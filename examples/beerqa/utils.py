import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from heapq import nlargest
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import wikipedia
from tensorzero import AsyncTensorZeroGateway, ToolCall, ToolResult

log = logging.getLogger(__name__)
tensorzero_semaphore = asyncio.Semaphore(10)
wikipedia_semaphore = asyncio.Semaphore(10)


async def fetch_wikipedia_page(title: str) -> Union[str, wikipedia.WikipediaPage]:
    try:
        result = await asyncio.to_thread(wikipedia.page, title, auto_suggest=False)
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for {title}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"This is a disambiguation page. Please provide a more specific title from these options: {e.options}"
    return result


async def get_wikipedia_summary(title: str) -> str:
    page = await fetch_wikipedia_page(title)
    if type(page) is str:
        return page
    return page.summary


async def get_wikipedia_full_text(title: str) -> str:
    page = await fetch_wikipedia_page(title)
    if type(page) is str:
        return page
    return page.content


async def execute_tool_call(tool_call: ToolCall) -> ToolResult:
    title = tool_call.arguments["article_title"]
    if tool_call.name == "get_summary":
        async with wikipedia_semaphore:
            summary = await get_wikipedia_summary(title)
        return ToolResult(name=tool_call.name, result=summary, id=tool_call.id)
    elif tool_call.name == "get_full_text":
        async with wikipedia_semaphore:
            full_text = await get_wikipedia_full_text(title)
        return ToolResult(name=tool_call.name, result=full_text, id=tool_call.id)
    else:
        raise ValueError(f"Unknown tool call: {tool_call.name}")


class BeerQA:
    def __init__(self, data_path: str):
        with Path(data_path).open("r") as f:
            self.data = json.load(f)

    def get_question(self, index: int) -> str:
        return self.data["data"][index]["question"]

    def get_answers(self, index: int) -> List[str]:
        return self.data["data"][index]["answers"]


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ToolCallResult:
    name: str
    arguments: str
    result: str
    id: str

    def render(self) -> List[Dict[str, Any]]:
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": self.id,
                        "name": self.name,
                        "arguments": self.arguments,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "name": self.name,
                        "result": self.result,
                        "id": self.id,
                    }
                ],
            },
        ]


@dataclass
class Message:
    role: Role
    content: Union[str, ToolCallResult, ToolCall]

    def render(self) -> List[Dict[str, Any]]:
        """
        Render the message to a list of TensorZero messages that can be sent to the Gateway.
        """
        messages = []
        for content in self.content:
            if isinstance(content, ToolCallResult):
                messages.extend(content.render())
            elif isinstance(content, ToolCall):
                raise ValueError(
                    "ToolCall messages should be converted to ToolCallResult messages before a Node is rendered."
                )
            else:
                messages.append({"role": self.role.value, "content": content})
        return messages


@dataclass
class Node:
    messages: List[Message]
    score: Optional[float] = None
    finished: bool = False

    def render(self) -> List[Dict[str, Any]]:
        # Flatten the list of messages
        return [item for message in self.messages for item in message.render()]

    def is_finished(self) -> bool:
        if len(self.messages) == 0:
            return False
        message = self.messages[-1]
        if (
            isinstance(message.content, ToolCall)
            and message.content.name == "submit_answer"
        ):
            return True
        return False


async def beam_search(
    root_node: Node,
    client: AsyncTensorZeroGateway,
    beam_width: int = 5,
    branching_factor: int = 3,
    max_depth: int = 10,
):
    """
    Perform beam search on an LLM action / conversation tree.
    """
    nodes = [root_node]
    solutions = []
    for depth in range(max_depth):
        queries_remaining = max_depth - depth
        nodes_with_scores = []
        for node in nodes:
            node.score = await heuristic_evaluate_node(node, client, queries_remaining)
            if node.score is not None:
                nodes_with_scores.append(node)
        nodes = nodes_with_scores
        # Use a heap to efficiently get the top beam_width nodes without a full sort
        nodes = nlargest(beam_width, nodes, key=lambda n: n.score)
        tasks = []
        for node in nodes:
            for i in range(branching_factor):
                tasks.append(generate_successor(node, client, queries_remaining))
        results = await asyncio.gather(*tasks)
        new_nodes = [result for result in results if result is not None]
        nodes_to_remove = []
        for node in new_nodes:
            if node.is_finished():
                solutions.append(node)
                nodes_to_remove.append(node)
                continue
            for message in node.messages:
                new_content = []
                for content in message.content:
                    if isinstance(content, ToolCall):
                        tool_result = await execute_tool_call(content)
                        tool_call_result = ToolCallResult(
                            name=content.name,
                            arguments=content.arguments,
                            result=tool_result.result,
                            id=content.id,
                        )
                        new_content.append(tool_call_result)
                    else:
                        new_content.append(content)
                message.content = new_content
        for node in nodes_to_remove:
            new_nodes.remove(node)


async def heuristic_evaluate_node(
    node: Node,
    client: AsyncTensorZeroGateway,
    queries_remaining: int,
    variant_name: Optional[str] = None,
) -> Optional[float]:
    """
    Heuristically evaluate a node.
    """
    try:
        async with tensorzero_semaphore:
            response = await client.inference(
                function_name="heuristic_evaluation",
                input={
                    "system": {
                        "question": node.messages[0].content,
                        "queries_remaining": queries_remaining,
                    },
                    "messages": node.render(),
                },
                variant_name=variant_name,
            )
    except Exception as e:
        log.warning(f"Error heuristically evaluating node: {e}")
        return None
    return response.output.parsed["score"]


async def generate_successor(
    node: Node,
    client: AsyncTensorZeroGateway,
    queries_remaining: int,
    variant_name: Optional[str] = None,
) -> Optional[Node]:
    """
    Generate a new node by calling the LLM.
    """
    try:
        async with tensorzero_semaphore:
            response = await client.inference(
                function_name="beerqa_solver",
                input={
                    "system": {"queries_remaining": queries_remaining},
                    "messages": node.render(),
                },
                variant_name=variant_name,
            )
    except Exception as e:
        log.warning(f"Error generating successor: {e}")
        return None
    new_node = deepcopy(node)
    new_node.messages.append(
        Message(Role.ASSISTANT, response.output.parsed["messages"])
    )
    return new_node
