import asyncio
import json
import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from heapq import nlargest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import wikipedia
from scipy.stats import t
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall, ToolResult
from tqdm.asyncio import trange

log = logging.getLogger(__name__)
tensorzero_semaphore = asyncio.Semaphore(60)
wikipedia_semaphore = asyncio.Semaphore(30)


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


async def execute_tool_call(tool_call: ToolCall) -> Optional[ToolResult]:
    try:
        title = tool_call.arguments["article_title"]
    except KeyError:
        log.error(f"Tool call {tool_call.name} missing argument: article_title")
        log.error(f"Tool call: {tool_call}")
        return None
    if tool_call.name == "get_summary":
        async with wikipedia_semaphore:
            try:
                summary = await get_wikipedia_summary(title)
            except Exception as e:
                log.error(f"Error getting summary for {title}: {e}")
                return None
        return ToolResult(name=tool_call.name, result=summary, id=tool_call.id)
    elif tool_call.name == "get_full_text":
        async with wikipedia_semaphore:
            try:
                full_text = await get_wikipedia_full_text(title)
            except Exception as e:
                log.error(f"Error getting full text for {title}: {e}")
                return None
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
    arguments: Any
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
                        "arguments": json.dumps(self.arguments),
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
    content: List[Union[Text, ToolCallResult, ToolCall]]

    def render(self) -> List[Dict[str, Any]]:
        """
        Render the message to a list of TensorZero messages that can be sent to the Gateway.
        """
        messages = []
        content_list = (
            self.content if isinstance(self.content, list) else [self.content]
        )
        for content in content_list:
            if isinstance(content, ToolCallResult):
                messages.extend(content.render())
            elif isinstance(content, ToolCall):
                if content.name == "submit_answer":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"The answer is {content.arguments["answer"]}",
                        }
                    )
                else:
                    raise ValueError(
                        "ToolCall messages should be converted to ToolCallResult messages before a Node is rendered."
                    )
            else:
                messages.append({"role": self.role.value, "content": content})
        return messages


def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    h = sem * t.ppf((1 + confidence) / 2.0, n - 1)
    return mean - h, mean + h


@dataclass
class Node:
    messages: List[Message]
    score: Optional[float] = None

    def render(self) -> List[Dict[str, Any]]:
        # Flatten the list of messages
        return [item for message in self.messages for item in message.render()]

    def is_finished(self) -> bool:
        if len(self.messages) == 0:
            return False
        message = self.messages[-1]
        for content in message.content:
            if isinstance(content, ToolCall) and content.name == "submit_answer":
                return True
        return False


async def beam_search(
    root_node: Node,
    client: AsyncTensorZeroGateway,
    beam_width: int = 5,
    branching_factor: int = 3,
    max_depth: int = 4,
) -> Tuple[str, List[List[float]]]:
    """
    Perform beam search on an LLM action / conversation tree.
    """
    nodes = [root_node]
    solutions = []
    scores = []
    for depth in range(max_depth):
        print(f"Depth: {depth}")
        queries_remaining = max_depth - depth
        nodes_with_scores = []
        node_scores = await asyncio.gather(
            *[
                heuristic_evaluate_node(node, client, queries_remaining)
                for node in nodes
            ]
        )
        for node, score in zip(nodes, node_scores):
            if score is not None:
                node.score = score
                nodes_with_scores.append(node)
        nodes = nodes_with_scores
        scores.append([node.score for node in nodes])
        print(f"Scores: {scores[-1]}")
        # Use a heap to efficiently get the top beam_width nodes without a full sort
        nodes = nlargest(beam_width, nodes, key=lambda n: n.score)
        tasks = []
        for node in nodes:
            for i in range(branching_factor):
                tasks.append(generate_successor(node, client, queries_remaining))
        results = await asyncio.gather(*tasks)
        new_nodes = [result for result in results if result is not None]
        nodes_to_remove = []

        # Collect all ToolCall tasks
        tool_call_tasks = []
        tool_call_info = []
        for node in new_nodes:
            if node.is_finished():
                solutions.append(node)
                nodes_to_remove.append(node)
                continue
            for message in node.messages:
                content_list = (
                    message.content
                    if isinstance(message.content, list)
                    else [message.content]
                )
                for content in content_list:
                    if isinstance(content, ToolCall):
                        tool_call_tasks.append(execute_tool_call(content))
                        tool_call_info.append((node, message, content))

        # Execute all tool calls concurrently
        tool_results = await asyncio.gather(*tool_call_tasks)

        for tool_result, (node, message, content) in zip(tool_results, tool_call_info):
            if tool_result is None:
                nodes_to_remove.append(node)
                continue
            tool_call_result = ToolCallResult(
                name=content.name,
                arguments=content.arguments,
                result=tool_result.result,
                id=content.id,
            )
            # Replace the ToolCall with ToolCallResult in message.content
            new_content = []
            content_list = (
                message.content
                if isinstance(message.content, list)
                else [message.content]
            )
            for c in content_list:
                if c == content:
                    new_content.append(tool_call_result)
                else:
                    new_content.append(c)
            message.content = new_content

        # Remove nodes marked for removal
        for node in nodes_to_remove:
            new_nodes.remove(node)
        nodes = new_nodes
        if len(nodes) == 0:
            break

    # Get final answers for each node and add them to solutions
    final_answer_tasks = []
    for node in nodes:
        final_answer_tasks.append(
            get_final_answer(client, node.messages[0].content[0], node)
        )

    final_answers = await asyncio.gather(*final_answer_tasks)

    for node, final_answer in zip(nodes, final_answers):
        final_node = deepcopy(node)
        final_node.messages.append(
            Message(
                Role.ASSISTANT,
                [
                    ToolCall(
                        type="tool_call",
                        name="submit_answer",
                        arguments={"answer": final_answer},
                        id="final_answer",
                        raw_arguments={"answer": final_answer},
                        raw_name="submit_answer",
                    )
                ],
            )
        )
        solutions.append(final_node)

    # Evaluate solutions
    # nodes_with_scores = []
    # node_scores = await asyncio.gather(*[heuristic_evaluate_node(node, client, queries_remaining) for node in solutions])
    # for node, score in zip(solutions, node_scores):
    #     if score is not None:
    #         node.score = score
    #         nodes_with_scores.append(node)
    # solutions = nodes_with_scores

    # Find the solution with the highest score
    # best_solution = None
    # best_score = float("-inf")
    # for solution in solutions:
    #     if solution.score is not None and solution.score > best_score:
    #         best_solution = solution
    #         best_score = solution.score
    # try:
    #     answer = best_solution.messages[-1].content[0].arguments["answer"]
    # except KeyError:
    #     answer = "No answer found"
    answer = await fuse_answers(client, root_node.messages[0].content[0], solutions)
    return answer, scores


async def get_final_answer(
    client: AsyncTensorZeroGateway, question: str, node: Node
) -> str:
    """
    Get the final answer from a node.
    """
    async with tensorzero_semaphore:
        response = await client.inference(
            function_name="final_answer",
            input={"system": {"question": question}, "messages": node.render()},
        )
    return response.output.parsed["answer"]


async def fuse_answers(
    client: AsyncTensorZeroGateway, question: str, solutions: List[Node]
) -> str:
    """
    Fuse together the answers to a question from multiple solutions.
    """
    prepared_solutions = []
    for solution in solutions:
        queries = []
        for message in solution.messages:
            if isinstance(message.content, list):
                for content in message.content:
                    if isinstance(content, ToolCallResult):
                        queries.append(
                            {
                                "title": content.arguments["article_title"],
                                "content": content.result,
                            }
                        )
        prepared_solutions.append(
            {
                "queries": queries,
                "answer": solution.messages[-1].content[0].arguments["answer"],
            }
        )
    user_input = {"solutions": prepared_solutions}
    async with tensorzero_semaphore:
        while prepared_solutions:
            try:
                response = await client.inference(
                    function_name="fuse_answers",
                    input={
                        "system": {"question": question},
                        "messages": [{"role": "user", "content": user_input}],
                    },
                )
                break  # If successful, exit the loop
            except Exception as e:
                if "too long" in str(e).lower():
                    # Remove one solution and try again
                    prepared_solutions.pop()
                    user_input = {"solutions": prepared_solutions}
                    log.warning(
                        f"Input too long. Removed a solution. Remaining solutions: {len(prepared_solutions)}"
                    )
                else:
                    log.warning(f"Error fusing answers: {e}")
                    log.warning(f"Question: {question}")
                    log.warning(f"Solutions: {prepared_solutions}")
                    # Randomly choose an answer from the prepared solutions
                    if prepared_solutions:
                        random_solution = random.choice(prepared_solutions)
                        return random_solution["answer"]
                    else:
                        return "No answer found"
        else:
            log.warning("All solutions removed. Unable to fuse answers.")
            return "No answer found"
    return response.output.parsed["answer"]


async def grade_answer(
    client: AsyncTensorZeroGateway,
    question: str,
    gt_answer: List[str],
    submitted_answer: str,
) -> float:
    async with tensorzero_semaphore:
        response = await client.inference(
            function_name="grade_answer",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "question": question,
                            "gt_answer": gt_answer,
                            "submitted_answer": submitted_answer,
                        },
                    }
                ]
            },
        )
    return response.output.parsed["score"]


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
                        "question": node.messages[0].content[0],
                        "queries_remaining": queries_remaining,
                    },
                    "messages": node.render()[1:],
                },
                variant_name=variant_name,
            )
    except Exception as e:
        log.warning(f"Error heuristically evaluating node: {e}")
        log.warning(f"Node: {node}")
        log.warning(f"Rendered Node: {node.render()}")
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
    new_node.messages.append(Message(Role.ASSISTANT, response.content))
    return new_node


async def solve_grade_question(
    client: AsyncTensorZeroGateway,
    question: str,
    gt_answers: List[str],
) -> Tuple[float, List[List[float]]]:
    root_node = Node(messages=[Message(Role.USER, question)])
    solution, scores = await beam_search(
        root_node, client, beam_width=5, branching_factor=3, max_depth=5
    )
    score = await grade_answer(client, question, gt_answers, solution)
    return score, scores


async def main():
    beerqa = BeerQA("data/beerqa_dev_v1.0.json")
    num_questions = 100
    scores = []

    async with AsyncTensorZeroGateway("http://localhost:3000") as client:
        tasks = []
        for i in range(num_questions):
            question = beerqa.get_question(i)
            gt_answer = beerqa.get_answers(i)
            tasks.append(
                solve_grade_question(
                    client,
                    question,
                    gt_answer,
                )
            )

        progress_bar = trange(num_questions, desc="Solving questions")
        for task in asyncio.as_completed(tasks):
            score, _ = await task
            scores.append(score)
            ci_lower, ci_upper = confidence_interval(scores)
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "Average Score": f"{np.mean(scores):.2f} CI: ({ci_lower:.2f}, {ci_upper:.2f})"
                },
                refresh=True,
            )
        progress_bar.close()

    print(f"Average score: {np.mean(scores)}")


if __name__ == "__main__":
    asyncio.run(main())
