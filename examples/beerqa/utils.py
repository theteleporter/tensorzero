import asyncio
import json
from typing import List
from pathlib import Path
import wikipedia

async def fetch_wikipedia_page(title: str) -> wikipedia.WikipediaPage:
    result = await asyncio.to_thread(wikipedia.page, title, auto_suggest=False)
    return result

async def get_wikipedia_summary(title: str) -> str:
    page = await fetch_wikipedia_page(title)
    return page.summary

async def get_wikipedia_full_text(title: str) -> str:
    page = await fetch_wikipedia_page(title)
    return page.content


class BeerQA:
    def __init__(self, data_path: str):
        with Path(data_path).open("r") as f:
            self.data = json.load(f)

    def get_question(self, index: int) -> str:
        return self.data['data'][index]["question"]

    def get_answers(self, index: int) -> List[str]:
        return self.data['data'][index]["answers"]
    
    