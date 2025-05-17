# %%
import pandas as pd
from inspect_ai.model import CachePolicy, get_model
import asyncio, json, os, requests
from anthropic import AsyncAnthropic
import arxiv
from arxiv import Result
from dataclasses import dataclass
from pathlib import Path

ARXIV_CLIENT = arxiv.Client()
DEFAULT_MODEL = "anthropic/claude-3-7-sonnet-20250219"
@dataclass
class ParsedAuthor:
    name: str
    paper_id: str
    paper_pdf: Path | None = None

def query_arxiv_for_authors(ids: list[str],client: arxiv.Client = ARXIV_CLIENT) -> list[ParsedAuthor]:
    # Note: We'd need to implement this with inspect_ai if arxiv has an API wrapper
    # For now, keeping the original implementation but would need to be updated
    search = arxiv.Search(id_list=ids).results() 
    results = client.results(arxiv.Search(id_list=ids))

    authors = [ParsedAuthor(a.name, r.entry_id.split('/')[-1]) for r in results for a in r.authors]

    return  authors

