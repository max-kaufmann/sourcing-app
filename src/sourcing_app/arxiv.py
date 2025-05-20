import arxiv
from pathlib import Path
import re

from pydantic import BaseModel

ARXIV_CLIENT = arxiv.Client()


class ArxivAuthor(BaseModel):
    """Basic author information parsed directly from ArXiv API."""

    name: str
    paper_id: str
    paper_title: str | None = None
    paper_abstract: str | None = None
    paper_url: str | None = None
    paper_pdf: Path | None = None


def get_authors_from_arxiv_ids(
    ids: list[str], client: arxiv.Client = ARXIV_CLIENT
) -> list[list[ArxivAuthor]]:
    """
    Retrieve authors from a list of ArXiv paper IDs.

    Args:
        ids: List of ArXiv paper IDs
        client: ArXiv client instance (optional)

    Returns:
        List of Author objects with basic information populated
    """
    # Create ArXiv search with the list of IDs
    search = arxiv.Search(id_list=ids)
    results = client.results(search)

    authors: list[list[ArxivAuthor]] = []
    for paper in results:
        # Extract paper ID from the entry_id (which is a URL)
        paper_id = paper.entry_id.split("/")[-1]
        paper_title = paper.title
        paper_abstract = paper.summary
        paper_url = paper.entry_id

        authors_for_paper: list[ArxivAuthor] = []
        # Create an Author object for each author of the paper
        for paper_author in paper.authors:
            author = ArxivAuthor(
                name=paper_author.name,
                paper_id=paper_id,
                paper_title=paper_title,
                paper_abstract=paper_abstract,
                paper_url=paper_url,
            )
            authors_for_paper.append(author)

        authors.append(authors_for_paper)

    return authors


def arxiv_id_from_url(url: str) -> str:
    search_result = re.search(r"arxiv\.org/([a-zA-Z]+)/(\d+\.\d+)", url)
    if search_result is None:
        raise ValueError(f"Could not parse arxiv id from url: {url}")
    return search_result.group(2)
