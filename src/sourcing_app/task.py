from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.util import SandboxEnvironmentSpec
import pandas as pd
from inspect_ai.agent import Agent
from sourcing_app.arxiv import get_authors_from_arxiv_ids, ArxivAuthor
import json
from inspect_ai.log import EvalLog

from pydantic import Field
from pydantic import BaseModel
from typing import Literal


class CandidateRating(BaseModel):
    """Enhanced author data with detailed information after web searching and scoring. Extra field annotations are added for the LLM to understand what its aim is."""

    # Professional information
    current_role: str | None = Field(
        description="The author's current role, if they have one."
    )
    organization: str | None = Field(
        description="The organization the author works at, if they have one."
    )
    location: str | None = Field(description="What country the author is based in.")
    # URLs to author profiles
    personal_website: str | None = Field(
        description="The URL to the author's personal website"
    )
    linkedin_url: str | None = Field(
        description="The URL to the author's LinkedIn profile"
    )
    scholar_url: str | None = Field(
        description="The URL to the author's Google Scholar profile"
    )

    # Summary of their background
    summary: str | None = Field(
        description="A high-level summary of the author's background, including their current role, organization, years of experience, and any other relevant information, from the perspective of sourcing."
    )

    # Level of difficulty while sourcing
    problem_level_while_soucing: Literal[1, 2, 3, 4, 5] | None = Field(
        description="The level of difficulty you had while sourcing the author. 1 is no problems, meaning that you feel confident that you got most of the relevant background information for the author, 5 is very difficult, meaning that you feel that you were not able to get much information about the author."
    )
    problems_while_sourcing: str | None = Field(
        description="Any problems you had while sourcing the author. Problems may include that the author has a common name, so that you were confused, or you were blocked."
    )

    candidate_rating: Literal[1, 2, 3, 4, 5] | None = Field(
        description="The rating of the candidate, from 1 to 5, where 1 is the worst and 5 is the best, according to the rating criteria."
    )
    candidate_rating_reasoning: str | None = Field(
        description="The reasoning behind the rating of the candidate, this should be detailed and explain why the candidate was rated as they were."
    )


DEFAULT_INPUT_PROMPT = """
Please fill out the information for the following candidate:

Name: {author_name}

Information about the paper that they wrote:

Paper Title: {paper_title}
Paper URL: {paper_url}
Paper Abstract: {paper_abstract}"""

DEFAULT_SANDBOX_CONFIG = Path(__file__).parent / "sandboxes" / "uk_web_tools.yaml"


@task(name="AISI Author Sourcing")
def source_authors_task(
    arxiv_ids: list[str],
    top_n_authors: int | None = None,
    sandbox_config: Path = DEFAULT_SANDBOX_CONFIG,
    input_prompt: str = DEFAULT_INPUT_PROMPT,
    solver: Agent | None = None,
) -> Task:
    """
    Inspect Task definition for the sourcing task. Returns a task where each Sample is a single author,
    created from its arxiv id.

    Args:
        arxiv_ids: List of ArXiv paper IDs to search for authors
        top_n_authors: Maximum number of authors to source for each paper (starting from left to right, lets you select the top N authors)
        sandbox_config: Path to the sandbox config file
        input_prompt: Prompt to use for the input
        solver: Solver to use for the task, defaults to sourcing_app.agents.candidate_rater_agent()
    """

    # Step 1: Get authors from ArXiv
    authors = get_authors_from_arxiv_ids(arxiv_ids)
    INPUT_JSON_SCHEMA = json.dumps(CandidateRating.model_json_schema())

    if top_n_authors is not None:
        authors = [authors_for_paper[:top_n_authors] for authors_for_paper in authors]

    # We flatten the list of authors
    authors = [author for authors_for_paper in authors for author in authors_for_paper]
    # Step 2: Create samples for each author
    samples: list[Sample] = []
    for i, author in enumerate(authors):
        # Create sample with author metadata
        sample = Sample(
            id=f"author_{i}",
            input=input_prompt.format(
                author_name=author.name,
                paper_title=author.paper_title,
                paper_url=author.paper_url,
                paper_abstract=author.paper_abstract,
                model_json_schema=INPUT_JSON_SCHEMA,
            ),
            # We'll add the basic author info to metadata
            metadata={"author": author.model_dump()},
        )
        samples.append(sample)

    from sourcing_app.agents import candidate_rater_agent

    # Step 3: Create and return the Task
    return Task(
        dataset=MemoryDataset(samples),
        solver=solver or candidate_rater_agent(),
        sandbox=SandboxEnvironmentSpec(type="docker", config=sandbox_config.as_posix()),
    )


def task_output_to_candidate_df(eval_log: EvalLog) -> pd.DataFrame:
    """
    Convert evaluation log to a pandas DataFrame containing candidate information.

    Args:
        eval_log: The evaluation log containing sample results

    Returns:
        A pandas DataFrame with author and candidate rating information
    """

    AUTHOR_CANDIDATE_COLUMNS = (
        list(ArxivAuthor.model_json_schema()["properties"].keys())
        + list(CandidateRating.model_json_schema()["properties"].keys())
        + ["log_location", "validation_error"]
    )
    if eval_log.samples is None:
        raise ValueError(
            "No samples found in the EvalLog, only pass completed tasks to this function."
        )

    df_rows = []
    for sample in eval_log.samples:
        candidate_rating = CandidateRating.model_validate(
            sample.output.metadata["candidate_rating"]
        )  # type: ignore[arg-type]
        author_info = ArxivAuthor.model_validate(sample.metadata["author"])  # type: ignore[arg-type]
        validation_error = sample.output.metadata["validation_error"]  # type: ignore[index]

        row = (
            candidate_rating.model_dump()
            | author_info.model_dump()
            | {"log_location": eval_log.location, "validation_error": validation_error}
        )
        df_rows.append(row)

    return pd.DataFrame(df_rows, columns=AUTHOR_CANDIDATE_COLUMNS, dtype=object)
