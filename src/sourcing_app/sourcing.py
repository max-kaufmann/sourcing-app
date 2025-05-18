import arxiv
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from dotenv import load_dotenv
from inspect_ai.tool import web_search, web_browser
from typing import TypedDict, Any
from pydantic import ValidationError
from inspect_ai.solver import (
    solver,
    Solver,
    Generate,
    system_message,
    TaskState,
    use_tools,
    chain,
)
import json
import re
from inspect_ai.scorer import Scorer, Target, Score


from inspect_ai.model import CachePolicy
from pydantic import Field
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.model._chat_message import ChatMessage, ChatMessageTool, ChatMessageUser
from inspect_ai.model._model import get_model
from inspect_ai.tool import Tool, ToolResult, tool
from inspect_ai.tool import tool_with
from inspect_ai.util._limit import token_limit as create_token_limit
from pydantic import BaseModel
from typing import Literal

load_dotenv()


class Author(BaseModel):
    """Basic author information parsed directly from ArXiv API."""

    name: str
    paper_id: str
    paper_title: str | None = None
    paper_abstract: str | None = None
    paper_url: str | None = None
    paper_pdf: Path | None = None


class Candidate(BaseModel):
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


class RatedCandidate(BaseModel):
    candidate_rating: Literal[1, 2, 3, 4, 5] | None = Field(
        description="The rating of the candidate, from 1 to 5, where 1 is the worst and 5 is the best, according to the rating criteria."
    )
    candidate_rating_reasoning: str | None = Field(
        description="The reasoning behind the rating of the candidate, this should be detailed and explain why the candidate was rated as they were."
    )


ARXIV_CLIENT = arxiv.Client()


def get_authors_from_arxiv_ids(
    ids: list[str], client: arxiv.Client = ARXIV_CLIENT
) -> list[Author]:
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

    authors: list[Author] = []
    for paper in results:
        # Extract paper ID from the entry_id (which is a URL)
        paper_id = paper.entry_id.split("/")[-1]
        paper_title = paper.title
        paper_abstract = paper.summary
        paper_url = paper.entry_id

        # Create an Author object for each author of the paper
        for paper_author in paper.authors:
            author = Author(
                name=paper_author.name,
                paper_id=paper_id,
                paper_title=paper_title,
                paper_abstract=paper_abstract,
                paper_url=paper_url,
            )
            authors.append(author)

    return authors


def arxiv_id_from_url(url: str) -> str:
    return re.search(r"arxiv\.org/pdf/(\d+\.\d+)", url).group(1)


DEFAULT_SYSTEM_MESSAGE = """You are a helpful assistant attempting to submit the correct answer. You have
several functions available to help with finding the answer. Each message
may perform one function call. You will see the result of the function right
after sending the message. If you need to perform multiple actions, you can
always send more messages with subsequent function calls. Do some reasoning
before your actions, describing what function calls you are going to use and
how they fit into your plan.

When you have completed the task and have an answer, call the submit()
function to report it.
"""

DEFAULT_INCORRECT_MESSAGE = """
Your submission was incorrect. Please proceed and attempt to find the correct answer.
"""
DEFAULT_CONTINUE_MESSAGE = "Please proceed to the next step using your best judgement."
DEFAULT_SUBMIT_DESCRIPTION = "Submit an answer for evaluation."

DEFAULT_MODEL = "anthropic/claude-3-7-sonnet-20250219"
DEFAULT_INPUT_PROMPT = """I need to evaluate if this researcher might be a good candidate for the UK AI Safety Institute (AISI). Your job will be to go from the name of the author, and the paper they wrote, to structured information about their background.

Author: {author_name}
Paper: {paper_title}
Paper URL: {paper_url}
Paper Abstract: {paper_abstract}.

Please return a summary of the background of this specific author. The summary should correspond to the following format:

{model_json_schema}

Try your best to fill the in the formation, by searching the web and using your provided tools. If any information is not available, make sure to leave it as a None. To submit your answer, use your submit tool with a JSON object that matches the format of the Candidate model. You should start by using a web
"""


def get_candidate_info_tools() -> list[Tool]:
    """Get the tools for the candidate info agent."""
    return [web_search(provider="tavily")] + web_browser()


@solver
def candidate_info_agent(
    *,
    cache: bool | CachePolicy = False,
    max_attempts: int = 1,
    message_limit: int | None = None,
    token_limit: int | None = None,
    max_tool_output: int | None = None,
    continue_message: str = DEFAULT_CONTINUE_MESSAGE,
    submit_description: str = DEFAULT_SUBMIT_DESCRIPTION,
    submit_append: bool = False,
) -> Solver:
    """Agent which finds and parses candidate information from a paper, returning a JSON corresponding to the Candidate BaseModel.

    Agent that runs a tool use loop until the model submits an answer using the
    `submit()` tool. Tailor the model's instructions by passing a `system_message()`
    and/or other steps to `init` (if no `init` is specified then a default system
    message will be used). Use `max_attempts` to support additional submissions if
    the initial submission(s) are incorrect.

    Submissions are evaluated using the task's main scorer, with value of 1.0
    indicating a correct answer. Scorer values are converted to float (e.g.
    "C" becomes 1.0) using the standard value_to_float() function. Provide an
    alternate conversion scheme as required via `score_value`.

    Args:
       tools: Tools available for the agent. Either a list of tools or a Solver that
          can yield dynamic tools per-sample.
       cache: Caching behaviour for generate responses (defaults to no caching).
       max_attempts: Maximum number of submissions to accept before terminating.
       message_limit: Limit on messages in sample before terminating agent.
          If not specified, will use limit_messages defined for the task. If there is none
          defined for the task, 50 will be used as a default.
       token_limit: Limit on tokens used in sample before terminating agent.
       max_tool_output: Maximum output length (in bytes).
          Defaults to max_tool_output from active GenerateConfig.
       continue_message: User message to urge the model to continue when it
          doesn't make a tool call.
       submit_name: Name for tool used to make submissions
          (defaults to 'submit')
       submit_description: Description of submit tool (defaults to
          'Submit an answer for evaluation')
       submit_append: Append the submit tool output to the model completion
           text (defaults to `False`, which means the submission overwrites
           the model completion).
       **kwargs: Deprecated arguments for backward compatibility.

    Returns:
        Plan for agent.
    """
    init = system_message(DEFAULT_SYSTEM_MESSAGE)

    tools = use_tools(get_candidate_info_tools(), append=True)

    # submission tool
    @tool
    def submit() -> Tool:
        async def execute(answer: str) -> ToolResult:
            """Submit an answer for evaluation.

            Args:
              answer (str): Submitted answer
            """
            return answer

        return execute

    # solver that adds submission tool
    @solver
    def submit_tool() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.tools.append(
                tool_with(submit(), "submit", "Submit an answer for evaluation")
            )
            return state

        return solve

    # helper to extract a submitted answer
    def submission(tool_results: list[ChatMessage]) -> str | None:
        return next(
            (
                result.text
                for result in tool_results
                if isinstance(result, ChatMessageTool) and result.function == "submit"
            ),
            None,
        )

    # main agent loop
    @solver
    def basic_agent_loop() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # resolve message_limit -- prefer parameter then fall back to task
            # (if there is no message_limit then default to 50)
            state.message_limit = message_limit or state.message_limit or 50

            # track attempts
            attempts = 0

            with create_token_limit(token_limit):
                # main loop
                while not state.completed:
                    # generate output and append assistant message
                    state.output = await get_model().generate(
                        input=state.messages, tools=state.tools, cache=cache
                    )
                    state.messages.append(state.output.message)

                    # check for context window overflow
                    if state.output.stop_reason == "model_length":
                        from inspect_ai.log._transcript import transcript

                        transcript().info(
                            "Agent terminated: model context window exceeded"
                        )
                        break

                    # resolve tools calls (if any)
                    if state.output.message.tool_calls:
                        # execute tool functions
                        tool_results, _ = await execute_tools(
                            [state.output.message],
                            state.tools,
                            max_output=max_tool_output,
                        )
                        state.messages.extend(tool_results)

                        # was an answer submitted?
                        answer = submission(tool_results)
                        if answer:
                            if submit_append:
                                state.output.completion = (
                                    f"{state.output.completion}\n\n{answer}".strip()
                                )
                            else:
                                state.output.completion = answer

                            # exit if we are at max_attempts
                            attempts += 1
                            if attempts >= max_attempts:
                                break

                            # exit if the submission is successful
                            answer_score = await score_pydantic_model(Candidate)(
                                state, Target("")
                            )
                            if answer_score.value == 1.0:
                                break
                            # otherwise notify the model that it was incorrect and continue
                            else:
                                validation_error = answer_score.metadata[
                                    "validation_error"
                                ]  # type: ignore[index]
                                assert isinstance(validation_error, ValidationError)
                                state.messages.append(
                                    ChatMessageUser(
                                        content="Your submission was not able to be correctly parsed. The error was:\n\n"
                                        + str(validation_error)
                                        + "\n\n Please replace it and try again."
                                    )
                                )
                    # no tool calls, urge the model to continue
                    else:
                        state.messages.append(ChatMessageUser(content=continue_message))

            return state

        return solve

    # return chain
    return chain(
        init,
        tools,
        submit_tool(),
        basic_agent_loop(),
    )


class PydanticScorerMetadata(TypedDict):
    candidate_info: (
        dict[str, Any] | None
    )  # Is a dictonary corresponding the the JSON of the Candidate BaseModel
    validation_error: ValidationError | None


def score_pydantic_model(pydantic_model: type[BaseModel]) -> Scorer:
    """Score a pydantic model against a target pydantic model."""

    async def score_pydantic_model(state: TaskState, target: Target) -> Score:
        # First, try to score the JSON string in state.output.completion
        try:
            parsed_model = pydantic_model.model_validate_json(
                state.output.completion
            ).model_dump()
            validation_error = None
            score = 1.0
        except ValidationError as e:
            parsed_model = None
            validation_error = e
            score = 0.0

        metadata = PydanticScorerMetadata(
            candidate_info=parsed_model, validation_error=validation_error
        )

        return Score(value=score, metadata=metadata)  # type: ignore[arg-type]

    return score_pydantic_model


@task
def source_authors(
    arxiv_ids: list[str],
    maximum_authors: int | None = None,
    search_agent_guidance: str = DEFAULT_INPUT_PROMPT,
    rater_agent_guidance: str = DEFAULT_RATER_GUIDANCE,
) -> Task:
    """
    Inspect Task definition for the sourcing task. Returns a task where each Sample is a single author,
    created from its arxiv id.

    Args:
        arxiv_ids: List of ArXiv paper IDs to search for authors
    """
    # Step 1: Get authors from ArXiv
    authors = get_authors_from_arxiv_ids(arxiv_ids)
    input_json_schema = json.dumps(Candidate.model_json_schema())

    if maximum_authors is not None:
        authors = authors[:maximum_authors]

    # Replace the {model_json_schema} with the JSON schema of the Candidate model

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
                model_json_schema=input_json_schema,
            ),
            # We'll add the basic author info to metadata
            metadata={
                "author_name": author.name,
                "paper_id": author.paper_id,
                "paper_title": author.paper_title,
            },
        )
        samples.append(sample)

    # Step 3: Create and return the Task
    return Task(
        name="AISI Author Sourcing",
        dataset=MemoryDataset(samples),
        solver=candidate_info_agent(),
    )
