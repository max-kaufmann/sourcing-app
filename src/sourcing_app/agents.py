from inspect_ai.tool import web_search, web_browser
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageSystem
from pydantic import ValidationError
from inspect_ai.agent import agent
import json
from inspect_ai.scorer import Target
from inspect_ai.agent._react import _remove_submit_tool, _agent_generate
from inspect_ai.model import Model

from inspect_ai.model._call_tools import execute_tools
from inspect_ai.model._chat_message import ChatMessage, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import Tool, ToolResult, tool
from sourcing_app.utils import score_pydantic_model
from sourcing_app.task import CandidateRating

DEFAULT_RATER_SYSTEM_PROMPT = """
You should rate the candidate on a scale of 1 to 5, where 1 is the worst and 5 is the best. The criteria should be, as well as general competence:

- Which of the authors have track records of publishing in ML conferences and a clear interest in adversarial ML or AI safety work (eg as demonstrated by publishing a at least few papers in either area) 
- Upweight people at frontier ai labs, who hsve finished phds, or who are towards the end of their phd

use your submit tool to return a rating in the following format:

{model_json_schema}

Make sure to first use your web search too to get some background information about the candidate, and then its very important that you actively try to find the different urls which are included in the candidate information using your web browsing tools. Your web search tool is good for a first pass of getting information for a candidate, but you then need to use your web browser to find the other information. DONT GIVE UP! Only once you are certain that you cannot find the relevant information about the candidate, then you should submit a rating - however, think carefully about the information that you have found."""


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


# default submit tool
@tool
def submit() -> Tool:
    async def execute(answer: str) -> ToolResult:
        """Submit an answer for evaluation.

        Args:
            answer (str): Submitted answer
        """
        return answer

    return execute


def get_candidate_info_tools() -> list[Tool]:
    """Get the tools for the candidate info agent."""
    return [web_search(provider="google")] + web_browser()


@agent
def candidate_rater_agent(
    *,
    system_prompt: str = DEFAULT_RATER_SYSTEM_PROMPT,
    max_attempts: int = 5,
    model: str | Model | Agent | None = None,
) -> Agent:
    """Extensible ReAct agent based on the paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).

    Provide a `name` and `description` for the agent if you plan on using it
    in a multi-agent system (this is so other agents can clearly identify
    its name and purpose). These fields are not required when using `react()`
    as a top-level solver.

    The agent runs a tool use loop until the model submits an answer using the
    `submit()` tool. Use `instructions` to tailor the agent's system message
    (the default `instructions` provides a basic ReAct prompt).

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    By default, the model will be urged to continue if it fails to call
    a tool. Customise this behavior using the `on_continue` option.

    Args:
       name: Agent name (required when using with `handoff()` or `as_tool()`)
       description: Agent description (required when using with `handoff()` or `as_tool()`)
       prompt: Prompt for agent. Includes agent-specific contextual `instructions`
          as well as an optional `assistant_prompt` and `handoff_prompt` (for agents
          that use handoffs). both are provided by default but can be removed or
          customized). Pass `str` to specify the instructions and use the defaults
          for handoff and prompt messages.
       tools: Tools available for the agent.
       model: Model to use for agent (defaults to currently evaluated model).
       attempts: Configure agent to make multiple attempts.
       submit: Use a submit tool for reporting the final answer. Defaults to `True`
          which uses the default submit behavior. Pass an `AgentSubmit` to
          customize the behavior or pass `False` to disable the submit tool.
       on_continue: Message to play back to the model to urge it to continue
          when it stops calling tools. Use the placeholder {submit} to refer to
          the submit tool within the message. Alternatively, an async function
          to call to determine whether the loop should continue and what message
          to play back. Note that this function is called on _every_ iteration of
          the loop so if you only want to send a message back when the model fails
          to call tools you need to code that behavior explicitly.
       truncation: Truncate the conversation history in the event of a context
          window overflow. Defaults to "disabled" which does no truncation. Pass
          "auto" to use `trim_messages()` to reduce the context size. Pass a
          `MessageFilter` function to do custom truncation.

    Returns:
        ReAct agent.
    """

    tools = get_candidate_info_tools() + [submit()]

    system_prompt = system_prompt.format(
        model_json_schema=json.dumps(CandidateRating.model_json_schema(), indent=2)
    )

    async def execute(state: AgentState) -> AgentState:
        state.messages.insert(0, ChatMessageSystem(content=system_prompt))
        # track attempts
        attempt_count = 0

        # main loop = will terminate after submit (subject to max_attempts)
        # or if a message or token limit is hit
        candidate_rating = None
        validation_error = None
        while True:
            # generate output and append assistant message
            state = await _agent_generate(model, state, tools)

            # check for context window overflow
            if state.output.stop_reason == "model_length":
                break

            # resolve tool calls (if any)
            if state.output.message.tool_calls:
                # call tool functions
                messages, output = await execute_tools(state.messages, tools)
                state.messages.extend(messages)
                if output:
                    state.output = output

                # check for a submission
                answer = submission(messages)
                if answer is not None:
                    # set the output to the answer for scoring
                    state.output.completion = answer

                    # exit if we are at max_attempts
                    attempt_count += 1
                    if attempt_count >= max_attempts:
                        break

                    answer_scores = await score_pydantic_model(CandidateRating)(
                        state, Target("")
                    )  # type: ignore[arg-type]
                    candidate_rating = answer_scores.metadata["model_dump"]  # type: ignore[index]
                    validation_error = answer_scores.metadata["validation_error"]  # type: ignore[index]
                    # exit if the submission is successful
                    if answer_scores.value == 1.0:
                        break
                    # otherwise notify the model that it was incorrect and continue
                    else:
                        validation_error = answer_scores.metadata["validation_error"]  # type: ignore[index]
                        assert isinstance(validation_error, ValidationError)
                        state.messages.append(
                            ChatMessageUser(
                                content="Your submission was not able to be correctly parsed. The error was:\n\n"
                                + str(validation_error)
                                + "\n\n Please replace it and try again."
                            )
                        )

            elif not state.output.message.tool_calls:
                state.messages.append(
                    ChatMessageUser(
                        content="Please proceed to the next step using your best judgement. If you think you have found the information you need, please call the `submit()` tool with your final answer."
                    )
                )

        # once we are complete, remove submit tool calls from the history
        # (as they will potentially confuse parent agents who also have
        # their own submit tools that they are 'watching' for)
        state.messages = _remove_submit_tool(state.messages, submit.__name__)
        state.output.metadata = (state.output.metadata or {}) | {
            "candidate_rating": candidate_rating,
            "validation_error": validation_error,
        }
        return state

    return execute
