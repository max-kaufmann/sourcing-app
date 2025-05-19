from typing import TypedDict, Any
from pydantic import ValidationError
from inspect_ai.solver import (
    TaskState,
)
from inspect_ai.scorer import Scorer, Target, Score

from pydantic import BaseModel


class PydanticScorerMetadata(TypedDict):
    model_dump: (
        dict[str, Any] | None
    )  # Is a dictonary corresponding the the JSON of the Candidate BaseModel
    validation_error: str | None


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
            validation_error = str(e)
            score = 0.0

        metadata = PydanticScorerMetadata(
            model_dump=parsed_model, validation_error=validation_error
        )

        return Score(value=score, metadata=metadata)  # type: ignore[arg-type]

    return score_pydantic_model
