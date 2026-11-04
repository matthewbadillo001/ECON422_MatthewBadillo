from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Lag = Annotated[int, Field(ge=1, le=12)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Plan(Record):
    target: str = Field(min_length=1)
    predictors: list[str]
    target_lags: list[Lag] = Field(min_length=1, max_length=12)
    predictor_lags: list[Lag] = Field(max_length=12)
    seasonality: bool
    holdout_months: int = Field(ge=3, le=60)

    @model_validator(mode="after")
    def check_roles(self):
        if self.target in self.predictors:
            raise ValueError("List the target only under target.")
        if bool(self.predictors) != bool(self.predictor_lags):
            raise ValueError("Set predictor_lags exactly when predictors are requested.")
        for values in (self.predictors, self.target_lags, self.predictor_lags):
            if len(values) != len(set(values)):
                raise ValueError("Remove duplicate predictors or lags.")
        return self


class Proposal(Record):
    decision: Literal["run", "ask_user"]
    plan: Plan | None
    reason: str = Field(min_length=1)


class Review(Record):
    decision: Literal["accept", "revise", "ask_user"]
    reason: str = Field(min_length=1)
