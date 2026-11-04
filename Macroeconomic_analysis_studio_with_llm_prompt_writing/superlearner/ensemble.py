"""Chronological validation and evaluation for the forecasting ensemble."""
from studio_core.spec import Plan
from studio_core.workflow import Result
from .forecasting import development, evaluate, score


def forecast_development(design, holdout_months):
    plan = Plan(**design.options, holdout_months=holdout_months)
    frame = design.frame
    if len(frame) <= holdout_months:
        raise ValueError("The holdout leaves no training months.")
    dev = frame.iloc[:-holdout_months]
    weights, oof, evidence = development(dev, plan)
    return plan, weights, oof, evidence


def super_learner(design, *, holdout_months):
    plan, weights, oof, evidence = forecast_development(design, holdout_months)
    predictions, future = evaluate(design.frame, plan, weights)
    audit = {"task": "forecast", "accepted_plan": plan.model_dump(), "weights": evidence["weights"],
             "holdout_start": str(design.frame.index[-holdout_months]),
             "holdout_end": str(design.frame.index[-1]), "holdout_months_scored": len(predictions),
             "next_month": future}
    return Result(predictions, score(predictions), oof, audit)


