"""Public entry points, revision loop and a reproducible result bundle."""
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from studio_core.agents import DEMO_PROMPT, DemoAgents
from studio_core.data_io import load_demo, monthly_frame
from superlearner.forecasting import development, evaluate, score


class ReviewStopped(ValueError):
    """No approved result was released. Inspect history for the decision trail."""
    def __init__(self, message, history):
        super().__init__(message)
        self.history = history


@dataclass
class Result:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    validation_predictions: pd.DataFrame
    audit: dict

    def save(self, destination):
        """Write into a new result folder only after all files are ready."""
        save_bundle(destination, {"holdout_forecasts.csv": self.predictions,
                                  "metrics.csv": self.metrics,
                                  "validation_forecasts.csv": self.validation_predictions}, self.audit)


def save_bundle(destination, tables, audit):
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError(f"Choose a new output directory: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as temp:
        stage = Path(temp) / "results"
        stage.mkdir()
        for name, table in tables.items():
            table.to_csv(stage / name)
        (stage / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        if "python_script" in audit:
            (stage / "analysis.py").write_text(audit["python_script"], encoding="utf-8")
        stage.rename(dest)


def _run(prompt, data, agents):
    if not prompt.strip():
        raise ValueError("Supply a forecasting question.")
    frame = monthly_frame(data)
    history = []
    context = {"prompt": prompt, "available_columns": list(frame.columns),
               "calendar_extent": [str(frame.index[0]), str(frame.index[-1])],
               "history": history}
    locked = None
    for attempt in range(1, 4):
        entry = {"round": attempt}
        try:
            proposal = agents.propose(context)
            entry["analyst"] = proposal.model_dump()
            if proposal.decision == "ask_user":
                raise ReviewStopped(proposal.reason, history + [entry])
            plan = proposal.plan
            if plan is None:
                raise ValueError("Provide a plan for decision=run.")
            if locked and (plan.target, plan.holdout_months) != locked:
                raise ValueError("Target and holdout length are locked after development feedback. "
                                 "Start a new analysis to change them.")
            if len(frame) <= plan.holdout_months:
                raise ValueError("The requested holdout leaves no training data.")
            dev = frame.iloc[:-plan.holdout_months]
            weights, oof, evidence = development(dev, plan)
            locked = (plan.target, plan.holdout_months)
            entry["evidence"] = evidence
            review = agents.review({**context, "plan": plan.model_dump(), "evidence": evidence})
        except ReviewStopped:
            raise
        except ValueError as exc:
            entry["validation_feedback"] = str(exc)
            history.append(entry)
            continue
        entry["critic"] = review.model_dump()
        history.append(entry)
        if review.decision == "ask_user":
            raise ReviewStopped(review.reason, history)
        if review.decision == "accept":
            break
    else:
        last = history[-1]
        feedback = last.get("validation_feedback") or last["critic"]["reason"]
        raise ReviewStopped(f"Revision budget exhausted. Last feedback: {feedback}", history)

    predictions, future = evaluate(frame, plan, weights)
    audit = {"agent_mode": agents.name, "original_prompt": prompt, "rounds": history,
             "accepted_plan": plan.model_dump(), "weights": evidence["weights"],
             "holdout_start": str(dev.index[-1] + 1), "holdout_end": str(frame.index[-1]),
             "holdout_months_scored": len(predictions), "next_month": future}
    return Result(predictions, score(predictions), oof, audit)


def run_demo():
    """Run the fixed teaching prompt with scripted analyst and critic roles."""
    return _run(DEMO_PROMPT, load_demo(), DemoAgents())


def forecast(prompt, data=None, model="llama3.1:8b"):
    """Interpret a new prompt with local Llama. data can be a CSV path or a DataFrame."""
    from studio_core.studio import analyze
    return analyze(prompt, data=data, model=model, required_task="forecast")
