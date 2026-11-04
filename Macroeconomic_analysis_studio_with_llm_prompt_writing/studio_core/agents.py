"""Two roles. The controller, never the model, is responsible for execution."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from studio_core.spec import Plan, Proposal, Review

DEMO_PROMPT = (
    "Forecast next month's fare_inflation with a Super Learner combining LASSO and "
    "random forest. Use fare_inflation lags 1, 2 and 3, fuel_inflation lag 1, "
    "and annual seasonality. Evaluate one-month-ahead forecasts over the last 24 months."
)


class DemoAgents:
    """Scripted fixture for DEMO_PROMPT only. The first plan deliberately misses seasonality."""
    name = "offline scripted demonstration"

    def propose(self, context):
        plan = Plan(target="fare_inflation", predictors=["fuel_inflation"],
                    target_lags=[1, 2, 3], predictor_lags=[1],
                    seasonality=bool(context["history"]), holdout_months=24)
        return Proposal(decision="run", plan=plan,
                        reason="Use a lagged model; add calendar terms after the critic requests them."
                        if plan.seasonality else "First demonstration plan uses lagged inflation.")

    def review(self, context):
        features = context["evidence"]["features"]
        if "calendar__sin" not in features:
            return Review(decision="revise", reason="The prompt asks for annual seasonality. "
                          "The executed design has only lagged inflation. Add month sine and cosine.")
        return Review(decision="accept", reason="The executed features match the requested lags "
                      "and seasonality. Every training fold ends before its validation block.")


class OllamaAgents:
    def __init__(self, model="llama3.1:8b"):
        if not model.strip() or "cloud" in model.lower():
            raise ValueError("Choose a locally installed Ollama model, such as llama3.1:8b.")
        self.model = model
        self.name = f"local Ollama / {model}"

    def _call(self, role, schema, context):
        body = {"model": self.model, "stream": False, "format": schema.model_json_schema(),
                "options": {"temperature": 0, "seed": 422},
                "messages": [{"role": "system", "content": role},
                             {"role": "user", "content": json.dumps(context)}]}
        req = Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=240) as response:
                content = json.load(response)["message"]["content"]
        except HTTPError as exc:
            detail = exc.read(1200).decode(errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError("Local Ollama did not respond. Start Ollama and pull "
                               f"{self.model}, then submit the prompt again.") from exc
        return schema.model_validate_json(content)

