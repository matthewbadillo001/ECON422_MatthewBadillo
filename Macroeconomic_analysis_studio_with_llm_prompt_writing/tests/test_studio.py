import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from studio_core.app import StudioServer

import numpy as np
import pandas as pd

from studio_core.api import DEMO_PROMPT, load_demo
from studio_core.app import Handler
from studio_core.example_prompt import FORECAST_PROMPT
from studio_core.studio import CompleteExampleAssessment, PromptAssessment, ClarificationCheck, StudioAgents, WorkedForecastDraft, analyze
from studio_core.spec import Review
from studio_core.studio import Draft, Interpretation, ESTIMATION_PROMPT, _analyze, parse_program
from studio_core.workflow import ReviewStopped

FEATURES = ('design = monthly_features(data, target="fare_inflation", target_lags=[1,2,3], '
            'predictors=["fuel_inflation"], predictor_lags=[1], seasonality=True)')
FORECAST = FEATURES + '\nresult = super_learner(design, holdout_months=24)'
ESTIMATE = FEATURES + '\nresult = ols_hac(design, term="fuel_inflation__lag1", hac_lags=3)'


WRAPPED_FORECAST = """import pandas as pd
from monthly_forecast import monthly_features, super_learner

# Load the data
data = pd.read_csv('data.csv')

# Define the design
design = monthly_features(data, target='fare_inflation', target_lags=[1, 2, 3], predictors=['fuel_inflation'], predictor_lags=[1], seasonality=True)

# Run the Super Learner
result = super_learner(design, holdout_months=24)

# Print the ensemble weights and January forecast
print(result['ensemble_weights'])
print(result['forecast']['2025-01'])"""

NOTES = dict(clarity="Outcome specified.", structure="A single supported workflow.", context="Monthly data available.",
             role="An economic perspective.", examples="Calendar lags are clear.")


class Replies:
    name = "controlled test replies"

    def __init__(self, code=FORECAST, revise=True):
        self.code = code
        self.count = 0
        self.revise = revise

    def review_prompt(self, context):
        return PromptAssessment(decision="accept", message="Ready for analysis.", criteria=NOTES, suggested_prompt="")

    def draft(self, context):
        self.count += 1
        code = self.code
        if self.count == 1 and self.revise:
            code = code.replace("seasonality=True", "seasonality=False")
        return Draft(decision="run", code=code, reason="Apply the requested functions.")

    def critic(self, context):
        if "seasonality=False" in context["code"]:
            return Review(decision="revise", reason="Add the requested seasonal terms.")
        return Review(decision="accept", reason="The executed design matches the question.")

    def confirm(self, context):
        return Review(decision="accept", reason="I agree with this exact code.")

    def interpret(self, context):
        return Interpretation(text="The table reports the computed results.")

    def audit_interpretation(self, context):
        return Review(decision="accept", reason="The interpretation matches the fixed result.")


class StudioTests(unittest.TestCase):
    def test_generated_program_and_dual_approval(self):
        res = _analyze(DEMO_PROMPT, load_demo(), Replies())
        self.assertEqual([r["critic"]["decision"] for r in res.audit["rounds"]], ["revise","accept"])
        self.assertEqual(res.audit["rounds"][-1]["analyst_confirmation"]["decision"], "accept")
        self.assertAlmostEqual(res.metrics.loc["super_learner","RMSE"], .5283, places=3)
        self.assertEqual(res.audit["approved_revision"], res.audit["rounds"][-1]["revision_id"])
        with tempfile.TemporaryDirectory() as t:
            res.save(Path(t)/"run")
            namespace = {}
            with patch("sys.argv", ["analysis.py"]), contextlib.redirect_stdout(io.StringIO()):
                exec((Path(t)/"run"/"analysis.py").read_text(), namespace)
            pd.testing.assert_frame_equal(namespace["result"].metrics, res.metrics)

    def test_analyst_can_disagree_after_critic_accepts(self):
        agents = Replies(revise=False)
        agents.confirm = lambda c: Review(decision="revise",reason="Unresolved methodological issue.")
        with patch("studio_core.studio.super_learner") as release:
            with self.assertRaises(ReviewStopped) as e:
                _analyze(DEMO_PROMPT, load_demo(), agents)
            self.assertEqual(len(e.exception.history), 3)
            release.assert_not_called()

    def test_source_is_restricted(self):
        for code in [FORECAST.replace("data,", "open('secret'),"), FORECAST+'\nimport os',
                     FORECAST.replace("24)", "__import__('os'))"),
                     FORECAST.replace("target_lags=[1,2,3]", "target_lags=[x for x in data]")]:
            with self.assertRaises(ValueError):
                parse_program(code)

    def test_reported_llama_script_is_reviewed_and_replayed_in_round_one(self):
        # Use the actual response from the failed local Llama run, including the fake import.
        agents = Replies(WRAPPED_FORECAST, revise=False)
        with patch.object(agents, "critic", wraps=agents.critic) as critic, \
             patch.object(agents, "confirm", wraps=agents.confirm) as confirm:
            result = _analyze(FORECAST_PROMPT, load_demo(), agents, built_in_forecast=True)
        self.assertEqual(agents.count, 1)
        entry, = result.audit["rounds"]
        self.assertEqual(entry["analyst"]["code"], WRAPPED_FORECAST)
        cleaned = entry["validated_code"]
        self.assertEqual(parse_program(cleaned), parse_program(FORECAST))
        for reviewer in (critic, confirm):
            self.assertEqual(reviewer.call_args.args[0]["code"], cleaned)
        script = result.audit["python_script"]
        self.assertNotIn("monthly_forecast", script)
        self.assertNotIn("pd.read_csv('data.csv')", script)
        self.assertNotIn("result['forecast']", script)
        namespace = {}
        with patch("sys.argv", ["analysis.py"]), contextlib.redirect_stdout(io.StringIO()):
            exec(script, namespace)
        pd.testing.assert_frame_equal(namespace["result"].metrics, result.metrics)
        self.assertAlmostEqual(result.audit["next_month"]["forecast"], 0.1023773, places=5)

    def test_wrapped_script_respects_input_and_rejects_other_operations(self):
        self.assertEqual(parse_program("```python\n"+WRAPPED_FORECAST+"\n```"), parse_program(FORECAST))
        data = load_demo()
        data["fare_inflation"] += 3
        result = _analyze(DEMO_PROMPT, data, Replies(WRAPPED_FORECAST, revise=False))
        np.testing.assert_allclose(result.predictions.actual, data["fare_inflation"].iloc[-24:])
        for bad in (
            WRAPPED_FORECAST.replace("data.csv", "other.csv"),
            WRAPPED_FORECAST.replace("data.csv", "https://example.org/data.csv"),
            WRAPPED_FORECAST.replace("import pandas as pd", "import os"),
            WRAPPED_FORECAST.replace("design =", "data = data.dropna()\ndesign ="),
            WRAPPED_FORECAST.replace("result = super_learner", "design = None\nresult = super_learner"),
            WRAPPED_FORECAST + "\nprint(open('extra.txt'))",
            WRAPPED_FORECAST + "\nprint(result, file=open('extra.txt', 'w'))",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_program(bad)

    def test_custom_estimation_hac_against_calendar_pair_sum(self):
        data = load_demo()
        run = _analyze(ESTIMATION_PROMPT, data, Replies(ESTIMATE))
        d = data.set_index(pd.PeriodIndex(pd.to_datetime(data.date), freq="M")).asfreq("M")
        d = d.reindex(pd.period_range(d.index.min(), d.index.max(), freq="M"))
        features = [np.ones(len(d)), *(d.fare_inflation.shift(l) for l in (1,2,3)),
                    d.fuel_inflation.shift(1), np.sin(2*np.pi*(d.index.month-1)/12),
                    np.cos(2*np.pi*(d.index.month-1)/12)]
        X = np.column_stack(features); y = d.fare_inflation.to_numpy()
        use = np.isfinite(X).all(axis=1) & np.isfinite(y)
        dates, A, y = d.index.asi8[use], X[use], y[use]
        inv = np.linalg.inv(A.T@A); b = inv@A.T@y
        weighted = A*(y-A@b)[:,None]
        kernel = np.maximum(1-abs(dates[:,None]-dates[None,:])/4, 0)
        expected = np.sqrt(np.diag(inv@weighted.T@kernel@weighted@inv)*len(y)/(len(y)-A.shape[1]))
        np.testing.assert_allclose(run.metrics.estimate,b)
        np.testing.assert_allclose(run.metrics.HAC_SE,expected)
        self.assertNotIn("estimate", run.audit["rounds"][-1]["evidence"])

    def test_result_review_revises_text_without_refitting(self):
        replies = Replies(ESTIMATE, revise=False)
        decisions = iter(["revise", "accept"])
        replies.audit_interpretation = lambda c: Review(decision=next(decisions), reason="Review the wording.")
        result = _analyze(ESTIMATION_PROMPT, load_demo(), replies)
        self.assertEqual(replies.count, 1)
        self.assertEqual(len(result.audit["result_review"]), 2)

    def test_complete_example_uses_runnable_schema_and_live_review(self):
        replies = Replies(revise=False)
        def structured(role, schema, context):
            if schema is CompleteExampleAssessment:
                return CompleteExampleAssessment(decision="accept", message="Ready.", criteria=NOTES)
            if schema is WorkedForecastDraft:
                return WorkedForecastDraft(decision="run", code=FORECAST, reason="Fit the one-month forecast.")
            if schema is Review:
                return Review(decision="accept", reason="This matches the supplied evidence.")
            self.assertIs(schema, Interpretation)
            return replies.interpret(context)
        with patch.object(StudioAgents, "_call", side_effect=structured) as calls:
            result = analyze(FORECAST_PROMPT)
        self.assertEqual(calls.call_args_list[0].args[1], CompleteExampleAssessment)
        self.assertEqual(calls.call_args_list[1].args[1], WorkedForecastDraft)
        self.assertEqual(calls.call_count, 6)
        self.assertAlmostEqual(result.audit["next_month"]["forecast"], 0.1023773, places=5)

    def test_spurious_clarification_can_be_recovered(self):
        replies = Replies(revise=False)
        initial = Draft(decision="ask_user", code="", reason="Specify how to combine the models.")
        with patch.object(replies, "draft", side_effect=[initial, replies.draft({})]), \
             patch.object(replies, "check_clarification", create=True, return_value=ClarificationCheck(
                 decision="revise", reason="The package already fits convex weights.")):
            result = _analyze(DEMO_PROMPT, load_demo(), replies)
        self.assertEqual(result.audit["rounds"][0]["clarification_check"]["decision"], "revise")
        self.assertEqual(result.audit["rounds"][-1]["analyst_confirmation"]["decision"], "accept")

    def test_page_request_to_final_result(self):
        with StudioServer(("127.0.0.1",0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                page = urlopen(url).read().decode()
                self.assertIn("Run with local Llama", page)
                self.assertNotIn("__FORECAST_FIELDS__", page)
                self.assertIn("Run reference forecast (Python only)", page)
                def reference(prompt, csv=None):
                    req = Request(url+"/analyze", data=json.dumps(
                        {"mode":"reference","prompt":prompt,"csv":csv}).encode(),
                        headers={"Content-Type":"application/json"})
                    return [json.loads(line) for line in urlopen(req)][-1]
                with patch("studio_core.app.analyze", side_effect=AssertionError("unexpected live call")):
                    self.assertEqual(reference(FORECAST_PROMPT+" edited")["type"],"stopped")
                    self.assertEqual(reference(FORECAST_PROMPT,load_demo().to_csv(index=False))["type"],"stopped")
                    fixed = reference(FORECAST_PROMPT)["report"]
                self.assertEqual(fixed["audit"]["agent_mode"],"Python reference only")
                self.assertEqual(len(fixed["holdout_forecasts"]),24)
                self.assertAlmostEqual(fixed["audit"]["next_month"]["forecast"],.1023773,places=5)
                request = Request(url+"/analyze",data=json.dumps({"prompt":ESTIMATION_PROMPT,
                    "csv":load_demo().to_csv(index=False)}).encode(),headers={"Content-Type":"application/json"})
                with patch("studio_core.studio.StudioAgents",return_value=Replies(ESTIMATE)):
                    rows = [json.loads(line) for line in urlopen(request)]
                self.assertEqual(rows[-1]["type"],"result")
                self.assertEqual(rows[-1]["report"]["task"],"estimation")
                self.assertTrue(any(r.get("decision")=="revise" for r in rows))
            finally:
                server.shutdown();thread.join()
