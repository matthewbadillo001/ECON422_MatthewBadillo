import json
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd

from studio_core.api import DEMO_PROMPT, forecast, load_demo, run_demo, simulate_data
from studio_core.agents import DemoAgents
from studio_core.data_io import monthly_frame
from superlearner.forecasting import make_design
from studio_core.spec import Plan
from studio_core.workflow import ReviewStopped, _run


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_demo()

    def test_simulation_and_calendar_gap(self):
        pd.testing.assert_frame_equal(load_demo(), simulate_data())
        frame = monthly_frame(load_demo())
        plan = Plan(**self.result.audit["accepted_plan"])
        features, _ = make_design(frame, plan)
        self.assertTrue(pd.isna(features.loc["2012-05", "fuel_inflation__lag1"]))
        self.assertEqual(features.loc["2012-03", "fuel_inflation__lag1"], frame.loc["2012-02", "fuel_inflation"])

    def test_review_and_chronological_folds(self):
        audit = self.result.audit
        self.assertEqual([r["critic"]["decision"] for r in audit["rounds"]], ["revise", "accept"])
        self.assertEqual(audit["holdout_months_scored"], 24)
        self.assertEqual(audit["next_month"]["month"], "2025-01")
        w = list(audit["weights"].values())
        self.assertAlmostEqual(sum(w), 1)
        self.assertGreaterEqual(min(w), 0)
        pred = self.result.predictions
        np.testing.assert_allclose(pred.super_learner, pred[["lasso", "random_forest"]].to_numpy() @ w)
        for fold in audit["rounds"][-1]["evidence"]["folds"]:
            self.assertLess(fold["train_last"], fold["validate_first"])
            self.assertLess(fold["validate_last"], audit["holdout_start"])

    def test_holdout_and_future_targets_do_not_leak(self):
        data = load_demo()
        data.loc[data.index[-24:], "fare_inflation"] += 10
        altered = _run(DEMO_PROMPT, data, DemoAgents())
        self.assertEqual(altered.audit["rounds"], self.result.audit["rounds"])
        self.assertEqual(altered.predictions.super_learner.iloc[0], self.result.predictions.super_learner.iloc[0])
        self.assertNotEqual(altered.metrics.loc["super_learner","RMSE"], self.result.metrics.loc["super_learner","RMSE"])

    def test_local_agent_protocol(self):
        # Emulate Ollama JSON over the public forecast() path. No model download required.
        code = ('design = monthly_features(data, target="fare_inflation", '
                'target_lags=[1,2,3], predictors=["fuel_inflation"], predictor_lags=[1], '
                'seasonality=True)\nresult = super_learner(design, holdout_months=24)')
        payloads = iter([{"decision": "accept", "message":"Clear enough.", "suggested_prompt":"", "criteria":dict.fromkeys(("clarity","structure","context","role","examples"), "Clear.")},
                         {"decision": "run", "code": code, "reason": "Requested lags."},
                         {"decision": "accept", "reason": "Features match the prompt."},
                         {"decision": "accept", "reason": "I agree with the reviewed code."},
                         {"text": "The holdout comparison uses the accepted forecast."},
                         {"decision": "accept", "reason": "Consistent with the computed table."}])
        def server(req, timeout):
            sent = json.loads(req.data)
            self.assertFalse(sent["stream"])
            self.assertIsInstance(sent["format"], dict)
            self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/chat")
            raw = json.dumps({"message": {"content": json.dumps(next(payloads))}})
            return BytesIO(raw.encode())
        with patch("studio_core.agents.urlopen", side_effect=server):
            result = forecast(DEMO_PROMPT)
        pd.testing.assert_frame_equal(result.predictions, self.result.predictions)

    def test_clarification_stops_before_fit(self):
        ask = {"decision": "ask_user", "code": "", "reason": "Which target?"}
        replies = [{"decision":"accept","message":"Ready.","suggested_prompt":"","criteria":dict.fromkeys(("clarity","structure","context","role","examples"),"Clear.")}, ask, {"decision": "ask_user", "reason": "Which target should be forecast?"}]
        responses = [BytesIO(json.dumps({"message": {"content": json.dumps(r)}}).encode()) for r in replies]
        with patch("studio_core.agents.urlopen", side_effect=responses):
            with self.assertRaisesRegex(ReviewStopped, "Which target"):
                forecast("Forecast something.")


if __name__ == "__main__":
    unittest.main()
