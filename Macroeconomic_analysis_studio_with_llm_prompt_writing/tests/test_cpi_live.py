"""Live time-series assessment regressions. Exercise the Ollama JSON boundary without a local model."""
import io
import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from studio_core.app import Handler, StudioServer
from studio_core.example_prompt import DML_CODE, DML_FIELDS, DML_FRED_FIELDS, fred_starter
from test_studio import NOTES


class CpiLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = StudioServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, body):
        req = Request(f"http://127.0.0.1:{self.server.server_port}"+path,
                      data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
        with urlopen(req) as response:
            return json.load(response) if path=="/sources/fred" else [json.loads(line) for line in response]

    def setUp(self):
        dates = pd.date_range("2015-01-01", periods=120, freq="MS")
        self.data = pd.DataFrame({"date":dates.strftime("%Y-%m-%d"), "CPIAUCSL":210 + .5*pd.RangeIndex(120)})
        self.tasks = []
        self.contexts = []
        self.critic_decision = "accept"
        self.findings_advice = "The assumptions need to be addressed before estimating a DML coefficient."
        self.proposal = {"design":{"exposure_lag":1, "control_lags":[], "seasonality":False}, "reason":"Use last month's CPI."}
        self.agent = patch("studio_core.agents.urlopen", side_effect=self.reply)
        self.agent.start()
        self.addCleanup(self.agent.stop)

    def reply(self, request, timeout):
        body=json.loads(request.data)
        ctx=json.loads(body["messages"][1]["content"])
        schema=body["format"]["title"]
        role=body["messages"][0]["content"]
        self.tasks.append(schema); self.contexts.append(ctx)
        if schema in ("PromptAssessment", "CompleteExampleAssessment"):
            answer={"decision":"accept","criteria":NOTES,"suggested_prompt":"","message":"Ready."}
        elif schema in ("Draft", "WorkedForecastDraft"):
            answer={"decision":"run","code":self.custom_code,"reason":"Use the specified exposure and outcome."}
        elif schema=="ScreenExplanation":
            self.assertIn("CPIAUCSL__lag1",ctx["confirmed_fred_screen"])
            answer={"reason":"Assess current CPI with last month's CPI as the exposure."}
        elif schema=="OwnLagProposal":
            answer=self.proposal
        elif schema=="ClarificationCheck":
            answer={"decision":"ask_user","reason":"Which lag is the exposure?"}
        elif schema=="Review":
            if ctx.get("review_stage") == "findings":
                self.assertIn("completed time-series DML screen",role)
                self.assertEqual(len(ctx["diagnostics"]),3)
                if self.findings_advice is None:
                    raise TimeoutError("local model unavailable")
                answer={"decision":"revise","reason":self.findings_advice}
            else:
                self.assertEqual(ctx.get("review_stage"),"specification")
                self.assertNotIn("verdict",ctx["evidence"])
                self.assertNotIn("diagnostics",ctx["evidence"])
                answer={"decision":self.critic_decision,"reason":"Review the requested lag."}
        elif schema=="Interpretation":
            answer={"text":"The level-stationarity screen raised a concern."}
        else:
            self.fail(f"Unexpected role or unbound analyst code: {schema}")
        return io.BytesIO(json.dumps({"message":{"content":json.dumps(answer)}}).encode())

    def test_time_series_assessment_reports_each_verdict_despite_false_critic_objection(self):
        self.custom_code = DML_CODE
        self.critic_decision = "revise"
        self.findings_advice = ("The question asks to study fare_inflation as Y and last month's fuel_inflation as D, "
                                "but the executed design uses fare_inflation lags 1, 2, and 3 as X, "
                                "which does not match the question.")
        rng = np.random.default_rng(123)
        quiet = pd.DataFrame({"date": pd.date_range("1980-01-01", periods=480, freq="MS"),
                              "fare_inflation": rng.normal(size=480), "fuel_inflation": rng.normal(size=480)})
        cases = [
            ("bundled", {}, "does_not_pass"),
            ("keep original", {"wording": {"action": "keep", "suggestion": "A shorter version."}}, "does_not_pass"),
            ("quiet", {"csv": quiet.to_csv(index=False)}, "no_flags"),
            ("short", {"csv": quiet.head(30).to_csv(index=False)}, "inconclusive"),
        ]
        for label, sample, status in cases:
            with self.subTest(label=label):
                self.tasks.clear()
                body = {"mode": "live", "source_mode": "auto", "data_kind": "auto", "prompt_fields": DML_FIELDS, **sample}
                end = self.post("/analyze", body)[-1]
                self.assertEqual(end["type"], "result", end)
                ref = self.post("/analyze", {**body, "mode": "reference"})[-1]["report"]
                report = end["report"]
                audit = report["audit"]
                self.assertEqual((audit["screen_status"], report["rows"]), (status, ref["rows"]))
                self.assertEqual(len(report["rows"]), 3)
                self.assertNotIn("estimate", audit)
                self.assertEqual(self.tasks, ["CompleteExampleAssessment", "WorkedForecastDraft", "Review"])
                self.assertEqual(self.contexts[-1]["review_stage"], "findings")
                self.assertEqual(self.contexts[-1]["variable_roles"], {
                    "Y": "fare_inflation", "D": "fuel_inflation__lag1",
                    "X": ["fare_inflation__lag1", "fare_inflation__lag2", "fare_inflation__lag3", "calendar sine/cosine"]})
                self.assertEqual(audit["interpretation"], audit["verdict"])
                self.assertEqual(len(audit["rounds"]), 1)
                trace = audit["events"]
                self.assertTrue(any("Exposure D = fuel_inflation__lag1" in event["message"] for event in trace))
                self.assertFalse(any(self.findings_advice in event["message"] for event in trace))

    def test_worked_assessment_rejects_code_that_uses_a_control_as_the_exposure(self):
        self.custom_code = DML_CODE.replace('term="fuel_inflation__lag1"', 'term="fare_inflation__lag1"')
        end = self.post("/analyze", {"mode": "live", "prompt_fields": DML_FIELDS})[-1]
        self.assertEqual(end["type"], "stopped")
        self.assertIn("Use the worked RCF-DML screen specification", end["message"])
        self.assertNotIn("Review", self.tasks)

    def test_import_confirm_run_and_report(self):
        initial={"mode":"live","source_mode":"fred","prompt_fields":DML_FRED_FIELDS}
        self.assertEqual(self.post("/analyze",initial)[-1]["type"],"needs_source")
        obs = [{"date":date,"value":str(val)} for date,val in self.data.itertuples(index=False)]
        with patch("studio_core.sources._fred_response",return_value=json.dumps({"observations":obs}).encode()):
            imported=self.post("/sources/fred",dict(series_ids="`CPIAUCSL`",api_key="test"*8,starter_task="rcf_dml_screen"))
        body={**initial,"prompt_fields":imported["starter"],"csv":imported["csv"],"filename":imported["filename"],"fred_task":"rcf_dml_screen"}
        events=self.post("/analyze",body)
        self.assertEqual(events[-1]["type"],"result",events[-1])
        report=events[-1]["report"]
        self.assertEqual(report["task"],"rcf_dml_screen")
        self.assertEqual(report["audit"]["term"],"CPIAUCSL__lag1")
        self.assertEqual(report["rows"][0][1],"Concern")
        self.assertEqual(len(report["audit"]["rounds"]),1)
        self.assertEqual(report["audit"]["screen_status"],"does_not_pass")
        self.assertIn("does not pass",report["summary"])
        self.assertEqual(report["audit"]["rounds"][0]["findings_comment"]["decision"],"revise")
        self.assertTrue(any(e.get("decision")=="screen_complete" for e in report["audit"]["events"]))
        self.assertFalse(any(e.get("decision") in ("stopped","revise") for e in report["audit"]["events"]))
        self.assertEqual(self.tasks,["CompleteExampleAssessment","ScreenExplanation","Review"])
        ref=self.post("/analyze",{**body,"mode":"reference"})[-1]["report"]
        self.assertEqual(ref["rows"],report["rows"])
        self.assertNotIn("estimate",report["audit"])

    def test_custom_blank_form_preserves_lags_and_real_clarification(self):
        fields,_=fred_starter(["CPIAUCSL"],"rcf_dml_screen")
        fields["clarity"]="Assess time-series DML suitability for monthly `CPIAUCSL` in levels."
        fields["context"]="I have not decided which lag of CPIAUCSL to use."
        body=dict(mode="live",source_mode="fred",prompt_fields=fields,csv=self.data.to_csv(index=False),filename="FRED monthly data (CPIAUCSL)")
        self.proposal={"design":None,"reason":"Which lag is the exposure?"}
        q=self.post("/analyze",body)[-1]
        self.assertEqual(q["type"],"needs_input")
        self.proposal={"design":{"exposure_lag":2,"control_lags":[1],"seasonality":True},"reason":"Use the user's clarification."}
        res=self.post("/analyze",{**body,"clarifications":[{"question":q["question"],"answer":"Use lag 2 as D, lag 1 as X, and annual seasonality."}]})[-1]
        self.assertEqual(res["type"],"result",res)
        audit=res["report"]["audit"]
        self.assertEqual(audit["term"],"CPIAUCSL__lag2")
        self.assertEqual(audit["controls"],["CPIAUCSL__lag1","calendar sine/cosine"])
        self.assertEqual(len(audit["rounds"]),1)

    def test_estimate_request_completes_with_no_coefficient(self):
        fields,self.custom_code=fred_starter(["CPIAUCSL"],"ts_dml")
        report=self.post("/analyze",dict(mode="live",source_mode="fred",prompt_fields=fields,csv=self.data.to_csv(index=False),filename="FRED monthly data (CPIAUCSL)"))[-1]["report"]
        self.assertEqual(report["task"],"ts_dml")
        self.assertEqual(report["audit"]["screen_status"],"does_not_pass")
        self.assertIsNone(report["audit"]["estimate"])
        self.assertIn("Coefficient and interval withheld.",report["summary"])
        self.assertTrue(any(ctx.get("review_stage")=="specification" for ctx in self.contexts))
        self.assertEqual(report["audit"]["rounds"][0]["findings_comment"]["decision"],"revise")

    def test_free_text_structure_asks_for_screen_and_same_revision_stops(self):
        fields,_=fred_starter(["CPIAUCSL"],"rcf_dml_screen")
        fields["clarity"]="Use time-series DML for monthly CPIAUCSL persistence."
        fields["structure"]="Assess the three conditions only. Report the diagnostic evidence."
        self.critic_decision="revise"
        res=self.post("/analyze",dict(mode="live",source_mode="fred",prompt_fields=fields,csv=self.data.to_csv(index=False),filename="FRED monthly data (CPIAUCSL)"))[-1]
        self.assertEqual(res["type"],"stopped")
        self.assertIn("same assessment again",res["message"])
        self.assertEqual(self.tasks.count("Review"),1)

    def test_inconclusive_returns_report_even_if_comment_unavailable(self):
        fields,_=fred_starter(["CPIAUCSL"],"rcf_dml_screen")
        self.findings_advice=None
        res=self.post("/analyze",dict(mode="live",source_mode="fred",prompt_fields=fields,csv=self.data.head(30).to_csv(index=False),filename="FRED monthly data (CPIAUCSL)"))[-1]
        self.assertEqual(res["type"],"result",res)
        report=res["report"]
        self.assertEqual(report["audit"]["screen_status"],"inconclusive")
        self.assertIn("inconclusive",report["summary"])
        self.assertNotIn("does not pass",report["summary"])
        self.assertEqual([row[1] for row in report["rows"]],["Inconclusive"]*3)
        self.assertIn("comment_error",report["audit"]["rounds"][0])
