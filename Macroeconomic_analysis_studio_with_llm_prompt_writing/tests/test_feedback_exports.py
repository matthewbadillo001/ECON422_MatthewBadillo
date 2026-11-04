import csv
import io
import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from studio_core.app import Handler, StudioServer
from studio_core.example_prompt import FORECAST_PROMPT
from test_studio import Replies, FORECAST, ESTIMATE


class FeedbackAndExportsTests(unittest.TestCase):
    def setUp(self):
        self.server = StudioServer(("127.0.0.1",0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def run_request(self, body):
        req = Request(self.url+"/analyze", data=json.dumps(body).encode(),
                      headers={"Content-Type":"application/json"})
        return [json.loads(line) for line in urlopen(req)]

    def download(self, report, fmt):
        with urlopen(self.url+"/reports/"+report["id"]+"."+fmt) as resp:
            self.assertIn("attachment",resp.headers["Content-Disposition"])
            return resp.read()

    def test_feedback_reruns_same_source_and_exports_each_version(self):
        first = self.run_request({"mode":"reference","prompt":FORECAST_PROMPT})[-1]["report"]
        original_csv = self.download(first,"csv")
        note = "Use two fare lags. Keep the fuel lag and 24-month holdout."
        replies = Replies(FORECAST.replace("[1,2,3]", "[1,2]"),revise=False)
        with patch("studio_core.studio.StudioAgents",return_value=replies), \
             patch.object(replies,"draft",wraps=replies.draft) as draft, \
             patch.object(replies,"critic",wraps=replies.critic) as critic, \
             patch.object(replies,"interpret",wraps=replies.interpret) as interpret:
            run = self.run_request({"parent_id":first["id"],"feedback":note})
        second = run[-1]["report"]
        for agent in (draft,critic,interpret):
            self.assertEqual(agent.call_args.args[0]["user_revision"]["latest_user_feedback"],note)
        self.assertEqual(second["audit"]["accepted_plan"]["target_lags"],[1,2])
        self.assertNotEqual(first["rows"],second["rows"])
        self.assertEqual(second["version"],2)
        self.assertEqual(second["parent_id"],first["id"])
        self.assertEqual(first["source"],second["source"])
        self.assertTrue(any("target_lags" in c for c in second["changes"]))
        pending = [r["role"] for r in run if r.get("phase")=="working"]
        self.assertIn("Critic",pending)
        self.assertIn("Python",pending)
        data = list(csv.DictReader(io.StringIO(self.download(second,"csv").decode("utf-8-sig"))))
        self.assertTrue(any(row["value"] == note for row in data))
        self.assertTrue(any(row["decision"] == "accept" for row in data))
        actual = next(row for row in data if row["section"]=="results" and row["model"]=="super_learner")
        self.assertAlmostEqual(float(actual["RMSE"]),dict((r[0],r[1]) for r in second["rows"])["super_learner"])
        self.assertEqual(sum(row["section"]=="holdout" for row in data),24)
        self.assertTrue(self.download(second,"pdf").startswith(b"%PDF"))
        self.assertEqual(self.download(first,"csv"),original_csv)

        # A failed follow-up must not create a version or remove the last approved one.
        with patch("studio_core.app.analyze",side_effect=ValueError("Unsupported request")):
            fail = self.run_request({"parent_id":second["id"],"feedback":"New estimator, please."})
        self.assertEqual(fail[-1]["type"],"stopped")
        self.assertEqual(len(self.server.runs),2)
        with self.assertRaises(HTTPError) as exc:
            self.run_request({"parent_id":"missing","feedback":"Adjust the lags."})
        self.assertEqual(exc.exception.code,410)

    def test_custom_estimation_data_is_reused_for_feedback(self):
        from studio_core.api import load_demo
        data=load_demo().iloc[:125]
        with patch("studio_core.studio.StudioAgents",return_value=Replies(ESTIMATE,revise=False)):
            one = self.run_request({"prompt":"Estimate the lagged fuel association.","csv":data.to_csv(index=False),
                                    "filename":"short.csv"})[-1]["report"]
            two = self.run_request({"parent_id":one["id"],"feedback":"Explain the interval for a student."})[-1]["report"]
        self.assertEqual(two["source"],"short.csv")
        self.assertEqual(one["rows"],two["rows"])
        self.assertEqual(one["audit"]["n"],two["audit"]["n"])
        self.assertIn("HAC_SE",self.download(two,"csv").decode("utf-8-sig"))
        self.assertTrue(self.download(two,"pdf").startswith(b"%PDF"))
