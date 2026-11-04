import io
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from studio_core.api import load_demo
from studio_core.app import Handler, StudioServer
from studio_core.example_prompt import FORECAST_FIELDS, INCOMPLETE_CLARITY
from studio_core.sources import NoRedirect
from studio_core.studio import Draft, PromptAssessment, chosen_forecast_target
from test_studio import FORECAST, NOTES, Replies


SCREEN_QUESTION = ("The user has not decided which series to forecast, so the analyst needs to ask the user "
                   "to choose a series to forecast. The package capabilities and user question do not match, "
                   "so the analyst cannot proceed without user input. The analyst will ask the user to specify "
                   "which series they would like to forecast (e.g. fare_inflation, fuel_inflation, or another series not listed).")
FAKE_KEY = "a0" * 16


class StudioSourceTests(unittest.TestCase):
    def setUp(self):
        self.server = StudioServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.worker.join()
        self.server.server_close()

    def post(self, path, body):
        request = Request(self.base+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
        with urlopen(request) as response:
            if path == "/analyze":
                return [json.loads(line) for line in response]
            return json.load(response)

    def test_submit_short_reply_once_even_if_reviewer_and_analyst_repeat_question(self):
        fields = {**FORECAST_FIELDS, "clarity": INCOMPLETE_CLARITY}
        args = {"prompt_fields":fields}
        question = self.post("/analyze", args)[-1]
        self.assertEqual(question["type"], "needs_input")
        turns = [{"question":question["question"], "answer":"fare_inflation"}]
        agents = Replies(revise=False)
        repeat = PromptAssessment(decision="ask_user", message=SCREEN_QUESTION, criteria=NOTES, suggested_prompt="")
        drafts = [Draft(decision="ask_user", reason=SCREEN_QUESTION, code=""),
                  Draft(decision="run", reason="Use the confirmed target.", code=FORECAST)]
        with patch("studio_core.studio.StudioAgents", return_value=agents), \
             patch.object(agents, "review_prompt", return_value=repeat), \
             patch.object(agents, "draft", side_effect=drafts) as draft:
            stream = self.post("/analyze", {**args, "clarifications":turns})
        report = stream[-1]["report"]
        self.assertFalse(any(e["type"] == "needs_input" for e in stream))
        self.assertEqual(report["audit"]["accepted_plan"]["target"], "fare_inflation")
        self.assertEqual(report["audit"]["prompt_clarifications"], turns)
        self.assertIn(INCOMPLETE_CLARITY, report["audit"]["original_prompt"])
        ctx = draft.call_args.args[0]
        self.assertNotIn(INCOMPLETE_CLARITY, ctx["original_prompt"])
        self.assertEqual(ctx["confirmed_forecast_target"], "fare_inflation")
        self.assertEqual(chosen_forecast_target([{"question":SCREEN_QUESTION,"answer":"fuel_inflation"}],load_demo()),"fuel_inflation")

    def test_distinct_missing_input_is_still_asked(self):
        question = "Which fuel lag would be available by the forecast origin?"
        agents = Replies()
        turns = [{"question":SCREEN_QUESTION,"answer":"fare_inflation"}]
        with patch("studio_core.studio.StudioAgents", return_value=agents), \
             patch.object(agents, "review_prompt", return_value=PromptAssessment(
                 decision="ask_user", message=question, criteria=NOTES, suggested_prompt="")):
            self.assertEqual(self.post("/analyze", {"prompt":INCOMPLETE_CLARITY,"clarifications":turns})[-1],
                             {"type":"needs_input", "question":question})
        for answer in ["date", "made_up", "fare_inflation or fuel_inflation"]:
            self.assertIsNone(chosen_forecast_target([{"question":SCREEN_QUESTION,"answer":answer}], load_demo()))

    def test_fred_import_forecast_and_revision_never_expose_key(self):
        args = {"prompt":"Forecast next month's UNRATE using FRED observations."}
        sample = load_demo().iloc[:160]
        observations = [{"date":d, "value":str(v)} for d,v in zip(sample.date,sample.fare_inflation)]
        def fred_reply(url, timeout):
            dest = urlsplit(url)
            self.assertEqual((dest.scheme, dest.hostname, dest.path),
                             ("https","api.stlouisfed.org","/fred/series/observations"))
            self.assertEqual(parse_qs(dest.query)["api_key"], [FAKE_KEY])
            self.assertEqual(parse_qs(dest.query)["frequency"], ["m"])
            return io.BytesIO(json.dumps({"observations":observations}).encode())
        with patch("studio_core.sources.build_opener") as opener, patch("studio_core.app.analyze") as start:
            self.assertEqual(self.post("/analyze", args)[-1]["type"], "needs_source")
            start.assert_not_called()
            opener.assert_not_called()
            opener.return_value.open.side_effect = fred_reply
            data = self.post("/sources/fred", {"series_ids":"UNRATE", "api_key":FAKE_KEY})
            self.assertIsInstance(opener.call_args.args[0], NoRedirect)
            opener.return_value.open.assert_called_once()
        self.assertNotIn(FAKE_KEY,json.dumps(data))
        self.assertTrue(data["csv"].startswith("date,UNRATE"))
        code = FORECAST.replace('"fare_inflation"', '"UNRATE"').replace('["fuel_inflation"]','[]').replace('predictor_lags=[1]', 'predictor_lags=[]')
        agents = Replies(code, revise=False)
        with patch("studio_core.studio.StudioAgents", return_value=agents), \
             patch.object(agents, "review_prompt", wraps=agents.review_prompt) as review, \
             patch("studio_core.app.import_fred", side_effect=AssertionError("Unexpected re-fetch")):
            args.update(csv=data["csv"],filename=data["filename"])
            first=self.post("/analyze", args)[-1]["report"]
            second=self.post("/analyze", {"parent_id":first["id"], "feedback":"Keep the same model."})[-1]["report"]
        for call in review.call_args_list:
            ctx = call.args[0]
            self.assertEqual(ctx["columns"], ["date","UNRATE"])
            self.assertIn("FRED", ctx["loaded_data_source"])
            self.assertNotIn(FAKE_KEY, json.dumps(ctx))
        for result in [first,second]:
            self.assertNotIn(FAKE_KEY, json.dumps(result))
            self.assertEqual(result["source"], data["filename"])
            with urlopen(self.base+"/reports/"+result["id"]+".csv") as response:
                self.assertNotIn(FAKE_KEY.encode(), response.read())

    def test_source_can_be_requested_in_a_followup_and_csv_needs_no_key(self):
        body = {"prompt": "Forecast the unemployment rate.", "clarifications": [
            {"question": "Which data should I use?", "answer": "Use FRED series UNRATE."}]}
        with patch("studio_core.app.analyze") as start:
            self.assertEqual(self.post("/analyze", body)[-1]["type"], "needs_source")
            start.assert_not_called()
        self.assertIsNone(chosen_forecast_target([{"question":"Which target variable should I estimate?",
                                                  "answer":"fare_inflation"}], load_demo()))
        with patch("studio_core.studio.StudioAgents", return_value=Replies(revise=False)):
            body = {"prompt": "Forecast fare_inflation from my uploaded CSV exported from FRED.",
                    "csv":load_demo().to_csv(index=False), "filename":"my_data.csv"}
            result = self.post("/analyze", body)[-1]["report"]
        self.assertEqual(result["source"], "my_data.csv")

    def test_bad_key_is_retryable_and_errors_do_not_echo_credentials(self):
        with patch("studio_core.sources.build_opener") as opener:
            opener.return_value.open.side_effect = HTTPError(
                "https://example.invalid/?api_key="+FAKE_KEY,400,"bad",{},None)
            with self.assertRaises(HTTPError) as error:
                self.post("/sources/fred", {"series_ids":"UNRATE", "api_key":FAKE_KEY})
            reply = error.exception.read().decode()
            self.assertNotIn(FAKE_KEY,reply)
            self.assertIn("Check the key",reply)
        with self.assertRaises(HTTPError) as error:
            self.post("/analyze", {"prompt":"Use FRED. My key is "+FAKE_KEY})
        self.assertEqual(error.exception.code,422)
        self.assertNotIn(FAKE_KEY, error.exception.read().decode())
        with patch("studio_core.app.analyze") as start:
            result=self.post("/analyze", {"prompt":"Use the BEA API to forecast a series."})[-1]
            self.assertEqual(result["type"], "needs_source")
            self.assertEqual(result["provider"], "custom")
            start.assert_not_called()
