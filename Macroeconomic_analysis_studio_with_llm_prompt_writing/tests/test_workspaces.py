import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from studio_core.app import Handler, StudioServer
from studio_core.example_prompt import FIELD_HELP
from studio_core.studio import parse_program


class Workspaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=StudioServer(("127.0.0.1",0),Handler)
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
        cls.url=f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close()

    def post(self,route,body):
        request=Request(self.url+route,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        with urlopen(request) as response: return json.load(response)

    def test_practice_reviews_only_the_five_written_fields(self):
        draft={k:"" for k in FIELD_HELP};draft["clarity"]="Forecast next month's retail sales."
        notes={k:"Writing feedback about "+k for k in draft}
        with patch("studio_core.app.StudioAgents.review_writing",return_value=notes) as writing, \
             patch("studio_core.app.analyze",side_effect=AssertionError("Analysis started")), \
             patch("studio_core.app.load_retail_demo",side_effect=AssertionError("Data loaded")):
            r=self.post("/practice-review",{"prompt_fields":draft})
        writing.assert_called_once_with(draft)
        self.assertEqual(r["source"],"Local Llama writing feedback")
        self.assertEqual(r["notes"],notes)
        self.assertFalse(self.server.runs)

    def test_empty_drafts_credentials_and_offline_review(self):
        fields=dict.fromkeys(FIELD_HELP,"")
        with self.assertRaises(HTTPError) as err:self.post("/practice-review",{"prompt_fields":fields})
        self.assertEqual(err.exception.code,400)
        fields["context"]="api_key=abcdefghijklmnopqrstuvwxyz"
        with self.assertRaises(HTTPError) as err:self.post("/practice-review",{"prompt_fields":fields})
        self.assertEqual(err.exception.code,422)
        fields["context"]="Use monthly CPI levels."
        with patch("studio_core.app.StudioAgents.review_writing",side_effect=RuntimeError("Ollama offline")):
            r=self.post("/practice-review",{"prompt_fields":fields})
        self.assertIn("Self-review",r["source"]);self.assertEqual(set(r["notes"]),set(FIELD_HELP))

    def test_removed_model_route_and_program_are_unavailable(self):
        with self.assertRaises(HTTPError) as err:self.post("/joint-template",{})
        self.assertEqual(err.exception.code,403)
        with self.assertRaises(ValueError):
            parse_program("design = joint_data(data, outcomes=['y','x'], first_controls=[], second_controls=[])\nresult = joint_var(design, lags=1, holdout_months=24)")
