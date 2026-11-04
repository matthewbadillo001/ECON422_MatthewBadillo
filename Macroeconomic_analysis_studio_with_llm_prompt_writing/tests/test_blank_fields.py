import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from studio_core.app import Handler, StudioServer
from studio_core.example_prompt import assembled_fields
from studio_core.studio import StudioAgents
from test_studio import Replies


class FieldCompletionTests(unittest.TestCase):
    def setUp(self):
        self.server = StudioServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.fields = {"clarity": "Forecast next month's fare_inflation.", "structure": " ",
                       "context": "Use monthly fare and fuel inflation.", "role": "", "examples": ""}

    def tearDown(self):
        self.server.shutdown()
        self.worker.join()
        self.server.server_close()

    def post(self):
        req = Request(self.url+"/analyze", data=json.dumps({"mode":"live", "prompt_fields":self.fields}).encode(),
                      headers={"Content-Type":"application/json"})
        return [json.loads(line) for line in urlopen(req)]

    def test_candidate_gate_is_driven_by_blanks_and_preserves_user_input(self):
        def provider(instructions, schema, context):
            self.assertEqual(context["written_fields"]["clarity"], self.fields["clarity"])
            self.assertIn("fuel_inflation", context["available_columns"])
            return schema(**{k:{"first":f"First {k}", "second":f"Second {k}"}
                             for k in context["requested_fields"]})
        with patch.object(StudioAgents, "_call", side_effect=provider), patch("studio_core.app.analyze") as live:
            response = self.post()[-1]
            self.assertEqual(response["type"], "needs_fields")
            self.assertEqual(set(response["candidates"]), {"structure","role","examples"})
            self.fields["role"] = response["candidates"]["role"][0]+" with my edit"
            self.fields["structure"] = "My own structure"
            self.assertEqual(set(self.post()[-1]["candidates"]), {"examples"})
            live.assert_not_called()
        self.fields["examples"] = "Use December fuel for a January prediction."
        agents=Replies(revise=False)
        with patch("studio_core.studio.StudioAgents",return_value=agents), \
             patch.object(agents,"draft",wraps=agents.draft) as draft:
            final=self.post()[-1]
        self.assertEqual(final["type"], "result")
        prompt=assembled_fields(self.fields)
        self.assertEqual(draft.call_args.args[0]["original_prompt"],prompt)
        self.assertEqual(final["report"]["audit"]["original_prompt"],prompt)

    def test_unavailable_reviewer_offers_starters_and_manual_entry(self):
        self.fields=dict.fromkeys(self.fields,"")
        with patch.object(StudioAgents, "_call", side_effect=RuntimeError("Offline")):
            result=self.post()[-1]
        self.assertEqual(set(result["candidates"]),set(self.fields))
        self.assertIn("unavailable",result["source"])
        self.assertTrue(all(len(opts)==2 for opts in result["candidates"].values()))

    def hints(self, targets, **extra):
        body = {"prompt_fields": self.fields, "targets": targets, **extra}
        req = Request(self.url + "/writing-hints", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req) as response:
            return json.load(response)

    def test_live_suggestions_use_draft_without_running_or_saving_analysis(self):
        self.fields["structure"] = "Start by fitting on earlier months."

        def reviewer(instructions, schema, ctx):
            self.assertEqual(ctx["written_fields"]["structure"], self.fields["structure"])
            self.assertEqual(set(ctx["requested_fields"]), {"structure", "role"})
            self.assertIn("NEXT sentence", instructions)
            self.assertEqual(ctx["available_columns"], [])
            return schema(structure={"first": "Then evaluate forecast errors.", "second": "Report units."},
                          role={"first": "Speak as an economist.", "second": "Address a student."})

        with patch.object(StudioAgents, "_call", side_effect=reviewer), patch("studio_core.app.analyze") as fit:
            result = self.hints(["structure", "role"], custom_csv=True)
        self.assertEqual(result["candidates"]["structure"][0], "Then evaluate forecast errors.")
        self.assertEqual(result["source"], "Reviewer suggestions")
        fit.assert_not_called()
        self.assertFalse(self.server.runs)

    def test_offline_live_structure_starter_matches_task(self):
        with patch.object(StudioAgents, "_call", side_effect=RuntimeError("Offline")):
            forecast = self.hints(["structure"])
            self.fields["clarity"] = "Estimate an association with fuel inflation."
            estimate = self.hints(["structure"])
        self.assertIn("unavailable", forecast["source"])
        self.assertIn("one-month-ahead", forecast["candidates"]["structure"][0])
        self.assertIn("coefficient", estimate["candidates"]["structure"][0])
        self.assertEqual(set(forecast["candidates"]), {"structure"})
        self.assertFalse(self.server.runs)
