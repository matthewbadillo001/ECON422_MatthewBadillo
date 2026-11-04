import csv
import io
import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from studio_core.api import load_demo
from studio_core.app import Handler, StudioServer
from studio_core.studio import PromptAssessment
from test_studio import Replies, NOTES, FORECAST


class PromptReviewerTests(unittest.TestCase):
    def test_pause_before_fitting_then_answer_and_export(self):
        prompt = "Forecast one of the two series."
        questions = ["Which column do you want to forecast?", "Which fuel lag is available at the forecast origin?"]
        answers = ["Forecast fare_inflation and include fuel.", "Use fuel_inflation lag 1."]
        data = load_demo().iloc[:170]
        request = {"prompt":prompt,"csv":data.to_csv(index=False),"filename":"my_months.csv"}
        agents = Replies(FORECAST, revise=False)
        original_check = agents.review_prompt

        def check(context):
            turns = context.get("clarification_answers", [])
            self.assertEqual(context["rows"], len(data))
            if len(turns) < 2:
                return PromptAssessment(decision="ask_user",message=questions[len(turns)],criteria=NOTES,suggested_prompt="")
            return original_check(context)

        with StudioServer(("127.0.0.1",0), Handler) as server:
            base = f"http://127.0.0.1:{server.server_port}"
            def post(body):
                req = Request(base+"/analyze",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
                return [json.loads(line) for line in urlopen(req)]
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                with patch("studio_core.studio.StudioAgents", return_value=agents), \
                     patch.object(agents, "review_prompt", side_effect=check), \
                     patch.object(agents, "draft", wraps=agents.draft) as draft, \
                     patch.object(agents, "critic", wraps=agents.critic) as critic:
                    turns=[]
                    for q, answer in zip(questions, answers):
                        stream=post({**request,"clarifications":turns})
                        self.assertEqual(stream[-1], {"type":"needs_input","question":q})
                        self.assertEqual(len(server.runs),0)
                        draft.assert_not_called()
                        self.assertFalse(any(e.get("role")=="Python" for e in stream))
                        turns.append({"question":q,"answer":answer})
                    stream=post({**request,"clarifications":turns})
                    r=stream[-1]["report"]
                    self.assertEqual(r["source"], "my_months.csv")
                    self.assertEqual(r["version"],1)
                    self.assertEqual(r["audit"]["original_prompt"],prompt)
                    self.assertEqual(r["audit"]["prompt_clarifications"],turns)
                    self.assertEqual(draft.call_args.args[0]["clarification_answers"],turns)
                    self.assertEqual(critic.call_args.args[0]["clarification_answers"],turns)
                records=list(csv.DictReader(io.StringIO(urlopen(base+"/reports/"+r["id"]+".csv").read().decode("utf-8-sig"))))
                self.assertTrue(any(x["role"]=="User" and x["value"]==answers[0] for x in records))
                self.assertTrue(any(x['criteria'] and 'structure' in x['criteria'] for x in records))
                self.assertTrue(urlopen(base+"/reports/"+r["id"]+".pdf").read().startswith(b"%PDF"))
                # Later user feedback inherits clarification answers from version 1.
                with patch("studio_core.studio.StudioAgents",return_value=agents), \
                     patch.object(agents,"draft",wraps=agents.draft) as revise:
                    r2=post({"parent_id":r["id"],"feedback":"Keep the model and explain the units."})[-1]["report"]
                    self.assertEqual(revise.call_args.args[0]["user_revision"]["prior_prompt_clarifications"],turns)
                    self.assertEqual(r2["audit"]["prompt_clarifications"],turns)
            finally:
                server.shutdown();thread.join()
