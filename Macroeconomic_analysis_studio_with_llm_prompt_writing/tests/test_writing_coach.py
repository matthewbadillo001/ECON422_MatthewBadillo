import unittest
from unittest.mock import patch

from studio_core.api import PromptNeedsAnswer, PromptRewriteOffered, analyze, load_demo
from studio_core.studio import PromptAssessment, _analyze
from test_studio import NOTES, Replies


ROUGH = "next month fares. fuel last month. forecast please"
REWRITE = ("Forecast next month's airfare inflation using last month's fuel inflation. "
           "Report the one-month-ahead prediction.")
EDITED = REWRITE + " Use plain language for my students."


class WritingCoachTests(unittest.TestCase):
    def setUp(self):
        self.agents = Replies(revise=False)
        self.review = PromptAssessment(decision="accept", message="This makes the forecast horizon explicit.",
                                       criteria=NOTES, suggested_prompt=REWRITE)

    def test_rough_prompt_offer_and_author_choices(self):
        with patch.object(self.agents, "review_prompt", return_value=self.review) as review, \
             patch.object(self.agents, "draft", wraps=self.agents.draft) as draft, \
             patch.object(self.agents, "critic", wraps=self.agents.critic) as critic:
            with self.assertRaises(PromptRewriteOffered) as offer:
                _analyze(ROUGH, load_demo(), self.agents, pause_for_rewrite=True)
            self.assertEqual(str(offer.exception), REWRITE)
            draft.assert_not_called()
            for action, expected in [("keep",ROUGH), ("use",REWRITE), ("edit",EDITED)]:
                with self.subTest(action=action):
                    choice={"action":action, "suggestion":REWRITE, "edited":EDITED if action=="edit" else ""}
                    r=_analyze(ROUGH, load_demo(), self.agents, pause_for_rewrite=True, wording=choice)
                    for worker in (review,draft,critic):
                        self.assertEqual(worker.call_args.args[0]["original_prompt"], expected)
                    self.assertEqual(r.audit["original_prompt"], ROUGH)
                    self.assertEqual(r.audit["analysis_prompt"], expected)
                    self.assertEqual(r.audit["writing_choice"]["used"], expected)

    def test_missing_information_still_requires_an_answer(self):
        assessment = PromptAssessment(decision="ask_user", message="Which series should I forecast?",
                                      criteria=NOTES, suggested_prompt="Forecast next month's [outcome].")
        with patch.object(self.agents, "review_prompt", return_value=assessment), patch.object(self.agents, "draft") as draft:
            for choice in (None, {"action":"edit","suggestion":REWRITE,"edited":"Forecast something."}):
                with self.assertRaises(PromptNeedsAnswer):
                    _analyze(ROUGH,load_demo(),self.agents,pause_for_rewrite=True,wording=choice)
            draft.assert_not_called()

    def test_revision_rewrites_only_feedback(self):
        revision={"latest_user_feedback":"two lags instead pls", "prior_analysis_prompt":EDITED,
                  "previous_approved_code":"Previously fitted forecast."}
        selected="Keep the original analysis, changing the number of airfare lags to two."
        with patch("studio_core.studio.StudioAgents",return_value=self.agents), \
             patch.object(self.agents,"review_prompt",return_value=self.review), \
             patch.object(self.agents,"draft",wraps=self.agents.draft) as draft:
            r=analyze(ROUGH,revision=revision,wording={"action":"use","suggestion":selected})
        ctx=draft.call_args.args[0]
        self.assertEqual(ctx["original_prompt"],EDITED)
        self.assertEqual(ctx["user_revision"]["latest_user_feedback"],selected)
        self.assertEqual(r.audit["writing_choice"]["original"],"two lags instead pls")
