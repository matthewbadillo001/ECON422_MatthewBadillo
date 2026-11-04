import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from studio_core.api import load_demo
from studio_core.analysis_tools import monthly_features, rcf_dml_screen
from studio_core.app import Handler, StudioServer
from studio_core.example_prompt import AIRFARE_FRED_COLUMNS, DML_CODE, DML_FIELDS, DML_FRED_FIELDS, DML_PROMPT
from rcf_screen import FLAG, QUIET, UNKNOWN, _edge_records
from studio_core.studio import _analyze, parse_program
from studio_core.workflow import ReviewStopped
from test_studio import Replies, ESTIMATE


def series(phi=0, n=480):
    rng = np.random.default_rng(123)
    y, d = rng.normal(size=n), rng.normal(size=n)
    for i in range(1,n):
        d[i] += phi*d[i-1]
    return pd.DataFrame({"date":pd.date_range("1980-01-01",periods=n,freq="MS"),"y":y,"d":d})


def check(frame):
    design = monthly_features(frame,target="y",target_lags=[1],predictors=["d"],predictor_lags=[1],seasonality=False)
    return rcf_dml_screen(design,term="d__lag1")


class RcfTests(unittest.TestCase):
    def test_quiet_and_boundary_concern_are_distinct(self):
        quiet = check(series())
        self.assertEqual(quiet.metrics.Assessment.tolist(), [QUIET]*3)
        self.assertIn("candidate",quiet.audit["verdict"])
        self.assertEqual(quiet.audit["screen_status"],"no_flags")
        persistent = check(series(.93))
        self.assertEqual(persistent.metrics.Assessment.iloc[-1],FLAG)
        self.assertGreater(persistent.audit["evidence"]["Conditional stability"]["gain_D"],.1)
        self.assertIn("does not pass",persistent.audit["verdict"])
        self.assertEqual(persistent.audit["screen_status"],"does_not_pass")

    def test_ragged_short_and_constant_inputs_are_inconclusive(self):
        frame=series(n=100)
        frame.loc[40:58,"d"]=np.nan
        res=check(frame)
        self.assertEqual(res.audit["n"],41)
        self.assertEqual(res.metrics.Assessment.tolist(),[UNKNOWN]*3)
        frame["d"]=2.0
        self.assertEqual(check(frame).metrics.Assessment.iloc[-1],UNKNOWN)
        frame["d"]=np.nan
        self.assertEqual(check(frame).audit["n"],0)
        frame["d"]=frame.y
        self.assertEqual(check(frame).metrics.Assessment.iloc[-1],UNKNOWN)

    def test_reversed_blocks_rebuild_features_without_using_held_out_months(self):
        sample=series(n=96)
        design=monthly_features(sample,target="y",target_lags=[1],predictors=["d"],predictor_lags=[1],seasonality=False)
        blocks=[design.frame.iloc[idx] for idx in np.array_split(np.arange(96),8)]
        forward=_edge_records(blocks,3,1,design.options,"d__lag1","d")
        backward=_edge_records(blocks,3,-1,design.options,"d__lag1","d")
        month=backward.index[0]
        self.assertEqual(backward.loc[month,"y__lag1"],blocks[3].loc[month+1,"y"])
        self.assertEqual(backward.loc[month,"__D"],blocks[3].loc[month+1,"d"])
        self.assertEqual(forward.iloc[0]["__donor"],2)
        self.assertEqual(backward.iloc[0]["__donor"],4)
        self.assertNotIn(blocks[3].index[-1],backward.index)
        self.assertNotIn(blocks[3].index[0],forward.index)

    def test_agents_cannot_silently_switch_a_suitability_request_to_ols(self):
        with self.assertRaises(ReviewStopped) as stop:
            _analyze("Assess DML suitability for airfare.",load_demo(),Replies(ESTIMATE,revise=False))
        self.assertIn("requires rcf_dml_screen",str(stop.exception))

    def test_browser_paths_reports_replay_and_feedback(self):
        with StudioServer(("127.0.0.1",0),Handler) as server:
            threading.Thread(target=server.serve_forever,daemon=True).start()
            base=f"http://127.0.0.1:{server.server_port}"
            def post(body):
                with urlopen(Request(base+"/analyze",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})) as response:
                    return [json.loads(row) for row in response][-1]
            try:
                body={"mode":"reference","prompt_fields":DML_FIELDS}
                with patch("studio_core.app.analyze",side_effect=AssertionError("reference used Llama")):
                    ref=post(body)["report"]
                with patch("studio_core.studio.StudioAgents",return_value=Replies(DML_CODE,revise=False)):
                    live=post({**body,"mode":"live"})["report"]
                self.assertEqual(ref["rows"],live["rows"])
                for source in ("retail_demo", "ar_demo"):
                    with self.subTest(source=source), patch("studio_core.studio.StudioAgents",return_value=Replies(DML_CODE,revise=False)):
                        for mode in ("live","reference"):
                            report=post({**body,"mode":mode,"source_mode":source})["report"]
                            self.assertEqual(report["source"],"Bundled simulated monthly data")
                            self.assertEqual(report["rows"],ref["rows"])
                            self.assertEqual(report["audit"]["term"],"fuel_inflation__lag1")
                with patch("studio_core.studio.StudioAgents",side_effect=AssertionError("incompatible CSV reached Llama")):
                    wrong=post({**body,"mode":"live","source_mode":"retail_demo",
                                "csv":"date,unrelated\n2020-01-01,42\n","filename":"my.csv"})
                    self.assertEqual(wrong["type"],"stopped")
                    self.assertIn("fare_inflation, fuel_inflation",wrong["message"])
                self.assertEqual(ref["task"],"rcf_dml_screen")
                self.assertEqual([row[1] for row in ref["rows"]],[FLAG,FLAG,QUIET])
                self.assertIn("2012-05",ref["interpretation"])
                self.assertEqual(ref["audit"]["n"],152)
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/"replay.py"
                    p.write_text(ref["audit"]["python_script"])
                    with patch("sys.argv",[str(p)]),contextlib.redirect_stdout(io.StringIO()):
                        scope={};exec(p.read_text(),scope)
                    self.assertEqual(scope["result"].metrics.reset_index().values.tolist(),ref["rows"])
                with urlopen(base+"/reports/"+ref["id"]+".csv") as response:
                    self.assertIn("Conditional stability",response.read().decode("utf-8-sig"))
                with urlopen(base+"/reports/"+ref["id"]+".pdf") as response:
                    self.assertTrue(response.read().startswith(b"%PDF"))
                revised=DML_CODE.replace('target_lags=[1, 2, 3]','target_lags=[1, 2]')
                with patch("studio_core.studio.StudioAgents",return_value=Replies(revised,revise=False)):
                    later=post({"parent_id":ref["id"],"feedback":"Keep only own lags 1 and 2."})["report"]
                self.assertEqual(later["task"],ref["task"])
                self.assertNotIn("fare_inflation__lag3",later["audit"]["controls"])
                self.assertIn("target_lags",str(later["changes"]))
                ext={"mode":"reference","prompt_fields":DML_FRED_FIELDS,"source_mode":"fred"}
                self.assertEqual(post(ext)["settings"]["series_ids"],list(AIRFARE_FRED_COLUMNS))
                ext.update(csv=load_demo().to_csv(index=False),filename="FRED fixture")
                fred=post(ext)["report"]
                self.assertEqual(fred["task"],"rcf_dml_screen")
                self.assertIn("Pass the downloaded monthly input CSV",fred["audit"]["python_script"])
            finally:
                server.shutdown()
