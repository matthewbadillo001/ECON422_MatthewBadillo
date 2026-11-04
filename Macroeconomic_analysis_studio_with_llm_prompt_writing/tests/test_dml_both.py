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

from studio_core.api import cross_section, cross_dml, ts_dml, load_cross_demo, load_activity_demo, load_demo
from studio_core.analysis_tools import monthly_features
from studio_core.app import StudioServer,Handler
from studio_core.example_prompt import (CROSS_FIELDS,CROSS_CODE,CROSS_SCREEN_FIELDS,
    TS_EST_CODE,TS_EST_FRED_CODE,TS_EST_FIELDS,TS_EST_FRED_FIELDS)
from studio_core.studio import parse_program
import dml
from test_studio import Replies
from studio_core.spec import Review


class DualDmlTests(unittest.TestCase):
    def test_cross_score_independent_folds_and_reproducible_robust_se(self):
        demo=load_cross_demo(n=300)
        demo.loc[7,"education"]=np.nan
        design=cross_section(demo,target="earnings",exposure="training_hours",controls=["education","experience"])
        residuals=[];test_ids=[]
        original=dml._residuals
        def record(train,test,*args):
            self.assertTrue(train.index.intersection(test.index).empty)
            test_ids.extend(test.index)
            residuals.append(original(train,test,*args))
            return residuals[-1]
        with patch.object(dml,"_residuals",side_effect=record):
            res=cross_dml(design)
        self.assertEqual(len(set(test_ids)),299)
        u,v=np.vstack(residuals).T
        theta=v@u/(v@v)
        psi=v*(u-theta*v)/(v@v)
        se=np.sqrt(psi@psi*299/298)
        self.assertAlmostEqual(res.audit["estimate"]["coefficient"],theta)
        self.assertAlmostEqual(res.audit["estimate"]["SE"],se)
        self.assertLess(abs(theta-1.5),.3)
        self.assertNotIn("Time reversibility",res.metrics.index)
        self.assertEqual(res.audit["n"],299)

    def test_cross_screen_weak_or_binary_overlap_stops_estimation(self):
        demo=load_cross_demo(n=300)
        demo["training_hours"]=demo.education
        terms=dict(target="earnings",exposure="training_hours",controls=["education","experience"])
        self.assertIsNone(cross_dml(cross_section(demo,**terms)).audit["estimate"])
        demo["training_hours"]=(demo.education>0).astype(int)
        res=cross_dml(cross_section(demo,**terms))
        self.assertEqual(res.metrics.loc["Binary exposure overlap","Assessment"],"Concern")
        self.assertIsNone(res.audit["estimate"])
        demo["training_hours"]=np.random.default_rng(17).binomial(1,.5,len(demo))
        res=cross_dml(cross_section(demo,**terms),estimate=False)
        self.assertIsNone(res.audit["estimate"]);self.assertIn("Binary exposure overlap",res.metrics.index)

    def test_ts_aggregation_hac_and_holdout_isolation(self):
        frame=load_activity_demo()
        (_,features),(_,opts)=parse_program(TS_EST_CODE)
        segments=[]
        original=dml._residuals
        def record(train,test,*args):
            self.assertTrue(train.index.intersection(test.index).empty)
            arr=original(train,test,*args);segments.append((test.index,arr))
            return arr
        with patch.object(dml,"_residuals",side_effect=record):
            res=ts_dml(monthly_features(frame,**features),**opts)
        self.assertIsNotNone(res.audit["estimate"])
        folded=[];influence=pd.Series(0.,index=pd.PeriodIndex(frame.date,freq="M"))
        for ix,arr in segments:
            u,v=arr.T;b=v@u/(v@v);folded.append(b)
            influence.loc[ix]=v*(u-b*v)/(len(segments)*(v@v))
        g=influence.to_numpy()
        variance=g@g + 2*sum((1-h/4)*(g[h:]@g[:-h]) for h in range(1,4))
        self.assertAlmostEqual(res.audit["estimate"]["coefficient"],np.mean(folded))
        self.assertAlmostEqual(res.audit["estimate"]["SE"],np.sqrt(variance))
        self.assertLess(abs(np.mean(folded)-.4),.15)
        (_,f),(_,o)=parse_program(TS_EST_FRED_CODE)
        with patch.object(dml,"_residuals",side_effect=AssertionError("estimation after warning")):
            self.assertIsNone(ts_dml(monthly_features(load_demo(),**f),**o).audit["estimate"])

    def test_ui_reference_live_feedback_replay_and_export(self):
        with StudioServer(("127.0.0.1",0),Handler) as server:
            threading.Thread(target=server.serve_forever,daemon=True).start()
            base=f"http://127.0.0.1:{server.server_port}"
            def post(body):
                with urlopen(Request(base+"/analyze",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})) as response:
                    return [json.loads(row) for row in response][-1]
            try:
                for fields,code,kind in [(CROSS_FIELDS,CROSS_CODE,"cross_section"),(TS_EST_FIELDS,TS_EST_CODE,"monthly")]:
                    body=dict(mode="reference",prompt_fields=fields,data_kind=kind)
                    ref=post(body)["report"]
                    worker=Replies(code,revise=False)
                    with patch("studio_core.studio.StudioAgents",return_value=worker), \
                         patch.object(worker,"critic",return_value=Review(decision="accept",reason="Matches the requested controls and timing.")):
                        live=post({**body,"mode":"live"})["report"]
                        rev=post(dict(parent_id=ref["id"],feedback="Keep the specification."))["report"]
                    for report in [ref,live,rev]:
                        self.assertEqual(report["data_kind"],kind)
                        self.assertIsNotNone(report["audit"]["estimate"])
                        self.assertEqual(report["audit"]["estimate"],ref["audit"]["estimate"])
                    with patch("sys.argv",["replay.py"]),contextlib.redirect_stdout(io.StringIO()):
                        scope={};exec(ref["audit"]["python_script"],scope)
                    self.assertEqual(scope["result"].audit["estimate"],ref["audit"]["estimate"])
                    with urlopen(base+"/reports/"+ref["id"]+".csv") as f:
                        self.assertIn("coefficient",f.read().decode("utf-8-sig"))
                    with urlopen(base+"/reports/"+ref["id"]+".pdf") as f:self.assertTrue(f.read().startswith(b"%PDF"))
                custom=post(dict(mode="reference",prompt_fields=CROSS_FIELDS,csv=load_cross_demo(n=180).to_csv(index=False),filename="households.csv"))["report"]
                self.assertEqual(custom["audit"]["n"],180);self.assertIn("Pass the downloaded",custom["audit"]["python_script"])
                assessment=post(dict(mode="reference",prompt_fields=CROSS_SCREEN_FIELDS))["report"]
                self.assertIsNone(assessment["audit"]["estimate"]);self.assertFalse(assessment["audit"]["requested_estimate"])
                self.assertEqual(post(dict(mode="reference",prompt_fields=TS_EST_FRED_FIELDS,source_mode="fred"))["type"],"needs_source")
                self.assertEqual(post(dict(mode="reference",prompt_fields=CROSS_FIELDS,source_mode="fred"))["type"],"stopped")
            finally: server.shutdown()
