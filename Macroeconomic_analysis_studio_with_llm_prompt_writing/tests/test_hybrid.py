import contextlib
import io
import json
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
import numpy as np
import pandas as pd

from hybrid_forecasting import _block,_ma1_forecasts,family_forecast,family_development
from studio_core.family_prompts import CONFIGS,FIELDS,template,fred_task
from studio_core.app import Handler,StudioServer
from exploration import monthly_series
from studio_core.data_io import load_retail_demo
from test_studio import Replies


class HybridMethods(unittest.TestCase):
    def setUp(self):
        self.series=monthly_series(load_retail_demo(),target="MRTSSM44000USN")
        self.options=dict(mode="compare",base="seasonal_trend",learner="forest",holdout_months=24)

    def test_selection_forecast_decomposition_and_purity(self):
        r=family_forecast(self.series,**self.options)
        parts=r.audit["next_components"]
        self.assertAlmostEqual(sum(parts.values()),r.metrics.loc["Hybrid","Next forecast"])
        for mode,label in [("hybrid","Hybrid"),("classical","Classical only"),("ml","ML only")]:
            only=family_forecast(self.series,**{**self.options,"mode":mode})
            np.testing.assert_allclose(only.metrics.loc[label],r.metrics.loc[label])
        with patch("hybrid_forecasting._learner",side_effect=AssertionError("ML in classical path")):
            family_forecast(self.series,**{**self.options,"mode":"classical"})
        with patch("hybrid_forecasting._fit_base",side_effect=AssertionError("base in ML-only path")):
            family_forecast(self.series,**{**self.options,"mode":"ml"})

    def test_holdout_exclusion_and_future_shock(self):
        changed=self.series.copy()
        changed.iloc[-24:]+=100000
        self.assertEqual(family_development(self.series,**self.options),family_development(changed,**self.options))
        end=len(changed);cut=end-24
        a,_=_block(self.series,["hybrid"],"seasonal_trend","forest",cut,end)
        b,_=_block(changed,["hybrid"],"seasonal_trend","forest",cut,end)
        self.assertAlmostEqual(a.iloc[0,0],b.iloc[0,0])
        self.assertNotAlmostEqual(a.iloc[-1,0],b.iloc[-1,0])

    def test_historical_residual_targets_are_forward_predictions(self):
        captured=[]
        from hybrid_forecasting import _learner
        class Spy:
            def fit(self,x,y):
                self.model=_learner("lasso").fit(x,y);captured.append(y.copy());return self
            def predict(self,x): return self.model.predict(x)
        cutoff=150
        with patch("hybrid_forecasting._learner",return_value=Spy()):
            _block(self.series,["hybrid"],"trend","lasso",cutoff,cutoff+1)
        first=24
        y=self.series.to_numpy()
        beta=np.polyfit(np.arange(first),y[:first],1)
        self.assertAlmostEqual(captured[0][0],y[first]-np.polyval(beta,first),places=6)

    def test_benchmarks_and_ar_use_only_the_requested_past_months(self):
        series = self.series
        start = 170
        history = series.iloc[:start]
        level = history.iloc[0]
        for observed in history.iloc[1:]:
            level = .3 * observed + .7 * level
        expected = {
            "ses": level,
            "naive": history.iloc[-1],
            "seasonal_naive": history.iloc[-12],
            "moving_average3": history.iloc[-3:].mean(),
            "mean": history.mean(),
        }
        for order in (3, 12):
            features = np.column_stack([np.ones(start), *(history.shift(k) for k in range(1, order+1))])
            weights = np.linalg.lstsq(features[order:], history.iloc[order:], rcond=None)[0]
            expected[f"ar{order}"] = np.r_[1, history.iloc[-order:][::-1]] @ weights
        for base, value in expected.items():
            with self.subTest(base=base):
                pred, _ = _block(series, ["classical"], base, "", start, start + 1)
                self.assertAlmostEqual(pred.iloc[0, 0], value, places=6)

        incomplete = series.copy()
        incomplete.iloc[start - 12] = np.nan
        pred, _ = _block(incomplete, ["classical"], "naive", "", start, start + 1)
        self.assertEqual(pred.iloc[0, 0], history.iloc[-1])
        pred, _ = _block(incomplete, ["hybrid"], "ar12", "ridge", start, start + 1)
        self.assertTrue(pred.empty)

    def test_ma1_recovers_signed_shocks_and_updates_only_after_observation(self):
        shocks = np.random.default_rng(422).normal(size=1601)
        for theta in (-.65, .65):
            with self.subTest(theta=theta):
                y = 5 + shocks[1:] + theta * shocks[:-1]
                cutoff = 1400
                forecast, fit = _ma1_forecasts(y, cutoff, cutoff + 3, False)
                self.assertAlmostEqual(fit["theta"], theta, delta=.08)
                self.assertAlmostEqual(fit["mean"], 5, delta=.15)
                self.assertAlmostEqual(forecast[cutoff], fit["mean"] + fit["theta"] * fit["last_innovation"])
                for t in (cutoff + 1, cutoff + 2):
                    self.assertAlmostEqual(forecast[t], fit["mean"] + fit["theta"] * (y[t-1] - forecast[t-1]))
                surprise = y.copy()
                surprise[cutoff] += 10
                changed, new_fit = _ma1_forecasts(surprise, cutoff, cutoff+2, False)
                self.assertEqual(fit, new_fit)
                self.assertEqual(forecast[cutoff], changed[cutoff])
                self.assertAlmostEqual(changed[cutoff+1] - forecast[cutoff+1], fit["theta"] * 10)

    def test_ma1_historical_forecasts_match_each_prefix_and_handle_a_gap(self):
        y = self.series.to_numpy().copy()
        y[80] = np.nan
        start = 130
        forecasts, _ = _ma1_forecasts(y, start, start + 1, True)
        for t in (24, 60, 81, 99, start):
            prefix, fit = _ma1_forecasts(y[:t+1], t, t+1, False)
            self.assertAlmostEqual(forecasts[t], prefix[t])
            if t == 81:
                self.assertEqual(fit["last_innovation"], 0)
                self.assertEqual(prefix[t], fit["mean"])
        forecast, fit = _ma1_forecasts(np.full(40, 9.), 36, 40, False)
        np.testing.assert_array_equal(forecast[36:], 9.)
        self.assertEqual(fit["theta"], 0.)

    def test_missing_calendar_and_short_input(self):
        s=self.series.copy();s.iloc[-10]=np.nan
        r=family_forecast(s,**self.options)
        dates=[v["month"] for v in r.audit["holdout_predictions"]]
        self.assertNotIn(str(s.index[-10]),dates);self.assertNotIn(str(s.index[-9]),dates)
        with self.assertRaises(ValueError):family_forecast(s.iloc[:45],**self.options)


class HybridInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=StudioServer(("127.0.0.1",0),Handler)
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
        cls.url=f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close()

    def post(self,path,body):
        req=Request(self.url+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        with urlopen(req) as resp:
            return [json.loads(l) for l in resp][-1] if path=="/analyze" else json.load(resp)

    def test_templates_export_and_replay(self):
        for key in ("seasonForest","mlLasso","classicAR","classicMA","familyCompare"):
            cfg=CONFIGS[key];fields,code=template(cfg)
            body=dict(prompt_fields=fields,source_mode="retail_demo")
            reference=self.post("/analyze",{**body,"mode":"reference"})["report"]
            with patch("studio_core.studio.StudioAgents",return_value=Replies(code,revise=False)):
                live=self.post("/analyze",{**body,"mode":"live"})["report"]
            self.assertEqual(live["rows"],reference["rows"])
            for ext in ("csv","pdf"):
                with urlopen(f'{self.url}/reports/{live["id"]}.{ext}') as f:
                    content=f.read()
                self.assertTrue(b"Validation RMSE" in content if ext=="csv" else content.startswith(b"%PDF"))
            with contextlib.redirect_stdout(io.StringIO()),patch.object(sys,"argv",["analysis.py"]):
                env={};exec(live["audit"]["python_script"],env)
            self.assertEqual(env["result"].metrics.reset_index().values.tolist(),live["rows"])

    def test_ma1_pairs_with_each_learner_and_compares_forecast_families(self):
        from hybrid_forecasting import LEARNERS
        for mode, learner in [*(("hybrid", learner) for learner in LEARNERS), ("compare", "ridge")]:
            with self.subTest(mode=mode, learner=learner):
                cfg = {**CONFIGS["arForest"], "mode":mode, "base":"ma1", "learner":learner}
                fields = self.post("/family-template", cfg)["fields"]
                report = self.post("/analyze", dict(family_config=cfg, prompt_fields=fields,
                                   mode="reference", source_mode="retail_demo"))["report"]
                self.assertEqual(report["audit"]["base"], "ma1")
                self.assertEqual(len(report["audit"]["holdout_predictions"]), 24)
                parts = report["audit"]["next_components"]
                ma = report["audit"]["ma1"]
                self.assertAlmostEqual(parts["Baseline"], ma["mean"] + ma["theta"] * ma["last_innovation"])
                hybrid = next(row for row in report["rows"] if row[0] == "Hybrid")
                self.assertAlmostEqual(sum(parts.values()), hybrid[-1])

    def test_custom_activity_forecast_and_replay(self):
        cfg = {**CONFIGS["seasonForest"], "base":"ar3", "learner":"boosting",
               "target":"activity", "units":"simulated index units", "source":"activity"}
        fields = self.post("/family-template", cfg)["fields"]
        event = self.post("/analyze", dict(family_config=cfg, prompt_fields=fields, mode="reference",
                                           source_mode="ar_demo", data_kind="monthly"))
        report = event["report"]
        self.assertEqual(report["audit"]["target"], "activity")
        self.assertEqual(report["source"], "Bundled simulated AR activity")
        with contextlib.redirect_stdout(io.StringIO()), patch.object(sys, "argv", ["analysis.py"]):
            env = {}
            exec(report["audit"]["python_script"], env)
        self.assertEqual(env["result"].metrics.reset_index().values.tolist(), report["rows"])

    def test_custom_csv_and_fred(self):
        cfg={**CONFIGS["seasonForest"],"base":"ar1","target":"sales","units":"USD","source":"csv"}
        data=load_retail_demo().rename(columns={"MRTSSM44000USN":"sales"})
        fields=self.post("/family-template",cfg)["fields"]
        args=dict(family_config=cfg,prompt_fields=fields,mode="reference",source_mode="local")
        self.assertEqual(self.post("/analyze",args)["type"],"stopped")
        r=self.post("/analyze",{**args,"csv":data.to_csv(index=False),"filename":"monthly.csv"})["report"]
        self.assertEqual(r["audit"]["base"],"ar1")
        self.assertIn("Pass the downloaded monthly input CSV",r["audit"]["python_script"])
        preset=CONFIGS["trendLasso"]
        e=self.post("/analyze",dict(mode="reference",prompt_fields=FIELDS["trendLasso"],source_mode="fred"))
        self.assertEqual(e["settings"]["series_ids"],["MRTSSM44000USN"])
        task=fred_task(preset)
        with patch("studio_core.sources._fred_response",return_value=json.dumps({"observations":[
                {"date":dt,"value":v} for dt,v in data.itertuples(index=False)]}).encode()):
            imported=self.post("/sources/fred",dict(series_ids="MYINDEX",api_key="a"*32,starter_task=task))
        result=self.post("/analyze",dict(mode="reference",prompt_fields=imported["starter"],source_mode="fred",
            fred_task=task,family_config=imported["family_config"],csv=imported["csv"],filename=imported["filename"]))["report"]
        self.assertEqual(result["audit"]["target"],"MYINDEX");self.assertEqual(result["audit"]["learner"],"lasso")
