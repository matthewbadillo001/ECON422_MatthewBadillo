import contextlib
import io
import json
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from studio_core.app import Handler, StudioServer
from studio_core.data_io import load_retail_demo
from studio_core.example_prompt import (RETAIL_FIELDS, RETAIL_FRED_FIELDS, RETAIL_ID, RETAIL_CODE,
    EXPLORE_FIELDS, EXPLORE_FRED_FIELDS, csv_explore_fields, exploration_code)
from exploration import explore_series, exploration_development, monthly_series
from test_studio import Replies


def profile(data, name=RETAIL_ID):
    return explore_series(monthly_series(data, target=name))


class RetailExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = StudioServer(("127.0.0.1",0), Handler)
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.data = load_retail_demo()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path, body):
        req = Request(self.url+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        with urlopen(req) as response:
            return json.load(response) if path=="/sources/fred" else [json.loads(line) for line in response][-1]

    def test_holdout_is_excluded_from_all_model_selection(self):
        result = exploration_development(monthly_series(self.data,target=RETAIL_ID))
        changed = self.data.copy()
        changed.loc[changed.index[-24:], RETAIL_ID] = 1
        other = exploration_development(monthly_series(changed,target=RETAIL_ID))
        pd.testing.assert_frame_equal(result.metrics, other.metrics)
        self.assertEqual(result.audit, other.audit)
        self.assertEqual(result.audit["recommended_model"], "Trend + month effects + AR(1)")
        self.assertAlmostEqual(min(result.audit["models"].values()), 21569.25, places=2)
        self.assertEqual(result.audit["common_months"],24)
        self.assertIn("2023-01",result.audit["reserved"])
        full = profile(self.data)
        altered = profile(changed)
        self.assertEqual(full.audit["recommended_model"],altered.audit["recommended_model"])
        self.assertNotEqual(full.audit["holdout_rmse"],altered.audit["holdout_rmse"])
        self.assertEqual(full.audit["forecast_month"],"2025-01")
        self.assertEqual(len(full.audit["holdout_scored_months"]),24)

    def test_other_series_can_select_a_different_model(self):
        data = self.data.rename(columns={RETAIL_ID:"season"})
        data["season"] = np.resize([17.,9.,-4.,2.,13.,7.,3.,-3.,6.,11.,1.,5.],len(data))
        self.assertEqual(profile(data,"season").audit["recommended_model"],"Same month last year")
        # A calendar gap must remain a gap, not shorten the lag.
        gap = self.data.drop([30,40,202])
        report=profile(gap)
        self.assertIn("190 of 192", report.metrics.loc["Coverage","Finding"])
        self.assertLess(report.audit["common_months"],24)
        self.assertIsNone(profile(self.data.iloc[:40]).audit["recommended_model"])

    def test_worked_reference_live_reports_and_replay(self):
        for fields,code in ((RETAIL_FIELDS,RETAIL_CODE),(EXPLORE_FIELDS,exploration_code(RETAIL_ID))):
            body = {"prompt_fields":fields,"source_mode":"retail_demo"}
            reference=self.request("/analyze",{**body,"mode":"reference"})["report"]
            with patch("studio_core.studio.StudioAgents",return_value=Replies(code,revise=False)):
                live=self.request("/analyze",{**body,"mode":"live"})["report"]
            self.assertEqual(live["rows"],reference["rows"])
            if fields is RETAIL_FIELDS:
                self.assertEqual(live["audit"]["next_month"]["month"],"2025-01")
                self.assertAlmostEqual(live["audit"]["next_month"]["forecast"],587398.56,places=2)
            else:
                self.assertIn("Recommend Trend", live["summary"])
                for fmt in ("csv","pdf"):
                    with urlopen(f'{self.url}/reports/{live["id"]}.{fmt}') as response:
                        contents=response.read()
                    self.assertTrue(contents.startswith(b"%PDF") if fmt=="pdf" else b"Rolling validation" in contents)
            for report in (live,reference):
                with patch.object(sys,"argv",["analysis.py"]), contextlib.redirect_stdout(io.StringIO()):
                    env={}
                    exec(report["audit"]["python_script"],env)
                self.assertEqual(env["result"].metrics.reset_index().values.tolist(),report["rows"])

    def test_generic_csv_and_fred_starters(self):
        data=self.data.rename(columns={RETAIL_ID:"sales"})
        fields=csv_explore_fields("sales","millions of dollars")
        args=dict(mode="reference", source_mode="local",prompt_fields=fields, exploration_target="sales",
                  exploration_units="millions of dollars", csv=data.to_csv(index=False),filename="my_series.csv")
        report=self.request("/analyze",args)["report"]
        self.assertIn("sales",report["summary"])
        self.assertIn("Pass the downloaded monthly input CSV",report["audit"]["python_script"])
        with patch("studio_core.studio.StudioAgents",return_value=Replies(exploration_code("sales"),revise=False)):
            live=self.request("/analyze",{**args,"mode":"live"})["report"]
        self.assertEqual(report["rows"],live["rows"])
        for preset in (RETAIL_FRED_FIELDS,EXPLORE_FRED_FIELDS):
            event=self.request("/analyze",dict(mode="reference",source_mode="fred",prompt_fields=preset))
            self.assertEqual(event["settings"]["series_ids"],[RETAIL_ID])
            self.assertEqual(event["settings"]["recipe"],"retail_2005_2024")
        def mock_fred(opener,url,sid):
            self.assertIn("observation_end=2024-12-31",url)
            return json.dumps({"observations":[{"date":d,"value":str(v)} for d,v in self.data.itertuples(index=False)]}).encode()
        with patch("studio_core.sources._fred_response",side_effect=mock_fred):
            imported=self.request("/sources/fred",dict(series_ids=RETAIL_ID,api_key="a"*32,recipe="retail_2005_2024"))
        for fields in (EXPLORE_FRED_FIELDS, RETAIL_FRED_FIELDS):
            actual=self.request("/analyze",dict(mode="reference",source_mode="fred",prompt_fields=fields,
                csv=imported["csv"],filename=imported["filename"]))
            self.assertEqual(actual["type"],"result")
        with patch("studio_core.sources._fred_response",return_value=json.dumps({"observations":[
                {"date":d,"value":str(v)} for d,v in self.data.itertuples(index=False)]}).encode()):
            other=self.request("/sources/fred",dict(series_ids="UNRATE",api_key="a"*32,starter_task="exploration"))
        r=self.request("/analyze",dict(mode="reference",source_mode="fred",fred_task="exploration",
                          prompt_fields=other["starter"],csv=other["csv"],filename=other["filename"]))["report"]
        self.assertEqual(r["audit"]["target"],"UNRATE")
