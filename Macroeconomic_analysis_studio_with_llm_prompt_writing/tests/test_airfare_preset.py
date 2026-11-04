import json
import threading
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from studio_core.app import Handler, StudioServer
from studio_core.data_io import prepare_airfare_sample
from studio_core.example_prompt import AIRFARE_FRED_COLUMNS, AIRFARE_FRED_FIELDS, FORECAST_CODE, FORECAST_FIELDS
from test_studio import Replies


class AirfareExampleTests(unittest.TestCase):
    def test_gap_is_not_bridged_and_levels_after_cutoff_are_excluded(self):
        ids = list(AIRFARE_FRED_COLUMNS)
        levels=pd.DataFrame({"date":["2024-06-01","2024-08-01","2024-09-01","2024-10-01","2024-11-01","2024-12-01","2025-01-01"],
                             ids[0]:[10,15,20,22,23,24,900], ids[1]:[4,5,6,7,8,10,90]})
        rates=prepare_airfare_sample(levels)
        self.assertEqual(rates.iloc[-1]["date"],"2024-12-01")
        self.assertTrue(pd.isna(rates.loc[rates.date=="2024-08-01","fare_inflation"].iloc[0]))
        self.assertAlmostEqual(rates.iloc[-1]["fuel_inflation"],25.0)
        self.assertAlmostEqual(rates.iloc[-1]["fare_inflation"],100*(24/23-1))
        pd.testing.assert_frame_equal(prepare_airfare_sample(rates),rates)
        with self.assertRaisesRegex(ValueError,"October–December"):
            prepare_airfare_sample(levels[levels.date!="2024-11-01"])

    def test_custom_fred_ids_use_raw_observations_and_matching_reference(self):
        months=pd.date_range("2016-01-01",periods=120,freq="MS")
        def response(opener,url,sid):
            params=parse_qs(urlsplit(url).query)
            self.assertEqual(params["units"],["lin"])
            self.assertNotIn("observation_end",params)
            return json.dumps({"observations":[{"date":str(d.date()),"value":str(210+i)}
                for i,d in enumerate(months)]}).encode()
        with StudioServer(("127.0.0.1",0),Handler) as server:
            threading.Thread(target=server.serve_forever,daemon=True).start()
            def post(path,body):
                req=Request(f"http://127.0.0.1:{server.server_port}"+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
                with urlopen(req) as r:
                    return json.load(r) if path=="/sources/fred" else [json.loads(line) for line in r][-1]
            try:
                for task in ("ts_dml","rcf_dml_screen","forecast"):
                    with patch("studio_core.sources._fred_response",side_effect=response):
                        imported=post("/sources/fred",{"series_ids":"cpiaucsl","api_key":"test"*8,"starter_task":task})
                    self.assertEqual(imported["series"],["CPIAUCSL"])
                    self.assertEqual(imported["csv"].splitlines()[1],"2016-01-01,210")
                    body=dict(prompt_fields=imported["starter"],csv=imported["csv"],filename=imported["filename"],
                              mode="reference",source_mode="fred",fred_task=task)
                    result=post("/analyze",body)["report"]
                    self.assertEqual(result["task"],task)
                    self.assertNotIn("prepare_airfare_sample",result["audit"]["python_script"])
                    self.assertIn("CPIAUCSL",result["audit"]["reference_code"])
                    if task!="forecast":
                        self.assertEqual(result["audit"]["term"],"CPIAUCSL__lag1")
                        self.assertEqual(result["rows"][0][1],"Concern")
                    changed={**imported["starter"],"role":"My own question."}
                    self.assertEqual(post("/analyze",{**body,"prompt_fields":changed})["type"],"stopped")
            finally:
                server.shutdown()

    def test_forms_import_percentage_changes_and_both_review_paths(self):
        ids=list(AIRFARE_FRED_COLUMNS)
        months=pd.date_range("2004-12-01", "2025-02-01",freq="MS")
        rng=np.random.default_rng(422)
        series={sid:pd.Series(100*np.cumprod(1+rng.normal(.002,.01,len(months))),index=months) for sid in ids}
        calls=[]
        def response(opener,url,sid):
            params=parse_qs(urlsplit(url).query)
            self.assertEqual(params["observation_start"],["2004-12-01"])
            self.assertEqual(params["observation_end"],["2024-12-31"])
            calls.append(sid)
            return json.dumps({"observations":[{"date":str(d.date()),"value":str(v)} for d,v in series[sid].items()]}).encode()
        with StudioServer(("127.0.0.1",0),Handler) as server:
            threading.Thread(target=server.serve_forever,daemon=True).start()
            def post(path,body):
                req=Request(f"http://127.0.0.1:{server.server_port}"+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
                with urlopen(req) as r:
                    return json.load(r) if path=="/sources/fred" else [json.loads(line) for line in r]
            try:
                for mode in ("live","reference"):
                    args={"prompt_fields":AIRFARE_FRED_FIELDS,"mode":mode,"source_mode":"fred"}
                    req=post("/analyze",args)[-1]
                    self.assertEqual(req["settings"]["series_ids"],ids)
                    self.assertEqual(req["type"],"needs_source")
                with patch("studio_core.sources._fred_response",side_effect=response):
                    got=post("/sources/fred",{"series_ids":", ".join(ids),"recipe":req["settings"]["recipe"],"api_key":"test"*8})
                self.assertEqual(calls,ids)
                self.assertEqual(got["csv"].splitlines()[0],"date,fare_inflation,fuel_inflation")
                self.assertEqual(got["months"],240)
                args.update(csv=got["csv"],filename=got["filename"])
                reference=post("/analyze",args)[-1]["report"]
                # Replay the analyst's pasted code, including the missing newline.
                agent=Replies(FORECAST_CODE.replace("\n"," "),revise=False)
                with patch("studio_core.studio.StudioAgents",return_value=agent):
                    live=post("/analyze",{**args,"mode":"live"})[-1]["report"]
                    revised=post("/analyze",{"parent_id":live["id"],"feedback":"Keep the model and explain the forecast."})[-1]["report"]
                for result in (reference,live,revised):
                    audit=result["audit"]
                    self.assertEqual(audit["next_month"]["month"],"2025-01")
                    self.assertAlmostEqual(audit["next_month"]["forecast"],reference["audit"]["next_month"]["forecast"])
                    self.assertEqual(audit["holdout_months_scored"],24)
                    self.assertIn("Pass the downloaded monthly input CSV",audit["python_script"])
                with patch("studio_core.app.analyze") as llm:
                    end=post("/analyze",{"prompt_fields":FORECAST_FIELDS,"mode":"live",
                                         "csv":"date,UNRATE\n2024-12-01,4.0\n"})[-1]
                    self.assertEqual(end["type"],"stopped");self.assertIn("CUSR0000SETG01",end["message"])
                    llm.assert_not_called()
            finally:
                server.shutdown()
