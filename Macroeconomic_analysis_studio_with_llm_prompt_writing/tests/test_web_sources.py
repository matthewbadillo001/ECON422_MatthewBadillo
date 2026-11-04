import json
import socket
import threading
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from studio_core.api import load_demo
from studio_core.app import Handler, StudioServer
from studio_core.sources import requested_source
from studio_core.web_sources import PROVIDERS, import_web_source
from test_studio import Replies, FORECAST

KEY = "fixture-key-7_zx"
RECORDS = [{"month":"2025-02","mean":4.1},{"month":"2025-01","mean":4.2}]


def options(provider="custom", **overrides):
    args = {**PROVIDERS[provider], "provider":provider, "api_key":KEY}
    if provider == "custom":
        args.update(endpoint="https://weather.example.com/months?q=A", path="data.observations",
                    date="month", values="temperature=mean")
    return {**args, **overrides}


class WebSourceTests(unittest.TestCase):
    def test_provider_names_and_custom_sites(self):
        for prompt, expected in [("Fetch electricity data from EIA", "eia"), ("Use Alpha Vantage", "alpha"),
                                 ("Use the BEA API", "custom"), ("Data from weather.example.com", "custom"),
                                 ("Use EIA instead of FRED", "eia"), ("Use the included data without FRED API", "local"),
                                 ("Forecast the simulated series", "local")]:
            self.assertEqual(requested_source(prompt),expected)
        self.assertEqual(requested_source("Use FRED", "custom"),"custom")

    def test_provider_presets_send_only_their_own_key_and_map_monthly_rows(self):
        fixtures = {"eia":{"response":{"data":[{"period":"2025-01","price":"16.5"}],"total":"1"}},
                    "alpha":{"data":[{"date":"2025-01-01","value":"4.1"}]}}
        for provider, fixture in fixtures.items():
            with self.subTest(provider=provider), patch("studio_core.web_sources._fetch", return_value=json.dumps(fixture).encode()) as transport:
                result=import_web_source(options(provider))
                url,host,header,custom=transport.call_args.args
                self.assertEqual(host,PROVIDERS[provider]["host"])
                self.assertEqual(parse_qs(urlsplit(url).query)[PROVIDERS[provider]["param"]],[KEY])
                self.assertIsNone(header); self.assertFalse(custom)
                self.assertEqual(result["months"],1)
                self.assertNotIn(KEY,json.dumps(result))

    def test_custom_auth_and_numeric_columns(self):
        for auth in ["query","header","bearer","none"]:
            with self.subTest(auth=auth), patch("studio_core.web_sources._fetch",return_value=json.dumps({"data":{"observations":RECORDS}}).encode()) as transport:
                args=options(auth=auth,param="subscription",api_key="" if auth=="none" else KEY)
                r=import_web_source(args)
                url,host,header,custom=transport.call_args.args
                self.assertTrue(custom); self.assertEqual(host,"weather.example.com")
                self.assertEqual(KEY in url,auth=="query")
                self.assertEqual(header,{"header":"subscription: "+KEY,"bearer":"Authorization: Bearer "+KEY}.get(auth))
                self.assertEqual(r["csv"],"date,temperature\n2025-01-01,4.2\n2025-02-01,4.1\n")
                self.assertNotIn(KEY,json.dumps(r))

    def test_unsafe_destinations_and_embedded_credentials_are_rejected(self):
        for endpoint in ["http://api.example.com/data","https://name:pass@api.example.com/data",
                         "https://api.example.com:81/data","https://api.example.com/data?apikey=secret",
                         "https://api.example.com/path#fragment","https://127.0.0.1/data"]:
            with self.subTest(endpoint=endpoint),patch("studio_core.web_sources._fetch") as transport:
                with self.assertRaises(ValueError): import_web_source(options(endpoint=endpoint))
                transport.assert_not_called()
        with self.assertRaisesRegex(ValueError,"official API"):
            import_web_source(options("alpha",endpoint="https://wrong.example.com/query"))
        with patch("studio_core.web_sources.socket.getaddrinfo",return_value=[
                (socket.AF_INET,socket.SOCK_STREAM,0,"",("127.0.0.1",443))]), patch("studio_core.web_sources.subprocess.Popen") as proc:
            with self.assertRaisesRegex(ValueError,"public internet"):
                import_web_source(options())
            proc.assert_not_called()

    def test_response_shape_errors_and_provider_limits(self):
        cases = [(options("alpha"),{"Information":"notice "+KEY},"notice"),
                 (options("eia"),{"response":{"total":"82","data":[]}},"pages"),
                 (options(),{"data":{"observations":RECORDS*2}},"per month"),
                 (options(),{"rows":[]},"path")]
        for args,data,message in cases:
            with self.subTest(error=message), patch("studio_core.web_sources._fetch",return_value=json.dumps(data).encode()):
                with self.assertRaisesRegex(ValueError,message) as err: import_web_source(args)
                self.assertNotIn(KEY,str(err.exception))

    def test_api_form_import_then_analyze_and_revise(self):
        sample=load_demo().head(160).rename(columns={"fare_inflation":"price","fuel_inflation":"fuel"})
        provider_csv=sample.to_csv(index=False).encode()
        program=FORECAST.replace("fare_inflation","price").replace("fuel_inflation","fuel")
        agents=Replies(program,revise=False)
        with StudioServer(("127.0.0.1",0),Handler) as server:
            threading.Thread(target=server.serve_forever,daemon=True).start()
            def post(path,body):
                url=f"http://127.0.0.1:{server.server_port}"+path
                with urlopen(Request(url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})) as response:
                    return [json.loads(x) for x in response][-1] if path=="/analyze" else json.load(response)
            try:
                body={"prompt":"Forecast monthly price from the weather.example.com API."}
                self.assertEqual(post("/analyze",body)["provider"],"custom")
                with patch("studio_core.web_sources._fetch",return_value=provider_csv) as connection:
                    r=post("/sources/web",options(format="csv",date="date",values="price,fuel"))
                    with patch("studio_core.studio.StudioAgents",return_value=agents), patch.object(agents,"review_prompt",wraps=agents.review_prompt) as review:
                        base=post("/analyze",{**body,"csv":r["csv"],"filename":r["filename"]})["report"]
                        revised=post("/analyze",{"parent_id":base["id"],"feedback":"Keep the same forecast target."})["report"]
                    connection.assert_called_once()
                    self.assertNotIn(KEY,str(review.call_args_list))
                    self.assertEqual(base["source"],"weather.example.com monthly data")
                    self.assertEqual(revised["source"],base["source"])
                    self.assertNotIn(KEY,json.dumps([base,revised]))
            finally:
                server.shutdown()
