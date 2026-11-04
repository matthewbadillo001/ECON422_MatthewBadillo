"""Monthly FRED observations. Credentials are used only during this import."""
import json
import re
import socket
import ssl
import subprocess
import sys
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, build_opener

import pandas as pd

from studio_core.data_io import prepare_airfare_sample
from studio_core.example_prompt import AIRFARE_FRED_COLUMNS, RETAIL_ID


def resembles_fred_key(text):
    return bool(re.search(r"(?<!\w)[a-zA-Z0-9]{32}(?!\w)", text))


def requested_source(prompt, mode="auto"):
    if mode in ("ar_demo", "retail_demo"):
        return "local"
    if mode in {"local", "fred", "eia", "alpha", "custom"}:
        return mode
    if mode != "auto":
        raise ValueError("Unknown data source setting.")
    patterns = {"fred": r"\bFRED\b|Federal Reserve Economic Data|(?:api|fred)\.stlouisfed\.org",
                "eia": r"\bEIA\b|Energy Information Administration|api\.eia\.gov",
                "alpha": r"Alpha\s*Vantage|alphavantage\.co"}
    text = prompt
    for pattern in patterns.values():
        text = re.sub(r"(?:do not use|don't use|without|instead of)\s+(?:the\s+)?(?:" + pattern + r")(?: API)?", "", text, flags=re.I)
    matches = [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]
    if len(matches) == 1:
        return matches[0]
    if matches or re.search(r"\b(?:API|BEA|BLS|World Bank|Census Bureau|Quandl|OECD|IMF)\b|https?://|\b[\w.-]+\.(?:gov|org|com|net|io)\b", text, re.I):
        return "custom"
    return "local"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
MAX_BYTES = 2_000_000
TLS_CERT = "the HTTPS certificate could not be verified"
DNS = "DNS could not resolve the host"
PROXY = "the HTTPS proxy could not connect"
TIMEOUT = "the request timed out"
CONNECT = "the connection was refused or interrupted"
TLS = "the TLS handshake failed"


def _transport_reason(exc):
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLCertVerificationError):
        return TLS_CERT
    if isinstance(reason, ssl.SSLError):
        return TLS
    if isinstance(reason, socket.gaierror):
        return DNS
    if isinstance(reason, TimeoutError):
        return TIMEOUT
    if "tunnel connection failed" in str(reason).lower():
        return PROXY
    if isinstance(reason, (ConnectionError, HTTPException)):
        return CONNECT
    return "the request ended before a complete HTTP response was received"


def _fred_http_error(status, sid):
    if status == 407:
        return "The HTTPS proxy requires authentication (HTTP 407). Check the Mac's proxy settings."
    if status == 429:
        return "FRED rate limit reached (HTTP 429). Wait a moment and try again."
    if 500 <= status < 600:
        return f"The FRED request returned HTTP {status}. Retry later."
    if 300 <= status < 400:
        return f"The FRED endpoint returned a redirect (HTTP {status}). The studio does not forward the key."
    if status in (401, 403):
        return f"The FRED request was denied (HTTP {status}). Check the key and your network's access to the API."
    return (f"FRED rejected the request for {sid} (HTTP {status}). Check the key and series ID. "
            "Use a monthly or higher-frequency series.")


def _macos_fred_request(url, sid, python_reason):
    # The URL contains a credential. Pass it via stdin, never argv, a file, or diagnostics.
    command = ["/usr/bin/curl", "--disable", "--silent", "--proto", "=https", "--no-location",
               "--max-filesize", str(MAX_BYTES), "--connect-timeout", "8", "--max-time", "22",
               "--write-out", "\n%{http_code}", "--config", "-"]
    try:
        result = subprocess.run(command, input=f'url = "{url}"\n'.encode(), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=25, check=False)
    except subprocess.TimeoutExpired:
        reason = TIMEOUT
    except OSError:
        reason = "the macOS HTTPS client could not start"
    else:
        if result.returncode == 0:
            body, _, code = result.stdout.rpartition(b"\n")
            if code.isdigit() and int(code) == 200:
                return body
            if code.isdigit() and int(code) >= 300:
                raise ValueError(_fred_http_error(int(code), sid))
            reason = "no valid HTTP status was received"
        elif result.returncode == 63:
            raise ValueError(f"The response for {sid} exceeded the studio's size limit.")
        else:
            reason = {5: PROXY, 6: DNS, 7: CONNECT, 28: TIMEOUT, 35: TLS,
                      60: TLS_CERT, 77: TLS_CERT, 97: PROXY}.get(
                          result.returncode, f"curl exited with code {result.returncode}")
    advice = {TLS_CERT: "Check the trusted certificates in macOS Keychain or your network's HTTPS proxy.",
              DNS: "Check DNS or VPN access to api.stlouisfed.org.",
              PROXY: "Check the Mac's HTTPS proxy settings.",
              TIMEOUT: "Retry once or check whether the API host is reachable."}.get(
                  reason, "Check whether your VPN or proxy permits access to the API host.")
    raise ValueError(f"FRED connection to api.stlouisfed.org failed. Python: {python_reason}. "
                     f"macOS HTTPS retry: {reason}. {advice}")


def _fred_response(opener, url, sid):
    try:
        with opener.open(url, timeout=20) as response:
            return response.read(MAX_BYTES + 1)
    except HTTPError as exc:
        exc.close()
        raise ValueError(_fred_http_error(exc.code, sid)) from None
    except (URLError, OSError, HTTPException) as exc:
        reason = _transport_reason(exc)
    if sys.platform == "darwin":
        return _macos_fred_request(url, sid, reason)
    raise ValueError(f"FRED connection to api.stlouisfed.org failed: {reason}.")


def import_fred(series_ids, api_key, *, airfare_example=False, retail_example=False):
    if not isinstance(series_ids, str) or len(series_ids) > 256:
        raise ValueError("Enter one to six comma-separated FRED series IDs.")
    ids = []
    for piece in series_ids.split(","):
        sid = piece.strip()
        if len(sid) >= 2 and sid[0] == sid[-1] and sid[0] in ("`", "'", '"'):
            sid = sid[1:-1].strip()
        ids.append(sid.upper())
    if (not 1 <= len(ids) <= 6 or len(set(ids)) != len(ids)
            or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,39}", sid) or sid == "DATE"
                   or resembles_fred_key(sid) for sid in ids)):
        raise ValueError("Use one to six unique FRED series IDs, separated by commas.")
    if not isinstance(api_key, str) or not re.fullmatch(r"[a-z0-9]{32}", api_key.strip()):
        raise ValueError("Enter your 32-character FRED API key in the masked field.")
    if airfare_example and set(ids) != set(AIRFARE_FRED_COLUMNS):
        raise ValueError("Use CUSR0000SETG01 and MJFUELUSGULF for the January 2025 airfare example.")
    if retail_example and ids != [RETAIL_ID]:
        raise ValueError("The retail sales preset requires " + RETAIL_ID + ".")
    opener = build_opener(NoRedirect())
    series = []
    for sid in ids:
        query = dict(series_id=sid, api_key=api_key.strip(), file_type="json",
                     frequency="m", aggregation_method="avg", units="lin")
        if airfare_example:
            query.update(observation_start="2004-12-01", observation_end="2024-12-31")
        if retail_example:
            query.update(observation_start="2005-01-01", observation_end="2024-12-31")
        params = urlencode(query)
        raw = _fred_response(opener, FRED_ENDPOINT+"?"+params, sid)
        if len(raw) > MAX_BYTES:
            raise ValueError(f"The response for {sid} exceeded the studio's size limit.")
        try:
            observations = json.loads(raw)["observations"]
            values = pd.DataFrame(observations, columns=["date", "value"])
            dates = pd.to_datetime(values["date"], errors="raise").dt.to_period("M")
            numbers = pd.to_numeric(values["value"].replace(".", None), errors="raise")
            if dates.hasnans or dates.duplicated().any() or not numbers.notna().any():
                raise ValueError("Missing or duplicate dates or no numeric observations")
            series.append(pd.Series(numbers.to_numpy(), index=pd.PeriodIndex(dates, name="date"), name=sid))
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"FRED returned no usable monthly observations for {sid}.") from None
    frame = pd.concat(series, axis=1).sort_index().dropna(how="all")
    frame.index = frame.index.to_timestamp().strftime("%Y-%m-%d")
    if airfare_example:
        rates = prepare_airfare_sample(frame.rename_axis("date").reset_index())
        return {"csv": rates.to_csv(index=False),
                "filename": "FRED CUSR0000SETG01 & MJFUELUSGULF | monthly % changes, Jan 2005–Dec 2024 (current vintage)",
                "series": list(AIRFARE_FRED_COLUMNS.values()), "months": len(rates)}
    if retail_example:
        frame = frame.loc["2005-01-01":"2024-12-31"]
        if frame.empty or frame.index[-1] != "2024-12-01":
            raise ValueError("The retail exercise needs data through December 2024.")
        return {"csv":frame.to_csv(index_label="date"), "filename":"FRED retail sales, Jan 2005–Dec 2024",
                "series":ids, "months":len(frame)}
    return {"csv": frame.to_csv(index_label="date"), "filename": "FRED monthly data ("+", ".join(ids)+")",
            "series": ids, "months": len(frame)}
