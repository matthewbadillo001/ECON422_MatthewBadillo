"""Import one page of monthly observations from a user-selected HTTPS data API."""
import csv
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

from studio_core.sources import MAX_BYTES


PROVIDERS = {
    "eia": {"name":"U.S. Energy Information Administration (EIA)", "host":"api.eia.gov",
            "endpoint":"https://api.eia.gov/v2/electricity/retail-sales/data/?frequency=monthly&data[0]=price&facets[stateid][]=CO&facets[sectorid][]=RES&length=5000",
            "auth":"query", "param":"api_key", "path":"response.data", "date":"period",
            "values":"electricity_price=price", "format":"json",
            "note":"The starting example is the Colorado residential electricity price in cents per kilowatthour. Change the query and fields for another monthly EIA series."},
    "alpha": {"name":"Alpha Vantage", "host":"www.alphavantage.co",
              "endpoint":"https://www.alphavantage.co/query?function=UNEMPLOYMENT",
              "auth":"query", "param":"apikey", "path":"data", "date":"date",
              "values":"unemployment_rate=value", "format":"json",
              "note":"The starting example is the monthly U.S. unemployment rate. Change the query and fields for another monthly Alpha Vantage series."},
    "custom": {"name":"Custom website API", "host":"", "endpoint":"", "auth":"header", "param":"X-API-Key",
               "path":"", "date":"date", "values":"value", "format":"json",
               "note":"Enter the provider’s documented HTTPS GET URL without a key. Choose how it authenticates and map a single page of monthly records."},
}


def _destination(endpoint, provider, key):
    if not isinstance(endpoint, str) or not 1 <= len(endpoint) <= 3500 or re.search(r'[\s\x00-\x1f\x7f"\\]', endpoint):
        raise ValueError("Enter a public HTTPS data endpoint without spaces or credentials.")
    try:
        parsed = urlsplit(endpoint)
        host = parsed.hostname or ""
        allowed_origin = (parsed.scheme == "https" and parsed.port in (None, 443)
                          and parsed.username is None and not parsed.fragment
                          and re.fullmatch(r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]*", host))
    except ValueError:
        allowed_origin = False
    if not allowed_origin:
        raise ValueError("Use a public HTTPS hostname on port 443, without a login or fragment.")
    if provider["host"] and host != provider["host"]:
        raise ValueError("The endpoint must use the selected provider’s official API hostname.")
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if any(re.search(r"(?:api.?key|access.?token|authorization|secret|password|^token$)", k, re.I) for k, _ in query) or (key and key in endpoint):
        raise ValueError("Remove credentials from the URL and enter the key only in the masked field.")
    return parsed, host, query


def _public_route(host):
    try:
        addresses = {addr[4][0] for addr in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
        ips = [ipaddress.ip_address(addr) for addr in sorted(addresses)]
    except (OSError, ValueError):
        raise ValueError("DNS could not resolve this API hostname.") from None
    if not ips or any(not ip.is_global or ip.is_multicast or getattr(ip, "ipv4_mapped", None)
                      or (ip.version == 6 and (ip.sixtofour or ip.teredo or ip in ipaddress.ip_network("64:ff9b::/96"))) for ip in ips):
        raise ValueError("The custom API must resolve exclusively to public internet addresses.")
    # Use the verified address for the actual connection, retaining hostname-based TLS.
    ip = min(ips, key=lambda addr: addr.version)
    return f"{host}:443:[{ip}]" if ip.version == 6 else f"{host}:443:{ip}"


def _fetch(url, host, header, custom):
    curl = "/usr/bin/curl" if os.path.exists("/usr/bin/curl") else shutil.which("curl")
    if not curl:
        raise ValueError("The HTTPS importer requires curl on this computer.")
    command = [curl, "--disable", "--silent", "--proto", "=https", "--no-location", "--globoff",
               "--connect-timeout", "8", "--max-time", "25", "--max-filesize", str(MAX_BYTES),
               "--write-out", "\n%{http_code}", "--config", "-"]
    if custom:
        command.extend(["--noproxy", "*", "--resolve", _public_route(host)])
    # Keep credentials out of subprocess arguments and log messages.
    config = "url = " + json.dumps(url) + "\n"
    if header:
        config += "header = " + json.dumps(header) + "\n"
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
            proc.stdin.write(config.encode())
            proc.stdin.close()
            raw = proc.stdout.read(MAX_BYTES + 5)
            if len(raw) > MAX_BYTES + 4:
                proc.kill()
                raise ValueError("The response is too large. Narrow the request to a smaller monthly sample.")
            rc = proc.wait()
    except OSError:
        raise ValueError("The HTTPS client could not complete the import.") from None
    if rc:
        detail = {5:"proxy lookup failed", 6:"DNS lookup failed", 7:"connection failed",
                  28:"connection timed out", 60:"certificate verification failed", 63:"response was too large"}
        raise ValueError(f"Could not retrieve data from {host}: {detail.get(rc, 'HTTPS client error '+str(rc))}.")
    body, sep, status = raw.rpartition(b"\n")
    if not sep or status != b"200":
        code = status.decode() if status.isdigit() else "unknown"
        raise ValueError(f"{host} returned HTTP {code}. Check the endpoint, access permissions and rate limit.")
    return body


def _monthly_data(raw, options, provider):
    try:
        text = raw.decode("utf-8-sig")
        if options["format"] == "csv":
            rows = list(csv.DictReader(io.StringIO(text)))
        elif options["format"] == "json":
            rows = json.loads(text)
            if isinstance(rows, dict) and any(k in rows for k in ("Error Message", "Information", "Note", "error")):
                raise ValueError("The provider returned a notice instead of data. Check the key, query and plan limits.")
            if provider == "eia" and isinstance(rows, dict):
                data = rows.get("response", {})
                if isinstance(data, dict) and isinstance(data.get("data"), list):
                    total = str(data.get("total", len(data["data"])))
                    if not re.fullmatch(r"\d{1,12}", total):
                        raise ValueError("The EIA response contains an invalid record count.")
                    if int(total) > len(data["data"]):
                        raise ValueError("The EIA result spans multiple pages. Narrow the filters or date range.")
            for part in filter(None, options.get("path", "").split(".")):
                if isinstance(rows, list):
                    if not re.fullmatch(r"\d{1,8}", part):
                        raise ValueError("Use a numeric index when the JSON path crosses a list.")
                    rows = rows[int(part)]
                else:
                    rows = rows[part]
        else:
            raise ValueError("Select JSON or CSV.")
    except (UnicodeError, json.JSONDecodeError, csv.Error, KeyError, IndexError, TypeError):
        raise ValueError("Check the response format and the JSON path to the observations.") from None
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("The records path must point to a nonempty list of objects with date and value fields.")
    frame = pd.DataFrame(rows)
    date_field = options["date"].strip()
    if date_field not in frame:
        raise ValueError("The selected date field is missing from the records.")
    dates = frame[date_field].astype(str)
    if not dates.str.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?").all():
        raise ValueError("Use monthly dates in YYYY-MM or YYYY-MM-DD form.")
    days = dates.where(dates.str.len() == 10, dates + "-01")
    months = pd.to_datetime(days, format="%Y-%m-%d", errors="coerce").dt.to_period("M")
    if months.isna().any() or months.duplicated().any():
        raise ValueError("Select one observation per month. Filter the series or aggregate it at the provider.")
    result = pd.DataFrame({"date":months.dt.strftime("%Y-%m-01")})
    for entry in options["values"].split(","):
        alias, _, field = entry.strip().partition("=")
        alias, field = alias.strip(), (field or alias).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", alias) or alias in result or field not in frame:
            raise ValueError("Match each value field to a unique analysis name, for example unemployment_rate=value.")
        values = frame[field].replace(["", ".", "NA", "N/A", "null"], None)
        numbers = pd.to_numeric(values, errors="coerce")
        if (values.notna() & numbers.isna()).any() or not numbers.notna().any() or np.isinf(numbers).any():
            raise ValueError("Each selected value field must contain numeric monthly observations.")
        result[alias] = numbers
    return result.dropna(subset=list(result.columns[1:]), how="all").sort_values("date")


def import_web_source(options):
    provider = options.get("provider")
    if provider not in PROVIDERS:
        raise ValueError("Select an API provider.")
    required = ("endpoint", "auth", "param", "path", "date", "values", "format", "api_key")
    if any(not isinstance(options.get(k), str) or len(options[k]) > 4096 for k in required):
        raise ValueError("Complete the API connection fields.")
    preset = PROVIDERS[provider]
    key = options["api_key"].strip()
    parsed, host, query = _destination(options["endpoint"], preset, key)
    auth, param = options["auth"], options["param"].strip()
    if preset["host"] and (auth != preset["auth"] or param != preset["param"]):
        raise ValueError("Use the selected provider's documented authentication method.")
    if auth not in {"query", "header", "bearer", "none"}:
        raise ValueError("Select a supported authentication method.")
    if auth != "none" and (not key or not re.fullmatch(r"[\x20-\x7e]{1,2048}", key)):
        raise ValueError("Enter the provider's key or access token in the masked field.")
    if auth == "none" and key:
        raise ValueError("Clear the key when using an unauthenticated API.")
    if auth in ("query", "header") and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", param):
        raise ValueError("Enter the provider's query parameter or header name.")
    header = None
    if auth == "query":
        if any(k.casefold() == param.casefold() for k, _ in query):
            raise ValueError("Remove the credential parameter from the endpoint URL.")
        query.append((param, key))
    elif auth == "header":
        if param.lower() in {"host","cookie","content-length","transfer-encoding","connection","origin","referer","upgrade","trailer","te","expect"} or param.lower().startswith("proxy-"):
            raise ValueError("Use a header intended for the provider's API key.")
        header = param + ": " + key
    elif auth == "bearer":
        header = "Authorization: Bearer " + key
    url = urlunsplit(("https", host, parsed.path or "/", urlencode(query), ""))
    raw = _fetch(url, host, header, provider == "custom")
    frame = _monthly_data(raw, options, provider)
    data = frame.to_csv(index=False)
    if len(data.encode()) > 2_000_000 or (key and (key in data or key in host)):
        raise ValueError("The mapped output is too large or contains the credential. Check the value fields.")
    return {"csv":data, "filename":f"{host} monthly data", "series":list(frame.columns[1:]), "months":len(frame)}
