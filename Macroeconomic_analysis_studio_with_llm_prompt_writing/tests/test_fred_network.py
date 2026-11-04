import json
import socket
import ssl
import subprocess
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from studio_core.sources import import_fred


KEY = "ab" * 16  # Test placeholder.


class FredNetworkTests(unittest.TestCase):
    def test_macos_recovers_from_python_certificate_failure(self):
        observations = {"observations": [{"date": "2023-01-01", "value": "3.4"}]}
        def system_client(command, **options):
            self.assertEqual(command[:2], ["/usr/bin/curl", "--disable"])
            self.assertIn("--no-location", command)
            self.assertEqual(command[command.index("--proto") + 1], "=https")
            self.assertNotIn("--insecure", command)
            self.assertNotIn(KEY, " ".join(command))
            self.assertIn(b"api_key=" + KEY.encode(), options["input"])
            return subprocess.CompletedProcess(command, 0, json.dumps(observations).encode() + b"\n200")
        with patch("studio_core.sources.sys.platform", "darwin"), \
             patch("studio_core.sources.build_opener") as opener, \
             patch("studio_core.sources.subprocess.run", side_effect=system_client) as curl:
            opener.return_value.open.side_effect = URLError(ssl.SSLCertVerificationError())
            result = import_fred("UNRATE", KEY)
        curl.assert_called_once()
        self.assertIn("2023-01-01,3.4", result["csv"])
        self.assertNotIn(KEY, str(result))

    def test_macos_reports_both_clients_without_disclosing_the_url(self):
        cases = [(socket.gaierror(), 6, "DNS", "DNS"), (TimeoutError(), 28, "timed out", "timed out"),
                 (ssl.SSLCertVerificationError(), 60, "certificate", "certificate"),
                 (OSError("Tunnel connection failed: 407 "+KEY), 5, "proxy", "proxy")]
        for original, status, py_text, curl_text in cases:
            with self.subTest(code=status), patch("studio_core.sources.sys.platform", "darwin"), \
                 patch("studio_core.sources.build_opener") as opener, \
                 patch("studio_core.sources.subprocess.run") as curl:
                opener.return_value.open.side_effect = URLError(original)
                curl.return_value = subprocess.CompletedProcess([], status, b"", b"URL with key "+KEY.encode())
                with self.assertRaises(ValueError) as err:
                    import_fred("UNRATE", KEY)
                python_message, mac_message = str(err.exception).split("macOS HTTPS retry:")
                self.assertIn(py_text, python_message)
                self.assertIn(curl_text, mac_message)
                self.assertNotIn(KEY, str(err.exception))

    def test_http_response_is_not_retried_or_called_a_network_failure(self):
        for code, message in [(400,"Check the key"), (407,"proxy"), (429,"rate limit"), (503,"Retry later"), (302,"redirect")]:
            with self.subTest(code=code), patch("studio_core.sources.sys.platform", "darwin"), \
                 patch("studio_core.sources.build_opener") as opener, \
                 patch("studio_core.sources.subprocess.run") as curl:
                opener.return_value.open.side_effect = HTTPError("https://example.invalid/?api_key="+KEY,
                                                                 code, "failure", {}, None)
                with self.assertRaisesRegex(ValueError, message) as err:
                    import_fred("UNRATE", KEY)
                curl.assert_not_called()
                self.assertNotIn(KEY, str(err.exception))

    def test_http_error_after_verified_retry_keeps_provider_status(self):
        with patch("studio_core.sources.sys.platform", "darwin"), \
             patch("studio_core.sources.build_opener") as opener, \
             patch("studio_core.sources.subprocess.run") as curl:
            opener.return_value.open.side_effect = TimeoutError()
            curl.return_value = subprocess.CompletedProcess([], 0, b"echoed key "+KEY.encode()+b"\n429")
            with self.assertRaisesRegex(ValueError, "HTTP 429") as err:
                import_fred("UNRATE", KEY)
            self.assertNotIn(KEY, str(err.exception))
