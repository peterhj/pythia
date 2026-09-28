"""Account services tested with local callbacks and fake HTTP/token data only."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPConnection
import io
import json
import os
from pathlib import Path
import queue
import socket
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit

from pythia.interaction import cli, CodexAuth, load_codex_auth, ToolCall
from pythia.interaction import load_codex_credentials
from pythia.interaction import DEFAULT_REQUEST_TIMEOUT_SECONDS
from pythia.interaction import USER_AGENT
from pythia.interaction import codex_login, codex_quota, user_tools
from pythia.interaction._account_http import AccountServiceError


def _tokens(account="account-one"):
    payload = base64.urlsafe_b64encode(json.dumps({
        "https://api.openai.com/auth": {"chatgpt_account_id": account},
    }).encode()).rstrip(b"=").decode()
    return {"access_token": "FAKE_ACCESS_SECRET", "refresh_token": "FAKE_REFRESH_SECRET",
            "id_token": f"header.{payload}.signature"}


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


class LoginServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "auth.json"

    def _callback(self, query, *, state=None, params=None, host=None):
        """Send one browser-style callback to the advertised address; return (status, body)."""
        redirect = urlsplit(query["redirect_uri"][0])
        pairs = [("state", query["state"][0] if state is None else state)]
        if params is None:
            params = {"code": "FAKE_CODE_SECRET"}
        pairs.extend(params.items() if isinstance(params, dict) else params)
        connection = HTTPConnection(redirect.hostname, redirect.port, timeout=2)
        try:
            connection.request("GET", redirect.path + "?" + urlencode(pairs),
                               headers={} if host is None else {"Host": host})
            response = connection.getresponse()
            body = response.read().decode()
            # Browser text is application-authored: no code, token, or provider value.
            self.assertNotIn("SECRET", body)
            return response.status, body
        finally:
            connection.close()

    def _run(self, *, complete=True, callback_params=None, callback_status=200,
             cancel_exchange=False, tokens=None, expected_account=None, fail_save=False,
             timeout_seconds=2, request_timeout_seconds=None):
        notices = queue.Queue()
        cancel = threading.Event()
        exchange_entered, exchange_release = threading.Event(), threading.Event()
        # Recorded on self so failure cases can inspect them after login raises.
        self.exchange_requests = request_values = []
        self.callback_response = None
        responses = []

        def opener(request, *, timeout):
            self.assertEqual(request.full_url, "https://auth.openai.com/oauth/token")
            self.assertEqual(request.get_header("User-agent"), USER_AGENT)
            http_timeout = (
                DEFAULT_REQUEST_TIMEOUT_SECONDS
                if request_timeout_seconds is None else request_timeout_seconds
            )
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, min(http_timeout, timeout_seconds))
            values = parse_qs(request.data.decode())
            request_values.append(values)
            if cancel_exchange:
                exchange_entered.set()
                if not exchange_release.wait(2):
                    raise AssertionError("exchange was not released")
            result = Response(tokens or _tokens())
            responses.append(result)
            return result

        request_options = (
            {} if request_timeout_seconds is None
            else {"request_timeout_seconds": request_timeout_seconds}
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(codex_login.login, self.path, notify=notices.put,
                                 cancel=cancel, callback_port=0, opener=opener,
                                 timeout_seconds=timeout_seconds, expected_account=expected_account,
                                 **request_options)
            try:
                notice = notices.get(timeout=2)
                query = parse_qs(urlsplit(notice.splitlines()[-1]).query)
                redirect_port = urlsplit(query["redirect_uri"][0]).port
                self.assertEqual(query["redirect_uri"],
                                 [f"http://127.0.0.1:{redirect_port}/auth/callback"])
                self.assertEqual(query["code_challenge_method"], ["S256"])
                self.assertEqual(query["response_type"], ["code"])
                if complete:
                    self.assertEqual(self._callback(query, state="wrong")[0], 400)
                    if fail_save:
                        patch = mock.patch.object(codex_login.os, "replace", side_effect=OSError("FAKE_ACCESS_SECRET"))
                    else:
                        patch = mock.patch.object(codex_login.os, "replace", wraps=os.replace)
                    with patch:
                        self.callback_response = self._callback(query, params=callback_params)
                        self.assertEqual(self.callback_response[0], callback_status)
                        if cancel_exchange:
                            self.assertTrue(exchange_entered.wait(2))
                            cancel.set()
                            exchange_release.set()
                        future.result(timeout=3)
                elif timeout_seconds >= 1:
                    cancel.set()
                    future.result(timeout=3)
                else:
                    future.result(timeout=3)
            finally:
                cancel.set()
                exchange_release.set()
                # The worker closes the listener even on denial/timeout/write errors.
                try:
                    future.result(timeout=3)
                except AccountServiceError:
                    pass
                self.assertTrue(all(response.closed for response in responses))
                if "query" in locals():
                    port = urlsplit(query["redirect_uri"][0]).port
                    with socket.socket() as probe:
                        probe.settimeout(0.2)
                        self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0)
        if request_values:
            verifier = request_values[0]["code_verifier"][0]
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            self.assertEqual(query["code_challenge"], [challenge])
            self.assertEqual(request_values[0]["redirect_uri"], query["redirect_uri"])
            self.assertEqual(request_values[0]["code"], ["FAKE_CODE_SECRET"])
            self.assertNotIn(verifier, notice)
        return request_values

    def test_login_pkce_callback_atomic_credentials_and_permissions(self):
        self.path.write_text('{"unrelated":"preserved","tokens":{"access_token":"old"}}')
        requests = self._run(expected_account="account-one")
        self.assertEqual(len(requests), 1)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["unrelated"], "preserved")
        self.assertEqual(saved["tokens"]["refresh_token"], "FAKE_REFRESH_SECRET")
        self.assertEqual(load_codex_auth(auth_file=self.path), CodexAuth("FAKE_ACCESS_SECRET", "account-one"))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.path.parent.glob(".auth.*.tmp")), [])

    def test_token_exchange_honors_explicit_http_timeout(self):
        self.assertEqual(len(self._run(request_timeout_seconds=0.25)), 1)

    def test_refresh_retains_omitted_tokens_and_rejects_account_switch(self):
        original_tokens = _tokens("account-one")
        self.path.write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "unrelated": "preserved",
                    "tokens": {**original_tokens, "account_id": "account-one"},
                }
            ),
            encoding="utf-8",
        )
        credentials = load_codex_credentials(auth_file=self.path)
        requests = []

        def successful(request, *, timeout):
            requests.append((json.loads(request.data), timeout))
            return Response({"access_token": "NEW_ACCESS_SECRET"})

        refreshed = codex_login.refresh_codex_credentials(
            credentials,
            opener=successful,
            timeout_seconds=0.25,
        )
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(refreshed.auth.access_token, "NEW_ACCESS_SECRET")
        self.assertEqual(refreshed.refresh_token, original_tokens["refresh_token"])
        self.assertEqual(refreshed.id_token, original_tokens["id_token"])
        self.assertEqual(saved["unrelated"], "preserved")
        self.assertEqual(requests[0][0]["grant_type"], "refresh_token")
        self.assertEqual(requests[0][0]["refresh_token"], "FAKE_REFRESH_SECRET")
        self.assertEqual(requests[0][1], 0.25)

        before = self.path.read_bytes()
        switched = _tokens("account-two")
        current = load_codex_credentials(auth_file=self.path)

        def different_account(request, *, timeout):
            del request, timeout
            return Response(switched)

        with self.assertRaisesRegex(AccountServiceError, "different"):
            codex_login.refresh_codex_credentials(
                current,
                opener=different_account,
            )
        self.assertEqual(self.path.read_bytes(), before)

    def test_denial_cancel_timeout_account_mismatch_and_save_failure_preserve_old_file(self):
        cases = (
            ({"callback_params": {"error": "access_denied"}}, "declined"),
            ({"complete": False}, "cancelled"),
            ({"complete": False, "timeout_seconds": 0.3}, "timed out"),
            ({"cancel_exchange": True}, "cancelled"),
            ({"expected_account": "different-account"}, "mismatch"),
            ({"fail_save": True}, "could not be saved"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                original = '{"tokens":{"access_token":"old"}}'
                self.path.write_text(original)
                with self.assertRaisesRegex(AccountServiceError, message) as error:
                    self._run(**kwargs)
                self.assertNotIn("SECRET", str(error.exception))
                self.assertEqual(self.path.read_text(), original)
                self.assertEqual(list(self.path.parent.glob(".auth.*.tmp")), [])

    def test_port_conflict_and_pre_cancel_do_not_publish_challenge_or_write(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            notify = mock.Mock()
            with self.assertRaisesRegex(AccountServiceError, "listener"):
                codex_login.login(self.path, notify=notify, cancel=threading.Event(),
                                  callback_port=occupied.getsockname()[1])
            notify.assert_not_called()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(AccountServiceError, "cancelled"):
            codex_login.login(self.path, notify=notify, cancel=cancel)
        self.assertFalse(self.path.exists())

    def test_provider_errors_and_unusable_codes_end_login_without_reflection_or_exchange(self):
        declined = "Sign-in was declined. Return to the terminal."
        failed = "Sign-in failed. Return to the terminal for details."
        cases = (
            ({"error": "access_denied",
              "error_description": "FAKE_DESCRIPTION_SECRET: missing_codex_entitlement"},
             200, "Codex is not enabled for this workspace",
             "Codex is not enabled for your workspace. Contact your workspace "
             "administrator, then return to the terminal."),
            ({"error": "access_denied", "error_description": "FAKE_DESCRIPTION_SECRET"},
             200, "declined by the provider", declined),
            ({"error": "server_error", "error_description": "FAKE_DESCRIPTION_SECRET"},
             200, "OAuth error server_error", failed),
            ({"error": "FAKE ERROR SECRET"}, 200, "unrecognized error", failed),
            ({"error": "\x1b[2JFAKE_ERROR_SECRET"}, 200, "unrecognized error", failed),
            ([("error", "access_denied"), ("error", "access_denied")],
             200, "unrecognized error", failed),
            ({"error_description": "FAKE_DESCRIPTION_SECRET"}, 400, "authorization code", failed),
            ({"code": "FAKE CODE SECRET"}, 400, "authorization code", failed),
        )
        for params, status, message, page in cases:
            with self.subTest(params=params):
                original = '{"tokens":{"access_token":"old"}}'
                self.path.write_text(original)
                # The harness waits at most 3 s, so a 30 s budget proves the callback ends login.
                with self.assertRaisesRegex(AccountServiceError, message) as error:
                    self._run(callback_params=params, callback_status=status, timeout_seconds=30)
                self.assertNotIn("SECRET", str(error.exception))
                self.assertEqual(self.callback_response, (status, page))
                self.assertEqual(self.exchange_requests, [])
                self.assertEqual(self.path.read_text(), original)

    def test_callback_host_must_match_the_advertised_loopback_address(self):
        notices = queue.Queue()
        cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(codex_login.login, self.path, notify=notices.put, cancel=cancel,
                                 callback_port=0, timeout_seconds=30,
                                 opener=lambda request, *, timeout: Response(_tokens()))
            try:
                query = parse_qs(urlsplit(notices.get(timeout=2).splitlines()[-1]).query)
                port = urlsplit(query["redirect_uri"][0]).port
                for host in (f"localhost:{port}", f"[::1]:{port}", "127.0.0.1",
                             f"attacker.example:{port}"):
                    with self.subTest(host=host):
                        # A valid state under another Host must neither answer nor end the flow.
                        self.assertEqual(self._callback(query, host=host),
                                         (400, "Invalid login callback."))
                self.assertFalse(future.done())
                self.assertEqual(self._callback(query)[0], 200)
                future.result(timeout=3)
            finally:
                cancel.set()
        self.assertEqual(load_codex_auth(auth_file=self.path),
                         CodexAuth("FAKE_ACCESS_SECRET", "account-one"))


class CallbackFailureTests(unittest.TestCase):
    state = "S" * 43
    declined = ("Login was declined by the provider.",
                "Sign-in was declined. Return to the terminal.")
    unrecognized = ("Login failed: the provider returned an unrecognized error.",
                    "Sign-in failed. Return to the terminal for details.")

    def _failure(self, *pairs):
        # Parse exactly as the callback handler does.
        params = parse_qs(urlencode(pairs), max_num_fields=16)
        return codex_login._callback_failure(params, self.state)

    def test_entitlement_marker_is_recognized_only_for_access_denied(self):
        terminal, page = self._failure(("error", "access_denied"),
                                       ("error_description", "Account MISSING_CODEX_ENTITLEMENT"))
        self.assertIn("Codex is not enabled for this workspace", terminal)
        self.assertIn("Contact your workspace administrator", page)
        self.assertNotIn("MISSING", terminal + page)
        self.assertEqual(
            self._failure(("error", "server_error"),
                          ("error_description", "missing_codex_entitlement"))[0],
            "Login failed: the provider returned OAuth error server_error.")
        self.assertEqual(
            self._failure(("error", "access_denied"),
                          ("error_description", "missing_codex_entitlement"),
                          ("error_description", "missing_codex_entitlement")),
            self.declined)

    def test_only_one_bounded_identifier_without_the_state_is_echoed(self):
        self.assertEqual(self._failure(("error", "access_denied")), self.declined)
        self.assertEqual(self._failure(("error", "a" * 64))[0],
                         f"Login failed: the provider returned OAuth error {'a' * 64}.")
        for value in ("a" * 65, "access denied", "server_error\n", "\x1b[31mserver_error",
                      "caf\u00e9", self.state, f"echo_{self.state}"):
            with self.subTest(value=value):
                self.assertEqual(self._failure(("error", value)), self.unrecognized)
        self.assertEqual(self._failure(("error", "access_denied"), ("error", "server_error")),
                         self.unrecognized)


class QuotaServiceTests(unittest.TestCase):
    def test_plan_identifiers_are_preserved_without_a_tier_whitelist(self):
        payload = {"rate_limit": {"primary_window": {"used_percent": 25}},
                   "credits": {"has_credits": False, "unlimited": False, "balance": "0"}}
        baseline = codex_quota.format_quota(payload, queried_at="fixed").splitlines()
        cases = [(name, name) for name in (
            "free", "plus", "pro", "team", "business", "enterprise", "edu",
            "prolite", "future-plan_v2", "0", "p" * 64,
        )]
        cases.extend(((" ProLite ", "ProLite"), (" " + "p" * 64 + " ", "p" * 64)))
        for value, expected in cases:
            with self.subTest(value=value):
                lines = codex_quota.format_quota({**payload, "plan_type": value}, queried_at="fixed").splitlines()
                self.assertEqual(lines, [baseline[0], f"plan: {expected}", *baseline[2:]])

    def test_invalid_plan_labels_are_unavailable_without_discarding_quota_fields(self):
        payload = {"rate_limit": {"primary_window": {"used_percent": 25}},
                   "additional_rate_limits": [{"metered_feature": "spark", "rate_limit": {}}]}
        expected = codex_quota.format_quota(payload, queried_at="fixed")
        self.assertIn("\nplan: unavailable\n", expected)
        for value in (
            None, "", "   ", 0, False, [], {}, b"prolite", "p" * 65,
            "pro lite", "pro.lite", "pro/lite", "\tprolite", "prolite\t",
            "prolite\n", "\rprolite", "prolite\x00", "prolite\x1b[2J",
            "prolite\u0085", "\u00a0prolite", "prölite",
        ):
            with self.subTest(value=value):
                self.assertEqual(codex_quota.format_quota(
                    {**payload, "plan_type": value}, queried_at="fixed",
                ), expected)

    def test_bearer_reflected_as_valid_plan_identifier_is_still_redacted(self):
        token = "FAKE_BEARER"
        response = Response({"plan_type": token})
        output = codex_quota.query_quota(CodexAuth(token), opener=mock.Mock(return_value=response))
        self.assertIn("\nplan: [redacted]\n", output)
        self.assertNotIn(token, output)
        self.assertTrue(response.closed)

    def test_normalized_quota_request_and_historical_output(self):
        response = Response({"plan_type": "pro", "rate_limit": {
            "primary_window": {"used_percent": 20, "limit_window_seconds": 18000, "reset_at": 2000000000},
        }, "credits": {"has_credits": True, "unlimited": False, "balance": "12.5"},
            "additional_rate_limits": [{"metered_feature": "spark", "rate_limit": {
                "secondary_window": {"used_percent": 5},
            }}]})

        def opener(request, *, timeout):
            self.assertEqual(request.full_url, "https://chatgpt.com/backend-api/wham/usage")
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.get_header("User-agent"), USER_AGENT)
            self.assertEqual(request.get_header("Authorization"), "Bearer FAKE_SECRET")
            self.assertEqual(request.get_header("Chatgpt-account-id"), "account")
            self.assertEqual(timeout, 17)
            return response

        text = codex_quota.query_quota(CodexAuth("FAKE_SECRET", "account"), timeout_seconds=17, opener=opener)
        self.assertTrue(response.closed)
        self.assertIn("Quota snapshot at", text)
        self.assertIn("\nplan: pro\n", text)
        self.assertIn("primary: used=20%", text)
        self.assertIn("secondary: unavailable", text)
        self.assertIn("credits.balance: 12.5", text)
        self.assertIn("spark.secondary: used=5%", text)
        self.assertNotIn("FAKE_SECRET", text)
        self.assertIn("unavailable", codex_quota.format_quota({}))

    def test_absent_null_and_empty_additional_limits_are_equivalent(self):
        payload = {
            "plan_type": "plus",
            "rate_limit": {"primary_window": {"used_percent": 25}},
            "credits": {"has_credits": True, "unlimited": False, "balance": "12.5"},
        }
        expected = codex_quota.format_quota(payload, queried_at="fixed")
        for fields in ({}, {"additional_rate_limits": None}, {"additional_rate_limits": []}):
            with self.subTest(fields=fields):
                value = {**payload, **fields}
                self.assertEqual(codex_quota.format_quota(value, queried_at="fixed"), expected)
                response = Response(value)
                output = codex_quota.query_quota(
                    CodexAuth("FAKE_SECRET"), opener=mock.Mock(return_value=response),
                )
                self.assertEqual(output.splitlines()[1:], expected.splitlines()[1:])
                self.assertTrue(response.closed)

    def test_invalid_additional_limits_remain_rejected(self):
        for value in (False, 0, "", "FAKE_SECRET", {}, {"metered_feature": "spark"}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(AccountServiceError, "invalid additional limits"):
                    codex_quota.format_quota({"additional_rate_limits": value})
        for value in (None, False, 0, "FAKE_SECRET"):
            with self.subTest(entry=value):
                with self.assertRaisesRegex(AccountServiceError, "an invalid additional limit"):
                    codex_quota.format_quota({"additional_rate_limits": [value]})

    def test_invalid_fields_and_http_errors_are_safe_and_closed(self):
        for value in (-1, 101, True, float("nan"), "FAKE_SECRET"):
            with self.subTest(value=value), self.assertRaises(AccountServiceError):
                codex_quota.format_quota({"rate_limit": {"primary_window": {"used_percent": value}}})
        for code in (302, 401, 403):
            body = io.BytesIO(b"FAKE_SECRET in provider response")
            headers = (
                {"x-oai-request-id": "request-auth", "cf-ray": "ray-auth"}
                if code == 401
                else {}
            )
            error = HTTPError(
                "https://example.invalid/FAKE_SECRET",
                code,
                "FAKE_SECRET",
                headers,
                body,
            )
            with self.subTest(code=code):
                with self.assertRaises(AccountServiceError) as raised:
                    codex_quota.query_quota(CodexAuth("FAKE_SECRET"), opener=mock.Mock(side_effect=error))
                self.assertNotIn("FAKE_SECRET", str(raised.exception))
                if code == 401:
                    self.assertIn("request_id=request-auth", str(raised.exception))
                    self.assertIn("cf_ray=ray-auth", str(raised.exception))
                self.assertTrue(body.closed)

    def test_unsupported_routes_unknown_account_and_exception_guard_do_not_expose_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ([], ["--endpoint-api", "codex", "--model", "muse-spark-1.3"],
                            ["--endpoint-api", "codex", "--model", "test",
                             "--endpoint-url", "https://example.org/responses",
                             "--endpoint-auth", "none"]):
                args = cli._build_parser().parse_args(options)
                environment = user_tools.create_user_environment(args, notify=mock.Mock(), cancel=threading.Event())
                with mock.patch.object(user_tools, "login") as login, mock.patch.object(user_tools, "query_quota") as quota:
                    for name in ("login", "quota"):
                        self.assertFalse(environment.execute_tool_calls((ToolCall(name, "one", "{}"),)).items[0].success)
                    login.assert_not_called()
                    quota.assert_not_called()
            args = cli._build_parser().parse_args([
                "--endpoint-api", "codex", "--model", "test",
                "--endpoint-auth-home", directory,
            ])
            environment = user_tools.create_user_environment(args, notify=mock.Mock(), cancel=threading.Event(), provider_history=True)
            with mock.patch.object(user_tools, "login") as login:
                result = environment.execute_tool_calls((ToolCall("login", "one", "{}"),)).items[0]
                self.assertFalse(result.success)
                login.assert_not_called()
            environment = user_tools.create_user_environment(args, notify=mock.Mock(), cancel=threading.Event())
            with mock.patch.object(user_tools, "login", side_effect=RuntimeError("FAKE_SECRET")):
                result = environment.execute_tool_calls((ToolCall("login", "one", "{}"),)).items[0]
            self.assertNotIn("FAKE_SECRET", result.output)


if __name__ == "__main__":
    unittest.main()
