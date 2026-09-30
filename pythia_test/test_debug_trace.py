"""--debug-trace: verbatim, append-only request/response logs next to the save."""

from __future__ import annotations

import asyncio
import base64
from contextlib import redirect_stderr, redirect_stdout
import http.server
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error
import urllib.request

from pythia.interaction import (
    CODEX_RESPONSES_API_URL, ChatCompletionsModel, CodexAuth, Environment, Init,
    InteractionContext, Message, MessagesModel, ModelError, ModelTransportError,
    PromptSummarizingCompactor, ResponsesOpaqueCompactor, ToolResult, cli,
    codex_quota, load_interaction_save,
)
from pythia.interaction import _debug_trace
from pythia.interaction._account_http import AccountServiceError
from pythia.interaction._debug_trace import (
    DebugTrace, TracingOpener, debug_trace_paths, trace_operation,
)
from pythia.interaction.user_tools import UserToolIntent
from pythia_test.interaction_helpers import (
    chat_endpoint, codex_model, messages_endpoint, responses_endpoint,
)
from pythia_test.test_interaction_cli import _Model, _answer
from pythia_test.test_responses import (
    _FailingSSEResponse, _FakeSSEResponse, _ScriptedOpener, _account_id_token,
    _compaction_event, _completed_event, _http_error as _codex_http_error,
    _message_event,
)


TIMESTAMP = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z$"
CHAT_OK = {"choices": [{"finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hello."}}]}
MESSAGES_OK = {"type": "message", "role": "assistant", "stop_reason": "end_turn",
               "content": [{"type": "text", "text": "Done."}],
               "usage": {"input_tokens": 1, "output_tokens": 1}}
CHAT_URL = "http://127.0.0.1:9/v1/chat/completions"


def _rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _http_error(url, status, body=b"", headers=None):
    return urllib.error.HTTPError(url, status, "HTTP failure", dict(headers or {}), io.BytesIO(body))


class _JSONResponse:
    def __init__(self, payload, *, status=200, headers=None):
        self.status = status
        self.headers = dict({"Content-Type": "application/json"} if headers is None else headers)
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self, *args):
        return self.body

    def close(self):
        self.closed = True


class _AccountResponse(io.BytesIO):
    status = 200


class _SizelessResponse:
    """A response whose read() takes no size, like small adapter-test fakes."""

    def __init__(self, status, body):
        self.status = status
        self.headers = {}
        self.body = body

    def read(self):
        return self.body

    def close(self):
        pass


class _ScriptedHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        status, headers, body = self.server.script.pop(0)
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server:
    """A loopback server replaying scripted (status, headers, body) responses."""

    def __init__(self, test, *script):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedHandler)
        self.httpd.script = list(script)
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()
        test.addCleanup(thread.join, 5)
        test.addCleanup(self.httpd.server_close)
        test.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"


class _TraceTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.save = self.root / "session.jsonl"
        self.trace = DebugTrace.open(self.save)
        self.context = InteractionContext((Init("session-1"), Message("user", "hello")))

    def pairs(self):
        """Return (request, response) rows in request order, checking pairing."""
        requests = _rows(self.trace.request_path)
        responses = _rows(self.trace.response_path)
        self.assertEqual(len({row["id"] for row in requests}), len(requests))
        self.assertEqual(sorted(row["id"] for row in requests),
                         sorted(row["id"] for row in responses))
        by_id = {row["id"]: row for row in responses}
        for request in requests:
            response = by_id[request["id"]]
            self.assertEqual(request["type"], "http_request")
            self.assertEqual(response["type"], "http_response")
            for key in ("op", "retry", "t0", "method", "url"):
                self.assertEqual(response[key], request[key])
            self.assertRegex(request["t0"], TIMESTAMP)
            self.assertRegex(response["t1"], TIMESTAMP)
            self.assertLessEqual(request["t0"], response["t1"])
        return [(request, by_id[request["id"]]) for request in requests]

    def chat_model(self, *outcomes, api_key=None):
        inner = _ScriptedOpener(*outcomes)
        model = ChatCompletionsModel(
            chat_endpoint(api_url="http://127.0.0.1:9", api_key=api_key),
            opener=self.trace.opener(inner),
            retry_sleep=lambda _delay: None,
        )
        return model, inner

    def codex(self, *outcomes):
        inner = _ScriptedOpener(*outcomes)
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL, model="codex-test",
                bearer_token="FAKE-BEARER", account_id="account-1", api_provider="codex",
            ),
            opener=self.trace.opener(inner),
            retry_sleep=lambda _delay: None,
        )
        return model, inner


class TraceFileTests(unittest.TestCase):
    def test_paths_append_trace_suffixes_to_the_full_save_name(self):
        self.assertEqual(
            debug_trace_paths(Path("/work/review.jsonl")),
            (Path("/work/review.jsonl.trace.req.jsonl"),
             Path("/work/review.jsonl.trace.res.jsonl")),
        )

    def test_open_creates_private_logs_and_never_truncates(self):
        with tempfile.TemporaryDirectory() as directory:
            save = Path(directory) / "session.jsonl"
            trace = DebugTrace.open(save)
            for path in (trace.request_path, trace.response_path):
                self.assertEqual(path.read_bytes(), b"")
                if os.name == "posix":
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            trace.request_path.write_bytes(b'{"type":"http_request"}\n')
            trace.response_path.write_bytes(b'{"type":"http_resp')  # Crashed mid-line.
            DebugTrace.open(save)
            DebugTrace.open(save)
            self.assertEqual(trace.request_path.read_bytes(), b'{"type":"http_request"}\n')
            self.assertEqual(trace.response_path.read_bytes(), b'{"type":"http_resp\n')

    def test_non_regular_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            save = Path(directory) / "session.jsonl"
            debug_trace_paths(save)[1].mkdir()
            with self.assertRaisesRegex(ValueError, "regular file"):
                DebugTrace.open(save)


class TraceFailureTests(_TraceTestCase):
    def test_write_failure_disables_tracing_without_changing_the_exchange(self):
        model, inner = self.chat_model(_JSONResponse(CHAT_OK), _JSONResponse(CHAT_OK))
        failure = OSError(28, "No space left on device")
        with mock.patch.object(_debug_trace, "_private_opener", side_effect=failure):
            with trace_operation("sample"):
                self.assertEqual(model.sample(self.context).last_assistant_text, "Hello.")
        warning = self.trace.take_warning()
        self.assertIn("debug trace disabled", warning)
        self.assertIn("No space left on device", warning)
        self.assertIsNone(self.trace.take_warning())
        self.assertFalse(self.trace.enabled)
        with trace_operation("sample"):
            self.assertEqual(model.sample(self.context).last_assistant_text, "Hello.")
        self.assertEqual(len(inner.calls), 2)
        self.assertEqual(self.trace.request_path.read_bytes(), b"")
        self.assertEqual(self.trace.response_path.read_bytes(), b"")

    def test_exchanges_outside_an_operation_have_null_op_and_retry(self):
        model, _ = self.chat_model(_JSONResponse(CHAT_OK))
        model.sample(self.context)
        [(request, _)] = self.pairs()
        self.assertIsNone(request["op"])
        self.assertIsNone(request["retry"])


class ModelExchangeTests(_TraceTestCase):
    def test_chat_request_and_response_are_verbatim_including_credentials(self):
        response = _JSONResponse(CHAT_OK, headers={
            "Content-Type": "application/json", "X-Request-Id": "req-1",
        })
        model, inner = self.chat_model(response, api_key="sk-FAKE-SECRET")
        with trace_operation("sample"):
            self.assertEqual(model.sample(self.context).last_assistant_text, "Hello.")
        self.assertTrue(response.closed)
        [(request, reply)] = self.pairs()
        sent, _ = inner.calls[0]
        self.assertEqual((request["op"], request["retry"], request["method"]),
                         ("sample", 0, "POST"))
        self.assertEqual(request["url"], CHAT_URL)
        self.assertEqual(request["payload"], sent.data.decode("utf-8"))
        self.assertIn(["Authorization", "Bearer sk-FAKE-SECRET"], request["headers"])
        self.assertEqual(reply["status"], 200)
        self.assertEqual(reply["headers"], [["Content-Type", "application/json"],
                                            ["X-Request-Id", "req-1"]])
        self.assertEqual(reply["payload"], response.body.decode("utf-8"))
        self.assertIsNone(reply["exception"])

    def test_messages_retry_keeps_the_error_payload_and_headers(self):
        url = "http://127.0.0.1:9/v1/messages"
        error_body = b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}'
        inner = _ScriptedOpener(
            _http_error(url, 429, error_body, {"Retry-After": "0"}),
            _JSONResponse(MESSAGES_OK),
        )
        model = MessagesModel(
            messages_endpoint("http://127.0.0.1:9", "model", api_key="FAKE-KEY",
                              max_output_tokens=100),
            opener=self.trace.opener(inner),
            retry_sleep=lambda _delay: None,
        )
        with trace_operation("sample"):
            self.assertEqual(model.sample(self.context).request_attempts, 2)
        (first, first_reply), (second, second_reply) = self.pairs()
        self.assertEqual([first["retry"], second["retry"]], [0, 1])
        self.assertEqual(first["url"], url)
        self.assertEqual(first["payload"], second["payload"])
        self.assertIn(["X-api-key", "FAKE-KEY"], first["headers"])
        self.assertEqual(first_reply["status"], 429)
        self.assertEqual(first_reply["payload"], error_body.decode())
        self.assertEqual(first_reply["headers"], [["Retry-After", "0"]])
        self.assertIsNone(first_reply["exception"])
        self.assertEqual(second_reply["status"], 200)

    def test_connection_failures_are_responses_with_exceptions(self):
        def refused():
            return urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

        model, _ = self.chat_model(refused(), refused(), refused())
        with trace_operation("sample"), self.assertRaises(ModelTransportError):
            model.sample(self.context)
        pairs = self.pairs()
        self.assertEqual([request["retry"] for request, _ in pairs], [0, 1, 2])
        for _, reply in pairs:
            self.assertIsNone(reply["status"])
            self.assertIsNone(reply["headers"])
            self.assertIsNone(reply["payload"])
            self.assertEqual(reply["exception"]["exc_type"], "URLError")
            self.assertIn("Connection refused", reply["exception"]["exc_val"])
            self.assertIn("Traceback", reply["exception"]["exc_tb"])

    def test_non_utf8_payloads_are_recorded_as_base64(self):
        body = b"\xff\xfe<html>proxy error</html>"
        model, _ = self.chat_model(_http_error(CHAT_URL, 400, body))
        with trace_operation("sample"), self.assertRaises(ModelTransportError):
            model.sample(self.context)
        [(_, reply)] = self.pairs()
        self.assertEqual(reply["status"], 400)
        self.assertIsNone(reply["payload"])
        self.assertEqual(base64.b64decode(reply["payload_base64"]), body)

    def test_outcomes_match_untraced_runs(self):
        context_window = b'{"error":{"message":"maximum context length exceeded"}}'

        def outcomes(wrap):
            results = []
            for outcome in (_JSONResponse(CHAT_OK), _http_error(CHAT_URL, 400, context_window)):
                model = ChatCompletionsModel(
                    chat_endpoint(api_url="http://127.0.0.1:9"),
                    opener=wrap(_ScriptedOpener(outcome)),
                    retry_sleep=lambda _delay: None,
                )
                try:
                    sample = model.sample(self.context)
                except ModelError as exc:
                    results.append((type(exc), str(exc), exc.failure))
                else:
                    results.append((sample.items, sample.usage,
                                    sample.request_attempts, sample.recovery))
            return results

        untraced = outcomes(lambda inner: inner)
        with trace_operation("sample"):
            traced = outcomes(self.trace.opener)
        self.assertEqual(traced, untraced)
        self.assertEqual(traced[1][2].category, "context_window")
        self.assertEqual(len(self.pairs()), 2)


class CodexExchangeTests(_TraceTestCase):
    def test_event_stream_is_recorded_as_consumed(self):
        response = _FakeSSEResponse(_message_event(0, "OK"), _completed_event(),
                                    headers={"x-codex-turn-state": "state-1"})
        model, inner = self.codex(response)
        with trace_operation("sample"):
            self.assertEqual(model.sample(self.context).last_assistant_text, "OK")
        self.assertTrue(response.closed)
        [(request, reply)] = self.pairs()
        sent, _ = inner.calls[0]
        self.assertEqual(request["url"], sent.full_url)
        self.assertEqual(request["payload"], sent.data.decode("utf-8"))
        self.assertIn(["Authorization", "Bearer FAKE-BEARER"], request["headers"])
        self.assertIn(["Chatgpt-account-id", "account-1"], request["headers"])
        self.assertEqual(reply["status"], 200)
        self.assertEqual(reply["headers"], [["x-codex-turn-state", "state-1"]])
        self.assertEqual(reply["payload"], b"".join(response._lines).decode("utf-8"))

    def test_http_error_body_is_complete_and_the_failure_is_unchanged(self):
        error_json = base64.b64encode(b'{"error":{"code":"token_expired"}}').decode()
        headers = {"x-error-json": error_json, "x-request-id": "req-1", "cf-ray": "ray-1"}
        # Larger than the adapter's 1 MiB error-body read.
        body = json.dumps({"error": {"code": "invalid_request", "message": "bad input"},
                           "padding": "x" * (1 << 20)}).encode()

        def failure(wrap):
            model = codex_model(
                responses_endpoint(
                    api_url=CODEX_RESPONSES_API_URL, model="codex-test",
                    bearer_token="FAKE-BEARER", api_provider="codex",
                ),
                opener=wrap(_ScriptedOpener(_codex_http_error(400, body=body, headers=headers))),
            )
            with self.assertRaises(ModelError) as raised:
                model.sample(self.context)
            return type(raised.exception), str(raised.exception), raised.exception.failure

        untraced = failure(lambda inner: inner)
        with trace_operation("sample"):
            traced = failure(self.trace.opener)
        self.assertEqual(traced, untraced)
        self.assertEqual(traced[2].auth_error_code, "token_expired")
        [(_, reply)] = self.pairs()
        self.assertEqual(reply["status"], 400)
        self.assertEqual(reply["payload"], body.decode())
        self.assertIn(["x-error-json", error_json], reply["headers"])
        self.assertIsNone(reply["exception"])

    def test_in_stream_failure_event_is_captured(self):
        failed = {"type": "response.failed", "response": {
            "error": {"code": "server_error", "message": "The model failed."},
        }}
        model, _ = self.codex(_FakeSSEResponse(_message_event(0, "partial"), failed))
        with trace_operation("sample"), self.assertRaises(ModelError) as raised:
            model.sample(self.context)
        self.assertEqual(raised.exception.failure.category, "response_failed")
        [(_, reply)] = self.pairs()
        self.assertEqual(reply["status"], 200)
        self.assertIn('"type":"response.failed"', reply["payload"])
        self.assertIn("The model failed.", reply["payload"])
        self.assertIsNone(reply["exception"])

    def test_mid_stream_timeout_keeps_the_partial_payload_and_exception(self):
        first = _FailingSSEResponse(_message_event(0, "partial"),
                                    failure=TimeoutError("The read operation timed out"))
        model, _ = self.codex(first, _FakeSSEResponse(_message_event(0, "OK"), _completed_event()))
        with trace_operation("sample"):
            sample = model.sample(self.context)
        self.assertEqual(sample.last_assistant_text, "OK")
        self.assertEqual(sample.recovery, ("stream_timeout_retry",))
        (first_request, first_reply), (second_request, second_reply) = self.pairs()
        self.assertEqual([first_request["retry"], second_request["retry"]], [0, 1])
        self.assertEqual(first_reply["status"], 200)
        self.assertEqual(first_reply["payload"], b"".join(first._lines).decode("utf-8"))
        self.assertEqual(first_reply["exception"]["exc_type"], "TimeoutError")
        self.assertEqual(first_reply["exception"]["exc_val"], "The read operation timed out")
        self.assertIsNone(second_reply["exception"])

    def test_read_fallback_and_non_iterable_bodies_behave_as_untraced(self):
        # The adapter answers a read(size) TypeError with read(); only the
        # last read attempt's failure belongs to the response.
        body = b'{"error":{"message":"overloaded"}}'
        model, _ = self.codex(_SizelessResponse(503, body),
                              _FakeSSEResponse(_message_event(0, "OK"), _completed_event()))
        with trace_operation("sample"):
            self.assertEqual(model.sample(self.context).recovery, ("http_503_retry",))
        (_, first_reply), _ = self.pairs()
        self.assertEqual((first_reply["status"], first_reply["payload"]), (503, body.decode()))
        self.assertIsNone(first_reply["exception"])

        def failure(wrap):
            model = codex_model(
                responses_endpoint(api_url=CODEX_RESPONSES_API_URL, model="codex-test",
                                   bearer_token="FAKE-BEARER", api_provider="codex"),
                opener=wrap(_ScriptedOpener(_SizelessResponse(200, b"{}"))),
            )
            with self.assertRaises(ModelError) as raised:
                model.sample(self.context)
            return type(raised.exception), str(raised.exception)

        untraced = failure(lambda inner: inner)
        self.assertEqual(failure(self.trace.opener), untraced)
        self.assertIn("iterable SSE stream", untraced[1])

    def test_oauth_refresh_is_traced_verbatim_under_its_own_op(self):
        auth_file = self.root / "auth.json"
        auth_file.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {
            "access_token": "old-token", "refresh_token": "old-refresh",
            "id_token": _account_id_token("account-1"), "account_id": "account-1",
        }}), encoding="utf-8")
        refreshed = json.dumps({
            "access_token": "new-token", "refresh_token": "new-refresh",
            "id_token": _account_id_token("account-1"),
        }).encode("utf-8")
        model = codex_model(
            model="codex-test",
            auth_file=auth_file,
            opener=self.trace.opener(_ScriptedOpener(
                _codex_http_error(401), _codex_http_error(401),
                _FakeSSEResponse(_message_event(0, "refreshed"), _completed_event()),
            )),
            auth_opener=self.trace.opener(
                _ScriptedOpener(_FakeSSEResponse(status=200, body=refreshed)),
                op="auth_refresh",
            ),
            retry_sleep=lambda _delay: None,
        )
        with trace_operation("sample"):
            sample = model.sample(InteractionContext((Message("user", "hello"),)))
        self.assertEqual(sample.recovery,
                         ("credential_reload_unchanged", "http_401_retry", "oauth_refresh"))
        pairs = self.pairs()
        self.assertEqual([(request["op"], request["retry"]) for request, _ in pairs],
                         [("sample", 0), ("sample", 1), ("auth_refresh", 0), ("sample", 2)])
        self.assertEqual([reply["status"] for _, reply in pairs], [401, 401, 200, 200])
        self.assertEqual(
            [dict(request["headers"]).get("Authorization") for request, _ in pairs],
            ["Bearer old-token", "Bearer old-token", None, "Bearer new-token"],
        )
        refresh, refresh_reply = pairs[2]
        self.assertEqual(refresh["method"], "POST")
        self.assertEqual(refresh["url"], "https://auth.openai.com/oauth/token")
        self.assertEqual(json.loads(refresh["payload"])["refresh_token"], "old-refresh")
        self.assertEqual(refresh_reply["payload"], refreshed.decode("utf-8"))


class CompactionTraceTests(_TraceTestCase):
    def test_remote_compaction_requests_are_tagged_compact(self):
        model, _ = self.codex(_FakeSSEResponse(_compaction_event(0, "checkpoint"),
                                               _completed_event()))
        with trace_operation("compact"):
            ResponsesOpaqueCompactor(model).compact(self.context)
        [(request, reply)] = self.pairs()
        self.assertEqual((request["op"], request["retry"]), ("compact", 0))
        self.assertEqual(json.loads(request["payload"])["input"][-1],
                         {"type": "compaction_trigger"})
        self.assertIn(["X-codex-beta-features", "remote_compaction_v2"], request["headers"])
        self.assertEqual(reply["status"], 200)

    def test_prompt_summary_compaction_requests_are_tagged_compact(self):
        summary = {"choices": [{"finish_reason": "stop",
                                "message": {"role": "assistant", "content": "Summary."}}]}
        model, _ = self.chat_model(_JSONResponse(summary))
        with trace_operation("compact"):
            PromptSummarizingCompactor(model).compact(self.context)
        [(request, reply)] = self.pairs()
        self.assertEqual((request["op"], request["retry"]), ("compact", 0))
        self.assertEqual(json.loads(reply["payload"]), summary)


class AccountTraceTests(_TraceTestCase):
    def test_quota_get_is_traced_with_its_method_and_no_payload(self):
        body = json.dumps({"plan_type": "pro"}).encode("utf-8")
        response = _AccountResponse(body)
        with trace_operation("quota"):
            text = codex_quota.query_quota(
                CodexAuth("FAKE_ACCESS", "account-1"), timeout_seconds=5,
                opener=self.trace.opener(mock.Mock(return_value=response)),
            )
        self.assertIn("\nplan: pro\n", text)
        self.assertTrue(response.closed)
        [(request, reply)] = self.pairs()
        self.assertEqual((request["op"], request["retry"], request["method"]),
                         ("quota", 0, "GET"))
        self.assertEqual(request["url"], "https://chatgpt.com/backend-api/wham/usage")
        self.assertIsNone(request["payload"])
        self.assertIn(["Authorization", "Bearer FAKE_ACCESS"], request["headers"])
        self.assertEqual(reply["status"], 200)
        self.assertEqual(reply["payload"], body.decode("utf-8"))

    def test_account_error_bodies_are_captured_although_callers_never_read_them(self):
        body = b'{"detail":"Unauthorized"}'
        error = _http_error("https://chatgpt.com/backend-api/wham/usage", 401, body,
                            {"x-request-id": "req-7"})
        with trace_operation("quota"), self.assertRaises(AccountServiceError) as raised:
            codex_quota.query_quota(CodexAuth("FAKE_ACCESS"),
                                    opener=self.trace.opener(mock.Mock(side_effect=error)))
        self.assertEqual(str(raised.exception), "Account service HTTP 401 (request_id=req-7).")
        [(_, reply)] = self.pairs()
        self.assertEqual(reply["status"], 401)
        self.assertEqual(reply["headers"], [["x-request-id", "req-7"]])
        self.assertEqual(reply["payload"], body.decode("utf-8"))


class LoopbackTraceTests(_TraceTestCase):
    def test_real_urllib_responses_and_errors_are_traced_transparently(self):
        error_body = b'{"error":{"message":"bad request"}}'
        stream = b"".join(_FakeSSEResponse(_message_event(0, "streamed"), _completed_event())._lines)
        server = _Server(
            self,
            (400, [("Content-Type", "application/json"), ("X-Request-Id", "req-400")], error_body),
            (200, [("Content-Type", "application/json")], json.dumps(CHAT_OK).encode()),
            (200, [("Content-Type", "text/event-stream")], stream),
        )
        chat = ChatCompletionsModel(chat_endpoint(api_url=server.url),
                                    opener=self.trace.opener(urllib.request.urlopen),
                                    retry_sleep=lambda _delay: None)
        with trace_operation("sample"), self.assertRaises(ModelTransportError):
            chat.sample(self.context)
        with trace_operation("sample"):
            self.assertEqual(chat.sample(self.context).last_assistant_text, "Hello.")
        codex = codex_model(
            responses_endpoint(api_url=server.url, model="model", bearer_token="token",
                               api_provider="api"),
            opener=self.trace.opener(urllib.request.urlopen),
        )
        with trace_operation("sample"):
            self.assertEqual(codex.sample(self.context).last_assistant_text, "streamed")
        (error_request, error_reply), (_, ok_reply), (stream_request, stream_reply) = self.pairs()
        self.assertEqual(error_request["url"], f"{server.url}/v1/chat/completions")
        self.assertEqual(error_reply["status"], 400)
        self.assertEqual(error_reply["payload"], error_body.decode())
        self.assertIn(["X-Request-Id", "req-400"], error_reply["headers"])
        self.assertEqual(ok_reply["status"], 200)
        self.assertEqual(json.loads(ok_reply["payload"]), CHAT_OK)
        self.assertEqual(stream_request["url"], f"{server.url}/responses")
        self.assertIn(["Content-Type", "text/event-stream"], stream_reply["headers"])
        self.assertEqual(stream_reply["payload"], stream.decode())


class CLITraceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.save = self.root / "session.jsonl"
        self.request_log, self.response_log = debug_trace_paths(self.save)

    def main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["--headless", "--cwd", str(self.root), "--save", str(self.save),
                             "--enable-default-tools=False", *argv])
        return code, stderr.getvalue()

    def test_flag_is_off_by_default(self):
        self.assertFalse(cli._build_parser().parse_args([]).debug_trace)
        self.assertTrue(cli._build_parser().parse_args(["--debug-trace"]).debug_trace)

    def test_headless_runs_append_both_logs_even_without_resume(self):
        ok = (200, [("Content-Type", "application/json")], json.dumps(CHAT_OK).encode())
        server = _Server(self, ok, ok)
        endpoint = ["--endpoint-api", "chat-completions",
                    "--endpoint-url", f"{server.url}/v1/chat/completions",
                    "--endpoint-auth", "none", "--endpoint-model", "test-model"]
        for prompt in ("first", "second"):
            code, stderr = self.main("--debug-trace", *endpoint, "--prompt", prompt)
            self.assertEqual(code, 0, stderr)
            self.assertIn(f"Debug trace: {self.request_log} and {self.response_log}", stderr)
        requests, responses = _rows(self.request_log), _rows(self.response_log)
        self.assertEqual([json.loads(row["payload"])["messages"][-1]["content"]
                          for row in requests], ["first", "second"])
        self.assertEqual([(row["op"], row["retry"], row["method"]) for row in requests],
                         [("sample", 0, "POST")] * 2)
        self.assertEqual([row["id"] for row in responses], [row["id"] for row in requests])
        self.assertEqual([row["status"] for row in responses], [200, 200])
        # The save was replaced by the second run; the trace kept both.
        saved = load_interaction_save(self.save).items
        self.assertIn(Message("user", "second"), saved)
        self.assertNotIn(Message("user", "first"), saved)
        self.assertIn(Message("assistant", "Hello."), saved)

    def test_unusable_trace_destination_fails_before_the_model_is_built(self):
        self.request_log.mkdir()
        with mock.patch.object(cli, "build_model") as build:
            code, stderr = self.main("--debug-trace", "--prompt", "hello")
        self.assertEqual(code, 1)
        build.assert_not_called()
        self.assertIn("debug trace destination must be a regular file", stderr)
        self.assertFalse(self.save.exists())

    def test_without_the_flag_no_logs_are_created(self):
        model = _Model(self.save, _answer())
        with mock.patch.object(cli, "build_model", side_effect=lambda args: model):
            code, stderr = self.main("--prompt", "hello")
        self.assertEqual(code, 0, stderr)
        self.assertFalse(self.request_log.exists())
        self.assertFalse(self.response_log.exists())
        self.assertNotIn("Debug trace", stderr)

    def test_model_rebuilds_and_account_tools_receive_traced_openers(self):
        trace = DebugTrace.open(self.save)
        args = cli._build_parser().parse_args([])
        with mock.patch.object(cli, "build_model") as build:
            cli._build_model(args, trace)
            cli._build_model(args, None)
        traced, plain = build.call_args_list
        self.assertEqual(plain, mock.call(args))
        opener, auth_opener = traced.kwargs["opener"], traced.kwargs["auth_opener"]
        self.assertIsInstance(opener, TracingOpener)
        self.assertIs(opener.inner, urllib.request.urlopen)
        self.assertIsNone(opener.op)
        self.assertEqual(auth_opener.op, "auth_refresh")
        # Account requests keep their non-redirecting opener underneath.
        self.assertTrue(any(type(handler).__name__ == "_NoRedirect"
                            for handler in auth_opener.inner.__self__.handlers))

        def execute(calls):
            return SimpleNamespace(items=(ToolResult(calls[0].call_id, "snapshot"),))

        state = cli._UIState(headless=True, trace=trace)
        with mock.patch.object(cli, "create_user_environment") as create:
            create.return_value.execute_tool_calls.side_effect = execute
            asyncio.run(cli._user_tool(
                UserToolIntent("quota", "{}"), None, InteractionContext((Init(model="m"),)),
                state, self.save, args, Environment(), mock.Mock(),
            ))
        account_opener = create.call_args.kwargs["opener"]
        self.assertIsInstance(account_opener, TracingOpener)
        self.assertIsNone(account_opener.op)
        self.assertEqual(type(account_opener.inner.__self__).__name__, "OpenerDirector")

    def test_trace_failure_is_reported_once_as_a_notice(self):
        trace = DebugTrace.open(self.save)
        state = cli._UIState(headless=True, trace=trace)
        trace._fail("Warning: debug trace disabled; could not write X: boom")
        with redirect_stderr(io.StringIO()) as stderr:
            for _ in range(2):
                with cli._traced_operation(state, "sample"):
                    pass
        self.assertEqual(stderr.getvalue().count("debug trace disabled"), 1)


if __name__ == "__main__":
    unittest.main()
