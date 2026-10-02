from __future__ import annotations

from pythia_test.interaction_helpers import responses_endpoint
from pythia_test.interaction_helpers import codex_model

import base64
import http.client
import io
import json
import os
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from pythia.interaction import BUILTIN_MODEL_CATALOG
from pythia.interaction import CODEX_RESPONSES_API_URL
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import CodexAuth
from pythia.interaction import CodexResponsesModel
from pythia.interaction import COMPACTION_SUMMARY_PREFIX
from pythia.interaction import COMPACTION_SUMMARY_SUFFIX
from pythia.interaction import CompactionError
from pythia.interaction import CompactionMetadata
from pythia.interaction import CompactionSettings
from pythia.interaction import ContextPrefix
from pythia.interaction import DEFAULT_REQUEST_TIMEOUT_SECONDS
from pythia.interaction import Environment
from pythia.interaction import Init
from pythia.interaction import Instructions
from pythia.interaction import META_RESPONSES_API_URL
from pythia.interaction import Message
from pythia.interaction import ModelAuthenticationError
from pythia.interaction import ModelConfigurationError
from pythia.interaction import ModelContextWindowError
from pythia.interaction import InteractionContext
from pythia.interaction import ModelResponseError
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import ModelTransportError
from pythia.interaction import OpaqueCompaction
from pythia.interaction import PiCompactor
from pythia.interaction import Reasoning
from pythia.interaction import REMOTE_COMPACTION_V2_RETAINED_USER_MESSAGE_TOKENS
from pythia.interaction import ResponsesOpaqueCompactor
from pythia.interaction import SampleParams
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import TokenUsage
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import SampleMetadata
from pythia.interaction import UserInteraction
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import USER_AGENT
from pythia.interaction import X_CODEX_TURN_STATE_HEADER
from pythia.interaction import create_default_compactor
from pythia.interaction import load_interaction_save
from pythia.interaction import save_interaction_save
from pythia.interaction.compaction import _LEGACY_SUMMARY_PREFIX
from pythia.interaction.demo import DEFAULT_PROMPT
from pythia.interaction.demo import EXPERIMENTAL_USER_MESSAGE_PROMPT
from pythia.interaction.demo import _build_model
from pythia.interaction.demo import _build_parser
from pythia.interaction.demo import run
from pythia.interaction.experimental_tools import create_inject_user_message_tool


def _event_lines(payload, *, event_name=None, crlf=False):
    ending = "\r\n" if crlf else "\n"
    lines = []
    if event_name is not None:
        lines.append(f"event: {event_name}{ending}".encode("utf-8"))
    lines.append(
        f"data: {json.dumps(payload, separators=(',', ':'))}{ending}".encode(
            "utf-8"
        )
    )
    lines.append(ending.encode("utf-8"))
    return lines


def _completed_event(
    *,
    response_id=None,
    input_tokens=0,
    output_tokens=0,
    total_tokens=0,
    cached_tokens=0,
):
    response = {
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_tokens_details": {
                "cached_tokens": cached_tokens,
            },
        }
    }
    if response_id is not None:
        response["id"] = response_id
    return {
        "type": "response.completed",
        "response": response,
    }


def _message_event(index, text):
    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                }
            ],
        },
    }
    if index is not None:
        event["output_index"] = index
    return event


def _reasoning_event(index, *, text, summary, encrypted_content):
    return {
        "type": "response.output_item.done",
        "output_index": index,
        "item": {
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": summary,
                }
            ],
            "content": [
                {
                    "type": "reasoning_text",
                    "text": text,
                }
            ],
            "encrypted_content": encrypted_content,
        },
    }


def _tool_call_event(index, *, call_id="call-1"):
    return {
        "type": "response.output_item.done",
        "output_index": index,
        "item": {
            "type": "function_call",
            "name": "lookup",
            "call_id": call_id,
            "arguments": '{"query":"pythia"}',
        },
    }


def _compaction_event(index, encrypted_content="encrypted-checkpoint"):
    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "compaction",
            "encrypted_content": encrypted_content,
        },
    }
    if index is not None:
        event["output_index"] = index
    return event


class _FakeSSEResponse:
    def __init__(
        self,
        *payloads,
        headers=None,
        status=200,
        body=b"",
        crlf=False,
    ):
        self.status = status
        self.headers = dict(headers or {})
        self.closed = False
        self._body = body
        self._lines = []
        for payload in payloads:
            self._lines.extend(_event_lines(payload, crlf=crlf))

    def __iter__(self):
        return iter(self._lines)

    def read(self, size=-1):
        return self._body if size < 0 else self._body[:size]

    def close(self):
        self.closed = True


class _FailingSSEResponse(_FakeSSEResponse):
    def __init__(self, *payloads, failure, **kwargs):
        super().__init__(*payloads, **kwargs)
        self.failure = failure

    def __iter__(self):
        yield from self._lines
        raise self.failure


class _ScriptedOpener:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if not self.outcomes:
            raise AssertionError("unexpected HTTP request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(request, timeout=timeout)
        return outcome


def _request_payload(opener, index=0):
    request, _ = opener.calls[index]
    return json.loads(request.data.decode("utf-8"))


def _request_headers(opener, index=0):
    request, _ = opener.calls[index]
    return {
        name.lower(): value
        for name, value in request.header_items()
    }


def _account_id_token(account_id):
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": account_id,
                }
            }
        ).encode("utf-8")
    ).rstrip(b"=").decode("ascii")
    return f"header.{payload}.signature"


def _http_error(status, *, body=b"", headers=None):
    return urllib.error.HTTPError(
        CODEX_RESPONSES_API_URL,
        status,
        "HTTP failure",
        dict(headers or {}),
        io.BytesIO(body),
    )


class StreamingResponsesEndpointTests(unittest.TestCase):
    def test_endpoint_normalizes_url_and_redacts_token(self):
        endpoint = responses_endpoint(
            api_url=" HTTPS://api.example.test:8443/proxy/root/ ",
            model=" codex-test ",
            bearer_token=" secret-token ",
            account_id=" account-1 ",
            api_provider=" CODEX ",
        )

        self.assertEqual(
            endpoint.url,
            "HTTPS://api.example.test:8443/proxy/root/responses",
        )
        self.assertEqual(endpoint.model, "codex-test")
        self.assertEqual(endpoint.account_id, "account-1")
        self.assertEqual(endpoint.api_provider, "codex")
        self.assertNotIn("secret-token", repr(endpoint))

    def test_endpoint_rejects_invalid_configuration(self):
        base = {
            "api_url": "https://api.example.test/v1",
            "model": "codex-test",
            "bearer_token": "token",
        }
        cases = (
            {"api_url": ""},
            {"api_url": "localhost:8000"},
            {"api_url": "ftp://localhost"},
            {"api_url": "http:///missing-host"},
            {"api_url": "http://user:password@localhost"},
            {"api_url": "http://localhost/prefix?query=value"},
            {"api_url": "http://localhost/prefix#fragment"},
            {"api_url": "http://localhost:not-a-port"},
            {"api_url": "http://localhost:0"},
            {"model": " "},
            {"bearer_token": " "},
            {"bearer_token": "two words"},
            {"api_provider": "unknown"},
            {"account_id": "account-1"},
            {"request_timeout_seconds": 0},
        )
        for override in cases:
            kwargs = dict(base)
            kwargs.update(override)
            with self.subTest(override=override):
                with self.assertRaises((ModelConfigurationError, ValueError)):
                    responses_endpoint(**kwargs)


class CodexResponsesConstructionTests(unittest.TestCase):
    def test_model_loads_auth_and_builds_default_codex_endpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "codex-token",
                            "account_id": "account-1",
                        }
                    }
                ),
                encoding="utf-8",
            )

            model = codex_model(
                model="codex-test",
                auth_file=auth_file,
            )

        self.assertEqual(model.endpoint.url, CODEX_RESPONSES_API_URL + "/responses")
        self.assertEqual(model.endpoint.model, "codex-test")
        self.assertEqual(model.endpoint.account_id, "account-1")
        self.assertEqual(model.endpoint.api_provider, "codex")
        self.assertEqual(
            model.endpoint.request_timeout_seconds,
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        self.assertNotIn("codex-token", repr(model.endpoint))

    def test_model_accepts_explicit_auth_and_endpoint_overrides(self):
        model = codex_model(
            model="codex-test",
            auth=CodexAuth(
                access_token="override-token",
                account_id="account-2",
            ),
            api_url="https://proxy.example.test/codex",
            request_timeout_seconds=12,
        )

        self.assertEqual(
            model.endpoint.url,
            "https://proxy.example.test/codex/responses",
        )
        self.assertEqual(model.endpoint.account_id, "account-2")
        self.assertEqual(model.endpoint.request_timeout_seconds, 12.0)

        endpoint = responses_endpoint(
            api_url="https://api.example.test/v1",
            model="generic-model",
            bearer_token="api-key",
        )
        overridden = codex_model(endpoint)
        self.assertIs(overridden.endpoint, endpoint)

    def test_context_token_metadata_respects_model_and_provider_routes(self):
        cases = (
            ("codex-gpt-5.6-sol", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-5.6-sol-medium", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-5.6-sol-max", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-astra", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-astra-medium", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-astra-max", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-sol", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-sol-medium", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-sol-max", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-luna", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-luna-medium", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6-luna-max", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6.1-sol", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6.1-sol-medium", (872_000, 1_000_000, 128_000)),
            ("codex-gpt-6.1-sol-max", (872_000, 1_000_000, 128_000)),
            ("muse-spark-1.3", (None, None, None)),
            ("codex-gpt-5.6-sol-high", (None, None, None)),
            ("unknown-model", (None, None, None)),
        )
        for api_provider in ("codex", "api"):
            for requested_model, codex_limits in cases:
                with self.subTest(
                    model=requested_model,
                    api_provider=api_provider,
                ):
                    model = codex_model(
                        responses_endpoint(
                            api_url="https://api.example.test/v1",
                            model=requested_model,
                            bearer_token="token",
                            api_provider=api_provider,
                        ),
                    )

                    self.assertEqual(
                        (
                            model.auto_compact_context_tokens,
                            model.max_context_tokens,
                            model.max_output_tokens,
                        ),
                        (
                            codex_limits
                            if api_provider == "codex"
                            else (None, None, None)
                        ),
                    )

    def test_muse_model_uses_meta_responses_endpoint_by_default(self):
        with mock.patch.dict(
            "os.environ",
            {"META_API_KEY": " meta-api-key "},
            clear=True,
        ):
            model = codex_model(model=" muse-spark-1.3 ")

        self.assertEqual(model.endpoint.url, META_RESPONSES_API_URL + "/responses")
        self.assertEqual(
            model.endpoint.url,
            "https://api.meta.ai/v1/responses",
        )
        self.assertEqual(model.endpoint.model, "muse-spark-1.3-contributor")
        self.assertEqual(model.endpoint.api_provider, "codex")
        self.assertEqual(model.endpoint.bearer_token, "meta-api-key")
        self.assertIsNone(model.endpoint.account_id)
        self.assertNotIn("meta-api-key", repr(model.endpoint))

    def test_muse_model_requires_meta_api_key_without_explicit_auth(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                ModelConfigurationError,
                "META_API_KEY",
            ):
                codex_model(model="muse-spark-1.3-xhigh")

    def test_explicit_api_url_overrides_muse_model_default(self):
        model = codex_model(
            model="muse-spark-1.3",
            auth=CodexAuth(access_token="codex-token"),
            api_url="https://proxy.example.test/meta",
        )

        self.assertEqual(
            model.endpoint.url,
            "https://proxy.example.test/meta/responses",
        )

    def test_model_rejects_missing_or_conflicting_construction_options(self):
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "model is required",
        ):
            codex_model()
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "model is required",
        ):
            codex_model(
                model=" ",
                auth=CodexAuth(access_token="token"),
            )

        endpoint = responses_endpoint(
            api_url="https://api.example.test/v1",
            model="generic-model",
            bearer_token="api-key",
        )
        with self.assertRaises(TypeError):
            CodexResponsesModel(endpoint, model="other-model")
        with self.assertRaisesRegex(TypeError, "retry_sleep"):
            codex_model(endpoint, retry_sleep=object())
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "credentials conflict with endpoint auth policy",
        ):
            codex_model(
                model="codex-test",
                auth=CodexAuth(access_token="token"),
                auth_file="/tmp/auth.json",
            )


class CodexResponsesModelTests(unittest.TestCase):
    def test_codex_maps_instructions_and_legacy_system_messages_to_developer(self):
        for provider, role in (("codex", "developer"), ("api", "system")):
            with self.subTest(provider=provider):
                context = InteractionContext((
                    Init("session-test"),
                    Instructions("old instructions"),
                    Message("user", "first request"),
                    Instructions("effective instructions"),
                    Message("system", "legacy system message"),
                    Message("developer", "explicit developer message"),
                    Message("user", "follow-up"),
                ))
                before = context.items
                opener = _ScriptedOpener(_FakeSSEResponse(_message_event(0, "OK"), _completed_event()))
                model = codex_model(responses_endpoint(
                    api_url=CODEX_RESPONSES_API_URL,
                    model="codex-gpt-6-astra-max", bearer_token="test-token", api_provider=provider,
                ), opener=opener)
                self.assertEqual(model.sample(context).last_assistant_text, "OK")
                encoded = _request_payload(opener)["input"]
                self.assertEqual([i["role"] for i in encoded], [role, "user", role, "developer", "user"])
                self.assertEqual(encoded[0]["content"], [{"type": "input_text", "text": "effective instructions"}])
                self.assertEqual(encoded[2]["content"], [{"type": "input_text", "text": "legacy system message"}])
                self.assertEqual(context.items, before)

    def test_codex_instruction_absence_and_empty_text_remain_distinct(self):
        model = codex_model(responses_endpoint(
            api_url=CODEX_RESPONSES_API_URL, model="codex-gpt-6-astra-max",
            bearer_token="test-token", api_provider="codex",
        ))
        for text in (None, "", " \t "):
            with self.subTest(text=text):
                items = (() if text is None else (Instructions(text),))
                context = InteractionContext((*items, Message("user", "hello")))
                before = context.items
                payload, _ = model._build_request_payload(context, (), None)
                encoded = payload["input"]
                self.assertEqual([i["role"] for i in encoded],
                                 ["user"] if text is None else ["developer", "user"])
                if text is not None:
                    self.assertEqual(encoded[0]["content"][0]["text"], text)
                self.assertEqual(context.items, before)

    def test_responses_opaque_compaction_subtype_is_enforced(self):
        response = _FakeSSEResponse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "compaction",
                    "encrypted_content": "new-encrypted-summary",
                },
            },
            _message_event(1, "continued"),
            _completed_event(),
        )
        opener = _ScriptedOpener(response)
        model = codex_model(
            responses_endpoint(
                api_url="http://localhost:8000/v1",
                model="model",
                bearer_token="token",
                api_provider="api",
            ),
            opener=opener,
        )

        sample = model.sample(
            InteractionContext(
                (
                    OpaqueCompaction.from_responses("encrypted-summary"),
                    Message(role="user", content="continue"),
                )
            )
        )

        self.assertEqual(
            _request_payload(opener)["input"][0],
            {
                "type": "compaction",
                "encrypted_content": "encrypted-summary",
            },
        )
        self.assertEqual(
            sample.items,
            (
                OpaqueCompaction.from_responses("new-encrypted-summary"),
                Message(role="assistant", content="continued"),
            ),
        )
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "Messages opaque compaction",
        ):
            model.sample(
                InteractionContext(
                    (
                        OpaqueCompaction.from_messages("summary"),
                        Message(role="user", content="continue"),
                    )
                )
            )

    def test_remote_v2_compaction_builds_client_prefix_and_preserves_state(self):
        response = _FakeSSEResponse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "future_unrelated_output", "secret": "ignored"},
            },
            _tool_call_event(1, call_id="must-not-execute"),
            _compaction_event(2, "new-encrypted-checkpoint"),
            _completed_event(
                response_id="response-compact-1",
                input_tokens=120,
                output_tokens=9,
                total_tokens=129,
                cached_tokens=80,
            ),
            headers={"x-codex-turn-state": "replacement-state"},
        )
        opener = _ScriptedOpener(response)
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="secret-token",
                account_id="account-1",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=lambda: "unexpected-new-identifier",
        )
        metadata = SampleMetadata(
            TokenUsage(),
            provider_turn_id="turn-1",
            provider_turn_state="sticky-state",
        )
        context = InteractionContext((
            Init("session-1"),
            Instructions("Keep these instructions."),
            Message("user", "First request."),
            UserInteractionBoundary(),
            Message("assistant", "Old answer."),
            metadata,
            Message("user", "Latest request."),
        ))
        before = context.items
        tool = ToolSpec("lookup", "Look things up.", {
            "type": "object", "properties": {},
        })

        with mock.patch(
            "pythia.interaction.compaction.perf_counter",
            side_effect=(200.0, 286.25),
        ):
            result = ResponsesOpaqueCompactor(model).compact(
                context,
                tools=(tool,),
            )

        self.assertEqual(context.items, before)
        self.assertEqual(
            result.usage,
            TokenUsage(120, 9, 129, 80),
        )
        self.assertEqual(result.protocol, "responses_compaction_v2")
        self.assertEqual(result.elapsed_seconds, 86.25)
        self.assertEqual(result.provider_turn_id, "turn-1")
        self.assertEqual(result.provider_turn_state, "sticky-state")
        self.assertEqual(result.provider_response_id, "response-compact-1")
        self.assertEqual(result.request_attempts, 1)
        self.assertEqual(result.recovery, ())
        checkpoint = result.items[0]
        self.assertIsInstance(checkpoint, ContextPrefix)
        self.assertEqual(checkpoint.prefix_items, (
            Instructions("Keep these instructions."),
            Message("user", "First request."),
            Message("user", "Latest request."),
            OpaqueCompaction.from_responses("new-encrypted-checkpoint"),
        ))
        payload = _request_payload(opener)
        self.assertEqual(payload["input"][0]["role"], "developer")
        self.assertEqual(payload["input"][0]["content"][0]["text"], "Keep these instructions.")
        self.assertEqual(payload["input"][-1], {"type": "compaction_trigger"})
        self.assertEqual(
            sum(item.get("type") == "compaction_trigger" for item in payload["input"]),
            1,
        )
        self.assertEqual([item["name"] for item in payload["tools"]], ["lookup"])
        headers = _request_headers(opener)
        self.assertEqual(headers["x-codex-beta-features"], "remote_compaction_v2")
        self.assertEqual(headers["x-codex-turn-state"], "sticky-state")
        self.assertEqual(headers["session_id"], "session-1")
        self.assertEqual(
            json.loads(headers["x-codex-turn-metadata"])["turn_id"],
            "turn-1",
        )
        self.assertTrue(response.closed)
        self.assertNotIn("compaction_trigger", repr(result))
        for private in (
            "turn-1",
            "sticky-state",
            "response-compact-1",
        ):
            self.assertNotIn(private, repr(result))
        self.assertFalse(any(isinstance(item, ToolCall) for item in result.items))
        self.assertEqual(
            result.context_items()[-1],
            CompactionMetadata(
                usage=TokenUsage(120, 9, 129, 80),
                protocol="responses_compaction_v2",
                provider_turn_id="turn-1",
                provider_turn_state="sticky-state",
                provider_response_id="response-compact-1",
                elapsed_seconds=86.25,
            ),
        )

    def test_remote_v2_compaction_retains_only_newest_real_user_messages(self):
        opener = _ScriptedOpener(_FakeSSEResponse(
            _compaction_event(None),
            _completed_event(),
        ))
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
        )
        context = InteractionContext((
            Message("user", "old-user"),
            Message("user", f"{_LEGACY_SUMMARY_PREFIX}\nold local summary"),
            Message("user", f"{COMPACTION_SUMMARY_PREFIX}pi summary{COMPACTION_SUMMARY_SUFFIX}"),
            Message("assistant", "old assistant output"),
            Message("user", "new-user"),
        ))

        result = ResponsesOpaqueCompactor(
            model,
            retained_user_message_tokens=2,
        ).compact(context)

        checkpoint = result.items[0]
        self.assertEqual(checkpoint.prefix_items, (
            Message("user", "new-user"),
            OpaqueCompaction.from_responses("encrypted-checkpoint"),
        ))
        # Retention affects only the client-built prefix; the server still
        # receives the complete effective source context before the trigger.
        request_text = json.dumps(_request_payload(opener)["input"])
        self.assertIn("old-user", request_text)
        self.assertIn("old local summary", request_text)
        self.assertIn("pi summary", request_text)
        # Neither summary style is retained as a user message.
        retained = ResponsesOpaqueCompactor(model)._is_retained_user_message
        self.assertFalse(retained(context[1]))
        self.assertFalse(retained(context[2]))
        self.assertTrue(retained(context[0]))

    def test_extra_sample_params_extend_samples_and_remote_v2_compaction(self):
        binding = BUILTIN_MODEL_CATALOG.bind(
            "codex", "codex-test",
            endpoint_url=CODEX_RESPONSES_API_URL + "/responses",
            endpoint_auth="supplied",
            extra_sample_params={"service_tier": "priority"},
        )
        opener = _ScriptedOpener(
            _FakeSSEResponse(_message_event(0, "OK"), _completed_event()),
            _FakeSSEResponse(_compaction_event(0), _completed_event()),
        )
        model = codex_model(
            StreamingResponsesEndpoint(binding=binding, bearer_token="token"),
            opener=opener,
        )
        self.assertTrue(model.supports_remote_compaction)
        context = InteractionContext((Message("user", "hello"),))

        self.assertEqual(model.sample(context).last_assistant_text, "OK")
        ResponsesOpaqueCompactor(model).compact(context)

        sampled, compacted = _request_payload(opener, 0), _request_payload(opener, 1)
        for payload in (sampled, compacted):
            self.assertEqual(payload["service_tier"], "priority")
            self.assertIs(payload["store"], False)
        self.assertEqual(compacted["input"][-1], {"type": "compaction_trigger"})

    def test_remote_v2_compaction_metadata_records_transport_recovery(self):
        response = _FakeSSEResponse(
            _compaction_event(0),
            _completed_event(response_id="response-after-retry"),
        )
        opener = _ScriptedOpener(
            _http_error(500, body=b'{"error":{"code":"temporary"}}'),
            response,
        )
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
            retry_sleep=lambda _delay: None,
        )

        result = ResponsesOpaqueCompactor(model).compact(
            InteractionContext((Message("user", "compact me"),))
        )

        metadata = result.context_items()[-1]
        self.assertIsInstance(metadata, CompactionMetadata)
        self.assertEqual(metadata.request_attempts, 2)
        self.assertEqual(metadata.recovery, ("http_500_retry",))
        self.assertEqual(metadata.provider_response_id, "response-after-retry")
        self.assertEqual(len(opener.calls), 2)
        self.assertTrue(response.closed)

    def test_remote_v2_compaction_retries_interrupted_stream(self):
        discarded = _FailingSSEResponse(
            _compaction_event(0, "discarded-checkpoint"),
            failure=OSError("stream reset"),
        )
        accepted = _FakeSSEResponse(
            _compaction_event(0, "accepted-checkpoint"),
            _completed_event(response_id="accepted-response"),
        )
        opener = _ScriptedOpener(discarded, accepted)
        sleeps = []
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
            retry_sleep=sleeps.append,
        )

        result = ResponsesOpaqueCompactor(model).compact(
            InteractionContext((Message("user", "compact me"),))
        )

        self.assertEqual(
            result.items[0].prefix_items[-1],
            OpaqueCompaction.from_responses("accepted-checkpoint"),
        )
        self.assertEqual(result.request_attempts, 2)
        self.assertEqual(result.recovery, ("stream_transport_retry",))
        self.assertEqual(result.provider_response_id, "accepted-response")
        self.assertEqual(sleeps, [0.25])
        self.assertTrue(discarded.closed)
        self.assertTrue(accepted.closed)

    def test_remote_v2_compaction_requires_one_checkpoint_and_completion(self):
        cases = (
            (
                (_message_event(0, "unrelated"), _completed_event()),
                "exactly one opaque checkpoint",
            ),
            (
                (_compaction_event(0, "one"), _compaction_event(1, "two"),
                 _completed_event()),
                "exactly one opaque checkpoint",
            ),
            (
                (_compaction_event(0),),
                "closed before response.completed",
            ),
            (
                ({
                    "type": "response.output_item.done",
                    "item": {"type": "compaction", "encrypted_content": ""},
                }, _completed_event()),
                "must not be empty",
            ),
        )
        for payloads, message in cases:
            with self.subTest(message=message):
                attempts = 3 if message == "closed before response.completed" else 1
                model = codex_model(
                    responses_endpoint(
                        api_url=CODEX_RESPONSES_API_URL,
                        model="codex-test",
                        bearer_token="token",
                        api_provider="codex",
                    ),
                    opener=_ScriptedOpener(*(
                        _FakeSSEResponse(*payloads)
                        for _ in range(attempts)
                    )),
                    retry_sleep=lambda _delay: None,
                )
                with self.assertRaisesRegex(ModelResponseError, message):
                    ResponsesOpaqueCompactor(model).compact(
                        InteractionContext((Message("user", "compact me"),))
                    )

    def test_default_compactor_uses_remote_only_for_known_codex_route(self):
        official = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            )
        )
        custom = codex_model(
            responses_endpoint(
                api_url="https://example.test/v1",
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            )
        )
        meta = codex_model(
            responses_endpoint(
                api_url=META_RESPONSES_API_URL,
                model="muse-spark-1.3",
                bearer_token="token",
                api_provider="codex",
            )
        )

        self.assertEqual(
            REMOTE_COMPACTION_V2_RETAINED_USER_MESSAGE_TOKENS,
            64_000,
        )
        self.assertTrue(official.supports_remote_compaction)
        self.assertFalse(custom.supports_remote_compaction)
        self.assertFalse(meta.supports_remote_compaction)
        provider = CompactionSettings(mode="provider")
        self.assertIsInstance(
            create_default_compactor(official, provider),
            ResponsesOpaqueCompactor,
        )
        # Pi mode, the default settings, uses pi even on the official route.
        for settings in (CompactionSettings(), CompactionSettings(mode="pi")):
            self.assertIsInstance(create_default_compactor(official, settings), PiCompactor)
        for model in (custom, meta):
            for settings in (CompactionSettings(), provider):
                with self.subTest(model=model.endpoint.url, mode=settings.mode):
                    self.assertIsInstance(
                        create_default_compactor(model, settings), PiCompactor,
                    )

    def test_remote_compactor_forwards_only_the_turn_extra_and_rejects_focus(self):
        opener = _ScriptedOpener(_FakeSSEResponse(_compaction_event(0), _completed_event()))
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
        )
        context = InteractionContext((Message("user", "compact me"),))
        compactor = ResponsesOpaqueCompactor(model)
        with self.assertRaisesRegex(CompactionError, "focus text"):
            compactor.compact(context, instructions="keep paths")
        self.assertEqual(opener.calls, [])
        compactor.compact(
            context,
            sample_params=SampleParams(
                max_output_tokens=77,
                enable_auto_compaction=True,
                auto_compact_tokens=100,
                extra={"service_tier": "flex"},
            ),
            instructions="   ",
        )
        payload = _request_payload(opener)
        self.assertEqual(payload["service_tier"], "flex")
        self.assertNotIn("max_output_tokens", payload)
        self.assertEqual(payload["input"][-1], {"type": "compaction_trigger"})

    def test_pi_mode_on_the_codex_route_keeps_session_and_turn_headers(self):
        opener = _ScriptedOpener(_FakeSSEResponse(_message_event(0, "Summary."), _completed_event()))
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=lambda: "unexpected-new-identifier",
        )
        context = InteractionContext((
            Init("session-1"),
            Message("user", "request"),
            UserInteractionBoundary(),
            Message("assistant", "answer"),
            SampleMetadata(
                TokenUsage(total_tokens=10),
                provider_turn_id="turn-1",
                provider_turn_state="state-1",
            ),
            ModelSampleBoundary(),
        ))

        result = create_default_compactor(model, CompactionSettings(keep_recent_tokens=0)).compact(
            context, sample_params=SampleParams(),
        )

        headers = _request_headers(opener)
        payload = _request_payload(opener)
        self.assertEqual(headers["session_id"], "session-1")
        self.assertEqual(json.loads(headers["x-codex-turn-metadata"])["turn_id"], "turn-1")
        self.assertEqual(headers[X_CODEX_TURN_STATE_HEADER], "state-1")
        self.assertEqual(payload["prompt_cache_key"], "session-1")
        self.assertEqual(payload["tools"], [])
        self.assertEqual([item["role"] for item in payload["input"]], ["developer", "user"])
        self.assertNotIn("x-codex-beta-features", headers)
        self.assertEqual(result.protocol, "pi")

    def test_remote_compactor_rejects_pending_calls_without_network(self):
        opener = _ScriptedOpener()
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
        )
        with self.assertRaisesRegex(CompactionError, "unresolved tool calls"):
            ResponsesOpaqueCompactor(model).compact(InteractionContext((
                ToolCall("lookup", "pending", "{}"),
            )))
        self.assertEqual(opener.calls, [])

    def test_compaction_metadata_can_preserve_codex_turn_continuity(self):
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            identifier_factory=lambda: "unexpected-new-turn",
        )
        metadata = CompactionMetadata(
            TokenUsage(),
            "responses_compaction_v2",
            provider_turn_id="compact-turn",
            provider_turn_state="compact-state",
        )
        context = InteractionContext((
            Init("session"),
            Message("user", "request"),
            UserInteractionBoundary(),
            metadata,
        ))

        payload, state = model._build_request_payload(context, (), None)

        self.assertEqual(state.turn_id, "compact-turn")
        self.assertEqual(state.turn_state, "compact-state")
        self.assertNotIn("compaction_metadata", json.dumps(payload))

    def test_init_prefix_id_owns_codex_session_and_prompt_cache_key(self):
        opener = _ScriptedOpener(
            _FakeSSEResponse(
                _message_event(0, "done"),
                _completed_event(),
            )
        )

        turn_identifiers = iter(("turn-1",))
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=lambda: next(turn_identifiers),
        )
        context = InteractionContext(
            (
                Init(prefix_id="session-from-context"),
                Message(role="user", content="hello"),
            )
        )

        sample = model.sample(context)

        payload = _request_payload(opener)
        headers = _request_headers(opener)
        self.assertEqual(
            payload["prompt_cache_key"],
            "session-from-context",
        )
        self.assertEqual(headers["session_id"], "session-from-context")
        self.assertIsNone(sample.provider_session_id)
        metadata = sample.context_items()[-2]
        self.assertIsNone(metadata.provider_session_id)

    def test_codex_reasoning_model_aliases_set_base_model_and_effort(self):
        cases = (
            ("codex-gpt-5.6-sol-medium", "gpt-5.6-sol", "medium"),
            ("codex-gpt-5.6-sol-max", "gpt-5.6-sol", "max"),
            ("codex-gpt-5.6-sol", "gpt-5.6-sol", None),
            ("codex-gpt-5.6-sol-high", "codex-gpt-5.6-sol-high", None),
        )
        for requested_model, expected_model, expected_effort in cases:
            with self.subTest(model=requested_model):
                opener = _ScriptedOpener(
                    _FakeSSEResponse(
                        _message_event(0, "done"),
                        _completed_event(),
                    )
                )
                model = codex_model(
                    responses_endpoint(
                        api_url=CODEX_RESPONSES_API_URL,
                        model=requested_model,
                        bearer_token="token",
                        api_provider="codex",
                    ),
                    opener=opener,
                )

                model.sample(
                    InteractionContext([Message(role="user", content="hello")])
                )

                payload = _request_payload(opener)
                self.assertEqual(payload["model"], expected_model)
                self.assertNotIn("auto_compact_context_tokens", payload)
                self.assertNotIn("max_context_tokens", payload)
                if expected_effort is None:
                    self.assertNotIn("reasoning", payload)
                else:
                    self.assertEqual(
                        payload["reasoning"],
                        {"effort": expected_effort},
                    )

    def test_astra_models_set_reasoning_and_low_verbosity(self):
        cases = (
            ("codex-gpt-6-astra", {"summary": "auto"}),
            ("codex-gpt-6-astra-medium", {"effort": "medium", "summary": "auto"}),
            ("codex-gpt-6-astra-max", {"effort": "max", "summary": "auto"}),
        )
        for requested_model, expected_reasoning in cases:
            with self.subTest(model=requested_model):
                opener = _ScriptedOpener(
                    _FakeSSEResponse(
                        _message_event(0, "done"),
                        _completed_event(),
                    )
                )
                model = codex_model(
                    model=requested_model,
                    auth=CodexAuth(access_token="token"),
                    opener=opener,
                )

                model.sample(
                    InteractionContext([Message(role="user", content="hello")])
                )

                request, _ = opener.calls[0]
                payload = _request_payload(opener)
                self.assertEqual(
                    request.full_url,
                    f"{CODEX_RESPONSES_API_URL}/responses",
                )
                self.assertEqual(payload["model"], "gpt-6-astra")
                self.assertEqual(payload["reasoning"], expected_reasoning)
                self.assertEqual(payload["text"], {"verbosity": "low"})
                self.assertNotIn("auto_compact_context_tokens", payload)
                self.assertNotIn("max_context_tokens", payload)

    def test_reasoning_model_aliases_are_not_applied_to_generic_responses(self):
        for requested_model in (
            "codex-gpt-5.6-sol-max",
            "codex-gpt-6-astra",
            "codex-gpt-6-astra-medium",
            "codex-gpt-6-astra-max",
        ):
            with self.subTest(model=requested_model):
                opener = _ScriptedOpener(
                    _FakeSSEResponse(
                        _message_event(0, "done"),
                        _completed_event(),
                    )
                )
                model = codex_model(
                    responses_endpoint(
                        api_url="https://api.example.test/v1",
                        model=requested_model,
                        bearer_token="api-key",
                        api_provider="api",
                    ),
                    opener=opener,
                )

                model.sample(
                    InteractionContext([Message(role="user", content="hello")])
                )

                payload = _request_payload(opener)
                self.assertEqual(payload["model"], requested_model)
                self.assertNotIn("reasoning", payload)
                self.assertNotIn("text", payload)

    def test_muse_models_route_to_contributor_model_and_reasoning(self):
        cases = (
            ("muse-spark-1.3", None),
            ("muse-spark-1.3-xhigh", "xhigh"),
        )
        for requested_model, expected_effort in cases:
            with self.subTest(model=requested_model):
                opener = _ScriptedOpener(
                    _FakeSSEResponse(
                        _message_event(0, "done"),
                        _completed_event(),
                    )
                )
                model = codex_model(
                    model=requested_model,
                    auth=CodexAuth(access_token="meta-api-key"),
                    opener=opener,
                    identifier_factory=(
                        iter(("session-1", "turn-1")).__next__
                    ),
                )

                model.sample(
                    InteractionContext([Message(role="user", content="hello")])
                )

                request, _ = opener.calls[0]
                payload = _request_payload(opener)
                self.assertEqual(
                    request.full_url,
                    "https://api.meta.ai/v1/responses",
                )
                self.assertEqual(
                    payload["model"],
                    "muse-spark-1.3-contributor",
                )
                if expected_effort is None:
                    self.assertNotIn("reasoning", payload)
                else:
                    self.assertEqual(
                        payload["reasoning"],
                        {"effort": expected_effort},
                    )

    def test_sample_encodes_request_and_collects_ordered_output_items(self):
        response = _FakeSSEResponse(
            _tool_call_event(2),
            _reasoning_event(
                0,
                text="Need repository facts.",
                summary="Inspect the repository.",
                encrypted_content="encrypted-reasoning",
            ),
            _message_event(1, "I will inspect it."),
            _completed_event(
                input_tokens=30,
                output_tokens=8,
                total_tokens=38,
                cached_tokens=6,
            ),
            headers={"x-codex-turn-state": "turn-state-1"},
            crlf=True,
        )
        opener = _ScriptedOpener(response)
        identifiers = iter(("session-1", "turn-1"))
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="secret-token",
                account_id="account-1",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=lambda: next(identifiers),
        )
        context = InteractionContext(
            UserInteraction(
                items=(Message(role="user", content="Summarize this repo."),)
            ).context_items()
        )
        before = context.items
        tool = ToolSpec(
            name="lookup",
            description="Look up repository facts.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        )

        sample = model.sample(
            context,
            tools=(tool,),
            sample_params=SampleParams(max_output_tokens=200),
        )

        self.assertEqual(context.items, before)
        self.assertEqual(
            sample.items,
            (
                Reasoning(
                    content="Need repository facts.",
                    summary=("Inspect the repository.",),
                    encrypted_content="encrypted-reasoning",
                ),
                Message(role="assistant", content="I will inspect it."),
                ToolCall(
                    name="lookup",
                    call_id="call-1",
                    arguments_json='{"query":"pythia"}',
                ),
            ),
        )
        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(sample.usage.input_tokens, 30)
        self.assertEqual(sample.usage.output_tokens, 8)
        self.assertEqual(sample.usage.total_tokens, 38)
        self.assertEqual(sample.usage.cached_input_tokens, 6)
        self.assertEqual(sample.provider_session_id, "session-1")
        self.assertEqual(sample.provider_turn_id, "turn-1")
        self.assertEqual(sample.provider_turn_state, "turn-state-1")
        metadata = sample.context_items()[-2]
        self.assertIsInstance(metadata, SampleMetadata)
        self.assertEqual(metadata.provider_session_id, "session-1")
        self.assertEqual(metadata.provider_turn_id, "turn-1")
        self.assertEqual(metadata.provider_turn_state, "turn-state-1")
        self.assertNotIn("turn-state-1", repr(sample))
        self.assertNotIn("turn-state-1", repr(metadata))
        self.assertTrue(response.closed)

        request, timeout = opener.calls[0]
        payload = _request_payload(opener)
        headers = _request_headers(opener)
        self.assertEqual(timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(
            request.full_url,
            f"{CODEX_RESPONSES_API_URL}/responses",
        )
        self.assertEqual(headers["user-agent"], USER_AGENT)
        self.assertEqual(headers["authorization"], "Bearer secret-token")
        self.assertEqual(headers["chatgpt-account-id"], "account-1")
        self.assertEqual(headers["session_id"], "session-1")
        self.assertNotIn("x-codex-turn-state", headers)
        self.assertEqual(
            json.loads(headers["x-codex-turn-metadata"]),
            {
                "turn_id": "turn-1",
                "sandbox": "none",
            },
        )
        self.assertEqual(payload["model"], "codex-test")
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["store"])
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(
            payload["include"],
            ["reasoning.encrypted_content"],
        )
        self.assertEqual(payload["prompt_cache_key"], "session-1")
        self.assertEqual(payload["max_output_tokens"], 200)
        self.assertEqual(
            payload["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Summarize this repo.",
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            payload["tools"],
            [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up repository facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                    "strict": False,
                }
            ],
        )

    def test_sol_medium_experiment_encodes_user_message_and_preserves_turn(self):
        tool = create_inject_user_message_tool()
        opener = _ScriptedOpener(
            _FakeSSEResponse(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "name": tool.spec.name,
                        "call_id": "inject-1",
                        "arguments": "{}",
                    },
                },
                _completed_event(),
                headers={"x-codex-turn-state": "sticky-state"},
            ),
            _FakeSSEResponse(
                _message_event(0, "received: hello world"), _completed_event(),
            ),
        )
        model = codex_model(
            model="codex-gpt-5.6-sol-medium",
            auth=CodexAuth(access_token="test-token"),
            opener=opener,
            # A second generated turn ID would exhaust the iterator.
            identifier_factory=iter(("turn-1",)).__next__,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.jsonl"
            with mock.patch("builtins.print"):
                answer = run(
                    model, Environment((tool,)),
                    prompt=EXPERIMENTAL_USER_MESSAGE_PROMPT,
                    max_samples=2, save_path=path,
                )
            restored = load_interaction_save(path)
        self.assertEqual(answer, "received: hello world")
        self.assertEqual(len(opener.calls), 2)
        for index in (0, 1):
            payload = _request_payload(opener, index)
            self.assertEqual(payload["model"], "gpt-5.6-sol")
            self.assertEqual(payload["reasoning"], {"effort": "medium"})
            self.assertEqual(payload["tool_choice"], "auto")
            self.assertEqual(tuple(spec["name"] for spec in payload["tools"]), (tool.spec.name,))
        self.assertNotIn("hello world", json.dumps(_request_payload(opener, 0)))
        self.assertEqual(_request_payload(opener, 1)["input"][-2:], [
            {
                "type": "function_call_output",
                "call_id": "inject-1",
                "output": "Synthetic user message queued.",
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello world"}],
            },
        ])
        first_headers = _request_headers(opener, 0)
        second_headers = _request_headers(opener, 1)
        self.assertEqual(second_headers["session_id"], first_headers["session_id"])
        self.assertEqual(second_headers["x-codex-turn-state"], "sticky-state")
        self.assertEqual(
            json.loads(second_headers["x-codex-turn-metadata"])["turn_id"], "turn-1",
        )
        self.assertEqual(restored.items.count(Message("user", "hello world")), 1)
        self.assertEqual(restored.items.count(UserInteractionBoundary()), 1)

    def test_context_owns_sticky_turn_state_and_resets_it_for_new_user_turn(self):
        first_response = _FakeSSEResponse(
            _reasoning_event(
                0,
                text="Use the tool.",
                summary="Tool needed.",
                encrypted_content="encrypted-1",
            ),
            _tool_call_event(1),
            _completed_event(),
            headers={"x-codex-turn-state": "sticky-state"},
        )
        second_response = _FakeSSEResponse(
            _message_event(0, "First turn complete."),
            _completed_event(),
            headers={"x-codex-turn-state": "replacement-state"},
        )
        third_response = _FakeSSEResponse(
            _message_event(0, "Second turn complete."),
            _completed_event(),
            headers={"x-codex-turn-state": "second-turn-state"},
        )
        opener = _ScriptedOpener(
            first_response,
            second_response,
            third_response,
        )
        identifiers = iter(("session-1", "turn-1", "turn-2"))
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=lambda: next(identifiers),
        )
        context = InteractionContext(
            UserInteraction(
                items=(Message(role="user", content="First turn"),)
            ).context_items()
        )

        first = model.sample(context)
        context.extend(first.context_items())
        context.append(
            ToolResult(
                call_id="call-1",
                output="repository facts",
            )
        )
        second = model.sample(context)
        context.extend(second.context_items())
        context.extend(
            UserInteraction(
                items=(Message(role="user", content="Second turn"),)
            ).context_items()
        )
        third = model.sample(context)

        second_headers = _request_headers(opener, 1)
        self.assertEqual(second_headers["session_id"], "session-1")
        self.assertEqual(
            second_headers["x-codex-turn-state"],
            "sticky-state",
        )
        self.assertEqual(
            json.loads(second_headers["x-codex-turn-metadata"])["turn_id"],
            "turn-1",
        )
        second_input = _request_payload(opener, 1)["input"]
        self.assertIn(
            {
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "Tool needed.",
                    }
                ],
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "Use the tool.",
                    }
                ],
                "encrypted_content": "encrypted-1",
            },
            second_input,
        )
        self.assertEqual(
            second_input[-1],
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "repository facts",
            },
        )
        self.assertEqual(second.provider_turn_state, "sticky-state")

        third_headers = _request_headers(opener, 2)
        self.assertEqual(third_headers["session_id"], "session-1")
        self.assertNotIn("x-codex-turn-state", third_headers)
        self.assertEqual(
            json.loads(third_headers["x-codex-turn-metadata"])["turn_id"],
            "turn-2",
        )
        self.assertEqual(third.provider_session_id, "session-1")
        self.assertEqual(third.provider_turn_id, "turn-2")
        self.assertEqual(
            third.provider_turn_state,
            "second-turn-state",
        )

    def test_generic_responses_endpoint_omits_codex_state(self):
        response = _FakeSSEResponse(
            _message_event(0, "generic response"),
            _completed_event(),
        )
        opener = _ScriptedOpener(response)

        def unexpected_identifier():
            raise AssertionError("generic Responses must not create Codex IDs")

        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=opener,
            identifier_factory=unexpected_identifier,
        )

        sample = model.sample(
            InteractionContext([Message(role="user", content="hello")])
        )

        headers = _request_headers(opener)
        payload = _request_payload(opener)
        self.assertNotIn("chatgpt-account-id", headers)
        self.assertNotIn("session_id", headers)
        self.assertNotIn("x-codex-turn-metadata", headers)
        self.assertNotIn("x-codex-turn-state", headers)
        self.assertNotIn("prompt_cache_key", payload)
        self.assertIsNone(sample.provider_session_id)
        self.assertIsNone(sample.provider_turn_id)
        self.assertIsNone(sample.provider_turn_state)
        self.assertEqual(sample.last_assistant_text, "generic response")

    def test_session_resume_replays_state_after_pending_tool_call(self):
        first_response = _FakeSSEResponse(
            _tool_call_event(0),
            _completed_event(),
            headers={"x-codex-turn-state": "resume-state"},
        )
        endpoint = responses_endpoint(
            api_url=CODEX_RESPONSES_API_URL,
            model="codex-test",
            bearer_token="token",
            api_provider="codex",
        )
        identifiers = iter(("session-1", "turn-1"))
        first_model = codex_model(
            endpoint,
            opener=_ScriptedOpener(first_response),
            identifier_factory=lambda: next(identifiers),
        )
        context = InteractionContext(
            UserInteraction(
                items=(Message(role="user", content="Use the tool"),)
            ).context_items()
        )
        first_sample = first_model.sample(context)
        context.extend(first_sample.context_items())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_save(save_path, context)
            resumed = load_interaction_save(save_path)

        resumed.append(
            ToolResult(
                call_id="call-1",
                output="resumed result",
            )
        )
        second_response = _FakeSSEResponse(
            _message_event(0, "resumed answer"),
            _completed_event(),
        )
        opener = _ScriptedOpener(second_response)

        def unexpected_identifier():
            raise AssertionError("resumed context must supply provider IDs")

        second_model = codex_model(
            endpoint,
            opener=opener,
            identifier_factory=unexpected_identifier,
        )
        second_sample = second_model.sample(resumed)

        headers = _request_headers(opener)
        self.assertEqual(headers["session_id"], "session-1")
        self.assertEqual(
            headers["x-codex-turn-state"],
            "resume-state",
        )
        self.assertEqual(
            json.loads(headers["x-codex-turn-metadata"])["turn_id"],
            "turn-1",
        )
        self.assertEqual(
            second_sample.last_assistant_text,
            "resumed answer",
        )

    def test_named_multiline_sse_event_is_supported(self):
        item = _message_event(0, "multiline")
        completed = _completed_event()
        item_json = json.dumps(item, separators=(",", ":"))
        split_at = item_json.index('"item"')
        lines = [
            b"event: response.output_item.done\r\n",
            f"data: {item_json[:split_at]}\r\n".encode("utf-8"),
            f"data: {item_json[split_at:]}\r\n".encode("utf-8"),
            b"\r\n",
            b": keepalive\r\n",
            *_event_lines(completed, crlf=True),
        ]
        response = _FakeSSEResponse()
        response._lines = lines
        opener = _ScriptedOpener(response)
        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=opener,
        )

        sample = model.sample(
            InteractionContext([Message(role="user", content="hello")])
        )

        self.assertEqual(sample.last_assistant_text, "multiline")

    def test_output_items_without_indices_preserve_event_order(self):
        response = _FakeSSEResponse(
            _message_event(None, "first"),
            _message_event(None, "second"),
            _completed_event(),
        )
        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=_ScriptedOpener(response),
        )

        sample = model.sample(
            InteractionContext([Message(role="user", content="hello")])
        )

        self.assertEqual(
            sample.items,
            (
                Message(role="assistant", content="first"),
                Message(role="assistant", content="second"),
            ),
        )

    def test_errors_are_typed_and_unsupported_options_are_rejected(self):
        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=_ScriptedOpener(),
        )
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "^Responses does not support these sampling options: seed$",
        ):
            model.sample(
                InteractionContext([Message(role="user", content="hello")]),
                sample_params=SampleParams(seed=7),
            )

        partial_message = _message_event(0, "partial")
        partial_message["sequence_number"] = 2
        incomplete_response = _FakeSSEResponse(
            {
                "type": "response.created",
                "sequence_number": 1,
                "response": {"id": "response-partial"},
            },
            partial_message,
            headers={
                "x-request-id": "request-partial",
                "cf-ray": "ray-partial",
            },
        )
        with self.assertRaisesRegex(ModelResponseError, "before") as raised:
            codex_model(
                model.endpoint,
                opener=_ScriptedOpener(
                    incomplete_response,
                    incomplete_response,
                    incomplete_response,
                ),
                retry_sleep=lambda _delay: None,
            ).sample(
                InteractionContext([Message(role="user", content="hello")])
            )
        self.assertEqual(
            raised.exception.completed_items,
            (Message(role="assistant", content="partial"),),
        )
        failure = raised.exception.failure
        self.assertIsNotNone(failure)
        self.assertEqual(failure.category, "stream_closed")
        self.assertEqual(failure.attempt_count, 3)
        self.assertEqual(
            failure.recovery,
            ("stream_closed_retry", "stream_closed_retry"),
        )
        self.assertEqual(failure.event_count, 2)
        self.assertEqual(
            failure.event_types,
            ("response.created:1", "response.output_item.done:1"),
        )
        self.assertEqual(failure.completed_item_count, 1)
        self.assertEqual(failure.last_event_type, "response.output_item.done")
        self.assertEqual(failure.last_sequence_number, 2)
        self.assertEqual(failure.response_id, "response-partial")
        self.assertEqual(failure.request_id, "request-partial")
        self.assertEqual(failure.cf_ray, "ray-partial")
        self.assertTrue(incomplete_response.closed)

        unsupported_response = _FakeSSEResponse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "custom_tool_call",
                    "name": "custom",
                },
            },
            _completed_event(),
        )
        with self.assertRaisesRegex(ModelResponseError, "unsupported"):
            codex_model(
                model.endpoint,
                opener=_ScriptedOpener(unsupported_response),
            ).sample(
                InteractionContext([Message(role="user", content="hello")])
            )

        malformed_response = _FakeSSEResponse()
        malformed_response._lines = [b"data: {\n", b"\n"]
        with self.assertRaisesRegex(ModelResponseError, "invalid JSON"):
            codex_model(
                model.endpoint,
                opener=_ScriptedOpener(malformed_response),
            ).sample(
                InteractionContext([Message(role="user", content="hello")])
            )

    def test_codex_401_is_actionable_and_does_not_expose_token(self):
        def error():
            return urllib.error.HTTPError(
                CODEX_RESPONSES_API_URL,
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"error":{"message":"expired"}}'),
            )
        opener = _ScriptedOpener(error(), error())
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="secret-token",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=iter(("session-1", "turn-1")).__next__,
            retry_sleep=lambda _delay: None,
        )

        with self.assertRaisesRegex(
            ModelTransportError,
            "codex login",
        ) as raised:
            model.sample(
                InteractionContext([Message(role="user", content="hello")])
            )

        self.assertNotIn("secret-token", str(raised.exception))
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(raised.exception.failure.attempt_count, 2)
        self.assertEqual(
            raised.exception.failure.recovery,
            ("http_401_retry",),
        )

    def test_generic_responses_401_is_not_retried(self):
        opener = _ScriptedOpener(_http_error(401))
        sleeper = mock.Mock(
            side_effect=AssertionError("generic 401 must not retry")
        )
        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=opener,
            retry_sleep=sleeper,
        )

        with self.assertRaises(ModelAuthenticationError) as raised:
            model.sample(InteractionContext((Message("user", "hello"),)))

        self.assertEqual(len(opener.calls), 1)
        sleeper.assert_not_called()
        self.assertEqual(raised.exception.failure.attempt_count, 1)
        self.assertEqual(raised.exception.failure.recovery, ())

    def test_codex_401_reloads_changed_auth_file_before_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"

            def write(token):
                auth_file.write_text(
                    json.dumps(
                        {
                            "auth_mode": "chatgpt",
                            "tokens": {
                                "access_token": token,
                                "account_id": "account-1",
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            write("old-token")
            response = _FakeSSEResponse(
                _message_event(0, "recovered"),
                _completed_event(),
            )

            def first(request, *, timeout):
                del request, timeout
                write("new-token")
                raise _http_error(401)

            opener = _ScriptedOpener(first, response)
            model = codex_model(
                model="codex-test",
                auth_file=auth_file,
                opener=opener,
            )
            sample = model.sample(
                InteractionContext([Message(role="user", content="hello")])
            )

        self.assertEqual(sample.last_assistant_text, "recovered")
        self.assertEqual(sample.request_attempts, 2)
        self.assertEqual(sample.recovery, ("credential_reload",))
        self.assertEqual(_request_headers(opener, 0)["authorization"], "Bearer old-token")
        self.assertEqual(_request_headers(opener, 1)["authorization"], "Bearer new-token")

    def test_spurious_401_retries_unchanged_auth_file_before_refresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps({
                    "auth_mode": "chatgpt",
                    "tokens": {
                        "access_token": "valid-token",
                        "refresh_token": "unused-refresh-token",
                        "id_token": _account_id_token("account-1"),
                        "account_id": "account-1",
                    },
                }),
                encoding="utf-8",
            )
            opener = _ScriptedOpener(
                _http_error(401, headers={"retry-after": "0.75"}),
                _FakeSSEResponse(
                    _message_event(0, "accepted unchanged"),
                    _completed_event(),
                ),
            )
            auth_opener = mock.Mock(
                side_effect=AssertionError("spurious 401 must not refresh")
            )
            sleeps = []
            model = codex_model(
                model="codex-test",
                auth_file=auth_file,
                opener=opener,
                auth_opener=auth_opener,
                retry_sleep=sleeps.append,
            )

            sample = model.sample(
                InteractionContext([Message(role="user", content="hello")])
            )

        self.assertEqual(sample.last_assistant_text, "accepted unchanged")
        self.assertEqual(sample.request_attempts, 2)
        self.assertEqual(
            sample.recovery,
            ("credential_reload_unchanged", "http_401_retry"),
        )
        self.assertEqual(sleeps, [0.75])
        auth_opener.assert_not_called()
        self.assertEqual(
            [_request_headers(opener, i)["authorization"] for i in range(2)],
            ["Bearer valid-token", "Bearer valid-token"],
        )

    def test_codex_401_refreshes_oauth_token_and_persists_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "unrelated": "preserved",
                        "tokens": {
                            "access_token": "old-token",
                            "refresh_token": "old-refresh",
                            "id_token": _account_id_token("account-1"),
                            "account_id": "account-1",
                        },
                    }
                ),
                encoding="utf-8",
            )
            opener = _ScriptedOpener(
                _http_error(401),
                _http_error(401),
                _FakeSSEResponse(
                    _message_event(0, "refreshed"),
                    _completed_event(),
                ),
            )
            auth_opener = _ScriptedOpener(
                _FakeSSEResponse(
                    status=200,
                    body=json.dumps(
                        {
                            "access_token": "new-token",
                            "refresh_token": "new-refresh",
                            "id_token": _account_id_token("account-1"),
                        }
                    ).encode("utf-8"),
                )
            )
            model = codex_model(
                model="codex-test",
                auth_file=auth_file,
                opener=opener,
                auth_opener=auth_opener,
                retry_sleep=lambda _delay: None,
            )

            sample = model.sample(
                InteractionContext([Message(role="user", content="hello")])
            )
            saved = json.loads(auth_file.read_text(encoding="utf-8"))
            saved_mode = auth_file.stat().st_mode & 0o777

        self.assertEqual(sample.last_assistant_text, "refreshed")
        self.assertEqual(sample.request_attempts, 3)
        self.assertEqual(
            sample.recovery,
            (
                "credential_reload_unchanged",
                "http_401_retry",
                "oauth_refresh",
            ),
        )
        self.assertEqual(_request_headers(opener, 0)["authorization"], "Bearer old-token")
        self.assertEqual(_request_headers(opener, 1)["authorization"], "Bearer old-token")
        self.assertEqual(_request_headers(opener, 2)["authorization"], "Bearer new-token")
        refresh_request = json.loads(auth_opener.calls[0][0].data.decode("utf-8"))
        self.assertEqual(refresh_request["grant_type"], "refresh_token")
        self.assertEqual(refresh_request["refresh_token"], "old-refresh")
        self.assertEqual(saved["tokens"]["access_token"], "new-token")
        self.assertEqual(saved["tokens"]["refresh_token"], "new-refresh")
        self.assertEqual(saved["unrelated"], "preserved")
        if os.name == "posix":
            self.assertEqual(saved_mode, 0o600)

    def test_environment_credential_is_reloaded_after_401(self):
        response = _FakeSSEResponse(
            _message_event(0, "recovered"),
            _completed_event(),
        )
        calls = []

        def opener(request, *, timeout):
            calls.append((request, timeout))
            if len(calls) == 1:
                os.environ["META_API_KEY"] = "new-meta-key"
                raise _http_error(401)
            return response

        with mock.patch.dict(os.environ, {"META_API_KEY": "old-meta-key"}, clear=True):
            model = codex_model(
                model="muse-spark-1.3",
                opener=opener,
            )
            sample = model.sample(
                InteractionContext([Message(role="user", content="hello")])
            )

        headers = [
            {name.lower(): value for name, value in request.header_items()}
            for request, _ in calls
        ]
        self.assertEqual(sample.request_attempts, 2)
        self.assertEqual(sample.recovery, ("credential_reload",))
        self.assertEqual(headers[0]["authorization"], "Bearer old-meta-key")
        self.assertEqual(headers[1]["authorization"], "Bearer new-meta-key")

    def test_unchanged_environment_credential_does_not_loop_on_401(self):
        opener = _ScriptedOpener(
            _http_error(401),
            _http_error(401, headers={"x-request-id": "request-env"}),
        )
        with mock.patch.dict(os.environ, {"META_API_KEY": "meta-key"}, clear=True):
            model = codex_model(
                model="muse-spark-1.3",
                opener=opener,
                retry_sleep=lambda _delay: None,
            )
            with self.assertRaises(ModelAuthenticationError) as raised:
                model.sample(
                    InteractionContext([Message(role="user", content="hello")])
                )

        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(
            raised.exception.failure.recovery,
            ("credential_reload_unchanged", "http_401_retry"),
        )
        self.assertEqual(raised.exception.failure.request_id, "request-env")
        self.assertIn("restart", str(raised.exception))

    def test_auth_file_account_change_is_rejected_without_refresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"

            def write(account, token):
                auth_file.write_text(
                    json.dumps(
                        {
                            "auth_mode": "chatgpt",
                            "tokens": {
                                "access_token": token,
                                "refresh_token": "refresh-secret",
                                "id_token": _account_id_token(account),
                                "account_id": account,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            write("account-1", "old-token")
            calls = []

            def opener(request, *, timeout):
                calls.append((request, timeout))
                write("account-2", "other-account-token")
                raise _http_error(401)

            auth_opener = mock.Mock(
                side_effect=AssertionError("refresh must not be attempted")
            )
            model = codex_model(
                model="codex-test",
                auth_file=auth_file,
                opener=opener,
                auth_opener=auth_opener,
            )
            with self.assertRaises(ModelAuthenticationError) as raised:
                model.sample(
                    InteractionContext([Message(role="user", content="hello")])
                )

            saved = json.loads(auth_file.read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        auth_opener.assert_not_called()
        self.assertEqual(saved["tokens"]["account_id"], "account-2")
        self.assertEqual(
            raised.exception.failure.recovery,
            ("credential_reload_rejected",),
        )

    def test_repeated_401_is_bounded_and_has_safe_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "tokens": {
                            "access_token": "FAKE_OLD_SECRET",
                            "refresh_token": "FAKE_REFRESH_SECRET",
                            "id_token": _account_id_token("account-1"),
                            "account_id": "account-1",
                        },
                    }
                ),
                encoding="utf-8",
            )
            error_json = base64.b64encode(
                b'{"error":{"code":"token_expired"}}'
            ).decode("ascii")
            opener = _ScriptedOpener(
                _http_error(401),
                _http_error(401),
                _http_error(
                    401,
                    body=b'{"error":{"message":"FAKE_BODY_SECRET"}}',
                    headers={
                        "x-oai-request-id": "request-final",
                        "cf-ray": "ray-final",
                        "x-openai-authorization-error": "expired_token",
                        "x-error-json": error_json,
                    },
                ),
            )
            auth_opener = _ScriptedOpener(
                _FakeSSEResponse(
                    status=200,
                    body=json.dumps(
                        {
                            "access_token": "FAKE_NEW_SECRET",
                            "id_token": _account_id_token("account-1"),
                        }
                    ).encode("utf-8"),
                )
            )
            model = codex_model(
                model="codex-test",
                auth_file=auth_file,
                opener=opener,
                auth_opener=auth_opener,
                retry_sleep=lambda _delay: None,
            )

            with self.assertRaises(ModelAuthenticationError) as raised:
                model.sample(
                    InteractionContext([Message(role="user", content="hello")])
                )

        self.assertEqual(len(opener.calls), 3)
        failure = raised.exception.failure
        self.assertEqual(failure.http_status, 401)
        self.assertEqual(failure.attempt_count, 3)
        self.assertEqual(failure.request_id, "request-final")
        self.assertEqual(failure.cf_ray, "ray-final")
        self.assertEqual(failure.authorization_error, "expired_token")
        self.assertEqual(failure.auth_error_code, "token_expired")
        self.assertEqual(
            failure.recovery,
            (
                "credential_reload_unchanged",
                "http_401_retry",
                "oauth_refresh",
            ),
        )
        rendered = f"{raised.exception!r}\n{raised.exception}\n{failure!r}"
        for secret in (
            "FAKE_OLD_SECRET",
            "FAKE_NEW_SECRET",
            "FAKE_REFRESH_SECRET",
            "FAKE_BODY_SECRET",
        ):
            self.assertNotIn(secret, rendered)

    def test_retryable_and_terminal_http_errors_have_diagnostics(self):
        recovered_opener = _ScriptedOpener(
            _http_error(503, headers={"x-request-id": "request-overload"}),
            _FakeSSEResponse(
                _message_event(0, "recovered"),
                _completed_event(),
            ),
        )
        recovered = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=recovered_opener,
            retry_sleep=lambda _delay: None,
        ).sample(InteractionContext([Message(role="user", content="hello")]))
        self.assertEqual(recovered.request_attempts, 2)
        self.assertEqual(recovered.recovery, ("http_503_retry",))

        terminal_opener = _ScriptedOpener(
            _http_error(
                400,
                body=b'{"error":{"code":"invalid_request","message":"FAKE_SECRET"}}',
                headers={
                    "x-request-id": "request-invalid",
                    "x-openai-authorization-error": "reflected-api-key",
                },
            )
        )
        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=terminal_opener,
        )
        with self.assertRaises(ModelTransportError) as raised:
            model.sample(InteractionContext([Message(role="user", content="hello")]))
        self.assertEqual(raised.exception.failure.http_status, 400)
        self.assertEqual(raised.exception.failure.request_id, "request-invalid")
        self.assertEqual(raised.exception.failure.error_code, "invalid_request")
        self.assertIsNone(raised.exception.failure.authorization_error)
        self.assertNotIn("FAKE_SECRET", str(raised.exception))

    def test_system_role_rejection_has_safe_actionable_diagnostics(self):
        for body in (
            {"detail": "System messages are not allowed", "debug": "FAKE_SECRET"},
            {"error": {"message": "System messages are not allowed", "debug": "FAKE_SECRET"}},
        ):
            with self.subTest(body=body):
                opener = _ScriptedOpener(_http_error(400, body=json.dumps(body).encode()))
                model = codex_model(responses_endpoint(
                    api_url=CODEX_RESPONSES_API_URL, model="codex-gpt-6-astra-max",
                    bearer_token="test-token", api_provider="codex",
                ), opener=opener)
                with self.assertRaises(ModelTransportError) as raised:
                    model.sample(InteractionContext((Message("user", "hello"),)))
                failure = raised.exception.failure
                self.assertIn("system input messages are not allowed; use developer messages", failure.message)
                self.assertEqual(failure.http_status, 400)
                self.assertEqual(failure.attempt_count, 1)
                self.assertEqual(len(opener.calls), 1)
                self.assertNotIn("FAKE_SECRET", repr(failure) + str(raised.exception))
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "failure.jsonl"
                    save_interaction_save(path, InteractionContext((failure,)))
                    self.assertEqual(load_interaction_save(path).items, (failure,))
                    self.assertNotIn("FAKE_SECRET", path.read_text())

    def test_arbitrary_http_error_detail_is_still_withheld(self):
        for detail in ("FAKE_SECRET", "System messages are not allowed: FAKE_SECRET"):
            with self.subTest(detail=detail):
                opener = _ScriptedOpener(_http_error(400, body=json.dumps({"detail": detail}).encode()))
                model = codex_model(responses_endpoint(
                    api_url=CODEX_RESPONSES_API_URL, model="codex-gpt-6-astra-max",
                    bearer_token="test-token", api_provider="codex",
                ), opener=opener)
                with self.assertRaises(ModelTransportError) as raised:
                    model.sample(InteractionContext((Message("user", "hello"),)))
                self.assertEqual(raised.exception.failure.message, "Codex Responses HTTP 400: request failed")
                self.assertNotIn("FAKE_SECRET", repr(raised.exception.failure) + str(raised.exception))

    def test_http_and_midstream_failures_share_two_retry_budget(self):
        partial = _FailingSSEResponse(
            _message_event(0, "discarded partial output"),
            failure=http.client.IncompleteRead(b"truncated stream"),
            headers={X_CODEX_TURN_STATE_HEADER: "retry-sticky-state"},
        )
        success = _FakeSSEResponse(
            _message_event(0, "final output"),
            _completed_event(),
        )
        opener = _ScriptedOpener(
            partial,
            _http_error(503, headers={"retry-after": "1.25"}),
            success,
        )
        sleeps = []
        model = codex_model(
            responses_endpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
            retry_sleep=sleeps.append,
        )

        sample = model.sample(
            InteractionContext((Init("session"), Message("user", "hello")))
        )

        self.assertEqual(sample.items, (Message("assistant", "final output"),))
        self.assertEqual(sample.request_attempts, 3)
        self.assertEqual(
            sample.recovery,
            ("stream_transport_retry", "http_503_retry"),
        )
        self.assertEqual(sample.provider_turn_state, "retry-sticky-state")
        self.assertEqual(sleeps, [0.25, 1.25])
        self.assertTrue(partial.closed)
        self.assertTrue(success.closed)
        self.assertNotIn(
            "x-codex-turn-state",
            _request_headers(opener, 0),
        )
        self.assertEqual(
            _request_headers(opener, 1)["x-codex-turn-state"],
            "retry-sticky-state",
        )
        self.assertEqual(
            tuple(_request_payload(opener, index) for index in range(3)),
            (_request_payload(opener, 0),) * 3,
        )

    def test_stream_progress_does_not_reset_retry_budget(self):
        responses = tuple(
            _FakeSSEResponse(_message_event(0, text))
            for text in ("discarded one", "discarded two", "d final")
        )
        opener = _ScriptedOpener(*responses)
        sleeps = []
        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=opener,
            retry_sleep=sleeps.append,
        )

        with self.assertRaises(ModelResponseError) as raised:
            model.sample(InteractionContext((Message("user", "hello"),)))

        self.assertEqual(len(opener.calls), 3)
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertEqual(
            raised.exception.completed_items,
            (Message("assistant", "d final"),),
        )
        self.assertEqual(raised.exception.failure.attempt_count, 3)
        self.assertEqual(
            raised.exception.failure.recovery,
            ("stream_closed_retry", "stream_closed_retry"),
        )
        self.assertTrue(all(response.closed for response in responses))

    def test_connection_and_timeout_failures_are_retried(self):
        opener = _ScriptedOpener(
            urllib.error.URLError(OSError("connection reset")),
            socket.timeout("read timed out"),
            _FakeSSEResponse(
                _message_event(0, "reconnected"),
                _completed_event(),
            ),
        )
        sleeps = []
        sample = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=opener,
            retry_sleep=sleeps.append,
        ).sample(InteractionContext((Message("user", "hello"),)))

        self.assertEqual(sample.last_assistant_text, "reconnected")
        self.assertEqual(sample.request_attempts, 3)
        self.assertEqual(
            sample.recovery,
            ("connection_retry", "request_timeout_retry"),
        )
        self.assertEqual(sleeps, [0.25, 0.5])

    def test_retryable_http_failure_stops_after_two_retries(self):
        opener = _ScriptedOpener(*(
            _http_error(503, headers={"x-request-id": f"request-{index}"})
            for index in range(1, 4)
        ))
        sleeps = []
        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=opener,
            retry_sleep=sleeps.append,
        )

        with self.assertRaises(ModelTransportError) as raised:
            model.sample(InteractionContext((Message("user", "hello"),)))

        self.assertEqual(len(opener.calls), 3)
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertEqual(raised.exception.failure.attempt_count, 3)
        self.assertEqual(raised.exception.failure.request_id, "request-3")
        self.assertEqual(
            raised.exception.failure.recovery,
            ("http_503_retry", "http_503_retry"),
        )


class DemoConfigurationTests(unittest.TestCase):
    def test_demo_max_samples_is_unbounded_by_default(self):
        args = _build_parser().parse_args([])

        self.assertIsNone(args.max_samples)
        self.assertEqual(
            _build_parser().parse_args(["--max-samples=25"]).max_samples,
            25,
        )

    def test_demo_defaults_to_chat_completions_and_repository_prompt(self):
        args = _build_parser().parse_args([])
        model = _build_model(args)

        self.assertIsInstance(model, ChatCompletionsModel)
        self.assertEqual(
            model.endpoint.url,
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(
            DEFAULT_PROMPT,
            "Summarize the repository in the current working directory.",
        )

    def test_demo_builds_optional_codex_responses_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "codex-token",
                            "account_id": "account-1",
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = _build_parser().parse_args(
                [
                    "--endpoint-api",
                    "codex",
                    "--model",
                    "codex-test",
                    "--endpoint-auth-file",
                    str(auth_file),
                ]
            )

            model = _build_model(args)

        self.assertIsInstance(model, CodexResponsesModel)
        self.assertEqual(model.endpoint.url, CODEX_RESPONSES_API_URL + "/responses")
        self.assertEqual(model.endpoint.model, "codex-test")
        self.assertEqual(model.endpoint.account_id, "account-1")
        self.assertNotIn("codex-token", repr(model.endpoint))

    def test_demo_accepts_codex_model_api_shorthand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "codex-token",
                            "account_id": "account-1",
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = _build_parser().parse_args(
                [
                    "--endpoint-api=codex",
                    "--model",
                    "codex-test",
                    "--endpoint-auth-file",
                    str(auth_file),
                ]
            )

            model = _build_model(args)

        self.assertIsInstance(model, CodexResponsesModel)
        self.assertEqual(model.endpoint.url, CODEX_RESPONSES_API_URL + "/responses")
        self.assertEqual(model.endpoint.model, "codex-test")
        self.assertEqual(model.endpoint.account_id, "account-1")

    def test_demo_builds_muse_model_from_meta_api_key(self):
        args = _build_parser().parse_args(
            [
                "--endpoint-api=codex",
                "--model=muse-spark-1.3-xhigh",
            ]
        )

        with mock.patch.dict(
            "os.environ",
            {"META_API_KEY": "meta-api-key"},
            clear=True,
        ):
            model = _build_model(args)

        self.assertIsInstance(model, CodexResponsesModel)
        self.assertEqual(model.endpoint.url, META_RESPONSES_API_URL + "/responses")
        self.assertEqual(model.endpoint.model, "muse-spark-1.3-contributor")
        self.assertEqual(model.endpoint.bearer_token, "meta-api-key")
        self.assertIsNone(model.endpoint.account_id)
        self.assertNotIn("meta-api-key", repr(model.endpoint))

    def test_demo_rejects_ambiguous_auth_options(self):
        with self.assertRaisesRegex(ValueError, "required"):
            _build_model(
                _build_parser().parse_args(
                    ["--endpoint-api", "codex"]
                )
            )
        with self.assertRaisesRegex(ValueError, "not used"):
            _build_model(
                _build_parser().parse_args(
                    [
                        "--endpoint-api",
                        "codex",
                        "--model",
                        "codex-test",
                        "--endpoint-api-key",
                        "token",
                    ]
                )
            )
        with self.assertRaisesRegex(ValueError, "require"):
            _build_model(
                _build_parser().parse_args(
                    ["--endpoint-auth-home", "/tmp/codex"]
                )
            )

    def test_demo_rejects_unsupported_model_api_namespace(self):
        args = _build_parser().parse_args([])
        args.model_api = "unsupported-api"

        with self.assertRaisesRegex(
            ValueError,
            "unsupported model API: 'unsupported-api'",
        ):
            _build_model(args)


class GenericResponsesWireTests(unittest.TestCase):
    """Public Responses request and response shapes; Codex requests unchanged."""

    TOOL = ToolSpec("lookup", "Look things up.", {"type": "object", "properties": {}})

    def _sample(self, api_provider, *outcomes, tools=(), sample_params=None):
        opener = _ScriptedOpener(*outcomes)
        model = codex_model(
            responses_endpoint(
                api_url="https://api.example.test/v1",
                model="wire-model",
                bearer_token="api-key",
                api_provider=api_provider,
            ),
            opener=opener,
            retry_sleep=lambda _delay: None,
        )
        sample = model.sample(
            InteractionContext([Message(role="user", content="hello")]),
            tools=tools,
            sample_params=sample_params,
        )
        return sample, opener

    def test_tool_fields_need_tools_except_on_codex(self):
        fields = {"tools", "tool_choice", "parallel_tool_calls"}
        for api_provider, tools, sent in (
            ("api", (), False), ("api", (self.TOOL,), True), ("codex", (), True),
        ):
            with self.subTest(api_provider=api_provider, tools=len(tools)):
                _, opener = self._sample(
                    api_provider,
                    _FakeSSEResponse(_message_event(0, "done"), _completed_event()),
                    tools=tools,
                )
                payload = _request_payload(opener)
                self.assertEqual(fields & payload.keys(), fields if sent else set())
                if sent:
                    self.assertEqual(
                        (payload["tool_choice"], payload["parallel_tool_calls"]),
                        ("auto", False),
                    )
                self.assertEqual(
                    (payload["store"], payload["stream"], payload["include"]),
                    (False, True, ["reasoning.encrypted_content"]),
                )

    def test_generic_route_sends_temperature_and_top_p_and_codex_rejects_them(self):
        _, opener = self._sample(
            "api",
            _FakeSSEResponse(_message_event(0, "done"), _completed_event()),
            sample_params=SampleParams(temperature=0.25, top_p=0.5, max_output_tokens=64),
        )
        payload = _request_payload(opener)
        self.assertEqual(
            (payload["temperature"], payload["top_p"], payload["max_output_tokens"]),
            (0.25, 0.5, 64),
        )
        for api_provider, params, message in (
            ("api", SampleParams(stop=("END",), temperature=0.25),
             "^Responses does not support these sampling options: stop$"),
            ("codex", SampleParams(temperature=0.25),
             "^Codex Responses does not support these sampling options yet: temperature$"),
        ):
            with self.subTest(api_provider=api_provider), \
                    self.assertRaisesRegex(ModelConfigurationError, message):
                self._sample(api_provider, sample_params=params)

    def test_refusal_parts_decode_with_a_refusal_stop_reason(self):
        refusal = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "I can't help with that."}],
            },
        }
        sample, _ = self._sample("api", _FakeSSEResponse(
            {"type": "response.refusal.delta", "output_index": 0, "delta": "I can't"},
            refusal,
            _completed_event(),
        ))
        self.assertEqual(sample.last_assistant_text, "I can't help with that.")
        self.assertEqual(sample.stop_reason, "refusal")
        sample, _ = self._sample(
            "api", _FakeSSEResponse(_message_event(0, "Sure."), _completed_event()),
        )
        self.assertEqual(sample.stop_reason, "end_turn")
        malformed = dict(refusal, item=dict(refusal["item"], content=[{"type": "refusal"}]))
        with self.assertRaises(ModelResponseError) as raised:
            self._sample("api", _FakeSSEResponse(malformed, _completed_event()))
        self.assertEqual(raised.exception.failure.category, "invalid_output_item")

    def test_context_window_codes_classify_http_and_stream_failures(self):
        body = json.dumps({"error": {
            "type": "invalid_request_error", "code": "context_length_exceeded",
            "message": "Input is too large.",  # No text marker: the code decides.
        }}).encode()
        outcomes = {
            "http": lambda: _http_error(400, body=body),
            "response.failed": lambda: _FakeSSEResponse({
                "type": "response.failed",
                "response": {"error": {"code": "context_length_exceeded", "message": "x"}},
            }),
            "public error event": lambda: _FakeSSEResponse({
                "type": "error", "code": "context_length_exceeded", "message": "x",
                "param": None, "sequence_number": 1,
            }),
            "nested error event": lambda: _FakeSSEResponse({
                "type": "error", "error": {"code": "context_length_exceeded"},
            }),
        }
        for name, outcome in outcomes.items():
            for api_provider in ("api", "codex"):
                with self.subTest(name, api_provider=api_provider):
                    with self.assertRaises(ModelContextWindowError) as raised:
                        self._sample(api_provider, outcome())
                    self.assertEqual(raised.exception.failure.category, "context_window")
                    self.assertEqual(
                        raised.exception.failure.error_code, "context_length_exceeded",
                    )
        with self.assertRaises(ModelResponseError) as raised:
            self._sample("api", _FakeSSEResponse({
                "type": "response.failed",
                "response": {"error": {"code": "server_error", "message": "x"}},
            }))
        self.assertNotIsInstance(raised.exception, ModelContextWindowError)
        self.assertEqual(raised.exception.failure.category, "response_failed")


if __name__ == "__main__":
    unittest.main()
