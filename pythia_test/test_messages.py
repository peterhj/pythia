from __future__ import annotations

from pythia_test.interaction_helpers import messages_endpoint

import http.client
import io
import json
import socket
import urllib.error
import unittest
from typing import Any
from unittest import mock

from pythia.interaction import ANTHROPIC_MESSAGES_API_URL
from pythia.interaction import DEFAULT_REQUEST_TIMEOUT_SECONDS
from pythia.interaction import Environment
from pythia.interaction import MESSAGES_COMPACTION_BETA
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import MessagesServerCompaction
from pythia.interaction import ModelConfigurationError
from pythia.interaction import InteractionContext
from pythia.interaction import ModelContextWindowError
from pythia.interaction import ModelResponseError
from pythia.interaction import ModelSample
from pythia.interaction import ModelTimeoutError
from pythia.interaction import ModelTransportError
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import SamplingParams
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import USER_AGENT
from pythia.interaction.demo import _build_model
from pythia.interaction.demo import _build_parser
from pythia.interaction.demo import run
from pythia.interaction.experimental_tools import create_inject_user_message_tool


class _FakeResponse:
    def __init__(self, payload: Any, *, status: int = 200, headers=None):
        self.status = status
        self.headers = dict(headers or {})
        self.payload = payload
        self.closed = False

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")

    def close(self) -> None:
        self.closed = True


class _Opener:
    def __init__(self, response: Any):
        self.response = response
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _ScriptedOpener:
    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _ReadFailureResponse(_FakeResponse):
    def read(self) -> bytes:
        raise self.payload


def _http_error(status, *, body=b"", headers=None):
    return urllib.error.HTTPError(
        "https://api.anthropic.com/v1/messages",
        status,
        "HTTP failure",
        dict(headers or {}),
        io.BytesIO(body),
    )


def _payload(opener: _Opener) -> dict[str, Any]:
    request, _ = opener.calls[-1]
    return json.loads(request.data.decode("utf-8"))


def _endpoint(*args, **kwargs) -> MessagesEndpoint:
    kwargs.setdefault("max_output_tokens", 100)
    return messages_endpoint(*args, **kwargs)


class MessagesEndpointTests(unittest.TestCase):
    def test_output_limit_prefers_explicit_then_catalog_and_requires_a_source(self):
        catalogued = messages_endpoint(
            api_url="http://localhost",
            model="claude-fable-5-1",
        )
        explicit = messages_endpoint(
            api_url="http://localhost",
            model="claude-fable-5-1",
            max_output_tokens=100,
        )
        self.assertEqual(catalogued.max_output_tokens, 128_000)
        self.assertEqual(explicit.max_output_tokens, 100)

        with self.assertRaisesRegex(
            ModelConfigurationError,
            "model catalog.*--max-output-tokens",
        ):
            messages_endpoint(api_url="http://localhost", model="model")

    def test_normalizes_url_and_redacts_key(self):
        endpoint = _endpoint(
            api_url=" HTTPS://api.example.test:8443/proxy/ ",
            model=" model-name ",
            api_key=" secret-key ",
        )

        self.assertEqual(
            endpoint.url,
            "HTTPS://api.example.test:8443/proxy/v1/messages",
        )
        self.assertEqual(endpoint.model, "model-name")
        self.assertEqual(endpoint.api_key, "secret-key")
        self.assertNotIn("secret-key", repr(endpoint))
        self.assertEqual(
            ANTHROPIC_MESSAGES_API_URL,
            "https://api.anthropic.com",
        )

    def test_rejects_invalid_configuration(self):
        cases = [
            {"api_url": "", "model": "model"},
            {"api_url": "localhost", "model": "model"},
            {"api_url": "ftp://localhost", "model": "model"},
            {"api_url": "http://user:pass@localhost", "model": "model"},
            {"api_url": "http://localhost", "model": ""},
            {"api_url": "http://localhost", "model": "model", "api_key": ""},
            {
                "api_url": "http://localhost",
                "model": "model",
                "anthropic_version": "bad\nheader",
            },
            {
                "api_url": "http://localhost",
                "model": "model",
                "max_output_tokens": 0,
            },
            {
                "api_url": "http://localhost",
                "model": "model",
                "request_timeout_seconds": 0,
            },
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ModelConfigurationError, ValueError)):
                    _endpoint(**kwargs)
        with self.assertRaisesRegex(TypeError, "retry_sleep"):
            MessagesModel(
                _endpoint(api_url="http://localhost", model="model"),
                retry_sleep=object(),
            )

    def test_server_compaction_configuration_is_validated(self):
        default = MessagesServerCompaction()
        self.assertEqual(
            default.request_edit(),
            {"type": "compact_20260112"},
        )
        configured = MessagesServerCompaction(
            trigger_input_tokens=150_000,
            pause_after_compaction=True,
            instructions=" summarize without tools ",
        )
        self.assertEqual(
            configured.request_edit(),
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 150_000},
                "pause_after_compaction": True,
                "instructions": "summarize without tools",
            },
        )
        with self.assertRaisesRegex(ModelConfigurationError, "50000"):
            MessagesServerCompaction(trigger_input_tokens=49_999)
        with self.assertRaisesRegex(ModelConfigurationError, "instructions"):
            MessagesServerCompaction(instructions=" ")
        with self.assertRaisesRegex(TypeError, "server_compaction"):
            _endpoint(
                api_url="http://localhost:8000",
                model="model",
                server_compaction=object(),
            )

    def test_server_compaction_honors_sampling_options_override(self):
        policy = MessagesServerCompaction()
        model = MessagesModel(_endpoint(
            api_url="https://api.anthropic.com",
            model="claude-fable-5-1",
            api_key="test-key",
            server_compaction=policy,
        ))

        payload = model._build_request_payload(
            InteractionContext((Message("user", "Hello."),)),
            (),
            SamplingParams(auto_compact_tokens=123_456),
        )

        self.assertEqual(
            payload["context_management"]["edits"][0]["trigger"]["value"],
            123_456,
        )
        with self.assertRaisesRegex(ModelConfigurationError, "at least"):
            model._build_request_payload(
                InteractionContext((Message("user", "Hello."),)),
                (),
                SamplingParams(auto_compact_tokens=49_999),
            )

        explicit = MessagesModel(_endpoint(
            api_url="https://api.anthropic.com",
            model="claude-fable-5-1",
            api_key="test-key",
            server_compaction=MessagesServerCompaction(
                trigger_input_tokens=150_000,
            ),
        ))
        explicit_payload = explicit._build_request_payload(
            InteractionContext((Message("user", "Hello."),)),
            (),
            SamplingParams(auto_compact_tokens=200_000),
        )
        self.assertEqual(
            explicit_payload["context_management"]["edits"][0]["trigger"]["value"],
            150_000,
        )

    def test_server_compaction_uses_catalog_auto_compact_threshold(self):
        policy = MessagesServerCompaction()
        model = MessagesModel(_endpoint(
            api_url="https://api.anthropic.com",
            model="claude-fable-5-1",
            api_key="test-key",
            server_compaction=policy,
        ))

        payload = model._build_request_payload(
            InteractionContext((Message("user", "Hello."),)),
            (),
            None,
        )

        self.assertEqual(payload["context_management"], {
            "edits": [{
                "type": "compact_20260112",
                "trigger": {
                    "type": "input_tokens",
                    "value": 872_000,
                },
            }],
        })
        self.assertIs(model.endpoint.server_compaction, policy)
        self.assertIsNone(policy.trigger_input_tokens)

        disabled = model._build_request_payload(
            InteractionContext((Message("user", "Hello."),)),
            (),
            SamplingParams(enable_auto_compaction=False),
        )
        self.assertNotIn("context_management", disabled)

        runtime_enabled = MessagesModel(_endpoint(
            api_url="https://api.anthropic.com",
            model="claude-fable-5-1",
            api_key="test-key",
        ))
        enabled_after_startup = runtime_enabled._build_request_payload(
            InteractionContext((Message("user", "Hello."),)),
            (),
            SamplingParams(enable_auto_compaction=True),
        )
        self.assertEqual(
            enabled_after_startup["context_management"],
            payload["context_management"],
        )
        self.assertIsNone(runtime_enabled.endpoint.server_compaction)

        response = _FakeResponse({
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Done."}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        opener = _Opener(response)
        MessagesModel(model.endpoint, opener=opener).sample(
            InteractionContext((Message("user", "Hello."),)),
            sampling_params=SamplingParams(enable_auto_compaction=False),
        )
        request, _ = opener.calls[0]
        self.assertIsNone(request.get_header("Anthropic-beta"))


class MessagesModelTests(unittest.TestCase):
    def test_encodes_context_tools_options_and_decodes_response(self):
        response = _FakeResponse(
            {
                "type": "message",
                "role": "assistant",
                "model": "resolved-model",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Need the tool.",
                        "signature": "new-signature",
                    },
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "call-new",
                        "name": "lookup",
                        "input": {"q": "new"},
                    },
                ],
                "usage": {
                    "input_tokens": 20,
                    "cache_creation_input_tokens": 7,
                    "cache_read_input_tokens": 4,
                    "output_tokens": 5,
                },
            }
        )
        opener = _Opener(response)
        model = MessagesModel(
            _endpoint(
                api_url="https://api.example.test/anthropic",
                model="claude-sonnet-5",
                api_key="secret-key",
                max_output_tokens=2048,
            ),
            opener=opener,
        )
        context = InteractionContext(
            (
                Message(role="system", content="system text"),
                Message(role="developer", content="developer text"),
                Message(role="user", content="question"),
                UserInteractionBoundary(),
                Reasoning(
                    content="prior thought",
                    encrypted_content="responses-only-data",
                    content_signature="prior-signature",
                ),
                Message(role="assistant", content="calling"),
                ToolCall(
                    name="lookup",
                    call_id="call-old",
                    arguments_json='{"q":"old"}',
                ),
                ToolResult(call_id="call-old", output="result", success=False),
            )
        )
        tool = ToolSpec(
            name="lookup",
            description="Look something up.",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )

        sample = model.sample(
            context,
            tools=(tool,),
            sampling_params=SamplingParams(
                max_output_tokens=100,
                temperature=0.25,
                top_p=0.9,
                stop=("END",),
            ),
        )

        request, timeout = opener.calls[0]
        payload = _payload(opener)
        self.assertEqual(timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(
            request.full_url,
            "https://api.example.test/anthropic/v1/messages",
        )
        self.assertEqual(request.get_header("User-agent"), USER_AGENT)
        self.assertEqual(request.get_header("X-api-key"), "secret-key")
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(payload["model"], "claude-sonnet-5")
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["stop_sequences"], ["END"])
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["system"],
            [
                {"type": "text", "text": "system text"},
                {"type": "text", "text": "developer text"},
            ],
        )
        self.assertEqual(
            payload["messages"],
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "question"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "prior thought",
                            "signature": "prior-signature",
                        },
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "call-old",
                            "name": "lookup",
                            "input": {"q": "old"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-old",
                            "content": "result",
                            "is_error": True,
                        }
                    ],
                },
            ],
        )
        self.assertNotIn("responses-only-data", request.data.decode("utf-8"))
        self.assertEqual(
            payload["tools"],
            [
                {
                    "name": "lookup",
                    "description": "Look something up.",
                    "input_schema": tool.parameters,
                }
            ],
        )
        self.assertEqual(
            sample.items,
            (
                Reasoning(
                    content="Need the tool.",
                    content_signature="new-signature",
                ),
                Message(role="assistant", content="Checking."),
                ToolCall(
                    name="lookup",
                    call_id="call-new",
                    arguments_json='{"q":"new"}',
                ),
            ),
        )
        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(sample.usage.input_tokens, 31)
        self.assertEqual(sample.usage.output_tokens, 5)
        self.assertEqual(sample.usage.total_tokens, 36)
        self.assertEqual(sample.usage.cached_input_tokens, 4)
        self.assertTrue(response.closed)

    def test_signed_thinking_without_text_is_redacted_and_replayed(self):
        # A thinking block may return empty text with only a signature. Show
        # the same placeholder as encrypted-only Responses reasoning, and
        # replay the block unchanged so the provider can verify it.
        signature = "thinking-signature-must-not-be-displayed"
        thinking = {"type": "thinking", "thinking": "", "signature": signature}
        opener = _ScriptedOpener(
            _FakeResponse({
                "type": "message",
                "role": "assistant",
                "content": [thinking, {"type": "text", "text": "Done."}],
                "stop_reason": "end_turn",
            }),
            _FakeResponse({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Again."}],
                "stop_reason": "end_turn",
            }),
        )
        model = MessagesModel(
            _endpoint(api_url="http://localhost:8000", model="model"),
            opener=opener,
        )
        context = InteractionContext((Message("user", "Inspect."),))

        sample = model.sample(context)

        self.assertEqual(sample.items, (
            Reasoning(content="", content_signature=signature),
            Message("assistant", "Done."),
        ))
        displayed = tuple(item.text for item in sample.display_items())
        self.assertEqual(displayed[:2], ("[reasoning] ...", "[assistant] Done."))
        self.assertNotIn(signature, "\n".join(displayed))

        context.extend(sample.context_items())
        context.append(Message("user", "Continue."))
        model.sample(context)

        self.assertEqual(_payload(opener)["messages"][1], {
            "role": "assistant",
            "content": [thinking, {"type": "text", "text": "Done."}],
        })

    def test_injected_message_is_user_text_after_tool_result_block(self):
        tool = create_inject_user_message_tool()
        call = ToolCall(tool.spec.name, "inject-1", "{}")
        context = InteractionContext((Message("user", "Run the experiment."),))
        context.extend(ModelSample(items=(call,)).context_items())
        environment = Environment((tool,))
        context.extend(environment.execute_tool_calls((call,)).context_items())
        opener = _Opener(_FakeResponse({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "received: hello world"}],
            "stop_reason": "end_turn",
        }))
        model = MessagesModel(
            _endpoint(api_url="http://localhost:8000", model="model"),
            opener=opener,
        )
        model.sample(context, tools=environment.tool_specs)
        self.assertEqual(_payload(opener)["messages"][-1], {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.call_id,
                    "content": "Synthetic user message queued.",
                    "is_error": False,
                },
                {"type": "text", "text": "hello world"},
            ],
        })

    def test_uses_configured_tokens_and_omits_optional_fields(self):
        opener = _Opener(
            _FakeResponse(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                }
            )
        )
        model = MessagesModel(
            _endpoint(
                api_url="http://localhost:8000",
                model="model",
                max_output_tokens=77,
            ),
            opener=opener,
        )

        model.sample(InteractionContext((Message(role="user", content="hello"),)))

        payload = _payload(opener)
        self.assertEqual(payload["max_tokens"], 77)
        self.assertNotIn("system", payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("context_management", payload)
        request, _ = opener.calls[0]
        self.assertIsNone(request.get_header("X-api-key"))
        self.assertIsNone(request.get_header("Anthropic-beta"))

    def test_rejects_unsupported_or_unsafe_context(self):
        model = MessagesModel(
            _endpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(_FakeResponse({})),
        )
        cases = [
            (
                InteractionContext(
                    (
                        Message(role="user", content="hello"),
                        Message(role="system", content="late"),
                    )
                ),
                None,
                "appears after",
            ),
            (
                InteractionContext(
                    (
                        Message(role="user", content="hello"),
                        ToolCall(
                            name="tool",
                            call_id="call",
                            arguments_json="[]",
                        ),
                        ToolResult(call_id="call", output="done"),
                    )
                ),
                None,
                "decode to an object",
            ),
            (
                InteractionContext((Message(role="user", content="hello"),)),
                SamplingParams(seed=1),
                "seed",
            ),
        ]
        for context, options, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ModelConfigurationError, message):
                    model.sample(context, sampling_params=options)

    def test_rejects_unknown_response_blocks(self):
        model = MessagesModel(
            _endpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(
                _FakeResponse(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "x"}],
                    }
                )
            ),
        )
        with self.assertRaisesRegex(ModelResponseError, "redacted_thinking"):
            model.sample(InteractionContext((Message(role="user", content="hello"),)))

    def test_context_window_http_error_is_typed(self):
        error = urllib.error.HTTPError(
            url="http://localhost:8000/v1/messages",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"context window exceeded"}}'),
        )
        model = MessagesModel(
            _endpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(error),
        )

        with self.assertRaises(ModelContextWindowError):
            model.sample(InteractionContext((Message(role="user", content="hello"),)))

    def test_server_compaction_round_trips_and_projects_latest_block(self):
        first_response = _FakeResponse(
            {
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "compaction",
                        "content": "Summary of old work.",
                    },
                    {"type": "text", "text": "First answer."},
                ],
                "usage": {
                    "input_tokens": 23_000,
                    "output_tokens": 1_000,
                    "cache_read_input_tokens": 500,
                    "iterations": [
                        {
                            "type": "compaction",
                            "input_tokens": 180_000,
                            "output_tokens": 3_500,
                        },
                        {
                            "type": "message",
                            "input_tokens": 23_000,
                            "output_tokens": 1_000,
                            "cache_read_input_tokens": 500,
                        },
                    ],
                },
            }
        )
        second_response = _FakeResponse(
            {
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Second answer."}],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            }
        )
        opener = _ScriptedOpener(first_response, second_response)
        model = MessagesModel(
            _endpoint(
                api_url="http://localhost:8000",
                model="claude-sonnet-5",
                server_compaction=MessagesServerCompaction(),
            ),
            opener=opener,
        )
        context = InteractionContext(
            (
                Message(role="system", content="instructions"),
                Message(role="user", content="old question"),
            )
        )

        first = model.sample(context)

        first_request, _ = opener.calls[0]
        first_payload = json.loads(first_request.data.decode("utf-8"))
        self.assertEqual(
            first_request.get_header("Anthropic-beta"),
            MESSAGES_COMPACTION_BETA,
        )
        self.assertEqual(
            first_payload["context_management"],
            {"edits": [{"type": "compact_20260112"}]},
        )
        self.assertEqual(
            first.items,
            (
                OpaqueCompaction.from_messages("Summary of old work."),
                Message(role="assistant", content="First answer."),
            ),
        )
        self.assertEqual(first.usage.input_tokens, 203_500)
        self.assertEqual(first.usage.output_tokens, 4_500)
        self.assertEqual(first.usage.total_tokens, 208_000)
        self.assertEqual(first.usage.cached_input_tokens, 500)

        context.extend(first.context_items())
        context.extend(
            (
                Message(role="user", content="new question"),
                UserInteractionBoundary(),
            )
        )
        second = model.sample(context)

        self.assertEqual(second.last_assistant_text, "Second answer.")
        second_request, _ = opener.calls[1]
        second_payload = json.loads(second_request.data.decode("utf-8"))
        self.assertEqual(
            second_payload["messages"],
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "compaction",
                            "content": "Summary of old work.",
                        },
                        {"type": "text", "text": "First answer."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "new question"}
                    ],
                },
            ],
        )
        self.assertNotIn("old question", second_request.data.decode("utf-8"))
        self.assertEqual(
            second_payload["system"],
            [{"type": "text", "text": "instructions"}],
        )

    def test_rejects_null_compaction_and_responses_subtype(self):
        null_model = MessagesModel(
            _endpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(
                _FakeResponse(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "compaction", "content": None}
                        ],
                    }
                )
            ),
        )
        with self.assertRaisesRegex(ModelResponseError, "content"):
            null_model.sample(
                InteractionContext((Message(role="user", content="hello"),))
            )

        wrong_protocol_model = MessagesModel(
            _endpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(_FakeResponse({})),
        )
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "Responses opaque compaction",
        ):
            wrong_protocol_model.sample(
                InteractionContext(
                    (
                        OpaqueCompaction.from_responses("encrypted"),
                        Message(role="user", content="hello"),
                    )
                )
            )

    def test_http_413_is_a_context_window_error(self):
        error = urllib.error.HTTPError(
            url="http://localhost:8000/v1/messages",
            code=413,
            msg="request too large",
            hdrs=None,
            fp=io.BytesIO(b'{"type":"error","error":{"type":"request_too_large"}}'),
        )
        model = MessagesModel(
            _endpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(error),
        )
        with self.assertRaises(ModelContextWindowError):
            model.sample(InteractionContext((Message(role="user", content="hello"),)))

    def test_retryable_http_statuses_use_two_retries_and_metadata(self):
        returned_overload = _FakeResponse(
            {"type": "error"},
            status=503,
        )
        success = _FakeResponse({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "recovered"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        })
        opener = _ScriptedOpener(
            _http_error(529, headers={"retry-after": "0.75"}),
            returned_overload,
            success,
        )
        sleeps = []
        model = MessagesModel(
            _endpoint(api_url="https://api.anthropic.com", model="model"),
            opener=opener,
            retry_sleep=sleeps.append,
        )

        sample = model.sample(
            InteractionContext((Message(role="user", content="hello"),))
        )

        self.assertEqual(sample.last_assistant_text, "recovered")
        self.assertEqual(sample.request_attempts, 3)
        self.assertEqual(
            sample.recovery,
            ("http_529_retry", "http_503_retry"),
        )
        self.assertEqual(sleeps, [0.75, 0.5])
        self.assertTrue(returned_overload.closed)
        self.assertTrue(success.closed)
        request_bodies = tuple(call[0].data for call in opener.calls)
        self.assertEqual(request_bodies, (request_bodies[0],) * 3)

    def test_connection_and_body_read_failures_are_retried(self):
        failed_read = _ReadFailureResponse(
            http.client.IncompleteRead(b"truncated body")
        )
        success = _FakeResponse({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "recovered"}],
            "stop_reason": "end_turn",
            "usage": {},
        })
        opener = _ScriptedOpener(
            urllib.error.URLError(OSError("connection reset")),
            failed_read,
            success,
        )
        sleeps = []
        sample = MessagesModel(
            _endpoint(api_url="https://api.anthropic.com", model="model"),
            opener=opener,
            retry_sleep=sleeps.append,
        ).sample(InteractionContext((Message("user", "hello"),)))

        self.assertEqual(sample.request_attempts, 3)
        self.assertEqual(
            sample.recovery,
            ("connection_retry", "connection_retry"),
        )
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertTrue(failed_read.closed)

    def test_timeout_retry_exhaustion_has_safe_attempt_metadata(self):
        opener = _ScriptedOpener(*(
            socket.timeout("private timeout detail") for _ in range(3)
        ))
        sleeps = []
        model = MessagesModel(
            _endpoint(api_url="https://api.anthropic.com", model="model"),
            opener=opener,
            retry_sleep=sleeps.append,
        )

        with self.assertRaises(ModelTimeoutError) as raised:
            model.sample(InteractionContext((Message("user", "hello"),)))

        self.assertEqual(len(opener.calls), 3)
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertEqual(raised.exception.failure.attempt_count, 3)
        self.assertEqual(
            raised.exception.failure.recovery,
            ("request_timeout_retry", "request_timeout_retry"),
        )
        self.assertNotIn("private timeout detail", str(raised.exception))


class ReasoningSignatureTests(unittest.TestCase):
    def test_signature_is_validated_and_redacted(self):
        reasoning = Reasoning(
            content="thought",
            content_signature="signature-value",
        )
        self.assertNotIn("signature-value", repr(reasoning))
        with self.assertRaisesRegex(ValueError, "content_signature"):
            Reasoning(content="thought", content_signature="")


class MessagesDemoTests(unittest.TestCase):
    def test_uncatalogued_frontend_model_requires_an_explicit_output_limit(self):
        args = _build_parser().parse_args([
            "--endpoint-api",
            "messages",
            "--model",
            "claude-sonnet-5",
        ])

        with self.assertRaisesRegex(ValueError, "--max-output-tokens"):
            _build_model(args)

    def test_frontend_uses_catalogued_output_limit_when_omitted(self):
        args = _build_parser().parse_args([
            "--endpoint-api",
            "messages",
            "--model",
            "claude-fable-5-1",
            "--endpoint-auth", "none",
        ])

        model = _build_model(args)

        self.assertEqual(model.endpoint.max_output_tokens, 128_000)

    def test_builds_anthropic_messages_model(self):
        args = _build_parser().parse_args(
            [
                "--endpoint-api",
                "messages",
                "--model",
                "claude-sonnet-5",
                "--max-output-tokens",
                "100",
            ]
        )

        with mock.patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "secret-key"},
        ):
            model = _build_model(args)

        self.assertIsInstance(model, MessagesModel)
        self.assertEqual(
            model.endpoint.url,
            ANTHROPIC_MESSAGES_API_URL + "/v1/messages",
        )
        self.assertEqual(
            model.endpoint.model,
            "claude-sonnet-5",
        )
        self.assertNotIn("secret-key", repr(model.endpoint))

    def test_auto_compaction_flag_configures_messages_server_policy(self):
        base = [
            "--endpoint-api",
            "messages",
            "--model",
            "claude-sonnet-5",
            "--max-output-tokens",
            "100",
            "--endpoint-auth", "none",
        ]
        enabled = _build_model(_build_parser().parse_args(base))
        disabled = _build_model(_build_parser().parse_args([
            *base,
            "--enable-auto-compaction=False",
        ]))

        self.assertEqual(
            enabled.endpoint.server_compaction,
            MessagesServerCompaction(),
        )
        self.assertIsNone(disabled.endpoint.server_compaction)

        for removed in (
            "--messages-server-compaction",
            "--messages-compaction-trigger-tokens=200000",
            "--messages-pause-after-compaction",
            "--messages-compaction-instructions=value",
        ):
            with self.subTest(removed=removed):
                with mock.patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        _build_parser().parse_args([*base, removed])
                self.assertEqual(raised.exception.code, 2)

    def test_demo_continues_after_paused_compaction(self):
        class Model:
            def __init__(self):
                self.calls = []

            def sample(self, context, *, tools=(), sampling_params=None):
                del tools, sampling_params
                self.calls.append(context.copy())
                if len(self.calls) == 1:
                    return ModelSample(
                        items=(
                            OpaqueCompaction.from_messages("summary"),
                        ),
                        stop_reason="compaction",
                    )
                return ModelSample(
                    items=(Message(role="assistant", content="done"),),
                    stop_reason="end_turn",
                )

        model = Model()
        with mock.patch("builtins.print"):
            result = run(
                model,
                Environment(),
                prompt="hello",
                max_samples=2,
            )

        self.assertEqual(result, "done")
        self.assertEqual(len(model.calls), 2)
        self.assertIn(
            OpaqueCompaction.from_messages("summary"),
            model.calls[1].items,
        )


if __name__ == "__main__":
    unittest.main()
