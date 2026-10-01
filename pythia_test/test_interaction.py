from __future__ import annotations

from pythia_test.interaction_helpers import chat_endpoint

import io
import json
import socket
import urllib.error
import urllib.request
import unittest
from unittest import mock

import pythia.interaction as interaction
import pythia.interaction.context as context_module

from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import COMPACTION_SUMMARY_PREFIX
from pythia.interaction import COMPACTION_SUMMARY_SUFFIX
from pythia.interaction import CompactionContextWindowError
from pythia.interaction import CompactionError
from pythia.interaction import CompactionMetadata
from pythia.interaction import CompactionSettings
from pythia.interaction import ContextPrefix
from pythia.interaction import ContextValidationError
from pythia.interaction import DEFAULT_KEEP_RECENT_TOKENS
from pythia.interaction import DEFAULT_REQUEST_TIMEOUT_SECONDS
from pythia.interaction import Environment
from pythia.interaction import EnvironmentError
from pythia.interaction import EnvironmentResult
from pythia.interaction import Init
from pythia.interaction import InteractionConfigSnapshot
from pythia.interaction import Instructions
from pythia.interaction import MediaPart
from pythia.interaction import Message
from pythia.interaction import ModelConfigurationError
from pythia.interaction import InteractionContext
from pythia.interaction import ModelContextWindowError
from pythia.interaction import ModelFailure
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import ModelTimeoutError
from pythia.interaction import ModelTransportError
from pythia.interaction import NothingToCompact
from pythia.interaction import OpaqueCompaction
from pythia.interaction import PiCompactor
from pythia.interaction import Reasoning
from pythia.interaction import SampleParams
from pythia.interaction import SUMMARIZATION_PROMPT
from pythia.interaction import SUMMARIZATION_SYSTEM_PROMPT
from pythia.interaction import TextPart
from pythia.interaction import TokenUsage
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolOutcome
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import SampleMetadata
from pythia.interaction import TURN_PREFIX_SUMMARIZATION_PROMPT
from pythia.interaction import TurnSummary
from pythia.interaction import UPDATE_SUMMARIZATION_PROMPT
from pythia.interaction import UserInteraction
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import UserToolCall
from pythia.interaction import UserToolResult
from pythia.interaction import USER_AGENT
from pythia.interaction import auto_compaction_due
from pythia.interaction import create_default_compactor
from pythia.interaction import estimate_context_tokens
from pythia.interaction import is_compaction_summary
from pythia.interaction import should_auto_compact
from pythia.interaction.compaction import _LEGACY_SUMMARY_PREFIX
from pythia.interaction.compaction import _cut_points
from pythia.interaction.compaction import estimate_item_tokens
from pythia.interaction.experimental_tools import create_inject_user_message_tool


class _FakeHTTPResponse:
    def __init__(self, payload, *, status=200, headers=None):
        self.status = status
        self.headers = dict(headers or {})
        self._payload = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self):
        return self._payload

    def close(self):
        self.closed = True


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
        return outcome


class _ReadFailureResponse:
    def __init__(self, failure):
        self.status = 200
        self.headers = {}
        self.failure = failure
        self.closed = False

    def read(self):
        raise self.failure

    def close(self):
        self.closed = True


def _http_error(status, *, body=b"", headers=None):
    return urllib.error.HTTPError(
        "https://api.example.test/v1/chat/completions",
        status,
        "HTTP failure",
        dict(headers or {}),
        io.BytesIO(body),
    )


def _request_payload(opener, index=0):
    request, _ = opener.calls[index]
    return json.loads(request.data.decode("utf-8"))


class InteractionContextTests(unittest.TestCase):
    def test_public_context_name_and_repr(self):
        self.assertIs(context_module.InteractionContext, InteractionContext)
        for module in (interaction, context_module):
            self.assertIn("InteractionContext", module.__all__)
            self.assertNotIn("ModelContext", module.__all__)
            self.assertFalse(hasattr(module, "ModelContext"))

        self.assertEqual(repr(InteractionContext()), "InteractionContext([])")
        items = [Message(role="user", content="hello")]
        self.assertEqual(
            repr(InteractionContext(items)), f"InteractionContext({items!r})"
        )

    def test_context_is_append_only_and_projects_compaction(self):
        original = [
            Message(role="user", content="old request"),
            Message(role="assistant", content="old answer"),
        ]
        context = InteractionContext(original)
        checkpoint = ContextPrefix(
            prefix_items=(
                Message(role="user", content="summary"),
            )
        )

        context.append(checkpoint)
        context.append(Message(role="user", content="new request"))

        self.assertEqual(
            context.items,
            (
                *original,
                checkpoint,
                Message(role="user", content="new request"),
            ),
        )
        self.assertEqual(
            context.model_items(),
            (
                Message(role="user", content="summary"),
                Message(role="user", content="new request"),
            ),
        )

    def test_nested_compaction_is_rejected_without_mutation(self):
        context = InteractionContext([Message(role="user", content="hello")])
        nested = ContextPrefix(
            prefix_items=(
                ContextPrefix(
                    prefix_items=(Message(role="user", content="summary"),)
                ),
            )
        )

        with self.assertRaisesRegex(
            ContextValidationError,
            "must not contain another ContextPrefix",
        ):
            context.append(nested)

        self.assertEqual(
            context.items,
            (Message(role="user", content="hello"),),
        )

    def test_compaction_metadata_is_log_only_and_not_prefix_content(self):
        metadata = CompactionMetadata(
            TokenUsage(10, 2, 12, 4),
            "responses_compaction_v2",
            elapsed_seconds=3.5,
        )
        prefix = ContextPrefix((Message("user", "summary"),))
        context = InteractionContext((prefix, metadata))
        self.assertEqual(context.items, (prefix, metadata))
        self.assertEqual(context.model_items(), prefix.prefix_items)
        with self.assertRaisesRegex(
            ContextValidationError,
            "compaction metadata cannot appear in context prefixes",
        ):
            InteractionContext((ContextPrefix((metadata,)),))

    def test_new_interaction_is_rejected_before_tool_results(self):
        call = ToolCall(
            name="lookup",
            call_id="call-1",
            arguments_json="{}",
        )
        context = InteractionContext(
            [
                call,
                ModelSampleBoundary(),
            ]
        )

        with self.assertRaisesRegex(
            ContextValidationError,
            "before unresolved tool results",
        ):
            context.append(Message(role="user", content="continue"))

        self.assertEqual(
            context.items,
            (
                call,
                ModelSampleBoundary(),
            ),
        )

    def test_compaction_projection_preserves_interaction_boundaries(self):
        user_boundary = UserInteractionBoundary()
        sample_boundary = ModelSampleBoundary()
        checkpoint = ContextPrefix(
            prefix_items=(
                Message(role="user", content="retained request"),
                user_boundary,
                Message(role="assistant", content="retained answer"),
                sample_boundary,
            )
        )
        context = InteractionContext(
            [
                Message(role="user", content="old request"),
                checkpoint,
                Message(role="user", content="new request"),
            ]
        )

        self.assertEqual(
            context.model_items(),
            (
                Message(role="user", content="retained request"),
                user_boundary,
                Message(role="assistant", content="retained answer"),
                sample_boundary,
                Message(role="user", content="new request"),
            ),
        )

    def test_pending_tool_calls_are_derived_from_effective_context(self):
        context = InteractionContext(
            [
                ToolCall(
                    name="lookup",
                    call_id="call-1",
                    arguments_json="{}",
                )
            ]
        )

        self.assertEqual(
            tuple(call.call_id for call in context.pending_tool_calls()),
            ("call-1",),
        )
        with self.assertRaisesRegex(ContextValidationError, "unresolved"):
            context.assert_model_ready()

        context.append(
            ToolResult(call_id="call-1", output="done")
        )
        context.assert_model_ready()

    def test_sample_metadata_is_a_transparent_control_item(self):
        call = ToolCall(
            name="lookup",
            call_id="call-1",
            arguments_json="{}",
        )
        metadata = SampleMetadata(
            usage=TokenUsage(
                input_tokens=20,
                output_tokens=5,
                total_tokens=25,
                cached_input_tokens=4,
            )
        )
        context = InteractionContext(
            [
                Message(role="user", content="lookup"),
                UserInteractionBoundary(),
                call,
                metadata,
                ModelSampleBoundary(),
            ]
        )

        self.assertIn(call, context.pending_tool_calls())
        self.assertEqual(
            context.model_items(),
            (
                Message(role="user", content="lookup"),
                UserInteractionBoundary(),
                call,
                metadata,
                ModelSampleBoundary(),
            ),
        )

        context.append(ToolResult(call_id="call-1", output="done"))
        context.assert_model_ready()

    def test_unknown_tool_result_is_rejected_atomically(self):
        context = InteractionContext([Message(role="user", content="hello")])

        with self.assertRaisesRegex(ContextValidationError, "does not match"):
            context.extend(
                [
                    Message(role="assistant", content="answer"),
                    ToolResult(call_id="missing", output="bad"),
                ]
            )

        self.assertEqual(
            context.items,
            (Message(role="user", content="hello"),),
        )

    def test_copy_branches_the_context(self):
        context = InteractionContext([Message(role="user", content="root")])
        branch = context.copy()
        self.assertIsInstance(branch, InteractionContext)
        self.assertIsNot(branch, context)
        branch.append(Message(role="assistant", content="branch"))

        self.assertEqual(len(context), 1)
        self.assertEqual(len(branch), 2)

        original_only = Message(role="assistant", content="original")
        context.append(original_only)
        self.assertNotIn(original_only, branch)


class UserInteractionTests(unittest.TestCase):
    def test_context_items_adds_non_emitting_boundary(self):
        interaction = UserInteraction(
            items=[Message(role="user", content="hello")],
        )

        self.assertEqual(
            interaction.items,
            (Message(role="user", content="hello"),),
        )
        self.assertEqual(
            interaction.context_items(),
            (
                Message(role="user", content="hello"),
                UserInteractionBoundary(),
            ),
        )

    def test_rejects_empty_or_non_user_items(self):
        cases = (
            (),
            (Message(role="assistant", content="answer"),),
            (Reasoning(content="thought"),),
        )

        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(ValueError):
                    UserInteraction(items=items)


class EndpointTests(unittest.TestCase):
    def test_user_agent_matches_urllib_default(self):
        urllib_headers = dict(urllib.request.OpenerDirector().addheaders)
        self.assertEqual(USER_AGENT, urllib_headers["User-agent"])

    def test_endpoint_builds_url_after_path_prefix(self):
        endpoint = chat_endpoint(
            api_url=" HTTPS://api.example.test:8443/proxy/root/ ",
            model=" example-model ",
        )

        self.assertEqual(
            endpoint.url,
            "HTTPS://api.example.test:8443/proxy/root/v1/chat/completions",
        )
        self.assertEqual(endpoint.model, "example-model")

    def test_endpoint_normalizes_absent_model(self):
        endpoint = chat_endpoint(
            api_url="http://localhost:9000/",
            model=" ",
        )

        self.assertIsNone(endpoint.model)
        self.assertEqual(
            endpoint.url,
            "http://localhost:9000/v1/chat/completions",
        )

    def test_endpoint_rejects_invalid_configuration(self):
        with self.assertRaises((TypeError, AttributeError)):
            chat_endpoint(api_url=object())

        cases = [
            {"api_url": ""},
            {"api_url": "localhost:8000"},
            {"api_url": "ftp://localhost"},
            {"api_url": "http:///missing-host"},
            {"api_url": "http://user:password@localhost"},
            {"api_url": "http://localhost/prefix?query=value"},
            {"api_url": "http://localhost/prefix#fragment"},
            {"api_url": "http://localhost:not-a-port"},
            {"api_url": "http://localhost:0"},
            {
                "api_url": "http://localhost",
                "request_timeout_seconds": 0,
            },
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ModelConfigurationError, ValueError)):
                    chat_endpoint(**kwargs)
        with self.assertRaisesRegex(TypeError, "retry_sleep"):
            ChatCompletionsModel(
                chat_endpoint(api_url="http://localhost"),
                retry_sleep=object(),
            )

    def test_endpoint_validates_and_redacts_api_key(self):
        endpoint = chat_endpoint(
            api_url="http://localhost:8000",
            api_key=" example-secret ",
        )

        self.assertEqual(endpoint.api_key, "example-secret")
        self.assertNotIn("example-secret", repr(endpoint))

        with self.assertRaisesRegex(TypeError, "api_key"):
            chat_endpoint(
                api_url="http://localhost:8000",
                api_key=object(),
            )
        for api_key in ("", " ", "two words"):
            with self.subTest(api_key=api_key):
                with self.assertRaisesRegex(
                    ModelConfigurationError,
                    "api_key",
                ):
                    chat_endpoint(
                        api_url="http://localhost:8000",
                        api_key=api_key,
                    )


class ChatCompletionsModelTests(unittest.TestCase):
    def test_sample_context_items_adds_non_emitting_boundary(self):
        sample = ModelSample(
            items=(Message(role="assistant", content="answer"),),
        )
        usage = TokenUsage(
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
            cached_input_tokens=4,
        )
        sample_with_usage = ModelSample(
            items=(Message(role="assistant", content="answer"),),
            usage=usage,
        )

        self.assertEqual(
            sample.context_items(),
            (
                Message(role="assistant", content="answer"),
                SampleMetadata(usage=TokenUsage()),
                ModelSampleBoundary(),
            ),
        )
        self.assertEqual(
            sample_with_usage.context_items(),
            (
                Message(role="assistant", content="answer"),
                SampleMetadata(usage=usage),
                ModelSampleBoundary(),
            ),
        )
        self.assertEqual(
            sample.items,
            (Message(role="assistant", content="answer"),),
        )

    def test_sample_encodes_request_and_decodes_tool_call(self):
        response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I will check.",
                            "reasoning_content": "Need the tool.",
                            "tool_calls": [
                                {
                                    "id": "call-weather",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            }
        )
        opener = _ScriptedOpener(response)
        model = ChatCompletionsModel(
            chat_endpoint(
                api_url="http://localhost:8000",
                model="demo",
            ),
            opener=opener,
        )
        context = InteractionContext(
            [
                Message(role="system", content="Be concise."),
                Message(role="user", content="Weather in Paris?"),
            ]
        )
        before = context.items
        spec = ToolSpec(
            name="lookup_weather",
            description="Look up weather.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        )

        sample = model.sample(
            context,
            tools=(spec,),
            sample_params=SampleParams(
                max_output_tokens=100,
                temperature=0.25,
                stop=("END",),
                seed=7,
            ),
        )

        self.assertEqual(context.items, before)
        self.assertEqual(
            sample.items,
            (
                Reasoning(content="Need the tool."),
                Message(role="assistant", content="I will check."),
                ToolCall(
                    name="lookup_weather",
                    call_id="call-weather",
                    arguments_json='{"city":"Paris"}',
                ),
            ),
        )
        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(sample.usage.total_tokens, 25)
        self.assertEqual(sample.usage.cached_input_tokens, 4)
        self.assertIsNotNone(sample.elapsed_seconds)
        self.assertEqual(
            sample.context_items()[-2:],
            (
                SampleMetadata(usage=sample.usage, elapsed_seconds=sample.elapsed_seconds),
                ModelSampleBoundary(),
            ),
        )
        self.assertEqual(
            sample.display_items()[-1].text,
            "[sample] input=20 output=5 total=25 cached=4 "
            f"elapsed={sample.elapsed_seconds:.2f}s",
        )
        self.assertTrue(response.closed)

        request, timeout = opener.calls[0]
        payload = _request_payload(opener)
        self.assertEqual(timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(
            request.full_url,
            "http://localhost:8000/v1/chat/completions",
        )
        self.assertEqual(request.get_header("User-agent"), USER_AGENT)
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(payload["model"], "demo")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["stop"], ["END"])
        self.assertEqual(payload["seed"], 7)

    def test_api_key_uses_bearer_authorization(self):
        response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "authenticated",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        opener = _ScriptedOpener(response)
        endpoint = chat_endpoint(
            api_url="https://api.example.test/proxy",
            api_key=" example-secret ",
        )
        model = ChatCompletionsModel(
            endpoint,
            opener=opener,
        )

        sample = model.sample(
            InteractionContext([Message(role="user", content="hello")])
        )

        request, _ = opener.calls[0]
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer example-secret",
        )
        self.assertEqual(
            request.full_url,
            "https://api.example.test/proxy/v1/chat/completions",
        )
        self.assertEqual(sample.last_assistant_text, "authenticated")

    def test_caller_drives_tool_follow_up(self):
        first_response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"value":"hello"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        final_response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The tool said hello.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        opener = _ScriptedOpener(first_response, final_response)
        model = ChatCompletionsModel(
            chat_endpoint(api_url="http://localhost:8000"),
            opener=opener,
        )

        def echo(arguments, *, timeout_seconds=None):
            self.assertEqual(timeout_seconds, 3.0)
            return ToolOutcome(output=str(arguments["value"]))

        environment = Environment(
            tools=(
                Tool(
                    spec=ToolSpec(
                        name="echo",
                        description="Echo a value.",
                        parameters={"type": "object"},
                    ),
                    handler=echo,
                    timeout_seconds=3.0,
                ),
            )
        )
        context = InteractionContext()
        user_interaction = UserInteraction(
            items=(Message(role="user", content="Use echo."),),
        )
        context.extend(user_interaction.context_items())

        first_sample = model.sample(
            context,
            tools=environment.tool_specs,
        )
        context.extend(first_sample.context_items())
        environment_result = environment.execute_tool_calls(
            first_sample.tool_calls
        )
        context.extend(environment_result.context_items())
        final_sample = model.sample(
            context,
            tools=environment.tool_specs,
        )
        context.extend(final_sample.context_items())

        self.assertEqual(
            final_sample.last_assistant_text,
            "The tool said hello.",
        )
        second_payload = _request_payload(opener, 1)
        self.assertEqual(
            second_payload["messages"],
            [
                {"role": "user", "content": "Use echo."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"hello"}',
                            },
                        }
                    ],
                    "content": None,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "hello",
                },
            ],
        )

    def test_injected_user_message_is_encoded_after_tool_result(self):
        tool = create_inject_user_message_tool()
        call = ToolCall(tool.spec.name, "inject-1", "{}")
        context = InteractionContext((
            Message("user", "Run the experiment."), call, ModelSampleBoundary(),
        ))
        environment = Environment((tool,))
        context.extend(environment.execute_tool_calls((call,)).context_items())
        opener = _ScriptedOpener(_FakeHTTPResponse({
            "choices": [{
                "message": {"role": "assistant", "content": "received: hello world"},
                "finish_reason": "stop",
            }],
        }))
        model = ChatCompletionsModel(
            chat_endpoint(api_url="http://localhost:8000"), opener=opener,
        )
        model.sample(context, tools=environment.tool_specs)
        self.assertEqual(_request_payload(opener)["messages"][-2:], [
            {"role": "tool", "tool_call_id": call.call_id, "content": "Synthetic user message queued."},
            {"role": "user", "content": "hello world"},
        ])

    def test_sample_boundaries_separate_adjacent_assistant_messages(self):
        opener = _ScriptedOpener(
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "new answer",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
        model = ChatCompletionsModel(
            chat_endpoint(api_url="http://localhost:8000"),
            opener=opener,
        )
        context = InteractionContext(
            [
                Message(role="user", content="question"),
                UserInteractionBoundary(),
                Reasoning(content="first reasoning"),
                Message(role="assistant", content="first answer"),
                ModelSampleBoundary(),
                Reasoning(content="second reasoning"),
                Message(role="assistant", content="second answer"),
                ModelSampleBoundary(),
                Message(role="user", content="continue"),
            ]
        )

        model.sample(context)

        self.assertEqual(
            _request_payload(opener)["messages"],
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "reasoning_content": "first reasoning",
                    "content": "first answer",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "second reasoning",
                    "content": "second answer",
                },
                {"role": "user", "content": "continue"},
            ],
        )

    def test_sample_metadata_is_not_sent_to_chat_completions(self):
        opener = _ScriptedOpener(
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "answer",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
        model = ChatCompletionsModel(
            chat_endpoint(api_url="http://localhost:8000"),
            opener=opener,
        )
        context = InteractionContext(
            [
                Message(role="user", content="question"),
                UserInteractionBoundary(),
                Message(role="assistant", content="prior answer"),
                SampleMetadata(
                    usage=TokenUsage(
                        input_tokens=20,
                        output_tokens=5,
                        total_tokens=25,
                        cached_input_tokens=4,
                    )
                ),
                ModelSampleBoundary(),
                Message(role="user", content="continue"),
            ]
        )

        model.sample(context)

        self.assertEqual(
            _request_payload(opener)["messages"],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": "continue"},
            ],
        )

    def test_context_without_boundaries_keeps_adjacency_collation(self):
        opener = _ScriptedOpener(
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "new answer",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
        model = ChatCompletionsModel(
            chat_endpoint(api_url="http://localhost:8000"),
            opener=opener,
        )
        context = InteractionContext(
            [
                Message(role="user", content="question"),
                Message(role="assistant", content="first"),
                Message(role="assistant", content="second"),
                Message(role="user", content="continue"),
            ]
        )

        model.sample(context)

        self.assertEqual(
            _request_payload(opener)["messages"],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "firstsecond"},
                {"role": "user", "content": "continue"},
            ],
        )

    def test_optional_model_is_omitted(self):
        opener = _ScriptedOpener(
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
        model = ChatCompletionsModel(
            chat_endpoint(api_url="http://localhost:8000"),
            opener=opener,
        )

        model.sample(InteractionContext([Message(role="user", content="hello")]))

        self.assertNotIn("model", _request_payload(opener))

    def test_context_window_http_error_is_typed(self):
        error = urllib.error.HTTPError(
            url="http://localhost:8000/v1/chat/completions",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(
                b'{"error":{"message":"maximum context length exceeded"}}'
            ),
        )
        model = ChatCompletionsModel(
            chat_endpoint(api_url="http://localhost:8000"),
            opener=_ScriptedOpener(error),
        )

        with self.assertRaises(ModelContextWindowError):
            model.sample(InteractionContext([Message(role="user", content="hello")]))

    def test_retryable_http_statuses_use_two_retries_and_metadata(self):
        returned_error = _FakeHTTPResponse(
            {"error": {"message": "temporary"}},
            status=503,
        )
        success = _FakeHTTPResponse({
            "choices": [{
                "message": {"role": "assistant", "content": "recovered"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })
        opener = _ScriptedOpener(
            _http_error(429, headers={"Retry-After": "0.75"}),
            returned_error,
            success,
        )
        sleeps = []
        model = ChatCompletionsModel(
            chat_endpoint(api_url="https://api.example.test"),
            opener=opener,
            retry_sleep=sleeps.append,
        )

        sample = model.sample(InteractionContext((Message("user", "hello"),)))

        self.assertEqual(sample.last_assistant_text, "recovered")
        self.assertEqual(sample.request_attempts, 3)
        self.assertEqual(
            sample.recovery,
            ("http_429_retry", "http_503_retry"),
        )
        self.assertEqual(sleeps, [0.75, 0.5])
        self.assertTrue(returned_error.closed)
        self.assertTrue(success.closed)
        request_bodies = tuple(call[0].data for call in opener.calls)
        self.assertEqual(request_bodies, (request_bodies[0],) * 3)

    def test_connection_and_body_read_failures_are_retried(self):
        failed_read = _ReadFailureResponse(OSError("body disconnected"))
        success = _FakeHTTPResponse({
            "choices": [{
                "message": {"role": "assistant", "content": "recovered"},
                "finish_reason": "stop",
            }],
        })
        opener = _ScriptedOpener(
            urllib.error.URLError(OSError("connection reset")),
            failed_read,
            success,
        )
        sleeps = []
        sample = ChatCompletionsModel(
            chat_endpoint(api_url="https://api.example.test"),
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
        model = ChatCompletionsModel(
            chat_endpoint(api_url="https://api.example.test"),
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


class UserMessageOutcomeTests(unittest.TestCase):
    def test_legacy_outcomes_and_results_have_no_user_messages(self):
        self.assertEqual(ToolOutcome("ok").user_messages, ())
        self.assertEqual(ToolOutcome("failed", False).user_messages, ())
        self.assertEqual(EnvironmentResult(()).context_items(), ())

    def test_user_message_collections_are_copied_to_tuples(self):
        message = Message("user", "synthetic")
        messages = [message]
        outcome = ToolOutcome("ok", user_messages=messages)
        result = EnvironmentResult(
            (ToolResult("1", "ok"),), user_messages=messages,
        )
        messages.clear()
        self.assertEqual(outcome.user_messages, (message,))
        self.assertEqual(result.user_messages, (message,))

    def test_rejects_non_user_messages_in_outcomes_and_results(self):
        for item in (
            Message("system", "bad"),
            Message("developer", "bad"),
            Message("assistant", "bad"),
            UserInteractionBoundary(),
            ToolResult("1", "bad"),
            "hello",
        ):
            with self.subTest(item=item):
                with self.assertRaisesRegex(EnvironmentError, "user-role Message"):
                    ToolOutcome("ok", user_messages=(item,))
                with self.assertRaisesRegex(EnvironmentError, "user-role Message"):
                    EnvironmentResult(
                        (ToolResult("1", "ok"),), user_messages=(item,),
                    )

    def test_unsuccessful_outcomes_cannot_inject(self):
        messages = (Message("user", "synthetic"),)
        with self.assertRaisesRegex(EnvironmentError, "unsuccessful"):
            ToolOutcome("failed", False, user_messages=messages)
        for items in ((), (ToolResult("1", "failed", False),)):
            with self.subTest(items=items):
                with self.assertRaisesRegex(EnvironmentError, "successful tool result"):
                    EnvironmentResult(items, user_messages=messages)

    def test_result_items_remain_tool_result_only(self):
        with self.assertRaisesRegex(EnvironmentError, "only ToolResult"):
            EnvironmentResult(items=(Message("user", "synthetic"),))

    def test_projection_and_display_include_messages_without_consuming_them(self):
        item = ToolResult("1", "ok")
        message = Message("user", "synthetic")
        result = EnvironmentResult((item,), user_messages=(message,))
        self.assertEqual(result.items, (item,))
        for _ in range(2):
            self.assertEqual(result.context_items(), (item, message))
            self.assertEqual(
                tuple(item.text for item in result.display_items(
                    source_calls=(ToolCall("test", "1", "{}"),),
                )),
                ("[tool-ret]  test (1) [ok]\nok", "[user] synthetic"),
            )


class EnvironmentTests(unittest.TestCase):
    def test_environment_result_context_items_are_directly_appendable(self):
        item = ToolResult(call_id="call-1", output="done")
        result = EnvironmentResult(items=(item,))

        self.assertEqual(result.context_items(), (item,))

    def test_environment_executes_sequentially_and_preserves_order(self):
        seen = []

        def handler(arguments, *, timeout_seconds=None):
            seen.append((arguments["value"], timeout_seconds))
            return ToolOutcome(output=f"out:{arguments['value']}")

        environment = Environment(
            tools=(
                Tool(
                    spec=ToolSpec(
                        name="ordered",
                        description="Record order.",
                        parameters={"type": "object"},
                    ),
                    handler=handler,
                    timeout_seconds=2.0,
                ),
            )
        )
        calls = (
            ToolCall("ordered", "1", '{"value":"a"}'),
            ToolCall("ordered", "2", '{"value":"b"}'),
        )

        result = environment.execute_tool_calls(calls)

        self.assertEqual(seen, [("a", 2.0), ("b", 2.0)])
        self.assertEqual(
            result.items,
            (
                ToolResult("1", "out:a"),
                ToolResult("2", "out:b"),
            ),
        )

    def test_user_messages_follow_the_entire_batch_in_call_and_message_order(self):
        def handler(arguments, *, timeout_seconds=None):
            value = arguments["value"]
            return ToolOutcome(
                f"out:{value}",
                user_messages=(Message("user", value), Message("user", value + "!")),
            )

        environment = Environment((Tool(
            ToolSpec("inject", "test", {"type": "object"}), handler,
        ),))
        calls = (
            ToolCall("inject", "1", '{"value":"a"}'),
            ToolCall("missing", "2", "{}"),
            ToolCall("inject", "3", '{"value":"b"}'),
        )
        context = InteractionContext((*calls, ModelSampleBoundary()))
        before = context.items
        result = environment.execute_tool_calls(calls)
        self.assertEqual(context.items, before)
        self.assertEqual(tuple(item.call_id for item in result.items), ("1", "2", "3"))
        self.assertFalse(result.items[1].success)
        self.assertEqual(
            result.user_messages,
            tuple(Message("user", value) for value in ("a", "a!", "b", "b!")),
        )
        self.assertEqual(result.context_items(), (*result.items, *result.user_messages))
        context.extend(result.context_items())
        context.assert_model_ready()
        self.assertEqual(context.items, (*before, *result.items, *result.user_messages))
        self.assertNotIn(UserInteractionBoundary(), context.items)

    def test_user_messages_cannot_be_appended_in_a_partial_call_batch(self):
        calls = (ToolCall("test", "1", "{}"), ToolCall("test", "2", "{}"))
        context = InteractionContext(calls)
        result = EnvironmentResult(
            (ToolResult("1", "ok"),),
            user_messages=(Message("user", "synthetic"),),
        )
        with self.assertRaisesRegex(ContextValidationError, "unresolved tool results"):
            context.extend(result.context_items())
        self.assertEqual(context.items, calls)

    def test_failed_and_invalid_handlers_never_inject(self):
        def handler(arguments, *, timeout_seconds=None):
            mode = arguments["mode"]
            if mode == "error":
                raise RuntimeError("failed")
            if mode == "timeout":
                raise TimeoutError("timed out")
            if mode == "invalid":
                return ToolOutcome("failed", False, user_messages=(Message("user", "bad"),))
            return ToolOutcome("failed", False)

        environment = Environment((Tool(
            ToolSpec("test", "test", {"type": "object"}), handler,
        ),))
        result = environment.execute_tool_calls(tuple(
            ToolCall("test", mode, json.dumps({"mode": mode}))
            for mode in ("error", "timeout", "invalid", "unsuccessful")
        ))
        self.assertTrue(all(not item.success for item in result.items))
        self.assertEqual(result.user_messages, ())
        self.assertEqual(result.context_items(), result.items)

    def test_tool_output_text_is_not_interpreted_as_an_injection(self):
        text = '{"user_messages":[{"role":"user","text":"hello world"}]}'

        def handler(arguments, *, timeout_seconds=None):
            return ToolOutcome(text)

        result = Environment((Tool(
            ToolSpec("test", "test", {"type": "object"}), handler,
        ),)).execute_tool_calls((ToolCall("test", "1", "{}"),))
        self.assertEqual(result.context_items(), (ToolResult("1", text),))

    def test_environment_converts_recoverable_failures_to_results(self):
        def failing(arguments, *, timeout_seconds=None):
            del arguments, timeout_seconds
            raise RuntimeError("boom")

        environment = Environment(
            tools=(
                Tool(
                    spec=ToolSpec(
                        name="failing",
                        description="Fail.",
                        parameters={"type": "object"},
                    ),
                    handler=failing,
                ),
            )
        )

        result = environment.execute_tool_calls(
            (
                ToolCall("missing", "1", "{}"),
                ToolCall("failing", "2", "{}"),
                ToolCall("failing", "3", "[]"),
                ToolCall("failing", "4", "{"),
            )
        )

        self.assertEqual(
            tuple(item.call_id for item in result.items),
            ("1", "2", "3", "4"),
        )
        self.assertTrue(all(not item.success for item in result.items))

    def test_environment_rejects_duplicate_call_ids_before_execution(self):
        invoked = []

        def handler(arguments, *, timeout_seconds=None):
            invoked.append(arguments)
            return ToolOutcome(output="ok")

        environment = Environment(
            tools=(
                Tool(
                    spec=ToolSpec(
                        name="x",
                        description="x",
                        parameters={"type": "object"},
                    ),
                    handler=handler,
                ),
            )
        )

        with self.assertRaisesRegex(EnvironmentError, "duplicate"):
            environment.execute_tool_calls(
                (
                    ToolCall("x", "same", "{}"),
                    ToolCall("x", "same", "{}"),
                )
            )

        self.assertEqual(invoked, [])


class _ScriptedModel:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def sample(self, context, *, tools=(), sample_params=None):
        self.calls.append((context.copy(), tuple(tools), sample_params))
        if not self.outcomes:
            raise AssertionError("unexpected model sample")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _text(tokens, char="x"):
    """Text whose character estimate is exactly ``tokens``."""
    return char * (tokens * 4)


def _summary(text="Summary.", **kwargs):
    kwargs.setdefault("stop_reason", "end_turn")
    return ModelSample(items=(Message("assistant", text),), **kwargs)


def _wrapped(summary):
    return f"{COMPACTION_SUMMARY_PREFIX}{summary}{COMPACTION_SUMMARY_SUFFIX}"


def _prompt(call):
    """The user prompt of one recorded summary request."""
    context, _tools, _params = call
    instructions, message = context.model_items()
    return message.content


def _two_turns():
    """A finished first turn, then a long agentic second turn.

    Estimates: first turn 4 + 2 + 4; second request 4; call-1 6; its result
    30; "Working." 2; call-2 7; its result 30.
    """
    return InteractionContext((
        Init("session"),
        Instructions("Be brief."),
        Message("user", "First request."),
        UserInteractionBoundary(),
        Reasoning("Plan it."),
        Message("assistant", "First answer."),
        SampleMetadata(TokenUsage(total_tokens=50)),
        ModelSampleBoundary(),
        TurnSummary(sample_count=1),
        Message("user", "Second request."),
        UserInteractionBoundary(),
        ToolCall("exec_command", "call-1", '{"cmd":"ls"}'),
        SampleMetadata(TokenUsage(total_tokens=60)),
        ModelSampleBoundary(),
        ToolResult("call-1", _text(30, "r")),
        Message("assistant", "Working."),
        ToolCall("exec_command", "call-2", '{"cmd":"pwd"}'),
        SampleMetadata(TokenUsage(total_tokens=90)),
        ModelSampleBoundary(),
        ToolResult("call-2", _text(30, "s")),
    ))


class CompactionEstimateTests(unittest.TestCase):
    def test_item_estimates_are_characters_over_four_rounded_up(self):
        self.assertEqual(estimate_item_tokens(Message("user", "abcde")), 2)
        self.assertEqual(estimate_item_tokens(Message("user", (
            TextPart("abcd"), MediaPart("https://example.test/image.png"),
        ))), (4 + 4_800) // 4)
        self.assertEqual(estimate_item_tokens(Reasoning("abcd")), 1)
        self.assertEqual(estimate_item_tokens(Reasoning("", summary=("ab", "cd"))), 2)
        self.assertEqual(estimate_item_tokens(Reasoning("", content_signature="sig")), 0)
        self.assertEqual(estimate_item_tokens(ToolCall("ab", "call", "{}")), 1)
        self.assertEqual(estimate_item_tokens(ToolResult("call", "abcdefgh")), 2)
        self.assertEqual(estimate_item_tokens(Instructions("abc")), 1)
        self.assertEqual(estimate_item_tokens(OpaqueCompaction.from_messages("abcde")), 2)
        for item in (
            ModelSampleBoundary(), UserInteractionBoundary(), TurnSummary(),
            SampleMetadata(TokenUsage(total_tokens=5)), Init("session"),
            UserToolCall(ToolCall("compact", "user_1", "{}")),
            UserToolResult(ToolResult("user_1", "private text")),
        ):
            with self.subTest(item=type(item).__name__):
                self.assertEqual(estimate_item_tokens(item), 0)

    def test_pure_estimate_without_reported_usage(self):
        context = InteractionContext((
            Instructions(_text(3)),
            Message("user", _text(10)),
            Message("assistant", _text(5)),
            SampleMetadata(TokenUsage()),
            ModelSampleBoundary(),
        ))
        self.assertEqual(estimate_context_tokens(context), 18)

    def test_anchor_counts_items_after_the_latest_reported_usage(self):
        context = InteractionContext((
            Message("user", _text(10)),
            ToolCall("t", "call", "{}"),
            SampleMetadata(TokenUsage(total_tokens=1_000)),
            ModelSampleBoundary(),
            ToolResult("call", _text(7)),
        ))
        # Trailing tool results are counted on top of the reported usage.
        self.assertEqual(estimate_context_tokens(context), 1_007)
        context.extend((
            UserToolCall(ToolCall("config", "user_1", "{}")),
            UserToolResult(ToolResult("user_1", _text(50))),
            Message("assistant", _text(3)),
            SampleMetadata(TokenUsage(total_tokens=0)),
            ModelSampleBoundary(),
        ))
        # Zero usage is not an anchor, and user tools count as nothing.
        self.assertEqual(estimate_context_tokens(context), 1_010)

    def test_anchor_is_taken_after_the_latest_prefix(self):
        context = InteractionContext((
            Message("user", _text(1)),
            Message("assistant", _text(1)),
            SampleMetadata(TokenUsage(total_tokens=5_000)),
            ModelSampleBoundary(),
            ContextPrefix((Message("user", _text(8)),)),
            CompactionMetadata(TokenUsage(total_tokens=9_000), "pi"),
        ))
        self.assertEqual(estimate_context_tokens(context), 8)
        context.append(Message("user", _text(4)))
        self.assertEqual(estimate_context_tokens(context), 12)
        context.extend((
            Message("assistant", _text(2)),
            SampleMetadata(TokenUsage(total_tokens=300)),
            ModelSampleBoundary(),
        ))
        self.assertEqual(estimate_context_tokens(context), 300)

    def test_trigger_compares_the_estimate_with_the_threshold(self):
        context = InteractionContext((
            Message("user", _text(1)),
            Message("assistant", _text(1)),
            SampleMetadata(TokenUsage(total_tokens=98)),
            ModelSampleBoundary(),
            Message("user", _text(2)),
        ))
        self.assertTrue(should_auto_compact(context, 100))
        self.assertFalse(should_auto_compact(context, 101))
        # Usage alone, with nothing model-visible, never triggers.
        self.assertFalse(should_auto_compact(InteractionContext((
            SampleMetadata(TokenUsage(total_tokens=100)),
        )), 100))
        for threshold in (True, 0, -1, 1.5):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                should_auto_compact(InteractionContext(), threshold)

    def test_just_compacted_context_is_not_compacted_again(self):
        context = InteractionContext((
            Message("user", _text(100)),
            ContextPrefix((Message("user", _text(100)),)),
            CompactionMetadata(TokenUsage(), "pi"),
        ))
        self.assertEqual(estimate_context_tokens(context), 100)
        self.assertFalse(should_auto_compact(context, 50))
        context.extend((
            ModelFailure(category="timeout", message="timed out"),
            ModelSampleBoundary(),
        ))
        self.assertFalse(should_auto_compact(context, 50))
        context.append(Message("user", _text(1)))
        self.assertTrue(should_auto_compact(context, 50))

    def test_auto_compaction_due_needs_policy_threshold_and_host_ownership(self):
        class Host:
            auto_compaction_owner = "host"

        class Server:
            auto_compaction_owner = "server"

        context = InteractionContext((Message("user", _text(200)),))
        due = InteractionConfigSnapshot(auto_compact_tokens=100)
        self.assertTrue(auto_compaction_due(Host(), context, due))
        self.assertTrue(auto_compaction_due(object(), context, due))
        self.assertFalse(auto_compaction_due(Server(), context, due))
        for snapshot in (
            InteractionConfigSnapshot(auto_compact_tokens=None),
            InteractionConfigSnapshot(auto_compact_tokens=100, enable_auto_compaction=False),
            InteractionConfigSnapshot(auto_compact_tokens=201),
        ):
            with self.subTest(snapshot=snapshot):
                self.assertFalse(auto_compaction_due(Host(), context, snapshot))


class CompactionTests(unittest.TestCase):
    def test_settings_and_default_compactor(self):
        self.assertEqual(DEFAULT_KEEP_RECENT_TOKENS, 20_000)
        self.assertEqual(
            CompactionSettings(),
            CompactionSettings(mode="pi", keep_recent_tokens=20_000, max_output_tokens=None),
        )
        for kwargs in (
            {"mode": "remote"}, {"keep_recent_tokens": -1},
            {"keep_recent_tokens": True}, {"max_output_tokens": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CompactionSettings(**kwargs)
        model = _ScriptedModel()
        compactor = create_default_compactor(model)
        self.assertIsInstance(compactor, PiCompactor)
        self.assertEqual(
            (compactor.keep_recent_tokens, compactor.max_output_tokens),
            (20_000, None),
        )
        configured = create_default_compactor(model, CompactionSettings(
            mode="provider", keep_recent_tokens=0, max_output_tokens=77,
        ))
        # Provider mode without remote compaction, as for Messages.
        self.assertIsInstance(configured, PiCompactor)
        self.assertEqual(
            (configured.keep_recent_tokens, configured.max_output_tokens),
            (0, 77),
        )
        for kwargs in ({"keep_recent_tokens": -1}, {"max_output_tokens": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PiCompactor(model, **kwargs)
        with self.assertRaises(TypeError):
            PiCompactor(object())

    def test_cut_points_skip_tool_results_and_later_user_messages_in_a_run(self):
        span = (
            Message("user", "one"),                      # 0: cut point
            UserInteractionBoundary(),
            Message("user", "two"),                      # 2: same run
            Reasoning("think"),                          # 3: first output
            Message("assistant", "text"),
            ToolCall("t", "call-1", "{}"),
            SampleMetadata(TokenUsage()),
            ModelSampleBoundary(),
            ToolResult("call-1", "result"),              # 8: never
            ToolCall("t", "call-2", "{}"),               # 9: next sample
            ModelSampleBoundary(),
            ToolResult("call-2", "result"),
            Message("user", "three"),                    # 12: new run
            Message("assistant", "answer"),              # 13: first output
        )
        self.assertEqual(_cut_points(span), [0, 3, 9, 12, 13])

    def test_turn_boundary_cut_summarizes_history_and_keeps_the_tail(self):
        model = _ScriptedModel(_summary(
            "History summary.",
            usage=TokenUsage(80, 20, 100, 10),
            provider_session_id="provider-session",
            provider_turn_id="provider-turn",
            provider_turn_state="provider-state",
            request_attempts=2,
            recovery=("credential_reload",),
        ))
        context = _two_turns()
        before = context.items

        with mock.patch(
            "pythia.interaction.compaction.perf_counter",
            side_effect=(100.0, 112.5),
        ):
            result = PiCompactor(model, keep_recent_tokens=78).compact(context)

        self.assertEqual(context.items, before)
        self.assertEqual(len(model.calls), 1)
        prompt = _prompt(model.calls[0])
        self.assertEqual(prompt, (
            "<conversation>\n"
            "[User]: First request.\n\n"
            "[Assistant thinking]: Plan it.\n\n"
            "[Assistant]: First answer.\n"
            "</conversation>\n\n" + SUMMARIZATION_PROMPT
        ))
        checkpoint, = result.items
        self.assertEqual(checkpoint.prefix_items, (
            Instructions("Be brief."),
            Message("user", _wrapped("History summary.")),
            # Metadata leaves the tail; boundaries stay for the encoders.
            Message("user", "Second request."),
            UserInteractionBoundary(),
            ToolCall("exec_command", "call-1", '{"cmd":"ls"}'),
            ModelSampleBoundary(),
            ToolResult("call-1", _text(30, "r")),
            Message("assistant", "Working."),
            ToolCall("exec_command", "call-2", '{"cmd":"pwd"}'),
            ModelSampleBoundary(),
            ToolResult("call-2", _text(30, "s")),
        ))
        self.assertEqual(result.protocol, "pi")
        self.assertEqual(result.usage, TokenUsage(80, 20, 100, 10))
        self.assertEqual(result.elapsed_seconds, 12.5)
        self.assertEqual(result.request_attempts, 2)
        self.assertEqual(result.recovery, ("credential_reload",))
        self.assertEqual(result.context_items()[-1], CompactionMetadata(
            usage=TokenUsage(80, 20, 100, 10),
            protocol="pi",
            provider_session_id="provider-session",
            provider_turn_id="provider-turn",
            provider_turn_state="provider-state",
            elapsed_seconds=12.5,
            request_attempts=2,
            recovery=("credential_reload",),
        ))
        context.extend(result.context_items())
        self.assertEqual(context.model_items(), checkpoint.prefix_items)
        self.assertEqual(context.items[:len(before)], before)
        self.assertTrue(is_compaction_summary(checkpoint.prefix_items[1]))

    def test_split_turn_sends_two_requests_and_merges_them(self):
        model = _ScriptedModel(
            _summary(
                "History summary.",
                usage=TokenUsage(10, 2, 12, 3),
                provider_turn_id="first",
                recovery=("http_500_retry",),
            ),
            _summary(
                "Turn summary.",
                usage=TokenUsage(20, 4, 24, 5),
                provider_turn_id="second",
                request_attempts=2,
                recovery=("connection_retry", "http_429_retry"),
            ),
        )
        # The budget is reached at call-1's result, so the cut falls on the
        # next sample's first output item, inside the second turn.
        result = PiCompactor(model, keep_recent_tokens=40).compact(_two_turns())

        self.assertEqual(len(model.calls), 2)
        history, turn = (_prompt(call) for call in model.calls)
        self.assertTrue(history.startswith("<conversation>\n[User]: First request."))
        self.assertTrue(history.endswith(SUMMARIZATION_PROMPT))
        self.assertNotIn("Second request.", history)
        self.assertEqual(turn, (
            "# Conversation\n"
            "[User]: Second request.\n\n"
            '[Assistant tool calls]: exec_command(cmd="ls")\n\n'
            f"[Tool result]: {_text(30, 'r')}\n\n"
            "# Instructions\n" + TURN_PREFIX_SUMMARIZATION_PROMPT
        ))
        checkpoint, = result.items
        self.assertEqual(checkpoint.prefix_items, (
            Instructions("Be brief."),
            Message("user", _wrapped(
                "History summary.\n\n---\n\n**Turn Context (split turn):**\n\n"
                "Turn summary."
            )),
            Message("assistant", "Working."),
            ToolCall("exec_command", "call-2", '{"cmd":"pwd"}'),
            ModelSampleBoundary(),
            ToolResult("call-2", _text(30, "s")),
        ))
        self.assertEqual(result.usage, TokenUsage(30, 6, 36, 8))
        self.assertEqual(result.request_attempts, 3)
        self.assertEqual(
            result.recovery,
            ("http_500_retry", "connection_retry", "http_429_retry"),
        )
        self.assertEqual(result.provider_turn_id, "second")
        InteractionContext(result.context_items())

    def test_split_turn_without_history_uses_previous_summary_or_placeholder(self):
        for previous, history_text in ((None, "No prior history."), ("Old.", "Old.")):
            with self.subTest(previous=previous):
                items = [Instructions("Rules.")]
                if previous is not None:
                    items.append(Message("user", _wrapped(previous)))
                items.extend((
                    Message("user", "Only request."),
                    ToolCall("t", "call-1", "{}"),
                    ModelSampleBoundary(),
                    ToolResult("call-1", _text(10)),
                    ToolCall("t", "call-2", "{}"),
                    ModelSampleBoundary(),
                    ToolResult("call-2", _text(10)),
                ))
                model = _ScriptedModel(_summary("Turn summary."))
                result = PiCompactor(model, keep_recent_tokens=10).compact(
                    InteractionContext(items),
                    instructions="  the parser  ",
                )
                self.assertEqual(len(model.calls), 1)
                prompt = _prompt(model.calls[0])
                self.assertTrue(prompt.startswith("# Conversation\n[User]: Only request."))
                # Without a history request, focus goes to the turn prefix.
                self.assertTrue(prompt.endswith(
                    TURN_PREFIX_SUMMARIZATION_PROMPT + "\n\nAdditional focus: the parser"
                ))
                self.assertEqual(result.items[0].prefix_items[:2], (
                    Instructions("Rules."),
                    Message("user", _wrapped(
                        f"{history_text}\n\n---\n\n**Turn Context (split turn):**"
                        "\n\nTurn summary."
                    )),
                ))

    def test_budget_reached_inside_tool_results_keeps_their_sample(self):
        model = _ScriptedModel(_summary("History."), _summary("Turn."))
        result = PiCompactor(model, keep_recent_tokens=100).compact(InteractionContext((
            Message("user", "Earlier."),
            Message("assistant", "Earlier answer."),
            ModelSampleBoundary(),
            Message("user", "Run it."),
            ToolCall("t", "call-1", "{}"),
            ModelSampleBoundary(),
            ToolResult("call-1", _text(500)),
        )))
        self.assertEqual(result.items[0].prefix_items, (
            Message("user", _wrapped(
                "History.\n\n---\n\n**Turn Context (split turn):**\n\nTurn."
            )),
            ToolCall("t", "call-1", "{}"),
            ModelSampleBoundary(),
            ToolResult("call-1", _text(500)),
        ))

    def test_zero_keep_summarizes_everything(self):
        model = _ScriptedModel(_summary("All of it."))
        result = PiCompactor(model, keep_recent_tokens=0).compact(_two_turns())
        prompt = _prompt(model.calls[0])
        self.assertIn("[User]: Second request.", prompt)
        self.assertIn('exec_command(cmd="pwd")', prompt)
        self.assertEqual(result.items[0].prefix_items, (
            Instructions("Be brief."),
            Message("user", _wrapped("All of it.")),
        ))

    def test_nothing_to_compact_sends_no_request(self):
        model = _ScriptedModel()
        for keep, context, reason in (
            (DEFAULT_KEEP_RECENT_TOKENS, _two_turns(), "fits"),
            (10, InteractionContext((Message("user", _text(50)),)), "precedes"),
            (0, InteractionContext((Instructions("Rules."), Message("user", _wrapped("Old.")))), "nothing new"),
        ):
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(NothingToCompact, reason):
                    PiCompactor(model, keep_recent_tokens=keep).compact(context)
        self.assertEqual(model.calls, [])
        self.assertTrue(issubclass(NothingToCompact, CompactionError))

    def test_repeated_compaction_summarizes_previously_kept_items(self):
        model = _ScriptedModel(_summary("First summary."), _summary("Second summary."))
        context = _two_turns()
        context.extend(PiCompactor(model, keep_recent_tokens=78).compact(context).context_items())
        context.extend((
            Message("assistant", "Done."),
            SampleMetadata(TokenUsage(total_tokens=40)),
            ModelSampleBoundary(),
            Message("user", "Third request."),
            UserInteractionBoundary(),
            Message("assistant", _text(60, "t")),
            ModelSampleBoundary(),
        ))

        # 60 + 4 tokens reach the budget at the third request.
        result = PiCompactor(model, keep_recent_tokens=62).compact(context)

        # The window moved past the previously kept second turn, which is
        # summarized now, into the previous summary.
        self.assertEqual(_prompt(model.calls[1]), (
            "<conversation>\n"
            "[User]: Second request.\n\n"
            '[Assistant tool calls]: exec_command(cmd="ls")\n\n'
            f"[Tool result]: {_text(30, 'r')}\n\n"
            "[Assistant]: Working.\n\n"
            '[Assistant tool calls]: exec_command(cmd="pwd")\n\n'
            f"[Tool result]: {_text(30, 's')}\n\n"
            "[Assistant]: Done.\n"
            "</conversation>\n\n"
            "<previous-summary>\nFirst summary.\n</previous-summary>\n\n"
            + UPDATE_SUMMARIZATION_PROMPT
        ))
        self.assertEqual(result.items[0].prefix_items, (
            Instructions("Be brief."),
            Message("user", _wrapped("Second summary.")),
            Message("user", "Third request."),
            UserInteractionBoundary(),
            Message("assistant", _text(60, "t")),
            ModelSampleBoundary(),
        ))

    def test_previous_summary_sources(self):
        tail = (
            Message("user", "Next request."),
            Message("assistant", "Next answer."),
            ModelSampleBoundary(),
        )
        for label, source, previous in (
            ("new-style", Message("user", _wrapped("Pi summary.")), "Pi summary."),
            (
                "old-style",
                Message("user", f"{_LEGACY_SUMMARY_PREFIX}\nPrompt summary."),
                "Prompt summary.",
            ),
        ):
            with self.subTest(label=label):
                model = _ScriptedModel(_summary("Updated."))
                context = InteractionContext((
                    Instructions("Rules."),
                    ContextPrefix((Instructions("Rules."), source)),
                    *tail,
                ))
                result = PiCompactor(model, keep_recent_tokens=0).compact(context)
                prompt = _prompt(model.calls[0])
                self.assertEqual(prompt, (
                    "<conversation>\n[User]: Next request.\n\n"
                    "[Assistant]: Next answer.\n</conversation>\n\n"
                    f"<previous-summary>\n{previous}\n</previous-summary>\n\n"
                    + UPDATE_SUMMARIZATION_PROMPT
                ))
                self.assertEqual(result.items[0].prefix_items, (
                    Instructions("Rules."),
                    Message("user", _wrapped("Updated.")),
                ))
                self.assertTrue(is_compaction_summary(source))
        self.assertFalse(is_compaction_summary(Message("user", "An ordinary request.")))
        self.assertFalse(is_compaction_summary(Message("assistant", _wrapped("Not user."))))

    def test_messages_compaction_block_is_the_previous_summary(self):
        model = _ScriptedModel(_summary("Updated."))
        context = InteractionContext((
            Instructions("Rules."),
            Message("user", "Ignored by Anthropic."),
            Message("assistant", "Also ignored."),
            ModelSampleBoundary(),
            Message("user", "Current request."),
            OpaqueCompaction.from_messages("Server summary."),
            Message("assistant", "Continued."),
            ModelSampleBoundary(),
            Message("user", "Follow up."),
            Message("assistant", _text(30)),
            ModelSampleBoundary(),
        ))
        # 30 + 3 tokens reach the budget at "Follow up.", a turn boundary.
        result = PiCompactor(model, keep_recent_tokens=32).compact(context)
        prompt = _prompt(model.calls[0])
        self.assertEqual(prompt, (
            "<conversation>\n[Assistant]: Continued.\n</conversation>\n\n"
            "<previous-summary>\nServer summary.\n</previous-summary>\n\n"
            + UPDATE_SUMMARIZATION_PROMPT
        ))
        self.assertEqual(result.items[0].prefix_items, (
            Instructions("Rules."),
            Message("user", _wrapped("Updated.")),
            Message("user", "Follow up."),
            Message("assistant", _text(30)),
            ModelSampleBoundary(),
        ))

    def test_responses_checkpoints_are_carried_verbatim(self):
        model = _ScriptedModel(_summary("Summary."))
        checkpoint = OpaqueCompaction.from_responses("encrypted-checkpoint")
        context = InteractionContext((
            Instructions("Rules."),
            ContextPrefix((
                Instructions("Rules."),
                Message("user", "Retained request."),
                checkpoint,
            )),
            Message("assistant", "Continued."),
            ModelSampleBoundary(),
            Message("user", "Next request."),
            Message("assistant", _text(30)),
            ModelSampleBoundary(),
        ))
        result = PiCompactor(model, keep_recent_tokens=32).compact(context)
        self.assertEqual(_prompt(model.calls[0]), (
            "<conversation>\n[User]: Retained request.\n\n"
            "[Assistant]: Continued.\n</conversation>\n\n" + SUMMARIZATION_PROMPT
        ))
        self.assertEqual(result.items[0].prefix_items, (
            Instructions("Rules."),
            checkpoint,
            Message("user", _wrapped("Summary.")),
            Message("user", "Next request."),
            Message("assistant", _text(30)),
            ModelSampleBoundary(),
        ))

    def test_transcript_labels_grouping_arguments_truncation_and_media(self):
        model = _ScriptedModel(_summary())
        PiCompactor(model, keep_recent_tokens=0).compact(InteractionContext((
            Init("session"),
            Instructions("Rules."),
            Message("developer", "Leading note stays in the prefix."),
            Message("user", (
                TextPart("Look at this:"),
                MediaPart("data:image/png;base64,AAAA"),
            )),
            UserInteractionBoundary(),
            Reasoning("", summary=("Summary one.", "Summary two.")),
            Reasoning("", content_signature="signature-only"),
            Message("assistant", "Let me check."),
            ToolCall(
                "exec_command", "call-1",
                '{"cmd":"echo ü","yield_time_ms":10000,"env":{"A":[1,2]}}',
            ),
            ToolCall("apply_patch", "call-2", "*** Begin Patch"),
            SampleMetadata(TokenUsage(total_tokens=5)),
            ModelSampleBoundary(),
            ToolResult("call-1", "a" * 2_050),
            ToolResult("call-2", "", success=False),
            Message("system", "Mid-span system note."),
            Message("developer", "Mid-span developer note."),
            Reasoning("Direct thinking."),
            Message("assistant", "Part one."),
            Message("assistant", "Part two."),
            ModelFailure(category="stream_closed", message="stream closed"),
            ModelSampleBoundary(),
            TurnSummary(sample_count=2),
        )))
        self.assertEqual(_prompt(model.calls[0]), (
            "<conversation>\n"
            "[User]: Look at this:\n[image]\n\n"
            "[Assistant thinking]: Summary one.\nSummary two.\n\n"
            "[Assistant]: Let me check.\n\n"
            '[Assistant tool calls]: exec_command(cmd="echo ü", '
            'yield_time_ms=10000, env={"A":[1,2]}); '
            "apply_patch(*** Begin Patch)\n\n"
            f"[Tool result]: {'a' * 2_000}\n\n[... 50 more characters truncated]\n\n"
            "[System]: Mid-span system note.\n\n"
            "[Developer]: Mid-span developer note.\n\n"
            "[Assistant thinking]: Direct thinking.\n\n"
            "[Assistant]: Part one.\nPart two.\n"
            "</conversation>\n\n" + SUMMARIZATION_PROMPT
        ))

    def test_request_replaces_the_effective_context_and_inherits_turn_params(self):
        turn = SampleParams(
            max_output_tokens=900, temperature=0.3, top_p=0.9, stop=("END",),
            seed=7, enable_auto_compaction=True, auto_compact_tokens=5_000,
            extra={"thinking": {"type": "adaptive"}},
        )
        for budget, sample_params, expected in (
            (None, turn, SampleParams(
                max_output_tokens=900, temperature=0.3, top_p=0.9, stop=("END",),
                seed=7, enable_auto_compaction=False,
                extra={"thinking": {"type": "adaptive"}},
            )),
            (321, turn, SampleParams(
                max_output_tokens=321, temperature=0.3, top_p=0.9, stop=("END",),
                seed=7, enable_auto_compaction=False,
                extra={"thinking": {"type": "adaptive"}},
            )),
            (None, None, SampleParams(enable_auto_compaction=False)),
        ):
            with self.subTest(budget=budget, sample_params=sample_params):
                model = _ScriptedModel(_summary())
                context = _two_turns()
                PiCompactor(model, keep_recent_tokens=78, max_output_tokens=budget).compact(
                    context,
                    tools=(ToolSpec("exec_command", "Run.", {"type": "object"}),),
                    sample_params=sample_params,
                )
                request, tools, params = model.calls[0]
                self.assertEqual(params, expected)
                self.assertEqual(tools, ())
                # The raw log is kept, so Init and metadata still identify
                # the provider session; the model sees only the request.
                self.assertEqual(request.items[:-1], context.items)
                self.assertIsInstance(request.items[-1], ContextPrefix)
                self.assertEqual(request.model_items(), (
                    Instructions(SUMMARIZATION_SYSTEM_PROMPT),
                    Message("user", _prompt(model.calls[0])),
                ))

    def test_focus_text_goes_to_the_history_request(self):
        model = _ScriptedModel(_summary("History."), _summary("Turn."))
        PiCompactor(model, keep_recent_tokens=40).compact(
            _two_turns(), instructions="Keep file paths.",
        )
        history, turn = (_prompt(call) for call in model.calls)
        self.assertTrue(history.endswith(
            SUMMARIZATION_PROMPT + "\n\nAdditional focus: Keep file paths."
        ))
        self.assertNotIn("Additional focus", turn)
        with self.assertRaises(TypeError):
            PiCompactor(model).compact(_two_turns(), instructions=3)

    def test_response_checks_reject_incomplete_summaries(self):
        for label, sample, pattern in (
            ("tool calls", ModelSample(items=(
                Message("assistant", "Summary."), ToolCall("t", "call", "{}"),
            )), "returned tool calls"),
            ("max_tokens", _summary(stop_reason="max_tokens"), "stop_reason max_tokens"),
            ("refusal", _summary(stop_reason="refusal"), "stop_reason refusal"),
            ("empty", ModelSample(items=(
                Reasoning("thinking only"), Message("assistant", "  "),
            )), "returned no text"),
        ):
            with self.subTest(label=label):
                model = _ScriptedModel(sample)
                with self.assertRaisesRegex(CompactionError, pattern) as raised:
                    PiCompactor(model, keep_recent_tokens=0).compact(_two_turns())
                self.assertIn("summary request for the history", str(raised.exception))
                self.assertNotIsInstance(raised.exception, NothingToCompact)

    def test_all_text_blocks_are_joined(self):
        model = _ScriptedModel(ModelSample(items=(
            Message("assistant", "## Goal"),
            Reasoning("aside"),
            Message("assistant", "Finish it."),
        ), stop_reason="end_turn"))
        result = PiCompactor(model, keep_recent_tokens=0).compact(_two_turns())
        self.assertEqual(
            result.items[0].prefix_items[1],
            Message("user", _wrapped("## Goal\nFinish it.")),
        )

    def test_context_window_errors_name_the_part_and_the_settings(self):
        provider_text = "prompt is too long: 1234567 tokens > 1000000 maximum"
        for failing, part, count in (
            (0, "the history", 3),
            (1, "the earlier part of the current turn", 3),
        ):
            with self.subTest(part=part):
                error = ModelContextWindowError(provider_text)
                outcomes = [_summary("History."), _summary("Turn.")]
                outcomes[failing] = error
                model = _ScriptedModel(*outcomes)
                with self.assertRaises(CompactionContextWindowError) as raised:
                    PiCompactor(model, keep_recent_tokens=40).compact(_two_turns())
                message = str(raised.exception)
                self.assertIs(raised.exception.__cause__, error)
                self.assertIsInstance(raised.exception, CompactionError)
                tokens = (len(SUMMARIZATION_SYSTEM_PROMPT) + len(_prompt(model.calls[failing])) + 3) // 4
                self.assertTrue(message.startswith(
                    f"summary request for {part} ({count} items, ~{tokens:,} "
                    "estimated tokens) exceeded the model's context window"
                ), message)
                self.assertIn("compaction_max_output_tokens", message)
                self.assertIn("compaction_keep_recent_tokens", message)
                self.assertNotIn("1234567", message)
                self.assertNotIn("\n", message)
                self.assertLessEqual(len(message), 512)
                self.assertEqual(len(model.calls), failing + 1)

    def test_rejects_unready_contexts_and_invalid_arguments(self):
        compactor = PiCompactor(_ScriptedModel())
        with self.assertRaisesRegex(CompactionError, "unresolved tool calls"):
            compactor.compact(InteractionContext((ToolCall("t", "pending", "{}"),)))
        with self.assertRaises(TypeError):
            compactor.compact(_two_turns().items)
        with self.assertRaises(TypeError):
            compactor.compact(_two_turns(), sample_params={})

    def test_non_sample_response_is_rejected(self):
        model = _ScriptedModel(object())
        with self.assertRaisesRegex(CompactionError, "expected ModelSample"):
            PiCompactor(model, keep_recent_tokens=0).compact(_two_turns())


if __name__ == "__main__":
    unittest.main()
