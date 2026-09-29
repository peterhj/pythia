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
from pythia.interaction import CompactionError
from pythia.interaction import CompactionMetadata
from pythia.interaction import ContextPrefix
from pythia.interaction import ContextValidationError
from pythia.interaction import DEFAULT_COMPACTION_MAX_OUTPUT_TOKENS
from pythia.interaction import DEFAULT_REQUEST_TIMEOUT_SECONDS
from pythia.interaction import DEFAULT_SUMMARY_PREFIX
from pythia.interaction import Environment
from pythia.interaction import EnvironmentError
from pythia.interaction import EnvironmentResult
from pythia.interaction import Message
from pythia.interaction import ModelConfigurationError
from pythia.interaction import InteractionContext
from pythia.interaction import ModelContextWindowError
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import ModelTimeoutError
from pythia.interaction import ModelTransportError
from pythia.interaction import PromptSummarizingCompactor
from pythia.interaction import Reasoning
from pythia.interaction import SampleParams
from pythia.interaction import TokenUsage
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolOutcome
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import SampleMetadata
from pythia.interaction import UserInteraction
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import USER_AGENT
from pythia.interaction import should_auto_compact
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

    def test_auto_compaction_uses_latest_uncompacted_sample_usage(self):
        low = SampleMetadata(TokenUsage(total_tokens=99))
        high = SampleMetadata(TokenUsage(total_tokens=100))
        self.assertFalse(
            should_auto_compact(InteractionContext((low,)), 100)
        )
        self.assertTrue(
            should_auto_compact(InteractionContext((low, high)), 100)
        )

        prefix = ContextPrefix((Message("user", "summary"),))
        compacted = InteractionContext((low, high, prefix))
        self.assertFalse(should_auto_compact(compacted, 100))
        compacted.append(CompactionMetadata(
            TokenUsage(total_tokens=101),
            "responses_compaction_v2",
        ))
        self.assertFalse(should_auto_compact(compacted, 100))
        compacted.append(SampleMetadata(TokenUsage(total_tokens=100)))
        self.assertTrue(should_auto_compact(compacted, 100))

        for threshold in (True, 0, -1, 1.5):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                should_auto_compact(InteractionContext(), threshold)

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


class CompactionTests(unittest.TestCase):
    def test_prompt_compaction_output_limit_can_be_overridden(self):
        model = mock.Mock()
        options = SampleParams(max_output_tokens=321)

        compactor = PromptSummarizingCompactor(
            model, sample_params=options,
        )

        self.assertIs(compactor._sample_params, options)

    def test_prompt_compactor_returns_append_only_checkpoint(self):
        usage = TokenUsage(input_tokens=80, output_tokens=20, total_tokens=100)
        model = _ScriptedModel(
            ModelSample(
                items=(Message(role="assistant", content="Condensed work."),),
                stop_reason="end_turn",
                usage=usage,
                provider_session_id="provider-session",
                provider_turn_id="provider-turn",
                provider_turn_state="provider-state",
                request_attempts=2,
                recovery=("credential_reload",),
            )
        )
        context = InteractionContext(
            [
                Message(role="system", content="Base instructions."),
                Message(role="user", content="First request."),
                Message(role="assistant", content="First answer."),
                Message(
                    role="user",
                    content=f"{DEFAULT_SUMMARY_PREFIX}\nOld summary.",
                ),
                Message(role="user", content="Latest request."),
                Message(role="assistant", content="Latest answer."),
            ]
        )
        before = context.items
        compactor = PromptSummarizingCompactor(model)

        with mock.patch(
            "pythia.interaction.compaction.perf_counter",
            side_effect=(100.0, 112.5),
        ):
            result = compactor.compact(context)

        self.assertEqual(context.items, before)
        self.assertEqual(result.usage, usage)
        self.assertEqual(result.protocol, "prompt_summarization")
        self.assertEqual(result.elapsed_seconds, 12.5)
        self.assertEqual(result.request_attempts, 2)
        self.assertEqual(result.recovery, ("credential_reload",))
        self.assertEqual(len(result.items), 1)
        checkpoint = result.items[0]
        self.assertIsInstance(checkpoint, ContextPrefix)
        assert isinstance(checkpoint, ContextPrefix)
        self.assertEqual(
            checkpoint.prefix_items,
            (
                Message(role="system", content="Base instructions."),
                Message(role="user", content="First request."),
                Message(role="user", content="Latest request."),
                Message(
                    role="user",
                    content=f"{DEFAULT_SUMMARY_PREFIX}\nCondensed work.",
                ),
            ),
        )
        temporary_context, tools, options = model.calls[0]
        self.assertEqual(tools, ())
        self.assertEqual(
            options,
            SampleParams(
                temperature=0.0,
                max_output_tokens=DEFAULT_COMPACTION_MAX_OUTPUT_TOKENS,
            ),
        )
        self.assertEqual(DEFAULT_COMPACTION_MAX_OUTPUT_TOKENS, 2_000)
        self.assertNotEqual(temporary_context.items, context.items)
        self.assertIn("CONTEXT CHECKPOINT COMPACTION", temporary_context[-1].content)

        metadata = result.context_items()[-1]
        self.assertEqual(
            metadata,
            CompactionMetadata(
                usage=usage,
                protocol="prompt_summarization",
                provider_session_id="provider-session",
                provider_turn_id="provider-turn",
                provider_turn_state="provider-state",
                elapsed_seconds=12.5,
                request_attempts=2,
                recovery=("credential_reload",),
            ),
        )
        context.extend(result.context_items())
        self.assertEqual(context.model_items(), checkpoint.prefix_items)
        self.assertEqual(context.items[: len(before)], before)

    def test_prompt_compactor_fits_only_temporary_request(self):
        model = _ScriptedModel(
            ModelContextWindowError("too large"),
            ModelSample(
                items=(Message(role="assistant", content="summary"),),
            ),
        )
        context = InteractionContext(
            [
                Message(role="system", content="instructions"),
                Message(role="user", content="old"),
                Message(role="assistant", content="old answer"),
                Message(role="user", content="new"),
                Message(role="assistant", content="new answer"),
            ]
        )
        before = context.items
        compactor = PromptSummarizingCompactor(model)

        result = compactor.compact(context)

        self.assertEqual(context.items, before)
        self.assertEqual(len(model.calls), 2)
        self.assertLess(len(model.calls[1][0]), len(model.calls[0][0]))
        self.assertIsInstance(result.items[0], ContextPrefix)
        self.assertEqual(result.request_attempts, 2)
        self.assertEqual(result.recovery, ("context_window_trim",))

    def test_prompt_compactor_rejects_tool_calls(self):
        model = _ScriptedModel(
            ModelSample(
                items=(
                    ToolCall(
                        name="unexpected",
                        call_id="call-1",
                        arguments_json="{}",
                    ),
                ),
            )
        )
        compactor = PromptSummarizingCompactor(model)

        with self.assertRaisesRegex(CompactionError, "must not contain tool calls"):
            compactor.compact(
                InteractionContext([Message(role="user", content="hello")])
            )


if __name__ == "__main__":
    unittest.main()
