"""ResponsesModel: the public Responses API route, never Codex auth (offline)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from pythia.interaction import (
    BUILTIN_MODEL_CATALOG, CodexResponsesModel, CompactionError, ConfigError,
    ContextPrefix, Init, InteractionConfig, InteractionConfigSnapshot,
    InteractionContext, Instructions, Message, ModelAuthenticationError,
    ModelConfigurationError, ModelSampleBoundary, PiCompactor, ResponsesModel,
    ResponsesOpaqueCompactor, ToolSpec, TurnSummary, UserInteractionBoundary,
    create_default_compactor, parse_model_catalog,
)
from pythia.interaction import responses
from pythia_test.interaction_helpers import responses_endpoint
from pythia_test.test_responses import (
    _FakeSSEResponse, _ScriptedOpener, _completed_event, _http_error,
    _message_event, _request_headers, _request_payload,
)


CATALOG = parse_model_catalog('''[catalog]
version = 4
[model.public-test]
endpoint.api = responses
endpoint.url = https://api.example.test/v1/responses
endpoint.model = public-wire
endpoint.auth = env:PUBLIC_TEST_KEY
limits.auto_compact_context_tokens = 90000
limits.max_context_tokens = 128000
extra_sample_params.truncation = "auto"
''')
TOOL_FIELDS = {"tools", "tool_choice", "parallel_tool_calls"}


def _ok(text="done"):
    return _FakeSSEResponse(_message_event(0, text), _completed_event())


def _context():
    return InteractionContext((Instructions("Be brief."), Message("user", "hello")))


def _anonymous_binding():
    return BUILTIN_MODEL_CATALOG.bind(
        "responses", "local-wire",
        endpoint_url="http://127.0.0.1:9/v1/responses", endpoint_auth="none",
    )


class ResponsesModelConstructionTests(unittest.TestCase):
    def test_environment_supplied_and_anonymous_auth(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-1"}):
            opener = _ScriptedOpener(_ok(), _ok())
            model = ResponsesModel(
                binding=BUILTIN_MODEL_CATALOG.bind("responses", "gpt-test"), opener=opener,
            )
            self.assertEqual(model.binding.endpoint.auth, "env:OPENAI_API_KEY")
            model.sample(_context())
            os.environ["OPENAI_API_KEY"] = "env-key-2"  # Reread before each sample.
            model.sample(_context())
        self.assertEqual(opener.calls[0][0].full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(
            [_request_headers(opener, index)["authorization"] for index in (0, 1)],
            ["Bearer env-key-1", "Bearer env-key-2"],
        )

        supplied = BUILTIN_MODEL_CATALOG.bind("responses", "gpt-test", endpoint_auth="supplied")
        opener = _ScriptedOpener(_ok())
        ResponsesModel(binding=supplied, api_key=" literal-key ", opener=opener).sample(_context())
        self.assertEqual(_request_headers(opener)["authorization"], "Bearer literal-key")

        opener = _ScriptedOpener(_ok())
        model = ResponsesModel(binding=_anonymous_binding(), opener=opener)
        self.assertIsNone(model.endpoint.bearer_token)
        model.sample(_context())
        self.assertNotIn("authorization", _request_headers(opener))
        self.assertEqual(opener.calls[0][0].full_url, "http://127.0.0.1:9/v1/responses")

        with mock.patch.dict(os.environ, {}, clear=True), \
                self.assertRaisesRegex(ModelConfigurationError, "OPENAI_API_KEY"):
            ResponsesModel(binding=BUILTIN_MODEL_CATALOG.bind("responses", "gpt-test"))

    def test_endpoint_construction_uses_the_endpoint_credential(self):
        endpoint = responses_endpoint(
            api_url="https://api.example.test/v1", model="wire", bearer_token="endpoint-key",
        )
        opener = _ScriptedOpener(_ok())
        model = ResponsesModel(endpoint, opener=opener)
        self.assertIs(model.endpoint, endpoint)
        model.sample(_context())
        self.assertEqual(_request_headers(opener)["authorization"], "Bearer endpoint-key")

    def test_codex_routes_conflicting_options_and_bad_keys_are_rejected(self):
        generic = responses_endpoint(api_url="https://x.test/v1", model="m", bearer_token="k")
        codex = responses_endpoint(
            api_url="https://x.test/v1", model="m", bearer_token="k", api_provider="codex",
        )
        environment = BUILTIN_MODEL_CATALOG.bind("responses", "gpt-test")
        supplied = BUILTIN_MODEL_CATALOG.bind("responses", "gpt-test", endpoint_auth="supplied")
        codex_binding = BUILTIN_MODEL_CATALOG.bind(
            "codex", "codex-gpt-6.1-sol", endpoint_auth="none",
        )
        for kwargs, error, pattern in (
            ({"binding": codex_binding}, ModelConfigurationError, "requires a responses binding"),
            ({"endpoint": codex}, ModelConfigurationError, "requires a responses endpoint"),
            ({"endpoint": generic, "api_key": "k"}, ModelConfigurationError, "options: api_key$"),
            ({"endpoint": generic, "binding": environment, "request_timeout_seconds": 5},
             ModelConfigurationError, "options: binding, request_timeout_seconds$"),
            ({"endpoint": object()}, TypeError, "endpoint must be StreamingResponsesEndpoint"),
            ({}, TypeError, "binding must be ModelBinding"),
            ({"binding": environment, "api_key": "k"}, ModelConfigurationError,
             "api_key requires endpoint auth 'supplied'"),
            ({"binding": supplied}, ModelConfigurationError, "pass api_key"),
            ({"binding": supplied, "api_key": b"k"}, TypeError, "api_key must be a string"),
            ({"binding": supplied, "api_key": "  "}, ModelConfigurationError, "nonempty"),
            ({"binding": supplied, "api_key": "two words"}, ModelConfigurationError, "whitespace"),
        ):
            with self.subTest(options=sorted(kwargs)), self.assertRaisesRegex(error, pattern):
                ResponsesModel(**kwargs)
        # Codex credential, OAuth, and identifier options do not exist here.
        for option in ("auth", "auth_opener", "identifier_factory"):
            with self.subTest(option=option), self.assertRaises(TypeError):
                ResponsesModel(binding=supplied, api_key="k", **{option: None})


class ResponsesModelIsolationTests(unittest.TestCase):
    def setUp(self):
        for name in ("_resolve_auth_file", "load_codex_auth", "load_codex_credentials",
                     "refresh_codex_credentials"):
            patcher = mock.patch.object(
                responses, name, side_effect=AssertionError(f"{name} called"),
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_requests_carry_no_codex_state_and_a_401_is_final(self):
        tool = ToolSpec("lookup", "Look things up.", {"type": "object", "properties": {}})
        unauthorized = _http_error(401, body=b'{"error": {"code": "invalid_api_key"}}')
        opener = _ScriptedOpener(_ok(), _ok(), unauthorized)
        with mock.patch.dict(os.environ, {"PUBLIC_TEST_KEY": "public-key"}):
            model = ResponsesModel(
                binding=CATALOG.bind(name="public-test"), opener=opener,
                retry_sleep=lambda _delay: None,
            )
            model.sample(_context(), tools=(tool,))
            model.sample(_context())
            with self.assertRaises(ModelAuthenticationError) as raised:
                model.sample(_context())
        self.assertEqual(len(opener.calls), 3)  # The 401 was not retried.
        self.assertEqual(raised.exception.failure.auth_source, "environment")
        with_tools, without_tools = _request_payload(opener, 0), _request_payload(opener, 1)
        self.assertEqual(opener.calls[0][0].full_url, "https://api.example.test/v1/responses")
        self.assertEqual(with_tools["model"], "public-wire")
        self.assertEqual([item.get("role") for item in with_tools["input"]], ["system", "user"])
        self.assertEqual(with_tools["truncation"], "auto")
        self.assertEqual((with_tools["tool_choice"], with_tools["parallel_tool_calls"]),
                         ("auto", False))
        self.assertFalse(TOOL_FIELDS & without_tools.keys())
        for index in (0, 1):
            payload = _request_payload(opener, index)
            self.assertEqual((payload["store"], payload["stream"]), (False, True))
            self.assertNotIn("prompt_cache_key", payload)
            self.assertEqual(set(_request_headers(opener, index)),
                             {"accept", "authorization", "content-type", "user-agent"})

    def test_pi_compaction_sends_no_tools_and_never_installs_a_refusal(self):
        context = InteractionContext((
            Init("session"),
            Instructions("Be brief."),
            Message("user", "First request."),
            UserInteractionBoundary(),
            Message("assistant", "First answer."),
            ModelSampleBoundary(),
            TurnSummary(sample_count=1),
            Message("user", "Second request."),
        ))
        refusal = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant",
                     "content": [{"type": "refusal", "refusal": "I can't do that."}]},
        }
        opener = _ScriptedOpener(_ok("## Goal\n- Finish."), _FakeSSEResponse(refusal, _completed_event()))
        model = ResponsesModel(binding=_anonymous_binding(), opener=opener)
        self.assertIsInstance(create_default_compactor(model), PiCompactor)
        compactor = PiCompactor(model, keep_recent_tokens=0)  # Summarize everything.
        result = compactor.compact(context)
        self.assertEqual(result.protocol, "pi")
        self.assertIsInstance(result.items[0], ContextPrefix)
        self.assertFalse(TOOL_FIELDS & _request_payload(opener).keys())
        with self.assertRaises(CompactionError):
            compactor.compact(context)
        self.assertEqual(len(opener.calls), 2)


class ResponsesModelConfigurationTests(unittest.TestCase):
    def test_python_configuration_resolves_pi_and_catalog_policy(self):
        with mock.patch.dict(os.environ, {"PUBLIC_TEST_KEY": "public-key"}):
            model = ResponsesModel(binding=CATALOG.bind(name="public-test"))
        config = InteractionConfig.from_model(model)
        self.assertEqual(config.get("compaction_mode"), "pi")
        self.assertEqual(config.get("auto_compact_tokens"), 90000)
        self.assertEqual(config.get("max_context_tokens"), 128000)
        self.assertEqual(dict(config.get("extra_sample_params")), {"truncation": "auto"})
        self.assertFalse(model.supports_remote_compaction)
        self.assertNotIsInstance(model, CodexResponsesModel)
        with self.assertRaisesRegex(ConfigError, "no provider compaction"):
            InteractionConfig.from_model(
                model, InteractionConfigSnapshot(compaction_mode="provider"),
            )
        with self.assertRaisesRegex(TypeError, "CodexResponsesModel"):
            ResponsesOpaqueCompactor(model)


if __name__ == "__main__":
    unittest.main()
