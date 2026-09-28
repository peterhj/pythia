"""Endpoint authority and catalog/CLI request semantics (offline)."""

from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.interaction_helpers import codex_model

from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pythia.interaction import (
    BUILTIN_MODEL_CATALOG, EndpointSpec, InteractionConfig, InteractionContext,
    Message, ModelSpec, ModelLimits, parse_model_catalog,
)
from pythia.interaction import cli, demo, responses
from pythia.interaction._auto_config import resolve_config, namespace
from pythia.interaction.chat_completions import ChatCompletionsEndpoint, ChatCompletionsModel
from pythia.interaction.messages import MessagesEndpoint, MessagesModel
from pythia.interaction.codex_auth import CodexAuth
from pythia.interaction.model_config import build_model, prepare_namespace, supports_account_services


V2 = '''[catalog]
version = 2
[model.local-max]
endpoint.api = chat-completions
endpoint.url = http://127.0.0.1:8000/custom/invoke/
endpoint.model = wire-model
endpoint.auth = none
aliases = ["local"]
limits.auto_compact_context_tokens = 100000
limits.max_context_tokens = 150000
request_params.thinking = {"type": "enabled"}
request_params.reasoning_effort = "max"
'''


def prepared(*flags, catalog=None):
    return prepare_namespace(cli._build_parser().parse_args(flags), catalog)


def ctx():
    return InteractionContext((Message("user", "hello"),))


class EndpointSpecTests(unittest.TestCase):
    def test_exact_urls_auth_and_connection_identity(self):
        endpoint = EndpointSpec("chat-completions", "https://host.test/prefix/invoke/", "wire", "env:KEY")
        self.assertEqual(endpoint.url, "https://host.test/prefix/invoke/")
        self.assertEqual(endpoint.environment_variable, "KEY")
        self.assertEqual(endpoint.connection_identity, replace(endpoint, model="other").connection_identity)
        for changes in ({"url": "https://user:password@host.test"}, {"auth": "env:BAD-NAME"},
                        {"auth": "environment"}, {"auth": "codex-login"}, {"auth_file": "unused"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(endpoint, **changes)

    def test_model_definition_has_one_endpoint(self):
        original = BUILTIN_MODEL_CATALOG.get_model_spec("codex", "codex-gpt-6-astra")
        changed = replace(
            original,
            endpoint=replace(original.endpoint, model="renamed"),
        )
        self.assertEqual(changed.endpoint.model, "renamed")
        self.assertEqual(original.endpoint.model, "gpt-6-astra")

    def test_old_route_catalog_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_model_catalog('''[catalog]
version = 1
[model.example]
api = chat-completions
api_model = wire
route.provider = label
route.api_url = http://host.test
route.auth_source = explicit
''')

    def test_v2_needs_no_provider_and_granular_patches_preserve_endpoint(self):
        registry = parse_model_catalog(V2)
        spec = registry.get_model_spec("chat-completions", "local")
        self.assertEqual(spec.endpoint.auth, "none")
        changed = parse_model_catalog('''[catalog]
version = 2
[model.local-max]
override = true
limits.max_context_tokens = 200000
request_params.thinking = {"type": "disabled"}
''', base=registry).get_model_spec("chat-completions", "local")
        self.assertEqual(changed.endpoint, spec.endpoint)
        self.assertEqual(changed.limits.max_context_tokens, 200000)
        self.assertEqual(dict(changed.request_params["thinking"]), {"type": "disabled"})

    def test_mixed_and_incomplete_schemas_rejected(self):
        for text in (V2.replace("version = 2", "version = 1"), V2 + "route.api = chat-completions\n",
                     V2 + "api_model = other\n", V2.replace("endpoint.auth = none\n", ""),
                     V2.replace("endpoint.model = wire-model", "endpoint.model = null")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_catalog(text)
        with self.assertRaises((ValueError, TypeError)):
            replace(
                BUILTIN_MODEL_CATALOG.specs[0],
                endpoint=replace(BUILTIN_MODEL_CATALOG.specs[0].endpoint, model=None),
            )
        with self.assertRaises(ValueError):
            parse_model_catalog(V2.replace('request_params.thinking = {"type": "enabled"}\n', '')
                                .replace('request_params.reasoning_effort = "max"', 'request_params = null'))

    def test_credentialed_url_change_requires_explicit_auth_rebinding(self):
        registry = parse_model_catalog(V2.replace('endpoint.auth = none', 'endpoint.auth = env:KEY'))
        with self.assertRaisesRegex(ValueError, "endpoint-auth"):
            registry.bind(name="local", endpoint_url="https://other.test/infer")
        with self.assertRaises(ValueError):
            parse_model_catalog('''[catalog]
version=2
[model.local-max]
override=true
endpoint.url=https://other.test/infer
''', base=registry)
        binding = registry.bind(name="local", endpoint_url="https://other.test/infer", endpoint_auth="env:KEY")
        self.assertEqual(binding.endpoint.url, "https://other.test/infer")


class EndpointRuntimeTests(unittest.TestCase):
    def test_selector_is_distinct_from_endpoint_model_and_url(self):
        registry = parse_model_catalog(V2)
        args = prepared("--model", "local", "--endpoint-model", "wire-override", catalog=registry)
        model = build_model(args)
        config = InteractionConfig.from_namespace(args)
        self.assertEqual(args.model_binding.selector, "local")
        self.assertEqual(model.endpoint.url, "http://127.0.0.1:8000/custom/invoke/")
        payload = model._build_request_payload(ctx(), (), config.snapshot().sampling_params())
        self.assertEqual(payload["model"], "wire-override")
        self.assertEqual(payload["reasoning_effort"], "max")

    def test_endpoint_only_cli_does_not_resolve_wire_id_as_catalog_selector(self):
        args = prepared("--endpoint-api", "chat-completions", "--endpoint-model", "gpt-6-astra",
                        "--endpoint-url", "http://localhost:8000/infer", "--endpoint-auth", "none")
        self.assertIsNone(args.model_binding.spec)
        self.assertIsNone(args.model_binding.selector)
        self.assertEqual(build_model(args).endpoint.url, "http://localhost:8000/infer")
        self.assertIsNone(InteractionConfig.from_namespace(args).get("auto_compact_tokens"))
        self.assertEqual(
            cli._build_parser().parse_args([
                "--endpoint-api", "messages",
            ]).model_api,
            "messages",
        )

    def test_old_endpoint_flags_are_rejected(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            cli._build_parser().parse_args(["--api-url", "http://host"])
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            cli._build_parser().parse_args(["--model-api", "messages"])

    def test_prepared_raw_mutations_cannot_override_actual_url_model_or_login_file(self):
        with tempfile.TemporaryDirectory() as directory:
            args = prepared("--model", "codex-gpt-6-astra", "--endpoint-auth-home", directory)
            endpoint = args.model_binding.endpoint
            args.endpoint_url = "https://another.example.test/infer"
            args.endpoint_model = "other-wire"
            args.model = "other-selector"
            args.codex_auth_file = "/must-not-read"
            with mock.patch.object(responses, "load_codex_auth", return_value=CodexAuth("FAKE")) as load:
                model = build_model(args)
            self.assertEqual(model.endpoint.url, endpoint.url)
            self.assertEqual(model.binding.endpoint.model, endpoint.model)
            self.assertEqual(supports_account_services(args), model.binding.supports_account_services)
            load.assert_called_once_with(auth_file=endpoint.auth_file)

    def test_bound_transport_constructors_reject_late_url_changes(self):
        binding = parse_model_catalog(V2).bind(name="local")
        with self.assertRaises(TypeError):
            ChatCompletionsEndpoint(
                binding=binding,
                api_url="http://different.example.test",
            )
        binding = BUILTIN_MODEL_CATALOG.bind("codex", "codex-gpt-6-astra")
        with self.assertRaises(TypeError):
            responses.CodexResponsesModel(
                binding=binding,
                api_url="https://other.example.test",
            )

    def test_environment_requirement_does_not_depend_on_catalog_origin(self):
        changed = parse_model_catalog('''[catalog]
version = 2
[model.claude-fable-5-1]
override = true
source = metadata only
''')
        for registry in (BUILTIN_MODEL_CATALOG, changed):
            with self.subTest(registry=registry), mock.patch.dict("os.environ", {}, clear=True):
                args = prepared("--model", "claude-fable-5-1", catalog=registry)
                with self.assertRaisesRegex(ValueError, "credential"):
                    build_model(args)
                anonymous = prepared("--model", "claude-fable-5-1", "--endpoint-auth", "none", catalog=registry)
                self.assertIsNone(build_model(anonymous).endpoint.api_key)

    def test_explicit_env_and_supplied_credentials(self):
        with mock.patch.dict("os.environ", {"ENDPOINT_KEY": "fake-env"}):
            args = prepared("--endpoint-model", "wire", "--endpoint-auth", "env:ENDPOINT_KEY")
            self.assertEqual(build_model(args).endpoint.api_key, "fake-env")
            args.api_key = "stale-raw"
            self.assertEqual(build_model(args).endpoint.api_key, "fake-env")
        args = prepared("--endpoint-model", "wire", "--endpoint-auth", "supplied")
        with self.assertRaisesRegex(ValueError, "credential"):
            build_model(args)
        args = prepared("--endpoint-model", "wire", "--endpoint-auth", "supplied", "--endpoint-api-key", "fake")
        self.assertEqual(build_model(args).endpoint.api_key, "fake")
        with self.assertRaises(ValueError):
            prepared("--endpoint-auth", "none", "--endpoint-api-key", "fake")

    def test_messages_uses_exact_endpoint_model_and_anonymous_headers(self):
        args = prepared("--endpoint-api", "messages", "--endpoint-url", "http://host.test/custom/messages",
                        "--endpoint-model", "wire", "--endpoint-auth", "none", "--max-output-tokens", "77")
        model = build_model(args)
        self.assertEqual(model.endpoint.url, "http://host.test/custom/messages")
        self.assertEqual(model._build_request_payload(ctx(), (), None)["model"], "wire")
        self.assertIsNone(model.endpoint.api_key)

    def test_anonymous_generic_responses_requires_no_dummy_bearer(self):
        spec = EndpointSpec("responses", "http://host.test/custom/respond", "wire", "none")
        registry = type(BUILTIN_MODEL_CATALOG)((ModelSpec(name="test", endpoint=spec),))
        model = codex_model(binding=registry.bind(name="test"))
        self.assertIsNone(model.endpoint.bearer_token)
        self.assertNotIn("Authorization", model._build_headers(responses._ProviderState()))
        snapshot = model._checked_credential()
        self.assertIsNone(snapshot.auth)
        self.assertEqual(model.endpoint.url, spec.url)
        from pythia_test.test_responses import _FakeSSEResponse, _message_event, _completed_event
        requests = []
        def opener(request, **kwargs):
            requests.append(request)
            return _FakeSSEResponse(_message_event(0, "done"), _completed_event())
        model._opener = opener
        self.assertEqual(model.sample(ctx()).last_assistant_text, "done")
        self.assertEqual(requests[0].full_url, spec.url)
        self.assertIsNone(requests[0].get_header("Authorization"))

    def test_real_request_builders_send_exact_urls_and_no_anonymous_auth(self):
        class Response:
            status = 200
            headers = {}
            def __init__(self, payload):
                self.payload = payload
            def read(self):
                return json.dumps(self.payload).encode()
            def close(self):
                pass
        for api in ("chat-completions", "messages"):
            with self.subTest(api=api):
                args = prepared("--endpoint-api", api, "--endpoint-url", "http://host.test/exact/",
                                "--endpoint-model", "wire", "--endpoint-auth", "none", "--max-output-tokens", "77")
                model = build_model(args)
                requests = []
                payload = ({"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}
                           if api == "chat-completions" else
                           {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"})
                def opener(request, **kwargs):
                    requests.append(request)
                    return Response(payload)
                model._opener = opener
                self.assertEqual(model.sample(ctx()).last_assistant_text, "done")
                self.assertEqual(requests[0].full_url, "http://host.test/exact/")
                self.assertIsNone(requests[0].get_header("Authorization"))
                self.assertIsNone(requests[0].get_header("X-api-key"))

    def test_auto_endpoint_fields_override_catalog_values(self):
        registry = parse_model_catalog(V2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1, "contexts": {"2": {
                "endpoint_url": "http://host.test/exact", "endpoint_model": "worker-wire",
                "endpoint_auth": "none",
            }}}))
            settings = resolve_config(
                path,
                {
                    "model": "local",
                    "endpoint_url": "http://launch.test/exact",
                },
                catalog=registry,
            )
            endpoint = namespace(settings[2], registry).model_binding.endpoint
            self.assertEqual(endpoint.url, "http://host.test/exact")
            self.assertEqual(endpoint.model, "worker-wire")

if __name__ == "__main__":
    unittest.main()
