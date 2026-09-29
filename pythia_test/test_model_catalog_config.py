from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.interaction_helpers import responses_endpoint
from pythia_test.interaction_helpers import codex_model

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    BUILTIN_MODEL_CATALOG, CodexResponsesModel, ConfigError, Environment, Init,
    Instructions, InteractionConfig, InteractionContext, Message, ModelCatalog,
    ModelConfigurationError, ModelSample, ResolvedSamplingParams, SamplingParams,
    StreamingResponsesEndpoint, ToolSpec, LATEST_MODEL_CATALOG_VERSION,
    load_model_catalog, parse_model_catalog,
)
from pythia.interaction import auto, cli, demo, model_catalog, responses
from pythia.interaction._auto_config import build_parser, load_saved_config, namespace, resolve_config
from pythia.interaction._model_binding_debug import (
    debug_model_binding_path, save_debug_model_bindings,
)
from pythia.interaction.chat_completions import ChatCompletionsEndpoint, ChatCompletionsModel
from pythia.interaction.messages import (
    MessagesEndpoint, MessagesModel, MessagesPromptCaching, MessagesServerCompaction,
)
from pythia.interaction.model_config import build_model, prepare_namespace, supports_account_services
from pythia.interaction.model_catalog_config import MAX_CATALOG_BYTES


HEADER = "[catalog]\nversion = 2\n"
LOCAL = """
[model.local-max]
endpoint.api = chat-completions
endpoint.model = served-local
endpoint.url = http://127.0.0.1:8000/v1/chat/completions
endpoint.auth = none
aliases = ["local-alias"]
limits.auto_compact_context_tokens = 100000
limits.max_context_tokens = 150000
limits.max_output_tokens = 32000
request_params.thinking = {"type": "enabled", "budget_tokens": 1000}
request_params.reasoning_effort = "max"
"""
MESSAGE = """
[model.worker]
endpoint.api = messages
endpoint.model = served-messages
endpoint.url = https://messages.example.test/v1/messages
endpoint.auth = none
limits.auto_compact_context_tokens = 60000
limits.max_context_tokens = 80000
limits.max_output_tokens = 8000
messages.output_effort = high
"""
CODEX = """
[model.code-env]
endpoint.api = codex
endpoint.model = served-code
endpoint.url = https://responses.example.test/v1/responses
endpoint.auth = env:CATALOG_TEST_TOKEN
responses.reasoning_effort = high
"""


def catalog(text=LOCAL, **kwargs):
    return parse_model_catalog(HEADER + text, **kwargs)


def args_for(registry, *flags):
    return prepare_namespace(cli._build_parser().parse_args(flags), registry)


def context():
    return InteractionContext((Message("user", "hello"),))


_environment_get = os.environ.get


def no_credentials(key, default=None):
    if key == "CATALOG_TEST_TOKEN" or key == "CODEX_HOME" or key.endswith("_API_KEY"):
        raise AssertionError("unexpected credential lookup")
    return _environment_get(key, default)  # argparse/gettext may read locale settings.


class CatalogParserTests(unittest.TestCase):
    def test_missing_header_or_version_assumes_latest_and_warns_after_validation(self):
        for text in (
            (HEADER + LOCAL).replace("version = 2\n", ""),
            LOCAL,
        ):
            with self.subTest(text=text), self.assertWarnsRegex(
                UserWarning,
                r"does not specify a version; assuming latest supported version 2",
            ):
                registry = parse_model_catalog(text, source="catalog.ini")
            self.assertEqual(
                registry.get_model_spec("chat-completions", "local-max").name,
                "local-max",
            )
        self.assertEqual(LATEST_MODEL_CATALOG_VERSION, 2)

        invalid = LOCAL + "unknown = value\n"
        with mock.patch("warnings.warn") as warn:
            with self.assertRaises(ValueError):
                parse_model_catalog(invalid, source="catalog.ini")
        warn.assert_not_called()

    def test_new_entry_routes_aliases_and_preserves_structured_values(self):
        registry = catalog(LOCAL + MESSAGE)
        spec = registry.get_model_spec("chat-completions", "local-alias")
        self.assertEqual(spec.name, "local-max")
        self.assertEqual(spec.endpoint.model, "served-local")
        self.assertEqual(spec.request_params["thinking"]["budget_tokens"], 1000)
        self.assertEqual(spec.limits.max_context_tokens, 150000)
        self.assertEqual(registry.bind(name="worker").api, "messages")
        self.assertIsNone(model_catalog.get_model_spec("chat-completions", "local-max"))
        self.assertIn(spec, registry.list_model_specs("chat-completions"))

    def test_parser_preserves_case_percent_comments_and_multiline_json(self):
        registry = catalog(LOCAL + '''request_params.CaseSensitive = "100% # literal ; text"
request_params.a.b = 7
request_params.nested = {
    "inner": [true, null, "value"]
    }
''')
        params = registry.bind(name="local-max").request_params
        self.assertEqual(params["CaseSensitive"], "100% # literal ; text")
        self.assertNotIn("casesensitive", params)
        self.assertEqual(params["a.b"], 7)
        self.assertEqual(params["nested"]["inner"], (True, None, "value"))

    def test_granular_override_preserves_other_fields_and_builtin_object(self):
        original = model_catalog.get_model_spec("codex", "codex-gpt-6-astra")
        registry = catalog("""
[model.codex-gpt-6-astra]
override = true
limits.auto_compact_context_tokens = 700000
responses.text_verbosity = medium
""")
        changed = registry.get_model_spec("codex", original.name)
        self.assertEqual(changed.limits.auto_compact_context_tokens, 700000)
        self.assertEqual(changed.responses.text_verbosity, "medium")
        self.assertEqual(changed.endpoint, original.endpoint)
        self.assertEqual(changed.endpoint.model, original.endpoint.model)
        self.assertEqual(changed.responses.reasoning_summary, original.responses.reasoning_summary)
        self.assertEqual(changed.limits.max_context_tokens, original.limits.max_context_tokens)
        self.assertEqual(original.limits.auto_compact_context_tokens, 872000)
        self.assertIs(registry.get_model_spec("codex", "codex-gpt-6-astra-max"),
                      model_catalog.get_model_spec("codex", "codex-gpt-6-astra-max"))

    def test_override_request_values_are_atomic_and_map_can_be_cleared(self):
        base = catalog()
        changed = catalog('''[model.local-max]
override = true
request_params.thinking = {"type": "disabled"}
''', base=base)
        params = changed.bind(name="local-max").request_params
        self.assertEqual(dict(params["thinking"]), {"type": "disabled"})
        self.assertEqual(params["reasoning_effort"], "max")
        literal_null = catalog('''[model.local-max]
override = true
request_params.thinking = null
''', base=base)
        self.assertIsNone(literal_null.bind(name="local-max").request_params["thinking"])
        cleared = catalog('''[model.local-max]
override = true
request_params = {}
''', base=base)
        self.assertEqual(dict(cleared.bind(name="local-max").request_params), {})

    def test_alias_replacement_and_nullable_field_clear(self):
        base = catalog('[model.claude-fable-5.1]\noverride = true\naliases = ["old-fable"]\n')
        registry = catalog('''[model.claude-fable-5.1]
override = true
endpoint.api = messages
aliases = ["my-fable"]
limits.auto_compact_context_tokens = null
''', base=base)
        self.assertIsNone(registry.get_model_spec("messages", "old-fable"))
        self.assertEqual(registry.get_model_spec("messages", "my-fable").name, "claude-fable-5.1")
        self.assertIsNone(registry.bind("messages", "my-fable").limits.auto_compact_context_tokens)
        self.assertIsNotNone(base.get_model_spec("messages", "old-fable"))

    def test_old_spellings_and_invalid_schema_are_rejected(self):
        bad = (
            LOCAL + "api = chat-completions\n",
            LOCAL + "api_model = other\n",
            LOCAL.replace("endpoint.api =", "route.api ="),
            LOCAL.replace("endpoint.model =", "route.api_model ="),
            LOCAL + "unknown = value\n",
            LOCAL.replace("limits.max_context_tokens = 150000", "limits.max_context_tokens = true"),
            LOCAL.replace('aliases = ["local-alias"]', 'aliases = "alias"'),
            LOCAL.replace('request_params.reasoning_effort = "max"', 'request_params.reasoning_effort = max'),
            LOCAL + "request_params = {}\n",
            LOCAL + "request_params.duplicate = {\"x\":1,\"x\":2}\n",
            LOCAL + "request_params.nonfinite = NaN\n",
            LOCAL + "request_params.nonfinite = 1e999\n",
            LOCAL.replace("endpoint.auth = none", "endpoint.auth = codex-login"),
            LOCAL.replace("http://127.0.0.1:8000", "https://user:secret@example.test"),
            LOCAL.replace("endpoint.auth = none", "endpoint.auth = env:BAD-NAME"),
            MESSAGE.replace("60000", "49999"),
            MESSAGE + 'request_params.system = "replacement instructions"\n',
            CODEX + 'request_params.reasoning = {"effort": "low"}\n',
            '[model.codex-gpt-6-astra]\noverride = true\nendpoint.api = null\n',
            '[model.missing]\noverride = true\n',
            LOCAL.replace("local-max", "codex-gpt-6-astra").replace("chat-completions", "codex"),
        )
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                catalog(text)
        with self.assertRaises(ValueError):  # Overrides name the canonical model, not an alias.
            catalog('[model.local-alias]\noverride = true\n', base=catalog())

    def test_duplicate_sections_keys_defaults_and_versions_rejected(self):
        for text in (
            HEADER + LOCAL + LOCAL,
            HEADER + LOCAL + 'request_params.reasoning_effort = "low"\n',
            HEADER + "[DEFAULT]\noverride = true\n" + LOCAL,
            HEADER.replace("version = 2", "version = 1") + LOCAL,
            HEADER.replace("version = 2", "version = 3") + LOCAL,
            HEADER + "[other]\nx = 1\n", HEADER + "unknown = 1\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_catalog(text)

    def test_alias_collision_is_not_file_order_dependent(self):
        other = LOCAL.replace("[model.local-max]", "[model.other]")
        for text in (LOCAL + other, other + LOCAL):
            with self.assertRaisesRegex(ValueError, "collision"):
                catalog(text)

    def test_protected_fields_and_errors_do_not_disclose_values(self):
        for key in ("model", "messages", "stream", "tools", "max_tokens", "max_output_tokens",
                    "max_completion_tokens", "temperature", "seed", "api_key", "Authorization"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as error:
                    catalog(LOCAL + f'request_params.{key} = "VERY_SECRET_TOKEN"\n')
                self.assertNotIn("VERY_SECRET_TOKEN", str(error.exception))

    def test_messages_and_responses_entries_accept_request_params(self):
        generic = CODEX.replace("[model.code-env]", "[model.generic]").replace(
            "endpoint.api = codex", "endpoint.api = responses",
        )
        registry = catalog(
            MESSAGE + 'request_params.thinking = {"type": "enabled", "budget_tokens": 1024}\n'
            + CODEX + 'request_params.service_tier = "flex"\n'
            + generic + 'request_params.truncation = "auto"\n'
        )
        thinking = registry.bind(name="worker").request_params["thinking"]
        self.assertEqual(dict(thinking), {"type": "enabled", "budget_tokens": 1024})
        self.assertEqual(registry.bind(name="code-env").request_params["service_tier"], "flex")
        self.assertEqual(registry.bind("responses", "generic").request_params["truncation"], "auto")
        patched = catalog('''[model.codex-gpt-6-astra]
override = true
endpoint.api = codex
request_params.service_tier = "priority"
''')
        self.assertEqual(patched.bind("codex", "codex-gpt-6-astra").request_params["service_tier"], "priority")
        self.assertFalse(BUILTIN_MODEL_CATALOG.bind("codex", "codex-gpt-6-astra").request_params)

    def test_request_params_protect_the_fields_each_api_adapter_owns(self):
        responses_owned = ("tool_choice", "store", "include", "reasoning", "text",
                           "prompt_cache_key", "previous_response_id", "conversation")
        owned = {
            "messages": ("system", "output_config", "cache_control"),
            "codex": responses_owned,
            "responses": responses_owned,
        }
        entries = {
            "chat-completions": ("local-max", LOCAL),
            "messages": ("worker", MESSAGE),
            "codex": ("code-env", CODEX),
            "responses": ("code-env", CODEX.replace("endpoint.api = codex", "endpoint.api = responses")),
        }
        every_owned = sorted({key for keys in owned.values() for key in keys})
        for api, (name, text) in entries.items():
            for key in every_owned:
                with self.subTest(api=api, key=key):
                    if key in owned.get(api, ()):
                        for spelling in (key, key.upper()):
                            with self.assertRaises(ValueError):
                                catalog(text + f"request_params.{spelling} = 1\n")
                            with self.assertRaises(ValueError):
                                catalog(text).bind(api, name, request_params={spelling: 1})
                    else:
                        # Protection is API-specific: e.g. Chat Completions keeps these.
                        registry = catalog(text + f"request_params.{key} = 1\n")
                        self.assertEqual(registry.bind(api, name).request_params[key], 1)

    def test_catalog_and_nested_request_params_are_immutable(self):
        registry = catalog()
        binding = registry.bind(name="local-max")
        with self.assertRaises(FrozenInstanceError):
            registry.specs = ()
        with self.assertRaises(TypeError):
            registry.origins[("chat-completions", "local-max")] = "other"
        with self.assertRaises(TypeError):
            binding.request_params["thinking"]["type"] = "disabled"
        with self.assertRaises(TypeError):
            binding.spec.request_params["reasoning_effort"] = "low"

    def test_explicit_api_isolation_and_bare_name_ambiguity(self):
        registry = catalog(LOCAL.replace("local-max", "codex-gpt-6-astra"))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            registry.bind(name="codex-gpt-6-astra")
        self.assertEqual(registry.bind("codex", "codex-gpt-6-astra").endpoint.model, "gpt-6-astra")
        self.assertEqual(registry.bind("chat-completions", "codex-gpt-6-astra").endpoint.model, "served-local")
        isolated = registry.bind("messages", "codex-gpt-6-astra")
        self.assertIsNone(isolated.spec)
        self.assertFalse(isolated.request_params)
        with self.assertRaises(ValueError):
            catalog('[model.codex-gpt-6-astra]\noverride = true\nsource = patch\n', base=registry)
        patched = catalog('[model.codex-gpt-6-astra]\noverride = true\nendpoint.api = codex\nsource = patch\n', base=registry)
        self.assertEqual(patched.get_model_spec("codex", "codex-gpt-6-astra").source, "patch")

    def test_discovery_is_explicit_bounded_and_missing_default_is_optional(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(Path, "home", return_value=Path(directory)):
            self.assertIs(load_model_catalog(), BUILTIN_MODEL_CATALOG)
            path = Path(directory) / ".pythia" / "model-catalog.ini"
            path.parent.mkdir()
            path.write_text(HEADER + LOCAL)
            loaded = load_model_catalog()
            self.assertEqual(loaded.bind(name="local-max").origin, str(path))
            path.write_text("malformed")
            with self.assertRaises(ValueError):
                load_model_catalog()
            self.assertIs(load_model_catalog(enabled=False), BUILTIN_MODEL_CATALOG)
            with self.assertRaises(ValueError):
                load_model_catalog(Path(directory) / "missing.ini")
            with self.assertRaises(ValueError):
                load_model_catalog(path, enabled=False)
            path.write_bytes(b"x" * (MAX_CATALOG_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "size limit"):
                load_model_catalog(path)

    def test_catalog_reader_rejects_special_files_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_model_catalog(directory)
            if hasattr(os, "mkfifo"):
                path = Path(directory) / "pipe.ini"
                os.mkfifo(path)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    load_model_catalog(path)


class BoundRequestTests(unittest.TestCase):
    def test_catalog_route_model_and_policy_feed_one_config_and_request(self):
        args = args_for(catalog(), "--model", "local-alias")
        model = build_model(args)
        config = InteractionConfig.from_namespace(args)
        self.assertEqual(args.model_api, "chat-completions")
        self.assertEqual(config.get("auto_compact_tokens"), 100000)
        self.assertEqual(config.get("max_context_tokens"), 150000)
        self.assertIsNone(config.get("max_output_tokens"))  # The ceiling is not a budget.
        self.assertEqual(config.initial_values(), config.values())
        for params in (None, SamplingParams(), SamplingParams(max_output_tokens=77),
                       config.snapshot().sampling_params()):
            payload = model._build_request_payload(context(), (), params)
            self.assertEqual(payload["model"], "served-local")
            self.assertEqual(payload["reasoning_effort"], "max")
            self.assertEqual(payload["thinking"]["type"], "enabled")
        payload["thinking"]["type"] = "changed"
        self.assertEqual(model._build_request_payload(context(), (), None)["thinking"]["type"], "enabled")
        self.assertEqual(config.get("request_params")["thinking"]["type"], "enabled")
        with self.assertRaisesRegex(ConfigError, "launch-only"):
            config.set("request_params", {})
        rendered = json.loads(config.render(json_output=True))
        self.assertEqual(next(iter(rendered)), "__init__")
        self.assertEqual(rendered["request_params"], rendered["__init__"]["request_params"])

    def test_explicit_request_overlay_and_resolved_empty_never_fall_back(self):
        args = args_for(catalog(), "--model", "local-max", "--request-params",
                        '{"thinking":{"type":"disabled"},"custom":true}')
        cfg = InteractionConfig.from_namespace(args)
        model = build_model(args)
        payload = model._build_request_payload(context(), (), cfg.snapshot().sampling_params())
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertIs(payload["custom"], True)
        empty = model._build_request_payload(context(), (), ResolvedSamplingParams())
        self.assertNotIn("thinking", empty)
        self.assertNotIn("reasoning_effort", empty)

    def test_user_messages_limits_defaults_and_wire_model_do_not_lookup_builtins(self):
        args = args_for(catalog(MESSAGE), "--model", "worker")
        cfg = InteractionConfig.from_namespace(args)
        model = build_model(args)
        self.assertIsInstance(model, MessagesModel)
        self.assertEqual(model.endpoint.max_output_tokens, 8000)
        self.assertEqual(model.max_context_tokens, 80000)
        payload = model._build_request_payload(
            context(), (), cfg.snapshot().sampling_params(),
        )
        self.assertEqual(payload["model"], "served-messages")
        self.assertEqual(payload["output_config"], {"effort": "high"})
        self.assertEqual(payload["context_management"]["edits"][0]["trigger"]["value"], 60000)
        self.assertEqual(payload["max_tokens"], 8000)

    def test_user_responses_route_credentials_and_presets_are_bound(self):
        args = args_for(catalog(CODEX), "--model", "code-env")
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}), \
                mock.patch.object(responses, "load_codex_auth", side_effect=AssertionError("ambient credentials")):
            model = build_model(args)
        self.assertEqual(model.endpoint.bearer_token, "fake-token")
        self.assertEqual(
            model.endpoint.url,
            "https://responses.example.test/v1/responses",
        )
        payload, _ = model._build_request_payload(context(), (), None)
        self.assertEqual(payload["model"], "served-code")
        self.assertEqual(payload["reasoning"]["effort"], "high")
        self.assertFalse(model.supports_remote_compaction)
        self.assertFalse(supports_account_services(args))
        args.codex_home = "/must-not-read"
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}), \
                mock.patch.object(responses, "load_codex_auth", side_effect=AssertionError("stale raw path")):
            self.assertEqual(build_model(args).endpoint.bearer_token, "fake-token")

    def test_user_messages_request_params_extend_every_request(self):
        registry = catalog(MESSAGE + 'request_params.thinking = {"type": "enabled", "budget_tokens": 1024}\n'
                           + "request_params.top_k = 5\n")
        args = args_for(registry, "--model", "worker",
                        "--request-params", '{"metadata": {"user_id": "catalog-test"}}')
        cfg = InteractionConfig.from_namespace(args)
        model = build_model(args)
        self.assertEqual(cfg.get("request_params")["top_k"], 5)
        for params in (None, SamplingParams(temperature=1.0), cfg.snapshot().sampling_params()):
            payload = model._build_request_payload(context(), (), params)
            self.assertEqual(payload["thinking"], {"type": "enabled", "budget_tokens": 1024})
            self.assertEqual(payload["top_k"], 5)
            self.assertEqual(payload["metadata"], {"user_id": "catalog-test"})
            # Typed preferences and adapter-owned fields remain authoritative.
            self.assertEqual(payload["model"], "served-messages")
            self.assertEqual(payload["max_tokens"], 8000)
            self.assertEqual(payload["output_config"], {"effort": "high"})
            self.assertEqual(payload["context_management"]["edits"][0]["trigger"]["value"], 60000)
        payload["thinking"]["budget_tokens"] = 0
        self.assertEqual(model._build_request_payload(context(), (), None)["thinking"]["budget_tokens"], 1024)
        empty = model._build_request_payload(context(), (), ResolvedSamplingParams(max_output_tokens=10))
        self.assertFalse({"thinking", "top_k", "metadata"} & empty.keys())
        with self.assertRaises(ValueError):
            args_for(registry, "--model", "worker", "--request-params", '{"system": "replacement"}')

    def test_user_codex_request_params_extend_every_request(self):
        registry = catalog(CODEX + 'request_params.service_tier = "flex"\n')
        args = args_for(registry, "--model", "code-env", "--request-params", '{"truncation": "auto"}')
        cfg = InteractionConfig.from_namespace(args)
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}):
            model = build_model(args)
        for params in (None, SamplingParams(max_output_tokens=77), cfg.snapshot().sampling_params()):
            payload, _ = model._build_request_payload(context(), (), params)
            self.assertEqual(payload["service_tier"], "flex")
            self.assertEqual(payload["truncation"], "auto")
            self.assertEqual(payload["model"], "served-code")
            self.assertEqual(payload["reasoning"], {"effort": "high"})
            self.assertIs(payload["store"], False)
            self.assertEqual(payload["include"], ["reasoning.encrypted_content"])
        empty, _ = model._build_request_payload(context(), (), ResolvedSamplingParams())
        self.assertFalse({"service_tier", "truncation"} & empty.keys())
        with self.assertRaises(ValueError):
            args_for(registry, "--model", "code-env", "--request-params", '{"store": true}')

    def test_resolved_request_params_are_checked_against_the_adapter_api(self):
        reasoning = {"reasoning": {"effort": "low"}}
        chat = ChatCompletionsModel(chat_endpoint("http://localhost:8000", "literal"))
        payload = chat._build_request_payload(context(), (), ResolvedSamplingParams(request_params=reasoning))
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        codex = codex_model(responses_endpoint(
            "http://localhost:8000/v1", "literal", "token", api_provider="codex",
        ))
        with self.assertRaisesRegex(ModelConfigurationError, "adapter-owned"):
            codex._build_request_payload(context(), (), ResolvedSamplingParams(request_params=reasoning))
        messages = MessagesModel(MessagesEndpoint(binding=catalog(MESSAGE).bind(name="worker")))
        with self.assertRaisesRegex(ModelConfigurationError, "adapter-owned"):
            messages._build_request_payload(context(), (), ResolvedSamplingParams(
                max_output_tokens=10, request_params={"cache_control": {"type": "ephemeral"}},
            ))

    def test_every_field_an_adapter_writes_is_a_protected_request_param(self):
        tool = ToolSpec("lookup", "Look things up.", {"type": "object", "properties": {}})
        full = InteractionContext((Init("session-1"), Instructions("Be brief."), Message("user", "hello")))
        sampling = {"max_output_tokens": 5, "temperature": 0.5, "top_p": 0.5, "stop": ("END",)}
        chat = ChatCompletionsModel(chat_endpoint("http://localhost:8000", "literal"))
        messages = MessagesModel(MessagesEndpoint(
            binding=catalog(MESSAGE).bind(name="worker"),
            server_compaction=MessagesServerCompaction(),
            prompt_caching=MessagesPromptCaching(),
        ))
        codex = CodexResponsesModel(endpoint=StreamingResponsesEndpoint(
            binding=BUILTIN_MODEL_CATALOG.bind("codex", "codex-gpt-6-astra-max", endpoint_auth="supplied"),
            bearer_token="token",
        ))
        codex_payload, _ = codex._build_request_payload(full, (tool,), SamplingParams(max_output_tokens=5))
        written = {
            "chat-completions": chat._build_request_payload(full, (tool,), SamplingParams(seed=1, **sampling)),
            "messages": messages._build_request_payload(full, (tool,), SamplingParams(**sampling)),
            # One adapter serves both Responses dialects.
            "codex": codex_payload,
            "responses": codex_payload,
        }
        # Guard against a vacuous check: optional fields must be present.
        self.assertLessEqual({"system", "output_config", "cache_control", "context_management"},
                             written["messages"].keys())
        self.assertLessEqual({"reasoning", "text", "prompt_cache_key", "max_output_tokens"},
                             codex_payload.keys())
        for api, payload in written.items():
            for key in payload:
                with self.subTest(api=api, key=key), self.assertRaises(ValueError):
                    model_catalog.freeze_request_params({key: None}, api)

    def test_adapter_never_silently_replaces_an_unprotected_field(self):
        # Simulate an adapter field that was not added to the protected names.
        drifted = dict(model_catalog._PROFILE_RESERVED_REQUEST_PARAMS)
        drifted["messages"] = drifted["messages"] - {"output_config"}
        with mock.patch.object(model_catalog, "_PROFILE_RESERVED_REQUEST_PARAMS", drifted):
            binding = catalog(MESSAGE + 'request_params.output_config = {"effort": "low"}\n').bind(name="worker")
        model = MessagesModel(MessagesEndpoint(binding=binding))
        with self.assertRaisesRegex(ModelConfigurationError, "replace adapter-owned request fields: output_config"):
            model._build_request_payload(context(), (), None)

    def test_endpoint_location_alone_cannot_grant_account_or_compaction_services(self):
        text = CODEX.replace(
            "https://responses.example.test/v1",
            model_catalog.CODEX_RESPONSES_API_URL,
        )
        args = args_for(catalog(text), "--model", "code-env")
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}):
            model = build_model(args)
        self.assertFalse(supports_account_services(args))
        self.assertFalse(model.supports_remote_compaction)

    def test_chat_environment_reference_resolved_only_for_selected_model(self):
        text = LOCAL.replace("endpoint.auth = none", "endpoint.auth = env:CATALOG_TEST_TOKEN")
        with mock.patch("os.environ.get", side_effect=no_credentials):
            registry = catalog(text)
            args = args_for(registry, "--model", "local-max")
            InteractionConfig.from_namespace(args)
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}):
            self.assertEqual(build_model(args).endpoint.api_key, "fake-token")
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            build_model(args)

    def test_bindings_are_stable_when_file_or_other_catalog_changes(self):
        first = catalog()
        args = args_for(first, "--model", "local-max")
        cfg, model = InteractionConfig.from_namespace(args), build_model(args)
        second = catalog(LOCAL.replace("served-local", "other-wire").replace("100000", "110000"))
        self.assertEqual(second.bind(name="local-max").endpoint.model, "other-wire")
        self.assertEqual(cfg.set("auto_compact_tokens", None), 100000)
        self.assertEqual(model._build_request_payload(context(), (), None)["model"], "served-local")
        self.assertIsNone(model_catalog.get_model_spec("chat-completions", "local-max"))

    def test_direct_library_endpoints_do_not_load_a_home_catalog(self):
        with mock.patch("os.open", side_effect=AssertionError("implicit catalog read")):
            endpoint = chat_endpoint("http://localhost:8000", "local-max",
                                               request_params={"reasoning_effort": "high"})
            payload = ChatCompletionsModel(endpoint)._build_request_payload(context(), (), None)
        self.assertEqual(payload["model"], "local-max")
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_namespace_helpers_require_explicit_catalog_binding(self):
        args = cli._build_parser().parse_args(["--model", "local-max", "--model-catalog", "not-loaded.ini"])
        with self.assertRaisesRegex(ValueError, "Load and bind"):
            build_model(args)
        registry = catalog()
        self.assertEqual(build_model(args, catalog=registry).binding.endpoint.model, "served-local")
        self.assertEqual(InteractionConfig.from_namespace(args, catalog=registry).get("auto_compact_tokens"), 100000)

    def test_unconfigured_chat_sampling_does_not_enable_reasoning(self):
        model = ChatCompletionsModel(chat_endpoint("http://localhost:8000", "literal"))
        for params in (None, SamplingParams(max_output_tokens=77), ResolvedSamplingParams(max_output_tokens=77)):
            payload = model._build_request_payload(context(), (), params)
            self.assertNotIn("thinking", payload)
            self.assertNotIn("reasoning_effort", payload)

    def test_api_inference_does_not_send_ambient_codex_login_to_a_proxy(self):
        args = cli._build_parser().parse_args([
            "--model", "codex-gpt-6-astra",
            "--endpoint-url", "http://localhost:8000/v1/chat/completions",
        ])
        with mock.patch.object(responses, "load_codex_auth", side_effect=AssertionError("credential read")):
            with self.assertRaisesRegex(ValueError, "explicit --endpoint-auth"):
                build_model(args)
        args.model_api = "chat-completions"
        self.assertIsInstance(build_model(args), ChatCompletionsModel)

class AutoCatalogTests(unittest.TestCase):
    def test_context_api_can_disambiguate_a_common_model_without_repeating_it(self):
        registry = catalog(LOCAL.replace("local-max", "codex-gpt-6-astra"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1,
                "defaults": {"model": "codex-gpt-6-astra"},
                "contexts": {"1": {"model_api": "codex"},
                             "2": {"model_api": "chat-completions"},
                             "-1": {"model_api": "codex"}},
            }))
            settings = resolve_config(path, catalog=registry)
            self.assertEqual(settings[1]["model"], "codex-gpt-6-astra")
            self.assertEqual(namespace(settings[1], registry).model_binding.endpoint.model, "gpt-6-astra")
            self.assertEqual(namespace(settings[2], registry).model_binding.endpoint.model, "served-local")

    def test_context_api_inference_clearing_and_params_do_not_leak(self):
        registry = catalog(LOCAL + MESSAGE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1, "contexts": {
                "2": {"model_api": None, "model": "worker"},
            }}))
            settings = resolve_config(path, {"model": "local-max", "model_api": "chat-completions",
                                            "request_params": {"custom": True}}, catalog=registry)
            main = InteractionConfig.from_namespace(namespace(settings[1], registry))
            worker_args = namespace(settings[2], registry)
            worker = InteractionConfig.from_namespace(worker_args)
            self.assertEqual(worker_args.model_api, "messages")
            self.assertEqual(worker.get("auto_compact_tokens"), 60000)
            self.assertEqual(worker.get("request_params"), {})
            self.assertTrue(main.get("request_params")["custom"])
            self.assertIsNone(settings[2]["model_api"])
            self.assertIsNone(settings[2]["request_params"])

    def test_missing_map_inherits_null_resets_and_nested_values_replace(self):
        registry = catalog()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1, "contexts": {
                "2": {"request_params": None},
                "-1": {"request_params": {"thinking": {"type": "disabled"}}},
            }}))
            settings = resolve_config(path, {"model": "local-max", "request_params": {"custom": 1}}, catalog=registry)
            one, two, watch = (InteractionConfig.from_namespace(namespace(settings[i], registry)) for i in (1, 2, -1))
            self.assertEqual(one.get("request_params")["custom"], 1)
            self.assertNotIn("custom", two.get("request_params"))
            self.assertEqual(watch.get("request_params")["thinking"], {"type": "disabled"})
            self.assertEqual(watch.get("request_params")["custom"], 1)

    def test_api_clear_retains_selector_and_old_concrete_api_stays_scoped(self):
        saved = resolve_config(overrides={"model_api": "chat-completions", "model": "worker"})
        registry = catalog(MESSAGE)
        unchanged = resolve_config(saved=saved, catalog=registry)
        self.assertEqual(namespace(unchanged[1], registry).model_api, "chat-completions")
        self.assertIsNone(namespace(unchanged[1], registry).model_binding.spec)
        cleared = resolve_config(saved=saved, overrides={"model_api": None}, catalog=registry)
        self.assertEqual(cleared[1]["model"], "worker")
        self.assertEqual(namespace(cleared[1], registry).model_api, "messages")

    def test_different_provider_route_clears_inherited_credentials(self):
        registry = catalog(CODEX)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1, "contexts": {"2": {"model": "code-env"}}}))
            settings = resolve_config(path, {"model_api": "codex", "model": "codex-gpt-6-astra",
                                            "codex_home": directory}, catalog=registry)
            self.assertIsNone(settings[2]["codex_home"])
            self.assertEqual(settings[1]["codex_home"], directory)

    def test_saved_config_requires_the_current_complete_schema(self):
        registry = catalog()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            settings = resolve_config(overrides={"model": "local-max"}, catalog=registry)
            document = {"version": 1, "contexts": {str(i): dict(s) for i, s in settings.items()}}
            for row in document["contexts"].values():
                row.pop("request_params")
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "Invalid saved"):
                load_saved_config(path)
            path.write_text(json.dumps({
                "version": 1,
                "contexts": {str(i): dict(s) for i, s in settings.items()},
            }))
            saved = load_saved_config(path)
            current = resolve_config(saved=saved, catalog=registry)
            self.assertIsNone(current[1]["request_params"])
            self.assertIsNone(current[1]["model_api"])
            self.assertEqual(namespace(current[1], registry).model_binding.endpoint.model, "served-local")


class CatalogEntrypointTests(unittest.TestCase):
    def test_endpoint_api_flag_and_help_are_independent_of_home_catalog(self):
        for parser in (cli._build_parser(), demo._build_parser(), build_parser()):
            self.assertEqual(
                parser.parse_args(["--endpoint-api", "messages"]).model_api,
                "messages",
            )
        for frontend in (cli, demo, auto):
            with mock.patch("os.open", side_effect=AssertionError("help must not read catalog")), \
                    redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                frontend.main(["--help"])
            self.assertEqual(result.exception.code, 0)

    def test_listing_loads_selected_catalog_without_tty_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.conf"
            path.write_text(HEADER + LOCAL + CODEX)
            for frontend in (cli, demo, auto):
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(io.StringIO()), \
                        mock.patch("os.environ.get", side_effect=no_credentials):
                    result = frontend.main(["--list-models", "--model-catalog", str(path)])
                self.assertEqual(result, 0)
                self.assertIn("local-max", output.getvalue())
                self.assertIn("served-local", output.getvalue())

    def test_cli_debug_binding_snapshot_is_opt_in(self):
        requests = []
        class Response:
            status = 200
            headers = {}
            def read(self):
                return b'{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}'
            def close(self):
                pass
        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return Response()
        original = cli.build_model
        def model_factory(args):
            model = original(args)
            model._opener = opener
            return model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.ini"
            log = Path(directory) / "log.jsonl"
            path.write_text(HEADER + LOCAL)
            stale_sidecar = log.with_name(log.name + ".catalog.json")
            stale_sidecar.write_text("not valid catalog provenance")
            with mock.patch.object(cli, "build_model", side_effect=model_factory), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = cli.main(["--headless", "--enable-default-tools", "false",
                                   "--model-catalog", str(path), "--model", "local-max",
                                   "--prompt", "hello", "--save", str(log)])
            self.assertEqual(result, 0)
            self.assertTrue(log.is_file())
            self.assertFalse(debug_model_binding_path(log).exists())
            self.assertEqual(stale_sidecar.read_text(), "not valid catalog provenance")
            self.assertEqual(requests[0]["model"], "served-local")
            self.assertEqual(requests[0]["reasoning_effort"], "max")
            debug_log = Path(directory) / "debug.jsonl"
            with mock.patch.object(cli, "build_model", side_effect=model_factory), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = cli.main([
                    "--headless", "--enable-default-tools", "false",
                    "--model-catalog", str(path), "--model", "local-max",
                    "--prompt", "hello", "--save", str(debug_log),
                    "--debug-save-model-binding",
                ])
            self.assertEqual(result, 0)
            snapshot_path = debug_model_binding_path(debug_log)
            self.assertTrue(snapshot_path.is_file())
            snapshot = json.loads(snapshot_path.read_text())
            self.assertEqual(snapshot["bindings"]["main"]["selector"], "local-max")
            self.assertEqual(
                snapshot["bindings"]["main"]["endpoint"]["model"],
                "served-local",
            )

    def test_cli_sends_catalog_and_launch_request_params_to_messages(self):
        requests = []
        class Response:
            status = 200
            headers = {}
            def read(self):
                return json.dumps({
                    "type": "message", "role": "assistant", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }).encode("utf-8")
            def close(self):
                pass
        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return Response()
        original = cli.build_model
        def model_factory(args):
            model = original(args)
            model._opener = opener
            return model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.ini"
            path.write_text(HEADER + MESSAGE
                            + 'request_params.thinking = {"type": "enabled", "budget_tokens": 1024}\n')
            with mock.patch.object(cli, "build_model", side_effect=model_factory), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = cli.main(["--headless", "--enable-default-tools", "false",
                                   "--model-catalog", str(path), "--model", "worker",
                                   "--request-params", '{"top_k": 5}',
                                   "--prompt", "hello", "--save", str(Path(directory) / "log.jsonl")])
        self.assertEqual(result, 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["thinking"], {"type": "enabled", "budget_tokens": 1024})
        self.assertEqual(requests[0]["top_k"], 5)
        self.assertEqual(requests[0]["output_config"], {"effort": "high"})
        self.assertEqual(requests[0]["model"], "served-messages")

    def test_demo_debug_binding_snapshot_is_opt_in(self):
        class Response:
            status = 200
            headers = {}
            def read(self):
                return b'{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}'
            def close(self):
                pass
        original = demo.build_model
        def model_factory(args):
            model = original(args)
            model._opener = lambda request, **kwargs: Response()
            return model
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.ini"
            log = Path(directory) / "demo.jsonl"
            catalog_path.write_text(HEADER + LOCAL)
            with mock.patch.object(demo, "_build_model", side_effect=model_factory), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = demo.main([
                    "--model-catalog", str(catalog_path),
                    "--model", "local-max",
                    "--prompt", "hello",
                    "--save", str(log),
                    "--debug-save-model-binding",
                ])
            self.assertEqual(result, 0)
            snapshot = json.loads(debug_model_binding_path(log).read_text())
            self.assertEqual(snapshot["bindings"]["main"]["selector"], "local-max")


class DebugModelBindingTests(unittest.TestCase):
    def test_snapshot_is_readable_and_omits_credential_paths(self):
        binding = catalog().bind(name="local-max")
        codex = BUILTIN_MODEL_CATALOG.bind("codex", "codex-gpt-6-astra")
        codex = replace(
            codex,
            endpoint=replace(codex.endpoint, auth_file="/secret/auth.json"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bindings.json"
            self.assertIsNone(save_debug_model_bindings(
                path, {"main": binding, "codex": codex},
            ))
            document = json.loads(path.read_text())
        entry = document["bindings"]["main"]
        self.assertEqual(document["version"], 1)
        self.assertNotIn("auth_file", document["bindings"]["codex"]["endpoint"])
        self.assertNotIn("/secret/auth.json", json.dumps(document))
        self.assertEqual(entry["request_params"]["reasoning_effort"], "max")

    def test_snapshot_failure_returns_warning_and_removes_temporary_file(self):
        binding = catalog().bind(name="local-max")
        with tempfile.TemporaryDirectory() as directory:
            warning = save_debug_model_bindings(directory, {"main": binding})
            self.assertIn("could not save debug model-binding snapshot", warning)
            self.assertEqual(tuple(Path(directory).iterdir()), ())


if __name__ == "__main__":
    unittest.main()
