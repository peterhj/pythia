from __future__ import annotations

from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.interaction_helpers import messages_endpoint
from pythia_test.interaction_helpers import responses_endpoint
from pythia_test.interaction_helpers import codex_model

from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from dataclasses import replace
import importlib.util
import sys
import unittest
from unittest import mock

import pythia.interaction as interaction
from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import CodexAuth
from pythia.interaction import CodexResponsesModel
from pythia.interaction import Init
from pythia.interaction import Message
from pythia.interaction import MessagesDefaults
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import ModelConfigurationError
from pythia.interaction import InteractionContext
from pythia.interaction import ModelLimits
from pythia.interaction import ModelSpec
from pythia.interaction import ResponsesDefaults
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import cli
from pythia.interaction import demo
from pythia.interaction import get_model_spec
from pythia.interaction import list_model_specs
from pythia.interaction import messages
from pythia.interaction import model_catalog
from pythia.interaction import model_config
from pythia.interaction import responses
from pythia.interaction.model_config import build_model
from pythia.interaction.model_config import supports_account_services


_CODEX_PRESETS = (
    ("codex-gpt-5.6-sol", "gpt-5.6-sol", {}, None, "chatgpt"),
    ("codex-gpt-5.6-sol-medium", "gpt-5.6-sol", {"effort": "medium"}, None, "chatgpt"),
    ("codex-gpt-5.6-sol-max", "gpt-5.6-sol", {"effort": "max"}, None, "chatgpt"),
    ("codex-gpt-6-astra", "gpt-6-astra", {"summary": "auto"}, "low", "chatgpt"),
    ("codex-gpt-6-astra-medium", "gpt-6-astra", {"effort": "medium", "summary": "auto"}, "low", "chatgpt"),
    ("codex-gpt-6-astra-max", "gpt-6-astra", {"effort": "max", "summary": "auto"}, "low", "chatgpt"),
    ("codex-gpt-6-sol", "gpt-6-sol", {"summary": "auto"}, "low", "chatgpt"),
    ("codex-gpt-6-sol-medium", "gpt-6-sol", {"effort": "medium", "summary": "auto"}, "low", "chatgpt"),
    ("codex-gpt-6-sol-max", "gpt-6-sol", {"effort": "max", "summary": "auto"}, "low", "chatgpt"),
    ("codex-gpt-6-luna", "gpt-6-luna", {"summary": "auto"}, "low", "chatgpt"),
    ("codex-gpt-6-luna-medium", "gpt-6-luna", {"effort": "medium", "summary": "auto"}, "low", "chatgpt"),
    ("codex-gpt-6-luna-max", "gpt-6-luna", {"effort": "max", "summary": "auto"}, "low", "chatgpt"),
    ("muse-spark-1.3", "muse-spark-1.3-contributor", {}, None, "meta"),
    ("muse-spark-1.3-xhigh", "muse-spark-1.3-contributor", {"effort": "xhigh"}, None, "meta"),
)


class ModelCatalogTests(unittest.TestCase):
    def test_presets_preserve_routes_defaults_limits_and_request_payloads(self):
        self.assertEqual(tuple(spec.name for spec in list_model_specs("codex")),
                         tuple(row[0] for row in _CODEX_PRESETS))
        for name, wire_model, reasoning, verbosity, provider in _CODEX_PRESETS:
            with self.subTest(name=name):
                spec = get_model_spec("codex", name)
                self.assertIsInstance(spec, ModelSpec)
                self.assertEqual(spec.endpoint.model, wire_model)
                self.assertEqual(spec.responses, ResponsesDefaults(
                    reasoning_effort=reasoning.get("effort"),
                    reasoning_summary=reasoning.get("summary"), text_verbosity=verbosity,
                ))
                expected_url = (model_catalog.META_RESPONSES_API_URL if provider == "meta"
                                else model_catalog.CODEX_RESPONSES_API_URL)
                self.assertEqual(spec.endpoint.url, expected_url + "/responses")
                self.assertEqual(
                    spec.endpoint.auth,
                    "env:META_API_KEY" if provider == "meta" else "codex-login",
                )
                limits = (
                    ModelLimits()
                    if provider == "meta"
                    else ModelLimits(872_000, 1_000_000, 128_000)
                )
                self.assertEqual(spec.limits, limits)
                self.assertTrue(spec.source)

                model = codex_model(model=name, auth=CodexAuth("FAKE"),
                                            identifier_factory=lambda: "fixed-turn")
                context = InteractionContext((Init("session"), Message("user", "Hello.")))
                payload, _ = model._build_request_payload(context, (), None)
                self.assertEqual(model.endpoint.url, expected_url + "/responses")
                self.assertEqual(model.endpoint.model, wire_model)
                self.assertEqual(payload["model"], wire_model)
                self.assertEqual(payload.get("reasoning", {}), reasoning)
                self.assertEqual(payload.get("text"), None if verbosity is None else {"verbosity": verbosity})
                self.assertEqual(
                    model.auto_compact_context_tokens,
                    limits.auto_compact_context_tokens,
                )
                self.assertEqual(model.max_context_tokens, limits.max_context_tokens)
                self.assertEqual(model.max_output_tokens, limits.max_output_tokens)
                for field in (
                    "auto_compact_context_tokens",
                    "max_context_tokens",
                    "max_output_tokens",
                ):
                    self.assertNotIn(field, payload)

    def test_aliases_are_identical_but_effort_presets_are_distinct(self):
        fable = get_model_spec("messages", "claude-fable-5-1")
        self.assertIs(get_model_spec("messages", " claude-fable-5.1 "), fable)
        self.assertEqual(fable.aliases, ("claude-fable-5.1",))
        self.assertEqual(
            fable.limits,
            ModelLimits(
                auto_compact_context_tokens=872_000,
                max_context_tokens=1_000_000,
                max_output_tokens=128_000,
            ),
        )
        self.assertEqual(fable.endpoint.auth, "env:ANTHROPIC_API_KEY")
        self.assertIsNone(fable.responses)
        self.assertIsNone(fable.messages)

        fable_max = get_model_spec("messages", "claude-fable-5-1-max")
        self.assertIs(
            get_model_spec("messages", " claude-fable-5.1-max "),
            fable_max,
        )
        self.assertEqual(fable_max.aliases, ("claude-fable-5.1-max",))
        self.assertEqual(fable_max.endpoint.model, "claude-fable-5-1")
        self.assertIs(fable_max.limits, fable.limits)
        self.assertIs(fable_max.endpoint, fable.endpoint)
        self.assertEqual(
            fable_max.messages,
            MessagesDefaults(
                output_effort="max",
            ),
        )

        self.assertEqual(len(list_model_specs()), 20)
        for base_name, preset_name in (("codex-gpt-5.6-sol", "codex-gpt-5.6-sol-medium"),
                                       ("codex-gpt-6-astra", "codex-gpt-6-astra-max"),
                                       ("muse-spark-1.3", "muse-spark-1.3-xhigh")):
            base = get_model_spec("codex", base_name)
            preset = get_model_spec("codex", preset_name)
            self.assertIsNot(base, preset)
            self.assertIs(base.limits, preset.limits)
            self.assertIs(base.endpoint, preset.endpoint)
            self.assertNotEqual(base.responses, preset.responses)
        with self.assertRaises(ValueError):
            get_model_spec("codex-responses", "codex-gpt-6-astra")

    def test_fable_max_uses_output_effort(self):
        context = InteractionContext((Message("user", "Hello."),))
        cases = (
            ("claude-fable-5-1", None),
            ("claude-fable-5.1", None),
            ("claude-fable-5-1-max", "max"),
            ("claude-fable-5.1-max", "max"),
        )
        for name, effort in cases:
            with self.subTest(name=name):
                model = MessagesModel(messages_endpoint(
                    api_url="https://api.anthropic.com",
                    model=name,
                    max_output_tokens=100,
                    api_key="FAKE",
                ))

                payload = model._build_request_payload(context, (), None)

                self.assertEqual(payload["model"], "claude-fable-5-1")
                self.assertNotIn("reasoning", payload)
                self.assertNotIn("thinking", payload)
                if effort is None:
                    self.assertNotIn("output_config", payload)
                else:
                    self.assertEqual(
                        payload["output_config"],
                        {"effort": effort},
                    )

    def test_unknown_models_and_other_profiles_do_not_inherit_presets(self):
        for profile in ("responses", "chat-completions"):
            self.assertEqual(list_model_specs(profile), ())
            for spec in list_model_specs():
                for name in (spec.name, *spec.aliases):
                    self.assertIsNone(get_model_spec(profile, name))
        # Bare wire model IDs are not catalog selectors.
        for name in (None, "", "unknown", "CODEX-GPT-6-ASTRA", "codex-gpt-6-astra-low",
                     "codex-gpt-5.6-sol-high", "muse-spark-1.3-max",
                     "gpt-5.6-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"):
            with self.subTest(name=name):
                self.assertIsNone(get_model_spec("codex", name))
        for name in ("claude-fable-5.2", "claude-fable-5-1-20260901",
                     "claude-fable-5-1-high", "codex-gpt-6-astra"):
            self.assertIsNone(get_model_spec("messages", name))
        self.assertIsNone(get_model_spec("codex", "claude-fable-5.1"))

    def test_generic_responses_and_chat_completions_remain_pass_through(self):
        for spec in list_model_specs():
            for name in (spec.name, *spec.aliases):
                with self.subTest(name=name):
                    context = InteractionContext((Message("user", "Hello."),))
                    model = codex_model(responses_endpoint(
                        api_url="https://proxy.example.test/v1", model=name, bearer_token="FAKE",
                    ))
                    payload, state = model._build_request_payload(context, (), None)
                    self.assertEqual(payload["model"], name)
                    self.assertNotIn("reasoning", payload)
                    self.assertNotIn("text", payload)
                    self.assertIsNone(model.auto_compact_context_tokens)
                    self.assertIsNone(model.max_context_tokens)
                    self.assertIsNone(model.max_output_tokens)
                    self.assertNotIn("session_id", model._build_headers(state))
                    chat = ChatCompletionsModel(chat_endpoint("http://localhost", model=name))
                    self.assertEqual(chat._build_request_payload(context, (), None)["model"], name)

    def test_catalog_is_immutable_and_rejects_colliding_selectors(self):
        spec = get_model_spec("messages", "claude-fable-5-1")
        for obj, field, value in ((spec, "name", "other"), (spec.limits, "max_context_tokens", 1),
                                  (spec.endpoint, "url", "https://other.test")):
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, field, value)
        with self.assertRaises(TypeError):
            model_catalog._MODEL_INDEX[("messages", "other")] = spec
        for specs in (
            (spec, spec),
            (spec, replace(spec, name="other")),  # Colliding alias.
            (spec, replace(spec, name="claude-fable-5.1", aliases=())),
            (replace(spec, aliases=(spec.name,)),),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate model selector"):
                model_catalog._build_index(specs)
        other_profile = replace(
            spec,
            endpoint=replace(spec.endpoint, api="responses"),
        )
        self.assertEqual(len(model_catalog._build_index((spec, other_profile))), 4)

    def test_catalog_validation(self):
        for field in (
            "auto_compact_context_tokens",
            "max_context_tokens",
            "max_output_tokens",
        ):
            for value in (True, 0, -1, 1.5, "100"):
                with self.assertRaises(ValueError):
                    ModelLimits(**{field: value})
        with self.assertRaises(ValueError):
            ModelLimits(auto_compact_context_tokens=200, max_context_tokens=100)
        spec = get_model_spec("messages", "claude-fable-5-1")
        for fields in ({"name": ""}, {"endpoint": "bad"},
                       {"aliases": "alias"}, {"aliases": [" "]},
                       {"responses": ResponsesDefaults(reasoning_effort="max")}):
            with self.assertRaises((TypeError, ValueError)):
                replace(spec, **fields)
        with self.assertRaises((TypeError, ValueError)):
            replace(
                get_model_spec("codex", "codex-gpt-6-astra"),
                messages=MessagesDefaults(
                    output_effort="max",
                ),
            )
        for fields in (
            {"output_effort": ""},
            {"output_effort": "max effort"},
            {"output_effort": True},
        ):
            with self.assertRaises((TypeError, ValueError)):
                MessagesDefaults(**fields)
        with self.assertRaises(ValueError):
            get_model_spec("unknown-profile", "model")

    def test_import_and_lookup_do_not_access_credentials_or_network(self):
        module_name = "_isolated_interaction_model_catalog"
        module_spec = importlib.util.spec_from_file_location(module_name, model_catalog.__file__)
        module = importlib.util.module_from_spec(module_spec)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {module_name: module}))
            for target in ("os.environ.get", "builtins.open", "pathlib.Path.read_text",
                           "urllib.request.urlopen"):
                stack.enter_context(mock.patch(target, side_effect=AssertionError("unexpected effect")))
            module_spec.loader.exec_module(module)
            self.assertEqual(len(module.list_model_specs()), 20)
            self.assertEqual(
                module.get_model_spec("codex", "muse-spark-1.3").endpoint.auth,
                "env:META_API_KEY",
            )
            self.assertIsNone(module.get_model_spec("responses", "codex-gpt-6-astra"))

    def test_cli_and_demo_help_list_catalog_presets_and_aliases(self):
        for frontend in (cli, demo):
            parser = frontend._build_parser()
            model_help = next(action.help for action in parser._actions if action.dest == "model")
            for spec in list_model_specs():
                for name in (spec.name, *spec.aliases):
                    self.assertIn(name, model_help)
            self.assertIn("META_API_KEY", model_help)
            self.assertNotIn("codex-gpt-6-astra-low", model_help)
            self.assertNotIn("muse-spark-1.3-max", model_help)
            self.assertEqual(parser.parse_args(["--model", "future-model"]).model, "future-model")


class CatalogAuthParityTests(unittest.TestCase):
    def test_meta_environment_defaults_and_explicit_auth_url_and_file_precedence(self):
        with mock.patch.dict("os.environ", {"META_API_KEY": " env-token "}, clear=True):
            model = codex_model(model="muse-spark-1.3")
            self.assertEqual(model.endpoint.bearer_token, "env-token")
            with mock.patch.object(responses, "_load_default_model_auth",
                                   side_effect=AssertionError("must use explicit auth")):
                model = codex_model(
                    model="muse-spark-1.3", auth=CodexAuth("explicit-token"),
                    api_url="https://proxy.example.test", request_timeout_seconds=7,
                )
            self.assertEqual(model.endpoint.bearer_token, "explicit-token")
            self.assertEqual(
                model.endpoint.url,
                "https://proxy.example.test/responses",
            )
            self.assertEqual(model.endpoint.request_timeout_seconds, 7)
            for options in ({"codex_home": "/fake-home"}, {"auth_file": "/fake-auth.json"}):
                with self.subTest(options=options):
                    with mock.patch.object(responses, "load_codex_auth", return_value=CodexAuth("file-token")) as load:
                        model = codex_model(model="muse-spark-1.3", **options)
                    load.assert_called_once_with(
                        auth_file=model.binding.endpoint.auth_file,
                    )
                    self.assertEqual(model.endpoint.bearer_token, "file-token")
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ModelConfigurationError, "META_API_KEY"):
                codex_model(model="muse-spark-1.3-xhigh")

    def test_frontend_messages_auth_and_chat_defaults_do_not_cross_profiles(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "anthropic-token",
                                            "META_API_KEY": "meta-token"}, clear=True):
            for flags, expected in (([], "anthropic-token"), ([
                "--endpoint-auth", "supplied",
                "--endpoint-api-key", "explicit",
            ], "explicit")):
                args = demo._build_parser().parse_args([
                    "--endpoint-api", "messages", "--model", "claude-fable-5.1",
                    "--max-output-tokens", "100", *flags,
                ])
                self.assertEqual(build_model(args).endpoint.api_key, expected)
            args = demo._build_parser().parse_args([
                "--endpoint-api", "chat-completions",
                "--model", "muse-spark-1.3",
            ])
            model = build_model(args)
            self.assertEqual(
                model.endpoint.url,
                "http://127.0.0.1:8000/v1/chat/completions",
            )
            self.assertIsNone(model.endpoint.api_key)
            self.assertNotIn("anthropic-token", repr(list_model_specs()))
            self.assertNotIn("meta-token", repr(list_model_specs()))

    def test_account_services_require_actual_trusted_route_not_just_model_identity(self):
        parser = cli._build_parser()
        official = parser.parse_args([
            "--endpoint-api", "codex", "--model", "codex-gpt-6-astra",
        ])
        self.assertTrue(supports_account_services(official))

        custom = parser.parse_args([
            "--endpoint-api", "codex", "--model", "codex-gpt-6-astra",
            "--endpoint-url", "https://proxy.example.test/responses",
            "--endpoint-auth", "none",
        ])
        self.assertFalse(supports_account_services(custom))

        wrong_auth = parser.parse_args([
            "--endpoint-api", "codex", "--model", "codex-gpt-6-astra",
            "--endpoint-auth", "env:OTHER_TOKEN",
        ])
        self.assertFalse(supports_account_services(wrong_auth))

        messages_args = parser.parse_args([
            "--endpoint-api", "messages", "--model", "codex-gpt-6-astra",
        ])
        self.assertFalse(supports_account_services(messages_args))


if __name__ == "__main__":
    unittest.main()
