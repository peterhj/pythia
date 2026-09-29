"""Offline regression tests for config as the effective frontend policy."""

from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.interaction_helpers import messages_endpoint
from pythia_test.interaction_helpers import responses_endpoint
from pythia_test.interaction_helpers import codex_model

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    CompactionResult, ConfigError, ContextPrefix, Environment, Init,
    InteractionConfig, InteractionConfigSnapshot, InteractionContext, Message,
    ModelConfigurationError, ModelSample, ModelSampleBoundary,
    SampleMetadata, SampleParams, TokenUsage,
    TurnSummary,
)
from pythia.interaction import auto, cli, demo
from pythia.interaction._auto_config import (
    DEFAULTS, build_parser, load_saved_config, namespace, resolve_config,
)
from pythia.interaction.chat_completions import ChatCompletionsEndpoint, ChatCompletionsModel
from pythia.interaction.codex_auth import CodexAuth
from pythia.interaction.messages import MessagesEndpoint, MessagesModel, MessagesServerCompaction
from pythia.interaction.responses import CodexResponsesModel, StreamingResponsesEndpoint


def arguments(api="codex", model="codex-gpt-6-astra", **overrides):
    args = cli._build_parser().parse_args([])
    args.model_api, args.model = api, model
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def previous_context(tokens=900_000):
    return InteractionContext((
        Init("old"), Message("assistant", "previous answer"),
        SampleMetadata(TokenUsage(total_tokens=tokens)), ModelSampleBoundary(),
        TurnSummary(sample_count=1, context_tokens=tokens), Message("user", "continue"),
    ))


class CaptureMessages(MessagesModel):
    def __init__(self, trigger=None):
        super().__init__(messages_endpoint(
            api_url="https://api.anthropic.com", model="claude-fable-5.1",
            max_output_tokens=321,
            server_compaction=MessagesServerCompaction(trigger_input_tokens=trigger),
        ))
        self.payloads = []

    @property
    def auto_compact_context_tokens(self):
        raise AssertionError("frontend must not look up a model threshold")

    def sample(self, context, *, tools=(), sample_params=None):
        self.payloads.append(self._build_request_payload(context, tools, sample_params))
        return ModelSample((Message("assistant", "done"),))


class CaptureHost:
    auto_compaction_owner = "host"

    def __init__(self):
        self.sample_params = []

    @property
    def auto_compact_context_tokens(self):
        raise AssertionError("frontend must not look up a model threshold")

    def sample(self, context, *, tools=(), sample_params=None):
        self.sample_params.append(sample_params)
        return ModelSample((Message("assistant", "done"),))


class ConfigSeedingTests(unittest.TestCase):
    def test_catalog_seeds_current_and_initial_without_constructing_a_model(self):
        thinking = {"thinking": {"type": "adaptive"}}
        for api, model, budget, extra_sample_params in (
            ("codex", "codex-gpt-6-astra", None, {}),
            ("messages", "claude-fable-5.1", 128_000, thinking),
            ("messages", "claude-fable-5.1-max", 128_000, {**thinking, "output_config": {"effort": "max"}}),
        ):
            with self.subTest(api=api, model=model), mock.patch.object(
                CodexResponsesModel, "__init__", side_effect=AssertionError("credentials"),
            ):
                config = InteractionConfig.from_namespace(arguments(api, model))
                self.assertEqual(config.snapshot(), config.initial_snapshot())
                self.assertEqual(config.get("auto_compact_tokens"), 872_000)
                self.assertEqual(config.get("max_context_tokens"), 1_000_000)
                self.assertEqual(config.get("max_output_tokens"), budget)
                # Catalog extras (thinking, effort) seed the read-only view; the
                # projection leaves them to the binding, as for every request.
                self.assertEqual(config.get("extra_sample_params"), extra_sample_params)
                self.assertEqual(config.snapshot().sample_params(), SampleParams(
                    auto_compact_tokens=872_000, max_output_tokens=budget,
                    enable_auto_compaction=True,
                ))

    def test_unknown_or_wrong_profile_never_inherits_codex_limits(self):
        for api, model in (
            ("chat-completions", "codex-gpt-6-astra"), ("responses", "codex-gpt-6-astra"),
            ("codex", "unknown"), ("codex", "muse-spark-1.3"),
        ):
            with self.subTest(api=api, model=model):
                config = InteractionConfig.from_namespace(arguments(api, model))
                self.assertIsNone(config.get("auto_compact_tokens"))
                self.assertIsNone(config.get("max_context_tokens"))
                self.assertIsNone(config.get("max_output_tokens"))

    def test_python_generic_responses_binding_does_not_inherit_codex_limits(self):
        model = codex_model(responses_endpoint(
            api_url="https://api.openai.com/v1", model="codex-gpt-6-astra", bearer_token="FAKE",
        ))
        config = InteractionConfig.from_model(model)
        self.assertEqual(config.snapshot(), InteractionConfigSnapshot())

    def test_python_endpoint_preferences_seed_but_do_not_override_runtime_config(self):
        model = CaptureMessages(trigger=150_000)
        config = InteractionConfig.from_model(model)
        self.assertEqual(config.initial_values()["auto_compact_tokens"], 150_000)
        self.assertEqual(config.initial_values()["max_output_tokens"], 321)
        config.set("auto_compact_tokens", None)
        config.set("max_output_tokens", 77)
        payload = model._build_request_payload(previous_context(), (), config.snapshot().sample_params())
        self.assertEqual(payload["max_tokens"], 77)
        self.assertEqual(payload["context_management"]["edits"][0]["trigger"]["value"], 872_000)

    def test_null_resolves_immediately_but_reset_restores_launch_override(self):
        config = InteractionConfig.from_namespace(arguments(
            auto_compact_tokens=500_000, max_context_tokens=950_000,
        ))
        initial = config.initial_snapshot()
        config.set("auto_compact_tokens", 100_000)
        config.set("max_context_tokens", 1_000)  # Informational, not a constraint.
        self.assertEqual(config.set("auto_compact_tokens", None), 872_000)
        self.assertEqual(config.set("max_context_tokens", None), 1_000_000)
        self.assertEqual(config.get("auto_compact_tokens"), 872_000)
        self.assertIn("auto_compact_tokens = 872000", config.render(json_output=False))
        payload = json.loads(config.render(json_output=True))
        self.assertEqual(payload["auto_compact_tokens"], 872_000)
        self.assertEqual(payload["__init__"]["auto_compact_tokens"], 500_000)
        self.assertEqual(config.snapshot().sample_params().auto_compact_tokens, 872_000)
        config.reset("auto_compact_tokens")
        self.assertEqual(config.get("auto_compact_tokens"), 500_000)
        self.assertEqual(config.get("max_context_tokens"), 1_000_000)
        self.assertEqual(config.reset(), initial)
        self.assertIs(config.initial_snapshot(), initial)

    def test_messages_validation_is_atomic_even_while_disabled(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                args = arguments("messages", "claude-fable-5.1", enable_auto_compaction=enabled)
                config = InteractionConfig.from_namespace(args)
                before = config.snapshot()
                with self.assertRaisesRegex(ConfigError, "50000"):
                    config.set("auto_compact_tokens", 49_999)
                self.assertIs(config.snapshot(), before)
                args.auto_compact_tokens = 49_999
                with self.assertRaisesRegex(ConfigError, "50000"):
                    InteractionConfig.from_namespace(args)

    def test_reset_workspace_callback_is_atomic(self):
        changes = []
        def apply(value):
            if value:
                raise RuntimeError("cannot enable")
            changes.append(value)
        config = InteractionConfig(on_enable_workspace=apply)
        config.set("enable_workspace", False)
        before = config.snapshot()
        with self.assertRaisesRegex(RuntimeError, "cannot enable"):
            config.reset()
        self.assertIs(config.snapshot(), before)
        self.assertEqual(changes, [False])

    def test_equal_visible_policy_has_equal_behavior_regardless_of_history(self):
        config = InteractionConfig()
        initial = config.snapshot().sample_params()
        config.set("enable_auto_compaction", False)
        config.set("enable_auto_compaction", True)
        self.assertEqual(config.snapshot().sample_params(), initial)
        self.assertEqual(config.values(), config.initial_values())
        config.set("auto_compact_tokens", 100)
        self.assertIn("# init: auto_compact_tokens = None", config.render(json_output=False))
        self.assertIsNone(config.set("auto_compact_tokens", None))
        self.assertIsNone(config.snapshot().sample_params().auto_compact_tokens)

    def test_projection_keeps_per_call_fields_but_not_old_policy(self):
        base = SampleParams(max_output_tokens=7, auto_compact_tokens=8,
                            enable_auto_compaction=False, temperature=0.5,
                            top_p=0.8, stop=("stop",), seed=4, extra={"top_k": 5})
        snapshot = InteractionConfigSnapshot(
            max_output_tokens=10, extra_sample_params={"thinking": {"type": "enabled"}},
        )
        self.assertEqual(snapshot.sample_params(base), SampleParams(
            max_output_tokens=10, enable_auto_compaction=True, temperature=0.5,
            top_p=0.8, stop=("stop",), seed=4, extra={"top_k": 5},
        ))
        # The config's extras are a read-only view; requests inherit the binding's.
        self.assertIsNone(snapshot.sample_params().extra)


class SampleParamsPrecedenceTests(unittest.TestCase):
    def test_messages_per_call_policy_beats_endpoint_and_catalog(self):
        model = CaptureMessages(trigger=150_000)
        for threshold, expected in ((100_000, 100_000), (None, 150_000)):
            with self.subTest(threshold=threshold):
                payload = model._build_request_payload(previous_context(), (), SampleParams(
                    max_output_tokens=77, auto_compact_tokens=threshold,
                ))
                self.assertEqual(payload["max_tokens"], 77)
                edit = payload["context_management"]["edits"][0]
                self.assertEqual(edit["trigger"]["value"], expected)
        payload = model._build_request_payload(previous_context(), (), SampleParams(
            max_output_tokens=77, enable_auto_compaction=False,
        ))
        self.assertNotIn("context_management", payload)
        self.assertEqual(model.endpoint.server_compaction.trigger_input_tokens, 150_000)

    def test_messages_unset_fields_inherit_endpoint_then_catalog(self):
        for params in (None, SampleParams()):
            with self.subTest(params=params):
                payload = CaptureMessages()._build_request_payload(previous_context(), (), params)
                self.assertEqual(payload["max_tokens"], 321)
                # No endpoint trigger, so the catalog threshold applies.
                self.assertEqual(payload["context_management"]["edits"][0]["trigger"]["value"], 872_000)

    def test_messages_per_call_trigger_is_validated_not_ignored(self):
        with self.assertRaisesRegex(ModelConfigurationError, "at least"):
            CaptureMessages(trigger=150_000)._build_request_payload(
                previous_context(), (), SampleParams(auto_compact_tokens=49_999),
            )

    def test_enabling_messages_compaction_without_endpoint_policy(self):
        model = MessagesModel(messages_endpoint(
            api_url="https://api.anthropic.com", model="claude-fable-5.1",
        ))
        payload = model._build_request_payload(previous_context(), (), SampleParams(
            max_output_tokens=77, enable_auto_compaction=True,
        ))
        self.assertEqual(payload["context_management"]["edits"], [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 872_000},
        }])
        uncatalogued = MessagesModel(messages_endpoint(
            api_url="https://api.anthropic.com", model="custom-claude", max_output_tokens=77,
        ))
        payload = uncatalogued._build_request_payload(previous_context(), (), SampleParams(
            enable_auto_compaction=True,
        ))
        # Without an endpoint or catalog trigger, the server default applies.
        self.assertEqual(payload["context_management"]["edits"], [{"type": "compact_20260112"}])

    def test_default_responses_does_not_turn_output_ceiling_into_budget(self):
        model = codex_model(model="codex-gpt-6-astra", auth=CodexAuth("FAKE"))
        config = InteractionConfig.from_namespace(arguments())
        payload, _ = model._build_request_payload(previous_context(), (), config.snapshot().sample_params())
        for key in ("max_output_tokens", "auto_compact_tokens", "max_context_tokens", "enable_auto_compaction"):
            self.assertNotIn(key, payload)

    def test_host_only_options_do_not_change_chat_sampling_payload(self):
        model = ChatCompletionsModel(chat_endpoint(api_url="http://localhost:8000"))
        payload = model._build_request_payload(previous_context(), (), SampleParams(
            enable_auto_compaction=True, auto_compact_tokens=100_000,
        ))
        ordinary = model._build_request_payload(previous_context(), (), None)
        self.assertEqual(payload, ordinary)


class FrontendPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_messages_has_only_server_compaction(self):
        for threshold, enabled in ((None, True), (100_000, True), (100_000, False)):
            with self.subTest(threshold=threshold, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                config = InteractionConfig.from_namespace(arguments(
                    "messages", "claude-fable-5.1", auto_compact_tokens=threshold,
                    enable_auto_compaction=enabled,
                ))
                model = CaptureMessages(trigger=150_000)
                with mock.patch.object(cli, "create_default_compactor", side_effect=AssertionError("host compaction")):
                    await cli._turn(previous_context(), model, Environment(), cli._UIState(headless=True),
                                    Path(directory) / "log.jsonl", config)
                self.assertEqual(len(model.payloads), 1)
                if enabled:
                    self.assertEqual(model.payloads[0]["context_management"]["edits"][0]["trigger"]["value"],
                                     config.get("auto_compact_tokens"))
                else:
                    self.assertNotIn("context_management", model.payloads[0])

    async def test_cli_host_uses_only_config_including_none(self):
        for threshold in (None, 100):
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as directory:
                model = CaptureHost()
                config = InteractionConfig(InteractionConfigSnapshot(auto_compact_tokens=threshold))
                compactor = mock.Mock()
                compactor.compact.return_value = CompactionResult((ContextPrefix((Message("assistant", "summary"),)),))
                with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
                    await cli._turn(previous_context(), model, Environment(), cli._UIState(headless=True),
                                    Path(directory) / "log.jsonl", config)
                self.assertEqual(compactor.compact.call_count, int(threshold is not None))
                self.assertEqual(model.sample_params, [config.snapshot().sample_params()])

    def test_auto_turn_uses_context_policy_not_model_metadata(self):
        session = SimpleNamespace(
            _check_running=lambda: None, _phase=lambda *args: None, _emit=lambda *args: None,
            _checkpoint=lambda index, context, items: context.extend(items),
        )
        for model in (CaptureHost(), CaptureMessages(trigger=150_000)):
            for threshold in (None, 100_000):
                with self.subTest(model=type(model), threshold=threshold):
                    config = InteractionConfigSnapshot(auto_compact_tokens=threshold, max_output_tokens=77)
                    compactor = mock.Mock()
                    compactor.compact.return_value = CompactionResult((ContextPrefix((Message("assistant", "summary"),)),))
                    with mock.patch.object(auto, "create_default_compactor", return_value=compactor):
                        result = auto._Session._turn(session, 1, model, Environment(), config, previous_context())
                    self.assertEqual(result, "done")
                    self.assertEqual(compactor.compact.call_count,
                                     int(isinstance(model, CaptureHost) and threshold is not None))

    def test_demo_normalizes_keyword_and_options_before_sampling(self):
        for kwargs in (
            {"auto_compact_tokens": 100_000},
            {"sample_params": SampleParams(auto_compact_tokens=100_000)},
            {"auto_compact_tokens": 100_000, "sample_params": SampleParams(auto_compact_tokens=100_000)},
        ):
            with self.subTest(kwargs=kwargs):
                model = CaptureMessages()
                with redirect_stdout(io.StringIO()), mock.patch.object(
                    demo, "create_default_compactor", side_effect=AssertionError("host compaction"),
                ):
                    demo.run(model, Environment(), prompt="hello", **kwargs)
                self.assertEqual(model.payloads[0]["context_management"]["edits"][0]["trigger"]["value"], 100_000)
                self.assertEqual(model.payloads[0]["max_tokens"], 321)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            demo.run(CaptureMessages(), Environment(), auto_compact_tokens=100_000,
                     sample_params=SampleParams(auto_compact_tokens=200_000))

    def test_demo_binds_endpoint_policy_once_and_keeps_per_call_extra(self):
        model = CaptureMessages(trigger=150_000)
        with redirect_stdout(io.StringIO()):
            demo.run(model, Environment(), prompt="hello", sample_params=SampleParams(
                max_output_tokens=77, extra={"thinking": {"type": "disabled"}},
            ))
        payload = model.payloads[0]
        self.assertEqual(payload["max_tokens"], 77)
        self.assertEqual(payload["context_management"]["edits"][0]["trigger"]["value"], 150_000)
        # The per-call map replaces the catalog's adaptive thinking.
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_demo_samples_after_custom_metadata_is_bound_once(self):
        class OneReadModel(CaptureHost):
            reads = 0
            @property
            def auto_compact_context_tokens(self):
                self.reads += 1
                if self.reads != 1:
                    raise AssertionError("late model lookup")
                return 100
        model = OneReadModel()
        with redirect_stdout(io.StringIO()):
            demo.run(model, Environment(), prompt="hello")
        self.assertEqual(model.reads, 1)
        self.assertEqual(model.sample_params[0].auto_compact_tokens, 100)


class AutoSeedingTests(unittest.TestCase):
    def test_common_inputs_are_not_global_live_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({
                "version": 1,
                "defaults": {"model_api": "codex", "model": "codex-gpt-6-astra", "auto_compact_tokens": 600_000},
                "contexts": {
                    "2": {"model_api": "messages", "model": "claude-fable-5.1",
                          "auto_compact_tokens": None, "max_context_tokens": None},
                    "-1": {"model": "muse-spark-1.3", "auto_compact_tokens": None},
                },
            }))
            args = build_parser().parse_args([
                "--auto-compact-tokens", "500000", "--max-context-tokens", "950000",
            ])
            raw = resolve_config(path, {key: value for key, value in vars(args).items() if key in DEFAULTS})
            configs = {i: InteractionConfig.from_namespace(namespace(value)) for i, value in raw.items()}
            self.assertEqual(configs[1].get("auto_compact_tokens"), 500_000)
            self.assertEqual(configs[2].get("auto_compact_tokens"), 872_000)
            self.assertEqual(configs[1].get("max_context_tokens"), 950_000)
            self.assertEqual(configs[2].get("max_context_tokens"), 1_000_000)
            self.assertIsNone(configs[-1].get("auto_compact_tokens"))
            self.assertIsNone(raw[2]["auto_compact_tokens"])
            configs[1].set("auto_compact_tokens", 100_000)
            self.assertEqual(configs[2].get("auto_compact_tokens"), 872_000)
            saved = Path(directory) / "saved.json"
            saved.write_text(json.dumps({"version": 1, "contexts": {str(i): s for i, s in raw.items()}}))
            resumed = resolve_config(saved=load_saved_config(saved))
            self.assertEqual(resumed, raw)

    def test_saved_config_rejects_missing_current_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            raw = resolve_config(overrides={"model_api": "codex", "model": "codex-gpt-6-astra"})
            for settings in raw.values():
                del settings["auto_compact_tokens"], settings["max_context_tokens"]
            document = {"version": 1, "contexts": {str(i): s for i, s in raw.items()}}
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "Invalid saved"):
                load_saved_config(path)

    def test_new_cli_defaults_are_suppressed_and_values_are_validated_per_context(self):
        args = build_parser().parse_args([])
        for key in ("auto_compact_tokens", "max_context_tokens"):
            self.assertFalse(hasattr(args, key))
            for value in (0, -1, True, "100"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resolve_config(overrides={key: value})
        with self.assertRaisesRegex(ValueError, "50000"):
            resolve_config(overrides={"model_api": "messages", "model": "claude-fable-5.1",
                                      "auto_compact_tokens": 49_999})


if __name__ == "__main__":
    unittest.main()
