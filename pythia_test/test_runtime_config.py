from __future__ import annotations

import json
import unittest

from pythia.interaction import BUILTIN_MODEL_CATALOG
from pythia.interaction import CONFIG_KEYS
from pythia.interaction import CompactionSettings
from pythia.interaction import ConfigError
from pythia.interaction import InteractionConfig
from pythia.interaction import InteractionConfigSnapshot
from pythia.interaction import ModelConfigurationError
from pythia.interaction import SampleParams
from pythia.interaction import cli
from pythia.interaction.runtime_config import parse_config_literal
from pythia.interaction.runtime_config import resolve_compaction_mode


class InteractionConfigTests(unittest.TestCase):
    def test_output_limit_uses_the_explicit_shared_name(self):
        self.assertIn("max_output_tokens", CONFIG_KEYS)
        self.assertNotIn("max_tokens", CONFIG_KEYS)
        self.assertTrue(hasattr(SampleParams(), "max_output_tokens"))
        self.assertFalse(hasattr(SampleParams(), "max_tokens"))
        self.assertTrue(
            hasattr(InteractionConfigSnapshot(), "max_output_tokens")
        )
        self.assertFalse(hasattr(InteractionConfigSnapshot(), "max_tokens"))

    def test_defaults_and_python_and_json_dumps_are_stable(self):
        config = InteractionConfig()

        # A bare config binds the unresolved mode to pi.
        self.assertEqual(
            config.snapshot(), InteractionConfigSnapshot(compaction_mode="pi"),
        )
        self.assertEqual(
            config.render(json_output=False),
            "\n".join((
                "enable_workspace = True",
                "max_samples = None",
                "max_output_tokens = None",
                "enable_auto_compaction = True",
                "auto_compact_tokens = None",
                "max_context_tokens = None",
                "compaction_mode = 'pi'",
                "compaction_keep_recent_tokens = 20000",
                "compaction_max_output_tokens = None",
                "extra_sample_params = {}",
            )),
        )
        rendered_json = config.render(json_output=True)
        defaults = {
            "enable_workspace": True,
            "max_samples": None,
            "max_output_tokens": None,
            "enable_auto_compaction": True,
            "auto_compact_tokens": None,
            "max_context_tokens": None,
            "compaction_mode": "pi",
            "compaction_keep_recent_tokens": 20000,
            "compaction_max_output_tokens": None,
            "extra_sample_params": {},
        }
        self.assertEqual(json.loads(rendered_json), {**defaults, "__init__": defaults})
        self.assertEqual(list(json.loads(rendered_json)), ["__init__", *CONFIG_KEYS])
        self.assertEqual(
            config.render("max_output_tokens", json_output=False),
            "max_output_tokens = None",
        )
        self.assertEqual(
            config.render("max_output_tokens", json_output=True),
            '{\n  "__init__": {\n    "max_output_tokens": null\n  },\n'
            '  "max_output_tokens": null\n}',
        )

    def test_initial_values_render_as_comments_and_json_defaults(self):
        config = InteractionConfig(
            InteractionConfigSnapshot(auto_compact_tokens=100),
            initial=InteractionConfigSnapshot(
                auto_compact_tokens=500_000,
            ),
        )
        self.assertEqual(
            config.render(json_output=False),
            "\n".join((
                "enable_workspace = True",
                "max_samples = None",
                "max_output_tokens = None",
                "enable_auto_compaction = True",
                "# init: auto_compact_tokens = 500000",
                "auto_compact_tokens = 100",
                "max_context_tokens = None",
                "compaction_mode = 'pi'",
                "compaction_keep_recent_tokens = 20000",
                "compaction_max_output_tokens = None",
                "extra_sample_params = {}",
            )),
        )
        payload = json.loads(config.render(json_output=True))
        self.assertEqual(payload["auto_compact_tokens"], 100)
        self.assertEqual(payload["__init__"]["auto_compact_tokens"], 500_000)

        config.set("auto_compact_tokens", 500_000)
        self.assertNotIn("# init:", config.render(json_output=False))

    def test_namespace_values_seed_process_state(self):
        args = cli._build_parser().parse_args([
            "--enable-workspace=False",
            "--enable-auto-compaction=False",
            "--max-samples", "3",
            "--max-output-tokens", "2048",
            "--auto-compact-tokens", "500000",
            "--max-context-tokens", "1000000",
            "--compaction-keep-recent-tokens", "0",
            "--compaction-max-output-tokens", "4096",
        ])

        config = InteractionConfig.from_namespace(args)

        self.assertEqual(config.values(), {
            "enable_workspace": False,
            "max_samples": 3,
            "max_output_tokens": 2048,
            "enable_auto_compaction": False,
            "auto_compact_tokens": 500000,
            "max_context_tokens": 1000000,
            "compaction_mode": "pi",
            "compaction_keep_recent_tokens": 0,
            "compaction_max_output_tokens": 4096,
            "extra_sample_params": {},
        })
        self.assertEqual(
            config.snapshot().compaction_settings(),
            CompactionSettings(mode="pi", keep_recent_tokens=0, max_output_tokens=4096),
        )
        self.assertEqual(
            config.snapshot().sample_params(),
            SampleParams(
                max_output_tokens=2048,
                enable_auto_compaction=False,
                auto_compact_tokens=500000,
            ),
        )

    def test_context_limit_keys_validate_and_stay_independent(self):
        for key in ("auto_compact_tokens", "max_context_tokens"):
            self.assertIsNone(parse_config_literal(key, "null"))
            self.assertEqual(parse_config_literal(key, "123"), 123)
            for bad in ("0", "-1", "1.5", "True", "x"):
                with self.subTest(key=key, bad=bad):
                    with self.assertRaises(ConfigError):
                        parse_config_literal(key, bad)

        config = InteractionConfig()
        self.assertEqual(config.snapshot().sample_params(), SampleParams(enable_auto_compaction=True))
        config.set("auto_compact_tokens", 500000)
        self.assertEqual(
            config.snapshot().sample_params(),
            SampleParams(auto_compact_tokens=500000, enable_auto_compaction=True),
        )
        # max_context_tokens is informational: no auto<=max gate.
        config.set("max_context_tokens", 1000)
        self.assertEqual(config.get("auto_compact_tokens"), 500000)
        self.assertEqual(config.get("max_context_tokens"), 1000)
        config.set("auto_compact_tokens", None)
        self.assertEqual(config.snapshot().sample_params(), SampleParams(enable_auto_compaction=True))

    def test_compaction_keys_validate_and_resolve_null(self):
        self.assertEqual(parse_config_literal("compaction_keep_recent_tokens", "0"), 0)
        self.assertIsNone(parse_config_literal("compaction_keep_recent_tokens", "null"))
        self.assertEqual(parse_config_literal("compaction_max_output_tokens", "77"), 77)
        self.assertIsNone(parse_config_literal("compaction_max_output_tokens", "None"))
        for key, bad in (
            ("compaction_keep_recent_tokens", "-1"),
            ("compaction_keep_recent_tokens", "True"),
            ("compaction_keep_recent_tokens", "1.5"),
            ("compaction_max_output_tokens", "0"),
            ("compaction_max_output_tokens", "False"),
        ):
            with self.subTest(key=key, bad=bad):
                with self.assertRaises(ConfigError) as raised:
                    parse_config_literal(key, bad)
                self.assertNotIn(bad, str(raised.exception))

        config = InteractionConfig()
        self.assertEqual(config.set("compaction_keep_recent_tokens", 0), 0)
        self.assertEqual(config.snapshot().compaction_settings().keep_recent_tokens, 0)
        # null restores the default, rather than storing None.
        self.assertEqual(config.set("compaction_keep_recent_tokens", None), 20_000)
        self.assertEqual(config.get("compaction_keep_recent_tokens"), 20_000)
        self.assertEqual(config.set("compaction_max_output_tokens", 99), 99)
        self.assertEqual(config.snapshot().compaction_settings().max_output_tokens, 99)
        self.assertIsNone(config.set("compaction_max_output_tokens", None))
        # Summary budgets never change the sampling projection.
        self.assertEqual(config.snapshot().sample_params(), SampleParams(enable_auto_compaction=True))
        with self.assertRaises(ConfigError):
            config.set("compaction_keep_recent_tokens", -1)
        with self.assertRaises(ConfigError):
            InteractionConfigSnapshot(compaction_max_output_tokens=0)

    def test_compaction_mode_is_launch_only(self):
        config = InteractionConfig()
        for mode in ("pi", "provider"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ConfigError, "compaction_mode is launch-only"):
                    parse_config_literal("compaction_mode", mode)
                with self.assertRaisesRegex(ConfigError, "compaction_mode is launch-only"):
                    config.set("compaction_mode", mode)
        self.assertEqual(config.get("compaction_mode"), "pi")
        self.assertEqual(config.render("compaction_mode", json_output=False),
                         "compaction_mode = 'pi'")
        with self.assertRaisesRegex(ConfigError, "'pi' or 'provider'"):
            InteractionConfigSnapshot(compaction_mode="remote")

    def test_compaction_mode_resolves_per_route(self):
        catalog = BUILTIN_MODEL_CATALOG
        official = catalog.bind("codex", "codex-gpt-6-astra")
        meta = catalog.bind("codex", "muse-spark-1.3")
        custom = catalog.bind(
            "codex", "codex-gpt-6-astra",
            endpoint_url="https://example.test/v1/responses",
            endpoint_auth="none",
        )
        claude = catalog.bind("messages", "claude-opus-5.5")
        local = catalog.bind("chat-completions", "local")
        self.assertTrue(official.supports_remote_compaction)
        for binding, default in (
            (official, "provider"), (meta, "pi"), (custom, "pi"),
            (claude, "pi"), (local, "pi"),
        ):
            with self.subTest(binding=binding.selector, url=binding.endpoint.url):
                self.assertEqual(resolve_compaction_mode(binding), default)
                self.assertEqual(resolve_compaction_mode(binding, "pi"), "pi")
        for binding in (official, claude):
            self.assertEqual(resolve_compaction_mode(binding, "provider"), "provider")
        for binding in (meta, custom, local):
            with self.subTest(binding=binding.selector, url=binding.endpoint.url):
                with self.assertRaisesRegex(ConfigError, "no provider compaction"):
                    resolve_compaction_mode(binding, "provider")
        with self.assertRaises(ConfigError):
            resolve_compaction_mode(official, "remote")

    def test_namespace_resolves_the_mode_and_applies_the_minimum_only_to_provider(self):
        for argv, mode in (
            (["--endpoint-api", "codex", "--model", "codex-gpt-6-astra"], "provider"),
            (["--endpoint-api", "codex", "--model", "codex-gpt-6-astra",
              "--compaction-mode", "pi"], "pi"),
            (["--endpoint-api", "messages", "--model", "claude-fable-5.1"], "pi"),
            (["--endpoint-api", "messages", "--model", "claude-fable-5.1",
              "--compaction-mode", "provider"], "provider"),
        ):
            with self.subTest(argv=argv):
                config = InteractionConfig.from_namespace(cli._build_parser().parse_args(argv))
                self.assertEqual(config.get("compaction_mode"), mode)
                self.assertEqual(config.snapshot().compaction_settings().mode, mode)
        pi = InteractionConfig.from_namespace(cli._build_parser().parse_args([
            "--endpoint-api", "messages", "--model", "claude-fable-5.1",
            "--auto-compact-tokens", "20000",
        ]))
        self.assertEqual(pi.get("auto_compact_tokens"), 20_000)
        self.assertEqual(pi.set("auto_compact_tokens", 1_000), 1_000)
        with self.assertRaisesRegex(ConfigError, "at least 50000"):
            InteractionConfig.from_namespace(cli._build_parser().parse_args([
                "--endpoint-api", "messages", "--model", "claude-fable-5.1",
                "--compaction-mode", "provider", "--auto-compact-tokens", "20000",
            ]))
        with self.assertRaisesRegex(ConfigError, "no provider compaction"):
            InteractionConfig.from_namespace(cli._build_parser().parse_args([
                "--endpoint-api", "chat-completions", "--compaction-mode", "provider",
            ]))

    def test_messages_uses_catalog_fallback_and_explicit_precedence(self):
        catalogued = cli._build_parser().parse_args([
            "--endpoint-api", "messages",
            "--model", "claude-fable-5.1",
        ])
        config = InteractionConfig.from_namespace(catalogued)
        self.assertEqual(config.get("max_output_tokens"), 128_000)

        explicit = cli._build_parser().parse_args([
            "--endpoint-api", "messages",
            "--model", "claude-fable-5.1",
            "--max-output-tokens", "100",
        ])
        config = InteractionConfig.from_namespace(explicit)
        self.assertEqual(config.get("max_output_tokens"), 100)
        config.set("max_output_tokens", None)
        self.assertEqual(config.get("max_output_tokens"), 128_000)

    def test_uncatalogued_messages_requires_explicit_output_limit(self):
        missing = cli._build_parser().parse_args([
            "--endpoint-api", "messages",
            "--model", "model",
        ])
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "model catalog",
        ):
            InteractionConfig.from_namespace(missing)

        explicit = cli._build_parser().parse_args([
            "--endpoint-api", "messages",
            "--model", "model",
            "--max-output-tokens", "100",
        ])
        config = InteractionConfig.from_namespace(explicit)
        self.assertEqual(config.get("max_output_tokens"), 100)
        with self.assertRaisesRegex(ConfigError, "required for this Messages"):
            config.set("max_output_tokens", None)

    def test_json_and_python_literals_are_typed_per_key(self):
        for literal, expected in (
            ("true", True),
            ("True", True),
            ("false", False),
            ("False", False),
        ):
            with self.subTest(literal=literal):
                self.assertIs(
                    parse_config_literal("enable_workspace", literal),
                    expected,
                )
        for literal in ("null", "None"):
            with self.subTest(literal=literal):
                self.assertIsNone(parse_config_literal("max_output_tokens", literal))
        self.assertEqual(parse_config_literal("max_samples", "+12"), 12)

        for key, literal in (
            ("enable_workspace", "None"),
            ("enable_workspace", "1"),
            ("max_output_tokens", "True"),
            ("max_output_tokens", "0"),
            ("max_output_tokens", "1.5"),
            ("max_samples", "-1"),
        ):
            with self.subTest(key=key, literal=literal):
                with self.assertRaises(ConfigError) as raised:
                    parse_config_literal(key, literal)
                self.assertNotIn(literal, str(raised.exception))

    def test_set_validates_before_callback_and_is_atomic_on_callback_failure(self):
        updates = []
        config = InteractionConfig(on_enable_workspace=updates.append)

        self.assertIs(config.set("enable_workspace", False), False)
        self.assertEqual(updates, [False])
        self.assertIs(config.get("enable_workspace"), False)
        config.set("max_output_tokens", 100)
        self.assertEqual(config.get("max_output_tokens"), 100)
        config.set("max_output_tokens", None)
        self.assertIsNone(config.get("max_output_tokens"))

        failed = InteractionConfig(
            on_enable_workspace=lambda value: (_ for _ in ()).throw(
                RuntimeError("apply failed")
            )
        )
        with self.assertRaisesRegex(RuntimeError, "apply failed"):
            failed.set("enable_workspace", False)
        self.assertIs(failed.get("enable_workspace"), True)

    def test_sample_params_preserve_startup_defaults_and_runtime_override(self):
        config = InteractionConfig()
        self.assertEqual(config.snapshot().sample_params(), SampleParams(enable_auto_compaction=True))

        config.set("max_output_tokens", 99)
        self.assertEqual(
            config.snapshot().sample_params(),
            SampleParams(max_output_tokens=99, enable_auto_compaction=True),
        )
        config.set("max_output_tokens", None)
        self.assertEqual(config.snapshot().sample_params(), SampleParams(enable_auto_compaction=True))

        config.set("enable_auto_compaction", False)
        self.assertEqual(
            config.snapshot().sample_params(),
            SampleParams(enable_auto_compaction=False),
        )
        config.set("enable_auto_compaction", True)
        self.assertEqual(
            config.snapshot().sample_params(),
            SampleParams(enable_auto_compaction=True),
        )


if __name__ == "__main__":
    unittest.main()
