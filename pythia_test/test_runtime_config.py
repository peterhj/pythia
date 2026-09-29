from __future__ import annotations

import json
import unittest

from pythia.interaction import CONFIG_KEYS
from pythia.interaction import ConfigError
from pythia.interaction import InteractionConfig
from pythia.interaction import InteractionConfigSnapshot
from pythia.interaction import ModelConfigurationError
from pythia.interaction import SamplingParams
from pythia.interaction import ResolvedSamplingParams
from pythia.interaction import cli
from pythia.interaction.runtime_config import parse_config_literal


class InteractionConfigTests(unittest.TestCase):
    def test_output_limit_uses_the_explicit_shared_name(self):
        self.assertIn("max_output_tokens", CONFIG_KEYS)
        self.assertNotIn("max_tokens", CONFIG_KEYS)
        self.assertTrue(hasattr(SamplingParams(), "max_output_tokens"))
        self.assertFalse(hasattr(SamplingParams(), "max_tokens"))
        self.assertTrue(
            hasattr(InteractionConfigSnapshot(), "max_output_tokens")
        )
        self.assertFalse(hasattr(InteractionConfigSnapshot(), "max_tokens"))

    def test_defaults_and_python_and_json_dumps_are_stable(self):
        config = InteractionConfig()

        self.assertEqual(config.snapshot(), InteractionConfigSnapshot())
        self.assertEqual(
            config.render(json_output=False),
            "\n".join((
                "enable_workspace = True",
                "max_samples = None",
                "max_output_tokens = None",
                "enable_auto_compaction = True",
                "auto_compact_tokens = None",
                "max_context_tokens = None",
                "request_params = {}",
            )),
        )
        rendered_json = config.render(json_output=True)
        self.assertEqual(json.loads(rendered_json), {
            "enable_workspace": True,
            "max_samples": None,
            "max_output_tokens": None,
            "enable_auto_compaction": True,
            "auto_compact_tokens": None,
            "max_context_tokens": None,
            "request_params": {},
            "__init__": {
                "enable_workspace": True,
                "max_samples": None,
                "max_output_tokens": None,
                "enable_auto_compaction": True,
                "auto_compact_tokens": None,
                "max_context_tokens": None,
                "request_params": {},
            },
        })
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
                "request_params = {}",
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
        ])

        config = InteractionConfig.from_namespace(args)

        self.assertEqual(config.values(), {
            "enable_workspace": False,
            "max_samples": 3,
            "max_output_tokens": 2048,
            "enable_auto_compaction": False,
            "auto_compact_tokens": 500000,
            "max_context_tokens": 1000000,
            "request_params": {},
        })
        self.assertEqual(
            config.snapshot().sampling_params(),
            ResolvedSamplingParams(
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
        self.assertEqual(config.snapshot().sampling_params(), ResolvedSamplingParams())
        config.set("auto_compact_tokens", 500000)
        self.assertEqual(
            config.snapshot().sampling_params(),
            ResolvedSamplingParams(auto_compact_tokens=500000),
        )
        # max_context_tokens is informational: no auto<=max gate.
        config.set("max_context_tokens", 1000)
        self.assertEqual(config.get("auto_compact_tokens"), 500000)
        self.assertEqual(config.get("max_context_tokens"), 1000)
        config.set("auto_compact_tokens", None)
        self.assertEqual(config.snapshot().sampling_params(), ResolvedSamplingParams())

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

    def test_sampling_options_preserve_startup_defaults_and_runtime_override(self):
        config = InteractionConfig()
        self.assertEqual(config.snapshot().sampling_params(), ResolvedSamplingParams())

        config.set("max_output_tokens", 99)
        self.assertEqual(
            config.snapshot().sampling_params(),
            ResolvedSamplingParams(max_output_tokens=99),
        )
        config.set("max_output_tokens", None)
        self.assertEqual(config.snapshot().sampling_params(), ResolvedSamplingParams())

        config.set("enable_auto_compaction", False)
        self.assertEqual(
            config.snapshot().sampling_params(),
            ResolvedSamplingParams(enable_auto_compaction=False),
        )
        config.set("enable_auto_compaction", True)
        self.assertEqual(
            config.snapshot().sampling_params(),
            ResolvedSamplingParams(enable_auto_compaction=True),
        )


if __name__ == "__main__":
    unittest.main()
