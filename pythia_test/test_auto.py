from __future__ import annotations

import asyncio
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.request import urlopen

from pythia.interaction import DisplayItem, Environment, Message, ModelSample, ModelSampleBoundary, Tool, ToolCall
from pythia.interaction import ToolOutcome, ToolSpec, ToolResult, TurnSummary, Tools
from pythia.interaction import ModelFailure, ModelTransportError, Reasoning, OpaqueCompaction
from pythia.interaction import CompactionContextWindowError, CompactionResult, CompactionSettings
from pythia.interaction import ContextPrefix, ModelContextWindowError, NothingToCompact
from pythia.interaction import load_interaction_save
from pythia.interaction import auto
from pythia.interaction import SampleParams
from pythia.interaction._auto_board import BoardError
from pythia.interaction._auto_config import FOLLOWS_MAIN, build_parser, namespace, resolve_config
from pythia.interaction.messages import resolve_messages_max_output_tokens
from pythia.interaction.runtime_config import InteractionConfig


def answer(text="done"):
    return ModelSample((Message("assistant", text),))


def resume(content, call_id="resume"):
    """A watcher sample that resumes main with content."""
    return ModelSample((ToolCall("resume_main", call_id, json.dumps({"content": content})),))


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("timed out waiting for test condition")


class DisplayTests(unittest.TestCase):
    def setUp(self):
        self.session = SimpleNamespace(names={1: "main", 2: "custom worker", -1: "watcher"})

    def test_context_is_folded_into_every_item_label(self):
        sample = ModelSample((
            Reasoning("", summary=("first thought", "second thought")),
            Message("assistant", "answer\n[reasoning] quoted body stays unchanged"),
        ))
        original = sample.display_items()
        rendered = auto._display_events(self.session, (
            auto._Event(1, original),
            auto._Event(2, answer("worker answer").display_items()),
            auto._Event(-1, (DisplayItem("[debug] condition fired", label="debug"),), "debug"),
        ))
        self.assertEqual([item.text for item in rendered], [
            "[#1 (main) - reasoning] first thought",
            "[#1 (main) - reasoning] second thought",
            "[#1 (main) - assistant] answer\n[reasoning] quoted body stays unchanged",
            "[#1 (main) - sample] input=0 output=0 total=0 cached=0",
            "[#2 (custom worker) - assistant] worker answer",
            "[#2 (custom worker) - sample] input=0 output=0 total=0 cached=0",
            "[#-1 (watcher) - debug] condition fired",
        ])
        self.assertEqual(original, sample.display_items())
        self.assertEqual(original[0].text, "[reasoning] first thought")

    def test_global_events_and_empty_batches_have_no_context_heading(self):
        global_items = (DisplayItem("Board: http://localhost/README.md"),
                        DisplayItem("[assistant] global notice", label="assistant"),
                        DisplayItem("-old\n+new", is_diff=True))
        rendered = auto._display_events(self.session, (
            auto._Event(1, ()),
            auto._Event(None, global_items),
            auto._Event(-1, ()),
        ))
        self.assertEqual(rendered, list(global_items))
        self.assertTrue(all(a is b for a, b in zip(rendered, global_items)))

    def test_context_listing_is_one_multiline_item_for_each_selection(self):
        statuses = {
            1: "#1 (main custom) - quiescent",
            2: "#2 (worker custom) - sampling (1.2s)",
            -1: "#-1 (watcher custom) - waiting",
        }
        session = SimpleNamespace(status=lambda index: statuses[index], roles=(1, 2, -1))
        for selected in (1, 2, -1):
            with self.subTest(selected=selected):
                actual_selected, items, stop = auto._local_command("/contexts", selected, session)
                self.assertEqual(actual_selected, selected)
                self.assertFalse(stop)
                self.assertEqual(len(items), 1)
                expected = [
                    ("* " if index == selected else "  ") + statuses[index]
                    for index in (1, 2, -1)
                ]
                self.assertEqual(items[0].text.splitlines(), expected)
                self.assertEqual(len(items[0].text.split("\n")), 3)
                self.assertFalse(items[0].text.endswith("\n"))

    def test_unlabeled_context_notice_stays_in_one_item(self):
        notice = DisplayItem("Task failed; details withheld.")
        rendered = auto._display_events(self.session, (auto._Event(1, (notice,), "error"),))
        self.assertEqual(rendered, [DisplayItem("[#1 (main)]\nTask failed; details withheld.")])

    def test_arbitrary_message_role_is_not_parsed_as_bracket_syntax(self):
        original = auto.render_interaction_items((Message("custom] role", "body"),))
        rendered = auto._display_events(self.session, (auto._Event(1, original),))
        self.assertEqual(rendered[0].text, "[#1 (main) - custom] role] body")
        self.assertEqual(rendered[0].label, "#1 (main) - custom] role")

    def test_tool_payloads_keep_their_body_and_diff_colors(self):
        patch = "--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new"
        for name, payload in (("apply_patch", patch),
                              ("write_file", "[reasoning]\nthis is file content"),
                              ("write_file", '["a", "b"]')):
            with self.subTest(name=name, payload=payload):
                call = ToolCall(name, "edit", json.dumps({"content": payload}))
                original = auto.render_interaction_items((call,))
                rendered = auto._display_events(self.session, (auto._Event(2, original),))
                self.assertEqual(len(rendered), 2)
                self.assertEqual(rendered[0].text, f"[#2 (custom worker) - tool-call] {name} (edit)")
                self.assertEqual(rendered[1].text, f"[#2 (custom worker)]\n{payload}")
                self.assertEqual([item.is_diff for item in rendered],
                                 [item.is_diff for item in original])
                self.assertEqual(original[1].text, payload)
                if name == "apply_patch":
                    printed = str(rendered[1])
                    self.assertIn("\x1b[31m-old\x1b[0m", printed)
                    self.assertIn("\x1b[32m+new\x1b[0m", printed)
                    self.assertNotIn("\x1b[31m---", printed)
                    self.assertNotIn("\x1b[32m+++", printed)

    def test_tool_result_label_is_folded_without_rewriting_body_labels(self):
        call = ToolCall("exec_command", "diff", '{"cmd":"git diff"}')
        original = auto.render_interaction_items((
            ToolResult("diff", "[assistant] literal output\n@@ -1 +1 @@\n-old\n+new"),
        ), source_calls=(call,))
        rendered = auto._display_events(self.session, (auto._Event(1, original),))
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0].text,
                         "[#1 (main) - tool-ret]  exec_command (diff) [ok]\n"
                         "[assistant] literal output\n@@ -1 +1 @@\n-old\n+new")
        self.assertTrue(rendered[0].is_diff)
        self.assertIn("\x1b[31m-old\x1b[0m", str(rendered[0]))
        self.assertIn("\x1b[32m+new\x1b[0m", str(rendered[0]))


class ConfigTests(unittest.TestCase):
    def test_board_auth_boolean_argument_is_frontend_only(self):
        parser = build_parser()
        cases = (((), True), (("--enable-board-auth",), True),
                 (("--enable-board-auth", "TRUE"), True),
                 (("--enable-board-auth", "false"), False),
                 (("--enable-board-auth", "FaLsE"), False))
        for argv, expected in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertIs(args.enable_board_auth, expected)
                settings = resolve_config(overrides={
                    key: value for key, value in vars(args).items() if key in auto.DEFAULTS
                })
                self.assertTrue(all("enable_board_auth" not in value
                                    for value in settings.values()))
        for invalid in ("yes", "0", "enabled"):
            with self.subTest(invalid=invalid), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--enable-board-auth", invalid])

    def test_worker_board_flag_is_frontend_only_and_off_by_default(self):
        parser = build_parser()
        flag = "--enable-experimental-worker-board"
        for argv, expected in (((), False), ((flag,), True), ((flag, "TRUE"), True),
                               ((flag, "false"), False)):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertIs(args.enable_experimental_worker_board, expected)
                settings = resolve_config(overrides={
                    key: value for key, value in vars(args).items() if key in auto.DEFAULTS
                })
                self.assertEqual(set(settings), {1, 2, -1})
                self.assertTrue(all("enable_experimental_worker_board" not in value
                                    for value in settings.values()))
        for invalid in ("yes", "1"):
            with self.subTest(invalid=invalid), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([flag, invalid])

    def test_saved_config_statically_validates_and_normalizes_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("relative-cwd", "relative-home", "auth-parent"):
                (root / directory).mkdir()
            snapshot = auto.saved_document(resolve_config(overrides={"cwd": tmp}))
            snapshot["main"].update({
                "model_api": "codex", "model": "saved-codex",
                "cwd": "relative-cwd", "codex_home": "relative-home",
            })
            snapshot["worker"].update({
                "model_api": "codex", "model": "saved-codex-worker",
                "cwd": "relative-cwd", "codex_auth_file": "auth-parent/auth.json",
            })
            path = root / "config.json"

            def write():
                path.write_text(json.dumps(snapshot))

            write()
            saved = auto.load_saved_config(path)
            self.assertEqual(saved[1]["cwd"], str((root / "relative-cwd").absolute()))
            self.assertEqual(saved[1]["codex_home"], str((root / "relative-home").absolute()))
            self.assertEqual(saved[2]["codex_auth_file"],
                             str((root / "auth-parent/auth.json").absolute()))
            merged = resolve_config(saved=saved)
            self.assertEqual(merged[1]["codex_home"], saved[1]["codex_home"])
            self.assertEqual(merged[2]["codex_auth_file"], saved[2]["codex_auth_file"])

            for key in ("cwd", "codex_home", "codex_auth_file"):
                original = snapshot["main"][key]
                for invalid in (123, "", "bad\x00path"):
                    with self.subTest(key=key, invalid=invalid):
                        snapshot["main"][key] = invalid
                        write()
                        with self.assertRaises(ValueError):
                            auto.load_saved_config(path)
                snapshot["main"][key] = original

    def test_saved_config_is_version_3_and_older_saves_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            current = resolve_config(overrides={"cwd": tmp, "model": "saved-model"})
            document = auto.saved_document(current)
            self.assertEqual(set(document), {"version", "main", "worker", "watcher"})
            self.assertEqual(document["version"], 3)
            self.assertEqual(set(document["main"]), set(auto.DEFAULTS) | {"name"})
            path.write_text(json.dumps(document))
            self.assertEqual(resolve_config(saved=auto.load_saved_config(path)), current)
            for version in (1, 2):
                with self.subTest(version=version):
                    path.write_text(json.dumps({"version": version, "contexts": {}}))
                    with self.assertRaisesRegex(ValueError, "earlier version of auto"):
                        auto.load_saved_config(path)
            incomplete = {key: value for key, value in document["main"].items() if key != "cwd"}
            for broken in ({**document, "version": 4}, {**document, "main": incomplete},
                           {**document, "watcher": {"model": "no identity"}},
                           {**document, "observer": {"name": "x", "instructions": None}}):
                with self.subTest(broken=broken):
                    path.write_text(json.dumps(broken))
                    with self.assertRaisesRegex(ValueError, "Invalid saved"):
                        auto.load_saved_config(path)

    def test_saved_settings_are_base_for_explicit_resume_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Created with --model: every role chose saved-model itself.
            document = auto.saved_document(resolve_config(
                overrides={"model": "saved-model", "cwd": str(Path.cwd())}))
            document["worker"].update(name="saved worker",
                                      instructions="saved custom worker instructions")
            saved_path = Path(tmp) / "config.json"
            saved_path.write_text(json.dumps(document))
            saved = auto.load_saved_config(saved_path)
            path = Path(tmp) / "override.json"
            path.write_text(json.dumps({
                "version": 3,
                "main": {"max_samples": 7},
                "worker": {"model": "role-model", "name": "new worker"},
            }))
            settings = resolve_config(path, saved=saved, role_models={1: "launch-model"})
            self.assertEqual(settings[1]["model"], "launch-model")
            self.assertEqual(settings[2]["model"], "role-model")
            # --main-model is main's only: the watcher keeps the model it chose.
            self.assertEqual(settings[-1]["model"], "saved-model")
            self.assertEqual(settings.sources, {1: "command line", 2: "config file", -1: "saved"})
            self.assertTrue(all(value["max_samples"] == 7 for value in settings.values()))
            self.assertEqual(settings[2]["name"], "new worker")
            self.assertEqual(settings[2]["instructions"], "saved custom worker instructions")
            # --model sets every role's model, over the file and the save.
            settings = resolve_config(path, {"model": "launch-model"}, saved=saved)
            self.assertEqual({value["model"] for value in settings.values()}, {"launch-model"})

    def test_headless_boolean_argument_is_frontend_only(self):
        parser = build_parser()
        for argv, expected in (((), False), (("--headless",), True),
                               (("--headless", "True"), True),
                               (("--headless", "tRuE"), True),
                               (("--headless", "False"), False),
                               (("--headless", "fAlSe"), False)):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertIs(args.headless, expected)
                settings = resolve_config(overrides={
                    key: value for key, value in vars(args).items() if key in auto.DEFAULTS
                })
                self.assertTrue(all("headless" not in value for value in settings.values()))
        for value in ("yes", "1", "", "none"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--headless", value])

    def test_default_limits_are_unset(self):
        args = build_parser().parse_args([])
        self.assertFalse(hasattr(args, "max_samples"))
        self.assertFalse(hasattr(args, "max_output_tokens"))
        for settings in resolve_config().values():
            self.assertIsNone(settings["max_samples"])
            self.assertIsNone(settings["max_output_tokens"])
            snapshot = InteractionConfig.from_namespace(namespace(settings)).snapshot()
            self.assertIsNone(snapshot.max_samples)
            self.assertEqual(snapshot.sample_params(), SampleParams(enable_auto_compaction=True))

    def test_explicit_limits_and_per_role_null_overrides(self):
        args = build_parser().parse_args(["--max-samples", "3", "--max-output-tokens", "128"])
        overrides = {"max_samples": args.max_samples, "max_output_tokens": args.max_output_tokens}
        for settings in resolve_config(overrides=overrides).values():
            self.assertEqual(settings["max_samples"], 3)
            self.assertEqual(settings["max_output_tokens"], 128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({
                "version": 3,
                "main": {"max_samples": 5, "max_output_tokens": 512},
                "worker": {"max_samples": None, "max_output_tokens": None},
            }))
            settings = resolve_config(path)
            limits = {i: (s["max_samples"], s["max_output_tokens"]) for i, s in settings.items()}
            # Roles inherit main's settings; a role's null clears them for that role.
            self.assertEqual(limits, {1: (5, 512), -1: (5, 512), 2: (None, None)})
            # The command line wins over the file.
            settings = resolve_config(path, overrides)
            limits = {i: (s["max_samples"], s["max_output_tokens"]) for i, s in settings.items()}
            self.assertEqual(limits, {1: (3, 128), -1: (3, 128), 2: (3, 128)})
        for key in ("max_samples", "max_output_tokens"):
            for value in (0, -1, True, False, 1.5, "3"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resolve_config(overrides={key: value})

    def test_unset_messages_budget_uses_catalog_or_requires_explicit_limit(self):
        model = "claude-fable-5.1"
        settings = resolve_config(overrides={"model_api": "messages", "model": model})[2]
        self.assertIsNone(settings["max_output_tokens"])
        args = namespace(settings)
        snapshot = InteractionConfig.from_namespace(args).snapshot()
        self.assertEqual(
            snapshot.max_output_tokens,
            resolve_messages_max_output_tokens(args.model_binding, None),
        )
        with self.assertRaisesRegex(ValueError, "provide it explicitly"):
            resolve_config(overrides={"model_api": "messages", "model": "uncatalogued-auto-test-model"})

    def test_command_line_wins_and_role_routes_reset_main_provider_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({
                "version": 3,
                "main": {"model_api": "codex", "model": "file-main", "codex_auth_file": "auth.json"},
                "worker": {"model_api": "messages", "model": "worker",
                           "endpoint_auth": "env:WORKER_KEY", "max_output_tokens": 64},
            }))
            parsed = build_parser().parse_args(["--context-config", str(path), "--prompt", "hi"])
            self.assertFalse(hasattr(parsed, "model_api"))
            settings = resolve_config(path, role_models={1: "launch-main"})
            auth = str(Path(tmp) / "auth.json")
            self.assertEqual((settings[1]["model_api"], settings[1]["model"]), ("codex", "launch-main"))
            self.assertEqual(settings[1]["codex_auth_file"], auth)
            # The watcher follows main; the worker keeps its own route, without
            # main's Codex credential file.
            self.assertEqual((settings[-1]["model"], settings[-1]["codex_auth_file"]), ("launch-main", auth))
            self.assertEqual((settings[2]["model_api"], settings[2]["model"]), ("messages", "worker"))
            self.assertIsNone(settings[2]["codex_auth_file"])
            self.assertEqual(settings[2]["endpoint_auth"], "env:WORKER_KEY")
            self.assertEqual(settings[2]["max_output_tokens"], 64)
            self.assertEqual([settings[i]["name"] for i in (1, 2, -1)], ["main", "worker", "watcher"])
            self.assertEqual(settings.sources,
                             {1: "command line", 2: "config file", -1: FOLLOWS_MAIN})
            # --model is every role's model: the worker's file route gives way whole.
            settings = resolve_config(path, {"model": "launch-all"})
            self.assertEqual({value["model"] for value in settings.values()}, {"launch-all"})
            self.assertEqual(settings[2]["model_api"], "codex")
            self.assertIsNone(settings[2]["endpoint_auth"])

    def test_config_validation_and_instruction_scope(self):
        settings = resolve_config(overrides={"instructions": ""})
        self.assertEqual(settings[1]["instructions"], "")
        self.assertIsNone(settings[2]["instructions"])
        self.assertIsNone(settings[-1]["instructions"])
        bad = ({"max_samples": 0}, {"request_timeout_seconds": float("nan")},
               {"endpoint_url": "http://user:secret@localhost"}, {"api_key": "SECRET"},
               {"model_api": "other"}, {"max_output_tokens": True},
               {"cwd": None}, {"endpoint_url": True},
               {"endpoint_url": "http://localhost:bad"})
        for overrides in bad:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_config(overrides=overrides)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            for content in ('{"version":3,"observer":{}}',
                            '{"version":3,"version":3}',
                            '{"version":3,"watcher":{"name":"bad\\nname"}}',
                            '{"version":3,"main":{"api_key":"SECRET"}}',
                            '{"version":1,"contexts":{}}'):
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        resolve_config(path)
            path.write_text('{"version":1,"contexts":{}}')
            with self.assertRaisesRegex(ValueError, "version 3"):
                resolve_config(path)
            # Instructions are per role; the command line's win over the file's.
            path.write_text(json.dumps({"version": 3, "main": {"instructions": "file main"},
                                        "watcher": {"instructions": "file watcher"}}))
            self.assertEqual(resolve_config(path)[1]["instructions"], "file main")
            settings = resolve_config(path, {"instructions": "launch main"})
            self.assertEqual(settings[1]["instructions"], "launch main")
            self.assertEqual(settings[-1]["instructions"], "file watcher")
            self.assertIsNone(settings[2]["instructions"])


class RoleModelTests(unittest.TestCase):
    """Every role uses main's model unless it chooses its own."""

    def resolve(self, overrides=None, **kwargs):
        return resolve_config(overrides={"cwd": str(Path.cwd()), **(overrides or {})},
                              roles=(1, -1), **kwargs)

    def route(self, settings, index):
        binding = namespace(settings[index], settings.catalog).model_binding
        return binding.api, binding.endpoint.model, binding.endpoint.url, binding.endpoint.auth

    def test_model_options(self):
        codex = "https://chatgpt.com/backend-api/codex/responses"
        local = "http://127.0.0.1:8000/v1/chat/completions"
        gpu = "http://gpu:8000/v1/chat/completions"
        for overrides, role_models, main, watcher, sources in (
            ({}, {}, ("chat-completions", None, local, "none"),
             ("chat-completions", None, local, "none"), ("built-in default", FOLLOWS_MAIN)),
            ({"model": "codex-gpt-6-astra-max"}, {},
             ("codex", "gpt-6-astra", codex, "codex-login"),
             ("codex", "gpt-6-astra", codex, "codex-login"), ("command line", "command line")),
            ({"model": "codex-gpt-6-astra-max"}, {-1: "codex-gpt-6-luna"},
             ("codex", "gpt-6-astra", codex, "codex-login"),
             ("codex", "gpt-6-luna", codex, "codex-login"), ("command line", "command line")),
            ({"model": "claude-opus-5.5"}, {-1: "codex-gpt-6-luna"},
             ("messages", "claude-opus-5-5", "https://api.anthropic.com/v1/messages",
              "env:ANTHROPIC_API_KEY"),
             ("codex", "gpt-6-luna", codex, "codex-login"), ("command line", "command line")),
            # Another name is a model ID on main's endpoint.
            ({"endpoint_url": gpu, "model": "big"}, {-1: "small"},
             ("chat-completions", "big", gpu, "none"),
             ("chat-completions", "small", gpu, "none"), ("command line", "command line")),
            # --main-model is main's only, with main's endpoint options.
            ({"endpoint_url": gpu}, {1: "big"}, ("chat-completions", "big", gpu, "none"),
             ("chat-completions", "big", gpu, "none"), ("command line", FOLLOWS_MAIN)),
        ):
            with self.subTest(overrides=overrides, role_models=role_models):
                settings = self.resolve(overrides, role_models=role_models)
                self.assertEqual(self.route(settings, 1), main)
                self.assertEqual(self.route(settings, -1), watcher)
                self.assertEqual((settings.sources[1], settings.sources[-1]), sources)

    def test_catalog_defaults_are_independent_per_role(self):
        defaults = {1: "codex-gpt-6-astra-max", -1: "codex-gpt-6-luna"}
        for overrides, role_models, models, sources in (
            ({}, {}, ("codex-gpt-6-astra-max", "codex-gpt-6-luna"),
             ("catalog default", "catalog default")),
            ({"model": "codex-gpt-6-sol"}, {}, ("codex-gpt-6-sol", "codex-gpt-6-sol"),
             ("command line", "command line")),
            ({}, {1: "codex-gpt-6-sol"}, ("codex-gpt-6-sol", "codex-gpt-6-luna"),
             ("command line", "catalog default")),
            ({}, {-1: "codex-gpt-6-sol"}, ("codex-gpt-6-astra-max", "codex-gpt-6-sol"),
             ("catalog default", "command line")),
        ):
            with self.subTest(overrides=overrides, role_models=role_models):
                settings = self.resolve(overrides, role_models=role_models, role_defaults=defaults)
                self.assertEqual((settings[1]["model"], settings[-1]["model"]), models)
                self.assertEqual((settings.sources[1], settings.sources[-1]), sources)
        # A role without a default follows main.
        settings = self.resolve({}, role_models={1: "codex-gpt-6-sol"},
                                role_defaults={1: "codex-gpt-6-astra-max"})
        self.assertEqual(settings[-1]["model"], "codex-gpt-6-sol")
        # An explicit endpoint means that server, not the default model on it.
        settings = self.resolve({"endpoint_url": "http://gpu:8000/v1/chat/completions"},
                                role_defaults={1: "codex-gpt-6-astra-max"})
        self.assertIsNone(settings[1]["model"])
        self.assertEqual(self.route(settings, 1)[0], "chat-completions")
        # Connection-only options adjust the default model's connection.
        settings = self.resolve({"codex_auth_file": "/x/auth.json"},
                                role_defaults={1: "codex-gpt-6-astra-max"})
        self.assertEqual((settings[1]["model"], settings[1]["codex_auth_file"]),
                         ("codex-gpt-6-astra-max", "/x/auth.json"))
        self.assertEqual(settings.sources[1], "catalog default")

    def test_a_role_on_another_model_takes_nothing_model_specific(self):
        explicit = {"model_api": "codex", "model": "gpt-6-astra-max",
                    "codex_auth_file": "/x/auth.json"}
        # A catalog name selects its own route even when main's API is explicit.
        settings = self.resolve(explicit, role_models={-1: "claude-sonnet-5.5"})
        self.assertEqual(self.route(settings, -1)[0], "messages")
        self.assertIsNone(settings[-1]["codex_auth_file"])
        # Main's credentials carry only to the same connection.
        settings = self.resolve(explicit, role_models={-1: "codex-gpt-6-luna"})
        self.assertEqual(self.route(settings, -1)[:2], ("codex", "gpt-6-luna"))
        self.assertEqual(settings[-1]["codex_auth_file"], "/x/auth.json")
        # Model-specific options stay with main's model.
        limits = {"model": "claude-opus-5.5", "max_output_tokens": 64000,
                  "auto_compact_tokens": 100000, "extra_sample_params": {"custom": 1}}
        settings = self.resolve(limits, role_models={-1: "codex-gpt-6-luna"})
        self.assertEqual(settings[1]["max_output_tokens"], 64000)
        for key in ("max_output_tokens", "auto_compact_tokens", "extra_sample_params"):
            self.assertIsNone(settings[-1][key], key)
        self.assertEqual(self.resolve(limits)[-1]["max_output_tokens"], 64000)
        # Shared settings still apply to every role.
        settings = self.resolve({"max_samples": 3}, role_models={-1: "codex-gpt-6-luna"})
        self.assertEqual(settings[-1]["max_samples"], 3)

    def test_ambiguous_names_and_unrunnable_role_models_fail(self):
        from pythia.interaction.model_catalog_config import parse_model_catalog
        registry = parse_model_catalog(
            "[catalog]\nversion = 4\n\n[model.codex-gpt-6-luna]\nendpoint.api = chat-completions\n"
            "endpoint.url = http://127.0.0.1:9/v1/chat/completions\nendpoint.model = luna\n"
            "endpoint.auth = none\n")
        with self.assertRaisesRegex(ValueError, "several APIs"):
            self.resolve(role_models={-1: "codex-gpt-6-luna"}, catalog=registry)
        with self.assertRaisesRegex(ValueError, "running role"):
            self.resolve(role_models={2: "codex-gpt-6-luna"})

    def test_config_file_entries_for_roles_that_do_not_run_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auto.json"
            # Not runnable: Messages needs a token budget for an uncatalogued model.
            path.write_text(json.dumps({"version": 3, "worker": {
                "model_api": "messages", "model": "uncatalogued-worker"}}))
            self.assertEqual(set(self.resolve(path=path)), {1, -1})
            with self.assertRaisesRegex(ValueError, "provide it explicitly"):
                resolve_config(path)

    def test_resume_keeps_each_role_model_and_ignores_catalog_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"

            def create(**kwargs):
                path.write_text(json.dumps(auto.saved_document(self.resolve(**kwargs))))

            def resume(overrides=None, **kwargs):
                settings = self.resolve(overrides, saved=auto.load_saved_config(path), **kwargs)
                return settings[1]["model"], settings[-1]["model"], settings.sources[-1]

            create(overrides={"model": "codex-gpt-6-astra"})
            self.assertEqual(resume(role_models={1: "codex-gpt-6-sol"}),
                             ("codex-gpt-6-sol", "codex-gpt-6-astra", "saved"))
            self.assertEqual(resume({"model": "codex-gpt-6-luna"}),
                             ("codex-gpt-6-luna", "codex-gpt-6-luna", "command line"))
            create(role_defaults={1: "codex-gpt-6-astra"})
            self.assertEqual(json.loads(path.read_text())["watcher"],
                             {"name": "watcher", "instructions": None})
            self.assertEqual(resume(role_models={1: "codex-gpt-6-sol"}),
                             ("codex-gpt-6-sol", "codex-gpt-6-sol", FOLLOWS_MAIN))
            create(role_defaults={1: "codex-gpt-6-astra", -1: "codex-gpt-6-luna"})
            self.assertEqual(resume(role_defaults={1: "codex-gpt-6-sol", -1: "codex-gpt-6-sol"}),
                             ("codex-gpt-6-astra", "codex-gpt-6-luna", "saved"))


class InstructionTests(unittest.TestCase):
    def setUp(self):
        self.settings = resolve_config()
        self.base_url = "http://127.0.0.1:54321"

    def text(self, index):
        return auto._instructions(index, self.settings[index], self.base_url).text

    def test_cooperation_paragraph_precedes_each_default_role(self):
        preambles = []
        for index, name in ((1, "main"), (2, "worker"), (-1, "watcher")):
            with self.subTest(index=index):
                text = self.text(index)
                preamble, rest = text.split("\n\n", 1)
                preambles.append(preamble)
                self.assertIn("cooperative effort to complete the user's task", preamble)
                self.assertIn("shared message board", preamble)
                self.assertIn("iterative revisions toward a verified result", preamble)
                self.assertTrue(rest.startswith(f"You are {name}"))
                self.assertEqual(text.count("# Shared message board instructions"), 1)
                self.assertIn(f"Address: {self.base_url}\n", text)
                self.assertIn(f"Read {self.base_url}/README.md", text)
        self.assertEqual(len(set(preambles)), 1)
        self.assertIn("The host displays a debug event", self.text(-1))
        self.assertIn("No model polling is needed", self.text(-1))

    def test_main_delegates_implementation_and_reviews_iteratively(self):
        text = self.text(1)
        for requirement in (
            "You are main, the user-facing planner and reviewer",
            "acceptance criteria, and required checks",
            "Call board_post_plan to assign implementation and tests to worker #2",
            "Worker owns code and test edits",
            "do not instead assign worker a read-only review and implement the change yourself",
            "Use board_read_thread to obtain the result for each plan",
            "Review the diff and reported checks against the user's requirements",
            "post a concrete follow-up execution plan in the same thread",
            "review the revised work",
            "remain active through worker execution and your review",
            "later worker results do not automatically restart it",
            "If blocked, report what remains unresolved",
            "Respect explicitly planning-only or review-only user requests",
            "Do not resubmit a plan after an uncertain tool outcome without checking the board",
            "Local update_plan only maintains your checklist; it does not delegate work",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, text)

    def test_worker_implements_revises_and_reports_checks(self):
        text = self.text(2)
        for requirement in (
            "You are worker, the implementer",
            "including code changes and tests when implementation is requested",
            "Do not substitute another implementation proposal",
            "For follow-up assignments, revise the existing work according to main's review",
            "Respect explicit read-only assignments",
            "Finish relevant commands before handing work back to main for review",
            "changed files, checks and their outcomes, and remaining blockers",
            "do not claim unperformed work",
            "The host will publish your final outcome",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, text)

    def test_custom_instructions_replace_defaults_but_keep_board_discovery(self):
        for index in (1, 2, -1):
            for custom in ("Use my custom role contract.", "", " \n"):
                with self.subTest(index=index, custom=custom):
                    self.settings[index]["instructions"] = custom
                    body, board = self.text(index).split("\n\n# Shared message board instructions\n\n", 1)
                    self.assertEqual(body, custom)
                    self.assertIn(f"Address: {self.base_url}\n", board)
                    self.assertIn(f"Read {self.base_url}/README.md", board)

    def test_without_board_main_has_no_default_and_bodies_are_verbatim(self):
        # Main's default role text is about delegation, so without the board it
        # has none, like the CLI; the watcher keeps its host-only note.
        self.assertIsNone(auto._instructions(1, self.settings[1], None))
        watcher = auto._instructions(-1, self.settings[-1], None).text
        self.assertEqual(watcher, auto._ROLE_INSTRUCTIONS[-1])
        self.assertNotIn("board", watcher.lower())
        for index in (1, -1):
            for custom in ("Use my custom role contract.", ""):
                with self.subTest(index=index, custom=custom):
                    self.settings[index]["instructions"] = custom
                    self.assertEqual(auto._instructions(index, self.settings[index], None).text, custom)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "session"
        self.calls = {1: [], 2: []}
        self.options = {1: [], 2: []}
        self.effects = []
        self.threads = {i: [] for i in (1, 2, -1)}
        self.closed = []
        self.tool_names = {}
        self.models_built = []

    def session(self, scripts, extra_tools=(), *, settings_overrides=None,
                settings_updates=None, role_models=None, worker_board=True, **kwargs):
        """Start a scripted session; most runtime tests opt into the worker/board.

        worker_board=False omits the flag to exercise _Session's default.
        """
        test = self
        if worker_board:
            kwargs["enable_experimental_worker_board"] = True
        class Model:
            def __init__(self, index):
                self.index = index
                self.outcomes = deque(scripts.get(index, ()))
                test.threads[index].append(threading.get_ident())
                test.models_built.append(index)

            def sample(self, context, **params):
                test.threads[self.index].append(threading.get_ident())
                # A watcher (#-1) entry appears only once the watcher samples.
                test.calls.setdefault(self.index, []).append(context.copy())
                test.options.setdefault(self.index, []).append(params.get("sample_params"))
                saved = load_interaction_save(test.path / "contexts" / f"{self.index}.jsonl")
                test.assertEqual(saved.items, context.items)
                test.assertFalse(context.pending_tool_calls())
                if not self.outcomes:
                    raise AssertionError("unexpected model request")
                value = self.outcomes.popleft()
                if isinstance(value, BaseException):
                    raise value
                return value(context) if callable(value) else value

        class Tools(Environment):
            def __init__(self, index, tools):
                self.index = index
                test.threads[index].append(threading.get_ident())
                test.tool_names[index] = [tool.spec.name for tool in tools]
                super().__init__((*tools, *extra_tools))
            def execute_tool_calls(self, calls):
                test.threads[self.index].append(threading.get_ident())
                saved = load_interaction_save(test.path / "contexts" / f"{self.index}.jsonl")
                test.assertEqual(saved.pending_tool_calls(), tuple(calls))
                return super().execute_tool_calls(calls)
            def close(self):
                test.threads[self.index].append(threading.get_ident())
                test.closed.append(self.index)

        saved = (auto.load_saved_config(self.path / "config.json")
                 if kwargs.get("resume") and self.path.exists() else None)
        settings = resolve_config(
            overrides={"cwd": self.temp.name, **(settings_overrides or {})}, saved=saved,
            role_models=role_models,
        )
        for index, updates in (settings_updates or {}).items():
            settings[index].update(updates)
        session = auto._Session(self.path, settings,
                               model_factory=lambda i, args: Model(i),
                               environment_factory=lambda i, args, tools: Tools(i, tools), **kwargs).start()
        self.addCleanup(session.close)
        return session

    def finished(self, session, thread):
        wait_for(lambda: session.thread_result(thread) is not None)
        return session.thread_result(thread)

    def test_role_tool_snapshots_resume_only_when_changed(self):
        def extra(description):
            return (Tool(ToolSpec("extra", description, {}),
                         lambda args, **kwargs: ToolOutcome("unused")),)

        session = self.session({}, extra_tools=extra("first"), worker_board=False)
        session.close()
        for index in (1, -1):
            path = self.path / "contexts" / f"{index}.jsonl"
            context = load_interaction_save(path)
            self.assertEqual([s.name for s in context.latest_tools().specs],
                             [*self.tool_names[index], "extra"])
            # A compaction prefix drops tool snapshots from model_items, but
            # resume must still compare the raw log's latest snapshot.
            context.append(ContextPrefix((Message("assistant", "summary"),)))
            auto.save_interaction_save(path, context)

        for tools, expected_count in ((extra("first"), 1), (extra("changed"), 2), ((), 3), ((), 3)):
            session = self.session({}, extra_tools=tools, worker_board=False, resume=True)
            session.close()
            for index in (1, -1):
                context = load_interaction_save(self.path / "contexts" / f"{index}.jsonl")
                snapshots = [item for item in context if isinstance(item, Tools)]
                self.assertEqual(len(snapshots), expected_count)
                self.assertEqual([s.name for s in snapshots[-1].specs],
                                 [*self.tool_names[index], *(t.spec.name for t in tools)])
                self.assertFalse(any(isinstance(item, Tools) for item in context.model_items()))
        self.assertEqual(self.calls, {1: [], 2: []})

        # Legacy context files get one fresh snapshot on their next resume.
        for index in (1, -1):
            path = self.path / "contexts" / f"{index}.jsonl"
            context = load_interaction_save(path)
            auto.save_interaction_save(path, auto.InteractionContext(
                item for item in context if not isinstance(item, Tools)
            ))
        session = self.session({}, worker_board=False, resume=True)
        session.close()
        for index in (1, -1):
            context = load_interaction_save(self.path / "contexts" / f"{index}.jsonl")
            self.assertEqual(sum(isinstance(item, Tools) for item in context), 1)

    def settled(self, session, submitted):
        wait_for(lambda: session.task_result(submitted) is not None)
        return session.task_result(submitted)

    def files(self):
        return {path.relative_to(self.path): path.read_bytes()
                for path in self.path.rglob("*") if path.is_file()}

    def test_startup_context_summaries_share_one_global_display_item(self):
        session = self.session({}, role_models={2: "worker-model"}, settings_updates={
            1: {"name": "lead custom"},
            2: {"name": "builder custom"},
            -1: {"name": "observer custom"},
        })
        events = session.drain_events()
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event.index is None for event in events))
        self.assertEqual([item.text for item in events[0].items], [
            f"Save directory: {self.path}",
            "Warning: local tools are unsandboxed; use a trusted model and workspace.",
        ])
        self.assertEqual(len(events[1].items), 1)
        summary = events[1].items[0].text
        # Each role's model and its source; the board watcher runs no model.
        self.assertEqual(summary.splitlines(), [
            "#1 (lead custom): chat-completions / (server default) at 127.0.0.1:8000 (built-in default)",
            "#2 (builder custom): chat-completions / worker-model at 127.0.0.1:8000 (command line)",
            "#-1 (observer custom): no model (observe-only)",
        ])
        self.assertEqual(len(summary.split("\n")), 3)
        self.assertFalse(summary.endswith("\n"))
        self.assertEqual(auto._display_events(session, events)[-1], events[1].items[0])
        self.assertFalse((self.path / "model-bindings.json").exists())

    def test_debug_model_binding_snapshot_is_opt_in(self):
        self.session({}, debug_save_model_binding=True)
        path = self.path / "model-bindings.json"
        self.assertTrue(path.is_file())
        document = json.loads(path.read_text())
        self.assertEqual(set(document["bindings"]), {"1", "2", "-1"})
        self.assertEqual(
            document["bindings"]["1"]["endpoint"]["api"],
            "chat-completions",
        )

    def test_busy_phase_classification(self):
        session = self.session({})
        for phase in ("starting", "sampling", "compacting", "executing tools", "saving",
                      "awaiting watcher"):
            with self.subTest(phase=phase):
                session._phase(2, phase)
                self.assertTrue(session._is_busy(2))
        for phase in ("quiescent", "closed"):
            with self.subTest(phase=phase):
                session._phase(2, phase)
                self.assertFalse(session._is_busy(2))

    def test_headless_wait_keeps_idle_board_reachable_and_processes_work(self):
        session = self.session({1: [answer("headless answer")]})
        runner = threading.Thread(target=auto._headless, args=(session,))
        runner.start()
        try:
            with urlopen(session.service.base_url + "/README.md", timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"Auto message board", response.read())
            self.assertTrue(runner.is_alive())
            self.assertEqual(self.calls, {1: [], 2: []})
            submitted = session.submit("headless task", request_id="headless-test")
            self.assertTrue(self.finished(session, submitted["thread_id"]))
            self.assertEqual([record.kind for record in
                              session.service.board.records(submitted["thread_id"])],
                             ["user", "answer"])
            self.assertTrue(runner.is_alive())
        finally:
            session.request_stop()
            runner.join(3)
            session.close()
        self.assertFalse(runner.is_alive())
        self.assertFalse(any(thread.is_alive() for thread in session._threads.values()))

    def test_quiet_one_prompt_success_has_no_context_display(self):
        session = self.session({1: [answer("quiet answer")]})
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(auto._one_prompt(session, "quiet task", display=False), 0)
        self.assertEqual(output.getvalue(), "")
        records = session.service.board.records()
        self.assertEqual([record.kind for record in records], ["user", "answer"])
        saved = load_interaction_save(self.path / "contexts" / "1.jsonl")
        self.assertTrue(any(isinstance(item, Message) and item.content == "quiet answer"
                            for item in saved))

    def test_quiet_one_prompt_failure_has_no_context_display(self):
        failure = ModelTransportError("SECRET", failure=ModelFailure("transport", "safe"))
        session = self.session({1: [failure]})
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(auto._one_prompt(session, "failing task", display=False), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual([record.kind for record in session.service.board.records()],
                         ["user", "answer"])
        self.assertFalse(session.service.board.records()[-1].success)

    def test_headless_wait_stops_on_board_persistence_failure(self):
        session = self.session({})
        with session.service.board.changed:
            session.service.board.failed = True
            session.service.board.changed.notify_all()
        self.assertEqual(auto._headless(session), 1)
        self.assertEqual(session.drain_events(), [])

    def test_main_and_worker_can_run_past_eight_samples_by_default(self):
        tool = Tool(ToolSpec("noop", "continue the test", {}),
                    lambda *args, **kwargs: ToolOutcome("ok"))
        def rounds(prefix, count):
            return [ModelSample((ToolCall("noop", f"{prefix}-{i}", "{}"),)) for i in range(count)]
        session = self.session({
            1: [ModelSample((ToolCall("board_post_plan", "delegate", '{"content":"long task"}'),)),
                *rounds("main", 8), answer("main complete")],
            2: [*rounds("worker", 9), answer("worker complete")],
        }, (tool,))
        source = session.submit("long conversation")
        self.assertTrue(self.finished(session, source["thread_id"]))
        for index in (1, 2):
            self.assertEqual(len(self.calls[index]), 10)
            self.assertTrue(all(options == SampleParams(enable_auto_compaction=True) for options in self.options[index]))

    def test_explicit_limits_still_apply(self):
        tool = Tool(ToolSpec("noop", "continue the test", {}),
                    lambda *args, **kwargs: ToolOutcome("ok"))
        session = self.session({1: [
            ModelSample((ToolCall("noop", "first", "{}"),)),
            ModelSample((ToolCall("noop", "second", "{}"),)),
            answer("must not be sampled"),
        ]}, (tool,), settings_overrides={"max_samples": 2, "max_output_tokens": 128})
        source = session.submit("bounded conversation")
        self.assertFalse(self.finished(session, source["thread_id"]))
        self.assertEqual(len(self.calls[1]), 2)
        self.assertTrue(all(options.max_output_tokens == 128 for options in self.options[1]))
        self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))

    def test_final_sample_at_explicit_limit_succeeds(self):
        session = self.session({1: [answer()]}, settings_overrides={"max_samples": 1})
        source = session.submit("one sample")
        self.assertTrue(self.finished(session, source["thread_id"]))
        self.assertEqual(len(self.calls[1]), 1)

    def test_quiescent_board_first_mixed_roles_thread_affinity_and_debug_only_watcher(self):
        tool = Tool(ToolSpec("do_work", "Write proof", {"type": "object"}),
                    lambda args, **kwargs: (self.effects.append(threading.get_ident()) or ToolOutcome("proof written")))
        session = self.session({
            1: [ModelSample((Message("assistant", "Delegating, not final"),
                             ToolCall("board_post_plan", "p1", '{"content":"Write proof"}'))), answer("main done")],
            2: [ModelSample((ToolCall("do_work", "p1", "{}"),)), answer("worker done")],
        }, (tool,))
        for i in (1, 2, -1):
            self.assertIn("quiescent", session.status(i))
            context = load_interaction_save(self.path / "contexts" / f"{i}.jsonl")
            self.assertEqual(len(context), 3)
            self.assertIsInstance(context[-1], Tools)
            self.assertIn(session.service.base_url + "/README.md", context[1].text)
        self.assertEqual(session.service.board.records(), ())
        self.assertEqual(self.calls, {1: [], 2: []})
        self.assertEqual(self.effects, [])
        first = session.submit("original task", request_id="user-request")
        self.assertEqual(session.submit("original task", request_id="user-request"), first)
        self.assertTrue(self.finished(session, first["thread_id"]))
        records = session.service.board.records()
        self.assertEqual([r.kind for r in records].count("user"), 1)
        self.assertEqual([r.kind for r in records].count("plan"), 1)
        self.assertTrue(all(r.thread_id == first["thread_id"] for r in records))
        plan = next(r for r in records if r.kind == "plan")
        result = next(r for r in records if r.kind == "result")
        self.assertEqual(result.reply_to, plan.record_id)
        self.assertEqual(result.content, "worker done")
        self.assertEqual(len(self.calls[1]), 2)
        self.assertEqual(len(self.calls[2]), 2)
        self.assertEqual(len(self.effects), 1)
        debug = [e for e in session.drain_events() if e.kind == "debug"]
        self.assertEqual(len(debug), 1)
        self.assertEqual(debug[0].index, -1)
        self.assertIn("condition fired", debug[0].items[0].text)
        # The latest user request only asks for a debug event, not a watcher model turn/log schema.
        self.assertEqual(len(load_interaction_save(self.path / "contexts" / "-1.jsonl")), 3)
        session.close()
        self.assertCountEqual(self.closed, (1, 2, -1))
        for index, thread_ids in self.threads.items():
            self.assertEqual(len(set(thread_ids)), 1, index)
        self.assertEqual(len({v[0] for v in self.threads.values()}), 3)
        self.assertEqual(self.effects[0], self.threads[2][0])
        self.assertFalse(any(t.is_alive() for t in session._threads.values()))

    def test_waits_for_worker_and_fast_completion_watch_notification(self):
        worker_entered, release_worker, release_watcher = (threading.Event() for _ in range(3))
        original_watch = auto._Session._watch
        def delayed_watch(session):
            release_watcher.wait(5)
            original_watch(session)
        def worker(_context):
            worker_entered.set()
            self.assertTrue(release_worker.wait(5))
            return answer("worker late")
        with mock.patch.object(auto._Session, "_watch", delayed_watch):
            session = self.session({
                1: [ModelSample((ToolCall("board_post_plan", "p", '{"content":"slow"}'),)), answer()],
                2: [worker],
            })
        try:
            source = session.submit("task")
            self.assertTrue(worker_entered.wait(3))
            wait_for(lambda: source["record_id"] in session._done)
            self.assertIsNone(session.thread_result(source["thread_id"]))
            release_worker.set()
            wait_for(lambda: any(r.kind == "result" for r in session.service.board.records()))
            self.assertIsNone(session.thread_result(source["thread_id"]))
            release_watcher.set()
            self.assertTrue(self.finished(session, source["thread_id"]))
        finally:
            release_worker.set()
            release_watcher.set()
            session.close()

    def test_failure_and_reasoning_only_do_not_fire_watch_or_execute_recovered_calls(self):
        failure = ModelTransportError("SECRET", completed_items=(
            Message("assistant", "partial"), ToolCall("forbidden", "bad", "{}")),
            failure=ModelFailure("transport", "safe failure"))
        tool = Tool(ToolSpec("forbidden", "must not execute", {}),
                    lambda *args, **kwargs: self.fail("recovered call executed"))
        session = self.session({1: [failure, ModelSample((Reasoning("thought only"),))]}, (tool,))
        for prompt in ("first", "second"):
            source = session.submit(prompt)
            self.assertFalse(self.finished(session, source["thread_id"]))
        context = load_interaction_save(self.path / "contexts" / "1.jsonl")
        self.assertFalse(context.pending_tool_calls())
        self.assertTrue(any(isinstance(i, ToolResult) and not i.success for i in context))
        self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))
        self.assertNotIn("SECRET", (self.path / "index.jsonl").read_text())

    def test_whole_tool_batch_places_injected_messages_after_all_results(self):
        tool = Tool(ToolSpec("inject", "inject a user message", {}),
                    lambda *args, **kwargs: ToolOutcome("ok", user_messages=(Message("user", "injected"),)))
        session = self.session({1: [ModelSample((ToolCall("inject", "a", "{}"),
                                                ToolCall("inject", "b", "{}"))), answer()]}, (tool,))
        source = session.submit("batch")
        self.assertTrue(self.finished(session, source["thread_id"]))
        items = self.calls[1][1].items
        results = [n for n, i in enumerate(items) if isinstance(i, ToolResult)]
        messages = [n for n, i in enumerate(items) if isinstance(i, Message) and i.content == "injected"]
        self.assertEqual(len(results), 2)
        self.assertLess(max(results), min(messages))

    def test_summary_save_failure_blocks_watcher(self):
        session = self.session({1: [answer()]})
        real_save = auto.save_interaction_save
        def save(path, context):
            if isinstance(context.items[-1], TurnSummary):
                from pythia.interaction import SaveError
                raise SaveError("disk")
            return real_save(path, context)
        with mock.patch.object(auto, "save_interaction_save", save):
            source = session.submit("fail summary")
            self.assertFalse(self.finished(session, source["thread_id"]))
            session.close()
        self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))

    def test_encoding_failure_is_a_fatal_checkpoint_failure(self):
        session = self.session({1: [answer("unencodable: \ud800")]})
        source = session.submit("bad model text")
        self.assertFalse(self.finished(session, source["thread_id"]))
        self.assertTrue(session._stop.is_set())
        session.close()
        saved = load_interaction_save(self.path / "contexts" / "1.jsonl")
        self.assertFalse(any(isinstance(i, Message) and i.role == "assistant" for i in saved))
        self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))

    def test_saved_config_is_version_3_with_running_roles(self):
        session = self.session({}, role_models={2: "worker-model"})
        document = json.loads((self.path / "config.json").read_text())
        self.assertEqual(document["version"], 3)
        self.assertEqual(set(document), {"version", "main", "worker", "watcher"})
        for key in ("compaction_mode", "compaction_keep_recent_tokens",
                    "compaction_max_output_tokens"):
            self.assertIn(key, document["main"])
        # The worker keeps the model it chose; the watcher follows main.
        self.assertEqual(document["worker"]["model"], "worker-model")
        self.assertEqual(document["watcher"], {"name": "watcher", "instructions": None})
        session.close()

    def compactor(self, **kwargs):
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult(
            (ContextPrefix((Message("user", "summary"),)),), protocol="pi",
        )
        for name, value in kwargs.items():
            setattr(compactor.compact, name, value)
        return compactor

    def test_threshold_compaction_uses_the_turn_params_and_settings(self):
        compactor = self.compactor()
        with mock.patch.object(auto, "create_default_compactor", return_value=compactor) as create:
            session = self.session({1: [answer()]}, settings_overrides={
                "auto_compact_tokens": 1, "compaction_keep_recent_tokens": 0,
                "compaction_max_output_tokens": 64,
            })
            source = session.submit("compact first")
            self.assertTrue(self.finished(session, source["thread_id"]))
        create.assert_called_once()
        self.assertEqual(create.call_args.args[1], CompactionSettings(
            mode="pi", keep_recent_tokens=0, max_output_tokens=64,
        ))
        self.assertEqual(compactor.compact.call_args.kwargs["sample_params"], self.options[1][0])
        self.assertEqual(self.calls[1][0].model_items(), (Message("user", "summary"),))

    def test_threshold_nothing_to_compact_samples_normally(self):
        compactor = self.compactor(side_effect=NothingToCompact("nothing precedes the recent tail"))
        with mock.patch.object(auto, "create_default_compactor", return_value=compactor):
            session = self.session({1: [answer()]}, settings_overrides={"auto_compact_tokens": 1})
            source = session.submit("nothing to compact")
            self.assertTrue(self.finished(session, source["thread_id"]))
        compactor.compact.assert_called_once()
        self.assertEqual(len(self.calls[1]), 1)

    def test_overflow_compacts_once_and_retries_the_sample(self):
        compactor = self.compactor()
        overflow = ModelContextWindowError("prompt is too long", failure=ModelFailure(
            "context_window", "Messages HTTP 400: context window exceeded"))
        with mock.patch.object(auto, "create_default_compactor", return_value=compactor):
            session = self.session({1: [overflow, answer("recovered")]},
                                   settings_overrides={"max_samples": 1})
            source = session.submit("too long")
            self.assertTrue(self.finished(session, source["thread_id"]))
        compactor.compact.assert_called_once()
        # The failed attempt does not count against max_samples.
        self.assertEqual(len(self.calls[1]), 2)
        self.assertEqual(self.calls[1][1].model_items(), (Message("user", "summary"),))
        context = load_interaction_save(self.path / "contexts" / "1.jsonl")
        failure = next(i for i, item in enumerate(context) if isinstance(item, ModelFailure))
        self.assertIsInstance(context[failure + 2], ContextPrefix)

    def test_failed_overflow_compaction_is_reported_by_class_name(self):
        compactor = self.compactor(side_effect=CompactionContextWindowError(
            "summary request for the history (3 items, ~9 estimated tokens) exceeded"))
        overflow = ModelContextWindowError("prompt is too long")
        with mock.patch.object(auto, "create_default_compactor", return_value=compactor):
            session = self.session({1: [overflow]})
            source = session.submit("too long")
            self.assertFalse(self.finished(session, source["thread_id"]))
        errors = [item.text for event in session.drain_events() if event.kind == "error"
                  for item in event.items]
        self.assertIn("Task failed (CompactionContextWindowError); effects may have occurred. "
                      "Details withheld.", errors)
        self.assertEqual(len(self.calls[1]), 1)

    def test_compaction_continuation_is_not_an_end_of_turn(self):
        reached, release = threading.Event(), threading.Event()
        def final(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return answer("after compaction")
        session = self.session({1: [ModelSample((OpaqueCompaction("opaque", "messages"),),
                                               stop_reason="compaction"), final]})
        try:
            source = session.submit("compact")
            self.assertTrue(reached.wait(3))
            self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))
            release.set()
            self.assertTrue(self.finished(session, source["thread_id"]))
            self.assertEqual(len([e for e in session.drain_events() if e.kind == "debug"]), 1)
        finally:
            release.set()
            session.close()

    def test_stop_during_sample_checkpoints_but_does_not_execute_new_calls(self):
        reached, release = threading.Event(), threading.Event()
        def sample(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return ModelSample((ToolCall("effect", "pending", "{}"),))
        tool = Tool(ToolSpec("effect", "must not run after stop", {}),
                    lambda *args, **kwargs: (self.effects.append("bad") or ToolOutcome("bad")))
        session = self.session({1: [sample]}, (tool,))
        session.submit("stop in flight")
        self.assertTrue(reached.wait(3))
        closer = threading.Thread(target=session.close)
        closer.start()
        try:
            wait_for(session._stop.is_set)
            release.set()
            closer.join(3)
            self.assertFalse(closer.is_alive())
            self.assertEqual(self.effects, [])
            saved = load_interaction_save(self.path / "contexts" / "1.jsonl")
            self.assertEqual([c.call_id for c in saved.pending_tool_calls()], ["pending"])
            self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))
        finally:
            release.set()
            closer.join(5)

    def test_board_failure_during_sample_blocks_the_following_tool_batch(self):
        reached, release = threading.Event(), threading.Event()
        def sample(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return ModelSample((ToolCall("effect", "pending", "{}"),))
        tool = Tool(ToolSpec("effect", "must not run with failed persistence", {}),
                    lambda *args, **kwargs: (self.effects.append("bad") or ToolOutcome("bad")))
        session = self.session({1: [sample]}, (tool,))
        try:
            source = session.submit("first")
            self.assertTrue(reached.wait(3))
            with mock.patch("pythia.interaction._auto_board.os.fsync", side_effect=OSError("disk")):
                with self.assertRaises(BoardError):
                    session.submit("failed commit")
            release.set()
            wait_for(session._stop.is_set)
            session.close()
            self.assertEqual(self.effects, [])
            self.assertFalse(session.thread_result(source["thread_id"]))
            self.assertEqual([r.content for r in session.service.board.records()], ["first"])
        finally:
            release.set()
            session.close()

    def test_startup_failure_closes_constructed_environments_without_role_work(self):
        closed = []
        class Env(Environment):
            def __init__(self, index):
                super().__init__()
                self.index = index
                self.owner = threading.get_ident()
            def close(self):
                closed.append((self.index, self.owner, threading.get_ident()))
        def model(index, args):
            if index == 2:
                raise RuntimeError("FAKE_SECRET")
            return mock.Mock()
        session = auto._Session(self.path, resolve_config(), model_factory=model,
                               environment_factory=lambda i, a, t: Env(i),
                               enable_experimental_worker_board=True)
        with self.assertRaises(RuntimeError):
            session.start()
        self.assertCountEqual([i for i, _, _ in closed], (1, 2, -1))
        self.assertTrue(all(a == b for _, a, b in closed))
        self.assertFalse(any(t.is_alive() for t in session._threads.values()))
        self.assertEqual(session.service.board.records(), ())
        self.assertNotIn("FAKE_SECRET", repr(session.drain_events()))

    def test_interactive_navigation_is_responsive_without_direct_worker_input(self):
        reached, release = threading.Event(), threading.Event()
        def sample(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return answer("interactive done")
        session = self.session({1: [sample]})
        test = self
        class Terminal:
            closed = False
            def __init__(self):
                self.keys = deque()
                self.frames, self.items = [], []
                self.stage = 0
                self.busy_frames = 0
                self.closing_frames = 0
                self.main_busy_prompts = []
                self.waiting_close_prompts = []
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, text):
                self.keys.extend(SimpleNamespace(key=k, data=v) for k, v in (
                    ("c-u", ""), ("c-k", ""), ("<bracketed-paste>", text), ("c-m", "\r")))
            def read_keys(self):
                keys = tuple(self.keys)
                self.keys.clear()
                return keys
            def render(self, editor, status, items, prompt=":> "):
                self.frames.append((editor, status, prompt))
                self.items.extend(items)
                if self.stage == 0:
                    self.submit("/context 2")
                    self.stage = 1
                elif self.stage == 1 and status.startswith("#2"):
                    self.submit("do not send to worker")
                    self.stage = 2
                elif self.stage == 2:
                    test.assertEqual(editor.text, "do not send to worker")
                    test.assertEqual(session.service.board.records(), ())
                    self.submit("/context 1")
                    self.stage = 3
                elif self.stage == 3:
                    self.submit("actual task")
                    self.stage = 4
                elif self.stage == 4 and reached.is_set():
                    self.busy_frames += 1
                    self.main_busy_prompts.append(prompt)
                    if self.busy_frames == 18:
                        self.submit("/context 2")
                        self.stage = 5
                elif self.stage == 5 and status.startswith("#2"):
                    test.assertEqual(prompt, ":> ")
                    self.submit("/context #-1")
                    self.stage = 6
                elif self.stage == 6 and status.startswith("#-1"):
                    test.assertEqual(prompt, ":> ")
                    self.submit("/context 1")
                    self.stage = 7
                elif self.stage == 7 and status.startswith("#1"):
                    test.assertNotEqual(prompt, ":> ")
                    self.keys.extend((SimpleNamespace(key="<bracketed-paste>", data="draft"),
                                      SimpleNamespace(key="left", data="")))
                    self.stage = 8
                elif self.stage == 8 and editor.text == "draft":
                    test.assertEqual(editor.cursor, 4)
                    test.assertNotEqual(prompt, ":> ")
                    self.keys.append(SimpleNamespace(key="c-c", data=""))
                    self.stage = 9
                elif self.stage == 9 and status.startswith("closing"):
                    test.assertEqual(editor.text, "draft")
                    test.assertEqual(editor.cursor, 4)
                    test.assertNotEqual(prompt, ":> ")
                    self.closing_frames += 1
                    self.waiting_close_prompts.append(prompt)
                    if self.closing_frames == 18:
                        release.set()
                        self.stage = 10
        terminal = Terminal()
        try:
            result = asyncio.run(asyncio.wait_for(auto._interactive(session, terminal), timeout=6))
            self.assertEqual(result, 0)
            self.assertEqual(len(self.calls[1]), 1)
            self.assertEqual(self.calls[2], [])
            self.assertEqual([r.kind for r in session.service.board.records()], ["user", "answer"])
            self.assertTrue(any(s.startswith("#1 (main) - sampling") for _, s, _ in terminal.frames))
            self.assertTrue(any(s.startswith("#-1") for _, s, _ in terminal.frames))
            busy_prompts = set(terminal.main_busy_prompts)
            closing_prompts = set(terminal.waiting_close_prompts)
            self.assertGreaterEqual(len(busy_prompts), 2)
            self.assertGreaterEqual(len(closing_prompts), 2)
            self.assertNotIn(":> ", busy_prompts | closing_prompts)
            self.assertTrue(any(i.text == "[#1 (main) - assistant] interactive done" for i in terminal.items))
            self.assertTrue(any(i.text.startswith("[#-1 (watcher) - debug]") for i in terminal.items))
            self.assertFalse(any(i.text == "[#1 (main)]" for i in terminal.items))
            self.assertFalse(any(t.is_alive() for t in session._threads.values()))
        finally:
            release.set()
            session.close()

    def test_interactive_pending_submission_animates_before_context_work(self):
        session = self.session({1: [answer("accepted")]})
        entered, release = threading.Event(), threading.Event()
        submit = session.submit

        def delayed_submit(text, *, request_id):
            entered.set()
            self.assertTrue(release.wait(5))
            return submit(text, request_id=request_id)

        session.submit = delayed_submit
        test = self

        class Terminal:
            closed = False

            def __init__(self):
                self.keys = deque()
                self.prompts = []
                self.stage = 0
                self.completed_thread = None
                self.submission_complete = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, text):
                self.keys.extend(SimpleNamespace(key=key, data=data) for key, data in (
                    ("c-u", ""), ("c-k", ""), ("<bracketed-paste>", text), ("c-m", "\r")))

            def read_keys(self):
                keys = tuple(self.keys)
                self.keys.clear()
                return keys

            def render(self, editor, status, items, prompt=":> "):
                self.submission_complete |= any(
                    item.text.startswith("Posted user thread ") for item in items
                )
                if self.stage == 0:
                    self.submit("pending draft")
                    self.stage = 1
                elif self.stage == 1 and entered.is_set():
                    test.assertFalse(session._is_busy(1))
                    test.assertEqual(editor.text, "pending draft")
                    self.prompts.append(prompt)
                    if len(self.prompts) == 18:
                        release.set()
                        self.stage = 2
                elif self.stage == 2:
                    users = [record for record in session.service.board.records()
                             if record.kind == "user"]
                    if (self.submission_complete
                            and len(users) == 1
                            and session.thread_result(users[0].thread_id) is True
                            and status.startswith("#1 (main) - quiescent")):
                        test.assertEqual(prompt, ":> ")
                        self.completed_thread = users[0].thread_id
                        self.submit("/quit")
                        self.stage = 3

        terminal = Terminal()
        try:
            self.assertEqual(asyncio.run(asyncio.wait_for(
                auto._interactive(session, terminal), timeout=6)), 0)
            self.assertGreaterEqual(len(set(terminal.prompts)), 2)
            self.assertNotIn(":> ", terminal.prompts)
            records = session.service.board.records(terminal.completed_thread)
            self.assertEqual([record.kind for record in records], ["user", "answer"])
            self.assertTrue(session.thread_result(terminal.completed_thread))
            self.assertEqual(len(self.calls[1]), 1)
            self.assertFalse(any(thread.is_alive() for thread in session._threads.values()))
        finally:
            release.set()
            session.close()

    def test_interactive_global_close_animates_on_idle_selected_context(self):
        session = self.session({})
        entered, release = threading.Event(), threading.Event()
        close = session.close

        def delayed_close():
            entered.set()
            self.assertTrue(release.wait(5))
            close()

        session.close = delayed_close
        test = self

        class Terminal:
            closed = False

            def __init__(self):
                self.keys = deque()
                self.closing_prompts = []
                self.stage = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, text):
                self.keys.extend(SimpleNamespace(key=key, data=data) for key, data in (
                    ("c-u", ""), ("c-k", ""), ("<bracketed-paste>", text), ("c-m", "\r")))

            def read_keys(self):
                keys = tuple(self.keys)
                self.keys.clear()
                return keys

            def render(self, editor, status, items, prompt=":> "):
                if self.stage == 0:
                    self.submit("/context -1")
                    self.stage = 1
                elif self.stage == 1 and status.startswith("#-1"):
                    test.assertFalse(session._is_busy(-1))
                    test.assertEqual(prompt, ":> ")
                    self.keys.extend((SimpleNamespace(key="<bracketed-paste>", data="draft"),
                                      SimpleNamespace(key="left", data="")))
                    self.stage = 2
                elif self.stage == 2 and editor.text == "draft":
                    test.assertEqual(editor.cursor, 4)
                    self.keys.append(SimpleNamespace(key="c-c", data=""))
                    self.stage = 3
                elif self.stage == 3 and status.startswith("closing") and entered.is_set():
                    test.assertFalse(session._is_busy(-1))
                    test.assertEqual((editor.text, editor.cursor), ("draft", 4))
                    test.assertNotEqual(prompt, ":> ")
                    self.closing_prompts.append(prompt)
                    if len(self.closing_prompts) == 18:
                        release.set()
                        self.stage = 4

        terminal = Terminal()
        try:
            self.assertEqual(asyncio.run(asyncio.wait_for(
                auto._interactive(session, terminal), timeout=6)), 0)
            self.assertGreaterEqual(len(set(terminal.closing_prompts)), 2)
            self.assertNotIn(":> ", terminal.closing_prompts)
            self.assertEqual(session.service.board.records(), ())
            self.assertEqual(self.calls, {1: [], 2: []})
            self.assertFalse(any(thread.is_alive() for thread in session._threads.values()))
        finally:
            release.set()
            session.close()

    def test_stop_before_input_and_local_navigation_do_not_wake_owners(self):
        session = self.session({})
        for text, target in (("/context 2", 2), ("/context #-1", -1), ("/contexts", 1)):
            selected, items, stop = auto._local_command(text, 1, session)
            self.assertEqual(selected, target)
            self.assertFalse(stop)
        for text in ("/context 0", "/context -2", "/context -1\nhello", "/resume"):
            with self.assertRaises(ValueError):
                auto._local_command(text, 1, session)
        session.close()
        session.close()
        self.assertEqual(self.calls, {1: [], 2: []})
        self.assertEqual(session.service.board.records(), ())
        self.assertCountEqual(self.closed, (1, 2, -1))

    def test_existing_directory_is_not_modified(self):
        self.path.mkdir()
        marker = self.path / "keep"
        marker.write_text("untouched")
        with self.assertRaises(FileExistsError):
            self.session({})
        self.assertEqual(list(self.path.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "untouched")

    def test_invalid_session_board_auth_policy_creates_no_save(self):
        settings = resolve_config(overrides={"cwd": self.temp.name})
        for invalid in (None, 0, 1, "False"):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                auto._Session(self.path, settings, enable_board_auth=invalid)
            self.assertFalse(self.path.exists())

    def test_resume_restores_history_without_replay_then_runs_one_new_task(self):
        session = self.session({
            1: [ModelSample((ToolCall("board_post_plan", "delegate", '{"content":"old plan"}'),)),
                answer("old main")],
            2: [answer("old worker")],
        }, settings_updates={2: {"instructions": "saved custom worker instructions"}})
        old = session.submit("old task", request_id="old-request")
        self.assertTrue(self.finished(session, old["thread_id"]))
        old_records = session.service.board.records()
        old_contexts = {index: load_interaction_save(
            self.path / "contexts" / f"{index}.jsonl").items for index in (1, 2, -1)}
        old_calls = {index: len(calls) for index, calls in self.calls.items()}
        old_url = session.service.base_url
        session.close()

        resumed = self.session({1: [answer("new main")]}, resume=True)
        self.assertEqual({index: len(calls) for index, calls in self.calls.items()}, old_calls)
        self.assertEqual(resumed.service.board.records(), old_records)
        for index in (1, 2, -1):
            restored = load_interaction_save(self.path / "contexts" / f"{index}.jsonl")
            self.assertEqual(restored.items[:len(old_contexts[index])], old_contexts[index])
            self.assertIsInstance(restored[-1], auto.Instructions)
            self.assertIn(resumed.service.base_url, restored[-1].text)
            self.assertIn("without restoring old command-session IDs or runtime state",
                          restored[-1].text)
        self.assertIn("saved custom worker instructions",
                      load_interaction_save(self.path / "contexts" / "2.jsonl")[-1].text)
        new = resumed.submit("new task", request_id="new-request")
        self.assertTrue(self.finished(resumed, new["thread_id"]))
        records = resumed.service.board.records()
        self.assertEqual(records[:len(old_records)], old_records)
        self.assertEqual([record.kind for record in records[len(old_records):]], ["user", "answer"])
        self.assertNotEqual(old_url, "")
        self.assertEqual(len(self.calls[1]), old_calls[1] + 1)
        self.assertTrue(any("old main" in item.content for item in self.calls[1][-1]
                            if isinstance(item, Message)))

    def test_resume_lock_and_corrupt_context_fail_without_replacing_data(self):
        session = self.session({})
        config = (self.path / "config.json").read_bytes()
        settings = resolve_config(saved=auto.load_saved_config(self.path / "config.json"))
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            auto._Session(self.path, settings, resume=True,
                          enable_experimental_worker_board=True,
                          model_factory=lambda *_: self.fail("model initialized"),
                          environment_factory=lambda *_: self.fail("environment initialized")).start()
        self.assertEqual((self.path / "config.json").read_bytes(), config)
        session.close()
        context_path = self.path / "contexts" / "2.jsonl"
        context_path.write_text("broken\n")
        broken = context_path.read_bytes()
        with self.assertRaises(Exception):
            auto._Session(self.path, settings, resume=True,
                          enable_experimental_worker_board=True).start()
        self.assertEqual(context_path.read_bytes(), broken)

    def test_resume_lock_cannot_be_bypassed_by_directory_symlink(self):
        session = self.session({})
        alias = self.path.parent / "alias"
        alias.symlink_to(self.path, target_is_directory=True)
        settings = resolve_config(saved=auto.load_saved_config(self.path / "config.json"))
        before = {path.relative_to(self.path): path.read_bytes()
                  for path in self.path.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            auto._Session(alias, settings, resume=True,
                          enable_experimental_worker_board=True,
                          model_factory=lambda *_: self.fail("model initialized"),
                          environment_factory=lambda *_: self.fail("environment initialized")).start()
        self.assertEqual({path.relative_to(self.path): path.read_bytes()
                          for path in self.path.rglob("*") if path.is_file()}, before)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(auto.main([
                "--resume", "--headless", "--enable-experimental-worker-board", "--save", str(alias)
            ]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("auto failed: Auto save directory is already in use.",
                      stderr.getvalue())
        session.close()
        resumed = auto._Session(alias, settings, resume=True,
                                enable_experimental_worker_board=True).start()
        try:
            self.assertTrue(resumed._resumed)
            self.assertEqual(resumed.service.board.records(), ())
        finally:
            resumed.close()

    def test_resume_rejects_nonempty_context_without_initial_metadata(self):
        session = self.session({})
        session.close()
        context_path = self.path / "contexts" / "2.jsonl"
        auto.save_interaction_save(
            context_path, auto.InteractionContext((Message("user", "not auto metadata"),))
        )
        before = {path.relative_to(self.path): path.read_bytes()
                  for path in self.path.rglob("*") if path.is_file()}
        settings = resolve_config(saved=auto.load_saved_config(self.path / "config.json"))
        with self.assertRaisesRegex(ValueError, "initialization metadata"):
            auto._Session(self.path, settings, resume=True,
                          enable_experimental_worker_board=True,
                          model_factory=lambda *_: self.fail("model initialized"),
                          environment_factory=lambda *_: self.fail("environment initialized")).start()
        self.assertEqual({path.relative_to(self.path): path.read_bytes()
                          for path in self.path.rglob("*") if path.is_file()}, before)

    def test_resume_explicit_cwd_repairs_moved_saved_workspace(self):
        old = Path(self.temp.name) / "old-workspace"
        new = Path(self.temp.name) / "new-workspace"
        old.mkdir()
        session = self.session({}, settings_overrides={"cwd": str(old)})
        session.close()
        prefixes = {index: load_interaction_save(
            self.path / "contexts" / f"{index}.jsonl").items for index in (1, 2, -1)}
        before = {path.relative_to(self.path): path.read_bytes()
                  for path in self.path.rglob("*") if path.is_file()}
        old.rename(new)
        saved = auto.load_saved_config(self.path / "config.json")
        with self.assertRaisesRegex(ValueError, "cwd must be an existing"):
            resolve_config(saved=saved)
        self.assertEqual({path.relative_to(self.path): path.read_bytes()
                          for path in self.path.rglob("*") if path.is_file()}, before)

        settings = resolve_config(overrides={"cwd": str(new)}, saved=saved)
        self.assertTrue(all(value["cwd"] == str(new.absolute()) for value in settings.values()))
        resumed = auto._Session(self.path, settings, resume=True,
                                enable_experimental_worker_board=True).start()
        try:
            for index in (1, 2, -1):
                context = load_interaction_save(self.path / "contexts" / f"{index}.jsonl")
                self.assertEqual(context.items[:len(prefixes[index])], prefixes[index])
        finally:
            resumed.close()
        updated = auto.load_saved_config(self.path / "config.json")
        again = resolve_config(saved=updated)
        self.assertTrue(all(value["cwd"] == str(new.absolute()) for value in again.values()))
        repeated = auto._Session(self.path, again, resume=True,
                                 enable_experimental_worker_board=True).start()
        repeated.close()

    def test_resume_missing_directory_is_fresh_and_pending_call_is_not_rerun(self):
        session = self.session({}, resume=True)
        self.assertFalse(session._resumed)
        session.close()
        context_path = self.path / "contexts" / "1.jsonl"
        context = load_interaction_save(context_path)
        call = ToolCall("exec_command", "old command", '{"cmd":"must-not-run"}')
        context.extend((call, ModelSampleBoundary()))
        auto.save_interaction_save(context_path, context)
        prefix = context.items
        calls = dict((index, len(values)) for index, values in self.calls.items())

        resumed = self.session({}, resume=True)
        restored = load_interaction_save(context_path)
        self.assertEqual(restored.items[:len(prefix)], prefix)
        result = restored[len(prefix)]
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.call_id, call.call_id)
        self.assertFalse(result.success)
        self.assertIn("not rerun", result.output)
        self.assertFalse(restored.pending_tool_calls())
        self.assertEqual(dict((index, len(values)) for index, values in self.calls.items()), calls)
        resumed.close()
        repeated = self.session({}, resume=True)
        again = load_interaction_save(context_path)
        self.assertEqual(again.items[:len(restored)], restored.items)
        self.assertEqual(sum(isinstance(item, ToolResult) and item.call_id == call.call_id
                             for item in again), 1)
        repeated.close()

    def test_default_runs_only_main_and_watcher_without_board(self):
        session = self.session({1: [answer("main done")], -1: [answer("Complete.")]},
                               worker_board=False)
        self.assertFalse(session.worker_board)
        self.assertEqual(session.roles, (1, -1))
        self.assertIsNone(session.service)
        for name in ("index.jsonl", "index.md", "index.html", "contexts/2.jsonl"):
            self.assertFalse((self.path / name).exists(), name)
        # No worker model/environment and no board tools; the watcher supervises.
        self.assertEqual(self.threads[2], [])
        self.assertEqual(self.tool_names, {1: [], -1: ["resume_main", "read_main_context"]})
        self.assertCountEqual(self.models_built, (1, -1))
        self.assertEqual(len(load_interaction_save(self.path / "contexts" / "1.jsonl")), 2)
        watcher = load_interaction_save(self.path / "contexts" / "-1.jsonl")
        self.assertEqual(watcher[1], auto.Instructions(auto._SUPERVISOR_INSTRUCTIONS))
        self.assertEqual(set(json.loads((self.path / "config.json").read_text())),
                         {"version", "main", "watcher"})
        events = session.drain_events()
        self.assertEqual(events[-1].items[0].text.splitlines(), [
            "#1 (main): chat-completions / (server default) at 127.0.0.1:8000 (built-in default)",
            "#-1 (watcher): chat-completions / (server default) at 127.0.0.1:8000 (same as main)",
        ])
        self.assertEqual(self.calls, {1: [], 2: []})

        submitted = session.submit("plain task", request_id="unused-without-board")
        self.assertEqual(submitted, {"record_id": "1"})
        self.assertTrue(self.settled(session, submitted))
        self.assertEqual([item for item in self.calls[1][0] if isinstance(item, Message)],
                         [Message("user", "plain task")])
        report = [item for item in self.calls[-1][0] if isinstance(item, Message)][-1].content
        for expected in ("yielded on task 1 (watcher resumes so far: 0)", "User request:\nplain task",
                         "Outcome: ended", "Main's final answer:\nmain done"):
            self.assertIn(expected, report)
        debug = [event for event in session.drain_events() if event.kind == "debug"]
        self.assertEqual([event.index for event in debug], [-1, -1])
        self.assertEqual([event.items[0].text for event in debug], [
            "[debug] main end-of-turn condition fired: #1 source=1",
            "[debug] watcher released main: #1 source=1",
        ])
        self.assertFalse(session.has_errors)
        with self.assertRaisesRegex(BoardError, "Unknown task"):
            session.task_result({"record_id": "2"})
        session.close()
        self.assertCountEqual(self.closed, (1, -1))
        for index in (1, -1):  # Each context's model/tools stay on its owner thread.
            self.assertEqual(len(set(self.threads[index])), 1, index)
        self.assertFalse(any(thread.is_alive() for thread in session._threads.values()))

    def test_default_tasks_run_in_fifo_order_and_every_yield_reaches_the_watcher(self):
        reached, release = threading.Event(), threading.Event()
        def first(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return answer("first done")
        failure = ModelTransportError("SECRET", failure=ModelFailure("transport", "safe"))
        # Observe-only: the watcher sees every yield but builds no model.
        session = self.session({1: [first, failure, answer("third done")]}, worker_board=False,
                               watcher_max_resumes=0)
        self.assertEqual(self.models_built, [1])
        self.assertEqual(self.tool_names[-1], [])
        self.assertEqual(load_interaction_save(self.path / "contexts" / "-1.jsonl")[1],
                         auto.Instructions(auto._ROLE_INSTRUCTIONS[-1]))
        try:
            handles = [session.submit("first")]
            self.assertTrue(reached.wait(3))
            handles += [session.submit("second"), session.submit("third")]
            self.assertEqual([handle["record_id"] for handle in handles], ["1", "2", "3"])
            self.assertEqual([session.task_result(handle) for handle in handles], [None] * 3)
            release.set()
            self.assertEqual([self.settled(session, handle) for handle in handles],
                             [True, False, True])
        finally:
            release.set()
        self.assertEqual([[item.content for item in call if isinstance(item, Message)
                           and item.role == "user"][-1] for call in self.calls[1]],
                         ["first", "second", "third"])
        debug = [event.items[0].text for event in session.drain_events() if event.kind == "debug"]
        self.assertEqual(debug, [
            "[debug] main end-of-turn condition fired: #1 source=1",
            "[debug] watcher released main: #1 source=1",
            "[debug] main yielded: #1 source=2 failed (ModelTransportError)",
            "[debug] watcher released main: #1 source=2",
            "[debug] main end-of-turn condition fired: #1 source=3",
            "[debug] watcher released main: #1 source=3",
        ])
        self.assertNotIn(-1, self.calls)
        self.assertTrue(session.has_errors)  # The failed task is reported, not fatal.
        self.assertFalse(session._stop.is_set())
        self.assertNotIn("SECRET", (self.path / "contexts" / "1.jsonl").read_text())

    def test_default_submission_limits_and_close_report_unexecuted_tasks(self):
        reached, release = threading.Event(), threading.Event()
        def first(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return answer("first done")
        session = self.session({1: [first]}, worker_board=False)
        closer = threading.Thread(target=session.close)
        try:
            for invalid, reason in ((" \n", "Nonempty"), ("bad \ud800", "UTF-8")):
                with self.subTest(reason=reason), self.assertRaisesRegex(BoardError, reason):
                    session.submit(invalid)
            session.submit("running")
            self.assertTrue(reached.wait(3))
            for n in range(auto._MAX_PENDING_TASKS - 1):
                session.submit(f"queued {n}")
            with self.assertRaisesRegex(BoardError, "Pending work limit"):
                session.submit("one too many")
            closer.start()
            wait_for(session._stop.is_set)
            with self.assertRaisesRegex(BoardError, "not accepting"):
                session.submit("after stop")
            release.set()
            closer.join(5)
            self.assertFalse(closer.is_alive())
        finally:
            release.set()
            if closer.is_alive():
                closer.join(5)
        self.assertEqual(len(self.calls[1]), 1)
        self.assertTrue(session.task_result({"record_id": "1"}))
        notices = [item.text for event in session.drain_events() for item in event.items]
        self.assertIn(f"Stopped with {auto._MAX_PENDING_TASKS - 1} queued user tasks that were not "
                      "executed; without the board, queued tasks are not saved or replayed.", notices)

    def test_default_resume_restores_without_board_and_rejects_board_flag(self):
        session = self.session({1: [answer("old main")], -1: [answer("Complete.")]},
                               worker_board=False)
        self.assertTrue(self.settled(session, session.submit("old task")))
        session.close()
        old = {index: load_interaction_save(self.path / "contexts" / f"{index}.jsonl").items
               for index in (1, -1)}
        before = self.files()
        settings = resolve_config(saved=auto.load_saved_config(self.path / "config.json"))
        with self.assertRaisesRegex(ValueError, "created without the experimental worker/board"):
            auto._Session(self.path, settings, resume=True, enable_experimental_worker_board=True,
                          model_factory=lambda *_: self.fail("model initialized"),
                          environment_factory=lambda *_: self.fail("environment initialized")).start()
        self.assertEqual(self.files(), before)

        resumed = self.session({1: [answer("new main")], -1: [answer("Complete.")]},
                               worker_board=False, resume=True)
        restored = {index: load_interaction_save(self.path / "contexts" / f"{index}.jsonl")
                    for index in (1, -1)}
        for index, context in restored.items():
            self.assertEqual(context.items[:len(old[index])], old[index])
        self.assertEqual(restored[1][-1], auto.Instructions(auto._RESTART_NOTICE))
        self.assertEqual(restored[-1][-1], auto.Instructions(
            auto._SUPERVISOR_INSTRUCTIONS + "\n\n" + auto._RESTART_NOTICE))
        # The watcher's own history (report, decision) was restored, not replayed.
        self.assertTrue(any(isinstance(item, Message) and "User request:\nold task" in item.content
                            for item in restored[-1]))
        self.assertFalse((self.path / "index.jsonl").exists())
        new = resumed.submit("new task")
        self.assertEqual(new, {"record_id": "1"})
        self.assertTrue(self.settled(resumed, new))
        self.assertTrue(any(isinstance(item, Message) and item.content == "old main"
                            for item in self.calls[1][-1]))

    def test_board_save_requires_the_flag_to_resume(self):
        self.session({}).close()
        before = self.files()
        settings = resolve_config(saved=auto.load_saved_config(self.path / "config.json"))
        with self.assertRaisesRegex(ValueError, "uses the experimental worker/board; resume it "
                                    "with --enable-experimental-worker-board"):
            auto._Session(self.path, settings, resume=True,
                          model_factory=lambda *_: self.fail("model initialized"),
                          environment_factory=lambda *_: self.fail("environment initialized")).start()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(auto.main(["--resume", "--no-user-model-catalog", "--prompt", "task",
                                        "--save", str(self.path)]), 1)
        self.assertIn("auto failed: This auto save uses the experimental worker/board",
                      stderr.getvalue())
        self.assertEqual(self.files(), before)

    def test_default_navigation_lists_only_main_and_watcher(self):
        session = self.session({}, worker_board=False)
        _, items, _ = auto._local_command("/contexts", 1, session)
        self.assertEqual(items[0].text.splitlines(),
                         ["* #1 (main) - quiescent", "  #-1 (watcher) - quiescent"])
        self.assertEqual(auto._local_command("/context #-1", 1, session)[0], -1)
        for text in ("/context 2", "/context #2", "/context 0"):
            with self.subTest(text=text), self.assertRaisesRegex(
                    ValueError, r"^Use /context 1 or /context -1\.$"):
                auto._local_command(text, 1, session)
        self.assertEqual(self.calls, {1: [], 2: []})

    def test_default_one_prompt_displays_main_and_watcher_only(self):
        for display in (False, True):
            with self.subTest(display=display):
                self.path = Path(self.temp.name) / f"session-{display}"
                session = self.session({1: [answer("shown answer")], -1: [answer("Complete.")]},
                                       worker_board=False)
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(auto._one_prompt(session, "shown task", display=display), 0)
                text = output.getvalue()
                if not display:
                    self.assertEqual(text, "")
                    continue
                self.assertIn("[#1 (main) - assistant] shown answer", text)
                self.assertIn("[#-1 (watcher) - debug] main end-of-turn condition fired: "
                              "#1 source=1", text)
                self.assertIn("[#-1 (watcher) - user] Main (#1) yielded on task 1", text)
                self.assertIn("[#-1 (watcher) - assistant] Complete.", text)
                self.assertIn("[#-1 (watcher) - debug] watcher released main: #1 source=1", text)
                self.assertNotIn("decision failed", text)
                self.assertNotIn("#2", text)
                self.assertNotIn("Board", text)

    def test_default_interactive_queues_tasks_for_main_and_shows_watcher(self):
        session = self.session({1: [answer("interactive done")], -1: [answer("Complete.")]},
                               worker_board=False)
        test = self

        class Terminal:
            closed = False

            def __init__(self):
                self.keys = deque()
                self.texts = []
                self.stage = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, text):
                self.keys.extend(SimpleNamespace(key=key, data=data) for key, data in (
                    ("c-u", ""), ("c-k", ""), ("<bracketed-paste>", text), ("c-m", "\r")))

            def read_keys(self):
                keys = tuple(self.keys)
                self.keys.clear()
                return keys

            def render(self, editor, status, items, prompt=":> "):
                self.texts.extend(item.text for item in items)
                if self.stage == 0:
                    self.submit("/context 2")
                    self.stage = 1
                elif self.stage == 1 and "Use /context 1 or /context -1." in self.texts:
                    self.submit("/context -1")
                    self.stage = 2
                elif self.stage == 2 and status.startswith("#-1 (watcher)"):
                    self.submit("watcher draft")
                    self.stage = 3
                elif self.stage == 3 and ("Switch to /context 1 to submit a new user task. "
                                          "Draft preserved.") in self.texts:
                    test.assertEqual(editor.text, "watcher draft")
                    self.submit("/context 1")
                    self.stage = 4
                elif self.stage == 4 and status.startswith("#1 (main)"):
                    self.submit("interactive task")
                    self.stage = 5
                elif (self.stage == 5 and "Queued user task 1 for #1 (main)." in self.texts
                      and session.task_result({"record_id": "1"}) is True
                      and status.startswith("#1 (main) - quiescent")):
                    test.assertEqual(prompt, ":> ")
                    self.submit("/quit")
                    self.stage = 6

        terminal = Terminal()
        self.assertEqual(asyncio.run(asyncio.wait_for(
            auto._interactive(session, terminal), timeout=6)), 0)
        self.assertEqual(terminal.stage, 6)
        self.assertEqual(len(self.calls[1]), 1)
        self.assertEqual([item for item in self.calls[1][0] if isinstance(item, Message)],
                         [Message("user", "interactive task")])
        self.assertIn("[#1 (main) - assistant] interactive done", terminal.texts)
        self.assertIn("[#-1 (watcher) - debug] main end-of-turn condition fired: #1 source=1",
                      terminal.texts)
        self.assertIn("[#-1 (watcher) - assistant] Complete.", terminal.texts)
        self.assertFalse(any("decision failed" in text for text in terminal.texts))
        self.assertFalse(any(thread.is_alive() for thread in session._threads.values()))

    def debug_texts(self, session):
        return [item.text for event in session.drain_events() if event.kind == "debug"
                for item in event.items]

    @staticmethod
    def last_user(context):
        return [item for item in context if isinstance(item, Message) and item.role == "user"][-1]

    def test_watcher_recovers_main_from_a_failed_turn(self):
        failure = ModelTransportError("SECRET", failure=ModelFailure("transport", "safe summary"))
        session = self.session({
            1: [failure, answer("recovered answer")],
            -1: [resume("Retry the request."), answer("Resuming main."), answer("Complete.")],
        }, worker_board=False)
        self.assertTrue(self.settled(session, session.submit("flaky task")))
        self.assertFalse(session.has_errors)  # Recovery contributes to success.
        self.assertEqual(len(self.calls[1]), 2)
        self.assertEqual(self.last_user(self.calls[1][1]),
                         Message("user", auto._FOLLOW_UP_HEADER + "Retry the request."))
        first, second = (self.last_user(self.calls[-1][i]).content for i in (0, 2))
        self.assertIn("Outcome: failed (ModelTransportError)", first)
        self.assertIn("Failure: transport: safe summary", first)
        self.assertIn("(watcher resumes so far: 1)", second)
        self.assertIn("Main's final answer:\nrecovered answer", second)
        self.assertEqual(self.debug_texts(session), [
            "[debug] main yielded: #1 source=1 failed (ModelTransportError)",
            "[debug] watcher resumed main: #1 source=1",
            "[debug] main end-of-turn condition fired: #1 source=1",
            "[debug] watcher released main: #1 source=1",
        ])
        for index in (1, -1):
            self.assertNotIn("SECRET", (self.path / "contexts" / f"{index}.jsonl").read_text())

    def test_watcher_continues_main_after_its_sample_limit(self):
        tool = Tool(ToolSpec("noop", "continue the test", {}),
                    lambda *args, **kwargs: ToolOutcome("ok"))
        session = self.session({
            1: [ModelSample((ToolCall("noop", "n1", "{}"),)), answer("finished")],
            -1: [resume("Continue where you stopped."), answer("Resuming."), answer("Complete.")],
        }, (tool,), worker_board=False, settings_updates={1: {"max_samples": 1}})
        self.assertTrue(self.settled(session, session.submit("long task")))
        self.assertIn("Outcome: failed (SampleLimitExceeded)",
                      self.last_user(self.calls[-1][0]).content)
        self.assertEqual(len(self.calls[1]), 2)  # The resumed turn gets a fresh limit.
        self.assertFalse(session.has_errors)

    def test_resume_budget_bounds_watcher_resumes(self):
        session = self.session({
            1: [answer("draft 1"), answer("draft 2"), answer("draft 3")],
            -1: [resume("More.", "r1"), answer("Again."), resume("More.", "r2"), answer("Again.")],
        }, worker_board=False, watcher_max_resumes=2)
        self.assertTrue(self.settled(session, session.submit("polish")))
        self.assertEqual(len(self.calls[1]), 3)
        self.assertEqual(len(self.calls[-1]), 4)  # The over-budget yield costs no sample.
        self.assertEqual([text for text in self.debug_texts(session) if "watcher" in text], [
            "[debug] watcher resumed main: #1 source=1",
            "[debug] watcher resumed main: #1 source=1",
            "[debug] watcher released main: #1 source=1",
        ])

    def test_watcher_failure_releases_main_and_keeps_its_outcome(self):
        failure = ModelTransportError("SECRET", failure=ModelFailure("transport", "down"))
        session = self.session({1: [answer("first"), answer("second")],
                                -1: [failure, answer("Complete.")]}, worker_board=False)
        self.assertTrue(self.settled(session, session.submit("one")))
        self.assertTrue(self.settled(session, session.submit("two")))
        errors = [item.text for event in session.drain_events() if event.kind == "error"
                  for item in event.items]
        self.assertEqual(errors, ["Watcher decision failed (ModelTransportError); main was "
                                  "released. Details withheld."])
        self.assertFalse(session.has_errors)
        self.assertFalse(session._stop.is_set())

    def test_fault_reaches_the_watcher_before_it_stops_the_session(self):
        session = self.session({1: [answer()]}, worker_board=False)
        real_save = auto.save_interaction_save

        def save(path, context):
            if path.name == "1.jsonl" and isinstance(context.items[-1], TurnSummary):
                from pythia.interaction import SaveError
                raise SaveError("disk")
            return real_save(path, context)

        with mock.patch.object(auto, "save_interaction_save", save):
            submitted = session.submit("fail summary")
            self.assertFalse(self.settled(session, submitted))
            wait_for(session._stop.is_set)
        texts = [(event.kind, item.text) for event in session.drain_events() for item in event.items]
        fault = ("debug", "[debug] main yielded: #1 source=1 failed (SaveError), non-resumable")
        error = ("error", "Task failed (SaveError); effects may have occurred. Details withheld.")
        # Main waited for the watcher (which propagates, for now) before stopping.
        self.assertLess(texts.index(fault), texts.index(error))
        self.assertNotIn(-1, self.calls)  # A fault costs no watcher sample.
        with self.assertRaisesRegex(BoardError, "not accepting"):
            session.submit("after the fault")

    def test_stop_while_main_awaits_the_watcher_releases_without_resuming(self):
        reached, release = threading.Event(), threading.Event()

        def deciding(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return resume("Keep going.")

        session = self.session({1: [answer("done")], -1: [deciding]}, worker_board=False)
        closer = threading.Thread(target=session.close)
        try:
            submitted = session.submit("task")
            self.assertTrue(reached.wait(3))
            self.assertTrue(session.status(1).startswith("#1 (main) - awaiting watcher"))
            self.assertTrue(session._is_busy(1))
            self.assertIsNone(session.task_result(submitted))
            closer.start()
            wait_for(lambda: session.task_result(submitted) is not None)
            self.assertTrue(session.task_result(submitted))  # Settled from main's last yield.
            self.assertTrue(closer.is_alive())  # Close drains the watcher's in-flight sample.
            release.set()
            closer.join(5)
            self.assertFalse(closer.is_alive())
        finally:
            release.set()
            if closer.is_alive():
                closer.join(5)
        self.assertEqual(len(self.calls[1]), 1)  # The late resume was never applied.
        self.assertIn("[debug] watcher released main: #1 source=1", self.debug_texts(session))
        self.assertFalse(any(thread.is_alive() for thread in session._threads.values()))

    def test_user_tasks_queued_during_a_decision_run_after_main_is_released(self):
        reached, release = threading.Event(), threading.Event()

        def deciding(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return resume("Follow up on A.")

        session = self.session({
            1: [answer("A1"), answer("A2"), answer("B1")],
            -1: [deciding, answer("Resuming."), answer("Complete."), answer("Complete.")],
        }, worker_board=False)
        try:
            a = session.submit("task A")
            self.assertTrue(reached.wait(3))
            b = session.submit("task B")
            release.set()
            self.assertTrue(self.settled(session, a))
            self.assertTrue(self.settled(session, b))
        finally:
            release.set()
        self.assertEqual([self.last_user(call).content for call in self.calls[1]],
                         ["task A", auto._FOLLOW_UP_HEADER + "Follow up on A.", "task B"])

    def test_read_main_context_addresses_main_log_as_of_the_yield(self):
        session = self.session({
            1: [answer("visible answer")],
            -1: [ModelSample((ToolCall("read_main_context", "tail", "{}"),
                              ToolCall("read_main_context", "head", '{"start": 0, "limit": 1}'),
                              ToolCall("read_main_context", "bad", '{"limit": 0}'))),
                 answer("Complete.")],
        }, worker_board=False)
        self.assertTrue(self.settled(session, session.submit("inspect me")))
        main_log = load_interaction_save(self.path / "contexts" / "1.jsonl")
        results = {item.call_id: item for item in
                   load_interaction_save(self.path / "contexts" / "-1.jsonl")
                   if isinstance(item, ToolResult)}
        tail = json.loads(results["tail"].output)
        self.assertEqual(tail["revision"], len(main_log))
        self.assertEqual(tail["items"][-1]["index"], len(main_log) - 1)
        self.assertFalse(tail["has_more"])
        texts = "\n".join(entry["text"] for entry in tail["items"])
        self.assertIn("inspect me", texts)
        self.assertIn("visible answer", texts)
        head = json.loads(results["head"].output)
        self.assertEqual((head["start"], head["next"], head["has_more"]), (0, 1, True))
        self.assertEqual(head["items"][0]["type"], "Init")
        self.assertFalse(results["bad"].success)
        self.assertIn("limit must be an integer", results["bad"].output)

    def test_invalid_watcher_budget_creates_no_save(self):
        settings = resolve_config(overrides={"cwd": self.temp.name})
        for kwargs in ({"watcher_max_resumes": -1}, {"watcher_max_resumes": True},
                       {"watcher_max_resumes": 1.5},
                       {"watcher_max_resumes": 1, "enable_experimental_worker_board": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                auto._Session(self.path, settings, **kwargs)
            self.assertFalse(self.path.exists())


class EntryPointTests(unittest.TestCase):
    def test_default_rejects_board_only_options_before_any_session(self):
        for argv in (["--headless"], ["--headless", "--resume"],
                     ["--board-port", "8123", "--prompt", "task"],
                     ["--enable-board-auth", "False", "--prompt", "task"]):
            with self.subTest(argv=argv):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (mock.patch.object(auto, "_Session") as session,
                      redirect_stdout(stdout), redirect_stderr(stderr)):
                    self.assertEqual(auto.main([*argv, "--no-user-model-catalog",
                                                "--save", "unused"]), 1)
                session.assert_not_called()
                self.assertEqual(stdout.getvalue(), "")
                self.assertTrue(stderr.getvalue().startswith("auto failed: "), stderr.getvalue())
                self.assertIn("require", stderr.getvalue())
                self.assertIn("--enable-experimental-worker-board", stderr.getvalue())

    def test_default_passes_flag_and_prints_no_board_line(self):
        created = []

        class Session:
            has_errors = False
            service = None

            def __init__(self, *args, **kwargs):
                created.append(kwargs)

            def start(self):
                return self

            def close(self):
                pass

            def drain_events(self):
                return ()

        # Explicitly supplied default board options are harmless.
        for argv, budget in (([], None), (["--board-port", "0", "--enable-board-auth", "True"], None),
                             (["--enable-experimental-worker-board", "False"], None),
                             (["--watcher-max-resumes", "0"], 0),
                             (["--watcher-max-resumes", "3"], 3)):
            with self.subTest(argv=argv):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (mock.patch.object(auto, "_Session", Session),
                      mock.patch.object(auto, "_one_prompt", return_value=0) as run,
                      redirect_stdout(stdout), redirect_stderr(stderr)):
                    self.assertEqual(auto.main([*argv, "--no-user-model-catalog", "--prompt", "task",
                                                "--save", "unused"]), 0)
                run.assert_called_once()
                self.assertIs(created[-1]["enable_experimental_worker_board"], False)
                self.assertEqual(created[-1]["watcher_max_resumes"], budget)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(stderr.getvalue(), "")

    def test_watcher_budget_is_validated_before_any_session(self):
        for argv, message in ((["--watcher-max-resumes", "-1"], "nonnegative"),
                              (["--enable-experimental-worker-board", "--watcher-max-resumes", "2"],
                               "does not apply")):
            with self.subTest(argv=argv):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (mock.patch.object(auto, "_Session") as session,
                      redirect_stdout(stdout), redirect_stderr(stderr)):
                    self.assertEqual(auto.main([*argv, "--no-user-model-catalog", "--prompt", "task",
                                                "--save", "unused"]), 1)
                session.assert_not_called()
                self.assertIn(message, stderr.getvalue())

    def stub_main(self, argv, *, environ=None, start_error=None):
        """Run auto.main with a recording stub session: (code, stdout, stderr, settings)."""
        created = []

        class Session:
            has_errors = False
            service = SimpleNamespace(base_url="http://127.0.0.1:1")  # Board mode prints it.

            def __init__(self, path, settings, **kwargs):
                created.append(settings)

            def start(self):
                if start_error is not None:
                    raise start_error
                return self

            def close(self):
                pass

            def drain_events(self):
                return ()

        stdout, stderr = io.StringIO(), io.StringIO()
        with (mock.patch.object(auto, "_Session", Session),
              mock.patch.object(auto, "_one_prompt", return_value=0),
              mock.patch.dict(os.environ, environ or {}),
              redirect_stdout(stdout), redirect_stderr(stderr)):
            code = auto.main([*argv, "--prompt", "task", "--save", "unused"])
        return code, stdout.getvalue(), stderr.getvalue(), created[-1] if created else None

    def test_role_model_options_and_catalog_defaults_reach_the_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.ini"
            catalog_path.write_text(
                "[catalog]\nversion = 4\n\n[auto]\nmain.model = codex-gpt-6-astra-max\n"
                "watcher.model = codex-gpt-6-luna\nworker.model = codex-gpt-6-sol\n")
            auth = Path(tmp) / "auth.json"
            auth.write_text("{}")
            common = ["--model-catalog", str(catalog_path), "--endpoint-auth-file", str(auth)]
            for argv, models, sources in (
                ([], ("codex-gpt-6-astra-max", "codex-gpt-6-luna"),
                 ("catalog default", "catalog default")),
                (["--model", "codex-gpt-6-sol"], ("codex-gpt-6-sol", "codex-gpt-6-sol"),
                 ("command line", "command line")),
                (["--main-model", "codex-gpt-6-sol"], ("codex-gpt-6-sol", "codex-gpt-6-luna"),
                 ("command line", "catalog default")),
                (["--watcher-model", "codex-gpt-6-sol"], ("codex-gpt-6-astra-max", "codex-gpt-6-sol"),
                 ("catalog default", "command line")),
                # A watcher that runs no model takes no default.
                (["--watcher-max-resumes", "0"], ("codex-gpt-6-astra-max", "codex-gpt-6-astra-max"),
                 ("catalog default", FOLLOWS_MAIN)),
            ):
                with self.subTest(argv=argv):
                    code, _, stderr, settings = self.stub_main([*common, *argv])
                    self.assertEqual(code, 0, stderr)
                    self.assertEqual(set(settings), {1, -1})
                    self.assertEqual((settings[1]["model"], settings[-1]["model"]), models)
                    self.assertEqual((settings.sources[1], settings.sources[-1]), sources)
                    # The same Codex route keeps main's credential file.
                    self.assertEqual(settings[-1]["codex_auth_file"], str(auth))
            code, _, stderr, settings = self.stub_main(
                [*common, "--enable-experimental-worker-board"])
            self.assertEqual(code, 0, stderr)
            self.assertEqual({i: settings[i]["model"] for i in settings}, {
                1: "codex-gpt-6-astra-max", 2: "codex-gpt-6-sol", -1: "codex-gpt-6-astra-max"})

    def test_role_options_that_cannot_take_effect_fail_before_any_session(self):
        for argv, message in (
            (["--worker-model", "x"], "--worker-model requires --enable-experimental-worker-board"),
            (["--enable-experimental-worker-board", "--watcher-model", "x"], "has no effect"),
            (["--watcher-max-resumes", "0", "--watcher-model", "x"], "has no effect"),
        ):
            with self.subTest(argv=argv):
                code, _, stderr, settings = self.stub_main(["--no-user-model-catalog", *argv])
                self.assertEqual(code, 1)
                self.assertIsNone(settings)
                self.assertIn(message, stderr)

    def test_shared_runtime_flags_reach_every_role(self):
        code, _, stderr, settings = self.stub_main([
            "--no-user-model-catalog", "--enable-workspace", "False",
            "--enable-auto-compaction", "False", "--max-samples", "4"])
        self.assertEqual(code, 0, stderr)
        for index in (1, -1):
            self.assertIs(settings[index]["enable_workspace"], False)
            self.assertIs(settings[index]["enable_auto_compaction"], False)
            self.assertEqual(settings[index]["max_samples"], 4)

    def test_print_config_shows_each_role_and_saves_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp) / "run"
            stdout, stderr = io.StringIO(), io.StringIO()
            with (mock.patch.object(auto, "_Session") as session,
                  redirect_stdout(stdout), redirect_stderr(stderr)):
                code = auto.main([
                    "--no-user-model-catalog", "--endpoint-url", "http://gpu:8000/v1/chat/completions",
                    "--model", "big", "--watcher-model", "small", "--save", str(save),
                    "--print-config"])
            self.assertEqual(code, 0, stderr.getvalue())
            session.assert_not_called()
            self.assertFalse(save.exists())
            lines = stdout.getvalue().splitlines()
            self.assertEqual(lines[:2], [
                "#1 (main): chat-completions / big at gpu:8000 (command line)",
                "#-1 (watcher): chat-completions / small at gpu:8000 (command line)",
            ])
            document = json.loads("\n".join(lines[2:]))
            self.assertEqual(set(document), {"version", "main", "watcher"})
            self.assertEqual((document["main"]["model"], document["watcher"]["model"]), ("big", "small"))

    def test_missing_credentials_fail_before_any_save_without_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth = Path(tmp) / "auth.json"
            auth.write_text("{}")
            missing = Path(tmp) / "missing.json"
            for argv, message in (
                (["--model", "claude-sonnet-5.5"],
                 "#1 (main) uses claude-sonnet-5.5, which needs the ANTHROPIC_API_KEY "
                 "environment variable; it is not set."),
                (["--model", "codex-gpt-6-luna", "--endpoint-auth-file", str(missing)],
                 f"#1 (main) uses codex-gpt-6-luna, which needs a Codex login at {missing}; "
                 "none was found."),
                (["--model", "codex-gpt-6-luna", "--endpoint-auth-file", str(auth),
                  "--watcher-model", "claude-sonnet-5.5"],
                 "#-1 (watcher) uses claude-sonnet-5.5, which needs the ANTHROPIC_API_KEY"),
            ):
                with self.subTest(argv=argv):
                    code, _, stderr, settings = self.stub_main(
                        ["--no-user-model-catalog", *argv], environ={"ANTHROPIC_API_KEY": " "})
                    self.assertEqual(code, 1)
                    self.assertIsNone(settings)
                    self.assertIn(message, stderr)
            # Observe-only, the watcher needs no credential.
            code, _, stderr, _ = self.stub_main(
                ["--no-user-model-catalog", "--main-model", "codex-gpt-6-luna",
                 "--endpoint-auth-file", str(auth), "--model", "claude-sonnet-5.5",
                 "--watcher-max-resumes", "0"], environ={"ANTHROPIC_API_KEY": ""})
            self.assertEqual(code, 0, stderr)
            code, _, stderr, _ = self.stub_main(
                ["--no-user-model-catalog", "--model", "claude-sonnet-5.5"],
                environ={"ANTHROPIC_API_KEY": "set"})
            self.assertEqual(code, 0, stderr)

    def test_startup_failure_reports_its_safe_message(self):
        error = auto._StartupError("Auto context initialization failed (see context error notices).")
        code, _, stderr, _ = self.stub_main(["--no-user-model-catalog"], start_error=error)
        self.assertEqual(code, 1)
        self.assertIn("auto failed: Auto context initialization failed (see context error notices).",
                      stderr)

    def test_catalog_role_models_run_end_to_end_and_resume_keeps_them(self):
        seen = []

        class Gateway(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(request["model"])
                text = "main answer" if request["model"] == "wire-main" else "Complete."
                data = json.dumps({
                    "choices": [{"message": {"role": "assistant", "content": text},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        server_thread = threading.Thread(target=gateway.serve_forever, kwargs={"poll_interval": .01})
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                url = f"http://127.0.0.1:{gateway.server_port}/v1/chat/completions"

                def write_catalog(watcher):
                    entries = "".join(
                        f"[model.{name}]\nendpoint.api = chat-completions\nendpoint.url = {url}\n"
                        f"endpoint.model = {wire}\nendpoint.auth = none\n\n"
                        for name, wire in (("gw-main", "wire-main"), ("gw-watch", "wire-watch")))
                    (root / "catalog.ini").write_text(
                        f"[catalog]\nversion = 4\n\n{entries}"
                        f"[auto]\nmain.model = gw-main\nwatcher.model = {watcher}\n")

                def run(*extra):
                    return subprocess.run([
                        sys.executable, "-m", "pythia.interaction.auto",
                        "--model-catalog", str(root / "catalog.ini"),
                        "--request-timeout-seconds", "3", "--cwd", tmp,
                        "--save", str(root / "run"), *extra, "--prompt", "Do the task",
                    ], capture_output=True, text=True, timeout=30)

                write_catalog("gw-watch")
                result = run()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("#1 (main): chat-completions / gw-main (catalog default)", result.stdout)
                self.assertIn("#-1 (watcher): chat-completions / gw-watch (catalog default)",
                              result.stdout)
                self.assertEqual(seen, ["wire-main", "wire-watch"])
                # The save keeps its models when the catalog defaults change.
                write_catalog("gw-main")
                seen.clear()
                result = run("--resume")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("#-1 (watcher): chat-completions / gw-watch (saved)", result.stdout)
                self.assertEqual(seen, ["wire-main", "wire-watch"])
                # --model sets both roles.
                seen.clear()
                result = run("--resume", "--model", "gw-main")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(seen, ["wire-main", "wire-main"])
        finally:
            gateway.shutdown()
            server_thread.join()
            gateway.server_close()

    def test_default_one_prompt_subprocess_runs_main_and_watcher_without_board(self):
        seen = []

        def tools(request):
            return {tool["function"]["name"] for tool in request.get("tools", ())}

        def call(name, call_id, arguments):
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)}}]}

        def text(content):
            return {"role": "assistant", "content": content}

        # Main works and answers; the watcher resumes it once, then releases it.
        script = {
            "main": [call("exec_command", "work-1",
                          {"cmd": "printf main > proof.txt", "yield_time_ms": 1000}),
                     text("main answer"),
                     call("exec_command", "work-2",
                          {"cmd": "printf done > done.txt", "yield_time_ms": 1000}),
                     text("all done")],
            "watcher": [call("resume_main", "resume-1", {"content": "Also write done.txt."}),
                        text("Resumed main."), text("Complete.")],
        }

        class Gateway(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(request)
                message = script["watcher" if "resume_main" in tools(request) else "main"].pop(0)
                data = json.dumps({"choices": [{"message": message, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                             "total_tokens": 7}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        server_thread = threading.Thread(target=gateway.serve_forever, kwargs={"poll_interval": .01})
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = root / "run"
                result = subprocess.run([
                    sys.executable, "-m", "pythia.interaction.auto", "--no-user-model-catalog",
                    "--endpoint-url", f"http://127.0.0.1:{gateway.server_port}/v1/chat/completions",
                    "--model", "main", "--request-timeout-seconds", "3", "--cwd", tmp,
                    "--save", str(run), "--prompt", "Do the task",
                ], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertIn("[#1 (main) - assistant] main answer", result.stdout)
                self.assertIn("[#-1 (watcher) - debug] main end-of-turn condition fired: "
                              "#1 source=1", result.stdout)
                self.assertIn("[#-1 (watcher) - debug] watcher resumed main: #1 source=1",
                              result.stdout)
                self.assertIn("[#1 (main) - user] Automated follow-up from the watcher (#-1):",
                              result.stdout)
                self.assertIn("[#1 (main) - assistant] all done", result.stdout)
                self.assertIn("[#-1 (watcher) - debug] watcher released main: #1 source=1",
                              result.stdout)
                self.assertNotIn("Board", result.stdout)
                self.assertNotIn("#2", result.stdout)
                self.assertEqual((root / "proof.txt").read_text(), "main")
                self.assertEqual((root / "done.txt").read_text(), "done")
                for name in ("config.json", "contexts/1.jsonl", "contexts/-1.jsonl"):
                    self.assertTrue((run / name).is_file(), name)
                for name in ("index.jsonl", "index.md", "index.html", "contexts/2.jsonl"):
                    self.assertFalse((run / name).exists(), name)
                self.assertEqual(script, {"main": [], "watcher": []})
                main = [request for request in seen if "resume_main" not in tools(request)]
                watcher = [request for request in seen if "resume_main" in tools(request)]
                self.assertEqual((len(main), len(watcher)), (4, 3))
                for request in main:
                    self.assertIn("exec_command", tools(request))
                    self.assertFalse(any(name.startswith("board_") for name in tools(request)))
                    self.assertFalse(any(message["role"] in {"system", "developer"}
                                         for message in request["messages"]))
                self.assertEqual(main[0]["messages"][0]["role"], "user")
                self.assertIn("Do the task", json.dumps(main[0]["messages"][0]))
                self.assertIn(json.dumps("Automated follow-up from the watcher (#-1):\n\n"
                                         "Also write done.txt.")[1:-1], json.dumps(main[2]["messages"]))
                for request in watcher:
                    self.assertEqual(tools(request), {"resume_main", "read_main_context"})
                    self.assertIn(request["messages"][0]["role"], {"system", "developer"})
                    self.assertIn("the supervisor of main (#1)", json.dumps(request["messages"][0]))
                    self.assertIn(json.dumps("User request:\nDo the task")[1:-1],
                                  json.dumps(request["messages"]))
                self.assertNotIn("Board thread", json.dumps(seen))
        finally:
            gateway.shutdown()
            server_thread.join()
            gateway.server_close()

    @unittest.skipUnless(os.name == "posix", "requires a POSIX pseudo-terminal")
    def test_default_pty_lists_main_and_watcher_and_exits_without_model_work(self):
        import fcntl
        import pty
        import select
        import struct
        import termios

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pty-run"
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0))
            process = subprocess.Popen([
                sys.executable, "-m", "pythia.interaction.auto", "--no-user-model-catalog",
                "--save", str(path), "--endpoint-url", "http://127.0.0.1:1/v1/chat/completions",
                "--request-timeout-seconds", "1",
            ], stdin=slave, stdout=slave, stderr=subprocess.PIPE,
               env={**os.environ, "TERM": "xterm-256color"})
            os.close(slave)
            output = bytearray()

            def expect(marker, after=0):
                deadline = time.monotonic() + 5
                while marker not in output[after:]:
                    if time.monotonic() >= deadline:
                        self.fail(f"PTY did not show {marker!r}: {bytes(output)!r}")
                    if select.select([master], [], [], .05)[0]:
                        try:
                            output.extend(os.read(master, 65536))
                        except OSError:
                            self.fail(f"PTY closed early: {bytes(output)!r}")
            try:
                expect(b"#1 (main) - quiescent")
                start = len(output)
                os.write(master, b"/contexts\r")
                expect(b"#-1 (watcher) - quiescent", start)
                start = len(output)
                os.write(master, b"/context 2\r")
                expect(b"Use /context 1 or /context -1.", start)
                os.write(master, b"\x04")
                self.assertEqual(process.wait(timeout=5), 0)
                self.assertEqual(process.stderr.read(), b"")
                self.assertNotIn(b"Board", output)
                self.assertNotIn(b"#2", output)
                self.assertFalse((path / "index.jsonl").exists())
                self.assertFalse((path / "contexts" / "2.jsonl").exists())
                self.assertEqual(len(load_interaction_save(path / "contexts" / "1.jsonl")), 2)
                self.assertEqual(len(load_interaction_save(path / "contexts" / "-1.jsonl")), 3)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stderr.close()
                os.close(master)

    def test_board_auth_cli_propagation_warning_and_resume_secure_default(self):
        created = []
        class Session:
            has_errors = False
            service = SimpleNamespace(base_url="http://127.0.0.1:43210")
            def __init__(self, *args, **kwargs):
                created.append(kwargs)
            def start(self):
                return self
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "guaranteed-missing-save"
            board = "--enable-experimental-worker-board"
            for argv, expected, warned in (([board, "--headless", "--enable-board-auth", "False"], False, True),
                                           ([board, "--headless"], True, False),
                                           ([board, "--headless", "--enable-board-auth"], True, False),
                                           ([board, "--headless", "--resume"], True, False)):
                with self.subTest(argv=argv):
                    self.assertFalse(missing.exists())
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (mock.patch.object(auto, "_Session", Session),
                          mock.patch.object(auto, "_headless", return_value=0),
                          redirect_stdout(stdout), redirect_stderr(stderr)):
                        self.assertEqual(auto.main([*argv, "--save", str(missing)]), 0)
                    self.assertIs(created[-1]["enable_board_auth"], expected)
                    self.assertIs(created[-1]["enable_experimental_worker_board"], True)
                    self.assertEqual(stdout.getvalue(),
                                     "Board: http://127.0.0.1:43210/README.md\n")
                    self.assertEqual("authentication is disabled" in stderr.getvalue(), warned)

    def test_headless_main_prints_flushed_board_and_bypasses_tui(self):
        class Output(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flushes = 0

            def flush(self):
                self.flushes += 1
                super().flush()

        class Session:
            has_errors = False
            service = SimpleNamespace(base_url="http://127.0.0.1:43210")

            def __init__(self, *args, **kwargs):
                self.closed = False

            def start(self):
                return self

            def close(self):
                self.closed = True

            def drain_events(self):
                return ()

        for prompt in (None, "task"):
            with self.subTest(prompt=prompt):
                output = Output()
                runner = "_headless" if prompt is None else "_one_prompt"
                argv = ["--enable-experimental-worker-board", "--headless", "--save", "unused"]
                if prompt is not None:
                    argv += ["--prompt", prompt]
                def run_after_board(*args, **kwargs):
                    self.assertEqual(output.getvalue(),
                                     "Board: http://127.0.0.1:43210/README.md\n")
                    self.assertGreaterEqual(output.flushes, 1)
                    return 0
                with (mock.patch.object(auto, "_Session", Session),
                      mock.patch.object(auto, runner, side_effect=run_after_board) as run,
                      mock.patch.object(auto, "_interactive",
                                        side_effect=AssertionError("TUI entered")),
                      mock.patch.object(auto, "PosixTerminal",
                                        side_effect=AssertionError("terminal constructed")),
                      mock.patch.object(sys.stdin, "isatty", return_value=False),
                      mock.patch.object(sys.stdout, "isatty", return_value=False),
                      redirect_stdout(output)):
                    self.assertEqual(auto.main(argv), 0)
                self.assertEqual(output.getvalue(),
                                 "Board: http://127.0.0.1:43210/README.md\n")
                self.assertGreaterEqual(output.flushes, 1)
                if prompt is None:
                    run.assert_called_once()
                else:
                    self.assertFalse(run.call_args.kwargs["display"])

        output = Output()
        with (mock.patch.object(auto, "_Session", Session),
              mock.patch.object(auto, "_one_prompt", return_value=0) as run,
              redirect_stdout(output)):
            self.assertEqual(auto.main([
                "--enable-experimental-worker-board", "--headless", "False",
                "--save", "unused", "--prompt", "task"
            ]), 0)
        self.assertTrue(run.call_args.kwargs["display"])
        self.assertEqual(output.getvalue().count("Board: "), 1)

    def test_headless_false_keeps_non_tty_validation(self):
        for argv in ([], ["--headless", "False"]):
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with (mock.patch.object(sys.stdin, "isatty", return_value=False),
                      mock.patch.object(sys.stdout, "isatty", return_value=False),
                      mock.patch.object(auto, "_Session") as session,
                      redirect_stderr(stderr)):
                    self.assertEqual(auto.main(argv), 1)
                session.assert_not_called()
                self.assertIn("--headless", stderr.getvalue())

    def test_headless_main_suppresses_events_on_startup_runtime_and_interrupt(self):
        class Session:
            has_errors = False
            service = SimpleNamespace(base_url="http://127.0.0.1:43210")

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                return self

            def close(self):
                pass

        for failure, expected, board in ((RuntimeError("runtime"), 1, True),
                                         (KeyboardInterrupt(), 130, True)):
            with self.subTest(failure=type(failure).__name__):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (mock.patch.object(auto, "_Session", Session),
                      mock.patch.object(auto, "_headless", side_effect=failure),
                      mock.patch.object(auto, "_print_events",
                                        side_effect=AssertionError("display leaked")),
                      redirect_stdout(stdout), redirect_stderr(stderr)):
                    self.assertEqual(auto.main(["--enable-experimental-worker-board", "--headless",
                                                "--save", "unused"]), expected)
                self.assertEqual(stdout.getvalue().count("Board: "), int(board))
                self.assertNotIn("Save directory", stdout.getvalue())

        class StartupFailure(Session):
            def start(self):
                raise RuntimeError("startup")

        stdout, stderr = io.StringIO(), io.StringIO()
        with (mock.patch.object(auto, "_Session", StartupFailure),
              mock.patch.object(auto, "_print_events",
                                side_effect=AssertionError("display leaked")),
              redirect_stdout(stdout), redirect_stderr(stderr)):
            self.assertEqual(auto.main(["--enable-experimental-worker-board", "--headless",
                                        "--save", "unused"]), 1)
        self.assertEqual(stdout.getvalue(), "")

    def test_codex_auto_flow_rejects_system_wire_messages_like_the_real_endpoint(self):
        # The original auto fixtures covered Chat Completions/Messages, but did
        # not enforce Codex's role restriction on generated default instructions.
        seen = []

        class CodexGateway(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(request)
                if any(item.get("role") == "system" for item in request["input"]):
                    data = b'{"detail":"System messages are not allowed"}'
                    status, content_type = 400, "application/json"
                else:
                    main = request["model"] == "gpt-6-astra"
                    main_calls = sum(r["model"] == "gpt-6-astra" for r in seen)
                    if main and main_calls == 1:
                        item = {"type": "function_call", "name": "board_post_plan",
                                "call_id": "delegate", "arguments": '{"content":"Review the worker task"}'}
                    else:
                        item = {"type": "message", "role": "assistant", "content": [{
                            "type": "output_text", "text": "codex main done" if main else "codex worker done"}]}
                    events = (
                        {"type": "response.output_item.done", "output_index": 0, "item": item},
                        {"type": "response.completed", "response": {
                            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}},
                    )
                    data = b"".join(("data: " + json.dumps(event) + "\n\n").encode() for event in events)
                    status, content_type = 200, "text/event-stream"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        gateway = ThreadingHTTPServer(("127.0.0.1", 0), CodexGateway)
        server_thread = threading.Thread(target=gateway.serve_forever, kwargs={"poll_interval": .01})
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                auth = root / "auth.json"
                auth.write_text(json.dumps({"tokens": {
                    "access_token": "FAKE_CODEX_SECRET", "account_id": "test-account"}}))
                settings = root / "auto.json"
                # The worker's Codex model shares main's route, so it keeps
                # main's gateway and login.
                settings.write_text(json.dumps({
                    "version": 3,
                    "main": {"model_api": "codex", "model": "codex-gpt-6-astra-max",
                             "endpoint_url": f"http://127.0.0.1:{gateway.server_port}/responses",
                             "endpoint_auth": "codex-login",
                             "codex_auth_file": str(auth), "request_timeout_seconds": 3,
                             "cwd": tmp},
                    "worker": {"model": "codex-gpt-5.6-sol-max"},
                }))
                result = subprocess.run([
                    sys.executable, "-m", "pythia.interaction.auto", "--context-config", str(settings),
                    "--enable-experimental-worker-board",
                    "--save", str(root / "run"), "--prompt", "Review auto startup.",
                ], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("codex main done", result.stdout)
                self.assertIn("codex worker done", result.stdout)
                self.assertIn("condition fired", result.stdout)
                self.assertEqual(result.stdout.count("Board: "), 1)
                self.assertIn("/README.md", result.stdout)
                self.assertEqual(len(seen), 3)
                self.assertEqual({r["model"] for r in seen}, {"gpt-6-astra", "gpt-5.6-sol"})
                for request in seen:
                    self.assertEqual(request["input"][0]["role"], "developer")
                    self.assertIn("/README.md", request["input"][0]["content"][0]["text"])
                    self.assertEqual(request["reasoning"]["effort"], "max")
                    self.assertFalse(any(item.get("role") == "system" for item in request["input"]))
                for path in (root / "run").rglob("*"):
                    if path.is_file():
                        self.assertNotIn("FAKE_CODEX_SECRET", path.read_text())
                self.assertNotIn("FAKE_CODEX_SECRET", result.stdout + result.stderr)
        finally:
            gateway.shutdown()
            server_thread.join()
            gateway.server_close()

    @unittest.skipUnless(os.name == "posix", "requires a POSIX pseudo-terminal")
    def test_pty_quiescence_navigation_and_exit_without_model_work(self):
        import fcntl
        import pty
        import select
        import struct
        import termios

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pty-run"
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0))
            process = subprocess.Popen([
                sys.executable, "-m", "pythia.interaction.auto", "--enable-experimental-worker-board",
                "--save", str(path),
                "--endpoint-url", "http://127.0.0.1:1/v1/chat/completions",
                "--request-timeout-seconds", "1",
            ], stdin=slave, stdout=slave, stderr=subprocess.PIPE,
               env={**os.environ, "TERM": "xterm-256color"})
            os.close(slave)
            output = bytearray()
            def expect(marker, after=0):
                deadline = time.monotonic() + 5
                while marker not in output[after:]:
                    if time.monotonic() >= deadline:
                        self.fail(f"PTY did not show {marker!r}: {bytes(output)!r}")
                    if select.select([master], [], [], .05)[0]:
                        try:
                            output.extend(os.read(master, 65536))
                        except OSError:
                            self.fail(f"PTY closed early: {bytes(output)!r}")
            try:
                expect(b"Board: http://127.0.0.1:")
                expect(b"#1 (main) - quiescent")
                start = len(output)
                os.write(master, b"/context -1\r")
                expect(b"#-1 (watcher) - quiescent", start)
                start = len(output)
                os.write(master, b"not a task\r")
                expect(b"Switch to /context 1", start)
                os.write(master, b"\x04")
                self.assertEqual(process.wait(timeout=5), 0)
                self.assertEqual(process.stderr.read(), b"")
                self.assertEqual((path / "index.jsonl").read_text(), "")
                for index in (1, 2, -1):
                    self.assertEqual(len(load_interaction_save(path / "contexts" / f"{index}.jsonl")), 3)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stderr.close()
                os.close(master)

    def test_one_prompt_real_http_adapters_with_separate_apis_and_no_watcher_sample(self):
        seen = []
        class Gateway(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, request))
                if self.path.endswith("/messages"):
                    worker_calls = sum(p.endswith("/messages") for p, _ in seen)
                    content = ([{"type": "tool_use", "id": "work", "name": "exec_command",
                                 "input": {"cmd": "printf proof > proof.txt", "yield_time_ms": 1000}}]
                               if worker_calls == 1 else [{"type": "text", "text": "worker proof"}])
                    payload = {"id": "m", "type": "message", "role": "assistant",
                               "model": "worker", "content": content,
                               "stop_reason": "tool_use" if worker_calls == 1 else "end_turn",
                               "usage": {"input_tokens": 5, "output_tokens": 2}}
                else:
                    main_calls = sum(not p.endswith("/messages") for p, _ in seen)
                    message = ({"role": "assistant", "content": None, "tool_calls": [{
                        "id": "delegate", "type": "function", "function": {
                            "name": "board_post_plan", "arguments": '{"content":"Produce a proof"}'}}]}
                        if main_calls == 1 else {"role": "assistant", "content": "main answer"})
                    payload = {"choices": [{"message": message, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        server_thread = threading.Thread(target=gateway.serve_forever, kwargs={"poll_interval": .01})
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = root / "contexts.json"
                url = f"http://127.0.0.1:{gateway.server_port}"
                settings.write_text(json.dumps({
                    "version": 3, "main": {"model_api": "chat-completions",
                        "endpoint_url": url + "/v1/chat/completions",
                        "model": "main", "request_timeout_seconds": 3, "cwd": tmp},
                    "worker": {"model_api": "messages", "model": "worker",
                               "endpoint_url": url + "/v1/messages",
                               "endpoint_auth": "env:AUTO_TEST_KEY",
                               "max_output_tokens": 128},
                }))
                result = subprocess.run([sys.executable, "-m", "pythia.interaction.auto",
                    "--enable-experimental-worker-board",
                    "--context-config", str(settings), "--save", str(root / "run"), "--prompt", "Do the task"],
                    env={**os.environ, "AUTO_TEST_KEY": "FAKE_PROVIDER_SECRET"},
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("[#1 (main) - assistant] main answer", result.stdout)
                self.assertIn("[#2 (worker) - assistant] worker proof", result.stdout)
                self.assertIn("[#-1 (watcher) - debug]", result.stdout)
                self.assertNotIn("[#1 (main)]", result.stdout)
                self.assertEqual((root / "proof.txt").read_text(), "proof")
                self.assertIn("condition fired", result.stdout)
                self.assertIn("#1 (main)", result.stdout)
                self.assertNotIn("FAKE_PROVIDER_SECRET", result.stdout + result.stderr)
                for file in (root / "run").rglob("*"):
                    if file.is_file():
                        self.assertNotIn("FAKE_PROVIDER_SECRET", file.read_text())
                records = [json.loads(line) for line in (root / "run" / "index.jsonl").read_text().splitlines()]
                self.assertEqual([r["kind"] for r in records].count("user"), 1)
                self.assertEqual([r["kind"] for r in records].count("plan"), 1)
                self.assertEqual(len({r["thread_id"] for r in records}), 1)
                self.assertEqual(len(seen), 4)
                self.assertTrue(any(p.endswith("/messages") for p, _ in seen))
                for path, request in seen:
                    if path.endswith("/messages"):
                        self.assertEqual(request["max_tokens"], 128)
                    else:
                        self.assertNotIn("max_tokens", request)
                        self.assertNotIn("max_completion_tokens", request)
                self.assertEqual(len(load_interaction_save(root / "run" / "contexts" / "-1.jsonl")), 3)
        finally:
            gateway.shutdown()
            server_thread.join()
            gateway.server_close()

    def test_help_documents_resume_without_replay(self):
        result = subprocess.run([sys.executable, "-m", "pythia.interaction.auto", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--context-config", result.stdout)
        self.assertIn("--resume", result.stdout)
        self.assertIn("historical work is not\n                        replayed", result.stdout)


if __name__ == "__main__":
    unittest.main()
