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
from pythia.interaction import ToolOutcome, ToolSpec, ToolResult, TurnSummary
from pythia.interaction import ModelFailure, ModelTransportError, Reasoning, OpaqueCompaction
from pythia.interaction import load_interaction_save
from pythia.interaction import auto
from pythia.interaction import SampleParams
from pythia.interaction._auto_board import BoardError
from pythia.interaction._auto_config import build_parser, namespace, resolve_config
from pythia.interaction.messages import resolve_messages_max_output_tokens
from pythia.interaction.runtime_config import InteractionConfig


def answer(text="done"):
    return ModelSample((Message("assistant", text),))


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
        session = SimpleNamespace(status=lambda index: statuses[index])
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

    def test_saved_config_statically_validates_and_normalizes_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("relative-cwd", "relative-home", "auth-parent"):
                (root / directory).mkdir()
            snapshot = resolve_config(overrides={"cwd": tmp})
            snapshot[1].update({
                "model_api": "codex", "model": "saved-codex",
                "cwd": "relative-cwd", "codex_home": "relative-home",
            })
            snapshot[2].update({
                "model_api": "codex", "model": "saved-codex-worker",
                "cwd": "relative-cwd", "codex_auth_file": "auth-parent/auth.json",
            })
            path = root / "config.json"

            def write():
                path.write_text(json.dumps({
                    "version": 1,
                    "contexts": {str(index): value for index, value in snapshot.items()},
                }))

            write()
            saved = auto.load_saved_config(path)
            self.assertEqual(saved[1]["cwd"], str((root / "relative-cwd").absolute()))
            self.assertEqual(saved[1]["codex_home"], str((root / "relative-home").absolute()))
            self.assertEqual(saved[2]["codex_auth_file"],
                             str((root / "auth-parent/auth.json").absolute()))
            merged = resolve_config(saved=saved)
            self.assertEqual(merged[1]["codex_home"], saved[1]["codex_home"])

            for key in ("cwd", "codex_home", "codex_auth_file"):
                original = snapshot[1][key]
                for invalid in (123, "", "bad\x00path"):
                    with self.subTest(key=key, invalid=invalid):
                        snapshot[1][key] = invalid
                        write()
                        with self.assertRaises(ValueError):
                            auto.load_saved_config(path)
                snapshot[1][key] = original

    def test_saved_settings_are_base_for_explicit_resume_overrides(self):
        saved = resolve_config(overrides={"model": "saved-model", "cwd": str(Path.cwd())})
        saved[2]["name"] = "saved worker"
        saved[2]["instructions"] = "saved custom worker instructions"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "override.json"
            path.write_text(json.dumps({
                "version": 1,
                "defaults": {"max_samples": 7},
                "contexts": {"2": {"model": "role-model", "name": "new worker"}},
            }))
            settings = resolve_config(path, {"model": "launch-model"}, saved=saved)
        self.assertEqual(settings[1]["model"], "launch-model")
        self.assertEqual(settings[2]["model"], "role-model")
        self.assertEqual(settings[-1]["model"], "launch-model")
        self.assertTrue(all(value["max_samples"] == 7 for value in settings.values()))
        self.assertEqual(settings[2]["name"], "new worker")
        self.assertEqual(settings[2]["instructions"], "saved custom worker instructions")

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

    def test_explicit_limits_and_per_context_null_overrides(self):
        args = build_parser().parse_args(["--max-samples", "3", "--max-output-tokens", "128"])
        overrides = {"max_samples": args.max_samples, "max_output_tokens": args.max_output_tokens}
        for settings in resolve_config(overrides=overrides).values():
            self.assertEqual(settings["max_samples"], 3)
            self.assertEqual(settings["max_output_tokens"], 128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({
                "version": 1,
                "defaults": {"max_samples": 5, "max_output_tokens": 512},
                "contexts": {"2": {"max_samples": None, "max_output_tokens": None}},
            }))
            settings = resolve_config(path, overrides)
            self.assertEqual(settings[1]["max_samples"], 3)
            self.assertEqual(settings[1]["max_output_tokens"], 128)
            self.assertIsNone(settings[2]["max_samples"])
            self.assertIsNone(settings[2]["max_output_tokens"])
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

    def test_file_overrides_launch_defaults_and_resets_cross_api_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({
                "version": 1,
                "defaults": {"model_api": "codex", "model": "file-main", "codex_auth_file": "auth.json"},
                "contexts": {"2": {"model_api": "messages", "model": "worker",
                                       "endpoint_auth": "env:WORKER_KEY",
                                       "max_output_tokens": 64}},
            }))
            parsed = build_parser().parse_args(["--context-config", str(path), "--prompt", "hi"])
            self.assertFalse(hasattr(parsed, "model_api"))
            settings = resolve_config(path, {"model": "launch-main"})
            self.assertEqual(settings[1]["model"], "launch-main")
            self.assertEqual(settings[-1]["model"], "launch-main")
            self.assertEqual(settings[1]["codex_auth_file"], str(Path(tmp) / "auth.json"))
            self.assertEqual(settings[2]["model_api"], "messages")
            self.assertEqual(settings[2]["model"], "worker")
            self.assertIsNone(settings[2]["codex_auth_file"])
            self.assertEqual(settings[2]["endpoint_auth"], "env:WORKER_KEY")
            self.assertEqual([settings[i]["name"] for i in (1, 2, -1)], ["main", "worker", "watcher"])

    def test_config_validation_and_instruction_scope(self):
        settings = resolve_config(overrides={"instructions": ""})
        self.assertEqual(settings[1]["instructions"], "")
        self.assertIsNone(settings[2]["instructions"])
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
            for content in ('{"version":1,"contexts":{"0":{}}}',
                            '{"version":1,"version":1}',
                            '{"version":1,"contexts":{"-1":{"name":"bad\\nname"}}}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    resolve_config(path)


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

    def session(self, scripts, extra_tools=(), *, settings_overrides=None,
                settings_updates=None, **kwargs):
        test = self
        class Model:
            def __init__(self, index):
                self.index = index
                self.outcomes = deque(scripts.get(index, ()))
                test.threads[index].append(threading.get_ident())

            def sample(self, context, **params):
                test.threads[self.index].append(threading.get_ident())
                test.calls[self.index].append(context.copy())
                test.options[self.index].append(params.get("sample_params"))
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
            overrides={"cwd": self.temp.name, **(settings_overrides or {})}, saved=saved
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

    def test_startup_context_summaries_share_one_global_display_item(self):
        session = self.session({}, settings_updates={
            1: {"name": "lead custom"},
            2: {"name": "builder custom", "model_api": "messages", "model": "worker-model",
                "max_output_tokens": 128},
            -1: {"name": "observer custom", "model": "watch-model"},
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
        self.assertEqual(summary.splitlines(), [
            "#1 (lead custom): chat-completions / (server default)",
            "#2 (builder custom): messages / worker-model",
            "#-1 (observer custom): chat-completions / watch-model",
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
        for phase in ("starting", "sampling", "compacting", "executing tools", "saving"):
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
            self.assertEqual(len(context), 2)
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
        self.assertEqual(len(load_interaction_save(self.path / "contexts" / "-1.jsonl")), 2)
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
                               environment_factory=lambda i, a, t: Env(i))
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
        settings = resolve_config(self.path / "config.json")
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            auto._Session(self.path, settings, resume=True,
                          model_factory=lambda *_: self.fail("model initialized"),
                          environment_factory=lambda *_: self.fail("environment initialized")).start()
        self.assertEqual((self.path / "config.json").read_bytes(), config)
        session.close()
        context_path = self.path / "contexts" / "2.jsonl"
        context_path.write_text("broken\n")
        broken = context_path.read_bytes()
        with self.assertRaises(Exception):
            auto._Session(self.path, settings, resume=True).start()
        self.assertEqual(context_path.read_bytes(), broken)

    def test_resume_lock_cannot_be_bypassed_by_directory_symlink(self):
        session = self.session({})
        alias = self.path.parent / "alias"
        alias.symlink_to(self.path, target_is_directory=True)
        settings = resolve_config(self.path / "config.json")
        before = {path.relative_to(self.path): path.read_bytes()
                  for path in self.path.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            auto._Session(alias, settings, resume=True,
                          model_factory=lambda *_: self.fail("model initialized"),
                          environment_factory=lambda *_: self.fail("environment initialized")).start()
        self.assertEqual({path.relative_to(self.path): path.read_bytes()
                          for path in self.path.rglob("*") if path.is_file()}, before)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(auto.main([
                "--resume", "--headless", "--save", str(alias)
            ]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("auto failed: Auto save directory is already in use.",
                      stderr.getvalue())
        session.close()
        resumed = auto._Session(alias, settings, resume=True).start()
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
        settings = resolve_config(self.path / "config.json")
        with self.assertRaisesRegex(ValueError, "initialization metadata"):
            auto._Session(self.path, settings, resume=True,
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
        resumed = auto._Session(self.path, settings, resume=True).start()
        try:
            for index in (1, 2, -1):
                context = load_interaction_save(self.path / "contexts" / f"{index}.jsonl")
                self.assertEqual(context.items[:len(prefixes[index])], prefixes[index])
        finally:
            resumed.close()
        updated = auto.load_saved_config(self.path / "config.json")
        again = resolve_config(saved=updated)
        self.assertTrue(all(value["cwd"] == str(new.absolute()) for value in again.values()))
        repeated = auto._Session(self.path, again, resume=True).start()
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


class EntryPointTests(unittest.TestCase):
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
            for argv, expected, warned in ((["--headless", "--enable-board-auth", "False"], False, True),
                                           (["--headless"], True, False),
                                           (["--headless", "--enable-board-auth"], True, False),
                                           (["--headless", "--resume"], True, False)):
                with self.subTest(argv=argv):
                    self.assertFalse(missing.exists())
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (mock.patch.object(auto, "_Session", Session),
                          mock.patch.object(auto, "_headless", return_value=0),
                          redirect_stdout(stdout), redirect_stderr(stderr)):
                        self.assertEqual(auto.main([*argv, "--save", str(missing)]), 0)
                    self.assertIs(created[-1]["enable_board_auth"], expected)
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
                argv = ["--headless", "--save", "unused"]
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
                "--headless", "False", "--save", "unused", "--prompt", "task"
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
                    self.assertEqual(auto.main(["--headless", "--save", "unused"]), expected)
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
            self.assertEqual(auto.main(["--headless", "--save", "unused"]), 1)
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
                settings.write_text(json.dumps({
                    "version": 1,
                    "defaults": {"model_api": "codex", "model": "codex-gpt-6-astra-max",
                                 "endpoint_url": f"http://127.0.0.1:{gateway.server_port}/responses",
                                 "endpoint_auth": "codex-login",
                                 "codex_auth_file": str(auth), "request_timeout_seconds": 3,
                                 "cwd": tmp},
                    "contexts": {"2": {"model": "codex-gpt-5.6-sol-max"}},
                }))
                result = subprocess.run([
                    sys.executable, "-m", "pythia.interaction.auto", "--context-config", str(settings),
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
                sys.executable, "-m", "pythia.interaction.auto", "--save", str(path),
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
                    self.assertEqual(len(load_interaction_save(path / "contexts" / f"{index}.jsonl")), 2)
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
                    "version": 1, "defaults": {"model_api": "chat-completions",
                        "endpoint_url": url + "/v1/chat/completions",
                        "model": "main", "request_timeout_seconds": 3, "cwd": tmp},
                    "contexts": {"2": {"model_api": "messages", "model": "worker",
                                           "endpoint_url": url + "/v1/messages",
                                           "endpoint_auth": "env:AUTO_TEST_KEY",
                                           "max_output_tokens": 128}},
                }))
                result = subprocess.run([sys.executable, "-m", "pythia.interaction.auto",
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
                self.assertEqual(len(load_interaction_save(root / "run" / "contexts" / "-1.jsonl")), 2)
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
