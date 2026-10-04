from __future__ import annotations

import asyncio
from collections import deque
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import DefaultEnvironment
from pythia.interaction import CompactionMetadata
from pythia.interaction import CompactionContextWindowError
from pythia.interaction import CompactionResult
from pythia.interaction import CompactionSettings
from pythia.interaction import ContextPrefix
from pythia.interaction import DisplayItem
from pythia.interaction import Environment
from pythia.interaction import Init
from pythia.interaction import Instructions
from pythia.interaction import Message
from pythia.interaction import InteractionContext
from pythia.interaction import ModelAuthenticationError
from pythia.interaction import ModelFailure
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import SampleParams
from pythia.interaction import SampleMetadata
from pythia.interaction import SaveError
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolOutcome
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import Tools
from pythia.interaction import TokenUsage
from pythia.interaction import TurnSummary
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import UserToolCall
from pythia.interaction import UserToolResult
from pythia.interaction import cli
from pythia.interaction import demo
from pythia.interaction import load_interaction_save
from pythia.interaction import save_interaction_save
from pythia.interaction._cli_editor import Editor
from pythia.interaction._cli_editor import Layout
from pythia.interaction._cli_editor import cell_width
from pythia.interaction._cli_editor import layout_editor
from pythia.interaction._cli_terminal import PosixTerminal


def _answer(text="done"):
    return ModelSample(items=(Message(role="assistant", content=text),))


class _Terminal:
    def __init__(self, on_frame):
        self.on_frame = on_frame
        self.keys = deque()
        self.frames = []
        self.items = []
        self.closed = False
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exited = True

    def key(self, key, data=""):
        self.keys.append(SimpleNamespace(key=key, data=data))

    def submit(self, text):
        self.key("c-u")
        self.key("c-k")
        self.key("<bracketed-paste>", text)
        self.key("c-m", "\r")

    def read_keys(self):
        keys = tuple(self.keys)
        self.keys.clear()
        return keys

    def render(self, editor, status, items, prompt=":> "):
        self.frames.append((editor, status, prompt))
        self.items.extend(items)
        self.on_frame(self, editor, status)


class _Model:
    def __init__(self, path, *outcomes):
        self.path = path
        self.outcomes = deque(outcomes)
        self.calls = []
        self.checkpoints = []
        self.threads = []

    def sample(self, context, *, tools=(), sample_params=None):
        self.calls.append((context.copy(), tuple(tools), sample_params))
        self.checkpoints.append(load_interaction_save(self.path).items)
        self.threads.append(threading.get_ident())
        if not self.outcomes:
            raise AssertionError("unexpected sample")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome(context) if callable(outcome) else outcome


class EditorTests(unittest.TestCase):
    def test_editing_is_value_based_and_keeps_cursor_and_suffix(self):
        initial = Editor("first second", 12)
        left = initial.edit("word-left")
        self.assertEqual(left, Editor("first second", 6))
        self.assertEqual(initial, Editor("first second", 12))
        self.assertEqual(left.edit("c-w"), Editor("second", 0))
        self.assertEqual(left.edit("c-u"), Editor("second", 0))
        self.assertEqual(left.edit("c-k"), Editor("first ", 6))
        self.assertEqual(left.edit("x", "x").edit("c-h"), left)
        self.assertEqual(left.edit("right"), Editor(initial.text, 7))
        self.assertEqual(left.edit("word-right"), initial)
        self.assertEqual(initial.edit("c-a").edit("c-h"), Editor(initial.text, 0))
        self.assertEqual(initial.edit("up"), initial)

    def test_paste_and_multiline_editing_do_not_become_commands(self):
        pasted = Editor().edit("<bracketed-paste>", "/quit\r\n\x03界")
        self.assertEqual(pasted, Editor("/quit\n\x03界", 8))
        self.assertEqual(pasted.edit("c-j"), Editor("/quit\n\x03界\n", 9))

    def test_layout_wraps_and_counts_wide_and_combining_characters(self):
        self.assertEqual(
            layout_editor(Editor("abcd", 4), 7),
            Layout((":> abc", " > d"), 1, 4),
        )
        self.assertEqual(
            layout_editor(Editor("a\n界e\u0301", 5), 8),
            Layout((":> a", " > 界e\u0301"), 1, 6),
        )
        for columns in (1, 2, 3, 8, 80):
            with self.subTest(columns=columns):
                text = "a\t界\n\x1b[2J"
                layout = layout_editor(Editor(text, len(text)), columns)
                self.assertGreaterEqual(layout.cursor_column, 0)
                self.assertLess(layout.cursor_column, columns)
                self.assertNotIn("\x1b", "".join(layout.lines))
                self.assertTrue(all(
                    sum(cell_width(char) for char in line) <= max(1, columns - 1)
                    for line in layout.lines
                ))


class CLIConfigurationTests(unittest.TestCase):
    def test_working_tree_imports_help_and_legacy_target_without_posix_imports(self):
        script = '''
import builtins
import sys
from pathlib import Path

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "termios":
        raise AssertionError("help/import must not load POSIX terminal support")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from pythia.interaction import cli, demo
import pythia.auto as legacy
assert "pythia.auto.cli" not in sys.modules
project = Path(cli.__file__).resolve().parents[2] / "pyproject.toml"
assert 'autopythia = "pythia.auto:_main"' in project.read_text()
assert callable(legacy._main)
for module in (cli, demo):
    try:
        module.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("help did not exit")
'''
        result = subprocess.run([sys.executable, "-c", script],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parser_matches_shared_demo_defaults_and_overrides(self):
        for argv in (
            [],
            ["--endpoint-api", "codex", "--model", "codex-gpt-6-astra", "--resume",
             "--prompt", "/quit\nA literal query", "--instructions", "",
             "--max-samples", "3", "--max-output-tokens", "77", "--cwd", "work",
             "--save", "chosen.jsonl", "--enable-auto-compaction=False",
             "--enable-workspace=False", "--compaction-mode", "pi",
             "--compaction-keep-recent-tokens", "0",
             "--compaction-max-output-tokens", "512"],
        ):
            demo_args = vars(demo._build_parser().parse_args(argv))
            self.assertFalse(demo_args.pop("experimental_user_message_injection"))
            cli_args = vars(cli._build_parser().parse_args(argv))
            self.assertTrue(cli_args.pop("enable_default_tools"))
            self.assertFalse(cli_args.pop("headless"))
            self.assertFalse(cli_args.pop("debug_trace"))
            self.assertIsNone(cli_args.pop("prompt_file"))
            # Same option, different defaults: only the CLI resumes unless told otherwise.
            self.assertTrue(cli_args.pop("resume"))
            self.assertEqual(demo_args.pop("resume"), "--resume" in argv)
            self.assertEqual(cli_args, demo_args)

    def test_resume_is_an_optional_boolean_defaulting_to_true_only_in_the_cli(self):
        for frontend, default in ((cli, True), (demo, False)):
            parser = frontend._build_parser()
            for argv, expected in (([], default), (["--resume"], True), (["--resume=tRuE"], True),
                                   (["--resume=False"], False), (["--resume", "fAlSe"], False)):
                with self.subTest(frontend=frontend.__name__, argv=argv):
                    self.assertIs(parser.parse_args(argv).resume, expected)
            for value in ("", "0", "1", "yes", "no"):
                with self.subTest(frontend=frontend.__name__, value=value):
                    with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                        parser.parse_args(["--resume=" + value])
                    self.assertEqual(error.exception.code, 2)
            help_text = " ".join(parser.format_help().split())
            self.assertIn(
                f"a missing file starts a new one. A bare flag means True (default: {default})",
                help_text,
            )

    def test_context_limit_arguments_validate(self):
        # Context-limit args are positive ints (or unset) and chat-completions
        # accepts them; Messages enforces its compaction minimum.
        for option in ("--auto-compact-tokens", "--max-context-tokens"):
            args = cli._build_parser().parse_args(["--model", "m", option, "0"])
            with self.assertRaisesRegex(ValueError, option.lstrip("-")):
                cli.build_model(args)
        # The minimum binds only Anthropic's server compaction.
        messages = cli._build_parser().parse_args([
            "--endpoint-api", "messages", "--model", "claude-fable-5.1",
            "--auto-compact-tokens", "100", "--compaction-mode", "provider",
        ])
        with self.assertRaisesRegex(ValueError, "at least 50000"):
            cli.build_model(messages)
        pi = cli._build_parser().parse_args([
            "--endpoint-api", "messages", "--model", "claude-fable-5.1",
            "--auto-compact-tokens", "100", "--endpoint-auth", "none",
        ])
        self.assertIsNone(cli.build_model(pi).endpoint.server_compaction)
        for option, value, message in (
            ("--compaction-keep-recent-tokens", "-1", "nonnegative"),
            ("--compaction-max-output-tokens", "0", "positive"),
        ):
            args = cli._build_parser().parse_args(["--model", "m", option, value])
            with self.assertRaisesRegex(ValueError, message):
                cli.build_model(args)
        # A route without provider compaction rejects it before any request.
        for argv in (
            ["--model", "m", "--compaction-mode", "provider"],
            ["--endpoint-api", "codex", "--model", "muse-spark-1.3",
             "--compaction-mode", "provider"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, "no provider compaction"):
                    cli.build_model(cli._build_parser().parse_args(argv))

    def test_enable_arguments_accept_bare_and_explicit_booleans(self):
        cases = (
            ((), True),
            (("--enable-workspace",), True),
            (("--enable-workspace=True",), True),
            (("--enable-workspace=False",), False),
            (("--enable-workspace", "false"), False),
            (("--enable-workspace=tRuE",), True),
            (("--enable-workspace=FaLsE",), False),
        )
        for frontend in (cli, demo):
            options = ("enable-auto-compaction", "enable-workspace")
            if frontend is cli:
                options += ("enable-default-tools",)
            for option in options:
                attribute = option.replace("-", "_")
                for suffix, expected in cases:
                    argv = tuple(
                        value.replace("enable-workspace", option)
                        for value in suffix
                    )
                    with self.subTest(
                        frontend=frontend.__name__,
                        option=option,
                        argv=argv,
                    ):
                        args = frontend._build_parser().parse_args(argv)
                        self.assertIs(getattr(args, attribute), expected)

    def test_enable_arguments_reject_other_boolean_spellings(self):
        for frontend in (cli, demo):
            options = ("enable-auto-compaction", "enable-workspace")
            if frontend is cli:
                options += ("enable-default-tools",)
            for option in options:
                for value in ("", "0", "1", "yes", "no", "enabled"):
                    with self.subTest(
                        frontend=frontend.__name__,
                        option=option,
                        value=value,
                    ):
                        with mock.patch("sys.stderr", new=io.StringIO()):
                            with self.assertRaises(SystemExit) as raised:
                                frontend._build_parser().parse_args([
                                    f"--{option}={value}",
                                ])
                        self.assertEqual(raised.exception.code, 2)

    def test_user_message_experiment_is_demo_only(self):
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli._build_parser().parse_args(["--experimental-user-message-injection"])
        self.assertEqual(raised.exception.code, 2)

    def test_default_tools_flag_is_cli_only(self):
        from pythia.interaction._auto_config import build_parser as auto_parser
        for factory in (demo._build_parser, auto_parser):
            with self.subTest(factory=factory.__module__):
                parser = factory()
                self.assertFalse(hasattr(parser.parse_args([]), "enable_default_tools"))
                with mock.patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parser.parse_args(["--enable-default-tools=False"])
                self.assertEqual(raised.exception.code, 2)

    def test_non_tty_fails_before_model_environment_or_session_effects(self):
        with mock.patch.object(cli.sys, "stdin", io.StringIO()):
            with mock.patch.object(cli, "build_model") as model:
                with mock.patch.object(cli, "DefaultEnvironment") as environment:
                    with mock.patch.object(cli, "save_interaction_save") as save:
                        with mock.patch("builtins.print"):
                            self.assertEqual(cli.main(["--prompt", "hello"]), 1)
        model.assert_not_called()
        environment.assert_not_called()
        save.assert_not_called()

    def test_main_forwards_disabled_workspace_policy(self):
        terminal_stream = SimpleNamespace(isatty=lambda: True)
        context_manager = mock.MagicMock()
        selected_environment = object()
        context_manager.__enter__.return_value = selected_environment

        def skip_run(coroutine):
            coroutine.close()
            return 0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            with mock.patch.object(cli.sys, "stdin", terminal_stream):
                with mock.patch.object(cli.sys, "stdout", terminal_stream):
                    with mock.patch.object(cli, "build_model", return_value=object()):
                        with mock.patch.object(
                            cli,
                            "DefaultEnvironment",
                            return_value=context_manager,
                        ) as environment:
                            with mock.patch.object(cli, "PosixTerminal"):
                                with mock.patch.object(
                                    cli.asyncio,
                                    "run",
                                    side_effect=skip_run,
                                ):
                                    result = cli.main([
                                        "--cwd", str(root),
                                        "--save", str(root / "session.jsonl"),
                                        "--enable-workspace=False",
                                    ])

        self.assertEqual(result, 0)
        environment.assert_called_once_with(
            cwd=root,
            enable_workspace=False,
        )

    def test_main_without_default_tools_does_not_construct_the_runtime(self):
        terminal_stream = SimpleNamespace(isatty=lambda: True)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(cli.sys, "stdin", terminal_stream), \
                    mock.patch.object(cli.sys, "stdout", terminal_stream), \
                    mock.patch.object(cli, "build_model", return_value=object()), \
                    mock.patch.object(cli, "DefaultEnvironment") as default, \
                    mock.patch.object(cli, "PosixTerminal"), \
                    mock.patch.object(cli, "_run", new_callable=mock.AsyncMock, return_value=0) as run:
                self.assertEqual(cli.main([
                    "--save", str(Path(tmpdir) / "session.jsonl"),
                    "--enable-default-tools=False",
                ]), 0)
        default.assert_not_called()
        run.assert_awaited_once()
        environment = run.call_args.args[1]
        self.assertIs(type(environment), Environment)
        self.assertEqual(environment.tool_specs, ())
        self.assertFalse(run.call_args.args[3].enable_default_tools)

    def test_module_help_works_without_a_tty(self):
        result = subprocess.run(
            [sys.executable, "-m", "pythia.interaction.cli", "--help"],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", result.stdout)
        self.assertIn("--prompt", result.stdout)
        self.assertIn("--save PATH", result.stdout)
        self.assertIn("--enable-default-tools", result.stdout)
        self.assertIn("--headless", result.stdout)
        self.assertIn("--prompt-file PATH", result.stdout)

    def test_invalid_initial_options_fail_before_effects_even_with_a_tty(self):
        for argv in (
            ["--prompt", " "],
            ["--max-samples", "0"],
            ["--max-output-tokens", "0"],
        ):
            with self.subTest(argv=argv):
                terminal_stream = SimpleNamespace(isatty=lambda: True)
                with mock.patch.object(cli.sys, "stdin", terminal_stream):
                    with mock.patch.object(cli.sys, "stdout", terminal_stream):
                        with mock.patch.object(cli, "build_model") as model:
                            with mock.patch("builtins.print"):
                                self.assertEqual(cli.main(argv), 1)
                model.assert_not_called()


class TerminalRenderTests(unittest.TestCase):
    def test_content_controls_are_visible_and_decoration_is_applied_once(self):
        class Output(io.StringIO):
            def fileno(self):
                return 1

        output = Output()
        terminal = PosixTerminal(io.StringIO(), output)
        item = DisplayItem("-old\n+new\x1b[2J", is_diff=True)
        with mock.patch("os.get_terminal_size", return_value=os.terminal_size((40, 8))):
            terminal.render(Editor("draft", 5), "tool: name\nextra", (item,))
            rendered = output.getvalue()
            self.assertIn("tool: name extra", rendered)
            self.assertNotIn("\x1b[2J", rendered)
            self.assertIn("\\u001b[2J", rendered)
            self.assertEqual(rendered.count("\x1b[90m⌜"), 1)
            self.assertEqual(rendered.count("\x1b[90m⌞"), 1)
            self.assertIn("\x1b[31m-old", rendered)
            self.assertIn("\x1b[32m+new", rendered)
            terminal.render(Editor("draft", 5), "tool: name\nextra", ())
            self.assertEqual(output.getvalue(), rendered)
        self.assertEqual(item.text, "-old\n+new\x1b[2J")

    def test_prompt_change_redraws_without_mutating_editor_or_cursor(self):
        class Output(io.StringIO):
            def fileno(self):
                return 1

        output = Output()
        terminal = PosixTerminal(io.StringIO(), output)
        editor = Editor("editable draft", 8)
        with mock.patch("os.get_terminal_size", return_value=os.terminal_size((40, 8))):
            terminal.render(editor, "sampling", (), "⠋> ")
            first = output.getvalue()
            terminal.render(editor, "sampling", (), "⠙> ")
        self.assertGreater(len(output.getvalue()), len(first))
        self.assertIn("⠋> editable draft", first)
        self.assertIn("⠙> editable draft", output.getvalue()[len(first):])
        self.assertEqual(editor, Editor("editable draft", 8))


class _ControllerTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"

    async def _run(self, model, terminal, argv=(), environment=None):
        args = cli._build_parser().parse_args(argv)
        result = await asyncio.wait_for(
            cli._run(model, environment or Environment(), terminal, args, self.path),
            timeout=4,
        )
        self.assertTrue(terminal.exited)
        self.assertEqual(
            [context.items for context, _tools, _options in model.calls],
            model.checkpoints,
        )
        self.assertTrue(all(t != threading.get_ident() for t in model.threads))
        return result


class CLIToolsSnapshotTests(_ControllerTestCase):
    def test_tools_update_does_not_hide_completed_manual_compaction(self):
        context = InteractionContext((
            Init("saved"), UserToolCall(ToolCall("compact", "user_1", "{}")),
            UserToolResult(ToolResult("user_1", "compacted")),
            ContextPrefix((Message("user", "summary"),)),
            CompactionMetadata(TokenUsage(), "pi"), Tools(),
        ))
        self.assertIsNone(cli._resume_notice(context))

    async def test_startup_logs_tools_before_sampling(self):
        handler = mock.Mock(return_value=ToolOutcome("unused"))
        environment = Environment((Tool(ToolSpec("current", "Current tool", {}), handler),))
        model = _Model(self.path, _answer())
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], environment), 0)
        context, runtime_tools, _ = model.calls[0]
        self.assertEqual(context.items[1], Tools(environment.tool_specs))
        self.assertEqual(runtime_tools, environment.tool_specs)
        handler.assert_not_called()

    async def test_resume_compares_raw_latest_without_restoring_or_sampling(self):
        handler = mock.Mock(return_value=ToolOutcome("unused"))
        environment = Environment((Tool(ToolSpec("current", "Current tool", {}), handler),))
        current = Tools(environment.tool_specs)
        old = Tools((ToolSpec("obsolete", "Old tool", {}),))
        for previous in (None, current, old, Tools()):
            with self.subTest(previous=previous):
                original = (Init("saved"), *((previous,) if previous is not None else ()),
                            ContextPrefix((Message("assistant", "previous"),)), TurnSummary())
                save_interaction_save(self.path, InteractionContext(original))
                expected = original if previous == current else (*original, current)
                for _ in range(2):
                    model = _Model(self.path)
                    terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
                    self.assertEqual(await self._run(model, terminal, ["--resume"], environment), 0)
                    self.assertEqual(load_interaction_save(self.path).items, expected)
                    self.assertEqual(model.calls, [])
                    self.assertFalse(any("without recorded turn completion" in i.text for i in terminal.items))
                terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
                self.assertEqual(await self._run(_Model(self.path), terminal, ["--resume"]), 0)
                self.assertEqual(load_interaction_save(self.path).items, (*expected, Tools()))
        handler.assert_not_called()

    async def test_snapshot_checkpoint_failure_blocks_sampling(self):
        original = (Init("saved"), Message("assistant", "previous"), TurnSummary())
        save_interaction_save(self.path, InteractionContext(original))
        model = _Model(self.path, _answer())
        terminal = _Terminal(lambda t, e, s: t.submit("/quit") if s == "failed" else None)
        with mock.patch.object(cli, "save_interaction_save", side_effect=OSError("no space")):
            self.assertEqual(await self._run(model, terminal, ["--resume", "--prompt", "hello"]), 1)
        self.assertEqual(model.calls, [])
        self.assertEqual(load_interaction_save(self.path).items, original)


class CLIDisabledToolsTests(_ControllerTestCase):
    async def test_unexpected_default_calls_fail_without_effects_or_tool_warnings(self):
        marker = self.path.parent / "must-not-exist"
        arguments = (
            ("exec_command", {"cmd": f"touch {marker}"}),
            ("write_stdin", {"session_id": 1, "chars": "input"}),
            ("apply_patch", {"patch": f"*** Begin Patch\n*** Add File: {marker}\n+bad\n*** End Patch"}),
            ("update_plan", {"plan": [{"step": "bad", "status": "completed"}]}),
        )
        calls = tuple(ToolCall(name, name, json.dumps(args)) for name, args in arguments)
        model = _Model(self.path, ModelSample(calls), _answer())
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        self.assertEqual(await self._run(model, terminal, [
            "--enable-default-tools=False", "--enable-workspace=False", "--prompt", "hello",
        ]), 0)
        self.assertTrue(all(tools == () for _, tools, _ in model.calls))
        saved = load_interaction_save(self.path)
        results = [item for item in saved if isinstance(item, ToolResult)]
        self.assertEqual([r.call_id for r in results], [c.call_id for c in calls])
        self.assertTrue(all(not r.success and r.output.startswith("Unknown tool:") for r in results))
        self.assertFalse(saved.pending_tool_calls())
        self.assertFalse(marker.exists())
        notices = "\n".join(item.text for item in terminal.items)
        self.assertIn("Default model tools disabled; user commands remain available.", notices)
        self.assertNotIn("runs without a sandbox", notices)
        self.assertNotIn("workspace path restrictions are disabled", notices)

    async def test_resume_keeps_tool_history_and_does_not_replay_pending_calls(self):
        original = (
            Init("old"), ToolCall("update_plan", "old-plan", "{}"),
            ToolResult("old-plan", "historical result"), Message("assistant", "old answer"),
            TurnSummary(sample_count=1),
            ToolCall("exec_command", "pending", '{"cmd":"touch must-not-run"}'),
        )
        save_interaction_save(self.path, InteractionContext(original))
        model = _Model(self.path, _answer("continued"))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        environment = Environment()
        with mock.patch.object(environment, "execute_tool_calls") as execute:
            self.assertEqual(await self._run(model, terminal, [
                "--enable-default-tools=False", "--resume", "--prompt", "continue",
            ], environment), 0)
        execute.assert_not_called()
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:len(original)], original)
        recovered = saved[len(original)]
        self.assertIsInstance(recovered, ToolResult)
        self.assertEqual(recovered.call_id, "pending")
        self.assertFalse(recovered.success)
        self.assertIn("was not rerun", recovered.output)
        self.assertEqual(model.calls[0][1], ())
        self.assertFalse(saved.pending_tool_calls())

    async def test_config_and_both_compaction_paths_keep_empty_model_tools(self):
        save_interaction_save(self.path, InteractionContext((
            Init("old"), Message("assistant", "old answer"),
            SampleMetadata(TokenUsage(total_tokens=100)),
            ModelSampleBoundary(), TurnSummary(sample_count=1, context_tokens=100),
        )))
        model = _Model(self.path, _answer("continued"))
        commands = deque(("/config max_output_tokens 17", "/config.json", "/compact", "/quit"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft())
                             if s == "idle" and commands else None)
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult((
            ContextPrefix((Message("assistant", "summary"),)),
        ))
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, [
                "--enable-default-tools=False", "--resume", "--prompt", "continue",
                "--auto-compact-tokens", "100",
            ]), 0)
        self.assertEqual(compactor.compact.call_count, 2)  # automatic, then /compact
        for call in compactor.compact.call_args_list:
            self.assertEqual(call.kwargs["tools"], ())
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][1], ())
        saved = load_interaction_save(self.path)
        results = [item.result for item in saved if isinstance(item, UserToolResult)]
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.success for result in results))
        config = json.loads(results[1].output)
        self.assertEqual(config["max_output_tokens"], 17)
        self.assertNotIn("enable_default_tools", config)


class CLIControllerTests(_ControllerTestCase):
    async def test_auto_compact_tokens_config_drives_compaction(self):
        save_interaction_save(self.path, InteractionContext((
            Init("old"), Message("assistant", "old answer"),
            SampleMetadata(TokenUsage(total_tokens=100)),
            ModelSampleBoundary(), TurnSummary(sample_count=1, context_tokens=100),
        )))
        model = _Model(self.path, _answer("continued"))
        # The config value drives compaction even though the model exposes no
        # auto-compaction attribute of its own.
        self.assertFalse(hasattr(model, "auto_compact_context_tokens"))
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult((
            ContextPrefix((Message("assistant", "summary"),)),
        ))
        commands = deque(("/quit",))
        terminal = _Terminal(
            lambda t, e, s: t.submit(commands.popleft())
            if s == "idle" and commands else None
        )
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, [
                "--enable-default-tools=False", "--resume",
                "--auto-compact-tokens", "100", "--prompt", "continue",
            ]), 0)
        self.assertEqual(compactor.compact.call_count, 1)
        self.assertEqual(len(model.calls), 1)
    async def test_sampling_spinner_changes_and_returns_to_idle_prompt(self):
        release = threading.Event()
        sampling_prompts = []

        def sample(_context):
            self.assertTrue(release.wait(3))
            return _answer("done")

        def frame(terminal, _editor, status):
            prompt = terminal.frames[-1][2]
            if status.startswith("sampling"):
                sampling_prompts.append(prompt)
                if len(sampling_prompts) == 18:
                    release.set()
            elif status == "idle" and release.is_set():
                self.assertEqual(prompt, ":> ")
                terminal.key("c-d")

        terminal = _Terminal(frame)
        try:
            self.assertEqual(await self._run(
                _Model(self.path, sample), terminal, ["--prompt", "spin"]), 0)
        finally:
            release.set()
        self.assertGreaterEqual(len(set(sampling_prompts)), 2)
        self.assertNotIn(":> ", sampling_prompts)

    async def test_empty_start_keeps_timer_and_never_submits_demo_default(self):
        ticks = 0

        def frame(terminal, editor, status):
            nonlocal ticks
            if status == "idle":
                ticks += 1
                if ticks == 4:
                    terminal.key("c-d")

        model = _Model(self.path)
        terminal = _Terminal(frame)
        with mock.patch.object(cli.asyncio, "sleep", wraps=asyncio.sleep) as sleep:
            self.assertEqual(await self._run(model, terminal), 0)
        self.assertGreaterEqual(sleep.await_count, 4)
        self.assertTrue(all(call.args == (1 / 128,) for call in sleep.await_args_list))
        self.assertEqual(model.calls, [])
        self.assertGreaterEqual(ticks, 4)
        self.assertTrue(all(not editor.text for editor, _, _ in terminal.frames))
        saved = load_interaction_save(self.path)
        self.assertEqual(len(saved.items), 2)
        self.assertIsInstance(saved.items[0], Init)
        self.assertEqual(saved.items[1], Tools())

    async def test_full_input_queue_preserves_unaccepted_draft(self):
        state = cli._UIState(ready=True)
        for index in range(9):
            text = f"query-{index}"
            state.editor = Editor(text, len(text))
            state.handle_key("c-m", "\r")
        self.assertEqual(tuple(state.pending), tuple(f"query-{i}" for i in range(8)))
        self.assertEqual(state.editor, Editor("query-8", 7))
        state.handle_key("c-d", "")
        self.assertTrue(state.closing)
        self.assertEqual(tuple(state.pending), ())

    async def test_initial_literal_query_and_second_turn_each_submit_once(self):
        query = "/quit\nA single multiline query.\n"
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status == "idle":
                step += 1
                terminal.submit("second" if step == 1 else "/quit")

        terminal = _Terminal(frame)
        model = _Model(self.path, _answer("first"), _answer("second"))
        code = await self._run(
            model,
            terminal,
            [
                "--prompt", query,
                "--max-samples", "1",
                "--max-output-tokens", "77",
            ],
        )
        self.assertEqual(code, 0)
        self.assertEqual(terminal.frames[0][0], Editor(query, len(query)))
        self.assertEqual(len(model.calls), 2)
        saved = load_interaction_save(self.path)
        users = tuple(i for i in saved if isinstance(i, Message) and i.role == "user")
        self.assertEqual(users, (Message("user", query), Message("user", "second")))
        self.assertEqual(saved.items.count(UserInteractionBoundary()), 2)
        self.assertEqual(
            tuple(i.sample_count for i in saved if isinstance(i, TurnSummary)), (1, 2)
        )
        self.assertEqual(
            [call[2] for call in model.calls],
            [SampleParams(max_output_tokens=77, enable_auto_compaction=True)] * 2,
        )
        texts = [item.text for item in terminal.items]
        self.assertEqual(texts.count("[assistant] first"), 1)
        self.assertEqual(texts.count("[assistant] second"), 1)

    async def _assert_reasoning_is_visible_but_redacted(self, reasoning, secret):
        terminal = _Terminal(
            lambda terminal, _editor, status: (
                terminal.key("c-d") if status == "idle" else None
            )
        )
        model = _Model(
            self.path,
            ModelSample(
                items=(
                    reasoning,
                    Message(role="assistant", content="done"),
                ),
            ),
        )

        self.assertEqual(
            await self._run(model, terminal, ["--prompt", "inspect"]),
            0,
        )
        displayed = tuple(item.text for item in terminal.items)
        self.assertEqual(
            displayed.count("[reasoning] ..."),
            1,
        )
        self.assertNotIn(secret, "\n".join(displayed))

    async def test_encrypted_only_reasoning_is_visible_but_redacted(self):
        ciphertext = "provider-ciphertext-must-not-be-displayed"
        await self._assert_reasoning_is_visible_but_redacted(
            Reasoning(content="", encrypted_content=ciphertext), ciphertext,
        )

    async def test_signed_thinking_without_text_is_visible_but_redacted(self):
        signature = "thinking-signature-must-not-be-displayed"
        await self._assert_reasoning_is_visible_but_redacted(
            Reasoning(content="", content_signature=signature), signature,
        )

    async def test_unknown_slash_command_is_not_a_query(self):
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status == "idle":
                step += 1
                terminal.submit("/model" if step == 1 else "/exit")

        model = _Model(self.path)
        terminal = _Terminal(frame)
        self.assertEqual(await self._run(model, terminal), 0)
        self.assertEqual(model.calls, [])
        self.assertTrue(any("Unsupported command." in i.text for i in terminal.items))

    async def test_tool_results_are_checkpointed_individually_before_follow_up(self):
        calls = (ToolCall("record", "one", "{}"), ToolCall("record", "two", "{}"))
        checkpoints = []

        def record(arguments, *, timeout_seconds=None):
            checkpoints.append(load_interaction_save(self.path).items)
            return ToolOutcome("recorded")

        environment = Environment((Tool(ToolSpec("record", "", {}), record),))
        terminal = _Terminal(lambda t, e, s: t.submit("/quit") if s == "idle" else None)
        model = _Model(self.path, ModelSample(items=calls), _answer())
        self.assertEqual(await self._run(model, terminal, ["--prompt", "tools"], environment), 0)
        self.assertEqual(len(checkpoints), 2)
        self.assertFalse(any(isinstance(i, ToolResult) for i in checkpoints[0]))
        self.assertEqual(checkpoints[1][-1], ToolResult("one", "recorded"))
        self.assertEqual(model.calls[1][0].items[-2:], (
            ToolResult("one", "recorded"), ToolResult("two", "recorded"),
        ))

    async def test_pending_resume_precedes_empty_override_and_follow_up(self):
        call = ToolCall("missing", "pending", "{}")
        original = (Init("resumed"), Instructions("old"), Message("user", "old"),
                    UserInteractionBoundary(), call, ModelSampleBoundary())
        save_interaction_save(self.path, InteractionContext(original))
        terminal = _Terminal(lambda t, e, s: t.submit("/quit") if s == "idle" else None)
        model = _Model(self.path, _answer())
        self.assertEqual(await self._run(
            model, terminal, ["--resume", "--instructions", "", "--prompt", "follow-up"]
        ), 0)
        received = model.calls[0][0]
        self.assertEqual(received.items[:len(original)], original)
        result = received.items[-5]
        self.assertEqual(received.items[-4], Tools())
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)
        self.assertEqual(result.call_id, "pending")
        self.assertIn("was not rerun", result.output)
        self.assertEqual(received.items[-3:], (
            Instructions(""), Message("user", "follow-up"), UserInteractionBoundary(),
        ))
        self.assertEqual(received.model_items()[0], Instructions(""))

    async def test_resume_without_query_marks_pending_calls_unrecoverable_and_waits(self):
        original = (Init("saved"), Message("user", "original"),
                    UserInteractionBoundary(), ToolCall("missing", "pending", "{}"))
        save_interaction_save(self.path, InteractionContext(original))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:-2], original)
        self.assertEqual(saved.items[-1], Tools())
        self.assertEqual(saved.items[-2].call_id, "pending")
        self.assertFalse(saved.items[-2].success)
        self.assertIn("session restart", saved.items[-2].output)
        self.assertEqual(saved.pending_tool_calls(), ())
        self.assertEqual(model.calls, [])

    async def test_instructions_only_resume_samples_without_a_new_user_message(self):
        original = (Init("saved"), Instructions("old"),
                    Message("assistant", "previous"), TurnSummary(sample_count=1))
        save_interaction_save(self.path, InteractionContext(original))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        model = _Model(self.path, _answer())
        self.assertEqual(await self._run(
            model, terminal, ["--resume", "--instructions", ""]
        ), 0)
        self.assertEqual(model.calls[0][0].items, (*original, Tools(), Instructions("")))

    async def test_new_session_replaces_existing_log_and_missing_resume_keeps_query(self):
        for resume in (False, True):
            with self.subTest(resume=resume):
                if resume:
                    self.path.unlink()
                else:
                    save_interaction_save(self.path, InteractionContext((Init("old"),)))
                terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
                model = _Model(self.path, _answer())
                argv = ["--prompt", "fresh"] + (["--resume"] if resume else ["--resume=False"])
                self.assertEqual(await self._run(model, terminal, argv), 0)
                self.assertIn(
                    "[cli] Warning: exec_command runs without a sandbox; use a trusted model and workspace.",
                    [item.text for item in terminal.items],
                )
                context = model.calls[0][0]
                self.assertNotEqual(context.items[0], Init("old"))
                self.assertEqual(context.items[1:], (
                    Tools(),
                    Message("user", "fresh"), UserInteractionBoundary(),
                ))

    async def test_disabled_workspace_policy_is_warned(self):
        terminal = _Terminal(
            lambda terminal, editor, status: (
                terminal.key("c-d") if status == "idle" else None
            )
        )
        model = _Model(self.path, _answer())

        self.assertEqual(await self._run(
            model,
            terminal,
            ["--prompt", "fresh", "--enable-workspace=False"],
        ), 0)

        self.assertTrue(any(
            "workspace path restrictions are disabled" in item.text
            for item in terminal.items
        ))

    async def test_completed_resume_does_not_repeat_answer_or_summary(self):
        original = (Init("saved"), Tools(), Message("assistant", "previous"),
                    ModelSampleBoundary(), TurnSummary(sample_count=1))
        save_interaction_save(self.path, InteractionContext(original))
        terminal = _Terminal(lambda t, e, s: t.submit("/exit") if s == "idle" else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        self.assertEqual(load_interaction_save(self.path).items, original)
        self.assertEqual([i.text for i in terminal.items].count("[assistant] previous"), 1)
        self.assertEqual(model.calls, [])

    async def test_resume_is_the_default(self):
        original = (Init("saved"), Tools(), Message("assistant", "previous"),
                    ModelSampleBoundary(), TurnSummary(sample_count=1))
        save_interaction_save(self.path, InteractionContext(original))
        terminal = _Terminal(lambda t, e, s: t.submit("/exit") if s == "idle" else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, []), 0)
        self.assertEqual(load_interaction_save(self.path).items, original)
        self.assertEqual([i.text for i in terminal.items].count("[assistant] previous"), 1)
        self.assertEqual(model.calls, [])

    async def test_missing_resume_is_fresh_but_does_not_inject_default_query(self):
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        self.assertEqual(model.calls, [])
        self.assertTrue(any("no existing interaction.jsonl" in i.text for i in terminal.items))

    async def test_paused_messages_compaction_continues(self):
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        checkpoint = OpaqueCompaction.from_messages("summary")
        model = _Model(self.path, ModelSample(items=(checkpoint,), stop_reason="compaction"), _answer())
        with mock.patch.object(
            cli.time,
            "perf_counter",
            side_effect=(50.0, 57.25),
        ):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        self.assertIn(checkpoint, model.calls[1][0].items)
        self.assertEqual(
            load_interaction_save(self.path).items[-1],
            TurnSummary(
                sample_count=2,
                compaction_count=1,
                elapsed_seconds=7.25,
            ),
        )

    async def test_config_threshold_auto_compacts_before_sampling(self):
        original = (
            Init("saved"),
            Message("user", "old request"),
            UserInteractionBoundary(),
            Message("assistant", "old answer"),
            SampleMetadata(TokenUsage(90, 10, 100, 20)),
            ModelSampleBoundary(),
            TurnSummary(
                input_tokens_sum=90,
                output_tokens_sum=10,
                cached_input_tokens_sum=20,
                cached_input_tokens_max=20,
                non_cached_input_tokens_sum=70,
                context_tokens=100,
                sample_count=1,
            ),
        )
        save_interaction_save(self.path, InteractionContext(original))
        model = _Model(self.path, _answer("after auto compact"))
        compacted = []

        class Compactor:
            def compact(inner_self, source, *, tools=(), sample_params=None, instructions=None):
                compacted.append((source.items, tuple(tools), sample_params, instructions))
                return CompactionResult(
                    (ContextPrefix((Message("user", "follow up"),)),),
                    usage=TokenUsage(100, 5, 105, 50),
                    protocol="responses_compaction_v2",
                    elapsed_seconds=3.0,
                )

        terminal = _Terminal(
            lambda t, e, s: t.key("c-d") if s == "idle" else None
        )
        with mock.patch.object(
            cli,
            "create_default_compactor",
            return_value=Compactor(),
        ) as create:
            self.assertEqual(
                await self._run(
                    model,
                    terminal,
                    ["--resume", "--prompt", "follow up", "--auto-compact-tokens", "100"],
                ),
                0,
            )

        create.assert_called_once_with(model, CompactionSettings())
        self.assertEqual(len(compacted), 1)
        self.assertIn(Message("user", "follow up"), compacted[0][0])
        # The turn's params, without focus text.
        self.assertEqual(compacted[0][2], model.calls[0][2])
        self.assertEqual(compacted[0][2], SampleParams(
            enable_auto_compaction=True, auto_compact_tokens=100,
        ))
        self.assertIsNone(compacted[0][3])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0].model_items(), (
            Message("user", "follow up"),
        ))
        saved = load_interaction_save(self.path)
        self.assertEqual(
            len([item for item in saved if isinstance(item, CompactionMetadata)]),
            1,
        )
        self.assertFalse(any(
            isinstance(item, UserToolCall) and item.call.name == "compact"
            for item in saved
        ))
        summary = saved.items[-1]
        self.assertIsInstance(summary, TurnSummary)
        self.assertEqual(summary.input_tokens_sum, 190)
        self.assertEqual(summary.output_tokens_sum, 15)
        self.assertEqual(summary.cached_input_tokens_sum, 70)
        self.assertEqual(summary.non_cached_input_tokens_sum, 120)
        self.assertEqual(summary.context_tokens, 0)
        self.assertEqual(summary.sample_count, 2)
        self.assertEqual(summary.compaction_count, 1)
        transcript = "\n".join(item.text for item in terminal.items)
        self.assertIn("[context prefix]", transcript)
        self.assertIn(
            "[compaction] protocol=responses_compaction_v2",
            transcript,
        )

    async def test_auto_compaction_can_be_disabled(self):
        original = (
            Init("saved"),
            Message("assistant", "uncompacted answer"),
            SampleMetadata(TokenUsage(total_tokens=100)),
            ModelSampleBoundary(),
            TurnSummary(sample_count=1, context_tokens=100),
        )
        save_interaction_save(self.path, InteractionContext(original))
        model = _Model(self.path, _answer("done"))
        terminal = _Terminal(
            lambda t, e, s: t.key("c-d") if s == "idle" else None
        )

        with mock.patch.object(cli, "create_default_compactor") as create:
            self.assertEqual(
                await self._run(
                    model,
                    terminal,
                    [
                        "--resume",
                        "--prompt",
                        "follow up",
                        "--enable-auto-compaction=False",
                        "--auto-compact-tokens", "100",
                    ],
                ),
                0,
            )

        create.assert_not_called()
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            model.calls[0][2],
            SampleParams(enable_auto_compaction=False, auto_compact_tokens=100),
        )
        self.assertIn(
            Message("assistant", "uncompacted answer"),
            model.calls[0][0].model_items(),
        )
        self.assertFalse(any(
            isinstance(item, CompactionMetadata)
            for item in load_interaction_save(self.path)
        ))

    async def test_failure_after_tool_keeps_checkpoint_and_tui_alive(self):
        called = []

        def record(arguments, *, timeout_seconds=None):
            called.append(True)
            return ToolOutcome("effect done")

        environment = Environment((Tool(ToolSpec("record", "", {}), record),))
        terminal = _Terminal(lambda t, e, s: t.key("c-c") if s == "failed" else None)
        model = _Model(self.path, ModelSample(items=(ToolCall("record", "one", "{}"),)), RuntimeError("sample failed\n"))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], environment), 1)
        self.assertEqual(called, [True])
        self.assertEqual(load_interaction_save(self.path).items[-1], ToolResult("one", "effect done"))
        self.assertTrue(any("RuntimeError: sample failed" in i.text for i in terminal.items))

    async def test_queue_does_not_mutate_active_request(self):
        entered, release = threading.Event(), threading.Event()

        def first(context):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release model")
            return _answer("first")

        queued = False

        def frame(t, editor, status):
            nonlocal queued
            if entered.is_set() and not queued:
                queued = True
                t.submit("queued")
            elif "queued=1" in status:
                release.set()
            if status == "idle" and any(i.text == "[assistant] second" for i in t.items):
                t.key("c-d")

        terminal = _Terminal(frame)
        model = _Model(self.path, first, _answer("second"))
        try:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "first"]), 0)
        finally:
            release.set()
        self.assertEqual(len(model.calls), 2)
        self.assertNotIn(Message("user", "queued"), model.calls[0][0].items)
        self.assertIn(Message("user", "queued"), model.calls[1][0].items)

    async def test_quit_drains_sample_without_executing_returned_tools(self):
        entered, release = threading.Event(), threading.Event()
        call = ToolCall("must_not_run", "one", "{}")

        def sample(context):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release model")
            return ModelSample(items=(call,))

        def frame(t, editor, status):
            if entered.is_set():
                t.key("c-c")
            if status.startswith("closing"):
                release.set()

        terminal = _Terminal(frame)
        model = _Model(self.path, sample)
        try:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        finally:
            release.set()
        self.assertEqual(load_interaction_save(self.path).pending_tool_calls(), (call,))

    async def test_quit_during_tool_checkpoints_its_result_and_skips_remaining_calls(self):
        entered, release = threading.Event(), threading.Event()
        executions = []

        def record(arguments, *, timeout_seconds=None):
            executions.append(True)
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release tool")
            return ToolOutcome("completed before exit")

        def frame(t, editor, status):
            if entered.is_set():
                t.key("c-d")
            if status.startswith("closing"):
                release.set()

        calls = (ToolCall("record", "one", "{}"), ToolCall("record", "two", "{}"))
        environment = Environment((Tool(ToolSpec("record", "", {}), record),))
        model = _Model(self.path, ModelSample(items=calls))
        try:
            self.assertEqual(await self._run(
                model, _Terminal(frame), ["--prompt", "hello"], environment
            ), 0)
        finally:
            release.set()
        saved = load_interaction_save(self.path)
        self.assertEqual(executions, [True])
        self.assertEqual(saved.items[-1], ToolResult("one", "completed before exit"))
        self.assertEqual(saved.pending_tool_calls(), (calls[1],))

    async def test_command_session_survives_across_user_turns(self):
        step = 0

        def frame(t, editor, status):
            nonlocal step
            if status == "idle":
                step += 1
                t.submit("continue" if step == 1 else "/quit")

        start = ToolCall(
            "exec_command", "start",
            json.dumps({
                "cmd": "read line; printf '%s' \"$line\"",
                "yield_time_ms": 1,
            }),
        )
        write = ToolCall(
            "write_stdin", "write",
            '{"session_id":1,"chars":"persistent\\n","yield_time_ms":1000}',
        )
        model = _Model(self.path, ModelSample(items=(start,)), _answer("first"),
                       ModelSample(items=(write,)), _answer("second"))
        with DefaultEnvironment(cwd=self.path.parent) as environment:
            self.assertEqual(await self._run(
                model, _Terminal(frame), ["--prompt", "start"], environment
            ), 0)
            result = model.calls[-1][0].items[-1]
            self.assertIsInstance(result, ToolResult)
            self.assertTrue(result.success)
            self.assertIn("persistent", result.output)

    async def test_failed_checkpoint_blocks_follow_up_and_new_queries(self):
        real_save = save_interaction_save

        def fail_sample_save(path, context):
            if any(isinstance(i, ModelSampleBoundary) for i in context):
                raise SaveError("disk unavailable")
            real_save(path, context)

        step = 0

        def frame(t, editor, status):
            nonlocal step
            if status == "failed":
                step += 1
                t.submit("must not run" if step == 1 else "/quit")

        terminal = _Terminal(frame)
        model = _Model(self.path, _answer())
        with mock.patch.object(cli, "save_interaction_save", side_effect=fail_sample_save):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(load_interaction_save(self.path).items[-1], UserInteractionBoundary())


class AuthNoticeTests(unittest.TestCase):
    def test_notice_follows_the_rejected_credential_source(self):
        for source, expected in (
            ("environment", "Environment credential rejected; update it and restart the process."),
            ("static", "Configured static credential rejected; restart with updated credentials."),
            ("none", "Endpoint rejected anonymous access; restart with "
                     "--endpoint-auth env:NAME or supplied."),
            ("codex_file", "Model authentication needed; use /login."),
        ):
            with self.subTest(source=source):
                state = cli._UIState()
                failure = ModelFailure(
                    category="authentication", message="rejected", provider="api",
                    model="wire", auth_source=source,
                )
                cli._mark_auth_required(
                    state, ModelAuthenticationError("rejected", failure=failure),
                )
                self.assertTrue(state.auth_required)
                self.assertEqual(state.auth_notice, expected)


if __name__ == "__main__":
    unittest.main()
