"""One-shot CLI runs without terminal input, preserving the CLI's save policy."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from pythia.interaction import (
    Environment, Init, Instructions, InteractionContext, Message, ModelFailure,
    ModelResponseError, ModelSample, ModelSampleBoundary, ToolCall, ToolResult, Tools,
    TurnSummary, cli, demo, load_interaction_save, save_interaction_save,
)
from pythia.interaction.codex_auth import CodexAuthUnavailable
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer
from pythia_test.test_interaction_cli_pty import _SCRIPT


class NoInput:
    def read(self, *args):
        raise AssertionError("headless CLI read stdin")

    def isatty(self):
        raise AssertionError("headless CLI checked stdin TTY")


class HeadlessCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "session.jsonl"

    def run_main(self, argv, *outcomes, factory=None):
        model = _Model(self.path, *outcomes)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(cli, "build_model", side_effect=factory or (lambda args: model)), \
                mock.patch.object(cli, "PosixTerminal", side_effect=AssertionError("TUI constructed")), \
                mock.patch.object(cli.sys, "stdin", NoInput()), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["--headless", "--cwd", str(self.root), "--save", str(self.path), *argv])
        self.assertEqual([context.items for context, _, _ in model.calls], model.checkpoints)
        return code, model, stdout.getvalue(), stderr.getvalue()

    def test_optional_boolean_and_demo_scope(self):
        for argv, expected in (([], False), (["--headless"], True),
                               (["--headless=tRuE"], True), (["--headless", "fAlSe"], False)):
            self.assertIs(cli._build_parser().parse_args(argv).headless, expected)
        for value in ("", "0", "1", "yes", "no"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli._build_parser().parse_args(["--headless=" + value])
        self.assertFalse(hasattr(demo._build_parser().parse_args([]), "headless"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            demo._build_parser().parse_args(["--headless"])

    def test_headless_false_still_requires_tty(self):
        for argv in ([], ["--headless=False"]):
            with mock.patch.object(cli.sys, "stdin", io.StringIO()), \
                    mock.patch.object(cli, "build_model") as model, redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--prompt", "hello", *argv]), 1)
            model.assert_not_called()

    def test_no_explicit_work_fails_before_effects_or_save_changes(self):
        for argv in ([], ["--resume"], ["--resume=False", "--instructions", "new instructions"]):
            with self.subTest(argv=argv):
                self.path.write_bytes(b"keep this save\n")
                with mock.patch.object(cli, "build_model") as model, \
                        mock.patch.object(cli, "DefaultEnvironment") as environment, \
                        redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(["--headless", "--save", str(self.path), *argv]), 1)
                model.assert_not_called()
                environment.assert_not_called()
                self.assertEqual(self.path.read_bytes(), b"keep this save\n")
        self.path.unlink()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main([
                "--headless", "--save", str(self.path), "--resume", "--instructions", "",
            ]), 1)
        self.assertFalse(self.path.exists())

    def test_prompt_tools_and_answer_are_saved_without_context_display(self):
        call = ToolCall("update_plan", "plan", '{"plan":[{"step":"work","status":"completed"}]}')
        code, model, stdout, stderr = self.run_main(
            ["--prompt", "PRIVATE_PROMPT"], ModelSample((call,)), _answer("PRIVATE_ANSWER"),
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual({spec.name for spec in model.calls[0][1]},
                         {"exec_command", "write_stdin", "apply_patch", "update_plan"})
        saved = load_interaction_save(self.path)
        self.assertIn(Message("user", "PRIVATE_PROMPT"), saved)
        self.assertIn(Message("assistant", "PRIVATE_ANSWER"), saved)
        self.assertTrue(next(item for item in saved if isinstance(item, ToolResult)).success)
        self.assertIsInstance(saved[-1], TurnSummary)
        self.assertEqual(stdout, "")
        self.assertNotIn("PRIVATE_", stderr)
        self.assertNotIn("\x1b", stderr)
        self.assertIn(f"Save log: {self.path}", stderr)

    def test_prompt_file_is_loaded_once_before_model_construction(self):
        source = self.root / "task.md"
        prompt = " \t/retry\r\nLiteral café prompt\n\n"
        source.write_bytes(prompt.encode("utf-8"))
        model = _Model(self.path, _answer())

        def factory(args):
            self.assertEqual(args.prompt, prompt.rstrip())
            source.write_text("must not replace loaded text")
            return model

        code, _, _, stderr = self.run_main(
            ["--prompt-file", str(source), "--enable-default-tools=False"], factory=factory,
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(model.calls[0][1], ())
        self.assertEqual([i for i in load_interaction_save(self.path) if isinstance(i, Message) and i.role == "user"],
                         [Message("user", prompt.rstrip())])

    def test_sample_failure_saves_partial_calls_without_execution_or_retry_wait(self):
        marker = self.root / "must-not-run"
        partial = ToolCall("exec_command", "partial", json.dumps({"cmd": f"touch {marker}"}))
        failure = ModelFailure(category="stream_closed", message="stream failed")
        error = ModelResponseError("stream failed", failure=failure, completed_items=(partial,))
        code, model, stdout, stderr = self.run_main(["--prompt", "hello"], error)
        self.assertEqual(code, 1)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(stdout, "")
        self.assertIn("ModelResponseError: stream failed", stderr)
        self.assertNotIn("Use /retry", stderr)
        self.assertFalse(marker.exists())
        saved = load_interaction_save(self.path)
        self.assertIn(failure, saved)
        self.assertFalse(saved.pending_tool_calls())
        self.assertFalse(saved[-1].success)

    def test_save_failure_exits_without_waiting_for_quit(self):
        self.path.write_bytes(b"old save\n")
        with mock.patch.object(cli, "save_interaction_save", side_effect=OSError("disk failed")):
            code, model, _, stderr = self.run_main(["--resume=False", "--prompt", "hello"])
        self.assertEqual(code, 1)
        self.assertEqual(model.calls, [])
        self.assertIn("No further work will run", stderr)
        self.assertNotIn("Use /quit", stderr)
        self.assertEqual(self.path.read_bytes(), b"old save\n")

    def test_missing_auth_is_fail_fast_without_creating_a_save_or_login_shell(self):
        with mock.patch.object(cli, "DefaultEnvironment") as environment:
            code, _, _, stderr = self.run_main(
                ["--prompt", "hello", "--endpoint-api", "codex", "--model", "test"],
                factory=mock.Mock(side_effect=CodexAuthUnavailable("missing credentials")),
            )
        self.assertEqual(code, 1)
        self.assertIn("missing credentials", stderr)
        environment.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_resume_closes_pending_calls_then_runs_one_new_prompt(self):
        old = (Init("old"), ToolCall("exec_command", "pending", '{"cmd":"touch must-not-run"}'),
               ModelSampleBoundary())
        save_interaction_save(self.path, InteractionContext(old))
        code, model, _, stderr = self.run_main(["--resume", "--prompt", "continue"], _answer())
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(model.calls), 1)
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:len(old)], old)
        self.assertFalse(saved[len(old)].success)
        self.assertIn("was not rerun", saved[len(old)].output)
        self.assertFalse((self.root / "must-not-run").exists())

    def test_instructions_only_continuation_requires_an_existing_save(self):
        old = (Init("old"), Message("user", "old task"), Message("assistant", "old answer"), TurnSummary())
        for resume in (["--resume"], []):  # resume is the default
            with self.subTest(resume=resume):
                save_interaction_save(self.path, InteractionContext(old))
                code, model, _, stderr = self.run_main([*resume, "--instructions", ""], _answer())
                self.assertEqual(code, 0, stderr)
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(model.calls[0][0].items, (*old, Tools(model.calls[0][1]), Instructions("")))

    @unittest.skipUnless(os.name == "posix", "interactive CLI requires POSIX")
    def test_interactive_prompt_file_preloads_editor_and_submits_literal_text(self):
        class TTY(io.StringIO):
            def isatty(self):
                return True
        source = self.root / "task.md"
        text = "/quota\nThis is a literal task.\n"
        source.write_text(text)
        model = _Model(self.path, _answer())
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(cli.sys, "stdin", TTY()), mock.patch.object(cli.sys, "stdout", TTY()), \
                mock.patch.object(cli, "build_model", return_value=model), \
                mock.patch.object(cli, "PosixTerminal", return_value=terminal):
            self.assertEqual(cli.main([
                "--prompt-file", str(source), "--save", str(self.path), "--enable-default-tools=False",
            ]), 0)
        self.assertEqual(terminal.frames[0][0].text, text.rstrip())
        self.assertEqual(len(model.calls), 1)
        self.assertIn(Message("user", text.rstrip()), model.calls[0][0])

    def test_real_non_tty_entrypoint_success_failure_and_cleanup(self):
        source = self.root / "task.md"
        source.write_text("headless task\n")
        root = str(Path(__file__).resolve().parents[1])
        for failure, expected in (("", 0), ("model", 1), ("save", 1)):
            with self.subTest(failure=failure):
                result = subprocess.run([
                    sys.executable, "-c", _SCRIPT, "--headless", "--prompt-file", str(source),
                ], cwd=self.root, env={**os.environ, "PYTHONPATH": root, "PYTHIA_TEST_FAILURE": failure},
                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("\x1b", result.stderr)
                self.assertNotIn("Use /retry", result.stderr)
                cleanup = json.loads((self.root / "cleanup.json").read_text())
                self.assertEqual(cleanup["active_after"], 0)
                self.assertTrue(cleanup["terminated"])
                self.assertTrue(cleanup["pipes_closed"])

    @unittest.skipUnless(os.name == "posix", "uses POSIX signals and inherited pipe FDs")
    def test_sigint_drains_a_request_and_returns_130_without_executing_its_calls(self):
        script = _SCRIPT.replace(
            'os.read(int(os.environ["PYTHIA_TEST_RELEASE_FD"]), 1)',
            'Path("sample-entered").touch(); os.read(int(os.environ["PYTHIA_TEST_RELEASE_FD"]), 1)',
        )
        reader, writer = os.pipe()
        process = None
        try:
            process = subprocess.Popen([
                sys.executable, "-c", script, "--headless", "--prompt", "hello",
            ], cwd=self.root, env={**os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                "PYTHIA_TEST_FAILURE": "", "PYTHIA_TEST_RELEASE_FD": str(reader)},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=(reader,))
            deadline = time.monotonic() + 5
            while not (self.root / "sample-entered").exists() and time.monotonic() < deadline:
                self.assertIsNone(process.poll())
                time.sleep(0.01)
            self.assertTrue((self.root / "sample-entered").exists())
            process.send_signal(signal.SIGINT)
            time.sleep(0.05)
            self.assertIsNone(process.poll())
            os.write(writer, b"x")
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertEqual(stdout, b"")
            saved = load_interaction_save(self.root / "interaction.jsonl")
            self.assertEqual([call.call_id for call in saved.pending_tool_calls()], ["pending"])
            self.assertFalse((self.root / "must-not-run").exists())
            self.assertEqual(json.loads((self.root / "cleanup.json").read_text())["active_after"], 0)
        finally:
            os.close(reader)
            os.close(writer)
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)


class HeadlessDrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_drains_sample_and_checkpoints_without_starting_tools(self):
        previous_sigint = signal.getsignal(signal.SIGINT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "save.jsonl"
            entered, release = threading.Event(), threading.Event()

            def sample(context):
                entered.set()
                if not release.wait(3):
                    raise AssertionError("sample was not released")
                return ModelSample((ToolCall("record", "pending", "{}"),))

            model = _Model(path, sample)
            environment = Environment()
            args = cli._build_parser().parse_args(["--headless", "--prompt", "hello"])
            with redirect_stderr(io.StringIO()), mock.patch.object(environment, "execute_tool_calls") as execute:
                task = asyncio.create_task(cli._run_headless(model, environment, args, path))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                    task.cancel()
                    await asyncio.sleep(0.02)
                    self.assertFalse(task.done())
                finally:
                    release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
            execute.assert_not_called()
            self.assertEqual([i.call_id for i in load_interaction_save(path).pending_tool_calls()], ["pending"])
        self.assertIs(signal.getsignal(signal.SIGINT), previous_sigint)


if __name__ == "__main__":
    unittest.main()
