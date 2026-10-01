"""Terminal fault injection and orderly worker draining, without provider I/O."""

from __future__ import annotations

from contextlib import contextmanager
import io
import os
import threading
import unittest
from unittest import mock

from pythia.interaction import Environment
from pythia.interaction import ModelSample
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolOutcome
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import load_interaction_save
from pythia.interaction._cli_terminal import PosixTerminal
from pythia_test.test_interaction_cli import _ControllerTestCase
from pythia_test.test_interaction_cli import _Model
from pythia_test.test_interaction_cli import _Terminal


@unittest.skipUnless(os.name == "posix", "uses the POSIX input adapter")
class TerminalLifetimeTests(unittest.TestCase):
    def test_partial_entry_and_output_failures_always_restore_input(self):
        for failure in ("attach", "entry-write", "entry-flush", "exit-write", "exit-flush", "detach"):
            with self.subTest(failure=failure):
                events = []
                phase = "entry"

                class Output(io.StringIO):
                    def write(self, text):
                        if failure == phase + "-write":
                            raise OSError(failure)
                        return super().write(text)

                    def flush(self):
                        if failure == phase + "-flush":
                            raise OSError(failure)
                        return super().flush()

                class Input:
                    @contextmanager
                    def raw_mode(self):
                        events.append("raw")
                        try:
                            yield
                        finally:
                            events.append("cooked")

                    @contextmanager
                    def attach(self, callback):
                        if failure == "attach":
                            raise OSError("attach")
                        events.append("attached")
                        try:
                            yield
                        finally:
                            events.append("detached")
                            if failure == "detach":
                                raise OSError("detach")

                output = Output()
                with mock.patch("pythia.term_input.create_input", return_value=Input()):
                    with self.assertRaisesRegex(OSError, failure):
                        with PosixTerminal(io.StringIO(), output):
                            phase = "exit"
                self.assertEqual(events[0], "raw")
                self.assertEqual(events[-1], "cooked")
                if "attached" in events:
                    self.assertIn("detached", events)

    def test_cleanup_failure_preserves_original_exception_and_runs_all_callbacks(self):
        terminal = PosixTerminal(io.StringIO(), io.StringIO())
        restored = []
        terminal._stack.callback(lambda: restored.append(True))
        terminal._stack.callback(mock.Mock(side_effect=OSError("detach failed")))
        original = ValueError("original failure")
        terminal.__exit__(type(original), original, None)
        self.assertEqual(restored, [True])

    def test_reader_callback_error_is_raised_by_frame_not_event_loop(self):
        terminal = PosixTerminal(io.StringIO(), io.StringIO())
        terminal._input = mock.Mock()
        terminal._input.read_keys.side_effect = OSError("input failed")
        terminal._read()
        terminal._read()
        terminal._input.read_keys.assert_called_once_with()
        with self.assertRaisesRegex(OSError, "input failed"):
            terminal.read_keys()

    def test_real_input_adapter_reports_eof_without_synthesizing_a_query(self):
        from pythia.term_input import TerminalInput

        reader, writer = os.pipe()
        os.close(writer)
        with os.fdopen(reader, "r") as stream:
            with mock.patch("sys.stderr", io.StringIO()):
                terminal = PosixTerminal(stream, io.StringIO())
                terminal._input = TerminalInput(stream)
            terminal._read()
            self.assertTrue(terminal.closed)
            self.assertEqual(terminal.read_keys(), ())


class CLIShutdownTests(_ControllerTestCase):
    async def test_terminal_errors_and_eof_drain_tool_then_restore_terminal(self):
        for failure in ("read", "render", "eof"):
            with self.subTest(failure=failure):
                entered, release = threading.Event(), threading.Event()
                finished = []
                calls = (ToolCall("record", "one", "{}"), ToolCall("record", "two", "{}"))

                class Terminal(_Terminal):
                    def read_keys(self):
                        if failure == "read" and entered.is_set():
                            release.set()
                            raise OSError("read failure")
                        return super().read_keys()

                def frame(terminal, editor, status):
                    if entered.is_set():
                        if failure == "render":
                            release.set()
                            raise OSError("render failure")
                        if failure == "eof":
                            terminal.closed = True
                    if status.startswith("closing"):
                        release.set()

                terminal = Terminal(frame)

                def record(arguments, *, timeout_seconds=None):
                    entered.set()
                    if not release.wait(2):
                        raise AssertionError("tool was not released")
                    self.assertFalse(terminal.exited)
                    finished.append(True)
                    return ToolOutcome("drained result")

                model = _Model(self.path, ModelSample(items=calls))
                environment = Environment((Tool(ToolSpec("record", "", {}), record),))
                argv = ["--resume=False", "--prompt", "hello"]  # each failure mode starts fresh
                try:
                    if failure == "eof":
                        self.assertEqual(await self._run(model, terminal, argv, environment), 0)
                    else:
                        with self.assertRaisesRegex(OSError, failure + " failure"):
                            await self._run(model, terminal, argv, environment)
                finally:
                    release.set()
                self.assertTrue(terminal.exited)
                self.assertEqual(finished, [True])
                self.assertEqual(len(model.calls), 1)
                saved = load_interaction_save(self.path)
                self.assertEqual(saved.items[-1], ToolResult("one", "drained result"))
                self.assertEqual(saved.pending_tool_calls(), (calls[1],))

    async def test_initial_render_error_precedes_session_and_model_effects(self):
        def frame(terminal, editor, status):
            raise OSError("cannot render")

        terminal = _Terminal(frame)
        model = _Model(self.path)
        with self.assertRaisesRegex(OSError, "cannot render"):
            await self._run(model, terminal, ["--prompt", "hello"])
        self.assertTrue(terminal.exited)
        self.assertFalse(self.path.exists())
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
