"""CLI/auto prompt-file parsing and preflight, with no provider requests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import auto, cli, demo
from pythia.interaction._auto_config import build_parser as auto_parser
from pythia.interaction._prompt import load_prompt


class PromptFileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_mutual_exclusion_is_checked_before_any_file_read(self):
        for factory in (cli._build_parser, auto_parser):
            for argv in (["--prompt", "text", "--prompt-file", "missing"],
                         ["--prompt-file", "missing", "--prompt", ""]):
                with self.subTest(frontend=factory.__module__, argv=argv):
                    with mock.patch("pythia.interaction._prompt.os.open") as opened, redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            factory().parse_args(argv)
                    self.assertEqual(raised.exception.code, 2)
                    opened.assert_not_called()

    def test_invalid_path_arguments_and_demo_scope(self):
        for factory in (cli._build_parser, auto_parser):
            for value in ("", " \n", "-", "bad\x00path"):
                with self.subTest(frontend=factory.__module__, value=value), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        factory().parse_args(["--prompt-file", value])
        self.assertFalse(hasattr(demo._build_parser().parse_args([]), "prompt_file"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            demo._build_parser().parse_args(["--prompt-file", "file"])

    def test_file_rstrip_preserves_leading_internal_whitespace_and_paths(self):
        text = " \t/retry \t\r\nLiteral café prompt\n\n \t\u2003"
        expected = " \t/retry \t\r\nLiteral café prompt"
        source = self.root / "prompt with spaces.md"
        source.write_bytes(text.encode("utf-8"))
        workspace = self.root / "workspace"
        workspace.mkdir()
        before = Path.cwd()
        try:
            os.chdir(self.root)
            with mock.patch.dict(os.environ, {"HOME": str(self.root)}):
                for factory in (cli._build_parser, auto_parser):
                    for value in (source.name, "~/" + source.name, str(source)):
                        with self.subTest(frontend=factory.__module__, path=value):
                            args = factory().parse_args(["--prompt-file", value, "--cwd", str(workspace)])
                            self.assertEqual(load_prompt(args), expected)
        finally:
            os.chdir(before)
        for factory in (cli._build_parser, auto_parser):
            self.assertIsNone(load_prompt(factory().parse_args([])))
            self.assertEqual(load_prompt(factory().parse_args(["--prompt", "  literal\n"])),
                             "  literal\n")

    def test_nontext_empty_missing_directory_and_fifo_are_rejected(self):
        paths = [self.root, self.root / "missing"]
        for index, body in enumerate((b"", b" \t\r\n", b"FAKE_SECRET\xff")):
            path = self.root / str(index)
            path.write_bytes(body)
            paths.append(path)
        if hasattr(os, "mkfifo"):
            fifo = self.root / "fifo"
            os.mkfifo(fifo)
            paths.append(fifo)
            if hasattr(os, "O_NONBLOCK"):
                # Simulate replacement by a FIFO after the regular-file check.
                with mock.patch.object(Path, "is_file", return_value=True), self.assertRaises(ValueError):
                    load_prompt(SimpleNamespace(prompt=None, prompt_file=fifo))
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as raised:
                    load_prompt(SimpleNamespace(prompt=None, prompt_file=path))
                self.assertNotIn("FAKE_SECRET", str(raised.exception))
        unreadable = self.root / "unreadable"
        unreadable.write_text("contents")
        with mock.patch("pythia.interaction._prompt.os.open", side_effect=PermissionError):
            with self.assertRaises(ValueError):
                load_prompt(SimpleNamespace(prompt=None, prompt_file=unreadable))

    def test_file_errors_precede_model_environment_board_and_save_effects(self):
        source = self.root / "empty"
        source.write_text("")
        save = self.root / "keep"
        save.write_bytes(b"existing data\n")
        for frontend in (cli, auto):
            with self.subTest(frontend=frontend.__name__):
                with mock.patch.object(frontend, "build_model") as model, \
                        mock.patch.object(frontend, "DefaultEnvironment") as environment, \
                        mock.patch.object(frontend, "save_interaction_save") as checkpoint, \
                        mock.patch.object(auto, "_Session") as session, \
                        redirect_stderr(io.StringIO()):
                    self.assertEqual(frontend.main([
                        "--headless", "--prompt-file", str(source), "--save", str(save),
                    ]), 1)
                model.assert_not_called()
                environment.assert_not_called()
                checkpoint.assert_not_called()
                session.assert_not_called()
                self.assertEqual(save.read_bytes(), b"existing data\n")

    def test_auto_prompt_file_uses_one_prompt_path_with_or_without_headless(self):
        source = self.root / "task.md"
        text = "  /contexts\nThis is a literal board task.\n"
        source.write_text(text, encoding="utf-8")
        for headless in (False, True):
            with self.subTest(headless=headless):
                session = mock.Mock(has_errors=False)
                session.service.base_url = "http://127.0.0.1:43210"
                output = io.StringIO()

                def start():
                    # The file is already loaded; startup must not read it again.
                    source.write_text("changed after loading")
                    return session

                source.write_text(text, encoding="utf-8")
                session.start.side_effect = start
                with mock.patch.object(auto, "_Session", return_value=session) as factory, \
                        mock.patch.object(auto, "_one_prompt", return_value=0) as run, \
                        mock.patch.object(auto, "_headless") as wait, \
                        mock.patch.object(auto, "_interactive") as interactive, \
                        mock.patch.object(auto, "PosixTerminal") as terminal, \
                        mock.patch.object(auto, "_print_events"), redirect_stdout(output):
                    self.assertEqual(auto.main([
                        "--enable-experimental-worker-board",
                        f"--headless={headless}", "--prompt-file", str(source),
                        "--save", str(self.root / "run"),
                    ]), 0)
                run.assert_called_once_with(session, text.rstrip(), display=not headless)
                wait.assert_not_called()
                interactive.assert_not_called()
                terminal.assert_not_called()
                session.close.assert_called_once()
                settings = factory.call_args.args[1]
                self.assertTrue(all("prompt_file" not in value for value in settings.values()))
                self.assertEqual(output.getvalue(), "Board: http://127.0.0.1:43210/README.md\n")


if __name__ == "__main__":
    unittest.main()
