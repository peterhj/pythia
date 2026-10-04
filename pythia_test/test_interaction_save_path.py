"""Working-tree CLI/demo acceptance for --save, with no live provider I/O."""

from __future__ import annotations

from contextlib import ExitStack
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import cli, demo
from pythia.interaction import (
    Environment, Init, Instructions, Message, InteractionContext, ModelSample,
    ToolCall, ToolResult, Tools, TurnSummary, UserInteractionBoundary,
    UserToolCall, UserToolResult, load_interaction_save, save_interaction_save,
)
from pythia.interaction.model_config import DEFAULT_SAVE_PATH, resolve_save_path
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer


_COMPLETED = (Init("previous"), Message("assistant", "previous answer"), TurnSummary())
_PLAN_CALL = ToolCall("update_plan", "plan-one", '{"plan":[{"step":"Inspect","status":"pending"}]}')


class _SavePathTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.launch = self.root / "launch"
        self.workspace = self.root / "workspace"
        self.launch.mkdir()
        self.workspace.mkdir()
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.launch)
        self.default = self.launch / "interaction.jsonl"
        self.default_bytes = b"default log must not be touched\n"
        self.default.write_bytes(self.default_bytes)
        (self.launch / "logs").mkdir()
        self.selected = self.launch / "logs" / "chosen log.data"

    def assert_default_untouched(self):
        self.assertEqual(self.default.read_bytes(), self.default_bytes)
        self.assertFalse((self.workspace / "interaction.jsonl").exists())
        self.assertEqual(Path.cwd(), self.launch)


class SaveArgumentTests(_SavePathTestCase):
    def test_shared_default_and_explicit_save_paths_do_not_change_resume(self):
        for frontend, resume_default in ((cli, True), (demo, False)):
            with self.subTest(frontend=frontend.__name__):
                parser = frontend._build_parser()
                self.assertEqual(parser.parse_args([]).save_path, DEFAULT_SAVE_PATH)
                self.assertIs(parser.parse_args([]).resume, resume_default)
                for raw in ("review.jsonl", "~/review.jsonl", str(self.selected), " spaced name "):
                    args = parser.parse_args(["--save", raw])
                    self.assertIsInstance(args.save_path, Path)
                    self.assertEqual(args.save_path, Path(raw))
                    self.assertIs(args.resume, resume_default)
                help_text = " ".join(parser.format_help().split())
                self.assertIn("--save PATH", help_text)
                self.assertIn("default: interaction.jsonl", help_text)
                self.assertIn("selected --save file", help_text)
        argv = ["--save", "review", "--resume"]
        demo_args = vars(demo._build_parser().parse_args(argv))
        self.assertFalse(demo_args.pop("experimental_user_message_injection"))
        cli_args = vars(cli._build_parser().parse_args(argv))
        self.assertTrue(cli_args.pop("enable_default_tools"))
        self.assertFalse(cli_args.pop("headless"))
        self.assertFalse(cli_args.pop("debug_trace"))
        self.assertIsNone(cli_args.pop("prompt_file"))
        self.assertEqual(cli_args, demo_args)

    def test_raw_validation_rejects_empty_nul_stream_and_old_flag(self):
        for frontend in (cli, demo):
            for argv in (["--save", ""], ["--save", " \n\t"], ["--save", "bad\x00path"],
                         ["--save", "-"], ["--save"], ["--session", "old.jsonl"]):
                with self.subTest(frontend=frontend.__name__, argv=argv):
                    with mock.patch("sys.stderr", io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            frontend._build_parser().parse_args(argv)
                    self.assertEqual(error.exception.code, 2)
        self.assert_default_untouched()

    def test_paths_expand_home_but_not_environment_variables_or_filename_spaces(self):
        home = self.root / "home"
        home.mkdir()
        (self.launch / "$HOME").mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            self.assertEqual(resolve_save_path(Path("~/session")), home / "session")
            self.assertEqual(resolve_save_path(Path("$HOME/session")), self.launch / "$HOME/session")
        self.assertEqual(resolve_save_path(Path("logs/chosen log.data")), self.selected)
        self.assertEqual(resolve_save_path(self.selected), self.selected)
        self.assertEqual(resolve_save_path(Path(" spaced name ")), self.launch / " spaced name ")
        self.assertEqual(resolve_save_path(DEFAULT_SAVE_PATH), self.default)
        self.assert_default_untouched()

    @unittest.skipUnless(os.name == "posix", "uses POSIX symlinks")
    def test_normalization_does_not_resolve_file_symlinks(self):
        target = self.root / "target.jsonl"
        target.write_text("target sentinel\n")
        link = self.launch / "linked.jsonl"
        link.symlink_to(target)
        self.assertEqual(resolve_save_path(Path("linked.jsonl")), link)
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_text(), "target sentinel\n")


@unittest.skipUnless(os.name == "posix", "interactive main requires POSIX")
class SaveEntrypointTests(_SavePathTestCase):
    def _main(self, frontend, argv=(), *, samples=(_answer(),), save_argument=None):
        model = _Model(self.selected, *samples)
        model.endpoint = SimpleNamespace(model="initial-model")
        terminal = _Terminal(
            lambda t, e, s: t.key("c-d") if s in {"idle", "failed", "auth needed"} else None
        )
        argument = str(self.selected.relative_to(self.launch)) if save_argument is None else save_argument
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                frontend, "build_model" if frontend is cli else "_build_model", return_value=model,
            ))
            printed = stack.enter_context(mock.patch("builtins.print"))
            if frontend is cli:
                stream = SimpleNamespace(isatty=lambda: True)
                stack.enter_context(mock.patch.object(cli.sys, "stdin", stream))
                stack.enter_context(mock.patch.object(cli.sys, "stdout", stream))
                stack.enter_context(mock.patch.object(cli, "PosixTerminal", return_value=terminal))
            code = frontend.main(["--cwd", str(self.workspace), "--save", argument, *argv])
        self.assertEqual([context.items for context, _tools, _options in model.calls], model.checkpoints)
        self.assert_default_untouched()
        if frontend is cli:
            self.assertTrue(terminal.exited)
            self.assertEqual(sum(i.text == f"[cli] Save log: {self.selected}" for i in terminal.items), 1)
        return code, model, terminal, tuple(call.args[0] for call in printed.call_args_list)

    def test_missing_experimental_demo_resume_seeds_only_selected_log(self):
        status, model, _terminal, _printed = self._main(
            demo, ["--resume", "--experimental-user-message-injection"],
        )
        self.assertEqual(status, 0)
        self.assertEqual(model.calls[0][0].items[1:], (
            Tools(model.calls[0][1]),
            Message("user", demo.EXPERIMENTAL_USER_MESSAGE_PROMPT),
            UserInteractionBoundary(),
        ))
        self.assertIn(
            "experimental_inject_user_message",
            tuple(spec.name for spec in model.calls[0][1]),
        )
        self.assert_default_untouched()

    def test_fresh_custom_file_replaces_only_selected_log_and_checkpoints_all_batches(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                self.selected.write_text("old selected file\n")
                code, model, _terminal, _printed = self._main(
                    frontend, ["--resume=False", "--prompt", "fresh query"],
                    samples=(ModelSample(items=(_PLAN_CALL,)), _answer()),
                )
                self.assertEqual(code, 0)
                self.assertEqual(len(model.calls), 2)
                saved = load_interaction_save(self.selected)
                self.assertEqual(saved.items[0].model, "initial-model")
                self.assertEqual(saved.items[2:4], (Message("user", "fresh query"), UserInteractionBoundary()))
                self.assertIn(ToolResult("plan-one", "Plan updated"), saved.items)
                self.assertEqual(saved.items[-1].sample_count, 2)
                self.assertNotIn("Save log:", repr(saved.items))
                self.assertNotIn(str(self.selected), repr(saved.items))

    def test_absolute_and_home_paths_reach_both_entrypoints(self):
        for frontend in (cli, demo):
            for argument in (str(self.selected), "~/logs/chosen log.data"):
                with self.subTest(frontend=frontend.__name__, argument=argument):
                    with mock.patch.dict(os.environ, {"HOME": str(self.launch)}):
                        code, model, _terminal, _printed = self._main(
                            frontend, ["--prompt", "hello"], save_argument=argument,
                        )
                    self.assertEqual(code, 0)
                    self.assertEqual(len(model.calls), 1)
                    self.assertIn(Message("user", "hello"), load_interaction_save(self.selected).items)

    def test_completed_resume_replays_only_chosen_file_without_sampling(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                with demo.DefaultEnvironment(cwd=self.workspace) as environment:
                    snapshot = Tools(environment.tool_specs)
                save_interaction_save(self.selected, InteractionContext((*_COMPLETED, snapshot)))
                before = self.selected.read_bytes()
                code, model, terminal, printed = self._main(frontend, ["--resume"], samples=())
                self.assertEqual(code, 0)
                self.assertEqual(model.calls, [])
                self.assertEqual(self.selected.read_bytes(), before)
                texts = [str(i) for i in terminal.items] if frontend is cli else [str(i) for i in printed]
                self.assertEqual(sum("[assistant] previous answer" in text for text in texts), 1)

    def test_missing_chosen_resume_never_falls_back_to_existing_default(self):
        for frontend in (cli, demo):
            for prompt in (None, "explicit"):
                with self.subTest(frontend=frontend.__name__, prompt=prompt):
                    self.selected.unlink(missing_ok=True)
                    argv = ["--resume"] + (["--prompt", prompt] if prompt is not None else [])
                    samples = () if frontend is cli and prompt is None else (_answer(),)
                    code, model, terminal, printed = self._main(frontend, argv, samples=samples)
                    self.assertEqual(code, 0)
                    self.assertEqual(len(model.calls), len(samples))
                    notices = "\n".join(str(i) for i in (*terminal.items, *printed))
                    self.assertIn("no existing chosen log.data was found", notices)
                    saved = load_interaction_save(self.selected)
                    users = [i.content for i in saved if isinstance(i, Message) and i.role == "user"]
                    expected = [] if not samples else [prompt if prompt is not None else demo.DEFAULT_PROMPT]
                    self.assertEqual(users, expected)

    def test_malformed_chosen_resume_preserves_both_files(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                self.selected.write_bytes(b"invalid JSON\n")
                code, model, _terminal, _printed = self._main(frontend, ["--resume"], samples=())
                self.assertEqual(code, 1)
                self.assertEqual(model.calls, [])
                self.assertEqual(self.selected.read_bytes(), b"invalid JSON\n")

    def test_pending_call_override_and_follow_up_keep_frontend_recovery_policies(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                original = (Init("old"), Message("user", "old query"), _PLAN_CALL)
                save_interaction_save(self.selected, InteractionContext(original))
                code, model, _terminal, _printed = self._main(
                    frontend, ["--resume", "--instructions", "", "--prompt", "follow-up"],
                )
                self.assertEqual(code, 0)
                received = model.calls[0][0].items
                self.assertEqual(received[:len(original)], original)
                result = received[-5]
                self.assertEqual(received[-4], Tools(model.calls[0][1]))
                self.assertEqual(result.call_id, _PLAN_CALL.call_id)
                self.assertEqual(result.success, frontend is demo)
                if frontend is cli:
                    self.assertIn("was not rerun", result.output)
                else:
                    self.assertEqual(result.output, "Plan updated")
                self.assertEqual(received[-3:], (
                    Instructions(""), Message("user", "follow-up"), UserInteractionBoundary(),
                ))

    def test_instructions_only_resume_on_selected_file_adds_no_user_turn(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                save_interaction_save(self.selected, InteractionContext(_COMPLETED))
                code, model, _terminal, _printed = self._main(
                    frontend, ["--resume", "--instructions", "override"],
                )
                self.assertEqual(code, 0)
                self.assertEqual(model.calls[0][0].items, (*_COMPLETED, Tools(model.calls[0][1]), Instructions("override")))

    def test_custom_user_tool_recovery_never_dispatches_or_samples(self):
        call = UserToolCall(ToolCall("login", "user_pending", "{}"))
        original = (*_COMPLETED, call)
        save_interaction_save(self.selected, InteractionContext(original))
        with mock.patch.object(cli, "create_user_environment") as create:
            code, model, _terminal, _printed = self._main(cli, ["--resume"], samples=())
        self.assertEqual(code, 0)
        create.assert_not_called()
        self.assertEqual(model.calls, [])
        saved = load_interaction_save(self.selected)
        self.assertEqual(saved.items[:-2], original)
        self.assertIsInstance(saved.items[-1], Tools)
        self.assertIsInstance(saved.items[-2], UserToolResult)
        self.assertFalse(saved.items[-2].result.success)

    def test_checkpoint_failure_keeps_chosen_file_and_does_not_redirect_to_default(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                self.selected.write_bytes(b"previous selected contents\n")
                with mock.patch.object(frontend, "save_interaction_save", side_effect=OSError("disk failure")) as save:
                    code, model, terminal, printed = self._main(
                        frontend, ["--resume=False", "--prompt", "hello"], samples=(),
                    )
                self.assertEqual(code, 1)
                self.assertEqual(model.calls, [])
                save.assert_called_once()
                self.assertEqual(Path(save.call_args.args[0]), self.selected)
                self.assertEqual(self.selected.read_bytes(), b"previous selected contents\n")
                if frontend is cli:
                    self.assertTrue(any("unsaved state remains in memory" in i.text for i in terminal.items))

    def test_path_errors_fail_before_model_environment_or_terminal_creation(self):
        not_directory = self.launch / "not-directory"
        not_directory.write_text("not a directory")
        fifo = self.launch / "pipe"
        os.mkfifo(fifo)
        for frontend in (cli, demo):
            for path in (self.workspace, self.launch / "missing/log", not_directory / "log", fifo, Path(os.devnull)):
                with self.subTest(frontend=frontend.__name__, path=path):
                    stream = SimpleNamespace(isatty=lambda: True)
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(cli.sys, "stdin", stream))
                        stack.enter_context(mock.patch.object(cli.sys, "stdout", stream))
                        stack.enter_context(mock.patch("builtins.print"))
                        model = stack.enter_context(mock.patch.object(
                            frontend, "build_model" if frontend is cli else "_build_model",
                            side_effect=AssertionError("model should not be constructed"),
                        ))
                        environment = stack.enter_context(mock.patch.object(frontend, "DefaultEnvironment"))
                        terminal = stack.enter_context(mock.patch.object(cli, "PosixTerminal"))
                        self.assertEqual(frontend.main(["--save", str(path)]), 1)
                    model.assert_not_called()
                    environment.assert_not_called()
                    terminal.assert_not_called()
                    self.assert_default_untouched()
        self.assertFalse((self.launch / "missing").exists())
        self.assertFalse(self.selected.exists())

    def test_unavailable_home_fails_before_model_construction(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                stream = SimpleNamespace(isatty=lambda: True)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(cli.sys, "stdin", stream))
                    stack.enter_context(mock.patch.object(cli.sys, "stdout", stream))
                    stack.enter_context(mock.patch("builtins.print"))
                    stack.enter_context(mock.patch("os.path.expanduser", return_value="~"))
                    model = stack.enter_context(mock.patch.object(
                        frontend, "build_model" if frontend is cli else "_build_model",
                        side_effect=AssertionError("model should not be constructed"),
                    ))
                    self.assertEqual(frontend.main(["--save", "~/session.jsonl"]), 1)
                model.assert_not_called()
                self.assert_default_untouched()

    def test_non_tty_rejection_precedes_path_resolution(self):
        with mock.patch.object(cli.sys, "stdin", io.StringIO()), mock.patch("builtins.print"):
            with mock.patch.object(cli, "resolve_save_path") as resolve:
                self.assertEqual(cli.main(["--save", "missing/log.jsonl"]), 1)
        resolve.assert_not_called()
        self.assert_default_untouched()

    def test_programmatic_demo_can_still_disable_persistence(self):
        model = mock.Mock(spec=["sample"])
        model.sample.return_value = _answer()
        with mock.patch("builtins.print"), mock.patch.object(demo, "save_interaction_save") as save:
            self.assertEqual(demo.run(model, Environment(), prompt="hello", save_path=None), "done")
        save.assert_not_called()
        self.assert_default_untouched()


if __name__ == "__main__":
    unittest.main()
