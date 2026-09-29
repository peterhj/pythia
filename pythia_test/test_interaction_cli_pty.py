"""POSIX PTY tests of the real entry point with a scripted, offline model."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if os.name == "posix":
    import fcntl
    import pty
    import struct
    import termios

from pythia.interaction import Init
from pythia.interaction import Message
from pythia.interaction import InteractionContext
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import TurnSummary
from pythia.interaction import UserToolCall
from pythia.interaction import UserToolResult
from pythia.interaction import load_interaction_save
from pythia.interaction import save_interaction_save


_SCRIPT = '''
import json
import os
from pathlib import Path
from pythia.interaction import cli, Message, ModelSample, ModelTimeoutError, ToolCall, ToolResult, load_interaction_save

failure = os.environ.get("PYTHIA_TEST_FAILURE")

class Environment(cli.DefaultEnvironment):
    def close(self):
        processes = [s.process for s in self.command_runtime._sessions.values()]
        super().close()
        Path("cleanup.json").write_text(json.dumps({
            "active_before": len(processes),
            "active_after": len(self.command_runtime.active_session_ids),
            "terminated": all(p.poll() is not None for p in processes),
            "pipes_closed": all(p.stdin.closed and p.stdout.closed for p in processes),
        }))

cli.DefaultEnvironment = Environment

if failure in {"initial-save", "save"}:
    save = cli.save_interaction_save
    def fail_save(path, context):
        if failure == "initial-save" or any(isinstance(i, ToolResult) for i in context):
            raise OSError("injected checkpoint failure")
        save(path, context)
    cli.save_interaction_save = fail_save
elif failure == "render":
    render = cli.PosixTerminal.render
    def fail_render(self, editor, status, items, prompt=":> "):
        render(self, editor, status, items, prompt)
        if status == "idle":
            raise OSError("injected render failure")
    cli.PosixTerminal.render = fail_render
elif failure == "attach":
    import pythia.term_input as term_input
    create = term_input.create_input
    def fail_create(stdin):
        result = create(stdin)
        def attach(callback):
            raise OSError("injected attach failure")
        result.attach = attach
        return result
    term_input.create_input = fail_create

class Model:
    def __init__(self, save_path, enable_default_tools):
        self.tool_done = False
        self.save_path = Path(save_path).expanduser().absolute()
        self.enable_default_tools = enable_default_tools

    def sample(self, context, *, tools=(), sample_params=None):
        assert load_interaction_save(self.save_path).items == context.items
        assert {tool.name for tool in tools} == (
            {"exec_command", "write_stdin", "apply_patch", "update_plan"}
            if self.enable_default_tools else set()
        )
        if "PYTHIA_TEST_RELEASE_FD" in os.environ:
            os.read(int(os.environ["PYTHIA_TEST_RELEASE_FD"]), 1)
            return ModelSample(items=(ToolCall(
                "exec_command", "pending", '{"cmd":"touch must-not-run"}'
            ),))
        if not self.tool_done:
            self.tool_done = True
            if failure == "retry":
                raise ModelTimeoutError("injected sample timeout")
            if failure in {"model", "save", "render"}:
                return ModelSample(items=(ToolCall(
                    "exec_command", "command-1", '{"cmd":"read line","yield_time_ms":0}'
                ),))
            return ModelSample(items=(ToolCall(
                "update_plan", "plan-1",
                '{"plan":[{"step":"Inspect","status":"in_progress"}]}'
            ),))
        if failure == "model":
            raise RuntimeError("injected model failure")
        count = sum(isinstance(i, Message) and i.role == "user" for i in context)
        return ModelSample(items=(Message("assistant", f"answer-{count}"),))

if failure in {"auth", "auth-cancel"}:
    from functools import partial
    from pythia.interaction import user_tools
    from pythia.interaction.codex_login import login
    build = cli.build_model
    def build_when_authenticated(args):
        build(args)  # Real routing/validation/loading, but no provider sampling.
        return Model(args.save_path, args.enable_default_tools)
    cli.build_model = build_when_authenticated
    if failure == "auth":
        def fake_login(path, **kwargs):
            path.write_text(json.dumps({"tokens": {"access_token": "FAKE_PTY_SECRET", "account_id": "account"}}))
        user_tools.login = fake_login
        user_tools.query_quota = lambda auth, **kwargs: "offline quota snapshot"
    else:
        user_tools.login = partial(login, callback_port=0)
else:
    cli.build_model = lambda args: Model(args.save_path, args.enable_default_tools)
raise SystemExit(cli.main())
'''


@unittest.skipUnless(os.name == "posix", "requires POSIX PTYs")
class PosixCLITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.master, self.slave = pty.openpty()
        self.resize(80, 24)
        self.attributes = termios.tcgetattr(self.slave)
        os.set_blocking(self.master, False)
        self.output = bytearray()
        self.changed = asyncio.Event()
        self.process = None
        self.release = None
        asyncio.get_running_loop().add_reader(self.master, self._read)

    def resize(self, columns, rows):
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))

    def _read(self):
        try:
            data = os.read(self.master, 65536)
        except BlockingIOError:
            return
        except OSError:
            asyncio.get_running_loop().remove_reader(self.master)
            return
        if data:
            self.output.extend(data)
            self.changed.set()

    def start(self, *args, blocked=False, failure=None):
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        if failure:
            env["PYTHIA_TEST_FAILURE"] = failure
        inherited = ()
        if blocked:
            reader, self.release = os.pipe()
            env["PYTHIA_TEST_RELEASE_FD"] = str(reader)
            inherited = (reader,)
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-c", _SCRIPT, *args],
                stdin=self.slave, stdout=self.slave, stderr=subprocess.PIPE,
                cwd=self.root, env=env, pass_fds=inherited,
            )
        finally:
            for descriptor in inherited:
                os.close(descriptor)

    async def wait_output(self, text):
        async def wait():
            while text not in self.output:
                self.changed.clear()
                await self.changed.wait()
        try:
            await asyncio.wait_for(wait(), 4)
        except asyncio.TimeoutError:
            self.fail(f"did not see {text!r}; terminal tail: {bytes(self.output[-2000:])!r}")

    async def wait_exit(self, expected=0, paste_enabled=True, default_tools=True):
        status = await asyncio.to_thread(self.process.wait, timeout=4)
        stderr = self.process.stderr.read().decode()
        self.assertEqual(status, expected, stderr)
        self.assertEqual(termios.tcgetattr(self.slave), self.attributes)
        # Consume remaining output queued by the child before checking cleanup.
        self._read()
        if paste_enabled:
            self.assertIn(b"\x1b[?2004h", self.output)
        self.assertIn(b"\x1b[?2004l", self.output)
        if not default_tools:
            self.assertFalse((self.root / "cleanup.json").exists())
            return
        cleanup = json.loads((self.root / "cleanup.json").read_text())
        self.assertEqual(cleanup["active_after"], 0)
        self.assertTrue(cleanup["terminated"])
        self.assertTrue(cleanup["pipes_closed"])

    async def asyncTearDown(self):
        if self.release is not None:
            os.close(self.release)
            self.release = None
        if self.process is not None:
            if self.process.poll() is None:
                os.write(self.master, b"\x04")
                try:
                    await asyncio.to_thread(self.process.wait, timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait, timeout=2)
            self.process.stderr.close()
        asyncio.get_running_loop().remove_reader(self.master)
        os.close(self.master)
        os.close(self.slave)
        self.temporary.cleanup()

    async def test_initial_query_tools_multiline_edit_resize_and_quit(self):
        self.start("--prompt", "initial")
        await self.wait_output(b"[assistant] answer-1")
        await self.wait_output(b"idle")
        self.resize(12, 6)
        os.write(self.master, "\x1b[200~next\ncafé\x1b[201~".encode())
        await self.wait_output("café".encode())
        users_before_enter = [
            i for i in load_interaction_save(self.root / "interaction.jsonl")
            if isinstance(i, Message) and i.role == "user"
        ]
        self.assertEqual(users_before_enter, [Message("user", "initial")])
        os.write(self.master, b"\x1b[D!\r")
        await self.wait_output(b"[assistant] answer-2")
        os.write(self.master, b"/quit\r")
        await self.wait_exit()
        saved = load_interaction_save(self.root / "interaction.jsonl")
        self.assertEqual(
            [i for i in saved if isinstance(i, Message) and i.role == "user"],
            [Message("user", "initial"), Message("user", "next\ncaf!é")],
        )
        self.assertEqual([i.sample_count for i in saved if isinstance(i, TurnSummary)], [2, 3])
        self.assertEqual(self.output.count(b"[user] initial"), 1)
        self.assertEqual(self.output.count(b"[assistant] answer-1"), 1)
        self.assertIn(b"[tool-ret]  update_plan (plan-1) [ok]", self.output)

    async def test_ctrl_d_exits_with_nonempty_draft_without_sampling(self):
        self.start()
        await self.wait_output(b"idle")
        os.write(self.master, b"unfinished draft\x04")
        await self.wait_exit()
        saved = load_interaction_save(self.root / "interaction.jsonl")
        self.assertEqual(len(saved.items), 1)

    async def test_save_path_stays_in_launch_directory_not_tool_workspace(self):
        workspace = self.root / "tools"
        workspace.mkdir()
        self.start("--cwd", str(workspace))
        await self.wait_output(b"idle")
        os.write(self.master, b"\x04")
        await self.wait_exit()
        self.assertTrue((self.root / "interaction.jsonl").exists())
        self.assertFalse((workspace / "interaction.jsonl").exists())

    async def test_custom_save_path_and_resume_preserve_default_and_workspace_logs(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        (self.root / "logs").mkdir()
        selected = self.root / "logs" / "chosen file.jsonl"
        selected.write_text("old selected log\n")
        default = self.root / "interaction.jsonl"
        default.write_bytes(b"default sentinel\n")
        workspace_log = workspace / "interaction.jsonl"
        workspace_log.write_bytes(b"workspace sentinel\n")
        self.start("--save", "logs/chosen file.jsonl", "--cwd", str(workspace), "--prompt", "first")
        await self.wait_output(b"[assistant] answer-1")
        await self.wait_output(b"idle")
        os.write(self.master, b"/quit\r")
        await self.wait_exit()
        before = selected.read_bytes()
        self.assertEqual(default.read_bytes(), b"default sentinel\n")
        self.assertEqual(workspace_log.read_bytes(), b"workspace sentinel\n")
        self.assertIn(f"Save log: {selected}".encode(), self.output)
        self.assertEqual([i for i in load_interaction_save(selected) if isinstance(i, Message) and i.role == "user"],
                         [Message("user", "first")])

        self.process.stderr.close()
        self.output.clear()
        self.start("--save", str(selected), "--resume", "--cwd", str(workspace))
        await self.wait_output(b"[assistant] answer-1")
        await self.wait_output(b"idle")
        os.write(self.master, b"/quit\r")
        await self.wait_exit()
        self.assertEqual(self.output.count(b"[assistant] answer-1"), 1)
        self.assertEqual(selected.read_bytes(), before)
        self.assertEqual(default.read_bytes(), b"default sentinel\n")
        self.assertEqual(workspace_log.read_bytes(), b"workspace sentinel\n")

    async def test_custom_save_path_survives_auth_activation_and_quota(self):
        selected = self.root / "auth session.jsonl"
        auth = self.root / "credentials.json"
        default = self.root / "interaction.jsonl"
        default.write_bytes(b"default sentinel\n")
        self.start("--endpoint-api", "codex", "--model", "test", "--endpoint-auth-file", str(auth),
                   "--save", "auth session.jsonl", "--prompt", "blocked initial", failure="auth")
        await self.wait_output(b"auth needed")
        self.assertEqual(len(load_interaction_save(selected).items), 1)
        os.write(self.master, b"\x15\x0b/login\r")
        await self.wait_output(b"Model ready")
        os.write(self.master, b"/quota\r")
        await self.wait_output(b"offline quota snapshot")
        os.write(self.master, b"explicit query\r")
        await self.wait_output(b"[assistant] answer-1")
        os.write(self.master, b"/quit\r")
        await self.wait_exit()
        saved = load_interaction_save(selected)
        self.assertEqual([i.call.name for i in saved if isinstance(i, UserToolCall)], ["login", "quota"])
        self.assertEqual([i for i in saved if isinstance(i, Message) and i.role == "user"],
                         [Message("user", "explicit query")])
        self.assertEqual(default.read_bytes(), b"default sentinel\n")
        self.assertEqual(json.loads(auth.read_text())["tokens"]["access_token"], "FAKE_PTY_SECRET")
        self.assertNotIn("FAKE_PTY_SECRET", selected.read_text())

    async def test_custom_save_path_checkpoints_inflight_result_on_exit(self):
        selected = self.root / "stopped.jsonl"
        default = self.root / "interaction.jsonl"
        default.write_bytes(b"default sentinel\n")
        self.start("--save", "stopped.jsonl", "--prompt", "initial", blocked=True)
        await self.wait_output(b"sampling")
        os.write(self.master, b"\x03")
        await self.wait_output(b"closing")
        os.write(self.release, b"x")
        await self.wait_exit()
        self.assertEqual([call.call_id for call in load_interaction_save(selected).pending_tool_calls()], ["pending"])
        self.assertEqual(default.read_bytes(), b"default sentinel\n")
        self.assertFalse((self.root / "must-not-run").exists())

    async def test_ctrl_c_during_sample_drains_worker_and_leaves_calls_pending(self):
        self.start("--prompt", "initial", blocked=True)
        await self.wait_output(b"sampling")
        os.write(self.master, b"\x03")
        await self.wait_output(b"closing")
        os.write(self.release, b"x")
        await self.wait_exit()
        saved = load_interaction_save(self.root / "interaction.jsonl")
        self.assertEqual(saved.pending_tool_calls(), (
            ToolCall("exec_command", "pending", '{"cmd":"touch must-not-run"}'),
        ))
        self.assertFalse((self.root / "must-not-run").exists())

    async def test_startup_error_restores_terminal_and_preserves_bad_session(self):
        path = self.root / "interaction.jsonl"
        path.write_text("invalid JSON\n")
        self.start("--resume")
        await self.wait_exit(expected=1)
        self.assertEqual(path.read_text(), "invalid JSON\n")

    async def test_resume_never_executes_unresolved_command_and_waits_for_input(self):
        path = self.root / "interaction.jsonl"
        original = (Init("old"), Message("user", "old query"),
                    ToolCall("exec_command", "pending", '{"cmd":"touch must-not-run"}'))
        save_interaction_save(path, InteractionContext(original))
        self.start("--resume")
        await self.wait_output(b"was not rerun")
        await self.wait_output(b"idle")
        os.write(self.master, b"/quit\r")
        await self.wait_exit()
        saved = load_interaction_save(path)
        self.assertEqual(saved.items[:-1], original)
        self.assertIsInstance(saved.items[-1], ToolResult)
        self.assertFalse(saved.items[-1].success)
        self.assertFalse((self.root / "must-not-run").exists())
        self.assertNotIn(b"[assistant] answer", self.output)

    async def test_initial_checkpoint_failure_keeps_ui_alive_and_preserves_old_log(self):
        path = self.root / "interaction.jsonl"
        original = b"old log not yet replaced\n"
        path.write_bytes(original)
        self.start("--prompt", "hello", failure="initial-save")
        await self.wait_output(b"unsaved state remains in memory")
        os.write(self.master, b"\x04")
        await self.wait_exit(expected=1)
        self.assertEqual(path.read_bytes(), original)

    async def test_model_failure_closes_retained_command_and_restores_terminal(self):
        self.start("--prompt", "hello", failure="model")
        await self.wait_output(b"injected model failure")
        os.write(self.master, b"/quit\r")
        await self.wait_exit(expected=1)
        self.assertEqual(json.loads((self.root / "cleanup.json").read_text())["active_before"], 1)

    async def test_tool_result_save_failure_closes_command_and_preserves_pending_call(self):
        self.start("--prompt", "hello", failure="save")
        await self.wait_output(b"unsaved state remains in memory")
        os.write(self.master, b"/exit\r")
        await self.wait_exit(expected=1)
        self.assertEqual(json.loads((self.root / "cleanup.json").read_text())["active_before"], 1)
        saved = load_interaction_save(self.root / "interaction.jsonl")
        self.assertEqual([call.call_id for call in saved.pending_tool_calls()], ["command-1"])

    async def test_render_failure_restores_terminal_and_closes_command(self):
        self.start("--prompt", "hello", failure="render")
        await self.wait_exit(expected=1)
        self.assertEqual(json.loads((self.root / "cleanup.json").read_text())["active_before"], 1)

    async def test_partial_terminal_entry_restores_raw_mode_and_closes_environment(self):
        self.start(failure="attach")
        await self.wait_exit(expected=1, paste_enabled=False)
        self.assertFalse((self.root / "interaction.jsonl").exists())

    async def test_missing_auth_shell_login_then_explicit_query(self):
        self.start("--endpoint-api", "codex", "--model", "test", "--endpoint-auth-file",
                   str(self.root / "auth.json"), "--prompt", "blocked initial", failure="auth")
        await self.wait_output(b"auth needed")
        os.write(self.master, b"\r")
        await self.wait_output(b"Draft was not submitted")
        os.write(self.master, b"\x15\x0b/login\r")
        await self.wait_output(b"Model ready")
        saved = load_interaction_save(self.root / "interaction.jsonl")
        self.assertTrue(saved.items[-1].result.success)
        self.assertIsInstance(saved.items[-1], UserToolResult)
        self.assertFalse(any(isinstance(i, Message) and i.role == "user" for i in saved))
        self.assertNotIn(b"[assistant] answer", self.output)
        os.write(self.master, b"explicit query\r")
        await self.wait_output(b"[assistant] answer-1")
        os.write(self.master, b"/quit\r")
        await self.wait_exit()
        self.assertNotIn(b"FAKE_PTY_SECRET", self.output)
        self.assertNotIn("FAKE_PTY_SECRET", (self.root / "interaction.jsonl").read_text())

    async def test_disabled_default_tools_preserve_config_login_quota_and_chat(self):
        self.start("--enable-default-tools=False", "--enable-workspace=False",
                   "--endpoint-api", "codex", "--model", "test", "--endpoint-auth-file",
                   str(self.root / "auth.json"), failure="auth")
        await self.wait_output(b"Default model tools disabled")
        await self.wait_output(b"auth needed")
        os.write(self.master, b"/config enable_default_tools True\r")
        await self.wait_output(b"Unknown config key")
        os.write(self.master, b"/config.json\r")
        await self.wait_output(b'"enable_workspace": false')
        os.write(self.master, b"/login\r")
        await self.wait_output(b"Model ready")
        os.write(self.master, b"/quota\r")
        await self.wait_output(b"offline quota snapshot")
        os.write(self.master, b"hello\r")
        await self.wait_output(b"Unknown tool: update_plan")
        await self.wait_output(b"[assistant] answer-1")
        os.write(self.master, b"/quit\r")
        await self.wait_exit(default_tools=False)
        saved = load_interaction_save(self.root / "interaction.jsonl")
        self.assertEqual([i.call.name for i in saved if isinstance(i, UserToolCall)],
                         ["config", "login", "quota"])
        self.assertTrue(all(i.result.success for i in saved if isinstance(i, UserToolResult)))
        self.assertFalse(next(i for i in saved if isinstance(i, ToolResult)).success)
        self.assertNotIn(b"FAKE_PTY_SECRET", self.output)
        self.assertNotIn("FAKE_PTY_SECRET", (self.root / "interaction.jsonl").read_text())

    async def test_retry_after_sample_failure_preserves_diagnostics_and_one_user_turn(self):
        self.start("--prompt", "original", "--enable-default-tools=False", failure="retry")
        await self.wait_output(b"ModelTimeoutError: injected sample timeout")
        await self.wait_output(b"Sampling failed. Use /retry to try again.")
        self.assertLess(self.output.index(b"ModelTimeoutError: injected sample timeout"),
                        self.output.index(b"Sampling failed. Use /retry to try again."))
        os.write(self.master, b"/retry\r")
        await self.wait_output(b"[assistant] answer-1")
        os.write(self.master, b"/retry\r")
        await self.wait_output(b"No retryable sampling failure")
        os.write(self.master, b"/quit\r")
        await self.wait_exit(expected=1, default_tools=False)
        saved = load_interaction_save(self.root / "interaction.jsonl")
        self.assertEqual([i for i in saved if isinstance(i, Message) and i.role == "user"],
                         [Message("user", "original")])
        self.assertFalse(any(isinstance(i, (UserToolCall, ToolCall)) for i in saved))
        self.assertIsInstance(saved[-1], TurnSummary)

    async def test_exit_during_real_login_callback_wait_restores_terminal(self):
        self.start("--endpoint-api", "codex", "--model", "test", "--endpoint-auth-file",
                   str(self.root / "auth.json"), failure="auth-cancel")
        await self.wait_output(b"auth needed")
        os.write(self.master, b"/login\r")
        await self.wait_output(b"https://auth.openai.com/oauth/authorize?")
        os.write(self.master, b"\x03")
        await self.wait_exit()
        saved = load_interaction_save(self.root / "interaction.jsonl")
        self.assertFalse(saved.items[-1].result.success)
        self.assertFalse(saved.pending_user_tool_calls())
        self.assertFalse((self.root / "auth.json").exists())
        self.assertNotIn("code_challenge", (self.root / "interaction.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
