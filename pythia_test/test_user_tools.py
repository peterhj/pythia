from __future__ import annotations

from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.interaction_helpers import messages_endpoint
from pythia_test.interaction_helpers import codex_model

import asyncio
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    ChatCompletionsEndpoint, ChatCompletionsModel, CodexAuth, CodexAuthUnavailable,
    CodexResponsesModel, CompactionContextWindowError, CompactionError, CompactionMetadata,
    CompactionResult, CompactionSettings, ContextPrefix,
    ContextValidationError, DefaultEnvironment, Environment,
    Instructions, InteractionConfig, Message, MessagesEndpoint, MessagesModel, InteractionContext, ModelSample,
    ModelSampleBoundary, NothingToCompact, OpaqueCompaction, PiCompactor, SampleParams, Init,
    TokenUsage, ToolCall, ToolResult, Tools, SampleMetadata, TurnSummary, UserInteraction,
    UserInteractionBoundary, UserToolCall, UserToolResult, load_interaction_save,
    render_interaction_items, save_interaction_save,
)
from pythia.interaction import cli, demo, user_tools
from pythia.interaction._cli_editor import Editor
from pythia.interaction.model_config import build_model, supports_account_services
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer


def _records(name="quota", call_id="user_one"):
    return (UserToolCall(ToolCall(name, call_id, "{}")),
            UserToolResult(ToolResult(call_id, "private account snapshot")))


class UserToolValueTests(unittest.TestCase):
    def test_config_command_parses_python_and_json_forms_canonically(self):
        cases = (
            ("/config", "{}"),
            ("/config.json", '{"format":"json"}'),
            (
                "/config enable_workspace False",
                '{"key":"enable_workspace","value":false}',
            ),
            (
                "/config.json enable_auto_compaction True",
                '{"key":"enable_auto_compaction","value":true,"format":"json"}',
            ),
            (
                "/config max_output_tokens None",
                '{"key":"max_output_tokens","value":null}',
            ),
            (
                "/config.json max_samples 3",
                '{"key":"max_samples","value":3,"format":"json"}',
            ),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                intent = user_tools.parse_user_tool(command)
                self.assertEqual(intent.name, "config")
                self.assertEqual(intent.arguments_json, expected)

    def test_config_parser_rejects_unknown_or_invalid_values_without_echo(self):
        for command, private in (
            ("/config unknown_key", "unknown_key"),
            ("/config enable_workspace None", "None"),
            ("/config max_output_tokens True", "True"),
            ("/config max_samples FAKE_SECRET", "FAKE_SECRET"),
            ("/config max_output_tokens 1 extra", "1 extra"),
            ("/config.json\nFAKE_SECRET", "FAKE_SECRET"),
        ):
            with self.subTest(command=command):
                with self.assertRaises(ValueError) as raised:
                    user_tools.parse_user_tool(command)
                self.assertNotIn(private, str(raised.exception))

    def test_config_user_tool_reads_and_writes_without_account_support(self):
        args = cli._build_parser().parse_args([])
        workspace_update = mock.Mock()
        config = InteractionConfig(
            on_enable_workspace=workspace_update,
        )
        environment = user_tools.create_user_environment(
            args,
            notify=mock.Mock(),
            cancel=threading.Event(),
            config=config,
        )

        def execute(command):
            intent = user_tools.parse_user_tool(command)
            return environment.execute_tool_calls((ToolCall(
                intent.name,
                "config-call",
                intent.arguments_json,
            ),)).items[0]

        python_dump = execute("/config")
        self.assertTrue(python_dump.success)
        self.assertEqual(python_dump.output, "\n".join((
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
        )))
        json_dump = execute("/config.json")
        self.assertTrue(json_dump.success)
        self.assertEqual(
            json.loads(json_dump.output),
            {**config.values(), "__init__": config.initial_values()},
        )

        updated = execute("/config max_output_tokens 2048")
        self.assertTrue(updated.success)
        self.assertEqual(updated.output, "# init: max_output_tokens = None\nmax_output_tokens = 2048")
        self.assertEqual(config.get("max_output_tokens"), 2048)
        cleared = execute("/config.json max_output_tokens None")
        self.assertTrue(cleared.success)
        self.assertEqual(
            json.loads(cleared.output),
            {"max_output_tokens": None, "__init__": {"max_output_tokens": None}},
        )
        self.assertIsNone(config.get("max_output_tokens"))
        workspace = execute("/config enable_workspace False")
        self.assertTrue(workspace.success)
        workspace_update.assert_called_once_with(False)

        login = environment.execute_tool_calls((
            ToolCall("login", "login-call", "{}"),
        )).items[0]
        self.assertFalse(login.success)

        forged = environment.execute_tool_calls((ToolCall(
            "config",
            "forged-config",
            '{"key":"max_output_tokens","value":"FAKE_SECRET"}',
        ),)).items[0]
        self.assertFalse(forged.success)
        self.assertNotIn("FAKE_SECRET", forged.output)

    def test_empty_user_tool_arguments_are_elided_only_from_display(self):
        for name in ("quota", "login"):
            for raw_arguments in ("{}", "{ }", " \n{\n}\t "):
                for show_arguments in (False, True):
                    with self.subTest(name=name, raw=raw_arguments, show=show_arguments):
                        call = UserToolCall(ToolCall(name, "user_one", raw_arguments))
                        display = render_interaction_items(
                            (call,), show_generic_arguments=show_arguments,
                        )
                        self.assertEqual(tuple(i.text for i in display), (
                            f"[user-tool-call] {name} (user_one)",
                        ))
                        self.assertEqual(call.call.arguments_json, raw_arguments)

    def test_nonempty_falsy_and_malformed_user_arguments_remain_visible(self):
        cases = (
            (' { "workspace_id": "workspace" } ', '{"workspace_id":"workspace"}'),
            ("[]", "[]"), ("null", "null"), ("false", "false"), ("0", "0"),
            ('""', '""'), ('{"broken"', '{"broken"'),
        )
        for raw_arguments, expected in cases:
            for show_arguments in (False, True):
                with self.subTest(raw=raw_arguments, show=show_arguments):
                    call = UserToolCall(ToolCall("login", "user_one", raw_arguments))
                    display = render_interaction_items(
                        (call,), show_generic_arguments=show_arguments,
                    )
                    self.assertEqual(tuple(i.text for i in display), (
                        "[user-tool-call] login (user_one)", expected,
                    ))
                    self.assertEqual(call.call.arguments_json, raw_arguments)

    def test_model_empty_arguments_and_all_empty_object_results_are_unchanged(self):
        call = ToolCall("lookup", "model_one", "{}")
        self.assertEqual(tuple(i.text for i in render_interaction_items((call,))), (
            "[tool-call] lookup (model_one)",
        ))
        self.assertEqual(tuple(i.text for i in render_interaction_items(
            (call,), show_generic_arguments=True,
        )), ("[tool-call] lookup (model_one)", "{}"))
        result = ToolResult("model_one", "{}")
        for show_arguments in (False, True):
            with self.subTest(show=show_arguments):
                self.assertEqual(tuple(i.text for i in render_interaction_items(
                    (result,), source_calls=(call,), show_generic_arguments=show_arguments,
                )), ("[tool-ret]  lookup (model_one) [ok]\n{}",))
                self.assertEqual(tuple(i.text for i in render_interaction_items(
                    (UserToolResult(result),), source_user_calls=(UserToolCall(call),),
                    show_generic_arguments=show_arguments,
                )), ("[user-tool-ret]  lookup (model_one) [ok]\n{}",))

    def test_empty_arguments_round_trip_and_old_quota_result_replay_without_rewriting(self):
        raw_arguments = " {\n } "
        old_output = "Quota snapshot at earlier\nplan: unavailable"
        original = (Init("old"),
                    UserToolCall(ToolCall("quota", "user_one", raw_arguments)),
                    UserToolResult(ToolResult("user_one", old_output)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interaction.jsonl"
            save_interaction_save(path, InteractionContext(original))
            before = path.read_bytes()
            restored = load_interaction_save(path)
            for _ in range(2):
                self.assertEqual(tuple(i.text for i in render_interaction_items(restored.items)), (
                    "[user-tool-call] quota (user_one)",
                    f"[user-tool-ret]  quota (user_one) [ok]\n{old_output}",
                ))
                self.assertEqual(restored.items, original)
                self.assertEqual(path.read_bytes(), before)

    def test_round_trip_projection_provider_payloads_and_turn_state(self):
        base = (Init("session"), Message("user", "hello"), UserInteractionBoundary(),
                Message("assistant", "answer"),
                SampleMetadata(TokenUsage(), provider_turn_id="turn", provider_turn_state="opaque"),
                ModelSampleBoundary(), TurnSummary(sample_count=1))
        context = InteractionContext((*base, *_records()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            save_interaction_save(path, context)
            self.assertIn('"type": "user_tool_call"', path.read_text())
            self.assertEqual(load_interaction_save(path).items, context.items)
        self.assertEqual(context.model_items(), InteractionContext(base).model_items())
        models = (
            ChatCompletionsModel(chat_endpoint("http://localhost:8000")),
            MessagesModel(messages_endpoint(
                api_url="https://api.anthropic.com",
                model="test",
                max_output_tokens=100,
                api_key="fake",
            )),
            codex_model(model="test", auth=CodexAuth("fake")),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model._build_request_payload(context, (), None),
                                 model._build_request_payload(InteractionContext(base), (), None))
                self.assertNotIn("private account", repr(model._build_request_payload(context, (), None)))
        self.assertIsNone(cli._resume_notice(context))
        self.assertEqual(demo._final_assistant_text(context), "answer")

    def test_origin_aware_display_and_validation(self):
        call, result = _records(call_id="same")
        model_call, model_result = ToolCall("model_tool", "same", "{}"), ToolResult("same", "model result")
        context = InteractionContext((model_call, model_result, call, result))
        display = "\n".join(i.text for i in render_interaction_items(context.items))
        self.assertIn("[tool-ret]  model_tool (same)", display)
        self.assertIn("[user-tool-ret]  quota (same)", display)
        for items in (
            (result,), (call, model_result), (model_call, result),
            (call, call), (call, Message("user", "early")),
            (call, result, call), (model_call, call),
            (ContextPrefix((call, result)),),
        ):
            with self.subTest(items=items), self.assertRaises(ContextValidationError):
                InteractionContext(items)
        pending = InteractionContext((call,))
        self.assertEqual(pending.pending_tool_calls(), ())
        self.assertEqual(pending.pending_user_tool_calls(), (call,))
        with self.assertRaises(ContextValidationError):
            pending.assert_model_ready()
        with self.assertRaises(ValueError):
            UserInteraction((call,))
        with self.assertRaises(Exception):
            ModelSample((call,))
        with self.assertRaises(TypeError):
            UserToolResult(call)

    def test_compaction_excludes_user_tools_but_retains_raw_replay(self):
        model = mock.Mock()
        model.sample.return_value = _answer("summary")
        context = InteractionContext((Message("user", "hello"), *_records()))
        result = PiCompactor(model, keep_recent_tokens=0).compact(context)
        # The request keeps the raw log for provider continuity, but the model
        # sees only the summary request, whose transcript omits user tools.
        submitted = model.sample.call_args.args[0].model_items()
        self.assertFalse(any(isinstance(i, (UserToolCall, UserToolResult)) for i in submitted))
        self.assertIn("[User]: hello", submitted[-1].content)
        self.assertNotIn("private account", repr(submitted))
        context.extend(result.context_items())
        self.assertNotIn("private account", repr(context.model_items()))
        self.assertIn("private account", "\n".join(i.text for i in render_interaction_items(context.items)))

    def test_syntax_no_secret_echo_and_model_dispatch_cannot_call_user_tools(self):
        self.assertEqual(user_tools.parse_user_tool("/login workspace").arguments_json,
                         '{"workspace_id": "workspace"}')
        self.assertEqual(user_tools.parse_user_tool("/compact").arguments_json, "{}")
        for text in ("/login secret.code", "/quota secret-token",
                     "/compact\nsecret-token", "/login\ncode", "/unknown-secret"):
            with self.subTest(text=text), self.assertRaises(ValueError) as error:
                user_tools.parse_user_tool(text)
            self.assertNotIn("secret", str(error.exception))
        for name in ("compact", "config", "login", "quota"):
            result = Environment().execute_tool_calls((ToolCall(name, "model", "{}"),))
            self.assertFalse(result.items[0].success)
            self.assertIn("Unknown tool", result.items[0].output)

    def test_compact_takes_optional_single_line_focus_text(self):
        for command, arguments in (
            ("/compact", {}),
            ("  /compact   ", {}),
            ("/compact keep the parser changes", {"instructions": "keep the parser changes"}),
            ("/compact   spaced   focus  ", {"instructions": "spaced   focus"}),
        ):
            with self.subTest(command=command):
                intent = user_tools.parse_user_tool(command)
                self.assertEqual(intent.name, "compact")
                self.assertEqual(json.loads(intent.arguments_json), arguments)
                self.assertEqual(cli._compact_focus(intent), arguments.get("instructions"))
        with self.assertRaisesRegex(ValueError, "single line"):
            user_tools.parse_user_tool("/compact first\nsecond")
        with self.assertRaisesRegex(ValueError, r"/compact \[focus\]"):
            user_tools.parse_user_tool("/compactify")

    def test_auth_needed_editor_blocks_only_model_submissions_and_quit_bypasses_queue(self):
        state = cli._UIState(ready=True, auth_required=True, editor=Editor("hello", 5))
        state.handle_key("c-m", "")
        self.assertEqual(state.editor.text, "hello")
        self.assertEqual(tuple(state.pending), ())
        for _ in range(9):
            state.editor = Editor("/quota", 6)
            state.handle_key("c-m", "")
        self.assertEqual(len(state.pending), 8)
        self.assertEqual(state.editor.text, "/quota")
        self.assertTrue(all(isinstance(i, user_tools.UserToolIntent) for i in state.pending))
        state.editor = Editor("/quit", 5)
        state.handle_key("c-m", "")
        self.assertTrue(state.closing)
        self.assertTrue(state.login_cancel.is_set())
        self.assertFalse(state.pending)


class AuthConfigurationTests(unittest.TestCase):
    def test_main_opens_only_missing_auth_case_and_rejects_other_config_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            base = ["--endpoint-api", "codex", "--model", "test", "--endpoint-auth-home", directory]
            for extra, code, opens in (([], 0, True), (["--endpoint-url", "bad-url"], 1, False),
                                       (["--request-timeout-seconds", "0"], 1, False)):
                with self.subTest(extra=extra):
                    stream = SimpleNamespace(isatty=lambda: True)
                    with mock.patch.object(cli.sys, "stdin", stream), mock.patch.object(cli.sys, "stdout", stream):
                        with mock.patch.object(cli, "DefaultEnvironment") as environment:
                            with mock.patch.object(cli, "PosixTerminal"):
                                with mock.patch.object(cli, "_run", new_callable=mock.AsyncMock, return_value=0) as run:
                                    with mock.patch("builtins.print"):
                                        self.assertEqual(cli.main([*base, *extra]), code)
                    if opens:
                        self.assertIsNone(run.call_args.args[0])
                        self.assertEqual(environment.call_count, 1)
                    else:
                        environment.assert_not_called()
                        run.assert_not_called()

    def test_non_auth_validation_precedes_missing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ["--endpoint-api", "codex", "--model", "test", "--endpoint-auth-home", directory]
            with self.assertRaises(CodexAuthUnavailable):
                build_model(cli._build_parser().parse_args(argv))
            for extra in (["--endpoint-url", "not-a-url"], ["--request-timeout-seconds", "nan"],
                          ["--endpoint-api-key", "secret"], ["--endpoint-auth-home", ""]):
                with self.subTest(extra=extra):
                    with self.assertRaises(ValueError) as error:
                        build_model(cli._build_parser().parse_args([*argv, *extra]))
                    self.assertNotIsInstance(error.exception, CodexAuthUnavailable)

    def test_capabilities_do_not_follow_wire_api_name_and_demo_stays_fail_fast(self):
        for argv, supported in (
            (["--endpoint-api", "codex", "--model", "test"], True),
            (["--endpoint-api", "codex", "--model", "muse-spark-1.3"], False),
            (["--endpoint-api", "codex", "--model", "test",
              "--endpoint-url", "https://example.org/responses",
              "--endpoint-auth", "none"], False),
            ([], False), (["--endpoint-api", "messages", "--model", "test"], False),
        ):
            self.assertEqual(supports_account_services(cli._build_parser().parse_args(argv)), supported)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(demo, "DefaultEnvironment") as environment:
                with mock.patch("sys.stderr", io.StringIO()):
                    self.assertEqual(demo.main(["--endpoint-api", "codex", "--model", "test",
                                                "--endpoint-auth-home", directory]), 1)
                environment.assert_not_called()


class UserToolControllerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "interaction.jsonl"
        self.auth_path = self.root / "auth.json"
        self.args = cli._build_parser().parse_args([
            "--endpoint-api", "codex", "--model", "test", "--endpoint-auth-file", str(self.auth_path),
        ])

    async def run_cli(self, model, terminal):
        result = await asyncio.wait_for(cli._run(model, Environment(), terminal, self.args, self.path), 4)
        self.assertTrue(terminal.exited)
        return result

    async def test_config_updates_next_turn_sampling_and_is_durable_but_hidden(self):
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status != "idle":
                return
            if step == 0:
                terminal.submit("/config.json")
                step = 1
            elif step == 1 and sum(
                item.text.startswith("[user-tool-ret]  config")
                for item in terminal.items
            ) >= 1:
                terminal.submit("/config max_output_tokens 17")
                step = 2
            elif step == 2 and sum(
                item.text.startswith("[user-tool-ret]  config")
                for item in terminal.items
            ) >= 2:
                terminal.submit("next query")
                step = 3
            elif step == 3 and any(
                item.text == "[assistant] configured"
                for item in terminal.items
            ):
                terminal.key("c-d")

        model = _Model(self.path, _answer("configured"))
        terminal = _Terminal(frame)

        self.assertEqual(await self.run_cli(model, terminal), 0)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][2], SampleParams(max_output_tokens=17, enable_auto_compaction=True))
        self.assertFalse(any(
            isinstance(item, (UserToolCall, UserToolResult))
            for item in model.calls[0][0].model_items()
        ))
        saved = load_interaction_save(self.path)
        config_calls = [
            item for item in saved if isinstance(item, UserToolCall)
            and item.call.name == "config"
        ]
        config_results = [
            item for item in saved if isinstance(item, UserToolResult)
            and item.result.call_id in {call.call.call_id for call in config_calls}
        ]
        self.assertEqual(len(config_calls), 2)
        self.assertEqual(len(config_results), 2)
        # The official Codex route defaults to provider compaction.
        defaults = {
            "enable_workspace": True,
            "max_samples": None,
            "max_output_tokens": None,
            "enable_auto_compaction": True,
            "auto_compact_tokens": None,
            "max_context_tokens": None,
            "compaction_mode": "provider",
            "compaction_keep_recent_tokens": 20000,
            "compaction_max_output_tokens": None,
            "extra_sample_params": {},
        }
        self.assertEqual(
            json.loads(config_results[0].result.output),
            {**defaults, "__init__": defaults},
        )
        self.assertEqual(config_results[1].result.output,
                         "# init: max_output_tokens = None\nmax_output_tokens = 17")

    async def test_config_is_available_without_an_authenticated_model(self):
        submitted = False

        def frame(terminal, editor, status):
            nonlocal submitted
            if status == "auth needed" and not submitted:
                submitted = True
                terminal.submit("/config enable_workspace")
            elif status == "auth needed" and submitted and self.path.exists():
                saved = load_interaction_save(self.path)
                if isinstance(saved.items[-1], UserToolResult):
                    terminal.key("c-d")

        self.assertEqual(
            await self.run_cli(None, _Terminal(frame)),
            0,
        )
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[-1].result.output, "enable_workspace = True")
        self.assertTrue(saved.items[-1].result.success)

    async def test_unfinished_config_is_closed_without_replay(self):
        call = UserToolCall(ToolCall(
            "config",
            "pending-config",
            '{"key":"max_output_tokens","value":17}',
        ))
        save_interaction_save(
            self.path,
            InteractionContext((Init("saved"), Tools(), call)),
        )
        self.args.resume = True
        terminal = _Terminal(
            lambda terminal, editor, status: (
                terminal.key("c-d") if status == "idle" else None
            )
        )

        self.assertEqual(await self.run_cli(_Model(self.path), terminal), 0)

        result = load_interaction_save(self.path).items[-1]
        self.assertIsInstance(result, UserToolResult)
        self.assertFalse(result.result.success)
        self.assertIn("was not rerun", result.result.output)
        self.assertIn("current launch arguments", result.result.output)
        self.assertNotIn("credential", result.result.output)

    async def test_resume_does_not_replay_in_memory_config(self):
        first_submitted = False

        def first_frame(terminal, editor, status):
            nonlocal first_submitted
            if status == "idle" and not first_submitted:
                first_submitted = True
                terminal.submit("/config max_output_tokens 17")
            elif status == "idle" and first_submitted and self.path.exists():
                if isinstance(
                    load_interaction_save(self.path).items[-1],
                    UserToolResult,
                ):
                    terminal.key("c-d")

        self.assertEqual(
            await self.run_cli(_Model(self.path), _Terminal(first_frame)),
            0,
        )

        self.args.resume = True
        second_submitted = False

        def second_frame(terminal, editor, status):
            nonlocal second_submitted
            if status == "idle" and not second_submitted:
                second_submitted = True
                terminal.submit("after restart")
            elif status == "idle" and any(
                item.text == "[assistant] default config"
                for item in terminal.items
            ):
                terminal.key("c-d")

        model = _Model(self.path, _answer("default config"))
        terminal = _Terminal(second_frame)
        self.assertEqual(await self.run_cli(model, terminal), 0)

        self.assertEqual(model.calls[0][2], SampleParams(enable_auto_compaction=True))
        self.assertTrue(any(
            "saved config commands were not replayed" in item.text
            for item in terminal.items
        ))

    async def test_config_disables_auto_compaction_for_the_next_turn(self):
        original = (
            Init("saved"),
            Message("assistant", "old answer"),
            SampleMetadata(TokenUsage(total_tokens=100)),
            ModelSampleBoundary(),
            TurnSummary(sample_count=1, context_tokens=100),
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status != "idle":
                return
            if step == 0:
                terminal.submit("/config enable_auto_compaction False")
                step = 1
            elif step == 1 and any(
                item.text.startswith("[user-tool-ret]  config")
                for item in terminal.items
            ):
                terminal.submit("follow up")
                step = 2
            elif step == 2 and any(
                item.text == "[assistant] uncompacted"
                for item in terminal.items
            ):
                terminal.key("c-d")

        model = _Model(self.path, _answer("uncompacted"))
        self.args.auto_compact_tokens = 100
        with mock.patch.object(cli, "create_default_compactor") as create:
            self.assertEqual(await self.run_cli(model, _Terminal(frame)), 0)

        create.assert_not_called()
        self.assertEqual(
            model.calls[0][2],
            SampleParams(enable_auto_compaction=False, auto_compact_tokens=100),
        )

    async def test_config_max_samples_applies_to_the_next_turn(self):
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status == "idle" and step == 0:
                terminal.submit("/config max_samples 1")
                step = 1
            elif status == "idle" and step == 1 and any(
                item.text.startswith("[user-tool-ret]  config")
                for item in terminal.items
            ):
                terminal.submit("one sample only")
                step = 2
            elif status == "failed":
                terminal.key("c-d")

        model = _Model(
            self.path,
            ModelSample(items=(ToolCall("missing", "model-call", "{}"),)),
        )
        terminal = _Terminal(frame)

        self.assertEqual(await self.run_cli(model, terminal), 1)

        self.assertEqual(len(model.calls), 1)
        self.assertTrue(any(
            "within 1 samples" in item.text
            for item in terminal.items
        ))

    async def test_config_changes_live_workspace_policy_without_rebuilding(self):
        workspace = self.root / "workspace"
        outside = self.root / "outside"
        workspace.mkdir()
        outside.mkdir()
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status != "idle":
                return
            if step == 0:
                terminal.submit("/config enable_workspace False")
                step = 1
            elif step == 1 and any(
                item.text.startswith("[user-tool-ret]  config")
                for item in terminal.items
            ):
                terminal.submit("inspect outside")
                step = 2
            elif step == 2 and any(
                item.text == "[assistant] inspected"
                for item in terminal.items
            ):
                terminal.key("c-d")

        model = _Model(
            self.path,
            ModelSample(items=(ToolCall(
                "exec_command",
                "outside-command",
                json.dumps({
                    "cmd": "pwd",
                    "workdir": str(outside),
                    "yield_time_ms": 1_000,
                }),
            ),)),
            _answer("inspected"),
        )
        with DefaultEnvironment(cwd=workspace) as environment:
            result = await asyncio.wait_for(
                cli._run(
                    model,
                    environment,
                    _Terminal(frame),
                    self.args,
                    self.path,
                ),
                4,
            )

        self.assertEqual(result, 0)
        tool_result = next(
            item for item in load_interaction_save(self.path)
            if isinstance(item, ToolResult)
            and item.call_id == "outside-command"
        )
        self.assertTrue(tool_result.success)
        self.assertIn(str(outside), tool_result.output)

    async def test_quota_plan_propagates_http_to_log_and_replay_without_empty_argument_block(self):
        token = "FAKE_BEARER"
        self.auth_path.write_text(json.dumps({
            "tokens": {"access_token": token, "account_id": "account"},
        }))
        for plan_type, expected_plan in (("prolite", "prolite"), (token, "[redacted]")):
            with self.subTest(plan_type=plan_type):
                self.args.resume = False
                submitted = False
                model = _Model(self.path)
                response = io.BytesIO(json.dumps({
                    "plan_type": plan_type,
                    "rate_limit": {"primary_window": {"used_percent": 25}},
                }).encode())

                def open_request(request, *, timeout):
                    pending = load_interaction_save(self.path).pending_user_tool_calls()
                    self.assertEqual(len(pending), 1)
                    self.assertEqual(pending[0].call.name, "quota")
                    self.assertEqual(pending[0].call.arguments_json, "{}")
                    self.assertEqual(request.get_method(), "GET")
                    self.assertEqual(request.full_url, "https://chatgpt.com/backend-api/wham/usage")
                    self.assertEqual(request.get_header("Authorization"), f"Bearer {token}")
                    self.assertEqual(timeout, self.args.request_timeout_seconds)
                    return response

                def frame(t, editor, status):
                    nonlocal submitted
                    if status == "idle":
                        if not submitted:
                            submitted = True
                            t.submit("/quota")
                        elif any(i.text.startswith("[user-tool-ret]  quota") for i in t.items):
                            t.key("c-d")

                terminal = _Terminal(frame)
                with mock.patch("pythia.interaction._account_http.urllib.request.build_opener") as factory:
                    opener = factory.return_value.open
                    opener.side_effect = open_request
                    self.assertEqual(await self.run_cli(model, terminal), 0)
                    opener.assert_called_once()
                    self.assertTrue(response.closed)
                    saved = load_interaction_save(self.path)
                    self.assertEqual(tuple(type(i) for i in saved), (
                        Init, Tools, UserToolCall, UserToolResult,
                    ))
                    self.assertEqual(saved.items[2].call.arguments_json, "{}")
                    self.assertTrue(saved.items[3].result.success)
                    self.assertIn(f"\nplan: {expected_plan}\n", saved.items[3].result.output)
                    self.assertNotIn(token, self.path.read_text())
                    self.assertNotIn("{}", tuple(i.text for i in terminal.items))
                    transcript = tuple(i for i in terminal.items if not i.text.startswith("[cli]"))
                    self.assertEqual(len(transcript), 3)
                    self.assertEqual(transcript, render_interaction_items(saved.items))

                    before = self.path.read_bytes()
                    self.args.resume = True
                    replay = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
                    with mock.patch.object(cli, "save_interaction_save") as save:
                        self.assertEqual(await self.run_cli(model, replay), 0)
                    save.assert_not_called()
                    opener.assert_called_once()
                    self.assertEqual(self.path.read_bytes(), before)
                    self.assertEqual(tuple(i for i in replay.items if not i.text.startswith("[cli]")), transcript)
                self.assertEqual(model.calls, [])

    async def test_missing_auth_login_then_explicit_query_never_replays_initial_prompt(self):
        self.args.prompt = "initial draft"
        model = _Model(self.path, _answer())
        step = 0

        def frame(t, editor, status):
            nonlocal step
            if status == "auth needed" and step == 0:
                self.assertEqual(editor.text, "initial draft")
                t.key("c-m")  # blocked, no user boundary
                step = 1
            elif status == "auth needed" and step == 1:
                self.assertEqual(editor.text, "initial draft")
                t.submit("/quota")
                step = 2
            elif status == "auth needed" and step == 2 and any(isinstance(i, UserToolResult) for i in load_interaction_save(self.path)):
                t.submit("/login")
                step = 3
            elif status == "idle" and step == 3:
                self.assertEqual(model.calls, [])
                t.submit("explicit query")
                step = 4
            elif status == "idle" and step == 4:
                t.submit("/quit")

        def login(path, **kwargs):
            pending = load_interaction_save(self.path).pending_user_tool_calls()
            self.assertEqual(pending[0].call.name, "login")
            self.assertFalse(any(isinstance(i, UserInteractionBoundary) for i in load_interaction_save(self.path)))
            path.write_text('{"tokens":{"access_token":"FAKE_SECRET","account_id":"account"}}')

        def activate(args):
            saved = load_interaction_save(self.path)
            self.assertIsInstance(saved.items[-1], UserToolResult)
            self.assertTrue(saved.items[-1].result.success)
            return model

        terminal = _Terminal(frame)
        with mock.patch.object(user_tools, "login", side_effect=login):
            with mock.patch.object(cli, "build_model", side_effect=activate):
                self.assertEqual(await self.run_cli(None, terminal), 0)
        saved = load_interaction_save(self.path)
        self.assertEqual([i for i in saved if isinstance(i, Message) and i.role == "user"], [Message("user", "explicit query")])
        self.assertEqual([i.call.name for i in saved if isinstance(i, UserToolCall)], ["quota", "login"])
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn("FAKE_SECRET", self.path.read_text())
        self.assertNotIn("FAKE_SECRET", "\n".join(i.text for i in terminal.items))
        self.assertEqual(model.calls[0][0].items, model.checkpoints[0])

    async def test_resume_unfinished_user_tool_never_executes_and_preserves_model_tail(self):
        original = (Init("old"), Tools(), Message("assistant", "answer"), TurnSummary(), _records("login")[0])
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        model = _Model(self.path)
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(user_tools, "login") as login:
            self.assertEqual(await self.run_cli(model, terminal), 0)
        login.assert_not_called()
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:-1], original)
        self.assertFalse(saved.items[-1].result.success)
        self.assertIsNone(cli._resume_notice(saved))
        self.assertEqual(model.calls, [])
        before = self.path.read_bytes()
        await self.run_cli(model, _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None))
        self.assertEqual(self.path.read_bytes(), before)

    async def test_user_call_and_result_save_failures_block_login_or_activation(self):
        for fail_result in (False, True):
            with self.subTest(fail_result=fail_result):
                attempted = []

                def save(path, context):
                    target = UserToolResult if fail_result else UserToolCall
                    if any(isinstance(i, target) for i in context):
                        raise OSError("disk failed")
                    save_interaction_save(path, context)

                def login(path, **kwargs):
                    attempted.append(True)

                def frame(t, editor, status):
                    if status == "auth needed":
                        t.submit("/login")
                    if status == "failed":
                        t.key("c-d")

                with mock.patch.object(cli, "save_interaction_save", side_effect=save):
                    with mock.patch.object(user_tools, "login", side_effect=login):
                        with mock.patch.object(cli, "build_model") as activate:
                            self.assertEqual(await self.run_cli(None, _Terminal(frame)), 1)
                activate.assert_not_called()
                self.assertEqual(len(attempted), int(fail_result))
                saved = load_interaction_save(self.path)
                self.assertEqual(bool(saved.pending_user_tool_calls()), fail_result)

    async def test_quit_during_login_cancels_wait_and_checkpoints_safe_result(self):
        entered = threading.Event()

        def login(path, *, notify, cancel, **kwargs):
            notify("fake transient challenge")
            entered.set()
            if not cancel.wait(2):
                raise AssertionError("login was not cancelled")
            raise user_tools.AccountServiceError("Login cancelled.")

        def frame(t, editor, status):
            if status == "auth needed" and not entered.is_set():
                t.submit("/login")
            if any("fake transient challenge" in i.text for i in t.items):
                t.key("c-c")

        with mock.patch.object(user_tools, "login", side_effect=login):
            self.assertEqual(await self.run_cli(None, _Terminal(frame)), 0)
        saved = load_interaction_save(self.path)
        self.assertFalse(saved.pending_user_tool_calls())
        self.assertFalse(saved.items[-1].result.success)
        self.assertNotIn("fake transient challenge", self.path.read_text())

    async def test_login_failure_and_activation_failure_are_safe_and_do_not_sample(self):
        for activation in (False, True):
            with self.subTest(activation=activation):
                calls = []

                def login(path, **kwargs):
                    calls.append(True)
                    if not activation:
                        raise RuntimeError("FAKE_SECRET")

                def frame(t, editor, status):
                    if status == "auth needed":
                        t.submit("/quit" if calls else "/login")

                terminal = _Terminal(frame)
                with mock.patch.object(user_tools, "login", side_effect=login):
                    with mock.patch.object(cli, "build_model", side_effect=RuntimeError("FAKE_SECRET")):
                        self.assertEqual(await self.run_cli(None, terminal), int(activation))
                self.assertNotIn("FAKE_SECRET", self.path.read_text())
                self.assertNotIn("FAKE_SECRET", "\n".join(i.text for i in terminal.items))
                self.assertFalse(any(isinstance(i, UserInteractionBoundary) for i in load_interaction_save(self.path)))

    async def test_authenticated_prompt_starting_with_login_is_literal_text(self):
        self.args.prompt = "/login"
        model = _Model(self.path, _answer())
        with mock.patch.object(user_tools, "login") as login:
            await self.run_cli(model, _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None))
        login.assert_not_called()
        self.assertIn(Message("user", "/login"), model.calls[0][0].items)

    async def test_quota_queued_during_sample_runs_after_turn_and_before_follow_up(self):
        self.auth_path.write_text('{"tokens":{"access_token":"fake","account_id":"account"}}')
        entered, release = threading.Event(), threading.Event()
        step = 0

        def first(context):
            entered.set()
            if not release.wait(2):
                raise AssertionError("sample not released")
            return _answer("first")

        def quota(auth, **kwargs):
            items = load_interaction_save(self.path).items
            self.assertIsInstance(items[-1], UserToolCall)
            self.assertIsInstance(items[-2], TurnSummary)
            self.assertNotIn(Message("user", "follow-up"), items)
            return "quota snapshot"

        def frame(t, editor, status):
            nonlocal step
            if entered.is_set() and step == 0:
                t.submit("/quota")
                t.submit("follow-up")
                step = 1
            if "queued=2" in status:
                release.set()
            if status == "idle" and any(i.text == "[assistant] second" for i in t.items):
                t.key("c-d")

        self.args.prompt = "first"
        model = _Model(self.path, first, _answer("second"))
        try:
            with mock.patch.object(user_tools, "query_quota", side_effect=quota) as query:
                self.assertEqual(await self.run_cli(model, _Terminal(frame)), 0)
        finally:
            release.set()
        query.assert_called_once()
        self.assertEqual(len(model.calls), 2)
        saved = load_interaction_save(self.path)
        self.assertEqual(len([i for i in saved if isinstance(i, TurnSummary)]), 2)

    async def test_same_account_login_rebinds_without_sampling_or_changing_provider_state(self):
        metadata = SampleMetadata(TokenUsage(), provider_turn_id="turn", provider_turn_state="state")
        original = (Init("old"), Tools(), Message("assistant", "answer"), metadata, TurnSummary())
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        model = _Model(self.path)
        model.endpoint = SimpleNamespace(account_id="account")
        rebuilt = []
        submitted = False

        def login(path, **kwargs):
            self.assertEqual(kwargs["expected_account"], "account")
            path.write_text('{"tokens":{"access_token":"new-token","account_id":"account"}}')

        def activate(args):
            candidate = build_model(args)
            rebuilt.append(candidate)
            return candidate

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle":
                if rebuilt:
                    t.submit("/quit")
                elif not submitted:
                    submitted = True
                    t.submit("/login")

        with mock.patch.object(user_tools, "login", side_effect=login):
            with mock.patch.object(cli, "build_model", side_effect=activate):
                self.assertEqual(await self.run_cli(model, _Terminal(frame)), 0)
        self.assertEqual(len(rebuilt), 1)
        self.assertEqual(rebuilt[0].endpoint.bearer_token, "new-token")
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:len(original)], original)
        self.assertEqual(saved.model_items(), InteractionContext(original).model_items())
        self.assertEqual(len(saved.items), len(original) + 2)
        self.assertEqual(model.calls, [])

    async def test_compact_records_audit_pair_and_installs_checkpoint_atomically(self):
        original = (
            Init("old"),
            Tools(),
            Message("assistant", "previous answer"),
            TurnSummary(sample_count=1),
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        model = _Model(self.path)
        observations = []
        parameters = []

        class FakeCompactor:
            def compact(inner_self, source, *, tools=(), sample_params=None, instructions=None):
                persisted = load_interaction_save(self.path)
                observations.append((source.items, tuple(tools), persisted.items))
                parameters.append((sample_params, instructions))
                return CompactionResult(
                    items=(ContextPrefix((
                        Message("user", "retained request"),
                        OpaqueCompaction.from_responses("private checkpoint"),
                    )),),
                    usage=TokenUsage(100, 8, 108, 75),
                    protocol="responses_compaction_v2",
                    elapsed_seconds=86.25,
                )

        submitted = False

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle" and not submitted:
                submitted = True
                t.submit("/compact")
            elif status == "idle" and submitted and self.path.exists():
                if isinstance(load_interaction_save(self.path).items[-1], CompactionMetadata):
                    t.key("c-d")

        terminal = _Terminal(frame)
        compactor = FakeCompactor()
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor) as create:
            self.assertEqual(await self.run_cli(model, terminal), 0)
        # The official Codex route defaults to provider compaction.
        create.assert_called_once_with(model, CompactionSettings(mode="provider"))
        self.assertEqual(parameters, [(SampleParams(enable_auto_compaction=True), None)])
        self.assertEqual(len(observations), 1)
        source_items, tools, persisted_items = observations[0]
        self.assertEqual(source_items, original)
        self.assertEqual(tools, ())
        self.assertIsInstance(persisted_items[-1], UserToolCall)
        self.assertEqual(persisted_items[-1].call.name, "compact")
        self.assertEqual(persisted_items[-1].call.arguments_json, "{}")
        self.assertEqual(InteractionContext(source_items).pending_user_tool_calls(), ())

        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:len(original)], original)
        call, result, checkpoint, metadata = saved.items[-4:]
        self.assertIsInstance(call, UserToolCall)
        self.assertIsInstance(result, UserToolResult)
        self.assertTrue(result.result.success)
        self.assertEqual(result.result.call_id, call.call.call_id)
        self.assertIsInstance(checkpoint, ContextPrefix)
        self.assertIsInstance(metadata, CompactionMetadata)
        self.assertIn("remote opaque checkpoint", result.result.output)
        self.assertNotIn("input=100", result.result.output)
        self.assertEqual(metadata.usage, TokenUsage(100, 8, 108, 75))
        self.assertEqual(metadata.protocol, "responses_compaction_v2")
        self.assertEqual(metadata.elapsed_seconds, 86.25)
        self.assertEqual(saved.model_items(), checkpoint.prefix_items)
        self.assertFalse(any(
            isinstance(item, (UserToolCall, UserToolResult))
            for item in saved.model_items()
        ))
        self.assertNotIn("input=100", repr(saved.model_items()))
        self.assertEqual(model.calls, [])
        self.assertEqual(
            len([item for item in saved if isinstance(item, TurnSummary)]),
            1,
        )
        transcript = "\n".join(item.text for item in terminal.items)
        self.assertIn("[user-tool-call] compact", transcript)
        self.assertIn("[user-tool-ret]  compact", transcript)
        self.assertIn("[context prefix]", transcript)
        self.assertIn(
            "[compaction] protocol=responses_compaction_v2 input=100 "
            "output=8 total=108 cached=75 elapsed=86.25s",
            transcript,
        )
        self.assertNotIn("private checkpoint", transcript)
        self.assertIsNone(cli._resume_notice(saved))

        # Replaying a completed manual checkpoint neither reruns compaction nor
        # emits the incomplete-model-turn warning used for provider pauses.
        before = self.path.read_bytes()
        replay = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(cli, "create_default_compactor") as replay_create:
            with mock.patch.object(cli, "save_interaction_save") as save:
                self.assertEqual(await self.run_cli(model, replay), 0)
        replay_create.assert_not_called()
        save.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(
            "without recorded turn completion" in item.text
            for item in replay.items
        ))

    def compact_frame(self, command):
        submitted = False

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle" and not submitted:
                submitted = True
                t.submit(command)
            elif status == "idle" and submitted and self.path.exists():
                items = load_interaction_save(self.path).items
                if isinstance(items[-1], (CompactionMetadata, UserToolResult)):
                    t.key("c-d")
        return frame

    async def test_compact_focus_reaches_a_pi_summary_request(self):
        original = (
            Init("old"),
            Message("user", "old request"),
            UserInteractionBoundary(),
            Message("assistant", "previous answer"),
            SampleMetadata(TokenUsage(total_tokens=10)),
            ModelSampleBoundary(),
            TurnSummary(sample_count=1),
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        self.args.compaction_mode = "pi"
        self.args.compaction_keep_recent_tokens = 0
        self.args.compaction_max_output_tokens = 512
        model = _Model(self.path, _answer("## Goal\nShip it."))
        terminal = _Terminal(self.compact_frame("/compact keep file paths"))
        self.assertEqual(await self.run_cli(model, terminal), 0)

        [(request, tools, params)] = model.calls
        self.assertEqual(tools, ())
        self.assertEqual(params, SampleParams(max_output_tokens=512, enable_auto_compaction=False))
        prompt = request.model_items()[-1].content
        self.assertTrue(prompt.startswith("<conversation>\n[User]: old request"))
        self.assertTrue(prompt.endswith("\n\nAdditional focus: keep file paths"))
        saved = load_interaction_save(self.path)
        call, result, checkpoint, metadata = saved.items[-4:]
        self.assertEqual(json.loads(call.call.arguments_json), {"instructions": "keep file paths"})
        self.assertTrue(result.result.success)
        self.assertEqual(result.result.output, "Context compacted using a pi summary checkpoint.")
        self.assertIsInstance(checkpoint, ContextPrefix)
        self.assertEqual(metadata.protocol, "pi")
        texts = [item.text for item in terminal.items]
        self.assertIn("[context prefix] 1 item", texts)
        self.assertIn("## Goal\nShip it.", texts)

    async def test_compact_reports_nothing_to_compact_as_unsuccessful(self):
        original = (Init("old"), Message("assistant", "previous answer"), TurnSummary(sample_count=1))
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        self.args.compaction_mode = "pi"
        model = _Model(self.path)
        self.assertEqual(await self.run_cli(model, _Terminal(self.compact_frame("/compact"))), 0)
        self.assertEqual(model.calls, [])
        saved = load_interaction_save(self.path)
        self.assertIsInstance(saved.items[-1], UserToolResult)
        self.assertFalse(saved.items[-1].result.success)
        self.assertEqual(
            saved.items[-1].result.output,
            "Nothing to compact: the context fits in compaction_keep_recent_tokens.",
        )
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in saved))

    async def test_compact_shows_a_context_window_failure(self):
        original = (Init("old"), Message("assistant", "previous answer"), TurnSummary(sample_count=1))
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        message = (
            "summary request for the history (412 items, ~905,000 estimated tokens) "
            "exceeded the model's context window; pi compaction sends each part in one "
            "request and does not split it. Lower compaction_max_output_tokens, or raise "
            "compaction_keep_recent_tokens to summarize less."
        )
        compactor = mock.Mock()
        compactor.compact.side_effect = CompactionContextWindowError(message)
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(
                await self.run_cli(_Model(self.path), _Terminal(self.compact_frame("/compact"))), 0,
            )
        result = load_interaction_save(self.path).items[-1]
        self.assertFalse(result.result.success)
        self.assertEqual(result.result.output, f"Compaction failed: {message}")

    async def test_compact_busy_status_is_compacting_with_live_elapsed_time(self):
        original = (
            Init("old"),
            Message("assistant", "previous answer"),
            TurnSummary(sample_count=1),
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        entered = threading.Event()
        release = threading.Event()
        statuses = []

        class BlockingCompactor:
            def compact(inner_self, source, *, tools=(), sample_params=None, instructions=None):
                del source, tools, sample_params, instructions
                entered.set()
                if not release.wait(2):
                    raise AssertionError("test did not release compaction")
                return CompactionResult(
                    (ContextPrefix((Message("user", "summary"),)),),
                    protocol="pi",
                )

        submitted = False

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle" and not submitted:
                submitted = True
                t.submit("/compact")
            elif entered.is_set() and status.startswith("compacting "):
                statuses.append(status)
                release.set()
            elif status == "idle" and submitted and self.path.exists():
                if isinstance(
                    load_interaction_save(self.path).items[-1],
                    CompactionMetadata,
                ):
                    t.key("c-d")

        try:
            with mock.patch.object(
                cli,
                "create_default_compactor",
                return_value=BlockingCompactor(),
            ):
                self.assertEqual(
                    await self.run_cli(_Model(self.path), _Terminal(frame)),
                    0,
                )
        finally:
            release.set()
        self.assertTrue(statuses)
        self.assertRegex(statuses[0], r"^compacting \d+s$")

    async def test_compact_uses_remote_v2_by_default_for_codex_responses(self):
        original = (
            Init("session"),
            Message("user", "retain this request"),
            Message("assistant", "old answer"),
            TurnSummary(sample_count=1),
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        observed = []

        class Response:
            status = 200
            headers = {}

            def __init__(inner_self):
                payloads = (
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "compaction",
                            "encrypted_content": "server checkpoint",
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "response-compact",
                            "usage": {
                                "input_tokens": 20,
                                "output_tokens": 3,
                                "total_tokens": 23,
                            },
                        },
                    },
                )
                inner_self.lines = []
                for payload in payloads:
                    inner_self.lines.extend((
                        f"data: {json.dumps(payload)}\n".encode(),
                        b"\n",
                    ))
                inner_self.closed = False

            def __iter__(inner_self):
                return iter(inner_self.lines)

            def close(inner_self):
                inner_self.closed = True

        response = Response()

        def open_request(request, *, timeout):
            saved = load_interaction_save(self.path)
            payload = json.loads(request.data)
            observed.append((saved.items, payload, dict(request.header_items()), timeout))
            return response

        model = codex_model(
            model="test",
            auth=CodexAuth("token", "account"),
            opener=open_request,
        )
        submitted = False

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle" and not submitted:
                submitted = True
                t.submit("/compact")
            elif status == "idle" and submitted and observed:
                if isinstance(load_interaction_save(self.path).items[-1], CompactionMetadata):
                    t.key("c-d")

        self.assertEqual(await self.run_cli(model, _Terminal(frame)), 0)
        self.assertEqual(len(observed), 1)
        request_items, payload, headers, timeout = observed[0]
        self.assertIsInstance(request_items[-1], UserToolCall)
        self.assertEqual(payload["input"][-1], {"type": "compaction_trigger"})
        self.assertNotIn("user_tool", json.dumps(payload))
        self.assertEqual(
            {name.lower(): value for name, value in headers.items()}[
                "x-codex-beta-features"
            ],
            "remote_compaction_v2",
        )
        saved = load_interaction_save(self.path)
        self.assertEqual(tuple(type(item) for item in saved.items[-4:]), (
            UserToolCall, UserToolResult, ContextPrefix, CompactionMetadata,
        ))
        metadata = saved.items[-1]
        self.assertEqual(metadata.protocol, "responses_compaction_v2")
        self.assertEqual(metadata.usage, TokenUsage(20, 3, 23, 0))
        self.assertIsNotNone(metadata.elapsed_seconds)
        self.assertEqual(metadata.provider_response_id, "response-compact")
        self.assertEqual(saved.model_items(), (
            Message("user", "retain this request"),
            OpaqueCompaction.from_responses("server checkpoint"),
        ))
        self.assertNotIn("server checkpoint", "\n".join(
            item.text for item in render_interaction_items(saved.items)
        ))
        self.assertTrue(response.closed)

    async def test_failed_compact_records_result_and_keeps_effective_context(self):
        original = (
            Init("old"),
            Message("assistant", "previous answer"),
            TurnSummary(sample_count=1),
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        model = _Model(self.path)
        submitted = False

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle" and not submitted:
                submitted = True
                t.submit("/compact")
            elif status == "idle" and submitted:
                saved = load_interaction_save(self.path)
                if isinstance(saved.items[-1], UserToolResult):
                    t.key("c-d")

        compactor = mock.Mock()
        compactor.compact.side_effect = CompactionError("invalid remote checkpoint")
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self.run_cli(model, _Terminal(frame)), 0)
        saved = load_interaction_save(self.path)
        self.assertIsInstance(saved.items[-2], UserToolCall)
        self.assertIsInstance(saved.items[-1], UserToolResult)
        self.assertFalse(saved.items[-1].result.success)
        self.assertIn("invalid remote checkpoint", saved.items[-1].result.output)
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in saved))
        self.assertFalse(any(isinstance(item, CompactionMetadata) for item in saved))
        self.assertEqual(saved.model_items(), InteractionContext(original).model_items())
        self.assertEqual(model.calls, [])

    async def test_resume_never_reruns_an_unfinished_compact(self):
        call = UserToolCall(ToolCall("compact", "user_pending", "{}"))
        original = (
            Init("old"),
            Tools(),
            Message("assistant", "previous answer"),
            TurnSummary(sample_count=1),
            call,
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(cli, "create_default_compactor") as create:
            self.assertEqual(await self.run_cli(_Model(self.path), terminal), 0)
        create.assert_not_called()
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:-1], original)
        self.assertIsInstance(saved.items[-1], UserToolResult)
        self.assertFalse(saved.items[-1].result.success)
        self.assertIn("was not rerun", saved.items[-1].result.output)
        self.assertIn("no durable compaction checkpoint", saved.items[-1].result.output)
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in saved))

    async def test_compact_result_checkpoint_save_failure_is_not_rerun(self):
        original = (
            Init("old"),
            Tools(),
            Message("assistant", "previous answer"),
            TurnSummary(sample_count=1),
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        model = _Model(self.path)
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult((ContextPrefix((
            Message("user", "summary"),
        )),))
        real_save = save_interaction_save

        def fail_checkpoint(path, context):
            if any(isinstance(item, ContextPrefix) for item in context):
                raise OSError("disk failed")
            real_save(path, context)

        submitted = False

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle" and not submitted:
                submitted = True
                t.submit("/compact")
            elif status == "failed":
                t.key("c-d")

        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            with mock.patch.object(cli, "save_interaction_save", side_effect=fail_checkpoint):
                self.assertEqual(await self.run_cli(model, _Terminal(frame)), 1)
        compactor.compact.assert_called_once()
        interrupted = load_interaction_save(self.path)
        self.assertEqual(interrupted.items[:-1], original)
        self.assertIsInstance(interrupted.items[-1], UserToolCall)
        self.assertEqual(interrupted.pending_user_tool_calls(), (interrupted.items[-1],))
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in interrupted))

        # Startup closes the pending audit record but cannot infer or recreate
        # the checkpoint that failed to persist.
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(cli, "create_default_compactor") as create:
            self.assertEqual(await self.run_cli(model, terminal), 0)
        create.assert_not_called()
        recovered = load_interaction_save(self.path)
        self.assertIsInstance(recovered.items[-1], UserToolResult)
        self.assertFalse(recovered.items[-1].result.success)
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in recovered))

    async def test_compact_without_an_active_model_is_a_recorded_failure(self):
        submitted = False

        def frame(t, editor, status):
            nonlocal submitted
            if status == "auth needed" and not submitted:
                submitted = True
                t.submit("/compact")
            elif status == "auth needed" and submitted and self.path.exists():
                if isinstance(load_interaction_save(self.path).items[-1], UserToolResult):
                    t.key("c-d")

        with mock.patch.object(cli, "create_default_compactor") as create:
            self.assertEqual(await self.run_cli(None, _Terminal(frame)), 0)
        create.assert_not_called()
        saved = load_interaction_save(self.path)
        self.assertEqual(tuple(type(item) for item in saved), (
            Init, Tools, UserToolCall, UserToolResult,
        ))
        self.assertFalse(saved.items[-1].result.success)
        self.assertIn("authentication needed", saved.items[-1].result.output.lower())
        self.assertEqual(saved.model_items(), ())

    async def test_follow_up_queued_after_compact_uses_only_prefix_context(self):
        original = (
            Init("old"),
            Message("assistant", "discarded old answer"),
            TurnSummary(sample_count=1),
        )
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume = True
        model = _Model(self.path, _answer("answer after compact"))
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult((ContextPrefix((
            Message("user", "compacted state"),
        )),))
        submitted = False

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle" and not submitted:
                submitted = True
                t.submit("/compact")
                t.submit("follow up")
            elif status == "idle" and submitted and model.calls:
                t.key("c-d")

        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self.run_cli(model, _Terminal(frame)), 0)
        self.assertEqual(len(model.calls), 1)
        materialized = model.calls[0][0].model_items()
        self.assertIn(Message("user", "compacted state"), materialized)
        self.assertIn(Message("user", "follow up"), materialized)
        self.assertNotIn(Message("assistant", "discarded old answer"), materialized)
        self.assertFalse(any(
            isinstance(
                item,
                (CompactionMetadata, UserToolCall, UserToolResult, ContextPrefix),
            )
            for item in materialized
        ))
        saved = load_interaction_save(self.path)
        self.assertEqual(
            [item.call.name for item in saved if isinstance(item, UserToolCall)],
            ["compact"],
        )
        self.assertEqual(
            len([item for item in saved if isinstance(item, TurnSummary)]),
            2,
        )

    async def test_auth_needed_instructions_resume_does_not_sample_or_add_a_user_boundary(self):
        original = (Init("old"), Message("assistant", "answer"), TurnSummary())
        save_interaction_save(self.path, InteractionContext(original))
        self.args.resume, self.args.instructions = True, "new instructions"
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "auth needed" else None)
        self.assertEqual(await self.run_cli(None, terminal), 0)
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items, (*original, Tools(), Instructions("new instructions")))


if __name__ == "__main__":
    unittest.main()
