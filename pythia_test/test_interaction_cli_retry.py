"""Explicit, CLI-local retries: no new user turn and no old tool replay."""

from collections import deque
import json
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    CompactionContextWindowError, CompactionMetadata, CompactionResult,
    CompactionSettings, ContextPrefix, Environment, Init, InteractionContext,
    Message, ModelAuthenticationError, ModelContextWindowError, ModelFailure,
    ModelResponseError, NothingToCompact,
    ModelSample, ModelSampleBoundary, ModelTimeoutError, OpaqueCompaction,
    Reasoning, SampleParams, SaveError, Tool, ToolCall, ToolOutcome,
    ToolResult, ToolSpec, TurnSummary, UserInteractionBoundary, UserToolCall,
    cli, load_interaction_save, save_interaction_save,
)
from pythia.interaction import user_tools
from pythia.interaction._cli_editor import Editor
from pythia_test.test_interaction_cli import _ControllerTestCase, _Model, _Terminal, _answer


HINT = "[cli] Sampling failed. Use /retry to try again."
AUTH_HINT = "[cli] Model authentication needed; use /login."


def submit(state, text="/retry"):
    state.editor = Editor(text, len(text))
    state.handle_key("c-m", "\r")


def retry_then_quit(count=1):
    retries = 0

    def frame(terminal, editor, status):
        nonlocal retries
        if status == "failed" and retries < count:
            retries += 1
            terminal.submit("/retry")
        elif status == "idle":
            terminal.key("c-d")
    return frame


class RetryInputTests(unittest.TestCase):
    def test_retry_is_typed_deduplicated_and_not_auth_gated(self):
        ticket = cli._RetryIntent()
        state = cli._UIState(ready=True, phase="failed", retry=ticket, auth_required=True)
        submit(state)
        submit(state)
        self.assertEqual(tuple(state.pending), (ticket,))
        self.assertTrue(any("already queued" in item.text for item in state.displays))
        state.request_exit()
        self.assertIsNone(state.retry)
        self.assertFalse(state.pending)

    def test_retry_rejects_busy_missing_failure_and_invalid_arguments(self):
        cases = [("idle", None, "/retry", "No retryable sampling failure")]
        for phase in ("sampling", "saving", "compacting", "loading model", "tool: exec_command"):
            cases.append((phase, cli._RetryIntent(), "/retry", "work is in progress"))
        for text in ("/retry FAKE_SECRET", "/retry\nFAKE_SECRET", "/retry\r"):
            cases.append(("failed", cli._RetryIntent(), text, "Usage: /retry"))
        for phase, ticket, text, notice in cases:
            with self.subTest(phase=phase, text=text):
                state = cli._UIState(ready=True, phase=phase, retry=ticket)
                submit(state, text)
                self.assertFalse(state.pending)
                displayed = "\n".join(item.text for item in state.displays)
                self.assertIn(notice, displayed)
                self.assertNotIn("FAKE_SECRET", displayed)

    def test_full_queue_and_checkpoint_failure_preserve_unaccepted_draft(self):
        for persistence_failed in (False, True):
            with self.subTest(persistence_failed=persistence_failed):
                state = cli._UIState(ready=True, phase="failed", retry=cli._RetryIntent(),
                                     persistence_failed=persistence_failed)
                state.pending.extend(str(i) for i in range(8))
                submit(state)
                self.assertEqual(len(state.pending), 8)
                self.assertEqual(state.editor.text, "/retry")


class RetryControllerTests(_ControllerTestCase):
    async def test_errors_are_preserved_then_hint_and_retry_does_not_add_user_input(self):
        for outcome, diagnostic in (
            (ModelTimeoutError("timed out"), "ModelTimeoutError: timed out"),
            (RuntimeError("adapter failed"), "RuntimeError: adapter failed"),
            (_answer(" "), "RuntimeError: model returned no final assistant text"),
        ):
            with self.subTest(diagnostic=diagnostic):
                model = _Model(self.path, outcome, _answer("recovered"))
                terminal = _Terminal(retry_then_quit())
                with mock.patch.object(cli, "build_model") as build:
                    self.assertEqual(await self._run(model, terminal, [
                        "--prompt", "original", "--enable-default-tools=False",
                    ]), 1)  # Existing sticky failure exit status is unchanged.
                build.assert_not_called()
                self.assertEqual(len(model.calls), 2)
                self.assertTrue(all(tools == () for _, tools, _ in model.calls))
                saved = load_interaction_save(self.path)
                self.assertEqual([i for i in saved if isinstance(i, Message) and i.role == "user"],
                                 [Message("user", "original")])
                self.assertEqual(sum(isinstance(i, UserInteractionBoundary) for i in saved), 1)
                self.assertFalse(any(isinstance(i, UserToolCall) for i in saved))
                shown = [item.text for item in terminal.items]
                self.assertEqual(shown.count(HINT), 1)
                self.assertLess(shown.index("[cli] " + diagnostic), shown.index(HINT))
                self.assertIsInstance(saved[-1], TurnSummary)

    async def test_repeated_failures_require_fresh_explicit_retries(self):
        model = _Model(self.path, ModelTimeoutError("first"), ModelTimeoutError("second"), _answer())
        terminal = _Terminal(retry_then_quit(2))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(sum(item.text == HINT for item in terminal.items), 2)

    async def test_retry_uses_a_fresh_budget_but_does_not_bypass_its_limit(self):
        model = _Model(self.path, ModelTimeoutError("first"),
                       ModelSample((ToolCall("noop", "new", "{}"),)))
        effects = []
        environment = Environment((Tool(ToolSpec("noop", "", {}),
            lambda args, **kwargs: (effects.append(True) or ToolOutcome("ok"))),))
        commands = deque(("/retry", "/retry", "/quit"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft())
                             if s == "failed" and commands else None)
        self.assertEqual(await self._run(model, terminal, [
            "--prompt", "hello", "--max-samples", "1",
        ], environment), 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(effects, [True])
        shown = [i.text for i in terminal.items]
        self.assertEqual(shown.count(HINT), 1)
        self.assertTrue(any("within 1 samples" in text for text in shown))
        self.assertTrue(any("No retryable" in text for text in shown))

    async def test_partial_output_is_retained_old_calls_stay_closed_and_new_calls_execute(self):
        effects = []
        environment = Environment((Tool(ToolSpec("record", "", {}),
            lambda args, **kwargs: (effects.append(args["label"]) or ToolOutcome("done"))),))
        old = ToolCall("record", "old", '{"label":"old"}')
        partial = ToolCall("record", "partial", '{"label":"must not run"}')
        new = ToolCall("record", "new", '{"label":"new"}')
        failure = ModelFailure(category="stream_closed", message="stream interrupted")
        error = ModelResponseError("stream interrupted", failure=failure,
                                   completed_items=(Message("assistant", "partial output"), partial))
        model = _Model(self.path, ModelSample((old,)), error, ModelSample((new,)), _answer())
        self.assertEqual(await self._run(model, _Terminal(retry_then_quit()),
                                        ["--prompt", "hello"], environment), 1)
        self.assertEqual(effects, ["old", "new"])
        retry_context = model.calls[2][0]
        self.assertIn(failure, retry_context)
        self.assertIn(Message("assistant", "partial output"), retry_context)
        self.assertEqual([i.call_id for i in retry_context if isinstance(i, ToolResult)],
                         ["old", "partial"])
        self.assertFalse(next(i for i in retry_context if isinstance(i, ToolResult)
                              and i.call_id == "partial").success)
        self.assertFalse(retry_context.pending_tool_calls())

    async def test_ticket_survives_config_quota_login_and_manual_compaction(self):
        auth = self.path.parent / "auth.json"
        auth.write_text(json.dumps({"tokens": {"access_token": "FAKE_SECRET", "account_id": "same"}}))
        model = _Model(self.path, ModelTimeoutError("first"), _answer())
        model.endpoint = SimpleNamespace(account_id="same")
        commands = deque(("/config max_output_tokens 17", "/quota", "/login", "/compact", "/retry", "/quit"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft())
                             if s in {"failed", "idle"} and commands else None)
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult((ContextPrefix((Message("user", "summary"),)),))
        with mock.patch.object(user_tools, "query_quota", return_value="quota"), \
                mock.patch.object(user_tools, "login") as login, \
                mock.patch.object(cli, "build_model", return_value=model) as build, \
                mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, [
                "--prompt", "hello", "--endpoint-api", "codex", "--model", "test",
                "--endpoint-auth-file", str(auth), "--enable-default-tools=False",
            ]), 1)
        login.assert_called_once()
        build.assert_called_once()  # /login activation only; retry reuses it.
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1][2], SampleParams(max_output_tokens=17, enable_auto_compaction=True))
        self.assertEqual(model.calls[1][1], ())
        self.assertIn(Message("user", "summary"), model.calls[1][0].model_items())
        self.assertEqual([i.call.name for i in load_interaction_save(self.path) if isinstance(i, UserToolCall)],
                         ["config", "quota", "login", "compact"])

    async def test_new_query_invalidates_queued_retry_and_success_disables_retry(self):
        step = 0

        def frame(t, e, s):
            nonlocal step
            if s == "failed" and step == 0:
                t.submit("new task")
                t.submit("/retry")
                step = 1
            elif s == "idle" and step == 1 and any("no longer applies" in i.text for i in t.items):
                t.submit("/retry")
                step = 2
            elif s == "idle" and step == 2 and any("No retryable" in i.text for i in t.items):
                t.key("c-d")

        model = _Model(self.path, ModelTimeoutError("first"), _answer())
        self.assertEqual(await self._run(model, _Terminal(frame), ["--prompt", "old task"]), 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([i.content for i in model.calls[1][0] if isinstance(i, Message) and i.role == "user"],
                         ["old task", "new task"])

    async def test_resumed_failure_is_not_retryable_and_prompt_retry_is_literal(self):
        original = InteractionContext((Init("old"), Message("user", "hello"),
            ModelFailure(category="timeout", message="old failure"), ModelSampleBoundary()))
        save_interaction_save(self.path, original)
        before = self.path.read_bytes()
        commands = deque(("/retry", "/quit"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft()) if s == "idle" and commands else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(any("No retryable" in i.text for i in terminal.items))
        model = _Model(self.path, _answer())
        self.assertEqual(await self._run(model, _Terminal(retry_then_quit()), ["--prompt", "/retry"]), 0)
        self.assertIn(Message("user", "/retry"), model.calls[0][0])

    async def test_non_sampling_failures_do_not_offer_retry(self):
        for kind in ("tool", "compaction", "limit"):
            with self.subTest(kind=kind):
                sample = (ModelSample((OpaqueCompaction.from_messages("summary"),), stop_reason="compaction")
                          if kind == "limit" else ModelSample((ToolCall("record", "call", "{}"),)))
                model = _Model(self.path, sample)
                environment = Environment()
                submitted = False

                def frame(t, e, s):
                    nonlocal submitted
                    if s == "failed":
                        if not submitted:
                            submitted = True
                            t.submit("/retry")
                        else:
                            t.key("c-d")

                compactor = mock.Mock()
                compactor.compact.side_effect = ModelTimeoutError("compaction failed")
                terminal = _Terminal(frame)
                with mock.patch.object(environment, "execute_tool_calls", side_effect=RuntimeError("lost tool result")) as execute, \
                        mock.patch.object(cli, "auto_compaction_due", return_value=kind == "compaction"), \
                        mock.patch.object(cli, "create_default_compactor", return_value=compactor):
                    self.assertEqual(await self._run(model, terminal, [
                        "--prompt", "hello", "--max-samples", "1", "--auto-compact-tokens", "100",
                    ], environment), 1)
                self.assertEqual(execute.call_count, int(kind == "tool"))
                self.assertEqual(len(model.calls), int(kind != "compaction"))
                self.assertNotIn(HINT, [i.text for i in terminal.items])
                self.assertTrue(any("No retryable" in i.text for i in terminal.items))

    async def test_checkpoint_failure_blocks_retry_even_during_failure_cleanup(self):
        failure = ModelFailure(category="timeout", message="timed out")
        model = _Model(self.path, ModelTimeoutError("timed out", failure=failure))
        submitted = False

        def frame(t, e, s):
            nonlocal submitted
            if s == "failed":
                if not submitted:
                    submitted = True
                    t.submit("/retry")
                else:
                    t.key("c-d")

        def save(path, context):
            if failure in context:
                raise SaveError("disk failed")
            save_interaction_save(path, context)

        terminal = _Terminal(frame)
        with mock.patch.object(cli, "save_interaction_save", side_effect=save):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn(HINT, [i.text for i in terminal.items])
        self.assertTrue(any("no further work" in i.text for i in terminal.items))
        self.assertNotIn(failure, load_interaction_save(self.path))

    async def test_queued_query_diagnostic_precedes_retry_hint(self):
        entered, release = threading.Event(), threading.Event()
        step = 0

        def fail(context):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release sample")
            raise ModelTimeoutError("timed out")

        def frame(t, e, s):
            nonlocal step
            if entered.is_set() and step == 0:
                t.submit("queued task")
                step = 1
            if "queued=1" in s:
                release.set()
            if s == "failed":
                t.key("c-d")

        terminal = _Terminal(frame)
        try:
            self.assertEqual(await self._run(_Model(self.path, fail), terminal, ["--prompt", "hello"]), 1)
        finally:
            release.set()
        shown = [i.text for i in terminal.items]
        self.assertLess(shown.index("[cli] ModelTimeoutError: timed out"), shown.index(HINT))
        self.assertLess(next(n for n, text in enumerate(shown) if "Queued queries discarded" in text), shown.index(HINT))


class RetryAuthenticationTests(_ControllerTestCase):
    def failed_model(self, *, source="codex_file", account="same", partial=()):
        failure = ModelFailure(category="authentication", message="credentials rejected", auth_source=source)
        model = _Model(self.path, ModelAuthenticationError(failure.message, failure=failure, completed_items=partial))
        model.endpoint = SimpleNamespace(account_id=account)
        return model

    async def test_auth_retry_reloads_without_login_and_preserves_account(self):
        for source in ("codex_file", "environment", "static"):
            with self.subTest(source=source):
                model = self.failed_model(source=source)
                replacement = _Model(self.path, _answer())
                replacement.endpoint = SimpleNamespace(account_id="same")
                terminal = _Terminal(retry_then_quit())
                with mock.patch.object(cli, "build_model", return_value=replacement) as build, \
                        mock.patch.object(user_tools, "login") as login:
                    self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
                build.assert_called_once()
                login.assert_not_called()
                self.assertEqual(len(replacement.calls), 1)
                self.assertEqual(replacement.calls[0][0].items, replacement.checkpoints[0])
                shown = [i.text for i in terminal.items]
                self.assertNotIn(HINT, shown)
                if source == "codex_file":
                    self.assertLess(shown.index("[cli] ModelAuthenticationError: credentials rejected"), shown.index(AUTH_HINT))
                else:
                    self.assertTrue(any("restart" in text for text in shown))

    async def test_reload_failure_keeps_retry_available_and_withholds_details(self):
        model = self.failed_model()
        replacement = _Model(self.path, _answer())
        replacement.endpoint = SimpleNamespace(account_id="same")
        commands = deque(("/retry", "/retry"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft())
                             if s in {"failed", "auth needed"} and commands
                             else t.key("c-d") if s == "idle" else None)
        with mock.patch.object(cli, "build_model", side_effect=[ValueError("FAKE_SECRET"), replacement]) as build:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(len(replacement.calls), 1)
        self.assertNotIn("FAKE_SECRET", "\n".join(i.text for i in terminal.items))
        self.assertNotIn("FAKE_SECRET", self.path.read_text())

    async def test_login_can_recover_auth_but_does_not_automatically_retry(self):
        model = self.failed_model()
        replacement = _Model(self.path, _answer())
        replacement.endpoint = SimpleNamespace(account_id="same")
        commands = deque(("/login", "/retry", "/quit"))

        def frame(t, e, s):
            if s in {"failed", "idle"} and commands:
                command = commands.popleft()
                if command == "/retry":
                    self.assertEqual(replacement.calls, [])
                t.submit(command)

        with mock.patch.object(user_tools, "login") as login, \
                mock.patch.object(cli, "build_model", return_value=replacement) as build:
            self.assertEqual(await self._run(model, _Terminal(frame), [
                "--prompt", "hello", "--endpoint-api", "codex", "--model", "test",
                "--endpoint-auth-file", str(self.path.parent / "auth.json"),
            ]), 1)
        login.assert_called_once()
        build.assert_called_once()  # Retry reuses the model activated by /login.
        self.assertEqual(len(replacement.calls), 1)
        self.assertEqual([i for i in replacement.calls[0][0] if isinstance(i, Message) and i.role == "user"],
                         [Message("user", "hello")])

    async def test_reloading_a_different_or_unknown_account_does_not_sample(self):
        for account in ("same", None):
            with self.subTest(account=account):
                model = self.failed_model(account=account, partial=(Reasoning("", encrypted_content="opaque"),))
                replacement = _Model(self.path, _answer())
                replacement.endpoint = SimpleNamespace(account_id="different")
                submitted = False

                def frame(t, e, s):
                    nonlocal submitted
                    if s == "failed" and not submitted:
                        submitted = True
                        t.submit("/retry")
                    elif s == "auth needed":
                        t.key("c-d")

                with mock.patch.object(cli, "build_model", return_value=replacement) as build:
                    self.assertEqual(await self._run(model, _Terminal(frame), [
                        "--prompt", "hello", "--endpoint-api", "codex", "--model", "test",
                    ]), 1)
                self.assertEqual(build.call_count, int(account is not None))
                self.assertEqual(replacement.calls, [])

    async def test_exit_during_auth_reload_does_not_sample_the_replacement(self):
        model = self.failed_model()
        replacement = _Model(self.path, _answer())
        replacement.endpoint = SimpleNamespace(account_id="same")
        entered, release = threading.Event(), threading.Event()
        submitted = False

        def build(args):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release model reload")
            return replacement

        def frame(t, e, s):
            nonlocal submitted
            if s == "failed" and not submitted:
                submitted = True
                t.submit("/retry")
            if entered.is_set():
                t.key("c-c")
            if s.startswith("closing"):
                release.set()

        try:
            with mock.patch.object(cli, "build_model", side_effect=build):
                self.assertEqual(await self._run(model, _Terminal(frame), ["--prompt", "hello"]), 1)
        finally:
            release.set()
        self.assertTrue(entered.is_set())
        self.assertEqual(replacement.calls, [])


OVERFLOW = ModelFailure(category="context_window", message="Messages HTTP 400: context window exceeded")
OVERFLOW_NOTICE = "[cli] Model context window exceeded; compacting before one retry."


def overflow():
    return ModelContextWindowError("prompt is too long", failure=OVERFLOW)


def quit_when_settled(terminal, editor, status):
    if status in {"idle", "failed"}:
        terminal.key("c-d")


class OverflowRecoveryTests(_ControllerTestCase):
    def compactor(self, **kwargs):
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult(
            (ContextPrefix((Message("user", "summary"),)),), protocol="pi",
        )
        for name, value in kwargs.items():
            setattr(compactor.compact, name, value)
        return compactor

    async def test_context_window_error_compacts_once_and_retries_the_sample(self):
        model = _Model(self.path, overflow(), _answer("recovered"))
        compactor = self.compactor()
        terminal = _Terminal(quit_when_settled)
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor) as create:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        create.assert_called_once_with(model, CompactionSettings())
        compactor.compact.assert_called_once()
        self.assertEqual(compactor.compact.call_args.kwargs["sample_params"], model.calls[0][2])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1][0].model_items(), (Message("user", "summary"),))
        saved = load_interaction_save(self.path)
        # The failure is recorded before the compaction that recovers it.
        failure_index = saved.items.index(OVERFLOW)
        self.assertIsInstance(saved.items[failure_index + 1], ModelSampleBoundary)
        self.assertIsInstance(saved.items[failure_index + 2], ContextPrefix)
        self.assertIsInstance(saved.items[failure_index + 3], CompactionMetadata)
        self.assertIsInstance(saved.items[-1], TurnSummary)
        texts = [item.text for item in terminal.items]
        self.assertIn(OVERFLOW_NOTICE, texts)
        self.assertNotIn(HINT, texts)

    async def test_second_overflow_in_a_turn_fails_and_arms_retry(self):
        model = _Model(self.path, overflow(), overflow(), _answer("after retry"))
        compactor = self.compactor()
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, _Terminal(retry_then_quit()), ["--prompt", "hello"]), 1)
        # One recovery per turn; the explicit /retry starts a new attempt.
        compactor.compact.assert_called_once()
        self.assertEqual(len(model.calls), 3)

    async def test_nothing_to_compact_keeps_the_sampling_error(self):
        model = _Model(self.path, overflow())
        compactor = self.compactor(side_effect=NothingToCompact(
            "the context fits in compaction_keep_recent_tokens",
        ))
        terminal = _Terminal(quit_when_settled)
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 1)
        texts = [item.text for item in terminal.items]
        self.assertIn("[cli] ModelContextWindowError: prompt is too long", texts)
        self.assertIn(HINT, texts)
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in load_interaction_save(self.path)))

    async def test_failed_recovery_compaction_fails_the_turn_with_retry(self):
        message = (
            "summary request for the history (412 items, ~905,000 estimated tokens) "
            "exceeded the model's context window"
        )
        model = _Model(self.path, overflow(), _answer("after retry"))
        compactor = self.compactor(side_effect=CompactionContextWindowError(message))
        terminal = _Terminal(retry_then_quit())
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        texts = [item.text for item in terminal.items]
        self.assertIn(f"[cli] CompactionContextWindowError: {message}", texts)
        self.assertIn(HINT, texts)
        self.assertEqual(len(model.calls), 2)

    async def test_no_recovery_without_host_owned_automatic_compaction(self):
        for argv, owner in (
            (["--enable-auto-compaction=False"], "host"),
            ([], "server"),
        ):
            with self.subTest(argv=argv, owner=owner):
                model = _Model(self.path, overflow())
                model.auto_compaction_owner = owner
                compactor = self.compactor()
                terminal = _Terminal(quit_when_settled)
                with mock.patch.object(cli, "create_default_compactor", return_value=compactor) as create:
                    self.assertEqual(await self._run(model, terminal, ["--prompt", "hello", *argv]), 1)
                create.assert_not_called()
                self.assertIn(HINT, [item.text for item in terminal.items])

    async def test_threshold_nothing_to_compact_samples_normally(self):
        model = _Model(self.path, _answer("done"))
        compactor = self.compactor(side_effect=NothingToCompact("nothing precedes the recent tail"))
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, _Terminal(quit_when_settled), [
                "--prompt", "hello", "--auto-compact-tokens", "1",
            ]), 0)
        compactor.compact.assert_called_once()
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(any(isinstance(item, CompactionMetadata) for item in load_interaction_save(self.path)))

    async def test_failed_threshold_compaction_fails_the_turn_without_retry(self):
        message = "summary request for the history (3 items, ~9 estimated tokens) exceeded the model's context window"
        model = _Model(self.path)
        compactor = self.compactor(side_effect=CompactionContextWindowError(message))
        terminal = _Terminal(quit_when_settled)
        with mock.patch.object(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, [
                "--prompt", "hello", "--auto-compact-tokens", "1",
            ]), 1)
        self.assertEqual(model.calls, [])
        texts = [item.text for item in terminal.items]
        self.assertIn(f"[cli] CompactionContextWindowError: {message}", texts)
        self.assertNotIn(HINT, texts)


if __name__ == "__main__":
    unittest.main()
