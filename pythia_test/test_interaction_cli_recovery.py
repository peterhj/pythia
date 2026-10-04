"""Offline recovery and persistence-failure acceptance for the working-tree CLI."""

from __future__ import annotations

import threading
import unittest
import weakref
from unittest import mock

from pythia.interaction import ContextPrefix
from pythia.interaction import DefaultEnvironment
from pythia.interaction import Environment
from pythia.interaction import Init
from pythia.interaction import Instructions
from pythia.interaction import Message
from pythia.interaction import ModelAuthenticationError
from pythia.interaction import InteractionContext
from pythia.interaction import ModelFailure
from pythia.interaction import ModelResponseError
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import SaveError
from pythia.interaction import TokenUsage
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolOutcome
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import Tools
from pythia.interaction import SampleMetadata
from pythia.interaction import TurnSummary
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import cli
from pythia.interaction import load_interaction_save
from pythia.interaction import render_interaction_items
from pythia.interaction import save_interaction_save
from pythia_test.test_interaction_cli import _ControllerTestCase
from pythia_test.test_interaction_cli import _Model
from pythia_test.test_interaction_cli import _Terminal
from pythia_test.test_interaction_cli import _answer


def _quit_when_idle(terminal, editor, status):
    if status == "idle":
        terminal.key("c-d")


class CLIRecoveryTests(_ControllerTestCase):
    def test_stop_reason_is_not_part_of_the_durable_sample_projection(self):
        items = (Message("assistant", "text that could be truncated"),)
        self.assertEqual(
            ModelSample(items=items, stop_reason="end_turn").context_items(),
            ModelSample(items=items, stop_reason="max_tokens").context_items(),
        )
        items = (OpaqueCompaction.from_messages("checkpoint"),)
        self.assertEqual(
            ModelSample(items=items, stop_reason="compaction").context_items(),
            ModelSample(items=items).context_items(),
        )

    async def test_resume_tail_matrix_replays_raw_history_without_saving_or_sampling(self):
        old = (Init("old"), Tools(), Message("assistant", "historical answer"),
               TurnSummary(sample_count=1))
        tails = (
            ((), None),
            ((Message("user", "unfinished"), UserInteractionBoundary()), "user submission"),
            ((ToolCall("record", "one", "{}"), ModelSampleBoundary(),
              ToolResult("one", "saved result")), "tool results"),
            ((Message("assistant", "possibly truncated"), ModelSampleBoundary()), "assistant output"),
            ((OpaqueCompaction.from_messages("secret messages checkpoint"),
              ModelSampleBoundary()), "compaction checkpoint"),
            ((OpaqueCompaction.from_responses("secret responses checkpoint"),), "compaction checkpoint"),
            ((ContextPrefix((Message("assistant", "replacement, not transcript"),)),),
             "context-prefix checkpoint"),
            ((Instructions("changed"),), "instructions update"),
            ((Reasoning("thinking"), ModelSampleBoundary()), "incomplete model output"),
        )
        for tail, notice in tails:
            with self.subTest(tail=tail):
                context = InteractionContext((*old, *tail))
                save_interaction_save(self.path, context)
                original_bytes = self.path.read_bytes()
                model = _Model(self.path)
                terminal = _Terminal(_quit_when_idle)
                with mock.patch.object(cli, "save_interaction_save") as save:
                    self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
                save.assert_not_called()
                self.assertEqual(self.path.read_bytes(), original_bytes)
                self.assertEqual(model.calls, [])
                self.assertEqual(
                    tuple(i for i in terminal.items if not i.text.startswith("[cli]")),
                    render_interaction_items(context.items),
                )
                notices = "\n".join(i.text for i in terminal.items if i.text.startswith("[cli]"))
                self.assertEqual("No model request was started" in notices, notice is not None)
                if notice:
                    self.assertIn(notice, notices)
                self.assertNotIn("secret", "\n".join(i.text for i in terminal.items))

    async def test_incomplete_response_contributes_completed_items_and_diagnostics(self):
        failure = ModelFailure(
            category="stream_closed",
            message="Responses stream closed before response.completed",
            provider="codex",
            model="model",
            auth_source="codex_file",
            request_id="request-1",
            event_count=2,
            completed_item_count=1,
        )
        error = ModelResponseError(
            failure.message,
            failure=failure,
            completed_items=(Message("assistant", "partial but complete item"),),
        )
        terminal = _Terminal(
            lambda terminal, _editor, status: (
                terminal.key("c-d") if status == "failed" else None
            )
        )
        model = _Model(self.path, error)

        self.assertEqual(
            await self._run(model, terminal, ["--prompt", "hello"]),
            1,
        )

        saved = load_interaction_save(self.path)
        self.assertEqual(
            saved.items[-3:],
            (
                Message("assistant", "partial but complete item"),
                failure,
                ModelSampleBoundary(),
            ),
        )
        self.assertFalse(any(isinstance(i, SampleMetadata) for i in saved))
        self.assertFalse(any(isinstance(i, TurnSummary) for i in saved))
        displayed = tuple(item.text for item in terminal.items)
        self.assertIn("[assistant] partial but complete item", displayed)
        self.assertTrue(any(text.startswith("[model failure]") for text in displayed))

    async def test_unauthorized_failure_blocks_more_model_queries_until_login(self):
        failure = ModelFailure(
            category="authentication",
            message="Codex Responses HTTP 401: authentication failed",
            provider="codex",
            model="model",
            auth_source="codex_file",
            http_status=401,
        )
        error = ModelAuthenticationError(failure.message, failure=failure)
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status == "failed" and step == 0:
                terminal.submit("must remain a draft")
                step = 1
            elif status == "failed" and step == 1:
                self.assertEqual(editor.text, "must remain a draft")
                terminal.key("c-d")

        model = _Model(self.path, error)
        terminal = _Terminal(frame)

        self.assertEqual(
            await self._run(model, terminal, ["--prompt", "hello"]),
            1,
        )
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(
            any("authentication needed" in item.text.lower() for item in terminal.items)
        )

    async def test_tool_call_recovered_from_failed_stream_is_not_executed(self):
        call = ToolCall("record", "partial-call", "{}")
        failure = ModelFailure(
            category="stream_closed",
            message="Responses stream closed before response.completed",
            completed_item_count=1,
        )
        error = ModelResponseError(
            failure.message,
            failure=failure,
            completed_items=(call,),
        )
        executed = []

        def record(arguments, *, timeout_seconds=None):
            del arguments, timeout_seconds
            executed.append(True)
            return ToolOutcome("executed")

        environment = Environment(
            (Tool(ToolSpec("record", "", {}), record),)
        )
        terminal = _Terminal(
            lambda terminal, _editor, status: (
                terminal.key("c-d") if status == "failed" else None
            )
        )

        self.assertEqual(
            await self._run(
                _Model(self.path, error),
                terminal,
                ["--prompt", "hello"],
                environment,
            ),
            1,
        )

        self.assertEqual(executed, [])
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.pending_tool_calls(), ())
        result = next(
            item for item in reversed(saved.items)
            if isinstance(item, ToolResult)
        )
        self.assertEqual(result.call_id, call.call_id)
        self.assertFalse(result.success)
        self.assertIn("Not executed", result.output)

    async def test_partial_batch_recovery_is_durable_per_call_and_idempotent(self):
        calls = tuple(ToolCall("record", name, "{}") for name in ("one", "two", "three"))
        original = (Init("old"), Tools(), *calls, ModelSampleBoundary(),
                    ToolResult("one", "already saved"))
        save_interaction_save(self.path, InteractionContext(original))
        environment = mock.Mock(spec=Environment, tool_specs=())
        writes = []

        def save(path, context):
            writes.append((load_interaction_save(path).items, context.items))
            save_interaction_save(path, context)

        terminal = _Terminal(_quit_when_idle)
        with mock.patch.object(cli, "save_interaction_save", side_effect=save):
            self.assertEqual(await self._run(_Model(self.path), terminal, ["--resume"], environment), 0)
        environment.execute_tool_calls.assert_not_called()
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:len(original)], original)
        self.assertEqual(len(writes), 2)
        self.assertEqual(writes[0][0], original)
        self.assertEqual(writes[1][0], writes[0][1])
        self.assertEqual([i.call_id for i in saved.items[-2:]], ["two", "three"])
        for result in saved.items[-2:]:
            self.assertFalse(result.success)
            self.assertIn("may already have produced side effects", result.output)
        for name in ("two", "three"):
            self.assertEqual(sum(f"record ({name}) [error]" in i.text for i in terminal.items), 1)
        with mock.patch.object(cli, "save_interaction_save") as save_again:
            await self._run(_Model(self.path), _Terminal(_quit_when_idle), ["--resume"], environment)
        save_again.assert_not_called()
        environment.execute_tool_calls.assert_not_called()

    async def test_pending_recovery_with_instructions_only_is_explicit_continuation(self):
        original = (Init("old"), Tools(), Message("user", "old query"),
                    ToolCall("record", "one", "{}"))
        save_interaction_save(self.path, InteractionContext(original))
        environment = Environment()
        model = _Model(self.path, _answer())
        with mock.patch.object(environment, "execute_tool_calls") as execute:
            await self._run(model, _Terminal(_quit_when_idle),
                            ["--resume", "--instructions", ""], environment)
        execute.assert_not_called()
        items = model.calls[0][0].items
        self.assertEqual(items[:len(original)], original)
        self.assertIsInstance(items[-2], ToolResult)
        self.assertFalse(items[-2].success)
        self.assertEqual(items[-1], Instructions(""))
        self.assertEqual(len(model.calls), 1)

    async def test_resume_preserves_provider_state_compaction_and_follow_up_once(self):
        reasoning = Reasoning("", encrypted_content="encrypted", content_signature="signature")
        metadata = SampleMetadata(TokenUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                                provider_session_id="session-id", provider_turn_id="turn-id",
                                provider_turn_state="opaque-turn-state")
        call = ToolCall("record", "call_codex_id", "{}")
        original = (Init("old"), Tools(), Message("user", "old query"), reasoning,
                    OpaqueCompaction.from_responses("opaque-checkpoint"), call, metadata,
                    ModelSampleBoundary())
        save_interaction_save(self.path, InteractionContext(original))
        model = _Model(self.path, _answer())
        terminal = _Terminal(_quit_when_idle)
        await self._run(model, terminal, ["--resume", "--prompt", "follow-up\nunchanged"])
        received = model.calls[0][0]
        self.assertEqual(received.items[:len(original)], original)
        self.assertEqual(received.items[-3].call_id, call.call_id)
        self.assertFalse(received.items[-3].success)
        self.assertEqual(received.items[-2:], (
            Message("user", "follow-up\nunchanged"), UserInteractionBoundary(),
        ))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual([i for i in received if isinstance(i, SampleMetadata)], [metadata])
        self.assertNotIn("[cli]", repr(received.items))
        self.assertNotIn("opaque-turn-state", "\n".join(i.text for i in terminal.items))

    async def test_new_environment_does_not_restore_plan_or_old_command_handles(self):
        with DefaultEnvironment(cwd=self.path.parent) as previous:
            calls = (
                ToolCall("update_plan", "plan", '{"plan":[{"step":"old","status":"pending"}]}'),
                ToolCall("exec_command", "start", '{"cmd":"read line","yield_time_ms":0}'),
            )
            results = previous.execute_tool_calls(calls)
            self.assertEqual(previous.command_runtime.active_session_ids, (1,))
            self.assertIsNotNone(previous.latest_plan)
        original = (Init("old"), *calls, *results.items,
                    Message("assistant", "previous"), TurnSummary(sample_count=1))
        save_interaction_save(self.path, InteractionContext(original))
        with DefaultEnvironment(cwd=self.path.parent) as environment:
            model = _Model(self.path, ModelSample(items=(ToolCall(
                "write_stdin", "lost", '{"session_id":1,"chars":"must not write"}'
            ),)), _answer())
            terminal = _Terminal(_quit_when_idle)
            await self._run(model, terminal, ["--resume", "--prompt", "check handle"], environment)
            self.assertIsNone(environment.latest_plan)
            self.assertEqual(environment.command_runtime.active_session_ids, ())
            result = model.calls[-1][0].items[-1]
            self.assertFalse(result.success)
            self.assertIn("unknown session_id: 1", result.output)
            self.assertTrue(any("not restored" in i.text for i in terminal.items))


class CLIPersistenceFailureTests(_ControllerTestCase):
    async def test_save_failure_matrix_retains_unsaved_context_and_stops_all_work(self):
        # Fresh: init, user, sample, tool 1, tool 2, final sample, summary.
        # Resume: recovery 1, recovery 2, instructions, user.
        for resume, count in ((False, 7), (True, 4)):
            for fail_at in range(1, count + 1):
                with self.subTest(resume=resume, fail_at=fail_at):
                    calls = (ToolCall("record", "one", "{}"), ToolCall("record", "two", "{}"))
                    original = (Init("old"), *calls) if resume else (Init("old"),)
                    save_interaction_save(self.path, InteractionContext(original))
                    executions, attempts, references = [], [], []
                    durable = self.path.read_bytes()
                    real_checkpoint = cli._checkpoint

                    async def checkpoint(context, state, path):
                        references.append(weakref.ref(context))
                        await real_checkpoint(context, state, path)

                    def save(path, context):
                        nonlocal durable
                        attempts.append(context.items)
                        if len(attempts) == fail_at:
                            raise SaveError("injected disk failure")
                        save_interaction_save(path, context)
                        durable = path.read_bytes()

                    def record(arguments, *, timeout_seconds=None):
                        executions.append(load_interaction_save(self.path).items)
                        return ToolOutcome("completed effect")

                    step = 0

                    def frame(terminal, editor, status):
                        nonlocal step
                        if status == "failed":
                            # This is a weak reference to the live context, not the save's copy.
                            self.assertIsNotNone(references[-1]())
                            self.assertEqual(references[-1]().items, attempts[-1])
                            step += 1
                            if step == 1:
                                terminal.submit("must not run")
                            else:
                                self.assertEqual(editor.text, "must not run")
                                terminal.key("c-d")

                    environment = Environment((Tool(ToolSpec("record", "", {}), record),))
                    model = _Model(self.path, ModelSample(items=calls), _answer())
                    terminal = _Terminal(frame)
                    argv = ["--prompt", "hello"]
                    if resume:
                        argv += ["--resume", "--instructions", "override"]
                    else:
                        argv += ["--resume=False"]
                    with mock.patch.object(cli, "_checkpoint", side_effect=checkpoint):
                        with mock.patch.object(cli, "save_interaction_save", side_effect=save):
                            self.assertEqual(await self._run(model, terminal, argv, environment), 1)
                    self.assertEqual(len(attempts), fail_at)
                    self.assertEqual(self.path.read_bytes(), durable)
                    self.assertNotIn("must not run", repr(attempts))
                    self.assertTrue(any("unsaved state remains in memory" in i.text for i in terminal.items))
                    if resume:
                        self.assertEqual(executions, [])
                        self.assertEqual(model.calls, [])
                    else:
                        self.assertEqual(len(model.calls), int(fail_at >= 3) + int(fail_at >= 6))
                        self.assertEqual(len(executions), int(fail_at >= 4) + int(fail_at >= 5))
                    # Unsaved summaries/results must not masquerade as saved output.
                    shown = tuple(i for i in terminal.items if not i.text.startswith("[cli]"))
                    self.assertEqual(shown, render_interaction_items(load_interaction_save(self.path).items))

    async def test_failed_checkpoint_during_exit_reports_failure_without_another_effect(self):
        entered, release = threading.Event(), threading.Event()

        def save(path, context):
            if any(isinstance(i, ModelSampleBoundary) for i in context):
                entered.set()
                if not release.wait(2):
                    raise AssertionError("checkpoint not released")
                raise SaveError("disk failed during exit")
            save_interaction_save(path, context)

        def frame(terminal, editor, status):
            if entered.is_set():
                terminal.key("c-c")
            if status.startswith("closing"):
                release.set()

        model = _Model(self.path, ModelSample(items=(ToolCall("record", "one", "{}"),)))
        terminal = _Terminal(frame)
        environment = Environment()
        try:
            with mock.patch.object(cli, "save_interaction_save", side_effect=save):
                with mock.patch.object(environment, "execute_tool_calls") as execute:
                    self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], environment), 1)
        finally:
            release.set()
        execute.assert_not_called()
        self.assertEqual(load_interaction_save(self.path).items[-1], UserInteractionBoundary())
        self.assertTrue(any("disk failed during exit" in i.text for i in terminal.items))

    async def test_effect_failure_discards_queued_queries_and_never_retries_old_tools(self):
        entered, release = threading.Event(), threading.Event()
        environment = Environment()
        calls = (ToolCall("record", "one", "{}"), ToolCall("record", "two", "{}"))

        def fail(calls):
            entered.set()
            if not release.wait(2):
                raise AssertionError("tool not released")
            raise RuntimeError("lost tool outcome")

        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if step == 0 and entered.is_set():
                terminal.submit("queued, not permission to retry")
                step = 1
            if "queued=1" in status:
                release.set()
            if status == "failed" and step == 1:
                terminal.submit("explicit new query")
                step = 2
            if status == "idle" and step == 2:
                terminal.key("c-d")

        model = _Model(self.path, ModelSample(items=calls), _answer())
        try:
            with mock.patch.object(environment, "execute_tool_calls", side_effect=fail) as execute:
                self.assertEqual(await self._run(model, _Terminal(frame), ["--prompt", "hello"], environment), 1)
        finally:
            release.set()
        execute.assert_called_once_with((calls[0],))
        received = model.calls[-1][0].items
        self.assertNotIn(Message("user", "queued, not permission to retry"), received)
        self.assertEqual(received[-2:], (Message("user", "explicit new query"), UserInteractionBoundary()))
        results = [i for i in received if isinstance(i, ToolResult)]
        self.assertEqual([r.call_id for r in results], ["one", "two"])
        self.assertTrue(all(not r.success and "was not rerun" in r.output for r in results))

    async def test_sample_limit_and_empty_answer_fail_without_summary_or_retry(self):
        for sample, options in (
            (ModelSample(items=(OpaqueCompaction.from_messages("checkpoint"),), stop_reason="compaction"),
             ["--max-samples", "1"]),
            (ModelSample(items=(Message("assistant", " "),)), []),
        ):
            with self.subTest(sample=sample):
                model = _Model(self.path, sample)
                terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "failed" else None)
                self.assertEqual(await self._run(model, terminal, ["--prompt", "hello", *options]), 1)
                self.assertEqual(len(model.calls), 1)
                self.assertFalse(any(isinstance(i, TurnSummary) for i in load_interaction_save(self.path)))


if __name__ == "__main__":
    unittest.main()
