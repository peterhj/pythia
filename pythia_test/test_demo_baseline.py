"""Executable demo baselines for the interaction CLI port's stage 0.

These exercise the existing one-shot demo, not an unimplemented REPL. The CLI's
empty-editor default, timer, exit keys, and visual pre-fill are later-stage tests.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import DisplayItem
from pythia.interaction import DEFAULT_REQUEST_TIMEOUT_SECONDS
from pythia.interaction import Init
from pythia.interaction import Instructions
from pythia.interaction import Message
from pythia.interaction import InteractionContext
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import Reasoning
from pythia.interaction import ResolvedSamplingParams
from pythia.interaction import TokenUsage
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import SampleMetadata
from pythia.interaction import TurnSummary
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import demo
from pythia.interaction import load_interaction_save
from pythia.interaction import save_interaction_save


DEMO_ARGUMENT_DEFAULTS = {
    "model_api": None,
    "endpoint_url": None,
    "endpoint_model": None,
    "endpoint_auth": None,
    "model_catalog": None,
    "no_user_model_catalog": False,
    "list_models": False,
    "debug_save_model_binding": False,
    "request_params": None,
    "model": None,
    "api_key": None,
    "codex_home": None,
    "codex_auth_file": None,
    "cwd": ".",
    "enable_auto_compaction": True,
    "enable_workspace": True,
    "enable_experimental_media": False,
    "auto_compact_tokens": None,
    "max_context_tokens": None,
    "max_samples": None,
    "max_output_tokens": None,
    "request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
    "prompt": None,
    "instructions": None,
    "save_path": Path("interaction.jsonl"),
    "resume": False,
    "experimental_user_message_injection": False,
}

COMPLETED_SESSION_ITEMS = (
    Init("baseline-session"),
    Instructions("Original instructions."),
    Message(role="user", content="Original request."),
    UserInteractionBoundary(),
    Message(role="assistant", content="Previous answer."),
    SampleMetadata(
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            cached_input_tokens=4,
        ),
        provider_turn_id="previous-turn",
        provider_turn_state="previous-state",
    ),
    ModelSampleBoundary(),
    TurnSummary(
        input_tokens_sum=10,
        output_tokens_sum=2,
        cached_input_tokens_sum=4,
        cached_input_tokens_max=4,
        non_cached_input_tokens_sum=6,
        context_tokens=12,
        sample_count=1,
    ),
)

PLAN_CALL = ToolCall(
    name="update_plan",
    call_id="plan-1",
    arguments_json='{"plan":[{"step":"Inspect","status":"in_progress"}]}',
)
ANSWER = ModelSample(
    items=(Message(role="assistant", content="Done."),),
    stop_reason="end_turn",
)
INJECTION_CALL = ToolCall(
    name="experimental_inject_user_message",
    call_id="inject-1",
    arguments_json="{}",
)
INJECTION_RESULT = ToolResult(INJECTION_CALL.call_id, "Synthetic user message queued.")
INJECTED_MESSAGE = Message(role="user", content="hello world")


class _CheckpointRecordingModel:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.calls = []
        self.checkpoints = []

    def sample(self, context, *, tools=(), sampling_params=None):
        self.calls.append((context.copy(), tuple(tools), sampling_params))
        self.checkpoints.append(load_interaction_save("interaction.jsonl"))
        try:
            return next(self.samples)
        except StopIteration:
            raise AssertionError("unexpected model sample") from None


class DemoArgumentBaselineTests(unittest.TestCase):
    def test_argument_defaults(self):
        self.assertEqual(
            vars(demo._build_parser().parse_args([])),
            DEMO_ARGUMENT_DEFAULTS,
        )

    def test_experimental_flag_does_not_change_provider_or_model_selection(self):
        args = demo._build_parser().parse_args([
            "--experimental-user-message-injection",
            "--endpoint-api", "codex",
            "--model", "codex-gpt-5.6-sol-medium",
        ])
        self.assertEqual(vars(args), {
            **DEMO_ARGUMENT_DEFAULTS,
            "experimental_user_message_injection": True,
            "model_api": "codex",
            "model": "codex-gpt-5.6-sol-medium",
        })

    def test_explicit_arguments_preserve_empty_instructions_and_query_text(self):
        query = " /quit\nTreat this as one user query.\n"
        args = demo._build_parser().parse_args(
            [
                "--endpoint-api", "codex",
                "--model", "codex-gpt-6-astra",
                "--endpoint-url", "https://proxy.example.test/codex/responses",
                "--endpoint-auth", "codex-login",
                "--endpoint-auth-file", "auth.json",
                "--cwd", "workspace",
                "--save", "custom.jsonl",
                "--max-samples", "2",
                "--max-output-tokens", "77",
                "--request-timeout-seconds", "9",
                "--instructions", "",
                "--prompt", query,
                "--resume",
            ]
        )
        self.assertEqual(
            vars(args),
            {
                **DEMO_ARGUMENT_DEFAULTS,
                "model_api": "codex",
                "model": "codex-gpt-6-astra",
                "endpoint_url": "https://proxy.example.test/codex/responses",
                "endpoint_auth": "codex-login",
                "codex_auth_file": "auth.json",
                "cwd": "workspace",
                "save_path": Path("custom.jsonl"),
                "max_samples": 2,
                "max_output_tokens": 77,
                "request_timeout_seconds": 9.0,
                "instructions": "",
                "prompt": query,
                "resume": True,
            },
        )


class DemoStartupBaselineTests(unittest.TestCase):
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
        self.path = self.launch / "interaction.jsonl"

    def _run_demo(self, argv=(), samples=(ANSWER,)):
        model = _CheckpointRecordingModel(samples)
        with mock.patch.object(demo, "_build_model", return_value=model):
            with mock.patch("builtins.print") as print_mock:
                status = demo.main(argv)
        printed = tuple(call.args[0] for call in print_mock.call_args_list)
        self.assertEqual(len(model.calls), len(model.checkpoints))
        for (context, _tools, _options), checkpoint in zip(
            model.calls, model.checkpoints
        ):
            self.assertEqual(context.items, checkpoint.items)
        return status, model, printed

    def test_workspace_flag_is_forwarded_and_warned(self):
        with mock.patch.object(
            demo,
            "DefaultEnvironment",
            wraps=demo.DefaultEnvironment,
        ) as environment:
            status, _model, printed = self._run_demo([
                "--cwd", str(self.workspace),
                "--enable-workspace=False",
                "--prompt", "No restricted workdir.",
            ])

        self.assertEqual(status, 0)
        environment.assert_called_once_with(
            cwd=self.workspace,
            enable_workspace=False,
            extra_tools=(),
        )
        self.assertTrue(any(
            "workspace path restrictions are disabled" in text
            for text in printed
        ))

    def test_one_shot_default_query_is_still_injected(self):
        status, model, _printed = self._run_demo()

        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 1)
        context, tools, options = model.calls[0]
        self.assertEqual(tuple(spec.name for spec in tools), (
            "exec_command", "write_stdin", "update_plan", "apply_patch",
        ))
        self.assertIsInstance(context.items[0], Init)
        self.assertEqual(
            context.items[1:],
            (
                Message(role="user", content=demo.DEFAULT_PROMPT),
                UserInteractionBoundary(),
            ),
        )
        self.assertEqual(options, ResolvedSamplingParams())

    def test_initial_query_is_one_item_once_across_tool_follow_up(self):
        query = "/quit\nInspect café without splitting this query.\n"
        samples = (
            ModelSample(
                items=(PLAN_CALL,),
                usage=TokenUsage(
                    input_tokens=20,
                    output_tokens=4,
                    total_tokens=24,
                    cached_input_tokens=5,
                ),
            ),
            ModelSample(
                items=ANSWER.items,
                usage=TokenUsage(
                    input_tokens=30,
                    output_tokens=6,
                    total_tokens=36,
                    cached_input_tokens=10,
                ),
            ),
        )
        with mock.patch.object(
            demo,
            "perf_counter",
            side_effect=(10.0, 12.5),
        ):
            status, model, printed = self._run_demo(
                [
                    "--prompt", query,
                    "--max-samples", "2",
                    "--max-output-tokens", "77",
                ],
                samples,
            )

        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 2)
        for context, _tools, options in model.calls:
            self.assertEqual(
                tuple(item for item in context if isinstance(item, Message)),
                (Message(role="user", content=query),),
            )
            self.assertEqual(
                options,
                ResolvedSamplingParams(max_output_tokens=77),
            )
        self.assertEqual(
            tuple(
                item.text for item in printed if isinstance(item, DisplayItem)
            ),
            (
                f"[user] {query.rstrip()}",
                "[tool-call] update_plan (plan-1)",
                "[sample] input=20 output=4 total=24 cached=5",
                "[tool-ret]  update_plan (plan-1) [ok]\n"
                "[plan] Updated plan\n[plan] [>] Inspect",
                "[assistant] Done.",
                "[sample] input=30 output=6 total=36 cached=10",
                "[turn] input_sum=50 output_sum=10 cold_sum=35 "
                "cached_sum=15 cached_max=10 context=36 samples=2 "
                "compactions=0 elapsed=2.50s",
            ),
        )
        restored = load_interaction_save(self.path)
        self.assertEqual(
            restored.items.count(Message(role="user", content=query)), 1
        )
        self.assertEqual(restored.items.count(UserInteractionBoundary()), 1)
        self.assertEqual(
            restored.items[-1],
            TurnSummary(
                input_tokens_sum=50,
                output_tokens_sum=10,
                cached_input_tokens_sum=15,
                cached_input_tokens_max=10,
                non_cached_input_tokens_sum=35,
                context_tokens=36,
                sample_count=2,
                elapsed_seconds=2.5,
            ),
        )

    def _assert_reasoning_is_visible_but_redacted_live_and_on_resume(
        self,
        reasoning,
        secret,
    ):
        sample = ModelSample(
            items=(
                reasoning,
                Message(role="assistant", content="Done."),
            ),
        )

        status, _model, printed = self._run_demo(
            ["--prompt", "Inspect."],
            (sample,),
        )

        self.assertEqual(status, 0)
        display_text = tuple(
            item.text for item in printed if isinstance(item, DisplayItem)
        )
        self.assertEqual(
            display_text.count("[reasoning] ..."),
            1,
        )
        self.assertNotIn(secret, "\n".join(display_text))

        status, replay_model, replayed = self._run_demo(
            ["--resume"],
            samples=(),
        )

        self.assertEqual(status, 0)
        self.assertEqual(replay_model.calls, [])
        replayed_text = tuple(
            item.text for item in replayed if isinstance(item, DisplayItem)
        )
        self.assertEqual(
            replayed_text.count("[reasoning] ..."),
            1,
        )
        self.assertNotIn(secret, "\n".join(replayed_text))

    def test_encrypted_only_reasoning_is_visible_but_redacted_live_and_on_resume(
        self,
    ):
        ciphertext = "provider-ciphertext-must-not-be-displayed"
        self._assert_reasoning_is_visible_but_redacted_live_and_on_resume(
            Reasoning(content="", encrypted_content=ciphertext), ciphertext,
        )

    def test_signed_thinking_without_text_is_visible_but_redacted_live_and_on_resume(
        self,
    ):
        signature = "thinking-signature-must-not-be-displayed"
        self._assert_reasoning_is_visible_but_redacted_live_and_on_resume(
            Reasoning(content="", content_signature=signature), signature,
        )

    def test_fresh_start_replaces_launch_session_not_workspace_session(self):
        save_interaction_save(
            self.path, InteractionContext(COMPLETED_SESSION_ITEMS)
        )
        workspace_path = self.workspace / "interaction.jsonl"
        workspace_path.write_text("workspace sentinel\n", encoding="utf-8")

        status, model, _printed = self._run_demo(
            ["--cwd", str(self.workspace), "--prompt", "Fresh query."],
            (
                ModelSample(
                    items=(
                        ToolCall(
                            name="exec_command",
                            call_id="cwd-1",
                            arguments_json='{"cmd":"pwd","yield_time_ms":1000}',
                        ),
                    ),
                ),
                ANSWER,
            ),
        )

        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(Path.cwd(), self.launch)
        restored = load_interaction_save(self.path)
        self.assertIsInstance(restored.items[0], Init)
        self.assertNotEqual(restored.items[0], COMPLETED_SESSION_ITEMS[0])
        self.assertEqual(
            model.calls[0][0].items[1:],
            (
                Message(role="user", content="Fresh query."),
                UserInteractionBoundary(),
            ),
        )
        tool_result = model.calls[1][0].items[-1]
        self.assertIsInstance(tool_result, ToolResult)
        self.assertTrue(tool_result.success)
        self.assertIn(f"\n{self.workspace}\n", tool_result.output)
        self.assertEqual(workspace_path.read_text(), "workspace sentinel\n")

    def test_completed_resume_replays_summary_without_sampling_or_appending(self):
        save_interaction_save(
            self.path, InteractionContext(COMPLETED_SESSION_ITEMS)
        )
        before = self.path.read_bytes()

        status, model, printed = self._run_demo(["--resume"], samples=())

        self.assertEqual(status, 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(
            tuple(
                item.text for item in printed if isinstance(item, DisplayItem)
            ),
            (
                "[instructions] Original instructions.",
                "[user] Original request.",
                "[assistant] Previous answer.",
                "[sample] input=10 output=2 total=12 cached=4",
                "[turn] input_sum=10 output_sum=2 cold_sum=6 "
                "cached_sum=4 cached_max=4 context=12 samples=1 compactions=0",
            ),
        )

    def test_resume_sweeps_tools_before_override_and_initial_query(self):
        interrupted = (
            *COMPLETED_SESSION_ITEMS[:4],
            PLAN_CALL,
            ModelSampleBoundary(),
        )
        save_interaction_save(self.path, InteractionContext(interrupted))

        status, model, _printed = self._run_demo(
            ["--resume", "--instructions", "", "--prompt", "Follow-up."]
        )

        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 1)
        expected = (
            *interrupted,
            ToolResult(call_id=PLAN_CALL.call_id, output="Plan updated"),
            Instructions(""),
            Message(role="user", content="Follow-up."),
            UserInteractionBoundary(),
        )
        self.assertEqual(model.calls[0][0].items, expected)
        self.assertEqual(model.calls[0][0].model_items()[0], Instructions(""))
        self.assertEqual(
            load_interaction_save(self.path).items[:len(expected)], expected
        )

    def test_instructions_only_resume_appends_empty_or_nonempty_override(self):
        for instructions in ("", "New instructions."):
            with self.subTest(instructions=instructions):
                save_interaction_save(
                    self.path, InteractionContext(COMPLETED_SESSION_ITEMS)
                )
                status, model, _printed = self._run_demo(
                    ["--resume", "--instructions", instructions]
                )

                self.assertEqual(status, 0)
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(
                    model.calls[0][0].items,
                    (*COMPLETED_SESSION_ITEMS, Instructions(instructions)),
                )

    def test_missing_resume_file_warns_and_uses_supplied_or_demo_default_query(self):
        for prompt in (None, "Explicit query."):
            with self.subTest(prompt=prompt):
                self.path.unlink(missing_ok=True)
                argv = ["--resume"]
                if prompt is not None:
                    argv.extend(("--prompt", prompt))
                status, model, printed = self._run_demo(argv)

                self.assertEqual(status, 0)
                self.assertIn(
                    "Warning: no existing interaction.jsonl was found; "
                    "a fresh one was created.",
                    printed,
                )
                self.assertEqual(
                    model.calls[0][0].items[1:],
                    (
                        Message(role="user", content=prompt or demo.DEFAULT_PROMPT),
                        UserInteractionBoundary(),
                    ),
                )

    def test_malformed_resume_file_fails_without_replacement_or_sampling(self):
        malformed = b'{"type":\n'
        self.path.write_bytes(malformed)

        status, model, printed = self._run_demo(
            ["--resume", "--prompt", "Follow-up."], samples=()
        )

        self.assertEqual(status, 1)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), malformed)
        self.assertTrue(any("invalid JSON" in str(item) for item in printed))

    def test_empty_initial_query_fails_before_replacing_session_or_sampling(self):
        save_interaction_save(
            self.path, InteractionContext(COMPLETED_SESSION_ITEMS)
        )
        before = self.path.read_bytes()
        for prompt in ("", " \n\t"):
            with self.subTest(prompt=prompt):
                status, model, _printed = self._run_demo(
                    ["--prompt", prompt], samples=()
                )
                self.assertEqual(status, 1)
                self.assertEqual(model.calls, [])
                self.assertEqual(self.path.read_bytes(), before)

    def test_experiment_injects_after_entire_batch_and_checkpoints_before_sampling(self):
        status, model, printed = self._run_demo(
            ["--experimental-user-message-injection", "--max-samples", "2"],
            (ModelSample(items=(INJECTION_CALL, PLAN_CALL)), ANSWER),
        )
        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 2)
        initial, tools, _options = model.calls[0]
        self.assertEqual(initial.items[1:], (
            Message("user", demo.EXPERIMENTAL_USER_MESSAGE_PROMPT),
            UserInteractionBoundary(),
        ))
        self.assertEqual(tuple(spec.name for spec in tools), (
            "exec_command", "write_stdin", "update_plan", "apply_patch", INJECTION_CALL.name,
        ))
        following = model.calls[1][0]
        following.assert_model_ready()
        self.assertEqual(following.items[-3:], (
            INJECTION_RESULT,
            ToolResult(PLAN_CALL.call_id, "Plan updated"),
            INJECTED_MESSAGE,
        ))
        self.assertEqual(following.items.count(UserInteractionBoundary()), 1)
        displays = tuple(item.text for item in printed if isinstance(item, DisplayItem))
        self.assertEqual(displays.count("[user] hello world"), 1)
        self.assertLess(
            next(i for i, text in enumerate(displays) if text.startswith("[tool-ret]  update_plan")),
            displays.index("[user] hello world"),
        )
        restored = load_interaction_save(self.path)
        self.assertEqual(restored.items[:len(following)], following.items)
        self.assertEqual(restored.items.count(INJECTED_MESSAGE), 1)

        before = self.path.read_bytes()
        status, replay_model, replay = self._run_demo(
            ["--resume", "--experimental-user-message-injection"], samples=(),
        )
        self.assertEqual(status, 0)
        self.assertEqual(replay_model.calls, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(
            tuple(item.text for item in replay if isinstance(item, DisplayItem)).count("[user] hello world"),
            1,
        )

    def test_experimental_prompt_is_overridable_without_disabling_the_tool(self):
        for resume in (False, True):
            with self.subTest(resume=resume):
                self.path.unlink(missing_ok=True)
                argv = ["--experimental-user-message-injection", "--prompt", "Custom test."]
                if resume:
                    argv.append("--resume")
                status, model, _printed = self._run_demo(argv)
                self.assertEqual(status, 0)
                context, tools, _options = model.calls[0]
                self.assertEqual(context.items[1:], (
                    Message("user", "Custom test."), UserInteractionBoundary(),
                ))
                self.assertIn(INJECTION_CALL.name, tuple(tool.name for tool in tools))
                # Enabling the tool alone does not append the synthetic message.
                self.assertNotIn(INJECTED_MESSAGE, load_interaction_save(self.path).items)

    def test_missing_experimental_resume_uses_experimental_seed_prompt(self):
        status, model, printed = self._run_demo(
            ["--resume", "--experimental-user-message-injection"],
        )
        self.assertEqual(status, 0)
        self.assertEqual(model.calls[0][0].items[1:], (
            Message("user", demo.EXPERIMENTAL_USER_MESSAGE_PROMPT),
            UserInteractionBoundary(),
        ))
        self.assertIn(
            "Warning: no existing interaction.jsonl was found; a fresh one was created.",
            printed,
        )

    def test_experimental_resume_does_not_append_seed_to_completed_save(self):
        save_interaction_save(self.path, InteractionContext(COMPLETED_SESSION_ITEMS))
        before = self.path.read_bytes()
        status, model, _printed = self._run_demo(
            ["--resume", "--experimental-user-message-injection"], samples=(),
        )
        self.assertEqual(status, 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_experimental_resume_sweeps_pending_call_without_reseeding(self):
        interrupted = (
            *COMPLETED_SESSION_ITEMS[:4], INJECTION_CALL, ModelSampleBoundary(),
        )
        save_interaction_save(self.path, InteractionContext(interrupted))
        status, model, _printed = self._run_demo(
            ["--resume", "--experimental-user-message-injection"],
        )
        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 1)
        expected = (*interrupted, INJECTION_RESULT, INJECTED_MESSAGE)
        self.assertEqual(model.calls[0][0].items, expected)
        restored = load_interaction_save(self.path)
        self.assertEqual(restored.items[:len(expected)], expected)
        self.assertEqual(restored.items.count(INJECTED_MESSAGE), 1)

    def test_pending_experiment_without_opt_in_is_unknown_and_cannot_inject(self):
        interrupted = (
            *COMPLETED_SESSION_ITEMS[:4], INJECTION_CALL, ModelSampleBoundary(),
        )
        save_interaction_save(self.path, InteractionContext(interrupted))
        status, model, _printed = self._run_demo(["--resume"])
        self.assertEqual(status, 0)
        result = model.calls[0][0].items[-1]
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.output)
        self.assertNotIn(INJECTED_MESSAGE, load_interaction_save(self.path).items)


if __name__ == "__main__":
    unittest.main()
