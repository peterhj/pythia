from __future__ import annotations

import json
import unittest

from pythia.interaction import COMPACTION_SUMMARY_PREFIX
from pythia.interaction import COMPACTION_SUMMARY_SUFFIX
from pythia.interaction import CompactionMetadata
from pythia.interaction import CompactionResult
from pythia.interaction import ContextPrefix
from pythia.interaction import DisplayItem
from pythia.interaction import EnvironmentResult
from pythia.interaction import Instructions
from pythia.interaction import InteractionItemRenderer
from pythia.interaction import Message
from pythia.interaction import ModelFailure
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import TokenUsage
from pythia.interaction import SampleMetadata
from pythia.interaction import UserInteraction
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import render_interaction_items
from pythia.interaction.compaction import _LEGACY_SUMMARY_PREFIX


ANSI_GREEN = "\x1b[32m"
ANSI_RED = "\x1b[31m"
ANSI_BRIGHT_BLACK = "\x1b[90m"
ANSI_RESET = "\x1b[0m"


class DisplayItemTests(unittest.TestCase):
    def test_label_is_validated_non_rendering_metadata(self):
        plain = DisplayItem("[reasoning] inspect")
        labeled = DisplayItem(plain.text, label="reasoning")
        self.assertIsNone(plain.label)
        self.assertEqual(labeled.label, "reasoning")
        self.assertEqual(labeled, plain)
        self.assertEqual(str(labeled), str(plain))
        self.assertEqual(repr(labeled), repr(plain))
        for invalid in (1, False):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                DisplayItem(plain.text, label=invalid)
        for invalid in ("", "assistant"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DisplayItem(plain.text, label=invalid)

    def test_display_item_is_nominal_printable_text(self):
        item = DisplayItem("line one\nline two")

        self.assertEqual(item.text, "line one\nline two")
        self.assertEqual(
            str(item),
            (
                f"   {ANSI_BRIGHT_BLACK}⌜{ANSI_RESET}line one\n"
                f"   {ANSI_BRIGHT_BLACK}⌞{ANSI_RESET}line two"
            ),
        )

        self.assertEqual(
            str(DisplayItem("line one")),
            f"   {ANSI_BRIGHT_BLACK}[{ANSI_RESET}line one",
        )
        self.assertEqual(
            str(DisplayItem("line one\nline two\n\nline three")),
            (
                f"   {ANSI_BRIGHT_BLACK}⌜{ANSI_RESET}line one\n"
                "    line two\n"
                "\n"
                f"   {ANSI_BRIGHT_BLACK}⌞{ANSI_RESET}line three"
            ),
        )
        self.assertIn("DisplayItem", repr(item))

        for invalid in (None, 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    DisplayItem(invalid)
        for invalid in ("", "text\n", "text\r"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    DisplayItem(invalid)


class InteractionItemRendererTests(unittest.TestCase):
    def test_messages_reasoning_boundaries_and_compaction(self):
        checkpoint = ContextPrefix(
            prefix_items=(
                Message(role="user", content="summary"),
                UserInteractionBoundary(),
            )
        )
        rendered = render_interaction_items(
            (
                Message(role="system", content="system"),
                Message(role="developer", content="developer"),
                Message(role="user", content="user"),
                Message(role="assistant", content="assistant"),
                Message(role="assistant", content="   "),
                Reasoning(
                    content="hidden fallback",
                    summary=("first", " ", "second"),
                ),
                Reasoning(content="fallback"),
                Reasoning(content=" "),
                Reasoning(
                    content="",
                    summary=(),
                    encrypted_content="provider-ciphertext",
                ),
                ModelFailure(
                    category="stream_closed",
                    message="Responses stream closed before response.completed",
                    provider="codex",
                    model="model",
                    auth_source="codex_file",
                    request_id="request-1",
                    attempt_count=2,
                    event_count=3,
                    completed_item_count=1,
                    last_event_type="response.output_item.done",
                    last_sequence_number=4,
                    recovery=("credential_reload",),
                    elapsed_seconds=1.25,
                ),
                SampleMetadata(
                    usage=TokenUsage(
                        input_tokens=20,
                        output_tokens=5,
                        total_tokens=25,
                        cached_input_tokens=4,
                    )
                ),
                ModelSampleBoundary(),
                UserInteractionBoundary(),
                OpaqueCompaction.from_responses("secret"),
                checkpoint,
            )
        )

        self.assertEqual(
            tuple(item.text for item in rendered),
            (
                "[system] system",
                "[developer] developer",
                "[user] user",
                "[assistant] assistant",
                "[reasoning] first",
                "[reasoning] second",
                "[reasoning] fallback",
                "[reasoning] ...",
                "[model failure] Responses stream closed before "
                "response.completed kind=stream_closed provider=codex "
                "model=model auth_source=codex_file attempts=2 "
                "request_id=request-1 events=3 completed_items=1 "
                "last_event=response.output_item.done last_sequence=4 "
                "elapsed=1.25s recovery=credential_reload",
                "[sample] input=20 output=5 total=25 cached=4",
                "[compaction] opaque checkpoint",
                "[context prefix] 2 items",
            ),
        )
        self.assertTrue(all(isinstance(item, DisplayItem) for item in rendered))
        self.assertEqual(tuple(item.label for item in rendered), (
            "system", "developer", "user", "assistant",
            "reasoning", "reasoning", "reasoning", "reasoning",
            "model failure", "sample", "compaction", "context prefix",
        ))

    def test_signed_thinking_without_text_renders_redacted_placeholder(self):
        # Messages counterpart of encrypted-only Responses reasoning: the
        # signature is replay data, so only the placeholder may be shown.
        signature = "thinking-signature-must-not-be-displayed"
        rendered = render_interaction_items(
            (
                Reasoning(content="", content_signature=signature),
                Reasoning(content=" \n ", content_signature=signature),
                Reasoning(content="visible thought", content_signature=signature),
            )
        )

        self.assertEqual(
            tuple(item.text for item in rendered),
            (
                "[reasoning] ...",
                "[reasoning] ...",
                "[reasoning] visible thought",
            ),
        )
        self.assertEqual(
            tuple(item.label for item in rendered),
            ("reasoning",) * 3,
        )
        self.assertNotIn(signature, "\n".join(str(item) for item in rendered))

    def test_context_prefix_shows_its_compaction_summary(self):
        kept = (
            Message("user", "Kept request."),
            Message("assistant", "Kept answer."),
            ModelSampleBoundary(),
        )
        for label, summary in (
            ("new-style", f"{COMPACTION_SUMMARY_PREFIX}## Goal\nShip it.{COMPACTION_SUMMARY_SUFFIX}"),
            ("old-style", f"{_LEGACY_SUMMARY_PREFIX}\n## Goal\nShip it."),
        ):
            with self.subTest(label=label):
                rendered = render_interaction_items((ContextPrefix((
                    Instructions("Rules."),
                    OpaqueCompaction.from_responses("carried-checkpoint"),
                    Message("user", summary),
                    *kept,
                )),))
                # The kept items were shown where they first appeared.
                self.assertEqual(tuple(item.text for item in rendered), (
                    "[context prefix] 6 items",
                    "## Goal\nShip it.",
                ))
                self.assertEqual(tuple(item.label for item in rendered), ("context prefix", None))
        # Summaries are literal payloads, even when they start like a label.
        literal = render_interaction_items((ContextPrefix((
            Message("user", f"{COMPACTION_SUMMARY_PREFIX}[user] quoted{COMPACTION_SUMMARY_SUFFIX}"),
        )),))
        self.assertEqual(literal[1], DisplayItem("[user] quoted"))
        self.assertIsNone(literal[1].label)
        # Remote checkpoints render as before.
        remote = render_interaction_items((ContextPrefix((
            Message("user", "Retained request."),
            OpaqueCompaction.from_responses("secret"),
        )),))
        self.assertEqual(tuple(item.text for item in remote), ("[context prefix] 2 items",))

    def test_message_label_retains_the_full_role(self):
        rendered = render_interaction_items((Message("  custom] role  ", "body"),))
        self.assertEqual(rendered[0].label, "custom] role")
        self.assertEqual(rendered[0].text, "[custom] role] body")

    def test_producer_display_items_and_compaction_context_items(self):
        user = UserInteraction(
            items=(Message(role="user", content="hello"),),
        )
        sample = ModelSample(
            items=(
                Reasoning(content="inspect"),
                Message(role="assistant", content="answer"),
            ),
            usage=TokenUsage(
                input_tokens=20,
                output_tokens=5,
                total_tokens=25,
                cached_input_tokens=4,
            ),
        )
        checkpoint = ContextPrefix(
            prefix_items=(Message(role="user", content="summary"),)
        )
        compaction = CompactionResult(items=(checkpoint,))

        self.assertEqual(
            user.display_items(),
            (DisplayItem("[user] hello"),),
        )
        self.assertEqual(
            sample.display_items(),
            (
                DisplayItem("[reasoning] inspect"),
                DisplayItem("[assistant] answer"),
                DisplayItem(
                    "[sample] input=20 output=5 total=25 cached=4"
                ),
            ),
        )
        metadata = CompactionMetadata(
            usage=TokenUsage(),
            protocol="unspecified",
        )
        self.assertEqual(compaction.context_items(), (checkpoint, metadata))
        self.assertEqual(
            compaction.display_items(),
            (
                DisplayItem("[context prefix] 1 item"),
                DisplayItem(
                    "[compaction] protocol=unspecified input=0 output=0 "
                    "total=0 cached=0"
                ),
            ),
        )

    def test_shell_generic_and_malformed_tool_calls(self):
        exec_call = ToolCall(
            name="exec_command",
            call_id="exec-1",
            arguments_json=json.dumps(
                {
                    "cmd": "git status --short",
                    "yield_time_ms": 1_000,
                }
            ),
        )
        stdin_call = ToolCall(
            name="write_stdin",
            call_id="stdin-1",
            arguments_json=json.dumps(
                {
                    "session_id": 7,
                    "chars": "x\n",
                }
            ),
        )
        generic_call = ToolCall(
            name="lookup",
            call_id="lookup-1",
            arguments_json='{"b":2,"a":1}',
        )
        malformed_call = ToolCall(
            name="lookup",
            call_id="lookup-2",
            arguments_json='{"broken"',
        )

        self.assertEqual(
            render_interaction_items(
                (exec_call, stdin_call, generic_call)
            ),
            (
                DisplayItem(
                    "[tool-call] exec_command (exec-1)\n"
                    "git status --short"
                ),
                DisplayItem(
                    "[tool-call] write_stdin (stdin-1)\n"
                    "session_id=7 chars=2 bytes"
                ),
                DisplayItem("[tool-call] lookup (lookup-1)"),
            ),
        )

        renderer = InteractionItemRenderer(
            show_generic_arguments=True,
        )
        self.assertEqual(
            tuple(item.label for item in renderer.render_items((generic_call, malformed_call))),
            ("tool-call", None, "tool-call", None),
        )
        self.assertEqual(
            renderer.render_items((generic_call, malformed_call)),
            (
                DisplayItem("[tool-call] lookup (lookup-1)"),
                DisplayItem('{"a":1,"b":2}'),
                DisplayItem("[tool-call] lookup (lookup-2)"),
                DisplayItem('{"broken"'),
            ),
        )

    def test_environment_results_resolve_source_calls_without_storing_them(self):
        exec_call = ToolCall(
            name="exec_command",
            call_id="exec-1",
            arguments_json='{"cmd":"printf hello"}',
        )
        result = EnvironmentResult(
            items=(
                ToolResult(
                    call_id="exec-1",
                    output="Process exited with code 0\nOutput:\nhello\n",
                ),
            )
        )

        self.assertEqual(
            result.display_items(source_calls=(exec_call,)),
            (
                DisplayItem(
                    "[tool-ret]  exec_command (exec-1) [ok]\n"
                    "Process exited with code 0\n"
                    "Output:\n"
                    "hello"
                ),
            ),
        )
        self.assertEqual(
            result.display_items(),
            (
                DisplayItem(
                    "[tool-ret]  tool (exec-1) [ok]\n"
                    "Process exited with code 0\n"
                    "Output:\n"
                    "hello"
                ),
            ),
        )
        self.assertEqual(
            result.context_items(),
            result.items,
        )
        self.assertFalse(hasattr(result, "source_calls"))

        with self.assertRaisesRegex(ValueError, "duplicate source"):
            result.display_items(source_calls=(exec_call, exec_call))
        with self.assertRaisesRegex(TypeError, "source_calls"):
            result.display_items(source_calls=(object(),))

    def test_item_sequence_correlates_calls_and_results(self):
        call = ToolCall(
            name="lookup",
            call_id="call-1",
            arguments_json="{}",
        )
        rendered = render_interaction_items(
            (
                call,
                ModelSampleBoundary(),
                ToolResult(
                    call_id="call-1",
                    output="not found",
                    success=False,
                ),
            )
        )

        self.assertEqual(
            rendered,
            (
                DisplayItem("[tool-call] lookup (call-1)"),
                DisplayItem(
                    "[tool-ret]  lookup (call-1) [error]\nnot found"
                ),
            ),
        )

    def test_update_plan_result_uses_source_arguments(self):
        call = ToolCall(
            name="update_plan",
            call_id="plan-1",
            arguments_json=json.dumps(
                {
                    "explanation": "Use a clear sequence.",
                    "plan": [
                        {"step": "Inspect", "status": "completed"},
                        {"step": "Implement", "status": "in_progress"},
                        {"step": "Verify", "status": "pending"},
                    ],
                }
            ),
        )
        success = EnvironmentResult(
            items=(
                ToolResult(
                    call_id="plan-1",
                    output="Plan updated",
                ),
            )
        )
        failure = EnvironmentResult(
            items=(
                ToolResult(
                    call_id="plan-1",
                    output="observer failed",
                    success=False,
                ),
            )
        )

        self.assertEqual(
            success.display_items(source_calls=(call,)),
            (
                DisplayItem(
                    "[tool-ret]  update_plan (plan-1) [ok]\n"
                    "[plan] Updated plan\n"
                    "[plan] note: Use a clear sequence.\n"
                    "[plan] [x] Inspect\n"
                    "[plan] [>] Implement\n"
                    "[plan] [ ] Verify"
                ),
            ),
        )
        self.assertEqual(
            failure.display_items(source_calls=(call,)),
            (
                DisplayItem(
                    "[tool-ret]  update_plan (plan-1) [error]\n"
                    "observer failed"
                ),
            ),
        )

    def test_apply_patch_payload_uses_separate_blocks_and_diff_color(self):
        patch = "\n".join(
            (
                "*** Begin Patch",
                "*** Update File: example.txt",
                "@@",
                "-old",
                "+new",
                "*** End Patch",
            )
        )
        call = ToolCall(
            name="apply_patch",
            call_id="patch-1",
            arguments_json=json.dumps({"patch": f"{patch}\n"}),
        )

        rendered = render_interaction_items((call,))
        self.assertEqual(tuple(item.label for item in rendered), ("tool-call", None))
        self.assertEqual(
            rendered,
            (
                DisplayItem("[tool-call] apply_patch (patch-1)"),
                DisplayItem(patch, is_diff=True),
            ),
        )
        self.assertEqual(
            str(rendered[1]),
            "\n".join(
                (
                    f"   {ANSI_BRIGHT_BLACK}⌜{ANSI_RESET}*** Begin Patch",
                    "    *** Update File: example.txt",
                    "    @@",
                    f"    {ANSI_RED}-old{ANSI_RESET}",
                    f"    {ANSI_GREEN}+new{ANSI_RESET}",
                    f"   {ANSI_BRIGHT_BLACK}⌞{ANSI_RESET}*** End Patch",
                )
            ),
        )

        uncolored = InteractionItemRenderer(color=False).render_items((call,))
        self.assertEqual(
            tuple(item.text for item in uncolored),
            (
                "[tool-call] apply_patch (patch-1)",
                patch,
            ),
        )
        self.assertFalse(uncolored[1].is_diff)
        self.assertEqual(
            str(uncolored[1]),
            "\n".join(
                (
                    f"   {ANSI_BRIGHT_BLACK}⌜{ANSI_RESET}*** Begin Patch",
                    "    *** Update File: example.txt",
                    "    @@",
                    "    -old",
                    "    +new",
                    f"   {ANSI_BRIGHT_BLACK}⌞{ANSI_RESET}*** End Patch",
                )
            ),
        )

    def test_git_diff_output_color_requires_git_diff_command(self):
        diff = "\n".join(
            (
                "diff --git a/example.txt b/example.txt",
                "--- a/example.txt",
                "+++ b/example.txt",
                "@@ -1 +1 @@",
                "-old",
                "+new",
            )
        )
        git_call = ToolCall(
            name="exec_command",
            call_id="git-1",
            arguments_json='{"cmd":"git diff -- example.txt"}',
        )
        cat_call = ToolCall(
            name="exec_command",
            call_id="cat-1",
            arguments_json='{"cmd":"cat example.diff"}',
        )
        renderer = InteractionItemRenderer()

        git_result = renderer.render_items(
            (ToolResult(call_id="git-1", output=diff),),
            source_calls=(git_call,),
        )
        cat_result = renderer.render_items(
            (ToolResult(call_id="cat-1", output=diff),),
            source_calls=(cat_call,),
        )

        expected_text = (
            "[tool-ret]  exec_command (git-1) [ok]\n"
            "diff --git a/example.txt b/example.txt\n"
            "--- a/example.txt\n"
            "+++ b/example.txt\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new"
        )
        self.assertEqual(
            git_result,
            (
                DisplayItem(expected_text, is_diff=True),
            ),
        )
        self.assertEqual(
            str(git_result[0]),
            "\n".join(
                (
                    f"   {ANSI_BRIGHT_BLACK}⌜{ANSI_RESET}[tool-ret]  "
                    "exec_command (git-1) [ok]",
                    "    diff --git a/example.txt b/example.txt",
                    "    --- a/example.txt",
                    "    +++ b/example.txt",
                    "    @@ -1 +1 @@",
                    f"    {ANSI_RED}-old{ANSI_RESET}",
                    f"   {ANSI_BRIGHT_BLACK}⌞{ANSI_RESET}"
                    f"{ANSI_GREEN}+new{ANSI_RESET}",
                )
            ),
        )
        self.assertEqual(
            cat_result,
            (
                DisplayItem(
                    "[tool-ret]  exec_command (cat-1) [ok]\n"
                    f"{diff}"
                ),
            ),
        )
        self.assertFalse(cat_result[0].is_diff)


if __name__ == "__main__":
    unittest.main()
