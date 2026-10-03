from __future__ import annotations

from pythia_test.interaction_helpers import messages_endpoint

import copy
import io
import unittest
from contextlib import redirect_stderr
from dataclasses import replace

from pythia.interaction import ContextPrefix
from pythia.interaction import Instructions
from pythia.interaction import MESSAGES_COMPACTION_BETA
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import MessagesPromptCaching
from pythia.interaction import MessagesServerCompaction
from pythia.interaction import ModelConfigurationError
from pythia.interaction import InteractionContext
from pythia.interaction import ModelTransportError
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import SampleParams
from pythia.interaction import TokenUsage
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import cli
from pythia.interaction import demo
from pythia.interaction import interaction_item_to_dict
from pythia.interaction import summarize_turn_usage
from pythia.interaction.model_config import build_model
from pythia_test.test_messages import _FakeResponse
from pythia_test.test_messages import _ScriptedOpener
from pythia_test.test_messages import _payload


def _response(*, content=None, usage=None, stop_reason="end_turn"):
    return _FakeResponse({
        "type": "message",
        "role": "assistant",
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": "Done."}] if content is None else content,
        "usage": {} if usage is None else usage,
    })


def _marked_blocks(payload):
    """``(message index, block index, block)`` for each block breakpoint."""
    return [
        (message_index, block_index, block)
        for message_index, message in enumerate(payload["messages"])
        for block_index, block in enumerate(message["content"])
        if "cache_control" in block
    ]


def _without_block_markers(payload):
    """A copy of ``payload`` without block-level cache controls."""
    stripped = copy.deepcopy(payload)
    for message in stripped["messages"]:
        for block in message["content"]:
            block.pop("cache_control", None)
    return stripped


class MessagesPromptCachingTests(unittest.TestCase):
    def test_configuration_defaults_and_validation(self):
        self.assertEqual(MessagesPromptCaching().ttl, "5m")
        for ttl in ("5m", "1h"):
            with self.subTest(ttl=ttl):
                policy = MessagesPromptCaching(ttl=ttl)
                self.assertEqual(
                    policy.request_cache_control(),
                    {"type": "ephemeral", "ttl": ttl},
                )
                control = policy.request_cache_control()
                control["ttl"] = "invalid"
                self.assertEqual(policy.request_cache_control()["ttl"], ttl)
        for ttl in ("", "5s", "60m", "1H", " 5m "):
            with self.subTest(ttl=ttl):
                with self.assertRaisesRegex(ModelConfigurationError, "ttl"):
                    MessagesPromptCaching(ttl=ttl)
        for ttl in (None, True, 300, [], {}):
            with self.subTest(ttl=ttl):
                with self.assertRaisesRegex(TypeError, "ttl"):
                    MessagesPromptCaching(ttl=ttl)
        for policy in (True, False, "5m", {}, {"type": "ephemeral"}):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(TypeError, "prompt_caching"):
                    messages_endpoint(
                        api_url="http://localhost", model="model",
                        max_output_tokens=100,
                        prompt_caching=policy,
                    )

    def test_direct_endpoint_caching_is_off_by_default(self):
        opener = _ScriptedOpener(_response())
        endpoint = messages_endpoint(
            api_url="http://localhost",
            model="model",
            max_output_tokens=100,
        )
        self.assertIsNone(endpoint.prompt_caching)
        MessagesModel(endpoint, opener=opener).sample(
            InteractionContext((Message("user", "Hello."),))
        )
        # Neither the top-level control nor a block breakpoint is sent.
        self.assertNotIn(b"cache_control", opener.calls[0][0].data)
        self.assertIsNone(opener.calls[0][0].get_header("Anthropic-beta"))

    def test_unsupported_cache_control_is_not_silently_retried(self):
        opener = _ScriptedOpener(
            _FakeResponse({"error": "cache_control is unsupported"}, status=400),
            _response(),
        )
        model = MessagesModel(messages_endpoint(
            api_url="http://localhost", model="model",
            max_output_tokens=100,
            prompt_caching=MessagesPromptCaching(),
        ), opener=opener)
        with self.assertRaisesRegex(ModelTransportError, "cache_control"):
            model.sample(InteractionContext((Message("user", "Hello."),)))
        self.assertEqual(len(opener.calls), 1)
        self.assertIn("cache_control", _payload(opener))

    def test_caching_across_tool_turns_advances_breakpoint_and_preserves_blocks(self):
        for ttl in ("5m", "1h"):
            with self.subTest(ttl=ttl):
                opener = _ScriptedOpener(
                    _response(content=[
                        {"type": "thinking", "thinking": "Check facts.", "signature": "sig"},
                        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}},
                    ], stop_reason="tool_use"),
                    _response(),
                )
                endpoint = messages_endpoint(
                    api_url="http://localhost", model="model",
                    max_output_tokens=100,
                    prompt_caching=MessagesPromptCaching(ttl=ttl),
                )
                model = MessagesModel(endpoint, opener=opener)
                uncached = MessagesModel(replace(endpoint, prompt_caching=None))
                context = InteractionContext((
                    Instructions("Be concise."), Message("user", "Check facts."),
                ))
                tools = (ToolSpec("lookup", "Look up facts.", {"type": "object"}),)
                options = SampleParams(max_output_tokens=128)
                control = {"type": "ephemeral", "ttl": ttl}

                for turn in range(2):
                    before = [interaction_item_to_dict(item) for item in context]
                    # The uncached request plus the top-level control and the
                    # same control on the last block: the user text on turn 0,
                    # then the tool result.
                    expected = uncached._build_request_payload(context, tools, options)
                    expected["cache_control"] = control
                    expected["messages"][-1]["content"][-1]["cache_control"] = control
                    sample = model.sample(
                        context, tools=tools, sample_params=options,
                    )
                    self.assertEqual(_payload(opener), expected)
                    self.assertEqual(
                        [interaction_item_to_dict(item) for item in context], before,
                    )
                    self.assertIsNone(opener.calls[-1][0].get_header("Anthropic-beta"))
                    if turn == 0:
                        first_payload = _payload(opener)
                        context.extend(sample.context_items())
                        context.append(ToolResult("call-1", "Found facts."))

                second_payload = _payload(opener)
                self.assertEqual(second_payload["system"], first_payload["system"])
                self.assertEqual(second_payload["tools"], first_payload["tools"])
                # Only the breakpoint moves; the earlier blocks are unchanged.
                first_messages = _without_block_markers(first_payload)["messages"]
                self.assertEqual(
                    _without_block_markers(second_payload)["messages"][:len(first_messages)],
                    first_messages,
                )
                self.assertEqual(len(_marked_blocks(second_payload)), 1)
                self.assertEqual(second_payload["messages"][1]["content"][0], {
                    "type": "thinking", "thinking": "Check facts.", "signature": "sig",
                })
                self.assertEqual(second_payload["messages"][-1]["content"], [{
                    "type": "tool_result", "tool_use_id": "call-1",
                    "content": "Found facts.", "is_error": False,
                    "cache_control": control,
                }])

    def test_block_breakpoint_marks_the_rightmost_eligible_block(self):
        def call(call_id):
            return ToolCall(name="lookup", call_id=call_id, arguments_json="{}")

        tools = (ToolSpec("lookup", "Look up facts.", {"type": "object"}),)
        # Context items and the (message, block) index of the marked block.
        cases = (
            ((Instructions("Be concise."), Message("user", "Hello.")), (0, 0)),
            # Thinking and empty text blocks never carry a breakpoint.
            ((Instructions(""), Message("user", "Hello."), Reasoning("thought")), (0, 0)),
            ((Message("user", "Hello."), Message("assistant", "")), (0, 0)),
            ((Message("user", "Hello."), Message("assistant", " \n")), (0, 0)),
            # Parallel tool results: the last one.
            ((
                Message("user", "Go."), Reasoning("Plan.", content_signature="sig"),
                call("call-1"), call("call-2"),
                ToolResult("call-1", "One."), ToolResult("call-2", "Two."),
            ), (2, 1)),
            # Blank tool results are skipped: an earlier result, else the call.
            ((
                Message("user", "Go."), call("call-1"), call("call-2"),
                ToolResult("call-1", "One."), ToolResult("call-2", ""),
            ), (2, 0)),
            ((
                Message("user", "Go."), Message("assistant", "Looking."),
                call("call-1"), ToolResult("call-1", " "),
            ), (1, 1)),
            # After a paused server compaction, its block is the last one.
            ((Message("user", "Old question."), OpaqueCompaction.from_messages("Summary.")), (0, 0)),
            # Nothing eligible: only the top-level control is sent.
            ((Instructions("Be concise."), Message("user", "")), None),
        )
        for items, position in cases:
            for ttl in ("5m", "1h"):
                with self.subTest(items=items, ttl=ttl):
                    context = InteractionContext(items)
                    opener = _ScriptedOpener(_response())
                    endpoint = messages_endpoint(
                        api_url="http://localhost", model="model",
                        max_output_tokens=100,
                        prompt_caching=MessagesPromptCaching(ttl=ttl),
                    )
                    uncached = MessagesModel(
                        replace(endpoint, prompt_caching=None),
                    )._build_request_payload(context, tools, None)
                    MessagesModel(endpoint, opener=opener).sample(context, tools=tools)
                    payload = _payload(opener)
                    control = {"type": "ephemeral", "ttl": ttl}
                    self.assertEqual(payload["cache_control"], control)
                    self.assertEqual(
                        [(m, b, block["cache_control"]) for m, b, block in _marked_blocks(payload)],
                        [] if position is None else [(*position, control)],
                    )
                    # Purely additive: system, tools, and every other block
                    # are exactly as in the uncached request.
                    self.assertEqual(
                        _without_block_markers(payload),
                        {**uncached, "cache_control": control},
                    )

    def test_caching_composes_with_compaction_without_changing_the_log(self):
        for checkpoint in (
            OpaqueCompaction.from_messages("Summary."),
            ContextPrefix((Instructions("Be concise."), Message("user", "Summary."))),
        ):
            with self.subTest(checkpoint=checkpoint):
                context = InteractionContext((
                    Instructions("Be concise."), Message("user", "Old question."),
                    checkpoint, Message("user", "Continue."),
                ))
                before = [interaction_item_to_dict(item) for item in context]
                opener = _ScriptedOpener(_response())
                model = MessagesModel(messages_endpoint(
                    api_url="http://localhost", model="model",
                    max_output_tokens=100,
                    prompt_caching=MessagesPromptCaching(ttl="1h"),
                    server_compaction=MessagesServerCompaction(),
                ), opener=opener)
                model.sample(context)
                payload = _payload(opener)
                self.assertEqual(payload["cache_control"], {"type": "ephemeral", "ttl": "1h"})
                # The block breakpoint is placed after the projection.
                self.assertEqual([block for _, _, block in _marked_blocks(payload)], [{
                    "type": "text", "text": "Continue.",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }])
                self.assertEqual(payload["context_management"], {
                    "edits": [{"type": "compact_20260112"}],
                })
                self.assertEqual(
                    opener.calls[0][0].get_header("Anthropic-beta"), MESSAGES_COMPACTION_BETA,
                )
                self.assertEqual(payload["system"], [{"type": "text", "text": "Be concise."}])
                self.assertNotIn("Old question.", str(payload["messages"]))
                self.assertEqual([interaction_item_to_dict(item) for item in context], before)

    def test_cache_usage_counts_writes_and_reads_once(self):
        cases = (
            ({"input_tokens": 100, "output_tokens": 10}, TokenUsage(100, 10, 110, 0)),
            ({
                "input_tokens": 20, "output_tokens": 10,
                "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0,
            }, TokenUsage(1020, 10, 1030, 0)),
            ({
                "input_tokens": 20, "output_tokens": 10,
                "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
            }, TokenUsage(1120, 10, 1130, 1000)),
            ({
                "input_tokens": 0, "output_tokens": 10,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 2048,
            }, TokenUsage(2048, 10, 2058, 2048)),
            ({
                "input_tokens": 2048, "output_tokens": 503,
                "cache_creation_input_tokens": 248, "cache_read_input_tokens": 1800,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 148, "ephemeral_1h_input_tokens": 100,
                },
            }, TokenUsage(4096, 503, 4599, 1800)),
        )
        for usage, expected in cases:
            with self.subTest(usage=usage):
                opener = _ScriptedOpener(_response(usage=usage))
                model = MessagesModel(messages_endpoint(
                    api_url="http://localhost", model="model",
                    max_output_tokens=100,
                    prompt_caching=MessagesPromptCaching(),
                ), opener=opener)
                sample = model.sample(InteractionContext((Message("user", "Hello."),)))
                self.assertEqual(sample.usage, expected)
                summary = summarize_turn_usage(sample.context_items())
                self.assertEqual(summary.cached_input_tokens_sum, expected.cached_input_tokens)
                self.assertEqual(
                    summary.non_cached_input_tokens_sum,
                    expected.input_tokens - expected.cached_input_tokens,
                )

    def test_compaction_cache_usage_is_not_added_again_to_iteration_totals(self):
        for per_iteration_cache in (False, True):
            with self.subTest(per_iteration_cache=per_iteration_cache):
                iterations = [
                    {"type": "compaction", "input_tokens": 100, "output_tokens": 10},
                    {"type": "message", "input_tokens": 200, "output_tokens": 5},
                ]
                if per_iteration_cache:
                    iterations[0].update(cache_creation_input_tokens=20, cache_read_input_tokens=50)
                    iterations[1].update(cache_creation_input_tokens=30, cache_read_input_tokens=60)
                opener = _ScriptedOpener(_response(usage={
                    "input_tokens": 300, "output_tokens": 15,
                    "cache_creation_input_tokens": 50, "cache_read_input_tokens": 110,
                    "iterations": iterations,
                }))
                model = MessagesModel(messages_endpoint(
                    api_url="http://localhost", model="model",
                    max_output_tokens=100,
                    prompt_caching=MessagesPromptCaching(),
                    server_compaction=MessagesServerCompaction(),
                ), opener=opener)
                sample = model.sample(InteractionContext((Message("user", "Hello."),)))
                self.assertEqual(sample.usage, TokenUsage(460, 15, 475, 110))


class MessagesPromptCachingCLITests(unittest.TestCase):
    def test_cli_and_demo_default_to_automatic_five_minute_caching(self):
        for frontend in (cli, demo):
            for flags in (
                [],
                ["--enable-auto-compaction"],
                ["--enable-auto-compaction=False"],
                ["--resume"],
                ["--endpoint-url", "http://localhost/v1/messages"],
                ["--compaction-mode", "provider"],
                ["--compaction-mode", "provider", "--enable-auto-compaction=False"],
            ):
                with self.subTest(frontend=frontend.__name__, flags=flags):
                    args = frontend._build_parser().parse_args([
                        "--endpoint-api", "messages", "--model", "model",
                        "--max-output-tokens", "100", "--endpoint-auth", "none", *flags,
                    ])
                    model = build_model(args)
                    self.assertEqual(model.endpoint.prompt_caching, MessagesPromptCaching())
                    self.assertEqual(model.endpoint.prompt_caching.ttl, "5m")
                    # Server compaction belongs to provider mode only; the
                    # config's per-call flag suppresses it there.
                    self.assertEqual(
                        model.endpoint.server_compaction,
                        (
                            MessagesServerCompaction()
                            if args.compaction_mode == "provider"
                            else None
                        ),
                    )
                    payload = model._build_request_payload(
                        InteractionContext((Message("user", "Hello."),)), (), None,
                    )
                    self.assertEqual(
                        payload["cache_control"], {"type": "ephemeral", "ttl": "5m"},
                    )
                    self.assertEqual(payload["messages"], [{"role": "user", "content": [{
                        "type": "text", "text": "Hello.",
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    }]}])

    def test_frontends_do_not_expose_cache_options(self):
        for frontend in (cli, demo):
            for flags in (
                ["--messages-prompt-caching"],
                ["--messages-cache-ttl", "5m"],
                ["--messages-cache-ttl", "1h"],
            ):
                with self.subTest(frontend=frontend.__name__, flags=flags):
                    with redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            frontend._build_parser().parse_args(flags)
                    self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
