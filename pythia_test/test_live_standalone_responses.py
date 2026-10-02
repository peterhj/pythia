"""Opt-in live smoke test of the standalone Responses route (paid requests).

Skipped unless ``PYTHIA_LIVE_RESPONSES_MODEL`` names a public model and
``OPENAI_API_KEY`` is set. Requests go to https://api.openai.com/v1/responses
through ``ResponsesModel`` and the CLI, never Codex auth::

    PYTHIA_LIVE_RESPONSES_MODEL=gpt-6.1-luna OPENAI_API_KEY=... \\
        python3 -m pythia_test pythia_test.test_live_standalone_responses

Only request bodies are recorded, never headers. Do not add ``--debug-trace``:
traces record the key verbatim.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.request

from pythia.interaction import (
    BUILTIN_MODEL_CATALOG, DefaultEnvironment, Init, Instructions, InteractionContext,
    Message, ModelAuthenticationError, ModelSampleBoundary, PiCompactor, Reasoning,
    ResponsesModel, ToolCall, ToolResult, ToolSpec, TurnSummary,
    UserInteractionBoundary, cli, load_interaction_save, parse_model_catalog,
)
from pythia.interaction.compaction import _plan_compaction, estimate_item_tokens


MODEL = os.environ.get("PYTHIA_LIVE_RESPONSES_MODEL")
TOOL_FIELDS = {"tools", "tool_choice", "parallel_tool_calls"}
LOOKUP = ToolSpec(
    "lookup_code",
    "Return the secret code registered for a name.",
    {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    },
)


class _Recorder:
    """The real HTTP opener, keeping each request body (never headers)."""

    def __init__(self):
        self.payloads = []

    def __call__(self, request, *, timeout):
        self.payloads.append(json.loads(request.data))
        return urllib.request.urlopen(request, timeout=timeout)


def _model(recorder, **bind_options):
    return ResponsesModel(
        binding=BUILTIN_MODEL_CATALOG.bind("responses", MODEL, **bind_options),
        opener=recorder,
    )


def _reasoning_model(recorder):
    """The same route from a user catalog with typed high effort, so samples reason."""
    catalog = parse_model_catalog(f"""[catalog]
version = 4
[model.live-reasoning]
endpoint.api = responses
endpoint.url = https://api.openai.com/v1/responses
endpoint.model = {MODEL}
endpoint.auth = env:OPENAI_API_KEY
responses.reasoning_effort = high
""")
    return ResponsesModel(binding=catalog.bind(name="live-reasoning"), opener=recorder)


def _encrypted_reasoning(items):
    return [item for item in items if isinstance(item, Reasoning) and item.encrypted_content]


def _shape(items):
    """Item types for failure messages, without content."""
    return [
        type(item).__name__
        + ("[encrypted]" if isinstance(item, Reasoning) and item.encrypted_content else "")
        for item in items
    ]


def _run_to_answer(model, context, samples=3):
    """Sample with lookup_code until a text answer, answering every lookup."""
    for _ in range(samples):
        sample = model.sample(context, tools=(LOOKUP,))
        calls = [item for item in sample.items if isinstance(item, ToolCall)]
        context.extend((
            *sample.items,
            ModelSampleBoundary(),
            *(ToolResult(call.call_id, "ZEBRA-42") for call in calls),
        ))
        if not calls:
            return sample
    raise AssertionError(f"no text answer after {samples} samples")


@unittest.skipUnless(
    MODEL and os.environ.get("OPENAI_API_KEY"),
    "live: set PYTHIA_LIVE_RESPONSES_MODEL and OPENAI_API_KEY",
)
class LiveStandaloneResponsesTests(unittest.TestCase):
    def test_text_request_without_tools(self):
        recorder = _Recorder()
        sample = _model(recorder).sample(InteractionContext((
            Instructions("Answer with one lowercase word."),
            Message("user", "What colour is a clear daytime sky?"),
        )))
        self.assertIn("blue", sample.last_assistant_text.lower())
        self.assertEqual(sample.stop_reason, "end_turn")
        self.assertGreater(sample.usage.total_tokens, 0)
        self.assertFalse(TOOL_FIELDS & recorder.payloads[0].keys())

    def test_default_tool_schemas_are_accepted(self):
        recorder = _Recorder()
        with tempfile.TemporaryDirectory() as directory, \
                DefaultEnvironment(directory) as environment:
            specs = environment.tool_specs
            sample = _model(recorder).sample(
                InteractionContext((
                    Message("user", "Do not call any tools. Reply with the single word: ready"),
                )),
                tools=specs,
            )
        self.assertEqual(
            {tool["name"] for tool in recorder.payloads[0]["tools"]},
            {spec.name for spec in specs},
        )
        self.assertTrue(sample.items)

    def test_rejected_key_is_one_unretried_request(self):
        recorder = _Recorder()
        model = ResponsesModel(
            binding=BUILTIN_MODEL_CATALOG.bind("responses", MODEL, endpoint_auth="supplied"),
            api_key="sk-pythia-live-smoke-test-invalid",
            opener=recorder,
        )
        with self.assertRaises(ModelAuthenticationError) as raised:
            model.sample(InteractionContext((Message("user", "hi"),)))
        self.assertEqual(len(recorder.payloads), 1)
        self.assertEqual(raised.exception.failure.auth_source, "static")

    def test_tool_round_trip_then_pi_compaction_and_continuation(self):
        recorder = _Recorder()
        model = _reasoning_model(recorder)
        context = InteractionContext((
            Init("live"),
            Instructions("Be brief. Use lookup_code whenever asked for a code."),
            Message("user", "Remember: the project codename is BLUEBIRD. Reply with just: noted"),
            UserInteractionBoundary(),
        ))
        reply = model.sample(context)
        context.extend((*reply.items, ModelSampleBoundary(), TurnSummary(sample_count=1)))

        # A function-tool round trip; the second request replays its reasoning.
        context.extend((
            Message("user", "Work out 37 * 43 - 12, call lookup_code with that number "
                            "as the name, then reply with just the code."),
            UserInteractionBoundary(),
        ))
        first = model.sample(context, tools=(LOOKUP,))
        calls = [item for item in first.items if isinstance(item, ToolCall)]
        self.assertTrue(calls, f"the model did not call lookup_code: {_shape(first.items)}")
        context.extend((
            *first.items,
            ModelSampleBoundary(),
            *(ToolResult(call.call_id, "ZEBRA-42") for call in calls),
        ))
        final = _run_to_answer(model, context)
        context.append(TurnSummary(sample_count=2))
        self.assertIn("ZEBRA-42", final.last_assistant_text, _shape(final.items))
        replayed = [item for item in recorder.payloads[2]["input"]
                    if item.get("type") == "reasoning" and item.get("encrypted_content")]
        self.assertEqual(len(replayed), len(_encrypted_reasoning(reply.items + first.items)))

        # Keep the whole second turn verbatim, so the summary replaces only the
        # first. Cuts fall on user and assistant messages, never inside a sample.
        items = context.model_items()
        second_turn = [index for index, item in enumerate(items)
                       if isinstance(item, Message) and item.role == "user"][1]
        keep = sum(estimate_item_tokens(item) for item in items[second_turn:])
        if not _encrypted_reasoning(_plan_compaction(items, keep).tail):
            # Reasoning is the model's choice; without it this run proves nothing.
            self.skipTest("inconclusive: no encrypted reasoning in the second turn "
                          f"(first={_shape(first.items)}, final={_shape(final.items)})")
        requests_before = len(recorder.payloads)
        result = PiCompactor(model, keep_recent_tokens=keep).compact(context)
        prefix = result.items[0]
        self.assertTrue(_encrypted_reasoning(prefix.prefix_items))
        for payload in recorder.payloads[requests_before:]:
            self.assertFalse(TOOL_FIELDS & payload.keys())

        context.extend(result.context_items())
        context.extend((
            Message("user", "Without looking anything up again, reply with the project "
                            "codename, a space, and the secret code."),
            UserInteractionBoundary(),
        ))
        continuation = len(recorder.payloads)
        answer = _run_to_answer(model, context).last_assistant_text
        self.assertIn("BLUEBIRD", answer.upper(), f"from the summary: {answer!r}")
        self.assertIn("ZEBRA-42", answer, f"from the kept turn: {answer!r}")
        self.assertTrue([
            item for item in recorder.payloads[continuation]["input"]
            if item.get("type") == "reasoning" and item.get("encrypted_content")
        ], "the continuation did not replay the kept encrypted reasoning")

    def test_cli_headless_text_task(self):
        with tempfile.TemporaryDirectory() as directory:
            save = Path(directory) / "live.jsonl"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli.main([
                    "--headless", "--endpoint-api", "responses", "--model", MODEL,
                    "--cwd", directory, "--save", str(save), "--resume=False",
                    "--enable-default-tools=False",
                    "--prompt", "Reply with the single word: pong",
                ])
            self.assertEqual(code, 0, stderr.getvalue())
            answers = [item.content for item in load_interaction_save(save).items
                       if isinstance(item, Message) and item.role == "assistant"]
        self.assertIn("pong", " ".join(answers).lower())


if __name__ == "__main__":
    unittest.main()
