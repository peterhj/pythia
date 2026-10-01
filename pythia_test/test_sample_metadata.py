from __future__ import annotations

from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.interaction_helpers import messages_endpoint
from pythia_test.interaction_helpers import responses_endpoint
from pythia_test.interaction_helpers import codex_model

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import CompactionMetadata
from pythia.interaction import CompactionResult
from pythia.interaction import CompactionSettings
from pythia.interaction import CodexResponsesModel
from pythia.interaction import ConfigError
from pythia.interaction import ContextPrefix
from pythia.interaction import Environment
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import InteractionContext
from pythia.interaction import ModelContextWindowError
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import ModelTimeoutError
from pythia.interaction import NothingToCompact
from pythia.interaction import SampleMetadata
from pythia.interaction import SaveError
from pythia.interaction import SampleParams
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import TokenUsage
from pythia.interaction import TurnSummary
from pythia.interaction import chat_completions
from pythia.interaction import demo
from pythia.interaction import interaction_item_from_dict
from pythia.interaction import interaction_item_to_dict
from pythia.interaction import load_interaction_save
from pythia.interaction import messages
from pythia.interaction import render_interaction_items
from pythia.interaction import responses
from pythia.interaction import save_interaction_save
from pythia.interaction import summarize_turn_usage


_USAGE = TokenUsage(20, 5, 25, 4)
_BASE_RECORD = {
    "usage": {
        "input_tokens": 20, "output_tokens": 5,
        "total_tokens": 25, "cached_input_tokens": 4,
    },
    "provider_session_id": "session-private",
    "provider_turn_id": "turn-private",
    "provider_turn_state": "state-private",
}
_BAD_TYPES = (True, False, "12.5", [], {})
_BAD_NUMBERS = (-1, -0.1, float("nan"), float("inf"), -float("inf"), 10**1000)


class SampleMetadataTests(unittest.TestCase):
    def test_sampling_auto_compaction_override_is_optional_boolean(self):
        for value in (None, False, True):
            self.assertIs(
                SampleParams(enable_auto_compaction=value).enable_auto_compaction,
                value,
            )
        for value in (0, 1, "false", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(
                TypeError,
                "enable_auto_compaction",
            ):
                SampleParams(enable_auto_compaction=value)

    def test_elapsed_validation_and_unknown_default(self):
        for item_type, fields in (
            (SampleMetadata, {"usage": _USAGE}),
            (CompactionMetadata, {
                "usage": _USAGE,
                "protocol": "responses_compaction_v2",
            }),
            (ModelSample, {"items": (Message("assistant", "Done."),)}),
            (TurnSummary, {}),
        ):
            with self.subTest(item_type=item_type):
                self.assertIsNone(item_type(**fields).elapsed_seconds)
                for value in (None, 0, 0.0, 12, 12.3456789):
                    item = item_type(**fields, elapsed_seconds=value)
                    self.assertEqual(item.elapsed_seconds, value)
                    if value is not None:
                        self.assertIsInstance(item.elapsed_seconds, float)
                for value in _BAD_TYPES:
                    with self.assertRaisesRegex(TypeError, "elapsed_seconds"):
                        item_type(**fields, elapsed_seconds=value)
                for value in _BAD_NUMBERS:
                    with self.assertRaisesRegex(ValueError, "elapsed_seconds"):
                        item_type(**fields, elapsed_seconds=value)

    def test_elapsed_is_appended_after_existing_positional_fields(self):
        metadata = SampleMetadata(_USAGE, "session", "turn", "state")
        sample = ModelSample((Message("assistant", "Done."),), "end_turn",
                             _USAGE, "session", "turn", "state")
        self.assertIsNone(metadata.elapsed_seconds)
        self.assertEqual(sample.context_items()[-2], metadata)

    def test_codec_reads_legacy_and_canonical_types_and_writes_canonical(self):
        for kind in ("turn_metadata", "sample_metadata"):
            for timing in ({}, {"elapsed_seconds": None}, {"elapsed_seconds": 0},
                           {"elapsed_seconds": 12.3456789}):
                with self.subTest(kind=kind, timing=timing):
                    item = interaction_item_from_dict({"type": kind, **_BASE_RECORD, **timing})
                    self.assertIsInstance(item, SampleMetadata)
                    self.assertEqual(item.usage, _USAGE)
                    self.assertEqual(item.elapsed_seconds, timing.get("elapsed_seconds"))
                    canonical = {"type": "sample_metadata", **_BASE_RECORD}
                    if item.elapsed_seconds is not None:
                        canonical["elapsed_seconds"] = item.elapsed_seconds
                    self.assertEqual(interaction_item_to_dict(item), canonical)

    def test_codec_rejects_invalid_elapsed_values(self):
        for kind in ("turn_metadata", "sample_metadata"):
            for value in (*_BAD_TYPES, *_BAD_NUMBERS):
                with self.subTest(kind=kind, value=value):
                    with self.assertRaisesRegex(SaveError, kind + r"\.elapsed_seconds"):
                        interaction_item_from_dict({
                            "type": kind, **_BASE_RECORD, "elapsed_seconds": value,
                        })

    def test_compaction_metadata_codec_display_and_private_fields(self):
        metadata = CompactionMetadata(
            usage=_USAGE,
            protocol="responses_compaction_v2",
            provider_session_id="session-private",
            provider_turn_id="turn-private",
            provider_turn_state="state-private",
            provider_response_id="response-private",
            elapsed_seconds=86.25,
            request_attempts=2,
            recovery=("http_500_retry",),
        )
        encoded = interaction_item_to_dict(metadata)
        self.assertEqual(encoded, {
            "type": "compaction_metadata",
            **_BASE_RECORD,
            "protocol": "responses_compaction_v2",
            "provider_response_id": "response-private",
            "elapsed_seconds": 86.25,
            "request_attempts": 2,
            "recovery": ["http_500_retry"],
        })
        self.assertEqual(interaction_item_from_dict(encoded), metadata)
        self.assertEqual(
            render_interaction_items((metadata,))[0].text,
            "[compaction] protocol=responses_compaction_v2 "
            "input=20 output=5 total=25 cached=4 elapsed=86.25s "
            "attempts=2 recovery=http_500_retry",
        )
        for private in (
            "session-private",
            "turn-private",
            "state-private",
            "response-private",
        ):
            self.assertNotIn(private, repr(metadata))
            self.assertNotIn(
                private,
                render_interaction_items((metadata,))[0].text,
            )

        minimal = CompactionMetadata(TokenUsage(), "prompt_summarization")
        self.assertEqual(interaction_item_to_dict(minimal), {
            "type": "compaction_metadata",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
            },
            "protocol": "prompt_summarization",
        })
        for protocol in (
            "",
            " ",
            "bad\nprotocol",
            "responses/compaction_v2",
            "responses.compaction_v2",
            "responses-compaction-v2",
            "Responses_compaction_v2",
            "x" * 129,
        ):
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                CompactionMetadata(TokenUsage(), protocol)

    def test_mixed_legacy_and_nested_jsonl_normalizes_only_on_save(self):
        records = [
            {"type": "message", "role": "user", "text": "Old request."},
            {"type": "turn_metadata", **_BASE_RECORD},
            {"type": "sample_metadata", **_BASE_RECORD, "elapsed_seconds": 12.3456789},
            {
                "type": "context_compaction",
                "replacement_items": [
                    {"type": "message", "role": "user", "content": "Summary."},
                    {"type": "turn_metadata", **_BASE_RECORD},
                    {"type": "sample_metadata", **_BASE_RECORD, "elapsed_seconds": 0},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interaction.jsonl"
            original = "".join(json.dumps(record) + "\n" for record in records)
            path.write_text(original, encoding="utf-8")
            restored = load_interaction_save(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertIsInstance(restored[1], SampleMetadata)
            self.assertIsNone(restored[1].elapsed_seconds)
            self.assertEqual(restored[2].elapsed_seconds, 12.3456789)
            checkpoint = restored[3]
            self.assertIsInstance(checkpoint, ContextPrefix)
            self.assertIsInstance(checkpoint.prefix_items[1], SampleMetadata)
            self.assertIsNone(checkpoint.prefix_items[1].elapsed_seconds)
            self.assertEqual(checkpoint.prefix_items[2].elapsed_seconds, 0.0)
            save_interaction_save(path, restored)
            encoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for record in (encoded[1], encoded[2], *encoded[3]["prefix_items"][1:]):
                self.assertEqual(record["type"], "sample_metadata")
            self.assertNotIn("elapsed_seconds", encoded[1])
            self.assertEqual(load_interaction_save(path).items, restored.items)

    def test_live_and_replay_display_share_timing_and_hide_provider_fields(self):
        for elapsed, suffix in (
            (None, ""), (0, " elapsed=0.00s"), (12.3456789, " elapsed=12.35s"),
        ):
            with self.subTest(elapsed=elapsed), tempfile.TemporaryDirectory() as directory:
                sample = ModelSample(
                    items=(Message("assistant", "Done."),), usage=_USAGE,
                    provider_session_id="session-private", provider_turn_id="turn-private",
                    provider_turn_state="state-private", elapsed_seconds=elapsed,
                )
                with mock.patch("pythia.interaction.model.perf_counter",
                                side_effect=AssertionError("must not remeasure")):
                    metadata = sample.context_items()[-2]
                    self.assertEqual(metadata.elapsed_seconds, elapsed)
                    self.assertEqual(sample.context_items()[-1], ModelSampleBoundary())
                    context = InteractionContext(sample.context_items())
                    path = Path(directory) / "interaction.jsonl"
                    save_interaction_save(path, context)
                    replay = render_interaction_items(load_interaction_save(path).items)
                    self.assertEqual(replay, sample.display_items())
                self.assertEqual(
                    replay[-1].text, "[sample] input=20 output=5 total=25 cached=4" + suffix,
                )
                for private in ("session-private", "turn-private", "state-private"):
                    self.assertNotIn(private, repr(sample))
                    self.assertNotIn(private, repr(metadata))
                    self.assertNotIn(private, "\n".join(item.text for item in replay))

    def test_turn_summary_keeps_usage_semantics_and_uses_short_label(self):
        items = tuple(SampleMetadata(_USAGE, elapsed_seconds=value) for value in (None, 1, 2))
        compact_usage = TokenUsage(100, 10, 110, 60)
        compaction = CompactionMetadata(
            compact_usage,
            "responses_compaction_v2",
            elapsed_seconds=4,
        )
        prefix = ContextPrefix((Message("user", "summary"),))
        summary = summarize_turn_usage(
            (*items, prefix, compaction, TurnSummary(sample_count=99)),
            elapsed_seconds=9.5,
        )
        self.assertEqual(summary, TurnSummary(
            input_tokens_sum=160, output_tokens_sum=25, cached_input_tokens_sum=72,
            cached_input_tokens_max=60, non_cached_input_tokens_sum=88,
            context_tokens=25, sample_count=3, compaction_count=1,
            elapsed_seconds=9.5,
        ))
        self.assertEqual(render_interaction_items((summary,))[0].text,
                         "[turn] input_sum=160 output_sum=25 cold_sum=88 "
                         "cached_sum=72 cached_max=60 context=25 samples=3 "
                         "compactions=1 elapsed=9.50s")
        self.assertEqual(interaction_item_to_dict(summary)["type"], "turn_summary")
        self.assertEqual(summary.elapsed_seconds, 9.5)
        self.assertIsNone(summarize_turn_usage(items).elapsed_seconds)
        self.assertNotEqual(
            summary.elapsed_seconds,
            sum(value for value in (1, 2, 4)),
        )

    def test_turn_summary_elapsed_codec_is_optional_and_replay_stable(self):
        base = {
            "type": "turn_summary",
            "input_tokens_sum": 1,
            "output_tokens_sum": 2,
            "cached_input_tokens_sum": 0,
            "cached_input_tokens_max": 0,
            "non_cached_input_tokens_sum": 1,
            "context_tokens": 3,
            "sample_count": 1,
            "compaction_count": 0,
        }
        legacy = interaction_item_from_dict(base)
        self.assertIsNone(legacy.elapsed_seconds)
        self.assertEqual(interaction_item_to_dict(legacy), base)
        timed = interaction_item_from_dict({**base, "elapsed_seconds": 12.25})
        self.assertEqual(timed.elapsed_seconds, 12.25)
        self.assertEqual(
            interaction_item_to_dict(timed),
            {**base, "elapsed_seconds": 12.25},
        )
        self.assertTrue(
            render_interaction_items((timed,))[0].text.endswith(
                "elapsed=12.25s"
            )
        )
        for value in (*_BAD_TYPES, *_BAD_NUMBERS):
            with self.subTest(value=value), self.assertRaisesRegex(
                SaveError,
                r"turn_summary\.elapsed_seconds",
            ):
                interaction_item_from_dict({**base, "elapsed_seconds": value})

    def test_timed_metadata_remains_invisible_to_provider_content(self):
        items = (Message("user", "Question."), Message("assistant", "Answer."))
        metadata_items = (
            SampleMetadata(_USAGE, elapsed_seconds=12.3456789),
            CompactionMetadata(
                _USAGE,
                "responses_compaction_v2",
                elapsed_seconds=86.25,
            ),
        )
        for metadata in metadata_items:
            if isinstance(metadata, CompactionMetadata):
                self.assertEqual(
                    InteractionContext((*items, metadata)).model_items(),
                    items,
                )
            for encode in (chat_completions._encode_context_messages,
                           messages._encode_context, responses._encode_context_items):
                with self.subTest(
                    encoder=encode.__module__,
                    metadata=type(metadata).__name__,
                ):
                    self.assertEqual(encode((*items, metadata)), encode(items))

    def test_demo_records_independent_active_turn_elapsed_time(self):
        model = mock.Mock(spec=["sample"])
        model.sample.return_value = ModelSample((Message("assistant", "Done."),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interaction.jsonl"
            with mock.patch("builtins.print"):
                with mock.patch.object(
                    demo,
                    "perf_counter",
                    side_effect=(100.0, 112.5),
                ):
                    answer = demo.run(
                        model,
                        Environment(),
                        prompt="Hello.",
                        save_path=path,
                    )
            summary = load_interaction_save(path).items[-1]
        self.assertEqual(answer, "Done.")
        self.assertIsInstance(summary, TurnSummary)
        self.assertEqual(summary.elapsed_seconds, 12.5)

    def test_demo_runs_configured_auto_compaction_before_sample(self):
        class Model:
            auto_compact_context_tokens = 100

            def __init__(self):
                self.contexts = []
                self.sample_params = []

            def sample(self, context, *, tools=(), sample_params=None):
                del tools
                self.contexts.append(context.copy())
                self.sample_params.append(sample_params)
                return ModelSample((Message("assistant", "Done."),))

        model = Model()
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult(
            (ContextPrefix((Message("user", "follow up"),)),),
            usage=TokenUsage(100, 5, 105, 50),
            protocol="responses_compaction_v2",
            elapsed_seconds=3,
        )
        original = InteractionContext((
            Message("user", "old"),
            Message("assistant", "answer"),
            SampleMetadata(TokenUsage(90, 10, 100, 20)),
            ModelSampleBoundary(),
            TurnSummary(sample_count=1, context_tokens=100),
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interaction.jsonl"
            save_interaction_save(path, original)
            with mock.patch("builtins.print"):
                with mock.patch.object(
                    demo,
                    "create_default_compactor",
                    return_value=compactor,
                ):
                    answer = demo.run(
                        model,
                        Environment(),
                        prompt="follow up",
                        save_path=path,
                        resume=True,
                    )
            saved = load_interaction_save(path)

        self.assertEqual(answer, "Done.")
        compactor.compact.assert_called_once()
        self.assertEqual(len(model.contexts), 1)
        self.assertEqual(
            model.sample_params,
            [SampleParams(auto_compact_tokens=100, enable_auto_compaction=True)],
        )
        self.assertEqual(model.contexts[0].model_items(), (
            Message("user", "follow up"),
        ))
        self.assertEqual(
            len([item for item in saved if isinstance(item, CompactionMetadata)]),
            1,
        )

    def test_demo_binds_compaction_keywords_and_recovers_from_overflow(self):
        class Model:
            def __init__(self, *outcomes):
                self.outcomes = list(outcomes)
                self.sample_params = []

            def sample(self, context, *, tools=(), sample_params=None):
                self.sample_params.append(sample_params)
                outcome = self.outcomes.pop(0)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult(
            (ContextPrefix((Message("user", "summary"),)),), protocol="pi",
        )
        model = Model(ModelContextWindowError("too long"), ModelSample((Message("assistant", "Done."),)))
        with mock.patch("builtins.print"):
            with mock.patch.object(demo, "create_default_compactor", return_value=compactor) as create:
                answer = demo.run(
                    model, Environment(), prompt="hello", max_samples=1,
                    compaction_mode="pi", compaction_keep_recent_tokens=0,
                    compaction_max_output_tokens=64,
                )
        self.assertEqual(answer, "Done.")
        # One compact-and-retry; the failed attempt does not count.
        create.assert_called_once_with(model, CompactionSettings(
            mode="pi", keep_recent_tokens=0, max_output_tokens=64,
        ))
        self.assertEqual(compactor.compact.call_args.kwargs["sample_params"], model.sample_params[0])
        self.assertEqual(len(model.sample_params), 2)

        # Nothing to compact leaves the sampling error.
        compactor.compact.side_effect = NothingToCompact("the context fits")
        with mock.patch("builtins.print"):
            with mock.patch.object(demo, "create_default_compactor", return_value=compactor):
                with self.assertRaises(ModelContextWindowError):
                    demo.run(Model(ModelContextWindowError("too long")), Environment(), prompt="hello")
        with self.assertRaisesRegex(ConfigError, "no provider compaction"):
            demo.run(Model(), Environment(), prompt="hello", compaction_mode="provider")

    def test_demo_auto_compaction_can_be_disabled(self):
        class Model:
            auto_compact_context_tokens = 100

            def __init__(self):
                self.contexts = []
                self.sample_params = []

            def sample(self, context, *, tools=(), sample_params=None):
                del tools
                self.contexts.append(context.copy())
                self.sample_params.append(sample_params)
                return ModelSample((Message("assistant", "Done."),))

        model = Model()
        original = InteractionContext((
            Message("assistant", "uncompacted"),
            SampleMetadata(TokenUsage(total_tokens=100)),
            ModelSampleBoundary(),
            TurnSummary(sample_count=1, context_tokens=100),
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interaction.jsonl"
            save_interaction_save(path, original)
            with mock.patch("builtins.print"):
                with mock.patch.object(demo, "create_default_compactor") as create:
                    answer = demo.run(
                        model,
                        Environment(),
                        prompt="follow up",
                        save_path=path,
                        resume=True,
                        enable_auto_compaction=False,
                    )

        self.assertEqual(answer, "Done.")
        create.assert_not_called()
        self.assertEqual(len(model.contexts), 1)
        self.assertEqual(
            model.sample_params,
            [SampleParams(enable_auto_compaction=False, auto_compact_tokens=100)],
        )
        self.assertIn(
            Message("assistant", "uncompacted"),
            model.contexts[0].model_items(),
        )


class _Clock:
    def __init__(self):
        self.now = 100.0
        self.reads = []

    def __call__(self):
        self.reads.append(self.now)
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _TimedResponse:
    status = 200
    headers = {"x-codex-turn-state": "state-private"}

    def __init__(self, name, clock, *, fail=False):
        self.clock = clock
        self.fail = fail
        self.closed = False
        self.body = (
            {"choices": [{"message": {"role": "assistant", "content": "Done."},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25,
                       "prompt_tokens_details": {"cached_tokens": 4}}}
            if name == "chat" else
            {"type": "message", "role": "assistant", "stop_reason": "end_turn",
             "content": [{"type": "text", "text": "Done."}],
             "usage": {"input_tokens": 16, "cache_read_input_tokens": 4, "output_tokens": 5}}
        )

    def read(self):
        self.clock.advance(3)
        if self.fail:
            raise TimeoutError("offline body timeout")
        return json.dumps(self.body).encode()

    def __iter__(self):
        self.clock.advance(1)
        yield "data: " + json.dumps({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Done."}]},
        }) + "\n"
        yield "\n"
        self.clock.advance(2)
        if self.fail:
            raise TimeoutError("offline stream timeout")
        yield "data: " + json.dumps({
            "type": "response.completed", "response": {"usage": {
                "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                "input_tokens_details": {"cached_tokens": 4},
            }},
        }) + "\n"
        yield "\n"

    def close(self):
        self.clock.advance(5)
        self.closed = True


def _timing_case(name, clock, *, fail=False):
    response = _TimedResponse(name, clock, fail=fail)

    def opener(request, *, timeout):
        clock.advance(2)
        return response

    if name == "chat":
        model = ChatCompletionsModel(chat_endpoint("http://localhost"), opener=opener)
        return model, response, chat_completions, "_decode_response"
    if name == "messages":
        model = MessagesModel(
            messages_endpoint(
                "http://localhost",
                "model",
                max_output_tokens=100,
            ),
            opener=opener,
        )
        return model, response, messages, "_decode_response"
    identifiers = iter(("session-private", "turn-private"))
    model = codex_model(responses_endpoint(
        api_url="http://localhost", model="model", bearer_token="FAKE",
        api_provider="codex" if name == "codex" else "api",
    ), opener=opener, identifier_factory=lambda: next(identifiers))
    return model, response, responses, "_collect_sample"


class SampleTimingTests(unittest.TestCase):
    def test_full_adapter_call_is_timed_through_decode_and_cleanup(self):
        for name in ("chat", "messages", "responses", "codex"):
            with self.subTest(name=name):
                clock = _Clock()
                model, response, provider, decoder_name = _timing_case(name, clock)
                build = model._build_request_payload
                decode = getattr(provider, decoder_name)

                def timed_build(*args, **kwargs):
                    clock.advance(1)
                    return build(*args, **kwargs)

                def timed_decode(*args, **kwargs):
                    clock.advance(4)
                    return decode(*args, **kwargs)

                context = InteractionContext((Message("user", "Hello."),))
                before = context.items
                with ExitStack() as stack:
                    stack.enter_context(mock.patch("pythia.interaction.model.perf_counter", clock))
                    stack.enter_context(mock.patch.object(model, "_build_request_payload", timed_build))
                    stack.enter_context(mock.patch.object(provider, decoder_name, timed_decode))
                    sample = model.sample(context)
                    self.assertEqual(sample.elapsed_seconds, 15.0)
                    self.assertTrue(response.closed)
                    self.assertEqual(sample.usage, _USAGE)
                    self.assertEqual(sample.stop_reason, "end_turn")
                    self.assertEqual(sample.items, (Message("assistant", "Done."),))
                    self.assertEqual(context.items, before)
                    clock.advance(1000)  # Later tools, saving, and display must not change it.
                    self.assertEqual(sample.context_items()[-2].elapsed_seconds, 15.0)
                    self.assertTrue(sample.display_items()[-1].text.endswith("elapsed=15.00s"))
                    self.assertEqual(clock.reads, [100.0, 115.0])
                if name == "codex":
                    metadata = sample.context_items()[-2]
                    self.assertEqual(metadata.provider_session_id, "session-private")
                    self.assertEqual(metadata.provider_turn_id, "turn-private")
                    self.assertEqual(metadata.provider_turn_state, "state-private")

    def test_timeouts_close_responses_without_producing_sample_metadata(self):
        for name in ("chat", "messages", "responses", "codex"):
            with self.subTest(name=name):
                clock = _Clock()
                model, response, _, _ = _timing_case(name, clock, fail=True)
                context = InteractionContext((Message("user", "Hello."),))
                before = context.items
                with mock.patch("pythia.interaction.model.perf_counter", clock):
                    with self.assertRaises(ModelTimeoutError):
                        model.sample(context)
                self.assertTrue(response.closed)
                self.assertEqual(context.items, before)
                self.assertEqual(clock.reads, [100.0])

    def test_messages_compaction_result_gets_one_sample_measurement(self):
        clock = _Clock()
        model, response, _, _ = _timing_case("messages", clock)
        response.body.update(stop_reason="compaction", content=[{
            "type": "compaction", "content": "Summary.",
        }])
        with mock.patch("pythia.interaction.model.perf_counter", clock):
            sample = model.sample(InteractionContext((Message("user", "Hello."),)))
        self.assertEqual(sample.stop_reason, "compaction")
        self.assertEqual(sample.elapsed_seconds, 10.0)
        self.assertEqual(
            [item for item in sample.context_items() if isinstance(item, SampleMetadata)],
            [SampleMetadata(_USAGE, elapsed_seconds=10.0)],
        )
        self.assertEqual(summarize_turn_usage(sample.context_items()).sample_count, 1)
        self.assertEqual(clock.reads, [100.0, 110.0])


if __name__ == "__main__":
    unittest.main()
