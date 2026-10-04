from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pythia.interaction import (
    ChatCompletionsModel, CodexResponsesModel, ContextPrefix, Environment,
    Init, Instructions, InteractionContext, Message, MessagesModel, ModelSample,
    OpaqueCompaction, PiCompactor, ResponsesModel, ResponsesOpaqueCompactor, SaveError, Tool, ToolOutcome,
    Tools, ToolSpec, TurnSummary, UserInteractionBoundary,
    estimate_context_tokens, interaction_item_from_dict, interaction_item_to_dict,
    load_interaction_save, render_interaction_items, save_interaction_save,
)
from pythia.interaction import chat_completions, demo, messages, responses
from pythia_test.interaction_helpers import chat_endpoint, messages_endpoint, responses_endpoint


def spec(name="lookup", description="Look up data."):
    return ToolSpec(name, description, {
        "type": "object", "properties": {"query": {"type": "string"}},
        "required": ["query"], "additionalProperties": False,
    })


class ToolsItemTests(unittest.TestCase):
    def test_round_trip_empty_and_nonempty_snapshots(self):
        for item in (Tools(), Tools([spec(), spec("other", "世界")])):
            with self.subTest(item=item):
                record = {"type": "tools", "specs": [
                    {"name": s.name, "description": s.description, "parameters": s.parameters}
                    for s in item.specs
                ]}
                self.assertEqual(interaction_item_to_dict(item), record)
                self.assertEqual(interaction_item_from_dict(record), item)
                context = InteractionContext((Init("test"), item, Message("user", "hello")))
                with tempfile.TemporaryDirectory() as root:
                    path = Path(root) / "context.jsonl"
                    save_interaction_save(path, context)
                    self.assertEqual(load_interaction_save(path).items, context.items)

    def test_snapshot_and_serialized_record_do_not_alias_runtime_schema(self):
        runtime = spec()
        item = Tools((runtime,))
        expected = interaction_item_to_dict(item)
        runtime.parameters["properties"]["query"]["type"] = "number"
        runtime.parameters["required"].append("extra")
        self.assertEqual(interaction_item_to_dict(item), expected)
        record = interaction_item_to_dict(item)
        record["specs"][0]["parameters"]["required"].clear()
        self.assertEqual(interaction_item_to_dict(item), expected)

    def test_invalid_specs_and_schemas_are_rejected(self):
        for specs in ((object(),), (spec(), spec())):
            with self.subTest(specs=specs), self.assertRaises((TypeError, ValueError)):
                Tools(specs)
        for invalid in (object(), float("nan"), float("inf"), {1: "not a string key"}):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                Tools((ToolSpec("bad", "bad", {"nested": invalid}),))
        cycle = {}
        cycle["cycle"] = cycle
        with self.assertRaisesRegex(ValueError, "nesting depth"):
            Tools((ToolSpec("bad", "bad", cycle),))

    def test_malformed_records_fail_with_save_errors(self):
        valid = interaction_item_to_dict(Tools((spec(),)))["specs"][0]
        for entries in (None, {}, [None], [{}], [valid, valid],
                        [{**valid, "parameters": []}],
                        [{**valid, "parameters": {"bad": float("nan")}}]):
            with self.subTest(entries=entries), self.assertRaisesRegex(SaveError, "tools.specs"):
                interaction_item_from_dict({"type": "tools", "specs": entries})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "context.jsonl"
            before = json.dumps({"type": "tools", "specs": [valid, valid]}) + "\n"
            path.write_text(before)
            with self.assertRaisesRegex(SaveError, "line 1"):
                load_interaction_save(path)
            self.assertEqual(path.read_text(), before)

    def test_raw_latest_survives_compaction_but_never_enters_model_view(self):
        older, latest = Tools((spec(),)), Tools()
        message = Message("user", "summary")
        context = InteractionContext((Init("test"), older, latest, ContextPrefix((message,))))
        self.assertEqual(context.latest_tools(), latest)
        self.assertEqual(context.model_items(), (message,))
        self.assertEqual(estimate_context_tokens(context), estimate_context_tokens(InteractionContext((message,))))
        self.assertEqual(context.items[1:3], (older, latest))
        self.assertIsNone(InteractionContext((message,)).latest_tools())
        context.append(older)
        self.assertEqual(context.latest_tools(), older)
        self.assertEqual(context.model_items(), (message,))

    def test_rendering_is_concise_and_shows_explicit_empty(self):
        rendered = render_interaction_items((Tools((spec(), spec("other"))), Tools()))
        self.assertEqual([item.text for item in rendered], ["[tools] lookup, other", "[tools] (none)"])
        self.assertEqual([item.label for item in rendered], ["tools", "tools"])

    def test_provider_encoders_skip_snapshots_without_splitting_messages(self):
        plain = (Instructions("rules"), Message("user", "hello"),
                 Message("assistant", "one"), Message("assistant", "two"))
        logged = (Tools(), plain[0], Tools((spec(),)), *plain[1:3], Tools(), plain[3])
        for encode in (chat_completions._encode_context_messages,
                       messages._encode_context, responses._encode_context_items):
            with self.subTest(encode=encode):
                self.assertEqual(encode(logged), encode(plain))
        compacted = (*plain, OpaqueCompaction.from_messages("checkpoint"), Message("user", "next"))
        self.assertEqual(messages._encode_context((Tools((spec(),)), *compacted)),
                         messages._encode_context(compacted))

    def test_requests_use_runtime_tools_never_logged_tools(self):
        models = (
            ChatCompletionsModel(chat_endpoint(model="test")),
            MessagesModel(messages_endpoint("https://example.test", "test", api_key="test", max_output_tokens=1024)),
            ResponsesModel(responses_endpoint("https://example.test", "test")),
            CodexResponsesModel(responses_endpoint("https://example.test", "test", api_provider="codex")),
        )
        for model in models:
            for runtime in ((), (spec("runtime"),)):
                for logged in (Tools(), Tools((spec("old"),))):
                    with self.subTest(model=type(model).__name__, runtime=runtime, logged=logged):
                        context = InteractionContext((Init("test"), logged, Message("user", "hello")))
                        before = context.items
                        payload = model._build_request_payload(context, runtime, None)
                        if isinstance(payload, tuple):
                            payload = payload[0]
                        wire_tools = payload.get("tools", [])
                        self.assertEqual(len(wire_tools), len(runtime))
                        if runtime:
                            self.assertEqual(wire_tools[0].get("function", wire_tools[0])["name"], "runtime")
                        self.assertNotIn("old", json.dumps(payload))
                        self.assertEqual(context.items, before)

    def test_pi_compaction_retains_raw_audit_not_tools_in_summary_or_prefix(self):
        snapshot = Tools((spec(),))
        context = InteractionContext((Init("test"), snapshot,
                                      Message("user", "old task"), UserInteractionBoundary(),
                                      Message("assistant", "old answer"), TurnSummary(),
                                      Message("user", "new task"), UserInteractionBoundary()))
        model = mock.Mock(spec=["sample"])
        model.sample.return_value = ModelSample((Message("assistant", "summary"),), stop_reason="end_turn")
        result = PiCompactor(model, keep_recent_tokens=0).compact(context, tools=(spec("runtime"),))
        for call in model.sample.call_args_list:
            self.assertEqual(call.kwargs["tools"], ())
            self.assertFalse(any(isinstance(item, Tools) for item in call.args[0].model_items()))
        context.extend(result.context_items())
        self.assertEqual(context.latest_tools(), snapshot)
        self.assertFalse(any(isinstance(item, Tools) for item in context.model_items()))

    def test_remote_compaction_keeps_raw_snapshot_and_explicit_runtime_tools(self):
        snapshot = Tools((spec("old"),))
        runtime = (spec("runtime"),)
        context = InteractionContext((Init("test"), snapshot, Message("user", "hello")))
        model = CodexResponsesModel(responses_endpoint(
            "https://example.test", "test", api_provider="codex",
        ))
        response = responses._RemoteCompactionResponse(OpaqueCompaction.from_responses("checkpoint"))
        with mock.patch.object(model, "_compact_responses_v2", return_value=response) as compact:
            result = ResponsesOpaqueCompactor(model).compact(context, tools=runtime)
        self.assertIs(compact.call_args.args[0], context)
        self.assertEqual(compact.call_args.args[1], runtime)
        context.extend(result.context_items())
        self.assertEqual(context.latest_tools(), snapshot)
        self.assertFalse(any(isinstance(item, Tools) for item in context.model_items()))


class DemoToolsSnapshotTests(unittest.TestCase):
    def test_fresh_resume_change_clear_and_legacy_upgrade(self):
        handler = mock.Mock(return_value=ToolOutcome("not called"))
        runtime = spec("current")
        environment = Environment((Tool(runtime, handler),))
        current = Tools(environment.tool_specs)
        for previous in (None, current, Tools((spec("old"),)), Tools()):
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "session.jsonl"
                initial = (Init("test"), *((previous,) if previous is not None else ()),
                           ContextPrefix((Message("assistant", "saved answer"),)), TurnSummary())
                save_interaction_save(path, InteractionContext(initial))
                model = mock.Mock(spec=["sample"])
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(demo.run(model, environment, save_path=path, resume=True, prompt=None), "saved answer")
                expected = initial if previous == current else (*initial, current)
                self.assertEqual(load_interaction_save(path).items, expected)
                model.sample.assert_not_called()
                # Same runtime does not append again, even after compaction.
                with redirect_stdout(io.StringIO()):
                    demo.run(model, environment, save_path=path, resume=True, prompt=None)
                self.assertEqual(load_interaction_save(path).items, expected)
                with redirect_stdout(io.StringIO()):
                    demo.run(model, Environment(), save_path=path, resume=True, prompt=None)
                self.assertEqual(load_interaction_save(path).items, (*expected, Tools()))
                handler.assert_not_called()

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "session.jsonl"
            def sample(context, *, tools, sample_params):
                self.assertEqual(context.latest_tools(), current)
                self.assertEqual(tuple(tools), environment.tool_specs)
                self.assertEqual(load_interaction_save(path).items, context.items)
                return ModelSample((Message("assistant", "done"),), stop_reason="end_turn")
            model.sample.side_effect = sample
            with redirect_stdout(io.StringIO()):
                demo.run(model, environment, save_path=path, prompt="hello", enable_auto_compaction=False)
            self.assertEqual(load_interaction_save(path).items[1], current)
