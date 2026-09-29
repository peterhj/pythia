"""Experimental media user messages: parsing, codec, encoders, CLI."""

from __future__ import annotations

import base64
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import (
    Environment,
    MediaPart,
    InteractionContext,
    Message,
    ModelConfigurationError,
    SaveError,
    TextPart,
    cli,
    demo,
    interaction_item_from_dict,
    interaction_item_to_dict,
    load_interaction_save,
)
from pythia.interaction.chat_completions import _encode_context_messages
from pythia.interaction.messages import _encode_context
from pythia.interaction.media import (
    AttachmentError,
    ContentLimits,
    content_item_to_responses,
    parse_user_prompt,
    resolve_content,
    split_leading_references,
)
from pythia.interaction.responses import _encode_context_items
from pythia_test.test_interaction_cli import _Model, _answer


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _write_png(path, data=_PNG):
    path.write_bytes(data)
    return path


class SplitLeadingReferencesTests(unittest.TestCase):
    def test_no_leading_reference_is_identity(self):
        for prompt in ("hello", "", "   "):
            with self.subTest(prompt=prompt):
                self.assertEqual(split_leading_references(prompt), ((), prompt))

    def test_leading_references_and_text(self):
        self.assertEqual(
            split_leading_references("@a.png @b.jpg  describe  this"),
            (("a.png", "b.jpg"), "describe  this"),
        )
        self.assertEqual(split_leading_references("  @a.png tail"), (("a.png",), "tail"))

    def test_attachment_only(self):
        self.assertEqual(split_leading_references("@a.png"), (("a.png",), ""))
        self.assertEqual(split_leading_references("@a.png   "), (("a.png",), ""))

    def test_bare_at_and_escape_end_the_run(self):
        self.assertEqual(split_leading_references("@ @x"), ((), "@ @x"))
        self.assertEqual(split_leading_references("@"), ((), "@"))
        self.assertEqual(split_leading_references("@@literal"), ((), "@literal"))
        self.assertEqual(
            split_leading_references("@a.png @@lit"),
            (("a.png",), "@lit"),
        )

    def test_non_at_token_ends_the_run(self):
        self.assertEqual(
            split_leading_references("@a.png\nb.png rest"),
            (("a.png",), "b.png rest"),
        )


class ResolveContentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)

    def resolve(self, reference, **kwargs):
        return resolve_content(
            (reference,),
            cwd=self.cwd,
            enable_workspace=True,
            **kwargs,
        )

    def test_local_image_is_inlined_as_data_url(self):
        _write_png(self.cwd / "pic.png")
        parts = self.resolve("pic.png")
        self.assertEqual(len(parts), 1)
        self.assertIsInstance(parts[0], MediaPart)
        self.assertTrue(parts[0].source_uri.startswith("data:image/png;base64,"))
        encoded = parts[0].source_uri.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), _PNG)

    def test_magic_sniff_when_extension_is_unknown(self):
        _write_png(self.cwd / "blob.dat")
        (part,) = self.resolve("blob.dat")
        self.assertTrue(part.source_uri.startswith("data:image/png;base64,"))

    def test_absolute_path_resolution(self):
        path = _write_png(self.cwd / "pic.png")
        (part,) = self.resolve(str(path))
        self.assertTrue(part.source_uri.startswith("data:image/png;"))

    def test_workspace_containment(self):
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside = _write_png(Path(outside_dir.name) / "out.png")
        with self.assertRaisesRegex(AttachmentError, "outside --cwd"):
            resolve_content((str(outside),), cwd=self.cwd, enable_workspace=True)
        parts = resolve_content((str(outside),), cwd=self.cwd, enable_workspace=False)
        self.assertTrue(parts[0].source_uri.startswith("data:image/png;"))

    def test_missing_directory_and_non_image_are_rejected(self):
        (self.cwd / "empty.png").mkdir()
        (self.cwd / "notes.txt").write_text("hello")
        for reference, message in (
            ("missing.png", "cannot read attachment"),
            ("empty.png", "not a regular file"),
            ("notes.txt", "not a supported image"),
        ):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(AttachmentError, message):
                    self.resolve(reference)

    def test_limits(self):
        _write_png(self.cwd / "pic.png")
        with self.assertRaisesRegex(AttachmentError, "too large"):
            self.resolve("pic.png", limits=ContentLimits(max_item_bytes=4))
        with self.assertRaisesRegex(AttachmentError, "too many"):
            resolve_content(
                ("a", "b", "c"),
                cwd=self.cwd,
                enable_workspace=True,
                limits=ContentLimits(max_items=2),
            )
        with self.assertRaisesRegex(AttachmentError, "in total"):
            resolve_content(
                ("pic.png", "pic.png"),
                cwd=self.cwd,
                enable_workspace=True,
                limits=ContentLimits(max_total_bytes=len(_PNG)),
            )

    def test_remote_urls(self):
        (part,) = self.resolve("https://example.test/a.png")
        self.assertEqual(part.source_uri, "https://example.test/a.png")
        for reference in ("https://example.test/a.txt", "https://x"):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(
                    AttachmentError, "not a supported image|no host"
                ):
                    self.resolve(reference)

    def test_other_schemes_rejected(self):
        for reference in ("data:image/png;base64,AA", "file:///etc/passwd"):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(AttachmentError, "unsupported URL scheme"):
                    self.resolve(reference)

    def test_errors_never_embed_payload(self):
        _write_png(self.cwd / "pic.png")
        with self.assertRaises(AttachmentError) as raised:
            self.resolve("pic.png", limits=ContentLimits(max_item_bytes=4))
        self.assertNotIn(
            base64.b64encode(_PNG).decode("ascii"), str(raised.exception)
        )


class ParseUserPromptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        _write_png(self.cwd / "pic.png")

    def test_disabled_is_identity_and_reads_nothing(self):
        message = parse_user_prompt(
            "@missing.png describe",
            cwd=self.cwd,
            enabled=False,
            enable_workspace=True,
        )
        self.assertEqual(message, Message(role="user", content="@missing.png describe"))

    def test_enabled_without_references_is_identity(self):
        message = parse_user_prompt(
            "plain text", cwd=self.cwd, enabled=True, enable_workspace=True
        )
        self.assertEqual(message.content, "plain text")

    def test_enabled_builds_parts_in_order(self):
        message = parse_user_prompt(
            "@pic.png describe it",
            cwd=self.cwd,
            enabled=True,
            enable_workspace=True,
        )
        self.assertIsInstance(message.content, tuple)
        self.assertIsInstance(message.content[0], MediaPart)
        self.assertEqual(message.content[1], TextPart("describe it"))
        self.assertEqual(message.content_text, "describe it")

    def test_attachment_only_and_escape(self):
        message = parse_user_prompt(
            "@pic.png", cwd=self.cwd, enabled=True, enable_workspace=True
        )
        self.assertEqual(len(message.content), 1)
        escaped = parse_user_prompt(
            "@@literal", cwd=self.cwd, enabled=True, enable_workspace=True
        )
        self.assertEqual(escaped.content, "@literal")


class MessageModelTests(unittest.TestCase):
    def test_string_content_is_unchanged(self):
        message = Message(role="user", content="hello")
        self.assertEqual(message.content, "hello")
        self.assertEqual(message.content_text, "hello")
        self.assertEqual(message.parts, (TextPart("hello"),))
        self.assertFalse(message.has_media)
        self.assertFalse(hasattr(message, "text"))

    def test_text_only_tuple_canonicalizes(self):
        message = Message(role="user", content=(TextPart("a"), TextPart("b")))
        self.assertEqual(message.content, "a\nb")
        self.assertFalse(message.has_media)

    def test_mixed_tuple_preserved_in_order(self):
        parts = (MediaPart("data:image/png;base64,AA"), TextPart("look"))
        message = Message(role="user", content=parts)
        self.assertEqual(message.content, parts)
        self.assertEqual(message.content_text, "look")
        self.assertTrue(message.has_media)

    def test_non_user_media_rejected(self):
        with self.assertRaisesRegex(ValueError, "require role 'user'"):
            Message(role="assistant", content=(MediaPart("x"),))

    def test_invalid_content_rejected(self):
        for content in (None, True, 123, [], {}, [{"type": "text", "text": "x"}]):
            with self.subTest(content=content):
                with self.assertRaisesRegex(TypeError, "content must be a string"):
                    Message(role="user", content=content)


class CodecTests(unittest.TestCase):
    def test_string_message_round_trips_unchanged(self):
        for role in ("user", "assistant"):
            message = Message(role=role, content="hello")
            self.assertEqual(
                interaction_item_to_dict(message),
                {"type": "message", "role": role, "content": "hello"},
            )

    def test_media_message_round_trips_in_memory_shape(self):
        message = Message(
            role="user",
            content=(MediaPart("data:image/png;base64,AA"), TextPart("look")),
        )
        encoded = interaction_item_to_dict(message)
        self.assertEqual(
            encoded,
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "media", "source_uri": "data:image/png;base64,AA"},
                    {"type": "text", "text": "look"},
                ],
            },
        )
        self.assertEqual(interaction_item_from_dict(encoded), message)

    def test_attachment_only_omits_text_part(self):
        message = Message(role="user", content=(MediaPart("http://x/y.png"),))
        encoded = interaction_item_to_dict(message)
        self.assertEqual(
            encoded["content"],
            [{"type": "media", "source_uri": "http://x/y.png"}],
        )
        self.assertEqual(interaction_item_from_dict(encoded), message)

    def test_text_only_array_loads_as_string(self):
        item = interaction_item_from_dict(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            }
        )
        self.assertEqual(item, Message(role="user", content="hi"))

    def test_invalid_arrays_rejected(self):
        for content, message in (
            ([], "must not be empty"),
            ([{"type": "input_image", "image_url": "x"}], "unsupported"),
            ([{"type": "nope"}], "unsupported"),
            ([{"type": "text"}], "must be a string"),
            ([{"type": "media"}], "source_uri"),
        ):
            with self.subTest(content=content):
                with self.assertRaisesRegex(SaveError, message):
                    interaction_item_from_dict(
                        {"type": "message", "role": "user", "content": content}
                    )


class EncoderTests(unittest.TestCase):
    def test_responses_parts_and_role_text(self):
        context = InteractionContext(
            (
                Message(
                    role="user",
                    content=(MediaPart("data:image/png;base64,AA"), TextPart("look")),
                ),
                Message(role="assistant", content="done"),
            )
        )
        encoded = _encode_context_items(context.model_items(), system_role="developer")
        self.assertEqual(
            encoded,
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,AA"},
                        {"type": "input_text", "text": "look"},
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
        )

    def test_chat_completions_parts(self):
        context = InteractionContext(
            (Message(role="user", content=(MediaPart("https://x/y.png"), TextPart("q"))),)
        )
        self.assertEqual(
            _encode_context_messages(context.model_items()),
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                        {"type": "text", "text": "q"},
                    ],
                }
            ],
        )

    def test_string_chat_completions_unchanged(self):
        context = InteractionContext((Message(role="user", content="hi"),))
        self.assertEqual(
            _encode_context_messages(context.model_items()),
            [{"role": "user", "content": "hi"}],
        )

    def test_messages_rejects_media(self):
        context = InteractionContext((Message(role="user", content=(MediaPart("x"),)),))
        with self.assertRaisesRegex(ModelConfigurationError, "not supported"):
            _encode_context(context.model_items())

    def test_content_item_to_responses(self):
        self.assertEqual(
            content_item_to_responses(TextPart("t"), "assistant"),
            {"type": "output_text", "text": "t"},
        )
        self.assertEqual(
            content_item_to_responses(TextPart("t"), "user"),
            {"type": "input_text", "text": "t"},
        )
        self.assertEqual(
            content_item_to_responses(MediaPart("s"), "user"),
            {"type": "input_image", "image_url": "s"},
        )


class OptionParserTests(unittest.TestCase):
    def test_option_forms(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                self.assertIs(
                    frontend._build_parser()
                    .parse_args([])
                    .enable_experimental_media,
                    False,
                )
                self.assertIs(
                    frontend._build_parser()
                    .parse_args(["--enable-experimental-media"])
                    .enable_experimental_media,
                    True,
                )
                self.assertIs(
                    frontend._build_parser()
                    .parse_args(["--enable-experimental-media=False"])
                    .enable_experimental_media,
                    False,
                )


class HeadlessMediaTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "session.jsonl"
        _write_png(self.root / "pic.png")

    def run_main(self, argv, *outcomes):
        model = _Model(self.path, *outcomes)
        stderr = io.StringIO()
        with mock.patch.object(cli, "build_model", return_value=model), \
                mock.patch.object(
                    cli, "PosixTerminal",
                    side_effect=AssertionError("TUI constructed"),
                ), \
                redirect_stderr(stderr):
            code = cli.main(
                ["--headless", "--cwd", str(self.root), "--save", str(self.path), *argv]
            )
        return code, model, stderr.getvalue()

    def _saved_user(self):
        items = load_interaction_save(self.path).items
        return [
            item
            for item in items
            if isinstance(item, Message) and item.role == "user"
        ]

    def test_flag_on_saves_responses_array_and_sends_parts(self):
        code, model, stderr = self.run_main(
            ["--enable-experimental-media", "--prompt", "@pic.png describe"],
            _answer("ok"),
        )
        self.assertEqual(code, 0, stderr)
        (user,) = self._saved_user()
        self.assertIsInstance(user.content, tuple)
        self.assertTrue(user.content[0].source_uri.startswith("data:image/png;base64,"))
        self.assertEqual(user.content[-1], TextPart("describe"))
        sent = model.calls[0][0].items
        sent_user = [
            item
            for item in sent
            if isinstance(item, Message) and item.role == "user"
        ][0]
        self.assertEqual(sent_user.content_text, "describe")
        self.assertTrue(sent_user.has_media)

    def test_flag_off_keeps_literal_text_and_reads_nothing(self):
        code, _model, stderr = self.run_main(
            ["--prompt", "@missing.png describe"], _answer("ok")
        )
        self.assertEqual(code, 0, stderr)
        (user,) = self._saved_user()
        self.assertEqual(user.content, "@missing.png describe")

    def test_attachment_error_exits_nonzero_without_sampling(self):
        code, model, stderr = self.run_main(
            ["--enable-experimental-media", "--prompt", "@missing.png describe"],
            _answer("ok"),
        )
        self.assertEqual(code, 1)
        self.assertEqual(model.calls, [])
        self.assertIn("cannot read attachment", stderr)
        self.assertEqual(self._saved_user(), [])

    def test_messages_api_rejected_before_sampling(self):
        code, model, stderr = self.run_main(
            [
                "--enable-experimental-media",
                "--endpoint-api",
                "messages",
                "--model",
                "claude-test",
                "--max-output-tokens",
                "77",
                "--prompt",
                "@pic.png describe",
            ],
            _answer("ok"),
        )
        self.assertEqual(code, 1)
        self.assertEqual(model.calls, [])
        self.assertIn("does not support media", stderr)


class DisplayTests(unittest.TestCase):
    def test_media_message_renders_without_payload(self):
        from pythia.interaction import render_interaction_items

        message = Message(
            role="user",
            content=(MediaPart("data:image/png;base64," + "A" * 400), TextPart("look")),
        )
        blocks = [item.text for item in render_interaction_items((message,))]
        self.assertEqual(blocks[0], "[user] look")
        self.assertTrue(blocks[1].startswith("[user] [image] image/png"))
        self.assertNotIn("A" * 8, "".join(blocks))

    def test_text_only_message_unchanged(self):
        from pythia.interaction import render_interaction_items

        message = Message(role="user", content="hello")
        self.assertEqual(
            [item.text for item in render_interaction_items((message,))],
            ["[user] hello"],
        )


class DemoMediaTests(unittest.TestCase):
    def test_run_converts_leading_references(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        _write_png(root / "pic.png")
        path = root / "interaction.jsonl"

        class Model:
            def __init__(self):
                self.calls = []

            def sample(self, context, *, tools=(), sample_params=None):
                self.calls.append(context.copy())
                return _answer("ok")

        model = Model()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            text = demo.run(
                model,
                Environment(),
                prompt="@pic.png hello",
                save_path=path,
                enable_media=True,
                enable_workspace=True,
                cwd=root,
            )
        self.assertEqual(text, "ok")
        (user,) = [
            item
            for item in load_interaction_save(path).items
            if isinstance(item, Message) and item.role == "user"
        ]
        self.assertIsInstance(user.content, tuple)
        self.assertEqual(user.content_text, "hello")


if __name__ == "__main__":
    unittest.main()
