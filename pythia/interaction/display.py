"""Static display formatting for interaction item batches."""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from .items import CompactionMetadata
from .items import ContextPrefix
from .items import Init
from .items import Instructions
from .items import InteractionItem
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import ToolResult
from .items import SampleMetadata
from .items import TextPart
from .items import TurnSummary
from .items import UserInteractionBoundary
from .items import UserToolCall
from .items import UserToolResult
from .items import is_interaction_item


_SHELL_TOOL_NAMES = frozenset(
    {
        "shell",
        "container.exec",
        "local_shell",
        "shell_command",
        "exec_command",
        "write_stdin",
    }
)

_CODE_EDIT_TOOL_NAMES = frozenset(
    {
        "apply_patch",
        "create_file",
        "edit_file",
        "insert_into_file",
        "replace_in_file",
        "write_file",
    }
)

_SHELL_COMMAND_SEPARATOR_TOKENS = frozenset(
    {"&&", "||", ";", "|", "&"}
)

_GIT_GLOBAL_OPTIONS_WITH_VALUE = frozenset(
    {
        "-c",
        "-C",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--exec-path",
        "--config-env",
        "--super-prefix",
    }
)

_GIT_GLOBAL_OPTION_PREFIXES_WITH_VALUE = (
    "--git-dir=",
    "--work-tree=",
    "--namespace=",
    "--exec-path=",
    "--config-env=",
    "--super-prefix=",
)

_ANSI_RED = "\x1b[31m"
_ANSI_GREEN = "\x1b[32m"
_ANSI_BRIGHT_BLACK = "\x1b[90m"
_ANSI_RESET = "\x1b[0m"


@dataclass(frozen=True)
class DisplayItem:
    """One complete human-readable interaction display block.

    ``text`` is the canonical, undecorated block contents.  Printing an item
    applies the Autopythia/Contradex left-hand quote gutter and, for diff
    blocks, the Contradex diff colorscheme.

    ``label`` optionally identifies the renderer-owned leading ``[label]``
    token in ``text``. It does not affect printing or equality; frontends can
    use it to add attribution without mistaking bracket-leading payloads for
    labels.
    """

    text: str
    is_diff: bool = field(default=False)
    label: Optional[str] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("display item text must be a string")
        if not self.text:
            raise ValueError("display item text must not be empty")
        if self.text.endswith(("\n", "\r")):
            raise ValueError(
                "display item text must not have a trailing newline"
            )
        if not isinstance(self.is_diff, bool):
            raise TypeError("display item is_diff must be a bool")
        if self.label is not None:
            if not isinstance(self.label, str):
                raise TypeError("display item label must be a string or None")
            if not self.label or not self.text.startswith(f"[{self.label}]"):
                raise ValueError("display item label must match its leading [label] token")

    def __str__(self) -> str:
        text = _colorize_diff_text(self.text) if self.is_diff else self.text
        return _quote_wrap_display_text(text)


@dataclass(frozen=True)
class InteractionItemRenderer:
    """Render completed interaction items using Contradex-style labels."""

    color: bool = True
    show_generic_arguments: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.color, bool):
            raise TypeError("color must be a bool")
        if not isinstance(self.show_generic_arguments, bool):
            raise TypeError("show_generic_arguments must be a bool")

    def render_items(
        self,
        items: Iterable[InteractionItem],
        *,
        source_calls: Iterable[ToolCall] = (),
        source_user_calls: Iterable[UserToolCall] = (),
    ) -> Tuple[DisplayItem, ...]:
        call_by_id = self._source_call_map(source_calls)
        user_call_by_id = self._source_call_map(i.call for i in source_user_calls)
        rendered: List[DisplayItem] = []

        for index, item in enumerate(tuple(items)):
            if not is_interaction_item(item):
                raise TypeError(
                    "items must contain only InteractionItem values; "
                    f"item {index} is {type(item).__name__}"
                )

            blocks: Tuple[str, ...]
            label: Optional[str] = None
            diff_block_indices: Set[int] = set()
            if isinstance(item, UserToolCall):
                label = "user-tool-call"
                user_call_by_id[item.call.call_id] = item.call
                blocks = self._render_tool_call(item.call, diff_block_indices, user=True)
            elif isinstance(item, UserToolResult):
                label = "user-tool-ret"
                blocks = self._render_tool_result(
                    item.result, user_call_by_id.get(item.result.call_id),
                    diff_block_indices, user=True,
                )
            elif isinstance(item, Instructions):
                label = "instructions"
                blocks = _render_instructions(item)
            elif isinstance(item, Message):
                label = item.role.strip() or "message"
                blocks = _render_message(item)
            elif isinstance(item, Reasoning):
                label = "reasoning"
                blocks = _render_reasoning(item)
            elif isinstance(item, ToolCall):
                label = "tool-call"
                call_by_id[item.call_id] = item
                blocks = self._render_tool_call(
                    item,
                    diff_block_indices,
                )
            elif isinstance(item, ToolResult):
                label = "tool-ret"
                blocks = self._render_tool_result(
                    item,
                    call_by_id.get(item.call_id),
                    diff_block_indices,
                )
            elif isinstance(item, SampleMetadata):
                label = "sample"
                blocks = _render_sample_metadata(item)
            elif isinstance(item, CompactionMetadata):
                label = "compaction"
                blocks = _render_compaction_metadata(item)
            elif isinstance(item, ModelFailure):
                label = "model failure"
                blocks = _render_model_failure(item)
            elif isinstance(item, TurnSummary):
                label = "turn"
                blocks = _render_turn_summary(item)
            elif isinstance(
                item,
                (ModelSampleBoundary, Init, UserInteractionBoundary),
            ):
                blocks = ()
            elif isinstance(item, OpaqueCompaction):
                label = "compaction"
                blocks = ("[compaction] opaque checkpoint",)
            elif isinstance(item, ContextPrefix):
                label = "context prefix"
                item_count = len(item.prefix_items)
                noun = "item" if item_count == 1 else "items"
                blocks = (
                    f"[context prefix] {item_count} {noun}",
                )
            else:
                raise TypeError(
                    f"unsupported interaction item: {type(item).__name__}"
                )

            for block_index, block in enumerate(blocks):
                normalized = _normalize_display_block(block)
                if normalized is not None:
                    rendered.append(
                        DisplayItem(
                            normalized,
                            is_diff=(
                                self.color
                                and block_index in diff_block_indices
                            ),
                            # Extra tool-call blocks are literal payloads, even
                            # when their text happens to start with [a label].
                            label=(None if block_index > 0 and isinstance(
                                item, (ToolCall, UserToolCall)
                            ) else label),
                        )
                    )

        return tuple(rendered)

    def _source_call_map(
        self,
        source_calls: Iterable[ToolCall],
    ) -> Dict[str, ToolCall]:
        calls: Dict[str, ToolCall] = {}
        for index, call in enumerate(tuple(source_calls)):
            if not isinstance(call, ToolCall):
                raise TypeError(
                    "source_calls must contain only ToolCall values; "
                    f"item {index} is {type(call).__name__}"
                )
            if call.call_id in calls:
                raise ValueError(
                    f"duplicate source tool call id: {call.call_id!r}"
                )
            calls[call.call_id] = call
        return calls

    def _render_tool_call(
        self,
        item: ToolCall,
        diff_block_indices: Set[int],
        *, user: bool = False,
    ) -> Tuple[str, ...]:
        label = (f"[user-tool-call] {item.name} ({item.call_id})" if user
                 else _format_tool_call_label(item.name, item.call_id))
        parsed, raw_arguments = _parse_tool_arguments(item.arguments_json)
        payload = parsed if isinstance(parsed, Mapping) else {}

        command = _extract_shell_command(item.name, payload)
        if command is not None:
            return (f"{label}\n{command}",)

        code_payload = _extract_code_payload(
            item.name,
            parsed,
            raw_arguments,
        )
        if code_payload is not None:
            if item.name == "apply_patch":
                diff_block_indices.add(1)
            return (label, code_payload)

        if item.name == "update_plan":
            return (label,)

        if self.show_generic_arguments or user:
            # Elide only empty user-call objects, not model debug arguments or
            # falsy non-object values that may be useful in diagnostics.
            if user and isinstance(parsed, Mapping) and not parsed:
                return (label,)
            generic_arguments = _format_generic_arguments(
                parsed,
                raw_arguments,
            )
            if generic_arguments is not None:
                return (label, generic_arguments)

        return (label,)

    def _render_tool_result(
        self,
        item: ToolResult,
        source_call: Optional[ToolCall],
        diff_block_indices: Set[int],
        *, user: bool = False,
    ) -> Tuple[str, ...]:
        name = source_call.name if source_call is not None else "tool"
        status = "ok" if item.success else "error"
        lines = [
            f"[user-tool-ret]  {name} ({item.call_id}) [{status}]" if user else _format_tool_result_label(
                name,
                item.call_id,
                status=status,
            )
        ]

        plan_lines = ()
        if item.success and name == "update_plan" and source_call is not None:
            parsed, _ = _parse_tool_arguments(source_call.arguments_json)
            if isinstance(parsed, Mapping):
                plan_lines = _render_update_plan_lines(parsed)
                lines.extend(plan_lines)

        output = _normalize_tool_output(item.output)
        if output is not None and not (
            plan_lines and output == "Plan updated"
        ):
            if (
                source_call is not None
                and name in _SHELL_TOOL_NAMES
            ):
                parsed, _ = _parse_tool_arguments(
                    source_call.arguments_json
                )
                payload = parsed if isinstance(parsed, Mapping) else {}
                command = _extract_shell_command(name, payload)
                if (
                    command is not None
                    and _looks_like_git_diff_command(command)
                ):
                    diff_block_indices.add(0)
            lines.append(output)

        return ("\n".join(lines),)


def render_interaction_items(
    items: Iterable[InteractionItem],
    *,
    source_calls: Iterable[ToolCall] = (),
    source_user_calls: Iterable[UserToolCall] = (),
    color: bool = True,
    show_generic_arguments: bool = False,
) -> Tuple[DisplayItem, ...]:
    return InteractionItemRenderer(
        color=color,
        show_generic_arguments=show_generic_arguments,
    ).render_items(
        items,
        source_calls=source_calls,
        source_user_calls=source_user_calls,
    )


def _format_byte_count(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size} B"


def _describe_non_text_source(source_uri: str) -> str:
    """A compact, payload-free label for a ``MediaPart`` source URI."""
    if not source_uri.startswith("data:"):
        return source_uri
    header, _, encoded = source_uri.partition(",")
    media_type = header[len("data:"):].split(";", 1)[0]
    if not media_type:
        media_type = "application/octet-stream"
    padding = encoded[-2:].count("=")
    size = max(0, len(encoded) * 3 // 4 - padding)
    return f"{media_type} {_format_byte_count(size)}"


def _render_message(item: Message) -> Tuple[str, ...]:
    role = item.role.strip() or "message"
    blocks = []
    text = item.content_text
    if text.strip():
        blocks.append(f"[{role}] {text}")
    for part in item.parts:
        if isinstance(part, TextPart):
            continue
        blocks.append(
            f"[{role}] [image] {_describe_non_text_source(part.source_uri)}"
        )
    return tuple(blocks)


def _render_instructions(item: Instructions) -> Tuple[str, ...]:
    # Unlike Message, empty/whitespace-only instructions are still shown
    # so the audit trail preserves presence vs absence.
    if not item.text.strip():
        return ("[instructions]",)
    return (f"[instructions] {item.text}",)


def _render_sample_metadata(item: SampleMetadata) -> Tuple[str, ...]:
    usage = item.usage
    elapsed = (
        "" if item.elapsed_seconds is None else f" elapsed={item.elapsed_seconds:.2f}s"
    )
    attempts = (
        "" if item.request_attempts == 1 else f" attempts={item.request_attempts}"
    )
    recovery = (
        "" if not item.recovery else f" recovery={','.join(item.recovery)}"
    )
    return (
        "[sample] "
        f"input={usage.input_tokens} "
        f"output={usage.output_tokens} "
        f"total={usage.total_tokens} "
        f"cached={usage.cached_input_tokens}{elapsed}{attempts}{recovery}",
    )


def _render_compaction_metadata(
    item: CompactionMetadata,
) -> Tuple[str, ...]:
    usage = item.usage
    fields = [
        "[compaction]",
        f"protocol={item.protocol}",
        f"input={usage.input_tokens}",
        f"output={usage.output_tokens}",
        f"total={usage.total_tokens}",
        f"cached={usage.cached_input_tokens}",
    ]
    if item.elapsed_seconds is not None:
        fields.append(f"elapsed={item.elapsed_seconds:.2f}s")
    if item.request_attempts != 1:
        fields.append(f"attempts={item.request_attempts}")
    if item.recovery:
        fields.append(f"recovery={','.join(item.recovery)}")
    return (" ".join(fields),)


def _render_model_failure(item: ModelFailure) -> Tuple[str, ...]:
    fields = [f"[model failure] {item.message}", f"kind={item.category}"]
    for name, value in (
        ("provider", item.provider),
        ("model", item.model),
        ("auth_source", item.auth_source),
        ("status", item.http_status),
        ("attempts", item.attempt_count),
        ("request_id", item.request_id),
        ("response_id", item.response_id),
        ("cf_ray", item.cf_ray),
        ("auth_error", item.authorization_error),
        ("auth_code", item.auth_error_code),
        ("error_code", item.error_code),
        ("events", item.event_count or None),
        ("completed_items", item.completed_item_count or None),
        ("last_event", item.last_event_type),
        ("last_sequence", item.last_sequence_number),
        (
            "elapsed",
            None if item.elapsed_seconds is None else f"{item.elapsed_seconds:.2f}s",
        ),
    ):
        if value is not None:
            fields.append(f"{name}={value}")
    if item.recovery:
        fields.append(f"recovery={','.join(item.recovery)}")
    if item.event_types:
        fields.append(f"event_types={','.join(item.event_types)}")
    return (" ".join(fields),)


def _render_turn_summary(item: TurnSummary) -> Tuple[str, ...]:
    # Contradex-style end-of-turn aggregate: warm = cached, cold = non-cached.
    # Keep field-order stable for tests: input/output sums, cold sum,
    # warm sum/max, context window, sample/compaction counts.
    return (
        "[turn] "
        f"input_sum={item.input_tokens_sum} "
        f"output_sum={item.output_tokens_sum} "
        f"cold_sum={item.non_cached_input_tokens_sum} "
        f"cached_sum={item.cached_input_tokens_sum} "
        f"cached_max={item.cached_input_tokens_max} "
        f"context={item.context_tokens} "
        f"samples={item.sample_count} "
        f"compactions={item.compaction_count}"
        + (
            ""
            if item.elapsed_seconds is None
            else f" elapsed={item.elapsed_seconds:.2f}s"
        ),
    )


def _render_reasoning(item: Reasoning) -> Tuple[str, ...]:
    summaries = tuple(
        value.strip()
        for value in item.summary
        if value.strip()
    )
    if summaries:
        return tuple(f"[reasoning] {value}" for value in summaries)
    fallback = item.content.strip()
    if fallback:
        return (f"[reasoning] {fallback}",)
    # Providers can return a valid reasoning item whose only payload is opaque:
    # a Responses item containing only ``encrypted_content``, or a Messages
    # thinking block with empty ``thinking`` text and only a ``signature``
    # (``content_signature``).  That data is needed for provider replay but is
    # not human-readable and must not be printed.  Still render a redacted
    # marker so the transcript does not silently lose the item's position.
    if item.encrypted_content is not None or item.content_signature is not None:
        return ("[reasoning] ...",)
    return ()


def _parse_tool_arguments(
    arguments_json: str,
) -> Tuple[Optional[object], str]:
    raw = arguments_json
    stripped = raw.strip()
    if not stripped:
        return None, raw
    try:
        return json.loads(stripped), raw
    except json.JSONDecodeError:
        return None, raw


def _stringify_command(value: object) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, list):
        normalized = [
            str(part).strip()
            for part in value
            if str(part).strip()
        ]
        if normalized:
            return " ".join(normalized)
    return None


def _extract_shell_command(
    name: str,
    payload: Mapping[str, object],
) -> Optional[str]:
    if name not in _SHELL_TOOL_NAMES:
        return None
    if name == "write_stdin":
        return _format_write_stdin_summary(payload)
    command = payload.get("cmd")
    if command is None:
        command = payload.get("command")
    return _stringify_command(command)


def _format_write_stdin_summary(
    payload: Mapping[str, object],
) -> Optional[str]:
    parts = []
    if "session_id" in payload:
        parts.append(f"session_id={payload['session_id']}")
    if "chars" in payload:
        chars = str(payload.get("chars") or "")
        parts.append(f"chars={len(chars.encode('utf-8'))} bytes")
    return " ".join(parts) if parts else None


def _extract_code_payload(
    name: str,
    parsed: Optional[object],
    raw_arguments: str,
) -> Optional[str]:
    if name not in _CODE_EDIT_TOOL_NAMES:
        return None

    if isinstance(parsed, Mapping):
        for key in (
            "patch",
            "input",
            "diff",
            "raw",
            "value",
            "content",
            "text",
            "code",
        ):
            payload = _normalize_tool_payload(parsed.get(key))
            if payload is not None:
                return payload
        return None

    if isinstance(parsed, str):
        return _normalize_tool_payload(parsed)
    if parsed is None:
        return _normalize_tool_payload(raw_arguments)
    return None


def _format_generic_arguments(
    parsed: Optional[object],
    raw_arguments: str,
) -> Optional[str]:
    if parsed is not None:
        return json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return _normalize_tool_payload(raw_arguments)


def _format_tool_call_label(name: str, call_id: str) -> str:
    return f"[tool-call] {name} ({call_id})"


def _format_tool_result_label(
    name: str,
    call_id: str,
    *,
    status: str,
) -> str:
    return f"[tool-ret]  {name} ({call_id}) [{status}]"


def _render_update_plan_lines(
    payload: Mapping[str, object],
) -> Tuple[str, ...]:
    raw_plan = payload.get("plan")
    if not isinstance(raw_plan, list):
        return ()

    lines = ["[plan] Updated plan"]
    explanation = payload.get("explanation")
    if isinstance(explanation, str):
        normalized_explanation = explanation.strip()
        if normalized_explanation:
            lines.append(f"[plan] note: {normalized_explanation}")

    rendered_steps = 0
    for item in raw_plan:
        if not isinstance(item, Mapping):
            continue
        step = str(item.get("step") or "").strip()
        if not step:
            continue
        status = str(item.get("status") or "").strip()
        lines.append(f"[plan] {_format_plan_status(status)} {step}")
        rendered_steps += 1

    if rendered_steps == 0:
        lines.append("[plan] (no steps provided)")
    return tuple(lines)


def _format_plan_status(status: str) -> str:
    if status == "completed":
        return "[x]"
    if status == "in_progress":
        return "[>]"
    if status == "pending":
        return "[ ]"
    return f"[{status or '?'}]"


def _normalize_tool_payload(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.rstrip("\r\n")
    if not normalized.strip():
        return None
    return normalized


def _normalize_tool_output(value: object) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    normalized = text.rstrip("\r\n")
    if not normalized.strip():
        return None
    return normalized


def _normalize_display_block(value: object) -> Optional[str]:
    if not isinstance(value, str):
        raise TypeError("display blocks must be strings")
    normalized = value.rstrip("\r\n")
    if not normalized.strip():
        return None
    return normalized


def _colorize_diff_text(payload: str) -> str:
    lines = payload.split("\n")
    metadata_header_indices = _collect_diff_metadata_header_indices(lines)
    return "\n".join(
        _colorize_diff_line(
            line,
            is_metadata_header=(index in metadata_header_indices),
        )
        for index, line in enumerate(lines)
    )


def _bright_black(text: str) -> str:
    return f"{_ANSI_BRIGHT_BLACK}{text}{_ANSI_RESET}"


def _quote_wrap_display_text(text: str) -> str:
    """Apply the Autopythia display gutter to a completed block."""
    lines = text.split("\n")
    if len(lines) == 1:
        return f"   {_bright_black('[')}{lines[0]}"

    parts = [f"   {_bright_black('⌜')}{lines[0]}"]
    parts.extend(f"    {line}" if line else "" for line in lines[1:-1])
    parts.append(f"   {_bright_black('⌞')}{lines[-1]}")
    return "\n".join(parts)


def _colorize_diff_line(
    line: str,
    *,
    is_metadata_header: bool = False,
) -> str:
    if not line or is_metadata_header:
        return line
    if line.startswith("+"):
        return f"{_ANSI_GREEN}{line}{_ANSI_RESET}"
    if line.startswith("-"):
        return f"{_ANSI_RED}{line}{_ANSI_RESET}"
    return line


def _collect_diff_metadata_header_indices(lines: List[str]) -> set:
    metadata_indices = set()
    pending_old_indices = []
    in_hunk = False

    for index, line in enumerate(lines):
        if in_hunk:
            if _is_unified_diff_hunk_line(line):
                continue
            in_hunk = False

        if _is_unified_diff_hunk_header(line):
            in_hunk = True
            pending_old_indices.clear()
            continue

        if _is_unified_old_file_header_line(line):
            pending_old_indices.append(index)
            continue

        if _is_unified_new_file_header_line(line):
            if pending_old_indices:
                metadata_indices.update(pending_old_indices)
                metadata_indices.add(index)
            pending_old_indices.clear()
            continue

        pending_old_indices.clear()

    return metadata_indices


def _is_unified_diff_hunk_header(line: str) -> bool:
    return line.startswith("@@")


def _is_unified_diff_hunk_line(line: str) -> bool:
    if not line:
        return False
    return line.startswith((" ", "+", "-", "\\"))


def _is_unified_old_file_header_line(line: str) -> bool:
    return _is_unified_file_header_line(line, prefix="---")


def _is_unified_new_file_header_line(line: str) -> bool:
    return _is_unified_file_header_line(line, prefix="+++")


def _is_unified_file_header_line(line: str, *, prefix: str) -> bool:
    if not line.startswith(prefix):
        return False
    if len(line) == len(prefix):
        return False
    return line[len(prefix)] in {" ", "\t"}


def _looks_like_git_diff_command(command: str) -> bool:
    stripped = command.strip()
    if not stripped:
        return False

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()

    if not tokens:
        return False

    segment = []
    for token in tokens:
        if token in _SHELL_COMMAND_SEPARATOR_TOKENS:
            if _segment_invokes_git_diff(segment):
                return True
            segment = []
            continue
        segment.append(token)

    return _segment_invokes_git_diff(segment)


def _segment_invokes_git_diff(tokens: List[str]) -> bool:
    if not tokens:
        return False

    index = 0
    while index < len(tokens) and _looks_like_env_assignment(tokens[index]):
        index += 1

    if index < len(tokens) and tokens[index] == "env":
        index += 1
        while index < len(tokens):
            token = tokens[index]
            if token == "-u" and index + 1 < len(tokens):
                index += 2
                continue
            if token.startswith("-") or _looks_like_env_assignment(token):
                index += 1
                continue
            break

    if index >= len(tokens) or tokens[index] != "git":
        return False

    index += 1
    while index < len(tokens):
        token = tokens[index]
        if token == "diff":
            return True
        if token in _GIT_GLOBAL_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if token.startswith(_GIT_GLOBAL_OPTION_PREFIXES_WITH_VALUE):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return False

    return False


def _looks_like_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    name, _, _ = token.partition("=")
    if not name or not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(char.isalnum() or char == "_" for char in name)


__all__ = [
    "DisplayItem",
    "InteractionItemRenderer",
    "render_interaction_items",
]
