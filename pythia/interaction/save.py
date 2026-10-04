from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

from .context import InteractionContext
from .items import CompactionMetadata
from .items import ContentPart
from .items import ContextPrefix
from .items import Init
from .items import Instructions
from .items import InteractionItem
from .items import MediaPart
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import ToolResult
from .items import Tools
from .items import SampleMetadata
from .items import TextPart
from .items import TurnSummary
from .items import UserInteractionBoundary
from .items import UserToolCall
from .items import UserToolResult
from .items import _validate_elapsed_seconds
from .usage import TokenUsage
from ._tool_spec import ToolSpec
from ._tool_spec import _copy_schema


SavePath = Union[str, os.PathLike]


class SaveError(ValueError):
    pass


_ITEM_TYPES = {
    Init: "init",
    Instructions: "instructions",
    Tools: "tools",
    Message: "message",
    Reasoning: "reasoning",
    ToolCall: "tool_call",
    ToolResult: "tool_result",
    UserToolCall: "user_tool_call",
    UserToolResult: "user_tool_result",
    ModelSampleBoundary: "model_sample_boundary",
    SampleMetadata: "sample_metadata",
    CompactionMetadata: "compaction_metadata",
    ModelFailure: "model_failure",
    TurnSummary: "turn_summary",
    UserInteractionBoundary: "user_interaction_boundary",
    OpaqueCompaction: "opaque_compaction",
    ContextPrefix: "context_prefix",
}
# Accept legacy records on read; new saves use only the canonical types above.
_ITEM_TYPE_NAMES = frozenset(_ITEM_TYPES.values()) | {
    "context_compaction",
    "session_init",
    "turn_metadata",
}


def _item_type_name(item: InteractionItem) -> str:
    item_type = type(item)
    if item_type not in _ITEM_TYPES:
        raise SaveError(
            f"cannot encode interaction item type {item_type.__name__}"
        )
    return _ITEM_TYPES[item_type]


def _content_part_to_dict(part: ContentPart) -> Dict[str, Any]:
    """Encode one in-memory content part using its own (non-wire) shape.

    The durable record mirrors the in-memory ``TextPart``/``MediaPart`` types;
    the Responses ``input_text``/``input_image`` mapping happens only when a
    request is assembled (see ``responses._encode_context_items``).
    """
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, MediaPart):
        return {"type": "media", "source_uri": part.source_uri}
    raise SaveError(f"cannot encode content part type {type(part).__name__}")


def interaction_item_to_dict(item: InteractionItem) -> Dict[str, Any]:
    """Encode an interaction item as a JSON-compatible dictionary."""
    encoded: Dict[str, Any] = {"type": _item_type_name(item)}

    if isinstance(item, UserToolCall):
        encoded["call"] = interaction_item_to_dict(item.call)
    elif isinstance(item, UserToolResult):
        encoded["result"] = interaction_item_to_dict(item.result)
    elif isinstance(item, Instructions):
        encoded.update(text=item.text)
    elif isinstance(item, Tools):
        encoded["specs"] = [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": _copy_schema(spec.parameters),
            }
            for spec in item.specs
        ]
    elif isinstance(item, Message):
        if isinstance(item.content, str):
            encoded.update(role=item.role, content=item.content)
        else:
            encoded.update(
                role=item.role,
                content=[_content_part_to_dict(part) for part in item.content],
            )
    elif isinstance(item, Init):
        encoded["prefix_id"] = item.prefix_id
        if item.model is not None:
            encoded["model"] = item.model
    elif isinstance(item, Reasoning):
        encoded.update(content=item.content, summary=list(item.summary))
        if item.encrypted_content is not None:
            encoded["encrypted_content"] = item.encrypted_content
        if item.content_signature is not None:
            encoded["content_signature"] = item.content_signature
    elif isinstance(item, ToolCall):
        encoded.update(
            name=item.name,
            call_id=item.call_id,
            arguments_json=item.arguments_json,
        )
    elif isinstance(item, ToolResult):
        encoded.update(
            call_id=item.call_id,
            output=item.output,
            success=item.success,
        )
    elif isinstance(item, OpaqueCompaction):
        encoded.update(payload=item.payload, protocol=item.protocol)
    elif isinstance(item, ContextPrefix):
        encoded["prefix_items"] = [
            interaction_item_to_dict(nested)
            for nested in item.prefix_items
        ]
    elif isinstance(item, (SampleMetadata, CompactionMetadata)):
        encoded["usage"] = {
            "input_tokens": item.usage.input_tokens,
            "output_tokens": item.usage.output_tokens,
            "total_tokens": item.usage.total_tokens,
            "cached_input_tokens": item.usage.cached_input_tokens,
        }
        if isinstance(item, CompactionMetadata):
            encoded["protocol"] = item.protocol
        provider_fields = (
            "provider_session_id",
            "provider_turn_id",
            "provider_turn_state",
        )
        if isinstance(item, CompactionMetadata):
            provider_fields = (*provider_fields, "provider_response_id")
        for field_name in provider_fields:
            value = getattr(item, field_name)
            if value is not None:
                encoded[field_name] = value
        if item.elapsed_seconds is not None:
            encoded["elapsed_seconds"] = item.elapsed_seconds
        if item.request_attempts != 1:
            encoded["request_attempts"] = item.request_attempts
        if item.recovery:
            encoded["recovery"] = list(item.recovery)
    elif isinstance(item, ModelFailure):
        for field_name in (
            "category",
            "message",
            "provider",
            "model",
            "auth_source",
            "http_status",
            "request_id",
            "response_id",
            "cf_ray",
            "authorization_error",
            "auth_error_code",
            "error_code",
            "attempt_count",
            "event_count",
            "event_types",
            "completed_item_count",
            "last_event_type",
            "last_sequence_number",
            "recovery",
            "elapsed_seconds",
        ):
            value = getattr(item, field_name)
            if field_name in {"event_types", "recovery"}:
                if value:
                    encoded[field_name] = list(value)
            elif value is not None:
                encoded[field_name] = value
    elif isinstance(item, TurnSummary):
        encoded.update(
            input_tokens_sum=item.input_tokens_sum,
            output_tokens_sum=item.output_tokens_sum,
            cached_input_tokens_sum=item.cached_input_tokens_sum,
            cached_input_tokens_max=item.cached_input_tokens_max,
            non_cached_input_tokens_sum=item.non_cached_input_tokens_sum,
            context_tokens=item.context_tokens,
            sample_count=item.sample_count,
            compaction_count=item.compaction_count,
        )
        if item.elapsed_seconds is not None:
            encoded["elapsed_seconds"] = item.elapsed_seconds
    elif isinstance(
        item,
        (ModelSampleBoundary, UserInteractionBoundary),
    ):
        pass
    else:
        raise SaveError(
            f"cannot encode interaction item type {type(item).__name__}"
        )

    return encoded


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SaveError("interaction item must be an object")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SaveError(f"{field_name} must be a string")
    return value


def _require_content(mapping: Mapping[str, Any], item_type: str) -> str:
    """Read content or legacy text, rejecting invalid or conflicting fields."""
    if "content" not in mapping and "text" in mapping:
        return _require_string(mapping["text"], f"{item_type}.text")

    content = _require_string(mapping.get("content"), f"{item_type}.content")
    if "text" in mapping:
        legacy_text = _require_string(mapping["text"], f"{item_type}.text")
        if content != legacy_text:
            raise SaveError(
                f"{item_type}.content and {item_type}.text must match"
            )
    return content


def _require_init_prefix_id(mapping: Mapping[str, Any]) -> str:
    """Read the canonical prefix ID or its legacy session ID spelling."""
    if "prefix_id" not in mapping and "session_id" in mapping:
        return _require_string(mapping["session_id"], "init.session_id")

    prefix_id = _require_string(mapping.get("prefix_id"), "init.prefix_id")
    if "session_id" in mapping:
        session_id = _require_string(mapping["session_id"], "init.session_id")
        if prefix_id != session_id:
            raise SaveError("init.prefix_id and init.session_id must match")
    return prefix_id


def _message_content_parts(value: Any) -> Tuple[ContentPart, ...]:
    """Decode a durable content array into in-memory content parts."""
    parts = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise SaveError(f"message.content[{index}] must be an object")
        part_type = entry.get("type")
        if part_type == "text":
            parts.append(
                TextPart(
                    text=_require_string(
                        entry.get("text"),
                        f"message.content[{index}].text",
                    )
                )
            )
        elif part_type == "media":
            source_uri = entry.get("source_uri")
            if not isinstance(source_uri, str) or not source_uri:
                raise SaveError(
                    f"message.content[{index}].source_uri must be a "
                    "non-empty string"
                )
            parts.append(MediaPart(source_uri=source_uri))
        else:
            raise SaveError(
                f"unsupported message.content[{index}] type: {part_type!r}"
            )
    if not parts:
        raise SaveError("message.content list must not be empty")
    return tuple(parts)


def _optional_string(
    mapping: Mapping[str, Any],
    key: str,
    field_name: str,
) -> Optional[str]:
    value = mapping.get(key)
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SaveError(f"{field_name} must be a nonnegative integer")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SaveError(f"{field_name} must be a positive integer")
    return value


def _optional_http_status(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 599:
        raise SaveError("model_failure.http_status must be from 100 through 599")
    return value


def _optional_nonnegative_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _require_nonnegative_int(value, field_name)


def interaction_item_from_dict(value: Any) -> InteractionItem:
    """Decode a dictionary produced by ``interaction_item_to_dict``."""
    mapping = _require_mapping(value)
    item_type = mapping.get("type")
    if not isinstance(item_type, str) or item_type not in _ITEM_TYPE_NAMES:
        raise SaveError(f"unknown interaction item type: {item_type!r}")

    if item_type == "user_tool_call":
        return UserToolCall(interaction_item_from_dict(mapping.get("call")))
    if item_type == "user_tool_result":
        return UserToolResult(interaction_item_from_dict(mapping.get("result")))
    if item_type == "instructions":
        return Instructions(
            text=_require_string(mapping.get("text"), "instructions.text"),
        )
    if item_type == "tools":
        specs = mapping.get("specs")
        if not isinstance(specs, list):
            raise SaveError("tools.specs must be a list")
        decoded = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, Mapping):
                raise SaveError(f"tools.specs[{index}] must be an object")
            try:
                decoded.append(ToolSpec(
                    name=spec.get("name"),
                    description=spec.get("description"),
                    parameters=spec.get("parameters"),
                ))
            except (TypeError, ValueError) as exc:
                raise SaveError(f"tools.specs[{index}]: {exc}") from exc
        try:
            return Tools(tuple(decoded))
        except (TypeError, ValueError) as exc:
            raise SaveError(f"tools.specs: {exc}") from exc
    if item_type == "message":
        role = _require_string(mapping.get("role"), "message.role")
        raw_content = mapping.get("content")
        if isinstance(raw_content, list):
            return Message(
                role=role,
                content=_message_content_parts(raw_content),
            )
        return Message(
            role=role,
            content=_require_content(mapping, "message"),
        )
    if item_type in {"init", "session_init"}:
        return Init(
            prefix_id=_require_init_prefix_id(mapping),
            model=_optional_string(mapping, "model", "init.model"),
        )
    if item_type == "reasoning":
        summary_value = mapping.get("summary", ())
        if not isinstance(summary_value, list):
            raise SaveError("reasoning.summary must be a list")
        return Reasoning(
            content=_require_content(mapping, "reasoning"),
            summary=[
                _require_string(value, f"reasoning.summary[{index}]")
                for index, value in enumerate(summary_value)
            ],
            encrypted_content=_optional_string(
                mapping,
                "encrypted_content",
                "reasoning.encrypted_content",
            ),
            content_signature=_optional_string(
                mapping,
                "content_signature",
                "reasoning.content_signature",
            ),
        )
    if item_type == "tool_call":
        return ToolCall(
            name=_require_string(mapping.get("name"), "tool_call.name"),
            call_id=_require_string(
                mapping.get("call_id"),
                "tool_call.call_id",
            ),
            arguments_json=_require_string(
                mapping.get("arguments_json"),
                "tool_call.arguments_json",
            ),
        )
    if item_type == "tool_result":
        success = mapping.get("success", True)
        if not isinstance(success, bool):
            raise SaveError("tool_result.success must be a boolean")
        return ToolResult(
            call_id=_require_string(
                mapping.get("call_id"),
                "tool_result.call_id",
            ),
            output=_require_string(
                mapping.get("output"),
                "tool_result.output",
            ),
            success=success,
        )
    if item_type == "opaque_compaction":
        # Existing save files stored only Responses encrypted content.
        if "protocol" not in mapping and "payload" not in mapping:
            return OpaqueCompaction.from_responses(
                _require_string(
                    mapping.get("encrypted_content"),
                    "opaque_compaction.encrypted_content",
                )
            )
        protocol = _require_string(
            mapping.get("protocol"),
            "opaque_compaction.protocol",
        )
        return OpaqueCompaction(
            payload=_require_string(
                mapping.get("payload"),
                "opaque_compaction.payload",
            ),
            protocol=protocol,
        )
    if item_type in {"context_prefix", "context_compaction"}:
        key = (
            "prefix_items"
            if "prefix_items" in mapping or "replacement_items" not in mapping
            else "replacement_items"
        )
        prefix = mapping.get(key)
        if not isinstance(prefix, list):
            raise SaveError(
                f"{item_type}.{key} must be a list"
            )
        prefix_items = tuple(
            interaction_item_from_dict(item) for item in prefix
        )
        if key == "prefix_items" and "replacement_items" in mapping:
            replacement = mapping["replacement_items"]
            if not isinstance(replacement, list):
                raise SaveError(
                    f"{item_type}.replacement_items must be a list"
                )
            replacement_items = tuple(
                interaction_item_from_dict(item) for item in replacement
            )
            if prefix_items != replacement_items:
                raise SaveError(
                    f"{item_type}.prefix_items and "
                    f"{item_type}.replacement_items must match"
                )
        return ContextPrefix(prefix_items)
    if item_type == "compaction_metadata":
        usage = mapping.get("usage")
        if not isinstance(usage, Mapping):
            raise SaveError("compaction_metadata.usage must be an object")
        try:
            elapsed = _validate_elapsed_seconds(mapping.get("elapsed_seconds"))
        except (TypeError, ValueError) as exc:
            raise SaveError(f"compaction_metadata.{exc}") from exc
        recovery = mapping.get("recovery", [])
        if not isinstance(recovery, list):
            raise SaveError("compaction_metadata.recovery must be a list")
        try:
            return CompactionMetadata(
                usage=TokenUsage(
                    input_tokens=_require_nonnegative_int(
                        usage.get("input_tokens"),
                        "compaction_metadata.usage.input_tokens",
                    ),
                    output_tokens=_require_nonnegative_int(
                        usage.get("output_tokens"),
                        "compaction_metadata.usage.output_tokens",
                    ),
                    total_tokens=_require_nonnegative_int(
                        usage.get("total_tokens"),
                        "compaction_metadata.usage.total_tokens",
                    ),
                    cached_input_tokens=_require_nonnegative_int(
                        usage.get("cached_input_tokens"),
                        "compaction_metadata.usage.cached_input_tokens",
                    ),
                ),
                protocol=_require_string(
                    mapping.get("protocol"),
                    "compaction_metadata.protocol",
                ),
                provider_session_id=_optional_string(
                    mapping,
                    "provider_session_id",
                    "compaction_metadata.provider_session_id",
                ),
                provider_turn_id=_optional_string(
                    mapping,
                    "provider_turn_id",
                    "compaction_metadata.provider_turn_id",
                ),
                provider_turn_state=_optional_string(
                    mapping,
                    "provider_turn_state",
                    "compaction_metadata.provider_turn_state",
                ),
                provider_response_id=_optional_string(
                    mapping,
                    "provider_response_id",
                    "compaction_metadata.provider_response_id",
                ),
                elapsed_seconds=elapsed,
                request_attempts=_require_positive_int(
                    mapping.get("request_attempts", 1),
                    "compaction_metadata.request_attempts",
                ),
                recovery=tuple(
                    _require_string(
                        value,
                        f"compaction_metadata.recovery[{index}]",
                    )
                    for index, value in enumerate(recovery)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise SaveError(f"compaction_metadata.{exc}") from exc
    if item_type in {"sample_metadata", "turn_metadata"}:
        usage = mapping.get("usage")
        if not isinstance(usage, Mapping):
            raise SaveError(f"{item_type}.usage must be an object")
        try:
            elapsed = _validate_elapsed_seconds(mapping.get("elapsed_seconds"))
        except (TypeError, ValueError) as exc:
            raise SaveError(f"{item_type}.{exc}") from exc
        recovery = mapping.get("recovery", [])
        if not isinstance(recovery, list):
            raise SaveError(f"{item_type}.recovery must be a list")
        return SampleMetadata(
            usage=TokenUsage(
                input_tokens=_require_nonnegative_int(
                    usage.get("input_tokens"),
                    f"{item_type}.usage.input_tokens",
                ),
                output_tokens=_require_nonnegative_int(
                    usage.get("output_tokens"),
                    f"{item_type}.usage.output_tokens",
                ),
                total_tokens=_require_nonnegative_int(
                    usage.get("total_tokens"),
                    f"{item_type}.usage.total_tokens",
                ),
                cached_input_tokens=_require_nonnegative_int(
                    usage.get("cached_input_tokens"),
                    f"{item_type}.usage.cached_input_tokens",
                ),
            ),
            provider_session_id=_optional_string(
                mapping,
                "provider_session_id",
                f"{item_type}.provider_session_id",
            ),
            provider_turn_id=_optional_string(
                mapping,
                "provider_turn_id",
                f"{item_type}.provider_turn_id",
            ),
            provider_turn_state=_optional_string(
                mapping,
                "provider_turn_state",
                f"{item_type}.provider_turn_state",
            ),
            elapsed_seconds=elapsed,
            request_attempts=_require_positive_int(
                mapping.get("request_attempts", 1),
                f"{item_type}.request_attempts",
            ),
            recovery=tuple(
                _require_string(value, f"{item_type}.recovery[{index}]")
                for index, value in enumerate(recovery)
            ),
        )
    if item_type == "model_failure":
        recovery = mapping.get("recovery", [])
        if not isinstance(recovery, list):
            raise SaveError("model_failure.recovery must be a list")
        event_types = mapping.get("event_types", [])
        if not isinstance(event_types, list):
            raise SaveError("model_failure.event_types must be a list")
        try:
            elapsed = _validate_elapsed_seconds(mapping.get("elapsed_seconds"))
        except (TypeError, ValueError) as exc:
            raise SaveError(f"model_failure.{exc}") from exc
        try:
            return ModelFailure(
                category=_require_string(
                    mapping.get("category"), "model_failure.category",
                ),
                message=_require_string(
                    mapping.get("message"), "model_failure.message",
                ),
                provider=_optional_string(
                    mapping, "provider", "model_failure.provider",
                ),
                model=_optional_string(
                    mapping, "model", "model_failure.model",
                ),
                auth_source=_optional_string(
                    mapping,
                    "auth_source",
                    "model_failure.auth_source",
                ),
                http_status=_optional_http_status(mapping.get("http_status")),
                request_id=_optional_string(
                    mapping, "request_id", "model_failure.request_id",
                ),
                response_id=_optional_string(
                    mapping, "response_id", "model_failure.response_id",
                ),
                cf_ray=_optional_string(
                    mapping, "cf_ray", "model_failure.cf_ray",
                ),
                authorization_error=_optional_string(
                    mapping,
                    "authorization_error",
                    "model_failure.authorization_error",
                ),
                auth_error_code=_optional_string(
                    mapping,
                    "auth_error_code",
                    "model_failure.auth_error_code",
                ),
                error_code=_optional_string(
                    mapping,
                    "error_code",
                    "model_failure.error_code",
                ),
                attempt_count=_require_positive_int(
                    mapping.get("attempt_count", 1),
                    "model_failure.attempt_count",
                ),
                event_count=_require_nonnegative_int(
                    mapping.get("event_count", 0),
                    "model_failure.event_count",
                ),
                event_types=tuple(
                    _require_string(
                        value,
                        f"model_failure.event_types[{index}]",
                    )
                    for index, value in enumerate(event_types)
                ),
                completed_item_count=_require_nonnegative_int(
                    mapping.get("completed_item_count", 0),
                    "model_failure.completed_item_count",
                ),
                last_event_type=_optional_string(
                    mapping,
                    "last_event_type",
                    "model_failure.last_event_type",
                ),
                last_sequence_number=_optional_nonnegative_int(
                    mapping.get("last_sequence_number"),
                    "model_failure.last_sequence_number",
                ),
                recovery=tuple(
                    _require_string(
                        value,
                        f"model_failure.recovery[{index}]",
                    )
                    for index, value in enumerate(recovery)
                ),
                elapsed_seconds=elapsed,
            )
        except (TypeError, ValueError) as exc:
            raise SaveError(f"model_failure.{exc}") from exc
    if item_type == "turn_summary":
        try:
            elapsed = _validate_elapsed_seconds(mapping.get("elapsed_seconds"))
            return TurnSummary(
                input_tokens_sum=_require_nonnegative_int(
                    mapping.get("input_tokens_sum"),
                    "turn_summary.input_tokens_sum",
                ),
                output_tokens_sum=_require_nonnegative_int(
                    mapping.get("output_tokens_sum"),
                    "turn_summary.output_tokens_sum",
                ),
                cached_input_tokens_sum=_require_nonnegative_int(
                    mapping.get("cached_input_tokens_sum"),
                    "turn_summary.cached_input_tokens_sum",
                ),
                cached_input_tokens_max=_require_nonnegative_int(
                    mapping.get("cached_input_tokens_max"),
                    "turn_summary.cached_input_tokens_max",
                ),
                non_cached_input_tokens_sum=_require_nonnegative_int(
                    mapping.get("non_cached_input_tokens_sum"),
                    "turn_summary.non_cached_input_tokens_sum",
                ),
                context_tokens=_require_nonnegative_int(
                    mapping.get("context_tokens"),
                    "turn_summary.context_tokens",
                ),
                sample_count=_require_nonnegative_int(
                    mapping.get("sample_count"),
                    "turn_summary.sample_count",
                ),
                compaction_count=_require_nonnegative_int(
                    mapping.get("compaction_count"),
                    "turn_summary.compaction_count",
                ),
                elapsed_seconds=elapsed,
            )
        except (TypeError, ValueError) as exc:
            raise SaveError(f"turn_summary.{exc}") from exc
    if item_type == "model_sample_boundary":
        return ModelSampleBoundary()
    if item_type == "user_interaction_boundary":
        return UserInteractionBoundary()

    raise SaveError(f"unknown interaction item type: {item_type!r}")


def iter_interaction_items(
    items: Iterable[InteractionItem],
) -> Iterator[Dict[str, Any]]:
    for item in items:
        yield interaction_item_to_dict(item)


def save_interaction_save(path: SavePath, context: InteractionContext) -> None:
    """Atomically write a context as one JSON interaction item per line."""
    destination = Path(path)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            for encoded in iter_interaction_items(context.items):
                temporary.write(
                    json.dumps(
                        encoded,
                        ensure_ascii=False,
                    )
                )
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except OSError as exc:
        raise SaveError(
            f"could not write save to {destination}: {exc}"
        ) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def load_interaction_save(path: SavePath) -> InteractionContext:
    """Load and validate a JSONL interaction-save file."""
    source = Path(path)
    items: list[InteractionItem] = []
    try:
        with source.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SaveError(
                        f"invalid JSON in save {source} at line "
                        f"{line_number}: {exc}"
                    ) from exc

                try:
                    items.append(interaction_item_from_dict(value))
                except (SaveError, TypeError, ValueError) as exc:
                    raise SaveError(
                        f"invalid save item in {source} at line "
                        f"{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise SaveError(f"could not load save {source}: {exc}") from exc

    try:
        return InteractionContext(items)
    except (TypeError, ValueError) as exc:
        raise SaveError(f"invalid save {source}: {exc}") from exc


__all__ = [
    "SaveError",
    "interaction_item_from_dict",
    "interaction_item_to_dict",
    "iter_interaction_items",
    "load_interaction_save",
    "save_interaction_save",
]
