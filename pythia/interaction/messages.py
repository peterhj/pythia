from __future__ import annotations

import http.client
import json
import math
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

from ._http import USER_AGENT
from ._transport_retry import DEFAULT_MAX_TRANSIENT_RETRIES
from ._transport_retry import retry_delay_seconds
from .context import ContextValidationError
from .context import InteractionContext
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
from .items import TurnSummary
from .items import UserInteractionBoundary
from .model import ModelConfigurationError
from .model import ModelContextWindowError
from .model import ModelError
from .model import ModelResponseError
from .model import ModelSample
from .model import ModelTimeoutError
from .model import ModelTransportError
from .model import SampleParams
from .model import _apply_extra_sample_params
from .model import _timed_sample
from .model_catalog import ModelBinding
from .model_catalog import MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .usage import TokenUsage


DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
MESSAGES_COMPACTION_BETA = "compact-2026-01-12"
_RETRYABLE_HTTP_STATUSES = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504, 529}
)


def resolve_messages_max_output_tokens(
    binding: ModelBinding,
    explicit_value: Optional[int],
) -> int:
    """Resolve an explicit request limit or a catalogued model maximum."""
    if explicit_value is not None:
        if (
            isinstance(explicit_value, bool)
            or not isinstance(explicit_value, int)
            or explicit_value <= 0
        ):
            raise ModelConfigurationError(
                "max_output_tokens must be a positive integer"
            )
        return explicit_value
    if not isinstance(binding, ModelBinding) or binding.api != "messages":
        raise ModelConfigurationError("A Messages model binding is required")
    spec = binding.spec
    catalog_value = (
        None if spec is None else spec.limits.max_output_tokens
    )
    if catalog_value is None:
        raise ModelConfigurationError(
            "Messages max_output_tokens is not available from the model "
            "catalog; provide it explicitly (--max-output-tokens in "
            "CLI/demo)"
        )
    return catalog_value


@dataclass(frozen=True)
class MessagesPromptCaching:
    """Prompt caching at the last cacheable block of each request.

    Each request carries the top-level (automatic) ``cache_control`` and the
    same control on the rightmost ``messages`` block that accepts one.
    """

    ttl: Literal["5m", "1h"] = "5m"

    def __post_init__(self) -> None:
        if not isinstance(self.ttl, str):
            raise TypeError("prompt caching ttl must be a string")
        if self.ttl not in {"5m", "1h"}:
            raise ModelConfigurationError(
                "prompt caching ttl must be '5m' or '1h'"
            )

    def request_cache_control(self) -> Dict[str, str]:
        return {"type": "ephemeral", "ttl": self.ttl}


@dataclass(frozen=True)
class MessagesServerCompaction:
    """Server compaction policy; its trigger is the endpoint-level default.

    A per-call ``SampleParams.auto_compact_tokens`` overrides the trigger, and
    an unset trigger uses the known auto-compaction limit when encoding.
    Without either, the trigger is omitted to use the server default.
    """

    trigger_input_tokens: Optional[int] = None
    pause_after_compaction: bool = False
    instructions: Optional[str] = None

    def __post_init__(self) -> None:
        trigger = self.trigger_input_tokens
        if trigger is not None and (
            isinstance(trigger, bool)
            or not isinstance(trigger, int)
            or trigger < MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
        ):
            raise ModelConfigurationError(
                "trigger_input_tokens must be an integer of at least 50000 "
                "or None"
            )
        if not isinstance(self.pause_after_compaction, bool):
            raise TypeError("pause_after_compaction must be a bool")
        if self.instructions is not None:
            if not isinstance(self.instructions, str):
                raise TypeError("instructions must be a string or None")
            instructions = self.instructions.strip()
            if not instructions:
                raise ModelConfigurationError(
                    "compaction instructions must not be empty"
                )
            object.__setattr__(self, "instructions", instructions)

    def request_edit(self) -> Dict[str, Any]:
        edit: Dict[str, Any] = {"type": "compact_20260112"}
        if self.trigger_input_tokens is not None:
            edit["trigger"] = {
                "type": "input_tokens",
                "value": self.trigger_input_tokens,
            }
        if self.pause_after_compaction:
            edit["pause_after_compaction"] = True
        if self.instructions is not None:
            edit["instructions"] = self.instructions
        return edit


@dataclass(frozen=True)
class MessagesEndpoint:
    binding: ModelBinding = field(repr=False)
    api_key: Optional[str] = field(default=None, repr=False)
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    max_output_tokens: Optional[int] = None
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    server_compaction: Optional[MessagesServerCompaction] = None
    prompt_caching: Optional[MessagesPromptCaching] = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ModelBinding):
            raise TypeError("binding must be ModelBinding")
        if self.binding.api != "messages":
            raise ModelConfigurationError(
                "Messages endpoint requires a messages binding"
            )
        if self.binding.endpoint.model is None:
            raise ModelConfigurationError("Messages endpoint model is required")

        if self.api_key is not None:
            if not isinstance(self.api_key, str):
                raise TypeError("api_key must be a string or None")
            api_key = self.api_key.strip()
            if not api_key:
                raise ModelConfigurationError("api_key must not be empty")
            if any(character.isspace() for character in api_key):
                raise ModelConfigurationError(
                    "api_key must not contain whitespace"
                )
            object.__setattr__(self, "api_key", api_key)

        if not isinstance(self.anthropic_version, str):
            raise TypeError("anthropic_version must be a string")
        anthropic_version = self.anthropic_version.strip()
        if not anthropic_version:
            raise ModelConfigurationError(
                "anthropic_version must not be empty"
            )
        if "\r" in anthropic_version or "\n" in anthropic_version:
            raise ModelConfigurationError(
                "anthropic_version must not contain newlines"
            )
        object.__setattr__(self, "anthropic_version", anthropic_version)
        if ((self.binding.endpoint.auth == "none") != (self.api_key is None)):
            raise ModelConfigurationError("Credentials do not match the resolved endpoint auth policy")

        object.__setattr__(
            self,
            "max_output_tokens",
            resolve_messages_max_output_tokens(
                self.binding,
                self.max_output_tokens,
            ),
        )

        timeout = self.request_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ModelConfigurationError(
                "request_timeout_seconds must be positive and finite"
            )
        object.__setattr__(self, "request_timeout_seconds", float(timeout))

        if self.server_compaction is not None and not isinstance(
            self.server_compaction,
            MessagesServerCompaction,
        ):
            raise TypeError(
                "server_compaction must be MessagesServerCompaction or None"
            )
        if self.prompt_caching is not None and not isinstance(
            self.prompt_caching,
            MessagesPromptCaching,
        ):
            raise TypeError(
                "prompt_caching must be MessagesPromptCaching or None"
            )

    @property
    def model(self) -> str:
        return self.binding.endpoint.model

    @property
    def url(self) -> str:
        return self.binding.endpoint.url


def _append_block(
    messages: List[Dict[str, Any]],
    pending: Optional[Dict[str, Any]],
    role: str,
    block: Dict[str, Any],
) -> Dict[str, Any]:
    if pending is None or pending["role"] != role:
        if pending is not None:
            messages.append(pending)
        pending = {"role": role, "content": []}
    pending["content"].append(block)
    return pending


def _tool_input(arguments_json: str, index: int) -> Dict[str, Any]:
    try:
        value = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ModelConfigurationError(
            f"tool call at item {index} has invalid JSON arguments"
        ) from exc
    if not isinstance(value, dict):
        raise ModelConfigurationError(
            f"tool call at item {index} arguments must decode to an object"
        )
    return value


def _encode_context(
    items: Sequence[InteractionItem],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    # Anthropic ignores content before the latest Messages compaction block.
    # Keep the append-only InteractionContext intact while avoiding an ever-growing
    # outbound HTTP body.
    latest_compaction = -1
    for index, item in enumerate(items):
        if isinstance(item, OpaqueCompaction) and item.protocol == "messages":
            latest_compaction = index
    if latest_compaction >= 0:
        instruction_prefix: List[InteractionItem] = []
        for item in items[:latest_compaction]:
            if isinstance(item, Instructions):
                instruction_prefix.append(item)
                continue
            if isinstance(item, Message) and item.role in {
                "system",
                "developer",
            }:
                instruction_prefix.append(item)
                continue
            if isinstance(
                item,
                (
                    ModelSampleBoundary,
                    ModelFailure,
                    Init,
                    SampleMetadata,
                    CompactionMetadata,
                    TurnSummary,
                    UserInteractionBoundary,
                ),
            ):
                continue
            break
        items = (*instruction_prefix, *items[latest_compaction:])

    system: List[Dict[str, str]] = []
    messages: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    conversation_started = False

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            messages.append(pending)
            pending = None

    # Last-wins Instructions: effective maps to system prompt. Empty text
    # preserved; absence means no entry. model_items() already collapses,
    # this is defensive for direct encoder calls.
    effective: Optional[Instructions] = None
    for item in items:
        if isinstance(item, Instructions):
            effective = item
    if effective is not None:
        system.append({"type": "text", "text": effective.text})

    for index, item in enumerate(items):
        if isinstance(
            item,
            (
                ModelSampleBoundary,
                ModelFailure,
                Init,
                SampleMetadata,
                CompactionMetadata,
                TurnSummary,
                UserInteractionBoundary,
            ),
        ):
            flush()
            continue

        if isinstance(item, Instructions):
            continue

        if isinstance(item, Message):
            if not isinstance(item.content, str):
                raise ModelConfigurationError(
                    "media message content is not supported with the "
                    "Messages API yet"
                )
            if item.role in {"system", "developer"}:
                if conversation_started:
                    raise ModelConfigurationError(
                        f"{item.role} message at item {index} appears after "
                        "Messages conversation content"
                    )
                system.append({"type": "text", "text": item.content})
                continue
            if item.role not in {"user", "assistant"}:
                raise ModelConfigurationError(
                    f"unsupported message role at item {index}: {item.role!r}"
                )
            conversation_started = True
            pending = _append_block(
                messages,
                pending,
                item.role,
                {"type": "text", "text": item.content},
            )
            continue

        if isinstance(item, Reasoning):
            conversation_started = True
            block: Dict[str, Any] = {
                "type": "thinking",
                "thinking": item.content or "\n".join(item.summary),
            }
            if item.content_signature is not None:
                block["signature"] = item.content_signature
            pending = _append_block(messages, pending, "assistant", block)
            continue

        if isinstance(item, ToolCall):
            conversation_started = True
            pending = _append_block(
                messages,
                pending,
                "assistant",
                {
                    "type": "tool_use",
                    "id": item.call_id,
                    "name": item.name,
                    "input": _tool_input(item.arguments_json, index),
                },
            )
            continue

        if isinstance(item, ToolResult):
            conversation_started = True
            pending = _append_block(
                messages,
                pending,
                "user",
                {
                    "type": "tool_result",
                    "tool_use_id": item.call_id,
                    "content": item.output,
                    "is_error": not item.success,
                },
            )
            continue

        if isinstance(item, OpaqueCompaction):
            if item.protocol != "messages":
                raise ModelConfigurationError(
                    "Messages cannot encode a Responses opaque compaction"
                )
            conversation_started = True
            pending = _append_block(
                messages,
                pending,
                "assistant",
                {"type": "compaction", "content": item.payload},
            )
            continue
        if isinstance(item, ContextPrefix):
            raise ModelConfigurationError(
                "ContextPrefix must be projected before request encoding"
            )
        raise ModelConfigurationError(
            f"unsupported interaction item at index {index}: {item!r}"
        )

    flush()
    if not messages:
        raise ModelConfigurationError("cannot sample an empty model context")
    return system, messages


def _accepts_cache_control(block: Mapping[str, Any]) -> bool:
    """Whether an encoded block may carry an explicit cache breakpoint.

    This is an allow-list. Anthropic rejects breakpoints on thinking and empty
    text blocks, and unlisted block types are never marked. Blank tool results
    are skipped because it is unverified whether they accept a breakpoint.
    """
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        return isinstance(text, str) and bool(text.strip())
    if block_type == "tool_result":
        content = block.get("content")
        return isinstance(content, str) and bool(content.strip())
    return block_type in {"tool_use", "compaction"}


def _mark_last_cacheable_block(
    messages: List[Dict[str, Any]],
    cache_control: Dict[str, str],
) -> None:
    """Mark the rightmost block that accepts ``cache_control``, if any."""
    for message in reversed(messages):
        for block in reversed(message["content"]):
            if _accepts_cache_control(block):
                block["cache_control"] = cache_control
                return


def _encode_tools(tools: Sequence[Any]) -> List[Dict[str, Any]]:
    encoded: List[Dict[str, Any]] = []
    seen = set()
    for index, tool in enumerate(tools):
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        parameters = getattr(tool, "parameters", None)
        if not isinstance(name, str) or not name.strip():
            raise ModelConfigurationError(
                f"tool {index} must have a non-empty name"
            )
        if name in seen:
            raise ModelConfigurationError(f"duplicate tool name: {name!r}")
        seen.add(name)
        if not isinstance(description, str):
            raise ModelConfigurationError(
                f"tool {name!r} must have a string description"
            )
        if not isinstance(parameters, Mapping):
            raise ModelConfigurationError(
                f"tool {name!r} parameters must be a mapping"
            )
        encoded.append(
            {
                "name": name,
                "description": description,
                "input_schema": dict(parameters),
            }
        )
    return encoded


def _apply_sample_params(
    payload: Dict[str, Any],
    sample_params: Optional[SampleParams],
) -> None:
    """Encode typed knobs; the builder resolves the required budget."""
    if sample_params is None:
        return
    if sample_params.seed is not None:
        raise ModelConfigurationError(
            "Messages does not support the seed sampling option"
        )
    if sample_params.temperature is not None:
        payload["temperature"] = sample_params.temperature
    if sample_params.top_p is not None:
        payload["top_p"] = sample_params.top_p
    if sample_params.stop:
        payload["stop_sequences"] = list(sample_params.stop)


def _require_string(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ModelResponseError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ModelResponseError(f"{field_name} must not be empty")
    return value


def _decode_content(value: Any) -> Tuple[InteractionItem, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ModelResponseError("message.content must be a list")
    items: List[InteractionItem] = []
    for index, block in enumerate(value):
        if not isinstance(block, Mapping):
            raise ModelResponseError(
                f"message.content[{index}] must be an object"
            )
        block_type = block.get("type")
        if block_type == "text":
            items.append(
                Message(
                    role="assistant",
                    content=_require_string(
                        block.get("text"),
                        f"message.content[{index}].text",
                    ),
                )
            )
            continue
        if block_type == "thinking":
            signature_value = block.get("signature")
            content_signature = (
                None
                if signature_value is None
                else _require_string(
                    signature_value,
                    f"message.content[{index}].signature",
                    allow_empty=False,
                )
            )
            items.append(
                Reasoning(
                    content=_require_string(
                        block.get("thinking"),
                        f"message.content[{index}].thinking",
                    ),
                    content_signature=content_signature,
                )
            )
            continue
        if block_type == "compaction":
            items.append(
                OpaqueCompaction.from_messages(
                    _require_string(
                        block.get("content"),
                        f"message.content[{index}].content",
                        allow_empty=False,
                    )
                )
            )
            continue
        if block_type == "tool_use":
            tool_input = block.get("input", {})
            if not isinstance(tool_input, Mapping):
                raise ModelResponseError(
                    f"message.content[{index}].input must be an object"
                )
            try:
                arguments_json = json.dumps(
                    dict(tool_input),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ModelResponseError(
                    f"message.content[{index}].input is not JSON-compatible"
                ) from exc
            items.append(
                ToolCall(
                    name=_require_string(
                        block.get("name"),
                        f"message.content[{index}].name",
                        allow_empty=False,
                    ),
                    call_id=_require_string(
                        block.get("id"),
                        f"message.content[{index}].id",
                        allow_empty=False,
                    ),
                    arguments_json=arguments_json,
                )
            )
            continue
        raise ModelResponseError(
            f"unsupported message.content[{index}] type: {block_type!r}"
        )
    return tuple(items)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _usage_counts(value: Mapping[str, Any]) -> Tuple[int, int, int]:
    # Anthropic input_tokens excludes cache writes and reads. The optional
    # cache_creation TTL breakdown is already included in its aggregate below.
    cache_creation_tokens = _nonnegative_int(
        value.get("cache_creation_input_tokens")
    )
    cache_read_tokens = _nonnegative_int(
        value.get("cache_read_input_tokens")
    )
    return (
        _nonnegative_int(value.get("input_tokens"))
        + cache_creation_tokens
        + cache_read_tokens,
        _nonnegative_int(value.get("output_tokens")),
        cache_read_tokens,
    )


def _decode_usage(value: Any) -> TokenUsage:
    if not isinstance(value, Mapping):
        return TokenUsage()
    input_tokens, output_tokens, cached_input_tokens = _usage_counts(value)
    iterations = value.get("iterations")
    if isinstance(iterations, Sequence) and not isinstance(
        iterations,
        (str, bytes, bytearray),
    ):
        iteration_input_tokens = 0
        iteration_output_tokens = 0
        iteration_cached_input_tokens = 0
        iteration_cache_creation_tokens = 0
        iteration_count = 0
        for iteration in iterations:
            if not isinstance(iteration, Mapping):
                continue
            iteration_count += 1
            iteration_input, iteration_output, iteration_cached = (
                _usage_counts(iteration)
            )
            iteration_input_tokens += iteration_input
            iteration_output_tokens += iteration_output
            iteration_cached_input_tokens += iteration_cached
            iteration_cache_creation_tokens += _nonnegative_int(
                iteration.get("cache_creation_input_tokens")
            )
        if iteration_count:
            # Some beta response versions report cache fields only at the
            # top level, outside the per-iteration breakdown.
            if (
                iteration_cached_input_tokens == 0
                and iteration_cache_creation_tokens == 0
            ):
                top_level_cache_creation = _nonnegative_int(
                    value.get("cache_creation_input_tokens")
                )
                iteration_input_tokens += (
                    cached_input_tokens + top_level_cache_creation
                )
            input_tokens = iteration_input_tokens
            output_tokens = iteration_output_tokens
            if iteration_cached_input_tokens:
                cached_input_tokens = iteration_cached_input_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=min(
            input_tokens,
            cached_input_tokens,
        ),
    )


def _decode_response(payload: Any) -> ModelSample:
    if not isinstance(payload, Mapping):
        raise ModelResponseError("Messages response must be an object")
    if payload.get("type") != "message":
        raise ModelResponseError(
            f"Messages response has unsupported type: {payload.get('type')!r}"
        )
    if payload.get("role") != "assistant":
        raise ModelResponseError(
            f"Messages response has unsupported role: {payload.get('role')!r}"
        )
    return ModelSample(
        items=_decode_content(payload.get("content")),
        stop_reason=(
            None
            if payload.get("stop_reason") is None
            else _require_string(payload.get("stop_reason"), "stop_reason")
        ),
        usage=_decode_usage(payload.get("usage")),
    )


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read()
    except Exception:
        return ""
    if not isinstance(payload, (bytes, bytearray)):
        return ""
    return bytes(payload).decode("utf-8", errors="replace")[:4096]


def _is_context_window_error(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "context window",
            "maximum context length",
            "context length exceeded",
            "too many tokens",
        )
    )


def _messages_http_failure(
    status: int,
    detail: str,
    *,
    model: str,
    attempt_count: int,
    recovery: Tuple[str, ...],
) -> ModelError:
    context_window = status == 413 or _is_context_window_error(detail)
    message = (
        f"Messages HTTP {status}: {detail}"
        if detail
        else f"Messages HTTP {status}: request failed"
    )
    failure = ModelFailure(
        category="context_window" if context_window else "http_error",
        message=(
            f"Messages HTTP {status}: context window exceeded"
            if context_window
            else f"Messages HTTP {status}: request failed"
        ),
        provider="messages",
        model=model,
        http_status=status,
        attempt_count=attempt_count,
        recovery=recovery,
    )
    error_type = ModelContextWindowError if context_window else ModelTransportError
    return error_type(message, failure=failure)


def _messages_transport_failure(
    *,
    timeout: bool,
    model: str,
    attempt_count: int,
    recovery: Tuple[str, ...],
) -> ModelTransportError:
    message = (
        "Messages request timed out"
        if timeout
        else "Messages request failed before a response was completed"
    )
    failure = ModelFailure(
        category="request_timeout" if timeout else "request_transport",
        message=message,
        provider="messages",
        model=model,
        attempt_count=attempt_count,
        recovery=recovery,
    )
    error_type = ModelTimeoutError if timeout else ModelTransportError
    return error_type(message, failure=failure)


def _close_response(response: Any) -> None:
    if response is None:
        return
    close = getattr(response, "close", None)
    if callable(close):
        close()


class MessagesModel:
    def __init__(
        self,
        endpoint: MessagesEndpoint,
        *,
        opener: Optional[Callable[..., Any]] = None,
        retry_sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        if not isinstance(endpoint, MessagesEndpoint):
            raise TypeError("endpoint must be MessagesEndpoint")
        if retry_sleep is not None and not callable(retry_sleep):
            raise TypeError("retry_sleep must be callable or None")
        self.endpoint = endpoint
        self.binding = endpoint.binding
        self._opener = opener or urllib.request.urlopen
        self._retry_sleep = time.sleep if retry_sleep is None else retry_sleep

    @property
    def auto_compaction_owner(self) -> str:
        """``server`` with configured server compaction, else ``host``."""
        return "server" if self.endpoint.server_compaction is not None else "host"

    @property
    def max_context_tokens(self) -> Optional[int]:
        """Known context ceiling and implicit compaction trigger, or None."""
        spec = self.binding.spec
        return None if spec is None else spec.limits.max_context_tokens

    @property
    def max_output_tokens(self) -> Optional[int]:
        """Known output ceiling, or None; does not change request max_tokens."""
        spec = self.binding.spec
        return None if spec is None else spec.limits.max_output_tokens

    def _build_request_payload(
        self,
        context: InteractionContext,
        tools: Sequence[Any],
        sample_params: Optional[SampleParams],
    ) -> Dict[str, Any]:
        if not isinstance(context, InteractionContext):
            raise TypeError("context must be InteractionContext")
        context.assert_model_ready()
        system, messages = _encode_context(context.model_items())
        spec = self.binding.spec
        # Per-call > endpoint; the endpoint already resolved explicit > catalog.
        output_budget = (
            self.endpoint.max_output_tokens
            if sample_params is None or sample_params.max_output_tokens is None
            else sample_params.max_output_tokens
        )
        payload: Dict[str, Any] = {
            "model": (
                self.binding.endpoint.model
            ),
            "max_tokens": output_budget,
            "messages": messages,
            "stream": False,
        }
        if system:
            payload["system"] = system
        encoded_tools = _encode_tools(tools)
        if encoded_tools:
            payload["tools"] = encoded_tools
        if self.endpoint.prompt_caching is not None:
            # The top-level control lets the API place its automatic breakpoint
            # at the last cacheable block. The same control also marks the
            # rightmost block that accepts an explicit breakpoint (never
            # thinking or empty blocks, nor system or tools). The TTLs must
            # match: a different TTL on the block the automatic breakpoint
            # lands on is an HTTP 400, while the same TTL makes it a no-op.
            # TODO: Add a per-call ``SampleParams.enable_prompt_caching``
            # (None inherits this policy, False omits both the top-level and
            # the block cache_control, True creates no policy), and set it to
            # False for pi summary requests: nothing reads their cache
            # entries, and 5-minute writes cost 1.25x base input (pi sends
            # summaries with cacheRetention "none").
            prompt_caching = self.endpoint.prompt_caching
            payload["cache_control"] = prompt_caching.request_cache_control()
            _mark_last_cacheable_block(
                payload["messages"],
                prompt_caching.request_cache_control(),
            )
        compaction = self.endpoint.server_compaction
        # A per-call False suppresses configured server compaction; None and
        # True leave it as configured and never create a policy.
        suppressed = (
            sample_params is not None
            and sample_params.enable_auto_compaction is False
        )
        if compaction is not None and not suppressed:
            # Per-call > endpoint > catalog; if none is known, omit the
            # trigger so the server uses its default.
            auto_compact_context = (
                None if sample_params is None else sample_params.auto_compact_tokens
            )
            if auto_compact_context is None:
                auto_compact_context = compaction.trigger_input_tokens
            if auto_compact_context is None and spec is not None:
                auto_compact_context = spec.limits.auto_compact_context_tokens
            if (
                auto_compact_context is not None
                and auto_compact_context
                < MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
            ):
                raise ModelConfigurationError(
                    "auto-compaction trigger must be at least "
                    f"{MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS} tokens"
                )
            compaction = replace(
                compaction,
                trigger_input_tokens=auto_compact_context,
            )
            payload["context_management"] = {
                "edits": [compaction.request_edit()]
            }
        _apply_sample_params(payload, sample_params)
        _apply_extra_sample_params(payload, self.binding, sample_params)
        return payload

    @_timed_sample
    def sample(
        self,
        context: InteractionContext,
        *,
        tools: Sequence[Any] = (),
        sample_params: Optional[SampleParams] = None,
    ) -> ModelSample:
        if sample_params is not None and not isinstance(sample_params, SampleParams):
            raise TypeError("sample_params must be SampleParams or None")
        payload = self._build_request_payload(context, tools, sample_params)
        try:
            request_data = json.dumps(payload, ensure_ascii=False).encode(
                "utf-8"
            )
        except (TypeError, ValueError) as exc:
            raise ModelConfigurationError(
                "Messages request is not JSON-serializable"
            ) from exc
        headers = {
            "Accept": "application/json",
            "Anthropic-Version": self.endpoint.anthropic_version,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if "context_management" in payload:
            headers["Anthropic-Beta"] = MESSAGES_COMPACTION_BETA
        if self.endpoint.api_key is not None:
            headers["X-API-Key"] = self.endpoint.api_key
        attempts = 0
        retries = 0
        recovery: List[str] = []
        while True:
            attempts += 1
            request = urllib.request.Request(
                self.endpoint.url,
                data=request_data,
                headers=headers,
                method="POST",
            )
            response = None
            try:
                try:
                    response = self._opener(
                        request,
                        timeout=self.endpoint.request_timeout_seconds,
                    )
                except urllib.error.HTTPError as exc:
                    detail = _read_http_error_body(exc) or str(exc)
                    status = exc.code
                    response_headers = exc.headers
                    exc.close()
                    if (
                        status in _RETRYABLE_HTTP_STATUSES
                        and not _is_context_window_error(detail)
                        and retries < DEFAULT_MAX_TRANSIENT_RETRIES
                    ):
                        retries += 1
                        recovery.append(f"http_{status}_retry")
                        self._retry_sleep(
                            retry_delay_seconds(retries, response_headers)
                        )
                        continue
                    raise _messages_http_failure(
                        status,
                        detail,
                        model=self.endpoint.model,
                        attempt_count=attempts,
                        recovery=tuple(recovery),
                    ) from exc

                status = getattr(response, "status", None)
                response_headers = getattr(response, "headers", None)
                raw = response.read()
                if not isinstance(raw, (bytes, bytearray)):
                    raise ModelResponseError("HTTP response body must be bytes")
                if isinstance(status, int) and not 200 <= status < 300:
                    detail = bytes(raw).decode("utf-8", errors="replace")[:4096]
                    if (
                        status in _RETRYABLE_HTTP_STATUSES
                        and not _is_context_window_error(detail)
                        and retries < DEFAULT_MAX_TRANSIENT_RETRIES
                    ):
                        retries += 1
                        recovery.append(f"http_{status}_retry")
                        _close_response(response)
                        response = None
                        self._retry_sleep(
                            retry_delay_seconds(retries, response_headers)
                        )
                        continue
                    raise _messages_http_failure(
                        status,
                        detail,
                        model=self.endpoint.model,
                        attempt_count=attempts,
                        recovery=tuple(recovery),
                    )
                try:
                    text = bytes(raw).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ModelResponseError("response is not UTF-8 JSON") from exc
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ModelResponseError("response is not valid JSON") from exc
                try:
                    sample = _decode_response(decoded)
                except ContextValidationError as exc:
                    raise ModelResponseError(str(exc)) from exc
                except ModelResponseError:
                    raise
                except (TypeError, ValueError) as exc:
                    raise ModelResponseError(str(exc)) from exc
                return replace(
                    sample,
                    request_attempts=attempts,
                    recovery=tuple(recovery),
                )
            except urllib.error.URLError as exc:
                timeout = isinstance(
                    exc.reason,
                    (TimeoutError, socket.timeout),
                )
                if retries < DEFAULT_MAX_TRANSIENT_RETRIES:
                    retries += 1
                    recovery.append(
                        "request_timeout_retry"
                        if timeout
                        else "connection_retry"
                    )
                    _close_response(response)
                    response = None
                    self._retry_sleep(retry_delay_seconds(retries))
                    continue
                raise _messages_transport_failure(
                    timeout=timeout,
                    model=self.endpoint.model,
                    attempt_count=attempts,
                    recovery=tuple(recovery),
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                if retries < DEFAULT_MAX_TRANSIENT_RETRIES:
                    retries += 1
                    recovery.append("request_timeout_retry")
                    _close_response(response)
                    response = None
                    self._retry_sleep(retry_delay_seconds(retries))
                    continue
                raise _messages_transport_failure(
                    timeout=True,
                    model=self.endpoint.model,
                    attempt_count=attempts,
                    recovery=tuple(recovery),
                ) from exc
            except (OSError, http.client.HTTPException) as exc:
                if retries < DEFAULT_MAX_TRANSIENT_RETRIES:
                    retries += 1
                    recovery.append("connection_retry")
                    _close_response(response)
                    response = None
                    self._retry_sleep(retry_delay_seconds(retries))
                    continue
                raise _messages_transport_failure(
                    timeout=False,
                    model=self.endpoint.model,
                    attempt_count=attempts,
                    recovery=tuple(recovery),
                ) from exc
            finally:
                _close_response(response)


__all__ = [
    "DEFAULT_ANTHROPIC_VERSION",
    "MESSAGES_COMPACTION_BETA",
    "MessagesEndpoint",
    "MessagesModel",
    "MessagesPromptCaching",
    "MessagesServerCompaction",
    "resolve_messages_max_output_tokens",
]
