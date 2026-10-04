from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

from .usage import TokenUsage
from ._tool_spec import ToolSpec
from ._tool_spec import _copy_schema


_COMPACTION_PROTOCOL_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


def _require_string(value: object, field_name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_elapsed_seconds(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("elapsed_seconds must be numeric or None")
    try:
        elapsed = float(value)
    except OverflowError:
        raise ValueError("elapsed_seconds must be nonnegative and finite") from None
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_seconds must be nonnegative and finite")
    return elapsed


def _fresh_prefix_id() -> str:
    return f"session_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class Init:
    """Durable identity established before an interaction begins."""

    prefix_id: str = field(default_factory=_fresh_prefix_id)
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model is not None:
            _require_string(self.model, "model", allow_empty=False)
        _require_string(self.prefix_id, "prefix_id", allow_empty=False)
        if "\r" in self.prefix_id or "\n" in self.prefix_id:
            raise ValueError("prefix_id must not contain newlines")


@dataclass(frozen=True)
class TextPart:
    """A plain-text content item inside a media ``Message``.

    Mirrors the Responses ``input_text``/``output_text`` content item. A
    text-only message is stored as a plain ``str`` and never carries this type,
    so it is normally only observed alongside a non-text part.
    """

    text: str

    def __post_init__(self) -> None:
        _require_string(self.text, "TextPart.text")


@dataclass(frozen=True)
class MediaPart:
    """A non-text media content item.

    ``source_uri`` is an ``http(s)`` URL or a ``data:`` URL. This slice only
    produces images, and every media part is serialized as the Responses
    ``input_image`` content item and the Chat Completions ``image_url`` part.

    TODO: always emitted as ``input_image`` for now. Add Responses
    ``input_file``, filename, and ``detail`` handling once file/document
    support is designed.
    """

    source_uri: str

    def __post_init__(self) -> None:
        _require_string(self.source_uri, "MediaPart.source_uri", allow_empty=False)
        if "\r" in self.source_uri or "\n" in self.source_uri:
            raise ValueError("MediaPart.source_uri must not contain newlines")


ContentPart = Union[TextPart, MediaPart]


@dataclass(frozen=True)
class Message:
    """An input or output message (there is no separate output message type).

    ``content`` is either plain text (a ``str``) or an ordered tuple of content
    parts mirroring a Responses ``content`` array. A tuple containing only
    ``TextPart`` items is canonicalized back to a ``str`` in ``__post_init__``,
    so ``content`` is a string exactly when the message is text-only. Non-text
    parts are only permitted for ``role == "user"``.
    """

    role: str
    content: Union[str, Tuple[ContentPart, ...]]

    def __post_init__(self) -> None:
        _require_string(self.role, "role", allow_empty=False)
        content = self.content
        if isinstance(content, str):
            return
        if isinstance(content, (list, tuple)):
            parts = tuple(content)
            if not parts:
                raise TypeError(
                    "content must be a string or a non-empty sequence of "
                    "content parts"
                )
            for index, part in enumerate(parts):
                if not isinstance(part, (TextPart, MediaPart)):
                    raise TypeError(
                        "content must be a string or a tuple of typed content "
                        f"parts; content[{index}] is {type(part).__name__}"
                    )
            object.__setattr__(self, "content", parts)
            if all(isinstance(part, TextPart) for part in parts):
                object.__setattr__(
                    self, "content", "\n".join(part.text for part in parts)
                )
                return
            if self.role != "user":
                raise ValueError(
                    "non-text content parts require role 'user'; "
                    f"got {self.role!r}"
                )
            return
        raise TypeError("content must be a string or a tuple of content parts")

    @property
    def has_media(self) -> bool:
        return not isinstance(self.content, str)

    @property
    def content_text(self) -> str:
        """The concatenated text projection, for every role."""
        content = self.content
        if isinstance(content, str):
            return content
        return "\n".join(
            part.text for part in content if isinstance(part, TextPart)
        )

    @property
    def parts(self) -> Tuple[ContentPart, ...]:
        """The ordered content view; a string becomes a single text part."""
        content = self.content
        if isinstance(content, str):
            return (TextPart(content),)
        return content


@dataclass(frozen=True)
class Instructions:
    """Optional persistent system instructions.

    Corresponds to a Chat Completions ``system`` message. Empty and
    whitespace-only text is supported; ``no instructions`` is represented
    only by the absence of an ``Instructions`` item. When several
    ``Instructions`` items appear in a log, the last one overrides all
    earlier ones (see ``InteractionContext.model_items``).
    """

    text: str

    def __post_init__(self) -> None:
        _require_string(self.text, "text")


@dataclass(frozen=True)
class Tools:
    """A durable snapshot of the runtime's advertised tool specifications.

    Later snapshots supersede earlier ones for audit purposes; an empty tuple
    records no tools. Unlike Instructions, these are not restored or used as
    provider input: the runtime environment and explicit ``tools=`` argument
    remain authoritative. Schemas are defensively copied, and no handlers or
    runtime state are retained.
    """

    specs: Tuple[ToolSpec, ...] = ()

    def __post_init__(self) -> None:
        specs = []
        names = set()
        for index, spec in enumerate(self.specs):
            if not isinstance(spec, ToolSpec):
                raise TypeError(f"specs[{index}] must be ToolSpec")
            if spec.name in names:
                raise ValueError(f"duplicate tool name: {spec.name!r}")
            names.add(spec.name)
            specs.append(ToolSpec(
                spec.name, spec.description, _copy_schema(spec.parameters),
            ))
        object.__setattr__(self, "specs", tuple(specs))


@dataclass(frozen=True)
class Reasoning:
    content: str
    summary: Tuple[str, ...] = ()
    encrypted_content: Optional[str] = field(default=None, repr=False)
    content_signature: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_string(self.content, "content")
        summary = tuple(self.summary)
        for index, value in enumerate(summary):
            _require_string(value, f"summary[{index}]")
        object.__setattr__(self, "summary", summary)
        if self.encrypted_content is not None:
            _require_string(
                self.encrypted_content,
                "encrypted_content",
                allow_empty=False,
            )
        if self.content_signature is not None:
            _require_string(
                self.content_signature,
                "content_signature",
                allow_empty=False,
            )


@dataclass(frozen=True)
class ToolCall:
    name: str
    call_id: str
    arguments_json: str

    def __post_init__(self) -> None:
        _require_string(self.name, "name", allow_empty=False)
        _require_string(self.call_id, "call_id", allow_empty=False)
        _require_string(self.arguments_json, "arguments_json")


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: str
    success: bool = True

    def __post_init__(self) -> None:
        _require_string(self.call_id, "call_id", allow_empty=False)
        _require_string(self.output, "output")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")


@dataclass(frozen=True)
class UserToolCall:
    """A user-authorized tool invocation, durable but never provider input."""

    call: ToolCall

    def __post_init__(self) -> None:
        if not isinstance(self.call, ToolCall):
            raise TypeError("call must be ToolCall")


@dataclass(frozen=True)
class UserToolResult:
    """The safe, log-only outcome of a user tool (not an assistant tool)."""

    result: ToolResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, ToolResult):
            raise TypeError("result must be ToolResult")


@dataclass(frozen=True)
class ModelSampleBoundary:
    """Marks the end of one model sample without emitting provider content."""


@dataclass(frozen=True)
class UserInteractionBoundary:
    """Marks the end of one user interaction without emitting provider content."""


@dataclass(frozen=True)
class SampleMetadata:
    """Durable per-sample usage, elapsed time, and provider continuity.

    One ``SampleMetadata`` is recorded per completed model sample
    (see ``ModelSample.context_items``). It preserves the provider's
    ``TokenUsage`` for that sample plus the provider continuity tokens.
    ``elapsed_seconds`` is host-measured sample latency, not provider compute
    time; ``None`` means it was not measured (including in legacy logs).
    ``request_attempts`` and ``recovery`` record bounded transport/auth recovery
    without retaining credentials or provider response bodies.
    It is *not* a cumulative end-of-turn aggregate; see ``TurnSummary``.
    """

    usage: TokenUsage
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)
    elapsed_seconds: Optional[float] = None
    request_attempts: int = 1
    recovery: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        object.__setattr__(
            self, "elapsed_seconds", _validate_elapsed_seconds(self.elapsed_seconds)
        )
        if (
            isinstance(self.request_attempts, bool)
            or not isinstance(self.request_attempts, int)
            or self.request_attempts <= 0
        ):
            raise ValueError("request_attempts must be a positive integer")
        recovery = tuple(self.recovery)
        for index, value in enumerate(recovery):
            _require_string(value, f"recovery[{index}]", allow_empty=False)
            if "\r" in value or "\n" in value:
                raise ValueError(f"recovery[{index}] must not contain newlines")
        object.__setattr__(self, "recovery", recovery)
        for field_name in (
            "provider_session_id",
            "provider_turn_id",
            "provider_turn_state",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            _require_string(value, field_name, allow_empty=False)
            if "\r" in value or "\n" in value:
                raise ValueError(f"{field_name} must not contain newlines")


@dataclass(frozen=True)
class CompactionMetadata:
    """Durable metadata for one successful explicit compaction operation.

    ``protocol`` identifies the compaction procedure (for example, ``pi`` or
    ``responses_compaction_v2``), while ``OpaqueCompaction.protocol``
    identifies only the provider wire family used to replay an opaque payload.
    The item is operational metadata and is never encoded into a model request.
    """

    # TODO: Record estimated context tokens before and after compaction (pi's
    # ``tokensBefore`` and ``estimatedTokensAfter``).
    usage: TokenUsage
    protocol: str
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)
    provider_response_id: Optional[str] = field(default=None, repr=False)
    elapsed_seconds: Optional[float] = None
    request_attempts: int = 1
    recovery: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        protocol = _require_string(
            self.protocol,
            "protocol",
            allow_empty=False,
        )
        if _COMPACTION_PROTOCOL_RE.fullmatch(protocol) is None:
            raise ValueError(
                "protocol must be a lowercase snake-case identifier of at "
                "most 128 characters"
            )
        object.__setattr__(
            self,
            "elapsed_seconds",
            _validate_elapsed_seconds(self.elapsed_seconds),
        )
        if (
            isinstance(self.request_attempts, bool)
            or not isinstance(self.request_attempts, int)
            or self.request_attempts <= 0
        ):
            raise ValueError("request_attempts must be a positive integer")
        recovery = tuple(self.recovery)
        for index, value in enumerate(recovery):
            _require_string(value, f"recovery[{index}]", allow_empty=False)
            if "\r" in value or "\n" in value:
                raise ValueError(f"recovery[{index}] must not contain newlines")
        object.__setattr__(self, "recovery", recovery)
        for field_name in (
            "provider_session_id",
            "provider_turn_id",
            "provider_turn_state",
            "provider_response_id",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            _require_string(value, field_name, allow_empty=False)
            if "\r" in value or "\n" in value:
                raise ValueError(f"{field_name} must not contain newlines")


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class ModelFailure:
    """Safe, durable diagnostics for a model attempt that did not complete.

    These fields are deliberately bounded to metadata. Request/response bodies,
    credentials, user content, and provider continuity tokens do not belong in
    this item. Model adapters never encode it back to a provider.
    """

    category: str
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    auth_source: Optional[str] = None
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    response_id: Optional[str] = None
    cf_ray: Optional[str] = None
    authorization_error: Optional[str] = None
    auth_error_code: Optional[str] = None
    error_code: Optional[str] = None
    attempt_count: int = 1
    event_count: int = 0
    event_types: Tuple[str, ...] = ()
    completed_item_count: int = 0
    last_event_type: Optional[str] = None
    last_sequence_number: Optional[int] = None
    recovery: Tuple[str, ...] = ()
    elapsed_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        for field_name in ("category", "message"):
            value = _require_string(
                getattr(self, field_name), field_name, allow_empty=False,
            )
            if "\r" in value or "\n" in value:
                raise ValueError(f"{field_name} must not contain newlines")
            limit = 128 if field_name == "category" else 1024
            if len(value) > limit:
                raise ValueError(f"{field_name} must not exceed {limit} characters")
        for field_name in (
            "provider",
            "model",
            "auth_source",
            "request_id",
            "response_id",
            "cf_ray",
            "authorization_error",
            "auth_error_code",
            "error_code",
            "last_event_type",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            _require_string(value, field_name, allow_empty=False)
            if "\r" in value or "\n" in value:
                raise ValueError(f"{field_name} must not contain newlines")
            if len(value) > 256:
                raise ValueError(f"{field_name} must not exceed 256 characters")
        if self.http_status is not None and (
            isinstance(self.http_status, bool)
            or not isinstance(self.http_status, int)
            or not 100 <= self.http_status <= 599
        ):
            raise ValueError("http_status must be from 100 through 599 or None")
        if (
            isinstance(self.attempt_count, bool)
            or not isinstance(self.attempt_count, int)
            or self.attempt_count <= 0
        ):
            raise ValueError("attempt_count must be a positive integer")
        _require_nonnegative_int(self.event_count, "event_count")
        event_types = tuple(self.event_types)
        for index, value in enumerate(event_types):
            _require_string(value, f"event_types[{index}]", allow_empty=False)
            if "\r" in value or "\n" in value:
                raise ValueError(f"event_types[{index}] must not contain newlines")
            if len(value) > 320:
                raise ValueError(
                    f"event_types[{index}] must not exceed 320 characters"
                )
        object.__setattr__(self, "event_types", event_types)
        _require_nonnegative_int(
            self.completed_item_count,
            "completed_item_count",
        )
        if self.last_sequence_number is not None:
            _require_nonnegative_int(
                self.last_sequence_number,
                "last_sequence_number",
            )
        recovery = tuple(self.recovery)
        for index, value in enumerate(recovery):
            _require_string(value, f"recovery[{index}]", allow_empty=False)
            if "\r" in value or "\n" in value:
                raise ValueError(f"recovery[{index}] must not contain newlines")
            if len(value) > 256:
                raise ValueError(f"recovery[{index}] must not exceed 256 characters")
        object.__setattr__(self, "recovery", recovery)
        object.__setattr__(
            self,
            "elapsed_seconds",
            _validate_elapsed_seconds(self.elapsed_seconds),
        )


@dataclass(frozen=True)
class TurnSummary:
    """Cumulative end-of-turn usage derived from per-sample ``SampleMetadata``.

    Ports the earlier autopythia/contradex ``AgentState`` accounting
    (``output_tokens_sum``, ``cache_hit_*`` warm stats, ``non_cache_hit`` cold
    stats, ``total_usage_tokens`` context, ``compaction_count``) to
    ``pythia.interaction`` without changing ``SampleMetadata`` semantics.

    ``TurnSummary`` is encoder-transparent (never sent to the provider) and
    durable. It is derived via :func:`summarize_turn_usage`, which folds over
    the raw log's request metadata and counts compaction markers.
    Existing ``TurnSummary`` items are skipped by the fold so re-summarizing
    a context that already contains summaries does not double-count.

    ``elapsed_seconds`` is independently measured active-turn wall time. It is
    not the sum of request metadata durations and is not session-cumulative.
    ``None`` means that the caller did not supply a turn measurement.
    """

    input_tokens_sum: int = 0
    output_tokens_sum: int = 0
    cached_input_tokens_sum: int = 0
    cached_input_tokens_max: int = 0
    non_cached_input_tokens_sum: int = 0
    context_tokens: int = 0
    sample_count: int = 0
    compaction_count: int = 0
    elapsed_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens_sum",
            "output_tokens_sum",
            "cached_input_tokens_sum",
            "cached_input_tokens_max",
            "non_cached_input_tokens_sum",
            "context_tokens",
            "sample_count",
            "compaction_count",
        ):
            _require_nonnegative_int(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "elapsed_seconds",
            _validate_elapsed_seconds(self.elapsed_seconds),
        )

    @property
    def goal_accounting_tokens(self) -> int:
        """Billable tokens: cold (non-cached) input + output."""
        return self.non_cached_input_tokens_sum + self.output_tokens_sum

    @property
    def has_detailed_usage(self) -> bool:
        """Whether any folded sample carried nonzero usage."""
        return bool(
            self.input_tokens_sum
            or self.output_tokens_sum
            or self.cached_input_tokens_sum
            or self.cached_input_tokens_max
            or self.non_cached_input_tokens_sum
            or self.sample_count
        )


def summarize_turn_usage(
    items,
    *,
    elapsed_seconds: Optional[float] = None,
) -> "TurnSummary":
    """Fold request metadata into a cumulative-usage ``TurnSummary``.

    Mirrors ``contradex.kernel.AgentState.add_response_usage``:
    warm per sample is ``min(cached, input)``, cold is ``max(0, input-warm)``.
    Successful explicit compaction usage contributes to the cumulative token
    sums but not ``sample_count``. ``context_tokens`` tracks the last ordinary
    sample's ``total_tokens`` (contradex ``total_usage_tokens`` semantics:
    current window, not a sum); a compaction request's total is not the size of
    its installed prefix.
    ``compaction_count`` counts ``OpaqueCompaction``/``ContextPrefix``
    markers. A context prefix counts here because compaction is currently its
    only producer. ``TurnSummary`` items in the input are skipped.

    Pass the raw log (``context.items``) for session-cumulative stats, or a
    slice after the last ``UserInteractionBoundary`` for per-turn stats.
    """
    input_tokens_sum = 0
    output_tokens_sum = 0
    cached_input_tokens_sum = 0
    cached_input_tokens_max = 0
    non_cached_input_tokens_sum = 0
    context_tokens = 0
    sample_count = 0
    compaction_count = 0
    for item in items:
        if isinstance(item, (SampleMetadata, CompactionMetadata)):
            usage = item.usage
            warm = min(usage.cached_input_tokens, usage.input_tokens)
            cold = max(0, usage.input_tokens - warm)
            input_tokens_sum += usage.input_tokens
            output_tokens_sum += usage.output_tokens
            cached_input_tokens_sum += warm
            cached_input_tokens_max = max(cached_input_tokens_max, warm)
            non_cached_input_tokens_sum += cold
            if isinstance(item, SampleMetadata):
                context_tokens = usage.total_tokens
                sample_count += 1
        elif isinstance(item, (OpaqueCompaction, ContextPrefix)):
            compaction_count += 1
        elif isinstance(item, TurnSummary):
            continue
    return TurnSummary(
        input_tokens_sum=input_tokens_sum,
        output_tokens_sum=output_tokens_sum,
        cached_input_tokens_sum=cached_input_tokens_sum,
        cached_input_tokens_max=cached_input_tokens_max,
        non_cached_input_tokens_sum=non_cached_input_tokens_sum,
        context_tokens=context_tokens,
        sample_count=sample_count,
        compaction_count=compaction_count,
        elapsed_seconds=elapsed_seconds,
    )


@dataclass(frozen=True)
class OpaqueCompaction:
    payload: str = field(repr=False)
    protocol: Literal["responses", "messages"] = "responses"

    def __post_init__(self) -> None:
        _require_string(
            self.payload,
            "payload",
            allow_empty=False,
        )
        if not isinstance(self.protocol, str):
            raise TypeError("protocol must be a string")
        if self.protocol not in {"responses", "messages"}:
            raise ValueError(
                "protocol must be 'responses' or 'messages'"
            )

    @classmethod
    def from_responses(cls, encrypted_content: str) -> "OpaqueCompaction":
        return cls(payload=encrypted_content, protocol="responses")

    @classmethod
    def from_messages(cls, content: str) -> "OpaqueCompaction":
        return cls(payload=content, protocol="messages")


@dataclass(frozen=True)
class ContextPrefix:
    """Establish a new model-visible prefix at this point in the log.

    Earlier effective items are replaced by ``prefix_items``. Items appended
    after this interaction item follow that prefix. The underlying interaction
    log remains append-only.
    """

    prefix_items: Tuple["InteractionItem", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefix_items", tuple(self.prefix_items))


InteractionItem = Union[
    Init,
    Instructions,
    Tools,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    UserToolCall,
    UserToolResult,
    ModelSampleBoundary,
    SampleMetadata,
    CompactionMetadata,
    ModelFailure,
    TurnSummary,
    UserInteractionBoundary,
    OpaqueCompaction,
    ContextPrefix,
]

INTERACTION_ITEM_TYPES = (
    Init,
    Instructions,
    Tools,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    UserToolCall,
    UserToolResult,
    ModelSampleBoundary,
    SampleMetadata,
    CompactionMetadata,
    ModelFailure,
    TurnSummary,
    UserInteractionBoundary,
    OpaqueCompaction,
    ContextPrefix,
)


def is_interaction_item(value: object) -> bool:
    return isinstance(value, INTERACTION_ITEM_TYPES)


__all__ = [
    "ContextPrefix",
    "ContentPart",
    "MediaPart",
    "Instructions",
    "InteractionItem",
    "Message",
    "ModelSampleBoundary",
    "ModelFailure",
    "OpaqueCompaction",
    "Reasoning",
    "Init",
    "TextPart",
    "ToolCall",
    "ToolResult",
    "Tools",
    "UserToolCall",
    "UserToolResult",
    "SampleMetadata",
    "CompactionMetadata",
    "TurnSummary",
    "UserInteractionBoundary",
    "is_interaction_item",
    "summarize_turn_usage",
]
