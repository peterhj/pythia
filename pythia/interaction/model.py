from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import wraps
from time import perf_counter
from typing import Callable
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Mapping

from .context import InteractionContext
from .items import InteractionItem
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import SampleMetadata
from .items import _validate_elapsed_seconds
from .usage import TokenUsage
from .model_catalog import freeze_extra_sample_params
from .model_catalog import thaw_json

if TYPE_CHECKING:
    from .display import DisplayItem
    from .environment import ToolSpec
    from .model_catalog import ModelBinding


class ModelError(RuntimeError):
    """Base model failure with optional safe diagnostics and completed output."""

    def __init__(
        self,
        *args,
        failure: Optional[ModelFailure] = None,
        completed_items: Sequence[InteractionItem] = (),
    ) -> None:
        super().__init__(*args)
        if failure is not None and not isinstance(failure, ModelFailure):
            raise TypeError("failure must be ModelFailure or None")
        items = tuple(completed_items)
        for index, item in enumerate(items):
            if not isinstance(item, (Message, Reasoning, ToolCall, OpaqueCompaction)):
                raise TypeError(
                    "completed_items must contain model output items; "
                    f"item {index} is {type(item).__name__}"
                )
        self.failure = failure
        self.completed_items = items


class ModelConfigurationError(ModelError, ValueError):
    pass


class ModelTransportError(ModelError):
    pass


class ModelAuthenticationError(ModelTransportError):
    pass


class ModelTimeoutError(ModelTransportError):
    pass


class ModelResponseError(ModelError):
    pass


class ModelContextWindowError(ModelResponseError):
    pass


def _validate_optional_finite_number(
    value: object,
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric or None")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


@dataclass(frozen=True)
class SampleParams:
    """Per-call parameters of ``Model.sample()``.

    Every ``None`` field (and an empty ``stop``) inherits the model's default,
    with one precedence order: this call, then adapter settings such as a
    Messages endpoint's budget or compaction trigger, then the model binding
    and catalog. Adapters reject typed fields that their API cannot send.

    ``extra`` holds request-body extensions for this call. A mapping replaces
    the binding's ``extra_sample_params`` (``{}`` sends none); ``None``
    inherits them.
    """

    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Tuple[str, ...] = ()
    seed: Optional[int] = None
    # Host/provider context-management control, never a sampling wire field.
    enable_auto_compaction: Optional[bool] = None
    # Host context-management threshold override, never a sampling wire field.
    auto_compact_tokens: Optional[int] = None
    extra: Optional[Mapping] = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None:
            if (
                isinstance(self.max_output_tokens, bool)
                or not isinstance(self.max_output_tokens, int)
                or self.max_output_tokens <= 0
            ):
                raise ValueError(
                    "max_output_tokens must be a positive integer or None"
                )

        temperature = _validate_optional_finite_number(
            self.temperature,
            "temperature",
        )
        if temperature is not None and temperature < 0:
            raise ValueError("temperature must be nonnegative")
        object.__setattr__(self, "temperature", temperature)

        top_p = _validate_optional_finite_number(self.top_p, "top_p")
        if top_p is not None and not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        object.__setattr__(self, "top_p", top_p)

        stop = tuple(self.stop)
        for index, value in enumerate(stop):
            if not isinstance(value, str):
                raise TypeError(f"stop[{index}] must be a string")
        object.__setattr__(self, "stop", stop)

        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if self.enable_auto_compaction is not None and not isinstance(
            self.enable_auto_compaction,
            bool,
        ):
            raise TypeError(
                "enable_auto_compaction must be a bool or None"
            )
        if self.auto_compact_tokens is not None and (
            isinstance(self.auto_compact_tokens, bool)
            or not isinstance(self.auto_compact_tokens, int)
            or self.auto_compact_tokens <= 0
        ):
            raise ValueError(
                "auto_compact_tokens must be a positive integer or None"
            )
        if self.extra is not None:
            object.__setattr__(self, "extra", freeze_extra_sample_params(self.extra))


def _apply_extra_sample_params(
    payload: dict,
    binding: "ModelBinding",
    sample_params: Optional[SampleParams],
) -> None:
    """Add one request's body extensions without replacing adapter fields.

    A per-call ``extra`` map replaces the binding's map, even when empty;
    ``None`` params or ``extra`` use the binding's map. Per-call maps are
    validated without an API, so the names reserved by this binding's API are
    checked here before sending.
    """
    override = None if sample_params is None else sample_params.extra
    if override is None:
        params = binding.extra_sample_params
    else:
        try:
            params = freeze_extra_sample_params(override, binding.api)
        except ValueError as exc:
            raise ModelConfigurationError(str(exc)) from None
    extensions = thaw_json(params)
    # Typed and structural fields stay authoritative; never override silently.
    replaced = sorted(extensions.keys() & payload.keys())
    if replaced:
        raise ModelConfigurationError(
            "extra_sample_params cannot replace adapter-owned request fields: "
            + ", ".join(replaced)
        )
    payload.update(extensions)


@dataclass(frozen=True)
class ModelSample:
    items: Tuple[InteractionItem, ...]
    stop_reason: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)
    elapsed_seconds: Optional[float] = None
    request_attempts: int = 1
    recovery: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items:
            raise ModelResponseError("model sample must contain at least one item")
        for index, item in enumerate(items):
            if not isinstance(
                item,
                (Message, Reasoning, ToolCall, OpaqueCompaction),
            ):
                raise ModelResponseError(
                    "model sample items must be assistant messages, reasoning, "
                    "tool calls, or opaque compactions; "
                    f"item {index} is {type(item).__name__}"
                )
            if isinstance(item, Message) and item.role != "assistant":
                raise ModelResponseError(
                    f"model message at item {index} must have role 'assistant'"
                )
        object.__setattr__(self, "items", items)

        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise TypeError("stop_reason must be a string or None")
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
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"recovery[{index}] must be a non-empty string")
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
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            if "\r" in value or "\n" in value:
                raise ValueError(f"{field_name} must not contain newlines")

    @property
    def tool_calls(self) -> Tuple[ToolCall, ...]:
        return tuple(item for item in self.items if isinstance(item, ToolCall))

    def context_items(self) -> Tuple[InteractionItem, ...]:
        """Return this sample's output, sample metadata, and durable boundary."""
        return (
            *self.items,
            self._metadata(),
            ModelSampleBoundary(),
        )

    def _metadata(self) -> SampleMetadata:
        return SampleMetadata(
            usage=self.usage,
            provider_session_id=self.provider_session_id,
            provider_turn_id=self.provider_turn_id,
            provider_turn_state=self.provider_turn_state,
            elapsed_seconds=self.elapsed_seconds,
            request_attempts=self.request_attempts,
            recovery=self.recovery,
        )

    def display_items(self) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(
            (*self.items, self._metadata()),
        )

    @property
    def assistant_messages(self) -> Tuple[Message, ...]:
        return tuple(item for item in self.items if isinstance(item, Message))

    @property
    def last_assistant_text(self) -> Optional[str]:
        for item in reversed(self.items):
            if isinstance(item, Message) and item.content_text:
                return item.content_text
        return None


def _timed_sample(method: Callable[..., ModelSample]) -> Callable[..., ModelSample]:
    """Measure a complete adapter call, including its response cleanup.

    Timing starts inside the calling worker, not while waiting for one. Errors
    propagate unchanged and do not produce a sample or fabricated usage data.
    """
    @wraps(method)
    def measured(*args, **kwargs) -> ModelSample:
        started = perf_counter()
        sample = method(*args, **kwargs)
        return replace(sample, elapsed_seconds=perf_counter() - started)

    return measured


class Model(Protocol):
    def sample(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        sample_params: Optional[SampleParams] = None,
    ) -> ModelSample:
        ...


__all__ = [
    "Model",
    "ModelAuthenticationError",
    "ModelConfigurationError",
    "ModelContextWindowError",
    "ModelError",
    "ModelResponseError",
    "ModelSample",
    "ModelTimeoutError",
    "ModelTransportError",
    "SampleParams",
    "TokenUsage",
]
