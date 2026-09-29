from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import wraps
from time import perf_counter
from typing import List
from typing import Optional
from typing import Protocol
from typing import TYPE_CHECKING
from typing import Tuple

from .context import ContextValidationError
from .context import InteractionContext
from .items import CompactionMetadata
from .items import ContextPrefix
from .items import InteractionItem
from .items import Instructions
from .items import Message
from .items import SampleMetadata
from .model import Model
from .model import ModelContextWindowError
from .model import ModelSample
from .model import SampleParams
from .model import TokenUsage

if TYPE_CHECKING:
    from .environment import ToolSpec
    from .display import DisplayItem


DEFAULT_COMPACTION_MAX_OUTPUT_TOKENS = 2_000
# TODO: Add a nullable ``compaction_max_output_tokens`` runtime-config key.
# ``None`` should inherit the interaction's resolved ``max_output_tokens``
# instead of selecting this independent prompt-compaction policy default.

DEFAULT_COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff "
    "summary for another LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly "
    "continue the work."
)

DEFAULT_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a "
    "summary of its thinking process. You also have access to the state of "
    "the tools that were used by that language model. Use this to build on "
    "the work that has already been done and avoid duplicating work. Here is "
    "the summary produced by the other language model, use the information "
    "in this summary to assist with your own analysis:"
)


class CompactionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompactionResult:
    items: Tuple[InteractionItem, ...]
    usage: TokenUsage = field(default_factory=TokenUsage)
    protocol: str = "unspecified"
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)
    provider_response_id: Optional[str] = field(default=None, repr=False)
    elapsed_seconds: Optional[float] = None
    request_attempts: int = 1
    recovery: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if len(items) != 1 or not isinstance(items[0], ContextPrefix):
            raise CompactionError(
                "compaction result must contain exactly one ContextPrefix"
            )
        try:
            InteractionContext(items)
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc
        metadata = self._metadata()
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "elapsed_seconds", metadata.elapsed_seconds)
        object.__setattr__(self, "recovery", metadata.recovery)

    def context_items(self) -> Tuple[InteractionItem, ...]:
        """Return the installed prefix followed by durable operation metadata."""
        return (*self.items, self._metadata())

    def _metadata(self) -> CompactionMetadata:
        return CompactionMetadata(
            usage=self.usage,
            protocol=self.protocol,
            provider_session_id=self.provider_session_id,
            provider_turn_id=self.provider_turn_id,
            provider_turn_state=self.provider_turn_state,
            provider_response_id=self.provider_response_id,
            elapsed_seconds=self.elapsed_seconds,
            request_attempts=self.request_attempts,
            recovery=self.recovery,
        )

    def display_items(self) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(self.context_items())


# TODO: Pass the turn's SampleParams to compactors. Compaction requests now
# inherit only the model binding's defaults: a per-call ``extra`` never reaches
# them, and a Messages prompt summary inherits catalog thinking/effort extras
# and server compaction alongside its own temperature=0 and output budget.
class Compactor(Protocol):
    def compact(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
    ) -> CompactionResult:
        ...


def uses_host_auto_compaction(model: Model) -> bool:
    """Ownership is independent of numeric limits; custom models default to host."""
    return getattr(model, "auto_compaction_owner", "host") == "host"


def should_auto_compact(
    context: InteractionContext,
    threshold_tokens: int,
) -> bool:
    """Return whether the latest uncompacted sample reached a threshold."""
    if not isinstance(context, InteractionContext):
        raise TypeError("context must be InteractionContext")
    if (
        isinstance(threshold_tokens, bool)
        or not isinstance(threshold_tokens, int)
        or threshold_tokens <= 0
    ):
        raise ValueError("threshold_tokens must be a positive integer")

    # A prefix/metadata newer than the last sample means that usage belongs to
    # the pre-compaction window. Do not immediately compact the new prefix again
    # before it has produced a fresh provider usage measurement.
    for item in reversed(context.items):
        if isinstance(item, (CompactionMetadata, ContextPrefix)):
            return False
        if isinstance(item, SampleMetadata):
            return item.usage.total_tokens >= threshold_tokens
    return False


def _timed_compact(
    method: Callable[..., CompactionResult],
) -> Callable[..., CompactionResult]:
    """Measure a complete successful compactor call inside its worker."""
    @wraps(method)
    def measured(*args, **kwargs) -> CompactionResult:
        started = perf_counter()
        result = method(*args, **kwargs)
        if not isinstance(result, CompactionResult):
            raise TypeError(
                "compactor must return CompactionResult, got "
                f"{type(result).__name__}"
            )
        return replace(
            result,
            elapsed_seconds=perf_counter() - started,
        )

    return measured


def create_default_compactor(model: Model) -> Compactor:
    """Select the interaction default without probing a provider at runtime.

    The built-in ChatGPT/Codex Responses route advertises remote compaction and
    therefore uses an opaque server checkpoint. Other adapters use the portable
    prompt summarizer. A Codex-compatible third-party endpoint is not assumed to
    implement private remote-compaction controls merely because it speaks the
    Responses wire format.
    """
    # Keep the generic compaction module independent of the Responses adapter at
    # import time. ``responses`` itself imports the result and retention helpers
    # below for its concrete compactor.
    from .responses import CodexResponsesModel
    from .responses import ResponsesOpaqueCompactor

    if isinstance(model, CodexResponsesModel):
        if model.supports_remote_compaction:
            return ResponsesOpaqueCompactor(model)
        # Responses intentionally supports only max_output_tokens from the
        # generic SampleParams surface. Do not give its local fallback the
        # prompt compactor's temperature=0 default.
        return PromptSummarizingCompactor(
            model,
            sample_params=SampleParams(
                max_output_tokens=DEFAULT_COMPACTION_MAX_OUTPUT_TOKENS,
            ),
        )
    return PromptSummarizingCompactor(model)


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return "(tokens truncated)"
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()} ...(tokens truncated)"


def _leading_instruction_prefix(
    items: Sequence[InteractionItem],
) -> Tuple[InteractionItem, ...]:
    effective = None
    for _item in items:
        if isinstance(_item, Instructions):
            effective = _item
    prefix = []
    if effective is not None:
        prefix.append(effective)
    for item in items:
        if isinstance(item, Instructions):
            continue
        if isinstance(item, Message) and item.role in {"system", "developer"}:
            prefix.append(item)
            continue
        break
    return tuple(prefix)


def _select_retained_user_messages(
    messages: Sequence[Message],
    max_tokens: int,
) -> Tuple[Message, ...]:
    selected_reversed = []
    remaining = max(0, max_tokens)
    for message in reversed(messages):
        if remaining == 0:
            break
        tokens = _approx_token_count(message.content_text)
        if tokens <= remaining:
            selected_reversed.append(message)
            remaining -= tokens
            continue
        # Truncation drops any non-text parts: a partial image/file is not
        # meaningful and would still consume provider budget.
        selected_reversed.append(
            Message(
                role="user",
                content=_truncate_text_to_tokens(message.content_text, remaining),
            )
        )
        break
    selected_reversed.reverse()
    return tuple(selected_reversed)


def _drop_oldest_non_instruction_item(
    items: List[InteractionItem],
    compaction_prompt: Message,
) -> bool:
    index = 0
    while index < len(items) and (
        isinstance(items[index], Instructions)
        or (
            isinstance(items[index], Message)
            and items[index].role in {"system", "developer"}
        )
    ):
        index += 1
    if index >= len(items):
        return False

    del items[index]
    while True:
        try:
            InteractionContext((*items, compaction_prompt))
            return True
        except ContextValidationError:
            if index >= len(items):
                return False
            del items[index]


class PromptSummarizingCompactor:
    def __init__(
        self,
        model: Model,
        *,
        prompt: str = DEFAULT_COMPACTION_PROMPT,
        summary_prefix: str = DEFAULT_SUMMARY_PREFIX,
        retained_user_message_tokens: int = 20_000,
        sample_params: Optional[SampleParams] = None,
        retain_user_message: Optional[Callable[[Message], bool]] = None,
    ) -> None:
        if not hasattr(model, "sample") or not callable(model.sample):
            raise TypeError("model must provide sample(...)")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not isinstance(summary_prefix, str) or not summary_prefix.strip():
            raise ValueError("summary_prefix must not be empty")
        if (
            isinstance(retained_user_message_tokens, bool)
            or not isinstance(retained_user_message_tokens, int)
            or retained_user_message_tokens < 0
        ):
            raise ValueError(
                "retained_user_message_tokens must be a nonnegative integer"
            )
        if sample_params is not None and not isinstance(sample_params, SampleParams):
            raise TypeError("sample_params must be SampleParams or None")
        if retain_user_message is not None and not callable(retain_user_message):
            raise TypeError("retain_user_message must be callable or None")

        self._model = model
        self._prompt = prompt
        self._summary_prefix = summary_prefix
        self._retained_user_message_tokens = retained_user_message_tokens
        self._sample_params = sample_params or SampleParams(
            temperature=0.0,
            max_output_tokens=DEFAULT_COMPACTION_MAX_OUTPUT_TOKENS,
        )
        self._retain_user_message = retain_user_message

    def _is_retained_user_message(self, message: Message) -> bool:
        if message.role != "user":
            return False
        if message.content_text.startswith(f"{self._summary_prefix}\n"):
            return False
        if self._retain_user_message is not None:
            return bool(self._retain_user_message(message))
        return True

    @_timed_compact
    def compact(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
    ) -> CompactionResult:
        del tools
        if not isinstance(context, InteractionContext):
            raise TypeError("context must be InteractionContext")
        try:
            context.assert_model_ready()
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc

        active_items = list(context.model_items())
        instruction_prefix = _leading_instruction_prefix(active_items)
        user_messages = tuple(
            item
            for item in active_items
            if isinstance(item, Message)
            and self._is_retained_user_message(item)
        )
        compaction_prompt = Message(role="user", content=self._prompt)
        request_items = list(active_items)
        request_attempts = 0
        recovery = []

        while True:
            try:
                temporary_context = InteractionContext(
                    (*request_items, compaction_prompt)
                )
                sample = self._model.sample(
                    temporary_context,
                    tools=(),
                    sample_params=self._sample_params,
                )
                if not isinstance(sample, ModelSample):
                    raise CompactionError(
                        f"model returned {type(sample).__name__}, expected ModelSample"
                    )
                request_attempts += sample.request_attempts
                recovery.extend(sample.recovery)
                break
            except ModelContextWindowError as exc:
                request_attempts += (
                    exc.failure.attempt_count
                    if exc.failure is not None
                    else 1
                )
                if exc.failure is not None:
                    recovery.extend(exc.failure.recovery)
                recovery.append("context_window_trim")
                if not _drop_oldest_non_instruction_item(
                    request_items,
                    compaction_prompt,
                ):
                    raise CompactionError(
                        "compaction request exceeds the model context window "
                        "after all removable items were discarded"
                    ) from exc

        if sample.tool_calls:
            raise CompactionError(
                "compaction model response must not contain tool calls"
            )
        summary_text = sample.last_assistant_text
        if summary_text is None or not summary_text.strip():
            raise CompactionError(
                "compaction model response has no non-empty assistant summary"
            )

        retained_users = _select_retained_user_messages(
            user_messages,
            self._retained_user_message_tokens,
        )
        summary_message = Message(
            role="user",
            content=f"{self._summary_prefix}\n{summary_text.strip()}",
        )
        prefix_items = (
            *instruction_prefix,
            *retained_users,
            summary_message,
        )
        checkpoint = ContextPrefix(
            prefix_items=tuple(prefix_items),
        )
        return CompactionResult(
            items=(checkpoint,),
            usage=sample.usage,
            protocol="prompt_summarization",
            provider_session_id=sample.provider_session_id,
            provider_turn_id=sample.provider_turn_id,
            provider_turn_state=sample.provider_turn_state,
            request_attempts=request_attempts,
            recovery=tuple(recovery),
        )


__all__ = [
    "CompactionError",
    "CompactionResult",
    "Compactor",
    "DEFAULT_COMPACTION_MAX_OUTPUT_TOKENS",
    "DEFAULT_COMPACTION_PROMPT",
    "DEFAULT_SUMMARY_PREFIX",
    "PromptSummarizingCompactor",
    "create_default_compactor",
    "should_auto_compact",
    "uses_host_auto_compaction",
]
