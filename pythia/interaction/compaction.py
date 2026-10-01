"""Context compaction: the compactor protocol and pi-style summaries.

``PiCompactor`` ports the coding agent's compaction from pi
(``packages/coding-agent/src/core/compaction/``). It keeps the most recent
context verbatim except signed thinking, summarizes the older span from a
plain-text transcript under a dedicated summarizer system prompt, updates the
previous summary iteratively, and summarizes the start of a split turn
separately. The remote Responses compactor lives with its adapter
(``responses.ResponsesOpaqueCompactor``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Iterable
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
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import SampleMetadata
from .items import TextPart
from .items import ToolCall
from .items import ToolResult
from .items import TurnSummary
from .items import UserInteractionBoundary
from .model import Model
from .model import ModelContextWindowError
from .model import ModelSample
from .model import SampleParams
from .model import TokenUsage

if TYPE_CHECKING:
    from .display import DisplayItem
    from .environment import ToolSpec


# ``pi`` summarizes on the host. ``provider`` selects the provider's own
# compaction: Codex remote compaction, or Anthropic server-side compaction.
COMPACTION_MODES = ("pi", "provider")
# Pi's ``keepRecentTokens``, ``TOOL_RESULT_MAX_CHARS``, and
# ``ESTIMATED_IMAGE_CHARS``.
DEFAULT_KEEP_RECENT_TOKENS = 20_000
TOOL_RESULT_MAX_CHARS = 2_000
ESTIMATED_IMAGE_CHARS = 4_800

# Pi's prompt texts, verbatim.
SUMMARIZATION_SYSTEM_PROMPT = """\
You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

SUMMARIZATION_PROMPT = """\
The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """\
The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """\
The messages above are earlier context from an ongoing conversation. Later messages are stored separately and do not need to be reconstructed.

Create a concise checkpoint of the user's request and the progress shown above. This checkpoint will be placed before the later messages so the conversation can continue with the necessary context.

## Original Request
[What did the user ask for?]

## Progress So Far
- [Key decisions and work completed in these messages]

## Context Needed to Continue
- [Information from these messages needed to understand the later work]

Only summarize information explicitly present above. Do not infer or recreate later messages."""

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the "
    "following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

# The removed prompt summarizer's prefix. Its summaries remain in older saves,
# where they become the previous summary at the next compaction.
_LEGACY_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a "
    "summary of its thinking process. You also have access to the state of "
    "the tools that were used by that language model. Use this to build on "
    "the work that has already been done and avoid duplicating work. Here is "
    "the summary produced by the other language model, use the information "
    "in this summary to assist with your own analysis:"
)

_SPLIT_TURN_SEPARATOR = "\n\n---\n\n**Turn Context (split turn):**\n\n"
_NO_PRIOR_HISTORY = "No prior history."
_HISTORY_PART = "the history"
_TURN_PREFIX_PART = "the earlier part of the current turn"
# A truncated, refused, or paused summary must never become a checkpoint.
_INCOMPLETE_STOP_REASONS = frozenset({
    "max_tokens",
    "refusal",
    "compaction",
    "model_context_window_exceeded",
})
_MODEL_VISIBLE_TYPES = (
    Instructions,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    OpaqueCompaction,
)


class CompactionError(RuntimeError):
    pass


class NothingToCompact(CompactionError):
    """The context has nothing to summarize outside the kept recent tail.

    This is not a failure: automatic compaction skips it and samples.
    """


class CompactionContextWindowError(CompactionError):
    """A summary request exceeded the model's context window."""


def _require_keep_recent_tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("keep_recent_tokens must be a nonnegative integer")
    return value


def _require_summary_budget(value: object) -> Optional[int]:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError("max_output_tokens must be a positive integer or None")
    return value


@dataclass(frozen=True)
class CompactionSettings:
    """Compactor selection and pi options, from one config snapshot.

    ``max_output_tokens=None`` gives each summary request the turn's budget.
    """

    mode: str = "pi"
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    max_output_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in COMPACTION_MODES:
            raise ValueError("mode must be 'pi' or 'provider'")
        _require_keep_recent_tokens(self.keep_recent_tokens)
        _require_summary_budget(self.max_output_tokens)


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


class Compactor(Protocol):
    """Build a replacement prefix for a model-ready context.

    ``sample_params`` are the turn's params; compactors derive their own
    request params from them. ``instructions`` is optional focus text from
    ``/compact [focus]``.
    """

    def compact(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        sample_params: Optional[SampleParams] = None,
        instructions: Optional[str] = None,
    ) -> CompactionResult:
        ...


def uses_host_auto_compaction(model: Model) -> bool:
    """Ownership is independent of numeric limits; custom models default to host."""
    return getattr(model, "auto_compaction_owner", "host") == "host"


def _estimate_text_tokens(chars: int) -> int:
    return (chars + 3) // 4


def _reasoning_text(item: Reasoning) -> str:
    return item.content or "\n".join(item.summary)


def _item_chars(item: object) -> int:
    if isinstance(item, Message):
        if isinstance(item.content, str):
            return len(item.content)
        return sum(
            len(part.text) if isinstance(part, TextPart)
            else ESTIMATED_IMAGE_CHARS
            for part in item.content
        )
    if isinstance(item, Reasoning):
        return len(_reasoning_text(item))
    if isinstance(item, ToolCall):
        return len(item.name) + len(item.arguments_json)
    if isinstance(item, ToolResult):
        return len(item.output)
    if isinstance(item, Instructions):
        return len(item.text)
    if isinstance(item, OpaqueCompaction):
        return len(item.payload)
    return 0


def estimate_item_tokens(item: InteractionItem) -> int:
    """Estimate one item as characters / 4, rounded up (pi's ``estimateTokens``).

    Media parts count as ``ESTIMATED_IMAGE_CHARS`` characters each. Boundaries,
    metadata, and user-tool records count as 0.
    """
    return _estimate_text_tokens(_item_chars(item))


def estimate_context_tokens(context: InteractionContext) -> int:
    """Estimate the effective context, anchored on provider-reported usage.

    This follows pi's ``estimateContextTokens``. The anchor is the latest
    top-level ``SampleMetadata`` after the latest ``ContextPrefix`` that
    reports usage; the estimate is its ``total_tokens`` plus the estimates of
    the raw items after it. Without an anchor, for example just after a
    compaction or on a server that reports no usage, it estimates all of
    ``context.model_items()``. Character estimates undercount contexts whose
    reasoning is stored only as signatures, hence the anchor.
    """
    if not isinstance(context, InteractionContext):
        raise TypeError("context must be InteractionContext")
    items = context.items
    start = 0
    for index in range(len(items) - 1, -1, -1):
        if isinstance(items[index], ContextPrefix):
            start = index + 1
            break
    for index in range(len(items) - 1, start - 1, -1):
        item = items[index]
        if isinstance(item, SampleMetadata) and item.usage.total_tokens > 0:
            return item.usage.total_tokens + sum(
                estimate_item_tokens(later) for later in items[index + 1:]
            )
    return sum(estimate_item_tokens(item) for item in context.model_items())


def should_auto_compact(
    context: InteractionContext,
    threshold_tokens: int,
) -> bool:
    """Return whether the estimated context reached a threshold.

    A context with no model-visible item after its latest ``ContextPrefix``
    was just compacted, and is not compacted again.
    """
    if not isinstance(context, InteractionContext):
        raise TypeError("context must be InteractionContext")
    if (
        isinstance(threshold_tokens, bool)
        or not isinstance(threshold_tokens, int)
        or threshold_tokens <= 0
    ):
        raise ValueError("threshold_tokens must be a positive integer")
    for item in reversed(context.items):
        if isinstance(item, ContextPrefix):
            return False
        if isinstance(item, _MODEL_VISIBLE_TYPES):
            break
    else:
        return False
    return estimate_context_tokens(context) >= threshold_tokens


def auto_compaction_due(model: Model, context: InteractionContext, snapshot) -> bool:
    """Return whether a frontend should compact before its next sample.

    ``snapshot`` is the turn's ``InteractionConfigSnapshot``. A failed
    automatic compaction fails the turn; ``NothingToCompact`` is not a
    failure.
    """
    # TODO: Pi reports a failed threshold compaction and keeps sampling
    # (``_runAutoCompaction`` in pi's ``agent-session.ts``); here the failure
    # fails the turn.
    threshold = snapshot.auto_compact_tokens
    return bool(
        snapshot.enable_auto_compaction
        and uses_host_auto_compaction(model)
        and threshold is not None
        and should_auto_compact(context, threshold)
    )


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


def create_default_compactor(
    model: Model,
    settings: CompactionSettings = CompactionSettings(),
) -> Compactor:
    """Select a compactor for one mode without probing a provider at runtime.

    ``provider`` mode uses remote opaque compaction when the model advertises
    it: the built-in ChatGPT/Codex Responses route. Everything else uses
    ``PiCompactor``, including manual compaction for Messages in ``provider``
    mode, since Anthropic has no on-demand server compaction.
    """
    if not isinstance(settings, CompactionSettings):
        raise TypeError("settings must be CompactionSettings")
    if (
        settings.mode == "provider"
        and getattr(model, "supports_remote_compaction", False) is True
    ):
        # ``responses`` imports this module for its concrete compactor.
        from .responses import ResponsesOpaqueCompactor

        return ResponsesOpaqueCompactor(model)
    return PiCompactor(
        model,
        keep_recent_tokens=settings.keep_recent_tokens,
        max_output_tokens=settings.max_output_tokens,
    )


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
    return _split_instruction_prefix(items)[0]


def _split_instruction_prefix(
    items: Sequence[InteractionItem],
) -> Tuple[Tuple[InteractionItem, ...], Tuple[InteractionItem, ...]]:
    """Split the effective instructions and leading system/developer messages."""
    effective = None
    for item in items:
        if isinstance(item, Instructions):
            effective = item
    prefix: List[InteractionItem] = []
    if effective is not None:
        prefix.append(effective)
    rest: List[InteractionItem] = []
    leading = True
    for item in items:
        if isinstance(item, Instructions):
            continue
        if (
            leading
            and isinstance(item, Message)
            and item.role in {"system", "developer"}
        ):
            prefix.append(item)
            continue
        leading = False
        rest.append(item)
    return tuple(prefix), tuple(rest)


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


def _compaction_summary_text(item: object) -> Optional[str]:
    """Return a summary message's text, for new-style and old-style summaries."""
    if not isinstance(item, Message) or item.role != "user":
        return None
    text = item.content
    if not isinstance(text, str):
        return None
    if (
        len(text) >= len(COMPACTION_SUMMARY_PREFIX) + len(COMPACTION_SUMMARY_SUFFIX)
        and text.startswith(COMPACTION_SUMMARY_PREFIX)
        and text.endswith(COMPACTION_SUMMARY_SUFFIX)
    ):
        return text[len(COMPACTION_SUMMARY_PREFIX):len(text) - len(COMPACTION_SUMMARY_SUFFIX)]
    legacy_prefix = f"{_LEGACY_SUMMARY_PREFIX}\n"
    if text.startswith(legacy_prefix):
        return text[len(legacy_prefix):]
    return None


def is_compaction_summary(message: object) -> bool:
    """Whether a message is a compaction summary, in either style.

    New-style summaries are user messages wrapped in
    ``COMPACTION_SUMMARY_PREFIX`` and ``COMPACTION_SUMMARY_SUFFIX``. Old-style
    summaries, from the removed prompt summarizer, remain in older saves.
    """
    return _compaction_summary_text(message) is not None


def _is_user_message(item: object) -> bool:
    return isinstance(item, Message) and item.role == "user"


def _is_output_item(item: object) -> bool:
    """A sample's output: reasoning, assistant text, a tool call, or a checkpoint."""
    if isinstance(item, (Reasoning, ToolCall, OpaqueCompaction)):
        return True
    return isinstance(item, Message) and item.role == "assistant"


def _is_signed_thinking(item: object) -> bool:
    """Messages thinking, which is valid only after its original history."""
    return isinstance(item, Reasoning) and item.content_signature is not None


def _has_content(items: Iterable[InteractionItem]) -> bool:
    return any(isinstance(item, _MODEL_VISIBLE_TYPES) for item in items)


def _message_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        part.text if isinstance(part, TextPart) else "[image]"
        for part in message.content
    )


def _format_tool_call(call: ToolCall) -> str:
    try:
        arguments = json.loads(call.arguments_json)
    except (ValueError, RecursionError):
        arguments = None
    if isinstance(arguments, dict):
        rendered = ", ".join(
            f"{key}="
            + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            for key, value in arguments.items()
        )
    else:
        rendered = call.arguments_json
    return f"{call.name}({rendered})"


def _truncate_for_summary(text: str) -> str:
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    omitted = len(text) - TOOL_RESULT_MAX_CHARS
    return (
        f"{text[:TOOL_RESULT_MAX_CHARS]}\n\n"
        f"[... {omitted} more characters truncated]"
    )


_ROLE_LABELS = {"user": "User", "system": "System", "developer": "Developer"}


def _serialize_transcript(items: Sequence[InteractionItem]) -> str:
    """Render items as text, following pi's ``serializeConversation``.

    Each sample's output is grouped as thinking, then text, then tool calls,
    so the summarizer reads a transcript rather than a conversation to
    continue. Tool results are truncated; boundaries and metadata are skipped.
    """
    entries: List[str] = []
    thinking: List[str] = []
    texts: List[str] = []
    calls: List[str] = []

    def flush_sample() -> None:
        if thinking:
            entries.append("[Assistant thinking]: " + "\n".join(thinking))
        if texts:
            entries.append("[Assistant]: " + "\n".join(texts))
        if calls:
            entries.append("[Assistant tool calls]: " + "; ".join(calls))
        thinking.clear()
        texts.clear()
        calls.clear()

    for item in items:
        if isinstance(item, Reasoning):
            text = _reasoning_text(item)
            if text.strip():
                thinking.append(text)
        elif isinstance(item, Message) and item.role == "assistant":
            if item.content_text.strip():
                texts.append(item.content_text)
        elif isinstance(item, ToolCall):
            calls.append(_format_tool_call(item))
        elif isinstance(item, OpaqueCompaction):
            # A Messages compaction block is readable summary text. Responses
            # checkpoints are encrypted and never reach the transcript.
            if item.protocol == "messages" and item.payload.strip():
                texts.append(item.payload)
        elif isinstance(item, Message):
            flush_sample()
            text = _message_text(item)
            if text.strip():
                label = _ROLE_LABELS.get(item.role, item.role.strip().title())
                entries.append(f"[{label}]: {text}")
        elif isinstance(item, ToolResult):
            flush_sample()
            if item.output:
                entries.append(
                    f"[Tool result]: {_truncate_for_summary(item.output)}"
                )
        elif isinstance(item, (ModelSampleBoundary, UserInteractionBoundary)):
            flush_sample()
    flush_sample()
    return "\n\n".join(entries)


def _focus_suffix(focus: Optional[str]) -> str:
    return "" if focus is None else f"\n\nAdditional focus: {focus}"


def _history_prompt(
    history: Sequence[InteractionItem],
    previous_summary: Optional[str],
    focus: Optional[str],
) -> str:
    prompt = f"<conversation>\n{_serialize_transcript(history)}\n</conversation>\n\n"
    if previous_summary is None:
        instructions = SUMMARIZATION_PROMPT
    else:
        prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        instructions = UPDATE_SUMMARIZATION_PROMPT
    return prompt + instructions + _focus_suffix(focus)


def _turn_prefix_prompt(
    turn_prefix: Sequence[InteractionItem],
    focus: Optional[str],
) -> str:
    return (
        f"# Conversation\n{_serialize_transcript(turn_prefix)}\n\n"
        f"# Instructions\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
        + _focus_suffix(focus)
    )


@dataclass(frozen=True)
class _CompactionPlan:
    instruction_prefix: Tuple[InteractionItem, ...]
    checkpoints: Tuple[InteractionItem, ...]
    previous_summary: Optional[str]
    history: Tuple[InteractionItem, ...]
    # Non-empty only when the cut splits a turn.
    turn_prefix: Tuple[InteractionItem, ...]
    tail: Tuple[InteractionItem, ...]


def _extract_summary_sources(
    rest: Tuple[InteractionItem, ...],
) -> Tuple[Optional[str], Tuple[InteractionItem, ...], Tuple[InteractionItem, ...]]:
    """Return the previous summary, carried checkpoints, and the span."""
    for index in range(len(rest) - 1, -1, -1):
        item = rest[index]
        if isinstance(item, OpaqueCompaction) and item.protocol == "messages":
            # Anthropic ignores everything before its latest compaction block
            # except the instructions, and so does the summary.
            rest = rest[index:]
            break
    source = None
    previous_summary = None
    for index, item in enumerate(rest):
        if isinstance(item, OpaqueCompaction) and item.protocol == "messages":
            source, previous_summary = index, item.payload
            continue
        text = _compaction_summary_text(item)
        if text is not None:
            source, previous_summary = index, text
    checkpoints: List[InteractionItem] = []
    span: List[InteractionItem] = []
    for index, item in enumerate(rest):
        if index == source:
            continue
        if isinstance(item, OpaqueCompaction) and item.protocol == "responses":
            # Encrypted: carried into the new prefix verbatim.
            checkpoints.append(item)
            continue
        span.append(item)
    if previous_summary is not None and not previous_summary.strip():
        previous_summary = None
    return previous_summary, tuple(checkpoints), tuple(span)


def _cut_points(span: Sequence[InteractionItem]) -> List[int]:
    """Indices where the kept tail may start (pi's ``findValidCutPoints``).

    These are the first of each run of consecutive user messages and the
    first output item of each sample. Tool results are never cut points.
    """
    points: List[int] = []
    after_user = False
    in_sample = False
    for index, item in enumerate(span):
        if isinstance(item, ModelSampleBoundary):
            in_sample = False
            continue
        if not isinstance(item, _MODEL_VISIBLE_TYPES):
            continue
        if _is_user_message(item):
            if not after_user:
                points.append(index)
            after_user = True
            in_sample = False
            continue
        after_user = False
        if _is_output_item(item):
            if not in_sample:
                points.append(index)
            in_sample = True
        else:
            in_sample = False
    return points


def _find_cut(
    span: Sequence[InteractionItem],
    points: Sequence[int],
    keep_recent_tokens: int,
) -> int:
    """Follow pi's ``findProjectedCutPoint``, except that 0 keeps nothing."""
    if keep_recent_tokens == 0:
        return len(span)
    total = 0
    for index in range(len(span) - 1, -1, -1):
        tokens = estimate_item_tokens(span[index])
        if not tokens:
            continue
        total += tokens
        if total >= keep_recent_tokens:
            for point in points:
                if point >= index:
                    return point
            # Trailing tool results alone exceed the budget: keep their
            # sample's first output item rather than cutting between them.
            if points:
                return points[-1]
            raise NothingToCompact("nothing precedes the recent tail")
    raise NothingToCompact("the context fits in compaction_keep_recent_tokens")


def _plan_compaction(
    items: Sequence[InteractionItem],
    keep_recent_tokens: int,
) -> _CompactionPlan:
    instruction_prefix, rest = _split_instruction_prefix(items)
    previous_summary, checkpoints, span = _extract_summary_sources(rest)
    if not _has_content(span):
        raise NothingToCompact("there is nothing new to summarize")
    points = _cut_points(span)
    cut = _find_cut(span, points, keep_recent_tokens)
    turn_start = None
    if cut < len(span) and _is_output_item(span[cut]):
        # The cut splits a turn when a user message starts it earlier in the
        # span: that run of user messages starts the turn prefix.
        for point in reversed(points):
            if point < cut and _is_user_message(span[point]):
                turn_start = point
                break
    if turn_start is None:
        history, turn_prefix = span[:cut], ()
    else:
        history, turn_prefix = span[:turn_start], span[turn_start:cut]
    if not _has_content((*history, *turn_prefix)):
        raise NothingToCompact("nothing precedes the recent tail")
    return _CompactionPlan(
        instruction_prefix=instruction_prefix,
        checkpoints=checkpoints,
        previous_summary=previous_summary,
        history=tuple(history),
        turn_prefix=tuple(turn_prefix),
        tail=tuple(span[cut:]),
    )


def _context_window_message(
    part: str,
    items: Sequence[InteractionItem],
    prompt: str,
) -> str:
    count = sum(1 for item in items if isinstance(item, _MODEL_VISIBLE_TYPES))
    noun = "item" if count == 1 else "items"
    tokens = _estimate_text_tokens(len(SUMMARIZATION_SYSTEM_PROMPT) + len(prompt))
    return (
        f"summary request for {part} ({count} {noun}, ~{tokens:,} estimated "
        "tokens) exceeded the model's context window; pi compaction sends "
        "each part in one request and does not split it. Lower "
        "compaction_max_output_tokens, or raise compaction_keep_recent_tokens "
        "to summarize less."
    )


def _summary_text(sample: ModelSample, part: str) -> str:
    """Check a summary response, as pi's ``getSummarizationFailure`` does."""
    if sample.tool_calls:
        raise CompactionError(f"summary request for {part} returned tool calls")
    if sample.stop_reason in _INCOMPLETE_STOP_REASONS:
        raise CompactionError(
            f"summary request for {part} stopped early "
            f"(stop_reason {sample.stop_reason}); an incomplete summary is "
            "never installed"
        )
    # All text blocks, as pi's ``contentText``, not only the last one.
    text = "\n".join(
        message.content_text for message in sample.assistant_messages
    ).strip()
    if not text:
        raise CompactionError(f"summary request for {part} returned no text")
    return text


def _sum_usage(usages: Iterable[TokenUsage]) -> TokenUsage:
    input_tokens = output_tokens = total_tokens = cached_input_tokens = 0
    for usage in usages:
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        total_tokens += usage.total_tokens
        cached_input_tokens += usage.cached_input_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


class PiCompactor:
    """Pi's compaction: summarize older context and keep a recent tail.

    The new prefix is the instructions, any Responses checkpoints, the summary
    as a user message, and the kept tail verbatim, minus signed thinking.
    ``keep_recent_tokens`` is the estimated size of that tail; 0 keeps nothing
    (pi still keeps the last unit). ``max_output_tokens`` overrides the turn's
    budget for summary requests. ``tools`` are accepted for the protocol but
    never sent: the requests carry the transcript as text.
    """

    def __init__(
        self,
        model: Model,
        *,
        keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        if not hasattr(model, "sample") or not callable(model.sample):
            raise TypeError("model must provide sample(...)")
        self._model = model
        self.keep_recent_tokens = _require_keep_recent_tokens(keep_recent_tokens)
        self.max_output_tokens = _require_summary_budget(max_output_tokens)

    @_timed_compact
    def compact(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        sample_params: Optional[SampleParams] = None,
        instructions: Optional[str] = None,
    ) -> CompactionResult:
        del tools
        if not isinstance(context, InteractionContext):
            raise TypeError("context must be InteractionContext")
        if sample_params is not None and not isinstance(sample_params, SampleParams):
            raise TypeError("sample_params must be SampleParams or None")
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None")
        focus = None if instructions is None else instructions.strip() or None
        try:
            context.assert_model_ready()
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc

        plan = _plan_compaction(context.model_items(), self.keep_recent_tokens)
        params = self._summary_params(sample_params)
        samples: List[ModelSample] = []
        has_history = _has_content(plan.history)
        # TODO: Run the history and turn-prefix requests in parallel. They are
        # independent and merge in a fixed order, so wall time would be the
        # longer request rather than the sum (plan section 9.2). Build both
        # request contexts first (InteractionContext is not thread-safe), run
        # each on a two-worker ThreadPoolExecutor in its own
        # contextvars.copy_context() (--debug-trace tags need it), wait for
        # both, raise the first failure in fixed order, and combine accounting
        # in that order.
        history_text = plan.previous_summary or _NO_PRIOR_HISTORY
        if has_history:
            history_text, sample = self._summarize(
                context,
                _history_prompt(plan.history, plan.previous_summary, focus),
                params,
                part=_HISTORY_PART,
                items=plan.history,
            )
            samples.append(sample)
        summary = history_text
        if plan.turn_prefix:
            # Without a history request, focus text goes here rather than
            # being dropped (pi drops it).
            turn_text, sample = self._summarize(
                context,
                _turn_prefix_prompt(
                    plan.turn_prefix,
                    None if has_history else focus,
                ),
                params,
                part=_TURN_PREFIX_PART,
                items=plan.turn_prefix,
            )
            samples.append(sample)
            summary = f"{history_text}{_SPLIT_TURN_SEPARATOR}{turn_text}"

        # Boundaries stay: the Messages and Chat Completions encoders flush
        # assistant blocks at them. The other items are log-only, and
        # ModelFailure is not allowed in prefixes.
        # Signed thinking leaves too. Anthropic binds each thinking block to
        # the conversation that preceded it, which the summary replaces, so
        # the next request would fail with HTTP 400 ("Invalid `signature` in
        # `thinking` block. The block is bound to a different conversation.").
        # Responses reasoning (``encrypted_content``) stays.
        # TODO: Preserve thinking across compaction if Anthropic adds a way;
        # the binding has no keep option today. Its
        # ``thinking-binding-controls-2026-08-01`` beta header, together with
        # ``thinking.block_binding.prefix_mismatch_behavior: "drop_block"``
        # (default ``"error"``), makes the server drop unbound blocks instead.
        # That would also unstick saves whose prefix already kept them, but it
        # was verified only on claude-opus-5-5 with adaptive thinking, and it
        # would silently drop thinking after any other history bug too.
        tail = tuple(
            item
            for item in plan.tail
            if not isinstance(item, (SampleMetadata, TurnSummary, ModelFailure))
            and not _is_signed_thinking(item)
        )
        checkpoint = ContextPrefix(
            prefix_items=(
                *plan.instruction_prefix,
                *plan.checkpoints,
                Message(
                    role="user",
                    content=(
                        f"{COMPACTION_SUMMARY_PREFIX}{summary}"
                        f"{COMPACTION_SUMMARY_SUFFIX}"
                    ),
                ),
                *tail,
            ),
        )
        last = samples[-1]
        return CompactionResult(
            items=(checkpoint,),
            usage=_sum_usage(sample.usage for sample in samples),
            protocol="pi",
            provider_session_id=last.provider_session_id,
            provider_turn_id=last.provider_turn_id,
            provider_turn_state=last.provider_turn_state,
            request_attempts=sum(sample.request_attempts for sample in samples),
            recovery=tuple(
                entry for sample in samples for entry in sample.recovery
            ),
        )

    def _summary_params(self, sample_params: Optional[SampleParams]) -> SampleParams:
        """Inherit the turn's params, without provider compaction.

        Temperature and model extras (thinking, effort) are inherited, as pi
        inherits the session's thinking level.
        """
        turn = sample_params or SampleParams()
        return replace(
            turn,
            max_output_tokens=(
                self.max_output_tokens
                if self.max_output_tokens is not None
                else turn.max_output_tokens
            ),
            enable_auto_compaction=False,
            auto_compact_tokens=None,
        )

    def _summarize(
        self,
        context: InteractionContext,
        prompt: str,
        params: SampleParams,
        *,
        part: str,
        items: Sequence[InteractionItem],
    ) -> Tuple[str, ModelSample]:
        # The model sees only the summarizer prompt and one user message. The
        # raw log keeps Init and all metadata, so provider session and turn
        # continuity (Codex headers, prompt_cache_key) still match the session.
        request = context.copy()
        request.append(ContextPrefix((
            Instructions(SUMMARIZATION_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        )))
        try:
            sample = self._model.sample(request, tools=(), sample_params=params)
        except ModelContextWindowError as exc:
            # TODO: Split the span in half at a unit boundary and fold the
            # halves through the update prompt, with a bounded depth and a
            # ``recovery`` entry, instead of failing (plan section 9.3).
            raise CompactionContextWindowError(
                _context_window_message(part, items, prompt)
            ) from exc
        if not isinstance(sample, ModelSample):
            raise CompactionError(
                f"model returned {type(sample).__name__}, expected ModelSample"
            )
        return _summary_text(sample, part), sample


__all__ = [
    "COMPACTION_MODES",
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "CompactionContextWindowError",
    "CompactionError",
    "CompactionResult",
    "CompactionSettings",
    "Compactor",
    "DEFAULT_KEEP_RECENT_TOKENS",
    "ESTIMATED_IMAGE_CHARS",
    "NothingToCompact",
    "PiCompactor",
    "SUMMARIZATION_PROMPT",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "TOOL_RESULT_MAX_CHARS",
    "TURN_PREFIX_SUMMARIZATION_PROMPT",
    "UPDATE_SUMMARIZATION_PROMPT",
    "auto_compaction_due",
    "create_default_compactor",
    "estimate_context_tokens",
    "estimate_item_tokens",
    "is_compaction_summary",
    "should_auto_compact",
    "uses_host_auto_compaction",
]
