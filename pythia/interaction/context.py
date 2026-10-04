from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import overload

from .items import CompactionMetadata
from .items import ContextPrefix
from .items import Init
from .items import Instructions
from .items import InteractionItem
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import ToolCall
from .items import ToolResult
from .items import Tools
from .items import SampleMetadata
from .items import TurnSummary
from .items import UserToolCall
from .items import UserToolResult
from .items import is_interaction_item


class ContextValidationError(ValueError):
    pass


def _validate_item(item: object, field_name: str) -> InteractionItem:
    if not is_interaction_item(item):
        raise ContextValidationError(
            f"{field_name} must be an InteractionItem, got {type(item).__name__}"
        )
    return item


def _validate_tool_sequence(
    items: Sequence[InteractionItem],
    *,
    allow_pending: bool,
) -> Tuple[ToolCall, ...]:
    seen_call_ids = set()
    pending = {}
    call_batch_closed = False
    results_started = False

    for index, item in enumerate(items):
        if isinstance(item, ToolCall):
            if pending and (call_batch_closed or results_started):
                raise ContextValidationError(
                    "tool call appears before results for the preceding model "
                    f"sample at item {index}: {item.call_id!r}"
                )
            if item.call_id in seen_call_ids:
                raise ContextValidationError(
                    f"duplicate tool call id at item {index}: {item.call_id!r}"
                )
            seen_call_ids.add(item.call_id)
            pending[item.call_id] = item
            continue

        if isinstance(
            item,
            (
                ModelSampleBoundary,
                SampleMetadata,
                CompactionMetadata,
                ModelFailure,
                TurnSummary,
            ),
        ):
            if pending:
                call_batch_closed = True
            continue

        if isinstance(item, ToolResult):
            if item.call_id not in pending:
                raise ContextValidationError(
                    "tool result does not match an unresolved tool call at "
                    f"item {index}: {item.call_id!r}"
                )
            results_started = True
            call_batch_closed = True
            del pending[item.call_id]
            if not pending:
                call_batch_closed = False
                results_started = False
            continue

        if pending:
            call_ids = ", ".join(call.call_id for call in pending.values())
            raise ContextValidationError(
                f"item {index} appears before unresolved tool results: {call_ids}"
            )

    pending_calls = tuple(pending.values())
    if pending_calls and not allow_pending:
        call_ids = ", ".join(call.call_id for call in pending_calls)
        raise ContextValidationError(
            f"context contains unresolved tool calls: {call_ids}"
        )
    return pending_calls


def _validate_context_prefix(
    prefix_items: Sequence[InteractionItem],
) -> Tuple[InteractionItem, ...]:
    prefix = tuple(prefix_items)
    for index, item in enumerate(prefix):
        _validate_item(item, f"prefix_items[{index}]")
        if isinstance(item, (UserToolCall, UserToolResult)):
            raise ContextValidationError("user tools cannot appear in context prefixes")
        if isinstance(item, CompactionMetadata):
            raise ContextValidationError(
                "compaction metadata cannot appear in context prefixes"
            )
        if isinstance(item, ModelFailure):
            raise ContextValidationError(
                "model failures cannot appear in context prefixes"
            )
        if isinstance(item, ContextPrefix):
            raise ContextValidationError(
                "ContextPrefix prefix_items must not contain "
                "another ContextPrefix"
            )
        if isinstance(item, Init):
            raise ContextValidationError(
                "ContextPrefix prefix_items must not contain "
                "Init"
            )
    _validate_tool_sequence(prefix, allow_pending=False)
    return prefix


def _project_items(
    items: Sequence[InteractionItem],
) -> Tuple[InteractionItem, ...]:
    active: List[InteractionItem] = []
    for item in items:
        if isinstance(item, ContextPrefix):
            _validate_tool_sequence(active, allow_pending=False)
            active = list(
                _validate_context_prefix(item.prefix_items)
            )
        else:
            active.append(item)
    _validate_tool_sequence(active, allow_pending=True)
    return tuple(
        i
        for i in active
        if not isinstance(
            i,
            (CompactionMetadata, UserToolCall, UserToolResult, Tools),
        )
    )


def _pending_user_tools(items: Sequence[InteractionItem]) -> Tuple[UserToolCall, ...]:
    pending = None
    seen = set()
    for item in items:
        if isinstance(item, UserToolResult):
            if pending is None or item.result.call_id != pending.call.call_id:
                raise ContextValidationError("user tool result has no matching unresolved user call")
            pending = None
        elif pending is not None:
            raise ContextValidationError("item appears before unresolved user tool result")
        elif isinstance(item, UserToolCall):
            if item.call.call_id in seen:
                raise ContextValidationError("duplicate user tool call id")
            seen.add(item.call.call_id)
            pending = item
    return (pending,) if pending is not None else ()


def _validate_log(items: Sequence[InteractionItem]) -> None:
    _pending_user_tools(items)
    for index, item in enumerate(items):
        _validate_item(item, f"items[{index}]")
        if isinstance(item, Init) and index != 0:
            raise ContextValidationError(
                "Init must be the first interaction item"
            )
        if isinstance(item, ContextPrefix):
            _validate_context_prefix(item.prefix_items)
    _project_items(items)


def _collapse_instructions(
    items: Sequence[InteractionItem],
) -> Tuple[InteractionItem, ...]:
    """Collapse ``Instructions`` history to its effective item.

    The raw log keeps every ``Instructions`` for audit, but the model view
    exposes only the last one (later overrides earlier). ``No instructions``
    is represented solely by absence. Empty and whitespace-only text remains
    effective when it is the last item. The survivor is hoisted to the front
    so all backends encode a single leading system prompt regardless of
    where overrides were appended.
    """
    effective: Optional[InteractionItem] = None
    for item in items:
        if isinstance(item, Instructions):
            effective = item
    if effective is None:
        return tuple(items)
    remaining = tuple(item for item in items if not isinstance(item, Instructions))
    return (effective, *remaining)


class InteractionContext(Sequence[InteractionItem]):
    """Caller-owned, validated append-only interaction log.

    ``items`` retains the full log; ``model_items()`` derives its effective
    model view. This container is synchronous and not thread-safe. Scheduling
    and serialization of mutations remain the caller's responsibility.
    """

    def __init__(self, items: Iterable[InteractionItem] = ()) -> None:
        initial_items = list(items)
        _validate_log(initial_items)
        self._items = initial_items

    def __iter__(self) -> Iterator[InteractionItem]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> InteractionItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[InteractionItem]:
        ...

    def __getitem__(
        self,
        index: Union[int, slice],
    ) -> Union[InteractionItem, List[InteractionItem]]:
        return self._items[index]

    @property
    def items(self) -> Tuple[InteractionItem, ...]:
        return tuple(self._items)

    def model_items(self) -> Tuple[InteractionItem, ...]:
        projected = tuple(
            item
            for item in _project_items(self._items)
            if not isinstance(item, Init)
        )
        return _collapse_instructions(projected)

    def latest_tools(self) -> Optional[Tools]:
        """Last top-level tool snapshot in the raw log, not restored config.

        ContextPrefix projection intentionally does not affect this lookup:
        compaction must not cause unchanged runtime tools to be logged again.
        """
        return next(
            (item for item in reversed(self._items) if isinstance(item, Tools)),
            None,
        )

    def pending_tool_calls(self) -> Tuple[ToolCall, ...]:
        return _validate_tool_sequence(
            self.model_items(),
            allow_pending=True,
        )

    def assert_model_ready(self) -> None:
        if self.pending_user_tool_calls():
            raise ContextValidationError("cannot sample with unresolved user tool calls")
        pending = self.pending_tool_calls()
        if pending:
            call_ids = ", ".join(call.call_id for call in pending)
            raise ContextValidationError(
                f"cannot sample with unresolved tool calls: {call_ids}"
            )

    def pending_user_tool_calls(self) -> Tuple[UserToolCall, ...]:
        return _pending_user_tools(self._items)

    def append(self, item: InteractionItem) -> None:
        self.extend((item,))

    def extend(self, items: Iterable[InteractionItem]) -> None:
        new_items = list(items)
        candidate = [*self._items, *new_items]
        _validate_log(candidate)
        self._items.extend(new_items)

    def copy(self) -> "InteractionContext":
        return InteractionContext(self._items)

    def __repr__(self) -> str:
        return f"InteractionContext({self._items!r})"


__all__ = [
    "ContextValidationError",
    "InteractionContext",
]
