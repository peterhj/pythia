from __future__ import annotations

import json
import math
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Tuple

from .items import InteractionItem
from .items import Message
from .items import ToolCall
from .items import ToolResult
from ._tool_spec import ToolSpec

if TYPE_CHECKING:
    from .display import DisplayItem


class EnvironmentError(ValueError):
    pass


def _validate_user_messages(messages: Iterable[Message]) -> Tuple[Message, ...]:
    messages = tuple(messages)
    for index, message in enumerate(messages):
        if not isinstance(message, Message) or message.role != "user":
            raise EnvironmentError(
                f"user_messages[{index}] must be a user-role Message"
            )
    return messages


@dataclass(frozen=True)
class ToolOutcome:
    """Tool output plus optional synthetic user messages for the caller to append.

    Messages are explicit, trusted handler data, never parsed from output text.
    They are permitted only on successful outcomes and do not start a new turn.
    """

    output: str
    success: bool = True
    user_messages: Tuple[Message, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise TypeError("tool output must be a string")
        if not isinstance(self.success, bool):
            raise TypeError("tool success must be a bool")
        messages = _validate_user_messages(self.user_messages)
        if messages and not self.success:
            raise EnvironmentError(
                "unsuccessful tool outcomes must not contain user messages"
            )
        object.__setattr__(self, "user_messages", messages)


class ToolHandler(Protocol):
    def __call__(
        self,
        arguments: Mapping[str, object],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ToolOutcome:
        ...


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: ToolHandler
    timeout_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ToolSpec):
            raise TypeError("spec must be ToolSpec")
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if self.timeout_seconds is not None:
            value = self.timeout_seconds
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    "timeout_seconds must be positive and finite or None"
                )
            object.__setattr__(self, "timeout_seconds", float(value))


@dataclass(frozen=True)
class EnvironmentResult:
    """A tool-result batch followed by optional synthetic user messages.

    ``items`` remains tool-result-only. Append ``context_items()`` as one batch
    and checkpoint it before sampling. When tools can return user messages,
    execute the entire pending call batch, not one call at a time: messages
    cannot be appended while other tool calls remain unresolved.
    """

    items: Tuple[InteractionItem, ...]
    user_messages: Tuple[Message, ...] = ()

    def __post_init__(self) -> None:
        items = tuple(self.items)
        for index, item in enumerate(items):
            if not isinstance(item, ToolResult):
                raise EnvironmentError(
                    "environment results must contain only ToolResult items; "
                    f"item {index} is {type(item).__name__}"
                )
        object.__setattr__(self, "items", items)
        messages = _validate_user_messages(self.user_messages)
        if messages and not any(item.success for item in items):
            raise EnvironmentError(
                "user messages require a successful tool result"
            )
        object.__setattr__(self, "user_messages", messages)

    def context_items(self) -> Tuple[InteractionItem, ...]:
        # No UserInteractionBoundary: synthetic messages continue the current
        # turn, including its provider continuity metadata.
        return (*self.items, *self.user_messages)

    def display_items(
        self,
        *,
        source_calls: Iterable[ToolCall] = (),
    ) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(
            self.context_items(),
            source_calls=source_calls,
        )


def _failed_result(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        output=message,
        success=False,
    )


class Environment:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        registrations: Dict[str, Tool] = {}
        for index, tool in enumerate(tools):
            if not isinstance(tool, Tool):
                raise TypeError(f"tools[{index}] must be Tool")
            if tool.spec.name in registrations:
                raise EnvironmentError(
                    f"duplicate tool registration: {tool.spec.name!r}"
                )
            registrations[tool.spec.name] = tool
        self._tools = registrations

    @property
    def tool_specs(self) -> Tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def execute_tool_calls(
        self,
        calls: Sequence[ToolCall],
    ) -> EnvironmentResult:
        ordered_calls = tuple(calls)
        seen_call_ids = set()
        for index, call in enumerate(ordered_calls):
            if not isinstance(call, ToolCall):
                raise EnvironmentError(f"calls[{index}] must be ToolCall")
            if call.call_id in seen_call_ids:
                raise EnvironmentError(
                    f"duplicate tool call id: {call.call_id!r}"
                )
            seen_call_ids.add(call.call_id)

        results = []
        user_messages = []
        for call in ordered_calls:
            try:
                parsed_arguments = json.loads(call.arguments_json)
            except json.JSONDecodeError as exc:
                results.append(
                    _failed_result(
                        call,
                        f"Malformed JSON arguments for tool {call.name!r}: {exc}",
                    )
                )
                continue

            if not isinstance(parsed_arguments, Mapping):
                results.append(
                    _failed_result(
                        call,
                        f"Tool {call.name!r} arguments must decode to an object",
                    )
                )
                continue

            registration = self._tools.get(call.name)
            if registration is None:
                results.append(
                    _failed_result(call, f"Unknown tool: {call.name}")
                )
                continue

            try:
                outcome = registration.handler(
                    dict(parsed_arguments),
                    timeout_seconds=registration.timeout_seconds,
                )
                if not isinstance(outcome, ToolOutcome):
                    raise TypeError(
                        "tool handler must return ToolOutcome, got "
                        f"{type(outcome).__name__}"
                    )
            except TimeoutError as exc:
                detail = str(exc).strip()
                suffix = f": {detail}" if detail else ""
                results.append(
                    _failed_result(
                        call,
                        f"Tool {call.name!r} timed out{suffix}",
                    )
                )
                continue
            except Exception as exc:
                results.append(
                    _failed_result(
                        call,
                        f"Tool {call.name!r} failed: "
                        f"{exc.__class__.__name__}: {exc}",
                    )
                )
                continue

            results.append(
                ToolResult(
                    call_id=call.call_id,
                    output=outcome.output,
                    success=outcome.success,
                )
            )
            if outcome.success:
                user_messages.extend(outcome.user_messages)

        return EnvironmentResult(
            items=tuple(results),
            user_messages=tuple(user_messages),
        )


__all__ = [
    "Environment",
    "EnvironmentError",
    "EnvironmentResult",
    "Tool",
    "ToolHandler",
    "ToolOutcome",
    "ToolSpec",
]
