"""Tool descriptions shared by runtime registrations and durable snapshots."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must not be empty")
        name = self.name.strip()
        if _TOOL_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                "tool name may contain only letters, digits, underscores, and hyphens"
            )
        object.__setattr__(self, "name", name)
        if not isinstance(self.description, str):
            raise TypeError("tool description must be a string")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a mapping")
        object.__setattr__(self, "parameters", dict(self.parameters))


def _copy_schema(value, depth=0):
    """Copy finite JSON data without retaining any caller-owned containers."""
    if depth > 64:
        raise ValueError("tool parameters exceed maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("tool parameter keys must be strings")
        return {key: _copy_schema(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy_schema(item, depth + 1) for item in value]
    raise TypeError("tool parameters must contain only finite JSON values")
