"""Resolved, non-secret configuration for one interaction context binding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import json
import re
from threading import RLock
from types import SimpleNamespace
from typing import Dict
from typing import Optional
from typing import Union
from typing import Mapping

from .messages import MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
from .messages import resolve_messages_max_output_tokens
from .model import SampleParams
from .model_catalog import binding_from_namespace, freeze_extra_sample_params, thaw_json


ConfigValue = Union[bool, int, None, Mapping]
CONFIG_KEYS = (
    "enable_workspace",
    "max_samples",
    "max_output_tokens",
    "enable_auto_compaction",
    "auto_compact_tokens",
    "max_context_tokens",
    "extra_sample_params",
)
_BOOLEAN_KEYS = frozenset(("enable_workspace", "enable_auto_compaction"))
_OPTIONAL_POSITIVE_INTEGER_KEYS = frozenset((
    "max_samples",
    "max_output_tokens",
    "auto_compact_tokens",
    "max_context_tokens",
))
_INTEGER_LITERAL_RE = re.compile(r"^[+-]?[0-9]+$")


class ConfigError(ValueError):
    pass


def _require_key(key: object) -> str:
    if not isinstance(key, str) or key not in CONFIG_KEYS:
        raise ConfigError(
            "Unknown config key; use /config to list supported keys."
        )
    return key


def validate_config_value(key: str, value: object) -> ConfigValue:
    key = _require_key(key)
    if key == "extra_sample_params":
        try:
            return freeze_extra_sample_params(value)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
    if key in _BOOLEAN_KEYS:
        if not isinstance(value, bool):
            raise ConfigError(f"{key} requires True or False.")
        return value
    if key in _OPTIONAL_POSITIVE_INTEGER_KEYS:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(
                f"{key} requires a positive integer or None."
            )
        return value
    raise AssertionError(f"missing config validator for {key}")


def parse_config_literal(key: str, text: object) -> ConfigValue:
    """Parse the intentionally small JSON/Python scalar input grammar."""
    key = _require_key(key)
    if key == "extra_sample_params":
        raise ConfigError("extra_sample_params is launch-only; use /config.json to inspect it.")
    if not isinstance(text, str):
        raise ConfigError(f"Invalid value for {key}.")
    literal = text.strip()
    aliases = {
        "null": None,
        "None": None,
        "false": False,
        "False": False,
        "true": True,
        "True": True,
    }
    if literal in aliases:
        value = aliases[literal]
    elif _INTEGER_LITERAL_RE.fullmatch(literal) is not None:
        value = int(literal)
    else:
        raise ConfigError(f"Invalid value for {key}.")
    return validate_config_value(key, value)


@dataclass(frozen=True)
class InteractionConfigSnapshot:
    """Immutable context values; snapshots published by config are resolved.

    ``extra_sample_params`` is a launch-only, read-only view of the model
    binding's map. It is not projected into sample params: requests inherit
    the binding's map, so samples and compactions send the same extensions.
    """

    enable_workspace: bool = True
    max_samples: Optional[int] = None
    max_output_tokens: Optional[int] = None
    enable_auto_compaction: bool = True
    auto_compact_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None
    extra_sample_params: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in CONFIG_KEYS:
            object.__setattr__(self, key, validate_config_value(key, getattr(self, key)))

    def as_dict(self) -> Dict[str, ConfigValue]:
        return {key: thaw_json(getattr(self, key)) for key in CONFIG_KEYS}

    def sample_params(
        self, base: Optional[SampleParams] = None,
    ) -> SampleParams:
        """Project config-owned fields over a caller's other per-call params.

        Budget and compaction policy come from this snapshot. Other fields,
        including a per-call ``extra`` (usually None), come from ``base``.
        """
        if base is not None and not isinstance(base, SampleParams):
            raise TypeError("base must be SampleParams or None")
        return replace(
            base or SampleParams(),
            max_output_tokens=self.max_output_tokens,
            enable_auto_compaction=self.enable_auto_compaction,
            auto_compact_tokens=self.auto_compact_tokens,
        )

class InteractionConfig:
    """Thread-safe effective policy with immutable launch and default baselines.

    Factories bind raw frontend inputs to defaults. ``set(key, None)`` resolves
    against that captured default, while ``reset`` restores the launch value.
    Reads and request projection never consult models or the catalog.
    """

    def __init__(
        self,
        snapshot: InteractionConfigSnapshot = InteractionConfigSnapshot(),
        *,
        initial: Optional[InteractionConfigSnapshot] = None,
        on_enable_workspace: Optional[Callable[[bool], None]] = None,
        require_max_output_tokens: bool = False,
        max_output_tokens_fallback: Optional[int] = None,
        auto_compact_tokens_fallback: Optional[int] = None,
        max_context_tokens_fallback: Optional[int] = None,
        min_auto_compact_tokens: Optional[int] = None,
    ) -> None:
        if not isinstance(snapshot, InteractionConfigSnapshot):
            raise TypeError("snapshot must be InteractionConfigSnapshot")
        if initial is not None and not isinstance(
            initial, InteractionConfigSnapshot
        ):
            raise TypeError("initial must be InteractionConfigSnapshot or None")
        if on_enable_workspace is not None and not callable(
            on_enable_workspace
        ):
            raise TypeError("on_enable_workspace must be callable or None")
        if not isinstance(require_max_output_tokens, bool):
            raise TypeError("require_max_output_tokens must be a bool")
        # Bound defaults are used only at initialization/mutation, never on read.
        self._fallbacks = {
            key: validate_config_value(key, value)
            for key, value in (
                ("max_output_tokens", max_output_tokens_fallback),
                ("auto_compact_tokens", auto_compact_tokens_fallback),
                ("max_context_tokens", max_context_tokens_fallback),
            )
        }
        self._require_max_output_tokens = require_max_output_tokens
        self._min_auto_compact_tokens = validate_config_value(
            "auto_compact_tokens", min_auto_compact_tokens,
        )
        snapshot = self._resolve(snapshot)
        initial = snapshot if initial is None else self._resolve(initial)
        self._validate(snapshot)
        self._validate(initial)
        self._lock = RLock()
        self._snapshot = snapshot
        # The resolved launch values remain fixed after live config changes.
        self._initial = initial
        self._on_enable_workspace = on_enable_workspace

    def _resolve(
        self, snapshot: InteractionConfigSnapshot,
    ) -> InteractionConfigSnapshot:
        return replace(snapshot, **{
            key: value for key, value in self._fallbacks.items()
            if getattr(snapshot, key) is None
        })

    def _validate(self, snapshot: InteractionConfigSnapshot) -> None:
        if self._require_max_output_tokens and snapshot.max_output_tokens is None:
            raise ConfigError("max_output_tokens is required for this Messages model.")
        threshold = snapshot.auto_compact_tokens
        minimum = self._min_auto_compact_tokens
        if threshold is not None and minimum is not None and threshold < minimum:
            raise ConfigError(
                f"auto_compact_tokens must be at least {minimum} for this model."
            )

    def _publish(self, candidate: InteractionConfigSnapshot) -> None:
        """Validate and apply under the caller's lock; failed effects don't publish."""
        self._validate(candidate)
        if (candidate.enable_workspace != self._snapshot.enable_workspace
                and self._on_enable_workspace is not None):
            self._on_enable_workspace(candidate.enable_workspace)
        self._snapshot = candidate

    @classmethod
    def from_namespace(
        cls,
        args,
        *,
        on_enable_workspace: Optional[Callable[[bool], None]] = None,
        catalog=None,
    ) -> "InteractionConfig":
        """Seed one final namespace; do not write resolved values back into it."""
        binding = binding_from_namespace(args, catalog)
        spec = binding.spec
        limits = None if spec is None else spec.limits
        max_output_tokens = args.max_output_tokens
        max_output_tokens_fallback = (
            limits.max_output_tokens
            if binding.api == "messages" and limits is not None else None
        )
        if binding.api == "messages":
            max_output_tokens = resolve_messages_max_output_tokens(
                binding,
                max_output_tokens,
            )
        return cls(
            InteractionConfigSnapshot(
                enable_workspace=args.enable_workspace,
                max_samples=args.max_samples,
                max_output_tokens=max_output_tokens,
                enable_auto_compaction=args.enable_auto_compaction,
                auto_compact_tokens=getattr(args, "auto_compact_tokens", None),
                max_context_tokens=getattr(args, "max_context_tokens", None),
                extra_sample_params=binding.extra_sample_params,
            ),
            on_enable_workspace=on_enable_workspace,
            require_max_output_tokens=binding.api == "messages",
            max_output_tokens_fallback=max_output_tokens_fallback,
            auto_compact_tokens_fallback=(
                None if limits is None else limits.auto_compact_context_tokens
            ),
            max_context_tokens_fallback=(
                None if limits is None else limits.max_context_tokens
            ),
            min_auto_compact_tokens=(
                MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
                if binding.api == "messages" else None
            ),
        )

    @classmethod
    def from_model(
        cls,
        model,
        snapshot: InteractionConfigSnapshot = InteractionConfigSnapshot(),
    ) -> "InteractionConfig":
        """Bind Python frontend inputs once, without loading credentials.

        Endpoint budgets/triggers are initial preferences, not later overrides.
        Custom models may supply limit metadata at this boundary only. A
        built-in model's binding owns ``extra_sample_params``; a snapshot may
        repeat that map but not change it.
        """
        from .chat_completions import ChatCompletionsModel
        from .messages import MessagesModel
        from .responses import CodexResponsesModel

        if not isinstance(snapshot, InteractionConfigSnapshot):
            raise TypeError("snapshot must be InteractionConfigSnapshot")
        if isinstance(model, MessagesModel):
            endpoint = model.endpoint
            compaction = endpoint.server_compaction
            snapshot = replace(
                snapshot,
                max_output_tokens=(
                    endpoint.max_output_tokens if snapshot.max_output_tokens is None
                    else snapshot.max_output_tokens
                ),
                auto_compact_tokens=(
                    compaction.trigger_input_tokens
                    if snapshot.auto_compact_tokens is None and compaction is not None
                    else snapshot.auto_compact_tokens
                ),
            )
            profile = "messages"
        elif isinstance(model, CodexResponsesModel):
            endpoint = model.endpoint
            profile = model.binding.endpoint.api
        elif isinstance(model, ChatCompletionsModel):
            endpoint = model.endpoint
            profile = "chat-completions"
        else:
            return cls(
                snapshot,
                auto_compact_tokens_fallback=getattr(
                    model, "auto_compact_context_tokens", None,
                ),
                max_context_tokens_fallback=getattr(model, "max_context_tokens", None),
            )
        binding = model.binding
        if (snapshot.extra_sample_params
                and snapshot.extra_sample_params != binding.extra_sample_params):
            raise ConfigError(
                "extra_sample_params come from the model binding; bind them on "
                "the model, or pass SampleParams(extra=...) for per-call extras."
            )
        return cls.from_namespace(SimpleNamespace(
            model_api=profile, model=endpoint.model,
            model_binding=binding,
            **snapshot.as_dict(),
        ))

    def snapshot(self) -> InteractionConfigSnapshot:
        with self._lock:
            return self._snapshot

    def initial_snapshot(self) -> InteractionConfigSnapshot:
        with self._lock:
            return self._initial

    def get(self, key: str) -> ConfigValue:
        key = _require_key(key)
        return getattr(self.snapshot(), key)

    def set(self, key: str, value: object) -> ConfigValue:
        key = _require_key(key)
        if key == "extra_sample_params":
            raise ConfigError("extra_sample_params is launch-only; use /config.json to inspect it.")
        value = validate_config_value(key, value)
        if value is None and key in self._fallbacks:
            value = self._fallbacks[key]
        with self._lock:
            current = self._snapshot
            if getattr(current, key) == value:
                return value
            self._publish(replace(current, **{key: value}))
        return value

    def reset(self, key: Optional[str] = None) -> InteractionConfigSnapshot:
        """Restore the resolved launch state (not the catalog fallback)."""
        if key is not None:
            key = _require_key(key)
        with self._lock:
            candidate = self._initial if key is None else replace(
                self._snapshot, **{key: getattr(self._initial, key)},
            )
            self._publish(candidate)
            return self._snapshot

    def values(self, key: Optional[str] = None) -> Dict[str, ConfigValue]:
        snapshot = self.snapshot()
        if key is not None:
            key = _require_key(key)
            return {key: snapshot.as_dict()[key]}
        return snapshot.as_dict()

    def initial_values(self, key: Optional[str] = None) -> Dict[str, ConfigValue]:
        initial = self.initial_snapshot()
        if key is not None:
            key = _require_key(key)
            return {key: initial.as_dict()[key]}
        return initial.as_dict()

    def render(self, key: Optional[str] = None, *, json_output: bool) -> str:
        values = self.values(key)
        initial = self.initial_values(key)
        if json_output:
            # ``__init__`` carries the launch-seeded defaults (a dict) alongside
            # the current values; `/config-reset` (TODO) will restore them.
            return json.dumps(
                {"__init__": initial, **values},
                ensure_ascii=False,
                indent=2,
            )
        lines = []
        for name, value in values.items():
            seeded = initial[name]
            if seeded != value:
                lines.append(f"# init: {name} = {seeded!r}")
            lines.append(f"{name} = {value!r}")
        return "\n".join(lines)


__all__ = [
    "CONFIG_KEYS",
    "ConfigError",
    "ConfigValue",
    "InteractionConfig",
    "InteractionConfigSnapshot",
    "parse_config_literal",
    "validate_config_value",
]
