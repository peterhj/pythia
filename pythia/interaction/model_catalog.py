"""Static, profile-scoped model facts and Pythia request presets.

This module has no adapter, credential, environment, or network dependencies.
Maximum/output limits are metadata rather than request budgets;
``auto_compact_context_tokens`` is caller policy consumed by interaction
frontends and server-compaction configuration. Endpoints contain non-secret
delivery policy, not resolved credential values or permission to use account
services. Unknown names remain valid pass-through candidates for the owning
adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import json
import math
import re
from types import MappingProxyType
from collections.abc import Mapping as MappingABC
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Tuple
from urllib.parse import urlsplit


OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1"
CODEX_RESPONSES_API_URL = "https://chatgpt.com/backend-api/codex"
META_RESPONSES_API_URL = "https://api.meta.ai/v1"
ANTHROPIC_MESSAGES_API_URL = "https://api.anthropic.com"
# Shared protocol constraint for catalog validation and Messages request policy.
MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS = 50_000

_PROFILES = frozenset(("codex", "responses", "messages", "chat-completions"))

# Extensions are not a second path for typed policy or adapter-owned structure.
_RESERVED_REQUEST_PARAMS = frozenset((
    "model", "messages", "input", "instructions", "stream", "stream_options",
    "tools", "parallel_tool_calls", "max_tokens", "max_output_tokens",
    "max_completion_tokens", "max_new_tokens", "temperature", "top_p", "stop",
    "stop_sequences", "seed", "enable_auto_compaction", "auto_compact_tokens",
    "max_context_tokens", "context_management", "api", "model_api", "api_model",
    "api_url", "api_key", "api_key_env", "headers", "authorization",
    "bearer_token", "access_token", "account_id", "request_timeout_seconds",
))
# Each API's adapter also owns these top-level fields, including those holding
# typed catalog defaults (Messages output_config; Responses reasoning/text).
# They are protected even when a particular request omits them.
_RESPONSES_RESERVED_REQUEST_PARAMS = frozenset((
    "tool_choice", "store", "include", "reasoning", "text", "prompt_cache_key",
    # Stateless requests carry the complete context; no server-side history.
    "previous_response_id", "conversation",
))
_PROFILE_RESERVED_REQUEST_PARAMS = MappingProxyType({
    "chat-completions": _RESERVED_REQUEST_PARAMS,
    "messages": _RESERVED_REQUEST_PARAMS | {"system", "output_config", "cache_control"},
    "responses": _RESERVED_REQUEST_PARAMS | _RESPONSES_RESERVED_REQUEST_PARAMS,
    "codex": _RESERVED_REQUEST_PARAMS | _RESPONSES_RESERVED_REQUEST_PARAMS,
})
MAX_REQUEST_PARAMS_BYTES = 65_536


def parse_json_value(text: str):
    """Strict JSON, also used for structured INI values; never evaluate Python."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise ValueError("Invalid JSON value.") from None


def _freeze_json(value, depth=0):
    if depth > 32:
        raise ValueError("Request params exceed maximum nesting depth.")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, MappingABC):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Request param object keys must be strings.")
        return MappingProxyType({key: _freeze_json(item, depth + 1) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, depth + 1) for item in value)
    raise ValueError("Request params must contain only finite JSON values.")


def thaw_json(value):
    """Return fresh JSON containers; never expose a frozen config's internals."""
    if isinstance(value, MappingABC):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def freeze_request_params(value, profile=None):
    """Validate and freeze request-body extensions.

    Without a profile only names reserved for every API are checked; bindings
    and adapters also reject the names owned by their API's adapter.
    """
    if not isinstance(value, MappingABC):
        raise ValueError("request_params must be an object.")
    reserved = (_RESERVED_REQUEST_PARAMS if profile is None
                else _PROFILE_RESERVED_REQUEST_PARAMS[_normalize_profile(profile)])
    if any(not isinstance(key, str) or not key or key.lower() in reserved
           or any(ord(char) < 32 for char in key) for key in value):
        raise ValueError("Invalid or adapter-owned request parameter.")
    frozen = _freeze_json(value)
    try:
        encoded = json.dumps(thaw_json(frozen), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("Request params are not valid JSON.") from None
    if len(encoded) > MAX_REQUEST_PARAMS_BYTES:
        raise ValueError("Request params exceed the size limit.")
    return frozen


def validate_endpoint_url(value):
    """Validate a complete external endpoint URL."""
    try:
        if not isinstance(value, str) or not value or any(char.isspace() or ord(char) < 32 for char in value):
            raise ValueError()
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.port == 0):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(
            "Endpoint URL must be HTTP(S), without credentials, query, or fragment."
        ) from None


def _normalize_profile(profile: str) -> str:
    if not isinstance(profile, str) or profile not in _PROFILES:
        raise ValueError("unknown model catalog profile")
    return profile


def _require_identifier(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field_name} must be nonempty and contain no whitespace or controls")


@dataclass(frozen=True)
class EndpointSpec:
    """The single non-secret delivery authority. url is a complete POST URL."""

    api: str
    url: str
    model: Optional[str] = None
    auth: str = "none"
    # Resolved login reference, not file contents. Filled at the launch boundary.
    auth_file: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "api", _normalize_profile(self.api))
        validate_endpoint_url(self.url)
        if self.model is not None:
            _require_identifier(self.model, "endpoint.model")
        if not isinstance(self.auth, str) or not (
            self.auth in {"none", "supplied", "codex-login"}
            or (self.auth.startswith("env:") and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.auth[4:]))
        ):
            raise ValueError("endpoint.auth must be none, supplied, codex-login, or env:NAME")
        if self.auth == "codex-login" and self.api != "codex":
            raise ValueError("codex-login requires the Codex API")
        if self.auth_file is not None:
            if self.auth != "codex-login" or not isinstance(self.auth_file, str) or not self.auth_file.strip() or "\x00" in self.auth_file:
                raise ValueError("auth_file requires a nonempty Codex login reference")

    @property
    def environment_variable(self):
        return self.auth[4:] if self.auth.startswith("env:") else None

    @property
    def connection_identity(self):
        return self.api, self.url, self.auth, self.auth_file

    @property
    def is_official_codex(self):
        return self.api == "codex" and self.url.rstrip("/") == CODEX_RESPONSES_API_URL + "/responses"

@dataclass(frozen=True)
class ModelLimits:
    """Context policy and known ceilings for one catalogued model."""

    auto_compact_context_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "auto_compact_context_tokens",
            "max_context_tokens",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if (
            self.auto_compact_context_tokens is not None
            and self.max_context_tokens is not None
            and self.auto_compact_context_tokens > self.max_context_tokens
        ):
            raise ValueError(
                "auto-compact context must not exceed maximum context"
            )


@dataclass(frozen=True)
class ResponsesDefaults:
    """Pythia request preferences, not claims about a model's native defaults."""

    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    text_verbosity: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("reasoning_effort", "reasoning_summary", "text_verbosity"):
            value = getattr(self, name)
            if value is not None:
                _require_identifier(value, name)


@dataclass(frozen=True)
class MessagesDefaults:
    """Pythia Messages request preferences, not native model defaults."""

    output_effort: Optional[str] = None

    def __post_init__(self) -> None:
        if self.output_effort is not None:
            _require_identifier(self.output_effort, "output_effort")


@dataclass(frozen=True)
class ModelSpec:
    """A named policy and its single authoritative endpoint."""

    name: str
    endpoint: EndpointSpec
    limits: ModelLimits = field(default_factory=ModelLimits)
    responses: Optional[ResponsesDefaults] = None
    aliases: Tuple[str, ...] = ()
    source: Optional[str] = None
    messages: Optional[MessagesDefaults] = None
    request_params: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.name, "name")
        if not isinstance(self.endpoint, EndpointSpec):
            raise TypeError("endpoint must be EndpointSpec")
        _require_identifier(self.endpoint.model, "endpoint.model")
        if not isinstance(self.limits, ModelLimits):
            raise TypeError("limits must be ModelLimits")
        if (self.endpoint.api == "messages" and self.limits.auto_compact_context_tokens is not None
                and self.limits.auto_compact_context_tokens < MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS):
            raise ValueError("Messages compaction threshold must be at least 50000.")
        if self.responses is not None:
            if not isinstance(self.responses, ResponsesDefaults):
                raise TypeError("responses must be ResponsesDefaults or None")
            if self.endpoint.api not in {"codex", "responses"}:
                raise ValueError("Responses defaults require a Responses endpoint API")
        if self.messages is not None:
            if not isinstance(self.messages, MessagesDefaults):
                raise TypeError("messages must be MessagesDefaults or None")
            if self.endpoint.api != "messages":
                raise ValueError(
                    "Messages defaults require the Messages endpoint API"
                )
        if isinstance(self.aliases, (str, bytes)):
            raise TypeError("aliases must be an iterable of strings")
        aliases = tuple(self.aliases)
        for alias in aliases:
            _require_identifier(alias, "alias")
        object.__setattr__(self, "aliases", aliases)
        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("source must be a nonempty string or None")
        object.__setattr__(self, "request_params", freeze_request_params(
            self.request_params, self.endpoint.api,
        ))


_CHATGPT = EndpointSpec(
    "codex", CODEX_RESPONSES_API_URL + "/responses", auth="codex-login",
)
_META = EndpointSpec(
    "codex", META_RESPONSES_API_URL + "/responses", auth="env:META_API_KEY",
)
_ANTHROPIC = EndpointSpec(
    "messages", ANTHROPIC_MESSAGES_API_URL + "/v1/messages",
    auth="env:ANTHROPIC_API_KEY",
)
_PROFILE_DEFAULT_ENDPOINTS = MappingProxyType({
    "codex": _CHATGPT,
    "responses": EndpointSpec(
        "responses", OPENAI_RESPONSES_API_URL + "/responses", auth="supplied",
    ),
    "messages": _ANTHROPIC,
    "chat-completions": EndpointSpec(
        "chat-completions", "http://127.0.0.1:8000/v1/chat/completions",
    ),
})

# Shared immutable facts, inherited by effort presets rather than copied.
_CODEX_LIMITS = ModelLimits(
    auto_compact_context_tokens=872_000,
    max_context_tokens=1_000_000,
    max_output_tokens=128_000,
)
_SOL = ModelSpec(
    name="codex-gpt-5.6-sol", endpoint=replace(_CHATGPT, model="gpt-5.6-sol"),
    limits=_CODEX_LIMITS, responses=ResponsesDefaults(),
    source=(
        "codex-latest-20260904/codex-rs/models-manager/models.json; "
        "Pythia auto-compaction/max-context policy override"
    ),
)
_ASTRA = replace(
    _SOL, name="codex-gpt-6-astra",
    endpoint=replace(_SOL.endpoint, model="gpt-6-astra"),
    # Pythia deliberately requests summaries; the bundled catalog default is none.
    responses=ResponsesDefaults(reasoning_summary="auto", text_verbosity="low"),
)
# GPT-6 siblings match Astra's catalog capacities and summary/verbosity support,
# so they share its limits and request policy.
_GPT6_SOL = replace(
    _ASTRA, name="codex-gpt-6-sol",
    endpoint=replace(_ASTRA.endpoint, model="gpt-6-sol"),
    source=(
        "codex-latest-20260927/codex-rs/models-manager/models.json; "
        "Pythia auto-compaction/max-context policy override"
    ),
)
_GPT6_LUNA = replace(
    _GPT6_SOL, name="codex-gpt-6-luna",
    endpoint=replace(_GPT6_SOL.endpoint, model="gpt-6-luna"),
)
_SPARK = ModelSpec(
    name="muse-spark-1.3",
    endpoint=replace(_META, model="muse-spark-1.3-contributor"),
    responses=ResponsesDefaults(),
    source="Existing Pythia Meta Responses integration presets; token capacities unknown",
)
_FABLE = ModelSpec(
    name="claude-fable-5.1",
    endpoint=replace(_ANTHROPIC, model="claude-fable-5-1"),
    limits=ModelLimits(
        auto_compact_context_tokens=872_000,
        max_context_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    source=(
        "https://platform.claude.com/docs/en/models/fable-5-1/overview; "
        "https://platform.claude.com/docs/en/build-with-claude/effort"
    ),
)
_OPUS = ModelSpec(
    name="claude-opus-5.5",
    endpoint=replace(_ANTHROPIC, model="claude-opus-5-5"),
    limits=ModelLimits(
        auto_compact_context_tokens=872_000,
        max_context_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    source=(
        "https://platform.claude.com/docs/en/models/opus-5-5/overview; "
        "https://platform.claude.com/docs/en/build-with-claude/effort"
    ),
)


def _with_effort(base: ModelSpec, name: str, effort: str) -> ModelSpec:
    return replace(
        base, name=name, aliases=(),
        responses=replace(base.responses or ResponsesDefaults(), reasoning_effort=effort),
    )


def _with_messages_effort(
    base: ModelSpec,
    name: str,
    effort: str,
    *,
    aliases: Tuple[str, ...] = (),
) -> ModelSpec:
    return replace(
        base,
        name=name,
        aliases=aliases,
        messages=MessagesDefaults(
            output_effort=effort,
        ),
    )


_MODEL_SPECS = (
    _SOL,
    _with_effort(_SOL, "codex-gpt-5.6-sol-medium", "medium"),
    _with_effort(_SOL, "codex-gpt-5.6-sol-max", "max"),
    _ASTRA,
    _with_effort(_ASTRA, "codex-gpt-6-astra-medium", "medium"),
    _with_effort(_ASTRA, "codex-gpt-6-astra-max", "max"),
    _GPT6_SOL,
    _with_effort(_GPT6_SOL, "codex-gpt-6-sol-medium", "medium"),
    _with_effort(_GPT6_SOL, "codex-gpt-6-sol-max", "max"),
    _GPT6_LUNA,
    _with_effort(_GPT6_LUNA, "codex-gpt-6-luna-medium", "medium"),
    _with_effort(_GPT6_LUNA, "codex-gpt-6-luna-max", "max"),
    _SPARK,
    _with_effort(_SPARK, "muse-spark-1.3-xhigh", "xhigh"),
    _FABLE,
    _with_messages_effort(
        _FABLE,
        "claude-fable-5.1-max",
        "max",
    ),
    _OPUS,
    _with_messages_effort(
        _OPUS,
        "claude-opus-5.5-max",
        "max",
    ),
)


def _build_index(specs: Iterable[ModelSpec]) -> Mapping[Tuple[str, str], ModelSpec]:
    index = {}
    for spec in specs:
        if not isinstance(spec, ModelSpec):
            raise TypeError("catalog entries must be ModelSpec")
        for name in (spec.name, *spec.aliases):
            key = (spec.endpoint.api, name)
            if key in index:
                raise ValueError(
                    f"duplicate model selector in {spec.endpoint.api}: {name}"
                )
            index[key] = spec
    return MappingProxyType(index)


_MODEL_INDEX = _build_index(_MODEL_SPECS)


@dataclass(frozen=True)
class ModelBinding:
    """One resolved endpoint, including explicitly bound pass-through models."""

    selector: Optional[str]
    endpoint: EndpointSpec
    spec: Optional[ModelSpec] = None
    request_params: Mapping = field(default_factory=dict)
    origin: str = "builtin"
    api_explicit: bool = True

    def __post_init__(self):
        if not isinstance(self.endpoint, EndpointSpec):
            raise TypeError("binding endpoint must be EndpointSpec")
        if not isinstance(self.api_explicit, bool):
            raise TypeError("api_explicit must be a bool")
        if not isinstance(self.origin, str) or not self.origin:
            raise TypeError("binding origin must be a nonempty string")
        if self.selector is not None and not isinstance(self.selector, str):
            raise TypeError("selector must be a string or None")
        if self.spec is not None:
            if (not isinstance(self.spec, ModelSpec)
                    or self.spec.endpoint.api != self.api):
                raise ValueError("binding spec must match the selected API")
            if self.selector not in (self.spec.name, *self.spec.aliases):
                raise ValueError("binding selector does not select its spec")
        object.__setattr__(self, "request_params", freeze_request_params(self.request_params, self.api))

    @property
    def api(self):
        return self.endpoint.api

    @property
    def limits(self):
        return ModelLimits() if self.spec is None else self.spec.limits

    @property
    def supports_account_services(self):
        return self.endpoint.is_official_codex and self.endpoint.auth == "codex-login"

    @property
    def supports_remote_compaction(self):
        return self.endpoint.is_official_codex and self.endpoint.auth in {"codex-login", "supplied"}

    def with_request_params(self, overlay=None):
        params = dict(self.request_params)
        if overlay is not None:
            params.update(freeze_request_params(overlay, self.api))
        return replace(self, request_params=params)

@dataclass(frozen=True)
class ModelCatalog:
    """Immutable registry. File loading belongs to model_catalog_config, not here."""

    specs: Tuple[ModelSpec, ...] = ()
    origins: Mapping = field(default_factory=dict, repr=False)
    _index: Mapping = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        specs = tuple(self.specs)
        index = _build_index(specs)
        if not isinstance(self.origins, MappingABC) or any(
            not isinstance(value, str) or not value for value in self.origins.values()
        ):
            raise TypeError("catalog origins must be nonempty strings")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "origins", MappingProxyType({
            (spec.endpoint.api, spec.name): self.origins.get(
                (spec.endpoint.api, spec.name), "python",
            )
            for spec in specs
        }))

    def get_model_spec(self, api, name):
        api = _normalize_profile(api)
        if name is None:
            return None
        if not isinstance(name, str):
            raise TypeError("model name must be a string or None")
        return self._index.get((api, name.strip()))

    def list_model_specs(self, api=None):
        if api is None:
            return self.specs
        api = _normalize_profile(api)
        return tuple(spec for spec in self.specs if spec.endpoint.api == api)

    def matches(self, name, *, canonical_only=False):
        if name is None:
            return ()
        if not isinstance(name, str):
            raise TypeError("model name must be a string or None")
        name = name.strip()
        return tuple(spec for spec in self.specs if name == spec.name or (
            not canonical_only and name in spec.aliases
        ))

    def bind(self, api=None, name=None, *, request_params=None,
             endpoint_url=None, endpoint_model=None, endpoint_auth=None):
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("model name must be a string or None")
            name = name.strip() or None
        explicit = api is not None
        if api is None:
            matches = self.matches(name)
            if len(matches) > 1:
                raise ValueError("Ambiguous catalog model; select its API with --endpoint-api.")
            api = matches[0].endpoint.api if matches else "chat-completions"
        api = _normalize_profile(api)
        spec = self.get_model_spec(api, name)
        endpoint = (
            replace(_PROFILE_DEFAULT_ENDPOINTS[api], model=name)
            if spec is None else spec.endpoint
        )
        changes = {}
        if endpoint_url is not None:
            if endpoint_url != endpoint.url and endpoint.auth != "none" and endpoint_auth is None:
                raise ValueError("Changing a credentialed endpoint URL requires explicit --endpoint-auth.")
            changes["url"] = endpoint_url
        if endpoint_model is not None:
            changes["model"] = endpoint_model
        if endpoint_auth is not None:
            changes.update(auth=endpoint_auth, auth_file=None)
        endpoint = replace(endpoint, **changes)
        binding = ModelBinding(
            name, endpoint, spec,
            {} if spec is None else spec.request_params,
            "builtin" if spec is None else self.origins[(spec.endpoint.api, spec.name)],
            explicit,
        )
        return binding.with_request_params(request_params)


BUILTIN_MODEL_CATALOG = ModelCatalog(
    _MODEL_SPECS,
    {(spec.endpoint.api, spec.name): "builtin" for spec in _MODEL_SPECS},
)


def binding_from_namespace(args, catalog=None):
    """Prepared bindings are authoritative; raw args cannot reconfigure them."""
    existing = getattr(args, "model_binding", None)
    if isinstance(existing, ModelBinding):
        return existing
    if existing is not None:
        raise TypeError("model_binding must be ModelBinding or None")
    if catalog is None and getattr(args, "model_catalog", None) is not None:
        raise ValueError("Load and bind the explicit model catalog before resolving model configuration.")
    catalog = BUILTIN_MODEL_CATALOG if catalog is None else catalog
    auth = getattr(args, "endpoint_auth", None)
    supplied = getattr(args, "api_key", None) is not None
    inferred_supplied = supplied and auth is None
    if supplied:
        if auth is not None and auth != "supplied":
            raise ValueError("A supplied API key requires endpoint-auth supplied.")
        auth = "supplied"
    binding = catalog.bind(
        getattr(args, "model_api", None), getattr(args, "model", None),
        endpoint_url=getattr(args, "endpoint_url", None),
        endpoint_model=getattr(args, "endpoint_model", None),
        endpoint_auth=auth,
        request_params=getattr(args, "request_params", None),
    )
    if inferred_supplied and binding.api == "codex" and (
        binding.spec is None or binding.spec.endpoint.auth == "codex-login"
    ):
        raise ValueError(
            "--endpoint-api-key is not used with Codex login; select "
            "--endpoint-auth supplied explicitly."
        )
    return binding


def get_model_spec(profile: str, name: Optional[str]) -> Optional[ModelSpec]:
    """Look up an exact, case-sensitive selector; unknown models return None.

    Outer whitespace is stripped at the selector boundary. Generic Responses
    (``responses``) and Chat Completions do not inherit Codex presets.
    """
    profile = _normalize_profile(profile)
    if name is None:
        return None
    if not isinstance(name, str):
        raise TypeError("model name must be a string or None")
    return _MODEL_INDEX.get((profile, name.strip()))


def list_model_specs(profile: Optional[str] = None) -> Tuple[ModelSpec, ...]:
    """List canonical presets in declaration order; aliases do not duplicate entries."""
    if profile is None:
        return _MODEL_SPECS
    profile = _normalize_profile(profile)
    return tuple(spec for spec in _MODEL_SPECS if spec.endpoint.api == profile)


__all__ = [
    "ANTHROPIC_MESSAGES_API_URL",
    "CODEX_RESPONSES_API_URL",
    "META_RESPONSES_API_URL",
    "OPENAI_RESPONSES_API_URL",
    "ModelLimits",
    "ModelCatalog",
    "ModelBinding",
    "BUILTIN_MODEL_CATALOG",
    "EndpointSpec",
    "ModelSpec",
    "MessagesDefaults",
    "ResponsesDefaults",
    "get_model_spec",
    "list_model_specs",
]
