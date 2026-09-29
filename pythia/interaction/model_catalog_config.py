"""Explicit file I/O for the user catalog. Importing model_catalog stays pure."""

from __future__ import annotations

import configparser
from dataclasses import replace
from pathlib import Path
import re
import warnings

from ._config_file import read_config_bytes
from .model_catalog import BUILTIN_MODEL_CATALOG
from .model_catalog import ModelCatalog, ModelLimits, ModelSpec
from .model_catalog import EndpointSpec
from .model_catalog import ResponsesDefaults
from .model_catalog import _normalize_profile, parse_json_value


MAX_CATALOG_BYTES = 1_048_576
MAX_CATALOG_MODELS = 1024
# Version 3 replaced messages.output_effort with request_params.output_config.
LATEST_MODEL_CATALOG_VERSION = 3
_LIMIT_FIELDS = frozenset(("auto_compact_context_tokens", "max_context_tokens", "max_output_tokens"))
_RESPONSES_FIELDS = frozenset(("reasoning_effort", "reasoning_summary", "text_verbosity"))
_INTEGER = re.compile(r"^[+-]?[0-9]+$")


def default_model_catalog_path():
    return Path.home() / ".pythia" / "model-catalog.ini"


def _text(value):
    if value == "null":
        return None
    if value.startswith('"'):
        value = parse_json_value(value)
        if not isinstance(value, str):
            raise ValueError("Expected text.")
    return value


def _entry(section, values, base):
    name = section.removeprefix("model.")
    values = dict(values)
    override = values.pop("override", "false").lower()
    if override not in configparser.ConfigParser.BOOLEAN_STATES:
        raise ValueError("override must be a boolean.")
    override = configparser.ConfigParser.BOOLEAN_STATES[override]
    api_key = "endpoint.api"
    has_api = api_key in values
    api = _text(values.pop(api_key)) if has_api else None
    if has_api and api is None:
        raise ValueError("endpoint.api cannot be null.")
    if api is not None:
        api = _normalize_profile(api)
    if override:
        candidates = tuple(spec for spec in base.matches(name, canonical_only=True)
                           if api is None or spec.endpoint.api == api)
        if len(candidates) != 1:
            raise ValueError("Override requires one existing canonical model; specify endpoint.api if ambiguous.")
        original = candidates[0]
        api = original.endpoint.api
    else:
        if api is None:
            raise ValueError("New entries require endpoint.api.")
        if base.get_model_spec(api, name) is not None:
            raise ValueError("Existing entries require override = true.")
        original = None

    fields = {}
    endpoint = {} if original is None else dict(vars(original.endpoint))
    if api is not None:
        endpoint["api"] = api
    limits = {} if original is None else dict(vars(original.limits))
    responses = {} if original is None or original.responses is None else dict(vars(original.responses))
    params = {} if original is None else dict(original.request_params)
    if "request_params" in values and any(key.startswith("request_params.") for key in values):
        raise ValueError("Cannot combine whole-map and per-key request params.")
    for key, value in values.items():
        if key in {"endpoint.model", "endpoint.url", "endpoint.auth"}:
            endpoint[key[9:]] = _text(value)
            if key == "endpoint.auth":
                endpoint["auth_file"] = None
        elif key == "aliases":
            fields[key] = parse_json_value(value)
            if not isinstance(fields[key], list):
                raise ValueError("aliases must be a JSON array.")
        elif key == "source":
            fields[key] = _text(value)
        elif key.startswith("limits.") and key[7:] in _LIMIT_FIELDS:
            if value != "null" and _INTEGER.fullmatch(value) is None:
                raise ValueError("Limits require integers or null.")
            limits[key[7:]] = None if value == "null" else int(value)
        elif key.startswith("responses.") and key[10:] in _RESPONSES_FIELDS:
            responses[key[10:]] = _text(value)
        elif key == "request_params":
            params = parse_json_value(value)
        elif key.startswith("request_params."):
            params[key[15:]] = parse_json_value(value)
        else:
            raise ValueError("Unknown model field.")

    if original is None and not {"api", "url", "model", "auth"} <= endpoint.keys():
        raise ValueError("New entries require endpoint.api, url, model, and auth.")
    if (original is not None and "endpoint.url" in values
            and endpoint["url"] != original.endpoint.url and original.endpoint.auth != "none"
            and "endpoint.auth" not in values):
        raise ValueError("Changing a credentialed endpoint URL requires endpoint.auth.")
    endpoint = EndpointSpec(**endpoint)
    if endpoint.auth == "codex-login" and not endpoint.is_official_codex:
        raise ValueError("Catalog Codex login requires the official Codex endpoint.")
    fields.update(endpoint=endpoint, limits=ModelLimits(**limits), request_params=params)
    if responses:
        fields["responses"] = ResponsesDefaults(**responses)
    spec = (ModelSpec(name=name, **fields) if original is None
            else replace(original, **fields))
    return spec


def parse_model_catalog(text, *, base=BUILTIN_MODEL_CATALOG, source="user"):
    """Parse transactionally; omitted version metadata uses the latest schema."""
    if not isinstance(base, ModelCatalog):
        raise TypeError("base must be ModelCatalog")
    if not isinstance(text, str):
        raise TypeError("catalog text must be a string")
    if len(text.encode("utf-8")) > MAX_CATALOG_BYTES:
        raise ValueError("Model catalog exceeds the size limit.")
    parser = configparser.ConfigParser(
        interpolation=None, strict=True, delimiters=("=",), allow_no_value=False,
        inline_comment_prefixes=None, empty_lines_in_values=False,
    )
    parser.optionxform = str
    assumed_version = False
    try:
        parser.read_string(text)
        if parser.defaults():
            raise ValueError()
        if parser.has_section("catalog"):
            catalog_fields = set(parser["catalog"])
            if catalog_fields - {"version"}:
                raise ValueError()
            if "version" in catalog_fields:
                version = parser.getint("catalog", "version")
            else:
                version = LATEST_MODEL_CATALOG_VERSION
                assumed_version = True
        else:
            version = LATEST_MODEL_CATALOG_VERSION
            assumed_version = True
        sections = [section for section in parser.sections() if section != "catalog"]
        if len(sections) > MAX_CATALOG_MODELS or any(not section.startswith("model.") for section in sections):
            raise ValueError()
    except (configparser.Error, ValueError):
        raise ValueError(f"Invalid model catalog header/INI syntax: {source}") from None
    if version != LATEST_MODEL_CATALOG_VERSION:
        raise ValueError(
            f"Unsupported model catalog version {version} in {source}; "
            f"expected version {LATEST_MODEL_CATALOG_VERSION}"
        )
    specs = {(spec.endpoint.api, spec.name): spec for spec in base.specs}
    origins = dict(base.origins)
    seen = set()
    for section in sections:
        # Name the replacement rather than failing as a generic unknown field.
        if any(key.startswith("messages.") for key in parser[section]):
            raise ValueError(
                f"Invalid model catalog entry {section!r} in {source}: messages.* fields "
                'were removed in catalog version 3; use request_params.output_config = {"effort": ...}'
            )
        try:
            spec = _entry(section, parser[section], base)
            identity = (spec.endpoint.api, spec.name)
            if identity in seen:
                raise ValueError("Duplicate normalized model identity.")
            seen.add(identity)
            specs[identity] = spec
            origins[identity] = str(source)
        except (ValueError, TypeError, KeyError, RecursionError):
            # No values from request params/credentials should appear in errors.
            raise ValueError(f"Invalid model catalog entry {section!r} in {source}") from None
    try:
        catalog = ModelCatalog(tuple(specs.values()), origins)
    except ValueError:
        raise ValueError(f"Model catalog selector/alias collision in {source}") from None
    if assumed_version:
        warnings.warn(
            f"Model catalog {source!r} does not specify a version; assuming latest "
            f"supported version {version}.",
            UserWarning,
            stacklevel=2,
        )
    return catalog


def load_model_catalog(path=None, *, enabled=True, base=BUILTIN_MODEL_CATALOG):
    """Only this opt-in boundary reads the default home file; a missing default is OK."""
    if not enabled:
        if path is not None:
            raise ValueError("Cannot select and disable the user model catalog.")
        return base
    explicit = path is not None
    path = (default_model_catalog_path() if path is None else Path(path)).expanduser().absolute()
    try:
        data = read_config_bytes(path, MAX_CATALOG_BYTES)
        text = data.decode("utf-8")
    except FileNotFoundError:
        if not explicit:
            return base
        raise ValueError(f"Model catalog file not found: {path}") from None
    except (OSError, UnicodeError):
        raise ValueError(f"Could not read model catalog: {path}") from None
    return parse_model_catalog(text, base=base, source=str(path))


__all__ = [
    "LATEST_MODEL_CATALOG_VERSION",
    "default_model_catalog_path",
    "load_model_catalog",
    "parse_model_catalog",
]
