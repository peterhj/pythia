"""Small fixed-role configuration resolver. It never loads credential values."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from urllib.parse import urlsplit

from ._auto_board import parse_json
from ._prompt import add_prompt_arguments
from .compaction import COMPACTION_MODES
from .model import SampleParams
from .model_config import _boolean_argument
from .model_config import add_catalog_arguments, add_endpoint_arguments, prepare_namespace
from .model_catalog import BUILTIN_MODEL_CATALOG, freeze_extra_sample_params, thaw_json
from .runtime_config import InteractionConfig
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS


NAMES = {1: "main", 2: "worker", -1: "watcher"}
DEFAULTS = {
    "model_api": None, "model": None,
    "endpoint_url": None, "endpoint_model": None, "endpoint_auth": None,
    "codex_home": None, "codex_auth_file": None,
    "cwd": ".", "max_samples": None, "max_output_tokens": None,
    "request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
    "enable_workspace": True, "enable_auto_compaction": True,
    "auto_compact_tokens": None, "max_context_tokens": None,
    "compaction_mode": None, "compaction_keep_recent_tokens": None,
    "compaction_max_output_tokens": None,
    "extra_sample_params": None,
    "instructions": None,
}
# Saved config.json versions and their exact per-context key sets. Version 1
# predates the compaction keys, which load with their defaults.
SAVED_CONFIG_VERSION = 2
_COMPACTION_KEYS = ("compaction_mode", "compaction_keep_recent_tokens",
                    "compaction_max_output_tokens")
_SAVED_KEYS = {
    1: frozenset(DEFAULTS).difference(_COMPACTION_KEYS) | {"name"},
    2: frozenset(DEFAULTS) | {"name"},
}
_APIS = {"chat-completions", "messages", "codex"}
_PROVIDER_FIELDS = ("model", "endpoint_url", "endpoint_model", "endpoint_auth",
                    "codex_home", "codex_auth_file")
_PATHS = ("cwd", "codex_home", "codex_auth_file")


class AutoSettings(dict):
    """Raw JSON settings plus non-serialized invocation metadata."""
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog


def _layer(value, base):
    if not isinstance(value, dict) or set(value) - (set(DEFAULTS) | {"name"}):
        raise ValueError("Invalid auto configuration fields (use credential references, not API keys).")
    value = dict(value)
    api = value.get("model_api")
    if api is not None and (not isinstance(api, str) or api not in _APIS):
        raise ValueError("Unsupported auto model API.")
    if value.get("extra_sample_params") is not None:
        value["extra_sample_params"] = thaw_json(freeze_extra_sample_params(value["extra_sample_params"]))
    for key in _PATHS:
        if key == "cwd" and key in value and value[key] is None:
            raise ValueError("cwd must be an existing directory path.")
        if key in value and value[key] is not None:
            raw = value[key]
            if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
                raise ValueError(f"Invalid {key} path.")
            path = Path(raw).expanduser()
            value[key] = str((base / path).absolute())
    return value


def _identity(settings, catalog):
    api, name = settings["model_api"], settings["model"]
    if api is None:
        matches = catalog.matches(name) if isinstance(name, str) else ()
        api = (matches[0].endpoint.api if len(matches) == 1
               else (None if matches else "chat-completions"))
    if api in _APIS and isinstance(name, str):
        spec = catalog.get_model_spec(api, name)
        name = spec.name if spec is not None else name.strip()
    return api, name


def _merge(current, value, catalog):
    current = dict(current)
    before = _identity(current, catalog)
    after = _identity({**current, **value}, catalog)
    def route_for(identity):
        api, name = identity
        if api in _APIS:
            spec = catalog.get_model_spec(api, name)
            return None if spec is None else spec.endpoint.connection_identity
        return None

    route_change = before != after and route_for(before) != route_for(after)
    if after[0] != before[0] or route_change:
        for key in _PROVIDER_FIELDS:
            # Clearing the API asks the catalog about this selector; retain it.
            if key == "model" and (
                current["model_api"] is None or value.get("model_api") is None
                or after[0] == before[0]
            ):
                continue
            current[key] = None
    if before != after:
        current["extra_sample_params"] = None
    if value.get("endpoint_auth") is not None:
        if value["endpoint_auth"] != "codex-login":
            current["codex_home"] = current["codex_auth_file"] = None
    previous_params = current.get("extra_sample_params") or {}
    current.update(value)
    if isinstance(value.get("extra_sample_params"), dict):
        current["extra_sample_params"] = {**previous_params, **value["extra_sample_params"]}
    return current


def resolve_config(path=None, overrides=None, saved=None, *, catalog=BUILTIN_MODEL_CATALOG):
    """Merge raw per-context inputs; runtime owners resolve catalog policy later."""
    document = {}
    base = Path.cwd()
    if path is not None:
        path = Path(path).expanduser().absolute()
        base = path.parent
        try:
            document = parse_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            raise ValueError("Could not load auto context configuration.") from None
        if (not isinstance(document, dict) or set(document) - {"version", "defaults", "contexts"}
                or type(document.get("version")) is not int or document["version"] != 1):
            raise ValueError("Expected auto configuration version 1.")
    contexts = document.get("contexts", {})
    if not isinstance(contexts, dict) or set(contexts) - {"1", "2", "-1"}:
        raise ValueError("Auto contexts must use keys 1, 2, and -1.")
    launch = _layer(overrides or {}, Path.cwd())
    if "name" in launch:
        raise ValueError("Names must be configured per context.")
    main_instructions = launch.pop("instructions", None)
    main_instruction_override = overrides is not None and "instructions" in overrides
    defaults = _layer(document.get("defaults", {}), base)
    if "name" in defaults:
        raise ValueError("Names must be configured per context.")
    resolved = AutoSettings(catalog)
    for index, name in NAMES.items():
        initial = {**DEFAULTS, "name": name} if saved is None else dict(saved[index])
        specific = _layer(contexts.get(str(index), {}), base)
        settings = initial
        for layer in (defaults, launch, specific):
            settings = _merge(settings, layer, catalog)
        if index == 1 and main_instruction_override and "instructions" not in contexts.get("1", {}):
            settings["instructions"] = main_instructions
        settings["cwd"] = str(Path(settings["cwd"]).expanduser().absolute())
        _validate(settings, catalog)
        resolved[index] = settings
    return resolved


def load_saved_config(path):
    path = Path(path).expanduser().absolute()
    try:
        document = parse_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise ValueError("Could not load saved auto configuration.") from None
    contexts = document.get("contexts") if isinstance(document, dict) else None
    version = document.get("version") if isinstance(document, dict) else None
    expected = _SAVED_KEYS.get(version) if type(version) is int else None
    if (not isinstance(document, dict)
            or set(document) != {"version", "contexts"}
            or expected is None
            or not isinstance(contexts, dict) or set(contexts) != {"1", "2", "-1"}
            or any(not isinstance(value, dict) or set(value) != expected
                   for value in contexts.values())):
        raise ValueError("Invalid saved auto configuration.")
    return {
        index: _layer({**{key: DEFAULTS[key] for key in _COMPACTION_KEYS},
                       **contexts[str(index)]}, path.parent)
        for index in NAMES
    }


def _validate(settings, catalog):
    api = settings["model_api"]
    if api is not None and (not isinstance(api, str) or api not in _APIS):
        raise ValueError("Unsupported auto model API.")
    model = settings["model"]
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a nonempty string or null.")
    args = namespace(settings, catalog)
    api = args.model_api
    if api != "chat-completions" and args.model is None:
        raise ValueError("A model is required for Messages/Codex.")
    name = settings["name"]
    if (not isinstance(name, str) or not name.strip() or len(name) > 64 or
            any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise ValueError("Context name must be a short single-line label.")
    instructions = settings["instructions"]
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("instructions must be text or null.")
    timeout = settings["request_timeout_seconds"]
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
            not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("request_timeout_seconds must be positive and finite.")
    samples = settings["max_samples"]
    if samples is not None and (type(samples) is not int or samples <= 0):
        raise ValueError("max_samples must be a positive integer or null.")
    url = settings["endpoint_url"]
    if url is not None:
        try:
            if not isinstance(url, str):
                raise ValueError()
            parsed = urlsplit(url)
            if (parsed.scheme not in {"http", "https"}
                    or not parsed.hostname or parsed.username is not None
                    or parsed.password is not None or parsed.query or parsed.fragment):
                raise ValueError()
            parsed.port  # Validate the numeric/range syntax without connecting.
        except (TypeError, ValueError):
            raise ValueError("endpoint_url must be an HTTP(S) URL without credentials/query/fragment.") from None
    if not Path(settings["cwd"]).is_dir():
        raise ValueError("Context cwd must be an existing directory.")
    SampleParams(max_output_tokens=settings["max_output_tokens"])
    # Reuse provider-aware max-output-token and runtime setting validation.
    InteractionConfig.from_namespace(args)


def namespace(settings, catalog=BUILTIN_MODEL_CATALOG, *, binding=None):
    args = argparse.Namespace(**settings, api_key=None)
    if binding is not None:
        args.model_binding = binding
    return prepare_namespace(args, catalog)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Board-first three-context auto MVP (local tools are unsandboxed).",
        allow_abbrev=False,
    )
    parser.add_argument("--context-config", type=Path, help="Version-1 JSON configuration for contexts 1, 2, -1.")
    parser.add_argument("--save", type=Path, default=Path("interaction-auto"),
                        help=("New or resumed save directory; existing paths require --resume "
                              "(default: interaction-auto)."))
    add_prompt_arguments(
        parser, allow_file=True,
        prompt_help="Post one user task as a fresh board thread; run without a TTY.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=("resume the auto save selected by --save; a missing directory starts fresh, "
              "and historical work is not replayed (default: %(default)s)"),
    )
    parser.add_argument(
        "--headless", nargs="?", const=True, default=False,
        type=_boolean_argument, metavar="{False,True}",
        help=("run without the TUI or context display; a bare flag means True "
              "and no TTY is required (default: %(default)s)"),
    )
    parser.add_argument(
        "--enable-board-auth", nargs="?", const=True, default=True,
        type=_boolean_argument, metavar="{False,True}",
        help=("require bearer authentication for board data and HTML routes; "
              "False enables unsafe local debugging (default: %(default)s)"),
    )
    parser.add_argument("--board-port", type=int, default=0, help="Loopback port (0 chooses an available port).")
    add_endpoint_arguments(parser, auto=True)
    add_catalog_arguments(parser, suppress_extra_sample_params=True)
    for key in ("model", "cwd", "instructions"):
        parser.add_argument("--" + key.replace("_", "-"), default=argparse.SUPPRESS)
    parser.add_argument("--max-samples", type=int, default=argparse.SUPPRESS,
                        help="Optional sample limit per turn; unlimited when unset.")
    parser.add_argument("--max-output-tokens", type=int, default=argparse.SUPPRESS,
                        help="Optional output-token limit; uses model/API defaults when unset.")
    parser.add_argument("--auto-compact-tokens", type=int, default=argparse.SUPPRESS,
                        help="Common initial compaction threshold; defaults to each context's catalog.")
    parser.add_argument("--compaction-mode", choices=COMPACTION_MODES, default=argparse.SUPPRESS,
                        help=("Common compaction procedure: pi, or provider for Codex remote or "
                              "Anthropic server-side compaction; defaults to each context's route."))
    parser.add_argument("--compaction-keep-recent-tokens", type=int, default=argparse.SUPPRESS,
                        help="Common recent context kept verbatim by pi compaction (default: 20000).")
    parser.add_argument("--compaction-max-output-tokens", type=int, default=argparse.SUPPRESS,
                        help="Common pi summary output budget; defaults to each turn's budget.")
    parser.add_argument("--max-context-tokens", type=int, default=argparse.SUPPRESS,
                        help="Common informational context ceiling; defaults to each context's catalog.")
    parser.add_argument("--request-timeout-seconds", type=float, default=argparse.SUPPRESS)
    return parser
