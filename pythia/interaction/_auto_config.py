"""Small fixed-role configuration resolver. It never loads credential values.

Main (#1) takes the unprefixed options. Every other role is configured like main
except for what it sets itself: its own model (a per-role option, a config-file
entry, a save, or a catalog [auto] default) and its other per-role settings. A
role on a different model inherits none of main's model-specific settings, and
main's connection settings only when it uses the same connection.
"""

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
PRIMARY = 1
ROLE_INDEX = {name: index for index, name in NAMES.items()}
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
# One per-role JSON format for --context-config files and saved config.json:
# {"version": 3, "main": {...}, "watcher": {...}, "worker": {...}}.
CONFIG_VERSION = 3
_KEYS = frozenset(DEFAULTS) | {"name"}
# Per-role identity; never inherited.
_IDENTITY = ("name", "instructions")
# The model and how to reach it. A config-file entry chooses its own model only
# with _MODEL_CHOICE keys; otherwise its route keys adjust main's connection.
_ROUTE = ("model_api", "model", "endpoint_url", "endpoint_model", "endpoint_auth",
          "codex_home", "codex_auth_file")
_MODEL_CHOICE = ("model_api", "model", "endpoint_url", "endpoint_model")
_CONNECTION = ("endpoint_url", "endpoint_auth", "codex_home", "codex_auth_file")
# Valid only for a particular model; never carried to another model.
_MODEL_SPECIFIC = ("extra_sample_params", "max_output_tokens", "auto_compact_tokens",
                   "max_context_tokens", "compaction_mode", "compaction_max_output_tokens")
_SHARED = ("cwd", "max_samples", "request_timeout_seconds", "enable_workspace",
           "enable_auto_compaction", "compaction_keep_recent_tokens")
assert _KEYS == frozenset(_IDENTITY + _ROUTE + _MODEL_SPECIFIC + _SHARED)
# Where a role's model came from, for display; FOLLOWS_MAIN roles save none.
FOLLOWS_MAIN = "same as main"
_APIS = {"chat-completions", "messages", "codex", "responses"}
_PROVIDER_FIELDS = ("model", "endpoint_url", "endpoint_model", "endpoint_auth",
                    "codex_home", "codex_auth_file")
_PATHS = ("cwd", "codex_home", "codex_auth_file")


class AutoSettings(dict):
    """Raw JSON settings plus non-serialized invocation metadata."""
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog
        # Role index -> where its model came from (command line, config file,
        # saved, catalog default, FOLLOWS_MAIN, or built-in default).
        self.sources = {}


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


def _chooses(layer):
    return any(key in layer for key in _MODEL_CHOICE)


def _switch_model(main, choice, catalog):
    """Main's settings on another model; nothing model-specific carries over.

    A name from an option or the catalog: a catalog model brings its own route,
    keeping main's endpoint options (a gateway URL, credentials) only when main's
    model has the same route; any other name is a model ID on main's endpoint. A
    route from a file or save is used as given, with main's model when it names
    none.
    """
    settings = dict(main)
    for key in _MODEL_SPECIFIC:
        settings[key] = DEFAULTS[key]
    if isinstance(choice, dict):
        for key in _ROUTE:
            settings[key] = choice.get(key, DEFAULTS[key])
        if "model" not in choice:
            # Main's model elsewhere: its name, and its API unless one is given.
            settings["model"] = main["model"]
            if "model_api" not in choice:
                settings["model_api"] = main["model_api"]
        return settings
    name = choice.strip()
    matches = catalog.matches(name)
    if len(matches) > 1:
        raise ValueError(f"Model {name!r} is in the catalog for several APIs; "
                         "select one with model_api in a config file.")
    endpoint = namespace(main, catalog).model_binding.endpoint
    settings["endpoint_model"] = None
    if matches:
        settings.update(model_api=None, model=name)
        target = matches[0].endpoint
        # Main's model's own route, before main's endpoint options.
        route = catalog.bind(main["model_api"], main["model"]).endpoint
        if (target.api, target.url, target.auth) != (route.api, route.url, route.auth):
            for key in _CONNECTION:
                settings[key] = None
        elif main["model_api"] == target.api:
            # Main's options apply as given, e.g. a custom Codex URL needs the API.
            settings["model_api"] = target.api
    else:
        settings.update(model_api=endpoint.api, model=name,
                        endpoint_url=endpoint.url, endpoint_auth=endpoint.auth)
    return settings


def _read_document(path, what):
    try:
        document = parse_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise ValueError(f"Could not load {what}.") from None
    if not isinstance(document, dict):
        raise ValueError(f"Invalid {what}.")
    return document


def _entries(document, base):
    """Version-3 per-role entries by role index, with paths made absolute."""
    unknown = set(document) - {"version"} - set(ROLE_INDEX)
    if unknown:
        raise ValueError("Auto configuration roles must be main, watcher, and worker.")
    entries = {}
    for name, index in ROLE_INDEX.items():
        if name in document:
            entries[index] = _layer(document[name], base)
    return entries


def load_config_file(path):
    """A --context-config file: {"version": 3, "main": {...}, "watcher": {...}}."""
    path = Path(path).expanduser().absolute()
    document = _read_document(path, "auto context configuration")
    version = document.get("version")
    if type(version) is not int or version != CONFIG_VERSION:
        raise ValueError(
            f"Expected auto configuration version {CONFIG_VERSION}, with one entry per "
            'role: {"version": 3, "main": {...}, "watcher": {...}}.'
        )
    # Relative paths are relative to the file.
    return _entries(document, path.parent)


def load_saved_config(path):
    """A save's config.json: main in full, other roles only what they set."""
    path = Path(path).expanduser().absolute()
    document = _read_document(path, "saved auto configuration")
    version = document.get("version")
    if type(version) is int and version < CONFIG_VERSION:
        raise ValueError("This save was created by an earlier version of auto and cannot "
                         "be resumed; start a new save.")
    try:
        if type(version) is not int or version != CONFIG_VERSION:
            raise ValueError()
        entries = _entries(document, path.parent)
    except ValueError:
        raise ValueError("Invalid saved auto configuration.") from None
    if (set(entries.get(PRIMARY, ())) != _KEYS
            or any(not set(_IDENTITY) <= set(entry) for entry in entries.values())):
        raise ValueError("Invalid saved auto configuration.")
    return entries


def resolve_config(path=None, overrides=None, saved=None, *, catalog=BUILTIN_MODEL_CATALOG,
                   role_models=None, role_defaults=None, roles=None):
    """Merge raw per-role inputs; runtime owners resolve catalog policy later.

    overrides holds the unprefixed options: main takes them all; their model is
    every role's model; the other model and endpoint options stay with main's
    model, which roles on the same model share; instructions are main's only.
    role_models holds per-role model options (--main-model, ...); role_defaults
    the catalog [auto] defaults, which seed new saves only; saved the entries of
    a resumed save; roles the role indices to resolve (default: all).

    Per role, most specific first: per-role model option, --model, config-file
    entry, save, catalog default, then main's model.
    """
    roles = tuple(NAMES) if roles is None else tuple(roles)
    if PRIMARY not in roles or set(roles) - set(NAMES):
        raise ValueError("Auto roles must include main.")
    entries = {} if path is None else load_config_file(path)
    launch = _layer(overrides or {}, Path.cwd())
    if "name" in launch:
        raise ValueError("Names must be configured per role.")
    main_instructions = launch.pop("instructions", None)
    main_instruction_override = overrides is not None and "instructions" in overrides
    role_models = dict(role_models or {})
    for index, name in role_models.items():
        if index not in roles or not isinstance(name, str) or not name.strip():
            raise ValueError("Per-role models must name a model for a running role.")
    # Catalog defaults seed new saves only; a resumed save keeps its own models.
    role_defaults = {} if saved is not None else dict(role_defaults or {})
    saved = saved or {}
    resolved = AutoSettings(catalog)

    entry, stored = entries.get(PRIMARY, {}), saved.get(PRIMARY, {})
    settings, source = {**DEFAULTS, "name": NAMES[PRIMARY]}, "built-in default"
    if stored:
        settings, source = _merge(settings, stored, catalog), "saved"
    elif PRIMARY in role_defaults and not (
            _chooses(entry) or _chooses(launch) or PRIMARY in role_models):
        settings = _merge(settings, {"model_api": None, "model": role_defaults[PRIMARY]}, catalog)
        source = "catalog default"
    if _chooses(entry):
        source = "config file"
    settings = _merge(settings, {k: v for k, v in entry.items() if k not in _IDENTITY}, catalog)
    # The command line wins over the file. --main-model replaces --model for main
    # within the same layer, so endpoint options given with it stay.
    options = dict(launch)
    if PRIMARY in role_models:
        options["model"] = role_models[PRIMARY]
    if _chooses(options):
        source = "command line"
    settings = _merge(settings, options, catalog)
    settings["name"] = entry.get("name", stored.get("name", NAMES[PRIMARY]))
    settings["instructions"] = (main_instructions if main_instruction_override
                                else entry.get("instructions", stored.get("instructions")))
    settings["cwd"] = str(Path(settings["cwd"]).expanduser().absolute())
    _validate(settings, catalog)
    resolved[PRIMARY], resolved.sources[PRIMARY] = settings, source
    main = settings

    for index in roles:
        if index == PRIMARY:
            continue
        entry, stored = entries.get(index, {}), saved.get(index, {})
        choice, source = None, FOLLOWS_MAIN
        for candidate, label in (
            (role_defaults.get(index), "catalog default"),
            ({k: stored[k] for k in _ROUTE if k in stored} if _chooses(stored) else None, "saved"),
            ({k: entry[k] for k in _ROUTE if k in entry} if _chooses(entry) else None, "config file"),
            (launch.get("model"), "command line"),
            (role_models.get(index), "command line"),
        ):
            if candidate is not None:
                choice, source = candidate, label
        if isinstance(choice, dict) and set(choice) == {"model"}:
            choice = choice["model"]  # A bare name in a file or save acts like an option.
        shares = choice is None or (
            isinstance(choice, str) and main["model"] is not None
            and choice.strip() == main["model"].strip())
        settings = dict(main) if shares else _switch_model(main, choice, catalog)
        for layer in (stored, entry):
            # A route comes whole from whatever chose the model. On main's model,
            # a file or save may still adjust main's connection for this role.
            own = {k: v for k, v in layer.items() if k not in _ROUTE and k not in _IDENTITY}
            if shares and not _chooses(layer):
                own.update({k: layer[k] for k in _CONNECTION if k in layer})
            settings = _merge(settings, own, catalog)
        # The command line wins over the file: shared options for every role,
        # model-specific ones for roles on main's model.
        options = {k: v for k, v in launch.items()
                   if k in _SHARED or (shares and k in _MODEL_SPECIFIC)}
        settings = _merge(settings, options, catalog)
        settings["name"] = entry.get("name", stored.get("name", NAMES[index]))
        settings["instructions"] = entry.get("instructions", stored.get("instructions"))
        settings["cwd"] = str(Path(settings["cwd"]).expanduser().absolute())
        _validate(settings, catalog)
        resolved[index], resolved.sources[index] = settings, source
    return resolved


def saved_document(settings, sources=None):
    """The version-3 document a save keeps: main in full, other roles only what
    they set (their whole model selection when they chose their own model)."""
    sources = getattr(settings, "sources", {}) if sources is None else sources
    main = settings[PRIMARY]
    document = {"version": CONFIG_VERSION, NAMES[PRIMARY]: dict(main)}
    for index, value in settings.items():
        if index == PRIMARY:
            continue
        own_model = sources.get(index, FOLLOWS_MAIN) != FOLLOWS_MAIN
        document[NAMES[index]] = {
            key: item for key, item in value.items()
            if key in _IDENTITY or (own_model and key in _ROUTE + _MODEL_SPECIFIC)
            or item != main[key]
        }
    return document


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
        raise ValueError("A model is required for Messages, Codex, and Responses.")
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
        description=("Fixed-role auto: #1 main takes user tasks and #-1 watcher supervises "
                     "main; the #2 worker and shared message board are experimental opt-ins "
                     "(local tools are unsandboxed). Options configure main, and every role "
                     "uses main's configuration unless it sets its own; --model sets every "
                     "role's model. Default role models can be set in the model catalog's "
                     "[auto] section."),
        allow_abbrev=False,
    )
    parser.add_argument("--context-config", type=Path,
                        help=("precise per-role settings: a version-3 JSON file with main, "
                              "watcher, and worker entries; options on the command line win"))
    parser.add_argument("--save", type=Path, default=Path("interaction-auto"),
                        help=("New or resumed save directory; existing paths require --resume "
                              "(default: interaction-auto)."))
    add_prompt_arguments(
        parser, allow_file=True,
        prompt_help=("Submit one user task to main (a fresh board thread with the "
                     "experimental worker/board); run without a TTY."),
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
        "--watcher-max-resumes", type=int, default=None, metavar="N",
        help=("follow-up messages the watcher may resume main with per task, from its "
              "own model turn after each main yield; 0 makes it observe-only, with no "
              "watcher model; not with the experimental worker/board "
              "(default: unlimited)"),
    )
    parser.add_argument(
        "--enable-experimental-worker-board", nargs="?", const=True, default=False,
        type=_boolean_argument, metavar="{False,True}",
        help=("experimental: also run the #2 worker and the shared message board "
              "(loopback HTTP board, board tools and instructions, plan delegation, "
              "and idle --headless task intake); otherwise main takes tasks "
              "directly. --resume requires the save's original setting; a bare "
              "flag means True (default: %(default)s)"),
    )
    parser.add_argument(
        "--enable-board-auth", nargs="?", const=True, default=True,
        type=_boolean_argument, metavar="{False,True}",
        help=("require bearer authentication for board data and HTML routes; "
              "False enables unsafe local debugging; requires the experimental "
              "worker/board (default: %(default)s)"),
    )
    parser.add_argument("--board-port", type=int, default=0,
                        help=("Loopback board port (0 chooses an available port); requires "
                              "the experimental worker/board."))
    add_endpoint_arguments(parser, auto=True)
    add_catalog_arguments(parser, suppress_extra_sample_params=True)
    parser.add_argument("--model", default=argparse.SUPPRESS, metavar="NAME",
                        help=("every role's model: a catalog name, or another model ID on "
                              "main's endpoint; the per-role options below win"))
    parser.add_argument("--main-model", metavar="NAME",
                        help="main's model only; roles without their own model follow it")
    parser.add_argument("--watcher-model", metavar="NAME",
                        help=("the watcher's model (default: the catalog [auto] default, "
                              "else main's); not with the experimental worker/board or "
                              "--watcher-max-resumes 0, where the watcher runs no model"))
    parser.add_argument("--worker-model", metavar="NAME",
                        help=("the worker's model (default: the catalog [auto] default, "
                              "else main's); requires the experimental worker/board"))
    parser.add_argument("--print-config", action="store_true",
                        help=("print each role's resolved model and the configuration a save "
                              "would keep, then exit without starting or saving"))
    parser.add_argument("--cwd", default=argparse.SUPPRESS)
    parser.add_argument("--instructions", default=argparse.SUPPRESS,
                        help="main's instructions; other roles keep their own")
    parser.add_argument(
        "--enable-workspace", nargs="?", const=True, default=argparse.SUPPRESS,
        type=_boolean_argument, metavar="{False,True}",
        help=("restrict exec_command workdir and apply_patch paths to --cwd; a bare "
              "flag means True (default: True)"),
    )
    parser.add_argument(
        "--enable-auto-compaction", nargs="?", const=True, default=argparse.SUPPRESS,
        type=_boolean_argument, metavar="{False,True}",
        help="enable automatic compaction; a bare flag means True (default: True)",
    )
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
