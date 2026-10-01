"""User-only tool adapters. Never register these with the model environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from ._account_http import AccountServiceError
from .codex_auth import CodexAuthError, _resolve_auth_file, load_codex_auth
from .codex_login import login
from .codex_quota import query_quota
from .environment import Environment, Tool, ToolOutcome, ToolSpec
from .model_config import supports_account_services
from .model_catalog import binding_from_namespace
from .runtime_config import CONFIG_KEYS
from .runtime_config import ConfigError
from .runtime_config import InteractionConfig
from .runtime_config import parse_config_literal
from .timeouts import DEFAULT_LOGIN_TIMEOUT_SECONDS


@dataclass(frozen=True)
class UserToolIntent:
    name: str
    arguments_json: str


def parse_user_tool(text: str) -> UserToolIntent:
    if "\n" in text or "\r" in text:
        raise ValueError("User-tool commands must be a single line.")
    words = text.split()
    if not words or words[0] not in {
        "/compact",
        "/config",
        "/config.json",
        "/login",
        "/quota",
    }:
        raise ValueError(
            "Unsupported command. Use /retry, /compact [focus], /config, "
            "/config.json, /login, /quota, /quit, or /exit."
        )
    command = words[0]
    if command in {"/config", "/config.json"}:
        if len(words) > 3:
            raise ValueError("Usage: /config[.json] [KEY [VALUE]].")
        arguments = {}
        if len(words) >= 2:
            key = words[1]
            if key not in CONFIG_KEYS:
                raise ValueError(
                    "Unknown config key; use /config to list supported keys."
                )
            arguments["key"] = key
            if len(words) == 3:
                try:
                    arguments["value"] = parse_config_literal(key, words[2])
                except ConfigError as exc:
                    raise ValueError(str(exc)) from None
        if command == "/config.json":
            arguments["format"] = "json"
        return UserToolIntent(
            "config",
            json.dumps(arguments, separators=(",", ":")),
        )

    name = command[1:]
    if name == "compact":
        # The rest of the line is focus text for the summary, as in pi.
        rest = text.split(maxsplit=1)
        focus = rest[1].strip() if len(rest) == 2 else ""
        arguments = {"instructions": focus} if focus else {}
    elif name == "quota":
        if len(words) != 1:
            raise ValueError(f"Usage: /{name} (no arguments).")
        arguments = {}
    else:
        if len(words) > 2 or (len(words) == 2 and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", words[1])):
            raise ValueError("Usage: /login [workspace-id]. Do not paste codes or tokens.")
        arguments = {"workspace_id": words[1]} if len(words) == 2 else {}
    return UserToolIntent(name, json.dumps(arguments, sort_keys=True))


def create_user_environment(
    args,
    *,
    notify,
    cancel,
    config=None,
    expected_account=None,
    provider_history=False,
    opener=None,
):
    """Create invocation-scoped adapters with a safe error boundary and notice sink.

    ``opener`` replaces the account-request opener for login and quota.
    """
    supported = supports_account_services(args)
    if config is None:
        config = InteractionConfig.from_namespace(args)
    if not isinstance(config, InteractionConfig):
        raise TypeError("config must be InteractionConfig")
    # Passed only when supplied, so default calls keep their exact signature.
    account_options = {} if opener is None else {"opener": opener}

    def guard(handler):
        def execute(arguments, *, timeout_seconds=None):
            try:
                if not supported:
                    return ToolOutcome("This configuration does not support ChatGPT account services.", False)
                if cancel.is_set():
                    return ToolOutcome("User tool cancelled before execution.", False)
                return handler(arguments, timeout_seconds)
            except CodexAuthError:
                return ToolOutcome("Model authentication needed; use /login.", False)
            except AccountServiceError as exc:
                return ToolOutcome(str(exc), False)
            except Exception:
                # The ordinary executor prints str(exc); do not let secrets reach it.
                return ToolOutcome("User tool failed; provider and credential details were withheld.", False)
        return execute

    def sign_in(arguments, timeout_seconds):
        if set(arguments) - {"workspace_id"}:
            return ToolOutcome("Unexpected login arguments.", False)
        workspace = arguments.get("workspace_id")
        if workspace is not None and (not isinstance(workspace, str) or
                                      not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", workspace)):
            return ToolOutcome("Invalid workspace ID. Do not supply authorization codes or tokens.", False)
        if provider_history and expected_account is None:
            return ToolOutcome("Cannot verify the account for saved provider state; start a fresh session to log in.", False)
        endpoint = binding_from_namespace(args).endpoint
        path = _resolve_auth_file(codex_home=None if endpoint.auth_file else args.codex_home,
                                  auth_file=endpoint.auth_file or args.codex_auth_file).resolve()
        login(path, notify=notify, cancel=cancel, workspace_id=workspace,
              expected_account=expected_account, timeout_seconds=timeout_seconds,
              request_timeout_seconds=args.request_timeout_seconds,
              **account_options)
        return ToolOutcome("Codex credentials saved. Model activation is handled separately.")

    def quota(arguments, timeout_seconds):
        if arguments:
            return ToolOutcome("Quota takes no arguments.", False)
        endpoint = binding_from_namespace(args).endpoint
        auth = load_codex_auth(codex_home=None if endpoint.auth_file else args.codex_home,
                               auth_file=endpoint.auth_file or args.codex_auth_file)
        if expected_account is not None and auth.account_id != expected_account:
            return ToolOutcome("Credential account changed; start a fresh session before using it.", False)
        return ToolOutcome(query_quota(auth, timeout_seconds=timeout_seconds,
                                       **account_options))

    def configure(arguments, timeout_seconds):
        del timeout_seconds
        try:
            if set(arguments) - {"key", "value", "format"}:
                raise ConfigError("Unexpected config arguments.")
            output_format = arguments.get("format", "python")
            if output_format not in {"python", "json"}:
                raise ConfigError("Config format must be python or json.")
            if "key" not in arguments:
                key = None
                if "value" in arguments:
                    raise ConfigError("A config value requires a key.")
            else:
                key = arguments["key"]
                if not isinstance(key, str) or key not in CONFIG_KEYS:
                    raise ConfigError(
                        "Unknown config key; use /config to list supported keys."
                    )
            if "value" in arguments:
                assert key is not None
                config.set(key, arguments["value"])
            return ToolOutcome(
                config.render(key, json_output=output_format == "json")
            )
        except ConfigError as exc:
            return ToolOutcome(str(exc), False)
        except Exception:
            return ToolOutcome("Config operation failed; details were withheld.", False)

    return Environment((
        Tool(ToolSpec("config", "Read or update in-memory interaction configuration.", {
            "type": "object",
            "properties": {
                "key": {"type": "string", "enum": list(CONFIG_KEYS)},
                "value": {},
                "format": {"type": "string", "enum": ["python", "json"]},
            },
            "additionalProperties": False,
        }), configure),
        Tool(ToolSpec("login", "Sign in to the selected ChatGPT/Codex account.", {
            "type": "object", "properties": {"workspace_id": {"type": "string"}},
            "additionalProperties": False,
        }), guard(sign_in), timeout_seconds=DEFAULT_LOGIN_TIMEOUT_SECONDS),
        Tool(ToolSpec("quota", "Query a historical Codex account quota snapshot.", {
            "type": "object", "properties": {}, "additionalProperties": False,
        }), guard(quota), timeout_seconds=args.request_timeout_seconds),
    ))
