"""Explicit ChatGPT OAuth login; no model, terminal, or interaction-log ownership."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request

from ._account_http import AccountServiceError, request_json
from .codex_auth import CodexAuth
from .codex_auth import CodexCredentials
from .codex_auth import load_codex_credentials
from .timeouts import DEFAULT_LOGIN_TIMEOUT_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS


_ISSUER = "https://auth.openai.com"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
# Bound, advertised in redirect_uri, and required as the callback Host (RFC 8252 section 8.3).
_CALLBACK_HOST = "127.0.0.1"
_OAUTH_ERROR_CODE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_CALLBACK_FAILED_TEXT = "Sign-in failed. Return to the terminal for details."


def _token(payload, key):
    value = payload.get(key)
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise AccountServiceError("Login token response was incomplete or invalid.")
    return value


def _optional_token(payload, key, fallback):
    value = payload.get(key)
    if value is None:
        return fallback
    return _token(payload, key)


def _account_id(id_token):
    # The token comes from the trusted TLS token exchange, not command input.
    try:
        part = id_token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        account = payload["https://api.openai.com/auth"]["chatgpt_account_id"]
        if not isinstance(account, str) or not account or any(c.isspace() for c in account):
            raise ValueError
        return account
    except Exception:
        raise AccountServiceError("Login response did not identify a Codex account.") from None


def _callback_failure(params, state):
    """Return fixed (terminal, browser) text for a provider error callback.

    Only a validated OAuth error identifier is echoed. The description is consulted
    for the known entitlement marker but is never displayed or recorded.
    """
    errors = params.get("error", [])
    descriptions = params.get("error_description", [])
    code = errors[0] if len(errors) == 1 else None
    if code is not None and (not _OAUTH_ERROR_CODE.fullmatch(code) or state in code):
        code = None
    description = descriptions[0] if len(descriptions) == 1 else ""
    if code == "access_denied" and "missing_codex_entitlement" in description.lower():
        return ("Codex is not enabled for this workspace. Ask your workspace administrator "
                "for access, then run /login again.",
                "Codex is not enabled for your workspace. Contact your workspace "
                "administrator, then return to the terminal.")
    if code == "access_denied":
        return ("Login was declined by the provider.",
                "Sign-in was declined. Return to the terminal.")
    if code is not None:
        return f"Login failed: the provider returned OAuth error {code}.", _CALLBACK_FAILED_TEXT
    return "Login failed: the provider returned an unrecognized error.", _CALLBACK_FAILED_TEXT


def _save_credentials(path: Path, tokens: dict) -> None:
    temporary_name = None
    try:
        try:
            original = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(original, dict):
                raise ValueError
        except FileNotFoundError:
            original = {}
        # Keep unrelated fields, but replace the previous authentication mode.
        payload = {**original, "auth_mode": "chatgpt", "OPENAI_API_KEY": None,
                   "tokens": tokens, "last_refresh": datetime.now(timezone.utc).isoformat()}
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".auth.", suffix=".tmp", delete=False) as out:
            temporary_name = out.name
            os.chmod(out.name, 0o600)
            json.dump(payload, out, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    except Exception:
        raise AccountServiceError(
            "Credentials could not be saved; check the credential file and its directory."
        ) from None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def login(
    auth_file: Path, *, notify, cancel, workspace_id=None, expected_account=None,
    timeout_seconds=DEFAULT_LOGIN_TIMEOUT_SECONDS,
    request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    callback_port=1457, opener=None,
) -> None:
    """Write credentials after explicit login. Challenge text is transient only."""
    if cancel.is_set():
        raise AccountServiceError("Login cancelled.")
    if workspace_id and expected_account and workspace_id != expected_account:
        raise AccountServiceError("Account switching requires a fresh session.")
    state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    code = None
    failure = None

    class Callback(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            nonlocal code, failure
            status, message = 400, "Invalid login callback."
            try:
                parsed = urlsplit(self.path)
                params = parse_qs(parsed.query, max_num_fields=16)
                candidate = params.get("state", [])
                if (len(self.path) <= 8192 and self.headers.get("Host") == callback_host
                        and parsed.path == "/auth/callback"
                        and len(candidate) == 1
                        and secrets.compare_digest(candidate[0].encode(), state.encode())):
                    values = params.get("code", [])
                    if code is not None or failure is not None:
                        status, message = 409, "Callback already received."
                    elif "error" in params:
                        failure, message = _callback_failure(params, state)
                        status = 200
                    elif len(values) == 1 and values[0] and not any(c.isspace() for c in values[0]):
                        code = values[0]
                        status, message = 200, "Callback received. Check sign-in status in the terminal."
                    else:
                        # The provider answered this flow without a usable code; say so now
                        # instead of reporting a timeout at the deadline.
                        failure = ("Login failed: the provider callback did not include a usable "
                                   "authorization code.")
                        message = _CALLBACK_FAILED_TEXT
            except Exception:
                pass
            try:
                body = message.encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

    class Server(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(1.0)
            return connection, address

        def handle_error(self, request, client_address):
            # A malformed/slow callback must not dump request data to stderr.
            pass

    try:
        server = Server((_CALLBACK_HOST, callback_port), Callback)
    except OSError:
        raise AccountServiceError("Could not open the loopback login listener (port may be in use).") from None
    deadline = time.monotonic() + timeout_seconds
    callback_host = f"{_CALLBACK_HOST}:{server.server_port}"
    redirect_uri = f"http://{callback_host}/auth/callback"
    try:
        query = {
            "response_type": "code", "client_id": _CLIENT_ID, "redirect_uri": redirect_uri,
            "scope": "openid profile email offline_access api.connectors.read api.connectors.invoke",
            "code_challenge": challenge, "code_challenge_method": "S256",
            "id_token_add_organizations": "true", "codex_cli_simplified_flow": "true",
            "state": state, "originator": "codex_cli_rs",
        }
        if workspace_id or expected_account:
            query["allowed_workspace_id"] = workspace_id or expected_account
        notify(f"Open this URL to sign in (waiting up to {int(timeout_seconds)}s):\n"
               f"{_ISSUER}/oauth/authorize?{urlencode(query)}")
        server.timeout = 0.1
        while code is None and failure is None and not cancel.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if cancel.is_set():
        raise AccountServiceError("Login cancelled.")
    if failure is not None:
        raise AccountServiceError(failure)
    if code is None:
        raise AccountServiceError("Login timed out.")
    data = urlencode({"grant_type": "authorization_code", "client_id": _CLIENT_ID,
                      "code": code, "code_verifier": verifier, "redirect_uri": redirect_uri}).encode()
    response = request_json(Request(f"{_ISSUER}/oauth/token", data=data,
                                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                                    method="POST"),
                            min(request_timeout_seconds, max(0.001, deadline - time.monotonic())),
                            opener=opener)
    tokens = {key: _token(response, key) for key in ("access_token", "refresh_token", "id_token")}
    account = _account_id(tokens["id_token"])
    if any(expected and account != expected for expected in (expected_account, workspace_id)):
        raise AccountServiceError("Account mismatch; previous credentials were not replaced.")
    CodexAuth(tokens["access_token"], account)  # Validate before replacing any file.
    tokens["account_id"] = account
    if cancel.is_set():
        raise AccountServiceError("Login cancelled; credentials were not saved.")
    _save_credentials(auth_file, tokens)


def refresh_codex_credentials(
    credentials: CodexCredentials,
    *,
    timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    opener=None,
) -> CodexCredentials:
    """Refresh one same-account ChatGPT credential snapshot and persist it."""
    if not isinstance(credentials, CodexCredentials):
        raise TypeError("credentials must be CodexCredentials")
    current = load_codex_credentials(auth_file=credentials.auth_file)
    if current != credentials:
        if (
            credentials.auth.account_id is not None
            and current.auth.account_id != credentials.auth.account_id
        ):
            raise AccountServiceError(
                "Codex credential account changed before token refresh."
            )
        return current
    if credentials.refresh_token is None:
        raise AccountServiceError(
            "Codex credentials do not contain a refresh token; use /login."
        )
    request_body = json.dumps(
        {
            "client_id": _CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    response = request_json(
        Request(
            f"{_ISSUER}/oauth/token",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout_seconds,
        opener=opener,
    )
    access_token = _optional_token(
        response,
        "access_token",
        credentials.auth.access_token,
    )
    refresh_token = _optional_token(
        response,
        "refresh_token",
        credentials.refresh_token,
    )
    returned_id_token = response.get("id_token")
    id_token = _optional_token(response, "id_token", credentials.id_token)
    if returned_id_token is not None:
        account_id = _account_id(id_token)
    elif credentials.auth.account_id is not None:
        account_id = credentials.auth.account_id
    else:
        account_id = _account_id(id_token) if id_token is not None else None
    if (
        credentials.auth.account_id is not None
        and account_id != credentials.auth.account_id
    ):
        raise AccountServiceError(
            "Refreshed credentials identified a different Codex account."
        )
    CodexAuth(access_token, account_id)
    latest = load_codex_credentials(auth_file=credentials.auth_file)
    if latest != credentials:
        if (
            credentials.auth.account_id is not None
            and latest.auth.account_id != credentials.auth.account_id
        ):
            raise AccountServiceError(
                "Codex credential account changed during token refresh."
            )
        return latest
    try:
        current_payload = json.loads(
            credentials.auth_file.read_text(encoding="utf-8")
        )
        current_tokens = current_payload.get("tokens", {})
        if not isinstance(current_tokens, dict):
            current_tokens = {}
    except Exception:
        current_tokens = {}
    tokens = {
        **current_tokens,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "account_id": account_id,
    }
    if id_token is not None:
        tokens["id_token"] = id_token
    _save_credentials(credentials.auth_file, tokens)
    return load_codex_credentials(auth_file=credentials.auth_file)


__all__ = [
    "login",
    "refresh_codex_credentials",
]
