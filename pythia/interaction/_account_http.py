"""Bounded, non-redirecting account requests with secret-safe diagnostics."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ._http import USER_AGENT


class AccountServiceError(ValueError):
    """Only application-authored, credential-free messages belong here."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def default_account_opener():
    """Return the non-redirecting opener used by account requests by default."""
    return urllib.request.build_opener(_NoRedirect()).open


def _safe_header(headers, name):
    try:
        value = headers.get(name)
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if (
        not value
        or len(value) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return None
    return value


def _http_diagnostic_suffix(headers):
    request_id = _safe_header(headers, "x-request-id") or _safe_header(
        headers,
        "x-oai-request-id",
    )
    cf_ray = _safe_header(headers, "cf-ray")
    fields = []
    if request_id is not None:
        fields.append(f"request_id={request_id}")
    if cf_ray is not None:
        fields.append(f"cf_ray={cf_ray}")
    return "" if not fields else f" ({' '.join(fields)})"


def request_json(request, timeout_seconds, *, opener=None):
    if request.get_header("User-agent") is None:
        request.add_header("User-Agent", USER_AGENT)
    open_request = opener or default_account_opener()
    try:
        response = open_request(request, timeout=timeout_seconds)
        try:
            if not 200 <= getattr(response, "status", 200) < 300:
                raise AccountServiceError("Account service returned an unsuccessful response.")
            body = response.read(1_048_577)
        finally:
            response.close()
        if len(body) > 1_048_576:
            raise AccountServiceError("Account service response was too large.")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise AccountServiceError("Account service returned invalid data.")
        return value
    except urllib.error.HTTPError as exc:
        suffix = _http_diagnostic_suffix(exc.headers)
        exc.close()
        raise AccountServiceError(
            f"Account service HTTP {exc.code}{suffix}."
        ) from None
    except AccountServiceError:
        raise
    except Exception:
        # Do not echo response bodies, request URLs, headers, or arbitrary errors.
        raise AccountServiceError("Account service request failed or returned invalid data.") from None
