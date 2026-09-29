"""Canonical endpoint builders used by interaction adapter tests."""

from dataclasses import replace

from pythia.interaction import BUILTIN_MODEL_CATALOG
from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import CodexResponsesModel
from pythia.interaction import MessagesEndpoint
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction.codex_auth import _resolve_auth_file


def _full_url(prefix, suffix):
    return prefix.strip().rstrip("/") + suffix


def chat_endpoint(
    api_url="http://127.0.0.1:8000",
    model=None,
    *,
    api_key=None,
    binding=None,
    extra_sample_params=None,
    **kwargs,
):
    if binding is None:
        binding = BUILTIN_MODEL_CATALOG.bind(
            "chat-completions",
            model,
            endpoint_url=_full_url(api_url, "/v1/chat/completions"),
            endpoint_auth="supplied" if api_key is not None else "none",
            extra_sample_params=extra_sample_params,
        )
    return ChatCompletionsEndpoint(binding=binding, api_key=api_key, **kwargs)


def messages_endpoint(
    api_url,
    model,
    *,
    api_key=None,
    binding=None,
    **kwargs,
):
    if binding is None:
        binding = BUILTIN_MODEL_CATALOG.bind(
            "messages",
            model,
            endpoint_url=_full_url(api_url, "/v1/messages"),
            endpoint_auth="supplied" if api_key is not None else "none",
        )
    return MessagesEndpoint(binding=binding, api_key=api_key, **kwargs)


def responses_endpoint(
    api_url,
    model,
    bearer_token=None,
    *,
    account_id=None,
    api_provider="api",
    binding=None,
    **kwargs,
):
    if binding is None:
        if not isinstance(api_provider, str):
            raise TypeError("api_provider must be a string")
        api_provider = api_provider.strip().lower()
        if api_provider not in {"api", "codex"}:
            raise ValueError("api_provider must be api or codex")
        api = "codex" if api_provider == "codex" else "responses"
        binding = BUILTIN_MODEL_CATALOG.bind(
            api,
            model,
            endpoint_url=_full_url(api_url, "/responses"),
            endpoint_auth="supplied" if bearer_token is not None else "none",
        )
    return StreamingResponsesEndpoint(
        binding=binding,
        bearer_token=bearer_token,
        account_id=account_id,
        **kwargs,
    )


def codex_model(
    endpoint=None,
    *,
    model=None,
    auth=None,
    api_url=None,
    request_timeout_seconds=None,
    codex_home=None,
    auth_file=None,
    binding=None,
    **kwargs,
):
    if endpoint is not None:
        return CodexResponsesModel(endpoint=endpoint, **kwargs)
    if binding is None:
        binding = BUILTIN_MODEL_CATALOG.bind(
            "codex",
            model,
            endpoint_url=(
                None if api_url is None else _full_url(api_url, "/responses")
            ),
            endpoint_auth=(
                "supplied" if auth is not None
                else "codex-login" if codex_home is not None or auth_file is not None
                else None
            ),
        )
    if codex_home is not None or auth_file is not None:
        binding = replace(
            binding,
            endpoint=replace(
                binding.endpoint,
                auth="codex-login",
                auth_file=str(_resolve_auth_file(
                    codex_home=codex_home,
                    auth_file=auth_file,
                ).resolve()),
            ),
        )
    return CodexResponsesModel(
        binding=binding,
        auth=auth,
        request_timeout_seconds=request_timeout_seconds,
        **kwargs,
    )
