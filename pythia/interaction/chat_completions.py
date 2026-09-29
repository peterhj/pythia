from __future__ import annotations

import http.client
import json
import math
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from ._http import USER_AGENT
from ._transport_retry import DEFAULT_MAX_TRANSIENT_RETRIES
from ._transport_retry import retry_delay_seconds
from .context import ContextValidationError
from .context import InteractionContext
from .items import CompactionMetadata
from .items import ContextPrefix
from .items import Init
from .items import Instructions
from .items import InteractionItem
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import ToolResult
from .items import SampleMetadata
from .items import TextPart
from .items import TurnSummary
from .items import UserInteractionBoundary
from .model import ModelConfigurationError
from .model import ModelContextWindowError
from .model import ModelError
from .model import ModelResponseError
from .model import ModelSample
from .model import ModelTimeoutError
from .model import ModelTransportError
from .model import SampleParams
from .model import _apply_extra_sample_params
from .model_catalog import ModelBinding
from .model import TokenUsage
from .model import _timed_sample
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ChatCompletionsEndpoint:
    binding: ModelBinding = field(repr=False)
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    api_key: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ModelBinding):
            raise TypeError("binding must be ModelBinding")
        if self.binding.api != "chat-completions":
            raise ModelConfigurationError(
                "Chat Completions endpoint requires a chat-completions binding"
            )

        if self.api_key is not None:
            if not isinstance(self.api_key, str):
                raise TypeError("api_key must be a string or None")
            api_key = self.api_key.strip()
            if not api_key:
                raise ModelConfigurationError("api_key must not be empty")
            if any(character.isspace() for character in api_key):
                raise ModelConfigurationError(
                    "api_key must not contain whitespace"
                )
            object.__setattr__(self, "api_key", api_key)

        timeout = self.request_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ModelConfigurationError(
                "request_timeout_seconds must be positive and finite"
            )
        object.__setattr__(self, "request_timeout_seconds", float(timeout))
        if ((self.binding.endpoint.auth == "none") != (self.api_key is None)):
            raise ModelConfigurationError("Credentials do not match the resolved endpoint auth policy")

    @property
    def model(self) -> Optional[str]:
        return self.binding.endpoint.model

    @property
    def url(self) -> str:
        return self.binding.endpoint.url


def _append_text(existing: Optional[str], value: str, separator: str = "") -> str:
    if not existing:
        return value
    if not value:
        return existing
    return f"{existing}{separator}{value}"


def _encode_context_messages(
    items: Sequence[InteractionItem],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    assistant: Optional[Dict[str, Any]] = None

    def ensure_assistant() -> Dict[str, Any]:
        nonlocal assistant
        if assistant is None:
            assistant = {"role": "assistant"}
        return assistant

    def flush_assistant() -> None:
        nonlocal assistant
        if assistant is None:
            return
        if "content" not in assistant:
            assistant["content"] = None if assistant.get("tool_calls") else ""
        messages.append(assistant)
        assistant = None

    # Last-wins Instructions: emit effective once at front (system role).
    # Empty/whitespace-only text is preserved; absence means no system item.
    # model_items() already collapses+hoists, this is defensive for direct
    # encoder calls.
    effective: Optional[Instructions] = None
    for item in items:
        if isinstance(item, Instructions):
            effective = item
    if effective is not None:
        messages.append({"role": "system", "content": effective.text})

    for index, item in enumerate(items):
        if isinstance(
            item,
            (
                ModelSampleBoundary,
                ModelFailure,
                Init,
                SampleMetadata,
                CompactionMetadata,
                TurnSummary,
                UserInteractionBoundary,
            ),
        ):
            flush_assistant()
            continue

        if isinstance(item, Instructions):
            # Superseded or already emitted above; invisible to encoding.
            continue

        if isinstance(item, Message):
            if item.role == "assistant":
                pending = ensure_assistant()
                pending["content"] = _append_text(
                    pending.get("content"),
                    item.content,
                )
                continue
            if item.role not in {"system", "developer", "user"}:
                raise ModelConfigurationError(
                    f"unsupported message role at item {index}: {item.role!r}"
                )
            flush_assistant()
            if isinstance(item.content, str):
                messages.append({"role": item.role, "content": item.content})
            else:
                # Non-text content is user-only (constructor-enforced); the
                # OpenAI chat format uses text/image_url parts here.
                if item.role != "user":
                    raise ModelConfigurationError(
                        "media content requires the user role at item "
                        f"{index}: {item.role!r}"
                    )
                # TODO: an attachment-only message emits an image_url-only
                # array. If a provider rejects that, add a leading empty
                # {"type": "text", "text": ""} part (waiting on a real error).
                messages.append(
                    {
                        "role": item.role,
                        "content": [
                            (
                                {"type": "text", "text": part.text}
                                if isinstance(part, TextPart)
                                else {
                                    "type": "image_url",
                                    "image_url": {"url": part.source_uri},
                                }
                            )
                            for part in item.content
                        ],
                    }
                )
            continue

        if isinstance(item, Reasoning):
            pending = ensure_assistant()
            reasoning_text = item.content or "\n".join(item.summary)
            pending["reasoning_content"] = _append_text(
                pending.get("reasoning_content"),
                reasoning_text,
                separator="\n",
            )
            continue

        if isinstance(item, ToolCall):
            pending = ensure_assistant()
            tool_calls = pending.setdefault("tool_calls", [])
            if not isinstance(tool_calls, list):
                raise ModelConfigurationError(
                    "assistant tool_calls accumulator must be a list"
                )
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments_json,
                    },
                }
            )
            continue

        if isinstance(item, ToolResult):
            flush_assistant()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": item.output,
                }
            )
            continue

        if isinstance(item, OpaqueCompaction):
            raise ModelConfigurationError(
                "Chat Completions cannot encode OpaqueCompaction"
            )

        if isinstance(item, ContextPrefix):
            raise ModelConfigurationError(
                "ContextPrefix must be projected before request encoding"
            )

        raise ModelConfigurationError(
            f"unsupported interaction item at index {index}: {item!r}"
        )

    flush_assistant()
    if not messages:
        raise ModelConfigurationError("cannot sample an empty model context")
    return messages


def _encode_tools(tools: Sequence[Any]) -> List[Dict[str, Any]]:
    encoded: List[Dict[str, Any]] = []
    seen = set()
    for index, tool in enumerate(tools):
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        parameters = getattr(tool, "parameters", None)
        if not isinstance(name, str) or not name.strip():
            raise ModelConfigurationError(
                f"tool {index} must have a non-empty name"
            )
        if name in seen:
            raise ModelConfigurationError(f"duplicate tool name: {name!r}")
        seen.add(name)
        if not isinstance(description, str):
            raise ModelConfigurationError(
                f"tool {name!r} must have a string description"
            )
        if not isinstance(parameters, Mapping):
            raise ModelConfigurationError(
                f"tool {name!r} parameters must be a mapping"
            )
        encoded.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": dict(parameters),
                },
            }
        )
    return encoded


def _apply_sample_params(
    payload: Dict[str, Any],
    sample_params: Optional[SampleParams],
) -> None:
    if sample_params is None:
        return
    if sample_params.max_output_tokens is not None:
        payload["max_tokens"] = sample_params.max_output_tokens
    if sample_params.temperature is not None:
        payload["temperature"] = sample_params.temperature
    if sample_params.top_p is not None:
        payload["top_p"] = sample_params.top_p
    if sample_params.stop:
        payload["stop"] = list(sample_params.stop)
    if sample_params.seed is not None:
        payload["seed"] = sample_params.seed


def _coerce_content_text(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        parts: List[str] = []
        for index, part in enumerate(value):
            if not isinstance(part, Mapping):
                raise ModelResponseError(
                    f"{field_name}[{index}] must be an object"
                )
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            content = part.get("content")
            if isinstance(content, str):
                parts.append(content)
        return "".join(parts)
    raise ModelResponseError(f"{field_name} must be text, a content list, or null")


def _extract_reasoning_and_content(message: Mapping[str, Any]) -> Tuple[str, str]:
    content = _coerce_content_text(message.get("content"), "message.content")

    # Provider field names vary.  Prefer a non-empty "reasoning" value, but
    # retain compatibility with inference engines that still use
    # "reasoning_content".  A present-but-empty field must not hide the other.
    reasoning = ""
    for field_name in ("reasoning", "reasoning_content"):
        reasoning_value = message.get(field_name)
        if reasoning_value is None:
            continue
        reasoning = _coerce_content_text(
            reasoning_value,
            f"message.{field_name}",
        )
        if reasoning:
            break

    if not reasoning:
        match = _THINK_RE.search(content)
        if match is not None:
            reasoning = match.group(1).strip()
            content = f"{content[:match.start()]}{content[match.end():]}".strip()
    return reasoning, content


def _decode_tool_calls(raw_tool_calls: Any) -> Tuple[ToolCall, ...]:
    if raw_tool_calls is None:
        return ()
    if not isinstance(raw_tool_calls, Sequence) or isinstance(
        raw_tool_calls,
        (str, bytes, bytearray),
    ):
        raise ModelResponseError("message.tool_calls must be a list")

    calls: List[ToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, Mapping):
            raise ModelResponseError(f"tool call {index} must be an object")
        call_id = raw_call.get("id", raw_call.get("call_id"))
        function = raw_call.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments", "")
        else:
            name = raw_call.get("name")
            arguments = raw_call.get("arguments", "")

        if not isinstance(call_id, str) or not call_id.strip():
            raise ModelResponseError(f"tool call {index} has no call id")
        if not isinstance(name, str) or not name.strip():
            raise ModelResponseError(f"tool call {index} has no name")
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise ModelResponseError(
                    f"tool call {index} arguments are not JSON-compatible"
                ) from exc

        calls.append(
            ToolCall(
                name=name,
                call_id=call_id,
                arguments_json=arguments,
            )
        )
    return tuple(calls)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _decode_usage(raw_usage: Any) -> TokenUsage:
    if not isinstance(raw_usage, Mapping):
        return TokenUsage()
    input_tokens = _nonnegative_int(raw_usage.get("prompt_tokens"))
    output_tokens = _nonnegative_int(raw_usage.get("completion_tokens"))
    total_tokens = _nonnegative_int(raw_usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    cached_input_tokens = 0
    details = raw_usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        cached_input_tokens = min(
            input_tokens,
            _nonnegative_int(details.get("cached_tokens")),
        )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _map_finish_reason(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelResponseError("finish_reason must be a string or null")
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "tool_call": "tool_use",
        "content_filter": "refusal",
    }.get(value, value)


def _decode_response(payload: Any) -> ModelSample:
    if not isinstance(payload, Mapping):
        raise ModelResponseError("Chat Completions response must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or isinstance(
        choices,
        (str, bytes, bytearray),
    ) or not choices:
        raise ModelResponseError("Chat Completions response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ModelResponseError("choices[0] must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ModelResponseError("choices[0].message must be an object")

    reasoning, content = _extract_reasoning_and_content(message)
    calls = _decode_tool_calls(message.get("tool_calls"))
    items: List[InteractionItem] = []
    if reasoning:
        items.append(Reasoning(content=reasoning))
    if content or not calls:
        items.append(Message(role="assistant", content=content))
    items.extend(calls)

    return ModelSample(
        items=tuple(items),
        stop_reason=_map_finish_reason(choice.get("finish_reason")),
        usage=_decode_usage(payload.get("usage")),
    )


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read()
    except Exception:
        return ""
    if not isinstance(payload, (bytes, bytearray)):
        return ""
    return bytes(payload).decode("utf-8", errors="replace")


def _is_context_window_error(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "context window",
            "maximum context length",
            "context length exceeded",
            "too many tokens",
        )
    )


def _chat_http_failure(
    status: int,
    detail: str,
    *,
    model: Optional[str],
    attempt_count: int,
    recovery: Tuple[str, ...],
) -> ModelError:
    context_window = _is_context_window_error(detail)
    message = (
        f"Chat Completions HTTP {status}: {detail}"
        if detail
        else f"Chat Completions HTTP {status}: request failed"
    )
    failure = ModelFailure(
        category="context_window" if context_window else "http_error",
        message=(
            f"Chat Completions HTTP {status}: context window exceeded"
            if context_window
            else f"Chat Completions HTTP {status}: request failed"
        ),
        provider="chat-completions",
        model=model,
        http_status=status,
        attempt_count=attempt_count,
        recovery=recovery,
    )
    error_type = ModelContextWindowError if context_window else ModelTransportError
    return error_type(message, failure=failure)


def _chat_transport_failure(
    *,
    timeout: bool,
    model: Optional[str],
    attempt_count: int,
    recovery: Tuple[str, ...],
) -> ModelTransportError:
    message = (
        "Chat Completions request timed out"
        if timeout
        else "Chat Completions request failed before a response was completed"
    )
    failure = ModelFailure(
        category="request_timeout" if timeout else "request_transport",
        message=message,
        provider="chat-completions",
        model=model,
        attempt_count=attempt_count,
        recovery=recovery,
    )
    error_type = ModelTimeoutError if timeout else ModelTransportError
    return error_type(message, failure=failure)


def _close_response(response: Any) -> None:
    if response is None:
        return
    close = getattr(response, "close", None)
    if callable(close):
        close()


class ChatCompletionsModel:
    auto_compaction_owner = "host"

    def __init__(
        self,
        endpoint: ChatCompletionsEndpoint,
        *,
        opener: Optional[Callable[..., Any]] = None,
        retry_sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        if not isinstance(endpoint, ChatCompletionsEndpoint):
            raise TypeError("endpoint must be ChatCompletionsEndpoint")
        if retry_sleep is not None and not callable(retry_sleep):
            raise TypeError("retry_sleep must be callable or None")
        self.endpoint = endpoint
        self.binding = endpoint.binding
        self._opener = opener or urllib.request.urlopen
        self._retry_sleep = time.sleep if retry_sleep is None else retry_sleep

    def _build_request_payload(
        self,
        context: InteractionContext,
        tools: Sequence[Any],
        sample_params: Optional[SampleParams],
    ) -> Dict[str, Any]:
        if not isinstance(context, InteractionContext):
            raise TypeError("context must be InteractionContext")
        context.assert_model_ready()
        payload: Dict[str, Any] = {
            "messages": _encode_context_messages(context.model_items()),
            "stream": False,
        }
        if self.binding.endpoint.model is not None:
            payload["model"] = self.binding.endpoint.model
        encoded_tools = _encode_tools(tools)
        if encoded_tools:
            payload["tools"] = encoded_tools
            payload["parallel_tool_calls"] = False
        _apply_sample_params(payload, sample_params)
        _apply_extra_sample_params(payload, self.binding, sample_params)
        return payload

    @_timed_sample
    def sample(
        self,
        context: InteractionContext,
        *,
        tools: Sequence[Any] = (),
        sample_params: Optional[SampleParams] = None,
    ) -> ModelSample:
        if sample_params is not None and not isinstance(sample_params, SampleParams):
            raise TypeError("sample_params must be SampleParams or None")
        payload = self._build_request_payload(context, tools, sample_params)
        try:
            request_data = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ModelConfigurationError(
                "Chat Completions request is not JSON-serializable"
            ) from exc
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.endpoint.api_key is not None:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        attempts = 0
        retries = 0
        recovery: List[str] = []
        while True:
            attempts += 1
            request = urllib.request.Request(
                self.endpoint.url,
                data=request_data,
                headers=headers,
                method="POST",
            )
            response = None
            try:
                try:
                    response = self._opener(
                        request,
                        timeout=self.endpoint.request_timeout_seconds,
                    )
                except urllib.error.HTTPError as exc:
                    body = _read_http_error_body(exc)
                    detail = body or str(exc)
                    status = exc.code
                    response_headers = exc.headers
                    exc.close()
                    if (
                        status in _RETRYABLE_HTTP_STATUSES
                        and not _is_context_window_error(detail)
                        and retries < DEFAULT_MAX_TRANSIENT_RETRIES
                    ):
                        retries += 1
                        recovery.append(f"http_{status}_retry")
                        self._retry_sleep(
                            retry_delay_seconds(retries, response_headers)
                        )
                        continue
                    raise _chat_http_failure(
                        status,
                        detail,
                        model=self.endpoint.model,
                        attempt_count=attempts,
                        recovery=tuple(recovery),
                    ) from exc

                status = getattr(response, "status", None)
                response_headers = getattr(response, "headers", None)
                raw = response.read()
                if not isinstance(raw, (bytes, bytearray)):
                    raise ModelResponseError("HTTP response body must be bytes")
                if isinstance(status, int) and not 200 <= status < 300:
                    detail = bytes(raw).decode("utf-8", errors="replace")
                    if (
                        status in _RETRYABLE_HTTP_STATUSES
                        and not _is_context_window_error(detail)
                        and retries < DEFAULT_MAX_TRANSIENT_RETRIES
                    ):
                        retries += 1
                        recovery.append(f"http_{status}_retry")
                        _close_response(response)
                        response = None
                        self._retry_sleep(
                            retry_delay_seconds(retries, response_headers)
                        )
                        continue
                    raise _chat_http_failure(
                        status,
                        detail,
                        model=self.endpoint.model,
                        attempt_count=attempts,
                        recovery=tuple(recovery),
                    )

                try:
                    text = bytes(raw).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ModelResponseError("response is not UTF-8 JSON") from exc
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ModelResponseError("response is not valid JSON") from exc
                try:
                    sample = _decode_response(decoded)
                except ContextValidationError as exc:
                    raise ModelResponseError(str(exc)) from exc
                except ModelResponseError:
                    raise
                except (TypeError, ValueError) as exc:
                    raise ModelResponseError(str(exc)) from exc
                return replace(
                    sample,
                    request_attempts=attempts,
                    recovery=tuple(recovery),
                )
            except urllib.error.URLError as exc:
                timeout = isinstance(
                    exc.reason,
                    (TimeoutError, socket.timeout),
                )
                if retries < DEFAULT_MAX_TRANSIENT_RETRIES:
                    retries += 1
                    recovery.append(
                        "request_timeout_retry"
                        if timeout
                        else "connection_retry"
                    )
                    _close_response(response)
                    response = None
                    self._retry_sleep(retry_delay_seconds(retries))
                    continue
                raise _chat_transport_failure(
                    timeout=timeout,
                    model=self.endpoint.model,
                    attempt_count=attempts,
                    recovery=tuple(recovery),
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                if retries < DEFAULT_MAX_TRANSIENT_RETRIES:
                    retries += 1
                    recovery.append("request_timeout_retry")
                    _close_response(response)
                    response = None
                    self._retry_sleep(retry_delay_seconds(retries))
                    continue
                raise _chat_transport_failure(
                    timeout=True,
                    model=self.endpoint.model,
                    attempt_count=attempts,
                    recovery=tuple(recovery),
                ) from exc
            except (OSError, http.client.HTTPException) as exc:
                if retries < DEFAULT_MAX_TRANSIENT_RETRIES:
                    retries += 1
                    recovery.append("connection_retry")
                    _close_response(response)
                    response = None
                    self._retry_sleep(retry_delay_seconds(retries))
                    continue
                raise _chat_transport_failure(
                    timeout=False,
                    model=self.endpoint.model,
                    attempt_count=attempts,
                    recovery=tuple(recovery),
                ) from exc
            finally:
                _close_response(response)


__all__ = [
    "ChatCompletionsEndpoint",
    "ChatCompletionsModel",
]
