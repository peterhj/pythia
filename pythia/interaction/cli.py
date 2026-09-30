"""Caller-owned interaction loop with a scrollback POSIX terminal shell.

Run with ``python3 -m pythia.interaction.cli``. ``--headless`` runs one explicit
task without a terminal. Line-shell input and active-effect interruption are
deliberately deferred.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from collections.abc import Iterable
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import urllib.request
import uuid
from typing import Optional
from typing import Sequence
from typing import Union

from ._account_http import default_account_opener
from ._cli_editor import Editor
from ._cli_editor import safe_text
from ._cli_terminal import PosixTerminal
from ._debug_trace import DebugTrace
from ._debug_trace import trace_operation
from ._prompt import load_prompt
from .codex_auth import CodexAuthUnavailable
from .compaction import CompactionError
from .compaction import CompactionResult
from .compaction import create_default_compactor
from .compaction import should_auto_compact
from .compaction import uses_host_auto_compaction
from .context import InteractionContext
from .default_environment import DefaultEnvironment
from .display import DisplayItem
from .display import render_interaction_items
from .environment import Environment
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
from .items import TurnSummary
from .items import UserInteractionBoundary
from .items import UserToolCall
from .items import UserToolResult
from .items import summarize_turn_usage
from .media import AttachmentError
from .media import parse_user_prompt
from .model import Model
from .model import ModelAuthenticationError
from .model import ModelError
from .model import SampleParams
from .model_config import DEFAULT_SAVE_PATH
from .model_config import _boolean_argument
from .model_config import build_model
from .model_config import build_parser
from .model_config import initial_model_name
from .model_config import resolve_save_path
from .model_config import supports_account_services
from .model_config import frontend_catalog, prepare_namespace, render_model_catalog
from ._model_binding_debug import debug_model_binding_path
from ._model_binding_debug import save_debug_model_bindings
from .runtime_config import InteractionConfig
from .save import load_interaction_save
from .save import save_interaction_save
from .user import UserInteraction
from .user_tools import UserToolIntent
from .user_tools import create_user_environment
from .user_tools import parse_user_tool


FRAME_INTERVAL = 1 / 128
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_MAX_PENDING_QUERIES = 8


@dataclass(frozen=True, eq=False)
class _RetryIntent:
    """Identity ticket for one live sampling failure, never model input."""


@dataclass
class _UIState:
    editor: Editor = field(default_factory=Editor)
    pending: deque[Union[str, UserToolIntent, _RetryIntent]] = field(default_factory=deque)
    retry: Optional[_RetryIntent] = None
    headless: bool = False
    displays: deque[DisplayItem] = field(default_factory=deque)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    closing: bool = False
    ready: bool = False
    persistence_failed: bool = False
    phase: str = "starting"
    phase_started: float = field(default_factory=time.monotonic)
    exit_code: int = 0
    auth_required: bool = False
    auth_notice: str = "Model authentication needed; use /login."
    bound_account_id: Optional[str] = None
    login_cancel: threading.Event = field(default_factory=threading.Event)
    transient: queue.Queue[tuple[str, str]] = field(default_factory=lambda: queue.Queue(maxsize=8))
    active_user_call: Optional[str] = None
    trace: Optional[DebugTrace] = None

    def set_phase(self, phase: str) -> None:
        self.phase, self.phase_started = phase, time.monotonic()

    def notice(self, text: str) -> None:
        text = text.rstrip("\r\n")
        if self.headless:
            print(safe_text(f"[cli] {text}"), file=sys.stderr, flush=True)
        else:
            self.displays.append(DisplayItem(f"[cli] {text}"))

    def request_exit(self) -> None:
        self.closing = True
        self.retry = None
        self.pending.clear()
        self.login_cancel.set()
        self.changed.set()

    def handle_key(self, key: str, data: str) -> None:
        if key in {"c-c", "c-d"}:
            self.request_exit()
        elif not self.closing and self.ready:
            if key != "c-m":
                self.editor = self.editor.edit(key, data)
                return
            text = self.editor.text
            head = text.split(maxsplit=1)[0] if text.strip() else ""
            if head in {"/quit", "/exit"}:
                self.request_exit()
                return
            if not text.strip():
                self.editor = Editor()
                return
            if self.persistence_failed:
                self.notice("Checkpoint failed; no further work will run. Use /quit.")
                return
            intent = text
            if head == "/retry":
                error = None
                if text.strip() != "/retry" or "\n" in text or "\r" in text:
                    error = "Usage: /retry (no arguments; single line)."
                elif self.phase not in {"idle", "failed", "auth needed"}:
                    error = "Cannot retry while work is in progress."
                elif self.retry is None:
                    error = "No retryable sampling failure in this session."
                elif any(isinstance(item, _RetryIntent) for item in self.pending):
                    error = "A retry is already queued."
                if error is not None:
                    self.notice(error)
                    self.editor = Editor()
                    return
                intent = self.retry
            elif head.startswith("/"):
                try:
                    intent = parse_user_tool(text)
                except ValueError as exc:
                    # Never echo an arbitrary slash argument (possibly a secret).
                    self.notice(str(exc))
                    self.editor = Editor()
                    return
            elif self.auth_required:
                self.notice(f"{self.auth_notice} Draft was not submitted.")
                return
            if len(self.pending) >= _MAX_PENDING_QUERIES:
                self.notice("Query queue is full; the draft has not been submitted.")
            else:
                self.pending.append(intent)
                self.editor = Editor()
                self.changed.set()


def _build_model(args: argparse.Namespace, trace: Optional[DebugTrace]) -> Model:
    """Build the model; with --debug-trace, route its HTTP through the trace."""
    if trace is None:
        return build_model(args)
    return build_model(
        args,
        opener=trace.opener(urllib.request.urlopen),
        # Codex OAuth refresh during a sample's 401 recovery.
        auth_opener=trace.opener(default_account_opener(), op="auth_refresh"),
    )


@contextmanager
def _traced_operation(state: _UIState, op: str):
    """Tag the enclosed worker call's HTTP exchanges; report trace failures."""
    if state.trace is None:
        yield
        return
    try:
        with trace_operation(op):
            yield
    finally:
        warning = state.trace.take_warning()
        if warning is not None:
            state.notice(warning)


async def _checkpoint(context: InteractionContext, state: _UIState, path: Path) -> None:
    state.set_phase("saving")
    try:
        await asyncio.to_thread(save_interaction_save, path, context.copy())
    except Exception:
        state.persistence_failed = True
        raise


async def _append(
    context: InteractionContext,
    items: Iterable[InteractionItem],
    state: _UIState,
    path: Path,
) -> None:
    context.extend(items)
    await _checkpoint(context, state, path)


async def _sweep_tools(
    context: InteractionContext, environment: Environment, state: _UIState, path: Path
) -> None:
    for call in context.pending_tool_calls():
        if state.closing:
            return
        state.set_phase(f"tool: {call.name}")
        result = await asyncio.to_thread(environment.execute_tool_calls, (call,))
        await _append(context, result.context_items(), state, path)
        state.displays.extend(result.display_items(source_calls=(call,)))


async def _fail_pending_tools(
    context: InteractionContext, state: _UIState, path: Path, *, reason: str
) -> None:
    """Close missing outcomes without claiming that their effects did not happen."""
    for call in context.pending_tool_calls():
        if state.closing:
            return
        result = ToolResult(
            call_id=call.call_id,
            success=False,
            output=(
                f"Result unavailable after {reason}. "
                "This call cannot be resumed and was not rerun. "
                "It may already have produced side effects."
            ),
        )
        await _append(context, (result,), state, path)
        state.displays.extend(
            render_interaction_items((result,), source_calls=(call,))
        )


async def _fail_pending_user_tools(
    context: InteractionContext, state: _UIState, path: Path,
) -> None:
    for call in context.pending_user_tool_calls():
        if state.closing:
            return
        if call.call.name == "compact":
            output = (
                "Compaction outcome unavailable after interruption. The command "
                "was not rerun; no durable compaction checkpoint was installed."
            )
        elif call.call.name == "config":
            output = (
                "Config outcome unavailable after interruption. The command "
                "was not rerun; in-memory configuration was initialized from "
                "the current launch arguments."
            )
        else:
            output = (
                "User-tool outcome unavailable after interruption. The command "
                "was not rerun; credential side effects may already have occurred."
            )
        result = UserToolResult(ToolResult(
            call.call.call_id,
            output,
            success=False,
        ))
        await _append(context, (result,), state, path)
        state.displays.extend(render_interaction_items((result,), source_user_calls=(call,)))


def _has_provider_history(context: InteractionContext) -> bool:
    return any(
        (
            isinstance(i, (SampleMetadata, CompactionMetadata))
            and (
                i.provider_turn_id
                or i.provider_turn_state
                or i.provider_session_id
            )
        )
        or (isinstance(i, Reasoning) and i.encrypted_content)
        or (isinstance(i, OpaqueCompaction) and i.protocol == "responses")
        for i in (*context.items, *context.model_items())
    )


def _mark_auth_required(
    state: _UIState,
    exc: ModelAuthenticationError,
) -> None:
    state.auth_required = True
    if exc.failure is not None and exc.failure.auth_source == "environment":
        state.auth_notice = (
            "Environment credential rejected; update it and restart the process."
        )
    elif exc.failure is not None and exc.failure.auth_source == "static":
        state.auth_notice = (
            "Configured static credential rejected; restart with updated credentials."
        )
    else:
        state.auth_notice = "Model authentication needed; use /login."


def _compaction_failure_output(exc: BaseException) -> str:
    if isinstance(exc, ModelAuthenticationError):
        if exc.failure is not None:
            return f"Compaction failed: {exc.failure.message}"
        return "Model authentication needed; use /login."
    if isinstance(exc, CompactionError):
        detail = str(exc).replace("\r", " ").replace("\n", " ").strip()
        if detail:
            return f"Compaction failed: {detail[:512]}"
    if isinstance(exc, ModelError) and exc.failure is not None:
        return f"Compaction failed: {exc.failure.message}"
    return "Compaction failed; provider and response details were withheld."


def _compaction_success_output(result: CompactionResult) -> str:
    checkpoint = result.items[0]
    assert isinstance(checkpoint, ContextPrefix)
    opaque = any(
        isinstance(item, OpaqueCompaction)
        for item in checkpoint.prefix_items
    )
    mode = "a remote opaque checkpoint" if opaque else "a prompt summary checkpoint"
    return f"Context compacted using {mode}."


async def _compact_user_tool(
    intent: UserToolIntent,
    model: Optional[Model],
    context: InteractionContext,
    model_environment: Environment,
    state: _UIState,
    path: Path,
) -> Optional[Model]:
    # A pending UserToolCall deliberately makes a context non-sampleable. Take
    # the immutable compaction source first, then durably record authorization
    # for the effect before starting it.
    source_context = context.copy()
    call = UserToolCall(
        ToolCall(
            intent.name,
            "user_" + uuid.uuid4().hex,
            intent.arguments_json,
        )
    )
    await _append(context, (call,), state, path)
    state.displays.extend(render_interaction_items((call,)))
    state.active_user_call = call.call.call_id

    if state.closing:
        result_item = UserToolResult(
            ToolResult(
                call.call.call_id,
                "Compaction cancelled before execution.",
                success=False,
            )
        )
        await _append(context, (result_item,), state, path)
        state.displays.extend(
            render_interaction_items(
                (result_item,),
                source_user_calls=(call,),
            )
        )
        state.active_user_call = None
        return model

    state.set_phase("compacting")
    try:
        if model is None:
            result_item = UserToolResult(
                ToolResult(
                    call.call.call_id,
                    state.auth_notice,
                    success=False,
                )
            )
            contribution: tuple[InteractionItem, ...] = (result_item,)
        else:
            try:
                compactor = create_default_compactor(model)
                with _traced_operation(state, "compact"):
                    compaction = await asyncio.to_thread(
                        compactor.compact,
                        source_context,
                        tools=model_environment.tool_specs,
                    )
                if not isinstance(compaction, CompactionResult):
                    raise TypeError(
                        "compactor must return CompactionResult, got "
                        f"{type(compaction).__name__}"
                    )
            except Exception as exc:
                if isinstance(exc, ModelAuthenticationError):
                    _mark_auth_required(state, exc)
                    model = None
                result_item = UserToolResult(
                    ToolResult(
                        call.call.call_id,
                        _compaction_failure_output(exc),
                        success=False,
                    )
                )
                contribution = (result_item,)
            else:
                result_item = UserToolResult(
                    ToolResult(
                        call.call.call_id,
                        _compaction_success_output(compaction),
                    )
                )
                # Validate and save these together: a durable success result
                # must never exist without the checkpoint it describes.
                contribution = (
                    result_item,
                    *compaction.context_items(),
                )

        await _append(context, contribution, state, path)
        state.displays.extend(
            render_interaction_items(
                contribution,
                source_user_calls=(call,),
            )
        )
    finally:
        state.active_user_call = None
    return model


async def _user_tool(
    intent: UserToolIntent, model: Optional[Model], context: InteractionContext,
    state: _UIState, path: Path, args: argparse.Namespace,
    model_environment: Environment, config: InteractionConfig,
) -> Optional[Model]:
    if intent.name == "compact":
        return await _compact_user_tool(
            intent,
            model,
            context,
            model_environment,
            state,
            path,
        )
    expected_account = state.bound_account_id
    call = UserToolCall(ToolCall(intent.name, "user_" + uuid.uuid4().hex, intent.arguments_json))
    await _append(context, (call,), state, path)
    state.displays.extend(render_interaction_items((call,)))
    if state.closing:
        return model
    state.login_cancel.clear()
    state.active_user_call = call.call.call_id

    def notify(text):
        try:
            state.transient.put_nowait((call.call.call_id, text))
        except queue.Full:
            pass

    try:
        environment = create_user_environment(
            args, notify=notify, cancel=state.login_cancel,
            config=config,
            expected_account=expected_account,
            provider_history=_has_provider_history(context),
            **({} if state.trace is None else {
                "opener": state.trace.opener(default_account_opener()),
            }),
        )
        state.set_phase(f"user tool: {intent.name}")
        with _traced_operation(state, intent.name):
            outcome = await asyncio.to_thread(
                environment.execute_tool_calls, (call.call,),
            )
        result = UserToolResult(outcome.items[0])
        await _append(context, (result,), state, path)
        state.displays.extend(render_interaction_items((result,), source_user_calls=(call,)))
    finally:
        state.active_user_call = None
    if intent.name == "login" and result.result.success and not state.closing:
        state.set_phase("loading model")
        try:
            model = await asyncio.to_thread(_build_model, args, state.trace)
            if (expected_account is not None and
                    getattr(getattr(model, "endpoint", None), "account_id", None) != expected_account):
                raise ValueError("credential account changed during activation")
        except Exception:
            model = None
            state.exit_code = 1
            state.pending.clear()
            state.notice("Credentials were saved, but model activation failed. No model request was started.")
        else:
            state.bound_account_id = getattr(getattr(model, "endpoint", None), "account_id", None)
            state.notice(
                "Model ready with reloaded credentials. Backend authorization "
                "will be verified by the next model request; blocked drafts were not "
                "automatically submitted."
            )
        state.auth_required = model is None
        if model is not None:
            state.auth_notice = "Model authentication needed; use /login."
    return model


async def _turn(
    context: InteractionContext,
    model: Model,
    environment: Environment,
    state: _UIState,
    path: Path,
    config: InteractionConfig,
) -> None:
    # Each explicit attempt consumes the preceding failure's ticket. Only a
    # sampling failure below can arm a new one, not a tool/compaction/save error.
    state.retry = None
    turn_config = config.snapshot()
    sample_params = turn_config.sample_params()
    turn_started = time.perf_counter()
    samples = 0
    while not state.closing:
        if (
            turn_config.max_samples is not None
            and samples >= turn_config.max_samples
        ):
            raise RuntimeError(
                "model did not produce a final answer within "
                f"{turn_config.max_samples} samples"
            )
        threshold = turn_config.auto_compact_tokens
        if (
            turn_config.enable_auto_compaction
            and uses_host_auto_compaction(model)
            and threshold is not None
            and should_auto_compact(context, threshold)
        ):
            state.set_phase("compacting")
            compactor = create_default_compactor(model)
            with _traced_operation(state, "compact"):
                compaction = await asyncio.to_thread(
                    compactor.compact,
                    context.copy(),
                    tools=environment.tool_specs,
                )
            if not isinstance(compaction, CompactionResult):
                raise TypeError(
                    "compactor must return CompactionResult, got "
                    f"{type(compaction).__name__}"
                )
            await _append(
                context,
                compaction.context_items(),
                state,
                path,
            )
            state.displays.extend(compaction.display_items())
            if state.closing:
                return
        state.set_phase("sampling")
        try:
            with _traced_operation(state, "sample"):
                sample = await asyncio.to_thread(
                    model.sample,
                    context.copy(),
                    tools=environment.tool_specs,
                    sample_params=sample_params,
                )
        except ModelError as exc:
            contribution = (
                *exc.completed_items,
                *((exc.failure,) if exc.failure is not None else ()),
            )
            if contribution:
                recovered = (*contribution, ModelSampleBoundary())
                await _append(context, recovered, state, path)
                state.displays.extend(render_interaction_items(contribution))
                recovered_calls = tuple(
                    item for item in exc.completed_items
                    if isinstance(item, ToolCall)
                )
                if recovered_calls:
                    results = tuple(
                        ToolResult(
                            call_id=call.call_id,
                            output=(
                                "Not executed because the model response did "
                                "not complete."
                            ),
                            success=False,
                        )
                        for call in recovered_calls
                    )
                    await _append(context, results, state, path)
                    state.displays.extend(
                        render_interaction_items(
                            results,
                            source_calls=recovered_calls,
                        )
                    )
            state.retry = _RetryIntent()
            raise
        except Exception:
            # Adapters normally raise ModelError, but an exception at this
            # sampling boundary is still distinct from a failed local effect.
            state.retry = _RetryIntent()
            raise
        samples += 1
        model_account_id = getattr(
            getattr(model, "endpoint", None),
            "account_id",
            None,
        )
        if state.bound_account_id is None and model_account_id is not None:
            state.bound_account_id = model_account_id
        await _append(context, sample.context_items(), state, path)
        state.displays.extend(sample.display_items())
        if sample.stop_reason == "compaction":
            continue
        if not sample.tool_calls:
            final_text = sample.last_assistant_text
            if not final_text or not final_text.strip():
                state.retry = _RetryIntent()
                raise RuntimeError("model returned no final assistant text")
            summary = summarize_turn_usage(
                context.items,
                elapsed_seconds=time.perf_counter() - turn_started,
            )
            await _append(context, (summary,), state, path)
            state.displays.extend(render_interaction_items((summary,)))
            return
        await _sweep_tools(context, environment, state, path)


async def _reload_retry_model(
    context: InteractionContext, state: _UIState, args: argparse.Namespace,
) -> Optional[Model]:
    """Reload credentials without initiating login or switching accounts."""
    state.set_phase("loading model")
    state.auth_required = True
    expected_account = state.bound_account_id
    if (
        expected_account is None
        and supports_account_services(args)
        and _has_provider_history(context)
    ):
        state.notice(
            "Cannot verify the account for saved provider state; start a fresh session."
        )
        return None
    try:
        model = await asyncio.to_thread(_build_model, args, state.trace)
        account = getattr(getattr(model, "endpoint", None), "account_id", None)
        if expected_account is not None and account != expected_account:
            state.notice(
                "Credential account changed; no model request was started. "
                "Restore the original account or start a fresh session."
            )
            return None
    except Exception:
        # As with /login activation, do not reflect credential/provider details.
        state.notice("Model reload failed; no model request was started. Details withheld.")
        state.notice(state.auth_notice)
        return None
    state.bound_account_id = account
    state.auth_required = False
    state.auth_notice = "Model authentication needed; use /login."
    return model


def _ends_with_completed_manual_compaction(context: InteractionContext) -> bool:
    items = context.items
    end = len(items)
    if end and isinstance(items[end - 1], CompactionMetadata):
        end -= 1
    if end < 3 or not isinstance(items[end - 1], ContextPrefix):
        return False
    result = items[end - 2]
    call = items[end - 3]
    return (
        isinstance(result, UserToolResult)
        and result.result.success
        and isinstance(call, UserToolCall)
        and call.call.name == "compact"
        and result.result.call_id == call.call.call_id
    )


def _resume_notice(context: InteractionContext) -> Optional[str]:
    # Inspect the raw tail, not the model context established by a ContextPrefix.
    # A sample boundary does not record stop_reason or turn completion.
    if _ends_with_completed_manual_compaction(context):
        return None
    for item in reversed(context.items):
        if isinstance(
            item,
            (
                ModelSampleBoundary,
                SampleMetadata,
                CompactionMetadata,
                UserInteractionBoundary,
                UserToolCall,
                UserToolResult,
            ),
        ):
            continue
        if isinstance(item, (TurnSummary, Init)):
            return None
        if isinstance(item, OpaqueCompaction):
            tail = "a compaction checkpoint"
        elif isinstance(item, ContextPrefix):
            tail = "a context-prefix checkpoint"
        elif isinstance(item, ToolResult):
            tail = "tool results"
        elif isinstance(item, Message) and item.role == "user":
            tail = "a user submission"
        elif isinstance(item, Message) and item.role == "assistant":
            tail = "assistant output"
        elif isinstance(item, Instructions):
            tail = "an instructions update"
        elif isinstance(item, ModelFailure):
            tail = "a failed model attempt"
        else:
            tail = "incomplete model output"
        return (
            f"Resumed save ends with {tail}, without recorded turn completion. "
            "The model stop reason is not saved. No model request was started; "
            "enter a query to continue."
        )
    return None


async def _drive_interaction(
    model: Optional[Model],
    environment: Environment,
    state: _UIState,
    args: argparse.Namespace,
    path: Path,
    config: InteractionConfig,
) -> None:
    attachment_cwd = Path(args.cwd).expanduser().resolve()
    existing = args.resume and await asyncio.to_thread(path.exists)
    if existing:
        context = await asyncio.to_thread(load_interaction_save, path)
        state.displays.extend(render_interaction_items(context.items))
        state.notice(
            "Command sessions and plan state were not restored. "
            "Old command session IDs are not resumable; use only IDs from this run."
        )
        if any(
            isinstance(item, UserToolCall)
            and item.call.name == "config"
            for item in context.items
        ):
            state.notice(
                "In-memory configuration was reset from the current launch "
                "arguments; saved config commands were not replayed."
            )
    else:
        if args.resume:
            state.notice(
                f"Warning: no existing {path.name} was found; a fresh one was created."
            )
        initial = [Init(model=args.model or initial_model_name(model))]
        if args.instructions is not None:
            initial.append(Instructions(args.instructions))
        context = InteractionContext(initial)
    if state.closing:
        return
    initial_query = args.prompt
    startup = True
    while not state.closing:
        sampling_attempt = False
        try:
            if startup:
                if existing:
                    await _fail_pending_user_tools(context, state, path)
                    await _fail_pending_tools(
                        context, state, path, reason="session restart"
                    )
                else:
                    # Initial-save failures retain the fresh context too.
                    await _checkpoint(context, state, path)
                if state.closing:
                    return
                if existing and args.instructions is not None:
                    instructions = Instructions(args.instructions)
                    await _append(context, (instructions,), state, path)
                    state.displays.extend(render_interaction_items((instructions,)))
                if (getattr(args, "debug_save_model_binding", False)
                        and getattr(args, "model_binding", None) is not None):
                    warning = await asyncio.to_thread(
                        save_debug_model_bindings,
                        debug_model_binding_path(path),
                        {"main": args.model_binding},
                    )
                    if warning is not None:
                        state.notice(warning)
                query = initial_query
                should_sample = model is not None and (query is not None or (
                    existing and args.instructions is not None
                ))
                if existing and not should_sample:
                    notice = _resume_notice(context)
                    if notice is not None:
                        state.notice(notice)
                if model is None:
                    query = None
                    state.notice(
                        f"{state.auth_notice} No model query was submitted."
                    )
                else:
                    state.editor = Editor()
                state.ready = True
                startup = False
            else:
                await state.changed.wait()
                state.changed.clear()
                if state.closing:
                    return
                if not state.pending:
                    continue
                query = state.pending.popleft()
                if state.pending:
                    state.changed.set()
                if isinstance(query, _RetryIntent):
                    # Recheck on the owner: an earlier queued query may have
                    # superseded the failure since Enter accepted this ticket.
                    if query is not state.retry:
                        state.notice("Retry no longer applies to the current task.")
                        continue
                    if context.pending_tool_calls() or context.pending_user_tool_calls():
                        state.retry = None
                        state.notice("Cannot retry with unresolved tool outcomes.")
                        continue
                    if model is None:
                        model = await _reload_retry_model(context, state, args)
                        if model is None:
                            state.set_phase("auth needed")
                            continue
                    query = None  # Continue context; do not append a user turn.
                else:
                    await _fail_pending_user_tools(context, state, path)
                    # A new query is not permission to retry old calls whose
                    # side effects may already have happened.
                    await _fail_pending_tools(
                        context, state, path, reason="an interrupted operation"
                    )
                    if state.closing:
                        return
                    if isinstance(query, UserToolIntent):
                        model = await _user_tool(
                            query,
                            model,
                            context,
                            state,
                            path,
                            args,
                            environment,
                            config,
                        )
                        state.set_phase("auth needed" if state.auth_required else "idle")
                        continue
                    if model is None:
                        state.editor = Editor(query, len(query))
                        state.notice(f"{state.auth_notice} Draft was not submitted.")
                        state.set_phase("auth needed")
                        continue
                    state.retry = None
                should_sample = True
            if state.closing:
                return
            if query is not None:
                try:
                    message = parse_user_prompt(
                        query,
                        cwd=attachment_cwd,
                        enabled=args.enable_experimental_media,
                        enable_workspace=args.enable_workspace,
                    )
                    if message.has_media and args.model_api not in {
                        "codex",
                        "chat-completions",
                    }:
                        raise AttachmentError(
                            f"--endpoint-api {args.model_api} does not support "
                            "media prompts; use codex or "
                            "chat-completions"
                        )
                except AttachmentError as exc:
                    state.notice(str(exc))
                    if state.headless:
                        state.exit_code = 1
                        state.set_phase("failed")
                        return
                    state.editor = Editor(query, len(query))
                    state.set_phase("idle")
                    continue
                user = UserInteraction((message,))
                await _append(context, user.context_items(), state, path)
                state.displays.extend(user.display_items())
            if should_sample:
                sampling_attempt = True
                await _turn(
                    context,
                    model,
                    environment,
                    state,
                    path,
                    config,
                )
            state.set_phase("auth needed" if state.auth_required else "idle")
            if state.headless:
                return
        except Exception as exc:
            state.exit_code = 1
            state.ready = True
            startup = False
            state.set_phase("failed")
            if isinstance(exc, ModelAuthenticationError):
                if state.bound_account_id is None:
                    state.bound_account_id = getattr(
                        getattr(model, "endpoint", None), "account_id", None
                    )
                model = None
                _mark_auth_required(state, exc)
            state.notice(f"{type(exc).__name__}: {exc}")
            if state.pending:
                state.notice(
                    "Queued queries discarded after failure; submit again explicitly."
                )
            state.pending.clear()
            state.changed.clear()
            if state.persistence_failed:
                state.retry = None
                # Retain the unsaved context here until exit; never replay the effect.
                state.notice(
                    "Checkpoint failed; unsaved state remains in memory. "
                    + ("No further work will run." if state.headless else "Use /quit.")
                )
                if state.headless:
                    return
                while not state.closing:
                    await state.changed.wait()
                    state.changed.clear()
                return
            if state.headless:
                return
            if sampling_attempt and state.retry is not None and not state.closing:
                # Keep all existing diagnostics above, then append guidance.
                # Authentication guidance suggests /login; it is not a gate
                # on /retry, which can reload externally refreshed credentials.
                state.notice(
                    state.auth_notice if isinstance(exc, ModelAuthenticationError)
                    else "Sampling failed. Use /retry to try again."
                )


def _runtime_config(environment, args):
    workspace_update = getattr(environment, "set_enable_workspace", None)
    config = InteractionConfig.from_namespace(
        args,
        on_enable_workspace=(
            workspace_update if callable(workspace_update) else None
        ),
    )
    if callable(workspace_update):
        workspace_update(config.get("enable_workspace"))
    return config


def _startup_notices(state, args, path):
    state.notice(f"Save log: {path}")
    if state.trace is not None:
        state.notice(
            f"Debug trace: {state.trace.request_path} and "
            f"{state.trace.response_path} (append-only; verbatim HTTP "
            "payloads and headers, including credentials)."
        )
    if args.enable_default_tools:
        state.notice(
            "Warning: exec_command runs without a sandbox; use a trusted model and workspace."
        )
        if not args.enable_workspace:
            state.notice(
                "Warning: workspace path restrictions are disabled; "
                "exec_command workdir and apply_patch paths may resolve "
                "outside --cwd."
            )
    else:
        state.notice("Default model tools disabled; user commands remain available."
                     if not state.headless else "Default model tools disabled.")


async def _run_headless(model, environment, args, path, *, trace=None):
    """One explicit task, quiet context display, and orderly effect draining."""
    if model is None:
        raise ValueError("Headless execution requires an available model.")
    path = Path(path).absolute()
    config = _runtime_config(environment, args)
    state = _UIState(
        headless=True,
        bound_account_id=getattr(getattr(model, "endpoint", None), "account_id", None),
        trace=trace,
    )
    _startup_notices(state, args, path)
    loop = asyncio.get_running_loop()
    interrupted = False
    previous_sigint = None

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        loop.call_soon_threadsafe(state.request_exit)

    if threading.current_thread() is threading.main_thread():
        # asyncio.run on older Python versions cancels *all* tasks on SIGINT.
        # Request a cooperative stop instead, so the effect owner can save.
        previous_sigint = signal.signal(signal.SIGINT, interrupt)
    worker = asyncio.create_task(_drive_interaction(model, environment, state, args, path, config))
    try:
        while not worker.done():
            # The controller's display queue is transient, not a second log.
            state.displays.clear()
            await asyncio.wait((worker,), timeout=0.05)
        worker.result()
    finally:
        try:
            state.request_exit()
            # Cancellation/SIGINT must not close command resources before an
            # in-flight request/tool has finished and checkpointed its outcome.
            await asyncio.shield(worker)
            state.displays.clear()
        finally:
            if previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)
    return 130 if interrupted else state.exit_code


async def _run(
    model: Optional[Model],
    environment: Environment,
    terminal: PosixTerminal,
    args: argparse.Namespace,
    path: Path = DEFAULT_SAVE_PATH,
    *,
    trace: Optional[DebugTrace] = None,
) -> int:
    path = Path(path).absolute()
    config = _runtime_config(environment, args)
    prompt = args.prompt or ""
    state = _UIState(
        editor=Editor(prompt, len(prompt)), auth_required=model is None,
        bound_account_id=getattr(getattr(model, "endpoint", None), "account_id", None),
        trace=trace,
    )
    state.notice(
        "pythia.interaction — /retry, /compact, /config, /config.json, /login, /quota; "
        "/quit or /exit; Ctrl-C/Ctrl-D exit."
    )
    _startup_notices(state, args, path)
    worker = None
    frame = 0
    with terminal:
        try:
            # Draw the pre-filled editor before starting startup I/O/auto-submit.
            terminal.render(state.editor, "starting", tuple(state.displays))
            state.displays.clear()
            worker = asyncio.create_task(
                _drive_interaction(model, environment, state, args, path, config)
            )
            while True:
                for key in terminal.read_keys():
                    state.handle_key(key.key, key.data or "")
                if terminal.closed:
                    state.request_exit()
                while True:
                    try:
                        call_id, text = state.transient.get_nowait()
                    except queue.Empty:
                        break
                    if call_id == state.active_user_call:
                        state.notice(text)
                busy = state.phase not in {"idle", "failed", "auth needed"}
                status = (
                    "closing — waiting for current operation"
                    if state.closing else state.phase
                )
                if busy:
                    status += f" {int(time.monotonic() - state.phase_started)}s"
                if state.pending:
                    status += f" | queued={len(state.pending)}"
                prompt = (
                    f"{_SPINNER[(frame // 16) % len(_SPINNER)]}> "
                    if busy else ":> "
                )
                terminal.render(state.editor, status, tuple(state.displays), prompt)
                state.displays.clear()
                if worker.done():
                    worker.result()
                    break
                frame += 1
                await asyncio.sleep(FRAME_INTERVAL)
        finally:
            state.request_exit()
            if worker is not None:
                # Do not cancel a thread-backed effect or close its environment early.
                await asyncio.shield(worker)
    return state.exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "Interactive POSIX shell or headless task using Pythia's caller-owned interaction API.",
        allow_prompt_file=True,
    )
    parser.add_argument(
        "--headless", nargs="?", const=True, default=False,
        type=_boolean_argument, metavar="{False,True}",
        help=("run one explicit prompt without a TUI, stdin reads, or context display; "
              "save the interaction and exit. Requires --prompt/--prompt-file or "
              "--resume with --instructions and an existing save. A bare flag "
              "means True (default: %(default)s)"),
    )
    parser.add_argument(
        "--enable-default-tools",
        nargs="?",
        const=True,
        default=True,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            "enable default model tools (exec_command, write_stdin, apply_patch, "
            "update_plan); user commands remain available when False. "
            "Launch-only: repeat on --resume. A bare flag means True "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help=(
            "append every HTTP request and response (model sampling and "
            "compaction, /quota, and Codex OAuth) verbatim, including "
            "credentials, to SAVE.trace.req.jsonl and SAVE.trace.res.jsonl; "
            "never truncated, even without --resume; launch-only"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        catalog = frontend_catalog(args)
        if args.list_models:
            print(render_model_catalog(catalog))
            return 0
        args = prepare_namespace(args, catalog)
        args.prompt = load_prompt(args)
        if not args.headless and (os.name != "posix" or not sys.stdin.isatty() or not sys.stdout.isatty()):
            raise ValueError(
                "interaction CLI requires POSIX terminal stdin/stdout; "
                "use --headless with --prompt or --prompt-file for one task"
            )
        if args.prompt is not None and not args.prompt.strip():
            raise ValueError("prompt must be a non-empty string or None")
        if args.max_samples is not None and args.max_samples <= 0:
            raise ValueError("max_samples must be a positive integer or None")
        if args.max_output_tokens is not None:
            SampleParams(max_output_tokens=args.max_output_tokens)
        save_path = resolve_save_path(args.save_path)
        if (args.headless and args.prompt is None
                and not (args.resume and args.instructions is not None and save_path.is_file())):
            raise ValueError(
                "--headless requires --prompt or --prompt-file, or "
                "--resume with --instructions and an existing save."
            )
        trace = DebugTrace.open(save_path) if args.debug_trace else None
        try:
            model = _build_model(args, trace)
        except CodexAuthUnavailable:
            if args.headless or not supports_account_services(args):
                raise
            model = None
        cwd = Path(args.cwd).expanduser().resolve()
        environment_manager = (
            DefaultEnvironment(cwd=cwd, enable_workspace=args.enable_workspace)
            if args.enable_default_tools else nullcontext(Environment())
        )
        with environment_manager as environment:
            if args.headless:
                return asyncio.run(_run_headless(
                    model, environment, args, path=save_path, trace=trace,
                ))
            terminal = PosixTerminal(sys.stdin, sys.stdout)
            return asyncio.run(_run(
                model, environment, terminal, args, path=save_path, trace=trace,
            ))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"interaction CLI failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
