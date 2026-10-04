"""Fixed-role auto app. Run ``python3 -m pythia.interaction.auto --help``.

All contexts initially wait. By default only #1 (main) and #-1 (watcher) run:
frontend tasks wake main directly, and whenever main's turn loop stops it yields
to the watcher, whose own model turn may resume main with a follow-up message
(see ``_supervision``). ``--enable-experimental-worker-board`` adds the
experimental #2 (worker) and shared message board: user tasks then become board
threads that wake main, main's posted plans wake the worker, and the watcher only
displays main's end-of-turn condition.
This module deliberately does not change the existing CLI or demo.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional
from urllib.parse import urlsplit
import uuid

from ._auto_board import Board, BoardError, BoardService, atomic_text
from ._auto_config import (DEFAULTS, NAMES, ROLE_INDEX, build_parser, load_saved_config,
                           namespace, resolve_config, saved_document)
from ._model_binding_debug import save_debug_model_bindings
from ._cli_editor import Editor, safe_text
from ._cli_terminal import PosixTerminal
from ._prompt import load_prompt
from ._supervision import Fault, SupervisedHandle, Yield, YieldChannel, supervise
from .compaction import CompactionResult, NothingToCompact, auto_compaction_due
from .compaction import create_default_compactor, uses_host_auto_compaction
from .context import InteractionContext
from .default_environment import DefaultEnvironment
from .display import DisplayItem, render_interaction_items
from .environment import Environment, Tool, ToolOutcome, ToolSpec
from .items import Init, Instructions, Message, ModelSampleBoundary, ToolCall, ToolResult, Tools
from .items import UserToolResult
from .items import summarize_turn_usage
from .model import ModelContextWindowError, ModelError, ModelSample
from .model_config import build_model
from .model_config import frontend_catalog, render_model_catalog
from .model_catalog import BUILTIN_MODEL_CATALOG
from .runtime_config import InteractionConfig
from .save import SaveError, load_interaction_save, save_interaction_save
from .user import UserInteraction


_FRAME_INTERVAL = 1 / 128
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_TIMED_PHASES = {"sampling", "compacting", "executing tools", "saving", "awaiting watcher"}
_ACTIVE_PHASES = {"starting", *_TIMED_PHASES}
# Unfinished in-process tasks admitted without the board (its user-task limit).
_MAX_PENDING_TASKS = 16
# read_main_context paging: default/maximum items per read, characters per item.
_READ_LIMIT, _READ_MAX_LIMIT, _READ_ITEM_CHARS = 20, 50, 2000


_COOPERATION_PREAMBLE = (
    "This session is a cooperative effort to complete the user's task through a "
    "shared message board. Main plans and reviews, worker carries out assigned "
    "work, and feedback guides iterative revisions toward a verified result. "
    "Coordinate complementary work rather than duplicating it, and respect the "
    "user's requested scope."
)


_ROLE_INSTRUCTIONS = {
    1: (
        "You are main, the user-facing planner and reviewer. Your input is a "
        "shared-board task thread. For implementation requests, inspect enough "
        "context to form a self-contained, bounded execution plan with scope, "
        "acceptance criteria, and required checks. Call board_post_plan to assign "
        "implementation and tests to worker #2. Worker owns code and test edits; "
        "do not instead assign worker a read-only review and implement the change "
        "yourself. Use local tools to inspect actual changes and run independent "
        "checks, not to take over implementation.\n\n"
        "Use board_read_thread to obtain the result for each plan. Review the "
        "diff and reported checks against the user's requirements. If corrections "
        "are needed, post a concrete follow-up execution plan in the same thread, "
        "referring to the prior plan/result, and review the revised work. Continue "
        "this cycle until accepted or blocked. Do not finalize merely because a "
        "plan was posted: remain active through worker execution and your review. "
        "A final answer ends your turn; later worker results do not automatically "
        "restart it. If blocked, report what remains unresolved.\n\n"
        "Respect explicitly planning-only or review-only user requests; do not "
        "authorize implementation for those tasks. The host publishes your final "
        "answer to the board. Report what is actually verified. "
        "Do not resubmit a plan after an uncertain tool outcome without checking "
        "the board. Local update_plan only maintains your checklist; it does not "
        "delegate work."
    ),
    2: (
        "You are worker, the implementer. Execute main's current assignment with "
        "the allowed local tools, including code changes and tests when "
        "implementation is requested. Do not substitute another implementation "
        "proposal for the requested implementation. board_read_thread supplies "
        "the shared task context. For follow-up assignments, revise the existing "
        "work according to main's review rather than starting an unrelated task. "
        "Respect explicit read-only assignments and the user's requested scope. "
        "Finish relevant commands before handing work back to main for review. "
        "Return a concrete account of changed files, checks and their outcomes, "
        "and remaining blockers; do not claim unperformed work. The host will "
        "publish your final outcome; do not create new tasks or invent credentials."
    ),
    -1: "You are watcher. The host displays a debug event when main's end-of-turn condition fires. No model polling is needed.",
}
# Without the board, main defaults to no instructions (like the CLI): its
# board-era role text is about delegating to the worker.
_STANDALONE_ROLE_INSTRUCTIONS = {-1: _ROLE_INSTRUCTIONS[-1]}
_SUPERVISOR_INSTRUCTIONS = (
    "You are watcher (#-1), the supervisor of main (#1), an agent that works on the "
    "user's tasks in a shared workspace. Whenever main's turn loop stops, the host "
    "sends you a report: the user's request, how main stopped (ended or failed), "
    "main's final answer or a safe failure summary, and how many times main was "
    "already resumed for that task. Your purpose is to recover main from errors and "
    "to continue tasks main left unfinished.\n\n"
    "Use read_main_context to inspect main's log when the report is not enough. If "
    "main's work completes the request, or the task is blocked on the user, end your "
    "turn without resuming main. Otherwise call resume_main once with a concise "
    "message for main's next turn: what to retry or continue, and why. Do not add "
    "requirements beyond the user's request or repeat finished work. Main receives "
    "your message as an automated follow-up, not as the user. You have no "
    "workspace tools."
)
_FOLLOW_UP_HEADER = "Automated follow-up from the watcher (#-1):\n\n"
_RESTART_NOTICE = ("Restart notice: saved history was resumed without "
                   "restoring old command-session IDs or runtime state.")


def _instructions(index, settings, base_url, *, supervisor=False):
    """Effective role instructions; None without the board and any body."""
    body = settings["instructions"]
    if base_url is None:
        if body is None:
            body = (_SUPERVISOR_INSTRUCTIONS if supervisor and index == -1
                    else _STANDALONE_ROLE_INSTRUCTIONS.get(index))
        return None if body is None else Instructions(body)
    if body is None:
        body = _COOPERATION_PREAMBLE + "\n\n" + _ROLE_INSTRUCTIONS[index]
    return Instructions(body + "\n\n# Shared message board instructions\n\n"
                        "<INSTRUCTIONS>\n"
                        "This session has a shared message board for user task threads, plans, and results.\n"
                        f"Address: {base_url}\n"
                        f"Read {base_url}/README.md for API and usage instructions.\n"
                        "Use the bound board tools; the host supplies their credentials privately.\n"
                        "</INSTRUCTIONS>")


@dataclass(frozen=True)
class _Event:
    index: Optional[int]
    items: tuple[DisplayItem, ...]
    kind: str = "output"


@dataclass(frozen=True)
class _Completion:
    record_id: str
    thread_id: str


@dataclass(frozen=True)
class _Task:
    """One in-process user task for main when the experimental board is off."""
    sequence: int
    record_id: str
    content: str


class _Stopping(RuntimeError):
    pass


class _StartupError(RuntimeError):
    """A role failed to start; its notice has the details, and this text is safe."""


def _role_summary(index, settings, binding, source, runs_model):
    """One startup line: the role's API and model, and where the model came from."""
    name = settings["name"]
    if not runs_model:
        return f"#{index} ({name}): no model (observe-only)"
    text = f"#{index} ({name}): {binding.api} / {settings['model'] or '(server default)'}"
    if binding.spec is None:
        # Not catalogued: name the endpoint, which may be a local default.
        text += f" at {urlsplit(binding.endpoint.url).netloc}"
    return text if source is None else f"{text} ({source})"


def _check_credentials(settings, indices, catalog):
    """Fail before a save exists when a sampling role's credential reference is
    unusable; messages name the variable or file, never a value."""
    for index in indices:
        value = settings[index]
        endpoint = namespace(value, catalog).model_binding.endpoint
        who = (f"#{index} ({value['name']}) uses "
               f"{value['model'] or 'the server default model'}")
        if endpoint.auth == "supplied":
            raise ValueError(f"{who}, which needs a supplied API key; auto takes credentials "
                             "only by reference (--endpoint-auth env:NAME).")
        variable = endpoint.environment_variable
        if variable is not None and not os.environ.get(variable, "").strip():
            raise ValueError(f"{who}, which needs the {variable} environment variable; "
                             "it is not set.")
        if (endpoint.auth == "codex-login" and endpoint.auth_file is not None
                and not Path(endpoint.auth_file).is_file()):
            raise ValueError(f"{who}, which needs a Codex login at {endpoint.auth_file}; "
                             "none was found.")


class SampleLimitExceeded(RuntimeError):
    """A turn reached its explicit per-turn sample limit."""


class MissingFinalText(RuntimeError):
    """A turn ended without nonblank final assistant text."""


class _Binding:
    """Only touched on its context's owner thread; credentials never reach a model."""
    def __init__(self, index, client):
        self.index, self.client = index, client
        self.source = None

    def tools(self):
        def read(arguments, *, timeout_seconds=None):
            del timeout_seconds
            if self.source is None or set(arguments) - {"after", "limit"}:
                raise BoardError("Invalid board_read_thread arguments or no active task.")
            after, limit = arguments.get("after", 0), arguments.get("limit", 20)
            if (type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 100):
                raise BoardError("Expected after >= 0 and 1 <= limit <= 100.")
            value = self.client.read(self.source.thread_id, after=after, limit=limit)
            return ToolOutcome(json.dumps(value, ensure_ascii=False))

        def post(arguments, *, timeout_seconds=None):
            del timeout_seconds
            if self.source is None or set(arguments) != {"content"}:
                raise BoardError("board_post_plan requires only content and an active task.")
            value = self.client.post(self.source.thread_id, "plan", arguments["content"],
                                     self.source.record_id)
            return ToolOutcome(json.dumps({"thread_id": value["thread_id"],
                                           "plan_id": value["record_id"], "status": "accepted"}))

        tools = [Tool(ToolSpec(
            "board_read_thread", "Read this task's board thread, including plans/results. Follow next_after when has_more is true.",
            {"type": "object", "properties": {
                "after": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
             "additionalProperties": False}), read, timeout_seconds=10)]
        if self.index == 1:
            tools.append(Tool(ToolSpec(
                "board_post_plan", "Submit a self-contained plan to worker #2 in this task thread. Returns a plan ID, not a completed result. Author/thread/credentials are supplied by the host.",
                {"type": "object", "properties": {"content": {"type": "string"}},
                 "required": ["content"], "additionalProperties": False}),
                post, timeout_seconds=10))
        return tuple(tools)


def _model_factory(index, args):
    del index
    return build_model(args)


def _read_items(view, start, limit):
    """Render items start..start+limit-1 of a read-only log view as plain text."""
    calls = [item for item in view[:start] if isinstance(item, ToolCall)]
    entries = []
    for index in range(start, min(len(view), start + limit)):
        item = view[index]
        text = "\n".join(display.text for display in render_interaction_items(
            (item,), source_calls=calls, color=False))
        if isinstance(item, ToolCall):
            calls.append(item)
        if len(text) > _READ_ITEM_CHARS:
            text = (text[:_READ_ITEM_CHARS]
                    + f"\n[{len(text) - _READ_ITEM_CHARS} more characters omitted]")
        entries.append({"index": index, "type": type(item).__name__, "text": text})
    end = start + len(entries)
    return {"revision": len(view), "start": start, "next": end,
            "has_more": end < len(view), "items": entries}


class _WatcherTools:
    """Bound supervisor tools; touched only on the watcher's owner thread."""

    def __init__(self):
        self.handle = None  # SupervisedHandle while supervising
        self._open = False
        self._resume = None

    def begin(self):
        self._open, self._resume = True, None

    def end(self):
        resume, self._open, self._resume = self._resume, False, None
        return resume

    def tools(self):
        def resume_main(arguments, *, timeout_seconds=None):
            del timeout_seconds
            content = arguments.get("content")
            if set(arguments) != {"content"} or not isinstance(content, str) or not content.strip():
                raise ValueError("resume_main requires only nonempty content.")
            if not self._open:
                raise ValueError("No main yield is awaiting a decision.")
            if self._resume is not None:
                raise ValueError("Main is already being resumed for this report; end your turn.")
            self._resume = content
            return ToolOutcome("Recorded: main resumes with this message after your turn ends.")

        def read_main_context(arguments, *, timeout_seconds=None):
            del timeout_seconds
            view = () if self.handle is None else self.handle.view
            if set(arguments) - {"start", "limit"}:
                raise ValueError("read_main_context accepts only start and limit.")
            limit = arguments.get("limit", _READ_LIMIT)
            if type(limit) is not int or not 1 <= limit <= _READ_MAX_LIMIT:
                raise ValueError(f"limit must be an integer from 1 to {_READ_MAX_LIMIT}.")
            start = arguments.get("start", max(0, len(view) - limit))
            if type(start) is not int or not 0 <= start <= len(view):
                raise ValueError(f"start must be an integer from 0 to {len(view)}.")
            return ToolOutcome(json.dumps(_read_items(view, start, limit), ensure_ascii=False))

        return (
            Tool(ToolSpec(
                "resume_main",
                "Resume main (#1) with a follow-up message once your turn ends. Call at "
                "most once per report; end your turn without calling it to leave main idle.",
                {"type": "object", "properties": {"content": {"type": "string"}},
                 "required": ["content"], "additionalProperties": False}),
                resume_main, timeout_seconds=10),
            Tool(ToolSpec(
                "read_main_context",
                "Read main's log as of the current report. Items are addressed by index "
                "0..revision-1; omit start for the last items. Follow next while has_more.",
                {"type": "object", "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _READ_MAX_LIMIT}},
                 "additionalProperties": False}),
                read_main_context, timeout_seconds=10),
        )


def _yield_report(yield_):
    """The watcher's user message for one main yield (no log contents)."""
    outcome = yield_.kind if yield_.reason is None else f"{yield_.kind} ({yield_.reason})"
    lines = [f"Main (#{yield_.context}) yielded on task {yield_.job_id} "
             f"(watcher resumes so far: {yield_.resumes}).",
             "", "User request:", yield_.job_text, "", f"Outcome: {outcome}"]
    if yield_.failure is not None:
        lines.append(f"Failure: {yield_.failure.category}: {yield_.failure.message}")
    if yield_.final_text is not None:
        lines += ["", "Main's final answer:", yield_.final_text]
    lines += ["", f"Main's log has {yield_.revision} items (read_main_context indices "
                  f"0..{yield_.revision - 1})."]
    return "\n".join(lines)


def _yield_debug(yield_):
    if yield_.kind == "ended":
        return f"[debug] main end-of-turn condition fired: #1 source={yield_.job_id}"
    reason = "" if yield_.reason is None else f" ({yield_.reason})"
    resumable = "" if yield_.resumable else ", non-resumable"
    return f"[debug] main yielded: #1 source={yield_.job_id} {yield_.kind}{reason}{resumable}"


def _environment_factory(index, args, tools):
    if index == -1:
        return Environment(tools)  # Only bound watcher tools; no workspace tools.
    return DefaultEnvironment(cwd=args.cwd, enable_workspace=args.enable_workspace,
                              extra_tools=tools)


def _compact(session, index, model, environment, config, context, sample_params):
    """Install one automatic compaction; False when there is nothing to compact."""
    session._phase(index, "compacting")
    compactor = create_default_compactor(model, config.compaction_settings())
    try:
        result = compactor.compact(context.copy(), tools=environment.tool_specs,
                                   sample_params=sample_params)
    except NothingToCompact:
        return False
    if not isinstance(result, CompactionResult):
        raise TypeError("Expected CompactionResult.")
    session._checkpoint(index, context, result.context_items())
    session._emit(index, result.display_items())
    return True


class _Session:
    """Private fixed-role runtime; no dynamic manager/template API.

    The #2 worker and the board it coordinates through are one experimental,
    opt-in unit. Without them, main takes in-process tasks in FIFO order and the
    watcher supervises it: main yields whenever its turn loop stops, and the
    watcher's own model turn may resume it with a follow-up message.
    """
    def __init__(self, path, settings, *, board_port=0,
                 model_factory=_model_factory, environment_factory=_environment_factory,
                 resume=False, enable_board_auth=True,
                 debug_save_model_binding=False,
                 enable_experimental_worker_board=False,
                 watcher_max_resumes=None):
        if type(enable_board_auth) is not bool:
            raise TypeError("enable_board_auth must be a bool.")
        if type(debug_save_model_binding) is not bool:
            raise TypeError("debug_save_model_binding must be a bool.")
        if type(enable_experimental_worker_board) is not bool:
            raise TypeError("enable_experimental_worker_board must be a bool.")
        if watcher_max_resumes is not None and (
                type(watcher_max_resumes) is not int or watcher_max_resumes < 0):
            raise ValueError("watcher_max_resumes must be a nonnegative integer or None.")
        if watcher_max_resumes is not None and enable_experimental_worker_board:
            raise ValueError("watcher_max_resumes applies only without the experimental worker/board.")
        self.worker_board = enable_experimental_worker_board
        # Per-task resume budget (None: unlimited). With 0 the watcher only
        # observes main's yields and needs no model.
        self._max_resumes = watcher_max_resumes
        self._supervising = not self.worker_board and watcher_max_resumes != 0
        # Only these roles run, resolve, and are kept in config.json.
        self.roles = tuple(i for i in NAMES if i != 2 or self.worker_board)
        if set(self.roles) - set(settings):
            raise ValueError("Auto settings must include every running role.")
        self.path = Path(path).expanduser().absolute()
        self.catalog = getattr(settings, "catalog", BUILTIN_MODEL_CATALOG)
        self.settings = {i: deepcopy(s) for i, s in settings.items()}
        self.sources = dict(getattr(settings, "sources", {}))
        self.bindings = {i: namespace(self.settings[i], self.catalog).model_binding for i in self.roles}
        self.names = {i: self.settings[i]["name"] for i in self.roles}
        self._model_factory, self._environment_factory = model_factory, environment_factory
        self._port = board_port
        self._resume = resume
        self._enable_board_auth = enable_board_auth
        self._debug_save_model_binding = debug_save_model_binding
        self._started = False
        self._resumed = False
        self._baseline = 0
        self._contexts = {}
        self._lock_file = None
        self._stop = threading.Event()
        self._changed = threading.Condition()
        self._states = {i: ("starting", time.monotonic()) for i in self.roles}
        self._events = deque(maxlen=512)
        self._dropped = 0
        self._board_failure_shown = False
        self._view_stale_shown = False
        self._done = {}
        self._expected_watches = set()
        self._watched = set()
        self._watch_queue = queue.Queue()
        # Without the board, main yields to the watcher over the same queue.
        self._channel = None if self.worker_board else YieldChannel(self._stop, self._watch_queue)
        self._watcher_tools = _WatcherTools()
        self._ready = {i: threading.Event() for i in self.roles}
        # Without the board: retained in-process tasks, guarded by _changed.
        self._tasks = []
        self._accepting = False
        self._threads = {}
        self._errors = []
        self._fatal = False
        self._close_lock = threading.Lock()
        self._closed = False
        self.service = None

    def _lock(self):
        lock_path = self.path / ".lock"
        file = lock_path.open("a+")
        try:
            os.chmod(lock_path, 0o600)
            if os.name == "posix":
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                if file.tell() == 0:
                    file.write("\0")
                    file.flush()
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except (OSError, ImportError):
            file.close()
            raise BoardError("Auto save directory is already in use.", 409) from None
        self._lock_file = file

    def _unlock(self):
        if self._lock_file is not None:
            file, self._lock_file = self._lock_file, None
            try:
                if os.name == "posix":
                    import fcntl
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                else:
                    import msvcrt
                    file.seek(0)
                    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                file.close()

    def start(self):
        if self._started or self._closed:
            raise RuntimeError("Auto session cannot be started twice.")
        exists = self.path.exists()
        if exists and not self._resume:
            raise FileExistsError(self.path)
        if not exists:
            self.path.mkdir(mode=0o700)
        self._started = True
        try:
            self._lock()
            self._resumed = exists
            restored = None
            if self._resumed:
                # The board journal marks a worker/board save. Neither mode
                # adopts the other's history; nothing is modified on mismatch.
                journal = (self.path / "index.jsonl").exists()
                if journal and not self.worker_board:
                    raise ValueError("This auto save uses the experimental worker/board; resume it "
                                     "with --enable-experimental-worker-board.")
                if self.worker_board and not journal:
                    raise ValueError("This auto save was created without the experimental "
                                     "worker/board; resume it without "
                                     "--enable-experimental-worker-board.")
                contexts_path = self.path / "contexts"
                for index in self.roles:
                    context = load_interaction_save(contexts_path / f"{index}.jsonl")
                    if not len(context) or not isinstance(context[0], Init):
                        raise ValueError("Auto context history must begin with initialization metadata.")
                    self._contexts[index] = context
                if self.worker_board:
                    restored = Board.restore(self.path)
                    self._baseline = len(restored)
                    for record, _size in restored:
                        if record.kind in {"answer", "result"}:
                            self._done[record.reply_to] = record.success
            else:
                (self.path / "contexts").mkdir(mode=0o700)
            atomic_text(self.path / "config.json", json.dumps(saved_document(
                {i: self.settings[i] for i in self.roles}, self.sources,
            ), indent=2, ensure_ascii=False) + "\n")
            if self.worker_board:
                self.service = BoardService(
                    self.path, port=self._port, restored=restored,
                    enable_board_auth=self._enable_board_auth,
                )
            for index in self.roles:
                thread = threading.Thread(target=self._owner, args=(index,),
                                          name=f"auto-context-{index}")
                self._threads[index] = thread
                thread.start()
            for ready in self._ready.values():
                ready.wait()
            if self._fatal:
                raise _StartupError("Auto context initialization failed (see context error notices).")
            if self._debug_save_model_binding:
                warning = save_debug_model_bindings(
                    self.path / "model-bindings.json",
                    {str(i): binding for i, binding in self.bindings.items()},
                )
                if warning is not None:
                    self._emit(None, (DisplayItem(warning),))
            if self.service is not None:
                with self.service.board.changed:
                    self.service.board.accepting = True
            else:
                with self._changed:
                    self._accepting = True
            self._emit(None, (DisplayItem(f"Save directory: {self.path}"),
                              DisplayItem("Warning: local tools are unsandboxed; use a trusted model and workspace.")))
            if self._resumed:
                self._emit(None, (DisplayItem(
                    "Resumed saved history without replaying old work; command-session IDs and runtime state were not restored."
                ),))
            summaries = "\n".join(
                _role_summary(i, self.settings[i], self.bindings[i], self.sources.get(i),
                              i != -1 or self._supervising)
                for i in self.roles
            )
            self._emit(None, (DisplayItem(summaries),))
            return self
        except BaseException:
            self.close()
            raise

    def _emit(self, index, items, kind="output"):
        with self._changed:
            if len(self._events) == self._events.maxlen:
                self._dropped += 1
            self._events.append(_Event(index, tuple(items), kind))
            self._changed.notify_all()

    def drain_events(self):
        with self._changed:
            events = list(self._events)
            self._events.clear()
            if self._dropped:
                events.insert(0, _Event(None, (DisplayItem(
                    f"{self._dropped} display events omitted; consult the saved context/board logs."),)))
                self._dropped = 0
            if self.service is not None:
                if self.service.board.failed and not self._board_failure_shown:
                    events.append(_Event(None, (DisplayItem("Board persistence failed; no further work will run."),), "error"))
                    self._board_failure_shown = True
                stale = self.service.board.view_stale
                if stale and not self._view_stale_shown:
                    events.append(_Event(None, (DisplayItem(
                        "Board derived views are stale; index.jsonl remains authoritative."
                    ),)))
                self._view_stale_shown = stale
            return events

    def _phase(self, index, phase):
        with self._changed:
            self._states[index] = (phase, time.monotonic())
            self._changed.notify_all()

    def status(self, index):
        with self._changed:
            phase, started = self._states[index]
        text = f"#{index} ({self.names[index]}) - {phase}"
        if phase in _TIMED_PHASES:
            text += f"... {int(time.monotonic() - started)}s"
        return text

    def _is_busy(self, index):
        with self._changed:
            return self._states[index][0] in _ACTIVE_PHASES

    @property
    def board_failed(self):
        return self.service is not None and self.service.board.failed

    @property
    def has_errors(self):
        with self._changed:
            return bool(self._errors) or self.board_failed

    def _error(self, index, message, *, fatal=False):
        with self._changed:
            self._errors.append(message)
            self._fatal = self._fatal or fatal
        self._emit(index, (DisplayItem(message),), "error")
        if fatal:
            self.request_stop()

    def _checkpoint(self, index, context, items=()):
        self._phase(index, "saving")
        context.extend(items)
        try:
            save_interaction_save(self.path / "contexts" / f"{index}.jsonl", context)
        except Exception:
            # Serialization/encoding failures are persistence failures too, not
            # permission to continue from accepted-but-unsaved state.
            raise SaveError("Auto context checkpoint failed; further effects are blocked.") from None

    def _owner(self, index):
        environment = context = None
        try:
            args = namespace(self.settings[index], self.catalog, binding=self.bindings[index])
            config = InteractionConfig.from_namespace(args).snapshot()
            service = self.service
            binding = None if service is None else _Binding(index, service.client(str(index)))
            if index == -1:
                tools = self._watcher_tools.tools() if self._supervising else ()
            else:
                tools = () if binding is None else binding.tools()
            environment = self._environment_factory(index, args, tools)
            if not isinstance(environment, Environment):
                raise TypeError("Environment factory must return an Environment.")
            # An observe-only watcher is host control: no provider/auth
            # initialization. A supervising watcher runs its own model turns.
            needs_model = index != -1 or self._supervising
            model = self._model_factory(index, args) if needs_model else None
            if needs_model and not callable(getattr(model, "sample", None)):
                raise TypeError("Model factory must return a model with sample().")
            current = _instructions(index, self.settings[index],
                                    None if service is None else service.base_url,
                                    supervisor=self._supervising)
            tools_snapshot = Tools(environment.tool_specs)
            if self._resumed:
                context = self._contexts[index]
                recovered = []
                for call in context.pending_tool_calls():
                    recovered.append(ToolResult(
                        call.call_id,
                        "Result unavailable after restart. This call was not rerun and may already have produced side effects.",
                        success=False,
                    ))
                for call in context.pending_user_tool_calls():
                    recovered.append(UserToolResult(ToolResult(
                        call.call.call_id,
                        "User-tool outcome unavailable after restart. The command was not rerun and may already have produced side effects.",
                        success=False,
                    )))
                notice = Instructions(_RESTART_NOTICE if current is None
                                      else current.text + "\n\n" + _RESTART_NOTICE)
                if tools_snapshot != context.latest_tools():
                    recovered.append(tools_snapshot)
                context.extend((*recovered, notice))
                self._checkpoint(index, context)
            else:
                context = InteractionContext((Init(model=args.model),
                                              *(() if current is None else (current,)),
                                              tools_snapshot))
                self._checkpoint(index, context)
            self._phase(index, "quiescent")
            self._ready[index].set()
            if index == -1:
                if self.worker_board:
                    self._watch()
                else:
                    self._supervise(model, environment, config, context)
                return
            cursor = self._baseline
            while not self._stop.is_set():
                source = self._wait_input(index, cursor)
                if source is None or self._stop.is_set():
                    break
                cursor = source.sequence
                if binding is None:
                    self._supervised_job(source, model, environment, config, context)
                else:
                    binding.source = source
                    self._job(index, source, model, environment, config, context, binding)
                    binding.source = None
                self._phase(index, "quiescent")
        except BaseException as exc:
            self._error(index, f"Auto context failed ({type(exc).__name__}); details withheld.", fatal=True)
            if index == 1 and self._channel is not None:
                # Main's loop halted outside a turn; the watcher still observes it.
                self._channel.signal(Yield(
                    context=1, job_id=None, job_text="", kind="failed", resumable=False,
                    reason=type(exc).__name__, revision=0 if context is None else len(context)))
        finally:
            self._ready[index].set()
            if environment is not None:
                close = getattr(environment, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        self._error(index, f"Environment cleanup failed ({type(exc).__name__}).", fatal=True)
            self._phase(index, "closed")

    def _wait_input(self, index, cursor):
        """Block for this role's next source after cursor; None once stopping."""
        if self.service is not None:
            return self.service.board.wait_input("user" if index == 1 else "plan", cursor, self._stop)
        with self._changed:
            while not self._stop.is_set():
                if len(self._tasks) > cursor:
                    return self._tasks[cursor]
                self._changed.wait()
            return None

    def _watch(self):
        """Board-mode watcher: display main's end-of-turn condition only."""
        while True:
            completion = self._watch_queue.get()
            if completion is None:
                return
            with self._changed:
                if completion.record_id in self._watched:
                    continue
                self._emit(-1, (DisplayItem(
                    "[debug] main end-of-turn condition fired: "
                    f"#1 thread={completion.thread_id} source={completion.record_id}",
                    label="debug",
                ),), "debug")
                self._watched.add(completion.record_id)
                self._changed.notify_all()

    def _supervise(self, model, environment, config, context):
        """Watcher owner without the board: drive main as a supervised coroutine."""
        bridge = ThreadPoolExecutor(max_workers=1, thread_name_prefix="auto-yields")
        handle = self._watcher_tools.handle = SupervisedHandle(self._channel, bridge)

        async def decide(yield_):
            # Inline on this owner thread: the watcher's model, tools, and
            # checkpoints stay single-threaded, like every other context.
            return self._decide(yield_, model, environment, config, context)

        def on_fault(fault):
            self._emit(-1, (DisplayItem(_yield_debug(fault.yield_), label="debug"),), "debug")

        try:
            asyncio.run(supervise(handle, decide, on_fault=on_fault))
        except Fault:
            pass  # Propagated: main reports its failure and stops the session.
        finally:
            self._watcher_tools.handle = None
            bridge.shutdown(wait=False)  # No read is in flight once supervise returns.

    def _decide(self, yield_, model, environment, config, context):
        """One watcher decision: a follow-up message for main, or None to release it."""
        self._emit(-1, (DisplayItem(_yield_debug(yield_), label="debug"),), "debug")
        within_budget = self._max_resumes is None or yield_.resumes < self._max_resumes
        if model is not None and within_budget and not self._stop.is_set():
            report = UserInteraction((Message("user", _yield_report(yield_)),))
            self._watcher_tools.begin()
            try:
                self._checkpoint(-1, context, report.context_items())
                self._emit(-1, report.display_items())
                self._turn(-1, model, environment, config, context)
            except _Stopping:
                pass
            except Exception as exc:
                # The watcher's own failure releases main; it is not main's error.
                message = (f"Watcher decision failed ({type(exc).__name__}); main was "
                           "released. Details withheld.")
                if isinstance(exc, SaveError) or context.pending_tool_calls():
                    self._error(-1, message, fatal=True)
                else:
                    self._emit(-1, (DisplayItem(message),), "error")
            finally:
                # A recorded resume_main call stands even if the turn then failed.
                resume = self._watcher_tools.end()
                self._phase(-1, "quiescent")
        else:
            resume = None
        if self._stop.is_set():
            resume = None
        verb = "released" if resume is None else "resumed"
        self._emit(-1, (DisplayItem(f"[debug] watcher {verb} main: #1 source={yield_.job_id}",
                                    label="debug"),), "debug")
        return resume

    def _supervised_job(self, source, model, environment, config, context):
        """One user task for main without the board: turns until the watcher releases it."""
        text, resumes, yield_ = source.content, 0, None
        try:
            while True:
                yield_ = self._attempt(source, text, resumes, model, environment, config, context)
                if not yield_.resumable:
                    with self._changed:
                        self._accepting = False  # A faulted main admits no new tasks.
                self._phase(1, "awaiting watcher")
                resume = self._channel.signal(yield_, context.items)
                if not yield_.resumable:
                    # TODO(supervision): apply a watcher repair verdict here (e.g.
                    # rewrite main's context, hot-reload code) instead of always
                    # propagating the fault.
                    break
                if resume is None or self._stop.is_set():
                    break
                text, resumes = resume, resumes + 1
        finally:
            self._settle(source, yield_, context)

    def _attempt(self, source, text, resumes, model, environment, config, context):
        """Run one main turn on text and describe how its loop stopped."""
        fields = {"context": 1, "job_id": source.record_id, "job_text": source.content,
                  "resumes": resumes}
        try:
            content = text if resumes == 0 else _FOLLOW_UP_HEADER + text
            user = UserInteraction((Message("user", content),))
            self._checkpoint(1, context, user.context_items())
            self._emit(1, user.display_items())
            final = self._turn(1, model, environment, config, context)
        except _Stopping:
            return Yield(kind="stopped", resumable=False, revision=len(context), **fields)
        except Exception as exc:
            # Unsaved state or unresolved tool calls make the history unsafe to continue.
            resumable = not (isinstance(exc, SaveError) or context.pending_tool_calls())
            return Yield(kind="failed", resumable=resumable, reason=type(exc).__name__,
                         failure=exc.failure if isinstance(exc, ModelError) else None,
                         revision=len(context), **fields)
        return Yield(kind="ended", resumable=True, final_text=final,
                     revision=len(context), **fields)

    def _settle(self, source, yield_, context):
        """Record a task's outcome from main's last yield; recovery counts as success."""
        success = yield_ is not None and yield_.kind == "ended"
        if not success:
            if yield_ is not None and yield_.kind == "stopped":
                message = "Stopped before further effects; prior effects may have occurred."
            else:
                reason = "unknown" if yield_ is None else yield_.reason
                message = f"Task failed ({reason}); effects may have occurred. Details withheld."
            # As before, a fault (or unresolved calls at a stop) stops the session.
            fatal = yield_ is None or (not yield_.resumable and (
                yield_.kind == "failed" or bool(context.pending_tool_calls())))
            self._error(1, message, fatal=fatal)
        with self._changed:
            self._done[source.record_id] = success
            self._changed.notify_all()

    def _job(self, index, source, model, environment, config, context, binding):
        """Board-mode job for main or worker (one turn; outcome posted to the board)."""
        success = False
        try:
            if index == 2:
                binding.client.post(source.thread_id, "started", "Worker started this plan.", source.record_id)
            label = "User task" if index == 1 else "Assigned plan from main (#1)"
            user = UserInteraction((Message("user", f"{label}\nBoard thread: {source.thread_id}\nSource record: {source.record_id}\n\n{source.content}"),))
            self._checkpoint(index, context, user.context_items())
            self._emit(index, user.display_items())
            final = self._turn(index, model, environment, config, context)
            if index == 1:
                with self._changed:
                    self._expected_watches.add(source.record_id)
                    self._watch_queue.put(_Completion(source.record_id, source.thread_id))
            binding.client.post(source.thread_id, "answer" if index == 1 else "result",
                                final, source.record_id, success=True)
            success = True
        except Exception as exc:
            message = ("Stopped before further effects; prior effects may have occurred."
                       if isinstance(exc, _Stopping) else
                       f"Task failed ({type(exc).__name__}); effects may have occurred. Details withheld.")
            fatal = (isinstance(exc, SaveError) or self.board_failed or
                     bool(context.pending_tool_calls()))
            self._error(index, message, fatal=fatal)
            try:
                binding.client.post(source.thread_id, "answer" if index == 1 else "result",
                                    message, source.record_id, success=False)
            except BoardError:
                self._error(index, "Outcome could not be published; inspect the saved logs. No automatic retry will run.", fatal=True)
        finally:
            with self._changed:
                self._done[source.record_id] = success
                self._changed.notify_all()

    def _turn(self, index, model, environment, config, context):
        started = time.perf_counter()
        sample_params = config.sample_params()
        samples = 0
        # Pi's overflow recovery: one compact-and-retry per turn.
        overflow_recovered = False
        while config.max_samples is None or samples < config.max_samples:
            self._check_running()
            if auto_compaction_due(model, context, config):
                _compact(self, index, model, environment, config, context, sample_params)
                self._check_running()
            self._phase(index, "sampling")
            try:
                sample = model.sample(
                    context.copy(), tools=environment.tool_specs,
                    sample_params=sample_params,
                )
            except ModelError as exc:
                contribution = (*exc.completed_items, *((exc.failure,) if exc.failure is not None else ()))
                if contribution:
                    self._checkpoint(index, context, (*contribution, ModelSampleBoundary()))
                    self._emit(index, render_interaction_items(contribution))
                calls = tuple(i for i in exc.completed_items if isinstance(i, ToolCall))
                if calls:
                    results = tuple(ToolResult(c.call_id, "Not executed: the model response did not complete.", success=False) for c in calls)
                    self._checkpoint(index, context, results)
                    self._emit(index, render_interaction_items(results, source_calls=calls))
                if (isinstance(exc, ModelContextWindowError) and not overflow_recovered
                        and config.enable_auto_compaction and uses_host_auto_compaction(model)):
                    overflow_recovered = True
                    # A failed compaction fails the task; nothing to compact
                    # leaves the sampling error.
                    if _compact(self, index, model, environment, config, context, sample_params):
                        continue
                raise
            # The failed attempt before an overflow retry does not count.
            samples += 1
            if not isinstance(sample, ModelSample):
                raise TypeError("Expected ModelSample.")
            self._checkpoint(index, context, sample.context_items())
            self._emit(index, sample.display_items())
            if sample.stop_reason == "compaction":
                continue
            if not sample.tool_calls:
                text = sample.last_assistant_text
                if not text or not text.strip():
                    raise MissingFinalText("Model returned no final assistant text.")
                summary = summarize_turn_usage(context.items, elapsed_seconds=time.perf_counter() - started)
                self._checkpoint(index, context, (summary,))
                self._emit(index, render_interaction_items((summary,)))
                return text
            self._check_running()
            self._phase(index, "executing tools")
            outcome = environment.execute_tool_calls(sample.tool_calls)
            self._checkpoint(index, context, outcome.context_items())
            self._emit(index, outcome.display_items(source_calls=sample.tool_calls))
        raise SampleLimitExceeded("Model exceeded the per-turn sample limit.")

    def _check_running(self):
        if self.board_failed:
            raise BoardError("Board persistence failed; no further effects may start.", 503)
        if self._stop.is_set():
            raise _Stopping()

    def submit(self, text, *, request_id=None):
        """Submit one user task for main; returns its handle for task_result().

        With the board this posts a fresh user thread (request_id makes an
        uncertain HTTP outcome retryable). Otherwise the task is queued in
        process, which has no uncertain outcome, so request_id is unused.
        """
        if self._stop.is_set():
            raise BoardError("Auto is not accepting new work.", 503)
        if self.worker_board:
            if self.service is None:
                raise BoardError("Auto is not accepting new work.", 503)
            return self.service.client("user").create_thread(text, request_id=request_id)
        del request_id
        if not isinstance(text, str) or not text.strip():
            raise BoardError("Nonempty content is required.")
        try:
            text.encode("utf-8")
        except UnicodeError:
            # Unencodable text would only fail later, as a fatal checkpoint.
            raise BoardError("Content must be valid UTF-8.") from None
        with self._changed:
            if self._stop.is_set() or not self._accepting:
                raise BoardError("Auto is not accepting new work.", 503)
            if sum(task.record_id not in self._done for task in self._tasks) >= _MAX_PENDING_TASKS:
                raise BoardError("Pending work limit reached; request was not accepted.", 429)
            sequence = len(self._tasks) + 1
            self._tasks.append(_Task(sequence, str(sequence), text))
            self._changed.notify_all()
        return {"record_id": str(sequence)}

    def task_result(self, submitted):
        """None while a submit() handle's task is pending; then its bool outcome.

        With the board this is thread_result() for the posted thread. Without
        it, a task is one main job plus its applicable watch.
        """
        if self.worker_board:
            return self.thread_result(submitted["thread_id"])
        source = submitted["record_id"]
        with self._changed:
            if self._fatal:
                return False
            if not any(task.record_id == source for task in self._tasks):
                raise BoardError("Unknown task.", 404)
            if source not in self._done or (source in self._expected_watches
                                            and source not in self._watched):
                return None
            return self._done[source]

    def thread_result(self, thread_id):
        """None while pending; bool once all jobs and applicable watches settle."""
        if self.service is None:
            raise BoardError("Unknown task thread.", 404)
        # Snapshot done BEFORE records: once main is done its board answer is
        # committed and no further plans may be added. The later board snapshot
        # therefore cannot miss a plan just published by a finishing main.
        with self._changed:
            done = dict(self._done)
            expected, watched = set(self._expected_watches), set(self._watched)
            fatal = self._fatal
        if fatal or self.service.board.failed:
            return False
        sources = [r.record_id for r in self.service.board.records(thread_id) if r.kind in {"user", "plan"}]
        if not sources:
            raise BoardError("Unknown task thread.", 404)
        if not all(key in done for key in sources) or (expected.intersection(sources) - watched):
            return None
        return all(done[key] for key in sources)

    def request_stop(self):
        self._stop.set()
        with self._changed:
            self._accepting = False
            self._changed.notify_all()  # Wakes an idle main waiting for a task.
        if self._channel is not None:
            self._channel.wake()  # Releases a main waiting for the watcher.
        if self.service is not None:
            with self.service.board.changed:
                self.service.board.accepting = False
                self.service.board.changed.notify_all()

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.request_stop()
            # Producers finish before the watcher sentinel and HTTP shutdown.
            for index in (1, 2):
                thread = self._threads.get(index)
                if thread is not None and thread.ident is not None:
                    thread.join()
            # The yield channel's transport: no yield can follow this sentinel.
            self._watch_queue.put(None)
            watcher = self._threads.get(-1)
            if watcher is not None and watcher.ident is not None:
                watcher.join()
            if self.service is not None:
                pending = sum(r.kind in {"user", "plan"} and r.record_id not in self._done
                              for r in self.service.board.records())
                if pending:
                    self._emit(None, (DisplayItem(
                        f"Stopped with {pending} queued/unresolved tasks in the saved board; they were not executed or replayed."),))
                self.service.close()
            else:
                # Owners have exited, so every task without an outcome never started.
                with self._changed:
                    queued = sum(task.record_id not in self._done for task in self._tasks)
                if queued:
                    self._emit(None, (DisplayItem(
                        f"Stopped with {queued} queued user tasks that were not executed; "
                        "without the board, queued tasks are not saved or replayed."),))
            self._unlock()


def _display_events(session, events):
    items = []
    for event in events:
        if event.index is None:
            items.extend(event.items)
            continue
        context = f"#{event.index} ({session.names[event.index]})"
        for item in event.items:
            if item.label is not None:
                label = f"{context} - {item.label}"
                text = f"[{label}]{item.text[len(item.label) + 2:]}"
            else:
                # Keep raw payload lines intact, including diff headers and
                # code that starts with brackets. This is still one item.
                label = context
                text = f"[{label}]\n{item.text}"
            items.append(DisplayItem(text, is_diff=item.is_diff, label=label))
    return items


def _print_events(session):
    for item in _display_events(session, session.drain_events()):
        print(DisplayItem(safe_text(item.text), is_diff=item.is_diff), flush=True)


def _one_prompt(session, prompt, *, display=True):
    output = _print_events if display else lambda value: value.drain_events()
    output(session)
    submitted = session.submit(prompt)
    while True:
        output(session)
        result = session.task_result(submitted)
        if result is not None:
            output(session)
            return 0 if result else 1
        time.sleep(0.01)


def _headless(session):
    board = session.service.board
    while not session._stop.is_set() and not board.failed:
        session.drain_events()
        with board.changed:
            if not session._stop.is_set() and not board.failed:
                board.changed.wait()
    session.drain_events()
    return 1 if session.has_errors else 0


def _local_command(text, selected, session):
    """Pure navigation/exit dispatch; never submit local slash commands."""
    if "\n" in text or "\r" in text:
        raise ValueError("Local commands must be a single line.")
    words = text.split()
    if words in (["/quit"], ["/exit"]):
        return selected, (), True
    if words == ["/contexts"]:
        summaries = "\n".join(
            ("* " if i == selected else "  ") + session.status(i) for i in session.roles
        )
        return selected, (DisplayItem(summaries),), False
    if words and words[0] == "/context" and len(words) in {1, 2}:
        target = selected
        if len(words) == 2:
            roles = {str(i): i for i in session.roles}
            target = roles.get(words[1].removeprefix("#"))
            if target is None:
                choices = [f"/context {i}" for i in session.roles]
                hint = (" or ".join(choices) if len(choices) == 2
                        else ", ".join(choices[:-1]) + ", or " + choices[-1])
                raise ValueError(f"Use {hint}.")
        return target, (DisplayItem(f"Selected #{target} ({session.names[target]})."),), False
    raise ValueError("Use /contexts, /context N, /quit, or /exit.")


def _posted_notice(session, posted):
    if session.worker_board:
        return f"Posted user thread {posted['thread_id']}."
    return f"Queued user task {posted['record_id']} for #1 ({session.names[1]})."


async def _interactive(session, terminal):
    editor, selected = Editor(), 1
    frame = 0
    pending = None
    submitted_text = None
    retry = None
    closing = None
    notices = []
    board_failure_shown = False
    try:
        with terminal:
            while True:
                for key in terminal.read_keys():
                    if key.key in {"c-c", "c-d"}:
                        if closing is None:
                            session.request_stop()
                            closing = asyncio.create_task(asyncio.to_thread(session.close))
                    elif closing is None:
                        if key.key != "c-m":
                            editor = editor.edit(key.key, key.data or "")
                        elif editor.text.strip():
                            if editor.text.lstrip().startswith("/"):
                                try:
                                    selected, result, quit_ = _local_command(editor.text, selected, session)
                                    notices.extend(result)
                                    editor = Editor()
                                    if quit_:
                                        session.request_stop()
                                        closing = asyncio.create_task(asyncio.to_thread(session.close))
                                except ValueError as exc:
                                    notices.append(DisplayItem(str(exc)))
                            elif selected != 1:
                                kind = "user board thread" if session.worker_board else "user task"
                                notices.append(DisplayItem(f"Switch to /context 1 to submit a new {kind}. Draft preserved."))
                            elif pending is not None:
                                notices.append(DisplayItem("A submission is still pending. Draft preserved."))
                            else:
                                submitted_text = editor.text
                                request_id = retry[1] if retry and retry[0] == submitted_text else uuid.uuid4().hex
                                retry = (submitted_text, request_id)
                                pending = asyncio.create_task(asyncio.to_thread(session.submit, submitted_text, request_id=request_id))
                if terminal.closed and closing is None:
                    session.request_stop()
                    closing = asyncio.create_task(asyncio.to_thread(session.close))
                if pending is not None and pending.done():
                    try:
                        posted = pending.result()
                        notices.append(DisplayItem(_posted_notice(session, posted)))
                        retry = None
                        if editor.text == submitted_text:
                            editor = Editor()
                    except Exception as exc:
                        if session.worker_board:
                            notice = "Board submission failed or is uncertain. Retry the unchanged draft to reuse its request ID."
                        else:
                            # An in-process rejection is certain; its reason is host text.
                            reason = str(exc) if isinstance(exc, BoardError) else "Submission failed."
                            notice = f"Task was not queued: {reason} Draft preserved."
                        notices.append(DisplayItem(notice))
                    pending = None
                notices.extend(_display_events(session, session.drain_events()))
                if session.board_failed and not board_failure_shown:
                    session.request_stop()
                    notices.append(DisplayItem("Use /quit to close the failed session."))
                    board_failure_shown = True
                status = "closing - waiting for current work..." if closing else session.status(selected)
                busy = (session._is_busy(selected) or pending is not None
                        or (closing is not None and not closing.done()))
                prompt = f"{_SPINNER[(frame // 16) % len(_SPINNER)]}> " if busy else ":> "
                terminal.render(editor, status, tuple(notices), prompt)
                notices.clear()
                if closing is not None and closing.done():
                    closing.result()
                    break
                frame += 1
                await asyncio.sleep(_FRAME_INTERVAL)
    finally:
        session.request_stop()
        try:
            if pending is not None:
                try:
                    await asyncio.shield(pending)
                except Exception:
                    pass
        finally:
            await asyncio.shield(asyncio.to_thread(session.close))
    return 1 if session.has_errors else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    session = None
    exit_code = 1
    try:
        catalog = frontend_catalog(args)
        if args.list_models:
            print(render_model_catalog(catalog))
            return 0
        args.prompt = load_prompt(args)
        if not 0 <= args.board_port <= 65535:
            raise ValueError("board-port must be between 0 and 65535.")
        if not args.enable_experimental_worker_board:
            # Non-default board options would otherwise be silently ignored.
            if args.board_port != 0 or not args.enable_board_auth:
                raise ValueError("--board-port and --enable-board-auth require "
                                 "--enable-experimental-worker-board.")
            if args.headless and args.prompt is None:
                raise ValueError("--headless without --prompt or --prompt-file requires "
                                 "--enable-experimental-worker-board, whose board accepts tasks.")
        if args.watcher_max_resumes is not None:
            if args.watcher_max_resumes < 0:
                raise ValueError("--watcher-max-resumes must be a nonnegative integer.")
            if args.enable_experimental_worker_board:
                raise ValueError("--watcher-max-resumes does not apply with "
                                 "--enable-experimental-worker-board.")
        board = args.enable_experimental_worker_board
        if args.worker_model is not None and not board:
            raise ValueError("--worker-model requires --enable-experimental-worker-board.")
        if args.watcher_model is not None and (board or args.watcher_max_resumes == 0):
            raise ValueError("--watcher-model has no effect: the watcher runs no model with "
                             "--enable-experimental-worker-board or --watcher-max-resumes 0.")
        if args.prompt is not None and not args.prompt.strip():
            raise ValueError("prompt must not be empty.")
        if (not args.headless and args.prompt is None and not args.print_config
                and (os.name != "posix" or not sys.stdin.isatty() or not sys.stdout.isatty())):
            raise ValueError(
                "Interactive auto requires a POSIX terminal; use --prompt for one-shot mode "
                "or --headless to run without a TTY."
            )
        overrides = {k: v for k, v in vars(args).items() if k in DEFAULTS}
        roles = (1, 2, -1) if board else (1, -1)
        # Only roles that run a model take catalog defaults and need credentials.
        sampling = (1, 2) if board else (1, -1) if args.watcher_max_resumes != 0 else (1,)
        role_models = {index: name for index, name in (
            (1, args.main_model), (-1, args.watcher_model), (2, args.worker_model),
        ) if name is not None}
        role_defaults = {ROLE_INDEX[role]: name for role, name in catalog.auto_models.items()
                         if ROLE_INDEX[role] in sampling}
        save_path = Path(args.save).expanduser().absolute()
        saved = None
        if args.resume and save_path.exists():
            saved = load_saved_config(save_path / "config.json")
        settings = resolve_config(args.context_config, overrides, saved=saved, catalog=catalog,
                                  role_models=role_models, role_defaults=role_defaults,
                                  roles=roles)
        if args.print_config:
            for index in roles:
                print(_role_summary(index, settings[index],
                                    namespace(settings[index], catalog).model_binding,
                                    settings.sources.get(index), index in sampling))
            print(json.dumps(saved_document({i: settings[i] for i in roles}, settings.sources),
                             indent=2, ensure_ascii=False))
            return 0
        _check_credentials(settings, sampling, catalog)
        session = _Session(save_path, settings, board_port=args.board_port,
                           resume=args.resume, enable_board_auth=args.enable_board_auth,
                           debug_save_model_binding=args.debug_save_model_binding,
                           enable_experimental_worker_board=args.enable_experimental_worker_board,
                           watcher_max_resumes=args.watcher_max_resumes)
        session.start()
        if args.enable_experimental_worker_board:
            if not args.enable_board_auth:
                print("Warning: board authentication is disabled; local clients can read board data "
                      "and submit tasks that may run unsandboxed tools.", file=sys.stderr, flush=True)
            print(f"Board: {session.service.base_url}/README.md", flush=True)
        if args.prompt is not None:
            exit_code = _one_prompt(session, args.prompt, display=not args.headless)
        elif args.headless:
            exit_code = _headless(session)
        else:
            exit_code = asyncio.run(_interactive(session, PosixTerminal(sys.stdin, sys.stdout)))
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        # Config errors contain no credential values; unexpected provider/runtime
        # exceptions are deliberately not stringified here.
        detail = (str(exc) if isinstance(exc, (ValueError, BoardError, FileExistsError, _StartupError))
                  else type(exc).__name__)
        print(f"auto failed: {detail}", file=sys.stderr)
    finally:
        if session is not None:
            session.close()
            if not args.headless:
                try:
                    _print_events(session)
                except (OSError, ValueError):
                    exit_code = 1 if exit_code == 0 else exit_code
    return 1 if exit_code == 0 and session.has_errors else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
