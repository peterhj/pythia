"""Board-first fixed-role MVP. Run ``python3 -m pythia.interaction.auto --help``.

All contexts initially wait. User board threads wake #1 (main), its plans wake
#2 (worker), and finalized main turns produce a debug event from #-1 (watcher).
This module deliberately does not change the existing CLI or demo.
"""

from __future__ import annotations

import asyncio
from collections import deque
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
import uuid

from ._auto_board import Board, BoardError, BoardService, atomic_text
from ._auto_config import DEFAULTS, NAMES, SAVED_CONFIG_VERSION, build_parser, load_saved_config, namespace, resolve_config
from ._model_binding_debug import save_debug_model_bindings
from ._cli_editor import Editor, safe_text
from ._cli_terminal import PosixTerminal
from ._prompt import load_prompt
from .compaction import CompactionResult, NothingToCompact, auto_compaction_due
from .compaction import create_default_compactor, uses_host_auto_compaction
from .context import InteractionContext
from .default_environment import DefaultEnvironment
from .display import DisplayItem, render_interaction_items
from .environment import Environment, Tool, ToolOutcome, ToolSpec
from .items import Init, Instructions, Message, ModelSampleBoundary, ToolCall, ToolResult
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
_ACTIVE_PHASES = {"starting", "sampling", "compacting", "executing tools", "saving"}


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


def _instructions(index, settings, base_url):
    body = settings["instructions"]
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


class _Stopping(RuntimeError):
    pass


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


def _environment_factory(index, args, tools):
    if index == -1:
        return Environment()
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
    """Private fixed-role runtime; no dynamic manager/template API."""
    def __init__(self, path, settings, *, board_port=0,
                 model_factory=_model_factory, environment_factory=_environment_factory,
                 resume=False, enable_board_auth=True,
                 debug_save_model_binding=False):
        if type(enable_board_auth) is not bool:
            raise TypeError("enable_board_auth must be a bool.")
        if type(debug_save_model_binding) is not bool:
            raise TypeError("debug_save_model_binding must be a bool.")
        self.path = Path(path).expanduser().absolute()
        self.catalog = getattr(settings, "catalog", BUILTIN_MODEL_CATALOG)
        self.settings = {i: deepcopy(s) for i, s in settings.items()}
        self.bindings = {i: namespace(s, self.catalog).model_binding for i, s in self.settings.items()}
        self.names = {i: self.settings[i]["name"] for i in NAMES}
        self._model_factory, self._environment_factory = model_factory, environment_factory
        self._port = board_port
        self._resume = resume
        self._enable_board_auth = enable_board_auth
        self._debug_save_model_binding = debug_save_model_binding
        self._resumed = False
        self._baseline = 0
        self._contexts = {}
        self._lock_file = None
        self._stop = threading.Event()
        self._changed = threading.Condition()
        self._states = {i: ("starting", time.monotonic()) for i in NAMES}
        self._events = deque(maxlen=512)
        self._dropped = 0
        self._board_failure_shown = False
        self._view_stale_shown = False
        self._done = {}
        self._expected_watches = set()
        self._watched = set()
        self._watch_queue = queue.Queue()
        self._ready = {i: threading.Event() for i in NAMES}
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
        if self.service is not None or self._closed:
            raise RuntimeError("Auto session cannot be started twice.")
        exists = self.path.exists()
        if exists and not self._resume:
            raise FileExistsError(self.path)
        if not exists:
            self.path.mkdir(mode=0o700)
        try:
            self._lock()
            self._resumed = exists
            restored = None
            if self._resumed:
                contexts_path = self.path / "contexts"
                for index in NAMES:
                    context = load_interaction_save(contexts_path / f"{index}.jsonl")
                    if not len(context) or not isinstance(context[0], Init):
                        raise ValueError("Auto context history must begin with initialization metadata.")
                    self._contexts[index] = context
                restored = Board.restore(self.path)
                self._baseline = len(restored)
                for record, _size in restored:
                    if record.kind in {"answer", "result"}:
                        self._done[record.reply_to] = record.success
            else:
                (self.path / "contexts").mkdir(mode=0o700)
            atomic_text(self.path / "config.json", json.dumps({
                "version": SAVED_CONFIG_VERSION,
                "contexts": {str(i): s for i, s in self.settings.items()},
            }, indent=2, ensure_ascii=False) + "\n")
            self.service = BoardService(
                self.path, port=self._port, restored=restored,
                enable_board_auth=self._enable_board_auth,
            )
            for index in NAMES:
                thread = threading.Thread(target=self._owner, args=(index,),
                                          name=f"auto-context-{index}")
                self._threads[index] = thread
                thread.start()
            for ready in self._ready.values():
                ready.wait()
            if self._fatal:
                raise RuntimeError("Auto context initialization failed (see context error notices).")
            if self._debug_save_model_binding:
                warning = save_debug_model_bindings(
                    self.path / "model-bindings.json",
                    {str(i): binding for i, binding in self.bindings.items()},
                )
                if warning is not None:
                    self._emit(None, (DisplayItem(warning),))
            with self.service.board.changed:
                self.service.board.accepting = True
            self._emit(None, (DisplayItem(f"Save directory: {self.path}"),
                              DisplayItem("Warning: local tools are unsandboxed; use a trusted model and workspace.")))
            if self._resumed:
                self._emit(None, (DisplayItem(
                    "Resumed saved history without replaying old work; command-session IDs and runtime state were not restored."
                ),))
            summaries = "\n".join(
                f"#{i} ({s['name']}): {self.bindings[i].api} / {s['model'] or '(server default)'}"
                for i, s in self.settings.items()
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
        if phase in {"sampling", "compacting", "executing tools", "saving"}:
            text += f"... {int(time.monotonic() - started)}s"
        return text

    def _is_busy(self, index):
        with self._changed:
            return self._states[index][0] in _ACTIVE_PHASES

    @property
    def has_errors(self):
        with self._changed:
            return bool(self._errors) or (self.service is not None and self.service.board.failed)

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
        environment = None
        try:
            args = namespace(self.settings[index], self.catalog, binding=self.bindings[index])
            config = InteractionConfig.from_namespace(args).snapshot()
            binding = _Binding(index, self.service.client(str(index)))
            environment = self._environment_factory(index, args, () if index == -1 else binding.tools())
            if not isinstance(environment, Environment):
                raise TypeError("Environment factory must return an Environment.")
            # The watcher is host control, so no provider/auth initialization is
            # needed to display its debug event. Its settings remain independent.
            model = None if index == -1 else self._model_factory(index, args)
            if index != -1 and not callable(getattr(model, "sample", None)):
                raise TypeError("Model factory must return a model with sample().")
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
                current = _instructions(index, self.settings[index], self.service.base_url)
                current = Instructions(
                    current.text + "\n\nRestart notice: saved history was resumed without "
                    "restoring old command-session IDs or runtime state."
                )
                context.extend((*recovered, current))
                self._checkpoint(index, context)
            else:
                context = InteractionContext((Init(model=args.model),
                                              _instructions(index, self.settings[index], self.service.base_url)))
                self._checkpoint(index, context)
            self._phase(index, "quiescent")
            self._ready[index].set()
            if index == -1:
                self._watch()
                return
            cursor = self._baseline
            while not self._stop.is_set():
                source = self.service.board.wait_input("user" if index == 1 else "plan", cursor, self._stop)
                if source is None or self._stop.is_set():
                    break
                cursor = source.sequence
                binding.source = source
                self._job(index, source, model, environment, config, context, binding)
                binding.source = None
                self._phase(index, "quiescent")
        except BaseException as exc:
            self._error(index, f"Auto context failed ({type(exc).__name__}); details withheld.", fatal=True)
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

    def _watch(self):
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

    def _job(self, index, source, model, environment, config, context, binding):
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
            fatal = (isinstance(exc, SaveError) or self.service.board.failed or
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
                    raise RuntimeError("Model returned no final assistant text.")
                summary = summarize_turn_usage(context.items, elapsed_seconds=time.perf_counter() - started)
                self._checkpoint(index, context, (summary,))
                self._emit(index, render_interaction_items((summary,)))
                return text
            self._check_running()
            self._phase(index, "executing tools")
            outcome = environment.execute_tool_calls(sample.tool_calls)
            self._checkpoint(index, context, outcome.context_items())
            self._emit(index, outcome.display_items(source_calls=sample.tool_calls))
        raise RuntimeError("Model exceeded the per-turn sample limit.")

    def _check_running(self):
        if self.service.board.failed:
            raise BoardError("Board persistence failed; no further effects may start.", 503)
        if self._stop.is_set():
            raise _Stopping()

    def submit(self, text, *, request_id=None):
        if self._stop.is_set() or self.service is None:
            raise BoardError("Auto is not accepting new work.", 503)
        return self.service.client("user").create_thread(text, request_id=request_id)

    def thread_result(self, thread_id):
        """None while pending; bool once all jobs and applicable watches settle."""
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
        result = session.thread_result(submitted["thread_id"])
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
            ("* " if i == selected else "  ") + session.status(i) for i in NAMES
        )
        return selected, (DisplayItem(summaries),), False
    if words and words[0] == "/context" and len(words) in {1, 2}:
        target = selected
        if len(words) == 2:
            value = words[1].removeprefix("#")
            if value not in {"1", "2", "-1"}:
                raise ValueError("Use /context 1, /context 2, or /context -1.")
            target = int(value)
        return target, (DisplayItem(f"Selected #{target} ({session.names[target]})."),), False
    raise ValueError("Use /contexts, /context N, /quit, or /exit.")


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
                                notices.append(DisplayItem("Switch to /context 1 to submit a new user board thread. Draft preserved."))
                            elif pending is not None:
                                notices.append(DisplayItem("A board submission is still pending. Draft preserved."))
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
                        notices.append(DisplayItem(f"Posted user thread {posted['thread_id']}."))
                        retry = None
                        if editor.text == submitted_text:
                            editor = Editor()
                    except Exception:
                        notices.append(DisplayItem("Board submission failed or is uncertain. Retry the unchanged draft to reuse its request ID."))
                    pending = None
                notices.extend(_display_events(session, session.drain_events()))
                if session.service.board.failed and not board_failure_shown:
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
        if args.prompt is not None and not args.prompt.strip():
            raise ValueError("prompt must not be empty.")
        if (not args.headless and args.prompt is None
                and (os.name != "posix" or not sys.stdin.isatty() or not sys.stdout.isatty())):
            raise ValueError(
                "Interactive auto requires a POSIX terminal; use --prompt for one-shot mode "
                "or --headless to run without a TTY."
            )
        overrides = {k: v for k, v in vars(args).items() if k in DEFAULTS}
        save_path = Path(args.save).expanduser().absolute()
        saved = None
        if args.resume and save_path.exists():
            saved = load_saved_config(save_path / "config.json")
        settings = resolve_config(args.context_config, overrides, saved=saved, catalog=catalog)
        session = _Session(save_path, settings, board_port=args.board_port,
                           resume=args.resume, enable_board_auth=args.enable_board_auth,
                           debug_save_model_binding=args.debug_save_model_binding)
        session.start()
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
        detail = str(exc) if isinstance(exc, (ValueError, BoardError, FileExistsError)) else type(exc).__name__
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
