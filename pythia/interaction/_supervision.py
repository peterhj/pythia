"""Role-agnostic supervision: a supervised context yields to its supervisor.

A supervised context publishes one :class:`Yield` each time its turn loop
stops, then waits (unless stopping or unsupervised) for the supervisor's
answer: a resume message, or None to release it. The supervisor drives it as a
coroutine through :class:`SupervisedHandle`::

    yield_ = await handle(resume)   # answer the outstanding yield, await the next

A non-resumable yield is raised to the supervisor as :class:`Fault`. The
transport is a plain ``queue.Queue`` whose ``None`` sentinel is sent only after
the supervised owner has exited, so no yield is lost at shutdown.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import queue
import threading
from typing import Awaitable, Callable, Optional, Sequence, Tuple

from .items import InteractionItem, ModelFailure


YIELD_KINDS = ("ended", "failed", "stopped")


@dataclass(frozen=True)
class Yield:
    """Why a supervised turn loop stopped. Pure data; safe to show a model."""

    context: int
    job_id: Optional[str]
    job_text: str
    kind: str
    resumable: bool
    # The supervised log length at the yield: items 0..revision-1 are
    # addressable through the handle's view; the log only grows.
    revision: int
    reason: Optional[str] = None
    failure: Optional[ModelFailure] = None
    final_text: Optional[str] = None
    resumes: int = 0

    def __post_init__(self) -> None:
        if self.kind not in YIELD_KINDS:
            raise ValueError(f"Yield kind must be one of {YIELD_KINDS}.")
        if type(self.resumable) is not bool:
            raise TypeError("resumable must be a bool.")
        for name in ("revision", "resumes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.failure is not None and not isinstance(self.failure, ModelFailure):
            raise TypeError("failure must be ModelFailure or None.")


class Fault(Exception):
    """A non-resumable yield, raised to the supervisor."""

    def __init__(self, yield_: Yield) -> None:
        reason = f" ({yield_.reason})" if yield_.reason else ""
        super().__init__(f"#{yield_.context} {yield_.kind}{reason}")
        self.yield_ = yield_


class _Reply:
    """One-shot answer slot guarded by its channel's condition; first write wins."""

    def __init__(self, condition: threading.Condition) -> None:
        self._condition = condition
        self.done = False
        self.value: Optional[str] = None

    def set(self, value: Optional[str]) -> None:
        with self._condition:
            if not self.done:
                self.done, self.value = True, value
                self._condition.notify_all()


class YieldChannel:
    """Supervised-side end of the yield queue."""

    def __init__(self, stop: threading.Event, transport: Optional[queue.Queue] = None) -> None:
        self._stop = stop
        self._queue = queue.Queue() if transport is None else transport
        self._changed = threading.Condition()
        self._detached = False

    def signal(self, yield_: Yield, view: Sequence[InteractionItem] = ()) -> Optional[str]:
        """Publish a yield and its read-only log view; return the answer.

        Waits until the supervisor answers unless stopping or unsupervised, in
        which case the yield is only a notification. None releases.
        """
        if not isinstance(yield_, Yield):
            raise TypeError("signal() requires a Yield.")
        view = tuple(view)
        with self._changed:
            if self._stop.is_set() or self._detached:
                self._queue.put((yield_, view, None))
                return None
            reply = _Reply(self._changed)
            self._queue.put((yield_, view, reply))
            while not (reply.done or self._stop.is_set() or self._detached):
                self._changed.wait()
            return reply.value if reply.done else None

    def wake(self) -> None:
        """Re-check waits; call after setting the stop event."""
        with self._changed:
            self._changed.notify_all()

    def detach(self) -> None:
        """No supervisor remains: release waits; later yields only notify."""
        with self._changed:
            self._detached = True
            self._changed.notify_all()

    def close(self) -> None:
        """Send the shutdown sentinel, after the supervised owner has exited."""
        self._queue.put(None)


class SupervisedHandle:
    """Supervisor-side coroutine handle: ``yield_ = await handle(resume)``.

    Use from one task on one event loop. ``bridge`` is a dedicated executor for
    blocking queue reads (a single thread suffices).
    """

    def __init__(self, channel: YieldChannel, bridge) -> None:
        self._channel, self._bridge = channel, bridge
        self._pending = None  # (yield_, view, reply) for the outstanding yield
        self._read = None  # in-flight queue read; survives caller cancellation
        self._busy = False

    @property
    def view(self) -> Tuple[InteractionItem, ...]:
        """The supervised log as of the outstanding yield (empty if none)."""
        return () if self._pending is None else self._pending[1]

    async def __call__(self, resume: Optional[str] = None) -> Optional[Yield]:
        """Answer the outstanding yield, then await the next one.

        Returns a resumable Yield, or None once the supervised side has exited.
        Raises Fault for a non-resumable yield; the handle stays usable. A
        cancelled call loses nothing: its read stays in flight for the next
        call, and any resume it carried was already delivered.
        """
        if self._busy:
            raise RuntimeError("The supervised handle already has a caller.")
        if resume is not None and not isinstance(resume, str):
            raise TypeError("resume must be a string or None.")
        if self._pending is not None:
            yield_, _view, reply = self._pending
            if resume is not None and not yield_.resumable:
                raise ValueError("A non-resumable yield cannot be resumed.")
            self._pending = None
            if reply is not None:
                reply.set(resume)
        elif resume is not None:
            raise ValueError("There is no yield to resume.")
        self._busy = True
        try:
            if self._read is None:
                loop = asyncio.get_running_loop()
                self._read = loop.run_in_executor(self._bridge, self._channel._queue.get)
            item = await asyncio.shield(self._read)
            self._read = None
        finally:
            self._busy = False
        if item is None:
            return None
        self._pending = item
        if not item[0].resumable:
            raise Fault(item[0])
        return item[0]

    def detach(self) -> None:
        """Stop supervising: release a waiting supervised side for good."""
        pending, self._pending = self._pending, None
        self._channel.detach()
        if pending is not None and pending[2] is not None:
            pending[2].set(None)


async def supervise(
    handle: SupervisedHandle,
    decide: Callable[[Yield], Awaitable[Optional[str]]],
    *,
    on_fault: Optional[Callable[[Fault], None]] = None,
) -> None:
    """Drive a supervised context until it exits; never leave it waiting.

    ``decide`` maps each resumable yield to a resume message or None. A Fault
    is observed through ``on_fault`` and then propagated.
    """
    resume: Optional[str] = None
    try:
        while True:
            try:
                yield_ = await handle(resume)
            except Fault as fault:
                if on_fault is not None:
                    on_fault(fault)
                # TODO(supervision): handle non-resumable yields here instead
                # of propagating, e.g. rewrite the supervised context or
                # hot-reload code, then resume it (needs a richer reply type).
                raise
            if yield_ is None:
                return
            resume = await decide(yield_)
    finally:
        handle.detach()


__all__ = ["Fault", "SupervisedHandle", "Yield", "YieldChannel", "YIELD_KINDS", "supervise"]
