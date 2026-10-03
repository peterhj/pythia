"""Role-agnostic supervision primitives, independent of any auto session."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

from pythia.interaction import Message, ModelFailure
from pythia.interaction._supervision import Fault, SupervisedHandle, Yield, YieldChannel, supervise


def make(kind="ended", resumable=True, **fields):
    fields.setdefault("revision", 0)
    return Yield(context=1, job_id=fields.pop("job_id", "1"), job_text="task",
                 kind=kind, resumable=resumable, **fields)


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for test condition")
        time.sleep(0.005)


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.stop = threading.Event()
        self.channel = YieldChannel(self.stop)
        self.bridge = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.bridge.shutdown, wait=False)
        self.answers = []

    def supervised(self, *yields):
        """Publish yields in order on a thread, then close the channel."""
        def run():
            for yield_, view in yields:
                self.answers.append(self.channel.signal(yield_, view))
            self.channel.close()
        thread = threading.Thread(target=run)
        thread.start()
        self.addCleanup(thread.join, 5)
        return thread

    def run_supervisor(self, coroutine):
        return asyncio.run(asyncio.wait_for(coroutine, 5))

    def test_resume_then_release_round_trip_with_log_view(self):
        items = (Message("user", "task"), Message("assistant", "partial"))
        thread = self.supervised((make(revision=2), items),
                                 (make("failed", reason="ModelTransportError", resumes=1), items))
        handle = SupervisedHandle(self.channel, self.bridge)

        async def supervisor():
            first = await handle()
            self.assertEqual((first.kind, first.revision), ("ended", 2))
            self.assertEqual(handle.view, items)
            second = await handle("continue")
            self.assertEqual((second.kind, second.reason, second.resumes),
                             ("failed", "ModelTransportError", 1))
            return await handle(None)

        self.assertIsNone(self.run_supervisor(supervisor()))
        thread.join(5)
        self.assertEqual(self.answers, ["continue", None])
        self.assertEqual(handle.view, ())

    def test_stop_wakes_a_waiting_supervised_side_and_later_yields_only_notify(self):
        result = []
        thread = threading.Thread(target=lambda: result.append(self.channel.signal(make())))
        thread.start()
        wait_for(lambda: self.channel._queue.qsize() == 1)
        self.assertTrue(thread.is_alive())
        self.stop.set()
        self.channel.wake()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])
        self.assertIsNone(self.channel.signal(make(job_id="2")))
        self.assertEqual(self.channel._queue.qsize(), 2)

    def test_fault_is_raised_and_the_handle_stays_usable(self):
        failure = ModelFailure("save", "safe summary")
        self.supervised((make("failed", False, reason="SaveError", failure=failure), ()))
        handle = SupervisedHandle(self.channel, self.bridge)

        async def supervisor():
            with self.assertRaises(Fault) as raised:
                await handle()
            self.assertEqual(raised.exception.yield_.failure, failure)
            self.assertEqual(str(raised.exception), "#1 failed (SaveError)")
            with self.assertRaisesRegex(ValueError, "non-resumable"):
                await handle("resume anyway")
            return await handle(None)

        self.assertIsNone(self.run_supervisor(supervisor()))
        self.assertEqual(self.answers, [None])

    def test_cancelled_calls_lose_no_yield(self):
        def late():
            time.sleep(0.1)
            self.answers.append(self.channel.signal(make(job_id="late")))
            self.channel.close()
        thread = threading.Thread(target=late)
        thread.start()
        handle = SupervisedHandle(self.channel, self.bridge)

        async def supervisor():
            timeouts = 0
            while True:
                try:
                    yield_ = await asyncio.wait_for(handle(), 0.01)
                    break
                except asyncio.TimeoutError:
                    timeouts += 1
            return timeouts, yield_, await handle(None)

        timeouts, yield_, end = self.run_supervisor(supervisor())
        thread.join(5)
        self.assertGreater(timeouts, 0)
        self.assertEqual(yield_.job_id, "late")
        self.assertIsNone(end)
        self.assertEqual(self.answers, [None])

    def test_one_caller_at_a_time_and_resume_requires_a_yield(self):
        handle = SupervisedHandle(self.channel, self.bridge)

        async def supervisor():
            with self.assertRaisesRegex(ValueError, "no yield"):
                await handle("too early")
            first = asyncio.ensure_future(handle())
            await asyncio.sleep(0)
            with self.assertRaisesRegex(RuntimeError, "already has a caller"):
                await handle()
            self.channel.close()
            return await first

        self.assertIsNone(self.run_supervisor(supervisor()))

    def test_supervise_decides_observes_faults_and_always_detaches(self):
        thread = self.supervised(
            (make("failed", reason="ModelTransportError"), ()),
            (make(resumes=1), ()),
            (make("failed", False, reason="SaveError", resumes=1), ()),
            (make(job_id="2"), ()),  # after the fault propagated: notification only
        )
        handle = SupervisedHandle(self.channel, self.bridge)
        decided, faults = [], []

        async def decide(yield_):
            decided.append(yield_.kind)
            return "retry" if yield_.kind == "failed" else None

        with self.assertRaises(Fault):
            self.run_supervisor(supervise(handle, decide, on_fault=faults.append))
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(decided, ["failed", "ended"])
        self.assertEqual(self.answers, ["retry", None, None, None])
        self.assertEqual([fault.yield_.reason for fault in faults], ["SaveError"])

    def test_supervise_returns_when_the_supervised_side_exits(self):
        self.supervised((make(), ()))
        handle = SupervisedHandle(self.channel, self.bridge)

        async def decide(yield_):
            return None

        self.assertIsNone(self.run_supervisor(supervise(handle, decide)))
        self.assertEqual(self.answers, [None])
        self.assertIsNone(self.channel.signal(make(job_id="after")))  # detached: no wait

    def test_yield_validation(self):
        for fields, error in (({"kind": "halted"}, ValueError),
                              ({"resumable": 1}, TypeError),
                              ({"revision": -1}, ValueError),
                              ({"resumes": True}, ValueError),
                              ({"failure": "not safe"}, TypeError)):
            with self.subTest(fields=fields), self.assertRaises(error):
                make(**{"kind": "ended", "resumable": True, **fields})
        with self.assertRaises(TypeError):
            self.channel.signal("not a yield")


if __name__ == "__main__":
    unittest.main()
