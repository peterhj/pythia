from __future__ import annotations

import json
import os
import re
import select
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import CommandRuntime
from pythia.interaction import DefaultEnvironment
from pythia.interaction import DisplayItem
from pythia.interaction import Environment
from pythia.interaction import Init
from pythia.interaction import Message
from pythia.interaction import ModelSample
from pythia.interaction import PlanState
from pythia.interaction import PlanStep
from pythia.interaction import PlanStore
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import create_apply_patch_tool
from pythia.interaction import create_exec_command_tool
from pythia.interaction import create_update_plan_tool
from pythia.interaction import create_write_stdin_tool
from pythia.interaction.demo import DEFAULT_PROMPT
from pythia.interaction.demo import run_repository_summary


def _execute(environment, name, call_id, arguments):
    result = environment.execute_tool_calls(
        (
            ToolCall(
                name=name,
                call_id=call_id,
                arguments_json=json.dumps(arguments),
            ),
        )
    )
    return result.items[0]


class LocalToolFactoryTests(unittest.TestCase):
    def test_all_tools_can_be_registered_in_a_plain_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CommandRuntime(tmpdir)
            store = PlanStore()
            environment = Environment(
                tools=(
                    create_exec_command_tool(runtime),
                    create_write_stdin_tool(runtime),
                    create_update_plan_tool(store),
                    create_apply_patch_tool(tmpdir),
                )
            )

            self.assertEqual(
                tuple(spec.name for spec in environment.tool_specs),
                (
                    "exec_command",
                    "write_stdin",
                    "update_plan",
                    "apply_patch",
                ),
            )
            self.assertEqual(
                create_apply_patch_tool(tmpdir).spec.parameters["required"],
                ["patch"],
            )
            self.assertIn(
                "unified",
                create_apply_patch_tool(tmpdir).spec.description,
            )
            self.assertIn(
                "absolute paths within the workspace",
                create_apply_patch_tool(tmpdir).spec.description,
            )
            runtime.close()


class PlanToolTests(unittest.TestCase):
    def test_update_plan_stores_immutable_state_and_notifies(self):
        updates = []
        observed_latest = []
        stores = []

        def on_update(state):
            updates.append(state)
            observed_latest.append(stores[0].latest)

        store = PlanStore(on_update=on_update)
        stores.append(store)
        environment = Environment((create_update_plan_tool(store),))

        result = _execute(
            environment,
            "update_plan",
            "plan-1",
            {
                "explanation": "Start with inspection.",
                "plan": [
                    {"step": "Inspect files", "status": "in_progress"},
                    {"step": "Summarize", "status": "pending"},
                ],
            },
        )

        expected = PlanState(
            explanation="Start with inspection.",
            plan=(
                PlanStep(step="Inspect files", status="in_progress"),
                PlanStep(step="Summarize", status="pending"),
            ),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Plan updated")
        self.assertEqual(store.latest, expected)
        self.assertEqual(updates, [expected])
        self.assertEqual(observed_latest, [expected])

    def test_callback_failure_keeps_committed_plan(self):
        def fail_update(state):
            del state
            raise RuntimeError("observer failed")

        store = PlanStore(on_update=fail_update)
        environment = Environment((create_update_plan_tool(store),))

        result = _execute(
            environment,
            "update_plan",
            "plan-1",
            {
                "plan": [
                    {"step": "Inspect files", "status": "completed"},
                ],
            },
        )

        self.assertFalse(result.success)
        self.assertIn("observer failed", result.output)
        self.assertEqual(
            store.latest,
            PlanState(
                explanation=None,
                plan=(
                    PlanStep(
                        step="Inspect files",
                        status="completed",
                    ),
                ),
            ),
        )

    def test_invalid_plan_does_not_replace_previous_state(self):
        store = PlanStore()
        environment = Environment((create_update_plan_tool(store),))
        _execute(
            environment,
            "update_plan",
            "plan-1",
            {
                "plan": [
                    {"step": "Existing", "status": "completed"},
                ],
            },
        )
        before = store.latest

        result = _execute(
            environment,
            "update_plan",
            "plan-2",
            {
                "plan": [
                    {"step": "One", "status": "in_progress"},
                    {"step": "Two", "status": "in_progress"},
                ],
            },
        )

        self.assertFalse(result.success)
        self.assertIn("at most one", result.output)
        self.assertEqual(store.latest, before)


class CommandToolTests(unittest.TestCase):
    def test_exec_failure_cleans_up_child_before_and_after_session_registration(self):
        for method in ("_read_process_output", "_format_response"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmpdir:
                with CommandRuntime(tmpdir) as runtime:
                    terminated = []
                    terminate = runtime._terminate_process

                    def cleanup(process):
                        terminate(process)
                        terminated.append(process)

                    environment = Environment((create_exec_command_tool(runtime),))
                    with mock.patch.object(runtime, method, side_effect=OSError("injected failure")):
                        with mock.patch.object(runtime, "_terminate_process", side_effect=cleanup):
                            result = _execute(environment, "exec_command", "one", {
                                "cmd": "read line", "yield_time_ms": 0,
                            })
                    self.assertFalse(result.success)
                    self.assertEqual(runtime.active_session_ids, ())
                    self.assertEqual(len(terminated), 1)
                    self.assertIsNotNone(terminated[0].poll())
                    self.assertTrue(terminated[0].stdin.closed)
                    self.assertTrue(terminated[0].stdout.closed)

    def test_default_shell_falls_back_to_bash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch(
                    "pythia.interaction.local_tools.command.shutil.which",
                    side_effect=lambda value: (
                        "/usr/bin/bash"
                        if value in {"bash", "/usr/bin/bash"}
                        else None
                    ),
                ):
                    runtime = CommandRuntime(tmpdir)

            self.assertEqual(runtime.shell, "/usr/bin/bash")
            runtime.close()

    def test_exec_command_returns_completed_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CommandRuntime(tmpdir)
            environment = Environment((create_exec_command_tool(runtime),))

            result = _execute(
                environment,
                "exec_command",
                "exec-1",
                {
                    "cmd": "printf hello",
                    "yield_time_ms": 1_000,
                },
            )

            self.assertTrue(result.success)
            self.assertIn("Process exited with code 0", result.output)
            self.assertTrue(result.output.endswith("hello"))
            runtime.close()

    def test_nonzero_exit_is_successful_tool_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CommandRuntime(tmpdir)
            environment = Environment((create_exec_command_tool(runtime),))

            result = _execute(
                environment,
                "exec_command",
                "exec-1",
                {
                    "cmd": "exit 7",
                    "yield_time_ms": 1_000,
                },
            )

            self.assertTrue(result.success)
            self.assertIn("Process exited with code 7", result.output)
            runtime.close()

    def test_write_stdin_uses_short_poll_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with CommandRuntime(tmpdir) as runtime:
                environment = Environment(
                    (
                        create_exec_command_tool(runtime),
                        create_write_stdin_tool(runtime),
                    )
                )
                first = _execute(
                    environment,
                    "exec_command",
                    "exec-1",
                    {
                        "cmd": "sleep 30",
                        "yield_time_ms": 0,
                    },
                )
                match = re.search(r"session ID (\d+)", first.output)
                self.assertIsNotNone(match)
                assert match is not None
                session_id = int(match.group(1))

                with mock.patch.object(
                    runtime,
                    "_read_process_output",
                    return_value="",
                ) as read_output:
                    result = _execute(
                        environment,
                        "write_stdin",
                        "stdin-1",
                        {"session_id": session_id},
                    )

                self.assertTrue(result.success)
                self.assertEqual(read_output.call_args.args[1], 0.25)
                self.assertIn(
                    "Process running with session ID",
                    result.output,
                )

    def test_zero_duration_poll_drains_buffered_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with CommandRuntime(tmpdir) as runtime:
                environment = Environment(
                    (
                        create_exec_command_tool(runtime),
                        create_write_stdin_tool(runtime),
                    )
                )
                first = _execute(
                    environment,
                    "exec_command",
                    "exec-1",
                    {
                        "cmd": "sleep 0.1; printf ready; sleep 30",
                        "yield_time_ms": 0,
                    },
                )
                match = re.search(r"session ID (\d+)", first.output)
                self.assertIsNotNone(match)
                assert match is not None
                session_id = int(match.group(1))
                with runtime._lock:
                    session = runtime._sessions[session_id]
                self.assertTrue(
                    select.select(
                        [session.process.stdout.fileno()],
                        [],
                        [],
                        1.0,
                    )[0]
                )

                result = _execute(
                    environment,
                    "write_stdin",
                    "stdin-1",
                    {
                        "session_id": session_id,
                        "yield_time_ms": 0,
                    },
                )

                self.assertTrue(result.success)
                self.assertIn("ready", result.output)

    def test_exec_command_and_write_stdin_share_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CommandRuntime(
                tmpdir,
                default_exec_yield_time_ms=0,
            )
            environment = Environment(
                (
                    create_exec_command_tool(runtime),
                    create_write_stdin_tool(runtime),
                )
            )
            first = _execute(
                environment,
                "exec_command",
                "exec-1",
                {
                    "cmd": 'read line; printf "got:%s\\n" "$line"',
                    "yield_time_ms": 0,
                },
            )
            match = re.search(r"session ID (\d+)", first.output)
            self.assertIsNotNone(match)
            assert match is not None
            session_id = int(match.group(1))

            second = _execute(
                environment,
                "write_stdin",
                "stdin-1",
                {
                    "session_id": session_id,
                    "chars": "hello\n",
                    "yield_time_ms": 1_000,
                },
            )

            self.assertTrue(second.success)
            self.assertIn("got:hello", second.output)
            self.assertIn("Process exited with code 0", second.output)
            self.assertEqual(runtime.active_session_ids, ())
            runtime.close()

    def test_workdir_escape_and_unknown_session_become_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            runtime = CommandRuntime(workspace)
            environment = Environment(
                (
                    create_exec_command_tool(runtime),
                    create_write_stdin_tool(runtime),
                )
            )

            escaped = _execute(
                environment,
                "exec_command",
                "exec-1",
                {
                    "cmd": "pwd",
                    "workdir": str(outside),
                },
            )
            unknown = _execute(
                environment,
                "write_stdin",
                "stdin-1",
                {"session_id": 999},
            )

            self.assertFalse(escaped.success)
            self.assertIn("escapes configured cwd", escaped.output)
            self.assertFalse(unknown.success)
            self.assertIn("unknown session_id", unknown.output)
            runtime.close()

    def test_output_is_bounded_and_close_terminates_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = CommandRuntime(
                tmpdir,
                default_exec_yield_time_ms=0,
            )
            environment = Environment((create_exec_command_tool(runtime),))

            truncated = _execute(
                environment,
                "exec_command",
                "exec-1",
                {
                    "cmd": (
                        "python3 -c "
                        "'print(\"a\" * 200 + \"TAIL\")'"
                    ),
                    "yield_time_ms": 1_000,
                    "max_output_tokens": 20,
                },
            )
            running = _execute(
                environment,
                "exec_command",
                "exec-2",
                {
                    "cmd": "sleep 30",
                    "yield_time_ms": 0,
                },
            )

            self.assertIn("output truncated", truncated.output)
            self.assertIn("TAIL", truncated.output)
            self.assertIn("Process running with session ID", running.output)
            self.assertTrue(runtime.active_session_ids)
            runtime.close()
            runtime.close()
            self.assertEqual(runtime.active_session_ids, ())


class ApplyPatchToolTests(unittest.TestCase):
    def test_apply_patch_adds_updates_deletes_and_moves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "update.txt").write_text("old\n", encoding="utf-8")
            (root / "delete.txt").write_text("delete\n", encoding="utf-8")
            (root / "move.txt").write_text("before\n", encoding="utf-8")
            environment = Environment((create_apply_patch_tool(root),))
            patch = "\n".join(
                (
                    "*** Begin Patch",
                    "*** Update File: update.txt",
                    "@@",
                    "-old",
                    "+new",
                    "*** Add File: added.txt",
                    "+added",
                    "*** Delete File: delete.txt",
                    "*** Update File: move.txt",
                    "*** Move to: moved.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            )

            result = _execute(
                environment,
                "apply_patch",
                "patch-1",
                {"patch": patch},
            )

            self.assertTrue(result.success)
            self.assertEqual(
                result.output.splitlines(),
                [
                    "M update.txt",
                    "A added.txt",
                    "D delete.txt",
                    "R move.txt -> moved.txt",
                ],
            )
            self.assertEqual(
                (root / "update.txt").read_text(encoding="utf-8"),
                "new\n",
            )
            self.assertEqual(
                (root / "added.txt").read_text(encoding="utf-8"),
                "added\n",
            )
            self.assertFalse((root / "delete.txt").exists())
            self.assertFalse((root / "move.txt").exists())
            self.assertEqual(
                (root / "moved.txt").read_text(encoding="utf-8"),
                "after\n",
            )

    def test_apply_patch_accepts_absolute_paths_within_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            update = root / "update.txt"
            delete = root / "delete.txt"
            move = root / "move.txt"
            added = root / "nested" / "added.txt"
            moved = root / "nested" / "moved.txt"
            update.write_text("old\n", encoding="utf-8")
            delete.write_text("delete\n", encoding="utf-8")
            move.write_text("before\n", encoding="utf-8")
            environment = Environment((create_apply_patch_tool(root),))
            patch = "\n".join(
                (
                    "*** Begin Patch",
                    f"*** Update File: {update}",
                    "@@",
                    "-old",
                    "+new",
                    f"*** Add File: {added}",
                    "+added",
                    f"*** Delete File: {delete}",
                    f"*** Update File: {move}",
                    f"*** Move to: {moved}",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            )

            result = _execute(
                environment,
                "apply_patch",
                "patch-absolute",
                {"patch": patch},
            )

            self.assertTrue(result.success, result.output)
            self.assertEqual(
                result.output.splitlines(),
                [
                    f"M {update}",
                    f"A {added}",
                    f"D {delete}",
                    f"R {move} -> {moved}",
                ],
            )
            self.assertEqual(update.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(added.read_text(encoding="utf-8"), "added\n")
            self.assertFalse(delete.exists())
            self.assertFalse(move.exists())
            self.assertEqual(moved.read_text(encoding="utf-8"), "after\n")

    def test_apply_patch_accepts_unified_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "update.txt").write_text("old\n", encoding="utf-8")
            (root / "delete.txt").write_text("delete\n", encoding="utf-8")
            (root / "move.txt").write_text("before\n", encoding="utf-8")
            (root / "rename-only.txt").write_text(
                "unchanged\n",
                encoding="utf-8",
            )
            environment = Environment((create_apply_patch_tool(root),))
            patch = "\n".join(
                (
                    "diff --git a/update.txt b/update.txt",
                    "--- a/update.txt",
                    "+++ b/update.txt",
                    "@@ -1 +1 @@",
                    "-old",
                    "+new",
                    "diff --git a/added.txt b/added.txt",
                    "new file mode 100644",
                    "--- /dev/null",
                    "+++ b/added.txt",
                    "@@ -0,0 +1 @@",
                    "+added",
                    "diff --git a/delete.txt b/delete.txt",
                    "deleted file mode 100644",
                    "--- a/delete.txt",
                    "+++ /dev/null",
                    "@@ -1 +0,0 @@",
                    "-delete",
                    "diff --git a/move.txt b/moved.txt",
                    "--- a/move.txt",
                    "+++ b/moved.txt",
                    "@@ -1 +1 @@",
                    "-before",
                    "+after",
                    "diff --git a/rename-only.txt b/renamed-only.txt",
                    "similarity index 100%",
                    "rename from rename-only.txt",
                    "rename to renamed-only.txt",
                )
            )

            result = _execute(
                environment,
                "apply_patch",
                "patch-1",
                {"patch": patch},
            )

            self.assertTrue(result.success)
            self.assertEqual(
                result.output.splitlines(),
                [
                    "M update.txt",
                    "A added.txt",
                    "D delete.txt",
                    "R move.txt -> moved.txt",
                    "R rename-only.txt -> renamed-only.txt",
                ],
            )
            self.assertEqual(
                (root / "update.txt").read_text(encoding="utf-8"),
                "new\n",
            )
            self.assertEqual(
                (root / "added.txt").read_text(encoding="utf-8"),
                "added\n",
            )
            self.assertFalse((root / "delete.txt").exists())
            self.assertFalse((root / "move.txt").exists())
            self.assertEqual(
                (root / "moved.txt").read_text(encoding="utf-8"),
                "after\n",
            )
            self.assertFalse((root / "rename-only.txt").exists())
            self.assertEqual(
                (root / "renamed-only.txt").read_text(encoding="utf-8"),
                "unchanged\n",
            )

    def test_apply_patch_accepts_absolute_unified_diff_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            update = root / "update.txt"
            added = root / "nested" / "added.txt"
            update.write_text("old\n", encoding="utf-8")
            environment = Environment((create_apply_patch_tool(root),))
            patch = "\n".join(
                (
                    f"--- {update}",
                    f"+++ {update}",
                    "@@ -1 +1 @@",
                    "-old",
                    "+new",
                    "--- /dev/null",
                    f"+++ {added}",
                    "@@ -0,0 +1 @@",
                    "+added",
                )
            )

            result = _execute(
                environment,
                "apply_patch",
                "patch-absolute-unified",
                {"patch": patch},
            )

            self.assertTrue(result.success, result.output)
            self.assertEqual(
                result.output.splitlines(),
                [f"M {update}", f"A {added}"],
            )
            self.assertEqual(update.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(added.read_text(encoding="utf-8"), "added\n")

    def test_later_invalid_hunk_leaves_all_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("one\n", encoding="utf-8")
            second.write_text("two\n", encoding="utf-8")
            environment = Environment((create_apply_patch_tool(root),))
            patch = "\n".join(
                (
                    "*** Begin Patch",
                    "*** Update File: first.txt",
                    "@@",
                    "-one",
                    "+changed",
                    "*** Update File: second.txt",
                    "@@",
                    "-missing",
                    "+changed",
                    "*** End Patch",
                )
            )

            result = _execute(
                environment,
                "apply_patch",
                "patch-1",
                {"patch": patch},
            )

            self.assertFalse(result.success)
            self.assertIn("unable to apply hunk", result.output)
            self.assertEqual(first.read_text(encoding="utf-8"), "one\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "two\n")

    def test_later_invalid_unified_hunk_leaves_all_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("one\n", encoding="utf-8")
            second.write_text("two\n", encoding="utf-8")
            environment = Environment((create_apply_patch_tool(root),))
            patch = "\n".join(
                (
                    "--- a/first.txt",
                    "+++ b/first.txt",
                    "@@ -1 +1 @@",
                    "-one",
                    "+changed",
                    "--- a/second.txt",
                    "+++ b/second.txt",
                    "@@ -1 +1 @@",
                    "-missing",
                    "+changed",
                )
            )

            result = _execute(
                environment,
                "apply_patch",
                "patch-1",
                {"patch": patch},
            )

            self.assertFalse(result.success)
            self.assertIn("unable to apply hunk", result.output)
            self.assertEqual(first.read_text(encoding="utf-8"), "one\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "two\n")

    def test_apply_patch_rejects_workspace_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            environment = Environment((create_apply_patch_tool(workspace),))

            escaped = _execute(
                environment,
                "apply_patch",
                "patch-1",
                {
                    "patch": "\n".join(
                        (
                            "*** Begin Patch",
                            "*** Add File: ../outside.txt",
                            "+bad",
                            "*** End Patch",
                        )
                    )
                },
            )
            self.assertFalse(escaped.success)
            self.assertIn("escapes workspace", escaped.output)
            self.assertFalse((root / "outside.txt").exists())

            if hasattr(os, "symlink"):
                link = workspace / "link"
                try:
                    link.symlink_to(outside, target_is_directory=True)
                except OSError:
                    return
                symlink_escape = _execute(
                    environment,
                    "apply_patch",
                    "patch-2",
                    {
                        "patch": "\n".join(
                            (
                                "*** Begin Patch",
                                "*** Add File: link/new.txt",
                                "+bad",
                                "*** End Patch",
                            )
                        )
                    },
                )
                self.assertFalse(symlink_escape.success)
                self.assertIn("escapes workspace", symlink_escape.output)
                self.assertFalse((outside / "new.txt").exists())

    def test_apply_patch_rejects_external_absolute_paths_and_prefix_collisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            prefix_collision = Path(f"{workspace}-other")
            prefix_collision.mkdir()
            environment = Environment((create_apply_patch_tool(workspace),))

            for index, target in enumerate(
                (outside / "new.txt", prefix_collision / "new.txt"),
                1,
            ):
                with self.subTest(target=target):
                    result = _execute(
                        environment,
                        "apply_patch",
                        f"patch-external-{index}",
                        {
                            "patch": "\n".join(
                                (
                                    "*** Begin Patch",
                                    f"*** Add File: {target}",
                                    "+bad",
                                    "*** End Patch",
                                )
                            )
                        },
                    )

                    self.assertFalse(result.success)
                    self.assertIn("escapes workspace", result.output)
                    self.assertFalse(target.exists())

    def test_external_absolute_move_rejects_entire_patch_before_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            source = workspace / "source.txt"
            source.write_text("unchanged\n", encoding="utf-8")
            valid_add = workspace / "must-not-exist.txt"
            external_move = outside / "moved.txt"
            environment = Environment((create_apply_patch_tool(workspace),))
            patch = "\n".join(
                (
                    "*** Begin Patch",
                    f"*** Add File: {valid_add}",
                    "+should not be written",
                    f"*** Update File: {source}",
                    f"*** Move to: {external_move}",
                    "*** End Patch",
                )
            )

            result = _execute(
                environment,
                "apply_patch",
                "patch-external-move",
                {"patch": patch},
            )

            self.assertFalse(result.success)
            self.assertIn("escapes workspace", result.output)
            self.assertFalse(valid_add.exists())
            self.assertEqual(source.read_text(encoding="utf-8"), "unchanged\n")
            self.assertFalse(external_move.exists())

    def test_absolute_path_through_symlink_cannot_escape_workspace(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            link = workspace / "link"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are unavailable")
            target = link / "new.txt"
            environment = Environment((create_apply_patch_tool(workspace),))

            result = _execute(
                environment,
                "apply_patch",
                "patch-absolute-symlink",
                {
                    "patch": "\n".join(
                        (
                            "*** Begin Patch",
                            f"*** Add File: {target}",
                            "+bad",
                            "*** End Patch",
                        )
                    )
                },
            )

            self.assertFalse(result.success)
            self.assertIn("escapes workspace", result.output)
            self.assertFalse((outside / "new.txt").exists())


class DefaultEnvironmentTests(unittest.TestCase):
    def test_default_environment_composes_tools_and_plan_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            updates = []
            with DefaultEnvironment(
                cwd=tmpdir,
                on_plan_update=updates.append,
            ) as environment:
                self.assertEqual(
                    tuple(spec.name for spec in environment.tool_specs),
                    (
                        "exec_command",
                        "write_stdin",
                        "update_plan",
                        "apply_patch",
                    ),
                )
                result = _execute(
                    environment,
                    "update_plan",
                    "plan-1",
                    {
                        "plan": [
                            {"step": "Inspect", "status": "completed"},
                        ],
                    },
                )
                self.assertTrue(result.success)
                self.assertEqual(environment.latest_plan, updates[0])

            environment.close()
            self.assertEqual(environment.command_runtime.active_session_ids, ())

    def test_default_environment_enable_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with DefaultEnvironment(
                cwd=tmpdir,
                enable_write_stdin=False,
                enable_apply_patch=False,
            ) as environment:
                self.assertEqual(
                    tuple(spec.name for spec in environment.tool_specs),
                    ("exec_command", "update_plan"),
                )

    def test_disabled_workspace_allows_external_workdir_and_patch_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            absolute_target = outside / "absolute.txt"
            parent_target = root / "parent.txt"

            with DefaultEnvironment(
                cwd=workspace,
                enable_workspace=False,
            ) as environment:
                command = _execute(
                    environment,
                    "exec_command",
                    "external-command",
                    {
                        "cmd": "pwd",
                        "workdir": "../outside",
                        "yield_time_ms": 1_000,
                    },
                )
                patch = _execute(
                    environment,
                    "apply_patch",
                    "external-patch",
                    {
                        "patch": "\n".join((
                            "*** Begin Patch",
                            f"*** Add File: {absolute_target}",
                            "+absolute",
                            "*** Add File: ../parent.txt",
                            "+parent",
                            "*** End Patch",
                        ))
                    },
                )
                patch_spec = next(
                    spec
                    for spec in environment.tool_specs
                    if spec.name == "apply_patch"
                )

            self.assertTrue(command.success)
            self.assertIn(str(outside), command.output)
            self.assertTrue(patch.success)
            self.assertEqual(
                absolute_target.read_text(encoding="utf-8"),
                "absolute\n",
            )
            self.assertEqual(parent_target.read_text(encoding="utf-8"), "parent\n")
            self.assertIn(
                "absolute paths within the workspace",
                patch_spec.description,
            )

    def test_workspace_policy_can_change_without_rebuilding_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()

            with DefaultEnvironment(cwd=workspace) as environment:
                descriptions = tuple(
                    spec.description for spec in environment.tool_specs
                )
                restricted = _execute(
                    environment,
                    "exec_command",
                    "restricted",
                    {"cmd": "pwd", "workdir": str(outside)},
                )
                self.assertFalse(restricted.success)

                environment.set_enable_workspace(False)
                self.assertFalse(environment.enable_workspace)
                unrestricted = _execute(
                    environment,
                    "exec_command",
                    "unrestricted",
                    {
                        "cmd": "pwd",
                        "workdir": str(outside),
                        "yield_time_ms": 1_000,
                    },
                )
                allowed_target = outside / "allowed.txt"
                allowed = _execute(
                    environment,
                    "apply_patch",
                    "allowed",
                    {"patch": "\n".join((
                        "*** Begin Patch",
                        f"*** Add File: {allowed_target}",
                        "+allowed",
                        "*** End Patch",
                    ))},
                )

                environment.set_enable_workspace(True)
                self.assertTrue(environment.enable_workspace)
                denied_target = outside / "denied.txt"
                denied = _execute(
                    environment,
                    "apply_patch",
                    "denied",
                    {"patch": "\n".join((
                        "*** Begin Patch",
                        f"*** Add File: {denied_target}",
                        "+denied",
                        "*** End Patch",
                    ))},
                )

                self.assertEqual(
                    tuple(spec.description for spec in environment.tool_specs),
                    descriptions,
                )

            self.assertTrue(unrestricted.success)
            self.assertIn(str(outside), unrestricted.output)
            self.assertTrue(allowed.success)
            self.assertEqual(
                allowed_target.read_text(encoding="utf-8"),
                "allowed\n",
            )
            self.assertFalse(denied.success)
            self.assertFalse(denied_target.exists())

    def test_workspace_flags_require_real_booleans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for value in (None, 0, 1, "False"):
                with self.subTest(constructor="DefaultEnvironment", value=value):
                    with self.assertRaisesRegex(TypeError, "enable_workspace"):
                        DefaultEnvironment(tmpdir, enable_workspace=value)
                with self.subTest(constructor="CommandRuntime", value=value):
                    with self.assertRaisesRegex(TypeError, "enable_workspace"):
                        CommandRuntime(tmpdir, enable_workspace=value)
                with self.subTest(constructor="apply_patch", value=value):
                    with self.assertRaisesRegex(TypeError, "enable_workspace"):
                        create_apply_patch_tool(
                            tmpdir,
                            enable_workspace=value,
                        )


class _ScriptedRepositoryModel:
    def __init__(self):
        self.calls = []

    def sample(self, context, *, tools=(), sample_params=None):
        self.calls.append((context.copy(), tuple(tools), sample_params))
        if len(self.calls) == 1:
            return ModelSample(
                items=(
                    ToolCall(
                        name="exec_command",
                        call_id="inspect-1",
                        arguments_json=json.dumps(
                            {
                                "cmd": "find . -maxdepth 2 -type f -print",
                                "yield_time_ms": 1_000,
                            }
                        ),
                    ),
                ),
                stop_reason="tool_use",
            )

        results = [
            item
            for item in context.model_items()
            if isinstance(item, ToolResult)
        ]
        if not results or "README.md" not in results[-1].output:
            raise AssertionError("tool result was not appended before follow-up")
        return ModelSample(
            items=(
                Message(
                    role="assistant",
                    content="The repository contains a README and Python source.",
                ),
            ),
            stop_reason="end_turn",
        )


class _LoopingRepositoryModel:
    def __init__(self):
        self.index = 0

    def sample(self, context, *, tools=(), sample_params=None):
        del context, tools, sample_params
        self.index += 1
        return ModelSample(
            items=(
                ToolCall(
                    name="exec_command",
                    call_id=f"loop-{self.index}",
                    arguments_json=json.dumps(
                        {
                            "cmd": "printf loop",
                            "yield_time_ms": 1_000,
                        }
                    ),
                ),
            ),
            stop_reason="tool_use",
        )


class DemoTests(unittest.TestCase):
    def test_repository_summary_demo_accepts_unbounded_sample_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            model = _ScriptedRepositoryModel()
            with mock.patch("builtins.print"):
                with DefaultEnvironment(cwd=root) as environment:
                    summary = run_repository_summary(
                        model,
                        environment,
                        prompt=DEFAULT_PROMPT,
                        max_samples=None,
                    )

        self.assertEqual(
            summary,
            "The repository contains a README and Python source.",
        )
        self.assertEqual(len(model.calls), 2)

    def test_repository_summary_demo_rejects_invalid_sample_limits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with DefaultEnvironment(cwd=tmpdir) as environment:
                for max_samples in (True, 0, -1, 1.5, "2"):
                    with self.subTest(max_samples=max_samples):
                        with self.assertRaisesRegex(
                            (TypeError, ValueError),
                            "positive integer or None",
                        ):
                            run_repository_summary(
                                _LoopingRepositoryModel(),
                                environment,
                                max_samples=max_samples,
                            )

    def test_repository_summary_demo_runs_explicit_tool_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            model = _ScriptedRepositoryModel()
            with mock.patch("builtins.print") as print_mock:
                with DefaultEnvironment(cwd=root) as environment:
                    with mock.patch(
                        "pythia.interaction.demo.perf_counter",
                        side_effect=(20.0, 23.0),
                    ):
                        summary = run_repository_summary(
                            model,
                            environment,
                            prompt=DEFAULT_PROMPT,
                            max_samples=4,
                        )

            self.assertEqual(
                summary,
                "The repository contains a README and Python source.",
            )
            self.assertEqual(len(model.calls), 2)
            self.assertIsInstance(model.calls[0][0].items[0], Init)
            self.assertEqual(
                tuple(spec.name for spec in model.calls[0][1]),
                (
                    "exec_command",
                    "write_stdin",
                    "update_plan",
                    "apply_patch",
                ),
            )
            emitted = tuple(
                call.args[0] for call in print_mock.call_args_list
            )
            self.assertTrue(
                all(isinstance(item, DisplayItem) for item in emitted)
            )
            self.assertEqual(emitted[0].text, "[tools] exec_command, write_stdin, update_plan, apply_patch")
            emitted = emitted[1:]
            rendered = tuple(item.text for item in emitted)
            self.assertEqual(
                str(emitted[0]),
                f"   \x1b[90m[\x1b[0m[user] {DEFAULT_PROMPT}",
            )
            self.assertEqual(
                rendered[1],
                "[tool-call] exec_command (inspect-1)\n"
                "find . -maxdepth 2 -type f -print",
            )
            self.assertTrue(
                rendered[3].startswith(
                    "[tool-ret]  exec_command (inspect-1) [ok]\n"
                )
            )
            self.assertIn("README.md", rendered[3])
            self.assertEqual(
                rendered[4:],
                (
                    "[assistant] The repository contains a README and Python "
                    "source.",
                    "[sample] input=0 output=0 total=0 cached=0",
                    "[turn] input_sum=0 output_sum=0 cold_sum=0 "
                    "cached_sum=0 cached_max=0 context=0 samples=2 "
                    "compactions=0 elapsed=3.00s",
                ),
            )

    def test_repository_summary_demo_bounds_tool_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with DefaultEnvironment(cwd=tmpdir) as environment:
                with mock.patch("builtins.print"):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "within 2 samples",
                    ):
                        run_repository_summary(
                            _LoopingRepositoryModel(),
                            environment,
                            max_samples=2,
                        )


if __name__ == "__main__":
    unittest.main()
