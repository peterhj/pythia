"""Installer tests use fake checkouts and never touch the user's installation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


INSTALLER = Path(__file__).resolve().parents[1] / "install.py"
COMMANDS = {"ipythia": "cli", "autopythia": "auto"}


class InstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "home with 'quotes'"
        self.home.mkdir()
        self.env = {**os.environ, "HOME": str(self.home), "PYTHONPATH": "ignored"}
        self.checkout = self.make_checkout("checkout one")
        self.bin = self.home / ".local" / "bin"
        self.link = self.home / ".pythia" / "lib" / "pythia-dev"

    def make_checkout(self, name):
        checkout = self.root / name
        package = checkout / "pythia" / "interaction"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").touch()
        (package / "__init__.py").touch()
        for module in COMMANDS.values():
            (package / f"{module}.py").write_text(
                "import json, sys\n"
                f"print(json.dumps([{name!r}, {module!r}, sys.argv[1:]]))\n"
            )
        return checkout

    def install(self, *args, checkout=None, success=True):
        result = subprocess.run(
            [sys.executable, str(INSTALLER), *args], cwd=checkout or self.checkout,
            env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def assert_launchers(self, bin_dir, checkout_name):
        launch = self.root / "launch"
        launch.mkdir(exist_ok=True)
        # Neither a local pythia module nor a local stdlib module may shadow imports.
        for module in ("pythia", "json"):
            (launch / f"{module}.py").write_text("raise AssertionError('shadowed')\n")
        args = ["--help", "spaces and 'quotes'", "", "$HOME"]
        for command, module in COMMANDS.items():
            path = bin_dir / command
            self.assertEqual(path.stat().st_mode & 0o777, 0o755)
            result = subprocess.run(
                [str(path), *args], cwd=launch, env=self.env,
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(result.stdout), [checkout_name, module, args])

    def test_both_commands_and_checkout_switch(self):
        result = self.install("-e")
        for command in COMMANDS:
            self.assertIn(str(self.bin / command), result.stdout)
        self.assertEqual(self.link.resolve(), self.checkout)
        self.assert_launchers(self.bin, "checkout one")
        before = {name: (self.bin / name).read_bytes() for name in COMMANDS}
        self.install("-e")
        other = self.make_checkout("checkout two")
        self.install("-e", checkout=other)
        self.assertEqual(self.link.resolve(), other)
        self.assertEqual(before, {name: (self.bin / name).read_bytes() for name in COMMANDS})
        self.assert_launchers(self.bin, "checkout two")

    def test_custom_prefix_replaces_legacy_wrapper_or_symlink(self):
        prefix = self.root / "prefix with 'quotes'"
        bin_dir = prefix / "bin"
        bin_dir.mkdir(parents=True)
        target = self.root / "legacy"
        target.write_text("legacy target must not change")
        (bin_dir / "autopythia").symlink_to(target)
        (bin_dir / "ipythia").write_text("old wrapper")
        self.install("-e", "--prefix", str(prefix))
        self.assertEqual(target.read_text(), "legacy target must not change")
        self.assertFalse((bin_dir / "autopythia").is_symlink())
        self.assert_launchers(bin_dir, "checkout one")

    def test_preflight_checks_both_modules_and_destinations(self):
        auto = self.checkout / "pythia" / "interaction" / "auto.py"
        content = auto.read_text()
        auto.unlink()
        self.install("-e", success=False)
        self.assertFalse(self.link.exists())
        self.assertFalse(self.bin.exists())
        auto.write_text(content)
        (self.bin / "autopythia").mkdir(parents=True)
        (self.bin / "ipythia").write_text("unchanged")
        self.install("-e", success=False)
        self.assertFalse(self.link.exists())
        self.assertEqual((self.bin / "ipythia").read_text(), "unchanged")

    def test_requires_editable_and_preserves_non_symlink_checkout_path(self):
        self.install(success=False)
        self.assertFalse(self.link.exists())
        self.link.mkdir(parents=True)
        self.install("-e", success=False)
        self.assertTrue(self.link.is_dir())
        self.assertFalse(self.bin.exists())
