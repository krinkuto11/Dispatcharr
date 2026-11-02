"""Exercise the first compatible-release bootstrap in disposable Git repos."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import qoe_release_sync as sync


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


class ReleaseBootstrapTest(unittest.TestCase):
    def test_first_compatible_release_replays_qoe_and_preserves_scheduler(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            repo = root / "fork"
            repo.mkdir()
            git(repo, "init", "-b", "main")
            git(repo, "config", "user.name", "QoE CI")
            git(repo, "config", "user.email", "qoe@example.invalid")
            (repo / "app.txt").write_text("dev base\n")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "dev base")
            base = git(repo, "rev-parse", "HEAD")

            git(repo, "switch", "-c", "feat/qoe-observability")
            (repo / "qoe.txt").write_text("delivery diagnostics\n")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "qoe feature")
            origin = root / "origin.git"
            git(root, "init", "--bare", str(origin))
            git(repo, "remote", "add", "origin", str(origin))
            git(repo, "push", "origin", "feat/qoe-observability")

            git(repo, "switch", "main")
            (repo / "scripts").mkdir()
            (repo / ".github/workflows").mkdir(parents=True)
            (repo / "scripts/qoe_release_sync.py").write_text("scheduler v2\n")
            (repo / ".github/workflows/qoe-release-sync.yml").write_text("workflow v2\n")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "install scheduler")

            git(repo, "switch", "-c", "release", base)
            (repo / "release.txt").write_text("stable compatible\n")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "stable release")
            git(repo, "tag", "v0.34.0")
            upstream = root / "upstream.git"
            git(root, "init", "--bare", str(upstream))
            git(repo, "remote", "add", "upstream", str(upstream))
            git(repo, "push", "upstream", "v0.34.0")
            git(repo, "switch", "main")
            git(repo, "tag", "-d", "v0.34.0")

            original_cwd = Path.cwd()
            try:
                os.chdir(repo)
                with patch.object(sync, "DEV_BASE", base), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(sync.bootstrap("v0.34.0"), 0)
                self.assertIn("status=updated", output.getvalue())
                self.assertEqual((repo / "qoe.txt").read_text(), "delivery diagnostics\n")
                self.assertEqual((repo / "scripts/qoe_release_sync.py").read_text(), "scheduler v2\n")
                self.assertEqual((repo / ".qoe-upstream-tag").read_text(), "v0.34.0\n")
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
