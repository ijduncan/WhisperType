from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from single_instance import SingleInstance


ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import os
import sys
from single_instance import SingleInstance

guard = SingleInstance(sys.argv[1])
if not guard.acquire():
    raise SystemExit(3)
print('acquired', flush=True)
if sys.argv[2] == 'hold':
    sys.stdin.readline()
elif sys.argv[2] == 'abrupt':
    os._exit(0)
guard.release()
"""


class SingleInstanceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "user-data" / "instance.lock"

    def guard(self, path=None):
        guard = SingleInstance(path or self.path)
        self.addCleanup(guard.release)
        return guard

    def probe(self, mode="probe"):
        return subprocess.run(
            [sys.executable, "-c", CHILD, str(self.path), mode],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )

    def test_second_process_cannot_acquire_until_owner_releases(self):
        owner = self.guard()
        self.assertTrue(owner.acquire())
        blocked = self.probe()
        self.assertEqual(blocked.returncode, 3, blocked.stderr)
        self.assertEqual(blocked.stdout, "")
        owner.release()
        accepted = self.probe()
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(accepted.stdout.strip(), "acquired")

    def test_two_handles_in_one_process_also_conflict(self):
        owner, other = self.guard(), self.guard()
        self.assertTrue(owner.acquire())
        self.assertTrue(owner.acquire())
        self.assertFalse(other.acquire())
        owner.release()
        self.assertTrue(other.acquire())

    def test_distinct_user_paths_do_not_conflict(self):
        first = self.guard()
        other_user = self.guard(Path(self.folder.name) / "other-user" / "instance.lock")
        self.assertTrue(first.acquire())
        self.assertTrue(other_user.acquire())

    def test_abrupt_exit_does_not_leave_stale_lock(self):
        exited = self.probe("abrupt")
        self.assertEqual(exited.returncode, 0, exited.stderr)
        self.assertTrue(self.path.exists())
        self.assertTrue(self.guard().acquire())

    def test_killed_process_releases_its_lock(self):
        with subprocess.Popen(
            [sys.executable, "-c", CHILD, str(self.path), "hold"],
            cwd=ROOT,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as child:
            try:
                self.assertEqual(child.stdout.readline().strip(), "acquired")
                waiting = self.guard()
                self.assertFalse(waiting.acquire())
                child.kill()
                child.communicate(timeout=10)
                self.assertTrue(waiting.acquire())
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=10)

    def test_release_keeps_same_lock_file_and_is_idempotent(self):
        owner = self.guard()
        self.assertTrue(owner.acquire())
        owner.release()
        owner.release()
        self.assertTrue(self.path.exists())
        self.assertTrue(owner.acquire())

    def test_filesystem_errors_are_not_reported_as_another_instance(self):
        self.path.parent.mkdir(parents=True)
        self.path.mkdir()
        with self.assertRaises(OSError):
            self.guard().acquire()


if __name__ == "__main__":
    unittest.main()
