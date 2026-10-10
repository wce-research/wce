"""APP_ROOT: every Makefile target runs from the repository root, wherever the repository is cloned."""
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(shutil.which("make"), "make is not installed")
class AppRootTest(unittest.TestCase):
    def run_make(self, cwd, *args):
        return subprocess.run(["make", "--no-print-directory", *args], cwd=cwd, capture_output=True, text=True)

    def test_targets_work_from_another_directory(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_make(d, "-f", os.path.join(ROOT, "Makefile"), "-n", "check")
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("compileall", r.stdout)

    def test_same_recipe_inside_and_outside(self):
        inside = self.run_make(ROOT, "-n", "deep-hp-test").stdout.strip().splitlines()[-1]
        with tempfile.TemporaryDirectory() as d:
            outside = self.run_make(d, "-f", os.path.join(ROOT, "Makefile"), "-n", "deep-hp-test").stdout.strip().splitlines()[-1]
        self.assertEqual(inside, outside)

    def test_no_machine_path_is_written(self):
        mk = open(os.path.join(ROOT, "Makefile")).read()
        self.assertIn("APP_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))", mk); self.assertNotIn("/home/", mk)


if __name__ == "__main__":
    unittest.main()
