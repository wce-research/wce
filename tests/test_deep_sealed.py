"""Sealed set v2: private salt, public commitments, no default secret; version 1 is retired."""
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import core.deep_sealed as S
import core.deep_world as DW

SALT_A, SALT_B = "ab" * 32, "cd" * 32


class SealedV2Test(unittest.TestCase):
    def test_no_secret_no_set(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {S.SALT_ENV: os.path.join(d, "absent.salt")}):
            for call in (S.load_salt, S.salt_fingerprint, lambda: S.derived_seed(0), lambda: S.sealed_task(0), S.sealed_hash):
                with self.assertRaises(S.SealedSecretMissing): call()

    def test_malformed_secret_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.salt"); open(p, "w").write("short\n")
            with patch.dict(os.environ, {S.SALT_ENV: p}), self.assertRaises(S.SealedSecretMissing): S.load_salt()

    def test_creation_is_explicit_private_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sub", "x.salt")
            with patch.dict(os.environ, {S.SALT_ENV: p}):
                fp = S.create_salt(); self.assertEqual(fp, S.salt_fingerprint()); self.assertEqual(os.stat(p).st_mode & 0o777, 0o600); self.assertEqual(len(S.load_salt()), 64)
                with self.assertRaises(FileExistsError): S.create_salt()

    def test_salt_cannot_live_in_the_repository(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); p = os.path.join(root, "never_created_test.salt")
        with patch.dict(os.environ, {S.SALT_ENV: p}), self.assertRaises(ValueError): S.create_salt()
        self.assertFalse(os.path.exists(p))

    def test_tasks_depend_on_the_salt_only(self):
        a, b, c = S.sealed_task(3, SALT_A), S.sealed_task(3, SALT_A), S.sealed_task(3, SALT_B)
        self.assertEqual(a.world.uid, b.world.uid); self.assertTrue((a.X == b.X).all()); self.assertNotEqual(a.world.uid, c.world.uid); self.assertEqual(a.split, "sealed_v2")
        self.assertNotEqual(S.derived_seed(0, SALT_A), S.derived_seed(1, SALT_A)); self.assertGreaterEqual(S.derived_seed(0, SALT_A), S.SEED_FLOOR)

    def test_layout_is_public_and_carries_no_seed(self):
        spec = S.sealed_spec(); self.assertEqual(len(spec), 72); self.assertTrue(all("seed" not in s for s in spec)); self.assertEqual(sum(s["category"] == "in_class" for s in spec), 36)

    def test_v2_differs_from_the_retired_set(self):
        self.assertNotEqual(S.sealed_task(0, SALT_A).world.uid, DW.sealed_task(0).world.uid)

    def test_public_commitments(self):
        for h in (S.SALT_SHA256, S.SEALED_V2_SHA256): self.assertRegex(h, r"^[0-9a-f]{64}$")
        self.assertNotEqual(S.SEALED_V2_SHA256, DW.SEALED_SHA256); self.assertNotEqual(S.salt_fingerprint(SALT_A), S.SALT_SHA256)

    def test_source_holds_no_salt(self):
        import inspect
        src = inspect.getsource(S); self.assertNotIn("token_hex(32)\")", src); self.assertEqual(src.count("secrets.token_hex"), 1); self.assertIsNone(None if "DEFAULT_SALT =" not in src else 1)

    def test_git_ignores_private_material(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("sealed_v2.salt", "anything.salt", ".secrets/x", ".env.local"):
            self.assertEqual(subprocess.run(["git", "check-ignore", "-q", name], cwd=root).returncode, 0, name)
        self.assertEqual(subprocess.run(["git", "ls-files", "*.salt"], cwd=root, capture_output=True, text=True).stdout, "")


class RetiredV1Test(unittest.TestCase):
    def test_version_1_is_marked_retired(self):
        self.assertEqual(DW.SEALED_STATUS, "RETIRED BEFORE EVALUATION — NEVER CONSUMED"); self.assertEqual(S.RETIRED_STATUS, DW.SEALED_STATUS)

    def test_version_1_fingerprint_is_unchanged(self):
        self.assertEqual(DW.sealed_hash(), DW.SEALED_SHA256)

    def test_no_run_path_uses_a_sealed_set(self):
        import json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); mk = open(os.path.join(root, "Makefile")).read()
        self.assertNotIn("--kind deepsealed", mk); self.assertEqual(json.load(open(os.path.join(root, "docs", "sealed_deep_manifest.json")))["status"], DW.SEALED_STATUS)


if __name__ == "__main__":
    unittest.main()
