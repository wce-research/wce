"""Sealed evaluation set, version 2 — generated from a PRIVATE salt that never enters the repository.

Why a second version. The first deep sealed set (`core.deep_world.sealed_task`) is generated from seeds written in
the source code. Publishing the repository publishes its generator, its seeds and therefore its answers. That set is

    RETIRED BEFORE EVALUATION — NEVER CONSUMED

It was never evaluated and must never be used for a scientific decision. Its functions and its fingerprint remain
in `core.deep_world` for historical integrity only.

Protocol of version 2
  1. A random salt (256 bits) is created once, by an explicit command, in a file outside the repository
     (`~/.config/wce/sealed_v2.salt` by default, or the path in `WCE_SEALED_SALT_FILE`). Nothing creates it
     implicitly and there is no default value: without the salt every function below raises `SealedSecretMissing`.
  2. Each sealed task takes its seed from SHA-256(salt, purpose, index). The public generator is generic; the
     tasks, their labels and their answers cannot be reconstructed without the salt.
  3. Two commitments are public: `SALT_SHA256` (fingerprint of the salt) and `SEALED_V2_SHA256` (fingerprint of the
     whole set: task identifiers, true labels on fixed probes, first observations).
  4. Integrity check by the holder of the salt, at any time: `make sealed-verify`.
  5. After the single confirmatory evaluation, the salt can be disclosed. Anyone can then check that
     SHA-256(salt) equals `SALT_SHA256`, regenerate the set, and check `SEALED_V2_SHA256`: the set that was
     evaluated is provably the one committed before any tuning.

This module only generates and fingerprints. It runs no method on the sealed tasks.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from typing import List, Optional

RETIRED_STATUS = "RETIRED BEFORE EVALUATION — NEVER CONSUMED"
PURPOSE = "wce-deep-sealed-v2"
SALT_ENV = "WCE_SEALED_SALT_FILE"
DEFAULT_SALT_FILE = os.path.join("~", ".config", "wce", "sealed_v2.salt")
SALT_SHA256: Optional[str] = "2858653867e2c8b4dd56be8f29d2e720836aca4753bfd7a9d19a99a049942aed"        # public commitment to the private salt (committed 2026-10-10, before any use)
SEALED_V2_SHA256: Optional[str] = "62ddf846d154833ea9041a442453073049001a3fca9b9c13c63d6f17a2078f5c"   # public commitment to the sealed set (72 tasks)
SEED_FLOOR = 10 ** 12                    # derived seeds live far above every dev / validation / retired seed range


class SealedSecretMissing(RuntimeError):
    """The private salt is not available: the sealed set cannot be generated, and nothing replaces it."""


def salt_path() -> str: return os.path.expanduser(os.environ.get(SALT_ENV) or DEFAULT_SALT_FILE)


def _inside_repository(path: str) -> bool:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); return os.path.commonpath([root, os.path.abspath(path)]) == root


def load_salt() -> str:
    """The private salt. No default, no fallback, no implicit creation."""
    p = salt_path()
    if not os.path.isfile(p): raise SealedSecretMissing(f"sealed salt not found at {p}; it is private and is never stored in the repository")
    with open(p, encoding="ascii") as fh: s = fh.read().strip()
    if len(s) < 64 or any(c not in "0123456789abcdef" for c in s): raise SealedSecretMissing(f"sealed salt at {p} is malformed (expected at least 64 hexadecimal characters)")
    return s


def create_salt() -> str:
    """Create the salt ONCE, by explicit request. Refuses to overwrite, and refuses any location inside the
    repository. Returns the public fingerprint of the salt, never the salt."""
    p = salt_path()
    if os.path.exists(p): raise FileExistsError(f"a sealed salt already exists at {p}; refusing to overwrite it")
    if _inside_repository(p): raise ValueError("the sealed salt must live outside the repository")
    os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True); fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as fh: fh.write(secrets.token_hex(32) + "\n")
    return salt_fingerprint()


def salt_fingerprint(salt: Optional[str] = None) -> str: return hashlib.sha256((salt or load_salt()).encode("ascii")).hexdigest()


def derived_seed(index: int, salt: Optional[str] = None) -> int:
    h = hashlib.sha256(f"{salt or load_salt()}|{PURPOSE}|{int(index)}".encode("ascii")).digest(); return SEED_FLOOR + int.from_bytes(h[:6], "big")


def sealed_spec() -> List[dict]:
    """Public layout of the set (72 tasks: categories, depths, regimes, noise kinds). It carries no seed."""
    from core.deep_world import sealed_spec as layout
    return [{k: v for k, v in s.items() if k != "seed"} for s in layout()]


def sealed_task(index: int, salt: Optional[str] = None):
    """Sealed task `index`. Needs the private salt."""
    from core.deep_world import make_task
    s = sealed_spec()[index]; return make_task(derived_seed(index, salt), s["depth"], s["regime"], s["category"], s["kind"], s["unseen"], split="sealed_v2")


def sealed_hash(salt: Optional[str] = None) -> str:
    """Fingerprint of the sealed set: identifiers, true labels on fixed probes, first observations. Computed without
    ever exposing the content to inference code."""
    import torch
    from core.deep_world import full_region
    probe = full_region(96, torch.Generator().manual_seed(7)); sig = []
    for i in range(len(sealed_spec())):
        t = sealed_task(i, salt); sig.append([t.world.uid, "".join(str(int(v)) for v in t.world.label(probe, 0.5).tolist()), t.regime, t.spec.kind, round(t.spec.rate, 3), int(t.y.sum()), len(t.y)])
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()


def verify() -> dict:
    """Integrity check by the holder of the salt: both public commitments must match."""
    if SALT_SHA256 is None or SEALED_V2_SHA256 is None: raise RuntimeError("the sealed set has not been committed yet")
    a, b = salt_fingerprint(), sealed_hash(); return {"salt_matches": a == SALT_SHA256, "set_matches": b == SEALED_V2_SHA256, "ok": a == SALT_SHA256 and b == SEALED_V2_SHA256}


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    try:
        if cmd == "init": print("salt created at", salt_path()); print("SALT_SHA256      =", create_salt()); print("SEALED_V2_SHA256 =", sealed_hash())
        elif cmd == "fingerprint": print("SALT_SHA256      =", salt_fingerprint()); print("SEALED_V2_SHA256 =", sealed_hash())
        else:
            r = verify(); print("sealed set v2:", "INTACT" if r["ok"] else "MISMATCH", r); sys.exit(0 if r["ok"] else 1)
    except (SealedSecretMissing, FileExistsError, ValueError, RuntimeError) as e:
        print("sealed set v2:", e); sys.exit(2)
