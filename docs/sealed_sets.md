# Sealed evaluation sets

A sealed set is an evaluation set frozen before any tuning and used once, for a confirmatory evaluation. It is
never inspected adaptively.

## Status of every sealed set

| Set | Generator | Status |
|---|---|---|
| Deep compositional discovery, version 1 (72 tasks) | `core/deep_world.py` — `sealed_task`, seeds in source | **RETIRED BEFORE EVALUATION — NEVER CONSUMED** |
| Deep compositional discovery, version 2 (72 tasks) | `core/deep_sealed.py` — private salt | **LIVE — never evaluated** |
| Earlier milestones (robust reconstruction, active structure, boundary induction, measurement, perturbation, compositional, continuum, temporal, source, epistemic) | seeds in source | Consumed — historical artifacts; their results are in `results/` |

### Version 1 is retired

Version 1 is generated from seeds written in the source code, and its full specification is in
`docs/sealed_deep_manifest.json`. Publishing the repository therefore publishes its generator, its seeds and its
answers. It was never evaluated. It must never be used for a scientific decision. Its functions and its
fingerprint (`d57e9d761d165ef9d86ec7ee5c9c5c7909c4b1622425c2d2379cbdddaf1a6ca2`) remain in the code and in the
history for integrity only.

The sets of earlier milestones were each evaluated once. Publishing their generators no longer affects any future
evaluation.

## Version 2: private salt, public commitments

The repository contains the generic generator and two fingerprints. It contains neither the salt, nor the sealed
tasks, nor their labels, nor anything that allows them to be reconstructed.

| Public commitment | Value |
|---|---|
| `SALT_SHA256` — SHA-256 of the private salt | `2858653867e2c8b4dd56be8f29d2e720836aca4753bfd7a9d19a99a049942aed` |
| `SEALED_V2_SHA256` — fingerprint of the 72 sealed tasks | `62ddf846d154833ea9041a442453073049001a3fca9b9c13c63d6f17a2078f5c` |

Committed on 2026-10-10, before any use. The set has not been run.

### How it works

1. The salt is 256 random bits, created once by `make sealed-init`, stored outside the repository
   (`~/.config/wce/sealed_v2.salt`, or the file named by `WCE_SEALED_SALT_FILE`), readable by its owner only.
   Nothing creates it implicitly; there is no default value. Without it every sealed function raises
   `SealedSecretMissing`. `*.salt` is ignored by git, and `make sealed-init` refuses a location inside the
   repository and refuses to overwrite an existing salt.
2. The seed of sealed task `i` is derived from `SHA-256(salt | "wce-deep-sealed-v2" | i)`. The layout of the set
   (categories, depths, regimes, noise kinds) is public; the worlds and the observations are not.
3. The fingerprint of the set covers, for each task: the opaque world identifier, the true labels on 96 fixed
   probe points, the regime, the noise kind and rate, the number of positive observations and the number of
   observations.

### Verifying integrity

- **Holder of the salt, at any time:** `make sealed-verify` recomputes both fingerprints and compares them with
  the public commitments.
- **Anyone, after disclosure:** once the single confirmatory evaluation has been run and reported, the salt can be
  published. Check that `SHA-256(salt)` equals `SALT_SHA256`, place the salt in a file, point
  `WCE_SEALED_SALT_FILE` to it and run `make sealed-verify`. A match proves that the evaluated set is the one
  committed before any tuning, and that it was not chosen after seeing results.

### Rules

- The salt is never added to git, never printed, never copied into logs or run artifacts.
- No sealed task, label or answer is written to the repository or to a public artifact.
- Sealed tasks are generated only inside an explicitly authorised confirmatory evaluation.
- Losing the salt destroys the set: keep a private backup.
