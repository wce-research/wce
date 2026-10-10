# Security Policy

## Reporting a vulnerability

Please report security issues privately, through GitHub private vulnerability reporting on this repository
(**Security → Report a vulnerability**). Do not open a public issue for a vulnerability or for anything that
exposes credentials.

A dedicated security e-mail address will be listed here once it is confirmed operational.

Include what you found, how to reproduce it, and the affected commit. We will acknowledge the report and keep you
informed of the fix.

## Scope

This repository contains research code, experiment tooling and a documentation web application. It is not a
production service. Reports are especially useful for:

- credentials, tokens or private data accidentally committed;
- anything that would compromise a sealed evaluation set;
- the cloud-execution tooling under `wce_aws/` (resource creation, tagging, termination, permissions);
- the web application under `app/`.

## Credentials

No credential belongs in this repository. Cloud access uses short-lived single sign-on sessions held outside the
repository. If you find a secret in the tree or in the history, report it privately so it can be rotated.
