# Security Policy

## Reporting a vulnerability

No dedicated private security channel is operational yet. Until one is listed here:

- do **not** publish the details of a vulnerability, a credential or anything that would compromise a sealed
  evaluation set in a public issue;
- open a minimal public issue on this repository asking for a private channel, without technical details. A
  maintainer will reply with a way to send the report privately.

This section will be replaced by a confirmed private contact as soon as one is operational.

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
