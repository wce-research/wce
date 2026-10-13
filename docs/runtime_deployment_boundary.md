# Portable runtime and deployment boundary

WCE's public repository contains the research engine, scientific protocols/results,
frontend, portable Docker runtime and self-contained branded outage artifacts.
Production-specific resource identifiers, addresses, administration and recovery
procedures are not part of the public reproducibility contract.

For development use `make runtime-bootstrap frontend-test`, `make docker-up` and
the documented local port. Runtime configuration is explicit in `.env.example`:
production mode uses Secure/HttpOnly/SameSite cookies, bounded sessions, input/work
limits, a production HTTP server and graceful termination. Health/readiness are
lightweight. Read `docs/production_runtime_hardening.md` for generic behavior/tests.

No production deployment is triggered by cloning, tests or a push to this repository.
No private repository dependency is needed to run or reproduce public WCE.
Do not commit tokens, private keys, live cookies, environment secrets or private
sealed material. Research results remain public; tenant operational data does not.
