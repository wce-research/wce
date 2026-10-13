# WCE Labs — production runtime hardening

Date: 2026-10-12. See the [portable deployment boundary](runtime_deployment_boundary.md) for the public runtime scope. This milestone changes the local application runtime only. No AWS identity was used; no cloud resource, certificate, DNS, Cloudflare, mail record, public port or tunnel was created or changed.

## Initial blockers and implementation

| Blocker | Remediation | Evidence |
| --- | --- | --- |
| Development HTTP server | Waitress 3.0.2, hash-pinned lightweight WSGI server; transport-independent existing route handlers | HTTP/frontend/browser regressions, explicit transport configuration tests |
| SIGTERM ignored / forced exit 137 | Python exec-form PID 1; explicit SIGTERM/SIGINT handling, readiness drops, accepts stop, active work/output drains within configured deadline, dispatcher/channels close | In-flight SIGTERM regression, real Docker stop and SIGTERM tests |
| Unbounded history/work/concurrency | Fixed connection/thread/queue caps, API admission, per-session lock, cumulative budgets, recent history and causal checkpoints | Adversarial repetition, history, concurrent producers/sessions, simulation/input tests |
| Missing Secure cookie | Explicit production mode requires canonical HTTPS origin; Secure + HttpOnly + Strict + Max-Age | Development/production cookie tests and real production Docker API |

No framework migration, route redesign, template/CSS/branding change or scientific core modification. No `.github/workflows` directory exists in the inspected checkout; production promotion CI remains a proposal and no workflow was activated. New messages are authored in English with French translations in the existing language adapter. Test inventory was regenerated from source rather than copying a new count into templates.

## Server and concurrency choice

[Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/) is pure Python, supports WSGI and a fixed worker-thread pool, and avoids pre-fork workers or a database. The application is now a WSGI callable; the old Handler has become a transport-independent request/response adapter. Production no longer imports/runs `http.server`. Existing scientific `Session` remains unchanged; only web sessions receive bounded execution behavior.

One process, four worker threads, one I/O loop. No replication/shared state claimed. Two API executions at most, nonblocking per-session lock, fixed request admission and fixed dispatcher queue. Excess session work does not serialize every visitor behind a global scientific lock. Waitress buffers/parses requests before occupying a worker; no pipeline lookahead, bounded accepted channels, backlog, headers/body and output-buffer thresholds. Protocol-overload/invalid framing may close a connection; this is distinct from controlled application saturation responses.

Measurement caught an initial fractional event-loop timeout being coerced by Waitress to integer zero. That provisional configuration consumed CPU at idle and its benchmark results were discarded. The final timeout is the supported integer 1 second, with a configuration regression and a real Docker idle-CPU median assertion.

Thread dispatcher lifecycle is explicitly owned because closing a single Waitress listener is not sufficient to shut down its worker pool. The pinned server's create_server/dispatcher integration is tested; dependency upgrades require replay of lifecycle, queue and proxy regressions. No new scientific compute/training dependencies are installed.

## Resource boundaries and scientific meaning

| Boundary | Default / hard configuration ceiling | Result |
| --- | --- | --- |
| Worker threads | 4 / 16 | Fixed thread pool |
| Accepted connections / dispatcher queue / backlog | 128 / 256 each | Bounded socket/queue retention; overload can close/reject |
| API executions | 2, fixed | Excess execution returns 503 with Retry-After |
| Concurrent commands in one session | 1, fixed | 409 with Retry-After |
| Global API token bucket | 50 units/s, burst 100; ceilings 100/200 | 429 with Retry-After; no growing per-IP map |
| Command cost | 1 unit; advance adds ceil(seconds/10) | New cookies cannot bypass global expensive-work budget |
| Live stored sessions | 100 ceiling | New session refused with 429; active users not evicted to admit strangers |
| Inactivity / absolute lifetime | 900 / 3,600 seconds; maximum 3,600 | Cleanup on session access; expired state replaced; dictionary always capped |
| Per-session attempts | 1,000 ceiling | 429 on exhausted lifetime command budget |
| Per-session physics steps | 6,000 ceiling; 60 seconds/command, dt=.1 | Reserve cumulative work before execution, even if later rejected |
| Recent world history | 128 entries ceiling | Old entries discarded, current world preserved |
| Causal graph | 64 nodes ceiling | Rebase at a named checkpoint before next transformation |
| Body / command / headers | 4,096 bytes / 500 characters / 8,192 bytes | 413/400 or transport rejection |
| Numeric parameters | Finite, absolute value ≤1,000,000 | Invalid values rejected before scientific execution |
| Idle channel cleanup | 10 seconds, scan interval 1 second | Inactive clients released; trickling clients still consume only bounded channels |
| Local Compose container | 256 MiB, 1 CPU, 64 PIDs, 16 MiB tmpfs | Resource isolation supplements application limits |

Every admitted API POST consumes a rate token before body/command parsing, including malformed input; simulations consume additional work units. Per-session budgets count parsed execution attempts; malformed/rejected transport inputs cannot grow histories. Histories and graphs previously accumulated path copies indefinitely. At a checkpoint, old graph parent/child references are disconnected and the new root records current world properties. The recent history remains bounded separately. This does **not** preserve a complete lifetime causal trace: API responses explicitly state bounded recent history/checkpoint scope and checkpoint count. Core/research experiments retain their original full histories and protocols.

Current world entities/relations are fixed by the laboratory command vocabulary; there is no user-defined entity/upload/file creation endpoint. Numeric inputs, repeated actions and simulation cannot introduce an obvious unbounded session structure. Absolute lifetime and cumulative budgets prevent indefinitely extending one session. New-session abuse still consumes the global budget and reaches the fixed session cap. A distributed adversary can exhaust these budgets and reduce availability; future proxy/edge admission remains necessary.

429 indicates capacity/rate/session-work exhaustion; 503 indicates temporary execution/service saturation; 409 indicates a command already running in that session. No indefinite wait queue is introduced. A browser session expires or a new browser session can be started when its budget is exhausted. A 60-second simulation remains supported; a web session is not an unlimited experiment runner.

## Configuration, sessions and reverse proxy contract

`make runtime-bootstrap` creates ignored `.venv-runtime` and installs hash-pinned `requirements-web.txt`. `make frontend-serve` uses that runtime. All verification commands below are Makefile targets.

Production requires **both**:

```text
WCE_MODE=production
WCE_PUBLIC_ORIGIN=https://wcelabs.com
```

Missing origin, HTTP origin, credentials, path/query/fragment/port in origin, invalid mode and out-of-range/malformed limits fail startup before listening. Development mode defaults explicitly and rejects a production-origin setting. Local Compose remains development mode unless these variables are supplied; localhost HTTP remains usable. Never enable production cookies on a plain HTTP browser origin.

Session cookie: `HttpOnly; SameSite=Strict; Path=/; Max-Age=3600`, plus `Secure` exclusively in production. Tokens use cryptographic randomness and are stored only in memory/HttpOnly cookie. No token, AWS credentials or secret enters logs/images. Session loss on restart and lack of cross-process persistence are intentional launch limitations. No database or server-secret registry was introduced.

Production validates the original Host against the configured canonical host and requires exact HTTPS Origin on API POST. The proxy must preserve public Host and Origin and terminate authenticated HTTPS. Relative redirects preserve the browser's public origin. Health/readiness remain usable on loopback without public Host. Forwarded/CF client headers are never used for cookies, Host admission or budgets; Waitress strips untrusted proxy headers and no trusted proxy is configured. The request scheme is explicitly HTTPS in production; client address is the immediate peer, not an asserted forwarded visitor. This prevents spoofing and avoids pretending to have per-user/IP quotas.

Future proxy ownership: authenticate Cloudflare, reject unknown SNI/Host, bind app port to localhost/private container network, apply connection/body/header/deadline/edge rate controls, and set trusted headers only if a later feature actually needs them. Direct callers spoofing canonical Host/Origin still require network/authenticated-origin protection; application Host checks are not origin authentication. No proxy, certificate or TLS listener was installed here.

Existing CSP/nosniff/frame protection retained; Referrer-Policy and Permissions-Policy added. HSTS remains a future edge/proxy decision after real HTTPS validation; no preload/subdomain policy applied. HTTP caching remains no-store: no cache policy or static migration in this milestone.

## Health, readiness and lifecycle

- `/healthz`: inexpensive process liveness, no experiment/session allocation.
- `/readyz`: 200 when ready, 503 when draining/not ready. Startup verifies required public files and representative home/lab rendering before listening; it does not run an experiment.
- SIGTERM/SIGINT: mark not ready, stop accepts, finish bounded active tasks, cancel pending tasks, flush output within the same bounded deadline, close channels/listener and exit normally.
- `WCE_SHUTDOWN_SECONDS`: default 4, allowed 1–10; Docker Compose grace period 8 seconds for the default. If changed, configure container grace greater than app deadline plus margin.
- Unexpected I/O-loop death exits instead of leaving a falsely live process. Readiness reflects lifecycle, not a full dependency/route audit; external representative route probes remain necessary.

Docker uses exec-form Python PID 1, UID 999, read-only filesystem, restricted tmpfs, dropped capabilities, no-new-privileges, loopback host port 3018, rotated stdout/stderr logs. Image healthcheck now uses readiness. Compose resource limits are explicit. Unhealthy-running containers still need a future bounded watchdog; Docker restart policy alone handles process exits only.

Lifecycle checks also inspect /app for excluded credential/private paths and key/salt files, image environment for AWS/Google credentials and layer command history for credential-variable injection. These are targeted hygiene checks, not a claim that arbitrary secrets can be perfectly detected. Build context explicitly excludes local runtime environment, AWS/secrets/keys and tools. Dependency install uses version+wheel hash, no cache. No scanner currently certifies OS-layer CVE status; SBOM/vulnerability scanning remains required before public promotion.

## Measured before / after

Same short, warm, closed-loop local benchmark as the review: 1/10/50/100 active clients, no think time, equal ten-route mix including 10% lab queries, four-second client timeout, six-second request window. Raw evidence records success and all-attempt latencies separately. Limits/transport differ intentionally; sessions persist across test levels, so later levels can hit the now-non-evicting session cap. Controlled refusals must not be described as successful requests. This is not AWS capacity or real-browser traffic.

| Measurement | Before | After |
| --- | --- | --- |
| Docker-reported image bytes | 47,946,694 | 48,073,013 (45.85 MiB) |
| No-cache layer build, base available | 28.21 s | 32.863 s; includes pinned dependency fetch |
| Run to first health, 1 CPU | 0.772 s | 0.825 s |
| Median idle CPU, 1 CPU | 0.01% | 0.15% |
| Stop after benchmark | 5.45 s, exit 137 | 0.567 s, exit 0, no OOM |

Idle CPU samples include occasional short probe/startup spikes; medians describe quiet behavior, not the maximum. No cold-base-download claim. Local Compose uses a separately rebuilt image with the current test-inventory fingerprint; its runtime Python sources and inventory were checked against the workspace. Raw benchmark `commit` records the checkout base before the hardening commit; per-file runtime source hashes and image IDs identify the modified measured artifact. No final release SHA is falsely attributed to a pre-commit build.

| CPU | Clients | Before req/s | After req/s | Before p50/p95/p99 ms | After p50/p95/p99 ms | Before errors | After errors / attempts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1 | 280.39 | 273.47 | 1.984/6.337/7.842 | 2.081/6.339/7.821 | 0 | 0 / 2163 |
| 1.0 | 10 | 165.32 | 197.52 | 30.241/119.727/191.552 | 42.891/101.717/201.651 | 0 | 0 / 1196 |
| 1.0 | 50 | 158.76 | 149.31 | 71.298/1359.908/2353.91 | 244.524/363.386/1255.876 | {'TimeoutError': 1} | 0 / 1192 |
| 1.0 | 100 | 157.98 | 132.92 | 64.8/2115.541/3148.178 | 691.909/1003.366/1079.78 | {'TimeoutError': 38} | {'429': 53} / 914 |
| 0.25 | 1 | 69.51 | 38.32 | 2.44/76.652/84.65 | 4.085/89.289/94.791 | 0 | 0 / 282 |
| 0.25 | 10 | 26.9 | 39.05 | 290.548/1005.181/1388.943 | 205.081/504.495/687.066 | 0 | 0 / 243 |
| 0.25 | 50 | 15.71 | 13.19 | 897.627/3598.31/3796.707 | 2889.127/3999.04/4005.28 | {'TimeoutError': 45} | {'TimeoutError': 20} / 140 |
| 0.25 | 100 | 39.35 | 21.96 | 300.208/3397.282/3928.218 | 3093.374/3977.643/4004.278 | {'TimeoutError': 111} | {'TimeoutError': 61} / 281 |

At 1 CPU, 1/10/50 levels have no failures; at 100, **53/914 attempts receive controlled HTTP 429**, with no socket timeout, 500 or OOM. These refusals are not successes. Stored sessions accumulate across levels, and the new cap rejects new sessions instead of evicting existing users. At .25 CPU, **20 timeouts at 50 and 61 at 100 remain**; even the 10-client p95 is about 504 ms. That CPU class is insufficient for this stress profile. Bounded resources do not guarantee every overloaded connection receives a JSON refusal: transport queues/slow clients still need future proxy admission/deadline controls. Do not claim the cheapest .25 shared-CPU container passes the launch criterion.

The 1-CPU 10-client p95 is approximately 102 ms, meeting the proposed local <500 ms criterion for the mixed workload. The actual Lightsail burstable host, TLS/Cloudflare path and realistic API-heavy workload still require AWS production validation.

| CPU | Phase | Peak sampled RAM MiB | Peak sampled CPU % |
| --- | --- | --- | --- |
| 1.0 | 1 clients | 29.50 | 90.62 |
| 1.0 | 10 clients | 31.81 | 102.82 |
| 1.0 | 50 clients | 41.85 | 107.24 |
| 1.0 | 100 clients | 33.56 | 102.31 |
| 0.25 | 1 clients | 29.25 | 25.58 |
| 0.25 | 10 clients | 30.82 | 25.89 |
| 0.25 | 50 clients | 41.77 | 26.14 |
| 0.25 | 100 clients | 44.20 | 25.72 |

1-CPU sampled workload RAM peaks at 41.85 MiB versus 47.05 MiB before; quiet RAM is about 18.9 MiB versus 19.0 MiB before. The ten lift/60-second simulation cycles finish with about 32.04 MiB versus 32.03 MiB before the probe; this short probe is not a multi-hour memory guarantee. An adversarial regression also keeps the ball airborne for all 600 simulation steps and verifies default history/graph caps, supplementing 200 repeated action tests. CPU percentages use one-core normalization; short Docker samples can exceed quota percentages. Fixed worker PIDs remain bounded.

Final live-image lifecycle checks: `docker stop` **1.258 seconds**, direct SIGTERM **0.248 seconds**, both exit **0** without OOM or forced termination. An in-flight request also completes with HTTP 200 in the SIGTERM regression.

## Reproducible validation

```text
make runtime-bootstrap
make test-inventory
make runtime-test
make runtime-check
make runtime-build
make runtime-lifecycle
make runtime-browser
make runtime-benchmark
make runtime-benchmark-quarter
make runtime-local
make runtime-frontier
make runtime-proxy-browser
make runtime-status
make runtime-evidence
make runtime-lifecycle RUNTIME_IMAGE=wce-web:local
```

Browser prerequisites: previously installed Playwright/axe tools and Chromium, explicitly supplied through NODE_PATH and PLAYWRIGHT_BROWSERS_PATH. No browser dependency enters the image. Artifacts/logs stay local; evidence summaries are sanitized before versioning. runtime-local rebuilds/restarts only the existing local web service on its reserved loopback port. It is not an AWS deployment.

- **225 unit/frontend/runtime tests passed**, including 23 added runtime regressions. Generated test inventory checked against current source. The original 202 frontend/engine tests remain included.
- **408 FR/EN desktop/tablet/mobile/theme renders**, **160 WCAG AA scans with zero violations**, keyboard/reduced-motion/language persistence/no-JS/source/social-link checks and **112 existing browser regression paths** passed on the production server. The final numeric scheduler configuration fix was separately regressed; no UI source changed.
- **112 regression paths repeated against the rebuilt existing Docker service** on loopback 3018; laboratory actions/session continuity and FR/EN remained functional.
- Additional real-browser test uses an intercepted HTTPS origin with responses relayed to loopback production runtime: Secure/HttpOnly/Strict/expiry, session continuity FR/EN and JavaScript token invisibility pass. No certificate, proxy deployment, public DNS or Internet request is involved; real origin TLS is still a future gate.
- Docker build, dev/prod lifecycle, missing production config, idle CPU, image/credential hygiene and constrained 1/.25 CPU measurements completed. Lifecycle evidence references its exact image ID. Final local status verifies 70 endpoints (68 pages plus health/readiness), non-root, effective resource caps, read-only filesystem and source/inventory equality with workspace.
- `make runtime-check`: compilation, dependency consistency and diff whitespace checks pass. No full base-image CVE claim. All validation invoked through Makefile targets; transient helper/configuration failures were corrected and rerun before acceptance.

## Residual risks and next production validation

- Single process/host, ephemeral sessions, bounded capacity and command budgets; no HA or persistent lab state promised.
- An abusive client can consume global capacity. Proxy/edge network authentication, slow-client total deadlines, TLS, rate controls and genuine cloud load/soak tests are still required.
- Python worker threads are not a hard sandbox. Drain is bounded; pathological host I/O or a future non-cooperative command cannot be safely interrupted mid-operation. Current finite command vocabulary/budgets and CPU/RAM/PID isolation are tested; newly introduced commands require budget review.
- No kernel/base-image CVE scan, registry/promotion workflow, on-call notification, origin certificate, firewall or actual cloud quota validation in this task. The prior infrastructure permissions/quotas remain unresolved and must use a separate approved production identity.
- No cache benefit assumed. No DNS/mail changes, scientific sealed material or research AWS credentials introduced.

The four application blockers are evaluated by the test and measurement results below. Readiness for AWS production validation is not deployment approval, and does not execute any infrastructure gate.


## Files in this milestone

Runtime: `app/web.py`, `app/server.py`, `app/runtime_config.py`, `app/lab_runtime.py`, `app/localization.py`, `app/test_inventory.json`.

Packaging: `requirements-web.txt`, `Dockerfile`, `compose.yaml`, `.dockerignore`, `.gitignore`, `Makefile`.

Tests/tooling: `tests/test_web.py`, `tests/test_runtime.py`, `tests/runtime_signal_fixture.py`, `tests/browser/runtime.cjs`, `tools/production_benchmark.py`, `tools/runtime_build.py`, `tools/runtime_lifecycle.py`, `tools/runtime_status.py`, `tools/runtime_evidence_check.py`.

Documentation: this report, public scope in `docs/runtime_deployment_boundary.md`, sanitized JSON evidence in `docs/runtime_hardening_evidence/`. Source scientific core, templates, styles and AWS tooling unchanged during the original runtime-hardening milestone.

**RUNTIME HARDENING COMPLETE — READY FOR AWS PRODUCTION VALIDATION**
