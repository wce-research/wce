# WCE Labs — public identity and operations

Update of 10 October 2026. The existing art direction and C2 symbol are kept.

## Audit and scope

The frontend is rendered server-side in Python (app/site.py, templates and CSS/JS), with the Session lab and the API in the same web service. No additional frontend server, database or worker is needed for this update. The accounts of the future hub remain an ambition and their private architecture is not published.

The mentions of the WCE project and the scientific documents remain WCE / World Configuration Engine. The identity of the laboratory in the header, footer, titles, metadata and new exports becomes WCE Labs. The historical logo research boards are kept as archives, not used as the main identity.

Audit: shared shell, FR/EN routes, social links and .env.example configuration, exports/favicon, manifest, metadata and research resources. There was no centralised canonical/OG/JSON-LD; app/public_identity.py now provides them. No personal email added. No old domain introduced in the new exports.

## Configuration

- Domain and organisation centralised in app/public_identity.py: https://wcelabs.com and https://github.com/wce-research.
- PUBLIC_CONTACT_EMAIL: empty by default; only a valid address under wcelabs.com is displayed. Provide a real mailbox before activation.
- PUBLIC_WCE_REPO_ENABLED=false: the link https://github.com/wce-research/wce remains hidden by default. The repository is now public with an Apache-2.0 licence, according to the state communicated by the owner; the local scientific code has not yet been pushed. Its existence does not amount to publication of the code. Any scientific publication requires a prior audit.
- WCE_GITHUB_URL is kept for Compose compatibility; the frontend uses the centralised official organisation. X, LinkedIn and Instagram keep their existing configuration.

## Assets and SEO

The future canonical language of the public scientific repository is English: code, comments and docstrings, README, technical documentation, reports, hypotheses, artefacts and public metadata. The site remains multilingual and will be able to translate this canonical source. This rule does not mean that the existing local documents have already been translated or published.

Sources: tools/build_labs_brand.py, SVG files in app/static/brand. Exports: square PNG/SVG avatars light and dark, monochrome SVG, horizontal WCE Labs logos light/dark/monochrome, social image 1200 × 630. Adaptive SVG favicon, ICO 16/32/48 and apple-touch-icon 180. The original geometry is unchanged. Regeneration: python3 tools/build_labs_brand.py (Pillow and the DejaVu Sans system font required; no tool added to the production image).

Each FR/EN page receives a WCE Labs title, a cautious description, a localised canonical, hreflang, OpenGraph, Twitter summary_large_image and Organization JSON-LD. The images use the official domain. The manifest remains display=browser: no PWA capability promised.

## Machines and ports

Development: GEEKOM A6. Execution: Dell dell-host, ~/workspace/wce. Existing SSH <user>@<dell-host> with <ssh-key>; no change of key/configuration.

The authority on the GEEKOM is ~/workspace/_machine_registry (effective equivalent of the requested folder). ports.json projects.wce and the README table already reserve web: 3018, binding 127.0.0.1. ./ports check validates the registry. No new reservation needed. The Dell copy ~/workspace/portfolio-port-governance is historical: do not replace the recent authority with this copy.

Compose publishes only 127.0.0.1:3018 → web:8000 for the browser and the local API. The Dell runs Linux; no active Windows/WSL layer to be assumed. ss and docker compose ps confirmed that this port already belongs to WCE, without collision.

## Commands on the Dell

```bash
cd ~/workspace/wce
docker compose up -d --build --wait
docker compose ps
docker compose logs --tail=100 web
docker compose logs -f web
docker compose down
docker compose up -d --wait
```

Local URL: http://localhost:3018/fr/ or http://localhost:3018/en/.
Check: curl -fsS http://127.0.0.1:3018/healthz.
No LAN or Internet exposure added. The lab sessions are temporary; no persistent storage of accounts is announced.

## Future connection of wcelabs.com — validation required

The Cloudflare DNS is not modified. A DNS entry does not make the private port 127.0.0.1:3018 reachable from the Internet. Documentation: https://developers.cloudflare.com/dns/proxy-status/ and https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/.

After explicit authorisation:

1. Audit the current ingress of the Dell (reverse proxy, existing connectors, TLS, Cloudflare rules), choose to reuse it before any creation.
2. Validate an HTTPS route for wcelabs.com to WCE. If an existing local connector is retained, origin http://127.0.0.1:3018; if a Docker connector, origin http://web:8000 on an explicitly controlled shared network. No new arbitrary port.
3. Check Host/Origin, public TLS, cookie policy and protection of the interactive API behind the ingress; keep the current protections.
4. Validate the DNS/route change together then execute it. Do not create a tunnel, router redirection or secret as long as this step is not authorised.
5. Verify HTTPS, FR/EN, canonical, OG assets, API and logs from an external access. Declare the domain operational only after these checks.

Still to be provided: official contact address, publication of the scientific repository and choice/validation of the ingress. No partnership or cloud credit displayed. The frontend and the architecture of the hub must not be included in the publication of the scientific repository.

## Verification of this delivery

- GitHub SSH authentication: existing project account; Dell connection confirmed. No push performed and administration rights on the wce-research organisation not presumed.
- 184 relevant tests pass on the current frontend (web, engine, publication boundary, identity). This check is not a certification of all the scientific research.
- 166 tests pass on an isolated copy of the tree prepared for the identity commit (without the old uncommitted modifications).
- Dell: 158 identity/engine tests, Docker build, healthcheck, FR/EN pages, Open Research, manifest, favicon and images HTTP 200. Compose stop/restart validated; user wce; port 3018 restricted to 127.0.0.1.
- Browser connected to the Dell: FR/EN footer on desktop/mobile, five sections, localised titles; no JavaScript error or overflow. Temporary check screenshots in /tmp/wce-labs-browser on the GEEKOM.
- Review of the dedicated diff and check of secret patterns carried out. No .env, private key, credential or scientific file in the commit.
- Compose reported the absence of buildx for Bake; the build nevertheless succeeded with the existing Docker driver. No system installation needed for this delivery.
