# WCE Labs standalone outage experience

Prepared on 2026-10-13. This milestone creates local artifacts only. No AWS,
Cloudflare, DNS, mail, TLS material or production routing changes are involved.

## Purpose and design

The outage experience remains usable when the origin, reverse proxy, application,
APIs and application assets are unavailable. Each document contains its own CSS
and the official `app/static/brand/wce-labs-logo-dark.svg` inline, unchanged apart
from accessible SVG attributes. The existing dark palette (#161817, #efeee8,
#b0b3ad), Georgia editorial heading, Arial body, monospace signature, restrained
rules and spacing preserve the WCE Labs identity. There are no illustrations,
decorative animations, infrastructure details or claims about the outage cause.

## Artifacts and language

- `ops/outage/fr.html`: 2,495 UTF-8 bytes; primary French experience.
- `ops/outage/en.html`: 2,443 UTF-8 bytes; equivalent English experience.
- `ops/outage/template.html` and `tools/outage.py`: deterministic build sources.

Both files work independently with JavaScript disabled and system fonts. There
are zero external runtime dependencies or tracking requests. A data favicon
prevents an implicit application favicon fetch. No script, API, font, image or
stylesheet is fetched. SVG namespace metadata is not a network request.

Language is explicit per artifact: `lang=fr` or `lang=en`, with corresponding
document title and retry label. No client detection or application language state
is needed. Gate 2 may select the English artifact from a documented request
language policy, with French fallback; the delivery policy remains undecided.
Each artifact remains complete even if only one language can be configured.

French copy:

> WCE Labs
>
> Service temporairement indisponible
>
> Notre plateforme est momentanément indisponible.
> Nos systèmes sont en cours de rétablissement.
>
> Merci de réessayer dans quelques instants.
>
> Réessayer
>
> WCE Labs — World Configuration Engine

English copy:

> WCE Labs
>
> Service temporarily unavailable
>
> Our platform is temporarily unavailable.
> Our systems are being restored.
>
> Please try again shortly.
>
> Try again
>
> WCE Labs — World Configuration Engine

The retry link uses an empty URL reference: it reloads the current document,
preserving its path and query, with or without JavaScript. No automatic refresh
or retry loop adds load to an unavailable service.

## Local preview and verification

Run from the repository:

```sh
make outage-build
make outage-test
make outage-browser-bootstrap
make outage-browser
make frontend-test
make outage-preview
```

The preview binds only to `127.0.0.1`, uses an automatically allocated free port
and prints its URL. `/` and `/fr/` show French; `/en/` shows English. It is a
separate standard-library server, not a route in the WCE application. Stop with
Ctrl+C. An explicit available port can be passed as `OUTAGE_PORT`; check local
port governance before choosing one. No stable port reservation is required.

The preview returns **503** and `Cache-Control: no-store`; unknown paths return
404. It is a development tool, not a production HTTP server. Browser tools are
pinned and installed under `/tmp`, outside the production image. Browser results
and screenshots are written to `/tmp/wce-outage-review/`.

Validation covers six artifact contract tests plus the 225 existing frontend and
runtime tests. Browser checks cover 20 renders: French/English, JavaScript
enabled/disabled, 320×568, 390×844, 768×1024, 1440×900 and 2560×1440.
Ten WCAG A/AA scans cover the JavaScript-enabled equivalent DOM; keyboard retry,
visible focus, request isolation and overflow checks cover both modes. Short
narrow screens may scroll vertically to preserve readable content; normal
screens should not. No animation is present, including under reduced motion.
Automated accessibility scans supplement visual review; they do not constitute
an accessibility certification or assistive-technology user study.

The repository's generated test inventory is refreshed to include the six new
tests; normal application templates, styles, APIs and routing are unchanged.

### Executed results

- `make outage-build outage-test`: both artifacts reproduced; six tests passed.
- `make frontend-test`: 225 tests passed. Expected negative-path log messages
  were emitted by existing tests; they were not test failures.
- `make outage-browser`: 20 responsive renders, ten WCAG AA scans with zero
  detected violations, zero external requests, keyboard retry successful with
  JavaScript enabled and disabled. Both committed files also rendered directly
  using `file:` URLs without an HTTP or application server.
- Screenshots were visually inspected at mobile, narrow mobile, tablet, desktop
  and large desktop sizes, including both languages. No clipped text or
  horizontal overflow was found. The 320×568 layout scrolls vertically as intended
  to retain legibility; all larger tested viewports fit without scrolling.
- `git diff --check` passed. Human visual approval remains pending; no edge
  delivery, subscription capability or real Cloudflare failure trigger was tested.

## Future Cloudflare boundary

No Cloudflare product, subscription capability or integration is assumed.
Gate 2 must verify the selected mechanism accepts inline SVG/CSS and complete
HTML of this size, and select the appropriate language document. Serve an
appropriate failure response, such as **HTTP 503** for service unavailability,
never a successful 200 merely because this document rendered. Preserve meaningful
platform failure statuses where applicable. Do not invent a recovery time in a
Retry-After header. Prevent indexing and unintended caching of the failure page;
set suitable response headers in the delivery mechanism. Any future CSP must
allow the artifact's inline style and data favicon, without allowing scripts.

The artifact's content carries no credentials, build paths, source maps,
repository information, internal identifiers or origin infrastructure metadata.
This milestone stops at human visual review. Public edge delivery, outage
triggering/fallback semantics and real edge response verification belong to
Gate 2 and require separate authorization.
