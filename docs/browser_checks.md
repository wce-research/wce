# WCE browser checks

Run `make frontend-test` in the canonical workspace. It verifies HTTP routes, metadata, branding, sessions, input validation, safe API 500 responses, public-page privacy, the static inventory and the symbolic engine. It does not reproduce the whole scientific research suite.

Browser checks run locally. Playwright and axe are test dependencies outside the repository and production image. The scientific audit was verified with Node.js 20, Playwright/Playwright Core 1.64.0, axe-core and @axe-core/playwright 4.13.0, and bundled Chromium 156.

## Complete local scientific and accessibility audit

Install the pinned test tools in a dedicated temporary directory; they are never copied into the web image:

```sh
npm install --prefix /tmp/wce-frontend-audit-tools playwright@1.64.0 @axe-core/playwright@4.13.0
PLAYWRIGHT_BROWSERS_PATH=/tmp/wce-frontend-audit-tools/browsers \
  /tmp/wce-frontend-audit-tools/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/wce-frontend-audit-tools/node_modules \
PLAYWRIGHT_BROWSERS_PATH=/tmp/wce-frontend-audit-tools/browsers \
WCE_BROWSER_ARTIFACTS=/tmp/wce-frontend-audit \
  make frontend-science-browser
```

This target starts a loopback-only server on an ephemeral port through `make frontend-serve`, audits the sitemap's complete bilingual route catalog at 1440, 768 and 390 pixels in both themes, and stops the entire temporary server process group. It does not start Docker services, use SSH, create a tunnel or call infrastructure APIs. It runs WCAG AA scans in both themes on every desktop route and on the new mobile evidence views, checks internal links, all four official secure social links, diagrams, keyboard transactions, language detection/persistence, reduced motion and no-JavaScript reading. It also opens all fourteen milestones and their nested report evidence, checks expanded content at every viewport/theme, and runs additional WCAG AA scans on the expanded desktop/mobile programme, and then runs the established browser regression suite against the same server. Screenshots, local timing samples and the complete accessibility record are written to `audit.json` in the artifact directory. A host `ERR_NETWORK_CHANGED` is retried at most twice; other errors fail immediately.

`WCE_AUDIT_SCOPE=science` limits the page matrix to the four research views and the homepage for focused debugging; it is not the complete acceptance run. Automated axe scans are not a guarantee of accessibility for every assistive technology. Local timings are not production Core Web Vitals.

## Existing browser regression target

With Playwright Core installed in an external tools directory and a local Chromium browser available:

```sh
NODE_PATH=<external-tools>/node_modules \
WCE_CHROMIUM=<chromium-executable> \
WCE_BASE_URL=http://127.0.0.1:3018 \
WCE_BROWSER_ARTIFACTS=<temporary-artifacts-directory> \
make frontend-browser
```

The script checks French/English pages at desktop, tablet and mobile widths, day/night themes, geometry controls, nine metrics, hypothesis verdicts, source search, navigation, footer and laboratory responses. The optional SSH adapter is retained as a test facility, not a source synchronization mechanism. Frontend changes and scientific updates are synchronized through GitHub.
