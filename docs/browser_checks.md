# WCE browser checks

Run `make frontend-test` on the Dell canonical GitHub clone. It verifies HTTP routes, metadata, branding, sessions, input validation, safe API 500 responses, public-page privacy, the static inventory and the symbolic engine. It does not reproduce the whole scientific research suite.

Browser checks run directly on the Dell, against the local Docker service. Playwright is a test dependency outside the repository and production image; Node.js 20 or later is required for Playwright Core 1.63.0.

With Playwright Core installed in an external tools directory and a local Chromium browser available:

```sh
NODE_PATH=<external-tools>/node_modules \
WCE_CHROMIUM=<chromium-executable> \
WCE_BASE_URL=http://127.0.0.1:3018 \
WCE_BROWSER_ARTIFACTS=<temporary-artifacts-directory> \
make frontend-browser
```

The script checks French/English pages at desktop, tablet and mobile widths, day/night themes, geometry controls, nine metrics, hypothesis verdicts, source search, navigation, footer and laboratory responses. The optional SSH adapter is retained as a test facility, not a source synchronization mechanism. Frontend changes and scientific updates are synchronized through GitHub.
