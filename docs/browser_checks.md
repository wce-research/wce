# WCE browser checks

The targeted Python tests are run with `python3 -m unittest tests.test_web tests.test_wce`. In particular they verify the displayed values and counts against the served JSON files, as well as the update of the render when the values of an artefact change.

The script `tests/browser/frontier.cjs` uses Playwright Core 1.63.0 and Chromium. It is not part of the Docker image. Example run from the GEEKOM with the existing SSH connection, without a new port or TCP forwarding:

```bash
npm install --prefix /tmp/wce-browser --no-audit --no-fund playwright-core@1.63.0
NODE_PATH=/tmp/wce-browser/node_modules \
WCE_BASE_URL=http://127.0.0.1:3018 \
WCE_SSH_TARGET=<user>@<dell-host> \
WCE_SSH_IDENTITY=~/.ssh/<ssh-key> \
WCE_CHROMIUM=~/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome \
WCE_BROWSER_ARTIFACTS=/tmp/wce-frontier-dell \
node tests/browser/frontier.cjs
```

The Chromium path corresponds to the installation present on the GEEKOM. For a test directly on the Dell, omit the `WCE_SSH_*` variables and use a Chromium installed on that machine. The SSH relay only queries the existing HTTP server of the Dell; it starts no server or stack on the GEEKOM and does not modify its SSH configuration.

In the agent's isolated execution environment, `WCE_BROWSER_NO_SANDBOX=1` was needed for the test Chromium. This option is reserved for this verification process; it does not modify any protection of the web container.

Checks: 36 bilingual renders at widths 1440, 768 and 390 px; transformations and frame of the illustration; nine metrics and nine methods; tables and intervals; ten hypotheses; five source groups; day/night modes; ten diagrams; absence of horizontal overflow and of JavaScript errors.
