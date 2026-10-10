"""Small same-origin web interface for the existing WCE Session."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from app.site import PAGES, STATIC, render_page
from app.i18n import resolve_route
from app.localization import canonical_command, answer_in_language, error_in_language
from app.session import Session
from language.command import AdvanceTimeCommand, parse_command


ASSETS = {
    "/site.webmanifest": ("site.webmanifest", "application/manifest+json"),
    "/theme.js": ("theme.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/favicon.ico": ("favicon.ico", "image/vnd.microsoft.icon"),
    "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png"),
    "/wce-logo.svg": ("wce-logo.svg", "image/svg+xml"),
    "/api.js": ("api.js", "text/javascript; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/site.js": ("site.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
# Fixed brand allowlist: no arbitrary filesystem serving.
for asset in ('wce-labs-avatar-light.svg','wce-labs-avatar-dark.svg','wce-labs-avatar-mono.svg','wce-labs-avatar-light.png','wce-labs-avatar-dark.png','wce-labs-logo-light.svg','wce-labs-logo-dark.svg','wce-labs-logo-mono.svg','wce-labs-og.png'):
    ASSETS['/brand/'+asset] = ('brand/'+asset,'image/svg+xml' if asset.endswith('.svg') else 'image/png')
ROOT = STATIC.parents[1]
RESEARCH = {
    "/research/predictive_coverage.json": (ROOT / "results/predictive_coverage.json", "application/json; charset=utf-8"),
    "/research/continuum_summary.json": (ROOT / "results/continuum_summary.json", "application/json; charset=utf-8"),
    "/research/active_summary.json": (ROOT / "results/active_summary.json", "application/json; charset=utf-8"),
    "/research/hypotheses_generative_certificates.md": (ROOT / "docs/hypotheses_generative_certificates.md", "text/markdown; charset=utf-8"),
    "/research/architecture_wce.md": (ROOT / "docs/architecture_wce.md", "text/markdown; charset=utf-8"),
}
IDENTITY_ROOT = ROOT / "docs/identity"
IDENTITY_TYPES = {
    ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/vnd.microsoft.icon",
    ".zip": "application/zip", ".md": "text/markdown; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}
IDENTITY_FILES = {
    "/identity/" + str(file.relative_to(IDENTITY_ROOT)): (file, IDENTITY_TYPES[file.suffix])
    for file in IDENTITY_ROOT.rglob("*")
    if file.is_file() and file.suffix in IDENTITY_TYPES
}
IDENTITY_FILES["/identity/assets/typeface-license.txt"] = (
    IDENTITY_ROOT / "assets/typeface-license.txt", "text/plain; charset=utf-8"
)

MAX_BODY = 4096
MAX_COMMAND = 500
SESSION_TTL = 24 * 60 * 60
MAX_SESSIONS = 100
_sessions: dict[str, tuple[Session, float]] = {}
_lock = threading.RLock()


def _session(token: str | None) -> tuple[str, Session, bool]:
    now = time.monotonic()
    with _lock:
        if token and token in _sessions:
            session, touched = _sessions[token]
            if now - touched < SESSION_TTL:
                _sessions[token] = (session, now)
                return token, session, False
        expired = [key for key, (_, touched) in _sessions.items() if now - touched >= SESSION_TTL]
        for key in expired:
            del _sessions[key]
        if len(_sessions) >= MAX_SESSIONS:
            oldest = min(_sessions, key=lambda key: _sessions[key][1])
            del _sessions[oldest]
        token = secrets.token_urlsafe(32)
        session = Session()
        _sessions[token] = (session, now)
        return token, session, True


class Handler(BaseHTTPRequestHandler):
    server_version = "WCE/1.0"

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, cookie: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'")
        if cookie:
            self.send_header("Set-Cookie", f"wce_session={cookie}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict, cookie: str | None = None) -> None:
        if 'error' in payload:
            payload = {**payload, 'error': error_in_language(payload['error'],self.headers.get('X-WCE-Language','fr'))}
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8", cookie)

    def _token(self) -> str | None:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return None
        return cookie["wce_session"].value if "wce_session" in cookie else None

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        route, language = resolve_route(path)
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok"})
        elif route in PAGES:
            preference = SimpleCookie()
            try:
                preference.load(self.headers.get("Cookie", ""))
            except Exception:
                preference = SimpleCookie()
            if path == "/" and "wce_language" in preference and preference["wce_language"].value == "en":
                self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
                self.send_header("Location", "/en/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(HTTPStatus.OK, render_page(route,language), "text/html; charset=utf-8")
        elif path in ASSETS:
            filename, content_type = ASSETS[path]
            self._send(HTTPStatus.OK, (STATIC / filename).read_bytes(), content_type)
        elif path == "/identity":
            self.send_response(HTTPStatus.PERMANENT_REDIRECT)
            self.send_header("Location", "/identity/")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path in ("/identity/", "/identity/index.html"):
            self._send(HTTPStatus.OK, (IDENTITY_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path in IDENTITY_FILES:
            file, content_type = IDENTITY_FILES[path]
            self._send(HTTPStatus.OK, file.read_bytes(), content_type)
        elif path in RESEARCH:
            file, content_type = RESEARCH[path]
            self._send(HTTPStatus.OK, file.read_bytes(), content_type)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Route inconnue."})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/command":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Route inconnue."})
            return
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and urlsplit(origin).netloc != host:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Origine refusée."})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON requis."})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Requête trop longue ou vide."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            command = payload["command"]
            if not isinstance(command, str) or not 0 < len(command.strip()) <= MAX_COMMAND:
                raise ValueError("La commande doit contenir entre 1 et 500 caractères.")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc) or "Requête invalide."})
            return
        language = "en" if self.headers.get("X-WCE-Language") == "en" else "fr"
        command = canonical_command(command.strip(), language)
        try:
            parsed = parse_command(command)
        except ValueError as exc:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
            return
        if isinstance(parsed, AdvanceTimeCommand) and parsed.duration > 60:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "La durée maximale est de 60 secondes."})
            return
        token, session, created = _session(self._token())
        try:
            with _lock:
                answer = session.execute(command.strip())
        except ValueError as exc:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)}, token if created else None)
            return
        except Exception:
            logging.exception("Command execution failed")
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Erreur interne."}, token if created else None)
            return
        self._json(HTTPStatus.OK, {"answer": answer_in_language(answer,language)}, token if created else None)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("WCE_HOST", "127.0.0.1")
    port = int(os.environ.get("WCE_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    logging.info("WCE listening on %s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
