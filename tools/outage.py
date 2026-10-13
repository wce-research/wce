"""Build and preview portable outage documents; never import the application."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = {
    "fr": ("Service temporairement indisponible", "Notre plateforme est momentanément indisponible.", "Nos systèmes sont en cours de rétablissement.", "Merci de réessayer dans quelques instants.", "Réessayer"),
    "en": ("Service temporarily unavailable", "Our platform is temporarily unavailable.", "Our systems are being restored.", "Please try again shortly.", "Try again"),
}


def render(language):
    title, first, second, patience, retry = COPY[language]
    logo = (ROOT / "app/static/brand/wce-labs-logo-dark.svg").read_text().strip()
    logo = logo.replace('<svg ', '<svg role="img" aria-label="WCE Labs" ')
    template = (ROOT / "ops/outage/template.html").read_text()
    for key, value in dict(language=language, title=title, first=first, second=second,
                           patience=patience, retry=retry, logo=logo).items():
        template = template.replace("{{" + key + "}}", value)
    return template.encode("utf-8")


class Preview(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        language = {"/": "fr", "/fr/": "fr", "/en/": "en"}.get(route)
        body = render(language) if language else b"Not found"
        self.send_response(503 if language else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "preview"])
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if args.action == "build":
        for language in COPY:
            body = render(language)
            (ROOT / f"ops/outage/{language}.html").write_bytes(body)
            print(f"{language}: {len(body)} bytes")
    else:
        with HTTPServer(("127.0.0.1", args.port), Preview) as server:
            print(f"Outage preview: http://127.0.0.1:{server.server_port}/", flush=True)
            server.serve_forever()


if __name__ == "__main__":
    main()
