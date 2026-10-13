"""Standalone edge artifact contract, independent of application imports."""
import unittest
from html.parser import HTMLParser
from tools.outage import COPY, ROOT, render


class Tags(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.tags = []
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class OutageTests(unittest.TestCase):
    def test_committed_artifacts_are_reproducible(self):
        for language in COPY:
            self.assertEqual(render(language), (ROOT / f"ops/outage/{language}.html").read_bytes())

    def test_content_and_accessibility(self):
        for language, copy in COPY.items():
            body = render(language).decode()
            tags = Tags(body).tags
            self.assertIn(("html", {"lang": language}), tags)
            for text in copy:
                self.assertIn(text, body)
            for tag in ("main", "h1", "header", "footer"):
                self.assertEqual(sum(name == tag for name, _ in tags), 1)
            self.assertIn('name="viewport"', body)
            self.assertIn(':focus-visible', body)
            self.assertIn('min-height:48px', body)

    def test_official_logo(self):
        logo = (ROOT / "app/static/brand/wce-labs-logo-dark.svg").read_text().strip()
        for language in COPY:
            self.assertIn(logo.replace('<svg ', '<svg role="img" aria-label="WCE Labs" '), render(language).decode())

    def test_no_runtime_or_network_dependencies(self):
        for language in COPY:
            for tag, attrs in Tags(render(language).decode()).tags:
                self.assertNotIn(tag, ("script", "iframe", "object", "embed"))
                for key in ("href", "src", "action"):
                    if key in attrs:
                        self.assertIn(attrs[key], ("", "data:,"))
            self.assertNotIn("url(", render(language).decode())

    def test_retry_is_current_document(self):
        for language in COPY:
            links = [attrs for tag, attrs in Tags(render(language).decode()).tags if tag == "a"]
            self.assertEqual(links, [{"class": "retry", "href": ""}])

    def test_compact_and_no_internal_information(self):
        for language in COPY:
            body = render(language)
            self.assertLess(len(body), 8192)
            for forbidden in ("aws", "lightsail", "nginx", "docker", "cloudflare", "stack trace", "github", "/home/", "sourceMappingURL", "analytics", "8000", "18080"):
                self.assertNotIn(forbidden.lower(), body.decode().lower())


if __name__ == "__main__":
    unittest.main()
