import unittest
import hashlib
import tempfile
from tools.prepare_public_source import build
from tools.prepare_public_source import ROOT, selected_files
from app.site import PAGES, render_page


class PublicBoundaryTests(unittest.TestCase):
    def test_scientific_export_excludes_platform(self):
        paths = {p.relative_to(ROOT).as_posix() for p in selected_files()}
        self.assertIn('app/session.py', paths)
        self.assertIn('core/world.py', paths)
        self.assertEqual({p for p in paths if p.startswith('app/')}, {'app/__init__.py', 'app/session.py', 'app/repl.py'})
        self.assertFalse(any(p.startswith(('wce_aws/', 'app/static/', 'app/templates/')) for p in paths))
        self.assertNotIn('docs/hub_evolution.md', paths)
        self.assertNotIn('tests/test_web.py', paths)

    def test_public_pages_exclude_private_deployment(self):
        for language in ('fr', 'en'):
            for route in PAGES:
                with self.subTest(route=route, language=language):
                    page = render_page(route, language=language).decode()
                    for private in ('app/web.py', 'app/localization.py', 'compose.yaml', '127.0.0.1:3018', '192.168.0.12'):
                        self.assertNotIn(private, page)

    def test_export_is_deterministic_and_manifest_matches(self):
        import json
        import tarfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'core').mkdir()
            (root / 'core/world.py').write_text('class World: pass\n')
            archive = build(root)
            first = hashlib.sha256(archive.read_bytes()).hexdigest()
            build(root)
            self.assertEqual(first, hashlib.sha256(archive.read_bytes()).hexdigest())
            with tarfile.open(archive) as tar:
                manifest = json.load(tar.extractfile('wce/MANIFEST.json'))
                for name, digest in manifest['files'].items():
                    self.assertEqual(digest, hashlib.sha256(tar.extractfile('wce/'+name).read()).hexdigest())

    def test_reading_aids_are_bilingual_and_scoped(self):
        from app.understanding import QUESTIONS
        for language in ('fr', 'en'):
            home = render_page('/', language).decode()
            self.assertIn('understanding-grid', home)
            self.assertIn('World Configuration Engine', home)
            architecture = render_page('/architecture', language).decode()
            self.assertEqual(architecture.count('<details>'), 6)
            self.assertIn('vocabulary-grid', architecture)
            for route in QUESTIONS:
                page = render_page(route, language).decode()
                self.assertIn('research-question', page)
                self.assertIn('breadcrumbs', page)
                self.assertIn(QUESTIONS[route][language == 'en'][0], page)

    def test_mechanism_proof_boundaries_and_links(self):
        from app.research_catalog import FAMILIES
        from app.understanding import EVIDENCE_ROUTES
        for lang in ('fr', 'en'):
            for family in FAMILIES:
                page = render_page(family['route'], lang).decode()
                self.assertIn('mechanism-scope', page)
                self.assertEqual(page.count('<div class="scope-grid">'), 1)
                if family['key'] in EVIDENCE_ROUTES:
                    self.assertIn('/'+lang+EVIDENCE_ROUTES[family['key']], page)
                else:
                    self.assertIn('Aucun dossier de résultats dédié' if lang == 'fr' else 'No dedicated results dossier', page)
                self.assertNotIn('tests/test_web.py', page)

    def test_architecture_search_preserves_full_map(self):
        for lang in ('fr', 'en'):
            page=render_page('/architecture', lang).decode()
            self.assertIn('class="atlas-search" hidden', page)
            self.assertEqual(page.count('data-mechanism='), 10)
            self.assertIn('aria-controls="mechanism-results"', page)
            self.assertNotIn('mechanism-search', render_page('/', lang).decode())

    def test_related_mechanisms_have_localised_destinations(self):
        from app.content import RELATED_MECHANISMS
        for lang in ('fr', 'en'):
            for route, related in RELATED_MECHANISMS.items():
                page=render_page(route, lang).decode()
                for destination in related:
                    self.assertIn(f'href="/{lang}{destination}"', page)

    def test_metrics_have_protocol_specific_guidance_and_hashes(self):
        from app.evidence import METRICS, METRIC_GUIDES, artifact
        self.assertEqual(set(METRICS), set(METRIC_GUIDES))
        for lang in ('fr', 'en'):
            page=render_page('/research/continual', lang).decode()
            self.assertEqual(page.count('class="metric-guide"'), 9)
            self.assertIn('retention', page)
            self.assertIn('average_accuracy', page)
            for route, name in (('/research/measurement','predictive_coverage.json'),('/research/continual','continuum_summary.json'),('/research/discovery','active_summary.json')):
                self.assertIn(artifact(name)[1], render_page(route, lang).decode())

    def test_discovery_questions_preserve_verdicts(self):
        from app.evidence import HYPOTHESIS_GUIDES, artifact
        data, _ = artifact('active_summary.json')
        self.assertEqual(set(data['hypotheses']), set(HYPOTHESIS_GUIDES))
        for lang in ('fr','en'):
            page=render_page('/research/discovery',lang).decode()
            self.assertEqual(page.count('class="hypothesis-title"'), 10)
            for record in data['hypotheses'].values():
                self.assertIn(record['verdict'], page)
            self.assertIn('/research/hypotheses_active_structure.md',page)
            self.assertIn('class="verdict-filters" hidden',page)

    def test_open_research_state_and_search_are_public_metadata(self):
        for lang in ('fr','en'):
            source=render_page('/open/source',lang).decode()
            self.assertIn('class="release-grid"', source)
            self.assertIn('class="inventory-search" hidden', source)
            self.assertIn('data-source-module', source)
            self.assertIn('aria-controls="source-results"', source)
            brief=render_page('/open/contribute',lang).decode()
            self.assertIn('contribution-brief', brief)
            self.assertIn('Baseline', brief)
            self.assertNotIn('<form', brief)

    def test_laboratory_has_accessible_command_feedback(self):
        for lang in ('fr','en'):
            page=render_page('/lab',lang).decode()
            self.assertIn('id="command-count"',page)
            self.assertIn('id="lab-status" role="status"',page)
            self.assertIn('aria-describedby="command-help lab-status"',page)
            self.assertIn('/'+lang+'/lab/guide',page)
            self.assertIn('Run' if lang=='en' else 'Exécuter',page)

    def test_every_public_page_has_valid_internal_links_and_unique_ids(self):
        from html.parser import HTMLParser
        from app.site import PAGES
        from app.web import RESEARCH, ASSETS
        class PageParser(HTMLParser):
            def __init__(self):
                super().__init__();self.ids=[];self.links=[];self.headings=0
            def handle_starttag(self,tag,attributes):
                attrs=dict(attributes)
                if 'id' in attrs:self.ids.append(attrs['id'])
                if tag=='a' and 'href' in attrs:self.links.append(attrs['href'])
                if tag=='h1':self.headings+=1
        for lang in ('fr','en'):
            for route in PAGES:
                with self.subTest(lang=lang,route=route):
                    parser=PageParser();parser.feed(render_page(route,lang).decode())
                    self.assertEqual(parser.headings,1)
                    self.assertEqual(len(parser.ids),len(set(parser.ids)))
                    for target in parser.links:
                        if target.startswith('#'):
                            self.assertIn(target[1:],parser.ids)
                        elif target.startswith('/'):
                            prefix=next((p for p in ('/fr','/en') if target.startswith(p+'/')),None)
                            if prefix:
                                self.assertIn(target[len(prefix):].rstrip('/') or '/',PAGES)
                            else:
                                self.assertTrue(target in RESEARCH or target in ASSETS,target)

    def test_footer_directory_and_current_section_are_localised(self):
        for lang in ('fr','en'):
            page=render_page('/architecture/planning',lang).decode()
            self.assertIn('class="footer-directory"',page)
            self.assertIn('data-current-section="true"',page)
            self.assertNotIn(f'href="/{lang}/architecture" aria-current="page"',page)
            for target in ('/open/reproduce','/research/limitations','/vision/now'):
                self.assertIn(f'href="/{lang}{target}"',page)
