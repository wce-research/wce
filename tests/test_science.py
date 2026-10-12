"""Public evidence provenance, publication scope, language choice and source packaging."""
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

from app.i18n import preferred_language
from app.science import summary
from app.science_catalog import MILESTONES, PUBLIC_REPORTS, PUBLIC_SUMMARIES, PUBLIC_METHODS, SCIENCE_ROUTES, LATEST_VERDICT
from app.site import PAGES, render_page
from tools.build_test_inventory import inventory

ROOT = Path(__file__).resolve().parents[1]


class EvidenceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'data-science-artifact' in attrs:
            self.cells.append(attrs)
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])
        if 'id' in attrs:
            self.ids.append(attrs['id'])


class ScienceTests(unittest.TestCase):
    def test_all_displayed_scientific_values_match_exact_artifact_paths(self):
        for language in ('en','fr'):
            for route in ('/research/deep-composition','/research/high-precision'):
                parser=EvidenceParser();parser.feed(render_page(route,language).decode())
                self.assertGreater(len(parser.cells),20)
                for attrs in parser.cells:
                    value,_=summary(attrs['data-science-artifact'])
                    for key in json.loads(attrs['data-science-path']):
                        value=value[key]
                    self.assertEqual(value,json.loads(attrs['data-science-value']))

    def test_allowlisted_sources_exist_and_are_packaged(self):
        from app.web import RESEARCH
        docker=(ROOT/'Dockerfile').read_text(); ignore=(ROOT/'.dockerignore').read_text()
        for path in PUBLIC_REPORTS+PUBLIC_SUMMARIES+PUBLIC_METHODS:
            self.assertTrue((ROOT/path).is_file(),path)
            self.assertIn(path,docker)
            self.assertIn('!'+path,ignore)
            self.assertEqual(RESEARCH['/research/'+Path(path).name][0],ROOT/path)
        with self.assertRaises(ValueError):
            summary('../docs/sealed_deep_manifest.json')
        for path in RESEARCH.values():
            self.assertNotIn('sealed_deep_manifest',path[0].name)
            self.assertNotEqual(path[0].suffix,'.salt')

    def test_milestone_verdicts_and_hp_hypotheses_are_not_reinterpreted(self):
        report=(ROOT/'docs/rapport_high_precision_primitives.md').read_text()
        self.assertIn(LATEST_VERDICT,report.split('## 11. Verdict')[1])
        for language in ('en','fr'):
            page=render_page('/research/programme',language).decode()
            for milestone in MILESTONES:
                self.assertIn('data-milestone="'+milestone[0]+'"',page)
                self.assertIn(milestone[2],page)
            hp=render_page('/research/high-precision',language).decode()
            for hypothesis in range(1,14):
                self.assertIn(f'<th scope="row">HP{hypothesis}</th>',hp)
            self.assertIn('POST-HOC / EXPLORATORY',hp)
            self.assertIn('lang="en"',hp)

    def test_public_pages_link_to_real_canonical_files_and_valid_internal_anchors(self):
        for language in ('en','fr'):
            for route in PAGES:
                parser=EvidenceParser();parser.feed(render_page(route,language).decode())
                self.assertEqual(len(parser.ids),len(set(parser.ids)),route)
                for href in parser.links:
                    if href.startswith('#'):
                        self.assertIn(href[1:],parser.ids,(route,href))
                    prefix='https://github.com/wce-research/wce/blob/main/'
                    if href.startswith(prefix):
                        self.assertTrue((ROOT/href[len(prefix):]).is_file(),href)

    def test_generated_test_inventory_is_current(self):
        self.assertEqual(json.loads((ROOT/'app/test_inventory.json').read_text()),inventory())

    def test_scientific_scope_remains_explicit(self):
        for language in ('en','fr'):
            status=render_page('/vision/now',language).decode()
            for label in ('IMPLEMENTED / CURRENT','EXPERIMENTAL','BLOCKED / OPEN PROBLEM','FUTURE / RESEARCH DIRECTION','LONG-TERM VISION'):
                self.assertIn(label,status)
            for route in SCIENCE_ROUTES:
                page=render_page(route,language).decode()
                self.assertNotRegex(page,r'AGI achieved|proven superior architecture|solved perception|revolutionary AI')
        self.assertIn('exact atoms supplied',render_page('/research/deep-composition','en').decode().lower())
        self.assertIn('not perception benchmarks on real scenes',render_page('/research/high-precision','en').decode())

    def test_language_priority_and_quality_values(self):
        cases=[(None,'','en'),(None,'de-DE,fr-FR;q=0.8,en;q=0.7','fr'),
               (None,'fr;q=0.2,en-US;q=0.9','en'),('en','fr-FR','en'),('fr','en-US','fr'),
               ('invalid','fr;q=0,en;q=0.5','en'),(None,'fr;q=invalid','en'),
               (None,'fr;q=2','en')]
        for cookie,header,expected in cases:
            self.assertEqual(preferred_language(cookie,header),expected)
        self.assertIn('<html lang="en">',render_page('/').decode())
        self.assertIn('programme de recherche ouvert',render_page('/','fr').decode())

    def test_template_source_language_is_english(self):
        for path in (ROOT/'app/templates/pages').glob('*.html'):
            self.assertNotRegex(path.read_text(),r'\b(?:aujourd’hui|recherche ouverte|Où est la balle|Lire le dossier)\b',path.name)

    def test_all_scrollable_tables_are_keyboard_reachable(self):
        for language in ('en','fr'):
            for route in PAGES:
                page=render_page(route,language).decode()
                for attributes in re.findall(r'<div([^>]*class="table-scroll"[^>]*)>',page):
                    self.assertIn('tabindex="0"',attributes)
                    self.assertIn('role="region"',attributes)
                    self.assertIn('aria-label=',attributes)
