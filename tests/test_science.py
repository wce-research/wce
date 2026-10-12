"""Public evidence provenance, publication scope, language choice and source packaging."""
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

from app.i18n import preferred_language
from app.science import summary, report_sections, report_cell
from app.milestone_details import DETAILS
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
    def test_milestone_reading_notes_and_report_sections_are_source_backed(self):
        from hashlib import sha256
        self.assertEqual(set(DETAILS),{row[0] for row in MILESTONES})
        make=(ROOT/'Makefile').read_text()
        for name,notes in DETAILS.items():
            self.assertTrue((ROOT/notes[3]).is_file())
            self.assertRegex(make,r'(?m)^'+re.escape(notes[4][5:])+':')
            path,digest,sections=report_sections(name)
            raw=(ROOT/path).read_bytes();lines=raw.decode().splitlines()
            self.assertEqual(digest,sha256(raw).hexdigest())
            for heading,blocks in sections:
                self.assertIn(heading,lines)
                for kind,start,block in blocks:
                    if kind=='table':
                        self.assertEqual(tuple(lines[start:start+len(block)]),block)
                        width=len(block[0].strip('|').split('|'))
                        for row in block[2:]:
                            self.assertEqual(len(row.strip('|').split('|')),width,(name,row))
                    else:
                        self.assertEqual(lines[start][2:],block[0])
        with self.assertRaises(KeyError):
            report_sections('../private')

    def test_rendered_report_evidence_preserves_source_rows_in_both_languages(self):
        class ReportRows(HTMLParser):
            def __init__(self):
                super().__init__();self.rows=[];self.current=None
            def handle_starttag(self,tag,attrs):
                attrs=dict(attrs)
                if 'data-report-source' in attrs:
                    self.current=[tag,attrs,[]]
            def handle_data(self,value):
                if self.current is not None:self.current[2].append(value)
            def handle_endtag(self,tag):
                if self.current is not None and self.current[0]==tag:
                    self.rows.append(self.current);self.current=None
        for language in ('en','fr'):
            parser=ReportRows();parser.feed(render_page('/research/programme',language).decode())
            self.assertGreater(len(parser.rows),100)
            for tag,attrs,values in parser.rows:
                self.assertIn(attrs['data-report-source'],PUBLIC_REPORTS)
                line=(ROOT/attrs['data-report-source']).read_text().splitlines()[int(attrs['data-report-line'])-1]
                expected=[report_cell(c) or '—' for c in line.strip('|').split('|')] if tag=='tr' else [report_cell(line[2:])]
                self.assertEqual(''.join(values),''.join(expected))
            page=render_page('/research/programme',language).decode()
            self.assertIn('DC7',page);self.assertIn('LP1',page);self.assertIn('HP13',page)
            self.assertIn('lang="en"',page)
        objective=DETAILS['objective_alignment'][2][0]
        self.assertIn('None of the tested objectives',objective)
        self.assertIn('predefined guardrail criterion is met',DETAILS['validated_optimizer'][2][0])

    def test_historical_evidence_metrics_match_their_structured_sources(self):
        from app.evidence import artifact, METRICS
        class Attributes(HTMLParser):
            def __init__(self):super().__init__();self.nodes=[]
            def handle_starttag(self,tag,attrs):self.nodes.append(dict(attrs))
        for language in ('en','fr'):
            data,_=artifact('continuum_summary.json');parser=Attributes();parser.feed(render_page('/research/continual',language).decode())
            rows=[a for a in parser.nodes if 'data-estimate' in a]
            self.assertEqual(len(rows),len(data['methods'])*len(METRICS))
            for a in rows:
                r=data['methods'][a['data-method']][a['data-metric']]
                self.assertEqual(float(a['data-estimate']),r['mean'])
                self.assertEqual(float(a['data-ci-low']),r['ci95'][0])
                self.assertEqual(float(a['data-ci-high']),r['ci95'][1])
                self.assertEqual(int(a['data-n']),r['n'])
            data,_=artifact('active_summary.json');parser=Attributes();parser.feed(render_page('/research/discovery',language).decode())
            for a in parser.nodes:
                if 'data-hypothesis' in a:self.assertEqual(a['data-verdict'],data['hypotheses'][a['data-hypothesis']]['verdict'])
            data,_=artifact('predictive_coverage.json');parser=Attributes();parser.feed(render_page('/research/measurement',language).decode())
            coverage=[a for a in parser.nodes if 'data-predictive' in a]
            self.assertEqual(len(coverage),5)
            for a in coverage:
                r=data[a['data-source']]
                self.assertEqual(float(a['data-predictive']),r['predictive'])
                self.assertEqual(float(a['data-credible']),r['credible'])
                self.assertEqual(int(a['data-n']),r['n'])

    def test_architecture_explains_missing_measurement_and_discovery_stages(self):
        en=render_page('/architecture/research-map','en').decode()
        for key in ('Observable','ReferenceFrame / context','Fragment extraction','Type inference','Complete set','EvidenceGraph'):
            self.assertIn(key,en)
        fr=render_page('/architecture/research-map','fr').decode()
        for key in ('Observable','ReferenceFrame / context','Extraction des fragments','Inférence du type','Ensemble complet'):
            self.assertIn(key,fr)

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
