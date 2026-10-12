import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from app.web import Handler


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, data=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        conn.request(method, path, body=data, headers=headers or {})
        response = conn.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return result

    def command(self, text, cookie=None):
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        return self.request("POST", "/api/command", json.dumps({"command": text}), headers)

    def test_page_health_and_script(self):
        status, _, body = self.request("GET", "/healthz")
        self.assertEqual((status, json.loads(body)["status"]), (200, "ok"))
        status, _, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b'/site.js', body)
        self.assertIn(b'Open weights are not enough', body)
        self.assertEqual(self.request("GET", "/app.js")[0], 200)
        status, _, body = self.request("GET", "/research/predictive_coverage.json")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["all"]["n"], 96)
        status, _, body = self.request("GET", "/research/architecture_wce.md")
        self.assertEqual(status, 200)
        self.assertIn(b"Open weights are not enough", body)

    def test_distinct_pages_assets_and_route_safety(self):
        from app.web import PAGES, ASSETS
        pages = []
        for route in PAGES:
            status, headers, body = self.request("GET", route)
            self.assertEqual(status, 200, route)
            self.assertIn("text/html", headers["Content-Type"])
            self.assertIn(b'id="main"', body)
            pages.append(body)
        self.assertEqual(len(set(pages)), len(PAGES))
        self.assertNotIn(b'id="form"', pages[0])
        status, _, body = self.request("GET", "/lab")
        self.assertIn(b'id="form"', body)
        self.assertIn(b'/app.js', body)
        for route, (_, content_type) in ASSETS.items():
            status, headers, _ = self.request("GET", route)
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], content_type)
        self.assertEqual(self.request("GET", "/research/")[0], 200)
        for path in ("/../app/web.py", "/static/index.html", "/research/missing"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_social_configuration_is_public_validated_and_shared(self):
        import os
        from unittest.mock import patch
        from app.site import render_page
        with patch.dict(os.environ, {"WCE_GITHUB_URL": "https://github.com/wce-test", "WCE_X_URL": "javascript:alert(1)", "WCE_LINKEDIN_URL": "https://linkedin.com.evil.test/wce", "WCE_INSTAGRAM_URL": "https://instagram.com/wce-test"}):
            for route in ("/", "/architecture", "/lab"):
                body = render_page(route)
                self.assertIn(b'href="https://github.com/wce-research"', body)
                self.assertIn(b'href="https://instagram.com/wce-test"', body)
                self.assertIn(b'rel="noopener noreferrer"', body)
                self.assertNotIn(b'javascript:', body)
                self.assertNotIn(b'evil.test', body)

    def test_official_social_defaults_in_every_localized_footer(self):
        import os
        from unittest.mock import patch
        from app.site import PAGES, render_page
        from html.parser import HTMLParser

        class SocialLinks(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'a' and attrs.get('target') == '_blank':
                    self.links.append(attrs)

        expected = {
            'https://github.com/wce-research',
            'https://x.com/WCELabs',
            'https://www.instagram.com/wcelabs/',
            'https://www.linkedin.com/company/wce-labs/',
        }
        # Empty environment values preserve the confirmed official defaults.
        with patch.dict(os.environ, {key: '' for key in (
                'WCE_GITHUB_URL', 'WCE_X_URL', 'WCE_LINKEDIN_URL', 'WCE_INSTAGRAM_URL')}):
            for language in ('fr', 'en'):
                for route in PAGES:
                    footer = render_page(route, language).decode().split('<footer', 1)[1].split('</footer>', 1)[0]
                    parser = SocialLinks()
                    parser.feed(footer)
                    self.assertEqual({link['href'] for link in parser.links}, expected)
                    for link in parser.links:
                        self.assertEqual(link['rel'], 'noopener noreferrer')
                        self.assertIn('aria-label', link)
                    self.assertNotIn('class="social-pending"', footer)
                    self.assertNotIn('Coming soon', footer)
                    self.assertNotIn('À venir', footer)
                    self.assertIn('LinkedIn', footer)

    def test_identity_review_and_assets_are_explicit(self):
        import struct
        self.assertEqual(self.request("GET", "/identity")[0], 308)
        status, _, body = self.request("GET", "/identity/")
        self.assertEqual(status, 200)
        self.assertIn(b'finalists/c2/logo-horizontal.svg', body)
        self.assertEqual(self.request("GET", "/identity/boards/initial-12.svg")[0], 200)
        status, headers, body = self.request("GET", "/identity/finalists/c2/favicon.ico")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/vnd.microsoft.icon")
        self.assertEqual(struct.unpack("<HHH", body[:6]), (0, 1, 3))
        for route in ("/identity/build.py", "/identity/assets/DejaVuSans.ttf", "/identity/../web.py"):
            self.assertEqual(self.request("GET", route)[0], 404)

    def test_browser_language_and_saved_preference(self):
        for header, cookie, expected in (
                ('', '', 'en'), ('fr-FR,fr;q=0.9,en;q=0.8', '', 'fr'),
                ('en-US', 'wce_language=fr', 'fr'), ('fr-FR', 'wce_language=en', 'en')):
            _,_,body=self.request('GET','/research',headers={'Accept-Language':header,'Cookie':cookie})
            self.assertIn(f'<html lang="{expected}">'.encode(),body)
        _,_,body=self.request('GET','/en/research',headers={'Accept-Language':'fr','Cookie':'wce_language=fr'})
        self.assertIn(b'<html lang="en">',body)
        status,headers,_=self.request('GET','/',headers={'Cookie':'wce_language=fr'})
        self.assertEqual((status,headers['Location']),(307,'/fr/'))

    def test_sitemap_robots_and_browser_404(self):
        from app.site import PAGES
        import xml.etree.ElementTree as ET
        status,_,body=self.request('GET','/sitemap.xml')
        self.assertEqual(status,200)
        locations=[node.text for node in ET.fromstring(body).iter() if node.tag.endswith('loc')]
        self.assertEqual(len(locations),len(PAGES)*2)
        self.assertIn('https://wcelabs.com/en/research/high-precision',locations)
        self.assertIn(b'Disallow: /identity/',self.request('GET','/robots.txt')[2])
        status,headers,body=self.request('GET','/en/missing',headers={'Accept':'text/html'})
        self.assertEqual(status,404)
        self.assertIn('text/html',headers['Content-Type'])
        self.assertIn(b'noindex',body)
        self.assertIn(b'Return to WCE Labs',body)

    def test_bilingual_routes_and_deep_navigation(self):
        from app.site import PAGES, page_title
        from app.i18n import resolve_route
        from html.parser import HTMLParser
        class Links(HTMLParser):
            def __init__(self):
                super().__init__(); self.links=[]
            def handle_starttag(self, tag, attrs):
                if tag == 'a' and not any(key == 'data-language' for key,_ in attrs):
                    self.links.extend(value for key,value in attrs if key == 'href')
        for language in ('fr','en'):
            for route in PAGES:
                url = '/' + language + (route if route != '/' else '/')
                status,_,body = self.request('GET',url)
                self.assertEqual(status,200,url)
                text=body.decode()
                self.assertIn(f'lang="{language}"',text)
                self.assertIn(page_title(route,language).replace('&','&amp;'),text)
                self.assertIn('footer-brand',text)
                self.assertEqual(text.count('class="nav-toggle"'),5)
                parser=Links();parser.feed(text)
                for href in parser.links:
                    target, locale=resolve_route(href)
                    if target in PAGES:
                        self.assertEqual(locale,language,href)
                        self.assertTrue(href.startswith('/'+language+'/'),href)
        self.assertEqual(self.request('GET','/de/research')[0],404)
        self.assertEqual(self.request('GET','/fr/research/methodology')[0],200)
        status,headers,_=self.request('GET','/',headers={'Cookie':'wce_language=en'})
        self.assertEqual((status,headers['Location']),(307,'/en/'))
        self.assertEqual(self.request('GET','/fr/',headers={'Cookie':'wce_language=en'})[0],200)

    def test_english_laboratory_preserves_session_and_validation(self):
        headers={'Content-Type':'application/json','X-WCE-Language':'en'}
        status,response,body=self.request('POST','/api/command',json.dumps({'command':'Lift the ball to 2 metres'}),headers)
        self.assertEqual((status,json.loads(body)['answer']),(200,'OK.'))
        headers['Cookie']=response['Set-Cookie'].split(';',1)[0]
        status,_,body=self.request('POST','/api/command',json.dumps({'command':'Where is the ball?'}),headers)
        self.assertEqual((status,json.loads(body)['answer']),(200,'The ball is at a height of 2 m.'))
        status,_,body=self.request('POST','/api/command',json.dumps({'command':'Advance time by 1000000 seconds'}),headers)
        self.assertEqual((status,json.loads(body)['error']),(422,'The maximum duration is 60 seconds.'))
        headers['Origin']='http://evil.test'; headers['Host']='127.0.0.1'
        self.assertEqual(self.request('POST','/api/command',json.dumps({'command':'Where is the ball?'}),headers)[0],403)

    def test_session_isolation_and_command_flow(self):
        status, headers, body = self.command("Soulève la balle à 2 mètres")
        self.assertEqual((status, json.loads(body)["answer"]), (200, "OK."))
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        status, _, body = self.command("Où est la balle ?", cookie)
        self.assertEqual((status, json.loads(body)["answer"]), (200, "Balle se trouve à 2 m de hauteur."))
        status, _, body = self.command("Où est la balle ?")
        self.assertEqual((status, json.loads(body)["answer"]), (200, "Balle est sur sol."))

    def test_rejects_invalid_input_and_cross_origin(self):
        self.assertEqual(self.command(" ")[0], 400)
        self.assertEqual(self.command("x" * 501)[0], 400)
        self.assertEqual(self.command("Avance le temps de 1000000 secondes")[0], 422)
        self.assertEqual(self.request("POST", "/api/command", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/command", "{}", {"Content-Type": "application/json", "Origin": "http://evil.test", "Host": "127.0.0.1"})[0], 403)
        self.assertEqual(self.request("GET", "/missing")[0], 404)

    def test_unexpected_engine_failure_is_a_safe_500(self):
        from unittest.mock import patch
        with patch('app.web.Session.execute', side_effect=RuntimeError('internal diagnostic')):
            status, _, body = self.command('Soulève la balle à 2 mètres')
        self.assertEqual(status, 500)
        self.assertNotIn(b'internal diagnostic', body)
        self.assertIn('error', json.loads(body))

    def test_backend_catalog_sources_and_inventory_are_current(self):
        from pathlib import Path
        from app.research_catalog import FAMILIES
        from tools.build_backend_inventory import inventory
        root=Path(__file__).resolve().parents[1]
        for family in FAMILIES:
            for source in family['sources'] + family['tests']:
                self.assertTrue((root/source).is_file(),source)
        self.assertEqual(json.loads((root/'app/backend_inventory.json').read_text()),inventory(),
                         'Regenerate the source inventory after changing backend modules')

    def test_backend_presentations_and_cards(self):
        from app.research_catalog import FAMILIES
        for language in ('fr','en'):
            status,_,body=self.request('GET',f'/{language}/architecture')
            self.assertEqual(status,200)
            html=body.decode()
            self.assertEqual(html.count('class="engine-card engine-'),10)
            self.assertIn('language/command.py',html)
            for family in FAMILIES:
                self.assertIn(f'href="/{language}{family["route"]}"',html)
                _,_,page=self.request('GET',f'/{language}{family["route"]}')
                for source in family['sources']:
                    self.assertIn(source.encode(),page)
            _,_,page=self.request('GET',f'/{language}/open/source')
            self.assertIn(b'core/language_core.py',page)
            self.assertIn(b'StructuralVerifier',page)
            self.assertIn(b'core/planner.py',page)
            self.assertIn(b'TransformationPlanner',page)
            for section in ('research','vision','open','lab'):
                _,_,page=self.request('GET',f'/{language}/{section}')
                self.assertIn(b'class="section-card"',page)

    def test_research_views_follow_the_served_artifacts(self):
        from html.parser import HTMLParser
        from app.evidence import METRICS
        class Records(HTMLParser):
            def __init__(self):
                super().__init__(); self.rows=[]
            def handle_starttag(self,tag,attrs):
                values=dict(attrs)
                if any(key in values for key in ('data-source','data-estimate','data-hypothesis')):
                    self.rows.append(values)
        for language in ('fr','en'):
            _,_,raw=self.request('GET','/research/predictive_coverage.json')
            data=json.loads(raw)
            _,_,body=self.request('GET',f'/{language}/research/measurement')
            parser=Records();parser.feed(body.decode())
            self.assertEqual(len(parser.rows),len(data))
            for row in parser.rows:
                record=data[row['data-source']]
                self.assertEqual(float(row['data-predictive']),record['predictive'])
                self.assertEqual(float(row['data-credible']),record['credible'])
                self.assertEqual(int(row['data-n']),record['n'])
            _,_,raw=self.request('GET','/research/continuum_summary.json')
            methods=json.loads(raw)['methods']
            _,_,body=self.request('GET',f'/{language}/research/continual')
            parser=Records();parser.feed(body.decode())
            self.assertEqual(len(parser.rows),len(methods)*len(METRICS))
            for row in parser.rows:
                record=methods[row['data-method']][row['data-metric']]
                self.assertEqual(float(row['data-estimate']),record['mean'])
                self.assertEqual([float(row['data-ci-low']),float(row['data-ci-high'])],record['ci95'])
                self.assertEqual(int(row['data-n']),record['n'])
            _,_,raw=self.request('GET','/research/active_summary.json')
            hypotheses=json.loads(raw)['hypotheses']
            _,_,body=self.request('GET',f'/{language}/research/discovery')
            parser=Records();parser.feed(body.decode())
            self.assertEqual(len(parser.rows),len(hypotheses))
            for row in parser.rows:
                self.assertEqual(row['data-verdict'],hypotheses[row['data-hypothesis']]['verdict'])

    def test_evidence_updates_when_artifact_values_change(self):
        from copy import deepcopy
        from unittest.mock import patch
        from app.evidence import artifact,predictive
        data,digest=artifact('predictive_coverage.json');updated=deepcopy(data)
        updated['all']['predictive']=0.8123
        updated['all']['n']=101
        with patch('app.evidence.artifact',return_value=(updated,digest)):
            html=predictive('en')
        self.assertIn('data-predictive="0.8123"',html)
        self.assertIn('data-n="101"',html)
        self.assertIn(digest[:12],html)

    def test_configuration_and_diagrams_have_explicit_scope(self):
        from app.research_catalog import FAMILIES
        self.assertEqual(self.request('GET','/frontier.js')[0],200)
        _,_,body=self.request('GET','/en/')
        self.assertEqual(body.count(b'data-primitive='),3)
        self.assertEqual(body.count(b'data-configuration='),3)
        self.assertIn(b'without inference or scientific simulation',body)
        for family in FAMILIES:
            _,_,body=self.request('GET','/en'+family['route'])
            self.assertIn(b'MECHANISM SCHEMATIC',body)
            self.assertIn(b'neither an execution trace nor a performance measurement',body)


if __name__ == "__main__":
    unittest.main()
