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
                self.assertIn(b'href="https://github.com/wce-test"', body)
                self.assertIn(b'href="https://instagram.com/wce-test"', body)
                self.assertIn(b'rel="noopener noreferrer"', body)
                self.assertNotIn(b'javascript:', body)
                self.assertNotIn(b'evil.test', body)

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


if __name__ == "__main__":
    unittest.main()
