import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from app.site import PAGES, render_page
from app.public_identity import GITHUB_REPO

class PublicIdentityTests(unittest.TestCase):
    def test_metadata_and_branding_on_all_pages(self):
        for language in ('fr','en'):
            for route in PAGES:
                page=render_page(route,language).decode()
                canonical='https://wcelabs.com/'+language+(route if route!='/' else '/')
                self.assertIn(f'rel="canonical" href="{canonical}"',page)
                self.assertIn('property="og:site_name" content="WCE Labs"',page)
                self.assertIn('name="twitter:card" content="summary_large_image"',page)
                self.assertIn('>LABS</span>',page)
                self.assertNotIn('>RESEARCH</span>',page)
                self.assertIn('href="https://github.com/wce-research"',page)
                self.assertIn('href="'+GITHUB_REPO+'"',page)
                self.assertIn('mailto:security@wcelabs.com',page)
                self.assertEqual(page.count('mailto:'),1)
    def test_contact_validation_and_repo_switch(self):
        with patch.dict(os.environ,{'PUBLIC_CONTACT_EMAIL':'research@wcelabs.com','PUBLIC_WCE_REPO_ENABLED':'true'}):
            page=render_page('/').decode()
            self.assertIn('mailto:research@wcelabs.com',page)
            self.assertIn('href="'+GITHUB_REPO+'"',page)
        for value in ('personal@example.com','a@wcelabs.com?subject=x','a@wcelabs.com\nBcc:x','javascript:alert(1)'):
            with patch.dict(os.environ,{'PUBLIC_CONTACT_EMAIL':value}):self.assertEqual(render_page('/').decode().count('mailto:'),1)
    def test_canonical_release_and_protocol_language(self):
        page = render_page('/open/source', 'en').decode()
        self.assertIn('Apache-2.0', page)
        self.assertIn('canonical repository', page)
        self.assertNotIn('French source documents', page)
        self.assertIn('protocol (English)', render_page('/research/discovery', 'en').decode())
        with patch.dict(os.environ, {'PUBLIC_WCE_REPO_ENABLED': 'false'}):
            self.assertNotIn('href="'+GITHUB_REPO+'"', render_page('/').decode())

    def test_assets_and_manifest(self):
        from app.web import ASSETS, STATIC
        for route,(path,_) in ASSETS.items():
            if route.startswith('/brand/') or route=='/site.webmanifest':self.assertTrue((STATIC/path).is_file())
        manifest=json.loads((STATIC/'site.webmanifest').read_text())
        self.assertEqual(manifest['name'],'WCE Labs')
        self.assertEqual(manifest['display'],'browser')
