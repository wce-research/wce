"""Shared server-rendered shell; page content and public configuration stay separate."""
from html import escape
import os
from pathlib import Path
from string import Template
from urllib.parse import urlsplit
import re
import xml.etree.ElementTree as ET

from app.content import DOSSIERS, render_dossier
from app.i18n import localized_path, translate_content

STATIC = Path(__file__).with_name("static")
TEMPLATES = Path(__file__).with_name("templates")
PAGES = {
    "/": ("index.html", "Open architecture research"),
    "/research": ("research.html", "Recherche"),
    "/architecture": ("architecture.html", "Architecture"),
    "/vision": ("vision.html", "Vision"),
    "/open": ("open.html", "Open research"),
    "/lab": ("lab.html", "Laboratoire"),
    "/research/measurement": ("measurement.html", "Mesure et incertitude"),
    "/research/continual": ("continual.html", "Apprentissage continu"),
    "/research/discovery": ("discovery.html", "Découverte active"),
}
for route, record in DOSSIERS.items():
    PAGES[route] = (None, record['fr'][0])

EN_TITLES = {
    "/": "Open architecture research", "/research": "Research", "/architecture": "Architecture",
    "/vision": "Vision", "/open": "Open research", "/lab": "Laboratory",
    "/research/measurement": "Measurement & uncertainty", "/research/continual": "Continual learning",
    "/research/discovery": "Active discovery",
}
NAVIGATION = [
    ("/research", ("Recherche", "Research"),
     ("Mesures, protocoles et questions ouvertes.", "Measurements, protocols and open questions."),
     ["/research", "/research/measurement", "/research/continual", "/research/discovery", "/research/methodology", "/research/limitations"]),
    ("/architecture", ("Architecture", "Architecture"),
     ("Inspecter les mécanismes du prototype.", "Inspect the mechanisms of the prototype."),
     ["/architecture", "/architecture/primitives", "/architecture/world-state", "/architecture/learning", "/architecture/planning"]),
    ("/vision", ("Vision", "Vision"),
     ("L’état actuel, l’exploration et l’ambition.", "Current state, exploration and ambition."),
     ["/vision", "/vision/now", "/vision/exploration", "/vision/frontier-hub"]),
    ("/open", ("Recherche ouverte", "Open research"),
     ("Reproduire, inspecter et prolonger.", "Reproduce, inspect and extend."),
     ["/open", "/open/reproduce", "/open/source", "/open/contribute"]),
    ("/lab", ("Laboratoire", "Laboratory"),
     ("Expérimenter dans un monde explicite.", "Experiment in an explicit world."),
     ["/lab", "/lab/guide", "/lab/protocol", "/lab/limits"]),
]


def page_title(route, language):
    if route in DOSSIERS:
        return DOSSIERS[route][language][0]
    return EN_TITLES[route] if language == 'en' else ('Recherche en architectures ouvertes' if route == '/' else PAGES[route][1])


def navigation_html(route, language):
    index = 1 if language == 'en' else 0
    pieces=[]
    for category, names, descriptions, children in NAVIGATION:
        slug=category[1:]; name=names[index]
        current = route == category or route.startswith(category + '/')
        links=[]
        for child in children:
            label = ('Vue d’ensemble' if language == 'fr' else 'Overview') if child == category else page_title(child,language)
            aria = ' aria-current="page"' if child == route else ''
            links.append(f'<a href="{localized_path(child,language)}"{aria}>{escape(label)} <span aria-hidden="true">↗</span></a>')
        sublabel=('Sous-pages : ' if language == 'fr' else 'Subpages: ') + name
        pieces.append(f'<div class="nav-group" data-section="{slug}"><div class="nav-top"><a class="nav-heading" href="{localized_path(category,language)}"'+(' aria-current="page"' if current else '')+f'>{name}</a><button class="nav-toggle" aria-expanded="false" aria-controls="nav-{slug}" aria-label="{escape(sublabel,quote=True)}"><span aria-hidden="true">⌄</span></button></div><div class="nav-panel" id="nav-{slug}" hidden><div class="nav-intro"><p class="eyebrow">WCE / {name}</p><h2>{escape(descriptions[index])}</h2></div><div class="nav-links">'+''.join(links)+'</div></div></div>')
    return ''.join(pieces)

SOCIALS = [
    ("GitHub", "WCE_GITHUB_URL", {"github.com", "www.github.com"}, '<path d="M9 19c-4 1-4-2-6-2m12 4v-4c0-1-.3-1.8-1-2.2 3-.3 6-1.5 6-5.5 0-1.5-.5-2.7-1.5-3.7.2-.9.2-2-.2-3.1-1.5 0-3 1.1-3.5 1.5-2-.6-4-.6-6 0C8.8 3.6 7.3 2.5 5.8 2.5c-.4 1.1-.4 2.2-.2 3.1C4.6 6.6 4 7.8 4 9.3c0 4 3 5.2 6 5.5-.7.4-1 1.2-1 2.2v4"/>'),
    ("X", "WCE_X_URL", {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}, '<path d="m4 3 16 18M20 3 4 21M4 3h4l12 18h-4"/>'),
    ("LinkedIn", "WCE_LINKEDIN_URL", {"linkedin.com", "www.linkedin.com"}, '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 10v7m0-10v.1M11 17v-7m0 3c0-4 6-4 6 0v4"/>'),
    ("Instagram", "WCE_INSTAGRAM_URL", {"instagram.com", "www.instagram.com"}, '<rect x="3" y="3" width="18" height="18" rx="5"/><circle cx="12" cy="12" r="4"/><path d="M17 7h.01"/>'),
]


def social_links(language='fr') -> str:
    links = []
    for name, variable, hosts, icon in SOCIALS:
        from app.public_identity import GITHUB_ORG
        url = (GITHUB_ORG if name == "GitHub" else os.environ.get(variable, "").strip())
        try:
            parsed = urlsplit(url)
            valid = (parsed.scheme == "https" and parsed.hostname in hosts
                     and not parsed.username and not parsed.password
                     and parsed.port in (None, 443) and not any(c.isspace() for c in url))
        except ValueError:
            valid = False
        asset = STATIC / 'social' / (name.lower() + '.svg')
        if asset.exists():
            source = ET.fromstring(asset.read_text())
            paths = ''.join(f'<path d="{escape(node.attrib["d"], quote=True)}"/>' for node in source.iter() if node.tag.endswith('path'))
            svg = f'<svg viewBox="0 0 24 24" aria-hidden="true" class="social-logo">{paths}</svg>'
        else:
            svg = f'<svg viewBox="0 0 24 24" aria-hidden="true">{icon}</svg>'
        content = svg + f'<span>{name}</span>'
        action_label = f'WCE on {name} (new tab)' if language == 'en' else f'WCE sur {name} (nouvel onglet)'
        pending_label = f'{name}, official link to come' if language == 'en' else f'{name}, lien officiel à venir'
        pending_text = 'Coming soon' if language == 'en' else 'À venir' 
        if valid:
            links.append(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer" aria-label="{action_label}">{content}</a>')
        else:
            links.append(f'<span class="social-pending" aria-label="{pending_label}">{content}<small>{pending_text}</small></span>')
    return "".join(links)


def render_page(route: str, language='fr') -> bytes:
    if language not in ('fr', 'en'):
        raise ValueError('Unsupported language')
    filename, _ = PAGES[route]
    content = render_dossier(route,language) if route in DOSSIERS else translate_content((TEMPLATES / 'pages' / filename).read_text(),language)
    if route == '/open':
        from app.public_identity import research_context
        content += research_context(language)
    if language == 'en' and re.search(r'href="/research/[^"]+\.md"',content):
        content += '<p class="section source-language-note note">Source documents are currently in French. Scientific JSON artifacts are unchanged.</p>'
    # Localise page links only. Source documents, JSON and exported assets keep stable URLs.
    def page_link(match):
        target=match.group(1)
        return f'href="{localized_path(target,language)}"' if target in PAGES else match.group(0)
    content=re.sub(r'href="([^"]+)"',page_link,content)
    ui = {
      'fr': dict(skip='Aller au contenu',home_label='WCE Labs, accueil',menu_open='Menu +',menu_close='Fermer −',nav_label='Navigation principale',theme_dark='Mode nuit',theme_light='Mode jour',language_label='Langue',footer_message='Une architecture à explorer.<br>Une recherche à partager.',prototype='Prototype expérimental · 2026',open_research='Ouvrir la recherche',follow='SUIVRE WCE',social_label='Réseaux sociaux WCE',description='WCE explore des architectures alternatives d’intelligence artificielle. Architecture explicite, résultats mesurés et recherche ouverte.'),
      'en': dict(skip='Skip to content',home_label='WCE Labs, home',menu_open='Menu +',menu_close='Close −',nav_label='Main navigation',theme_dark='Dark mode',theme_light='Light mode',language_label='Language',footer_message='An architecture to explore.<br>Research to share.',prototype='Experimental prototype · 2026',open_research='Open the research',follow='FOLLOW WCE',social_label='WCE social profiles',description='WCE explores alternative AI architectures. Explicit mechanisms, measured results and open research.'),
    }[language]
    from app.public_identity import metadata, footer_links, DESCRIPTIONS
    ui['description']=DESCRIPTIONS[language]
    return Template((TEMPLATES / 'layout.html').read_text()).substitute(
        **ui,public_metadata=metadata(route,language,page_title(route,language)),official_links=footer_links(language),language=language,title=escape(page_title(route,language)),navigation=navigation_html(route,language),socials=social_links(language),
        home=localized_path('/',language),open_url=localized_path('/open',language),fr_url=localized_path(route,'fr'),en_url=localized_path(route,'en'),
        fr_current=' aria-current="true"' if language == 'fr' else '',en_current=' aria-current="true"' if language == 'en' else '',
        page_script='<script type="module" src="/app.js"></script>' if route == '/lab' else '',content=content,
    ).encode()
