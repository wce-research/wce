"""Public identity settings only; no deployment credentials or private hub metadata."""
from html import escape
import json
import os
import re
from urllib.parse import urlsplit

SITE_NAME = 'WCE Labs'
SITE_URL = 'https://wcelabs.com'
GITHUB_ORG = 'https://github.com/wce-research'
GITHUB_REPO = 'https://github.com/wce-research/wce'
SOCIAL_URLS = {
 'WCE_GITHUB_URL': GITHUB_ORG,
 'WCE_X_URL': 'https://x.com/WCELabs',
 'WCE_LINKEDIN_URL': '',
 'WCE_INSTAGRAM_URL': 'https://www.instagram.com/wcelabs/',
}
DESCRIPTIONS = {
 'en': 'WCE Labs researches compact, compositional and verifiable AI architectures through the World Configuration Engine.',
 'fr': 'WCE Labs étudie des architectures d’IA compactes, compositionnelles et vérifiables à travers le World Configuration Engine.',
}


def contact_email():
    value=os.environ.get('PUBLIC_CONTACT_EMAIL','').strip()
    return value if re.fullmatch(r'[A-Za-z0-9.!#$%&\'*+/=?^_`{|}~-]+@wcelabs\.com',value) else ''


def repository_link():
    return GITHUB_REPO if os.environ.get('PUBLIC_WCE_REPO_ENABLED','true').lower()=='true' else ''


def metadata(route, language, title):
    from app.i18n import localized_path
    canonical=SITE_URL+localized_path(route,language)
    full_title=title+' — '+SITE_NAME
    description=DESCRIPTIONS[language]
    image=SITE_URL+'/brand/wce-labs-og.png'
    tags=[f'<link rel="canonical" href="{canonical}">']
    for lang in ('fr','en'):
        tags.append(f'<link rel="alternate" hreflang="{lang}" href="{SITE_URL+localized_path(route,lang)}">')
    tags.append(f'<link rel="alternate" hreflang="x-default" href="{SITE_URL+localized_path(route,"en")}">')
    for key,value in {'og:type':'website','og:site_name':SITE_NAME,'og:title':full_title,'og:description':description,'og:url':canonical,'og:image':image,'og:image:width':'1200','og:image:height':'630','og:image:alt':'WCE Labs — World Configuration Engine','og:locale':'fr_FR' if language=='fr' else 'en_US'}.items():
        tags.append(f'<meta property="{key}" content="{escape(value,quote=True)}">')
    for key,value in {'twitter:card':'summary_large_image','twitter:title':full_title,'twitter:description':description,'twitter:image':image,'twitter:image:alt':'WCE Labs — World Configuration Engine'}.items():
        tags.append(f'<meta name="{key}" content="{escape(value,quote=True)}">')
    org={'@context':'https://schema.org','@type':'Organization','name':SITE_NAME,'url':SITE_URL,'logo':SITE_URL+'/brand/wce-labs-avatar-light.png','description':description,'sameAs':[GITHUB_ORG]}
    if contact_email():org['email']=contact_email()
    tags.append('<script type="application/ld+json">'+json.dumps(org,ensure_ascii=False).replace('<','\\u003c')+'</script>')
    return ''.join(tags)


def footer_links(language):
    label='Contact' if language=='fr' else 'Contact'
    links=f'<a href="{SITE_URL}">wcelabs.com ↗</a>'
    links+='<a href="mailto:security@wcelabs.com">'+('Sécurité' if language=='fr' else 'Security')+'</a>'
    if contact_email():links+=f'<a href="mailto:{escape(contact_email(),quote=True)}">{label}</a>'
    if repository_link():links+=f'<a href="{GITHUB_REPO}">'+('Code scientifique ↗' if language=='fr' else 'Scientific source ↗')+'</a>'
    return '<div class="official-links">'+links+'</div>'


def research_context(language):
    fr=language=='fr'
    heading='Le programme de recherche' if fr else 'The research programme'
    paragraphs=(
      ('WCE Labs est l’identité publique du laboratoire. World Configuration Engine est son projet de recherche principal ; wce-research est son organisation GitHub officielle.','WCE Labs is the laboratory’s public identity. World Configuration Engine is its main research project; wce-research is its official GitHub organisation.'),
      ('Le moteur symbolique, les prototypes et les artefacts d’évaluation existent. Leur intégration en un modèle entraînable unique reste expérimentale ; un hub communautaire opérationnel relève de la vision.','The symbolic engine, prototypes and evaluation artifacts exist. Integration into a single trainable model remains experimental; an operational community hub belongs to the vision.'),
      ('Le calcul sert à répéter les expériences sur plusieurs graines, comparer les stratégies, mesurer leurs coûts et tester leurs limites. Les scripts et protocoles définissent les budgets ; aucun crédit, partenariat ou allocation cloud n’est annoncé ici.','Compute is used to repeat experiments across seeds, compare strategies, measure their costs and test their limitations. Scripts and protocols define budgets; no credits, partnerships or cloud allocations are claimed here.'))
    text=''.join('<p>'+escape(a if fr else b)+'</p>' for a,b in paragraphs)
    return '<section class="section article"><p class="eyebrow">WCE LABS / OPEN RESEARCH</p><h2>'+heading+'</h2>'+text+'<div class="resource-links"><a href="/'+language+'/architecture">'+('Architecture' if fr else 'Architecture')+'</a><a href="/'+language+'/research/methodology">'+('Protocoles et hypothèses' if fr else 'Protocols and hypotheses')+'</a><a href="/'+language+'/research">'+('Expériences et résultats' if fr else 'Experiments and results')+'</a><a href="/'+language+'/open/reproduce">'+('Reproductibilité' if fr else 'Reproducibility')+'</a><a href="'+GITHUB_ORG+'">GitHub / wce-research ↗</a></div></section>'
