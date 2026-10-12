"""Two explicit locales; URL language and scientific source artifacts stay separate."""
import json
from pathlib import Path
import re

ENGLISH = json.loads(Path(__file__).with_name('copy.en.json').read_text())
FRENCH_COPY = json.loads(Path(__file__).with_name('copy.fr.json').read_text())
FRENCH = {
    'Intelligence,': 'Une intelligence,', 'with its architecture': 'une architecture', 'wide open.': 'grande ouverte.',
    'Make the mechanisms': 'Rendre les mécanismes', 'visible.': 'visibles.',
    'OPEN ARCHITECTURE RESEARCH': 'RECHERCHE EN ARCHITECTURES OUVERTES',
    'RESEARCH INDEX': 'INDEX DE RECHERCHE', 'RESEARCH DIRECTION': 'DIRECTION DE RECHERCHE',
    'OPEN RESEARCH': 'RECHERCHE OUVERTE', 'INTERACTIVE LAB': 'LABORATOIRE INTERACTIF',
    'WHAT EXISTS NOW': 'CE QUI EXISTE AUJOURD’HUI', 'WHAT IS BEING EXPLORED': 'CE QUI EST EXPLORÉ',
    'WHAT WCE AIMS TO BECOME': 'CE QUE WCE AMBITIONNE DE DEVENIR', 'RESEARCH STATUS': 'ÉTAT DE LA RECHERCHE',
    'RESEARCH NOTE': 'NOTE DE RECHERCHE', 'LOCAL ENGINE': 'MOTEUR LOCAL',
    'WCE / SYMBOLIC SESSION': 'WCE / SESSION SYMBOLIQUE', 'OBSERVE → STRUCTURE → EXPERIMENT': 'OBSERVER → STRUCTURER → EXPÉRIMENTER',
    'WCE / FIELD NOTES': 'WCE / NOTES DE TERRAIN', 'Reasoning & world state': 'Raisonnement & état du monde',
    'Choisissez une commande, puis exécutez-la. Les commandes acceptées utilisent une grammaire française limitée.': 'Choisissez une commande, puis exécutez-la. L’interface accepte une grammaire explicite et limitée, en français ou en anglais.',
}

def localized_path(route, language):
    return f'/{language}' + (route if route != '/' else '/')

def resolve_route(path):
    parts = path.split('/')
    language = parts[1] if len(parts) > 1 and parts[1] in ('fr', 'en') else 'en'
    route = '/' + '/'.join(parts[2:]) if len(parts) > 1 and parts[1] in ('fr', 'en') else path
    return route.rstrip('/') or '/', language


def preferred_language(cookie_value, accept_language):
    """Explicit saved choice wins; otherwise select the highest-ranked supported locale."""
    if cookie_value in ('en', 'fr'):
        return cookie_value
    candidates = []
    for position, entry in enumerate(accept_language.split(',')[:32]):
        fields = entry.strip().split(';')
        tag = fields[0].strip().lower().split('-')[0]
        quality = 1.0
        try:
            for field in fields[1:]:
                if field.strip().startswith('q='):
                    quality = float(field.strip()[2:])
        except ValueError:
            continue
        if tag in ('en', 'fr') and 0 < quality <= 1:
            candidates.append((quality, -position, tag))
    return max(candidates)[2] if candidates else 'en'

def translate_content(content, language):
    from html import escape
    catalog = ENGLISH if language == 'en' else FRENCH_COPY
    def translate(match):
        text = match.group(1); key = text.strip()
        if key not in catalog:
            return match.group(0)
        leading = text[:len(text)-len(text.lstrip())]
        trailing = text[len(text.rstrip()):]
        return '>' + leading + escape(catalog[key]) + trailing + '<'
    content = re.sub(r'>([^<>]+)<', translate, content)
    if language == 'en':
        for before, after in {
            'placeholder="Où est la balle ?"': 'placeholder="Where is the ball?"',
            'aria-label="Nombre de caractères"': 'aria-label="Character count"',
            'aria-label="Exécuter la commande"': 'aria-label="Run the command"',
            'aria-label="Filtrer les dossiers"': 'aria-label="Filter research notes"',
            'data-command="Soulève la balle à 2 mètres"': 'data-command="Lift the ball to 2 metres"',
            'data-command="Où est la balle ?"': 'data-command="Where is the ball?"',
            'data-command="Avance le temps de 1 seconde"': 'data-command="Advance time by 1 second"',
            'aria-label="Composition abstraite : trajectoires, observations et structure"': 'aria-label="Abstract composition: trajectories, observations and structure"',
            'aria-label="Étude graphique de trajectoires convergeant vers une structure ouverte"': 'aria-label="Graphical study of trajectories converging towards an open structure"',
        }.items():content = content.replace(before,after)
    if language == 'fr':
        for before, after in {
            'placeholder="Where is the ball?"': 'placeholder="Où est la balle ?"',
            'aria-label="Character count"': 'aria-label="Nombre de caractères"',
            'aria-label="Run the command"': 'aria-label="Exécuter la commande"',
            'aria-label="Filter research notes"': 'aria-label="Filtrer les dossiers"',
            'data-command="Lift the ball to 2 metres"': 'data-command="Soulève la balle à 2 mètres"',
            'data-command="Where is the ball?"': 'data-command="Où est la balle ?"',
            'data-command="Advance time by 1 second"': 'data-command="Avance le temps de 1 seconde"',
        }.items():
            content = content.replace(before,after)
    return content
