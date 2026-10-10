"""Two explicit locales; URL language and scientific source artifacts stay separate."""
import json
from pathlib import Path
import re

ENGLISH = json.loads(Path(__file__).with_name('copy.en.json').read_text())
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
    language = parts[1] if len(parts) > 1 and parts[1] in ('fr', 'en') else 'fr'
    route = '/' + '/'.join(parts[2:]) if len(parts) > 1 and parts[1] in ('fr', 'en') else path
    return route.rstrip('/') or '/', language

def translate_content(content, language):
    from html import escape
    catalog = ENGLISH if language == 'en' else FRENCH
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
    return content
