"""Code-native conceptual diagrams, kept separate from measured research data."""
from html import escape


def box(x,y,w,h,label,accent=False):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" class="diagram-box'+(' diagram-accent' if accent else '')+f'"/><text x="{x+w/2}" y="{y+h/2+4}" text-anchor="middle">{escape(label)}</text>'


def link(x1,y1,x2,y2,dashed=False):
    return f'<path d="M{x1} {y1}H{x2}" class="diagram-link'+(' diagram-dashed' if dashed else '')+'"/>' if y1==y2 else f'<path d="M{x1} {y1}H{(x1+x2)/2}V{y2}H{x2}" class="diagram-link"/>'


def diagram_svg(key,language='fr',decorative=True):
    # Schematic relationships, never an execution trace or a result plot.
    en=language=='en'
    labels={
      'composition':('Primitives et composition de transformations','Primitives and transformation composition'),
      'world':('État, action et nouvel état','State, action and next state'),
      'planning':('Recherche bornée entre un état et un objectif','Bounded search between a state and a goal'),
      'language':('Texte, interprétation et décision vérifiée','Text, interpretation and checked decision'),
      'measurement':('État latent, mesure et états compatibles','Latent state, measurement and compatible states'),
      'sources':('Sources pondérées et ancrage indépendant','Weighted sources and independent grounding'),
      'certificates':('Générateur, sondes et vérification','Generator, probes and verification'),
      'learning':('Adaptation candidate et consolidation conditionnelle','Candidate adaptation and conditional consolidation'),
      'discovery':('Hypothèses, expérience et révision','Hypotheses, experiment and revision'),
      'branching':('Base partagée et modifications de branches','Shared base and branch changes'),
    }
    if key=='composition':
        body=box(15,20,62,38,'A')+box(15,80,62,38,'B')+link(77,39,138,69)+link(77,99,138,69)+box(138,45,85,48,'A ∘ B',True)+link(223,69,271,69)+box(271,45,74,48,'MACRO')
    elif key=='world':
        body=box(10,42,85,50,'S₀')+link(95,67,135,67)+box(135,42,85,50,'ACTION',True)+link(220,67,260,67)+box(260,42,90,50,'S₁')
    elif key=='planning':
        body=box(12,45,55,44,'S')+link(67,67,125,27)+link(67,67,125,107)+box(125,9,80,36,'T₁')+box(125,89,80,36,'T₂',True)+link(205,27,280,67)+link(205,107,280,67)+box(280,45,65,44,'GOAL')
    elif key=='language':
        body=box(12,45,85,44,'TEXT')+link(97,67,125,67)+box(125,45,104,44,'STRUCTURE',True)+link(229,67,257,67)+box(257,45,91,44,'CHECK')+'<path d="M302 89V117H178V89" class="diagram-link diagram-dashed"/>'
    elif key=='measurement':
        body=box(10,45,80,44,'LATENT')+link(90,67,120,67)+box(120,45,110,44,'OPERATOR',True)+link(230,67,268,27)+link(230,67,268,107)+box(268,9,80,36,'S₁')+box(268,89,80,36,'S₂')
    elif key=='sources':
        body=box(12,9,90,36,'SOURCE A')+box(12,89,90,36,'SOURCE B')+link(102,27,142,67)+link(102,107,142,67)+box(142,45,95,44,'WEIGHT',True)+link(237,67,268,67)+box(268,45,80,44,'ANCHOR')
    elif key=='certificates':
        body=box(12,45,80,44,'GEN')+link(92,67,138,27)+link(92,67,138,107)+box(138,9,68,36,'PROBE')+box(138,89,68,36,'PROBE')+link(206,27,258,67)+link(206,107,258,67)+box(258,45,90,44,'CHECK',True)
    elif key=='learning':
        body=box(12,45,65,44,'Δθ')+link(77,67,120,67)+box(120,45,90,44,'CHECK',True)+link(210,67,257,27)+link(210,67,257,107)+box(257,9,91,36,'COMMIT')+box(257,89,91,36,'ROLLBACK')
    elif key=='discovery':
        body=box(12,45,90,44,'H₁ / H₂')+link(102,67,132,67)+box(132,45,100,44,'EXPERIMENT',True)+link(232,67,258,67)+box(258,45,90,44,'UPDATE')+'<path d="M303 89V117H57V89" class="diagram-link diagram-dashed"/>'
    else:
        body=box(20,45,90,44,'BASE')+link(110,67,220,27)+link(110,67,220,107)+box(220,9,125,36,'BASE + Δ₁',True)+box(220,89,125,36,'BASE + Δ₂')
    attrs='aria-hidden="true"' if decorative else 'role="img" aria-label="'+escape(labels[key][en],quote=True)+'"'
    return f'<svg class="mechanism-svg" viewBox="0 0 360 140" {attrs}>{body}</svg>'


def render_mechanism(route,language):
    from app.research_catalog import FAMILIES
    family=next((f for f in FAMILIES if f['route']==route),None)
    if not family:return ''
    fr=language=='fr'
    _,_,chain,_=family[language]
    return '<section class="section mechanism-detail"><div><p class="eyebrow">'+('SCHÉMA DU MÉCANISME' if fr else 'MECHANISM SCHEMATIC')+'</p><h2>'+escape(chain)+'</h2><p class="note">'+('Relations conceptuelles d’après le code. Ce schéma n’est ni une trace d’exécution ni une mesure de performance.' if fr else 'Conceptual relationships based on the code. This diagram is neither an execution trace nor a performance measurement.')+'</p></div>'+diagram_svg(family['key'],language,False)+'</section>'


def render_configuration(language):
    fr=language=='fr'
    options=[('primitives','Primitives' if fr else 'Primitives'),('rotation','Rotation'),('composition','Composition')]
    buttons=''.join(f'<button type="button" data-configuration="{key}" aria-pressed="'+('true' if i==0 else 'false')+f'">{name}</button>' for i,(key,name) in enumerate(options))
    descriptions=('Trois formes indépendantes. Le repère reste fixe.','Rotation des mêmes primitives autour de leur centre.','Rotation et translation composent une nouvelle configuration.') if fr else ('Three independent shapes. The reference stays fixed.','Rotate the same primitives around their centres.','Rotation and translation compose a new configuration.')
    texts=' '.join(f'data-description-{key}="{escape(text,quote=True)}"' for (key,_),text in zip(options,descriptions))
    return f'''<div class="configuration-panel" {texts}><div class="configuration-heading"><span class="eyebrow">WCE / CONFIGURATION STUDY</span><!-- three configuration states --></div><svg class="configuration-svg" viewBox="0 0 440 300" role="img" aria-label="{'Étude illustrative de trois configurations géométriques' if fr else 'Illustrative study of three geometric configurations'}"><g class="configuration-grid"><path d="M40 30V270M100 30V270M160 30V270M220 30V270M280 30V270M340 30V270M400 30V270M20 60H420M20 120H420M20 180H420M20 240H420"/></g><g id="configuration-primitives"><rect data-primitive="0" x="0" y="0" width="64" height="64" transform="translate(65 120)"/><rect data-primitive="1" x="0" y="0" width="64" height="64" transform="translate(185 120)"/><path data-primitive="2" d="M0 64V0L64 64Z" transform="translate(305 120)"/></g><g class="configuration-anchor"><path d="M210 150H230M220 140V160"/><circle cx="220" cy="150" r="15"/></g><text x="24" y="285" class="configuration-coordinate">x / y — {'repère fixe' if fr else 'fixed reference'}</text></svg><div class="configuration-controls" role="group" aria-label="{'Choisir une configuration' if fr else 'Choose a configuration'}">{buttons}</div><p id="configuration-description" aria-live="polite">{descriptions[0]}</p><div class="configuration-readout"><span>{'Primitives' if fr else 'Primitives'} <strong>03</strong></span><span>{'État' if fr else 'State'} <strong id="configuration-state">A</strong></span><span>{'Opérateur' if fr else 'Operator'} <strong id="configuration-operator">I</strong></span></div><p class="configuration-note">{'Géométrie illustrative, sans inférence ni simulation scientifique.' if fr else 'Illustrative geometry, without inference or scientific simulation.'}</p></div>'''
