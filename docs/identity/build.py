"""Reproducible, monochrome SVG identity exploration for WCE (SVG sources plus FontTools for outlined type)."""
from pathlib import Path
import json
from html import escape

ROOT = Path(__file__).resolve().parent
INK = '#161616'
WORDMARK = '<path d="M0 0h5l5 22 6-16h5l6 16 5-22h5l-8 30h-5l-6-17-6 17H8Z M75 0H60C51 0 46 6 46 15s5 15 14 15h15v-5H60c-6 0-9-4-9-10s3-10 9-10h15Z M84 0h22v5H89v7h15v5H89v8h17v5H84Z"/>'

def item(id,name,direction,shape,principle,fit,risk,use,keep=False,weakness=''):
    return dict(id=id,name=name,direction=direction,shape=shape,principle=principle,fit=fit,risk=risk,use=use,keep=keep,weakness=weakness)
CONCEPTS = [
 item('a1','Accrétion','A — Primitive → structure','<path d="M8 40h8v16H8zM24 24h8v32h-8zM40 8h16v48H40z"/>','La répétition et la croissance d’une primitive rectangulaire produisent une structure.','Une grammaire élémentaire immédiatement visible.','Évoque un graphique de croissance ou un égaliseur ; peu propriétaire.','Favicon 16/32 px et avatars lisibles ; Docker, documents et site faciles, mais confusion avec un pictogramme de statistiques.'),
 item('a2','Pli générateur','A — Primitive → structure','<path d="M8 8h32v8H16v16h16v16h24v8H24V40H8Z"/><path d="M40 16h8v16h8v8H40Z"/>','Deux parcours issus d’une même bande se replient et créent une configuration asymétrique.','Primitives identiques, changements de direction et composition ; aucun effet ajouté.','Peut rappeler un monogramme angulaire ou un circuit imprimé très simplifié.','16 px : coudes robustes ; 32 px : deux parcours perceptibles. Avatar, favicon, package Docker et en-tête scientifique sobres.',True,'La lecture du pli est moins immédiate que celle de l’assemblage ; proximité possible avec un signe typographique.'),
 item('b1','Plans compatibles','B — Espace des configurations','<path fill-rule="evenodd" d="M8 8h32v32H8ZM16 16v16h16V16Z M24 24h32v32H24ZM32 32v16h16V32Z"/>','Deux configurations décalées partagent une zone compatible.','Un même espace autorise plusieurs états.','Très proche des icônes de duplication, calques ou fenêtres.','16 px : intersection peu claire ; 32 px acceptable. Avatars, Docker et documentation fonctionnels mais caractère utilitaire dominant.'),
 item('b2','Chambre variable','B — Espace des configurations','<path d="M8 8h20v8H16v32h12v8H8Z"/><path d="M36 8c14 0 20 9 20 24S50 56 36 56v-8c8 0 12-6 12-16s-4-16-12-16Z"/><path d="M28 24h8v16h-8Z"/>','Une limite fixe et une limite déformable encadrent le même état.','Plusieurs géométries compatibles entourent une primitive commune.','Peut être lu comme une parenthèse, un D abstrait ou un opérateur mathématique.','16 px : contraste droite/courbe visible ; 32 px : état central clair. Bon en avatar rond, vignette Docker, favicon, équations et header de hub.',True,'L’espace des configurations reste abstrait ; la courbe donne une personnalité moins radicale que les signes orthogonaux.'),
 item('c1','Noyau stable','C — Invariant au milieu du changement','<path d="M8 8h20v8H16v12H8ZM56 8v20h-8V16H36V8ZM56 56H36v-8h12V36h8ZM8 56V36h8v12h12v8Z"/><path d="M26 26h12v12H26Z"/>','Des fragments de cadre entourent un noyau inchangé.','L’invariant est indépendant de l’enveloppe.','Cadrage photographique, autofocus ou scanner : association trop immédiate.','16 px : détail central chargé ; 32 px correct. Avatar et favicon utilisables, mais lecture « scan » gênante sur GitHub, Docker et site scientifique.'),
 item('c2','Invariant décalé','C — Invariant au milieu du changement','<path d="M8 8h32v8H16v24H8Z"/><path d="M56 56H24v-8h24V24h8Z"/><path d="M28 24h8v16h-8Z"/>','Deux angles identiques, tournés et décalés, changent l’enveloppe autour d’une barre stable.','Trois primitives seulement : une relation stable rend plusieurs configurations possibles.','Évoque des crochets de code ou un cadre de focus ; le décalage et la barre doivent rester caractéristiques.','16 px : barre de 2 px et ouvertures nettes ; 32 px : invariant identifiable. Compact pour favicon, avatar GitHub, Docker, documentation et futur hub.',True,'Le changement est suggéré par les deux orientations, pas animé ; le risque « brackets » reste à examiner.'),
 item('d1','Cellules liées','D — Composition','<path d="M8 8h24v8H16v16H8ZM24 24h24v8H32v16h-8ZM40 40h16v16H40Z"/>','Une primitive en angle est répétée puis prolongée.','Le système peut croître par ajout de modules.','Escalier, tuiles de logiciel ou motif de pavage ; silhouette peu mémorisable.','16 px clair mais générique ; 32 px révèle les modules. Avatars et Docker possibles, documents/site manquent de singularité.'),
 item('d2','Joint ouvert','D — Composition','<path d="M8 8h16v16h16v16H8Z"/><path d="M48 24h8v32H24v-8h24Z"/><path d="M32 8h24v8H32Z"/>','Deux modules en angle et une traverse constituent un ensemble sans fermer ses joints.','Une composition finie reste extensible ; l’ouverture est une propriété de construction.','Parenté possible avec des marques de fabrication ou de construction modulaire ; la silhouette dissymétrique évite le simple cadre.','16 px : trois masses et des joints de 2 px ; 32 px : assemblage plus évident. Excellent avatar rond, package Docker et favicon ; lisible en légende scientifique.',True,'Plus dense que C2 ; la logique des joints peut se lire comme un signe industriel plutôt que comme une transformation.'),
 item('e1','Monde discontinu','E — World Configuration Engine','<path d="M28 8A24 24 0 0 0 8 28h8a16 16 0 0 1 12-12ZM36 8v8a16 16 0 0 1 12 12h8A24 24 0 0 0 36 8ZM56 36h-8a16 16 0 0 1-12 12v8A24 24 0 0 0 56 36ZM28 56v-8a16 16 0 0 1-12-12H8a24 24 0 0 0 20 20Z"/>','Un domaine est défini par quatre frontières indépendantes.','État du monde incomplet et architecture ouverte.','Roue de chargement, viseur ou globe abstrait : trop courant.','16/32 px faciles ; favicon et avatar corrects. Docker/docs/site : connotation chargement dominante.'),
 item('e2','Repère mobile','E — World Configuration Engine','<path d="M8 8h8v40h40v8H8Z"/><path d="M24 8h32v32h-8V16H24Z"/><path d="m24 24 8-8 16 16-8 8Z"/>','Deux repères orthogonaux et un plan oblique décrivent un état depuis plusieurs projections.','Le monde est une représentation configurable plutôt qu’une sphère.','Icône de transformation 3D, recadrage ou outil CAD.','16 px : diagonale moins nette ; 32 px utile. Plus convaincant dans documentation technique que comme avatar GitHub ou marque Docker.'),
 item('f1','Front ouvert','F — Nouvelle frontière','<path d="M8 8h48v8H16v32h16v8H8Z"/><path d="M24 24h32v8H24Z"/><path d="M40 40h16v16h-8v-8h-8Z"/>','Une enceinte laisse une trajectoire ouverte vers un autre espace.','Une école architecturale doit pouvoir prolonger ses propres limites.','F ou G stylisé, pictogramme de sortie ou cadre ouvert.','16 px correct, 32 px net. Favicon, Docker, documents et hub fonctionnels mais monogramme involontaire.'),
 item('f2','Incision','F — Nouvelle frontière','<path d="M8 8h48v16H40L24 40v16H8Z"/><path d="M56 32v24H32V40l8-8Z"/>','Une découpe commune transforme un bloc unique en deux primitives compatibles.','La frontière crée des composants réassemblables ; le vide est opérateur.','Carré entaillé, tuile industrielle ou monogramme abstrait ; parenté générale avec marques géométriques et signes de partition.','16 px : deux masses très lisibles ; 32 px : découpe diagonale précise. Fort en favicon, avatar, image Docker et marque de plateforme.',True,'Moins explicite sur l’invariant ; la masse pleine peut sembler plus institutionnelle que laboratoire ouvert.'),
]
# Five readings: folded paths, curved space, invariant, modular joints, negative-space incision.
FINALISTS = [c for c in CONCEPTS if c['keep']]

def svg(body,w=64,h=64,title='WCE',bg=None,color=INK):
    rect=f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ''
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>{escape(title)}</title>{rect}<g fill="{color}">{body}</g></svg>'

def symbol(c,size=64,color=INK):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 64 64" role="img" aria-label="{escape(c["name"])}"><g fill="{color}">{c["shape"]}</g></svg>'

def lockup(c,long=False,stack=False,color=INK):
    if stack:
        return svg(f'<g transform="translate(52 0)">{c["shape"]}</g><g transform="translate(30 88)">{WORDMARK}</g>',166,124,c['name']+' — WCE empilé',color=color)
    if long:
        # Exact full name, outlined using an installed open font by outline_text().
        body=c['shape']+f'<g transform="translate(84 7)">{WORDMARK}</g>'+f'<g transform="translate(84 56)">{outline_text("World Configuration Engine",12)}</g>'
        return svg(body,258,64,c['name']+' — World Configuration Engine',color=color)
    return svg(c['shape']+f'<g transform="translate(84 17)">{WORDMARK}</g>',194,64,c['name']+' — WCE',color=color)

# Font is only used for long descriptive labels, not for the custom WCE wordmark.
def outline_text(text,size):
    from fontTools.ttLib import TTFont
    from fontTools.pens.svgPathPen import SVGPathPen
    font=TTFont(ROOT / 'assets/DejaVuSans.ttf')
    glyphs=font.getGlyphSet(); cmap=font.getBestCmap(); upm=font['head'].unitsPerEm
    x=0; paths=[]
    for character in text:
        name=cmap[ord(character)]; pen=SVGPathPen(glyphs); glyphs[name].draw(pen)
        paths.append(f'<path transform="translate({x} 0)" d="{pen.getCommands()}"/>')
        x+=font['hmtx'][name][0]
    return f'<g transform="scale({size/upm} {-size/upm})">'+''.join(paths)+'</g>'


def avatar(c,size=1024,bg='#fff',color=INK):
    margin=size*.20; scale=size*.6/64
    return svg(f'<g transform="translate({margin} {margin}) scale({scale})">{c["shape"]}</g>',size,size,c['name']+' — avatar',bg,color)


def banner(c,w,h,kind):
    # Typography sits in the centre/right so the lower-left avatar does not cover it.
    sym_size=min(h*.40,210); scale=sym_size/64
    x=w*.40; y=h*.5-sym_size*.5
    wm_scale=min((w*.25)/106, sym_size*.58/30)
    wx=x+sym_size+42; wy=h*.5-30*wm_scale*.65
    body=f'<g transform="translate({x} {y}) scale({scale})">{c["shape"]}</g><g transform="translate({wx} {wy}) scale({wm_scale})">{WORDMARK}</g>'
    font_size=min(19,h*.065)
    body+=f'<g transform="translate({wx} {h*.5+sym_size*.34})">{outline_text("World Configuration Engine",font_size)}</g>'
    # Quiet primitive on left, separated from main mark; no claim or slogan needed.
    body+=f'<path d="M{w*.08} {h*.3}h{h*.1}v{h*.1}h-{h*.1}Z" fill="none" stroke="{INK}" stroke-width="1"/>'
    return svg(body,w,h,c['name']+' — '+kind,'#fff')


def instagram(c,w,h,kind):
    size=w*.40; x=(w-size)/2; y=h*.32-size/2
    wm_scale=w*.33/106
    body=f'<g transform="translate({x} {y}) scale({size/64})">{c["shape"]}</g><g transform="translate({w*.335} {h*.59}) scale({wm_scale})">{WORDMARK}</g>'
    body+=f'<g transform="translate({w*.22} {h*.72})">{outline_text("World Configuration Engine",w*.038)}</g>'
    return svg(body,w,h,c['name']+' — '+kind,'#fff')


def build():
    for folder in ('concepts','finalists','boards'):(ROOT/folder).mkdir(exist_ok=True)
    for c in CONCEPTS:(ROOT/'concepts'/f'{c["id"]}.svg').write_text(svg(c['shape'],title=c['name']))
    for c in FINALISTS:
        folder=ROOT/'finalists'/c['id'];folder.mkdir(exist_ok=True)
        exports={
            'symbol-black.svg':svg(c['shape'],title=c['name']),
            'symbol-white.svg':svg(c['shape'],title=c['name'],color='#fff'),
            'logo-horizontal.svg':lockup(c),
            'logo-horizontal-white.svg':lockup(c,color='#fff'),
            'logo-stacked.svg':lockup(c,stack=True),
            'logo-full-name.svg':lockup(c,long=True),
            'avatar-1024.svg':avatar(c),
            'avatar-dark-1024.svg':avatar(c,bg=INK,color='#fff'),
            'x-avatar-400.svg':avatar(c,size=400),
            'favicon.svg':svg(c['shape'],title='WCE',bg='#fff'),
            'x-header-1500x500.svg':banner(c,1500,500,'X header'),
            'linkedin-profile-1584x396.svg':banner(c,1584,396,'LinkedIn profile'),
            'linkedin-page-1512x256.svg':banner(c,1512,256,'LinkedIn page'),
            'github-preview-1280x640.svg':banner(c,1280,640,'GitHub social preview'),
            'instagram-square-1080.svg':instagram(c,1080,1080,'Instagram square'),
            'instagram-portrait-1080x1350.svg':instagram(c,1080,1350,'Instagram portrait'),
            'instagram-story-1080x1920.svg':instagram(c,1080,1920,'Instagram story'),
        }
        for name,content in exports.items():(folder/name).write_text(content)
    # A 12-symbol visual overview, deliberately monochrome.
    body='<text x="48" y="60" font-family="Arial" font-size="30" letter-spacing="-1">WCE / 12 hypothèses de forme</text><text x="48" y="91" font-family="monospace" font-size="11">MONOCHROME · SIX DIRECTIONS · UNE LOGIQUE PAR SYMBOLE</text>'
    for i,c in enumerate(CONCEPTS):
        col=i%4;row=i//4;x=48+col*288;y=135+row*260
        body+=f'<path d="M{x} {y}h250" stroke="#d2d2d2" fill="none"/><text x="{x}" y="{y+25}" font-family="monospace" font-size="11">{c["id"].upper()} / {"FINALISTE" if c["keep"] else "ÉCARTÉ"}</text><g transform="translate({x+75} {y+52}) scale(1.5)">{c["shape"]}</g><text x="{x}" y="{y+183}" font-family="Arial" font-size="19">{escape(c["name"])}</text><text x="{x}" y="{y+207}" font-family="monospace" font-size="10">{escape(c["direction"])}</text>'
    (ROOT/'boards'/'initial-12.svg').write_text(svg(body,1248,950,'WCE — 12 concepts','#fff'))
    # Each row proves identical geometry across scale and contrast.
    body='<text x="48" y="57" font-family="Arial" font-size="30">WCE / les cinq finalistes</text><text x="48" y="85" font-family="monospace" font-size="11">LA FORME AVANT LA COULEUR · ÉCHELLES RÉELLES À 100 %</text>'
    for i,c in enumerate(FINALISTS):
        y=120+i*235
        body+=f'<path d="M48 {y}h1152" fill="none" stroke="#ccc"/><text x="48" y="{y+30}" font-family="Arial" font-size="18">{c["id"].upper()} / {escape(c["name"])}</text><g transform="translate(48 {y+56}) scale(1.9)">{c["shape"]}</g>'
        body+=f'<g transform="translate(240 {y+74})">{c["shape"]}<g transform="translate(84 17)">{WORDMARK}</g></g>'
        body+=f'<rect x="502" y="{y+49}" width="260" height="144" fill="{INK}"/><g fill="#fff" transform="translate(527 {y+87})">{c["shape"]}<g transform="translate(84 17)">{WORDMARK}</g></g>'
        body+=f'<g transform="translate(810 {y+106}) scale(.25)">{c["shape"]}</g><g transform="translate(875 {y+98}) scale(.5)">{c["shape"]}</g><circle cx="1050" cy="{y+115}" r="48" fill="{INK}"/><g fill="#fff" transform="translate(1018 {y+83})">{c["shape"]}</g>'
        for xx,label in [(48,'GRAND FORMAT / 122 PX'),(240,'SYMBOLE + WCE'),(502,'BLANC SUR NOIR'),(805,'16 PX'),(869,'32 PX'),(1004,'AVATAR ROND')]:
            body+=f'<text x="{xx}" y="{y+216}" font-family="monospace" font-size="9">{label}</text>'
    (ROOT/'boards'/'finalists-5.svg').write_text(svg(body,1248,1320,'WCE — 5 finalistes','#fff'))
    (ROOT/'concepts.json').write_text(json.dumps(CONCEPTS,ensure_ascii=False,indent=2))
    build_html()


def build_html():
    cards=''
    for c in CONCEPTS:
        cards+=f'''<article class="concept"><div class="meta">{c['id'].upper()} / {c['direction']} <span>{'Finaliste' if c['keep'] else 'Écarté'}</span></div><div class="symbol">{symbol(c,120)}</div><h3>{c['name']}</h3><p>{c['principle']}</p><details><summary>Logique, risques et usages</summary><p><b>Pourquoi WCE.</b> {c['fit']}</p><p><b>Confusion possible.</b> {c['risk']}</p><p><b>Usages.</b> {c['use']}</p></details></article>'''
    finals=''
    assessments={
      'a2':['Fort','Bon','Bon','Fort','Fort','Fort','Bon','Bon','Fort','Bon'],
      'b2':['Bon','Fort','Bon','Fort','Bon','Bon','Fort','Fort','Fort','Bon'],
      'c2':['Bon','Fort','Fort','Fort','Fort','Fort','Fort','Fort','Fort','Fort'],
      'd2':['Bon','Bon','Fort','Fort','Fort','Bon','Fort','Fort','Fort','Fort'],
      'f2':['Fort','Fort','Fort','Bon','Fort','Fort','Fort','Fort','Fort','Fort'],
    }
    criteria=['Distinctivité','Simplicité','Mémorisation','Thèse WCE','Transformation / composition','Frontier research','Intemporalité','Petite taille','Open source / GitHub','Futur hub']
    for c in FINALISTS:
        folder='finalists/'+c['id']
        finals+=f'''<section class="finalist" id="{c['id']}"><div class="final-head"><span>{c['id'].upper()} / {c['direction']}</span><h2>{c['name']}</h2><a href="{folder}/logo-horizontal.svg" download>SVG du logo ↗</a></div><div class="proof-grid"><div class="proof"><span>Symbole seul</span>{symbol(c,160)}</div><div class="proof"><span>Logo horizontal</span>{lockup(c)}</div><div class="proof dark"><span>Monochrome sombre</span>{lockup(c,color='#fff')}</div><div class="proof"><span>Nom complet</span>{lockup(c,long=True)}</div><div class="proof"><span>Relation empilée</span>{lockup(c,stack=True)}</div><div class="proof"><span>Favicon / 16 px / 32 px / avatar</span><div class="sizes">{symbol(c,16)}{symbol(c,32)}<span class="circle">{symbol(c,64,color='#fff')}</span></div></div></div><div class="argument"><div><h3>Ce qui tient.</h3><p>{c['principle']} {c['fit']}</p></div><div><h3>Ce qui résiste.</h3><p>{c['weakness']}</p><p>{c['risk']}</p></div></div><div class="evaluation">{''.join(f'<div><span>{label}</span><b>{value}</b></div>' for label,value in zip(criteria,assessments[c['id']]))}</div><details class="social-preview"><summary>Déclinaisons sociales et fichiers prêts à utiliser</summary><div class="social-gallery"><figure><img src="{folder}/x-header-1500x500.svg" alt="Bannière X"><figcaption>X / 1500 × 500</figcaption></figure><figure><img src="{folder}/linkedin-profile-1584x396.svg" alt="Bannière LinkedIn"><figcaption>LinkedIn profil / 1584 × 396</figcaption></figure><figure><img src="{folder}/github-preview-1280x640.svg" alt="Aperçu GitHub"><figcaption>GitHub / 1280 × 640</figcaption></figure><figure class="square"><img src="{folder}/instagram-square-1080.svg" alt="Publication Instagram"><figcaption>Instagram / 1080 × 1080</figcaption></figure></div><p>Dans <code>{folder}/</code> : SVG, PNG, ICO multi-résolution, avatars clairs/sombres, formats portrait et story, LinkedIn Page et nom complet.</p></details></section>'''
    html='''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WCE — Identity research</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5f2;color:#161616;font:15px Arial,sans-serif}a{color:inherit;text-underline-offset:4px}header{padding:30px 5%;border-bottom:1px solid #ccc;display:flex;justify-content:space-between;font:11px monospace;letter-spacing:1px}main{padding:0 5%}h1,h2{font-weight:400;letter-spacing:-2px}h1{font-size:clamp(46px,6vw,86px);line-height:1.05;max-width:900px;margin:50px 0 25px}h1 em{font-family:Georgia,serif}h2{font-size:38px}h3{font-size:22px;font-weight:400;letter-spacing:-.5px}p{line-height:1.7}small,.label{font:11px monospace}.intro{max-width:700px;padding-bottom:50px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;border-top:1px solid #ccc;padding:20px 0 40px}.toolbar a{background:#161616;color:#fff;padding:12px 16px;text-decoration:none;font-size:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}.concept{padding:24px;background:#fff;border:1px solid #ddd}.meta{font:10px monospace;line-height:1.7;display:flex;justify-content:space-between;gap:10px}.meta span{white-space:nowrap}.symbol{height:210px;display:grid;place-items:center}.concept p{font-size:13px}details{border-top:1px solid #ddd;padding-top:15px}summary{cursor:pointer;font-size:12px}details p{font-size:13px}.section-title{margin:60px 0 25px}.finalist{border-top:1px solid #aaa;padding-top:40px;margin-top:45px}.final-head{display:flex;align-items:baseline;gap:35px;flex-wrap:wrap}.final-head>span{font:10px monospace;letter-spacing:1px}.final-head a{margin-left:auto;font-size:12px}.proof-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.proof{height:230px;background:#fff;display:flex;align-items:center;justify-content:center;position:relative;padding:35px 20px}.proof>span:first-child{position:absolute;top:20px;left:20px;font:10px monospace}.proof>svg{max-width:100%;height:auto}.proof.dark{background:#161616;color:#fff}.sizes{display:flex;align-items:center;gap:28px}.circle{background:#161616;display:grid;place-items:center;width:100px;height:100px;border-radius:50%}.argument{display:grid;grid-template-columns:1fr 1fr;gap:50px;margin:25px 0}.argument p{font-size:14px}.evaluation{display:grid;grid-template-columns:repeat(5,1fr);gap:0 20px;margin-bottom:28px}.evaluation div{padding:15px 0;border-top:1px solid #ccc}.evaluation span,.evaluation b{display:block;font-size:11px;line-height:1.6}.evaluation b{font-weight:400;margin-top:10px;font-family:monospace}.social-gallery{display:grid;grid-template-columns:1fr 1fr;gap:25px;margin-top:30px}.social-gallery figure{margin:0}.social-gallery img{width:100%;border:1px solid #ccc}.social-gallery .square img{width:60%}.social-gallery figcaption{font:10px monospace;margin-top:10px}.closing{margin-top:65px;padding:40px 0;border-top:1px solid #aaa;max-width:850px}.palette{display:flex;gap:20px;margin:25px 0}.swatch{padding:25px;min-width:150px;font:11px monospace;border:1px solid #ccc}.page-footer{padding:25px 5%;border-top:1px solid #ccc;font:11px monospace}@media(max-width:900px){.grid,.proof-grid{grid-template-columns:1fr 1fr}.evaluation{grid-template-columns:repeat(2,1fr)}}@media(max-width:600px){header{gap:20px}.grid,.proof-grid,.argument,.social-gallery{grid-template-columns:1fr}.final-head{gap:10px}.final-head a{margin-left:0}.palette{flex-wrap:wrap}.concept{padding:22px}.argument{gap:10px}.sizes{gap:20px}.intro{padding-bottom:25px}}@media print{details{display:block}.toolbar{display:none}.finalist{break-before:page}.proof{break-inside:avoid}}
</style></head><body><header><span>WCE / IDENTITY RESEARCH</span><span>ÉTUDE 01 · MONOCHROME</span></header><main><h1>Small primitives.<br><em>A recognizable world.</em></h1><div class="intro"><p>Douze hypothèses graphiques, six familles, cinq finalistes. Le symbole condense une logique : quelques primitives et leurs relations peuvent engendrer une structure.</p><p>Aucun slogan n’est nécessaire au logo. Chaque proposition est d’abord évaluée en noir et blanc et aux tailles réelles. Les appréciations sont qualitatives, pas des mesures de reconnaissance auprès d’un public.</p></div><div class="toolbar"><a href="#concepts">Les 12 concepts</a><a href="#finalists">Les 5 finalistes</a><a href="boards/initial-12.svg">Planche initiale SVG</a><a href="boards/finalists-5.svg">Planche finalistes SVG</a><a href="README.md">Dossier et critères</a><a href="wce-identity-study-01.zip" download>Télécharger tous les kits</a></div><h2 id="concepts">01 / Exploration</h2><div class="grid">'''+cards+'''</div><h2 class="section-title" id="finalists">02 / Les cinq propositions retenues</h2>'''+finals+'''<section class="closing"><h2>03 / Recommandation de travail</h2><p><b>C2 — Invariant décalé</b> est la piste conceptuelle la plus cohérente : trois primitives, un invariant visible et deux orientations de la même forme. <b>F2 — Incision</b> est le meilleur challenger pour une marque de plateforme : silhouette plus mémorisable, lien à l’invariance moins direct.</p><p>D2 privilégie des joints ouverts entre trois modules, sans invariant central. A2 est plus transformationnel ; B2 apporte une famille courbe distincte. Aucun gagnant n’est imposé ni installé comme identité définitive.</p><p>La disponibilité visuelle n’est pas une preuve d’exclusivité : cette étude relève les risques de confusion et ne remplace pas une recherche d’antériorité.</p><h3>Après la forme : une palette secondaire.</h3><div class="palette"><div class="swatch" style="background:#161616;color:white">Graphite<br>#161616</div><div class="swatch" style="background:#f4f2ec">Ivoire<br>#F4F2EC</div><div class="swatch" style="background:#343cf2;color:white">Cobalt<br>#343CF2</div></div><p>La couleur accompagne le site ; le symbole principal reste monochrome. Pas de gradient, pas de version dépendante d’une couleur pour être identifiable.</p></section></main><footer class="page-footer">WCE — World Configuration Engine / Étude de direction, pas identité définitive.</footer></body></html>'''
    (ROOT/'index.html').write_text(html)

if __name__=='__main__':build()
