# WCE — identité visuelle / étude 01

Ouvrir `index.html` pour examiner les 12 concepts et les 5 finalistes. Les planches sont également disponibles en SVG et PNG sous `boards/`.

## Intention

Condenser **complexity from composition**. La forme doit porter l'identité sans slogan, sans couleur obligatoire et sans pictogramme conventionnel d'IA.

Les concepts sont dessinés en SVG à partir de primitives et d'opérateurs géométriques. Aucun cerveau, robot, puce, constellation, hexagone, étoile AI, éclair ou symbole infini n'est utilisé. Le wordmark WCE est dessiné en tracés vectoriels dédiés ; il ne sert pas de symbole.

## Sélection

| Famille | Explorations | Décision |
|---|---|---|
| A — transformation | A1 Accrétion ; A2 Pli générateur | A1 écarté : graphique de croissance. A2 conservé : parcours et replis. |
| B — configurations | B1 Plans compatibles ; B2 Chambre variable | B1 écarté : duplication de calques. B2 conservé : frontière fixe/variable. |
| C — invariant | C1 Noyau stable ; C2 Invariant décalé | C1 écarté : autofocus/scanner. C2 conservé : invariant, décalage et rotation. |
| D — composition | D1 Cellules liées ; D2 Joint ouvert | D1 écarté : escalier/pavage. D2 conservé : modules et joints ouverts. |
| E — monde/état | E1 Monde discontinu ; E2 Repère mobile | Écartés : chargement/viseur et outils CAD. Ces idées restent utilisables dans les diagrammes. |
| F — frontière | F1 Front ouvert ; F2 Incision | F1 écarté : lettre F ou sortie. F2 conservé : découpe structurante et silhouette compacte. |

Chaque concept possède dans la page et `concepts.json` : nom, symbole, principe, pertinence WCE, risque de confusion et comportement prévu en favicon/GitHub/Docker/documentation/site. Les risques sont des appréciations visuelles ; aucune disponibilité exclusive de marque n'est affirmée.

## Les cinq finalistes

- **A2 — Pli générateur** : met la transformation au premier plan. Deux parcours issus d'une bande composent une configuration asymétrique. Force : logique évolutive. Faiblesse : risque de monogramme ou circuit angulaire.
- **B2 — Chambre variable** : une limite fixe, une limite courbe et une primitive centrale. Force : famille distincte, bonne lisibilité. Faiblesse : peut se lire comme un opérateur ou une lettre D.
- **C2 — Invariant décalé** : deux angles identiques à orientations opposées et une barre centrale stable. Force : lien direct avec invariants et transformations ; trois primitives seulement. Faiblesse : rapprochement possible avec crochets de code.
- **D2 — Joint ouvert** : deux angles et une traverse assemblés sans fermer tous les joints. Force : composition extensible et silhouette compacte. Faiblesse : plus dense, connotation industrielle possible.
- **F2 — Incision** : le vide découpe un bloc en deux éléments compatibles. Force : masse et contreforme mémorisables, lisibilité robuste. Faiblesse : lien à l'invariant moins direct et ton plus institutionnel.

## Évaluation qualitative

Ces appréciations servent à choisir une direction à développer ; elles ne sont pas des mesures expérimentales de mémorisation ni une garantie de reconnaissance dans dix ans. Pas de classement chiffré.

| Critère | A2 | B2 | C2 | D2 | F2 |
|---|---|---|---|---|---|
| Distinctivité | Forte | Bonne | Bonne | Bonne | Forte |
| Simplicité | Bonne | Forte | Forte | Bonne | Forte |
| Mémorisation potentielle | Bonne | Bonne | Forte | Forte | Forte |
| Cohérence avec la thèse WCE | Forte | Forte | Forte | Forte | Bonne |
| Transformation / composition | Forte | Bonne | Forte | Forte | Forte |
| Potentiel frontier research | Fort | Bon | Fort | Bon | Fort |
| Intemporalité | Bonne | Forte | Forte | Forte | Forte |
| Petite taille | Bonne | Forte | Forte | Forte | Forte |
| Open source / GitHub | Fort | Fort | Fort | Fort | Fort |
| Futur hub | Bon | Bon | Fort | Fort | Fort |

**Recommandation : C2**, pour sa logique architecturale autant que sa forme. **Challenger : F2**, plus puissant comme silhouette de plateforme. La reconnaissance réelle doit encore être confrontée à des lecteurs et aux marques voisines.

## Livrables par finaliste

Dans `finalists/a2`, `b2`, `c2`, `d2` et `f2` :

| Fichier / famille | Usage |
|---|---|
| `symbol-black`, `symbol-white` | Symbole seul, transparent, SVG et PNG |
| `logo-horizontal`, `logo-horizontal-white` | Symbole + wordmark WCE, SVG et PNG |
| `logo-stacked` | Relation verticale, SVG et PNG |
| `logo-full-name` | WCE + World Configuration Engine, tracés vectoriels |
| `favicon.svg`, `favicon.ico` | Navigateur ; ICO contenant 16, 32 et 48 px |
| `favicon-16/32/48/180/192/512.png` | Tailles exactes ; 180 px adapté à apple-touch-icon |
| `avatar-1024`, `avatar-dark-1024`, `x-avatar-400` | Avatars carrés, symbole centré compatible recadrage circulaire |
| `x-header-1500x500` | Bannière X |
| `linkedin-profile-1584x396` | Couverture de profil LinkedIn |
| `linkedin-page-1512x256` | Couverture de Page LinkedIn selon la documentation consultée |
| `github-preview-1280x640` | Aperçu social de dépôt GitHub |
| `instagram-square-1080` | Publication carrée |
| `instagram-portrait-1080x1350` | Publication portrait 4:5 |
| `instagram-story-1080x1920` | Composition verticale 9:16, éléments importants éloignés des extrémités |

Les bannières ménagent l'angle inférieur gauche et placent l'identité au centre/droite. Un aperçu de recadrage reste nécessaire dans le compte réel, car les plateformes varient selon l'écran. Instagram ne propose pas de bannière de profil : les formats livrés sont avatar, publications et story.

## Palette secondaire

Après validation de la forme monochrome : graphite `#161616`, ivoire `#F4F2EC`, cobalt `#343CF2`. La couleur est un accent du système éditorial, pas une condition d'identification du symbole. Pas de gradient. Les kits initiaux restent volontairement monochromes.

## Construction / reproductibilité

```bash
python3 -m pip install -r docs/identity/requirements.txt
python3 docs/identity/build.py
# Pour les exports PNG : installer playwright-core 1.63.0 dans un environnement d'outillage.
NODE_PATH=/chemin/node_modules WCE_CHROMIUM=/chemin/chromium node docs/identity/render.cjs
python3 docs/identity/package.py
```

`assets/DejaVuSans.ttf` est utilisé pour les petits labels de nom complet, convertis en tracés. La licence est conservée dans `assets/typeface-license.txt`. Le wordmark WCE et les symboles sont dessinés indépendamment de cette police. Aucun fichier de police, outil de génération ni donnée privée n'est nécessaire pour afficher les logos exportés.

## Références vérifiées le 3 octobre 2026

- [X — dimensions de profil et header](https://help.x.com/en/managing-your-account/how-to-customize-your-profile) : avatar 400 × 400 ; header 1500 × 500.
- [LinkedIn — couverture de profil](https://www.linkedin.com/help/linkedin/answer/a568217/?lang=en-US) : 1584 × 396.
- [LinkedIn — images des Pages](https://www.linkedin.com/help/linkedin/answer/a563309/image-specifications-for-your-linkedin-pages-and-career-pages) : la page ouverte indique 1512 × 256, malgré des extraits de recherche plus anciens indiquant d'autres tailles.
- [GitHub — aperçu social](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview) : 1280 × 640 pour le meilleur affichage ; moins de 1 Mo.
- Les aides Instagram/Meta demandées n'ont pas été lisibles (limitation d'accès). Les exports Instagram sont des compositions usuelles aux ratios 1:1, 4:5 et 9:16, pas une certification des contraintes du compte.
- Références visuelles examinées : [OpenAI](https://openai.com/brand/), [Hugging Face](https://huggingface.co/brand), [Anthropic](https://www.anthropic.com/research), [AMI Labs](https://amilabs.xyz/). Les symboles de ces marques ne sont pas réutilisés.

## Intégration au site

C2 est l’identité de travail utilisée pour le symbole du header et les favicons. Le site sert `/favicon.svg` (adaptation automatique clair/sombre), `/favicon.ico`, `/apple-touch-icon.png` et `/wce-logo.svg`. L’étude conserve les cinq finalistes pour poursuivre la décision ; aucun compte social n’a été modifié ou publié.
