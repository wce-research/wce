# WCE — visual identity / study 01

Open `index.html` to examine the 12 concepts and the 5 finalists. The boards are also available as SVG and PNG under `boards/`.

## Intent

Condense **complexity from composition**. The shape must carry the identity without a slogan, without a mandatory colour and without a conventional AI pictogram.

The concepts are drawn in SVG from geometric primitives and operators. No brain, robot, chip, constellation, hexagon, AI star, lightning bolt or infinity symbol is used. The WCE wordmark is drawn as dedicated vector paths; it does not serve as a symbol.

## Selection

| Family | Explorations | Decision |
|---|---|---|
| A — transformation | A1 Accretion; A2 Generating fold | A1 discarded: growth chart. A2 kept: path and folds. |
| B — configurations | B1 Compatible planes; B2 Variable chamber | B1 discarded: duplication of layers. B2 kept: fixed/variable boundary. |
| C — invariant | C1 Stable core; C2 Shifted invariant | C1 discarded: autofocus/scanner. C2 kept: invariant, shift and rotation. |
| D — composition | D1 Linked cells; D2 Open joint | D1 discarded: staircase/tiling. D2 kept: modules and open joints. |
| E — world/state | E1 Discontinuous world; E2 Moving frame | Discarded: loading/viewfinder and CAD tools. These ideas remain usable in the diagrams. |
| F — boundary | F1 Open front; F2 Incision | F1 discarded: letter F or exit. F2 kept: structuring cut and compact silhouette. |

Each concept has in the page and `concepts.json`: name, symbol, principle, WCE relevance, risk of confusion and intended behaviour as favicon/GitHub/Docker/documentation/site. The risks are visual assessments; no exclusive trademark availability is asserted.

## The five finalists

- **A2 — Generating fold**: puts transformation in the foreground. Two paths issued from a band compose an asymmetric configuration. Strength: evolutive logic. Weakness: risk of monogram or angular circuit.
- **B2 — Variable chamber**: a fixed limit, a curved limit and a central primitive. Strength: distinct family, good legibility. Weakness: can be read as an operator or a letter D.
- **C2 — Shifted invariant**: two identical angles with opposite orientations and a stable central bar. Strength: direct link with invariants and transformations; three primitives only. Weakness: possible association with code brackets.
- **D2 — Open joint**: two angles and a crossbar assembled without closing all the joints. Strength: extensible composition and compact silhouette. Weakness: denser, possible industrial connotation.
- **F2 — Incision**: the void cuts a block into two compatible elements. Strength: memorable mass and counterform, robust legibility. Weakness: less direct link to the invariant and more institutional tone.

## Qualitative evaluation

These assessments serve to choose a direction to develop; they are not experimental measurements of memorability nor a guarantee of recognition in ten years. No numerical ranking.

| Criterion | A2 | B2 | C2 | D2 | F2 |
|---|---|---|---|---|---|
| Distinctiveness | Strong | Good | Good | Good | Strong |
| Simplicity | Good | Strong | Strong | Good | Strong |
| Potential memorability | Good | Good | Strong | Strong | Strong |
| Consistency with the WCE thesis | Strong | Strong | Strong | Strong | Good |
| Transformation / composition | Strong | Good | Strong | Strong | Strong |
| Frontier research potential | Strong | Good | Strong | Good | Strong |
| Timelessness | Good | Strong | Strong | Strong | Strong |
| Small size | Good | Strong | Strong | Strong | Strong |
| Open source / GitHub | Strong | Strong | Strong | Strong | Strong |
| Future hub | Good | Good | Strong | Strong | Strong |

**Recommendation: C2**, for its architectural logic as much as its shape. **Challenger: F2**, more powerful as a platform silhouette. Real recognition still has to be confronted with readers and with neighbouring brands.

## Deliverables per finalist

In `finalists/a2`, `b2`, `c2`, `d2` and `f2`:

| File / family | Use |
|---|---|
| `symbol-black`, `symbol-white` | Symbol alone, transparent, SVG and PNG |
| `logo-horizontal`, `logo-horizontal-white` | Symbol + WCE wordmark, SVG and PNG |
| `logo-stacked` | Vertical relation, SVG and PNG |
| `logo-full-name` | WCE + World Configuration Engine, vector paths |
| `favicon.svg`, `favicon.ico` | Browser; ICO containing 16, 32 and 48 px |
| `favicon-16/32/48/180/192/512.png` | Exact sizes; 180 px suited to apple-touch-icon |
| `avatar-1024`, `avatar-dark-1024`, `x-avatar-400` | Square avatars, centred symbol compatible with circular cropping |
| `x-header-1500x500` | X banner |
| `linkedin-profile-1584x396` | LinkedIn profile cover |
| `linkedin-page-1512x256` | LinkedIn Page cover according to the documentation consulted |
| `github-preview-1280x640` | GitHub repository social preview |
| `instagram-square-1080` | Square post |
| `instagram-portrait-1080x1350` | Portrait post 4:5 |
| `instagram-story-1080x1920` | Vertical composition 9:16, important elements kept away from the ends |

The banners spare the lower left corner and place the identity at the centre/right. A cropping preview remains necessary in the real account, because the platforms vary with the screen. Instagram does not offer a profile banner: the formats delivered are avatar, posts and story.

## Secondary palette

After validation of the monochrome shape: graphite `#161616`, ivory `#F4F2EC`, cobalt `#343CF2`. Colour is an accent of the editorial system, not a condition for identifying the symbol. No gradient. The initial kits remain deliberately monochrome.

## Construction / reproducibility

```bash
python3 -m pip install -r docs/identity/requirements.txt
python3 docs/identity/build.py
# Pour les exports PNG : installer playwright-core 1.63.0 dans un environnement d'outillage.
NODE_PATH=/chemin/node_modules WCE_CHROMIUM=/chemin/chromium node docs/identity/render.cjs
python3 docs/identity/package.py
```

`assets/DejaVuSans.ttf` is used for the small full-name labels, converted to paths. The licence is kept in `assets/typeface-license.txt`. The WCE wordmark and the symbols are drawn independently of this font. No font file, generation tool or private data is needed to display the exported logos.

## References verified on 3 October 2026

- [X — profile and header dimensions](https://help.x.com/en/managing-your-account/how-to-customize-your-profile): avatar 400 × 400; header 1500 × 500.
- [LinkedIn — profile cover](https://www.linkedin.com/help/linkedin/answer/a568217/?lang=en-US): 1584 × 396.
- [LinkedIn — Page images](https://www.linkedin.com/help/linkedin/answer/a563309/image-specifications-for-your-linkedin-pages-and-career-pages): the opened page indicates 1512 × 256, despite older search snippets indicating other sizes.
- [GitHub — social preview](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview): 1280 × 640 for the best display; less than 1 MB.
- The requested Instagram/Meta help pages could not be read (access limitation). The Instagram exports are usual compositions at the 1:1, 4:5 and 9:16 ratios, not a certification of the account constraints.
- Visual references examined: [OpenAI](https://openai.com/brand/), [Hugging Face](https://huggingface.co/brand), [Anthropic](https://www.anthropic.com/research), [AMI Labs](https://amilabs.xyz/). The symbols of these brands are not reused.

## Site integration

C2 is the working identity used for the header symbol and the favicons. The site serves `/favicon.svg` (automatic light/dark adaptation), `/favicon.ico`, `/apple-touch-icon.png` and `/wce-logo.svg`. The study keeps the five finalists to continue the decision; no social account was modified or published.
