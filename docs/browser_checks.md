# Contrôles navigateur WCE

Les tests ciblés Python sont lancés avec `python3 -m unittest tests.test_web tests.test_wce`. Ils vérifient notamment les valeurs et effectifs affichés face aux JSON servis, ainsi que la mise à jour du rendu lorsque les valeurs d’un artefact changent.

Le script `tests/browser/frontier.cjs` utilise Playwright Core 1.63.0 et Chromium. Il ne fait pas partie de l’image Docker. Exemple exécuté depuis le GEEKOM avec la connexion SSH existante, sans nouveau port ni forwarding TCP :

```bash
npm install --prefix /tmp/wce-browser --no-audit --no-fund playwright-core@1.63.0
NODE_PATH=/tmp/wce-browser/node_modules \
WCE_BASE_URL=http://127.0.0.1:3018 \
WCE_SSH_TARGET=<OPERATOR_PROMPT> \
WCE_SSH_IDENTITY=<USER_HOME>/.ssh/geekom_to_dell_ed25519 \
WCE_CHROMIUM=<USER_HOME>/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome \
WCE_BROWSER_ARTIFACTS=/tmp/wce-frontier-dell \
node tests/browser/frontier.cjs
```

Le chemin Chromium correspond à l’installation présente sur le GEEKOM. Pour un test directement sur le Dell, omettre les variables `WCE_SSH_*` et utiliser un Chromium installé sur cette machine. Le relais SSH interroge seulement le serveur HTTP existant du Dell ; il ne lance aucun serveur ni stack sur le GEEKOM et ne modifie pas sa configuration SSH.

Dans l’environnement d’exécution isolé de l’agent, `WCE_BROWSER_NO_SANDBOX=1` a été nécessaire pour Chromium de test. Cette option est réservée à ce processus de vérification ; elle ne modifie aucune protection du conteneur web.

Contrôles : 36 rendus bilingues aux largeurs 1440, 768 et 390 px ; transformations et repère de l’illustration ; neuf métriques et neuf méthodes ; tableaux et intervalles ; dix hypothèses ; cinq groupes de sources ; modes jour/nuit ; dix schémas ; absence de débordement horizontal et d’erreurs JavaScript.
