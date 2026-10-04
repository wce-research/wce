"""Banc de discours « tiers » : écrit séparément du résolveur, avec ses PROPRES conventions.

Ce module n'importe AUCUN composant de WCE (ni analyseur, ni résolveur, ni grammaire, ni planificateur) :
bibliothèque standard seulement (vérifié par test AST). Il décrit des scénarios dans son propre format :

    objets   : {"A": {"sorte": "cube", "teinte": "rouge", "lieu": "table"}, ...}      (vérité, jamais montrée)
    capteurs : faits rapportés par le capteur au début (liste de (objet, champ))
    historique : bilan passé des locuteurs {"bob": (justes, fausses)} — une donnée d'entrée, pas la vérité
    tours    : ("dit", locuteur, phrase) | ("capteur", objet, champ, valeur) | ("demande", phrase, attendu)

Attendus (conventions tierces) : un mot de lieu ("table", "étagère", "sol", "boîte"), une teinte ("rouge"…),
"oui" / "non", "agir:A" (commande sur l'objet A), "?" (il faut demander ou s'abstenir),
"présupposé" (la description ne désigne rien), "conflit" (sources contradictoires, pas d'engagement).

Deux parties : (1) scénarios écrits à la main (adversariaux, 12 catégories) ; (2) un second générateur,
indépendant des gabarits de reference_world, qui fait varier l'ordre, la redondance, les omissions,
les reformulations, les retours à un ancien référent et les interruptions.
"""
from __future__ import annotations

import random

NON_ENGAGEMENT = ("?", "présupposé", "conflit")
LIEUX = ("table", "étagère", "sol")
TEINTES = ("rouge", "bleu", "vert", "jaune")
FEMININ = {"balle": True, "boîte": True, "clé": True, "cube": False, "livre": False}
ACCORD = {("rouge", True): "rouge", ("bleu", True): "bleue", ("vert", True): "verte", ("jaune", True): "jaune",
          ("rouge", False): "rouge", ("bleu", False): "bleu", ("vert", False): "vert", ("jaune", False): "jaune"}
ARTICLE_LIEU = {"table": "la table", "étagère": "l'étagère", "sol": "le sol", "boîte": "la boîte"}


def S(id, cat, objets, capteurs, tours, historique=None, note=""):
    return {"id": id, "catégorie": cat, "objets": objets, "capteurs": capteurs, "tours": tours,
            "historique": historique or {}, "note": note, "origine": "main"}


def O(sorte, teinte, lieu): return {"sorte": sorte, "teinte": teinte, "lieu": lieu}


# ================================================================ (1) scénarios écrits à la main

def scenarios_main():
    X = []
    fiable = {"alice": (30, 1), "carla": (30, 1)}
    douteux = {"alice": (30, 1), "bob": (2, 8)}
    # --- référents implicites ---
    X.append(S("m01", "implicite", {"A": O("cube", "rouge", "table"), "B": O("balle", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "Un cube est sur la table."), ("demande", "Où est le cube ?", "table"),
                ("demande", "Où est la balle ?", "?")], fiable))
    X.append(S("m02", "implicite", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("B", "teinte")],
               [("dit", "alice", "Le cube rouge est sur la table."), ("demande", "Où est le cube rouge ?", "table"),
                ("demande", "Où est le cube bleu ?", "?")], fiable))
    # --- descriptions incomplètes ---
    X.append(S("m03", "incomplète", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "étagère")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "Le cube est sur la table."), ("demande", "Où est le cube rouge ?", "?"),
                ("demande", "Prends le cube.", "?")], fiable))
    X.append(S("m04", "incomplète", {"A": O("livre", "vert", "étagère"), "B": O("livre", "jaune", "table")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("demande", "Prends le livre.", "?"), ("demande", "Prends le livre vert.", "agir:A")], fiable))
    # --- descriptions contradictoires ---
    X.append(S("m05", "contradictoire", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "alice", "Le cube rouge est bleu."), ("demande", "De quelle couleur est le cube rouge ?", "rouge")], fiable,
               "affirmation incohérente avec les capteurs : ne change pas une teinte observée"))
    X.append(S("m06", "contradictoire", {"A": O("balle", "vert", "table")}, [("A", "sorte")],
               [("dit", "alice", "La balle est verte."), ("dit", "carla", "La balle est rouge."),
                ("demande", "De quelle couleur est la balle ?", "conflit")], fiable))
    # --- deux entités très similaires ---
    X.append(S("m07", "similaires", {"A": O("clé", "jaune", "table"), "B": O("clé", "jaune", "étagère")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte"), ("A", "lieu"), ("B", "lieu")],
               [("demande", "Où est la clé jaune ?", "?"), ("demande", "Prends la clé sur l'étagère.", "agir:B")], fiable))
    X.append(S("m08", "similaires", {"A": O("cube", "bleu", "table"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "lieu"), ("B", "lieu")],
               [("demande", "Le cube sur le sol est-il bleu ?", "?"), ("capteur", "B", "teinte", "bleu"),
                ("demande", "Le cube sur le sol est-il bleu ?", "oui")], fiable))
    # --- pronoms éloignés ---
    X.append(S("m09", "pronom_éloigné", {"A": O("cube", "rouge", "table"), "B": O("balle", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte")],
               [("dit", "alice", "Le cube rouge est sur la table."), ("dit", "alice", "La balle est sur le sol."),
                ("dit", "alice", "Elle est bleue."), ("dit", "alice", "Il reste sur la table."),
                ("demande", "De quelle couleur est la balle ?", "bleu"), ("demande", "Où est le cube ?", "table")], fiable))
    X.append(S("m10", "pronom_éloigné", {"A": O("cube", "vert", "étagère"), "B": O("cube", "jaune", "table")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "Le cube vert est sur l'étagère."), ("dit", "alice", "Le cube jaune est sur la table."),
                ("dit", "alice", "Il tombe."), ("demande", "Où est le cube jaune ?", "sol"),
                ("demande", "Où est le cube vert ?", "étagère")], fiable))
    # --- changement d'état ---
    X.append(S("m11", "changement_état", {"A": O("balle", "rouge", "sol"), "B": O("balle", "bleu", "table")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte"), ("A", "lieu")],
               [("capteur", "A", "lieu", "table"), ("dit", "alice", "La balle rouge tombe."),
                ("demande", "Où est la balle rouge ?", "sol"), ("demande", "De quelle couleur est la balle sur le sol ?", "?")], fiable,
               "la balle sur le sol : les deux balles ont pu y être ; seul le capteur l'établirait"))
    X.append(S("m12", "changement_état", {"A": O("cube", "rouge", "étagère")}, [("A", "sorte"), ("A", "teinte")],
               [("dit", "alice", "Le cube est sur la table."), ("capteur", "A", "lieu", "étagère"),
                ("demande", "Où est le cube ?", "étagère"), ("demande", "Le cube rouge est-il sur l'étagère ?", "oui")], fiable))
    # --- faux présupposés ---
    X.append(S("m13", "faux_présupposé", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("demande", "Où est le cube vert ?", "présupposé"), ("demande", "Prends le cube jaune.", "présupposé")], fiable))
    X.append(S("m14", "faux_présupposé", {"A": O("balle", "rouge", "table")}, [("A", "sorte"), ("A", "teinte")],
               [("demande", "Où est le livre ?", "présupposé"), ("demande", "De quelle couleur est la balle ?", "rouge")], fiable))
    # --- source non fiable ---
    X.append(S("m15", "source_douteuse", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "bob", "Le cube rouge est sur l'étagère."), ("demande", "Où est le cube rouge ?", "?")], douteux))
    X.append(S("m16", "source_douteuse", {"A": O("balle", "vert", "table")}, [("A", "sorte")],
               [("dit", "bob", "La balle est jaune."), ("dit", "alice", "La balle est verte."),
                ("demande", "De quelle couleur est la balle ?", "vert")], douteux))
    # --- source contradictoire ---
    X.append(S("m17", "sources_contraires", {"A": O("livre", "rouge", "table")}, [("A", "sorte"), ("A", "teinte")],
               [("dit", "alice", "Le livre est sur la table."), ("dit", "carla", "Le livre est sur l'étagère."),
                ("demande", "Où est le livre ?", "conflit"), ("capteur", "A", "lieu", "table"),
                ("demande", "Où est le livre ?", "table")], fiable))
    # --- « l'autre » ---
    X.append(S("m18", "autre", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "étagère")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "alice", "Le cube rouge tombe."), ("dit", "alice", "L'autre reste sur l'étagère."),
                ("demande", "Où est le cube bleu ?", "étagère"), ("demande", "Où est le cube rouge ?", "sol")], fiable))
    X.append(S("m19", "autre", {"A": O("balle", "rouge", "table"), "B": O("balle", "vert", "sol"), "C": O("balle", "jaune", "étagère")},
               [("A", "sorte"), ("B", "sorte"), ("C", "sorte"), ("A", "teinte")],
               [("dit", "alice", "La balle rouge est sur la table."), ("dit", "alice", "L'autre est sur le sol."),
                ("demande", "Où est la balle verte ?", "?")], fiable, "« l'autre » parmi trois : non déterminé"))
    # --- descriptions relationnelles ---
    X.append(S("m20", "relationnelle", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "étagère")},
               [("A", "sorte"), ("B", "sorte"), ("A", "lieu"), ("B", "lieu"), ("A", "teinte"), ("B", "teinte")],
               [("demande", "De quelle couleur est le cube sur la table ?", "rouge"),
                ("demande", "De quelle couleur est le cube sur l'étagère ?", "bleu"),
                ("demande", "Le cube sur la table est-il bleu ?", "non")], fiable))
    X.append(S("m21", "relationnelle", {"A": O("clé", "vert", "table"), "B": O("clé", "rouge", "table")},
               [("A", "sorte"), ("B", "sorte"), ("A", "lieu"), ("B", "lieu")],
               [("demande", "De quelle couleur est la clé sur la table ?", "?")], fiable))
    # --- ellipses simples ---
    X.append(S("m22", "ellipse", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "alice", "Le rouge est sur la table."), ("demande", "Où est le cube rouge ?", "table")], fiable))
    X.append(S("m23", "ellipse", {"A": O("balle", "jaune", "étagère"), "B": O("balle", "vert", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "alice", "La jaune est sur l'étagère."), ("dit", "alice", "La verte est sur le sol."),
                ("demande", "Où est la balle verte ?", "sol"), ("demande", "Où est la balle jaune ?", "étagère")], fiable))
    # --- mots inconnus : l'abstention est la bonne réponse ---
    X.append(S("m24", "hors_lexique", {"A": O("cube", "rouge", "table")}, [("A", "sorte")],
               [("dit", "alice", "Le cube est sur le rayon."), ("demande", "Où est le cube ?", "?")], fiable))
    # --- auto-correction : une réponse probable est révisée par une contradiction fiable ---
    X.append(S("m25", "correction", {"A": O("cube", "rouge", "étagère"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("B", "teinte")],
               [("dit", "alice", "Un cube est sur la table."), ("dit", "alice", "Il est rouge."),
                ("capteur", "A", "teinte", "rouge"), ("capteur", "A", "lieu", "étagère"),
                ("demande", "Où est le cube rouge ?", "étagère")], fiable))
    X.append(S("m26", "correction", {"A": O("balle", "bleu", "table")}, [("A", "sorte")],
               [("dit", "bob", "La balle est rouge."), ("capteur", "A", "teinte", "bleu"),
                ("demande", "De quelle couleur est la balle ?", "bleu")], douteux))
    return X


def scenarios_aveugles():
    """Second lot écrit à la main APRÈS la mise au point : évalué une seule fois, sans retouche du système."""
    X = []
    f = {"alice": (30, 1), "carla": (30, 1)}
    d = {"alice": (30, 1), "bob": (2, 8)}
    X.append(S("b01", "pronom_éloigné", {"A": O("livre", "rouge", "table"), "B": O("clé", "vert", "étagère")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "Le livre est sur la table."), ("dit", "alice", "La clé est sur l'étagère."),
                ("dit", "alice", "Il est rouge."), ("demande", "De quelle couleur est le livre ?", "rouge"),
                ("demande", "De quelle couleur est la clé ?", "?")], f))
    X.append(S("b02", "autre", {"A": O("livre", "bleu", "sol"), "B": O("livre", "vert", "table")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "alice", "Le livre bleu est sur le sol."), ("dit", "alice", "L'autre est sur la table."),
                ("demande", "Où est le livre vert ?", "table")], f))
    X.append(S("b03", "faux_présupposé", {"A": O("clé", "jaune", "table")}, [("A", "sorte"), ("A", "teinte")],
               [("demande", "Où est la clé rouge ?", "présupposé"), ("demande", "Prends la clé rouge.", "présupposé")], f))
    X.append(S("b04", "source_douteuse", {"A": O("livre", "vert", "étagère")}, [("A", "sorte")],
               [("dit", "bob", "Le livre est sur la table."), ("demande", "Où est le livre ?", "?")], d))
    X.append(S("b05", "sources_contraires", {"A": O("cube", "jaune", "sol")}, [("A", "sorte")],
               [("dit", "alice", "Le cube est jaune."), ("dit", "carla", "Le cube est vert."),
                ("demande", "De quelle couleur est le cube ?", "conflit"), ("capteur", "A", "teinte", "jaune"),
                ("demande", "De quelle couleur est le cube ?", "jaune")], f))
    X.append(S("b06", "changement_état", {"A": O("livre", "rouge", "sol"), "B": O("cube", "bleu", "table")},
               [("A", "sorte"), ("B", "sorte"), ("A", "lieu")],
               [("capteur", "A", "lieu", "étagère"), ("dit", "alice", "Le livre tombe."),
                ("demande", "Où est le livre ?", "sol"), ("demande", "Le livre est-il sur l'étagère ?", "non")], f))
    X.append(S("b07", "similaires", {"A": O("balle", "rouge", "table"), "B": O("balle", "rouge", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("demande", "Prends la balle rouge.", "?"), ("capteur", "A", "lieu", "table"), ("capteur", "B", "lieu", "sol"),
                ("demande", "Prends la balle sur le sol.", "agir:B")], f))
    X.append(S("b08", "relationnelle", {"A": O("cube", "vert", "étagère"), "B": O("cube", "jaune", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "lieu"), ("B", "lieu")],
               [("dit", "alice", "Le cube sur l'étagère est vert."), ("demande", "De quelle couleur est le cube sur l'étagère ?", "vert"),
                ("demande", "De quelle couleur est le cube sur le sol ?", "?")], f))
    X.append(S("b09", "ellipse", {"A": O("clé", "bleu", "table"), "B": O("clé", "rouge", "étagère")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "alice", "La bleue est sur la table."), ("demande", "Où est la clé bleue ?", "table"),
                ("demande", "Où est la clé rouge ?", "?")], f))
    X.append(S("b10", "implicite", {"A": O("balle", "vert", "sol"), "B": O("livre", "jaune", "table")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "Une balle est sur le sol."), ("dit", "alice", "Elle est verte."),
                ("demande", "Où est la balle verte ?", "sol"), ("demande", "De quelle couleur est le livre ?", "?")], f))
    X.append(S("b11", "contradictoire", {"A": O("livre", "jaune", "table")}, [("A", "sorte"), ("A", "teinte")],
               [("dit", "alice", "Le livre jaune est rouge."), ("demande", "De quelle couleur est le livre ?", "jaune")], f))
    X.append(S("b12", "incomplète", {"A": O("balle", "bleu", "table"), "B": O("balle", "jaune", "sol")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "La balle est sur la table."), ("demande", "Où est la balle bleue ?", "?"),
                ("demande", "De quelle couleur est la balle sur la table ?", "?")], f))
    X.append(S("b13", "correction", {"A": O("cube", "vert", "sol")}, [("A", "sorte")],
               [("dit", "bob", "Le cube est sur la table."), ("capteur", "A", "lieu", "sol"),
                ("demande", "Où est le cube ?", "sol")], d))
    X.append(S("b14", "hors_lexique", {"A": O("livre", "rouge", "table")}, [("A", "sorte")],
               [("dit", "alice", "Le bouquin est sur la table."), ("demande", "Où est le livre ?", "?")], f))
    X.append(S("b15", "pronom_éloigné", {"A": O("cube", "rouge", "table"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "alice", "Le cube bleu est sur le sol."), ("dit", "alice", "Le cube rouge est sur la table."),
                ("dit", "alice", "Il reste sur la table."), ("demande", "Où est le cube bleu ?", "sol")], f))
    return X


# ================================================================ (2) second générateur (conventions différentes)

VARIATIONS = ("ordre", "redondance", "omission", "reformulation", "retour", "interruption")


def _np(sorte, teinte=None, defini=True):
    f = FEMININ[sorte]
    art = ("la " if f else "le ") if defini else ("une " if f else "un ")
    if sorte[0] in "aeiouéè" and defini: art = "l'"
    return art + sorte + (" " + ACCORD[(teinte, f)] if teinte else "")


def _pron(sorte): return "Elle" if FEMININ[sorte] else "Il"


def _cap(s): return s[0].upper() + s[1:]


def scenario_genere(i, rng: random.Random, variation):
    sortes = ["cube", "balle", "livre", "clé"]
    s1 = rng.choice(sortes)
    t1, t2 = rng.sample(TEINTES, 2)
    l1, l2 = rng.sample(LIEUX, 2)
    autre_sorte = rng.choice([x for x in sortes if x != s1])
    objets = {"A": O(s1, t1, l1), "B": O(s1, t2, l2), "C": O(autre_sorte, rng.choice(TEINTES), rng.choice(LIEUX))}
    capteurs = [("A", "sorte"), ("B", "sorte"), ("C", "sorte"), ("B", "teinte")]
    hist = {"dora": (25, 1)}
    tours = []
    # le locuteur situe A par sa teinte (B est connu du capteur : A est déterminé par élimination de teinte)
    phrase_a = f"{_cap(_np(s1, t1))} est sur {ARTICLE_LIEU[l1]}."
    if variation == "reformulation":
        tours.append(("dit", "dora", f"{_cap(_np(s1, defini=False))} est sur {ARTICLE_LIEU[l1]}."))
        tours.append(("dit", "dora", f"{_pron(s1)} est {ACCORD[(t1, FEMININ[s1])]}."))
    elif variation == "omission":
        tours.append(("dit", "dora", f"{_cap(_np(s1))} est sur {ARTICLE_LIEU[l1]}."))       # teinte omise
    else:
        tours.append(("dit", "dora", phrase_a))
    if variation == "redondance": tours.append(("dit", "dora", phrase_a))
    if variation == "interruption":
        tours.append(("dit", "dora", f"{_cap(_np(autre_sorte))} est sur {ARTICLE_LIEU[objets['C']['lieu']]}."))
        tours.append(("dit", "dora", f"{_pron(autre_sorte)} est {ACCORD[(objets['C']['teinte'], FEMININ[autre_sorte])]}."))
    if variation == "retour":
        tours.append(("dit", "dora", f"{_cap(_np(autre_sorte))} est sur {ARTICLE_LIEU[objets['C']['lieu']]}."))
        tours.append(("dit", "dora", f"{_cap(_np(s1, t1))} tombe."))
        objets["A"]["lieu"] = "sol"
    attendu_a = "?" if variation == "omission" else objets["A"]["lieu"]
    questions = [("demande", f"Où est {_np(s1, t1)} ?", attendu_a),
                 ("demande", f"Où est {_np(s1, t2)} ?", "?"),
                 ("demande", f"Où est {_np(s1, rng.choice([t for t in TEINTES if t not in (t1, t2)]))} ?", "présupposé")]
    if variation == "ordre": rng.shuffle(questions)
    if variation == "interruption":
        questions.append(("demande", f"De quelle couleur est {_np(autre_sorte)} ?", objets["C"]["teinte"]))
    tours += questions
    return {"id": f"g{i:03d}", "catégorie": f"variation_{variation}", "objets": objets, "capteurs": capteurs,
            "tours": tours, "historique": hist, "note": "", "origine": "générateur_2"}


def scenarios_generes(n=120, seed=7):
    rng = random.Random(seed)
    return [scenario_genere(i, rng, VARIATIONS[i % len(VARIATIONS)]) for i in range(n)]


def scenarios_trompeurs(n=60, seed=3, taux=0.4):
    """Un locuteur à bon historique se trompe parfois de lieu (la vérité n'est révélée que par le capteur)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s1 = rng.choice(["cube", "balle", "livre", "clé"]); t1, t2 = rng.sample(TEINTES, 2)
        vrai, faux = rng.sample(LIEUX, 2)
        dit = faux if rng.random() < taux else vrai
        objets = {"A": O(s1, t1, vrai), "B": O(s1, t2, rng.choice(LIEUX))}
        tours = [("dit", "dora", f"{_cap(_np(s1, t1))} est sur {ARTICLE_LIEU[dit]}."),
                 ("demande", f"Où est {_np(s1, t1)} ?", vrai)]
        out.append({"id": f"t{i:03d}", "catégorie": "trompeur" if dit != vrai else "sincère", "objets": objets,
                    "capteurs": [("A", "sorte"), ("B", "sorte"), ("B", "teinte")], "tours": tours,
                    "historique": {"dora": (25, 1)}, "note": "", "origine": "trompeur"})
    return out


def scenarios_commandes(n=60, seed=5):
    """Commande sur un objet décrit ; les teintes ne sont pas observées : l'identité doit être établie (0,99)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s1 = rng.choice(["cube", "balle", "livre", "clé"]); t1, t2 = rng.sample(TEINTES, 2)
        l1, l2 = rng.sample(LIEUX, 2)
        objets = {"A": O(s1, t1, l1), "B": O(s1, t2, l2)}
        tours = [("demande", f"Prends {_np(s1, t1)}.", "agir:A")]
        out.append({"id": f"c{i:03d}", "catégorie": "commande", "objets": objets, "capteurs": [("A", "sorte"), ("B", "sorte")],
                    "tours": tours, "historique": {}, "note": "", "origine": "commandes"})
    return out


def banc_tiers(n_generes=120, seed=7):
    return scenarios_main() + scenarios_generes(n_generes, seed)


def nombre_questions(banc):
    return sum(1 for s in banc for t in s["tours"] if t[0] == "demande")
