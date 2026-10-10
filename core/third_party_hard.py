"""HARD third-party benchmark: development + SEALED set (evaluated only once, with no adjustment of the system).

Standard library only (no WCE component). Same conventions as core.third_party_discourse:
objects (hidden truth), sensors, speaker history, turns ("dit" / "capteur" / "demande").
The expected answers are those of a competent AND cautious reader: an answer is expected only if it is legitimate.

Categories: unseen vocabulary, new expressions, lexical ambiguity, misleading sources, elimination,
partial knowledge, false presupposition, temporality, long reference chain, polysemy.
"""
from __future__ import annotations

import hashlib
import json
import random


def S(id, cat, objets, capteurs, tours, historique=None):
    return {"id": id, "catégorie": cat, "objets": objets, "capteurs": capteurs, "tours": tours,
            "historique": historique or {"alice": (30, 1), "carla": (30, 1), "bob": (2, 8)}, "note": "", "origine": "difficile"}


def O(sorte, teinte, lieu): return {"sorte": sorte, "teinte": teinte, "lieu": lieu}


# ================================================================ development set (tuning allowed)

def dev_set():
    X = []
    X.append(S("d01", "vocabulaire_non_vu", {"A": O("cube", "rouge", "table")}, [("A", "sorte")],
               [("dit", "alice", "Le gobelet est sur la table."), ("demande", "Où est le gobelet ?", "?"),
                ("demande", "Où est le cube ?", "?")]))
    X.append(S("d02", "expression_nouvelle", {"A": O("livre", "vert", "étagère")}, [("A", "sorte")],
               [("dit", "alice", "Le livre se trouve sur l'étagère."), ("demande", "Où est le livre ?", "étagère")]))
    X.append(S("d03", "élimination", {"A": O("cube", "bleu", "sol"), "B": O("cube", "rouge", "table")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte")],
               [("dit", "alice", "Le cube rouge est sur la table."), ("demande", "Où est le cube rouge ?", "table"),
                ("demande", "Prends le cube rouge.", "agir:B")]))
    X.append(S("d04", "source_trompeuse", {"A": O("balle", "vert", "sol")}, [("A", "sorte")],
               [("dit", "bob", "La balle est sur la table."), ("dit", "bob", "La balle est rouge."),
                ("demande", "Où est la balle ?", "?"), ("demande", "De quelle couleur est la balle ?", "?")]))
    X.append(S("d05", "temporalité", {"A": O("cube", "jaune", "sol")}, [("A", "sorte")],
               [("dit", "alice", "Le cube était sur la table."), ("dit", "alice", "Il tombe."),
                ("demande", "Où est le cube ?", "sol")]))
    X.append(S("d06", "chaîne_longue", {"A": O("livre", "rouge", "table"), "B": O("balle", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "Le livre est sur la table."), ("dit", "alice", "Il est rouge."), ("dit", "alice", "La balle est sur le sol."),
                ("dit", "alice", "Elle est bleue."), ("dit", "alice", "Il reste sur la table."),
                ("demande", "De quelle couleur est le livre ?", "rouge"), ("demande", "Où est la balle ?", "sol")]))
    X.append(S("d07", "polysémie", {"A": O("clé", "jaune", "table")}, [("A", "sorte")],
               [("demande", "Range la clé.", "?")]))
    X.append(S("d08", "connaissance_partielle", {"A": O("cube", "vert", "table"), "B": O("cube", "bleu", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte")],
               [("demande", "Où est le cube vert ?", "?"), ("demande", "De quelle couleur est le cube sur le sol ?", "?")]))
    X.append(S("d09", "présupposition", {"A": O("livre", "rouge", "table")}, [("A", "sorte"), ("A", "teinte")],
               [("demande", "Pourquoi le livre bleu est-il sur la table ?", "présupposé")]))
    return X


def dev_generated(n=60, seed=17):
    """Generated variants (conventions specific to this module): elimination, misleading source, chains."""
    rng = random.Random(seed)
    sortes = [("cube", False), ("balle", True), ("livre", False), ("clé", True)]
    teintes = {("rouge", False): "rouge", ("bleu", False): "bleu", ("vert", False): "vert", ("jaune", False): "jaune",
               ("rouge", True): "rouge", ("bleu", True): "bleue", ("vert", True): "verte", ("jaune", True): "jaune"}
    lieux = {"table": "la table", "étagère": "l'étagère", "sol": "le sol"}
    out = []
    for i in range(n):
        (s, f), (t1, t2) = rng.choice(sortes), rng.sample(["rouge", "bleu", "vert", "jaune"], 2)
        l1, l2 = rng.sample(list(lieux), 2)
        art = "La" if f else "Le"; art_l = "la" if f else "le"
        kind = ("élimination", "source_trompeuse", "chaîne_longue")[i % 3]
        objets = {"A": O(s, t1, l1), "B": O(s, t2, l2)}
        if kind == "élimination":
            tours = [("dit", "alice", f"{art} {s} {teintes[(t1, f)]} est sur {lieux[l1]}."),
                     ("demande", f"Où est {art_l} {s} {teintes[(t1, f)]} ?", l1)]
            capt = [("A", "sorte"), ("B", "sorte"), ("B", "teinte")]
        elif kind == "source_trompeuse":
            tours = [("dit", "bob", f"{art} {s} {teintes[(t1, f)]} est sur {lieux[l2]}."),
                     ("demande", f"Où est {art_l} {s} {teintes[(t1, f)]} ?", "?")]
            capt = [("A", "sorte"), ("B", "sorte"), ("B", "teinte")]
        else:
            p = "Elle" if f else "Il"
            tours = [("dit", "alice", f"{art} {s} {teintes[(t1, f)]} est sur {lieux[l1]}."),
                     ("dit", "alice", f"{art} {s} {teintes[(t2, f)]} est sur {lieux[l2]}."),
                     ("dit", "alice", f"{p} tombe."),
                     ("demande", f"Où est {art_l} {s} {teintes[(t2, f)]} ?", "sol"),
                     ("demande", f"Où est {art_l} {s} {teintes[(t1, f)]} ?", l1)]
            capt = [("A", "sorte"), ("B", "sorte")]
            objets["B"]["lieu"] = "sol"
        out.append(S(f"g{i:03d}", kind, objets, capt, tours))
    return out


# ================================================================ SEALED set (written before any evaluation; do not modify)

def sealed_set():
    X = []
    X.append(S("s01", "vocabulaire_non_vu", {"A": O("balle", "rouge", "table")}, [("A", "sorte")],
               [("dit", "alice", "Le caillou est sur l'étagère."), ("demande", "Où est le caillou ?", "?"),
                ("demande", "Où est la balle ?", "?")]))
    X.append(S("s02", "vocabulaire_non_vu", {"A": O("cube", "bleu", "sol")}, [("A", "sorte"), ("A", "teinte")],
               [("dit", "alice", "Le cube est turquoise."), ("demande", "De quelle couleur est le cube ?", "bleu")]))
    X.append(S("s03", "expression_nouvelle", {"A": O("clé", "vert", "table")}, [("A", "sorte")],
               [("dit", "alice", "La clé repose sur la table."), ("demande", "Où est la clé ?", "table")]))
    X.append(S("s04", "expression_nouvelle", {"A": O("livre", "jaune", "sol")}, [("A", "sorte")],
               [("dit", "alice", "Le livre traîne par terre."), ("demande", "Où est le livre ?", "sol")]))
    X.append(S("s05", "ambiguïté_lexicale", {"A": O("balle", "vert", "table"), "B": O("balle", "rouge", "étagère")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "La verte est sur la table."), ("dit", "alice", "L'autre est sur l'étagère."),
                ("demande", "Où est la balle rouge ?", "?"), ("demande", "Où est la balle verte ?", "table")]))
    X.append(S("s06", "source_trompeuse", {"A": O("cube", "rouge", "étagère"), "B": O("cube", "jaune", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "bob", "Le cube rouge est sur la table."), ("dit", "alice", "Le cube rouge est sur l'étagère."),
                ("demande", "Où est le cube rouge ?", "étagère")]))
    X.append(S("s07", "source_trompeuse", {"A": O("livre", "bleu", "table")}, [("A", "sorte")],
               [("dit", "bob", "Le livre est vert."), ("dit", "bob", "Le livre est sur le sol."),
                ("demande", "De quelle couleur est le livre ?", "?"), ("demande", "Le livre est-il sur le sol ?", "?")]))
    X.append(S("s08", "élimination", {"A": O("balle", "jaune", "sol"), "B": O("balle", "bleu", "table"), "C": O("balle", "vert", "étagère")},
               [("A", "sorte"), ("B", "sorte"), ("C", "sorte"), ("A", "teinte"), ("C", "teinte")],
               [("dit", "alice", "La balle bleue est sur la table."), ("demande", "Où est la balle bleue ?", "table"),
                ("demande", "Prends la balle bleue.", "agir:B")]))
    X.append(S("s09", "élimination", {"A": O("clé", "rouge", "table"), "B": O("clé", "vert", "sol"), "C": O("clé", "jaune", "étagère")},
               [("A", "sorte"), ("B", "sorte"), ("C", "sorte"), ("A", "teinte")],
               [("demande", "Prends la clé verte.", "?")]))
    X.append(S("s10", "connaissance_partielle", {"A": O("livre", "rouge", "table"), "B": O("cube", "vert", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte")],
               [("dit", "alice", "Le livre est sur la table."), ("demande", "Où est le livre ?", "table"),
                ("demande", "De quelle couleur est le cube ?", "?"), ("demande", "Où est le cube ?", "?")]))
    X.append(S("s11", "connaissance_partielle", {"A": O("balle", "bleu", "sol"), "B": O("balle", "rouge", "table")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "Une balle est sur le sol."), ("demande", "Où est la balle bleue ?", "?"),
                ("demande", "De quelle couleur est la balle sur le sol ?", "?")]))
    X.append(S("s12", "présupposition", {"A": O("cube", "jaune", "table"), "B": O("cube", "vert", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("demande", "Où est le cube rouge ?", "présupposé"), ("demande", "Prends le cube bleu.", "présupposé"),
                ("demande", "Le cube jaune est-il sur la table ?", "?")]))
    X.append(S("s13", "présupposition", {"A": O("livre", "vert", "étagère")}, [("A", "sorte"), ("A", "teinte")],
               [("demande", "Pourquoi la balle est-elle sur l'étagère ?", "présupposé")]))
    X.append(S("s14", "temporalité", {"A": O("balle", "rouge", "sol")}, [("A", "sorte"), ("A", "teinte")],
               [("capteur", "A", "lieu", "table"), ("dit", "alice", "La balle rouge tombe."),
                ("demande", "Où est la balle rouge ?", "sol"), ("demande", "La balle rouge est-elle sur la table ?", "non")]))
    X.append(S("s15", "temporalité", {"A": O("cube", "bleu", "étagère")}, [("A", "sorte")],
               [("dit", "alice", "Le cube était sur la table."), ("capteur", "A", "lieu", "étagère"),
                ("demande", "Où est le cube ?", "étagère")]))
    X.append(S("s16", "temporalité", {"A": O("livre", "rouge", "sol")}, [("A", "sorte")],
               [("dit", "alice", "Le livre est sur la table."), ("dit", "carla", "Le livre est sur le sol."),
                ("demande", "Où est le livre ?", "conflit")]))
    X.append(S("s17", "chaîne_longue", {"A": O("cube", "rouge", "table"), "B": O("clé", "jaune", "étagère"), "C": O("balle", "vert", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("C", "sorte")],
               [("dit", "alice", "Le cube est sur la table."), ("dit", "alice", "La clé est sur l'étagère."),
                ("dit", "alice", "Elle est jaune."), ("dit", "alice", "La balle est sur le sol."), ("dit", "alice", "Elle est verte."),
                ("dit", "alice", "Le cube est rouge."), ("dit", "alice", "Il reste sur la table."),
                ("demande", "De quelle couleur est la clé ?", "jaune"), ("demande", "Où est la balle verte ?", "sol"),
                ("demande", "Où est le cube rouge ?", "table")]))
    X.append(S("s18", "chaîne_longue", {"A": O("livre", "bleu", "sol"), "B": O("livre", "vert", "table")},
               [("A", "sorte"), ("B", "sorte")],
               [("dit", "alice", "Le livre bleu est sur le sol."), ("dit", "alice", "Le livre vert est sur la table."),
                ("dit", "alice", "Il tombe."), ("dit", "alice", "L'autre reste sur le sol."),
                ("demande", "Où est le livre vert ?", "sol"), ("demande", "Où est le livre bleu ?", "sol")]))
    X.append(S("s19", "polysémie", {"A": O("livre", "rouge", "table")}, [("A", "sorte")],
               [("dit", "alice", "Alice range le livre."), ("demande", "Où est le livre ?", "?")]))
    X.append(S("s20", "polysémie", {"A": O("cube", "vert", "étagère")}, [("A", "sorte"), ("A", "lieu")],
               [("demande", "Le cube est-il sur l'étagère ?", "oui"), ("demande", "Le cube est-il vert ?", "?")]))
    X.append(S("s21", "similaires", {"A": O("cube", "rouge", "table"), "B": O("cube", "rouge", "sol")},
               [("A", "sorte"), ("B", "sorte"), ("A", "teinte"), ("B", "teinte")],
               [("dit", "alice", "Le cube rouge est sur la table."), ("demande", "Où est le cube rouge ?", "?")]))
    X.append(S("s22", "source_trompeuse", {"A": O("clé", "bleu", "table")}, [("A", "sorte"), ("A", "lieu")],
               [("dit", "alice", "La clé est sur le sol."), ("demande", "Où est la clé ?", "table")]))
    return X


def sealed_hash():
    return hashlib.sha256(json.dumps(sealed_set(), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


SEALED_SHA256 = "82490dcd725b0fc9fbbe5d52a2fb5afe6b9323941606632507b80a92979e6d67"   # fingerprint of the sealed set (frozen at its creation)


def n_questions(bench): return sum(1 for s in bench for t in s["tours"] if t[0] == "demande")
