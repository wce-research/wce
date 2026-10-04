"""Recherche structurelle HIÉRARCHIQUE (jalon Deep Compositional Structure Discovery).

Stratégies comparées, toutes sur le même budget d'équivalents-ajustement :
  flat          : candidats plats (chaîne + atome, ajustement joint depuis des initialisations heuristiques) — ablation
                  « sans hiérarchie » ;
  bottom_up     : sous-structures ajustées puis composées ; le nouvel opérande est ajusté sur la CIBLE RÉSIDUELLE
                  (ce que le parent n'explique pas) ; compositions de sous-structures déjà ajustées criblées sans
                  nouvel ajustement (prédictions en cache) ;
  top_down      : décomposition récursive de la cible en (atome ancre, combinateur, cible résiduelle à support
                  restreint) ;
  bidirectional : la cible résiduelle descendante est d'abord appariée aux sous-structures montantes déjà ajustées ;
  best_first    : file de priorité sur un score explicite (évidence, réutilisation, nouveauté, coût).

Ce module n'importe aucun module de monde et ne reçoit jamais la vérité.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from core.boundary_program import BNode
from core.deep_fit import DT, FitCounter, FitData, approx_evidence, fit_batch, heuristic_inits, loglik_batch, predict_batch, profiled_loglik, validated_fit
from core.deep_structure import (COMBINATORS, COMPACT_PARAMS, FragmentLibrary, StructureStore, WorldHypothesis, description_bits, family_of, is_composite,
                                 n_atoms, normalize, skel_key)

STRATEGIES = ("flat", "bottom_up", "top_down", "bidirectional", "best_first", "factorized")


def _leaves(p: BNode):
    return [l for c in p.children for l in _leaves(c)] if is_composite(p) else [p]


def estimate_noise_nn(X, y):
    """Estimation du bruit SANS modèle : désaccord d'étiquette avec le plus proche voisin, d ≈ 2ε(1−ε) loin des
    frontières (borne haute : les frontières l'augmentent). Bornée à [0,02 ; 0,30]."""
    D = torch.cdist(X.to(DT), X.to(DT)); D.fill_diagonal_(float("inf")); d = float((y[D.argmin(1)] != y).double().mean())
    eps = (1 - math.sqrt(max(0.0, 1 - 2 * min(d, 0.499)))) / 2
    return min(0.30, max(0.02, eps))


def residual_target(comb: str, a, y):
    """Cible et support de l'opérande R tel que comb(a, R) = y, a étant les prédictions dures (0/1) du parent.
    Renvoie (cible, poids, part d'erreurs INÉVITABLES quel que soit R)."""
    a = a.to(DT); y = y.to(DT)
    if comb == "xor": return (y - a).abs(), torch.ones_like(y), 0.0
    if comb == "or": return y, (a == 0).to(DT), float(((a == 1) & (y == 0)).double().mean())
    if comb == "and": return y, (a == 1).to(DT), float(((a == 0) & (y == 1)).double().mean())
    raise KeyError(comb)


def combine(comb: str, a, b):
    return a * b if comb == "and" else a + b - a * b if comb == "or" else (a - b).abs()


@dataclass
class SearchOutcome:
    hypotheses: List[WorldHypothesis]
    trace: List[dict]
    stats: dict
    noise: float
    store: StructureStore
    preds: Dict[str, torch.Tensor] = field(default_factory=dict)
    all: Dict[str, WorldHypothesis] = field(default_factory=dict)        # toutes les structures ajustées (sans dédoublonnage fonctionnel)
    discovery: object = None                                             # résultat de PrimitiveDiscovery (si utilisée)

    def best(self): return self.hypotheses[0] if self.hypotheses else None


class DeepSearch:
    """Recherche hiérarchique. `hierarchy=False` désactive résidus, cache et réutilisation (ablation)."""

    def __init__(self, strategy="bottom_up", budget=150.0, max_depth=6, beam=3, gamma=0.02, seed=0, library: Optional[FragmentLibrary] = None,
                 use_families=True, hierarchy=True, top_operands=2, atom_steps=50, dedup=True, time_budget=None, patience=1,
                 fact_inits=6, fact_rounds=3, fact_top=4, fact_polish=8, geometry=False, fact_time=30.0, fact_feedback=True, fact_separation=False, fact_literals=None, fact_discovery=None, fact_residual=True, validated=None):
        assert strategy in STRATEGIES
        self.strategy, self.budget, self.max_depth, self.beam, self.gamma, self.seed = strategy, budget, max_depth, beam, gamma, seed
        self.library, self.use_families, self.hierarchy, self.top_operands, self.atom_steps, self.dedup = library, use_families, hierarchy and strategy != "flat", top_operands, atom_steps, dedup
        self.time_budget, self.patience = time_budget, patience
        self.fact_inits, self.fact_rounds, self.fact_top, self.fact_polish, self.geometry, self.fact_time = fact_inits, fact_rounds, fact_top, fact_polish, geometry, fact_time
        self.validated = validated                                                               # None (toujours optimiser) | "profiled" | "surrogate" (garde-fou) | "never" | "gate"
        self.fact_feedback, self.fact_separation, self.fact_literals, self.fact_discovery, self.fact_residual, self.discovery = fact_feedback, fact_separation, fact_literals, fact_discovery, fact_residual, None

    # ---- état ---------------------------------------------------------------------------------
    def _setup(self, data: FitData, noise):
        self.data = data.with_flip(noise) if noise is not None else data; self.noise = float(self.data.flip.mean())
        self.store = StructureStore(); self.counter = FitCounter(self.budget); self.hyps: Dict[str, WorldHypothesis] = {}; self.preds: Dict[str, torch.Tensor] = {}
        self.trace: List[dict] = []; self.n = len(self.data); self.t0 = time.time(); self.generated = 0; self.fitted = 0; self.pruned = 0; self.fit_seconds = 0.0
        self.per_depth = {}; self._best_ev = float("-inf"); self._stall = -1; self.synth_seconds = 0.0; self.compositions = 0; self.gen_stats = []; self.transactions = []; self.decisions = []

    def _stop(self): return self.counter.exhausted() or (self.time_budget is not None and time.time() - self.t0 > self.time_budget)

    def _log(self, stage, **kw): self.trace.append({"stage": stage, "fit_equivalents": round(self.counter.fit_equivalents, 2), **kw})

    def _register(self, prog: BNode, ll: float, origin: str) -> Optional[WorldHypothesis]:
        """Enregistre une structure ajustée : internement (partage des sous-structures), cache, meilleure version."""
        prog = normalize(prog); node = self.store.intern(prog.skeleton()); d = n_atoms(prog)
        pred = predict_batch(prog.skeleton(), torch.tensor([prog.param_vector()], dtype=DT), self.data.X)[0].to(DT)
        ll = profiled_loglik(pred, self.data); ev = approx_evidence(prog.n_params(), d, ll, self.n)
        st = self.per_depth.setdefault(d, {"generated": 0, "fitted": 0, "kept": 0}); st["fitted"] += 1; self.fitted += 1
        old = self.hyps.get(node.key)
        if prog.n_params() > COMPACT_PARAMS: self._log("pruning", key=node.key, depth=d, reason="compact_budget"); self.pruned += 1; return None
        if old is not None and old.loglik >= ll: return old
        h = WorldHypothesis(node.key, prog, ll, ev, family=family_of(prog), origin=origin, fit_cost=round(self.counter.fit_equivalents, 2)); self.hyps[node.key] = h
        node.fits["current"] = (tuple(prog.param_vector()), ll)
        self.preds[node.key] = pred
        self._log("fit", key=node.key, depth=d, loglik=round(ll, 3), evidence=round(ev, 3), origin=origin); return h

    def _fit(self, skel: BNode, inits, steps, origin, data: Optional[FitData] = None):
        """Ajuste un squelette (K initialisations en lot). Avec `data` (cible résiduelle), renvoie les paramètres sans
        enregistrer d'hypothèse de monde."""
        t = time.time()
        if data is None and self.validated in ("never", "gate") and len(inits) == 1:
            # Politiques d'optimisation conditionnelle (seul étage modifié) : `never` préserve le candidat tel quel ;
            # `gate` décide PRESERVE / OPTIMIZE sur des signaux observables puis valide (core.deep_optgate).
            if self.validated == "never": row = torch.as_tensor(inits[0], dtype=DT).reshape(-1)
            else:
                from core.deep_optgate import conditional_fit
                dec = conditional_fit(skel, self.data, inits[0], counter=self.counter, steps=steps); self.decisions.append(dec); row = torch.tensor(dec.result, dtype=DT)
            th = row.reshape(1, -1); ll = loglik_batch(skel, th, self.data, self.gamma)
        elif data is None and self.validated is not None and len(inits) == 1:
            # GARDE-FOU (seul étage modifié) : un candidat EXISTANT n'est remplacé par sa version ajustée que si le
            # critère observable s'améliore — propose → evaluate → commit / rollback (core.deep_fit.validated_fit).
            row, tx = validated_fit(skel, self.data, inits[0], steps=steps, gamma=self.gamma, counter=self.counter, criterion=self.validated); self.transactions.append(tx)
            th = row.reshape(1, -1); ll = loglik_batch(skel, th, self.data, self.gamma)
        else: th, ll = fit_batch(skel, data or self.data, inits, steps=steps, gamma=self.gamma, counter=self.counter)
        self.fit_seconds += time.time() - t
        self.generated += 1; self.per_depth.setdefault(n_atoms(skel), {"generated": 0, "fitted": 0, "kept": 0})["generated"] += 1
        i = int(ll.argmax()); prog = skel.skeleton().with_params(th[i].tolist())
        return prog if data is not None else self._register(prog, float(ll[i]), origin)

    def _ranked(self, depth=None):
        hs = [h for h in self.hyps.values() if depth is None or h.depth == depth]; hs.sort(key=lambda h: -h.evidence)
        if not self.dedup: return hs
        out, sigs = [], []                                                # nouveauté : un seul représentant par fonction observée
        for h in hs:
            p = self.preds[h.key]
            if any(float((p != s).double().mean()) < 0.005 for s in sigs): continue
            out.append(h); sigs.append(p)
        return out

    def _errors(self, h): return float((self.preds[h.key] != self.data.y).double().mean())

    def _stalled(self):
        """Règle d'arrêt : `patience` étapes d'approfondissement consécutives sans gain d'évidence (facteur de Bayes
        < 3) de la meilleure hypothèse. Aucun bruit supposé connu."""
        b = self._ranked()[0].evidence
        if b > self._best_ev + math.log(3.0): self._best_ev = b; self._stall = 0
        else: self._stall += 1
        return self._stall >= self.patience

    # ---- briques ------------------------------------------------------------------------------
    def _atoms(self): return [self.store.nodes[k].skeleton for k in self.store.primitives]

    def _fit_atoms(self, n_init=2):
        for a in self._atoms():
            if self._stop(): break
            self._fit(a, heuristic_inits(a, self.data, n_init, self.seed), 100, "primitive")
        self._log("substructure_discovery", level=1, kept=len(self.hyps))

    def _operand_skeletons(self, room: int):
        """Opérandes candidats : atomes, puis fragments de la bibliothèque (et instances de familles) s'ils tiennent
        dans la profondeur restante."""
        ops = [(a, "atom") for a in self._atoms()]
        if self.library is not None and room >= 2:
            ops += [(f, "fragment") for f in self.library.proposals(max_atoms=room)[:6]]
            if self.use_families:
                seen = {skel_key(o) for o, _ in ops}
                for fam in self.library.families[:3]:
                    if len(fam.variables) != 1: continue
                    for a in self._atoms():
                        s = fam.instantiate({fam.variables[0]: a})
                        if n_atoms(s) <= room and skel_key(s) not in seen: seen.add(skel_key(s)); ops.append((s, "family"))
        return ops

    def _extend(self, parent: WorldHypothesis, combs=COMBINATORS, origin="composition"):
        """Compose `parent` avec un nouvel opérande ajusté sur la cible résiduelle, puis polit l'ensemble conjointement."""
        out = []; a = self.preds[parent.key]; room = self.max_depth - parent.depth
        if room < 1: return out
        for comb in combs:
            if self._stop(): break
            r, w, unavoidable = residual_target(comb, a, self.data.y)
            if float(w.sum()) < 8 or unavoidable >= self._errors(parent): self.pruned += 1; continue    # élagage : ce combinateur ne peut pas améliorer le parent
            rd = self.data.with_target(r, w * self.data.weight); cands = []
            for skel, kind in self._operand_skeletons(room):
                if self._stop(): break
                steps = self.atom_steps if kind == "atom" else 100
                op = self._fit(skel, heuristic_inits(skel, rd, 1, self.seed + len(self.trace)), steps, f"residual_{kind}", data=rd)
                th = torch.tensor([op.param_vector()], dtype=DT); cands.append((float(loglik_batch(skel, th, rd, self.gamma)[0]), op, kind))
            cands.sort(key=lambda c: -c[0])
            for _, op, kind in cands[: self.top_operands]:
                if self._stop(): break
                joint = normalize(BNode(comb, children=(parent.program, op)))
                h = self._fit(joint.skeleton(), torch.tensor([joint.param_vector()], dtype=DT), 100, f"{origin}_{comb}_{kind}")
                if h is not None: out.append(h)
        return out

    def _compose_cached(self, limit=3):
        """Réutilisation : compositions de deux sous-structures DÉJÀ ajustées, criblées sur leurs prédictions en cache
        (aucun ajustement), puis polies conjointement pour les meilleures."""
        hs = self._ranked()[:8]; scored = []; y = self.data.y
        for i, a in enumerate(hs):
            for b in hs[i + 1:]:
                if a.depth + b.depth > self.max_depth: continue
                for comb in COMBINATORS:
                    k = skel_key(BNode(comb, children=(a.program, b.program)))
                    if k in self.hyps: continue
                    err = float((combine(comb, self.preds[a.key], self.preds[b.key]) != y).double().mean())
                    if err < min(self._errors(a), self._errors(b)) - 0.01: scored.append((err, comb, a, b))
        scored.sort(key=lambda s: s[0]); out = []
        for err, comb, a, b in scored[:limit]:
            if self._stop(): break
            joint = normalize(BNode(comb, children=(a.program, b.program)))
            h = self._fit(joint.skeleton(), torch.tensor([joint.param_vector()], dtype=DT), 100, f"reuse_{comb}")
            if h is not None: out.append(h)
        if scored: self._log("composition", reused_pairs=len(scored[:limit]), screened=len(scored))
        return out

    # ---- stratégies ---------------------------------------------------------------------------
    def _run_flat(self):
        """Sans hiérarchie : chaîne + atome, ajustement joint depuis une initialisation heuristique (pas de résidu, pas
        de cache, pas de réutilisation)."""
        level = 1
        while level < self.max_depth and not self._stop():
            parents = self._ranked(level)[: self.beam]
            if not parents or self._stalled(): break
            for par in parents:
                for comb in COMBINATORS:
                    for a in self._atoms():
                        if self._stop(): break
                        ai = heuristic_inits(a, self.data, 1, self.seed + self.generated)[0]
                        joint = normalize(BNode(comb, children=(par.program, a.with_params(ai.tolist()))))
                        self._fit(joint.skeleton(), torch.tensor([joint.param_vector()], dtype=DT), 100, f"flat_{comb}")
            level += 1

    def _run_bottom_up(self):
        level = 1
        while level < self.max_depth and not self._stop():
            if self._stalled(): self._log("selection", reason="no_evidence_gain", level=level); break
            parents = [h for h in self._ranked() if h.depth == level][: self.beam] or self._ranked()[: self.beam]
            for par in parents:
                if self._stop(): break
                self._extend(par)
            self._compose_cached(); level += 1

    def _decompose(self, y, w, room, depth_left, tag):
        """Top-down : meilleur atome ancre sur la cible, puis récursion sur la cible résiduelle. Renvoie un programme
        (non enregistré) et ses prédictions dures."""
        td = self.data.with_target(y, w * self.data.weight); best = None
        if self.strategy == "bidirectional":                              # rencontre : une sous-structure montante explique-t-elle déjà la cible ?
            for h in self._ranked()[:12]:
                if h.depth > room: continue
                err = float(((self.preds[h.key] != y).to(DT) * w).sum() / w.sum().clamp_min(1))
                if best is None or err < best[0]: best = (err, h.program, self.preds[h.key])
            if best is not None: self._log("composition", stage_detail="meet", err=round(best[0], 4), tag=tag)
        if best is None or best[0] > 0.02:
            for a in self._atoms():
                if self._stop(): break
                op = self._fit(a, heuristic_inits(a, td, 1, self.seed + len(self.trace)), self.atom_steps, "decompose_atom", data=td)
                p = predict_batch(a, torch.tensor([op.param_vector()], dtype=DT), self.data.X)[0].to(DT)
                err = float(((p != y).to(DT) * w).sum() / w.sum().clamp_min(1))
                if best is None or err < best[0]: best = (err, op, p)
        err, prog, pred = best
        tol = 0.5 / math.sqrt(max(1.0, float(w.sum())))                    # échelle du bruit d'échantillonnage sur ce support
        if room - n_atoms(prog) < 1 or depth_left <= 1 or err <= tol or self._stop(): return prog, pred
        opts = []
        for comb in COMBINATORS:
            r, rw, unavoidable = residual_target(comb, pred, y); rw = rw * w
            if float(rw.sum()) < 8: continue
            minority = float((torch.minimum((r * rw).sum(), ((1 - r) * rw).sum())) / rw.sum().clamp_min(1))
            opts.append((unavoidable + 0.25 * minority, comb, r, rw))
        opts.sort(key=lambda o: o[0]); res = (err, prog, pred)
        for _, comb, r, rw in opts[: 2 if depth_left == self.max_depth else 1]:
            if self._stop(): break
            sub, sp = self._decompose(r, rw, room - n_atoms(prog), depth_left - 1, tag + comb[0])
            cp = combine(comb, pred, sp); e2 = float(((cp != y).to(DT) * w).sum() / w.sum().clamp_min(1))
            if e2 < res[0] - tol: res = (e2, BNode(comb, children=(prog, sub)), cp)
        return res[1], res[2]

    def _run_top_down(self):
        rounds = 0
        while not self._stop() and rounds < 3:
            if self._stalled(): self._log("selection", reason="no_evidence_gain", round=rounds); break
            prog, _ = self._decompose(self.data.y, torch.ones_like(self.data.y), self.max_depth, self.max_depth, f"r{rounds}")
            joint = normalize(prog)
            if skel_key(joint) in self.hyps and rounds: break
            self._fit(joint.skeleton(), torch.tensor([joint.param_vector()], dtype=DT), 100, "top_down_joint"); rounds += 1; self.seed += 101

    def _score(self, h: WorldHypothesis):
        """Score explicite de best-first : évidence + bits économisés par réutilisation − coût de recherche déjà payé."""
        reuse = (description_bits(h.program) - description_bits(h.program, library=self.library)) * math.log(2) if self.library is not None else 0.0
        return h.evidence + reuse - 0.02 * h.fit_cost

    def _run_best_first(self):
        expanded = set()
        while not self._stop():
            cand = [h for h in self._ranked() if h.key not in expanded and h.depth < self.max_depth]
            if not cand: break
            if self._stalled(): self._log("selection", reason="no_evidence_gain"); break
            h = max(cand[: 2 * self.beam], key=self._score); expanded.add(h.key); self._extend(h, origin="best_first")
            if len(expanded) % 2 == 0: self._compose_cached(limit=2)

    # ---- simplification, sortie ---------------------------------------------------------------
    def _simplify(self):
        """Retire un opérande si l'évidence n'en souffre pas (facteur de Bayes < 3) : contrôle de complexité."""
        best = self._ranked()[0]; changed = True
        while changed and is_composite(best.program):
            changed = False
            for c in best.program.children:
                k = skel_key(c); h = self.hyps.get(k)
                if h is None:
                    h = self._fit(c.skeleton(), torch.tensor([c.param_vector()], dtype=DT), 60, "simplify")
                if h is not None and h.evidence >= best.evidence - math.log(3.0): best = h; changed = True; self._log("pruning", reason="simplified", depth=h.depth); break
        return best

    def run(self, data: FitData, noise: Optional[float] = None) -> SearchOutcome:
        """`noise` : taux de retournement NOMINAL de la vraisemblance adoucie pendant l'ajustement par gradient (défaut
        0,10). Le classement des hypothèses utilise un bruit profilé par hypothèse : aucun bruit vrai n'est requis."""
        if noise is None: noise = 0.10
        self._setup(data, noise); self._log("candidate_generation", strategy=self.strategy, noise=round(self.noise, 4), budget=self.budget)
        self._fit_atoms()
        {"flat": self._run_flat, "bottom_up": self._run_bottom_up, "top_down": self._run_top_down, "bidirectional": self._run_bidirectional, "best_first": self._run_best_first, "factorized": self._run_factorized}[self.strategy]()
        sel = self._simplify(); ranked = self._ranked(); ranked = [sel] + [h for h in ranked if h.key != sel.key]
        ev = torch.tensor([h.evidence for h in ranked], dtype=DT); m = torch.softmax(ev, 0)
        for h, mm in zip(ranked, m.tolist()): h.mass = mm
        for d, st in self.per_depth.items(): st["kept"] = sum(1 for h in ranked if h.depth == d)
        self._log("selection", key=sel.key, depth=sel.depth, evidence=round(sel.evidence, 3))
        stats = {"strategy": self.strategy, "fit_equivalents": round(self.counter.fit_equivalents, 2), "generated": self.generated, "fitted": self.fitted, "pruned": self.pruned,
                 "distinct_structures": len(self.hyps), "families": len({h.family for h in self.hyps.values()}), "seconds": round(time.time() - self.t0, 3), "fit_seconds": round(self.fit_seconds, 3), "synthesis_seconds": round(self.synth_seconds, 3), "compositions": self.compositions,
                 "generation": self.gen_stats[-1] if self.gen_stats else {}, "validated": self.validated, "transactions": len(self.transactions), "commits": sum(t.accepted for t in self.transactions),
                 "rollbacks": sum(not t.accepted for t in self.transactions), "decisions": len(self.decisions), "preserved": sum(d.action == "PRESERVE" for d in self.decisions),
                 "gate_rollbacks": sum(d.rolled_back for d in self.decisions),
                 "per_depth": {str(k): v for k, v in sorted(self.per_depth.items())}, "store": self.store.stats(), "budget_exhausted": self.counter.exhausted()}
        return SearchOutcome(ranked, self.trace, stats, self.noise, self.store, self.preds, dict(self.hyps), self.discovery)

    def _atom_literals(self, data: FitData, n_init, steps, origin):
        """Littéraux par gradient : TOUTES les initialisations ajustées de chaque type d'atome sont gardées (des optimums
        locaux différents correspondent à des atomes différents de la structure)."""
        out = []
        for a in self._atoms():
            if self._stop(): break
            t = time.time(); th, _ = fit_batch(a, data, heuristic_inits(a, data, n_init, self.seed + len(out)), steps=steps, gamma=self.gamma, counter=self.counter); self.fit_seconds += time.time() - t
            self.generated += n_init; out += [a.with_params(row.tolist()) for row in th]
        self._log("substructure_discovery", origin=origin, literals=len(out)); return out

    def _run_factorized(self):
        """Génération factorisée (core.deep_generation) : littéraux (gradient multi-initialisations + géométrie des
        frontières) → synthèse booléenne sur signatures → quelques candidats polis conjointement. Des tours résiduels
        ajoutent les atomes que la meilleure formule n'explique pas encore."""
        from core.deep_generation import generate_candidates
        Xf = self.data.X.float(); self.gen_stats = []; seen = set()
        if self.fact_literals is not None: lits = list(self.fact_literals)                       # littéraux fournis (diagnostic oracle, ou primitives déjà découvertes)
        elif self.fact_discovery is not None:                                                  # PrimitiveDiscovery : proposition + validation WCE
            from core.deep_primitives import discover
            dsc = discover(Xf, self.data.y.long(), self.fact_discovery, seed=self.seed); self.counter.fit_equivalents += dsc.fit_equivalents; self.discovery = dsc
            lits = dsc.primitives.literals(alternatives=2); self._log("substructure_discovery", origin="primitive_discovery", classes=len(dsc.primitives), proposals=dsc.proposals)
        else: lits = self._atom_literals(self.data, self.fact_inits, self.atom_steps, "global")
        if self.fact_separation:
            from core.deep_generation import discover_atoms_by_separation
            more, st = discover_atoms_by_separation(self._atoms(), Xf, self.data.y, seed=self.seed, counter=self.counter); lits += more; self._log("substructure_discovery", origin="separation", literals=len(more))
        for rnd in range(self.fact_rounds + 1):
            t = time.time(); gen = generate_candidates(Xf, self.data.y, max_depth=self.max_depth, top=self.fact_top, seed=self.seed, extra_literals=lits, geometry=self.geometry, time_budget=self.fact_time)
            self.synth_seconds += time.time() - t; self.gen_stats.append(gen.stats); self.compositions += gen.stats.get("attempted", 0) + gen.stats.get("pairs_scored", 0)
            self._log("composition", round=rnd, literals=gen.stats.get("S1"), candidates=len(gen.programs), pairs_scored=gen.stats.get("pairs_scored"))
            order = sorted(range(len(gen.programs)), key=lambda i: -gen.info[i]["score"])
            for i in order[: self.fact_polish]:
                p = gen.programs[i]; k = skel_key(p)
                if self._stop(): break
                if (k, rnd) in seen or k in self.hyps: continue
                seen.add((k, rnd)); h = self._fit(p.skeleton(), torch.tensor([p.param_vector()], dtype=DT), 100, f"factorized_r{rnd}")
                if h is not None and self.fact_feedback: lits += _leaves(h.program)                  # atomes polis EN CONTEXTE (masquage pris en compte) → nouveaux littéraux
            if rnd == self.fact_rounds or self._stop() or not self._ranked(): break
            if not self.fact_residual: continue
            best = self._ranked()[0]; a = self.preds[best.key]
            for comb in COMBINATORS:                                          # atomes manquants : ajustés sur ce que la meilleure structure n'explique pas
                r, w, _ = residual_target(comb, a, self.data.y)
                if float(w.sum()) >= 8: lits += self._atom_literals(self.data.with_target(r, w * self.data.weight), 2, self.atom_steps, f"residual_{comb}")

    def _run_bidirectional(self):
        """Montée courte (niveau 2 par résidu) pour peupler la bibliothèque locale, puis descente avec rencontre."""
        for par in self._ranked(1)[:2]:
            if self._stop(): break
            self._extend(par)
        self._run_top_down()
