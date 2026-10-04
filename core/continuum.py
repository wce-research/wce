"""Système de compétences CONTINU (sans task_id, sans énumération, sans données brutes anciennes pour le système
complet) : Skill = (GenerativeCertificate, invariants, géométrie de stabilité, macro, dépendances).

Planificateur (par lot d'un régime) :
  contexte connu & compétence OK            → réactivation
  contexte connu & écart                     → BRUIT (réparé par nettoyage robuste) | DÉRIVE graduelle (raffinement en
                                               place, sûr au 1er / 2nd ordre) | changement ABRUPT (nouvelle compétence)
  contexte nouveau / abrupt                  → réutilisation > composition > règle découverte > adaptateur > nouveau module
Puis certificats (génératifs ou sondes stockées), gestion de capacité (élagage, fusion, distillation sous budget),
régénération sans instantané. Neuf méthodes comparées (4 monolithiques + 5 configurations du système).
"""
from __future__ import annotations

import copy
import itertools
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from core.continuous_world import D, D_CUE, OPS, Rule, rule_eval, superseded
from core.curvature import KEYS, actual_flips, flat, mlp, null_projector, predicted_margins, safe_step, unflat, ellipsoid
from core.generative_certificate import (ContinuousGenerator, GenerativeInvariantCertificate, StoredProbeCertificate, TinySurrogate,
                                         clean_inputs, FLOAT_BYTES)

H = 16


def new_expert(seed, h=H, d=D, out=2):
    g = torch.Generator().manual_seed(seed)
    return {"w1": torch.randn(h, d, generator=g) * (1.0 / d) ** 0.5, "b1": torch.zeros(h),
            "w2": torch.randn(out, h, generator=g) * (1.0 / h) ** 0.5, "b2": torch.zeros(out)}


def nparams(p): return sum(v.numel() for v in p.values())


def train(p, X, y, steps=200, lr=0.03, seed=0, project=None):
    torch.manual_seed(seed); q = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    opt = torch.optim.Adam(q.values(), lr=lr) if project is None else torch.optim.SGD(q.values(), lr=0.1, momentum=0.9)
    Xc = clean_inputs(X)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.cross_entropy(mlp(q, Xc), y).backward()
        if project is not None:
            gv = project(torch.cat([q[k].grad.flatten() for k in KEYS])); gd = unflat(gv, q)
            for k in KEYS: q[k].grad = gd[k]
        opt.step()
    return {k: v.detach() for k, v in q.items()}


# ================================================================ découverte de règles (bibliothèque de familles)

def fit_halfspace(X, y, steps=300):
    w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam([w, b], lr=0.1)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(X @ w - b, y.float()).backward(); opt.step()
    n = float(w.detach().norm()) or 1.0
    return Rule("halfspace", (tuple((w / n).tolist()), float(b) / n))


def fit_ellipsoid(X, y, steps=400):
    c = X[y == 1].mean(0).clone().requires_grad_(True) if int(y.sum()) else torch.zeros(D, requires_grad=True)
    a = torch.zeros(D, requires_grad=True); r = torch.ones(1, requires_grad=True); opt = torch.optim.Adam([c, a, r], lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); w = torch.nn.functional.softplus(a)
        s = r ** 2 - ((X - c) ** 2 * w).sum(1)
        (torch.nn.functional.binary_cross_entropy_with_logits(4 * s, y.float()) + 1e-3 * w.sum()).backward(); opt.step()
    return Rule("ellipsoid", (tuple(torch.nn.functional.softplus(a).tolist()), tuple(c.tolist()), float(r) ** 2))


def fit_sector(X, y):
    best = None
    for i, j in itertools.combinations(range(D), 2):
        for (a, b) in ((i, j), (j, i)):
            ang = torch.atan2(X[y == 1][:, b], X[y == 1][:, a])
            if len(ang) == 0: continue
            th = float(torch.atan2(torch.sin(ang).mean(), torch.cos(ang).mean()))
            r = Rule("sector", (a, b, th, math.pi / 2)); acc = float((rule_eval(r, X) == y).float().mean())
            if best is None or acc > best[0]: best = (acc, r)
    return best[1]


def fit_box(X, y):
    best = None; P = X[y == 1]
    if len(P) < 4: return Rule("box", (0, 1, 0.0, 0.0))
    for i, j in itertools.combinations(range(D), 2):
        a, b = float(P[:, i].abs().quantile(0.98)), float(P[:, j].abs().quantile(0.98))
        r = Rule("box", (i, j, a, b)); acc = float((rule_eval(r, X) == y).float().mean())
        if best is None or acc > best[0]: best = (acc, r)
    return best[1]


_orig_eval = rule_eval


def _eval_ext(r, X):
    if r.family == "ellipsoid":
        X = clean_inputs(X); w, c, rr = torch.tensor(r.params[0]), torch.tensor(r.params[1]), r.params[2]
        return (((X - c) ** 2 * w).sum(1) < rr).long()
    return _orig_eval(r, X)


import core.continuous_world as _cw
_cw.rule_eval = _eval_ext
Rule.__call__ = lambda self, X: _eval_ext(self, X)


def discover_rule(X, y, Xv, yv, target):
    """Recherche bornée sur les familles d'hypothèses (demi-espace, ellipsoïde, secteur, boîte) ; renvoie la
    meilleure règle et son exactitude de validation (mémorisation vs règle compositionnelle)."""
    Xc, Xvc = clean_inputs(X), clean_inputs(Xv); best = None; evals = 0
    for fit in (fit_halfspace, fit_ellipsoid, fit_sector, fit_box):
        r = fit(Xc, y); evals += 1; acc = float((r(Xvc) == yv).float().mean())
        if best is None or acc > best[1]: best = (r, acc)
        if acc >= target + 0.02: break
    return best[0], best[1], evals


# ================================================================ système

@dataclass
class Config:
    name: str = "full"
    cert: str = "generative"              # none | stored | generative
    reuse: bool = True
    compose: bool = True
    adapter: bool = True
    discovery: bool = True
    drift: str = "auto"                   # regrow | refine1 | refine2 | auto (2nd ordre + détection bruit/dérive)
    capacity: Optional[int] = None        # budget (paramètres neuronaux) ; None = illimité
    capacity_mgmt: bool = True
    target: float = 0.93
    steps: int = 300
    robust: bool = True                   # nettoyage robuste (NaN, outliers) avant décision


@dataclass
class Ctx:
    name: str
    proto: torch.Tensor
    count: int
    skill: Optional[str] = None
    history: List[float] = field(default_factory=list)
    ref_acc: float = 1.0                  # exactitude de validation de la compétence à son acquisition / raffinement


class ContinuumSystem:
    def __init__(self, cfg: Config, seed=0):
        self.cfg, self.seed = cfg, seed
        self.modules: Dict[str, dict] = {}; self.skills: Dict[str, tuple] = {}; self.contexts: List[Ctx] = []
        self.certs: Dict[str, object] = {}; self.log: List[dict] = []
        self.counters = Counter(); self.compute = Counter(); self.macros = Counter()
        self.g = torch.Generator().manual_seed(seed)

    # ---- exécution des compétences -------------------------------------------------------------
    def logits(self, skill, X):
        prog = self.skills[skill]; k = prog[0]; Xc = clean_inputs(X)
        if k == "expert": return mlp(self.modules[prog[1]], Xc)
        if k == "head":
            p = self.modules[prog[1]]; o = mlp(p, Xc); i = prog[2]; return o[:, 2 * i: 2 * i + 2]
        if k == "adapter":
            p, a = self.modules[prog[1]], self.modules[prog[2]]
            return mlp({"w1": p["w1"] + a["u"] @ a["v"], "b1": p["b1"] + a["db1"], "w2": p["w2"] + a["dw2"], "b2": p["b2"] + a["db2"]}, Xc)
        y = self.predict_skill(skill, X); return torch.stack([(1 - y).float(), y.float()], 1) * 10.0

    def predict_skill(self, skill, X):
        prog = self.skills[skill]; k = prog[0]
        if k == "rule": return prog[1](X)
        if k == "not": return 1 - self.predict_skill(prog[1], X)
        if k == "compose":
            a, b = self.predict_skill(prog[2], X), self.predict_skill(prog[3], X)
            return (a * b) if prog[1] == "and" else ((a + b) > 0).long() if prog[1] == "or" else (a != b).long()
        with torch.no_grad(): return self.logits(skill, X).argmax(1)

    def modules_of(self, skill, seen=None):
        seen = seen or set()
        if skill in seen or skill not in self.skills: return set()
        seen.add(skill); p = self.skills[skill]
        if p[0] in ("expert", "head"): return {p[1]}
        if p[0] == "adapter": return {p[1], p[2]}
        if p[0] == "not": return self.modules_of(p[1], seen)
        if p[0] == "compose": return self.modules_of(p[2], seen) | self.modules_of(p[3], seen)
        return set()

    def dependents(self, module): return [s for s in self.skills if module in self.modules_of(s)]
    def neural_params(self): return sum(nparams(p) for p in self.modules.values())
    def symbolic_floats(self): return sum(p[1].size() for p in self.skills.values() if p[0] == "rule")

    # ---- routeur ---------------------------------------------------------------------------------
    def route(self, C):
        if not self.contexts: return torch.full((len(C),), -1, dtype=torch.long)
        P = torch.stack([c.proto for c in self.contexts]); return torch.cdist(C, P).argmin(1)

    def discover(self, C, thr=1.0):
        m = C.mean(0)
        if self.contexts:
            d = torch.stack([(c.proto - m).norm() for c in self.contexts]); i = int(d.argmin())
            if float(d[i]) <= thr:
                c = self.contexts[i]; c.proto = (c.proto * c.count + C.sum(0)) / (c.count + len(C)); c.count += len(C); return c, False
        c = Ctx(f"ctx{len(self.contexts)}", m, len(C)); self.contexts.append(c); return c, True

    def predict(self, X, C):
        r = self.route(C); out = torch.zeros(len(X), dtype=torch.long)
        for ci in r.unique().tolist():
            m = r == ci
            if ci >= 0 and self.contexts[ci].skill is not None: out[m] = self.predict_skill(self.contexts[ci].skill, X[m])
        return out

    def confidence(self, X, C):
        r = self.route(C); out = torch.full((len(X),), 0.5)
        for ci in r.unique().tolist():
            m = r == ci; s = self.contexts[ci].skill if ci >= 0 else None
            if s is None: continue
            with torch.no_grad(): out[m] = torch.softmax(self.logits(s, X[m]), 1).max(1).values
        return out

    # ---- certificats -----------------------------------------------------------------------------
    def certify(self, skill, X, t):
        if self.cfg.cert == "none": return
        if self.cfg.cert == "stored":
            idx = torch.randperm(len(X), generator=self.g)[:24]; P = clean_inputs(X[idx])
            self.certs[skill] = StoredProbeCertificate(skill, P, self.predict_skill(skill, P), dependencies=tuple(self.modules_of(skill))); return
        gen = ContinuousGenerator().fit(X); prog = self.skills[skill]
        if prog[0] == "rule": ref, kind, fl = prog[1], "rule", prog[1].size()
        elif prog[0] in ("not", "compose") and all(self.skills.get(s, ("",))[0] == "rule" for s in _children(prog)):
            ref, kind, fl = (lambda X, s=skill, snap=dict(self.skills): _eval_prog(snap, s, X)), "program", 4
        else:
            sur = TinySurrogate(gen, lambda Z, s=skill: self.predict_skill(s, Z), D, seed=self.seed + len(self.certs))
            self.compute["generation"] += 5300; self.compute["surrogate_train"] += 1200; ref, kind, fl = sur, "surrogate", sur.floats()
        self.certs[skill] = GenerativeInvariantCertificate(skill, "continuous", gen, ref, kind, fl, provenance=f"regime@{t}",
                                                           dependencies=tuple(sorted(self.modules_of(skill))), created_at=t)
        self.certs[skill].calibrate(lambda Z, s=skill: self.predict_skill(s, Z)); self.compute["verification"] += 3 * 96 * 4

    def check_certs(self, skills, n=32):
        ok = True
        for s in skills:
            c = self.certs.get(s)
            if c is None or s not in self.skills: continue
            o, _ = c.check(lambda X, s=s: self.predict_skill(s, X), n=n, seed=self.counters["checks"])
            self.counters["checks"] += 1; self.compute["verification"] += n * 4
            ok &= o
        return ok

    # ---- apprentissage d'un lot ------------------------------------------------------------------
    def learn(self, X, C, y, t=0):
        cfg = self.cfg; ctx, new = self.discover(C); self.counters["t"] = t
        n = len(X); cut = int(0.75 * n); Xt, yt, Xv, yv = X[:cut], y[:cut], X[cut:], y[cut:]
        rec = {"t": t, "context": ctx.name, "new_context": new, "kind": None, "steps": 0, "params_before": self.neural_params()}
        steps0 = self.compute["train"]
        if not new and ctx.skill is not None:
            acc = float((self.predict_skill(ctx.skill, Xv) == yv).float().mean()); rec["acc_before"] = acc
            if acc >= min(cfg.target - 0.03, ctx.ref_acc - 0.06):          # écart relatif à la référence de la compétence
                rec["kind"] = "reactivate"; ctx.history.append(acc); return self._done(rec, steps0)
            regime = self.classify_change(ctx, X, y, acc)
            rec["change"] = regime
            if regime == "noise": rec["kind"] = "noise_ignored"; return self._done(rec, steps0)
            if regime == "drift" and cfg.drift != "regrow":
                if self.refine(ctx, Xt, yt, Xv, yv, rec): return self._done(rec, steps0)
        self.acquire(ctx, Xt, yt, Xv, yv, X, y, rec, t)
        if cfg.capacity is not None and cfg.capacity_mgmt: self.manage_capacity(rec)
        return self._done(rec, steps0)

    def _done(self, rec, steps0):
        rec["steps"] = self.compute["train"] - steps0; rec["params_after"] = self.neural_params(); self.log.append(rec); return rec

    def classify_change(self, ctx, X, y, acc):
        """Bruit vs vrai changement de régime : si l'exactitude se rétablit sur les exemples PROPRES (sans NaN ni
        outliers ni dimension corrompue détectable), c'est du bruit ; sinon dérive graduelle (≥ 0,6) ou abrupte."""
        if self.cfg.drift == "auto" and self.cfg.robust:
            Xc = X.clone(); bad = torch.isnan(Xc).any(1) | (Xc.abs() > 4.5).any(1)
            if int((~bad).sum()) > 30:
                acc_clean = float((self.predict_skill(ctx.skill, Xc[~bad]) == y[~bad]).float().mean())
                if acc_clean >= min(self.cfg.target - 0.03, ctx.ref_acc - 0.06): return "noise"
        return "drift" if (ctx.ref_acc - acc) <= 0.3 and acc >= 0.6 else "abrupt"

    def refine(self, ctx, Xt, yt, Xv, yv, rec):
        """Raffinement EN PLACE d'une compétence dérivée. Règle : réajustement des paramètres symboliques.
        Expert : gradient de la nouvelle perte, projeté (1er ordre) hors des directions sensibles des compétences
        DÉPENDANTES, puis amplitude contrôlée au 2nd ordre (refine2 / auto). Adaptateur : seul l'adaptateur bouge."""
        s = ctx.skill; prog = self.skills[s]; cfg = self.cfg
        if prog[0] == "rule":
            r, acc, ev = discover_rule(Xt, yt, Xv, yv, cfg.target); self.compute["search"] += ev * len(Xt)
            if acc >= cfg.target:
                self.skills[s] = ("rule", r); ctx.ref_acc = acc; self.certify(s, torch.cat([Xt, Xv]), rec["t"]); self.counters["refine"] += 1
                rec.update(kind="refine_rule", acc=acc); return True
            return False
        if prog[0] == "adapter":                                   # adaptateur : seul l'adaptateur bouge (base partagée intacte)
            base, an = prog[1], prog[2]; old = self.modules[an]
            self.modules[an] = train_adapter(self.modules[base], Xt, yt, cfg.steps, self.seed, init=old); self.compute["train"] += cfg.steps
            acc = float((self.predict_skill(s, Xv) == yv).float().mean()); rec["refine_gain"] = acc - rec.get("acc_before", 0.0)
            if acc >= rec.get("acc_before", 0.0) + 0.05 or acc >= cfg.target - 0.03:
                ctx.ref_acc = acc; self.certify(s, torch.cat([Xt, Xv]), rec["t"]); self.counters["refine"] += 1; rec.update(kind="refine_adapter", acc=acc); return True
            self.modules[an] = old; self.counters["rollback"] += 1; return False
        if prog[0] != "expert": return False
        m = prog[1]; deps = [d for d in self.dependents(m) if d != s]
        p0 = self.modules[m]; project = None; probes = None
        if deps and self.cfg.cert == "generative":
            probes = torch.cat([self.certs[d].generator.sample(48, self.g) for d in deps if d in self.certs])
            yp = torch.cat([self.predict_skill(d, self.certs[d].generator.sample(0, self.g)) for d in []] or [torch.zeros(0, dtype=torch.long)])
        if probes is not None and len(probes):
            yp = self.predict_skill(s, probes)      # comportement actuel du module sur la région des dépendants
            project, _ = null_projector(p0, probes, yp)
        new = train(p0, Xt, yt, cfg.steps, seed=self.seed, project=project); self.compute["train"] += cfg.steps
        delta = flat(new) - flat(p0)
        if probes is not None and len(probes) and cfg.drift in ("refine2", "auto"):
            a, _ = safe_step(p0, probes, yp, delta, max_flip=0.02, order=2); self.compute["curvature"] += 2 * len(probes) * 12
            delta = a * delta; rec["second_order_scale"] = a
        cand = unflat(flat(p0) + delta, p0); old = self.modules[m]; self.modules[m] = cand
        acc = float((self.predict_skill(s, Xv) == yv).float().mean()); rec["refine_gain"] = acc - rec.get("acc_before", 0.0)
        if (acc >= cfg.target - 0.03 or acc >= rec.get("acc_before", 0.0) + 0.05) and self.check_certs(deps):
            ctx.ref_acc = acc; self.certify(s, torch.cat([Xt, Xv]), rec["t"]); self.counters["refine"] += 1; rec.update(kind="refine", acc=acc); return True
        self.modules[m] = old; self.counters["rollback"] += 1; return False

    def acquire(self, ctx, Xt, yt, Xv, yv, X, y, rec, t):
        cfg = self.cfg; cands = []
        accv = lambda pred: float((pred == yv).float().mean())
        if cfg.reuse and self.skills:
            for s in list(self.skills):
                pv = self.predict_skill(s, Xv); a = accv(pv); self.compute["search"] += len(Xv)
                cands.append(("reuse", a, 0, lambda s=s: s))
                if 1 - a > a: cands.append(("reuse", 1 - a, 0, lambda s=s: self._add(f"not({s})", ("not", s))))
        best = max([c[1] for c in cands], default=0)
        if cfg.compose and best < cfg.target and len(self.skills) >= 2:
            names = list(self.skills)[-30:]; P = {s: self.predict_skill(s, Xv) for s in names}
            for a, b in itertools.combinations(names, 2):
                for op in OPS:
                    pa, pb = P[a], P[b]; pr = (pa * pb) if op == "and" else ((pa + pb) > 0).long() if op == "or" else (pa != pb).long()
                    ac = accv(pr); self.compute["search"] += len(Xv)
                    if ac >= cfg.target: cands.append(("compose", ac, 0, lambda a=a, b=b, op=op: self._add(f"{op}({a},{b})", ("compose", op, a, b))))
        best = max([c[1] for c in cands], default=0)
        if cfg.discovery and best < cfg.target:
            r, ac, ev = discover_rule(Xt, yt, Xv, yv, cfg.target); self.compute["search"] += ev * len(Xt) * 3
            cands.append(("rule", ac, 0, lambda r=r: self._add(f"rule_{ctx.name}_{t}", ("rule", r))))
        best = max([c[1] for c in cands], default=0)
        if cfg.adapter and best < cfg.target:
            bases = [s for s, p in self.skills.items() if p[0] == "expert"]
            if bases:
                b = max(bases, key=lambda s: max(accv(self.predict_skill(s, Xv)), 0)); m = self.skills[b][1]
                ad = train_adapter(self.modules[m], Xt, yt, cfg.steps, self.seed); self.compute["train"] += cfg.steps
                tmp = f"adp_{ctx.name}_{t}"; self.modules[tmp] = ad; self.skills["__tmp"] = ("adapter", m, tmp)
                ac = accv(self.predict_skill("__tmp", Xv)); self.skills.pop("__tmp"); self.modules.pop(tmp)
                cands.append(("adapter", ac, nparams(ad), lambda m=m, ad=ad, tmp=tmp: (self.modules.__setitem__(tmp, ad), self._add(f"skill_{tmp}", ("adapter", m, tmp)))[1]))
        best = max([c[1] for c in cands], default=0)
        if best < cfg.target or not cands:
            name = f"exp_{ctx.name}_{t}"; p = train(new_expert(self.seed + t), Xt, yt, cfg.steps, seed=self.seed); self.compute["train"] += cfg.steps
            ac = accv(self.predict_fn_params(p, Xv)); p = train(p, torch.cat([Xt, Xv]), torch.cat([yt, yv]), cfg.steps // 4, seed=self.seed); self.compute["train"] += cfg.steps // 4
            cands.append(("grow", ac, nparams(p), lambda name=name, p=p: (self.modules.__setitem__(name, p), self._add(f"skill_{name}", ("expert", name)))[1]))
        # coût : erreur + paramètres ; à exactitude ≥ cible, préférer reuse > compose > rule > adapter > grow
        order = {"reuse": 0, "compose": 1, "rule": 2, "adapter": 3, "grow": 4}
        good = [c for c in cands if c[1] >= cfg.target]
        pick = min(good, key=lambda c: (order[c[0]], -c[1])) if good else max(cands, key=lambda c: (c[1] - 0.00005 * c[2]))
        skill = pick[3](); ctx.skill = skill; ctx.ref_acc = pick[1]; self.counters[pick[0]] += 1; self.macros[pick[0]] += 1
        rec.update(kind=pick[0], acc=pick[1], params_added=pick[2])
        if skill not in self.certs: self.certify(skill, X, t)

    def predict_fn_params(self, p, X):
        with torch.no_grad(): return mlp(p, clean_inputs(X)).argmax(1)

    def _add(self, name, prog):
        if name not in self.skills: self.skills[name] = prog
        return name

    # ---- capacité : élagage, fusion, distillation -------------------------------------------------
    def manage_capacity(self, rec):
        budget = self.cfg.capacity; rep = Counter()
        if self.neural_params() <= budget: return
        used = {s for c in self.contexts if c.skill for s in [c.skill]}
        live = set()
        def mark(s):
            if s in live or s not in self.skills: return
            live.add(s); p = self.skills[s]
            if p[0] == "not": mark(p[1])
            if p[0] == "compose": mark(p[2]); mark(p[3])
        for s in used: mark(s)
        for s in [s for s in self.skills if s not in live]: self.skills.pop(s); self.certs.pop(s, None); rep["prune_skill"] += 1
        mods_used = set().union(*[self.modules_of(s) for s in self.skills]) if self.skills else set()
        for m in [m for m in self.modules if m not in mods_used]: self.modules.pop(m); rep["prune"] += 1; self.counters["prune"] += 1
        if self.neural_params() > budget: self._merge(rep)
        if self.neural_params() > budget: self._distill_symbolic(rep, budget)
        if self.neural_params() > budget and self.counters["t"] >= self.counters["distill_cooldown_until"]:
            before = self.counters["distill_rejected"]; self._distill(rep)
            if self.counters["distill_rejected"] > before: self.counters["distill_cooldown_until"] = self.counters["t"] + 5   # pas de nouvel essai avant 5 régimes
        rec["capacity"] = dict(rep)

    def _merge(self, rep):
        """Fusion de compétences atomiques ÉQUIVALENTES sur leurs régions génératives (sorties ≥ 97 % identiques
        sur les échantillons des deux certificats) et sans conflit de dépendances ; validée par les certificats."""
        atoms = [s for s, p in self.skills.items() if p[0] in ("expert", "adapter")]
        for a, b in itertools.combinations(atoms, 2):
            if a not in self.skills or b not in self.skills: continue
            ca, cb = self.certs.get(a), self.certs.get(b)
            if not isinstance(ca, GenerativeInvariantCertificate) or not isinstance(cb, GenerativeInvariantCertificate): continue
            Z = torch.cat([ca.generator.sample(64, self.g), cb.generator.sample(64, self.g)])
            if float((self.predict_skill(a, Z) == self.predict_skill(b, Z)).float().mean()) < 0.97: continue
            saved = (dict(self.skills), [c.skill for c in self.contexts])
            for c in self.contexts:
                if c.skill == b: c.skill = a
            for s, p in list(self.skills.items()):
                if p[0] in ("not",) and p[1] == b: self.skills[s] = ("not", a)
                if p[0] == "compose": self.skills[s] = tuple(a if x == b else x for x in p)
            self.skills.pop(b)
            if self.check_certs([s for s in self.certs if s in self.skills]) and self.certs[b].check(lambda X: self.predict_skill(a, X))[0]:
                self.certs[a].dependencies = tuple(sorted(set(self.certs[a].dependencies)))
                self.certs.pop(b); rep["merge"] += 1; self.counters["merge"] += 1
                for m in [m for m in self.modules if not any(m in self.modules_of(s) for s in self.skills)]: self.modules.pop(m)
            else:
                self.skills = saved[0]
                for c, s in zip(self.contexts, saved[1]): c.skill = s

    def _distill(self, rep, hidden=None, steps=2500, n=1500):
        """Distille plusieurs experts vers UN module multi-têtes, entraîné sur des échantillons GÉNÉRÉS par leurs
        certificats (étiquettes = références) ; essais par groupes de 6 puis 4 puis 3 ; conservé seulement si TOUS
        les certificats passent (sinon rollback)."""
        for k in (6, 4, 3):
            experts = [s for s, p in self.skills.items() if p[0] == "expert" and not self.dependents_other(s)
                       and isinstance(self.certs.get(s), GenerativeInvariantCertificate)][:k]
            if len(experts) < k: continue
            Xs, Ys = [], []
            for s in experts:
                c = self.certs[s]; Z = torch.cat([c.generator.sample(n, self.g), c.generator.sample_boundary(n // 3, self.g, c.reference), c.generator.sample_rare(n // 6, self.g)])
                Xs.append(Z); Ys.append(c.reference(Z))
            self.compute["generation"] += sum(len(x) for x in Xs)
            h = hidden or (10 * k + 8)                                  # ≈ 10 unités par compétence (vs 16 par expert)
            p = new_expert(self.seed + 777, h, D, 2 * len(experts)); q = {kk: v.clone().requires_grad_(True) for kk, v in p.items()}
            opt = torch.optim.Adam(q.values(), lr=0.02)
            for _ in range(steps):
                opt.zero_grad(); loss = 0.0
                for i, (Z, Y) in enumerate(zip(Xs, Ys)): loss = loss + torch.nn.functional.cross_entropy(mlp(q, Z)[:, 2 * i: 2 * i + 2], Y)
                loss.backward(); opt.step()
            self.compute["train"] += steps
            name = f"multi_{len(self.modules)}_{self.counters['distill']}"; before = self.neural_params(); saved = dict(self.skills)
            old_mods = {self.skills[s][1]: self.modules[self.skills[s][1]] for s in experts}
            self.modules[name] = {kk: v.detach() for kk, v in q.items()}
            for i, s in enumerate(experts): self.skills[s] = ("head", name, i)
            for m in old_mods: self.modules.pop(m)
            if self.check_certs([s for s in self.certs if s in self.skills]):
                rep["distill"] += 1; self.counters["distill"] += 1; self.counters["distill_params_saved"] += before - self.neural_params()
                self.counters["distill_before_params"] += sum(nparams(v) for v in old_mods.values()); self.counters["distill_after_params"] += nparams(self.modules[name])
                return
            self.skills = saved; self.modules.pop(name); self.modules.update(old_mods); self.counters["distill_rejected"] += 1

    def _distill_symbolic(self, rep, budget):
        """Distillation SYMBOLIQUE : un expert est remplacé par une règle de la bibliothèque ajustée sur des
        échantillons GÉNÉRÉS (étiquettes = référence du certificat), seulement si TOUS les certificats passent."""
        for s in [s for s, p in self.skills.items() if p[0] == "expert" and not self.dependents_other(s)]:
            if self.neural_params() <= budget: return
            c = self.certs.get(s)
            if not isinstance(c, GenerativeInvariantCertificate): continue
            Z = torch.cat([c.generator.sample(900, self.g), c.generator.sample_boundary(300, self.g, c.reference)]); Y = c.reference(Z)
            Zv = c.generator.sample(300, self.g); Yv = c.reference(Zv); self.compute["generation"] += 1500
            r, acc, ev = discover_rule(Z, Y, Zv, Yv, 0.97); self.compute["search"] += ev * len(Z)
            if acc < 0.95: continue
            saved, m = self.skills[s], self.skills[s][1]; mod = self.modules.pop(m); self.skills[s] = ("rule", r)
            if self.check_certs([s]): rep["distill_symbolic"] += 1; self.counters["distill_symbolic"] += 1
            else: self.skills[s] = saved; self.modules[m] = mod

    def dependents_other(self, s):
        m = self.skills[s][1]; return [d for d in self.dependents(m) if d != s]

    # ---- régénération 2.0 --------------------------------------------------------------------------
    def regenerate(self, module, steps=1500, n=2400):
        """Reconstruction d'un module corrompu SANS instantané : échantillons générés par les certificats de toutes
        les compétences qui en dépendent (centre + frontière + rares), étiquetés par leurs références (règle ou
        substitut), réajustement conjoint en partant des poids corrompus ; vérification par certificats."""
        deps = [s for s in self.dependents(module) if isinstance(self.certs.get(s), GenerativeInvariantCertificate)
                and self.skills[s][0] in ("expert", "head", "adapter")]
        if not deps: return False
        cur = self.modules[module]; h, d = cur["w1"].shape; o = cur["w2"].shape[0]
        fresh = new_expert(self.seed + 991, h, d, o)                         # poids corrompus abandonnés (tanh saturé) : réinitialisation
        q = {k: v.clone().requires_grad_(True) for k, v in fresh.items()}; opt = torch.optim.Adam(q.values(), lr=0.02)
        data = []
        for s in deps:
            c = self.certs[s]; Z = torch.cat([c.generator.sample(n, self.g), c.generator.sample_boundary(n // 3, self.g, c.reference), c.generator.sample_rare(n // 6, self.g)])
            data.append((s, Z, c.reference(Z)))
        for _ in range(steps):
            opt.zero_grad(); loss = 0.0
            for s, Z, Y in data:
                prog = self.skills[s]
                if prog[0] == "expert": o = mlp(q, Z)
                elif prog[0] == "head": o = mlp(q, Z)[:, 2 * prog[2]: 2 * prog[2] + 2]
                else:
                    a = self.modules[prog[2]]; o = mlp({"w1": q["w1"] + a["u"] @ a["v"], "b1": q["b1"] + a["db1"], "w2": q["w2"] + a["dw2"], "b2": q["b2"] + a["db2"]}, Z)
                loss = loss + torch.nn.functional.cross_entropy(o, Y)
            loss.backward(); opt.step()
        self.modules[module] = {k: v.detach() for k, v in q.items()}; self.compute["train"] += steps; self.compute["generation"] += len(deps) * n * 3 // 2
        return self.check_certs(deps)

    # ---- comptabilité mémoire -----------------------------------------------------------------------
    def memory(self):
        cert = sum(c.memory_bytes() for c in self.certs.values())
        return {"neural_params": self.neural_params(), "neural_bytes": FLOAT_BYTES * self.neural_params(),
                "symbolic_bytes": FLOAT_BYTES * self.symbolic_floats(), "certificate_bytes": cert,
                "graph_bytes": 24 * len(self.skills) + FLOAT_BYTES * (D_CUE + 2) * len(self.contexts), "macro_bytes": 16 * len(self.macros),
                "raw_data_bytes": 0}


def _children(prog):
    if prog[0] == "not": return [prog[1]]
    if prog[0] == "compose": return [prog[2], prog[3]]
    return []


def _eval_prog(skills, s, X):
    p = skills[s]
    if p[0] == "rule": return p[1](X)
    if p[0] == "not": return 1 - _eval_prog(skills, p[1], X)
    a, b = _eval_prog(skills, p[2], X), _eval_prog(skills, p[3], X)
    return (a * b) if p[1] == "and" else ((a + b) > 0).long() if p[1] == "or" else (a != b).long()


def train_adapter(p, X, y, steps, seed, rank=2, init=None):
    g = torch.Generator().manual_seed(seed); h = p["w1"].shape[0]
    a = init if init is not None else {"u": torch.randn(h, rank, generator=g) * 0.1, "v": torch.randn(rank, D, generator=g) * 0.1, "db1": torch.zeros(h),
         "dw2": torch.zeros_like(p["w2"]), "db2": torch.zeros(2)}
    q = {k: v.clone().requires_grad_(True) for k, v in a.items()}; opt = torch.optim.Adam(q.values(), lr=0.05); Xc = clean_inputs(X)
    for _ in range(steps):
        opt.zero_grad()
        w = {"w1": p["w1"] + q["u"] @ q["v"], "b1": p["b1"] + q["db1"], "w2": p["w2"] + q["dw2"], "b2": p["b2"] + q["db2"]}
        torch.nn.functional.cross_entropy(mlp(w, Xc), y).backward(); opt.step()
    return {k: v.detach() for k, v in q.items()}


# ================================================================ bases monolithiques

MONO_H = 64


def mono_init(seed):
    g = torch.Generator().manual_seed(seed); d = D + D_CUE
    return {"w1": torch.randn(MONO_H, d, generator=g) * (1.0 / d) ** 0.5, "b1": torch.zeros(MONO_H),
            "w2": torch.randn(2, MONO_H, generator=g) * (1.0 / MONO_H) ** 0.5, "b2": torch.zeros(2)}


class Monolithic:
    """vanilla | raw_replay | generative_replay | first_order (projection espace nul aux sondes générées)."""

    def __init__(self, kind, seed=0, steps=200):
        self.kind, self.seed, self.steps = kind, seed, steps; self.p = mono_init(seed); self.buf = []; self.gens = []; self.basis = None
        self.compute = Counter(); self.raw_bytes = 0; self.gen_bytes = 0; self.g = torch.Generator().manual_seed(seed)

    def Z(self, X, C): return torch.cat([clean_inputs(X), C], 1)

    def learn(self, X, C, y, t=0):
        Z = self.Z(X, C); extra = None
        if self.kind == "raw_replay" and self.buf: extra = (torch.cat([b[0] for b in self.buf]), torch.cat([b[1] for b in self.buf]))
        if self.kind == "generative_replay" and self.gens:
            Zs = torch.cat([gz.sample(48, self.g) for gz in self.gens])
            with torch.no_grad(): extra = (Zs, mlp(self.p, Zs).argmax(1))
            self.compute["generation"] += len(Zs)
        q = {k: v.clone().requires_grad_(True) for k, v in self.p.items()}
        opt = torch.optim.Adam(q.values(), lr=0.02) if self.basis is None else torch.optim.SGD(q.values(), lr=0.1, momentum=0.9)
        for _ in range(self.steps):
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(mlp(q, Z), y)
            if extra is not None: loss = loss + torch.nn.functional.cross_entropy(mlp(q, extra[0]), extra[1])
            loss.backward()
            if self.basis is not None:
                gv = torch.cat([q[k].grad.flatten() for k in KEYS]); gv = gv - self.basis @ (self.basis.T @ gv); gd = unflat(gv, q)
                for k in KEYS: q[k].grad = gd[k]
            opt.step()
        self.p = {k: v.detach() for k, v in q.items()}; self.compute["train"] += self.steps
        if self.kind == "raw_replay":
            self.buf.append((Z[:32], y[:32])); self.raw_bytes += 32 * (Z.shape[1] * FLOAT_BYTES + 1)
        if self.kind in ("generative_replay", "first_order"):
            gz = ContinuousGenerator().fit(Z); self.gens.append(gz); self.gen_bytes += gz.memory_bytes()
        if self.kind == "first_order":
            P = gz.sample(24, self.g)
            with torch.no_grad(): yp = mlp(self.p, P).argmax(1)
            from core.curvature import jacobian
            J = jacobian(self.p, P, yp)
            if self.basis is not None: J = J - (J @ self.basis) @ self.basis.T
            U, S, Vh = torch.linalg.svd(J, full_matrices=False); r = int((torch.cumsum(S ** 2, 0) / (S ** 2).sum() < 0.99).sum()) + 1
            B = Vh[:r].T; self.basis = B if self.basis is None else torch.linalg.qr(torch.cat([self.basis, B], 1))[0]
        return {"kind": "train", "steps": self.steps}

    def predict(self, X, C):
        with torch.no_grad(): return mlp(self.p, self.Z(X, C)).argmax(1)

    def confidence(self, X, C):
        with torch.no_grad(): return torch.softmax(mlp(self.p, self.Z(X, C)), 1).max(1).values

    def memory(self):
        nb = sum(v.numel() for v in self.p.values())
        return {"neural_params": nb, "neural_bytes": FLOAT_BYTES * nb, "symbolic_bytes": 0, "certificate_bytes": self.gen_bytes,
                "graph_bytes": 0, "macro_bytes": 0, "raw_data_bytes": self.raw_bytes,
                "basis_bytes": 0 if self.basis is None else FLOAT_BYTES * self.basis.numel()}


METHODS = {
    "vanilla": ("mono", "vanilla"), "raw_replay": ("mono", "raw_replay"), "generative_replay": ("mono", "generative_replay"),
    "first_order_constraints": ("mono", "first_order"),
    "structural_growth": ("sys", Config("structural_growth", cert="stored", reuse=False, compose=False, adapter=False, discovery=False, drift="regrow", capacity_mgmt=False, robust=False)),
    "compositional_old": ("sys", Config("compositional_old", cert="stored", discovery=False, drift="regrow", capacity_mgmt=False, robust=False)),
    "generative_certificates": ("sys", Config("generative_certificates", cert="generative", discovery=False, drift="refine1", capacity_mgmt=False, robust=False)),
    "generative_second_order": ("sys", Config("generative_second_order", cert="generative", discovery=False, drift="refine2", capacity_mgmt=False, robust=False)),
    "full": ("sys", Config("full", cert="generative", discovery=True, drift="auto", capacity_mgmt=True, robust=True)),
}


def make(method, seed, capacity=None):
    kind, spec = METHODS[method]
    if kind == "mono": return Monolithic(spec, seed)
    cfg = copy.deepcopy(spec); cfg.capacity = capacity
    return ContinuumSystem(cfg, seed)


def ece(conf, correct, bins=10):
    e = 0.0
    for b in range(bins):
        m = (conf >= 0.5 + b * 0.05) & (conf < 0.5 + (b + 1) * 0.05 + (1e-9 if b == bins - 1 else 0))
        if int(m.sum()): e += float(m.float().mean()) * abs(float(conf[m].mean()) - float(correct[m].float().mean()))
    return e


def run_stream(method, regs, seed=0, capacity=None, checkpoints=(10, 25, 50, 100), n_support=512, eval_n=384):
    """Exécute un flux ; renvoie métriques finales + points de contrôle (mise à l'échelle)."""
    sysm = make(method, seed, capacity); g = torch.Generator().manual_seed(seed + 11)
    tests = [r.test_set(seed * 1000 + i, eval_n) for i, r in enumerate(regs)]
    learn_acc, recs, cps, t0 = [], [], {}, time.time()
    for i, r in enumerate(regs):
        X, C, y = r.sample(n_support, g); rec = sysm.learn(X, C, y, t=i); recs.append(rec)
        learn_acc.append(float((sysm.predict(*tests[i][:2]) == tests[i][2]).float().mean()))
        if (i + 1) in checkpoints:
            cps[i + 1] = checkpoint_metrics(sysm, regs[: i + 1], tests[: i + 1], learn_acc, recs)
    out = cps[max(cps)] if cps else checkpoint_metrics(sysm, regs, tests, learn_acc, recs)
    out["checkpoints"] = cps; out["wall_seconds"] = round(time.time() - t0, 1)
    out["learning_curve"] = [r.get("steps", 0) for r in recs]
    out["params_curve"] = [sysm.memory()["neural_params"]] if not hasattr(sysm, "log") else [r.get("params_after", 0) for r in recs]
    out["kinds_curve"] = [r.get("kind") for r in recs]
    return out, sysm


def checkpoint_metrics(sysm, regs, tests, learn_acc, recs):
    sup = superseded(regs); idx = [j for j in range(len(regs)) if j not in sup]
    finals = {j: float((sysm.predict(*tests[j][:2]) == tests[j][2]).float().mean()) for j in idx}
    conf = torch.cat([sysm.confidence(*tests[j][:2]) for j in idx]); corr = torch.cat([sysm.predict(*tests[j][:2]) == tests[j][2] for j in idx])
    acc = sum(finals.values()) / len(finals); newt = sum(learn_acc) / len(learn_acc)
    fgt = sum(max(0.0, learn_acc[j] - finals[j]) for j in idx) / len(idx); bwt = sum(finals[j] - learn_acc[j] for j in idx) / len(idx)
    mem = sysm.memory(); kinds = Counter(r.get("kind") for r in recs)
    total_mem = sum(v for k, v in mem.items() if k.endswith("bytes"))
    comp = dict(sysm.compute)
    return {"average_accuracy": round(acc, 4), "new_task": round(newt, 4), "retention": round(acc, 4), "forgetting": round(fgt, 4),
            "backward_transfer": round(bwt, 4), "calibration_ece": round(ece(conf, corr), 4), "memory": mem, "total_memory_bytes": int(total_mem),
            "kinds": dict(kinds), "compute": comp, "n": len(regs),
            "reuse_ratio": round(sum(kinds.get(k, 0) for k in ("reuse", "compose", "reactivate")) / max(1, len(recs)), 4),
            "counters": dict(getattr(sysm, "counters", {}))}
