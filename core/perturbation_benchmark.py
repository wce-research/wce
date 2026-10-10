"""INDEPENDENT perturbation benchmark (own generators, disjoint seeds) + sealed set + multi-seed.

Item families (truth = invariant RECOMPUTED on the transformed object):
  small         : one known transformation on a NEW object                → does the invariant survive?
  composed      : chain of 2 to 4 known transformations                   → survival (compositional prediction)
  ood           : never-seen parameters (values, scales, lengths)         → generalization by family
  interaction   : pairs of module perturbations of a neural model         → non-additive interaction?
  transition    : step response with a hidden threshold                   → threshold detected (absolute error)
The duality index is built on the fixed TEST objects of the domain (core/invariant_algebra), never on the benchmark.
Methods compared: duality (stabilizer), "always preserved", "surface unchanged ⇒ preserved".
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict

import torch

from core.invariant_algebra import (
    DOMAINS, LEX_ACTION, LEX_ENTITY, Equation, DomainTransformation, X, Add, Multiply, Number, _both, build_index,
    algebra_transformations, sequence_transformations, geometry_transformations,
)

FAMILIES = ("small", "composed", "ood", "interaction", "transition")


# ================================================================ generators of new objects

def new_object(domain, rng):
    if domain == "algebra":
        a = rng.choice([v for v in range(-6, 7) if v != 0]); return Equation(Add(Multiply(Number(a), X), Number(rng.randint(-9, 9))), Number(rng.randint(-9, 9)))
    if domain == "language":
        return f"{rng.choice(list(LEX_ENTITY))} {rng.choice(list(LEX_ACTION))}"
    if domain == "sequence":
        return [rng.randint(0, 9) for _ in range(rng.randint(3, 7))]
    if domain == "geometry":
        return torch.tensor([[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(4)], dtype=torch.float64)
    if domain == "code":
        return random_program(rng)
    raise KeyError(domain)


def random_program(rng, n=None):
    n = n or rng.randint(3, 5); prog, names = [], []
    for i in range(n):
        name = "abcdefgh"[i]
        def operand():
            return rng.choice(names) if names and rng.random() < 0.6 else rng.randint(1, 9)
        if i == 0 or rng.random() < 0.3: e = rng.randint(1, 9) if rng.random() < 0.5 or not names else (rng.choice("+-*"), operand(), operand())
        else:
            e = (rng.choice("+-*"), operand(), operand())
            if rng.random() < 0.5: e = (rng.choice("+-*"), e, operand())
        prog.append((name, e)); names.append(name)
    last = prog[-1][0]; prog.append(("out", ("+", last, names[0] if len(names) > 1 else 0)))
    return prog


def ood_transformation(domain, rng):
    """Transformation from a known FAMILY with a never-seen parameter."""
    if domain == "algebra":
        v = rng.choice([7, -5, 11, 0, 0, 13]); fam = rng.choice(["AddBothSides", "MultiplyBothSides", "SubtractBothSides"])
        op = {"AddBothSides": Add, "MultiplyBothSides": Multiply, "SubtractBothSides": __import__("core.algebra", fromlist=["Subtract"]).Subtract}[fam]
        return _both(op, Number(v), f"{fam}({v})")
    if domain == "sequence":
        k = rng.choice([2, 3, -4]); return DomainTransformation(f"Shift({k:+d})", lambda s, k=k: [v + k for v in s], "sequence")
    if domain == "geometry":
        from core.invariant_algebra import _rot
        if rng.random() < 0.5:
            th = rng.uniform(0.1, 3.0); return DomainTransformation(f"Rotate({th:.2f})", lambda P, th=th: P @ _rot(th).T, "geometry")
        s = rng.choice([0.5, 3.0]); return DomainTransformation(f"Scale({s})", lambda P, s=s: s * P, "geometry")
    raise KeyError(domain)


def predict_family(name, ix):
    """Generalization by family: preserves what ALL the known transformations of the same family preserve
    (known exception: zero or symbolic parameter for the multiplicative operations)."""
    fam = name.split("(")[0]; arg = name[len(fam):]
    known = [t for t in ix.transformations if t.name.split("(")[0] == fam and t.name.split("(")[1:] != ["0)"] and "(x)" not in t.name]
    if not known: return frozenset()
    if fam == "MultiplyBothSides" and arg in ("(0)", "(x)"): return frozenset(i for i in ix.preserves(fam + "(0)"))
    out = frozenset(ix.invariants.items)
    for t in known: out &= ix.preserves(t)
    return out


# ================================================================ items

def make_items(seed, n_per=12):
    rng = random.Random(seed); items = []
    doms = ("algebra", "language", "sequence", "geometry", "code")
    for d in doms:
        tf = DOMAINS[d][1]()
        for _ in range(n_per):
            x = new_object(d, rng); T = rng.choice(tf); inv = rng.choice(list(DOMAINS[d][0].items))
            items.append({"family": "small", "domain": d, "chain": [T.name], "invariant": inv, "obj_seed": rng.randrange(10 ** 9), "x": x})
        for _ in range(n_per):
            x = new_object(d, rng); ch = [rng.choice(tf).name for _ in range(rng.randint(2, 4))]
            items.append({"family": "composed", "domain": d, "chain": ch, "invariant": rng.choice(list(DOMAINS[d][0].items)), "x": x})
        if d in ("algebra", "sequence", "geometry"):
            for _ in range(n_per):
                x = new_object(d, rng); T = ood_transformation(d, rng)
                items.append({"family": "ood", "domain": d, "chain": [T.name], "T": T, "invariant": rng.choice(list(DOMAINS[d][0].items)), "x": x})
    for _ in range(n_per):
        items.append({"family": "interaction", "modules": tuple(rng.sample(["enc", "expert_a", "head"], 2)), "mag": rng.choice([0.3, 0.5, 0.8]),
                      "seed": rng.randrange(1000)})
    for _ in range(n_per):
        items.append({"family": "transition", "t_star": rng.uniform(0.5, 3.5), "sharp": rng.choice([8.0, 20.0, 50.0]), "noise": rng.choice([0.0, 0.02])})
    return items


def _truth_chain(item):
    d = item["domain"]; inv = DOMAINS[d][0][item["invariant"]]
    by = {t.name: t for t in DOMAINS[d][1]()}
    x = item["x"]; y = x
    try:
        for name in item["chain"]: y = (item["T"] if item["family"] == "ood" else by[name])(y)
    except Exception: return False
    return inv.signature(y) == inv.signature(x)


def _surface_unchanged(item):
    d = item["domain"]; by = {t.name: t for t in DOMAINS[d][1]()}; y = item["x"]
    try:
        for name in item["chain"]: y = (item["T"] if item["family"] == "ood" else by[name])(y)
    except Exception: return False
    return str(y) == str(item["x"])


_INDEX = {}


def index(d):
    if d not in _INDEX: _INDEX[d] = build_index(d)
    return _INDEX[d]


_MODEL = {}


def _model():
    if "m" not in _MODEL:
        from core.generalized_credit import base_model
        _MODEL["m"] = base_model(0)
    return _MODEL["m"]


def evaluate_items(items):
    from core.stability import perturbed_model, detect_transitions
    res = defaultdict(lambda: defaultdict(list))
    m, xv, yv = _model()
    for it in items:
        f = it["family"]
        if f in ("small", "composed", "ood"):
            truth = _truth_chain(it); ix = index(it["domain"])
            if f == "ood": pred = it["invariant"] in predict_family(it["T"].name, ix)
            else: pred = it["invariant"] in ix.chain_preserves(it["chain"])
            res[f]["duality"].append(pred == truth); res[f]["always_preserved"].append(truth)
            res[f]["surface"].append(_surface_unchanged(it) == truth)
            res[f]["truth_rate"].append(truth)
        elif f == "interaction":
            def loss_of(mods):
                with torch.no_grad():
                    return float(torch.nn.functional.cross_entropy(perturbed_model(m, mods, it["mag"], it["seed"]).forward(xv), yv)
                                 - torch.nn.functional.cross_entropy(m.forward(xv), yv))
            a, b = it["modules"]; inter = loss_of((a, b)) - loss_of((a,)) - loss_of((b,))
            joint = loss_of((a, b)); singles = abs(loss_of((a,))) + abs(loss_of((b,)))
            rel = abs(inter) / max(1e-3, abs(joint) + singles)
            res[f]["abs_interaction"].append(abs(inter)); res[f]["relative_interaction"].append(rel)
            res[f]["resonance_candidate(>25%)"].append(rel > 0.25 and abs(inter) > 0.01)
        elif f == "transition":
            rng = random.Random(int(it["t_star"] * 1e6))
            obs = lambda v: {"perf": 1.0 / (1.0 + math.exp(it["sharp"] * (v - it["t_star"]))) + rng.gauss(0, it["noise"])}
            tb = detect_transitions([0.25 * k for k in range(17)], obs, jump=0.2, refine=8)
            err = min((abs(b.threshold - it["t_star"]) for b in tb), default=4.0)
            res[f]["abs_error"].append(err); res[f]["detected(<0.1)"].append(err < 0.1)
    out = {}
    for f, d in res.items():
        out[f] = {k: round(sum(map(float, v)) / len(v), 4) for k, v in d.items()}; out[f]["n"] = len(next(iter(d.values())))
    return out


# ================================================================ sealed set

def sealed_items(): return make_items(990001, n_per=16)


def _serializable(items):
    out = []
    for it in items:
        d = {k: v for k, v in it.items() if k not in ("x", "T")}
        d["x"] = str(it.get("x")); d["T"] = it["T"].name if "T" in it else None
        out.append(d)
    return out


def sealed_hash():
    return hashlib.sha256(json.dumps(_serializable(sealed_items()), sort_keys=True, default=str).encode()).hexdigest()


SEALED_SHA256 = "cbe97b61906fb5db53f4859204266db55fef899d4ed90b634e74eafd25c49f68"   # fingerprint frozen BEFORE any evaluation


# ================================================================ compression by equivalence classes

def compression_experiment(seed=0, n=200):
    """Storing all the examples vs (unique canonical forms + path of transformations); also checks that
    each example is RECONSTRUCTIBLE (same signature as the original for the invariant of the class)."""
    from core.invariant_algebra import (
        compression, canonical_equation, canonical_sentence, canonical_program, canonical_geometry, solution_set, LANG_INVARIANTS,
        CODE_INVARIANTS, GEO_INVARIANTS, ALG_INVARIANTS, SEQ_INVARIANTS,
    )
    rng = random.Random(seed); out = {}
    specs = {
        "algebra": (ALG_INVARIANTS, "solution_set", lambda: new_object("algebra", rng),
                    [t for t in algebra_transformations() if t.name in index("algebra").stabilizer("solution_set")],
                    canonical_equation),
        "language": (LANG_INVARIANTS, "semantic_meaning", lambda: new_object("language", rng),
                     [t for t in DOMAINS["language"][1]() if t.name in ("VerbSynonym", "NounSynonym")], canonical_sentence),
        "code": (CODE_INVARIANTS, "program_output", lambda: random_program(rng),
                 [t for t in DOMAINS["code"][1]() if t.name in ("RenameVariable", "ReorderIndependentStatements")], canonical_program),
        "geometry": (GEO_INVARIANTS, "distances", lambda: new_object("geometry", rng),
                     [t for t in geometry_transformations() if t.name in ("Rotate(30°)", "Translate(1,2)")],
                     lambda P: torch.round(canonical_geometry(P), decimals=3) + 0.0),
        "sequence": (SEQ_INVARIANTS, "cyclic_order", lambda: new_object("sequence", rng),
                     [t for t in sequence_transformations() if t.name == "Rotate"],
                     lambda s: list(min(tuple(s[i:] + s[:i]) for i in range(len(s))))),
    }
    for d, (invs, inv_name, gen, G, canon) in specs.items():
        bases = [gen() for _ in range(max(4, n // 25))]; items, paths = [], {}
        for i in range(n):
            x = rng.choice(bases); path = []
            for _ in range(rng.randint(0, 3)):
                T = rng.choice(G); x = T(x); path.append(T.name)
            items.append(x); paths[id(x)] = path
        inv = invs[inv_name]
        r = compression(items, lambda x: (canon(x), paths.get(id(x), [])))
        if d == "geometry":       # continuous invariant: equality up to a tolerance (canonical rounding to 1e-3)
            ok = [bool(torch.allclose(torch.cdist(canon(x), canon(x)), torch.cdist(x, x), atol=1e-2)) for x in items]
        else: ok = [inv.signature(canon(x)) == inv.signature(x) for x in items]
        r["faithful"] = round(sum(ok) / len(items), 3)
        out[d] = r
    return out


# ================================================================ multi-seed

def multi_seed(seeds=range(16), log=print):
    from core.generalized_credit import credit_benchmark
    from core.perturbation_diagnostics import guided_benchmark, learn_profile_table
    from core.safe_plasticity import METHODS, continual_benchmark
    from core.model_diagnostics import learn_tables
    from core.grounded_benchmark import summarize_values, paired
    tables, counts = learn_tables(seed=0), learn_profile_table(seed=0)
    per = []
    for s in seeds:
        r = {"bench": evaluate_items(make_items(1000 + s)), "credit": credit_benchmark(40, seed=100 + s),
             "continual": continual_benchmark(seed=s), "diagnosis": guided_benchmark(36, seed=200 + s, tables=tables, counts=counts),
             "compression": compression_experiment(seed=s)}
        per.append(r); log(s, {m: (r["continual"][m]["new_task"], r["continual"][m]["retention"]) for m in METHODS},
                           r["credit"]["generalized"], r["credit"]["gradient_only"])
    S = lambda f: summarize_values([f(r) for r in per])
    summary = {
        "bench": {fam: {k: S(lambda r, fam=fam, k=k: r["bench"][fam][k]) for k in per[0]["bench"][fam] if k != "n"} for fam in per[0]["bench"]},
        "credit": {k: S(lambda r, k=k: r["credit"][k]) for k in ("generalized", "gradient_only", "random", "repaired")},
        "credit_symbolic": {k: S(lambda r, k=k: r["credit"]["symbolic"][k]) for k in ("generalized", "gradient_only")},
        "credit_neural": {k: S(lambda r, k=k: r["credit"]["neural"][k]) for k in ("generalized", "gradient_only")},
        "continual": {m: {k: S(lambda r, m=m, k=k: r["continual"][m][k]) for k in ("new_task", "retention", "forgetting", "parameters_changed", "rollbacks", "adaptation_cost")} for m in METHODS},
        "diagnosis": {p: {k: S(lambda r, p=p, k=k: r["diagnosis"][p][k]) for k in ("cause_accuracy", "measurements", "cost", "repaired")} for p in ("planner_only", "perturbation_guided")},
        "compression": {d: {k: S(lambda r, d=d, k=k: r["compression"][d][k]) for k in ("ratio", "faithful", "classes")} for d in per[0]["compression"]},
    }
    comps = {f"{a}_vs_{b}": {k: paired([r["continual"][a][k] for r in per], [r["continual"][b][k] for r in per]) for k in ("new_task", "retention")}
             for a, b in (("invariant_constrained", "vanilla"), ("perturbation_guided", "vanilla"), ("invariant_constrained", "replay"),
                          ("perturbation_guided", "invariant_constrained"), ("fast_weights", "vanilla"))}
    comps["credit_generalized_vs_gradient"] = paired([r["credit"]["generalized"] for r in per], [r["credit"]["gradient_only"] for r in per])
    comps["diagnosis_cost_guided_vs_planner"] = paired([r["diagnosis"]["perturbation_guided"]["cost"] for r in per], [r["diagnosis"]["planner_only"]["cost"] for r in per])
    return {"summary": summary, "comparisons": comps, "per_seed": per}
