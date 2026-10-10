"""Temporal dynamics of beliefs: STATE_t + EVENTS + PERSISTENCE + UNOBSERVED CHANGE -> BELIEF_t+1.

For each dynamic property (entity, key), a discrete hidden Markov model, vectorised (torch):
  transition t -> t+1:
      without a reported event: stay with prob. exp(−λ_k); otherwise change to ANOTHER value, uniform
                                (unobserved change: possible, never invented precisely)
      reported event (source s, reliability r_e): with prob. r_e the event took place -> target value,
                                with prob. 1 − r_e it did not take place -> persistence transition above
  observation at time t by a source of reliability r: P(says v | state x) = r if v = x, (1 − r)/(K − 1) otherwise
Forward filtering = PRESENT belief; forward-backward smoothing = HISTORICAL belief (and automatic retroactive
correction when information about the past arrives). "I have not seen any change" ≠ "nothing has changed":
confidence decays with the age of the last piece of evidence, according to the rate λ learned per key type.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

STATIC, SLOW, FAST = "STATIC_LIKE", "DYNAMIC_SLOW", "DYNAMIC_FAST"
STALE = "STALE"
ASSERTED, QUALIFIED, WITHHELD = "ASSERTED", "QUALIFIED", "WITHHELD"
KNOWN, LIKELY, UNKNOWN, CONFLICTED = "KNOWN", "LIKELY", "UNKNOWN", "CONFLICTED"


# ================================================================ dynamics per key type

@dataclass
class KeyDynamics:
    key_type: str
    hazard: float                    # λ: rate of change per time step
    klass: str = FAST
    changes: float = 0.0
    exposure: float = 0.0

    def persistence(self, dt): return math.exp(-self.hazard * max(dt, 0))


CLASS_PRIORS = {STATIC: (0.5, 2000.0), SLOW: (1.0, 100.0), FAST: (2.0, 10.0)}   # Gamma(a, b) on λ per class


def classify(hazard):
    return STATIC if hazard < 0.002 else SLOW if hazard < 0.05 else FAST


class HazardModel:
    """λ_k estimated on TRAINING trajectories (changes / exposure), shrunk towards the class (Gamma)."""

    def __init__(self, default=None):
        self.dyn: Dict[str, KeyDynamics] = {}
        for k, (lam, kl) in (default or {}).items(): self.dyn[k] = KeyDynamics(k, lam, kl)

    def fit(self, trajectories, classes: Dict[str, str] = None):
        """trajectories: list of (key type, [values at times 0..T]) — truth of the training worlds."""
        stats = defaultdict(lambda: [0.0, 0.0])
        for kt, values in trajectories:
            stats[kt][0] += sum(1 for a, b in zip(values, values[1:]) if a != b)
            stats[kt][1] += max(0, len(values) - 1)
        for kt, (ch, ex) in stats.items():
            raw = ch / max(ex, 1e-9)
            kl = (classes or {}).get(kt, classify(raw))
            a, b = CLASS_PRIORS[kl]
            lam = (ch + a) / (ex + b)                                 # Gamma posterior mean
            self.dyn[kt] = KeyDynamics(kt, lam, classify(lam), ch, ex)
        return self

    def get(self, kt): return self.dyn.get(kt, KeyDynamics(kt, 0.1, FAST))


# ================================================================ temporal state of a property

@dataclass
class Report:
    t: int
    value: Any
    source: str
    reliability: float
    kind: str = "observation"         # observation | event
    ingested_at: int = 0             # ingestion time (≠ time of the observation or of the event)


@dataclass
class Segment:
    value: Any
    valid_from: int
    valid_until: int
    confidence: float


@dataclass
class TemporalBeliefState:
    entity: str
    key: str
    domain: List[Any]
    dynamics: KeyDynamics
    reports: List[Report] = field(default_factory=list)
    prior: Optional[torch.Tensor] = None
    _cache: Dict = field(default_factory=dict)

    @property
    def K(self): return len(self.domain)

    def idx(self, v): return self.domain.index(v)

    # ---- evidence ----------------------------------------------------------------------------
    def observe(self, t, value, source, reliability, ingested_at=None):
        self.reports.append(Report(t, value, source, reliability, "observation", ingested_at if ingested_at is not None else t))
        self._cache.clear()

    def event(self, t, target_value, source, reliability, ingested_at=None):
        """Reported event that leads to target_value at time t (applied with its evidence weight)."""
        self.reports.append(Report(t, target_value, source, reliability, "event", ingested_at if ingested_at is not None else t))
        self._cache.clear()

    def known_until(self): return max((r.ingested_at for r in self.reports), default=0)

    # ---- matrices -------------------------------------------------------------------------------
    def _persist(self, steps=1):
        K = self.K; p = self.dynamics.persistence(steps)
        M = torch.full((K, K), (1 - p) / max(K - 1, 1)); M.fill_diagonal_(p)
        return M

    def _event_matrix(self, events):
        M = self._persist()
        events = [r for r in events if not getattr(r, "defeated", False)]
        if not events: return M
        if len({r.value for r in events}) == 1 and len(events) == 1:   # one event: took place (r) or not (1 − r)
            r = events[0]; E = torch.zeros(self.K, self.K); E[:, self.idx(r.value)] = 1.0
            return r.reliability * E + (1 - r.reliability) * M
        # CONCURRENT events: P(a change took place) = 1 − Π(1 − r_i); target voted among the reports
        p_h = 1.0 - math.prod(1 - r.reliability for r in events)
        q = torch.ones(self.K)
        for r in events:
            v = torch.full((self.K,), (1 - r.reliability) / max(self.K - 1, 1)); v[self.idx(r.value)] = r.reliability; q = q * v
        q = q / q.sum()
        return p_h * q.expand(self.K, self.K) + (1 - p_h) * M

    def _likelihood(self, obs):
        L = torch.ones(self.K)
        for r in obs:
            if getattr(r, "defeated", False): continue
            rr = min(max(r.reliability, 1e-3), 1 - 1e-3)
            v = torch.full((self.K,), (1 - rr) / max(self.K - 1, 1)); v[self.idx(r.value)] = rr
            L = L * v
        return L

    # ---- filtering / smoothing -----------------------------------------------------------------------
    def posterior(self, T):
        """(filtered[0..T], smoothed[0..T]): [T+1, K] — vectorised, renormalised at each step."""
        key = ("post", T, len(self.reports), sum(getattr(r, "defeated", False) for r in self.reports))
        if key in self._cache: return self._cache[key]
        obs, ev, hidden = defaultdict(list), defaultdict(list), set()
        for r in self.reports:
            if r.t > T: continue
            if r.kind == "hidden_change": hidden.add(r.t); continue
            (obs if r.kind == "observation" else ev)[r.t].append(r)
        prior = self.prior if self.prior is not None else torch.full((self.K,), 1.0 / self.K)
        trans = [self._event_matrix(ev[t]) if ev[t] else None for t in range(T + 1)]
        U = torch.full((self.K, self.K), 1.0 / self.K)
        for t in hidden:
            if t <= T: trans[t] = U if trans[t] is None else 0.5 * (U + trans[t])      # hidden change possible: not invented
        P1 = self._persist()
        f = torch.zeros(T + 1, self.K)
        a = prior * self._likelihood(obs[0]); f[0] = a / a.sum()
        for t in range(1, T + 1):
            M = trans[t] if trans[t] is not None else P1
            a = (f[t - 1] @ M) * self._likelihood(obs[t]); f[t] = a / a.sum().clamp_min(1e-30)
        s = torch.zeros_like(f); b = torch.ones(self.K); s[T] = f[T]
        for t in range(T - 1, -1, -1):
            M = trans[t + 1] if trans[t + 1] is not None else P1
            b = M @ (self._likelihood(obs[t + 1]) * b); b = b / b.sum().clamp_min(1e-30)
            x = f[t] * b; s[t] = x / x.sum().clamp_min(1e-30)
        self._cache[key] = (f, s)
        return f, s

    def current(self, now):
        """PRESENT belief (filtered up to now)."""
        f, _ = self.posterior(now); return self._dist(f[now])

    def historical(self, t, now=None):
        """HISTORICAL belief at time t (smoothed with all the information available up to now)."""
        now = max(now if now is not None else self.known_until(), t)
        _, s = self.posterior(now); return self._dist(s[t])

    def _dist(self, v): return {self.domain[i]: float(v[i]) for i in range(self.K)}

    # ---- status, entitlement to assert, conflicts --------------------------------------------------------
    def last_observed(self, now):
        ts = [r.t for r in self.reports if r.kind == "observation" and r.t <= now and not getattr(r, "defeated", False)]
        return max(ts) if ts else None

    def status(self, now, assert_t=0.9, qualify_t=0.67):
        d = self.current(now); v, p = max(d.items(), key=lambda kv: kv[1])
        ranked = sorted(d.values(), reverse=True)
        lo = self.last_observed(now)
        if self.conflict_at(lo) if lo is not None else False: return CONFLICTED, None, p
        if p >= assert_t: return KNOWN, v, p
        if lo is not None:
            h = self.historical(lo, now); hv, hp = max(h.items(), key=lambda kv: kv[1])
            if hp >= assert_t and lo < now: return STALE, hv, p              # true then, too old for now
        if p >= qualify_t: return LIKELY, v, p
        return UNKNOWN, None, p

    def conflict_at(self, t, min_second=0.2, credible=0.6):
        """TEMPORAL conflict: two credible sources contradict each other AT THE SAME instant (≠ change of state)."""
        same = [r for r in self.reports if r.kind == "observation" and r.t == t]
        vals = {r.value for r in same if r.reliability >= credible}
        if len(vals) < 2: return False
        L = self._likelihood(same); post = L / L.sum()
        top = sorted(post.tolist(), reverse=True)
        return top[1] >= min_second

    def entitlement(self, now, assert_t=0.9, qualify_t=0.67):
        st, v, p = self.status(now, assert_t, qualify_t)
        if st == KNOWN: return ASSERTED, v, p
        if st == LIKELY or (st == STALE and p >= qualify_t): return QUALIFIED, v, p
        return WITHHELD, None, p

    # ---- trajectory, branches, compression ----------------------------------------------------------
    def trajectory(self, now=None):
        now = now if now is not None else self.known_until()
        _, s = self.posterior(now)
        out = []
        for t in range(now + 1):
            p, i = float(s[t].max()), int(s[t].argmax()); out.append((t, self.domain[i], p))
        return out

    def branches(self, t, now=None, min_p=0.1):
        """Event lattice: at an instant of a reported event, the possible outcomes (took place / not)."""
        now = now if now is not None else self.known_until()
        d = self.historical(t, now)
        return sorted(((v, p) for v, p in d.items() if p >= min_p), key=lambda kv: -kv[1])

    def segments(self, now=None, min_conf=0.9):
        """Compression: stable periods -> intervals (value, from, to, minimum confidence)."""
        segs = []
        for t, v, p in self.trajectory(now):
            if segs and segs[-1].value == v and (p >= min_conf) == (segs[-1].confidence >= min_conf):
                segs[-1].valid_until = t; segs[-1].confidence = min(segs[-1].confidence, p)
            else: segs.append(Segment(v, t, t, p))
        return segs

    # ---- minimal temporal self-correction ------------------------------------------------------
    def minimal_revision(self, now, refuted_value):
        """Refuted answer ("it is not refuted_value"): the possible causes are ranked by suspicion and only
        the most suspect one is revised: report from an unreliable source, reported event, or PERSISTENCE
        assumption (evidence too old) replaced by a possible hidden change starting from the last piece of evidence."""
        cands = []
        for r in self.reports:
            if r.t > now or getattr(r, "defeated", False) or r.value != refuted_value: continue
            cands.append((1 - r.reliability, "event" if r.kind == "event" else "source", r))
        lo = self.last_observed(now)
        if lo is not None and lo < now:
            cands.append((1 - self.dynamics.persistence(now - lo), "persistence", lo))
        if not cands: return None
        susp, cause, obj = max(cands, key=lambda c: c[0])
        if cause == "persistence":
            self.reports.append(Report(lo + 1, None, "hidden_change", 1.0, "hidden_change"))
        else:
            obj.defeated = True
        self._cache.clear()
        return {"cause": cause, "suspicion": round(susp, 3), "revised": 1, "candidates": len(cands)}

    def prediction_set(self, t, level=0.9, now=None, present=False):
        """Set of highest-probability values covering ≥ level (categorical predictive interval)."""
        d = self.current(t) if present else self.historical(t, now)
        out, acc = [], 0.0
        for v, p in sorted(d.items(), key=lambda kv: -kv[1]):
            out.append(v); acc += p
            if acc >= level: break
        return out


# ================================================================ memory of all the properties

class TemporalMemory:
    """All the dynamic properties; maintains the history (the temporal TMS does not destroy t1 when t2 changes)."""

    def __init__(self, hazards: HazardModel, domains: Dict[str, List[Any]]):
        self.hz, self.domains = hazards, domains
        self.states: Dict[Tuple[str, str], TemporalBeliefState] = {}

    def state(self, entity, key):
        if (entity, key) not in self.states:
            self.states[(entity, key)] = TemporalBeliefState(entity, key, list(self.domains[key]), self.hz.get(key))
        return self.states[(entity, key)]

    def size(self, now):
        raw = sum(len(s.reports) for s in self.states.values())
        segs = sum(len(s.segments(now)) for s in self.states.values())
        return {"properties": len(self.states), "reports": raw, "segments": segs, "time_steps": now * len(self.states)}


# ================================================================ predictive intervals of the sources

def beta_binomial_interval(alpha, beta, n, level=0.9):
    """PREDICTIVE interval on the proportion of correct answers among n FUTURE claims:
    k ~ BetaBinomial(n, α, β) (posterior uncertainty + future sampling noise). Vectorised."""
    k = torch.arange(n + 1, dtype=torch.float64)
    a, b = torch.tensor(float(alpha), dtype=torch.float64), torch.tensor(float(beta), dtype=torch.float64)
    logp = (torch.lgamma(torch.tensor(n + 1.0, dtype=torch.float64)) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
            + torch.lgamma(k + a) + torch.lgamma(n - k + b) - torch.lgamma(torch.tensor(n, dtype=torch.float64) + a + b)
            + torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b))
    p = torch.exp(logp - logp.max()); p = p / p.sum()
    cdf = torch.cumsum(p, 0)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lo = int(torch.searchsorted(cdf, torch.tensor(lo_q, dtype=torch.float64)))
    hi = int(torch.searchsorted(cdf, torch.tensor(hi_q, dtype=torch.float64)))
    return lo / max(n, 1), min(hi, n) / max(n, 1)


# ================================================================ property derived from a joint state (e.g. height)

class JointDerivedState:
    """Property DERIVED from a joint hidden state (base × modifier), e.g. height = f(support, elevation).
    The base follows its own dynamics (events, persistence); the modifier toggles with a learned rate;
    the reports on the base and on the derived value are emissions of the same state (vectorised)."""

    def __init__(self, base: TemporalBeliefState, modifiers, derive, toggle, derived_domain, mod_prior=None):
        self.base, self.mods, self.derive, self.toggle = base, list(modifiers), derive, toggle
        self.mod_prior = torch.tensor(mod_prior if mod_prior is not None else [1.0 / len(self.mods)] * len(self.mods))
        self.ddom = list(derived_domain)
        self.states = [(b, m) for b in base.domain for m in self.mods]
        self.dval = [derive(b, m) for b, m in self.states]
        self.reports: List[Report] = []

    def observe(self, t, value, source, reliability):
        self.reports.append(Report(t, value, source, reliability))

    def _lik(self, t, T):
        S = len(self.states); L = torch.ones(S)
        nb = len(self.base.domain); nd = len(self.ddom)
        for r in self.base.reports:
            if r.kind != "observation" or r.t != t: continue
            rr = min(max(r.reliability, 1e-3), 1 - 1e-3)
            L = L * torch.tensor([rr if b == r.value else (1 - rr) / (nb - 1) for b, _ in self.states])
        for r in self.reports:
            if r.t != t: continue
            rr = min(max(r.reliability, 1e-3), 1 - 1e-3)
            L = L * torch.tensor([rr if v == r.value else (1 - rr) / (nd - 1) for v in self.dval])
        return L

    def _trans(self, t):
        evs = [r for r in self.base.reports if r.kind == "event" and r.t == t]
        Mb = self.base._event_matrix(evs) if evs else self.base._persist()
        nm = len(self.mods)
        Mm = torch.full((nm, nm), self.toggle / max(nm - 1, 1)); Mm.fill_diagonal_(1 - self.toggle)
        return torch.kron(Mb, Mm)

    def posterior(self, T, smooth_at=None):
        S = len(self.states)
        prior = torch.kron(torch.full((len(self.base.domain),), 1.0 / len(self.base.domain)), self.mod_prior)
        f = torch.zeros(T + 1, S); a = prior * self._lik(0, T); f[0] = a / a.sum().clamp_min(1e-30)
        for t in range(1, T + 1):
            a = (f[t - 1] @ self._trans(t)) * self._lik(t, T); f[t] = a / a.sum().clamp_min(1e-30)
        post = f[T]
        if smooth_at is not None:
            b = torch.ones(S)
            for t in range(T - 1, smooth_at - 1, -1):
                b = self._trans(t + 1) @ (self._lik(t + 1, T) * b); b = b / b.sum().clamp_min(1e-30)
            x = f[smooth_at] * b; post = x / x.sum().clamp_min(1e-30)
        out = {v: 0.0 for v in self.ddom}
        for i, p in enumerate(post.tolist()): out[self.dval[i]] = out.get(self.dval[i], 0.0) + p
        return out

    def current(self, now): return self.posterior(now)

    def historical(self, t, now): return self.posterior(now, smooth_at=t)



# ================================================================ temporal truth maintenance

class TemporalTMS:
    """History preserved: a belief at time t1 is not destroyed when the state changes at t2; each dated
    belief keeps its justifications (reports); late information about the past revises the trajectory from
    that point on (smoothing) without erasing the earlier instants."""

    def __init__(self, state: TemporalBeliefState): self.state = state; self.log = []

    def belief_at(self, t, now):
        d = self.state.historical(t, now); v, p = max(d.items(), key=lambda kv: kv[1])
        just = [(r.source, r.t, r.kind) for r in self.state.reports if r.value == v and r.t <= t and not getattr(r, "defeated", False)]
        return {"t": t, "value": v, "p": round(p, 4), "justifications": just}

    def timeline(self, now): return [self.belief_at(t, now) for t in range(now + 1)]

    def late_report(self, t, value, source, reliability, kind="event", ingested_at=None):
        """DELAYED report (e.g. at t5: "it fell at t3"): the trajectory is recomputed from t3 on."""
        before = [b["value"] for b in self.timeline(ingested_at or t)]
        (self.state.event if kind == "event" else self.state.observe)(t, value, source, reliability, ingested_at)
        after = [b["value"] for b in self.timeline(ingested_at or t)]
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        self.log.append({"t": t, "ingested_at": ingested_at, "changed_from": min(changed) if changed else None})
        return changed
