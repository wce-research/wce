"""Instrumentation exécutée SUR AWS : profilage du vrai point chaud et micro-benchmark contrôlé.

* `profile` : une sous-unité par méthode, sur UNE tâche du banc principal (graine 0), sous cProfile, avec plafond de
  temps. Sortie : temps par fonction du projet (phases), fonctions les plus coûteuses, répartition du temps propre
  (interpréteur Python / noyaux Torch / couche Python de Torch / E-S et sérialisation / autre).
* `bench`   : même charge sur chaque type d'instance : (A) séquentiel 1 thread, (B) séquentiel avec autant de threads
  Torch que de vCPU, (C) autant de processus que de vCPU en parallèle. Temps de calcul pur, CPU moyen/pic, mémoire,
  efficacité parallèle.
Ces unités mesurent du TEMPS ; leurs résultats scientifiques ne sont pas utilisés.
"""
import cProfile
import json
import os
import pstats
import signal
import subprocess
import sys
import time

PROFILE_SEED = 0
PROFILE_CAP_S = int(os.environ.get("WCE_PROFILE_CAP_S", "900"))
BENCH_WORKLOAD = (("main", 0, "t1_robust_posterior"), ("main", 0, "t1_robust_active_efa"))   # petite sous-unité représentative (profondeur 2, bruit 8 %)
BENCH_CAP_S = int(os.environ.get("WCE_BENCH_CAP_S", "1500"))


class _Cap(Exception):
    pass


def _alarm(sig, frm): raise _Cap()


def category(filename, func):
    """Classe une entrée de profil selon l'endroit où le temps PROPRE est passé."""
    f = filename.replace("\\", "/")
    if f == "~":                                                       # fonction native (C/C++)
        if "torch" in func or "Tensor" in func or "_C." in func: return "torch_noyaux_natifs"
        if any(k in func for k in ("posix", "_io.", "json", "pickle", "write", "read", "fsync")): return "es_serialisation"
        return "python_natifs_divers"
    if "/torch/" in f: return "torch_couche_python"
    if "/core/" in f or "/wce_aws/" in f: return "python_projet"
    if f.endswith(("json/encoder.py", "json/decoder.py")) or "/json/" in f: return "es_serialisation"
    return "python_stdlib_autre"


def summarize(pr, top=18):
    st = pstats.Stats(pr); rows = []; cats = {}
    for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
        c = category(fn, name); cats[c] = cats.get(c, 0.0) + tt
        rows.append({"where": f"{os.path.basename(fn)}:{line}" if fn != "~" else "natif", "func": name[:70], "calls": nc, "self_s": round(tt, 3), "cum_s": round(ct, 3), "cat": c,
                     "project": "/core/" in fn.replace("\\", "/")})
    total = sum(cats.values()) or 1.0
    return {"profiled_s": round(total, 2), "self_time_by_category_s": {k: round(v, 2) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
            "self_time_by_category_pct": {k: round(100 * v / total, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
            "top_self": sorted(rows, key=lambda r: -r["self_s"])[:top],
            "project_phases_by_cumulative": [{k: r[k] for k in ("where", "func", "calls", "self_s", "cum_s")} for r in sorted((r for r in rows if r["project"]), key=lambda r: -r["cum_s"])[:top]]}


def profile_sub(task_k, method, cap=None):
    """Profile UNE sous-unité (tâche k, méthode) du banc principal. Le plafond interrompt le calcul, pas la mesure."""
    from core.robust_experiments import MAIN_SPECS, run_subunit
    cap = PROFILE_CAP_S if cap is None else cap; pr = cProfile.Profile(); t0 = time.time(); timed_out = False
    old = signal.signal(signal.SIGALRM, _alarm); signal.alarm(cap)
    try:
        pr.enable(); run_subunit("main", PROFILE_SEED, f"t{task_k}_{method}")
    except _Cap: timed_out = True
    finally: pr.disable(); signal.alarm(0); signal.signal(signal.SIGALRM, old)
    nz, d = MAIN_SPECS[task_k]
    return [{"task": task_k, "noise": nz, "depth": d, "method": method, "wall_s": round(time.time() - t0, 2), "timed_out": timed_out, "cap_s": cap, "note": "cProfile ajoute un surcoût sur le code Python pur",
             **summarize(pr)}]


def profile_subunits(key):
    from core.robust_experiments import METHODS
    return [(m, (lambda m=m: profile_sub(int(key), m))) for m in METHODS]


def _workload():
    """La charge de référence, séquentiellement. Renvoie (temps mur, temps CPU du processus)."""
    from core.robust_experiments import run_subunit
    t0 = time.time(); c0 = time.process_time()
    for kind, key, sid in BENCH_WORKLOAD: run_subunit(kind, key, sid)
    return round(time.time() - t0, 2), round(time.process_time() - c0, 2)


def _phase(name, threads, copies):
    """Lance `copies` processus indépendants exécutant la charge avec `threads` threads Torch ; mesure CPU machine et mémoire."""
    from wce_aws.sm_entry import CpuSampler
    env = dict(os.environ, WCE_TORCH_THREADS=str(threads), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    s = CpuSampler(1.0); s.start(); t0 = time.time()
    ps = [subprocess.Popen([sys.executable, "-m", "wce_aws.bench", "--workload"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(copies)]
    outs = []
    for p in ps:
        try: o, e = p.communicate(timeout=BENCH_CAP_S); outs.append(json.loads(o.strip().splitlines()[-1]) if p.returncode == 0 else {"error": e[-400:]})
        except subprocess.TimeoutExpired: p.kill(); outs.append({"error": f"plafond {BENCH_CAP_S} s"})
    wall = round(time.time() - t0, 2); s.stop()
    ok = [o for o in outs if "wall_s" in o]
    return {"phase": name, "torch_threads": threads, "processes": copies, "wall_s": wall, "compute_s_per_copy": [o["wall_s"] for o in ok], "cpu_s_per_copy": [o["cpu_s"] for o in ok],
            "peak_rss_mb": max([o["rss_mb"] for o in ok], default=None), "errors": [o["error"] for o in outs if "error" in o], **s.summary()}


def bench_sub():
    n = os.cpu_count() or 1
    A = _phase("A_sequentiel_1_thread", 1, 1); B = _phase("B_sequentiel_threads_torch", n, 1); C = _phase("C_processus_paralleles", 1, n)
    a = A["compute_s_per_copy"][0] if A["compute_s_per_copy"] else None; c = max(C["compute_s_per_copy"]) if len(C["compute_s_per_copy"]) == n else None
    b = B["compute_s_per_copy"][0] if B["compute_s_per_copy"] else None
    return [{"vcpus": n, "instance_type": os.environ.get("WCE_INSTANCE", "unknown"), "workload": [list(w) for w in BENCH_WORKLOAD], "phases": [A, B, C],
             "compute_s_1proc_1thread": a, "thread_speedup": round(a / b, 2) if a and b else None,
             "process_parallel_efficiency": round(a / c, 2) if a and c else None,                    # 1,0 = n processus aussi rapides chacun qu'un seul
             "throughput_workloads_per_hour": round(3600 * n / c, 1) if c else None, "throughput_1proc_per_hour": round(3600 / a, 1) if a else None,
             "cpu_model": _cpu_model()}]


def _cpu_model():
    try: return [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception: return "inconnu"


def bench_subunits(key): return [("all", bench_sub)]


SUBS = {"profile": profile_subunits, "bench": bench_subunits}


if __name__ == "__main__":
    if "--workload" in sys.argv:
        sys.path.insert(0, os.getcwd())
        import torch
        torch.set_num_threads(int(os.environ.get("WCE_TORCH_THREADS", "1")))
        from wce_aws.sm_entry import rss_mb
        w, c = _workload(); print(json.dumps({"wall_s": w, "cpu_s": c, "rss_mb": rss_mb(), "torch_threads": torch.get_num_threads()}))
