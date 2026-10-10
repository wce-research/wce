"""Entry point on the compute machine (SageMaker container or EC2 instance).

A UNIT (kind, key) is split into reproducible SUB-UNITS (core.robust_experiments.ALL_SUBUNITS). The parent:
  * distributes the sub-units of all units over a `spawn` Pool (one process per vCPU, 1 Torch thread each);
  * writes one CHECKPOINT per finished sub-unit (ckpt_<kind>_<key>.jsonl) → a resume recomputes only the rest;
  * publishes `_progress.json` (step, items processed, elapsed time, ETA, CPU, memory, last checkpoint);
each compute process publishes a HEARTBEAT `_hb_<pid>.json` (current sub-unit, Python stack, CPU, memory).
The output directory is synchronised to S3 continuously by the launch script.
"""
import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import threading
import time
import traceback

OUT = os.environ.get("WCE_OUT", "/opt/ml/processing/output")
SPECIAL = ("profile", "bench")                      # instrumentation units (wce_aws.bench)
HEARTBEAT_S = float(os.environ.get("WCE_HEARTBEAT_S", "20"))
_HB = {"sub": None, "since": 0.0, "started": False}


def _instance_type():
    try: return json.load(open("/opt/ml/config/processingjobconfig.json"))["ProcessingResources"]["ClusterConfig"]["InstanceType"]
    except Exception: return os.environ.get("WCE_INSTANCE", "unknown")


def _write(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), default=str); os.replace(tmp, path)        # atomic write (never a truncated JSON on S3)


def rss_mb():
    try: return round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except Exception: return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def cpu_times():
    """Cumulative (busy time, total time) of the machine, in ticks (/proc/stat); None outside Linux."""
    try: v = [int(x) for x in open("/proc/stat").readline().split()[1:9]]; return sum(v) - v[3] - v[4], sum(v)
    except Exception: return None


def mem_available_mb():
    try: return round(int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1024)
    except Exception: return None


class CpuSampler(threading.Thread):
    """Samples the CPU usage of the machine (mean / peak, in % of all vCPUs) and the minimal available memory."""
    def __init__(self, period=2.0): super().__init__(daemon=True); self.period = period; self.samples = []; self.min_mem = None; self._halt = threading.Event()

    def run(self):
        prev = cpu_times()
        while not self._halt.wait(self.period):
            cur = cpu_times(); m = mem_available_mb()
            if prev and cur and cur[1] > prev[1]: self.samples.append(100.0 * (cur[0] - prev[0]) / (cur[1] - prev[1]))
            if m is not None: self.min_mem = m if self.min_mem is None else min(self.min_mem, m)
            prev = cur

    def stop(self): self._halt.set()

    def summary(self):
        s = self.samples; return {"cpu_avg_pct": round(sum(s) / len(s), 1) if s else None, "cpu_peak_pct": round(max(s), 1) if s else None, "mem_available_min_mb": self.min_mem, "samples": len(s)}


def _stack(main_id, depth=6):
    """Python stack of the main thread, reduced to the functions of the project (where the computation stands, without interrupting it)."""
    fr = sys._current_frames().get(main_id); out = []
    while fr is not None:
        f = fr.f_code.co_filename
        if "/core/" in f or "/wce_aws/" in f: out.append(f"{os.path.basename(f)}:{fr.f_code.co_name}:{fr.f_lineno}")
        fr = fr.f_back
    return out[:depth]


def _heartbeat_loop(main_id):
    while True:
        time.sleep(HEARTBEAT_S)
        try:
            _write(os.path.join(OUT, f"_hb_{os.getpid()}.json"), {"pid": os.getpid(), "time": time.time(), "sub": _HB["sub"], "sub_elapsed_s": round(time.time() - _HB["since"], 1) if _HB["sub"] else None,
                                                                 "cpu_s": round(time.process_time(), 1), "rss_mb": rss_mb(), "stack": _stack(main_id) if _HB["sub"] else []})
        except Exception: pass


def _start_heartbeat():
    if not _HB["started"]:
        _HB["started"] = True; os.makedirs(OUT, exist_ok=True); threading.Thread(target=_heartbeat_loop, args=(threading.main_thread().ident,), daemon=True).start()


def run_sub(task):
    """Executed in a compute process: one sub-unit, 1 Torch thread (the parallelism is that of the processes)."""
    kind, key, sid = task
    import torch
    torch.set_num_threads(int(os.environ.get("WCE_TORCH_THREADS", "1")))
    _start_heartbeat(); _HB["sub"] = f"{kind}_{key}:{sid}"; _HB["since"] = time.time(); t0 = time.time(); c0 = time.process_time()
    try:
        if kind in SPECIAL:
            from wce_aws import bench
            rows = dict(bench.SUBS[kind](key))[sid]()
        elif kind.startswith("deep"):                                   # milestone Deep Compositional Structure Discovery
            from core.deep_experiments import run_subunit as deep_run
            rows = deep_run(kind, key, sid)
        else:
            from core.robust_experiments import run_subunit
            rows = run_subunit(kind, key, sid)
        err = None
    except Exception:
        rows, err = None, traceback.format_exc()
    _HB["sub"] = None
    return {"kind": kind, "key": key, "sub": sid, "rows": rows, "error": err, "seconds": round(time.time() - t0, 2), "cpu_s": round(time.process_time() - c0, 2), "rss_mb": rss_mb(), "pid": os.getpid()}


def sub_ids(kind, key):
    if kind in SPECIAL:
        from wce_aws import bench
        return [sid for sid, _ in bench.SUBS[kind](key)]
    if kind.startswith("deep"):
        from core.deep_experiments import subunit_ids as deep_ids
        return deep_ids(kind, key)
    from core.robust_experiments import subunit_ids
    return subunit_ids(kind, key)


def ckpt_path(kind, key): return os.path.join(OUT, f"ckpt_{kind}_{key}.jsonl")


def load_ckpt(kind, key):
    """Sub-units already finished {identifier: record}; a truncated last line (interruption) is ignored."""
    done = {}
    try:
        for line in open(ckpt_path(kind, key)):
            try: r = json.loads(line); done[r["sub"]] = r
            except (ValueError, KeyError): continue
    except OSError: pass
    return done


def finalize(kind, key, ids, done, errors, resumed):
    """Final result of the unit = ORDERED concatenation of the rows of the sub-units (identical with or without resume)."""
    ok = all(s in done for s in ids)
    rec = {"kind": kind, "key": key, "status": "ok" if ok else "error", "runtime_s": round(sum(done[s]["seconds"] for s in ids if s in done), 2),
           "subunits": {s: {"seconds": done[s]["seconds"], "cpu_s": done[s].get("cpu_s")} for s in ids if s in done},
           "result": [row for s in ids for row in done[s]["rows"]] if ok else {"error": errors.get(f"{kind}_{key}", "sous-unités manquantes"), "missing": [s for s in ids if s not in done]}}
    _write(os.path.join(OUT, f"{kind}_{key}.json"), rec)
    return {"kind": kind, "key": key, "status": rec["status"], "runtime_s": rec["runtime_s"], "subunits": len(ids), "resumed": resumed}


def run_units(units, procs=0, pool=True):
    """Executes the units (with resume). Returns the summary of the shard. `pool=False`: sequential in this process (tests)."""
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); procs = procs or os.cpu_count() or 1
    plan, done, todo, resumed = {}, {}, [], {}
    for u in units:
        kind, key = u["kind"], str(u["key"])
        try: ids = sub_ids(kind, key)
        except Exception:
            _write(os.path.join(OUT, f"{kind}_{key}.json"), {"kind": kind, "key": key, "status": "error", "runtime_s": 0.0, "result": {"error": traceback.format_exc()}})
            plan[(kind, key)] = None; continue
        plan[(kind, key)] = ids; d = {s: r for s, r in load_ckpt(kind, key).items() if s in ids}; done[(kind, key)] = d; resumed[(kind, key)] = len(d)
        todo += [(kind, key, s) for s in ids if s not in d]
    valid = {k: v for k, v in plan.items() if v is not None}
    total = sum(len(v) for v in valid.values()); n_resumed = sum(resumed.values()); errors = {}; fresh = []; last = None
    sampler = CpuSampler(); sampler.start(); nproc = max(1, min(procs, len(todo)))

    def publish(state):
        el = time.time() - t0; n_done = sum(len(v) for v in done.values()); new = len(fresh)
        eta = round(el / new * (total - n_done - len(errors)), 0) if new and state == "running" else (0 if state != "running" else None)
        hb = []
        for f in os.listdir(OUT):
            if f.startswith("_hb_") and f.endswith(".json"):
                try: h = json.load(open(os.path.join(OUT, f)))
                except (OSError, ValueError): continue
                if h.get("sub") and time.time() - h.get("time", 0) < 4 * HEARTBEAT_S: hb.append(h)
        _write(os.path.join(OUT, "_progress.json"), {
            "state": state, "time": time.time(), "step": last["sub_id"] if last else None, "in_flight": [{k: h.get(k) for k in ("sub", "sub_elapsed_s", "cpu_s", "rss_mb", "stack")} for h in hb],
            "items_done": n_done, "items_total": total, "items_resumed": n_resumed, "items_failed": len(errors), "elapsed_s": round(el, 1), "eta_s": eta,
            "mean_item_s": round(sum(r["seconds"] for r in fresh) / new, 2) if new else None, "processes": nproc, "vcpus": os.cpu_count(), "load_avg": list(os.getloadavg()),
            **sampler.summary(), "last_checkpoint": last, "instance_type": _instance_type(),
            "units": {f"{k}_{key}": {"done": len(done[(k, key)]), "total": len(ids)} for (k, key), ids in valid.items()}})

    def record(r):
        nonlocal last
        k = (r["kind"], r["key"])
        if r["error"]: errors[f"{r['kind']}_{r['key']}"] = r["error"]
        else:
            with open(ckpt_path(*k), "a") as f: f.write(json.dumps({x: r[x] for x in ("sub", "rows", "seconds", "cpu_s", "rss_mb")}, default=str) + "\n"); f.flush(); os.fsync(f.fileno())
            done[k][r["sub"]] = json.loads(json.dumps(r, default=str)); fresh.append(r); last = {"sub_id": f"{r['kind']}_{r['key']}:{r['sub']}", "time": time.time(), "seconds": r["seconds"]}
        publish("running")

    stop_tick = threading.Event()

    def ticker():                                      # the progress is republished even if no sub-unit finishes
        while not stop_tick.wait(HEARTBEAT_S):
            try: publish("running")
            except Exception: pass

    publish("running"); threading.Thread(target=ticker, daemon=True).start()
    if todo:
        if pool:
            with mp.get_context("spawn").Pool(nproc) as p:
                for r in p.imap_unordered(run_sub, todo, chunksize=1): record(r)
        else:
            for t in todo: record(run_sub(t))
    stop_tick.set()
    summ = [finalize(k, key, ids, done[(k, key)], errors, resumed[(k, key)]) if ids is not None else {"kind": k, "key": key, "status": "error", "runtime_s": 0.0, "subunits": 0, "resumed": 0} for (k, key), ids in plan.items()]
    sampler.stop(); wall = time.time() - t0; work = sum(r["seconds"] for r in fresh); publish("done" if all(s["status"] == "ok" for s in summ) else "failed")
    shard = {"units": summ, "wall_s": round(wall, 2), "instance_type": _instance_type(), "cpus": os.cpu_count(), "procs": nproc, "subunits_run": len(fresh), "subunits_resumed": n_resumed,
             "work_s": round(work, 2), "cpu_s": round(sum(r["cpu_s"] for r in fresh), 2), "parallel_efficiency": round(work / (wall * nproc), 3) if wall > 0 and fresh else None,
             "peak_rss_mb": max([r["rss_mb"] for r in fresh], default=None), "load_avg": list(os.getloadavg()), **sampler.summary(), "errors": errors}
    _write(os.path.join(OUT, "_shard_summary.json"), shard)
    return shard


def run_unit(u):
    """One unit, sequentially in this process (tests, debugging)."""
    s = run_units([u], pool=False)["units"][0]; return {"kind": s["kind"], "key": s["key"], "status": s["status"], "runtime_s": s["runtime_s"]}


def serialize_units(units): return json.dumps(units, separators=(",", ":"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--units", required=True); ap.add_argument("--procs", type=int, default=0); a = ap.parse_args()
    sys.path.insert(0, os.getcwd()); shard = run_units(json.loads(a.units), a.procs)
    print(json.dumps({"done": len(shard["units"]), "errors": sum(s["status"] != "ok" for s in shard["units"]), "wall_s": shard["wall_s"]}))
    sys.exit(1 if any(s["status"] != "ok" for s in shard["units"]) else 0)
