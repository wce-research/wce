"""LOCAL orchestration → SageMaker Processing (via AWS CLI, profile wce-runner). Invoked ONLY by the Makefile.

smoke | package | submit | status | logs | fetch | clean-safe
Each submission writes results/aws/<run_id>/manifest.json: run_id, git commit, timestamp, code and configuration
hash, instance type, region, jobs, status, durations, artefact paths. No IAM action is ever
attempted (only an IAM READ is tried at the smoke test to verify that it is denied)."""
import argparse
import glob
import hashlib
import io
import json
import re
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

REGION = os.environ.get("WCE_REGION", "eu-north-1")
PROFILE = os.environ.get("AWS_PROFILE", "wce-runner")
BUCKET = os.environ.get("WCE_BUCKET", "wce-research")
PREFIX = "robust"
EXPECTED_ROLE = "WCE-Research-Runner"
EXEC_ROLE_NAME = "WCE-SageMaker-ExecutionRole"
IMAGE = os.environ.get("WCE_IMAGE", f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker")
TAGS = [{"Key": "Project", "Value": "WCE"}, {"Key": "Purpose", "Value": "Research"}, {"Key": "Component", "Value": "RobustReconstruction"},
        {"Key": "ManagedBy", "Value": "WCE"}]
PROBE_INSTANCES = ("ml.c5.4xlarge", "ml.m5.4xlarge", "ml.c5.2xlarge", "ml.m5.2xlarge", "ml.m5.xlarge", "ml.m5.large", "ml.t3.xlarge", "ml.t3.large", "ml.t3.medium")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIRS = ("core", "physics", "language", "logic", "app", "wce_aws")


class AwsDenied(RuntimeError):
    pass


def aws(*args, parse=True, check=True):
    cmd = ["aws", *args, "--region", REGION, "--profile", PROFILE, "--output", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        if check:
            err = p.stderr.strip()
            if "AccessDenied" in err or "not authorized" in err or "UnauthorizedOperation" in err:
                raise AwsDenied(f"MISSING PERMISSION — {' '.join(args[:2])} : {err[:400]}")
            raise RuntimeError(f"aws {' '.join(args[:3])} : {err[:600]}")
        return None
    return json.loads(p.stdout) if parse and p.stdout.strip() else p.stdout


def check_identity(arn):
    """Fails cleanly if the identity is not the runner (never an administrator)."""
    if EXPECTED_ROLE not in arn or "Admin" in arn: raise SystemExit(f"INCORRECT IDENTITY: {arn} (expected: role {EXPECTED_ROLE})")
    return True


def identity():
    ident = aws("sts", "get-caller-identity"); check_identity(ident["Arn"]); return ident


def git_commit():
    p = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else "no-git"


def build_tarball():
    buf = io.BytesIO(); h = hashlib.sha256()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in CODE_DIRS:
            for f in sorted(glob.glob(os.path.join(ROOT, d, "**", "*"), recursive=True)):
                if os.path.isfile(f) and "__pycache__" not in f and not f.endswith((".pyc", ".bak")):
                    rel = os.path.relpath(f, ROOT); h.update(rel.encode()); h.update(open(f, "rb").read()); tar.add(f, arcname=rel)
        res = os.path.join(ROOT, "results", "active_stability_calibration.json")
        if os.path.exists(res): tar.add(res, arcname="results/active_stability_calibration.json")
        for f in sorted(glob.glob(os.path.join(ROOT, "results", "deep_configs", "*.json"))):                 # frozen configurations and weights of the compact proposer
            rel = os.path.relpath(f, ROOT); h.update(rel.encode()); h.update(open(f, "rb").read()); tar.add(f, arcname=rel)
    return buf.getvalue(), h.hexdigest()[:16]


def package():
    data, code_hash = build_tarball(); key = f"{PREFIX}/code/{code_hash}/code.tar.gz"; tmp = os.path.join(ROOT, "results", "aws", f"code_{code_hash}.tar.gz")
    os.makedirs(os.path.dirname(tmp), exist_ok=True); open(tmp, "wb").write(data)
    aws("s3", "cp", tmp, f"s3://{BUCKET}/{key}", "--quiet", parse=False); os.remove(tmp)
    return code_hash, f"s3://{BUCKET}/{PREFIX}/code/{code_hash}/"


def smoke(submit_job=False):
    report = {"time": datetime.now(timezone.utc).isoformat()}
    ident = identity(); report["identity"] = ident["Arn"].split("/")[1]; report["region"] = REGION
    key = f"{PREFIX}/smoke/probe_{int(time.time())}.txt"; tmp = os.path.join(ROOT, "results", "aws", "probe.txt")
    os.makedirs(os.path.dirname(tmp), exist_ok=True); open(tmp, "w").write("wce smoke")
    aws("s3", "cp", tmp, f"s3://{BUCKET}/{key}", "--quiet", parse=False); aws("s3", "rm", f"s3://{BUCKET}/{key}", "--quiet", parse=False); os.remove(tmp)
    report["s3_write_delete"] = "OK"
    aws("sagemaker", "list-processing-jobs", "--max-results", "1"); report["sagemaker_list"] = "OK"
    try:
        aws("iam", "list-roles", "--max-items", "1"); report["iam_read_expected_denied"] = "ÉCHEC : IAM accessible (séparation non respectée)"
    except (AwsDenied, RuntimeError) as e: report["iam_read_expected_denied"] = "OK (refusé comme attendu)"
    if submit_job:                                   # quota probe: first accepted instance type (no quota is modified)
        report["quota_probe"] = {}
        probes = ("t3.large", "t3.medium") if BACKEND == "ec2" else PROBE_INSTANCES   # types allowed to the runner (c5/m5 denied by policy)
        for inst in probes:
            try: report["smoke_run"] = submit("smoke", ["0"], shards=1, instance=inst, max_runtime=1800); report["quota_probe"][inst] = "OK"; break
            except RuntimeError as e:
                if "ResourceLimitExceeded" in str(e) or "VcpuLimitExceeded" in str(e) or "InstanceLimitExceeded" in str(e): report["quota_probe"][inst] = "quota 0"
                else: raise
    print(json.dumps(report, indent=1, ensure_ascii=False)); return report


def quotas():
    """READ ONLY: Service Quotas quotas useful to WCE and history of the requests. Calls only `get-service-quota`,
    `list-service-quotas` and `list-requested-service-quota-change-history`; submits and modifies nothing."""
    def call(*args):
        cmd = ["aws", *args, "--region", REGION, "--profile", PROFILE, "--output", "json"]; p = subprocess.run(cmd, capture_output=True, text=True)
        return (json.loads(p.stdout or "{}"), None) if p.returncode == 0 else (None, (p.stderr or "").strip().splitlines()[-1][:300] if p.stderr.strip() else "erreur")
    out = {"region": REGION, "errors": []}
    q, e = call("service-quotas", "get-service-quota", "--service-code", "ec2", "--quota-code", "L-1216C47A")
    out["ec2_standard"] = {"code": "L-1216C47A", "name": q["Quota"]["QuotaName"], "value": q["Quota"]["Value"], "adjustable": q["Quota"].get("Adjustable")} if q else {"code": "L-1216C47A", "error": e}
    sm, e = call("service-quotas", "list-service-quotas", "--service-code", "sagemaker", "--no-paginate", "--max-results", "100"); items = []
    cmd_all, e2 = call("service-quotas", "list-service-quotas", "--service-code", "sagemaker")
    for x in (cmd_all or sm or {}).get("Quotas", []):
        n = x["QuotaName"].lower()
        if ("processing job usage" in n or "training job usage" in n) and "spot" not in n and any(t in n for t in ("ml.c5.", "ml.m5.", "ml.t3.", "ml.c6i.", "ml.c7i.")): items.append({"code": x["QuotaCode"], "name": x["QuotaName"], "value": x["Value"]})
    if cmd_all is None and sm is None: out["errors"].append({"call": "list-service-quotas sagemaker", "error": e2 or e})
    out["sagemaker_nonzero_or_requested"] = sorted(items, key=lambda z: z["name"]); hist = []
    for svc in ("ec2", "sagemaker"):
        h, e = call("service-quotas", "list-requested-service-quota-change-history", "--service-code", svc)
        if h is None: out["errors"].append({"call": f"list-requested-service-quota-change-history {svc}", "error": e}); continue
        hist += [{"service": r.get("ServiceCode"), "code": r.get("QuotaCode"), "quota": r.get("QuotaName"), "desired": r.get("DesiredValue"), "status": r.get("Status"), "created": str(r.get("Created")), "last_updated": str(r.get("LastUpdated")), "case": r.get("CaseId"), "level": r.get("QuotaRequestedAtLevel")} for r in h.get("RequestedQuotas", [])]
    out["requests"] = sorted(hist, key=lambda r: r["last_updated"], reverse=True); req = {r["code"] for r in hist}
    out["sagemaker_nonzero_or_requested"] = [x for x in out["sagemaker_nonzero_or_requested"] if x["value"] > 0 or x["code"] in req]
    print(json.dumps(out, indent=1, ensure_ascii=False)); return out


def bench_report(kind="deepbenchhp"):
    """Instance bench: reads the `kind` runs already fetched (no AWS call). Per type: time, throughput, cost, memory,
    errors, and NUMERICAL comparison of the results with the c6i.xlarge reference (time and memory fields excluded)."""
    skip = ("seconds", "rss_mb", "runtime_s", "cpu_s"); runs = {}
    def strip(x):
        if isinstance(x, dict): return {k: strip(v) for k, v in sorted(x.items()) if k not in skip}
        return [strip(v) for v in x] if isinstance(x, list) else x
    def diff(a, b, out):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b): diff(a.get(k), b.get(k), out)
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for u, v in zip(a, b): diff(u, v, out)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool): out["numeric"] += 1; out["max_abs"] = max(out["max_abs"], abs(a - b)); out["differing"] += abs(a - b) > 1e-9
        else: out["other"] += 1; out["other_differing"] += a != b
    for p, man in manifests():
        if man.get("kind") != kind: continue
        for j in man["jobs"]:
            jd = os.path.join(os.path.dirname(p), "artifacts", j["name"])
            try: t = json.load(open(os.path.join(jd, "_timing.json"))); sm = json.load(open(os.path.join(jd, "_shard_summary.json")))
            except (OSError, ValueError): runs[man["run_id"]] = {"instance": man.get("instance_type"), "error": "artefacts absents ou incomplets"}; continue
            res = {os.path.basename(f): strip(json.load(open(f))) for f in sorted(glob.glob(os.path.join(jd, f"{kind}_*.json")))}; inst = man.get("instance_type"); price = PRICE_USD_H.get(inst); n = sm.get("subunits_run", 0); billed = t["setup_s"] + t["job_s"]
            runs[man["run_id"]] = {"instance": inst, "setup_s": t["setup_s"], "job_s": t["job_s"], "wall_s": sm.get("wall_s"), "subunits": n, "subunits_per_s": round(n / sm["wall_s"], 5) if sm.get("wall_s") else None, "work_s": sm.get("work_s"), "parallel_efficiency": sm.get("parallel_efficiency"), "cpu_avg_pct": sm.get("cpu_avg_pct"),
                                   "peak_rss_mb": sm.get("peak_rss_mb"), "mem_available_min_mb": sm.get("mem_available_min_mb"), "mem_total_mb": round(t.get("mem_total_kb", 0) / 1024), "errors": sm.get("errors"), "unit_status": [u["status"] for u in sm.get("units", [])], "exit_code": t.get("exit_code"),
                                   "price_usd_h": price, "cost_run_usd": round(billed / 3600 * price, 4) if price else None, "cost_per_subunit_usd": round(sm["wall_s"] / 3600 * price / n, 6) if price and n and sm.get("wall_s") else None, "_res": res}
    res = {k: r.pop("_res") for k, r in runs.items() if "_res" in r}; rk = next((k for k in res if runs[k]["instance"] == "c6i.xlarge"), None)
    for k in res:
        if rk is None: continue
        d = {"numeric": 0, "differing": 0, "max_abs": 0.0, "other": 0, "other_differing": 0}; diff(res[rk], res[k], d); runs[k]["vs_c6i"] = d; runs[k]["speedup_vs_c6i"] = round(runs[rk]["wall_s"] / runs[k]["wall_s"], 3) if runs[k].get("wall_s") else None
    out = os.path.join(ROOT, "results", "aws_instance_bench.json"); json.dump(runs, open(out, "w"), indent=1); print(json.dumps(runs, indent=1, ensure_ascii=False)); print("->", out); return runs


def whoami():
    """`sts get-caller-identity` with the runner profile; fails if the identity is not WCE-Research-Runner."""
    ident = aws("sts", "get-caller-identity"); arn = ident["Arn"]
    print(json.dumps({"profile": PROFILE, "region": REGION, "account": ident["Account"], "arn": arn}, indent=1)); check_identity(arn)
    print(f"IDENTITY OK: {EXPECTED_ROLE}"); return ident


GUARDRAIL_IAM_PROBES = (("list-roles", "--max-items", "1"), ("list-users", "--max-items", "1"))   # READS only: no mutation attempted


def check_guardrails():
    """Guardrails: (1) identity = runner, never admin; (2) the administrative IAM operations remain DENIED.
    Read-only probes (nothing is created or modified, even if the guardrail were absent). Exit code 4 on failure."""
    report = {"time": datetime.now(timezone.utc).isoformat(), "profile": PROFILE, "region": REGION}; ok = True
    arn = aws("sts", "get-caller-identity")["Arn"]; report["arn"] = arn
    try: check_identity(arn); report["identity"] = f"OK ({EXPECTED_ROLE})"
    except SystemExit as e: report["identity"] = f"ÉCHEC : {e}"; ok = False
    for probe in GUARDRAIL_IAM_PROBES:
        try: aws("iam", *probe); report[f"iam {probe[0]}"] = "ÉCHEC : opération IAM AUTORISÉE (séparation non respectée)"; ok = False
        except AwsDenied: report[f"iam {probe[0]}"] = "OK (refusé comme attendu)"
    report["guardrails"] = "OK" if ok else "ÉCHEC"; print(json.dumps(report, indent=1, ensure_ascii=False))
    if not ok: sys.exit(4)
    return report


def job_spec(name, role, units, code_uri, out, instance, max_runtime, run_id):
    entry = ("cd /opt/ml/processing/input/code && tar xzf code.tar.gz && "
             f"python3 -m wce_aws.sm_entry --units '{json.dumps(units, separators=(',', ':'))}'")
    return {"ProcessingJobName": name, "RoleArn": role,
            "AppSpecification": {"ImageUri": IMAGE, "ContainerEntrypoint": ["bash", "-c", entry]},
            "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1, "InstanceType": instance, "VolumeSizeInGB": 10}},
            "ProcessingInputs": [{"InputName": "code", "S3Input": {"S3Uri": code_uri, "LocalPath": "/opt/ml/processing/input/code", "S3DataType": "S3Prefix", "S3InputMode": "File"}}],
            "ProcessingOutputConfig": {"Outputs": [{"OutputName": "results", "S3Output": {"S3Uri": out, "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "Continuous"}}]},
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime}, "Environment": {"WCE_RUN_ID": run_id, "OMP_NUM_THREADS": "1"}, "Tags": TAGS}


BACKEND = os.environ.get("WCE_BACKEND", "processing")        # processing (main) | training | ec2 (EXPLICIT fallback, never silent)
EC2_PROFILE = "WCE-EC2-ExecutionRole"
EC2_AMI = "resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
EC2_TAGS = TAGS + [{"Key": "Name", "Value": "wce-robust-job"}]


def ec2_user_data(max_runtime):
    """Guardrail: shutdown (= termination, InstanceInitiatedShutdownBehavior=terminate) after max_runtime even if the job fails."""
    return f"#!/bin/bash\nshutdown -h +{max(10, max_runtime // 60)}\n"


SYNC_PERIOD_S = 45
PRICE_USD_H = {"t3.medium": 0.0432, "t3.large": 0.0864, "c6i.large": 0.091, "c6i.xlarge": 0.182, "c7i.xlarge": 0.1911, "c7i-flex.xlarge": 0.1816, "c8i.xlarge": 0.2007}   # APPROXIMATE on-demand eu-north-1 prices (to be verified on the invoice)


def ec2_job_script(units, code_uri, out, instance, run_id, procs=0):
    """SSM script. (1) RESUME: the checkpoints already present under the output prefix are pulled back before the computation;
    (2) OBSERVABILITY: the output directory (progress, heartbeats, checkpoints, log) is synchronised to S3
    every SYNC_PERIOD_S seconds; (3) any failure, installation included, is logged and REPORTED through the exit
    code (SSM status Failed); (4) the instance shuts down (= terminates) in all cases."""
    u = json.dumps(units, separators=(",", ":")); sync = f"aws s3 sync /opt/wce/out {out} --quiet --region {REGION} --exclude '*.tmp'"
    setup = " && ".join(["dnf install -y -q python3.11 python3.11-pip tar gzip",
                         "python3.11 -m pip install --no-cache-dir -q --only-binary=:all: torch numpy --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple",
                         f"aws s3 cp {code_uri}code.tar.gz . --quiet --region {REGION}", "tar xzf code.tar.gz"])
    job = f"WCE_OUT=/opt/wce/out WCE_INSTANCE={instance} WCE_RUN_ID={run_id} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 -m wce_aws.sm_entry --procs {procs} --units '{u}'"
    timing = ('echo "{\\"setup_s\\": $((t1-t0)), \\"job_s\\": $((t2-t1)), \\"exit_code\\": $rc, \\"vcpus\\": $(nproc), '
              '\\"mem_total_kb\\": $(awk \'/MemTotal/{print $2}\' /proc/meminfo), \\"mem_available_kb\\": $(awk \'/MemAvailable/{print $2}\' /proc/meminfo), '
              f'\\"instance_type\\": \\"{instance}\\"}}" > /opt/wce/out/_timing.json')
    return ["set -x", "trap 'shutdown -h now' EXIT",                                   # automatic termination at the end of the job, whatever happens
            "export HOME=/root TMPDIR=/opt/wce/tmp", "mkdir -p /opt/wce/out /opt/wce/tmp && cd /opt/wce", "rc=0; t0=$(date +%s); t1=$t0",
            f"aws s3 sync {out} /opt/wce/out --quiet --region {REGION} --exclude '*' --include 'ckpt_*.jsonl' || true",        # resume
            f"( set +x; while true; do sleep {SYNC_PERIOD_S}; {sync}; done ) >/dev/null 2>&1 &", "syncpid=$!",
            "{ " + setup + " && t1=$(date +%s) && echo SETUP_OK && " + job + "; } > /opt/wce/out/_log.txt 2>&1 || rc=$?",
            "t2=$(date +%s); kill $syncpid 2>/dev/null || true", timing, "tail -c 1500 /opt/wce/out/_log.txt >&2", sync, "exit $rc"]


def ec2_diagnose():
    """Diagnostic WITHOUT creation: `run-instances --dry-run` under several variants (instance type, set of tags,
    profile) to locate the policy condition that denies. No resource is created, no permission modified."""
    project = [{"Key": "Project", "Value": "WCE"}]; out = {}
    def tags(t, kinds): return json.dumps([{"ResourceType": k, "Tags": t} for k in kinds])
    variants = {"complet (tags WCE + Name + WCE-Run, instance+volume, profil)": ("c5.xlarge", tags(EC2_TAGS + [{"Key": "WCE-Run", "Value": "diag"}], ("instance", "volume")), True),
                "tags TAGS seuls (4 clés), instance+volume": ("c5.xlarge", tags(TAGS, ("instance", "volume")), True),
                "tag Project=WCE seul, instance+volume": ("c5.xlarge", tags(project, ("instance", "volume")), True),
                "tag Project=WCE seul, instance uniquement": ("c5.xlarge", tags(project, ("instance",)), True),
                "tag Project=WCE seul, sans profil d'instance": ("c5.xlarge", tags(project, ("instance", "volume")), False),
                "aucun tag": ("c5.xlarge", None, True)}
    for inst in ("t3.medium", "t3.large", "t3.xlarge", "t3.2xlarge", "c6i.large", "c6i.xlarge", "c6i.2xlarge", "c7i.large", "m6i.large", "m5.large", "m5.xlarge", "c5.large", "c5.2xlarge", "c5.4xlarge"):
        variants[f"tag Project=WCE seul, type {inst}"] = (inst, tags(project, ("instance", "volume")), True)
    for extra in (TAGS[1], TAGS[2], TAGS[3], {"Key": "Name", "Value": "wce-robust-job"}, {"Key": "WCE-Run", "Value": "diag"}):   # which tag key is denied?
        variants[f"t3.large, Project=WCE + {extra['Key']}"] = ("t3.large", tags(project + [extra], ("instance", "volume")), True)
    variants["t3.large, jeu complet de tags"] = ("t3.large", tags(EC2_TAGS + [{"Key": "WCE-Run", "Value": "diag"}], ("instance", "volume")), True)
    variants["t3.large, aucun tag"] = ("t3.large", None, True)
    for label, (inst, ts, prof) in variants.items():
        cmd = ["aws", "ec2", "run-instances", "--dry-run", "--image-id", EC2_AMI, "--instance-type", inst, "--count", "1", "--region", REGION, "--profile", PROFILE]
        if ts: cmd += ["--tag-specifications", ts]
        if prof: cmd += ["--iam-instance-profile", f"Name={EC2_PROFILE}"]
        err = subprocess.run(cmd, capture_output=True, text=True).stderr
        if "c6i" in label and "DryRunOperation" not in err: out[label + " — message complet"] = err.strip()[:900]
        out[label] = "AUTORISÉ (DryRunOperation)" if "DryRunOperation" in err else ("REFUSÉ : " + (" sur ".join(re.search(r"perform: (\S+) on resource: (\S+)", err).groups()) if "on resource" in err else err.strip()[-200:]))
    print(json.dumps(out, indent=1, ensure_ascii=False)); return out


BENCH_TYPES = ("c6i.xlarge", "c7i.xlarge", "c7i-flex.xlarge", "c8i.xlarge")
EC2_AMI_ARM = "resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"


def ec2_probe(types=BENCH_TYPES):
    """READ ONLY / WITHOUT creation: for each candidate type, availability in the region, characteristics, and
    `run-instances --dry-run` with EXACTLY the tags and the profile of the real launch. No resource is created."""
    out = {}; spec = json.dumps([{"ResourceType": t, "Tags": EC2_TAGS} for t in ("instance", "volume")])
    for inst in types:
        row = {}; off = aws("ec2", "describe-instance-type-offerings", "--location-type", "availability-zone", "--filters", f"Name=instance-type,Values={inst}", check=False)
        row["zones"] = sorted(o["Location"] for o in (off or {}).get("InstanceTypeOfferings", [])) if off is not None else "lecture refusée"
        d = aws("ec2", "describe-instance-types", "--instance-types", inst, check=False); arch = "x86_64"
        if d and d.get("InstanceTypes"):
            t = d["InstanceTypes"][0]; arch = t["ProcessorInfo"]["SupportedArchitectures"][0]; row.update({"vcpus": t["VCpuInfo"]["DefaultVCpus"], "memory_gib": t["MemoryInfo"]["SizeInMiB"] / 1024, "arch": arch, "cpu": t["ProcessorInfo"].get("Manufacturer"), "ghz": t["ProcessorInfo"].get("SustainedClockSpeedInGhz")})
        cmd = ["aws", "ec2", "run-instances", "--dry-run", "--image-id", EC2_AMI_ARM if arch == "arm64" else EC2_AMI, "--instance-type", inst, "--count", "1", "--region", REGION, "--profile", PROFILE, "--tag-specifications", spec, "--iam-instance-profile", f"Name={EC2_PROFILE}"]
        err = subprocess.run(cmd, capture_output=True, text=True).stderr; m = re.search(r"perform: (\S+) on resource: (\S+)", err)
        row["dry_run"] = "AUTORISÉ (DryRunOperation)" if "DryRunOperation" in err else "REFUSÉ : " + (" sur ".join(m.groups()) if m else err.strip()[-260:]); out[inst] = row
    print(json.dumps(out, indent=1, ensure_ascii=False)); return out


def ec2_launch(name, units, code_uri, out, instance, max_runtime, run_id):
    """Launches ONE instance (profile WCE-EC2-ExecutionRole, tagged Project=WCE), waits for SSM, sends the job through SSM
    (no SSH, no key). Returns (instance_id, command_id)."""
    spec = [{"ResourceType": t, "Tags": EC2_TAGS} for t in ("instance", "volume")]     # allowed keys only (the WCE-Run key is denied by policy; the run id stays in the manifest)
    r = aws("ec2", "run-instances", "--image-id", EC2_AMI, "--instance-type", instance, "--count", "1",
            "--iam-instance-profile", f"Name={EC2_PROFILE}", "--instance-initiated-shutdown-behavior", "terminate",
            "--tag-specifications", json.dumps(spec), "--user-data", ec2_user_data(max_runtime),
            "--metadata-options", "HttpTokens=required", "--block-device-mappings", json.dumps([{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 20, "DeleteOnTermination": True}}]))
    iid = r["Instances"][0]["InstanceId"]
    try:
        for _ in range(60):                                                             # wait for the SSM agent (≤ 5 min)
            info = aws("ssm", "describe-instance-information", "--filters", f"Key=InstanceIds,Values={iid}")
            if info.get("InstanceInformationList"): break
            time.sleep(5)
        else: raise RuntimeError(f"instance {iid} never visible to SSM")
        c = aws("ssm", "send-command", "--instance-ids", iid, "--document-name", "AWS-RunShellScript", "--timeout-seconds", "600",
                "--parameters", json.dumps({"commands": ec2_job_script(units, code_uri, out, instance, run_id), "executionTimeout": [str(max_runtime)]}),
                "--comment", f"WCE {name}"[:100])
        return iid, c["Command"]["CommandId"]
    except Exception:
        aws("ec2", "terminate-instances", "--instance-ids", iid, check=False); raise


def training_spec(name, role, units, code_uri, out, instance, max_runtime, run_id):
    """Training Job fallback: same container, same entry point; outputs written to /opt/ml/output/data (→ output.tar.gz)."""
    entry = ("mkdir -p /opt/ml/output/data && cd /opt/ml/input/data/code && tar xzf code.tar.gz && "
             f"WCE_OUT=/opt/ml/output/data python3 -m wce_aws.sm_entry --units '{json.dumps(units, separators=(',', ':'))}'")
    return {"TrainingJobName": name, "RoleArn": role,
            "AlgorithmSpecification": {"TrainingImage": IMAGE, "TrainingInputMode": "File", "ContainerEntrypoint": ["bash", "-c", entry]},
            "InputDataConfig": [{"ChannelName": "code", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": code_uri, "S3DataDistributionType": "FullyReplicated"}}}],
            "OutputDataConfig": {"S3OutputPath": out}, "ResourceConfig": {"InstanceCount": 1, "InstanceType": instance, "VolumeSizeInGB": 10},
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime}, "Environment": {"WCE_RUN_ID": run_id, "OMP_NUM_THREADS": "1"}, "Tags": TAGS}


def submit(kind, keys, shards, instance, max_runtime=6 * 3600, run_id=None):
    ident = identity(); acct = ident["Account"]; role = f"arn:aws:iam::{acct}:role/{EXEC_ROLE_NAME}"
    code_hash, code_uri = package(); ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if ":" in kind:                                                                     # multi-kind campaign: "noise:0-15,caltrain:0-5,…"
        units = [{"kind": part.split(":")[0], "key": k} for part in kind.split(";") for k in parse_keys(part.split(":")[1])]
        keys = [f"{u['kind']}:{u['key']}" for u in units]; kind = "campaign"; run_id = run_id or f"wce-campaign-{ts}"; man_kind = True
    else: units = [{"kind": kind, "key": k} for k in keys]
    run_id = run_id or f"wce-{kind}-{ts}"; cfg = {"kind": kind, "keys": keys, "shards": shards, "instance": instance, "image": IMAGE, "code_hash": code_hash}
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    groups = [units[i::shards] for i in range(shards)]; jobs = []
    d = os.path.join(ROOT, "results", "aws", run_id); os.makedirs(d, exist_ok=True)
    man = {"run_id": run_id, "git_commit": git_commit(), "code_hash": code_hash, "config_hash": cfg_hash, "timestamp": ts, "region": REGION,
           "identity": ident["Arn"].split("/")[1], "instance_type": instance, "image": IMAGE, "backend": BACKEND, "kind": kind, "keys": keys, "jobs": jobs, "status": "Submitted", "max_runtime": max_runtime}
    save = lambda: json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)        # written after EACH launch: no instance orphaned from the manifest
    for i, grp in enumerate(groups):
        if not grp: continue
        name = f"{run_id}-s{i}"[:63]; out = f"s3://{BUCKET}/{PREFIX}/runs/{run_id}/{name}/"
        tmp = os.path.join(d, f"{name}.spec.json")
        if BACKEND == "ec2":
            print(f"[EXPLICIT EC2 BACKEND] {name} on {instance}")
            iid, cid = ec2_launch(name, grp, code_uri, out, instance, max_runtime, run_id)
            jobs.append({"name": name, "units": len(grp), "unit_list": grp, "output": out, "instance_id": iid, "command_id": cid}); save(); continue
        if BACKEND == "training":
            open(tmp, "w").write(json.dumps(training_spec(name, role, grp, code_uri, out, instance, max_runtime, run_id)))
            try: aws("sagemaker", "create-training-job", "--cli-input-json", f"file://{tmp}")
            finally: os.remove(tmp)
        else:
            open(tmp, "w").write(json.dumps(job_spec(name, role, grp, code_uri, out, instance, max_runtime, run_id)))
            try: aws("sagemaker", "create-processing-job", "--cli-input-json", f"file://{tmp}")
            finally: os.remove(tmp)
        jobs.append({"name": name, "units": len(grp), "unit_list": grp, "output": out}); save()
    save()
    print(json.dumps({"run_id": run_id, "jobs": [j["name"] for j in jobs]})); return run_id


def manifests(run_id=None):
    paths = sorted(glob.glob(os.path.join(ROOT, "results", "aws", "*", "manifest.json")))
    if run_id: paths = [p for p in paths if f"/{run_id}/" in p]
    return [(p, json.load(open(p))) for p in paths]


def status(run_id=None):
    for p, man in manifests(run_id):
        if not run_id and man.get("status") in ("Completed", "Failed", "Stopped"): continue
        states = []
        for j in man["jobs"]:
            if man.get("backend") == "ec2":
                inv = aws("ssm", "get-command-invocation", "--command-id", j["command_id"], "--instance-id", j["instance_id"], check=False)
                st_i = aws("ec2", "describe-instances", "--instance-ids", j["instance_id"], check=False)
                state = st_i["Reservations"][0]["Instances"][0]["State"]["Name"] if st_i and st_i.get("Reservations") else "gone"
                ssm = inv["Status"] if inv else "Unknown"; j["instance_state"] = state
                j["status"] = {"Success": "Completed", "Failed": "Failed", "TimedOut": "Failed", "Cancelled": "Stopped"}.get(ssm, "InProgress")
                if inv and inv.get("ExecutionStartDateTime") and inv.get("ExecutionEndDateTime"):
                    try: j["runtime_s"] = round((datetime.fromisoformat(inv["ExecutionEndDateTime"].replace("Z", "+00:00")) - datetime.fromisoformat(inv["ExecutionStartDateTime"].replace("Z", "+00:00"))).total_seconds(), 1)
                    except ValueError: pass
                j["failure"] = (inv.get("StandardErrorContent", "")[-300:] if inv and j["status"] == "Failed" else ""); states.append(j["status"]); continue
            if man.get("backend") == "training":
                d = aws("sagemaker", "describe-training-job", "--training-job-name", j["name"]); st, en = d.get("TrainingStartTime"), d.get("TrainingEndTime")
                j["status"] = d["TrainingJobStatus"]; j["billable_s"] = d.get("BillableTimeInSeconds")
            else:
                d = aws("sagemaker", "describe-processing-job", "--processing-job-name", j["name"]); st, en = d.get("ProcessingStartTime"), d.get("ProcessingEndTime")
                j["status"] = d["ProcessingJobStatus"]
            j["failure"] = d.get("FailureReason", "")[:300]
            if st and en: j["runtime_s"] = round((datetime.fromisoformat(en) - datetime.fromisoformat(st)).total_seconds(), 1)
            states.append(j["status"])
        man["status"] = "Completed" if all(s == "Completed" for s in states) else ("Failed" if "Failed" in states else "Stopped" if "Stopped" in states else "InProgress")
        json.dump(man, open(p, "w"), indent=1)
        print(man["run_id"], man["status"], {s: states.count(s) for s in set(states)}, [j["failure"] for j in man["jobs"] if j.get("failure")][:2])


def logs(run_id):
    for p, man in manifests(run_id):
        for j in man["jobs"][:3]:
            if man.get("backend") == "ec2":                                               # output of the SSM command (no SSH)
                inv = aws("ssm", "get-command-invocation", "--command-id", j["command_id"], "--instance-id", j["instance_id"], check=False)
                print(f"== {j['name']} [{inv['Status'] if inv else 'unknown'}]")
                if inv: print("-- stdout\n" + inv.get("StandardOutputContent", "")[-3000:] + "\n-- stderr\n" + inv.get("StandardErrorContent", "")[-3000:])
                continue
            try:
                streams = aws("logs", "describe-log-streams", "--log-group-name", "/aws/sagemaker/ProcessingJobs", "--log-stream-name-prefix", j["name"])
                for s in streams.get("logStreams", [])[:1]:
                    ev = aws("logs", "get-log-events", "--log-group-name", "/aws/sagemaker/ProcessingJobs", "--log-stream-name", s["logStreamName"], "--limit", "60")
                    print(f"== {j['name']}")
                    for e in ev.get("events", []): print(e["message"][:400])
            except AwsDenied as e: print(str(e)); return


def fetch(run_id=None):
    for p, man in manifests(run_id):
        d = os.path.dirname(p)
        aws("s3", "sync", f"s3://{BUCKET}/{PREFIX}/runs/{man['run_id']}/", os.path.join(d, "artifacts"), "--quiet", parse=False)
        for tgz in glob.glob(os.path.join(d, "artifacts", "**", "output.tar.gz"), recursive=True):       # training backend: packaged outputs
            dest = os.path.join(d, "artifacts", tgz.split(os.sep + "artifacts" + os.sep)[1].split(os.sep)[0])
            with tarfile.open(tgz) as t: t.extractall(dest)
        print(man["run_id"], "artefacts:", len(glob.glob(os.path.join(d, "artifacts", "*", "*.json"))))


def timing():
    """Tracking of the execution times per job: installation, computation, cumulative work of the sub-units, parallel efficiency,
    mean/peak CPU, memory, estimated cost — to judge whether the resources suffice. Reads only the artefacts already
    fetched (no AWS call)."""
    rows = []
    for p, man in manifests():
        d = os.path.join(os.path.dirname(p), "artifacts")
        for j in man["jobs"]:
            jd = os.path.join(d, j["name"]); inst = man.get("instance_type")
            row = {"run": man["run_id"], "job": j["name"], "kind": man.get("kind"), "backend": man.get("backend"), "instance": inst, "status": j.get("status"), "total_s": j.get("runtime_s")}
            for f, keys in (("_timing.json", ("setup_s", "job_s", "vcpus", "mem_total_kb")),
                            ("_shard_summary.json", ("wall_s", "procs", "subunits_run", "subunits_resumed", "work_s", "cpu_s", "parallel_efficiency", "cpu_avg_pct", "cpu_peak_pct", "peak_rss_mb", "mem_available_min_mb"))):
                try: t = json.load(open(os.path.join(jd, f))); row.update({k: t.get(k) for k in keys})
                except (OSError, ValueError): continue
                if f == "_shard_summary.json":
                    rt = [u["runtime_s"] for u in t.get("units", [])]; row["units"] = len(rt); row["unit_max_s"] = max(rt, default=0); row["errors"] = sum(u["status"] != "ok" for u in t.get("units", []))
            billed = (row.get("setup_s") or 0) + (row.get("job_s") or 0) or row.get("total_s")
            if billed and inst in PRICE_USD_H: row["cost_usd_est"] = round(billed / 3600 * PRICE_USD_H[inst], 4)
            rows.append(row)
    out = os.path.join(ROOT, "results", "aws_timing.json"); json.dump(rows, open(out, "w"), indent=1)
    for r in rows: print(json.dumps(r, ensure_ascii=False))
    print(f"{len(rows)} jobs -> {out}"); return rows


def _ssm_run(iid, commands, wait=180):
    """Short SSM command on a running instance; returns (status, stdout, stderr)."""
    c = aws("ssm", "send-command", "--instance-ids", iid, "--document-name", "AWS-RunShellScript", "--timeout-seconds", "120",
            "--parameters", json.dumps({"commands": commands, "executionTimeout": [str(wait)]}), "--comment", "WCE inspect")["Command"]["CommandId"]
    for _ in range(wait // 4):
        time.sleep(4); inv = aws("ssm", "get-command-invocation", "--command-id", c, "--instance-id", iid, check=False)
        if inv and inv["Status"] not in ("Pending", "InProgress", "Delayed"): return inv["Status"], inv.get("StandardOutputContent", ""), inv.get("StandardErrorContent", "")
    return "TimedOut", "", ""


def inspect_run(run_id, pyspy=True):
    """Snapshot of a RUNNING EC2 job, without interrupting it: processes/threads, load, memory, Python stack (py-spy),
    and synchronisation of the output directory (log, heartbeats, checkpoints) to S3."""
    for p, man in manifests(run_id):
        if man.get("backend") != "ec2": continue
        for j in man["jobs"]:
            cmds = ["echo '== uptime'; uptime; nproc; free -m | head -2",
                    "echo '== processus python (pid, nlwp=threads, %cpu, %mem, rss Ko, temps cpu, durée)'; ps -eo pid,ppid,nlwp,pcpu,pmem,rss,time,etime,args | grep -E 'python3.11|PID' | grep -v grep | cut -c1-220",
                    "echo '== cpu par thread'; top -H -b -n 1 | head -14",
                    "echo '== sortie'; ls -la /opt/wce/out | tail -15; tail -c 600 /opt/wce/out/_log.txt"]
            if pyspy: cmds.append("echo '== piles python'; python3.11 -m pip install -q py-spy >/dev/null 2>&1; for pid in $(pgrep -f 'multiprocessing.spawn'); do /usr/local/bin/py-spy dump --pid $pid 2>&1 | head -45; done")
            cmds.append(f"aws s3 sync /opt/wce/out {j['output']} --quiet --region {REGION}; echo sync-ok")
            st, out, err = _ssm_run(j["instance_id"], cmds); print(f"== {j['name']} [{st}]\n{out}\n{err[-600:]}")
            snap = os.path.join(os.path.dirname(p), f"inspect_{j['name']}_{int(time.time())}.txt"); open(snap, "w").write(out + "\n" + err)


def stop_run(run_id, hard=False):
    """CLEAN stop of an EC2 run: snapshot + synchronisation of the outputs, cancellation of the SSM command, then
    termination of the instance — only if it is listed in OUR manifest AND tagged Project=WCE / ManagedBy=WCE."""
    if not run_id: raise SystemExit("stop requires RUN=<run_id>")
    for p, man in manifests(run_id):
        if man.get("backend") != "ec2": continue
        if hard: print("HARD STOP (interruption test): no final synchronisation, only the checkpoints already published survive")
        else:
            try: inspect_run(run_id)
            except RuntimeError as e: print("snapshot impossible:", str(e)[:200])
        for j in man["jobs"]:
            aws("ssm", "cancel-command", "--command-id", j["command_id"], "--instance-ids", j["instance_id"], check=False, parse=False)
            di = aws("ec2", "describe-instances", "--instance-ids", j["instance_id"], check=False)
            for res in (di or {}).get("Reservations", []):
                for ins in res["Instances"]:
                    tags = {t["Key"]: t["Value"] for t in ins.get("Tags", [])}
                    if ins["State"]["Name"] in ("pending", "running", "stopping", "stopped") and tags.get("Project") == "WCE" and tags.get("ManagedBy") == "WCE":
                        aws("ec2", "terminate-instances", "--instance-ids", ins["InstanceId"], parse=False); print("terminated", ins["InstanceId"])
            j["status"] = "Stopped"
        man["status"] = "Stopped"; json.dump(man, open(p, "w"), indent=1); print(man["run_id"], "stopped")


def progress(run_id=None, quiet=False):
    """INTERNAL progress of the active runs: pulls back only the progress / heartbeat files from S3."""
    rows = []
    for p, man in manifests(run_id):
        if not run_id and man.get("status") in ("Completed", "Failed", "Stopped"): continue
        d = os.path.join(os.path.dirname(p), "artifacts")
        aws("s3", "sync", f"s3://{BUCKET}/{PREFIX}/runs/{man['run_id']}/", d, "--quiet", "--exclude", "*", "--include", "*_progress.json", "--include", "*_hb_*.json", "--include", "*_timing.json", parse=False)
        for j in man["jobs"]:
            try: g = json.load(open(os.path.join(d, j["name"], "_progress.json")))
            except (OSError, ValueError): rows.append({"job": j["name"], "state": "aucune progression publiée (installation en cours ?)"}); continue
            age = round(time.time() - g["time"]); lc = g.get("last_checkpoint") or {}
            rows.append({"job": j["name"], "instance": man.get("instance_type"), "state": g["state"], "step": g.get("step"), "items": f"{g['items_done']}/{g['items_total']}", "resumed": g.get("items_resumed"),
                         "failed": g.get("items_failed"), "elapsed_s": g["elapsed_s"], "eta_s": g.get("eta_s"), "mean_item_s": g.get("mean_item_s"), "cpu_avg_pct": g.get("cpu_avg_pct"), "cpu_peak_pct": g.get("cpu_peak_pct"),
                         "load_avg": g.get("load_avg"), "mem_available_min_mb": g.get("mem_available_min_mb"), "processes": g.get("processes"), "vcpus": g.get("vcpus"),
                         "last_checkpoint": lc.get("sub_id"), "last_checkpoint_age_s": round(time.time() - lc["time"]) if lc.get("time") else None, "progress_age_s": age,
                         "in_flight": [{"sub": h["sub"], "elapsed_s": h["sub_elapsed_s"], "rss_mb": h["rss_mb"], "at": (h.get("stack") or ["?"])[0]} for h in g.get("in_flight", [])]})
    if not quiet:
        for r in rows: print(json.dumps(r, ensure_ascii=False))
        if not rows: print("no active run")
    return rows


def resume(run_id):
    """Relaunches ONLY the unfinished jobs of a run, under the SAME names and S3 prefixes: the checkpoints already
    published are pulled back by the job and only the missing sub-units are recomputed."""
    if not run_id: raise SystemExit("resume requires RUN=<run_id>")
    (p, man), = manifests(run_id)
    if BACKEND != man.get("backend"): raise SystemExit(f"current backend {BACKEND} ≠ backend of the run {man.get('backend')} (set WCE_BACKEND explicitly)")
    if BACKEND != "ec2": raise SystemExit("resume implemented for the ec2 backend (SageMaker: resubmit the run)")
    identity(); code_hash, code_uri = package()
    if code_hash != man["code_hash"]: print(f"WARNING: the code has changed since the run ({man['code_hash']} -> {code_hash}); noted in the manifest.")
    for j in man["jobs"]:
        if j.get("status") == "Completed": continue
        print(f"[EXPLICIT EC2 BACKEND] resume {j['name']} on {man['instance_type']}")
        j["instance_id"], j["command_id"] = ec2_launch(j["name"], j["unit_list"], code_uri, j["output"], man["instance_type"], man.get("max_runtime", 6 * 3600), run_id); j["status"] = "InProgress"
        j["resumes"] = j.get("resumes", 0) + 1
    man["status"] = "Submitted"; man.setdefault("resume_code_hashes", []).append(code_hash); json.dump(man, open(p, "w"), indent=1); print(run_id, "resumed")


def clean_safe():
    """Stops ONLY the jobs listed in OUR manifests, still active AND tagged Project=WCE; deletes only robust/smoke/."""
    for p, man in manifests():
        for j in man["jobs"]:
            if man.get("backend") == "ec2":
                di = aws("ec2", "describe-instances", "--instance-ids", j["instance_id"], check=False)
                for res in (di or {}).get("Reservations", []):
                    for ins in res["Instances"]:
                        tags = {t["Key"]: t["Value"] for t in ins.get("Tags", [])}
                        if ins["State"]["Name"] in ("pending", "running", "stopping", "stopped") and tags.get("Project") == "WCE" and tags.get("ManagedBy") == "WCE":
                            aws("ec2", "terminate-instances", "--instance-ids", ins["InstanceId"], parse=False); print("terminated", ins["InstanceId"])
                continue
            tr = man.get("backend") == "training"; what = "training" if tr else "processing"
            d = aws("sagemaker", f"describe-{what}-job", f"--{what}-job-name", j["name"], check=False)
            if d and d["TrainingJobStatus" if tr else "ProcessingJobStatus"] == "InProgress":
                tags = aws("sagemaker", "list-tags", "--resource-arn", d["TrainingJobArn" if tr else "ProcessingJobArn"]).get("Tags", [])
                if any(t["Key"] == "Project" and t["Value"] == "WCE" for t in tags):
                    aws("sagemaker", f"stop-{what}-job", f"--{what}-job-name", j["name"], parse=False); print("stopped", j["name"])
    aws("s3", "rm", f"s3://{BUCKET}/{PREFIX}/smoke/", "--recursive", "--quiet", parse=False); print("safe cleanup finished")


def parse_keys(spec):
    keys = []
    for part in spec.split(","):
        if "-" in part: lo, hi = part.split("-"); keys += [str(i) for i in range(int(lo), int(hi) + 1)]
        else: keys.append(part)
    return keys


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--kind"); ap.add_argument("--keys", default="0")
    ap.add_argument("--shards", type=int, default=1); ap.add_argument("--instance", default="ml.c5.4xlarge"); ap.add_argument("--run")
    ap.add_argument("--submit", action="store_true"); ap.add_argument("--hard", action="store_true"); ap.add_argument("--max-runtime", type=int, default=6 * 3600); a = ap.parse_args()
    try:
        if a.cmd == "smoke": smoke(a.submit)
        elif a.cmd == "whoami": whoami()
        elif a.cmd == "quotas": quotas()
        elif a.cmd == "ec2-probe": ec2_probe()
        elif a.cmd == "bench-report": bench_report()
        elif a.cmd == "progress": progress(a.run)
        elif a.cmd == "resume": resume(a.run)
        elif a.cmd == "inspect": identity(); inspect_run(a.run)
        elif a.cmd == "stop": identity(); stop_run(a.run, a.hard)
        elif a.cmd == "timing": timing()
        elif a.cmd == "ec2-diagnose": identity(); ec2_diagnose()
        elif a.cmd == "check-guardrails": check_guardrails()
        elif a.cmd == "package": print(package())
        elif a.cmd == "submit": submit(a.kind, parse_keys(a.keys), a.shards, a.instance, a.max_runtime)
        elif a.cmd == "status": status(a.run)
        elif a.cmd == "logs": logs(a.run)
        elif a.cmd == "fetch": fetch(a.run)
        elif a.cmd == "clean-safe": clean_safe()
        else: raise SystemExit(f"unknown command {a.cmd}")
    except AwsDenied as e:
        print(f"CLEAN STOP — {e}\nNo permission was broadened."); sys.exit(3)
