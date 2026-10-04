"""Orchestration LOCALE → SageMaker Processing (via AWS CLI, profil wce-runner). Invoqué UNIQUEMENT par le Makefile.

smoke | package | submit | status | logs | fetch | clean-safe
Chaque soumission écrit results/aws/<run_id>/manifest.json : run_id, git commit, timestamp, hash de code et de
configuration, type d'instance, région, jobs, statut, durées, chemins d'artefacts. Aucune action IAM n'est jamais
tentée (seule une LECTURE IAM est essayée au smoke test pour vérifier qu'elle est refusée)."""
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
                raise AwsDenied(f"PERMISSION MANQUANTE — {' '.join(args[:2])} : {err[:400]}")
            raise RuntimeError(f"aws {' '.join(args[:3])} : {err[:600]}")
        return None
    return json.loads(p.stdout) if parse and p.stdout.strip() else p.stdout


def check_identity(arn):
    """Échoue proprement si l'identité n'est pas le runner (jamais d'administrateur)."""
    if EXPECTED_ROLE not in arn or "Admin" in arn: raise SystemExit(f"IDENTITÉ INCORRECTE : {arn} (attendu : rôle {EXPECTED_ROLE})")
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
        for f in sorted(glob.glob(os.path.join(ROOT, "results", "deep_configs", "*.json"))):                 # configurations figées et poids du proposeur compact
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
    if submit_job:                                   # sonde de quota : premier type d'instance accepté (aucun quota n'est modifié)
        report["quota_probe"] = {}
        probes = ("t3.large", "t3.medium") if BACKEND == "ec2" else PROBE_INSTANCES   # types autorisés au runner (c5/m5 refusés par policy)
        for inst in probes:
            try: report["smoke_run"] = submit("smoke", ["0"], shards=1, instance=inst, max_runtime=1800); report["quota_probe"][inst] = "OK"; break
            except RuntimeError as e:
                if "ResourceLimitExceeded" in str(e) or "VcpuLimitExceeded" in str(e) or "InstanceLimitExceeded" in str(e): report["quota_probe"][inst] = "quota 0"
                else: raise
    print(json.dumps(report, indent=1, ensure_ascii=False)); return report


def whoami():
    """`sts get-caller-identity` avec le profil runner ; échoue si l'identité n'est pas WCE-Research-Runner."""
    ident = aws("sts", "get-caller-identity"); arn = ident["Arn"]
    print(json.dumps({"profile": PROFILE, "region": REGION, "account": ident["Account"], "arn": arn}, indent=1)); check_identity(arn)
    print(f"IDENTITÉ OK : {EXPECTED_ROLE}"); return ident


GUARDRAIL_IAM_PROBES = (("list-roles", "--max-items", "1"), ("list-users", "--max-items", "1"))   # LECTURES seulement : aucune mutation tentée


def check_guardrails():
    """Garde-fous : (1) identité = runner, jamais admin ; (2) les opérations IAM administratives restent REFUSÉES.
    Sondes en lecture seule (rien n'est créé ni modifié, même si le garde-fou était absent). Code de sortie 4 si échec."""
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


BACKEND = os.environ.get("WCE_BACKEND", "processing")        # processing (principal) | training | ec2 (repli EXPLICITE, jamais silencieux)
EC2_PROFILE = "WCE-EC2-ExecutionRole"
EC2_AMI = "resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
EC2_TAGS = TAGS + [{"Key": "Name", "Value": "wce-robust-job"}]


def ec2_user_data(max_runtime):
    """Garde-fou : extinction (= terminaison, InstanceInitiatedShutdownBehavior=terminate) après max_runtime même si le job échoue."""
    return f"#!/bin/bash\nshutdown -h +{max(10, max_runtime // 60)}\n"


SYNC_PERIOD_S = 45
PRICE_USD_H = {"t3.medium": 0.0432, "t3.large": 0.0864, "c6i.large": 0.091, "c6i.xlarge": 0.182}   # tarifs à la demande eu-north-1 APPROXIMATIFS (à vérifier sur la facture)


def ec2_job_script(units, code_uri, out, instance, run_id, procs=0):
    """Script SSM. (1) REPRISE : les checkpoints déjà présents sous le préfixe de sortie sont rapatriés avant le calcul ;
    (2) OBSERVABILITÉ : le répertoire de sortie (progression, heartbeats, checkpoints, journal) est synchronisé vers S3
    toutes les SYNC_PERIOD_S secondes ; (3) tout échec, installation comprise, est journalisé et REMONTÉ par le code de
    sortie (statut SSM Failed) ; (4) l'instance s'éteint (= se termine) dans tous les cas."""
    u = json.dumps(units, separators=(",", ":")); sync = f"aws s3 sync /opt/wce/out {out} --quiet --region {REGION} --exclude '*.tmp'"
    setup = " && ".join(["dnf install -y -q python3.11 python3.11-pip tar gzip",
                         "python3.11 -m pip install --no-cache-dir -q --only-binary=:all: torch numpy --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple",
                         f"aws s3 cp {code_uri}code.tar.gz . --quiet --region {REGION}", "tar xzf code.tar.gz"])
    job = f"WCE_OUT=/opt/wce/out WCE_INSTANCE={instance} WCE_RUN_ID={run_id} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 -m wce_aws.sm_entry --procs {procs} --units '{u}'"
    timing = ('echo "{\\"setup_s\\": $((t1-t0)), \\"job_s\\": $((t2-t1)), \\"exit_code\\": $rc, \\"vcpus\\": $(nproc), '
              '\\"mem_total_kb\\": $(awk \'/MemTotal/{print $2}\' /proc/meminfo), \\"mem_available_kb\\": $(awk \'/MemAvailable/{print $2}\' /proc/meminfo), '
              f'\\"instance_type\\": \\"{instance}\\"}}" > /opt/wce/out/_timing.json')
    return ["set -x", "trap 'shutdown -h now' EXIT",                                   # terminaison automatique à la fin du job, quoi qu'il arrive
            "export HOME=/root TMPDIR=/opt/wce/tmp", "mkdir -p /opt/wce/out /opt/wce/tmp && cd /opt/wce", "rc=0; t0=$(date +%s); t1=$t0",
            f"aws s3 sync {out} /opt/wce/out --quiet --region {REGION} --exclude '*' --include 'ckpt_*.jsonl' || true",        # reprise
            f"( set +x; while true; do sleep {SYNC_PERIOD_S}; {sync}; done ) >/dev/null 2>&1 &", "syncpid=$!",
            "{ " + setup + " && t1=$(date +%s) && echo SETUP_OK && " + job + "; } > /opt/wce/out/_log.txt 2>&1 || rc=$?",
            "t2=$(date +%s); kill $syncpid 2>/dev/null || true", timing, "tail -c 1500 /opt/wce/out/_log.txt >&2", sync, "exit $rc"]


def ec2_diagnose():
    """Diagnostic SANS création : `run-instances --dry-run` selon plusieurs variantes (type d'instance, jeu de tags,
    profil) pour localiser la condition de policy qui refuse. Aucune ressource n'est créée, aucune permission modifiée."""
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
    for extra in (TAGS[1], TAGS[2], TAGS[3], {"Key": "Name", "Value": "wce-robust-job"}, {"Key": "WCE-Run", "Value": "diag"}):   # quelle clé de tag est refusée ?
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


def ec2_launch(name, units, code_uri, out, instance, max_runtime, run_id):
    """Lance UNE instance (profil WCE-EC2-ExecutionRole, taguée Project=WCE), attend SSM, envoie le job par SSM
    (aucun SSH, aucune clé). Renvoie (instance_id, command_id)."""
    spec = [{"ResourceType": t, "Tags": EC2_TAGS} for t in ("instance", "volume")]     # clés autorisées seulement (la clé WCE-Run est refusée par policy ; l'id de run reste dans le manifeste)
    r = aws("ec2", "run-instances", "--image-id", EC2_AMI, "--instance-type", instance, "--count", "1",
            "--iam-instance-profile", f"Name={EC2_PROFILE}", "--instance-initiated-shutdown-behavior", "terminate",
            "--tag-specifications", json.dumps(spec), "--user-data", ec2_user_data(max_runtime),
            "--metadata-options", "HttpTokens=required", "--block-device-mappings", json.dumps([{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 20, "DeleteOnTermination": True}}]))
    iid = r["Instances"][0]["InstanceId"]
    try:
        for _ in range(60):                                                             # attente de l'agent SSM (≤ 5 min)
            info = aws("ssm", "describe-instance-information", "--filters", f"Key=InstanceIds,Values={iid}")
            if info.get("InstanceInformationList"): break
            time.sleep(5)
        else: raise RuntimeError(f"instance {iid} jamais visible par SSM")
        c = aws("ssm", "send-command", "--instance-ids", iid, "--document-name", "AWS-RunShellScript", "--timeout-seconds", "600",
                "--parameters", json.dumps({"commands": ec2_job_script(units, code_uri, out, instance, run_id), "executionTimeout": [str(max_runtime)]}),
                "--comment", f"WCE {name}"[:100])
        return iid, c["Command"]["CommandId"]
    except Exception:
        aws("ec2", "terminate-instances", "--instance-ids", iid, check=False); raise


def training_spec(name, role, units, code_uri, out, instance, max_runtime, run_id):
    """Repli Training Job : même conteneur, même point d'entrée ; sorties écrites dans /opt/ml/output/data (→ output.tar.gz)."""
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
    if ":" in kind:                                                                     # campagne multi-types : "noise:0-15,caltrain:0-5,…"
        units = [{"kind": part.split(":")[0], "key": k} for part in kind.split(";") for k in parse_keys(part.split(":")[1])]
        keys = [f"{u['kind']}:{u['key']}" for u in units]; kind = "campaign"; run_id = run_id or f"wce-campaign-{ts}"; man_kind = True
    else: units = [{"kind": kind, "key": k} for k in keys]
    run_id = run_id or f"wce-{kind}-{ts}"; cfg = {"kind": kind, "keys": keys, "shards": shards, "instance": instance, "image": IMAGE, "code_hash": code_hash}
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    groups = [units[i::shards] for i in range(shards)]; jobs = []
    d = os.path.join(ROOT, "results", "aws", run_id); os.makedirs(d, exist_ok=True)
    man = {"run_id": run_id, "git_commit": git_commit(), "code_hash": code_hash, "config_hash": cfg_hash, "timestamp": ts, "region": REGION,
           "identity": ident["Arn"].split("/")[1], "instance_type": instance, "image": IMAGE, "backend": BACKEND, "kind": kind, "keys": keys, "jobs": jobs, "status": "Submitted", "max_runtime": max_runtime}
    save = lambda: json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)        # écrit après CHAQUE lancement : aucune instance orpheline du manifeste
    for i, grp in enumerate(groups):
        if not grp: continue
        name = f"{run_id}-s{i}"[:63]; out = f"s3://{BUCKET}/{PREFIX}/runs/{run_id}/{name}/"
        tmp = os.path.join(d, f"{name}.spec.json")
        if BACKEND == "ec2":
            print(f"[BACKEND EC2 EXPLICITE] {name} sur {instance}")
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
            if man.get("backend") == "ec2":                                               # sortie de la commande SSM (aucun SSH)
                inv = aws("ssm", "get-command-invocation", "--command-id", j["command_id"], "--instance-id", j["instance_id"], check=False)
                print(f"== {j['name']} [{inv['Status'] if inv else 'inconnu'}]")
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
        for tgz in glob.glob(os.path.join(d, "artifacts", "**", "output.tar.gz"), recursive=True):       # backend training : sorties empaquetées
            dest = os.path.join(d, "artifacts", tgz.split(os.sep + "artifacts" + os.sep)[1].split(os.sep)[0])
            with tarfile.open(tgz) as t: t.extractall(dest)
        print(man["run_id"], "artefacts :", len(glob.glob(os.path.join(d, "artifacts", "*", "*.json"))))


def timing():
    """Suivi des temps d'exécution par job : installation, calcul, travail cumulé des sous-unités, efficacité parallèle,
    CPU moyen/pic, mémoire, coût estimé — pour juger si les ressources suffisent. Lit uniquement les artefacts déjà
    récupérés (aucun appel AWS)."""
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
    """Commande SSM courte sur une instance en marche ; renvoie (statut, stdout, stderr)."""
    c = aws("ssm", "send-command", "--instance-ids", iid, "--document-name", "AWS-RunShellScript", "--timeout-seconds", "120",
            "--parameters", json.dumps({"commands": commands, "executionTimeout": [str(wait)]}), "--comment", "WCE inspect")["Command"]["CommandId"]
    for _ in range(wait // 4):
        time.sleep(4); inv = aws("ssm", "get-command-invocation", "--command-id", c, "--instance-id", iid, check=False)
        if inv and inv["Status"] not in ("Pending", "InProgress", "Delayed"): return inv["Status"], inv.get("StandardOutputContent", ""), inv.get("StandardErrorContent", "")
    return "TimedOut", "", ""


def inspect_run(run_id, pyspy=True):
    """Instantané d'un job EC2 EN COURS, sans l'interrompre : processus/threads, charge, mémoire, pile Python (py-spy),
    et synchronisation du répertoire de sortie (journal, heartbeats, checkpoints) vers S3."""
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
    """Arrêt PROPRE d'un run EC2 : instantané + synchronisation des sorties, annulation de la commande SSM, puis
    terminaison de l'instance — uniquement si elle est listée dans NOTRE manifeste ET taguée Project=WCE / ManagedBy=WCE."""
    if not run_id: raise SystemExit("stop exige RUN=<run_id>")
    for p, man in manifests(run_id):
        if man.get("backend") != "ec2": continue
        if hard: print("ARRÊT BRUTAL (test d'interruption) : aucune synchronisation finale, seuls les checkpoints déjà publiés survivent")
        else:
            try: inspect_run(run_id)
            except RuntimeError as e: print("instantané impossible :", str(e)[:200])
        for j in man["jobs"]:
            aws("ssm", "cancel-command", "--command-id", j["command_id"], "--instance-ids", j["instance_id"], check=False, parse=False)
            di = aws("ec2", "describe-instances", "--instance-ids", j["instance_id"], check=False)
            for res in (di or {}).get("Reservations", []):
                for ins in res["Instances"]:
                    tags = {t["Key"]: t["Value"] for t in ins.get("Tags", [])}
                    if ins["State"]["Name"] in ("pending", "running", "stopping", "stopped") and tags.get("Project") == "WCE" and tags.get("ManagedBy") == "WCE":
                        aws("ec2", "terminate-instances", "--instance-ids", ins["InstanceId"], parse=False); print("terminé", ins["InstanceId"])
            j["status"] = "Stopped"
        man["status"] = "Stopped"; json.dump(man, open(p, "w"), indent=1); print(man["run_id"], "arrêté")


def progress(run_id=None, quiet=False):
    """Progression INTERNE des runs actifs : rapatrie uniquement les fichiers de progression / heartbeat depuis S3."""
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
        if not rows: print("aucun run actif")
    return rows


def resume(run_id):
    """Relance UNIQUEMENT les jobs non terminés d'un run, sous les MÊMES noms et préfixes S3 : les checkpoints déjà
    publiés sont rapatriés par le job et seules les sous-unités manquantes sont recalculées."""
    if not run_id: raise SystemExit("resume exige RUN=<run_id>")
    (p, man), = manifests(run_id)
    if BACKEND != man.get("backend"): raise SystemExit(f"backend courant {BACKEND} ≠ backend du run {man.get('backend')} (fixer WCE_BACKEND explicitement)")
    if BACKEND != "ec2": raise SystemExit("reprise implémentée pour le backend ec2 (SageMaker : resoumettre le run)")
    identity(); code_hash, code_uri = package()
    if code_hash != man["code_hash"]: print(f"ATTENTION : le code a changé depuis le run ({man['code_hash']} -> {code_hash}) ; noté dans le manifeste.")
    for j in man["jobs"]:
        if j.get("status") == "Completed": continue
        print(f"[BACKEND EC2 EXPLICITE] reprise {j['name']} sur {man['instance_type']}")
        j["instance_id"], j["command_id"] = ec2_launch(j["name"], j["unit_list"], code_uri, j["output"], man["instance_type"], man.get("max_runtime", 6 * 3600), run_id); j["status"] = "InProgress"
        j["resumes"] = j.get("resumes", 0) + 1
    man["status"] = "Submitted"; man.setdefault("resume_code_hashes", []).append(code_hash); json.dump(man, open(p, "w"), indent=1); print(run_id, "repris")


def clean_safe():
    """Arrête UNIQUEMENT les jobs listés dans NOS manifestes, encore actifs ET tagués Project=WCE ; supprime uniquement robust/smoke/."""
    for p, man in manifests():
        for j in man["jobs"]:
            if man.get("backend") == "ec2":
                di = aws("ec2", "describe-instances", "--instance-ids", j["instance_id"], check=False)
                for res in (di or {}).get("Reservations", []):
                    for ins in res["Instances"]:
                        tags = {t["Key"]: t["Value"] for t in ins.get("Tags", [])}
                        if ins["State"]["Name"] in ("pending", "running", "stopping", "stopped") and tags.get("Project") == "WCE" and tags.get("ManagedBy") == "WCE":
                            aws("ec2", "terminate-instances", "--instance-ids", ins["InstanceId"], parse=False); print("terminé", ins["InstanceId"])
                continue
            tr = man.get("backend") == "training"; what = "training" if tr else "processing"
            d = aws("sagemaker", f"describe-{what}-job", f"--{what}-job-name", j["name"], check=False)
            if d and d["TrainingJobStatus" if tr else "ProcessingJobStatus"] == "InProgress":
                tags = aws("sagemaker", "list-tags", "--resource-arn", d["TrainingJobArn" if tr else "ProcessingJobArn"]).get("Tags", [])
                if any(t["Key"] == "Project" and t["Value"] == "WCE" for t in tags):
                    aws("sagemaker", f"stop-{what}-job", f"--{what}-job-name", j["name"], parse=False); print("arrêté", j["name"])
    aws("s3", "rm", f"s3://{BUCKET}/{PREFIX}/smoke/", "--recursive", "--quiet", parse=False); print("nettoyage sûr terminé")


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
        else: raise SystemExit(f"commande inconnue {a.cmd}")
    except AwsDenied as e:
        print(f"ARRÊT PROPRE — {e}\nAucune permission n'a été élargie."); sys.exit(3)
