"""Check versioned local hardening evidence against source; no network/cloud calls."""
from hashlib import sha256
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parents[1]
evidence=root/'docs/runtime_hardening_evidence'
load=lambda name:json.loads((evidence/name).read_text())
image=load('build.json')['image_id']
for name in ('local_1cpu.json','local_025cpu.json'):
    record=load(name)
    assert record['image_id']==image
    assert record['tool_sha256']==sha256((root/'tools/production_benchmark.py').read_bytes()).hexdigest()
    for path,digest in record['source_sha256'].items():
        assert (root/path).is_relative_to(root/'app')
        assert sha256((root/path).read_bytes()).hexdigest()==digest,path
    assert record['exit_state']['ExitCode']==0 and not record['exit_state']['OOMKilled']
    for scenario in record['scenarios']:
        assert scenario['requests']==scenario['successes']+sum(scenario['errors'].values())
        assert abs(scenario['successful_rps']-scenario['successes']/scenario['elapsed_seconds'])<.1
one=load('local_1cpu.json')
assert all(not scenario['errors'] for scenario in one['scenarios'][:3])
assert set(one['scenarios'][3]['errors'])<= {'429','503','409'}
assert one['scenarios'][1]['successful_latency_ms']['p95']<500
status=load('local_status.json')
assert status['runtime_sources_match_workspace'] and status['test_inventory_matches_workspace']
for lifecycle in load('lifecycle.json')[:2]:
    assert lifecycle['image_id']==status['image_id']
    assert lifecycle['exit_code']==0 and not lifecycle['oom']
    assert statistics.median(lifecycle['idle_cpu_samples'])<5
browser=load('browser_summary.json')
assert browser['renders']==408 and browser['wcag_aa_scans']==160 and not browser['violations']
print('Runtime/image/source/benchmark/lifecycle/browser evidence checks passed')
