"""Verify the existing local web container without changing it or exposing a port."""
from http.client import HTTPConnection
import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def docker(*args):
    return subprocess.check_output(['docker',*args],text=True).strip()

container=docker('compose','ps','-q','web')
assert container,'Existing web service is not running'
state=json.loads(docker('inspect',container))[0]
ports=state['NetworkSettings']['Ports']['8000/tcp']
assert len(ports)==1 and ports[0]['HostIp']=='127.0.0.1'
port=int(ports[0]['HostPort'])
assert port==3018,'Reserved local web port changed'
assert state['State']['Health']['Status']=='healthy'
assert state['HostConfig']['ReadonlyRootfs']
assert state['HostConfig']['Memory']==256*1024*1024
assert state['HostConfig']['PidsLimit']==64
assert state['HostConfig']['NanoCpus']==1_000_000_000
assert docker('exec',container,'python','-c','import os;print(os.getuid())')=='999'
root=Path(__file__).resolve().parents[1]
source_hashes=json.loads(docker('exec',container,'python','-c',"from pathlib import Path; import json,hashlib; root=Path('/app'); print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'app').glob('*.py')}))"))
assert all(sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in source_hashes.items())
inventory_digest=docker('exec',container,'python','-c',"from pathlib import Path; import hashlib; print(hashlib.sha256(Path('/app/app/test_inventory.json').read_bytes()).hexdigest())")
assert inventory_digest==sha256((root/'app/test_inventory.json').read_bytes()).hexdigest()
from app.site import PAGES
checks=[]
for path in ['/healthz','/readyz']+['/'+locale+(route if route!='/' else '/') for locale in ('en','fr') for route in PAGES]:
    connection=HTTPConnection('127.0.0.1',port,timeout=3)
    try:
        connection.request('GET',path)
        response=connection.getresponse();body=response.read()
        assert response.status==200,(path,response.status)
        checks.append(path)
    finally:connection.close()
report={'loopback_port':port,'health':'healthy','checked_routes':len(checks),'uid':999,'memory_limit_mib':256,'cpu_limit':1,'pids_limit':64,'read_only':True,'image_id':state['Image'],'runtime_sources_match_workspace':True,'test_inventory_matches_workspace':True}
Path('/tmp/wce-runtime-local-status.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
