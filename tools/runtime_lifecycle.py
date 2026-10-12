"""Local-only Docker lifecycle and image hygiene checks; no AWS calls."""
import json
from pathlib import Path
import subprocess
import statistics
import os
import time
from http.client import HTTPConnection


def docker(*args):
    return subprocess.check_output(['docker',*args],text=True,stderr=subprocess.STDOUT).strip()


def request(port,path='/readyz',body=None):
    connection=HTTPConnection('127.0.0.1',port,timeout=2)
    headers={'Host':'wcelabs.com','Origin':'https://wcelabs.com','Content-Type':'application/json','X-WCE-Language':'en'}
    try:
        connection.request('POST' if body else 'GET',path,body,headers)
        response=connection.getresponse();data=response.read()
        return response.status,dict(response.getheaders()),data
    finally:connection.close()


def main():
    image=os.environ.get('WCE_RUNTIME_IMAGE','wce-web:hardened')
    image_id=json.loads(docker('image','inspect',image))[0]['Id']
    history=docker('history','--no-trunc','--format','{{.CreatedBy}}',image)
    assert not any(name in history for name in ('AWS_ACCESS_KEY','AWS_SECRET_ACCESS_KEY','GOOGLE_APPLICATION_CREDENTIALS','PRIVATE_SENTINEL'))
    results=[]
    for mode in ('development','production'):
        container=None
        try:
            env=['-e','WCE_MODE='+mode]
            if mode=='production':env+=['-e','WCE_PUBLIC_ORIGIN=https://wcelabs.com']
            container=docker('run','-d','--read-only','--tmpfs','/tmp:rw,noexec,nosuid,size=16m',
                '--cap-drop','ALL','--security-opt','no-new-privileges','--memory','256m','--cpus','1','--pids-limit','64',
                '-p','127.0.0.1::8000',*env,image)
            port=int(docker('port',container,'8000/tcp').rsplit(':',1)[1])
            for _ in range(100):
                try:
                    if request(port)[0]==200:break
                except (OSError,TimeoutError):pass
                time.sleep(.05)
            else:raise RuntimeError('Readiness failed')
            idle_cpu=[float(docker('stats','--no-stream','--format','{{.CPUPerc}}',container).rstrip('%')) for _ in range(3)]
            assert statistics.median(idle_cpu)<5,idle_cpu
            status,headers,_=request(port,'/api/command',json.dumps({'command':'Lift the ball to 2 metres'}))
            assert status==200
            cookie=headers['Set-Cookie']
            assert ('; Secure' in cookie)==(mode=='production')
            assert all(item in cookie for item in ('HttpOnly','SameSite=Strict','Max-Age='))
            assert docker('exec',container,'python','-c','import os;print(os.getuid())')!='0'
            hygiene=docker('exec',container,'python','-c',"from pathlib import Path; import os; p=Path('/app'); assert not any((p/x).exists() for x in ('.aws','.git','.env','secrets','.secrets','.venv-runtime')); assert not any(x.suffix in ('.key','.pem','.salt') for x in p.rglob('*')); assert not any(k.startswith(('AWS_','GOOGLE_')) for k in os.environ); print('application image hygiene passed')")
            started=time.perf_counter()
            if mode=='production':
                docker('kill','--signal','TERM',container)
                docker('wait',container)
            else:docker('stop','--time','8',container)
            elapsed=time.perf_counter()-started
            state=json.loads(docker('inspect',container))[0]['State']
            assert state['ExitCode']==0 and not state['OOMKilled'],state
            assert elapsed<6,elapsed
            log=docker('logs',container)
            assert 'shutdown complete' in log
            assert 'PRIVATE_SENTINEL' not in log
            results.append({'image_id':image_id,'mode':mode,'signal':'SIGTERM' if mode=='production' else 'docker stop','stop_seconds':round(elapsed,3),'exit_code':state['ExitCode'],'oom':state['OOMKilled'],'idle_cpu_samples':idle_cpu,'hygiene':hygiene})
        finally:
            if container:docker('rm','-f',container)
    # Missing production configuration must fail before listening.
    container=None
    try:
        container=docker('run','-d','--network','none','-e','WCE_MODE=production',image)
        exit_code=int(docker('wait',container));assert exit_code!=0
        results.append({'mode':'invalid production','exit_code':exit_code})
    finally:
        if container:docker('rm','-f',container)
    Path('/tmp/wce-runtime-lifecycle.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
