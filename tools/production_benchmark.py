"""Bounded local Docker measurement. No cloud calls; never targets a public URL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from http.client import HTTPConnection
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True).strip()


def percentile(values, share):
    ordered = sorted(values)
    return round(ordered[min(len(ordered)-1, int((len(ordered)-1)*share))], 3) if ordered else None


def request(port, path, command=None, cookie=None):
    connection = HTTPConnection('127.0.0.1', port, timeout=4)
    started = time.perf_counter()
    headers = {'X-WCE-Language': 'en'}
    if cookie: headers['Cookie'] = cookie
    if command is not None:
        body = json.dumps({'command': command})
        headers['Content-Type'] = 'application/json'
        headers['Origin'] = 'http://127.0.0.1:'+str(port)
    else: body = None
    try:
        connection.request('POST' if command else 'GET', path, body, headers)
        response = connection.getresponse()
        data = response.read()
        return response.status, (time.perf_counter()-started)*1000, len(data), response.getheader('Set-Cookie')
    finally: connection.close()


def sample(container):
    raw=json.loads(docker('stats', '--no-stream', '--format', '{{json .}}', container))
    return {key:raw[key] for key in ('CPUPerc','MemUsage','PIDs')}


def measure(container, port, users, seconds, workload):
    stop = time.perf_counter()+seconds
    records = []; lock = threading.Lock(); samples = []
    def worker(index):
        cookie = None; local = []; sequence = index
        time.sleep(index/users*0.1)
        while time.perf_counter()<stop:
            item = workload[sequence % len(workload)]; sequence += 1
            attempt_started = time.perf_counter()
            try:
                status, ms, count, new_cookie = request(port,*item,cookie=cookie)
                if new_cookie: cookie = new_cookie.split(';')[0]
                local.append((status,ms,count))
            except Exception as error:
                local.append((type(error).__name__,(time.perf_counter()-attempt_started)*1000,0))
        with lock: records.extend(local)
    def monitor():
        while time.perf_counter()<stop:
            samples.append(sample(container))
    started = time.perf_counter()
    observer = threading.Thread(target=monitor);observer.start()
    with ThreadPoolExecutor(max_workers=users) as pool:
        list(pool.map(worker,range(users)))
    observer.join(); elapsed = time.perf_counter()-started
    successful = [row[1] for row in records if row[0]==200]
    errors = {}
    for status,_,_ in records:
        if status!=200: errors[str(status)] = errors.get(str(status),0)+1
    return {'users':users,'requested_seconds':seconds,'elapsed_seconds':round(elapsed,3),
            'requests':len(records),'successes':len(successful),'errors':errors,
            'successful_rps':round(len(successful)/elapsed,2),
            'successful_latency_ms':{name:percentile(successful,p) for name,p in [('p50',.5),('p95',.95),('p99',.99)]},
            'all_attempt_latency_ms':{name:percentile([r[1] for r in records],p) for name,p in [('p50',.5),('p95',.95),('p99',.99)]},
            'bytes':sum(row[2] for row in records),'docker_samples':samples}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',default='wce-web:local')
    parser.add_argument('--cpus',type=float,default=1);parser.add_argument('--seconds',type=int,default=6)
    parser.add_argument('--output',default='/tmp/wce-production-benchmark.json');args=parser.parse_args()
    if not 0<args.cpus<=2 or not 2<=args.seconds<=20: parser.error('Bounded local test only')
    image=json.loads(docker('image','inspect',args.image))[0]
    result={'time_utc':datetime.now(timezone.utc).isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'tool_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'image_id':image['Id'],'image_bytes':image['Size'],'cpu_limit':args.cpus,'memory_limit_mib':256,
            'method':'Closed-loop users without think time, fresh HTTP/1.0 connections; successful and attempted latency separated; not an AWS capacity test'}
    container=None
    try:
        started=time.perf_counter()
        container=docker('run','-d','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
                         '--memory','256m','--cpus',str(args.cpus),'--pids-limit','256',
                         '--log-opt','max-size=2m','--log-opt','max-file=1','-p','127.0.0.1::8000',args.image)
        port=int(docker('port',container,'8000/tcp').rsplit(':',1)[1])
        for attempt in range(100):
            try:
                if request(port,'/healthz')[0]==200:break
            except Exception:time.sleep(.05)
        else:raise RuntimeError('Local container not ready')
        result['docker_run_to_ready_seconds']=round(time.perf_counter()-started,3)
        result['runtime']=docker('exec',container,'python','-c','import platform,os;print(platform.python_version(),os.getuid())')
        result['idle_samples']=[sample(container) for _ in range(3)]
        workload=[('/en/',None),('/fr/',None),('/en/research/programme',None),('/en/research/high-precision',None),('/en/architecture/research-map',None),('/styles.css',None),('/science.js',None),('/research/deep_high_precision_summary.json',None),('/healthz',None),('/api/command','Where is the ball?')]
        for path,command in workload:assert request(port,path,command)[0]==200,(path,command)
        result['workload']=[{'path':p,'command':c} for p,c in workload]
        result['scenarios']=[]
        for users in (1,10,50,100):
            record=measure(container,port,users,args.seconds,workload);result['scenarios'].append(record)
            print('users',users,'rps',record['successful_rps'],'latency',record['successful_latency_ms'],'errors',record['errors'],flush=True)
        result['bounded_history_probe']={'cycles':10,'commands_per_cycle':2,'duration_per_cycle':60,'before':sample(container)}
        cookie=None;timings=[]
        for _ in range(10):
            status,_,_,new_cookie=request(port,'/api/command','Lift the ball to 2 metres',cookie)
            assert status==200
            if new_cookie:cookie=new_cookie.split(';')[0]
            status,ms,_,new_cookie=request(port,'/api/command','Advance time by 60 seconds',cookie)
            assert status==200;timings.append(ms)
            if new_cookie:cookie=new_cookie.split(';')[0]
        result['bounded_history_probe'].update(after=sample(container),latency_ms=timings)
        started=time.perf_counter();docker('stop','--time','5',container)
        result['stop_seconds']=round(time.perf_counter()-started,3)
        result['exit_state']=json.loads(docker('inspect',container))[0]['State']
        # No paths, host identity or environment configuration is recorded.
        result['exit_state']={k:result['exit_state'][k] for k in ('ExitCode','OOMKilled','Error')}
    finally:
        if container:docker('rm','-f',container)  # Only the container ID created by this invocation.
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('Recorded',args.output)


if __name__=='__main__':main()
