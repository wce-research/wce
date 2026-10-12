"""Production runtime/security regressions; all commands exposed through Makefile."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import json
import threading
import time
import unittest
from unittest.mock import patch

from app.lab_runtime import BudgetExceeded, LabSession, TokenBucket
from app.runtime_config import RuntimeConfig
from app.web import Application
from app.server import build_server, close_server


def request(app,path='/api/command',command='Where is the ball?',host='localhost',origin=None,cookie='',extra=None,method='POST',raw=None):
    body = raw if raw is not None else json.dumps({'command':command}).encode()
    environ = {'PATH_INFO':path,'REQUEST_METHOD':method,'HTTP_HOST':host,
        'HTTP_X_WCE_LANGUAGE':'en','CONTENT_TYPE':'application/json','CONTENT_LENGTH':str(len(body)),
        'wsgi.input':BytesIO(body),'HTTP_COOKIE':cookie}
    if origin is not None:environ['HTTP_ORIGIN']=origin
    if extra:environ.update(extra)
    response={}
    def start(status,headers):response.update(status=int(status.split()[0]),headers=dict(headers))
    response['body']=b''.join(app(environ,start))
    return response


class RuntimeTests(unittest.TestCase):
    def test_production_requires_valid_explicit_origin(self):
        for env in ({'WCE_MODE':'oops'},{'WCE_MODE':'production'},
                    {'WCE_MODE':'production','WCE_PUBLIC_ORIGIN':'http://wcelabs.com'},
                    {'WCE_MODE':'production','WCE_PUBLIC_ORIGIN':'https://wcelabs.com/'},
                    {'WCE_MODE':'production','WCE_PUBLIC_ORIGIN':'https://bad host.com'},
                    {'WCE_MODE':'production','WCE_PUBLIC_ORIGIN':'https://wcelabs.\ncom'},
                    {'WCE_THREADS':'0'},{'WCE_MAX_SESSIONS':'101'},{'WCE_THREADS':'many'},
                    {'WCE_SESSION_TTL':'900','WCE_SESSION_LIFETIME':'30'}):
            with self.subTest(env=env),self.assertRaises(ValueError):RuntimeConfig.from_env(env)
        self.assertEqual(RuntimeConfig.from_env({'WCE_MODE':'production','WCE_PUBLIC_ORIGIN':'https://wcelabs.com'}).mode,'production')

    def test_cookie_production_is_secure_expiring_httponly(self):
        app=Application(RuntimeConfig(mode='production',public_origin='https://wcelabs.com'))
        res=request(app,host='wcelabs.com',origin='https://wcelabs.com')
        self.assertEqual(res['status'],200)
        for item in ('; Secure','HttpOnly','SameSite=Strict','Max-Age=3600','Path=/'):
            self.assertIn(item,res['headers']['Set-Cookie'])

    def test_cookie_development_ignores_spoofed_proxy_scheme(self):
        app=Application(RuntimeConfig())
        res=request(app,extra={'HTTP_X_FORWARDED_PROTO':'https','HTTP_FORWARDED':'proto=https'})
        self.assertEqual(res['status'],200)
        self.assertNotIn('Secure',res['headers']['Set-Cookie'])

    def test_production_host_origin_and_forwarded_headers(self):
        app=Application(RuntimeConfig(mode='production',public_origin='https://wcelabs.com'))
        for host,origin in [('evil.test','https://wcelabs.com'),('wcelabs.com',None),('wcelabs.com','http://wcelabs.com'),('wcelabs.com','https://evil.test')]:
            res=request(app,host=host,origin=origin,extra={'HTTP_X_FORWARDED_HOST':'wcelabs.com','HTTP_X_FORWARDED_PROTO':'https'})
            self.assertIn(res['status'],(400,403))
        self.assertEqual(request(app,path='/healthz',host='127.0.0.1',method='GET')['status'],200)
        self.assertEqual(request(app,path='/en/',host='evil.test',method='GET')['status'],400)

    def test_health_readiness_and_draining(self):
        app=Application(RuntimeConfig())
        self.assertEqual(request(app,path='/readyz',method='GET')['status'],200)
        app.ready=False
        self.assertEqual(request(app,path='/readyz',method='GET')['status'],503)
        self.assertEqual(request(app,path='/healthz',method='GET')['status'],200)
        self.assertEqual(request(app,path='/en/',method='GET')['status'],503)
        self.assertEqual(len(app.sessions),0)

    def test_get_failure_is_controlled_and_redacted(self):
        with patch('app.web.render_page',side_effect=RuntimeError('PRIVATE_SENTINEL')):
            res=request(Application(),path='/en/',method='GET')
        self.assertEqual(res['status'],500)
        self.assertNotIn(b'PRIVATE_SENTINEL',res['body'])

    def test_session_capacity_does_not_evict_active_users(self):
        app=Application(replace(RuntimeConfig(),sessions=1))
        first=request(app);cookie=first['headers']['Set-Cookie'].split(';')[0]
        self.assertEqual(request(app)['status'],429)
        self.assertEqual(request(app,cookie=cookie)['status'],200)
        self.assertEqual(len(app.sessions),1)

    def test_session_ttl_and_absolute_expiry(self):
        app=Application(RuntimeConfig())
        token,session,_=app.session(None)
        session.created-=app.config.session_lifetime+1
        replacement,_,created=app.session(token)
        self.assertTrue(created);self.assertNotEqual(token,replacement)
        with app.lock:app.sessions[replacement]=(app.sessions[replacement][0],time.monotonic()-app.config.session_ttl-1)
        self.assertNotEqual(app.session(replacement)[0],replacement)

    def test_repeated_commands_stop_at_budget(self):
        session=LabSession(replace(RuntimeConfig(),commands=3))
        for _ in range(3):session.execute('Où est la balle ?')
        with self.assertRaises(BudgetExceeded):session.execute('Où est la balle ?')
        self.assertEqual(session.commands,3)

    def test_simulation_budget_reserves_before_work(self):
        session=LabSession(replace(RuntimeConfig(),simulation_steps=10))
        session.execute('Avance le temps de 1 seconde')
        old=session.world.time
        with self.assertRaises(BudgetExceeded):session.execute('Avance le temps de 1 seconde')
        self.assertEqual(session.world.time,old)
        self.assertEqual(session.steps,10)

    def test_history_and_graph_are_bounded_without_resetting_world(self):
        session=LabSession(replace(RuntimeConfig(),history=8,graph_nodes=4))
        for _ in range(200):
            session.execute('Soulève la balle à 2 mètres')
            self.assertLessEqual(len(session.world.history),8)
            self.assertLessEqual(len(session.world.transform_graph.nodes),4)
        self.assertGreater(session.checkpoints,0)
        self.assertEqual(session.world.entities['balle'].properties['y'],2)
        self.assertIn('2',session.execute('Où est la balle ?'))
        # Keep the ball airborne for all 600 steps, exercising default graph/history caps.
        airborne=LabSession(RuntimeConfig())
        airborne.execute('Soulève la balle à 1000000 mètres')
        airborne.execute('Avance le temps de 60 secondes')
        self.assertEqual(airborne.steps,600)
        self.assertLessEqual(len(airborne.world.history),128)
        self.assertLessEqual(len(airborne.world.transform_graph.nodes),64)
        self.assertGreater(airborne.checkpoints,1)
        self.assertAlmostEqual(airborne.world.time,60,places=8)


    def test_numeric_parameters_are_finite_and_bounded(self):
        for text in ('Soulève la balle à '+('9'*400)+' mètres','Soulève la balle à 1000001 mètres','Avance le temps de '+('9'*400)+' secondes'):
            with self.subTest(text=text[:40]),self.assertRaises(ValueError):LabSession(RuntimeConfig()).execute(text)

    def test_body_command_and_malformed_inputs(self):
        app=Application()
        self.assertEqual(request(app,raw=b'x'*4097)['status'],413)
        self.assertEqual(request(app,command='x'*501)['status'],400)
        self.assertEqual(request(app,raw=b'not json')['status'],400)
        self.assertEqual(request(app,raw=b'[]')['status'],400)

    def test_saturation_returns_retryable_status_without_new_state(self):
        app=Application()
        app.api_slots.acquire();app.api_slots.acquire()
        try:
            res=request(app)
            self.assertEqual(res['status'],503)
            self.assertEqual(res['headers']['Retry-After'],'1')
            self.assertFalse(app.sessions)
        finally:app.api_slots.release();app.api_slots.release()

    def test_concurrent_same_session_is_rejected_without_waiting(self):
        app=Application();first=request(app);cookie=first['headers']['Set-Cookie'].split(';')[0]
        session=next(iter(app.sessions.values()))[0]
        session.lock.acquire()
        try:self.assertEqual(request(app,cookie=cookie)['status'],409)
        finally:session.lock.release()

    def test_concurrent_session_creation_is_bounded(self):
        app=Application(replace(RuntimeConfig(),sessions=2))
        with ThreadPoolExecutor(max_workers=20) as pool:
            results=list(pool.map(lambda _:request(app),range(20)))
        self.assertLessEqual(len(app.sessions),2)
        self.assertTrue(all(res['status'] in (200,429,503) for res in results))

    def test_global_budget_has_no_unbounded_key_registry(self):
        app=Application(replace(RuntimeConfig(),api_burst=1,api_rate=1))
        self.assertEqual(request(app)['status'],200)
        self.assertEqual(request(app)['status'],429)
        self.assertLessEqual(app.bucket.tokens,1)

    def test_transport_limits_and_untrusted_proxy_configuration(self):
        app=Application();server=build_server(app)
        try:
            self.assertEqual(server.adj.max_request_body_size,4096)
            self.assertEqual(server.adj.max_request_header_size,8192)
            self.assertEqual(server.adj.threads,4)
            self.assertEqual(server.adj.connection_limit,128)
            self.assertGreaterEqual(server.adj.asyncore_loop_timeout,1)
            self.assertIsNone(server.adj.trusted_proxy)
            self.assertTrue(server.adj.clear_untrusted_proxy_headers)
        finally:close_server(server,app)

    def test_sigterm_drains_an_inflight_response(self):
        import http.client
        import os
        from pathlib import Path
        import re
        import signal
        import subprocess
        root=Path(__file__).resolve().parents[1]
        process=subprocess.Popen(['make','--no-print-directory','runtime-signal-fixture'],cwd=root,
            env={**os.environ,'WCE_HOST':'127.0.0.1','WCE_PORT':'0'},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        result=[]
        try:
            while True:
                line=process.stderr.readline()
                match=re.search(r'WCE listening on 127\.0\.0\.1:(\d+)',line)
                if match:break
                if process.poll() is not None:raise AssertionError('Fixture failed before startup')
            port=int(match[1])
            def command():
                connection=http.client.HTTPConnection('127.0.0.1',port,timeout=3)
                try:
                    connection.request('POST','/api/command',json.dumps({'command':'Où est la balle ?'}),{'Content-Type':'application/json'})
                    response=connection.getresponse();result.append((response.status,response.read()))
                finally:connection.close()
            worker=threading.Thread(target=command);worker.start()
            while 'INFLIGHT_READY' not in process.stdout.readline():
                if process.poll() is not None:raise AssertionError('Fixture failed before command')
            # Signal the Python process, not make's wrapper process.
            children=subprocess.check_output(['ps','--ppid',str(process.pid),'-o','pid='],text=True).split()
            self.assertEqual(len(children),1)
            started=time.monotonic();os.kill(int(children[0]),signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5),0)
            self.assertLess(time.monotonic()-started,2)
            worker.join(timeout=3)
            self.assertEqual(result[0][0],200)
        finally:
            if process.poll() is None:process.kill();process.wait()
            process.stdout.close();process.stderr.close()

    def test_startup_validation_rejects_missing_public_files(self):
        app=Application()
        with patch('pathlib.Path.is_file',return_value=False),self.assertRaises(RuntimeError):
            app.validate_startup()
        self.assertFalse(app.ready)

    def test_queue_is_bounded_under_concurrent_producers(self):
        from app.server import BoundedDispatcher
        class Task:
            def cancel(self):pass
        dispatcher=BoundedDispatcher(5)
        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(lambda _:dispatcher.add_task(Task()),range(100)))
        self.assertEqual(len(dispatcher.queue),5)
        dispatcher.shutdown(timeout=1)

    def test_real_http_rejects_large_inputs_and_proxy_host_spoof(self):
        import http.client
        app=Application(RuntimeConfig(mode='production',public_origin='https://wcelabs.com'))
        server=build_server(app)
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        try:
            port=int(server.effective_port)
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=2)
            connection.request('POST','/api/command',b'x'*5000,{'Content-Type':'application/json','Host':'wcelabs.com','Origin':'https://wcelabs.com'})
            response=connection.getresponse();self.assertEqual(response.status,413);response.read();connection.close()
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=2)
            connection.request('GET','/en/',headers={'Host':'evil.test','X-Forwarded-Host':'wcelabs.com','X-Forwarded-Proto':'https'})
            response=connection.getresponse();self.assertEqual(response.status,400);response.read();connection.close()
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=2)
            connection.request('GET','/healthz',headers={'X-Large':'x'*9000})
            response=connection.getresponse();self.assertEqual(response.status,431);response.read();connection.close()
        finally:
            close_server(server,app);thread.join(timeout=2)

    def test_malformed_api_requests_cannot_bypass_budget(self):
        app=Application(replace(RuntimeConfig(),api_burst=1,api_rate=1))
        self.assertEqual(request(app,raw=b'not json')['status'],400)
        self.assertEqual(request(app,raw=b'not json')['status'],429)
        self.assertFalse(app.sessions)
