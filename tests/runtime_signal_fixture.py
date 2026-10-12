"""Test-only delayed command, used to prove SIGTERM drains an in-flight response."""
import time
from app.lab_runtime import LabSession
from app.web import Application
from app.server import serve

execute=LabSession.execute

def delayed(self,text):
    print('INFLIGHT_READY',flush=True)
    time.sleep(.3)
    return execute(self,text)

LabSession.execute=delayed
serve(Application())
