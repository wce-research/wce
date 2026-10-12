"""Timed local Docker build; no infrastructure or deployment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

started=time.perf_counter()
subprocess.run(['docker','build','--no-cache','-t','wce-web:hardened','.'],check=True)
elapsed=time.perf_counter()-started
image=json.loads(subprocess.check_output(['docker','image','inspect','wce-web:hardened'],text=True))[0]
record={'time_utc':datetime.now(timezone.utc).isoformat(),'build_seconds':round(elapsed,3),'image_id':image['Id'],'image_bytes':image['Size'],'base_download_included':'only if registry/cache required it; not a guaranteed cold-base benchmark'}
Path('/tmp/wce-runtime-build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
