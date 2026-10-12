"""Generate test scale from tracked source without importing or running research tests."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    paths = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','tests/test_*.py'],cwd=ROOT,text=True).splitlines()
    records = []
    for path in sorted(set(paths)):
        raw = (ROOT/path).read_bytes()
        tree = ast.parse(raw)
        count = sum(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_') for node in ast.walk(tree))
        records.append({'path':path,'declared_methods':count,'sha256':sha256(raw).hexdigest()})
    digest = sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'definition':'Static test_ function declarations; no execution or dynamic-case count',
            'declared_methods':sum(row['declared_methods'] for row in records),
            'source_sha256':digest,'files':records}


if __name__ == '__main__':
    (ROOT/'app/test_inventory.json').write_text(json.dumps(inventory(),indent=2)+'\n')
