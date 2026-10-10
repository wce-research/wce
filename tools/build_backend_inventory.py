"""Regenerate public source metadata without executing or exposing module contents."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ('core', 'language', 'physics', 'logic', 'app')
EXCLUDE = {'__init__.py', 'research_catalog.py', 'presentation.py', 'content.py', 'i18n.py', 'site.py', 'localization.py', 'diagrams.py', 'evidence.py', 'understanding.py'}


def inventory():
    records=[]
    for folder in FOLDERS:
        for path in sorted((ROOT/folder).glob('*.py')):
            if path.name in EXCLUDE or (folder == 'app' and path.name not in {'session.py', 'repl.py'}):continue
            tree=ast.parse(path.read_text())
            classes=[node.name for node in tree.body if isinstance(node,ast.ClassDef)]
            functions=[node.name for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and not node.name.startswith('_')]
            records.append({'path':path.relative_to(ROOT).as_posix(),'group':folder,'classes':classes,'functions':functions})
    return records

if __name__ == '__main__':
    target=ROOT/'app/backend_inventory.json'
    target.write_text(json.dumps(inventory(),ensure_ascii=False,indent=2)+'\n')
    print(f'{len(inventory())} modules inventoried in {target.relative_to(ROOT)}')
