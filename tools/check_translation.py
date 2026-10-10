"""Translation invariant checker.

The only transformation allowed by the English normalisation is `French text -> equivalent English text`.
This tool compares the working tree with a git reference and fails on anything else.

Python files
    * the AST must be identical once every string constant is masked (same code, numbers, identifiers);
    * a string constant may change only if it is a docstring or a human-facing message; strings in protected
      positions (dict keys, subscripts, comparisons, keyword names) must be byte-identical;
    * changed non-docstring strings are listed so that a human can audit them.

Markdown / text files
    * the sequence of numbers must be identical (decimal comma -> decimal point and thin-space thousands ->
      comma are the only accepted renderings);
    * the multiset of backticked code spans must be identical;
    * the table shape (number of cells per row) must be identical.

Usage: python tools/check_translation.py [--ref REF] [--json] FILE...
"""
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter

NUM = re.compile(r"\d+(?:[    ,.]\d+)*")
SPAN = re.compile(r"`[^`\n]+`")


def git_show(ref, path):
    p = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True)
    return p.stdout.decode("utf-8", errors="replace") if p.returncode == 0 else None


def _docstring_nodes(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            f = n.body[0]
            if isinstance(f, ast.Expr) and isinstance(f.value, ast.Constant) and isinstance(f.value.value, str): out.add(id(f.value))
    return out


def _protected_nodes(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict): out.update(id(k) for k in n.keys if k is not None)
        elif isinstance(n, ast.Subscript): out.update(id(c) for c in ast.walk(n.slice))
        elif isinstance(n, ast.Compare): out.update(id(c) for part in [n.left] + n.comparators for c in ast.walk(part))
    return out


def _strings(tree):
    """Ordered string constants with their role: doc | protected | other."""
    doc, prot = _docstring_nodes(tree), _protected_nodes(tree); out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str): out.append(("doc" if id(n) in doc else "protected" if id(n) in prot else "other", n.value, getattr(n, "lineno", 0)))
    return out


class _Mask(ast.NodeTransformer):
    def visit_Constant(self, node): return ast.copy_location(ast.Constant("S"), node) if isinstance(node.value, str) else node


def check_python(old, new):
    errors, changed = [], []
    try: a, b = ast.parse(old), ast.parse(new)
    except SyntaxError as e: return [f"syntax error: {e}"], []
    sa, sb = _strings(a), _strings(b)
    if ast.dump(_Mask().visit(ast.parse(old))) != ast.dump(_Mask().visit(ast.parse(new))): errors.append("code structure changed (AST differs beyond string contents)")
    if len(sa) != len(sb): errors.append(f"number of string constants changed ({len(sa)} -> {len(sb)})"); return errors, changed
    for (ra, va, la), (rb, vb, lb) in zip(sa, sb):
        if va == vb: continue
        if ra == "protected" or rb == "protected": errors.append(f"line {lb}: protected string changed: {va!r} -> {vb!r}")
        elif ra != "doc": changed.append({"line": lb, "old": va, "new": vb})
        if NUM.findall(va) and _nums(va) != _nums(vb): errors.append(f"line {lb}: numbers changed inside a string or docstring")
        if va.count("{") != vb.count("{") or va.count("%") != vb.count("%"): errors.append(f"line {lb}: format placeholders changed: {va[:60]!r}")
    ca, cb = _comments(old), _comments(new)
    if len(ca) != len(cb): errors.append(f"number of comments changed ({len(ca)} -> {len(cb)})")
    elif [_nums(c) for c in ca] != [_nums(c) for c in cb]: errors.append("numbers changed inside a comment")
    return errors, changed


def _comments(src):
    try: return [t.string for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError): return []


def _nums(text): return [re.sub(r"[    ,.]", "", m) for m in NUM.findall(text)]


def check_text(old, new):
    errors = []; na, nb = _nums(old), _nums(new)
    if na != nb:
        i = next((k for k, (x, y) in enumerate(zip(na, nb)) if x != y), min(len(na), len(nb))); errors.append(f"numbers differ at position {i}: {na[i:i + 4]} -> {nb[i:i + 4]} ({len(na)} vs {len(nb)} numbers)")
    ca, cb = Counter(SPAN.findall(old)), Counter(SPAN.findall(new))
    if ca != cb: errors.append(f"code spans differ: removed {list((ca - cb).items())[:6]} added {list((cb - ca).items())[:6]}")
    ta = [l.count("|") for l in old.splitlines() if l.lstrip().startswith("|")]; tb = [l.count("|") for l in new.splitlines() if l.lstrip().startswith("|")]
    if ta != tb: errors.append(f"table shape differs ({len(ta)} vs {len(tb)} table rows)")
    return errors


def check(path, ref="HEAD"):
    old = git_show(ref, path)
    if old is None: return {"file": path, "status": "new", "errors": [], "changed_strings": []}
    new = open(path, encoding="utf-8").read()
    if old == new: return {"file": path, "status": "unchanged", "errors": [], "changed_strings": []}
    errors, changed = check_python(old, new) if path.endswith(".py") else (check_text(old, new), [])
    return {"file": path, "status": "ok" if not errors else "FAILED", "errors": errors, "changed_strings": changed}


def main(argv):
    ref = "HEAD"; as_json = False; files = []
    it = iter(argv)
    for a in it:
        if a == "--ref": ref = next(it)
        elif a == "--json": as_json = True
        else: files.append(a)
    res = [check(f, ref) for f in files]; bad = [r for r in res if r["errors"]]
    if as_json: print(json.dumps(res, indent=1, ensure_ascii=False))
    else:
        for r in res:
            if r["status"] != "unchanged": print(r["status"].ljust(7), r["file"], f"({len(r['changed_strings'])} message strings changed)" if r["changed_strings"] else "")
            for e in r["errors"]: print("   ", e)
        print(f"{len(res)} files checked, {len(bad)} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
