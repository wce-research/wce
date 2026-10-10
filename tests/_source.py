"""Source inspection helpers for guard tests.

`code_only(obj)` returns the source of a module, class or function with comments and docstrings removed, so that a
guard such as "this module never touches the world truth" checks the CODE and does not depend on the wording or the
language of the documentation."""
import ast
import inspect
import textwrap


def code_only(obj) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)
