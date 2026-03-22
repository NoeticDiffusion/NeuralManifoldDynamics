"""Insert one-line docstrings for functions missing them."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _make_docstring(name: str) -> str:
    if name.startswith("test_"):
        rest = name[5:].replace("_", " ").strip() or "case"
        return f"Test {rest}."
    if name == "__init__":
        return "Initialize the instance."
    if name == "main":
        return "Program entry point."
    if name == "build_parser":
        return "Build and return the argument parser."
    if name in ("run",):
        return "Run the main workflow for this component."
    if name in ("from_mapping", "from_config", "from_resolved"):
        return f"Construct instance {name.replace('_', ' ')}."
    if name.startswith("__") and name.endswith("__"):
        return f"Dunder method {name}."
    readable = name.strip("_").replace("_", " ") or name
    if name.startswith("_"):
        return f"Internal helper: {readable}."
    return f"Handle {readable}."


def _should_skip(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if ast.get_docstring(node):
        return True
    if not node.body:
        return True
    if node.body[0].lineno <= node.lineno:
        return True
    return False


def _collect_missing(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if not _should_skip(node):
                out.append(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if not _should_skip(node):
                out.append(node)
            self.generic_visit(node)

    V().visit(tree)
    return out


def process_file(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0

    missing = _collect_missing(tree)
    if not missing:
        return 0

    lines = src.splitlines(keepends=True)
    missing.sort(key=lambda n: n.body[0].lineno, reverse=True)
    added = 0
    for node in missing:
        first = node.body[0]
        insert_idx = first.lineno - 1
        line = lines[insert_idx]
        indent = line[: len(line) - len(line.lstrip(" \t"))]
        doc = _make_docstring(node.name)
        lines.insert(insert_idx, f'{indent}"""{doc}"""\n')
        added += 1

    path.write_text("".join(lines), encoding="utf-8")
    return added


def main() -> None:
    roots = [ROOT / r for r in sys.argv[1:]] or [ROOT / "core/src/core", ROOT / "mndm/src/mndm"]
    total = 0
    for base in roots:
        for py in sorted(base.rglob("*.py")):
            n = process_file(py)
            if n:
                print(f"{py.relative_to(ROOT)}: +{n}")
                total += n
    print(f"Total docstrings added: {total}")


if __name__ == "__main__":
    main()
