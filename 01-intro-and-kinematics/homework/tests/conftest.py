"""
Session-wide checks: forbid student code (solutions/) from importing reference_solution or hidden_tests.

Set BLOCK_REFERENCE_IMPORT=1 to block runtime imports of those top-level packages (optional).
"""
import ast
import os
import sys
from pathlib import Path

import pytest

FORBIDDEN_MODULES = ("reference_solution", "hidden_tests")


def _install_import_blocker():
    """Block imports of reference_solution and hidden_tests when BLOCK_REFERENCE_IMPORT is set."""
    if os.environ.get("BLOCK_REFERENCE_IMPORT") != "1":
        return

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            top = fullname.split(".")[0]
            if top in FORBIDDEN_MODULES:
                raise ImportError(
                    f"Import of '{fullname}' is disabled in this run. "
                    "Do not import reference_solution or hidden_tests in your solution."
                )
            return None

    sys.meta_path.insert(0, _Blocker())


_install_import_blocker()


def _collect_imported_modules(tree: ast.AST) -> set[str]:
    """Return top-level module names that appear in import statements."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                modules.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split(".")[0]
                modules.add(name)
    return modules


@pytest.fixture(scope="session", autouse=True)
def _forbid_reference_imports():
    """Fail the test session if any solution file imports reference_solution or hidden_tests."""
    solutions_dir = Path(__file__).resolve().parent.parent / "solutions"
    if not solutions_dir.is_dir():
        return
    for py_file in solutions_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        imported = _collect_imported_modules(tree)
        for forbidden in FORBIDDEN_MODULES:
            if forbidden in imported:
                pytest.fail(
                    f"{py_file.name} must not import '{forbidden}'. "
                    "Implement the solution yourself; do not use the reference or hidden test modules."
                )
