"""Every module an embedded harness imports must itself be embedded.

The failure this guards cost a full training run on 6 Sep: the harnesses were
rewired to import the single `qwen_loader`, EMBED in the generator was not
updated, and kernel E trained for 8.6 hours then died in its eval cell with
`No module named 'qwen_loader'`. Nothing in the notebook's build or smoke test
touches the eval cell, so the only place the miss could show was after the
GPU-hours were spent.

Static check, no torch: parse each embedded script and resolve its
`import X` / `from X import` against the files in scripts/. Anything that
resolves to a sibling file must be in EMBED too.
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from make_train_notebook import EMBED  # noqa: E402


def _sibling_imports(path):
    """Names imported by `path` that are files in scripts/ (top-level or nested)."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return {n for n in names if os.path.isfile(os.path.join(SCRIPTS, n + ".py"))}


def _missing(embed):
    embedded = {os.path.basename(rel)[:-3] for rel in embed}
    gaps = {}
    for rel in embed:
        need = _sibling_imports(os.path.join(ROOT, rel)) - embedded
        if need:
            gaps[rel] = sorted(need)
    return gaps


def test_every_sibling_import_is_embedded():
    assert _missing(EMBED) == {}, (
        "an embedded script imports a scripts/ module the notebook does not ship; "
        "the eval cell will die with ModuleNotFoundError after training")


def test_the_loader_is_shipped():
    assert "scripts/qwen_loader.py" in EMBED


def test_control_dropping_the_loader_is_caught():
    without = [r for r in EMBED if not r.endswith("qwen_loader.py")]
    gaps = _missing(without)
    assert gaps, "the check let the 6 Sep failure through"
    assert all("qwen_loader" in v for v in gaps.values())
    assert "scripts/eval_cdvqa.py" in gaps
