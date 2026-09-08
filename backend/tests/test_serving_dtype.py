"""Which precision answered is part of the result, so it goes in the trace.

nf4 is not free. On RSVQA-LR the joint adapter reads 84.5 in 4-bit and 87.5 in
half precision on the same 400 questions - three points paid to quantisation
rather than to the model. int8 is the rung between them and is the only one
that fits the 8 GB demo card alongside four adapters, so it is now selectable
by SATQUERY_DTYPE and by --dtype on all four harnesses.

Two things have to hold, and the second is the one that bites:

  1. an unknown value is refused at import, not defaulted;
  2. the trace records the precision, so replaying an nf4 trace under int8
     FAILS instead of quietly reporting a byte-identical match between two
     different sets of weights. A replay that cannot tell those apart is not
     checking anything.
"""

import importlib
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from qwen_loader import DTYPES  # noqa: E402

from satquery import models, tools  # noqa: E402


def test_the_three_precisions_are_the_ones_the_loader_offers():
    assert DTYPES == ("nf4", "int8", "fp16")


def test_default_is_nf4_so_every_recorded_number_reproduces():
    assert models.DTYPE == "nf4"


def test_an_unknown_precision_is_refused_at_import():
    """THE CONTROL. A typo in demo_env.ps1 must stop the run, not silently
    serve 4-bit under a name that says otherwise."""
    env = dict(os.environ, SATQUERY_DTYPE="fp8", SATQUERY_BACKEND="stub")
    proc = subprocess.run(
        [sys.executable, "-c", "import satquery.models"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "SATQUERY_DTYPE" in proc.stderr


@pytest.mark.parametrize("value", ["nf4", "int8", "fp16"])
def test_each_precision_is_accepted(value):
    env = dict(os.environ, SATQUERY_DTYPE=value, SATQUERY_BACKEND="stub")
    proc = subprocess.run(
        [sys.executable, "-c",
         "import satquery.models as m; print(m.DTYPE)"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == value


def test_dtype_name_is_empty_without_a_loaded_model():
    """Same contract as adapter_name: nothing loaded, nothing claimed. Under
    the stub backend no weights exist, so the trace must not assert a
    precision that never ran."""
    assert models.dtype_name() == ""


def test_dtype_name_reports_the_precision_once_weights_are_loaded(monkeypatch):
    monkeypatch.setitem(models._STATE, "model", object())
    monkeypatch.setattr(models, "DTYPE", "int8")
    assert models.dtype_name() == "int8"


def test_every_real_qwen_row_records_the_precision():
    """The four tool functions that run the model all put `dtype` beside
    `adapter`. Read from the source, because reaching the branch needs a GPU."""
    src = open(os.path.join(ROOT, "satquery", "tools.py"), encoding="utf-8").read()
    assert src.count('"adapter": models.adapter_name(') == 3   # vqa, change, fusion
    assert src.count('"adapter": adapter,') == 1               # grounding
    # 4 task rows + zoom.crop, whose two real branches (region parsed / not)
    # both ran the model and both say so.
    assert src.count('"dtype": models.dtype_name()') == 6


def test_loader_rejects_an_unknown_dtype():
    from qwen_loader import load_model
    with pytest.raises(ValueError, match="dtype must be one of"):
        load_model("whatever", "int4")


def test_every_harness_offers_the_three_precisions():
    """All four eval scripts, one flag, one spelling. A harness stuck on two
    choices would silently score the demo's precision as something else."""
    for name in ("eval_rsvqa_lr", "eval_ben_vqa", "eval_cdvqa", "eval_ground"):
        src = open(os.path.join(ROOT, "scripts", name + ".py"), encoding="utf-8").read()
        assert 'choices=["nf4", "int8", "fp16"]' in src, name
        assert "from qwen_loader import load_model" in src, name
        # the old inline branch must be gone, or two definitions drift apart again
        assert "bnb_4bit_quant_type" not in src, name
