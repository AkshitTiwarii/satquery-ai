"""The --ref-rows lever must change the data it names.

Two grounding runs asked for 16,000 and 40,000 reference rows and both trained on 10,986,
because the Lithuania/Summer train cell holds exactly that many and the generator only
raised a per-patch cap that never bound. Above the cell's count the cut must leave the
season, and the patch budget must be able to supply the rows asked for.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_train_notebook import (  # noqa: E402
    CELL_REF_ROWS, REF_ROWS_PER_PATCH, build, ground_ref_cell_flags, ground_ref_max_patches,
)


def _cut_cell(ref_rows):
    nb = build(data="ground-ref", vision_lora=True, ref_rows=ref_rows)
    src = "".join(nb["cells"][4]["source"])
    assert "make_subset.py" in src
    return src


def test_the_recipe_cut_is_unchanged():
    src = _cut_cell(CELL_REF_ROWS)
    assert "--country Lithuania --season Summer" in src
    assert "--max-patches 4000 --per-patch 14" in src
    assert "ref[:10986]" in src


def test_above_the_cell_the_cut_leaves_the_season():
    src = _cut_cell(40000)
    assert "--country Lithuania" in src
    assert "--season" not in src, "40k reference rows cannot come from one season"
    assert "ref[:40000]" in src


def test_the_patch_budget_can_supply_the_rows():
    for ref_rows in (CELL_REF_ROWS + 1, 20000, 40000, 60000):
        assert ground_ref_max_patches(ref_rows) * REF_ROWS_PER_PATCH >= ref_rows
    assert ground_ref_max_patches(40000) == 13334


def test_control_the_old_per_patch_trick_would_fail_here():
    # What the generator did before 8 Sep: 4,000 patches at 3.54 reference rows each is
    # 14,160 at most, whatever the per-patch cap says. The check above must reject it.
    old_budget = 4000
    assert old_budget * 3.54 < 40000
    assert ground_ref_cell_flags(40000) != "--country Lithuania --season Summer"
