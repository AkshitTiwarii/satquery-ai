"""VRSBench is a prescribed grading split we hold no number on. This pins the
two things that would make a number from it wrong rather than merely bad.

THE BOX SCALE. VRSBench gold is "{<25><40><33><60>}" where the integers are
PERCENTAGES. BigEarthNet.txt uses 0-1 decimals for the same thing and some Qwen
builds emit 0-1000. Read 25 as 0.25 of a unit square when it means 25%, or as
1/1000, and every gold box collapses into a dot in the corner: the score comes
out near zero and looks like a model that cannot localise. So the parser
refuses an out-of-range value instead of rescaling on a guess, and the harness
parses every gold box before it loads the model rather than three hours in.

THE ANSWER NORMALISER. 37,409 open-vocabulary answers, 710 distinct golds on
one type alone. Normalise too little and "Yes." misses "yes"; too much and
"north-south" matches "south-north", which is a different answer to a
direction question. The cases below fix where that line sits.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_vrsbench import (BRACE_BOX, REF_PROMPTS, VQA_PROMPT,  # noqa: E402
                           normalise, parse_gold_box, parse_pred_box,
                           stratified)


# --- the box scale ---------------------------------------------------------

def test_gold_box_is_read_as_percentages():
    assert parse_gold_box("{<25><40><33><60>}") == pytest.approx((0.25, 0.40, 0.33, 0.60))


def test_gold_box_handles_a_full_frame():
    assert parse_gold_box("{<0><0><100><100>}") == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_gold_box_orders_the_corners():
    """A box given bottom-right first must still have x0 <= x1."""
    assert parse_gold_box("{<60><80><20><30>}") == pytest.approx((0.2, 0.3, 0.6, 0.8))


def test_an_out_of_range_scale_raises_rather_than_rescaling():
    """THE CONTROL. 0-1000 is the scale some Qwen builds use. Silently
    dividing it by 100 would put every box outside the image; silently
    dividing by 1000 would be a guess that happens to work here and fail on
    the next dataset. Refuse."""
    with pytest.raises(ValueError, match="0-100 percentages"):
        parse_gold_box("{<250><400><330><600>}")


def test_a_box_with_no_braces_is_not_a_box():
    assert parse_gold_box("the vehicle is on the left") is None
    assert parse_gold_box("") is None
    assert parse_gold_box(None) is None


def test_the_regex_does_not_match_three_numbers():
    assert BRACE_BOX.search("{<25><40><33>}") is None


def test_predicted_brace_box_is_read_and_labelled():
    got = parse_pred_box("{<10><20><30><40>}", "vrsbench", (448, 448))
    assert got is not None
    box, how = got
    assert how == "brace"
    assert box == pytest.approx((0.10, 0.20, 0.30, 0.40))


def test_a_ben_format_answer_still_counts_under_the_vrsbench_format():
    """Our grounding adapter answers in BigEarthNet's string. Asking it in the
    brace format and scoring its real answer as unparsed would measure the
    prompt, not the model - so the fallback exists and the record says which
    format the answer arrived in."""
    got = parse_pred_box("[0.1 0.2, 0.3 0.4]", "vrsbench", (448, 448))
    assert got is not None
    box, how = got
    assert how == "ben"
    assert box == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_unreadable_output_is_none_not_a_silent_zero_box():
    assert parse_pred_box("I cannot tell", "ben", (448, 448)) is None


# --- the answer normaliser -------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Yes.", "yes"),
    ("yes", "yes"),
    ("  YES  ", "yes"),
    ("The answer is a ship", "answer is ship"),
    ("2", "2"),
    ("two", "2"),
    ("Grayscale", "grayscale"),
    ("North-South", "north-south"),
])
def test_normalise_cases(raw, expected):
    assert normalise(raw) == expected


def test_direction_answers_stay_distinct():
    """Collapsing the hyphen would make these equal, and they are opposite
    answers to a direction question."""
    assert normalise("North-South") != normalise("South-North")


def test_empty_normalises_to_none():
    assert normalise("") is None
    assert normalise("   ") is None
    assert normalise(None) is None


# --- sampling --------------------------------------------------------------

def _rows(n_per_type=50, types=("a", "b", "c")):
    out = []
    for t in types:
        for i in range(n_per_type):
            out.append({"image_id": f"img{i // 4:04d}.png", "question": "q?",
                        "ground_truth": "Yes", "type": t, "unique": i % 2 == 0})
    return out


def test_stratified_spreads_over_types_and_images():
    got = stratified(_rows(), 30, key="type")
    assert len(got) == 30
    per = {}
    for r in got:
        per[r["type"]] = per.get(r["type"], 0) + 1
    assert set(per.values()) == {10}
    assert len({r["image_id"] for r in got}) >= 8


def test_stratified_returns_everything_when_limit_is_zero():
    rows = _rows()
    assert stratified(rows, 0) is rows
    assert len(stratified(rows, 10_000)) == len(rows)


def test_stratified_can_group_by_the_unique_flag():
    got = stratified(_rows(), 20, key="unique")
    assert len(got) == 20
    assert len({r["unique"] for r in got}) == 2


# --- prompts ---------------------------------------------------------------

def test_the_ben_prompt_carries_the_suffix_the_adapter_learned():
    from eval_ground import PROMPT_SUFFIX
    assert REF_PROMPTS["ben"].endswith(PROMPT_SUFFIX)
    assert "<ref>" in REF_PROMPTS["ben"]


def test_the_vrsbench_prompt_survives_formatting():
    """The brace format contains { and }, which str.format eats unless they are
    doubled. If this broke, every question would be asked with the braces
    missing and the model would have no idea what shape to answer in."""
    got = REF_PROMPTS["vrsbench"].format(q="the red car")
    assert "{<x1><y1><x2><y2>}" in got
    assert "<p>the red car</p>" in got


def test_the_vqa_prompt_pins_answer_length():
    got = VQA_PROMPT.format(q="How many ships are there?")
    assert got.startswith("How many ships are there?")
    assert "single word or phrase" in got


# --- the real annotation files, when they are on this machine --------------

_ANN = os.path.join(ROOT, "data", "vrsbench")


@pytest.mark.skipif(not os.path.exists(os.path.join(_ANN, "VRSBench_EVAL_referring.json")),
                    reason="VRSBench annotations are not on this machine")
def test_every_real_gold_box_parses():
    rows = json.load(open(os.path.join(_ANN, "VRSBench_EVAL_referring.json"),
                          encoding="utf-8"))
    for r in rows:
        assert parse_gold_box(r["ground_truth"]) is not None, r["ground_truth"]
