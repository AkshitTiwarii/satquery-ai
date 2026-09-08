"""RSVQA-LR counting: the model may answer with a bucket label, not an integer.

Counting is 2,947 of the 10,004 questions in the test split and the weakest
type we serve (63.0 binned, against 92/96/87 for the other three). The reason
is visible in the per-row records: of 19 questions whose gold count falls in
the 11-100 bucket, the adapter put exactly one in that bucket - it answers a
small integer or a large one and lands a bucket either side. It is being asked
to regress an exact number and then scored on a five-way bucket.

So the adapter is retrained to answer with the bucket label itself
(make_rsvqa_subset.py --count-answer bucket) and the harness has to read one.
Two properties matter and both are tested here:

  1. a label parses to its own bucket - including ">1000", which the integer
     parser silently mis-bins to "101-1000" because it reads the 1000 and
     ignores the ">". That is the control: this test fails on the parser as
     it stood before bucket support.
  2. an integer-answering adapter scores EXACTLY as it did before, so the new
     number stays comparable with the recorded 84.5. The regression fixture is
     the 200 count rows of two earlier adapters with their recorded
     correct_binned column; re-reading their raw text must reproduce it.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_rsvqa_lr import COUNT_LABELS, bin_count, normalise  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "rsvqa_count_rows.jsonl")


# --- 1. bucket labels read back as themselves -------------------------------

@pytest.mark.parametrize("label", ["0", "1-10", "11-100", "101-1000", ">1000"])
def test_label_round_trips(label):
    """Every label the subset writes must survive parse -> bin unchanged."""
    assert normalise(label, "count") == label
    assert bin_count(normalise(label, "count")) == label


def test_gt_1000_is_the_control():
    """The bucket the integer parser gets wrong.

    ">1000" holds the integer 1000, which bins to "101-1000". Before bucket
    support this assertion failed, and every >1000 answer was scored against
    the wrong bucket.
    """
    assert bin_count(normalise(">1000", "count")) == ">1000"


@pytest.mark.parametrize("raw,expected", [
    (">1000", ">1000"),
    ("> 1000", ">1000"),
    ("greater than 1000", ">1000"),
    ("more than 1000", ">1000"),
    ("101-1000", "101-1000"),
    ("101 - 1000", "101-1000"),
    ("101 to 1000", "101-1000"),
    ("between 101 and 1000", "101-1000"),
    ("11-100", "11-100"),
    ("11 to 100", "11-100"),
    ("1-10", "1-10"),
    ("between 1 and 10", "1-10"),
    ("The answer is 11-100.", "11-100"),
])
def test_verbose_label_forms(raw, expected):
    assert normalise(raw, "count") == expected


def test_labels_constant_matches_the_bins():
    """COUNT_LABELS is what the subset cutter writes; it must not drift."""
    from eval_rsvqa_lr import COUNT_BINS
    assert COUNT_LABELS == tuple(label for _, _, label in COUNT_BINS)


# --- 2. integers behave exactly as before -----------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("197", "197"),
    ("0", "0"),
    ("1023", "1023"),
    ("There are 3 buildings", "3"),
    ("none", "0"),
    ("seven", "7"),
    ("  12  ", "12"),
])
def test_integers_unchanged(raw, expected):
    assert normalise(raw, "count") == expected


@pytest.mark.parametrize("value,expected", [
    ("0", "0"), ("1", "1-10"), ("10", "1-10"), ("11", "11-100"),
    ("100", "11-100"), ("101", "101-1000"), ("1000", "101-1000"),
    ("1001", ">1000"), ("1023", ">1000"),
])
def test_bin_count_integers_unchanged(value, expected):
    assert bin_count(value) == expected


def test_bin_count_rejects_nonsense():
    assert bin_count("banana") is None
    assert bin_count(None) is None


def test_recorded_scores_reproduce():
    """The comparability check.

    Two integer-answering adapters, 200 count rows, their correct_binned
    column as the harness recorded it on the office box. Re-reading the same
    raw text through the current parser must give the same verdict on every
    row - otherwise a post-change score cannot be set beside the 84.5 that
    those runs produced.
    """
    rows = [json.loads(line) for line in open(FIXTURE, encoding="utf-8")]
    assert len(rows) == 200, "fixture lost rows"
    for r in rows:
        pred = normalise(r["raw"], "count")
        assert pred == r["pred"], f"{r['adapter']} {r['qid']}: {r['raw']!r}"
        got = bin_count(pred) == bin_count(r["gold"])
        assert got == r["correct_binned"], f"{r['adapter']} {r['qid']}: {r['raw']!r}"


def test_fixture_contains_the_hard_bucket():
    """A fixture that cannot fail proves nothing.

    The 11-100 bucket is the one the integer adapter misses; if the fixture
    held only rows it got right, the regression check above would pass on a
    broken parser too.
    """
    rows = [json.loads(line) for line in open(FIXTURE, encoding="utf-8")]
    hard = [r for r in rows if bin_count(r["gold"]) == "11-100"]
    assert len(hard) >= 20, "fixture has too few 11-100 rows to be a regression"
    assert any(not r["correct_binned"] for r in hard), "fixture holds no failures"
