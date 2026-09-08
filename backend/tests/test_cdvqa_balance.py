"""The change adapter was paid to guess, so the sampler stopped paying it.

CDVQA's class and ratio types have a strong answer prior. The v4 training mix
balanced the eight question TYPES but left the answers at their natural
frequency, and the adapter learned the cheapest thing available: on the 50
smallest_change test questions it answered 'trees' 49 times, scoring 14.0
against a 34.0 majority baseline. change_to_what collapsed onto 'buildings'
and change_ratio onto the bucket below the true one.

balanced(..., balance_answers=True) flattens the answer distribution inside
each type, so playing the prior is worth ~1/k instead of ~1/2 and the only
remaining way to gain is to tell the classes apart. This pins that:

  * the flattening actually happens, on a pool built to collapse;
  * rare answers are not dropped, and the row count does not shrink;
  * files are still strided, so a type is never drawn from a few pictures;
  * balance_answers=False reproduces v4 exactly, which is the control - a
    committed subset file must still hash the same through the new code.
"""

import collections
import hashlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from make_cdvqa_subset import _round_robin_files, balanced  # noqa: E402


def _pool(spec, files=500, per_file=16):
    """spec: {type: {answer: share}} -> rows shaped like load_split's.

    500 files x 16 rephrasings = 8,000 rows per type, against a 1,000-row
    budget per type below. That ratio matters: the real split holds 13,882
    change_or_not questions and the subset takes 2,500, so a minority answer
    has room to fill its share. A sampler can only flatten what the pool
    contains, and test_an_unflattenable_pool... pins the other regime.
    """
    rows = []
    qid = 0
    for qtype, answers in spec.items():
        bag = []
        for ans, share in answers.items():
            bag += [ans] * share
        for f in range(files):
            for k in range(per_file):
                ans = bag[(f * per_file + k) % len(bag)]
                rows.append({"qid": qid, "img_id": qid, "file": f"{f:04d}.png",
                             "type": qtype, "question": "q?", "gold": ans})
                qid += 1
    return rows


# The shares CDVQA's train split actually has, measured off data/cdvqa_train.jsonl:
# smallest_change is 28% 'trees' over six classes, change_or_not is 88% 'yes'.
REAL = {"smallest_change": {"trees": 28, "buildings": 21, "low_vegetation": 19,
                            "water": 16, "NVG_surface": 13, "playgrounds": 3},
        "change_or_not": {"yes": 88, "no": 12}}

# A pool so lopsided that no reordering can flatten it: 90% of the rows carry
# one answer, so a 1,000-row budget cannot avoid them. Used to pin what the
# sampler does at the limit, which is take everything else and top up.
EXTREME = {"smallest_change": {"trees": 90, "buildings": 4, "low_vegetation": 3,
                               "NVG_surface": 2, "water": 1}}


def _shares(rows, qtype):
    c = collections.Counter(r["gold"] for r in rows if r["type"] == qtype)
    n = sum(c.values())
    return {a: k / n for a, k in c.items()}


def test_the_control_first_unbalanced_stays_skewed():
    """If this did NOT come out skewed, the pool could not detect a fix."""
    rows = balanced(_pool(REAL), 2000, seed=1, balance_answers=False)
    assert _shares(rows, "smallest_change")["trees"] > 0.25
    assert _shares(rows, "change_or_not")["yes"] > 0.80


def test_balancing_flattens_the_prior():
    """On the real distribution every answer ends near its uniform share."""
    rows = balanced(_pool(REAL), 2000, seed=1, balance_answers=True)
    small = _shares(rows, "smallest_change")
    assert max(small.values()) <= 0.25, small
    yesno = _shares(rows, "change_or_not")
    assert abs(yesno["yes"] - 0.5) < 0.02, yesno


def test_balancing_strictly_reduces_the_dominant_share():
    for spec, qtype in ((REAL, "smallest_change"), (EXTREME, "smallest_change")):
        pool = _pool(spec)
        before = _shares(balanced(pool, 1000, 1, balance_answers=False), qtype)
        after = _shares(balanced(pool, 1000, 1, balance_answers=True), qtype)
        top = max(before, key=before.get)
        assert after[top] < before[top], (qtype, before[top], after[top])


def test_an_unflattenable_pool_is_documented_not_silently_broken():
    """When 90% of the rows carry one answer, a full-size subset cannot be
    balanced: every other row is already taken. The sampler keeps the budget
    and the mix stays skewed - that is the honest outcome, and the notice the
    cutter prints is what tells a reader the prior is still there."""
    # 100 files = 1,600 rows against a 1,000-row budget, so 90% 'trees' leaves
    # only 160 other rows in existence and the budget cannot avoid the rest.
    rows = balanced(_pool(EXTREME, files=100), 1000, seed=1, balance_answers=True)
    small = _shares(rows, "smallest_change")
    assert small["trees"] > 0.5, "the pool really is unflattenable at this size"
    # Every minority answer is exhausted rather than under-sampled: the pool
    # holds 16 'water' and 32 'NVG_surface' rows and the subset takes them all.
    counts = collections.Counter(r["gold"] for r in rows)
    assert counts["water"] == 16 and counts["NVG_surface"] == 32, counts


def test_rare_answers_survive():
    """'playgrounds' is 3% of the pool. Flattening must not drop it, and must
    not invent rows for it either - it is taken whole."""
    rows = balanced(_pool(REAL), 2000, seed=1, balance_answers=True)
    small = collections.Counter(
        r["gold"] for r in rows if r["type"] == "smallest_change")
    assert set(small) == set(REAL["smallest_change"]), small


def test_row_count_is_not_lost_to_balancing():
    """A rare answer that cannot fill its share hands the slack on, so the
    subset is the size it was asked for."""
    plain = balanced(_pool(REAL), 2000, seed=1, balance_answers=False)
    bal = balanced(_pool(REAL), 2000, seed=1, balance_answers=True)
    assert len(bal) == len(plain) == 2000


def test_files_are_still_strided():
    """The split holds 16 rephrasings per picture. Drawing a type from a
    handful of files would make the mix look balanced and the imagery not be."""
    rows = balanced(_pool(REAL), 2000, seed=1, balance_answers=True)
    files = {r["file"] for r in rows if r["type"] == "smallest_change"}
    assert len(files) >= 90, len(files)


def test_round_robin_visits_every_file_before_repeating():
    pool = [{"file": f"{f}.png", "gold": "x", "qid": f * 10 + k}
            for f in range(10) for k in range(5)]
    got = _round_robin_files(pool, 10)
    assert len({r["file"] for r in got}) == 10


def test_round_robin_cannot_exceed_the_pool():
    pool = [{"file": "a.png", "gold": "x", "qid": 1}]
    assert len(_round_robin_files(pool, 50)) == 1


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "cdvqa_train.jsonl")),
                    reason="the committed v4 subset is not on this machine")
def test_v4_subset_still_reproduces_byte_for_byte():
    """The comparability control.

    data/cdvqa_train.jsonl is the file the 58.5 adapter trained on. Cutting it
    again with balance_answers=False must produce the identical bytes, or the
    new sampler changed the old path and no post-change number could be set
    beside 58.5.
    """
    import io
    import json
    import subprocess
    import tempfile

    committed = os.path.join(ROOT, "data", "cdvqa_train.jsonl")
    if not os.path.exists(os.path.join(ROOT, "data", "cdvqa", "Train_questions.json")):
        pytest.skip("CDVQA split files are not on this machine")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "again.jsonl")
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "make_cdvqa_subset.py"),
             "--data-dir", os.path.join(ROOT, "data", "cdvqa"),
             "--limit", "20000", "--no-balance-answers", "--out", out],
            check=True, capture_output=True)
        a = hashlib.sha256(io.open(committed, "rb").read()).hexdigest()
        b = hashlib.sha256(io.open(out, "rb").read()).hexdigest()
        assert a == b, "the v4 recipe no longer reproduces"
        assert json.loads(io.open(out, encoding="utf-8").readline())
