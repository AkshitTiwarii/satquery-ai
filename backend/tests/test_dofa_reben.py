"""The DOFA run's scoring must be right before a GPU-hour is spent on it.

Plain numpy. AP is checked against hand-computed values, the macro filter against its
own definition, and the control that matters: a score that carries no information about
the label must land at the class prevalence, not above it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from dofa_reben import BANDS, CLASSES, WAVES, average_precision, micro_ap, summarise  # noqa: E402


def test_constants_line_up():
    assert len(CLASSES) == 19 and len(set(CLASSES)) == 19
    assert len(BANDS) == len(WAVES) == 12
    assert WAVES == sorted(WAVES), "wavelengths must be in band order, and S2 bands ascend"


def test_ap_hand_computed():
    # ranking by score: pos, neg, pos, neg -> precision at the two hits = 1/1, 2/3 -> AP = (1 + 2/3)/2
    y = np.array([[1], [0], [1], [0]], np.float32)
    s = np.array([[0.9], [0.8], [0.7], [0.1]], np.float32)
    assert abs(average_precision(y, s)[0] - (1 + 2 / 3) / 2) < 1e-9
    # perfect ranking -> 1, inverted ranking -> the worst case for 2 positives in 4
    assert average_precision(y, np.array([[0.9], [0.1], [0.8], [0.2]]))[0] == 1.0
    worst = average_precision(y, np.array([[0.1], [0.9], [0.2], [0.8]]))[0]
    assert abs(worst - (1 / 3 + 2 / 4) / 2) < 1e-9


def test_ap_is_nan_for_absent_class_and_macro_skips_it():
    y = np.zeros((50, 19), np.float32)
    y[:25, 0] = 1        # class 0 has 25 positives
    y[:5, 1] = 1         # class 1 has 5 - below the 20 floor
    s = np.random.RandomState(0).rand(50, 19).astype(np.float32)
    ap = average_precision(y, s)
    assert np.isnan(ap[2:]).all()
    r = summarise("x", y, s)
    assert r["classes_in_macro"] == 1
    assert r["per_class_ap"][CLASSES[2]] is None
    assert abs(r["macro_ap_20pos"] - ap[0]) < 1e-9


def test_control_uninformative_scores_sit_at_prevalence():
    rng = np.random.RandomState(1)
    y = (rng.rand(4000, 1) < 0.3).astype(np.float32)
    s = rng.rand(4000, 1).astype(np.float32)
    ap = average_precision(y, s)[0]
    assert abs(ap - 0.3) < 0.03, ap
    # and the prior row of the run (constant score) is exactly the prevalence
    const = np.full_like(s, 0.3)
    assert abs(average_precision(y, const)[0] - y.mean()) < 1e-6
    assert abs(micro_ap(y, const) - y.mean()) < 1e-6


def test_informative_scores_beat_the_control():
    rng = np.random.RandomState(2)
    y = (rng.rand(2000, 3) < 0.4).astype(np.float32)
    s = y * 0.6 + rng.rand(2000, 3) * 0.5
    r = summarise("good", y, s)
    assert r["macro_ap_20pos"] > 0.8 and r["micro_ap"] > 0.8
