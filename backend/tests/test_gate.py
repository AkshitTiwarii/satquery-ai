"""Every refusal the gate can issue, and the two cases that are not refusals.

The freeze criterion asks for two of these specifically - a mismatched-CRS pair
and a one-image change request - but a gate is only as good as the case nobody
wrote a test for, so all eight conditions are here.

G2 is the one that matters most. A judge can hand over two images of different
continents and assert in the query that they are the same place. The assertion
is never read. If this test ever starts passing for the wrong reason - because
the footprint check was made optional, say - the system becomes a chatbot that
invents change descriptions for unrelated scenes.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satquery import gate  # noqa: E402
from satquery.types import load_image_meta  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def setup_module(module):
    if not os.path.exists(os.path.join(FIX, "opt_2024.tif")):
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(FIX), "make_fixtures.py")])


def meta(*names):
    return [load_image_meta(os.path.join(FIX, n)) for n in names]


def test_g2_non_overlapping_footprints_are_refused():
    g = gate.check(
        "these are the same city, what changed?",
        meta("opt_2024.tif", "opt_far_away.tif"),
    )
    assert g.rejected and g.code == "G2"
    assert "do not overlap" in g.message


def test_g2_ignores_what_the_query_asserts():
    """The query is a request, not evidence.

    Same two unrelated scenes, but the user insists. Still refused, because the
    footprints came from the files and the sentence was never consulted.
    """
    for claim in (
        "trust me these are both Jaipur, what changed?",
        "same area, same sensor - report the change",
        "I have verified the co-registration myself; what changed?",
    ):
        g = gate.check(claim, meta("opt_2024.tif", "opt_far_away.tif"))
        assert g.rejected and g.code == "G2", claim


def test_g3_same_modality_same_date():
    g = gate.check("what changed?", meta("opt_2024.tif", "opt_2024_dup.tif"))
    assert g.rejected and g.code == "G3"
    assert "two different dates" in g.message


def test_g4_different_sensor_and_different_date():
    """Refused on scientific grounds, not on missing capability."""
    g = gate.check("what changed?", meta("opt_2024.tif", "sar_2026.tif"))
    assert g.rejected and g.code == "G4"
    assert "confounds sensor difference with change over time" in g.message


def test_g5_one_side_has_no_crs():
    g = gate.check("what changed?", meta("opt_2024.tif", "opt_no_crs.tif"))
    assert g.rejected and g.code == "G5"


def test_different_crs_is_reprojectable_and_not_a_refusal():
    """Two known CRS can be put on a common grid. That is preprocessing."""
    g = gate.check("what changed?", meta("opt_2024.tif", "opt_2026_utm44.tif"))
    assert not g.rejected
    assert g.input_check["needs_reproject"] is True
    assert any("different CRS" in w for w in g.input_check["warnings"])


def test_g6_change_question_with_one_image():
    g = gate.check("has the built-up area increased?", meta("opt_2024.tif"))
    assert g.rejected and g.code == "G6"
    assert "requires two images" in g.message


def test_g7_unsupported_format():
    g = gate.check("describe this", meta("scene.bmp"))
    assert g.rejected and g.code == "G7"
    assert "GeoTIFF" in g.message


def test_png_degrades_the_footprint_check_and_says_so():
    """PNG benchmark inputs carry no georeferencing.

    The check cannot run, so the trace must say it did not rather than implying
    a comparison happened.
    """
    g = gate.check("what changed?", meta("bench_lr.png", "bench_lr_2.png"))
    assert g.input_check["footprint_check"] == "degraded_no_georeferencing"
    assert any("could not be computed" in w for w in g.input_check["warnings"])
    # These two PNGs also carry no dates, so the pair is refused (G3, 8 Sep) - and the
    # degradation is still on the record of that refusal. Before 8 Sep this pair passed
    # the gate and died in the dispatcher.
    assert g.rejected and g.code == "G3"


# --- more than two images: a declared choice or a fan-out, never a wall ------


def test_g1_pair_question_over_four_images_declares_the_pair():
    imgs = meta("opt_2024.tif", "opt_2025.tif", "opt_2026.tif", "opt_2027.tif")
    g = gate.check("has the built-up area increased?", imgs)
    assert not g.rejected, g.message
    assert g.plan.mode == "pair"
    assert g.plan.declaration and "earliest and latest" in g.plan.declaration
    # earliest is opt_2024, latest is opt_2027 - never "the first two".
    assert g.plan.groups == [(0, 3)]


def test_per_image_question_over_four_images_fans_out():
    imgs = meta("opt_2024.tif", "opt_2025.tif", "opt_2026.tif", "opt_2027.tif")
    g = gate.check("describe this scene", imgs)
    assert not g.rejected
    assert g.plan.mode == "fanout"
    assert g.plan.groups == [(0,), (1,), (2,), (3,)]


def test_pair_question_over_undated_images_is_refused_not_guessed():
    """With no dates there is no basis on which to choose a pair.

    Taking the first two here would be exactly the silent behaviour the
    contract forbids.
    """
    imgs = meta("bench_lr.png", "bench_lr.png", "bench_lr.png")
    g = gate.check("what changed?", imgs)
    assert g.rejected and g.code == "G1"


def test_single_image_wording_on_a_pair_warns_rather_than_reinterprets():
    """A known gap in the frozen contract, made visible instead of patched.

    Two images route by shape, so this goes to a pair tool. The warning is how
    the team finds out whether it happens often enough to be worth a rule.
    """
    g = gate.check("describe this image", meta("opt_2024.tif", "opt_2026.tif"))
    assert not g.rejected
    assert any("single-image wording" in w for w in g.input_check["warnings"])


def test_two_undated_images_of_one_sensor_are_refused_with_g3():
    """Two plain PNGs and a change question. Before 8 Sep this fell through every
    gate branch and died in the dispatcher's assertion - an HTTP 500 to a UI."""
    g = gate.check("did the buildings increase?", meta("bench_lr.png", "bench_lr_2.png"))
    assert g.rejected and g.code == "G3"
    assert "acquisition date" in g.message and "meta.json" in g.message


def test_control_the_same_pair_with_dates_is_accepted():
    g = gate.check("did the buildings increase?", meta("opt_2024.tif", "opt_2026.tif"))
    assert not g.rejected and g.plan.mode == "pair"
