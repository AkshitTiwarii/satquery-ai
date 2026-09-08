"""Gate 1: thirty hand-written queries must route to the correct tool.

This is the freeze criterion from PLAN.md. It is deliberately hand-written
rather than generated - generated queries test the regex against itself, and
the thing worth knowing is whether a sentence a judge would actually type lands
on the right row.

Read the shape rules before adding a case: with two images the dispatcher
routes on INPUT SHAPE and never consults the query, so the wording of a
two-image case does not change where it goes. That is per the contract, and it
is why the two-image cases below vary the wording on purpose - all four
phrasings of a change question must land on the same row.
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satquery import dispatcher, gate  # noqa: E402
from satquery.types import load_image_meta  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def setup_module(module):
    if not os.path.exists(os.path.join(FIX, "opt_2024.tif")):
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(FIX), "make_fixtures.py")])


def meta(*names):
    return [load_image_meta(os.path.join(FIX, n)) for n in names]


ONE = ("opt_2024.tif",)
OPT_PAIR = ("opt_2024.tif", "opt_2026.tif")
SAR_PAIR = ("sar_2024.tif", "sar_2026.tif")
CROSS = ("opt_2024.tif", "sar_2024_same_day.tif")

# (query, fixtures, expected rule, expected tool)
CASES = [
    # --- R4: grounding. Eight, because grounding is the second mandatory
    # single-image capability and the wording varies most.
    ("highlight the airport",                      ONE, "R4", "ground.rs"),
    ("where is the reservoir?",                    ONE, "R4", "ground.rs"),
    ("locate all ships in the harbour",            ONE, "R4", "ground.rs"),
    ("show me the runway",                         ONE, "R4", "ground.rs"),
    ("give me a bounding box for the stadium",     ONE, "R4", "ground.rs"),
    ("point out the bridge over the river",        ONE, "R4", "ground.rs"),
    ("mark the flooded fields",                    ONE, "R4", "ground.rs"),
    ("find the solar farm",                        ONE, "R4", "ground.rs"),

    # --- R5: captioning
    ("describe this image",                        ONE, "R5", "caption.rs"),
    ("caption this scene",                         ONE, "R5", "caption.rs"),
    ("what does this show?",                       ONE, "R5", "caption.rs"),
    ("summarise the land cover here",              ONE, "R5", "caption.rs"),
    ("give me an overview of the area",            ONE, "R5", "caption.rs"),

    # --- R6: the catch-all. Every RSVQA question type is in here, because
    # these are the sentences the benchmark actually contains.
    ("is there a water body?",                     ONE, "R6", "vqa.rs"),
    ("how many aircraft are visible?",             ONE, "R6", "vqa.rs"),
    ("is this a rural or an urban scene?",         ONE, "R6", "vqa.rs"),
    ("are there more buildings than trees?",       ONE, "R6", "vqa.rs"),
    ("what is the dominant crop type?",            ONE, "R6", "vqa.rs"),
    ("does the scene contain a port?",             ONE, "R6", "vqa.rs"),

    # --- R3: two optical, two dates. Four phrasings, one destination.
    ("has the built-up area increased?",           OPT_PAIR, "R3", "change.vqa"),
    ("what changed between these dates?",          OPT_PAIR, "R3", "change.vqa"),
    ("was there deforestation?",                   OPT_PAIR, "R3", "change.vqa"),
    ("did the lake shrink?",                       OPT_PAIR, "R3", "change.vqa"),

    # --- R2: two radar, two dates. Log-ratio, never subtraction.
    ("has the built-up area increased?",           SAR_PAIR, "R2", "change.sar_logratio"),
    ("what changed?",                              SAR_PAIR, "R2", "change.sar_logratio"),
    ("show new construction since the earlier date", SAR_PAIR, "R2", "change.sar_logratio"),

    # --- R1: one optical and one radar, same date.
    ("combine both sensors to map built-up area",  CROSS, "R1", "fusion.optical_sar"),
    ("what can you extract using both?",           CROSS, "R1", "fusion.optical_sar"),
    ("map surface water using optical and radar",  CROSS, "R1", "fusion.optical_sar"),
    ("identify built-up area from this pair",      CROSS, "R1", "fusion.optical_sar"),
]


def test_there_are_thirty_of_them():
    """The freeze criterion says thirty. Fewer is not the same test."""
    assert len(CASES) == 30


@pytest.mark.parametrize("query,names,rule,tool", CASES)
def test_routes_to_the_right_tool(query, names, rule, tool):
    images = meta(*names)
    g = gate.check(query, images)
    assert not g.rejected, "gate refused a query that should route: %s" % g.message
    group = g.plan.groups[0]
    r = dispatcher.route(query, [images[i] for i in group])
    assert r["rule_id"] == rule, "%r took %s, expected %s" % (query, r["rule_id"], rule)
    assert r["tool_id"] == tool


def test_every_rule_is_exercised():
    """A routing suite that never reaches a rule has not tested it."""
    assert {c[2] for c in CASES} == {"R1", "R2", "R3", "R4", "R5", "R6"}


def test_grounding_beats_captioning():
    """'show me the ships' reads like a description and is not one.

    R4 is checked before R5 for exactly this. If the order is ever swapped,
    every 'show me the X' becomes a caption and grounding quietly stops being
    reachable from natural wording.
    """
    images = meta("opt_2024.tif")
    assert dispatcher.route("show me the ships", images)["rule_id"] == "R4"


def test_two_images_route_by_shape_not_wording():
    """The same sentence goes to different rows depending on the input.

    This is the contract's design and worth pinning: 'has it increased' is
    change on an optical pair and log-ratio change on a radar pair, because the
    sensor decides the method, not the phrasing.
    """
    q = "has the built-up area increased?"
    assert dispatcher.route(q, meta(*OPT_PAIR))["tool_id"] == "change.vqa"
    assert dispatcher.route(q, meta(*SAR_PAIR))["tool_id"] == "change.sar_logratio"


def test_dispatcher_refuses_a_shape_the_gate_should_have_resolved():
    """Three images must never reach the dispatcher.

    The gate resolves N>2 into groups of one or two. If that ever regresses,
    this fails loudly here rather than producing a confident answer about an
    arbitrary pair.
    """
    with pytest.raises(AssertionError):
        dispatcher.route("what changed?", meta("opt_2024.tif", "opt_2025.tif", "opt_2026.tif"))


def test_no_routing_case_is_mistaken_for_a_compound_query():
    """The guard on `is_compound`.

    The planner path refuses rather than guessing, so a false positive here
    turns a perfectly good single-task question into a wall. Every one of the
    thirty must stay on the rule table.
    """
    for query, _, _, _ in CASES:
        assert not dispatcher.is_compound(query), query
