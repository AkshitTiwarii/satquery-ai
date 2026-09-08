"""The trace is the graded artefact, so these are the tests that matter most.

Three properties, each of which the statement asks for by name:

  * every emitted trace validates, INCLUDING refusals - "an auditable
    execution summary" is not conditional on the answer being produced;
  * the GUI reads the trace and nothing else, so anything the demo shows must
    be in here;
  * `--replay` reproduces byte-identical output.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satquery import registry, run, trace as tracelib  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def setup_module(module):
    if not os.path.exists(os.path.join(FIX, "opt_2024.tif")):
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(FIX), "make_fixtures.py")])


def fx(*names):
    return [os.path.join(FIX, n) for n in names]


def test_registry_validates_and_has_fourteen_rows():
    rows = registry.load()
    assert len(rows) == 14
    assert len(registry.served(rows)) == 13, "baseline.geochat must not be served"


def test_shared_weights_are_not_double_counted():
    """Residency is keyed on weights, not on row id.

    vqa.rs, ground.rs and caption.rs are one resident model. A loader that adds
    their vram_mb together books 7.5 GB for 2.5 GB of weights and evicts things
    it never needed to.
    """
    rows = registry.load()
    keys = {r.weights_key for r in rows}
    assert registry.by_id(rows, "ground.rs").weights_key == "vqa.rs"
    assert registry.by_id(rows, "caption.rs").weights_key == "vqa.rs"
    # change.vqa joined the shared Qwen weights on 4 Sep 2026 (a LoRA adapter
    # trained on CDVQA, not the Siamese head it was first specified as).
    assert registry.by_id(rows, "change.vqa").weights_key == "vqa.rs"
    total = sum(r["vram_mb"] for r in rows if r.id == r.weights_key and r.served)
    assert total < 4000, "resident budget %d MB does not fit 8.6 GB with headroom" % total
    assert len(keys) == 10


def test_a_normal_query_emits_a_valid_trace():
    t = run.answer("is there a water body?", fx("opt_2024.tif"))
    tracelib.validate(t)
    assert t["classified_task"] == "vqa"
    assert t["routing"]["rule_id"] == "R6"
    assert t["abstained"] is False


def test_the_encoder_runs_on_every_query():
    """encode.rs_gsd is not optional. It is where 12-band and VV/VH enter."""
    for q, names in [
        ("is there a water body?", ("opt_2024.tif",)),
        ("what changed?", ("sar_2024.tif", "sar_2026.tif")),
        ("combine both sensors", ("opt_2024.tif", "sar_2024_same_day.tif")),
    ]:
        t = run.answer(q, fx(*names))
        assert any(s["tool"] == "encode.rs_gsd" for s in t["steps"]), q


def test_sar_is_preprocessed_as_sar_and_it_shows_in_the_trace():
    """Declared as tools, not hidden in a loader, so a reviewer can see it."""
    t = run.answer("what changed?", fx("sar_2024.tif", "sar_2026.tif"))
    used = [s["tool"] for s in t["steps"]]
    assert "preproc.speckle_lee" in used
    assert "preproc.db_normalise" in used
    assert used.index("preproc.speckle_lee") < used.index("encode.rs_gsd")


def test_a_refusal_is_a_complete_trace_not_an_error():
    t = run.answer("has it increased?", fx("opt_2024.tif"))
    tracelib.validate(t)
    assert t["abstained"] is True
    assert t["routing"]["rule_id"] == "R0"
    assert t["classified_task"] is None
    assert t["input_check"]["code"] == "G6"
    assert t["steps"] == [], "nothing may run after a refusal"


def test_every_gate_condition_still_validates():
    cases = [
        ("what changed?", ("opt_2024.tif", "opt_far_away.tif")),
        ("what changed?", ("opt_2024.tif", "opt_2024_dup.tif")),
        ("what changed?", ("opt_2024.tif", "sar_2026.tif")),
        ("what changed?", ("opt_2024.tif", "opt_no_crs.tif")),
        ("has it increased?", ("opt_2024.tif",)),
        ("describe this", ("scene.bmp",)),
    ]
    for q, names in cases:
        t = run.answer(q, fx(*names))
        tracelib.validate(t)
        assert t["abstained"] is True, (q, names)


def test_fanout_records_one_step_per_image():
    imgs = fx("opt_2024.tif", "opt_2025.tif", "opt_2026.tif", "opt_2027.tif")
    t = run.answer("describe this scene", imgs)
    tracelib.validate(t)
    assert t["routing"]["fanout"]["n_calls"] == 4
    assert sum(1 for s in t["steps"] if s["tool"] == "caption.rs") == 4
    assert t["output"]["text"].count("Image ") == 4


def test_the_declared_pair_appears_in_the_trace():
    imgs = fx("opt_2024.tif", "opt_2025.tif", "opt_2026.tif", "opt_2027.tif")
    t = run.answer("has the built-up area increased?", imgs)
    tracelib.validate(t)
    assert t["routing"]["pair_declaration"]
    assert "earliest and latest" in t["routing"]["pair_declaration"]
    assert t["abstained"] is False


def test_fanout_confidence_is_the_minimum_not_the_mean():
    """An answer is only as trustworthy as its weakest part."""
    imgs = fx("opt_2024.tif", "opt_2025.tif", "opt_2026.tif")
    t = run.answer("describe this scene", imgs)
    per = [s["confidence"] for s in t["steps"] if s["tool"] == "caption.rs"]
    assert t["output"]["confidence"] == round(min(per), 3)


def test_grounding_exports_a_georeferenced_shape():
    """The difference between a demo and a tool."""
    t = run.answer("highlight the airport", fx("opt_2024.tif"))
    assert t["output"]["geojson"]
    gj = json.loads(t["output"]["geojson"])
    assert gj["crs"]["properties"]["name"] == "EPSG:32643"
    x, y = gj["features"][0]["geometry"]["coordinates"][0][0]
    assert 600000.0 <= x <= 620000.0 and 1500000.0 <= y <= 1520000.0
    assert any(s["tool"] == "export.geojson" for s in t["steps"])


def test_stub_results_are_labelled_as_stubs():
    """Nothing may present a canned answer as a real one."""
    t = run.answer("is there a water body?", fx("opt_2024.tif"))
    assert all(s["stub"] for s in t["steps"] if s["tool"] != "encode.rs_gsd" or True)


def test_a_compound_query_takes_the_planner_path_and_says_so():
    t = run.answer(
        "compare land cover between these two dates, and tell me if water increased",
        fx("opt_2024.tif", "opt_2026.tif"),
    )
    tracelib.validate(t)
    assert t["routing"]["by"] == "planner"
    assert t["routing"]["planner_used"] is True
    assert t["abstained"] is True


def test_replay_reproduces_byte_identical_output(tmp_path):
    p = str(tmp_path / "trace.json")
    t = run.answer("highlight the airport", fx("opt_2024.tif"))
    tracelib.write(t, p)
    assert run.replay(p) == 0


def test_replay_fails_when_the_registry_changes(tmp_path):
    """A different registry can produce a different answer for honest reasons.

    The lock exists so that shows up as a failed audit rather than a silently
    different result.
    """
    p = str(tmp_path / "trace.json")
    t = run.answer("is there a water body?", fx("opt_2024.tif"))
    t["replay"]["registry_lock"] = "0" * 64
    tracelib.write(t, p)
    assert run.replay(p) == 2


def test_the_same_query_twice_gives_the_same_answer():
    a = run.answer("highlight the airport", fx("opt_2024.tif"))
    b = run.answer("highlight the airport", fx("opt_2024.tif"))
    assert a["output"] == b["output"]


def test_replay_takes_the_input_files_explicitly(tmp_path):
    """A trace records file NAMES only - no machine paths in the graded
    artefact - so from anywhere but the fixture folder the caller hands the
    files back in. Their sha256 must match; the count must match."""
    import shutil
    img = tmp_path / "elsewhere" / "opt_2024.tif"
    img.parent.mkdir()
    shutil.copy(fx("opt_2024.tif")[0], img)
    p = str(tmp_path / "trace.json")
    t = run.answer("is there a water body?", [str(img)])
    tracelib.write(t, p)
    assert run.replay(p, [str(img)]) == 0
    assert run.replay(p, [str(img), str(img)]) == 2
