"""Zoom-and-re-ask: the predicate reads the file not the query, the crop maths round-trips,
and the trace shows the step before the task tool - only where the scale calls for it.

Stub backend throughout (conftest). The real-model gain is measured by
scripts/eval_vrsbench.py --zoom, which imports the same functions.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from satquery import run, zoom  # noqa: E402
from satquery.types import NormBox  # noqa: E402


@pytest.fixture
def zoom_on(monkeypatch):
    """Z1 is off by default since the 7 Sep measurement; these tests exercise the
    mechanism, so they turn it on explicitly."""
    monkeypatch.setenv("SATQUERY_ZOOM", "1")

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def setup_module(module):
    if not os.path.exists(os.path.join(FIX, "bench_hr.png")):
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(FIX), "make_fixtures.py")])


def fx(*names):
    return [os.path.join(FIX, n) for n in names]


def _im(width, height, gsd):
    return SimpleNamespace(width=width, height=height, gsd_m=gsd)


# --- Z1, the predicate -------------------------------------------------------

def test_zoom_is_off_by_default_because_it_measured_negative(monkeypatch):
    monkeypatch.delenv("SATQUERY_ZOOM", raising=False)
    assert not zoom.enabled()
    assert zoom.wants("grounding", _im(512, 512, None)) is None
    t = run.answer("where is the runway?", fx("bench_hr.png"))
    assert "zoom.crop" not in [s["tool"] for s in t["steps"]]


def test_a_10m_benchmark_tile_is_never_zoomed_whatever_its_pixels(zoom_on):
    assert zoom.wants("grounding", _im(256, 256, 10.0)) is None
    assert zoom.wants("vqa", _im(512, 512, 10.0)) is None      # gsd in the file wins over pixels


def test_sub_metre_gsd_zooms_even_at_120px(zoom_on):
    z = zoom.wants("grounding", _im(120, 120, 0.65))
    assert z and z["rule_id"] == "Z1" and "gsd" in z["reason"]


def test_no_gsd_falls_back_to_the_pixel_rule(zoom_on):
    assert zoom.wants("vqa", _im(512, 512, None))["rule_id"] == "Z1"
    assert zoom.wants("vqa", _im(256, 256, None)) is None
    assert zoom.wants("vqa", _im(300, 300, None)) is None       # strictly greater


def test_only_grounding_and_vqa_zoom(zoom_on):
    hr = _im(512, 512, None)
    assert zoom.wants("captioning", hr) is None
    assert zoom.wants("change_vqa", hr) is None
    assert zoom.wants("fusion", hr) is None


# --- S1: grounding on base weights for sub-metre tiles, always on ------------------

def test_scale_rule_reads_the_file_like_z1_but_needs_no_switch(monkeypatch):
    monkeypatch.delenv("SATQUERY_ZOOM", raising=False)
    assert zoom.sub_metre(_im(512, 512, None)) is not None
    assert zoom.sub_metre(_im(120, 120, 0.65)) is not None
    assert zoom.sub_metre(_im(256, 256, 10.0)) is None
    assert zoom.sub_metre(_im(512, 512, 10.0)) is None


def test_grounding_on_a_sub_metre_tile_records_base_weights_in_the_trace(monkeypatch):
    monkeypatch.delenv("SATQUERY_ZOOM", raising=False)
    t = run.answer("where is the runway?", fx("bench_hr.png"))
    q = t["output"]["quantities"]
    assert q["adapter_rule"] and q["adapter_rule"].startswith("S1 base weights")
    t = run.answer("where is the runway?", fx("bench_lr.png"))
    assert t["output"]["quantities"]["adapter_rule"] is None


# --- the crop maths ------------------------------------------------------------

def test_expand_adds_margin_and_clamps_to_the_image():
    r = zoom.expand(NormBox(0.4, 0.4, 0.6, 0.6), 1000, 1000, margin=0.25, min_px=10)
    assert (r.x0, r.y0, r.x1, r.y1) == (0.35, 0.35, 0.65, 0.65)
    r = zoom.expand(NormBox(0.0, 0.0, 0.1, 0.1), 1000, 1000, margin=0.25, min_px=10)
    assert r.x0 == 0.0 and r.y0 == 0.0 and r.x1 > 0.1        # pushed inward, not clipped smaller


def test_expand_never_crops_below_min_px_or_beyond_the_image():
    r = zoom.expand(NormBox(0.50, 0.50, 0.51, 0.51), 512, 512, min_px=224)
    assert abs((r.x1 - r.x0) * 512 - 224) < 1.0
    r = zoom.expand(NormBox(0.1, 0.1, 0.9, 0.9), 200, 200, min_px=224)
    assert (r.x0, r.y0, r.x1, r.y1) == (0.0, 0.0, 1.0, 1.0)   # image smaller than min: whole image


def test_map_back_round_trips_the_crop_centre_and_corners():
    region = NormBox(0.2, 0.3, 0.6, 0.8)
    assert zoom.map_back(NormBox(0.5, 0.5, 0.5, 0.5), region) == NormBox(0.4, 0.55, 0.4, 0.55)
    assert zoom.map_back(NormBox(0.0, 0.0, 1.0, 1.0), region) == region
    l, t, r, b = zoom.crop_pixels(region, 500, 400)
    assert (l, t, r, b) == (100, 120, 300, 320)


# --- the trace ---------------------------------------------------------------

def test_a_high_res_grounding_query_zooms_and_the_trace_shows_it_before_the_tool(zoom_on):
    t = run.answer("where is the runway?", fx("bench_hr.png"))
    tools = [s["tool"] for s in t["steps"]]
    assert "zoom.crop" in tools and "ground.rs" in tools
    assert tools.index("zoom.crop") < tools.index("ground.rs")
    z = next(s for s in t["steps"] if s["tool"] == "zoom.crop")
    assert z["params"]["rule_id"] == "Z1"
    assert t["routing"]["rule_id"] == "R4"          # the routing rule is untouched
    q = t["output"]["quantities"]
    assert q["zoomed"] is True and len(q["crop_region"]) == 4
    assert all(0.0 <= v <= 1.0 for v in q["norm_box"])   # the box is in the FULL image frame


def test_a_low_res_query_does_not_zoom_and_the_trace_is_unchanged(zoom_on):
    t = run.answer("where is the runway?", fx("bench_lr.png"))
    assert "zoom.crop" not in [s["tool"] for s in t["steps"]]
    assert "zoomed" not in t["output"]["quantities"]


def test_captioning_on_a_high_res_tile_does_not_zoom(zoom_on):
    t = run.answer("describe this image", fx("bench_hr.png"))
    assert "zoom.crop" not in [s["tool"] for s in t["steps"]]


def test_a_zoomed_trace_replays_byte_identical(zoom_on, tmp_path):
    t = run.answer("how many aircraft are visible?", fx("bench_hr.png"))
    assert any(s["tool"] == "zoom.crop" for s in t["steps"])
    p = tmp_path / "t.json"
    p.write_text(json.dumps(t), encoding="utf-8")
    assert run.replay(str(p), fx("bench_hr.png")) == 0
