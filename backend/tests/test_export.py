"""The QGIS moment, asserted.

Gate I says a georeferenced output must open in QGIS in the right place on
Earth. QGIS cannot be driven from a test, but everything that would make it
land in the wrong place can be: the CRS, the affine transform, the bounds, the
units of the coordinates, and the units the area is computed in.

The failure this guards against is specific and quiet. A file written in UTM
metres but labelled EPSG:4326 opens at longitude 600000, which is off the map
entirely; a file whose area was computed in degrees reports a number that is
wrong by a factor that changes with latitude. Neither raises an error. Both
look like a broken model.
"""

import glob
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satquery import export, run as sq_run  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
rasterio = pytest.importorskip("rasterio", reason="export writes real rasters")


def setup_module(module):
    if not os.path.exists(os.path.join(FIX, "opt_2024.tif")):
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(FIX), "make_fixtures.py")])


def fx(*names):
    return [os.path.join(FIX, n) for n in names]


@pytest.fixture()
def out_dir(tmp_path, monkeypatch):
    d = str(tmp_path / "out")
    monkeypatch.setattr(export, "OUT_DIR", d)
    return d


def test_geojson_comes_out_in_lon_lat_not_utm(out_dir):
    """The whole point. UTM metres labelled 4326 land off the map."""
    t = sq_run.answer("highlight the largest building", fx("opt_2024.tif"))
    path = glob.glob(os.path.join(out_dir, "*_shapes.geojson"))[0]
    gj = json.load(open(path, encoding="utf-8"))
    ring = gj["features"][0]["geometry"]["coordinates"][0]

    for lon, lat in ring:
        assert -180 <= lon <= 180, "longitude %r is not degrees - still in UTM?" % lon
        assert -90 <= lat <= 90, "latitude %r is not degrees - still in UTM?" % lat

    # The fixtures are a 1.2 km square in UTM 43N, which is Karnataka.
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    assert 75.9 < min(lons) < 76.0, lons
    assert 13.5 < min(lats) < 13.6, lats


def test_geojson_keeps_its_provenance(out_dir):
    sq_run.answer("highlight the largest building", fx("opt_2024.tif"))
    gj = json.load(open(glob.glob(os.path.join(out_dir, "*_shapes.geojson"))[0], encoding="utf-8"))
    props = gj["features"][0]["properties"]
    assert props["source_crs"] == "EPSG:32643"
    assert props["scene"] == "opt_2024.tif"
    assert props["stub"] is True, "a stubbed box must say so in the file it writes"


def test_area_is_computed_in_metres_not_degrees(out_dir):
    """A square degree is not a constant area.

    The box is ~0.22 x 0.18 of a 1.2 km square, so roughly 0.057 km2. Computed
    in degrees the same shape would come out around 4e-6 - small enough to look
    like a plausible number and be completely wrong.
    """
    sq_run.answer("highlight the largest building", fx("opt_2024.tif"))
    gj = json.load(open(glob.glob(os.path.join(out_dir, "*_shapes.geojson"))[0], encoding="utf-8"))
    area = gj["features"][0]["properties"]["area_km2"]
    assert 0.01 < area < 0.2, "area %r km2 is not a metric area" % area


def test_geotiff_preserves_crs_transform_and_bounds_exactly(out_dir):
    """The raster is evidence, so it must not be resampled on the way out."""
    sq_run.answer("has the built-up area increased?", fx("sar_2024.tif", "sar_2026.tif"))
    path = glob.glob(os.path.join(out_dir, "*_change_mask.tif"))[0]

    with rasterio.open(os.path.join(FIX, "sar_2024.tif")) as src:
        with rasterio.open(path) as dst:
            assert dst.crs == src.crs
            assert dst.transform == src.transform
            assert dst.bounds == src.bounds
            assert (dst.width, dst.height) == (src.width, src.height)
            assert dst.count == 1 and dst.dtypes[0] == "uint8"
            assert dst.nodata == 0
            assert dst.tags()["SATQUERY_STUB"] == "true"
            assert dst.tags()["SATQUERY_SOURCE"] == "sar_2024.tif"


def test_the_trace_carries_the_real_area_and_path(out_dir):
    t = sq_run.answer("has the built-up area increased?", fx("sar_2024.tif", "sar_2026.tif"))
    q = t["output"]["quantities"]
    assert q["export.crs"] == "EPSG:32643"
    assert q["export.area_km2"] > 0
    assert q["export.pixels"] > 0
    assert os.path.exists(t["output"]["raster"])
    assert any(s["tool"] == "export.geotiff" and not s["stub"] for s in t["steps"]), \
        "the export is real even while the mask content is stubbed"


def test_output_names_are_content_addressed_not_random(out_dir):
    """Two identical runs must produce ONE file, not two.

    The output paths go into the trace, and `--replay` compares that block byte
    for byte. A random name in there makes every replay fail for a reason that
    has nothing to do with the answer.
    """
    a = sq_run.answer("highlight the largest building", fx("opt_2024.tif"))
    b = sq_run.answer("highlight the largest building", fx("opt_2024.tif"))
    assert a["output"] == b["output"]
    assert len(glob.glob(os.path.join(out_dir, "*_shapes.geojson"))) == 1


def test_a_different_question_gets_a_different_file(out_dir):
    sq_run.answer("highlight the largest building", fx("opt_2024.tif"))
    sq_run.answer("highlight the runway", fx("opt_2024.tif"))
    assert len(glob.glob(os.path.join(out_dir, "*_shapes.geojson"))) == 2


def test_replay_still_byte_identical_with_real_exports(out_dir, tmp_path):
    from satquery import trace as tracelib

    p = str(tmp_path / "trace.json")
    t = sq_run.answer("highlight the largest building", fx("opt_2024.tif"))
    tracelib.write(t, p)
    assert sq_run.replay(p) == 0
