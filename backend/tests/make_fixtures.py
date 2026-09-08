"""Write the test fixtures.

With rasterio installed these are REAL GeoTIFFs - real CRS, real affine
transform, real band counts, acquisition date in the file's own metadata tags.
That matters for more than tidiness: the gate's whole job is to reason about
what a file actually declares, and a test that feeds it a sidecar is testing
the sidecar. It is also what makes the QGIS check possible at all, because you
cannot drag a placeholder into a map.

Without rasterio it falls back to placeholder bytes plus a `.meta.json`
sidecar, so the suite still runs on a laptop with no geospatial stack.

Pixel content is a deterministic pattern, never random: the trace records the
sha256 of every input and `--replay` verifies it, so a fixture that changed
byte-for-byte between runs would break replay for a reason that has nothing to
do with the answer.

    python tests/make_fixtures.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")

# A 1.2 km square in UTM 43N - Karnataka, around 75.92 E, 13.57 N - and a
# second square far enough away that the footprints cannot overlap, for G2.
#
# The size is not arbitrary: 120 px at 10 m/px IS 1200 m. GSD is read from the
# affine transform, so a fixture that claims 10 m in its sidecar while its
# transform says otherwise is a fixture that lies, and the loader will believe
# the transform - correctly. Keep these three numbers in step.
HERE_BBOX = [600000.0, 1500000.0, 601200.0, 1501200.0]
FAR_BBOX = [200000.0, 900000.0, 201200.0, 901200.0]

FIXTURES = {
    # optical pair, same place, two dates -> R3
    "opt_2024.tif": dict(bands=12, modality="msi", crs="EPSG:32643", bbox=HERE_BBOX,
                         gsd_m=10.0, acquired_at="2024-01-14T05:31:00Z", width=120, height=120),
    "opt_2026.tif": dict(bands=12, modality="msi", crs="EPSG:32643", bbox=HERE_BBOX,
                         gsd_m=10.0, acquired_at="2026-02-02T05:29:00Z", width=120, height=120),
    # radar pair, same place, two dates -> R2
    "sar_2024.tif": dict(bands=2, modality="sar", crs="EPSG:32643", bbox=HERE_BBOX,
                         gsd_m=10.0, acquired_at="2024-01-14T05:31:00Z", width=120, height=120),
    "sar_2026.tif": dict(bands=2, modality="sar", crs="EPSG:32643", bbox=HERE_BBOX,
                         gsd_m=10.0, acquired_at="2026-02-02T05:29:00Z", width=120, height=120),
    # radar on the SAME day as opt_2024 -> R1 cross-modal
    "sar_2024_same_day.tif": dict(bands=2, modality="sar", crs="EPSG:32643", bbox=HERE_BBOX,
                                  gsd_m=10.0, acquired_at="2024-01-14T05:31:00Z", width=120, height=120),
    # a third and fourth optical date, for the N>2 cases
    "opt_2025.tif": dict(bands=12, modality="msi", crs="EPSG:32643", bbox=HERE_BBOX,
                         gsd_m=10.0, acquired_at="2025-03-08T05:30:00Z", width=120, height=120),
    "opt_2027.tif": dict(bands=12, modality="msi", crs="EPSG:32643", bbox=HERE_BBOX,
                         gsd_m=10.0, acquired_at="2027-04-19T05:28:00Z", width=120, height=120),
    # G2 - a different part of the world, same CRS
    "opt_far_away.tif": dict(bands=12, modality="msi", crs="EPSG:32643", bbox=FAR_BBOX,
                             gsd_m=10.0, acquired_at="2026-02-02T05:29:00Z", width=120, height=120),
    # G3 - same sensor, same day as opt_2024
    "opt_2024_dup.tif": dict(bands=12, modality="msi", crs="EPSG:32643", bbox=HERE_BBOX,
                             gsd_m=10.0, acquired_at="2024-01-14T05:31:00Z", width=120, height=120),
    # a different CRS - reprojectable, so NOT a refusal
    "opt_2026_utm44.tif": dict(bands=12, modality="msi", crs="EPSG:32644", bbox=HERE_BBOX,
                               gsd_m=10.0, acquired_at="2026-02-02T05:29:00Z", width=120, height=120),
    # G5 - a real TIFF with no CRS at all, so there is nothing to reproject from
    "opt_no_crs.tif": dict(bands=12, modality="msi", crs=None, bbox=None,
                           gsd_m=10.0, acquired_at="2026-02-02T05:29:00Z", width=120, height=120),
    # benchmark PNGs: no georeferencing, so the footprint check degrades
    "bench_lr.png": dict(bands=3, modality="optical", crs=None, bbox=None,
                         gsd_m=10.0, acquired_at=None, width=256, height=256),
    "bench_lr_2.png": dict(bands=3, modality="optical", crs=None, bbox=None,
                           gsd_m=10.0, acquired_at="2026-02-02T05:29:00Z", width=256, height=256),
    # a sub-metre benchmark tile (VRSBench-shaped): 512 px, no GSD in the file,
    # so rule Z1 fires on the pixel count. bench_lr.png above is 256 px at 10 m
    # and must NOT zoom, whatever the pixels say.
    "bench_hr.png": dict(bands=3, modality="optical", crs=None, bbox=None,
                         gsd_m=None, acquired_at=None, width=512, height=512),
    # G7 - a format we do not accept
    "scene.bmp": dict(bands=3, modality="optical", crs=None, bbox=None,
                      gsd_m=None, acquired_at=None, width=64, height=64),
}


def _pattern(width, height, band, seed_name):
    """Deterministic pixels: a gradient plus a block whose position is stable.

    Never random. A fixture whose bytes move between runs breaks `--replay`,
    which verifies the sha256 of every input.
    """
    import numpy as np

    off = (sum(bytearray(seed_name.encode())) + band * 37) % 64
    y, x = np.mgrid[0:height, 0:width]
    arr = ((x + y + off) % 256).astype("uint16") * 16
    arr[height // 4: height // 2, width // 4: width // 2] = 40000 + band * 100
    return arr


def _write_tif(path, meta):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    w, h, n = meta["width"], meta["height"], meta["bands"]
    name = os.path.basename(path)
    data = np.stack([_pattern(w, h, b, name) for b in range(n)])

    if meta["bbox"]:
        transform = from_bounds(*meta["bbox"], width=w, height=h)
    else:
        transform = rasterio.Affine.identity()

    profile = dict(
        driver="GTiff", width=w, height=h, count=n, dtype="uint16",
        transform=transform, compress="deflate", predictor=2,
    )
    if meta["crs"]:
        profile["crs"] = rasterio.crs.CRS.from_string(meta["crs"])

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        # MODALITY and ACQUISITION_DATE are read back by load_image_meta. They
        # live in the file, which is the point - the gate must never take these
        # from anything the user typed.
        tags = {"MODALITY": meta["modality"]}
        if meta["acquired_at"]:
            tags["ACQUISITION_DATE"] = meta["acquired_at"]
        dst.update_tags(**tags)


def _write_png(path, meta):
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        with open(path, "wb") as fh:
            fh.write(b"SATQUERY-FIXTURE:" + os.path.basename(path).encode() + b"\n")
        return
    w, h = meta["width"], meta["height"]
    name = os.path.basename(path)
    arr = np.dstack([(_pattern(w, h, b, name) >> 8).astype("uint8") for b in range(3)])
    Image.fromarray(arr, mode="RGB").save(path, optimize=True)


def main():
    os.makedirs(FIX, exist_ok=True)
    try:
        import rasterio  # noqa: F401
        have_rio = True
    except ImportError:
        have_rio = False

    real = 0
    for name, meta in FIXTURES.items():
        path = os.path.join(FIX, name)
        ext = os.path.splitext(name)[1].lower()
        if ext == ".tif" and have_rio:
            _write_tif(path, meta)
            real += 1
        elif ext == ".png":
            _write_png(path, meta)
        else:
            with open(path, "wb") as fh:
                fh.write(b"SATQUERY-FIXTURE:" + name.encode("ascii") + b"\n")
        # The sidecar is written either way: it is the fallback on a machine
        # with no rasterio, and it documents what each fixture is FOR.
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
            fh.write("\n")

    print("wrote %d fixtures to %s (%d real GeoTIFFs)" % (len(FIXTURES), FIX, real))
    if not have_rio:
        print("  rasterio not installed - GeoTIFFs are placeholders read via sidecar")


if __name__ == "__main__":
    main()
