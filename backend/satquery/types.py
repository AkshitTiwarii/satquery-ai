"""The data shapes every other module is written against.

Read this file before any of the others. Most of the bugs this system can have
are data-shape bugs, and two of them are worth naming up front:

  1. `ImageMeta` is built ONLY by `load_image_meta`, which reads the file. No
     field on it may ever come from the query. A judge can hand over two images
     of different continents and assert in the query that they are the same
     place; the gate compares footprints computed from each file's own CRS and
     affine transform and refuses. THE QUERY IS A REQUEST, NOT EVIDENCE. Every
     claim in it loses to the file: claimed dates lose to timestamps, claimed
     modality loses to band count.

  2. Bounding boxes. BigEarthNet.txt normalises boxes 0-1, VRSBench 0-100. A
     `BBox` here is always in CRS units and a `NormBox` is always 0-1. They are
     different types on purpose, because mixing them is wrong by 100x and
     raises no error.

`Plan` is the piece that keeps the dispatcher honest. The statement never
capped input at two images, so the gate resolves N images into groups of one or
two BEFORE routing - a fan-out for per-image questions, one declared pair for
change and fusion. The dispatcher therefore only ever sees a shape it has a
rule for, which is what makes R0-R6 exhaustive.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

# Modalities, exactly as the registry spells them.
OPTICAL = "optical"
MSI = "msi"
SAR = "sar"

SUPPORTED_FORMATS = ("GeoTIFF", "TIFF", "PNG", "JPEG")

# PNG and JPEG carry no georeferencing. The statement allows them only for the
# prescribed public benchmarks, and for those the footprint check degrades to a
# visual comparison - which the trace has to say out loud.
UNGEOREFERENCED = ("PNG", "JPEG")


@dataclass(frozen=True)
class BBox:
    """A footprint in CRS units. Never normalised, never 0-1."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    def overlaps(self, other: "BBox") -> bool:
        return not (
            self.maxx <= other.minx
            or other.maxx <= self.minx
            or self.maxy <= other.miny
            or other.maxy <= self.miny
        )

    def as_list(self) -> list:
        return [self.minx, self.miny, self.maxx, self.maxy]


@dataclass(frozen=True)
class NormBox:
    """A box normalised to 0-1. The ONLY normalisation this codebase uses.

    VRSBench ships 0-100 and BigEarthNet.txt ships 0-1; both are converted here
    at load time and the range is asserted, so nothing downstream has to know
    which file a box came from.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for v in (self.x0, self.y0, self.x1, self.y1):
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    "box coordinate %r is outside 0-1 - a 0-100 box was probably "
                    "passed in without conversion" % (v,)
                )

    @classmethod
    def from_scale(cls, x0: float, y0: float, x1: float, y1: float, scale: float) -> "NormBox":
        """Build from a box expressed on a 0-`scale` grid (1 or 100)."""
        return cls(x0 / scale, y0 / scale, x1 / scale, y1 / scale)


@dataclass(frozen=True)
class ImageMeta:
    """What the gate is allowed to know about one input image.

    Every field is read from the file itself. See the module docstring.
    """

    path: str
    sha256: str
    fmt: str
    n_bands: int
    modality: str
    width: int
    height: int
    crs: Optional[str] = None
    bbox: Optional[BBox] = None
    gsd_m: Optional[float] = None
    acquired_at: Optional[str] = None  # ISO-8601, from the file's own metadata

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.bbox is not None

    def summary(self) -> dict:
        """The form that goes into the trace's input_check block."""
        return {
            "path": os.path.basename(self.path),
            "sha256": self.sha256,
            "format": self.fmt,
            "bands": self.n_bands,
            "modality": self.modality,
            "size_px": [self.width, self.height],
            "crs": self.crs,
            "bbox": self.bbox.as_list() if self.bbox else None,
            "gsd_m": self.gsd_m,
            "acquired_at": self.acquired_at,
        }


@dataclass
class Plan:
    """How N supplied images resolve into calls the dispatcher can route.

    mode is one of:
      "single"  - one image, one call
      "pair"    - exactly two images, one call
      "fanout"  - more than two images and a per-image question, N calls

    `groups` holds indices into the image list. Every group has length 1 or 2,
    which is what lets the dispatcher rules stay exhaustive.

    `declaration` is set only when a pair was CHOSEN out of more than two
    images. It is not a refusal - it is the system saying which two it used and
    why, so the user can take the decision back.
    """

    mode: str
    groups: list = field(default_factory=list)
    declaration: Optional[str] = None


@dataclass
class ToolResult:
    """What every registry row returns. Stub or real, the shape is the same."""

    text: str
    # None means "the component does not report one". Real model answers have
    # no calibrated confidence yet, and inventing one would defeat abstention.
    confidence: Optional[float]
    quantities: dict = field(default_factory=dict)
    geojson: Optional[dict] = None
    raster: Optional[str] = None
    stub: bool = False


def sha256_of(path: str) -> str:
    """Content hash, recorded in the trace so --replay can verify its inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _modality_from_bands(n_bands: int) -> str:
    """Derive modality from band count.

    reBEN excludes the three 60 m bands, so Sentinel-2 arrives as 12 and not 13.
    Code that assumes 13 mis-aligns every channel, so both are accepted here and
    anything else with ten or more bands is treated as multispectral.
    """
    if n_bands == 2:
        return SAR  # Sentinel-1 GRD: VV and VH
    if n_bands >= 10:
        return MSI
    if n_bands in (3, 4):
        return OPTICAL
    raise ValueError(
        "cannot derive modality from %d bands - declare it in the file's "
        "metadata tags" % n_bands
    )


def _fmt_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".tif": "GeoTIFF",
        ".tiff": "GeoTIFF",
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
    }.get(ext, ext.lstrip(".").upper() or "UNKNOWN")


def load_image_meta(path: str) -> ImageMeta:
    """Read one input file into an ImageMeta.

    Three sources, in order of authority:

      1. rasterio, when it is installed and the file is a GeoTIFF. This is the
         real path and the only one that runs against the hidden ISRO set.
      2. A `<path>.meta.json` sidecar. Used for fixtures and for benchmark PNGs
         that carry no georeferencing of their own. A sidecar may only supply
         what the file genuinely records - it is not a place to assert a
         footprint the pixels do not have.
      3. Pillow, for the size of a plain PNG or JPEG.

    Note what is NOT a source: the user's query.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    fmt = _fmt_from_path(path)
    side = path + ".meta.json"
    sidecar = {}
    if os.path.exists(side):
        with open(side, "r", encoding="utf-8") as fh:
            sidecar = json.load(fh)

    if fmt in ("GeoTIFF", "TIFF"):
        try:
            import rasterio  # noqa: F401  (optional; absent on dev laptops)
        except ImportError:
            pass
        else:
            try:
                return _meta_from_rasterio(path, fmt, sidecar)
            except Exception:
                # rasterio refused the file. Fall back ONLY when a sidecar
                # exists - that is the fixture and dev-laptop case. With no
                # sidecar this is a genuinely unreadable raster and the caller
                # must hear about it rather than get invented metadata.
                if not sidecar:
                    raise

    if fmt in UNGEOREFERENCED:
        width, height, n_bands = _size_from_pillow(path, sidecar)
        return ImageMeta(
            path=path,
            sha256=sha256_of(path),
            fmt=fmt,
            n_bands=n_bands,
            modality=sidecar.get("modality") or _modality_from_bands(n_bands),
            width=width,
            height=height,
            crs=None,
            bbox=None,
            gsd_m=sidecar.get("gsd_m"),
            acquired_at=sidecar.get("acquired_at"),
        )

    if not sidecar:
        raise RuntimeError(
            "cannot read %s: rasterio is not installed and there is no "
            "%s sidecar" % (path, os.path.basename(side))
        )
    return _meta_from_sidecar(path, fmt, sidecar)


def _meta_from_rasterio(path: str, fmt: str, sidecar: dict) -> ImageMeta:
    import rasterio

    with rasterio.open(path) as src:
        b = src.bounds
        tags = src.tags()
        # GSD comes from the affine transform, not from anyone's assertion.
        # With no CRS there is no georeferenced transform to read it from, so
        # fall back to the sidecar - which is how a benchmark PNG documents the
        # 10 m its dataset is published at.
        gsd = abs(src.transform.a) if src.crs else sidecar.get("gsd_m")
        n_bands = src.count
        return ImageMeta(
            path=path,
            sha256=sha256_of(path),
            fmt=fmt,
            n_bands=n_bands,
            modality=tags.get("MODALITY") or _modality_from_bands(n_bands),
            width=src.width,
            height=src.height,
            crs=str(src.crs) if src.crs else None,
            bbox=BBox(b.left, b.bottom, b.right, b.top) if src.crs else None,
            gsd_m=gsd,
            acquired_at=tags.get("ACQUISITION_DATE") or sidecar.get("acquired_at"),
        )


def _meta_from_sidecar(path: str, fmt: str, s: dict) -> ImageMeta:
    bbox = BBox(*s["bbox"]) if s.get("bbox") else None
    n_bands = int(s["bands"])
    return ImageMeta(
        path=path,
        sha256=sha256_of(path),
        fmt=fmt,
        n_bands=n_bands,
        modality=s.get("modality") or _modality_from_bands(n_bands),
        width=int(s.get("width", 0)),
        height=int(s.get("height", 0)),
        crs=s.get("crs"),
        bbox=bbox,
        gsd_m=s.get("gsd_m"),
        acquired_at=s.get("acquired_at"),
    )


def _size_from_pillow(path: str, sidecar: dict):
    """Size and band count of a PNG or JPEG.

    Falls back to the sidecar when Pillow is missing or the file will not open,
    which is what test fixtures are: a name and a header, no pixels.
    """
    fallback = (
        int(sidecar.get("width", 0)),
        int(sidecar.get("height", 0)),
        int(sidecar.get("bands", 3)),
    )
    try:
        from PIL import Image
    except ImportError:
        return fallback
    try:
        with Image.open(path) as im:
            return im.width, im.height, len(im.getbands())
    except Exception:
        if sidecar:
            return fallback
        raise
