"""Zoom-and-re-ask: the one training-free lever the literature measured, as pure functions.

What it is. On a sub-metre tile the thing a question asks about can be a few pixels of a
512 px image, and the visual token cap (256) sees it as a smear. MAP-Agent (UHR-Micro,
arXiv 2605.12237) and ZoomEarth (arXiv 2511.12267, on Qwen2.5-VL-3B) both measured the same
fix: predict the region the question is about, crop it FROM THE ORIGINAL pixels, ask again
on the crop. Grounding went 6.3 -> 35.8 and counting 20 -> 33 on an 8B; on our base model
one crop step was worth +4.1 at 1.55x latency. Feeding a bigger image instead HURT.

Where it applies, and where it must not. A 256 px tile at 10 m has nothing to zoom into:
the first pass already saw every pixel. So the predicate is scale, read from the file - GSD
under 2 m when the file carries one, else more than 300 px on the long side - and never the
query. This is rule Z1 of the dispatcher; the routing rule (R4/R6) is unchanged, the zoom is
a step BEFORE the task tool, and the trace shows both.

Who proposes the region. Base weights. Our grounding adapter learned 120 px / 10 m tiles and
scores 13.0 on VRSBench referring where the untrained model scores 37.8 (7 Sep): the skill
does not survive the scale change, so on exactly the imagery this step exists for, the
adapter is the worse proposer. The task tool that then runs on the crop keeps its adapter.

Every function here is pure so the engine and scripts/eval_vrsbench.py --zoom share it byte
for byte, and the harness number is a measurement of the engine's behaviour, not of a copy.

MEASURED 7 Sep 2026, and it did not help. Kaggle T4, fp16, the same 400 stratified VRSBench
rows with and without the step (kernel ichigooo0o/satquery-vrsbench-zoom v2, records in
docs/results/vrsbench_*_fp16_kaggle_*.json): referring Acc@0.5 38.5 -> 34.2, VQA 42.7 -> 42.2,
at 3x the latency, with a region proposed on every row. The proposer is the weak link: a 3B
base model places the referring box right 38.5% of the time, so the crop misses the object
about as often as it finds it, and existence and quantity questions then answer on the wrong
patch (existence 69.7 -> 54.5, quantity 39.4 -> 33.3). The published gains came from an 8B
proposer (MAP-Agent) or a TRAINED zoom policy (ZoomEarth, SFT+GRPO); training-free with a 3B
proposer is a null. So Z1 is OFF unless SATQUERY_ZOOM=1: the row, the rule and the tests
stay, because the next version of this idea is a trained proposer, and everything but the
proposer is built and measured.
"""

from __future__ import annotations

import os

from .types import NormBox

MIN_LONG_SIDE_PX = 300     # a 256 px benchmark tile stays below this, a 512 px one is above
MAX_GSD_M = 2.0            # "sub-metre-ish": Cartosat-2S is 0.65 m, VRSBench ~0.1-1 m
MARGIN = 0.25              # context kept around the proposed region, as a fraction of it
MIN_CROP_PX = 224          # never crop below what the vision tower was built to see
ZOOM_TASKS = ("grounding", "vqa")
RULE_ID = "Z1"


def sub_metre(im) -> str | None:
    """The scale predicate on its own: the reason string when the FILE says the tile is
    sub-metre (GSD < MAX_GSD_M, or no GSD and long side > MIN_LONG_SIDE_PX), else None.
    Used by rule Z1 (zoom, opt-in) and by rule S1 (grounding on base weights, always on).

    S1, measured 7-8 Sep on VRSBench referring, 400 rows, nf4: the 10 m grounding adapter
    13.0, base weights 37.8, adapter behind a base-proposed crop 30.5. The adapter learned
    120 px tiles and is the wrong weights for sub-metre imagery; until a VRSBench adapter
    exists, ground.rs answers there on base weights and says so in the trace.
    """
    long_side = max(im.width, im.height)
    if im.gsd_m is not None:
        if im.gsd_m < MAX_GSD_M:
            return "gsd %.3g m < %g m: sub-metre tile" % (im.gsd_m, MAX_GSD_M)
        return None
    if long_side > MIN_LONG_SIDE_PX:
        return "no gsd in file; long side %d px > %d px" % (long_side, MIN_LONG_SIDE_PX)
    return None


def enabled() -> bool:
    """Z1 is opt-in since the 7 Sep measurement (module docstring)."""
    return os.environ.get("SATQUERY_ZOOM", "0").strip().lower() in ("1", "true", "yes", "on")


def wants(task: str, im) -> dict | None:
    """Rule Z1. None means no zoom; otherwise the reason, for the trace.

    `im` is an ImageMeta: width/height/gsd_m come from the FILE. The query is
    not consulted - a request cannot make a 10 m tile sub-metre.
    """
    if not enabled():
        return None
    if task not in ZOOM_TASKS:
        return None
    why = sub_metre(im)
    return {"rule_id": RULE_ID, "reason": why + ", detail exceeds the token cap"} if why else None


def expand(box: NormBox, width: int, height: int,
           margin: float = MARGIN, min_px: int = MIN_CROP_PX) -> NormBox:
    """The crop region for a proposed box: margin around it, at least min_px a
    side (but never larger than the image), clamped to the image. Pure."""
    bw, bh = box.x1 - box.x0, box.y1 - box.y0
    cx, cy = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
    w = max(bw * (1 + 2 * margin), min(1.0, min_px / float(width)))
    h = max(bh * (1 + 2 * margin), min(1.0, min_px / float(height)))
    w, h = min(w, 1.0), min(h, 1.0)
    x0 = min(max(cx - w / 2.0, 0.0), 1.0 - w)
    y0 = min(max(cy - h / 2.0, 0.0), 1.0 - h)
    return NormBox(round(x0, 4), round(y0, 4), round(x0 + w, 4), round(y0 + h, 4))


def crop_pixels(region: NormBox, width: int, height: int) -> tuple:
    """(left, top, right, bottom) integer pixels for PIL.crop, at least 1 px each way."""
    l, t = int(round(region.x0 * width)), int(round(region.y0 * height))
    r, b = int(round(region.x1 * width)), int(round(region.y1 * height))
    r, b = max(r, l + 1), max(b, t + 1)
    return l, t, min(r, width), min(b, height)


def map_back(box_in_crop: NormBox, region: NormBox) -> NormBox:
    """A box normalised inside the crop -> the same box normalised in the full image."""
    w, h = region.x1 - region.x0, region.y1 - region.y0
    return NormBox(
        round(region.x0 + box_in_crop.x0 * w, 4),
        round(region.y0 + box_in_crop.y0 * h, 4),
        round(region.x0 + box_in_crop.x1 * w, 4),
        round(region.y0 + box_in_crop.y1 * h, 4),
    )


def region_query(task: str, query: str) -> str:
    """What the proposer is asked to locate. For grounding it is the query itself;
    for a question it is the area the answer depends on."""
    if task == "grounding":
        return query
    return "the area of the image you would need to look at closely to answer: %s" % query
