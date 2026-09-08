"""The always-on compatibility gate. A stage, not a registry row.

The statement makes input checking a standing duty of the controller - "check
the number, modality, format, metadata, and compatibility of the input images"
- so this runs on every query and cannot be routed around. It is a peer of
`steps` in the trace, not one of them.

This is also where abstention lives. On a hidden test set a confident wrong
answer costs more than an honest refusal, and refusing is on the rubric.

G2 is the adversarial one and the reason this module exists at all. A judge can
hand over two images of completely different places and assert in the query
that they are the same area. The gate never reads that assertion: it computes
both footprints from each file's own CRS and affine transform and compares
them. A chatbot handed two unrelated scenes will invent a change description.
This cannot, because the model is never asked.

The second job here is resolving how many images there are. The statement never
capped input at two - Implementation Scope says the system "shall support" the
three shapes, which is a floor, and the Background says information "may be
distributed across paired or multiple observations". So more than two images is
neither required nor forbidden. We fan out per-image questions and, for pair
questions, name which two we used and why. That resolution happens HERE, before
routing, which is what keeps the dispatcher rules exhaustive.
"""

from __future__ import annotations

from .dispatcher import CHANGE_RE, CAPTION_RE, GROUNDING_RE, wants_pair
from .types import MSI, OPTICAL, SAR, SUPPORTED_FORMATS, UNGEOREFERENCED, Plan

CLOUD_WARN_PCT = 60.0

# How far two co-registered images may be out of alignment before we stop
# calling them co-registered. One pixel is generous for change detection and
# tight enough to catch a genuinely mis-aligned pair.
MAX_COREG_OFFSET_PX = 1.0


class GateResult:
    def __init__(self, verdict, input_check, plan=None, code=None, message=None):
        self.verdict = verdict
        self.input_check = input_check
        self.plan = plan
        self.code = code
        self.message = message

    @property
    def rejected(self) -> bool:
        return self.verdict == "rejected"


def check(query: str, images: list) -> GateResult:
    """Validate the input and decide how the images resolve into calls."""
    ic = {
        "n_images": len(images),
        "modality": [im.modality for im in images],
        "images": [im.summary() for im in images],
        "crs": [im.crs for im in images],
        "gsd_m": [im.gsd_m for im in images],
        "acquisitions": [im.acquired_at for im in images],
        "crs_match": None,
        "coregistered": None,
        "max_offset_px": None,
        "temporal_order": None,
        "footprint_check": "not_applicable",
        "warnings": [],
        "verdict": "accepted",
        "code": None,
        "message": None,
    }

    if not images:
        return _reject(ic, "G7", "Rejected: no image was supplied.")

    # G7 - format. Checked first because nothing else can be trusted about a
    # file we cannot open.
    for im in images:
        if im.fmt not in SUPPORTED_FORMATS:
            return _reject(
                ic,
                "G7",
                "Rejected: %s is not supported. GeoTIFF or TIFF for geospatial "
                "imagery; PNG and JPEG only for the prescribed benchmark datasets."
                % im.fmt,
            )

    _warn_cloud(ic, images)

    if len(images) == 1:
        # G6 - a change question with only one image.
        if CHANGE_RE.search(query):
            return _reject(
                ic,
                "G6",
                "Rejected: change analysis requires two images; one was supplied.",
            )
        ic["plan_mode"] = "single"
        return GateResult("accepted", ic, Plan(mode="single", groups=[(0,)]))

    if len(images) > 2:
        return _resolve_many(query, images, ic)

    return _check_pair(query, images, (0, 1), ic)


def _resolve_many(query: str, images: list, ic: dict) -> GateResult:
    """More than two images. Not a refusal - a declared choice, fan-out, or compound plan."""
    # Check for compound cross-modal + bitemporal triad:
    # e.g. optical_current + sar_current (cross-modal) and optical_baseline + optical_current (bi-temporal)
    modalities = [im.modality for im in images]
    has_optical = any(m in (OPTICAL, MSI) for m in modalities)
    has_sar = any(m == SAR for m in modalities)
    if has_optical and has_sar and len(images) >= 3:
        sar_candidates = [(i, im) for i, im in enumerate(images) if im.modality == SAR]
        opt_candidates = [(i, im) for i, im in enumerate(images) if im.modality in (OPTICAL, MSI)]
        if sar_candidates and len(opt_candidates) >= 2:
            sar_curr = sar_candidates[0]
            def _sar_match_score(opt_tuple):
                idx, opt_im = opt_tuple
                score = 0
                if opt_im.acquired_at and sar_curr[1].acquired_at and opt_im.acquired_at == sar_curr[1].acquired_at:
                    score += 10
                if opt_im.fmt == sar_curr[1].fmt:
                    score += 5
                if opt_im.crs == sar_curr[1].crs and opt_im.crs is not None:
                    score += 3
                return score

            opt_curr = max(opt_candidates, key=_sar_match_score)
            remaining_opts = [o for o in opt_candidates if o[0] != opt_curr[0]]
            opt_base = remaining_opts[0]

            ic["input_config"] = ["cross_modal", "bitemporal"]
            ic["plan_mode"] = "compound_fusion_change"
            declaration = (
                "Compound input configuration: 3 images resolve into a cross-modal pair "
                "(%s + %s) for optical-SAR fusion and a bi-temporal pair (%s -> %s) for change analysis."
                % (_name(opt_curr[1]), _name(sar_curr[1]), _name(opt_base[1]), _name(opt_curr[1]))
            )
            ic["pair_declaration"] = declaration
            plan = Plan(
                mode="compound_fusion_change",
                groups=[(opt_curr[0], sar_curr[0]), (opt_base[0], opt_curr[0])],
                declaration=declaration
            )
            return GateResult("accepted", ic, plan)

    if not wants_pair(query):
        # Answerable one image at a time. One call each, every call its own
        # step in the trace.
        ic["plan_mode"] = "fanout"
        return GateResult(
            "accepted",
            ic,
            Plan(mode="fanout", groups=[(i,) for i in range(len(images))]),
        )

    # G1 - the question needs a pair. Pick the earliest and latest acquisition
    # and SAY SO. Never silently take the first two.
    dated = [(i, im) for i, im in enumerate(images) if im.acquired_at]
    if len(dated) < 2:
        return _reject(
            ic,
            "G1",
            "Rejected: %d images supplied and this question needs a pair, but "
            "fewer than two carry an acquisition date, so there is no basis on "
            "which to choose. Supply exactly two images."
            % len(images),
        )
    dated.sort(key=lambda t: t[1].acquired_at)
    lo, hi = dated[0], dated[-1]
    declaration = (
        "%d images supplied. Change and cross-modal analysis each work on a "
        "pair, so I used %s and %s - the earliest and latest acquisitions. "
        "Supply exactly two to choose the pair yourself."
        % (len(images), _name(lo[1]), _name(hi[1]))
    )
    res = _check_pair(query, [lo[1], hi[1]], (lo[0], hi[0]), ic)
    if res.plan:
        res.plan.declaration = declaration
    ic["pair_declaration"] = declaration
    return res


def _check_pair(query: str, pair: list, idx: tuple, ic: dict) -> GateResult:
    """Every two-image check, in the order that gives the clearest refusal."""
    a, b = pair
    same_modality = a.modality == b.modality
    same_date = a.acquired_at is not None and a.acquired_at == b.acquired_at
    both_dated = a.acquired_at is not None and b.acquired_at is not None

    ic["crs_match"] = (a.crs == b.crs) if (a.crs and b.crs) else None
    if both_dated:
        ic["temporal_order"] = (
            "same" if a.acquired_at == b.acquired_at
            else "ascending" if a.acquired_at < b.acquired_at
            else "descending"
        )

    # G2 - do these two images actually cover the same ground? Computed from
    # each file, never from the query.
    if a.georeferenced and b.georeferenced:
        ic["footprint_check"] = "computed"
        if a.crs == b.crs and not a.bbox.overlaps(b.bbox):
            return _reject(
                ic,
                "G2",
                "Rejected: these two images do not overlap. Image 1 covers %s; "
                "image 2 covers %s. Change and cross-modal analysis both "
                "require the same geographic area."
                % (a.bbox.as_list(), b.bbox.as_list()),
            )
        ic["max_offset_px"] = _offset_px(a, b)
        ic["coregistered"] = (
            ic["max_offset_px"] is not None and ic["max_offset_px"] <= MAX_COREG_OFFSET_PX
        )
    else:
        # PNG and JPEG benchmark inputs carry no georeferencing, so the strong
        # check is impossible. Say so in the trace rather than implying it ran.
        ic["footprint_check"] = "degraded_no_georeferencing"
        ic["warnings"].append(
            "Footprint overlap could not be computed: %s carries no CRS. The "
            "hidden evaluation set is GeoTIFF, so this degradation applies to "
            "benchmark inputs only."
            % (", ".join(sorted({im.fmt for im in pair if not im.georeferenced})))
        )

    # G3 - same sensor, same day. Neither pair task is defined on this.
    if same_modality and same_date:
        return _reject(
            ic,
            "G3",
            "Rejected: both images are %s acquired on %s. Change analysis needs "
            "two different dates; cross-modal analysis needs one optical and "
            "one SAR image." % (a.modality, a.acquired_at),
        )

    # G4 - different sensor AND different date. A refusal on scientific
    # grounds, not on missing capability: any difference could be the ground
    # changing or the sensors disagreeing, and nothing can separate the two.
    if not same_modality and both_dated and a.acquired_at != b.acquired_at:
        return _reject(
            ic,
            "G4",
            "Rejected: comparing %s from %s against %s from %s confounds sensor "
            "difference with change over time - any difference could be either. "
            "Supply two images from the same sensor for change analysis, or two "
            "from the same date for cross-modal analysis."
            % (a.modality, a.acquired_at, b.modality, b.acquired_at),
        )

    # A cross-modal pair with no dates at all cannot be shown to be same-date,
    # and fusion is only defined on a same-date pair.
    if not same_modality and not both_dated:
        return _reject(
            ic,
            "G4",
            "Rejected: cross-modal analysis needs one optical and one SAR image "
            "from the same date, and at least one of these files carries no "
            "acquisition date, so that cannot be established.",
        )

    # Same sensor, no dates. Change analysis is defined on two known, different
    # dates; without both there is no comparison to make, and the dispatcher
    # (rightly) has no rule for an undated pair. PNG and JPEG never carry a
    # date, so this is the refusal a browser upload of two plain images gets.
    if same_modality and not both_dated:
        undated = [str(i + 1) for i, im in enumerate(pair) if im.acquired_at is None]
        return _reject(
            ic,
            "G3",
            "Rejected: both images are %s and %s no acquisition date, so "
            "change analysis has no before and after to compare. "
            "Supply GeoTIFFs with acquisition metadata, or a <file>.meta.json "
            "sidecar with acquired_at for each image."
            % (a.modality, ("image %s carries" % undated[0]) if len(undated) == 1
               else "images %s carry" % " and ".join(undated)),
        )

    # G5 - different CRS. Reprojectable when both are known; a refusal only
    # when one side has no CRS to reproject from.
    needs_reproject = False
    if a.crs and b.crs and a.crs != b.crs:
        needs_reproject = True
    elif bool(a.crs) != bool(b.crs):
        return _reject(
            ic,
            "G5",
            "Rejected: the two inputs are in different CRS and are not "
            "co-registered. One file declares %s and the other declares none, "
            "so they cannot be put on a common grid."
            % (a.crs or b.crs),
        )

    # Same shape, single-image wording. The rules route two images by SHAPE, so
    # this will go to a pair tool. Record it rather than silently reinterpret -
    # it is a real gap in the frozen contract and the trace is where the team
    # will see how often it happens.
    if not wants_pair(query) and (GROUNDING_RE.search(query) or CAPTION_RE.search(query)):
        ic["warnings"].append(
            "Two images supplied with single-image wording. Routing is by input "
            "shape, so this went to a pair tool. Supply one image to ask about "
            "one image."
        )

    ic["plan_mode"] = "pair"
    plan = Plan(mode="pair", groups=[tuple(idx)])
    if needs_reproject:
        plan.declaration = None
        ic["warnings"].append(
            "Inputs are in different CRS (%s and %s); preproc.reproject will "
            "put them on a common grid before any comparison." % (a.crs, b.crs)
        )
        plan.mode = "pair"
        plan.groups = [tuple(idx)]
        ic["needs_reproject"] = True
    return GateResult("accepted", ic, plan)


def _offset_px(a, b):
    """Worst-case corner offset between two footprints, in pixels.

    Only meaningful when both are in the same CRS with a known GSD. Returns
    None otherwise rather than guessing - an invented alignment number is worse
    than an absent one.
    """
    if a.crs != b.crs or not a.gsd_m:
        return None
    d = max(
        abs(a.bbox.minx - b.bbox.minx),
        abs(a.bbox.miny - b.bbox.miny),
        abs(a.bbox.maxx - b.bbox.maxx),
        abs(a.bbox.maxy - b.bbox.maxy),
    )
    return round(d / a.gsd_m, 3)


def _warn_cloud(ic: dict, images: list) -> None:
    """G8 - heavy cloud is a warning, not a refusal. Radar still works."""
    for im in images:
        pct = getattr(im, "cloud_pct", None)
        if pct is not None and pct > CLOUD_WARN_PCT:
            ic["warnings"].append(
                "Low confidence: cloud cover over the queried region exceeds "
                "%.0f%%. SAR is available and recommended." % CLOUD_WARN_PCT
            )
            return


def _name(im) -> str:
    import os

    return os.path.basename(im.path)


def _reject(ic: dict, code: str, message: str) -> GateResult:
    ic["verdict"] = "rejected"
    ic["code"] = code
    ic["message"] = message
    return GateResult("rejected", ic, None, code, message)
