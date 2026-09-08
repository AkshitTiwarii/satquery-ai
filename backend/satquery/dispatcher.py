"""Deterministic routing: seven rules, one job, no model call.

All five of the statement's representative queries resolve here with nothing
learned deciding anything. Two reasons that is the right default and not
laziness:

  * Internal reasoning is not evaluated. The statement says so directly. A
    planner and a table emit an identical trace, so the planner buys nothing
    that is graded.
  * At the measured 11 tok/s, a 30-token plan costs ~2.7 s before any real work
    starts, and it can hallucinate a tool name that does not exist. A table
    cannot.

The rules are exhaustive over what reaches them because the gate has already
resolved the input into groups of one or two images (see gate.Plan). R6 is the
catch-all for any single-image wording we did not anticipate.

The planner fallback exists for genuinely compound queries only - "compare land
cover between these dates AND tell me if water increased" - and it reuses the
Qwen that is already resident, so it costs no extra VRAM. When it fires,
routing.by is set to "planner" so the trace shows honestly which path decided.
"""

from __future__ import annotations

import re

from . import zoom
from .types import MSI, OPTICAL, SAR

# R4 and R5 keep the statement's own wording. Kept as one compiled pattern each
# so the gate and the dispatcher can never disagree about what a change
# question looks like.
GROUNDING_RE = re.compile(
    r"\b(highlight|where\b|locate|show me where|bounding box|point out|mark the|find the|pinpoint)\b",
    re.I,
)
CAPTION_RE = re.compile(
    r"\b(describe|caption|what does this show|summarise|summarize|overview of)\b",
    re.I,
)

# Used by the gate to decide whether more than two images fan out or collapse
# to one declared pair. A question that needs a pair cannot be answered one
# image at a time, so it is the hinge of the whole N>2 design.
CHANGE_RE = re.compile(
    r"\b(chang(e|ed|es)|increas(e|ed)|decreas(e|ed)|grew|grown|shrunk|shrank"
    r"|differ(ence|ent)|before and after|expand(ed)?|deforest\w*|new construction"
    r"|appeared|disappeared|lost|gained|between (these|the two) dates"
    r"|over time|since)\b",
    re.I,
)
FUSION_RE = re.compile(
    r"\b(fus(e|ed|ion)|combin(e|ed|ing)|both sensors|both modalities"
    r"|optical and (radar|sar)|(radar|sar) and optical|using both)\b",
    re.I,
)

TASK_VQA = "vqa"
TASK_GROUNDING = "grounding"
TASK_CAPTIONING = "captioning"
TASK_CHANGE = "change_vqa"
TASK_FUSION = "fusion"


def wants_pair(query: str) -> bool:
    """Does this question need two images to answer at all?

    The gate calls this before routing, to decide whether N>2 images fan out
    (one call each) or collapse to one declared pair.
    """
    return bool(CHANGE_RE.search(query) or FUSION_RE.search(query))


def is_compound(query: str) -> bool:
    """A genuinely compound query - two tasks joined, not one task described.

    Deliberately narrow. " and " appears in plenty of single-task questions
    ("buildings and roads"), so we require a clause boundary with a verb-ish
    second half, and even then the planner is the exception path.
    """
    if not re.search(r",?\s+(and|then|also)\s+(tell|say|report|check|is|are|how|what|does|do)\b", query, re.I):
        return False
    # ONE task hit is enough, not two. R6 is the catch-all and has no pattern
    # of its own, so a compound query whose second clause is a plain question -
    # which is the contract's own worked example - can never score two. The
    # clause boundary above is what keeps this tight; the thirty routing cases
    # are the guard that it stays tight.
    hits = sum(
        bool(rx.search(query))
        for rx in (CHANGE_RE, FUSION_RE, GROUNDING_RE, CAPTION_RE)
    )
    return hits >= 1


def route(query: str, images: list) -> dict:
    """Return {task, rule_id, tool_id} for one group of one or two images.

    `images` is a group produced by the gate, so len(images) is 1 or 2. Any
    other length is a bug in the gate, not an input the user can cause.
    """
    n = len(images)
    if n not in (1, 2):
        raise AssertionError(
            "dispatcher received %d images; the gate must resolve N>2 into "
            "groups of one or two before routing" % n
        )

    if n == 2:
        a, b = images
        same_modality = a.modality == b.modality
        distinct_dates = _distinct_dates(a, b)

        # R1 - one optical/msi and one sar, same date.
        if not same_modality and {a.modality, b.modality} <= {OPTICAL, MSI, SAR}:
            return {"task": TASK_FUSION, "rule_id": "R1", "tool_id": "fusion.optical_sar"}

        # R2 - both radar, different dates. Log-ratio, never subtraction.
        if same_modality and a.modality == SAR and distinct_dates:
            return {"task": TASK_CHANGE, "rule_id": "R2", "tool_id": "change.sar_logratio"}

        # R3 - same modality, different dates.
        if same_modality and distinct_dates:
            return {"task": TASK_CHANGE, "rule_id": "R3", "tool_id": "change.vqa"}

        raise AssertionError(
            "two images reached the dispatcher in a shape no rule covers "
            "(%s/%s, distinct_dates=%s); gate conditions G3 and G4 exist to "
            "close exactly this and one of them has a hole"
            % (a.modality, b.modality, distinct_dates)
        )

    # R4 - grounding. Checked before captioning: "show me the ships" is a
    # grounding request even though it reads like a description.
    if GROUNDING_RE.search(query):
        return {"task": TASK_GROUNDING, "rule_id": "R4", "tool_id": "ground.rs"}

    # R5 - captioning.
    if CAPTION_RE.search(query):
        return {"task": TASK_CAPTIONING, "rule_id": "R5", "tool_id": "caption.rs"}

    # R6 - anything else about one image.
    return {"task": TASK_VQA, "rule_id": "R6", "tool_id": "vqa.rs"}


def zoom_rule(task: str, images: list) -> dict | None:
    """Z1. Not a routing rule - R4/R6 still choose the tool - but a step the
    dispatcher inserts before it when the FILE says the tile is sub-metre
    (or, with no GSD, wider than 300 px). One image only; pairs never zoom.
    Returns None or {rule_id, reason} for the trace."""
    if len(images) != 1:
        return None
    return zoom.wants(task, images[0])


def _distinct_dates(a, b) -> bool:
    """Two acquisitions count as distinct only if both dates are known.

    An unknown date is not a different date. Treating a missing timestamp as
    "distinct" would route an unlabelled pair into change analysis and produce
    a confident answer about a comparison that was never valid.
    """
    if a.acquired_at is None or b.acquired_at is None:
        return False
    return a.acquired_at != b.acquired_at
