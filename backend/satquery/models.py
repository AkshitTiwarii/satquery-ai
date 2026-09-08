"""The real Qwen backend, behind the same interface the stubs use.

One model in memory, three registry rows. `vqa.rs`, `ground.rs` and
`caption.rs` are the same weights with different prompts (and, later, different
LoRA adapters), which is why residency is keyed on weights and not on row id.

Backend selection, via SATQUERY_BACKEND:

    auto   (default)  real if torch, CUDA, transformers and the weights are all
                      present; stub otherwise
    real              force real, and fail loudly if it cannot load
    stub               force stub, even on a machine that could run the model

The point of "auto" is that the same command runs on a laptop with no GPU and
on a T4, and the trace says which happened - `stub` is recorded per step, so a
run can never quietly present a canned answer as a measured one.

THE COORDINATE TRAP. Qwen2.5-VL emits grounding boxes in ABSOLUTE PIXELS of the
image AFTER the processor's own resize, not of the file you handed it, and not
normalised. BigEarthNet.txt normalises 0-1 and VRSBench 0-100. Three
conventions, no error raised if you mix them, and a box that is wrong by 100x
still plots - just in the wrong place. Everything here converts to NormBox
(0-1) at the boundary and asserts the range, and the trace records which
denominator was used so a wrong box is diagnosable rather than mysterious.
"""

from __future__ import annotations

import json
import os
import re
import sys

# One visual token covers a 28x28 pixel block: patch_size 14, merge_size 2.
# So a cap of N tokens is a cap of N * 28 * 28 pixels, which is the knob
# Akshit's study varies.
PIXELS_PER_TOKEN = 28 * 28

DEFAULT_MODEL = os.environ.get("SATQUERY_QWEN", "Qwen/Qwen2.5-VL-3B-Instruct")
DEFAULT_VISUAL_TOKENS = int(os.environ.get("SATQUERY_VISUAL_TOKENS", "256"))

# How the weights are quantised: nf4 (default, ~2.5 GB, what every recorded
# number was measured on), int8 (~4 GB), or fp16 (7.51 GB, too large for the
# 8 GB demo card). Quantisation is not free - the joint adapter reads 84.5 on
# RSVQA-LR in nf4 and 87.5 in fp16 on the same questions - so the precision is
# part of the result and goes into the trace beside the adapter name.
DTYPE = os.environ.get("SATQUERY_DTYPE", "nf4")
if DTYPE not in ("nf4", "int8", "fp16"):
    raise ValueError(
        f"SATQUERY_DTYPE must be nf4, int8 or fp16, got {DTYPE!r}")


def dtype_name() -> str:
    """The precision the weights are actually served at, for the trace."""
    return DTYPE if _STATE.get("model") is not None else ""

# LoRA adapters, one per registry row that has been fine-tuned, all resident
# on the ONE copy of the weights (peft holds several and switches by name;
# each is ~60 MB). Unset means base weights for that row:
#
#   SATQUERY_VQA_ADAPTER     vqa.rs      base 60.75% -> 85.5% on RSVQA-LR (PLAN §9)
#   SATQUERY_CHANGE_ADAPTER  change.vqa  base 40.0% vs 51.5% majority on CDVQA
#
# Rows without an adapter of their own run with every adapter DISABLED:
# ground.rs and caption.rs were never trained on boxes or prose and would
# have to be re-measured before an adapter is allowed near them.
ADAPTER_DIRS = {
    name: d for name, d in (
        ("vqa", os.environ.get("SATQUERY_VQA_ADAPTER")),
        ("change", os.environ.get("SATQUERY_CHANGE_ADAPTER")),
        ("ground", os.environ.get("SATQUERY_GROUND_ADAPTER")),
        ("fusion", os.environ.get("SATQUERY_FUSION_ADAPTER")),
    ) if d
}

_STATE = {"model": None, "processor": None, "why_not": None, "adapters": {}}


def adapter_name(which: str = None) -> str:
    """Which adapter answers for `which` ('vqa', 'change'), for the trace.
    Empty string means base weights. No argument: the first loaded, or ''."""
    ads = _STATE["adapters"]
    if which is None:
        d = next(iter(ads.values()), None)
    else:
        d = ads.get(which)
    return os.path.basename(os.path.normpath(d)) if d else ""


def backend() -> str:
    return os.environ.get("SATQUERY_BACKEND", "auto").lower()


def available() -> bool:
    """Can the real model run here? Cached, because the answer cannot change."""
    if backend() == "stub":
        return False
    if _STATE["model"] is not None:
        return True
    if _STATE["why_not"] is not None:
        return False
    try:
        _load()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "use stubs"
        _STATE["why_not"] = "%s: %s" % (type(exc).__name__, exc)
        if backend() == "real":
            raise
        return False


def why_not() -> str:
    return _STATE["why_not"] or ""


def _load():
    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")

    # The harnesses' loader, imported rather than reimplemented. Four copies of
    # this branch is how the engine came to serve nf4 at fp16 compute while the
    # harnesses used bf16 - a difference nobody chose. scripts/ is not a
    # package, so it goes on the path here.
    _scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from qwen_loader import load_model

    model = load_model(DEFAULT_MODEL, DTYPE, device="auto")
    for name, d in ADAPTER_DIRS.items():
        # Loud on purpose. A configured adapter that quietly fails to load
        # would present base-model answers under the fine-tuned name, and the
        # trace would say the wrong weights answered.
        if not os.path.isfile(os.path.join(d, "adapter_config.json")):
            raise FileNotFoundError(
                "SATQUERY_%s_ADAPTER=%r has no adapter_config.json" % (name.upper(), d))
        if not _STATE["adapters"]:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, d, adapter_name=name)
        else:
            model.load_adapter(d, adapter_name=name)
        _STATE["adapters"][name] = d
    model.eval()
    proc = AutoProcessor.from_pretrained(
        DEFAULT_MODEL,
        min_pixels=64 * PIXELS_PER_TOKEN,
        max_pixels=DEFAULT_VISUAL_TOKENS * PIXELS_PER_TOKEN,
    )
    _STATE["model"], _STATE["processor"] = model, proc
    return model, proc


def set_visual_tokens(n: int) -> None:
    """Change the visual-token cap WITHOUT reloading the model.

    The cap is a processor setting - it decides how many 28x28 blocks an image
    is cut into - not a model setting. Reloading the module to change it puts a
    second 2.5 GB copy of the weights on the card before the first is
    collected, and sweeping 128/256/512/1024 that way OOMs a T4 on the third
    step. The model stays; only the processor is rebuilt.
    """
    from transformers import AutoProcessor

    if _STATE["model"] is None:
        _load()
    _STATE["processor"] = AutoProcessor.from_pretrained(
        DEFAULT_MODEL,
        min_pixels=64 * PIXELS_PER_TOKEN,
        max_pixels=n * PIXELS_PER_TOKEN,
    )


def generate(pil_images: list, prompt: str, max_new_tokens: int = 64,
             adapter: str = None) -> dict:
    """Run the model on one or more images and return text plus the geometry.

    The returned `resized_wh` is the size the processor actually fed the model,
    which is the denominator any coordinate in the answer is expressed in.

    `adapter` names which resident LoRA answers ('vqa', 'change'); None, or a
    name that is not loaded, means base weights with every adapter disabled.
    peft switches by name and disable_adapter() is a context, not a reload,
    so this costs nothing and the one copy of the weights stays put.
    """
    import contextlib

    import torch

    model, proc = _STATE["model"], _STATE["processor"]
    if model is None:
        model, proc = _load()

    content = [{"type": "image", "image": im} for im in pil_images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=pil_images, return_tensors="pt").to(model.device)

    scope = contextlib.nullcontext()
    if _STATE["adapters"]:
        if adapter in _STATE["adapters"]:
            model.set_adapter(adapter)
        else:
            scope = model.disable_adapter()

    with torch.inference_mode(), scope:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy: --replay has to reproduce this
            temperature=None,
            top_p=None,
            top_k=None,
        )
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    answer = proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    return {
        "text": answer,
        "resized_wh": _resized_wh(inputs),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(trimmed.shape[1]),
    }


def _resized_wh(inputs) -> tuple:
    """The pixel size the processor resized to, per image.

    image_grid_thw is in patch units, so multiply by the patch size. This is
    the denominator for any box the model returns; using the file's own size
    instead is wrong by whatever the resize factor happened to be, which
    changes per image and therefore looks like a model that is bad at
    localisation rather than a bug.
    """
    grid = inputs.get("image_grid_thw")
    if grid is None:
        return None
    patch = 14
    try:
        patch = _STATE["processor"].image_processor.patch_size
    except Exception:  # noqa: BLE001
        pass
    out = []
    for row in grid.tolist():
        _, h, w = row
        out.append((w * patch, h * patch))
    return out[0] if len(out) == 1 else out


# --- grounding: parse a box out of whatever the model wrote -----------------

_JSON_BOX = re.compile(r'"bbox(?:_2d)?"\s*:\s*\[([^\]]+)\]')
_BARE_BOX = re.compile(r"\[\s*([0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+)\s*\]")


# BigEarthNet.txt writes a box as "[x0 y0, x1 y1]" - a space inside each
# corner, a comma between corners, normalised 0-1. The grounding adapter is
# trained on that string verbatim, so the engine has to read it back.
_BEN_BOX = re.compile(r"\[\s*(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s*\]")


def parse_box(text: str, resized_wh, original_wh):
    """Pull a box out of the model's answer and normalise it to 0-1.

    Returns (NormBox, denominator_used) or (None, reason). Never guesses: if
    the numbers do not fit either candidate frame, it says so rather than
    scaling something into range and producing a plausible wrong answer.
    """
    from .types import NormBox

    nums = None
    m = _BEN_BOX.search(text)
    if m:
        nums = ",".join(m.groups())
    if not nums and (m := _JSON_BOX.search(text)):
        nums = m.group(1)
    if not nums:
        try:
            blob = json.loads(text[text.index("{"): text.rindex("}") + 1])
            for k in ("bbox_2d", "bbox", "box"):
                if isinstance(blob, dict) and k in blob:
                    nums = ",".join(str(v) for v in blob[k])
                    break
        except Exception:  # noqa: BLE001
            m = _BARE_BOX.search(text)
            if m:
                nums = m.group(1)
    if not nums:
        return None, "no box in the answer"

    try:
        x0, y0, x1, y1 = [float(v.strip()) for v in nums.split(",")]
    except ValueError:
        return None, "box did not parse as four numbers"

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if (x1 - x0) <= 0 or (y1 - y0) <= 0:
        # Zero area in any frame. Plotting it would draw a point and call it a
        # region; abstaining is the honest answer.
        return None, "degenerate box (zero area)"

    # Already normalised? A real 0-1 box has non-zero area (checked above), so
    # anything that fits in the unit square is taken as such.
    if max(x1, y1) <= 1.0:
        return NormBox(x0, y0, x1, y1), "already_0_1"

    for name, wh in (("resized", resized_wh), ("original", original_wh)):
        if not wh:
            continue
        w, h = wh
        if x1 <= w and y1 <= h and w and h:
            try:
                return NormBox(x0 / w, y0 / h, x1 / w, y1 / h), name
            except ValueError:
                continue

    return None, (
        "box %s fits neither the resized frame %s nor the original %s - "
        "refusing to scale it into range"
        % ([x0, y0, x1, y1], resized_wh, original_wh)
    )


# --- prompts -----------------------------------------------------------------
# Kept here rather than inline so the eval harness and the served tool ask the
# model the same question. A prompt difference between the two is the quietest
# possible way to make a benchmark number meaningless.

VQA_PROMPT = (
    "You are looking at a satellite image. Answer the question in as few words "
    "as possible - 'yes', 'no', a number, or a single noun phrase. Do not "
    "explain.\n\nQuestion: {query}"
)

# The four instruction templates the vqa.rs adapter was TRAINED on, verbatim
# from scripts/eval_rsvqa_lr.py (a test pins the two copies equal; the eval
# script cannot import this package on Kaggle, so it keeps its own). An
# adapter that learned "Answer with one word: yes or no." and is then asked
# the same question in VQA_PROMPT's wording is being scored on a format it
# never saw - the 85.5% was measured with THESE strings.
RSVQA_PROMPTS = {
    "presence":    "{q}? Answer with one word: yes or no.",
    "comp":        "{q}? Answer with one word: yes or no.",
    "rural_urban": "{q}? Answer with one word: rural or urban.",
    "count":       "{q}? Answer with a single number and nothing else.",
}

# An adapter cut with make_rsvqa_subset.py --count-answer bucket answers a
# counting question with the published bucket ("11-100") and was trained under
# this instruction instead, because "a single number and nothing else" would be
# an instruction it has to disobey. The engine must ask the way the LIVE
# adapter was taught, so the form is an env var set beside the adapter path in
# scripts/demo_env.ps1 rather than guessed from the adapter directory. Both
# strings are pinned equal to the harness's copies by tests/test_vqa_prompts.py.
RSVQA_COUNT_BUCKET_PROMPT = "{q}? Answer with one of: 0, 1-10, 11-100, 101-1000, >1000."
COUNT_FORM = os.environ.get("SATQUERY_COUNT_FORM", "number")
if COUNT_FORM not in ("number", "bucket"):
    raise ValueError("SATQUERY_COUNT_FORM must be 'number' or 'bucket', "
                     f"got {COUNT_FORM!r}")


def rsvqa_templates() -> dict:
    """The four templates in force, with counting posed for the live adapter."""
    t = dict(RSVQA_PROMPTS)
    if COUNT_FORM == "bucket":
        t["count"] = RSVQA_COUNT_BUCKET_PROMPT
    return t


_COUNT_Q = re.compile(r"^\s*(how many|how much|what is the (number|count|amount) of|"
                      r"what number of|count the)\b", re.I)
_RURAL_URBAN_Q = re.compile(r"\b(rural|urban)\b", re.I)
_YESNO_Q = re.compile(r"^\s*(is|are|does|do|has|have|was|were|can|could|will|would|"
                      r"should|did|isn't|aren't|doesn't|don't)\b", re.I)


def vqa_prompt(query: str) -> tuple:
    """Pick the prompt for vqa.rs: a trained template when the question has
    one of the four RSVQA shapes, the generic VQA_PROMPT otherwise.

    Returns (prompt, kind), and the kind goes in the trace so a wrong answer
    can be read as "asked in the wrong format" rather than "the model is bad".
    """
    q = query.strip().rstrip("?").strip()
    if _COUNT_Q.search(q):
        kind = "count"
    elif _RURAL_URBAN_Q.search(q):
        kind = "rural_urban"
    elif _YESNO_Q.search(q):
        kind = "presence"   # presence and comp share one template
    else:
        return VQA_PROMPT.format(query=query), "open"
    return rsvqa_templates()[kind].format(q=q), kind


# The change adapter's templates, verbatim from scripts/eval_cdvqa.py (pinned
# equal by a test, same reason as RSVQA_PROMPTS above). CDVQA answers are
# dataset tokens - 'NVG_surface', '10_to_20' - so the template has to name
# them or the model answers in English and is marked wrong on a technicality.
CDVQA_CLASSES = ["NVG_surface", "buildings", "low_vegetation", "trees", "water", "playgrounds"]
CDVQA_RATIOS = ["0", "0_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_50",
                "50_to_60", "60_to_70", "70_to_80", "80_to_90", "90_to_100"]
CDVQA_PROMPTS = {
    "change_or_not": "{q} Answer with one word: yes or no.",
    "increase_or_not": "{q} Answer with one word: yes or no.",
    "decrease_or_not": "{q} Answer with one word: yes or no.",
    "change_to_what": "{q} Answer with one of: " + ", ".join(CDVQA_CLASSES) + ".",
    "smallest_change": "{q} Answer with one of: " + ", ".join(CDVQA_CLASSES) + ".",
    "largest_change": "{q} Answer with one of: " + ", ".join(CDVQA_CLASSES) + ".",
    "change_ratio": "{q} Answer with one of: " + ", ".join(CDVQA_RATIOS) + ".",
    "change_ratio_types": "{q} Answer with one of: " + ", ".join(CDVQA_RATIOS) + ".",
}
CDVQA_PAIR_PREFIX = ("The first image is the pre-change image and the second is the "
                     "post-change image of the same place. ")

_RATIO_Q = re.compile(r"\b(percentage|percent|fraction|how much of|what proportion|ratio)\b", re.I)
_WHAT_CHANGE_Q = re.compile(r"\b(what (type of )?change|largest change|smallest change|"
                            r"biggest change|changed to|change(d)? into|mainly changed)\b", re.I)
_CHANGE_YESNO_Q = re.compile(r"^\s*(did|do|does|has|have|is|are|was|were)\b.*\b(change|changed|"
                             r"increase|increased|decrease|decreased|grow|grown|shrink|shrunk|"
                             r"appear|appeared|disappear|disappeared|new|removed|built|demolished)\b", re.I)


def change_prompt(query: str) -> tuple:
    """Prompt for change.rs: a trained CDVQA template when the question has one
    of its shapes (ratio / which-class / yes-no about change), CHANGE_PROMPT
    otherwise. Returns (prompt, kind); the kind goes in the trace."""
    q = query.strip()
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"
    if _RATIO_Q.search(q):
        kind = "change_ratio"
    elif _WHAT_CHANGE_Q.search(q):
        kind = "change_to_what"
    elif _CHANGE_YESNO_Q.search(q):
        kind = "change_or_not"
    else:
        return CHANGE_PROMPT.format(query=query), "open"
    return CDVQA_PAIR_PREFIX + CDVQA_PROMPTS[kind].format(q=q), kind


# The grounding adapter is trained on BigEarthNet.txt's referring-expression
# rows, whose target sits in <ref>...</ref> and whose answer is the "[x0 y0,
# x1 y1]" string that parse_box reads. Verbatim from scripts/eval_ground.py
# (pinned by a test). The engine's query is a question, not a noun phrase,
# so the leading "where is / find / locate" and the trailing "?" come off.
GROUND_PROMPT_SUFFIX = " Answer with the box as [x0 y0, x1 y1], normalised 0-1."
_GROUND_LEAD = re.compile(r"^\s*(where\s+(is|are)\s+|locate\s+|find\s+|show\s+me\s+|"
                          r"identify\s+(the\s+location\s+of\s+)?|point\s+(out|to)\s+)", re.I)


def ground_prompt(query: str) -> tuple:
    """Prompt for ground.rs when the grounding adapter is resident: the
    referring-expression template it was trained on. Returns (prompt, target)."""
    q = query.strip().rstrip("?.! ").strip()
    target = _GROUND_LEAD.sub("", q).strip() or q
    return ("Identify the location of the <ref>%s</ref>." % target) + GROUND_PROMPT_SUFFIX, target


# The fusion adapter's form, verbatim from scripts/eval_ben_vqa.py (pinned by
# a test): BigEarthNet.txt's own binary / multiple-choice templates, behind a
# prefix that says which image is optical and which is radar. The SAR render
# (VV, VH, VV-VH in dB on FIXED windows) is duplicated from
# scripts/ben_images.py for the same reason and pinned the same way.
BEN_VQA_PROMPTS = {
    "binary": "{q} Answer with one word: yes or no.",
    "mcq": "{q} Answer with the letter only: a, b, c or d.",
}
SAR_PREFIX = ("The first image is optical and the second is radar (SAR, VV/VH) of the "
              "same place on the same date. ")
SAR_DB_WINDOW = {"VV": (-25.0, 0.0), "VH": (-32.0, -5.0), "RATIO": (0.0, 20.0)}
_MCQ_Q = re.compile(r"\ba\)\s*.+\bb\)\s*", re.I | re.S)


def fusion_prompt(query: str) -> tuple:
    """Prompt for fusion.optical_sar when the fusion adapter is resident.
    Multiple-choice if the query carries a) b) options, else yes/no.
    Returns (prompt, kind)."""
    q = query.strip()
    kind = "mcq" if _MCQ_Q.search(q) else "binary"
    return SAR_PREFIX + BEN_VQA_PROMPTS[kind].format(q=q), kind


def sar_to_rgb(vv, vh, size=None):
    """VV/VH arrays (dB, or linear which is converted) -> 8-bit RGB PIL
    (VV, VH, VV-VH) on the fixed dB windows. Same maths as the harness."""
    import numpy as np
    from PIL import Image

    vv = np.asarray(vv, dtype="float32")
    vh = np.asarray(vh, dtype="float32")
    if vv.ndim == 3:
        vv = vv[0]
    if vh.ndim == 3:
        vh = vh[0]
    if vv.min() >= 0 and vh.min() >= 0:
        vv = 10 * np.log10(np.clip(vv, 1e-6, None))
        vh = 10 * np.log10(np.clip(vh, 1e-6, None))
    chans = []
    for arr, key in ((vv, "VV"), (vh, "VH"), (vv - vh, "RATIO")):
        lo, hi = SAR_DB_WINDOW[key]
        chans.append(np.clip((arr - lo) / (hi - lo), 0, 1))
    img = Image.fromarray((np.dstack(chans) * 255).astype("uint8"), "RGB")
    return img.resize((size, size), Image.BILINEAR) if size else img


def open_sar_rgb(meta):
    """A SAR ImageMeta (2-band VV/VH GeoTIFF) -> the fusion adapter's render.
    One band is replicated as both polarisations, with a warning-grade loss
    of the ratio channel, rather than refusing the file."""
    import numpy as np
    import rasterio

    with rasterio.open(meta.path) as src:
        vv = src.read(1).astype("float32")
        vh = src.read(2).astype("float32") if src.count >= 2 else vv
    return sar_to_rgb(vv, vh)


GROUND_PROMPT = (
    "Locate what the question asks about in this satellite image and output "
    'its bounding box as JSON: {{"bbox_2d": [x1, y1, x2, y2]}}. Output only '
    "the JSON.\n\nQuestion: {query}"
)

CAPTION_PROMPT = (
    "Describe this satellite image in two sentences. Name the land-cover types "
    "and any built structures you can see.\n\nQuestion: {query}"
)

CHANGE_PROMPT = (
    "These are two satellite images of the same place, the first earlier and "
    "the second later. Answer the question about what changed between them, in "
    "as few words as possible.\n\nQuestion: {query}"
)

FUSION_PROMPT = (
    "These are two images of the same place on the same date: one optical and "
    "one radar. Use both to answer, in as few words as possible.\n\n"
    "Question: {query}"
)


def open_rgb(meta, tokens: int = None):
    """Load one ImageMeta as a 3-channel RGB PIL image for Qwen.

    The conversion Qwen needs and DOFA must not get. A 12-band reBEN array has
    to have bands 4/3/2 picked out, percentile-stretched and cast to 8-bit
    before Qwen sees anything; skip it and you either crash or - far worse -
    train and infer happily on black images. DOFA wants the opposite: all 12
    bands, raw, unstretched. Two loaders off one file, and this is the Qwen one.
    """
    from PIL import Image

    if meta.fmt in ("PNG", "JPEG"):
        return Image.open(meta.path).convert("RGB")

    try:
        import numpy as np
        import rasterio
    except ImportError:
        return Image.open(meta.path).convert("RGB")

    with rasterio.open(meta.path) as src:
        if src.count >= 4:
            # Sentinel-2 band order: B04 red, B03 green, B02 blue.
            idx = (4, 3, 2)
        elif src.count == 3:
            idx = (1, 2, 3)
        else:
            idx = (1, 1, 1)  # radar: VV replicated, only so Qwen has 3 channels
        bands = [src.read(i).astype("float32") for i in idx]

    out = []
    for b in bands:
        lo, hi = np.percentile(b, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        out.append(np.clip((b - lo) / (hi - lo), 0, 1))
    arr = (np.dstack(out) * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")
