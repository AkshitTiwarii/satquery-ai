"""Score a Qwen2.5-VL model on VRSBench - the third split the statement names.

    python scripts/eval_vrsbench.py --task vqa   --limit 400
    python scripts/eval_vrsbench.py --task refer --limit 400 --adapter F:\\sih\\runs\\lora_ground_ref_vis

VRSBench (arXiv:2406.12384) is 29,614 human-verified aerial images at 512x512
with captions, VQA and referring expressions. We hold no number on it at all,
and it is prescribed for grading, so this is the gap this file closes.

WHAT MAKES IT DIFFERENT FROM OUR OTHER HARNESSES
------------------------------------------------
1. The imagery is sub-metre. RSVQA-LR and BigEarthNet are 10 m. This is the
   scale gap the ISRO set will also have, so a zero-shot number here is the
   closest thing we have to a rehearsal for the hidden set.

2. VQA answers are open vocabulary. 37,409 questions over twelve types, and
   `object position` alone has 710 distinct gold answers. Exact match after
   normalisation is therefore harsh, so a `contains` column is reported beside
   it as an upper bound. Quote exact; show contains only to say how much of the
   gap is wording rather than perception.

3. Boxes are 0-100 INTEGERS in braces - "{<25><40><33><60>}" - not the 0-1
   decimals BigEarthNet.txt uses and not the 0-1000 some Qwen builds emit.
   Getting that wrong scores every box as a miss while looking like a model
   problem, so parse_gold refuses anything above 100 rather than guessing.

4. Two answer formats, because two different adapters can be asked. Our
   grounding adapter learned BigEarthNet's "[x0 y0, x1 y1]" string, so
   --answer-format ben asks in the wording it was trained on and converts the
   gold to match; a future VRSBench-trained adapter would use the dataset's own
   brace format. Asking in a format the adapter never saw measures the prompt,
   not the model.

WHAT THE PUBLISHED NUMBERS ARE
------------------------------
VQA, fine-tuned on VRSBench train (paper Table 5): LLaVA-1.5 60.9, GeoChat
60.6, Mini-Gemini 60.2, MiniGPT-v2 37.1; GeoChat zero-shot 40.8.
Referring, Acc@0.5 over all objects (Table 4): GeoChat 39.6, LLaVA-1.5 36.3,
MiniGPT-v2 35.8, Mini-Gemini 29.0. GeoGround reports 75.93 on VRSBench-Ref.
Print them beside our own so nobody has to look them up mid-review.
"""

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _zoom_deps():
    """--zoom uses the ENGINE's crop maths and proposer prompt, imported, so the number
    this harness produces is a measurement of what satquery.run does, not of a copy.
    Imported lazily: a training kernel embeds scripts/ alone and never passes --zoom."""
    from satquery import zoom as zoomlib
    from satquery.models import GROUND_PROMPT as proposer_prompt
    from satquery.models import parse_box as parse_any_box
    return zoomlib, proposer_prompt, parse_any_box

# The referring geometry is identical to BigEarthNet's, so it is imported
# rather than rewritten - one definition of IoU across both harnesses.
from eval_ground import PROMPT_SUFFIX as BEN_SUFFIX  # noqa: E402
from eval_ground import iou as box_iou  # noqa: E402
from eval_ground import parse as parse_ben_box  # noqa: E402

# --- prompts ---------------------------------------------------------------
# VRSBench trains on "<image>\n[vqa] {question}" with the answer a bare short
# phrase, so the instruction here only has to pin the LENGTH: without it the
# base model writes a sentence and the normaliser has to guess where the answer
# ends. This is the standard short-answer wording and is what the published
# numbers were produced under.
VQA_PROMPT = "{q} Answer the question using a single word or phrase."
VQA_MAX_NEW = 12

# Two ways to ask for a box. `ben` is the string our grounding adapter learned;
# `vrsbench` is the dataset's own, for an adapter trained on it.
REF_PROMPTS = {
    "ben": "Identify the location of <ref>{q}</ref>." + BEN_SUFFIX,
    "vrsbench": "Give me the location of <p>{q}</p>. "
                "Answer with the box as {{<x1><y1><x2><y2>}}, integers 0-100.",
}
REF_MAX_NEW = 32

PUBLISHED = {
    "vqa": "fine-tuned on VRSBench train: LLaVA-1.5 60.9 | GeoChat 60.6 | "
           "Mini-Gemini 60.2 | MiniGPT-v2 37.1 | GeoChat zero-shot 40.8",
    "refer": "Acc@0.5 all objects: GeoGround 75.9 | GeoChat 39.6 | "
             "LLaVA-1.5 36.3 | MiniGPT-v2 35.8 | Mini-Gemini 29.0",
}

# --- answer normalisation --------------------------------------------------
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s-]")
_WS = re.compile(r"\s+")
WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalise(text):
    """Model text or gold -> a comparable string, or None if empty.

    Lowercase, drop punctuation and articles, collapse whitespace, spell
    numbers as digits. Deliberately mild: 'Yes.' and 'yes' must match, but
    'north-south' and 'south-north' must not.
    """
    if text is None:
        return None
    # Hyphens survive _PUNCT on purpose: "north-south" and "south-north" are
    # different answers and collapsing the hyphen would merge them.
    t = _PUNCT.sub(" ", str(text).lower())
    t = _WS.sub(" ", t).strip()
    if not t:
        return None
    words = [WORD_NUMBERS.get(w, w) for w in t.split() if w not in _ARTICLES]
    return " ".join(words) or None


# --- referring boxes -------------------------------------------------------
BRACE_BOX = re.compile(r"\{?\s*<\s*(\d+)\s*>\s*<\s*(\d+)\s*>\s*"
                       r"<\s*(\d+)\s*>\s*<\s*(\d+)\s*>\s*\}?")


def parse_gold_box(text):
    """VRSBench gold '{<x1><y1><x2><y2>}' -> (x0,y0,x1,y1) normalised 0-1.

    The integers are PERCENTAGES. Reading them as 0-1 decimals, or as the
    0-1000 scale some Qwen builds use, silently turns every gold box into a
    dot in the corner and scores the model at zero - so anything above 100 is
    refused rather than rescaled on a guess.
    """
    m = BRACE_BOX.search(text or "")
    if not m:
        return None
    v = [int(g) for g in m.groups()]
    if max(v) > 100:
        raise ValueError(
            f"VRSBench boxes are 0-100 percentages; got {v} from {text!r}. "
            "Refusing to guess a scale.")
    x0, y0, x1, y1 = (n / 100.0 for n in v)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def parse_pred_box(text, fmt, resized_wh):
    """Model text -> (x0,y0,x1,y1) normalised 0-1, or None."""
    if fmt == "vrsbench":
        m = BRACE_BOX.search(text or "")
        if m:
            v = [int(g) for g in m.groups()]
            if max(v) <= 100:
                x0, y0, x1, y1 = (n / 100.0 for n in v)
                return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), "brace"
        # fall through: a BEN-format answer from a BEN-trained adapter still
        # counts, and saying so in the record is more useful than a miss
    got = parse_ben_box(text, resized_wh)
    if got is None:
        return None
    return got, "ben"


def _ask(model, proc, img, prompt, max_new, base=False):
    """One generation. Returns (text, resized_wh) - the size the processor fed
    the model, which is the denominator of any coordinate in the answer.
    base=True runs with every adapter disabled (only meaningful with one loaded)."""
    import contextlib
    import torch
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    scope = model.disable_adapter() if (base and hasattr(model, "disable_adapter")) else contextlib.nullcontext()
    with torch.inference_mode(), scope:
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0].strip()
    grid = inputs.get("image_grid_thw")
    resized = ((int(grid[0][2]) * 14, int(grid[0][1]) * 14) if grid is not None else img.size)
    return raw, resized


# --- data ------------------------------------------------------------------
def load_rows(ann_dir, task):
    name = "VRSBench_EVAL_vqa.json" if task == "vqa" else "VRSBench_EVAL_referring.json"
    path = os.path.join(ann_dir, name)
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    for r in rows:
        r["type"] = r.get("type") or "unknown"
    return rows


def stratified(rows, limit, key="type"):
    """`limit` rows spread across type AND image.

    The file is ordered by image and each image carries several questions, so
    the first N of a type would come from a handful of pictures. Stride instead,
    exactly as the RSVQA and BigEarthNet harnesses do.
    """
    if not limit or limit >= len(rows):
        return rows
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    per = max(1, limit // len(by))
    out = []
    for k in sorted(by):
        pool = by[k]
        step = max(1, len(pool) // per)
        out.extend(pool[::step][:per])
    return out[:limit]


class ZipImages:
    """Lazy reader over Images_val.zip. 9,350 PNGs at 512x512 is about 7 GB
    decoded, so they are never all resident; a small cache covers the several
    questions that share one image."""

    def __init__(self, zip_path, cache=64):
        self.zip_path = zip_path
        self._zf = None
        self._index = None
        self._cache = {}
        self._order = []
        self._cap = cache

    def _open(self):
        # Opened lazily and per process: a ZipFile handle does not survive a
        # DataLoader fork, and this is also called from worker processes.
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
            self._index = {os.path.basename(n): n
                           for n in self._zf.namelist() if not n.endswith("/")}
        return self._zf

    def __contains__(self, name):
        self._open()
        return name in self._index

    def get(self, name):
        if name in self._cache:
            return self._cache[name]
        zf = self._open()
        member = self._index.get(name)
        if member is None:
            return None
        img = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
        self._cache[name] = img
        self._order.append(name)
        if len(self._order) > self._cap:
            self._cache.pop(self._order.pop(0), None)
        return img


def main():
    from transformers import AutoProcessor

    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir", default=r"F:\sih\data\vrsbench")
    p.add_argument("--images", default=r"F:\sih\data\vrsbench\Images_val.zip")
    p.add_argument("--task", choices=["vqa", "refer"], default="vqa")
    p.add_argument("--answer-format", choices=["ben", "vrsbench"], default="ben",
                   help="refer only: the box wording to ask for. 'ben' is what "
                        "our grounding adapter learned; 'vrsbench' is the "
                        "dataset's own brace format.")
    p.add_argument("--model", default=r"F:\sih\models\Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--limit", type=int, default=400, help="0 = the whole split")
    p.add_argument("--max-pixels", type=int, default=256)
    p.add_argument("--dtype", choices=["nf4", "int8", "fp16"], default="int8")
    p.add_argument("--out", default=r"F:\sih\runs\vrsbench")
    p.add_argument("--zoom", action="store_true",
                   help="rule Z1: propose the region on BASE weights, crop from the "
                        "original pixels, ask the task prompt on the crop, map boxes "
                        "back. VRSBench is 512 px sub-metre, so Z1 fires on every row.")
    args = p.parse_args()

    import torch

    tag = "fine-tuned" if args.adapter else "BASELINE (no adapter)"
    print(f"=== VRSBench {args.task} \u00b7 {tag} ===")
    print(f"  published: {PUBLISHED[args.task]}")

    rows = load_rows(args.ann_dir, args.task)
    print(f"  {len(rows):,} rows over {len({r['image_id'] for r in rows}):,} images")
    counts = Counter(r["type"] for r in rows)
    if args.task == "vqa":
        print("  by type: " + " \u00b7 ".join(f"{t} {counts[t]:,}" for t in sorted(counts)))
    else:
        print("  unique: " + str(dict(Counter(bool(r.get("unique")) for r in rows))))
        print(f"  asking in the {args.answer_format} box format")

    key = "type" if args.task == "vqa" else "unique"
    rows = stratified(rows, args.limit, key=key)
    if args.limit:
        print(f"  limited to {len(rows):,} (stratified across {key})")

    if args.task == "refer":
        # Parse every gold box up front. parse_gold_box raises on an
        # out-of-range scale, and discovering that three hours into a run
        # would throw the run away; it costs milliseconds here.
        bad = 0
        for r in rows:
            try:
                r["_gold_box"] = parse_gold_box(r["ground_truth"])
            except ValueError as exc:
                raise SystemExit(f"gold box out of range: {exc}")
            if r["_gold_box"] is None:
                bad += 1
        if bad:
            raise SystemExit(
                f"{bad} of {len(rows)} gold boxes did not parse. The brace "
                "format changed, or the wrong file was loaded - either way "
                "the score would be meaningless.")
        print(f"  all {len(rows):,} gold boxes parsed, 0-100 scale confirmed")

    images = ZipImages(args.images)
    missing = [r for r in rows if r["image_id"] not in images]
    if missing:
        print(f"  WARNING: {len(missing)} rows have no image in the zip; skipped")
        rows = [r for r in rows if r["image_id"] in images]

    t0 = time.perf_counter()
    from qwen_loader import load_model
    model = load_model(args.model, args.dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  adapter loaded from {args.adapter}")
    model.eval()
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=64 * 28 * 28,
                                         max_pixels=args.max_pixels * 28 * 28)
    print(f"  weights: {args.dtype} \u00b7 ready in {time.perf_counter() - t0:.1f}s, "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB resident")

    # Majority baseline, per type, on the rows actually scored. Every accuracy
    # in this project is quoted with one beside it: 'object existence' is 71.7%
    # 'Yes', so a model that only ever says yes scores 71.7 on that type.
    gold_by_type = defaultdict(Counter)
    for r in rows:
        gold_by_type[r["type"]][normalise(r["ground_truth"])] += 1
    majority = {t: c.most_common(1)[0][0] for t, c in gold_by_type.items()}

    hits, contains_hits, seen, unparsed = Counter(), Counter(), Counter(), Counter()
    zoomed_rows = 0
    ious, acc50, acc70 = defaultdict(list), Counter(), Counter()
    whole_ious, whole50 = defaultdict(list), Counter()
    fmt_seen = Counter()
    samples = []

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sink = open(args.out + ".jsonl", "w", encoding="utf-8")
    t_start = time.perf_counter()

    for i, r in enumerate(rows, 1):
        img = images.get(r["image_id"])
        if img is None:
            continue
        if args.task == "vqa":
            prompt = VQA_PROMPT.format(q=r["question"].strip())
            max_new = VQA_MAX_NEW
        else:
            prompt = REF_PROMPTS[args.answer_format].format(q=r["question"].strip())
            max_new = REF_MAX_NEW

        region, region_how = None, None
        ask_img = img
        if args.zoom:
            zoomlib, PROPOSER_PROMPT, parse_any_box = _zoom_deps()
            # The proposal, on base weights: the grounding adapter learned 120 px
            # 10 m tiles and scores 13.0 here against the base's 37.8, so on this
            # imagery it is the worse proposer (satquery/zoom.py).
            rq = zoomlib.region_query("grounding" if args.task == "refer" else "vqa",
                                      r["question"].strip())
            raw_r, resized_r = _ask(model, proc, img, PROPOSER_PROMPT.format(query=rq),
                                    64, base=bool(args.adapter))
            prop, region_how = parse_any_box(raw_r, resized_r, img.size)
            if prop is not None:
                region = zoomlib.expand(prop, img.width, img.height)
                ask_img = img.crop(zoomlib.crop_pixels(region, img.width, img.height))
        raw, resized = _ask(model, proc, ask_img, prompt, max_new)
        if region is not None:
            zoomed_rows += 1

        grp = r["type"] if args.task == "vqa" else ("unique" if r.get("unique")
                                                    else "non-unique")
        seen[grp] += 1

        if args.task == "vqa":
            pred, gold = normalise(raw), normalise(r["ground_truth"])
            ok = pred is not None and pred == gold
            near = bool(pred and gold and (gold in pred or pred in gold))
            hits[grp] += int(ok)
            contains_hits[grp] += int(near)
            if pred is None:
                unparsed[grp] += 1
            rec = {**{k: r[k] for k in ("image_id", "question", "ground_truth", "type")},
                   "raw": raw, "pred": pred, "correct": ok, "contains": near,
                   "zoomed": region is not None,
                   "crop_region": [region.x0, region.y0, region.x1, region.y1] if region else None,
                   "region_parse": region_how}
        else:
            gold_box = r["_gold_box"]
            got = parse_pred_box(raw, args.answer_format, resized)
            if got is None:
                unparsed[grp] += 1
                v = 0.0
                how = "unparsed"
            else:
                box, how = got
                if region is not None:
                    # answered in the crop's frame; scored in the full image's
                    from satquery.types import NormBox
                    b = zoomlib.map_back(NormBox(*box), region)
                    box = (b.x0, b.y0, b.x1, b.y1)
                v = box_iou(box, gold_box)
            fmt_seen[how] += 1
            ious[grp].append(v)
            acc50[grp] += int(v >= 0.5)
            acc70[grp] += int(v >= 0.7)
            w = box_iou((0.0, 0.0, 1.0, 1.0), gold_box)
            whole_ious[grp].append(w)
            whole50[grp] += int(w >= 0.5)
            rec = {**{k: r[k] for k in ("image_id", "question", "ground_truth", "type")},
                   "unique": bool(r.get("unique")), "size_group": r.get("size_group", ""),
                   "raw": raw, "iou": round(v, 4), "box_format": how,
                   "zoomed": region is not None,
                   "crop_region": [region.x0, region.y0, region.x1, region.y1] if region else None,
                   "region_parse": region_how}

        sink.write(json.dumps(rec) + "\n")
        if len(samples) < 12:
            samples.append((grp, r["question"][:44], str(r["ground_truth"])[:22], raw[:26]))

        if i % 250 == 0 or i == len(rows):
            rate = i / (time.perf_counter() - t_start)
            print(f"  {i}/{len(rows)}  {rate:4.1f} q/s  "
                  f"eta {(len(rows) - i) / rate / 60:5.1f} min", flush=True)
    sink.close()
    elapsed = time.perf_counter() - t_start

    print("\n  --- a few raw outputs, to check the normaliser is not eating answers ---")
    for grp, q, gold, raw in samples:
        print(f"    {grp:<18} gold={gold:<24} raw={raw!r}")

    total = sum(seen.values())
    summary = {"run": tag, "task": args.task, "adapter": args.adapter,
               "dtype": args.dtype, "max_pixels_tokens": args.max_pixels,
               "zoom": bool(args.zoom), "zoomed_rows": zoomed_rows,
               "n": total, "elapsed_min": round(elapsed / 60, 2),
               "published": PUBLISHED[args.task]}

    print(f"\n=== VRSBench {args.task} \u00b7 {tag} ===")
    if args.task == "vqa":
        print(f"{'type':<20}{'n':>7}{'exact':>8}{'maj':>7}{'delta':>8}"
              f"{'contains':>10}{'unp':>5}")
        print("-" * 65)
        per = {}
        for t in sorted(seen):
            acc = 100 * hits[t] / seen[t]
            maj = 100 * gold_by_type[t][majority[t]] / seen[t]
            con = 100 * contains_hits[t] / seen[t]
            per[t] = {"n": seen[t], "accuracy": round(acc, 2),
                      "majority": round(maj, 2), "majority_label": majority[t],
                      "contains": round(con, 2), "unparsed": unparsed[t]}
            print(f"{t:<20}{seen[t]:>7}{acc:>7.1f}%{maj:>6.1f}%{acc - maj:>+8.1f}"
                  f"{con:>9.1f}%{unparsed[t]:>5}")
        overall = 100 * sum(hits.values()) / max(1, total)
        maj_overall = 100 * sum(gold_by_type[t][majority[t]] for t in seen) / max(1, total)
        con_overall = 100 * sum(contains_hits.values()) / max(1, total)
        avg = sum(v["accuracy"] for v in per.values()) / max(1, len(per))
        print("-" * 65)
        print(f"{'OVERALL':<20}{total:>7}{overall:>7.1f}%{maj_overall:>6.1f}%"
              f"{overall - maj_overall:>+8.1f}{con_overall:>9.1f}%"
              f"{sum(unparsed.values()):>5}")
        print(f"{'per-type average':<20}{'':>7}{avg:>7.1f}%")
        print("\n  Exact match after normalisation. `contains` is the upper bound if")
        print("  every wording difference were forgiven - the gap between them is")
        print("  phrasing, not perception. Quote exact.")
        summary.update({"overall_accuracy": round(overall, 2),
                        "majority_baseline": round(maj_overall, 2),
                        "beats_majority_by": round(overall - maj_overall, 2),
                        "contains_accuracy": round(con_overall, 2),
                        "per_type_average": round(avg, 2), "per_type": per})
    else:
        print(f"{'group':<14}{'n':>7}{'mIoU':>8}{'Acc@.5':>9}{'Acc@.7':>9}"
              f"{'whole mIoU':>12}{'whole@.5':>10}{'unp':>5}")
        print("-" * 74)
        per = {}
        for g in sorted(seen):
            n = seen[g]
            mi = sum(ious[g]) / n
            wm = sum(whole_ious[g]) / n
            per[g] = {"n": n, "miou": round(mi, 4),
                      "acc50": round(100 * acc50[g] / n, 2),
                      "acc70": round(100 * acc70[g] / n, 2),
                      "whole_miou": round(wm, 4),
                      "whole_acc50": round(100 * whole50[g] / n, 2),
                      "unparsed": unparsed[g]}
            print(f"{g:<14}{n:>7}{mi:>8.3f}{100 * acc50[g] / n:>8.1f}%"
                  f"{100 * acc70[g] / n:>8.1f}%{wm:>12.3f}"
                  f"{100 * whole50[g] / n:>9.1f}%{unparsed[g]:>5}")
        allv = [v for g in ious for v in ious[g]]
        allw = [v for g in whole_ious for v in whole_ious[g]]
        mi = sum(allv) / max(1, len(allv))
        print("-" * 74)
        print(f"{'OVERALL':<14}{total:>7}{mi:>8.3f}"
              f"{100 * sum(acc50.values()) / max(1, total):>8.1f}%"
              f"{100 * sum(acc70.values()) / max(1, total):>8.1f}%"
              f"{sum(allw) / max(1, len(allw)):>12.3f}"
              f"{100 * sum(whole50.values()) / max(1, total):>9.1f}%"
              f"{sum(unparsed.values()):>5}")
        print(f"\n  answer formats seen: {dict(fmt_seen)}")
        print("  The whole-image box is the trivial baseline, as in eval_ground.")
        summary.update({"miou": round(mi, 4),
                        "acc50": round(100 * sum(acc50.values()) / max(1, total), 2),
                        "acc70": round(100 * sum(acc70.values()) / max(1, total), 2),
                        "whole_miou": round(sum(allw) / max(1, len(allw)), 4),
                        "whole_acc50": round(100 * sum(whole50.values()) / max(1, total), 2),
                        "answer_format": args.answer_format,
                        "box_formats_seen": dict(fmt_seen),
                        "unparsed": sum(unparsed.values()), "per_group": per})

    print(f"\n  {elapsed / 60:.1f} min \u00b7 {total / max(1e-9, elapsed):.1f} q/s")
    print(f"  published for comparison: {PUBLISHED[args.task]}")
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  per-question -> {args.out}.jsonl")
    print(f"  summary      -> {args.out}.json")


if __name__ == "__main__":
    main()
