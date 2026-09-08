"""Score grounding on BigEarthNet.txt's bounding-box rows: IoU and Acc@0.5.

    python scripts/eval_ground.py --limit 400                    # base model
    python scripts/eval_ground.py --limit 400 --adapter runs/x   # fine-tuned

BigEarthNet.txt carries 2,205,686 box annotations in two categories:
  point      "Provide a bounding box for the land cover class instance at
              <point>(0.82, 0.28)</point> ..."            -> "[0.64 0.0, 1.0 0.71]"
  reference  "Identify the location of the <ref>largest connected region of
              pastures</ref>."                             -> "[0.0 0.33, 0.28 0.8]"
Boxes are normalised 0-1 in BigEarthNet.txt's own "[x0 y0, x1 y1]" string, and
the adapter is trained to answer in exactly that string (PROMPT_SUFFIX below
tells the model so; satquery/models.py parses it back and a test pins the
regex). Metric: mean IoU and the share of boxes with IoU >= 0.5, per category,
against a trivial baseline - the whole-image box, which is what the base
model answers for most queries and which scores well on large regions.

Imagery: the Lithuania/Summer reBEN LMDB (the only cell we hold), so the test
rows are that cell's test split. Same loader as training (scripts/ben_images.py).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT_SUFFIX = " Answer with the box as [x0 y0, x1 y1], normalised 0-1."

BEN_BOX = re.compile(r"\[\s*(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s*\]")
JSON_BOX = re.compile(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]")


def parse(text, resized_wh):
    """-> (x0, y0, x1, y1) normalised, or None. BEN string first, then a Qwen
    JSON/pixel box scaled by the frame the processor actually fed the model."""
    m = BEN_BOX.search(text or "")
    if m:
        b = [float(v) for v in m.groups()]
        if max(b) <= 1.0:
            return _order(b)
    m = JSON_BOX.search(text or "")
    if m and resized_wh:
        w, h = resized_wh
        b = [float(v) for v in m.groups()]
        if max(b) <= 1.0:
            return _order(b)
        if b[2] <= w and b[3] <= h:
            return _order([b[0] / w, b[1] / h, b[2] / w, b[3] / h])
    return None


def _order(b):
    x0, y0, x1, y1 = b
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return (x0, y0, x1, y1)


def parse_gold(s):
    m = BEN_BOX.search(s)
    return _order([float(v) for v in m.groups()]) if m else None


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_rows(parquet, split, country, season):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    filters = [("type", "=", "bounding box"), ("split", "=", split)]
    if country:
        filters.append(("country", "=", country))
    if season:
        filters.append(("season", "=", season))
    t = pq.read_table(parquet, columns=["ID", "patch_id", "input", "output", "category"], filters=filters)
    return t.to_pylist()


def stratified(rows, limit, seed=1337):
    """Half point, half reference, strided across patches."""
    import random
    if not limit or limit >= len(rows):
        return rows
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    per = limit // len(by_cat)
    out = []
    for c in sorted(by_cat):
        pool = by_cat[c]
        step = max(1, len(pool) // per)
        out.extend(pool[::step][:per])
    random.Random(seed).shuffle(out)
    return out[:limit]


def main():
    import torch
    from transformers import AutoProcessor
    from ben_images import BENImages   # lazy: lmdb is not on every machine that imports PROMPT_SUFFIX

    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default=r"F:\sih\data\BigEarthNet.txt.parquet")
    p.add_argument("--lmdb", default=r"F:\sih\data\ben_lt_summer")
    p.add_argument("--split", default="test")
    p.add_argument("--country", default="Lithuania")
    p.add_argument("--season", default="Summer")
    p.add_argument("--model", default=r"F:\sih\models\Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--max-pixels", type=int, default=256)
    p.add_argument("--size", type=int, default=224, help="tile size fed to the processor")
    p.add_argument("--dtype", choices=["nf4", "int8", "fp16"], default="nf4")
    p.add_argument("--out", default=r"F:\sih\runs\ground_baseline")
    args = p.parse_args()

    tag = "fine-tuned" if args.adapter else "BASELINE (no adapter)"
    print(f"=== BigEarthNet.txt grounding · {args.split} · {tag} ===")
    rows = load_rows(args.parquet, args.split, args.country, args.season)
    print(f"  {len(rows)} box rows over {len({r['patch_id'] for r in rows})} patches "
          f"({args.country or 'all'} / {args.season or 'all'})")
    print("  by category:", dict(Counter(r["category"] for r in rows)))
    rows = stratified(rows, args.limit)
    print(f"  limited to {len(rows)}")

    ben = BENImages(args.lmdb)
    rows = [r for r in rows if ben.has(r["patch_id"])]
    print(f"  {len(rows)} rows have imagery in {ben.path}")

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
    print(f"  weights: {args.dtype} · ready in {time.perf_counter() - t0:.1f}s")

    ious = defaultdict(list)
    whole = defaultdict(list)     # the whole-image box, the trivial baseline
    unparsed = Counter()
    per_q = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    jsonl = open(args.out + ".jsonl", "w", encoding="utf-8")
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        img = ben.get(r["patch_id"], size=args.size)
        prompt = r["input"] + PROMPT_SUFFIX
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        grid = inputs.get("image_grid_thw")
        resized = None
        if grid is not None:
            _, gh, gw = grid.tolist()[0]
            resized = (gw * 14, gh * 14)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        gold = parse_gold(r["output"])
        pred = parse(raw, resized)
        v = iou(pred, gold)
        ious[r["category"]].append(v)
        whole[r["category"]].append(iou((0.0, 0.0, 1.0, 1.0), gold))
        unparsed[r["category"]] += pred is None
        rec = dict(r, raw=raw, pred=pred, gold_box=gold, iou=round(v, 4))
        per_q.append(rec)
        jsonl.write(json.dumps(rec) + "\n")
        if i % 50 == 0 or i == len(rows):
            el = time.perf_counter() - t0
            allv = [x for c in ious.values() for x in c]
            print(f"  {i}/{len(rows)}  mean IoU {sum(allv) / len(allv):.3f}  "
                  f"Acc@0.5 {100 * sum(x >= 0.5 for x in allv) / len(allv):5.1f}%  "
                  f"{i / el:.1f} q/s  eta {(len(rows) - i) / (i / el) / 60:.1f} min")
    jsonl.close()

    print("\n  --- a few raw outputs ---")
    for rec in per_q[:8]:
        print(f"    iou={rec['iou']:.2f} {rec['category']:9s} gold={rec['output']:<24s} raw={rec['raw'][:60]!r}")

    def acc(v):
        return 100 * sum(x >= 0.5 for x in v) / max(1, len(v))

    allv = [x for c in ious.values() for x in c]
    allw = [x for c in whole.values() for x in c]
    print(f"\n=== BigEarthNet.txt grounding · {tag} ===")
    print(f"{'category':10s} {'n':>5s} {'mIoU':>7s} {'Acc@.5':>7s} {'whole mIoU':>11s} {'whole Acc':>10s} {'unp':>4s}")
    print("-" * 52)
    per_cat = {}
    for c in sorted(ious):
        v, w = ious[c], whole[c]
        per_cat[c] = {"n": len(v), "miou": round(sum(v) / len(v), 4), "acc50": round(acc(v), 2),
                      "whole_miou": round(sum(w) / len(w), 4), "whole_acc50": round(acc(w), 2),
                      "unparsed": unparsed[c]}
        print(f"{c:10s} {len(v):5d} {sum(v) / len(v):7.3f} {acc(v):6.1f}% {sum(w) / len(w):11.3f} {acc(w):9.1f}% {unparsed[c]:4d}")
    print("-" * 52)
    print(f"{'OVERALL':10s} {len(allv):5d} {sum(allv) / len(allv):7.3f} {acc(allv):6.1f}% "
          f"{sum(allw) / len(allw):11.3f} {acc(allw):9.1f}% {sum(unparsed.values()):4d}")
    print("\n  'whole' is the whole-image box - the trivial baseline any number here must beat.")
    elapsed = time.perf_counter() - t0
    print(f"\n  {elapsed / 60:.1f} min · {len(rows) / elapsed:.1f} q/s")

    summary = {"run": tag, "adapter": args.adapter, "dtype": args.dtype, "split": args.split,
               "country": args.country, "season": args.season, "size": args.size,
               "max_pixels_tokens": args.max_pixels, "n": len(allv),
               "miou": round(sum(allv) / len(allv), 4), "acc50": round(acc(allv), 2),
               "whole_miou": round(sum(allw) / len(allw), 4), "whole_acc50": round(acc(allw), 2),
               "unparsed": sum(unparsed.values()), "elapsed_min": round(elapsed / 60, 2),
               "per_category": per_cat}
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  per-question -> {args.out}.jsonl\n  summary      -> {args.out}.json")


if __name__ == "__main__":
    sys.exit(main())
