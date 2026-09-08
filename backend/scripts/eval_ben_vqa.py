"""Score VQA on BigEarthNet.txt's own questions - the statement's named dataset.

    python scripts/eval_ben_vqa.py --limit 400                    # base model
    python scripts/eval_ben_vqa.py --limit 400 --adapter runs/x   # fine-tuned

BigEarthNet.txt (arXiv:2603.29630) has 3.6 M binary and 3.3 M multiple-choice
questions. MEASURED 5 Sep: binary categories are presence / area / count /
adjacency, answered yes|no (51% no); mcq categories are presence / season /
climate zone / area / count / country, answered a|b|c|d (uniform 25%). The
inputs are complete questions already, so the template only pins the answer
form. Exact match, majority baseline beside every number, per type and per
category - the season / climate / country mcqs are not answerable from the
pixels of one tile, and the per-category rows say so rather than hiding it
in an average.

Imagery: the Lithuania/Summer reBEN LMDB (the only cell we hold). The split
is filtered to it, and rows whose patch is not in the LMDB are dropped and
counted. Use --split bench for the statement's benchmark split (its Lithuania
patches may be other seasons; the count printed says how many survived).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPTS = {
    "binary": "{q} Answer with one word: yes or no.",
    "mcq": "{q} Answer with the letter only: a, b, c or d.",
}
MAX_NEW = {"binary": 4, "mcq": 4}

# Fusion mode (--sar): the same questions asked over an optical + SAR pair.
# The prefix names which image is which; the fusion adapter is trained on
# it verbatim (a test pins the engine's copy to this string).
SAR_PREFIX = ("The first image is optical and the second is radar (SAR, VV/VH) of the "
              "same place on the same date. ")

_YES = re.compile(r"\byes\b", re.I)
_NO = re.compile(r"\bno\b", re.I)
_LETTER = re.compile(r"(?<![a-z])([abcd])(?![a-z])", re.I)


def normalise(raw, qtype):
    t = (raw or "").strip().lower().replace("**", "")
    if not t:
        return None
    if qtype == "binary":
        y, n = _YES.search(t), _NO.search(t)
        if y and n:
            return "yes" if y.start() < n.start() else "no"
        return "yes" if y else ("no" if n else None)
    m = _LETTER.search(t)
    return m.group(1).lower() if m else None


def load_rows(parquet, split, country, season):
    import pyarrow.parquet as pq
    filters = [("split", "=", split), ("type", "in", ["binary", "mcq"])]
    if country:
        filters.append(("country", "=", country))
    if season:
        filters.append(("season", "=", season))
    t = pq.read_table(parquet, columns=["ID", "patch_id", "s1_name", "input", "output", "type", "category"],
                      filters=filters)
    rows = t.to_pylist()
    for r in rows:
        r["gold"] = str(r["output"]).strip().lower()
    return rows


def stratified(rows, limit, seed=1337):
    """Equal share per (type, category), strided across patches."""
    import random
    if not limit or limit >= len(rows):
        return rows
    by = defaultdict(list)
    for r in rows:
        by[(r["type"], r["category"])].append(r)
    per = max(1, limit // len(by))
    out = []
    for k in sorted(by):
        pool = by[k]
        step = max(1, len(pool) // per)
        out.extend(pool[::step][:per])
    random.Random(seed).shuffle(out)
    return out[:limit]


def main():
    import torch
    from transformers import AutoProcessor
    from ben_images import BENImages

    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default=r"F:\sih\data\BigEarthNet.txt.parquet")
    p.add_argument("--lmdb", default=r"F:\sih\data\ben_lt_summer")
    p.add_argument("--split", default="test", choices=["train", "validation", "test", "bench"])
    p.add_argument("--country", default="Lithuania")
    p.add_argument("--season", default="Summer")
    p.add_argument("--model", default=r"F:\sih\models\Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--max-pixels", type=int, default=256)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--dtype", choices=["nf4", "int8", "fp16"], default="nf4")
    p.add_argument("--sar", action="store_true",
                   help="fusion mode: feed the optical tile AND its Sentinel-1 render, with SAR_PREFIX")
    p.add_argument("--blank-optical", action="store_true",
                   help="with --sar: replace the optical tile by mid-grey, so the score is what SAR alone answers")
    p.add_argument("--out", default=r"F:\sih\runs\ben_vqa_baseline")
    args = p.parse_args()

    tag = ("fine-tuned" if args.adapter else "BASELINE (no adapter)") + (" · optical+SAR" if args.sar else "") + (" · optical BLANKED" if args.blank_optical else "")
    print(f"=== BigEarthNet.txt VQA · {args.split} · {tag} ===")
    rows = load_rows(args.parquet, args.split, args.country, args.season)
    print(f"  {len(rows)} binary+mcq rows over {len({r['patch_id'] for r in rows})} patches "
          f"({args.country or 'all'} / {args.season or 'all'})")
    print("  by type:", dict(Counter(r["type"] for r in rows)))
    rows = stratified(rows, args.limit)
    print(f"  limited to {len(rows)} (stratified across type x category)")
    ben = BENImages(args.lmdb)
    have = [r for r in rows if ben.has(r["patch_id"]) and (not args.sar or ben.get_sar(r["s1_name"]) is not None)]
    print(f"  {len(have)} of {len(rows)} rows have imagery in {ben.path}")
    rows = have
    if not rows:
        print("  nothing to score - no patch of this split is in the LMDB")
        return 1

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

    gold_by = defaultdict(Counter)
    for r in rows:
        gold_by[(r["type"], r["category"])][r["gold"]] += 1
    majority = {k: c.most_common(1)[0][0] for k, c in gold_by.items()}

    hits, seen, unparsed = Counter(), Counter(), Counter()
    per_q = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    jsonl = open(args.out + ".jsonl", "w", encoding="utf-8")
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        img = ben.get(r["patch_id"], size=args.size)
        if args.blank_optical:
            from PIL import Image as _I
            img = _I.new("RGB", img.size, (128, 128, 128))
        q = r["input"].strip()
        prompt = PROMPTS[r["type"]].format(q=q)
        imgs = [img]
        if args.sar:
            imgs.append(ben.get_sar_rgb(r["s1_name"], size=args.size))
            prompt = SAR_PREFIX + prompt
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW[r["type"]], do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        pred = normalise(raw, r["type"])
        k = (r["type"], r["category"])
        ok = pred is not None and pred == r["gold"]
        hits[k] += ok
        seen[k] += 1
        unparsed[k] += pred is None
        rec = dict(r, raw=raw, pred=pred, correct=bool(ok))
        per_q.append(rec)
        jsonl.write(json.dumps(rec) + "\n")
        if i % 50 == 0 or i == len(rows):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(rows)}  running acc {100 * sum(hits.values()) / i:5.1f}%  "
                  f"{i / el:.1f} q/s  eta {(len(rows) - i) / (i / el) / 60:.1f} min")
    jsonl.close()

    print("\n  --- a few raw outputs ---")
    for rec in per_q[:10]:
        print(f"    {'OK' if rec['correct'] else 'XX'}  {rec['type']:7s} {rec['category']:13s} "
              f"gold={rec['gold']:<4s} raw={rec['raw'][:30]!r}")

    total = sum(seen.values())
    overall = 100 * sum(hits.values()) / total
    maj_overall = 100 * sum(gold_by[k][majority[k]] for k in seen) / total
    print(f"\n=== BigEarthNet.txt VQA · {args.split} · {tag} ===")
    print(f"{'type/category':24s} {'n':>5s} {'acc':>7s} {'majority':>9s} {'delta':>7s} {'unp':>4s}")
    print("-" * 60)
    per = {}
    by_type = defaultdict(lambda: [0, 0, 0])
    for k in sorted(seen):
        acc = 100 * hits[k] / seen[k]
        maj = 100 * gold_by[k][majority[k]] / seen[k]
        per["%s/%s" % k] = {"n": seen[k], "correct": hits[k], "accuracy": round(acc, 2),
                            "majority": round(maj, 2), "unparsed": unparsed[k]}
        by_type[k[0]][0] += hits[k]
        by_type[k[0]][1] += seen[k]
        by_type[k[0]][2] += gold_by[k][majority[k]]
        print(f"{k[0] + '/' + k[1]:24s} {seen[k]:5d} {acc:6.1f}% {maj:8.1f}% {acc - maj:+7.1f} {unparsed[k]:4d}")
    print("-" * 60)
    for ty, (h, n, m) in sorted(by_type.items()):
        print(f"{ty.upper():24s} {n:5d} {100 * h / n:6.1f}% {100 * m / n:8.1f}% {100 * (h - m) / n:+7.1f}")
    print(f"{'OVERALL':24s} {total:5d} {overall:6.1f}% {maj_overall:8.1f}% {overall - maj_overall:+7.1f} "
          f"{sum(unparsed.values()):4d}")
    print("\n  Exact match. Never quote the accuracy without the majority column beside it.")
    elapsed = time.perf_counter() - t0
    print(f"\n  {elapsed / 60:.1f} min · {total / elapsed:.1f} q/s")

    summary = {"run": tag, "adapter": args.adapter, "dtype": args.dtype, "split": args.split, "sar": args.sar, "blank_optical": args.blank_optical,
               "country": args.country, "season": args.season, "max_pixels_tokens": args.max_pixels,
               "n": total, "overall_accuracy": round(overall, 2),
               "majority_baseline": round(maj_overall, 2),
               "beats_majority_by": round(overall - maj_overall, 2),
               "by_type": {ty: {"n": n, "accuracy": round(100 * h / n, 2), "majority": round(100 * m / n, 2)}
                           for ty, (h, n, m) in by_type.items()},
               "elapsed_min": round(elapsed / 60, 2), "per_category": per}
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  per-question -> {args.out}.jsonl\n  summary      -> {args.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
