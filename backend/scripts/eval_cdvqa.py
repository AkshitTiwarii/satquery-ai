"""Score Qwen2.5-VL on CDVQA - the bi-temporal change-VQA benchmark.

    python scripts/eval_cdvqa.py --limit 400                     # base model
    python scripts/eval_cdvqa.py --limit 400 --adapter runs/x    # fine-tuned

CDVQA (Yuan et al., TGRS 2022) asks questions about a PAIR of co-registered
aerial images from the SECOND dataset: did class X change, increase, decrease;
what did it change to; which change is largest/smallest; what fraction changed.
Eight types, a closed vocabulary per type, scored by exact match - which is
how the paper reports it, so the number is comparable.

Two things the harness pins down so the score means something:

  * THE PROMPT NAMES THE VOCABULARY. The answers are dataset tokens
    ('NVG_surface', '10_to_20'), not English. A model asked in plain English
    answers in plain English and scores zero on a format technicality. The
    templates below are also what the adapter is trained on
    (scripts/make_cdvqa_subset.py imports them), so train and test agree.
  * A MAJORITY BASELINE BESIDE EVERY NUMBER. change_ratio_types is 47% '0'
    and change_or_not is 57% 'yes'; an adapter that learned the priors and
    nothing else would look respectable without it.

The base model's own answer to this task, measured 2 Sep: "I'm sorry, but I
can't see any images to compare." Capability 3 is fine-tune or nothing.
"""

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

from PIL import Image

# scripts/ on the path, so the shared weight loader is importable
# however this file is invoked - from the repo root, from inside
# scripts/, or as /kaggle/working/src/scripts/<name>.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ["NVG_surface", "buildings", "low_vegetation", "trees", "water", "playgrounds"]
RATIOS = ["0", "0_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_50",
          "50_to_60", "60_to_70", "70_to_80", "80_to_90", "90_to_100"]

_YESNO = "{q} Answer with one word: yes or no."
_CLASS = "{q} Answer with one of: " + ", ".join(CLASSES) + "."
_RATIO = "{q} Answer with one of: " + ", ".join(RATIOS) + "."

PROMPTS = {
    "change_or_not": _YESNO,
    "increase_or_not": _YESNO,
    "decrease_or_not": _YESNO,
    "change_to_what": _CLASS,
    "smallest_change": _CLASS,
    "largest_change": _CLASS,
    "change_ratio": _RATIO,
    "change_ratio_types": _RATIO,
}
KIND = {"change_or_not": "yesno", "increase_or_not": "yesno", "decrease_or_not": "yesno",
        "change_to_what": "class", "smallest_change": "class", "largest_change": "class",
        "change_ratio": "ratio", "change_ratio_types": "ratio"}
MAX_NEW = {"yesno": 4, "class": 8, "ratio": 8}

# The two images are "pre-change" and "post-change" in the questions' own
# words. Say so, or "the first image" in a question is ambiguous to the model.
PAIR_PREFIX = ("The first image is the pre-change image and the second is the "
               "post-change image of the same place. ")

_CLASS_ALIASES = {
    "nvg_surface": "NVG_surface", "non-vegetated ground surface": "NVG_surface",
    "non vegetated ground surface": "NVG_surface", "nvg": "NVG_surface", "ground": "NVG_surface",
    "buildings": "buildings", "building": "buildings",
    "low_vegetation": "low_vegetation", "low vegetation": "low_vegetation",
    "trees": "trees", "tree": "trees",
    "water": "water", "playgrounds": "playgrounds", "playground": "playgrounds",
}


def normalise(raw, qtype):
    t = (raw or "").strip().lower().replace("**", "")
    if not t:
        return None
    kind = KIND[qtype]
    if kind == "yesno":
        y, n = re.search(r"\byes\b", t), re.search(r"\bno\b", t)
        if y and n:
            return "yes" if y.start() < n.start() else "no"
        return "yes" if y else ("no" if n else None)
    if kind == "class":
        best = None
        for alias, canon in _CLASS_ALIASES.items():
            i = t.find(alias)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, canon)
        return best[1] if best else None
    m = re.search(r"\b(\d{1,3})\s*(?:_|-| )?to(?:_|-| )?(\d{1,3})\b", t)
    if m:
        cand = "%s_to_%s" % (m.group(1), m.group(2))
        return cand if cand in RATIOS else None
    if re.search(r"^\s*0\b|\bzero\b|\bnone\b|\bno change\b", t):
        return "0"
    return None


def load_split(data_dir, split):
    with open(os.path.join(data_dir, f"{split}_questions.json"), encoding="utf-8") as fh:
        questions = [q for q in json.load(fh)["questions"] if q.get("active")]
    with open(os.path.join(data_dir, f"{split}_answers.json"), encoding="utf-8") as fh:
        answers = {a["question_id"]: str(a["answer"]).strip()
                   for a in json.load(fh)["answers"] if a.get("active")}
    with open(os.path.join(data_dir, f"{split}_images.json"), encoding="utf-8") as fh:
        files = {im["id"]: im["file_name"] for im in json.load(fh)["images"] if im.get("active")}
    rows = []
    for q in questions:
        if q["id"] in answers and q["img_id"] in files and q["type"] in PROMPTS:
            rows.append({"qid": q["id"], "img_id": q["img_id"], "file": files[q["img_id"]],
                         "type": q["type"], "question": q["question"].strip(),
                         "gold": answers[q["id"]]})
    return rows


def stratified(rows, limit):
    """`limit` rows spread across type AND file: the split is ordered by image,
    and one file carries 16 augmented image ids, so the first N of a type would
    all be one picture asked sixteen ways."""
    if not limit or limit >= len(rows):
        return rows
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    per = max(1, limit // len(by_type))
    out = []
    for t in sorted(by_type):
        pool = by_type[t]
        step = max(1, len(pool) // per)
        out.extend(pool[::step][:per])
    return out[:limit]


def load_pairs(images_dir, files, im1="im1", im2="im2"):
    """file name -> (pre PIL, post PIL). Both must exist or the pair is skipped
    loudly; a silently missing second image would turn every change question
    into a single-image one."""
    pairs, missing = {}, []
    for f in sorted(files):
        a, b = os.path.join(images_dir, im1, f), os.path.join(images_dir, im2, f)
        if os.path.exists(a) and os.path.exists(b):
            pairs[f] = (Image.open(a).convert("RGB"), Image.open(b).convert("RGB"))
        else:
            missing.append(f)
    if missing:
        print(f"  WARNING: {len(missing)} pairs missing an image, e.g. {missing[:3]}")
    return pairs


def main():
    import torch
    from transformers import AutoProcessor

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=r"F:\sih\data\CDVQA-main")
    p.add_argument("--images-dir", default=r"F:\sih\data\second",
                   help="holds im1/ and im2/ with the SECOND PNGs")
    p.add_argument("--split", default="Test")
    p.add_argument("--model", default=r"F:\sih\models\Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--limit", type=int, default=0, help="0 = the whole split")
    p.add_argument("--max-pixels", type=int, default=256, help="visual tokens per image")
    p.add_argument("--dtype", choices=["nf4", "int8", "fp16"], default="nf4")
    p.add_argument("--out", default=r"F:\sih\runs\cdvqa_baseline")
    args = p.parse_args()

    tag = "fine-tuned" if args.adapter else "BASELINE (no adapter)"
    print(f"=== CDVQA {args.split} · {tag} ===")
    rows = load_split(args.data_dir, args.split)
    print(f"  {len(rows)} active questions over {len({r['file'] for r in rows})} image pairs")
    counts = Counter(r["type"] for r in rows)
    print("  by type: " + " · ".join(f"{t} {counts[t]}" for t in sorted(counts)))
    # The full split's own type mix, kept before the sample is balanced. The
    # CDVQA papers report Overall Accuracy over THIS distribution, which is
    # 35% change_or_not and only 4.9% change_ratio; a sample with 50 of each
    # type answers a harder question (Average Accuracy) and cannot be set
    # beside a published OA without being re-weighted. Both are reported.
    split_mix = dict(counts)
    split_total = sum(split_mix.values())
    rows = stratified(rows, args.limit)
    if args.limit:
        print(f"  limited to {len(rows)} (stratified across types)")
    pairs = load_pairs(args.images_dir, {r["file"] for r in rows})
    rows = [r for r in rows if r["file"] in pairs]
    print(f"  decoded {len(pairs)} pairs at {next(iter(pairs.values()))[0].size}")

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
    print(f"  weights: {args.dtype} · ready in {time.perf_counter() - t0:.1f}s, "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB resident")

    gold_by_type = defaultdict(Counter)
    for r in rows:
        gold_by_type[r["type"]][r["gold"]] += 1
    majority = {t: c.most_common(1)[0][0] for t, c in gold_by_type.items()}

    hits, seen, unparsed = Counter(), Counter(), Counter()
    per_q = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    jsonl = open(args.out + ".jsonl", "w", encoding="utf-8")
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        pre, post = pairs[r["file"]]
        prompt = PAIR_PREFIX + PROMPTS[r["type"]].format(q=r["question"])
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[pre, post], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW[KIND[r["type"]]],
                                 do_sample=False, temperature=None, top_p=None, top_k=None)
        raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0].strip()
        pred = normalise(raw, r["type"])
        ok = pred is not None and pred == r["gold"]
        hits[r["type"]] += ok
        seen[r["type"]] += 1
        unparsed[r["type"]] += pred is None
        rec = dict(r, raw=raw, pred=pred, correct=bool(ok))
        per_q.append(rec)
        jsonl.write(json.dumps(rec) + "\n")
        if i % 50 == 0 or i == len(rows):
            el = time.perf_counter() - t0
            acc = 100 * sum(hits.values()) / i
            print(f"  {i}/{len(rows)}  running acc {acc:5.1f}%  {i / el:.1f} q/s  "
                  f"eta {(len(rows) - i) / (i / el) / 60:.1f} min")
    jsonl.close()

    print("\n  --- a few raw outputs, to check the normaliser is not eating answers ---")
    for rec in per_q[:12]:
        print(f"    {'OK' if rec['correct'] else 'XX'}  {rec['type']:19s} gold={rec['gold']:<14s} raw={rec['raw']!r}")

    total = sum(seen.values())
    overall = 100 * sum(hits.values()) / total
    maj_hits = sum(gold_by_type[t][majority[t]] for t in seen)
    maj_overall = 100 * maj_hits / total

    # Overall Accuracy: each type weighted by its share of the real test
    # split, which is what the CDVQA papers mean by OA. Post-stratified from
    # the per-type rates we measured, so it is an estimate of the number a
    # full-split run would print - run --limit 0 for the real one.
    weight = {t: split_mix.get(t, 0) / split_total for t in seen}
    covered = sum(weight.values())
    natural = (sum(100 * hits[t] / seen[t] * weight[t] for t in seen) / covered
               if covered else 0.0)
    maj_natural = (sum(100 * gold_by_type[t][majority[t]] / seen[t] * weight[t]
                       for t in seen) / covered if covered else 0.0)
    print(f"\n=== CDVQA {args.split} · {tag} ===")
    print(f"{'type':20s} {'n':>5s} {'acc':>7s} {'majority':>9s} {'delta':>7s} {'unp':>4s}")
    print("-" * 58)
    per_type = {}
    for t in sorted(seen):
        acc = 100 * hits[t] / seen[t]
        maj = 100 * gold_by_type[t][majority[t]] / seen[t]
        per_type[t] = {"n": seen[t], "correct": hits[t], "accuracy": round(acc, 2),
                       "majority": round(maj, 2), "majority_label": majority[t],
                       "unparsed": unparsed[t]}
        print(f"{t:20s} {seen[t]:5d} {acc:6.1f}% {maj:8.1f}% {acc - maj:+7.1f} {unparsed[t]:4d}")
    print("-" * 58)
    print(f"{'OVERALL':20s} {total:5d} {overall:6.1f}% {maj_overall:8.1f}% {overall - maj_overall:+7.1f} "
          f"{sum(unparsed.values()):4d}")
    print(f"{'OA (natural mix)':20s} {split_total:5d} {natural:6.1f}% {maj_natural:8.1f}%"
          f" {natural - maj_natural:+7.1f}")
    print("\n  Exact match on the dataset's own tokens, as the CDVQA paper scores it.")
    print("  TWO protocols, and they differ by more than ten points:")
    print("    OVERALL here is Average Accuracy - every type weighted equally,")
    print("      which is the harder read because the weak types (ratio, smallest)")
    print("      are rare in the real split.")
    print("    OA (natural mix) re-weights the same per-type rates by the split's")
    print("      own type frequencies. THAT is the column CDVQA papers publish;")
    print("      the 2026 Qwen3.5-2B result is OA 74.74 / AA 68.59.")
    print("  Say which one a number is. Never quote either without its majority.")
    if sum(unparsed.values()) > total * 0.02:
        print("\n  WARNING: more than 2% of answers could not be parsed - a prompt problem, "
              "not a model problem. Read the raw outputs above before believing this score.")
    elapsed = time.perf_counter() - t0
    print(f"\n  {elapsed / 60:.1f} min · {total / elapsed:.1f} q/s")

    summary = {
        "run": tag, "adapter": args.adapter, "dtype": args.dtype, "split": args.split,
        "max_pixels_tokens": args.max_pixels, "n": total,
        "overall_accuracy": round(overall, 2), "majority_baseline": round(maj_overall, 2),
        "beats_majority_by": round(overall - maj_overall, 2),
        "average_accuracy": round(overall, 2),
        "overall_accuracy_natural": round(natural, 2),
        "majority_baseline_natural": round(maj_natural, 2),
        "beats_majority_by_natural": round(natural - maj_natural, 2),
        "split_type_mix": split_mix,
        "elapsed_min": round(elapsed / 60, 2), "per_type": per_type,
    }
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  per-question -> {args.out}.jsonl")
    print(f"  summary      -> {args.out}.json")


if __name__ == "__main__":
    sys.exit(main())
