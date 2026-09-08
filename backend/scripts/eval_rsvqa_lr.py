"""Score a Qwen2.5-VL model on the RSVQA-LR test split.

This is the harness behind Gate 1 and Gate 2 in PLAN.md. Run it once with no
adapter to get the baseline, then again with --adapter after the LoRA lands.
Both numbers must come from THIS script or the comparison means nothing.

Three things it is careful about, each of which silently ruins the score:

  1. The `active` flag. The split files contain every question in the dataset,
     33,212 of them, and only 10,004 belong to the test split. Inactive entries
     are bare stubs with no question text at all. Filter first, read second.

  2. Answer format. The model writes "Yes, there is a water body." and the gold
     answer is "yes". Exact string match scores that as wrong and you conclude
     the model is broken when it is not. Every answer is normalised, and the
     count of answers we could NOT parse is reported per type - a high unparsed
     count means the prompt is wrong, not the model.

  3. Counting. 456 distinct gold answers, scored on exact match. Every model on
     this benchmark scores badly here. Report it separately so a weak overall
     number can be attributed correctly.

Usage
    python scripts/eval_rsvqa_lr.py                       # full baseline
    python scripts/eval_rsvqa_lr.py --limit 400           # quick check
    python scripts/eval_rsvqa_lr.py --adapter runs/lora   # score the fine-tune
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

# scripts/ on the path, so the shared weight loader is importable
# however this file is invoked - from the repo root, from inside
# scripts/, or as /kaggle/working/src/scripts/<name>.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# One instruction per question type. Without these the model writes prose and
# the normaliser has to guess; with them nearly everything parses first try.
PROMPTS = {
    "presence":    "{q}? Answer with one word: yes or no.",
    "comp":        "{q}? Answer with one word: yes or no.",
    "rural_urban": "{q}? Answer with one word: rural or urban.",
    "count":       "{q}? Answer with a single number and nothing else.",
}
MAX_NEW = {"presence": 5, "comp": 5, "rural_urban": 5, "count": 8}

# The counting instruction an adapter trained with --count-answer bucket needs.
# Its answer is "11-100", so "a single number and nothing else" would be an
# instruction the model has to be taught to disobey. PROMPTS above is NOT
# edited: every recorded number - the 60.75 base, the 84.5 joint adapter - was
# measured with it, and changing it in place would silently invalidate them
# all. Pick with --count-prompt; the summary says which was used.
COUNT_BUCKET_PROMPT = "{q}? Answer with one of: 0, 1-10, 11-100, 101-1000, >1000."
COUNT_BUCKET_MAX_NEW = 12


def prompts_for(count_prompt):
    """The four templates, with counting posed the way the adapter expects."""
    if count_prompt == "number":
        return dict(PROMPTS), dict(MAX_NEW)
    if count_prompt == "bucket":
        p, m = dict(PROMPTS), dict(MAX_NEW)
        p["count"] = COUNT_BUCKET_PROMPT
        m["count"] = COUNT_BUCKET_MAX_NEW
        return p, m
    raise ValueError(f"unknown count prompt {count_prompt!r}")

YES = re.compile(r"\byes\b|\btrue\b", re.I)
NO = re.compile(r"\bno\b|\bfalse\b|\bnone\b", re.I)
RURAL = re.compile(r"\brural\b", re.I)
URBAN = re.compile(r"\burban\b", re.I)
INT = re.compile(r"-?\d+")

# The published RSVQA protocol bins counts into five buckets rather than scoring
# exact integers. GeoChat's paper states it plainly: "counting questions are
# quantified into five categories: 0, between 1 and 10, between 11 and 100,
# between 101 and 1000, and greater than 1000". Exact match against 456 raw
# integers is a much harder test, and a number produced that way cannot be
# compared with any published figure. We report both.
COUNT_BINS = [(0, 0, "0"), (1, 10, "1-10"), (11, 100, "11-100"),
              (101, 1000, "101-1000"), (1001, float("inf"), ">1000")]

# The five labels, in the exact spelling make_rsvqa_subset.py writes as the
# training answer under --count-answer bucket. An adapter trained that way
# answers "11-100" where the earlier ones answered "37", so the parser below
# reads a label first and an integer second. Do not respell these: the subset
# cutter imports this tuple, so a change here changes what the model is taught.
COUNT_LABELS = tuple(label for _, _, label in COUNT_BINS)

# Longest alternative first, so "101-1000" is never read as the "1-10" bucket.
# The lookarounds keep it off ordinary integers: a bare "197" carries no
# separator and no ">", so it falls straight through to INT as it always did.
_SEP = r"(?:-|–|—|to|through|and)"
# The lookarounds must reject a longer number ("1101", "11-1000", "100.5") but
# not ordinary sentence punctuation - "The answer is 11-100." is a label.
BUCKET = re.compile(
    r"(?<!\d)(?<!\d\.)"
    r"(?:"
    r"(?:>|greater\s+than|more\s+than|over)\s*1000"
    r"|101\s*" + _SEP + r"\s*1000"
    r"|11\s*" + _SEP + r"\s*100"
    r"|1\s*" + _SEP + r"\s*10"
    r")"
    r"(?!\d)(?!\.\d)",
    re.I,
)


def _canonical_bucket(text):
    """The label a bucket-shaped phrase denotes, or None if there is none."""
    m = BUCKET.search(text)
    if not m:
        return None
    found = m.group(0)
    if re.match(r">|greater|more|over", found, re.I):
        return ">1000"
    lo = int(INT.findall(found)[0])
    for _lo, _hi, label in COUNT_BINS:
        if _lo == lo:
            return label
    return None


def bin_count(value):
    """A count answer -> the published five-bucket label, or None.

    Takes either a raw integer - what the dataset's gold answers are, and what
    the integer-trained adapters emit - or one of the five labels, which the
    bucket-trained adapter emits and which is already its own bucket.
    """
    if value in COUNT_LABELS:
        return value
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    for lo, hi, label in COUNT_BINS:
        if lo <= n <= hi:
            return label
    return None


WORD_NUMBERS = {
    "zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def normalise(raw, qtype):
    """Model text -> a comparable answer, or None if we genuinely cannot read it."""
    t = (raw or "").strip().lower()
    if not t:
        return None

    if qtype in ("presence", "comp"):
        y, n = YES.search(t), NO.search(t)
        if y and not n:
            return "yes"
        if n and not y:
            return "no"
        if y and n:                       # both present: take whichever came first
            return "yes" if y.start() < n.start() else "no"
        return None

    if qtype == "rural_urban":
        r, u = RURAL.search(t), URBAN.search(t)
        if r and not u:
            return "rural"
        if u and not r:
            return "urban"
        if r and u:
            return "rural" if r.start() < u.start() else "urban"
        return None

    if qtype == "count":
        # A bucket label first. It has to come first because every label but
        # ">1000" also contains an integer, and ">1000" contains the wrong one:
        # the integer reader takes its 1000 and bins it as 101-1000.
        bucket = _canonical_bucket(t)
        if bucket:
            return bucket
        m = INT.search(t)
        if m:
            return str(int(m.group()))
        for word, val in WORD_NUMBERS.items():
            if re.search(r"\b%s\b" % word, t):
                return str(val)
        return None

    return t or None


def load_split(data_dir):
    """Active test questions joined to their gold answers."""
    with open(os.path.join(data_dir, "LR_split_test_questions.json"), encoding="utf-8") as fh:
        questions = [q for q in json.load(fh)["questions"] if q.get("active")]
    with open(os.path.join(data_dir, "LR_split_test_answers.json"), encoding="utf-8") as fh:
        answers = [a for a in json.load(fh)["answers"] if a.get("active")]

    gold = {a["question_id"]: str(a["answer"]).strip().lower() for a in answers}
    rows = []
    for q in questions:
        if q["id"] in gold:
            rows.append({
                "qid": q["id"], "img_id": q["img_id"],
                "type": q["type"], "question": q["question"].strip(),
                "gold": gold[q["id"]],
            })
    missing = len(questions) - len(rows)
    if missing:
        print(f"  warning: {missing} active questions had no active answer, dropped")
    return rows


def stratified(rows, limit):
    """Take `limit` rows spread across BOTH type and image.

    Taking the first N of each type looks stratified and is not: the split file
    is ordered by image, so the first 100 comparison questions come from three
    images. Every per-type number then describes three pictures. Stride through
    each type instead, so the sample spans all 100 images.
    """
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


def load_images(zip_path, needed):
    """Decode each image once. 100 images, so keeping them all in memory is fine."""
    images = {}
    with zipfile.ZipFile(zip_path) as z:
        names = {os.path.splitext(os.path.basename(n))[0]: n
                 for n in z.namelist() if n.lower().endswith((".tif", ".tiff", ".png", ".jpg"))}
        for img_id in sorted(needed):
            key = str(img_id)
            if key not in names:
                continue
            images[img_id] = Image.open(io.BytesIO(z.read(names[key]))).convert("RGB")
    return images


def main():
    # Lazy on purpose: make_rsvqa_subset.py imports PROMPTS from this file on
    # machines that have neither torch nor transformers.
    import torch
    from transformers import AutoProcessor

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=r"F:\sih\data\rsvqa_lr")
    p.add_argument("--model", default=r"F:\sih\models\Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA directory; omit for the baseline")
    p.add_argument("--limit", type=int, default=0, help="0 = all 10,004")
    p.add_argument("--max-pixels", type=int, default=256, help="visual tokens per image")
    p.add_argument("--dtype", choices=["nf4", "int8", "fp16"], default="nf4",
                   help="nf4 = 4-bit via bitsandbytes, the 8 GB path and the baseline's; "
                        "fp16 = full half-precision weights, 7.5 GB, needs a 16 GB card "
                        "and no bitsandbytes at all (the Kaggle T4 path)")
    p.add_argument("--out", default=r"F:\sih\runs\rsvqa_lr_baseline")
    p.add_argument("--count-prompt", choices=["number", "bucket"], default="number",
                   help="how counting questions are posed. 'number' is the template "
                        "every recorded run used and asks for a raw integer; "
                        "'bucket' lists the five published buckets and is what an "
                        "adapter cut with make_rsvqa_subset.py --count-answer bucket "
                        "was trained on. The binned score is the same metric either "
                        "way, so the two stay comparable; the summary records which.")
    args = p.parse_args()
    prompts, max_new = prompts_for(args.count_prompt)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tag = "fine-tuned" if args.adapter else "BASELINE (no adapter)"
    print(f"=== RSVQA-LR test · {tag} ===")
    print(f"  counting posed as {args.count_prompt}: {prompts['count']}")

    rows = load_split(args.data_dir)
    print(f"  {len(rows)} active questions over {len({r['img_id'] for r in rows})} images")
    counts = Counter(r["type"] for r in rows)
    print("  by type: " + " · ".join(f"{t} {counts[t]}" for t in sorted(counts)))
    if len(rows) != 10004:
        print(f"  WARNING: expected 10,004 active questions, got {len(rows)}. "
              f"Check the split files before trusting this score.")

    rows = stratified(rows, args.limit)
    if args.limit:
        print(f"  limited to {len(rows)} (stratified across types)")

    images = load_images(os.path.join(args.data_dir, "Images_LR.zip"),
                         {r["img_id"] for r in rows})
    print(f"  decoded {len(images)} images at {next(iter(images.values())).size}")

    # Two loaders on purpose. The Kaggle notebook trains in fp16 and never
    # installs bitsandbytes, and transformers refuses a BitsAndBytesConfig
    # unless the package's metadata is present - so the 4-bit import has to be
    # lazy or the fp16 path dies at the same line the v8 run died at.
    t0 = time.perf_counter()
    from qwen_loader import load_model
    model = load_model(args.model, args.dtype)
    print(f"  weights: {args.dtype}")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  adapter loaded from {args.adapter}")
    model.eval()
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=64 * 28 * 28,
                                         max_pixels=args.max_pixels * 28 * 28)
    print(f"  model ready in {time.perf_counter() - t0:.1f}s, "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB resident")

    # majority-class baseline: what you would score by always answering the most
    # common label for that type. Any honest report of accuracy needs it alongside.
    gold_by_type = defaultdict(Counter)
    gold_binned_by_type = defaultdict(Counter)
    imgs_per_type = defaultdict(set)
    for r in rows:
        gold_by_type[r["type"]][r["gold"]] += 1
        # the majority baseline has to be computed on the SAME scale as the score
        # it is compared against, or the comparison is meaningless
        gb = bin_count(r["gold"]) if r["type"] == "count" else r["gold"]
        gold_binned_by_type[r["type"]][gb] += 1
        imgs_per_type[r["type"]].add(r["img_id"])
    majority = {t: c.most_common(1)[0][1] for t, c in gold_by_type.items()}
    majority_binned = {t: c.most_common(1)[0][1] for t, c in gold_binned_by_type.items()}
    imgs_per_type = {t: len(v) for t, v in imgs_per_type.items()}

    hits = Counter()
    hits_binned = Counter()
    seen = Counter()
    unparsed = Counter()
    samples = []
    t_start = time.perf_counter()

    bucket_form_preds = 0
    with open(args.out + ".jsonl", "w", encoding="utf-8") as sink:
        for i, r in enumerate(rows, 1):
            img = images.get(r["img_id"])
            if img is None:
                continue
            qtype = r["type"]
            prompt = prompts.get(qtype, "{q}? Answer briefly.").format(q=r["question"].rstrip("?"))
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda:0")
            n_in = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new.get(qtype, 8),
                                     do_sample=False)
            raw = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()

            pred = normalise(raw, qtype)
            correct = pred is not None and pred == r["gold"]
            if qtype == "count":                      # published protocol bins counts
                pb, gb = bin_count(pred), bin_count(r["gold"])
                correct_binned = pb is not None and pb == gb
                # A bucket-trained adapter answers "11-100"; an integer-trained
                # one answers "37". Recording the share tells a later reader
                # which form produced the number, and catches an adapter that
                # has half-reverted to integers under the bucket prompt.
                bucket_form_preds += int(pred in COUNT_LABELS)
            else:
                correct_binned = correct
            seen[qtype] += 1
            hits[qtype] += int(correct)
            hits_binned[qtype] += int(correct_binned)
            if pred is None:
                unparsed[qtype] += 1
            if len(samples) < 12:
                samples.append((qtype, r["question"][:44], r["gold"], raw[:28], correct))

            sink.write(json.dumps({**r, "raw": raw, "pred": pred, "correct": correct,
                                   "correct_binned": correct_binned}) + "\n")

            if i % 250 == 0 or i == len(rows):
                rate = i / (time.perf_counter() - t_start)
                done = sum(hits.values()) / max(1, sum(seen.values())) * 100
                eta = (len(rows) - i) / rate / 60
                print(f"  {i}/{len(rows)}  running acc {done:5.1f}%  "
                      f"{rate:4.1f} q/s  eta {eta:4.1f} min", flush=True)

    elapsed = time.perf_counter() - t_start
    print("\n  --- a few raw outputs, to check the normaliser is not eating answers ---")
    for qtype, q, gold, raw, ok in samples:
        print(f"    {'OK ' if ok else 'XX '} {qtype:<12} gold={gold:<8} raw={raw!r}")

    print(f"\n=== RSVQA-LR test · {tag} ===")
    print(f"{'type':<14}{'n':>6}{'exact':>8}{'maj':>7}{'binned':>9}{'maj-b':>8}"
          f"{'delta':>8}{'unp':>5}{'imgs':>6}")
    print("-" * 71)
    for qtype in sorted(seen):
        acc = hits[qtype] / seen[qtype] * 100
        maj = majority[qtype] / seen[qtype] * 100
        majb = majority_binned[qtype] / seen[qtype] * 100
        binned = hits_binned[qtype] / seen[qtype] * 100
        print(f"{qtype:<14}{seen[qtype]:>6}{acc:>7.1f}%{maj:>6.1f}%{binned:>8.1f}%"
              f"{majb:>7.1f}%{binned - majb:>+8.1f}{unparsed[qtype]:>5}{imgs_per_type[qtype]:>6}")
    total_n, total_h = sum(seen.values()), sum(hits.values())
    total_maj = sum(majority.values())
    overall = total_h / max(1, total_n) * 100
    maj_overall = total_maj / max(1, total_n) * 100
    total_b = sum(hits_binned.values())
    total_majb = sum(majority_binned.values())
    binned_overall = total_b / max(1, total_n) * 100
    majb_overall = total_majb / max(1, total_n) * 100
    print("-" * 71)
    print(f"{'OVERALL':<14}{total_n:>6}{overall:>7.1f}%{maj_overall:>6.1f}%"
          f"{binned_overall:>8.1f}%{majb_overall:>7.1f}%{binned_overall - majb_overall:>+8.1f}"
          f"{sum(unparsed.values()):>5}")
    print()
    if args.count_prompt == "bucket":
        share = (bucket_form_preds / seen["count"] * 100) if seen.get("count") else 0.0
        print(f"  {share:.0f}% of counting answers came back as a bucket label.")
        print("  The EXACT column is meaningless for this adapter and will look bad:")
        print("  gold is a raw integer, the answer is a bucket, so they never match")
        print("  by string. Read the BINNED column, which is the published protocol")
        print("  and is what the 84.5 of the integer adapter was measured on too.")
        if share < 95:
            print("  WARNING: under 95% bucket form - the adapter is half-reverting to")
            print("  integers. Those still bin correctly, but the training did not take.")
        print()
    print("  exact / maj    exact string match, counts against 456 raw integers")
    print("  binned / maj-b published RSVQA protocol, counts in 5 buckets")
    print("  delta          binned score minus the binned majority baseline")
    print()
    print("  Quote the BINNED column against any published figure, and never quote")
    print("  either without its majority baseline beside it.")
    if binned_overall <= majb_overall:
        print()
        print("  NOTE: at or below the majority-class baseline on the published metric.")
        print("  The model is not yet reading these images usefully. Expected before")
        print("  fine-tuning; a red flag after it.")
    print(f"\n  {elapsed / 60:.1f} min · {total_n / elapsed:.1f} q/s")

    if sum(unparsed.values()) > total_n * 0.02:
        print("\n  WARNING: more than 2% of answers could not be parsed. That is a prompt "
              "problem, not a model problem. Read the raw outputs above before "
              "believing this score.")

    summary = {
        "run": tag, "adapter": args.adapter, "dtype": args.dtype,
        "max_pixels_tokens": args.max_pixels,
        "count_prompt": args.count_prompt,
        "count_answers_in_bucket_form": (
            round(bucket_form_preds / seen["count"] * 100, 2)
            if seen.get("count") else None),
        "n": total_n, "overall_accuracy": round(overall, 2),
        "elapsed_min": round(elapsed / 60, 2),
        "overall_accuracy_binned": round(binned_overall, 2),
        "majority_baseline": round(maj_overall, 2),
        "majority_baseline_binned": round(majb_overall, 2),
        "beats_majority_by": round(overall - maj_overall, 2),
        "beats_majority_by_binned": round(binned_overall - majb_overall, 2),
        "per_type": {t: {"n": seen[t], "correct": hits[t],
                         "accuracy": round(hits[t] / seen[t] * 100, 2),
                         "accuracy_binned": round(hits_binned[t] / seen[t] * 100, 2),
                         "majority": round(majority[t] / seen[t] * 100, 2),
                         "majority_binned": round(majority_binned[t] / seen[t] * 100, 2),
                         "images_sampled": imgs_per_type[t],
                         "unparsed": unparsed[t]} for t in sorted(seen)},
    }
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  per-question -> {args.out}.jsonl")
    print(f"  summary      -> {args.out}.json")


if __name__ == "__main__":
    sys.exit(main())
