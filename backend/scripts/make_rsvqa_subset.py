"""Cut a training subset out of RSVQA-LR's own train split.

    python scripts/make_rsvqa_subset.py --data-dir data/rsvqa_lr --limit 24000 \
        --out data/rsvqa_train.jsonl

WHY THIS SPLIT AND NOT BigEarthNet.txt
--------------------------------------
The first QLoRA (Kaggle v8, 3 Sep) trained on 24,000 BigEarthNet.txt
binary/MCQ questions and moved RSVQA-LR from 60.75 to 62.0 - a yes-bias shift,
not perception (PLAN section 9). RSHallu reached 92.11 on the same test set with
this exact LoRA config by training on RSVQA-LR's OWN train split. Same recipe,
different data, so this script produces rows in the shape the notebook already
consumes: patch_id / input / output / type.

TWO THINGS THAT WOULD QUIETLY RUIN THE NUMBER
---------------------------------------------
1. The eval harness wraps every question in a per-type instruction
   ("Answer with one word: yes or no."). Train on the bare question and the
   model learns one format and is scored on another. The templates are
   imported from eval_rsvqa_lr.py, not copied, so they cannot drift.
2. Image leakage. RSVQA-LR is split BY IMAGE (572 train, 100 test; the ids
   are not a clean range, so never assume one). The script asserts the train
   rows share no image with the test split when the test file is present.

The subset is balanced ACROSS TYPES, not sampled at the split's own ratio.
rural_urban is under 1% of the split (one question per image) and is kept
whole; the other three share the remainder equally, strided across images so
no type is drawn from a handful of pictures. 24,000 rows is v8's budget:
1,500 steps x 16 accumulation fit in 10.5 h on a T4.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_rsvqa_lr import (COUNT_LABELS, PROMPTS,  # noqa: E402  the harness's own,
                           bin_count, prompts_for)  # imported so they cannot drift


def load(data_dir, split):
    with open(os.path.join(data_dir, f"LR_split_{split}_questions.json"), encoding="utf-8") as fh:
        questions = [q for q in json.load(fh)["questions"] if q.get("active")]
    with open(os.path.join(data_dir, f"LR_split_{split}_answers.json"), encoding="utf-8") as fh:
        answers = [a for a in json.load(fh)["answers"] if a.get("active")]
    gold = {a["question_id"]: str(a["answer"]).strip().lower() for a in answers}
    rows = []
    for q in questions:
        if q["id"] in gold and q["type"] in PROMPTS:
            rows.append({"qid": q["id"], "img_id": q["img_id"], "type": q["type"],
                         "question": q["question"].strip().rstrip("?"),
                         "gold": gold[q["id"]]})
    return rows


def balanced(rows, limit, seed):
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    rng = random.Random(seed)
    keep = {"rural_urban": by_type.pop("rural_urban", [])}
    remaining = max(0, limit - len(keep["rural_urban"]))
    per = remaining // max(1, len(by_type))
    for t, pool in by_type.items():
        # stride first so every image contributes, then shuffle within the pick
        step = max(1, len(pool) // per) if per else 1
        pick = pool[::step][:per] if per else []
        rng.shuffle(pick)
        keep[t] = pick
    out = [r for t in sorted(keep) for r in keep[t]]
    rng.shuffle(out)
    return out


def apply_count_answer(rows):
    """Rewrite counting answers as bucket labels, in place. -> the bucket census.

    Counting is scored on five buckets, so teach the five buckets. Trained on
    raw integers the model has to regress an exact number and is then graded on
    whichever bucket it lands in: of the 19 test questions whose gold count sits
    in 11-100, the integer adapter placed exactly one of them there. The label
    is what the published protocol asks for, and eval_rsvqa_lr.py reads either
    form, so a bucket adapter and an integer adapter stay comparable on the
    binned column. Rows of every other type are left exactly as they were.
    """
    for r in rows:
        if r["type"] == "count":
            label = bin_count(r["gold"])
            assert label is not None, \
                f"count row {r['qid']} has non-numeric gold {r['gold']!r}"
            r["gold"] = label
    return Counter(r["gold"] for r in rows if r["type"] == "count")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/rsvqa_lr")
    ap.add_argument("--limit", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/rsvqa_train.jsonl")
    ap.add_argument("--count-answer", choices=["bucket", "integer"], default="bucket",
                    help="what a counting row is taught to answer. 'bucket' writes the "
                         "published five-bucket label (0 / 1-10 / 11-100 / 101-1000 / "
                         ">1000), which is what the protocol scores; 'integer' writes "
                         "the raw number, reproducing the v2 and joint_v3 adapters.")
    args = ap.parse_args()

    train = load(args.data_dir, "train")
    print(f"train split: {len(train)} active questions over "
          f"{len({r['img_id'] for r in train})} images")
    print("  by type:", dict(sorted(Counter(r["type"] for r in train).items())))

    test_q = os.path.join(args.data_dir, "LR_split_test_questions.json")
    if os.path.exists(test_q):
        test = load(args.data_dir, "test")
        overlap = {r["img_id"] for r in train} & {r["img_id"] for r in test}
        assert not overlap, f"train and test share {len(overlap)} images - refusing to leak"
        print(f"  no image overlap with the {len({r['img_id'] for r in test})}-image test split")
    else:
        print("  WARNING: test split not present, image-leak check skipped")

    rows = balanced(train, args.limit, args.seed)
    print(f"subset: {len(rows)} rows over {len({r['img_id'] for r in rows})} images")
    print("  by type:", dict(sorted(Counter(r["type"] for r in rows).items())))
    yes = Counter(r["gold"] for r in rows if r["type"] in ("presence", "comp"))
    print("  yes/no balance in presence+comp:", dict(yes))

    if args.count_answer == "bucket":
        buckets = apply_count_answer(rows)
        print("  count answers as buckets:",
              {k: buckets[k] for k in COUNT_LABELS if k in buckets})
        missing = [k for k in COUNT_LABELS if k not in buckets]
        if missing:
            print(f"  NOTE: no training row lands in {missing} - the model "
                  f"cannot learn a bucket it never sees")
    else:
        print("  count answers as raw integers (v2 / joint_v3 form)")

    # The instruction has to match the answer. Teaching "11-100" under
    # "Answer with a single number and nothing else." trains the model to
    # disobey its own prompt, so the template travels with the answer form
    # and the harness is run with the matching --count-prompt.
    templates, _ = prompts_for("bucket" if args.count_answer == "bucket" else "number")
    print("  counting instruction:", templates["count"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "patch_id": str(r["img_id"]), "type": r["type"], "qid": r["qid"],
                "input": templates[r["type"]].format(q=r["question"]),
                "output": r["gold"],
            }) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
