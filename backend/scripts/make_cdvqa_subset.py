"""Cut a training subset out of CDVQA's train split for the change adapter.

    python scripts/make_cdvqa_subset.py --data-dir data/cdvqa --limit 20000 \
        --out data/cdvqa_train.jsonl

Rows come out in the shape the training notebook already consumes -
patch_id / input / output / type - with patch_id the SECOND file name, which
the pair loader resolves to (im1/name, im2/name).

WHAT THE SPLIT LOOKS LIKE, MEASURED
------------------------------------
Train: 25,600 image-id entries over 1,600 FILES - each file appears sixteen
times with rephrased questions ("...changed?" / "...changed in the first
image?"). So "one row per image id" would be sixteen near-copies of one
picture; the subset strides over FILES. Eight question types with closed
vocabularies (yes/no; six land-cover classes; eleven ratio bins). Priors are
strong - change_ratio_types is 47% '0' - so the subset is balanced across
types and the harness prints a majority baseline beside every number.

The prompt templates are imported from eval_cdvqa.py, not copied, and the
harness prepends the same pre/post sentence, so train and test agree. A test
in tests/ pins the engine's copy to these.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cdvqa import PAIR_PREFIX, PROMPTS, load_split  # noqa: E402


def _round_robin_files(pool, want):
    """`want` rows out of `pool`, striding over FILES so that every picture
    contributes once before any picture contributes twice."""
    by_file = defaultdict(list)
    for r in pool:
        by_file[r["file"]].append(r)
    files = sorted(by_file)
    pick = []
    i = 0
    while len(pick) < min(want, len(pool)):
        f = files[i % len(files)]
        k = i // len(files)
        if k < len(by_file[f]):
            pick.append(by_file[f][k])
        i += 1
        if i > len(pool) * 2:
            break
    return pick


def balanced(rows, limit, seed, balance_answers=True):
    """`limit` rows spread evenly across type and, when balance_answers, across
    the gold ANSWERS within each type.

    Balancing types alone is not enough. The class and ratio types carry a
    strong answer prior, and a model that cannot tell the classes apart scores
    best by playing it: trained on a mix whose most common smallest_change
    answer is 'trees', the v4 adapter answered 'trees' to 49 of the 50
    smallest_change test questions and scored 14.0 against a 34.0 majority.
    Flattening the answer distribution removes the reward for collapsing, so
    the only way left to gain is to tell the classes apart. It cannot
    manufacture that ability - the best published result for this type is 35 -
    but it stops us paying the model for guessing.

    Rare answers (water, playgrounds) are taken whole and their unused share
    passes to the answers that still have rows, so the total is unchanged.
    """
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    rng = random.Random(seed)
    per = limit // len(by_type)
    out = []
    for t in sorted(by_type):
        pool = by_type[t]
        if not balance_answers:
            pick = _round_robin_files(pool, per)
        else:
            by_ans = defaultdict(list)
            for r in pool:
                by_ans[r["gold"]].append(r)
            # Smallest pools first, so an answer that cannot fill its share
            # hands the remainder to the answers that still can.
            order = sorted(by_ans, key=lambda a: (len(by_ans[a]), a))
            pick, left, remaining = [], per, len(order)
            for ans in order:
                share = left // remaining
                got = _round_robin_files(by_ans[ans], share)
                pick.extend(got)
                left -= len(got)
                remaining -= 1
        rng.shuffle(pick)
        out.extend(pick)
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cdvqa")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/cdvqa_train.jsonl")
    ap.add_argument("--balance-answers", dest="balance_answers",
                    action="store_true", default=True,
                    help="flatten the gold-answer distribution inside each type "
                         "(default). The v4 recipe balanced types only, and the "
                         "adapter collapsed onto each type's most common answer.")
    ap.add_argument("--no-balance-answers", dest="balance_answers",
                    action="store_false",
                    help="the v4 recipe: types balanced, answers left at their "
                         "natural prior. Kept so v4 can be reproduced exactly.")
    args = ap.parse_args()

    train = load_split(args.data_dir, "Train")
    files = {r["file"] for r in train}
    print(f"train split: {len(train)} active questions over {len(files)} files")
    print("  by type:", dict(sorted(Counter(r["type"] for r in train).items())))

    test_q = os.path.join(args.data_dir, "Test_questions.json")
    if os.path.exists(test_q):
        test_files = {r["file"] for r in load_split(args.data_dir, "Test")}
        overlap = files & test_files
        assert not overlap, f"train and test share {len(overlap)} files - refusing to leak"
        print(f"  no file overlap with the {len(test_files)}-file test split")
    else:
        print("  WARNING: test split not present, leak check skipped")

    rows = balanced(train, args.limit, args.seed,
                    balance_answers=args.balance_answers)
    print(f"subset: {len(rows)} rows over {len({r['file'] for r in rows})} files")
    print("  by type:", dict(sorted(Counter(r["type"] for r in rows).items())))
    yn = Counter(r["gold"] for r in rows if r["gold"] in ("yes", "no"))
    print("  yes/no balance:", dict(yn))
    print(f"  answers balanced within each type: {args.balance_answers}")
    print("  worst answer share per type (the prior a lazy model would play):")
    by_type_ans = defaultdict(Counter)
    for r in rows:
        by_type_ans[r["type"]][r["gold"]] += 1
    for t in sorted(by_type_ans):
        c = by_type_ans[t]
        n = sum(c.values())
        top, k = c.most_common(1)[0]
        print(f"    {t:20s} {len(c):2d} answers, top {top!r} at {100 * k / n:4.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "patch_id": r["file"], "type": r["type"], "qid": r["qid"],
                "input": PAIR_PREFIX + PROMPTS[r["type"]].format(q=r["question"]),
                "output": r["gold"],
            }) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
