"""Cut a VRSBench TRAIN subset in the shape the training notebook eats.

    python scripts/make_vrsbench_subset.py --train-json VRSBench_train.json --task vqa|refer \
        --limit 30000 --eval-json VRSBench_EVAL_vqa.json --out rows.jsonl

VRSBench_train.json is LLaVA-style: 142,390 conversations over 20,264 images, each a
"<image>\\n[tag] question" / answer pair, tag in {caption, vqa, refer}. Referring answers are
"{<x1><y1><x2><y2>}" on a 0-100 integer grid (trap 1 in PLAN.md section 10: BigEarthNet is
0-1). This script keeps the dataset's own answer strings and asks in the EVAL HARNESS'S OWN
prompt templates, imported from scripts/eval_vrsbench.py, so train and test are worded alike
and the adapter is scored in the format it learned.

Rows out: {"patch_id": <image filename in Images_train.zip>, "input": <prompt>,
           "output": <answer string>, "type": "vqa"|"refer", "tag": <sub-type if known>}
`patch_id` is the key the notebook's loader uses everywhere; here it is a PNG name.

Three refusals, each one a run saved:
  * an image that also appears in the EVAL split (--eval-json) - leakage, refuse;
  * a referring answer whose box does not parse or leaves 0-100 - refuse the row;
  * a task tag the file does not carry - refuse, do not silently emit an empty file.
"""
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_vrsbench import REF_PROMPTS, VQA_PROMPT, parse_gold_box  # noqa: E402

TAG_RE = re.compile(r"^\s*<image>\s*\[(\w+)\]\s*(.*)$", re.S)
P_RE = re.compile(r"<p>(.*?)</p>", re.S)


def parse_conversation(rec):
    """-> (tag, question, answer) or None when the record is not a two-turn pair."""
    conv = rec.get("conversations") or []
    if len(conv) < 2 or conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
        return None
    m = TAG_RE.match(conv[0].get("value", ""))
    if not m:
        return None
    return m.group(1).lower(), m.group(2).strip(), str(conv[1].get("value", "")).strip()


def refer_expression(question):
    """The phrase between <p>...</p>, else the question itself."""
    m = P_RE.search(question)
    return (m.group(1) if m else question).strip().rstrip("?").strip()


def build_rows(records, task):
    want = "vqa" if task == "vqa" else "refer"
    rows, dropped = [], collections.Counter()
    for rec in records:
        parsed = parse_conversation(rec)
        if parsed is None:
            dropped["unparsed_conversation"] += 1
            continue
        tag, q, a = parsed
        if tag != want:
            continue
        image = rec.get("image")
        if not image:
            dropped["no_image"] += 1
            continue
        if task == "vqa":
            if not q or not a:
                dropped["empty"] += 1
                continue
            rows.append({"patch_id": image, "input": VQA_PROMPT.format(q=q.rstrip("?").strip() + "?"),
                         "output": a, "type": "vqa", "tag": rec.get("type", "")})
        else:
            try:
                box = parse_gold_box(a)
            except ValueError:
                dropped["box_out_of_range"] += 1
                continue
            if box is None:
                dropped["box_unparsed"] += 1
                continue
            expr = refer_expression(q)
            if not expr:
                dropped["empty"] += 1
                continue
            rows.append({"patch_id": image, "input": REF_PROMPTS["vrsbench"].format(q=expr),
                         "output": a, "type": "refer", "tag": rec.get("type", "")})
    return rows, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--task", choices=["vqa", "refer"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = every row of that task")
    ap.add_argument("--eval-json", action="append", default=[],
                    help="EVAL split file(s); any shared image id is leakage and the run refuses")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.train_json, encoding="utf-8") as fh:
        records = json.load(fh)
    rows, dropped = build_rows(records, a.task)
    if not rows:
        raise SystemExit("no [%s] rows in %s - wrong file, or the tag format changed" % (a.task, a.train_json))

    eval_ids = set()
    for p in a.eval_json:
        with open(p, encoding="utf-8") as fh:
            eval_ids |= {r["image_id"] for r in json.load(fh)}
    if eval_ids:
        leak = {r["patch_id"] for r in rows} & eval_ids
        if leak:
            raise SystemExit("%d train images also appear in the EVAL split, e.g. %s - refusing"
                             % (len(leak), sorted(leak)[:3]))

    random.Random(a.seed).shuffle(rows)
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    hist = collections.Counter(r["tag"] or "-" for r in rows)
    print("%s rows: %d over %d images -> %s" % (a.task, len(rows), len({r["patch_id"] for r in rows}), a.out))
    print("  by sub-type:", dict(hist.most_common()))
    print("  dropped:", dict(dropped) or "none")
    print("  eval-split overlap: 0 of %d eval images" % len(eval_ids) if eval_ids else "  eval overlap NOT checked")
    print("  example:", json.dumps(rows[0])[:300])


if __name__ == "__main__":
    main()
