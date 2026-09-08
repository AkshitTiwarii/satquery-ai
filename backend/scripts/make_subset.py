"""Cut a training subset out of BigEarthNet.txt.

BigEarthNet.txt is 9,553,962 annotations over 464,044 patches. Annotations are
dense per patch (about 20 each), and 76% of them are binary/mcq questions, so an
uncapped sample drowns captioning and referring expressions. This script selects
a country/season cell (or the whole corpus), caps annotations per patch while
guaranteeing at least one of every present type, and writes:

    <out>              one JSON object per line, ready for a training loader
    <out>.patches.txt  the patch ids, for `tar --files-from` when extracting
                       imagery out of the Zenodo archives

Deterministic for a given --seed.
"""

import argparse
import json
import random

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS = [
    "patch_id", "s1_name", "input", "output", "type", "category",
    "latitude", "longitude", "country", "season", "climate_zone",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", default="data/BigEarthNet.txt.parquet")
    p.add_argument("--split", default="train",
                   choices=["train", "validation", "test", "bench"])
    p.add_argument("--country", help="e.g. Lithuania; omit for all 10")
    p.add_argument("--season", help="Spring | Summer | Fall | Winter; omit for all")
    p.add_argument("--max-patches", type=int, default=50000)
    p.add_argument("--per-patch", type=int, default=6,
                   help="cap on annotations kept per patch")
    p.add_argument("--types", default="binary,mcq,captioning",
                   help="comma separated subset of: binary,mcq,captioning,bounding box")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default="data/subset_train.jsonl")
    return p.parse_args()


def main():
    a = parse_args()

    table = pq.read_table(a.parquet, columns=COLUMNS + ["split"])

    mask = pc.equal(table["split"], a.split)
    if a.country:
        mask = pc.and_(mask, pc.equal(table["country"], a.country))
    if a.season:
        mask = pc.and_(mask, pc.equal(table["season"], a.season))

    wanted_types = [t.strip() for t in a.types.split(",") if t.strip()]
    mask = pc.and_(mask, pc.is_in(table["type"], value_set=pa.array(wanted_types)))

    rows = table.filter(mask).select(COLUMNS).to_pylist()
    if not rows:
        raise SystemExit(
            f"no rows for split={a.split} country={a.country} season={a.season} "
            f"types={wanted_types} — check the spelling against §3.1 of the runbook"
        )

    by_patch = {}
    for row in rows:
        by_patch.setdefault(row["patch_id"], []).append(row)

    rng = random.Random(a.seed)
    patches = sorted(by_patch)
    rng.shuffle(patches)
    patches = patches[: a.max_patches]

    written = 0
    kept_types = {}
    with open(a.out, "w", encoding="utf-8") as fh:
        for patch_id in patches:
            group = by_patch[patch_id]
            order = list(range(len(group)))
            rng.shuffle(order)

            # one of every type present, then top up at random
            first_of_type, top_up = [], []
            seen = set()
            for i in order:
                if group[i]["type"] in seen:
                    top_up.append(i)
                else:
                    seen.add(group[i]["type"])
                    first_of_type.append(i)

            chosen = first_of_type[: a.per_patch]
            chosen += top_up[: max(0, a.per_patch - len(chosen))]

            for i in chosen:
                row = group[i]
                kept_types[row["type"]] = kept_types.get(row["type"], 0) + 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    allowlist = a.out.rsplit(".", 1)[0] + ".patches.txt"
    with open(allowlist, "w", encoding="utf-8") as fh:
        fh.write("\n".join(patches) + "\n")

    print(f"{len(patches)} patches, {written} samples -> {a.out}")
    print(f"patch allowlist -> {allowlist}")
    for t in sorted(kept_types, key=kept_types.get, reverse=True):
        print(f"  {t:<14} {kept_types[t]:>8}")


if __name__ == "__main__":
    main()
