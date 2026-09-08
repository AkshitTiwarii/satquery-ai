"""Fetch VRSBench's validation split - the third grading split the statement names.

    python scripts/get_vrsbench.py --dest F:\\sih\\data\\vrsbench

Four files, 4.0 GB, from the authors' Hugging Face repo (xiang709/VRSBench):

  VRSBench_EVAL_vqa.json         9.4 MB   37,409 questions over 9,349 images
  VRSBench_EVAL_referring.json  10.3 MB   16,159 referring expressions
  VRSBench_EVAL_Cap.json         4.8 MB   captions (not scored yet)
  Images_val.zip                 4.0 GB   9,350 PNGs at 512x512

The zip is NOT extracted. 9,350 files at 512x512 decode to about 7 GB and the
harness reads them from the archive one at a time, so extracting would cost
disk for nothing. eval_vrsbench.py opens it lazily and per process, because a
ZipFile handle does not survive a DataLoader fork.

Resumable: hf_hub_download skips a file whose size and hash already match, so
re-running after a dropped connection costs a HEAD request per file rather
than the 4 GB again.
"""

import argparse
import os

REPO = "xiang709/VRSBench"
FILES = [
    ("VRSBench_EVAL_vqa.json", 9_358_245),
    ("VRSBench_EVAL_referring.json", 10_277_680),
    ("VRSBench_EVAL_Cap.json", 4_809_521),
    ("Images_val.zip", 3_976_656_690),
]
# The TRAIN split, for --split train: 20,264 images and 142,390 LLaVA-style
# conversations (caption / vqa / refer). 8.4 GB. Not extracted either.
TRAIN_FILES = [
    ("VRSBench_train.json", 64_923_606),
    ("Images_train.zip", 8_359_313_269),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=r"F:\sih\data\vrsbench")
    ap.add_argument("--annotations-only", action="store_true",
                    help="skip the 4 GB image archive")
    ap.add_argument("--split", choices=["val", "train", "both"], default="val",
                    help="val = the EVAL files (default); train adds Images_train.zip + "
                         "VRSBench_train.json (8.4 GB); both = everything")
    args = ap.parse_args()
    files = {"val": FILES, "train": TRAIN_FILES, "both": FILES + TRAIN_FILES}[args.split]

    from huggingface_hub import hf_hub_download

    os.makedirs(args.dest, exist_ok=True)
    for name, expect in files:
        if args.annotations_only and name.endswith(".zip"):
            print(f"  {name:30} skipped (--annotations-only)")
            continue
        target = os.path.join(args.dest, name)
        if os.path.exists(target) and os.path.getsize(target) == expect:
            print(f"  {name:30} already here, {expect / 1e6:9.1f} MB")
            continue
        print(f"  {name:30} downloading {expect / 1e6:9.1f} MB ...", flush=True)
        hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset",
                        local_dir=args.dest)
        got = os.path.getsize(target)
        # A truncated download is the failure that matters: the annotations
        # would still be valid JSON prefixes to nothing, and a short zip reads
        # as "image missing" for thousands of rows rather than as an error.
        if got != expect:
            raise SystemExit(
                f"{name}: got {got:,} bytes, expected {expect:,}. "
                "Delete it and re-run rather than scoring against a partial file.")
        print(f"  {name:30} done, {got / 1e6:9.1f} MB")

    print("\nVRSBench ready at", args.dest)
    print("  score with: python scripts/eval_vrsbench.py --task vqa --limit 400")


if __name__ == "__main__":
    main()
