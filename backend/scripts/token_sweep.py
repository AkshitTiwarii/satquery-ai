"""Visual-token sweep: the same questions at 128 / 256 / 512 / 1024 tokens per image.

    python scripts/token_sweep.py                    # every harness that has an adapter
    python scripts/token_sweep.py --only rsvqa ground

Answers Akshit's question ("is 256 right?") on the FINE-TUNED adapters, not the
base model, using the harnesses' own --max-pixels flag so train/score stay in
one frame. Runs on the office 4060 in nf4 (inference at 1024 tokens with two
images fits: 16 images at 1,366 tokens measured at 2.85 GB). Writes
runs/sweep/<harness>_<cap>.json per cell and one table to runs/sweep/SUMMARY.md.

What to expect, and why it is still worth measuring: BigEarthNet and RSVQA-LR
tiles are 120-256 px, i.e. 64-81 tokens native, so their curves must go flat
above 256 by construction - a real result for those benchmarks. The SECOND
pairs (512 px) are the only rows where 512 and 1024 can differ.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("SIH_ROOT", r"F:\sih")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
OUT = os.path.join(ROOT, "runs", "sweep")
CAPS = [128, 256, 512, 1024]

HARNESSES = {
    "rsvqa": {
        "script": "scripts/eval_rsvqa_lr.py",
        "adapter": os.path.join(ROOT, "runs", "lora_rsvqa_v2"),
        "args": ["--limit", "400"],
        "metric": ("overall_accuracy_binned", "majority_baseline_binned"),
        "label": "RSVQA-LR binned acc",
    },
    "ground": {
        "script": "scripts/eval_ground.py",
        "adapter": os.path.join(ROOT, "runs", "lora_ground_v3"),
        "args": ["--limit", "400"],
        "metric": ("acc50", "whole_acc50"),
        "label": "BEN grounding Acc@0.5",
    },
    "cdvqa": {
        "script": "scripts/eval_cdvqa.py",
        "adapter": os.path.join(ROOT, "runs", "lora_cdvqa_v4"),
        "args": ["--limit", "400"],
        "metric": ("overall_accuracy", "majority_baseline"),
        "label": "CDVQA exact acc (pairs, 512 px source)",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--caps", nargs="*", type=int, default=CAPS)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = args.only or list(HARNESSES)
    results = {}
    for name in names:
        h = HARNESSES[name]
        if not os.path.isfile(os.path.join(h["adapter"], "adapter_config.json")):
            print(f"[{name}] no adapter at {h['adapter']} - skipped")
            continue
        for cap in args.caps:
            out = os.path.join(OUT, f"{name}_{cap}")
            if os.path.exists(out + ".json"):
                print(f"[{name} @ {cap}] cached")
            else:
                t0 = time.time()
                cmd = [PY, "-u", h["script"], "--adapter", h["adapter"], "--max-pixels", str(cap),
                       "--out", out, *h["args"]]
                print(f"[{name} @ {cap}] running ...", flush=True)
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=7200)
                if r.returncode:
                    print(f"[{name} @ {cap}] FAILED rc={r.returncode}\n{r.stderr[-1500:]}")
                    continue
                print(f"[{name} @ {cap}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
            d = json.load(open(out + ".json"))
            m, b = h["metric"]
            results.setdefault(name, {})[cap] = (d[m], d[b], d.get("elapsed_min"))

    lines = ["# Visual-token sweep, fine-tuned adapters, office 4060 nf4, 400 questions each", ""]
    for name, per in results.items():
        h = HARNESSES[name]
        lines.append(f"## {h['label']}")
        lines.append("| tokens/image | score | baseline | minutes |")
        lines.append("|---:|---:|---:|---:|")
        for cap in sorted(per):
            s, b, el = per[cap]
            lines.append(f"| {cap} | {s} | {b} | {el} |")
        lines.append("")
    open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
