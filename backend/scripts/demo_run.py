"""Gate I: the demo query set, end to end, twice, unattended - with replay.

    python scripts/demo_run.py            # on the office box, after demo_env.ps1

Runs every demo query through `python -m satquery.run` with all four adapters
resident (pass 1), then runs each recorded trace back through `--replay`
(pass 2), which re-executes the plan and compares the output byte-for-byte.
A demo that cannot replay itself is a demo that may not repeat on stage.
Prints one line per query with the tool chain, the adapter that answered and
the wall time, and exits non-zero if any run fails, abstains unexpectedly, or
does not replay. Writes runs/demo/<n>_<slug>.json and runs/demo/REPORT.md.

The query set covers all five capabilities with the fixtures the plumbing
checks built: a BigEarthNet tile (VQA, grounding, fusion with its SAR pair)
and a SECOND pre/post pair (change). Timings include the model load, since
each run is a fresh process, exactly as the CLI demo would be.
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("SIH_ROOT", r"F:\sih")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
OUT = os.path.join(ROOT, "runs", "demo")
os.makedirs(OUT, exist_ok=True)

OPT = os.path.join(ROOT, "e2e", "gf", "opt.tif")
SAR = os.path.join(ROOT, "e2e", "gf", "sar.tif")
PRE = os.path.join(ROOT, "e2e", "pair", "pre.png")
POST = os.path.join(ROOT, "e2e", "pair", "post.png")
LR = os.path.join(ROOT, "e2e", "232.tif")

QUERIES = [
    ("vqa_presence",   "Is there a road?",                                     [LR]),
    ("vqa_count",      "How many buildings are there?",                       [LR]),
    ("vqa_rural",      "Is it a rural or an urban area?",                     [LR]),
    ("vqa_ben",        "Is there any water in the image?",                    [OPT]),
    ("ground_ref",     "Where is the largest connected region of pastures?",  [OPT]),
    ("change_yesno",   "Did the areas of buildings increase?",                [PRE, POST]),
    ("change_what",    "What is the largest change?",                         [PRE, POST]),
    ("fusion_water",   "Is there any water in the image?",                    [OPT, SAR]),
    ("fusion_mcq",     "Which class covers most of the image? a) Arable land, b) Pastures, c) Broad-leaved forest, d) Water", [OPT, SAR]),
]

REQUIRED = {"SATQUERY_BACKEND", "SATQUERY_VQA_ADAPTER", "SATQUERY_CHANGE_ADAPTER",
            "SATQUERY_GROUND_ADAPTER", "SATQUERY_FUSION_ADAPTER"}
missing = REQUIRED - set(os.environ)
if missing:
    print("demo env not loaded - missing", sorted(missing), "- dot-source scripts/demo_env.ps1 first")
    sys.exit(2)


def run(args, timeout=900):
    t0 = time.time()
    r = subprocess.run([PY, "-m", "satquery.run", *args], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return r, time.time() - t0


rows, failures = [], 0
print("PASS 1 - live runs")
for n, (slug, q, imgs) in enumerate(QUERIES, 1):
    out = os.path.join(OUT, "%02d_%s.json" % (n, slug))
    r, secs = run(["--query", q, *imgs, "--out", out])
    if r.returncode:
        failures += 1
        print("  FAIL %-14s rc=%d %s" % (slug, r.returncode, r.stderr[-300:].replace("\n", " | ")))
        rows.append((slug, "FAIL", "", "", secs, ""))
        continue
    t = json.load(open(out))
    qty = t["output"].get("quantities") or {}
    tools = [s["tool"] for s in t["steps"]]
    stubs = [s["tool"] for s in t["steps"] if s["stub"]]
    real = [s["tool"] for s in t["steps"] if not s["stub"]]
    ok = not t["abstained"] and any(x.split(".")[0] in ("vqa", "ground", "change", "fusion") for x in real)
    if not ok:
        failures += 1
    print("  %s %-14s %-9s %-40s adapter=%-16s %5.1fs  %s" % (
        "ok  " if ok else "FAIL", slug, t["classified_task"], "+".join(tools), qty.get("adapter") or "-",
        secs, t["output"]["text"][:50].replace("\n", " ")))
    rows.append((slug, t["classified_task"], "+".join(real), qty.get("adapter") or "-", secs,
                 t["output"]["text"][:60].replace("\n", " ")))

print("\nPASS 2 - replay every trace")
replayed = 0
for n, (slug, q, imgs) in enumerate(QUERIES, 1):
    out = os.path.join(OUT, "%02d_%s.json" % (n, slug))
    if not os.path.exists(out):
        continue
    # A trace records file NAMES only (no machine paths in the graded
    # artefact), so replay is handed the files again and checks their sha256.
    r, secs = run(["--replay", out, *imgs])
    good = r.returncode == 0
    replayed += good
    if not good:
        failures += 1
    print("  %s %-14s %5.1fs %s" % ("ok  " if good else "FAIL", slug, secs,
                                    "" if good else (r.stdout + r.stderr)[-300:].replace("\n", " | ")))

lines = ["# Demo run - %s" % time.strftime("%Y-%m-%d %H:%M"), "",
         "| query | task | real tools | adapter | seconds | answer |", "|---|---|---|---|---:|---|"]
for slug, task, real, ad, secs, text in rows:
    lines.append("| %s | %s | %s | %s | %.1f | %s |" % (slug, task, real, ad, secs, text))
lines += ["", "replayed %d / %d traces byte-identical; failures: %d" % (replayed, len(rows), failures)]
open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n" + lines[-1])
sys.exit(1 if failures else 0)
