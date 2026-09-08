"""End-to-end check of the vqa.rs adapter THROUGH THE REAL ENTRY POINT.

    python scripts/e2e_vqa_adapter.py [N]      # N = first N cases only

Runs on the office box (SIH_ROOT, default F:\sih) against e2e_cases.json,
a list of real RSVQA-LR test questions with gold. First run 4 Sep 2026:
13/14 with the adapter, 7/14 base - docs/results/e2e_vqa_adapter_2026-09-04.txt.

Runs `python -m satquery.run` as a subprocess per question on the office box,
reads the trace it writes, and compares the answer with the RSVQA-LR gold.
Then the same questions with the adapter unset, so the delta is the adapter's
and nothing else. Finally one grounding query WITH the adapter loaded, to show
ground.rs still runs on base weights (the disable_adapter path) and does not
crash.
"""

import json
import os
import re
import subprocess
import sys
import time
import zipfile

ROOT = os.environ.get("SIH_ROOT", r"F:\sih")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
ADAPTER = os.path.join(ROOT, "runs", "lora_rsvqa_v2")
# The engine's default is the hub id, which on this box means a 7.5 GB
# download into the HF cache. The weights are already here.
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-VL-3B-Instruct")
E2E = os.path.join(ROOT, "e2e")
os.makedirs(E2E, exist_ok=True)

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cases = json.load(open(os.path.join(ROOT, "e2e_cases.json")))
if LIMIT:
    cases = cases[:LIMIT]

# images out of the zip, by name, exactly as the harness does it
z = zipfile.ZipFile(os.path.join(ROOT, "data", "rsvqa_lr", "Images_LR.zip"))
names = {os.path.splitext(os.path.basename(n))[0]: n
         for n in z.namelist() if n.lower().endswith(".tif")}
for i in sorted({c["img_id"] for c in cases}):
    p = os.path.join(E2E, "%d.tif" % i)
    if not os.path.exists(p):
        open(p, "wb").write(z.read(names[str(i)]))


def norm(text, kind):
    t = (text or "").strip().lower()
    if kind in ("presence", "comp"):
        m = re.search(r"\byes\b|\bno\b", t)
        return m.group() if m else None
    if kind == "rural_urban":
        m = re.search(r"\brural\b|\burban\b", t)
        return m.group() if m else None
    m = re.search(r"-?\d+", t)
    return str(int(m.group())) if m else None


def binned(v):
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    for lo, hi, lab in [(0, 0, "0"), (1, 10, "1-10"), (11, 100, "11-100"),
                        (101, 1000, "101-1000"), (1001, 10**9, ">1000")]:
        if lo <= n <= hi:
            return lab


def run(query, img, out, adapter):
    env = dict(os.environ, SATQUERY_BACKEND="real", SATQUERY_QWEN=MODEL_DIR, PYTHONIOENCODING="utf-8")
    if adapter:
        env["SATQUERY_VQA_ADAPTER"] = ADAPTER
    else:
        env.pop("SATQUERY_VQA_ADAPTER", None)
    t0 = time.time()
    r = subprocess.run([PY, "-m", "satquery.run", "--query", query, img, "--out", out],
                       cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if r.returncode != 0:
        print("RUN FAILED rc=%d\n%s" % (r.returncode, r.stderr[-3000:]))
        sys.exit(1)
    tr = json.load(open(out))
    vqa = [s for s in tr["steps"] if s["tool"] == "vqa.rs"]
    q = tr["output"].get("quantities") or {}
    return {"text": tr["output"]["text"], "task": tr.get("classified_task"),
            "abstained": tr.get("abstained"), "stub": vqa[0]["stub"] if vqa else None,
            "adapter": q.get("adapter"), "kind": q.get("prompt_kind"),
            "tools": [s["tool"] for s in tr["steps"]], "secs": round(time.time() - t0, 1)}


def table(label, adapter):
    print("\n=== %s ===" % label)
    hits = 0
    for n, c in enumerate(cases):
        img = os.path.join(E2E, "%d.tif" % c["img_id"])
        out = os.path.join(E2E, "trace_%s_%02d.json" % ("lora" if adapter else "base", n))
        r = run(c["question"], img, out, adapter)
        pred = norm(r["text"], c["type"])
        ok = (binned(pred) == binned(c["gold"])) if c["type"] == "count" else (pred == c["gold"])
        hits += bool(ok)
        print("%s img %d %-12s gold=%-6s got=%-6s raw=%-14r adapter=%-14s kind=%-12s stub=%s task=%s %ss"
              % ("OK" if ok else "XX", c["img_id"], c["type"], c["gold"], pred, r["text"][:14],
                 r["adapter"], r["kind"], r["stub"], r["task"], r["secs"]))
        if adapter:
            assert r["stub"] is False, "vqa.rs ran as a STUB - the real backend did not load"
            assert r["adapter"] == "lora_rsvqa_v2", "trace does not name the adapter: %r" % r["adapter"]
            assert r["kind"] == ("presence" if c["type"] == "comp" else c["type"]), (r["kind"], c["type"])
        else:
            assert r["adapter"] == "", "adapter unset but trace names one: %r" % r["adapter"]
    print("%s: %d / %d" % (label, hits, len(cases)))
    return hits


with_adapter = table("WITH adapter (SATQUERY_VQA_ADAPTER set)", True)
without = table("WITHOUT adapter (base weights)", False)

print("\n=== grounding WITH the adapter loaded (must run on base weights, must not crash) ===")
env = dict(os.environ, SATQUERY_BACKEND="real", SATQUERY_QWEN=MODEL_DIR, SATQUERY_VQA_ADAPTER=ADAPTER, PYTHONIOENCODING="utf-8")
out = os.path.join(E2E, "trace_ground.json")
r = subprocess.run([PY, "-m", "satquery.run", "--query", "Where is the road?",
                    os.path.join(E2E, "%d.tif" % cases[0]["img_id"]), "--out", out],
                   cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
print("rc", r.returncode, r.stderr[-1500:] if r.returncode else "")
tr = json.load(open(out))
g = [s for s in tr["steps"] if s["tool"] == "ground.rs"]
print("tools:", [s["tool"] for s in tr["steps"]], "| stub:", g[0]["stub"] if g else None,
      "| quantities:", {k: v for k, v in (tr["output"].get("quantities") or {}).items()
                        if k in ("norm_box", "adapter", "denominator", "prompt_kind")},
      "| text:", tr["output"]["text"][:80])
assert g and g[0]["stub"] is False

print("\nSUMMARY  with adapter %d/%d   base %d/%d" % (with_adapter, len(cases), without, len(cases)))
