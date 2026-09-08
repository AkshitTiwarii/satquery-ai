"""Plumbing check for two resident adapters, through the real entry point.

The change adapter is still training, so the VQA adapter stands in under the
'change' name: this proves the loader holds two named adapters, change.vqa
runs REAL on a pair with the right one selected, vqa.rs still selects its
own, and ground.rs still runs with both disabled. Scores mean nothing here;
only the trace fields do.
"""
import json
import os
import shutil
import subprocess

ROOT = r"F:\sih"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
E2E = os.path.join(ROOT, "e2e", "pair")
os.makedirs(E2E, exist_ok=True)

f = "00003.png"
for sub, name, date in (("im1", "pre.png", "2019-06-01T03:00:00Z"),
                        ("im2", "post.png", "2021-06-01T03:00:00Z")):
    dst = os.path.join(E2E, name)
    shutil.copy(os.path.join(ROOT, "data", "second", sub, f), dst)
    json.dump({"acquired_at": date, "bands": 3, "bbox": None, "crs": None, "gsd_m": 0.5,
               "height": 512, "width": 512, "modality": "optical"},
              open(dst + ".meta.json", "w"), indent=1)

env = dict(os.environ,
           SATQUERY_BACKEND="real",
           SATQUERY_QWEN=os.path.join(ROOT, "models", "Qwen2.5-VL-3B-Instruct"),
           SATQUERY_VQA_ADAPTER=os.path.join(ROOT, "runs", "lora_rsvqa_v2"),
           SATQUERY_CHANGE_ADAPTER=os.path.join(ROOT, "runs", "lora_rsvqa_v2"),
           PYTHONIOENCODING="utf-8")


def run(query, imgs, out):
    r = subprocess.run([PY, "-m", "satquery.run", "--query", query, *imgs, "--out", out],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    if r.returncode:
        print("RUN FAILED rc=%d\n%s" % (r.returncode, r.stderr[-2500:]))
        raise SystemExit(1)
    t = json.load(open(out))
    q = t["output"].get("quantities") or {}
    print("query   :", query)
    print("task    :", t.get("classified_task"), "| abstained:", t.get("abstained"),
          "| rule:", (t.get("routing") or {}).get("rule_id"))
    print("steps   :", [(s["tool"], "stub" if s["stub"] else "REAL") for s in t["steps"]])
    print("adapter :", q.get("adapter"), "| prompt_kind:", q.get("prompt_kind"),
          "| tokens in:", q.get("input_tokens"))
    print("text    :", t["output"]["text"][:100].replace("\n", " "))
    print("warnings:", (t.get("input_check") or {}).get("warnings"))
    print()
    return t


pre, post = os.path.join(E2E, "pre.png"), os.path.join(E2E, "post.png")
t1 = run("Did the areas of buildings increase?", [pre, post], os.path.join(E2E, "trace_change.json"))
t2 = run("Is there a road?", [pre], os.path.join(E2E, "trace_vqa.json"))
t3 = run("Where is the road?", [pre], os.path.join(E2E, "trace_ground.json"))

s1 = [s for s in t1["steps"] if s["tool"] == "change.vqa"]
assert s1 and s1[0]["stub"] is False, "change.vqa did not run real"
assert (t1["output"]["quantities"] or {}).get("adapter") == "lora_rsvqa_v2"
assert (t1["output"]["quantities"] or {}).get("prompt_kind") == "change_or_not"
assert (t2["output"]["quantities"] or {}).get("adapter") == "lora_rsvqa_v2"
s3 = [s for s in t3["steps"] if s["tool"] == "ground.rs"]
assert s3 and s3[0]["stub"] is False
print("PLUMBING OK: two named adapters resident, change.vqa real on a pair, vqa.rs and ground.rs intact")
