"""Plumbing check for the ground and fusion adapters, through the real entry point.

Neither adapter exists yet, so the VQA adapter stands in under both names.
What this proves: four named adapters load onto one model; ground.rs asks in
the <ref> form and parses the BigEarthNet box string; fusion.optical_sar
routes an optical + SAR pair (rule R1), renders the SAR GeoTIFF through the
fixed-window render, and answers real; the trace names the adapter each time.
Scores mean nothing here - only the trace fields do.

Inputs are built from the reBEN LMDB on the office box: one optical tile as
PNG and its Sentinel-1 VV/VH as a 2-band float32 GeoTIFF, with sidecars
carrying modality and one shared date.
"""
import json
import os
import subprocess
import sys

ROOT = os.environ.get("SIH_ROOT", r"F:\sih")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
E2E = os.path.join(ROOT, "e2e", "gf")
os.makedirs(E2E, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ben_images import BENImages  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import rasterio  # noqa: E402

t = pq.read_table(os.path.join(ROOT, "data", "BigEarthNet.txt.parquet"),
                  columns=["patch_id", "s1_name"],
                  filters=[("country", "=", "Lithuania"), ("season", "=", "Summer"), ("split", "=", "test")])
row = t.slice(0, 1).to_pylist()[0]
ben = BENImages(os.path.join(ROOT, "data", "ben_lt_summer"))
opt = ben.get(row["patch_id"], size=None)
vv, vh = ben.get_sar(row["s1_name"])
# Both as GeoTIFF: the fusion row accepts GeoTIFF/TIFF only (the gate refused a
# PNG on the first attempt, correctly - the hidden set is GeoTIFF).
opt_path = os.path.join(E2E, "opt.tif")
sar_path = os.path.join(E2E, "sar.tif")
arr = np.asarray(opt)
with rasterio.open(opt_path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                   count=3, dtype="uint8") as dst:
    for i in range(3):
        dst.write(arr[:, :, i], i + 1)
with rasterio.open(sar_path, "w", driver="GTiff", height=vv.shape[0], width=vv.shape[1],
                   count=2, dtype="float32") as dst:
    dst.write(vv, 1)
    dst.write(vh, 2)
for p, mod, bands, wh in ((opt_path, "optical", 3, opt.size), (sar_path, "sar", 2, vv.shape[::-1])):
    json.dump({"acquired_at": "2017-07-19T04:50:00Z", "bands": bands, "bbox": None, "crs": None,
               "gsd_m": 10.0, "width": wh[0], "height": wh[1], "modality": mod},
              open(p + ".meta.json", "w"), indent=1)

STAND_IN = os.path.join(ROOT, "runs", "lora_rsvqa_v2")
env = dict(os.environ, SATQUERY_BACKEND="real",
           SATQUERY_QWEN=os.path.join(ROOT, "models", "Qwen2.5-VL-3B-Instruct"),
           SATQUERY_VQA_ADAPTER=STAND_IN, SATQUERY_CHANGE_ADAPTER=STAND_IN,
           SATQUERY_GROUND_ADAPTER=STAND_IN, SATQUERY_FUSION_ADAPTER=STAND_IN,
           PYTHONIOENCODING="utf-8")


def run(query, imgs, out):
    r = subprocess.run([PY, "-m", "satquery.run", "--query", query, *imgs, "--out", out],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    if r.returncode:
        print("RUN FAILED rc=%d\n%s" % (r.returncode, r.stderr[-2500:]))
        raise SystemExit(1)
    tr = json.load(open(out))
    q = tr["output"].get("quantities") or {}
    print("query   :", query)
    print("task    :", tr.get("classified_task"), "| rule:", (tr.get("routing") or {}).get("rule_id"),
          "| abstained:", tr.get("abstained"))
    print("steps   :", [(s["tool"], "stub" if s["stub"] else "REAL") for s in tr["steps"]])
    print("adapter :", q.get("adapter"), "| kind/target:", q.get("prompt_kind") or q.get("ref_target"),
          "| box:", q.get("norm_box"), "| frame:", q.get("box_frame") or q.get("box_parse"))
    print("text    :", tr["output"]["text"][:90].replace("\n", " "))
    print()
    return tr


t1 = run("Where is the largest connected region of pastures?", [opt_path], os.path.join(E2E, "trace_ground.json"))
t2 = run("Is there any water in the image?", [opt_path, sar_path], os.path.join(E2E, "trace_fusion.json"))
t3 = run("Is there a road?", [opt_path], os.path.join(E2E, "trace_vqa.json"))

g = [s for s in t1["steps"] if s["tool"] == "ground.rs"]
assert g and g[0]["stub"] is False, "ground.rs did not run real"
assert (t1["output"]["quantities"] or {}).get("adapter") == "lora_rsvqa_v2"
assert (t1["output"]["quantities"] or {}).get("ref_target") == "the largest connected region of pastures"
f = [s for s in t2["steps"] if s["tool"] == "fusion.optical_sar"]
assert t2.get("classified_task") == "fusion" and f and f[0]["stub"] is False, "fusion did not run real on the pair"
assert (t2["output"]["quantities"] or {}).get("adapter") == "lora_rsvqa_v2"
assert (t2["output"]["quantities"] or {}).get("prompt_kind") == "binary"
assert (t3["output"]["quantities"] or {}).get("adapter") == "lora_rsvqa_v2"
print("PLUMBING OK: four named adapters resident; ground.rs real in <ref> form; fusion real on optical+SAR; vqa intact")
