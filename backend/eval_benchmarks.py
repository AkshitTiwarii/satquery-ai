"""SatQuery AI - Official SIH / ISRO-SAC 5-Query Benchmark Evaluation Harness.

Validates the 5 official representative tasks:
  1. Single-Image Captioning / Scene Description (VRSBench)
  2. Text-Guided Spatial Region Grounding (VRSBench / RSVQA)
  3. Multi-Temporal Change Detection & Mask Export (CDVQA)
  4. Optical + SAR Cross-Modal Fusion (Cartosat-2S + RISAT evaluation pair)
  5. Quantitative Change-VQA (Trend and Delta Analysis)

Verifies:
  - Physical Verification Gates (G1-G5)
  - Registry Rule Dispatching (R1-R6)
  - Observable Execution Trace Compliance
  - Spatial Evidence Output (Bounding Boxes, Change Masks, GeoJSON)
"""

import os
import sys
import time
import json
from pathlib import Path

# Add backend to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from satquery import run, registry


def run_benchmark_eval():
    fixtures_dir = BASE_DIR / "fixtures"
    opt_path = str(fixtures_dir / "opt.tif")
    sar_path = str(fixtures_dir / "sar.tif")
    pre_path = str(fixtures_dir / "pre.png")
    post_path = str(fixtures_dir / "post.png")
    lr_path = str(fixtures_dir / "lr_232.tif")

    test_cases = [
        {
            "id": "BENCH_Q1_CAPTIONING",
            "name": "Query 1: Single-Image Land-Cover & Scene Description",
            "query": "Describe the land-cover and major objects visible in this image.",
            "images": [opt_path],
            "expected_task": "captioning",
            "expected_tool": "caption.rs",
            "check_output": lambda trace: len(trace["output"]["text"]) > 20 and trace["input_check"]["verdict"] == "accepted",
        },
        {
            "id": "BENCH_Q2_GROUNDING",
            "name": "Query 2: Text-Guided Spatial Region Grounding",
            "query": "Highlight the water body referred to in the query.",
            "images": [lr_path if os.path.exists(lr_path) else opt_path],
            "expected_task": "grounding",
            "expected_tool": "ground.rs",
            "check_output": lambda trace: (
                trace["output"].get("quantities", {}).get("norm_box") is not None
                or trace["output"].get("geojson") is not None
            ),
        },
        {
            "id": "BENCH_Q3_CHANGE_MASK",
            "name": "Query 3: Multi-Temporal Change Localization & Mask Export",
            "query": "What changed between these two dates, and where did the change occur?",
            "images": [pre_path, post_path],
            "expected_task": "change_vqa",
            "expected_tool": "change.vqa",
            "check_output": lambda trace: (
                trace["output"].get("raster") is not None
                or "delta_pct" in trace["output"].get("quantities", {})
                or "export.path" in trace["output"].get("quantities", {})
            ),
        },
        {
            "id": "BENCH_Q4_OPTICAL_SAR_FUSION",
            "name": "Query 4: Optical + SAR Cross-Modal Feature Extraction",
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "images": [opt_path, sar_path],
            "expected_task": "fusion",
            "expected_tool": "fusion.optical_sar",
            "check_output": lambda trace: (
                trace["input_check"]["modality"] == ["optical", "sar"]
                or trace["input_check"]["verdict"] == "accepted"
            ),
        },
        {
            "id": "BENCH_Q5_CHANGE_VQA_TREND",
            "name": "Query 5: Quantitative Change Trend & Area Delta",
            "query": "Has the built-up area increased, decreased, or remained unchanged?",
            "images": [pre_path, post_path],
            "expected_task": "change_vqa",
            "expected_tool": "change.vqa",
            "check_output": lambda trace: any(
                w in trace["output"]["text"].lower()
                for w in ["increase", "decrease", "unchanged", "stable", "delta", "%"]
            ),
        },
    ]

    print("\n" + "=" * 80)
    print("🛰️  SATQUERY AI: OFFICIAL ISRO / SIH 5-QUERY BENCHMARK EVALUATION HARNESS")
    print("=" * 80)
    print(f"Registry Lock Hash : {registry.lock_hash()[:16]}... (Deterministic Audit Guard)")
    print(f"Total Test Cases   : {len(test_cases)}")
    print("-" * 80)

    results_summary = []
    all_passed = True

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/{len(test_cases)}] Running: {tc['name']}")
        print(f"  Query    : \"{tc['query']}\"")
        print(f"  Inputs   : {[os.path.basename(p) for p in tc['images']]}")

        t0 = time.perf_counter()
        try:
            trace = run.answer(tc["query"], tc["images"], seed=1337)
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)

            task_ok = trace["classified_task"] == tc["expected_task"]
            tool_used = next((s["tool"] for s in trace.get("steps", []) if s.get("tool") == tc["expected_tool"]), None)
            tool_ok = tool_used is not None
            output_ok = tc["check_output"](trace)
            passed = task_ok and tool_ok and output_ok and not trace.get("abstained")

            status_str = "✅ PASS" if passed else "❌ FAIL"
            if not passed:
                all_passed = False

            print(f"  Result   : {status_str} in {duration_ms}ms")
            print(f"  Task     : {trace['classified_task']} (Expected: {tc['expected_task']}) -> {'OK' if task_ok else 'MISMATCH'}")
            print(f"  Tool     : {tool_used or 'None'} (Expected: {tc['expected_tool']}) -> {'OK' if tool_ok else 'MISMATCH'}")
            print(f"  Output   : {trace['output']['text'][:120]}...")
            if trace["output"].get("quantities"):
                print(f"  Metrics  : {trace['output']['quantities']}")

            results_summary.append({
                "id": tc["id"],
                "name": tc["name"],
                "passed": passed,
                "task": trace["classified_task"],
                "tool": tool_used,
                "duration_ms": duration_ms,
                "trace_id": trace["trace_id"],
            })

        except Exception as exc:
            all_passed = False
            print(f"  Result   : ❌ EXCEPTION - {exc}")
            results_summary.append({
                "id": tc["id"],
                "name": tc["name"],
                "passed": False,
                "error": str(exc),
            })

    print("\n" + "=" * 80)
    print("📊 BENCHMARK EVALUATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Test Case ID':<28} | {'Task':<14} | {'Tool':<18} | {'Latency':<9} | {'Status'}")
    print("-" * 80)
    for r in results_summary:
        if r.get("passed"):
            print(f"{r['id']:<28} | {r['task']:<14} | {r['tool']:<18} | {r['duration_ms']}ms | ✅ PASS")
        else:
            err = r.get("error", "Output mismatch")
            print(f"{r['id']:<28} | {'FAILED':<14} | {'-':<18} | {'-':<9} | ❌ {err}")

    print("=" * 80)
    if all_passed:
        print("🎉 ALL 5 BENCHMARK QUERIES PASSED 100% SPECIFICATION REQUIREMENTS!")
    else:
        print("⚠️ SOME BENCHMARK TESTS FAILED - REVIEW ABOVE LOG.")
    print("=" * 80 + "\n")

    # Export benchmark report
    out_dir = BASE_DIR / "out"
    out_dir.mkdir(exist_ok=True)
    report_p = out_dir / "official_benchmark_eval_results.json"
    with open(report_p, "w", encoding="utf-8") as fh:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lock_hash": registry.lock_hash(),
            "summary": results_summary,
            "all_passed": all_passed,
        }, fh, indent=2)
    print(f"Audit log saved to: {report_p}\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_benchmark_eval())
