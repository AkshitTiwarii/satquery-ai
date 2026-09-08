"""SatQuery AI - Runnable 8-Phase Pipeline Demonstration
Executes ISRO representative query:
"Use the optical and SAR images together to identify built-up and water-covered regions,
 and tell me if built-up area increased since last year."
"""

import json
import os
import sys
from pathlib import Path

# Put backend dir on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from satquery.pipeline_reference import execute_satquery_pipeline


def main():
    query = (
        "Use the optical and SAR images together to identify built-up and water-covered regions, "
        "and tell me if built-up area increased since last year."
    )

    fixtures_dir = BASE_DIR / "fixtures"
    opt_path = str(fixtures_dir / "opt.tif") if (fixtures_dir / "opt.tif").exists() else "optical_current_2025.tif"
    sar_path = str(fixtures_dir / "sar.tif") if (fixtures_dir / "sar.tif").exists() else "sar_current_2025.tif"
    past_path = str(fixtures_dir / "opt.tif") if (fixtures_dir / "opt.tif").exists() else "optical_prior_2024.tif"

    image_paths = {
        "optical_current": opt_path,
        "sar_current": sar_path,
        "optical_last_year": past_path,
    }

    dates = {
        "optical_current": "2025-11-15",
        "sar_current": "2025-11-15",
        "optical_last_year": "2024-11-12",
    }

    print("\n" + "=" * 76)
    print("🛰️  SATQUERY AI: 8-PHASE AUDITABLE EXECUTION PIPELINE (ISRO SIH26167)")
    print("=" * 76)
    print(f"User Query:\n  \"{query}\"\n")
    print("Input Observations:")
    print(f"  1. [Current Optical]    : {opt_path} (Dated {dates['optical_current']})")
    print(f"  2. [Current SAR]        : {sar_path} (Dated {dates['sar_current']})")
    print(f"  3. [Prior Year Optical] : {past_path} (Dated {dates['optical_last_year']})")
    print("-" * 76)

    # Execute
    res = execute_satquery_pipeline(query, image_paths, dates=dates)

    print("\n[PHASE 1 & 2: INGESTION & CLASSIFICATION]")
    print(f"  Detected Input Configuration : {res.input_config}")
    print(f"  Active Specialist Tasks      : {res.active_tasks}")

    print("\n[PHASE 3 & 4: PLANNING & SHARED BACKBONE]")
    plan_step = next(s for s in res.execution_trace if s["step"] == "planning")
    backbone_step = next(s for s in res.execution_trace if s["step"] == "backbone_encoding")
    print(f"  Topological Plan Order       : {' -> '.join(plan_step['order'])}")
    print(f"  Task Dependencies            : {plan_step['dependencies']}")
    print(f"  Shared Forward Passes        : {backbone_step['forward_passes']} forward passes across 3 images")
    print(f"  Compute Reduction            : ~{backbone_step['compute_saved_pct']}% reduction vs isolated pipelines")

    print("\n[PHASE 5 & 6: EXECUTION & CROSS-VALIDATION]")
    for s in res.execution_trace:
        if s["step"] == "execution":
            seed_info = f" (Seeded with: {s['seed']})" if s.get("seed") else ""
            print(f"  Executed Tool: {s['tool']}{seed_info} | Confidence: {s['confidence']:.0%} | Time: {s['time_ms']}ms")

    cv_step = next(s for s in res.execution_trace if s["step"] == "cross_validation")
    print(f"  Spatial IoU Agreement        : {cv_step['agreement'].upper()} (Overlap IoU: {cv_step['overlap_iou']})")
    print(f"  Calibrated Confidence        : {cv_step['final_confidence']:.0%}")

    print("\n" + "=" * 76)
    print("[PHASE 7 & 8: FINAL EVIDENCE-GROUNDED ANSWER]")
    print("=" * 76)
    print(res.answer_text)
    print("=" * 76)

    print("\n[AUDITABLE EXECUTION TRACE SCHEMA (EVALUATION READY)]:")
    print(json.dumps(res.execution_trace, indent=2))

    # Save exportable report
    out_dir = BASE_DIR / "out"
    out_dir.mkdir(exist_ok=True)
    report_file = out_dir / "satquery_8phase_audit_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(res.downloadable_report, f, indent=2)

    print(f"\n✅ Auditable Report exported to: {report_file}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
