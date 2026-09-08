"""Tests for the SatQuery 8-Phase Reference Execution Pipeline."""

from pathlib import Path
from satquery.pipeline_reference import execute_satquery_pipeline


def test_8phase_pipeline_end_to_end():
    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
    opt_path = str(fixtures_dir / "opt.tif")
    sar_path = str(fixtures_dir / "sar.tif")

    query = (
        "Use the optical and SAR images together to identify built-up and water-covered regions, "
        "and tell me if built-up area increased since last year."
    )

    image_paths = {
        "optical_current": opt_path,
        "sar_current": sar_path,
        "optical_last_year": opt_path,
    }

    dates = {
        "optical_current": "2025-11-15",
        "sar_current": "2025-11-15",
        "optical_last_year": "2024-11-12",
    }

    res = execute_satquery_pipeline(query, image_paths, dates=dates)

    # 1. Compound Input Configuration
    assert "cross_modal" in res.input_config
    assert "bitemporal" in res.input_config

    # 2. Intent Classification
    assert "rs_fusion" in res.active_tasks
    assert "rs_change" in res.active_tasks

    # 3. Execution Trace Steps
    step_names = [s["step"] for s in res.execution_trace]
    assert "ingestion" in step_names
    assert "compatibility_gate" in step_names
    assert "classification" in step_names
    assert "planning" in step_names
    assert "backbone_encoding" in step_names
    assert "execution" in step_names
    assert "cross_validation" in step_names

    # 4. Planning Dependency
    planning_step = next(s for s in res.execution_trace if s["step"] == "planning")
    assert planning_step["order"] == ["rs_fusion", "rs_change"]
    assert planning_step["dependencies"]["rs_change"] == ["rs_fusion"]

    # 5. Overlays and Answer Grounding
    assert "built_up_mask" in res.map_overlay
    assert "water_mask" in res.map_overlay
    assert "change_heatmap" in res.map_overlay
    assert res.confidence >= 0.80
    assert "Joint Optical-SAR Analysis" in res.answer_text
    assert "Bi-Temporal Change Analysis" in res.answer_text
