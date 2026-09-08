"""The grounding adapter's prompt form, pinned to the harness's copy."""

import importlib.util
import os

from satquery import models

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _harness():
    spec = importlib.util.spec_from_file_location(
        "eval_ground", os.path.join(HERE, "scripts", "eval_ground.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_suffix_matches_the_harness_byte_for_byte():
    assert models.GROUND_PROMPT_SUFFIX == _harness().PROMPT_SUFFIX


def test_where_is_question_becomes_a_ref_target():
    prompt, target = models.ground_prompt("Where is the road?")
    assert target == "the road"
    assert prompt == ("Identify the location of the <ref>the road</ref>."
                      + models.GROUND_PROMPT_SUFFIX)


def test_noun_phrase_passes_through():
    prompt, target = models.ground_prompt("largest connected region of pastures")
    assert target == "largest connected region of pastures"
    assert "<ref>largest connected region of pastures</ref>" in prompt


def test_other_leads_are_stripped():
    assert models.ground_prompt("Locate the water body")[1] == "the water body"
    assert models.ground_prompt("find the airport.")[1] == "the airport"
    assert models.ground_prompt("Show me the forest")[1] == "the forest"


def test_harness_parser_reads_the_engine_box_string():
    h = _harness()
    assert h.parse("[0.64 0.0, 1.0 0.71]", (448, 448)) == (0.64, 0.0, 1.0, 0.71)
    assert h.parse("[0.5 0.5, 0.5 0.5]", (448, 448)) is None
    assert h.parse('{"bbox_2d": [0, 0, 224, 448]}', (448, 448)) == (0.0, 0.0, 0.5, 1.0)
