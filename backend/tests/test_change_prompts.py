"""The change.rs adapter's templates, pinned to the CDVQA harness's copy.

Same reason as test_vqa_prompts.py: the engine and the Kaggle-side harness
each carry a copy, and the adapter only scores what it was asked in. No GPU.
"""

import importlib.util
import os

from satquery import models

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _harness():
    spec = importlib.util.spec_from_file_location(
        "eval_cdvqa", os.path.join(HERE, "scripts", "eval_cdvqa.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_engine_templates_match_the_harness_byte_for_byte():
    h = _harness()
    assert models.CDVQA_PROMPTS == h.PROMPTS
    assert models.CDVQA_PAIR_PREFIX == h.PAIR_PREFIX
    assert models.CDVQA_CLASSES == h.CLASSES
    assert models.CDVQA_RATIOS == h.RATIOS


def test_ratio_questions_get_the_ratio_template():
    prompt, kind = models.change_prompt("What is the percentage of changed regions?")
    assert kind == "change_ratio"
    assert prompt.startswith(models.CDVQA_PAIR_PREFIX)
    assert prompt.endswith("90_to_100.")


def test_which_class_questions_get_the_class_template():
    prompt, kind = models.change_prompt("What is the largest change?")
    assert kind == "change_to_what"
    assert "NVG_surface, buildings, low_vegetation, trees, water, playgrounds." in prompt


def test_yes_no_change_questions_get_the_yes_no_template():
    p1, k1 = models.change_prompt("Did the areas of trees decrease?")
    p2, k2 = models.change_prompt("Has built-up area increased")
    assert k1 == k2 == "change_or_not"
    assert p1.endswith("Did the areas of trees decrease? Answer with one word: yes or no.")
    assert p2.endswith("Has built-up area increased? Answer with one word: yes or no.")


def test_open_questions_fall_back_to_the_generic_change_prompt():
    prompt, kind = models.change_prompt("Describe what happened here between the two dates.")
    assert kind == "open"
    assert prompt == models.CHANGE_PROMPT.format(
        query="Describe what happened here between the two dates.")
