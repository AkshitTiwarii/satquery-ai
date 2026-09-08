"""The vqa.rs adapter was trained on four instruction templates. This pins them.

Two copies exist on purpose: satquery/models.py (the engine) and
scripts/eval_rsvqa_lr.py (the harness, which runs on Kaggle where the engine
package is not present). If they drift, the engine asks the adapter questions
in a format it never learned and the 85.5% quietly stops applying. No GPU
needed: this is string handling.
"""

import importlib.util
import os

from satquery import models

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _harness():
    spec = importlib.util.spec_from_file_location(
        "eval_rsvqa_lr", os.path.join(HERE, "scripts", "eval_rsvqa_lr.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _harness_prompts():
    return _harness().PROMPTS


def test_engine_templates_match_the_harness_byte_for_byte():
    assert models.RSVQA_PROMPTS == _harness_prompts()


def test_bucket_count_template_matches_the_harness():
    """The second counting instruction, for a bucket-trained adapter.

    Same drift risk as the four above, same consequence: the engine would ask
    for a bucket in wording the adapter never saw, or keep asking for an
    integer from an adapter that only answers buckets.
    """
    assert models.RSVQA_COUNT_BUCKET_PROMPT == _harness().COUNT_BUCKET_PROMPT


def test_count_form_defaults_to_the_integer_template():
    """Default stays 'number' so every recorded run still reproduces."""
    assert models.COUNT_FORM == "number"
    assert models.rsvqa_templates() == models.RSVQA_PROMPTS


def test_bucket_form_swaps_only_the_count_template(monkeypatch):
    monkeypatch.setattr(models, "COUNT_FORM", "bucket")
    t = models.rsvqa_templates()
    assert t["count"] == models.RSVQA_COUNT_BUCKET_PROMPT
    for kind in ("presence", "comp", "rural_urban"):
        assert t[kind] == models.RSVQA_PROMPTS[kind]


def test_bucket_form_reaches_vqa_prompt(monkeypatch):
    """The control for the swap: without it the engine would go on asking for
    an integer while a bucket-trained adapter was loaded."""
    monkeypatch.setattr(models, "COUNT_FORM", "bucket")
    prompt, kind = models.vqa_prompt("How many buildings are there?")
    assert kind == "count"
    assert prompt.endswith("Answer with one of: 0, 1-10, 11-100, 101-1000, >1000.")


def test_count_questions_get_the_count_template():
    prompt, kind = models.vqa_prompt("How many buildings are there?")
    assert kind == "count"
    assert prompt == "How many buildings are there? Answer with a single number and nothing else."


def test_rural_urban_questions_get_their_template():
    prompt, kind = models.vqa_prompt("Is it a rural or an urban area?")
    assert kind == "rural_urban"
    assert prompt.endswith("Answer with one word: rural or urban.")


def test_presence_and_comparison_share_the_yes_no_template():
    p1, k1 = models.vqa_prompt("Is there a road?")
    p2, k2 = models.vqa_prompt("Are there more roads than buildings")
    assert k1 == k2 == "presence"
    assert p1 == "Is there a road? Answer with one word: yes or no."
    assert p2 == "Are there more roads than buildings? Answer with one word: yes or no."


def test_trailing_question_mark_is_not_doubled():
    prompt, _ = models.vqa_prompt("Is there a road??  ")
    assert "??" not in prompt


def test_open_questions_fall_back_to_the_generic_prompt():
    prompt, kind = models.vqa_prompt("What is in this image?")
    assert kind == "open"
    assert prompt == models.VQA_PROMPT.format(query="What is in this image?")


def test_no_adapter_means_empty_name_in_the_trace():
    assert models.adapter_name() == ""
