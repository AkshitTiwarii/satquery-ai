"""The fusion adapter's prompt form and SAR render, pinned to the harness copies.

scripts/eval_ben_vqa.py and scripts/ben_images.py run on Kaggle where the
engine package is absent, so each side keeps a copy; these tests are what
stop them drifting. No GPU, no LMDB: the render is checked on synthetic dB.
"""

import importlib.util
import os

import numpy as np

from satquery import models

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefix_and_templates_match_the_harness():
    h = _load("scripts/eval_ben_vqa.py", "eval_ben_vqa")
    assert models.SAR_PREFIX == h.SAR_PREFIX
    assert models.BEN_VQA_PROMPTS == h.PROMPTS


def test_db_windows_match_the_loader():
    import pytest
    try:
        bi = _load("scripts/ben_images.py", "ben_images")
    except ImportError:
        pytest.skip("lmdb is not installed here; the window is checked where it is")
    assert models.SAR_DB_WINDOW == bi.SAR_DB_WINDOW


def test_yes_no_question_gets_the_binary_form():
    prompt, kind = models.fusion_prompt("Is there any water in the image?")
    assert kind == "binary"
    assert prompt == models.SAR_PREFIX + "Is there any water in the image? Answer with one word: yes or no."


def test_options_make_it_multiple_choice():
    q = "Which classes share a boundary? a) Arable land and Pastures, b) Water and Forest, c) none, d) all"
    prompt, kind = models.fusion_prompt(q)
    assert kind == "mcq"
    assert prompt.endswith("Answer with the letter only: a, b, c or d.")


def test_sar_render_is_deterministic_and_in_range():
    vv = np.full((8, 8), -18.0, dtype="float32")
    vh = np.full((8, 8), -28.0, dtype="float32")
    img = models.sar_to_rgb(vv, vh)
    r, g, b = np.asarray(img)[0, 0]
    # VV -18 on (-25, 0) -> 0.28; VH -28 on (-32, -5) -> 0.148; ratio 10 on (0, 20) -> 0.5
    assert (r, g, b) == (71, 37, 127)
    lin = models.sar_to_rgb(10 ** (vv / 10), 10 ** (vh / 10))      # linear in -> same dB out
    assert np.array_equal(np.asarray(lin), np.asarray(img))
