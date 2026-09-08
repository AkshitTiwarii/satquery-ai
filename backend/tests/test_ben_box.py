"""The grounding adapter answers in BigEarthNet.txt's own box string."""

from satquery import models


def test_ben_box_string_parses_as_normalised():
    box, how = models.parse_box("[0.64 0.0, 1.0 0.71]", (448, 448), (120, 120))
    assert how == "already_0_1"
    assert (box.x0, box.y0, box.x1, box.y1) == (0.64, 0.0, 1.0, 0.71)


def test_ben_box_with_prose_around_it():
    box, how = models.parse_box("The region is at [0.0 0.33, 0.28 0.8].", (448, 448), (120, 120))
    assert how == "already_0_1"
    assert (box.x0, box.y0, box.x1, box.y1) == (0.0, 0.33, 0.28, 0.8)


def test_qwen_json_pixel_box_still_parses():
    box, how = models.parse_box('{"bbox_2d": [0, 0, 224, 448]}', (448, 448), (120, 120))
    assert how == "resized"
    assert (box.x0, box.y0, box.x1, box.y1) == (0.0, 0.0, 0.5, 1.0)


def test_degenerate_ben_box_is_refused():
    box, how = models.parse_box("[0.5 0.5, 0.5 0.5]", (448, 448), (120, 120))
    assert box is None
