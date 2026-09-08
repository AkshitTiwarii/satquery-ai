"""The VRSBench train cut: parses the LLaVA form, keeps the dataset's answer strings,
asks in the harness's own templates, and refuses leakage and bad boxes."""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_vrsbench import REF_PROMPTS, VQA_PROMPT  # noqa: E402
from make_vrsbench_subset import build_rows, parse_conversation, refer_expression  # noqa: E402


def rec(image, tag, q, a, typ=""):
    return {"id": "x", "image": image, "type": typ,
            "conversations": [{"from": "human", "value": "<image>\n[%s] %s" % (tag, q)},
                              {"from": "gpt", "value": a}]}


RECS = [
    rec("00002_0000.png", "caption", "Could you describe this?", "A toll station."),
    rec("00002_0000.png", "refer", "could you tell me the location for <p>The toll station at the center</p>?",
        "{<45><45><59><59>}"),
    rec("00002_0000.png", "vqa", "How many vehicles are there?", "3", "object quantity"),
    rec("00003_0000.png", "vqa", "Is there a river", "No", "object existence"),
    rec("00003_0000.png", "refer", "where is <p>the bridge</p>?", "{<10><20><130><40>}"),   # out of range
    rec("00004_0000.png", "refer", "where is <p>the pier</p>?", "no box here"),
]


def test_parse_conversation_reads_tag_question_answer():
    assert parse_conversation(RECS[2]) == ("vqa", "How many vehicles are there?", "3")
    assert refer_expression("could you tell me the location for <p>The toll station at the center</p>?") == \
        "The toll station at the center"


def test_vqa_rows_use_the_harness_prompt_and_keep_the_answer():
    rows, dropped = build_rows(RECS, "vqa")
    assert [r["output"] for r in rows] == ["3", "No"]
    assert rows[0]["input"] == VQA_PROMPT.format(q="How many vehicles are there?")
    assert rows[1]["input"] == VQA_PROMPT.format(q="Is there a river?")      # question mark restored
    assert rows[0]["tag"] == "object quantity" and rows[0]["patch_id"] == "00002_0000.png"


def test_refer_rows_keep_the_brace_box_and_drop_the_bad_ones():
    rows, dropped = build_rows(RECS, "refer")
    assert len(rows) == 1
    assert rows[0]["output"] == "{<45><45><59><59>}"
    assert rows[0]["input"] == REF_PROMPTS["vrsbench"].format(q="The toll station at the center")
    assert dropped["box_out_of_range"] == 1 and dropped["box_unparsed"] == 1


def test_captions_never_leak_into_either_task():
    for task in ("vqa", "refer"):
        rows, _ = build_rows(RECS, task)
        assert all(r["type"] == task for r in rows)
        assert not any("describe" in r["input"] for r in rows)


def test_cli_refuses_an_image_shared_with_the_eval_split(tmp_path):
    train = tmp_path / "train.json"
    train.write_text(json.dumps(RECS), encoding="utf-8")
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps([{"image_id": "00003_0000.png", "question": "q", "ground_truth": "a"}]), encoding="utf-8")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_vrsbench_subset.py"),
                        "--train-json", str(train), "--task", "vqa", "--eval-json", str(ev),
                        "--out", str(tmp_path / "rows.jsonl")], capture_output=True, text=True)
    assert p.returncode != 0 and "refusing" in (p.stdout + p.stderr)
    # and without the leaking image it writes the file with a histogram
    ev.write_text(json.dumps([{"image_id": "99999_0000.png", "question": "q", "ground_truth": "a"}]), encoding="utf-8")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_vrsbench_subset.py"),
                        "--train-json", str(train), "--task", "vqa", "--eval-json", str(ev),
                        "--out", str(tmp_path / "rows.jsonl")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    rows = [json.loads(l) for l in (tmp_path / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and "eval-split overlap: 0" in p.stdout


def test_cli_refuses_a_file_without_the_task(tmp_path):
    train = tmp_path / "train.json"
    train.write_text(json.dumps([RECS[0]]), encoding="utf-8")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_vrsbench_subset.py"),
                        "--train-json", str(train), "--task", "refer", "--out", str(tmp_path / "r.jsonl")],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "no [refer] rows" in (p.stdout + p.stderr)
