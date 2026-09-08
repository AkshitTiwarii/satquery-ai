"""The HTTP seam returns the same trace the CLI would, and refuses bad input with a
readable error rather than a hang. Stub backend, real server on a free port."""

import base64
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satquery import api, run, trace as tracelib  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def setup_module(module):
    if not os.path.exists(os.path.join(FIX, "opt_2024.tif")):
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(FIX), "make_fixtures.py")])
    module.SRV = api.serve("127.0.0.1", 0)
    module.PORT = module.SRV.server_address[1]
    threading.Thread(target=module.SRV.serve_forever, daemon=True).start()


def teardown_module(module):
    module.SRV.shutdown()


def _post(path, payload):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), method="POST",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(path):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _file(name):
    with open(os.path.join(FIX, name), "rb") as fh:
        return {"name": name, "b64": base64.b64encode(fh.read()).decode()}


def test_health_says_which_backend_and_adapters():
    status, h = _get("/health")
    assert status == 200 and h["ok"] is True
    assert h["backend"] == "stub" and set(h["adapters"]) == {"vqa", "ground", "change", "fusion"}


def test_answer_returns_a_valid_trace_equal_to_the_cli_output():
    status, t = _post("/answer", {"query": "is there a water body?", "files": [_file("opt_2024.tif")]})
    assert status == 200, t
    tracelib.validate(t)
    cli = run.answer("is there a water body?", [os.path.join(FIX, "opt_2024.tif")])
    # everything that determines the answer is identical; only ids and timings may differ
    assert t["output"] == cli["output"]
    assert t["routing"] == cli["routing"]
    assert [s["tool"] for s in t["steps"]] == [s["tool"] for s in cli["steps"]]
    assert t["replay"]["input_sha256"] == cli["replay"]["input_sha256"]


def test_a_pair_goes_through_the_same_gate_and_rules():
    status, t = _post("/answer", {"query": "has the built-up area increased?",
                                  "files": [_file("opt_2024.tif"), _file("opt_2026.tif")]})
    assert status == 200 and t["routing"]["rule_id"] == "R3"
    status, again = _get("/traces/" + t["trace_id"])
    assert status == 200 and again["trace_id"] == t["trace_id"]


def test_a_refusal_is_a_trace_not_an_error():
    status, t = _post("/answer", {"query": "has it increased?", "files": [_file("opt_2024.tif")]})
    assert status == 200 and t["abstained"] is True and t["routing"]["rule_id"] == "R0"


def test_bad_input_gets_a_readable_400():
    assert _post("/answer", {"query": "", "files": [_file("opt_2024.tif")]})[0] == 400
    assert _post("/answer", {"query": "x", "files": []})[0] == 400
    status, e = _post("/answer", {"query": "x", "files": [{"name": "scene.bmp", "b64": "AAAA"}]})
    assert status == 400 and "GeoTIFF" in e["error"]
    status, e = _post("/answer", {"query": "x", "files": [{"name": "a.tif", "b64": "not base64!"}]})
    assert status == 400 and "decode" in e["error"]
    assert _get("/traces/nope")[0] == 404


def test_root_is_a_json_index_and_the_api_serves_no_page():
    status, idx = _get("/")
    assert status == 200 and "POST /answer" in idx["endpoints"]
    for path in ("/reference", "/ui", "/index.html"):
        assert _get(path)[0] == 404, path


def _png(name):
    with open(os.path.join(os.path.dirname(FIX), "fixtures", name), "rb") as fh:
        return {"name": name, "b64": base64.b64encode(fh.read()).decode()}


def test_two_plain_pngs_and_a_change_question_is_a_refusal_trace_not_a_500():
    status, t = _post("/answer", {"query": "did the buildings increase?",
                                  "files": [_png("bench_lr.png"), _png("bench_lr_2.png")]})
    assert status == 200 and t["abstained"] is True and t["input_check"]["code"] == "G3"


def test_sidecars_give_the_pngs_dates_and_the_pair_routes_to_change():
    def side(name, when):
        meta = json.dumps({"acquired_at": when, "modality": "optical", "gsd_m": 10.0})
        return {"name": name + ".meta.json", "b64": base64.b64encode(meta.encode()).decode()}
    status, t = _post("/answer", {"query": "did the buildings increase?",
                                  "files": [_png("bench_lr.png"), side("bench_lr.png", "2024-05-01T00:00:00Z"),
                                            _png("bench_lr_2.png"), side("bench_lr_2.png", "2026-05-01T00:00:00Z")]})
    assert status == 200, t
    assert t["abstained"] is False and t["routing"]["rule_id"] == "R3"
    assert t["input_check"]["n_images"] == 2
    status, e = _post("/answer", {"query": "x", "files": [side("a.png", "2024-01-01T00:00:00Z")]})
    assert status == 400 and "sidecar" in e["error"]


def test_token_gates_answer_but_not_health(monkeypatch):
    monkeypatch.setattr(api, "TOKEN", "s3cret")
    assert _get("/health")[0] == 200
    assert _post("/answer", {"query": "x", "files": [_file("opt_2024.tif")]})[0] == 401
    req = urllib.request.Request("http://127.0.0.1:%d/answer" % PORT, method="POST",
                                 data=json.dumps({"query": "is there a water body?", "files": [_file("opt_2024.tif")]}).encode(),
                                 headers={"Content-Type": "application/json", "X-SatQuery-Token": "s3cret"})
    with urllib.request.urlopen(req, timeout=60) as r:
        assert r.status == 200
