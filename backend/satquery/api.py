"""The HTTP seam between the engine and a web UI. Standard library only.

    python -m satquery.api --port 8765            # after scripts/demo_env.ps1 on the box

One rule from the contract carries over unchanged: the UI consumes traces and
nothing else. So this server has exactly one job - accept a query and its
files, call `satquery.run.answer`, and return the trace that call produced,
byte for byte the same JSON the CLI writes. Anything the screen shows is
therefore something the audit log recorded, and the two cannot disagree.

Endpoints
  GET  /                  JSON index of these endpoints; the API serves no page
  GET  /health            {"ok": true, "backend": ..., "adapters": {...}}
  POST /answer            body: {"query": str, "files": [{"name": str, "b64": str}, ...],
                          a "<image>.meta.json" entry is a sidecar for the image it names,
                                 "seed": int (optional)}
                          -> the trace (HTTP 200), or {"error": str} (HTTP 400)
  GET  /traces/<id>       a trace this process produced earlier, by trace_id

Files travel as base64 inside JSON rather than multipart because a browser
can produce that with FileReader in three lines and the standard library can
parse it with zero dependencies - nothing to install on the demo box, nothing
that can be missing on stage. Each request's files land in their own
temporary directory named by the request, and are deleted when the trace has
been built; the trace records only file NAMES and sha256, never machine paths.

CORS is open on purpose: the UI is served from a different port or a file://
page during the demo, and this box is not on the internet.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import time
import requests

from . import models, run, bhoonidhi, agent

MAX_BODY = 200 * 1024 * 1024     # 200 MB of JSON: a handful of GeoTIFFs, generously
# Optional shared secret. Set SATQUERY_API_TOKEN when the port is reachable beyond the
# tailnet (a Funnel or a VPS tunnel): POST /answer then requires the same value in the
# X-SatQuery-Token header. GET /health and the page stay open - they leak nothing.
TOKEN = os.environ.get("SATQUERY_API_TOKEN", "").strip()
_TRACES: dict = {}
_LOCK = threading.Lock()          # one resident model, one query at a time

# Load .env if present
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

def get_remote_gateway_config() -> tuple[str, str]:
    """Dynamically load SATQUERY_REMOTE_URL and SATQUERY_REMOTE_TOKEN from environment or .env."""
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        try:
            with open(_env_path, "r", encoding="utf-8") as _ef:
                for _line in _ef:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        os.environ[_k.strip()] = _v.strip()
        except Exception:
            pass
    url = os.environ.get("SATQUERY_REMOTE_URL", "").rstrip("/")
    token = os.environ.get("SATQUERY_REMOTE_TOKEN", "").strip()
    return url, token


REMOTE_GATEWAY_URL, REMOTE_GATEWAY_TOKEN = get_remote_gateway_config()
_REMOTE_CACHE = {"timestamp": 0.0, "status": None}


def check_remote_gpu_health(force: bool = False) -> dict:
    """Check if the high-power GPU gateway is online, with short TTL caching and detailed error diagnostics."""
    url, token = get_remote_gateway_config()
    now = time.time()
    if not force and now - _REMOTE_CACHE["timestamp"] < 3.0 and _REMOTE_CACHE["status"] is not None:
        return _REMOTE_CACHE["status"]
    if not url:
        return {"ok": False, "model_available": False, "reason": "SATQUERY_REMOTE_URL not configured"}

    headers = {"Accept": "application/json"}
    if token:
        headers["X-SatQuery-Token"] = token

    last_err = None
    for verify_ssl in (True, False):
        try:
            resp = requests.get(f"{url}/health", headers=headers, timeout=3.5, verify=verify_ssl)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    is_avail = bool(data.get("model_available") or data.get("backend") == "real")
                    result = {
                        **data,
                        "model_available": is_avail,
                        "remote_url": url,
                    }
                    _REMOTE_CACHE["timestamp"] = now
                    _REMOTE_CACHE["status"] = result
                    return result
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue

    status_fail = {
        "ok": False,
        "model_available": False,
        "reason": last_err or "Unreachable",
        "remote_url": url,
    }
    _REMOTE_CACHE["timestamp"] = now
    _REMOTE_CACHE["status"] = status_fail
    return status_fail


def _json(handler, status: int, payload: dict) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-SatQuery-Token")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def answer_request(payload: dict) -> dict:
    """Validate, materialise the files, run the engine, clean up. Pure enough to test."""
    query = payload.get("query")
    files = payload.get("files")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    if not isinstance(files, list) or not files:
        raise ValueError("'files' must be a non-empty list of {name, b64}")
    seed = payload.get("seed", run.DEFAULT_SEED)
    if not isinstance(seed, int):
        raise ValueError("'seed' must be an integer")

    # Auto-dispatch to high-performance 4060 GPU gateway if available and local CUDA is not resident
    if REMOTE_GATEWAY_URL and not models.available():
        remote_health = check_remote_gpu_health()
        if remote_health.get("ok") and remote_health.get("model_available"):
            try:
                headers = {"Content-Type": "application/json"}
                if REMOTE_GATEWAY_TOKEN:
                    headers["X-SatQuery-Token"] = REMOTE_GATEWAY_TOKEN
                resp = requests.post(
                    f"{REMOTE_GATEWAY_URL}/answer",
                    json=payload,
                    headers=headers,
                    timeout=4.0,
                )
                if resp.status_code == 200:
                    trace = resp.json()
                    _TRACES[trace["trace_id"]] = trace
                    return trace
            except Exception as exc:
                print(f"[api] Remote GPU gateway request failed ({exc}); seamlessly falling back to local engine")
    tmp = tempfile.mkdtemp(prefix="satquery_req_")
    try:
        paths = []
        for i, f in enumerate(files):
            name = os.path.basename(str(f.get("name") or "")) or "input_%d" % i
            sidecar = name.lower().endswith(".meta.json")
            if not sidecar and not name.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                raise ValueError("file %r: only GeoTIFF/TIFF/PNG/JPEG are accepted "
                                 "(plus a <file>.meta.json sidecar per image)" % name)
            try:
                raw = base64.b64decode(f.get("b64") or "", validate=True)
            except Exception:  # noqa: BLE001
                raise ValueError("file %r: b64 did not decode" % name)
            if not raw:
                raise ValueError("file %r is empty" % name)
            p = os.path.join(tmp, name)
            if sidecar:
                # Dates, GSD, modality for a PNG/JPEG that carries none. Read by
                # satquery.types beside the image it names; never an input itself.
                try:
                    json.loads(raw.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    raise ValueError("file %r: sidecar is not JSON" % name)
            with open(p, "wb") as fh:
                fh.write(raw)
            if not sidecar:
                paths.append(p)
        if not paths:
            raise ValueError("'files' holds only sidecars; at least one image is needed")
        with _LOCK:
            trace = run.answer(query, paths, seed=seed)
        _TRACES[trace["trace_id"]] = trace
        return trace
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "satquery-api/0.1"

    def log_message(self, fmt, *args):  # quieter than the default, still one line per request
        print("%s %s" % (self.address_string(), fmt % args), flush=True)

    def do_OPTIONS(self):  # noqa: N802 - http.server naming
        _json(self, 204, {})

    def do_GET(self):  # noqa: N802
        if self.path == "/":
            _json(self, 200, {"ok": True, "service": "satquery-api", "endpoints": {
                "GET /health": "backend, precision, adapters",
                "POST /answer": "{query, files:[{name,b64}], seed?} -> trace",
                "GET /traces/<id>": "a trace this process produced",
            }})
            return
        if self.path == "/api/reconnect":
            rem = check_remote_gpu_health(force=True)
            _json(self, 200, rem)
            return
        if self.path == "/health":
            rem = check_remote_gpu_health()
            if rem.get("ok") and rem.get("model_available"):
                _json(self, 200, {
                    **rem,
                    "proxy": "local_controller",
                    "mode": "remote_gpu_active",
                })
                return
            _json(self, 200, {
                "ok": True,
                "backend": models.backend(),
                "model_available": models.available(),
                "dtype": models.dtype_name(),
                "adapters": {k: models.adapter_name(k) for k in ("vqa", "ground", "change", "fusion")},
                "mode": "local_fallback",
                "remote_gpu": rem,
            })
            return
        if self.path.startswith("/bhoonidhi/archives"):
            filter_kw = None
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                filter_kw = qs.get("q", [None])[0]
            archives = bhoonidhi.list_archives(filter_kw)
            _json(self, 200, {"ok": True, "count": len(archives), "archives": archives})
            return
        if self.path == "/bhoonidhi/status":
            status = bhoonidhi.get_auth_status()
            _json(self, 200, status)
            return
        if self.path.startswith("/traces/"):
            tid = self.path[len("/traces/"):]
            t = _TRACES.get(tid)
            if t is None:
                _json(self, 404, {"error": "no trace %s in this process" % tid})
            else:
                _json(self, 200, t)
            return
        _json(self, 404, {"error": "unknown path; use GET /health, /bhoonidhi/archives, POST /answer, /bhoonidhi/search"})

    def do_POST(self):  # noqa: N802
        if self.path == "/bhoonidhi/search":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                _json(self, 400, {"error": "body is not valid JSON"})
                return
            s_date = body.get("start_date")
            e_date = body.get("end_date")
            sat = body.get("satellite")
            sensor = body.get("sensor")
            bbox = body.get("bbox")
            limit = int(body.get("limit", 15))
            if not s_date or not e_date or not sat:
                _json(self, 400, {"error": "Missing required fields: start_date, end_date, satellite"})
                return
            result = bhoonidhi.search_scenes(s_date, e_date, sat, sensor=sensor, bbox=bbox, limit=limit)
            _json(self, 200, result)
            return
        if self.path in ("/agent", "/api/agent"):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                _json(self, 400, {"error": "body is not valid JSON"})
                return
            query = payload.get("query", "").strip()
            if not query:
                _json(self, 400, {"error": "'query' must be a non-empty string"})
                return
            files = payload.get("files")
            seed = payload.get("seed", 1337)
            try:
                res = agent.run_agentic_workflow(query, attached_files=files, seed=seed)
                _json(self, 200, res)
            except Exception as exc:
                _json(self, 500, {"error": f"Agent error: {exc}"})
        if self.path == "/api/remote-gateway":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                _json(self, 400, {"error": "body is not valid JSON"})
                return
            new_url = body.get("url", "").strip().rstrip("/")
            new_tok = body.get("token", "").strip()
            if new_url:
                os.environ["SATQUERY_REMOTE_URL"] = new_url
                if new_tok:
                    os.environ["SATQUERY_REMOTE_TOKEN"] = new_tok
                _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
                lines = []
                if os.path.exists(_env_path):
                    with open(_env_path, "r", encoding="utf-8") as f:
                        lines = [l for l in f.readlines() if not l.startswith("SATQUERY_REMOTE_URL") and not l.startswith("SATQUERY_REMOTE_TOKEN")]
                lines.append(f"SATQUERY_REMOTE_URL={new_url}\n")
                if new_tok:
                    lines.append(f"SATQUERY_REMOTE_TOKEN={new_tok}\n")
                with open(_env_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            rem = check_remote_gpu_health(force=True)
            _json(self, 200, {"ok": True, "health": rem})
            return
        if self.path != "/answer":
            _json(self, 404, {"error": "Unknown POST endpoint; supported: /answer, /agent, /bhoonidhi/search, /api/remote-gateway"})
            return
        if TOKEN and (self.headers.get("X-SatQuery-Token") or "").strip() != TOKEN:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 0:
                try:
                    self.rfile.read(length)
                except Exception:
                    pass
            _json(self, 401, {"error": "missing or wrong X-SatQuery-Token header"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            _json(self, 400, {"error": "body must be 1 byte to %d MB of JSON" % (MAX_BODY // (1024 * 1024))})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            _json(self, 400, {"error": "body is not valid JSON"})
            return
        if payload.get("agentic") is True:
            query = payload.get("query", "").strip()
            files = payload.get("files")
            seed = payload.get("seed", 1337)
            try:
                res = agent.run_agentic_workflow(query, attached_files=files, seed=seed)
                _json(self, 200, res)
            except Exception as exc:
                _json(self, 500, {"error": f"Agent error: {exc}"})
            return
        try:
            trace = answer_request(payload)
        except ValueError as exc:
            _json(self, 400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - the UI must see the failure, not a hang
            _json(self, 500, {"error": "%s: %s" % (type(exc).__name__, exc)})
            return
        _json(self, 200, trace)


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), Handler)
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SatQuery HTTP API - traces in, traces out")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from another machine")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    srv = serve(a.host, a.port)
    print("satquery api on http://%s:%d  backend=%s  model_available=%s" % (
        a.host, a.port, models.backend(), models.available()), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
