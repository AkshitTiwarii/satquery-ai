"""Quick live test for Bhoonidhi integrations."""
import sys, io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("=== Test 1: bhoonidhi_downloader SDK (BhoonidhiClient) ===")
try:
    from bhoonidhi_downloader.sdk import BhoonidhiClient
    client = BhoonidhiClient()
    s_dt = datetime(2024, 6, 1)
    e_dt = datetime(2024, 9, 1)
    f_out = io.StringIO()
    with redirect_stdout(f_out), redirect_stderr(f_out):
        q = client.query.create(
            s_dt, e_dt,
            satellite="Sentinel-2A",
            minx=77.50, maxx=77.70,
            miny=12.90, maxy=13.10,
            save=False,
        )
    scenes = getattr(q, "scenes", []) or []
    print(f"  Sentinel-2A scenes found: {len(scenes)}")
    if scenes:
        s = scenes[0]
        s_dict = s if isinstance(s, dict) else (s.dict() if hasattr(s, "dict") else {})
        print(f"  First scene keys: {list(s_dict.keys())[:10]}")
        print(f"  Scene ID: {s_dict.get('ID') or s_dict.get('FILENAME')}")
        print(f"  DOP: {s_dict.get('DOP')}")
except Exception as e:
    print(f"  SDK ERROR: {e}")

print()
print("=== Test 2: bhoonidhi SmartSearch (NLP) ===")
try:
    from bhoonidhi.SmartSearch import bhoonidhiSmartSearch
    f_out = io.StringIO()
    with redirect_stdout(f_out), redirect_stderr(f_out):
        res = bhoonidhiSmartSearch("Get me Sentinel-2A data from Bengaluru last 3 months")
    n = len(res) if isinstance(res, list) else "N/A"
    print(f"  SmartSearch results: type={type(res).__name__} count={n}")
    if isinstance(res, list) and res:
        first = res[0]
        if isinstance(first, dict):
            print(f"  First result keys: {list(first.keys())[:8]}")
        else:
            print(f"  First result: {first}")
except Exception as e:
    print(f"  SmartSearch ERROR: {type(e).__name__}: {e}")

print()
print("=== Test 3: agent.py bhoonidhi integration ===")
try:
    from satquery import bhoonidhi as bh
    result = bh.search_scenes(
        start_date="2024-06-01",
        end_date="2024-09-01",
        satellite="Sentinel-2A",
        bbox=[77.50, 12.90, 77.70, 13.10],
        limit=5,
    )
    scenes = result.get("scenes", [])
    err = result.get("error")
    if err:
        print(f"  search_scenes ERROR: {err}")
    else:
        print(f"  search_scenes returned {result.get('total_found')} total, {len(scenes)} returned")
        if scenes:
            print(f"  First scene: {scenes[0]}")
except Exception as e:
    print(f"  satquery.bhoonidhi ERROR: {type(e).__name__}: {e}")
