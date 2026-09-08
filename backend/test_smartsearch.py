"""Test SmartSearch with various queries."""
import sys, io
from contextlib import redirect_stdout, redirect_stderr
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bhoonidhi.SmartSearch import bhoonidhiSmartSearch

queries = [
    "Get me Landsat 8 data from Hyderabad last month",
    "Show me Sentinel 2A data over Bengaluru",
    "Get data from kerala floods 2020",
    "Get me EOS-04 data from Delhi last year",
]
for q in queries:
    try:
        cap = io.StringIO()
        with redirect_stdout(cap), redirect_stderr(cap):
            res = bhoonidhiSmartSearch(q)
        n = len(res) if isinstance(res, list) else "N/A"
        print(f"Query: {q}")
        print(f"  count={n}")
        if isinstance(res, list) and res:
            print(f"  First: {res[0]}")
    except Exception as e:
        print(f"Query: {q}")
        print(f"  ERROR: {e}")
    print()
