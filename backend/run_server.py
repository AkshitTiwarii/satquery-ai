"""Convenience launcher for the SatQuery AI API server.

Usage:
    python run_server.py [--port 8765] [--host 0.0.0.0] [--backend stub|real]
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding across Windows consoles so Rich spinners & Unicode never fail
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure backend root is on sys.path so `satquery` can be imported
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the SatQuery API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument(
        "--backend",
        choices=["stub", "real"],
        default=os.environ.get("SATQUERY_BACKEND", "stub"),
        help="Backend mode: 'stub' (zero-GPU, canned answers, real traces) or 'real' (GPU models)",
    )
    args = parser.parse_args()

    os.environ["SATQUERY_BACKEND"] = args.backend

    from satquery import api, models

    print("=" * 60)
    print(f">> SatQuery AI API starting on http://{args.host}:{args.port}")
    print(f"   Mode: {args.backend} | Model Available: {models.available()}")
    print("=" * 60, flush=True)

    server = api.serve(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SatQuery server...")
