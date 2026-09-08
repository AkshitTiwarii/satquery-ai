"""Force the stub backend for the whole suite.

Two reasons this is not laziness:

  * A test that behaves differently on a machine with a GPU is not a test. The
    suite has to give the same answer on Vinayak's laptop, on the office box
    and in CI.
  * `models.available()` imports torch to find out, which costs ~10 s the first
    time and turns a 1.5 s inner loop into a 13 s one.

The REAL backend is verified on Kaggle by `scripts/make_kaggle_notebook.py`,
which runs the same pipeline against the actual model and prints the trace. It
cannot be verified here, and pretending otherwise with a mock would test the
mock.
"""

import os
import sys
from pathlib import Path

# Add backend root and scripts folder to sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SATQUERY_BACKEND", "stub")
