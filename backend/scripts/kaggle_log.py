"""Fetch ONLY the session log of a satquery Kaggle kernel.

    python scripts/kaggle_log.py vinayakameta/satquery-qlora-joint-e2 [out.log]

kernels_output() downloads every output file first, and these kernels' working
dirs carry the copied LMDB - a multi-GB download to read a traceback. The same
endpoint returns the log alongside the file list, so call it directly and take
only that.
"""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest)

if len(sys.argv) < 2 or "/" not in sys.argv[1]:
    sys.exit("usage: kaggle_log.py USER/SLUG [out.log]")
user, slug = sys.argv[1].split("/", 1)
out = sys.argv[2] if len(sys.argv) > 2 else f"{slug}.log"

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = user
    req.kernel_slug = slug
    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)

names = [f.file_name for f in resp.files]
print(f"output files: {len(names)}; adapter present:",
      any(n.startswith("lora_adapter/adapter_model") for n in names))
log = resp.log or ""
with open(out, "w", encoding="utf-8") as fh:
    fh.write(log)
print(f"log chars: {len(log)} -> {out}")
