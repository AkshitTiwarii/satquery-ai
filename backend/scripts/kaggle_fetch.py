"""Download ONLY the small records and the adapter from a satquery kernel's output.

    python scripts/kaggle_fetch.py vinayakameta/satquery-qlora-joint-e2 out_dir

`kaggle kernels output` pulls every file, and these kernels' working dirs hold
the copied LMDB and extracted tiles - gigabytes to read a 2 KB result. The
session-output listing carries a direct URL per file, so take the eval JSONs,
trainer states, READMEs and lora_adapter/ and nothing else. The listing caps
at 500 files a page and the adapter sorts after the extracted images, hence
the page loop (a 5 Sep gotcha: the first page hid the adapter).
"""
import os
import sys
import urllib.request

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest)

if len(sys.argv) < 3 or "/" not in sys.argv[1]:
    sys.exit("usage: kaggle_fetch.py USER/SLUG OUT_DIR")
user, slug = sys.argv[1].split("/", 1)
dest = sys.argv[2]

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = user
    req.kernel_slug = slug
    req.page_size = 500
    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    files = list(resp.files)
    while resp.next_page_token:
        req.page_token = resp.next_page_token
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        files += list(resp.files)
print(f"output files: {len(files)}")

want = [f for f in files
        if f.file_name.endswith((".json", ".md", ".txt"))
        or f.file_name.startswith("lora_adapter/")]
if not any(f.file_name.startswith("lora_adapter/adapter_model") for f in want):
    print("WARNING: no lora_adapter/adapter_model.* in this output")
for f in want:
    out = os.path.join(dest, f.file_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    urllib.request.urlretrieve(f.url, out)
    print(f"{os.path.getsize(out):>12}  {f.file_name}")
