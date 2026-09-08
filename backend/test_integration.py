import base64
import json
import urllib.request
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

QUERIES = [
    ("Q1 Road VQA", "Is there a road?", ["lr_232.tif"]),
    ("Q2 Count VQA", "How many buildings are there?", ["lr_232.tif"]),
    ("Q4 Water VQA", "Is there any water in the image?", ["opt.tif", "opt.tif.meta.json"]),
    ("Q5 Grounding", "Where is the largest connected region of pastures?", ["opt.tif", "opt.tif.meta.json"]),
    ("Q6 Change VQA", "Did the areas of buildings increase?", ["pre.png", "pre.png.meta.json", "post.png", "post.png.meta.json"]),
    ("Q8 Fusion", "Is there any water in the image?", ["opt.tif", "opt.tif.meta.json", "sar.tif", "sar.tif.meta.json"]),
]

def run_tests():
    print("Testing SatQuery API endpoints...")
    # 1. Health
    req = urllib.request.Request("http://localhost:8765/health")
    with urllib.request.urlopen(req) as resp:
        health = json.loads(resp.read())
        print(f"Health check: OK={health.get('ok')} backend={health.get('backend')}")

    # 2. Answer queries
    for label, query, files in QUERIES:
        payload = {"query": query, "files": []}
        for f in files:
            with open(FIXTURES_DIR / f, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")
            payload["files"].append({"name": f, "b64": b64})

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:8765/answer",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            trace = json.loads(resp.read())
            task = trace.get("classified_task")
            rule = trace.get("routing", {}).get("rule_id")
            text = trace.get("output", {}).get("text", "")
            print(f"  [{label}] Rule: {rule} | Task: {task} | Text: {text[:45]}...")

    print("\nAll integration queries passed successfully!")

if __name__ == "__main__":
    run_tests()
