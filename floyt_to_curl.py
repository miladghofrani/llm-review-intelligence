"""
Converts data_processing/floyt-reviews.json into a batch request payload
ready for the inference server.

Usage:
    python3 floyt_to_curl.py              # prints the curl command
    python3 floyt_to_curl.py --save       # saves payload to floyt_batch_request.json
"""

import json
import sys
from pathlib import Path

INPUT  = Path("data_processing/floyt-reviews.json")
OUTPUT = Path("floyt_batch_request.json")

with open(INPUT) as f:
    data = json.load(f)

reviews = [
    entry["fields"]["renterQuote"][0].strip()
    for entry in data
    if entry["fields"].get("renterQuote", [""])[0].strip()
]

payload = {"reviews": reviews}

if "--save" in sys.argv:
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(reviews)} reviews to {OUTPUT}")
    print(f"\ncurl -X POST http://localhost:8742/infer/batch \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d @{OUTPUT}")
else:
    print(f"# {len(reviews)} reviews extracted from {INPUT}")
    print(f"\ncurl -X POST http://localhost:8742/infer/batch \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(payload, ensure_ascii=False)}'")
