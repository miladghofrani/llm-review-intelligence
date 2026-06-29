"""
Converts floyt/reviews.json into floyt/batch_request.json
ready for the inference server.

Usage:
    python3 floyt/convert.py
"""

import json
from pathlib import Path

INPUT  = Path(__file__).parent / "reviews.json"
OUTPUT = Path(__file__).parent / "batch_request.json"

with open(INPUT) as f:
    data = json.load(f)

reviews = [
    entry["fields"]["renterQuote"][0].strip()
    for entry in data
    if entry["fields"].get("renterQuote", [""])[0].strip()
]

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump({"reviews": reviews}, f, ensure_ascii=False, indent=2)

print(f"Saved {len(reviews)} reviews to {OUTPUT}")
print(f"\ncurl -X POST http://localhost:8742/infer/batch \\")
print(f"  -H 'Content-Type: application/json' \\")
print(f"  -d @{OUTPUT}")
