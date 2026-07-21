"""
Converts floyt/reviews.json into floyt/batch_request.json
ready for the /infer/batch endpoint.

Usage:
    python3 floyt/convert.py
"""

import json
from pathlib import Path

INPUT  = Path(__file__).parent / "reviews.json"
OUTPUT = Path(__file__).parent / "batch_request.json"


def _extract(entry: dict) -> dict | None:
    fields = entry.get("fields", {})
    review = fields.get("renterQuote", [""])[0].strip()
    if not review:
        return None
    return {
        "review":                   review,
        "database_id":              fields.get("databaseId",             [None])[0],
        "car_condition_rating":     fields.get("carConditionRating",     [None])[0],
        "processing_speed_rating":  fields.get("processingSpeedRating",  [None])[0],
        "service_level_rating":     fields.get("serviceLevelRating",     [None])[0],
        "recommendation_rating":    fields.get("recommendationRating",   [None])[0],
    }


with open(INPUT) as f:
    data = json.load(f)

reviews = [r for entry in data if (r := _extract(entry))]

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump({"reviews": reviews}, f, ensure_ascii=False, indent=2)

print(f"Saved {len(reviews)} reviews to {OUTPUT}")
print(f"\ncurl -X POST http://localhost:8742/infer/batch \\")
print(f"  -H 'Content-Type: application/json' \\")
print(f"  -d @{OUTPUT}")
