"""
Converts floyt/rentalcars-1year.csv into an /infer/aggregate request payload.

Usage:
    python3 floyt/convert_csv.py
"""

import csv
import json
from pathlib import Path

INPUT  = Path(__file__).parent / "rentalcars-1year.csv"
OUTPUT = Path(__file__).parent / "rentalcars_aggregate_request.json"

RATING_FIELD_MAP = {
    "aggregateRating":       "aggregate_rating",
    "renterRating":          "renter_rating",
    "carConditionRating":    "car_condition_rating",
    "processingSpeedRating": "processing_speed_rating",
    "providerCareRating":    "provider_care_rating",
    "serviceLevelRating":    "service_level_rating",
    "recommendationRating":  "recommendation_rating",
}


def _extract(row: dict) -> dict | None:
    review = row.get("renterQuote", "").strip()
    if not review:
        return None

    database_id = row.get("_id", "").strip()
    result = {"review": review, "database_id": int(float(database_id)) if database_id else None}
    for csv_field, request_field in RATING_FIELD_MAP.items():
        value = row.get(csv_field, "").strip()
        result[request_field] = float(value) if value else None
    return result


with open(INPUT, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

reviews = [r for row in rows if (r := _extract(row))]
provider = rows[0]["provider"] if rows else None

payload = {"provider": provider, "reviews": reviews}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"Saved {len(reviews)} reviews (provider={provider!r}) to {OUTPUT}")
print(f"\ncurl -X POST http://localhost:8742/infer/aggregate \\")
print(f"  -H 'Content-Type: application/json' \\")
print(f"  -d @{OUTPUT}")
