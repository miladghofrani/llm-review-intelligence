"""
Converts a Floyt CSV export into an /infer/aggregate request payload.
Handles CSVs identified by either a "provider" or "renter" column.

Usage:
    python3 floyt/convert_csv.py <filename.csv>   # e.g. rentalcars-1year.csv
    python3 floyt/convert_csv.py                  # defaults to rentalcars-1year.csv
"""

import csv
import json
import sys
from pathlib import Path

INPUT  = Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "rentalcars-1year.csv")
OUTPUT = INPUT.parent / f"{INPUT.stem.replace('-', '_')}_aggregate_request.json"

RATING_FIELD_MAP = {
    "carConditionRating":    "car_condition_rating",
    "processingSpeedRating": "processing_speed_rating",
    "serviceLevelRating":    "service_level_rating",
    "recommendationRating":  "recommendation_rating",
}


def _extract(row: dict) -> dict | None:
    review = (row.get("renterQuote") or "").strip()
    if not review:
        return None

    database_id = (row.get("_id") or "").strip()
    result = {"review": review, "database_id": int(float(database_id)) if database_id else None}
    for csv_field, request_field in RATING_FIELD_MAP.items():
        value = (row.get(csv_field) or "").strip()
        result[request_field] = float(value) if value else None
    return result


with open(INPUT, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

reviews  = [r for row in rows if (r := _extract(row))]
provider = (rows[0].get("provider") or None) if rows else None
renter   = (rows[0].get("renter") or None) if rows else None

payload = {"reviews": reviews}
if provider:
    payload["provider"] = provider
if renter:
    payload["renter"] = renter

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"Saved {len(reviews)} reviews (provider={provider!r}, renter={renter!r}) to {OUTPUT}")
print(f"\ncurl -X POST http://localhost:8742/infer/aggregate \\")
print(f"  -H 'Content-Type: application/json' \\")
print(f"  -d @{OUTPUT}")
