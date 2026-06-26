"""
Generates 10,000 synthetic car rental reviews using HuggingFace Inference API.
Uses your existing HF_TOKEN from .env — no extra billing needed.
Saves to data_processing/car_rental_reviews.jsonl (resumes if interrupted).

Usage:
    python3 generate_dataset.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
OUTPUT_FILE = Path("data_processing/car_rental_reviews.jsonl")
TOTAL       = 10_000
BATCH_SIZE  = 25   # smaller batches are more reliable on free tier
MODEL       = "Qwen/Qwen2.5-72B-Instruct"

CATEGORIES = [
    "Cleanliness",
    "Vehicle Condition",
    "Pickup Experience",
    "Return Experience",
    "Hidden Fees & Billing",
    "Insurance & Upselling",
    "Staff & Communication",
    "Booking & App",
]

PROMPT_TEMPLATE = """Generate {n} diverse, realistic car rental customer reviews in JSONL format.
Output ONLY one JSON object per line — no array brackets, no markdown, no explanation.

Each line must have exactly these fields:
- "review_body": A realistic review (4-8 sentences). Written naturally as a real customer.
- "summary": 2-3 sentences capturing the main points and overall sentiment.
- "categories": 1-3 items from: {cats}

Rules:
- 40% positive, 40% negative, 20% mixed sentiment
- Vary: locations (airports, city centers, train stations), car types, rental companies
- Cover all 8 categories roughly equally across the batch
- Summaries must genuinely condense the review — not just repeat the first sentence

Example:
{{"review_body": "I picked up a compact car at Munich airport last week. The queue at the counter took nearly an hour despite my online reservation. Once I finally got the car, it was clean and in good shape. However, on my final invoice I noticed a 45 EUR airport surcharge that was never mentioned during booking.", "summary": "Long wait at pickup despite a pre-existing reservation. The car itself was fine, but an undisclosed airport surcharge appeared on the final invoice.", "categories": ["Pickup Experience", "Hidden Fees & Billing"]}}

Generate exactly {n} lines now:"""


def count_existing() -> int:
    if not OUTPUT_FILE.exists():
        return 0
    with open(OUTPUT_FILE) as f:
        return sum(1 for _ in f)


def generate_batch(client: InferenceClient, n: int) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(n=n, cats=", ".join(CATEGORIES))
    for attempt in range(3):
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=6000,
                temperature=0.9,
            )
            text = response.choices[0].message.content
            lines = [l.strip() for l in text.strip().splitlines() if l.strip().startswith("{")]
            reviews = [json.loads(l) for l in lines]
            if len(reviews) < n // 2:
                raise ValueError(f"Only got {len(reviews)} reviews, expected ~{n}")
            return reviews
        except Exception as e:
            print(f"\n  Attempt {attempt + 1}/3 failed: {e}")
            time.sleep(20)
    return []


def main():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN not found in .env")

    client = InferenceClient(model=MODEL, token=HF_TOKEN)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    existing = count_existing()
    if existing >= TOTAL:
        print(f"✅ Already have {existing:,} reviews. Nothing to do.")
        return

    remaining = TOTAL - existing
    batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Generating {remaining:,} reviews in {batches} batches (resuming from {existing:,})...\n")

    total_written = existing
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(batches):
            size = min(BATCH_SIZE, TOTAL - total_written)
            print(f"  Batch {i + 1}/{batches} ({size} reviews)...", end=" ", flush=True)

            reviews = generate_batch(client, size)

            for r in reviews:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

            total_written += len(reviews)
            print(f"✅  ({total_written:,} / {TOTAL:,} total)")
            time.sleep(2)

    print(f"\n✅ Done! {total_written:,} reviews saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
