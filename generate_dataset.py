"""
Generates synthetic car rental reviews using Groq API (Llama 3.3 70B).
Saves to data_processing/car_rental_reviews.jsonl — resumes if interrupted.

Usage:
    python3 generate_dataset.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OUTPUT_FILE  = Path("data_processing/car_rental_reviews.jsonl")
TOTAL        = 10_000
BATCH_SIZE   = 20    # Groq free tier: keep batches small
# Models tried in order — falls back when daily token limit is hit
MODELS = [
    "llama-3.3-70b-versatile",   # 100K TPD  — best quality
    "llama-3.1-8b-instant",      # 500K TPD  — fast fallback
    "gemma2-9b-it",              # 500K TPD  — second fallback
]

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
- For POSITIVE reviews the summary must NOT contain "but" or any negative clause
- For NEGATIVE reviews the summary must NOT contain "however" or any positive clause
- For MIXED reviews use "however" or "but" to contrast pros and cons

Examples:
{{"review_body": "Fantastic experience from start to finish. The car was spotless and ready on time. Staff were friendly and professional with no upselling pressure whatsoever. Return at Lyon airport took under five minutes.", "summary": "Smooth and pleasant rental experience with a clean car and professional staff. Pickup and return were both fast and hassle-free.", "categories": ["Cleanliness", "Pickup Experience", "Return Experience"]}}
{{"review_body": "I picked up a compact car at Munich airport last week. The queue at the counter took nearly an hour despite my online reservation. Once I finally got the car, it was clean and in good shape. However, on my final invoice I noticed a 45 EUR airport surcharge that was never mentioned during booking.", "summary": "Long wait at pickup despite a pre-existing reservation. An undisclosed airport surcharge appeared on the final invoice.", "categories": ["Pickup Experience", "Hidden Fees & Billing"]}}
{{"review_body": "Nightmare from start to finish. The car smelled of cigarettes and had visible damage not noted on the contract. Staff were dismissive when I raised concerns and refused to provide an updated damage report.", "summary": "Terrible experience with a dirty, damaged vehicle and unhelpful staff who refused to acknowledge pre-existing damage.", "categories": ["Cleanliness", "Vehicle Condition", "Staff & Communication"]}}

Generate exactly {n} lines now:"""


def count_existing() -> int:
    if not OUTPUT_FILE.exists():
        return 0
    with open(OUTPUT_FILE) as f:
        return sum(1 for _ in f)


def generate_batch(client: Groq, n: int, model: str) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(n=n, cats=", ".join(CATEGORIES))
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
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
            msg = str(e)
            if "rate_limit_exceeded" in msg and "tokens per day" in msg:
                raise  # let caller switch model
            print(f"\n  Attempt {attempt + 1}/3 failed: {e}")
            time.sleep(15)
    return []


def main():
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY not found in .env — get a free key at console.groq.com")

    client = Groq(api_key=GROQ_API_KEY)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    existing = count_existing()
    if existing >= TOTAL:
        print(f"✅ Already have {existing:,} reviews. Nothing to do.")
        return

    remaining = TOTAL - existing
    batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Generating {remaining:,} reviews in {batches} batches (resuming from {existing:,})...\n")

    total_written = existing
    model_index = 0
    current_model = MODELS[model_index]
    print(f"Using model: {current_model}\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(batches):
            size = min(BATCH_SIZE, TOTAL - total_written)
            print(f"  Batch {i + 1}/{batches} ({size} reviews) [{current_model}]...", end=" ", flush=True)

            try:
                reviews = generate_batch(client, size, current_model)
            except Exception as e:
                if "tokens per day" in str(e) and model_index + 1 < len(MODELS):
                    model_index += 1
                    current_model = MODELS[model_index]
                    print(f"\n  Daily limit hit — switching to {current_model}")
                    reviews = generate_batch(client, size, current_model)
                else:
                    print(f"\n  Failed: {e}")
                    reviews = []

            for r in reviews:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

            total_written += len(reviews)
            print(f"✅  ({total_written:,} / {TOTAL:,} total)")
            time.sleep(3)

    print(f"\n✅ Done! {total_written:,} reviews saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
