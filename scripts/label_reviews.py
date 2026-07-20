"""
Labels real Floyt reviews with sentiment and categories using Claude Haiku.
Reads  : data_processing/reviews_raw.json
Writes : data_processing/reviews_labeled.jsonl  (resumes if interrupted)

Usage:
    python3 scripts/label_reviews.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
INPUT_FILE     = Path("data_processing/reviews_raw.json")
OUTPUT_FILE    = Path("data_processing/reviews_labeled_short.jsonl")

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

SYSTEM_PROMPT = f"""You are a car rental review analyst. Given a customer review (may be in German or French),
you output a JSON object with exactly these fields:

- "review_body": the original review text, unchanged
- "sentiment": exactly one of: "positive", "negative", "mixed"
- "categories": a JSON array of 1-3 items from this exact list:
  {json.dumps(CATEGORIES)}

Output ONLY the JSON object — no markdown, no explanation."""

USER_PROMPT = "Review:\n{review}"


SPAM_KEYWORDS = [
    "poc review", "please disregard", "please ignore", "bugbounty",
    "placeholder", "test review", "test entry", "security validation",
    "controlled booking",
]


def _is_spam(review: str) -> bool:
    t = review.lower()
    return len(review) < 20 or any(k in t for k in SPAM_KEYWORDS)


def load_already_done() -> set[str]:
    done = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["review_body"][:80])
                except Exception:
                    pass
    return done


def label_review(client: anthropic.Anthropic, review: str) -> dict | None:
    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_PROMPT.format(review=review)}],
            )
            text = message.content[0].text.strip()
            # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            obj = json.loads(text)
            # Validate required fields
            if not all(k in obj for k in ("review_body", "sentiment", "categories")):
                raise ValueError("Missing fields in response")
            if obj["sentiment"] not in ("positive", "negative", "mixed"):
                obj["sentiment"] = "negative"
            # Normalize categories to valid values only
            obj["categories"] = [c for c in obj["categories"] if c in CATEGORIES]
            if not obj["categories"]:
                obj["categories"] = ["Staff & Communication"]
            return obj
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"\n  Failed after 3 attempts: {e}")
    return None


def main():
    if not CLAUDE_API_KEY:
        raise SystemExit("CLAUDE_API_KEY not found in .env")
    if not INPUT_FILE.exists():
        raise SystemExit(f"{INPUT_FILE} not found — run the CSV conversion first")

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    reviews = json.loads(INPUT_FILE.read_text())
    done    = load_already_done()

    clean     = [r for r in reviews if not _is_spam(r["review"])]
    spam_count = len(reviews) - len(clean)
    remaining  = [r for r in clean if r["review"][:80] not in done]
    print(f"Total: {len(reviews)} | Spam filtered: {spam_count} | Already done: {len(done)} | Remaining: {len(remaining)}\n")

    if not remaining:
        print("✅ All reviews already labeled.")
        return

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i, entry in enumerate(remaining, start=1):
            review = entry["review"]
            print(f"  [{i}/{len(remaining)}] {review[:60].replace(chr(10), ' ')}...", end=" ", flush=True)

            result = label_review(client, review)
            if result is None:
                print("❌ skipped")
                continue

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"✅  ({result['categories']})")

            # Avoid hitting rate limits
            time.sleep(0.3)

    total = sum(1 for _ in open(OUTPUT_FILE))
    print(f"\n✅ Done! {total} labeled reviews saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
