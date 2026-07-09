"""
Translates English summaries to match the review's native language (DE or FR).

Reads  : data_processing/reviews_labeled_short.jsonl
Writes : data_processing/reviews_labeled_mt5.jsonl  (new file, original untouched)

Each output row is identical to the input row except `summary` is now in the
same language as `review_body`, making it ready to fine-tune mT5_multilingual_XLSum
for monolingual summarization (DE→DE, FR→FR).

English reviews and all other languages keep their existing English summaries.
"""

import json
import os
from pathlib import Path

import deepl
from langdetect import detect, LangDetectException

INPUT_FILE  = Path(__file__).parent / "reviews_labeled_short.jsonl"
OUTPUT_FILE = Path(__file__).parent / "reviews_labeled_mt5.jsonl"

DEEPL_TARGET = {"de": "DE", "fr": "FR"}  # only translate these two


def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def main():
    api_key = os.environ.get("DEEPL_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPL_API_KEY not set in environment")

    client = deepl.Translator(api_key)

    print(f"Reading {INPUT_FILE.name}...")
    rows = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  {len(rows):,} rows loaded")

    # Detect language for every review
    print("Detecting review languages...")
    for row in rows:
        row["_lang"] = detect_lang(row["review_body"])

    # Group indices by target DeepL language
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        target = DEEPL_TARGET.get(row["_lang"])
        if target:
            groups.setdefault(target, []).append(i)

    total_to_translate = sum(len(v) for v in groups.values())
    print(f"  Will translate {total_to_translate:,} summaries "
          f"({', '.join(f'{len(v)} {k}' for k, v in groups.items())})")

    # Estimate character count
    chars = sum(len(rows[i]["summary"]) for idxs in groups.values() for i in idxs)
    print(f"  Estimated characters: {chars:,}  (free tier limit: 500,000)")

    # Batch translate per target language
    BATCH = 50  # DeepL batch size
    for target_lang, indices in groups.items():
        print(f"\nTranslating {len(indices):,} summaries → {target_lang}...")
        for batch_start in range(0, len(indices), BATCH):
            batch_idx = indices[batch_start: batch_start + BATCH]
            texts = [rows[i]["summary"] for i in batch_idx]
            results = client.translate_text(texts, target_lang=target_lang)
            for i, result in zip(batch_idx, results):
                rows[i]["summary"] = result.text
            done = min(batch_start + BATCH, len(indices))
            print(f"  {done}/{len(indices)}", end="\r")
        print(f"  {len(indices)}/{len(indices)} done")

    # Write output (remove temp _lang key)
    print(f"\nWriting {OUTPUT_FILE.name}...")
    with open(OUTPUT_FILE, "w") as f:
        for row in rows:
            row.pop("_lang", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(rows):,} rows to {OUTPUT_FILE}")
    print("\nSample (DE):")
    for row in rows[:3]:
        print(f"  review: {row['review_body'][:80]}...")
        print(f"  summary: {row['summary']}")
        print()


if __name__ == "__main__":
    main()
