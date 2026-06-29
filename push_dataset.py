"""
Pushes data_processing/car_rental_reviews.jsonl to HF Hub as
miladghofrani/car-rental-reviews.

Usage:
    python3 push_dataset.py
"""

import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HF_REPO   = "miladghofrani/car-rental-reviews"
JSONL     = Path("data_processing/car_rental_reviews.jsonl")


def main():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN not found in .env")
    if not JSONL.exists():
        raise SystemExit(f"{JSONL} not found")

    login(token=HF_TOKEN)

    print(f"Loading {JSONL}...")
    dataset = load_dataset("json", data_files=str(JSONL), split="train")
    print(f"Loaded {dataset.num_rows:,} reviews")

    print(f"Pushing to {HF_REPO}...")
    dataset.push_to_hub(HF_REPO, private=False)
    print(f"✅ Done — https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
