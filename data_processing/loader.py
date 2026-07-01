from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict

from config import CATEGORIES

LOCAL_FILE = Path(__file__).parent / "car_rental_reviews.jsonl"
HF_DATASET = "miladghofrani/car-rental-reviews"


def load_review_dataset():
    """
    Loads the car rental review dataset.
    Uses the local JSONL file if available, otherwise loads from HF Hub.
    Columns: review_body, summary, categories.
    """
    if LOCAL_FILE.exists():
        print(f"\n📥 Loading local dataset from {LOCAL_FILE.name}...")
        raw = load_dataset("json", data_files={"train": str(LOCAL_FILE)}, split="train")
    else:
        print(f"\n📥 Loading dataset from HF Hub ({HF_DATASET})...")
        raw = load_dataset(HF_DATASET, split="train")

    dataset = DatasetDict({"train": raw})
    print(f"✅ Loaded {dataset['train'].num_rows:,} reviews")

    sample = dataset["train"][0]
    print("\n🔍 Sample Review:")
    print("-" * 60)
    print(f"Review    : {sample['review_body'][:200]}...")
    print(f"Summary   : {sample['summary']}")
    print(f"Categories: {sample['categories']}")
    print("-" * 60)

    return dataset


def build_multitask_dataset(dataset):
    """
    Converts raw reviews into three training examples per review:
    1. Summarization  — input: full review  /  output: summary
    2. Classification — input: full review  /  output: comma-separated categories
    3. Sentiment      — input: full review  /  output: positive | negative | mixed

    Returns a DatasetDict with 'train' and 'test' splits (90/10).
    """
    print(f"\n⚙️  Building multi-task examples...")

    categories_str = ", ".join(CATEGORIES)
    valid_sentiments = {"positive", "negative", "mixed"}
    inputs, outputs = [], []

    for row in dataset["train"]:
        body       = (row.get("review_body") or "").strip()
        summary    = (row.get("summary") or "").strip()
        categories = row.get("categories") or []
        sentiment  = (row.get("sentiment") or "").strip().lower()

        if not body or not summary:
            continue

        inputs.append(f"Summarize the following car rental review.\n\n{body}\n\nSummary:")
        outputs.append(summary)

        inputs.append(
            f"Classify this car rental review into one or more of these categories: "
            f"{categories_str}.\n\nReview: {body}\n\nCategories:"
        )
        outputs.append(", ".join(categories) if categories else "Staff & Communication")

        if sentiment in valid_sentiments:
            inputs.append(
                f"What is the overall sentiment of this car rental review? "
                f"Answer with exactly one word: positive, negative, or mixed.\n\n"
                f"Review: {body}\n\nSentiment:"
            )
            outputs.append(sentiment)

    full = Dataset.from_dict({"input": inputs, "output": outputs})
    split = full.train_test_split(test_size=0.1, seed=42)

    print(f"✅ {split['train'].num_rows:,} train / {split['test'].num_rows:,} validation examples")
    return split
