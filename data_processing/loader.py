import requests
from datasets import load_dataset, Dataset, DatasetDict
from config import DATASET_NAME, DATASET_SUBSET, SUBSAMPLE_RATIO, CATEGORIES
from data_processing.labeler import label_review, format_labels


def load_review_dataset():
    """
    Loads automotive reviews via Parquet files from the HF Datasets Server.
    Bypasses script-based loading (removed in datasets 3.x).
    Fields used: review_body (input text), review_headline (summary target).
    """
    print(f"\n📥 Loading {DATASET_NAME} / {DATASET_SUBSET} from Hugging Face...")

    api_url = (
        f"https://datasets-server.huggingface.co/parquet"
        f"?dataset={DATASET_NAME}&config={DATASET_SUBSET}"
    )
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    parquet_urls = [f["url"] for f in resp.json().get("parquet_files", [])]

    if not parquet_urls:
        raise RuntimeError(
            f"No Parquet files found for {DATASET_NAME}/{DATASET_SUBSET}.\n"
            f"Check: {api_url}"
        )

    raw = load_dataset("parquet", data_files={"train": parquet_urls}, split="train")
    raw = raw.rename_columns({"text": "review_body", "title": "review_headline"})
    dataset = DatasetDict({"train": raw})

    print(f"✅ Loaded {dataset['train'].num_rows:,} reviews")
    sample = dataset["train"][0]
    print("\n🔍 Sample Review:")
    print("-" * 60)
    print(f"Review : {sample['review_body'][:200]}...")
    print(f"Headline: {sample['review_headline']}")
    print("-" * 60)

    return dataset


def build_multitask_dataset(dataset):
    """
    Converts raw reviews into two training example types per review:

    1. Summarization  → input: full review  /  output: review headline
    2. Classification → input: full review  /  output: comma-separated categories

    Returns a DatasetDict with 'train' and 'test' splits (90/10).
    """
    print(f"\n⚙️  Building multi-task examples (1 in every {SUBSAMPLE_RATIO} reviews)...")

    categories_str = ", ".join(CATEGORIES)
    inputs, outputs = [], []

    train_split = dataset["train"]
    for idx, row in enumerate(train_split):
        if idx % SUBSAMPLE_RATIO != 0:
            continue

        body = (row.get("review_body") or "").strip()
        headline = (row.get("review_headline") or "").strip()
        if not body or not headline:
            continue

        # --- Summarization example ---
        inputs.append(f"Summarize the following car rental review.\n\n{body}\n\nSummary:")
        outputs.append(headline)

        # --- Classification example ---
        inputs.append(
            f"Classify this car rental review into one or more of these categories: "
            f"{categories_str}.\n\n"
            f"Review: {body}\n\nCategories:"
        )
        outputs.append(format_labels(label_review(body)))

    full = Dataset.from_dict({"input": inputs, "output": outputs})
    split = full.train_test_split(test_size=0.1, seed=42)

    print(f"✅ {split['train'].num_rows:,} train / {split['test'].num_rows:,} validation examples")
    print(f"   (each review generates one summarization + one classification example)")
    return split
