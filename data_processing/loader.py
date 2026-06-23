def load_review_dataset():
    """
    Loads fancyzhx/amazon_polarity — a Parquet-native dataset with no loading script.
    Columns: title (headline), content (body), label (0=neg / 1=pos).
    """
    print(f"\n📥 Loading {DATASET_NAME} from Hugging Face...")

    raw = load_dataset(DATASET_NAME, split="train")
    raw = raw.rename_columns({"content": "review_body", "title": "review_headline"})
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
