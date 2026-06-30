def tokenize_dataset(tokenizer, dataset):
    """
    Tokenizes a DatasetDict with 'input' and 'output' columns into
    'input_ids' and 'labels' columns ready for seq2seq training.
    """
    print("\n⚙️  Tokenizing dataset for seq2seq training...")

    pad_id = tokenizer.pad_token_id

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input"],
            truncation=True,
            max_length=512,
        )
        labels = tokenizer(
            batch["output"],
            truncation=True,
            max_length=128,
        )
        # Replace padding in labels with -100 so the loss ignores them.
        # No padding here — DataCollatorForSeq2Seq pads dynamically per batch,
        # which is faster than padding everything to max_length upfront.
        model_inputs["labels"] = [
            [(t if t != pad_id else -100) for t in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["input", "output"])
    print("✅ Tokenization complete.")
    return tokenized
