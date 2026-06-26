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
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        labels = tokenizer(
            batch["output"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        # Replace padding in labels with -100 so the loss ignores them
        model_inputs["labels"] = [
            [(t if t != pad_id else -100) for t in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["input", "output"])
    print("✅ Tokenization complete.")
    return tokenized
