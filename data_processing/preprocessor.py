def tokenize_dataset(tokenizer, dataset):
    """
    Tokenizes a DatasetDict with 'input' and 'output' columns into
    'input_ids' and 'labels' columns ready for seq2seq training.
    """
    print("\n⚙️  Tokenizing dataset for seq2seq training...")

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
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["input", "output"])
    print("✅ Tokenization complete.")
    return tokenized
