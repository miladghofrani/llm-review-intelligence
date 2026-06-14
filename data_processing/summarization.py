from datasets import load_dataset

def load_dialogue_dataset(dataset_name = "knkarthick/dialogsum"):
    print("\n📥 Fetching data from Hugging Face (Mocking Company DB)...")
    dataset = load_dataset(dataset_name)
    
    print("✅ Dataset loaded successfully!")
    print(f"📊 Dataset Structure:\n{dataset}")
    
    # Print the very first review/dialogue from the test set to inspect it
    print("\n🔍 Sample Data (Review/Dialogue):")
    print("--------------------------------------------------")
    print(dataset['test'][0]['dialogue'])
    print("--------------------------------------------------")
    print("📝 Human Summary:")
    print(dataset['test'][0]['summary'])
    print("--------------------------------------------------")
    
    return dataset

def preprocess_dataset(tokenizer, dataset):
    print("\n⚙️ Step 4: Preprocessing and Tokenizing the Dataset...")

    def tokenize_function(example):
        start_prompt = 'Summarize the following dialogue.\n\n'
        end_prompt = '\n\nSummary: '

        safe_dialogues = [str(dialogue) if dialogue is not None else "" for dialogue in example["dialogue"]]
        safe_summaries = [str(summary) if summary is not None else "" for summary in example["summary"]]

        prompt = [start_prompt + dialogue + end_prompt for dialogue in safe_dialogues]
        example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(safe_summaries, padding="max_length", truncation=True, return_tensors="pt").input_ids
        return example

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(tokenized_datasets)
    columns_to_remove = ['id', 'dialogue', 'summary', 'topic']
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    print("✅ Dataset successfully tokenized and cleaned.")
    return tokenized_datasets

def subsample_dataset(tokenized_datasets):
    """
    Subsamples the dataset to speed up local testing.
    In a production AWS environment, you would bypass this to train on the full DB.
    """
    print("\n✂️ Step 5: Subsampling the Dataset for Faster Local Testing...")
    
    # Keep only 1 out of every 100 records
    small_dataset = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)
    
    print("✅ Dataset significantly reduced for rapid prototyping.")
    print(f"📊 Original Training Data Shape: {tokenized_datasets['train'].shape}")
    print(f"📊 New (Subsampled) Training: {small_dataset['train'].shape}")
    print(f"📊 New (Subsampled) Validation: {small_dataset['validation'].shape}")
    print(f"📊 New (Subsampled) Test: {small_dataset['test'].shape}")
    
    return small_dataset