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
        start_prompt = 'Summarize the following review.\n\n'
        end_prompt = '\n\nSummary: '

        safe_reviews = [str(review) if review is not None else "" for review in example["review_body"]]
        safe_titles = [str(title) if title is not None else "" for title in example["review_title"]]

        prompt = [start_prompt + review + end_prompt for review in safe_reviews]
        example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(safe_titles, padding="max_length", truncation=True, return_tensors="pt").input_ids
        return example

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(tokenized_datasets)
    columns_to_remove = ['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category']
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