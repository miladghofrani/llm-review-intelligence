import time
import torch
import pandas as pd
import numpy as np
import evaluate

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    GenerationConfig, 
    TrainingArguments, 
    Trainer
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def load_amazon_reviews_mock():
    print("\n📥 Step 1: Fetching Real-World Customer Reviews...")
    dataset_name = "goosmanlei/amazon_reviews_multi"
    
    dataset = load_dataset(dataset_name, "en")
    
    print("✅ Dataset loaded successfully!")
    
    print("\n🔍 Sample Real-World Review:")
    print("-" * 80)
    print(f"📦 Category: {dataset['train'][0]['product_category']}")
    print(f"📝 Summary (Title): {dataset['train'][0]['review_title']}")
    print(f"🗣️ Full Review: {dataset['train'][0]['review_body']}")
    print("-" * 80)
    
    return dataset

def load_company_reviews_mock():
    """
    Loads a mock dataset (DialogSum) to simulate company customer reviews.
    Later, this function will be replaced with an SQL connection to the company DB.
    """
    print("\n📥 Fetching data from Hugging Face (Mocking Company DB)...")
    dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(dataset_name)
    
    print("✅ Dataset loaded successfully!")
    # print(f"📊 Dataset Structure:\n{dataset}")
    
    # Print the very first review/dialogue from the test set to inspect it
    # print("\n🔍 Sample Data (Review/Dialogue):")
    # print("--------------------------------------------------")
    # print(dataset['test'][0]['dialogue'])
    # print("--------------------------------------------------")
    # print("📝 Human Summary:")
    # print(dataset['test'][0]['summary'])
    # print("--------------------------------------------------")
    
    return dataset

def print_number_of_trainable_model_parameters(model):
    """
    Calculates and prints the number of trainable parameters in the model.
    This is crucial for monitoring memory usage before and after applying PEFT/LoRA.
    """
    trainable_model_params = 0
    all_model_params = 0
    
    # Iterate through all layers and count parameters
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
            
    # Calculate percentage
    percentage = 100 * trainable_model_params / all_model_params
    
    # Format the output beautifully
    report = (
        f"📊 Model Parameter Report:\n"
        f"--------------------------------------------------\n"
        f"  Total Parameters:     {all_model_params:,}\n"
        f"  Trainable Parameters: {trainable_model_params:,}\n"
        f"  Percentage Trainable: {percentage:.2f}%\n"
        f"--------------------------------------------------"
    )
    return report

def load_llm_model(device):
    print("\n🧠 Step 2: Loading the LLM and Tokenizer...")
    model_name = 'google/flan-t5-base'
    
    # 1. Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer loaded.")

    target_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # 2. Load the Model with memory optimization (bfloat16)
    # This reduces RAM usage by half, perfect for local testing and AWS efficiency.
    original_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        dtype=target_dtype
    ).to(device)

    print(f"✅ FLAN-T5-Base and Tokenizer loaded securely using {target_dtype}.")
    print("\n" + print_number_of_trainable_model_parameters(original_model))
    
    return original_model, tokenizer

def test_zero_shot_inference(model, tokenizer, dataset, device):
    """
    Tests the base model's ability to summarize a dialogue without any examples.
    """
    print("\n🧪 Running Zero-Shot Inference Test...")
    
    # Select a random index (like in the lab)
    index = 200
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    # Create the prompt mimicking our future review summarization
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
    """

    # 1. Tokenize the input and move to the correct hardware device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # 2. Generate the output using the model
    output_tokens = model.generate(
        inputs["input_ids"], 
        max_new_tokens=200
    )[0]
    
    # 3. Decode the output back to human-readable text
    output = tokenizer.decode(output_tokens, skip_special_tokens=True)

    # Print the results beautifully
    dash_line = '-' * 80
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')
    print(dash_line)

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

def setup_peft_lora_model(original_model):
    """
    Applies LoRA to the base model, freezing the original weights and 
    adding a small, trainable adapter.
    """
    print("\n🪄 Step 6: Injecting LoRA Adapters into the Model (PEFT)...")
    
    # Define the LoRA configuration based on the lab parameters
    lora_config = LoraConfig(
        r=32, # Rank of the adapter matrices
        lora_alpha=32, # Scaling factor
        target_modules=["q", "v"], # Targeting the Attention mechanism
        lora_dropout=0.05, # Regularization to prevent overfitting
        bias="none", 
        task_type=TaskType.SEQ_2_SEQ_LM # Model type definition
    )
    
    # Wrap the original model with the PEFT model
    peft_model = get_peft_model(original_model, lora_config)
    
    print("✅ LoRA adapters injected successfully!")
    
    # Print the dramatic reduction in trainable parameters!
    print("\n" + print_number_of_trainable_model_parameters(peft_model))
    
    return peft_model

def train_and_save_peft_model(peft_model, tokenizer, tokenized_datasets):
    """
    Executes the PEFT training loop and saves ONLY the small adapter weights.
    We keep max_steps=1 for the dry run.
    """
    print("\n🏋️ Step 7: Starting PEFT Training (Dry Run)...")
    
    output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'
    
    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, 
        num_train_epochs=1,
        logging_steps=1,
        max_steps=1  # <--- CRITICAL: Remove this in AWS production to train fully!
    )
    
    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )
    
    print("⏳ Running PEFT Trainer...")
    peft_trainer.train()
    
    # Save the adapter and tokenizer locally
    peft_model_path = "./peft-dialogue-summary-checkpoint-local"
    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)
    
    print(f"✅ PEFT Model (Adapters) saved successfully in: {peft_model_path}")
    return peft_model_path

def load_saved_peft_model(device, adapter_path):
    """
    Simulates production deployment: Loads the base model and attaches 
    the trained LoRA adapters for fast inference.
    """
    print("\n🚀 Step 8: Assembling the Production Model for Inference...")
    
    # 1. Load the base model again
    model_name = "google/flan-t5-base"
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # 2. Attach the LoRA adapter we saved in Step 7
    # is_trainable=False is crucial here for saving memory during inference
    peft_model_for_inference = PeftModel.from_pretrained(
        peft_model_base, 
        adapter_path, 
        torch_dtype=torch.bfloat16,
        is_trainable=False
    ).to(device)
    
    print("✅ Base Model + LoRA Adapter assembled successfully!")
    return peft_model_for_inference

def test_custom_review():
    print("🚀 Loading the Production Model for Custom Inference...")
    
    device = "cpu"
    
    base_model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float32
    ).to(device)
    
    adapter_path = "./peft-dialogue-summary-checkpoint-local" 
    try:
        production_model = PeftModel.from_pretrained(
            base_model, 
            adapter_path, 
            torch_dtype=torch.float32,
            is_trainable=False
        ).to(device)
        print("✅ Custom LoRA Adapter attached successfully!\n")
    except Exception as e:
        print(f"❌ Error loading adapter. Ensure '{adapter_path}' exists.")
        return

    custom_review = """
    I rented a car for my weekend trip to the mountains. The engine ran fine, 
    but the interior was extremely dirty. There were sticky coffee stains on the passenger 
    seat and the whole cabin smelled like cheap cigarette smoke. Also, their mobile app 
    crashed twice when I tried to extend my rental period. Very frustrating experience!
    """

    prompt = f"Summarize the following review.\n\n{custom_review}\n\nSummary: "

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    output_tokens = production_model.generate(
        input_ids=inputs["input_ids"], 
        max_new_tokens=50,
        temperature=0.5,
        do_sample=True
    )[0]
    
    summary = tokenizer.decode(output_tokens, skip_special_tokens=True)

    print("-" * 80)
    print(f"📝 CUSTOM REVIEW:\n{custom_review.strip()}")
    print("-" * 80)
    print(f"✨ AI GENERATED SUMMARY:\n{summary}")
    print("-" * 80)

def main():
    """
    Main entry point for the LLM Review Classification & Summarization Pipeline.
    """
    print("🚀 Initializing the LLM Pipeline...")
    print("--------------------------------------------------")
    
    # Check if all libraries are loaded successfully
    print("✅ All required libraries imported successfully.")
    
    # Check PyTorch version
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # For Docker running on CPU (during local Mac testing)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Processing device set to: {device.upper()}")
    print("--------------------------------------------------")

    # Step 1: Load the Data
    # dataset = load_company_reviews_mock()

    dataset = load_amazon_reviews_mock()

    # Step 2: Load the Model & Tokenizer
    original_model, tokenizer = load_llm_model(device)

    # # Step 3: Run Zero-Shot Inference
    # test_zero_shot_inference(original_model, tokenizer, dataset, device)

    # Step 4: Tokenize Dataset
    tokenized_dataset = preprocess_dataset(tokenizer, dataset)

    # Step 5: Apply subsampling to save time
    small_tokenized_dataset = subsample_dataset(tokenized_dataset)

    # Step 6: Setup PEFT Model
    peft_model = setup_peft_lora_model(original_model)

    # Step 7: Train PEFT and Save
    saved_peft_path = train_and_save_peft_model(peft_model, tokenizer, small_tokenized_dataset)

    # Step 8: Load the saved PEFT model for inference (Production Simulation)
    production_model = load_saved_peft_model(device, saved_peft_path)

    print("Ready for the next step!")

if __name__ == "__main__":
    test_custom_review()
    # main()