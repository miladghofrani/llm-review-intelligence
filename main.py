import torch
from data_processing.summarization import load_dialogue_dataset, preprocess_dataset, subsample_dataset

from model_loader import load_llm_model
from peft_trainer import load_saved_peft_model, setup_peft_lora_model, train_and_save_peft_model

def get_device():
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
    return device

def train_peft_model_for_summarization():
    device = get_device()

    original_model, tokenizer = load_llm_model(device)

    # Train model for summarization base on dialogue dataset
    dataset = load_dialogue_dataset()
    tokenized_dataset = preprocess_dataset(tokenizer, dataset)
    small_tokenized_dataset = subsample_dataset(tokenized_dataset)

    peft_model = setup_peft_lora_model(original_model)
    train_and_save_peft_model(peft_model, tokenizer, small_tokenized_dataset)

    print("✅ PEFT training complete! You can now deploy the model for inference using the saved adapter weights.")

if __name__ == "__main__":
    train_peft_model_for_summarization()