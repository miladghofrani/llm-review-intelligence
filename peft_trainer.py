import time
import torch

from model_loader import print_number_of_trainable_model_parameters
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    GenerationConfig, 
    TrainingArguments, 
    Trainer
)

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