import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
            
    percentage = 100 * trainable_model_params / all_model_params
    report = (
        f"📊 Model Parameter Report:\n"
        f"--------------------------------------------------\n"
        f"  Total Parameters:     {all_model_params:,}\n"
        f"  Trainable Parameters: {trainable_model_params:,}\n"
        f"  Percentage Trainable: {percentage:.4f}%\n"
        f"--------------------------------------------------"
    )
    return report

def load_llm_model(device, model_name='google/flan-t5-base'):
    print("\n🧠 Step 2: Loading the LLM and Tokenizer...")
    
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

    print(f"✅ {model_name} and Tokenizer loaded securely using {target_dtype}.")
    print("\n" + print_number_of_trainable_model_parameters(original_model))
    
    return original_model, tokenizer


