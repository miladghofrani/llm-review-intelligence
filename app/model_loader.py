import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MODEL_NAME, MT5_MODEL_NAME, MT5_ADAPTER_PATH

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

def load_tokenizer(model_name=MODEL_NAME):
    print("\n🔤 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✅ Tokenizer loaded for {model_name}.")
    return tokenizer

def load_llm_model(device, model_name=MODEL_NAME):
    print("\n🧠 Loading the LLM...")
    target_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=target_dtype).to(device)
    print(f"✅ {model_name} loaded using {target_dtype}.")
    return model


def load_mt5_tokenizer(model_name=MT5_ADAPTER_PATH):
    print("\n🔤 Loading mT5 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✅ mT5 tokenizer loaded.")
    return tokenizer


def load_mt5_model(device, base_model_name=MT5_MODEL_NAME, adapter_path=MT5_ADAPTER_PATH):
    from peft import PeftModel
    print("\n🧠 Loading mT5 summarization model...")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    print(f"✅ mT5 + adapter loaded from {adapter_path}.")
    return model


