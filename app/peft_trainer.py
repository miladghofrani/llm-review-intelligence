import time
import torch

from config import (
    ADAPTER_PATH,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MAX_STEPS,
    NUM_EPOCHS,
)
from .model_loader import print_number_of_trainable_model_parameters
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import TrainingArguments, Trainer


def setup_peft_lora_model(original_model):
    """Injects LoRA adapters into the base model, freezing the original weights."""
    print("\n🪄 Injecting LoRA Adapters (PEFT)...")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q", "v"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    peft_model = get_peft_model(original_model, lora_config)
    print("✅ LoRA adapters injected.")
    return peft_model


def train_and_save_peft_model(peft_model, tokenizer, tokenized_datasets):
    """Runs the PEFT training loop and saves only the small adapter weights."""
    run_dir = f"./peft-training-run-{int(time.time())}"

    training_args = TrainingArguments(
        output_dir=run_dir,
        auto_find_batch_size=True,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=1,
        max_steps=MAX_STEPS if MAX_STEPS is not None else -1,
    )

    mode = f"max_steps={MAX_STEPS}" if MAX_STEPS is not None else f"{NUM_EPOCHS} full epoch(s)"
    print(f"\n🏋️  Starting PEFT training ({mode})...")

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    trainer.train()

    trainer.model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"✅ Adapter weights saved to: {ADAPTER_PATH}")
    return ADAPTER_PATH


def load_saved_peft_model(device, base_model, adapter_path):
    """Loads base model + LoRA adapters for inference."""
    print(f"\n🚀 Loading PEFT model from {adapter_path}...")

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        dtype=torch.bfloat16,
        is_trainable=False,
    ).to(device)

    print("✅ Model ready for inference.")
    return peft_model
