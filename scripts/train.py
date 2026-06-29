from config import ADAPTER_PATH, CATEGORIES
from data_processing.loader import load_review_dataset, build_multitask_dataset
from data_processing.preprocessor import tokenize_dataset
from app.device_utils import get_device
from app.model_loader import load_llm_model, load_tokenizer, print_number_of_trainable_model_parameters
from app.peft_trainer import load_saved_peft_model, setup_peft_lora_model, train_and_save_peft_model


def train():
    """Fine-tunes flan-t5 with LoRA on summarization + classification of car rental reviews."""
    device = get_device()

    tokenizer = load_tokenizer()
    original_model = load_llm_model(device)
    print("\n" + print_number_of_trainable_model_parameters(original_model))

    raw_dataset = load_review_dataset()
    multitask_dataset = build_multitask_dataset(raw_dataset)
    tokenized_dataset = tokenize_dataset(tokenizer, multitask_dataset)

    peft_model = setup_peft_lora_model(original_model)
    print("\n" + print_number_of_trainable_model_parameters(peft_model))

    train_and_save_peft_model(peft_model, tokenizer, tokenized_dataset)

    print("\n✅ Training complete. Adapter weights saved to:", ADAPTER_PATH)


def infer(review: str, adapter_path: str = ADAPTER_PATH):
    """Runs summarization and category classification on a single car rental review."""
    device = get_device()
    tokenizer = load_tokenizer()
    base_model = load_llm_model(device)
    model = load_saved_peft_model(device, base_model, adapter_path)

    categories_str = ", ".join(CATEGORIES)

    print("\n📝 Input Review:")
    print("-" * 60)
    print(review)
    print("-" * 60)

    # --- Summarization ---
    summary_prompt = f"Summarize the following car rental review.\n\n{review}\n\nSummary:"
    summary_inputs = tokenizer(summary_prompt, return_tensors="pt").to(device)
    summary_tokens = model.generate(
        input_ids=summary_inputs["input_ids"],
        max_new_tokens=60,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )[0]
    summary = tokenizer.decode(summary_tokens, skip_special_tokens=True)

    # --- Classification ---
    category_prompt = (
        f"Classify this car rental review into one or more of these categories: "
        f"{categories_str}.\n\n"
        f"Review: {review}\n\nCategories:"
    )
    category_inputs = tokenizer(category_prompt, return_tensors="pt").to(device)
    category_tokens = model.generate(
        input_ids=category_inputs["input_ids"],
        max_new_tokens=40,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )[0]
    categories = tokenizer.decode(category_tokens, skip_special_tokens=True)

    print(f"\n✨ Summary   : {summary}")
    print(f"🏷️  Categories: {categories}")
    print("-" * 60)

    return {"summary": summary, "categories": categories}


if __name__ == "__main__":
    sample_review = (
        "I rented a car for my weekend trip. The engine ran fine, "
        "but the interior was extremely dirty with sticky coffee stains on the passenger "
        "seat. Also, their app crashed twice when I tried to extend my rental. "
        "On top of that, they charged me an extra fee that was never mentioned at booking."
    )
    infer(sample_review)
