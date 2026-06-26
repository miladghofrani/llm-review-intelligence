import numpy as np
import pandas as pd
import evaluate
import torch
from transformers import GenerationConfig

from config import ADAPTER_PATH, MODEL_NAME
from device_utils import get_device
from model_loader import load_llm_model, load_tokenizer
from peft_trainer import load_saved_peft_model


class RougeEvaluator:
    """
    Compares the base model against the PEFT fine-tuned model on car rental
    review summarization using ROUGE metrics.
    """

    def __init__(self, adapter_path: str = ADAPTER_PATH, num_samples: int = 50):
        self.adapter_path = adapter_path
        self.num_samples  = num_samples
        self.device       = get_device()
        self.rouge        = evaluate.load("rouge")

        print("Loading tokenizer and models...")
        self.tokenizer  = load_tokenizer(MODEL_NAME)
        # Load base model twice so PEFT wrapping doesn't affect the reference copy
        self.base_model = load_llm_model(self.device, MODEL_NAME)
        self.base_model.eval()
        peft_base       = load_llm_model(self.device, MODEL_NAME)
        self.peft_model = load_saved_peft_model(self.device, peft_base, adapter_path)
        self.peft_model.eval()
        print("✅ Models ready.\n")

    def _summarize(self, model, review: str) -> str:
        prompt = f"Summarize the following car rental review.\n\n{review}\n\nSummary:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            tokens = model.generate(
                input_ids=inputs["input_ids"],
                generation_config=GenerationConfig(
                    max_new_tokens=80,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                ),
            )
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def evaluate(self, dataset) -> pd.DataFrame:
        """
        Runs evaluation on a sample of the test split.

        Args:
            dataset: DatasetDict with a 'test' split containing 'input' and 'output' columns.

        Returns:
            DataFrame with one row per sample showing reference, base, and peft summaries.
        """
        # Keep only summarization examples (not classification)
        summarization_rows = [
            row for row in dataset["test"]
            if row["input"].startswith("Summarize")
        ]
        n = min(self.num_samples, len(summarization_rows))
        print(f"Running inference on {n} summarization samples...\n")

        references, base_summaries, peft_summaries, reviews = [], [], [], []

        for i, row in enumerate(summarization_rows[:n]):
            review = row["input"].split("\n\n")[1]
            reference = row["output"]

            base_out = self._summarize(self.base_model, review)
            peft_out = self._summarize(self.peft_model, review)

            reviews.append(review)
            references.append(reference)
            base_summaries.append(base_out)
            peft_summaries.append(peft_out)

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n} done...")

        df = pd.DataFrame({
            "review":     reviews,
            "reference":  references,
            "base_model": base_summaries,
            "peft_model": peft_summaries,
        })

        self._print_scores(references, base_summaries, peft_summaries)
        return df

    def _print_scores(self, references, base_summaries, peft_summaries):
        base_results = self.rouge.compute(
            predictions=base_summaries,
            references=references,
            use_aggregator=True,
            use_stemmer=True,
        )
        peft_results = self.rouge.compute(
            predictions=peft_summaries,
            references=references,
            use_aggregator=True,
            use_stemmer=True,
        )

        print("\n" + "=" * 60)
        print(f"{'METRIC':<12} {'BASE MODEL':>14} {'PEFT MODEL':>14} {'IMPROVEMENT':>14}")
        print("=" * 60)

        for key in base_results:
            base_val = base_results[key]
            peft_val = peft_results[key]
            delta    = (peft_val - base_val) * 100
            sign     = "+" if delta >= 0 else ""
            print(f"{key:<12} {base_val*100:>13.2f}% {peft_val*100:>13.2f}% {sign}{delta:>12.2f}%")

        print("=" * 60)


if __name__ == "__main__":
    from data_processing.loader import load_review_dataset, build_multitask_dataset
    from data_processing.preprocessor import tokenize_dataset

    raw     = load_review_dataset()
    dataset = build_multitask_dataset(raw)

    evaluator = RougeEvaluator(num_samples=50)
    results   = evaluator.evaluate(dataset)

    print("\nSample predictions:")
    print(results[["reference", "base_model", "peft_model"]].head(5).to_string(index=False))
