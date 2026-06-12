import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

class ReviewSummarizer:
    def __init__(self, base_model_name="google/flan-t5-base", adapter_path="./peft-dialogue-summary-checkpoint-local"):
        """
        Constructor: Loads the base model, tokenizer, and attaches the LoRA adapter.
        This runs only once when the class is instantiated.
        """
        print("🚀 Initializing the Production Model...")
        self.device = "cpu"
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Load Base Model
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_name, 
            torch_dtype=torch.float32
        ).to(self.device)
        
        # Try Loading Production Model (PEFT/LoRA)
        try:
            self.production_model = PeftModel.from_pretrained(
                self.base_model, 
                self.adapter_path, 
                torch_dtype=torch.float32,
                is_trainable=False
            ).to(self.device)
            print("✅ Custom LoRA Adapter attached successfully!\n")
        except Exception as e:
            print(f"❌ Error loading adapter. Ensure '{self.adapter_path}' exists.")
            self.production_model = None

    def summarize_single_review(self, custom_review: str):
        """
        Method 1: Takes a single string (review) and returns the AI generated summary.
        """
        if not self.production_model:
            return "Error: Production model not loaded."

        prompt = f"Summarize the following review.\n\n{custom_review}\n\nSummary: "
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        output_tokens = self.production_model.generate(
            input_ids=inputs["input_ids"], 
            max_new_tokens=50,
            temperature=0.5,
            do_sample=True
        )[0]
        
        summary = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return summary

    def compare_models_visually(self, dataset, sample_size=10):
        """
        Method 2: Compares the base model against the trained PEFT model using a dataset.
        """
        if not self.production_model:
            print("Error: Production model not loaded.")
            return None

        print(f"\n📊 Visual Comparison: Base Model vs. PEFT Model (Sample Size: {sample_size})")
        
        reviews = dataset['test'][0:sample_size]['review_body']
        human_baseline_summaries = dataset['test'][0:sample_size]['review_title']

        original_model_summaries = []
        peft_model_summaries = []

        print(f"⏳ Generating summaries for {sample_size} reviews. Please wait...")
        
        for review in reviews:
            prompt = f"Summarize the following review.\n\n{str(review)}\n\nSummary: "
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids

            # Generate with Base Model
            orig_outputs = self.base_model.generate(input_ids=input_ids, max_new_tokens=50)
            original_model_summaries.append(self.tokenizer.decode(orig_outputs[0], skip_special_tokens=True))

            # Generate with PEFT Model
            peft_outputs = self.production_model.generate(input_ids=input_ids, max_new_tokens=50)
            peft_model_summaries.append(self.tokenizer.decode(peft_outputs[0], skip_special_tokens=True))
            
        zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))
        df = pd.DataFrame(zipped_summaries, columns=['Human_Baseline', 'Original_Model', 'PEFT_Model'])
        
        print("\n✅ Comparison Table Generated!\n")
        print("-" * 80)
        print(df.head(5))
        print("-" * 80)
        
        return df

# ==========================================
# HOW TO USE THE CLASS
# ==========================================
if __name__ == "__main__":
    # 1. Instantiate the class (Loads models into memory once)
    summarizer = ReviewSummarizer()

    # 2. Use Method 1: Test a single custom review
    sample_review = """
    I rented a car for my weekend trip to the mountains. The engine ran fine, 
    but the interior was extremely dirty. There were sticky coffee stains on the passenger 
    seat and the whole cabin smelled like cheap cigarette smoke. Also, their mobile app 
    crashed twice when I tried to extend my rental period. Very frustrating experience!
    """
    
    print("-" * 80)
    print(f"📝 CUSTOM REVIEW:\n{sample_review.strip()}")
    print("-" * 80)
    ai_summary = summarizer.summarize_single_review(sample_review)
    print(f"✨ AI GENERATED SUMMARY:\n{ai_summary}")
    print("-" * 80)

    # 3. Use Method 2: Compare models visually
    # (Uncomment the lines below and pass your loaded dataset to run this)
    # from datasets import load_dataset
    # dataset = load_dataset("goosmanlei/amazon_reviews_multi", "en")
    # comparison_df = summarizer.compare_models_visually(dataset, sample_size=5)