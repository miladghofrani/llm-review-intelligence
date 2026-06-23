import os

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "miladghofrani/car-rental-peft-adapter")

# McAuley-Lab/Amazon-Reviews-2023 is the script-free replacement for amazon_us_reviews
# raw_review_Automotive has 'text' (body) + 'title' (headline), loaded via Parquet
DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_SUBSET = os.getenv("DATASET_SUBSET", "raw_review_Automotive")

# Training — set MAX_STEPS=0 (or unset) for full training, any positive int for dry run
MAX_STEPS = int(os.getenv("MAX_STEPS", "1")) or None
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))

# How many reviews to skip between samples (100 = keep 1% for local testing)
SUBSAMPLE_RATIO = int(os.getenv("SUBSAMPLE_RATIO", "100"))

# LoRA adapter settings
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Car rental review categories for multi-label classification
CATEGORIES = [
    "Vehicle Condition",
    "Customer Service",
    "Pricing & Billing",
    "Insurance & Documents",
    "Pickup & Return",
    "App & Booking",
]
