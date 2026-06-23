import os

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "miladghofrani/car-rental-peft-adapter")

# fancyzhx/amazon_polarity is a Parquet-native re-upload with no loading script
# Columns: title (headline), content (body), label (0=neg / 1=pos)
DATASET_NAME = "fancyzhx/amazon_polarity"
DATASET_SUBSET = os.getenv("DATASET_SUBSET", "")

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
