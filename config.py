import os

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "miladghofrani/car-rental-peft-adapter")

# fancyzhx/amazon_polarity is a Parquet-native re-upload with no loading script
# Columns: title (headline), content (body), label (0=neg / 1=pos)
DATASET_NAME = "fancyzhx/amazon_polarity"
DATASET_SUBSET = os.getenv("DATASET_SUBSET", "")

# Training — set MAX_STEPS=0 (or unset) for full training, any positive int for dry run
MAX_STEPS = int(os.getenv("MAX_STEPS", "0")) or None
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))

# How many reviews to skip between samples (200 = keep 0.5% — good balance of speed vs quality)
SUBSAMPLE_RATIO = int(os.getenv("SUBSAMPLE_RATIO", "200"))

# LoRA adapter settings
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Car rental review categories for multi-label classification
CATEGORIES = [
    "Cleanliness",
    "Vehicle Condition",
    "Pickup Experience",
    "Return Experience",
    "Hidden Fees & Billing",
    "Insurance & Upselling",
    "Staff & Communication",
    "Booking & App",
]
