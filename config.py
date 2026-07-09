import os

# Classification model (categories + sentiment)
MODEL_NAME   = os.getenv("MODEL_NAME",   "google/flan-t5-large")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "miladghofrani/car-rental-peft-adapter-large")

# Summarization model
MT5_MODEL_NAME   = os.getenv("MT5_MODEL_NAME",   "csebuetnlp/mT5_multilingual_XLSum")
MT5_ADAPTER_PATH = os.getenv("MT5_ADAPTER_PATH", "miladghofrani/car-rental-mt5-summarizer")

# Training — set MAX_STEPS=0 (or unset) for full training, any positive int for dry run
MAX_STEPS = int(os.getenv("MAX_STEPS", "0")) or None
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))

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
