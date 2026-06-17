import torch

def get_device():
    print("🚀 Initializing the LLM Pipeline...")
    print("--------------------------------------------------")
    print("✅ All required libraries imported successfully.")
    print(f"✅ PyTorch version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Processing device set to: {device.upper()}")
    print("--------------------------------------------------")
    return device
