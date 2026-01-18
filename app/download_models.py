#!/usr/bin/env python3
"""
Pre-download models during Docker build to avoid runtime issues
"""
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/graphcodebert-base"

print(f"Downloading {MODEL_NAME}...")
print(f"PyTorch version: {torch.__version__}")

# Download tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✓ Tokenizer downloaded")

# Download model with explicit settings to avoid meta tensor issues
print("Downloading model...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
)
print("✓ Model downloaded")

# Test loading
print("\nTesting model load...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = model.to(device)
model.eval()
print("✓ Model loaded successfully")

print("\nAll models downloaded and verified!")
