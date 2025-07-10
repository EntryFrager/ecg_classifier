import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training will take on {device}")
