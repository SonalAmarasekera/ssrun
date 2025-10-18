import torch

# Load a random .pt file produced by your cache_latents_working.py
x = torch.load("your_latent.pt")  # or from the Dataset

print("Mean:", x.mean().item())
print("Std:", x.std().item())
print("Min:", x.min().item())
print("Max:", x.max().item())
