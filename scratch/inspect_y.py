import torch
import numpy as np
from pathlib import Path

cache_dir = Path("data/cache")
train_file = cache_dir / "final_token_aligned_soft_layer16_train.pt"

if train_file.exists():
    data = torch.load(train_file, map_location="cpu")
    y = data["y"].numpy()
    print("y shape:", y.shape)
    for d in range(7):
        unique_vals = np.unique(y[:, d])
        print(f"Slot {d}: unique values: {unique_vals}")
else:
    print("Cache file not found.")
