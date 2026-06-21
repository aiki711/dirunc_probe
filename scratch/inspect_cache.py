import torch
from pathlib import Path

cache_dir = Path("data/cache")
train_file = cache_dir / "final_token_aligned_soft_layer16_train.pt"

if train_file.exists():
    print(f"Loading {train_file}...")
    data = torch.load(train_file, map_location="cpu")
    print("Keys in data dict:", data.keys())
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: tensor of shape {v.shape}, dtype {v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list of length {len(v)}")
            if len(v) > 0:
                print(f"    First item type: {type(v[0])}")
                if isinstance(v[0], dict):
                    print(f"    First item keys: {v[0].keys()}")
                    print(f"    First item sample: {v[0]}")
else:
    print(f"File {train_file} not found.")
