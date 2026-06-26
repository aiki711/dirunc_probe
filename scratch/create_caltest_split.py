#!/usr/bin/env python3
"""
Create a stratified 50/50 cal/test split of the dev cache.

Cal split  → used for threshold calibration (and layer selection verification)
Test split → held-out for final evaluation (never touched before final report)

Saves indices to: data/cache/dev_cal_indices.npy
                  data/cache/dev_test_indices.npy
"""
import numpy as np
import torch
import json
from pathlib import Path
from collections import defaultdict

def main():
    SEED = 42
    rng = np.random.default_rng(SEED)

    CACHE_DIR = Path("data/cache")
    cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer26_dev.pt",
                       map_location="cpu")
    meta = cache["metadata"]   # list of dicts with 'case_role'
    N = len(meta)
    print(f"Total dev pairs: {N}")

    # ── Stratified split by case_role ─────────────────────────────────────
    role_to_indices = defaultdict(list)
    for i, m in enumerate(meta):
        role = m.get("case_role", "Unknown")
        role_to_indices[role].append(i)

    print("\n=== Case role distribution ===")
    for role, idxs in sorted(role_to_indices.items()):
        print(f"  {role:12s}: {len(idxs):4d} pairs")

    cal_indices  = []
    test_indices = []

    for role, idxs in role_to_indices.items():
        arr = np.array(idxs)
        rng.shuffle(arr)
        mid = len(arr) // 2
        cal_indices.extend(arr[:mid].tolist())
        test_indices.extend(arr[mid:].tolist())

    cal_indices  = np.array(sorted(cal_indices),  dtype=np.int64)
    test_indices = np.array(sorted(test_indices), dtype=np.int64)

    print(f"\nCal split:  {len(cal_indices):4d} pairs")
    print(f"Test split: {len(test_indices):4d} pairs")
    assert len(set(cal_indices) & set(test_indices)) == 0, "Overlap detected!"
    assert len(cal_indices) + len(test_indices) == N, "Total mismatch!"

    # ── Verify role balance ───────────────────────────────────────────────
    cal_roles  = [meta[i]["case_role"] for i in cal_indices]
    test_roles = [meta[i]["case_role"] for i in test_indices]
    from collections import Counter
    print("\n=== Role balance (cal / test) ===")
    all_roles = sorted(set(cal_roles + test_roles))
    cc, tc = Counter(cal_roles), Counter(test_roles)
    for r in all_roles:
        print(f"  {r:12s}: cal={cc[r]:3d}  test={tc[r]:3d}")

    # ── Save ─────────────────────────────────────────────────────────────
    np.save(CACHE_DIR / "dev_cal_indices.npy",  cal_indices)
    np.save(CACHE_DIR / "dev_test_indices.npy", test_indices)
    print(f"\nSaved to {CACHE_DIR / 'dev_cal_indices.npy'}")
    print(f"Saved to {CACHE_DIR / 'dev_test_indices.npy'}")

if __name__ == "__main__":
    main()
