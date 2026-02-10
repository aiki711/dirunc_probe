from datasets import load_dataset

# Try without trust_remote_code
datasets_to_try = [
    "multi_woz_v22",
    "Hotel-ID/MultiWOZ-2.4", # Check exact case if possible
    "convlab/multiwoz24",
    "google/multiwoz"
]

for ds_name in datasets_to_try:
    print(f"Trying to load {ds_name}...")
    try:
        # removed trust_remote_code=True
        ds = load_dataset(ds_name)
        print(f"Successfully loaded {ds_name}")
        print("Keys:", ds.keys())
        if 'train' in ds:
            print("First example:", ds['train'][0])
        break
    except Exception as e:
        print(f"Failed to load {ds_name}: {e}")
