import torch
from transformers import AutoModel, AutoTokenizer
import time

print("CUDA available:", torch.cuda.is_available())
print("BF16 supported:", torch.cuda.is_bf16_supported())

model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Selected dtype:", dtype)

print("Loading model...")
t0 = time.time()
model = AutoModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype=dtype)
print(f"Loaded in {time.time() - t0:.2f}s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Moving to device:", device)
t0 = time.time()
model.to(device)
print(f"Moved to device in {time.time() - t0:.2f}s")

# Check parameter devices
devices = set()
for name, param in model.named_parameters():
    devices.add(param.device)
print("Parameter devices:", devices)

# Run a test forward pass
print("Running test forward pass...")
batch_size = 16
seq_len = 100
input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
attention_mask = torch.ones((batch_size, seq_len)).to(device)

# Warmup
with torch.no_grad():
    _ = model(input_ids=input_ids, attention_mask=attention_mask)

# Measure
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
print(f"10 forward passes took {time.time() - t0:.4f}s (Average {(time.time() - t0)/10:.4f}s per pass)")
