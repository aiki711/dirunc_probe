import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b-it"

def debug_ppl(context, target_text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0")

    full_text = f"{context}\n{target_text}" if context and context != "None" else target_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
        logits = outputs.logits
        
    # Calculate loss per token
    # Shift so that tokens predict the next one
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print(f"\n--- Debug PPL ---")
    print(f"Full Text: {full_text}")
    
    # Find start of target_text
    target_tokens = tokenizer.tokenize(target_text)
    # This is a bit naive but works for debugging
    target_len = len(target_tokens)
    
    print(f"{'Token':<15} | {'Loss':<10} | {'PPL_contribution'}")
    print("-" * 40)
    
    # We ignore the first token (BOS) loss calculation as shift moves it
    for i in range(len(token_losses)):
        token = tokens[i+1] # The token being predicted
        loss = token_losses[i].item()
        ppl = torch.exp(token_losses[i]).item()
        print(f"{token:<15} | {loss:<10.4f} | {ppl:.2f}")

# Sample from SGD that had issues
ctx = "[Domain: Events_2 / Intent: BuyEventTickets]\nAssistant: Aloft Philadelphia Airport is 3 stars.\nUser: Good. Now book those tickets.\nAssistant: Number?"
filled = " Just 3."
mech = " Just."

print("DEBUGGING FILLED:")
debug_ppl(ctx, filled)
print("\nDEBUGGING MECHANICAL:")
debug_ppl(ctx, mech)
