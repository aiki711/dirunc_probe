"""
Experiment 9-Extension: Word-Level Checklist Evolution
=======================================================
Probes the model's internal checklist state after EACH WORD is added to the
context incrementally.

Methodology (identical to Exp9 sentence-level, but at word granularity):
  For each word w in the scenario text:
    - Append w to the running context
    - Append the full query block: [WHO?] [WHAT?] [WHEN?] [WHERE?] [WHY?] [HOW?] [WHICH?]
    - Run a forward pass and probe at hs[-1] (sequence end)
    - Record P(missing) for each label

This is methodologically valid because the probe was trained on hs[-1].
"""

import torch
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.getcwd())
from scripts.common import DIRS

QUERY_SUFFIX = " [WHO?] [WHAT?] [WHEN?] [WHERE?] [WHY?] [HOW?] [WHICH?]"


def run_probe_prediction(hs_last, W, b):
    logits = (hs_last.to(torch.float32) @ W.T.to(torch.float32)) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    return {d: float(probs[i].cpu().numpy()) for i, d in enumerate(DIRS)}


def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_lodo/lodo_query_layer8_multiwoz_train.pt"

    # Same scenario as Exp9 sentence-level (train booking)
    scenario_name = "train_booking_word_level"
    full_text = (
        "I need to find a train. "
        "I am going to cambridge from london kings cross. "
        "I need to leave on friday after 15:15. "
        "There will be 2 of us traveling."
    )

    output_dir = Path("runs/balanced/experiment9_word")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "word_trajectory.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    print("Loading probe weights...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)

    # Split on spaces, keeping punctuation attached to the preceding word
    words = full_text.split()

    results = {
        "scenario": scenario_name,
        "full_text": full_text,
        "word_evolution": [],
    }

    current_context = ""

    with torch.no_grad():
        for i, word in enumerate(words):
            # Build running context
            if current_context:
                current_context += " "
            current_context += word

            probe_text = current_context + QUERY_SUFFIX

            enc = tokenizer(probe_text, return_tensors="pt").to(device)
            out = lm(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx + 1][0]  # (seq_len, hidden)

            probs = run_probe_prediction(hs[-1], W, b)

            results["word_evolution"].append({
                "word_idx": i,
                "word": word,
                "context_so_far": current_context,
                "probs": probs,
            })

            if (i + 1) % 5 == 0 or i == len(words) - 1:
                print(f"  [{i+1}/{len(words)}] '{word}' | "
                      f"WHO={probs['who']:.3f} WHEN={probs['when']:.3f} WHERE={probs['where']:.3f}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved word trajectory data to {out_file}")


if __name__ == "__main__":
    main()
