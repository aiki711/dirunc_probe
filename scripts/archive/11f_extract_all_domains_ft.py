import json
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import glob
import sys
import os

sys.path.append(os.getcwd())
from scripts.common import DIRS, strip_query_tokens

def get_final_token_position(attention_mask):
    valid_indices = torch.nonzero(attention_mask[0]).squeeze(-1)
    if valid_indices.numel() == 0: return -1
    return valid_indices[-1].item()

def run_probe_final_token(hs, pos, W, b):
    if pos < 0: return {d: 0.0 for d in DIRS}
    vec = hs[0, pos].to(torch.float32)
    H = vec.unsqueeze(0).expand(len(DIRS), -1)
    logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    return {d: float(probs[i].cpu().numpy()) for i, d in enumerate(DIRS)}

def extract_domain_from_id(sample_id: str):
    parts = sample_id.split("::")
    source = parts[0]
    if source == "sgd" and len(parts) > 3:
        return f"sgd_{parts[3]}"
    elif source == "multiwoz" and len(parts) > 2:
        return f"multiwoz_{parts[2].replace('domain_', '')}"
    return None

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    data_jsonl = "data/processed/mixed/dirunc/all.jsonl"
    exp_dir = "runs/balanced/experiment6_final_token"
    neurons_path = "runs/balanced/experiment7_neurons_ft/neurons_report.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(neurons_path, "r") as f:
        neuron_report = json.load(f)
        top_neurons = {d: [n["index"] for n in neuron_report[d][:3]] for d in DIRS}

    print("Loading test dataset in memory...")
    # Group by domain -> dialogue_id -> turns
    domain_dialogues = defaultdict(lambda: defaultdict(list))
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            did = data.get("id", "")
            domain = extract_domain_from_id(did)
            if domain:
                diag_id = "::".join(did.split("::")[:4]) if "sgd" in domain else "::".join(did.split("::")[:3])
                tidx = data.get("_meta", {}).get("turn_idx", 0)
                domain_dialogues[domain][diag_id].append((tidx, data))

    # sort turns
    for dom in domain_dialogues:
        for did in domain_dialogues[dom]:
            domain_dialogues[dom][did].sort(key=lambda x: x[0])

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    enc_null = tokenizer([""], return_tensors="pt").to(device)
    with torch.no_grad():
        out_null = lm(**enc_null, output_hidden_states=True)
        hs_null = out_null.hidden_states[layer_idx + 1]
        pos_null = get_final_token_position(enc_null["attention_mask"])

    found_examples = {d: [] for d in DIRS}
    examples_needed = 2

    # Loop through each domain model
    for pt_file in glob.glob(f"{exp_dir}/*.pt"):
        # e.g. lodo_query_layer8_sgd_Events_1.pt -> domain is sgd_Events_1
        domain = os.path.basename(pt_file).replace("lodo_query_layer8_", "").replace(".pt", "")
        eval_json = f"{exp_dir}/{domain}_eval.json"
        
        if domain not in domain_dialogues or not os.path.exists(eval_json):
            continue
            
        with open(eval_json, "r") as f:
            thresholds = json.load(f).get("thresholds_used", {d: 0.5 for d in DIRS})
            
        print(f"\nProcessing domain: {domain}")
        weights = torch.load(pt_file, map_location="cpu")
        W = weights["W"].to(device).to(torch.bfloat16)
        b = weights["b"].to(device).to(torch.bfloat16)

        with torch.no_grad():
            p_null = run_probe_final_token(hs_null, pos_null, W, b)

        dialogues = domain_dialogues[domain]
        # Prioritize matching labels that we haven't found yet
        for did, turns in dialogues.items():
            if all(len(found_examples[d]) >= examples_needed for d in DIRS):
                break
                
            for i in range(len(turns)):
                tidx_a, data_a = turns[i]
                labels_a = data_a.get("labels", {})
                text_a = strip_query_tokens(data_a.get("text", "")).strip()
                
                # evaluate only labels we still need
                for target_label in DIRS:
                    if len(found_examples[target_label]) >= examples_needed: continue
                        
                    if int(labels_a.get(target_label, 0)) == 1:
                        for j in range(i+1, min(i+4, len(turns))):
                            tidx_b, data_b = turns[j]
                            labels_b = data_b.get("labels", {})
                            if int(labels_b.get(target_label, 0)) == 0:
                                text_b = strip_query_tokens(data_b.get("text", "")).strip()
                                
                                with torch.no_grad():
                                    enc_a = tokenizer([text_a], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
                                    out_a = lm(**enc_a, output_hidden_states=True)
                                    hs_a = out_a.hidden_states[layer_idx + 1]
                                    pos_a = get_final_token_position(enc_a["attention_mask"])
                                    p_real_a = run_probe_final_token(hs_a, pos_a, W, b)
                                    delta_p_a = p_real_a[target_label] - p_null[target_label]
                                    
                                    enc_b = tokenizer([text_b], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
                                    out_b = lm(**enc_b, output_hidden_states=True)
                                    hs_b = out_b.hidden_states[layer_idx + 1]
                                    pos_b = get_final_token_position(enc_b["attention_mask"])
                                    p_real_b = run_probe_final_token(hs_b, pos_b, W, b)
                                    delta_p_b = p_real_b[target_label] - p_null[target_label]
                                
                                th = thresholds.get(target_label, 0.5)
                                if delta_p_a >= th and delta_p_b < th:
                                    # Found!
                                    aa = [float(hs_a[0, pos_a][idx].cpu()) for idx in top_neurons[target_label]]
                                    ab = [float(hs_b[0, pos_b][idx].cpu()) for idx in top_neurons[target_label]]
                                    found_examples[target_label].append({
                                        "domain": domain,
                                        "A_text": text_a,
                                        "B_text": text_b,
                                        "A_delta_p": delta_p_a,
                                        "B_delta_p": delta_p_b,
                                        "threshold": th,
                                        "neurons": top_neurons[target_label],
                                        "A_activations": aa,
                                        "B_activations": ab
                                    })
                                break

    out_file = "runs/balanced/experiment7_neurons_ft/all_test_data_shifts.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=== Correctly Predicted Test Data Pairs (ALL LODO Domains) ===\n\n")
        for label, examples in found_examples.items():
            if not examples:
                f.write(f"--- [{label.upper()}] no valid test pairs found ---\n\n")
                continue
            for i, ex in enumerate(examples):
                f.write(f"--- [{label.upper()}] Correct Pair {i+1} (Domain: {ex['domain']}) ---\n")
                f.write(f"Threshold: {ex['threshold']:.3f}\n")
                f.write(f"Missing (A): {ex['A_text']}\n")
                f.write(f"Filled  (B): {ex['B_text']}\n")
                f.write(f"  DeltaP(A): {ex['A_delta_p']:+.3f} (Pred: Missing)\n")
                f.write(f"  DeltaP(B): {ex['B_delta_p']:+.3f} (Pred: Filled)\n")
                f.write("  Top Neurons Shift:\n")
                for j, n_idx in enumerate(ex["neurons"]):
                    aa = ex["A_activations"][j]
                    ab = ex["B_activations"][j]
                    shift = ab - aa
                    status = "SUPPRESSED" if shift < -0.05 else ("INCREASED" if shift > 0.05 else "STABLE")
                    f.write(f"    n{n_idx:<4} : A={aa:+.3f} -> B={ab:+.3f} | Δ={shift:+.3f} [{status}]\n")
                f.write("\n")

    print(f"Done. Outputs written to {out_file}")

if __name__ == "__main__":
    main()
