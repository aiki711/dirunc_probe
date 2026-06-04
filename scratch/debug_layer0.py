import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

import torch
import importlib.util
from transformers import AutoTokenizer
from scripts.common import NATURAL_QUERY_STR

spec = importlib.util.spec_from_file_location("nq_probe", "scripts/32_train_contrastive_nq_probe.py")
nq_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq_probe)
ProbeModelBase = nq_probe.ProbeModelBase
NaturalQueryProbe = nq_probe.NaturalQueryProbe

def main():
    model_name = "google/gemma-2-2b-it"
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    base = ProbeModelBase(model_name).to(device)
    model = NaturalQueryProbe(base, tokenizer).to(device)

    # Simple text pair (filled and missing)
    text_f = "Yes I'm looking for a train to Cambridge that same day." + NATURAL_QUERY_STR
    text_m = "Yes I'm looking for a train that same day." + NATURAL_QUERY_STR

    enc_f = tokenizer([text_f], return_tensors="pt").to(device)
    enc_m = tokenizer([text_m], return_tensors="pt").to(device)

    with torch.no_grad():
        # Get hidden states at Layer 0 (embeddings)
        out_f = base.lm(input_ids=enc_f["input_ids"], attention_mask=enc_f["attention_mask"])
        out_m = base.lm(input_ids=enc_m["input_ids"], attention_mask=enc_m["attention_mask"])
        
        hs_f = base.get_layer_hidden(out_f.hidden_states, 0)
        hs_m = base.get_layer_hidden(out_m.hidden_states, 0)

        # Forward pass on model to extract dir_vecs
        # Let's inspect the dir_vecs manually
        def get_dir_vecs(input_ids, hs):
            ids_list = input_ids[0].tolist()
            h_vec = hs[0]
            dir_vecs = []
            for d in ["who", "what"]:
                seq = model.query_token_seqs[d]
                pos = None
                for j in range(len(ids_list) - len(seq), -1, -1):
                    if ids_list[j:j+len(seq)] == seq:
                        pos = j + len(seq) - 1
                        break
                print(f"Direction {d} -> seq: {seq}, found at pos: {pos}, token: {tokenizer.decode([ids_list[pos]]) if pos is not None else None}")
                dir_vecs.append(h_vec[pos])
            return dir_vecs

        print("=== Filled ===")
        vecs_f = get_dir_vecs(enc_f["input_ids"], hs_f)
        print("Are 'who' and 'what' vectors identical for Filled?", torch.allclose(vecs_f[0], vecs_f[1]))

        print("\n=== Missing ===")
        vecs_m = get_dir_vecs(enc_m["input_ids"], hs_m)
        print("Are 'who' and 'what' vectors identical for Missing?", torch.allclose(vecs_m[0], vecs_m[1]))

        print("\nAre the 'who' vectors identical between Filled and Missing?", torch.allclose(vecs_f[0], vecs_m[0]))

if __name__ == "__main__":
    main()
