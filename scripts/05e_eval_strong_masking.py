import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import sys
import importlib.util
from sklearn.metrics import f1_score
import re
import spacy

def load_03_train_probe():
    path = Path(__file__).parent / "03_train_probe.py"
    spec = importlib.util.spec_from_file_location("train_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_probe"] = module
    spec.loader.exec_module(module)
    return module

train_probe = load_03_train_probe()
ProbeModelBase = train_probe.ProbeModelBase
extract_activations = train_probe.extract_activations
JsonlDirUncDataset = train_probe.JsonlDirUncDataset
FilterSpec = train_probe.FilterSpec
collate_batch = train_probe.collate_batch

from common import DIRS, SPECIAL_TOKENS, QUERY_LABEL_STR

# Specific keywords that strongly indicate the service or intent to MASK
MASK_KEYWORDS = [
    "restaurant", "restaurants", "food", "eat", "dining", "dine",
    "hotel", "hotels", "motel", "motels", "stay", "lodging", "guesthouse",
    "flight", "flights", "fly", "airplane", "airport",
    "train", "trains",
    "bus", "buses",
    "taxi", "cab", "uber", "ride",
    "book", "reserve", "booking", "reservation",
    "find", "search", "looking",
    "doctor", "hospital", "medical",
    "event", "events", "concert", "game", "movie", "movies", "cinema",
    "music", "song", "songs",
    "bank", "transfer", "balance", "account"
]

# NER categories to mask
MASK_ENTITIY_LABELS = {"GPE", "LOC", "FAC", "ORG", "DATE", "TIME"}

def get_mask_function():
    nlp = spacy.load("en_core_web_sm")
    keyword_pattern = re.compile(r'\b(' + '|'.join(MASK_KEYWORDS) + r')\b', re.IGNORECASE)
    
    def mask_text(text: str) -> str:
        # 1. First, find query tokens to preserve them
        query_start_idx = text.find(" [")
        if query_start_idx != -1 and "[WHO?]" in text:
            content_text = text[:query_start_idx]
            query_text = text[query_start_idx:]
        else:
            content_text = text
            query_text = ""

        # 2. Keyword masking
        masked_text = keyword_pattern.sub("[MASK]", content_text)
        
        # 3. NER and POS masking
        doc = nlp(masked_text)
        # We need to build the string token by token or use offsets. 
        # Using offsets is safer to avoid re-masking [MASK] tokens or artifacts.
        
        # Sort entities by start char to handle them in order
        spans = []
        for ent in doc.ents:
            if ent.label_ in MASK_ENTITIY_LABELS:
                spans.append((ent.start_char, ent.end_char))
        
        # Also mask Proper Nouns that aren't already masked
        for token in doc:
            if token.pos_ == "PROPN" and not any(s <= token.idx < e for s, e in spans):
                # Only if it's not already a "[MASK]" (spacy might treat [MASK] weirdly)
                if token.text != "[" and token.text != "MASK" and token.text != "]":
                     spans.append((token.idx, token.idx + len(token.text)))
        
        # Sort and merge overlapping spans
        spans.sort()
        merged_spans = []
        if spans:
            curr_s, curr_e = spans[0]
            for next_s, next_e in spans[1:]:
                if next_s < curr_e:
                    curr_e = max(curr_e, next_e)
                else:
                    merged_spans.append((curr_s, curr_e))
                    curr_s, curr_e = next_s, next_e
            merged_spans.append((curr_s, curr_e))
        
        # Apply spans in reverse to not mess up offsets
        result_chars = list(masked_text)
        for s, e in reversed(merged_spans):
            result_chars[s:e] = ["[MASK]"]
            
        return "".join(result_chars) + query_text
        
    return mask_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Evaluation dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained probe .pt file")
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary.json with thresholds")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mode", type=str, default="query")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    tokenizer.add_tokens(["[MASK]"])

    shared_model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    shared_model.resize_token_embeddings(len(tokenizer))
    shared_model.to(device)
    shared_model.eval()

    mask_fn = get_mask_function()

    with open(args.data_jsonl, "r") as f:
        rows = [json.loads(line) for line in f]

    print("Applying strong masking...")
    for r in tqdm(rows):
        r["text"] = mask_fn(r["text"])

    # Sample output for verification
    print("\n[Sample Masked Text]")
    for i in range(min(5, len(rows))):
        print(f"Original ID: {rows[i].get('id', 'N/A')}")
        print(f"Masked Text: {rows[i]['text']}")

    fs = FilterSpec(None, None, None)
    ds = JsonlDirUncDataset(rows, fs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, 
                    collate_fn=lambda b: collate_batch(tokenizer, b, args.max_length))

    base = ProbeModelBase(
        args.model_name,
        vocab_size=None,
        train_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS],
        pretrained_model=shared_model,
    ).to(device)

    print(f"Extracting activations for layer {args.layer_idx} (STRONG MASKING)...")
    X_dict, Y, _ = extract_activations(dl, base, [args.layer_idx], args.mode, device, tokenizer)
    X = X_dict[args.layer_idx].to(device)
    Y = Y.to(device)

    hidden_size = shared_model.config.hidden_size
    num_labels = len(DIRS)
    probe = train_probe.QueryHead(hidden_size, num_labels, torch.float32).to(device)
    probe.load_state_dict(torch.load(args.model_path, map_location=device))
    probe.eval()

    with open(args.summary_json, "r") as f:
        summary = json.load(f)
    best_metrics = summary["best_overall"]["best"]
    t_dict = best_metrics["per_class_tuned"]["threshold_dict"]
    thresholds = torch.tensor([t_dict[d] for d in DIRS], device=device)
    print(f"Loaded optimized thresholds: {t_dict}")

    print("Evaluating...")
    with torch.no_grad():
        logits = probe(X)
        probs = torch.sigmoid(logits)
        preds = (probs >= thresholds).float()

    Y_np = Y.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    macro_f1 = f1_score(Y_np, preds_np, average="macro", zero_division=0)
    print(f"\n[Strong Masking Overall] Macro F1: {macro_f1:.4f}")
    
    per_class_f1 = f1_score(Y_np, preds_np, average=None, zero_division=0)
    for j, cls in enumerate(DIRS):
        print(f"  - {cls}: {per_class_f1[j]:.4f}")

if __name__ == "__main__":
    main()
