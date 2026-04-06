import json
import re
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def normalize_natural_text(text: str, target_dir: str) -> str:
    """
    Makes a mechanically masked sentence (with multi-spaces or missing fragments) more natural.
    """
    # 1. Remove double spaces
    text = re.sub(r"\s{2,}", " ", text)
    
    # 2. Fix broken prepositions: "in ." -> ".", "at ," -> ","
    text = re.sub(r"\b(in|at|on|to|from|for|with)\s*([.,!?;])", r"\2", text, flags=re.IGNORECASE)
    
    # 3. Handle missing subjects for Name-like slots (what/who)
    # If the sentence starts with a verb like "seems", "is located", add "This" or "It"
    if target_dir in ["what", "who"]:
        # Match cases like "Assistant:  seems like" or "User:  is good"
        # Since we just normalized double spaces, it might be "Assistant: seems like"
        text = re.sub(r"(Assistant:|User:)\s*([a-z])", r"\1 It \2", text)
        # Also handle "Assistant: . seems like" -> "Assistant: It seems like"
        text = re.sub(r"([.!?])\s*([a-z])", r"\1 It \2", text)

    # 4. Clean up trailing spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()

def truncate_at_target(text: str, target_dir: str) -> str:
    """
    Removes trailing noise like [WHO?] and any turns that appear AFTER the 
    information was provided/missing.
    Actually, to keep it simple and high quality, we truncate everything after 
    the turn that contains the slot difference.
    """
    # 1. Strip QUERY TOKENS
    from common import strip_query_tokens
    text = strip_query_tokens(text)
    
    # 2. Split by turns
    lines = text.split("\n")
    
    # We want to find the turn that differs between A and B, or just the last turn 
    # if it's already relevant.
    # But for B (Resolved), the slot info is usually in one of the turns.
    # The current dataset preparation anchors on a turn_idx.
    # If we stop at the LAST turn provided in the context (before the query tokens),
    # that is usually the context we want.
    
    # HOWEVER, the user specifically mentioned they don't want trailing "booking..." noise.
    # If the last turn is "User: 予約をお願いします", we should probably go back one turn.
    
    # Heuristic: If the last turn contains common 'noise' intents like booking, thank you, is there anything else?
    # remove it.
    noise_patterns = [
        r"reserve", r"booking", r"anything else", r"thank you", r"thanks", r"all set", r"sounds good",
        r"予約", r"お願いします", r"ありがとう", r"ほかはありますか"
    ]
    
    # We iteratively remove the last turn if it matches noise and we have at least 1 turn left.
    while len(lines) > 1:
        last_turn = lines[-1].lower()
        if any(re.search(p, last_turn) for p in noise_patterns):
            lines.pop()
        else:
            break
            
    return "\n".join(lines).strip()

def main():
    input_path = Path("data/processed/sgd/dirunc_balanced/train.jsonl")
    output_path = Path("data/processed/sgd/dirunc_balanced/paired_minimal_v2.jsonl")
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    # Load and group by pair_id
    data_by_id = defaultdict(dict)
    print("Loading dataset...")
    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f):
            row = json.loads(line)
            p_id = row.get("pair_id")
            cond = row.get("condition")
            if p_id and cond in ["resolved", "unresolved"]:
                data_by_id[p_id][cond] = row

    paired_data = []
    dirs = ["who", "when", "where", "how", "which", "what"]
    
    print("Generating natural minimal pairs...")
    for p_id, variants in tqdm(data_by_id.items()):
        if "resolved" not in variants or "unresolved" not in variants:
            continue
            
        b_row = variants["resolved"]
        a_row = variants["unresolved"]
        target_dir = b_row["target_dir"]
        
        # 1. Truncate noise from both
        text_b = truncate_at_target(b_row["text"], target_dir)
        text_a_raw = truncate_at_target(a_row["text"], target_dir)
        
        # 2. Naturalize A
        text_a = normalize_natural_text(text_a_raw, target_dir)
        # B also needs basic normalization if it was truncated
        text_b = normalize_natural_text(text_b, target_dir)
        
        # 3. Final verification: They must be different and the same history length
        if text_a == text_b:
            continue
            
        # Ensure B actually has the info (heuristic check: A should be shorter or different)
        # Since we use Final Token probe, we'll extract the embedding from the end of context.
        paired_data.append({
            "pair_id": p_id + "::natural",
            "text_A": text_a,
            "text_B": text_b,
            "label_idx": dirs.index(target_dir),
            "direction": target_dir,
            "turn_idx": b_row["turn_idx"],
            "dialogue_id": b_row["dialogue_id"]
        })

    # 4. Balancing
    print(f"Total pairs extracted: {len(paired_data)}")
    data_by_dir = defaultdict(list)
    for p in paired_data:
        data_by_dir[p["direction"]].append(p)
    
    counts = {d: len(v) for d, v in data_by_dir.items()}
    print(f"Counts per direction: {counts}")
    
    min_count = min(counts.values()) if counts else 0
    target_count = min(min_count, 500) # Target 500 per slot for high quality
    
    final_data = []
    random.seed(42)
    for d in dirs:
        samples = data_by_dir.get(d, [])
        if len(samples) > target_count:
            samples = random.sample(samples, target_count)
        final_data.extend(samples)
    
    random.shuffle(final_data)
    
    print(f"Saving {len(final_data)} balanced natural minimal pairs to {output_path}")
    from common import write_jsonl
    write_jsonl(output_path, final_data)

if __name__ == "__main__":
    # Import common inside main or ensure it's in path
    import sys
    sys.path.append("scripts")
    main()
