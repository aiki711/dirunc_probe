import json
import re
from pathlib import Path
import spacy

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
MASK_ENTITIY_LABELS = {"GPE", "LOC", "FAC", "ORG", "DATE", "TIME"}

def get_mask_function():
    nlp = spacy.load("en_core_web_sm")
    keyword_pattern = re.compile(r'\b(' + '|'.join(MASK_KEYWORDS) + r')\b', re.IGNORECASE)
    
    def mask_text(text: str) -> str:
        query_start_idx = text.find(" [")
        if query_start_idx != -1 and "[WHO?]" in text:
            content_text = text[:query_start_idx]
            query_text = text[query_start_idx:]
        else:
            content_text = text
            query_text = ""
        masked_text = keyword_pattern.sub("[MASK]", content_text)
        doc = nlp(masked_text)
        spans = []
        for ent in doc.ents:
            if ent.label_ in MASK_ENTITIY_LABELS:
                spans.append((ent.start_char, ent.end_char))
        for token in doc:
            if token.pos_ == "PROPN" and not any(s <= token.idx < e for s, e in spans):
                if token.text not in ["[", "MASK", "]"]:
                     spans.append((token.idx, token.idx + len(token.text)))
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
        result_chars = list(masked_text)
        for s, e in reversed(merged_spans):
            result_chars[s:e] = ["[MASK]"]
        return "".join(result_chars) + query_text
    return mask_text

def analyze_masking(data_path, num_samples=1000):
    mask_fn = get_mask_function()
    total_samples = 0
    total_words = 0
    total_masked_words = 0
    
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples: break
            row = json.loads(line)
            original_text = row["text"]
            masked_text = mask_fn(original_text)
            
            orig_words = original_text.split()
            masked_words = masked_text.split()
            
            # Count mask tokens
            mask_count = masked_text.count("[MASK]")
            
            total_samples += 1
            total_words += len(orig_words)
            total_masked_words += mask_count
            
            if i < 3:
                print(f"--- Sample {i} ---")
                print(f"Original: {original_text}")
                print(f"Masked  : {masked_text}")
    
    return {
        "total_samples": total_samples,
        "total_words": total_words,
        "total_masked_tokens": total_masked_words,
        "word_mask_rate": total_masked_words / total_words if total_words > 0 else 0
    }

if __name__ == "__main__":
    test_file = "data/processed/mixed/dirunc/test.jsonl"
    if Path(test_file).exists():
        stats = analyze_masking(test_file)
        print("\n[Resulting Stats]")
        print(json.dumps(stats, indent=2))
    else:
        print(f"File {test_file} not found.")
