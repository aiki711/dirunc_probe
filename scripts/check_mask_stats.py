
import json
import re
from pathlib import Path

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

def analyze_masking(data_path):
    pattern = re.compile(r'\b(' + '|'.join(MASK_KEYWORDS) + r')\b', re.IGNORECASE)
    
    total_samples = 0
    masked_samples = 0
    total_words = 0
    total_masked_words = 0
    
    with open(data_path, "r") as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            
            # Remove query tokens for word count if possible, but they are part of the input
            # For simplicity, let's just count everything in row["text"]
            words = text.split()
            matches = pattern.findall(text)
            
            total_samples += 1
            if matches:
                masked_samples += 1
            
            total_words += len(words)
            total_masked_words += len(matches)
            
    return {
        "total_samples": total_samples,
        "masked_samples": masked_samples,
        "sample_mask_rate": masked_samples / total_samples if total_samples > 0 else 0,
        "total_words": total_words,
        "total_masked_words": total_masked_words,
        "word_mask_rate": total_masked_words / total_words if total_words > 0 else 0
    }

if __name__ == "__main__":
    test_file = "data/processed/mixed/dirunc/test.jsonl"
    if Path(test_file).exists():
        stats = analyze_masking(test_file)
        print(json.dumps(stats, indent=2))
    else:
        print(f"File {test_file} not found.")
