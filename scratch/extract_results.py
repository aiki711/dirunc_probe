#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    transcript_path = Path("/home/admin/.gemini/antigravity-ide/brain/e060296f-135d-45c6-8196-31dddb1a2016/.system_generated/logs/transcript.jsonl")
    if not transcript_path.exists():
        print("Transcript log not found.")
        return

    print("Searching for slotwise_logit_results.json content in transcript...")
    
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "slotwise_logit_results.json" in line:
                # Let's print lines that contain typical json contents or file read contents
                if "verify_accuracy" in line or "slot_f1" in line:
                    try:
                        obj = json.loads(line)
                        # Check if the content contains the file content
                        content = obj.get("content", "")
                        if "verify_accuracy" in content:
                            print("Found candidate content in step index:", obj.get("step_index"))
                            print("---")
                            print(content)
                            print("---")
                    except Exception as e:
                        pass

if __name__ == "__main__":
    main()
