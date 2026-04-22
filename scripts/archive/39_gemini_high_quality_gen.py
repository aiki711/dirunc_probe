import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.api_core import exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Configuration ---
SOURCE_PATH = Path("data/processed/case_grammar/natural_dev.jsonl") # Start with Dev set for analysis
OUTPUT_PATH = Path("data/processed/case_grammar/natural_dev_gemini.jsonl")
MODEL_ID = "gemini-flash-latest" # Using the most stable high-quota model

def load_api_key():
    env_path = Path(".env")
    if not env_path.exists():
        return os.environ.get("GOOGLE_API_KEY")
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                return line.strip().split("=", 1)[1].strip("'\"")
    return None

# Retry logic for 429/503 errors
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    retry=(retry_if_exception_type(exceptions.ResourceExhausted) | retry_if_exception_type(exceptions.ServiceUnavailable))
)
def generate_missing_versions(client, context, filled_text, role, dropped_span):
    prompt = f"""You are a linguist assisting in Case Grammar research.
Task: Create two "Missing" versions of the "Filled Sentence" where the specific information "{dropped_span}" (Role: {role}) is omitted or replaced.

Context: 
{context}

Filled Sentence: {filled_text}
Omit information: "{dropped_span}"

Instructions:
1. The "Missing" version MUST be perfectly natural and fluent English, sounding like a typical follow-up or reply.
2. The specific value "{dropped_span}" MUST be removed from BOTH versions.
3. Version A (Soft Placeholder): Replace with a context-appropriate placeholder (like "that place", "the city", "it", etc.) that indicates the category of the info but masks the specific detail.
4. Version B (Strong Omission): Completely remove the info or use a minimal pronoun (like "it", "there") ONLY if grammatically necessary.
5. Avoid "Generalization" (e.g., replacing "Thai restaurant" with just "restaurant"). Prefer placeholders or omission.

Output in valid JSON format:
{{
  "version_a": "...",
  "version_b": "..."
}}"""
    
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    text = response.text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    return json.loads(text)

def main():
    api_key = load_api_key()
    if not api_key:
        print("API Key not found.")
        return

    client = genai.Client(api_key=api_key)

    # 1. Check existing progress (Checkpointing)
    finished_ids = set()
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    finished_ids.add(json.loads(line)["id"])
                except:
                    continue
    print(f"Resuming from {len(finished_ids)} already generated samples.")

    # 2. Load target tasks (excluding QA-SRL for priority)
    target_tasks = []
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("dataset") == "qasrl": continue # Focus on dialogue context first
            if row["id"] not in finished_ids:
                target_tasks.append(row)
    
    print(f"Total tasks remaining: {len(target_tasks)}")
    if not target_tasks:
        print("Everything is finished!")
        return

    # 3. Execution
    with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
        for row in tqdm(target_tasks, desc="Generating"):
            role = row["metadata"].get("case_role", "unknown")
            dropped_span = row["metadata"].get("dropped_span", "")
            
            # Correctly extract filled_text and context from 'text' field
            full_text = row.get("text", "")
            
            # Pattern from original scripts: [Prefix] content
            # Often context and filled_text are separated by \nUser: in dialogue data
            if "\nUser: " in full_text:
                context, filled_text_raw = full_text.rsplit("\nUser: ", 1)
            else:
                context = "None"
                filled_text_raw = full_text

            # Strip prefixes like [Domain: ...] or [Target Verb: ...] for Gemini input
            filled_text = filled_text_raw.split("] ", 1)[-1] if "] " in filled_text_raw else filled_text_raw

            try:
                versions = generate_missing_versions(client, context, filled_text, role, dropped_span)
                row["gemini_soft"] = versions.get("version_a", "")
                row["gemini_strong"] = versions.get("version_b", "")
                
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                
                # Sleep briefly to avoid aggressive burst capping
                time.sleep(1.0) 
                
            except Exception as e:
                print(f"\nError generated ID {row['id']}: {e}")
                # We skip and continue to let checkpointing handle it next time
                continue

    print(f"Generation completed successfully. Results at {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
