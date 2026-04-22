import asyncio
import json
import os
import time
from pathlib import Path
from google import genai
from google.api_core import exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- CONFIGURATION (STRICT ABSOLUTE PATHS) ---
SOURCE_PATH = Path("/home/admin/work/s2550009/dirunc_probe/data/processed/case_grammar/natural_dev.jsonl")
OUTPUT_PATH = Path("/home/admin/work/s2550009/dirunc_probe/data/processed/case_grammar/natural_dev_gemini.jsonl")

MODELS = [
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.0-flash-lite-001"
]

MAX_CONCURRENT_REQUESTS = 10 

def load_api_key():
    env_path = Path("/home/admin/work/s2550009/dirunc_probe/.env")
    if not env_path.exists():
        return os.environ.get("GOOGLE_API_KEY")
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                return line.strip().split("=", 1)[1].strip("'\"")
    return None

class GeminiGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.model_index = 0

    def get_next_model(self):
        model = MODELS[self.model_index]
        self.model_index = (self.model_index + 1) % len(MODELS)
        return model

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=10, max=60),
        retry=(retry_if_exception_type(exceptions.ResourceExhausted) | retry_if_exception_type(exceptions.ServiceUnavailable))
    )
    async def generate_single(self, row):
        role = row["metadata"].get("case_role", "unknown")
        dropped_span = row["metadata"].get("dropped_span", "")
        full_text = row.get("text", "")
        
        if "\nUser: " in full_text:
            context, filled_text_raw = full_text.rsplit("\nUser: ", 1)
        else:
            context = "None"
            filled_text_raw = full_text
        filled_text = filled_text_raw.split("] ", 1)[-1] if "] " in filled_text_raw else filled_text_raw

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

        async with self.semaphore:
            model_id = self.get_next_model()
            try:
                response = await self.client.aio.models.generate_content(model=model_id, contents=prompt)
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                res_json = json.loads(text)
                
                row["gemini_soft"] = res_json.get("version_a", "")
                row["gemini_strong"] = res_json.get("version_b", "")
                return row
            except Exception as e:
                print(f"Error for ID {row['id']} using {model_id}: {e}")
                return None

async def main():
    api_key = load_api_key()
    if not api_key:
        print("API Key not found.")
        return

    # 1. Load progress
    finished_ids = set()
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    finished_ids.add(json.loads(line)["id"])
                except:
                    continue
    print(f"Resuming from {len(finished_ids)} already generated samples.")

    # 2. Load target tasks
    target_tasks = []
    if not SOURCE_PATH.exists():
        print(f"CRITICAL ERROR: Source file not found at {SOURCE_PATH}")
        return

    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("dataset") == "qasrl": continue
            if row["id"] not in finished_ids:
                target_tasks.append(row)
    
    print(f"Total tasks remaining: {len(target_tasks)}")
    if not target_tasks:
        print("Everything is finished!")
        return

    # 3. Execution
    generator = GeminiGenerator(api_key)
    chunk_size = 50 
    for i in range(0, len(target_tasks), chunk_size):
        chunk = target_tasks[i:i+chunk_size]
        tasks = [generator.generate_single(row) for row in chunk]
        
        print(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} samples)...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        saved_count = 0
        with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
            for res in results:
                if isinstance(res, dict):
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    saved_count += 1
            out_f.flush()
        print(f"Finished chunk {i//chunk_size + 1}. Saved {saved_count} samples. Total progress: {len(finished_ids) + (i + saved_count)}")

    print(f"Generation completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
