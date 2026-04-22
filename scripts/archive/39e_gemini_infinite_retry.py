import asyncio
import json
import os
import time
import sys
from pathlib import Path
from google import genai
from google.api_core import exceptions
from tenacity import retry, stop_never, wait_exponential, retry_if_exception_type

# --- CONFIGURATION (STRICT ABSOLUTE PATHS) ---
SOURCE_PATH = Path("/home/admin/work/s2550009/dirunc_probe/data/processed/case_grammar/natural_dev.jsonl")
OUTPUT_PATH = Path("/home/admin/work/s2550009/dirunc_probe/data/processed/case_grammar/natural_dev_gemini.jsonl")

MODELS = [
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.0-flash-lite-001"
]

MAX_CONCURRENT_REQUESTS = 5 # Reduced slightly to be more polite under 429

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

    # USE stop_never to wait for quota reset indefinitely
    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=2, min=30, max=600), # Wait up to 10 mins before next try
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
Task: Create two "Missing" versions (Soft and Strong) where specific info "{dropped_span}" (Role: {role}) is omitted.

Context: {context}
Filled Sentence: {filled_text}
Omit information: "{dropped_span}"

Output in valid JSON:
{{
  "version_a": "placeholder version",
  "version_b": "omission version"
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
            except exceptions.ResourceExhausted:
                # Let tenacity handle it via retry decorators
                print(f"Quota exhausted for {model_id}. Retrying after backoff...")
                raise
            except Exception as e:
                print(f"Permanent error for ID {row['id']}: {e}")
                return None

async def main():
    api_key = load_api_key()
    if not api_key:
        print("API Key not found.")
        return

    generator = GeminiGenerator(api_key)

    while True:
        # 1. Load progress
        finished_ids = set()
        if OUTPUT_PATH.exists():
            with OUTPUT_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        finished_ids.add(json.loads(line)["id"])
                    except:
                        continue
        
        # 2. Load targets
        target_tasks = []
        with SOURCE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("dataset") == "qasrl": continue
                if row["id"] not in finished_ids:
                    target_tasks.append(row)
        
        if not target_tasks:
            print("Everything finished successfully.")
            break

        print(f"Status: {len(target_tasks)} tasks remaining. Starting processing.")

        # 3. Process in larger chunks but with internal retries
        chunk_size = 20
        for i in range(0, len(target_tasks), chunk_size):
            chunk = target_tasks[i:i+chunk_size]
            tasks = [generator.generate_single(row) for row in chunk]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            saved_count = 0
            with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
                for res in results:
                    if isinstance(res, dict):
                        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        saved_count += 1
                out_f.flush()
            
            if saved_count > 0:
                print(f"Saved {saved_count} samples. Total progress: {len(finished_ids) + saved_count}")
            
            # If everything in chunk failed with ResourceExhausted, let's sleep heavily
            if saved_count == 0:
                print("No samples saved in this chunk (likely quota). Sleeping for 1 hour...")
                await asyncio.sleep(3600)
                break # Break out of inner loop to reload target_tasks and retry

        await asyncio.sleep(10) # Small breather between full loops

if __name__ == "__main__":
    asyncio.run(main())
