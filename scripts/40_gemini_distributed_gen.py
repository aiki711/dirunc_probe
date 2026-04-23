import asyncio
import json
import os
import time
import re
from pathlib import Path
from google import genai
from google.api_core import exceptions

# --- CONFIGURATION (STRICT ABSOLUTE PATHS) ---
BASE_DIR = Path("/home/admin/work/s2550009/dirunc_probe")
DATA_DIR = BASE_DIR / "data/processed/case_grammar"
LOG_PATH = BASE_DIR / "scripts/40_generation.log"

MODELS_CONFIG = {
    "models/gemini-2.0-flash": {"limit_rpm": 15},
    "models/gemini-2.0-flash-lite": {"limit_rpm": 15},
    "models/gemini-2.5-flash": {"limit_rpm": 15},
    "models/gemini-2.5-pro": {"limit_rpm": 2},
    "models/gemini-2.5-flash-lite": {"limit_rpm": 15},
    "models/gemini-flash-latest": {"limit_rpm": 15},
    "models/gemini-flash-lite-latest": {"limit_rpm": 15},
    "models/gemini-pro-latest": {"limit_rpm": 2},
    "models/gemini-3-flash-preview": {"limit_rpm": 15},
    "models/gemini-3.1-pro-preview": {"limit_rpm": 2},
    "models/gemini-3.1-flash-lite-preview": {"limit_rpm": 15},
    "models/gemini-3.1-flash-live-preview": {"limit_rpm": 15},
}

def load_api_keys():
    """Loads all GOOGLE_API_KEY* from .env file."""
    keys = []
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return [os.environ.get("GOOGLE_API_KEY")] if os.environ.get("GOOGLE_API_KEY") else []
    
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^GOOGLE_API_KEY(?:_\d+)?\s*=\s*(.*)", line.strip())
            if match:
                key = match.group(1).strip("'\"")
                if key: keys.append(key)
    return list(set(keys)) # unique keys

class DistributedGenerator:
    def __init__(self, api_keys):
        self.clients = [genai.Client(api_key=k, http_options={'api_version': 'v1beta'}) for k in api_keys]
        self.queue = asyncio.Queue()
        self.results_lock = asyncio.Lock()
        self.finished_ids = set()
        self.stopped = False
        print(f"Initialized with {len(self.clients)} API keys.")

    async def load_finished_ids(self, paths):
        for p in paths:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            self.finished_ids.add(json.loads(line)["id"])
                        except: pass
        print(f"Loaded {len(self.finished_ids)} already finished IDs.")

    async def worker(self, key_index, model_id, rpm):
        client = self.clients[key_index]
        key_label = f"Key{key_index+1}"
        delay = 60.0 / rpm
        print(f"Worker [{key_label}][{model_id}] started.")
        
        while not self.stopped:
            try:
                try:
                    row, output_path = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    if self.stopped: break
                    continue

                if row["id"] in self.finished_ids:
                    self.queue.task_done()
                    continue

                try:
                    res = await self.generate_content(client, model_id, row)
                    if res:
                        async with self.results_lock:
                            with output_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                                f.flush()
                            self.finished_ids.add(row["id"])
                    
                    self.queue.task_done()
                    await asyncio.sleep(delay)

                except exceptions.ResourceExhausted as e:
                    err_msg = str(e)
                    if "limit: 0" in err_msg or "Daily" in err_msg or "RPD" in err_msg:
                        print(f"[{key_label}][{model_id}] DAILY QUOTA EXHAUSTED (RPD). Sleeping for 1 hour...")
                        retry_delay = 3600
                    else:
                        print(f"[{key_label}][{model_id}] Temporary quota hit (TPM/RPM). Sleeping for 1 minute...")
                        retry_delay = 60
                    
                    self.queue.put_nowait((row, output_path))
                    self.queue.task_done()
                    await asyncio.sleep(retry_delay)
                
                except Exception as e:
                    print(f"[{key_label}][{model_id}] Error: {e}")
                    self.queue.put_nowait((row, output_path))
                    self.queue.task_done()
                    await asyncio.sleep(60)

            except Exception as e:
                print(f"Worker {key_label} fatal error: {e}")
                await asyncio.sleep(10)

    async def generate_content(self, client, model_id, row):
        role = row["metadata"].get("case_role", "unknown")
        dropped_span = row["metadata"].get("dropped_span", "")
        full_text = row.get("text", "")
        
        if "\nUser: " in full_text:
            context, filled_text_raw = full_text.rsplit("\nUser: ", 1)
        else:
            context = "None"
            filled_text_raw = full_text
        filled_text = filled_text_raw.split("] ", 1)[-1] if "] " in filled_text_raw else filled_text_raw

        prompt = f"""Task: Rewrite "Filled Sentence" into 2 Missing versions (Soft and Strong) where "{dropped_span}" (Role: {role}) is omitted.
Context: {context}
Filled Sentence: {filled_text}
Omit: "{dropped_span}"

JSON format: {{"version_a": "placeholder version", "version_b": "omission version"}}"""

        response = await client.aio.models.generate_content(model=model_id, contents=prompt)
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        res_json = json.loads(text)
        
        row["gemini_soft"] = res_json.get("version_a", "")
        row["gemini_strong"] = res_json.get("version_b", "")
        return row

async def main():
    api_keys = load_api_keys()
    if not api_keys:
        print("No API Keys found.")
        return

    generator = DistributedGenerator(api_keys)
    
    train_src = DATA_DIR / "natural_train.jsonl"
    dev_src = DATA_DIR / "natural_dev.jsonl"
    train_out = DATA_DIR / "natural_train_gemini.jsonl"
    dev_out = DATA_DIR / "natural_dev_gemini.jsonl"

    await generator.load_finished_ids([train_out, dev_out])

    def add_from_file(path, out_path):
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("dataset") == "qasrl": continue
                if row["id"] not in generator.finished_ids:
                    if row["id"].endswith("::filled"):
                        generator.queue.put_nowait((row, out_path))
                        count += 1
        return count

    print("Queuing tasks...")
    dev_count = add_from_file(dev_src, dev_out)
    train_count = add_from_file(train_src, train_out)
    print(f"Total tasks queued: {dev_count + train_count}")

    if (dev_count + train_count) == 0:
        print("No tasks remaining.")
        return

    # Start Workers for EVERY KEY x EVERY MODEL
    workers = []
    for k_idx in range(len(api_keys)):
        for model_id, cfg in MODELS_CONFIG.items():
            workers.append(asyncio.create_task(generator.worker(k_idx, model_id, cfg["limit_rpm"])))

    try:
        while not generator.queue.empty():
            await asyncio.sleep(60)
            print(f"Remaining tasks: {generator.queue.qsize()}")
        await generator.queue.join()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        generator.stopped = True
        for w in workers:
            w.cancel()

if __name__ == "__main__":
    asyncio.run(main())
