import asyncio
import json
import os
import time
from pathlib import Path
from google import genai
from google.api_core import exceptions
from tenacity import retry, stop_never, wait_exponential, retry_if_exception_type

# --- CONFIGURATION (STRICT ABSOLUTE PATHS) ---
BASE_DIR = Path("/home/admin/work/s2550009/dirunc_probe")
DATA_DIR = BASE_DIR / "data/processed/case_grammar"
LOG_PATH = BASE_DIR / "scripts/40_generation.log"

MODELS_CONFIG = {
    "models/gemini-2.0-flash": {"limit_rpm": 15, "priority": 1},
    "models/gemini-flash-latest": {"limit_rpm": 15, "priority": 1},
    "models/gemini-flash-lite-latest": {"limit_rpm": 15, "priority": 1},
    "models/gemini-pro-latest": {"limit_rpm": 2, "priority": 2}, # Pro has lower RPM in Free Tier
}

def load_api_key():
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return os.environ.get("GOOGLE_API_KEY")
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                return line.strip().split("=", 1)[1].strip("'\"")
    return None

class DistributedGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        self.queue = asyncio.Queue()
        self.results_lock = asyncio.Lock()
        self.finished_ids = set()
        self.stopped = False

    async def load_finished_ids(self, paths):
        for p in paths:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            self.finished_ids.add(json.loads(line)["id"])
                        except: pass
        print(f"Loaded {len(self.finished_ids)} already finished IDs.")

    async def worker(self, model_id, rpm):
        delay = 60.0 / rpm
        print(f"Worker {model_id} started (RPM: {rpm})")
        
        while not self.stopped:
            try:
                # Try to get a task without blocking forever if queue is empty
                try:
                    row, output_path = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    if self.stopped: break
                    continue

                if row["id"] in self.finished_ids:
                    self.queue.task_done()
                    continue

                # Generation attempt
                try:
                    res = await self.generate_content(model_id, row)
                    if res:
                        async with self.results_lock:
                            with output_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                                f.flush()
                            self.finished_ids.add(row["id"])
                    
                    self.queue.task_done()
                    await asyncio.sleep(delay) # Rate limiting

                except exceptions.ResourceExhausted:
                    print(f"[{model_id}] Quota hit. Sleeping for 1 hour...")
                    self.queue.put_nowait((row, output_path)) # Return task to queue
                    self.queue.task_done()
                    await asyncio.sleep(3600) # Heavy sleep on 429
                
                except Exception as e:
                    print(f"[{model_id}] Error for ID {row['id']}: {e}")
                    self.queue.put_nowait((row, output_path)) # Return to try with other models
                    self.queue.task_done()
                    await asyncio.sleep(60)

            except Exception as e:
                print(f"Worker {model_id} fatal error: {e}")
                await asyncio.sleep(10)

    async def generate_content(self, model_id, row):
        role = row["metadata"].get("case_role", "unknown")
        dropped_span = row["metadata"].get("dropped_span", "")
        full_text = row.get("text", "")
        
        # Simple extraction
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

        response = await self.client.aio.models.generate_content(model=model_id, contents=prompt)
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        res_json = json.loads(text)
        
        row["gemini_soft"] = res_json.get("version_a", "")
        row["gemini_strong"] = res_json.get("version_b", "")
        return row

async def main():
    api_key = load_api_key()
    generator = DistributedGenerator(api_key)
    
    # 1. Targets
    train_src = DATA_DIR / "natural_train.jsonl"
    dev_src = DATA_DIR / "natural_dev.jsonl"
    train_out = DATA_DIR / "natural_train_gemini.jsonl"
    dev_out = DATA_DIR / "natural_dev_gemini.jsonl"

    await generator.load_finished_ids([train_out, dev_out])

    # 2. Add tasks to queue (Dev Priority)
    def add_from_file(path, out_path):
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("dataset") == "qasrl": continue
                if row["id"] not in generator.finished_ids:
                    # ONLY Filled sentences as base for rewriting
                    if row["id"].endswith("::filled"):
                        generator.queue.put_nowait((row, out_path))
                        count += 1
        return count

    print("Queuing Dev tasks...")
    dev_count = add_from_file(dev_src, dev_out)
    print("Queuing Train tasks...")
    train_count = add_from_file(train_src, train_out)
    print(f"Total tasks queued: {dev_count + train_count} (Dev: {dev_count}, Train: {train_count})")

    if (dev_count + train_count) == 0:
        print("No tasks remaining.")
        return

    # 3. Start Workers
    workers = []
    for model_id, cfg in MODELS_CONFIG.items():
        workers.append(asyncio.create_task(generator.worker(model_id, cfg["limit_rpm"])))

    # 4. Monitor
    try:
        while not generator.queue.empty():
            await asyncio.sleep(60)
            print(f"Remaining tasks: {generator.queue.qsize()}")
        
        await generator.queue.join() # Wait until all items are processed
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        generator.stopped = True
        for w in workers:
            w.cancel()
        print("Generation finished or stopped.")

if __name__ == "__main__":
    asyncio.run(main())
