import asyncio
import json
import os
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from google import genai

# --- CONFIGURATION (STRICT ABSOLUTE PATHS) ---
BASE_DIR = Path("/home/admin/work/s2550009/dirunc_probe")
DATA_DIR = BASE_DIR / "data/processed/case_grammar"
LOG_PATH = BASE_DIR / "logs/40_generation.log"
STATS_PATH = BASE_DIR / "scripts/40_usage_stats.json"

# 無料枠の複数モデルを併用して合計RPDを稼ぐ設定
MODELS_CONFIG = {
    # 8B相当 (Flash-Lite 最新版)
    "models/gemini-flash-lite-latest": {"limit_rpm": 15, "limit_rpd": 1400},
    # 標準Flash (最新版)
    "models/gemini-flash-latest": {"limit_rpm": 15, "limit_rpd": 20},
    # Pro (最新版)
    "models/gemini-pro-latest": {"limit_rpm": 2, "limit_rpd": 45},
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
        self.api_keys = api_keys
        self.queue = asyncio.Queue()
        self.results_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()
        self.finished_ids = set()
        self.stopped = False
        self.usage_stats = self.load_stats()
        print(f"Initialized with {len(self.clients)} API keys and {len(MODELS_CONFIG)} models.")

    def load_stats(self):
        if STATS_PATH.exists():
            try:
                with STATS_PATH.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except: pass
        return {}

    async def save_stats(self):
        async with self.stats_lock:
            with STATS_PATH.open("w", encoding="utf-8") as f:
                json.dump(self.usage_stats, f, ensure_ascii=False, indent=2)

    def get_today_str(self):
        # 日本時間17時にリセットされるため、17時間引いた日付を「今日」として扱う
        return (datetime.now() - timedelta(hours=17)).strftime("%Y-%m-%d")

    async def increment_usage(self, key_index, model_id):
        today = self.get_today_str()
        stat_key = f"Key{key_index+1}:{model_id}"
        async with self.stats_lock:
            if today not in self.usage_stats:
                self.usage_stats[today] = {}
            self.usage_stats[today][stat_key] = self.usage_stats[today].get(stat_key, 0) + 1
        await self.save_stats()

    def get_usage(self, key_index, model_id):
        today = self.get_today_str()
        stat_key = f"Key{key_index+1}:{model_id}"
        return self.usage_stats.get(today, {}).get(stat_key, 0)

    async def wait_until_tomorrow(self, key_label, model_id):
        now = datetime.now()
        # 次の日本時間17:01:00を計算
        if now.hour < 17:
            target = now.replace(hour=17, minute=1, second=0, microsecond=0)
        else:
            target = (now + timedelta(days=1)).replace(hour=17, minute=1, second=0, microsecond=0)
        
        wait_seconds = (target - now).total_seconds()
        print(f"[{time.strftime('%H:%M:%S')}][{key_label}][{model_id}] RPD LIMIT REACHED. Sleeping until reset (JST 17:01, {target.strftime('%Y-%m-%d %H:%M:%S')})...")
        await asyncio.sleep(wait_seconds)

    async def load_finished_ids(self, paths):
        for p in paths:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            self.finished_ids.add(json.loads(line)["id"])
                        except: pass
        print(f"Loaded {len(self.finished_ids)} already finished IDs.")

    async def worker(self, key_index, model_id, rpm, rpd_limit):
        client = self.clients[key_index]
        key_label = f"Key{key_index+1}"
        delay = 60.0 / rpm
        print(f"Worker [{key_label}][{model_id}] started. (RPM:{rpm}, RPD_Limit:{rpd_limit})")
        
        while not self.stopped:
            # Check RPD limit before taking a task
            if self.get_usage(key_index, model_id) >= rpd_limit:
                await self.wait_until_tomorrow(key_label, model_id)
                continue

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
                        await self.increment_usage(key_index, model_id)
                    
                    self.queue.task_done()
                    await asyncio.sleep(delay)

                except Exception as e:
                    err_msg = str(e).upper()
                    now_str = time.strftime('%H:%M:%S')
                    
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "QUOTA" in err_msg:
                        # 429 Resource Exhausted Error
                        if any(x in err_msg for x in ["LIMIT: 0", "DAILY", "RPD", "DAY", "QUOTA_EXCEEDED"]):
                            # Daily Limit (RPD)
                            print(f"[{now_str}][{key_label}][{model_id}] DAILY QUOTA EXHAUSTED (RPD).")
                            # Mark as maxed out
                            async with self.stats_lock:
                                today = self.get_today_str()
                                if today not in self.usage_stats: self.usage_stats[today] = {}
                                self.usage_stats[today][f"{key_label}:{model_id}"] = rpd_limit
                            retry_delay = 1
                        else:
                            # RPM/TPM Limit or 503 Spike
                            # print(f"[{now_str}][{key_label}][{model_id}] RPM/TPM/Demand spike. Sleeping 60s...")
                            retry_delay = 60
                    else:
                        print(f"[{now_str}][{key_label}][{model_id}] Error: {e}")
                        retry_delay = 60
                    
                    self.queue.put_nowait((row, output_path))
                    self.queue.task_done()
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                print(f"Worker {key_label}:{model_id} fatal error: {e}")
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

    # Start Workers (Each Key x Each Model)
    workers = []
    for k_idx in range(len(api_keys)):
        for model_id, cfg in MODELS_CONFIG.items():
            workers.append(asyncio.create_task(generator.worker(k_idx, model_id, cfg["limit_rpm"], cfg["limit_rpd"])))

    try:
        while not generator.queue.empty():
            await asyncio.sleep(300)
            status_msg = f"[{time.strftime('%H:%M:%S')}] Remaining: {generator.queue.qsize()}"
            # Show summary
            print(status_msg)
        await generator.queue.join()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        generator.stopped = True
        for w in workers:
            w.cancel()

if __name__ == "__main__":
    asyncio.run(main())
