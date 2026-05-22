#!/usr/bin/env python3
"""
残り5件のJSONパースエラーを手動修正するスクリプト。
より強化されたプロンプトとパース処理を使って個別にリトライする。
"""
import asyncio
import json
import re
import time
from pathlib import Path
from google import genai

BASE_DIR = Path("/home/admin/work/s2550009/dirunc_probe")
DATA_DIR = BASE_DIR / "data/processed/case_grammar"


def load_api_keys():
    keys = []
    env_path = BASE_DIR / ".env"
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^GOOGLE_API_KEY(?:_\d+)?\s*=\s*(.*)", line.strip())
            if match:
                key = match.group(1).strip("'\"")
                if key:
                    keys.append(key)
    return list(set(keys))


def get_target_rows(path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("dataset") == "qasrl":
                continue
            if row["id"].endswith("::filled"):
                rows[row["id"]] = row
    return rows


def get_finished_ids(paths):
    finished = set()
    for p in paths:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        finished.add(json.loads(line)["id"])
                    except:
                        pass
    return finished


def robust_parse_json(text):
    """より強力なJSONパース。様々な形式のレスポンスに対応。"""
    # 1. まずそのままパース試行
    try:
        return json.loads(text.strip())
    except:
        pass

    # 2. ```json ... ``` ブロック
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass

    # 3. ``` ... ``` ブロック (言語指定なし)
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass

    # 4. テキスト中の { ... } を抽出
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass

    # 5. version_a / version_b をキーワード検索
    va = re.search(r'"version_a"\s*:\s*"([^"]*)"', text)
    vb = re.search(r'"version_b"\s*:\s*"([^"]*)"', text)
    if va and vb:
        return {"version_a": va.group(1), "version_b": vb.group(1)}

    return None


async def generate_with_stronger_prompt(client, model_id, row):
    role = row["metadata"].get("case_role", "unknown")
    dropped_span = row["metadata"].get("dropped_span", "")
    full_text = row.get("text", "")

    if "\nUser: " in full_text:
        context, filled_text_raw = full_text.rsplit("\nUser: ", 1)
    else:
        context = "None"
        filled_text_raw = full_text
    filled_text = (
        filled_text_raw.split("] ", 1)[-1] if "] " in filled_text_raw else filled_text_raw
    )

    # より明示的なプロンプト
    prompt = f"""You must output ONLY a JSON object. No explanation, no markdown, no extra text.

Task: Remove "{dropped_span}" (Role: {role}) from the sentence in two ways.
Context: {context}
Original: {filled_text}

Output exactly this JSON with your versions filled in:
{{"version_a": "soft version with placeholder or vague reference instead of {dropped_span}", "version_b": "strong version with {dropped_span} fully deleted"}}"""

    response = await client.aio.models.generate_content(model=model_id, contents=prompt)
    raw_text = response.text
    print(f"  Raw response: {repr(raw_text[:300])}")

    result = robust_parse_json(raw_text)
    return result


async def main():
    api_keys = load_api_keys()
    if not api_keys:
        print("No API keys found.")
        return

    dev_src = DATA_DIR / "natural_dev.jsonl"
    train_src = DATA_DIR / "natural_train.jsonl"
    dev_out = DATA_DIR / "natural_dev_gemini.jsonl"
    train_out = DATA_DIR / "natural_train_gemini.jsonl"

    dev_targets = get_target_rows(dev_src)
    train_targets = get_target_rows(train_src)
    finished = get_finished_ids([dev_out, train_out])

    all_targets = {**dev_targets, **train_targets}
    remaining = {k: v for k, v in all_targets.items() if k not in finished}

    print(f"Remaining: {len(remaining)} rows")

    # モデルの優先順（pro, flash, flash-lite）
    models_to_try = [
        "models/gemini-flash-lite-latest",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
    ]

    for i, (rid, row) in enumerate(remaining.items()):
        print(f"\n[{i+1}/{len(remaining)}] Processing: {rid}")
        success = False

        for key_idx, key in enumerate(api_keys):
            if success:
                break
            client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})

            for model_id in models_to_try:
                print(f"  Trying Key{key_idx+1} / {model_id}...")
                try:
                    res = await generate_with_stronger_prompt(client, model_id, row)
                    if res and res.get("version_a") and res.get("version_b"):
                        row["gemini_soft"] = res["version_a"]
                        row["gemini_strong"] = res["version_b"]

                        # dev か train か判定して保存
                        out_path = dev_out if rid in dev_targets else train_out
                        with out_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            f.flush()

                        print(f"  ✅ SUCCESS! version_a: {res['version_a'][:80]}")
                        success = True
                        break
                    else:
                        print(f"  ❌ Parse failed (None result)")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                await asyncio.sleep(5)

        if not success:
            print(f"  ⚠️  All attempts failed for {rid}")

        await asyncio.sleep(4)

    print("\n=== Done ===")
    finished2 = get_finished_ids([dev_out, train_out])
    remaining2 = set(all_targets.keys()) - finished2
    print(f"Remaining after fix: {len(remaining2)}")


if __name__ == "__main__":
    asyncio.run(main())
