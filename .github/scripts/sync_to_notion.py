import os
import requests
import time
import subprocess
from datetime import datetime
from google import genai

# 環境変数
TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
REPO_NAME = os.environ["REPO_NAME"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
REPO_PATH = os.environ["GITHUB_REPOSITORY"]

# クライアント初期化
client = genai.Client(api_key=GEMINI_API_KEY)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def get_weekly_commits():
    # コマンドを正確に分割してリストにする
    cmd = [
        'git', 'log', 
        '--since=1 week ago', 
        '--no-merges', 
        '--pretty=format:%s|%h'
    ]
    # shell=False (デフォルト) で実行することで引数を確実に渡す
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if not result.stdout.strip():
        return []
    
    commits = []
    for line in result.stdout.strip().split('\n'):
        if '|' in line:
            parts = line.rsplit('|', 1)
            if len(parts) == 2:
                commits.append(parts)
    return commits

def generate_ai_summary(commits):
    if not commits:
        return "今週の更新はない。"
    
    commit_list = "\n".join([f"- {msg}" for msg, _ in commits])
    prompt = f"以下はリポジトリ「{REPO_NAME}」の今週のコミット履歴である。簡潔に3項目程度の「だである調」で要約せよ。\n\n{commit_list}"
    
    # モデル名の候補
    model_candidates = ['gemini-2.5-flash-lite-preview-09-2025']
    
    for model_name in model_candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
            
    return "（AI要約はクォータ制限により生成できなかった。）"

def build_blocks(commits, ai_summary):
    blocks = [
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "🤖 AI Weekly Summary"}}]}},
        {"object": "block", "type": "callout", "callout": {
            "rich_text": [{"text": {"content": ai_summary}}],
            "icon": {"emoji": "💡"}, "color": "blue_background"
        }},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "📝 Detailed Commit Logs"}}]}}
    ]

    if not commits:
        blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "今週の更新はない。"}}]}})
        return blocks

    categories = {"Features ✨": ["feat"], "Fixes 🛠️": ["fix"], "Refactoring ♻️": ["refactor"], "Others 📄": []}
    grouped = {cat: [] for cat in categories}
    for msg, hash_id in commits:
        found = False
        for cat, keywords in categories.items():
            if any(msg.lower().startswith(k) for k in keywords):
                grouped[cat].append((msg, hash_id)); found = True; break
        if not found: grouped["Others 📄"].append((msg, hash_id))

    for cat, items in grouped.items():
        if not items: continue
        blocks.append({"object": "block", "type": "heading_3", "heading_3": {"rich_text": [{"text": {"content": cat}}]}})
        for msg, hash_id in items[:15]:
            url = f"https://github.com/{REPO_PATH}/commit/{hash_id}"
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"text": {"content": f"{msg} "}},
                        {"text": {"content": f"({hash_id})", "link": {"url": url}}, "annotations": {"code": True, "color": "gray"}}
                    ]
                }
            })
    return blocks

def create_notion_page(blocks):
    date_str = datetime.now().strftime("%Y-%m-%d")
    payload = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Name": {"title": [{"text": {"content": f"{REPO_NAME} 進捗 ({date_str}週)"}}]},
            "Date": {"date": {"start": date_str}}
        },
        "children": blocks[:100]
    }
    return requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload).json()

if __name__ == "__main__":
    commits = get_weekly_commits()
    summary = generate_ai_summary(commits)
    blocks = build_blocks(commits, summary)
    res = create_notion_page(blocks)
    
    if "url" in res:
        print(f"Final Success! Page URL: {res['url']}")
    else:
        print(f"Notion API Error: {res}")
