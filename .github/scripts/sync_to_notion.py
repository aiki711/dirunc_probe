import os
import requests
from datetime import datetime
from google import genai

# 環境変数
TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
REPO_NAME = os.environ["REPO_NAME"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
REPO_PATH = os.environ["GITHUB_REPOSITORY"]

# 最新のGemini Client設定
client = genai.Client(api_key=GEMINI_API_KEY)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def get_weekly_commits():
    import subprocess
    cmd = ['git', 'log', '--since="1 week ago"', '--pretty=format:%s|%h']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout.strip():
        return []
    return [line.split('|') for line in result.stdout.strip().split('\n')]

def generate_ai_summary(commits):
    if not commits:
        return "今週の更新はない。"
    
    commit_list = "\n".join([f"- {msg}" for msg, _ in commits])
    prompt = f"""
    以下はリポジトリ「{REPO_NAME}」の今週のコミット履歴である。
    内容を分析し、どのような進捗があったか簡潔な3項目程度の「だである調」で要約せよ。
    
    【コミット履歴】
    {commit_list}
    """
    try:
        # gemini-2.0-flash を使用
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"要約の生成に失敗した。 (詳細: {e})"

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
        blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "更新なし。"}}]}})
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
            "名前": {"title": [{"text": {"content": f"{REPO_NAME} 進捗 ({date_str}週)"}}]},
            "日付": {"date": {"start": date_str}}
        },
        "children": blocks[:100]
    }
    return requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload).json()

if __name__ == "__main__":
    commits = get_weekly_commits()
    summary = generate_ai_summary(commits)
    blocks = build_blocks(commits, summary)
    res = create_notion_page(blocks)
    print(f"Final Success: {res.get('url')}")
