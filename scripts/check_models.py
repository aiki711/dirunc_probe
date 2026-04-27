from google import genai
import os
import re
from pathlib import Path

BASE_DIR = Path("/home/admin/work/s2550009/dirunc_probe")

def load_api_key():
    env_path = BASE_DIR / ".env"
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^GOOGLE_API_KEY\s*=\s*(.*)", line.strip())
            if match: return match.group(1).strip("'\"")
    return None

def main():
    api_key = load_api_key()
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    print("Listing models for v1beta:")
    for m in client.models.list():
        print(m)

if __name__ == "__main__":
    main()
