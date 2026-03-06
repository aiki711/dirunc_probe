# scripts/archive/00_fetch_multiwoz.py
from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

MULTIWOZ_REPO_ZIP_URL = "https://github.com/budzianowski/multiwoz/archive/refs/heads/master.zip"

def download_zip(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        return r.read()

def main() -> None:
    out_dir = Path("data/raw/multiwoz")
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".download_complete"
    if marker.exists():
        print(f"[skip] already downloaded: {out_dir}")
        return

    print(f"[download] {MULTIWOZ_REPO_ZIP_URL}")
    blob = download_zip(MULTIWOZ_REPO_ZIP_URL)

    tmp_dir = Path("data/raw/_tmp_multiwoz_zip")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(tmp_dir)

    # Repo zip contains 'multiwoz-master/'
    repo_root = next(tmp_dir.glob("multiwoz-master*"))
    data_zip_path = repo_root / "data" / "MultiWOZ_2.1.zip"
    
    if not data_zip_path.exists():
        # Maybe it's directly in data/
        print("[warn] MultiWOZ_2.1.zip not found in repo, checking data/ folder directly")
        contents_dir = repo_root / "data"
    else:
        # Extract the inner zip
        print(f"[extract] inner zip: {data_zip_path}")
        inner_tmp = tmp_dir / "inner"
        inner_tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_zip_path) as zf:
            zf.extractall(inner_tmp)
        
        contents_dir = inner_tmp
        # It might be in a subfolder like 'MultiWOZ_2.1'
        subfolders = [p for p in inner_tmp.iterdir() if p.is_dir()]
        if subfolders:
            contents_dir = subfolders[0]

    print(f"[extract] contents_dir={contents_dir}")

    # Copy files
    # MultiWOZ 2.1 often uses .txt or .json for these
    for name_base in ["data", "valListFile", "testListFile"]:
        found = False
        for ext in [".json", ".txt"]:
            name = name_base + ext
            src = contents_dir / name
            if not src.exists():
                # Try recursive
                hits = list(contents_dir.glob(f"**/{name}"))
                if hits:
                    src = hits[0]
            
            if src.exists():
                # Save as .json in raw dir for 02c compatibility? 
                # Actually 02c expects .json extension for everything
                dst = out_dir / (name_base + ".json")
                shutil.copy2(src, dst)
                print(f"  - copied {name} to {dst.name}")
                found = True
                break
        if not found:
            print(f"  - [warn] {name_base} NOT found (.json or .txt)")

    marker.write_text("ok\n", encoding="utf-8")
    shutil.rmtree(tmp_dir)
    print(f"[done] saved to {out_dir}")

if __name__ == "__main__":
    main()
