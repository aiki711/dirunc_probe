# scripts/00_fetch_sgd.py
from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

REPO_ZIP_URL = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip"

def download_zip(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        return r.read()

def main() -> None:
    out_dir = Path("data/raw/sgd")
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".download_complete"
    if marker.exists():
        print(f"[skip] already downloaded: {out_dir}")
        return

    print(f"[download] {REPO_ZIP_URL}")
    blob = download_zip(REPO_ZIP_URL)

    tmp_dir = Path("data/raw/_tmp_sgd_zip")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(tmp_dir)

    # repo root inside zip
    repo_root = next(tmp_dir.glob("dstc8-schema-guided-dialogue-*"))
    print(f"[extract] repo_root={repo_root}")

    # Copy key folders/files
    # Official repo contains train/dev/test and schema files under those folders
    for name in ["train", "dev", "test", "test_unseen", "test_seen", "README.md", "LICENSE"]:
        src = repo_root / name
        if src.exists():
            dst = out_dir / name
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"  - copied {name}")
        else:
            # not all repos have all variants; that's ok
            pass

    marker.write_text("ok\n", encoding="utf-8")
    shutil.rmtree(tmp_dir)
    print(f"[done] saved to {out_dir}")

if __name__ == "__main__":
    main()
