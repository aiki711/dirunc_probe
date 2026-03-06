# scripts/archive/00_fetch_qasrl.py
from __future__ import annotations

import io
import os
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen, Request

QASRL_DATA_TAR_URL = "http://qasrl.org/data/qasrl-v2.tar"

def download_data(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        return r.read()

def main() -> None:
    # Based on scripts/02b_process_qasrl.py default
    out_dir = Path("temp_qasrl/qasrl-bank/data/qasrl-v2/orig")
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".download_complete"
    # If the marker exists but files are missing, we should re-download
    if marker.exists() and all((out_dir / f"{s}.jsonl.gz").exists() for s in ["train", "dev", "test"]):
        print(f"[skip] already downloaded: {out_dir}")
        return

    print(f"[download] {QASRL_DATA_TAR_URL}")
    blob = download_data(QASRL_DATA_TAR_URL)

    tmp_tar = Path("temp_qasrl/qasrl-v2.tar")
    tmp_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar.write_bytes(blob)

    print(f"[extract] {tmp_tar}")
    with tarfile.open(tmp_tar, "r") as tf:
        tf.extractall(path=Path("temp_qasrl"))

    # The tar extracts to 'data/qasrl-v2/...' relative to CWD? 
    # Actually download.sh says 'tar xf data/qasrl-v2.tar' which places it in 'data/qasrl-v2/'
    # In our case we extract to 'temp_qasrl', so it should be in 'temp_qasrl/data/qasrl-v2/orig'
    src_dir = Path("temp_qasrl/data/qasrl-v2/orig")
    if not src_dir.exists():
        # Fallback search
        found = list(Path("temp_qasrl").glob("**/train.jsonl.gz"))
        if found:
            src_dir = found[0].parent
            print(f"Found data at {src_dir}")
        else:
            print("[error] Could not find extracted data in temp_qasrl")
            return

    for name in ["train.jsonl.gz", "dev.jsonl.gz", "test.jsonl.gz"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            print(f"  - copied {name}")
        else:
            print(f"  - [warn] {name} not found in extracted tar")

    marker.write_text("ok\n", encoding="utf-8")
    if tmp_tar.exists():
        tmp_tar.unlink()
    
    # Optional cleanup of temp_qasrl/data if preferred, 
    # but we store final in temp_qasrl/qasrl-bank/... for 02b compatibility
    print(f"[done] saved to {out_dir}")

if __name__ == "__main__":
    main()
