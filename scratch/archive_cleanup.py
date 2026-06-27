#!/usr/bin/env python3
"""
Archive unused experimental folders/files from runs/ to archive_runs/
except for identify_verify_comparison.
"""
import shutil
from pathlib import Path

def main():
    root_dir = Path("/home/admin/work/s2550009/dirunc_probe")
    runs_dir = root_dir / "runs"
    archive_dir = root_dir / "archive_runs"

    if not runs_dir.exists():
        print("runs directory does not exist.")
        return

    archive_dir.mkdir(parents=True, exist_ok=True)

    # We keep only identify_verify_comparison in runs/
    keep_dirs = {"identify_verify_comparison"}

    print("Archiving files from runs/ to archive_runs/ ...")
    for item in runs_dir.iterdir():
        if item.name in keep_dirs:
            continue
        
        target = archive_dir / item.name
        
        # If target already exists, resolve conflicts by adding suffix or overwriting
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        
        shutil.move(str(item), str(target))
        print(f"  Moved: runs/{item.name} -> archive_runs/{item.name}")

    print("\nCleanup complete!")

if __name__ == "__main__":
    main()
