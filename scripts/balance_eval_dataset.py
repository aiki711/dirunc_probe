import json
import random
from pathlib import Path
from collections import defaultdict
import argparse

def balance_dataset(input_path: Path, output_path: Path, max_per_class: int = 5000, seed: int = 42):
    random.seed(seed)
    
    with input_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    
    print(f"Original dataset: {len(rows)} rows")
    
    # クラスごとにインデックスを分類
    # マルチラベルなので、一つの行が複数のクラスに属することがある
    class_indices = defaultdict(list)
    none_indices = []
    
    for i, row in enumerate(rows):
        labels = row.get("labels", {})
        pos_classes = [c for c, v in labels.items() if v == 1]
        
        if not pos_classes:
            none_indices.append(i)
        else:
            for c in pos_classes:
                class_indices[c].append(i)
    
    selected_indices = set()
    
    # 各クラスから最大 max_per_class 件をサンプリング
    for cls, indices in class_indices.items():
        if len(indices) <= max_per_class:
            selected_indices.update(indices)
            print(f"Class '{cls}': kept all {len(indices)} samples")
        else:
            sampled = random.sample(indices, max_per_class)
            selected_indices.update(sampled)
            print(f"Class '{cls}': downsampled from {len(indices)} to {max_per_class}")
            
    # ネガティブサンプル（全ラベル 0）もバランス調整
    # 最小のクラス数程度に合わせる
    min_class_size = min(len(indices) for indices in class_indices.values())
    neg_target_size = max(min_class_size, max_per_class)
    
    if len(none_indices) > neg_target_size:
        sampled_none = random.sample(none_indices, neg_target_size)
    else:
        sampled_none = none_indices
    selected_indices.update(sampled_none)
    print(f"Negative samples: kept {len(sampled_none)} samples")
    
    # 抽出して保存
    final_rows = [rows[i] for i in sorted(list(selected_indices))]
    print(f"Balanced dataset: {len(final_rows)} rows")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_per_class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    balance_dataset(Path(args.input), Path(args.output), args.max_per_class, args.seed)
