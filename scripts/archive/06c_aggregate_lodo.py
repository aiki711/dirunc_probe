import json
import argparse
import glob
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Aggregate LODO results across domains.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing eval JSON files")
    parser.add_argument("--out_summary", type=str, required=True, help="Path to save the summary JSON")
    args = parser.parse_args()

    json_files = glob.glob(str(Path(args.results_dir) / "*_eval.json"))
    
    if not json_files:
        print(f"No result files found in {args.results_dir}")
        return
        
    all_results = []
    macro_f1s = []
    macro_f1s_posonly = []
    
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_results.append(data)
            metrics = data.get("metrics", {})
            if "macro_f1" in metrics:
                macro_f1s.append(metrics["macro_f1"])
            if "macro_f1_posonly" in metrics:
                macro_f1s_posonly.append(metrics["macro_f1_posonly"])
                
    summary = {
        "num_domains_evaluated": len(all_results),
        "macro_average_across_domains": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
        "macro_average_posonly_across_domains": float(np.mean(macro_f1s_posonly)) if macro_f1s_posonly else 0.0,
        "domain_results": all_results
    }
    
    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"=== Aggregation Complete ===")
    print(f"Domains evaluated: {summary['num_domains_evaluated']}")
    print(f"Macro F1 (Overall): {summary['macro_average_across_domains']:.4f}")
    print(f"Macro F1 (Pos-only): {summary['macro_average_posonly_across_domains']:.4f}")
    print(f"Summary saved to {args.out_summary}")

if __name__ == "__main__":
    main()
