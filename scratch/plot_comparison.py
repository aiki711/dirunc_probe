import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Style configuration
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    
    # Data definition
    methods = ['One-step Prompting\n(Baseline 1)', 'Identify-then-Verify\n(Prior Work / Baseline 2)', 'Probing\n(Ours / Multi-label)']
    
    # 1. Verify Phase Data
    verify_acc = [53.67, 74.33, 79.33]
    verify_f1 = [66.67, 84.44, 86.22]
    
    # 2. Identify Phase Data
    identify_f1 = [0.0, 26.98, 64.83]  # One-step has no slot identification capability
    
    # Premium Color Palette
    # Soft Red/Grey for One-step, Warm Amber for Prior Work, Deep Emerald/Teal for Probing
    colors = ['#EF5350', '#FFA726', '#26A69A']
    
    # Setup Figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    
    # ------------------ Subplot 1: Verify Phase ------------------
    x = np.arange(len(methods))
    width = 0.35
    
    # Draw grouped bars (Accuracy and F1-score)
    rects_acc = ax1.bar(x - width/2, verify_acc, width, label='Accuracy', color='#90A4AE', edgecolor='none', alpha=0.85)
    rects_f1 = ax1.bar(x + width/2, verify_f1, width, label='Omission F1-score', color='#0288D1', edgecolor='none', alpha=0.85)
    
    ax1.set_ylabel('Score (%)', fontsize=11, fontweight='semibold')
    ax1.set_title('Verify Phase: Context Completeness Detection\n(Sufficient vs. Insufficient)', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10, fontweight='semibold')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#cccccc')
    ax1.spines['bottom'].set_color('#cccccc')
    
    # Add values on top of the bars
    def autolabel_verify(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    autolabel_verify(rects_acc, ax1)
    autolabel_verify(rects_f1, ax1)
    
    # ------------------ Subplot 2: Identify Phase ------------------
    # Draw single bars (Macro F1-score) with specific colors per method
    rects_id = ax2.bar(methods, identify_f1, width=0.5, color=colors, edgecolor='none', alpha=0.85)
    
    ax2.set_ylabel('Macro F1-score (%)', fontsize=11, fontweight='semibold')
    ax2.set_title('Identify Phase: Omitted Semantic Slot Identification\n(Multi-class / Multi-label Specificity)', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xticks(np.arange(len(methods)))
    ax2.set_xticklabels(methods, fontsize=10, fontweight='semibold')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#cccccc')
    ax2.spines['bottom'].set_color('#cccccc')
    
    # Add values on top of the bars
    for rect in rects_id:
        height = rect.get_height()
        label = f'{height:.2f}%'
        ax2.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    # Adjust layout
    plt.suptitle('Comparison on 300 Balanced Evaluation Samples (Gemma-2-2b-it)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the plot
    out_path = Path('runs/identify_verify_comparison/comprehensive_comparison.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Chart successfully saved to {out_path}")

if __name__ == "__main__":
    main()
