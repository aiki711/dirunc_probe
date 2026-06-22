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
    
    # Slots to display (ordered by F1-Score descending as provided by the user)
    labels = [
        'who\n(Agent)',
        'when\n(Time)',
        'how\n(Manner/Qty)',
        'what\n(Theme)',
        'which\n(Which)',
        'where\n(Location/Source/Goal)'
    ]
    
    # Performance metrics (%)
    precision = [100.00, 90.38, 86.27, 69.09, 100.00, 74.00]
    recall = [80.00, 87.04, 68.75, 71.70, 50.00, 54.41]
    f1_score = [88.89, 88.68, 76.52, 70.37, 66.67, 62.71]
    
    x = np.arange(len(labels))
    width = 0.25  # width of the bars
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    
    # Harmonious premium color palette
    color_precision = '#42A5F5'  # Soft Blue
    color_recall = '#FFA726'     # Soft Amber
    color_f1 = '#26A69A'         # Premium Teal
    
    rects_p = ax.bar(x - width, precision, width, label='Precision', color=color_precision, edgecolor='none', alpha=0.9)
    rects_r = ax.bar(x, recall, width, label='Recall', color=color_recall, edgecolor='none', alpha=0.9)
    rects_f = ax.bar(x + width, f1_score, width, label='F1-Score', color=color_f1, edgecolor='none', alpha=0.9)
    
    # Customizing axes
    ax.set_ylabel('Score (%)', fontsize=11, fontweight='semibold')
    ax.set_title('Detailed Probing Performance per Semantic Slot\n(Identify Phase / Multi-label Baseline Probes)', fontsize=13, fontweight='bold', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='semibold')
    ax.set_ylim(0, 115)
    
    # Grid and Spines
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Custom Legend
    # Since Japanese font can fail in some envs, we use bilingual labels for safety
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=10)
    
    # Function to add value labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    autolabel(rects_p)
    autolabel(rects_r)
    autolabel(rects_f)
    
    plt.tight_layout()
    
    # Save the plot
    out_path = Path('runs/identify_verify_comparison/slot_performance.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Slot performance chart successfully saved to {out_path}")

if __name__ == "__main__":
    main()
