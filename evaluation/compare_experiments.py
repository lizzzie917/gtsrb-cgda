import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.class_names import GTSRB_CLASSES
from utils import save_json

EXPERIMENTS = [
    ("E1_baseline", "E1: Baseline"),
    ("E2_uniform", "E2: Uniform Aug"),
    ("E3_static_cgda", "E3: Static CGDA"),
    ("E4_dynamic_cgda", "E4: Dynamic CGDA"),
]
FOCUS_CLASSES = [0, 1, 21, 30]

def main():
    analysis_dir = Path("results/analysis")
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for exp_key, label in EXPERIMENTS:
        path = analysis_dir / f"{exp_key}_val_confusion.json"
        if path.exists():
            loaded.append((exp_key, label, json.loads(path.read_text())))

    if not loaded:
        raise FileNotFoundError("No analysis JSONs found in results/analysis/.")

    print("\n=== Final Experiment Comparison (3-Seed Mean ± Std) ===")
    print(f"{'Experiment':<20} | {'Val Acc':<17} | {'Val Macro F1':<17} | {'Test Acc':<17} | {'Test Macro F1':<17}")
    print("-" * 98)

    summary_rows = []
    for exp_key, label, d in loaded:
        s = d["summary"]
        va = f"{s['val_acc_mean']:.4f} ± {s['val_acc_std']:.4f}"
        vf = f"{s['val_macro_f1_mean']:.4f} ± {s['val_macro_f1_std']:.4f}"
        ta = f"{s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f}"
        tf = f"{s['test_macro_f1_mean']:.4f} ± {s['test_macro_f1_std']:.4f}"
        print(f"{label:<20} | {va:<17} | {vf:<17} | {ta:<17} | {tf:<17}")
        summary_rows.append({"experiment": exp_key, "label": label, **s})

    save_json({"experiments": summary_rows}, analysis_dir / "final_comparison.json")

    # Side-by-side comparison plot: Test Acc, Test Macro F1, and Focus Class Test Recall
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels = [x[1] for x in loaded]
    x = np.arange(len(labels))
    test_acc = [x[2]["summary"]["test_acc_mean"] for x in loaded]
    test_f1 = [x[2]["summary"]["test_macro_f1_mean"] for x in loaded]

    axes[0].bar(x - 0.18, test_acc, width=0.35, label="Test Accuracy", color="#2b5c8f")
    axes[0].bar(x + 0.18, test_f1, width=0.35, label="Test Macro F1", color="#d95f02")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(0.90, 1.00)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall Test Performance Across Experiments")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right")

    width = 0.8 / len(loaded)
    cx = np.arange(len(FOCUS_CLASSES))
    for idx, (_, label, d) in enumerate(loaded):
        recalls = [d["test_per_class_recall"][c] for c in FOCUS_CLASSES]
        offset = (idx - (len(loaded) - 1) / 2) * width
        axes[1].bar(cx + offset, recalls, width=width, label=label)

    axes[1].set_xticks(cx)
    axes[1].set_xticklabels([f"{c}: {GTSRB_CLASSES[c]}" for c in FOCUS_CLASSES], rotation=15, ha="right")
    axes[1].set_ylim(0.55, 1.02)
    axes[1].set_ylabel("Test Recall")
    axes[1].set_title("Test Recall on High-Confusion Sign Classes")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    out_fig = fig_dir / "E1_to_E4_comparison.png"
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison table JSON to: {analysis_dir / 'final_comparison.json'}")
    print(f"Saved comparison figure to    : {out_fig}")

if __name__ == "__main__":
    main()