import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.class_names import GTSRB_CLASSES, SPEED_LIMIT_CLASSES
from utils import load_config, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="E1_baseline",
                    help="Experiment prefix in results/metrics/ (e.g., E1_baseline)")
    args = ap.parse_args()

    cfg = load_config()
    seeds = cfg["seeds"]
    metrics_dir = Path(cfg["metrics_dir"])

    runs = []
    for s in seeds:
        path = metrics_dir / f"{args.exp}_seed{s}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        runs.append(json.loads(path.read_text()))

    val_acc = [r["val"]["accuracy"] for r in runs]
    val_f1 = [r["val"]["macro_f1"] for r in runs]
    test_acc = [r["test"]["accuracy"] for r in runs]
    test_f1 = [r["test"]["macro_f1"] for r in runs]

    val_pcr = np.mean([r["val"]["per_class_recall"] for r in runs], axis=0)
    test_pcr = np.mean([r["test"]["per_class_recall"] for r in runs], axis=0)

    val_cm_sum = np.sum([r["val"]["confusion_matrix"] for r in runs], axis=0)
    row_sums = val_cm_sum.sum(axis=1, keepdims=True)
    val_cm_norm = np.divide(
        val_cm_sum, row_sums,
        out=np.zeros_like(val_cm_sum, dtype=float),
        where=row_sums > 0,
    )

    print(f"\n{args.exp} Summary Across Seeds {seeds}")
    print(f"Val  Accuracy : {np.mean(val_acc):.4f} ± {np.std(val_acc):.4f}")
    print(f"Val  Macro F1 : {np.mean(val_f1):.4f} ± {np.std(val_f1):.4f}")
    print(f"Test Accuracy : {np.mean(test_acc):.4f} ± {np.std(test_acc):.4f}")
    print(f"Test Macro F1 : {np.mean(test_f1):.4f} ± {np.std(test_f1):.4f}")

    worst_val_idx = np.argsort(val_pcr)[:10]
    print("\nTop 10 Lowest Validation Recall Classes")
    for c in worst_val_idx:
        print(f"Class {c:2d} ({GTSRB_CLASSES[int(c)]:<32}): "
              f"Val Recall={val_pcr[c]:.4f} | Test Recall={test_pcr[c]:.4f}")

    off_diag = val_cm_sum.copy()
    np.fill_diagonal(off_diag, 0)
    flat_top = np.argsort(off_diag.ravel())[::-1][:10]
    top_pairs = []
    print("\nTop 10 Validation Confusion Pairs (Across 3 Seeds)")
    for idx in flat_top:
        i, j = divmod(int(idx), off_diag.shape[1])
        cnt = int(off_diag[i, j])
        if cnt == 0:
            break
        top_pairs.append({
            "true_id": i,
            "true_name": GTSRB_CLASSES[i],
            "pred_id": j,
            "pred_name": GTSRB_CLASSES[j],
            "val_errors_total": cnt,
        })
        print(f"True {i:2d} ({GTSRB_CLASSES[i]}) -> Pred {j:2d} ({GTSRB_CLASSES[j]}): {cnt} errors")

    # Save structured validation analysis JSON for CGDA
    analysis_out = Path("results/analysis") / f"{args.exp}_val_confusion.json"
    save_json({
        "experiment": args.exp,
        "seeds": seeds,
        "summary": {
            "val_acc_mean": float(np.mean(val_acc)),
            "val_acc_std": float(np.std(val_acc)),
            "val_macro_f1_mean": float(np.mean(val_f1)),
            "val_macro_f1_std": float(np.std(val_f1)),
            "test_acc_mean": float(np.mean(test_acc)),
            "test_acc_std": float(np.std(test_acc)),
            "test_macro_f1_mean": float(np.mean(test_f1)),
            "test_macro_f1_std": float(np.std(test_f1)),
        },
        "val_per_class_recall": val_pcr.tolist(),
        "test_per_class_recall": test_pcr.tolist(),
        "val_confusion_matrix_sum": val_cm_sum.tolist(),
        "val_confusion_matrix_norm": val_cm_norm.tolist(),
        "top_val_confused_pairs": top_pairs,
    }, analysis_out)

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure A: 43x43 Normalized Validation Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(val_cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-Normalized Recall / Error Rate")
    ticks = np.arange(0, 43, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_xlabel("Predicted Class ID", fontsize=11)
    ax.set_ylabel("True Class ID", fontsize=11)
    ax.set_title(f"{args.exp}: Normalized Validation Confusion Matrix (3-Seed Mean)", fontsize=12)
    fig.savefig(fig_dir / f"{args.exp}_val_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure B: 12 Lowest Per-Class Recall (PCR) Classes
    worst12 = np.argsort(val_pcr)[:12][::-1]
    y_pos = np.arange(len(worst12))
    labels_w = [f"{c}: {GTSRB_CLASSES[int(c)]}" for c in worst12]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y_pos + 0.18, val_pcr[worst12], height=0.35, label="Validation Recall", color="#2b5c8f")
    ax.barh(y_pos - 0.18, test_pcr[worst12], height=0.35, label="Test Recall", color="#d95f02")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_w, fontsize=9)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_title(f"{args.exp}: 12 Lowest-Recall Classes (3-Seed Mean)", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.savefig(fig_dir / f"{args.exp}_worst_classes_pcr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure C: Speed-Limit Per-Class Recall (PCR)
    x_pos = np.arange(len(SPEED_LIMIT_CLASSES))
    speed_labels = [GTSRB_CLASSES[c].replace("Speed limit ", "") for c in SPEED_LIMIT_CLASSES]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x_pos - 0.18, val_pcr[SPEED_LIMIT_CLASSES], width=0.35, label="Validation Recall", color="#2b5c8f")
    ax.bar(x_pos + 0.18, test_pcr[SPEED_LIMIT_CLASSES], width=0.35, label="Test Recall", color="#d95f02")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(speed_labels, fontsize=10)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title(f"{args.exp}: Speed-Limit Per-Class Recall (3-Seed Mean)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.savefig(fig_dir / f"{args.exp}_speed_limits_pcr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved analysis JSON to: {analysis_out}")
    print(f"Saved 3 figures to    : {fig_dir}/")

if __name__ == "__main__":
    main()