import json
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from augmentation.basic import MEAN, STD


def build_transform_with_prob(p: float, size: int = 64):
    """Build an augmentation pipeline governed by probability p in [0, 1] using E1 transform families."""
    p = float(np.clip(p, 0.0, 1.0))
    return T.Compose([
        T.Resize((size, size)),
        T.RandomApply([T.RandomRotation(10)], p=p),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=p),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=p * 0.6),
        T.RandomPerspective(distortion_scale=0.1, p=p * 0.6),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def compute_cgda_probabilities(
    val_confusion_matrix,
    p_min: float = 0.2,
    p_max: float = 0.8,
    gamma: float = 0.75,
) -> dict:
    """Compute per class augmentation probabilities in [p_min, p_max] from a validation confusion matrix."""
    cm = np.asarray(val_confusion_matrix, dtype=np.float64)
    num_classes = cm.shape[0]

    tp = np.diag(cm)
    row_sums = cm.sum(axis=1)  # Actual samples per class (TP plus FN).
    col_sums = cm.sum(axis=0)  # Predicted samples per class (TP plus FP).

    denom = row_sums + col_sums
    f1 = np.divide(2.0 * tp, denom, out=np.ones(num_classes), where=denom > 0)
    confusion_score = 1.0 - f1  # Score of 0.0 means zero errors and 1.0 means total confusion.

    s_min = confusion_score.min()
    s_max = confusion_score.max()
    if s_max - s_min < 1e-12:
        # Fallback if all classes have identical confusion.
        mid = 0.5 * (p_min + p_max)
        return {c: float(mid) for c in range(num_classes)}

    norm_score = (confusion_score - s_min) / (s_max - s_min)
    probs = p_min + (p_max - p_min) * (norm_score ** gamma)
    return {c: float(np.clip(probs[c], p_min, p_max)) for c in range(num_classes)}


def load_static_cgda_probabilities(
    analysis_path: str = "results/analysis/E1_baseline_val_confusion.json",
    p_min: float = 0.2,
    p_max: float = 0.8,
    gamma: float = 0.75,
) -> dict:
    """Load the saved E1 validation confusion matrix and return per class probabilities for static CGDA."""
    data = json.loads(Path(analysis_path).read_text())
    return compute_cgda_probabilities(
        data["val_confusion_matrix_sum"], p_min=p_min, p_max=p_max, gamma=gamma
    )


def build_class_transforms(class_probs: dict, size: int = 64) -> dict:
    """Convert class probabilities into a dictionary of transforms for GTSRBDataset."""
    return {int(c): build_transform_with_prob(p, size=size) for c, p in class_probs.items()}


def compute_budget_matched_prob(class_probs: dict, train_targets: list) -> float:
    """Compute the dataset weighted average probability across the training set for uniform augmentation."""
    probs = [class_probs[int(y)] for y in train_targets]
    return float(np.mean(probs))