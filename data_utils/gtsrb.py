import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.datasets import GTSRB

def load_samples(root, split):
    """Returns a list of (image_path, class_id). split = 'train' or 'test'."""
    ds = GTSRB(root=root, split=split, download=True)
    return [(str(p), int(y)) for p, y in ds._samples]

def track_aware_split(samples, split_file, seed=42, n_splits=5):
    """80/20 split by physical-sign track, stratified by class.
    Saved to split_file so everyone (and every experiment) uses the identical split."""
    split_file = Path(split_file)
    if split_file.exists():
        d = json.loads(split_file.read_text())
        return d["train_idx"], d["val_idx"]

    labels = np.array([y for _, y in samples])
    # Filenames look like 00012_00003.ppm -> track 00012, frame 3.
    # Track IDs repeat across classes, so the group is class + track.
    groups = np.array([f"{y}_{Path(p).stem.split('_')[0]}" for p, y in samples])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(np.zeros(len(samples)), labels, groups))

    # Sanity check: no track appears in both sets
    assert not (set(groups[train_idx]) & set(groups[val_idx])), "Track leakage!"

    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(
        {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}))
    return train_idx.tolist(), val_idx.tolist()

class GTSRBDataset(Dataset):
    """Returns (image_tensor, label). Supports optional per-class transforms
    for static (E3) and dynamic (E4) CGDA."""

    def __init__(self, samples, transform=None, class_transforms=None):
        self.samples = samples
        self.transform = transform
        self.class_transforms = class_transforms  # dict: {class_id: transform}
        self.targets = [y for _, y in samples]

    def set_class_transforms(self, class_transforms):
        """Updates per-class transforms between epochs for dynamic CGDA (E4)."""
        self.class_transforms = class_transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.class_transforms is not None and y in self.class_transforms:
            img = self.class_transforms[y](img)
        elif self.transform is not None:
            img = self.transform(img)
        return img, y