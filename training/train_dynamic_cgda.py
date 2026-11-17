import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentation.basic import get_eval_transform, get_train_transform
from augmentation.cgda import (
    build_class_transforms,
    compute_cgda_probabilities,
)
from data_utils.gtsrb import GTSRBDataset, load_samples, track_aware_split
from evaluation.metrics import compute_metrics, predict
from models.resnet18 import build_resnet18
from utils import get_device, load_config, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--update_every", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    epochs = args.epochs or cfg["epochs"]
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    # data (starts with E1 basic augmentation, adapts every update_every epochs)
    all_train = load_samples(cfg["data_root"], "train")
    test_samples = load_samples(cfg["data_root"], "test")
    tr_idx, va_idx = track_aware_split(
        all_train, cfg["split_file"], cfg["split_seed"], cfg["n_splits"]
    )

    train_ds = GTSRBDataset(
        [all_train[i] for i in tr_idx], get_train_transform(cfg["img_size"])
    )
    val_ds = GTSRBDataset(
        [all_train[i] for i in va_idx], get_eval_transform(cfg["img_size"])
    )
    test_ds = GTSRBDataset(test_samples, get_eval_transform(cfg["img_size"]))
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    g = torch.Generator().manual_seed(args.seed)
    kw = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    # model / optimisation
    model = build_resnet18(cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"E4_dynamic_cgda_seed{args.seed}.pth"
    best_f1, history, prob_schedule, t0 = -1.0, [], [], time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        yv, pv = predict(model, val_loader, device)
        m = compute_metrics(yv, pv, cfg["num_classes"])
        train_loss = running / len(train_ds)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
        })
        print(
            f"epoch {epoch}: loss={train_loss:.4f} "
            f"val_acc={m['accuracy']:.4f} val_f1={m['macro_f1']:.4f}"
        )

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save(model.state_dict(), ckpt)

        # dynamically update per-class augmentation probabilities
        if epoch % args.update_every == 0 and epoch < epochs:
            class_probs = compute_cgda_probabilities(m["confusion_matrix"])
            train_ds.set_class_transforms(
                build_class_transforms(class_probs, cfg["img_size"])
            )
            prob_schedule.append({
                "after_epoch": epoch,
                "class_probs": {str(k): v for k, v in class_probs.items()},
            })
            print(f"  -> Updated dynamic CGDA probabilities after epoch {epoch}")

    train_time = time.time() - t0

    # final evaluation with the best-validation checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    val_m = compute_metrics(*predict(model, val_loader, device), cfg["num_classes"])
    test_m = compute_metrics(*predict(model, test_loader, device), cfg["num_classes"])
    print(f"TEST acc={test_m['accuracy']:.4f}  macro_f1={test_m['macro_f1']:.4f}")

    save_json(
        {
            "experiment": "E4_dynamic_cgda",
            "seed": args.seed,
            "update_every": args.update_every,
            "prob_schedule": prob_schedule,
            "train_time_sec": train_time,
            "history": history,
            "val": val_m,
            "test": test_m,
        },
        Path(cfg["metrics_dir"]) / f"E4_dynamic_cgda_seed{args.seed}.json",
    )

if __name__ == "__main__":
    main()