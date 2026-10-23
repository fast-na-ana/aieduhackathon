"""
Trains the pulse-consistency transformer.

Run from the project folder:
    py -3.12 -m pulse_model.train --data_dir data/pulse_windows

Writes to runs/<run_name>/:
    best.pt               best checkpoint (by validation AUC) + everything the demo needs to use it
    config.json           the exact settings used
    log.csv               one row per epoch
    test_predictions.csv  per-window test scores
    metrics.json          final test numbers (threshold chosen on VALIDATION, never on test)
    curves.png            loss / AUC / APCER curves
"""
import csv
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import parse_args
from .data import PulseWindows, assign_splits, load_windows
from .model import PulseTransformer


def detection_metrics(y, s, target_bpcer, threshold=None):
    """Scores s are P(fake). Reject when s > threshold.
    BPCER = real users rejected, APCER = attacks accepted (ISO/IEC 30107-3)."""
    real, fake = s[y == 0], s[y == 1]
    if threshold is None:
        threshold = float(np.quantile(real, 1 - target_bpcer)) if len(real) else 0.5
    return dict(
        auc=float(roc_auc_score(y, s)) if len(real) and len(fake) else float("nan"),
        threshold=threshold,
        bpcer=float((real > threshold).mean()) if len(real) else float("nan"),
        apcer=float((fake <= threshold).mean()) if len(fake) else float("nan"),
        acc_at_0_5=float(((s > 0.5) == y).mean()),
    )


@torch.no_grad()
def predict(model, loader, loss_fn, device):
    model.eval()
    scores, total, n = [], 0.0, 0
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        logits = model(x, m)
        total += loss_fn(logits, y).item() * len(y)
        n += len(y)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores), total / max(n, 1)


def plot_curves(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [r["train_loss"] for r in rows], label="train loss")
    ax[0].plot(ep, [r["val_loss"] for r in rows], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("loss")
    ax[1].plot(ep, [r["val_auc"] for r in rows], label="val AUC")
    ax[1].plot(ep, [r["val_apcer"] for r in rows], label="val APCER @ target BPCER")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.02); ax[1].legend(); ax[1].set_title("validation")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    cfg = parse_args()
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- data ----
    d, regions = load_windows(cfg.data_dir, cfg.fps)
    split = assign_splits(d, cfg)
    idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
    print("\nsplit   subjects  real  fake")
    for s, ii in idx.items():
        y = d["y"][ii]
        print(f"{s:<7} {len(set(d['subject'][ii])):>8}  {(y == 0).sum():>4}  {(y == 1).sum():>4}")
    print(f"regions hidden (mask=0): {100 * (1 - d['mask'].mean()):.1f}% of region-windows\n")

    loaders = {
        s: DataLoader(PulseWindows(d, idx[s], cfg, train=(s == "train")),
                      batch_size=cfg.batch_size, shuffle=(s == "train"))
        for s in idx
    }
    window_len = loaders["train"].dataset.L

    # ---- model ----
    model = PulseTransformer(num_regions=len(regions), in_features=loaders["train"].dataset.in_features,
                             d_model=cfg.d_model,
                             num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                             mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    # fakes usually outnumber reals; weight the loss so both classes count equally
    y_train = d["y"][idx["train"]]
    pos_weight = torch.tensor(float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(loaders["train"])
    total_steps = cfg.epochs * steps_per_epoch
    warmup = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):  # linear warm-up, then cosine decay
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---- output folder ----
    run_dir = os.path.join(cfg.out_dir, cfg.run_name or time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    ckpt_path = os.path.join(run_dir, "best.pt")
    print(f"writing to {run_dir}\n")

    # ---- training loop ----
    rows, best, bad_epochs = [], (-1.0, 0.0), 0
    y_val = d["y"][idx["val"]]
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, m, y in loaders["train"]:
            x, m, y = x.to(device), m.to(device), y.to(device)
            loss = loss_fn(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)

        s_val, val_loss = predict(model, loaders["val"], loss_fn, device)
        vm = detection_metrics(y_val, s_val, cfg.target_bpcer)
        row = dict(epoch=epoch, train_loss=total / n, val_loss=val_loss, val_auc=vm["auc"],
                   val_apcer=vm["apcer"], val_acc=vm["acc_at_0_5"], lr=sched.get_last_lr()[0],
                   seconds=time.time() - t0)
        rows.append(row)

        # best = highest AUC, ties broken by lower loss (AUC is NaN if val lacks a class)
        score = (0.0 if math.isnan(vm["auc"]) else vm["auc"], -val_loss)
        improved = score > best
        if improved:
            best, bad_epochs = score, 0
            torch.save(dict(model=model.state_dict(), model_args=model.args, regions=regions,
                            fps=cfg.fps, window_len=window_len, normalize=cfg.normalize,
                            xcorr_lags=cfg.xcorr_lags,
                            threshold=float(vm["threshold"]), target_bpcer=cfg.target_bpcer,
                            epoch=epoch, val_auc=float(vm["auc"])), ckpt_path)
        else:
            bad_epochs += 1

        print(f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val AUC {vm['auc']:.3f} | val APCER@BPCER{cfg.target_bpcer:.0%} {vm['apcer']:.3f} | "
              f"{row['seconds']:.1f}s{'  * saved' if improved else ''}")

        with open(os.path.join(run_dir, "log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        plot_curves(rows, os.path.join(run_dir, "curves.png"))

        if bad_epochs >= cfg.patience:
            print(f"no improvement for {cfg.patience} epochs, stopping early")
            break

    # ---- final test with the best checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    s_test, test_loss = predict(model, loaders["test"], loss_fn, device)
    y_test = d["y"][idx["test"]]
    tm = detection_metrics(y_test, s_test, cfg.target_bpcer, threshold=ckpt["threshold"])
    results = dict(best_epoch=ckpt["epoch"], val_auc=ckpt["val_auc"], test_loss=test_loss,
                   window_level=tm, per_attack={}, clip_level=None)

    attack = d["attack"][idx["test"]]
    for a in sorted(set(attack[y_test == 1])):
        sel = attack == a
        results["per_attack"][a] = dict(n=int(sel.sum()),
                                        apcer=float((s_test[sel] <= ckpt["threshold"]).mean()))

    clip = d["clip"][idx["test"]]
    clips = sorted(set(clip))
    if len(clips) < len(clip):  # several windows per clip: also score each clip by its mean
        cy = np.array([y_test[clip == c][0] for c in clips])
        cs = np.array([s_test[clip == c].mean() for c in clips])
        results["clip_level"] = detection_metrics(cy, cs, cfg.target_bpcer, threshold=ckpt["threshold"])

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "subject", "source_subject", "attack", "label", "score"])
        for i, s in zip(idx["test"], s_test):
            w.writerow([d["clip"][i], d["subject"][i], d["source_subject"][i], d["attack"][i],
                        int(d["y"][i]), f"{s:.5f}"])

    print(f"\nbest epoch {ckpt['epoch']} (val AUC {ckpt['val_auc']:.3f}), "
          f"threshold {ckpt['threshold']:.3f} chosen on validation")
    print(f"TEST window-level: AUC {tm['auc']:.3f} | APCER {tm['apcer']:.3f} | BPCER {tm['bpcer']:.3f}")
    if results["clip_level"]:
        c = results["clip_level"]
        print(f"TEST clip-level:   AUC {c['auc']:.3f} | APCER {c['apcer']:.3f} | BPCER {c['bpcer']:.3f}")
    for a, r in results["per_attack"].items():
        print(f"  attack '{a}': APCER {r['apcer']:.3f} over {r['n']} windows")
    print(f"\nsaved: {run_dir}")


if __name__ == "__main__":
    main()
