"""
Simple baseline: logistic regression on region agreement and timing (no waveform, no transformer).
Uses the exact same data, person split, threshold rule and metrics as train.py, so the two compare
fairly. Takes the same flags as train.py (only the data/split ones matter).

Run from the project folder:
    py -3.12 -m pulse_model.baseline --run_name baseline_v1

Writes to runs/<run_name>/: baseline.json (weights + everything needed to score a window),
metrics.json, test_predictions.csv.
"""
import csv
import json
import os
import time
from dataclasses import asdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import parse_args
from .data import assign_splits, load_windows
from .tokens import agreement_features
from .train import detection_metrics

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)  # regularisation strength, picked on validation AUC


def features(d, idx, cfg):
    L = int(round(cfg.fps * cfg.window_seconds))
    T = d["X"].shape[-1]
    off = (T - L) // 2
    return np.stack([agreement_features(d["X"][i, :, off:off + L], d["mask"][i], cfg.normalize,
                                        cfg.xcorr_lags) for i in idx])


def main():
    cfg = parse_args()
    d, regions = load_windows(cfg.data_dir, cfg.fps, cfg.scenes)
    split = assign_splits(d, cfg)
    idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
    F = {s: features(d, ii, cfg) for s, ii in idx.items()}
    y = {s: d["y"][ii] for s, ii in idx.items()}
    print(f"{F['train'].shape[1]} features; train {len(y['train'])}, val {len(y['val'])}, "
          f"test {len(y['test'])} windows")

    scaler = StandardScaler().fit(F["train"])
    Z = {s: scaler.transform(f) for s, f in F.items()}
    best = None
    for C in C_GRID:
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Z["train"], y["train"])
        auc = roc_auc_score(y["val"], lr.predict_proba(Z["val"])[:, 1])
        print(f"C={C:<6} val AUC {auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, C, lr)
    val_auc, C, lr = best

    s = {k: lr.predict_proba(z)[:, 1] for k, z in Z.items()}
    vm = detection_metrics(y["val"], s["val"], cfg.target_bpcer)
    threshold = vm["threshold"]
    tm = detection_metrics(y["test"], s["test"], cfg.target_bpcer, threshold=threshold)
    results = dict(model="logistic regression on region agreement", C=C, val_auc=val_auc,
                   window_level=tm, per_attack={}, clip_level=None)

    attack = d["attack"][idx["test"]]
    for a in sorted(set(attack[y["test"] == 1])):
        sel = attack == a
        results["per_attack"][a] = dict(n=int(sel.sum()), apcer=float((s["test"][sel] <= threshold).mean()))
    clip = d["clip"][idx["test"]]
    clips = sorted(set(clip))
    if len(clips) < len(clip):
        cy = np.array([y["test"][clip == c][0] for c in clips])
        cs = np.array([s["test"][clip == c].mean() for c in clips])
        results["clip_level"] = detection_metrics(cy, cs, cfg.target_bpcer, threshold=threshold)

    if cfg.cross_test:
        dx, _ = load_windows(cfg.cross_test, cfg.fps)
        sx = lr.predict_proba(scaler.transform(features(dx, np.arange(len(dx["y"])), cfg)))[:, 1]
        results["cross_test"] = dict(
            path=cfg.cross_test, n_real=int((dx["y"] == 0).sum()), n_fake=int((dx["y"] == 1).sum()),
            **detection_metrics(dx["y"], sx, cfg.target_bpcer, threshold=threshold))

    run_dir = os.path.join(cfg.out_dir, cfg.run_name or "baseline-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(run_dir, "baseline.json"), "w") as f:
        json.dump(dict(regions=regions, fps=cfg.fps, window_len=int(round(cfg.fps * cfg.window_seconds)),
                       normalize=cfg.normalize, xcorr_lags=cfg.xcorr_lags, C=C,
                       mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                       coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                       threshold=threshold, target_bpcer=cfg.target_bpcer), f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "subject", "source_subject", "attack", "label", "score"])
        for i, sc in zip(idx["test"], s["test"]):
            w.writerow([d["clip"][i], d["subject"][i], d["source_subject"][i], d["attack"][i],
                        int(d["y"][i]), f"{sc:.5f}"])

    print(f"\nbest C {C} (val AUC {val_auc:.3f}), threshold {threshold:.3f} chosen on validation")
    print(f"TEST window-level: AUC {tm['auc']:.3f} | APCER {tm['apcer']:.3f} | BPCER {tm['bpcer']:.3f}")
    if results["clip_level"]:
        c = results["clip_level"]
        print(f"TEST clip-level:   AUC {c['auc']:.3f} | APCER {c['apcer']:.3f} | BPCER {c['bpcer']:.3f}")
    if "cross_test" in results:
        c = results["cross_test"]
        print(f"CROSS-DATASET ({c['path']}): AUC {c['auc']:.3f} | APCER {c['apcer']:.3f} | "
              f"BPCER {c['bpcer']:.3f}  ({c['n_real']} real, {c['n_fake']} fake windows)")
    print(f"\nsaved: {run_dir}")


if __name__ == "__main__":
    main()
