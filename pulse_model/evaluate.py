"""
Scores a pulse-windows file with an EXPORTED model (pulse.onnx + pulse.json), the same way the demo
laptop will: numpy preprocessing from tokens.py + onnxruntime, no torch.

Run from the project folder:
    py -3.12 -m pulse_model.evaluate --windows data/pulse_windows_test/dfdc_test.npz
    py -3.12 -m pulse_model.evaluate --windows <file> --export runs/transformer_s0/export

Prints window- and clip-level AUC / APCER / BPCER at the exported threshold (chosen on DFD
validation, never on this data) and writes <windows>_scores_<kind>_<run>.csv (one row per clip)
next to the windows file.
"""
import argparse
import csv
import json
import os

import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

from .tokens import agreement_features, make_tokens

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "export")


def metrics(y, s, threshold):
    real, fake = s[y == 0], s[y == 1]
    return dict(n_real=int(len(real)), n_fake=int(len(fake)),
                auc=float(roc_auc_score(y, s)) if len(real) and len(fake) else float("nan"),
                apcer=float((fake <= threshold).mean()) if len(fake) else float("nan"),
                bpcer=float((real > threshold).mean()) if len(real) else float("nan"))


def main():
    p = argparse.ArgumentParser(description="Score pulse windows with an exported model.")
    p.add_argument("--windows", required=True, help=".npz written by the pulse pipeline")
    p.add_argument("--export", default=EXPORT_DIR, help="folder with pulse.onnx and pulse.json")
    a = p.parse_args()

    with open(os.path.join(a.export, "pulse.json")) as f:
        meta = json.load(f)
    sess = ort.InferenceSession(os.path.join(a.export, "pulse.onnx"), providers=["CPUExecutionProvider"])
    z = np.load(a.windows)
    if [str(r) for r in z["regions"]] != meta["regions"]:
        raise ValueError(f"regions {list(z['regions'])} != model regions {meta['regions']}")
    X = np.nan_to_num(z["X"].astype(np.float32))
    L, T = meta["window_len"], X.shape[-1]
    X = X[:, :, (T - L) // 2:(T - L) // 2 + L]
    mask = z["mask"].astype(bool) & (X.std(axis=-1) > 1e-8)
    y, clip = z["y"].astype(int), z["clip"].astype(str)

    norm, K = meta["normalize"], meta["xcorr_lags"]
    if meta["kind"] == "baseline":
        F = np.stack([agreement_features(x, m, norm, K) for x, m in zip(X, mask)])
        s = sess.run(None, {"features": F})[0]
    else:
        tok = np.stack([make_tokens(x, m, norm, K) for x, m in zip(X, mask)])
        s = sess.run(None, {"tokens": tok, "mask": mask.astype(np.float32)})[0]

    thr = meta["threshold"]
    clips = sorted(set(clip))
    cy = np.array([y[clip == c][0] for c in clips])
    cs = np.array([s[clip == c].mean() for c in clips])
    w, c = metrics(y, s, thr), metrics(cy, cs, thr)
    print(f"model: {meta['kind']} from {meta.get('source_run')}, threshold {thr:.3f} "
          f"(target BPCER {meta['target_bpcer']:.0%} on DFD validation)")
    print(f"data:  {a.windows}: {len(y)} windows from {len(clips)} clips, "
          f"{len(set(z['subject'].astype(str)))} subjects")
    for name, m in (("window", w), ("clip", c)):
        print(f"{name:>6}-level: AUC {m['auc']:.3f} | APCER {m['apcer']:.3f} (fakes let through) | "
              f"BPCER {m['bpcer']:.3f} (reals rejected)  [{m['n_real']} real, {m['n_fake']} fake]")

    out = os.path.splitext(a.windows)[0] + f"_scores_{meta['kind']}_{meta.get('source_run')}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip", "label", "windows", "mean_p_fake", "rejected"])
        for name, lab, sc in zip(clips, cy, cs):
            wr.writerow([name, int(lab), int((clip == name).sum()), f"{sc:.4f}", int(sc > thr)])
    print(f"per-clip scores: {out}")


if __name__ == "__main__":
    main()
