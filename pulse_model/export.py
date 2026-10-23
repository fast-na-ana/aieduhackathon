"""
Exports a trained pulse model to ONNX for the demo laptop (demo_app/PLAN.md section 11).

Run from the project folder, pointing at a run folder from train.py or baseline.py:
    py -3.12 -m pulse_model.export --run runs/baseline_s0

Writes pulse_model/export/:
    pulse.onnx   one output, p_fake [B] (sigmoid already applied)
    pulse.json   everything the demo needs to turn a window into the model input, plus the threshold

The demo never imports torch. Per 10 s window it does (numpy only, pulse_model/tokens.py):
    kind "transformer": tokens = make_tokens(X, mask, normalize, xcorr_lags)          -> inputs tokens [1, R, F], mask [1, R]
    kind "baseline":    feats  = agreement_features(X, mask, normalize, xcorr_lags)   -> input  features [1, F]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(__file__), "export")


class LogisticModel(nn.Module):
    """The baseline's StandardScaler + logistic regression as one graph."""

    def __init__(self, b):
        super().__init__()
        self.register_buffer("mean", torch.tensor(b["mean"], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(b["scale"], dtype=torch.float32))
        self.register_buffer("coef", torch.tensor(b["coef"], dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(b["intercept"], dtype=torch.float32))

    def forward(self, features):
        return torch.sigmoid(((features - self.mean) / self.scale) @ self.coef + self.intercept)


class TransformerProb(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, mask):
        return torch.sigmoid(self.model(tokens, mask))


def main():
    p = argparse.ArgumentParser(description="Export a trained pulse model to ONNX.")
    p.add_argument("--run", required=True, help="runs/<name> folder with best.pt or baseline.json")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, "pulse.onnx")
    with open(os.path.join(args.run, "metrics.json")) as f:
        metrics = json.load(f)

    if os.path.exists(os.path.join(args.run, "baseline.json")):
        with open(os.path.join(args.run, "baseline.json")) as f:
            b = json.load(f)
        module = LogisticModel(b).eval()
        n = len(b["mean"])
        dummy = (torch.zeros(1, n),)
        inputs, dyn = ["features"], {"features": {0: "batch"}, "p_fake": {0: "batch"}}
        meta = dict(kind="baseline", preprocess="pulse_model.tokens.agreement_features",
                    inputs={"features": [None, n]}, **{k: b[k] for k in (
                        "regions", "fps", "window_len", "normalize", "xcorr_lags", "threshold", "target_bpcer")})
    else:
        from .model import load_model
        model, ckpt = load_model(os.path.join(args.run, "best.pt"))
        module = TransformerProb(model).eval()
        R, F = model.args["num_regions"], model.args["in_features"]
        dummy = (torch.zeros(1, R, F), torch.ones(1, R))
        inputs = ["tokens", "mask"]
        dyn = {"tokens": {0: "batch"}, "mask": {0: "batch"}, "p_fake": {0: "batch"}}
        meta = dict(kind="transformer", preprocess="pulse_model.tokens.make_tokens",
                    inputs={"tokens": [None, R, F], "mask": [None, R]}, **{k: ckpt[k] for k in (
                        "regions", "fps", "window_len", "normalize", "xcorr_lags", "threshold", "target_bpcer")})

    torch.onnx.export(module, dummy, onnx_path, input_names=inputs, output_names=["p_fake"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    meta.update(output="p_fake", source_run=os.path.basename(os.path.normpath(args.run)),
                threshold_note="chosen on DFD validation people for the target BPCER; "
                               "re-tune on the demo camera (README checklist D)",
                val_auc=metrics.get("val_auc"), test=metrics.get("window_level"),
                cross_test=metrics.get("cross_test"))
    with open(os.path.join(args.out_dir, "pulse.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # check: onnxruntime gives the same scores as torch
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    test_in = [rng.normal(size=(8,) + tuple(t.shape[1:])).astype(np.float32) for t in dummy]
    if meta["kind"] == "transformer":
        test_in[1] = (rng.random(test_in[1].shape) > 0.2).astype(np.float32)
    got = sess.run(None, dict(zip(inputs, test_in)))[0]
    with torch.no_grad():
        want = module(*[torch.from_numpy(t) for t in test_in]).numpy()
    err = float(np.abs(got - want).max())
    print(f"exported {meta['kind']} from {args.run} -> {onnx_path} "
          f"({os.path.getsize(onnx_path) / 1024:.1f} KB); max |onnx - torch| = {err:.2e}")
    if err > 1e-4:
        raise SystemExit("ONNX output does not match torch")


if __name__ == "__main__":
    main()
