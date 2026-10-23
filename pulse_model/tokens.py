"""
Window preprocessing shared by training and the demo. Numpy only: the demo laptop runs the
exported ONNX model and must not need torch.

    make_tokens(x, mask, normalize, xcorr_lags)   -> transformer input [R, L + 2K + 1]
    agreement_features(x, mask, normalize, K)     -> baseline input (region agreement + timing)
"""
import numpy as np


def preprocess_window(x, mask, mode):
    """x: [R, L] raw pulse waveforms; mask: [R] bool."""
    x = np.nan_to_num(x.astype(np.float32))
    x = x - x.mean(axis=1, keepdims=True)
    if mode == "region":
        x = x / (x.std(axis=1, keepdims=True) + 1e-6)
    elif mode == "window":
        x = x / ((x[mask].std() if mask.any() else 1.0) + 1e-6)
    else:
        raise ValueError(f"unknown normalize mode {mode!r}")
    x = np.clip(x, -6, 6)
    x[~mask] = 0
    return x


def add_xcorr(x, mask, K):
    """Appends each region's correlation with the face-average pulse at lags -K..K samples.
    This states each region's relative timing explicitly (a lag of 1 frame is only 33 ms at 30 fps,
    which the raw waveform alone hides from the model). Returns [R, L + 2K + 1]."""
    if K <= 0:
        return x
    L = x.shape[1]
    ref = x[mask].mean(axis=0) if mask.any() else np.zeros(L, np.float32)
    core = ref[K:L - K]
    c = np.stack([(x[:, K + k:L - K + k] * core).sum(axis=1) for k in range(-K, K + 1)], axis=1)
    c = c / (np.linalg.norm(x[:, K:L - K], axis=1, keepdims=True) * np.linalg.norm(core) + 1e-6)
    c[~mask] = 0
    return np.concatenate([x, c.astype(np.float32)], axis=1)


def make_tokens(x, mask, normalize, xcorr_lags):
    """Full preprocessing for inference. The demo calls this on each rolling 10 s window."""
    return add_xcorr(preprocess_window(x, mask, normalize), mask, xcorr_lags)


def agreement_features(x, mask, normalize, K):
    """Baseline input: no waveform, only how the regions agree.
    [correlation of every region pair at lag 0 (R*(R-1)/2), each region's correlation with the
    face-average pulse at lags -K..K (R*(2K+1))]. Hidden regions give 0."""
    t = make_tokens(x, mask, normalize, K)
    w = t[:, :t.shape[1] - (2 * K + 1)] if K > 0 else t
    R = w.shape[0]
    n = np.linalg.norm(w, axis=1)
    c = (w @ w.T) / (np.outer(n, n) + 1e-6)
    c[~mask] = 0
    c[:, ~mask] = 0
    pairs = c[np.triu_indices(R, 1)]
    lags = t[:, -(2 * K + 1):].reshape(-1) if K > 0 else np.zeros(0, np.float32)
    return np.concatenate([pairs, lags]).astype(np.float32)
