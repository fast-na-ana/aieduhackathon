"""
Skin-colour traces -> pulse windows.

    colour per region per frame [frames, regions, 3] + frame times
      -> resample to a steady 30 Hz (DFD is 24 fps, DFDC 29.97, webcams jitter)
      -> cut 10 s windows (2 s stride offline; the latest 10 s live)
      -> per window and region: POS -> detrend -> band-pass 0.7-3.0 Hz

Every window is processed on its own, exactly like the live webcam buffer, so training windows and
live windows go through identical maths. All regions of a window get identical processing, so
their relative timing (what the model reads) is never shifted.
"""
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

FPS = 30
WINDOW_S = 10.0
STRIDE_S = 2.0
BAND_HZ = (0.7, 3.0)      # 42-180 bpm
POS_WIN_S = 1.6           # POS sliding window (Wang et al. 2017; rPPG-Toolbox default)
DETREND_LAMBDA = 100      # smoothness-priors detrend (Tarvainen et al. 2002; rPPG-Toolbox default)
BANDPASS_ORDER = 2        # Butterworth, run forwards and backwards (zero phase)
MIN_COVERAGE = 0.8        # a region must be visible in this fraction of a window's frames...
MAX_GAP_S = 0.5           # ...and never hidden longer than this at a time (short gaps are interpolated)
MIN_REGIONS = 3           # windows with fewer visible regions are dropped


def window_len(fps=FPS, seconds=WINDOW_S):
    return int(round(fps * seconds))


def pos(c, fps=FPS):
    """POS (Wang et al. 2017). c: [N, R, 3] RGB traces -> [R, N] pulse."""
    N = c.shape[0]
    l = int(np.ceil(POS_WIN_S * fps))
    w = sliding_window_view(c.astype(np.float64), l, axis=0)  # [N-l+1, R, 3, l]
    cn = w / w.mean(axis=-1, keepdims=True)
    s1 = cn[:, :, 1] - cn[:, :, 2]
    s2 = cn[:, :, 1] + cn[:, :, 2] - 2 * cn[:, :, 0]
    h = s1 + s1.std(-1, keepdims=True) / (s2.std(-1, keepdims=True) + 1e-12) * s2
    h -= h.mean(-1, keepdims=True)
    out = np.zeros((c.shape[1], N))
    for k in range(l):  # overlap-add
        out[:, k:k + N - l + 1] += h[:, :, k].T
    return out


@lru_cache(maxsize=8)
def _detrend_matrix(n, lam):
    eye = np.eye(n)
    d2 = np.diff(eye, 2, axis=0)
    return eye - np.linalg.inv(eye + lam ** 2 * d2.T @ d2)


def detrend(x, lam=DETREND_LAMBDA):
    """Smoothness-priors detrend along the last axis."""
    return x @ _detrend_matrix(x.shape[-1], lam)  # the matrix is symmetric


@lru_cache(maxsize=8)
def _bandpass_sos(fps, lo, hi, order):
    return butter(order, [lo, hi], btype="bandpass", fs=fps, output="sos")


def bandpass(x, fps=FPS, band=BAND_HZ):
    return sosfiltfilt(_bandpass_sos(fps, band[0], band[1], BANDPASS_ORDER), x, axis=-1)


def rgb_to_pulse(c, fps=FPS):
    """[N, R, 3] colour traces (no gaps) -> [R, N] pulse: POS, detrend, band-pass."""
    return bandpass(detrend(pos(c, fps)), fps)


def make_window(t, rgb, t0, fps=FPS, seconds=WINDOW_S):
    """One window starting at time t0 (seconds).

    t: [F] frame times in seconds (increasing). rgb: [F, R, 3], NaN where a region was hidden.
    Returns (X [R, L] float32 pulse, mask [R] bool), or None if too few regions were visible.
    """
    L = window_len(fps, seconds)
    grid = t0 + np.arange(L) / fps
    lo, hi = np.searchsorted(t, [grid[0] - 1.0 / fps, grid[-1] + 1.0 / fps])
    lo, hi = max(lo - 1, 0), min(hi + 1, len(t))  # one frame either side to interpolate the edges
    tw, cw = t[lo:hi], rgb[lo:hi]
    inside = (tw >= grid[0]) & (tw <= grid[-1])

    R = rgb.shape[1]
    mask = np.zeros(R, bool)
    c = np.ones((L, R, 3))
    for r in range(R):
        ok = ~np.isnan(cw[:, r, 0])
        if not ok[inside].any() or ok[inside].mean() < MIN_COVERAGE:
            continue
        tv = tw[ok]
        gaps = np.diff(np.concatenate([[grid[0]], tv[(tv >= grid[0]) & (tv <= grid[-1])], [grid[-1]]]))
        if gaps.max() > MAX_GAP_S:
            continue
        for ch in range(3):
            c[:, r, ch] = np.interp(grid, tv, cw[ok, r, ch])
        mask[r] = True
    if mask.sum() < MIN_REGIONS:
        return None
    X = np.zeros((R, L), np.float32)
    X[mask] = rgb_to_pulse(c[:, mask], fps)
    return X, mask


def window_starts(t, fps=FPS, seconds=WINDOW_S, stride=STRIDE_S):
    """Start times of every full window in a recording with frame times t."""
    span = (window_len(fps, seconds) - 1) / fps
    n = int(np.floor((t[-1] - t[0] - span + 0.5 / fps) / stride)) + 1
    return t[0] + stride * np.arange(max(n, 0))


def heart_rate(X, mask=None, fps=FPS, band=BAND_HZ):
    """Heart rate (bpm) and SNR (dB) of a window, from the average of its visible regions.

    SNR follows de Haan & Jeanne 2013: power within 0.1 Hz of the peak and 0.2 Hz of its first
    harmonic vs. the rest of the band. Above ~0 dB the pulse is clearly visible.
    """
    X = np.atleast_2d(X)
    if mask is None:
        mask = np.ones(len(X), bool)
    x = X[mask]
    x = (x / (x.std(axis=1, keepdims=True) + 1e-12)).mean(axis=0)
    nfft = 4096
    f = np.fft.rfftfreq(nfft, 1 / fps)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    inband = (f >= band[0]) & (f <= band[1])
    f0 = f[inband][np.argmax(p[inband])]
    sig = (np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2)
    rest = (f >= band[0]) & (f <= 2 * band[1]) & ~sig
    snr = 10 * np.log10(p[sig].sum() / (p[rest].sum() + 1e-20) + 1e-20)
    return 60.0 * f0, snr


def region_agreement(X, mask):
    """Mean correlation between every pair of visible regions in a window. A pulse is common to all
    regions while compression noise in separate patches of skin is not, so this is a quality score:
    near 0 means the regions share nothing the model could compare."""
    x = X[mask]
    if len(x) < 2:
        return np.nan
    c = np.corrcoef(x)
    return float(c[np.triu_indices(len(x), 1)].mean())


def recording_windows(t, rgb, fps=FPS, seconds=WINDOW_S, stride=STRIDE_S):
    """All windows of one recording: (starts [W], X [W, R, L], mask [W, R]) for those that passed."""
    starts, Xs, masks = [], [], []
    for t0 in window_starts(t, fps, seconds, stride):
        w = make_window(t, rgb, t0, fps, seconds)
        if w is not None:
            starts.append(t0)
            Xs.append(w[0])
            masks.append(w[1])
    R, L = rgb.shape[1], window_len(fps, seconds)
    if not Xs:
        return np.zeros(0), np.zeros((0, R, L), np.float32), np.zeros((0, R), bool)
    return np.array(starts), np.stack(Xs), np.stack(masks)
