"""
Loads pulse windows for training and splits them by person.

EXPECTED INPUT: one or more .npz files anywhere under cfg.data_dir (they are joined together).
Each file holds these arrays (N = number of windows, R = regions, T = samples per window):

  required
    X               float32 [N, R, T]  POS pulse waveform per face region, already detrended and
                                       band-passed 0.7-3.0 Hz. T >= fps * window_seconds (300).
    y               int     [N]        0 = real, 1 = fake
    subject         str     [N]        the person whose body/pulse is on camera
                                       (for a face swap: the DRIVER video's person)
  optional
    mask            bool    [N, R]     1 = region visible, 0 = hidden (e.g. neck). Default: all visible
    source_subject  str     [N]        for a face swap: whose FACE was pasted on. Default: = subject
    clip            str     [N]        which video the window came from (enables per-clip scores)
    attack          str     [N]        e.g. "real", "swap", "replay" (for per-attack breakdowns)
    regions         str     [R]        region names; must be identical across files
    fps             scalar             must match cfg.fps
    split           str     [N]        "train" / "val" / "test" to force a split

Save strings as normal numpy string arrays (np.array([...], dtype=str)), not Python objects.
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokens import add_xcorr, make_tokens, preprocess_window  # noqa: F401 (make_tokens re-exported)

REQUIRED = ("X", "y", "subject")
SPLITS = ("train", "val", "test")


def load_windows(data_dir, fps, scenes=""):
    """data_dir: one .npz file or a folder of them. scenes: comma-separated scene names to keep
    (empty = keep everything)."""
    if os.path.isfile(data_dir):
        files = [data_dir]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")

    parts, regions = [], None
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            missing = [k for k in REQUIRED if k not in z]
            if missing:
                raise ValueError(f"{f} is missing required arrays {missing}")
            X = np.nan_to_num(z["X"].astype(np.float32))
            if X.ndim != 3:
                raise ValueError(f"{f}: X must be [N, regions, samples], got shape {X.shape}")
            N, R, _ = X.shape
            if "fps" in z and int(z["fps"]) != fps:
                raise ValueError(f"{f} was recorded at {int(z['fps'])} fps but config says {fps}")

            names = [str(s) for s in z["regions"]] if "regions" in z else [f"region_{i}" for i in range(R)]
            if regions is None:
                regions = names
            elif names != regions:
                raise ValueError(f"{f} has regions {names}, expected {regions}")

            def col(name, default):
                return np.asarray(z[name]).astype(str).reshape(-1) if name in z else default

            subject = z["subject"].astype(str).reshape(-1)
            y = z["y"].astype(np.int64).reshape(-1)
            mask = z["mask"].astype(bool) if "mask" in z else np.ones((N, R), bool)
            base = os.path.basename(f)
            parts.append(dict(
                X=X, y=y, mask=mask, subject=subject,
                source_subject=col("source_subject", subject),
                clip=col("clip", np.array([f"{base}#{i}" for i in range(N)])),
                attack=col("attack", np.where(y == 1, "fake", "real")),
                split=col("split", np.full(N, "")),
                scene=col("scene", np.full(N, "")),
            ))
            for k, v in parts[-1].items():
                if len(v) != N:
                    raise ValueError(f"{f}: '{k}' has {len(v)} entries but X has {N} windows")
            if not np.isin(y, (0, 1)).all():
                raise ValueError(f"{f}: y must contain only 0 (real) and 1 (fake)")

    # files may have different window lengths; centre-crop all to the shortest
    T = min(p["X"].shape[-1] for p in parts)
    for p in parts:
        off = (p["X"].shape[-1] - T) // 2
        p["X"] = p["X"][:, :, off:off + T]

    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["mask"] &= d["X"].std(axis=-1) > 1e-8  # a flat signal means the region wasn't really there
    print(f"loaded {len(d['y'])} windows from {len(files)} file(s); regions = {regions}; "
          f"samples per window = {T}")

    keep_scenes = [s.strip() for s in scenes.split(",") if s.strip()]
    if keep_scenes:
        unknown = set(keep_scenes) - set(d["scene"])
        if unknown:
            raise ValueError(f"scenes {sorted(unknown)} are not in the data")
        keep = np.isin(d["scene"], keep_scenes)
        d = {k: v[keep] for k, v in d.items()}
        print(f"kept {keep.sum()} windows from scenes {keep_scenes} "
              f"({(d['y'] == 0).sum()} real, {(d['y'] == 1).sum()} fake)")
    return d, regions


def assign_splits(d, cfg):
    """Subject-disjoint split: nobody's pulse appears in more than one of train/val/test."""
    forced = d["split"]
    if np.isin(forced, SPLITS).all():
        print("using the 'split' column from the data files")
        split = forced.copy()
    else:
        if (forced != "").any():
            print("WARNING: 'split' column is incomplete, ignoring it and splitting randomly by subject")
        people = np.array(sorted(set(d["subject"])))
        np.random.default_rng(cfg.split_seed).shuffle(people)
        n = len(people)
        n_test = max(1, int(round(n * cfg.test_fraction)))
        n_val = max(1, int(round(n * cfg.val_fraction)))
        if n_test + n_val >= n:
            raise ValueError(f"only {n} subjects; need more to make train/val/test splits")
        group = {p: "test" for p in people[:n_test]}
        group.update({p: "val" for p in people[n_test:n_test + n_val]})
        group.update({p: "train" for p in people[n_test + n_val:]})
        split = np.array([group[s] for s in d["subject"]])

    # a val/test fake whose pasted-on face was also seen during training is a (mild) leak
    is_train = split == "train"
    seen = set(d["subject"][is_train]) | set(d["source_subject"][is_train])
    leak = ~is_train & (d["y"] == 1) & np.isin(d["source_subject"], list(seen))
    if leak.any():
        if cfg.strict_source_split:
            split[leak] = "drop"
            print(f"dropped {leak.sum()} val/test fakes whose source face appears in train")
        else:
            print(f"note: {leak.sum()} val/test fakes use a source face also seen in train "
                  f"(the driver's pulse is still unseen). --strict_source_split true drops them.")

    for s in SPLITS:
        ys = d["y"][split == s]
        if (ys == 0).sum() == 0 or (ys == 1).sum() == 0:
            print(f"WARNING: {s} split has {(ys == 0).sum()} real and {(ys == 1).sum()} fake windows")
    return split


class PulseWindows(Dataset):
    def __init__(self, d, idx, cfg, train):
        self.X, self.mask, self.y = d["X"][idx], d["mask"][idx], d["y"][idx]
        self.L = int(round(cfg.fps * cfg.window_seconds))
        if self.X.shape[-1] < self.L:
            raise ValueError(f"windows have {self.X.shape[-1]} samples, model needs {self.L}")
        self.in_features = self.L + (2 * cfg.xcorr_lags + 1 if cfg.xcorr_lags > 0 else 0)
        self.cfg, self.train = cfg, train
        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self):
        return len(self.y)

    def _drop_regions(self, m):
        visible = np.flatnonzero(m)
        drop = visible[self.rng.random(len(visible)) < self.cfg.aug_region_dropout]
        drop = drop[:max(0, len(visible) - 2)]  # always keep at least 2 regions to compare
        m[drop] = False
        return m

    def __getitem__(self, i):
        cfg, rng = self.cfg, self.rng
        m = self.mask[i].copy()
        T = self.X.shape[-1]
        start = rng.integers(0, T - self.L + 1) if self.train else (T - self.L) // 2
        x = self.X[i, :, start:start + self.L]

        if self.train:
            m = self._drop_regions(m)
        x = preprocess_window(x, m, cfg.normalize)
        if self.train:
            x = x * rng.uniform(cfg.aug_gain_min, cfg.aug_gain_max, size=(x.shape[0], 1))
            x = x + rng.normal(0, cfg.aug_noise_std, size=x.shape)
            x[~m] = 0
        x = add_xcorr(x, m, cfg.xcorr_lags)

        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(m.astype(np.float32)),
                torch.tensor(float(self.y[i])))
