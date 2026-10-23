"""
Every setting for the pulse-consistency transformer lives here.
Change numbers in this file (or on the command line, e.g. --epochs 30), never inside the other files.
"""
import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    # ---- data ----
    data_dir: str = "data/pulse_windows/dfd.npz"   # one .npz file or a folder of them (format: top of data.py)
    scenes: str = ("talking_against_wall,kitchen_still,outside_talking_still_laughing,"
                   "secret_conversation,meeting_serious")
                                          # comma-separated scenes to keep, empty = all. Default: the
                                          # 5 DFD scenes that have fakes, so "walking = real" can't be learned
    cross_test: str = "data/pulse_windows/dfdc.npz"  # another dataset, scored once at the end with the
                                          # validation threshold (never trained or tuned on). Empty = skip
    fps: int = 30
    window_seconds: float = 10.0          # model input = fps * window_seconds samples per region (300)
    normalize: str = "window"             # "window": one scale for all regions (keeps relative amplitude)
                                          # "region": every region scaled on its own
    xcorr_lags: int = 5                   # also give each region its timing vs the face-average pulse
                                          # at -5..+5 frames. 0 = raw waveform only
    val_fraction: float = 0.15            # fraction of SUBJECTS (not windows) held out for validation
    test_fraction: float = 0.15           # fraction of SUBJECTS held out for the final test
    strict_source_split: bool = False     # True: drop val/test fakes whose pasted-on face appears in train
    split_seed: int = 0

    # ---- augmentation (training only) ----
    # Never shift regions in time separately: the relative timing between regions IS the signal.
    aug_region_dropout: float = 0.15      # chance each region is hidden (simulates hijab/collar over the neck)
    aug_noise_std: float = 0.05
    aug_gain_min: float = 0.8             # random per-region amplitude gain
    aug_gain_max: float = 1.25

    # ---- model ----
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 3
    mlp_ratio: int = 4
    dropout: float = 0.1

    # ---- training ----
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    patience: int = 12                    # stop early if val AUC has not improved for this many epochs
    grad_clip: float = 1.0
    target_bpcer: float = 0.05            # report APCER at this BPCER (5% of real users rejected)
    seed: int = 0
    device: str = "auto"                  # "auto", "cuda" or "cpu"
    out_dir: str = "runs"
    run_name: str = ""                    # empty = timestamp


def _str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def parse_args(argv=None):
    """Builds a --flag for every field above, so any setting can be overridden from the command line."""
    cfg = Config()
    p = argparse.ArgumentParser(description="Train the pulse-consistency transformer.")
    for f in fields(Config):
        default = getattr(cfg, f.name)
        kind = _str2bool if isinstance(default, bool) else type(default)
        p.add_argument(f"--{f.name}", type=kind, default=default)
    for k, v in vars(p.parse_args(argv)).items():
        setattr(cfg, k, v)
    return cfg
