"""
Makes a SYNTHETIC dataset in the exact format train.py expects. Two uses:
  1. a smoke test that the code runs end to end (its scores mean nothing)
  2. a worked example of the .npz format for whoever builds the real pulse windows

    py -3.12 -m pulse_model.make_synthetic_data --out data/synthetic/synthetic.npz
"""
import argparse
import os

import numpy as np

REGIONS = ["forehead", "left_cheek", "right_cheek", "chin", "neck"]
LAG_MS = np.array([15.0, 0.0, 2.0, -10.0, -40.0])  # made-up "physically plausible" arrival offsets


def pulse(t, hr_hz, lag_s, rng):
    phase = 2 * np.pi * hr_hz * (t - lag_s)
    return np.sin(phase) + 0.3 * np.sin(2 * phase + 0.5) + rng.normal(0, 0.4, len(t))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic/synthetic.npz")
    p.add_argument("--subjects", type=int, default=30)
    p.add_argument("--windows", type=int, default=12, help="windows per clip")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=12.0, help="window length (longer than 10 s on purpose)")
    a = p.parse_args()

    rng = np.random.default_rng(0)
    t = np.arange(int(a.fps * a.seconds)) / a.fps
    X, y, mask, subject, source, clip, attack = [], [], [], [], [], [], []
    people = [f"S{i:02d}" for i in range(a.subjects)]
    for si, s in enumerate(people):
        hr = rng.uniform(1.0, 1.6)  # 60-96 bpm
        neck_hidden = rng.random() < 0.3  # e.g. hijab
        for fake in (0, 1):
            src = people[(si + 1 + rng.integers(a.subjects - 1)) % a.subjects] if fake else s
            for w in range(a.windows):
                if fake:  # same driver pulse, but region timing no longer consistent
                    lags = rng.uniform(-60, 60, len(REGIONS)) / 1000
                else:
                    lags = (LAG_MS + rng.normal(0, 3, len(REGIONS))) / 1000
                X.append([pulse(t, hr, lag, rng) for lag in lags])
                y.append(fake)
                mask.append([True, True, True, True, not neck_hidden])
                subject.append(s)
                source.append(src)
                clip.append(f"{s}_{'swap' if fake else 'real'}")
                attack.append("swap" if fake else "real")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, X=np.array(X, np.float32), y=np.array(y), mask=np.array(mask),
             subject=np.array(subject), source_subject=np.array(source), clip=np.array(clip),
             attack=np.array(attack), regions=np.array(REGIONS), fps=a.fps)
    print(f"wrote {len(y)} SYNTHETIC windows to {a.out}  X shape = {np.array(X).shape}")


if __name__ == "__main__":
    main()
