"""
Sanity checks for the pulse pipeline on a handful of videos.

For each video: a plot (frame with the five regions drawn, one typical window's pulse per region,
its spectrum, and heart rate across all windows) saved to data/pulse_checks/<video>.png, and a
console row with how many windows fall in a plausible 50-120 bpm.

A heart rate inside 50-120 bpm is NOT enough on its own: a static wall run through the same
filter lands there most of the time too. So every video also gets a control: five background
patches along the top of the frame (no skin, no pulse) go through the identical window code, and
the face must beat them on region agreement (do the regions share one pulse?) and SNR.

For each fake, its original is processed too and the two are compared window by window. A face
swap only replaces the face, so the neck should give (nearly) the same pulse in both, and the heart
rate should match: a swap inherits the driver's pulse. How close they get also shows how much the
two separate video encodes differ.

    py -3.12 -m pulse_pipeline.check --sample 3                # 3 random fakes per dataset + originals
    py -3.12 -m pulse_pipeline.check dfd/01__kitchen_still dfdc/aaqaifqrwn
"""
import argparse
import os
import random

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .datasets import dfd_clips, dfdc_clips
from .extract import cache_path, process_clip
from .pulse import BAND_HZ, FPS, STRIDE_S, WINDOW_S, heart_rate, recording_windows, region_agreement
from .regions import COLOURS_HEX, REGIONS, RegionExtractor, draw_regions

OUT = os.path.join("data", "pulse_checks")
HR_OK = (50, 120)
INK, INK2, MUTED, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#f0efec"


def load(clip):
    """Cached traces + windows + heart rates for one video, or the reason there are none."""
    if not os.path.exists(cache_path(clip)):
        status = process_clip(clip)[1]
        if status != "ok":
            return status
    with np.load(cache_path(clip)) as z:
        d = {k: z[k] for k in z.files}
    d["starts"], d["X"], d["mask"] = recording_windows(d["t"], d["rgb"])
    d["hr"] = np.array([heart_rate(x, m) for x, m in zip(d["X"], d["mask"])]).reshape(-1, 2)
    d["agree"] = np.array([region_agreement(x, m) for x, m in zip(d["X"], d["mask"])])
    d["bg"] = background_control(clip["path"], d["t"])
    return d


def background_control(path, t):
    """Same metrics as the face, from five static patches along the top of the frame."""
    cap = cv2.VideoCapture(path)
    tr = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        H, W = bgr.shape[:2]
        s = int(0.08 * min(H, W))
        y = int(0.02 * H)
        tr.append([bgr[y:y + s, int(x * W) - s // 2:int(x * W) + s // 2, ::-1].reshape(-1, 3).mean(0)
                   for x in (0.1, 0.3, 0.5, 0.7, 0.9)])
    cap.release()
    rgb = np.array(tr, np.float32)[:len(t)]
    _, X, mask = recording_windows(t[:len(rgb)], rgb)
    hr = np.array([heart_rate(x, m) for x, m in zip(X, mask)]).reshape(-1, 2)
    return dict(hr=hr, agree=np.array([region_agreement(x, m) for x, m in zip(X, mask)]))


def frame_with_regions(clip, d, t):
    """The frame at time t with the tracked face's regions drawn on it."""
    i = int(np.argmin(np.abs(d["t"] - t)))
    cap = cv2.VideoCapture(clip["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, bgr = cap.read()
    cap.release()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    face = d["face"][i]
    with RegionExtractor(target=None if np.isnan(face).any() else face) as ex:
        fr = ex(rgb, 0)
    draw_regions(rgb, fr, thickness=max(2, rgb.shape[0] // 400))
    if fr.polygons is not None:  # zoom to the face and neck
        pts = np.concatenate(fr.polygons)
        c, s = pts.mean(0), 1.4 * (pts.max(0) - pts.min(0)).max()
        x0, y0 = np.maximum((c - s / 2).astype(int), 0)
        rgb = rgb[y0:int(c[1] + s / 2), x0:int(c[0] + s / 2)]
    return rgb


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_clip(clip, d, path):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig = plt.figure(figsize=(15, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(9, 2, width_ratios=[1, 1.7], height_ratios=[1] * 5 + [0.3, 1.6, 0.7, 1.6],
                          hspace=0.35, wspace=0.12)
    title = (f"{clip['clip']}   {'FAKE (face swap)' if clip['label'] else 'REAL'}   "
             f"subject {clip['subject']}")
    fig.suptitle(title, x=0.06, ha="left", color=INK, fontsize=13)

    n = len(d["X"])
    if n == 0:
        fig.text(0.5, 0.5, "no usable windows", ha="center", color=INK2, fontsize=14)
        fig.savefig(path, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        return
    w = int(np.argsort(d["hr"][:, 1])[n // 2])  # the window with median SNR = a typical one
    t0 = d["starts"][w]

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(frame_with_regions(clip, d, t0 + WINDOW_S / 2))
    ax.set_axis_off()
    ax.set_title("regions at the middle of the window shown (grey = hidden)", color=INK2, fontsize=9)

    tt = t0 + np.arange(d["X"].shape[-1]) / FPS
    for r, name in enumerate(REGIONS):
        a = fig.add_subplot(gs[r, 1])
        style(a)
        a.set_yticks([])
        a.set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center", color=INK2, fontsize=9)
        if d["mask"][w, r]:
            x = d["X"][w, r]
            a.plot(tt, x / (x.std() + 1e-12), color=COLOURS_HEX[r], lw=1.4)
        else:
            a.text(0.5, 0.5, "hidden", transform=a.transAxes, ha="center", va="center", color=MUTED)
        a.set_xlim(tt[0], tt[-1])
        if r == 0:
            a.set_title(f"pulse per region, typical window ({t0:.0f}-{t0 + WINDOW_S:.0f} s, "
                        f"median SNR of {n})", color=INK2, fontsize=9, loc="left")
        if r < len(REGIONS) - 1:
            a.set_xticklabels([])
        else:
            a.set_xlabel("time (s)", color=MUTED, fontsize=8)

    a = fig.add_subplot(gs[6, 1])
    style(a)
    m = d["mask"][w]
    x = d["X"][w][m]
    x = (x / (x.std(axis=1, keepdims=True) + 1e-12)).mean(0)
    f = np.fft.rfftfreq(4096, 1 / FPS) * 60
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), 4096)) ** 2
    keep = (f >= 30) & (f <= 200)
    a.axvspan(*HR_OK, color=BAND, zorder=0)
    a.plot(f[keep], p[keep] / p[keep].max(), color=COLOURS_HEX[0], lw=1.4)
    hr, snr = d["hr"][w]
    a.plot([hr], [1.0], "o", ms=7, color=COLOURS_HEX[0], mec=SURFACE, mew=2)
    a.annotate(f"{hr:.0f} bpm, SNR {snr:.1f} dB", (hr, 1.0), xytext=(8, -4), textcoords="offset points",
               color=INK, fontsize=9)
    a.set_yticks([])
    a.set_xlim(30, 200)
    a.set_title("spectrum of the face-average pulse (shaded: plausible 50-120 bpm, filter passes 42-180)",
                color=INK2, fontsize=9, loc="left")
    a.set_xlabel("bpm", color=MUTED, fontsize=8)

    a = fig.add_subplot(gs[8, 1])
    style(a)
    a.axhspan(*HR_OK, color=BAND, zorder=0)
    mid = d["starts"] + WINDOW_S / 2
    a.plot(mid, d["hr"][:, 0], "o", ms=6, color=COLOURS_HEX[0], mec=SURFACE, mew=1.5)
    a.set_ylim(BAND_HZ[0] * 60 - 5, BAND_HZ[1] * 60 + 5)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    ok = ((d["hr"][:, 0] >= HR_OK[0]) & (d["hr"][:, 0] <= HR_OK[1])).mean()
    a.set_title(f"heart rate per window ({n} windows, {STRIDE_S:.0f} s stride; {ok:.0%} in 50-120 bpm; "
                f"median region agreement {np.nanmedian(d['agree']):.2f}, "
                f"background patches {np.nanmedian(d['bg']['agree']):.2f})", color=INK2, fontsize=9, loc="left")
    a.set_xlabel("window centre (s)", color=MUTED, fontsize=8)
    a.set_ylabel("bpm", color=MUTED, fontsize=8)

    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def compare(fake, orig):
    """Per-region correlation between a fake's and its original's pulse, in windows at the same time."""
    a, b = {round(s, 2): i for i, s in enumerate(fake["starts"])}, {round(s, 2): i for i, s in enumerate(orig["starts"])}
    common = sorted(set(a) & set(b))
    corr = np.full((len(common), len(REGIONS)), np.nan)
    dhr = []
    for k, s in enumerate(common):
        i, j = a[s], b[s]
        for r in range(len(REGIONS)):
            if fake["mask"][i, r] and orig["mask"][j, r]:
                corr[k, r] = np.corrcoef(fake["X"][i, r], orig["X"][j, r])[0, 1]
        dhr.append(fake["hr"][i, 0] - orig["hr"][j, 0])
    return corr, np.array(dhr)


def main():
    p = argparse.ArgumentParser(description="Plot and sanity-check pulses for a few videos.")
    p.add_argument("clips", nargs="*", help="video ids like dfd/01__kitchen_still or dfdc/aaqaifqrwn")
    p.add_argument("--sample", type=int, default=0, help="random fakes per dataset (plus their originals)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    every = {c["clip"]: c for c in dfdc_clips() + dfd_clips()}
    chosen = [every[c] for c in a.clips]
    rng = random.Random(a.seed)
    for ds in ("dfdc", "dfd"):
        fakes = [c for c in every.values() if c["dataset"] == ds and c["label"] == 1 and c["pair"]]
        chosen += rng.sample(fakes, a.sample) if a.sample else []
    by_path = {c["path"]: c for c in every.values()}
    for c in list(chosen):  # every fake brings its original
        if c["label"] == 1 and c["pair"] in by_path and by_path[c["pair"]] not in chosen:
            chosen.append(by_path[c["pair"]])

    os.makedirs(OUT, exist_ok=True)
    data = {}
    print(f"{'':58s} {'':5s} {'':4s} {'':9s} {'--- face':>10s} {'(background control) ---':>34s}")
    print(f"{'video':58s} {'label':5s} {'win':>4s} {'HR median':>9s} {'in 50-120':>16s} {'agreement':>16s} "
          f"{'SNR dB':>14s}")
    for c in chosen:
        d = data[c["clip"]] = load(c)
        if isinstance(d, str):
            print(f"{c['clip']:58s} {'fake' if c['label'] else 'real':5s}    0  (skipped: {d})")
            continue
        plot_clip(c, d, os.path.join(OUT, c["clip"].replace("/", "__") + ".png"))
        hr, bg = d["hr"], d["bg"]
        if len(hr):
            def in_range(h):
                return ((h[:, 0] >= HR_OK[0]) & (h[:, 0] <= HR_OK[1])).mean() if len(h) else np.nan
            print(f"{c['clip']:58s} {'fake' if c['label'] else 'real':5s} {len(hr):4d} "
                  f"{np.median(hr[:, 0]):9.1f} {in_range(hr):8.0%} ({in_range(bg['hr']):4.0%}) "
                  f"{np.nanmedian(d['agree']):9.2f} ({np.nanmedian(bg['agree']):5.2f}) "
                  f"{np.median(hr[:, 1]):7.1f} ({np.median(bg['hr'][:, 1]):5.1f})")
        else:
            print(f"{c['clip']:58s} {'fake' if c['label'] else 'real':5s}    0  (no usable windows)")

    print("\nfake vs. its original, same time windows: pulse correlation per region "
          "(the neck is untouched by a swap, so it should be near 1)")
    print(f"{'fake':58s} " + " ".join(f"{r[:11]:>11s}" for r in REGIONS) + "  |HR diff| median")
    for c in chosen:
        if c["label"] != 1 or c["pair"] not in by_path:
            continue
        fake, orig = data[c["clip"]], data[by_path[c["pair"]]["clip"]]
        if isinstance(fake, str) or isinstance(orig, str):
            continue
        corr, dhr = compare(fake, orig)
        if not len(dhr):
            print(f"{c['clip']:58s} (no common windows)")
            continue
        cells = " ".join(f"{np.nanmedian(corr[:, r]):11.2f}" if not np.isnan(corr[:, r]).all()
                         else f"{'-':>11s}" for r in range(len(REGIONS)))
        print(f"{c['clip']:58s} {cells}  {np.median(np.abs(dhr)):6.1f} bpm")
    print(f"\nplots in {OUT}")


if __name__ == "__main__":
    main()
