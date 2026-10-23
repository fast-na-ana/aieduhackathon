"""
Videos -> pulse windows for training.

Stage 1 (slow, run once): every video -> per-frame skin colour per region, cached in
    data/pulse_cache/<dataset>/<video>.npz. Already-cached videos are skipped, so it can be stopped
    and restarted.
Stage 2 (seconds): cache -> 10 s windows (2 s stride) -> POS pulse, written to
    data/pulse_windows/<dataset>.npz in the format pulse_model/data.py reads, plus
    data/pulse_windows/clips.csv with one quality-check row per video.
    Re-run it alone (--windows_only) after changing anything in pulse.py.

    py -3.12 -m pulse_pipeline.extract                          # everything
    py -3.12 -m pulse_pipeline.extract --datasets dfd --limit 20
    py -3.12 -m pulse_pipeline.extract --windows_only
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from .datasets import dfd_clips, dfdc_clips
from .pulse import FPS, heart_rate, recording_windows, region_agreement
from .regions import REGIONS, RegionExtractor, face_size, find_faces, landmarker

CACHE_DIR = os.path.join("data", "pulse_cache")
OUT_DIR = os.path.join("data", "pulse_windows")
MIN_SECONDS = 10.0


def locate_swap(fake, orig, frames=(0, 15, 30)):
    """Which face a fake swapped, given its frame-aligned original. Finds every face in the original,
    then picks the one whose pixels changed most, relative to the change in a ring just around it
    (re-encoding changes busy textures everywhere; only the swapped face changes much more than its
    surroundings). Returns (x, y, size) in pixels, or None if no face was found."""
    ca, cb = cv2.VideoCapture(fake), cv2.VideoCapture(orig)
    diff, ref = None, {}
    for i in range(max(frames) + 1):
        oka, fa = ca.read()
        okb, fb = cb.read()
        if not (oka and okb) or fa.shape != fb.shape:
            break
        if i in frames:
            d = cv2.absdiff(cv2.GaussianBlur(fa, (5, 5), 0), cv2.GaussianBlur(fb, (5, 5), 0))
            d = d.max(axis=2).astype(np.float32)
            diff = d if diff is None else diff + d
            ref[i] = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
    ca.release()
    cb.release()
    faces = []
    lm = landmarker(video=False, num_faces=4)
    for i in sorted(ref):
        faces = find_faces(ref[i], lm, stop_when_found=False)
        if faces:
            break
    lm.close()
    if not faces:
        return None
    H, W = diff.shape

    def mean_in(c, half):
        x0, y0 = np.maximum((c - half).astype(int), 0)
        x1, y1 = np.minimum((c + half).astype(int), [W, H])
        return diff[y0:y1, x0:x1].sum(), max((x1 - x0) * (y1 - y0), 1)

    best, best_score = None, -1.0
    for pts in faces:
        c, s = pts.mean(0), face_size(pts)
        inner, n_in = mean_in(c, 0.35 * s)
        outer, n_out = mean_in(c, 1.0 * s)
        score = (inner / n_in) / ((outer - inner) / max(n_out - n_in, 1) + 1.0)
        if score > best_score:
            best, best_score = (float(c[0]), float(c[1]), s), score
    return best


def extract_traces(path, target=None):
    """Decodes a video (no re-encoding, no JPEG) and returns per-frame region colours."""
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rgb, skin, nf, face = [], [], [], []
    with RegionExtractor(target=target) as ex:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            fr = ex(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), len(rgb) * 1000.0 / fps)
            rgb.append(fr.rgb)
            skin.append(fr.skin_frac)
            nf.append(fr.n_faces)
            face.append(fr.face)
    cap.release()
    n = len(rgb)
    R = len(REGIONS)
    return dict(t=np.arange(n) / fps, rgb=np.array(rgb, np.float32).reshape(n, R, 3),
                skin_frac=np.array(skin, np.float16).reshape(n, R), n_faces=np.array(nf, np.int8),
                face=np.array(face, np.float32).reshape(n, 3), fps_native=fps, width=W, height=H,
                target=np.array(target if target is not None else [np.nan] * 3, np.float32))


def cache_path(clip):
    return os.path.join(CACHE_DIR, *clip["clip"].split("/")) + ".npz"


def process_clip(clip):
    """Stage 1 for one video. Runs in a worker process."""
    out = cache_path(clip)
    t0 = time.time()
    cap = cv2.VideoCapture(clip["path"])
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or n / fps < MIN_SECONDS - 0.05:
        return clip["clip"], "too_short", time.time() - t0
    target = None
    if clip["pair"]:  # a real video's pair is one of its fakes
        fake, orig = (clip["path"], clip["pair"]) if clip["label"] else (clip["pair"], clip["path"])
        target = locate_swap(fake, orig)
    tr = extract_traces(clip["path"], target)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", **tr)
    os.replace(out + ".tmp.npz", out)
    return clip["clip"], "ok", time.time() - t0


def run_traces(clips, workers):
    todo = [c for c in clips if not os.path.exists(cache_path(c))]
    print(f"stage 1: {len(clips) - len(todo)} videos already cached, {len(todo)} to do, {workers} workers")
    if not todo:
        return
    t0 = time.time()
    skipped = []
    with ProcessPoolExecutor(workers) as pool:
        futs = {pool.submit(process_clip, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                name, status, secs = f.result()
            except Exception as e:  # a broken video must not stop the run
                name, status, secs = futs[f]["clip"], f"error: {e!r}", 0.0
            if status != "ok":
                skipped.append((name, status))
            if i % 25 == 0 or i == len(todo) or status != "ok":
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {name}  {status}  {secs:.1f}s   "
                      f"elapsed {el / 60:.1f} min, eta {el / i * (len(todo) - i) / 60:.1f} min", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "skipped.csv"), "a", newline="") as f:
        csv.writer(f).writerows(skipped)


def build_windows(clips):
    """Stage 2: cache -> one windows file per dataset + clips.csv."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, per_ds = [], {}
    for c in clips:
        p = cache_path(c)
        row = {k: c[k] for k in ("clip", "dataset", "label", "subject", "source_subject", "scene")}
        if not os.path.exists(p):
            rows.append(dict(row, status="no_cache"))
            continue
        with np.load(p) as z:
            t, rgb, nf = z["t"], z["rgb"], z["n_faces"]
            fps_native, target = float(z["fps_native"]), z["target"]
        starts, X, mask = recording_windows(t, rgb)
        hr = np.array([heart_rate(x, m) for x, m in zip(X, mask)]).reshape(-1, 2)
        agree = np.array([region_agreement(x, m) for x, m in zip(X, mask)], np.float32)
        face = ~np.isnan(rgb[:, :, 0]).all(axis=1)
        row.update(status="ok", frames=len(t), fps_native=round(fps_native, 3),
                   swap_located=int(not np.isnan(target).any()),
                   face_frac=round(face.mean(), 3), multi_face_frac=round((nf > 1).mean(), 3),
                   windows=len(X), hr_median=round(np.median(hr[:, 0]), 1) if len(X) else "",
                   snr_median=round(np.median(hr[:, 1]), 2) if len(X) else "",
                   agree_median=round(float(np.nanmedian(agree)), 3) if len(X) else "")
        for r, name in enumerate(REGIONS):
            row[f"vis_{name}"] = round((~np.isnan(rgb[:, r, 0])).mean(), 3)
        rows.append(row)
        if len(X):
            n = len(X)
            d = per_ds.setdefault(c["dataset"], {k: [] for k in (
                "X", "mask", "y", "subject", "source_subject", "clip", "attack", "scene",
                "start_s", "hr_bpm", "snr_db", "agree")})
            d["X"].append(X)
            d["mask"].append(mask)
            d["y"].append(np.full(n, c["label"]))
            for k in ("subject", "source_subject", "clip", "attack", "scene"):
                d[k].append(np.full(n, c[k]))
            d["start_s"].append(starts.astype(np.float32))
            d["hr_bpm"].append(hr[:, 0].astype(np.float32))
            d["snr_db"].append(hr[:, 1].astype(np.float32))
            d["agree"].append(agree)

    for ds, d in per_ds.items():
        arrays = {k: np.concatenate(v) for k, v in d.items()}
        for k in ("subject", "source_subject", "clip", "attack", "scene"):
            arrays[k] = arrays[k].astype(str)
        out = os.path.join(OUT_DIR, f"{ds}.npz")
        np.savez(out, **arrays, regions=np.array(REGIONS), fps=FPS)
        y = arrays["y"]
        print(f"stage 2: {out}: {len(y)} windows ({(y == 0).sum()} real, {(y == 1).sum()} fake) "
              f"from {len(set(arrays['clip']))} videos, X {arrays['X'].shape}")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT_DIR, "clips.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"stage 2: per-video quality report in {os.path.join(OUT_DIR, 'clips.csv')}")


def main():
    p = argparse.ArgumentParser(description="Turn DFDC + DFD videos into pulse windows.")
    p.add_argument("--datasets", default="dfdc,dfd")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--limit", type=int, default=0, help="only the first N videos per dataset (testing)")
    p.add_argument("--match", default="", help="only videos whose name contains this text")
    p.add_argument("--windows_only", action="store_true", help="skip stage 1, rebuild windows from the cache")
    a = p.parse_args()

    clips = []
    for ds in a.datasets.split(","):
        cs = {"dfdc": dfdc_clips, "dfd": dfd_clips}[ds.strip()]()
        cs = [c for c in cs if a.match in c["clip"]]
        clips += cs[:a.limit] if a.limit else cs
    if not a.windows_only:
        run_traces(clips, a.workers)
    build_windows(clips)


if __name__ == "__main__":
    main()
