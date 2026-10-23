"""
Gives every DFDC original video a person ID. DFDC publishes none, and the same actor appears in
several originals, so splitting DFDC by original video can put one person in two splits.

For each original: a few frames, cropped around the face the pulse tracker followed (cached in
data/pulse_cache/dfdc/<stem>.npz as face = x, y, size), aligned on 5 MediaPipe landmarks and
embedded with OpenCV's SFace face recogniser. The frame embeddings are averaged per video, and the
originals are clustered into people (average linkage on cosine distance).

Writes pulse_pipeline/dfdc_people.json (original stem -> person id), which datasets.py reads, and
data/dfdc_people/contact_sheet.jpg (one row per person) to check the clusters by eye. Crops and
embeddings are cached in data/dfdc_people/, so re-clustering takes seconds. Afterwards rebuild the
windows: py -3.12 -m pulse_pipeline.extract --windows_only

    py -3.12 -m pulse_pipeline.dfdc_people
    py -3.12 -m pulse_pipeline.dfdc_people --threshold 0.5
"""
import argparse
import json
import os
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .datasets import DFDC_DIR, DFDC_PEOPLE
from .regions import _faces_in, landmarker

CACHE_DIR = os.path.join("data", "pulse_cache", "dfdc")
OUT_DIR = os.path.join("data", "dfdc_people")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_recognition_sface_2021dec.onnx")
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
             "face_recognition_sface_2021dec.onnx")
FRAMES = 8  # frames per video, spread over the frames where the tracker had the face
CROP = 256  # crops are 2x the face size, resized to this many pixels
# ArcFace's 112x112 alignment template (SFace uses it too): eye centres, nose tip, mouth corners
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def ensure_model(path=MODEL_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading OpenCV's SFace face recogniser to {path}")
        urllib.request.urlretrieve(MODEL_URL, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def originals(folder=DFDC_DIR):
    with open(os.path.join(folder, "metadata.json")) as f:
        meta = json.load(f)
    return sorted(os.path.splitext(n)[0] for n, m in meta.items() if m["label"] == "REAL")


def square_crop(img, x, y, side, out):
    """A side x side square centred on (x, y), padded with black where it leaves the image."""
    H, W = img.shape[:2]
    x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    return cv2.resize(img[y0 + pad:y1 + pad, x0 + pad:x1 + pad], (out, out), interpolation=cv2.INTER_AREA)


def video_crops(stem):
    """FRAMES face crops [n, CROP, CROP, 3] (BGR) of one original. Runs in a worker process."""
    cv2.setNumThreads(1)
    with np.load(os.path.join(CACHE_DIR, f"{stem}.npz")) as z:
        face = z["face"]
    seen = np.flatnonzero(~np.isnan(face[:, 0]))
    want = set(seen[np.linspace(0, len(seen) - 1, FRAMES).round().astype(int)]) if len(seen) else set()
    crops = []
    cap = cv2.VideoCapture(os.path.join(DFDC_DIR, f"{stem}.mp4"))
    for i in range(max(want, default=-1) + 1):
        if not cap.grab():
            break
        if i in want:
            ok, bgr = cap.retrieve()
            if ok:
                x, y, s = face[i]
                crops.append(square_crop(bgr, x, y, 2 * s, CROP))
    cap.release()
    return stem, np.array(crops, np.uint8).reshape(-1, CROP, CROP, 3)


def five_points(pts):
    """Eye centres, nose tip and mouth corners from MediaPipe's 478 landmarks, image-left first."""
    eyes = sorted([pts[[33, 133]].mean(0), pts[[362, 263]].mean(0)], key=lambda p: p[0])
    mouth = sorted([pts[61], pts[291]], key=lambda p: p[0])
    return np.array([*eyes, pts[1], *mouth], np.float32)


def embed(crops_of):
    """stem -> [n, 128] unit-length SFace embeddings, one per crop; NaN where MediaPipe found no face."""
    rec = cv2.FaceRecognizerSF.create(ensure_model(), "")
    lm = landmarker(video=False, num_faces=4)
    out = {}
    for stem, crops in crops_of.items():
        feats = np.full((len(crops), 128), np.nan, np.float32)
        for i, bgr in enumerate(crops):
            faces = _faces_in(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), lm, (0, 0, CROP, CROP))
            if not faces:
                continue
            pts = min(faces, key=lambda f: np.linalg.norm(f.mean(0) - CROP / 2))  # the tracked face
            M, _ = cv2.estimateAffinePartial2D(five_points(pts), TEMPLATE, method=cv2.LMEDS)
            if M is None:
                continue
            f = rec.feature(cv2.warpAffine(bgr, M, (112, 112))).reshape(-1)
            feats[i] = f / np.linalg.norm(f)
        out[stem] = feats
    lm.close()
    return out


def load_or_make(workers):
    os.makedirs(OUT_DIR, exist_ok=True)
    crops_path, emb_path = os.path.join(OUT_DIR, "crops.npz"), os.path.join(OUT_DIR, "embeddings.npz")
    stems = originals()
    if os.path.exists(crops_path):
        with np.load(crops_path) as z:
            crops_of = {s: z[s] for s in stems}
    else:
        print(f"cropping {FRAMES} frames from each of {len(stems)} originals, {workers} workers")
        with ProcessPoolExecutor(workers) as pool:
            crops_of = dict(pool.map(video_crops, stems))
        np.savez_compressed(crops_path, **crops_of)
    if os.path.exists(emb_path):
        with np.load(emb_path) as z:
            feats_of = {s: z[s] for s in stems}
    else:
        print("embedding the crops")
        feats_of = embed(crops_of)
        np.savez(emb_path, **feats_of)
    return stems, crops_of, feats_of


def cluster(E, threshold):
    """Person label per row of E (unit vectors): average-linkage clusters on cosine distance."""
    D = np.clip(1 - E @ E.T, 0, 2)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=threshold).fit_predict(D)


def contact_sheet(people, crops_of, path, shown=None, per_row=10, thumb=128):
    """One row per person seen in 2+ originals, then everyone seen once, packed per_row a line.
    shown: stem -> which crop to show (default the middle one)."""
    members = {}
    for stem, pid in people.items():
        members.setdefault(pid, []).append(stem)
    multi = sorted((m for m in members.values() if len(m) > 1), key=len, reverse=True)
    single = sorted(m[0] for m in members.values() if len(m) == 1)
    lines = [m[i:i + per_row] for m in multi for i in range(0, len(m), per_row)]
    lines += [single[i:i + per_row] for i in range(0, len(single), per_row)]
    cell_h, gap = thumb + 18, 6
    sheet = np.full((len(lines) * (cell_h + gap) + gap, per_row * (thumb + gap) + gap, 3), 255, np.uint8)
    for r, line in enumerate(lines):
        y = gap + r * (cell_h + gap)
        for c, stem in enumerate(line):
            x = gap + c * (thumb + gap)
            crops = crops_of[stem]
            if len(crops):  # middle 3/4 of the crop = 1.5x the face size
                m, k = CROP // 8, (shown or {}).get(stem, len(crops) // 2)
                sheet[y:y + thumb, x:x + thumb] = cv2.resize(crops[k][m:-m, m:-m], (thumb, thumb),
                                                             interpolation=cv2.INTER_AREA)
            cv2.putText(sheet, f"{people[stem]} {stem}", (x + 1, y + thumb + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    p = argparse.ArgumentParser(description="Cluster DFDC originals into people.")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="cosine distance (1 - similarity) below which clusters merge")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()

    stems, crops_of, feats_of = load_or_make(a.workers)
    found = {s: f[~np.isnan(f[:, 0])] for s, f in feats_of.items()}
    for s in stems:
        if not len(found[s]):
            print(f"WARNING: no face embedded for {s}; it keeps its own person id")
    ok = [s for s in stems if len(found[s])]
    E = np.array([found[s].mean(0) for s in ok])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    labels = dict(zip(ok, cluster(E, a.threshold))) if len(ok) > 1 else {s: 0 for s in ok}
    # the sheet shows each video's crop most like its mean, so a face rather than a passing chair
    shown = {s: int(np.nanargmax(feats_of[s] @ E[i])) for i, s in enumerate(ok)}

    # person ids p00, p01, ... in order of each person's first original (alphabetical)
    ids, people = {}, {}
    for s in stems:
        key = labels.get(s, s)
        ids.setdefault(key, f"p{len(ids):02d}")
        people[s] = ids[key]
    with open(DFDC_PEOPLE, "w") as f:
        json.dump(people, f, indent=1, sort_keys=True)
    sheet = os.path.join(OUT_DIR, "contact_sheet.jpg")
    contact_sheet(people, crops_of, sheet, shown)

    # what a reviewer needs: how sure each video's embedding is, and the closest calls either side
    S = E @ E.T
    same = np.array([[people[a_] == people[b] for b in ok] for a_ in ok])
    iu = np.triu_indices(len(ok), 1)
    within = sorted((S[i, j], ok[i], ok[j]) for i, j in zip(*iu) if same[i, j])
    across = sorted(((S[i, j], ok[i], ok[j]) for i, j in zip(*iu) if not same[i, j]), reverse=True)
    spread = {s: float((found[s] @ E[i]).mean()) for i, s in enumerate(ok)}
    sizes = sorted((list(people.values()).count(pid) for pid in set(people.values())), reverse=True)
    print(f"{len(stems)} originals -> {len(set(people.values()))} people (threshold {a.threshold}); "
          f"videos per person: {sizes}")
    print("people at other thresholds:", {t: len(set(cluster(E, t))) for t in (0.3, 0.4, 0.5, 0.7)})
    print("weakest links inside a person:", *(f"{v:.2f} {x}-{y}" for v, x, y in within[:8]), sep="\n  ")
    print("closest pairs of different people:", *(f"{v:.2f} {x}-{y}" for v, x, y in across[:8]), sep="\n  ")
    print("least consistent videos (mean frame-to-video similarity):",
          *(f"{v:.2f} {s} ({len(found[s])} frames)" for s, v in sorted(spread.items(), key=lambda t: t[1])[:5]),
          sep="\n  ")
    print(f"wrote {DFDC_PEOPLE} and {sheet}")


if __name__ == "__main__":
    main()
