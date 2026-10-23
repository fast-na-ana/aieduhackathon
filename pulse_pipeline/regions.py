"""
Face regions for the pulse pipeline.

MediaPipe Face Landmarker -> five skin patches (forehead, left cheek, right cheek, chin, neck)
-> mean skin colour (RGB) per patch per frame.

Frames go in as raw decoded arrays straight from the decoder or the webcam. Nothing is ever
JPEG-compressed on the way, because compression wipes out the tiny colour changes we measure.

Each patch only averages pixels that look like the face's own skin (chroma close to the cheeks and
forehead in that frame), so hair, beard, glasses, collars and hijabs are left out. A patch that is
mostly not skin, or mostly outside the frame, is reported as hidden. This is how the neck gets
masked when a collar or hijab covers it. A cheek turned away from the camera is hidden too.

The face is followed with a moving crop around it. That keeps the tracker on one person when
several are in shot, and makes small faces (people far from the camera) big enough to detect.

"left" and "right" are the person's own left and right (on an unmirrored camera their left cheek is
on the right of the image).
"""
import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

REGIONS = ["forehead", "left_cheek", "right_cheek", "chin", "neck"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")

# MediaPipe face-mesh landmark indices, walked around each patch's outline.
# Chosen by eye on a landmark map; they stay clear of the eyes, brows, nose, lips and the face edge.
POLYGONS = {
    "forehead":    [103, 67, 109, 10, 338, 297, 332, 334, 296, 336, 9, 107, 66, 105],
    "left_cheek":  [345, 346, 347, 348, 329, 266, 423, 426, 427, 411, 376, 352],
    "right_cheek": [116, 117, 118, 119, 100, 36, 203, 206, 207, 187, 147, 123],
    "chin":        [194, 201, 200, 421, 418, 262, 369, 396, 175, 171, 140, 32],
}
# The face mesh stops at the jaw, so the neck patch is placed relative to the face:
# start NECK_GAP face-heights below the chin tip, NECK_LEN face-heights long, NECK_HALF_WIDTH
# face-widths either side of the centre line. "Down" follows the head's tilt (forehead -> chin).
FOREHEAD_TOP, CHIN_TIP, FACE_LEFT_EDGE, FACE_RIGHT_EDGE = 10, 152, 454, 234
NECK_GAP, NECK_LEN, NECK_HALF_WIDTH = 0.12, 0.25, 0.22

SKIN_REF = ("forehead", "left_cheek", "right_cheek")  # define "this person's skin" per frame
SKIN_K = 3.0           # a pixel is skin if its Cr and Cb are within SKIN_K robust std of the reference
SKIN_MIN_SPREAD = 3.0  # floor on that std (0-255 chroma units) so a very even face isn't over-strict
DARK, CLIPPED = 25, 250  # drop near-black pixels and any pixel with a saturated channel
MIN_PIXELS = 40          # fewer skin pixels than this -> hidden
MIN_SKIN_FRAC = 0.35     # fraction of the patch that must be skin (a covered neck scores ~0)
MIN_INSIDE_FRAC = 0.8    # fraction of the patch that must be inside the frame
MIN_CHEEK_RATIO = 0.4    # a cheek smaller than this x the other cheek is turned away -> hidden

# face tracking
CROP_SCALE = 3.0  # the crop around the face is this many face-sizes wide
MIN_CROP = 192    # pixels
MAX_MISSES = 5    # frames without the face before searching the whole frame again
MAX_FACES = 2     # faces MediaPipe may return per crop (we keep the one nearest the last position)
SEARCH_TILES = (1.0, 0.6, 0.33)  # whole-frame search: tile sides as a fraction of the frame's short side
SEARCH_EVERY = 10  # while no face is followed, search the whole frame every this many frames
DETECT_CONF = PRESENCE_CONF = TRACK_CONF = 0.5  # MediaPipe's thresholds for finding / keeping a face
# (0.3 and an instant re-detect on a miss were tried on 48 dropped DFDC videos: they recovered 2 and
# added tracker jumps. Those videos drop because the face is side-on, not because tracking is strict.)


def ensure_model(path=MODEL_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading the MediaPipe face landmarker to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


@dataclass
class FrameRegions:
    rgb: np.ndarray        # [R, 3] float32 mean skin colour per region, NaN where hidden
    valid: np.ndarray      # [R] bool
    skin_frac: np.ndarray  # [R] float32 fraction of each patch that counted as skin
    polygons: list         # R arrays of [K, 2] pixel coords (for drawing), or None if no face
    n_faces: int           # faces MediaPipe returned in this frame's search area
    face: np.ndarray       # [3] float32 followed face's centre x, y and size in pixels; NaN if none


def region_polygons(pts):
    """pts: [478, 2] landmark pixel coords -> list of [K, 2] float polygons in REGIONS order."""
    polys = {name: pts[idx] for name, idx in POLYGONS.items()}
    top, chin = pts[FOREHEAD_TOP], pts[CHIN_TIP]
    down = chin - top
    h = np.linalg.norm(down) + 1e-6
    down = down / h
    across = np.array([-down[1], down[0]])
    w = np.linalg.norm(pts[FACE_LEFT_EDGE] - pts[FACE_RIGHT_EDGE])
    a = chin + NECK_GAP * h * down
    b = a + NECK_LEN * h * down
    hw = NECK_HALF_WIDTH * w * across
    polys["neck"] = np.stack([a - hw, a + hw, b + hw, b - hw])
    return [polys[r] for r in REGIONS]


def _patch(frame, poly):
    """Pixels of one polygon: (rgb [n,3], YCrCb [n,3], polygon area, pixels inside the frame)."""
    H, W = frame.shape[:2]
    p = np.round(poly).astype(np.int32)
    area = abs(cv2.contourArea(p))
    x0, y0 = np.maximum(p.min(0), 0)
    x1, y1 = np.minimum(p.max(0) + 1, [W, H])
    if x1 <= x0 or y1 <= y0 or area < 1:
        return None, None, area, 0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [p - [x0, y0]], 1)
    m = m.astype(bool)
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    return crop[m], cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)[m], area, int(m.sum())


def region_means(frame, polys):
    """frame: HxWx3 uint8 RGB. polys: from region_polygons. Returns (rgb [R,3], valid [R], skin [R])."""
    patches = [_patch(frame, p) for p in polys]

    def usable(rgb, yc):
        return (yc[:, 0] >= DARK) & (rgb.max(axis=1) < CLIPPED)

    ref = [yc[usable(rgb, yc)][:, 1:] for name, (rgb, yc, _, _) in zip(REGIONS, patches)
           if name in SKIN_REF and rgb is not None]
    ref = np.concatenate(ref).astype(np.float32) if ref else np.zeros((0, 2), np.float32)

    R = len(REGIONS)
    out = np.full((R, 3), np.nan, np.float32)
    valid = np.zeros(R, bool)
    skin_frac = np.zeros(R, np.float32)
    if len(ref) < MIN_PIXELS:
        return out, valid, skin_frac
    med = np.median(ref, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(ref - med), axis=0), SKIN_MIN_SPREAD)

    for i, (rgb, yc, area, inside) in enumerate(patches):
        if rgb is None:
            continue
        skin = usable(rgb, yc) & (np.abs(yc[:, 1:] - med) <= SKIN_K * spread).all(axis=1)
        n = int(skin.sum())
        skin_frac[i] = n / max(inside, 1)
        if n >= MIN_PIXELS and skin_frac[i] >= MIN_SKIN_FRAC and inside >= MIN_INSIDE_FRAC * area:
            out[i] = rgb[skin].mean(axis=0, dtype=np.float64)
            valid[i] = True

    # a cheek much smaller than the other one is turned away from the camera
    li, ri = REGIONS.index("left_cheek"), REGIONS.index("right_cheek")
    la, ra = patches[li][2], patches[ri][2]
    for i, mine, other in ((li, la, ra), (ri, ra, la)):
        if mine < MIN_CHEEK_RATIO * other:
            out[i], valid[i] = np.nan, False
    return out, valid, skin_frac


def landmarker(video=False, num_faces=MAX_FACES, model_path=None):
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model(model_path or MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=num_faces, min_face_detection_confidence=DETECT_CONF,
        min_face_presence_confidence=PRESENCE_CONF, min_tracking_confidence=TRACK_CONF)
    return vision.FaceLandmarker.create_from_options(opts)


def _faces_in(frame, lm, box, t_ms=None):
    """Landmarks [478, 2] (frame pixels) of every face MediaPipe finds inside box = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[y0:y1, x0:x1]))
    res = lm.detect(img) if t_ms is None else lm.detect_for_video(img, t_ms)
    return [np.array([(p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0) for p in f], np.float32)
            for f in res.face_landmarks]


def face_size(pts):
    return float((pts.max(0) - pts.min(0)).max())


def find_faces(frame, lm, stop_when_found=True):
    """Every face in a frame, small ones included. MediaPipe's detector sees the image at 128 px, so
    a face far from a 1080p camera is too small for it; searching overlapping tiles fixes that.
    Goes coarse to fine; stop_when_found=True stops at the first level that finds a face.
    lm: an IMAGE-mode landmarker. Returns a list of landmark arrays, biggest face first."""
    H, W = frame.shape[:2]
    found = []
    for frac in SEARCH_TILES:
        s = min(int(min(H, W) * frac), H, W)
        step = max(1, int(s * 2 / 3))
        xs = sorted({*range(0, W - s, step), W - s}) if frac < 1 else [0]
        ys = sorted({*range(0, H - s, step), H - s}) if frac < 1 else [0]
        boxes = [(x, y, x + s, y + s) for y in ys for x in xs] if frac < 1 else [(0, 0, W, H)]
        for x0, y0, x1, y1 in boxes:
            for pts in _faces_in(frame, lm, (x0, y0, x1, y1)):
                lo, hi = pts.min(0), pts.max(0)
                cut = ((lo[0] < x0 + 2 and x0 > 0) or (lo[1] < y0 + 2 and y0 > 0) or
                       (hi[0] > x1 - 2 and x1 < W) or (hi[1] > y1 - 2 and y1 < H))
                if not cut:  # a face cut by the tile edge is found whole in a neighbouring tile
                    found.append(pts)
        if found and stop_when_found:
            break
    faces = []
    for pts in sorted(found, key=face_size, reverse=True):  # drop repeats of the same face
        if all(np.linalg.norm(pts.mean(0) - q.mean(0)) > 0.5 * face_size(q) for q in faces):
            faces.append(pts)
    return faces


class RegionExtractor:
    """One per video stream (MediaPipe VIDEO mode tracks the face from frame to frame).

        ex = RegionExtractor()                    # follow the biggest face
        ex = RegionExtractor(target=(x, y, size)) # follow the face at this spot in the first frame
        fr = ex(frame_rgb, t_ms)                  # t_ms must increase from call to call
    """

    def __init__(self, target=None, model_path=None):
        self.model_path = model_path
        self.landmarker = landmarker(video=True, model_path=model_path)
        self.finder = None  # IMAGE-mode landmarker for whole-frame searches, made when first needed
        self.centre = self.side = None
        if target is not None:
            x, y, size = target
            self.centre = np.array([x, y], np.float32)
            self.side = max(CROP_SCALE * size, MIN_CROP)
        self.misses = 0
        self.frame_i = 0
        self.searched_at = -SEARCH_EVERY
        self.last_t = -1

    def _box(self, H, W):
        if self.centre is None:
            return 0, 0, W, H
        sw, sh = min(int(self.side), W), min(int(self.side), H)
        x0 = int(np.clip(self.centre[0] - sw / 2, 0, W - sw))
        y0 = int(np.clip(self.centre[1] - sh / 2, 0, H - sh))
        return x0, y0, x0 + sw, y0 + sh

    def _follow(self, pts):
        want = max(CROP_SCALE * face_size(pts), MIN_CROP)
        # keep the crop size fixed unless the face size has really changed (MediaPipe's own tracker
        # works in crop coordinates, so a jittering crop would make the landmarks jitter)
        if self.side is None or self.misses > MAX_MISSES or abs(want / self.side - 1) > 0.3:
            self.side = want
        self.centre = pts.mean(0)
        self.misses = 0

    def _finder(self):
        if self.finder is None:
            self.finder = landmarker(video=False, num_faces=4, model_path=self.model_path)
        return self.finder

    def _search(self, frame):
        """No face followed yet, or lost for a while: search the whole frame (every few frames)."""
        if self.frame_i - self.searched_at < SEARCH_EVERY:
            return
        self.searched_at = self.frame_i
        faces = find_faces(frame, self._finder(), stop_when_found=self.centre is None)
        if faces:  # first time: the biggest face; after a loss: the one nearest where it was
            pts = faces[0] if self.centre is None else min(
                faces, key=lambda f: np.linalg.norm(f.mean(0) - self.centre))
            self._follow(pts)

    def landmarks(self, frame, t_ms):
        """Pixel landmarks [478, 2] of the followed face (or None) and the number of faces seen."""
        t_ms = max(int(t_ms), self.last_t + 1)
        self.last_t = t_ms
        self.frame_i += 1
        H, W = frame.shape[:2]
        if self.centre is None or self.misses > MAX_MISSES:
            self._search(frame)
        faces = _faces_in(frame, self.landmarker, self._box(H, W), t_ms)
        if not faces:
            self.misses += 1
            return None, 0
        if self.centre is None:
            pts = max(faces, key=face_size)
        else:
            pts = min(faces, key=lambda f: np.linalg.norm(f.mean(0) - self.centre))
        self._follow(pts)
        return pts, len(faces)

    def __call__(self, frame, t_ms):
        frame = np.ascontiguousarray(frame)
        pts, n = self.landmarks(frame, t_ms)
        if pts is None:
            R = len(REGIONS)
            return FrameRegions(np.full((R, 3), np.nan, np.float32), np.zeros(R, bool),
                                np.zeros(R, np.float32), None, 0, np.full(3, np.nan, np.float32))
        polys = region_polygons(pts)
        rgb, valid, skin = region_means(frame, polys)
        face = np.array([*pts.mean(0), face_size(pts)], np.float32)
        return FrameRegions(rgb, valid, skin, polys, n, face)

    def close(self):
        self.landmarker.close()
        if self.finder is not None:
            self.finder.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


COLOURS_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # one per region, fixed order
COLOURS = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in COLOURS_HEX]  # RGB


def draw_regions(frame, fr, thickness=2):
    """Draws the patches on an RGB frame in place: coloured = used, grey = hidden."""
    if fr.polygons is None:
        return frame
    for i, poly in enumerate(fr.polygons):
        colour = COLOURS[i] if fr.valid[i] else (120, 120, 120)
        cv2.polylines(frame, [np.round(poly).astype(np.int32)], True, colour, thickness, cv2.LINE_AA)
    return frame
