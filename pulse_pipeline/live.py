"""
Live pulse from a webcam: the same regions and pulse maths as training, on a rolling 10 s buffer.

    py -3.12 -m pulse_pipeline.live                       # webcam 0: regions, pulse, heart rate
    py -3.12 -m pulse_pipeline.live --camera 1 --lock     # another camera; lock exposure/white balance
    py -3.12 -m pulse_pipeline.live --selftest clip.mp4   # prove live == offline on a video file

In the demo app:

    lp = LivePulse()
    fr = lp.push(frame_rgb)    # every frame; returns the regions (for drawing)
    w = lp.window()            # None until 10 s are buffered, else
                               # dict(X [5, 300], mask [5], hr_bpm, snr_db, agree, fps)
    tokens = pulse_model.data.make_tokens(w["X"], w["mask"], cfg.normalize, cfg.xcorr_lags)

Frames are used as delivered by the camera and never saved or JPEG-compressed. The camera is asked
for uncompressed YUY2; many webcams send JPEG-compressed MJPG at high resolutions instead, which
hurts the pulse, so the format the camera actually delivers is printed at start-up.
"""
import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

from .pulse import FPS, WINDOW_S, heart_rate, make_window, recording_windows, region_agreement, window_len
from .regions import COLOURS, REGIONS, RegionExtractor, draw_regions


class LivePulse:
    """Rolling buffer of per-frame region colours with their capture times."""

    def __init__(self, seconds=WINDOW_S, fps=FPS, target=None):
        self.ex = RegionExtractor(target=target)
        self.fps, self.seconds = fps, seconds
        self.span = (window_len(fps, seconds) - 1) / fps
        self.t, self.rgb = deque(), deque()
        self.t0 = None

    def push(self, frame_rgb, t=None):
        """Adds one frame. t: capture time in seconds (default: now). Returns the FrameRegions."""
        t = time.perf_counter() if t is None else t
        if self.t0 is None:
            self.t0 = t
        t -= self.t0
        fr = self.ex(frame_rgb, t * 1000.0)
        self.t.append(t)
        self.rgb.append(fr.rgb)
        while self.t[-1] - self.t[0] > self.span + 1.0:  # keep 1 s spare for the edge interpolation
            self.t.popleft()
            self.rgb.popleft()
        return fr

    def filled(self):
        """Fraction of the 10 s window collected so far."""
        return min(1.0, (self.t[-1] - self.t[0]) / self.span) if len(self.t) > 1 else 0.0

    def fps_measured(self):
        return (len(self.t) - 1) / (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def window(self):
        """The latest 10 s as model input, or None if not enough is buffered or visible yet."""
        if self.filled() < 1.0:
            return None
        t, rgb = np.array(self.t), np.array(self.rgb)
        w = make_window(t, rgb, t[-1] - self.span, self.fps, self.seconds)
        if w is None:
            return None
        X, mask = w
        hr, snr = heart_rate(X, mask, self.fps)
        return dict(X=X, mask=mask, hr_bpm=hr, snr_db=snr, agree=region_agreement(X, mask),
                    fps=self.fps_measured(), t_end=t[-1])

    def close(self):
        self.ex.close()


def fourcc_str(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def open_camera(index, width, height, fps):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))  # uncompressed, no JPEG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        sys.exit(f"could not open camera {index}")
    fmt = fourcc_str(cap)
    print(f"camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"{fmt} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps requested")
    if fmt.upper() in ("MJPG", "JPEG"):
        print("WARNING: the camera sends JPEG-compressed frames; try --width 640 --height 480 for raw YUY2")
    return cap


def lock_camera(cap):
    """Turns off auto exposure and auto white balance (they add slow colour drifts). Not every
    camera/driver obeys; the values read back are printed."""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual, 0.75 = auto
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    print(f"locked? auto exposure = {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}, "
          f"auto white balance = {cap.get(cv2.CAP_PROP_AUTO_WB)}")


def draw_panel(view, lp, w):
    """Heart rate, buffer state and the latest face-average pulse, drawn onto the preview."""
    H, W = view.shape[:2]
    ink = (240, 239, 236)
    y = 28
    if w is None:
        lines = [f"collecting... {lp.filled():.0%} of {lp.seconds:.0f} s   ({lp.fps_measured():.1f} fps)"]
    else:
        lines = [f"heart rate {w['hr_bpm']:.0f} bpm   SNR {w['snr_db']:.1f} dB   "
                 f"region agreement {w['agree']:.2f}   ({w['fps']:.1f} fps)",
                 "regions: " + "  ".join(n.replace("_", " ") + ("" if m else " (hidden)")
                                         for n, m in zip(REGIONS, w["mask"]))]
    for line in lines:
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, 1, cv2.LINE_AA)
        y += 24
    if w is not None:
        x = w["X"][w["mask"]]
        x = (x / (x.std(axis=1, keepdims=True) + 1e-12)).mean(0)
        h0, h1 = H - 90, H - 10
        xs = np.linspace(10, W - 10, len(x))
        ys = (h0 + h1) / 2 - np.clip(x / 3, -1, 1) * (h1 - h0) / 2
        cv2.rectangle(view, (5, h0 - 5), (W - 5, h1 + 5), (26, 26, 25), -1)
        cv2.polylines(view, [np.stack([xs, ys], 1).astype(np.int32)], False, COLOURS[0][::-1], 2, cv2.LINE_AA)


def run_camera(a):
    cap = open_camera(a.camera, a.width, a.height, a.fps)
    lp = LivePulse()
    locked = not a.lock
    w, last = None, 0.0
    print("press q or Esc to quit")
    try:
        while True:
            ok, bgr = cap.read()
            t = time.perf_counter()
            if not ok:
                break
            if not locked and lp.t and lp.t[-1] > 2.0:  # let auto exposure settle first, then freeze it
                lock_camera(cap)
                locked = True
            fr = lp.push(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)
            if t - last > 0.5:  # a new window twice a second is plenty for the display
                w, last = lp.window(), t
            view = cv2.cvtColor(draw_regions(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fr), cv2.COLOR_RGB2BGR)
            view = cv2.flip(view, 1)  # mirror the preview only, never the frames we measure
            draw_panel(view, lp, w)
            cv2.imshow("PulseGuard live pulse", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        lp.close()


def selftest(path, jitter_ms=8.0, drop=0.03, seed=0):
    """Replays a video as if it were a webcam and checks the live buffer gives the offline answer.
    1. exact replay: every live window must equal the offline window at the same start time.
    2. webcam-like replay: timestamps jittered and a few frames dropped; heart rates must stay close."""
    from .extract import extract_traces

    off = extract_traces(path)
    t_off = off["t"]
    print(f"{path}: {len(t_off)} frames at {off['fps_native']:.2f} fps")

    def replay(times, keep):
        lp, out = LivePulse(), []
        next_at = lp.span
        cap = cv2.VideoCapture(path)
        for t, k in zip(times, keep):
            ok, bgr = cap.read()
            if not ok:
                break
            if not k:
                continue
            lp.push(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)
            if lp.t[-1] >= next_at:  # emit a window every 2 s, like the offline stride
                w = lp.window()
                if w is not None:
                    out.append(w)
                next_at += 2.0
        cap.release()
        lp.close()
        return out

    live = replay(t_off, np.ones(len(t_off), bool))
    worst = 0.0
    for w in live:
        ref = make_window(t_off, off["rgb"], w["t_end"] - (window_len() - 1) / FPS)
        same_mask = ref is not None and (ref[1] == w["mask"]).all()
        worst = max(worst, np.abs(ref[0] - w["X"]).max() if same_mask else np.inf)
    print(f"1. exact replay: {len(live)} live windows, max |live - offline| = {worst:.2e} "
          f"-> {'PASS' if worst < 1e-5 else 'FAIL'}")

    rng = np.random.default_rng(seed)
    times = t_off + rng.normal(0, jitter_ms / 1000, len(t_off))
    times = np.maximum.accumulate(times + np.arange(len(times)) * 1e-6)
    keep = rng.random(len(t_off)) >= drop
    jit = replay(times, keep)
    starts, X, mask = recording_windows(t_off, off["rgb"])
    hr_off = {round(s): heart_rate(x, m)[0] for s, x, m in zip(starts, X, mask)}
    diffs = [abs(w["hr_bpm"] - hr_off[round(w["t_end"] - (window_len() - 1) / FPS)])
             for w in jit if round(w["t_end"] - (window_len() - 1) / FPS) in hr_off]
    print(f"2. webcam-like replay ({jitter_ms:.0f} ms jitter, {drop:.0%} frames dropped): "
          f"{len(diffs)} windows, median |HR live - HR offline| = {np.median(diffs):.1f} bpm, "
          f"max {np.max(diffs):.1f} bpm")


def main():
    p = argparse.ArgumentParser(description="Live pulse from a webcam (rolling 10 s buffer).")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--lock", action="store_true", help="lock exposure and white balance after 2 s")
    p.add_argument("--selftest", metavar="VIDEO", help="replay a video file instead of a camera")
    a = p.parse_args()
    if a.selftest:
        selftest(a.selftest)
    else:
        run_camera(a)


if __name__ == "__main__":
    main()
