import cv2
import time
import csv
import json
import argparse
import numpy as np
from collections import defaultdict, deque
from deepface import DeepFace

EMO_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def clamp_bbox(x, y, w, h, W, H, pad=0):
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2 * pad)
    h = min(H - y, h + 2 * pad)
    return x, y, w, h

def bbox_center(b):
    x, y, w, h = b
    return (x + w * 0.5, y + h * 0.5)

def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class FPS:
    def __init__(self):
        self.t = time.time()
        self.fps = 0.0

    def tick(self):
        now = time.time()
        dt = now - self.t
        self.t = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps


class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "face_id", "emotion", "dominant_score", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "scores_json"])
        self.file.flush()

    def log(self, face_id, emo, emo_score, bbox, scores_dict):
        x, y, w, h = bbox
        self.writer.writerow([
            now_iso(),
            face_id,
            emo,
            float(emo_score),
            int(x), int(y), int(w), int(h),
            json.dumps(scores_dict, ensure_ascii=False)
        ])
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except:
            pass


class FaceTracker:
    def __init__(self, max_lost=10, max_dist_px=120):
        self.next_id = 1
        self.tracks = {}
        self.max_lost = max_lost
        self.max_dist2 = max_dist_px * max_dist_px

    def update(self, detections):
        det_centers = [bbox_center(b) for b in detections]
        assigned = set()

        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            best_j = None
            best_d2 = None

            for j, c in enumerate(det_centers):
                if j in assigned:
                    continue
                d2 = dist2(track["center"], c)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_j = j

            if best_j is not None and best_d2 is not None and best_d2 <= self.max_dist2:
                bbox = detections[best_j]
                self.tracks[tid] = {"bbox": bbox, "center": det_centers[best_j], "lost": 0, "last_seen": time.time()}
                assigned.add(best_j)
            else:
                track["lost"] += 1
                self.tracks[tid] = track

        for j, bbox in enumerate(detections):
            if j in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": bbox, "center": det_centers[j], "lost": 0, "last_seen": time.time()}

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["lost"] > self.max_lost:
                del self.tracks[tid]

        return self.tracks


class EmotionEngine:
    def __init__(self, analyze_every=6):
        self.analyze_every = analyze_every
        self.frame_idx = 0
        self.cache = {}

    def analyze_face(self, face_crop):
        try_backends = ["skip", "opencv"]
        last_err = None

        for backend in try_backends:
            try:
                r = DeepFace.analyze(
                    img_path=face_crop,
                    actions=["emotion"],
                    detector_backend=backend,
                    enforce_detection=False
                )
                if isinstance(r, list):
                    r = r[0]
                scores = r["emotion"]
                dominant = max(scores, key=scores.get)
                return {"dominant": dominant, "dominant_score": float(scores[dominant]), "scores": {k: float(scores.get(k, 0.0)) for k in scores.keys()}}
            except Exception as e:
                last_err = e

        raise last_err

    def step(self, tracks, frame_bgr, pad=10):
        self.frame_idx += 1
        if self.frame_idx % self.analyze_every != 0:
            return self.cache

        H, W = frame_bgr.shape[:2]
        results = {}

        for face_id, t in tracks.items():
            x, y, w, h = t["bbox"]
            x, y, w, h = clamp_bbox(x, y, w, h, W, H, pad=pad)

            crop = frame_bgr[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            try:
                emo = self.analyze_face(crop)
                results[face_id] = emo
            except Exception:
                if face_id in self.cache:
                    results[face_id] = self.cache[face_id]

        for fid, emo in results.items():
            self.cache[fid] = emo

        return self.cache


class EmotionHeatmap:
    def __init__(self, decay=0.94):
        self.decay = decay
        self.map = None

    def ensure(self, H, W):
        if self.map is None or self.map.shape != (H, W):
            self.map = np.zeros((H, W), dtype=np.float32)

    def add_blob(self, H, W, cx, cy, strength=1.0, radius=60):
        self.ensure(H, W)
        self.map *= self.decay

        x0 = int(max(0, cx - radius))
        x1 = int(min(W - 1, cx + radius))
        y0 = int(max(0, cy - radius))
        y1 = int(min(H - 1, cy + radius))

        if x1 <= x0 or y1 <= y0:
            return

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        sigma2 = (radius * 0.55) ** 2
        blob = np.exp(-d2 / (2.0 * sigma2)) * float(strength)

        self.map[y0:y1, x0:x1] += blob

    def render_overlay(self, frame_bgr, alpha=0.35):
        if self.map is None:
            return frame_bgr

        m = self.map.copy()
        m = np.clip(m, 0, np.percentile(m, 99) + 1e-6)

        if m.max() > 0:
            m = (m / m.max() * 255.0).astype(np.uint8)
        else:
            m = m.astype(np.uint8)

        heat = cv2.applyColorMap(m, cv2.COLORMAP_INFERNO)
        return cv2.addWeighted(frame_bgr, 1.0, heat, alpha, 0)


class RollingSeries:
    def __init__(self, window=600):
        self.window = window
        self.series = {k: deque(maxlen=window) for k in EMO_KEYS}
        self.ticks = deque(maxlen=window)

    def push(self, counts_dict):
        self.ticks.append((self.ticks[-1] + 1) if self.ticks else 1)
        for k in EMO_KEYS:
            self.series[k].append(int(counts_dict.get(k, 0)))

    def export_png(self, path="emotion_plot.png", title="Emotion counts per tick"):
        import matplotlib.pyplot as plt

        xs = list(range(len(self.ticks)))
        plt.figure()
        for k in EMO_KEYS:
            plt.plot(xs, list(self.series[k]), label=k)

        plt.title(title)
        plt.xlabel("tick")
        plt.ylabel("count")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path


def parse_args():
    p = argparse.ArgumentParser("Emotion Detection (RetinaFace + Tracking + Heatmap + CSV + PNG Plot)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--analyze-every", type=int, default=6, help="Analyze emotions every N frames")
    p.add_argument("--heatmap", action="store_true", help="Enable heatmap overlay")
    p.add_argument("--csv", type=str, default="emotion_log.csv", help="CSV output path")
    p.add_argument("--plot-png", type=str, default="emotion_plot.png", help="Final PNG output path")
    p.add_argument("--series-window", type=int, default=600, help="Ticks stored for final plot")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = FPS()
    tracker = FaceTracker(max_lost=12, max_dist_px=140)
    emo_engine = EmotionEngine(analyze_every=args.analyze_every)
    heatmap = EmotionHeatmap(decay=0.95)
    logger = CSVLogger(args.csv)
    rolling = RollingSeries(window=args.series_window)

    global_counts = defaultdict(int)
    per_face_counts = defaultdict(lambda: defaultdict(int))

    show_heatmap = args.heatmap
    show_hud = True

    print("Controls: q quit | h hud | m heatmap")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (args.width, args.height))
            H, W = frame.shape[:2]

            bboxes = []
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="retinaface",
                    enforce_detection=False,
                    align=False
                )
                for f in faces:
                    area = f.get("facial_area", {}) or {}
                    x = int(area.get("x", 0))
                    y = int(area.get("y", 0))
                    w = int(area.get("w", 0))
                    h = int(area.get("h", 0))
                    if w > 0 and h > 0:
                        bboxes.append((x, y, w, h))
            except Exception:
                bboxes = []

            tracks = tracker.update(bboxes)
            emotions_by_id = emo_engine.step(tracks, frame, pad=12)
            tick_counts = defaultdict(int)

            overlay = frame.copy()

            for face_id, t in tracks.items():
                x, y, w, h = t["bbox"]
                x, y, w, h = clamp_bbox(x, y, w, h, W, H, pad=0)

                cv2.rectangle(overlay, (x, y), (x + w, y + h), (240, 240, 240), 2)

                emo = emotions_by_id.get(face_id)
                if emo:
                    dom = emo["dominant"]
                    dom_score = emo["dominant_score"]

                    label = f"#{face_id} {dom} ({dom_score:.0f}%)"
                    cv2.putText(
                        overlay, label, (x, max(18, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (230, 230, 230), 2, cv2.LINE_AA
                    )

                    global_counts[dom] += 1
                    per_face_counts[face_id][dom] += 1
                    tick_counts[dom] += 1

                    logger.log(face_id, dom, dom_score, (x, y, w, h), emo["scores"])

                    if show_heatmap:
                        cx, cy = bbox_center((x, y, w, h))
                        strength = max(0.1, min(1.0, dom_score / 100.0))
                        heatmap.add_blob(H, W, cx, cy, strength=strength, radius=70)
                else:
                    cv2.putText(
                        overlay, f"#{face_id} ...", (x, max(18, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (150, 150, 150), 2, cv2.LINE_AA
                    )

            rolling.push(tick_counts)

            if show_heatmap:
                overlay = heatmap.render_overlay(overlay, alpha=0.33)

            f = fps.tick()
            if show_hud:
                top = [f"Faces: {len(tracks)}", f"FPS: {f:.1f}", f"Analyze every: {args.analyze_every} frames", f"Heatmap: {'on' if show_heatmap else 'off'}", "Keys: q quit | h hud | m heatmap" ]
                y0 = 22
                for line in top:
                    cv2.putText(
                        overlay, line, (12, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (210, 210, 210), 2, cv2.LINE_AA
                    )
                    y0 += 20

            cv2.imshow("Emotion Detector (RetinaFace)", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                show_hud = not show_hud
            if key == ord("m"):
                show_heatmap = not show_heatmap

    finally:
        logger.close()
        cap.release()
        cv2.destroyAllWindows()

        png_path = rolling.export_png(args.plot_png, title="Emotion counts per tick (rolling)")
        print(f"\nPNG saved: {png_path}")
        print(f"CSV saved: {args.csv}")

        print("\n=== Summary (global counts) ===")
        for k in EMO_KEYS:
            print(f"{k:>8}: {global_counts.get(k, 0)}")


if __name__ == "__main__":
    main()