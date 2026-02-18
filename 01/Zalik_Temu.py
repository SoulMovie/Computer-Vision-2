import os
import cv2
import csv
import time
import math
import numpy as np
from ultralytics import YOLO


# ----------------------------
# Config
# ----------------------------
PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_OUT_DIR = os.path.join(OUTPUT_DIR, "videos")
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

USE_WEBCAM = False
INPUT_VIDEO_PATH = os.path.join(PROJECT_DIR, "video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(VIDEO_OUT_DIR, "people_flow_output.mp4")
CSV_PATH = os.path.join(LOG_DIR, "tracks_summary.csv")

MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4
TRACKER = "bytetrack.yaml"

SAVE_VIDEO = True
SHOW_WINDOW = True

# Trajectory settings
MAX_TRAIL = 30
MIN_TRACK_POINTS = 8

# Heatmap settings (lighter overlay)
HEATMAP_DECAY = 0.99
HEATMAP_GAIN = 1.5


# ----------------------------
# Helpers
# ----------------------------
def bbox_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def direction_from_path(path):
    if len(path) < MIN_TRACK_POINTS:
        return "unknown"

    x0, y0 = path[0]
    x1, y1 = path[-1]
    dx = x1 - x0
    dy = y1 - y0

    if abs(dx) < 10 and abs(dy) < 10:
        return "unknown"

    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"

def path_distance_px(path):
    if len(path) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        dist += math.hypot(x1 - x0, y1 - y0)
    return dist


# ----------------------------
# Main
# ----------------------------
def main():
    model = YOLO(MODEL_PATH)

    source = 0 if USE_WEBCAM else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

    # State
    tracks = {}
    finished = set()
    flow_counts = {"up": 0, "down": 0, "left": 0, "right": 0}

    heat = np.zeros((frame_h, frame_w), dtype=np.float32)

    # CSV
    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["track_id", "first_seen", "last_seen", "duration_s", "direction", "distance_px"])

    PERSON_CLASS_ID = 0

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            results = model.track(
                frame,
                conf=CONF_THRESH,
                tracker=TRACKER,
                persist=True,
                verbose=False
            )
            r = results[0]

            heat *= HEATMAP_DECAY

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                conf = boxes.conf.cpu().numpy()

                ids = None
                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy()

                for i in range(len(xyxy)):
                    class_id = int(cls[i])
                    if class_id != PERSON_CLASS_ID:
                        continue

                    x1, y1, x2, y2 = xyxy[i].astype(int)
                    score = float(conf[i])

                    tid = -1
                    if ids is not None:
                        tid = int(ids[i])
                    if tid == -1:
                        continue

                    cx, cy = bbox_center(x1, y1, x2, y2)

                    if tid not in tracks:
                        tracks[tid] = {
                            "path": [],
                            "first_seen": now,
                            "last_seen": now,
                            "direction": "unknown",
                            "counted": False
                        }

                    tr = tracks[tid]
                    tr["last_seen"] = now
                    tr["path"].append((cx, cy))
                    if len(tr["path"]) > MAX_TRAIL:
                        tr["path"] = tr["path"][-MAX_TRAIL:]

                    tr["direction"] = direction_from_path(tr["path"])

                    heat[cy, cx] += HEATMAP_GAIN

                    if (not tr["counted"]) and tr["direction"] in flow_counts:
                        flow_counts[tr["direction"]] += 1
                        tr["counted"] = True

                    label = f"ID {tid} {tr['direction']}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 1)

                    path = tr["path"]
                    if len(path) > 1:
                        for j in range(1, len(path)):
                            cv2.line(frame, path[j - 1], path[j], (255, 0, 0), 1)

            # Heatmap overlay (lighter)
            heat_norm = np.clip(heat / (heat.max() + 1e-6), 0, 1)
            heat_img = (heat_norm * 255).astype(np.uint8)
            heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.85, heat_img, 0.15, 0)

            # FPS
            curr_time = time.time()
            fps_display = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # ---------------- HUD (thicker, readable) ----------------
            hud = frame.copy()

            # background rectangle
            cv2.rectangle(hud, (5, 5), (240, 90), (0, 0, 0), -1)

            # transparency
            frame = cv2.addWeighted(hud, 0.35, frame, 0.65, 0)

            # text (thicker)
            cv2.putText(frame, f"People: {len(tracks)}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame,
                        f"Up:{flow_counts['up']}  Down:{flow_counts['down']}",
                        (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame,
                        f"Left:{flow_counts['left']}  Right:{flow_counts['right']}",
                        (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # FPS on right side
            cv2.putText(frame, f"FPS: {fps_display:.1f}",
                        (frame_w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if SHOW_WINDOW:
                cv2.imshow("People Flow", frame)

            if writer is not None:
                writer.write(frame)

            if SHOW_WINDOW and (cv2.waitKey(1) & 0xFF == ord("q")):
                break

        # Save summary
        for tid, tr in tracks.items():
            duration = max(0.0, tr["last_seen"] - tr["first_seen"])
            dist_px = path_distance_px(tr["path"])
            csv_writer.writerow([
                tid,
                tr["first_seen"],
                tr["last_seen"],
                duration,
                tr["direction"],
                dist_px
            ])

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
