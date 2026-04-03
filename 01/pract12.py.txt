import cv2
import time
import csv
import yt_dlp
from ultralytics import YOLO

YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4
TRACKER = "bytetrack.yaml"
DISTANCE_METERS = 10  # distance between lines in meters

LINE1 = ((800, 400), (1800, 400))
LINE2 = ((200, 700), (1900, 700))

CSV_FILE = "speed_data.csv"

def get_youtube_stream(url):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

def box_overlaps_line(box, line):
    x1, y1, x2, y2 = box
    lx1, ly1 = line[0]
    lx2, ly2 = line[1]
    min_x, max_x = min(lx1, lx2), max(lx1, lx2)
    min_y, max_y = min(ly1, ly2), max(ly1, ly2)
    if x2 < min_x or x1 > max_x:
        return False
    if y2 < min_y or y1 > max_y:
        return False
    return True

source = get_youtube_stream(YOUTUBE_URL)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(source)

seen_id_total = set()
id_cross_time_1 = {}
id_cross_time_2 = {}
id_speed = {}
speed_list = []

# setup CSV file
with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "object_id", "class", "speed_kmh"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=False)
    r = result[0]

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        track_id = boxes.id.cpu().numpy() if boxes.id is not None else None

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            class_id = int(cls[i])
            class_name = model.names[class_id]

            # skip people
            if class_name.lower() == "person":
                continue

            score = conf[i]
            tid = int(track_id[i]) if track_id is not None else -1

            if tid != -1:
                seen_id_total.add(tid)

                # check overlap with LINE1
                if box_overlaps_line((x1, y1, x2, y2), LINE1):
                    if tid not in id_cross_time_1:
                        id_cross_time_1[tid] = time.time()
                        print(f"[LINE1] ID {tid}, Class {class_name}, Score {score:.2f}")

                # check overlap with LINE2
                if box_overlaps_line((x1, y1, x2, y2), LINE2):
                    if tid in id_cross_time_1 and tid not in id_cross_time_2:
                        t2 = time.time()
                        id_cross_time_2[tid] = t2
                        dt = t2 - id_cross_time_1[tid]
                        if dt > 0:
                            speed = (DISTANCE_METERS / dt) * 3.6  # km/h
                            id_speed[tid] = speed
                            speed_list.append(speed)
                            print(f"[LINE2] ID {tid}, Class {class_name}, Speed {speed:.1f} km/h")

                            # append to CSV
                            with open(CSV_FILE, mode="a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([time.time(), tid, class_name, f"{speed:.1f}"])

            speed_text = f"{id_speed[tid]:.1f} km/h" if tid in id_speed else "N/A"
            label = f'{class_name} ID {tid} | {speed_text}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.line(frame, LINE1[0], LINE1[1], (0, 0, 255), 3)
    cv2.line(frame, LINE2[0], LINE2[1], (255, 0, 0), 3)

    total = len(seen_id_total)
    cv2.putText(frame, f'unique objects: {total}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    avg_speed = sum(speed_list) / len(speed_list) if speed_list else 0
    cv2.putText(frame, f'average speed: {avg_speed:.1f} km/h', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    display_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    cv2.imshow('frame', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()