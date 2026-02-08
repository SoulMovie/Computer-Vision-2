import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'videos')

USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, '13403360-hd_1920_1080_30fps.mp4')
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO("yolov8n.pt")

CONF_THRESHOLD = 0.55
RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0

BICYCLE_CLASS_ID = 1
CAR_CLASS_ID = 2
MOTORCYCLE_CLASS_ID = 3
BUS_CLASS_ID = 5
TRUCK_CLASS_ID = 7


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    result = model(frame, conf=CONF_THRESHOLD, verbose=False)

    bicycle_count = 0
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == BICYCLE_CLASS_ID:
                bicycle_count += 1
                color = (161, 135, 199)
                text = f'bicicl {conf:.2f}'
            elif cls == CAR_CLASS_ID:
                car_count += 1
                color = (161, 135, 199)
                text = f'car {conf:.2f}'
            elif cls == MOTORCYCLE_CLASS_ID:
                motorcycle_count += 1
                color = (161, 135, 199)
                text = f'motorcycle {conf:.2f}'
            elif cls == BUS_CLASS_ID:
                bus_count += 1
                color = (161, 135, 199)
                text = f'bus {conf:.2f}'
            elif cls == TRUCK_CLASS_ID:
                truck_count += 1
                color = (161, 135, 199)
                text = f'truck {conf:.2f}'

            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    total_machines = bicycle_count + car_count + motorcycle_count + bus_count + truck_count

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 1.0 / dt

    cv2.putText(frame, f'Bicycles: {bicycle_count}', (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (161, 135, 199), 2)
    cv2.putText(frame, f'Cars: {car_count}', (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (161, 135, 199), 2)
    cv2.putText(frame, f'Motorcycles: {motorcycle_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (161, 135, 199), 2)
    cv2.putText(frame, f'Buses: {bus_count}', (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (161, 135, 199), 2)
    cv2.putText(frame, f'Trucks: {truck_count}', (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (161, 135, 199), 2)

    cv2.putText(frame, f'Total: {total_machines}', (20, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (199, 117, 174), 2)
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 520),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (199, 117, 174), 2)

    cv2.imshow('YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()