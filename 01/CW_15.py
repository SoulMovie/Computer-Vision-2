import os
import cv2
import shutil

PROJECT_DIR = os.path.dirname(__file__)


IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

OUT_DIR = os.path.join(PROJECT_DIR, "out")
PEOPLE_DIR = os.path.join(OUT_DIR, "people")
NO_PEOPLE_DIR = os.path.join(OUT_DIR, "no_people")


os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)


PROTOTXT_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")


net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)


CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]


PERSON_CLASS_ID = CLASSES.index("person")


CONF_THRESHOLD = 0.5


def detect_persons_on_image(image_bgr):

    (h, w) = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence >= CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            boxes.append((x1, y1, x2, y2))
            confidences.append(confidence)

    return boxes, confidences



allowed_ext = (".jpg", ".jpeg", ".png", ".bmp")

files = os.listdir(IMAGES_DIR)

count_people = 0
count_no_people = 0

for filename in files:
    if not filename.lower().endswith(allowed_ext):
        continue

    in_path = os.path.join(IMAGES_DIR, filename)


    img = cv2.imread(in_path)
    if img is None:
        print("Не вдалося прочитати:", filename)
        continue

    boxes, confidences = detect_persons_on_image(img)
    person_count = len(boxes)
    found = person_count > 0

    if found:

        out_path = os.path.join(PEOPLE_DIR, filename)
        shutil.copy2(in_path, out_path)
        count_people += 1

        boxed = img.copy()

        for (x1, y1, x2, y2), conf in zip(boxes, confidences):
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                boxed,
                f"{conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            boxed,
            f"People: {person_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        boxed_path = os.path.join(PEOPLE_DIR, "boxed_" + filename)
        cv2.imwrite(boxed_path, boxed)

        print(f"[PEOPLE] {filename}  conf={conf:.2f}")

    else:

        out_path = os.path.join(NO_PEOPLE_DIR, filename)
        shutil.copy2(in_path, out_path)
        count_no_people += 1

        print(f"[NO]     {filename}")

print("\nГотово!")
print("З людьми:", count_people)
print("Без людей:", count_no_people)
print("Поріг confidence:", CONF_THRESHOLD)
print("Результати в:", OUT_DIR)