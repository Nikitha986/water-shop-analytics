from ultralytics import YOLO
import config
from ultralytics import YOLO
from config import CAN_MODEL_PATH, CAN_CLASSES, CAN_CONFIDENCE

can_model = YOLO(CAN_MODEL_PATH)


model = YOLO("yolov8n.pt")

def detect(frame):
    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            if conf < config.CONFIDENCE_THRESHOLD:
                continue
            detections.append((int(x1), int(y1), int(x2), int(y2), int(cls)))

    return detections
def detect_cans(frame):
    results = can_model(frame, conf=CAN_CONFIDENCE)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = CAN_CLASSES.get(cls, "unknown")

            detections.append((label, conf, (x1, y1, x2, y2)))

    return detections
