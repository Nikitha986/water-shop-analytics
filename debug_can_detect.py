import cv2
from ultralytics import YOLO
import numpy as np

VIDEO_PATH = r"C:\Users\shiva\water-shop-analytics\videos\cctv.mp4"
CAN_MODEL = r"C:\Users\shiva\water-shop-analytics\water_can_model.pt"
SEG_MODEL = "yolov8n-seg"

print("Loading models...")
can_model = YOLO(CAN_MODEL)
seg_model = YOLO(SEG_MODEL)

cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Error opening video")
    exit()

# Skip to a few frames in
for _ in range(10):
    cap.read()

ret, frame = cap.read()
cap.release()
if not ret:
    print("Failed to read frame")
    exit()

print("\n=== CUSTOM MODEL ===")
cres = can_model.predict(frame, conf=0.03, imgsz=640, verbose=False)
print(f"Detections: {len(cres[0].boxes) if cres[0].boxes else 0}")
has_masks = cres[0].masks is not None
print(f"Has masks: {has_masks}")

print("\n=== SEGMENTATION MODEL (FALLBACK) ===")
sres = seg_model.predict(frame, conf=0.03, imgsz=640, verbose=False)
print(f"Detections: {len(sres[0].boxes) if sres[0].boxes else 0}")
has_seg_masks = sres[0].masks is not None
print(f"Has masks: {has_seg_masks}")

# Show areas of detected cans
if sres[0].boxes:
    print("\n=== DETECTED CAN DETAILS (SEG MODEL) ===")
    for i, box in enumerate(sres[0].boxes):
        coords = box.xyxy[0]
        if hasattr(coords, 'cpu'):
            coords = coords.cpu().numpy()
        x1, y1, x2, y2 = map(int, coords)
        w = x2 - x1
        h = y2 - y1
        area = w * h
        print(f"Can {i}: bbox=({x1},{y1},{x2},{y2}) area={area}")
        
        # Check mask contour area
        if has_seg_masks and i < len(sres[0].masks.data):
            mask_i = sres[0].masks.data[i]
            if hasattr(mask_i, 'cpu'):
                mask_i = mask_i.cpu().numpy()
            else:
                mask_i = np.asarray(mask_i)
            if mask_i.max() <= 1:
                mask_i = (mask_i*255).astype(np.uint8)
            else:
                mask_i = mask_i.astype(np.uint8)
            
            cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                contour_area = cv2.contourArea(c)
                print(f"  → Contour area: {contour_area}")
                mx, my, mw, mh = cv2.boundingRect(c)
                print(f"  → Contour bbox: ({mx},{my},{mx+mw},{my+mh})")

print("\n=== COUNTER ZONE ===")
COUNTER_ZONE = (780, 120, 1280, 720)
cx1, cy1, cx2, cy2 = COUNTER_ZONE
margin = 100
print(f"Zone: x=[{cx1-margin} to {cx2+margin}], y=[{cy1-margin} to {cy2+margin}]")
