import cv2
from ultralytics import YOLO
import numpy as np
import os

VIDEO_PATH = r"C:\Users\shiva\water-shop-analytics\videos\cctv.mp4"
CAN_MODEL = r"C:\Users\shiva\water-shop-analytics\water_can_model.pt"
SEG_TEST_MODEL = "yolov8n-seg"

out_path = "debug_output.jpg"

print('Loading can model...')
can_model = YOLO(CAN_MODEL)

cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print('Failed to open video:', VIDEO_PATH)
    raise SystemExit(1)

ret, frame = cap.read()
cap.release()
if not ret or frame is None:
    print('Failed to read frame from video')
    raise SystemExit(1)

print('Running can model on one frame...')
results = can_model.predict(frame, conf=0.03, imgsz=640, verbose=False)
print('Detections:', len(results[0].boxes))

# annotate
annot = frame.copy()
if len(results) > 0 and results[0].boxes is not None:
    for i, box in enumerate(results[0].boxes):
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
        except Exception:
            # fallback if xyxy is array-like
            coords = box.xyxy
            if hasattr(coords, 'cpu'):
                coords = coords.cpu().numpy()
            coords = coords[0]
            x1, y1, x2, y2 = map(int, coords)
        cv2.rectangle(annot, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(annot, 'CAN', (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        masks = getattr(results[0], 'masks', None)
        if masks is not None and getattr(masks, 'data', None) is not None:
            mask_data = masks.data
            if i < len(mask_data):
                mask_i = mask_data[i]
                if hasattr(mask_i, 'cpu'):
                    mask_i = mask_i.cpu().numpy()
                else:
                    mask_i = np.asarray(mask_i)
                if mask_i.max() <= 1:
                    mask_i = (mask_i * 255).astype(np.uint8)
                else:
                    mask_i = mask_i.astype(np.uint8)
                cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cv2.drawContours(annot, cnts, -1, (0,255,255), 2)
        else:
            # Fallback: try to extract object boundary inside bbox using edges
            roi = frame[y1:y2, x1:x2]
            if roi is not None and roi.size != 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                edges = cv2.Canny(blur, 50, 150)
                cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    # pick largest contour
                    c = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(c) > 100:  # ignore tiny noise
                        c[:,0,0] += x1
                        c[:,0,1] += y1
                        cv2.drawContours(annot, [c], -1, (0,255,255), 2)

cv2.imwrite(out_path, annot)
print('Wrote', os.path.abspath(out_path))

# print mask diagnostics
masks = getattr(results[0], 'masks', None)
print('has_masks:', masks is not None)
if masks is not None:
    print('masks.data present:', getattr(masks, 'data', None) is not None)
    try:
        print('masks count:', len(masks.data))
    except Exception as e:
        print('masks info error:', e)
else:
    # Try a segmentation-capable public model to verify mask drawing
    try:
        print('\nNo detections from custom model — trying', SEG_TEST_MODEL)
        seg = YOLO(SEG_TEST_MODEL)
        res2 = seg.predict(frame, conf=0.03, imgsz=640, verbose=False)
        print('Seg-model detections:', len(res2[0].boxes))
        annot2 = frame.copy()
        if len(res2) > 0 and res2[0].boxes is not None:
            for i, box in enumerate(res2[0].boxes):
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                except Exception:
                    coords = box.xyxy
                    if hasattr(coords, 'cpu'):
                        coords = coords.cpu().numpy()
                    coords = coords[0]
                    x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(annot2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                masks2 = getattr(res2[0], 'masks', None)
                if masks2 is not None and getattr(masks2, 'data', None) is not None:
                    md = masks2.data
                    if i < len(md):
                        m_i = md[i]
                        if hasattr(m_i, 'cpu'):
                            m_i = m_i.cpu().numpy()
                        else:
                            m_i = np.asarray(m_i)
                        if m_i.max() <= 1:
                            m_i = (m_i * 255).astype(np.uint8)
                        else:
                            m_i = m_i.astype(np.uint8)
                        cnts, _ = cv2.findContours(m_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            cv2.drawContours(annot2, cnts, -1, (0,255,255), 2)
        seg_out = 'debug_output_seg.jpg'
        cv2.imwrite(seg_out, annot2)
        print('Wrote', os.path.abspath(seg_out))
    except Exception as e:
        print('Seg-model test failed:', e)
