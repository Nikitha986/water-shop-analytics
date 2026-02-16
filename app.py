# import cv2
# from ultralytics import YOLO
# from datetime import datetime
# import numpy as np
# from collections import deque

# # ==============================
# # CONFIGURATION
# # ==============================

# VIDEO_PATH = r"C:\Users\shiva\water-shop-analytics\videos\cctv.mp4"
# PERSON_MODEL = "yolov8n.pt"
# CAN_MODEL = r"C:\Users\shiva\water-shop-analytics\water_can_model.pt"
# SEG_MODEL = "yolov8n-seg"

# ENTRY_LINE_Y = 320
# EXIT_Y_THRESHOLD = 220
# COUNTER_ZONE = (950, 100, 1280, 750)  # right side where counter actually is
# QR_ZONE = (960, 120, 1280, 500)

# # thresholds
# CAN_CONFIDENCE = 0.03
# CAN_MIN_SIZE = 40
# CAN_COOLDOWN = 5
# RECENT_CAN_DIST = 200  # increased to keep same can visible
# PERSON_CAN_ASSOC_DIST = 400  # increased to match cans further away
# CAN_MIN_AREA = 1000  # moderate threshold to avoid noise

# # dedupe queues
# recent_can_positions = deque(maxlen=100)

# # ==============================
# # LOAD MODELS
# # ==============================
# print("Loading models...")
# person_model = YOLO(PERSON_MODEL)
# can_model = YOLO(CAN_MODEL)
# seg_model = YOLO(SEG_MODEL)

# # ==============================
# # VIDEO CAPTURE
# # ==============================
# cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
# if not cap.isOpened():
#     print("Error opening video")
#     raise SystemExit(1)

# # ==============================
# # SIMPLE ANALYTICS STORE
# # ==============================
# customer_count = 0
# visit_state = {}
# payments = {"CASH": 0, "UPI": 0, "UNPAID": 0}
# hourly_visits = {}
# can_demand = {"20L": 0}
# last_can_time = None

# print("System running... Press ESC to stop.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     now = datetime.now()

#     # PERSON TRACKING (use track to get persistent IDs)
#     pres = person_model.track(frame, persist=True, classes=[0], conf=0.4, iou=0.5)
#     person_boxes = []  # list of (id, (x1,y1,x2,y2), centroid)
#     if len(pres) > 0 and getattr(pres[0], 'boxes', None) is not None:
#         try:
#             boxes_xy = pres[0].boxes.xyxy.cpu().numpy()
#             ids = pres[0].boxes.id.cpu().numpy().astype(int)
#             for box, tid in zip(boxes_xy, ids):
#                 x1, y1, x2, y2 = map(int, box)
#                 cx = (x1 + x2)//2
#                 cy = (y1 + y2)//2
#                 person_boxes.append((int(tid), (x1,y1,x2,y2), (cx,cy)))
#                 cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
#                 cv2.putText(frame, f"ID {tid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#         except Exception:
#             pass

#     # CAN DETECTION: try custom model, fallback to seg model
#     cres = can_model.predict(frame, conf=CAN_CONFIDENCE, imgsz=640, verbose=False)
#     use_res = cres
#     if len(cres) == 0 or getattr(cres[0], 'boxes', None) is None or getattr(cres[0], 'masks', None) is None:
#         try:
#             sres = seg_model.predict(frame, conf=CAN_CONFIDENCE, imgsz=640, verbose=False)
#             if len(sres) > 0 and getattr(sres[0], 'boxes', None) is not None:
#                 use_res = sres
#         except Exception:
#             use_res = cres

#     # extract candidate can boxes (prefer mask-derived bboxes)
#     can_candidates = []
#     if len(use_res) > 0 and getattr(use_res[0], 'boxes', None) is not None:
#         for i, box in enumerate(use_res[0].boxes):
#             try:
#                 coords = box.xyxy[0]
#                 if hasattr(coords, 'cpu'):
#                     coords = coords.cpu().numpy()
#             except Exception:
#                 coords = box.xyxy
#                 if hasattr(coords, 'cpu'):
#                     coords = coords.cpu().numpy()
#                 coords = coords[0]
#             x1, y1, x2, y2 = map(int, coords)
#             w = x2 - x1; h = y2 - y1
#             if w < CAN_MIN_SIZE or h < CAN_MIN_SIZE:
#                 continue
#             # prefer mask bbox if available
#             masks = getattr(use_res[0], 'masks', None)
#             bbox_abs = (x1,y1,x2,y2)
#             if masks is not None and getattr(masks, 'data', None) is not None and i < len(masks.data):
#                 mask_i = masks.data[i]
#                 if hasattr(mask_i, 'cpu'):
#                     mask_i = mask_i.cpu().numpy()
#                 else:
#                     mask_i = np.asarray(mask_i)
#                 if mask_i.max() > 0:
#                     if mask_i.max() <= 1:
#                         mask_i = (mask_i*255).astype(np.uint8)
#                     else:
#                         mask_i = mask_i.astype(np.uint8)
#                     cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                     if cnts:
#                         c = max(cnts, key=cv2.contourArea)
#                         area = cv2.contourArea(c)
#                         if area < CAN_MIN_AREA:
#                             continue
#                         mx, my, mw, mh = cv2.boundingRect(c)
#                         if mask_i.shape[0] == frame.shape[0] and mask_i.shape[1] == frame.shape[1]:
#                             bbox_abs = (mx, my, mx+mw, my+mh)
#                         else:
#                             bbox_abs = (x1+mx, y1+my, x1+mx+mw, y1+my+mh)
#             # only consider cans in or near counter zone
#             bx1, by1, bx2, by2 = bbox_abs
#             bcx = (bx1+bx2)//2; bcy = (by1+by2)//2
#             cx1, cy1, cx2, cy2 = COUNTER_ZONE
#             margin = 100
#             if not (cx1-margin <= bcx <= cx2+margin and cy1-margin <= bcy <= cy2+margin):
#                 continue
#             can_candidates.append(bbox_abs)

#     # merge overlapping candidates
#     def iou(a,b):
#         ax1, ay1, ax2, ay2 = a
#         bx1, by1, bx2, by2 = b
#         ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
#         ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
#         if ix2<=ix1 or iy2<=iy1: return 0.0
#         inter = (ix2-ix1)*(iy2-iy1)
#         union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
#         return inter/union if union>0 else 0.0

#     final_boxes = []
#     for cb in can_candidates:
#         merged = False
#         for j, fb in enumerate(final_boxes):
#             if iou(cb, fb) > 0.35:
#                 nx1 = min(cb[0], fb[0]); ny1 = min(cb[1], fb[1])
#                 nx2 = max(cb[2], fb[2]); ny2 = max(cb[3], fb[3])
#                 final_boxes[j] = (nx1, ny1, nx2, ny2)
#                 merged = True; break
#         if not merged:
#             final_boxes.append(cb)

#     # Associate at most one can per person (nearest)
#     # Cans ONLY shown when persons are present
#     used_can_boxes = []
#     if person_boxes:
#         # compute person centroids
#         person_centroids = [(p[0], p[2][0], p[2][1]) for p in person_boxes]
#         assigned = set()
#         for pid, pcx, pcy in person_centroids:
#             best_idx = -1; best_dist = PERSON_CAN_ASSOC_DIST
#             for idx, cb in enumerate(final_boxes):
#                 if idx in assigned: continue
#                 cx = (cb[0]+cb[2])//2; cy = (cb[1]+cb[3])//2
#                 dist = ((pcx-cx)**2 + (pcy-cy)**2)**0.5
#                 if dist < best_dist:
#                     best_dist = dist; best_idx = idx
#             if best_idx >=0:
#                 assigned.add(best_idx); used_can_boxes.append(final_boxes[best_idx])

#     # draw person boxes (already drawn) and draw can boxes
#     for bx in used_can_boxes:
#         x1c, y1c, x2c, y2c = map(int, bx)
#         cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (255,255,0), 2)
#         cv2.putText(frame, "20L CAN", (x1c, y1c-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
#         cx_can = (x1c+x2c)//2; cy_can = (y1c+y2c)//2
#         dup = any(abs(px-cx_can)<RECENT_CAN_DIST and abs(py-cy_can)<RECENT_CAN_DIST for px,py in recent_can_positions)
#         if not dup:
#             recent_can_positions.append((cx_can, cy_can))
#             if last_can_time is None or (now-last_can_time).seconds > CAN_COOLDOWN:
#                 can_demand['20L'] += 1; last_can_time = now

#     # UI overlay
#     cv2.putText(frame,f"Customers: {len(person_boxes)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
#     cv2.putText(frame,f"20L Cans: {can_demand['20L']}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

#     cv2.imshow('Water Shop Analytics', frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release(); cv2.destroyAllWindows()

# print("\n====== DAILY SUMMARY ======")
# print("Total Customers:", len(person_boxes))
# print("Can Demand:", can_demand)
