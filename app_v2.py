import cv2
import numpy as np
import time
from ultralytics import YOLO

VIDEO_SOURCE = r"C:\Users\shiva\water-shop-analytics\videos\cctv.mp4"

PIPE_ZONE = (290, 210, 520, 430)

MIN_VISIT_TIME = 2
EXIT_DELAY = 3
DIST_THRESHOLD = 170   # increased for reliability

visit_count = 0
customer_active = False
visit_start_time = 0
last_seen_time = 0
last_valid_box = None
counted_this_visit = False

fill_frames = 0
fill_detected = False
prev_edges = None

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

print("Processing CCTV footage...")

def near_pipe(box):
    if box is None:
        return False

    x1,y1,x2,y2 = box
    zx1,zy1,zx2,zy2 = PIPE_ZONE

    pcx = (x1+x2)//2
    pcy = (y1+y2)//2
    zcx = (zx1+zx2)//2
    zcy = (zy1+zy2)//2

    dist = ((pcx-zcx)**2 + (pcy-zcy)**2)**0.5
    return dist < DIST_THRESHOLD


def detect_fill_motion(frame, prev):
    x1,y1,x2,y2 = PIPE_ZONE
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gray,40,120)

    motion=False
    if prev is not None:
        diff=cv2.absdiff(edges,prev)
        motion=np.sum(diff)>20000

    return motion, edges


FRAME_SKIP = 2
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame,(960,540))
    now = time.time()

    # ---------------- PERSON DETECTION ----------------
    results = model(frame, conf=0.35, classes=[0])

    person_present = False
    best_box = None
    best_area = 0

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            area = (x2-x1)*(y2-y1)
            if area > best_area:
                best_area = area
                best_box = (x1,y1,x2,y2)

    if best_box:
        person_present = True
        last_valid_box = best_box   # store last valid position

        x1,y1,x2,y2 = best_box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,"CUSTOMER",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # ---------------- VISIT SESSION ----------------
    if person_present and not customer_active:
        customer_active = True
        visit_start_time = now
        counted_this_visit = False
        fill_frames = 0
        fill_detected = False

    if person_present:
        last_seen_time = now

    # PERSON LEFT
    if customer_active and not person_present:
        if now - last_seen_time > EXIT_DELAY:

            visit_duration = now - visit_start_time

            if (visit_duration > MIN_VISIT_TIME
                and near_pipe(last_valid_box)
                and not counted_this_visit):

                visit_count += 1
                counted_this_visit = True
                print(f"✅ Visit counted → {visit_count}")

            customer_active = False

    # ---------------- FILL DETECTION ----------------
    if customer_active:
        motion, edges = detect_fill_motion(frame, prev_edges)
        prev_edges = edges

        if motion:
            fill_frames += 1
        else:
            fill_frames = 0

        if fill_frames > 6:
            fill_detected = True

    # ---------------- STATUS ----------------
    status=""
    payment=""

    if customer_active:
        if fill_detected:
            status="PAID"
            payment="UPI"
        else:
            status="PAID"
            payment="CASH"

    # ---------------- DISPLAY ----------------
    cv2.putText(frame,f"Visits: {visit_count}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    if customer_active:
        cv2.putText(frame,f"Status: {status}",(20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        cv2.putText(frame,f"Payment: {payment}",(20,115),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Water Shop Analytics",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nTotal Visits:", visit_count)
