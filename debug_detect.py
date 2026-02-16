from ultralytics import YOLO
import cv2

VIDEO_PATH = r"C:\Users\shiva\water-shop-analytics\videos\cctv.mp4"
CAN_MODEL = r"C:\Users\shiva\water-shop-analytics\water_can_model.pt"
SEG_TEST_MODEL = "yolov8n-seg"

m = YOLO(CAN_MODEL)
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    print('Failed to read frame')
    raise SystemExit(1)

res = m.predict(frame, conf=0.03, imgsz=640, verbose=False)
print('boxes count:', len(res[0].boxes))
for i, b in enumerate(res[0].boxes):
    try:
        coords = b.xyxy[0]
        if hasattr(coords, 'cpu'):
            coords = coords.cpu().numpy()
        print(i, list(map(int, coords)))
    except Exception as e:
        print('error reading box coords:', e)

masks = getattr(res[0], 'masks', None)
print('has masks:', masks is not None)
if masks is not None:
    print('masks.data present:', getattr(masks, 'data', None) is not None)
    try:
        print('masks count:', len(masks.data))
    except Exception as e:
        print('masks info error:', e)
else:
    try:
        print('\nNo detections from custom model — trying', SEG_TEST_MODEL)
        seg = YOLO(SEG_TEST_MODEL)
        res2 = seg.predict(frame, conf=0.03, imgsz=640, verbose=False)
        print('seg boxes count:', len(res2[0].boxes))
        print('has masks:', getattr(res2[0], 'masks', None) is not None)
    except Exception as e:
        print('seg-model test failed:', e)
