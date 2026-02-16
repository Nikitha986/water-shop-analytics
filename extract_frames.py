import cv2

video = r"C:\Users\shiva\water-shop-analytics\videos\cctv.mp4"
cap = cv2.VideoCapture(video)

count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % 15 == 0:   # every 15 frames
        cv2.imwrite(f"dataset/images/train/frame_{saved}.jpg", frame)
        saved += 1

    count += 1

cap.release()
print("Saved:", saved)
