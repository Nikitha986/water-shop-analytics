import cv2

def draw_entry_line(frame, y):
    cv2.line(frame, (0, y), (frame.shape[1], y), (0, 0, 255), 2)

def draw_count(frame, count):
    cv2.putText(frame, f"Customers: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

def draw_box(frame, x1, y1, x2, y2, track_id):
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
