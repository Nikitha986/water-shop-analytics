"""
Person Tracking Module
Handles customer detection and persistent ID tracking
"""
import cv2
from ultralytics import YOLO

class PersonTracker:
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize person tracker"""
        self.model = YOLO(model_path)
    
    def track_persons(self, frame, conf=0.4):
        """
        Track persons in frame with persistent IDs
        Returns: list of {id, bbox, centroid}
        """
        persons = []
        
        results = self.model.track(frame, persist=True, classes=[0], conf=conf, iou=0.5)
        
        if len(results) > 0 and getattr(results[0], 'boxes', None) is not None:
            try:
                boxes_xy = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, person_id in zip(boxes_xy, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    person_info = {
                        'id': int(person_id),
                        'bbox': (x1, y1, x2, y2),
                        'centroid': (cx, cy)
                    }
                    persons.append(person_info)
            except Exception as e:
                print(f"Person tracking error: {e}")
        
        return persons
