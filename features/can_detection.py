"""
Can Detection Module
Handles water can detection and size classification
"""
import cv2
import numpy as np
from ultralytics import YOLO

class CanDetector:
    def __init__(self, custom_model_path, seg_model="yolov8n-seg"):
        """Initialize can detector with primary and fallback models"""
        self.custom_model = YOLO(custom_model_path)
        self.seg_model = YOLO(seg_model)
        self.can_sizes = {"5L": (30, 80), "10L": (80, 200), "20L": (200, 400), "30L": (400, 800)}
        
    def detect_cans(self, frame, conf=0.03, zone=None):
        """
        Detect water cans in frame using multiple strategies:
        1. Model-based detection (custom or segmentation)
        2. Color-based fallback (white bottle detection)
        Returns: list of {bbox, size_class, confidence, centroid}
        """
        cans = []
        
        # Try custom model first
        custom_results = self.custom_model.predict(frame, conf=conf, imgsz=640, verbose=False)
        use_results = custom_results
        
        # Fallback to segmentation model if no masks
        if (len(custom_results) == 0 or 
            getattr(custom_results[0], 'boxes', None) is None or 
            getattr(custom_results[0], 'masks', None) is None):
            seg_results = self.seg_model.predict(frame, conf=conf, imgsz=640, verbose=False)
            if len(seg_results) > 0:
                use_results = seg_results
        
        # Try model-based detection
        if len(use_results) > 0 and getattr(use_results[0], 'boxes', None) is not None:
            model_cans = self._detect_cans_from_model(frame, use_results, zone)
            cans.extend(model_cans)
        
        # If no cans detected by model, try color-based detection (white bottles)
        if len(cans) == 0:
            color_cans = self._detect_cans_by_color(frame, zone)
            cans.extend(color_cans)
        
        # If still no cans, try circle detection (cans appear circular from above)
        if len(cans) == 0:
            circle_cans = self._detect_cans_by_circle(frame, zone)
            cans.extend(circle_cans)
        
        return cans
    
    def _detect_cans_from_model(self, frame, use_results, zone):
        """Extract can detections from ML model results"""
        cans = []
        
        if len(use_results) == 0 or not hasattr(use_results[0], 'boxes'):
            return cans
        
        for i, box in enumerate(use_results[0].boxes):
            try:
                coords = box.xyxy[0]
                if hasattr(coords, 'cpu'):
                    coords = coords.cpu().numpy()
            except Exception:
                coords = box.xyxy
                if hasattr(coords, 'cpu'):
                    coords = coords.cpu().numpy()
                coords = coords[0]
            
            x1, y1, x2, y2 = map(int, coords)
            w, h = x2 - x1, y2 - y1
            
            # Filter by min size
            if w < 30 or h < 30:
                continue
            
            # Prefer mask-derived bbox if available
            bbox_final = (x1, y1, x2, y2)
            
            try:
                masks = getattr(use_results[0], 'masks', None)
                if masks is not None:
                    masks_data = getattr(masks, 'data', None)
                    if masks_data is not None and i < len(masks_data):
                        mask_i = masks_data[i]
                        if hasattr(mask_i, 'cpu'):
                            mask_i = mask_i.cpu().numpy()
                        else:
                            mask_i = np.asarray(mask_i)
                        
                        if mask_i.max() > 0:
                            if mask_i.max() <= 1:
                                mask_i = (mask_i * 255).astype(np.uint8)
                            else:
                                mask_i = mask_i.astype(np.uint8)
                            
                            cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if cnts:
                                c = max(cnts, key=cv2.contourArea)
                                area = cv2.contourArea(c)
                                if area >= 500:  # min area threshold
                                    mx, my, mw, mh = cv2.boundingRect(c)
                                    if (mask_i.shape[0] == frame.shape[0] and 
                                        mask_i.shape[1] == frame.shape[1]):
                                        bbox_final = (mx, my, mx + mw, my + mh)
                                    else:
                                        bbox_final = (x1 + mx, y1 + my, x1 + mx + mw, y1 + my + mh)
            except Exception:
                pass
            
            bx1, by1, bx2, by2 = bbox_final
            bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
            
            # Apply zone filter if provided
            if zone:
                cx1, cy1, cx2, cy2 = zone
                margin = 100
                if not (cx1 - margin <= bcx <= cx2 + margin and 
                        cy1 - margin <= bcy <= cy2 + margin):
                    continue
            
            # Classify can size
            size_class = self._classify_can_size(bx2 - bx1, by2 - by1)
            
            can_info = {
                'bbox': bbox_final,
                'size': size_class,
                'centroid': (bcx, bcy),
                'confidence': float(box.conf[0]) if hasattr(box, 'conf') else 0.5
            }
            cans.append(can_info)
        
        return cans
    
    def _detect_cans_by_color(self, frame, zone):
        """Detect white/light-colored cylindrical objects (water bottles/cans) using color detection"""
        cans = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # White/very light colors in HSV: low saturation, high value
            # More permissive range for white detection
            lower_white = np.array([0, 0, 150])      # Lower V threshold for darker whites
            upper_white = np.array([180, 80, 255])   # Higher S threshold for less saturated whites
            
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Also detect very saturated light grays/silvers
            lower_gray = np.array([0, 0, 100])
            upper_gray = np.array([180, 30, 200])
            mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
            
            mask = cv2.bitwise_or(mask, mask_gray)
            
            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 200:  # very low min area
                    continue
                
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter by aspect ratio (cans are roughly cylindrical, but allow more variation)
                aspect_ratio = h / (w + 0.001)
                if aspect_ratio < 0.4 or aspect_ratio > 6:  # Very lenient range
                    continue
                
                # SKIP zone filter - detect anywhere in frame
                bcx, bcy = x + w // 2, y + h // 2
                
                size_class = self._classify_can_size(w, h)
                
                can_info = {
                    'bbox': (x, y, x + w, y + h),
                    'size': size_class,
                    'centroid': (bcx, bcy),
                    'confidence': 0.6  # Lower confidence for color-based detection
                }
                cans.append(can_info)
        
        except Exception as e:
            pass
        
        return cans
    
    def _detect_cans_by_circle(self, frame, zone):
        """Detect cans using Hough Circle detection (cans appear circular from above)"""
        cans = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to smooth
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Hough Circle detection
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,         # Min distance between circles (lowered)
                param1=30,          # Canny edge detection threshold (lowered)
                param2=15,          # Circle accumulator threshold (much lower)
                minRadius=15,       # Minimum circle radius (lowered)
                maxRadius=200       # Maximum circle radius (increased)
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                for circle in circles[0, :]:
                    cx, cy, radius = circle
                    area = np.pi * radius ** 2
                    
                    if area < 200:  # min area
                        continue
                    
                    # Create bbox from circle
                    x1 = max(0, cx - radius)
                    y1 = max(0, cy - radius)
                    x2 = min(frame.shape[1], cx + radius)
                    y2 = min(frame.shape[0], cy + radius)
                    
                    # Skip if zone provided and circle outside zone
                    if zone:
                        cx1, cy1, cx2, cy2 = zone
                        margin = 300  # Very generous margin
                        if not (cx1 - margin <= cx <= cx2 + margin and 
                                cy1 - margin <= cy <= cy2 + margin):
                            continue
                    
                    size_class = self._classify_can_size(x2 - x1, y2 - y1)
                    
                    can_info = {
                        'bbox': (x1, y1, x2, y2),
                        'size': size_class,
                        'centroid': (cx, cy),
                        'confidence': 0.5  # Circle detection has moderate confidence
                    }
                    cans.append(can_info)
        
        except Exception as e:
            pass
        
        return cans
    
    def _classify_can_size(self, width, height):
        """Classify can size based on bbox dimensions (placeholder)"""
        # TODO: This will be trained as a separate classification model
        # For now, use simple heuristics
        avg_dim = (width + height) / 2
        if avg_dim < 80:
            return "5L"
        elif avg_dim < 200:
            return "10L"
        elif avg_dim < 400:
            return "20L"
        else:
            return "30L"
    
    def merge_overlapping_boxes(self, cans, iou_threshold=0.35):
        """Merge overlapping can detections"""
        if not cans:
            return []
        
        def iou(box1, box2):
            x1a, y1a, x2a, y2a = box1
            x1b, y1b, x2b, y2b = box2
            ix1 = max(x1a, x1b)
            iy1 = max(y1a, y1b)
            ix2 = min(x2a, x2b)
            iy2 = min(y2a, y2b)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = ((x2a - x1a) * (y2a - y1a) + 
                     (x2b - x1b) * (y2b - y1b) - inter)
            return inter / union if union > 0 else 0.0
        
        merged = []
        used = set()
        
        for i, can1 in enumerate(cans):
            if i in used:
                continue
            cluster = [can1]
            used.add(i)
            
            for j, can2 in enumerate(cans):
                if j <= i or j in used:
                    continue
                if iou(can1['bbox'], can2['bbox']) > iou_threshold:
                    cluster.append(can2)
                    used.add(j)
            
            # Merge cluster
            merged_can = self._merge_can_cluster(cluster)
            merged.append(merged_can)
        
        return merged
    
    def _merge_can_cluster(self, cluster):
        """Merge multiple overlapping can detections into one"""
        bboxes = [c['bbox'] for c in cluster]
        x1s = [b[0] for b in bboxes]
        y1s = [b[1] for b in bboxes]
        x2s = [b[2] for b in bboxes]
        y2s = [b[3] for b in bboxes]
        
        merged_bbox = (min(x1s), min(y1s), max(x2s), max(y2s))
        avg_conf = np.mean([c['confidence'] for c in cluster])
        # Use most common size in cluster
        sizes = [c['size'] for c in cluster]
        size_class = max(set(sizes), key=sizes.count)
        
        cx = (merged_bbox[0] + merged_bbox[2]) // 2
        cy = (merged_bbox[1] + merged_bbox[3]) // 2
        
        return {
            'bbox': merged_bbox,
            'size': size_class,
            'centroid': (cx, cy),
            'confidence': avg_conf
        }
