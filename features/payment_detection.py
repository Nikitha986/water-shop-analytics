"""
Payment Mode Detection Module
Handles identification of payment methods (Cash, Scan/UPI, Unpaid)
"""
import cv2
import numpy as np

# Try to import pyzbar for QR code detection (optional)
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

class PaymentDetector:
    """
    Detects payment mode from CCTV footage:
    - SCAN: QR code detected (Scan/UPI payment)
    - CASH: No QR code, payment completed (cash payment)
    - UNPAID: No payment indicators detected
    """
    
    def __init__(self):
        self.payment_modes = ["CASH", "SCAN", "UNPAID"]
    
    def detect_payment_mode(self, frame, bbox, detection_threshold=0.5):
        """
        Detect payment mode in region of interest
        Returns: payment_mode (str) - "SCAN", "CASH", or "UNPAID"
        """
        if bbox is None or frame is None:
            return "UNPAID"
        
        x1, y1, x2, y2 = bbox
        roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        
        if roi.size == 0:
            return "UNPAID"
        
        # Try to detect QR code (indicates SCAN/UPI payment)
        qr_detected = self._detect_qr_code(roi)
        if qr_detected:
            return "SCAN"
        
        # Check for payment completion indicators
        cash_detected = self._detect_cash_payment(roi)
        if cash_detected:
            return "CASH"
        
        # Default to UNPAID if no indicators found
        return "UNPAID"
    
    def _detect_qr_code(self, roi):
        """
        Detect QR code in ROI (indicates SCAN/UPI payment)
        Returns: True if QR code detected, False otherwise
        """
        # Try pyzbar QR code detection if available
        if PYZBAR_AVAILABLE:
            try:
                barcodes = pyzbar.decode(roi)
                if len(barcodes) > 0:
                    return True
            except Exception:
                pass
        
        # Fallback: Detect QR-like patterns using contours
        return self._detect_qr_pattern(roi)
    
    def _detect_qr_pattern(self, roi):
        """
        Fallback QR pattern detection using contours
        QR codes have distinctive geometric patterns
        """
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # QR codes typically have multiple rectangular contours
            if len(cnts) > 20:  # Heuristic: QR has many contours
                # Check for significant square/rectangular patterns
                approx_rects = 0
                for cnt in cnts:
                    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                    if len(approx) == 4:  # Rectangle
                        approx_rects += 1
                
                if approx_rects > 10:  # Multiple rectangles indicate QR pattern
                    return True
        except Exception:
            pass
        
        return False
    
    def _detect_cash_payment(self, roi):
        """
        Detect cash payment indicators (hand holding bills, coins, etc.)
        Uses simple heuristics: color contrast and motion patterns
        Returns: True if cash payment indicators detected
        """
        try:
            # Cash typically has specific color ranges (greenish/yellowish bills, metallic coins)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define color range for cash (bills/coins)
            # Green-yellow range for bills, gray/gold for coins
            lower1 = np.array([20, 50, 50])    # Yellow-green bills
            upper1 = np.array([40, 255, 255])
            lower2 = np.array([0, 0, 50])      # Gray/silver coins
            upper2 = np.array([180, 50, 200])
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # If significant area contains cash colors, likely cash payment
            cash_area_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
            if cash_area_ratio > 0.1:  # 10% of ROI contains cash-colored pixels
                return True
        except Exception:
            pass
        
        return False
    
    def classify_payment_with_state(self, payment_mode, customer_bbox, previous_frame_history):
        """
        Classify payment with state tracking over multiple frames
        More robust than single-frame detection
        """
        # If QR detected in any recent frame, classify as SCAN
        if any(p == "SCAN" for p in previous_frame_history[-5:]):
            return "SCAN"
        
        # If cash patterns detected in recent frames, classify as CASH
        if any(p == "CASH" for p in previous_frame_history[-5:]):
            return "CASH"
        
        # Default based on current frame
        return payment_mode
