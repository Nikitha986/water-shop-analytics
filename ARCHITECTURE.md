# Water Shop Analytics System

## Project Overview
Automated CCTV-based business intelligence system for water shops. Tracks customer visits, can demand (by size), payment modes, and generates daily/weekly/monthly reports.

## Architecture

### Core Modules (`features/`)

#### 1. `person_tracking.py` - PersonTracker
- **Purpose**: Detect and track customers with persistent IDs
- **Input**: Video frame
- **Output**: List of {id, bbox, centroid} for each detected person
- **Model**: YOLOv8n (person class only)
- **Status**: ✅ Working

#### 2. `can_detection.py` - CanDetector
- **Purpose**: Detect water cans and classify by size
- **Input**: Video frame, optional zone filter
- **Output**: List of {bbox, size, centroid, confidence}
- **Models**: 
  - Primary: Custom water_can_model.pt
  - Fallback: yolov8n-seg (segmentation model)
- **Features**:
  - Instance mask extraction from segmentation
  - Bbox merging using IoU clustering
  - Size classification (5L, 10L, 20L, 30L)
  - Zone filtering for counter area
- **Status**: ✅ Working (needs fine-tuning for size classification)

#### 3. `payment_detection.py` - PaymentDetector
- **Purpose**: Identify payment mode (Cash, UPI, Machine)
- **Input**: Video frame, ROI bbox
- **Output**: payment_mode (string) or None
- **Status**: 🔧 Stub - requires implementation

#### 4. `customer_database.py` - CustomerDatabase
- **Purpose**: Persistent storage of customer visit/payment history
- **Features**:
  - Track repeat customers
  - Record can purchase history by size
  - Flag unpaid transactions
  - Customer profile queries
- **Storage**: JSON file (customer_database.json)
- **Status**: ✅ Implemented (ready for use)

#### 5. `analytics.py` - AnalyticsEngine
- **Purpose**: Generate business insights and reports
- **Capabilities**:
  - Hourly customer counts and can sales
  - Peak hour identification
  - Daily summary (customers, cans by size, payment modes)
  - Weekly trend analysis
  - Report export to JSON
- **Status**: ✅ Implemented (ready for use)

---

## Main Pipeline (`app_v2.py`)

**Execution Flow:**
1. Load video frame
2. **Track persons** → Get persistent customer IDs
3. **Detect cans** → Get can detections with sizes
4. **Associate cans to persons** → Link each can to nearest person
5. **Merge overlapping detections** → IoU-based clustering
6. **Deduplicate** → Prevent counting same can multiple times
7. **Update analytics** → Record visit in database and analytics engine
8. **Render UI** → Display boxes and statistics
9. **Export reports** → Daily/weekly summaries

**UI Controls:**
- `ESC` - Exit
- `R` - Reset can counter
- `S` - Save daily report to JSON

**Output Files:**
- `customer_database.json` - Persistent customer records
- `report_YYYYMMDD.json` - Daily report

---

## Configuration

### COUNTER_ZONE
Defines the spatial region where cans should be detected (reduces false positives)
```python
COUNTER_ZONE = (950, 100, 1280, 750)  # (x1, y1, x2, y2)
```

### Key Thresholds
- `CAN_CONFIDENCE`: 0.03 (model confidence for detection)
- `PERSON_CAN_ASSOC_DIST`: 400px (max distance to link can to person)
- `CAN_DEDUP_DIST`: 200px (same can if centroid within this distance)
- `CAN_COOLDOWN`: 5s (minimum time between counting same can)

---

## Development Roadmap (Following ML Rules)

### Hour 1: ✅ Structure & Stub Code
- Project folder structure created
- 5 modular features implemented
- Main pipeline (`app_v2.py`) with integration
- **Git Commit**: "Hour 1 – Full architecture and module structure"

### Hour 2: Fine-tune Can Detection
- Adjust COUNTER_ZONE boundaries for accurate detection
- Test segmentation fallback model
- Validate bbox merging logic
- **Deliverable**: Can detection working reliably
- **Git Commit**: "Hour 2 – Can detection tuning and validation"

### Hour 3: Can Size Classification
- Collect 5-10 labeled images (5L, 10L, 20L, 30L)
- Annotate in YOLO format
- **Deliverable**: Annotated dataset
- **Git Commit**: "Hour 3 – Can size dataset preparation (5 images)"

### Hour 4: Size Classification Training (Dry Run)
- Train YOLOv8 classifier on 5 images, 1 epoch
- **Deliverable**: Verify training pipeline works
- **Git Commit**: "Hour 4 – Size classifier dry run (1 epoch)"

### Hour 5: Payment Detection Research
- Explore payment gesture recognition / QR code detection
- Define detection approach
- **Deliverable**: Solution design doc
- **Git Commit**: "Hour 5 – Payment detection approach"

### Hour 6+: Incremental Feature Addition
- Payment mode detection implementation
- Repeat customer recognition (face similarity)
- Mobile dashboard for shop owner
- Multi-branch analytics

---

## Running the System

### Prerequisites
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Main Pipeline
```bash
python app_v2.py
```

### Run Diagnostics
```bash
python check_frame.py  # Test can detection on single frame
python debug_can_detect.py  # Detailed can detection analysis
```

---

## File Structure
```
water-shop-analytics/
├── app.py                      # Original pipeline (legacy)
├── app_v2.py                   # New modular pipeline ✅
├── features/
│   ├── __init__.py
│   ├── person_tracking.py      # CustomerTracking ✅
│   ├── can_detection.py        # Can detection + size ✅
│   ├── payment_detection.py    # Payment detection 🔧
│   ├── customer_database.py    # Customer records ✅
│   └── analytics.py            # Reports & insights ✅
├── models/
│   ├── best.pt                 # Custom trained model
│   └── water_can_model.pt      # Can detection model
├── dataset.yaml                # YOLO dataset config
├── requirements.txt            # Dependencies
└── customer_database.json       # Generated at runtime
```

---

## Business Requirements Covered

| Requirement | Status | Module |
|---|---|---|
| Customer counting | ✅ | PersonTracker |
| Can demand (volume) | ✅ | AnalyticsEngine |
| Can size classification | 🔧 | CanDetector (needs training) |
| Payment mode detection | 🔧 | PaymentDetector |
| Repeat customer tracking | ✅ | CustomerDatabase |
| Unpaid customer flag | ✅ | CustomerDatabase |
| Peak hours analysis | ✅ | AnalyticsEngine |
| Daily reports | ✅ | AnalyticsEngine |
| Weekly reports | ✅ | AnalyticsEngine |

---

## Next Steps

1. **Run `app_v2.py`** with the updated COUNTER_ZONE and verify can detection
2. **Verify person-can associations** are correct in the UI
3. **Test deduplication logic** - same can should be counted only once per cooldown
4. **Begin training can size classifier** (Hour 3 per ML Rules)
5. **Implement payment detection** (Hour 5+ per ML Rules)

---

## Troubleshooting

### Can not detected
- Check COUNTER_ZONE boundaries match actual counter location
- Verify segmentation model (`yolov8n-seg`) is loaded correctly
- Increase `CAN_CONFIDENCE` temporarily for debugging

### False positives
- Reduce COUNTER_ZONE size
- Increase `CAN_MIN_AREA` threshold
- Check zone margin settings

### Person-can not associating
- Increase `PERSON_CAN_ASSOC_DIST`
- Verify person tracking is working (green boxes visible)
- Check person centroids vs can centroids

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-14  
**Owner**: Development Team
