# Water Shop Analytics

This project is designed to analyze CCTV footage from a water shop to track customer visits, detect payment modes, and monitor operational insights.

## Folder Structure

```
water-shop-analytics/
│
├── app_v2.py                # Main application script for analyzing CCTV footage
├── app.py                   # Legacy application script
├── analytics.py             # Analytics-related functions
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── data/                    # Data storage
│   ├── raw/                 # Raw data and datasets
│   ├── processed/           # Processed data
│   └── logs/                # Log files
├── models/                  # Pretrained models
│   └── pretrained/          # Pretrained model files
├── notebooks/               # Jupyter notebooks for analysis
├── tests/                   # Test scripts
│   ├── test_video.py
│   ├── test_modules.py
│   └── test_can.py
├── utils/                   # Utility scripts
│   ├── analytics_utils.py
│   ├── debug_can_detect.py
│   ├── debug_detect.py
│   ├── extract_frames.py
│   ├── reid.py
│   ├── tracker.py
│   └── utils.py
├── videos/                  # Video files
└── features/                # Feature-specific modules
    ├── analytics.py
    ├── can_detection.py
    ├── customer_database.py
    ├── payment_detection.py
    ├── person_tracking.py
    └── __init__.py
```

## Features

1. **Customer Visit Tracking**:
   - Detects customers entering the shop.
   - Tracks the number of visits.
   - Ensures accurate counting by verifying proximity to the filling area.

2. **Payment Mode Detection**:
   - Detects payment modes (Cash, UPI, etc.).
   - Tracks payment status (Paid/Unpaid).

3. **Operational Insights**:
   - Tracks peak hours and idle times.
   - Provides daily, weekly, and monthly reports.

4. **Object Detection**:
   - Uses YOLOv8 for detecting customers and objects in the shop.

5. **Data Management**:
   - Organized folder structure for raw data, processed data, logs, and models.

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd water-shop-analytics
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the main application:
   ```
   python app_v2.py
   ```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLO

## License

This project is licensed under the MIT License.