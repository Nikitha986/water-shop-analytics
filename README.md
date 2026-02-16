# Water Shop Analytics

This project is designed to analyze CCTV footage from a water shop to track customer visits, detect payment modes, and monitor operational insights.


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

##  
Contributions, suggestions, and improvements are welcome.

If you’d like to collaborate:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request
