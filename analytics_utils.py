import csv
import os
from datetime import datetime

# =============================
# MULTI SHOP CONFIG
# =============================

SHOP_ID = "SHOP_001"

# =============================
# SAVE DAILY REPORT
# =============================

def save_daily_report(data, file="daily_report.csv"):
    file_exists = os.path.isfile(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Date","Shop",
                "Customers","Repeat",
                "5L","10L","20L",
                "Cash","UPI","Unpaid",
                "Avg Dwell Time",
                "Peak Hour"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            SHOP_ID,
            data["customers"],
            data["repeat"],
            data["cans_5"],
            data["cans_10"],
            data["cans_20"],
            data["cash"],
            data["upi"],
            data["unpaid"],
            round(data["avg_dwell"],2),
            data["peak_hour"]
        ])

# =============================
# ALERT SYSTEM
# =============================

def save_defaulters(defaulters, file="defaulters.csv"):
    if not defaulters:
        return

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time","CustomerID"])

        for cid in defaulters:
            writer.writerow([datetime.now(), cid])
