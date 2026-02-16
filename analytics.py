from datetime import datetime
import pandas as pd

class ShopAnalytics:
    def __init__(self):
        self.customer_count = 0
        self.counted_ids = set()
        self.entry_times = {}
        self.visit_durations = []
        self.visit_log = []
        self.hourly_visits = {}
        self.opening_time = None
        self.closing_time = None

    def register_entry(self, track_id, now):
        if track_id in self.counted_ids:
            return

        self.counted_ids.add(track_id)
        self.customer_count += 1
        self.entry_times[track_id] = now

        hour = now.strftime("%H:00")
        self.hourly_visits[hour] = self.hourly_visits.get(hour, 0) + 1

        if self.opening_time is None:
            self.opening_time = now.strftime("%H:%M:%S")

        self.closing_time = now.strftime("%H:%M:%S")

    def register_exit(self, track_id, now):
        if track_id not in self.entry_times:
            return

        duration = (now - self.entry_times[track_id]).seconds
        self.visit_durations.append(duration)

        self.visit_log.append({
            "visitor_id": track_id,
            "entry_time": self.entry_times[track_id].strftime("%H:%M:%S"),
            "duration_seconds": duration
        })

        del self.entry_times[track_id]

    def generate_report(self, path):
        avg_duration = (
            sum(self.visit_durations) / len(self.visit_durations)
            if self.visit_durations else 0
        )

        df_summary = pd.DataFrame([{
            "total_customers": self.customer_count,
            "opening_time": self.opening_time,
            "closing_time": self.closing_time,
            "avg_visit_duration_sec": avg_duration
        }])

        df_visits = pd.DataFrame(self.visit_log)
        df_hours = pd.DataFrame(list(self.hourly_visits.items()),
                                columns=["hour", "visits"])

        df_summary.to_csv(path, index=False)
        df_visits.to_csv(path.replace(".csv", "_visits.csv"), index=False)
        df_hours.to_csv(path.replace(".csv", "_hourly.csv"), index=False)

        print("\nReports saved to:", path)
