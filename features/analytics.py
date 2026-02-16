"""
Analytics Module
Generates insights: peak hours, daily summaries, trends
"""
from datetime import datetime, timedelta
from collections import defaultdict
import json

class AnalyticsEngine:
    """Generate business analytics and reports"""
    
    def __init__(self):
        self.hourly_stats = defaultdict(lambda: {
            'customer_count': 0,
            'cans_sold': 0,
            'can_breakdown': defaultdict(int),  # {size: count}
            'payment_breakdown': defaultdict(int),  # {mode: count}
        })
        self.daily_stats = {}
    
    def record_visit(self, timestamp, num_cans, can_sizes, payment_mode):
        """Record visit for analytics"""
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
        
        self.hourly_stats[hour_key]['customer_count'] += 1
        self.hourly_stats[hour_key]['cans_sold'] += num_cans
        
        for size in can_sizes:
            self.hourly_stats[hour_key]['can_breakdown'][size] += 1
        
        self.hourly_stats[hour_key]['payment_breakdown'][payment_mode] += 1
    
    def get_peak_hours(self, date_str=None):
        """Identify peak hours on a given date"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        peak_hours = []
        for hour_key, stats in self.hourly_stats.items():
            if hour_key.startswith(date_str):
                peak_hours.append({
                    'hour': hour_key,
                    'customer_count': stats['customer_count'],
                    'cans_sold': stats['cans_sold']
                })
        
        # Sort by customer count descending
        return sorted(peak_hours, key=lambda x: x['customer_count'], reverse=True)
    
    def get_daily_summary(self, date_str=None):
        """Generate daily summary report"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        daily_customers = 0
        daily_cans = 0
        daily_can_breakdown = defaultdict(int)
        daily_payment_breakdown = defaultdict(int)
        
        for hour_key, stats in self.hourly_stats.items():
            if hour_key.startswith(date_str):
                daily_customers += stats['customer_count']
                daily_cans += stats['cans_sold']
                for size, count in stats['can_breakdown'].items():
                    daily_can_breakdown[size] += count
                for mode, count in stats['payment_breakdown'].items():
                    daily_payment_breakdown[mode] += count
        
        return {
            'date': date_str,
            'total_customers': daily_customers,
            'total_cans': daily_cans,
            'can_breakdown': dict(daily_can_breakdown),
            'payment_breakdown': dict(daily_payment_breakdown),
            'peak_hours': self.get_peak_hours(date_str)[:3]
        }
    
    def get_weekly_summary(self, end_date=None):
        """Generate weekly summary report"""
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        weekly_customers = 0
        weekly_cans = 0
        weekly_can_breakdown = defaultdict(int)
        daily_summaries = []
        
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            daily = self.get_daily_summary(date_str)
            daily_summaries.append(daily)
            
            weekly_customers += daily['total_customers']
            weekly_cans += daily['total_cans']
            for size, count in daily['can_breakdown'].items():
                weekly_can_breakdown[size] += count
            
            current += timedelta(days=1)
        
        return {
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'total_customers': weekly_customers,
            'total_cans': weekly_cans,
            'can_breakdown': dict(weekly_can_breakdown),
            'avg_customers_per_day': weekly_customers / 7,
            'daily_summaries': daily_summaries
        }
    
    def export_report(self, report_data, output_file):
        """Export report to JSON"""
        try:
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"Report exported to {output_file}")
        except Exception as e:
            print(f"Error exporting report: {e}")
