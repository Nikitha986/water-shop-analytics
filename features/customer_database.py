"""
Customer Database Module
Tracks repeat customers, visit frequency, and payment status
"""
import json
from datetime import datetime
from pathlib import Path

class CustomerDatabase:
    """Persistent store for customer visit and payment data"""
    
    def __init__(self, db_path="customer_database.json"):
        self.db_path = Path(db_path)
        self.customers = self._load_db()
    
    def _load_db(self):
        """Load customer database from file"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                return {}
        return {}
    
    def _save_db(self):
        """Save customer database to file"""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.customers, f, indent=2)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def add_visit(self, customer_id, timestamp, cans_bought, can_sizes, payment_mode="UNKNOWN", paid=False):
        """
        Record a customer visit
        Args:
            customer_id: unique customer identifier
            timestamp: datetime of visit
            cans_bought: number of cans
            can_sizes: list of can sizes (e.g., ["20L", "10L"])
            payment_mode: "CASH", "UPI", "MACHINE"
            paid: True if transaction completed
        """
        if customer_id not in self.customers:
            self.customers[customer_id] = {
                'first_visit': timestamp.isoformat(),
                'visits': [],
                'total_cans': 0,
                'total_visits': 0,
                'payment_history': [],
                'unpaid_transactions': []
            }
        
        visit_record = {
            'timestamp': timestamp.isoformat(),
            'cans': cans_bought,
            'can_sizes': can_sizes,
            'payment_mode': payment_mode,
            'paid': paid
        }
        
        self.customers[customer_id]['visits'].append(visit_record)
        self.customers[customer_id]['total_cans'] += cans_bought
        self.customers[customer_id]['total_visits'] += 1
        self.customers[customer_id]['payment_history'].append({
            'timestamp': timestamp.isoformat(),
            'mode': payment_mode,
            'paid': paid
        })
        
        if not paid:
            self.customers[customer_id]['unpaid_transactions'].append(visit_record)
        
        self._save_db()
    
    def get_repeat_customers(self, min_visits=2):
        """Get customers with at least min_visits"""
        return [cid for cid, data in self.customers.items() 
                if data['total_visits'] >= min_visits]
    
    def get_unpaid_customers(self):
        """Get list of customers with unpaid transactions"""
        return [cid for cid, data in self.customers.items() 
                if len(data['unpaid_transactions']) > 0]
    
    def get_customer_info(self, customer_id):
        """Get customer profile"""
        return self.customers.get(customer_id, None)
