"""
Water Shop Analytics Features Package
Main modules for CCTV-based business intelligence
"""

from .person_tracking import PersonTracker
from .can_detection import CanDetector
from .payment_detection import PaymentDetector
from .customer_database import CustomerDatabase
from .analytics import AnalyticsEngine

__all__ = [
    'PersonTracker',
    'CanDetector',
    'PaymentDetector',
    'CustomerDatabase',
    'AnalyticsEngine'
]
