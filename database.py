from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///data/shop.db")
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Visit(Base):
    __tablename__ = "visits"

    id = Column(Integer, primary_key=True)
    customer_id = Column(String)
    can_size = Column(String)
    payment_mode = Column(String)
    status = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)
