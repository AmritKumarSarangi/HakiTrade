from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base

class StockData(Base):
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Relationship to the features
    features = relationship("FeatureStore", back_populates="stock_data", uselist=False)

class FeatureStore(Base):
    __tablename__ = "feature_store"

    id = Column(Integer, primary_key=True, index=True)
    stock_data_id = Column(Integer, ForeignKey("stock_data.id"), unique=True, nullable=False)
    
    # Store arbitrary technical indicators (RSI, MACD, BB, etc.) in JSON format.
    # This allows adding new indicators in the future without database migrations!
    indicators = Column(JSON, nullable=False, default=dict)

    stock_data = relationship("StockData", back_populates="features")
