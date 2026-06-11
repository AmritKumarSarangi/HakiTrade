import yfinance as yf
import pandas as pd
import numpy as np
import ta
import logging
from datetime import datetime

from app.celery_worker import celery_app
from app.database import SessionLocal, engine
from app.models.db_models import Base, StockData, FeatureStore

# Ensure tables are created when tasks are imported
Base.metadata.create_all(bind=engine)

logger = logging.getLogger(__name__)

STOCKS_TO_TRACK = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS"
]

@celery_app.task(name="app.tasks.fetch_daily_market_data")
def fetch_daily_market_data():
    """
    Background task to fetch End-Of-Day market data and compute indicators.
    Scheduled to run at 4:15 PM IST.
    """
    logger.info("Starting daily market data fetch...")
    
    db = SessionLocal()
    try:
        for symbol in STOCKS_TO_TRACK:
            # Fetch last 30 days to ensure enough data for indicator lookback periods
            df = yf.download(symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            # Expand features using 'ta' library
            df = calculate_features(df)
            
            # We only want to save the latest trading day to the database in this daily task
            latest_row = df.iloc[-1]
            latest_date = df.index[-1].date()
            
            # Check if record already exists for this date
            existing_record = db.query(StockData).filter(
                StockData.symbol == symbol,
                StockData.date == latest_date
            ).first()
            
            if existing_record:
                logger.info(f"Record for {symbol} on {latest_date} already exists. Skipping.")
                continue
                
            # Create StockData (OHLCV)
            stock_data = StockData(
                symbol=symbol,
                date=latest_date,
                open=float(latest_row['Open']),
                high=float(latest_row['High']),
                low=float(latest_row['Low']),
                close=float(latest_row['Close']),
                volume=float(latest_row['Volume'])
            )
            db.add(stock_data)
            db.flush() # Flush to get the stock_data.id
            
            # Extract the ~20 technical indicators into a JSON dict
            # Exclude raw OHLCV columns from the JSON payload
            indicator_dict = {}
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for col in df.columns:
                if col not in exclude_cols and pd.notna(latest_row[col]):
                    indicator_dict[col] = float(latest_row[col])
                    
            # Create FeatureStore record linked to StockData
            feature_store = FeatureStore(
                stock_data_id=stock_data.id,
                indicators=indicator_dict
            )
            db.add(feature_store)
            logger.info(f"Successfully saved {symbol} EOD data and {len(indicator_dict)} features.")
            
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error during fetch task: {str(e)}")
    finally:
        db.close()
        
    logger.info("Daily market data fetch complete.")

def calculate_features(df):
    """
    Computes ~20 technical indicators.
    Returns the dataframe with new feature columns.
    """
    # 1. Momentum Indicators
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['Stoch_K'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
    df['Stoch_D'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch_signal()
    df['Williams_R'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close']).williams_r()
    df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'], window=12).roc()
    
    # 2. Trend Indicators
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    df['EMA_12'] = ta.trend.EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=df['Close'], window=26).ema_indicator()
    df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
    
    # 3. Volatility Indicators
    bb = ta.volatility.BollingerBands(close=df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    
    # 4. Volume Indicators
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).chaikin_money_flow()

    return df
