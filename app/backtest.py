import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import torch
import os
from abc import ABC, abstractmethod

# FIX 4: Indian equity transaction costs for Zerodha CNC (delivery) orders.
# These are deducted once per round-trip trade (buy + sell).
#
# Breakdown (approximate, 2024-25 rates):
#   STT                 : 0.10% on sell-side turnover
#   Stamp duty          : 0.015% on buy-side turnover
#   NSE exchange charge : 0.00322% on turnover
#   GST (18%) on charges: ~0.001%
#   SEBI charges        : negligible
# --------------------------------------------------
#   Round-trip total    : ~0.12%
INDIA_ROUNDTRIP_COST = 0.0012   # 0.12 % per completed trade (buy + sell)

class BaseStrategy(ABC):
    def __init__(self):
        pass
        
    @abstractmethod
    def generate_signals(self, df):
        """Should add a 'Signal' column to df (1 for buy, -1 for sell, 0 for neutral)"""
        pass

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, short_window=5, long_window=20):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, df):
        df[f"MA{self.short_window}"] = df["Close"].rolling(self.short_window).mean()
        df[f"MA{self.long_window}"] = df["Close"].rolling(self.long_window).mean()
        df["Signal"] = 0
        df.loc[df[f"MA{self.short_window}"] > df[f"MA{self.long_window}"], "Signal"] = 1
        df.loc[df[f"MA{self.short_window}"] < df[f"MA{self.long_window}"], "Signal"] = -1
        return df

class EnsembleMLStrategy(BaseStrategy):
    def __init__(self, model_dir="models"):
        super().__init__()
        self.xgb = joblib.load(os.path.join(model_dir, "xgb.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.features = ["RSI", "MACD", "Signal_MACD", "Momentum", "Volume_Ratio", "Volatility"]
        
    def build_features(self, df):
        delta = df["Close"].diff()
        gain  = delta.where(delta > 0, 0.0)
        loss  = -delta.where(delta < 0, 0.0)
        rs    = gain.rolling(14).mean() / loss.rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]   = ema12 - ema26
        df["Signal_MACD"] = df["MACD"].ewm(span=9, adjust=False).mean()

        df["Momentum"]     = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100
        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["Volatility"]   = df["Close"].pct_change().rolling(10).std() * 100
        return df
        
    def generate_signals(self, df):
        df = self.build_features(df)
        # We need to drop NAs to scale, but we want to keep index aligned
        # So we create a boolean mask for valid rows
        valid_mask = df[self.features].notna().all(axis=1)
        
        df["Signal"] = 0
        if valid_mask.sum() == 0:
            return df
            
        X_raw = df.loc[valid_mask, self.features].values
        X_sc = self.scaler.transform(X_raw)
        
        probs = self.xgb.predict_proba(X_sc)[:, 1]
        
        # 1 for buy, -1 for sell
        df.loc[valid_mask, "Signal"] = np.where(probs > 0.5, 1, -1)
        return df

class Backtester:
    def __init__(self, strategy, symbol="RELIANCE.NS", period="1y"):
        self.strategy = strategy
        self.symbol = symbol
        self.period = period
        
    def fetch_data(self):
        df = yf.download(
            self.symbol,
            period=self.period,
            interval="1d",
            progress=False,
            auto_adjust=True
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
        
    def run(self):
        df = self.fetch_data()
        if df.empty:
            return None
            
        df = self.strategy.generate_signals(df)
        
        # RETURNS
        df["Market Return"]   = df["Close"].pct_change()
        df["Strategy Return"] = df["Signal"].shift(1) * df["Market Return"]

        # FIX 4: Apply Indian equity transaction costs.
        # A round-trip cost is charged whenever the strategy reverses direction
        # (signal flips from +1 → -1 or vice versa, i.e. a new trade is opened).
        # Half the round-trip cost is applied on entry and half on exit, which
        # is equivalent to deducting the full cost at the moment of the flip.
        signal_shifted   = df["Signal"].shift(1)
        signal_prev      = df["Signal"].shift(2)
        trade_flip       = (signal_shifted != signal_prev) & signal_shifted.notna() & signal_prev.notna()
        df["Strategy Return"] -= trade_flip.astype(float) * INDIA_ROUNDTRIP_COST

        df = df.dropna()
        
        if df.empty:
            return None
            
        # CUMULATIVE RETURNS
        df["Cumulative Market"] = (1 + df["Market Return"]).cumprod()
        df["Cumulative Strategy"] = (1 + df["Strategy Return"]).cumprod()
        
        # PERFORMANCE
        strategy_return = (df["Cumulative Strategy"].iloc[-1] - 1) * 100
        market_return = (df["Cumulative Market"].iloc[-1] - 1) * 100
        
        # WIN RATE
        winning_trades = (df["Strategy Return"] > 0).sum()
        total_trades = len(df)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # SHARPE RATIO
        std = df["Strategy Return"].std()
        sharpe_ratio = (df["Strategy Return"].mean() / std) * np.sqrt(252) if std != 0 else 0
        
        # MAX DRAWDOWN
        rolling_max = df["Cumulative Strategy"].cummax()
        drawdown = (df["Cumulative Strategy"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # SORTINO RATIO (downside deviation)
        downside_returns = df.loc[df["Strategy Return"] < 0, "Strategy Return"]
        down_std = downside_returns.std()
        sortino_ratio = (df["Strategy Return"].mean() / down_std) * np.sqrt(252) if down_std != 0 else 0
        
        return {
            "symbol": self.symbol,
            "strategy": self.strategy.__class__.__name__,
            "strategy_return": round(strategy_return, 2),
            "market_return": round(market_return, 2),
            "win_rate": round(win_rate, 2),
            "sharpe_ratio": round(float(sharpe_ratio), 2),
            "sortino_ratio": round(float(sortino_ratio), 2),
            "max_drawdown": round(float(max_drawdown), 2)
        }

def backtest_strategy(symbol, strategy_type="ma"):
    # Wrapper to maintain backward compatibility with main.py
    if strategy_type == "ml":
        strategy = EnsembleMLStrategy()
    else:
        strategy = MovingAverageStrategy()
        
    bt = Backtester(strategy, symbol)
    return bt.run()