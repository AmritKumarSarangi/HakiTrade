import yfinance as yf
import pandas as pd
import numpy as np


def backtest_strategy(symbol):

    df = yf.download(
        symbol,
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=True
    )

    if df.empty:
        return None

    # MOVING AVERAGES
    df["MA5"] = df["Close"].rolling(5).mean()

    df["MA20"] = df["Close"].rolling(20).mean()

    # SIGNALS
    df["Signal"] = 0

    df.loc[df["MA5"] > df["MA20"], "Signal"] = 1

    df.loc[df["MA5"] < df["MA20"], "Signal"] = -1

    # RETURNS
    df["Market Return"] = df["Close"].pct_change()

    df["Strategy Return"] = (
        df["Signal"].shift(1)
        * df["Market Return"]
    )

    df = df.dropna()

    # CUMULATIVE RETURNS
    df["Cumulative Market"] = (
        1 + df["Market Return"]
    ).cumprod()

    df["Cumulative Strategy"] = (
        1 + df["Strategy Return"]
    ).cumprod()

    # PERFORMANCE
    strategy_return = (
        df["Cumulative Strategy"].iloc[-1] - 1
    ) * 100

    market_return = (
        df["Cumulative Market"].iloc[-1] - 1
    ) * 100

    # WIN RATE
    winning_trades = (
        df["Strategy Return"] > 0
    ).sum()

    total_trades = len(df)

    win_rate = (
        winning_trades / total_trades
    ) * 100

    # SHARPE RATIO
    sharpe_ratio = (
        df["Strategy Return"].mean()
        / df["Strategy Return"].std()
    ) * np.sqrt(252)

    # MAX DRAWDOWN
    rolling_max = df["Cumulative Strategy"].cummax()

    drawdown = (
        df["Cumulative Strategy"]
        - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min() * 100

    return {
        "symbol": symbol,

        "strategy_return": round(strategy_return, 2),

        "market_return": round(market_return, 2),

        "win_rate": round(win_rate, 2),

        "sharpe_ratio": round(float(sharpe_ratio), 2),

        "max_drawdown": round(float(max_drawdown), 2)
    }