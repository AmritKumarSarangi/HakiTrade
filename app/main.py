from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import json
import concurrent.futures
import warnings
from datetime import datetime
from dotenv import load_dotenv
from app.risk_engine import RiskManager
from app.portfolio_optimizer import PortfolioOptimizer
from app.database import SessionLocal, engine, Base
from app.models.db_models import StockData, FeatureStore

_nse_stocks_cache = None
def load_nse_stocks():
    global _nse_stocks_cache
    if _nse_stocks_cache is None:
        nse_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nse_stocks.json")
        with open(nse_path, "r") as f:
            _nse_stocks_cache = json.load(f)
    return _nse_stocks_cache

warnings.filterwarnings("ignore")

# Initialize database tables automatically
Base.metadata.create_all(bind=engine)

load_dotenv()

try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except Exception:
    KITE_AVAILABLE = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

def clean_nans(obj):
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    return obj

# ====================================================
# PyTorch Model Definitions (must match train_model.py)
# ====================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


class TransformerModel(nn.Module):
    def __init__(self, input_size=6, d_model=32, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.embed  = nn.Linear(input_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                          dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head   = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(1)


# ====================================================
# LOAD MODELS (ensemble with RF fallback)
# ====================================================
SEQ_LEN     = 10
USE_ENSEMBLE = False
xgb_model = lgb_model = lstm_model = tfm_model = meta_model = scaler = None
rf_model  = None

try:
    scaler     = joblib.load("models/scaler.pkl")
    xgb_model  = joblib.load("models/xgb.pkl")
    lgb_model  = joblib.load("models/lgb.pkl")
    meta_model = joblib.load("models/meta.pkl")

    lstm_model = LSTMModel()
    lstm_model.load_state_dict(torch.load("models/lstm.pt", map_location="cpu"))
    lstm_model.eval()

    tfm_model = TransformerModel()
    tfm_model.load_state_dict(torch.load("models/transformer.pt", map_location="cpu"))
    tfm_model.eval()

    USE_ENSEMBLE = True
    print("[OK] Ensemble loaded: XGBoost + LightGBM + LSTM + Transformer + Meta")
except Exception as e:
    print(f"[WARN] Ensemble not available ({e}). Falling back to RandomForest.")
    try:
        rf_model = joblib.load("ai_model.pkl")
    except Exception as e2:
        print(f"[ERROR] Fallback ai_model.pkl also failed: {e2}")
        rf_model = None

# ====================================================
# CORS
# ====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STOCK LIST
# =========================
stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "SBIN.NS", "INFY.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS",
    "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TATAMOTORS.NS", "KOTAKBANK.NS", "M&M.NS", "AXISBANK.NS", "ASIANPAINT.NS",
    "NTPC.NS", "TITAN.NS", "BAJAJFINSV.NS", "ULTRACEMCO.NS", "ONGC.NS",
    "WIPRO.NS", "NESTLEIND.NS", "POWERGRID.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "LTIM.NS", "TATASTEEL.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "GRASIM.NS", "TECHM.NS", "HINDALCO.NS", "EICHERMOT.NS",
    "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "INDUSINDBK.NS", "TATACONSUM.NS", "HEROMOTOCO.NS", "BPCL.NS", "JSWSTEEL.NS"
]

# =========================
# INDICATOR FUNCTIONS
# =========================

def get_stock_data(symbol):
    data = yf.download(symbol, period="3mo", interval="1d", auto_adjust=True, progress=False)

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    return data


def calculate_rsi(data, period=14):
    delta = data["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def calculate_macd(data):
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd.iloc[-1], signal.iloc[-1]


def calculate_momentum(data):
    momentum = (
        (data["Close"].iloc[-1] - data["Close"].iloc[-10])
        / data["Close"].iloc[-10]
    ) * 100

    return momentum


def calculate_volatility(data):

    returns = data["Close"].pct_change()

    volatility = float(returns.std())

    return volatility * 100


def calculate_volume_ratio(data):

    avg_volume = float(
        data["Volume"].rolling(20).mean().iloc[-1]
    )

    current_volume = float(
        data["Volume"].iloc[-1]
    )

    if avg_volume == 0:
        return 1.0

    return current_volume / avg_volume


# =========================
# ROOT
# =========================
@app.get("/")
def home():
    return {
        "message": "HakiTrade Quant Engine Running"
    }


# ====================================================
# ENSEMBLE INFERENCE HELPER
# ====================================================
def ensemble_predict(feat_row: np.ndarray, seq_matrix: np.ndarray):
    """
    feat_row  : shape (6,)  – one flat feature vector
    seq_matrix: shape (10, 6) – 10-day lookback window
    Returns (ai_probability, model_breakdown_dict)
    """
    if USE_ENSEMBLE:
        # Scale flat features
        flat_sc  = scaler.transform(feat_row.reshape(1, -1))
        # Scale sequence
        seq_sc   = scaler.transform(seq_matrix).reshape(1, SEQ_LEN, 6)
        seq_t    = torch.tensor(seq_sc, dtype=torch.float32)

        p_xgb  = float(xgb_model.predict_proba(flat_sc)[0][1])
        p_lgb  = float(lgb_model.predict_proba(flat_sc)[0][1])
        with torch.no_grad():
            p_lstm = float(lstm_model(seq_t).item())
            p_tfm  = float(tfm_model(seq_t).item())

        stack   = np.array([[p_xgb, p_lgb, p_lstm, p_tfm]])
        p_final = float(meta_model.predict_proba(stack)[0][1])

        breakdown = {
            "xgb":         round(p_xgb  * 100, 1),
            "lgb":         round(p_lgb  * 100, 1),
            "lstm":        round(p_lstm * 100, 1),
            "transformer": round(p_tfm  * 100, 1),
        }
        return p_final, breakdown
    else:
        # RandomForest fallback
        feat_df = pd.DataFrame([feat_row],
                               columns=["RSI","MACD","Signal",
                                        "Momentum","Volume_Ratio","Volatility"])
        p = float(rf_model.predict_proba(feat_df)[0][1])
        breakdown = {"xgb": None, "lgb": None, "lstm": None, "transformer": None}
        return p, breakdown

# =========================
# AUTHENTICATION
# =========================
class LoginRequest(BaseModel):
    password: str

@app.post("/login")
def login(req: LoginRequest):
    admin_password = os.getenv("ADMIN_PASSWORD", "hakitrade123")
    if req.password == admin_password:
        return {"success": True, "token": "haki_session_token_123"}
    return {"success": False, "error": "Invalid password"}

# =========================
# THREAD WORKER
# =========================
def process_stock(stock, data, news_title, sentiment_raw):
    try:
        if data is None:
            return None

        price        = float(data["Close"].iloc[-1])
        rsi          = float(calculate_rsi(data))
        macd, signal_line = calculate_macd(data)
        momentum     = float(calculate_momentum(data))
        volatility   = float(calculate_volatility(data))
        volume_ratio = float(calculate_volume_ratio(data))

        feat_row = np.array([
            rsi, float(macd), float(signal_line),
            momentum, volume_ratio, volatility
        ])

        # Build 10-day lookback sequence from recent Close prices
        close_vals = data["Close"].values
        if len(data) >= SEQ_LEN + 1:
            # Reconstruct feature matrix for the last SEQ_LEN days
            seq_rows = []
            for i in range(SEQ_LEN, 0, -1):
                c = data["Close"].iloc[-i - 1:-1] if i < len(data) else data["Close"]
                idx = -i
                delta = data["Close"].diff()
                gain  = delta.where(delta > 0, 0)
                loss  = -delta.where(delta < 0, 0)
                rs_   = gain.rolling(14).mean() / loss.rolling(14).mean()
                rsi_i = float((100 - 100 / (1 + rs_)).iloc[idx])
                e12   = data["Close"].ewm(span=12, adjust=False).mean().iloc[idx]
                e26   = data["Close"].ewm(span=26, adjust=False).mean().iloc[idx]
                macd_i = float(e12 - e26)
                sig_i  = float(data["Close"].ewm(span=12, adjust=False).mean().ewm(
                              span=9, adjust=False).mean().iloc[idx])
                mom_i  = float(
                    (data["Close"].iloc[idx] - data["Close"].iloc[idx - 10])
                    / data["Close"].iloc[idx - 10] * 100
                ) if abs(idx) < len(data) - 10 else 0.0
                avg_v  = float(data["Volume"].rolling(20).mean().iloc[idx])
                volr_i = float(data["Volume"].iloc[idx]) / avg_v if avg_v > 0 else 1.0
                vol_i  = float(data["Close"].pct_change().rolling(10).std().iloc[idx] * 100)
                seq_rows.append([rsi_i, macd_i, sig_i, mom_i, volr_i, vol_i])
            seq_matrix = np.array(seq_rows)  # (10, 6)
        else:
            seq_matrix = np.tile(feat_row, (SEQ_LEN, 1))  # fallback

        # Replace any NaN in seq_matrix with 0
        seq_matrix = np.nan_to_num(seq_matrix, nan=0.0)

        ai_probability, model_breakdown = ensemble_predict(feat_row, seq_matrix)
        targets = calculate_targets(stock, float(price), data)

        return {
            "symbol": stock,
            "price":  price,
            "rsi":    rsi,
            "macd":   float(macd),
            "macd_signal":  float(signal_line),
            "momentum":     momentum,
            "volume_ratio": volume_ratio,
            "volatility":   volatility,
            "ai_prediction":  int(ai_probability > 0.5),
            "ai_probability": ai_probability,
            "model_breakdown": model_breakdown,
            "news_title":    news_title,
            "sentiment_raw": sentiment_raw,
            "targets": targets,
        }

    except Exception as e:
        print(f"ERROR processing {stock}: {e}")
        return None

# =========================
# TOP STOCKS API
# =========================
@app.get("/top-stocks")
def top_stocks():

    results = []

    # Bulk download all stocks
    bulk_data = yf.download(stocks, period="3mo", interval="1d", auto_adjust=True, progress=False)

    def worker(stock):
        try:
            if stock not in bulk_data["Close"]:
                return None
            df = pd.DataFrame()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in bulk_data and stock in bulk_data[col]:
                    df[col] = bulk_data[col][stock]
            df = df.dropna()
            if df.empty:
                return None
            
            # NLP Sentiment Analysis
            news_title = "No recent news."
            sentiment_raw = 0.0
            try:
                news = yf.Ticker(stock).news
                if news and len(news) > 0 and 'title' in news[0]['content']:
                    news_title = news[0]['content']['title']
                    sentiment_raw = analyzer.polarity_scores(news_title)['compound']
            except Exception as e:
                print(f"[WARN] yfinance news failed for {stock}: {e}")

            return process_stock(stock, df, news_title, sentiment_raw)
        except Exception as e:
            print(f"ERROR extracting {stock}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = executor.map(worker, stocks)
        for result in futures:
            if result is not None:
                results.append(result)

    if not results:
        return {"success": False, "total_stocks": 0, "stocks": []}

    # =========================
    # NORMALIZE & RANK
    # =========================
    df_results = pd.DataFrame(results)

    def normalize(series, inverse=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            norm = pd.Series(0.5, index=series.index)
        else:
            norm = (series - s_min) / (s_max - s_min)
        return 1.0 - norm if inverse else norm

    norm_trend = normalize(df_results["macd"])
    norm_momentum = normalize(df_results["momentum"])
    norm_liquidity = normalize(df_results["volume_ratio"])
    norm_volatility = normalize(df_results["volatility"], inverse=True)
    norm_sentiment = normalize(df_results["sentiment_raw"])

    raw_ai = df_results["ai_probability"]
    
    # Dynamic AI weighting
    ai_weight = np.where(raw_ai > 0.65, 0.30 + (raw_ai - 0.65) * 0.5, 0.30)
    ai_weight = np.clip(ai_weight, 0.30, 0.45)
    
    rem_weight = 1.0 - ai_weight
    scale = rem_weight / 0.70

    score = (
        norm_momentum * (0.20 * scale) +
        norm_trend * (0.15 * scale) +
        norm_sentiment * (0.10 * scale) +
        norm_liquidity * (0.15 * scale) +
        norm_volatility * (0.10 * scale) +
        raw_ai * ai_weight
    ) * 100

    df_results["score"] = score.round(2)

    def get_signal(row):
        if row["score"] > 75 and row["ai_probability"] > 0.60:
            return "STRONG BUY"
        elif row["score"] > 60:
            return "BUY"
        elif row["score"] > 40:
            return "HOLD"
        else:
            return "SELL"

    df_results["signal"] = df_results.apply(get_signal, axis=1)

    # Format values for display
    df_results["price"] = df_results["price"].round(2)
    df_results["rsi"] = df_results["rsi"].round(2)
    df_results["macd"] = df_results["macd"].round(2)
    df_results["macd_signal"] = df_results["macd_signal"].round(2)
    df_results["momentum"] = df_results["momentum"].round(2)
    df_results["volume_ratio"] = df_results["volume_ratio"].round(2)
    df_results["volatility"] = df_results["volatility"].round(2)
    df_results["ai_probability"] = (df_results["ai_probability"] * 100).round(2)

    final_results = df_results.to_dict(orient="records")

    # =========================
    # SORT
    # =========================
    results = sorted(
        final_results,
        key=lambda x: x["score"],
        reverse=True
    )

    return clean_nans({
        "success": True,
        "total_stocks": len(results),
        "stocks": results
    })

# =========================
# BACKTEST API
# =========================
@app.get("/backtest")
def run_backtest():
    try:
        backtest_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        benchmark = "^NSEI"
        tickers = backtest_stocks + [benchmark]
        
        data = yf.download(tickers, period="2y", interval="1d", auto_adjust=True, progress=False)
        
        if data.empty or "Close" not in data:
            return {"success": False, "error": "No data downloaded"}
        
        close_prices = data["Close"].dropna(how="all")
        volume_data = data["Volume"] if "Volume" in data else pd.DataFrame(columns=tickers)
        
        stock_returns = pd.DataFrame(index=close_prices.index)
        
        for stock in backtest_stocks:
            if stock not in close_prices.columns:
                continue
                
            df = pd.DataFrame({
                "Close": close_prices[stock],
                "Volume": volume_data[stock] if stock in volume_data.columns else 0
            }).dropna()
            
            if df.empty or len(df) < 50:
                continue
                
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            rs = gain.rolling(14).mean() / loss.rolling(14).mean()
            df["RSI"] = 100 - (100 / (1 + rs))
            
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            
            df["Momentum"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100
            
            avg_vol = df["Volume"].rolling(20).mean()
            df["Volume_Ratio"] = df["Volume"] / avg_vol.replace(0, 1)
            
            df["Volatility"] = df["Close"].pct_change(fill_method=None).rolling(10).std() * 100
            
            df = df.dropna()
            if df.empty:
                continue
                
            features = df[["RSI", "MACD", "Signal", "Momentum", "Volume_Ratio", "Volatility"]]
            
            # Use ensemble or RF fallback for backtest
            if USE_ENSEMBLE:
                feat_sc = scaler.transform(features.values)
                probs   = xgb_model.predict_proba(feat_sc)[:, 1]
            else:
                probs = rf_model.predict_proba(features)[:, 1]
            df["AI_Prob"] = probs
            
            df["Buy_Signal"] = (df["AI_Prob"] > 0.60) & (df["Momentum"] > 0)
            
            df["Next_Ret"] = df["Close"].pct_change(fill_method=None).shift(-1)
            
            stock_returns[stock] = df["Next_Ret"] * df["Buy_Signal"].astype(int)
            
        port_ret = stock_returns.mean(axis=1).fillna(0)
        
        if benchmark in close_prices:
            bench_ret = close_prices[benchmark].pct_change(fill_method=None).shift(-1).fillna(0)
        else:
            bench_ret = pd.Series(0, index=port_ret.index)
            
        port_cum = (1 + port_ret).cumprod() * 100
        bench_cum = (1 + bench_ret).cumprod() * 100
        
        days = len(port_ret)
        years = days / 252.0
        
        port_final = port_cum.iloc[-2] if len(port_cum) > 1 else 100
        cagr = ((port_final / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        daily_vol = port_ret.std()
        ann_vol = daily_vol * np.sqrt(252)
        sharpe = (cagr / 100 - 0.05) / ann_vol if ann_vol > 0 else 0
        
        rolling_max = port_cum.cummax()
        drawdown = (port_cum - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        win_rate = (port_ret > 0).sum() / (port_ret != 0).sum() * 100 if (port_ret != 0).sum() > 0 else 0
        
        chart_data = []
        for date, p_val, b_val in zip(port_cum.index, port_cum, bench_cum):
            chart_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "portfolio": round(p_val, 2),
                "benchmark": round(b_val, 2)
            })
            
        return {
            "success": True,
            "metrics": {
                "cagr": round(cagr, 2),
                "sharpe": round(sharpe, 2),
                "max_drawdown": round(max_dd, 2),
                "win_rate": round(win_rate, 2)
            },
            "chart_data": chart_data[:-1]
        }
    except Exception as e:
        print("Backtest Error:", e)
        return {"success": False, "error": str(e)}


# ============================================================
# TRADING ENGINE
# ============================================================

PAPER_PORTFOLIO_PATH  = "data/paper_portfolio.json"
KITE_SESSION_PATH     = "data/kite_session.json"   # FIX 3: persistent token cache
KITE_API_KEY    = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_REDIRECT   = os.getenv("KITE_REDIRECT_URL", "http://127.0.0.1:8000/trading/callback")

# FIX 3: In-memory cache — now backed by file so it survives server restarts
_kite_instance   = None
_kite_access_tok = None


def _save_kite_token(access_token: str):
    """Persist the Zerodha access token to disk so it survives server restarts."""
    os.makedirs("data", exist_ok=True)
    with open(KITE_SESSION_PATH, "w") as f:
        json.dump({"access_token": access_token, "saved_at": datetime.now().isoformat()}, f)


def _load_kite_token() -> str | None:
    """Read a previously saved access token from disk, if it exists."""
    if not os.path.exists(KITE_SESSION_PATH):
        return None
    try:
        with open(KITE_SESSION_PATH, "r") as f:
            data = json.load(f)
        return data.get("access_token")
    except Exception:
        return None


def _clear_kite_token():
    """Remove the cached token (called when token validation fails)."""
    if os.path.exists(KITE_SESSION_PATH):
        os.remove(KITE_SESSION_PATH)


def get_kite() -> "KiteConnect | None":
    """Return an authenticated KiteConnect instance.

    Resolution order:
    1. In-memory instance (fast path — set during this server's lifetime).
    2. File-cached access token (FIX 3: survives server restarts).
    3. None — user must log in via the Kite login URL.

    Zerodha tokens expire at midnight IST; stale tokens are detected via a
    lightweight profile() call and cleared so the UI shows a re-login prompt.
    """
    global _kite_instance, _kite_access_tok
    if not KITE_AVAILABLE or not KITE_API_KEY:
        return None

    # Fast path: already validated in this server session
    if _kite_instance and _kite_access_tok:
        return _kite_instance

    # FIX 3: Try restoring session from file cache
    cached_token = _load_kite_token()
    if cached_token:
        try:
            kite = KiteConnect(api_key=KITE_API_KEY)
            kite.set_access_token(cached_token)
            kite.profile()          # lightweight call — raises if token expired
            _kite_instance   = kite
            _kite_access_tok = cached_token
            return _kite_instance
        except Exception:
            # Token has expired (midnight rollover) — clear stale cache
            _clear_kite_token()
            _kite_instance   = None
            _kite_access_tok = None

    return None


def _get_live_ltp(kite, symbol_ns: str) -> float | None:
    """FIX 1: Fetch real-time Last Traded Price from Zerodha.

    Uses kite.ltp() which returns the actual market price with zero delay,
    replacing the 15-minute delayed yfinance data that was previously used
    to determine the limit price for live broker orders.

    Args:
        kite: Authenticated KiteConnect instance.
        symbol_ns: yfinance-style ticker, e.g. "RELIANCE.NS".

    Returns:
        Float price, or None if the lookup failed.
    """
    symbol_clean = symbol_ns.replace(".NS", "").replace(".BO", "")
    exchange     = "BSE" if symbol_ns.endswith(".BO") else "NSE"
    kite_key     = f"{exchange}:{symbol_clean}"
    try:
        result = kite.ltp(kite_key)
        return float(result[kite_key]["last_price"])
    except Exception:
        return None


def _build_live_portfolio_state(kite) -> dict:
    """FIX 2: Build the portfolio state dict that RiskManager.evaluate_trade() expects.

    Fetches current holdings and open positions from Zerodha and converts them
    into the format:
        {
            "open_positions": ["RELIANCE.NS", ...],
            "allocations":    {"RELIANCE.NS": 0.08, ...},
            "sector_exposure":{"Energy": 0.08, ...},
        }

    Portfolio capital is derived from the total portfolio value via Kite margins.
    """
    try:
        margins    = kite.margins("equity")
        net_worth  = float(margins.get("net", 0)) or 1.0

        holdings   = kite.holdings()   # settled long-term positions
        positions  = kite.positions()  # intraday / short-term
        day_pos    = positions.get("day", []) + positions.get("net", [])

        open_positions: list[str] = []
        allocations:    dict[str, float] = {}
        sector_map     = RiskManager(portfolio_capital=1).sector_map
        sector_exposure: dict[str, float] = {}

        def _add(symbol_ns: str, value: float):
            if symbol_ns not in open_positions:
                open_positions.append(symbol_ns)
            allocations[symbol_ns] = allocations.get(symbol_ns, 0.0) + value / net_worth
            sector = sector_map.get(symbol_ns, "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + value / net_worth

        for h in holdings:
            qty = int(h.get("quantity", 0))
            ltp = float(h.get("last_price", 0))
            if qty > 0 and ltp > 0:
                sym = h.get("tradingsymbol", "") + ".NS"
                _add(sym, qty * ltp)

        for p in day_pos:
            qty = int(p.get("quantity", 0))
            avg = float(p.get("average_price", 0))
            if qty > 0 and avg > 0:
                sym = p.get("tradingsymbol", "") + ".NS"
                _add(sym, qty * avg)

        return {
            "open_positions":  open_positions,
            "allocations":     allocations,
            "sector_exposure": sector_exposure,
            "net_worth":       net_worth,
        }
    except Exception:
        # If we cannot read the live portfolio, return an empty state so risk
        # checks still run (they will use zero existing exposure).
        return {"open_positions": [], "allocations": {}, "sector_exposure": {}}


def _load_paper() -> dict:
    if not os.path.exists(PAPER_PORTFOLIO_PATH):
        return {"capital": 100000, "positions": [], "orders": []}
    with open(PAPER_PORTFOLIO_PATH, "r") as f:
        return json.load(f)


def _save_paper(data: dict):
    os.makedirs("data", exist_ok=True)
    with open(PAPER_PORTFOLIO_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Target Price Engine ────────────────────────────────────────
def calculate_targets(symbol: str, entry: float, data: pd.DataFrame) -> dict:
    """
    Returns ATR-based, Fibonacci, and Fixed-% targets + stop loss.
    """
    results = {"entry": round(entry, 2)}

    try:
        # ── 1. ATR-based (14-day Average True Range) ──────────────
        high = data["High"] if "High" in data.columns else data["Close"]
        low  = data["Low"]  if "Low"  in data.columns else data["Close"]
        close = data["Close"]

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        atr_target = round(float(entry + 3.0 * atr), 2)
        atr_sl     = round(float(entry - 1.5 * atr), 2)
        atr_rr     = round(float((atr_target - entry) / max(entry - atr_sl, 0.01)), 2)

        results["atr"] = {
            "target":     atr_target if pd.notna(atr_target) else round(entry * 1.06, 2),
            "stop_loss":  atr_sl if pd.notna(atr_sl) else round(entry * 0.97, 2),
            "risk_reward": atr_rr if pd.notna(atr_rr) else 2.0,
            "atr_value":  round(float(atr), 2) if pd.notna(atr) else round(entry * 0.02, 2)
        }
    except Exception:
        atr = entry * 0.02
        results["atr"] = {
            "target":     round(entry * 1.06, 2),
            "stop_loss":  round(entry * 0.97, 2),
            "risk_reward": 2.0,
            "atr_value":  round(atr, 2)
        }

    try:
        # ── 2. Fibonacci Extension (recent 20-day swing) ──────────
        recent = data.tail(20)
        swing_low  = float(recent["Low"].min()  if "Low"  in recent.columns else recent["Close"].min())
        swing_high = float(recent["High"].max() if "High" in recent.columns else recent["Close"].max())
        rng = swing_high - swing_low

        fib_127 = round(float(swing_high + 0.272 * rng), 2)
        fib_162 = round(float(swing_high + 0.618 * rng), 2)
        fib_sl  = round(float(entry - 0.382 * rng), 2)
        fib_rr  = round(float((fib_127 - entry) / max(entry - fib_sl, 0.01)), 2)

        results["fibonacci"] = {
            "target_127":  fib_127 if pd.notna(fib_127) else round(entry * 1.05, 2),
            "target_162":  fib_162 if pd.notna(fib_162) else round(entry * 1.10, 2),
            "stop_loss":   fib_sl if pd.notna(fib_sl) else round(entry * 0.96, 2),
            "swing_low":   round(float(swing_low), 2) if pd.notna(swing_low) else round(entry * 0.9, 2),
            "swing_high":  round(float(swing_high), 2) if pd.notna(swing_high) else round(entry * 1.05, 2),
            "risk_reward": fib_rr if pd.notna(fib_rr) else 1.67
        }
    except Exception:
        results["fibonacci"] = {
            "target_127":  round(entry * 1.05, 2),
            "target_162":  round(entry * 1.10, 2),
            "stop_loss":   round(entry * 0.96, 2),
            "swing_low":   round(entry * 0.9, 2),
            "swing_high":  round(entry * 1.05, 2),
            "risk_reward": 1.67
        }

    # ── 3. Fixed Percentage Tiers ─────────────────────────────────
    results["fixed"] = {
        "target_5pct":  round(entry * 1.05, 2),
        "target_10pct": round(entry * 1.10, 2),
        "target_15pct": round(entry * 1.15, 2),
        "stop_loss":    round(entry * 0.97, 2),
        "risk_reward_5":  round(0.05 / 0.03, 2),
        "risk_reward_10": round(0.10 / 0.03, 2),
        "risk_reward_15": round(0.15 / 0.03, 2),
    }

    return results


# ── Pydantic Request Models ─────────────────────────────────────
class BuyRequest(BaseModel):
    symbol: str           # e.g. "RELIANCE.NS"
    quantity: int
    mode: str = "paper"   # "paper" | "live"


class SellRequest(BaseModel):
    symbol: str
    quantity: int
    mode: str = "paper"


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/trading/config")
def trading_config():
    """Returns Kite login URL and connection status."""
    kite = get_kite()
    connected = kite is not None

    login_url = None
    if KITE_AVAILABLE and KITE_API_KEY and not connected:
        temp_kite = KiteConnect(api_key=KITE_API_KEY)
        login_url = temp_kite.login_url()

    return {
        "kite_available": KITE_AVAILABLE,
        "api_key_set":    bool(KITE_API_KEY),
        "connected":      connected,
        "login_url":      login_url
    }


@app.get("/trading/callback")
def trading_callback(request_token: str = Query(...)):
    """OAuth2 callback — exchanges request_token for access_token.

    FIX 3: After a successful handshake, persist the access token to disk so
    the session is automatically restored after a server restart without
    requiring the user to log in again (until Zerodha expires it at midnight).
    """
    global _kite_instance, _kite_access_tok
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        _kite_access_tok = data["access_token"]
        kite.set_access_token(_kite_access_tok)
        _kite_instance = kite
        # FIX 3: Persist token to disk for post-restart session recovery
        _save_kite_token(_kite_access_tok)
        # Redirect back to the frontend after successful auth
        return RedirectResponse("http://localhost:5174/?kite=connected")
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/trading/buy")
def trading_buy(req: BuyRequest):
    """Place a paper or live buy order."""
    try:
        symbol_clean = req.symbol.replace(".NS", "").replace(".BO", "")
        # Fetch current price + OHLCV for target calculation
        raw = yf.download(req.symbol, period="3mo", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if raw.empty or raw["Close"].dropna().empty:
            return {"success": False, "error": "Could not fetch price data"}

        entry = float(raw["Close"].dropna().iloc[-1])
        targets = calculate_targets(req.symbol, entry, raw)

        order_record = {
            "id":           f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symbol":       symbol_clean,
            "full_symbol":  req.symbol,
            "quantity":     req.quantity,
            "entry_price":  entry,
            "mode":         req.mode,
            "type":         "BUY",
            "timestamp":    datetime.now().isoformat(),
            "targets":      targets,
            "status":       "OPEN"
        }

        if req.mode == "paper":
            portfolio = _load_paper()
            cost = entry * req.quantity
            if cost > portfolio["capital"]:
                return {"success": False, "error": f"Insufficient paper capital. Need {cost:.2f}, have {portfolio['capital']:.2f}"}

            portfolio["capital"] -= cost

            # Check if position exists, if so add to it
            existing = next((p for p in portfolio["positions"] if p["full_symbol"] == req.symbol), None)
            if existing:
                total_qty  = existing["quantity"] + req.quantity
                avg_entry  = (existing["entry_price"] * existing["quantity"] + entry * req.quantity) / total_qty
                existing["quantity"]    = total_qty
                existing["entry_price"] = round(avg_entry, 2)
                existing["targets"]     = calculate_targets(req.symbol, avg_entry, raw)
            else:
                portfolio["positions"].append({
                    "symbol":       symbol_clean,
                    "full_symbol":  req.symbol,
                    "quantity":     req.quantity,
                    "entry_price":  round(entry, 2),
                    "targets":      targets,
                    "timestamp":    datetime.now().isoformat()
                })

            portfolio["orders"].append(order_record)
            _save_paper(portfolio)
            return clean_nans({"success": True, "mode": "paper", "order": order_record})

        elif req.mode == "live":
            kite = get_kite()
            if not kite:
                return {"success": False, "error": "Zerodha not connected. Please authenticate first."}

            # FIX 2: Risk gate — evaluate trade against institutional risk limits
            # before touching the broker. Rejected trades never reach Zerodha.
            live_state   = _build_live_portfolio_state(kite)
            net_worth    = live_state.get("net_worth", 100000)
            trade_value  = entry * req.quantity          # approx. capital to commit
            risk_manager = RiskManager(portfolio_capital=net_worth)
            risk_result  = risk_manager.evaluate_trade(
                symbol             = req.symbol,
                intended_capital   = trade_value,
                current_portfolio  = live_state,
            )
            if not risk_result["approved"]:
                return {
                    "success": False,
                    "error":   f"Risk gate rejected order: {risk_result['reason']}",
                    "risk":    risk_result,
                }

            # FIX 1: Use real-time LTP from Zerodha instead of 15-min delayed
            # yfinance data. Fall back to the yfinance price only if kite.ltp() fails.
            live_ltp = _get_live_ltp(kite, req.symbol)
            if live_ltp is not None:
                entry = live_ltp
                order_record["entry_price"] = entry

            # Place a LIMIT buy order 0.1% above real-time LTP for fill probability
            limit_price = round(entry * 1.001, 2)
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=symbol_clean,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=req.quantity,
                product=kite.PRODUCT_CNC,
                order_type=kite.ORDER_TYPE_LIMIT,
                price=limit_price,
                validity=kite.VALIDITY_DAY
            )
            order_record["kite_order_id"] = order_id
            order_record["limit_price"]   = limit_price
            order_record["risk_approved"] = True
            order_record["recommended_sl_pct"] = risk_result.get("recommended_stop_loss_pct", 0.05)
            return clean_nans({"success": True, "mode": "live", "order_id": order_id, "order": order_record})

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/trading/sell")
def trading_sell(req: SellRequest):
    """Close a paper or live position."""
    try:
        symbol_clean = req.symbol.replace(".NS", "").replace(".BO", "")
        raw = yf.download(req.symbol, period="5d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if raw.empty or raw["Close"].dropna().empty:
            return {"success": False, "error": "Could not fetch price data"}
        exit_price = float(raw["Close"].dropna().iloc[-1])

        if req.mode == "paper":
            portfolio = _load_paper()
            pos = next((p for p in portfolio["positions"] if p["full_symbol"] == req.symbol), None)
            if not pos:
                return {"success": False, "error": f"No open paper position for {req.symbol}"}

            sell_qty = min(req.quantity, pos["quantity"])
            pnl      = (exit_price - pos["entry_price"]) * sell_qty
            portfolio["capital"] += exit_price * sell_qty

            if sell_qty >= pos["quantity"]:
                portfolio["positions"].remove(pos)
            else:
                pos["quantity"] -= sell_qty

            portfolio["orders"].append({
                "id":         f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "symbol":     symbol_clean,
                "full_symbol": req.symbol,
                "quantity":   sell_qty,
                "exit_price": round(exit_price, 2),
                "entry_price": round(pos["entry_price"], 2),
                "pnl":        round(pnl, 2),
                "mode":       "paper",
                "type":       "SELL",
                "timestamp":  datetime.now().isoformat(),
                "status":     "CLOSED"
            })
            _save_paper(portfolio)
            return clean_nans({"success": True, "pnl": round(pnl, 2), "exit_price": round(exit_price, 2)})

        elif req.mode == "live":
            kite = get_kite()
            if not kite:
                return {"success": False, "error": "Zerodha not connected."}

            # FIX 2: Risk gate for sell orders — ensure the position actually exists
            # in the live portfolio and we are not over-selling.
            live_state  = _build_live_portfolio_state(kite)
            open_syms   = live_state.get("open_positions", [])
            if req.symbol not in open_syms:
                return {
                    "success": False,
                    "error":   f"Risk gate: No live position found for {req.symbol}. "
                               "Cannot sell a position you do not hold.",
                }

            # FIX 1: Fetch real-time exit price from Zerodha LTP
            live_ltp = _get_live_ltp(kite, req.symbol)
            if live_ltp is not None:
                exit_price = live_ltp

            # Place a LIMIT sell order 0.1% below real-time LTP for fill probability
            limit_price = round(exit_price * 0.999, 2)
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=symbol_clean,
                transaction_type=kite.TRANSACTION_TYPE_SELL,
                quantity=req.quantity,
                product=kite.PRODUCT_CNC,
                order_type=kite.ORDER_TYPE_LIMIT,
                price=limit_price,
                validity=kite.VALIDITY_DAY
            )
            return clean_nans({"success": True, "mode": "live", "order_id": order_id, "exit_price": limit_price})

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/trading/positions")
def trading_positions(mode: str = "paper"):
    """Active positions with live P&L."""
    try:
        if mode == "paper":
            portfolio = _load_paper()
            positions = portfolio.get("positions", [])
            enriched = []
            for pos in positions:
                try:
                    raw = yf.download(pos["full_symbol"], period="2d", progress=False, auto_adjust=True)
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.droplevel(1)
                    if not raw.empty and not raw["Close"].dropna().empty:
                        current = float(raw["Close"].dropna().iloc[-1])
                    else:
                        current = pos["entry_price"]
                except Exception:
                    current = pos["entry_price"]

                pnl     = (current - pos["entry_price"]) * pos["quantity"]
                pnl_pct = (current - pos["entry_price"]) / pos["entry_price"] * 100
                enriched.append({
                    **pos,
                    "current_price": round(current, 2),
                    "pnl":           round(pnl, 2),
                    "pnl_pct":       round(pnl_pct, 2),
                    "market_value":  round(current * pos["quantity"], 2)
                })
            return clean_nans({
                "success":    True,
                "mode":       "paper",
                "capital":    round(portfolio.get("capital", 0), 2),
                "positions":  enriched,
                "total_pnl":  round(sum(p["pnl"] for p in enriched), 2)
            })

        elif mode == "live":
            kite = get_kite()
            if not kite:
                return {"success": False, "error": "Not connected to Zerodha"}
            positions = kite.positions()
            return clean_nans({"success": True, "mode": "live", "positions": positions})

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/trading/orders")
def trading_orders(mode: str = "paper"):
    """Order history."""
    try:
        if mode == "paper":
            portfolio = _load_paper()
            return clean_nans({"success": True, "mode": "paper", "orders": portfolio.get("orders", [])})
        elif mode == "live":
            kite = get_kite()
            if not kite:
                return {"success": False, "error": "Not connected to Zerodha"}
            return clean_nans({"success": True, "mode": "live", "orders": kite.orders()})
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/trading/targets/{symbol}")
def get_targets(symbol: str):
    """Standalone target price calculation for any NSE symbol."""
    try:
        sym_ns = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        raw = yf.download(sym_ns, period="3mo", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if raw.empty:
            return {"success": False, "error": "No data"}
        entry = float(raw["Close"].iloc[-1])
        return {"success": True, "symbol": symbol, "targets": calculate_targets(sym_ns, entry, raw)}
    except Exception as e:
        return {"success": False, "error": str(e)}

# =========================
# RISK MANAGEMENT API
# =========================
@app.get("/risk/config")
def get_risk_config():
    """Returns the current risk parameters configured in the Risk Engine."""
    try:
        # Instantiate RiskManager (dummy capital just to read defaults)
        rm = RiskManager(portfolio_capital=100000)
        return {
            "success": True,
            "profile": "Balanced",
            "parameters": {
                "max_position_size_pct": rm.max_position_size * 100,
                "max_daily_var_pct": rm.max_daily_var * 100,
                "max_open_positions": rm.max_open_positions,
                "default_stop_loss_pct": rm.default_stop_loss * 100,
                "max_sector_exposure_pct": rm.max_sector_exposure * 100,
                "cash_reserve_pct": rm.cash_reserve * 100
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/portfolio/optimize")
def get_portfolio_optimization():
    """Runs Markowitz Mean-Variance optimization."""
    try:
        # We will run this on the main active stocks in the universe
        opt_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "LT.NS"]
        optimizer = PortfolioOptimizer(symbols=opt_stocks, max_position_size=0.15, cash_reserve=0.10)
        result = optimizer.optimize()
        if not result:
            return {"success": False, "error": "Optimizer failed to find a valid allocation."}
        
        return {
            "success": True,
            "allocation": result["allocation"],
            "metrics": result["metrics"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# =========================
# CHART DATA API
# =========================
@app.get("/chart-data/{symbol}")
def get_chart_data(symbol: str, period: str = "3M"):
    try:
        period_map = {
            "1W": ("5d", "15m"),
            "1M": ("1mo", "1h"),
            "3M": ("3mo", "1d"),
            "6M": ("6mo", "1d"),
            "1Y": ("1y", "1d"),
        }
        yf_period, yf_interval = period_map.get(period, ("3mo", "1d"))
        
        data = yf.download(symbol, period=yf_period, interval=yf_interval, progress=False)
        if data.empty:
            return {"success": False, "error": "No data found"}
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        data = data.dropna()
        
        result_data = []
        for index, row in data.iterrows():
            if yf_interval in ["15m", "1h"]:
                ts = int(index.timestamp())
            else:
                ts = int(pd.Timestamp(index.date()).timestamp())
                
            result_data.append({
                "time": ts,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
            
        return {"success": True, "symbol": symbol, "period": period, "data": result_data}
    except Exception as e:
        return {"success": False, "error": str(e)}

# =========================
# STOCK SEARCH API
# =========================
@app.get("/search-stocks")
def search_stocks(query: str = ""):
    if not query:
        return {"results": []}
        
    stocks_data = load_nse_stocks()
    q = query.lower()
    
    results = []
    for stock in stocks_data:
        if q in stock["symbol"].lower() or q in stock["name"].lower():
            results.append({
                "symbol": stock["symbol"],
                "name": stock["name"],
                "sector": stock.get("sector", ""),
                "full_symbol": f"{stock['symbol']}.NS"
            })
            if len(results) >= 20:
                break
                
    return {"results": results}

# =========================
# FEATURE STORE API
# =========================
@app.get("/data/features")
def get_feature_store():
    """Returns the historical OHLCV data merged with calculated technical indicators."""
    db = SessionLocal()
    try:
        # Fetch the last 100 records for the UI to prevent overwhelming the browser
        records = db.query(StockData, FeatureStore).join(
            FeatureStore, StockData.id == FeatureStore.stock_data_id
        ).order_by(StockData.date.desc(), StockData.symbol).limit(100).all()
        
        results = []
        for stock, feature in records:
            row = {
                "id": stock.id,
                "symbol": stock.symbol,
                "date": stock.date.isoformat() if hasattr(stock.date, 'isoformat') else str(stock.date),
                "open": stock.open,
                "high": stock.high,
                "low": stock.low,
                "close": stock.close,
                "volume": stock.volume,
            }
            # Merge JSONB indicators directly into the row
            if feature.indicators and isinstance(feature.indicators, dict):
                for k, v in feature.indicators.items():
                    # Format float keys to 2 decimals for cleaner UI
                    row[k] = round(v, 4) if isinstance(v, float) else v
            results.append(row)
            
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close()