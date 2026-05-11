"""
HakiTrade Multi-Factor Ensemble Training
=========================================
Models:  XGBoost · LightGBM · LSTM · Transformer
Meta:    Logistic Regression (stacking)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("models", exist_ok=True)

STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS"
]
SEQ_LEN   = 10   # lookback window for LSTM / Transformer
N_FOLDS   = 5
FEATURES  = ["RSI", "MACD", "Signal", "Momentum", "Volume_Ratio", "Volatility"]

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA COLLECTION & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
def build_features(df):
    """Return df with 6 technical features + Target.  Drops NaN rows."""
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    rs    = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Momentum"]     = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Volatility"]   = df["Close"].pct_change().rolling(10).std() * 100

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df[FEATURES + ["Target"]].dropna()


print("=" * 60)
print("  HakiTrade Ensemble Trainer")
print("=" * 60)

all_flat = []   # rows for XGB / LGB
all_seq  = []   # sequences for LSTM / Transformer
all_y    = []   # labels

for sym in STOCKS:
    print(f"\n  Downloading {sym} (5 years)…")
    raw = yf.download(sym, period="5y", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    if raw.empty or len(raw) < 60:
        print(f"  Skipping — insufficient data.")
        continue

    df = build_features(raw)
    print(f"  {len(df)} rows ready.")

    X_raw = df[FEATURES].values
    y_raw = df["Target"].values

    # --- flat records (XGB / LGB) ---
    all_flat.append(X_raw)
    all_y.append(y_raw)

    # --- overlapping sequences (LSTM / Transformer) ---
    for i in range(SEQ_LEN, len(X_raw)):
        all_seq.append(X_raw[i - SEQ_LEN : i])   # shape (10, 6)

X_flat = np.vstack(all_flat)
y_flat = np.concatenate(all_y)

# align sequences to flat labels (seq uses last row as label)
X_seq = np.array(all_seq)          # (N, 10, 6)
y_seq = y_flat[SEQ_LEN:][:len(X_seq)]  # match length

print(f"\n  Flat rows  : {len(X_flat)}")
print(f"  Sequences  : {len(X_seq)}")

# scale flat features
scaler = StandardScaler()
X_flat_sc = scaler.fit_transform(X_flat)

# scale sequences using same scaler
X_seq_sc = X_seq.reshape(-1, 6)
X_seq_sc = scaler.transform(X_seq_sc).reshape(-1, SEQ_LEN, 6)

joblib.dump(scaler, "models/scaler.pkl")
print("  Scaler saved.")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  PyTorch Model Definitions
# ──────────────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


class TransformerModel(nn.Module):
    def __init__(self, input_size=6, d_model=32, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.embed  = nn.Linear(input_size, d_model)
        enc_layer   = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                                  dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head   = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(1)


def train_torch_model(model, X_tr, y_tr, X_val, y_val, epochs=30, lr=1e-3):
    """Train a PyTorch model and return val probabilities."""
    device = torch.device("cpu")
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)

    ds     = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = model(Xv).numpy()
    return probs


# ──────────────────────────────────────────────────────────────────────────────
# 3.  5-FOLD OOF PREDICTIONS  (XGB, LGB, LSTM, Transformer)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Cross-validation OOF pass")
print("=" * 60)

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb  = np.zeros(len(X_flat))
oof_lgb  = np.zeros(len(X_flat))
oof_lstm = np.zeros(len(X_seq))
oof_tfm  = np.zeros(len(X_seq))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_flat_sc, y_flat), 1):
    print(f"\n  Fold {fold}/{N_FOLDS}")

    # --- XGBoost ---
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
        n_jobs=-1, verbosity=0
    )
    xgb_m.fit(X_flat_sc[tr_idx], y_flat[tr_idx],
               eval_set=[(X_flat_sc[val_idx], y_flat[val_idx])],
               verbose=False)
    oof_xgb[val_idx] = xgb_m.predict_proba(X_flat_sc[val_idx])[:, 1]
    print(f"    XGB AUC={roc_auc_score(y_flat[val_idx], oof_xgb[val_idx]):.4f}")

    # --- LightGBM ---
    lgb_m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_m.fit(X_flat_sc[tr_idx], y_flat[tr_idx],
               callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)],
               eval_set=[(X_flat_sc[val_idx], y_flat[val_idx])])
    oof_lgb[val_idx] = lgb_m.predict_proba(X_flat_sc[val_idx])[:, 1]
    print(f"    LGB AUC={roc_auc_score(y_flat[val_idx], oof_lgb[val_idx]):.4f}")

    # Seq fold alignment  (same ratio as flat)
    seq_tr  = tr_idx[tr_idx >= SEQ_LEN] - SEQ_LEN
    seq_val = val_idx[val_idx >= SEQ_LEN] - SEQ_LEN
    seq_tr  = seq_tr[seq_tr < len(X_seq)]
    seq_val = seq_val[seq_val < len(X_seq)]

    if len(seq_tr) == 0 or len(seq_val) == 0:
        continue

    # --- LSTM ---
    lstm_m = LSTMModel()
    lp = train_torch_model(
        lstm_m,
        X_seq_sc[seq_tr], y_seq[seq_tr].astype(np.float32),
        X_seq_sc[seq_val], y_seq[seq_val].astype(np.float32),
        epochs=30
    )
    oof_lstm[seq_val] = lp
    print(f"    LSTM AUC={roc_auc_score(y_seq[seq_val], lp):.4f}")

    # --- Transformer ---
    tfm_m = TransformerModel()
    tp = train_torch_model(
        tfm_m,
        X_seq_sc[seq_tr], y_seq[seq_tr].astype(np.float32),
        X_seq_sc[seq_val], y_seq[seq_val].astype(np.float32),
        epochs=30
    )
    oof_tfm[seq_val] = tp
    print(f"    Transformer AUC={roc_auc_score(y_seq[seq_val], tp):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FINAL FULL-DATA TRAINING  (save production models)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training final models on full data")
print("=" * 60)

# XGBoost
print("  XGBoost…")
xgb_final = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
)
xgb_final.fit(X_flat_sc, y_flat)
joblib.dump(xgb_final, "models/xgb.pkl")
print("    Saved models/xgb.pkl")

# LightGBM
print("  LightGBM…")
lgb_final = lgb.LGBMClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_final.fit(X_flat_sc, y_flat)
joblib.dump(lgb_final, "models/lgb.pkl")
print("    Saved models/lgb.pkl")

# LSTM (full data)
print("  LSTM…")
lstm_final = LSTMModel()
train_torch_model(
    lstm_final,
    X_seq_sc, y_seq.astype(np.float32),
    X_seq_sc, y_seq.astype(np.float32),
    epochs=40
)
torch.save(lstm_final.state_dict(), "models/lstm.pt")
print("    Saved models/lstm.pt")

# Transformer (full data)
print("  Transformer…")
tfm_final = TransformerModel()
train_torch_model(
    tfm_final,
    X_seq_sc, y_seq.astype(np.float32),
    X_seq_sc, y_seq.astype(np.float32),
    epochs=40
)
torch.save(tfm_final.state_dict(), "models/transformer.pt")
print("    Saved models/transformer.pt")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  META-LEARNER  (stack OOF probs → LogReg)
# ──────────────────────────────────────────────────────────────────────────────
print("\n  Training meta-learner…")

# align sequences to flat index
offset    = SEQ_LEN
n_seq     = len(X_seq)
n_flat    = len(X_flat)

# Take only the overlapping portion
oof_stack = np.column_stack([
    oof_xgb[offset:offset + n_seq],
    oof_lgb[offset:offset + n_seq],
    oof_lstm[:n_seq],
    oof_tfm[:n_seq]
])
y_meta    = y_flat[offset:offset + n_seq]

meta = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
meta.fit(oof_stack, y_meta)
joblib.dump(meta, "models/meta.pkl")
print("    Saved models/meta.pkl")

print("\n" + "=" * 60)
print("  ✅  ENSEMBLE TRAINING COMPLETE")
print("=" * 60)
print("  Models saved to models/")
print("    • xgb.pkl")
print("    • lgb.pkl")
print("    • lstm.pt")
print("    • transformer.pt")
print("    • meta.pkl")
print("    • scaler.pkl")
print(f"\n  Meta-learner weights: {dict(zip(['XGB','LGB','LSTM','TFM'], meta.coef_[0].round(3)))}")