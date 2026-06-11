# ⚡ HakiTrade — Institutional AI Quant Platform

HakiTrade is a sophisticated, real-time quantitative trading dashboard powered by machine learning. It provides institutional-grade stock analysis, AI-ranked signals, and a seamless trading interface for both paper and live execution.

![HakiTrade Dashboard](https://github.com/placeholder-image.png)

## 🌟 Key Features

- **🧠 Ensemble AI Engine**: Leverages XGBoost, LightGBM, LSTM, and Transformer models to predict stock price movements with high conviction.
- **📊 Real-time NSE Analytics**: Live tracking of 40+ high-liquidity NSE stocks with technical indicators like RSI, MACD, and Volume Ratios.
- **🎯 Dynamic AI Targets**: Automated calculation of Exit/Entry targets using ATR-based volatility, Fibonacci Extensions, and Fixed Profit/Loss ratios.
- **📱 Premium Trading Interface**: A glassmorphism-inspired UI for managing active positions, order history, and portfolio P&L.
- **📄 Paper & Live Trading**: Seamlessly toggle between risk-free simulation (₹100,000 starting capital) and live execution via Zerodha Kite Connect.
- **🔒 Secure Access**: Admin-level authentication gateway to protect your proprietary trading data.

## 🚀 Tech Stack

- **Frontend**: React.js, Vite, Axios, Recharts (Modern UI/UX with Glassmorphism)
- **Backend**: FastAPI (Python), Uvicorn, Concurrent Threading
- **Machine Learning**: PyTorch (LSTM/Transformers), XGBoost, LightGBM, Joblib
- **Data & APIs**: Yahoo Finance (Market Data), VADER (NLP Sentiment Analysis), Zerodha Kite Connect (Trading)

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Zerodha Kite API](https://developers.kite.trade/) Credentials (optional for live trading)

### 1. Clone & Install Backend
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Frontend
```bash
cd frontend
npm install
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
KITE_REDIRECT_URL=http://127.0.0.1:8000/trading/callback
ADMIN_PASSWORD=your_secure_password
```

### 4. Run the Platform
**Start Backend:**
```bash
uvicorn app.main:app --reload
```
**Start Frontend:**
```bash
cd frontend
npm run dev
```

## 🌐 Deployment

### Backend (FastAPI)
Deploy to **Render**, **Railway**, or **AWS EC2**.
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Frontend (React)
Deploy to **Vercel** or **Netlify**.
- Build Command: `npm run build`
- Output Directory: `dist`

## ⚖️ Disclaimer
*HakiTrade is for educational and research purposes. Quantitative trading involves significant risk. Always test strategies thoroughly in paper mode before committing real capital.*

---
Built with ⚡ by Antigravity AI
