# ⚡ HakiTrade — Institutional AI Quant Platform

HakiTrade is a highly sophisticated, real-time quantitative trading ecosystem powered by ensemble machine learning, distributed data pipelines, and a modern web stack. It bridges the gap between institutional-grade quantitative finance and modern full-stack web development.

![HakiTrade Dashboard](frontend/src/assets/hero.png)

## 🌟 The 12-Module Ecosystem

1. **🧠 Ensemble AI Engine**: Aggregates predictive probability scoring from deep learning (PyTorch LSTMs & Transformers) and gradient boosting (XGBoost & LightGBM) via a Logistic Regression meta-learner.
2. **🗞️ Sentiment Intensity Analyzer**: Dynamic NLP pipeline utilizing VADER to extract live market sentiment from breaking financial news headlines.
3. **📊 Backtest Engine**: Vectorized Pandas simulator that validates strategy historical performance (CAGR, Sharpe Ratio, Max Drawdown) against the NIFTY 50 benchmark.
4. **⚖️ Portfolio Optimizer**: Uses Markowitz Mean-Variance math (via `SciPy`) to compute covariance matrices and output the capital allocation that mathematically maximizes the Sharpe Ratio.
5. **🛡️ Risk Management Engine**: Enforces strict institutional constraints, including 1.5% Max Daily VaR (95% confidence), 10% position caps, and 25% sector exposure limits.
6. **📱 Algorithmic Trading Engine**: Automated execution router that seamlessly toggles between zero-risk Paper Trading and Live Broker Execution via the Zerodha Kite Connect API.
7. **🎯 Target Price Calculator**: Calculates dynamic exit/entry pricing using 14-day Average True Range (ATR), Fibonacci Extensions, and Fixed-Percentage scaling.
8. **⚙️ Automated ETL Pipeline**: Distributed `Celery` and `Redis` background worker ecosystem scheduled to ingest End-Of-Day market data daily at 4:15 PM IST.
9. **🗄️ PostgreSQL Feature Store**: Infinitely scalable JSONB database schema storing raw OHLCV prices alongside 20+ dynamically engineered technical indicators.
10. **📈 Multi-Factor Signal Generator**: High-performance ranking algorithm that synthesizes AI probabilities, momentum metrics, volatility, and sentiment into actionable "STRONG BUY" / "SELL" signals.
11. **⚡ High-Performance API**: Concurrent, asynchronous `FastAPI` backend engineered for rapid quantitative calculations.
12. **💻 Institutional Web Dashboard**: Dockerized `React` + `Vite` frontend, reverse-proxied by `Nginx`, featuring a sleek glassmorphism UI, Recharts analytics, and real-time portfolio tracking.

---

## 🚀 Tech Stack

- **Frontend:** React.js, Vite, Axios, Recharts
- **Backend:** FastAPI (Python), Uvicorn, Celery, Redis
- **Database:** PostgreSQL (SQLAlchemy ORM)
- **Machine Learning:** PyTorch, XGBoost, LightGBM, scikit-learn, VADER NLP
- **DevOps & Infrastructure:** Docker, Docker Compose, Nginx

---

## 🛠️ Deployment & Setup (Docker Compose)

HakiTrade is completely containerized. Deploying it locally or to a Production Cloud VPS (DigitalOcean/AWS) requires only a single command.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
- Git

### 1. Clone & Configure
```bash
git clone https://github.com/YourUsername/HakiTrade.git
cd HakiTrade
```

Create a `.env` file in the root directory. Add your database credentials and optional Zerodha keys:
```env
# Database & Redis
DATABASE_URL=postgresql://user:password@db:5432/hakitrade
CELERY_BROKER_URL=redis://redis:6379/0

# Live Trading (Optional)
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
KITE_REDIRECT_URL=http://your_domain_or_ip/api/trading/callback
```

### 2. Launch the Ecosystem
Run the pre-deployment tests and spin up the entire multi-container stack (FastAPI, React, Nginx, Celery, Redis, Postgres):

```bash
docker-compose up -d --build
```

### 3. Access the Platform
- **Dashboard:** `http://localhost` (or your Server's Public IP)
- **API Swagger Docs:** `http://localhost/api/docs`

---

## 🧠 Training the Models
The AI models are pre-trained and saved in the `/models` directory. To re-train the models on the latest 5-year historical data:
```bash
python train_model.py
```

## ⚖️ Disclaimer
*HakiTrade is built for educational and research purposes. Quantitative trading involves significant risk. Always test strategies thoroughly in paper mode before committing real capital.*

Built with ⚡ by Antigravity AI
