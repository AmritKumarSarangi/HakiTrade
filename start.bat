@echo off
echo Starting HakiTrade Quant Platform...

echo Starting Backend...
start "HakiTrade Backend" cmd /k "venv\Scripts\activate && uvicorn app.main:app --reload"

echo Starting Frontend...
start "HakiTrade Frontend" cmd /k "cd frontend && npm run dev"

echo Both processes have been started in new windows.
