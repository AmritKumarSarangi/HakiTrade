import { useEffect, useState, useCallback } from "react";
import axios from "axios";
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Cell, Area, AreaChart
} from "recharts";
import "./App.css";

// ── helpers ──────────────────────────────────────────────────────────────────
const shortSymbol = (s) => s.replace(".NS", "");

const signalClass = (s) => {
  if (!s) return "";
  if (s === "STRONG BUY") return "strong-buy";
  if (s === "BUY") return "buy";
  if (s === "HOLD") return "hold";
  return "sell";
};

const signalDot = (s) => {
  if (s === "STRONG BUY") return "▲▲";
  if (s === "BUY") return "▲";
  if (s === "HOLD") return "◆";
  return "▼";
};

const rsiClass = (v) => {
  if (v >= 70) return "overbought";
  if (v <= 30) return "oversold";
  return "neutral";
};

const scoreColor = (score) => {
  if (score > 75) return "#10b981";
  if (score > 60) return "#34d399";
  if (score > 40) return "#f59e0b";
  return "#ef4444";
};

const sentimentText = (s) => {
  if (s >= 0.15) return "Bullish";
  if (s <= -0.15) return "Bearish";
  return "Neutral";
};

const sentimentClass = (s) => {
  if (s >= 0.15) return "up";
  if (s <= -0.15) return "down";
  return "neutral";
};

const fmt = (n, d = 2) => (typeof n === "number" ? n.toFixed(d) : "—");
const fmtPrice = (n) => n ? `₹${n.toLocaleString("en-IN", { minimumFractionDigits: 2 })}` : "—";

// Custom Recharts tooltip
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0d1526", border: "1px solid rgba(99,179,237,0.2)",
      borderRadius: 10, padding: "10px 14px", fontFamily: "'JetBrains Mono',monospace",
      fontSize: 12, color: "#f0f4ff"
    }}>
      <div style={{ color: "#8899bb", marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || "#3b82f6" }}>
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

// ── Skeleton rows ─────────────────────────────────────────────────────────────
function SkeletonRows() {
  return Array.from({ length: 5 }).map((_, i) => (
    <tr key={i}>
      {[80, 70, 60, 80, 90, 70].map((w, j) => (
        <td key={j} style={{ padding: "14px 10px" }}>
          <div className="skeleton" style={{ width: `${w}%`, height: 14 }} />
        </td>
      ))}
    </tr>
  ));
}

// ── Model Breakdown ──────────────────────────────────────────────────────────
function ModelBreakdown({ breakdown }) {
  if (!breakdown || breakdown.xgb === null) return <span style={{ color: "#8899bb", fontSize: 11 }}>RF Fallback</span>;
  const models = [
    { key: "xgb",         label: "XGB",    color: "#3b82f6" },
    { key: "lgb",         label: "LGB",    color: "#8b5cf6" },
    { key: "lstm",        label: "LSTM",   color: "#10b981" },
    { key: "transformer", label: "TFM",    color: "#f59e0b" },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 120 }}>
      {models.map(m => (
        <div key={m.key} style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ fontSize: 9, color: "#8899bb", width: 26, textAlign: "right" }}>{m.label}</span>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", borderRadius: 3, height: 5, overflow: "hidden" }}>
            <div style={{ width: `${breakdown[m.key]}%`, background: m.color, height: "100%", borderRadius: 3, transition: "width 0.6s ease" }} />
          </div>
          <span style={{ fontSize: 9, color: m.color, width: 30 }}>{breakdown[m.key]}%</span>
        </div>
      ))}
    </div>
  );
}

// ── Ticker ────────────────────────────────────────────────────────────────────
function TickerBar({ stocks }) {
  if (!stocks.length) return null;
  const items = [...stocks, ...stocks];
  return (
    <div className="ticker-bar">
      <div className="ticker-track">
        {items.map((s, i) => {
          const up = s.momentum >= 0;
          return (
            <span className="ticker-item" key={i}>
              <span className="ticker-symbol">{shortSymbol(s.symbol)}</span>
              <span className="ticker-price">{fmtPrice(s.price)}</span>
              <span className={`ticker-change ${up ? "up" : "down"}`}>
                {up ? "+" : ""}{fmt(s.momentum)}%
              </span>
              <span style={{ color: "#1e2a40" }}>|</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ── Backtest View ─────────────────────────────────────────────────────────────
function BacktestView() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchBacktest() {
      try {
        const res = await axios.get("http://127.0.0.1:8000/backtest");
        if (res.data.success) {
          setData(res.data);
        } else {
          setError(res.data.error || "Failed to fetch backtest");
        }
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    fetchBacktest();
  }, []);

  if (loading) return <div className="empty-state">Running AI Simulation... (Takes ~2s)</div>;
  if (error) return <div className="error-state">⚠️ {error}</div>;
  if (!data) return null;

  const m = data.metrics;
  return (
    <div className="backtest-view" style={{ display: 'flex', flexDirection: 'column', gap: 24, marginTop: 16 }}>
      <div className="stats-row">
        <div className="stat-card green">
          <div className="stat-card-header"><span className="stat-label">CAGR</span><div className="stat-icon green">📈</div></div>
          <div className="stat-value">{m.cagr}%</div>
          <div className="stat-meta"><span className="stat-meta-text">Annualized Return</span></div>
        </div>
        <div className="stat-card blue">
          <div className="stat-card-header"><span className="stat-label">Sharpe Ratio</span><div className="stat-icon blue">⚖️</div></div>
          <div className="stat-value">{m.sharpe}</div>
          <div className="stat-meta"><span className="stat-meta-text">Risk-adjusted</span></div>
        </div>
        <div className="stat-card red">
          <div className="stat-card-header"><span className="stat-label">Max Drawdown</span><div className="stat-icon red">📉</div></div>
          <div className="stat-value">{m.max_drawdown}%</div>
          <div className="stat-meta"><span className="stat-meta-text">Peak-to-Trough Loss</span></div>
        </div>
        <div className="stat-card purple">
          <div className="stat-card-header"><span className="stat-label">Win Rate</span><div className="stat-icon purple">🏆</div></div>
          <div className="stat-value">{m.win_rate}%</div>
          <div className="stat-meta"><span className="stat-meta-text">Profitable Trades</span></div>
        </div>
      </div>

      <div className="chart-card">
        <div className="chart-header">
          <div>
            <div className="chart-title">Strategy Equity Curve (2 Years)</div>
            <div className="chart-subtitle">AI Portfolio vs NIFTY 50 Benchmark</div>
          </div>
        </div>
        <div className="chart-body" style={{ height: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.chart_data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.07)" />
              <XAxis dataKey="date" tick={{ fill: "#8899bb", fontSize: 10 }} minTickGap={30} />
              <YAxis tick={{ fill: "#8899bb", fontSize: 10 }} domain={['auto', 'auto']} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="portfolio" name="AI Portfolio" stroke="#3b82f6" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="benchmark" name="NIFTY 50" stroke="#8899bb" strokeWidth={2} strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ── Trading View ─────────────────────────────────────────────────────────────
function TradingView({ stocks }) {
  const [mode, setMode]           = useState("paper");
  const [kiteConfig, setKiteConfig] = useState(null);
  const [positions, setPositions] = useState([]);
  const [orders, setOrders]       = useState([]);
  const [capital, setCapital]     = useState(100000);
  const [totalPnl, setTotalPnl]   = useState(0);
  const [qty, setQty]             = useState({});
  const [targetMode, setTargetMode] = useState("atr");
  const [toast, setToast]         = useState(null);
  const [loadingPos, setLoadingPos] = useState(false);

  const showToast = (msg, ok = true) => {
    setToast({ msg, ok });
    setTimeout(() => setToast(null), 3500);
  };

  const fetchPositions = async () => {
    setLoadingPos(true);
    try {
      const res = await axios.get(`http://127.0.0.1:8000/trading/positions?mode=${mode}`);
      if (res.data.success) {
        setPositions(res.data.positions || []);
        setCapital(res.data.capital ?? 100000);
        setTotalPnl(res.data.total_pnl ?? 0);
      }
    } catch (e) { /* silent */ }
    setLoadingPos(false);
  };

  const fetchOrders = async () => {
    try {
      const res = await axios.get(`http://127.0.0.1:8000/trading/orders?mode=${mode}`);
      if (res.data.success) setOrders(res.data.orders || []);
    } catch (e) { /* silent */ }
  };

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/trading/config").then(r => setKiteConfig(r.data)).catch(() => {});
    fetchPositions();
    fetchOrders();
  }, [mode]);

  const handleBuy = async (symbol) => {
    const q = parseInt(qty[symbol] || 1);
    if (q < 1) return showToast("Quantity must be ≥ 1", false);
    try {
      const res = await axios.post("http://127.0.0.1:8000/trading/buy", { symbol, quantity: q, mode });
      if (res.data.success) {
        showToast(`Bought ${q}x ${symbol.replace(".NS", "")} @ ₹${res.data.order?.entry_price?.toFixed(2)}`);
        fetchPositions(); fetchOrders();
      } else showToast(res.data.error, false);
    } catch (e) { showToast(e.message, false); }
  };

  const handleSell = async (symbol, availQty) => {
    try {
      const res = await axios.post("http://127.0.0.1:8000/trading/sell", { symbol, quantity: availQty, mode });
      if (res.data.success) {
        showToast(`Closed ${symbol.replace(".NS","")} · P&L ₹${res.data.pnl?.toFixed(2)}`, res.data.pnl >= 0);
        fetchPositions(); fetchOrders();
      } else showToast(res.data.error, false);
    } catch (e) { showToast(e.message, false); }
  };

  // Stocks eligible for trading cards (STRONG BUY / BUY, show top 12)
  const buySignals = stocks.filter(s => s.signal === "STRONG BUY" || s.signal === "BUY").slice(0, 12);

  const tLabel = { atr: "ATR-Based", fibonacci: "Fibonacci Ext.", fixed: "Fixed %" };

  const getTargetDisplay = (s) => {
    if (!s.targets) return null;
    const t = s.targets;
    if (targetMode === "atr" && t.atr) return {
      target: t.atr.target, sl: t.atr.stop_loss, rr: t.atr.risk_reward,
      sub: `ATR = ₹${t.atr.atr_value}`
    };
    if (targetMode === "fibonacci" && t.fibonacci) return {
      target: t.fibonacci.target_127, sl: t.fibonacci.stop_loss, rr: t.fibonacci.risk_reward,
      sub: `Fib 1.272 Ext · 1.618 → ₹${t.fibonacci.target_162?.toFixed(2)}`
    };
    if (targetMode === "fixed" && t.fixed) return {
      target: t.fixed.target_5pct, sl: t.fixed.stop_loss, rr: t.fixed.risk_reward_5,
      sub: `+10% → ₹${t.fixed.target_10pct} · +15% → ₹${t.fixed.target_15pct}`
    };
    return null;
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Toast */}
      {toast && (
        <div style={{
          position: "fixed", bottom: 28, right: 28, zIndex: 999,
          background: toast.ok ? "rgba(16,185,129,0.12)" : "rgba(239,68,68,0.12)",
          border: `1px solid ${toast.ok ? "rgba(16,185,129,0.3)" : "rgba(239,68,68,0.3)"}`,
          color: toast.ok ? "var(--green)" : "var(--red)",
          padding: "12px 20px", borderRadius: 12, fontSize: 13, fontWeight: 600,
          backdropFilter: "blur(16px)", boxShadow: "0 8px 32px rgba(0,0,0,0.4)"
        }}>{toast.ok ? "✓" : "✕"} {toast.msg}</div>
      )}

      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div>
          <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.8px", fontWeight: 600 }}>Trading Mode</div>
          <div style={{ display: "flex", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 10, padding: 3, gap: 2 }}>
            {["paper", "live"].map(m => (
              <button key={m} onClick={() => setMode(m)} style={{
                padding: "7px 20px", borderRadius: 8, border: "none", cursor: "pointer",
                fontWeight: 700, fontSize: 12, fontFamily: "'JetBrains Mono',monospace",
                letterSpacing: "0.5px", textTransform: "uppercase",
                background: mode === m ? (m === "paper" ? "var(--accent-soft)" : "rgba(16,185,129,0.15)") : "transparent",
                color: mode === m ? (m === "paper" ? "var(--accent)" : "var(--green)") : "var(--text-muted)",
                border: mode === m ? `1px solid ${m === "paper" ? "rgba(59,130,246,0.3)" : "rgba(16,185,129,0.3)"}` : "1px solid transparent",
                transition: "all 0.2s"
              }}>{m === "paper" ? "📄 Paper" : "⚡ Live"}</button>
            ))}
          </div>
        </div>

        {/* Capital + P&L */}
        <div style={{ display: "flex", gap: 16 }}>
          <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 12, padding: "12px 20px", textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.8px", marginBottom: 4 }}>Available Capital</div>
            <div style={{ fontSize: 22, fontWeight: 800, fontFamily: "'JetBrains Mono',monospace", color: "var(--text-primary)" }}>₹{capital.toLocaleString("en-IN", { maximumFractionDigits: 0 })}</div>
          </div>
          <div style={{ background: "var(--bg-card)", border: `1px solid ${totalPnl >= 0 ? "rgba(16,185,129,0.25)" : "rgba(239,68,68,0.25)"}`, borderRadius: 12, padding: "12px 20px", textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.8px", marginBottom: 4 }}>Total P&L</div>
            <div style={{ fontSize: 22, fontWeight: 800, fontFamily: "'JetBrains Mono',monospace", color: totalPnl >= 0 ? "var(--green)" : "var(--red)" }}>{totalPnl >= 0 ? "+" : ""}₹{totalPnl.toFixed(2)}</div>
          </div>
        </div>

        {/* Kite connect (live mode) */}
        {mode === "live" && (
          <div>
            {kiteConfig?.connected ? (
              <div style={{ padding: "10px 18px", background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.3)", borderRadius: 10, color: "var(--green)", fontSize: 13, fontWeight: 700 }}>✓ Zerodha Connected</div>
            ) : (
              <a href={kiteConfig?.login_url || "#"} target="_blank" rel="noreferrer" style={{
                display: "inline-block", padding: "10px 18px",
                background: "linear-gradient(135deg,#387ed1,#3b82f6)",
                border: "1px solid rgba(59,130,246,0.4)", borderRadius: 10,
                color: "#fff", fontSize: 13, fontWeight: 700, textDecoration: "none",
                boxShadow: "0 4px 20px rgba(59,130,246,0.3)"
              }}>{kiteConfig?.api_key_set ? "🔑 Connect Zerodha" : "⚠ Set API Key in .env"}</a>
            )}
          </div>
        )}
      </div>

      {/* Target method selector */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.8px" }}>Target Method:</span>
        {["atr", "fibonacci", "fixed"].map(m => (
          <button key={m} onClick={() => setTargetMode(m)} style={{
            padding: "5px 14px", borderRadius: 20, border: "1px solid",
            borderColor: targetMode === m ? "rgba(59,130,246,0.5)" : "var(--border)",
            background: targetMode === m ? "var(--accent-soft)" : "transparent",
            color: targetMode === m ? "var(--accent)" : "var(--text-muted)",
            fontSize: 11, fontWeight: 700, cursor: "pointer", transition: "all 0.2s",
            fontFamily: "'JetBrains Mono',monospace"
          }}>{tLabel[m]}</button>
        ))}
      </div>

      {/* Buy Signal Cards */}
      <div className="card">
        <div className="card-header">
          <div><div className="card-title">Top Buy Opportunities</div><div className="card-subtitle">AI-ranked STRONG BUY / BUY signals with {tLabel[targetMode]} targets</div></div>
          <span className="card-badge live">● {buySignals.length} Signals</span>
        </div>
        <div style={{ padding: "16px 24px 20px", display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(320px,1fr))", gap: 14 }}>
          {buySignals.length === 0 ? (
            <div className="empty-state">No buy signals yet — refresh data</div>
          ) : buySignals.map((s, i) => {
            const td = getTargetDisplay(s);
            const symShort = s.symbol.replace(".NS", "");
            return (
              <div key={i} style={{
                background: "rgba(59,130,246,0.04)", border: "1px solid rgba(59,130,246,0.12)",
                borderRadius: 14, padding: "16px 18px", display: "flex", flexDirection: "column", gap: 10,
                transition: "border-color 0.2s"
              }}>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <div style={{ fontSize: 15, fontWeight: 800, fontFamily: "'JetBrains Mono',monospace", color: "var(--text-primary)" }}>{symShort}</div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>NSE · India</div>
                  </div>
                  <span className={`signal-badge ${signalClass(s.signal)}`} style={{ fontSize: 10 }}>{signalDot(s.signal)} {s.signal}</span>
                </div>

                {/* Price row */}
                <div style={{ display: "flex", gap: 10 }}>
                  <div style={{ flex: 1, background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "8px 12px" }}>
                    <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.8px" }}>Entry</div>
                    <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "'JetBrains Mono',monospace", color: "var(--text-primary)" }}>₹{s.price?.toFixed(2)}</div>
                  </div>
                  {td && <>
                    <div style={{ flex: 1, background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)", borderRadius: 8, padding: "8px 12px" }}>
                      <div style={{ fontSize: 9, color: "var(--green)", textTransform: "uppercase", letterSpacing: "0.8px" }}>🎯 Target</div>
                      <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "'JetBrains Mono',monospace", color: "var(--green)" }}>₹{td.target?.toFixed(2)}</div>
                    </div>
                    <div style={{ flex: 1, background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, padding: "8px 12px" }}>
                      <div style={{ fontSize: 9, color: "var(--red)", textTransform: "uppercase", letterSpacing: "0.8px" }}>🛡 Stop Loss</div>
                      <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "'JetBrains Mono',monospace", color: "var(--red)" }}>₹{td.sl?.toFixed(2)}</div>
                    </div>
                  </>}
                </div>

                {/* R/R + sub label */}
                {td && (
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{td.sub}</span>
                    <span style={{ fontSize: 11, fontWeight: 700, fontFamily: "'JetBrains Mono',monospace", padding: "3px 8px", borderRadius: 6, background: "rgba(245,158,11,0.1)", color: "var(--yellow)" }}>R/R {td.rr}:1</span>
                  </div>
                )}

                {/* Qty + Buy button */}
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <input type="number" min="1" value={qty[s.symbol] || 1}
                    onChange={e => setQty(prev => ({ ...prev, [s.symbol]: e.target.value }))}
                    style={{ width: 64, padding: "7px 10px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--bg-base)", color: "var(--text-primary)", fontSize: 13, fontFamily: "'JetBrains Mono',monospace", outline: "none" }}
                  />
                  <button onClick={() => handleBuy(s.symbol)} style={{
                    flex: 1, padding: "8px 0", borderRadius: 8, border: "1px solid rgba(16,185,129,0.4)",
                    background: "rgba(16,185,129,0.12)", color: "var(--green)",
                    fontSize: 12, fontWeight: 700, cursor: "pointer", transition: "all 0.2s",
                    fontFamily: "'JetBrains Mono',monospace"
                  }}>▲ BUY {mode === "paper" ? "(Paper)" : "(Live)"}</button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Active Positions */}
      <div className="card">
        <div className="card-header">
          <div><div className="card-title">Active Positions</div><div className="card-subtitle">Live P&L · {mode === "paper" ? "Simulated portfolio" : "Zerodha account"}</div></div>
          <button onClick={fetchPositions} style={{ padding: "5px 14px", borderRadius: 8, border: "1px solid var(--border)", background: "transparent", color: "var(--text-muted)", fontSize: 12, cursor: "pointer" }}>{loadingPos ? "…" : "↻ Refresh"}</button>
        </div>
        <div style={{ padding: "0 24px 20px" }}>
          {positions.length === 0 ? (
            <div className="empty-state" style={{ padding: 24 }}>No open positions</div>
          ) : (
            <table className="stock-table">
              <thead><tr>
                <th>Symbol</th><th>Qty</th><th>Entry</th><th>Current</th><th>P&L</th>
                <th>🎯 Target</th><th>🛡 Stop Loss</th><th>Action</th>
              </tr></thead>
              <tbody>
                {positions.map((p, i) => {
                  const pnlUp = p.pnl >= 0;
                  const td = p.targets ? (
                    targetMode === "atr" ? { t: p.targets.atr?.target, sl: p.targets.atr?.stop_loss } :
                    targetMode === "fibonacci" ? { t: p.targets.fibonacci?.target_127, sl: p.targets.fibonacci?.stop_loss } :
                    { t: p.targets.fixed?.target_5pct, sl: p.targets.fixed?.stop_loss }
                  ) : {};
                  return (
                    <tr key={i}>
                      <td style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 700, color: "var(--text-primary)" }}>{p.symbol}</td>
                      <td>{p.quantity}</td>
                      <td style={{ fontFamily: "'JetBrains Mono',monospace" }}>₹{p.entry_price?.toFixed(2)}</td>
                      <td style={{ fontFamily: "'JetBrains Mono',monospace" }}>₹{p.current_price?.toFixed(2)}</td>
                      <td style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 700, color: pnlUp ? "var(--green)" : "var(--red)" }}>{pnlUp ? "+" : ""}₹{p.pnl?.toFixed(2)} <span style={{ fontSize: 10 }}>({p.pnl_pct?.toFixed(1)}%)</span></td>
                      <td style={{ color: "var(--green)", fontFamily: "'JetBrains Mono',monospace" }}>{td.t ? `₹${td.t.toFixed(2)}` : "—"}</td>
                      <td style={{ color: "var(--red)", fontFamily: "'JetBrains Mono',monospace" }}>{td.sl ? `₹${td.sl.toFixed(2)}` : "—"}</td>
                      <td><button onClick={() => handleSell(p.full_symbol, p.quantity)} style={{ padding: "5px 14px", borderRadius: 8, border: "1px solid rgba(239,68,68,0.35)", background: "rgba(239,68,68,0.08)", color: "var(--red)", fontSize: 11, fontWeight: 700, cursor: "pointer", fontFamily: "'JetBrains Mono',monospace" }}>▼ SELL</button></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Order History */}
      <div className="card">
        <div className="card-header"><div><div className="card-title">Order History</div><div className="card-subtitle">All {mode} trades</div></div></div>
        <div style={{ padding: "0 24px 20px", maxHeight: 320, overflowY: "auto" }}>
          {orders.length === 0 ? <div className="empty-state" style={{ padding: 24 }}>No orders yet</div> : (
            <table className="stock-table">
              <thead><tr><th>Time</th><th>Symbol</th><th>Type</th><th>Qty</th><th>Price</th><th>P&L</th><th>Status</th></tr></thead>
              <tbody>
                {[...orders].reverse().map((o, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: 11, color: "var(--text-muted)" }}>{o.timestamp ? new Date(o.timestamp).toLocaleTimeString("en-IN") : "—"}</td>
                    <td style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 700, color: "var(--text-primary)" }}>{o.symbol}</td>
                    <td><span style={{ padding: "2px 8px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: o.type === "BUY" ? "rgba(16,185,129,0.1)" : "rgba(239,68,68,0.1)", color: o.type === "BUY" ? "var(--green)" : "var(--red)" }}>{o.type}</span></td>
                    <td>{o.quantity}</td>
                    <td style={{ fontFamily: "'JetBrains Mono',monospace" }}>₹{(o.entry_price || o.exit_price || 0).toFixed(2)}</td>
                    <td style={{ fontFamily: "'JetBrains Mono',monospace", color: (o.pnl ?? 0) >= 0 ? "var(--green)" : "var(--red)" }}>{o.pnl != null ? `${o.pnl >= 0 ? "+" : ""}₹${o.pnl.toFixed(2)}` : "—"}</td>
                    <td><span style={{ fontSize: 10, color: "var(--text-muted)" }}>{o.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Login View ──────────────────────────────────────────────────────────────
function LoginView({ onLogin }) {
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await axios.post("http://127.0.0.1:8000/login", { password });
      if (res.data.success) {
        onLogin(res.data.token);
      } else {
        setError(res.data.error || "Invalid password");
      }
    } catch (err) {
      setError("Server error. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-glass">
        <div className="login-header">
          <div className="login-logo">⚡</div>
          <h1>HakiTrade</h1>
          <p>Institutional Grade AI Quant Platform</p>
        </div>
        
        <form onSubmit={handleSubmit} className="login-form">
          <div className="input-group">
            <label>Admin Password</label>
            <input 
              type="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              autoFocus
            />
          </div>
          
          {error && <div className="login-error">{error}</div>}
          
          <button type="submit" className="login-btn" disabled={loading}>
            {loading ? "Authenticating..." : "Unlock Dashboard"}
          </button>
        </form>
        
        <div className="login-footer">
          <span>Secure Session Encryption Enabled</span>
        </div>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem("haki_auth"));
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  const handleLogin = (token) => {
    localStorage.setItem("haki_auth", token);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem("haki_auth");
    setIsAuthenticated(false);
  };

  const fetchStocks = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const res = await axios.get("http://127.0.0.1:8000/top-stocks");
      setStocks(res.data.stocks || []);
      setBackendOnline(true);
      setLastUpdated(new Date());
    } catch (e) {
      setError("Cannot reach backend. Start the FastAPI server on port 8000.");
      setBackendOnline(false);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    if (!isAuthenticated) return;
    fetchStocks();
    const id = setInterval(fetchStocks, 60000);
    return () => clearInterval(id);
  }, [fetchStocks, isAuthenticated]);

  // Derived stats
  const buyCount = stocks.filter(s => s.signal === "STRONG BUY" || s.signal === "BUY").length;
  const avgAI    = stocks.length ? (stocks.reduce((a, s) => a + s.ai_probability, 0) / stocks.length) : 0;
  const topScore = stocks.length ? Math.max(...stocks.map(s => s.score)) : 0;
  const maxScore = Math.max(...stocks.map(s => Math.abs(s.score)), 1);

  const engineItems = [
    { icon: "⚡", label: "FastAPI Backend", color: "var(--accent-soft)", status: backendOnline },
    { icon: "📡", label: "NSE Market Data", color: "var(--green-soft)", status: backendOnline },
    { icon: "🧠", label: "AI Quant Engine", color: "var(--purple-soft)", status: backendOnline },
    { icon: "🔮", label: "ML Predictions", color: "var(--cyan-soft)", status: backendOnline },
  ];

  if (!isAuthenticated) {
    return <LoginView onLogin={handleLogin} />;
  }

  return (
    <div className="app">
      {/* ── Navbar ── */}
      <nav className="navbar">
        <div className="navbar-logo">
          <div className="logo-icon">⚡</div>
          <span className="logo-text">HakiTrade</span>
          <span className="logo-badge">NSE · QUANT</span>
        </div>
        <div className="navbar-center">
          {["Dashboard", "Signals", "Backtest", "Analytics", "Trading"].map(l => (
            <span 
              key={l} 
              className={`nav-link${l === activeTab ? " active" : ""}`}
              onClick={() => setActiveTab(l)}
              style={{ cursor: "pointer" }}
            >
              {l}
            </span>
          ))}
        </div>
        <div className="navbar-right">
          <div className="status-dot" style={{ background: backendOnline ? "var(--green)" : "var(--red)" }} />
          <span className="status-text">{backendOnline ? "LIVE" : "OFFLINE"}</span>
          <button className={`refresh-btn${refreshing ? " loading" : ""}`} onClick={fetchStocks} disabled={refreshing}>
            <span className="btn-icon">↻</span>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
          <div className="logout-btn" onClick={handleLogout} title="Logout">
            🚪
          </div>
        </div>
      </nav>

      {/* ── Ticker ── */}
      <TickerBar stocks={stocks} />

      {/* ── Main ── */}
      <main className="main">
        {/* Header */}
        <div className="page-header">
          <div>
            <div className="page-title">AI Quant Dashboard</div>
            <div className="page-subtitle">Real-time NSE signals powered by machine learning</div>
          </div>
          <div className="last-updated">
            {lastUpdated
              ? `Last updated: ${lastUpdated.toLocaleTimeString()}`
              : "Awaiting data…"}
          </div>
        </div>

        {/* ── Backtest View ── */}
        {activeTab === "Backtest" && <BacktestView />}

        {/* ── Trading View ── */}
        {activeTab === "Trading" && <TradingView stocks={stocks} />}

        {/* Stat Cards */}
        {activeTab !== "Backtest" && activeTab !== "Trading" && (
        <div className="stats-row">
          {[
            {
              color: "blue", icon: "📊", iconCls: "blue",
              label: "Tracked Stocks", value: stocks.length || "—",
              badge: stocks.length ? `${stocks.length} Active` : null, badgeCls: "neutral"
            },
            {
              color: "green", icon: "🟢", iconCls: "green",
              label: "Buy Signals", value: loading ? "—" : buyCount,
              badge: stocks.length ? `${Math.round(buyCount / Math.max(stocks.length, 1) * 100)}% bullish` : null, badgeCls: "up"
            },
            {
              color: "purple", icon: "🧠", iconCls: "purple",
              label: "Avg AI Confidence", value: loading ? "—" : `${fmt(avgAI)}%`,
              badge: avgAI > 60 ? "High Conviction" : "Moderate", badgeCls: avgAI > 60 ? "up" : "neutral"
            },
            {
              color: "cyan", icon: "🏆", iconCls: "cyan",
              label: "Top Quant Score",
              value: loading ? "—" : (topScore > 0 ? fmt(topScore) : (topScore < 0 ? fmt(topScore) : "0")),
              badge: stocks[0]?.signal || null, badgeCls: stocks[0]?.signal === "STRONG BUY" ? "up" : "neutral"
            },
          ].map((c) => (
            <div key={c.label} className={`stat-card ${c.color}`}>
              <div className="stat-card-header">
                <span className="stat-label">{c.label}</span>
                <div className={`stat-icon ${c.iconCls}`}>{c.icon}</div>
              </div>
              <div className="stat-value">{c.value}</div>
              {c.badge && (
                <div className="stat-meta">
                  <span className={`stat-badge ${c.badgeCls}`}>{c.badge}</span>
                  <span className="stat-meta-text">NSE equities</span>
                </div>
              )}
            </div>
          ))}
        </div>
        )}

        {/* Dashboard Grid: Table + Score Chart */}
        {(activeTab === "Dashboard" || activeTab === "Signals") && (
        <div className="dashboard-grid" style={activeTab === "Signals" ? { gridTemplateColumns: '1fr' } : {}}>
          {/* Stock Table */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Top AI Ranked Stocks</div>
                <div className="card-subtitle">Sorted by composite quant score</div>
              </div>
              <span className="card-badge live">● LIVE</span>
            </div>
            <div className="card-body" style={{ padding: "0 24px 0" }}>
              {error ? (
                <div className="error-state">
                  <div className="error-icon">⚠️</div>
                  <div className="error-text">{error}</div>
                  <button className="retry-btn" onClick={fetchStocks}>Retry</button>
                </div>
              ) : (
                <div className={activeTab === "Dashboard" ? "stock-table-scroll" : ""}>
                  <table className="stock-table">
                    <thead>
                      <tr>
                        <th>Stock</th>
                        <th>Price</th>
                        <th>Score</th>
                        <th>Signal</th>
                        <th>Sentiment</th>
                        <th>AI Conf.</th>
                        {activeTab === "Signals" && <th>Model Consensus</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {loading ? <SkeletonRows /> : (activeTab === "Dashboard" ? stocks.slice(0, 15) : stocks).map((s, i) => (
                        <tr key={i}>
                          <td>
                            <div className="stock-name-cell">
                              <span className="stock-symbol">{shortSymbol(s.symbol)}</span>
                              <span className="stock-exchange">NSE · India</span>
                              <span className="news-title" style={{ fontSize: 10, color: '#8899bb', marginTop: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 220, display: 'block' }}>
                                {s.news_title || "No recent news."}
                              </span>
                            </div>
                          </td>
                          <td className="price-cell">{fmtPrice(s.price)}</td>
                          <td>
                            <div className="score-cell">
                              <span className="mono" style={{ color: scoreColor(s.score), fontWeight: 700, minWidth: 42 }}>
                                {fmt(s.score)}
                              </span>
                              <div className="score-bar-wrap">
                                <div className="score-bar-fill" style={{
                                  width: `${Math.min(Math.max((s.score / maxScore) * 100, 0), 100)}%`,
                                  background: scoreColor(s.score)
                                }} />
                              </div>
                            </div>
                          </td>
                          <td>
                            <span className={`signal-badge ${signalClass(s.signal)}`}>
                              {signalDot(s.signal)} {s.signal}
                            </span>
                          </td>
                          <td>
                            <span className={`stat-badge ${sentimentClass(s.sentiment_raw)}`} style={{ fontSize: 10, padding: '2px 6px' }}>
                              {sentimentText(s.sentiment_raw)}
                            </span>
                          </td>
                          <td>
                            <div className="ai-prob-wrap">
                              <div className="ai-prob-bar">
                                <div className="ai-prob-fill" style={{ width: `${s.ai_probability}%` }} />
                              </div>
                              <span className="ai-prob-text">{fmt(s.ai_probability)}%</span>
                            </div>
                          </td>
                          {activeTab === "Signals" && (
                            <td><ModelBreakdown breakdown={s.model_breakdown} /></td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {!loading && !error && activeTab === "Dashboard" && stocks.length > 15 && (
                <div style={{ padding: "12px 0 16px", borderTop: "1px solid rgba(99,179,237,0.06)", textAlign: "center" }}>
                  <span
                    onClick={() => setActiveTab("Signals")}
                    style={{ fontSize: 12, color: "var(--accent)", fontFamily: "'JetBrains Mono',monospace", cursor: "pointer", fontWeight: 600, letterSpacing: "0.5px" }}
                  >
                    View all {stocks.length} stocks in Signals →
                  </span>
                </div>
              )}
            </div>
          </div>


          {/* Quant Score Bar Chart */}
          {activeTab === "Dashboard" && (
          <div className="chart-card">
            <div className="chart-header">
              <div>
                <div className="chart-title">Quant Score Ranking</div>
                <div className="chart-subtitle">Composite momentum + AI signal</div>
              </div>
              <span className="card-badge live">● LIVE</span>
            </div>
            <div className="chart-body" style={{ height: 300 }}>
              {loading || !stocks.length ? (
                <div className="empty-state">
                  <div className="skeleton" style={{ width: "100%", height: "100%", borderRadius: 12 }} />
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stocks.map(s => ({ ...s, symbol: shortSymbol(s.symbol) }))}
                    margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.07)" />
                    <XAxis dataKey="symbol" tick={{ fill: "#8899bb", fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#8899bb", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="score" name="Score" radius={[6, 6, 0, 0]}>
                      {stocks.map((s, i) => (
                        <Cell key={i} fill={scoreColor(s.score)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
          )}
        </div>
        )}

        {/* Charts Row */}
        {(activeTab === "Dashboard" || activeTab === "Analytics") && (
        <div className="charts-grid">
          {/* AI Probability Line */}
          <div className="chart-card">
            <div className="chart-header">
              <div>
                <div className="chart-title">AI Prediction Confidence</div>
                <div className="chart-subtitle">ML bullish probability per stock</div>
              </div>
            </div>
            <div className="chart-body">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={stocks.map(s => ({ ...s, symbol: shortSymbol(s.symbol) }))}
                  margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="aiGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.07)" />
                  <XAxis dataKey="symbol" tick={{ fill: "#8899bb", fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#8899bb", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="ai_probability" name="AI %" stroke="#3b82f6" strokeWidth={2.5} fill="url(#aiGrad)" dot={{ fill: "#3b82f6", r: 4, strokeWidth: 0 }} activeDot={{ r: 6 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Momentum Area Chart */}
          <div className="chart-card">
            <div className="chart-header">
              <div>
                <div className="chart-title">Price Momentum (10D)</div>
                <div className="chart-subtitle">% change over last 10 trading days</div>
              </div>
            </div>
            <div className="chart-body">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={stocks.map(s => ({ ...s, symbol: shortSymbol(s.symbol) }))}
                  margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="momGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.07)" />
                  <XAxis dataKey="symbol" tick={{ fill: "#8899bb", fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#8899bb", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="momentum" name="Momentum %" stroke="#10b981" strokeWidth={2.5} fill="url(#momGrad)" dot={{ fill: "#10b981", r: 4, strokeWidth: 0 }} activeDot={{ r: 6 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        )}

        {/* Bottom Row */}
        {(activeTab === "Dashboard" || activeTab === "Analytics") && (
        <div className="bottom-row" style={activeTab === "Analytics" ? { gridTemplateColumns: '1fr' } : {}}>
          {/* Metrics Detail */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Technical Indicators</div>
                <div className="card-subtitle">RSI · MACD · Volatility · Volume</div>
              </div>
            </div>
            <div className="card-body" style={{ padding: "12px 24px 20px" }}>
              {loading ? (
                <div className="skeleton" style={{ width: "100%", height: 160, borderRadius: 8 }} />
              ) : !stocks.length ? (
                <div className="empty-state">No data available</div>
              ) : (
                <table className="metrics-table">
                  <thead>
                    <tr>
                      <th>Stock</th>
                      <th>RSI</th>
                      <th>MACD</th>
                      <th>Vol Ratio</th>
                      <th>Volatility</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stocks.map((s, i) => (
                      <tr key={i}>
                        <td style={{ color: "#f0f4ff", fontWeight: 700 }}>{shortSymbol(s.symbol)}</td>
                        <td><span className={`rsi-value ${rsiClass(s.rsi)}`}>{fmt(s.rsi)}</span></td>
                        <td style={{ color: s.macd >= 0 ? "var(--green)" : "var(--red)" }}>{fmt(s.macd)}</td>
                        <td style={{ color: s.volume_ratio > 1.2 ? "var(--cyan)" : "var(--text-secondary)" }}>{fmt(s.volume_ratio)}x</td>
                        <td style={{ color: s.volatility > 2 ? "var(--yellow)" : "var(--text-secondary)" }}>{fmt(s.volatility)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* Engine Status */}
          {activeTab === "Dashboard" && (
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Engine Status</div>
                <div className="card-subtitle">System health &amp; connectivity</div>
              </div>
              <span className="card-badge live" style={!backendOnline ? { background: "var(--red-soft)", color: "var(--red)", borderColor: "rgba(239,68,68,0.2)" } : {}}>
                {backendOnline ? "● ONLINE" : "● OFFLINE"}
              </span>
            </div>
            <div className="card-body">
              <div className="engine-grid">
                {engineItems.map((e, i) => (
                  <div className="engine-item" key={i}>
                    <div className="engine-icon" style={{ background: e.color }}>{e.icon}</div>
                    <div className="engine-info">
                      <div className="engine-name">{e.label}</div>
                      <div className={`engine-status ${e.status ? "online" : "offline"}`}>
                        {e.status ? "ONLINE" : "OFFLINE"}
                      </div>
                    </div>
                    <div className={`engine-indicator ${e.status ? "online" : "offline"}`} />
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 16, padding: "12px 14px", background: "rgba(59,130,246,0.04)", borderRadius: 10, border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "'JetBrains Mono',monospace" }}>
                  DATA SOURCE: Yahoo Finance (NSE) · MODEL: scikit-learn RandomForest · REFRESH: 60s
                </div>
              </div>
            </div>
          </div>
          )}
        </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <span><span className="footer-brand">HakiTrade</span> · AI Quantitative Trading Platform</span>
        <span>NSE · India · {new Date().getFullYear()}</span>
        <span>Data delayed ~15min · Not financial advice</span>
      </footer>
    </div>
  );
}