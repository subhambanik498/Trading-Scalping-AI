# APEX TRADING BOT v3.1
## Autonomous Algorithmic Trading System

---

## WHAT'S INCLUDED

| File | Purpose |
|------|---------|
| `index.html` | Full trading dashboard UI (open in any browser) |
| `backend.py` | Python FastAPI trading engine |
| `README.md` | This file |

---

## QUICK START (Frontend Only)
Just open `index.html` in your browser. The UI works standalone with simulated live data.

---

## FULL BACKEND SETUP

### 1. Install Python dependencies
```bash
pip install fastapi uvicorn alpaca-py yfinance pandas ta-lib scikit-learn \
            torch apscheduler python-dotenv websockets kiteconnect
```

### 2. Configure your .env file
```bash
# Create .env in the same folder
ADMIN_SECRET_KEY=YOUR_SECURE_PASSWORD_HERE

# Alpaca (US stocks) - get at alpaca.markets
ALPACA_API_KEY=pk_live_xxxxxxxxxxxx
ALPACA_SECRET=sk_live_xxxxxxxxxxxx
ALPACA_PAPER=true   # Set false for real money trading

# Zerodha (Indian markets) - get at kite.trade
ZERODHA_API_KEY=xxxxxxxxxxxxxxxx
ZERODHA_SECRET=xxxxxxxxxxxxxxxx

# Risk settings
MAX_POSITION_PCT=0.02    # 2% of portfolio per trade
STOP_LOSS_PCT=0.015      # 1.5% stop loss
TAKE_PROFIT_PCT=0.03     # 3% take profit
MAX_DAILY_LOSS=0.05      # Halt trading if down 5% in a day
MAX_POSITIONS=10         # Max simultaneous open trades
```

### 3. Run the server
```bash
python backend.py
# Server starts at: http://localhost:8000
# API docs at:      http://localhost:8000/docs
```

---

## API ENDPOINTS

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/status` | None | Bot status + ML accuracy |
| GET | `/positions` | None | All open positions |
| GET | `/trades` | None | Recent trade history |
| GET | `/scan` | None | Run stock scanner |
| GET | `/indicators` | None | List all indicators |
| POST | `/bot/start` | Admin | Start trading |
| POST | `/bot/stop` | Admin | Stop trading |
| POST | `/broker/add` | Admin | Add broker API |
| POST | `/indicators/add` | Admin | Add custom indicator |
| POST | `/retrain` | Admin | Trigger ML retraining |
| PUT | `/risk` | Admin | Update risk parameters |
| WS | `/ws` | None | Live WebSocket feed |

**Admin auth**: Pass header `X-Admin-Key: YOUR_ADMIN_PASSWORD`

---

## FEATURES

### Autonomous Operation
- Scans 5,000+ stocks every 30 seconds
- Places trades automatically with bracket orders (stop-loss + take-profit pre-set)
- After a profitable exit, immediately hunts for the next opportunity
- Runs 24/7 — no human needed

### Midnight Retraining
- Every midnight at 00:00, the ML model retrains on 12 months of fresh data
- Uses Gradient Boosting + Random Forest ensemble
- Accuracy reported after each retrain
- Can also be triggered manually via `/retrain` API (admin only)

### Risk Management (NEVER BYPASSED)
- Per-trade position sizing (default 2% of portfolio)
- Auto stop-loss on every trade (default 1.5%)
- Auto take-profit on every trade (default 3%)
- Daily loss circuit breaker (default 5% — halts all trading)
- Max simultaneous positions cap (default 10)
- Portfolio heat monitoring

### Technical Indicators
Built-in: RSI, MACD, Bollinger Bands, EMA(9/21/200), ATR, Stochastic, VWAP, OBV
Admin can add custom indicators at runtime via the API (no restart needed)

### Multi-Broker Support
- Alpaca Markets (US stocks, crypto)
- Zerodha Kite (Indian NSE/BSE)
- Interactive Brokers (coming soon)
- Custom REST API (any broker)

---

## ADMIN PROTECTION
- Admin password is required to: start/stop bot, add brokers, add indicators, change risk params, trigger retraining
- Default password: `APEX@admin2024` — **CHANGE THIS IN .env BEFORE GOING LIVE**
- In the UI: click the ⬡ ADMIN button, enter your password

---

## ⚠️ IMPORTANT DISCLAIMERS

1. **Start with paper trading.** Set `ALPACA_PAPER=true` and test for at least 30 days before risking real money.
2. **No system can guarantee profits.** Markets are partially random. This bot uses best-practice risk management to minimize losses, not eliminate them.
3. **Past backtest performance ≠ future results.**
4. **You are responsible** for your trades. Monitor the bot regularly.
5. **This is not financial advice.**

---

## ARCHITECTURE

```
Browser (index.html)
       │
       ▼
FastAPI Backend (backend.py)
       │
  ┌────┴─────────────────────┐
  │                          │
TradeEngine              ML Engine
  │                     (LSTM + GB + RF)
  ├── RiskManager            │
  ├── IndicatorEngine ───────┘
  ├── ProfitHunter
  └── Brokers
       ├── Alpaca
       ├── Zerodha
       └── Custom
```
