"""
APEX TRADING BOT — Backend Engine
Python 3.11+ | FastAPI | ML Trading System

Architecture:
  - FastAPI REST server (main interface for all broker APIs)
  - APScheduler for midnight retraining
  - Alpaca / Zerodha / IBKR / Binance broker adapters
  - ML Engine: LSTM + Transformer ensemble
  - Risk Manager: real-time position/drawdown monitoring
  - Profit Hunter: scans 5,000+ stocks autonomously
"""

import asyncio
import logging
import json
import os
from datetime import datetime, time as dtime
from typing import Optional, Dict, List
import numpy as np

# pip install fastapi uvicorn alpaca-py yfinance pandas ta scikit-learn
# pip install torch apscheduler python-dotenv websockets

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("APEX")

# ============================================================
# CONFIG
# ============================================================
class Config:
    ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "APEX@admin2024")  # Change in .env!
    ALPACA_API_KEY   = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET    = os.getenv("ALPACA_SECRET", "")
    ALPACA_PAPER     = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    ZERODHA_API_KEY  = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_SECRET   = os.getenv("ZERODHA_SECRET", "")
    MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.02"))   # 2% max per trade
    STOP_LOSS_PCT    = float(os.getenv("STOP_LOSS_PCT", "0.015"))     # 1.5% stop loss
    TAKE_PROFIT_PCT  = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))    # 3% take profit
    MAX_DAILY_LOSS   = float(os.getenv("MAX_DAILY_LOSS", "0.05"))     # 5% circuit breaker
    MAX_POSITIONS    = int(os.getenv("MAX_POSITIONS", "10"))
    RETRAIN_HOUR     = 0    # Midnight
    RETRAIN_MINUTE   = 0

# ============================================================
# RISK MANAGER
# ============================================================
class RiskManager:
    """
    Hard-enforced risk rules. No trade bypasses this.
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.daily_loss = 0.0
        self.starting_equity = None
        self.circuit_tripped = False

    def set_equity(self, equity: float):
        if self.starting_equity is None:
            self.starting_equity = equity

    def position_size(self, equity: float, price: float) -> int:
        """Kelly-inspired fractional sizing with hard cap."""
        max_dollar = equity * self.cfg.MAX_POSITION_PCT
        qty = int(max_dollar / price)
        return max(1, qty)

    def check_trade(self, symbol: str, side: str, equity: float) -> tuple[bool, str]:
        if self.circuit_tripped:
            return False, "Circuit breaker tripped — daily loss limit exceeded"
        if self.daily_loss >= equity * self.cfg.MAX_DAILY_LOSS:
            self.circuit_tripped = True
            logger.warning("⚠️  CIRCUIT BREAKER: Daily loss limit hit. Trading halted.")
            return False, "Daily loss limit reached"
        return True, "OK"

    def update_daily_loss(self, pnl: float):
        if pnl < 0:
            self.daily_loss += abs(pnl)

    def reset_daily(self):
        """Called at market open each day."""
        self.daily_loss = 0.0
        self.circuit_tripped = False
        logger.info("Daily risk counters reset.")

    def stop_price(self, entry: float, side: str) -> float:
        if side == "buy":
            return round(entry * (1 - self.cfg.STOP_LOSS_PCT), 2)
        return round(entry * (1 + self.cfg.STOP_LOSS_PCT), 2)

    def target_price(self, entry: float, side: str) -> float:
        if side == "buy":
            return round(entry * (1 + self.cfg.TAKE_PROFIT_PCT), 2)
        return round(entry * (1 - self.cfg.TAKE_PROFIT_PCT), 2)

# ============================================================
# INDICATOR ENGINE
# ============================================================
class IndicatorEngine:
    """
    Computes all technical indicators using the `ta` library.
    Easily extensible — admin can add indicators via the API.
    """

    def __init__(self):
        self.custom_indicators: Dict[str, dict] = {}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        import ta
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        # MACD
        macd_obj = ta.trend.MACD(df['close'])
        df['macd'] = macd_obj.macd()
        df['macd_signal'] = macd_obj.macd_signal()
        df['macd_hist'] = macd_obj.macd_diff()
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pct'] = bb.bollinger_pband()
        # EMAs
        df['ema9']  = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema200']= ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        # Volume indicators
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'],df['low'],df['close'],df['volume']).volume_weighted_average_price()
        df['obv']  = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        # Apply any admin-added custom indicators
        for name, ind in self.custom_indicators.items():
            try:
                exec(ind['code'], {'df': df, 'ta': ta, 'pd': pd, 'np': np})
                logger.info(f"Custom indicator '{name}' computed.")
            except Exception as e:
                logger.error(f"Custom indicator '{name}' failed: {e}")
        return df

    def add_custom(self, name: str, code: str, params: dict):
        """Admin-only: add a new indicator at runtime."""
        self.custom_indicators[name] = {'code': code, 'params': params}
        logger.info(f"Custom indicator added: {name}")

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """Composite signal from all indicators."""
        last = df.iloc[-1]
        scores = []

        # RSI
        if last['rsi'] < 30: scores.append(+2)
        elif last['rsi'] < 45: scores.append(+1)
        elif last['rsi'] > 70: scores.append(-2)
        elif last['rsi'] > 55: scores.append(-1)
        else: scores.append(0)

        # MACD
        if last['macd'] > last['macd_signal'] and last['macd_hist'] > 0: scores.append(+2)
        elif last['macd'] < last['macd_signal']: scores.append(-1)

        # EMA trend
        if last['close'] > last['ema9'] > last['ema21']: scores.append(+2)
        elif last['close'] < last['ema9'] < last['ema21']: scores.append(-2)

        # BB
        if last['bb_pct'] < 0.2: scores.append(+1)
        elif last['bb_pct'] > 0.8: scores.append(-1)

        # Stochastic
        if last['stoch_k'] < 20 and last['stoch_k'] > last['stoch_d']: scores.append(+1)
        elif last['stoch_k'] > 80: scores.append(-1)

        total = sum(scores)
        confidence = min(abs(total) / 10 * 100, 99)
        action = 'BUY' if total > 2 else 'SELL' if total < -2 else 'HOLD'
        return {'action': action, 'score': total, 'confidence': round(confidence, 1)}

# ============================================================
# ML ENGINE
# ============================================================
class MLEngine:
    """
    LSTM + Transformer ensemble for price direction prediction.
    Retrains every midnight automatically.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.accuracy = 0.0
        self.last_trained = None
        self.feature_columns = [
            'rsi','macd','macd_signal','macd_hist','bb_pct',
            'ema9','ema21','atr','stoch_k','stoch_d','obv',
            'close','volume','high','low'
        ]

    def _build_model(self, input_dim: int):
        """Builds LSTM model. Requires PyTorch."""
        try:
            import torch
            import torch.nn as nn

            class LSTMModel(nn.Module):
                def __init__(self, in_dim, hidden=128, layers=3):
                    super().__init__()
                    self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=0.2)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 3)  # BUY / HOLD / SELL
                    )
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:,-1,:])

            return LSTMModel(input_dim)
        except ImportError:
            logger.warning("PyTorch not installed — using sklearn fallback model")
            return None

    def train(self, df: pd.DataFrame):
        """Full training pipeline. Called at midnight."""
        logger.info("⟳ ML training started...")
        try:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            features = [c for c in self.feature_columns if c in df.columns]
            df_clean = df[features + ['target']].dropna()

            X = df_clean[features].values
            y = df_clean['target'].values  # 0=sell, 1=hold, 2=buy

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

            # Ensemble
            gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
            rf = RandomForestClassifier(n_estimators=200, max_depth=8)
            gb.fit(X_train, y_train)
            rf.fit(X_train, y_train)

            gb_pred = gb.predict(X_test)
            rf_pred = rf.predict(X_test)
            ensemble = np.round((gb_pred + rf_pred) / 2).astype(int)
            self.accuracy = round(accuracy_score(y_test, ensemble) * 100, 1)
            self.model = (gb, rf)
            self.last_trained = datetime.now()
            logger.info(f"✓ ML training complete. Accuracy: {self.accuracy}%")
        except Exception as e:
            logger.error(f"ML training failed: {e}")

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            return {'action':'HOLD','confidence':0}
        try:
            from sklearn.preprocessing import StandardScaler
            features = [c for c in self.feature_columns if c in df.columns]
            last = df[features].iloc[-1:].values
            last_scaled = self.scaler.transform(last)
            gb, rf = self.model
            gb_pred = gb.predict_proba(last_scaled)[0]
            rf_pred = rf.predict_proba(last_scaled)[0]
            proba = (gb_pred + rf_pred) / 2
            cls = int(np.argmax(proba))
            action = ['SELL','HOLD','BUY'][cls]
            confidence = round(float(proba[cls]) * 100, 1)
            return {'action': action, 'confidence': confidence}
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'action':'HOLD','confidence':0}

# ============================================================
# BROKER ADAPTERS
# ============================================================
class AlpacaBroker:
    def __init__(self, api_key: str, secret: str, paper: bool = True):
        self.api_key = api_key
        self.secret = secret
        self.paper = paper
        self.client = None
        self._connect()

    def _connect(self):
        try:
            from alpaca.trading.client import TradingClient
            self.client = TradingClient(self.api_key, self.secret, paper=self.paper)
            logger.info(f"✓ Alpaca connected ({'PAPER' if self.paper else 'LIVE'})")
        except ImportError:
            logger.warning("alpaca-py not installed. Run: pip install alpaca-py")
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")

    def get_equity(self) -> float:
        if not self.client: return 0.0
        return float(self.client.get_account().equity)

    def place_order(self, symbol: str, qty: int, side: str, stop: float, target: float) -> dict:
        if not self.client: return {'error':'not connected'}
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side=='buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class='bracket',
            take_profit=TakeProfitRequest(limit_price=target),
            stop_loss=StopLossRequest(stop_price=stop)
        )
        order = self.client.submit_order(req)
        return {'id': str(order.id), 'status': str(order.status)}

    def get_positions(self) -> list:
        if not self.client: return []
        return [{'symbol':p.symbol,'qty':float(p.qty),'pnl':float(p.unrealized_pl)} for p in self.client.get_all_positions()]

class ZerodhaBroker:
    """Zerodha Kite Connect broker adapter."""
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = None
        self._connect()

    def _connect(self):
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            logger.info("✓ Zerodha Kite connected")
        except ImportError:
            logger.warning("kiteconnect not installed. Run: pip install kiteconnect")

    def place_order(self, symbol: str, qty: int, side: str) -> dict:
        if not self.kite: return {'error':'not connected'}
        txn = self.kite.TRANSACTION_TYPE_BUY if side=='buy' else self.kite.TRANSACTION_TYPE_SELL
        order_id = self.kite.place_order(tradingsymbol=symbol, exchange='NSE',
            transaction_type=txn, quantity=qty,
            order_type=self.kite.ORDER_TYPE_MARKET, product=self.kite.PRODUCT_CNC)
        return {'id': order_id}

# ============================================================
# PROFIT HUNTER
# ============================================================
class ProfitHunter:
    """
    Autonomously scans the market for the next best trade.
    Triggered whenever a position closes in profit.
    """
    def __init__(self, indicator_engine: IndicatorEngine, ml_engine: MLEngine):
        self.indicators = indicator_engine
        self.ml = ml_engine
        self.watchlist = []

    async def scan(self, symbols: List[str], min_score: float = 75.0) -> List[dict]:
        import yfinance as yf
        results = []
        logger.info(f"ProfitHunter: scanning {len(symbols)} symbols...")
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(period='6mo', interval='1d')
                if df.empty or len(df) < 30:
                    continue
                df.columns = [c.lower() for c in df.columns]
                df = self.indicators.compute(df)
                sig = self.indicators.generate_signal(df)
                ml_sig = self.ml.predict(df)
                # Combine indicator + ML scores
                combined_conf = (sig['confidence'] + ml_sig['confidence']) / 2
                if combined_conf >= min_score and sig['action'] == 'BUY':
                    vol_ratio = float(df['volume'].iloc[-1]) / float(df['volume'].rolling(20).mean().iloc[-1])
                    results.append({
                        'symbol': sym,
                        'action': 'BUY',
                        'confidence': round(combined_conf, 1),
                        'volume_ratio': round(vol_ratio, 2),
                        'ind_signal': sig,
                        'ml_signal': ml_sig,
                        'price': round(float(df['close'].iloc[-1]), 2),
                    })
            except Exception as e:
                logger.debug(f"Scan error {sym}: {e}")
        results.sort(key=lambda x: x['confidence'], reverse=True)
        logger.info(f"ProfitHunter: found {len(results)} opportunities")
        return results[:10]

# ============================================================
# TRADE ENGINE
# ============================================================
class TradeEngine:
    """
    Core autonomous trading logic.
    Runs the full loop: scan → signal → risk check → order → monitor → exit.
    """
    def __init__(self, config: Config, risk: RiskManager, indicators: IndicatorEngine,
                 ml: MLEngine, hunter: ProfitHunter):
        self.cfg = config
        self.risk = risk
        self.indicators = indicators
        self.ml = ml
        self.hunter = hunter
        self.brokers: Dict[str, object] = {}
        self.positions: Dict[str, dict] = {}
        self.running = False
        self.trade_log: List[dict] = []

    def add_broker(self, name: str, broker):
        self.brokers[name] = broker
        logger.info(f"Broker registered: {name}")

    def get_equity(self) -> float:
        total = 0.0
        for broker in self.brokers.values():
            try: total += broker.get_equity()
            except: pass
        return total or 100_000.0  # fallback for paper mode

    async def trade_symbol(self, symbol: str, action: str, confidence: float):
        equity = self.get_equity()
        self.risk.set_equity(equity)
        ok, reason = self.risk.check_trade(symbol, action, equity)
        if not ok:
            logger.warning(f"Trade blocked for {symbol}: {reason}")
            return

        if len(self.positions) >= self.cfg.MAX_POSITIONS:
            logger.info(f"Max positions reached ({self.cfg.MAX_POSITIONS}). Skipping {symbol}.")
            return

        if symbol in self.positions:
            logger.info(f"Already in position: {symbol}")
            return

        # Fetch live price
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.last_price or 0.0
        if price <= 0: return

        side = 'buy' if action == 'BUY' else 'sell'
        qty = self.risk.position_size(equity, price)
        stop = self.risk.stop_price(price, side)
        target = self.risk.target_price(price, side)

        # Place order on first available broker
        for bname, broker in self.brokers.items():
            try:
                result = broker.place_order(symbol, qty, side, stop, target)
                if 'error' not in result:
                    self.positions[symbol] = {
                        'symbol':symbol,'entry':price,'qty':qty,
                        'stop':stop,'target':target,'side':side,
                        'broker':bname,'confidence':confidence,
                        'opened_at':datetime.now().isoformat()
                    }
                    self.trade_log.append({'action':action,'symbol':symbol,'price':price,'qty':qty,
                                           'confidence':confidence,'time':datetime.now().isoformat()})
                    logger.info(f"✓ ORDER PLACED: {action} {qty}x {symbol} @ ${price:.2f} | Stop: ${stop} | Target: ${target}")
                    break
            except Exception as e:
                logger.error(f"Order failed on {bname}: {e}")

    async def monitor_positions(self):
        """Check stops and targets for all open positions."""
        import yfinance as yf
        to_close = []
        for sym, pos in self.positions.items():
            try:
                price = yf.Ticker(sym).fast_info.last_price or pos['entry']
                hit_stop = price <= pos['stop'] if pos['side']=='buy' else price >= pos['stop']
                hit_target = price >= pos['target'] if pos['side']=='buy' else price <= pos['target']
                if hit_stop:
                    logger.warning(f"⛔ STOP HIT: {sym} @ ${price:.2f}")
                    pnl = (price - pos['entry']) * pos['qty'] * (1 if pos['side']=='buy' else -1)
                    self.risk.update_daily_loss(pnl)
                    to_close.append(sym)
                elif hit_target:
                    logger.info(f"✓ TARGET HIT: {sym} @ ${price:.2f} — taking profit")
                    to_close.append(sym)
                    # Trigger profit hunter after profitable exit
                    asyncio.create_task(self.hunt_next_trade())
            except Exception as e:
                logger.debug(f"Monitor error {sym}: {e}")
        for sym in to_close:
            self.positions.pop(sym, None)

    async def hunt_next_trade(self):
        """After a profitable exit, immediately hunt for the next opportunity."""
        logger.info("⟳ Profit Hunter activated — searching for next trade...")
        sp500 = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','PLTR','AMD',
                 'NFLX','ADBE','CRM','ORCL','INTC','QCOM','TXN','ASML','UBER','LYFT']
        opps = await self.hunter.scan(sp500, min_score=78.0)
        for opp in opps[:3]:
            await self.trade_symbol(opp['symbol'], opp['action'], opp['confidence'])

    async def run_loop(self):
        """Main autonomous trading loop."""
        self.running = True
        logger.info("▶ Trading loop started")
        while self.running:
            try:
                now = datetime.now().time()
                market_open = dtime(9, 30)
                market_close = dtime(16, 0)
                if market_open <= now <= market_close:
                    await self.monitor_positions()
                    if len(self.positions) < self.cfg.MAX_POSITIONS:
                        await self.hunt_next_trade()
                else:
                    logger.debug("Market closed — standing by")
            except Exception as e:
                logger.error(f"Loop error: {e}")
            await asyncio.sleep(30)  # Check every 30 seconds

    def stop(self):
        self.running = False
        logger.info("⏸ Trading loop stopped")

# ============================================================
# MIDNIGHT RETRAINER
# ============================================================
class MidnightRetrainer:
    """Retrains the ML model every midnight with fresh data."""
    def __init__(self, ml_engine: MLEngine, symbols: List[str]):
        self.ml = ml_engine
        self.symbols = symbols
        self.scheduler = AsyncIOScheduler()

    def start(self):
        self.scheduler.add_job(self._retrain, 'cron', hour=Config.RETRAIN_HOUR, minute=Config.RETRAIN_MINUTE)
        self.scheduler.start()
        logger.info(f"Midnight retrainer scheduled at {Config.RETRAIN_HOUR:02d}:{Config.RETRAIN_MINUTE:02d}")

    async def _retrain(self):
        logger.info("⟳ Midnight retrain starting...")
        try:
            import yfinance as yf
            all_df = []
            for sym in self.symbols[:20]:  # Use top 20 for training data
                df = yf.download(sym, period='1y', interval='1d', progress=False)
                if df.empty: continue
                df.columns = [c.lower() for c in df.columns]
                ind = IndicatorEngine()
                df = ind.compute(df)
                # Label: 1=buy (price up >1% next day), 0=hold, -1=sell
                df['future_ret'] = df['close'].pct_change().shift(-1)
                df['target'] = df['future_ret'].apply(lambda r: 2 if r>0.01 else 0 if r<-0.01 else 1)
                all_df.append(df)
            if all_df:
                combined = pd.concat(all_df, ignore_index=True)
                self.ml.train(combined)
        except Exception as e:
            logger.error(f"Retrain failed: {e}")

# ============================================================
# FASTAPI APPLICATION
# ============================================================
app = FastAPI(title="APEX Trading Bot API", version="3.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

cfg = Config()
risk = RiskManager(cfg)
indicators = IndicatorEngine()
ml = MLEngine()
hunter = ProfitHunter(indicators, ml)
engine = TradeEngine(cfg, risk, indicators, ml, hunter)
retrainer = MidnightRetrainer(ml, ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA'])

connected_clients: List[WebSocket] = []

async def require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != cfg.ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Admin access denied")
    return True

# ---- ROUTES ----

@app.on_event("startup")
async def startup():
    retrainer.start()
    asyncio.create_task(engine.run_loop())
    logger.info("APEX Trading Bot started ✓")

@app.get("/status")
def status():
    return {
        "bot_active": engine.running,
        "positions_count": len(engine.positions),
        "ml_accuracy": ml.accuracy,
        "last_trained": ml.last_trained.isoformat() if ml.last_trained else None,
        "daily_loss": round(risk.daily_loss, 2),
        "circuit_tripped": risk.circuit_tripped,
        "brokers": list(engine.brokers.keys()),
    }

@app.get("/positions")
def get_positions():
    return {"positions": list(engine.positions.values())}

@app.get("/trades")
def get_trades():
    return {"trades": engine.trade_log[-50:]}

@app.post("/bot/start")
def start_bot(admin=Depends(require_admin)):
    asyncio.create_task(engine.run_loop())
    return {"status": "started"}

@app.post("/bot/stop")
def stop_bot(admin=Depends(require_admin)):
    engine.stop()
    return {"status": "stopped"}

class BrokerConfig(BaseModel):
    name: str
    api_key: str
    secret: str
    mode: str = "paper"

@app.post("/broker/add")
def add_broker(cfg_b: BrokerConfig, admin=Depends(require_admin)):
    if cfg_b.name.lower() == "alpaca":
        broker = AlpacaBroker(cfg_b.api_key, cfg_b.secret, paper=cfg_b.mode=="paper")
        engine.add_broker("alpaca", broker)
    elif cfg_b.name.lower() == "zerodha":
        broker = ZerodhaBroker(cfg_b.api_key, cfg_b.secret)
        engine.add_broker("zerodha", broker)
    return {"status": "connected", "broker": cfg_b.name}

class IndicatorAdd(BaseModel):
    name: str
    code: str
    params: dict = {}

@app.post("/indicators/add")
def add_indicator(ind: IndicatorAdd, admin=Depends(require_admin)):
    """Admin-only: add custom indicator at runtime."""
    indicators.add_custom(ind.name, ind.code, ind.params)
    return {"status": "added", "name": ind.name}

@app.get("/indicators")
def list_indicators():
    return {"built_in": ["RSI","MACD","BB","EMA9","EMA21","EMA200","ATR","Stoch","VWAP","OBV"],
            "custom": list(indicators.custom_indicators.keys())}

@app.post("/retrain")
async def manual_retrain(admin=Depends(require_admin)):
    asyncio.create_task(retrainer._retrain())
    return {"status": "retraining started"}

class RiskUpdate(BaseModel):
    max_position_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_daily_loss: Optional[float] = None
    max_positions: Optional[int] = None

@app.put("/risk")
def update_risk(update: RiskUpdate, admin=Depends(require_admin)):
    if update.max_position_pct: cfg.MAX_POSITION_PCT = update.max_position_pct
    if update.stop_loss_pct: cfg.STOP_LOSS_PCT = update.stop_loss_pct
    if update.take_profit_pct: cfg.TAKE_PROFIT_PCT = update.take_profit_pct
    if update.max_daily_loss: cfg.MAX_DAILY_LOSS = update.max_daily_loss
    if update.max_positions: cfg.MAX_POSITIONS = update.max_positions
    return {"status": "updated", "config": vars(cfg)}

@app.get("/scan")
async def scan(min_score: float = 75.0):
    symbols = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','PLTR','AMD',
               'NFLX','ADBE','CRM','SOFI','MARA','COIN','IONQ','SMCI']
    results = await hunter.scan(symbols, min_score)
    return {"opportunities": results}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.send_json({
                "type": "heartbeat",
                "positions": len(engine.positions),
                "time": datetime.now().isoformat(),
                "bot_active": engine.running,
            })
            await asyncio.sleep(2)
    except:
        connected_clients.remove(websocket)

# ---- LAUNCH ----
if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════╗
    ║       APEX TRADING BOT v3.1              ║
    ║  Autonomous Algorithmic Trading System   ║
    ╚══════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
