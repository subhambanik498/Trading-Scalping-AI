"""
Indian Stock Market Data Handler
Fetches live market data from Indian exchanges (NSE/BSE) and provides analysis
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndianStockData:
    """Live market data for Indian stocks"""
    symbol: str
    company_name: str
    exchange: str  # NSE or BSE
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    market_cap: float
    pe_ratio: float
    dividend_yield: float
    timestamp: datetime
    
    def get_price_change(self) -> float:
        """Get price change from open"""
        return self.current_price - self.open_price
    
    def get_price_change_percent(self) -> float:
        """Get percentage change from open"""
        if self.open_price == 0:
            return 0
        return ((self.current_price - self.open_price) / self.open_price) * 100


class IndianStockMarketFetcher:
    """
    Fetch live market data for Indian stocks from NSE/BSE
    Uses multiple data sources for reliability
    """
    
    # Popular Indian stocks on NSE
    MAJOR_INDICES = {
        'NIFTY50': '^NSEI',
        'SENSEX': '^BSESN',
        'NIFTY_IT': '^NSEOIT',
        'NIFTY_BANK': '^NSEBANK',
        'NIFTY_AUTO': '^CNXIT',
    }
    
    # Sample of major Indian stocks
    MAJOR_STOCKS = {
        'TCS': 'Tata Consultancy Services',
        'INFY': 'Infosys Limited',
        'WIPRO': 'Wipro Limited',
        'HCL': 'HCL Technologies',
        'RELIANCE': 'Reliance Industries',
        'HDFC': 'Housing Development Finance',
        'ICICIBANK': 'ICICI Bank Limited',
        'SBIN': 'State Bank of India',
        'BAJAJFINSV': 'Bajaj Finserv',
        'MARUTI': 'Maruti Suzuki India',
        'BAJAJ-AUTO': 'Bajaj Auto',
        'SUNPHARMA': 'Sun Pharmaceutical',
        'ASIANPAINT': 'Asian Paints',
        'AXISBANK': 'Axis Bank',
        'LTTS': 'L&T Technology Services',
    }
    
    def __init__(self, config=None):
        self.config = config or {}
        self.cache = {}
        self.cache_duration = timedelta(minutes=1)
        self.last_fetch_time = {}
        
    def fetch_nse_stock_data(self, symbol: str) -> Optional[IndianStockData]:
        """
        Fetch live data for Indian stock from NSE
        
        Args:
            symbol: Stock symbol (e.g., 'TCS', 'INFY')
            
        Returns:
            IndianStockData object or None if fetch fails
        """
        try:
            # Check cache first
            if self._is_cache_valid(symbol):
                return self.cache.get(symbol)
            
            # Fetch from yfinance (supports Indian stocks)
            import yfinance as yf
            
            # NSE symbols need .NS suffix
            nse_symbol = f"{symbol}.NS"
            stock = yf.Ticker(nse_symbol)
            
            # Get current data
            info = stock.info
            
            stock_data = IndianStockData(
                symbol=symbol,
                company_name=info.get('longName', symbol),
                exchange='NSE',
                current_price=float(info.get('currentPrice', 0)),
                open_price=float(info.get('open', 0)),
                high_price=float(info.get('dayHigh', 0)),
                low_price=float(info.get('dayLow', 0)),
                close_price=float(info.get('previousClose', 0)),
                volume=int(info.get('volume', 0)),
                market_cap=float(info.get('marketCap', 0)),
                pe_ratio=float(info.get('trailingPE', 0)),
                dividend_yield=float(info.get('dividendYield', 0)),
                timestamp=datetime.now()
            )
            
            # Cache the data
            self.cache[symbol] = stock_data
            self.last_fetch_time[symbol] = datetime.now()
            
            logger.info(f"Fetched NSE data for {symbol}: ₹{stock_data.current_price}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching NSE data for {symbol}: {e}")
            return None
    
    def fetch_multiple_stocks(self, symbols: List[str]) -> Dict[str, IndianStockData]:
        """Fetch data for multiple Indian stocks"""
        results = {}
        for symbol in symbols:
            data = self.fetch_nse_stock_data(symbol)
            if data:
                results[symbol] = data
        return results
    
    def fetch_nifty50_constituents(self) -> Dict[str, IndianStockData]:
        """Fetch data for all NIFTY 50 constituent stocks"""
        return self.fetch_multiple_stocks(list(self.MAJOR_STOCKS.keys()))
    
    def fetch_index_data(self, index_name: str) -> Optional[Dict]:
        """
        Fetch major Indian index data
        
        Args:
            index_name: 'NIFTY50', 'SENSEX', 'NIFTY_IT', etc.
            
        Returns:
            Index data dictionary
        """
        try:
            import yfinance as yf
            
            symbol = self.MAJOR_INDICES.get(index_name)
            if not symbol:
                logger.warning(f"Unknown index: {index_name}")
                return None
            
            index = yf.Ticker(symbol)
            info = index.info
            
            return {
                'index': index_name,
                'symbol': symbol,
                'current_price': float(info.get('currentPrice', 0)),
                'open': float(info.get('open', 0)),
                'high': float(info.get('dayHigh', 0)),
                'low': float(info.get('dayLow', 0)),
                'close': float(info.get('previousClose', 0)),
                'volume': int(info.get('volume', 0)),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching index data for {index_name}: {e}")
            return None
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self.cache:
            return False
        
        last_fetch = self.last_fetch_time.get(symbol)
        if not last_fetch:
            return False
        
        return datetime.now() - last_fetch < self.cache_duration


class IndianMarketAnalyzer:
    """
    Analyze Indian stock market data for trading opportunities
    Includes technical analysis, momentum detection, and market screening
    """
    
    def __init__(self, fetcher: IndianStockMarketFetcher):
        self.fetcher = fetcher
        self.historical_data = {}
    
    def fetch_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for technical analysis
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
            
            nse_symbol = f"{symbol}.NS"
            stock = yf.Ticker(nse_symbol)
            
            # Fetch historical data
            hist = stock.history(period=f'{days}d')
            self.historical_data[symbol] = hist
            
            logger.info(f"Fetched {len(hist)} days of historical data for {symbol}")
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def calculate_sma(self, symbol: str, period: int = 20) -> Optional[List[float]]:
        """Calculate Simple Moving Average"""
        try:
            if symbol not in self.historical_data:
                self.fetch_historical_data(symbol)
            
            df = self.historical_data.get(symbol)
            if df is None or df.empty:
                return None
            
            sma = df['Close'].rolling(window=period).mean()
            return sma.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating SMA for {symbol}: {e}")
            return None
    
    def calculate_ema(self, symbol: str, period: int = 12) -> Optional[List[float]]:
        """Calculate Exponential Moving Average"""
        try:
            if symbol not in self.historical_data:
                self.fetch_historical_data(symbol)
            
            df = self.historical_data.get(symbol)
            if df is None or df.empty:
                return None
            
            ema = df['Close'].ewm(span=period, adjust=False).mean()
            return ema.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating EMA for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[List[float]]:
        """
        Calculate Relative Strength Index
        RSI > 70: Overbought (potential sell signal)
        RSI < 30: Oversold (potential buy signal)
        """
        try:
            if symbol not in self.historical_data:
                self.fetch_historical_data(symbol)
            
            df = self.historical_data.get(symbol)
            if df is None or df.empty:
                return None
            
            close = df['Close']
            delta = close.diff()
            
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return None
    
    def calculate_macd(self, symbol: str) -> Optional[Dict]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        MACD Line: 12-day EMA - 26-day EMA
        Signal Line: 9-day EMA of MACD
        """
        try:
            if symbol not in self.historical_data:
                self.fetch_historical_data(symbol)
            
            df = self.historical_data.get(symbol)
            if df is None or df.empty:
                return None
            
            close = df['Close']
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.tolist(),
                'signal': signal_line.tolist(),
                'histogram': histogram.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD for {symbol}: {e}")
            return None
    
    def detect_trend(self, symbol: str) -> Optional[str]:
        """
        Detect current trend (Uptrend, Downtrend, Sideways)
        Based on SMA comparison
        """
        try:
            if symbol not in self.historical_data:
                self.fetch_historical_data(symbol)
            
            df = self.historical_data.get(symbol)
            if df is None or df.empty or len(df) < 50:
                return None
            
            sma_20 = df['Close'].rolling(window=20).mean()
            sma_50 = df['Close'].rolling(window=50).mean()
            
            current_price = df['Close'].iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            
            if current_sma_20 > current_sma_50 and current_price > current_sma_20:
                return "UPTREND"
            elif current_sma_20 < current_sma_50 and current_price < current_sma_20:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
            
        except Exception as e:
            logger.error(f"Error detecting trend for {symbol}: {e}")
            return None
    
    def screen_bullish_stocks(self, symbols: List[str]) -> List[Dict]:
        """
        Screen for bullish trading opportunities
        Criteria:
        - Price above 20-day SMA
        - RSI < 70 (not overbought)
        - MACD positive
        - Uptrend detected
        """
        bullish_stocks = []
        
        for symbol in symbols:
            try:
                # Fetch data
                self.fetch_historical_data(symbol)
                df = self.historical_data.get(symbol)
                
                if df is None or len(df) < 50:
                    continue
                
                # Get indicators
                rsi = self.calculate_rsi(symbol)
                trend = self.detect_trend(symbol)
                macd = self.calculate_macd(symbol)
                
                current_price = df['Close'].iloc[-1]
                sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
                
                current_rsi = rsi[-1] if rsi else None
                current_macd = macd['macd'][-1] if macd else None
                
                # Apply screening criteria
                if (current_price > sma_20 and 
                    current_rsi and current_rsi < 70 and 
                    current_macd and current_macd > 0 and
                    trend == "UPTREND"):
                    
                    bullish_stocks.append({
                        'symbol': symbol,
                        'price': current_price,
                        'rsi': current_rsi,
                        'trend': trend,
                        'macd': current_macd
                    })
                
            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                continue
        
        return sorted(bullish_stocks, key=lambda x: x['rsi'])
    
    def get_market_summary(self) -> Dict:
        """Get summary of major Indian market indices"""
        indices = ['NIFTY50', 'SENSEX', 'NIFTY_IT', 'NIFTY_BANK']
        summary = {}
        
        for index in indices:
            data = self.fetcher.fetch_index_data(index)
            if data:
                summary[index] = data
        
        return summary
