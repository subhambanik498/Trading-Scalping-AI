"""
Bot Controller and Orchestrator
Main orchestrator that coordinates all modules and runs the trading bot
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from threading import Thread, Event
import json

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot controller
    Orchestrates all trading modules and makes trading decisions
    """
    
    def __init__(self, config: Dict, broker_api, market_analyzer, order_executor, 
                 risk_manager, ai_analyzer):
        """
        Initialize the trading bot
        
        Args:
            config: Configuration dictionary
            broker_api: Broker API connector
            market_analyzer: Market analysis module
            order_executor: Order execution module
            risk_manager: Risk management module
            ai_analyzer: AI chart analysis module
        """
        self.config = config
        self.broker_api = broker_api
        self.market_analyzer = market_analyzer
        self.order_executor = order_executor
        self.risk_manager = risk_manager
        self.ai_analyzer = ai_analyzer
        
        # Bot state
        self.is_running = False
        self.stop_event = Event()
        self.trading_active = config.get('trading_active', False)
        self.scan_interval = config.get('scan_interval', 60)
        self.watchlist = config.get('watchlist', [])
        
        # Statistics
        self.session_stats = {
            'scan_count': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Trading Bot...")
        self.is_running = True
        self.session_stats['start_time'] = datetime.now()
        
        # Reset daily limits
        self.risk_manager.reset_daily_limits()
        
        # Start monitoring thread
        monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
        logger.info("Trading Bot started successfully")
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping Trading Bot...")
        self.is_running = False
        self.stop_event.set()
        self.session_stats['end_time'] = datetime.now()
        
        # Close all positions
        self._close_all_positions()
        
        logger.info("Trading Bot stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running and not self.stop_event.is_set():
            try:
                self._scan_for_opportunities()
                self._manage_positions()
                self._check_order_fills()
                time.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)
    
    def _scan_for_opportunities(self):
        """Scan watchlist for opportunities using AI analysis"""
        self.session_stats['scan_count'] += 1
        logger.info(f"Scanning {len(self.watchlist)} stocks...")
        
        for symbol in self.watchlist:
            try:
                if self.order_executor.get_position(symbol):
                    continue
                
                # Get live data
                stock_data = self.market_analyzer.fetcher.fetch_nse_stock_data(symbol)
                if not stock_data:
                    continue
                
                # Get historical data
                hist_data = self.market_analyzer.fetch_historical_data(symbol, days=30)
                if hist_data is None or hist_data.empty:
                    continue
                
                # Create chart for AI analysis
                chart_path = self._create_chart(symbol, hist_data)
                if not chart_path:
                    continue
                
                # AI chart analysis with screenshot
                ai_rec = self.ai_analyzer.analyze_chart(
                    chart_path=chart_path,
                    symbol=symbol,
                    current_price=stock_data.current_price
                )
                
                if not ai_rec:
                    continue
                
                # Get technical indicators
                rsi = self.market_analyzer.calculate_rsi(symbol)
                trend = self.market_analyzer.detect_trend(symbol)
                macd = self.market_analyzer.calculate_macd(symbol)
                
                # Generate trading signal
                signal = self._generate_trading_signal(
                    symbol=symbol,
                    stock_data=stock_data,
                    rsi=rsi[-1] if rsi else None,
                    trend=trend,
                    macd=macd['macd'][-1] if macd else None,
                    ai_recommendation=ai_rec
                )
                
                if signal:
                    self.session_stats['signals_generated'] += 1
                    logger.info(f"Signal for {symbol}:")
                    logger.info(f"  Action: {signal['action']}")
                    logger.info(f"  Entry: ₹{signal['entry_price']:.2f}")
                    logger.info(f"  SL: ₹{signal['stop_loss']:.2f}")
                    logger.info(f"  TP: ₹{signal['take_profit']:.2f}")
                    logger.info(f"  Confidence: {signal['confidence']}%")
                    
                    if self.trading_active:
                        self._execute_trade(symbol, signal)
            
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
    
    def _generate_trading_signal(self, symbol: str, stock_data, rsi: float,
                                 trend: str, macd: float,
                                 ai_recommendation) -> Optional[Dict]:
        """Generate signal combining AI + technical analysis"""
        try:
            signal_confidence = 0
            
            # AI recommendation (40% weight)
            if ai_recommendation.action == "BUY":
                signal_confidence += ai_recommendation.confidence * 0.4
            elif ai_recommendation.action == "SELL":
                signal_confidence -= ai_recommendation.confidence * 0.4
            
            # Technical indicators (60% weight)
            # Trend (20% weight)
            if trend == "UPTREND":
                signal_confidence += 20
            elif trend == "DOWNTREND":
                signal_confidence -= 20
            
            # RSI (20% weight)
            if rsi and rsi < 30:
                signal_confidence += 20
            elif rsi and rsi > 70:
                signal_confidence -= 20
            
            # MACD (20% weight)
            if macd and macd > 0:
                signal_confidence += 20
            else:
                signal_confidence -= 20
            
            # Only BUY signals with confidence >= 60%
            if signal_confidence >= 60:
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'current_price': stock_data.current_price,
                    'entry_price': ai_recommendation.entry_price,
                    'stop_loss': ai_recommendation.stop_loss,
                    'take_profit': ai_recommendation.take_profit,
                    'confidence': int(signal_confidence),
                    'ai_confidence': ai_recommendation.confidence,
                    'rr_ratio': ai_recommendation.risk_reward_ratio
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _execute_trade(self, symbol: str, signal: Dict):
        """Execute trade based on signal"""
        try:
            logger.info(f"Executing trade for {symbol}")
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                max_risk_percent=self.config.get('max_risk_percent', 0.02)
            )
            
            if position_size <= 0:
                logger.warning(f"Position size invalid for {symbol}")
                return
            
            # Check risk criteria
            opportunity = {
                'symbol': symbol,
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'position_size': position_size
            }
            
            if not self.risk_manager.can_take_trade(opportunity):
                logger.warning(f"Trade rejected by risk manager")
                return
            
            # Place bracket order
            bracket_order = self.order_executor.place_bracket_order(
                symbol=symbol,
                quantity=int(position_size),
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
            
            if bracket_order:
                self.session_stats['trades_executed'] += 1
                logger.info(f"Trade executed successfully for {symbol}")
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _manage_positions(self):
        """Monitor open positions"""
        try:
            positions = self.order_executor.get_positions()
            
            for symbol, position in positions.items():
                try:
                    stock_data = self.market_analyzer.fetcher.fetch_nse_stock_data(symbol)
                    if stock_data:
                        position.current_price = stock_data.current_price
                        pnl = position.get_current_pnl()
                        pnl_percent = position.get_pnl_percent()
                        
                        logger.info(f"{symbol}: P&L ₹{pnl:.2f} ({pnl_percent:.2f}%)")
                        
                        if position.is_stop_hit():
                            logger.warning(f"SL hit for {symbol}")
                            self.order_executor.close_position(symbol, position.stop_loss)
                        elif position.is_target_hit():
                            logger.info(f"TP hit for {symbol}")
                            self.order_executor.close_position(symbol, position.take_profit)
                
                except Exception as e:
                    logger.error(f"Error managing {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error in position management: {e}")
    
    def _check_order_fills(self):
        """Check order fill status"""
        try:
            open_orders = self.order_executor.get_open_orders()
            
            for order in open_orders:
                try:
                    order_status = self.broker_api.get_order_status(order.order_id)
                    
                    if order_status:
                        self.order_executor.update_order_status(
                            order_id=order.order_id,
                            filled_quantity=order_status.get('filled_qty', 0),
                            average_price=order_status.get('avg_price', order.price)
                        )
                
                except Exception as e:
                    logger.error(f"Error checking order {order.order_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error checking fills: {e}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        try:
            positions = self.order_executor.get_positions()
            
            for symbol in positions:
                try:
                    stock_data = self.market_analyzer.fetcher.fetch_nse_stock_data(symbol)
                    if stock_data:
                        self.order_executor.close_position(symbol, stock_data.current_price)
                        logger.info(f"Closed {symbol}")
                
                except Exception as e:
                    logger.error(f"Error closing {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def _create_chart(self, symbol: str, hist_data) -> str:
        """Create chart screenshot for AI analysis"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Price chart
            ax1.plot(hist_data.index, hist_data['Close'], label='Close', linewidth=2, color='blue')
            ax1.fill_between(hist_data.index, hist_data['Low'], hist_data['High'], alpha=0.1, color='blue')
            ax1.set_title(f'{symbol} - Price Chart')
            ax1.set_ylabel('Price (₹)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            colors = ['green' if hist_data['Close'].iloc[i] >= hist_data['Open'].iloc[i] else 'red' 
                     for i in range(len(hist_data))]
            ax2.bar(hist_data.index, hist_data['Volume'], label='Volume', color=colors, alpha=0.7)
            ax2.set_title('Volume Analysis')
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Save
            from datetime import datetime as dt
            chart_path = f'/tmp/{symbol}_chart_{dt.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.tight_layout()
            plt.savefig(chart_path, dpi=100)
            plt.close()
            
            logger.info(f"Chart saved: {chart_path}")
            return chart_path
        
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return None
    
    def add_to_watchlist(self, symbol: str):
        """Add to watchlist"""
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            logger.info(f"Added {symbol}")
    
    def remove_from_watchlist(self, symbol: str):
        """Remove from watchlist"""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            logger.info(f"Removed {symbol}")
    
    def get_status(self) -> Dict:
        """Get bot status"""
        return {
            'is_running': self.is_running,
            'trading_active': self.trading_active,
            'watchlist_size': len(self.watchlist),
            'open_positions': len(self.order_executor.get_positions()),
            'open_orders': len(self.order_executor.get_open_orders()),
            'scans': self.session_stats['scan_count'],
            'signals': self.session_stats['signals_generated'],
            'trades': self.session_stats['trades_executed']
        }
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        return {
            **self.session_stats,
            'positions': len(self.order_executor.get_positions()),
            'orders': len(self.order_executor.get_open_orders()),
            'trades_summary': self.order_executor.get_trade_summary()
        }


def create_bot(config: Dict, broker_api, market_analyzer, order_executor, 
               risk_manager, ai_analyzer) -> TradingBot:
    """Factory function to create bot"""
    logger.info("Creating Trading Bot...")
    bot = TradingBot(config, broker_api, market_analyzer, order_executor, 
                     risk_manager, ai_analyzer)
    return bot
