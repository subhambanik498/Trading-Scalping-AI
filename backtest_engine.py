"""
Backtesting Engine
Historical simulation of trading strategy to validate performance
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single backtest trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    commission: float = 0.0
    slippage: float = 0.0
    
    def get_duration_minutes(self) -> float:
        """Get trade duration in minutes"""
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    avg_trade_duration_minutes: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    starting_capital: float = 0.0
    ending_capital: float = 0.0
    total_return_percent: float = 0.0
    
    trades: List[BacktestTrade] = field(default_factory=list)


class BacktestingEngine:
    """
    Backtesting engine for strategy validation
    Simulates historical trading based on generated signals
    """
    
    def __init__(self, starting_capital: float = 100000.0):
        """
        Initialize backtesting engine
        
        Args:
            starting_capital: Initial capital for backtesting
        """
        self.starting_capital = starting_capital
        self.capital = starting_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    def backtest_strategy(self, symbol: str, hist_data: pd.DataFrame,
                         signal_generator, config: Dict) -> BacktestMetrics:
        """
        Run backtest on historical data
        
        Args:
            symbol: Stock symbol
            hist_data: Historical OHLCV data
            signal_generator: Function to generate trading signals
            config: Strategy configuration
            
        Returns:
            BacktestMetrics with detailed performance stats
        """
        try:
            logger.info(f"Starting backtest for {symbol}")
            
            # Reset state
            self.capital = self.starting_capital
            self.positions = {}
            self.trades = []
            self.equity_curve = []
            
            # Iterate through historical data
            for idx in range(len(hist_data)):
                current_row = hist_data.iloc[idx]
                current_date = hist_data.index[idx]
                
                # Update position P&L
                self._update_positions(current_row, current_date)
                
                # Generate signal
                signal = signal_generator(
                    hist_data=hist_data.iloc[:idx+1],
                    current_price=current_row['Close'],
                    symbol=symbol,
                    config=config
                )
                
                # Execute trade
                if signal:
                    self._execute_backtest_trade(
                        symbol=symbol,
                        signal=signal,
                        current_price=current_row['Close'],
                        current_date=current_date,
                        config=config
                    )
                
                # Record equity
                self.equity_curve.append((current_date, self.capital))
            
            # Calculate metrics
            metrics = self._calculate_metrics(symbol)
            
            logger.info(f"Backtest complete: {metrics.total_trades} trades, "
                       f"Win Rate: {metrics.win_rate:.1f}%, "
                       f"Total P&L: ₹{metrics.total_pnl:.2f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return BacktestMetrics()
    
    def _execute_backtest_trade(self, symbol: str, signal: Dict, 
                                current_price: float, current_date: datetime,
                                config: Dict):
        """Execute a backtest trade"""
        try:
            if signal['action'] == 'BUY' and symbol not in self.positions:
                # Calculate position size
                risk_percent = config.get('max_risk_percent', 0.02)
                max_loss = self.capital * risk_percent
                
                stop_loss = signal['stop_loss']
                risk_per_share = abs(current_price - stop_loss)
                
                if risk_per_share > 0:
                    quantity = int(max_loss / risk_per_share)
                    
                    if quantity > 0:
                        position_value = quantity * current_price
                        
                        # Check if we have enough capital
                        if position_value <= self.capital * 0.5:  # Use max 50% of capital
                            # Add commission and slippage
                            commission = position_value * 0.0005  # 0.05% commission
                            
                            self.positions[symbol] = {
                                'entry_price': current_price,
                                'entry_date': current_date,
                                'quantity': quantity,
                                'stop_loss': stop_loss,
                                'take_profit': signal['take_profit'],
                                'commission': commission
                            }
                            
                            self.capital -= commission
            
            elif signal['action'] == 'SELL' and symbol in self.positions:
                # Close position
                position = self.positions[symbol]
                
                pnl = (current_price - position['entry_price']) * position['quantity']
                pnl -= position['commission']
                
                trade = BacktestTrade(
                    symbol=symbol,
                    entry_time=position['entry_date'],
                    exit_time=current_date,
                    entry_price=position['entry_price'],
                    exit_price=current_price,
                    quantity=position['quantity'],
                    pnl=pnl,
                    pnl_percent=(pnl / (position['entry_price'] * position['quantity'])) * 100,
                    commission=position['commission']
                )
                
                self.trades.append(trade)
                self.capital += (position['quantity'] * current_price)
                del self.positions[symbol]
        
        except Exception as e:
            logger.error(f"Error executing backtest trade: {e}")
    
    def _update_positions(self, current_row: pd.Series, current_date: datetime):
        """Update unrealized P&L for open positions"""
        try:
            for symbol, position in self.positions.items():
                # Check stop loss
                if current_row['Low'] <= position['stop_loss']:
                    pnl = (position['stop_loss'] - position['entry_price']) * position['quantity']
                    pnl -= position['commission']
                    
                    trade = BacktestTrade(
                        symbol=symbol,
                        entry_time=position['entry_date'],
                        exit_time=current_date,
                        entry_price=position['entry_price'],
                        exit_price=position['stop_loss'],
                        quantity=position['quantity'],
                        pnl=pnl,
                        pnl_percent=(pnl / (position['entry_price'] * position['quantity'])) * 100,
                        commission=position['commission']
                    )
                    
                    self.trades.append(trade)
                    self.capital += (position['quantity'] * position['stop_loss'])
                    del self.positions[symbol]
                
                # Check take profit
                elif current_row['High'] >= position['take_profit']:
                    pnl = (position['take_profit'] - position['entry_price']) * position['quantity']
                    pnl -= position['commission']
                    
                    trade = BacktestTrade(
                        symbol=symbol,
                        entry_time=position['entry_date'],
                        exit_time=current_date,
                        entry_price=position['entry_price'],
                        exit_price=position['take_profit'],
                        quantity=position['quantity'],
                        pnl=pnl,
                        pnl_percent=(pnl / (position['entry_price'] * position['quantity'])) * 100,
                        commission=position['commission']
                    )
                    
                    self.trades.append(trade)
                    self.capital += (position['quantity'] * position['take_profit'])
                    del self.positions[symbol]
        
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _calculate_metrics(self, symbol: str) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        try:
            if not self.trades:
                return BacktestMetrics()
            
            metrics = BacktestMetrics()
            metrics.trades = self.trades
            metrics.starting_capital = self.starting_capital
            metrics.ending_capital = self.capital
            metrics.total_return_percent = ((self.capital - self.starting_capital) / self.starting_capital) * 100
            
            # Trade statistics
            metrics.total_trades = len(self.trades)
            metrics.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
            metrics.losing_trades = sum(1 for t in self.trades if t.pnl < 0)
            
            if metrics.total_trades > 0:
                metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
            
            # P&L statistics
            metrics.total_pnl = sum(t.pnl for t in self.trades)
            metrics.gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
            metrics.gross_loss = sum(t.pnl for t in self.trades if t.pnl < 0)
            
            if metrics.winning_trades > 0:
                metrics.avg_win = metrics.gross_profit / metrics.winning_trades
            
            if metrics.losing_trades > 0:
                metrics.avg_loss = metrics.gross_loss / metrics.losing_trades
            
            if abs(metrics.gross_loss) > 0:
                metrics.profit_factor = metrics.gross_profit / abs(metrics.gross_loss)
            
            # Extremes
            metrics.largest_win = max((t.pnl for t in self.trades), default=0)
            metrics.largest_loss = min((t.pnl for t in self.trades), default=0)
            
            # Trade duration
            durations = [t.get_duration_minutes() for t in self.trades]
            if durations:
                metrics.avg_trade_duration_minutes = np.mean(durations)
            
            # Consecutive trades
            metrics.max_consecutive_wins = self._calculate_max_consecutive_wins()
            metrics.max_consecutive_losses = self._calculate_max_consecutive_losses()
            
            # Drawdown
            metrics.max_drawdown, metrics.max_drawdown_percent = self._calculate_max_drawdown()
            
            # Risk-adjusted returns
            metrics.sharpe_ratio = self._calculate_sharpe_ratio()
            metrics.sortino_ratio = self._calculate_sortino_ratio()
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return BacktestMetrics()
    
    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades"""
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades"""
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_max_drawdown(self) -> Tuple[float, float]:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0, 0
        
        equity_values = [e[1] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            
            dd = peak - value
            max_dd = max(max_dd, dd)
        
        max_dd_percent = (max_dd / self.starting_capital) * 100 if self.starting_capital > 0 else 0
        
        return max_dd, max_dd_percent
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe Ratio"""
        try:
            if not self.trades:
                return 0
            
            returns = [t.pnl_percent for t in self.trades]
            
            if len(returns) < 2:
                return 0
            
            avg_return = np.mean(returns)
            std_dev = np.std(returns)
            
            if std_dev == 0:
                return 0
            
            # Annualized
            sharpe = ((avg_return - risk_free_rate) / std_dev) * np.sqrt(252)
            
            return sharpe
        
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino Ratio"""
        try:
            if not self.trades:
                return 0
            
            returns = np.array([t.pnl_percent for t in self.trades])
            
            if len(returns) < 2:
                return 0
            
            avg_return = np.mean(returns)
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 0
            
            downside_std = np.std(downside_returns)
            
            if downside_std == 0:
                return 0
            
            # Annualized
            sortino = ((avg_return - risk_free_rate) / downside_std) * np.sqrt(252)
            
            return sortino
        
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0
    
    def print_report(self, metrics: BacktestMetrics):
        """Print detailed backtest report"""
        print("\n" + "="*60)
        print("BACKTEST REPORT")
        print("="*60)
        print(f"Starting Capital: ₹{metrics.starting_capital:,.2f}")
        print(f"Ending Capital:   ₹{metrics.ending_capital:,.2f}")
        print(f"Total Return:     {metrics.total_return_percent:.2f}%")
        print(f"Total P&L:        ₹{metrics.total_pnl:,.2f}")
        print()
        print(f"Total Trades:     {metrics.total_trades}")
        print(f"Winning Trades:   {metrics.winning_trades}")
        print(f"Losing Trades:    {metrics.losing_trades}")
        print(f"Win Rate:         {metrics.win_rate:.1f}%")
        print()
        print(f"Avg Win:          ₹{metrics.avg_win:,.2f}")
        print(f"Avg Loss:         ₹{metrics.avg_loss:,.2f}")
        print(f"Profit Factor:    {metrics.profit_factor:.2f}")
        print()
        print(f"Largest Win:      ₹{metrics.largest_win:,.2f}")
        print(f"Largest Loss:     ₹{metrics.largest_loss:,.2f}")
        print()
        print(f"Consecutive Wins: {metrics.max_consecutive_wins}")
        print(f"Consecutive Loss: {metrics.max_consecutive_losses}")
        print()
        print(f"Max Drawdown:     ₹{metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
        print()
        print(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:    {metrics.sortino_ratio:.2f}")
        print(f"Avg Trade Time:   {metrics.avg_trade_duration_minutes:.1f} minutes")
        print("="*60 + "\n")
