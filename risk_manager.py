"""
Risk Management System
Ensures the bot protects capital and never makes catastrophic losses
"""

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for a potential trade"""
    max_position_value: float
    risk_amount: float
    stop_loss_distance: float
    risk_reward_ratio: float
    portfolio_risk_percent: float
    
    def is_acceptable(self, min_rr_ratio: float = 1.0, max_risk_percent: float = 2.0) -> bool:
        """Check if trade meets minimum risk criteria"""
        return (self.risk_reward_ratio >= min_rr_ratio and 
                self.portfolio_risk_percent <= max_risk_percent)


class RiskManager:
    """
    Manages all risk aspects of trading
    - Position sizing
    - Loss limits
    - Risk/reward ratios
    """
    
    def __init__(self, config):
        self.config = config
        self.daily_trades = 0
        self.daily_losses = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        
    def can_take_trade(self, opportunity) -> bool:
        """
        Determine if a trade should be taken based on risk criteria
        
        Args:
            opportunity: TradeOpportunity object
            
        Returns:
            bool: True if trade meets all risk criteria
        """
        try:
            # Check daily loss limit
            if self.daily_losses >= self.config.MAX_DAILY_LOSS:
                logger.warning("Daily loss limit reached")
                return False
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.warning(f"Consecutive loss limit reached: {self.consecutive_losses}")
                return False
            
            # Calculate risk metrics
            metrics = self.calculate_risk_metrics(opportunity)
            
            # Verify metrics
            if not metrics.is_acceptable(
                min_rr_ratio=1.0,
                max_risk_percent=self.config.MAX_LOSS_PER_TRADE * 100
            ):
                logger.info(f"Trade {opportunity.symbol} rejected: insufficient risk/reward")
                return False
            
            # Check position size
            if metrics.max_position_value <= self.config.MIN_TRADE_CAPITAL:
                logger.info(f"Trade {opportunity.symbol} rejected: position too small")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False
    
    def calculate_risk_metrics(self, opportunity) -> RiskMetrics:
        """Calculate detailed risk metrics for a trade"""
        
        entry_price = opportunity.entry_price
        stop_loss = opportunity.stop_loss
        take_profit = opportunity.take_profit
        
        # Calculate distances
        stop_loss_distance = abs(entry_price - stop_loss)
        take_profit_distance = abs(take_profit - entry_price)
        
        # Risk/Reward ratio
        risk_reward_ratio = take_profit_distance / stop_loss_distance if stop_loss_distance > 0 else 0
        
        # Position sizing based on risk
        risk_amount = stop_loss_distance * opportunity.position_size
        portfolio_value = 100000  # Example portfolio value
        portfolio_risk_percent = (risk_amount / portfolio_value) * 100
        
        # Max position value
        max_position_value = entry_price * opportunity.position_size
        
        return RiskMetrics(
            max_position_value=max_position_value,
            risk_amount=risk_amount,
            stop_loss_distance=stop_loss_distance,
            risk_reward_ratio=risk_reward_ratio,
            portfolio_risk_percent=portfolio_risk_percent
        )
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               max_risk_percent: float) -> float:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price
            max_risk_percent: Maximum risk as percentage of capital
            
        Returns:
            Position size in shares
        """
        try:
            # Simplified calculation
            stop_distance = abs(entry_price - stop_loss)
            
            if stop_distance == 0:
                return 0
            
            # Risk amount based on max loss
            portfolio_value = 100000  # This should come from portfolio manager
            risk_amount = portfolio_value * max_risk_percent
            
            # Position size = risk amount / stop distance
            position_size = risk_amount / stop_distance
            
            # Apply maximum position size constraint
            max_shares = (portfolio_value * self.config.MAX_POSITION_SIZE) / entry_price
            position_size = min(position_size, max_shares)
            
            # Ensure minimum trade
            if position_size * entry_price < self.config.MIN_TRADE_CAPITAL:
                return 0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def record_trade_result(self, pnl: float, is_win: bool):
        """Record the result of a trade for risk tracking"""
        self.daily_trades += 1
        
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.daily_losses += abs(pnl)
    
    def reset_daily_limits(self):
        """Reset daily counters at start of new trading day"""
        self.daily_trades = 0
        self.daily_losses = 0.0
        self.consecutive_losses = 0
        logger.info("Daily limits reset")
    
    def get_risk_report(self) -> dict:
        """Get current risk metrics report"""
        return {
            'daily_trades': self.daily_trades,
            'daily_losses': self.daily_losses,
            'consecutive_losses': self.consecutive_losses,
            'daily_loss_remaining': self.config.MAX_DAILY_LOSS - self.daily_losses
        }


class PositionRiskCalculator:
    """Calculate real-time risk for open positions"""
    
    @staticmethod
    def calculate_current_risk(position, current_price: float) -> dict:
        """Calculate current risk metrics for a position"""
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        # Current P&L
        pnl = (current_price - entry_price) * quantity
        pnl_percent = (current_price - entry_price) / entry_price if entry_price != 0 else 0
        
        # Risk to current position
        risk_to_stop = (current_price - stop_loss) * quantity
        profit_to_target = (take_profit - current_price) * quantity
        
        return {
            'current_pnl': pnl,
            'pnl_percent': pnl_percent,
            'risk_to_stop': risk_to_stop,
            'profit_to_target': profit_to_target,
            'risk_reward_ratio': abs(profit_to_target / risk_to_stop) if risk_to_stop != 0 else 0
        }
