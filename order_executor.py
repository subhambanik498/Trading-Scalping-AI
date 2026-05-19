"""
Order Execution Module
Handles buy/sell logic, order placement, and order management for Indian stocks
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Buy or Sell"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    PLACED = "PLACED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a single order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float
    stop_price: Optional[float] = None
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def get_total_value(self) -> float:
        """Get total order value at execution price"""
        return self.quantity * self.price
    
    def get_filled_value(self) -> float:
        """Get value of filled portion"""
        return self.filled_quantity * self.average_fill_price
    
    def is_complete(self) -> bool:
        """Check if order is fully filled"""
        return self.filled_quantity == self.quantity
    
    def get_remaining_quantity(self) -> int:
        """Get remaining unfilled quantity"""
        return self.quantity - self.filled_quantity


@dataclass
class Position:
    """Represents an open position in a stock"""
    symbol: str
    side: OrderSide
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    
    def get_current_pnl(self) -> float:
        """Get current unrealized P&L"""
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) * self.quantity
        else:  # SELL
            return (self.entry_price - self.current_price) * self.quantity
    
    def get_pnl_percent(self) -> float:
        """Get current unrealized P&L percentage"""
        if self.entry_price == 0:
            return 0
        pnl = self.get_current_pnl()
        return (pnl / (self.entry_price * self.quantity)) * 100
    
    def is_stop_hit(self) -> bool:
        """Check if stop loss is hit"""
        if self.side == OrderSide.BUY:
            return self.current_price <= self.stop_loss
        else:  # SELL
            return self.current_price >= self.stop_loss
    
    def is_target_hit(self) -> bool:
        """Check if take profit target is hit"""
        if self.side == OrderSide.BUY:
            return self.current_price >= self.take_profit
        else:  # SELL
            return self.current_price <= self.take_profit


class OrderExecutor:
    """
    Executes buy/sell orders on Indian stock exchanges (NSE/BSE)
    Manages order placement, tracking, and execution
    """
    
    def __init__(self, broker_api, config):
        """
        Initialize order executor
        
        Args:
            broker_api: API connector for the broker (e.g., Zerodha, Angel Broking)
            config: Configuration with trading parameters
        """
        self.broker_api = broker_api
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_counter = 0
        self.trades_executed = []
        
    def place_buy_order(self, symbol: str, quantity: int, price: float,
                       order_type: OrderType = OrderType.MARKET,
                       stop_price: Optional[float] = None) -> Optional[Order]:
        """
        Place a buy order for an Indian stock
        
        Args:
            symbol: Stock symbol (e.g., 'TCS', 'INFY')
            quantity: Number of shares to buy
            price: Limit price (or current price for market orders)
            order_type: Type of order (MARKET, LIMIT, STOP_LOSS, STOP_LIMIT)
            stop_price: Stop price for stop orders
            
        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Validate inputs
            if quantity <= 0:
                logger.error(f"Invalid quantity: {quantity}")
                return None
            
            if price <= 0:
                logger.error(f"Invalid price: {price}")
                return None
            
            # Create order object
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )
            
            # Place order via broker API
            result = self.broker_api.place_order(
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                price=price,
                order_type=order_type.value,
                stop_price=stop_price
            )
            
            if result and result.get('status') == 'success':
                order.status = OrderStatus.PLACED
                self.orders[order_id] = order
                logger.info(f"Buy order placed: {symbol} x{quantity} @ ₹{price}")
                return order
            else:
                logger.error(f"Failed to place buy order for {symbol}: {result}")
                order.status = OrderStatus.REJECTED
                return None
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return None
    
    def place_sell_order(self, symbol: str, quantity: int, price: float,
                        order_type: OrderType = OrderType.MARKET,
                        stop_price: Optional[float] = None) -> Optional[Order]:
        """
        Place a sell order for an Indian stock
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            price: Limit price
            order_type: Type of order
            stop_price: Stop price for stop orders
            
        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Check if position exists and has sufficient quantity
            position = self.positions.get(symbol)
            if position and position.side == OrderSide.BUY:
                if position.quantity < quantity:
                    logger.error(f"Insufficient position for {symbol}. Have: {position.quantity}, Want: {quantity}")
                    return None
            
            # Create order object
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )
            
            # Place order via broker API
            result = self.broker_api.place_order(
                symbol=symbol,
                side='SELL',
                quantity=quantity,
                price=price,
                order_type=order_type.value,
                stop_price=stop_price
            )
            
            if result and result.get('status') == 'success':
                order.status = OrderStatus.PLACED
                self.orders[order_id] = order
                logger.info(f"Sell order placed: {symbol} x{quantity} @ ₹{price}")
                return order
            else:
                logger.error(f"Failed to place sell order for {symbol}: {result}")
                order.status = OrderStatus.REJECTED
                return None
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return None
    
    def place_bracket_order(self, symbol: str, quantity: int, entry_price: float,
                           stop_loss: float, take_profit: float) -> Optional[Dict]:
        """
        Place a bracket order (Entry + Stop Loss + Take Profit)
        Common strategy for Indian scalping
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Dictionary with all three orders, or None if failed
        """
        try:
            # Place main entry order
            entry_order = self.place_buy_order(
                symbol=symbol,
                quantity=quantity,
                price=entry_price,
                order_type=OrderType.LIMIT
            )
            
            if not entry_order:
                return None
            
            # Place stop loss order (will execute if SL is hit)
            sl_order = self.place_sell_order(
                symbol=symbol,
                quantity=quantity,
                price=stop_loss,
                order_type=OrderType.STOP_LOSS,
                stop_price=stop_loss
            )
            
            # Place take profit order (will execute if TP is hit)
            tp_order = self.place_sell_order(
                symbol=symbol,
                quantity=quantity,
                price=take_profit,
                order_type=OrderType.LIMIT
            )
            
            if entry_order and sl_order and tp_order:
                logger.info(f"Bracket order placed for {symbol}: Entry ₹{entry_price}, SL ₹{stop_loss}, TP ₹{take_profit}")
                return {
                    'entry': entry_order,
                    'stop_loss': sl_order,
                    'take_profit': tp_order
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing bracket order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            order = self.orders.get(order_id)
            if not order:
                logger.error(f"Order not found: {order_id}")
                return False
            
            if order.status in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
                logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
                return False
            
            # Cancel via broker API
            result = self.broker_api.cancel_order(order_id)
            
            if result and result.get('status') == 'success':
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def update_order_status(self, order_id: str, filled_quantity: int, 
                           average_price: float) -> bool:
        """Update order execution status"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return False
            
            order.filled_quantity = filled_quantity
            order.average_fill_price = average_price
            
            if filled_quantity == 0:
                order.status = OrderStatus.PENDING
            elif filled_quantity < order.quantity:
                order.status = OrderStatus.PARTIALLY_FILLED
            else:
                order.status = OrderStatus.FILLED
                # Create/update position
                self._update_position(order)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False
    
    def _update_position(self, order: Order):
        """Update position after order execution"""
        try:
            existing_pos = self.positions.get(order.symbol)
            
            if order.side == OrderSide.BUY:
                if existing_pos:
                    # Average up
                    total_value = (existing_pos.quantity * existing_pos.entry_price + 
                                 order.filled_quantity * order.average_fill_price)
                    total_quantity = existing_pos.quantity + order.filled_quantity
                    avg_price = total_value / total_quantity
                    
                    existing_pos.quantity = total_quantity
                    existing_pos.entry_price = avg_price
                else:
                    # New long position
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        side=OrderSide.BUY,
                        quantity=order.filled_quantity,
                        entry_price=order.average_fill_price,
                        entry_time=datetime.now(),
                        stop_loss=0,
                        take_profit=0
                    )
            
            elif order.side == OrderSide.SELL:
                if existing_pos and existing_pos.side == OrderSide.BUY:
                    # Close/reduce long position
                    existing_pos.quantity -= order.filled_quantity
                    
                    if existing_pos.quantity <= 0:
                        del self.positions[order.symbol]
                        self._record_trade(order, existing_pos)
            
            logger.info(f"Position updated for {order.symbol}")
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def _record_trade(self, exit_order: Order, position: Position):
        """Record completed trade to history"""
        try:
            pnl = position.get_current_pnl()
            pnl_percent = position.get_pnl_percent()
            
            trade_record = {
                'symbol': position.symbol,
                'entry_time': position.entry_time,
                'exit_time': exit_order.timestamp,
                'entry_price': position.entry_price,
                'exit_price': exit_order.average_fill_price,
                'quantity': position.quantity,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'duration': (exit_order.timestamp - position.entry_time).total_seconds()
            }
            
            self.trades_executed.append(trade_record)
            logger.info(f"Trade completed: {trade_record['symbol']} P&L: ₹{pnl:.2f} ({pnl_percent:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.order_counter}"
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [o for o in self.orders.values() 
                if o.status in [OrderStatus.PENDING, OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED]]
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all open positions"""
        return self.positions
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get specific position"""
        return self.positions.get(symbol)
    
    def close_position(self, symbol: str, current_price: float) -> Optional[Order]:
        """Close an open position at current market price"""
        position = self.positions.get(symbol)
        if not position:
            logger.warning(f"No position to close for {symbol}")
            return None
        
        return self.place_sell_order(
            symbol=symbol,
            quantity=position.quantity,
            price=current_price,
            order_type=OrderType.MARKET
        )
    
    def get_trade_summary(self) -> Dict:
        """Get summary of completed trades"""
        if not self.trades_executed:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': 0
            }
        
        total = len(self.trades_executed)
        wins = sum(1 for t in self.trades_executed if t['pnl'] > 0)
        losses = total - wins
        total_pnl = sum(t['pnl'] for t in self.trades_executed)
        
        return {
            'total_trades': total,
            'winning_trades': wins,
            'losing_trades': losses,
            'total_pnl': total_pnl,
            'win_rate': (wins / total * 100) if total > 0 else 0
        }
