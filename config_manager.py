"""
Configuration Manager
Centralized configuration management for the trading bot
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage all bot configurations"""
    
    DEFAULT_CONFIG = {
        # Bot Settings
        "bot": {
            "name": "Indian Stock Trading Bot",
            "version": "1.0.0",
            "trading_active": False,
            "scan_interval": 60,  # seconds
            "log_level": "INFO"
        },
        
        # Broker Configuration
        "broker": {
            "name": "zerodha",  # zerodha, angel_broking, 5paisa
            "api_key": "",
            "api_secret": "",
            "user_id": "",
            "password": "",
            "base_url": "https://api.kite.trade"
        },
        
        # Market Settings
        "market": {
            "exchange": "NSE",  # NSE, BSE
            "market_open": "09:15",
            "market_close": "15:30",
            "timezone": "Asia/Kolkata",
            "data_source": "yfinance"  # yfinance, kite, angel
        },
        
        # Trading Rules
        "trading": {
            "max_daily_loss": 5000.0,  # Max daily loss in rupees
            "max_loss_per_trade": 0.02,  # 2% per trade
            "max_position_size": 0.1,  # 10% of portfolio
            "min_trade_capital": 500.0,
            "max_consecutive_losses": 5,
            "min_rr_ratio": 1.0,
            "enable_scalping": True,
            "scalp_hold_time_max": 300,  # 5 minutes in seconds
        },
        
        # Risk Management
        "risk": {
            "max_open_positions": 3,
            "max_portfolio_heat": 0.05,  # 5% of portfolio at risk
            "use_trailing_stops": False,
            "trailing_stop_percent": 2.0,
            "enable_breakeven": True,
            "breakeven_trigger_percent": 1.5
        },
        
        # AI Configuration
        "ai": {
            "enabled": True,
            "provider": "openai",  # openai, claude
            "model": "gpt-4-vision-preview",
            "api_key": "",
            "ai_confidence_threshold": 60,
            "patterns_to_detect": [
                "breakout", "pullback", "double_top", "double_bottom",
                "triangle", "wedge", "head_shoulders", "flag"
            ]
        },
        
        # Technical Indicators
        "indicators": {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "sma_fast": 20,
            "sma_slow": 50,
            "ema_fast": 12,
            "ema_slow": 26,
            "macd_signal": 9
        },
        
        # Watchlist
        "watchlist": [
            "TCS", "INFY", "WIPRO", "HCL", "RELIANCE",
            "HDFC", "ICICIBANK", "SBIN", "BAJAJFINSV",
            "MARUTI", "SUNPHARMA", "ASIANPAINT", "AXISBANK"
        ],
        
        # Portfolio
        "portfolio": {
            "initial_capital": 100000.0,
            "leverage": 1,
            "commission_percent": 0.03  # Broker commission
        },
        
        # Logging
        "logging": {
            "log_file": "trading_bot.log",
            "log_level": "INFO",
            "max_file_size": 10485760,  # 10MB
            "backup_count": 5
        },
        
        # Notifications
        "notifications": {
            "telegram_enabled": False,
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "email_enabled": False,
            "email_from": "",
            "email_to": [],
            "notify_on_trade": True,
            "notify_on_signal": True,
            "notify_on_error": True
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    self._deep_merge(self.config, user_config)
                    logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Config file not found. Creating default at {self.config_file}")
            self.save()
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value by dot notation (e.g., 'broker.api_key')"""
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Config key not found: {key_path}")
            return default
    
    def set(self, key_path: str, value: Any):
        """Set config value by dot notation"""
        try:
            keys = key_path.split('.')
            config = self.config
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            config[keys[-1]] = value
            logger.info(f"Config updated: {key_path} = {value}")
        except Exception as e:
            logger.error(f"Error setting config: {e}")
    
    def get_broker_config(self) -> Dict:
        """Get broker configuration"""
        return self.config.get('broker', {})
    
    def get_trading_rules(self) -> Dict:
        """Get trading rules"""
        return self.config.get('trading', {})
    
    def get_risk_config(self) -> Dict:
        """Get risk management config"""
        return self.config.get('risk', {})
    
    def get_ai_config(self) -> Dict:
        """Get AI configuration"""
        return self.config.get('ai', {})
    
    def get_watchlist(self) -> list:
        """Get watchlist"""
        return self.config.get('watchlist', [])
    
    def add_to_watchlist(self, symbol: str):
        """Add symbol to watchlist"""
        watchlist = self.config.get('watchlist', [])
        if symbol not in watchlist:
            watchlist.append(symbol)
            self.config['watchlist'] = watchlist
            self.save()
    
    def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        watchlist = self.config.get('watchlist', [])
        if symbol in watchlist:
            watchlist.remove(symbol)
            self.config['watchlist'] = watchlist
            self.save()
    
    def validate(self) -> tuple[bool, list]:
        """
        Validate configuration
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required broker settings
        broker = self.config.get('broker', {})
        if not broker.get('api_key'):
            errors.append("Broker API key not configured")
        
        # Check AI settings if enabled
        ai = self.config.get('ai', {})
        if ai.get('enabled') and not ai.get('api_key'):
            errors.append("AI provider API key not configured")
        
        # Check trading rules
        trading = self.config.get('trading', {})
        if trading.get('max_daily_loss', 0) <= 0:
            errors.append("Max daily loss must be positive")
        
        if trading.get('min_rr_ratio', 0) < 0.5:
            errors.append("Min R:R ratio too low")
        
        # Check watchlist
        if not self.config.get('watchlist'):
            errors.append("Watchlist is empty")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_all(self) -> Dict:
        """Get all configuration"""
        return self.config.copy()
    
    def reset_to_defaults(self):
        """Reset to default configuration"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()
        logger.info("Configuration reset to defaults")
    
    def _deep_merge(self, target: Dict, source: Dict):
        """Deep merge source into target"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def export_safe_config(self) -> Dict:
        """Export config with sensitive data masked"""
        import copy
        safe_config = copy.deepcopy(self.config)
        
        # Mask sensitive data
        if 'broker' in safe_config:
            safe_config['broker']['api_key'] = '***MASKED***'
            safe_config['broker']['api_secret'] = '***MASKED***'
            safe_config['broker']['password'] = '***MASKED***'
        
        if 'ai' in safe_config:
            safe_config['ai']['api_key'] = '***MASKED***'
        
        if 'notifications' in safe_config:
            safe_config['notifications']['telegram_bot_token'] = '***MASKED***'
            safe_config['notifications']['telegram_chat_id'] = '***MASKED***'
        
        return safe_config
