"""
Centralized Configuration Management for TitanovaX Trading System
Handles all configuration loading, validation, and management
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import copy

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "titanovax"
    user: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_timeout: int = 30
    max_connections: int = 20
    pool_size: int = 5

@dataclass
class MT5Config:
    """MT5 broker configuration"""
    login: int = 0
    password: str = ""
    server: str = ""
    path: str = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: int = 5
    symbols: list = None
    default_lot_size: float = 0.1
    max_risk_percent: float = 2.0
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]

@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_type: str = "xgboost"
    feature_window: int = 100
    prediction_horizon: int = 1
    training_data_days: int = 365
    retrain_interval_hours: int = 24
    
    # XGBoost specific
    xgboost_params: Dict[str, Any] = None
    
    # Transformer specific
    transformer_params: Dict[str, Any] = None
    
    # Feature engineering
    feature_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.xgboost_params is None:
            self.xgboost_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
        
        if self.transformer_params is None:
            self.transformer_params = {
                "d_model": 128,
                "nhead": 8,
                "num_layers": 6,
                "dropout": 0.1,
                "max_seq_length": 500
            }
        
        if self.feature_config is None:
            self.feature_config = {
                "use_technical_indicators": True,
                "use_market_microstructure": True,
                "use_sentiment": False,
                "use_fundamental": False,
                "technical_periods": [5, 10, 20, 50, 100, 200],
                "microstructure_window": 20
            }

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_drawdown_percent: float = 10.0
    max_daily_loss_percent: float = 5.0
    max_positions: int = 5
    max_correlation: float = 0.8
    stop_loss_pips: int = 50
    take_profit_pips: int = 100
    trailing_stop_enabled: bool = True
    trailing_stop_distance: float = 0.02
    position_sizing_method: str = "fixed_risk"  # fixed_risk, kelly, volatility_targeting
    
    # Kelly criterion parameters
    kelly_fraction: float = 0.25
    
    # Volatility targeting
    volatility_target: float = 0.15
    volatility_lookback: int = 20

@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    enabled: bool = True
    bot_token: str = ""
    chat_id: str = ""
    admin_ids: list = None
    alert_on_trade: bool = True
    alert_on_error: bool = True
    alert_on_high_drawdown: bool = True
    
    def __post_init__(self):
        if self.admin_ids is None:
            self.admin_ids = []

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size_mb: int = 100
    backup_count: int = 5
    log_directory: str = "logs"
    
    # Component-specific log levels
    component_levels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.component_levels is None:
            self.component_levels = {
                "mt5_trader": "INFO",
                "ml_trainer": "INFO",
                "risk_manager": "INFO",
                "telegram_bot": "INFO"
            }

@dataclass
class ServerConfig:
    """Server/API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    cors_origins: list = None
    api_key: str = ""
    rate_limit: int = 1000  # requests per minute
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "https://localhost:3000"]

@dataclass
class SystemConfig:
    """System-wide configuration"""
    environment: str = "development"  # development, staging, production
    timezone: str = "UTC"
    data_directory: str = "data"
    models_directory: str = "models"
    cache_directory: str = "cache"
    temp_directory: str = "temp"
    
    # Performance settings
    max_workers: int = 4
    memory_limit_gb: float = 4.0
    cpu_limit_percent: float = 80.0
    
    # Monitoring
    metrics_enabled: bool = True
    health_check_interval: int = 60
    
    # Feature flags
    features: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = {
                "ml_pipeline": True,
                "telegram_integration": True,
                "web_dashboard": True,
                "paper_trading": True,
                "live_trading": False,
                "backtesting": True,
                "risk_management": True
            }

class ConfigManager:
    """Centralized configuration manager"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.logger = self._setup_logging()
        self.config_dir = Path("config")
        self.config_file = self.config_dir / "config.yaml"
        self.secrets_file = self.config_dir / "secrets.yaml"
        
        # Configuration sections
        self.database = DatabaseConfig()
        self.mt5 = MT5Config()
        self.ml = MLConfig()
        self.risk = RiskConfig()
        self.telegram = TelegramConfig()
        self.logging = LoggingConfig()
        self.server = ServerConfig()
        self.system = SystemConfig()
        
        # Load configuration
        self.load_config()
        
        # Watch for changes
        self._watch_config = False
        self._config_watcher = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup configuration manager logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # File handler
        handler = logging.FileHandler(f'logs/config_manager_{datetime.now().strftime("%Y%m%d")}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def load_config(self) -> bool:
        """Load configuration from files"""
        try:
            self.logger.info("Loading configuration...")
            
            # Load main config
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self._apply_config(config_data)
            
            # Load secrets (override main config)
            if self.secrets_file.exists():
                with open(self.secrets_file, 'r') as f:
                    secrets_data = yaml.safe_load(f)
                    self._apply_secrets(secrets_data)
            
            # Load environment variables (highest priority)
            self._load_env_vars()
            
            self.logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            self.logger.info("Saving configuration...")
            
            # Ensure config directory exists
            self.config_dir.mkdir(exist_ok=True)
            
            # Prepare configuration data
            config_data = {
                'database': asdict(self.database),
                'mt5': asdict(self.mt5),
                'ml': asdict(self.ml),
                'risk': asdict(self.risk),
                'telegram': asdict(self.telegram),
                'logging': asdict(self.logging),
                'server': asdict(self.server),
                'system': asdict(self.system)
            }
            
            # Remove sensitive data before saving
            self._remove_sensitive_data(config_data)
            
            # Save to file
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def create_default_config(self) -> bool:
        """Create default configuration files"""
        try:
            self.logger.info("Creating default configuration...")
            
            # Ensure config directory exists
            self.config_dir.mkdir(exist_ok=True)
            
            # Create main config
            self.save_config()
            
            # Create secrets template
            secrets_template = {
                'database': {
                    'password': 'your_database_password'
                },
                'mt5': {
                    'login': 12345,
                    'password': 'your_mt5_password',
                    'server': 'YourBroker-Server'
                },
                'telegram': {
                    'bot_token': 'your_telegram_bot_token',
                    'chat_id': 'your_chat_id'
                },
                'server': {
                    'api_key': 'your_api_key'
                }
            }
            
            with open(self.secrets_file, 'w') as f:
                yaml.dump(secrets_template, f, default_flow_style=False, indent=2)
            
            self.logger.info("Default configuration created")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create default configuration: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        errors = []
        warnings = []
        
        # Validate MT5 configuration
        if self.mt5.login == 0:
            errors.append("MT5 login is not configured")
        
        if not self.mt5.password:
            errors.append("MT5 password is not configured")
        
        if not self.mt5.server:
            errors.append("MT5 server is not configured")
        
        # Validate database configuration
        if not self.database.password:
            errors.append("Database password is not configured")
        
        # Validate Telegram configuration
        if self.telegram.enabled and not self.telegram.bot_token:
            warnings.append("Telegram is enabled but bot token is not configured")
        
        # Validate risk configuration
        if self.risk.max_risk_percent > 10.0:
            warnings.append("Maximum risk percentage is very high (>10%)")
        
        # Validate ML configuration
        if self.ml.model_type not in ["xgboost", "transformer", "ensemble"]:
            errors.append(f"Invalid ML model type: {self.ml.model_type}")
        
        # Validate system configuration
        if self.system.environment not in ["development", "staging", "production"]:
            errors.append(f"Invalid environment: {self.system.environment}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_errors': len(errors),
            'total_warnings': len(warnings)
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'environment': self.system.environment,
            'ml_model_type': self.ml.model_type,
            'mt5_symbols': len(self.mt5.symbols),
            'telegram_enabled': self.telegram.enabled,
            'risk_management_enabled': self.system.features.get('risk_management', False),
            'live_trading_enabled': self.system.features.get('live_trading', False),
            'paper_trading_enabled': self.system.features.get('paper_trading', True),
            'validation': self.validate_config()
        }
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update a specific configuration value"""
        try:
            config_section = getattr(self, section, None)
            if config_section is None:
                self.logger.error(f"Configuration section '{section}' not found")
                return False
            
            if hasattr(config_section, key):
                setattr(config_section, key, value)
                self.logger.info(f"Updated {section}.{key} = {value}")
                return True
            else:
                self.logger.error(f"Configuration key '{key}' not found in section '{section}'")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def get_env_var(self, var_name: str, default: Any = None) -> Any:
        """Get environment variable with type conversion"""
        value = os.getenv(var_name, default)
        
        if value is None:
            return default
        
        # Type conversion
        if isinstance(default, bool):
            return str(value).lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default, int):
            try:
                return int(value)
            except ValueError:
                return default
        elif isinstance(default, float):
            try:
                return float(value)
            except ValueError:
                return default
        elif isinstance(default, list):
            if isinstance(value, str):
                return [item.strip() for item in value.split(',')]
        
        return value
    
    def _apply_config(self, config_data: Dict[str, Any]):
        """Apply configuration data to sections"""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _apply_secrets(self, secrets_data: Dict[str, Any]):
        """Apply secrets configuration"""
        for section_name, section_data in secrets_data.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _load_env_vars(self):
        """Load environment variables"""
        # Database
        self.database.password = self.get_env_var('DB_PASSWORD', self.database.password)
        self.database.host = self.get_env_var('DB_HOST', self.database.host)
        self.database.port = self.get_env_var('DB_PORT', self.database.port)
        self.database.name = self.get_env_var('DB_NAME', self.database.name)
        self.database.user = self.get_env_var('DB_USER', self.database.user)
        
        # MT5
        self.mt5.login = self.get_env_var('MT5_LOGIN', self.mt5.login)
        self.mt5.password = self.get_env_var('MT5_PASSWORD', self.mt5.password)
        self.mt5.server = self.get_env_var('MT5_SERVER', self.mt5.server)
        
        # Telegram
        self.telegram.bot_token = self.get_env_var('TELEGRAM_BOT_TOKEN', self.telegram.bot_token)
        self.telegram.chat_id = self.get_env_var('TELEGRAM_CHAT_ID', self.telegram.chat_id)
        
        # Server
        self.server.api_key = self.get_env_var('API_KEY', self.server.api_key)
        self.server.host = self.get_env_var('SERVER_HOST', self.server.host)
        self.server.port = self.get_env_var('SERVER_PORT', self.server.port)
        
        # System
        self.system.environment = self.get_env_var('ENVIRONMENT', self.system.environment)
    
    def _remove_sensitive_data(self, config_data: Dict[str, Any]):
        """Remove sensitive data from configuration"""
        # Database password
        if 'database' in config_data and 'password' in config_data['database']:
            config_data['database']['password'] = "***REDACTED***"
        
        # MT5 credentials
        if 'mt5' in config_data:
            if 'password' in config_data['mt5']:
                config_data['mt5']['password'] = "***REDACTED***"
            if 'login' in config_data['mt5']:
                config_data['mt5']['login'] = 0
        
        # Telegram token
        if 'telegram' in config_data and 'bot_token' in config_data['telegram']:
            config_data['telegram']['bot_token'] = "***REDACTED***"
        
        # API key
        if 'server' in config_data and 'api_key' in config_data['server']:
            config_data['server']['api_key'] = "***REDACTED***"
    
    def start_config_watcher(self):
        """Start watching configuration files for changes"""
        if self._watch_config:
            return
        
        self._watch_config = True
        self._config_watcher = threading.Thread(target=self._watch_files, daemon=True)
        self._config_watcher.start()
        self.logger.info("Configuration file watcher started")
    
    def stop_config_watcher(self):
        """Stop watching configuration files"""
        self._watch_config = False
        if self._config_watcher:
            self._config_watcher.join(timeout=1)
        self.logger.info("Configuration file watcher stopped")
    
    def _watch_files(self):
        """Watch configuration files for changes"""
        last_modified = {}
        
        files_to_watch = [self.config_file, self.secrets_file]
        
        while self._watch_config:
            try:
                for file_path in files_to_watch:
                    if file_path.exists():
                        current_modified = file_path.stat().st_mtime
                        
                        if file_path not in last_modified:
                            last_modified[file_path] = current_modified
                        elif current_modified != last_modified[file_path]:
                            self.logger.info(f"Configuration file changed: {file_path}")
                            self.load_config()
                            last_modified[file_path] = current_modified
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error watching configuration files: {e}")
                time.sleep(10)  # Wait longer on error

def main():
    """Example usage"""
    config_manager = ConfigManager()
    
    # Create default config if it doesn't exist
    if not config_manager.config_file.exists():
        config_manager.create_default_config()
    
    # Load configuration
    if config_manager.load_config():
        print("✅ Configuration loaded successfully")
        
        # Validate configuration
        validation = config_manager.validate_config()
        if validation['valid']:
            print("✅ Configuration is valid")
        else:
            print(f"❌ Configuration validation failed:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        # Get configuration summary
        summary = config_manager.get_config_summary()
        print(f"Configuration Summary:")
        print(f"  Environment: {summary['environment']}")
        print(f"  ML Model: {summary['ml_model_type']}")
        print(f"  Telegram: {'Enabled' if summary['telegram_enabled'] else 'Disabled'}")
        print(f"  Live Trading: {'Enabled' if summary['live_trading_enabled'] else 'Disabled'}")
        
        # Start configuration watcher
        config_manager.start_config_watcher()
        
        try:
            # Keep running to test config watcher
            import time
            time.sleep(30)
        except KeyboardInterrupt:
            pass
        finally:
            config_manager.stop_config_watcher()
    else:
        print("❌ Failed to load configuration")

if __name__ == "__main__":
    main()