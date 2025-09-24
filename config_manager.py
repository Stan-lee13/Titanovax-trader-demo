"""
TitanovaX Trading System - Centralized Configuration Manager
Handles all environment variables, credentials, and configuration validation
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import hmac
from datetime import datetime, timedelta
import redis
import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    name: str = "titanovax"
    user: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_timeout: int = 30
    max_connections: int = 20
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.ssl_mode}"


@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 30
    connection_pool_timeout: int = 20
    
    @property
    def connection_url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class TradingConfig:
    """Trading system configuration"""
    default_lot_size: float = 0.1
    max_risk_percent: float = 2.0
    max_positions: int = 5
    stop_loss_pips: int = 50
    take_profit_pips: int = 100
    max_slippage_pips: int = 2
    max_daily_trades: int = 50
    max_concurrent_positions: int = 5
    min_account_balance: float = 1000.0
    trading_start_hour: int = 0
    trading_end_hour: int = 24
    weekend_trading: bool = False


@dataclass
class OANDAConfig:
    """OANDA API configuration"""
    api_key: str = ""
    account_id: str = ""
    environment: str = "practice"  # practice or live
    base_url: str = "https://api-fxpractice.oanda.com"
    streaming_url: str = "https://stream-fxpractice.oanda.com"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if self.environment == "live":
            self.base_url = "https://api-fxtrade.oanda.com"
            self.streaming_url = "https://stream-fxtrade.oanda.com"


@dataclass
class BinanceConfig:
    """Binance API configuration"""
    api_key: str = ""
    secret_key: str = ""
    testnet: bool = True
    timeout: int = 30
    max_retries: int = 3
    base_url: str = "https://testnet.binance.vision"  # Default to testnet
    
    def __post_init__(self):
        if not self.testnet:
            self.base_url = "https://api.binance.com"


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    bot_token: str = ""
    chat_id: str = ""
    channel_id: str = ""
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 30


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    hmac_secret_key: str = ""
    rate_limit_per_minute: int = 1000
    rate_limit_per_hour: int = 10000
    rate_limit_per_day: int = 100000
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    ip_whitelist: list = field(default_factory=list)


@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_type: str = "xgboost"
    retrain_interval_hours: int = 24
    feature_window: int = 100
    prediction_horizon: int = 1
    model_update_interval_hours: int = 6
    model_validation_threshold: float = 0.6
    model_retrain_trigger: float = 0.5
    ensemble_models: list = field(default_factory=lambda: ["xgboost", "lightgbm", "transformer"])
    onnx_export_enabled: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    metrics_enabled: bool = True
    health_check_interval: int = 60
    prometheus_port: int = 9090
    grafana_port: int = 3001
    alert_drawdown_percent: float = 10.0
    alert_daily_loss_percent: float = 5.0
    alert_cpu_percent: float = 90.0
    alert_memory_percent: float = 90.0
    alert_disk_percent: float = 90.0
    log_level: str = "INFO"
    log_file_enabled: bool = True
    log_console_enabled: bool = True
    log_max_file_size_mb: int = 100
    log_backup_count: int = 5


@dataclass
class StorageConfig:
    """Storage configuration for embeddings and logs"""
    embeddings_model: str = "all-MiniLM-L6-v2"
    embeddings_dimension: int = 384
    faiss_index_type: str = "IndexIVFPQ"
    faiss_pq_m: int = 16
    faiss_pq_nbits: int = 8
    faiss_nlist: int = 1024
    use_disk_storage: bool = True
    parquet_compression: str = "zstd"
    retention_days_raw: int = 90
    retention_days_summary: int = 365
    deduplication_enabled: bool = True
    memory_map_enabled: bool = True


class ConfigManager:
    """Centralized configuration manager for TitanovaX Trading System"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.logger = self._setup_logging()
        self._load_environment_variables()
        self._validate_configuration()
        self._test_connections()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("ConfigManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_environment_variables(self):
        """Load all environment variables with validation"""
        try:
            from dotenv import load_dotenv
            load_dotenv(self.env_file)
        except ImportError:
            self.logger.warning("python-dotenv not installed, using system environment variables")
        
        # Database Configuration
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "titanovax"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            ssl_mode=os.getenv("DB_SSL_MODE", "prefer"),
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "20"))
        )
        
        # Redis Configuration
        self.redis = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            db=int(os.getenv("REDIS_DB", "0")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        )
        
        # Trading Configuration
        self.trading = TradingConfig(
            default_lot_size=float(os.getenv("DEFAULT_LOT_SIZE", "0.1")),
            max_risk_percent=float(os.getenv("MAX_RISK_PERCENT", "2.0")),
            max_positions=int(os.getenv("MAX_POSITIONS", "5")),
            stop_loss_pips=int(os.getenv("STOP_LOSS_PIPS", "50")),
            take_profit_pips=int(os.getenv("TAKE_PROFIT_PIPS", "100")),
            max_slippage_pips=int(os.getenv("MAX_SLIPPAGE_PIPS", "2")),
            max_daily_trades=int(os.getenv("MAX_DAILY_TRADES", "50")),
            max_concurrent_positions=int(os.getenv("MAX_CONCURRENT_POSITIONS", "5")),
            min_account_balance=float(os.getenv("MIN_ACCOUNT_BALANCE", "1000.0")),
            trading_start_hour=int(os.getenv("TRADING_START_HOUR", "0")),
            trading_end_hour=int(os.getenv("TRADING_END_HOUR", "24")),
            weekend_trading=os.getenv("WEEKEND_TRADING", "false").lower() == "true"
        )
        
        # OANDA Configuration
        self.oanda = OANDAConfig(
            api_key=os.getenv("OANDA_API_KEY", ""),
            account_id=os.getenv("OANDA_ACCOUNT_ID", ""),
            environment=os.getenv("OANDA_ENVIRONMENT", "practice"),
            timeout=int(os.getenv("OANDA_TIMEOUT", "30")),
            max_retries=int(os.getenv("OANDA_MAX_RETRIES", "3"))
        )
        
        # Binance Configuration
        self.binance = BinanceConfig(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            secret_key=os.getenv("BINANCE_SECRET_KEY", ""),
            testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
            timeout=int(os.getenv("BINANCE_TIMEOUT", "30")),
            max_retries=int(os.getenv("BINANCE_MAX_RETRIES", "3"))
        )
        
        # Telegram Configuration
        self.telegram = TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            channel_id=os.getenv("TELEGRAM_CHANNEL_ID", ""),
            enabled=os.getenv("TELEGRAM_ENABLED", "true").lower() == "true",
            timeout=int(os.getenv("TELEGRAM_TIMEOUT", "30")),
            rate_limit_per_minute=int(os.getenv("TELEGRAM_RATE_LIMIT", "30"))
        )
        
        # Security Configuration
        self.security = SecurityConfig(
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", ""),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            hmac_secret_key=os.getenv("HMAC_SECRET_KEY", ""),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "1000")),
            ssl_enabled=os.getenv("SSL_ENABLED", "false").lower() == "true",
            ssl_cert_path=os.getenv("SSL_CERT_PATH", ""),
            ssl_key_path=os.getenv("SSL_KEY_PATH", "")
        )
        
        # ML Configuration
        self.ml = MLConfig(
            model_type=os.getenv("ML_MODEL_TYPE", "xgboost"),
            retrain_interval_hours=int(os.getenv("RETRAIN_INTERVAL_HOURS", "24")),
            feature_window=int(os.getenv("FEATURE_WINDOW", "100")),
            prediction_horizon=int(os.getenv("PREDICTION_HORIZON", "1")),
            model_validation_threshold=float(os.getenv("MODEL_VALIDATION_THRESHOLD", "0.6"))
        )
        
        # Monitoring Configuration
        self.monitoring = MonitoringConfig(
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "60")),
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            grafana_port=int(os.getenv("GRAFANA_PORT", "3001")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_max_file_size_mb=int(os.getenv("LOG_MAX_FILE_SIZE_MB", "100")),
            log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
        
        # Storage Configuration
        self.storage = StorageConfig(
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
            embeddings_dimension=int(os.getenv("EMBEDDINGS_DIMENSION", "384")),
            faiss_index_type=os.getenv("FAISS_INDEX_TYPE", "IndexIVFPQ"),
            faiss_pq_m=int(os.getenv("FAISS_PQ_M", "16")),
            faiss_pq_nbits=int(os.getenv("FAISS_PQ_NBITS", "8")),
            faiss_nlist=int(os.getenv("FAISS_NLIST", "1024")),
            use_disk_storage=os.getenv("USE_DISK_STORAGE", "true").lower() == "true",
            retention_days_raw=int(os.getenv("RETENTION_DAYS_RAW", "90")),
            retention_days_summary=int(os.getenv("RETENTION_DAYS_SUMMARY", "365"))
        )
        
        self.logger.info("Configuration loaded successfully")
    
    def _validate_configuration(self):
        """Validate all configuration parameters"""
        self.logger.info("Validating configuration...")
        
        # Validate critical credentials
        required_credentials = {
            "OANDA_API_KEY": self.oanda.api_key,
            "OANDA_ACCOUNT_ID": self.oanda.account_id,
            "BINANCE_API_KEY": self.binance.api_key,
            "BINANCE_SECRET_KEY": self.binance.secret_key,
            "TELEGRAM_BOT_TOKEN": self.telegram.bot_token,
            "TELEGRAM_CHAT_ID": self.telegram.chat_id,
            "JWT_SECRET_KEY": self.security.jwt_secret_key,
            "DB_PASSWORD": self.database.password
        }
        
        missing_credentials = []
        for key, value in required_credentials.items():
            if not value:
                missing_credentials.append(key)
        
        if missing_credentials:
            self.logger.warning(f"Missing credentials: {missing_credentials}")
            if "OANDA_API_KEY" in missing_credentials or "OANDA_ACCOUNT_ID" in missing_credentials:
                raise ValueError("OANDA credentials are required for trading operations")
            if "TELEGRAM_BOT_TOKEN" in missing_credentials:
                self.logger.warning("Telegram integration will be disabled")
                self.telegram.enabled = False
        
        # Validate numeric ranges
        if self.trading.max_risk_percent <= 0 or self.trading.max_risk_percent > 100:
            raise ValueError("MAX_RISK_PERCENT must be between 0 and 100")
        
        if self.trading.max_daily_trades <= 0:
            raise ValueError("MAX_DAILY_TRADES must be positive")
        
        if self.monitoring.alert_cpu_percent <= 0 or self.monitoring.alert_cpu_percent > 100:
            raise ValueError("ALERT_CPU_PERCENT must be between 0 and 100")
        
        self.logger.info("Configuration validation completed")
    
    def _test_connections(self):
        """Test critical service connections"""
        self.logger.info("Testing service connections...")
        
        # Test Redis connection
        try:
            redis_client = redis.from_url(self.redis.connection_url)
            redis_client.ping()
            self.logger.info("Redis connection successful")
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise ConnectionError(f"Cannot connect to Redis: {e}")
        
        # Test PostgreSQL connection
        try:
            conn = psycopg2.connect(
                host=self.database.host,
                port=self.database.port,
                database=self.database.name,
                user=self.database.user,
                password=self.database.password,
                connect_timeout=self.database.connection_timeout
            )
            conn.close()
            self.logger.info("PostgreSQL connection successful")
        except Exception as e:
            self.logger.error(f"PostgreSQL connection failed: {e}")
            raise ConnectionError(f"Cannot connect to PostgreSQL: {e}")
        
        self.logger.info("All connection tests passed")
    
    def generate_hmac_signature(self, message: str, timestamp: str = None) -> str:
        """Generate HMAC-SHA256 signature for message validation"""
        if not self.security.hmac_secret_key:
            raise ValueError("HMAC secret key not configured")
        
        if timestamp is None:
            timestamp = str(int(datetime.now().timestamp()))
        
        message_to_sign = f"{timestamp}:{message}"
        signature = hmac.new(
            self.security.hmac_secret_key.encode(),
            message_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    def verify_hmac_signature(self, signature: str, message: str, max_age_seconds: int = 300) -> bool:
        """Verify HMAC-SHA256 signature"""
        try:
            timestamp_str, provided_signature = signature.split(":", 1)
            timestamp = int(timestamp_str)
            
            # Check timestamp age
            current_time = int(datetime.now().timestamp())
            if current_time - timestamp > max_age_seconds:
                return False
            
            # Generate expected signature
            expected_signature = self.generate_hmac_signature(message, timestamp_str)
            
            # Compare signatures using hmac.compare_digest for security
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception:
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring"""
        return {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "ssl_enabled": self.database.ssl_mode != "disable"
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "max_connections": self.redis.max_connections
            },
            "trading": {
                "max_risk_percent": self.trading.max_risk_percent,
                "max_daily_trades": self.trading.max_daily_trades,
                "max_positions": self.trading.max_positions,
                "lot_size": self.trading.default_lot_size
            },
            "oanda": {
                "environment": self.oanda.environment,
                "account_configured": bool(self.oanda.account_id)
            },
            "binance": {
                "testnet": self.binance.testnet,
                "api_configured": bool(self.binance.api_key)
            },
            "telegram": {
                "enabled": self.telegram.enabled,
                "rate_limit": self.telegram.rate_limit_per_minute
            },
            "security": {
                "ssl_enabled": self.security.ssl_enabled,
                "jwt_expiration_hours": self.security.jwt_expiration_hours,
                "rate_limit_per_minute": self.security.rate_limit_per_minute
            },
            "ml": {
                "model_type": self.ml.model_type,
                "retrain_interval_hours": self.ml.retrain_interval_hours,
                "ensemble_models": self.ml.ensemble_models
            },
            "monitoring": {
                "metrics_enabled": self.monitoring.metrics_enabled,
                "prometheus_port": self.monitoring.prometheus_port,
                "grafana_port": self.monitoring.grafana_port,
                "log_level": self.monitoring.log_level
            },
            "storage": {
                "embeddings_model": self.storage.embeddings_model,
                "embeddings_dimension": self.storage.embeddings_dimension,
                "faiss_index_type": self.storage.faiss_index_type,
                "use_disk_storage": self.storage.use_disk_storage,
                "retention_days_raw": self.storage.retention_days_raw
            }
        }
    
    def export_config(self, output_file: str = "config_export.json"):
        """Export current configuration to JSON file (without sensitive data)"""
        config_summary = self.get_config_summary()
        
        with open(output_file, 'w') as f:
            json.dump(config_summary, f, indent=2, default=str)
        
        self.logger.info(f"Configuration exported to {output_file}")


# Global configuration instance
config = None

def initialize_config(env_file: str = ".env") -> ConfigManager:
    """Initialize global configuration"""
    global config
    config = ConfigManager(env_file)
    return config

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global config
    if config is None:
        raise RuntimeError("Configuration not initialized. Call initialize_config() first.")
    return config


if __name__ == "__main__":
    # Test configuration manager
    try:
        cfg = initialize_config()
        print("Configuration initialized successfully!")
        print(f"OANDA Environment: {cfg.oanda.environment}")
        print(f"Telegram Enabled: {cfg.telegram.enabled}")
        print(f"Trading Risk Limit: {cfg.trading.max_risk_percent}%")
        
        # Export configuration summary
        cfg.export_config("current_config.json")
        
        # Test HMAC signature
        test_message = "test_trading_signal"
        signature = cfg.generate_hmac_signature(test_message)
        print(f"HMAC Signature: {signature}")
        
        is_valid = cfg.verify_hmac_signature(signature, test_message)
        print(f"Signature validation: {is_valid}")
        
    except Exception as e:
        print(f"Configuration error: {e}")
        exit(1)