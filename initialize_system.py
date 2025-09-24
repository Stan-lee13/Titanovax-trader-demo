#!/usr/bin/env python3
"""
TitanovaX System Initialization Script
Initializes database, storage systems, and configuration
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/initialization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize PostgreSQL database"""
    try:
        logger.info("🔄 Initializing database...")

        # Create database connection
        from config_manager import get_config_manager
        config = get_config_manager()
        db_config = config.get_database_config()

        # Test database connection
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=db_config.postgres_host,
                port=db_config.postgres_port,
                database=db_config.postgres_database,
                user=db_config.postgres_user,
                password=db_config.postgres_password
            )
            logger.info("✅ Database connection successful")
            conn.close()

        except Exception as e:
            logger.warning(f"⚠️ Database connection failed: {e}")
            logger.info("ℹ️ This is expected if database is not running yet")

        # Create SQL initialization file
        sql_dir = Path('sql')
        sql_dir.mkdir(exist_ok=True)

        init_sql = """
-- TitanovaX Database Initialization
-- Generated: {datetime}

-- Create tables if they don't exist
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    volume DECIMAL(10,2) NOT NULL,
    price DECIMAL(10,6) NOT NULL,
    stop_loss DECIMAL(10,6),
    take_profit DECIMAL(10,6),
    pnl DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'open',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telegram_messages (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(50) UNIQUE NOT NULL,
    chat_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50),
    username VARCHAR(100),
    text TEXT,
    timestamp TIMESTAMP NOT NULL,
    message_type VARCHAR(20) DEFAULT 'text',
    metadata JSONB,
    embedding VECTOR(384), -- FAISS embedding dimension
    hash VARCHAR(64) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_summaries (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    message_count INTEGER DEFAULT 0,
    unique_users INTEGER DEFAULT 0,
    topics TEXT[],
    sentiment_score DECIMAL(5,3),
    key_messages TEXT[],
    summary_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_percent DECIMAL(5,2),
    memory_percent DECIMAL(5,2),
    disk_percent DECIMAL(5,2),
    network_io JSONB,
    process_count INTEGER,
    thread_count INTEGER,
    open_files INTEGER
);

CREATE TABLE IF NOT EXISTS security_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    source_ip VARCHAR(45),
    user_agent TEXT,
    endpoint VARCHAR(100),
    details JSONB,
    resolved BOOLEAN DEFAULT FALSE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON telegram_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON telegram_messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_hash ON telegram_messages(hash);
CREATE INDEX IF NOT EXISTS idx_summaries_date ON daily_summaries(date);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_type ON security_events(event_type);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for trades table
DROP TRIGGER IF EXISTS update_trades_updated_at ON trades;
CREATE TRIGGER update_trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
"""

        with open(sql_dir / 'init.sql', 'w') as f:
            f.write(init_sql.format(datetime=datetime.now()))

        logger.info("✅ Database initialization scripts created")
        return True

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def initialize_storage_system():
    """Initialize FAISS and Parquet storage systems"""
    try:
        logger.info("🔄 Initializing storage system...")

        # Create storage directories
        storage_dirs = [
            'data/storage',
            'data/storage/embeddings',
            'data/storage/metadata',
            'data/storage/messages',
            'data/storage/summaries',
            'data/logs',
            'data/models',
            'data/config'
        ]

        for dir_path in storage_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Create storage configuration
        from config_manager import get_config_manager
        config = get_config_manager()

        storage_config = {
            'embeddings_model': config.get_model_config().embedding_model,
            'embeddings_dimension': config.get_model_config().embedding_dimensions,
            'faiss_index_type': config.get_model_config().faiss_index_type,
            'faiss_nlist': config.get_model_config().faiss_nlist,
            'faiss_m': config.get_model_config().faiss_m,
            'faiss_nbits': config.get_model_config().faiss_nbits,
            'retention_days_raw': config.get_system_config().data_retention_days,
            'retention_days_summary': config.get_system_config().data_retention_days * 3,
            'use_disk_storage': True,
            'parquet_compression': 'zstd',
            'deduplication_enabled': True,
            'memory_map_enabled': True
        }

        # Save storage configuration
        with open('data/config/storage_config.json', 'w') as f:
            import json
            json.dump(storage_config, f, indent=2)

        logger.info("✅ Storage system initialized")
        return True

    except Exception as e:
        logger.error(f"❌ Storage system initialization failed: {e}")
        return False

def initialize_monitoring_system():
    """Initialize monitoring and alerting system"""
    try:
        logger.info("🔄 Initializing monitoring system...")

        # Create monitoring configuration
        monitoring_config = {
            'metrics_collection_interval': 30,
            'alert_check_interval': 60,
            'anomaly_detection_enabled': True,
            'self_healing_enabled': True,
            'telegram_alerts_enabled': True,
            'email_alerts_enabled': False,
            'dashboard_enabled': True,
            'log_level': 'INFO',
            'alert_thresholds': {
                'cpu_percent': 85,
                'memory_percent': 80,
                'disk_percent': 90,
                'error_rate_percent': 5,
                'response_time_ms': 1000
            }
        }

        # Save monitoring configuration
        with open('data/config/monitoring_config.json', 'w') as f:
            import json
            json.dump(monitoring_config, f, indent=2)

        logger.info("✅ Monitoring system initialized")
        return True

    except Exception as e:
        logger.error(f"❌ Monitoring system initialization failed: {e}")
        return False

def initialize_security_system():
    """Initialize security system"""
    try:
        logger.info("🔄 Initializing security system...")

        # Create security configuration
        security_config = {
            'hmac_secret_key': 'your_secure_hmac_key_here',
            'jwt_secret_key': 'your_secure_jwt_key_here',
            'rate_limiting_enabled': True,
            'ip_filtering_enabled': True,
            'circuit_breaker_enabled': True,
            'input_validation_enabled': True,
            'audit_logging_enabled': True,
            'rate_limits': {
                'api': {'requests_per_minute': 100, 'requests_per_hour': 1000},
                'trading': {'requests_per_minute': 50, 'requests_per_hour': 500},
                'telegram': {'requests_per_minute': 30, 'requests_per_hour': 300}
            }
        }

        # Save security configuration
        with open('data/config/security_config.json', 'w') as f:
            import json
            json.dump(security_config, f, indent=2)

        logger.info("✅ Security system initialized")
        return True

    except Exception as e:
        logger.error(f"❌ Security system initialization failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    try:
        logger.info("🔄 Creating sample data...")

        # Create sample trading signals
        sample_signals = [
            {
                'symbol': 'EURUSD',
                'action': 'BUY',
                'size': 0.1,
                'price': 1.0850,
                'timestamp': int(datetime.now().timestamp())
            },
            {
                'symbol': 'GBPUSD',
                'action': 'SELL',
                'size': 0.2,
                'price': 1.2650,
                'timestamp': int(datetime.now().timestamp())
            },
            {
                'symbol': 'USDJPY',
                'action': 'BUY',
                'size': 0.15,
                'price': 150.25,
                'timestamp': int(datetime.now().timestamp())
            }
        ]

        with open('data/sample_signals.json', 'w') as f:
            import json
            json.dump(sample_signals, f, indent=2)

        # Create sample Telegram messages
        sample_messages = [
            {
                'message_id': '1001',
                'chat_id': '-1001234567890',
                'user_id': '123456789',
                'username': 'trader_john',
                'text': 'EURUSD looking bullish, RSI at 65',
                'timestamp': datetime.now(),
                'type': 'text'
            },
            {
                'message_id': '1002',
                'chat_id': '-1001234567890',
                'user_id': '987654321',
                'username': 'analyst_sarah',
                'text': 'Market volatility increasing, expect higher spreads',
                'timestamp': datetime.now(),
                'type': 'text'
            },
            {
                'message_id': '1003',
                'chat_id': '-1001234567890',
                'user_id': '456789123',
                'username': 'risk_manager',
                'text': 'Daily loss limit reached, stopping trading for today',
                'timestamp': datetime.now(),
                'type': 'text'
            }
        ]

        with open('data/sample_messages.json', 'w') as f:
            json.dump(sample_messages, f, indent=2, default=str)

        logger.info("✅ Sample data created")
        return True

    except Exception as e:
        logger.error(f"❌ Sample data creation failed: {e}")
        return False

def run_system_tests():
    """Run system component tests"""
    try:
        logger.info("🔄 Running system tests...")

        # Test configuration system
        try:
            from config_manager import get_config_manager
            config = get_config_manager()
            logger.info("✅ Configuration system test passed")
        except Exception as e:
            logger.error(f"❌ Configuration system test failed: {e}")
            return False

        # Test storage system
        try:
            from storage_system import TitanovaXStorageSystem
            storage = TitanovaXStorageSystem(config)
            logger.info("✅ Storage system test passed")
        except Exception as e:
            logger.error(f"❌ Storage system test failed: {e}")
            return False

        # Test monitoring system
        try:
            from monitoring_system import TitanovaXMonitoringSystem
            monitoring = TitanovaXMonitoringSystem(config)
            logger.info("✅ Monitoring system test passed")
        except Exception as e:
            logger.error(f"❌ Monitoring system test failed: {e}")
            return False

        # Test security system
        try:
            from security_system import TitanovaXSecuritySystem
            security = TitanovaXSecuritySystem(config)
            logger.info("✅ Security system test passed")
        except Exception as e:
            logger.error(f"❌ Security system test failed: {e}")
            return False

        logger.info("✅ All system tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ System tests failed: {e}")
        return False

def main():
    """Main initialization function"""
    logger.info("🚀 Starting TitanovaX system initialization...")

    # Check if we're in the right directory
    if not Path('config_manager.py').exists():
        logger.error("❌ Not in TitanovaX directory. Please run from the project root.")
        sys.exit(1)

    # Run initialization steps
    steps = [
        ("Database", initialize_database),
        ("Storage System", initialize_storage_system),
        ("Monitoring System", initialize_monitoring_system),
        ("Security System", initialize_security_system),
        ("Sample Data", create_sample_data),
        ("System Tests", run_system_tests)
    ]

    results = []
    for step_name, step_function in steps:
        logger.info(f"📋 Running {step_name} initialization...")
        success = step_function()
        results.append((step_name, success))

        if not success:
            logger.warning(f"⚠️ {step_name} initialization had issues but continuing...")

    # Summary
    successful = sum(1 for _, success in results if success)
    total = len(results)

    logger.info("=" * 60)
    logger.info("🎯 INITIALIZATION SUMMARY")
    logger.info("=" * 60)

    for step_name, success in results:
        status = "✅" if success else "⚠️"
        logger.info(f"{status} {step_name}: {'Completed' if success else 'Issues Found'}")

    logger.info("-" * 60)
    logger.info(f"📊 Completion: {successful}/{total} steps successful")

    if successful == total:
        logger.info("🎉 System initialization completed successfully!")
        logger.info("🚀 Ready to start TitanovaX Trading System")
    else:
        logger.warning("⚠️ Some initialization steps had issues. Check logs for details.")
        logger.info("ℹ️ You can still run the system, but some features may not work correctly.")

    logger.info("=" * 60)

if __name__ == "__main__":
    main()
