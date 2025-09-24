#!/usr/bin/env python3
"""
Isolated configuration test to verify credentials without external dependencies
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables manually to test without full config manager
from dotenv import load_dotenv
load_dotenv()

def test_environment_variables():
    """Test all environment variables are loaded correctly"""
    print("=== Environment Variables Test ===")
    
    # Test Binance configuration
    print("\n🔍 Binance Configuration:")
    binance_api_key = os.getenv('BINANCE_API_KEY')
    binance_secret_key = os.getenv('BINANCE_SECRET_KEY')
    binance_testnet = os.getenv('BINANCE_USE_TESTNET', 'true').lower() == 'true'
    
    print(f"  API Key configured: {bool(binance_api_key)}")
    print(f"  Secret Key configured: {bool(binance_secret_key)}")
    print(f"  Testnet enabled: {binance_testnet}")
    if binance_api_key:
        print(f"  API Key length: {len(binance_api_key)}")
    if binance_secret_key:
        print(f"  Secret Key length: {len(binance_secret_key)}")
    
    # Test MT5 configuration
    print("\n🔍 MT5 Configuration:")
    mt5_login = os.getenv('MT5_LOGIN')
    mt5_password = os.getenv('MT5_PASSWORD')
    mt5_server = os.getenv('MT5_SERVER')
    
    print(f"  Login: {mt5_login}")
    print(f"  Password configured: {bool(mt5_password)}")
    print(f"  Server: {mt5_server}")
    
    # Test OANDA configuration
    print("\n🔍 OANDA Configuration:")
    oanda_api_key = os.getenv('OANDA_API_KEY')
    oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')
    oanda_use_demo = os.getenv('OANDA_USE_DEMO', 'true').lower() == 'true'
    
    print(f"  API Key configured: {bool(oanda_api_key)}")
    print(f"  Account ID: {oanda_account_id}")
    print(f"  Demo account: {oanda_use_demo}")
    
    # Test Database configuration
    print("\n🔍 Database Configuration:")
    db_host = os.getenv('DATABASE_HOST', 'localhost')
    db_port = os.getenv('DATABASE_PORT', '5432')
    db_name = os.getenv('DATABASE_NAME', 'titanovax')
    db_user = os.getenv('DATABASE_USER')
    
    print(f"  Host: {db_host}")
    print(f"  Port: {db_port}")
    print(f"  Database: {db_name}")
    print(f"  User configured: {bool(db_user)}")
    
    # Test Redis configuration
    print("\n🔍 Redis Configuration:")
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = os.getenv('REDIS_PORT', '6379')
    
    print(f"  Host: {redis_host}")
    print(f"  Port: {redis_port}")
    
    return {
        'binance': {
            'api_key': binance_api_key,
            'secret_key': binance_secret_key,
            'testnet': binance_testnet
        },
        'mt5': {
            'login': mt5_login,
            'password': mt5_password,
            'server': mt5_server
        },
        'oanda': {
            'api_key': oanda_api_key,
            'account_id': oanda_account_id,
            'demo': oanda_use_demo
        },
        'database': {
            'host': db_host,
            'port': db_port,
            'name': db_name,
            'user': db_user
        },
        'redis': {
            'host': redis_host,
            'port': redis_port
        }
    }

def test_binance_connectivity():
    """Test Binance API connectivity"""
    print("\n=== Binance Connectivity Test ===")
    
    try:
        from binance.client import Client
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        use_testnet = os.getenv('BINANCE_USE_TESTNET', 'true').lower() == 'true'
        
        if not api_key or not secret_key:
            print("❌ Binance credentials not configured")
            return False
        
        # Create client with testnet
        client = Client(api_key, secret_key, testnet=use_testnet)
        
        # Test API connectivity
        try:
            status = client.get_system_status()
            print(f"✅ Binance system status: {status}")
            
            # Test account info
            account = client.get_account()
            print(f"✅ Account connected: {account['accountType']}")
            
            return True
        except Exception as e:
            print(f"❌ Binance API error: {e}")
            return False
            
    except ImportError:
        print("⚠️  python-binance not installed, skipping connectivity test")
        return None
    except Exception as e:
        print(f"❌ Unexpected error testing Binance: {e}")
        return False

def test_file_permissions():
    """Test file permissions and accessibility"""
    print("\n=== File Permissions Test ===")
    
    test_files = [
        '.env',
        'config_manager.py',
        'binance_collector.py',
        'requirements.txt'
    ]
    
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read(100)  # Read first 100 chars
                print(f"✅ {file_path}: Accessible ({len(content)} chars)")
            else:
                print(f"⚠️  {file_path}: Not found")
        except Exception as e:
            print(f"❌ {file_path}: Error - {e}")

def main():
    """Main test function"""
    print("🚀 TitanovaX Configuration Verification")
    print("=" * 50)
    
    # Test environment variables
    config = test_environment_variables()
    
    # Test Binance connectivity
    binance_status = test_binance_connectivity()
    
    # Test file permissions
    test_file_permissions()
    
    # Summary
    print("\n=== Summary ===")
    
    # Binance status
    if config['binance']['api_key'] and config['binance']['secret_key']:
        print("✅ Binance credentials: Configured")
        if binance_status is True:
            print("✅ Binance API: Connected")
        elif binance_status is False:
            print("❌ Binance API: Connection failed")
        else:
            print("⚠️  Binance API: Skipped (library not available)")
    else:
        print("❌ Binance credentials: Missing")
    
    # MT5 status
    if config['mt5']['login'] and config['mt5']['password'] and config['mt5']['server']:
        if config['mt5']['login'] == 'YOUR_MT5_LOGIN' or config['mt5']['password'] == 'YOUR_MT5_PASSWORD':
            print("⚠️  MT5 credentials: Placeholder values detected - needs real credentials")
        else:
            print("✅ MT5 credentials: Configured")
    else:
        print("❌ MT5 credentials: Missing or incomplete")
    
    # OANDA status
    if config['oanda']['api_key']:
        print("✅ OANDA credentials: Configured")
    else:
        print("⚠️  OANDA credentials: Not configured (optional)")
    
    # Database status
    if config['database']['user']:
        print("✅ Database credentials: Configured")
    else:
        print("⚠️  Database credentials: Missing or incomplete")
    
    # Redis status
    print("⚠️  Redis: Connection not tested (service may not be running)")
    
    print("\n" + "=" * 50)
    print("🔍 Review the results above to identify any configuration issues")

if __name__ == "__main__":
    main()