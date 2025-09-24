#!/usr/bin/env python3
"""
Binance API Connection Test
Tests the Binance API connectivity and functionality
"""

import asyncio
import sys
from binance_collector import BinanceDataCollector
from config_manager import ConfigManager

async def test_binance_connection():
    """Test Binance API connection"""
    try:
        print("🔄 Testing Binance API Connection...")
        
        # Initialize config and collector
        config = ConfigManager()
        collector = BinanceDataCollector(config)
        
        print("📡 Connecting to Binance Testnet...")
        
        # Test server time (no authentication required)
        server_time = await collector.client.get_server_time()
        print(f"✅ Server Time: {server_time['serverTime']}")
        
        # Test exchange info
        exchange_info = await collector.client.get_exchange_info()
        print(f"✅ Exchange Info: {len(exchange_info['symbols'])} trading pairs available")
        
        # Test recent trades for BTCUSDT
        trades = await collector.client.get_recent_trades(symbol='BTCUSDT', limit=3)
        print(f"✅ Market Data: Retrieved {len(trades)} recent BTCUSDT trades")
        
        # Test account info (requires authentication)
        try:
            account_info = await collector.client.get_account()
            print(f"✅ Account Access: Account type: {account_info['accountType']}")
            print(f"✅ Account Status: {account_info['canTrade']}")
        except Exception as e:
            print(f"⚠️  Account Access: Limited (testnet mode): {e}")
        
        print("\n✅ Binance API connection successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Binance API Error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🔍 BINANCE API CONNECTION TEST")
    print("=" * 60)
    
    # Run the async test
    result = asyncio.run(test_binance_connection())
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL STATUS: {'✅ CONNECTED' if result else '❌ FAILED'}")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)