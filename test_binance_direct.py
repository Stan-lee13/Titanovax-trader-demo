#!/usr/bin/env python3
"""
Direct Binance API Test using python-binance library
Tests Binance API connectivity without Redis dependency
"""

import asyncio
import os
from binance import AsyncClient
from binance.exceptions import BinanceAPIException

async def test_binance_direct():
    """Test Binance API directly using python-binance"""
    try:
        print("🔄 Testing Direct Binance API Connection...")
        
        # Get API credentials from environment
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_KEY')
        testnet = os.getenv('BINANCE_USE_TESTNET', 'true').lower() == 'true'
        
        print(f"📡 Using Testnet: {testnet}")
        print(f"🔑 API Key configured: {'Yes' if api_key else 'No'}")
        print(f"🔐 Secret Key configured: {'Yes' if api_secret else 'No'}")
        
        # Initialize client
        client = AsyncClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        print("🔗 Connecting to Binance...")
        
        # Test server time (no authentication required)
        try:
            server_time = await client.get_server_time()
            print(f"✅ Server Time: {server_time['serverTime']}")
        except Exception as e:
            print(f"❌ Server Time Error: {e}")
            return False
        
        # Test exchange info
        try:
            exchange_info = await client.get_exchange_info()
            btc_pairs = [s for s in exchange_info['symbols'] if 'BTC' in s['symbol']]
            print(f"✅ Exchange Info: {len(exchange_info['symbols'])} total pairs, {len(btc_pairs)} BTC pairs")
        except Exception as e:
            print(f"❌ Exchange Info Error: {e}")
        
        # Test recent trades for BTCUSDT
        try:
            trades = await client.get_recent_trades(symbol='BTCUSDT', limit=3)
            print(f"✅ Market Data: Retrieved {len(trades)} recent BTCUSDT trades")
            if trades:
                print(f"   Latest trade: Price ${trades[0]['price']}, Volume {trades[0]['qty']}")
        except Exception as e:
            print(f"❌ Market Data Error: {e}")
        
        # Test account info (requires authentication)
        if api_key and api_secret:
            try:
                account_info = await client.get_account()
                print(f"✅ Account Access: Type: {account_info['accountType']}, Can Trade: {account_info['canTrade']}")
                print(f"✅ Account Balance: {len(account_info['balances'])} assets")
            except Exception as e:
                print(f"⚠️  Account Access Limited: {e}")
        else:
            print("⚠️  Skipping account test - no API credentials configured")
        
        # Test order book
        try:
            order_book = await client.get_order_book(symbol='BTCUSDT', limit=5)
            print(f"✅ Order Book: Top bid ${order_book['bids'][0][0]}, Top ask ${order_book['asks'][0][0]}")
        except Exception as e:
            print(f"❌ Order Book Error: {e}")
        
        # Close client
        await client.close_connection()
        
        print("\n✅ Binance API connection successful!")
        return True
        
    except BinanceAPIException as e:
        print(f"\n❌ Binance API Exception: {e}")
        print(f"   Status Code: {e.status_code}")
        print(f"   Error Code: {e.code}")
        return False
    except Exception as e:
        print(f"\n❌ General Error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🔍 DIRECT BINANCE API CONNECTION TEST")
    print("=" * 60)
    
    # Run the async test
    result = asyncio.run(test_binance_direct())
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL STATUS: {'✅ CONNECTED' if result else '❌ FAILED'}")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)