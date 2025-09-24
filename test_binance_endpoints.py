#!/usr/bin/env python3
"""
Binance API Endpoint Test
Tests different Binance API endpoints to find working connection
"""

import requests
import json

def test_binance_endpoints():
    """Test various Binance API endpoints"""
    
    endpoints = [
        ("Main API", "https://api.binance.com/api/v3/ping"),
        ("Main API (US)", "https://api.binance.us/api/v3/ping"),
        ("Testnet (old)", "https://testnet.binance.vision/api/v3/ping"),
        ("Testnet (new)", "https://testnet.binancefuture.com/api/v3/ping"),
        ("Testnet (futures)", "https://testnet.binancefuture.com/fapi/v1/ping"),
    ]
    
    print("🔍 Testing Binance API Endpoints...")
    print("=" * 60)
    
    working_endpoints = []
    
    for name, url in endpoints:
        try:
            print(f"\nTesting {name}: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {name}: SUCCESS - {result}")
                working_endpoints.append((name, url))
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ {name}: Connection Error - {e}")
        except requests.exceptions.Timeout:
            print(f"❌ {name}: Timeout")
        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Request Error - {e}")
        except Exception as e:
            print(f"❌ {name}: Unexpected Error - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY: {len(working_endpoints)} working endpoints found")
    
    if working_endpoints:
        print("✅ Working endpoints:")
        for name, url in working_endpoints:
            print(f"   - {name}: {url}")
        return True
    else:
        print("❌ No working endpoints found")
        return False

def test_binance_main_api():
    """Test main Binance API functionality"""
    print("\n🔍 Testing Main Binance API Functionality...")
    print("=" * 60)
    
    try:
        # Test server time
        response = requests.get("https://api.binance.com/api/v3/time", timeout=10)
        if response.status_code == 200:
            server_time = response.json()
            print(f"✅ Server Time: {server_time['serverTime']}")
        
        # Test exchange info
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
        if response.status_code == 200:
            exchange_info = response.json()
            btc_pairs = [s for s in exchange_info['symbols'] if 'BTC' in s['symbol'] and s['status'] == 'TRADING']
            print(f"✅ Exchange Info: {len(exchange_info['symbols'])} total pairs, {len(btc_pairs)} BTC pairs trading")
        
        # Test recent trades
        response = requests.get("https://api.binance.com/api/v3/trades?symbol=BTCUSDT&limit=3", timeout=10)
        if response.status_code == 200:
            trades = response.json()
            print(f"✅ Recent Trades: {len(trades)} BTCUSDT trades retrieved")
            if trades:
                print(f"   Latest: Price ${trades[0]['price']}, Volume {trades[0]['qty']}")
        
        # Test order book
        response = requests.get("https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5", timeout=10)
        if response.status_code == 200:
            order_book = response.json()
            if order_book['bids'] and order_book['asks']:
                print(f"✅ Order Book: Top bid ${order_book['bids'][0][0]}, Top ask ${order_book['asks'][0][0]}")
        
        print("\n✅ Main Binance API is fully functional!")
        return True
        
    except Exception as e:
        print(f"❌ Main API Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 BINANCE API ENDPOINT TESTING")
    print("=" * 60)
    
    # Test endpoints
    endpoints_working = test_binance_endpoints()
    
    # Test main API functionality if endpoints work
    if endpoints_working:
        api_functional = test_binance_main_api()
        
        print("\n" + "=" * 60)
        print("📋 FINAL STATUS:")
        if api_functional:
            print("✅ Binance API is accessible and functional")
            print("✅ Main API endpoint: https://api.binance.com")
            print("⚠️  Testnet may be unavailable - use main API for testing")
        else:
            print("❌ Binance API has connectivity issues")
    else:
        print("❌ No Binance API endpoints are accessible")