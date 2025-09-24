#!/usr/bin/env python3
"""
Redis Service Connectivity Test
Tests Redis service status and connectivity
"""

import redis
import time

def test_redis_connection():
    """Test Redis connection and basic operations"""
    print("🔍 Testing Redis Connection...")
    print("=" * 50)
    
    try:
        # Test connection to localhost:6379
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, socket_timeout=5)
        
        # Test basic ping
        ping_result = r.ping()
        print(f"✅ Redis Ping: {ping_result}")
        
        # Test set and get
        test_key = 'titanovax_test_key'
        test_value = 'Hello from TitanovaX'
        
        r.set(test_key, test_value, ex=60)  # Expire in 60 seconds
        retrieved_value = r.get(test_key)
        
        if retrieved_value == test_value:
            print(f"✅ Redis Set/Get: Successfully stored and retrieved value")
        else:
            print(f"❌ Redis Set/Get: Value mismatch")
        
        # Test info
        info = r.info()
        print(f"✅ Redis Version: {info.get('redis_version', 'Unknown')}")
        print(f"✅ Memory Usage: {info.get('used_memory_human', 'Unknown')}")
        print(f"✅ Connected Clients: {info.get('connected_clients', 'Unknown')}")
        
        # Clean up
        r.delete(test_key)
        
        print("\n✅ Redis connection successful!")
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Redis Connection Error: {e}")
        print("   Redis service may not be running")
        return False
    except Exception as e:
        print(f"❌ Redis Error: {e}")
        return False

def test_redis_service_status():
    """Check if Redis service is running"""
    print("\n🔍 Checking Redis Service Status...")
    print("=" * 50)
    
    try:
        import socket
        
        # Try to connect to Redis port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('localhost', 6379))
        sock.close()
        
        if result == 0:
            print("✅ Redis port 6379 is open and accepting connections")
            return True
        else:
            print(f"❌ Redis port 6379 is not accessible (error code: {result})")
            print("   Possible causes:")
            print("   - Redis service is not running")
            print("   - Redis is configured on a different port")
            print("   - Firewall is blocking the connection")
            return False
            
    except Exception as e:
        print(f"❌ Service check error: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 REDIS SERVICE CONNECTIVITY TEST")
    print("=" * 60)
    
    # First check if service is running
    service_running = test_redis_service_status()
    
    if service_running:
        # Test actual Redis operations
        connection_successful = test_redis_connection()
        
        print("\n" + "=" * 60)
        print("📋 FINAL STATUS:")
        if connection_successful:
            print("✅ Redis service is running and functional")
            print("✅ Connection: localhost:6379")
            print("✅ Basic operations: Working")
        else:
            print("⚠️  Redis service is running but has connection issues")
    else:
        print("❌ Redis service is not running or not accessible")
        print("\n🎯 NEXT STEPS:")
        print("   1. Start Redis service: redis-server")
        print("   2. Check Redis configuration")
        print("   3. Verify port 6379 is open")
    
    print("=" * 60)
    return service_running

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)