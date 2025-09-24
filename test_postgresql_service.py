#!/usr/bin/env python3
"""
PostgreSQL Service Connectivity Test
Tests PostgreSQL service status and database connectivity
"""

import psycopg2
import socket

def test_postgresql_service():
    """Test PostgreSQL service connectivity"""
    print("🔍 Testing PostgreSQL Service...")
    print("=" * 60)
    
    try:
        # Test port connectivity
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 5432))
        sock.close()
        
        if result == 0:
            print("✅ PostgreSQL port 5432 is open and accepting connections")
            return True
        else:
            print(f"❌ PostgreSQL port 5432 is not accessible (error code: {result})")
            print("   Possible causes:")
            print("   - PostgreSQL service is not running")
            print("   - PostgreSQL is configured on a different port")
            print("   - Firewall is blocking the connection")
            return False
            
    except Exception as e:
        print(f"❌ Service check error: {e}")
        return False

def test_postgresql_connection():
    """Test actual PostgreSQL database connection"""
    print("\n🔍 Testing PostgreSQL Database Connection...")
    print("=" * 60)
    
    # Database connection parameters from environment
    import os
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'titanovax'),
        'user': os.getenv('DB_USER', ''),
        'password': os.getenv('DB_PASSWORD', ''),
        'connect_timeout': 5
    }
    
    print(f"📡 Connection Parameters:")
    print(f"   Host: {db_config['host']}")
    print(f"   Port: {db_config['port']}")
    print(f"   Database: {db_config['database']}")
    print(f"   User: {'Configured' if db_config['user'] else 'Not configured'}")
    print(f"   Password: {'Configured' if db_config['password'] else 'Not configured'}")
    
    if not db_config['user'] or not db_config['password']:
        print("\n⚠️  Database credentials are incomplete")
        print("   Please check DB_USER and DB_PASSWORD environment variables")
        return False
    
    try:
        # Attempt connection
        conn = psycopg2.connect(**db_config)
        
        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ PostgreSQL Version: {version[0]}")
        
        # Test database info
        cursor.execute("SELECT current_database(), current_user;")
        db_info = cursor.fetchone()
        print(f"✅ Connected to: {db_info[0]} as {db_info[1]}")
        
        # Test table existence
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            LIMIT 5;
        """)
        tables = cursor.fetchall()
        
        if tables:
            print(f"✅ Found {len(tables)} tables in database")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("⚠️  No tables found in database (may be empty)")
        
        # Clean up
        cursor.close()
        conn.close()
        
        print("\n✅ PostgreSQL connection successful!")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ PostgreSQL Operational Error: {e}")
        print("   This usually indicates:")
        print("   - Database service is not running")
        print("   - Invalid credentials")
        print("   - Database does not exist")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL Error: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 POSTGRESQL SERVICE CONNECTIVITY TEST")
    print("=" * 60)
    
    # First check if service is running
    service_running = test_postgresql_service()
    
    if service_running:
        # Test actual database connection
        connection_successful = test_postgresql_connection()
        
        print("\n" + "=" * 60)
        print("📋 FINAL STATUS:")
        if connection_successful:
            print("✅ PostgreSQL service is running and functional")
            print("✅ Connection: localhost:5432")
            print("✅ Database operations: Working")
        else:
            print("⚠️  PostgreSQL service is running but has connection issues")
            print("   Check credentials and database configuration")
    else:
        print("❌ PostgreSQL service is not running or not accessible")
        print("\n🎯 NEXT STEPS:")
        print("   1. Start PostgreSQL service")
        print("   2. Check PostgreSQL configuration")
        print("   3. Verify port 5432 is open")
        print("   4. Check database credentials in .env file")
    
    print("=" * 60)
    return service_running

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)