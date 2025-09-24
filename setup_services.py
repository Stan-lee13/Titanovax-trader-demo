#!/usr/bin/env python3
"""
Windows Service Setup Helper for TitanovaX Trading System
Provides installation and setup guidance for Redis and PostgreSQL on Windows
"""

import os
import subprocess
import sys
import urllib.request
import zipfile
import shutil

def download_file(url, filename):
    """Download a file with progress indication"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def setup_redis_windows():
    """Setup Redis on Windows"""
    print("\n🔧 Setting up Redis for Windows...")
    print("=" * 60)
    
    # Check if Redis is already running
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        if ':6379' in result.stdout:
            print("✅ Redis appears to be running on port 6379")
            return True
    except:
        pass
    
    # Download Redis for Windows
    redis_url = "https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip"
    redis_zip = "redis-windows.zip"
    redis_dir = "redis-windows"
    
    if not os.path.exists(redis_dir):
        if download_file(redis_url, redis_zip):
            try:
                with zipfile.ZipFile(redis_zip, 'r') as zip_ref:
                    zip_ref.extractall(redis_dir)
                os.remove(redis_zip)
                print("✅ Redis extracted successfully")
            except Exception as e:
                print(f"❌ Failed to extract Redis: {e}")
                return False
    
    # Start Redis server
    redis_server = os.path.join(redis_dir, "redis-server.exe")
    if os.path.exists(redis_server):
        print("Starting Redis server...")
        try:
            # Start Redis in background
            subprocess.Popen([redis_server], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
            print("✅ Redis server started")
            return True
        except Exception as e:
            print(f"❌ Failed to start Redis: {e}")
            return False
    else:
        print("❌ Redis server executable not found")
        return False

def setup_postgresql_windows():
    """Setup PostgreSQL on Windows"""
    print("\n🔧 Setting up PostgreSQL for Windows...")
    print("=" * 60)
    
    # Check if PostgreSQL is already running
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        if ':5432' in result.stdout:
            print("✅ PostgreSQL appears to be running on port 5432")
            return True
    except:
        pass
    
    print("PostgreSQL setup requires manual installation.")
    print("Please follow these steps:")
    print("\n1. Download PostgreSQL installer:")
    print("   https://www.postgresql.org/download/windows/")
    print("\n2. Run the installer and choose:")
    print("   - Port: 5432")
    print("   - Password: postgres")
    print("   - Username: postgres")
    print("\n3. After installation, create the titanovax database:")
    print("   psql -U postgres -c \"CREATE DATABASE titanovax;\"")
    
    return False

def create_database_config():
    """Create database configuration file"""
    print("\n📝 Creating database configuration...")
    
    # Check current .env file
    env_file = ".env"
    db_config = """
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=titanovax
DB_USER=postgres
DB_PASSWORD=postgres
"""
    
    redis_config = """
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
"""
    
    # Read existing .env file
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Add database config if not present
        if 'DB_HOST' not in content:
            with open(env_file, 'a') as f:
                f.write(db_config)
            print("✅ Database configuration added to .env file")
        
        if 'REDIS_HOST' not in content:
            with open(env_file, 'a') as f:
                f.write(redis_config)
            print("✅ Redis configuration added to .env file")
    else:
        print("❌ .env file not found - please create it first")

def main():
    """Main setup function"""
    print("🚀 TitanovaX Service Setup Helper")
    print("=" * 60)
    print("This script will help you set up Redis and PostgreSQL services")
    print("=" * 60)
    
    # Setup Redis
    redis_success = setup_redis_windows()
    
    # Setup PostgreSQL (guide only)
    setup_postgresql_windows()
    
    # Create database configuration
    create_database_config()
    
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY:")
    print("=" * 60)
    
    if redis_success:
        print("✅ Redis: Running on localhost:6379")
    else:
        print("❌ Redis: Setup incomplete")
    
    print("⚠️  PostgreSQL: Manual installation required")
    print("✅ Configuration: Updated .env file")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Install PostgreSQL manually (see instructions above)")
    print("2. Start Redis server if not running")
    print("3. Run service tests to verify connectivity")
    print("4. Proceed with Binance API testing")
    
    return redis_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)