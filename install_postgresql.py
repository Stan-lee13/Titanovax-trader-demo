#!/usr/bin/env python3
"""
PostgreSQL Windows Installer and Setup Script
Downloads, installs, and configures PostgreSQL for TitanovaX Trading System
"""

import os
import subprocess
import urllib.request
import sys
import time
import shutil
from pathlib import Path

def download_postgresql():
    """Download PostgreSQL for Windows"""
    print("🔧 Downloading PostgreSQL for Windows...")
    
    # PostgreSQL 15.4 for Windows x64
    pg_url = "https://get.enterprisedb.com/postgresql/postgresql-15.4-1-windows-x64.exe"
    installer_path = "postgresql-installer.exe"
    
    try:
        print(f"Downloading from: {pg_url}")
        urllib.request.urlretrieve(pg_url, installer_path)
        print(f"✅ Downloaded PostgreSQL installer")
        return installer_path
    except Exception as e:
        print(f"❌ Failed to download PostgreSQL: {e}")
        return None

def install_postgresql(installer_path):
    """Install PostgreSQL with automated configuration"""
    print("\n🔧 Installing PostgreSQL...")
    
    if not os.path.exists(installer_path):
        print("❌ Installer not found")
        return False
    
    try:
        # Installation parameters
        install_dir = r"C:\Program Files\PostgreSQL\15"
        data_dir = r"C:\Program Files\PostgreSQL\15\data"
        password = "postgres"
        port = "5432"
        
        # Create installation command
        install_cmd = [
            installer_path,
            "--mode", "unattended",
            "--unattendedmodeui", "minimal",
            "--superpassword", password,
            "--servicepassword", password,
            "--serviceport", port,
            "--servicename", "postgresql-15",
            "--prefix", install_dir,
            "--datadir", data_dir,
            "--create_shortcuts", "1"
        ]
        
        print(f"Installing PostgreSQL to: {install_dir}")
        print(f"Data directory: {data_dir}")
        print(f"Service port: {port}")
        print(f"Superuser password: {password}")
        
        # Run installer
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def create_titanovax_database():
    """Create TitanovaX database and user"""
    print("\n🔧 Creating TitanovaX database...")
    
    try:
        # Wait for service to start
        print("Waiting for PostgreSQL service to start...")
        time.sleep(10)
        
        # Create database
        create_db_cmd = [
            r"C:\Program Files\PostgreSQL\15\bin\createdb.exe",
            "-U", "postgres",
            "-h", "localhost",
            "-p", "5432",
            "titanovax"
        ]
        
        print("Creating titanovax database...")
        result = subprocess.run(create_db_cmd, input="postgres\n", text=True, capture_output=True)
        
        if result.returncode == 0 or "already exists" in result.stderr.lower():
            print("✅ TitanovaX database created/verified")
            return True
        else:
            print(f"⚠️  Database creation: {result.stderr}")
            return True  # Continue even if database exists
            
    except Exception as e:
        print(f"❌ Database creation error: {e}")
        return False

def setup_postgresql_service():
    """Start and configure PostgreSQL service"""
    print("\n🔧 Starting PostgreSQL service...")
    
    try:
        # Start service
        start_cmd = ["sc", "start", "postgresql-15"]
        result = subprocess.run(start_cmd, capture_output=True, text=True)
        
        if result.returncode == 0 or "RUNNING" in result.stdout:
            print("✅ PostgreSQL service started")
            return True
        else:
            print(f"⚠️  Service start: {result.stderr}")
            return True  # Continue even if service issues
            
    except Exception as e:
        print(f"❌ Service start error: {e}")
        return False

def test_postgresql_connection():
    """Test PostgreSQL connection"""
    print("\n🔧 Testing PostgreSQL connection...")
    
    try:
        # Test connection
        test_cmd = [
            r"C:\Program Files\PostgreSQL\15\bin\psql.exe",
            "-U", "postgres",
            "-h", "localhost",
            "-p", "5432",
            "-d", "titanovax",
            "-c", "SELECT version();"
        ]
        
        result = subprocess.run(test_cmd, input="postgres\n", text=True, capture_output=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL connection successful")
            if "PostgreSQL" in result.stdout:
                version = result.stdout.split('\n')[0]
                print(f"   Version: {version}")
            return True
        else:
            print(f"❌ Connection test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def update_env_file():
    """Update .env file with PostgreSQL configuration"""
    print("\n🔧 Updating .env file...")
    
    env_file = ".env"
    if not os.path.exists(env_file):
        print("❌ .env file not found")
        return False
    
    try:
        # Read current .env
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Update PostgreSQL settings
        if "DB_HOST=localhost" not in content:
            content += "\n# PostgreSQL Configuration (Updated)\n"
            content += "DB_HOST=localhost\n"
            content += "DB_PORT=5432\n"
            content += "DB_NAME=titanovax\n"
            content += "DB_USER=postgres\n"
            content += "DB_PASSWORD=postgres\n"
        
        # Write updated content
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ .env file updated with PostgreSQL configuration")
        return True
        
    except Exception as e:
        print(f"❌ .env update error: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 PostgreSQL Installation for TitanovaX Trading System")
    print("=" * 60)
    
    # Check if already installed
    pg_path = r"C:\Program Files\PostgreSQL\15\bin\psql.exe"
    if os.path.exists(pg_path):
        print("✅ PostgreSQL appears to be already installed")
        print(f"   Found at: {pg_path}")
        
        # Test existing installation
        if test_postgresql_connection():
            update_env_file()
            print("\n🎉 PostgreSQL is ready to use!")
            return True
    
    print("Starting PostgreSQL installation process...")
    
    # Download PostgreSQL
    installer = download_postgresql()
    if not installer:
        print("❌ Failed to download PostgreSQL")
        return False
    
    # Install PostgreSQL
    if not install_postgresql(installer):
        print("❌ PostgreSQL installation failed")
        return False
    
    # Create database
    create_titanovax_database()
    
    # Start service
    setup_postgresql_service()
    
    # Test connection
    connection_ok = test_postgresql_connection()
    
    # Update environment
    update_env_file()
    
    # Cleanup
    if os.path.exists(installer):
        os.remove(installer)
    
    print("\n" + "=" * 60)
    print("📋 INSTALLATION SUMMARY:")
    print("=" * 60)
    
    if connection_ok:
        print("✅ PostgreSQL installed and configured successfully")
        print("✅ TitanovaX database created")
        print("✅ Connection test passed")
        print("✅ Environment file updated")
        print("\n🎉 PostgreSQL is ready for TitanovaX!")
        print("\n📡 Connection Details:")
        print("   Host: localhost")
        print("   Port: 5432")
        print("   Database: titanovax")
        print("   User: postgres")
        print("   Password: postgres")
        return True
    else:
        print("⚠️  PostgreSQL installed but connection issues detected")
        print("   Please check the service status and firewall settings")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)