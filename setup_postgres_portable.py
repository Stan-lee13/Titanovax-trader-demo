#!/usr/bin/env python3
"""
Lightweight PostgreSQL Setup Script for TitanovaX
Creates a local PostgreSQL installation using portable binaries
"""

import os
import subprocess
import zipfile
import urllib.request
import shutil
from pathlib import Path

def create_postgres_config():
    """Create PostgreSQL configuration files"""
    print("🔧 Creating PostgreSQL configuration...")
    
    # Create data directory
    data_dir = Path("postgres_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create config directory
    config_dir = Path("postgres_config")
    config_dir.mkdir(exist_ok=True)
    
    # Create postgresql.conf
    postgresql_conf = """
# TitanovaX PostgreSQL Configuration
listen_addresses = 'localhost'
port = 5432
max_connections = 100
shared_buffers = 128MB
dynamic_shared_memory_type = windows
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 10MB
"""
    
    with open(config_dir / "postgresql.conf", "w") as f:
        f.write(postgresql_conf)
    
    # Create pg_hba.conf
    pg_hba_conf = """
# PostgreSQL Client Authentication Configuration
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
"""
    
    with open(config_dir / "pg_hba.conf", "w") as f:
        f.write(pg_hba_conf)
    
    print("✅ Configuration files created")
    return True

def setup_postgres_portable():
    """Setup portable PostgreSQL using available tools"""
    print("🔧 Setting up portable PostgreSQL...")
    
    # Check if PostgreSQL binaries exist in system
    pg_paths = [
        r"C:\Program Files\PostgreSQL\15\bin",
        r"C:\Program Files\PostgreSQL\14\bin", 
        r"C:\Program Files\PostgreSQL\13\bin",
        r"C:\Program Files\PostgreSQL\12\bin",
    ]
    
    pg_bin = None
    for path in pg_paths:
        if os.path.exists(path):
            pg_bin = path
            break
    
    if not pg_bin:
        print("❌ No PostgreSQL installation found")
        return False
    
    print(f"✅ Found PostgreSQL binaries at: {pg_bin}")
    
    # Create configuration
    create_postgres_config()
    
    # Initialize database cluster
    initdb_path = os.path.join(pg_bin, "initdb.exe")
    data_dir = Path("postgres_data").absolute()
    
    try:
        print("Initializing database cluster...")
        subprocess.run([
            initdb_path,
            "-D", str(data_dir),
            "-U", "postgres",
            "--encoding=UTF8",
            "--locale=en_US.UTF-8"
        ], check=True)
        
        print("✅ Database cluster initialized")
        
        # Create TitanovaX database
        createdb_path = os.path.join(pg_bin, "createdb.exe")
        print("Creating TitanovaX database...")
        subprocess.run([
            createdb_path,
            "-U", "postgres",
            "-h", "localhost",
            "-p", "5432",
            "titanovax"
        ], check=True)
        
        print("✅ TitanovaX database created")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ PostgreSQL setup failed: {e}")
        return False

def create_startup_script():
    """Create startup script for PostgreSQL"""
    print("🔧 Creating startup script...")
    
    # Find PostgreSQL binaries
    pg_paths = [
        r"C:\Program Files\PostgreSQL\15\bin",
        r"C:\Program Files\PostgreSQL\14\bin", 
        r"C:\Program Files\PostgreSQL\13\bin",
        r"C:\Program Files\PostgreSQL\12\bin",
    ]
    
    pg_bin = None
    for path in pg_paths:
        if os.path.exists(path):
            pg_bin = path
            break
    
    if not pg_bin:
        print("❌ No PostgreSQL binaries found")
        return False
    
    # Create startup script
    startup_script = f"""@echo off
echo Starting PostgreSQL for TitanovaX...
cd /d "{os.getcwd()}"
"{pg_bin}\\postgres.exe" -D "postgres_data" -p 5432
echo PostgreSQL started on port 5432
pause
"""
    
    with open("start_postgres.bat", "w") as f:
        f.write(startup_script)
    
    print("✅ Startup script created: start_postgres.bat")
    return True

def test_postgres_connection():
    """Test PostgreSQL connection"""
    print("🔧 Testing PostgreSQL connection...")
    
    # Find psql
    pg_paths = [
        r"C:\Program Files\PostgreSQL\15\bin",
        r"C:\Program Files\PostgreSQL\14\bin", 
        r"C:\Program Files\PostgreSQL\13\bin",
        r"C:\Program Files\PostgreSQL\12\bin",
    ]
    
    psql_path = None
    for path in pg_paths:
        psql = os.path.join(path, "psql.exe")
        if os.path.exists(psql):
            psql_path = psql
            break
    
    if not psql_path:
        print("❌ psql not found")
        return False
    
    try:
        result = subprocess.run([
            psql_path,
            "-U", "postgres",
            "-h", "localhost",
            "-p", "5432",
            "-d", "titanovax",
            "-c", "SELECT version();"
        ], capture_output=True, text=True)
        
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

def main():
    """Main setup function"""
    print("🚀 PostgreSQL Setup for TitanovaX Trading System")
    print("=" * 60)
    
    # Setup PostgreSQL
    if setup_postgres_portable():
        # Create startup script
        create_startup_script()
        
        # Test connection
        if test_postgres_connection():
            print("\n🎉 PostgreSQL setup completed successfully!")
            print("\n📡 To start PostgreSQL:")
            print("   1. Run: start_postgres.bat")
            print("   2. Or run postgres.exe manually")
            print("\n📡 Connection Details:")
            print("   Host: localhost")
            print("   Port: 5432")
            print("   Database: titanovax")
            print("   User: postgres")
            print("   Password: (none required for local trust auth)")
            return True
        else:
            print("\n⚠️  Setup completed but connection test failed")
            print("   Please start PostgreSQL manually and test again")
            return False
    else:
        print("\n❌ PostgreSQL setup failed")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)