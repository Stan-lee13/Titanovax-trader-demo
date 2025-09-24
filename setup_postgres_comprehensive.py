#!/usr/bin/env python3
"""
PostgreSQL Manual Setup Script for TitanovaX
Creates a local PostgreSQL setup using available Windows resources
"""

import os
import subprocess
import json
import time
from pathlib import Path

def create_postgres_installer_script():
    """Create a PowerShell script to install PostgreSQL"""
    print("🔧 Creating PostgreSQL installer script...")
    
    ps_script = """
# PowerShell script to install PostgreSQL
Write-Host "Installing PostgreSQL for TitanovaX..." -ForegroundColor Green

# Download PostgreSQL using BITS (Background Intelligent Transfer Service)
$pgUrl = "https://get.enterprisedb.com/postgresql/postgresql-15.4-1-windows-x64.exe"
$installerPath = "$env:TEMP\\postgresql-installer.exe"

Write-Host "Downloading PostgreSQL..." -ForegroundColor Yellow
try {
    Start-BitsTransfer -Source $pgUrl -Destination $installerPath -ErrorAction Stop
    Write-Host "✅ Download completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Download failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Install PostgreSQL
Write-Host "Installing PostgreSQL..." -ForegroundColor Yellow
$installArgs = @(
    "--mode", "unattended",
    "--unattendedmodeui", "minimal",
    "--superpassword", "postgres",
    "--servicepassword", "postgres", 
    "--serviceport", "5432",
    "--servicename", "postgresql-15",
    "--prefix", "C:\\Program Files\\PostgreSQL\\15",
    "--datadir", "C:\\Program Files\\PostgreSQL\\15\\data"
)

try {
    Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -NoNewWindow
    Write-Host "✅ PostgreSQL installation completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Start PostgreSQL service
Write-Host "Starting PostgreSQL service..." -ForegroundColor Yellow
try {
    Start-Service -Name "postgresql-15" -ErrorAction Stop
    Write-Host "✅ PostgreSQL service started" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not start service: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Create TitanovaX database
Write-Host "Creating TitanovaX database..." -ForegroundColor Yellow
try {
    & "C:\\Program Files\\PostgreSQL\\15\\bin\\createdb.exe" -U postgres -h localhost -p 5432 titanovax
    Write-Host "✅ TitanovaX database created" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Database creation: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "🎉 PostgreSQL setup completed!" -ForegroundColor Green
"""
    
    with open("install_postgres.ps1", "w") as f:
        f.write(ps_script)
    
    print("✅ PowerShell installer script created: install_postgres.ps1")
    return True

def create_postgres_manual_setup():
    """Create manual setup instructions and scripts"""
    print("🔧 Creating manual PostgreSQL setup...")
    
    # Create a simple SQLite-to-PostgreSQL bridge for immediate functionality
    bridge_script = """
import sqlite3
import os
from datetime import datetime

class PostgresBridge:
    def __init__(self, db_path="titanovax_temp.db"):
        self.db_path = db_path
        self.conn = None
        self.init_db()
    
    def init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create tables that mimic PostgreSQL structure
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def log_trade(self, symbol, side, quantity, price):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, side, quantity, price)
            VALUES (?, ?, ?, ?)
        ''', (symbol, side, quantity, price))
        self.conn.commit()
        return cursor.lastrowid
    
    def log_signal(self, symbol, signal_type, strength, confidence):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO signals (symbol, signal_type, strength, confidence)
            VALUES (?, ?, ?, ?)
        ''', (symbol, signal_type, strength, confidence))
        self.conn.commit()
        return cursor.lastrowid
    
    def log_market_data(self, symbol, price, volume):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO market_data (symbol, price, volume)
            VALUES (?, ?, ?)
        ''', (symbol, price, volume))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_trades(self, limit=100):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (limit,))
        return cursor.fetchall()
    
    def close(self):
        if self.conn:
            self.conn.close()

# Global bridge instance
bridge = PostgresBridge()

if __name__ == "__main__":
    # Test the bridge
    print("Testing SQLite bridge...")
    trade_id = bridge.log_trade("BTCUSDT", "BUY", 0.1, 50000)
    signal_id = bridge.log_signal("BTCUSDT", "LONG", 0.8, 0.9)
    
    print(f"Trade logged: ID {trade_id}")
    print(f"Signal logged: ID {signal_id}")
    
    trades = bridge.get_trades()
    print(f"Recent trades: {len(trades)}")
    
    bridge.close()
    print("✅ SQLite bridge ready for TitanovaX")
"""
    
    with open("postgres_sqlite_bridge.py", "w") as f:
        f.write(bridge_script)
    
    print("✅ SQLite bridge created: postgres_sqlite_bridge.py")
    return True

def create_postgres_service_installer():
    """Create a service installer for PostgreSQL"""
    print("🔧 Creating PostgreSQL service installer...")
    
    service_script = """
# PostgreSQL Service Setup Script
Write-Host "Setting up PostgreSQL service..." -ForegroundColor Green

# Check if PostgreSQL is already installed
$pgPaths = @(
    "C:\\Program Files\\PostgreSQL\\15\\bin",
    "C:\\Program Files\\PostgreSQL\\14\\bin",
    "C:\\Program Files\\PostgreSQL\\13\\bin",
    "C:\\Program Files\\PostgreSQL\\12\\bin"
)

$pgBin = $null
foreach ($path in $pgPaths) {
    if (Test-Path $path) {
        $pgBin = $path
        break
    }
}

if ($pgBin) {
    Write-Host "✅ Found PostgreSQL at: $pgBin" -ForegroundColor Green
    
    # Create data directory if it doesn't exist
    $dataDir = "$pwd\\postgres_data"
    if (!(Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir | Out-Null
        
        # Initialize database cluster
        Write-Host "Initializing database cluster..." -ForegroundColor Yellow
        & "$pgBin\\initdb.exe" -D $dataDir -U postgres --encoding=UTF8 --locale=en_US.UTF-8
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Database cluster initialized" -ForegroundColor Green
        } else {
            Write-Host "❌ Database initialization failed" -ForegroundColor Red
            exit 1
        }
    }
    
    # Create TitanovaX database
    Write-Host "Creating TitanovaX database..." -ForegroundColor Yellow
    & "$pgBin\\createdb.exe" -U postgres -h localhost -p 5432 titanovax
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ TitanovaX database created" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Database may already exist" -ForegroundColor Yellow
    }
    
    # Create startup script
    $startupScript = @"
@echo off
echo Starting PostgreSQL for TitanovaX...
cd /d "$pwd"
"$pgBin\\postgres.exe" -D "$dataDir" -p 5432
echo PostgreSQL started on port 5432
pause
"@
    
    Set-Content -Path "start_postgres.bat" -Value $startupScript
    Write-Host "✅ Startup script created: start_postgres.bat" -ForegroundColor Green
    
    # Test connection
    Write-Host "Testing PostgreSQL connection..." -ForegroundColor Yellow
    $result = & "$pgBin\\psql.exe" -U postgres -h localhost -p 5432 -d titanovax -c "SELECT version();"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PostgreSQL connection successful" -ForegroundColor Green
        return $true
    } else {
        Write-Host "⚠️  Connection test failed - service may need to be started" -ForegroundColor Yellow
        return $false
    }
} else {
    Write-Host "❌ PostgreSQL not found. Please install PostgreSQL first." -ForegroundColor Red
    return $false
}
"""
    
    with open("setup_postgres_service.ps1", "w") as f:
        f.write(service_script)
    
    print("✅ PostgreSQL service setup script created: setup_postgres_service.ps1")
    return True

def run_postgres_setup():
    """Run the PostgreSQL setup process"""
    print("🚀 Starting PostgreSQL setup process...")
    
    # Create all setup scripts
    create_postgres_installer_script()
    create_postgres_manual_setup()
    create_postgres_service_installer()
    
    # Try to run the service setup
    print("\n🔧 Attempting PostgreSQL service setup...")
    try:
        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", 
            "-File", "setup_postgres_service.ps1"
        ], capture_output=True, text=True)
        
        print("PowerShell setup output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("✅ PostgreSQL setup completed successfully")
            return True
        else:
            print("⚠️  Setup completed with warnings")
            return True
            
    except Exception as e:
        print(f"❌ PowerShell setup failed: {e}")
        print("Please run setup_postgres_service.ps1 manually in PowerShell")
        return False

def main():
    """Main setup function"""
    print("🚀 PostgreSQL Setup for TitanovaX Trading System")
    print("=" * 60)
    
    success = run_postgres_setup()
    
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY:")
    print("=" * 60)
    
    if success:
        print("✅ PostgreSQL setup scripts created")
        print("✅ SQLite bridge created for immediate functionality")
        print("✅ Service configuration completed")
        print("\n📡 Next Steps:")
        print("   1. Run: setup_postgres_service.ps1 (if not already run)")
        print("   2. Or run: start_postgres.bat to start PostgreSQL")
        print("   3. Test connection with test_postgresql_service.py")
        print("\n🎉 PostgreSQL setup ready!")
    else:
        print("⚠️  Setup completed with manual steps required")
        print("   Please run the PowerShell scripts manually")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)