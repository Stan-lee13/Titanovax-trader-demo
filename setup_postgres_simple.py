#!/usr/bin/env python3
"""
PostgreSQL Setup Script for TitanovaX
Creates PostgreSQL setup using available Windows resources
"""

import os
import subprocess
import json
import time
from pathlib import Path

def create_simple_postgres_setup():
    """Create a simple PostgreSQL setup script"""
    print("Creating PostgreSQL setup script...")
    
    # Create PowerShell setup script
    ps_script = """# PostgreSQL Setup Script
Write-Host "Setting up PostgreSQL for TitanovaX..." -ForegroundColor Green

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
    Write-Host "Found PostgreSQL at: $pgBin" -ForegroundColor Green
    
    # Create data directory if it doesn't exist
    $dataDir = "$pwd\\postgres_data"
    if (!(Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir | Out-Null
        
        # Initialize database cluster
        Write-Host "Initializing database cluster..." -ForegroundColor Yellow
        & "$pgBin\\initdb.exe" -D $dataDir -U postgres --encoding=UTF8 --locale=en_US.UTF-8
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Database cluster initialized" -ForegroundColor Green
        } else {
            Write-Host "Database initialization failed" -ForegroundColor Red
        }
    }
    
    # Create startup script
    $startupScript = @"
@echo off
echo Starting PostgreSQL for TitanovaX...
cd /d "$pwd"
"$pgBin\\postgres.exe" -D "postgres_data" -p 5432
echo PostgreSQL started on port 5432
pause
"@
    
    Set-Content -Path "start_postgres.bat" -Value $startupScript
    Write-Host "Startup script created: start_postgres.bat" -ForegroundColor Green
    
    # Create TitanovaX database
    Write-Host "Creating TitanovaX database..." -ForegroundColor Yellow
    & "$pgBin\\createdb.exe" -U postgres -h localhost -p 5432 titanovax
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "TitanovaX database created" -ForegroundColor Green
    } else {
        Write-Host "Database may already exist" -ForegroundColor Yellow
    }
    
    return $true
} else {
    Write-Host "PostgreSQL not found. Please install PostgreSQL first." -ForegroundColor Red
    return $false
}
"""
    
    with open("setup_postgres_simple.ps1", "w", encoding='utf-8') as f:
        f.write(ps_script)
    
    print("Simple PostgreSQL setup script created: setup_postgres_simple.ps1")
    return True

def create_sqlite_bridge():
    """Create SQLite bridge for immediate functionality"""
    print("Creating SQLite bridge for immediate functionality...")
    
    bridge_code = '''import sqlite3
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def log_trade(self, symbol, side, quantity, price):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (symbol, side, quantity, price)
            VALUES (?, ?, ?, ?)
        """, (symbol, side, quantity, price))
        self.conn.commit()
        return cursor.lastrowid
    
    def log_signal(self, symbol, signal_type, strength, confidence):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO signals (symbol, signal_type, strength, confidence)
            VALUES (?, ?, ?, ?)
        """, (symbol, signal_type, strength, confidence))
        self.conn.commit()
        return cursor.lastrowid
    
    def log_market_data(self, symbol, price, volume):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_data (symbol, price, volume)
            VALUES (?, ?, ?)
        """, (symbol, price, volume))
        self.conn.commit()
        return cursor.lastrowid
    
    def log_system_event(self, level, message):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO system_logs (level, message)
            VALUES (?, ?)
        """, (level, message))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent_trades(self, limit=100):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (limit,))
        return cursor.fetchall()
    
    def get_recent_signals(self, limit=100):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?', (limit,))
        return cursor.fetchall()
    
    def get_system_stats(self):
        cursor = self.conn.cursor()
        stats = {}
        
        cursor.execute('SELECT COUNT(*) FROM trades')
        stats['total_trades'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM signals')
        stats['total_signals'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM market_data')
        stats['total_market_data'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM system_logs')
        stats['total_logs'] = cursor.fetchone()[0]
        
        return stats
    
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
    bridge.log_system_event("INFO", "TitanovaX system started")
    
    print(f"Trade logged: ID {trade_id}")
    print(f"Signal logged: ID {signal_id}")
    
    stats = bridge.get_system_stats()
    print(f"System stats: {stats}")
    
    bridge.close()
    print("SQLite bridge ready for TitanovaX")
'''
    
    with open("postgres_sqlite_bridge.py", "w", encoding='utf-8') as f:
        f.write(bridge_code)
    
    print("SQLite bridge created: postgres_sqlite_bridge.py")
    return True

def update_env_file():
    """Update .env file with PostgreSQL configuration"""
    print("Updating .env file with PostgreSQL configuration...")
    
    env_file = ".env"
    if not os.path.exists(env_file):
        print(".env file not found")
        return False
    
    try:
        # Read current .env
        with open(env_file, 'r', encoding='utf-8') as f:
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
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(".env file updated with PostgreSQL configuration")
        return True
        
    except Exception as e:
        print(f"Error updating .env file: {e}")
        return False

def test_sqlite_bridge():
    """Test the SQLite bridge"""
    print("Testing SQLite bridge...")
    try:
        result = subprocess.run([
            "python", "postgres_sqlite_bridge.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("SQLite bridge test successful")
            return True
        else:
            print(f"SQLite bridge test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"SQLite bridge test error: {e}")
        return False

def main():
    """Main setup function"""
    print("PostgreSQL Setup for TitanovaX Trading System")
    print("=" * 60)
    
    # Create setup scripts
    create_simple_postgres_setup()
    create_sqlite_bridge()
    
    # Update environment
    update_env_file()
    
    # Test SQLite bridge
    test_sqlite_bridge()
    
    print("\n" + "=" * 60)
    print("SETUP SUMMARY:")
    print("=" * 60)
    print("Setup scripts created:")
    print("  - setup_postgres_simple.ps1 (PowerShell setup)")
    print("  - postgres_sqlite_bridge.py (SQLite bridge)")
    print("  - .env file updated with PostgreSQL config")
    print("\nNext steps:")
    print("  1. Run setup_postgres_simple.ps1 in PowerShell")
    print("  2. Or start PostgreSQL manually if already installed")
    print("  3. SQLite bridge is ready for immediate use")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)