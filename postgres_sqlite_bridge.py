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
    
    def test_connection(self):
        """Test database connection and basic functionality"""
        try:
            # Test connection
            if not self.conn:
                return {'success': False, 'error': 'No database connection'}
            
            # Test basic query
            cursor = self.conn.cursor()
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            
            if result and result[0] == 1:
                return {'success': True, 'error': None}
            else:
                return {'success': False, 'error': 'Database query test failed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
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
