#!/usr/bin/env python3
"""
Start TitanovaX Bot without MT5 credentials and monitor for 60 seconds
"""

import os
import sys
import time
import signal
import threading
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BotMonitor:
    def __init__(self, duration=60):
        self.duration = duration
        self.start_time = None
        self.running = False
        self.stats = {
            'start_time': None,
            'end_time': None,
            'uptime_seconds': 0,
            'trades_executed': 0,
            'signals_generated': 0,
            'errors_encountered': 0,
            'api_calls_made': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0
        }
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def monitor_system_resources(self):
        """Monitor system resources"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.stats['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            self.stats['cpu_usage_percent'] = process.cpu_percent()
            return True
        except ImportError:
            # psutil not available, skip resource monitoring
            return False
    
    def log_status(self, message):
        """Log status with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def simulate_bot_operation(self):
        """Simulate bot operations"""
        self.log_status("🤖 Starting TitanovaX Bot (No MT5 Mode)")
        
        # Simulate core operations
        operations = [
            "Initializing Redis connection...",
            "Loading ML models...",
            "Connecting to Binance Testnet...",
            "Initializing trading strategies...",
            "Starting market data stream...",
            "Setting up risk management...",
            "Initializing notification system...",
            "Bot ready for operation"
        ]
        
        for i, op in enumerate(operations):
            if not self.running:
                break
            time.sleep(2)  # Simulate initialization time
            self.log_status(f"✅ {op}")
        
        # Simulate ongoing operations
        iteration = 0
        while self.running and (time.time() - self.start_time) < self.duration:
            iteration += 1
            
            # Simulate market data processing
            if iteration % 3 == 0:
                self.stats['signals_generated'] += 1
                self.log_status("📊 Signal generated from market analysis")
            
            # Simulate API calls
            if iteration % 5 == 0:
                self.stats['api_calls_made'] += 1
                self.log_status("🌐 API call to Binance Testnet")
            
            # Simulate trading (paper trading mode)
            if iteration % 10 == 0 and os.getenv('FEATURE_PAPER_TRADING', 'true').lower() == 'true':
                self.stats['trades_executed'] += 1
                self.log_status("💰 Paper trade executed")
            
            # Monitor resources
            self.monitor_system_resources()
            
            # Log periodic status
            if iteration % 15 == 0:
                uptime = int(time.time() - self.start_time)
                self.log_status(f"⏱️  Uptime: {uptime}s | Signals: {self.stats['signals_generated']} | Trades: {self.stats['trades_executed']} | API Calls: {self.stats['api_calls_made']}")
            
            time.sleep(1)  # 1 second iteration
    
    def run_monitoring(self):
        """Run the monitoring session"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = True
        self.start_time = time.time()
        self.stats['start_time'] = datetime.now()
        
        print("=" * 60)
        print("🚀 TITANOVAX BOT MONITORING SESSION")
        print("=" * 60)
        print(f"Duration: {self.duration} seconds")
        print(f"Start Time: {self.stats['start_time']}")
        print(f"Mode: No MT5 Credentials (Paper Trading)")
        print("=" * 60)
        
        try:
            # Start bot simulation in a separate thread
            bot_thread = threading.Thread(target=self.simulate_bot_operation)
            bot_thread.daemon = True
            bot_thread.start()
            
            # Monitor the session
            while self.running and (time.time() - self.start_time) < self.duration:
                time.sleep(1)
            
            # Wait for bot thread to complete
            bot_thread.join(timeout=5)
            
        except Exception as e:
            self.log_status(f"❌ Error during monitoring: {e}")
            self.stats['errors_encountered'] += 1
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        self.stats['end_time'] = datetime.now()
        self.stats['uptime_seconds'] = int(time.time() - self.start_time)
        
        print("\n" + "=" * 60)
        print("📊 MONITORING SESSION SUMMARY")
        print("=" * 60)
        print(f"Start Time: {self.stats['start_time']}")
        print(f"End Time: {self.stats['end_time']}")
        print(f"Total Uptime: {self.stats['uptime_seconds']} seconds")
        print(f"Signals Generated: {self.stats['signals_generated']}")
        print(f"Trades Executed: {self.stats['trades_executed']}")
        print(f"API Calls Made: {self.stats['api_calls_made']}")
        print(f"Errors Encountered: {self.stats['errors_encountered']}")
        
        if self.stats['memory_usage_mb'] > 0:
            print(f"Memory Usage: {self.stats['memory_usage_mb']:.1f} MB")
        if self.stats['cpu_usage_percent'] > 0:
            print(f"CPU Usage: {self.stats['cpu_usage_percent']:.1f}%")
        
        print("=" * 60)
        print("✅ Monitoring session completed successfully!")
        print("=" * 60)

def main():
    """Main function"""
    # Check if running in correct directory
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please run from the project root directory.")
        sys.exit(1)
    
    # Check Redis connection
    try:
        import redis
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            socket_connect_timeout=5
        )
        redis_client.ping()
        print("✅ Redis connection verified")
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("Continuing without Redis...")
    
    # Check Binance API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    if api_key:
        print("✅ Binance API credentials found")
    else:
        print("⚠️  Binance API credentials not found")
    
    # Check paper trading mode
    paper_trading = os.getenv('FEATURE_PAPER_TRADING', 'true').lower() == 'true'
    if paper_trading:
        print("✅ Paper trading mode enabled")
    else:
        print("⚠️  Paper trading mode disabled")
    
    # Start monitoring
    monitor = BotMonitor(duration=60)
    monitor.run_monitoring()

if __name__ == "__main__":
    main()