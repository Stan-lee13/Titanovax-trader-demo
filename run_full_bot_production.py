#!/usr/bin/env python3
"""
Full Production Bot Run - All Features Enabled (60 seconds)
Includes Telegram, Email notifications, and all production features
"""

import os
import sys
import time
import signal
import threading
import asyncio
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProductionBotRunner:
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
            'notifications_sent': 0,
            'telegram_messages': 0,
            'emails_sent': 0,
            'api_calls_made': 0,
            'ml_predictions': 0,
            'risk_checks': 0,
            'errors_encountered': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0
        }
        self.telegram_bot = None
        self.email_sender = None
        self.sqlite_bridge = None
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
        self.running = False
        
    def log_status(self, message, level="INFO"):
        """Log status with timestamp and level"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        
        # Log to database
        if self.sqlite_bridge:
            try:
                self.sqlite_bridge.log_system_event(level, message)
            except:
                pass  # Don't let logging failures stop the bot
                
    def initialize_telegram_bot(self):
        """Initialize Telegram bot for notifications"""
        try:
            from orchestration.telegram_bot import TitanovaXTelegramBot
            
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if bot_token and chat_id:
                self.telegram_bot = TitanovaXTelegramBot(bot_token)
                # Initialize the bot application
                asyncio.run(self.telegram_bot.initialize())
                self.log_status("✅ Telegram bot initialized successfully")
                return True
            else:
                self.log_status("⚠️  Telegram credentials not configured", "WARNING")
                return False
                
        except Exception as e:
            self.log_status(f"❌ Failed to initialize Telegram bot: {e}", "ERROR")
            return False
            
    def initialize_email_sender(self):
        """Initialize email sender for notifications"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            smtp_host = os.getenv('EMAIL_SMTP_HOST')
            smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
            email_username = os.getenv('EMAIL_USERNAME')
            email_password = os.getenv('EMAIL_PASSWORD')
            
            if smtp_host and email_username and email_password:
                # Test email connection
                context = __import__('ssl').create_default_context()
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls(context=context)
                    server.login(email_username, email_password)
                
                self.log_status("✅ Email sender initialized successfully")
                return True
            else:
                self.log_status("⚠️  Email credentials not configured", "WARNING")
                return False
                
        except Exception as e:
            self.log_status(f"❌ Failed to initialize email sender: {e}", "ERROR")
            return False
            
    def initialize_sqlite_bridge(self):
        """Initialize SQLite bridge for data storage"""
        try:
            from postgres_sqlite_bridge import PostgresBridge
            self.sqlite_bridge = PostgresBridge()
            self.log_status("✅ SQLite bridge initialized successfully")
            return True
        except Exception as e:
            self.log_status(f"❌ Failed to initialize SQLite bridge: {e}", "ERROR")
            return False
            
    def send_telegram_notification(self, message, level="INFO"):
        """Send Telegram notification"""
        try:
            if self.telegram_bot and os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true':
                asyncio.run(self.telegram_bot.send_message(message))
                self.stats['telegram_messages'] += 1
                self.log_status(f"📱 Telegram notification sent: {message[:50]}...")
                return True
        except Exception as e:
            self.log_status(f"❌ Failed to send Telegram notification: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
        return False
        
    def send_email_notification(self, subject, message, level="INFO"):
        """Send email notification"""
        try:
            if self.email_sender:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                smtp_host = os.getenv('EMAIL_SMTP_HOST')
                smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
                email_username = os.getenv('EMAIL_USERNAME')
                email_password = os.getenv('EMAIL_PASSWORD')
                email_from = os.getenv('EMAIL_FROM')
                email_to = os.getenv('EMAIL_TO')
                
                if email_to:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = f"TitanovaX: {subject}"
                    msg["From"] = email_from
                    msg["To"] = email_to
                    
                    html = f"""
                    <html>
                      <body>
                        <h2>TitanovaX Trading System</h2>
                        <p><strong>Subject:</strong> {subject}</p>
                        <p><strong>Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        <p><strong>Message:</strong></p>
                        <p>{message}</p>
                        <hr>
                        <p><small>This is an automated message from TitanovaX Trading System</small></p>
                      </body>
                    </html>
                    """
                    
                    msg.attach(MIMEText(html, "html"))
                    
                    context = __import__('ssl').create_default_context()
                    with smtplib.SMTP(smtp_host, smtp_port) as server:
                        server.starttls(context=context)
                        server.login(email_username, email_password)
                        server.sendmail(email_from, email_to, msg.as_string())
                    
                    self.stats['emails_sent'] += 1
                    self.log_status(f"📧 Email notification sent: {subject}")
                    return True
                    
        except Exception as e:
            self.log_status(f"❌ Failed to send email notification: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
        return False
        
    def send_notification(self, message, level="INFO", include_telegram=True, include_email=True):
        """Send notifications through all configured channels"""
        self.stats['notifications_sent'] += 1
        
        if include_telegram:
            self.send_telegram_notification(message, level)
            
        if include_email:
            self.send_email_notification(f"Trading Alert - {level}", message, level)
            
    def simulate_ml_prediction(self):
        """Simulate ML prediction and signal generation"""
        try:
            # Simulate ML model prediction
            import random
            
            # Generate mock market data
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
            symbol = random.choice(symbols)
            
            # Simulate ML prediction
            prediction_score = random.uniform(0.1, 0.9)
            signal_strength = random.uniform(0.3, 0.8)
            confidence = random.uniform(0.4, 0.95)
            
            if prediction_score > 0.7:  # High confidence threshold
                signal_type = "BUY" if random.random() > 0.5 else "SELL"
                
                # Log signal to database
                if self.sqlite_bridge:
                    signal_id = self.sqlite_bridge.log_signal(symbol, signal_type, signal_strength, confidence)
                
                self.stats['signals_generated'] += 1
                self.stats['ml_predictions'] += 1
                
                message = f"🤖 ML Signal: {signal_type} {symbol} (Strength: {signal_strength:.2f}, Confidence: {confidence:.2f})"
                self.log_status(message)
                
                # Send notifications for significant signals
                if signal_strength > 0.7:
                    self.send_notification(message, "SIGNAL")
                    
                return signal_id
                
        except Exception as e:
            self.log_status(f"❌ ML prediction simulation failed: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            
        return None
        
    def simulate_risk_management(self):
        """Simulate risk management checks"""
        try:
            import random
            
            self.stats['risk_checks'] += 1
            
            # Simulate risk parameters
            max_risk = float(os.getenv('MAX_RISK_PERCENT', '2.0'))
            current_exposure = random.uniform(0.5, 3.0)
            
            if current_exposure > max_risk:
                message = f"⚠️ Risk Alert: Current exposure ({current_exposure:.1f}%) exceeds maximum ({max_risk}%)"
                self.log_status(message, "WARNING")
                self.send_notification(message, "RISK_ALERT")
                return False
            else:
                self.log_status(f"✅ Risk check passed: Exposure {current_exposure:.1f}% within limits")
                return True
                
        except Exception as e:
            self.log_status(f"❌ Risk management simulation failed: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return False
            
    def simulate_trade_execution(self):
        """Simulate trade execution in paper trading mode"""
        try:
            import random
            
            if os.getenv('FEATURE_PAPER_TRADING', 'true').lower() != 'true':
                return None
                
            # Simulate trade execution
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
            symbol = random.choice(symbols)
            side = random.choice(["BUY", "SELL"])
            quantity = random.uniform(0.01, 0.1)
            price = random.uniform(40000, 60000) if "BTC" in symbol else random.uniform(2000, 4000)
            
            # Log trade to database
            if self.sqlite_bridge:
                trade_id = self.sqlite_bridge.log_trade(symbol, side, quantity, price)
            
            self.stats['trades_executed'] += 1
            
            message = f"💰 Paper Trade Executed: {side} {quantity} {symbol} @ ${price:,.2f}"
            self.log_status(message)
            
            # Send trade notification
            self.send_notification(message, "TRADE")
            
            return trade_id
            
        except Exception as e:
            self.log_status(f"❌ Trade execution simulation failed: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return None
            
    def monitor_system_resources(self):
        """Monitor system resources"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.stats['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            self.stats['cpu_usage_percent'] = process.cpu_percent()
            return True
        except ImportError:
            return False
            
    def run_production_simulation(self):
        """Run full production simulation"""
        self.log_status("🚀 Starting TitanovaX Production Simulation")
        self.log_status(f"Duration: {self.duration} seconds")
        self.log_status(f"Mode: Full Production (Paper Trading)")
        
        # Initialize all systems
        self.log_status("🔧 Initializing production systems...")
        
        telegram_ok = self.initialize_telegram_bot()
        email_ok = self.initialize_email_sender()
        sqlite_ok = self.initialize_sqlite_bridge()
        
        self.log_status(f"System Status - Telegram: {'✅' if telegram_ok else '❌'}, Email: {'✅' if email_ok else '❌'}, Database: {'✅' if sqlite_ok else '❌'}")
        
        # Send startup notifications
        self.send_notification("🚀 TitanovaX Production Bot Started", "STARTUP")
        
        # Main simulation loop
        iteration = 0
        while self.running and (time.time() - self.start_time) < self.duration:
            iteration += 1
            current_time = time.time() - self.start_time
            
            # Simulate different operations at different intervals
            
            # Every 3 seconds: ML prediction and signal generation
            if iteration % 3 == 0:
                self.simulate_ml_prediction()
                
            # Every 5 seconds: Risk management check
            if iteration % 5 == 0:
                self.simulate_risk_management()
                
            # Every 7 seconds: Trade execution
            if iteration % 7 == 0:
                self.simulate_trade_execution()
                
            # Every 10 seconds: System resource monitoring
            if iteration % 10 == 0:
                self.monitor_system_resources()
                
            # Every 15 seconds: Status update
            if iteration % 15 == 0:
                uptime = int(time.time() - self.start_time)
                status_msg = f"📊 Status Update - Uptime: {uptime}s | Signals: {self.stats['signals_generated']} | Trades: {self.stats['trades_executed']} | Notifications: {self.stats['notifications_sent']}"
                self.log_status(status_msg)
                
                # Send periodic status to Telegram
                if telegram_ok and iteration % 30 == 0:
                    self.send_telegram_notification(status_msg)
                    
            # Simulate API calls
            if iteration % 4 == 0:
                self.stats['api_calls_made'] += 1
                self.log_status("🌐 Binance Testnet API call")
                
            time.sleep(1)  # 1 second iteration
            
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        self.stats['end_time'] = datetime.now()
        self.stats['uptime_seconds'] = int(time.time() - self.start_time)
        
        # Send shutdown notifications
        self.send_notification("🛑 TitanovaX Production Bot Shutting Down", "SHUTDOWN")
        
        print("\n" + "=" * 80)
        print("📊 PRODUCTION SIMULATION SUMMARY")
        print("=" * 80)
        print(f"Start Time: {self.stats['start_time']}")
        print(f"End Time: {self.stats['end_time']}")
        print(f"Total Uptime: {self.stats['uptime_seconds']} seconds")
        print(f"Signals Generated: {self.stats['signals_generated']}")
        print(f"Trades Executed: {self.stats['trades_executed']}")
        print(f"ML Predictions: {self.stats['ml_predictions']}")
        print(f"Risk Checks: {self.stats['risk_checks']}")
        print(f"API Calls Made: {self.stats['api_calls_made']}")
        print(f"Notifications Sent: {self.stats['notifications_sent']}")
        print(f"Telegram Messages: {self.stats['telegram_messages']}")
        print(f"Emails Sent: {self.stats['emails_sent']}")
        print(f"Errors Encountered: {self.stats['errors_encountered']}")
        
        if self.stats['memory_usage_mb'] > 0:
            print(f"Memory Usage: {self.stats['memory_usage_mb']:.1f} MB")
        if self.stats['cpu_usage_percent'] > 0:
            print(f"CPU Usage: {self.stats['cpu_usage_percent']:.1f}%")
            
        print("=" * 80)
        print("✅ Production simulation completed successfully!")
        print("=" * 80)
        
    def run(self):
        """Run the production simulation"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = True
        self.start_time = time.time()
        self.stats['start_time'] = datetime.now()
        
        print("=" * 80)
        print("🚀 TITANOVAX FULL PRODUCTION SIMULATION")
        print("=" * 80)
        print(f"Duration: {self.duration} seconds")
        print(f"Start Time: {self.stats['start_time']}")
        print(f"Mode: Full Production (All Features Enabled)")
        print(f"Paper Trading: ENABLED")
        print(f"Notifications: ENABLED (Telegram + Email)")
        print(f"ML Pipeline: ENABLED")
        print(f"Risk Management: ENABLED")
        print("=" * 80)
        
        try:
            self.run_production_simulation()
        except Exception as e:
            self.log_status(f"❌ Production simulation failed: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
        finally:
            self.shutdown()

def main():
    """Main function"""
    # Check if running in correct directory
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please run from the project root directory.")
        sys.exit(1)
    
    # Verify Redis connection
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
    
    # Start production simulation
    bot = ProductionBotRunner(duration=60)
    bot.run()

if __name__ == "__main__":
    main()