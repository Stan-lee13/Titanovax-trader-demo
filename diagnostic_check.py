#!/usr/bin/env python3
"""
TitanovaX Comprehensive Diagnostic Check
Tests all systems before production run
"""

import os
import sys
import asyncio
import smtplib
import sqlite3
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class TitanovaXDiagnostic:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
    def log_result(self, test_name, status, details="", error=None):
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }
        self.results['details'].append(result)
        
        if status == "PASS":
            self.results['tests_passed'] += 1
            print(f"✅ {test_name}: {details}")
        else:
            self.results['tests_failed'] += 1
            print(f"❌ {test_name}: {details}")
            if error:
                print(f"   Error: {error}")
    
    def test_environment_variables(self):
        """Test all required environment variables"""
        # Load environment variables from .env file if it exists
        env_file = '.env'
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        required_vars = [
            'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'TELEGRAM_CHANNEL_ID',
            'EMAIL_SMTP_HOST', 'EMAIL_SMTP_PORT', 'EMAIL_USERNAME', 
            'EMAIL_PASSWORD', 'EMAIL_TO', 'EMAIL_FROM'
        ]
        
        optional_vars = [
            'MAX_RISK_PERCENT', 'TELEGRAM_ENABLED', 'REDIS_HOST', 'REDIS_PORT'
        ]
        
        missing_required = []
        configured_optional = []
        configured_values = {}
        
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.strip() == '':
                missing_required.append(var)
            else:
                # Mask sensitive data
                if 'TOKEN' in var or 'PASSWORD' in var:
                    configured_values[var] = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "..."
                else:
                    configured_values[var] = value
                
        for var in optional_vars:
            if os.getenv(var):
                configured_optional.append(var)
        
        if missing_required:
            self.log_result("Environment Variables", "FAIL", 
                          f"Missing required: {missing_required}")
        else:
            self.log_result("Environment Variables", "PASS", 
                          f"All required configured. Key values: {configured_values}")
    
    def test_telegram_bot(self):
        """Test Telegram bot connection and messaging"""
        try:
            from orchestration.telegram_bot import TitanovaXTelegramBot
            
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                self.log_result("Telegram Bot", "FAIL", "Missing credentials")
                return
            
            # Test bot initialization
            bot = TitanovaXTelegramBot(bot_token)
            success = asyncio.run(bot.initialize())
            
            if success:
                # Test sending a message
                test_message = f"🔍 TitanovaX Diagnostic Test - {datetime.now().strftime('%H:%M:%S')}"
                message_sent = asyncio.run(bot.send_message(test_message, chat_id))
                
                if message_sent:
                    self.log_result("Telegram Bot", "PASS", 
                                  "Bot initialized and test message sent successfully")
                else:
                    self.log_result("Telegram Bot", "FAIL", 
                                  "Bot initialized but message failed to send")
            else:
                self.log_result("Telegram Bot", "FAIL", "Bot initialization failed")
                
        except Exception as e:
            self.log_result("Telegram Bot", "FAIL", f"Exception: {str(e)}", e)
    
    def test_email_system(self):
        """Test email SMTP connection and sending"""
        try:
            smtp_host = os.getenv('EMAIL_SMTP_HOST')
            smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
            email_username = os.getenv('EMAIL_USERNAME')
            email_password = os.getenv('EMAIL_PASSWORD')
            email_from = os.getenv('EMAIL_FROM')
            email_to = os.getenv('EMAIL_TO')
            
            if not all([smtp_host, email_username, email_password, email_to]):
                self.log_result("Email System", "FAIL", "Missing email credentials")
                return
            
            # Test SMTP connection
            import ssl
            context = ssl.create_default_context()
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls(context=context)
                server.login(email_username, email_password)
                
                # Send test email
                msg = MIMEMultipart("alternative")
                msg["Subject"] = "TitanovaX Diagnostic Test"
                msg["From"] = email_from
                msg["To"] = email_to
                
                html = f"""
                <html>
                  <body>
                    <h2>🔍 TitanovaX Diagnostic Test</h2>
                    <p>This is a test email from the TitanovaX diagnostic system.</p>
                    <p><strong>Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>If you received this email, your email notification system is working correctly.</p>
                  </body>
                </html>
                """
                
                msg.attach(MIMEText(html, "html"))
                server.sendmail(email_from, email_to, msg.as_string())
            
            self.log_result("Email System", "PASS", "SMTP connection successful and test email sent")
            
        except Exception as e:
            self.log_result("Email System", "FAIL", f"Email test failed: {str(e)}", e)
    
    def test_sqlite_bridge(self):
        """Test SQLite database and bridge functionality"""
        try:
            from postgres_sqlite_bridge import PostgresBridge
            
            bridge = PostgresBridge()
            
            # Test database connection
            test_result = bridge.test_connection()
            if test_result['success']:
                # Test logging functionality
                signal_id = bridge.log_signal("BTCUSDT", "BUY", 0.75, 0.85)
                
                if signal_id:
                    self.log_result("SQLite Bridge", "PASS", 
                                  f"Connection successful, signal logged with ID: {signal_id}")
                else:
                    self.log_result("SQLite Bridge", "FAIL", "Connection successful but signal logging failed")
            else:
                self.log_result("SQLite Bridge", "FAIL", f"Connection failed: {test_result['error']}")
                
        except Exception as e:
            self.log_result("SQLite Bridge", "FAIL", f"SQLite bridge test failed: {str(e)}", e)
    
    def test_model_loading(self):
        """Test ML model loading and functionality"""
        try:
            # Check if model files exist
            model_files = [
                'models/ml_predictor.pkl',
                'models/signal_generator.pkl',
                'models/risk_manager.pkl'
            ]
            
            existing_models = []
            missing_models = []
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    existing_models.append(model_file)
                else:
                    missing_models.append(model_file)
            
            if existing_models:
                self.log_result("Model Loading", "PASS", 
                              f"Found models: {existing_models}. Missing: {missing_models}")
            else:
                self.log_result("Model Loading", "WARNING", 
                              "No model files found. Will use simulated predictions.")
                
        except Exception as e:
            self.log_result("Model Loading", "FAIL", f"Model test failed: {str(e)}", e)
    
    def test_binance_api(self):
        """Test Binance API connectivity"""
        try:
            import requests
            
            # Test Binance API endpoint
            response = requests.get('https://api.binance.com/api/v3/time')
            
            if response.status_code == 200:
                server_time = response.json()['serverTime']
                self.log_result("Binance API", "PASS", f"API reachable. Server time: {server_time}")
            else:
                self.log_result("Binance API", "FAIL", f"API returned status: {response.status_code}")
                
        except Exception as e:
            self.log_result("Binance API", "FAIL", f"Binance API test failed: {str(e)}", e)
    
    def test_system_resources(self):
        """Test system resources and performance"""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            resource_status = f"Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%, Disk: {disk_percent:.1f}%"
            
            if memory_percent < 90 and cpu_percent < 80 and disk_percent < 90:
                self.log_result("System Resources", "PASS", resource_status)
            else:
                self.log_result("System Resources", "WARNING", 
                              f"High resource usage: {resource_status}")
                
        except Exception as e:
            self.log_result("System Resources", "FAIL", f"Resource test failed: {str(e)}", e)
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "="*60)
        print("📊 TITANOVAX DIAGNOSTIC REPORT")
        print("="*60)
        
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        success_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.results['tests_passed']}")
        print(f"Failed: {self.results['tests_failed']}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Timestamp: {self.results['timestamp']}")
        
        # Categorize results
        critical_failures = []
        warnings = []
        
        for result in self.results['details']:
            if result['status'] == 'FAIL':
                if any(keyword in result['test'].lower() for keyword in ['telegram', 'email', 'model']):
                    critical_failures.append(result)
            elif result['status'] == 'WARNING':
                warnings.append(result)
        
        if critical_failures:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_failures)}):")
            for failure in critical_failures:
                print(f"  • {failure['test']}: {failure['details']}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"  • {warning['test']}: {warning['details']}")
        
        print("\n" + "="*60)
        
        # Save report to file
        report_file = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")
        
        return self.results
    
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print("🔍 Starting TitanovaX Comprehensive Diagnostic...")
        print("="*60)
        
        self.test_environment_variables()
        self.test_telegram_bot()
        self.test_email_system()
        self.test_sqlite_bridge()
        self.test_model_loading()
        self.test_binance_api()
        self.test_system_resources()
        
        return self.generate_report()

if __name__ == "__main__":
    diagnostic = TitanovaXDiagnostic()
    results = diagnostic.run_all_tests()
    
    # Exit with appropriate code
    if results['tests_failed'] == 0:
        print("\n✅ All systems ready for production!")
        sys.exit(0)
    elif results['tests_failed'] <= 2:
        print("\n⚠️  Some issues detected but system can run with limitations.")
        sys.exit(0)
    else:
        print("\n❌ Critical issues detected. Please fix before running production.")
        sys.exit(1)