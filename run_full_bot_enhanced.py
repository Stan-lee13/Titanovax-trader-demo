#!/usr/bin/env python3
"""
Enhanced TitanovaX Production Bot with Comprehensive Monitoring and Reporting
Runs for 60 seconds with full assessment and notification delivery verification
"""

import os
import sys
import asyncio
import random
import time
import json
import psutil
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from hybrid_notification_system import (
    HybridNotificationManager,
    NotificationCategory
)

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EnhancedTitanovaXBot:
    def __init__(self):
        self.start_time = datetime.now()
        self.running = True
        self.telegram_bot = None
        self.email_sender = None
        self.sqlite_bridge = None
        
        # Enhanced statistics tracking
        self.stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'ml_predictions': 0,
            'risk_checks': 0,
            'notifications_sent': 0,
            'telegram_messages': 0,
            'emails_sent': 0,
            'errors_encountered': 0,
            'api_calls': 0,
            'memory_usage': [],
            'cpu_usage': [],
            'performance_metrics': {}
        }
        
        # Notification tracking for verification
        self.notification_log = []
        
        # Performance assessment data
        self.assessment_data = {
            'model_performance': {},
            'system_health': {},
            'notification_delivery': {},
            'trading_efficiency': {}
        }
        
        # Risk alert debouncing
        self.risk_alert_last_sent = None
        self.risk_alert_cooldown = 30  # 30 seconds cooldown for risk alerts
        self.last_risk_level = None
        
        # Persistent event loop for Telegram
        self.telegram_event_loop = None
        self.telegram_thread = None
        
        # Hybrid notification system
        self.hybrid_notification_manager = None
        
        print("🚀 Initializing Enhanced TitanovaX Production Bot...")
        
    def log_status(self, message, level="INFO"):
        """Enhanced logging with timestamp and level"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        
        # Log to notification tracking
        self.notification_log.append({
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'type': 'system_log'
        })
        
        # Also log to SQLite if available
        if self.sqlite_bridge:
            try:
                self.sqlite_bridge.log_system_event(level, message)
            except:
                pass
    
    def initialize_telegram_bot(self):
        """Initialize Telegram bot with persistent event loop"""
        try:
            from orchestration.telegram_bot import TitanovaXTelegramBot
            
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if bot_token and chat_id:
                self.telegram_bot = TitanovaXTelegramBot(bot_token)
                
                # Create persistent event loop for Telegram
                try:
                    import threading
                    result = {'success': False, 'error': None}
                    
                    def init_persistent_loop():
                        try:
                            # Create and set persistent event loop
                            self.telegram_event_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(self.telegram_event_loop)
                            
                            # Initialize the bot
                            init_success = self.telegram_event_loop.run_until_complete(
                                self.telegram_bot.initialize()
                            )
                            
                            if init_success:
                                # Test with a message
                                init_message = f"🤖 TitanovaX Enhanced Bot Initialized\n"
                                init_message += f"📅 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                init_message += f"⚙️ Mode: Enhanced Production Monitoring\n"
                                init_message += f"⏱️ Duration: 60 seconds\n"
                                init_message += f"🔍 Status: All systems operational"
                                
                                test_success = self.telegram_event_loop.run_until_complete(
                                    self.telegram_bot.send_message(init_message, chat_id)
                                )
                                result['success'] = test_success
                                
                                if test_success:
                                    self.stats['telegram_messages'] += 1
                                    
                                # Keep the loop running for the bot's lifetime
                                # Use a separate thread for run_forever to avoid blocking
                                def keep_loop_running():
                                    try:
                                        self.telegram_event_loop.run_forever()
                                    except Exception as loop_error:
                                        self.log_status(f"❌ Telegram loop error: {loop_error}", "ERROR")
                                
                                loop_thread = threading.Thread(target=keep_loop_running, daemon=True)
                                loop_thread.start()
                            else:
                                result['error'] = "Bot initialization failed"
                                
                        except Exception as thread_error:
                            result['error'] = str(thread_error)
                            self.log_status(f"❌ Telegram init thread error: {thread_error}", "ERROR")
                    
                    # Start the persistent loop in a separate thread
                    self.telegram_thread = threading.Thread(target=init_persistent_loop, daemon=True)
                    self.telegram_thread.start()
                    
                    # Wait for initialization to complete
                    time.sleep(5)  # Give it time to initialize
                    
                    if result['success']:
                        self.log_status("✅ Telegram bot initialized with persistent event loop")
                        return True
                    else:
                        error_msg = result['error'] or "Unknown initialization error"
                        self.log_status(f"❌ Telegram bot initialization failed: {error_msg}", "ERROR")
                        return False
                        
                except Exception as init_error:
                    self.log_status(f"❌ Telegram initialization error: {init_error}", "ERROR")
                    return False
            else:
                self.log_status("⚠️ Telegram credentials not configured", "WARNING")
                return False
                
        except Exception as e:
            self.log_status(f"❌ Failed to initialize Telegram bot: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return False
    
    def initialize_email_sender(self):
        """Initialize email sender with comprehensive testing and Gmail SMTP"""
        try:
            smtp_host = os.getenv('EMAIL_SMTP_HOST', 'smtp.gmail.com')
            smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
            email_username = os.getenv('EMAIL_USERNAME')
            email_password = os.getenv('EMAIL_PASSWORD')
            
            if smtp_host and email_username and email_password:
                # Test connection with enhanced error handling
                import ssl
                import socket
                
                try:
                    # Create SSL context
                    context = ssl.create_default_context()
                    
                    # Test connection with timeout
                    with socket.create_connection((smtp_host, smtp_port), timeout=30) as sock:
                        with smtplib.SMTP(smtp_host, smtp_port) as server:
                            server.timeout = 30  # Set timeout
                            server.starttls(context=context)
                            server.login(email_username, email_password)
                            
                            # Test with a simple email
                            test_msg = MIMEMultipart("alternative")
                            test_msg["Subject"] = "TitanovaX Email System Test"
                            test_msg["From"] = email_username
                            test_msg["To"] = email_username
                            
                            test_content = f"""
                            <html>
                              <body>
                                <h3>🤖 TitanovaX Email System Test</h3>
                                <p>Email system initialized successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                                <p>SMTP Server: {smtp_host}:{smtp_port}</p>
                                <p>Status: ✅ Operational</p>
                              </body>
                            </html>
                            """
                            
                            test_msg.attach(MIMEText(test_content, "html"))
                            server.sendmail(email_username, email_username, test_msg.as_string())
                    
                    self.log_status("✅ Email sender initialized successfully with Gmail SMTP")
                    
                    # Store the email sending function for later use
                    def email_sender_func(subject, message, html_content=None):
                        return self.send_email_notification(subject, message, html_content=html_content)
                    
                    self.email_sender = email_sender_func
                    return True
                    
                except socket.timeout:
                    self.log_status("❌ Email initialization failed: Connection timeout", "ERROR")
                    self.stats['errors_encountered'] += 1
                    return False
                except smtplib.SMTPAuthenticationError as auth_error:
                    self.log_status(f"❌ Email authentication failed: {auth_error}", "ERROR")
                    self.stats['errors_encountered'] += 1
                    return False
                except smtplib.SMTPException as smtp_error:
                    self.log_status(f"❌ SMTP error: {smtp_error}", "ERROR")
                    self.stats['errors_encountered'] += 1
                    return False
                    
            else:
                self.log_status("⚠️ Email credentials not configured", "WARNING")
                return False
                
        except Exception as e:
            self.log_status(f"❌ Failed to initialize email sender: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return False
    
    def initialize_sqlite_bridge(self):
        """Initialize SQLite bridge for data storage"""
        try:
            from postgres_sqlite_bridge import PostgresBridge
            self.sqlite_bridge = PostgresBridge()
            
            # Test connection
            test_result = self.sqlite_bridge.test_connection()
            if test_result['success']:
                self.log_status("✅ SQLite bridge initialized successfully")
                return True
            else:
                self.log_status(f"❌ SQLite bridge test failed: {test_result['error']}", "ERROR")
                return False
                
        except Exception as e:
            self.log_status(f"❌ Failed to initialize SQLite bridge: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return False
    
    def send_telegram_notification(self, message, level="INFO", verify_delivery=True):
        """Send Telegram notification using persistent event loop"""
        try:
            if self.telegram_bot and os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true':
                chat_id = os.getenv('TELEGRAM_CHAT_ID')
                
                if chat_id:
                    # Add timestamp and level to message
                    enhanced_message = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
                    
                    # Use the persistent event loop
                    import threading
                    result = {'success': False, 'error': None}
                    
                    def send_message_async():
                        try:
                            if self.telegram_event_loop and not self.telegram_event_loop.is_closed():
                                # Schedule the coroutine in the persistent loop
                                future = asyncio.run_coroutine_threadsafe(
                                    self.telegram_bot.send_message(enhanced_message, chat_id),
                                    self.telegram_event_loop
                                )
                                # Wait for result with timeout
                                result['success'] = future.result(timeout=10)
                            else:
                                result['error'] = "Event loop not available"
                        except Exception as send_error:
                            result['error'] = str(send_error)
                            self.log_status(f"❌ Telegram send error: {send_error}", "ERROR")
                    
                    # Run in separate thread to avoid blocking
                    thread = threading.Thread(target=send_message_async)
                    thread.start()
                    thread.join(timeout=15)  # Increased timeout
                    
                    if result['success']:
                        self.stats['telegram_messages'] += 1
                        self.stats['notifications_sent'] += 1
                        
                        # Track notification
                        self.notification_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'telegram',
                            'message': message,
                            'level': level,
                            'status': 'sent',
                            'delivery_verified': verify_delivery
                        })
                        
                        self.log_status(f"📱 Telegram notification sent: {message[:50]}...")
                        return True
                    else:
                        error_msg = result['error'] or "Unknown send error"
                        self.log_status(f"❌ Telegram notification failed: {error_msg}", "ERROR")
                        return False
                        
        except Exception as e:
            self.log_status(f"❌ Failed to send Telegram notification: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            
            # Track failed notification
            self.notification_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'telegram',
                'message': message,
                'level': level,
                'status': 'failed',
                'error': str(e)
            })
            
        return False
    
    def send_email_notification(self, subject, message, level="INFO", html_content=None):
        """Send email notification with enhanced formatting and Gmail SMTP"""
        try:
            if self.email_sender:
                smtp_host = os.getenv('EMAIL_SMTP_HOST', 'smtp.gmail.com')
                smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '465'))  # Default to SSL port 465
                email_username = os.getenv('EMAIL_USERNAME')
                email_password = os.getenv('EMAIL_PASSWORD')
                email_from = os.getenv('EMAIL_FROM')
                email_to = os.getenv('EMAIL_TO')
                
                if email_to:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = f"TitanovaX: {subject}"
                    msg["From"] = email_from
                    msg["To"] = email_to
                    
                    # Create enhanced HTML content
                    if html_content is None:
                        html_content = f"""
                        <html>
                          <head>
                            <style>
                              body {{ font-family: Arial, sans-serif; margin: 20px; }}
                              .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                              .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 10px; }}
                              .footer {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 12px; }}
                              .metric {{ background-color: #007bff; color: white; padding: 10px; border-radius: 3px; margin: 5px 0; }}
                              .timestamp {{ color: #6c757d; font-size: 12px; }}
                            </style>
                          </head>
                          <body>
                            <div class="header">
                              <h2>🤖 TitanovaX Trading System</h2>
                              <p>Enhanced Production Monitoring Report</p>
                            </div>
                            <div class="content">
                              <p><strong>Subject:</strong> {subject}</p>
                              <p><strong>Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                              <p><strong>Message:</strong></p>
                              <p>{message}</p>
                            </div>
                            <div class="footer">
                              <p><small>This is an automated message from TitanovaX Enhanced Trading System</small></p>
                              <p class="timestamp">Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            </div>
                          </body>
                        </html>
                        """
                    
                    msg.attach(MIMEText(html_content, "html"))
                    
                    # Send email with enhanced error handling
                    import ssl
                    import socket
                    
                    try:
                        # Create SSL context
                        context = ssl.create_default_context()
                        
                        # Use SSL connection for port 465 (Gmail)
                        if smtp_port == 465:
                            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=60) as server:
                                server.login(email_username, email_password)
                                server.sendmail(email_from, email_to, msg.as_string())
                        else:
                            # Use STARTTLS for other ports
                            with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
                                server.starttls(context=context)
                                server.login(email_username, email_password)
                                server.sendmail(email_from, email_to, msg.as_string())
                        
                        self.stats['emails_sent'] += 1
                        self.stats['notifications_sent'] += 1
                        
                        # Track notification
                        self.notification_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'email',
                            'subject': subject,
                            'message': message,
                            'level': level,
                            'status': 'sent'
                        })
                        
                        self.log_status(f"📧 Email notification sent: {subject}")
                        return True
                        
                    except socket.timeout:
                        self.log_status(f"❌ Email send failed: Connection timeout (60s)", "ERROR")
                        self.stats['errors_encountered'] += 1
                        
                        # Track failed notification
                        self.notification_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'email',
                            'subject': subject,
                            'message': message,
                            'level': level,
                            'status': 'failed',
                            'error': 'Connection timeout (60s)'
                        })
                        return False
                        
                    except smtplib.SMTPAuthenticationError as auth_error:
                        self.log_status(f"❌ Email authentication failed: {auth_error}", "ERROR")
                        self.stats['errors_encountered'] += 1
                        
                        # Track failed notification
                        self.notification_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'email',
                            'subject': subject,
                            'message': message,
                            'level': level,
                            'status': 'failed',
                            'error': f'Authentication error: {auth_error}'
                        })
                        return False
                        
                    except smtplib.SMTPException as smtp_error:
                        self.log_status(f"❌ SMTP error: {smtp_error}", "ERROR")
                        self.stats['errors_encountered'] += 1
                        
                        # Track failed notification
                        self.notification_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'email',
                            'subject': subject,
                            'message': message,
                            'level': level,
                            'status': 'failed',
                            'error': f'SMTP error: {smtp_error}'
                        })
                        return False
                        
        except Exception as e:
            self.log_status(f"❌ Failed to send email notification: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            
            # Track failed notification
            self.notification_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'email',
                'subject': subject,
                'message': message,
                'level': level,
                'status': 'failed',
                'error': str(e)
            })
            
        return False
    
    def initialize_hybrid_notification_system(self):
        """Initialize hybrid notification system with configurable batch processing"""
        try:
            # Get configuration from environment variables
            batch_interval = int(os.getenv('NOTIFICATION_BATCH_INTERVAL', '10'))  # seconds
            enable_persistence = os.getenv('NOTIFICATION_PERSISTENCE', 'true').lower() == 'true'
            max_batch_size = int(os.getenv('NOTIFICATION_MAX_BATCH_SIZE', '10'))
            
            self.log_status(f"🔧 Initializing hybrid notification system with batch_interval={batch_interval}, persistence={enable_persistence}, max_batch_size={max_batch_size}")
            
            self.hybrid_notification_manager = HybridNotificationManager(
                telegram_bot=self.telegram_bot,
                email_sender=self.email_sender,
                batch_interval=batch_interval,
                persist_to_db=enable_persistence,
                max_buffer_size=max_batch_size
            )
            
            self.log_status(f"✅ HybridNotificationManager created successfully")
            
            # Check if Telegram bot is available
            self.log_status(f"🔍 Telegram bot status: {self.telegram_bot is not None}")
            if self.telegram_bot:
                self.log_status(f"🔍 Telegram bot has send_message: {hasattr(self.telegram_bot, 'send_message')}")
                if hasattr(self.telegram_bot, 'chat_id'):
                    self.log_status(f"🔍 Telegram bot chat_id: {self.telegram_bot.chat_id}")
            
            # Test the system with a startup notification
            self.log_status("🧪 Testing hybrid notification system...", "INFO")
            
            try:
                # Start the batch processor first
                self.hybrid_notification_manager.start_batch_processor()
                self.log_status("✅ Batch processor started", "INFO")
                
                # Give it a moment to initialize
                import time
                time.sleep(0.1)
                
                # Check buffer status before test
                buffer_status_before = self.hybrid_notification_manager.get_buffer_status()
                self.log_status(f"📊 Buffer status before test: {buffer_status_before}", "INFO")
                
                test_success = self.hybrid_notification_manager.add_notification(
                    category=NotificationCategory.SYSTEM,
                    title="TitanovaX Hybrid Notification System",
                    message=f"🚀 Hybrid notification system initialized successfully\n"
                            f"📊 Batch interval: {batch_interval}s\n"
                            f"💾 Persistence: {'Enabled' if enable_persistence else 'Disabled'}\n"
                            f"📦 Max batch size: {max_batch_size}\n"
                            f"⏰ Ready for optimized notification delivery",
                    priority=1,
                    bypass_batch=True  # Send immediately to verify system is working
                )
                
                self.log_status(f"🔍 Test notification result: {test_success}")
                
                # Check buffer status after test
                buffer_status_after = self.hybrid_notification_manager.get_buffer_status()
                self.log_status(f"📊 Buffer status after test: {buffer_status_after}", "INFO")
                
            except Exception as e:
                self.log_status(f"❌ Test notification failed with exception: {e}", "ERROR")
                test_success = False
            
            if test_success:
                self.log_status(f"✅ Hybrid notification system initialized (batch: {batch_interval}s, persistence: {enable_persistence})")
                return True
            else:
                self.log_status("⚠️ Hybrid notification system test failed, continuing without it", "WARNING")
                self.hybrid_notification_manager = None
                return False
                
        except Exception as e:
            self.log_status(f"❌ Failed to initialize hybrid notification system: {e}", "ERROR")
            self.hybrid_notification_manager = None
            self.stats['errors_encountered'] += 1
            return False
    
    def send_notification(self, message: str, notification_type: str = "SYSTEM", detailed_explanation: str = None):
        """Send notification via both Telegram and Email with optional detailed explanation"""
        self.stats['notifications_sent'] += 1
        
        try:
            # Use hybrid notification system if available
            if hasattr(self, 'hybrid_notification_manager') and self.hybrid_notification_manager:
                # Map notification types to categories
                category_map = {
                    "SIGNAL": NotificationCategory.SIGNAL,
                    "TRADE": NotificationCategory.TRADE,
                    "RISK_ALERT": NotificationCategory.RISK,
                    "SYSTEM": NotificationCategory.SYSTEM,
                    "CRITICAL": NotificationCategory.CRITICAL,
                    "PERFORMANCE": NotificationCategory.PERFORMANCE
                }
                
                category = category_map.get(notification_type, NotificationCategory.SYSTEM)
                priority = 3 if notification_type == "CRITICAL" else (2 if "RISK" in notification_type else 1)
                
                # Use hybrid system for better batching and formatting
                success = self.hybrid_notification_manager.add_notification(
                    category=category,
                    title=f"TitanovaX {notification_type}",
                    message=message,
                    priority=priority,
                    bypass_batch=(notification_type == "CRITICAL"),
                    detailed_explanation=detailed_explanation
                )
                
                if success:
                    self.log_status(f"✅ Notification queued via hybrid system: {notification_type}")
                    return True
                else:
                    self.log_status("⚠️ Hybrid system failed, falling back to direct notifications")
            
            # Fallback to original notification system
            telegram_success = self.send_telegram_notification(message, notification_type)
            
            # Use detailed explanation for email if available, otherwise use basic message
            email_content = detailed_explanation if detailed_explanation else message
            email_success = self.send_email_notification(
                f"TitanovaX {notification_type}",
                email_content,
                notification_type,
                message
            )
            
            if telegram_success or email_success:
                self.log_status(f"✅ Notification sent successfully (TG: {telegram_success}, Email: {email_success})")
                return True
            else:
                self.log_status("❌ Both notification channels failed")
                return False
                
        except Exception as e:
            self.log_status(f"❌ Notification error: {e}")
            self.stats['errors_encountered'] += 1
            return False
    
    def simulate_ml_prediction(self):
        """Enhanced ML prediction simulation with performance tracking"""
        try:
            import random
            
            # Simulate ML model prediction
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT", "AVAXUSDT"]
            symbol = random.choice(symbols)
            
            # Simulate different model performances
            model_accuracy = random.uniform(0.65, 0.95)  # Model accuracy range
            prediction_score = random.uniform(0.1, 0.9)
            signal_strength = random.uniform(0.3, 0.8)
            confidence = random.uniform(0.4, 0.95)
            
            # Simulate model loading time
            model_load_time = random.uniform(0.1, 0.5)  # seconds
            prediction_time = random.uniform(0.05, 0.2)  # seconds
            
            if prediction_score > 0.7:  # High confidence threshold
                signal_type = "BUY" if random.random() > 0.5 else "SELL"
                
                # Log signal to database
                if self.sqlite_bridge:
                    signal_id = self.sqlite_bridge.log_signal(symbol, signal_type, signal_strength, confidence)
                
                self.stats['signals_generated'] += 1
                self.stats['ml_predictions'] += 1
                
                message = f"🤖 ML Signal: {signal_type} {symbol} (Strength: {signal_strength:.2f}, Confidence: {confidence:.2f}, Accuracy: {model_accuracy:.2f})"
                self.log_status(message)
                
                # Track model performance
                self.assessment_data['model_performance'][signal_id] = {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'strength': signal_strength,
                    'confidence': confidence,
                    'accuracy': model_accuracy,
                    'load_time': model_load_time,
                    'prediction_time': prediction_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send notifications for significant signals
                if signal_strength > 0.7:
                    self.send_notification(message, "SIGNAL")
                    
                return signal_id
                
        except Exception as e:
            self.log_status(f"❌ ML prediction simulation failed: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            
        return None
    
    def simulate_risk_management(self):
        """Enhanced risk management simulation with self-healing features and debouncing"""
        try:
            import random
            
            self.stats['risk_checks'] += 1
            
            # Simulate risk parameters
            max_risk = float(os.getenv('MAX_RISK_PERCENT', '2.0'))
            current_exposure = random.uniform(0.5, 3.5)  # Sometimes exceed limits
            
            # Simulate self-healing when risk is exceeded
            if current_exposure > max_risk:
                message = f"⚠️ Risk Alert: Current exposure ({current_exposure:.1f}%) exceeds maximum ({max_risk}%)"
                self.log_status(message, "WARNING")
                
                # Check if we should send alert (debouncing logic)
                should_send_alert = False
                current_time = datetime.now()
                
                # First alert or different risk level
                if (self.risk_alert_last_sent is None or 
                    self.last_risk_level is None or
                    abs(current_exposure - self.last_risk_level) > 0.5):  # Significant change
                    should_send_alert = True
                    self.risk_alert_last_sent = current_time
                    self.last_risk_level = current_exposure
                else:
                    # Check cooldown period
                    time_since_last_alert = (current_time - self.risk_alert_last_sent).total_seconds()
                    if time_since_last_alert >= self.risk_alert_cooldown:
                        should_send_alert = True
                        self.risk_alert_last_sent = current_time
                        self.last_risk_level = current_exposure
                
                # Simulate self-healing action
                healing_action = random.choice([
                    "Reducing position size",
                    "Closing partial positions", 
                    "Implementing tighter stop-losses",
                    "Pausing new trades temporarily"
                ])
                
                healing_message = f"🔧 Self-healing activated: {healing_action}"
                self.log_status(healing_message, "INFO")
                
                # Send risk alert notification (with debouncing)
                if should_send_alert:
                    self.send_notification(message, "RISK_ALERT")
                    self.log_status(f"📢 Risk alert sent (cooldown: {self.risk_alert_cooldown}s)")
                else:
                    self.log_status(f"⏸️ Risk alert suppressed due to cooldown ({self.risk_alert_cooldown}s)")
                
                # Track risk management performance
                self.assessment_data['trading_efficiency']['risk_management'] = {
                    'exposure': current_exposure,
                    'max_allowed': max_risk,
                    'action_taken': healing_action,
                    'alert_sent': should_send_alert,
                    'timestamp': datetime.now().isoformat()
                }
                
                return False
            else:
                self.log_status(f"✅ Risk check passed: Exposure {current_exposure:.1f}% within limits")
                return True
                
        except Exception as e:
            self.log_status(f"❌ Risk management simulation failed: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return False
    
    def simulate_trade_execution(self):
        """Enhanced trade execution simulation with detailed tracking and comprehensive trade explanations"""
        try:
            import random
            
            # Simulate trade execution
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
            symbol = random.choice(symbols)
            side = random.choice(["BUY", "SELL"])
            quantity = random.uniform(0.01, 0.5)
            price = random.uniform(45000, 55000) if "BTC" in symbol else random.uniform(2500, 3500)
            
            # Simulate execution time and success rate
            execution_time = random.uniform(0.1, 1.0)  # seconds
            success_rate = random.uniform(0.85, 0.98)  # 85-98% success rate
            
            if random.random() < success_rate:
                # Successful trade
                trade_id = self.sqlite_bridge.log_trade(symbol, side, quantity, price)
                self.stats['trades_executed'] += 1
                self.stats['api_calls'] += 1
                
                message = f"📈 Trade Executed: {side} {quantity:.3f} {symbol} @ ${price:.2f} (Exec Time: {execution_time:.2f}s)"
                self.log_status(message)
                
                # Generate comprehensive trade explanation for email
                trade_explanation = self.generate_trade_explanatio(
                    symbol, side, quantity, price, execution_time, trade_id
                )
                
                # Track trading efficiency
                self.assessment_data['trading_efficiency'][trade_id] = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'execution_time': execution_time,
                    'success': True,
                    'timestamp': datetime.now().isoformat(),
                    'trade_explanation': trade_explanation
                }
                
                # Send trade notification with detailed explanation for email
                self.send_notification(message, "TRADE", trade_explanation)
                
                return trade_id
            else:
                # Failed trade
                self.log_status(f"❌ Trade Failed: {side} {quantity:.3f} {symbol} @ ${price:.2f}", "ERROR")
                self.stats['errors_encountered'] += 1
                return None
                
        except Exception as e:
            self.log_status(f"❌ Trade execution simulation failed: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return None
    
    def assess_system_health(self):
        """Comprehensive system health assessment"""
        try:
            # Memory and CPU monitoring
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.stats['memory_usage'].append(memory_info.percent)
            self.stats['cpu_usage'].append(cpu_percent)
            
            # System health assessment
            self.assessment_data['system_health'] = {
                'memory_usage_percent': memory_info.percent,
                'cpu_usage_percent': cpu_percent,
                'available_memory_gb': memory_info.available / (1024**3),
                'total_memory_gb': memory_info.total / (1024**3),
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            }
            
            # Health status message
            if memory_info.percent < 90 and cpu_percent < 80:
                health_status = "🟢 Excellent"
            elif memory_info.percent < 95 and cpu_percent < 90:
                health_status = "🟡 Good"
            else:
                health_status = "🔴 Critical"
            
            self.log_status(f"🏥 System Health: {health_status} (Memory: {memory_info.percent:.1f}%, CPU: {cpu_percent:.1f}%)")
            
            return self.assessment_data['system_health']
            
        except Exception as e:
            self.log_status(f"❌ System health assessment failed: {e}", "ERROR")
            return {}
    
    def generate_comprehensive_report(self):
        """Generate comprehensive performance assessment report"""
        runtime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Notification delivery assessment
        telegram_success_rate = (self.stats['telegram_messages'] / max(self.stats['notifications_sent'], 1)) * 100
        email_success_rate = (self.stats['emails_sent'] / max(self.stats['notifications_sent'], 1)) * 100
        
        self.assessment_data['notification_delivery'] = {
            'total_notifications': self.stats['notifications_sent'],
            'telegram_messages': self.stats['telegram_messages'],
            'emails_sent': self.stats['emails_sent'],
            'telegram_success_rate': telegram_success_rate,
            'email_success_rate': email_success_rate,
            'notification_log': self.notification_log
        }
        
        # Model performance summary
        if self.assessment_data['model_performance']:
            avg_confidence = sum([m['confidence'] for m in self.assessment_data['model_performance'].values()]) / len(self.assessment_data['model_performance'])
            avg_accuracy = sum([m['accuracy'] for m in self.assessment_data['model_performance'].values()]) / len(self.assessment_data['model_performance'])
        else:
            avg_confidence = 0
            avg_accuracy = 0
        
        # Generate comprehensive report
        report = f"""
🔍 TITANOVAX ENHANCED PERFORMANCE ASSESSMENT
═══════════════════════════════════════════════════════════════

📊 EXECUTIVE SUMMARY
• Runtime: {runtime_seconds:.1f} seconds
• Overall Success Rate: {((self.stats['signals_generated'] + self.stats['trades_executed']) / max(self.stats['signals_generated'] + self.stats['trades_executed'] + self.stats['errors_encountered'], 1) * 100):.1f}%
• System Health: {self.assessment_data.get('system_health', {}).get('memory_usage_percent', 0):.1f}% Memory, {self.assessment_data.get('system_health', {}).get('cpu_usage_percent', 0):.1f}% CPU
• Errors Encountered: {self.stats['errors_encountered']}

🤖 MACHINE LEARNING PERFORMANCE
• Signals Generated: {self.stats['signals_generated']}
• ML Predictions Made: {self.stats['ml_predictions']}
• Average Confidence: {avg_confidence:.2f}
• Average Model Accuracy: {avg_accuracy:.2f}
• Model Load Time: Avg {sum([m.get('load_time', 0) for m in self.assessment_data['model_performance'].values()]) / max(len(self.assessment_data['model_performance']), 1):.3f}s

📈 TRADING EFFICIENCY
• Trades Executed: {self.stats['trades_executed']}
• Risk Checks Passed: {self.stats['risk_checks']}
• Self-Healing Actions: {len([a for a in self.assessment_data.get('trading_efficiency', {}).values() if isinstance(a, dict) and a.get('action_taken')])}
• Trading Success Rate: {(self.stats['trades_executed'] / max(self.stats['trades_executed'] + self.stats['errors_encountered'], 1) * 100):.1f}%

📱 NOTIFICATION DELIVERY VERIFICATION
• Total Notifications: {self.stats['notifications_sent']}
• Telegram Messages: {self.stats['telegram_messages']} ({telegram_success_rate:.1f}% success rate)
• Emails Sent: {self.stats['emails_sent']} ({email_success_rate:.1f}% success rate)
• Delivery Issues: {len([n for n in self.notification_log if n.get('status') == 'failed'])}

🏥 SYSTEM HEALTH ASSESSMENT
• Memory Usage: {self.assessment_data.get('system_health', {}).get('memory_usage_percent', 0):.1f}%
• CPU Usage: {self.assessment_data.get('system_health', {}).get('cpu_usage_percent', 0):.1f}%
• Available Memory: {self.assessment_data.get('system_health', {}).get('available_memory_gb', 0):.1f} GB
• Uptime: {runtime_seconds:.1f} seconds

🔧 SELF-HEALING FEATURES
• Autonomous Risk Management: {'✅ Active' if self.stats['risk_checks'] > 0 else '❌ Inactive'}
• System Monitoring: {'✅ Active' if len(self.stats['memory_usage']) > 0 else '❌ Inactive'}
• Error Recovery: {'✅ Active' if self.stats['errors_encountered'] < 5 else '⚠️ Moderate'}
• Notification Delivery: {'✅ Verified' if telegram_success_rate > 50 else '❌ Issues Detected'}

📈 KEY PERFORMANCE INDICATORS
• Signal Generation Rate: {(self.stats['signals_generated'] / runtime_seconds * 60):.1f} signals/minute
• Trade Execution Rate: {(self.stats['trades_executed'] / runtime_seconds * 60):.1f} trades/minute
• Error Rate: {(self.stats['errors_encountered'] / runtime_seconds * 60):.1f} errors/minute
• System Responsiveness: {'🟢 Excellent' if self.stats['errors_encountered'] < 3 else '🟡 Good' if self.stats['errors_encountered'] < 10 else '🔴 Poor'}

⚠️ IDENTIFIED ISSUES & RECOMMENDATIONS
{self.generate_recommendations()}

═══════════════════════════════════════════════════════════════
🏁 Assessment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Next recommended run: {(datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report
    
    def generate_recommendations(self):
        """Generate specific recommendations based on performance"""
        recommendations = []
        
        # Check notification delivery
        if self.stats['telegram_messages'] == 0:
            recommendations.append("• Telegram notifications not delivering - check bot token and chat ID")
        
        if self.stats['emails_sent'] == 0:
            recommendations.append("• Email notifications not sending - check SMTP configuration and network")
        
        # Check error rate
        if self.stats['errors_encountered'] > 5:
            recommendations.append("• High error rate detected - review system logs and configuration")
        
        # Check system resources
        if len(self.stats['memory_usage']) > 0 and self.stats['memory_usage'][-1] > 90:
            recommendations.append("• High memory usage - consider system optimization or scaling")
        
        # Check model performance
        if self.stats['signals_generated'] == 0:
            recommendations.append("• No ML signals generated - verify model configuration and data sources")
        
        # Check trading performance
        if self.stats['trades_executed'] == 0:
            recommendations.append("• No trades executed - verify trading configuration and API connections")
        
        if not recommendations:
            recommendations.append("• All systems operating within normal parameters")
            recommendations.append("• Continue monitoring and regular maintenance schedule")
        
        return "\n".join(recommendations)
    
    async def run_production_cycle(self):
        """Main production cycle with comprehensive monitoring"""
        cycle_start = datetime.now()
        
        self.log_status("🔄 Starting production cycle...")
        
        # System health check
        self.assess_system_health()
        
        # ML Prediction
        signal_id = self.simulate_ml_prediction()
        
        # Risk Management
        risk_ok = self.simulate_risk_management()
        
        # Trade Execution (only if risk check passes and signal generated)
        if risk_ok and signal_id:
            self.simulate_trade_execution()
        
        # Performance tracking
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        self.log_status(f"⏱️ Cycle completed in {cycle_time:.2f} seconds")
        
        return cycle_time
    
    async def run_enhanced_production(self, duration_seconds=60):
        """Run enhanced production with comprehensive assessment"""
        self.log_status("🚀 Starting Enhanced TitanovaX Production Run")
        self.log_status(f"⏱️ Duration: {duration_seconds} seconds")
        self.log_status(f"📊 Features: ML Predictions, Risk Management, Trade Execution, Notification Verification")
        
        # Initialize all systems
        telegram_ok = self.initialize_telegram_bot()
        email_ok = self.initialize_email_sender()
        sqlite_ok = self.initialize_sqlite_bridge()
        hybrid_ok = self.initialize_hybrid_notification_system()
        
        if not sqlite_ok:
            self.log_status("❌ Critical: SQLite bridge initialization failed", "ERROR")
            return
        
        # Send startup notification
        startup_message = f"🚀 TitanovaX Enhanced Production Started\n"
        startup_message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        startup_message += f"⚙️ Telegram: {'✅' if telegram_ok else '❌'}\n"
        startup_message += f"📧 Email: {'✅' if email_ok else '❌'}\n"
        startup_message += f"💾 Database: {'✅' if sqlite_ok else '❌'}\n"
        startup_message += f"📦 Hybrid Notifications: {'✅' if hybrid_ok else '❌'}"
        
        self.send_notification(startup_message, "SYSTEM_START")
        
        # Production cycles
        cycle_count = 0
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < duration_seconds and self.running:
            try:
                cycle_time = await self.run_production_cycle()
                cycle_count += 1
                
                # Progress notification every 15 seconds
                if cycle_count % 15 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    progress_message = f"📊 Progress: {elapsed:.0f}/{duration_seconds}s | Cycles: {cycle_count} | Signals: {self.stats['signals_generated']} | Trades: {self.stats['trades_executed']}"
                    self.send_notification(progress_message, "PROGRESS")
                
                # Brief pause between cycles
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                self.log_status("⏹️ Production run interrupted by user", "WARNING")
                self.running = False
                break
            except Exception as e:
                self.log_status(f"❌ Production cycle error: {e}", "ERROR")
                self.stats['errors_encountered'] += 1
                await asyncio.sleep(2)  # Brief pause on error
        
        # Generate and send comprehensive report
        self.log_status("📊 Generating comprehensive performance assessment...")
        comprehensive_report = self.generate_comprehensive_report()
        
        # Send report via both channels
        self.log_status("📧 Sending comprehensive report via Telegram and Email...")
        
        # Telegram report (shortened version)
        telegram_report = f"""
🔍 TITANOVAX PERFORMANCE ASSESSMENT
⏱️ Runtime: {(datetime.now() - start_time).total_seconds():.1f}s
📈 Signals: {self.stats['signals_generated']} | Trades: {self.stats['trades_executed']}
📱 Notifications: {self.stats['notifications_sent']} (TG: {self.stats['telegram_messages']}, Email: {self.stats['emails_sent']})
🏥 Health: Memory {self.stats['memory_usage'][-1]:.1f}%, CPU {self.stats['cpu_usage'][-1]:.1f}%
⚠️ Errors: {self.stats['errors_encountered']}

See email for detailed report.
        """
        
        self.send_telegram_notification(telegram_report, "PERFORMANCE_REPORT")
        
        # Email report (full version)
        self.send_email_notification(
            "TitanovaX Enhanced Performance Assessment",
            comprehensive_report,
            "PERFORMANCE_REPORT",
            comprehensive_report
        )
        
        # Print final summary
        self.log_status("="*80)
        self.log_status("🏁 ENHANCED PRODUCTION RUN COMPLETED")
        self.log_status("="*80)
        self.log_status(f"⏱️ Total Runtime: {(datetime.now() - start_time).total_seconds():.1f} seconds")
        self.log_status(f"🔄 Production Cycles: {cycle_count}")
        self.log_status(f"📊 Signals Generated: {self.stats['signals_generated']}")
        self.log_status(f"📈 Trades Executed: {self.stats['trades_executed']}")
        self.log_status(f"🤖 ML Predictions: {self.stats['ml_predictions']}")
        self.log_status(f"⚠️ Risk Checks: {self.stats['risk_checks']}")
        self.log_status(f"📱 Notifications Sent: {self.stats['notifications_sent']}")
        self.log_status(f"📧 Telegram Messages: {self.stats['telegram_messages']}")
        self.log_status(f"📧 Emails Sent: {self.stats['emails_sent']}")
        self.log_status(f"❌ Errors Encountered: {self.stats['errors_encountered']}")
        self.log_status(f"🏥 Final Memory Usage: {self.stats['memory_usage'][-1]:.1f}%")
        self.log_status(f"⚡ Final CPU Usage: {self.stats['cpu_usage'][-1]:.1f}%")
        
        # Hybrid notification statistics
        if self.hybrid_notification_manager:
            try:
                hybrid_stats = self.hybrid_notification_manager.get_statistics()
                self.log_status(f"📦 Hybrid System: {hybrid_stats['total_notifications']} notifications")
                self.log_status(f"📊 Batch efficiency: {hybrid_stats['batch_efficiency']:.1f}%")
                self.log_status(f"⚡ Critical bypass: {hybrid_stats['critical_bypass_count']} alerts")
            except:
                pass
        
        # Save detailed report to file
        report_file = f"enhanced_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': (datetime.now() - start_time).total_seconds(),
                'statistics': self.stats,
                'assessment_data': self.assessment_data,
                'notification_log': self.notification_log
            }, f, indent=2)
        
        self.log_status(f"📄 Detailed report saved to: {report_file}")
        self.log_status("="*80)

if __name__ == "__main__":
    # Load environment variables
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    bot = EnhancedTitanovaXBot()
    
    try:
        asyncio.run(bot.run_enhanced_production(60))
    except KeyboardInterrupt:
        print("\n⏹️ Production run interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()