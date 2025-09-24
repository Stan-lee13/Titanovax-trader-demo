"""
TitanovaX Hybrid Batch Notification System

This module implements a sophisticated notification system that:
1. Buffers notifications for batch processing
2. Sends consolidated notifications at configurable intervals
3. Provides immediate alerts for critical events
4. Supports multiple notification categories with proper formatting
"""

import asyncio
import threading
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationCategory(Enum):
    """Notification categories with priority levels"""
    CRITICAL = "critical"           # System failures, margin calls
    TRADE = "trade"                  # Trade executions
    RISK = "risk"                    # Risk alerts
    SIGNAL = "signal"               # ML signals
    SYSTEM = "system"                # System events
    PERFORMANCE = "performance"     # Performance metrics


@dataclass
class Notification:
    """Individual notification data structure"""
    id: str
    category: NotificationCategory
    title: str
    message: str
    timestamp: datetime
    priority: int = 1  # 1=normal, 2=high, 3=critical
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchNotification:
    """Batch notification data structure"""
    batch_id: str
    start_time: datetime
    end_time: datetime
    notifications: List[Notification]
    category_counts: Dict[str, int]


class NotificationBuffer:
    """
    Thread-safe notification buffer that collects alerts for batch processing
    """
    
    def __init__(self, max_size: int = 1000, persist_to_db: bool = True):
        self.buffer: List[Notification] = []
        self.lock = threading.Lock()
        self.max_size = max_size
        self.persist_to_db = persist_to_db
        self.db_path = "notification_buffer.db"
        
        if self.persist_to_db:
            self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    metadata TEXT,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            conn.commit()
            conn.close()
            logger.info("✅ Notification buffer database initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize notification database: {e}")
            self.persist_to_db = False
    
    def add_notification(self, notification: Notification) -> bool:
        """
        Add a notification to the buffer
        
        Args:
            notification: Notification object to add
            
        Returns:
            bool: True if added successfully, False if buffer is full
        """
        with self.lock:
            if len(self.buffer) >= self.max_size:
                logger.warning(f"⚠️ Notification buffer full ({self.max_size}), dropping oldest")
                self.buffer.pop(0)
            
            self.buffer.append(notification)
            
            if self.persist_to_db:
                self._persist_notification(notification)
            
            logger.debug(f"📥 Added notification: {notification.category.value} - {notification.title}")
            return True
    
    def _persist_notification(self, notification: Notification):
        """Persist notification to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO notifications 
                (id, category, title, message, timestamp, priority, metadata, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                notification.id,
                notification.category.value,
                notification.title,
                notification.message,
                notification.timestamp.isoformat(),
                notification.priority,
                json.dumps(notification.metadata) if notification.metadata else None,
                False
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Failed to persist notification: {e}")
    
    def get_and_clear_buffer(self) -> List[Notification]:
        """
        Get all notifications from buffer and clear it
        
        Returns:
            List[Notification]: List of all buffered notifications
        """
        with self.lock:
            notifications = self.buffer.copy()
            self.buffer.clear()
            logger.info(f"📤 Retrieved {len(notifications)} notifications from buffer")
            return notifications
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def get_unprocessed_from_db(self) -> List[Notification]:
        """Get unprocessed notifications from database"""
        if not self.persist_to_db:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, category, title, message, timestamp, priority, metadata
                FROM notifications 
                WHERE processed = FALSE
                ORDER BY timestamp
            ''')
            
            rows = cursor.fetchall()
            notifications = []
            
            for row in rows:
                notification = Notification(
                    id=row[0],
                    category=NotificationCategory(row[1]),
                    title=row[2],
                    message=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    priority=row[5],
                    metadata=json.loads(row[6]) if row[6] else None
                )
                notifications.append(notification)
            
            conn.close()
            return notifications
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve unprocessed notifications: {e}")
            return []
    
    def mark_as_processed(self, notification_ids: List[str]):
        """Mark notifications as processed in database"""
        if not self.persist_to_db or not notification_ids:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE notifications 
                SET processed = TRUE 
                WHERE id IN ({seq})
            '''.format(seq=','.join(['?' for _ in notification_ids])), notification_ids)
            conn.commit()
            conn.close()
            logger.info(f"✅ Marked {len(notification_ids)} notifications as processed")
        except Exception as e:
            logger.error(f"❌ Failed to mark notifications as processed: {e}")


class NotificationFormatter:
    """
    Formats notifications into beautiful batch messages for Telegram
    """
    
    # Emoji mapping for different categories
    EMOJI_MAP = {
        NotificationCategory.CRITICAL: "🚨",
        NotificationCategory.TRADE: "💰",
        NotificationCategory.RISK: "⚠️",
        NotificationCategory.SIGNAL: "📊",
        NotificationCategory.SYSTEM: "⚙️",
        NotificationCategory.PERFORMANCE: "📈"
    }
    
    # Category display names
    CATEGORY_NAMES = {
        NotificationCategory.CRITICAL: "🚨 CRITICAL ALERTS",
        NotificationCategory.TRADE: "💰 TRADE EXECUTIONS",
        NotificationCategory.RISK: "⚠️ RISK ALERTS",
        NotificationCategory.SIGNAL: "📊 ML SIGNALS",
        NotificationCategory.SYSTEM: "⚙️ SYSTEM EVENTS",
        NotificationCategory.PERFORMANCE: "📈 PERFORMANCE METRICS"
    }
    
    @staticmethod
    def format_batch_notification(batch: BatchNotification) -> str:
        """
        Format a batch of notifications into a consolidated Telegram message
        
        Args:
            batch: BatchNotification object containing all notifications
            
        Returns:
            str: Formatted message ready for Telegram
        """
        # Header with time window
        start_str = batch.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = batch.end_time.strftime("%H:%M:%S")
        
        message_parts = []
        message_parts.append(f"📊 **TITANOVAX BATCH NOTIFICATIONS**")
        message_parts.append(f"⏰ **Time Window**: {start_str} → {end_str}")
        message_parts.append(f"📋 **Total Alerts**: {len(batch.notifications)}")
        message_parts.append("")
        
        # Group notifications by category
        categories = {}
        for notification in batch.notifications:
            if notification.category not in categories:
                categories[notification.category] = []
            categories[notification.category].append(notification)
        
        # Format each category section
        for category, notifications in categories.items():
            if not notifications:
                continue
                
            # Category header
            category_name = NotificationFormatter.CATEGORY_NAMES.get(category, f"{NotificationFormatter.EMOJI_MAP.get(category, '🔹')} {category.value.upper()}")
            message_parts.append(f"**{category_name}** ({len(notifications)})")
            
            # Format notifications in this category
            for i, notification in enumerate(notifications, 1):
                timestamp = notification.timestamp.strftime("%H:%M:%S")
                
                # Format based on category
                if category == NotificationCategory.TRADE:
                    message_parts.append(f"{i}. {notification.title}")
                    message_parts.append(f"   💬 {notification.message}")
                    
                elif category == NotificationCategory.RISK:
                    message_parts.append(f"{i}. {notification.title}")
                    message_parts.append(f"   📊 {notification.message}")
                    
                elif category == NotificationCategory.SIGNAL:
                    message_parts.append(f"{i}. {notification.title}")
                    message_parts.append(f"   📈 {notification.message}")
                    
                elif category == NotificationCategory.CRITICAL:
                    message_parts.append(f"🚨 {notification.title}")
                    message_parts.append(f"   ⚠️ {notification.message}")
                    
                else:
                    message_parts.append(f"{i}. {notification.title}")
                    message_parts.append(f"   📝 {notification.message}")
                
                message_parts.append(f"   ⏰ {timestamp}")
                message_parts.append("")
        
        # Footer
        message_parts.append("---")
        message_parts.append("🤖 *TitanovaX Trading System*")
        
        return "\n".join(message_parts)
    
    @staticmethod
    def format_single_notification(notification: Notification) -> str:
        """
        Format a single notification for immediate delivery
        
        Args:
            notification: Single notification object
            
        Returns:
            str: Formatted message for immediate delivery
        """
        emoji = NotificationFormatter.EMOJI_MAP.get(notification.category, "🔹")
        timestamp = notification.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""
{emoji} **{notification.title}**
⏰ {timestamp}

{notification.message}

🤖 *TitanovaX Trading System*
""".strip()


class HybridNotificationManager:
    """
    Main manager for hybrid batch notification system
    """
    
    def __init__(self, 
                 telegram_bot=None,
                 email_sender=None,
                 batch_interval: int = 10,
                 persist_to_db: bool = True,
                 max_buffer_size: int = 1000,
                 enable_email: bool = True,
                 email_batch_threshold: int = 3):
        """
        Initialize the hybrid notification manager
        
        Args:
            telegram_bot: Telegram bot instance for sending messages
            email_sender: Email sender function for sending emails
            batch_interval: Interval in seconds for batch processing (default: 10)
            persist_to_db: Whether to persist notifications to SQLite
            max_buffer_size: Maximum number of notifications to buffer
            enable_email: Whether to enable email notifications
            email_batch_threshold: Minimum number of notifications to trigger email batch
        """
        self.telegram_bot = telegram_bot
        self.email_sender = email_sender
        self.batch_interval = batch_interval
        self.buffer = NotificationBuffer(max_size=max_buffer_size, persist_to_db=persist_to_db)
        self.formatter = NotificationFormatter()
        self.running = False
        self.batch_thread = None
        self.last_batch_time = datetime.now()
        self.enable_email = enable_email
        self.email_batch_threshold = email_batch_threshold
        
        logger.info(f"🚀 Hybrid Notification Manager initialized (batch interval: {batch_interval}s, email: {enable_email})")
    
    def start_batch_processor(self):
        """Start the batch processing thread"""
        if self.running:
            logger.warning("⚠️ Batch processor already running")
            return
        
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processor_loop, daemon=True)
        self.batch_thread.start()
        logger.info("✅ Batch notification processor started")
    
    def stop_batch_processor(self):
        """Stop the batch processing thread"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5)
        logger.info("🛑 Batch notification processor stopped")
    
    def _batch_processor_loop(self):
        """Main loop for batch processing"""
        while self.running:
            try:
                # Check if it's time to process a batch
                now = datetime.now()
                if (now - self.last_batch_time).total_seconds() >= self.batch_interval:
                    self._process_batch()
                    self.last_batch_time = now
                
                # Sleep for a short interval to avoid busy waiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error in batch processor loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _process_batch(self):
        """Process buffered notifications into a batch"""
        try:
            # Get all notifications from buffer
            notifications = self.buffer.get_and_clear_buffer()
            
            if not notifications:
                logger.debug("📭 No notifications to process in batch")
                return
            
            # Create batch notification
            batch = BatchNotification(
                batch_id=f"batch_{int(time.time())}",
                start_time=self.last_batch_time,
                end_time=datetime.now(),
                notifications=notifications,
                category_counts=self._count_categories(notifications)
            )
            
            # Format and send batch message
            batch_message = self.formatter.format_batch_notification(batch)
            
            # Send via Telegram
            telegram_success = False
            if self.telegram_bot:
                try:
                    # Use the persistent event loop for sending
                    if hasattr(self.telegram_bot, 'send_message'):
                        self.telegram_bot.send_message(
                            text=batch_message,
                            parse_mode='Markdown'
                        )
                        telegram_success = True
                        logger.info(f"📤 Sent Telegram batch notification with {len(notifications)} messages")
                    else:
                        logger.warning("⚠️ Telegram bot missing send_message method")
                except Exception as e:
                    logger.error(f"❌ Failed to send Telegram batch notification: {e}")
            else:
                logger.warning("⚠️ No Telegram bot available for batch notification")
            
            # Send via Email if enabled and threshold met
            email_success = False
            if self.enable_email and self.email_sender and len(notifications) >= self.email_batch_threshold:
                try:
                    # Create detailed email content
                    email_subject = f"TitanovaX Batch Notification - {len(notifications)} Alerts"
                    
                    # Check if any notification has detailed explanation
                    detailed_explanations = []
                    for notification in notifications:
                        if notification.metadata and 'detailed_explanation' in notification.metadata:
                            detailed_explanations.append(notification.metadata['detailed_explanation'])
                    
                    # Use detailed explanations if available, otherwise use standard batch message
                    if detailed_explanations:
                        # Combine all detailed explanations
                        combined_explanation = "\n\n---\n\n".join(detailed_explanations)
                        email_html = self._create_email_batch_content(batch, combined_explanation)
                        email_content = combined_explanation
                    else:
                        email_html = self._create_email_batch_content(batch)
                        email_content = batch_message
                    
                    # Send email using the email sender function
                    email_success = self.email_sender(email_subject, email_content, html_content=email_html)
                    if email_success:
                        logger.info(f"📧 Sent Email batch notification with {len(notifications)} messages")
                    else:
                        logger.warning("⚠️ Email batch notification failed")
                except Exception as e:
                    logger.error(f"❌ Failed to send Email batch notification: {e}")
            
            # Mark notifications as processed if at least one channel succeeded
            if telegram_success or email_success:
                notification_ids = [n.id for n in notifications]
                self.buffer.mark_as_processed(notification_ids)
            else:
                # Re-add notifications to buffer on complete failure
                logger.warning("⚠️ All notification channels failed, re-adding to buffer")
                for notification in notifications:
                    self.buffer.add_notification(notification)
                
        except Exception as e:
            logger.error(f"❌ Error processing batch: {e}")
    
    def _count_categories(self, notifications: List[Notification]) -> Dict[str, int]:
        """Count notifications by category"""
        counts = {}
        for notification in notifications:
            category = notification.category.value
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def _create_email_batch_content(self, batch: BatchNotification, detailed_explanation: str = None) -> str:
        """Create detailed HTML content for email batch notification"""
        try:
            # Create summary by category
            category_summary = ""
            for category, count in batch.category_counts.items():
                category_summary += f"<li><strong>{category.title()}:</strong> {count} notifications</li>"
            
            # Create detailed notification list
            notification_details = ""
            for notification in batch.notifications:
                priority_class = "high" if notification.priority >= 3 else "medium" if notification.priority >= 2 else "low"
                
                # Check if this notification has detailed explanation
                if notification.metadata and 'detailed_explanation' in notification.metadata:
                    explanation_content = f"""
                    <div class="detailed-explanation">
                        <h5>📊 Detailed Analysis:</h5>
                        <div class="explanation-content">{notification.metadata['detailed_explanation'].replace(chr(10), '<br>')}</div>
                    </div>
                    """
                else:
                    explanation_content = ""
                
                notification_details += f"""
                <div class="notification-item priority-{priority_class}">
                    <h4>{notification.title}</h4>
                    <p><strong>Category:</strong> {notification.category.value.title()}</p>
                    <p><strong>Time:</strong> {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Priority:</strong> {notification.priority}/3</p>
                    <div class="message-content">{notification.message.replace(chr(10), '<br>')}</div>
                    {explanation_content}
                </div>
                """
            
            html_content = f"""
            <html>
              <head>
                <style>
                  body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                  .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                  .summary {{ background-color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                  .notifications {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                  .notification-item {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                  .priority-high {{ border-left: 4px solid #e74c3c; background-color: #fdf2f2; }}
                  .priority-medium {{ border-left: 4px solid #f39c12; background-color: #fffaf0; }}
                  .priority-low {{ border-left: 4px solid #27ae60; background-color: #f0fff4; }}
                  .message-content {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; margin-top: 10px; font-family: monospace; }}
                  .footer {{ background-color: #e9ecef; padding: 15px; border-radius: 8px; margin-top: 20px; font-size: 12px; text-align: center; }}
                  .metric {{ background-color: #007bff; color: white; padding: 10px; border-radius: 3px; margin: 5px 0; display: inline-block; }}
                  ul {{ margin: 10px 0; }}
                  li {{ margin: 5px 0; }}
                </style>
              </head>
              <body>
                <div class="header">
                  <h2>🤖 TitanovaX Trading System</h2>
                  <h3>📧 Batch Notification Report</h3>
                  <p><strong>Batch ID:</strong> {batch.batch_id}</p>
                  <p><strong>Time Range:</strong> {batch.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {batch.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                  <p><strong>Total Notifications:</strong> {len(batch.notifications)}</p>
                </div>
                
                <div class="summary">
                  <h3>📊 Notification Summary</h3>
                  <ul>
                    {category_summary}
                  </ul>
                  <p><span class="metric">Total: {len(batch.notifications)} notifications</span></p>
                </div>
                
                {(f'<div class="overall-explanation"><h3>📈 Comprehensive Trade Analysis</h3><div class="explanation-content">{detailed_explanation.replace(chr(10), "<br>")}</div></div>' if detailed_explanation else '')}
                
                <div class="notifications">
                  <h3>📝 Detailed Notifications</h3>
                  {notification_details}
                </div>
                
                <div class="footer">
                  <p><small>This is an automated batch notification from TitanovaX Enhanced Trading System</small></p>
                  <p><strong>Generated at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
              </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ Error creating email batch content: {e}")
            return f"<html><body><h3>TitanovaX Batch Notification</h3><p>{len(batch.notifications)} notifications processed.</p></body></html>"
    
    def add_notification(self, 
                        category: NotificationCategory,
                        title: str,
                        message: str,
                        priority: int = 1,
                        metadata: Optional[Dict[str, Any]] = None,
                        bypass_batch: bool = False,
                        detailed_explanation: str = None) -> bool:
        """
        Add a notification to the system
        
        Args:
            category: Notification category
            title: Notification title
            message: Notification message
            priority: Priority level (1=normal, 2=high, 3=critical)
            metadata: Optional metadata dictionary
            bypass_batch: Whether to send immediately (for critical alerts)
            detailed_explanation: Optional detailed explanation for email notifications
            
        Returns:
            bool: True if notification was added/sent successfully
        """
        try:
            notification = Notification(
                id=f"notif_{int(time.time() * 1000)}",
                category=category,
                title=title,
                message=message,
                timestamp=datetime.now(),
                priority=priority,
                metadata=metadata
            )
            
            # Store detailed explanation in metadata for email use
            if detailed_explanation:
                if metadata is None:
                    metadata = {}
                metadata['detailed_explanation'] = detailed_explanation
                notification.metadata = metadata
            
            # Handle critical notifications immediately
            if bypass_batch or category == NotificationCategory.CRITICAL or priority >= 3:
                return self._send_immediate_notification(notification)
            else:
                return self.buffer.add_notification(notification)
                
        except Exception as e:
            logger.error(f"❌ Failed to add notification: {e}")
            return False
    
    def _send_immediate_notification(self, notification: Notification) -> bool:
        """Send a notification immediately (bypass batch)"""
        try:
            formatted_message = self.formatter.format_single_notification(notification)
            
            if self.telegram_bot:
                try:
                    if hasattr(self.telegram_bot, 'send_message'):
                        self.telegram_bot.send_message(
                            text=formatted_message,
                            parse_mode='Markdown'
                        )
                        logger.info(f"🚀 Sent immediate notification: {notification.title}")
                        return True
                    else:
                        logger.warning("⚠️ Telegram bot missing send_message method")
                        return False
                except Exception as e:
                    logger.error(f"❌ Failed to send immediate notification: {e}")
                    return False
            else:
                logger.warning("⚠️ No Telegram bot available for immediate notification")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending immediate notification: {e}")
            return False
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status"""
        return {
            "buffer_size": self.buffer.get_buffer_size(),
            "batch_interval": self.batch_interval,
            "running": self.running,
            "last_batch_time": self.last_batch_time.isoformat(),
            "persist_to_db": self.buffer.persist_to_db
        }


# Convenience functions for easy integration
def create_notification_manager(telegram_bot=None, batch_interval: int = 10) -> HybridNotificationManager:
    """
    Create a hybrid notification manager instance
    
    Args:
        telegram_bot: Telegram bot instance
        batch_interval: Batch processing interval in seconds
        
    Returns:
        HybridNotificationManager: Configured notification manager
    """
    return HybridNotificationManager(
        telegram_bot=telegram_bot,
        batch_interval=batch_interval,
        persist_to_db=True,
        max_buffer_size=1000
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the notification system
    print("🧪 Testing Hybrid Notification System")
    
    # Create manager
    manager = create_notification_manager(batch_interval=5)
    
    # Add some test notifications
    manager.add_notification(
        category=NotificationCategory.TRADE,
        title="BUY Signal Detected",
        message="ETHUSDT showing strong buy momentum at $3,200",
        priority=1
    )
    
    manager.add_notification(
        category=NotificationCategory.RISK,
        title="Risk Level Elevated",
        message="Portfolio risk increased to 2.5%",
        priority=2
    )
    
    # Critical notification (should bypass batch)
    manager.add_notification(
        category=NotificationCategory.CRITICAL,
        title="System Alert",
        message="Connection timeout detected",
        priority=3,
        bypass_batch=True
    )
    
    print(f"📊 Buffer status: {manager.get_buffer_status()}")
    print("✅ Notification system test completed")