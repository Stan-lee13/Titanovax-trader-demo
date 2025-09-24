"""
TitanovaX Notification Batcher - Reliable & Batched Notifications
Replaces immediate alerts with batched, reliable delivery system
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Import our async manager
from async_manager import async_manager, async_safe

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class NotificationType(Enum):
    """Types of notifications"""
    TRADE = "trade"
    RISK = "risk"
    SIGNAL = "signal"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    CRITICAL = "critical"


@dataclass
class Notification:
    """Notification data structure"""
    id: str
    type: NotificationType
    title: str
    message: str
    timestamp: datetime
    priority: NotificationPriority
    metadata: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BatchMessage:
    """Batched message for delivery"""
    batch_id: str
    notifications: List[Notification]
    created_at: datetime
    delivery_attempts: int = 0


class NotificationBatcher:
    """
    Reliable notification batching system with fallback channels
    
    Features:
    - Batches notifications to prevent spam
    - Multiple delivery channels (Telegram, Email, Webhook)
    - Automatic retry with exponential backoff
    - Fallback channel support
    - Memory efficient buffering
    """
    
    def __init__(self, 
                 batch_interval: int = 10,
                 max_batch_size: int = 50,
                 telegram_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None,
                 email_config: Optional[Dict[str, str]] = None,
                 webhook_url: Optional[str] = None,
                 enable_fallback: bool = True):
        """
        Initialize the notification batcher
        
        Args:
            batch_interval: Seconds between batch deliveries
            max_batch_size: Maximum notifications per batch
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
            email_config: Email configuration dict
            webhook_url: Webhook URL for notifications
            enable_fallback: Enable fallback channels
        """
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.email_config = email_config or {}
        self.webhook_url = webhook_url
        self.enable_fallback = enable_fallback
        
        # Buffers
        self.notification_buffer: List[Notification] = []
        self.pending_batches: List[BatchMessage] = []
        
        # Control flags
        self.running = False
        self._lock = asyncio.Lock()
        
        # Channel availability
        self.channels_available = {
            'telegram': bool(telegram_token and telegram_chat_id),
            'email': bool(email_config),
            'webhook': bool(webhook_url)
        }
        
        logger.info(f"🚀 NotificationBatcher initialized - Interval: {batch_interval}s, Channels: {self.channels_available}")
    
    @async_safe
    async def start(self):
        """Start the batching service"""
        if self.running:
            return
            
        self.running = True
        # Start the batch processor
        await async_manager.ensure_task(self._batch_processor())
        logger.info("✅ NotificationBatcher started")
    
    async def stop(self):
        """Stop the batching service"""
        self.running = False
        # Process any remaining notifications
        await self._flush_buffer()
        logger.info("🛑 NotificationBatcher stopped")
    
    @async_safe
    async def add_notification(self, notification: Notification):
        """
        Add a notification to the buffer
        
        Args:
            notification: Notification to add
        """
        async with self._lock:
            self.notification_buffer.append(notification)
            
            # If critical, send immediately
            if notification.priority == NotificationPriority.CRITICAL:
                await self._send_critical_notification(notification)
            
            # Check if buffer is full
            if len(self.notification_buffer) >= self.max_batch_size:
                await self._create_batch()
    
    async def _batch_processor(self):
        """Main batch processing loop"""
        while self.running:
            try:
                await asyncio.sleep(self.batch_interval)
                await self._process_batches()
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _process_batches(self):
        """Process notification batches"""
        async with self._lock:
            if self.notification_buffer:
                await self._create_batch()
            
            # Process pending batches
            for batch in self.pending_batches[:]:
                await self._deliver_batch(batch)
    
    async def _create_batch(self):
        """Create a batch from current buffer"""
        if not self.notification_buffer:
            return
        
        batch_notifications = self.notification_buffer[:self.max_batch_size]
        self.notification_buffer = self.notification_buffer[self.max_batch_size:]
        
        batch = BatchMessage(
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            notifications=batch_notifications,
            created_at=datetime.now()
        )
        
        self.pending_batches.append(batch)
        logger.debug(f"📦 Created batch {batch.batch_id} with {len(batch.notifications)} notifications")
    
    async def _deliver_batch(self, batch: BatchMessage):
        """Deliver a batch through available channels"""
        delivery_success = False
        
        # Try primary channels
        for channel in ['telegram', 'email', 'webhook']:
            if self.channels_available[channel]:
                try:
                    await self._send_via_channel(channel, batch)
                    delivery_success = True
                    logger.info(f"✅ Batch {batch.batch_id} delivered via {channel}")
                    break
                except Exception as e:
                    logger.error(f"❌ Failed to deliver batch {batch.batch_id} via {channel}: {e}")
                    if not self.enable_fallback:
                        break
        
        if delivery_success:
            self.pending_batches.remove(batch)
        else:
            # Retry logic
            batch.delivery_attempts += 1
            if batch.delivery_attempts >= 3:
                logger.error(f"❌ Batch {batch.batch_id} failed after 3 attempts, dropping")
                self.pending_batches.remove(batch)
            else:
                # Exponential backoff
                backoff_time = 2 ** batch.delivery_attempts
                logger.warning(f"⏰ Batch {batch.batch_id} retrying in {backoff_time}s (attempt {batch.delivery_attempts + 1})")
                await asyncio.sleep(backoff_time)
    
    async def _send_via_channel(self, channel: str, batch: BatchMessage):
        """Send batch through specific channel"""
        message = self._format_batch_message(batch)
        
        if channel == 'telegram':
            await self._send_telegram(message)
        elif channel == 'email':
            await self._send_email(f"TitanovaX Batch Notifications ({len(batch.notifications)})", message)
        elif channel == 'webhook':
            await self._send_webhook({
                'batch_id': batch.batch_id,
                'notifications': [asdict(n) for n in batch.notifications],
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
    
    async def _send_critical_notification(self, notification: Notification):
        """Send critical notification immediately through all channels"""
        message = self._format_critical_notification(notification)
        
        for channel in ['telegram', 'email', 'webhook']:
            if self.channels_available[channel]:
                try:
                    if channel == 'telegram':
                        await self._send_telegram(message)
                    elif channel == 'email':
                        await self._send_email(f"🚨 CRITICAL: {notification.title}", message)
                    elif channel == 'webhook':
                        await self._send_webhook({
                            'type': 'critical',
                            'notification': asdict(notification),
                            'message': message,
                            'timestamp': datetime.now().isoformat()
                        })
                    logger.info(f"🚨 Critical notification sent via {channel}")
                except Exception as e:
                    logger.error(f"❌ Failed to send critical notification via {channel}: {e}")
    
    async def _send_telegram(self, message: str):
        """Send message via Telegram"""
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Telegram API error: {response.status}")
    
    async def _send_email(self, subject: str, body: str):
        """Send email notification"""
        if not self.email_config:
            return
            
        msg = MIMEMultipart()
        msg['From'] = self.email_config.get('from', 'titanovax@system.com')
        msg['To'] = self.email_config.get('to', 'admin@titanovax.com')
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Simple SMTP sending (you may want to use aiosmtplib for async)
        import smtplib
        try:
            with smtplib.SMTP(self.email_config.get('smtp_host', 'localhost'), 
                            self.email_config.get('smtp_port', 587)) as server:
                if self.email_config.get('use_tls', True):
                    server.starttls()
                if self.email_config.get('username'):
                    server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            raise
    
    async def _send_webhook(self, data: Dict[str, Any]):
        """Send webhook notification"""
        if not self.webhook_url:
            return
            
        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=data) as response:
                if response.status >= 400:
                    raise Exception(f"Webhook error: {response.status}")
    
    def _format_batch_message(self, batch: BatchMessage) -> str:
        """Format batch message for delivery"""
        message_parts = [
            f"📊 **TitanovaX Batch Notifications**",
            f"⏰ **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"📋 **Count**: {len(batch.notifications)} notifications",
            ""
        ]
        
        # Group by type
        type_counts = {}
        for notification in batch.notifications:
            type_name = notification.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        for type_name, count in type_counts.items():
            message_parts.append(f"• {type_name.title()}: {count}")
        
        # Add summary of high priority items
        high_priority = [n for n in batch.notifications if n.priority >= NotificationPriority.HIGH]
        if high_priority:
            message_parts.extend(["", "⚠️ **High Priority Items**:"])
            for notification in high_priority[:5]:  # Show first 5
                message_parts.append(f"• {notification.title}: {notification.message}")
        
        message_parts.extend(["", "🤖 *TitanovaX Trading System*"])
        
        return "\n".join(message_parts)
    
    def _format_critical_notification(self, notification: Notification) -> str:
        """Format critical notification"""
        return f"""
🚨 **CRITICAL ALERT: {notification.title}**
⏰ **Time**: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{notification.message}

🤖 *TitanovaX Trading System - IMMEDIATE ATTENTION REQUIRED*
""".strip()
    
    async def _flush_buffer(self):
        """Flush remaining notifications"""
        async with self._lock:
            if self.notification_buffer:
                await self._create_batch()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'running': self.running,
            'buffer_size': len(self.notification_buffer),
            'pending_batches': len(self.pending_batches),
            'channels_available': self.channels_available,
            'batch_interval': self.batch_interval,
            'max_batch_size': self.max_batch_size
        }


# Global notification batcher instance
notification_batcher: Optional[NotificationBatcher] = None


def initialize_notification_batcher(config: Dict[str, Any]) -> NotificationBatcher:
    """Initialize global notification batcher"""
    global notification_batcher
    
    notification_batcher = NotificationBatcher(
        batch_interval=config.get('batch_interval', 10),
        max_batch_size=config.get('max_batch_size', 50),
        telegram_token=config.get('telegram_token'),
        telegram_chat_id=config.get('telegram_chat_id'),
        email_config=config.get('email_config'),
        webhook_url=config.get('webhook_url'),
        enable_fallback=config.get('enable_fallback', True)
    )
    
    return notification_batcher


def get_notification_batcher() -> Optional[NotificationBatcher]:
    """Get global notification batcher instance"""
    return notification_batcher