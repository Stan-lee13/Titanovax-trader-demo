"""
Test script for the hybrid batch notification system
"""

import time
import threading
from datetime import datetime
from hybrid_notification_system import (
    HybridNotificationManager, 
    NotificationCategory, 
    create_notification_manager,
    NotificationBuffer,
    NotificationFormatter,
    Notification,
    BatchNotification
)


class MockTelegramBot:
    """Mock Telegram bot for testing"""
    def __init__(self):
        self.chat_id = "test_chat_123"
        self.messages_sent = []
    
    def send_message(self, chat_id, text, parse_mode=None):
        """Mock send_message method"""
        self.messages_sent.append({
            'chat_id': chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'timestamp': datetime.now()
        })
        print(f"📤 MOCK TELEGRAM: {text[:100]}...")
        return True


def test_notification_buffer():
    """Test the notification buffer functionality"""
    print("🧪 Testing NotificationBuffer...")
    
    buffer = NotificationBuffer(max_size=10, persist_to_db=False)
    
    # Add test notifications
    for i in range(5):
        notification = Notification(
            id=f"test_{i}",
            category=NotificationCategory.TRADE,
            title=f"Test Trade {i}",
            message=f"Test message {i}",
            timestamp=datetime.now()
        )
        buffer.add_notification(notification)
    
    # Check buffer size
    assert buffer.get_buffer_size() == 5, f"Expected 5, got {buffer.get_buffer_size()}"
    
    # Get and clear buffer
    notifications = buffer.get_and_clear_buffer()
    assert len(notifications) == 5, f"Expected 5, got {len(notifications)}"
    assert buffer.get_buffer_size() == 0, f"Expected 0, got {buffer.get_buffer_size()}"
    
    print("✅ NotificationBuffer test passed")


def test_notification_formatter():
    """Test the notification formatter"""
    print("🧪 Testing NotificationFormatter...")
    
    formatter = NotificationFormatter()
    
    # Test single notification formatting
    notification = Notification(
        id="test_1",
        category=NotificationCategory.TRADE,
        title="BUY Signal",
        message="ETHUSDT buy signal at $3200",
        timestamp=datetime.now()
    )
    
    formatted = formatter.format_single_notification(notification)
    assert "BUY Signal" in formatted, "Title not found in formatted message"
    assert "ETHUSDT" in formatted, "Message not found in formatted message"
    
    # Test batch formatting
    notifications = [
        Notification(
            id="batch_1",
            category=NotificationCategory.TRADE,
            title="Trade 1",
            message="Buy ETH",
            timestamp=datetime.now()
        ),
        Notification(
            id="batch_2",
            category=NotificationCategory.RISK,
            title="Risk Alert",
            message="Risk level high",
            timestamp=datetime.now()
        )
    ]
    
    batch = BatchNotification(
        batch_id="test_batch",
        start_time=datetime.now(),
        end_time=datetime.now(),
        notifications=notifications,
        category_counts={"trade": 1, "risk": 1}
    )
    
    batch_formatted = formatter.format_batch_notification(batch)
    assert "TRADE EXECUTIONS" in batch_formatted, "Trade section not found"
    assert "RISK ALERTS" in batch_formatted, "Risk section not found"
    
    print("✅ NotificationFormatter test passed")


def test_hybrid_manager():
    """Test the hybrid notification manager"""
    print("🧪 Testing HybridNotificationManager...")
    
    # Create mock telegram bot
    mock_bot = MockTelegramBot()
    
    # Create manager with short batch interval for testing
    manager = create_notification_manager(
        telegram_bot=mock_bot,
        batch_interval=3  # 3 seconds for quick testing
    )
    
    # Test adding normal notifications (should go to buffer)
    manager.add_notification(
        category=NotificationCategory.TRADE,
        title="Test Trade Signal",
        message="BTCUSDT showing buy momentum",
        priority=1
    )
    
    manager.add_notification(
        category=NotificationCategory.RISK,
        title="Risk Level Update",
        message="Portfolio risk at 1.2%",
        priority=2
    )
    
    # Test critical notification (should bypass batch)
    manager.add_notification(
        category=NotificationCategory.CRITICAL,
        title="CRITICAL: System Alert",
        message="Connection timeout detected",
        priority=3,
        bypass_batch=True
    )
    
    # Check buffer status
    status = manager.get_buffer_status()
    assert status['buffer_size'] == 2, f"Expected buffer size 2, got {status['buffer_size']}"
    assert status['batch_interval'] == 3, f"Expected interval 3, got {status['batch_interval']}"
    
    # Start batch processor
    manager.start_batch_processor()
    
    # Wait for batch processing
    print("⏳ Waiting for batch processing...")
    time.sleep(5)  # Wait for batch interval
    
    # Check that critical message was sent immediately
    critical_messages = [msg for msg in mock_bot.messages_sent if "CRITICAL" in msg['text']]
    assert len(critical_messages) > 0, "Critical message not sent immediately"
    
    # Add more messages and wait for batch
    manager.add_notification(
        category=NotificationCategory.SIGNAL,
        title="ML Signal Update",
        message="New ML prediction available",
        priority=1
    )
    
    time.sleep(4)  # Wait for next batch
    
    # Stop batch processor
    manager.stop_batch_processor()
    
    # Verify messages were sent
    assert len(mock_bot.messages_sent) > 0, "No messages were sent"
    print(f"📊 Total messages sent: {len(mock_bot.messages_sent)}")
    
    print("✅ HybridNotificationManager test passed")


def test_integration_simulation():
    """Simulate real-world usage scenario"""
    print("🧪 Testing Integration Simulation...")
    
    mock_bot = MockTelegramBot()
    manager = create_notification_manager(
        telegram_bot=mock_bot,
        batch_interval=2  # 2 seconds for quick testing
    )
    
    manager.start_batch_processor()
    
    # Simulate various notification types
    notifications = [
        (NotificationCategory.TRADE, "BUY Signal", "ETHUSDT buy signal at $3200", 1),
        (NotificationCategory.RISK, "Risk Alert", "Portfolio risk increased to 2.5%", 2),
        (NotificationCategory.SIGNAL, "ML Update", "New prediction confidence: 85%", 1),
        (NotificationCategory.CRITICAL, "System Warning", "API rate limit approaching", 3),
        (NotificationCategory.PERFORMANCE, "Performance Update", "Daily P&L: +2.3%", 1),
    ]
    
    # Send notifications with delays
    for category, title, message, priority in notifications:
        if priority >= 3:  # Critical notifications
            manager.add_notification(category, title, message, priority, bypass_batch=True)
        else:
            manager.add_notification(category, title, message, priority)
        time.sleep(0.5)  # Small delay between notifications
    
    # Wait for batch processing
    time.sleep(3)
    
    manager.stop_batch_processor()
    
    # Verify results
    print(f"📊 Total messages sent: {len(mock_bot.messages_sent)}")
    
    # Check for critical message (should be sent immediately)
    critical_found = any("System Warning" in msg['text'] or "CRITICAL" in msg['text'] for msg in mock_bot.messages_sent)
    assert critical_found, "Critical message not found in sent messages"
    
    # Check for batch message (should contain multiple notifications)
    batch_found = any("BATCH NOTIFICATIONS" in msg['text'] for msg in mock_bot.messages_sent)
    assert batch_found, "Batch notification not found in sent messages"
    
    print("✅ Integration Simulation test passed")


def main():
    """Run all tests"""
    print("🚀 Starting Hybrid Notification System Tests")
    print("=" * 50)
    
    try:
        test_notification_buffer()
        test_notification_formatter()
        test_hybrid_manager()
        test_integration_simulation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Hybrid notification system is working correctly.")
        print("✅ NotificationBuffer: Stores and manages notifications")
        print("✅ NotificationFormatter: Formats messages beautifully")
        print("✅ HybridNotificationManager: Handles batch and immediate notifications")
        print("✅ Integration: Works in realistic scenarios")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()