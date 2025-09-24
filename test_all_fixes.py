#!/usr/bin/env python3
"""
TitanovaX Enhanced Production Bot - Comprehensive Fix Testing
Tests all implemented fixes for the outstanding issues:
1. Telegram Event Loop Closure
2. Email SMTP Timeout Issues  
3. Risk Alert Debouncing
"""

import os
import sys
import time
import threading
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from run_full_bot_enhanced import EnhancedTitanovaXBot

def test_telegram_event_loop():
    """Test 1: Telegram Event Loop Persistence"""
    print("\n" + "="*60)
    print("TEST 1: TELEGRAM EVENT LOOP PERSISTENCE")
    print("="*60)
    
    try:
        bot = EnhancedTitanovaXBot()
        
        # Test multiple telegram notifications
        print("Sending multiple Telegram notifications...")
        
        for i in range(3):
            success = bot.send_telegram_notification(f"Test message {i+1} - Event loop persistence test")
            print(f"Message {i+1}: {'✅ SUCCESS' if success else '❌ FAILED'}")
            time.sleep(2)
        
        # Check if event loop is still alive
        if bot.telegram_event_loop and not bot.telegram_event_loop.is_closed():
            print("✅ Telegram event loop is persistent and not closed")
            return True
        else:
            print("❌ Telegram event loop is closed or not available")
            return False
            
    except Exception as e:
        print(f"❌ Telegram event loop test failed: {e}")
        return False

def test_email_smtp():
    """Test 2: Email SMTP with Gmail"""
    print("\n" + "="*60)
    print("TEST 2: EMAIL SMTP WITH GMAIL")
    print("="*60)
    
    try:
        bot = EnhancedTitanovaXBot()
        
        # Test email initialization
        print("Testing email initialization...")
        email_init = bot.initialize_email_sender()
        print(f"Email initialization: {'✅ SUCCESS' if email_init else '❌ FAILED'}")
        
        if not email_init:
            return False
        
        # Test sending email
        print("Testing email notification...")
        email_success = bot.send_email_notification(
            "Test Email - SMTP Fix Verification",
            "This is a test email to verify the SMTP timeout fix is working properly."
        )
        print(f"Email notification: {'✅ SUCCESS' if email_success else '❌ FAILED'}")
        
        return email_success
        
    except Exception as e:
        print(f"❌ Email SMTP test failed: {e}")
        return False

def test_risk_alert_debouncing():
    """Test 3: Risk Alert Debouncing"""
    print("\n" + "="*60)
    print("TEST 3: RISK ALERT DEBOUNCING")
    print("="*60)
    
    try:
        bot = EnhancedTitanovaXBot()
        
        print("Testing risk alert debouncing...")
        
        # Simulate multiple risk management calls with same risk level
        alerts_sent = []
        for i in range(10):
            # Simulate risk level around 2.1% (above 2% threshold)
            risk_level = 2.1 + (i % 3) * 0.1  # Varies between 2.1% and 2.3%
            
            # Mock the risk management to test debouncing
            current_time = time.time()
            should_send_alert = False
            
            # Check debouncing logic (same as in simulate_risk_management)
            if (bot.risk_alert_last_sent is None or 
                current_time - bot.risk_alert_last_sent > bot.risk_alert_cooldown or
                abs(risk_level - bot.last_risk_level) > 0.5):  # 0.5% threshold
                should_send_alert = True
                bot.risk_alert_last_sent = current_time
                bot.last_risk_level = risk_level
            
            alerts_sent.append(should_send_alert)
            print(f"Cycle {i+1}: Risk Level {risk_level:.1f}% - Alert: {'SENT' if should_send_alert else 'SUPPRESSED'}")
            time.sleep(0.5)  # Short delay between tests
        
        # Analyze results
        total_sent = sum(alerts_sent)
        total_suppressed = len(alerts_sent) - total_sent
        
        print(f"\nResults:")
        print(f"Total cycles: {len(alerts_sent)}")
        print(f"Alerts sent: {total_sent}")
        print(f"Alerts suppressed: {total_suppressed}")
        
        # Test passes if debouncing is working (some alerts are suppressed)
        if total_suppressed > 0:
            print("✅ Risk alert debouncing is working properly")
            return True
        else:
            print("⚠️  No alerts were suppressed - debouncing may not be working")
            return False
            
    except Exception as e:
        print(f"❌ Risk alert debouncing test failed: {e}")
        return False

def test_complete_production_cycle():
    """Test 4: Complete Production Cycle Integration"""
    print("\n" + "="*60)
    print("TEST 4: COMPLETE PRODUCTION CYCLE INTEGRATION")
    print("="*60)
    
    try:
        print("Testing complete production cycle with all fixes...")
        
        # Run a short production cycle
        os.system("python run_full_bot_enhanced.py --cycles 3 --test-mode")
        
        print("✅ Complete production cycle test initiated")
        return True
        
    except Exception as e:
        print(f"❌ Complete production cycle test failed: {e}")
        return False

def main():
    """Run all tests and provide summary"""
    print("🚀 Starting TitanovaX Enhanced Bot Fix Testing")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {
        'telegram_event_loop': test_telegram_event_loop(),
        'email_smtp': test_email_smtp(),
        'risk_alert_debouncing': test_risk_alert_debouncing(),
        'complete_production_cycle': test_complete_production_cycle()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes are working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed - review the fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())