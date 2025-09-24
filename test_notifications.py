#!/usr/bin/env python3
"""
TitanovaX Notification System Test
Tests Telegram and Email delivery with verification
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_telegram_notification():
    """Test Telegram notification delivery"""
    print("🧪 Testing Telegram Notifications...")
    
    try:
        from orchestration.telegram_bot import TitanovaXTelegramBot
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        channel_id = os.getenv('TELEGRAM_CHAT_ID')
        if not bot_token or not chat_id:
            print("⚠️  Telegram credentials not configured")
            return False
            
        print(f"📱 Bot Token: {bot_token[:10]}...")
        print(f"💬 Chat ID: {chat_id}")
        
        # Initialize bot
        bot = TitanovaXTelegramBot(bot_token)
        
        # Create new event loop for initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(bot.initialize())
        
        if success:
            print("✅ Telegram bot initialized successfully")
            
            # Send test message
            test_message = f"🧪 TitanovaX Test Message\n"
            test_message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            test_message += f"✅ Telegram notifications are working!\n"
            test_message += f"🤖 This is an automated test from TitanovaX"
            
            message_success = loop.run_until_complete(bot.send_message(test_message, chat_id))
            loop.close()
            
            if message_success:
                print("✅ Telegram test message sent successfully!")
                return True
            else:
                print("❌ Failed to send Telegram message")
                return False
        else:
            print("❌ Telegram bot initialization failed")
            loop.close()
            return False
            
    except Exception as e:
        print(f"❌ Telegram test failed: {e}")
        return False

def test_email_notification():
    """Test email notification delivery"""
    print("\n📧 Testing Email Notifications...")
    
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        smtp_host = os.getenv('EMAIL_SMTP_HOST')
        smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        email_username = os.getenv('EMAIL_USERNAME')
        email_password = os.getenv('EMAIL_PASSWORD')
        email_from = os.getenv('EMAIL_FROM')
        email_to = os.getenv('EMAIL_TO')
        
        if not all([smtp_host, email_username, email_password, email_to]):
            print("⚠️  Email credentials not configured")
            return False
            
        print(f"📧 SMTP Host: {smtp_host}:{smtp_port}")
        print(f"📧 From: {email_from}")
        print(f"📧 To: {email_to}")
        
        # Create test message
        subject = "TitanovaX Email Test"
        body = f"""
TitanovaX Email Notification Test

📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ Email notifications are working!
🤖 This is an automated test from TitanovaX

If you received this message, your email configuration is correct.
        """
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = email_from
        msg["To"] = email_to
        
        html_content = f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; margin: 20px; }}
              .header {{ background-color: #28a745; color: white; padding: 20px; border-radius: 5px; }}
              .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 10px; }}
              .footer {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 12px; }}
            </style>
          </head>
          <body>
            <div class="header">
              <h2>✅ TitanovaX Email Test</h2>
              <p>Notification System Verification</p>
            </div>
            <div class="content">
              <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
              <p><strong>Status:</strong> Email notifications are working!</p>
              <p><strong>System:</strong> TitanovaX Enhanced Trading System</p>
            </div>
            <div class="footer">
              <p><small>This is an automated test message from TitanovaX</small></p>
            </div>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(html_content, "html"))
        
        # Send email
        import ssl
        context = ssl.create_default_context()
        
        # Use SSL connection for better reliability (port 465)
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
                server.login(email_username, email_password)
                server.sendmail(email_from, email_to, msg.as_string())
        else:
            # Fallback to STARTTLS for port 587
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls(context=context)
                server.login(email_username, email_password)
                server.sendmail(email_from, email_to, msg.as_string())
        
        print("✅ Email test message sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Email test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 TitanovaX Notification System Test")
    print("="*50)
    
    # Load environment variables
    if os.path.exists('.env'):
        print("📋 Loading environment variables...")
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded")
    
    # Test Telegram
    telegram_success = test_telegram_notification()
    
    # Test Email
    email_success = test_email_notification()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    print(f"📱 Telegram: {'✅ WORKING' if telegram_success else '❌ FAILED'}")
    print(f"📧 Email: {'✅ WORKING' if email_success else '❌ FAILED'}")
    
    if telegram_success and email_success:
        print("\n🎉 ALL NOTIFICATION SYSTEMS ARE OPERATIONAL!")
        print("✅ Your TitanovaX system can now send alerts via both channels.")
    elif telegram_success or email_success:
        print("\n⚠️  PARTIAL SUCCESS: At least one notification channel is working.")
    else:
        print("\n❌ CRITICAL: No notification channels are working.")
        print("🔧 Please check your configuration and network connectivity.")
    
    # Save test results
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'telegram_success': telegram_success,
        'email_success': email_success,
        'overall_status': 'success' if telegram_success and email_success else 'partial' if telegram_success or email_success else 'failed'
    }
    
    with open('notification_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n📄 Test results saved to: notification_test_results.json")

if __name__ == "__main__":
    main()