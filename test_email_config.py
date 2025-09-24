#!/usr/bin/env python3
"""
Email Configuration Test for TitanovaX Trading System
Tests SMTP connection and email functionality
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

def test_email_configuration():
    """Test email configuration and connectivity"""
    print("Testing Email Configuration...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get email configuration
    smtp_host = os.getenv('EMAIL_SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
    email_username = os.getenv('EMAIL_USERNAME', '')
    email_password = os.getenv('EMAIL_PASSWORD', '')
    email_from = os.getenv('EMAIL_FROM', '')
    email_to = os.getenv('EMAIL_TO', '')
    
    print(f"SMTP Host: {smtp_host}")
    print(f"SMTP Port: {smtp_port}")
    print(f"Email Username: {email_username}")
    print(f"Email From: {email_from}")
    print(f"Email To: {email_to}")
    
    # Check if email credentials are configured
    if not email_username or not email_password:
        print("❌ Email credentials not configured properly")
        return False
    
    try:
        # Test SMTP connection
        print("\nTesting SMTP connection...")
        context = ssl.create_default_context()
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(email_username, email_password)
            print("✅ SMTP connection successful")
            
            # Send test email
            if email_to:
                print("\nSending test email...")
                
                message = MIMEMultipart("alternative")
                message["Subject"] = "TitanovaX Email Configuration Test"
                message["From"] = email_from
                message["To"] = email_to
                
                text = """\
Hi there,

This is a test email from TitanovaX Trading System.

Your email configuration is working correctly!

Best regards,
TitanovaX Team
"""
                
                html = """\
<html>
  <body>
    <h2>TitanovaX Email Configuration Test</h2>
    <p>Hi there,</p>
    <p>This is a test email from <strong>TitanovaX Trading System</strong>.</p>
    <p>Your email configuration is working correctly!</p>
    <br>
    <p>Best regards,<br>
    <strong>TitanovaX Team</strong></p>
  </body>
</html>
"""
                
                part1 = MIMEText(text, "plain")
                part2 = MIMEText(html, "html")
                
                message.attach(part1)
                message.attach(part2)
                
                server.sendmail(email_from, email_to, message.as_string())
                print("✅ Test email sent successfully")
            else:
                print("⚠️  No recipient email configured (EMAIL_TO is empty)")
                print("Email configuration is working but no test email sent")
            
            return True
            
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ SMTP Authentication failed: {e}")
        return False
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_email_configuration_basic():
    """Basic email configuration test without sending"""
    print("Testing Email Configuration (Basic)...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get email configuration
    smtp_host = os.getenv('EMAIL_SMTP_HOST')
    smtp_port = os.getenv('EMAIL_SMTP_PORT')
    email_username = os.getenv('EMAIL_USERNAME')
    email_password = os.getenv('EMAIL_PASSWORD')
    email_from = os.getenv('EMAIL_FROM')
    email_to = os.getenv('EMAIL_TO')
    
    print(f"SMTP Host: {smtp_host}")
    print(f"SMTP Port: {smtp_port}")
    print(f"Email Username: {email_username}")
    print(f"Email From: {email_from}")
    print(f"Email To: {email_to}")
    
    # Check configuration
    if smtp_host and smtp_port and email_username and email_password and email_from:
        print("✅ Email configuration is complete")
        if email_to:
            print("✅ Recipient email is configured")
        else:
            print("⚠️  Recipient email (EMAIL_TO) is not configured")
        return True
    else:
        print("❌ Email configuration is incomplete")
        return False

if __name__ == "__main__":
    print("TitanovaX Email Configuration Test")
    print("=" * 50)
    
    # Test basic configuration first
    basic_test = test_email_configuration_basic()
    
    if basic_test:
        print("\nTesting full email functionality...")
        full_test = test_email_configuration()
        
        if full_test:
            print("\n✅ Email configuration test completed successfully!")
        else:
            print("\n⚠️  Email configuration test completed with warnings.")
            print("Basic configuration is valid but connection test failed.")
    else:
        print("\n❌ Email configuration is incomplete. Please update your .env file.")
    
    print("\nEmail configuration test finished.")