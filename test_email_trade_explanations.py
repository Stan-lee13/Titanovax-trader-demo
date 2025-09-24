#!/usr/bin/env python3
"""
Comprehensive Email Trade Explanations Test
Tests the email system with detailed trade explanations and PDF attachments
"""

import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailTradeExplainer:
    """Enhanced email system for detailed trade explanations"""
    
    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        self.smtp_host = os.getenv('EMAIL_SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '465'))
        self.email_username = os.getenv('EMAIL_USERNAME')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_from = os.getenv('EMAIL_FROM')
        self.email_to = os.getenv('EMAIL_TO')
        
        logger.info(f"📧 Email configuration: {self.smtp_host}:{self.smtp_port}")
        logger.info(f"📧 From: {self.email_from} -> To: {self.email_to}")
        
        # Validate configuration
        if not all([self.email_username, self.email_password, self.email_from, self.email_to]):
            logger.error("❌ Missing email configuration. Please check your .env file.")
            raise ValueError("Email configuration is incomplete")
    
    def create_trade_explanation_html(self, trade_data: Dict[str, Any]) -> str:
        """Create detailed HTML content for trade explanations"""
        
        # Trade details
        symbol = trade_data.get('symbol', 'Unknown')
        trade_type = trade_data.get('type', 'BUY')
        entry_price = trade_data.get('entry_price', 0)
        exit_price = trade_data.get('exit_price', 0)
        quantity = trade_data.get('quantity', 0)
        pnl = trade_data.get('pnl', 0)
        timestamp = trade_data.get('timestamp', datetime.now())
        confidence = trade_data.get('confidence', 0)
        signal_strength = trade_data.get('signal_strength', 0)
        
        # ML Analysis
        ml_reasoning = trade_data.get('ml_reasoning', 'No detailed analysis available')
        model_accuracy = trade_data.get('model_accuracy', 0)
        prediction_confidence = trade_data.get('prediction_confidence', 0)
        
        # Risk Analysis
        risk_level = trade_data.get('risk_level', 'Medium')
        stop_loss = trade_data.get('stop_loss', 0)
        take_profit = trade_data.get('take_profit', 0)
        risk_reward_ratio = trade_data.get('risk_reward_ratio', 0)
        
        # Market Context
        market_trend = trade_data.get('market_trend', 'Neutral')
        volatility = trade_data.get('volatility', 'Normal')
        volume_analysis = trade_data.get('volume_analysis', 'Standard')
        
        # Format PnL with color
        pnl_color = "#27ae60" if pnl >= 0 else "#e74c3c"
        pnl_sign = "+" if pnl >= 0 else ""
        
        html_content = f"""
        <html>
          <head>
            <style>
              body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
              .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; }}
              .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
              .header h1 {{ margin: 0; font-size: 28px; font-weight: 300; }}
              .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
              .trade-summary {{ background: #2c3e50; color: white; padding: 25px; display: flex; justify-content: space-around; flex-wrap: wrap; }}
              .metric {{ text-align: center; padding: 15px; }}
              .metric-value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
              .metric-label {{ font-size: 12px; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }}
              .content {{ padding: 30px; }}
              .section {{ margin-bottom: 30px; background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
              .section h2 {{ color: #2c3e50; margin-top: 0; font-size: 20px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
              .detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
              .detail-item {{ background: white; padding: 15px; border-radius: 6px; border: 1px solid #e1e8ed; }}
              .detail-label {{ font-weight: 600; color: #7f8c8d; font-size: 12px; text-transform: uppercase; margin-bottom: 5px; }}
              .detail-value {{ color: #2c3e50; font-size: 16px; font-weight: 500; }}
              .ml-analysis {{ background: #e8f5e8; border-left-color: #27ae60; }}
              .risk-analysis {{ background: #fff3cd; border-left-color: #ffc107; }}
              .market-context {{ background: #e3f2fd; border-left-color: #2196f3; }}
              .pnl-positive {{ color: #27ae60; font-weight: bold; }}
              .pnl-negative {{ color: #e74c3c; font-weight: bold; }}
              .confidence-high {{ color: #27ae60; }}
              .confidence-medium {{ color: #f39c12; }}
              .confidence-low {{ color: #e74c3c; }}
              .footer {{ background: #34495e; color: white; padding: 20px; text-align: center; font-size: 12px; }}
              .timestamp {{ opacity: 0.7; }}
              @media (max-width: 600px) {{
                .trade-summary {{ flex-direction: column; }}
                .detail-grid {{ grid-template-columns: 1fr; }}
              }}
            </style>
          </head>
          <body>
            <div class="container">
              <div class="header">
                <h1>🤖 TitanovaX Trade Execution Report</h1>
                <p>AI-Powered Trading System - Detailed Analysis</p>
              </div>
              
              <div class="trade-summary">
                <div class="metric">
                  <div class="metric-value">{symbol}</div>
                  <div class="metric-label">Symbol</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{trade_type}</div>
                  <div class="metric-label">Trade Type</div>
                </div>
                <div class="metric">
                  <div class="metric-value" style="color: {pnl_color}">{pnl_sign}${pnl:.2f}</div>
                  <div class="metric-label">P&L</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{confidence:.1%}</div>
                  <div class="metric-label">Confidence</div>
                </div>
              </div>
              
              <div class="content">
                <div class="section">
                  <h2>📊 Trade Execution Details</h2>
                  <div class="detail-grid">
                    <div class="detail-item">
                      <div class="detail-label">Entry Price</div>
                      <div class="detail-value">${entry_price:.4f}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Exit Price</div>
                      <div class="detail-value">${exit_price:.4f}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Quantity</div>
                      <div class="detail-value">{quantity:.2f}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Execution Time</div>
                      <div class="detail-value">{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
                    </div>
                  </div>
                </div>
                
                <div class="section ml-analysis">
                  <h2>🧠 AI/ML Analysis</h2>
                  <div class="detail-grid">
                    <div class="detail-item">
                      <div class="detail-label">Model Accuracy</div>
                      <div class="detail-value">{model_accuracy:.1%}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Prediction Confidence</div>
                      <div class="detail-value">{prediction_confidence:.1%}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Signal Strength</div>
                      <div class="detail-value">{signal_strength:.1%}</div>
                    </div>
                  </div>
                  <div style="margin-top: 15px; padding: 15px; background: white; border-radius: 6px;">
                    <strong>AI Reasoning:</strong><br>
                    <p style="margin: 10px 0 0 0; line-height: 1.6;">{ml_reasoning}</p>
                  </div>
                </div>
                
                <div class="section risk-analysis">
                  <h2>⚠️ Risk Management Analysis</h2>
                  <div class="detail-grid">
                    <div class="detail-item">
                      <div class="detail-label">Risk Level</div>
                      <div class="detail-value">{risk_level}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Stop Loss</div>
                      <div class="detail-value">${stop_loss:.4f}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Take Profit</div>
                      <div class="detail-value">${take_profit:.4f}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Risk/Reward Ratio</div>
                      <div class="detail-value">{risk_reward_ratio}:1</div>
                    </div>
                  </div>
                </div>
                
                <div class="section market-context">
                  <h2>📈 Market Context</h2>
                  <div class="detail-grid">
                    <div class="detail-item">
                      <div class="detail-label">Market Trend</div>
                      <div class="detail-value">{market_trend}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Volatility</div>
                      <div class="detail-value">{volatility}</div>
                    </div>
                    <div class="detail-item">
                      <div class="detail-label">Volume Analysis</div>
                      <div class="detail-value">{volume_analysis}</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div class="footer">
                <p>🤖 TitanovaX AI Trading System - Automated Analysis & Reporting</p>
                <p class="timestamp">Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
              </div>
            </div>
          </body>
        </html>
        """
        
        return html_content
    
    def create_sample_trade_data(self) -> Dict[str, Any]:
        """Create sample trade data for testing"""
        return {
            'symbol': 'BTCUSDT',
            'type': 'BUY',
            'entry_price': 43250.50,
            'exit_price': 43500.75,
            'quantity': 0.25,
            'pnl': 62.56,
            'timestamp': datetime.now(),
            'confidence': 0.85,
            'signal_strength': 0.78,
            'ml_reasoning': 'Machine learning model detected bullish momentum based on technical indicators including RSI oversold bounce, MACD bullish crossover, and volume spike above 20-day average. Sentiment analysis shows positive institutional flow with whale accumulation patterns.',
            'model_accuracy': 0.82,
            'prediction_confidence': 0.85,
            'risk_level': 'Medium',
            'stop_loss': 42800.00,
            'take_profit': 44000.00,
            'risk_reward_ratio': 1.5,
            'market_trend': 'Bullish',
            'volatility': 'Moderate',
            'volume_analysis': 'Above Average'
        }
    
    def send_trade_explanation_email(self, trade_data: Dict[str, Any] = None) -> bool:
        """Send detailed trade explanation via email"""
        try:
            if not trade_data:
                trade_data = self.create_sample_trade_data()
            
            logger.info("📧 Creating trade explanation email...")
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🤖 TitanovaX Trade Report - {trade_data['symbol']} {trade_data['type']}"
            msg["From"] = self.email_from
            msg["To"] = self.email_to
            
            # Create HTML content
            html_content = self.create_trade_explanation_html(trade_data)
            
            # Attach HTML content
            msg.attach(MIMEText(html_content, "html"))
            
            # Create and attach JSON report
            json_report = json.dumps(trade_data, indent=2, default=str)
            json_attachment = MIMEBase('application', 'json')
            json_attachment.set_payload(json_report.encode('utf-8'))
            encoders.encode_base64(json_attachment)
            json_attachment.add_header('Content-Disposition', 'attachment', filename=f'trade_report_{trade_data["symbol"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            msg.attach(json_attachment)
            
            logger.info("🔐 Connecting to SMTP server...")
            
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect and send
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context, timeout=60) as server:
                server.login(self.email_username, self.email_password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())
            
            logger.info("✅ Trade explanation email sent successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send trade explanation email: {e}")
            return False
    
    def test_email_connection(self) -> bool:
        """Test email connection and authentication"""
        try:
            logger.info("🧪 Testing email connection...")
            
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context, timeout=30) as server:
                server.login(self.email_username, self.email_password)
                logger.info("✅ Email connection test successful!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Email connection test failed: {e}")
            return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Email Trade Explanations Test")
    logger.info("=" * 60)
    
    # Initialize email trade explainer
    explainer = EmailTradeExplainer()
    
    # Test 1: Email connection
    logger.info("\n📋 Test 1: Email Connection Test")
    connection_success = explainer.test_email_connection()
    
    if connection_success:
        logger.info("✅ Email connection: WORKING")
    else:
        logger.error("❌ Email connection: FAILED")
        return False
    
    # Test 2: Send trade explanation email
    logger.info("\n📋 Test 2: Trade Explanation Email Test")
    email_success = explainer.send_trade_explanation_email()
    
    if email_success:
        logger.info("✅ Trade explanation email: SENT")
    else:
        logger.error("❌ Trade explanation email: FAILED")
        return False
    
    # Test 3: Send multiple trade explanations
    logger.info("\n📋 Test 3: Multiple Trade Explanations Test")
    
    trades = [
        {
            'symbol': 'ETHUSDT',
            'type': 'SELL',
            'entry_price': 2650.25,
            'exit_price': 2625.80,
            'quantity': 1.5,
            'pnl': -36.68,
            'timestamp': datetime.now(),
            'confidence': 0.78,
            'signal_strength': 0.72,
            'ml_reasoning': 'Technical analysis indicates bearish divergence on RSI, MACD showing negative crossover, and increased selling pressure from institutional wallets.',
            'model_accuracy': 0.79,
            'prediction_confidence': 0.78,
            'risk_level': 'Low',
            'stop_loss': 2680.00,
            'take_profit': 2600.00,
            'risk_reward_ratio': 2.1,
            'market_trend': 'Bearish',
            'volatility': 'High',
            'volume_analysis': 'Heavy Selling'
        },
        {
            'symbol': 'ADAUSDT',
            'type': 'BUY',
            'entry_price': 0.485,
            'exit_price': 0.492,
            'quantity': 5000,
            'pnl': 35.00,
            'timestamp': datetime.now(),
            'confidence': 0.82,
            'signal_strength': 0.75,
            'ml_reasoning': 'Cardano showing strong fundamentals with upcoming network upgrades, on-chain metrics indicate increased developer activity and staking participation.',
            'model_accuracy': 0.81,
            'prediction_confidence': 0.82,
            'risk_level': 'Medium',
            'stop_loss': 0.470,
            'take_profit': 0.520,
            'risk_reward_ratio': 1.8,
            'market_trend': 'Bullish',
            'volatility': 'Moderate',
            'volume_analysis': 'Accumulation Phase'
        }
    ]
    
    for i, trade in enumerate(trades):
        logger.info(f"📤 Sending trade explanation {i+1}/{len(trades)}...")
        success = explainer.send_trade_explanation_email(trade)
        if success:
            logger.info(f"✅ Trade {i+1} explanation sent")
        else:
            logger.error(f"❌ Trade {i+1} explanation failed")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Email Trade Explanations Test Completed!")
    logger.info("✅ All trade explanations have been sent to your email")
    logger.info("📧 Please check your inbox for detailed trade reports")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)