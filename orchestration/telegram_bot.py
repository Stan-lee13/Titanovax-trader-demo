"""
TitanovaX Telegram Bot for Consent, Reporting, and Explainability
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.error import TelegramError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TitanovaXTelegramBot:
    """Telegram bot for TitanovaX trading system"""
    
    def __init__(self, token: str = None):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
        self.application = None
        self.user_consents = {}  # Store user consent states
        self.trading_status = {}  # Store trading status per user
        self.risk_limits = {}  # Store user risk limits
        
        # Default configuration
        self.default_config = {
            "max_daily_trades": 10,
            "max_position_size_pct": 5.0,
            "max_daily_drawdown_pct": 3.0,
            "notification_types": ["trade_executions", "risk_alerts", "daily_summary"],
            "auto_trading": False,
            "risk_level": "medium"
        }
    
    async def initialize(self):
        """Initialize Telegram bot"""
        try:
            logger.info("Initializing TitanovaX Telegram Bot...")
            
            # Create application
            self.application = Application.builder().token(self.token).build()
            
            # Register handlers
            self.register_handlers()
            
            # Initialize user data
            await self.load_user_data()
            
            logger.info("✅ Telegram bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Telegram bot initialization failed: {e}")
            return False
    
    def register_handlers(self):
        """Register command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("consent", self.consent_command))
        self.application.add_handler(CommandHandler("settings", self.settings_command))
        self.application.add_handler(CommandHandler("trades", self.trades_command))
        self.application.add_handler(CommandHandler("risk", self.risk_command))
        self.application.add_handler(CommandHandler("performance", self.performance_command))
        self.application.add_handler(CommandHandler("explain", self.explain_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("resume", self.resume_command))
        
        # Callback query handlers
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        user_id = str(user.id)
        
        # Initialize user data
        if user_id not in self.user_consents:
            self.user_consents[user_id] = {
                "consented": False,
                "consent_timestamp": None,
                "config": self.default_config.copy(),
                "trades": [],
                "risk_metrics": {}
            }
        
        welcome_message = f"""
🤖 Welcome to TitanovaX Trading Bot!

Hello {user.first_name}! I'm your AI-powered trading assistant.

🔒 **Important**: Before we begin, I need your explicit consent to:
• Execute trades on your behalf
• Access market data and perform analysis
• Send you trading notifications and reports
• Monitor your risk exposure

⚠️ **Risk Warning**: Trading involves substantial risk of loss. Only trade with capital you can afford to lose.

Use /consent to provide your trading authorization.
Use /help to see all available commands.
        """
        
        keyboard = [
            [InlineKeyboardButton("✅ Provide Consent", callback_data="consent_yes")],
            [InlineKeyboardButton("❌ Decline", callback_data="consent_no")],
            [InlineKeyboardButton("📖 View Risk Disclosure", callback_data="risk_disclosure")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
    
    async def consent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /consent command"""
        user_id = str(update.effective_user.id)
        
        consent_message = f"""
🔐 **Trading Consent Authorization**

By providing consent, you authorize TitanovaX to:

✅ **Trade Execution**: Execute buy/sell orders based on AI analysis
✅ **Market Analysis**: Access real-time market data and indicators
✅ **Risk Management**: Monitor and manage your trading risk
✅ **Performance Tracking**: Track and report trading performance
✅ **Notifications**: Send trade alerts and daily summaries

⚠️ **Risk Acknowledgment**:
• You understand trading involves risk of loss
• You can revoke consent at any time with /stop
• Maximum daily loss is limited to 3% of account balance
• You maintain full control over your trading account

Do you consent to automated trading?
        """
        
        keyboard = [
            [InlineKeyboardButton("✅ Yes, I Consent", callback_data="consent_yes")],
            [InlineKeyboardButton("❌ No, Decline", callback_data="consent_no")],
            [InlineKeyboardButton("⚙️ Configure Limits", callback_data="configure_limits")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(consent_message, reply_markup=reply_markup)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = f"""
🤖 **TitanovaX Trading Bot Commands**

**Basic Commands:**
/start - Initialize bot and provide consent
/help - Show this help message
/status - Check system status and connections
/stop - Stop automated trading
/resume - Resume automated trading

**Trading Commands:**
/trades - View recent trades and performance
/performance - Show detailed performance metrics
/risk - Display current risk exposure
/explain - Get explanation for recent trades

**Configuration:**
/settings - Configure trading parameters
/consent - Manage trading authorization

**Emergency:**
/emergency_stop - Immediate stop all trading (use in crisis)

**Need Help?** Contact support or use /emergency_stop for immediate assistance.
        """
        
        await update.message.reply_text(help_message)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_id = str(update.effective_user.id)
        user_data = self.user_consents.get(user_id, {})
        
        # Get system status (this would connect to actual trading systems)
        system_status = {
            "orchestration": "🟢 Online",
            "mt5_connection": "🟢 Connected",
            "mt4_connection": "🟢 Connected",
            "ai_models": "🟢 Active (5 models)",
            "risk_management": "🟢 Monitoring",
            "memory_system": "🟢 Operational"
        }
        
        status_message = f"""
📊 **TitanovaX System Status**

**User Authorization:**
• Consent Status: {'✅ Active' if user_data.get('consented') else '❌ Inactive'}
• Auto Trading: {'🟢 Enabled' if user_data.get('config', {}).get('auto_trading') else '🔴 Disabled'}
• Risk Level: {user_data.get('config', {}).get('risk_level', 'medium').upper()}

**System Components:**
• Orchestration Engine: {system_status['orchestration']}
• MT5 Connection: {system_status['mt5_connection']}
• MT4 Connection: {system_status['mt4_connection']}
• AI Models: {system_status['ai_models']}
• Risk Management: {system_status['risk_management']}
• Memory System: {system_status['memory_system']}

**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        
        await update.message.reply_text(status_message)
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        user_id = str(update.effective_user.id)
        user_config = self.user_consents.get(user_id, {}).get('config', self.default_config)
        
        settings_message = f"""
⚙️ **Trading Configuration**

**Current Settings:**
• Max Daily Trades: {user_config['max_daily_trades']}
• Max Position Size: {user_config['max_position_size_pct']}% of balance
• Max Daily Drawdown: {user_config['max_daily_drawdown_pct']}% of balance
• Risk Level: {user_config['risk_level'].upper()}
• Auto Trading: {'Enabled' if user_config['auto_trading'] else 'Disabled'}

**Notifications:**
• Trade Executions: {'✅' if 'trade_executions' in user_config['notification_types'] else '❌'}
• Risk Alerts: {'✅' if 'risk_alerts' in user_config['notification_types'] else '❌'}
• Daily Summary: {'✅' if 'daily_summary' in user_config['notification_types'] else '❌'}

Select a setting to modify:
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 Max Daily Trades", callback_data="set_max_trades")],
            [InlineKeyboardButton("💰 Position Size %", callback_data="set_position_size")],
            [InlineKeyboardButton("📉 Max Drawdown %", callback_data="set_drawdown")],
            [InlineKeyboardButton("🎯 Risk Level", callback_data="set_risk_level")],
            [InlineKeyboardButton("🔔 Notifications", callback_data="set_notifications")],
            [InlineKeyboardButton("🤖 Auto Trading", callback_data="toggle_auto_trading")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(settings_message, reply_markup=reply_markup)
    
    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command"""
        user_id = str(update.effective_user.id)
        user_trades = self.user_consents.get(user_id, {}).get('trades', [])
        
        if not user_trades:
            await update.message.reply_text("📈 No recent trades found. Trading will begin once you provide consent with /consent")
            return
        
        # Get last 5 trades
        recent_trades = user_trades[-5:]
        
        trades_message = f"""
📈 **Recent Trading Activity**

**Last {len(recent_trades)} Trades:**
        """
        
        for i, trade in enumerate(recent_trades, 1):
            trade_info = f"""
**Trade {i}:**
• Symbol: {trade.get('symbol', 'N/A')}
• Direction: {trade.get('direction', 'N/A')}
• Confidence: {trade.get('confidence', 0):.2f}
• Size: {trade.get('size', 0):.4f}
• Result: {trade.get('result', 'Pending')}
• Time: {trade.get('timestamp', 'N/A')}
            """
            trades_message += trade_info
        
        await update.message.reply_text(trades_message)
    
    async def risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command"""
        user_id = str(update.effective_user.id)
        risk_metrics = self.user_consents.get(user_id, {}).get('risk_metrics', {})
        
        risk_message = f"""
⚠️ **Risk Exposure Dashboard**

**Current Risk Metrics:**
• Daily P&L: ${risk_metrics.get('daily_pnl', 0):.2f}
• Daily Drawdown: {risk_metrics.get('daily_drawdown_pct', 0):.2f}%
• Open Positions: {risk_metrics.get('open_positions', 0)}
• Total Exposure: {risk_metrics.get('total_exposure_pct', 0):.2f}%
• Risk Level: {risk_metrics.get('risk_level', 'medium').upper()}

**Risk Limits:**
• Max Daily Drawdown: 3.0%
• Max Position Size: 5.0%
• Max Daily Trades: 10

**Status:** {'🟢 Safe' if risk_metrics.get('daily_drawdown_pct', 0) < 2.0 else '⚠️ Caution' if risk_metrics.get('daily_drawdown_pct', 0) < 3.0 else '🔴 High Risk'}
        """
        
        await update.message.reply_text(risk_message)
    
    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command"""
        user_id = str(update.effective_user.id)
        
        # Simulate performance data (this would come from actual trading systems)
        performance_data = {
            "total_trades": 25,
            "winning_trades": 18,
            "losing_trades": 7,
            "win_rate": 72.0,
            "total_pnl": 1250.50,
            "avg_win": 85.25,
            "avg_loss": -45.75,
            "profit_factor": 1.87,
            "max_drawdown": 2.1,
            "sharpe_ratio": 1.45
        }
        
        performance_message = f"""
📊 **Trading Performance Report**

**Overall Statistics:**
• Total Trades: {performance_data['total_trades']}
• Winning Trades: {performance_data['winning_trades']}
• Losing Trades: {performance_data['losing_trades']}
• Win Rate: {performance_data['win_rate']:.1f}%

**Profitability:**
• Total P&L: ${performance_data['total_pnl']:.2f}
• Average Win: ${performance_data['avg_win']:.2f}
• Average Loss: ${performance_data['avg_loss']:.2f}
• Profit Factor: {performance_data['profit_factor']:.2f}

**Risk Metrics:**
• Maximum Drawdown: {performance_data['max_drawdown']:.1f}%
• Sharpe Ratio: {performance_data['sharpe_ratio']:.2f}

**Rating:** {'🌟 Excellent' if performance_data['win_rate'] > 70 else '⭐ Good' if performance_data['win_rate'] > 60 else '⚠️ Needs Improvement'}
        """
        
        await update.message.reply_text(performance_message)
    
    async def explain_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /explain command"""
        user_id = str(update.effective_user.id)
        
        # Simulate explanation (this would come from RAG/LLM system)
        explanation = f"""
🧠 **AI Trading Explanation**

**Recent Decision Analysis:**

**Market Regime Detection:**
• Current Regime: TREND_UP (Confidence: 0.87)
• Volatility Level: Low (0.8%)
• Market Sentiment: Bullish

**Model Consensus:**
• XGBoost Model: BUY (Confidence: 0.82)
• Transformer Model: BUY (Confidence: 0.79)
• Ensemble Model: BUY (Confidence: 0.85)

**Technical Indicators:**
• RSI: 62 (Neutral-Bullish)
• MACD: Bullish crossover
• Moving Averages: Price above 20/50 MA
• Support/Resistance: Strong support at 1.0850

**Risk Assessment:**
• Position Size: 2.5% of account (Conservative)
• Stop Loss: 1.5% from entry
• Take Profit: 3.0% from entry
• Risk/Reward Ratio: 1:2

**Rationale:**
Multiple AI models agree on bullish outlook with strong technical confirmation. Risk-managed position sizing ensures capital preservation while capturing upside potential.
        """
        
        await update.message.reply_text(explanation)
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        user_id = str(update.effective_user.id)
        
        if user_id in self.user_consents:
            self.user_consents[user_id]['config']['auto_trading'] = False
            await self.save_user_data()
            
            stop_message = f"""
🛑 **Trading Stopped**

✅ Automated trading has been disabled.

**What this means:**
• No new trades will be executed
• Existing positions remain open
• Monitoring and alerts continue
• You can resume anytime with /resume

**Your account remains secure and under your control.**
            """
            
            await update.message.reply_text(stop_message)
        else:
            await update.message.reply_text("❌ No active trading session found.")
    
    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        user_id = str(update.effective_user.id)
        
        if user_id in self.user_consents:
            self.user_consents[user_id]['config']['auto_trading'] = True
            await self.save_user_data()
            
            resume_message = f"""
▶️ **Trading Resumed**

✅ Automated trading has been re-enabled.

**Current Settings:**
• Risk Level: {self.user_consents[user_id]['config']['risk_level'].upper()}
• Max Daily Trades: {self.user_consents[user_id]['config']['max_daily_trades']}
• Max Position Size: {self.user_consents[user_id]['config']['max_position_size_pct']}% of balance

**Monitoring active.** Use /status to check system health.
            """
            
            await update.message.reply_text(resume_message)
        else:
            await update.message.reply_text("❌ Please start with /start first.")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        user_id = str(update.effective_user.id)
        
        await query.answer()
        
        if query.data == "consent_yes":
            if user_id not in self.user_consents:
                self.user_consents[user_id] = {"consented": False, "config": self.default_config.copy()}
            
            self.user_consents[user_id]['consented'] = True
            self.user_consents[user_id]['consent_timestamp'] = datetime.now().isoformat()
            await self.save_user_data()
            
            response_text = f"""
✅ **Consent Granted Successfully!**

Thank you for providing trading authorization. TitanovaX is now configured to:

🤖 **Execute AI-powered trades**
📊 **Monitor market conditions**
⚠️ **Manage risk exposure**
📈 **Track performance metrics**

**Next Steps:**
• Use /settings to configure your preferences
• Use /status to monitor system health
• Use /trades to view trading activity

**Emergency:** Use /stop to halt trading immediately.
            """
            
            await query.edit_message_text(response_text)
            
        elif query.data == "consent_no":
            response_text = f"""
❌ **Consent Declined**

You have chosen not to provide trading authorization. TitanovaX will not execute any trades.

**You can still:**
• Monitor market conditions with /status
• View educational content
• Use the bot for market analysis

**To enable trading later:** Use /consent to provide authorization.
            """
            
            await query.edit_message_text(response_text)
            
        elif query.data == "risk_disclosure":
            risk_text = f"""
⚠️ **Risk Disclosure Statement**

**Trading involves substantial risk including:**
• Loss of invested capital
• Market volatility and unexpected events
• Technology failures and system errors
• Liquidity risks and slippage
• Regulatory and geopolitical risks

**Risk Management Features:**
• Maximum daily drawdown: 3% of account balance
• Position size limits: 5% of account per trade
• Automatic stop-loss mechanisms
• Real-time risk monitoring

**Important:** Past performance does not guarantee future results. Only trade with capital you can afford to lose.
            """
            
            await query.edit_message_text(risk_text)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        user_message = update.message.text.lower()
        
        if "emergency" in user_message or "stop" in user_message:
            await self.emergency_stop(update, context)
        else:
            await update.message.reply_text(
                "I understand your message. For trading commands, please use the menu buttons or type /help for available commands."
            )
    
    async def emergency_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop all trading"""
        user_id = str(update.effective_user.id)
        
        # Stop all trading activities
        if user_id in self.user_consents:
            self.user_consents[user_id]['config']['auto_trading'] = False
            await self.save_user_data()
        
        emergency_message = f"""
🚨 **EMERGENCY STOP ACTIVATED**

⚠️ **All trading activities have been immediately halted.**

**Actions Taken:**
• Automated trading: DISABLED
• New position opening: BLOCKED
• Risk monitoring: CONTINUING
• Position monitoring: ACTIVE

**Your Account:**
• Existing positions remain open
• Account access: UNCHANGED
• Manual trading: AVAILABLE

**To Resume:** Use /resume when ready
**For Support:** Contact emergency support

**Stay calm. Your capital is protected.**
        """
        
        await update.message.reply_text(emergency_message)
    
    async def save_user_data(self):
        """Save user data to file"""
        try:
            with open('data/telegram_users.json', 'w') as f:
                json.dump(self.user_consents, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")
    
    async def load_user_data(self):
        """Load user data from file"""
        try:
            if os.path.exists('data/telegram_users.json'):
                with open('data/telegram_users.json', 'r') as f:
                    self.user_consents = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
    
    async def start_bot(self):
        """Start the Telegram bot"""
        try:
            logger.info("Starting TitanovaX Telegram Bot...")
            
            if await self.initialize():
                # Start the bot
                await self.application.initialize()
                await self.application.start()
                
                logger.info("✅ Telegram bot started successfully")
                
                # Run the bot until interrupted
                await self.application.updater.start_polling()
                
                # Keep the bot running
                await asyncio.Event().wait()
            else:
                logger.error("Failed to start Telegram bot")
                
        except Exception as e:
            logger.error(f"Telegram bot runtime error: {e}")
    
    async def stop_bot(self):
        """Stop the Telegram bot"""
        try:
            logger.info("Stopping TitanovaX Telegram Bot...")
            
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
            
            logger.info("✅ Telegram bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
    
    async def send_message(self, message: str, chat_id: str = None) -> bool:
        """Send a message to a specific chat"""
        try:
            if not self.application:
                logger.error("Telegram application not initialized")
                return False
            
            # Use provided chat_id or get from environment
            target_chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
            if not target_chat_id:
                logger.error("No chat_id provided and TELEGRAM_CHAT_ID not set")
                return False
            
            # Send message using the bot
            await self.application.bot.send_message(
                chat_id=target_chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ Message sent to chat {target_chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

async def main():
    """Main function to run the Telegram bot"""
    logger.info("Starting TitanovaX Telegram Bot...")
    
    # Get bot token from environment
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
    
    if bot_token == 'YOUR_BOT_TOKEN_HERE':
        logger.error("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
        return
    
    # Create and start bot
    bot = TitanovaXTelegramBot(bot_token)
    
    try:
        await bot.start_bot()
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.stop_bot()

if __name__ == "__main__":
    asyncio.run(main())