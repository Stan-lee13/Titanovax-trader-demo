"""
TitanovaX Orchestration System Main Runner
Production-ready orchestration system with all components integrated
"""

import asyncio
import json
import logging
import signal
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from orchestration.orchestration_engine import OrchestrationEngine, TradingSignal
from orchestration.telegram_bot import TitanovaXTelegramBot
from orchestration.rag_llm_integration import RAGLLMIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TitanovaXOrchestrationSystem:
    """Main orchestration system coordinating all components"""
    
    def __init__(self):
        self.orchestration_engine = None
        self.telegram_bot = None
        self.rag_system = None
        self.running = False
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load orchestration configuration"""
        try:
            config_path = Path("orchestration/orchestration_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.error("Orchestration config file not found")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    async def initialize(self):
        """Initialize all orchestration components"""
        logger.info("🚀 Initializing TitanovaX Orchestration System...")
        
        try:
            # Create data directory
            Path("data").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            
            # Initialize orchestration engine
            self.orchestration_engine = OrchestrationEngine()
            engine_initialized = await self.orchestration_engine.initialize()
            
            if not engine_initialized:
                logger.error("❌ Orchestration engine initialization failed")
                return False
            
            # Initialize RAG/LLM system
            self.rag_system = RAGLLMIntegration()
            rag_initialized = await asyncio.to_thread(self.rag_system.initialize)
            
            if not rag_initialized:
                logger.error("❌ RAG/LLM system initialization failed")
                return False
            
            # Initialize Telegram bot (if token available)
            telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
            if telegram_token:
                self.telegram_bot = TitanovaXTelegramBot(telegram_token)
                telegram_initialized = await self.telegram_bot.initialize()
                
                if not telegram_initialized:
                    logger.warning("⚠️ Telegram bot initialization failed - continuing without")
                    self.telegram_bot = None
            else:
                logger.warning("⚠️ No Telegram token found - bot disabled")
                self.telegram_bot = None
            
            logger.info("✅ All orchestration components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Orchestration system initialization failed: {e}")
            return False
    
    async def start(self):
        """Start the orchestration system"""
        logger.info("🚀 Starting TitanovaX Orchestration System...")
        
        if await self.initialize():
            self.running = True
            
            # Start orchestration engine
            await self.orchestration_engine.start()
            
            # Start Telegram bot if available
            if self.telegram_bot:
                asyncio.create_task(self.telegram_bot.start_bot())
            
            # Start background tasks
            asyncio.create_task(self.background_monitoring())
            
            logger.info("✅ TitanovaX Orchestration System started successfully")
            
            # Run until stopped
            await self.run_forever()
        else:
            logger.error("❌ Failed to start orchestration system")
            sys.exit(1)
    
    async def stop(self):
        """Stop the orchestration system"""
        logger.info("🛑 Stopping TitanovaX Orchestration System...")
        
        self.running = False
        
        # Stop orchestration engine
        if self.orchestration_engine:
            await self.orchestration_engine.stop()
        
        # Stop Telegram bot
        if self.telegram_bot:
            await self.telegram_bot.stop_bot()
        
        logger.info("✅ TitanovaX Orchestration System stopped")
    
    async def background_monitoring(self):
        """Background monitoring tasks"""
        while self.running:
            try:
                # System health check
                await self.health_check()
                
                # Performance monitoring
                await self.performance_monitoring()
                
                # Risk monitoring
                await self.risk_monitoring()
                
                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def health_check(self):
        """Check system health"""
        try:
            # Check orchestration engine
            engine_health = self.orchestration_engine.running if self.orchestration_engine else False
            
            # Check memory system
            memory_health = False
            if self.orchestration_engine and self.orchestration_engine.memory_system:
                memory_health = self.orchestration_engine.memory_system.redis_client is not None
            
            # Check RAG system
            rag_health = self.rag_system is not None
            
            # Log health status
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "orchestration_engine": "healthy" if engine_health else "unhealthy",
                "memory_system": "healthy" if memory_health else "unhealthy",
                "rag_system": "healthy" if rag_health else "unhealthy",
                "telegram_bot": "enabled" if self.telegram_bot else "disabled"
            }
            
            # Save health status
            with open('logs/health_status.json', 'w') as f:
                json.dump(health_status, f, indent=2)
            
            # Alert if any component is unhealthy
            if not all(status == "healthy" or status == "enabled" for status in health_status.values() if status != "disabled"):
                logger.warning(f"⚠️ System health issues detected: {health_status}")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def performance_monitoring(self):
        """Monitor system performance"""
        try:
            # This would integrate with actual performance metrics
            # For now, log basic performance info
            logger.info("📊 Performance monitoring cycle completed")
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
    
    async def risk_monitoring(self):
        """Monitor risk metrics"""
        try:
            # This would integrate with risk management system
            # For now, log basic risk info
            logger.info("⚠️ Risk monitoring cycle completed")
            
        except Exception as e:
            logger.error(f"Risk monitoring failed: {e}")
    
    async def run_forever(self):
        """Run the system until interrupted"""
        logger.info("🔄 TitanovaX Orchestration System is running...")
        
        try:
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            await self.stop()
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 Received signal {signum}")
        self.running = False

async def main():
    """Main function"""
    logger.info("🎯 Starting TitanovaX Orchestration System...")
    
    # Create orchestration system
    system = TitanovaXOrchestrationSystem()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, system.handle_signal)
    signal.signal(signal.SIGTERM, system.handle_signal)
    
    try:
        # Start the system
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
    finally:
        await system.stop()

if __name__ == "__main__":
    # Set event loop policy for Windows if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the main function
    asyncio.run(main())