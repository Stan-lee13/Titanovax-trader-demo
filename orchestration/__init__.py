"""
TitanovaX Orchestration Package
Production-ready orchestration system for automated trading
"""

from .orchestration_engine import OrchestrationEngine, TradingSignal
from .telegram_bot import TitanovaXTelegramBot
from .rag_llm_integration import RAGLLMIntegration

__version__ = "1.0.0"
__all__ = [
    "OrchestrationEngine",
    "TradingSignal", 
    "TitanovaXTelegramBot",
    "RAGLLMIntegration"
]