#!/usr/bin/env python3
"""
Signal Processor Module for TitanovaX Trading System
Wrapper for signal processing functionality
"""

from signal_processing import SignalProcessor as CoreSignalProcessor, TradingSignal
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

class SignalProcessor:
    """
    Signal Processor Wrapper for Orchestration Integration
    """
    
    def __init__(self, config_path: str = 'config/signal_config.json'):
        """Initialize signal processor"""
        self.processor = CoreSignalProcessor(config_path)
        self.logger = logging.getLogger(__name__)
        
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data and return signal
        
        Args:
            market_data: Market data with OHLCV information
            
        Returns:
            Processed signal as dictionary
        """
        try:
            signal = self.processor.process_market_data(market_data)
            return {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'strength': signal.strength,
                'features': signal.features,
                'metadata': signal.metadata
            }
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': market_data.get('symbol', 'UNKNOWN'),
                'signal_type': 'HOLD',
                'confidence': 0.5,
                'strength': 0.0,
                'features': {},
                'metadata': {'error': str(e)}
            }
    
    def get_signal_performance(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Get signal performance metrics"""
        return self.processor.get_signal_performance(lookback_days)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance analysis"""
        return self.processor.get_feature_importance()