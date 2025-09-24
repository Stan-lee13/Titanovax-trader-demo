#!/usr/bin/env python3
"""
Signal Processing Module for TitanovaX Trading System
Advanced signal processing and feature engineering for trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import zscore
import logging
from pathlib import Path
import json

@dataclass
class SignalMetrics:
    """Signal quality metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

@dataclass
class TradingSignal:
    """Processed trading signal"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    strength: float
    features: Dict[str, float]
    metadata: Dict[str, Any]

class SignalProcessor:
    """
    Advanced Signal Processing for Trading Signals
    """

    def __init__(self, config_path: str = 'config/signal_config.json'):
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        self.signal_history: List[TradingSignal] = []
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.performance_metrics: Dict[str, SignalMetrics] = {}
        
        self.load_config()
        self.setup_logging()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'moving_averages': [5, 10, 20, 50, 200],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2,
            'stochastic_period': 14,
            'volume_sma_period': 20,
            'volatility_window': 20,
            'correlation_window': 50,
            'signal_threshold': 0.6,
            'confidence_threshold': 0.7,
            'feature_scaling': 'standard',
            'noise_reduction': True,
            'adaptive_filtering': True
        }

    def load_config(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            else:
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load signal config: {e}")

    def save_config(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def process_market_data(self, market_data: Dict[str, Any]) -> TradingSignal:
        """
        Process market data into trading signal
        
        Args:
            market_data: Market data with OHLCV information
            
        Returns:
            Processed trading signal
        """
        try:
            # Extract price and volume data
            prices = np.array(market_data.get('close_prices', []))
            volumes = np.array(market_data.get('volumes', []))
            highs = np.array(market_data.get('high_prices', prices))
            lows = np.array(market_data.get('low_prices', prices))
            
            if len(prices) < max(self.config['moving_averages']):
                return self._create_hold_signal(market_data)
            
            # Calculate technical indicators
            features = self._calculate_features(prices, volumes, highs, lows)
            
            # Apply signal processing techniques
            processed_features = self._apply_signal_processing(features)
            
            # Generate signal
            signal_type, confidence, strength = self._generate_signal(processed_features)
            
            # Create trading signal
            trading_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=market_data.get('symbol', 'UNKNOWN'),
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                features=processed_features,
                metadata={
                    'market_data': market_data,
                    'processing_timestamp': datetime.now().isoformat()
                }
            )
            
            # Store in history
            self.signal_history.append(trading_signal)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return self._create_hold_signal(market_data)

    def _calculate_features(self, prices: np.ndarray, volumes: np.ndarray, 
                          highs: np.ndarray, lows: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicator features"""
        features = {}
        
        # Price-based features
        for period in self.config['moving_averages']:
            if len(prices) >= period:
                features[f'ma_{period}'] = np.mean(prices[-period:])
                features[f'ma_ratio_{period}'] = prices[-1] / features[f'ma_{period}']
        
        # RSI
        features['rsi'] = self._calculate_rsi(prices)
        
        # MACD
        macd_line, signal_line = self._calculate_macd(prices)
        features['macd'] = macd_line[-1] if len(macd_line) > 0 else 0
        features['macd_signal'] = signal_line[-1] if len(signal_line) > 0 else 0
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
        features['bb_upper'] = bb_upper[-1] if len(bb_upper) > 0 else prices[-1]
        features['bb_lower'] = bb_lower[-1] if len(bb_lower) > 0 else prices[-1]
        features['bb_position'] = (prices[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Stochastic
        features['stochastic'] = self._calculate_stochastic(highs, lows, prices)
        
        # Volume features
        if len(volumes) >= self.config['volume_sma_period']:
            features['volume_sma'] = np.mean(volumes[-self.config['volume_sma_period']:])
            features['volume_ratio'] = volumes[-1] / features['volume_sma']
        
        # Volatility
        features['volatility'] = np.std(prices[-self.config['volatility_window']:]) / np.mean(prices[-self.config['volatility_window']:])
        
        # Price momentum
        features['price_momentum'] = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        features['price_change_24h'] = (prices[-1] - prices[-24]) / prices[-24] if len(prices) >= 24 else 0
        
        return features

    def _apply_signal_processing(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply advanced signal processing techniques"""
        processed = features.copy()
        
        # Feature scaling
        if self.config['feature_scaling'] == 'standard':
            processed = self._standard_scale_features(processed)
        elif self.config['feature_scaling'] == 'minmax':
            processed = self._minmax_scale_features(processed)
        
        # Noise reduction
        if self.config['noise_reduction']:
            processed = self._apply_noise_reduction(processed)
        
        # Adaptive filtering
        if self.config['adaptive_filtering']:
            processed = self._apply_adaptive_filtering(processed)
        
        return processed

    def _generate_signal(self, features: Dict[str, float]) -> Tuple[str, float, float]:
        """Generate trading signal from processed features"""
        # Simple rule-based signal generation
        # In production, this would use ML models
        
        buy_score = 0
        sell_score = 0
        
        # RSI-based signals
        rsi = features.get('rsi', 50)
        if rsi < 30:
            buy_score += 2
        elif rsi > 70:
            sell_score += 2
        
        # MACD signals
        macd_histogram = features.get('macd_histogram', 0)
        if macd_histogram > 0:
            buy_score += 1
        else:
            sell_score += 1
        
        # Bollinger band signals
        bb_position = features.get('bb_position', 0.5)
        if bb_position < 0.2:
            buy_score += 1
        elif bb_position > 0.8:
            sell_score += 1
        
        # Stochastic signals
        stochastic = features.get('stochastic', 50)
        if stochastic < 20:
            buy_score += 1
        elif stochastic > 80:
            sell_score += 1
        
        # Volume confirmation
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.2:
            # Volume supports the signal
            if buy_score > sell_score:
                buy_score += 1
            elif sell_score > buy_score:
                sell_score += 1
        
        # Determine final signal
        total_score = buy_score + sell_score
        if total_score == 0:
            return 'HOLD', 0.5, 0.0
        
        confidence = max(buy_score, sell_score) / total_score
        
        if buy_score > sell_score and confidence >= self.config['signal_threshold']:
            return 'BUY', confidence, (buy_score - sell_score) / total_score
        elif sell_score > buy_score and confidence >= self.config['signal_threshold']:
            return 'SELL', confidence, (sell_score - buy_score) / total_score
        else:
            return 'HOLD', confidence, abs(buy_score - sell_score) / total_score

    def _calculate_rsi(self, prices: np.ndarray, period: int = None) -> float:
        """Calculate RSI indicator"""
        if period is None:
            period = self.config['rsi_period']
        
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        exp1 = pd.Series(prices).ewm(span=self.config['macd_fast']).mean()
        exp2 = pd.Series(prices).ewm(span=self.config['macd_slow']).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
        return macd_line.values, signal_line.values

    def _calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=self.config['bollinger_period']).mean()
        std = pd.Series(prices).rolling(window=self.config['bollinger_period']).std()
        upper_band = sma + (std * self.config['bollinger_std'])
        lower_band = sma - (std * self.config['bollinger_std'])
        return upper_band.values, lower_band.values

    def _calculate_stochastic(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate Stochastic indicator"""
        period = self.config['stochastic_period']
        if len(highs) < period:
            return 50.0
        
        high_max = np.max(highs[-period:])
        low_min = np.min(lows[-period:])
        
        if high_max == low_min:
            return 50.0
        
        stochastic = 100 * (closes[-1] - low_min) / (high_max - low_min)
        return stochastic

    def _standard_scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply standard scaling to features"""
        # Simple scaling based on historical values
        scaled = {}
        for key, value in features.items():
            # Use z-score normalization with moving statistics
            if key in self.feature_cache:
                historical_values = self.feature_cache[key]
                if len(historical_values) > 1:
                    mean = np.mean(historical_values)
                    std = np.std(historical_values)
                    if std > 0:
                        scaled[key] = (value - mean) / std
                    else:
                        scaled[key] = value
                else:
                    scaled[key] = value
            else:
                scaled[key] = value
        
        return scaled

    def _minmax_scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply min-max scaling to features"""
        # Simple min-max scaling
        scaled = {}
        for key, value in features.items():
            if key in self.feature_cache:
                historical_values = self.feature_cache[key]
                if len(historical_values) > 1:
                    min_val = np.min(historical_values)
                    max_val = np.max(historical_values)
                    if max_val > min_val:
                        scaled[key] = (value - min_val) / (max_val - min_val)
                    else:
                        scaled[key] = value
                else:
                    scaled[key] = value
            else:
                scaled[key] = value
        
        return scaled

    def _apply_noise_reduction(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply noise reduction to features"""
        # Simple smoothing based on recent values
        smoothed = {}
        for key, value in features.items():
            if key in self.feature_cache:
                historical_values = self.feature_cache[key]
                if len(historical_values) >= 3:
                    # Simple moving average smoothing
                    recent_values = historical_values[-3:]
                    smoothed[key] = np.mean(recent_values)
                else:
                    smoothed[key] = value
            else:
                smoothed[key] = value
        
        return smoothed

    def _apply_adaptive_filtering(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply adaptive filtering to features"""
        # Simple adaptive filtering based on recent volatility
        filtered = {}
        for key, value in features.items():
            if key in self.feature_cache:
                historical_values = self.feature_cache[key]
                if len(historical_values) >= 10:
                    # Adaptive filter based on recent volatility
                    recent_values = historical_values[-10:]
                    volatility = np.std(recent_values)
                    mean_val = np.mean(recent_values)
                    
                    # Adjust filtering based on volatility
                    if volatility > 0:
                        alpha = min(0.9, 1.0 - (volatility / (abs(mean_val) + 1e-6)))
                        filtered[key] = alpha * value + (1 - alpha) * mean_val
                    else:
                        filtered[key] = value
                else:
                    filtered[key] = value
            else:
                filtered[key] = value
        
        return filtered

    def _create_hold_signal(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Create a default HOLD signal"""
        return TradingSignal(
            timestamp=datetime.now(),
            symbol=market_data.get('symbol', 'UNKNOWN'),
            signal_type='HOLD',
            confidence=0.5,
            strength=0.0,
            features={},
            metadata={'reason': 'Insufficient data for signal generation'}
        )

    def get_signal_performance(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Get signal performance metrics"""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_signals = [s for s in self.signal_history if s.timestamp > cutoff_date]
        
        if not recent_signals:
            return {'error': 'No signals in lookback period'}
        
        # Calculate basic metrics
        total_signals = len(recent_signals)
        buy_signals = len([s for s in recent_signals if s.signal_type == 'BUY'])
        sell_signals = len([s for s in recent_signals if s.signal_type == 'SELL'])
        hold_signals = len([s for s in recent_signals if s.signal_type == 'HOLD'])
        
        avg_confidence = np.mean([s.confidence for s in recent_signals])
        avg_strength = np.mean([s.strength for s in recent_signals])
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_confidence': avg_confidence,
            'avg_strength': avg_strength,
            'signal_distribution': {
                'buy_pct': buy_signals / total_signals * 100,
                'sell_pct': sell_signals / total_signals * 100,
                'hold_pct': hold_signals / total_signals * 100
            }
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance analysis"""
        # Simple correlation-based importance
        # In production, this would use more sophisticated methods
        
        importance = {}
        
        # Analyze recent signals
        if len(self.signal_history) >= 50:
            recent_signals = self.signal_history[-50:]
            
            # Calculate feature correlations with signal strength
            for feature_name in ['rsi', 'macd_histogram', 'bb_position', 'stochastic', 'volume_ratio']:
                feature_values = []
                strengths = []
                
                for signal in recent_signals:
                    if feature_name in signal.features:
                        feature_values.append(signal.features[feature_name])
                        strengths.append(signal.strength)
                
                if len(feature_values) > 10:
                    correlation = np.corrcoef(feature_values, strengths)[0, 1]
                    importance[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return importance