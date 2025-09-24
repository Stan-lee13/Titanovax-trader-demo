#!/usr/bin/env python3
"""
Regime-Aware Orchestration for TitanovaX
Real-time regime classifier that outputs: TREND / RANGE / VOLATILE / CRISIS
All strategy modules consult this before action
"""

import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import pandas as pd

class MarketRegime(Enum):
    """Market regime classifications"""
    TREND = "trend"
    RANGE = "range"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    UNKNOWN = "unknown"

@dataclass
class RegimeFeatures:
    """Features used for regime classification"""
    symbol: str
    timestamp: datetime

    # Price action features
    price_change_1m: float
    price_change_5m: float
    price_change_15m: float
    volatility_1m: float
    volatility_5m: float
    volatility_15m: float

    # Volume features
    volume_ratio_1m: float
    volume_ratio_5m: float
    volume_spike: bool

    # Technical indicators
    rsi_14: float
    bb_width: float
    adx: float

    # News and external factors
    news_impact: float
    market_hours: bool  # True if during main market hours

@dataclass
class RegimePrediction:
    """Regime prediction result"""
    symbol: str
    timestamp: datetime
    predicted_regime: MarketRegime
    confidence: float
    regime_scores: Dict[MarketRegime, float]
    features: RegimeFeatures
    reasoning: str

class RegimeClassifier:
    """Real-time regime classifier using ensemble of methods"""

    def __init__(self, config_path: str = 'config/regime_config.json'):
        self.config_path = Path(config_path)
        self.regime_history: Dict[str, deque] = {}
        self.feature_history: Dict[str, deque] = {}
        self.max_history = 500

        # Regime transition tracking
        self.regime_transitions: Dict[str, List[Tuple[MarketRegime, datetime]]] = {}

        self.load_config()
        self.setup_logging()

        # Initialize regime thresholds
        self._initialize_thresholds()

    def load_config(self):
        """Load regime classifier configuration"""
        default_config = {
            "regime_thresholds": {
                "trend_min_adx": 25,
                "range_max_adx": 20,
                "range_max_bb_width": 0.02,
                "volatile_min_volatility": 0.005,
                "crisis_min_volatility": 0.01,
                "crisis_volume_spike_threshold": 3.0,
                "news_impact_threshold": 0.3
            },
            "confidence_weights": {
                "price_action": 0.4,
                "technical_indicators": 0.3,
                "volume_analysis": 0.2,
                "external_factors": 0.1
            },
            "history_length_minutes": 60,
            "update_frequency_seconds": 30
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load regime config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def _initialize_thresholds(self):
        """Initialize regime classification thresholds"""
        self.thresholds = self.config["regime_thresholds"]

    def extract_regime_features(self, symbol: str, market_data: Dict) -> RegimeFeatures:
        """
        Extract features for regime classification from market data

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            market_data: Dict containing OHLCV and indicator data
        """

        # Extract basic price data
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])

        if not prices or len(prices) < 15:
            return self._create_empty_features(symbol)

        current_price = prices[-1]
        prev_1m_price = prices[-1] if len(prices) < 1 else prices[-1]
        prev_5m_price = prices[-5] if len(prices) < 5 else prices[-5]
        prev_15m_price = prices[-15] if len(prices) < 15 else prices[-15]

        # Calculate price changes (in pips for forex)
        price_change_1m = (current_price - prev_1m_price) * 10000
        price_change_5m = (current_price - prev_5m_price) * 10000
        price_change_15m = (current_price - prev_15m_price) * 10000

        # Calculate volatility (standard deviation of price changes)
        recent_prices = prices[-15:] if len(prices) >= 15 else prices
        price_returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility_1m = np.std(price_returns[-1:]) if len(price_returns) > 0 else 0
        volatility_5m = np.std(price_returns[-5:]) if len(price_returns) >= 5 else volatility_1m
        volatility_15m = np.std(price_returns[-15:]) if len(price_returns) >= 15 else volatility_5m

        # Volume analysis
        recent_volumes = volumes[-5:] if len(volumes) >= 5 else volumes
        avg_volume = np.mean(recent_volumes) if recent_volumes else 1
        current_volume = recent_volumes[-1] if recent_volumes else 1
        volume_ratio_1m = current_volume / avg_volume if avg_volume > 0 else 1
        volume_ratio_5m = np.mean(recent_volumes) / avg_volume if avg_volume > 0 else 1
        volume_spike = volume_ratio_1m > self.thresholds["crisis_volume_spike_threshold"]

        # Technical indicators (simplified calculations)
        rsi_14 = self._calculate_rsi(recent_prices[-14:] if len(recent_prices) >= 14 else recent_prices)
        bb_width = self._calculate_bb_width(recent_prices)
        adx = self._calculate_adx(recent_prices)

        # News impact (placeholder - would come from news feed)
        news_impact = market_data.get('news_impact', 0.0)

        # Market hours (simplified)
        market_hours = self._is_market_hours()

        features = RegimeFeatures(
            symbol=symbol,
            timestamp=datetime.now(),
            price_change_1m=price_change_1m,
            price_change_5m=price_change_5m,
            price_change_15m=price_change_15m,
            volatility_1m=volatility_1m,
            volatility_5m=volatility_5m,
            volatility_15m=volatility_15m,
            volume_ratio_1m=volume_ratio_1m,
            volume_ratio_5m=volume_ratio_5m,
            volume_spike=volume_spike,
            rsi_14=rsi_14,
            bb_width=bb_width,
            adx=adx,
            news_impact=news_impact,
            market_hours=market_hours
        )

        return features

    def _create_empty_features(self, symbol: str) -> RegimeFeatures:
        """Create empty features for when data is insufficient"""
        return RegimeFeatures(
            symbol=symbol,
            timestamp=datetime.now(),
            price_change_1m=0.0,
            price_change_5m=0.0,
            price_change_15m=0.0,
            volatility_1m=0.0,
            volatility_5m=0.0,
            volatility_15m=0.0,
            volume_ratio_1m=1.0,
            volume_ratio_5m=1.0,
            volume_spike=False,
            rsi_14=50.0,
            bb_width=0.0,
            adx=0.0,
            news_impact=0.0,
            market_hours=self._is_market_hours()
        )

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bb_width(self, prices: List[float], period: int = 20) -> float:
        """Calculate Bollinger Bands width"""
        if len(prices) < period:
            return 0.0

        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)

        if sma == 0:
            return 0.0

        upper_band = sma + 2 * std
        lower_band = sma - 2 * std

        return (upper_band - lower_band) / sma

    def _calculate_adx(self, prices: List[float], period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)"""
        if len(prices) < period + 1:
            return 0.0

        # Simplified ADX calculation
        # In practice, you'd calculate true range, directional movement, etc.
        recent_prices = prices[-period:]
        price_range = np.max(recent_prices) - np.min(recent_prices)
        avg_price = np.mean(recent_prices)

        if avg_price == 0:
            return 0.0

        return min(100, (price_range / avg_price) * 50)

    def _is_market_hours(self) -> bool:
        """Check if current time is during main market hours"""
        now = datetime.now()
        # Simplified: consider market hours 8 AM - 5 PM UTC for forex
        return 8 <= now.hour <= 17

    def classify_regime(self, symbol: str, features: RegimeFeatures) -> MarketRegime:
        """
        Classify market regime based on features

        Uses a combination of rule-based and learned thresholds
        """

        # Price action analysis
        price_trend_strength = abs(features.price_change_15m) / max(1, features.volatility_15m * 10000)
        volatility_level = features.volatility_5m

        # Technical indicator analysis
        trend_strength = features.adx / 100.0  # Normalize to 0-1
        range_bound = features.bb_width < self.thresholds["range_max_bb_width"]
        overbought_oversold = features.rsi_14 > 70 or features.rsi_14 < 30

        # Volume and external factors
        volume_anomaly = features.volume_spike or features.volume_ratio_1m > 2.0
        news_pressure = features.news_impact > self.thresholds["news_impact_threshold"]

        # Regime classification logic
        if news_pressure or (volatility_level > self.thresholds["crisis_min_volatility"] and volume_anomaly):
            return MarketRegime.CRISIS

        elif volatility_level > self.thresholds["volatile_min_volatility"] or volume_anomaly:
            return MarketRegime.VOLATILE

        elif trend_strength > 0.6 and price_trend_strength > 2.0:
            return MarketRegime.TREND

        elif range_bound and trend_strength < 0.3:
            return MarketRegime.RANGE

        else:
            return MarketRegime.VOLATILE  # Default to volatile if uncertain

    def predict_regime(self, symbol: str, market_data: Dict) -> RegimePrediction:
        """
        Predict market regime with confidence scores

        Returns:
            RegimePrediction with classification and confidence
        """

        # Extract features
        features = self.extract_regime_features(symbol, market_data)

        # Get base classification
        predicted_regime = self.classify_regime(symbol, features)

        # Calculate confidence scores for all regimes
        regime_scores = self._calculate_regime_scores(features, predicted_regime)

        # Overall confidence is the score of the predicted regime
        confidence = regime_scores[predicted_regime]

        # Generate reasoning
        reasoning = self._generate_regime_reasoning(features, predicted_regime, regime_scores)

        # Create prediction
        prediction = RegimePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            predicted_regime=predicted_regime,
            confidence=confidence,
            regime_scores=regime_scores,
            features=features,
            reasoning=reasoning
        )

        # Update history
        self._update_history(symbol, prediction)

        return prediction

    def _calculate_regime_scores(self, features: RegimeFeatures, primary_regime: MarketRegime) -> Dict[MarketRegime, float]:
        """Calculate confidence scores for all regimes"""

        scores = {}

        # Base scores from different analysis methods
        price_action_score = self._price_action_regime_score(features)
        technical_score = self._technical_regime_score(features)
        volume_score = self._volume_regime_score(features)
        external_score = self._external_regime_score(features)

        # Weight the scores
        weights = self.config["confidence_weights"]

        for regime in MarketRegime:
            if regime == MarketRegime.UNKNOWN:
                scores[regime] = 0.0
                continue

            # Combine scores based on regime characteristics
            if regime == MarketRegime.TREND:
                scores[regime] = (price_action_score * 0.4 +
                                technical_score * 0.4 +
                                volume_score * 0.1 +
                                external_score * 0.1)
            elif regime == MarketRegime.RANGE:
                scores[regime] = (technical_score * 0.5 +
                                price_action_score * 0.3 +
                                external_score * 0.2)
            elif regime == MarketRegime.VOLATILE:
                scores[regime] = (volume_score * 0.4 +
                                price_action_score * 0.3 +
                                technical_score * 0.2 +
                                external_score * 0.1)
            elif regime == MarketRegime.CRISIS:
                scores[regime] = (external_score * 0.4 +
                                volume_score * 0.4 +
                                price_action_score * 0.2)

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for regime in scores:
                scores[regime] /= total

        return scores

    def _price_action_regime_score(self, features: RegimeFeatures) -> float:
        """Calculate regime score based on price action"""
        trend_strength = min(1.0, abs(features.price_change_15m) / 100)  # Normalize
        volatility_ratio = min(1.0, features.volatility_5m / 0.01)

        # Trend regime: strong directional movement with moderate volatility
        trend_score = trend_strength * (1 - volatility_ratio)

        # Range regime: low directional movement, low volatility
        range_score = (1 - trend_strength) * (1 - volatility_ratio * 0.5)

        # Volatile regime: high volatility regardless of direction
        volatile_score = volatility_ratio

        return max(trend_score, range_score, volatile_score)

    def _technical_regime_score(self, features: RegimeFeatures) -> float:
        """Calculate regime score based on technical indicators"""
        adx_score = features.adx / 100  # Normalize ADX
        bb_width_score = min(1.0, features.bb_width / 0.05)
        rsi_extreme = 1.0 if features.rsi_14 > 70 or features.rsi_14 < 30 else 0.0

        # Trend: high ADX, moderate BB width
        trend_score = adx_score * (1 - bb_width_score * 0.5)

        # Range: low ADX, low BB width
        range_score = (1 - adx_score) * (1 - bb_width_score)

        # Volatile: wide BB, extreme RSI
        volatile_score = bb_width_score * (0.5 + rsi_extreme * 0.5)

        return max(trend_score, range_score, volatile_score)

    def _volume_regime_score(self, features: RegimeFeatures) -> float:
        """Calculate regime score based on volume analysis"""
        volume_ratio_score = min(1.0, (features.volume_ratio_1m - 1) / 2)
        volume_spike_score = 1.0 if features.volume_spike else 0.0

        # Crisis/Volatile regime: volume spikes indicate stress
        crisis_score = volume_spike_score * 0.8 + volume_ratio_score * 0.2

        # Normal regimes: moderate volume is good
        normal_score = 1 - volume_ratio_score * 0.5

        return max(crisis_score, normal_score)

    def _external_regime_score(self, features: RegimeFeatures) -> float:
        """Calculate regime score based on external factors"""
        news_score = min(1.0, features.news_impact / self.thresholds["news_impact_threshold"])
        market_hours_score = 1.0 if features.market_hours else 0.5

        return (news_score * 0.7 + market_hours_score * 0.3)

    def _generate_regime_reasoning(self, features: RegimeFeatures,
                                 predicted_regime: MarketRegime,
                                 scores: Dict[MarketRegime, float]) -> str:
        """Generate human-readable reasoning for regime classification"""

        reasoning_parts = []

        if predicted_regime == MarketRegime.TREND:
            reasoning_parts.append(f"Strong directional movement ({features.price_change_15m:.0f} pips in 15m)")
            reasoning_parts.append(f'ADX indicates trend strength ({features.adx:.1f})')
        elif predicted_regime == MarketRegime.RANGE:
            reasoning_parts.append("Price oscillating within Bollinger Bands")
            reasoning_parts.append(f'Low volatility ({features.volatility_5m:.4f})')
        elif predicted_regime == MarketRegime.VOLATILE:
            reasoning_parts.append(f'High volatility detected ({features.volatility_5m:.4f})')
            if features.volume_spike:
                reasoning_parts.append("Volume spike observed")
        elif predicted_regime == MarketRegime.CRISIS:
            reasoning_parts.append(f'Extreme volatility ({features.volatility_5m:.4f})')
            reasoning_parts.append(f'High news impact ({features.news_impact:.2f})')

        # Add confidence information
        top_regime = max(scores, key=scores.get)
        if top_regime != predicted_regime:
            reasoning_parts.append(f'Close to {top_regime.value} regime (score: {scores[top_regime]:.2f})')

        return "; ".join(reasoning_parts)

    def _update_history(self, symbol: str, prediction: RegimePrediction):
        """Update regime and feature history"""

        # Initialize history if needed
        if symbol not in self.regime_history:
            self.regime_history[symbol] = deque(maxlen=self.max_history)
            self.feature_history[symbol] = deque(maxlen=self.max_history)

        # Add to history
        self.regime_history[symbol].append(prediction)
        self.feature_history[symbol].append(prediction.features)

        # Track regime transitions
        if symbol not in self.regime_transitions:
            self.regime_transitions[symbol] = []

        # Check for regime change
        if len(self.regime_history[symbol]) >= 2:
            prev_regime = self.regime_history[symbol][-2].predicted_regime
            if prev_regime != prediction.predicted_regime:
                self.regime_transitions[symbol].append((prediction.predicted_regime, prediction.timestamp))

    def get_current_regime(self, symbol: str) -> Optional[RegimePrediction]:
        """Get the most recent regime prediction for a symbol"""
        if symbol in self.regime_history and self.regime_history[symbol]:
            return self.regime_history[symbol][-1]
        return None

    def get_regime_history(self, symbol: str, minutes: int = 60) -> List[RegimePrediction]:
        """Get regime history for the last N minutes"""
        if symbol not in self.regime_history:
            return []

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [p for p in self.regime_history[symbol] if p.timestamp >= cutoff_time]

    def get_regime_stability(self, symbol: str, minutes: int = 60) -> Dict:
        """Get regime stability metrics"""
        history = self.get_regime_history(symbol, minutes)

        if not history:
            return {"stability": 0.0, "transitions": 0, "dominant_regime": None}

        # Count regime occurrences
        regime_counts = {}
        for prediction in history:
            regime = prediction.predicted_regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Calculate stability (percentage of time in dominant regime)
        total_predictions = len(history)
        dominant_regime = max(regime_counts, key=regime_counts.get)
        dominant_count = regime_counts[dominant_regime]
        stability = dominant_count / total_predictions

        # Count transitions
        transitions = 0
        for i in range(1, len(history)):
            if history[i].predicted_regime != history[i-1].predicted_regime:
                transitions += 1

        return {
            "stability": stability,
            "transitions": transitions,
            "dominant_regime": dominant_regime.value,
            "regime_distribution": {regime.value: count/total_predictions
                                  for regime, count in regime_counts.items()}
        }

    def should_enable_strategy(self, symbol: str, strategy_type: str) -> Tuple[bool, str]:
        """
        Determine if a strategy should be enabled based on current regime

        Args:
            symbol: Trading symbol
            strategy_type: "scalping", "grid", "trend_following", "mean_reversion"

        Returns:
            (enabled: bool, reason: str)
        """

        current_regime = self.get_current_regime(symbol)

        if not current_regime:
            return False, "No regime data available"

        regime = current_regime.predicted_regime

        strategy_rules = {
            "scalping": {
                MarketRegime.RANGE: (True, "Range-bound market suitable for scalping"),
                MarketRegime.TREND: (False, "Trending market not suitable for scalping"),
                MarketRegime.VOLATILE: (False, "High volatility unsuitable for scalping"),
                MarketRegime.CRISIS: (False, "Crisis conditions disable scalping")
            },
            "grid": {
                MarketRegime.RANGE: (True, "Range-bound market suitable for grid trading"),
                MarketRegime.TREND: (False, "Trending market unsuitable for grid"),
                MarketRegime.VOLATILE: (False, "High volatility unsuitable for grid"),
                MarketRegime.CRISIS: (False, "Crisis conditions disable grid")
            },
            "trend_following": {
                MarketRegime.TREND: (True, "Trending market suitable for trend following"),
                MarketRegime.RANGE: (False, "Range-bound market unsuitable for trend following"),
                MarketRegime.VOLATILE: (True, "Can follow trends in volatile markets"),
                MarketRegime.CRISIS: (False, "Crisis conditions disable trend following")
            },
            "mean_reversion": {
                MarketRegime.RANGE: (True, "Range-bound market suitable for mean reversion"),
                MarketRegime.TREND: (False, "Trending market unsuitable for mean reversion"),
                MarketRegime.VOLATILE: (False, "High volatility unsuitable for mean reversion"),
                MarketRegime.CRISIS: (False, "Crisis conditions disable mean reversion")
            }
        }

        if strategy_type not in strategy_rules:
            return False, f"Unknown strategy type: {strategy_type}"

        regime_rules = strategy_rules[strategy_type]
        enabled, reason = regime_rules.get(regime, (False, f"Regime {regime.value} not suitable for {strategy_type}"))

        # Add confidence factor
        if enabled and current_regime.confidence < 0.6:
            return False, f'{reason} (low confidence: {current_regime.confidence:.2f})'

        return enabled, reason

if __name__ == "__main__":
    # Demo usage
    classifier = RegimeClassifier()

    # Simulate market data for EURUSD
    market_data = {
        'prices': [1.1234, 1.1235, 1.1233, 1.1236, 1.1234, 1.1237, 1.1235, 1.1238,
                  1.1236, 1.1239, 1.1237, 1.1240, 1.1238, 1.1241, 1.1239],
        'volumes': [100, 105, 98, 110, 102, 115, 108, 120, 105, 125, 110, 130, 115, 135, 120],
        'news_impact': 0.1
    }

    # Get regime prediction
    prediction = classifier.predict_regime("EURUSD", market_data)

    print(f"Symbol: {prediction.symbol}")
    print(f"Regime: {prediction.predicted_regime.value}")
    print(f'Confidence: {prediction.confidence:.2f}')
    print(f"Reasoning: {prediction.reasoning}")
    print(f"All regime scores: { {k.value: f'{v:.2f}' for k, v in prediction.regime_scores.items()} }")

    # Check strategy enablement
    strategies = ["scalping", "grid", "trend_following", "mean_reversion"]
    for strategy in strategies:
        enabled, reason = classifier.should_enable_strategy("EURUSD", strategy)
        print(f"{strategy}: {enabled} - {reason}")

    # Get regime stability
    stability = classifier.get_regime_stability("EURUSD")
    print(f'\\nRegime stability: {stability["stability"]:.2f}')
    print(f"Dominant regime: {stability['dominant_regime']}")
    print(f"Transitions: {stability['transitions']}")
