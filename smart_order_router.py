#!/usr/bin/env python3
"""
Smart Order Router (SOR) for TitanovaX
Dynamically selects optimal execution methods based on market conditions
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Literal
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

class ExecutionMethod(Enum):
    """Available execution methods"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    ICEBERG = "iceberg"  # Hidden large orders
    SEGMENTED = "segmented"  # Split into smaller orders

class MarketCondition(Enum):
    """Market condition classifications"""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    WIDE_SPREAD = "wide_spread"
    LOW_LIQUIDITY = "low_liquidity"
    NEWS_IMPACT = "news_impact"

@dataclass
class ExecutionParams:
    """Parameters for order execution"""
    method: ExecutionMethod
    price: Optional[float] = None
    segments: int = 1
    segment_delay_seconds: int = 0
    hidden_size: Optional[float] = None
    max_slippage_pips: float = 2.0
    urgency: str = "normal"  # "low", "normal", "high"

@dataclass
class MarketState:
    """Current market state for a symbol"""
    symbol: str
    spread_pips: float
    volatility_1m: float
    volume_1m: int
    news_impact: float  # 0.0 to 1.0
    condition: MarketCondition
    timestamp: datetime

class SmartOrderRouter:
    """Smart Order Router that selects optimal execution strategy"""

    def __init__(self, config_path: str = 'config/sor_config.json'):
        self.config_path = Path(config_path)
        self.market_state: Dict[str, MarketState] = {}
        self.execution_history: List[Dict] = []
        self.max_history = 1000

        self.load_config()
        self.setup_logging()

    def load_config(self):
        """Load SOR configuration"""
        default_config = {
            "spread_thresholds": {
                "normal": 2.0,
                "wide": 5.0,
                "very_wide": 10.0
            },
            "volatility_thresholds": {
                "normal": 0.001,
                "high": 0.005,
                "extreme": 0.01
            },
            "twap_segments": {
                "normal": 5,
                "wide_spread": 10,
                "high_volatility": 15
            },
            "iceberg_threshold_lots": 1.0,
            "news_impact_threshold": 0.3,
            "fallback_timeout_seconds": 30,
            "segment_delay_range": [5, 30]  # Random delay between segments
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load SOR config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def assess_market_condition(self, symbol: str, spread_pips: float,
                              volatility: float, volume: int, news_impact: float = 0.0) -> MarketCondition:
        """Assess current market condition"""

        # Update market state
        state = MarketState(
            symbol=symbol,
            spread_pips=spread_pips,
            volatility_1m=volatility,
            volume_1m=volume,
            news_impact=news_impact,
            timestamp=datetime.now(),
            condition=MarketCondition.NORMAL
        )
        self.market_state[symbol] = state

        # Determine condition
        if news_impact >= self.config["news_impact_threshold"]:
            state.condition = MarketCondition.NEWS_IMPACT
        elif spread_pips >= self.config["spread_thresholds"]["very_wide"]:
            state.condition = MarketCondition.LOW_LIQUIDITY
        elif spread_pips >= self.config["spread_thresholds"]["wide"]:
            state.condition = MarketCondition.WIDE_SPREAD
        elif volatility >= self.config["volatility_thresholds"]["extreme"]:
            state.condition = MarketCondition.HIGH_VOLATILITY
        else:
            state.condition = MarketCondition.NORMAL

        return state.condition

    def select_execution_method(self, symbol: str, side: str, volume: float,
                              current_price: float, spread_pips: float,
                              volatility: float, volume_1m: int,
                              news_impact: float = 0.0) -> ExecutionParams:
        """
        Select optimal execution method based on market conditions

        Returns:
            ExecutionParams with recommended execution method and parameters
        """

        # Assess market condition
        condition = self.assess_market_condition(symbol, spread_pips, volatility, volume_1m, news_impact)

        # Base parameters
        params = ExecutionParams(
            method=ExecutionMethod.MARKET,
            price=current_price,
            max_slippage_pips=2.0
        )

        # Select method based on condition
        if condition == MarketCondition.NEWS_IMPACT:
            # Use TWAP during news to avoid slippage
            params.method = ExecutionMethod.TWAP
            params.segments = self.config["twap_segments"]["high_volatility"]
            params.segment_delay_seconds = np.random.randint(*self.config["segment_delay_range"])

        elif condition == MarketCondition.WIDE_SPREAD:
            # Use limit orders when spreads are wide
            params.method = ExecutionMethod.LIMIT
            # Set limit price slightly better than market
            price_adjustment = spread_pips * 0.3  # 30% of spread
            if side == "BUY":
                params.price = current_price - price_adjustment / 10000  # Convert pips to price
            else:
                params.price = current_price + price_adjustment / 10000

        elif condition == MarketCondition.HIGH_VOLATILITY:
            # Use segmented execution for large orders
            if volume > 0.5:  # Large order threshold
                params.method = ExecutionMethod.SEGMENTED
                params.segments = self.config["twap_segments"]["high_volatility"]
                params.segment_delay_seconds = np.random.randint(*self.config["segment_delay_range"])
            else:
                params.method = ExecutionMethod.TWAP
                params.segments = self.config["twap_segments"]["normal"]

        elif condition == MarketCondition.LOW_LIQUIDITY:
            # Use iceberg orders for large volumes
            if volume > self.config["iceberg_threshold_lots"]:
                params.method = ExecutionMethod.ICEBERG
                params.hidden_size = volume * 0.1  # Show only 10% at a time
                params.segments = 10
                params.segment_delay_seconds = np.random.randint(10, 60)
            else:
                params.method = ExecutionMethod.TWAP
                params.segments = self.config["twap_segments"]["wide_spread"]

        else:  # NORMAL condition
            # Use market for small orders, TWAP for larger ones
            if volume > 0.3:  # Medium-large order
                params.method = ExecutionMethod.TWAP
                params.segments = self.config["twap_segments"]["normal"]
            else:
                params.method = ExecutionMethod.MARKET

        # Adjust for urgency (this could come from signal confidence)
        if hasattr(self, '_last_signal_confidence'):
            if self._last_signal_confidence > 0.8:
                params.urgency = "high"
                params.max_slippage_pips = 1.0  # Allow less slippage for high confidence
            elif self._last_signal_confidence < 0.5:
                params.urgency = "low"
                params.max_slippage_pips = 3.0  # Allow more slippage for low confidence

        return params

    def set_signal_confidence(self, confidence: float):
        """Set signal confidence for execution optimization"""
        self._last_signal_confidence = confidence

    def get_execution_recommendation(self, symbol: str, side: str, volume: float,
                                   current_price: float, spread_pips: float,
                                   volatility: float, volume_1m: int,
                                   news_impact: float = 0.0,
                                   signal_confidence: float = 0.5) -> Dict:
        """
        Get comprehensive execution recommendation

        Returns:
            {
                "method": ExecutionMethod,
                "params": ExecutionParams,
                "reasoning": str,
                "expected_slippage": float,
                "estimated_duration": int,
                "fallback_methods": [ExecutionMethod]
            }
        """

        self.set_signal_confidence(signal_confidence)

        params = self.select_execution_method(
            symbol, side, volume, current_price, spread_pips,
            volatility, volume_1m, news_impact
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(params, self.market_state.get(symbol))

        # Estimate slippage and duration
        expected_slippage = self._estimate_slippage(params, spread_pips, volatility)
        estimated_duration = self._estimate_duration(params, volume)

        # Prepare fallback methods
        fallback_methods = self._get_fallback_methods(params.method)

        recommendation = {
            "method": params.method.value,
            "params": {
                "price": params.price,
                "segments": params.segments,
                "segment_delay_seconds": params.segment_delay_seconds,
                "hidden_size": params.hidden_size,
                "max_slippage_pips": params.max_slippage_pips,
                "urgency": params.urgency
            },
            "reasoning": reasoning,
            "expected_slippage": expected_slippage,
            "estimated_duration_seconds": estimated_duration,
            "fallback_methods": [method.value for method in fallback_methods],
            "market_condition": self.market_state.get(symbol, {}).condition.value if symbol in self.market_state else "unknown"
        }

        # Log execution decision
        self._log_execution_decision(recommendation)

        return recommendation

    def _generate_reasoning(self, params: ExecutionParams, market_state: Optional[MarketState]) -> str:
        """Generate human-readable reasoning for execution choice"""

        if market_state:
            condition = market_state.condition.value.replace('_', ' ').title()
        else:
            condition = "Unknown"

        method_explanations = {
            ExecutionMethod.MARKET: "Immediate execution for small orders in normal conditions",
            ExecutionMethod.LIMIT: "Price improvement when spreads are wide",
            ExecutionMethod.TWAP: "Gradual execution to minimize market impact",
            ExecutionMethod.VWAP: "Volume-weighted execution for large orders",
            ExecutionMethod.ICEBERG: "Hidden execution for very large orders",
            ExecutionMethod.SEGMENTED: "Split execution to reduce slippage"
        }

        reasoning = f"Selected {params.method.value.upper()} execution due to {condition} conditions. "
        reasoning += method_explanations[params.method]

        if params.segments > 1:
            reasoning += f" Order will be split into {params.segments} segments"

        if params.method == ExecutionMethod.LIMIT and params.price:
            reasoning += f" with limit price {params.price}"

        return reasoning

    def _estimate_slippage(self, params: ExecutionParams, spread_pips: float, volatility: float) -> float:
        """Estimate expected slippage for the chosen method"""

        base_slippage = {
            ExecutionMethod.MARKET: spread_pips * 0.5,
            ExecutionMethod.LIMIT: spread_pips * 0.1,
            ExecutionMethod.TWAP: spread_pips * 0.2,
            ExecutionMethod.VWAP: spread_pips * 0.15,
            ExecutionMethod.ICEBERG: spread_pips * 0.1,
            ExecutionMethod.SEGMENTED: spread_pips * 0.25
        }

        estimated = base_slippage.get(params.method, spread_pips * 0.3)

        # Adjust for volatility
        if volatility > self.config["volatility_thresholds"]["high"]:
            estimated *= 1.5

        return estimated

    def _estimate_duration(self, params: ExecutionParams, volume: float) -> int:
        """Estimate execution duration in seconds"""

        base_duration = {
            ExecutionMethod.MARKET: 5,
            ExecutionMethod.LIMIT: 60,  # Up to 1 minute for limit orders
            ExecutionMethod.TWAP: params.segments * params.segment_delay_seconds,
            ExecutionMethod.VWAP: 300,  # 5 minutes
            ExecutionMethod.ICEBERG: params.segments * 30,
            ExecutionMethod.SEGMENTED: params.segments * params.segment_delay_seconds
        }

        duration = base_duration.get(params.method, 30)

        # Adjust for volume (larger orders take longer)
        if volume > 1.0:
            duration *= 1.5
        elif volume < 0.1:
            duration *= 0.7

        return int(duration)

    def _get_fallback_methods(self, primary_method: ExecutionMethod) -> List[ExecutionMethod]:
        """Get fallback execution methods if primary fails"""

        fallback_chains = {
            ExecutionMethod.MARKET: [ExecutionMethod.LIMIT, ExecutionMethod.TWAP],
            ExecutionMethod.LIMIT: [ExecutionMethod.TWAP, ExecutionMethod.MARKET],
            ExecutionMethod.TWAP: [ExecutionMethod.VWAP, ExecutionMethod.SEGMENTED],
            ExecutionMethod.VWAP: [ExecutionMethod.TWAP, ExecutionMethod.MARKET],
            ExecutionMethod.ICEBERG: [ExecutionMethod.TWAP, ExecutionMethod.MARKET],
            ExecutionMethod.SEGMENTED: [ExecutionMethod.TWAP, ExecutionMethod.MARKET]
        }

        return fallback_chains.get(primary_method, [ExecutionMethod.MARKET])

    def _log_execution_decision(self, recommendation: Dict):
        """Log execution decision for analysis"""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": recommendation.get("symbol", "unknown"),
            "method": recommendation["method"],
            "expected_slippage": recommendation["expected_slippage"],
            "estimated_duration": recommendation["estimated_duration_seconds"],
            "market_condition": recommendation["market_condition"],
            "reasoning": recommendation["reasoning"]
        }

        self.execution_history.append(log_entry)

        # Keep only recent history
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""

        if not self.execution_history:
            return {"total_decisions": 0}

        methods = [entry["method"] for entry in self.execution_history]
        conditions = [entry["market_condition"] for entry in self.execution_history]

        stats = {
            "total_decisions": len(self.execution_history),
            "method_distribution": {},
            "condition_distribution": {},
            "avg_expected_slippage": sum(entry["expected_slippage"] for entry in self.execution_history) / len(self.execution_history),
            "recent_decisions": self.execution_history[-10:]  # Last 10 decisions
        }

        # Count methods
        for method in methods:
            stats["method_distribution"][method] = stats["method_distribution"].get(method, 0) + 1

        # Count conditions
        for condition in conditions:
            stats["condition_distribution"][condition] = stats["condition_distribution"].get(condition, 0) + 1

        return stats

if __name__ == "__main__":
    # Demo usage
    sor = SmartOrderRouter()

    # Test normal conditions
    recommendation = sor.get_execution_recommendation(
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        current_price=1.1234,
        spread_pips=1.5,
        volatility=0.0008,
        volume_1m=1000,
        signal_confidence=0.75
    )

    print("EURUSD Normal Conditions:")
    for key, value in recommendation.items():
        print(f"  {key}: {value}")

    # Test high volatility
    recommendation = sor.get_execution_recommendation(
        symbol="GBPUSD",
        side="SELL",
        volume=0.5,
        current_price=1.2678,
        spread_pips=3.0,
        volatility=0.008,
        volume_1m=500,
        signal_confidence=0.6
    )

    print("\\nGBPUSD High Volatility:")
    for key, value in recommendation.items():
        print(f"  {key}: {value}")

    # Show execution stats
    stats = sor.get_execution_stats()
    print("\\nExecution Stats:")
    print(f"Total decisions: {stats['total_decisions']}")
    print(f'Average expected slippage: {stats["avg_expected_slippage"]:.3f} pips')
    print(f"Method distribution: {stats['method_distribution']}")
