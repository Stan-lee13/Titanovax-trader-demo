#!/usr/bin/env python3
"""
Micro-Slippage Model for TitanovaX
Maintains per-symbol slippage models learned online and requires minimum edge > slippage
"""

import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

@dataclass
class SlippageObservation:
    """Single slippage observation"""
    symbol: str
    timestamp: datetime
    expected_price: float
    executed_price: float
    order_type: str  # "market", "limit", "twap", etc.
    volume: float
    market_condition: str
    spread_at_time: float
    volatility_at_time: float
    actual_slippage: float
    expected_slippage: float

@dataclass
class SlippageModel:
    """Per-symbol slippage model"""
    symbol: str
    last_updated: datetime

    # Rolling statistics
    slippage_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    spread_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    volatility_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Model parameters
    base_slippage: float = 0.5  # Base slippage in pips
    spread_coefficient: float = 0.3
    volatility_coefficient: float = 0.2
    volume_coefficient: float = 0.1

    # Condition-specific adjustments
    condition_multipliers: Dict[str, float] = field(default_factory=dict)

    # Performance tracking
    model_accuracy: float = 0.5
    prediction_confidence: float = 0.5

class MicroSlippageModel:
    """Online slippage model that learns from execution results"""

    def __init__(self, config_path: str = 'config/slippage_config.json'):
        self.config_path = Path(config_path)
        self.models: Dict[str, SlippageModel] = {}
        self.observation_history: List[SlippageObservation] = []
        self.max_observations = 10000

        self.load_config()
        self.setup_logging()

        # Initialize condition multipliers
        self.default_condition_multipliers = {
            "normal": 1.0,
            "wide_spread": 1.5,
            "high_volatility": 2.0,
            "news_impact": 3.0,
            "low_liquidity": 2.5
        }

    def load_config(self):
        """Load configuration"""
        default_config = {
            "update_frequency_minutes": 5,
            "min_observations_for_model": 50,
            "learning_rate": 0.1,
            "max_slippage_tolerance_pips": 5.0,
            "model_persistence_enabled": True,
            "condition_multipliers": {
                "normal": 1.0,
                "wide_spread": 1.5,
                "high_volatility": 2.0,
                "news_impact": 3.0,
                "low_liquidity": 2.5
            }
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load slippage config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def record_slippage(self, symbol: str, expected_price: float, executed_price: float,
                       order_type: str, volume: float, market_condition: str,
                       spread_at_time: float, volatility_at_time: float) -> float:
        """
        Record a slippage observation and update model

        Returns:
            Current estimated slippage for this symbol
        """

        actual_slippage = abs(executed_price - expected_price) * 10000  # Convert to pips

        observation = SlippageObservation(
            symbol=symbol,
            timestamp=datetime.now(),
            expected_price=expected_price,
            executed_price=executed_price,
            order_type=order_type,
            volume=volume,
            market_condition=market_condition,
            spread_at_time=spread_at_time,
            volatility_at_time=volatility_at_time,
            actual_slippage=actual_slippage,
            expected_slippage=0.0  # Will be calculated after model update
        )

        # Add to history
        self.observation_history.append(observation)
        if len(self.observation_history) > self.max_observations:
            self.observation_history = self.observation_history[-self.max_observations:]

        # Update model
        self._update_slippage_model(symbol, observation)

        return self.estimate_slippage(symbol, order_type, volume, market_condition)

    def _update_slippage_model(self, symbol: str, observation: SlippageObservation):
        """Update slippage model with new observation"""

        if symbol not in self.models:
            self.models[symbol] = SlippageModel(
                symbol=symbol,
                last_updated=datetime.now(),
                condition_multipliers=self.config["condition_multipliers"].copy()
            )

        model = self.models[symbol]

        # Update histories
        model.slippage_history.append(observation.actual_slippage)
        model.spread_history.append(observation.spread_at_time)
        model.volatility_history.append(observation.volatility_at_time)

        # Online learning update (simple gradient-like update)
        if len(model.slippage_history) >= self.config["min_observations_for_model"]:
            self._recalculate_model_parameters(model, observation)

        model.last_updated = datetime.now()

    def _recalculate_model_parameters(self, model: SlippageModel, new_observation: SlippageObservation):
        """Recalculate model parameters based on recent history"""

        # Convert deques to numpy arrays
        slippages = np.array(model.slippage_history)
        spreads = np.array(model.spread_history)
        volatilities = np.array(model.volatility_history)

        # Update base slippage (exponential moving average)
        alpha = self.config["learning_rate"]
        model.base_slippage = (1 - alpha) * model.base_slippage + alpha * np.mean(slippages)

        # Update coefficients using linear regression-like approach
        if len(slippages) > 50:  # Need sufficient data
            # Simple linear model: slippage ~ spread + volatility + volume
            X = np.column_stack([spreads, volatilities, np.ones_like(spreads)])
            y = slippages

            try:
                # Use numpy least squares
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

                if len(coeffs) >= 3:
                    model.spread_coefficient = max(0, coeffs[0])
                    model.volatility_coefficient = max(0, coeffs[1])
                    model.base_slippage = max(0, coeffs[2])
            except:
                # Fall back to simple averages if regression fails
                model.spread_coefficient = 0.3
                model.volatility_coefficient = 0.2

        # Update condition multipliers based on recent performance
        self._update_condition_multipliers(model)

        # Calculate model accuracy
        if len(model.slippage_history) > 10:
            predicted = [self._predict_slippage_from_model(model, obs.market_condition, obs.spread_at_time, obs.volatility_at_time)
                        for obs in list(model.slippage_history)[-100:]]
            actual = list(model.slippage_history)[-100:]

            if len(predicted) == len(actual):
                mse = np.mean((np.array(predicted) - np.array(actual))**2)
                model.model_accuracy = max(0.1, 1.0 / (1.0 + mse))  # Convert MSE to accuracy-like score
                model.prediction_confidence = min(1.0, len(model.slippage_history) / 500.0)

    def _update_condition_multipliers(self, model: SlippageModel):
        """Update condition multipliers based on recent observations"""

        recent_observations = list(model.slippage_history)[-100:]  # Last 100 observations

        if len(recent_observations) < 10:
            return

        # Group by condition and calculate average slippage
        condition_slippages = defaultdict(list)

        # Get recent observations with their conditions (this is simplified)
        # In a real implementation, you'd need to track conditions with observations
        for i, obs in enumerate(self.observation_history[-100:]):
            if i < len(recent_observations):
                condition_slippages[obs.market_condition].append(recent_observations[i])

        # Update multipliers based on relative slippage
        overall_avg = np.mean(recent_observations)

        for condition, slippages in condition_slippages.items():
            if slippages:
                condition_avg = np.mean(slippages)
                multiplier = condition_avg / overall_avg if overall_avg > 0 else 1.0
                model.condition_multipliers[condition] = max(0.5, min(3.0, multiplier))

    def _predict_slippage_from_model(self, model: SlippageModel, condition: str,
                                    spread: float, volatility: float) -> float:
        """Predict slippage using current model parameters"""

        predicted = (model.base_slippage +
                    model.spread_coefficient * spread +
                    model.volatility_coefficient * volatility)

        # Apply condition multiplier
        multiplier = model.condition_multipliers.get(condition, 1.0)
        predicted *= multiplier

        return max(0, predicted)

    def estimate_slippage(self, symbol: str, order_type: str, volume: float,
                         market_condition: str, spread_pips: float = None,
                         volatility: float = None) -> float:
        """
        Estimate expected slippage for a trade

        Returns:
            Estimated slippage in pips
        """

        if symbol not in self.models:
            # Return conservative estimate for unknown symbols
            return self.config["max_slippage_tolerance_pips"] * 0.5

        model = self.models[symbol]

        # Use provided values or recent averages
        if spread_pips is None:
            spread_pips = np.mean(list(model.spread_history)[-10:]) if model.spread_history else 1.0

        if volatility is None:
            volatility = np.mean(list(model.volatility_history)[-10:]) if model.volatility_history else 0.001

        # Base prediction
        predicted = self._predict_slippage_from_model(model, market_condition, spread_pips, volatility)

        # Adjust for order type
        order_type_multipliers = {
            "market": 1.0,
            "limit": 0.3,
            "twap": 0.6,
            "vwap": 0.5,
            "iceberg": 0.4,
            "segmented": 0.7
        }

        multiplier = order_type_multipliers.get(order_type, 1.0)
        predicted *= multiplier

        # Adjust for volume (larger orders typically have more slippage)
        volume_multiplier = 1.0 + (volume - 0.01) * model.volume_coefficient
        predicted *= volume_multiplier

        return max(0, predicted)

    def should_execute_trade(self, symbol: str, expected_edge_pips: float,
                           order_type: str, volume: float, market_condition: str) -> Tuple[bool, str]:
        """
        Determine if a trade should execute based on slippage model

        Returns:
            (should_execute: bool, reason: str)
        """

        estimated_slippage = self.estimate_slippage(symbol, order_type, volume, market_condition)

        # Check if expected edge is sufficient
        min_required_edge = estimated_slippage + 0.5  # Add safety buffer

        if expected_edge_pips < min_required_edge:
            return False, f'Insufficient edge: {expected_edge_pips:.2f} pips < required {min_required_edge:.2f} pips (estimated slippage: {estimated_slippage:.2f})'

        # Check maximum tolerance
        if estimated_slippage > self.config["max_slippage_tolerance_pips"]:
            return False, f'Slippage too high: {estimated_slippage:.2f} > max {self.config["max_slippage_tolerance_pips"]} pips'

        return True, f'Edge sufficient: {expected_edge_pips:.2f} > required {min_required_edge:.2f}'

    def get_model_health(self, symbol: str) -> Dict:
        """Get model health metrics for a symbol"""

        if symbol not in self.models:
            return {"status": "no_data", "observations": 0, "accuracy": 0.0}

        model = self.models[symbol]

        return {
            "status": "active" if len(model.slippage_history) >= self.config["min_observations_for_model"] else "training",
            "observations": len(model.slippage_history),
            "accuracy": model.model_accuracy,
            "confidence": model.prediction_confidence,
            "last_updated": model.last_updated.isoformat(),
            "base_slippage": model.base_slippage,
            "spread_coefficient": model.spread_coefficient,
            "volatility_coefficient": model.volatility_coefficient,
            "condition_multipliers": model.condition_multipliers
        }

    def get_all_models_health(self) -> Dict:
        """Get health metrics for all models"""

        return {symbol: self.get_model_health(symbol) for symbol in self.models.keys()}

    def save_models(self, path: str = 'models/slippage_models.json'):
        """Save all models to disk"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        models_data = {}
        for symbol, model in self.models.items():
            models_data[symbol] = {
                "symbol": model.symbol,
                "last_updated": model.last_updated.isoformat(),
                "base_slippage": model.base_slippage,
                "spread_coefficient": model.spread_coefficient,
                "volatility_coefficient": model.volatility_coefficient,
                "condition_multipliers": model.condition_multipliers,
                "model_accuracy": model.model_accuracy,
                "prediction_confidence": model.prediction_confidence
            }

        with open(path, 'w') as f:
            json.dump(models_data, f, indent=2)

        self.logger.info(f"Saved {len(models_data)} slippage models to {path}")

    def load_models(self, path: str = 'models/slippage_models.json'):
        """Load models from disk"""

        path = Path(path)
        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                models_data = json.load(f)

            for symbol, data in models_data.items():
                model = SlippageModel(
                    symbol=data["symbol"],
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                    base_slippage=data["base_slippage"],
                    spread_coefficient=data["spread_coefficient"],
                    volatility_coefficient=data["volatility_coefficient"],
                    condition_multipliers=data["condition_multipliers"],
                    model_accuracy=data["model_accuracy"],
                    prediction_confidence=data["prediction_confidence"]
                )

                self.models[symbol] = model

            self.logger.info(f"Loaded {len(models_data)} slippage models from {path}")

        except Exception as e:
            self.logger.error(f"Error loading slippage models: {e}")

if __name__ == "__main__":
    # Production usage example - in real trading, this would receive actual trade execution data
    slippage_model = MicroSlippageModel()

    print("Micro-Slippage Model Ready")
    print("In production, this model would:")
    print("1. Receive real trade execution data from MT5 EA")
    print("2. Learn from actual slippage observations")
    print("3. Update models online with real market data")
    print("4. Provide real-time slippage estimates for trade decisions")
    print("")
    print("Example API usage:")

    # Show example of how the model would be used in production
    # In real trading, actual_price and expected_price would come from real broker data
    try:
        # This is a hypothetical example - in production, real trade data would be passed
        # slippage_model.record_slippage(
        #     symbol="EURUSD",
        #     expected_price=1.1234,
        #     executed_price=1.1235,  # Real execution price from broker
        #     order_type="market",
        #     volume=0.1,
        #     market_condition="normal",
        #     spread_at_time=1.2,
        #     volatility_at_time=0.0008
        # )
        print("  - record_slippage() would receive real broker execution data")
        print("  - estimate_slippage() would use learned model parameters")
        print("  - should_execute_trade() would make real trading decisions")

    except Exception as e:
        print(f"Example error (expected in demo mode): {e}")

    # Show current model status (would be empty without real data)
    print("\\nCurrent Model Status:")
    health = slippage_model.get_model_health("EURUSD")
    print(f"  EURUSD Model: {health['status']} ({health['observations']} observations)")
