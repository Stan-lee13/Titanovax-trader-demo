#!/usr/bin/env python3
"""
Ensemble Decision Engine for TitanovaX
Combines XGBoost + Transformer + TA signals with Bayesian meta-model
Outputs calibrated probabilities and recommended actions
"""

import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import pandas as pd
from scipy import stats

@dataclass
class ModelPrediction:
    """Prediction from a single model"""
    model_id: str
    model_type: str  # "xgboost", "transformer", "technical", "rl"
    symbol: str
    timestamp: datetime
    prob_up: float
    prob_down: float
    prob_sideways: float
    confidence: float
    prediction_horizon_minutes: int
    features_hash: str
    raw_output: Optional[Dict] = None

@dataclass
class EnsemblePrediction:
    """Final ensemble prediction"""
    symbol: str
    timestamp: datetime
    ensemble_prob_up: float
    ensemble_prob_down: float
    ensemble_prob_sideways: float
    ensemble_confidence: float
    model_weights: Dict[str, float]
    individual_predictions: List[ModelPrediction]
    meta_features: Dict[str, Any]
    recommended_action: str
    recommended_size: float
    reasoning: str
    risk_score: float

class BayesianMetaModel:
    """Bayesian meta-model for combining predictions"""

    def __init__(self):
        self.model_reliability_scores: Dict[str, float] = {}
        self.regime_adjustments: Dict[str, Dict[str, float]] = {}
        self.confidence_calibration: Dict[str, Tuple[float, float]] = {}  # mean, std

    def update_model_reliability(self, model_id: str, predictions: List[ModelPrediction],
                               actual_outcomes: List[int]):
        """Update model reliability based on prediction accuracy"""

        if len(predictions) != len(actual_outcomes):
            return

        # Calculate accuracy metrics
        probs_up = np.array([p.prob_up for p in predictions])
        actual_binary = np.array([1 if outcome > 0 else 0 for outcome in actual_outcomes])

        # Brier score (lower is better)
        brier_score = np.mean((probs_up - actual_binary) ** 2)

        # Reliability score (inverse of Brier score, normalized)
        reliability = max(0.1, 1.0 / (1.0 + brier_score))

        # Update rolling average
        alpha = 0.1
        current_reliability = self.model_reliability_scores.get(model_id, 0.5)
        self.model_reliability_scores[model_id] = (
            (1 - alpha) * current_reliability + alpha * reliability
        )

    def get_model_weight(self, model_id: str, regime: str) -> float:
        """Get model weight adjusted for current regime"""

        base_weight = self.model_reliability_scores.get(model_id, 0.5)

        if regime in self.regime_adjustments and model_id in self.regime_adjustments[regime]:
            regime_multiplier = self.regime_adjustments[regime][model_id]
            return base_weight * regime_multiplier

        return base_weight

class EnsembleDecisionEngine:
    """Main ensemble decision engine"""

    def __init__(self, config_path: str = 'config/ensemble_config.json'):
        self.config_path = Path(config_path)
        self.prediction_history: Dict[str, deque] = {}
        self.meta_model = BayesianMetaModel()
        self.decision_history: List[Dict] = []
        self.max_history = 1000

        self.load_config()
        self.setup_logging()

    def load_config(self):
        """Load ensemble configuration"""
        default_config = {
            "model_weights": {
                "xgboost": 0.4,
                "transformer": 0.3,
                "technical": 0.2,
                "rl": 0.1
            },
            "regime_adjustments": {
                "trend": {
                    "xgboost": 1.0,
                    "transformer": 1.2,
                    "technical": 0.8,
                    "rl": 1.0
                },
                "range": {
                    "xgboost": 1.0,
                    "transformer": 0.8,
                    "technical": 1.2,
                    "rl": 1.0
                },
                "volatile": {
                    "xgboost": 1.0,
                    "transformer": 1.0,
                    "technical": 0.6,
                    "rl": 1.4
                },
                "crisis": {
                    "xgboost": 0.8,
                    "transformer": 0.6,
                    "technical": 0.4,
                    "rl": 1.6
                }
            },
            "calibration_thresholds": {
                "min_confidence": 0.6,
                "high_confidence": 0.8,
                "action_threshold": 0.65
            },
            "risk_multipliers": {
                "normal": 1.0,
                "elevated": 0.7,
                "high": 0.4,
                "crisis": 0.1
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
            logging.warning(f"Could not load ensemble config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def combine_predictions(self, predictions: List[ModelPrediction],
                          regime: str, symbol: str) -> EnsemblePrediction:
        """
        Combine individual model predictions into ensemble prediction

        Args:
            predictions: List of individual model predictions
            regime: Current market regime
            symbol: Trading symbol

        Returns:
            EnsemblePrediction with combined probabilities and recommendations
        """

        if not predictions:
            return self._create_empty_ensemble(symbol)

        # Filter out low-confidence predictions
        valid_predictions = [p for p in predictions if p.confidence >= self.config["calibration_thresholds"]["min_confidence"]]

        if not valid_predictions:
            return self._create_empty_ensemble(symbol, reason="All predictions below confidence threshold")

        # Get model weights adjusted for regime
        weights = {}
        total_weight = 0

        for pred in valid_predictions:
            weight = self.meta_model.get_model_weight(pred.model_id, regime)
            weights[pred.model_id] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Combine probabilities using weighted average
        ensemble_prob_up = sum(p.prob_up * weights[p.model_id] for p in valid_predictions)
        ensemble_prob_down = sum(p.prob_down * weights[p.model_id] for p in valid_predictions)
        ensemble_prob_sideways = sum(p.prob_sideways * weights[p.model_id] for p in valid_predictions)

        # Normalize probabilities
        total_prob = ensemble_prob_up + ensemble_prob_down + ensemble_prob_sideways
        if total_prob > 0:
            ensemble_prob_up /= total_prob
            ensemble_prob_down /= total_prob
            ensemble_prob_sideways /= total_prob

        # Calculate ensemble confidence
        individual_confidences = [p.confidence for p in valid_predictions]
        ensemble_confidence = np.mean(individual_confidences) * min(1.0, len(valid_predictions) / 3.0)

        # Apply temperature scaling for calibration
        ensemble_prob_up, ensemble_prob_down, ensemble_prob_sideways = self._calibrate_probabilities(
            ensemble_prob_up, ensemble_prob_down, ensemble_prob_sideways, regime
        )

        # Generate meta features
        meta_features = self._extract_meta_features(predictions, regime)

        # Determine recommended action
        recommended_action, reasoning = self._determine_action(
            ensemble_prob_up, ensemble_prob_down, ensemble_prob_sideways,
            ensemble_confidence, regime
        )

        # Calculate recommended size based on confidence and regime
        recommended_size = self._calculate_position_size(
            ensemble_confidence, regime, meta_features
        )

        # Calculate risk score
        risk_score = self._calculate_risk_score(meta_features, regime)

        # Create ensemble prediction
        ensemble = EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            ensemble_prob_up=ensemble_prob_up,
            ensemble_prob_down=ensemble_prob_down,
            ensemble_prob_sideways=ensemble_prob_sideways,
            ensemble_confidence=ensemble_confidence,
            model_weights=weights,
            individual_predictions=predictions,
            meta_features=meta_features,
            recommended_action=recommended_action,
            recommended_size=recommended_size,
            reasoning=reasoning,
            risk_score=risk_score
        )

        # Update history
        self._update_history(symbol, ensemble)

        return ensemble

    def _create_empty_ensemble(self, symbol: str, reason: str = "No predictions available") -> EnsemblePrediction:
        """Create empty ensemble prediction"""

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            ensemble_prob_up=0.33,
            ensemble_prob_down=0.33,
            ensemble_prob_sideways=0.34,
            ensemble_confidence=0.0,
            model_weights={},
            individual_predictions=[],
            meta_features={},
            recommended_action="hold",
            recommended_size=0.0,
            reasoning=f"Empty prediction: {reason}",
            risk_score=1.0
        )

    def _calibrate_probabilities(self, prob_up: float, prob_down: float, prob_sideways: float,
                               regime: str) -> Tuple[float, float, float]:
        """Apply temperature scaling for probability calibration"""

        # Temperature scaling based on regime
        temperature_map = {
            "trend": 1.0,
            "range": 1.2,
            "volatile": 0.8,
            "crisis": 0.5
        }

        temperature = temperature_map.get(regime, 1.0)

        # Apply temperature scaling
        if temperature != 1.0:
            prob_up = prob_up ** (1/temperature)
            prob_down = prob_down ** (1/temperature)
            prob_sideways = prob_sideways ** (1/temperature)

            # Renormalize
            total = prob_up + prob_down + prob_sideways
            if total > 0:
                prob_up /= total
                prob_down /= total
                prob_sideways /= total

        return prob_up, prob_down, prob_sideways

    def _extract_meta_features(self, predictions: List[ModelPrediction], regime: str) -> Dict[str, Any]:
        """Extract meta features from individual predictions"""

        if not predictions:
            return {}

        # Agreement metrics
        probs_up = [p.prob_up for p in predictions]
        probs_down = [p.prob_down for p in predictions]

        agreement_up = np.std(probs_up)
        agreement_down = np.std(probs_down)
        overall_agreement = (agreement_up + agreement_down) / 2

        # Diversity metrics
        model_types = [p.model_type for p in predictions]
        unique_types = len(set(model_types))
        type_diversity = unique_types / len(predictions) if predictions else 0

        # Confidence metrics
        avg_confidence = np.mean([p.confidence for p in predictions])
        confidence_std = np.std([p.confidence for p in predictions])

        # Direction consensus
        up_votes = sum(1 for p in predictions if p.prob_up > p.prob_down and p.prob_up > p.prob_sideways)
        down_votes = sum(1 for p in predictions if p.prob_down > p.prob_up and p.prob_down > p.prob_sideways)
        sideways_votes = len(predictions) - up_votes - down_votes

        return {
            "agreement_score": 1.0 - overall_agreement,  # Higher is better
            "model_diversity": type_diversity,
            "avg_confidence": avg_confidence,
            "confidence_consistency": 1.0 - confidence_std,
            "direction_consensus": max(up_votes, down_votes, sideways_votes) / len(predictions),
            "regime": regime,
            "num_models": len(predictions),
            "prediction_horizons": list(set(p.prediction_horizon_minutes for p in predictions))
        }

    def _determine_action(self, prob_up: float, prob_down: float, prob_sideways: float,
                         confidence: float, regime: str) -> Tuple[str, str]:
        """Determine recommended trading action"""

        threshold = self.config["calibration_thresholds"]["action_threshold"]

        # Determine strongest direction
        if prob_up > prob_down and prob_up > prob_sideways and prob_up > threshold:
            action = "buy"
            strength = prob_up
        elif prob_down > prob_up and prob_down > prob_sideways and prob_down > threshold:
            action = "sell"
            strength = prob_down
        else:
            action = "hold"
            strength = max(prob_up, prob_down, prob_sideways)

        # Adjust for regime
        regime_caution = {
            "crisis": 0.1,  # Be very cautious in crisis
            "volatile": 0.05,  # Slightly more cautious in volatile
            "trend": 0.0,   # Normal in trend
            "range": 0.0    # Normal in range
        }

        caution_adjustment = regime_caution.get(regime, 0.0)
        if action != "hold" and strength < threshold + caution_adjustment:
            action = "hold"
            strength = 0.0

        # Generate reasoning
        if action == "buy":
            reasoning = f'Strong buy signal (prob: {prob_up:.2f}, confidence: {confidence:.2f})'
        elif action == "sell":
            reasoning = f'Strong sell signal (prob: {prob_down:.2f}, confidence: {confidence:.2f})'
        elif action == "hold":
            reasoning = f'Hold - insufficient confidence or mixed signals (max_prob: {strength:.2f})'
        else:
            reasoning = "Unknown action determination"

        return action, reasoning

    def _calculate_position_size(self, confidence: float, regime: str, meta_features: Dict) -> float:
        """Calculate recommended position size based on confidence and conditions"""

        # Base size from confidence
        base_size = min(1.0, confidence * 2.0)  # Max 100% of normal size

        # Risk multiplier based on regime
        risk_multiplier = self.config["risk_multipliers"].get(regime, 1.0)

        # Agreement adjustment (smaller size when models disagree)
        agreement_score = meta_features.get("agreement_score", 0.5)
        agreement_multiplier = 0.5 + (agreement_score * 0.5)  # 0.5 to 1.0

        # Model diversity adjustment (larger size with more diverse models)
        diversity = meta_features.get("model_diversity", 0.5)
        diversity_multiplier = 0.8 + (diversity * 0.4)  # 0.8 to 1.2

        # Final size calculation
        position_size = base_size * risk_multiplier * agreement_multiplier * diversity_multiplier

        return max(0.01, min(1.0, position_size))  # Clamp between 1% and 100%

    def _calculate_risk_score(self, meta_features: Dict, regime: str) -> float:
        """Calculate overall risk score (0.0 = low risk, 1.0 = high risk)"""

        # Base risk from regime
        regime_risk = {
            "trend": 0.2,
            "range": 0.1,
            "volatile": 0.7,
            "crisis": 1.0
        }.get(regime, 0.5)

        # Confidence risk (lower confidence = higher risk)
        avg_confidence = meta_features.get("avg_confidence", 0.5)
        confidence_risk = 1.0 - avg_confidence

        # Agreement risk (lower agreement = higher risk)
        agreement_score = meta_features.get("agreement_score", 0.5)
        agreement_risk = 1.0 - agreement_score

        # Model count risk (fewer models = higher risk)
        num_models = meta_features.get("num_models", 1)
        model_risk = max(0, 1.0 - (num_models - 1) * 0.2)  # Risk decreases with more models

        # Combine risks
        overall_risk = (regime_risk * 0.4 +
                       confidence_risk * 0.3 +
                       agreement_risk * 0.2 +
                       model_risk * 0.1)

        return min(1.0, overall_risk)

    def _update_history(self, symbol: str, ensemble: EnsemblePrediction):
        """Update prediction and decision history"""

        # Initialize history if needed
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = deque(maxlen=self.max_history)

        self.prediction_history[symbol].append(ensemble)

        # Store decision for analysis
        decision_entry = {
            "timestamp": ensemble.timestamp.isoformat(),
            "symbol": symbol,
            "action": ensemble.recommended_action,
            "confidence": ensemble.ensemble_confidence,
            "size": ensemble.recommended_size,
            "regime": ensemble.meta_features.get("regime", "unknown")
        }

        self.decision_history.append(decision_entry)

        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]

    def get_decision_stats(self, symbol: str = None, hours: int = 24) -> Dict:
        """Get decision statistics"""

        if symbol:
            history = list(self.prediction_history.get(symbol, []))
        else:
            history = []
            for predictions in self.prediction_history.values():
                history.extend(predictions)

        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [e for e in history if e.timestamp >= cutoff_time]

        if not recent_history:
            return {"total_decisions": 0}

        actions = [e.recommended_action for e in recent_history]
        confidences = [e.ensemble_confidence for e in recent_history]
        sizes = [e.recommended_size for e in recent_history]

        stats = {
            "total_decisions": len(recent_history),
            "action_distribution": {
                "buy": actions.count("buy"),
                "sell": actions.count("sell"),
                "hold": actions.count("hold")
            },
            "avg_confidence": np.mean(confidences),
            "avg_size": np.mean(sizes),
            "confidence_std": np.std(confidences),
            "size_std": np.std(sizes),
            "high_confidence_decisions": sum(1 for c in confidences if c > 0.8),
            "risk_score_trend": "increasing" if len(recent_history) > 1 else "stable"
        }

        # Calculate risk score trend
        if len(recent_history) >= 5:
            recent_risk = [e.risk_score for e in recent_history[-5:]]
            older_risk = [e.risk_score for e in recent_history[-10:-5]]
            if np.mean(recent_risk) > np.mean(older_risk):
                stats["risk_score_trend"] = "increasing"
            elif np.mean(recent_risk) < np.mean(older_risk):
                stats["risk_score_trend"] = "decreasing"
            else:
                stats["risk_score_trend"] = "stable"

        return stats

    def should_take_action(self, ensemble: EnsemblePrediction) -> Tuple[bool, str]:
        """
        Determine if the ensemble prediction warrants taking action

        Returns:
            (take_action: bool, reason: str)
        """

        # Check confidence threshold
        if ensemble.ensemble_confidence < self.config["calibration_thresholds"]["min_confidence"]:
            return False, f'Confidence too low: {ensemble.ensemble_confidence:.2f}'

        # Check action threshold
        max_prob = max(ensemble.ensemble_prob_up, ensemble.ensemble_prob_down, ensemble.ensemble_prob_sideways)
        if max_prob < self.config["calibration_thresholds"]["action_threshold"]:
            return False, f'Maximum probability too low: {max_prob:.2f}'

        # Check risk score
        if ensemble.risk_score > 0.8:
            return False, f'Risk score too high: {ensemble.risk_score:.2f}'

        # Check if recommended action is strong enough
        if ensemble.recommended_action == "hold":
            return False, "Recommended action is hold"

        # All checks passed
        return True, f'Strong {ensemble.recommended_action} signal (conf: {ensemble.ensemble_confidence:.2f}, size: {ensemble.recommended_size:.2f})'

if __name__ == "__main__":
    # Demo usage
    engine = EnsembleDecisionEngine()

    # Create sample predictions
    predictions = [
        ModelPrediction(
            model_id="xgboost_v1",
            model_type="xgboost",
            symbol="EURUSD",
            timestamp=datetime.now(),
            prob_up=0.7,
            prob_down=0.2,
            prob_sideways=0.1,
            confidence=0.8,
            prediction_horizon_minutes=5,
            features_hash="abc123"
        ),
        ModelPrediction(
            model_id="transformer_v1",
            model_type="transformer",
            symbol="EURUSD",
            timestamp=datetime.now(),
            prob_up=0.6,
            prob_down=0.3,
            prob_sideways=0.1,
            confidence=0.75,
            prediction_horizon_minutes=5,
            features_hash="abc123"
        ),
        ModelPrediction(
            model_id="technical_v1",
            model_type="technical",
            symbol="EURUSD",
            timestamp=datetime.now(),
            prob_up=0.65,
            prob_down=0.25,
            prob_sideways=0.1,
            confidence=0.7,
            prediction_horizon_minutes=5,
            features_hash="abc123"
        )
    ]

    # Combine predictions
    ensemble = engine.combine_predictions(predictions, "trend", "EURUSD")

    print(f"Symbol: {ensemble.symbol}")
    print(f"Action: {ensemble.recommended_action}")
    print(f'Confidence: {ensemble.ensemble_confidence:.2f}')
    print(f'Size: {ensemble.recommended_size:.2f}')
    print(f'Probabilities - Up: {ensemble.ensemble_prob_up:.2f}, Down: {ensemble.ensemble_prob_down:.2f}, Sideways: {ensemble.ensemble_prob_sideways:.2f}')
    print(f'Risk Score: {ensemble.risk_score:.2f}')
    print(f"Reasoning: {ensemble.reasoning}")

    # Check if action should be taken
    should_act, reason = engine.should_take_action(ensemble)
    print(f"Should take action: {should_act} - {reason}")

    # Show decision stats
    stats = engine.get_decision_stats("EURUSD")
    print(f'\\nDecision Stats: {stats["total_decisions"]} total decisions, avg confidence: {stats["avg_confidence"]:.2f}')
