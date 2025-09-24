#!/usr/bin/env python3
"""
Model Gating System for TitanovaX
Controls model access to live trading based on performance and risk metrics
"""

import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    last_updated: datetime
    confidence_score: float
    risk_adjusted_score: float

@dataclass
class GateDecision:
    """Gate decision result"""
    approved: bool
    confidence: float
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str
    timestamp: datetime

class ModelGatingSystem:
    """
    Model Gating System - Controls model access to live trading
    """

    def __init__(self, config_path: str = 'config/model_gating_config.json'):
        self.config_path = Path(config_path)
        self.models: Dict[str, ModelPerformance] = {}
        self.gate_thresholds = self._load_default_thresholds()
        self.trading_enabled = True
        self.emergency_stop = False
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        
        self.load_config()
        self.setup_logging()
        self.start_performance_monitor()

    def _load_default_thresholds(self) -> Dict[str, float]:
        """Load default gate thresholds"""
        return {
            'min_win_rate': 0.55,           # Minimum win rate (55%)
            'min_sharpe_ratio': 0.5,        # Minimum Sharpe ratio
            'max_drawdown': 0.15,           # Maximum drawdown (15%)
            'min_confidence_score': 0.7,    # Minimum confidence score
            'min_trades_for_validation': 20, # Minimum trades before validation
            'max_risk_score': 0.3,          # Maximum risk score
            'performance_window_days': 30,    # Performance evaluation window
            'emergency_threshold': 0.2       # Emergency stop threshold
        }

    def load_config(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.gate_thresholds.update(config.get('thresholds', {}))
                    self.trading_enabled = config.get('trading_enabled', True)
            else:
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load model gating config: {e}")

    def save_config(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            'thresholds': self.gate_thresholds,
            'trading_enabled': self.trading_enabled,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def start_performance_monitor(self):
        """Start performance monitoring thread"""
        self.monitor_thread = threading.Thread(target=self._performance_monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        while True:
            try:
                self._update_model_performance()
                self._evaluate_trading_conditions()
                time.sleep(60)  # Update every minute
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                time.sleep(10)

    def _update_model_performance(self):
        """Update model performance metrics"""
        current_time = datetime.now()
        
        for model_id, model_perf in self.models.items():
            # Calculate risk-adjusted score
            risk_score = self._calculate_risk_score(model_perf)
            model_perf.risk_adjusted_score = model_perf.confidence_score * (1 - risk_score)
            model_perf.last_updated = current_time
            
            # Store in history
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            self.performance_history[model_id].append(model_perf.copy())
            
            # Keep only recent history
            window_days = self.gate_thresholds['performance_window_days']
            cutoff_time = current_time - timedelta(days=window_days)
            self.performance_history[model_id] = [
                perf for perf in self.performance_history[model_id]
                if perf.last_updated > cutoff_time
            ]

    def _calculate_risk_score(self, model_perf: ModelPerformance) -> float:
        """Calculate risk score for a model"""
        risk_factors = []
        
        # Drawdown risk
        if model_perf.max_drawdown > self.gate_thresholds['max_drawdown']:
            risk_factors.append(1.0)
        else:
            risk_factors.append(model_perf.max_drawdown / self.gate_thresholds['max_drawdown'])
        
        # Win rate risk
        if model_perf.win_rate < self.gate_thresholds['min_win_rate']:
            risk_factors.append(1.0)
        else:
            risk_factors.append(1.0 - (model_perf.win_rate - self.gate_thresholds['min_win_rate']))
        
        # Sharpe ratio risk
        if model_perf.sharpe_ratio < self.gate_thresholds['min_sharpe_ratio']:
            risk_factors.append(1.0)
        else:
            risk_factors.append(max(0.0, 1.0 - model_perf.sharpe_ratio))
        
        return np.mean(risk_factors)

    def _evaluate_trading_conditions(self):
        """Evaluate overall trading conditions"""
        if not self.models:
            return
        
        # Calculate average performance metrics
        avg_confidence = np.mean([m.confidence_score for m in self.models.values()])
        avg_risk_score = np.mean([self._calculate_risk_score(m) for m in self.models.values()])
        
        # Check emergency conditions
        if avg_risk_score > self.gate_thresholds['emergency_threshold']:
            self.logger.critical(f"Emergency condition detected: avg risk score {avg_risk_score:.2f}")
            self.emergency_stop = True
            self.trading_enabled = False
        
        # Check if we should resume trading
        elif self.emergency_stop and avg_risk_score < self.gate_thresholds['emergency_threshold'] * 0.5:
            self.logger.info("Emergency conditions resolved - resuming trading")
            self.emergency_stop = False
            self.trading_enabled = True

    def register_model(self, model_id: str, initial_performance: Optional[ModelPerformance] = None):
        """Register a new model with the gating system"""
        if initial_performance:
            self.models[model_id] = initial_performance
        else:
            # Create default performance metrics
            self.models[model_id] = ModelPerformance(
                model_id=model_id,
                win_rate=0.0,
                avg_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                last_updated=datetime.now(),
                confidence_score=0.5,
                risk_adjusted_score=0.5
            )
        
        self.logger.info(f"Model {model_id} registered with gating system")

    def update_model_performance(self, model_id: str, performance_data: Dict[str, Any]):
        """Update model performance metrics"""
        if model_id not in self.models:
            self.register_model(model_id)
        
        model_perf = self.models[model_id]
        
        # Update metrics
        if 'win_rate' in performance_data:
            model_perf.win_rate = performance_data['win_rate']
        if 'avg_return' in performance_data:
            model_perf.avg_return = performance_data['avg_return']
        if 'sharpe_ratio' in performance_data:
            model_perf.sharpe_ratio = performance_data['sharpe_ratio']
        if 'max_drawdown' in performance_data:
            model_perf.max_drawdown = performance_data['max_drawdown']
        if 'total_trades' in performance_data:
            model_perf.total_trades = performance_data['total_trades']
        if 'confidence_score' in performance_data:
            model_perf.confidence_score = performance_data['confidence_score']
        
        model_perf.last_updated = datetime.now()

    def evaluate_model(self, model_id: str, market_conditions: Optional[Dict[str, Any]] = None) -> GateDecision:
        """
        Evaluate if a model should be allowed to trade
        
        Returns:
            GateDecision: Approval decision with confidence and reasoning
        """
        
        if self.emergency_stop:
            return GateDecision(
                approved=False,
                confidence=0.0,
                reasoning="Emergency stop activated",
                risk_level="CRITICAL",
                recommended_action="STOP_ALL_TRADING",
                timestamp=datetime.now()
            )
        
        if not self.trading_enabled:
            return GateDecision(
                approved=False,
                confidence=0.0,
                reasoning="Trading disabled by system",
                risk_level="HIGH",
                recommended_action="WAIT_FOR_CONDITIONS",
                timestamp=datetime.now()
            )
        
        if model_id not in self.models:
            return GateDecision(
                approved=False,
                confidence=0.0,
                reasoning=f"Model {model_id} not registered",
                risk_level="HIGH",
                recommended_action="REGISTER_MODEL",
                timestamp=datetime.now()
            )
        
        model_perf = self.models[model_id]
        
        # Check minimum trade requirement
        if model_perf.total_trades < self.gate_thresholds['min_trades_for_validation']:
            return GateDecision(
                approved=True,
                confidence=0.3,
                reasoning=f"Model has insufficient trades ({model_perf.total_trades}) for full validation",
                risk_level="MEDIUM",
                recommended_action="ALLOW_WITH_CAUTION",
                timestamp=datetime.now()
            )
        
        # Evaluate against thresholds
        checks = []
        
        # Win rate check
        if model_perf.win_rate >= self.gate_thresholds['min_win_rate']:
            checks.append(('win_rate', True, model_perf.win_rate))
        else:
            checks.append(('win_rate', False, model_perf.win_rate))
        
        # Sharpe ratio check
        if model_perf.sharpe_ratio >= self.gate_thresholds['min_sharpe_ratio']:
            checks.append(('sharpe_ratio', True, model_perf.sharpe_ratio))
        else:
            checks.append(('sharpe_ratio', False, model_perf.sharpe_ratio))
        
        # Drawdown check
        if model_perf.max_drawdown <= self.gate_thresholds['max_drawdown']:
            checks.append(('max_drawdown', True, model_perf.max_drawdown))
        else:
            checks.append(('max_drawdown', False, model_perf.max_drawdown))
        
        # Confidence score check
        if model_perf.confidence_score >= self.gate_thresholds['min_confidence_score']:
            checks.append(('confidence_score', True, model_perf.confidence_score))
        else:
            checks.append(('confidence_score', False, model_perf.confidence_score))
        
        # Calculate overall score
        passed_checks = sum(1 for _, passed, _ in checks if passed)
        total_checks = len(checks)
        approval_rate = passed_checks / total_checks
        
        # Determine decision
        if approval_rate >= 0.8:  # 80% of checks passed
            approved = True
            confidence = model_perf.confidence_score
            risk_level = "LOW"
            recommended_action = "ALLOW_TRADING"
            reasoning = f"Model passed {passed_checks}/{total_checks} checks"
        elif approval_rate >= 0.5:  # 50% of checks passed
            approved = True
            confidence = model_perf.confidence_score * 0.7
            risk_level = "MEDIUM"
            recommended_action = "ALLOW_WITH_CAUTION"
            reasoning = f"Model passed {passed_checks}/{total_checks} checks with caution"
        else:
            approved = False
            confidence = model_perf.confidence_score * 0.3
            risk_level = "HIGH"
            recommended_action = "BLOCK_TRADING"
            reasoning = f"Model failed {total_checks - passed_checks}/{total_checks} checks"
        
        # Add detailed check results to reasoning
        failed_checks = [name for name, passed, value in checks if not passed]
        if failed_checks:
            reasoning += f" - Failed: {', '.join(failed_checks)}"
        
        return GateDecision(
            approved=approved,
            confidence=confidence,
            reasoning=reasoning,
            risk_level=risk_level,
            recommended_action=recommended_action,
            timestamp=datetime.now()
        )

    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a model"""
        if model_id not in self.models:
            return None
        
        model_perf = self.models[model_id]
        risk_score = self._calculate_risk_score(model_perf)
        
        return {
            'model_id': model_id,
            'performance': {
                'win_rate': model_perf.win_rate,
                'avg_return': model_perf.avg_return,
                'sharpe_ratio': model_perf.sharpe_ratio,
                'max_drawdown': model_perf.max_drawdown,
                'total_trades': model_perf.total_trades,
                'confidence_score': model_perf.confidence_score,
                'risk_adjusted_score': model_perf.risk_adjusted_score
            },
            'risk_metrics': {
                'risk_score': risk_score,
                'risk_level': 'LOW' if risk_score < 0.3 else 'MEDIUM' if risk_score < 0.6 else 'HIGH'
            },
            'trading_status': {
                'system_enabled': self.trading_enabled,
                'emergency_stop': self.emergency_stop,
                'last_updated': model_perf.last_updated.isoformat()
            }
        }

    def get_all_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered models"""
        return {model_id: self.get_model_status(model_id) for model_id in self.models.keys()}

    def force_stop_all_trading(self, reason: str = "Manual intervention"):
        """Force stop all trading"""
        self.emergency_stop = True
        self.trading_enabled = False
        self.logger.critical(f"All trading stopped: {reason}")

    def resume_trading(self, reason: str = "Conditions resolved"):
        """Resume trading"""
        if not self.emergency_stop:
            self.trading_enabled = True
            self.logger.info(f"Trading resumed: {reason}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.models:
            return {
                'status': 'NO_MODELS',
                'trading_enabled': self.trading_enabled,
                'emergency_stop': self.emergency_stop,
                'active_models': 0,
                'avg_confidence': 0.0,
                'avg_risk_score': 0.0,
                'recommendation': 'Register models to begin'
            }
        
        avg_confidence = np.mean([m.confidence_score for m in self.models.values()])
        avg_risk_score = np.mean([self._calculate_risk_score(m) for m in self.models.values()])
        
        if self.emergency_stop:
            status = 'EMERGENCY_STOP'
            recommendation = 'Emergency stop active - resolve conditions'
        elif not self.trading_enabled:
            status = 'TRADING_DISABLED'
            recommendation = 'Trading disabled by system'
        elif avg_risk_score > self.gate_thresholds['emergency_threshold']:
            status = 'HIGH_RISK'
            recommendation = 'High risk conditions - monitor closely'
        elif avg_confidence < 0.5:
            status = 'LOW_CONFIDENCE'
            recommendation = 'Low model confidence - reduce exposure'
        else:
            status = 'HEALTHY'
            recommendation = 'System operating normally'
        
        return {
            'status': status,
            'trading_enabled': self.trading_enabled,
            'emergency_stop': self.emergency_stop,
            'active_models': len(self.models),
            'avg_confidence': avg_confidence,
            'avg_risk_score': avg_risk_score,
            'recommendation': recommendation,
            'last_updated': datetime.now().isoformat()
        }