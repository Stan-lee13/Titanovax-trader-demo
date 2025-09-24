#!/usr/bin/env python3
"""
Risk Management Module for TitanovaX Orchestration
Integrated risk management with orchestration system
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
class RiskMetrics:
    """Current risk metrics"""
    portfolio_value: float
    daily_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    exposure: float
    leverage: float
    correlation_risk: float

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_daily_loss: float
    max_drawdown: float
    max_exposure: float
    max_leverage: float
    min_liquidity_buffer: float
    max_correlation_risk: float
    var_limit: float
    max_position_size: float

class RiskManager:
    """
    Integrated Risk Management for TitanovaX
    """

    def __init__(self, config_path: str = 'config/risk_config.json'):
        self.config_path = Path(config_path)
        self.risk_limits = self._load_default_limits()
        self.current_metrics = RiskMetrics(
            portfolio_value=100000.0,
            daily_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            var_95=0.0,
            exposure=0.0,
            leverage=1.0,
            correlation_risk=0.0
        )
        self.position_history: List[Dict[str, Any]] = []
        self.risk_events: List[Dict[str, Any]] = []
        self.emergency_mode = False
        
        self.load_config()
        self.setup_logging()
        self.start_risk_monitoring()

    def _load_default_limits(self) -> RiskLimits:
        """Load default risk limits"""
        return RiskLimits(
            max_daily_loss=1000.0,
            max_drawdown=5.0,
            max_exposure=80.0,
            max_leverage=2.0,
            min_liquidity_buffer=10000.0,
            max_correlation_risk=0.7,
            var_limit=2000.0,
            max_position_size=100000.0
        )

    def load_config(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    limits_data = config.get('risk_limits', {})
                    self.risk_limits = RiskLimits(**{**self.risk_limits.__dict__, **limits_data})
            else:
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load risk config: {e}")

    def save_config(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            'risk_limits': self.risk_limits.__dict__,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def start_risk_monitoring(self):
        """Start risk monitoring thread"""
        self.monitor_thread = threading.Thread(target=self._risk_monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _risk_monitor_loop(self):
        """Risk monitoring loop"""
        while True:
            try:
                self._update_risk_metrics()
                self._check_risk_limits()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Risk monitor error: {e}")
                time.sleep(10)

    def _update_risk_metrics(self):
        """Update current risk metrics"""
        # This would connect to actual trading data in production
        # For now, we'll simulate with basic calculations
        
        # Calculate exposure based on position history
        total_exposure = sum(pos.get('value', 0) for pos in self.position_history[-10:])
        self.current_metrics.exposure = total_exposure
        
        # Calculate leverage
        if self.current_metrics.portfolio_value > 0:
            self.current_metrics.leverage = total_exposure / self.current_metrics.portfolio_value
        
        # Update drawdown
        if self.position_history:
            peak_value = max(pos.get('portfolio_value', self.current_metrics.portfolio_value) 
                           for pos in self.position_history[-50:])
            if peak_value > 0:
                current_value = self.current_metrics.portfolio_value
                self.current_metrics.max_drawdown = max(0.0, (peak_value - current_value) / peak_value * 100)

    def _check_risk_limits(self):
        """Check if risk limits are exceeded"""
        violations = []
        
        # Check daily loss limit
        if abs(self.current_metrics.daily_pnl) > self.risk_limits.max_daily_loss:
            violations.append(f"Daily loss limit exceeded: {abs(self.current_metrics.daily_pnl):.2f} > {self.risk_limits.max_daily_loss:.2f}")
        
        # Check drawdown limit
        if self.current_metrics.max_drawdown > self.risk_limits.max_drawdown:
            violations.append(f"Drawdown limit exceeded: {self.current_metrics.max_drawdown:.2f}% > {self.risk_limits.max_drawdown:.2f}%")
        
        # Check exposure limit
        if self.current_metrics.exposure > self.risk_limits.max_exposure:
            violations.append(f"Exposure limit exceeded: {self.current_metrics.exposure:.2f}% > {self.risk_limits.max_exposure:.2f}%")
        
        # Check leverage limit
        if self.current_metrics.leverage > self.risk_limits.max_leverage:
            violations.append(f"Leverage limit exceeded: {self.current_metrics.leverage:.2f} > {self.risk_limits.max_leverage:.2f}")
        
        # Handle violations
        if violations:
            self._handle_risk_violations(violations)

    def _handle_risk_violations(self, violations: List[str]):
        """Handle risk limit violations"""
        self.logger.critical(f"Risk violations detected: {violations}")
        
        # Record risk event
        risk_event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'RISK_VIOLATION',
            'violations': violations,
            'metrics': self.current_metrics.__dict__
        }
        self.risk_events.append(risk_event)
        
        # Trigger emergency mode if critical violations
        critical_violations = ['Daily loss limit', 'Drawdown limit']
        if any(any(crit in v for crit in critical_violations) for v in violations):
            self.emergency_mode = True
            self.logger.critical("EMERGENCY MODE ACTIVATED - Critical risk violations")

    def evaluate_trade_risk(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk of a proposed trade
        
        Args:
            trade_request: Trade request with symbol, size, direction, etc.
            
        Returns:
            Risk assessment result
        """
        if self.emergency_mode:
            return {
                'approved': False,
                'risk_level': 'CRITICAL',
                'reasoning': 'Emergency mode active',
                'risk_metrics': self.current_metrics.__dict__,
                'max_position_size': self.risk_limits.max_position_size
            }
        
        # Calculate trade impact
        trade_size = trade_request.get('size', 0)
        trade_value = trade_request.get('value', 0)
        
        # Simulate risk calculation
        projected_exposure = self.current_metrics.exposure + (trade_value / self.current_metrics.portfolio_value * 100)
        projected_var = self.current_metrics.var_95 * 1.1  # Simple projection
        
        risk_factors = []
        
        # Check if trade would exceed exposure limit
        if projected_exposure > self.risk_limits.max_exposure:
            risk_factors.append(f"Would exceed exposure limit: {projected_exposure:.2f}%")
        
        # Check liquidity impact
        if trade_value > self.current_metrics.portfolio_value - self.risk_limits.min_liquidity_buffer:
            risk_factors.append("Would breach liquidity buffer")
        
        # Check position size limit
        if trade_size > self.risk_limits.max_position_size:
            risk_factors.append(f"Would exceed max position size: {trade_size} > {self.risk_limits.max_position_size}")
        
        # Determine risk level
        if risk_factors:
            risk_level = 'HIGH'
            approved = False
        elif projected_exposure > self.risk_limits.max_exposure * 0.8:
            risk_level = 'MEDIUM'
            approved = True
        else:
            risk_level = 'LOW'
            approved = True
        
        return {
            'approved': approved,
            'risk_level': risk_level,
            'reasoning': f"Trade risk assessment: {', '.join(risk_factors) if risk_factors else 'Within limits'}",
            'projected_metrics': {
                'exposure': projected_exposure,
                'var_95': projected_var
            },
            'current_metrics': self.current_metrics.__dict__,
            'max_position_size': self.risk_limits.max_position_size
        }

    def update_position(self, position_data: Dict[str, Any]):
        """Update with new position data"""
        self.position_history.append({
            'timestamp': datetime.now().isoformat(),
            **position_data
        })
        
        # Keep only recent history
        max_history = 1000
        if len(self.position_history) > max_history:
            self.position_history = self.position_history[-max_history:]

    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': self.current_metrics.__dict__,
            'risk_limits': self.risk_limits.__dict__,
            'system_status': {
                'emergency_mode': self.emergency_mode,
                'violations_last_24h': len([e for e in self.risk_events 
                                          if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=24)]),
                'total_violations': len(self.risk_events)
            },
            'recent_events': self.risk_events[-10:],  # Last 10 events
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate risk recommendations"""
        recommendations = []
        
        if self.emergency_mode:
            recommendations.append("Emergency mode active - stop all trading")
        
        if self.current_metrics.max_drawdown > self.risk_limits.max_drawdown * 0.8:
            recommendations.append("High drawdown - reduce position sizes")
        
        if self.current_metrics.exposure > self.risk_limits.max_exposure * 0.9:
            recommendations.append("High exposure - consider reducing positions")
        
        if self.current_metrics.leverage > self.risk_limits.max_leverage * 0.8:
            recommendations.append("High leverage - monitor closely")
        
        if not recommendations:
            recommendations.append("Risk levels within acceptable limits")
        
        return recommendations

    def reset_emergency_mode(self):
        """Reset emergency mode after risk conditions improve"""
        if self.emergency_mode:
            self.emergency_mode = False
            self.logger.info("Emergency mode reset - risk conditions improved")
            
            # Record reset event
            self.risk_events.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'EMERGENCY_RESET',
                'reasoning': 'Risk conditions resolved'
            })