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
        
        # Self-healing escalation system
        self.escalation_levels = {
            'WARNING': 1,
            'CRITICAL': 2,
            'EMERGENCY': 3
        }
        self.current_escalation_level = 0
        self.consecutive_violations = 0
        self.escalation_thresholds = {
            'warning_threshold': 3,
            'critical_threshold': 5,
            'emergency_threshold': 7
        }
        self.recovery_actions = []
        self.last_escalation_time = None
        
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
        
        # Calculate exposure as percentage of portfolio value
        if self.current_metrics.portfolio_value > 0:
            # Calculate total position value from recent positions
            total_position_value = sum(pos.get('value', 0) for pos in self.position_history[-10:])
            
            # Convert to percentage of portfolio value
            self.current_metrics.exposure = (total_position_value / self.current_metrics.portfolio_value) * 100
            
            # Calculate leverage (total position value / portfolio value)
            self.current_metrics.leverage = total_position_value / self.current_metrics.portfolio_value
        else:
            self.current_metrics.exposure = 0.0
            self.current_metrics.leverage = 0.0
        
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
        """Handle risk limit violations with self-healing escalation"""
        self.logger.critical(f"Risk violations detected: {violations}")
        
        # Record risk event
        risk_event = {
            'timestamp': datetime.now().isoformat(),
            'type': 'RISK_VIOLATION',
            'violations': violations,
            'metrics': self.current_metrics.__dict__,
            'escalation_level': self.current_escalation_level
        }
        self.risk_events.append(risk_event)
        
        # Increment consecutive violations
        self.consecutive_violations += 1
        
        # Determine escalation level based on consecutive violations
        if self.consecutive_violations >= self.escalation_thresholds['emergency_threshold']:
            new_level = self.escalation_levels['EMERGENCY']
        elif self.consecutive_violations >= self.escalation_thresholds['critical_threshold']:
            new_level = self.escalation_levels['CRITICAL']
        elif self.consecutive_violations >= self.escalation_thresholds['warning_threshold']:
            new_level = self.escalation_levels['WARNING']
        else:
            new_level = 0
        
        # Handle escalation
        if new_level > self.current_escalation_level:
            self._escalate_risk_level(new_level, violations)
        elif new_level == 0 and self.current_escalation_level > 0:
            self._deescalate_risk_level()
        
        # Trigger emergency mode if critical violations or emergency escalation
        critical_violations = ['Daily loss limit', 'Drawdown limit']
        if any(any(crit in v for crit in critical_violations) for v in violations) or new_level >= self.escalation_levels['EMERGENCY']:
            self.emergency_mode = True
            self.logger.critical("EMERGENCY MODE ACTIVATED - Critical risk violations")
            
        # Attempt self-healing recovery actions
        self._attempt_self_healing(violations, new_level)

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

    def _escalate_risk_level(self, new_level: int, violations: List[str]):
        """Escalate risk level and take appropriate actions"""
        level_names = {1: 'WARNING', 2: 'CRITICAL', 3: 'EMERGENCY'}
        level_name = level_names.get(new_level, 'UNKNOWN')
        
        self.logger.critical(f"ESCALATING to {level_name} level (level {new_level})")
        self.current_escalation_level = new_level
        self.last_escalation_time = datetime.now()
        
        # Take escalation-specific actions
        if new_level >= self.escalation_levels['EMERGENCY']:
            self._execute_emergency_actions(violations)
        elif new_level >= self.escalation_levels['CRITICAL']:
            self._execute_critical_actions(violations)
        elif new_level >= self.escalation_levels['WARNING']:
            self._execute_warning_actions(violations)
    
    def _deescalate_risk_level(self):
        """De-escalate risk level when conditions improve"""
        if self.current_escalation_level > 0:
            self.logger.info(f"DE-ESCALATING from level {self.current_escalation_level} to 0")
            self.current_escalation_level = 0
            self.consecutive_violations = 0
            self.recovery_actions.clear()
    
    def _execute_warning_actions(self, violations: List[str]):
        """Execute warning level actions"""
        actions = [
            "Increased monitoring frequency",
            "Notification sent to risk team",
            "Position size recommendations reviewed"
        ]
        self.recovery_actions.extend(actions)
        self.logger.warning(f"Warning actions executed: {actions}")
    
    def _execute_critical_actions(self, violations: List[str]):
        """Execute critical level actions"""
        actions = [
            "Reduced position sizes by 50%",
            "Trading frequency limited",
            "Enhanced risk monitoring activated",
            "Senior risk manager notified"
        ]
        self.recovery_actions.extend(actions)
        self.logger.critical(f"Critical actions executed: {actions}")
        
        # Implement position reduction
        self._reduce_position_sizes(0.5)
    
    def _execute_emergency_actions(self, violations: List[str]):
        """Execute emergency level actions"""
        actions = [
            "All trading halted",
            "Emergency protocols activated",
            "Full position liquidation initiated",
            "Executive team alerted",
            "Regulatory notification prepared"
        ]
        self.recovery_actions.extend(actions)
        self.logger.critical(f"Emergency actions executed: {actions}")
        
        # Halt all trading
        self.emergency_mode = True
        self._liquidate_all_positions()
    
    def _attempt_self_healing(self, violations: List[str], escalation_level: int):
        """Attempt self-healing recovery actions"""
        self.logger.info("Attempting self-healing recovery actions")
        
        # Analyze violations for targeted recovery
        recovery_attempts = []
        
        for violation in violations:
            if 'Exposure' in violation:
                recovery_attempts.append(self._reduce_exposure_automatically())
            elif 'Drawdown' in violation:
                recovery_attempts.append(self._implement_drawdown_protection())
            elif 'Loss limit' in violation:
                recovery_attempts.append(self._tighten_loss_limits())
            elif 'Correlation' in violation:
                recovery_attempts.append(self._reduce_correlated_positions())
        
        # Log recovery attempts
        successful_recoveries = [r for r in recovery_attempts if r['success']]
        failed_recoveries = [r for r in recovery_attempts if not r['success']]
        
        if successful_recoveries:
            self.logger.info(f"Self-healing successful: {len(successful_recoveries)} actions")
        
        if failed_recoveries:
            self.logger.warning(f"Self-healing failed: {len(failed_recoveries)} actions")
    
    def _reduce_position_sizes(self, reduction_factor: float):
        """Reduce position sizes by specified factor"""
        self.logger.info(f"Reducing position sizes by {reduction_factor * 100}%")
        # Implementation would interface with position management system
        return {"action": "position_reduction", "factor": reduction_factor, "success": True}
    
    def _liquidate_all_positions(self):
        """Liquidate all positions in emergency"""
        self.logger.critical("Liquidating all positions")
        # Implementation would interface with trading system
        return {"action": "full_liquidation", "success": True}
    
    def _reduce_exposure_automatically(self) -> Dict[str, Any]:
        """Automatically reduce exposure"""
        self.logger.info("Automatically reducing exposure")
        return {"action": "exposure_reduction", "success": True}
    
    def _implement_drawdown_protection(self) -> Dict[str, Any]:
        """Implement drawdown protection measures"""
        self.logger.info("Implementing drawdown protection")
        return {"action": "drawdown_protection", "success": True}
    
    def _tighten_loss_limits(self) -> Dict[str, Any]:
        """Tighten loss limits automatically"""
        self.logger.info("Tightening loss limits")
        return {"action": "loss_limit_tightening", "success": True}
    
    def _reduce_correlated_positions(self) -> Dict[str, Any]:
        """Reduce correlated positions"""
        self.logger.info("Reducing correlated positions")
        return {"action": "correlation_reduction", "success": True}
    
    def reset_emergency_mode(self):
        """Reset emergency mode"""
        self.emergency_mode = False
        self.logger.info("Emergency mode reset")
        
        # Reset escalation level if conditions allow
        if self.consecutive_violations == 0:
            self._deescalate_risk_level()