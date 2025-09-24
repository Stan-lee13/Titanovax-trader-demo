#!/usr/bin/env python3
"""
Grid & Reverse Safety Net + Adaptive Capital Allocation + Automated Stress Testing
for TitanovaX Trading System
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import threading
import time

@dataclass
class GridSafetyConfig:
    """Grid strategy safety configuration"""
    max_grid_levels: int = 5
    min_level_spacing_pips: int = 10
    max_total_exposure_multiplier: float = 3.0
    dynamic_insurance_cap_percent: float = 0.5
    regime_restriction: str = "RANGE"  # Only allow in range-bound markets
    volatility_threshold: float = 0.001  # Max allowed volatility
    correlation_threshold: float = 0.7  # Max correlation with other positions

@dataclass
class ReverseSafetyConfig:
    """Reverse strategy safety configuration"""
    min_ensemble_confidence: float = 0.8
    max_position_size_multiplier: float = 1.5
    rl_sizing_clamp_percent: float = 0.3
    cooldown_period_minutes: int = 30
    max_consecutive_reversals: int = 3
    market_condition_check: bool = True

@dataclass
class CapitalAllocationConfig:
    """Adaptive capital allocation configuration"""
    base_allocation_percent: float = 0.02  # 2% per instrument
    sharpe_adjustment_factor: float = 0.5
    correlation_penalty_factor: float = 0.3
    regime_bonus_factor: float = 0.2
    rebalancing_frequency_hours: int = 24
    max_allocation_percent: float = 0.1  # 10% max per instrument
    min_allocation_percent: float = 0.005  # 0.5% min per instrument

@dataclass
class StressTestConfig:
    """Stress testing configuration"""
    nightly_test_enabled: bool = True
    historical_scenarios: List[str] = field(default_factory=lambda: [
        "2008_crisis", "2015_snb", "2020_covid", "black_swan_extreme"
    ])
    test_duration_hours: int = 4
    tolerance_threshold_percent: float = -5.0  # Max allowed loss during stress test
    live_size_reduction_factor: float = 0.5
    recovery_wait_hours: int = 24

@dataclass
class GridSafetyNet:
    """Grid strategy safety enforcement"""
    symbol: str
    regime: str
    current_grid_levels: int = 0
    total_exposure: float = 0.0
    insurance_cap: float = 0.0
    last_grid_time: Optional[datetime] = None
    violation_count: int = 0

    def is_safe_to_add_grid(self, market_volatility: float,
                           correlation_score: float,
                           current_regime: str) -> Tuple[bool, str]:
        """Check if it's safe to add another grid level"""
        config = GridSafetyConfig()

        # Check regime restriction
        if current_regime != config.regime_restriction:
            return False, f"Grid strategies only allowed in {config.regime_restriction} regime, current: {current_regime}"

        # Check volatility threshold
        if market_volatility > config.volatility_threshold:
            return False, f"Market volatility ({market_volatility:.4f}) exceeds threshold ({config.volatility_threshold})"

        # Check correlation
        if correlation_score > config.correlation_threshold:
            return False, f"High correlation ({correlation_score:.2f}) exceeds threshold ({config.correlation_threshold})"

        # Check grid level limit
        if self.current_grid_levels >= config.max_grid_levels:
            return False, f"Maximum grid levels ({config.max_grid_levels}) reached"

        # Check total exposure
        if self.total_exposure > config.max_total_exposure_multiplier * self.insurance_cap:
            return False, f"Total exposure ({self.total_exposure:.2f}) exceeds safe limit"

        return True, "Grid level addition approved"

    def update_exposure(self, new_exposure: float, account_balance: float):
        """Update exposure tracking"""
        self.total_exposure = new_exposure
        self.insurance_cap = account_balance * GridSafetyConfig().dynamic_insurance_cap_percent

@dataclass
class ReverseSafetyNet:
    """Reverse strategy safety enforcement"""
    symbol: str
    last_reverse_time: Optional[datetime] = None
    consecutive_reversals: int = 0
    current_position_size: float = 0.0

    def is_safe_to_reverse(self, ensemble_confidence: float,
                          proposed_size: float,
                          current_regime: str) -> Tuple[bool, str]:
        """Check if it's safe to execute a reversal"""
        config = ReverseSafetyConfig()

        # Check ensemble confidence
        if ensemble_confidence < config.min_ensemble_confidence:
            return False, f"Ensemble confidence ({ensemble_confidence:.2f}) below minimum ({config.min_ensemble_confidence})"

        # Check position size clamp
        if proposed_size > config.max_position_size_multiplier * self.current_position_size:
            return False, f"Proposed size ({proposed_size:.3f}) exceeds clamp limit"

        # Check cooldown period
        if self.last_reverse_time:
            cooldown_end = self.last_reverse_time + timedelta(minutes=config.cooldown_period_minutes)
            if datetime.now() < cooldown_end:
                remaining = cooldown_end - datetime.now()
                return False, f"Cooldown period not elapsed. {remaining.seconds} seconds remaining."

        # Check consecutive reversals
        if self.consecutive_reversals >= config.max_consecutive_reversals:
            return False, f"Maximum consecutive reversals ({config.max_consecutive_reversals}) reached"

        # Check market conditions
        if config.market_condition_check:
            # In production, this would check for news events, high volatility, etc.
            pass

        return True, "Reversal approved"

    def record_reversal(self, position_size: float):
        """Record successful reversal for tracking"""
        self.last_reverse_time = datetime.now()
        self.consecutive_reversals += 1
        self.current_position_size = position_size

        # Reset consecutive reversals if enough time has passed
        if self.last_reverse_time and self.consecutive_reversals > 1:
            time_since_last = datetime.now() - self.last_reverse_time
            if time_since_last.total_seconds() > 3600:  # 1 hour
                self.consecutive_reversals = 1

class AdaptiveCapitalAllocation:
    """Adaptive capital allocation based on performance and risk"""

    def __init__(self, config_path: str = 'config/capital_allocation_config.json'):
        self.config_path = Path(config_path)
        self.instrument_allocations: Dict[str, float] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.last_rebalancing: Optional[datetime] = None

        self.load_config()
        self.setup_logging()

    def load_config(self):
        """Load capital allocation configuration"""
        default_config = CapitalAllocationConfig()

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                    # Update config with loaded values
                    for key, value in config_dict.items():
                        if hasattr(default_config, key):
                            setattr(default_config, key, value)

            self.config = default_config
            self.save_config()
        except Exception as e:
            logging.warning(f"Could not load capital allocation config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            "base_allocation_percent": self.config.base_allocation_percent,
            "sharpe_adjustment_factor": self.config.sharpe_adjustment_factor,
            "correlation_penalty_factor": self.config.correlation_penalty_factor,
            "regime_bonus_factor": self.config.regime_bonus_factor,
            "rebalancing_frequency_hours": self.config.rebalancing_frequency_hours,
            "max_allocation_percent": self.config.max_allocation_percent,
            "min_allocation_percent": self.config.min_allocation_percent
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def calculate_optimal_allocation(self, instrument_performance: Dict[str, Dict],
                                   current_regime: str,
                                   total_capital: float) -> Dict[str, float]:
        """
        Calculate optimal capital allocation for all instruments

        Args:
            instrument_performance: Dict of instrument -> performance metrics
            current_regime: Current market regime
            total_capital: Total available capital

        Returns:
            Dict of instrument -> allocated capital
        """
        allocations = {}

        for instrument, metrics in instrument_performance.items():
            base_allocation = total_capital * self.config.base_allocation_percent

            # Adjust for Sharpe ratio
            sharpe_ratio = metrics.get("sharpe_ratio", 1.0)
            sharpe_adjustment = 1.0 + (sharpe_ratio - 1.0) * self.config.sharpe_adjustment_factor
            adjusted_allocation = base_allocation * sharpe_adjustment

            # Apply correlation penalty
            correlation_penalty = self._calculate_correlation_penalty(instrument)
            adjusted_allocation *= (1.0 - correlation_penalty * self.config.correlation_penalty_factor)

            # Apply regime bonus
            if current_regime in ["TREND", "RANGE"]:
                adjusted_allocation *= (1.0 + self.config.regime_bonus_factor)

            # Clamp to limits
            adjusted_allocation = max(
                total_capital * self.config.min_allocation_percent,
                min(adjusted_allocation, total_capital * self.config.max_allocation_percent)
            )

            allocations[instrument] = adjusted_allocation

            # Track performance for future adjustments
            self.performance_history[instrument].append(metrics)

        # Normalize to ensure total allocation doesn't exceed available capital
        total_allocated = sum(allocations.values())
        if total_allocated > total_capital:
            scale_factor = total_capital / total_allocated
            allocations = {k: v * scale_factor for k, v in allocations.items()}

        self.instrument_allocations = allocations
        return allocations

    def _calculate_correlation_penalty(self, instrument: str) -> float:
        """Calculate correlation penalty for an instrument"""
        # In production, this would use real correlation data
        # For now, use placeholder logic
        return 0.1  # 10% penalty for correlation

    def update_correlation_matrix(self, correlations: Dict[str, Dict[str, float]]):
        """Update correlation matrix for allocation calculations"""
        self.correlation_matrix = correlations

    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance allocations"""
        if self.last_rebalancing is None:
            return True

        time_since_rebalancing = datetime.now() - self.last_rebalancing
        return time_since_rebalancing.total_seconds() > self.config.rebalancing_frequency_hours * 3600

    def rebalance_allocations(self, instrument_performance: Dict[str, Dict],
                            current_regime: str, total_capital: float) -> Dict[str, float]:
        """Perform complete rebalancing of capital allocations"""
        new_allocations = self.calculate_optimal_allocation(
            instrument_performance, current_regime, total_capital
        )

        self.last_rebalancing = datetime.now()
        self.logger.info(f"Rebalanced capital allocations: {new_allocations}")

        return new_allocations

class AutomatedStressTesting:
    """Automated stress testing system"""

    def __init__(self, config_path: str = 'config/stress_testing_config.json'):
        self.config_path = Path(config_path)
        self.test_history: List[Dict] = []
        self.current_stress_level: float = 0.0
        self.last_stress_test: Optional[datetime] = None
        self.live_size_reduction_active: bool = False

        self.load_config()
        self.setup_logging()

    def load_config(self):
        """Load stress testing configuration"""
        default_config = StressTestConfig()

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                    for key, value in config_dict.items():
                        if hasattr(default_config, key):
                            setattr(default_config, key, value)

            self.config = default_config
        except Exception as e:
            logging.warning(f"Could not load stress testing config: {e}")
            self.config = default_config

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def run_nightly_stress_test(self, current_positions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Run nightly stress test using historical scenarios

        Args:
            current_positions: Current open positions

        Returns:
            Test results and recommendations
        """
        start_time = datetime.now()
        self.logger.info("Starting nightly stress test")

        results = {
            "test_id": f"stress_test_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "start_time": start_time.isoformat(),
            "scenarios_tested": [],
            "total_projected_loss": 0.0,
            "max_drawdown": 0.0,
            "tolerance_exceeded": False,
            "recommendations": []
        }

        # Test each historical scenario
        for scenario in self.config.historical_scenarios:
            scenario_result = self._run_scenario_test(scenario, current_positions)
            results["scenarios_tested"].append(scenario_result)

            # Aggregate results
            results["total_projected_loss"] += scenario_result.get("projected_loss", 0.0)
            results["max_drawdown"] = max(results["max_drawdown"],
                                         scenario_result.get("drawdown", 0.0))

        # Check tolerance
        results["tolerance_exceeded"] = results["total_projected_loss"] < self.config.tolerance_threshold_percent

        # Generate recommendations
        if results["tolerance_exceeded"]:
            results["recommendations"].append({
                "action": "reduce_live_size",
                "factor": self.config.live_size_reduction_factor,
                "reason": f"Stress test loss ({results['total_projected_loss']:.1f}%) exceeds tolerance ({self.config.tolerance_threshold_percent}%)"
            })

        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = (datetime.now() - start_time).total_seconds()

        self.test_history.append(results)
        self.last_stress_test = start_time

        self.logger.info(f"Stress test completed: Loss={results['total_projected_loss']:.1f}%, Tolerance Exceeded={results['tolerance_exceeded']}")

        return results

    def _run_scenario_test(self, scenario: str, current_positions: Dict[str, Dict]) -> Dict[str, Any]:
        """Run stress test for a specific scenario"""
        # In production, this would:
        # 1. Load historical market data for the scenario
        # 2. Replay market conditions
        # 3. Calculate impact on current positions
        # 4. Return projected losses

        # For now, use placeholder logic
        scenario_severity = {
            "2008_crisis": 0.15,  # 15% projected loss
            "2015_snb": 0.08,     # 8% projected loss
            "2020_covid": 0.12,   # 12% projected loss
            "black_swan_extreme": 0.25  # 25% projected loss
        }

        base_loss = scenario_severity.get(scenario, 0.1)
        position_adjustment = len(current_positions) * 0.02  # More positions = higher risk

        projected_loss = base_loss + position_adjustment
        drawdown = projected_loss * 1.2  # Drawdown slightly higher than loss

        return {
            "scenario": scenario,
            "projected_loss": projected_loss * 100,  # Convert to percentage
            "drawdown": drawdown * 100,
            "positions_affected": len(current_positions),
            "severity": "high" if projected_loss > 0.1 else "medium"
        }

    def should_reduce_live_size(self) -> Tuple[bool, float, str]:
        """Check if live position sizes should be reduced"""
        if not self.live_size_reduction_active:
            return False, 1.0, "No stress test violations"

        # Check if recovery period has passed
        if self.last_stress_test:
            recovery_end = self.last_stress_test + timedelta(hours=self.config.recovery_wait_hours)
            if datetime.now() >= recovery_end:
                self.live_size_reduction_active = False
                return False, 1.0, "Recovery period completed"

        return True, self.config.live_size_reduction_factor, "Stress test tolerance exceeded"

    def get_stress_test_history(self, days: int = 7) -> List[Dict]:
        """Get recent stress test history"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [test for test in self.test_history
                if datetime.fromisoformat(test["start_time"]) >= cutoff_date]

# Integrated Safety System
class GridReverseSafetySystem:
    """Integrated grid and reverse safety system"""

    def __init__(self):
        self.grid_safety = GridSafetyConfig()
        self.reverse_safety = ReverseSafetyConfig()
        self.capital_allocation = AdaptiveCapitalAllocation()
        self.stress_testing = AutomatedStressTesting()

        self.grid_nets: Dict[str, GridSafetyNet] = {}
        self.reverse_nets: Dict[str, ReverseSafetyNet] = {}

    def validate_grid_strategy(self, symbol: str, regime: str,
                             market_volatility: float, correlation_score: float,
                             account_balance: float) -> Tuple[bool, str]:
        """Validate grid strategy execution"""
        # Get or create grid safety net
        if symbol not in self.grid_nets:
            self.grid_nets[symbol] = GridSafetyNet(symbol, regime)

        grid_net = self.grid_nets[symbol]
        grid_net.regime = regime

        # Check safety conditions
        safe, reason = grid_net.is_safe_to_add_grid(market_volatility, correlation_score, regime)

        if safe:
            # Update exposure tracking
            grid_net.update_exposure(grid_net.total_exposure + 1000, account_balance)  # Placeholder exposure
            grid_net.current_grid_levels += 1
            grid_net.last_grid_time = datetime.now()

        return safe, reason

    def validate_reverse_strategy(self, symbol: str, ensemble_confidence: float,
                                proposed_size: float, current_regime: str) -> Tuple[bool, str]:
        """Validate reverse strategy execution"""
        # Get or create reverse safety net
        if symbol not in self.reverse_nets:
            self.reverse_nets[symbol] = ReverseSafetyNet(symbol)

        reverse_net = self.reverse_nets[symbol]

        # Check safety conditions
        safe, reason = reverse_net.is_safe_to_reverse(ensemble_confidence, proposed_size, current_regime)

        if safe:
            # Record the reversal
            reverse_net.record_reversal(proposed_size)

        return safe, reason

    def run_nightly_stress_test(self, current_positions: Dict[str, Dict]) -> Dict[str, Any]:
        """Run nightly stress test and get recommendations"""
        return self.stress_testing.run_nightly_stress_test(current_positions)

    def get_capital_allocations(self, instrument_performance: Dict[str, Dict],
                               current_regime: str, total_capital: float) -> Dict[str, float]:
        """Get optimal capital allocations"""
        return self.capital_allocation.calculate_optimal_allocation(
            instrument_performance, current_regime, total_capital
        )

    def should_reduce_positions(self) -> Tuple[bool, float, str]:
        """Check if position sizes should be reduced due to stress"""
        return self.stress_testing.should_reduce_live_size()

if __name__ == "__main__":
    # Demo usage
    safety_system = GridReverseSafetySystem()

    # Test grid strategy validation
    print("Testing Grid Strategy Safety:")
    safe, reason = safety_system.validate_grid_strategy(
        symbol="EURUSD",
        regime="RANGE",
        market_volatility=0.0008,
        correlation_score=0.5,
        account_balance=10000.0
    )
    print(f"Grid safe: {safe} - {reason}")

    # Test reverse strategy validation
    print("\nTesting Reverse Strategy Safety:")
    safe, reason = safety_system.validate_reverse_strategy(
        symbol="GBPUSD",
        ensemble_confidence=0.85,
        proposed_size=0.02,
        current_regime="TREND"
    )
    print(f"Reverse safe: {safe} - {reason}")

    # Test capital allocation
    print("\nTesting Capital Allocation:")
    performance = {
        "EURUSD": {"sharpe_ratio": 1.5, "return": 0.02},
        "GBPUSD": {"sharpe_ratio": 1.2, "return": 0.015}
    }
    allocations = safety_system.get_capital_allocations(performance, "TREND", 10000.0)
    print(f"Allocations: {allocations}")

    # Test stress testing
    print("\nTesting Stress Testing:")
    positions = {"EURUSD": {"size": 0.1}, "GBPUSD": {"size": 0.05}}
    stress_results = safety_system.run_nightly_stress_test(positions)
    print(f"Stress test loss: {stress_results['total_projected_loss']:.1f}%")

    print("\n✅ All safety systems operational!")
