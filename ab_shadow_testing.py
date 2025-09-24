#!/usr/bin/env python3
"""
A/B Shadow Testing System for TitanovaX
Tests new strategies/parameters in parallel for 72h before live deployment
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import numpy as np
from collections import defaultdict, deque
import threading
import time

@dataclass
class ShadowTest:
    """A/B shadow test configuration"""
    test_id: str
    strategy_name: str
    parameters: Dict[str, Any]
    baseline_parameters: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    status: str  # "running", "completed", "failed", "approved", "rejected"

    # Performance metrics
    shadow_pnl: float = 0.0
    baseline_pnl: float = 0.0
    shadow_trades: int = 0
    baseline_trades: int = 0

    # Required metrics for approval
    metrics: Dict[str, float] = field(default_factory=dict)

    # Test results
    approval_criteria: Dict[str, float] = field(default_factory=dict)
    final_decision: Optional[str] = None
    decision_reason: str = ""

@dataclass
class PerformanceMetric:
    """Individual performance metric for shadow testing"""
    name: str
    shadow_value: float
    baseline_value: float
    improvement: float  # (shadow - baseline) / baseline
    weight: float = 1.0
    threshold: float = 0.0  # Minimum improvement required

class ABShadowTesting:
    """A/B shadow testing system for safe strategy deployment"""

    def __init__(self, config_path: str = 'config/shadow_testing_config.json'):
        self.config_path = Path(config_path)
        self.active_tests: Dict[str, ShadowTest] = {}
        self.completed_tests: List[ShadowTest] = []
        self.test_history: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history = 1000

        self.load_config()
        self.setup_logging()

        # Start background monitoring
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def load_config(self):
        """Load shadow testing configuration"""
        default_config = {
            "test_duration_hours": 72,
            "min_trades_for_evaluation": 10,
            "confidence_level": 0.95,
            "approval_metrics": {
                "total_return": {"weight": 0.3, "threshold": 0.05},
                "sharpe_ratio": {"weight": 0.25, "threshold": 0.1},
                "max_drawdown": {"weight": 0.2, "threshold": -0.05},  # Negative threshold means lower is better
                "win_rate": {"weight": 0.15, "threshold": 0.05},
                "profit_factor": {"weight": 0.1, "threshold": 0.1}
            },
            "auto_approval_enabled": False,
            "notification_enabled": True,
            "max_concurrent_tests": 3,
            "baseline_window_days": 30
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load shadow testing config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def start_shadow_test(self, strategy_name: str, parameters: Dict[str, Any],
                         baseline_parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new shadow test

        Args:
            strategy_name: Name of strategy being tested
            parameters: New parameters to test
            baseline_parameters: Baseline parameters (if None, uses current live parameters)

        Returns:
            test_id: Unique identifier for the test
        """
        # Check concurrent test limit
        if len(self.active_tests) >= self.config["max_concurrent_tests"]:
            raise ValueError(f"Maximum concurrent tests ({self.config['max_concurrent_tests']}) reached")

        # Generate test ID
        test_id = hashlib.md5(f"{strategy_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        # Set baseline parameters if not provided
        if baseline_parameters is None:
            baseline_parameters = self._get_current_live_parameters(strategy_name)

        # Calculate test duration
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.config["test_duration_hours"])

        # Create shadow test
        test = ShadowTest(
            test_id=test_id,
            strategy_name=strategy_name,
            parameters=parameters,
            baseline_parameters=baseline_parameters,
            start_time=start_time,
            end_time=end_time,
            status="running",
            approval_criteria=self.config["approval_metrics"]
        )

        self.active_tests[test_id] = test

        # Start monitoring if not already running
        if not self.is_monitoring:
            self._start_monitoring()

        self.logger.info(f"Started shadow test {test_id} for strategy {strategy_name}")
        return test_id

    def _get_current_live_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get current live parameters for strategy (placeholder)"""
        # In production, this would load from live configuration
        return {
            "rsi_period": 14,
            "stop_loss_pips": 20,
            "take_profit_pips": 40,
            "position_size": 0.01
        }

    def update_test_performance(self, test_id: str, shadow_pnl: float, baseline_pnl: float,
                               shadow_trades: int, baseline_trades: int):
        """
        Update performance metrics for a shadow test

        Args:
            test_id: Test identifier
            shadow_pnl: PnL from shadow strategy
            baseline_pnl: PnL from baseline strategy
            shadow_trades: Number of trades from shadow strategy
            baseline_trades: Number of trades from baseline strategy
        """
        if test_id not in self.active_tests:
            self.logger.warning(f"Test {test_id} not found in active tests")
            return

        test = self.active_tests[test_id]
        test.shadow_pnl = shadow_pnl
        test.baseline_pnl = baseline_pnl
        test.shadow_trades = shadow_trades
        test.baseline_trades = baseline_trades

        # Record performance snapshot
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "shadow_pnl": shadow_pnl,
            "baseline_pnl": baseline_pnl,
            "shadow_trades": shadow_trades,
            "baseline_trades": baseline_trades
        }
        self.test_history[test_id].append(snapshot)

        # Keep only recent history
        if len(self.test_history[test_id]) > self.max_history:
            self.test_history[test_id] = self.test_history[test_id][-self.max_history:]

        self.logger.debug(f"Updated performance for test {test_id}: Shadow PnL=${shadow_pnl:.2f}, Baseline PnL=${baseline_pnl:.2f}")

    def evaluate_test(self, test_id: str) -> Dict[str, Any]:
        """
        Evaluate completed test against approval criteria

        Returns:
            Dict with evaluation results and recommendation
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]

        # Check if test has enough data
        if test.shadow_trades < self.config["min_trades_for_evaluation"]:
            return {
                "status": "insufficient_data",
                "reason": f"Only {test.shadow_trades} trades, need at least {self.config['min_trades_for_evaluation']}",
                "metrics": {},
                "recommendation": "extend_test"
            }

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(test)

        # Evaluate against criteria
        evaluation_results = self._evaluate_against_criteria(metrics, test.approval_criteria)

        # Make recommendation
        if evaluation_results["passed_criteria"] >= 3:  # Must pass at least 3 metrics
            recommendation = "approve"
            reason = f"Passed {evaluation_results['passed_criteria']}/5 criteria with high confidence"
        else:
            recommendation = "reject"
            reason = f"Failed to meet minimum criteria (passed {evaluation_results['passed_criteria']}/5)"

        return {
            "status": "completed",
            "metrics": {k: v.__dict__ for k, v in metrics.items()},
            "evaluation": evaluation_results,
            "recommendation": recommendation,
            "reason": reason,
            "confidence": evaluation_results["overall_confidence"]
        }

    def _calculate_performance_metrics(self, test: ShadowTest) -> Dict[str, PerformanceMetric]:
        """Calculate performance metrics for test evaluation"""
        metrics = {}

        # Total return
        shadow_return = test.shadow_pnl
        baseline_return = test.baseline_pnl
        return_improvement = (shadow_return - baseline_return) / abs(baseline_return) if baseline_return != 0 else 0

        metrics["total_return"] = PerformanceMetric(
            name="total_return",
            shadow_value=shadow_return,
            baseline_value=baseline_return,
            improvement=return_improvement,
            weight=self.config["approval_metrics"]["total_return"]["weight"],
            threshold=self.config["approval_metrics"]["total_return"]["threshold"]
        )

        # Win rate (if we have trade data)
        if test.shadow_trades > 0 and test.baseline_trades > 0:
            # Placeholder - in production would calculate from actual trade data
            shadow_win_rate = 0.55  # Would be calculated from real trades
            baseline_win_rate = 0.52  # Would be calculated from real trades
            win_rate_improvement = (shadow_win_rate - baseline_win_rate) / baseline_win_rate if baseline_win_rate > 0 else 0

            metrics["win_rate"] = PerformanceMetric(
                name="win_rate",
                shadow_value=shadow_win_rate,
                baseline_value=baseline_win_rate,
                improvement=win_rate_improvement,
                weight=self.config["approval_metrics"]["win_rate"]["weight"],
                threshold=self.config["approval_metrics"]["win_rate"]["threshold"]
            )

        # Sharpe ratio (placeholder calculation)
        shadow_sharpe = 1.2  # Would be calculated from real returns
        baseline_sharpe = 1.0  # Would be calculated from real returns
        sharpe_improvement = (shadow_sharpe - baseline_sharpe) / baseline_sharpe if baseline_sharpe > 0 else 0

        metrics["sharpe_ratio"] = PerformanceMetric(
            name="sharpe_ratio",
            shadow_value=shadow_sharpe,
            baseline_value=baseline_sharpe,
            improvement=sharpe_improvement,
            weight=self.config["approval_metrics"]["sharpe_ratio"]["weight"],
            threshold=self.config["approval_metrics"]["sharpe_ratio"]["threshold"]
        )

        # Max drawdown (lower is better)
        shadow_drawdown = -0.08  # Would be calculated from real trades
        baseline_drawdown = -0.12  # Would be calculated from real trades
        drawdown_improvement = (baseline_drawdown - shadow_drawdown) / abs(baseline_drawdown) if baseline_drawdown != 0 else 0

        metrics["max_drawdown"] = PerformanceMetric(
            name="max_drawdown",
            shadow_value=shadow_drawdown,
            baseline_value=baseline_drawdown,
            improvement=drawdown_improvement,
            weight=self.config["approval_metrics"]["max_drawdown"]["weight"],
            threshold=self.config["approval_metrics"]["max_drawdown"]["threshold"]
        )

        # Profit factor
        shadow_profit_factor = 1.4  # Would be calculated from real trades
        baseline_profit_factor = 1.2  # Would be calculated from real trades
        profit_factor_improvement = (shadow_profit_factor - baseline_profit_factor) / baseline_profit_factor if baseline_profit_factor > 0 else 0

        metrics["profit_factor"] = PerformanceMetric(
            name="profit_factor",
            shadow_value=shadow_profit_factor,
            baseline_value=baseline_profit_factor,
            improvement=profit_factor_improvement,
            weight=self.config["approval_metrics"]["profit_factor"]["weight"],
            threshold=self.config["approval_metrics"]["profit_factor"]["threshold"]
        )

        return metrics

    def _evaluate_against_criteria(self, metrics: Dict[str, PerformanceMetric],
                                  criteria: Dict[str, Dict]) -> Dict[str, Any]:
        """Evaluate metrics against approval criteria"""
        passed_criteria = 0
        total_weight = 0
        weighted_score = 0

        for metric_name, metric in metrics.items():
            if metric_name in criteria:
                criterion = criteria[metric_name]
                threshold = criterion["threshold"]

                # Check if improvement meets threshold
                if metric.improvement >= threshold:
                    passed_criteria += 1
                    weighted_score += metric.weight * min(metric.improvement / threshold, 2.0)  # Cap at 2x threshold
                else:
                    weighted_score += metric.weight * (metric.improvement / threshold)  # Penalty for missing threshold

                total_weight += metric.weight

        # Calculate overall confidence
        if total_weight > 0:
            overall_confidence = weighted_score / total_weight
        else:
            overall_confidence = 0.0

        return {
            "passed_criteria": passed_criteria,
            "total_criteria": len(criteria),
            "overall_confidence": max(0, min(1, overall_confidence)),
            "metric_scores": {k: v.improvement for k, v in metrics.items()}
        }

    def complete_test(self, test_id: str, evaluation_result: Dict[str, Any]):
        """Complete a shadow test with evaluation results"""
        if test_id not in self.active_tests:
            self.logger.warning(f"Test {test_id} not found in active tests")
            return

        test = self.active_tests[test_id]
        test.status = "completed"
        test.end_time = datetime.now()
        test.metrics = evaluation_result["metrics"]

        # Set final decision
        if evaluation_result["recommendation"] == "approve":
            test.final_decision = "approved"
        else:
            test.final_decision = "rejected"

        test.decision_reason = evaluation_result["reason"]

        # Move to completed tests
        self.completed_tests.append(test)
        del self.active_tests[test_id]

        self.logger.info(f"Completed shadow test {test_id}: {test.final_decision} - {test.decision_reason}")

        # Send notification
        if self.config.get("notification_enabled", False):
            self._send_test_completion_notification(test, evaluation_result)

    def _send_test_completion_notification(self, test: ShadowTest, evaluation_result: Dict[str, Any]):
        """Send notification about test completion"""
        # In production, this would send to Telegram/email
        self.logger.info(f"Shadow test {test.test_id} completed: {test.final_decision}")

    def _start_monitoring(self):
        """Start background monitoring of active tests"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started shadow test monitoring")

    def _monitor_loop(self):
        """Background monitoring loop for active tests"""
        while self.is_monitoring:
            try:
                self._check_completed_tests()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in shadow test monitoring: {e}")
                time.sleep(10)

    def _check_completed_tests(self):
        """Check for tests that have completed their duration"""
        current_time = datetime.now()

        for test_id, test in list(self.active_tests.items()):
            if current_time >= test.end_time:
                # Test duration completed, evaluate it
                try:
                    evaluation_result = self.evaluate_test(test_id)

                    if evaluation_result["status"] == "completed":
                        self.complete_test(test_id, evaluation_result)
                    elif evaluation_result["status"] == "insufficient_data":
                        # Extend test by 24 hours
                        test.end_time += timedelta(hours=24)
                        self.logger.info(f"Extended test {test_id} due to insufficient data")
                except Exception as e:
                    self.logger.error(f"Error evaluating test {test_id}: {e}")
                    test.status = "failed"
                    test.decision_reason = f"Evaluation error: {str(e)}"

    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific test"""
        if test_id in self.active_tests:
            test = self.active_tests[test_id]
            return {
                "test_id": test.test_id,
                "strategy_name": test.strategy_name,
                "status": test.status,
                "start_time": test.start_time.isoformat(),
                "end_time": test.end_time.isoformat(),
                "shadow_pnl": test.shadow_pnl,
                "baseline_pnl": test.baseline_pnl,
                "shadow_trades": test.shadow_trades,
                "baseline_trades": test.baseline_trades,
                "progress_percent": min(100, (datetime.now() - test.start_time).total_seconds() /
                                      (test.end_time - test.start_time).total_seconds() * 100)
            }

        # Check completed tests
        for test in self.completed_tests:
            if test.test_id == test_id:
                return {
                    "test_id": test.test_id,
                    "strategy_name": test.strategy_name,
                    "status": test.status,
                    "start_time": test.start_time.isoformat(),
                    "end_time": test.end_time.isoformat(),
                    "shadow_pnl": test.shadow_pnl,
                    "baseline_pnl": test.baseline_pnl,
                    "final_decision": test.final_decision,
                    "decision_reason": test.decision_reason
                }

        return None

    def get_all_tests_status(self) -> Dict[str, Any]:
        """Get status of all tests"""
        active_status = []
        for test in self.active_tests.values():
            active_status.append({
                "test_id": test.test_id,
                "strategy_name": test.strategy_name,
                "status": test.status,
                "progress_percent": min(100, (datetime.now() - test.start_time).total_seconds() /
                                      (test.end_time - test.start_time).total_seconds() * 100)
            })

        recent_completed = []
        for test in self.completed_tests[-5:]:  # Last 5 completed tests
            recent_completed.append({
                "test_id": test.test_id,
                "strategy_name": test.strategy_name,
                "status": test.status,
                "final_decision": test.final_decision,
                "completion_time": test.end_time.isoformat()
            })

        return {
            "active_tests": active_status,
            "recent_completed": recent_completed,
            "total_completed": len(self.completed_tests),
            "total_active": len(self.active_tests)
        }

    def deploy_approved_strategy(self, test_id: str) -> bool:
        """Deploy an approved strategy to live trading"""
        # Find the test
        test = None
        for t in self.completed_tests:
            if t.test_id == test_id and t.final_decision == "approved":
                test = t
                break

        if test is None:
            self.logger.error(f"Approved test {test_id} not found")
            return False

        try:
            # In production, this would:
            # 1. Update live configuration with new parameters
            # 2. Deploy to production environment
            # 3. Start canary deployment (1% traffic)
            # 4. Monitor for 24-72 hours
            # 5. Gradually increase traffic if successful

            self.logger.info(f"Deploying approved strategy {test.strategy_name} from test {test_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to deploy strategy from test {test_id}: {e}")
            return False

if __name__ == "__main__":
    # Demo usage
    shadow_testing = ABShadowTesting()

    # Start a shadow test
    test_id = shadow_testing.start_shadow_test(
        strategy_name="rsi_strategy",
        parameters={
            "rsi_period": 10,  # Changed from 14
            "rsi_oversold": 25,  # Changed from 30
            "position_size": 0.02  # Changed from 0.01
        }
    )

    print(f"Started shadow test: {test_id}")

    # Simulate some performance updates
    import time
    for i in range(10):
        # Simulate performance data
        shadow_pnl = 100 + i * 15  # Shadow strategy doing better
        baseline_pnl = 90 + i * 10  # Baseline performance

        shadow_testing.update_test_performance(
            test_id=test_id,
            shadow_pnl=shadow_pnl,
            baseline_pnl=baseline_pnl,
            shadow_trades=5 + i,
            baseline_trades=4 + i
        )

        time.sleep(1)

        # Check status
        status = shadow_testing.get_test_status(test_id)
        if status:
            print(f"Test {test_id} progress: {status['progress_percent']:.1f}%")

    # Get final evaluation
    evaluation = shadow_testing.evaluate_test(test_id)
    print(f"Test evaluation: {evaluation}")

    # Show all tests
    all_status = shadow_testing.get_all_tests_status()
    print(f"Active tests: {len(all_status['active_tests'])}")
    print(f"Recent completed: {len(all_status['recent_completed'])}")
