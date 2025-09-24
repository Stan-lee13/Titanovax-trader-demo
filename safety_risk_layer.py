#!/usr/bin/env python3
"""
Safety & Risk Management Layer for TitanovaX
Non-bypassable risk checks and safety mechanisms
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import threading
import time

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_loss_per_trade_currency: float
    daily_drawdown_cap_percent: float
    max_open_trades: int
    max_correlated_exposure_percent: float
    max_total_exposure_percent: float
    min_account_balance_protection_percent: float

@dataclass
class TradeRequest:
    """Trade request for validation"""
    symbol: str
    side: str  # "BUY" or "SELL"
    volume: float
    current_price: float
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    order_type: str = "market"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    """Current position information"""
    ticket: int
    symbol: str
    side: str
    volume: float
    open_price: float
    current_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    profit_loss: float
    open_time: datetime
    magic_number: int

@dataclass
class DailyPnL:
    """Daily profit/loss tracking"""
    date: str  # YYYY-MM-DD format
    starting_balance: float
    current_balance: float
    realized_pnl: float
    unrealized_pnl: float
    trades_count: int
    winning_trades: int
    losing_trades: int

class RiskEngine:
    """Core risk management engine with non-bypassable checks"""

    def __init__(self, config_path: str = 'config/risk_config.json'):
        self.config_path = Path(config_path)
        self.limits: RiskLimits = None
        self.positions: Dict[int, Position] = {}
        self.daily_pnl: Dict[str, DailyPnL] = {}
        self.account_balance: float = 0.0
        self.emergency_stop: bool = False
        self.admin_override: bool = False
        self.safety_lock_file = Path('data/state/disabled.lock')

        self.load_config()
        self.setup_logging()
        self.setup_emergency_monitor()

    def load_config(self):
        """Load risk configuration"""
        default_config = {
            "max_loss_per_trade_currency": 100.0,  # Max loss per trade in account currency
            "daily_drawdown_cap_percent": 5.0,     # Max daily drawdown percentage
            "max_open_trades": 5,                  # Maximum open trades
            "max_correlated_exposure_percent": 20.0, # Max exposure to correlated pairs
            "max_total_exposure_percent": 50.0,    # Max total exposure
            "min_account_balance_protection_percent": 20.0, # Keep this much in reserve
            "correlation_matrix": {
                "EURUSD": {"GBPUSD": 0.8, "USDCHF": 0.6, "AUDUSD": 0.7},
                "GBPUSD": {"EURUSD": 0.8, "USDCHF": 0.5, "AUDUSD": 0.6},
                "USDCHF": {"EURUSD": 0.6, "GBPUSD": 0.5, "AUDUSD": 0.4},
                "AUDUSD": {"EURUSD": 0.7, "GBPUSD": 0.6, "USDCHF": 0.4}
            },
            "emergency_contacts": ["admin@trading.com"],
            "auto_recovery_enabled": True,
            "circuit_breaker_threshold_percent": 10.0
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    self.config = {**default_config, **config_data}
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load risk config: {e}")
            self.config = default_config

        # Create risk limits object
        self.limits = RiskLimits(**{k: v for k, v in self.config.items()
                                   if k in RiskLimits.__dataclass_fields__})

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def setup_emergency_monitor(self):
        """Setup emergency monitoring thread"""
        self.monitor_thread = threading.Thread(target=self._emergency_monitor, daemon=True)
        self.monitor_thread.start()

    def _emergency_monitor(self):
        """Monitor for emergency conditions"""
        while True:
            try:
                self._check_emergency_conditions()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Emergency monitor error: {e}")

    def _check_emergency_conditions(self):
        """Check for emergency conditions that require immediate action"""

        # Check for safety lock file
        if self.safety_lock_file.exists():
            if not self.emergency_stop:
                self.logger.critical("Safety lock file detected - emergency stop activated")
                self.emergency_stop = True

        # Check for abnormal market conditions (placeholder)
        # In real implementation, this would check for extreme volatility, news events, etc.

        # Check for system health issues
        # In real implementation, this would check for connectivity, model health, etc.

    def validate_trade_request(self, request: TradeRequest) -> Tuple[bool, str]:
        """
        Validate trade request against all risk limits

        Returns:
            (approved: bool, reason: str)
        """

        # Emergency stop check (highest priority)
        if self.emergency_stop:
            return False, "Emergency stop activated"

        # Admin override check
        if not self.admin_override:
            # Check safety lock file
            if self.safety_lock_file.exists():
                return False, "Trading disabled by safety lock file"

        # Calculate potential loss for this trade
        potential_loss = self._calculate_potential_loss(request)
        if potential_loss > self.limits.max_loss_per_trade_currency:
            return False, f'Trade exceeds max loss per trade: {potential_loss:.2f} > {self.limits.max_loss_per_trade_currency:.2f}'

        # Check daily drawdown
        today = datetime.now().strftime('%Y-%m-%d')
        if today in self.daily_pnl:
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > self.limits.daily_drawdown_cap_percent:
                return False, f'Daily drawdown cap exceeded: {current_drawdown:.2f}% > {self.limits.daily_drawdown_cap_percent}%'

        # Check maximum open trades
        if len(self.positions) >= self.limits.max_open_trades:
            return False, f"Maximum open trades exceeded: {len(self.positions)} >= {self.limits.max_open_trades}"

        # Check correlated exposure
        correlated_exposure = self._calculate_correlated_exposure(request.symbol, request.volume)
        if correlated_exposure > self.limits.max_correlated_exposure_percent:
            return False, f'Correlated exposure too high: {correlated_exposure:.2f}% > {self.limits.max_correlated_exposure_percent}%'

        # Check total exposure
        total_exposure = self._calculate_total_exposure(request.volume)
        if total_exposure > self.limits.max_total_exposure_percent:
            return False, f'Total exposure too high: {total_exposure:.2f}% > {self.limits.max_total_exposure_percent}%'

        # Check account balance protection
        required_reserve = self.account_balance * (self.limits.min_account_balance_protection_percent / 100)
        if potential_loss > (self.account_balance - required_reserve):
            return False, f'Trade would violate account balance protection: required reserve {required_reserve:.2f}'

        # All checks passed
        return True, "All risk checks passed"

    def _calculate_potential_loss(self, request: TradeRequest) -> float:
        """Calculate potential loss for a trade request"""

        # For forex, potential loss is based on stop loss or max allowed loss
        if request.stop_loss_pips and request.stop_loss_pips > 0:
            # Convert pips to currency amount
            pip_value = request.volume * 10  # Standard lot pip value is $10
            potential_loss = request.stop_loss_pips * pip_value
        else:
            # Use maximum allowed loss if no stop loss specified
            potential_loss = self.limits.max_loss_per_trade_currency * 0.5  # Conservative estimate

        return abs(potential_loss)

    def _calculate_current_drawdown(self) -> float:
        """Calculate current daily drawdown percentage"""

        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.daily_pnl:
            return 0.0

        daily = self.daily_pnl[today]
        if daily.starting_balance == 0:
            return 0.0

        current_value = daily.starting_balance + daily.realized_pnl + daily.unrealized_pnl
        drawdown = (daily.starting_balance - current_value) / daily.starting_balance * 100

        return max(0, drawdown)

    def _calculate_correlated_exposure(self, symbol: str, volume: float) -> float:
        """Calculate correlated exposure percentage"""

        if symbol not in self.config["correlation_matrix"]:
            return 0.0

        correlated_exposure = 0.0
        correlations = self.config["correlation_matrix"][symbol]

        for position in self.positions.values():
            if position.symbol in correlations:
                correlation = correlations[position.symbol]
                correlated_exposure += position.volume * correlation

        # Add current trade's contribution
        for other_symbol, correlation in correlations.items():
            if other_symbol != symbol:
                correlated_exposure += volume * correlation

        # Convert to percentage of account balance
        if self.account_balance > 0:
            return (correlated_exposure / self.account_balance) * 100
        return 0.0

    def _calculate_total_exposure(self, additional_volume: float = 0.0) -> float:
        """Calculate total exposure percentage"""

        total_volume = sum(pos.volume for pos in self.positions.values()) + additional_volume

        # For forex, exposure is roughly volume * 100000 (for standard lots)
        # But we'll use a simpler calculation for demo
        exposure_value = total_volume * 10000  # Conservative estimate

        if self.account_balance > 0:
            return (exposure_value / self.account_balance) * 100
        return 0.0

    def update_positions(self, positions: List[Position]):
        """Update current positions"""

        self.positions = {pos.ticket: pos for pos in positions}

        # Recalculate unrealized PnL for daily tracking
        self._update_daily_pnl()

    def update_account_balance(self, balance: float):
        """Update account balance"""

        if self.account_balance == 0:  # First update
            today = datetime.now().strftime('%Y-%m-%d')
            self.daily_pnl[today] = DailyPnL(
                date=today,
                starting_balance=balance,
                current_balance=balance,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                trades_count=0,
                winning_trades=0,
                losing_trades=0
            )

        self.account_balance = balance

    def _update_daily_pnl(self):
        """Update daily PnL calculations"""

        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.daily_pnl:
            return

        daily = self.daily_pnl[today]
        daily.unrealized_pnl = sum(pos.profit_loss for pos in self.positions.values())
        daily.current_balance = daily.starting_balance + daily.realized_pnl + daily.unrealized_pnl

    def record_trade_result(self, ticket: int, profit_loss: float, winning: bool):
        """Record trade result for PnL tracking"""

        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.daily_pnl:
            self.daily_pnl[today] = DailyPnL(
                date=today,
                starting_balance=self.account_balance,
                current_balance=self.account_balance,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                trades_count=0,
                winning_trades=0,
                losing_trades=0
            )

        daily = self.daily_pnl[today]
        daily.realized_pnl += profit_loss
        daily.trades_count += 1

        if winning:
            daily.winning_trades += 1
        else:
            daily.losing_trades += 1

        daily.current_balance = daily.starting_balance + daily.realized_pnl + daily.unrealized_pnl

        # Check if daily drawdown cap exceeded
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.limits.daily_drawdown_cap_percent:
            self.logger.critical(f'Daily drawdown cap exceeded: {current_drawdown:.2f}%')
            self._activate_emergency_stop("Daily drawdown cap exceeded")

    def _activate_emergency_stop(self, reason: str):
        """Activate emergency stop"""

        self.emergency_stop = True

        # Create safety lock file
        self.safety_lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.safety_lock_file, 'w') as f:
            f.write(f"Emergency stop activated: {reason}\\n")
            f.write(f"Time: {datetime.now().isoformat()}\\n")

        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

        # In a real implementation, you might also:
        # - Close all positions
        # - Send emergency notifications
        # - Trigger circuit breaker

    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status"""

        today = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = self.daily_pnl.get(today)

        return {
            "emergency_stop": self.emergency_stop,
            "admin_override": self.admin_override,
            "account_balance": self.account_balance,
            "current_positions": len(self.positions),
            "total_exposure_percent": self._calculate_total_exposure(),
            "correlated_exposure_percent": max(self._calculate_correlated_exposure(symbol, 0) for symbol in self.config["correlation_matrix"].keys()),
            "daily_drawdown_percent": self._calculate_current_drawdown(),
            "daily_pnl": daily_pnl.__dict__ if daily_pnl else None,
            "risk_limits": {
                "max_loss_per_trade": self.limits.max_loss_per_trade_currency,
                "daily_drawdown_cap": self.limits.daily_drawdown_cap_percent,
                "max_open_trades": self.limits.max_open_trades,
                "max_correlated_exposure": self.limits.max_correlated_exposure_percent,
                "max_total_exposure": self.limits.max_total_exposure_percent
            },
            "safety_lock_file": str(self.safety_lock_file) if self.safety_lock_file.exists() else None
        }

    def activate_admin_override(self, admin_key: str) -> bool:
        """Activate admin override (requires proper authentication)"""

        # In a real implementation, this would verify against a secure key store
        expected_key_hash = hashlib.sha256("admin_secret_key".encode()).hexdigest()

        if hashlib.sha256(admin_key.encode()).hexdigest() == expected_key_hash:
            self.admin_override = True
            self.logger.warning("Admin override activated")
            return True

        return False

    def deactivate_admin_override(self):
        """Deactivate admin override"""

        self.admin_override = False
        self.logger.info("Admin override deactivated")

    def reset_emergency_stop(self, admin_key: str) -> bool:
        """Reset emergency stop (requires admin authentication)"""

        if self.activate_admin_override(admin_key):
            self.emergency_stop = False

            # Remove safety lock file
            if self.safety_lock_file.exists():
                self.safety_lock_file.unlink()

            self.logger.info("Emergency stop reset")
            return True

        return False

class AnomalyDetector:
    """Anomaly detection for trading system"""

    def __init__(self, config_path: str = 'config/anomaly_config.json'):
        self.config_path = Path(config_path)
        self.anomaly_history: List[Dict] = []
        self.max_history = 1000

        self.load_config()

    def load_config(self):
        """Load anomaly detection configuration"""
        default_config = {
            "slippage_threshold_multiplier": 3.0,
            "latency_threshold_ms": 500,
            "execution_failure_rate_threshold": 0.1,
            "spread_spike_threshold_multiplier": 2.0,
            "volume_anomaly_threshold": 5.0,
            "price_gap_threshold_pips": 50,
            "news_impact_threshold": 0.5
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load anomaly config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def detect_anomalies(self, market_data: Dict, execution_metrics: Dict) -> List[Dict]:
        """Detect anomalies in market data and execution"""

        anomalies = []

        # Check for slippage anomalies
        slippage_anomaly = self._check_slippage_anomaly(execution_metrics)
        if slippage_anomaly:
            anomalies.append(slippage_anomaly)

        # Check for latency anomalies
        latency_anomaly = self._check_latency_anomaly(execution_metrics)
        if latency_anomaly:
            anomalies.append(latency_anomaly)

        # Check for spread anomalies
        spread_anomaly = self._check_spread_anomaly(market_data)
        if spread_anomaly:
            anomalies.append(spread_anomaly)

        # Check for price gap anomalies
        price_gap_anomaly = self._check_price_gap_anomaly(market_data)
        if price_gap_anomaly:
            anomalies.append(price_gap_anomaly)

        # Store anomalies
        for anomaly in anomalies:
            anomaly["timestamp"] = datetime.now().isoformat()
            self.anomaly_history.append(anomaly)

        if len(self.anomaly_history) > self.max_history:
            self.anomaly_history = self.anomaly_history[-self.max_history:]

        return anomalies

    def _check_slippage_anomaly(self, execution_metrics: Dict) -> Optional[Dict]:
        """Check for slippage anomalies"""

        recent_slippages = execution_metrics.get('recent_slippages', [])
        if len(recent_slippages) < 5:
            return None

        avg_slippage = np.mean(recent_slippages)
        std_slippage = np.std(recent_slippages)

        if std_slippage == 0:
            return None

        # Check if current slippage is anomalous
        current_slippage = recent_slippages[-1] if recent_slippages else 0
        threshold = avg_slippage + self.config["slippage_threshold_multiplier"] * std_slippage

        if current_slippage > threshold:
            return {
                "type": "slippage_anomaly",
                "severity": "high" if current_slippage > threshold * 1.5 else "medium",
                "description": f'Unusual slippage detected: {current_slippage:.2f} pips',
                "value": current_slippage,
                "threshold": threshold,
                "recommendation": "Consider switching to limit orders or reducing position sizes"
            }

        return None

    def _check_latency_anomaly(self, execution_metrics: Dict) -> Optional[Dict]:
        """Check for latency anomalies"""

        recent_latencies = execution_metrics.get('recent_latencies', [])
        if not recent_latencies:
            return None

        avg_latency = np.mean(recent_latencies)
        threshold = self.config["latency_threshold_ms"]

        if avg_latency > threshold:
            return {
                "type": "latency_anomaly",
                "severity": "high" if avg_latency > threshold * 2 else "medium",
                "description": f'High execution latency detected: {avg_latency:.0f}ms',
                "value": avg_latency,
                "threshold": threshold,
                "recommendation": "Check connection and consider using closer servers"
            }

        return None

    def _check_spread_anomaly(self, market_data: Dict) -> Optional[Dict]:
        """Check for spread anomalies"""

        spread_history = market_data.get('spread_history', [])
        if len(spread_history) < 10:
            return None

        avg_spread = np.mean(spread_history)
        std_spread = np.std(spread_history)

        if std_spread == 0:
            return None

        current_spread = spread_history[-1] if spread_history else 0
        threshold = avg_spread + self.config["spread_spike_threshold_multiplier"] * std_spread

        if current_spread > threshold:
            return {
                "type": "spread_anomaly",
                "severity": "high" if current_spread > threshold * 2 else "medium",
                "description": f'Unusual spread detected: {current_spread:.2f} pips',
                "value": current_spread,
                "threshold": threshold,
                "recommendation": "Avoid scalping strategies during wide spreads"
            }

        return None

    def _check_price_gap_anomaly(self, market_data: Dict) -> Optional[Dict]:
        """Check for price gap anomalies"""

        price_history = market_data.get('price_history', [])
        if len(price_history) < 2:
            return None

        # Calculate price gaps
        gaps = []
        for i in range(1, len(price_history)):
            gap = abs(price_history[i] - price_history[i-1]) * 10000  # Convert to pips
            gaps.append(gap)

        if gaps:
            max_gap = max(gaps)
            threshold = self.config["price_gap_threshold_pips"]

            if max_gap > threshold:
                return {
                    "type": "price_gap_anomaly",
                    "severity": "critical",
                    "description": f'Large price gap detected: {max_gap:.0f} pips',
                    "value": max_gap,
                    "threshold": threshold,
                    "recommendation": "Exercise extreme caution - consider emergency stop"
                }

        return None

if __name__ == "__main__":
    # Demo usage
    risk_engine = RiskEngine()
    anomaly_detector = AnomalyDetector()

    # Update account balance
    risk_engine.update_account_balance(10000.0)

    # Create a trade request
    trade_request = TradeRequest(
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        current_price=1.1234,
        stop_loss_pips=20,
        take_profit_pips=40
    )

    # Validate trade
    approved, reason = risk_engine.validate_trade_request(trade_request)
    print(f"Trade approved: {approved}")
    print(f"Reason: {reason}")

    # Get risk status
    status = risk_engine.get_risk_status()
    print(f"\\nRisk Status:")
    print(f"Emergency stop: {status['emergency_stop']}")
    print(f"Account balance: ${status['account_balance']}")
    print(f"Current positions: {status['current_positions']}")
    print(f'Daily drawdown: {status["daily_drawdown_percent"]:.2f}%')

    # Test anomaly detection
    market_data = {
        'spread_history': [1.5, 1.6, 1.4, 1.7, 3.2, 4.1],  # Spike at the end
        'price_history': [1.1234, 1.1235, 1.1233, 1.1236, 1.1234, 1.1260]  # Big gap
    }

    execution_metrics = {
        'recent_slippages': [0.5, 0.8, 1.2, 2.1, 3.5],  # High slippage
        'recent_latencies': [25, 30, 45, 200, 350]  # High latency
    }

    anomalies = anomaly_detector.detect_anomalies(market_data, execution_metrics)
    print(f"\\nAnomalies detected: {len(anomalies)}")
    for anomaly in anomalies:
        print(f"- {anomaly['type']}: {anomaly['description']}")
