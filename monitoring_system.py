#!/usr/bin/env python3
"""
TitanovaX Monitoring & Alerting System
Comprehensive monitoring with anomaly detection and self-healing
"""

import psutil
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import requests
import numpy as np
from collections import defaultdict, deque

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    thread_count: int
    open_files: int

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    active_positions: int

@dataclass
class ModelMetrics:
    """ML model performance metrics"""
    timestamp: datetime
    model_name: str
    prediction_accuracy: float
    inference_latency_ms: float
    memory_usage_mb: float
    prediction_count: int
    error_count: int
    confidence_score: float

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # 'cpu_percent > 90', 'memory_percent > 85', etc.
    threshold: float
    duration_minutes: int  # How long condition must be true
    severity: str  # 'info', 'warning', 'critical', 'error'
    cooldown_minutes: int  # Cooldown period between alerts
    enabled: bool = True

class MetricsCollector:
    """Collect system and application metrics"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Metrics storage
        self.system_metrics = deque(maxlen=1000)
        self.trading_metrics = deque(maxlen=1000)
        self.model_metrics = defaultdict(lambda: deque(maxlen=100))

        # Historical data for anomaly detection
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))

        # Collection intervals
        self.system_interval = 30  # seconds
        self.trading_interval = 60  # seconds

        # Start collection threads
        self._start_collection_threads()

    def _start_collection_threads(self):
        """Start background metric collection"""
        # System metrics thread
        system_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        system_thread.start()

        # Trading metrics thread
        trading_thread = threading.Thread(
            target=self._collect_trading_metrics,
            daemon=True
        )
        trading_thread.start()

    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        while True:
            try:
                metrics = self._get_system_metrics()
                self.system_metrics.append(metrics)
                self.metrics_history['system'].append(metrics)

                time.sleep(self.system_interval)

            except Exception as e:
                self.logger.error(f"System metrics collection failed: {e}")
                time.sleep(self.system_interval)

    def _collect_trading_metrics(self):
        """Collect trading performance metrics"""
        while True:
            try:
                metrics = self._get_trading_metrics()
                self.trading_metrics.append(metrics)
                self.metrics_history['trading'].append(metrics)

                time.sleep(self.trading_interval)

            except Exception as e:
                self.logger.error(f"Trading metrics collection failed: {e}")
                time.sleep(self.trading_interval)

    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }

            # Process information
            process = psutil.Process()
            process_count = len(psutil.pids())
            thread_count = process.num_threads()
            open_files = len(process.open_files())

            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io=network_io,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files
            )

        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                process_count=0,
                thread_count=0,
                open_files=0
            )

    def _get_trading_metrics(self) -> TradingMetrics:
        """Get current trading metrics"""
        try:
            # This would normally connect to your trading database
            # For now, using placeholder values
            return TradingMetrics(
                timestamp=datetime.now(),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                daily_pnl=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                active_positions=0
            )

        except Exception as e:
            self.logger.error(f"Failed to get trading metrics: {e}")
            return TradingMetrics(
                timestamp=datetime.now(),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                daily_pnl=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                active_positions=0
            )

    def add_model_metrics(self, model_name: str, metrics: ModelMetrics):
        """Add model performance metrics"""
        self.model_metrics[model_name].append(metrics)
        self.metrics_history[f'model_{model_name}'].append(metrics)

    def get_system_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get system metrics summary for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        recent_metrics = [
            m for m in self.system_metrics
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {}

        # Calculate averages
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_disk = np.mean([m.disk_percent for m in recent_metrics])

        # Calculate peaks
        max_cpu = max([m.cpu_percent for m in recent_metrics])
        max_memory = max([m.memory_percent for m in recent_metrics])
        max_disk = max([m.disk_percent for m in recent_metrics])

        return {
            'time_range_minutes': minutes,
            'sample_count': len(recent_metrics),
            'averages': {
                'cpu_percent': round(avg_cpu, 2),
                'memory_percent': round(avg_memory, 2),
                'disk_percent': round(avg_disk, 2)
            },
            'peaks': {
                'cpu_percent': round(max_cpu, 2),
                'memory_percent': round(max_memory, 2),
                'disk_percent': round(max_disk, 2)
            },
            'current': {
                'cpu_percent': recent_metrics[-1].cpu_percent,
                'memory_percent': recent_metrics[-1].memory_percent,
                'disk_percent': recent_metrics[-1].disk_percent
            }
        }

class AnomalyDetector:
    """Detect anomalies in system metrics using statistical methods"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Anomaly detection parameters
        self.cpu_threshold_sigma = 3.0
        self.memory_threshold_sigma = 3.0
        self.disk_threshold_sigma = 2.5
        self.min_samples_for_detection = 30

        # Historical data for baseline calculation
        self.baselines = {
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'disk_percent': deque(maxlen=1000)
        }

    def update_baseline(self, metrics: SystemMetrics):
        """Update baseline data with new metrics"""
        self.baselines['cpu_percent'].append(metrics.cpu_percent)
        self.baselines['memory_percent'].append(metrics.memory_percent)
        self.baselines['disk_percent'].append(metrics.disk_percent)

    def detect_anomalies(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Detect anomalies in system metrics"""
        anomalies = []

        # Need sufficient historical data
        if len(self.baselines['cpu_percent']) < self.min_samples_for_detection:
            return anomalies

        # Check CPU anomaly
        cpu_anomaly = self._check_metric_anomaly(
            metrics.cpu_percent,
            self.baselines['cpu_percent'],
            'cpu_percent',
            self.cpu_threshold_sigma
        )
        if cpu_anomaly:
            anomalies.append(cpu_anomaly)

        # Check memory anomaly
        memory_anomaly = self._check_metric_anomaly(
            metrics.memory_percent,
            self.baselines['memory_percent'],
            'memory_percent',
            self.memory_threshold_sigma
        )
        if memory_anomaly:
            anomalies.append(memory_anomaly)

        # Check disk anomaly
        disk_anomaly = self._check_metric_anomaly(
            metrics.disk_percent,
            self.baselines['disk_percent'],
            'disk_percent',
            self.disk_threshold_sigma
        )
        if disk_anomaly:
            anomalies.append(disk_anomaly)

        return anomalies

    def _check_metric_anomaly(self, current_value: float, historical_values: deque,
                            metric_name: str, threshold_sigma: float) -> Optional[Dict[str, Any]]:
        """Check if a metric value is anomalous"""

        values_array = np.array(list(historical_values))
        mean = np.mean(values_array)
        std = np.std(values_array)

        if std == 0:
            return None

        z_score = abs(current_value - mean) / std

        if z_score > threshold_sigma:
            return {
                'metric': metric_name,
                'current_value': current_value,
                'mean': mean,
                'std': std,
                'z_score': z_score,
                'threshold_sigma': threshold_sigma,
                'severity': 'critical' if z_score > threshold_sigma * 1.5 else 'warning'
            }

        return None

class AlertManager:
    """Manage alerts and notifications"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Alert rules
        self.alert_rules = self._load_default_alert_rules()

        # Alert history to prevent spam
        self.alert_history = defaultdict(list)  # rule_name -> list of timestamps

        # Cooldown periods
        self.cooldown_periods = {
            'info': timedelta(minutes=5),
            'warning': timedelta(minutes=15),
            'error': timedelta(minutes=30),
            'critical': timedelta(minutes=60)
        }

    def _load_default_alert_rules(self) -> List[AlertRule]:
        """Load default alert rules"""
        return [
            AlertRule(
                name='high_cpu_usage',
                condition='cpu_percent > 90',
                threshold=90.0,
                duration_minutes=5,
                severity='critical',
                cooldown_minutes=15
            ),
            AlertRule(
                name='high_memory_usage',
                condition='memory_percent > 85',
                threshold=85.0,
                duration_minutes=5,
                severity='warning',
                cooldown_minutes=10
            ),
            AlertRule(
                name='high_disk_usage',
                condition='disk_percent > 90',
                threshold=90.0,
                duration_minutes=10,
                severity='warning',
                cooldown_minutes=30
            ),
            AlertRule(
                name='low_disk_space',
                condition='disk_percent > 95',
                threshold=95.0,
                duration_minutes=1,
                severity='critical',
                cooldown_minutes=60
            ),
            AlertRule(
                name='high_daily_loss',
                condition='daily_pnl < -5.0',
                threshold=-5.0,
                duration_minutes=1,
                severity='error',
                cooldown_minutes=30
            ),
            AlertRule(
                name='trading_drawdown',
                condition='current_drawdown > 10.0',
                threshold=10.0,
                duration_minutes=1,
                severity='warning',
                cooldown_minutes=15
            )
        ]

    def check_alerts(self, metrics_collector: MetricsCollector) -> List[Dict[str, Any]]:
        """Check all alert rules and return triggered alerts"""
        triggered_alerts = []

        # Get latest system metrics
        if not metrics_collector.system_metrics:
            return triggered_alerts

        latest_metrics = metrics_collector.system_metrics[-1]

        # Get latest trading metrics
        latest_trading = metrics_collector.trading_metrics[-1] if metrics_collector.trading_metrics else None

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Check cooldown
            if not self._is_cooldown_expired(rule.name, rule.severity):
                continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, latest_metrics, latest_trading):
                alert = {
                    'rule_name': rule.name,
                    'severity': rule.severity,
                    'message': self._generate_alert_message(rule, latest_metrics, latest_trading),
                    'timestamp': datetime.now(),
                    'metrics': {
                        'cpu_percent': latest_metrics.cpu_percent,
                        'memory_percent': latest_metrics.memory_percent,
                        'disk_percent': latest_metrics.disk_percent
                    }
                }

                triggered_alerts.append(alert)
                self.alert_history[rule.name].append(datetime.now())

                # Send alert
                self._send_alert(alert)

        return triggered_alerts

    def _evaluate_condition(self, condition: str, system_metrics: SystemMetrics,
                          trading_metrics: Optional[TradingMetrics]) -> bool:
        """Evaluate alert condition"""
        try:
            # Simple condition evaluation
            # In production, use a more sophisticated expression evaluator

            if condition == 'cpu_percent > 90':
                return system_metrics.cpu_percent > 90
            elif condition == 'memory_percent > 85':
                return system_metrics.memory_percent > 85
            elif condition == 'disk_percent > 90':
                return system_metrics.disk_percent > 90
            elif condition == 'disk_percent > 95':
                return system_metrics.disk_percent > 95
            elif condition == 'daily_pnl < -5.0':
                return trading_metrics and trading_metrics.daily_pnl < -5.0
            elif condition == 'current_drawdown > 10.0':
                return trading_metrics and trading_metrics.current_drawdown > 10.0
            else:
                return False

        except Exception as e:
            self.logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def _generate_alert_message(self, rule: AlertRule, system_metrics: SystemMetrics,
                              trading_metrics: Optional[TradingMetrics]) -> str:
        """Generate alert message"""
        messages = {
            'high_cpu_usage': f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
            'high_memory_usage': f"High memory usage: {system_metrics.memory_percent:.1f}%",
            'high_disk_usage': f"High disk usage: {system_metrics.disk_percent:.1f}%",
            'low_disk_space': f"Critical disk space: {system_metrics.disk_percent:.1f}%",
            'high_daily_loss': f"High daily loss: {trading_metrics.daily_pnl:.2f}%" if trading_metrics else "Trading data unavailable",
            'trading_drawdown': f"Trading drawdown: {trading_metrics.current_drawdown:.2f}%" if trading_metrics else "Trading data unavailable"
        }

        return messages.get(rule.name, f"Alert triggered: {rule.name}")

    def _is_cooldown_expired(self, rule_name: str, severity: str) -> bool:
        """Check if cooldown period has expired"""
        if rule_name not in self.alert_history:
            return True

        cooldown_period = self.cooldown_periods.get(severity, timedelta(minutes=15))
        last_alert = self.alert_history[rule_name][-1]

        return datetime.now() - last_alert >= cooldown_period

    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert via configured channels"""
        try:
            # Telegram alert
            if self.config_manager.get_system_config().enable_telegram_alerts:
                self._send_telegram_alert(alert)

            # Email alert
            if self.config_manager.get_system_config().enable_email_alerts:
                self._send_email_alert(alert)

            # Log alert
            self.logger.warning(f"ALERT: {alert['message']} (Severity: {alert['severity']})")

        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

    def _send_telegram_alert(self, alert: Dict[str, Any]):
        """Send alert via Telegram"""
        try:
            telegram_config = self.config_manager.telegram

            if not telegram_config.enabled:
                return

            message = f"""
🚨 **TitanovaX Alert**

**Severity:** {alert['severity'].upper()}
**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
**Message:** {alert['message']}

**System Metrics:**
• CPU: {alert['metrics']['cpu_percent']:.1f}%
• Memory: {alert['metrics']['memory_percent']:.1f}%
• Disk: {alert['metrics']['disk_percent']:.1f}%
"""

            # Send to Telegram bot
            url = f"https://api.telegram.org/bot{telegram_config.bot_token}/sendMessage"
            payload = {
                'chat_id': telegram_config.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code != 200:
                self.logger.error(f"Failed to send Telegram alert: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Telegram alert failed: {e}")

    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send alert via email (placeholder)"""
        # Implement email sending logic here
        self.logger.info(f"Email alert would be sent: {alert['message']}")

class SelfHealingManager:
    """Manage self-healing and auto-recovery"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Service status tracking
        self.service_status = {
            'trading_engine': True,
            'ml_models': True,
            'database': True,
            'redis': True,
            'telegram': True
        }

        # Restart counts to prevent infinite loops
        self.restart_counts = defaultdict(int)
        self.max_restarts = 3

    def check_and_heal(self):
        """Check services and attempt healing"""
        if not self.config_manager.get_system_config().auto_restart_failed_services:
            return

        # Check database connection
        if not self._check_database_connection():
            self._heal_database_connection()

        # Check Redis connection
        if not self._check_redis_connection():
            self._heal_redis_connection()

        # Check trading engine
        if not self._check_trading_engine():
            self._heal_trading_engine()

    def _check_database_connection(self) -> bool:
        """Check database connectivity"""
        try:
            # This would implement actual database connection check
            return True
        except Exception:
            return False

    def _check_redis_connection(self) -> bool:
        """Check Redis connectivity"""
        try:
            # This would implement actual Redis connection check
            return True
        except Exception:
            return False

    def _check_trading_engine(self) -> bool:
        """Check trading engine status"""
        try:
            # This would check if trading processes are running
            return True
        except Exception:
            return False

    def _heal_database_connection(self):
        """Attempt to heal database connection"""
        service_name = 'database'

        if self.restart_counts[service_name] >= self.max_restarts:
            self.logger.error(f"Max restarts reached for {service_name}")
            return

        try:
            self.logger.info(f"Attempting to heal {service_name} connection...")
            # Implement database restart logic
            self.restart_counts[service_name] += 1
            self.logger.info(f"{service_name} healing attempted")

        except Exception as e:
            self.logger.error(f"Failed to heal {service_name}: {e}")

    def _heal_redis_connection(self):
        """Attempt to heal Redis connection"""
        service_name = 'redis'

        if self.restart_counts[service_name] >= self.max_restarts:
            self.logger.error(f"Max restarts reached for {service_name}")
            return

        try:
            self.logger.info(f"Attempting to heal {service_name} connection...")
            # Implement Redis restart logic
            self.restart_counts[service_name] += 1
            self.logger.info(f"{service_name} healing attempted")

        except Exception as e:
            self.logger.error(f"Failed to heal {service_name}: {e}")

    def _heal_trading_engine(self):
        """Attempt to heal trading engine"""
        service_name = 'trading_engine'

        if self.restart_counts[service_name] >= self.max_restarts:
            self.logger.error(f"Max restarts reached for {service_name}")
            return

        try:
            self.logger.info(f"Attempting to restart {service_name}...")
            # Implement trading engine restart logic
            self.restart_counts[service_name] += 1
            self.logger.info(f"{service_name} restart attempted")

        except Exception as e:
            self.logger.error(f"Failed to restart {service_name}: {e}")

class TitanovaXMonitoringSystem:
    """Main monitoring system combining all components"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.metrics_collector = MetricsCollector(config_manager)
        self.anomaly_detector = AnomalyDetector(config_manager)
        self.alert_manager = AlertManager(config_manager)
        self.self_healing = SelfHealingManager(config_manager)

        # Monitoring thread
        self.monitoring_thread = None
        self._start_monitoring()

    def _start_monitoring(self):
        """Start monitoring thread"""
        def monitoring_worker():
            while True:
                try:
                    # Collect metrics
                    system_metrics = self.metrics_collector._get_system_metrics()

                    # Update anomaly detector baseline
                    self.anomaly_detector.update_baseline(system_metrics)

                    # Detect anomalies
                    anomalies = self.anomaly_detector.detect_anomalies(system_metrics)

                    if anomalies:
                        self.logger.warning(f"Anomalies detected: {len(anomalies)}")
                        for anomaly in anomalies:
                            self.logger.warning(f"Anomaly: {anomaly}")

                    # Check alerts
                    alerts = self.alert_manager.check_alerts(self.metrics_collector)

                    if alerts:
                        self.logger.info(f"Alerts triggered: {len(alerts)}")

                    # Self-healing check
                    self.self_healing.check_and_heal()

                    # Sleep for 1 minute
                    time.sleep(60)

                except Exception as e:
                    self.logger.error(f"Monitoring system error: {e}")
                    time.sleep(60)

        self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Get metrics summary
            system_summary = self.metrics_collector.get_system_metrics_summary(minutes=60)

            # Get trading metrics
            trading_summary = {}
            if self.metrics_collector.trading_metrics:
                latest_trading = self.metrics_collector.trading_metrics[-1]
                trading_summary = {
                    'total_trades': latest_trading.total_trades,
                    'win_rate': latest_trading.win_rate,
                    'total_pnl': latest_trading.total_pnl,
                    'active_positions': latest_trading.active_positions
                }

            # Get model metrics
            model_summaries = {}
            for model_name, metrics_list in self.metrics_collector.model_metrics.items():
                if metrics_list:
                    latest = metrics_list[-1]
                    model_summaries[model_name] = {
                        'accuracy': latest.prediction_accuracy,
                        'latency_ms': latest.inference_latency_ms,
                        'error_count': latest.error_count
                    }

            # Get service status
            service_status = {
                'database': self.self_healing._check_database_connection(),
                'redis': self.self_healing._check_redis_connection(),
                'trading_engine': self.self_healing._check_trading_engine()
            }

            return {
                'timestamp': datetime.now(),
                'system': system_summary,
                'trading': trading_summary,
                'models': model_summaries,
                'services': service_status,
                'alerts_enabled': self.config_manager.get_system_config().enable_telegram_alerts
            }

        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Demo usage
    from config_manager import get_config_manager

    try:
        config = get_config_manager()
        monitoring = TitanovaXMonitoringSystem(config)

        # Get system health
        health = monitoring.get_system_health()
        print(f"System Health: {json.dumps(health, indent=2, default=str)}")

        # Simulate some time passing
        import time
        print("Monitoring system started. Press Ctrl+C to exit.")
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        print("Monitoring stopped.")
    except Exception as e:
        print(f"Demo failed: {e}")
