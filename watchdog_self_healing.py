#!/usr/bin/env python3
"""
Watchdog & Self-Healing System for TitanovaX
Monitors services, restarts failed components, and provides immune-system like behavior
"""

import logging
import json
import psutil
import time
import threading
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import signal
import os
import requests
from collections import defaultdict

@dataclass
class ServiceStatus:
    """Status of a monitored service"""
    name: str
    process_id: Optional[int]
    is_running: bool
    last_heartbeat: Optional[datetime]
    restart_count: int
    last_restart_attempt: Optional[datetime]
    health_score: float  # 0.0 to 1.0
    status_message: str
    quarantine_until: Optional[datetime]

@dataclass
class SystemHealth:
    """Overall system health metrics"""
    timestamp: datetime
    overall_health: float
    service_health: Dict[str, float]
    resource_usage: Dict[str, float]
    anomaly_count: int
    quarantine_count: int

class WatchdogService:
    """Individual service monitoring and management"""

    def __init__(self, name: str, command: List[str], health_check_url: Optional[str] = None,
                 health_check_interval: int = 30, max_restarts: int = 5,
                 restart_delay: int = 60):
        self.name = name
        self.command = command
        self.health_check_url = health_check_url
        self.health_check_interval = health_check_interval
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay

        self.process: Optional[psutil.Process] = None
        self.status = ServiceStatus(
            name=name,
            process_id=None,
            is_running=False,
            last_heartbeat=None,
            restart_count=0,
            last_restart_attempt=None,
            health_score=1.0,
            status_message="Not started",
            quarantine_until=None
        )

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def start(self) -> bool:
        """Start the service"""

        if self.status.quarantine_until and datetime.now() < self.status.quarantine_until:
            self.logger.warning(f"Service {self.name} is in quarantine until {self.status.quarantine_until}")
            return False

        try:
            # Start the process
            self.process = psutil.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.status.process_id = self.process.pid
            self.status.is_running = True
            self.status.status_message = "Running"
            self.status.last_restart_attempt = datetime.now()

            self.logger.info(f"Started service {self.name} with PID {self.process.pid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start service {self.name}: {e}")
            self.status.is_running = False
            self.status.status_message = f"Failed to start: {str(e)}"
            return False

    def stop(self) -> bool:
        """Stop the service"""

        if not self.process:
            return True

        try:
            # Terminate the process
            self.process.terminate()

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=10)
            except psutil.TimeoutExpired:
                # Force kill if not terminated
                self.process.kill()
                self.process.wait()

            self.process = None
            self.status.process_id = None
            self.status.is_running = False
            self.status.status_message = "Stopped"

            self.logger.info(f"Stopped service {self.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop service {self.name}: {e}")
            return False

    def restart(self) -> bool:
        """Restart the service"""

        if self.status.restart_count >= self.max_restarts:
            self.logger.error(f"Service {self.name} has exceeded max restarts ({self.max_restarts})")
            return False

        self.logger.info(f"Restarting service {self.name} (attempt {self.status.restart_count + 1})")

        # Stop the service
        self.stop()

        # Wait before restarting
        time.sleep(2)

        # Start the service
        success = self.start()

        if success:
            self.status.restart_count += 1
            self.status.health_score = max(0.1, self.status.health_score * 0.9)  # Penalty for restart
        else:
            # Put in quarantine if restart failed
            self.status.quarantine_until = datetime.now() + timedelta(minutes=self.restart_delay)

        return success

    def check_health(self) -> float:
        """Check service health and return health score (0.0 to 1.0)"""

        if not self.process:
            self.status.is_running = False
            self.status.status_message = "Process not found"
            self.status.health_score = 0.0
            return 0.0

        try:
            # Check if process is still running
            if not self.process.is_running():
                self.status.is_running = False
                self.status.status_message = "Process died"
                self.status.health_score = 0.0
                return 0.0

            self.status.is_running = True

            # Check process health metrics
            health_score = 1.0

            try:
                # CPU usage check
                cpu_percent = self.process.cpu_percent(interval=1)
                if cpu_percent > 90:
                    health_score *= 0.8
                    self.status.status_message = f'High CPU usage: {cpu_percent:.1f}%'

                # Memory usage check
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                if memory_percent > 80:
                    health_score *= 0.7
                    self.status.status_message = f'High memory usage: {memory_percent:.1f}%'

                # Check for zombie processes
                if self.process.status() == psutil.STATUS_ZOMBIE:
                    health_score = 0.0
                    self.status.status_message = "Zombie process"

            except Exception as e:
                self.logger.warning(f"Could not get process metrics for {self.name}: {e}")
                health_score *= 0.9

            # HTTP health check if URL provided
            if self.health_check_url:
                try:
                    response = requests.get(self.health_check_url, timeout=5)
                    if response.status_code != 200:
                        health_score *= 0.5
                        self.status.status_message = f"HTTP health check failed: {response.status_code}"
                except Exception as e:
                    health_score *= 0.3
                    self.status.status_message = f"HTTP health check error: {str(e)}"

            # Update heartbeat
            self.status.last_heartbeat = datetime.now()
            self.status.health_score = max(0.0, min(1.0, health_score))

            return self.status.health_score

        except psutil.NoSuchProcess:
            self.status.is_running = False
            self.status.process_id = None
            self.status.status_message = "Process not found"
            self.status.health_score = 0.0
            return 0.0
        except Exception as e:
            self.logger.error(f"Error checking health for {self.name}: {e}")
            self.status.health_score = 0.0
            return 0.0

class Watchdog:
    """Main watchdog system for monitoring and self-healing"""

    def __init__(self, config_path: str = 'config/watchdog_config.json'):
        self.config_path = Path(config_path)
        self.services: Dict[str, WatchdogService] = {}
        self.health_history: List[SystemHealth] = []
        self.max_history = 100

        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.healing_thread: Optional[threading.Thread] = None

        self.load_config()
        self.setup_logging()

    def load_config(self):
        """Load watchdog configuration"""
        default_config = {
            "monitor_interval_seconds": 10,
            "health_check_interval_seconds": 30,
            "auto_restart_enabled": True,
            "quarantine_duration_minutes": 5,
            "max_consecutive_failures": 3,
            "services": {
                "ml_brain": {
                    "command": ["python", "-m", "ml_brain.inference.onnx_server"],
                    "health_check_url": "http://localhost:8000/health",
                    "health_check_interval": 30,
                    "max_restarts": 5,
                    "restart_delay": 60
                },
                "orchestrator": {
                    "command": ["python", "-m", "orchestration.decision_engine"],
                    "health_check_url": "http://localhost:8001/health",
                    "health_check_interval": 30,
                    "max_restarts": 3,
                    "restart_delay": 120
                }
            },
            "system_health_thresholds": {
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_percent": 90.0
            }
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    # Merge service configs
                    if "services" in config_data:
                        default_config["services"].update(config_data["services"])
                    self.config = {**default_config, **{k: v for k, v in config_data.items() if k != "services"}}
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load watchdog config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def add_service(self, name: str, command: List[str], health_check_url: Optional[str] = None,
                   health_check_interval: int = 30, max_restarts: int = 5,
                   restart_delay: int = 60):
        """Add a service to monitor"""

        service = WatchdogService(
            name=name,
            command=command,
            health_check_url=health_check_url,
            health_check_interval=health_check_interval,
            max_restarts=max_restarts,
            restart_delay=restart_delay
        )

        self.services[name] = service
        self.logger.info(f"Added service {name} for monitoring")

    def start_all_services(self) -> bool:
        """Start all configured services"""

        success_count = 0

        for name, service in self.services.items():
            if service.start():
                success_count += 1
            else:
                self.logger.error(f"Failed to start service {name}")

        self.logger.info(f"Started {success_count}/{len(self.services)} services")
        return success_count == len(self.services)

    def stop_all_services(self) -> bool:
        """Stop all services"""

        success_count = 0

        for name, service in self.services.items():
            if service.stop():
                success_count += 1
            else:
                self.logger.error(f"Failed to stop service {name}")

        self.logger.info(f"Stopped {success_count}/{len(self.services)} services")
        return success_count == len(self.services)

    def start(self):
        """Start the watchdog system"""

        if self.is_running:
            self.logger.warning("Watchdog is already running")
            return

        self.is_running = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        # Start healing thread
        self.healing_thread = threading.Thread(target=self._healing_loop, daemon=True)
        self.healing_thread.start()

        self.logger.info("Watchdog system started")

    def stop(self):
        """Stop the watchdog system"""

        if not self.is_running:
            return

        self.is_running = False

        # Stop all services
        self.stop_all_services()

        self.logger.info("Watchdog system stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""

        self.logger.info("Starting monitoring loop")

        while self.is_running:
            try:
                self._check_all_services()
                self._record_system_health()
                time.sleep(self.config["monitor_interval_seconds"])
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)

    def _healing_loop(self):
        """Main healing loop for automatic recovery"""

        self.logger.info("Starting healing loop")

        while self.is_running:
            try:
                self._perform_healing_actions()
                time.sleep(30)  # Check for healing opportunities every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in healing loop: {e}")
                time.sleep(10)

    def _check_all_services(self):
        """Check health of all services"""

        for name, service in self.services.items():
            try:
                health_score = service.check_health()

                if health_score == 0.0 and service.status.is_running:
                    self.logger.warning(f"Service {name} is unhealthy")

                    # Attempt restart if auto-restart is enabled
                    if self.config["auto_restart_enabled"]:
                        self._handle_service_failure(service)

            except Exception as e:
                self.logger.error(f"Error checking service {name}: {e}")

    def _handle_service_failure(self, service: WatchdogService):
        """Handle service failure with appropriate action"""

        if service.status.quarantine_until and datetime.now() < service.status.quarantine_until:
            self.logger.info(f"Service {service.name} is in quarantine, not restarting")
            return

        # Check consecutive failures
        if service.status.restart_count >= self.max_restarts:
            self.logger.error(f"Service {service.name} has exceeded max restarts, quarantining")
            service.status.quarantine_until = datetime.now() + timedelta(minutes=self.config["quarantine_duration_minutes"])
            return

        # Attempt restart
        self.logger.info(f"Attempting to restart service {service.name}")
        success = service.restart()

        if success:
            self.logger.info(f"Successfully restarted service {service.name}")
        else:
            self.logger.error(f"Failed to restart service {service.name}")

    def _perform_healing_actions(self):
        """Perform automatic healing actions"""

        # Check for quarantined services that can be released
        current_time = datetime.now()
        for name, service in self.services.items():
            if service.status.quarantine_until and current_time >= service.status.quarantine_until:
                self.logger.info(f"Releasing service {name} from quarantine")
                service.status.quarantine_until = None
                service.status.restart_count = 0  # Reset restart count

                # Try to start the service
                if service.start():
                    self.logger.info(f"Successfully started quarantined service {name}")
                else:
                    self.logger.error(f"Failed to start quarantined service {name}")

        # Check for system resource issues
        self._check_system_resources()

        # Perform garbage collection if needed
        self._perform_garbage_collection()

    def _check_system_resources(self):
        """Check system resource usage"""

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            thresholds = self.config["system_health_thresholds"]

            if cpu_percent > thresholds["cpu_percent"]:
                self.logger.warning(f'High CPU usage: {cpu_percent:.1f}%')

            if memory_percent > thresholds["memory_percent"]:
                self.logger.warning(f'High memory usage: {memory_percent:.1f}%')

            if disk_percent > thresholds["disk_percent"]:
                self.logger.warning(f'High disk usage: {disk_percent:.1f}%')

        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")

    def _perform_garbage_collection(self):
        """Perform garbage collection and cleanup"""

        try:
            # Clean up old log files
            self._cleanup_old_logs()

            # Clean up temporary files
            self._cleanup_temp_files()

        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")

    def _cleanup_old_logs(self):
        """Clean up old log files"""

        log_dir = Path("data/logs")
        if not log_dir.exists():
            return

        # Keep logs for last 7 days
        cutoff_time = datetime.now() - timedelta(days=7)

        for log_file in log_dir.glob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    log_file.unlink()
                    self.logger.info(f"Cleaned up old log file: {log_file}")
            except Exception as e:
                self.logger.warning(f"Could not clean up log file {log_file}: {e}")

    def _cleanup_temp_files(self):
        """Clean up temporary files"""

        temp_dir = Path("data/temp")
        if not temp_dir.exists():
            return

        # Clean up files older than 1 hour
        cutoff_time = datetime.now() - timedelta(hours=1)

        for temp_file in temp_dir.glob("*"):
            try:
                if temp_file.stat().st_mtime < cutoff_time.timestamp():
                    if temp_file.is_file():
                        temp_file.unlink()
                    else:
                        # Remove empty directories
                        temp_file.rmdir()
                    self.logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Could not clean up temp file {temp_file}: {e}")

    def _record_system_health(self):
        """Record overall system health"""

        service_health = {name: service.status.health_score for name, service in self.services.items()}

        # Calculate overall health
        if service_health:
            overall_health = sum(service_health.values()) / len(service_health)
        else:
            overall_health = 1.0

        # Get system resource usage
        resource_usage = {}
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            resource_usage = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            }
        except Exception as e:
            self.logger.warning(f"Could not get system resource usage: {e}")

        # Count anomalies and quarantined services
        anomaly_count = sum(1 for service in self.services.values() if service.status.health_score < 0.5)
        quarantine_count = sum(1 for service in self.services.values() if service.status.quarantine_until)

        health_record = SystemHealth(
            timestamp=datetime.now(),
            overall_health=overall_health,
            service_health=service_health,
            resource_usage=resource_usage,
            anomaly_count=anomaly_count,
            quarantine_count=quarantine_count
        )

        self.health_history.append(health_record)

        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""

        service_statuses = {}
        for name, service in self.services.items():
            service_statuses[name] = {
                "is_running": service.status.is_running,
                "health_score": service.status.health_score,
                "restart_count": service.status.restart_count,
                "last_heartbeat": service.status.last_heartbeat.isoformat() if service.status.last_heartbeat else None,
                "status_message": service.status.status_message,
                "quarantine_until": service.status.quarantine_until.isoformat() if service.status.quarantine_until else None
            }

        # Get recent health history
        recent_health = self.health_history[-10:] if self.health_history else []

        return {
            "is_running": self.is_running,
            "services": service_statuses,
            "overall_health": self.health_history[-1].overall_health if self.health_history else 1.0,
            "total_services": len(self.services),
            "healthy_services": sum(1 for s in self.services.values() if s.status.health_score > 0.8),
            "unhealthy_services": sum(1 for s in self.services.values() if s.status.health_score < 0.5),
            "quarantined_services": sum(1 for s in self.services.values() if s.status.quarantine_until),
            "recent_health_trend": "stable" if len(recent_health) < 2 else self._calculate_health_trend(recent_health)
        }

    def _calculate_health_trend(self, recent_health: List[SystemHealth]) -> str:
        """Calculate health trend from recent history"""

        if len(recent_health) < 2:
            return "stable"

        recent_scores = [h.overall_health for h in recent_health[-5:]]
        older_scores = [h.overall_health for h in recent_health[-10:-5]]

        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)

        if recent_avg > older_avg + 0.1:
            return "improving"
        elif recent_avg < older_avg - 0.1:
            return "degrading"
        else:
            return "stable"

if __name__ == "__main__":
    # Demo usage
    watchdog = Watchdog()

    # Add services (these would normally be real services)
    watchdog.add_service(
        name="demo_service",
        command=["python", "-c", "import time; time.sleep(30)"],
        health_check_url="http://localhost:8000/health"
    )

    # Start watchdog
    watchdog.start()

    try:
        # Monitor for a while
        print("Watchdog monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)

            # Print status
            status = watchdog.get_system_status()
            print(f"\\nSystem Status at {datetime.now()}:")
            print(f"  Overall health: {status['overall_health']:.2f}")
            print(f"  Healthy services: {status['healthy_services']}/{status['total_services']}")
            print(f"  Health trend: {status['recent_health_trend']}")

    except KeyboardInterrupt:
        print("\\nStopping watchdog...")
        watchdog.stop()
        print("Watchdog stopped.")
