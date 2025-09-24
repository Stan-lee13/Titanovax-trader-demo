#!/usr/bin/env python3
"""
TitanovaX Security System
Multi-layered security with HMAC validation, rate limiting, and intrusion detection
"""

import hmac
import hashlib
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict
import ipaddress
import re

@dataclass
class SecurityEvent:
    """Security event record"""
    timestamp: datetime
    event_type: str  # 'rate_limit_exceeded', 'invalid_signature', 'suspicious_ip', etc.
    severity: str  # 'low', 'medium', 'high', 'critical'
    source_ip: str
    user_agent: str
    endpoint: str
    details: Dict[str, Any]
    resolved: bool = False

@dataclass
class RateLimitRule:
    """Rate limiting rule"""
    name: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    cooldown_minutes: int

class HMACValidator:
    """HMAC signature validation system"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # HMAC configuration
        self.secret_key = config_manager.security.hmac_secret_key
        self.algorithm = config_manager.security.jwt_algorithm
        self.max_signature_age = 300  # 5 minutes

        if not self.secret_key:
            raise ValueError("HMAC secret key not configured")

    def generate_signature(self, message: str, timestamp: Optional[str] = None) -> str:
        """Generate HMAC signature for message"""
        if timestamp is None:
            timestamp = str(int(time.time()))

        message_to_sign = f"{timestamp}:{message}"

        signature = hmac.new(
            self.secret_key.encode(),
            message_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{timestamp}:{signature}"

    def validate_signature(self, signature: str, message: str) -> Tuple[bool, str]:
        """Validate HMAC signature"""
        try:
            if ':' not in signature:
                return False, "Invalid signature format"

            timestamp_str, provided_signature = signature.split(':', 1)
            timestamp = int(timestamp_str)

            # Check timestamp age
            current_time = int(time.time())
            if current_time - timestamp > self.max_signature_age:
                return False, f"Signature too old: {current_time - timestamp}s"

            # Generate expected signature
            expected_signature = self.generate_signature(message, timestamp_str)

            # Use compare_digest for security against timing attacks
            if not hmac.compare_digest(signature, expected_signature):
                return False, "Signature mismatch"

            return True, "Valid signature"

        except (ValueError, IndexError) as e:
            return False, f"Signature parsing failed: {e}"
        except Exception as e:
            return False, f"Signature validation error: {e}"

class RateLimiter:
    """Rate limiting system with multiple time windows"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Rate limit storage
        self.requests_per_minute = defaultdict(list)  # ip -> list of timestamps
        self.requests_per_hour = defaultdict(list)
        self.requests_per_day = defaultdict(list)

        # Configuration
        self.default_rules = {
            'api': RateLimitRule(
                name='api',
                requests_per_minute=100,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=20,
                cooldown_minutes=5
            ),
            'trading': RateLimitRule(
                name='trading',
                requests_per_minute=50,
                requests_per_hour=500,
                requests_per_day=5000,
                burst_limit=10,
                cooldown_minutes=10
            ),
            'telegram': RateLimitRule(
                name='telegram',
                requests_per_minute=30,
                requests_per_hour=300,
                requests_per_day=3000,
                burst_limit=5,
                cooldown_minutes=1
            )
        }

        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_requests, daemon=True)
        self.cleanup_thread.start()

    def check_rate_limit(self, ip: str, endpoint: str, rule_name: str = 'api') -> Tuple[bool, str]:
        """Check if request is within rate limits"""
        rule = self.default_rules.get(rule_name, self.default_rules['api'])
        current_time = time.time()

        # Clean up old requests first
        self._cleanup_expired(ip, current_time)

        # Get request counts
        minute_count = len(self.requests_per_minute[ip])
        hour_count = len(self.requests_per_hour[ip])
        day_count = len(self.requests_per_day[ip])

        # Check limits
        if minute_count >= rule.requests_per_minute:
            return False, f"Rate limit exceeded: {minute_count}/{rule.requests_per_minute} per minute"

        if hour_count >= rule.requests_per_hour:
            return False, f"Rate limit exceeded: {hour_count}/{rule.requests_per_hour} per hour"

        if day_count >= rule.requests_per_day:
            return False, f"Rate limit exceeded: {day_count}/{rule.requests_per_day} per day"

        # Record this request
        self.requests_per_minute[ip].append(current_time)
        self.requests_per_hour[ip].append(current_time)
        self.requests_per_day[ip].append(current_time)

        return True, "Request allowed"

    def _cleanup_expired(self, ip: str, current_time: float):
        """Clean up expired timestamps for an IP"""
        # Clean minute data (keep last 2 minutes)
        cutoff_minute = current_time - 120
        self.requests_per_minute[ip] = [
            t for t in self.requests_per_minute[ip] if t > cutoff_minute
        ]

        # Clean hour data (keep last 2 hours)
        cutoff_hour = current_time - 7200
        self.requests_per_hour[ip] = [
            t for t in self.requests_per_hour[ip] if t > cutoff_hour
        ]

        # Clean day data (keep last 48 hours)
        cutoff_day = current_time - 172800
        self.requests_per_day[ip] = [
            t for t in self.requests_per_day[ip] if t > cutoff_day
        ]

    def _cleanup_old_requests(self):
        """Background cleanup of old request data"""
        while True:
            try:
                current_time = time.time()
                inactive_ips = []

                # Find IPs to clean up
                for ip in list(self.requests_per_minute.keys()):
                    if not self.requests_per_minute[ip] and not self.requests_per_hour[ip] and not self.requests_per_day[ip]:
                        inactive_ips.append(ip)
                    else:
                        self._cleanup_expired(ip, current_time)

                # Remove inactive IPs
                for ip in inactive_ips:
                    del self.requests_per_minute[ip]
                    del self.requests_per_hour[ip]
                    del self.requests_per_day[ip]

                time.sleep(300)  # Clean up every 5 minutes

            except Exception as e:
                self.logger.error(f"Rate limiter cleanup failed: {e}")
                time.sleep(300)

class IPFilter:
    """IP address filtering and whitelist/blacklist management"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # IP lists
        self.whitelist = set()
        self.blacklist = set()
        self.suspicious_ips = set()

        # Load IP lists
        self._load_ip_lists()

        # Track failed attempts per IP
        self.failed_attempts = defaultdict(int)
        self.last_attempt = defaultdict(float)

    def _load_ip_lists(self):
        """Load IP whitelist/blacklist from configuration"""
        # These would typically be loaded from database or config files
        # For now, using empty lists - can be populated via admin interface
        pass

    def is_ip_allowed(self, ip: str) -> Tuple[bool, str]:
        """Check if IP address is allowed"""
        try:
            ip_obj = ipaddress.ip_address(ip)

            # Check blacklist
            if ip in self.blacklist:
                return False, "IP is blacklisted"

            # Check whitelist
            if self.whitelist and ip not in self.whitelist:
                return False, "IP not in whitelist"

            # Check if IP is suspicious
            if ip in self.suspicious_ips:
                return False, "IP marked as suspicious"

            return True, "IP allowed"

        except ValueError:
            return False, "Invalid IP address format"

    def record_failed_attempt(self, ip: str):
        """Record a failed authentication or request"""
        current_time = time.time()

        # Check for rapid fire attempts (brute force detection)
        if current_time - self.last_attempt[ip] < 1.0:  # Less than 1 second apart
            self.failed_attempts[ip] += 2  # Double penalty for rapid attempts
        else:
            self.failed_attempts[ip] += 1

        self.last_attempt[ip] = current_time

        # Mark as suspicious if too many failures
        if self.failed_attempts[ip] > 10:
            self.suspicious_ips.add(ip)
            self.logger.warning(f"IP {ip} marked as suspicious due to {self.failed_attempts[ip]} failed attempts")

    def record_successful_attempt(self, ip: str):
        """Record a successful authentication or request"""
        # Reset failed attempt counter on success
        if ip in self.failed_attempts:
            self.failed_attempts[ip] = 0
            self.suspicious_ips.discard(ip)

class CircuitBreaker:
    """Circuit breaker pattern for external services"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Circuit breaker states
        self.states = {}  # service_name -> state
        self.failure_counts = defaultdict(int)
        self.last_failure_time = defaultdict(float)
        self.success_counts = defaultdict(int)

        # Configuration
        self.failure_threshold = 5
        self.recovery_timeout = 60  # seconds
        self.success_threshold = 3  # successes needed to close circuit

    def call_service(self, service_name: str, service_function, *args, **kwargs):
        """Call a service with circuit breaker protection"""
        state = self._get_state(service_name)

        if state == 'open':
            if time.time() - self.last_failure_time[service_name] > self.recovery_timeout:
                # Try to close circuit
                self.states[service_name] = 'half_open'
                self.logger.info(f"Circuit breaker for {service_name} entering half-open state")
            else:
                raise Exception(f"Circuit breaker open for {service_name}")

        try:
            result = service_function(*args, **kwargs)
            self._record_success(service_name)
            return result

        except Exception as e:
            self._record_failure(service_name)
            raise e

    def _get_state(self, service_name: str) -> str:
        """Get current circuit breaker state"""
        if service_name not in self.states:
            self.states[service_name] = 'closed'
            return 'closed'

        return self.states[service_name]

    def _record_success(self, service_name: str):
        """Record successful service call"""
        self.success_counts[service_name] += 1

        if self.states[service_name] == 'half_open' and self.success_counts[service_name] >= self.success_threshold:
            self.states[service_name] = 'closed'
            self.failure_counts[service_name] = 0
            self.logger.info(f"Circuit breaker for {service_name} closed")

    def _record_failure(self, service_name: str):
        """Record failed service call"""
        self.failure_counts[service_name] += 1
        self.last_failure_time[service_name] = time.time()

        if self.failure_counts[service_name] >= self.failure_threshold:
            self.states[service_name] = 'open'
            self.logger.warning(f"Circuit breaker for {service_name} opened after {self.failure_counts[service_name]} failures")

class SecurityValidator:
    """Main security validation system"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.hmac_validator = HMACValidator(config_manager)
        self.rate_limiter = RateLimiter(config_manager)
        self.ip_filter = IPFilter(config_manager)
        self.circuit_breaker = CircuitBreaker(config_manager)

        # Security event storage
        self.security_events = []
        self.max_events = 10000

        # Input validation patterns
        self._compile_validation_patterns()

    def _compile_validation_patterns(self):
        """Compile regex patterns for input validation"""
        # SQL injection patterns
        self.sql_patterns = [
            r'(?i)\bselect\b.*\bfrom\b',
            r'(?i)\bunion\b.*\bselect\b',
            r'(?i)\bdrop\b.*\btable\b',
            r'(?i)\bdelete\b.*\bfrom\b',
            r';.*--',
            r';.*#',
        ]

        # XSS patterns
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
        ]

        # Path traversal patterns
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e%2f',
            r'%2e%2e%5c',
        ]

    def validate_request(self, request_data: Dict[str, Any], ip: str, user_agent: str = '') -> Tuple[bool, str]:
        """Validate incoming request"""
        try:
            # Check IP address
            allowed, reason = self.ip_filter.is_ip_allowed(ip)
            if not allowed:
                self._record_security_event('ip_blocked', 'high', ip, user_agent, 'api', {'reason': reason})
                return False, reason

            # Check rate limits
            endpoint = request_data.get('endpoint', 'api')
            rule_name = 'api'  # Could be determined by endpoint

            allowed, reason = self.rate_limiter.check_rate_limit(ip, endpoint, rule_name)
            if not allowed:
                self._record_security_event('rate_limit_exceeded', 'medium', ip, user_agent, endpoint, {'reason': reason})
                return False, reason

            # Validate input data
            validation_errors = self._validate_input_data(request_data)
            if validation_errors:
                self._record_security_event('input_validation_failed', 'medium', ip, user_agent, endpoint, {'errors': validation_errors})
                return False, f"Input validation failed: {', '.join(validation_errors)}"

            # Check for suspicious patterns
            suspicious_patterns = self._check_suspicious_patterns(request_data)
            if suspicious_patterns:
                self._record_security_event('suspicious_activity', 'high', ip, user_agent, endpoint, {'patterns': suspicious_patterns})
                self.ip_filter.record_failed_attempt(ip)
                return False, "Suspicious activity detected"

            return True, "Request validated successfully"

        except Exception as e:
            self.logger.error(f"Request validation failed: {e}")
            return False, f"Validation error: {e}"

    def validate_trading_signal(self, signal_data: Dict[str, Any], signature: str) -> Tuple[bool, str]:
        """Validate trading signal with HMAC signature"""
        try:
            # Validate HMAC signature
            message = json.dumps(signal_data, sort_keys=True)
            is_valid, reason = self.hmac_validator.validate_signature(signature, message)

            if not is_valid:
                self._record_security_event('invalid_signature', 'high', 'unknown', '', 'trading_signal', {'reason': reason})
                return False, reason

            # Validate signal structure
            validation_errors = self._validate_trading_signal(signal_data)
            if validation_errors:
                return False, f"Signal validation failed: {', '.join(validation_errors)}"

            return True, "Signal validated successfully"

        except Exception as e:
            self.logger.error(f"Trading signal validation failed: {e}")
            return False, f"Signal validation error: {e}"

    def _validate_input_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate input data for security issues"""
        errors = []

        def validate_value(value, key_path: str = ''):
            if isinstance(value, str):
                # Check length limits
                if len(value) > 10000:
                    errors.append(f"Value too long at {key_path}")

                # Check for SQL injection
                for pattern in self.sql_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(f"Potential SQL injection at {key_path}")

                # Check for XSS
                for pattern in self.xss_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(f"Potential XSS at {key_path}")

                # Check for path traversal
                for pattern in self.path_traversal_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(f"Potential path traversal at {key_path}")

            elif isinstance(value, dict):
                for k, v in value.items():
                    validate_value(v, f"{key_path}.{k}" if key_path else k)

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    validate_value(item, f"{key_path}[{i}]")

        validate_value(data)
        return errors

    def _validate_trading_signal(self, signal: Dict[str, Any]) -> List[str]:
        """Validate trading signal structure"""
        errors = []

        required_fields = ['symbol', 'action', 'size', 'timestamp']
        for field in required_fields:
            if field not in signal:
                errors.append(f"Missing required field: {field}")

        # Validate action
        if 'action' in signal:
            valid_actions = ['BUY', 'SELL', 'HOLD']
            if signal['action'] not in valid_actions:
                errors.append(f"Invalid action: {signal['action']}")

        # Validate size
        if 'size' in signal:
            try:
                size = float(signal['size'])
                if size <= 0:
                    errors.append("Size must be positive")
            except (ValueError, TypeError):
                errors.append("Size must be a valid number")

        # Validate symbol format
        if 'symbol' in signal:
            symbol = signal['symbol']
            if not isinstance(symbol, str) or len(symbol) < 3 or len(symbol) > 10:
                errors.append("Invalid symbol format")

        return errors

    def _check_suspicious_patterns(self, data: Dict[str, Any]) -> List[str]:
        """Check for suspicious patterns in request data"""
        suspicious = []

        def check_value(value):
            if isinstance(value, str):
                # Check for extremely long strings
                if len(value) > 5000:
                    suspicious.append("Unusually long string")

                # Check for repeated characters
                if len(value) > 100:
                    char_counts = defaultdict(int)
                    for char in value:
                        char_counts[char] += 1

                    max_repetitions = max(char_counts.values())
                    if max_repetitions > len(value) * 0.8:
                        suspicious.append("Suspicious character repetition")

            elif isinstance(value, dict):
                for v in value.values():
                    check_value(v)

            elif isinstance(value, list):
                for item in value:
                    check_value(item)

        check_value(data)
        return suspicious

    def _record_security_event(self, event_type: str, severity: str, source_ip: str,
                             user_agent: str, endpoint: str, details: Dict[str, Any]):
        """Record a security event"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            details=details
        )

        self.security_events.append(event)

        # Keep only recent events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]

        # Log security event
        log_level = {
            'low': logging.DEBUG,
            'medium': logging.INFO,
            'high': logging.WARNING,
            'critical': logging.ERROR
        }.get(severity, logging.INFO)

        self.logger.log(log_level, f"Security Event: {event_type} from {source_ip} - {details}")

    def get_security_report(self) -> Dict[str, Any]:
        """Get security system report"""
        recent_events = [
            event for event in self.security_events
            if (datetime.now() - event.timestamp).total_seconds() < 3600  # Last hour
        ]

        events_by_type = defaultdict(int)
        events_by_severity = defaultdict(int)

        for event in recent_events:
            events_by_type[event.event_type] += 1
            events_by_severity[event.severity] += 1

        return {
            'total_events_24h': len(self.security_events),
            'events_last_hour': len(recent_events),
            'events_by_type': dict(events_by_type),
            'events_by_severity': dict(events_by_severity),
            'rate_limited_ips': len(self.rate_limiter.requests_per_minute),
            'suspicious_ips': len(self.ip_filter.suspicious_ips)
        }

class TitanovaXSecuritySystem:
    """Main security system combining all components"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Initialize security components
        self.validator = SecurityValidator(config_manager)

    def validate_api_request(self, request_data: Dict[str, Any], ip: str, user_agent: str = '') -> Tuple[bool, str]:
        """Validate API request"""
        return self.validator.validate_request(request_data, ip, user_agent)

    def validate_trading_signal(self, signal_data: Dict[str, Any], signature: str) -> Tuple[bool, str]:
        """Validate trading signal"""
        return self.validator.validate_trading_signal(signal_data, signature)

    def generate_signal_signature(self, signal_data: Dict[str, Any]) -> str:
        """Generate signature for trading signal"""
        message = json.dumps(signal_data, sort_keys=True)
        return self.validator.hmac_validator.generate_signature(message)

    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report"""
        return self.validator.get_security_report()

if __name__ == "__main__":
    # Demo usage
    from config_manager import get_config_manager

    try:
        config = get_config_manager()
        security = TitanovaXSecuritySystem(config)

        # Test HMAC validation
        test_signal = {
            'symbol': 'EURUSD',
            'action': 'BUY',
            'size': 0.1,
            'timestamp': int(time.time())
        }

        signature = security.generate_signal_signature(test_signal)
        print(f"Generated signature: {signature}")

        is_valid, reason = security.validate_trading_signal(test_signal, signature)
        print(f"Signature validation: {is_valid} - {reason}")

        # Test security report
        report = security.get_security_report()
        print(f"Security report: {json.dumps(report, indent=2)}")

    except Exception as e:
        print(f"Security system demo failed: {e}")
