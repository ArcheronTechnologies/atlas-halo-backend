"""
Drift and Forecast Monitoring System
Automatically monitors data and model performance for drift and anomalies
"""

from __future__ import annotations
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..analytics.forecasting import moving_average_forecast, detect_drift, is_forecast_anomaly
from ..insights.alerts import alert_manager, AlertRule
from ..common.performance import performance_tracked


class MonitoringType(Enum):
    DATA_DRIFT = "data_drift"
    MODEL_PERFORMANCE = "model_performance" 
    FORECAST_ACCURACY = "forecast_accuracy"
    ANOMALY_DETECTION = "anomaly_detection"
    SYSTEM_HEALTH = "system_health"


@dataclass
class MonitoringRule:
    """Configuration for monitoring rule"""
    rule_id: str
    name: str
    monitoring_type: MonitoringType
    metric_name: str
    threshold: float
    window_size: int
    check_interval_seconds: int
    alert_cooldown_seconds: int
    enabled: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class DriftDetectionResult:
    """Result of drift detection"""
    drift_detected: bool
    drift_score: float
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime


class DriftMonitor:
    """
    Monitors data and model performance for drift and anomalies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Store monitoring data
        self.metrics_data: Dict[str, List[Tuple[float, float]]] = {}
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        
        # Track last alerts to prevent spam
        self.last_alert_times: Dict[str, float] = {}
        
        # Background monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize default monitoring rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default monitoring rules"""
        
        # Model performance drift monitoring
        self.add_monitoring_rule(MonitoringRule(
            rule_id="model_accuracy_drift",
            name="Model Accuracy Drift Detection",
            monitoring_type=MonitoringType.MODEL_PERFORMANCE,
            metric_name="model_accuracy",
            threshold=0.15,  # 15% drift threshold
            window_size=20,
            check_interval_seconds=300,  # 5 minutes
            alert_cooldown_seconds=1800,  # 30 minutes
            metadata={"severity": "HIGH"}
        ))
        
        # Data quality drift monitoring
        self.add_monitoring_rule(MonitoringRule(
            rule_id="data_quality_drift",
            name="Data Quality Drift Detection",
            monitoring_type=MonitoringType.DATA_DRIFT,
            metric_name="data_quality_score",
            threshold=0.20,
            window_size=50,
            check_interval_seconds=600,  # 10 minutes
            alert_cooldown_seconds=3600,  # 1 hour
            metadata={"severity": "MEDIUM"}
        ))
        
        # System response time monitoring
        self.add_monitoring_rule(MonitoringRule(
            rule_id="response_time_anomaly",
            name="Response Time Anomaly Detection",
            monitoring_type=MonitoringType.ANOMALY_DETECTION,
            metric_name="avg_response_time",
            threshold=2.0,  # 2 standard deviations
            window_size=30,
            check_interval_seconds=120,  # 2 minutes
            alert_cooldown_seconds=600,   # 10 minutes
            metadata={"severity": "MEDIUM"}
        ))
        
        # Forecast accuracy monitoring
        self.add_monitoring_rule(MonitoringRule(
            rule_id="forecast_accuracy_degradation",
            name="Forecast Accuracy Degradation",
            monitoring_type=MonitoringType.FORECAST_ACCURACY,
            metric_name="forecast_mape",  # Mean Absolute Percentage Error
            threshold=0.25,  # 25% increase in error
            window_size=15,
            check_interval_seconds=1800,  # 30 minutes
            alert_cooldown_seconds=7200,  # 2 hours
            metadata={"severity": "HIGH"}
        ))
    
    def add_monitoring_rule(self, rule: MonitoringRule):
        """Add a monitoring rule"""
        self.monitoring_rules[rule.rule_id] = rule
        
        # Start monitoring task for this rule
        if rule.enabled and rule.rule_id not in self.monitoring_tasks:
            task = asyncio.create_task(self._monitoring_loop(rule))
            self.monitoring_tasks[rule.rule_id] = task
        
        self.logger.info(f"Added monitoring rule: {rule.name}")
    
    def remove_monitoring_rule(self, rule_id: str):
        """Remove a monitoring rule"""
        if rule_id in self.monitoring_rules:
            del self.monitoring_rules[rule_id]
        
        if rule_id in self.monitoring_tasks:
            self.monitoring_tasks[rule_id].cancel()
            del self.monitoring_tasks[rule_id]
        
        self.logger.info(f"Removed monitoring rule: {rule_id}")
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value for monitoring"""
        if timestamp is None:
            timestamp = time.time()
        
        if metric_name not in self.metrics_data:
            self.metrics_data[metric_name] = []
        
        self.metrics_data[metric_name].append((timestamp, value))
        
        # Keep only recent data (last 1000 points per metric)
        self.metrics_data[metric_name] = self.metrics_data[metric_name][-1000:]
    
    async def _monitoring_loop(self, rule: MonitoringRule):
        """Background monitoring loop for a specific rule"""
        while rule.enabled and rule.rule_id in self.monitoring_rules:
            try:
                await asyncio.sleep(rule.check_interval_seconds)
                
                # Check if enough data is available
                if rule.metric_name not in self.metrics_data:
                    continue
                
                data_points = self.metrics_data[rule.metric_name]
                if len(data_points) < rule.window_size:
                    continue
                
                # Perform monitoring check based on type
                if rule.monitoring_type == MonitoringType.DATA_DRIFT:
                    await self._check_data_drift(rule, data_points)
                elif rule.monitoring_type == MonitoringType.MODEL_PERFORMANCE:
                    await self._check_model_performance_drift(rule, data_points)
                elif rule.monitoring_type == MonitoringType.ANOMALY_DETECTION:
                    await self._check_anomaly(rule, data_points)
                elif rule.monitoring_type == MonitoringType.FORECAST_ACCURACY:
                    await self._check_forecast_accuracy(rule, data_points)
                elif rule.monitoring_type == MonitoringType.SYSTEM_HEALTH:
                    await self._check_system_health(rule, data_points)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring error for rule {rule.rule_id}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_data_drift(self, rule: MonitoringRule, data_points: List[Tuple[float, float]]):
        """Check for data drift"""
        if len(data_points) < rule.window_size * 2:
            return
        
        # Split data into historical and recent
        split_point = len(data_points) - rule.window_size
        historical_data = data_points[:split_point]
        recent_data = data_points[split_point:]
        
        drift_result = detect_drift(historical_data, recent_data, rule.threshold)
        
        if drift_result["drift_detected"]:
            await self._emit_drift_alert(
                rule, 
                drift_result["drift_score"],
                {
                    "type": "data_drift",
                    "metric": rule.metric_name,
                    "drift_details": drift_result
                }
            )
    
    async def _check_model_performance_drift(self, rule: MonitoringRule, data_points: List[Tuple[float, float]]):
        """Check for model performance drift"""
        if len(data_points) < rule.window_size * 2:
            return
        
        # Use statistical drift detection
        recent_window = data_points[-rule.window_size:]
        previous_window = data_points[-(rule.window_size * 2):-rule.window_size]
        
        drift_result = detect_drift(previous_window, recent_window, rule.threshold)
        
        if drift_result["drift_detected"]:
            await self._emit_drift_alert(
                rule,
                drift_result["drift_score"],
                {
                    "type": "model_performance_drift",
                    "metric": rule.metric_name,
                    "historical_mean": drift_result["historical_mean"],
                    "recent_mean": drift_result["recent_mean"],
                    "drift_score": drift_result["drift_score"]
                }
            )
    
    async def _check_anomaly(self, rule: MonitoringRule, data_points: List[Tuple[float, float]]):
        """Check for anomalies using forecasting"""
        if len(data_points) < 10:
            return
        
        # Use recent data for forecasting
        forecast_data = data_points[-30:] if len(data_points) >= 30 else data_points
        forecast = moving_average_forecast(forecast_data, window=min(10, len(forecast_data) // 2))
        
        # Check if latest value is anomalous
        latest_value = data_points[-1][1]
        
        if is_forecast_anomaly(latest_value, forecast):
            deviation = abs(latest_value - forecast["forecast"])
            severity = "HIGH" if deviation > forecast.get("std_dev", 0) * 3 else "MEDIUM"
            
            await self._emit_anomaly_alert(
                rule,
                latest_value,
                {
                    "type": "anomaly_detected",
                    "metric": rule.metric_name,
                    "value": latest_value,
                    "expected": forecast["forecast"],
                    "bounds": [forecast["lower"], forecast["upper"]],
                    "deviation": deviation,
                    "severity": severity
                }
            )
    
    async def _check_forecast_accuracy(self, rule: MonitoringRule, data_points: List[Tuple[float, float]]):
        """Check forecast accuracy degradation"""
        if len(data_points) < rule.window_size * 2:
            return
        
        # Calculate forecast errors for recent predictions
        errors = []
        window_size = 5
        
        for i in range(window_size, len(data_points)):
            # Use historical data to forecast next point
            historical = data_points[i-window_size:i]
            actual = data_points[i][1]
            
            forecast = moving_average_forecast(historical, window=min(3, len(historical)))
            predicted = forecast["forecast"]
            
            # Calculate percentage error
            if actual != 0:
                error = abs(predicted - actual) / abs(actual)
                errors.append((data_points[i][0], error))
        
        if len(errors) < rule.window_size:
            return
        
        # Check for degradation in forecast accuracy
        recent_errors = errors[-rule.window_size:]
        earlier_errors = errors[-(rule.window_size * 2):-rule.window_size]
        
        recent_avg_error = sum(e[1] for e in recent_errors) / len(recent_errors)
        earlier_avg_error = sum(e[1] for e in earlier_errors) / len(earlier_errors)
        
        # Check if error has increased significantly
        error_increase = (recent_avg_error - earlier_avg_error) / (earlier_avg_error + 0.001)
        
        if error_increase > rule.threshold:
            await self._emit_drift_alert(
                rule,
                error_increase,
                {
                    "type": "forecast_accuracy_degradation",
                    "metric": rule.metric_name,
                    "recent_error": recent_avg_error,
                    "baseline_error": earlier_avg_error,
                    "degradation": error_increase
                }
            )
    
    async def _check_system_health(self, rule: MonitoringRule, data_points: List[Tuple[float, float]]):
        """Check system health metrics"""
        if len(data_points) < rule.window_size:
            return
        
        recent_values = [point[1] for point in data_points[-rule.window_size:]]
        avg_value = sum(recent_values) / len(recent_values)
        
        # Define health thresholds based on metric name
        thresholds = {
            "cpu_usage": 0.90,
            "memory_usage": 0.85,
            "disk_usage": 0.90,
            "error_rate": 0.05,
            "response_time": 5000,  # ms
        }
        
        threshold = thresholds.get(rule.metric_name, rule.threshold)
        
        if avg_value > threshold:
            await self._emit_health_alert(
                rule,
                avg_value,
                {
                    "type": "system_health_degradation",
                    "metric": rule.metric_name,
                    "current_value": avg_value,
                    "threshold": threshold,
                    "severity": rule.metadata.get("severity", "MEDIUM")
                }
            )
    
    async def _emit_drift_alert(self, rule: MonitoringRule, drift_score: float, details: Dict[str, Any]):
        """Emit alert for drift detection"""
        current_time = time.time()
        
        # Check cooldown period
        if rule.rule_id in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[rule.rule_id]
            if time_since_last < rule.alert_cooldown_seconds:
                return
        
        # Emit alert
        alert_manager.ingest(
            metric=f"drift_detected_{rule.monitoring_type.value}",
            value=drift_score,
            labels={
                "rule_id": rule.rule_id,
                "metric_name": rule.metric_name,
                "severity": rule.metadata.get("severity", "MEDIUM"),
                **{k: str(v) for k, v in details.items() if isinstance(v, (str, int, float))}
            }
        )
        
        self.last_alert_times[rule.rule_id] = current_time
        
        self.logger.warning(
            f"Drift detected by rule {rule.name}: "
            f"score={drift_score:.3f}, threshold={rule.threshold}"
        )
    
    async def _emit_anomaly_alert(self, rule: MonitoringRule, anomaly_value: float, details: Dict[str, Any]):
        """Emit alert for anomaly detection"""
        current_time = time.time()
        
        if rule.rule_id in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[rule.rule_id]
            if time_since_last < rule.alert_cooldown_seconds:
                return
        
        alert_manager.ingest(
            metric=f"anomaly_detected_{rule.metric_name}",
            value=anomaly_value,
            labels={
                "rule_id": rule.rule_id,
                "metric_name": rule.metric_name,
                "severity": details.get("severity", "MEDIUM"),
                "deviation": str(details.get("deviation", 0))
            }
        )
        
        self.last_alert_times[rule.rule_id] = current_time
        
        self.logger.warning(
            f"Anomaly detected by rule {rule.name}: "
            f"value={anomaly_value}, expected={details.get('expected', 'N/A')}"
        )
    
    async def _emit_health_alert(self, rule: MonitoringRule, current_value: float, details: Dict[str, Any]):
        """Emit alert for system health issues"""
        current_time = time.time()
        
        if rule.rule_id in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[rule.rule_id]
            if time_since_last < rule.alert_cooldown_seconds:
                return
        
        alert_manager.ingest(
            metric=f"system_health_{rule.metric_name}",
            value=current_value,
            labels={
                "rule_id": rule.rule_id,
                "metric_name": rule.metric_name,
                "severity": details.get("severity", "MEDIUM"),
                "threshold": str(details.get("threshold", 0))
            }
        )
        
        self.last_alert_times[rule.rule_id] = current_time
        
        self.logger.warning(
            f"Health issue detected by rule {rule.name}: "
            f"current={current_value}, threshold={details.get('threshold', 'N/A')}"
        )
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get status of monitoring system"""
        return {
            "active_rules": len([r for r in self.monitoring_rules.values() if r.enabled]),
            "total_rules": len(self.monitoring_rules),
            "monitoring_tasks": len(self.monitoring_tasks),
            "metrics_tracked": len(self.metrics_data),
            "rules": {
                rule_id: {
                    "name": rule.name,
                    "type": rule.monitoring_type.value,
                    "enabled": rule.enabled,
                    "metric": rule.metric_name,
                    "last_alert": self.last_alert_times.get(rule_id)
                }
                for rule_id, rule in self.monitoring_rules.items()
            },
            "metric_data_points": {
                metric: len(data) for metric, data in self.metrics_data.items()
            }
        }
    
    def start_monitoring(self):
        """Start all monitoring tasks"""
        for rule_id, rule in self.monitoring_rules.items():
            if rule.enabled and rule_id not in self.monitoring_tasks:
                task = asyncio.create_task(self._monitoring_loop(rule))
                self.monitoring_tasks[rule_id] = task
        
        self.logger.info(f"Started monitoring with {len(self.monitoring_tasks)} active tasks")
    
    def stop_monitoring(self):
        """Stop all monitoring tasks"""
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        self.monitoring_tasks.clear()
        self.logger.info("Stopped all monitoring tasks")


# Create monitoring hooks for common metrics
class MonitoringHooks:
    """Convenience hooks for common monitoring scenarios"""
    
    def __init__(self, drift_monitor: DriftMonitor):
        self.drift_monitor = drift_monitor
    
    def track_model_accuracy(self, model_name: str, accuracy: float):
        """Track model accuracy for drift detection"""
        self.drift_monitor.record_metric(f"{model_name}_accuracy", accuracy)
    
    def track_response_time(self, endpoint: str, response_time_ms: float):
        """Track API response times"""
        self.drift_monitor.record_metric(f"{endpoint}_response_time", response_time_ms)
    
    def track_error_rate(self, service: str, error_rate: float):
        """Track service error rates"""
        self.drift_monitor.record_metric(f"{service}_error_rate", error_rate)
    
    def track_data_quality(self, data_source: str, quality_score: float):
        """Track data quality scores"""
        self.drift_monitor.record_metric(f"{data_source}_quality", quality_score)
    
    def track_forecast_error(self, model_name: str, mape: float):
        """Track forecast mean absolute percentage error"""
        self.drift_monitor.record_metric(f"{model_name}_forecast_mape", mape)
    
    def track_system_metric(self, metric_name: str, value: float):
        """Track general system metrics"""
        self.drift_monitor.record_metric(metric_name, value)


# Global instances
drift_monitor = DriftMonitor()
monitoring_hooks = MonitoringHooks(drift_monitor)