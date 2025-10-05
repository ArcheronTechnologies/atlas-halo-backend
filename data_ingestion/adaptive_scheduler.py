#!/usr/bin/env python3
"""
Adaptive Collection Scheduler for Atlas AI
Dynamically adjusts collection intervals based on crime patterns, data quality, and system load
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from pathlib import Path

try:
    from .data_quality_validator import EnhancedDataQualityValidator, QualityScore
except ImportError:
    from data_quality_validator import EnhancedDataQualityValidator, QualityScore

logger = logging.getLogger(__name__)


class CollectionPriority(Enum):
    """Collection priority levels"""
    CRITICAL = 1    # Every 5 minutes
    HIGH = 2        # Every 15 minutes  
    NORMAL = 3      # Every 30 minutes
    LOW = 4         # Every 2 hours
    MINIMAL = 5     # Every 6 hours


@dataclass
class CollectionMetrics:
    """Metrics for adaptive scheduling decisions"""
    recent_incident_count: int
    average_quality_score: float
    error_rate: float
    duplicate_rate: float
    high_severity_count: int
    collection_success_rate: float
    data_freshness_hours: float
    peak_crime_time: bool
    weekend: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScheduleDecision:
    """Scheduling decision with reasoning"""
    next_interval_minutes: int
    priority_level: CollectionPriority
    reasoning: List[str]
    confidence: float
    adjustments_made: List[str]


class AdaptiveCollectionScheduler:
    """
    Adaptive scheduler that adjusts collection intervals based on:
    - Crime incident volume and severity
    - Data quality trends
    - Time-based patterns (peak crime hours, weekends)
    - System performance and error rates
    - Resource availability
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("data_lake/scheduler_config.json")
        self.metrics_file = Path("data_lake/collection_metrics.json") 
        self.validator = EnhancedDataQualityValidator()
        
        # Load configuration
        self.config = self._load_config()
        
        # Metrics tracking
        self.collection_history = []
        self.error_history = []
        self.quality_history = []
        
        # Peak crime time patterns (based on research on Swedish crime patterns)
        self.peak_crime_hours = {
            'weekday': [(16, 19), (21, 24)],  # Evening rush and late evening
            'weekend': [(14, 17), (20, 26)]   # Afternoon and late night (26 = 2 AM next day)
        }
        
        # Base intervals for different priorities (in minutes)
        self.base_intervals = {
            CollectionPriority.CRITICAL: 5,
            CollectionPriority.HIGH: 15,
            CollectionPriority.NORMAL: 30,
            CollectionPriority.LOW: 120,
            CollectionPriority.MINIMAL: 360
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load scheduler configuration"""
        
        default_config = {
            'min_interval_minutes': 5,
            'max_interval_minutes': 360,
            'quality_threshold_urgent': 2.0,
            'incident_spike_threshold': 1.5,  # 1.5x normal rate
            'error_rate_threshold': 0.15,
            'peak_hour_multiplier': 1.5,
            'weekend_multiplier': 1.2,
            'high_severity_threshold': 4,
            'duplicate_rate_threshold': 0.3,
            'enable_predictive_scaling': True,
            'resource_usage_threshold': 0.8
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def record_collection_result(self, 
                               incidents_collected: int,
                               quality_assessments: List[Any],
                               collection_time_seconds: float,
                               errors: List[str] = None):
        """Record results of a collection run for adaptive learning"""
        
        timestamp = datetime.utcnow()
        errors = errors or []
        
        # Calculate quality metrics
        if quality_assessments:
            avg_quality = statistics.mean(a.score for a in quality_assessments)
            high_quality_count = sum(1 for a in quality_assessments 
                                   if a.level in [QualityScore.EXCELLENT, QualityScore.GOOD])
            duplicate_count = sum(1 for a in quality_assessments if a.duplicate_risk > 0.7)
        else:
            avg_quality = 0.0
            high_quality_count = 0
            duplicate_count = 0
        
        collection_record = {
            'timestamp': timestamp.isoformat(),
            'incidents_collected': incidents_collected,
            'average_quality': avg_quality,
            'high_quality_count': high_quality_count,
            'duplicate_count': duplicate_count,
            'collection_time_seconds': collection_time_seconds,
            'error_count': len(errors),
            'errors': errors,
            'success': len(errors) == 0
        }
        
        self.collection_history.append(collection_record)
        
        # Keep only last 100 records
        if len(self.collection_history) > 100:
            self.collection_history = self.collection_history[-100:]
        
        # Save metrics
        self._save_metrics()
    
    def calculate_collection_metrics(self, hours_back: int = 4) -> CollectionMetrics:
        """Calculate current collection metrics for scheduling decisions"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_collections = [
            record for record in self.collection_history
            if datetime.fromisoformat(record['timestamp']) >= cutoff_time
        ]
        
        if not recent_collections:
            # No recent data - use conservative defaults
            return CollectionMetrics(
                recent_incident_count=0,
                average_quality_score=3.0,
                error_rate=0.0,
                duplicate_rate=0.0,
                high_severity_count=0,
                collection_success_rate=1.0,
                data_freshness_hours=24.0,
                peak_crime_time=self._is_peak_crime_time(),
                weekend=datetime.utcnow().weekday() >= 5
            )
        
        # Calculate metrics from recent collections
        total_incidents = sum(r['incidents_collected'] for r in recent_collections)
        avg_quality = statistics.mean(r['average_quality'] for r in recent_collections)
        error_rate = sum(r['error_count'] for r in recent_collections) / len(recent_collections)
        success_rate = sum(1 for r in recent_collections if r['success']) / len(recent_collections)
        
        # Calculate duplicate rate
        total_quality_incidents = sum(r['high_quality_count'] for r in recent_collections)
        total_duplicates = sum(r['duplicate_count'] for r in recent_collections)
        duplicate_rate = total_duplicates / max(1, total_incidents)
        
        # Estimate high severity count (assume 20% of incidents are high severity)
        high_severity_count = int(total_incidents * 0.2)
        
        # Data freshness
        if recent_collections:
            latest_collection = max(recent_collections, 
                                  key=lambda x: datetime.fromisoformat(x['timestamp']))
            data_freshness = (datetime.utcnow() - 
                            datetime.fromisoformat(latest_collection['timestamp'])).total_seconds() / 3600
        else:
            data_freshness = 24.0
        
        return CollectionMetrics(
            recent_incident_count=total_incidents,
            average_quality_score=avg_quality,
            error_rate=error_rate,
            duplicate_rate=duplicate_rate,
            high_severity_count=high_severity_count,
            collection_success_rate=success_rate,
            data_freshness_hours=data_freshness,
            peak_crime_time=self._is_peak_crime_time(),
            weekend=datetime.utcnow().weekday() >= 5
        )
    
    def decide_next_interval(self, current_metrics: CollectionMetrics = None) -> ScheduleDecision:
        """Decide the next collection interval based on current conditions"""
        
        if current_metrics is None:
            current_metrics = self.calculate_collection_metrics()
        
        reasoning = []
        adjustments = []
        priority = CollectionPriority.NORMAL
        
        # Start with normal interval
        interval_minutes = self.base_intervals[CollectionPriority.NORMAL]
        
        # Factor 1: Incident volume
        if current_metrics.recent_incident_count > 50:  # High volume
            priority = CollectionPriority.HIGH
            interval_minutes = self.base_intervals[CollectionPriority.HIGH]
            reasoning.append("High incident volume detected")
            adjustments.append("Increased frequency due to incident volume")
        
        elif current_metrics.recent_incident_count > 100:  # Very high volume
            priority = CollectionPriority.CRITICAL
            interval_minutes = self.base_intervals[CollectionPriority.CRITICAL]
            reasoning.append("Critical incident volume - very high activity")
            adjustments.append("Maximum frequency due to critical volume")
        
        # Factor 2: Data quality issues
        if current_metrics.average_quality_score < self.config['quality_threshold_urgent']:
            if priority.value > CollectionPriority.HIGH.value:
                priority = CollectionPriority.HIGH
                interval_minutes = self.base_intervals[CollectionPriority.HIGH]
            reasoning.append("Poor data quality requires more frequent validation")
            adjustments.append("Increased frequency for quality improvement")
        
        # Factor 3: Error rate
        if current_metrics.error_rate > self.config['error_rate_threshold']:
            # High errors - reduce frequency to avoid overloading
            if priority.value < CollectionPriority.LOW.value:
                priority = CollectionPriority.LOW
                interval_minutes = self.base_intervals[CollectionPriority.LOW]
            reasoning.append("High error rate - reducing collection frequency")
            adjustments.append("Reduced frequency due to errors")
        
        # Factor 4: Peak crime times
        if current_metrics.peak_crime_time:
            interval_minutes = int(interval_minutes / self.config['peak_hour_multiplier'])
            reasoning.append("Peak crime time - increased monitoring")
            adjustments.append("Reduced interval during peak crime hours")
        
        # Factor 5: Weekend patterns
        if current_metrics.weekend:
            interval_minutes = int(interval_minutes / self.config['weekend_multiplier'])
            reasoning.append("Weekend pattern - higher crime likelihood")
            adjustments.append("Reduced interval for weekend monitoring")
        
        # Factor 6: High severity incidents
        if current_metrics.high_severity_count > 5:
            if priority.value > CollectionPriority.HIGH.value:
                priority = CollectionPriority.HIGH
                interval_minutes = self.base_intervals[CollectionPriority.HIGH]
            reasoning.append("High severity incidents require close monitoring")
            adjustments.append("Increased frequency due to severity")
        
        # Factor 7: Data freshness
        if current_metrics.data_freshness_hours > 2:
            interval_minutes = max(5, int(interval_minutes * 0.7))  # Reduce interval
            reasoning.append("Stale data detected - need refresh")
            adjustments.append("Reduced interval to refresh data")
        
        # Factor 8: Duplicate rate
        if current_metrics.duplicate_rate > self.config['duplicate_rate_threshold']:
            interval_minutes = int(interval_minutes * 1.3)  # Slight increase
            reasoning.append("High duplicate rate - spacing out collections")
            adjustments.append("Increased interval to reduce duplicates")
        
        # Apply constraints
        interval_minutes = max(self.config['min_interval_minutes'], 
                             min(self.config['max_interval_minutes'], interval_minutes))
        
        # Calculate confidence based on data quality and history
        confidence = min(1.0, current_metrics.collection_success_rate + 
                        (current_metrics.average_quality_score / 5.0)) / 2.0
        
        return ScheduleDecision(
            next_interval_minutes=interval_minutes,
            priority_level=priority,
            reasoning=reasoning,
            confidence=confidence,
            adjustments_made=adjustments
        )
    
    def _is_peak_crime_time(self) -> bool:
        """Check if current time is during peak crime hours"""
        
        now = datetime.utcnow()
        current_hour = now.hour
        is_weekend = now.weekday() >= 5
        
        peak_hours = self.peak_crime_hours['weekend' if is_weekend else 'weekday']
        
        for start_hour, end_hour in peak_hours:
            if end_hour > 24:  # Handle overnight periods
                if current_hour >= start_hour or current_hour <= (end_hour - 24):
                    return True
            else:
                if start_hour <= current_hour < end_hour:
                    return True
        
        return False
    
    def _save_metrics(self):
        """Save collection metrics to file"""
        
        try:
            metrics_data = {
                'collection_history': self.collection_history[-50:],  # Keep last 50
                'last_updated': datetime.utcnow().isoformat(),
                'scheduler_config': self.config
            }
            
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def get_scheduling_recommendations(self) -> Dict[str, Any]:
        """Get current scheduling recommendations and analysis"""
        
        metrics = self.calculate_collection_metrics()
        decision = self.decide_next_interval(metrics)
        
        # Calculate trends
        recent_quality_trend = self._calculate_quality_trend()
        recent_volume_trend = self._calculate_volume_trend()
        
        return {
            'current_metrics': metrics.to_dict(),
            'scheduling_decision': {
                'next_interval_minutes': decision.next_interval_minutes,
                'priority_level': decision.priority_level.name,
                'reasoning': decision.reasoning,
                'confidence': decision.confidence,
                'adjustments': decision.adjustments_made
            },
            'trends': {
                'quality_trend': recent_quality_trend,
                'volume_trend': recent_volume_trend,
                'is_peak_time': self._is_peak_crime_time()
            },
            'recommendations': self._generate_recommendations(metrics, decision)
        }
    
    def _calculate_quality_trend(self) -> str:
        """Calculate recent quality trend"""
        
        if len(self.collection_history) < 5:
            return "insufficient_data"
        
        recent_quality = [r['average_quality'] for r in self.collection_history[-5:]]
        earlier_quality = [r['average_quality'] for r in self.collection_history[-10:-5]]
        
        if not earlier_quality:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent_quality)
        earlier_avg = statistics.mean(earlier_quality)
        
        if recent_avg > earlier_avg + 0.3:
            return "improving"
        elif recent_avg < earlier_avg - 0.3:
            return "declining"
        else:
            return "stable"
    
    def _calculate_volume_trend(self) -> str:
        """Calculate recent volume trend"""
        
        if len(self.collection_history) < 5:
            return "insufficient_data"
        
        recent_volume = [r['incidents_collected'] for r in self.collection_history[-5:]]
        earlier_volume = [r['incidents_collected'] for r in self.collection_history[-10:-5]]
        
        if not earlier_volume:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent_volume)
        earlier_avg = statistics.mean(earlier_volume)
        
        if recent_avg > earlier_avg * 1.2:
            return "increasing"
        elif recent_avg < earlier_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_recommendations(self, 
                                metrics: CollectionMetrics, 
                                decision: ScheduleDecision) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if metrics.error_rate > 0.1:
            recommendations.append("Consider investigating error sources to improve collection reliability")
        
        if metrics.average_quality_score < 3.0:
            recommendations.append("Review data sources - quality issues detected")
        
        if metrics.duplicate_rate > 0.2:
            recommendations.append("Enhance duplicate detection algorithms")
        
        if metrics.data_freshness_hours > 4:
            recommendations.append("Consider more frequent collection during active periods")
        
        if decision.priority_level == CollectionPriority.CRITICAL:
            recommendations.append("Monitor system resources during high-frequency collection")
        
        return recommendations


# Example usage and testing
async def main():
    """Test the adaptive scheduler"""
    
    logging.basicConfig(level=logging.INFO)
    
    scheduler = AdaptiveCollectionScheduler()
    
    # Simulate some collection results
    from .data_quality_validator import QualityAssessment, QualityScore
    
    # Simulate different scenarios
    test_assessments = [
        QualityAssessment(4.2, QualityScore.GOOD, [], ["Complete data"], 0.8, 0.1, 4.5, 4.0),
        QualityAssessment(3.8, QualityScore.FAIR, ["Minor issues"], ["Swedish content"], 0.7, 0.2, 4.0, 3.5),
        QualityAssessment(4.6, QualityScore.EXCELLENT, [], ["High quality"], 0.9, 0.05, 4.8, 4.5)
    ]
    
    # Record a normal collection
    scheduler.record_collection_result(
        incidents_collected=45,
        quality_assessments=test_assessments,
        collection_time_seconds=12.5,
        errors=[]
    )
    
    # Get recommendations
    recommendations = scheduler.get_scheduling_recommendations()
    
    print("=== Adaptive Collection Scheduler Analysis ===")
    print(f"Next interval: {recommendations['scheduling_decision']['next_interval_minutes']} minutes")
    print(f"Priority: {recommendations['scheduling_decision']['priority_level']}")
    print(f"Confidence: {recommendations['scheduling_decision']['confidence']:.2f}")
    print("\nReasoning:")
    for reason in recommendations['scheduling_decision']['reasoning']:
        print(f"  • {reason}")
    
    print("\nRecommendations:")
    for rec in recommendations['recommendations']:
        print(f"  • {rec}")


if __name__ == "__main__":
    asyncio.run(main())