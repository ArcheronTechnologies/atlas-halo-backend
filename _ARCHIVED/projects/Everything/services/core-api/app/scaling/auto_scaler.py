"""
Automatic Scaling and Load Management System

This module provides intelligent auto-scaling capabilities based on real-time metrics,
predictive analysis, and workload patterns for the SCIP platform.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import json
import statistics
from ..monitoring.performance import performance_monitor
from ..cache.redis_cache import cache
from ..db.optimization import db_optimizer

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    cpu_percent: float
    memory_percent: float
    request_rate: float
    response_time_p95: float
    error_rate: float
    queue_depth: int
    active_connections: int
    timestamp: datetime


@dataclass
class ScalingDecision:
    """Auto-scaling decision with reasoning"""
    action: str  # scale_up, scale_down, maintain
    component: str  # api, worker, database, cache
    current_instances: int
    target_instances: int
    confidence: float
    reasoning: str
    metrics_snapshot: ScalingMetrics
    estimated_cost_impact: float


class PredictiveAnalyzer:
    """Predictive analysis for proactive scaling"""
    
    def __init__(self):
        self.historical_patterns = {}
        self.seasonal_patterns = {}
        
    async def analyze_workload_patterns(self, hours_back: int = 168) -> Dict[str, Any]:
        """Analyze workload patterns over time (default: 1 week)"""
        try:
            # Get historical performance data
            trends = performance_monitor.get_performance_trends(hours=hours_back)
            
            if not trends.get('request_trends'):
                return {'patterns': [], 'predictions': []}
            
            # Analyze daily patterns
            daily_patterns = self._analyze_daily_patterns(hours_back)
            
            # Analyze weekly patterns  
            weekly_patterns = self._analyze_weekly_patterns(hours_back)
            
            # Generate predictions for next 24 hours
            predictions = self._predict_next_24h(daily_patterns, weekly_patterns)
            
            return {
                'daily_patterns': daily_patterns,
                'weekly_patterns': weekly_patterns,
                'predictions': predictions,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing workload patterns: {e}")
            return {'patterns': [], 'predictions': []}
    
    def _analyze_daily_patterns(self, hours_back: int) -> Dict[str, Any]:
        """Analyze daily traffic patterns"""
        # This would analyze historical request patterns by hour of day
        # For now, return typical business hours pattern
        return {
            'peak_hours': [9, 10, 11, 14, 15, 16],  # Business hours
            'low_hours': [0, 1, 2, 3, 4, 5, 22, 23],  # Night hours
            'average_peak_multiplier': 2.5,  # Peak is 2.5x average
            'average_low_multiplier': 0.3    # Low is 0.3x average
        }
    
    def _analyze_weekly_patterns(self, hours_back: int) -> Dict[str, Any]:
        """Analyze weekly traffic patterns"""
        return {
            'peak_days': [1, 2, 3, 4],  # Monday-Thursday
            'low_days': [5, 6, 0],      # Friday, Saturday, Sunday
            'weekend_multiplier': 0.4   # Weekend is 40% of weekday traffic
        }
    
    def _predict_next_24h(self, daily_patterns: Dict, weekly_patterns: Dict) -> List[Dict]:
        """Predict traffic for next 24 hours"""
        predictions = []
        now = datetime.now(timezone.utc)
        
        for hour in range(24):
            future_time = now + timedelta(hours=hour)
            hour_of_day = future_time.hour
            day_of_week = future_time.weekday()
            
            # Base load prediction
            base_multiplier = 1.0
            
            # Apply daily pattern
            if hour_of_day in daily_patterns['peak_hours']:
                base_multiplier *= daily_patterns['average_peak_multiplier']
            elif hour_of_day in daily_patterns['low_hours']:
                base_multiplier *= daily_patterns['average_low_multiplier']
            
            # Apply weekly pattern
            if day_of_week in weekly_patterns['low_days']:
                base_multiplier *= weekly_patterns['weekend_multiplier']
            
            predictions.append({
                'hour_offset': hour,
                'timestamp': future_time.isoformat(),
                'predicted_load_multiplier': base_multiplier,
                'confidence': 0.8 if hour < 12 else 0.6  # Higher confidence for near-term
            })
        
        return predictions


class AutoScaler:
    """Intelligent auto-scaling system"""
    
    def __init__(self):
        self.predictor = PredictiveAnalyzer()
        self.scaling_history = []
        
        # Scaling thresholds
        self.thresholds = {
            'cpu_scale_up': 70.0,      # Scale up at 70% CPU
            'cpu_scale_down': 30.0,    # Scale down at 30% CPU
            'memory_scale_up': 80.0,   # Scale up at 80% memory
            'response_time_scale_up': 2.0,  # Scale up if p95 > 2s
            'error_rate_scale_up': 5.0,     # Scale up if error rate > 5%
            'min_instances': 2,        # Minimum instances
            'max_instances': 20,       # Maximum instances
            'cooldown_minutes': 10     # Wait time between scaling actions
        }
        
        # Current instance counts (would be retrieved from orchestrator)
        self.current_instances = {
            'api': 3,
            'worker': 2,
            'database': 1,
            'cache': 2
        }
    
    async def evaluate_scaling_needs(self) -> List[ScalingDecision]:
        """Evaluate current metrics and determine scaling needs"""
        try:
            decisions = []
            
            # Get current metrics
            health = performance_monitor.get_system_health()
            endpoint_stats = performance_monitor.get_endpoint_statistics(hours=1)
            
            # Calculate aggregated metrics
            metrics = self._calculate_scaling_metrics(health, endpoint_stats)
            
            # Evaluate API server scaling
            api_decision = await self._evaluate_api_scaling(metrics)
            if api_decision:
                decisions.append(api_decision)
            
            # Evaluate worker scaling
            worker_decision = await self._evaluate_worker_scaling(metrics)
            if worker_decision:
                decisions.append(worker_decision)
            
            # Evaluate database scaling
            db_decision = await self._evaluate_database_scaling(metrics)
            if db_decision:
                decisions.append(db_decision)
            
            # Evaluate cache scaling
            cache_decision = await self._evaluate_cache_scaling(metrics)
            if cache_decision:
                decisions.append(cache_decision)
            
            # Store decisions in history
            for decision in decisions:
                self.scaling_history.append({
                    'timestamp': datetime.now(timezone.utc),
                    'decision': decision
                })
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error evaluating scaling needs: {e}")
            return []
    
    def _calculate_scaling_metrics(self, health: Dict, endpoint_stats: List) -> ScalingMetrics:
        """Calculate metrics used for scaling decisions"""
        
        # Calculate average response time and error rate
        if endpoint_stats:
            avg_response_time = statistics.mean(stat.avg_duration for stat in endpoint_stats)
            p95_response_time = statistics.mean(stat.p95_duration for stat in endpoint_stats)
            total_requests = sum(stat.total_requests for stat in endpoint_stats)
            total_errors = sum(stat.error_count for stat in endpoint_stats)
            error_rate = (total_errors / max(total_requests, 1)) * 100
            request_rate = total_requests / 3600  # Requests per second (assuming 1-hour window)
        else:
            avg_response_time = 0.0
            p95_response_time = 0.0
            error_rate = 0.0
            request_rate = 0.0
        
        return ScalingMetrics(
            cpu_percent=health.get('cpu_percent', 0.0),
            memory_percent=health.get('memory_percent', 0.0),
            request_rate=request_rate,
            response_time_p95=p95_response_time,
            error_rate=error_rate,
            queue_depth=0,  # Would be retrieved from actual queue
            active_connections=health.get('active_connections', 0),
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _evaluate_api_scaling(self, metrics: ScalingMetrics) -> Optional[ScalingDecision]:
        """Evaluate API server scaling needs"""
        current = self.current_instances['api']
        target = current
        action = 'maintain'
        reasoning = 'Metrics within normal range'
        confidence = 0.8
        
        # Scale up conditions
        scale_up_reasons = []
        if metrics.cpu_percent > self.thresholds['cpu_scale_up']:
            scale_up_reasons.append(f'High CPU: {metrics.cpu_percent:.1f}%')
        
        if metrics.memory_percent > self.thresholds['memory_scale_up']:
            scale_up_reasons.append(f'High memory: {metrics.memory_percent:.1f}%')
        
        if metrics.response_time_p95 > self.thresholds['response_time_scale_up']:
            scale_up_reasons.append(f'Slow response: {metrics.response_time_p95:.2f}s')
        
        if metrics.error_rate > self.thresholds['error_rate_scale_up']:
            scale_up_reasons.append(f'High error rate: {metrics.error_rate:.1f}%')
        
        if scale_up_reasons and current < self.thresholds['max_instances']:
            target = min(current + 1, self.thresholds['max_instances'])
            action = 'scale_up'
            reasoning = '; '.join(scale_up_reasons)
            confidence = 0.9
        
        # Scale down conditions
        elif (metrics.cpu_percent < self.thresholds['cpu_scale_down'] and 
              metrics.memory_percent < 50.0 and 
              metrics.response_time_p95 < 1.0 and 
              metrics.error_rate < 2.0 and
              current > self.thresholds['min_instances']):
            
            target = max(current - 1, self.thresholds['min_instances'])
            action = 'scale_down'
            reasoning = 'Low resource utilization and good performance'
            confidence = 0.7
        
        if action != 'maintain':
            return ScalingDecision(
                action=action,
                component='api',
                current_instances=current,
                target_instances=target,
                confidence=confidence,
                reasoning=reasoning,
                metrics_snapshot=metrics,
                estimated_cost_impact=self._estimate_cost_impact('api', current, target)
            )
        
        return None
    
    async def _evaluate_worker_scaling(self, metrics: ScalingMetrics) -> Optional[ScalingDecision]:
        """Evaluate background worker scaling needs"""
        current = self.current_instances['worker']
        
        # Worker scaling is primarily based on queue depth
        # For now, use CPU and memory as proxy metrics
        if metrics.cpu_percent > 80.0 and current < 8:
            return ScalingDecision(
                action='scale_up',
                component='worker',
                current_instances=current,
                target_instances=current + 1,
                confidence=0.8,
                reasoning='High CPU usage indicates worker bottleneck',
                metrics_snapshot=metrics,
                estimated_cost_impact=self._estimate_cost_impact('worker', current, current + 1)
            )
        
        return None
    
    async def _evaluate_database_scaling(self, metrics: ScalingMetrics) -> Optional[ScalingDecision]:
        """Evaluate database scaling needs"""
        # Database scaling is more complex and typically involves read replicas
        # For now, focus on identifying when database optimization is needed
        
        if metrics.response_time_p95 > 3.0:
            # Check if database optimization would help
            try:
                from sqlalchemy.orm import Session
                from ..db.session import get_session
                
                async with get_session() as session:
                    optimizations = await db_optimizer.optimize_queries(session)
                    
                    if optimizations['vacuum_needed'] or optimizations['analyze_needed']:
                        return ScalingDecision(
                            action='optimize',
                            component='database',
                            current_instances=1,
                            target_instances=1,
                            confidence=0.9,
                            reasoning='Database optimization needed to improve performance',
                            metrics_snapshot=metrics,
                            estimated_cost_impact=0.0  # Optimization has no direct cost
                        )
            except Exception as e:
                logger.error(f"Error evaluating database optimization: {e}")
        
        return None
    
    async def _evaluate_cache_scaling(self, metrics: ScalingMetrics) -> Optional[ScalingDecision]:
        """Evaluate cache scaling needs"""
        try:
            cache_stats = await cache.get_stats()
            hit_rate = cache_stats.get('hit_rate', 0.0)
            
            if hit_rate < 80.0 and metrics.response_time_p95 > 1.5:
                return ScalingDecision(
                    action='optimize',
                    component='cache',
                    current_instances=self.current_instances['cache'],
                    target_instances=self.current_instances['cache'],
                    confidence=0.8,
                    reasoning=f'Low cache hit rate ({hit_rate:.1f}%) affecting performance',
                    metrics_snapshot=metrics,
                    estimated_cost_impact=0.0
                )
        except Exception as e:
            logger.error(f"Error evaluating cache scaling: {e}")
        
        return None
    
    def _estimate_cost_impact(self, component: str, current: int, target: int) -> float:
        """Estimate monthly cost impact of scaling decision"""
        # Simplified cost estimation (would be more sophisticated in production)
        cost_per_instance = {
            'api': 150.0,      # $150/month per API instance
            'worker': 100.0,   # $100/month per worker instance
            'database': 500.0, # $500/month per DB instance
            'cache': 200.0     # $200/month per cache instance
        }
        
        base_cost = cost_per_instance.get(component, 100.0)
        return (target - current) * base_cost
    
    async def execute_scaling_decisions(self, decisions: List[ScalingDecision]) -> Dict[str, Any]:
        """Execute approved scaling decisions"""
        results = {
            'executed': [],
            'failed': [],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        for decision in decisions:
            try:
                if decision.action == 'optimize':
                    # Execute optimization tasks
                    success = await self._execute_optimization(decision.component)
                    if success:
                        results['executed'].append({
                            'component': decision.component,
                            'action': decision.action,
                            'reasoning': decision.reasoning
                        })
                    else:
                        results['failed'].append({
                            'component': decision.component,
                            'action': decision.action,
                            'error': 'Optimization failed'
                        })
                
                elif decision.action in ['scale_up', 'scale_down']:
                    # In production, this would call Kubernetes/Docker Swarm/ECS APIs
                    logger.info(f"Would execute scaling: {decision.component} from {decision.current_instances} to {decision.target_instances}")
                    
                    # Update internal tracking
                    self.current_instances[decision.component] = decision.target_instances
                    
                    results['executed'].append({
                        'component': decision.component,
                        'action': decision.action,
                        'from_instances': decision.current_instances,
                        'to_instances': decision.target_instances,
                        'reasoning': decision.reasoning,
                        'estimated_cost': decision.estimated_cost_impact
                    })
                
            except Exception as e:
                logger.error(f"Error executing scaling decision for {decision.component}: {e}")
                results['failed'].append({
                    'component': decision.component,
                    'action': decision.action,
                    'error': str(e)
                })
        
        return results
    
    async def _execute_optimization(self, component: str) -> bool:
        """Execute optimization tasks"""
        try:
            if component == 'database':
                from sqlalchemy.orm import Session
                from ..db.session import get_session
                
                async with get_session() as session:
                    results = await db_optimizer.run_maintenance_tasks(session)
                    return len(results.get('errors', [])) == 0
            
            elif component == 'cache':
                # Clear expired keys, optimize memory usage
                await cache.connect()
                # Would implement cache optimization logic
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing {component} optimization: {e}")
            return False
    
    async def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on predictive analysis"""
        try:
            # Get workload predictions
            patterns = await self.predictor.analyze_workload_patterns(hours_back=168)
            
            # Get current scaling decisions
            current_decisions = await self.evaluate_scaling_needs()
            
            # Generate proactive recommendations
            recommendations = []
            
            if patterns.get('predictions'):
                for prediction in patterns['predictions'][:6]:  # Next 6 hours
                    if prediction['predicted_load_multiplier'] > 2.0:
                        recommendations.append({
                            'type': 'proactive_scale_up',
                            'time_offset_hours': prediction['hour_offset'],
                            'predicted_load': prediction['predicted_load_multiplier'],
                            'recommendation': 'Consider scaling up API servers before peak load',
                            'confidence': prediction['confidence']
                        })
                    elif prediction['predicted_load_multiplier'] < 0.5:
                        recommendations.append({
                            'type': 'proactive_scale_down',
                            'time_offset_hours': prediction['hour_offset'],
                            'predicted_load': prediction['predicted_load_multiplier'],
                            'recommendation': 'Consider scaling down during low usage',
                            'confidence': prediction['confidence']
                        })
            
            return {
                'current_decisions': [decision.__dict__ for decision in current_decisions],
                'proactive_recommendations': recommendations,
                'workload_patterns': patterns,
                'current_instances': self.current_instances,
                'estimated_monthly_cost': self._calculate_monthly_cost(),
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating scaling recommendations: {e}")
            return {'error': str(e)}
    
    def _calculate_monthly_cost(self) -> float:
        """Calculate current monthly infrastructure cost"""
        costs = {
            'api': 150.0 * self.current_instances['api'],
            'worker': 100.0 * self.current_instances['worker'], 
            'database': 500.0 * self.current_instances['database'],
            'cache': 200.0 * self.current_instances['cache']
        }
        return sum(costs.values())


# Global auto-scaler instance
auto_scaler = AutoScaler()


async def run_scaling_evaluation() -> Dict[str, Any]:
    """Run complete scaling evaluation and return recommendations"""
    try:
        decisions = await auto_scaler.evaluate_scaling_needs()
        recommendations = await auto_scaler.get_scaling_recommendations()
        
        return {
            'scaling_decisions': [decision.__dict__ for decision in decisions],
            'recommendations': recommendations,
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in scaling evaluation: {e}")
        return {'error': str(e)}


async def execute_auto_scaling() -> Dict[str, Any]:
    """Execute automatic scaling based on current metrics"""
    try:
        decisions = await auto_scaler.evaluate_scaling_needs()
        
        # Filter for high-confidence decisions
        high_confidence_decisions = [d for d in decisions if d.confidence >= 0.8]
        
        if high_confidence_decisions:
            results = await auto_scaler.execute_scaling_decisions(high_confidence_decisions)
            logger.info(f"Auto-scaling executed: {len(results['executed'])} actions taken")
            return results
        else:
            return {
                'executed': [],
                'message': 'No high-confidence scaling decisions found',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error in auto-scaling execution: {e}")
        return {'error': str(e)}