"""
Predictions API endpoints for Levi frontend integration.
Provides real-time crime prediction data and explanations.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import asdict
import random
import math

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Import prediction modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from prediction.ensemble_models import create_crime_prediction_ensemble, EnsemblePrediction
from prediction.explainable_ai import create_prediction_explainer, generate_explanation_report
from analytics.forecasting import moving_average_forecast, exp_smoothing_forecast


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend


class PredictionsAPI:
    """API handler for crime predictions and explanations"""
    
    def __init__(self):
        self.ensemble = create_crime_prediction_ensemble()
        self.explainer = create_prediction_explainer(self.ensemble)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def generate_sample_series(self, hours: int = 168) -> List[tuple]:
        """Generate sample time series data for demo purposes"""
        series = []
        base_time = datetime.now().timestamp()
        
        for i in range(hours):
            timestamp = base_time - (hours - i) * 3600  # Hourly data going backwards
            
            # Simulate realistic crime patterns
            hour_of_day = i % 24
            day_of_week = (i // 24) % 7
            
            # Base level with daily and weekly patterns
            base_value = 8.0
            
            # Higher crime at night (peak around 23:00)
            daily_pattern = 2.0 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)
            
            # Higher crime on weekends
            weekend_boost = 1.5 if day_of_week >= 5 else 0
            
            # Add some trend and noise
            trend = 0.01 * i  # Slight upward trend
            noise = random.gauss(0, 0.5)
            
            value = max(0.1, base_value + daily_pattern + weekend_boost + trend + noise)
            series.append((timestamp, value))
        
        return series
    
    def get_cached_or_compute(self, cache_key: str, compute_func, ttl: int = None) -> Any:
        """Get cached result or compute and cache new result"""
        current_time = datetime.now().timestamp()
        cache_ttl = ttl or self.cache_ttl
        
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if current_time - cached_time < cache_ttl:
                return cached_data
        
        # Compute new result
        result = compute_func()
        self.cache[cache_key] = (result, current_time)
        return result
    
    def get_current_prediction(self) -> Dict[str, Any]:
        """Get current prediction with ensemble analysis"""
        
        def compute_prediction():
            # Generate recent data
            series = self.generate_sample_series(hours=168)  # 1 week of data
            
            # Get ensemble prediction
            prediction = self.ensemble.predict(series)
            
            # Get explanation
            context = {
                'geospatial': {
                    'is_hotspot': prediction.prediction > 12.0,
                    'cluster_density': 0.6 + 0.3 * (prediction.prediction / 15.0),
                    'area_classification': 'urban_high_activity'
                }
            }
            
            explanation = self.explainer.explain_prediction(series, prediction, context)
            
            return {
                'prediction': {
                    'value': prediction.prediction,
                    'confidence': prediction.confidence,
                    'variance': prediction.variance,
                    'trend': self._calculate_recent_trend(series),
                    'model_agreement': prediction.meta_features.get('model_agreement', 0.0),
                    'timestamp': prediction.timestamp.isoformat()
                },
                'explanation': {
                    'confidence_level': explanation.confidence_level,
                    'primary_factors': [
                        {
                            'name': factor.feature_name,
                            'score': factor.importance_score,
                            'contribution': factor.contribution,
                            'description': factor.description
                        }
                        for factor in explanation.primary_factors[:5]
                    ],
                    'risk_factors': explanation.risk_factors,
                    'mitigating_factors': explanation.mitigating_factors,
                    'recommendations': explanation.recommendations,
                    'summary': explanation.explanation_summary
                },
                'models': {
                    'individual_predictions': prediction.individual_predictions,
                    'weights': prediction.model_weights,
                    'models_used': prediction.models_used
                },
                'stats': {
                    'daily_average': sum(v for _, v in series[-24:]) / 24,
                    'weekly_trend': self._calculate_weekly_trend(series),
                    'peak_hour': self._find_peak_hour(series),
                    'data_quality': explanation.certainty_indicators.get('data_quality', 0.0)
                }
            }
        
        return self.get_cached_or_compute('current_prediction', compute_prediction, ttl=120)
    
    def get_prediction_timeline(self, period: str = '24h') -> Dict[str, Any]:
        """Get prediction timeline for specified period"""
        
        def compute_timeline():
            # Generate historical data
            historical_series = self.generate_sample_series(hours=168)
            
            # Generate future predictions based on period
            periods = {
                '1h': {'steps': 12, 'interval': 5},    # 5-minute intervals for 1 hour
                '24h': {'steps': 24, 'interval': 60},   # 1-hour intervals for 24 hours
                '7d': {'steps': 7, 'interval': 1440},   # Daily for 7 days
                '30d': {'steps': 30, 'interval': 1440}  # Daily for 30 days
            }
            
            config = periods.get(period, periods['24h'])
            timeline = []
            
            base_time = datetime.now()
            
            for i in range(config['steps']):
                # Use sliding window for prediction
                window_series = historical_series[-(168-i):] if i < 168 else historical_series
                
                prediction = self.ensemble.predict(window_series)
                
                timeline.append({
                    'time': (base_time + timedelta(minutes=i * config['interval'])).isoformat(),
                    'prediction': prediction.prediction,
                    'confidence': prediction.confidence,
                    'lower_bound': prediction.prediction - 2 * math.sqrt(prediction.variance),
                    'upper_bound': prediction.prediction + 2 * math.sqrt(prediction.variance)
                })
            
            return {
                'period': period,
                'timeline': timeline,
                'generated_at': datetime.now().isoformat()
            }
        
        cache_key = f'timeline_{period}'
        return self.get_cached_or_compute(cache_key, compute_timeline, ttl=180)
    
    def get_hotspot_data(self) -> Dict[str, Any]:
        """Get hotspot prediction data for map visualization"""
        
        def compute_hotspots():
            # Swedish cities/areas with coordinates
            locations = [
                {'name': 'Stockholm Central', 'lat': 59.3293, 'lng': 18.0686},
                {'name': 'Göteborg', 'lat': 57.7089, 'lng': 11.9746},
                {'name': 'Malmö', 'lat': 55.6050, 'lng': 13.0038},
                {'name': 'Uppsala', 'lat': 59.8586, 'lng': 17.6389},
                {'name': 'Linköping', 'lat': 58.4108, 'lng': 15.6214},
                {'name': 'Västerås', 'lat': 59.6099, 'lng': 16.5448},
                {'name': 'Örebro', 'lat': 59.2741, 'lng': 15.2066},
                {'name': 'Norrköping', 'lat': 58.5877, 'lng': 16.1924},
                {'name': 'Helsingborg', 'lat': 56.0465, 'lng': 12.6945},
                {'name': 'Jönköping', 'lat': 57.7826, 'lng': 14.1618}
            ]
            
            hotspots = []
            
            for location in locations:
                # Generate prediction for each location
                series = self.generate_sample_series(hours=72)  # 3 days of data
                prediction = self.ensemble.predict(series)
                
                # Calculate risk intensity
                intensity = min(1.0, prediction.prediction / 15.0)  # Normalize to 0-1
                
                hotspots.append({
                    'name': location['name'],
                    'lat': location['lat'],
                    'lng': location['lng'],
                    'intensity': intensity,
                    'prediction': prediction.prediction,
                    'confidence': prediction.confidence,
                    'risk_level': self._categorize_risk(intensity),
                    'estimated_incidents': round(prediction.prediction)
                })
            
            return {
                'hotspots': hotspots,
                'generated_at': datetime.now().isoformat(),
                'total_locations': len(hotspots),
                'high_risk_count': sum(1 for h in hotspots if h['risk_level'] == 'High')
            }
        
        return self.get_cached_or_compute('hotspots', compute_hotspots, ttl=300)
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get temporal pattern analysis data"""
        
        def compute_patterns():
            series = self.generate_sample_series(hours=168 * 4)  # 4 weeks of data
            
            # Daily pattern analysis
            daily_patterns = [0] * 24
            daily_counts = [0] * 24
            
            # Weekly pattern analysis  
            weekly_patterns = [0] * 7
            weekly_counts = [0] * 7
            
            for timestamp, value in series:
                dt = datetime.fromtimestamp(timestamp)
                hour = dt.hour
                weekday = dt.weekday()
                
                daily_patterns[hour] += value
                daily_counts[hour] += 1
                
                weekly_patterns[weekday] += value
                weekly_counts[weekday] += 1
            
            # Calculate averages
            daily_averages = [
                daily_patterns[i] / max(daily_counts[i], 1) 
                for i in range(24)
            ]
            
            weekly_averages = [
                weekly_patterns[i] / max(weekly_counts[i], 1) 
                for i in range(7)
            ]
            
            return {
                'daily_pattern': {
                    'hours': list(range(24)),
                    'values': daily_averages,
                    'peak_hour': daily_averages.index(max(daily_averages)),
                    'low_hour': daily_averages.index(min(daily_averages))
                },
                'weekly_pattern': {
                    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    'values': weekly_averages,
                    'peak_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][weekly_averages.index(max(weekly_averages))],
                    'low_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][weekly_averages.index(min(weekly_averages))]
                },
                'analysis': {
                    'weekend_factor': (weekly_averages[5] + weekly_averages[6]) / (sum(weekly_averages[:5]) / 5),
                    'night_factor': (sum(daily_averages[22:24]) + sum(daily_averages[0:6])) / (sum(daily_averages[6:22]) / 16),
                    'volatility': self._calculate_volatility([v for _, v in series])
                }
            }
        
        return self.get_cached_or_compute('patterns', compute_patterns, ttl=600)
    
    def _calculate_recent_trend(self, series: List[tuple]) -> float:
        """Calculate trend from recent data points"""
        if len(series) < 24:
            return 0.0
        
        recent_values = [v for _, v in series[-24:]]
        older_values = [v for _, v in series[-48:-24]]
        
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        return (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
    
    def _calculate_weekly_trend(self, series: List[tuple]) -> float:
        """Calculate weekly trend"""
        if len(series) < 168:
            return 0.0
        
        this_week = [v for _, v in series[-168:]]
        last_week = [v for _, v in series[-336:-168]]
        
        this_avg = sum(this_week) / len(this_week)
        last_avg = sum(last_week) / len(last_week)
        
        return (this_avg - last_avg) / last_avg if last_avg > 0 else 0.0
    
    def _find_peak_hour(self, series: List[tuple]) -> str:
        """Find the hour with highest average activity"""
        hourly_sums = [0] * 24
        hourly_counts = [0] * 24
        
        for timestamp, value in series[-168:]:  # Last week
            hour = datetime.fromtimestamp(timestamp).hour
            hourly_sums[hour] += value
            hourly_counts[hour] += 1
        
        hourly_averages = [
            hourly_sums[i] / max(hourly_counts[i], 1) 
            for i in range(24)
        ]
        
        peak_hour = hourly_averages.index(max(hourly_averages))
        return f"{peak_hour:02d}:00"
    
    def _categorize_risk(self, intensity: float) -> str:
        """Categorize risk level based on intensity"""
        if intensity > 0.8:
            return 'High'
        elif intensity > 0.6:
            return 'Medium'
        elif intensity > 0.4:
            return 'Low'
        else:
            return 'Very Low'
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of time series"""
        if len(values) < 2:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return math.sqrt(variance) / mean_val if mean_val > 0 else 0.0


# Initialize API handler
predictions_api = PredictionsAPI()


# API Routes
@app.route('/api/predictions/current', methods=['GET'])
def get_current_prediction():
    """Get current crime prediction with explanation"""
    try:
        data = predictions_api.get_current_prediction()
        return jsonify({
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/predictions/timeline', methods=['GET'])
def get_prediction_timeline():
    """Get prediction timeline for specified period"""
    period = request.args.get('period', '24h')
    
    if period not in ['1h', '24h', '7d', '30d']:
        return jsonify({
            'success': False,
            'error': 'Invalid period. Must be one of: 1h, 24h, 7d, 30d'
        }), 400
    
    try:
        data = predictions_api.get_prediction_timeline(period)
        return jsonify({
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/predictions/hotspots', methods=['GET'])
def get_hotspots():
    """Get hotspot prediction data for map visualization"""
    try:
        data = predictions_api.get_hotspot_data()
        return jsonify({
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/predictions/patterns', methods=['GET'])
def get_patterns():
    """Get temporal pattern analysis"""
    try:
        data = predictions_api.get_pattern_analysis()
        return jsonify({
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/predictions/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'service': 'Levi Predictions API',
        'timestamp': datetime.now().isoformat(),
        'cache_size': len(predictions_api.cache)
    })


@app.route('/api/predictions/cache/clear', methods=['POST'])
def clear_cache():
    """Clear prediction cache"""
    predictions_api.cache.clear()
    return jsonify({
        'success': True,
        'message': 'Cache cleared successfully',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("Starting Levi Predictions API...")
    print("Available endpoints:")
    print("  GET  /api/predictions/current   - Current prediction with explanation")
    print("  GET  /api/predictions/timeline  - Prediction timeline (?period=1h|24h|7d|30d)")
    print("  GET  /api/predictions/hotspots  - Hotspot map data")
    print("  GET  /api/predictions/patterns  - Temporal pattern analysis")
    print("  GET  /api/predictions/health    - Health check")
    print("  POST /api/predictions/cache/clear - Clear cache")
    print()
    
    app.run(host='0.0.0.0', port=5001, debug=True)