"""
Forecasting Capabilities

Price and demand forecasting capabilities with statistical models.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta, timezone
import statistics

from .base import AICapability, CapabilityResult, CapabilityConfig, CapabilityStatus

logger = logging.getLogger(__name__)


class PriceForecastCapability(AICapability):
    """
    Price forecasting capability using simple statistical models.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="price_forecast",
            version="1.0.0",
            timeout_seconds=15.0,
            retry_count=2
        )
        super().__init__(config)
    
    async def initialize(self) -> bool:
        """Initialize the forecaster"""
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Generate price forecast.
        
        Expected payload:
        {
            "component_id": "Component ID",
            "historical_prices": [{"date": "2024-01-01", "price": 10.50, "quantity": 100}],
            "forecast_days": 90,
            "supplier_id": "Optional supplier filter"
        }
        """
        component_id = payload.get('component_id', '')
        historical_prices = payload.get('historical_prices', [])
        forecast_days = payload.get('forecast_days', 90)
        supplier_id = payload.get('supplier_id')
        
        if not component_id:
            return CapabilityResult(
                success=False,
                error="No component_id provided for price forecast"
            )
        
        try:
            if len(historical_prices) < 3:
                # Not enough data for meaningful forecast
                return CapabilityResult(
                    success=True,
                    data={
                        "component_id": component_id,
                        "forecast": [],
                        "trend": "insufficient_data",
                        "confidence": 0.1,
                        "forecast_method": "insufficient_data"
                    },
                    confidence=0.1,
                    warnings=["Insufficient historical data for accurate forecast"]
                )
            
            # Prepare historical data
            price_data = self._prepare_price_data(historical_prices)
            
            # Generate forecast using simple methods
            forecast = self._generate_price_forecast(price_data, forecast_days)
            
            # Calculate trend and volatility
            trend = self._calculate_trend(price_data)
            volatility = self._calculate_volatility(price_data)
            
            # Determine confidence based on data quality
            confidence = self._calculate_confidence(price_data, volatility)
            
            return CapabilityResult(
                success=True,
                data={
                    "component_id": component_id,
                    "forecast": forecast,
                    "trend": trend,
                    "volatility": volatility,
                    "confidence": confidence,
                    "forecast_method": "moving_average_with_trend",
                    "data_points_used": len(price_data),
                    "forecast_period_days": forecast_days
                },
                confidence=confidence
            )
            
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Price forecast failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check with sample forecast"""
        try:
            sample_data = [
                {"date": "2024-01-01", "price": 10.0, "quantity": 100},
                {"date": "2024-01-15", "price": 10.5, "quantity": 150},
                {"date": "2024-02-01", "price": 11.0, "quantity": 120},
                {"date": "2024-02-15", "price": 10.8, "quantity": 130}
            ]
            
            test_result = await self.execute({
                "component_id": "test_component",
                "historical_prices": sample_data,
                "forecast_days": 30
            })
            
            return test_result.success and len(test_result.data.get("forecast", [])) > 0
        except:
            return False
    
    def _prepare_price_data(self, historical_prices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare and sort price data"""
        # Sort by date
        sorted_data = sorted(
            historical_prices,
            key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00'))
        )
        
        # Add parsed datetime for easier processing
        for item in sorted_data:
            item['datetime'] = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
        
        return sorted_data
    
    def _generate_price_forecast(self, price_data: List[Dict[str, Any]], forecast_days: int) -> List[Dict[str, Any]]:
        """Generate price forecast using moving average with trend"""
        if len(price_data) < 2:
            return []
        
        # Calculate moving average and trend
        prices = [item['price'] for item in price_data]
        
        # Simple trend calculation (linear regression slope approximation)
        n = len(prices)
        x_mean = (n - 1) / 2  # Mean of indices
        y_mean = statistics.mean(prices)
        
        numerator = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        trend_slope = numerator / denominator if denominator != 0 else 0
        
        # Generate forecast points
        last_date = price_data[-1]['datetime']
        last_price = price_data[-1]['price']
        
        # Calculate recent volatility for confidence bands
        if len(prices) >= 10:
            recent_prices = prices[-10:]
        else:
            recent_prices = prices
        
        volatility = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
        
        forecast = []
        days_increment = max(1, forecast_days // 30)  # At most 30 forecast points
        
        for i in range(0, forecast_days + 1, days_increment):
            forecast_date = last_date + timedelta(days=i)
            
            # Price prediction: last price + trend * days
            predicted_price = last_price + (trend_slope * i)
            
            # Add some mean reversion (prices don't trend forever)
            mean_price = statistics.mean(prices)
            reversion_factor = min(0.1, i / forecast_days * 0.1)
            predicted_price = predicted_price * (1 - reversion_factor) + mean_price * reversion_factor
            
            # Ensure price doesn't go negative
            predicted_price = max(0.01, predicted_price)
            
            # Calculate confidence bands
            confidence_width = volatility * (1 + i / forecast_days)  # Widening confidence
            
            forecast.append({
                "date": forecast_date.isoformat(),
                "predicted_price": round(predicted_price, 4),
                "confidence_lower": round(max(0.01, predicted_price - confidence_width), 4),
                "confidence_upper": round(predicted_price + confidence_width, 4),
                "days_ahead": i
            })
        
        return forecast
    
    def _calculate_trend(self, price_data: List[Dict[str, Any]]) -> str:
        """Calculate overall price trend"""
        if len(price_data) < 2:
            return "unknown"
        
        first_price = price_data[0]['price']
        last_price = price_data[-1]['price']
        
        change_percent = ((last_price - first_price) / first_price) * 100
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_volatility(self, price_data: List[Dict[str, Any]]) -> float:
        """Calculate price volatility"""
        if len(price_data) < 2:
            return 0.0
        
        prices = [item['price'] for item in price_data]
        return round(statistics.stdev(prices), 4)
    
    def _calculate_confidence(self, price_data: List[Dict[str, Any]], volatility: float) -> float:
        """Calculate forecast confidence based on data quality"""
        base_confidence = 0.5
        
        # More data points = higher confidence
        data_points_bonus = min(0.3, len(price_data) * 0.02)
        
        # Lower volatility = higher confidence
        volatility_penalty = min(0.4, volatility * 0.1)
        
        # Time span bonus (longer history = better)
        if len(price_data) >= 2:
            time_span_days = (price_data[-1]['datetime'] - price_data[0]['datetime']).days
            time_span_bonus = min(0.2, time_span_days / 365 * 0.2)
        else:
            time_span_bonus = 0
        
        confidence = base_confidence + data_points_bonus + time_span_bonus - volatility_penalty
        return round(min(0.95, max(0.1, confidence)), 2)


class DemandForecastCapability(AICapability):
    """
    Demand forecasting capability using historical patterns.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="demand_forecast",
            version="1.0.0",
            timeout_seconds=12.0,
            retry_count=2
        )
        super().__init__(config)
    
    async def initialize(self) -> bool:
        """Initialize the forecaster"""
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Generate demand forecast.
        
        Expected payload:
        {
            "component_id": "Component ID",
            "historical_demand": [{"date": "2024-01-01", "quantity": 100, "orders": 5}],
            "forecast_days": 90,
            "seasonality": true  # Whether to account for seasonal patterns
        }
        """
        component_id = payload.get('component_id', '')
        historical_demand = payload.get('historical_demand', [])
        forecast_days = payload.get('forecast_days', 90)
        seasonality = payload.get('seasonality', True)
        
        if not component_id:
            return CapabilityResult(
                success=False,
                error="No component_id provided for demand forecast"
            )
        
        try:
            if len(historical_demand) < 2:
                return CapabilityResult(
                    success=True,
                    data={
                        "component_id": component_id,
                        "forecast": [],
                        "trend": "insufficient_data",
                        "confidence": 0.1
                    },
                    confidence=0.1,
                    warnings=["Insufficient historical data for demand forecast"]
                )
            
            # Prepare demand data
            demand_data = self._prepare_demand_data(historical_demand)
            
            # Generate forecast
            forecast = self._generate_demand_forecast(demand_data, forecast_days, seasonality)
            
            # Calculate trend
            trend = self._calculate_demand_trend(demand_data)
            
            # Calculate confidence
            confidence = self._calculate_demand_confidence(demand_data)
            
            return CapabilityResult(
                success=True,
                data={
                    "component_id": component_id,
                    "forecast": forecast,
                    "trend": trend,
                    "confidence": confidence,
                    "forecast_method": "moving_average_seasonal",
                    "data_points_used": len(demand_data),
                    "seasonality_applied": seasonality
                },
                confidence=confidence
            )
            
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Demand forecast failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check with sample forecast"""
        try:
            sample_data = [
                {"date": "2024-01-01", "quantity": 100, "orders": 5},
                {"date": "2024-01-15", "quantity": 150, "orders": 8},
                {"date": "2024-02-01", "quantity": 120, "orders": 6}
            ]
            
            test_result = await self.execute({
                "component_id": "test_component",
                "historical_demand": sample_data,
                "forecast_days": 30
            })
            
            return test_result.success
        except:
            return False
    
    def _prepare_demand_data(self, historical_demand: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare and sort demand data"""
        sorted_data = sorted(
            historical_demand,
            key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00'))
        )
        
        for item in sorted_data:
            item['datetime'] = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
        
        return sorted_data
    
    def _generate_demand_forecast(
        self, demand_data: List[Dict[str, Any]], forecast_days: int, seasonality: bool
    ) -> List[Dict[str, Any]]:
        """Generate demand forecast"""
        quantities = [item['quantity'] for item in demand_data]
        orders = [item['orders'] for item in demand_data]
        
        # Calculate averages
        avg_quantity = statistics.mean(quantities)
        avg_orders = statistics.mean(orders)
        
        # Simple trend calculation
        if len(quantities) >= 2:
            recent_avg = statistics.mean(quantities[-3:]) if len(quantities) >= 3 else quantities[-1]
            trend_factor = recent_avg / avg_quantity
        else:
            trend_factor = 1.0
        
        # Generate forecast
        last_date = demand_data[-1]['datetime']
        forecast = []
        
        for i in range(0, forecast_days + 1, 7):  # Weekly forecasts
            forecast_date = last_date + timedelta(days=i)
            
            # Base prediction
            predicted_quantity = avg_quantity * trend_factor
            predicted_orders = avg_orders * trend_factor
            
            # Apply seasonality if requested
            if seasonality:
                seasonal_factor = self._get_seasonal_factor(forecast_date)
                predicted_quantity *= seasonal_factor
                predicted_orders *= seasonal_factor
            
            # Add some randomness reduction over time (regression to mean)
            if i > 0:
                regression_factor = 1 - (i / forecast_days * 0.1)
                predicted_quantity = predicted_quantity * regression_factor + avg_quantity * (1 - regression_factor)
                predicted_orders = predicted_orders * regression_factor + avg_orders * (1 - regression_factor)
            
            forecast.append({
                "date": forecast_date.isoformat(),
                "predicted_quantity": round(max(0, predicted_quantity)),
                "predicted_orders": round(max(0, predicted_orders)),
                "weeks_ahead": i // 7
            })
        
        return forecast
    
    def _get_seasonal_factor(self, date: datetime) -> float:
        """Get seasonal adjustment factor"""
        month = date.month
        
        # Simple seasonal patterns for electronics
        # Higher demand in Q4 (holiday season), lower in summer
        seasonal_factors = {
            1: 0.9,   # January - post-holiday low
            2: 0.95,  # February
            3: 1.05,  # March - spring projects
            4: 1.0,   # April
            5: 1.0,   # May
            6: 0.9,   # June - summer low
            7: 0.85,  # July - summer low
            8: 0.9,   # August
            9: 1.05,  # September - back to school/work
            10: 1.1,  # October - pre-holiday prep
            11: 1.15, # November - holiday season
            12: 1.2   # December - peak holiday
        }
        
        return seasonal_factors.get(month, 1.0)
    
    def _calculate_demand_trend(self, demand_data: List[Dict[str, Any]]) -> str:
        """Calculate demand trend"""
        if len(demand_data) < 2:
            return "unknown"
        
        quantities = [item['quantity'] for item in demand_data]
        
        first_half = quantities[:len(quantities)//2]
        second_half = quantities[len(quantities)//2:]
        
        if len(first_half) == 0 or len(second_half) == 0:
            return "stable"
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_demand_confidence(self, demand_data: List[Dict[str, Any]]) -> float:
        """Calculate demand forecast confidence"""
        base_confidence = 0.4
        
        # More data = higher confidence
        data_bonus = min(0.3, len(demand_data) * 0.03)
        
        # Calculate demand variability
        quantities = [item['quantity'] for item in demand_data]
        if len(quantities) > 1:
            cv = statistics.stdev(quantities) / statistics.mean(quantities)  # Coefficient of variation
            variability_penalty = min(0.3, cv * 0.2)
        else:
            variability_penalty = 0.1
        
        confidence = base_confidence + data_bonus - variability_penalty
        return round(min(0.9, max(0.1, confidence)), 2)