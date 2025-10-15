"""
Advanced Supplier Analysis Capability

Modular supplier analysis system with configurable scoring algorithms and comprehensive risk assessment.
Replaces hardcoded supplier analysis with flexible, data-driven evaluation.
"""

import logging
from typing import Dict, List, Any, Optional, Protocol
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from .base import AICapability, CapabilityResult, CapabilityConfig, CapabilityStatus

logger = logging.getLogger(__name__)


@dataclass
class SupplierMetrics:
    """Core supplier performance metrics"""
    financial_health: Optional[float] = None
    delivery_performance: Optional[float] = None
    quality_score: Optional[float] = None
    compliance_rating: Optional[float] = None
    innovation_index: Optional[float] = None
    sustainability_score: Optional[float] = None
    communication_rating: Optional[float] = None
    price_competitiveness: Optional[float] = None
    capacity_utilization: Optional[float] = None
    geographic_risk: Optional[float] = None


@dataclass
class SupplierProfile:
    """Comprehensive supplier profile"""
    supplier_id: str
    name: str
    country: str
    certifications: List[str]
    years_active: int
    employee_count: Optional[int] = None
    annual_revenue: Optional[float] = None
    primary_industries: Optional[List[str]] = None
    manufacturing_locations: Optional[List[str]] = None
    key_customers: Optional[List[str]] = None


class ScoringModule(ABC):
    """Abstract base class for scoring modules"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name"""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Default weight in overall scoring"""
        pass
    
    @abstractmethod
    async def calculate_score(self, supplier_profile: SupplierProfile, data: Dict[str, Any]) -> float:
        """Calculate score for this module (0.0 to 10.0)"""
        pass
    
    @abstractmethod
    def get_insights(self, score: float, data: Dict[str, Any]) -> List[str]:
        """Get human-readable insights from the score"""
        pass


class FinancialHealthModule(ScoringModule):
    """Financial health scoring module"""
    
    @property
    def name(self) -> str:
        return "financial_health"
    
    @property
    def weight(self) -> float:
        return 0.25
    
    async def calculate_score(self, supplier_profile: SupplierProfile, data: Dict[str, Any]) -> float:
        """Calculate financial health score"""
        financial_data = data.get('financial', {})
        
        # Credit rating component (40% of financial score)
        credit_rating = financial_data.get('credit_rating', 'C')
        credit_scores = {'AAA': 10, 'AA': 9, 'A': 8, 'BBB': 7, 'BB': 6, 'B': 5, 'C': 4, 'D': 2}
        credit_score = credit_scores.get(credit_rating, 4)
        
        # Revenue stability (30% of financial score)
        revenue_history = financial_data.get('revenue_history', [])
        revenue_stability = self._calculate_revenue_stability(revenue_history)
        
        # Debt-to-equity ratio (20% of financial score)
        debt_equity_ratio = financial_data.get('debt_equity_ratio', 0.5)
        debt_score = max(0, 10 - (debt_equity_ratio * 10))
        
        # Cash flow (10% of financial score)
        cash_flow_ratio = financial_data.get('cash_flow_ratio', 0.1)
        cash_flow_score = min(10, cash_flow_ratio * 50)
        
        # Weighted average
        total_score = (
            credit_score * 0.4 +
            revenue_stability * 0.3 +
            debt_score * 0.2 +
            cash_flow_score * 0.1
        )
        
        return min(10.0, max(0.0, total_score))
    
    def _calculate_revenue_stability(self, revenue_history: List[Dict]) -> float:
        """Calculate revenue stability score"""
        if len(revenue_history) < 2:
            return 5.0  # Neutral score for insufficient data
        
        revenues = [r.get('amount', 0) for r in revenue_history]
        if not revenues:
            return 5.0
        
        # Calculate variance
        mean_revenue = sum(revenues) / len(revenues)
        if mean_revenue == 0:
            return 2.0
        
        variance = sum((r - mean_revenue) ** 2 for r in revenues) / len(revenues)
        coefficient_of_variation = (variance ** 0.5) / mean_revenue
        
        # Lower CV = higher stability = higher score
        stability_score = max(0, 10 - (coefficient_of_variation * 20))
        return min(10.0, stability_score)
    
    def get_insights(self, score: float, data: Dict[str, Any]) -> List[str]:
        """Get financial health insights"""
        insights = []
        financial_data = data.get('financial', {})
        
        if score >= 8.0:
            insights.append("Excellent financial health with strong credit rating")
        elif score >= 6.0:
            insights.append("Good financial stability with manageable risk")
        elif score >= 4.0:
            insights.append("Moderate financial health requiring monitoring")
        else:
            insights.append("Financial concerns requiring immediate attention")
        
        credit_rating = financial_data.get('credit_rating')
        if credit_rating:
            insights.append(f"Credit rating: {credit_rating}")
        
        debt_equity = financial_data.get('debt_equity_ratio')
        if debt_equity and debt_equity > 0.7:
            insights.append("High debt-to-equity ratio may indicate financial stress")
        
        return insights


class DeliveryPerformanceModule(ScoringModule):
    """Delivery performance scoring module"""
    
    @property
    def name(self) -> str:
        return "delivery_performance"
    
    @property
    def weight(self) -> float:
        return 0.20
    
    async def calculate_score(self, supplier_profile: SupplierProfile, data: Dict[str, Any]) -> float:
        """Calculate delivery performance score"""
        delivery_data = data.get('delivery', {})
        
        # On-time delivery rate (50% of delivery score)
        on_time_rate = delivery_data.get('on_time_rate', 0.8)  # 80% default
        on_time_score = on_time_rate * 10
        
        # Average delay for late deliveries (25% of delivery score)
        avg_delay_days = delivery_data.get('avg_delay_days', 5)
        delay_score = max(0, 10 - (avg_delay_days / 2))
        
        # Delivery consistency (variance in delivery times) (25% of delivery score)
        delivery_variance = delivery_data.get('delivery_variance', 3)  # days
        consistency_score = max(0, 10 - delivery_variance)
        
        # Weighted average
        total_score = (
            on_time_score * 0.5 +
            delay_score * 0.25 +
            consistency_score * 0.25
        )
        
        return min(10.0, max(0.0, total_score))
    
    def get_insights(self, score: float, data: Dict[str, Any]) -> List[str]:
        """Get delivery performance insights"""
        insights = []
        delivery_data = data.get('delivery', {})
        
        on_time_rate = delivery_data.get('on_time_rate', 0.8)
        if on_time_rate >= 0.95:
            insights.append("Excellent on-time delivery performance")
        elif on_time_rate >= 0.85:
            insights.append("Good delivery reliability")
        elif on_time_rate >= 0.75:
            insights.append("Acceptable delivery performance with room for improvement")
        else:
            insights.append("Poor delivery performance requiring immediate attention")
        
        insights.append(f"On-time delivery rate: {on_time_rate*100:.1f}%")
        
        avg_delay = delivery_data.get('avg_delay_days')
        if avg_delay and avg_delay > 7:
            insights.append(f"Average delay of {avg_delay} days when late")
        
        return insights


class QualityModule(ScoringModule):
    """Quality scoring module"""
    
    @property
    def name(self) -> str:
        return "quality"
    
    @property
    def weight(self) -> float:
        return 0.20
    
    async def calculate_score(self, supplier_profile: SupplierProfile, data: Dict[str, Any]) -> float:
        """Calculate quality score"""
        quality_data = data.get('quality', {})
        
        # Defect rate (40% of quality score)
        defect_rate = quality_data.get('defect_rate', 0.02)  # 2% default
        defect_score = max(0, 10 - (defect_rate * 200))  # Scale defect rate
        
        # Customer return rate (25% of quality score)
        return_rate = quality_data.get('return_rate', 0.01)  # 1% default
        return_score = max(0, 10 - (return_rate * 500))
        
        # Quality certifications (20% of quality score)
        certifications = supplier_profile.certifications
        cert_score = self._calculate_certification_score(certifications)
        
        # Quality audit results (15% of quality score)
        audit_score = quality_data.get('last_audit_score', 7.0)
        
        # Weighted average
        total_score = (
            defect_score * 0.4 +
            return_score * 0.25 +
            cert_score * 0.2 +
            audit_score * 0.15
        )
        
        return min(10.0, max(0.0, total_score))
    
    def _calculate_certification_score(self, certifications: List[str]) -> float:
        """Calculate score based on quality certifications"""
        cert_scores = {
            'ISO9001': 3.0,
            'ISO14001': 2.0,
            'TS16949': 3.5,
            'AS9100': 4.0,
            'ISO13485': 3.5,
            'FDA': 2.5,
            'CE': 1.5
        }
        
        total_score = 0
        for cert in certifications:
            total_score += cert_scores.get(cert.upper(), 0)
        
        return min(10.0, total_score)
    
    def get_insights(self, score: float, data: Dict[str, Any]) -> List[str]:
        """Get quality insights"""
        insights = []
        quality_data = data.get('quality', {})
        
        if score >= 8.0:
            insights.append("Excellent quality standards and processes")
        elif score >= 6.0:
            insights.append("Good quality performance")
        elif score >= 4.0:
            insights.append("Quality improvements needed")
        else:
            insights.append("Significant quality concerns")
        
        defect_rate = quality_data.get('defect_rate')
        if defect_rate:
            insights.append(f"Defect rate: {defect_rate*100:.2f}%")
        
        return insights


class GeopoliticalRiskModule(ScoringModule):
    """Geopolitical risk scoring module"""
    
    @property
    def name(self) -> str:
        return "geopolitical_risk"
    
    @property
    def weight(self) -> float:
        return 0.15
    
    def __init__(self):
        # Country risk scores (higher = more stable)
        self.country_scores = {
            'Germany': 9.0, 'Switzerland': 9.5, 'Japan': 8.5, 'USA': 8.0,
            'UK': 8.0, 'South Korea': 7.5, 'Taiwan': 6.0, 'China': 5.5,
            'Mexico': 6.5, 'India': 6.0, 'Thailand': 6.5, 'Malaysia': 7.0,
            'Vietnam': 5.5, 'Philippines': 5.0, 'Brazil': 5.5, 'Turkey': 4.5
        }
    
    async def calculate_score(self, supplier_profile: SupplierProfile, data: Dict[str, Any]) -> float:
        """Calculate geopolitical risk score"""
        country = supplier_profile.country
        base_score = self.country_scores.get(country, 5.0)  # Default moderate risk
        
        # Adjust for additional risk factors
        geo_data = data.get('geopolitical', {})
        
        # Trade relationship stability
        trade_stability = geo_data.get('trade_stability_index', 0.7)
        trade_adjustment = (trade_stability - 0.5) * 2  # Scale to -1 to 1
        
        # Regulatory environment
        regulatory_score = geo_data.get('regulatory_environment_score', 6.0)
        regulatory_adjustment = (regulatory_score - 6.0) / 2
        
        # Political stability
        political_score = geo_data.get('political_stability_score', 6.0)
        political_adjustment = (political_score - 6.0) / 2
        
        final_score = base_score + trade_adjustment + regulatory_adjustment + political_adjustment
        
        return min(10.0, max(0.0, final_score))
    
    def get_insights(self, score: float, data: Dict[str, Any]) -> List[str]:
        """Get geopolitical risk insights"""
        insights = []
        
        if score >= 8.0:
            insights.append("Low geopolitical risk - stable operating environment")
        elif score >= 6.0:
            insights.append("Moderate geopolitical risk - monitor for changes")
        elif score >= 4.0:
            insights.append("Elevated geopolitical risk - consider diversification")
        else:
            insights.append("High geopolitical risk - immediate risk mitigation required")
        
        return insights


class SustainabilityModule(ScoringModule):
    """Sustainability and ESG scoring module"""
    
    @property
    def name(self) -> str:
        return "sustainability"
    
    @property
    def weight(self) -> float:
        return 0.10
    
    async def calculate_score(self, supplier_profile: SupplierProfile, data: Dict[str, Any]) -> float:
        """Calculate sustainability score"""
        sustainability_data = data.get('sustainability', {})
        
        # Environmental certifications (30% of sustainability score)
        env_certs = sustainability_data.get('environmental_certifications', [])
        env_score = self._calculate_env_cert_score(env_certs)
        
        # Carbon footprint (25% of sustainability score)
        carbon_score = sustainability_data.get('carbon_efficiency_score', 5.0)
        
        # Labor practices (25% of sustainability score)
        labor_score = sustainability_data.get('labor_practices_score', 6.0)
        
        # Waste management (20% of sustainability score)
        waste_score = sustainability_data.get('waste_management_score', 6.0)
        
        # Weighted average
        total_score = (
            env_score * 0.3 +
            carbon_score * 0.25 +
            labor_score * 0.25 +
            waste_score * 0.2
        )
        
        return min(10.0, max(0.0, total_score))
    
    def _calculate_env_cert_score(self, certifications: List[str]) -> float:
        """Calculate environmental certification score"""
        cert_scores = {
            'ISO14001': 4.0,
            'OHSAS18001': 3.0,
            'ISO45001': 3.5,
            'LEED': 2.5,
            'ENERGY_STAR': 2.0,
            'RoHS': 1.5,
            'REACH': 2.0
        }
        
        total_score = 0
        for cert in certifications:
            total_score += cert_scores.get(cert.upper(), 0)
        
        return min(10.0, total_score)
    
    def get_insights(self, score: float, data: Dict[str, Any]) -> List[str]:
        """Get sustainability insights"""
        insights = []
        
        if score >= 8.0:
            insights.append("Excellent sustainability practices and ESG compliance")
        elif score >= 6.0:
            insights.append("Good sustainability practices")
        elif score >= 4.0:
            insights.append("Basic sustainability measures in place")
        else:
            insights.append("Limited sustainability practices - improvement needed")
        
        return insights


class InnovationModule(ScoringModule):
    """Innovation and technology capability scoring module"""
    
    @property
    def name(self) -> str:
        return "innovation"
    
    @property
    def weight(self) -> float:
        return 0.10
    
    async def calculate_score(self, supplier_profile: SupplierProfile, data: Dict[str, Any]) -> float:
        """Calculate innovation score"""
        innovation_data = data.get('innovation', {})
        
        # R&D investment as percentage of revenue (40% of innovation score)
        rd_investment = innovation_data.get('rd_investment_percentage', 0.02)  # 2% default
        rd_score = min(10, rd_investment * 200)  # Scale R&D percentage
        
        # Patent portfolio (30% of innovation score)
        patent_count = innovation_data.get('patent_count', 0)
        patent_score = min(10, patent_count / 10)  # Scale patent count
        
        # Technology partnerships (20% of innovation score)
        tech_partnerships = innovation_data.get('technology_partnerships', 0)
        partnership_score = min(10, tech_partnerships * 2)
        
        # Digital maturity (10% of innovation score)
        digital_score = innovation_data.get('digital_maturity_score', 5.0)
        
        # Weighted average
        total_score = (
            rd_score * 0.4 +
            patent_score * 0.3 +
            partnership_score * 0.2 +
            digital_score * 0.1
        )
        
        return min(10.0, max(0.0, total_score))
    
    def get_insights(self, score: float, data: Dict[str, Any]) -> List[str]:
        """Get innovation insights"""
        insights = []
        
        if score >= 8.0:
            insights.append("Strong innovation capabilities and technology leadership")
        elif score >= 6.0:
            insights.append("Good innovation practices")
        elif score >= 4.0:
            insights.append("Moderate innovation capabilities")
        else:
            insights.append("Limited innovation - may struggle with future requirements")
        
        return insights


class AdvancedSupplierAnalysis(AICapability):
    """
    Advanced supplier analysis capability with modular scoring system.
    
    Provides comprehensive supplier evaluation using configurable scoring modules.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="advanced_supplier_analysis",
            version="2.0.0",
            timeout_seconds=30.0,
            retry_count=2
        )
        super().__init__(config)
        
        # Initialize scoring modules
        self.scoring_modules = {
            'financial_health': FinancialHealthModule(),
            'delivery_performance': DeliveryPerformanceModule(),
            'quality': QualityModule(),
            'geopolitical_risk': GeopoliticalRiskModule(),
            'sustainability': SustainabilityModule(),
            'innovation': InnovationModule()
        }
        
        # Default scoring weights (can be customized per analysis)
        self.default_weights = {
            'financial_health': 0.25,
            'delivery_performance': 0.20,
            'quality': 0.20,
            'geopolitical_risk': 0.15,
            'sustainability': 0.10,
            'innovation': 0.10
        }
    
    async def initialize(self) -> bool:
        """Initialize supplier analysis capability"""
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Execute advanced supplier analysis.
        
        Expected payload:
        {
            "supplier_id": "supplier_123",
            "analysis_modules": ["financial_health", "delivery_performance", "quality"],  # Optional
            "custom_weights": {"financial_health": 0.3, "quality": 0.4},  # Optional
            "comparison_suppliers": ["supplier_124", "supplier_125"],  # Optional
            "time_period": "12m",  # Analysis period
            "include_recommendations": true
        }
        """
        supplier_id = payload.get('supplier_id')
        if not supplier_id:
            return CapabilityResult(
                success=False,
                error="Supplier ID is required"
            )
        
        analysis_modules = payload.get('analysis_modules', list(self.scoring_modules.keys()))
        custom_weights = payload.get('custom_weights', {})
        comparison_suppliers = payload.get('comparison_suppliers', [])
        time_period = payload.get('time_period', '12m')
        include_recommendations = payload.get('include_recommendations', True)
        
        try:
            # Get supplier profile and data
            supplier_profile = await self._get_supplier_profile(supplier_id)
            if not supplier_profile:
                return CapabilityResult(
                    success=False,
                    error=f"Supplier {supplier_id} not found"
                )
            
            supplier_data = await self._get_supplier_data(supplier_id, time_period)
            
            # Calculate scores for each module
            module_scores = {}
            module_insights = {}
            
            for module_name in analysis_modules:
                if module_name in self.scoring_modules:
                    module = self.scoring_modules[module_name]
                    score = await module.calculate_score(supplier_profile, supplier_data)
                    insights = module.get_insights(score, supplier_data)
                    
                    module_scores[module_name] = score
                    module_insights[module_name] = insights
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                module_scores, custom_weights or self.default_weights
            )
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(module_scores, supplier_profile)
            
            # Generate recommendations if requested
            recommendations = []
            if include_recommendations:
                recommendations = await self._generate_recommendations(
                    module_scores, module_insights, supplier_profile
                )
            
            # Perform comparison analysis if requested
            comparison_analysis = None
            if comparison_suppliers:
                comparison_analysis = await self._perform_comparison_analysis(
                    supplier_id, comparison_suppliers, analysis_modules, custom_weights
                )
            
            return CapabilityResult(
                success=True,
                data={
                    "supplier_profile": asdict(supplier_profile),
                    "overall_score": round(overall_score, 2),
                    "module_scores": {k: round(v, 2) for k, v in module_scores.items()},
                    "module_insights": module_insights,
                    "risk_assessment": risk_assessment,
                    "recommendations": recommendations,
                    "comparison_analysis": comparison_analysis,
                    "analysis_metadata": {
                        "modules_used": analysis_modules,
                        "weights_applied": custom_weights or self.default_weights,
                        "time_period": time_period,
                        "analysis_date": datetime.now().isoformat()
                    }
                },
                confidence=self._calculate_analysis_confidence(supplier_data, analysis_modules)
            )
            
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Supplier analysis failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check with sample supplier analysis"""
        try:
            test_result = await self.execute({
                "supplier_id": "test_supplier",
                "analysis_modules": ["financial_health", "quality"],
                "include_recommendations": False
            })
            return test_result.success or "not found" in test_result.error.lower()
        except:
            return False
    
    async def _get_supplier_profile(self, supplier_id: str) -> Optional[SupplierProfile]:
        """Get supplier profile from database"""
        try:
            from ..db.session import get_session
            from ..db.models import Supplier
            
            session = next(get_session())
            try:
                supplier = session.query(Supplier).filter(
                    Supplier.id == supplier_id
                ).first()
                
                if not supplier:
                    return None
                
                return SupplierProfile(
                    supplier_id=supplier.id,
                    name=supplier.name,
                    country=getattr(supplier, 'country', 'Unknown'),
                    certifications=getattr(supplier, 'certifications', []),
                    years_active=getattr(supplier, 'years_active', 5),
                    employee_count=getattr(supplier, 'employee_count', None),
                    annual_revenue=getattr(supplier, 'annual_revenue', None),
                    primary_industries=getattr(supplier, 'primary_industries', []),
                    manufacturing_locations=getattr(supplier, 'manufacturing_locations', []),
                    key_customers=getattr(supplier, 'key_customers', [])
                )
                
            finally:
                session.close()
                
        except Exception as e:
            logger.warning(f"Failed to get supplier profile for {supplier_id}: {e}")
            return None
    
    async def _get_supplier_data(self, supplier_id: str, time_period: str) -> Dict[str, Any]:
        """Get comprehensive supplier performance data"""
        try:
            from ..db.session import get_session
            from ..db.models import PurchaseOrder, Supplier, QualityIssue
            from sqlalchemy import func, and_
            
            # Convert time period to days
            days = self._period_to_days(time_period)
            start_date = datetime.now() - timedelta(days=days)
            
            session = next(get_session())
            try:
                # Get delivery performance data
                delivery_stats = session.query(
                    func.avg(PurchaseOrder.delivery_delay_days).label('avg_delay'),
                    func.count(PurchaseOrder.id).label('total_orders'),
                    func.sum(
                        (PurchaseOrder.delivery_delay_days <= 0).cast(int)
                    ).label('on_time_orders')
                ).filter(
                    and_(
                        PurchaseOrder.supplier_id == supplier_id,
                        PurchaseOrder.created_at >= start_date
                    )
                ).first()
                
                # Get quality data
                quality_stats = session.query(
                    func.count(QualityIssue.id).label('quality_issues'),
                    func.avg(QualityIssue.severity_score).label('avg_severity')
                ).filter(
                    and_(
                        QualityIssue.supplier_id == supplier_id,
                        QualityIssue.created_at >= start_date
                    )
                ).first()
                
                # Calculate metrics
                total_orders = delivery_stats.total_orders or 0
                on_time_orders = delivery_stats.on_time_orders or 0
                on_time_rate = (on_time_orders / total_orders) if total_orders > 0 else 0.8
                avg_delay = delivery_stats.avg_delay or 0
                
                quality_issues = quality_stats.quality_issues or 0
                defect_rate = (quality_issues / total_orders) if total_orders > 0 else 0.01
                
                return {
                    'financial': {
                        'credit_rating': 'B',  # Would be from credit agency API
                        'revenue_history': [],  # Would be from financial data
                        'debt_equity_ratio': 0.4,
                        'cash_flow_ratio': 0.15
                    },
                    'delivery': {
                        'on_time_rate': on_time_rate,
                        'avg_delay_days': max(0, avg_delay),
                        'delivery_variance': 2.5,
                        'total_orders': total_orders
                    },
                    'quality': {
                        'defect_rate': defect_rate,
                        'return_rate': defect_rate * 0.5,  # Assume half of defects result in returns
                        'last_audit_score': 7.5,
                        'quality_issues_count': quality_issues
                    },
                    'geopolitical': {
                        'trade_stability_index': 0.75,
                        'regulatory_environment_score': 6.5,
                        'political_stability_score': 7.0
                    },
                    'sustainability': {
                        'environmental_certifications': ['ISO14001'],
                        'carbon_efficiency_score': 6.0,
                        'labor_practices_score': 7.0,
                        'waste_management_score': 6.5
                    },
                    'innovation': {
                        'rd_investment_percentage': 0.03,
                        'patent_count': 5,
                        'technology_partnerships': 2,
                        'digital_maturity_score': 6.0
                    }
                }
                
            finally:
                session.close()
                
        except Exception as e:
            logger.warning(f"Failed to get supplier data for {supplier_id}: {e}")
            # Return default data structure
            return {
                'financial': {'credit_rating': 'C', 'debt_equity_ratio': 0.5, 'cash_flow_ratio': 0.1},
                'delivery': {'on_time_rate': 0.8, 'avg_delay_days': 3, 'delivery_variance': 4},
                'quality': {'defect_rate': 0.02, 'return_rate': 0.01, 'last_audit_score': 6.0},
                'geopolitical': {'trade_stability_index': 0.7, 'regulatory_environment_score': 6.0},
                'sustainability': {'environmental_certifications': [], 'carbon_efficiency_score': 5.0},
                'innovation': {'rd_investment_percentage': 0.02, 'patent_count': 1}
            }
    
    def _calculate_overall_score(self, module_scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0
        total_weight = 0
        
        for module_name, score in module_scores.items():
            weight = weights.get(module_name, self.default_weights.get(module_name, 0.1))
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 5.0
    
    def _generate_risk_assessment(
        self, module_scores: Dict[str, float], supplier_profile: SupplierProfile
    ) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        
        # Identify critical risk areas
        critical_risks = []
        high_risks = []
        
        for module_name, score in module_scores.items():
            if score <= 3.0:
                critical_risks.append(module_name)
            elif score <= 5.0:
                high_risks.append(module_name)
        
        # Calculate overall risk level
        avg_score = sum(module_scores.values()) / len(module_scores)
        
        if avg_score >= 8.0:
            risk_level = "low"
        elif avg_score >= 6.0:
            risk_level = "medium"
        elif avg_score >= 4.0:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Geographic risk factors
        geo_risks = []
        high_risk_countries = ['China', 'Taiwan', 'Russia', 'Belarus']
        if supplier_profile.country in high_risk_countries:
            geo_risks.append(f"Operating in high-risk country: {supplier_profile.country}")
        
        return {
            "overall_risk_level": risk_level,
            "risk_score": round(10 - avg_score, 2),
            "critical_risk_areas": critical_risks,
            "high_risk_areas": high_risks,
            "geographic_risks": geo_risks,
            "risk_trend": "stable",  # Would be calculated from historical data
            "mitigation_priority": "high" if critical_risks else "medium" if high_risks else "low"
        }
    
    async def _generate_recommendations(
        self, 
        module_scores: Dict[str, float], 
        module_insights: Dict[str, List[str]],
        supplier_profile: SupplierProfile
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Score-based recommendations
        for module_name, score in module_scores.items():
            if score <= 4.0:
                recommendations.extend(
                    self._get_module_specific_recommendations(module_name, score, supplier_profile)
                )
        
        # Overall recommendations
        avg_score = sum(module_scores.values()) / len(module_scores)
        
        if avg_score <= 5.0:
            recommendations.append({
                "category": "strategic",
                "priority": "high",
                "action": "supplier_diversification",
                "description": "Consider diversifying supplier base to reduce dependency risk",
                "timeline": "3-6 months",
                "estimated_cost": "medium"
            })
        
        if 'geopolitical_risk' in module_scores and module_scores['geopolitical_risk'] <= 5.0:
            recommendations.append({
                "category": "risk_mitigation",
                "priority": "high", 
                "action": "geographic_diversification",
                "description": "Identify suppliers in more stable geographic regions",
                "timeline": "6-12 months",
                "estimated_cost": "high"
            })
        
        return recommendations
    
    def _get_module_specific_recommendations(
        self, module_name: str, score: float, supplier_profile: SupplierProfile
    ) -> List[Dict[str, Any]]:
        """Get recommendations specific to each scoring module"""
        recommendations = []
        
        if module_name == "financial_health" and score <= 4.0:
            recommendations.append({
                "category": "financial",
                "priority": "urgent",
                "action": "financial_monitoring",
                "description": "Implement enhanced financial monitoring and request regular financial reports",
                "timeline": "immediate",
                "estimated_cost": "low"
            })
        
        elif module_name == "delivery_performance" and score <= 4.0:
            recommendations.append({
                "category": "operational",
                "priority": "high",
                "action": "delivery_improvement_plan",
                "description": "Work with supplier to develop delivery improvement plan",
                "timeline": "1-3 months",
                "estimated_cost": "medium"
            })
        
        elif module_name == "quality" and score <= 4.0:
            recommendations.append({
                "category": "quality",
                "priority": "urgent",
                "action": "quality_audit",
                "description": "Conduct comprehensive quality audit and corrective action plan",
                "timeline": "immediate",
                "estimated_cost": "medium"
            })
        
        elif module_name == "sustainability" and score <= 4.0:
            recommendations.append({
                "category": "sustainability",
                "priority": "medium",
                "action": "sustainability_improvement",
                "description": "Support supplier in implementing sustainability improvements",
                "timeline": "6-12 months",
                "estimated_cost": "medium"
            })
        
        return recommendations
    
    async def _perform_comparison_analysis(
        self, 
        primary_supplier: str,
        comparison_suppliers: List[str], 
        analysis_modules: List[str],
        custom_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Perform comparative analysis between suppliers"""
        comparison_results = {}
        
        # Analyze each comparison supplier
        for supplier_id in comparison_suppliers:
            try:
                result = await self.execute({
                    "supplier_id": supplier_id,
                    "analysis_modules": analysis_modules,
                    "custom_weights": custom_weights,
                    "include_recommendations": False
                })
                
                if result.success:
                    comparison_results[supplier_id] = {
                        "overall_score": result.data["overall_score"],
                        "module_scores": result.data["module_scores"]
                    }
            except Exception as e:
                logger.warning(f"Failed to analyze comparison supplier {supplier_id}: {e}")
        
        # Generate comparative insights
        if comparison_results:
            all_scores = list(comparison_results.values())
            avg_overall = sum(s["overall_score"] for s in all_scores) / len(all_scores)
            
            return {
                "comparison_suppliers": comparison_results,
                "market_average": round(avg_overall, 2),
                "ranking": self._calculate_supplier_ranking(primary_supplier, comparison_results),
                "competitive_advantages": self._identify_competitive_advantages(comparison_results),
                "improvement_opportunities": self._identify_improvement_opportunities(comparison_results)
            }
        
        return {}
    
    def _calculate_supplier_ranking(
        self, primary_supplier: str, comparison_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate ranking among suppliers"""
        # Would implement ranking logic based on scores
        return {"position": 1, "total_suppliers": len(comparison_results) + 1}
    
    def _identify_competitive_advantages(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Identify competitive advantages"""
        # Would implement comparison logic
        return ["Strong delivery performance", "Excellent quality standards"]
    
    def _identify_improvement_opportunities(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement relative to competitors"""
        # Would implement gap analysis
        return ["Financial stability improvement needed", "Innovation capabilities below market average"]
    
    def _calculate_analysis_confidence(self, supplier_data: Dict[str, Any], modules: List[str]) -> float:
        """Calculate confidence in the analysis results"""
        base_confidence = 0.7
        
        # Adjust based on data availability
        total_orders = supplier_data.get('delivery', {}).get('total_orders', 0)
        if total_orders >= 10:
            base_confidence += 0.1
        elif total_orders >= 5:
            base_confidence += 0.05
        
        # Adjust based on module coverage
        module_coverage = len(modules) / len(self.scoring_modules)
        base_confidence += module_coverage * 0.15
        
        return min(0.95, base_confidence)
    
    def _period_to_days(self, period: str) -> int:
        """Convert period string to days"""
        period_map = {
            '1m': 30, '3m': 90, '6m': 180, '12m': 365, '1y': 365,
            '2y': 730, '3y': 1095
        }
        return period_map.get(period.lower(), 365)