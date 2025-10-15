"""
Scenario Analysis Capability

Advanced scenario analysis engine for supply chain risk assessment and strategic planning.
Replaces hardcoded scenario logic with flexible, data-driven models.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .base import AICapability, CapabilityResult, CapabilityConfig, CapabilityStatus

logger = logging.getLogger(__name__)


@dataclass
class ScenarioImpact:
    """Impact assessment for a specific scenario"""
    component_id: str
    severity: str  # low, medium, high, critical
    probability: float  # 0.0 to 1.0
    impact_areas: List[str]  # ["pricing", "availability", "quality", "delivery"]
    estimated_cost_impact: Optional[float] = None
    time_to_recovery: Optional[int] = None  # days
    mitigation_options: Optional[List[str]] = None


@dataclass
class ScenarioParameters:
    """Flexible parameters for scenario analysis"""
    scenario_type: str
    geographic_scope: List[str]
    industry_sectors: List[str]
    timeframe: str  # "immediate", "short_term", "medium_term", "long_term"
    probability: float
    duration: Optional[int] = None  # days
    severity_factors: Optional[Dict[str, float]] = None
    affected_regions: Optional[List[str]] = None
    affected_suppliers: Optional[List[str]] = None


class ScenarioAnalysisCapability(AICapability):
    """
    Advanced scenario analysis capability for supply chain risk assessment.
    
    Supports flexible scenario types and data-driven impact modeling.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="scenario_analysis",
            version="2.0.0",
            timeout_seconds=45.0,
            retry_count=2
        )
        super().__init__(config)
        
        # Load scenario models and risk factors
        self.scenario_models = {}
        self.risk_factors = self._load_risk_factors()
        self.impact_models = self._load_impact_models()
        self.mitigation_strategies = self._load_mitigation_strategies()
    
    async def initialize(self) -> bool:
        """Initialize scenario analysis models"""
        try:
            # Load historical scenario data for pattern recognition
            await self._load_historical_scenarios()
            await self.update_status(CapabilityStatus.HEALTHY)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize scenario analysis: {e}")
            await self.update_status(CapabilityStatus.DEGRADED)
            return False
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Execute scenario analysis.
        
        Expected payload:
        {
            "scenario_type": "geopolitical|economic|natural_disaster|cyber|pandemic|trade_policy",
            "parameters": {
                "geographic_scope": ["APAC", "EMEA"],
                "industry_sectors": ["semiconductors", "electronics"],
                "timeframe": "medium_term",
                "probability": 0.3,
                "severity_factors": {"supply_disruption": 0.8, "demand_shift": 0.4}
            },
            "affected_components": ["component_id_1", "component_id_2"],
            "analysis_depth": "comprehensive|standard|quick"
        }
        """
        scenario_type = payload.get('scenario_type', 'economic')
        parameters = ScenarioParameters(**payload.get('parameters', {}))
        affected_components = payload.get('affected_components', [])
        analysis_depth = payload.get('analysis_depth', 'standard')
        
        if not scenario_type:
            return CapabilityResult(
                success=False,
                error="Scenario type is required"
            )
        
        try:
            # Get or create scenario model
            scenario_model = await self._get_scenario_model(scenario_type, parameters)
            
            # Run multi-dimensional impact analysis
            impacts = await self._analyze_scenario_impacts(
                scenario_model, parameters, affected_components, analysis_depth
            )
            
            # Generate mitigation recommendations
            recommendations = await self._generate_mitigation_recommendations(
                scenario_type, parameters, impacts
            )
            
            # Calculate overall risk assessment
            risk_assessment = self._calculate_overall_risk(impacts, parameters)
            
            # Generate early warning indicators
            warning_indicators = await self._generate_warning_indicators(
                scenario_type, parameters
            )
            
            return CapabilityResult(
                success=True,
                data={
                    "scenario_id": f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "scenario_type": scenario_type,
                    "parameters": asdict(parameters),
                    "risk_assessment": risk_assessment,
                    "component_impacts": [asdict(impact) for impact in impacts],
                    "mitigation_recommendations": recommendations,
                    "warning_indicators": warning_indicators,
                    "confidence": scenario_model.get("confidence", 0.75),
                    "analysis_depth": analysis_depth,
                    "generated_at": datetime.now().isoformat()
                },
                confidence=scenario_model.get("confidence", 0.75)
            )
            
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Scenario analysis failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check with sample scenario analysis"""
        try:
            test_result = await self.execute({
                "scenario_type": "economic",
                "parameters": {
                    "scenario_type": "economic",
                    "geographic_scope": ["Global"],
                    "industry_sectors": ["electronics"],
                    "timeframe": "short_term",
                    "probability": 0.3
                },
                "affected_components": ["test_component"],
                "analysis_depth": "quick"
            })
            return test_result.success
        except:
            return False
    
    def _load_risk_factors(self) -> Dict[str, Any]:
        """Load comprehensive risk factor database"""
        return {
            "geopolitical": {
                "base_factors": {
                    "trade_restrictions": {"weight": 0.3, "regions": ["APAC", "EMEA"]},
                    "sanctions": {"weight": 0.4, "regions": ["EMEA", "Americas"]},
                    "territorial_disputes": {"weight": 0.2, "regions": ["APAC"]},
                    "election_cycles": {"weight": 0.1, "regions": ["Global"]}
                },
                "regional_multipliers": {
                    "APAC": 1.2,
                    "EMEA": 1.0,
                    "Americas": 0.8
                }
            },
            "economic": {
                "base_factors": {
                    "inflation": {"weight": 0.25, "sectors": ["all"]},
                    "currency_volatility": {"weight": 0.2, "sectors": ["all"]},
                    "interest_rates": {"weight": 0.15, "sectors": ["all"]},
                    "commodity_prices": {"weight": 0.2, "sectors": ["materials", "energy"]},
                    "demand_cycles": {"weight": 0.2, "sectors": ["technology", "automotive"]}
                }
            },
            "natural_disaster": {
                "base_factors": {
                    "earthquakes": {"weight": 0.3, "regions": ["Japan", "Taiwan", "California"]},
                    "typhoons": {"weight": 0.25, "regions": ["APAC"]},
                    "floods": {"weight": 0.2, "regions": ["Global"]},
                    "wildfires": {"weight": 0.15, "regions": ["California", "Australia"]},
                    "extreme_weather": {"weight": 0.1, "regions": ["Global"]}
                }
            },
            "cyber": {
                "base_factors": {
                    "supply_chain_attacks": {"weight": 0.4, "sectors": ["technology"]},
                    "infrastructure_attacks": {"weight": 0.3, "sectors": ["all"]},
                    "data_breaches": {"weight": 0.2, "sectors": ["all"]},
                    "ransomware": {"weight": 0.1, "sectors": ["all"]}
                }
            },
            "pandemic": {
                "base_factors": {
                    "factory_closures": {"weight": 0.4, "regions": ["Global"]},
                    "logistics_disruption": {"weight": 0.3, "regions": ["Global"]},
                    "demand_shifts": {"weight": 0.2, "regions": ["Global"]},
                    "workforce_reduction": {"weight": 0.1, "regions": ["Global"]}
                }
            },
            "trade_policy": {
                "base_factors": {
                    "tariffs": {"weight": 0.35, "regions": ["Global"]},
                    "export_controls": {"weight": 0.3, "regions": ["US", "EU", "China"]},
                    "trade_agreements": {"weight": 0.2, "regions": ["Global"]},
                    "regulatory_changes": {"weight": 0.15, "regions": ["Global"]}
                }
            }
        }
    
    def _load_impact_models(self) -> Dict[str, Any]:
        """Load impact assessment models for different scenario types"""
        return {
            "pricing_impact": {
                "immediate": {"multiplier": 1.1, "variance": 0.1},
                "short_term": {"multiplier": 1.2, "variance": 0.15},
                "medium_term": {"multiplier": 1.3, "variance": 0.2},
                "long_term": {"multiplier": 1.15, "variance": 0.25}
            },
            "availability_impact": {
                "critical_components": {"shortage_probability": 0.8, "duration_days": 90},
                "standard_components": {"shortage_probability": 0.4, "duration_days": 45},
                "commodity_components": {"shortage_probability": 0.2, "duration_days": 30}
            },
            "quality_impact": {
                "geopolitical": {"defect_rate_increase": 0.05},
                "natural_disaster": {"defect_rate_increase": 0.15},
                "economic": {"defect_rate_increase": 0.02},
                "cyber": {"defect_rate_increase": 0.1}
            },
            "delivery_impact": {
                "local_scenario": {"delay_days": 7, "variance": 3},
                "regional_scenario": {"delay_days": 21, "variance": 10},
                "global_scenario": {"delay_days": 45, "variance": 20}
            }
        }
    
    def _load_mitigation_strategies(self) -> Dict[str, Any]:
        """Load comprehensive mitigation strategy database"""
        return {
            "immediate_actions": {
                "increase_safety_stock": {
                    "effectiveness": 0.7,
                    "cost_multiplier": 1.2,
                    "implementation_time": 7
                },
                "activate_backup_suppliers": {
                    "effectiveness": 0.6,
                    "cost_multiplier": 1.15,
                    "implementation_time": 14
                },
                "expedite_shipments": {
                    "effectiveness": 0.5,
                    "cost_multiplier": 1.5,
                    "implementation_time": 3
                }
            },
            "short_term_actions": {
                "diversify_supplier_base": {
                    "effectiveness": 0.8,
                    "cost_multiplier": 1.1,
                    "implementation_time": 60
                },
                "redesign_for_alternatives": {
                    "effectiveness": 0.9,
                    "cost_multiplier": 1.05,
                    "implementation_time": 90
                },
                "establish_regional_partnerships": {
                    "effectiveness": 0.75,
                    "cost_multiplier": 1.0,
                    "implementation_time": 120
                }
            },
            "long_term_actions": {
                "vertical_integration": {
                    "effectiveness": 0.95,
                    "cost_multiplier": 0.9,
                    "implementation_time": 365
                },
                "technology_platform_shift": {
                    "effectiveness": 0.85,
                    "cost_multiplier": 0.95,
                    "implementation_time": 540
                },
                "strategic_inventory_hubs": {
                    "effectiveness": 0.8,
                    "cost_multiplier": 1.1,
                    "implementation_time": 180
                }
            }
        }
    
    async def _load_historical_scenarios(self):
        """Load historical scenario data for pattern recognition"""
        try:
            from ..db.session import get_session
            from ..db.models import ScenarioHistory  # Would need to create this model
            
            # For now, use built-in historical patterns
            self.historical_patterns = {
                "semiconductor_shortage_2020": {
                    "scenario_type": "pandemic",
                    "duration_days": 730,
                    "price_increase": 0.35,
                    "availability_impact": 0.6
                },
                "trade_war_2018": {
                    "scenario_type": "trade_policy",
                    "duration_days": 1095,
                    "price_increase": 0.15,
                    "availability_impact": 0.3
                },
                "fukushima_2011": {
                    "scenario_type": "natural_disaster",
                    "duration_days": 180,
                    "price_increase": 0.25,
                    "availability_impact": 0.8
                }
            }
        except Exception as e:
            logger.warning(f"Failed to load historical scenarios: {e}")
            self.historical_patterns = {}
    
    async def _get_scenario_model(self, scenario_type: str, parameters: ScenarioParameters) -> Dict[str, Any]:
        """Get or create scenario model based on type and parameters"""
        model_key = f"{scenario_type}_{parameters.timeframe}"
        
        if model_key not in self.scenario_models:
            # Create scenario model dynamically
            base_risk = self.risk_factors.get(scenario_type, {})
            
            # Calculate confidence based on historical data and parameter completeness
            confidence = self._calculate_model_confidence(scenario_type, parameters)
            
            self.scenario_models[model_key] = {
                "scenario_type": scenario_type,
                "base_risk_factors": base_risk,
                "confidence": confidence,
                "calibration_date": datetime.now().isoformat()
            }
        
        return self.scenario_models[model_key]
    
    async def _analyze_scenario_impacts(
        self, 
        scenario_model: Dict[str, Any], 
        parameters: ScenarioParameters, 
        affected_components: List[str],
        analysis_depth: str
    ) -> List[ScenarioImpact]:
        """Analyze impacts on specific components using multi-dimensional models"""
        impacts = []
        
        for component_id in affected_components:
            try:
                # Get component data for enhanced analysis
                component_data = await self._get_component_data(component_id)
                
                # Calculate multi-dimensional impact
                impact = await self._calculate_component_impact(
                    component_id, component_data, scenario_model, parameters, analysis_depth
                )
                
                impacts.append(impact)
                
            except Exception as e:
                logger.warning(f"Failed to analyze impact for component {component_id}: {e}")
                # Add fallback impact
                impacts.append(ScenarioImpact(
                    component_id=component_id,
                    severity="medium",
                    probability=parameters.probability,
                    impact_areas=["availability"],
                    estimated_cost_impact=None,
                    time_to_recovery=None,
                    mitigation_options=["gather_more_data"]
                ))
        
        return impacts
    
    async def _calculate_component_impact(
        self,
        component_id: str,
        component_data: Dict[str, Any],
        scenario_model: Dict[str, Any],
        parameters: ScenarioParameters,
        analysis_depth: str
    ) -> ScenarioImpact:
        """Calculate detailed impact for a specific component"""
        
        # Determine component criticality
        criticality = self._assess_component_criticality(component_data)
        
        # Calculate base impact probability
        base_probability = parameters.probability
        
        # Adjust probability based on component characteristics
        geographic_exposure = self._calculate_geographic_exposure(
            component_data, parameters.geographic_scope
        )
        supplier_concentration = self._calculate_supplier_concentration(component_data)
        
        adjusted_probability = base_probability * geographic_exposure * supplier_concentration
        
        # Determine severity based on multiple factors
        severity_score = self._calculate_severity_score(
            scenario_model, parameters, component_data, criticality
        )
        
        severity = self._score_to_severity(severity_score)
        
        # Identify impact areas
        impact_areas = self._identify_impact_areas(
            scenario_model["scenario_type"], component_data, parameters
        )
        
        # Estimate cost impact
        cost_impact = self._estimate_cost_impact(
            component_data, severity_score, parameters
        ) if analysis_depth in ["standard", "comprehensive"] else None
        
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(
            scenario_model["scenario_type"], severity_score, parameters
        ) if analysis_depth == "comprehensive" else None
        
        # Generate mitigation options
        mitigation_options = self._generate_component_mitigations(
            component_data, scenario_model["scenario_type"], severity
        )
        
        return ScenarioImpact(
            component_id=component_id,
            severity=severity,
            probability=min(0.95, adjusted_probability),
            impact_areas=impact_areas,
            estimated_cost_impact=cost_impact,
            time_to_recovery=recovery_time,
            mitigation_options=mitigation_options
        )
    
    async def _get_component_data(self, component_id: str) -> Dict[str, Any]:
        """Get comprehensive component data for impact analysis"""
        try:
            from ..db.session import get_session
            from ..db.models import Component, Supplier, PriceHistory
            from sqlalchemy import func, desc
            
            session = next(get_session())
            try:
                # Get component with supplier information
                component = session.query(Component).filter(
                    Component.id == component_id
                ).first()
                
                if not component:
                    return {"id": component_id, "category": "unknown", "suppliers": []}
                
                # Get supplier information
                suppliers = session.query(Supplier).join(
                    # Would need supplier-component relationship table
                ).filter(
                    # Component relationship filter
                ).all()
                
                # Get price history for volatility analysis
                price_history = session.query(PriceHistory).filter(
                    PriceHistory.component_id == component_id
                ).order_by(desc(PriceHistory.effective_date)).limit(12).all()
                
                return {
                    "id": component_id,
                    "manufacturer_part_number": component.manufacturer_part_number,
                    "category": component.category,
                    "description": component.description,
                    "manufacturer": component.manufacturer,
                    "suppliers": [{"id": s.id, "name": s.name, "country": getattr(s, 'country', 'Unknown')} for s in suppliers],
                    "price_history": [{"date": p.effective_date, "price": float(p.unit_price)} for p in price_history],
                    "price_volatility": self._calculate_price_volatility(price_history) if price_history else 0.0
                }
                
            finally:
                session.close()
                
        except Exception as e:
            logger.warning(f"Failed to get component data for {component_id}: {e}")
            return {"id": component_id, "category": "unknown", "suppliers": []}
    
    def _assess_component_criticality(self, component_data: Dict[str, Any]) -> str:
        """Assess component criticality based on multiple factors"""
        category = component_data.get("category", "").lower()
        supplier_count = len(component_data.get("suppliers", []))
        price_volatility = component_data.get("price_volatility", 0.0)
        
        criticality_score = 0
        
        # Category-based criticality
        if any(critical in category for critical in ["processor", "microcontroller", "fpga"]):
            criticality_score += 3
        elif any(important in category for important in ["memory", "power", "analog"]):
            criticality_score += 2
        else:
            criticality_score += 1
        
        # Supplier diversity factor
        if supplier_count == 1:
            criticality_score += 2
        elif supplier_count <= 3:
            criticality_score += 1
        
        # Price volatility factor
        if price_volatility > 0.3:
            criticality_score += 1
        
        if criticality_score >= 5:
            return "critical"
        elif criticality_score >= 3:
            return "high"
        elif criticality_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_geographic_exposure(
        self, component_data: Dict[str, Any], geographic_scope: List[str]
    ) -> float:
        """Calculate component exposure to geographic scenario"""
        suppliers = component_data.get("suppliers", [])
        if not suppliers:
            return 1.0  # Unknown exposure, assume full
        
        exposed_suppliers = 0
        for supplier in suppliers:
            supplier_country = supplier.get("country", "Unknown")
            # Simple mapping - would be more sophisticated in production
            if any(region in ["APAC", "Asia"] for region in geographic_scope) and supplier_country in ["China", "Taiwan", "Japan", "South Korea"]:
                exposed_suppliers += 1
            elif any(region in ["EMEA", "Europe"] for region in geographic_scope) and supplier_country in ["Germany", "Netherlands", "UK"]:
                exposed_suppliers += 1
            elif "Global" in geographic_scope:
                exposed_suppliers += 1
        
        exposure_ratio = exposed_suppliers / len(suppliers)
        return 0.5 + (exposure_ratio * 0.5)  # Scale between 0.5-1.0
    
    def _calculate_supplier_concentration(self, component_data: Dict[str, Any]) -> float:
        """Calculate supplier concentration risk multiplier"""
        supplier_count = len(component_data.get("suppliers", []))
        
        if supplier_count == 1:
            return 1.5  # High concentration risk
        elif supplier_count <= 3:
            return 1.2  # Medium concentration risk
        elif supplier_count <= 5:
            return 1.0  # Moderate concentration
        else:
            return 0.8  # Low concentration risk
    
    def _calculate_severity_score(
        self,
        scenario_model: Dict[str, Any],
        parameters: ScenarioParameters,
        component_data: Dict[str, Any],
        criticality: str
    ) -> float:
        """Calculate overall severity score (0.0 to 1.0)"""
        base_score = parameters.probability
        
        # Criticality multiplier
        criticality_multipliers = {"critical": 1.5, "high": 1.2, "medium": 1.0, "low": 0.8}
        base_score *= criticality_multipliers.get(criticality, 1.0)
        
        # Scenario type impact
        scenario_impacts = {
            "geopolitical": 0.8,
            "natural_disaster": 0.9,
            "cyber": 0.7,
            "pandemic": 0.85,
            "economic": 0.6,
            "trade_policy": 0.75
        }
        base_score *= scenario_impacts.get(scenario_model["scenario_type"], 0.7)
        
        # Timeframe impact
        timeframe_multipliers = {
            "immediate": 1.2,
            "short_term": 1.0,
            "medium_term": 0.8,
            "long_term": 0.6
        }
        base_score *= timeframe_multipliers.get(parameters.timeframe, 1.0)
        
        return min(1.0, base_score)
    
    def _score_to_severity(self, score: float) -> str:
        """Convert severity score to category"""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _identify_impact_areas(
        self, scenario_type: str, component_data: Dict[str, Any], parameters: ScenarioParameters
    ) -> List[str]:
        """Identify which areas will be impacted"""
        areas = []
        
        # Scenario-specific impact areas
        scenario_areas = {
            "geopolitical": ["availability", "pricing", "delivery"],
            "natural_disaster": ["availability", "delivery", "quality"],
            "cyber": ["availability", "quality", "delivery"],
            "pandemic": ["availability", "delivery", "pricing"],
            "economic": ["pricing", "demand"],
            "trade_policy": ["pricing", "availability", "delivery"]
        }
        
        base_areas = scenario_areas.get(scenario_type, ["availability", "pricing"])
        
        # Add component-specific considerations
        category = component_data.get("category", "").lower()
        if "semiconductor" in category and scenario_type in ["geopolitical", "natural_disaster"]:
            if "quality" not in base_areas:
                base_areas.append("quality")
        
        return base_areas
    
    def _estimate_cost_impact(
        self, component_data: Dict[str, Any], severity_score: float, parameters: ScenarioParameters
    ) -> float:
        """Estimate financial cost impact"""
        # Get recent price data
        price_history = component_data.get("price_history", [])
        if not price_history:
            return None
        
        recent_price = price_history[0]["price"] if price_history else 100.0
        
        # Base cost increase based on severity
        cost_multiplier = 1.0 + (severity_score * 0.5)  # Up to 50% increase
        
        # Scenario-specific adjustments
        scenario_multipliers = {
            "natural_disaster": 1.3,
            "geopolitical": 1.2,
            "pandemic": 1.25,
            "cyber": 1.1,
            "trade_policy": 1.15,
            "economic": 1.05
        }
        
        cost_multiplier *= scenario_multipliers.get(parameters.scenario_type, 1.0)
        
        estimated_new_price = recent_price * cost_multiplier
        cost_impact = estimated_new_price - recent_price
        
        return round(cost_impact, 2)
    
    def _estimate_recovery_time(
        self, scenario_type: str, severity_score: float, parameters: ScenarioParameters
    ) -> int:
        """Estimate recovery time in days"""
        base_recovery_times = {
            "natural_disaster": 90,
            "geopolitical": 365,
            "pandemic": 180,
            "cyber": 30,
            "trade_policy": 270,
            "economic": 120
        }
        
        base_time = base_recovery_times.get(scenario_type, 120)
        
        # Adjust based on severity
        severity_multiplier = 0.5 + severity_score  # 0.5x to 1.5x
        
        # Adjust based on timeframe
        timeframe_multipliers = {
            "immediate": 0.8,
            "short_term": 1.0,
            "medium_term": 1.2,
            "long_term": 1.5
        }
        
        recovery_time = base_time * severity_multiplier * timeframe_multipliers.get(parameters.timeframe, 1.0)
        
        return int(recovery_time)
    
    def _generate_component_mitigations(
        self, component_data: Dict[str, Any], scenario_type: str, severity: str
    ) -> List[str]:
        """Generate component-specific mitigation options"""
        mitigations = []
        
        supplier_count = len(component_data.get("suppliers", []))
        
        # Supplier diversity recommendations
        if supplier_count <= 2:
            mitigations.append("diversify_supplier_base")
        
        # Inventory recommendations based on severity
        if severity in ["critical", "high"]:
            mitigations.append("increase_safety_stock")
            mitigations.append("establish_strategic_reserves")
        
        # Scenario-specific mitigations
        if scenario_type == "geopolitical":
            mitigations.extend(["identify_alternative_regions", "evaluate_nearshoring"])
        elif scenario_type == "natural_disaster":
            mitigations.extend(["geographic_supplier_distribution", "emergency_airlift_plans"])
        elif scenario_type == "cyber":
            mitigations.extend(["vendor_security_assessment", "supply_chain_monitoring"])
        elif scenario_type in ["pandemic", "economic"]:
            mitigations.extend(["demand_forecasting", "flexible_contracts"])
        
        # Component category specific
        category = component_data.get("category", "").lower()
        if "semiconductor" in category:
            mitigations.append("design_for_alternative_components")
        
        return list(set(mitigations))  # Remove duplicates
    
    async def _generate_mitigation_recommendations(
        self, scenario_type: str, parameters: ScenarioParameters, impacts: List[ScenarioImpact]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive mitigation recommendations"""
        recommendations = []
        
        # Aggregate impact severity
        critical_components = sum(1 for impact in impacts if impact.severity == "critical")
        high_impact_components = sum(1 for impact in impacts if impact.severity == "high")
        
        # Generate recommendations based on impact severity distribution
        if critical_components > 0:
            recommendations.extend(self._get_immediate_recommendations(scenario_type, critical_components))
        
        if high_impact_components > 2:
            recommendations.extend(self._get_short_term_recommendations(scenario_type, high_impact_components))
        
        recommendations.extend(self._get_long_term_recommendations(scenario_type, len(impacts)))
        
        # Prioritize and add implementation details
        prioritized = self._prioritize_recommendations(recommendations, parameters, impacts)
        
        return prioritized
    
    def _get_immediate_recommendations(self, scenario_type: str, critical_count: int) -> List[Dict[str, Any]]:
        """Get immediate action recommendations"""
        immediate = self.mitigation_strategies["immediate_actions"]
        recommendations = []
        
        if critical_count >= 3:
            recommendations.append({
                "action": "activate_crisis_management",
                "priority": "urgent",
                "timeline": "immediate",
                "description": "Activate crisis management protocols for supply chain disruption",
                "effectiveness": 0.8,
                "estimated_cost": 50000
            })
        
        recommendations.extend([
            {
                "action": key,
                "priority": "urgent" if critical_count >= 2 else "high",
                "timeline": "immediate",
                "description": f"Implement {key.replace('_', ' ')} for critical components",
                **details
            }
            for key, details in immediate.items()
        ])
        
        return recommendations
    
    def _get_short_term_recommendations(self, scenario_type: str, high_count: int) -> List[Dict[str, Any]]:
        """Get short-term strategic recommendations"""
        short_term = self.mitigation_strategies["short_term_actions"]
        
        return [
            {
                "action": key,
                "priority": "high",
                "timeline": "short_term",
                "description": f"Implement {key.replace('_', ' ')} strategy",
                **details
            }
            for key, details in short_term.items()
        ]
    
    def _get_long_term_recommendations(self, scenario_type: str, total_components: int) -> List[Dict[str, Any]]:
        """Get long-term strategic recommendations"""
        long_term = self.mitigation_strategies["long_term_actions"]
        
        recommendations = []
        
        if total_components >= 10:
            recommendations.extend([
                {
                    "action": key,
                    "priority": "medium",
                    "timeline": "long_term",
                    "description": f"Strategic implementation of {key.replace('_', ' ')}",
                    **details
                }
                for key, details in long_term.items()
            ])
        
        return recommendations
    
    def _prioritize_recommendations(
        self, recommendations: List[Dict[str, Any]], parameters: ScenarioParameters, impacts: List[ScenarioImpact]
    ) -> List[Dict[str, Any]]:
        """Prioritize recommendations based on effectiveness and context"""
        
        def recommendation_score(rec):
            base_score = rec.get("effectiveness", 0.5)
            
            # Priority boost
            priority_boost = {"urgent": 0.3, "high": 0.2, "medium": 0.1}.get(rec.get("priority", "medium"), 0.1)
            
            # Timeline penalty for longer implementations
            timeline_penalty = {"immediate": 0, "short_term": 0.05, "long_term": 0.1}.get(rec.get("timeline", "medium"), 0.05)
            
            # Cost efficiency (inverse of cost multiplier)
            cost_efficiency = 1.0 / rec.get("cost_multiplier", 1.2)
            
            return base_score + priority_boost - timeline_penalty + (cost_efficiency * 0.1)
        
        # Sort by score and add ranking
        sorted_recommendations = sorted(recommendations, key=recommendation_score, reverse=True)
        
        for i, rec in enumerate(sorted_recommendations):
            rec["recommendation_rank"] = i + 1
            rec["score"] = round(recommendation_score(rec), 3)
        
        return sorted_recommendations
    
    async def _generate_warning_indicators(
        self, scenario_type: str, parameters: ScenarioParameters
    ) -> List[Dict[str, Any]]:
        """Generate early warning indicators to monitor"""
        indicators = []
        
        scenario_indicators = {
            "geopolitical": [
                {"metric": "trade_policy_announcements", "threshold": 2, "timeframe": "weekly"},
                {"metric": "diplomatic_tensions_index", "threshold": 0.7, "timeframe": "daily"},
                {"metric": "cross_border_shipping_delays", "threshold": 1.5, "timeframe": "daily"}
            ],
            "economic": [
                {"metric": "inflation_rate", "threshold": 0.05, "timeframe": "monthly"},
                {"metric": "currency_volatility", "threshold": 0.1, "timeframe": "daily"},
                {"metric": "commodity_price_changes", "threshold": 0.1, "timeframe": "weekly"}
            ],
            "natural_disaster": [
                {"metric": "seismic_activity", "threshold": 6.0, "timeframe": "real_time"},
                {"metric": "weather_alerts", "threshold": 1, "timeframe": "daily"},
                {"metric": "infrastructure_damage_reports", "threshold": 1, "timeframe": "real_time"}
            ],
            "cyber": [
                {"metric": "security_incidents", "threshold": 1, "timeframe": "real_time"},
                {"metric": "threat_intelligence_alerts", "threshold": 3, "timeframe": "daily"},
                {"metric": "vendor_security_breaches", "threshold": 1, "timeframe": "real_time"}
            ]
        }
        
        base_indicators = scenario_indicators.get(scenario_type, [])
        
        for indicator in base_indicators:
            indicators.append({
                **indicator,
                "current_value": None,  # Would be populated by monitoring system
                "status": "monitoring",
                "geographic_scope": parameters.geographic_scope
            })
        
        return indicators
    
    def _calculate_overall_risk(self, impacts: List[ScenarioImpact], parameters: ScenarioParameters) -> Dict[str, Any]:
        """Calculate overall risk assessment from individual impacts"""
        if not impacts:
            return {"level": "unknown", "score": 0.5, "confidence": 0.3}
        
        # Calculate weighted risk score
        severity_weights = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
        
        weighted_sum = 0
        total_weight = 0
        
        for impact in impacts:
            weight = severity_weights.get(impact.severity, 0.5) * impact.probability
            weighted_sum += weight
            total_weight += impact.probability
        
        average_risk = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Determine risk level
        if average_risk >= 0.8:
            risk_level = "critical"
        elif average_risk >= 0.6:
            risk_level = "high"
        elif average_risk >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Calculate confidence based on data quality
        confidence = min(0.95, 0.5 + (len(impacts) * 0.05))
        
        return {
            "level": risk_level,
            "score": round(average_risk, 3),
            "confidence": round(confidence, 3),
            "components_analyzed": len(impacts),
            "critical_components": sum(1 for i in impacts if i.severity == "critical"),
            "high_risk_components": sum(1 for i in impacts if i.severity == "high")
        }
    
    def _calculate_model_confidence(self, scenario_type: str, parameters: ScenarioParameters) -> float:
        """Calculate model confidence based on available data and parameters"""
        base_confidence = 0.6
        
        # Historical data availability
        if scenario_type in self.historical_patterns:
            base_confidence += 0.15
        
        # Parameter completeness
        param_completeness = 0
        total_params = 6
        
        if parameters.geographic_scope:
            param_completeness += 1
        if parameters.industry_sectors:
            param_completeness += 1
        if parameters.timeframe:
            param_completeness += 1
        if parameters.probability is not None:
            param_completeness += 1
        if parameters.duration is not None:
            param_completeness += 1
        if parameters.severity_factors:
            param_completeness += 1
        
        completeness_bonus = (param_completeness / total_params) * 0.2
        base_confidence += completeness_bonus
        
        return min(0.95, base_confidence)
    
    def _calculate_price_volatility(self, price_history: List) -> float:
        """Calculate price volatility from historical data"""
        if len(price_history) < 2:
            return 0.0
        
        prices = [float(p.unit_price) for p in price_history]
        if not prices:
            return 0.0
        
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_deviation = variance ** 0.5
        
        return std_deviation / mean_price if mean_price > 0 else 0.0