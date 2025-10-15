"""
Recommendation Capabilities

Component recommendation and alternative suggestion capabilities.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio

from .base import AICapability, CapabilityResult, CapabilityConfig, CapabilityStatus

logger = logging.getLogger(__name__)


class ComponentRecommender(AICapability):
    """
    Component recommendation system based on specifications and usage patterns.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="component_recommender",
            version="1.0.0",
            timeout_seconds=20.0,
            retry_count=2
        )
        super().__init__(config)
        
        # Dynamic recommendation system - loads patterns from database/config
        self.recommendation_cache = {}
        self.component_patterns = self._load_component_patterns()
        self.similarity_threshold = 0.3
    
    async def initialize(self) -> bool:
        """Initialize the recommender"""
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Generate component recommendations.
        
        Expected payload:
        {
            "component": "Current component part number or description",
            "category": "Optional component category",
            "specifications": {"voltage": "5V", "current": "1A"},  # Optional
            "max_recommendations": 5,
            "include_alternatives": true,
            "price_range": {"min": 0, "max": 100}  # Optional
        }
        """
        component = payload.get('component', '')
        category = payload.get('category', '')
        specifications = payload.get('specifications', {})
        max_recommendations = payload.get('max_recommendations', 5)
        include_alternatives = payload.get('include_alternatives', True)
        price_range = payload.get('price_range', {})
        
        if not component:
            return CapabilityResult(
                success=False,
                error="No component provided for recommendation"
            )
        
        try:
            recommendations = []
            
            # Database-driven recommendations (alternatives used together)
            db_recommendations = await self._get_database_recommendations(
                component, category, max_recommendations
            )
            recommendations.extend(db_recommendations)
            
            # Pattern-based recommendations (similar part numbers/functions)
            pattern_recommendations = self._get_pattern_based_recommendations(
                component, category, max_recommendations
            )
            recommendations.extend(pattern_recommendations)
            
            # Specification-based recommendations
            if specifications:
                spec_recommendations = self._get_specification_based_recommendations(
                    component, specifications, max_recommendations
                )
                recommendations.extend(spec_recommendations)
            
            # Fallback to similarity-based recommendations
            if len(recommendations) < max_recommendations // 2:
                similarity_recommendations = self._get_similarity_based_recommendations(
                    component, max_recommendations - len(recommendations)
                )
                recommendations.extend(similarity_recommendations)
            
            # Remove duplicates and sort by confidence
            recommendations = self._deduplicate_recommendations(recommendations)
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply price filtering if specified
            if price_range:
                recommendations = await self._filter_by_price(recommendations, price_range)
            
            # Limit results
            recommendations = recommendations[:max_recommendations]
            
            return CapabilityResult(
                success=True,
                data={
                    "recommendations": recommendations,
                    "total_found": len(recommendations),
                    "input_component": component,
                    "recommendation_methods": ["rule_based", "specification_based"]
                },
                confidence=0.7 if recommendations else 0.1
            )
            
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Component recommendation failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check with sample recommendation"""
        try:
            test_result = await self.execute({
                "component": "STM32F429ZIT6",
                "max_recommendations": 3
            })
            return test_result.success and len(test_result.data.get("recommendations", [])) > 0
        except:
            return False
    
    def _load_component_patterns(self) -> Dict[str, Any]:
        """Load component patterns and families for recommendations"""
        return {
            "microcontroller_families": {
                "stm32": ["STM32F0", "STM32F1", "STM32F2", "STM32F3", "STM32F4", "STM32F7", "STM32H7", "STM32L0", "STM32L1", "STM32L4"],
                "atmega": ["ATMEGA8", "ATMEGA16", "ATMEGA32", "ATMEGA64", "ATMEGA128", "ATMEGA168", "ATMEGA328", "ATMEGA2560"],
                "pic": ["PIC16", "PIC18", "PIC24", "PIC32"],
                "msp430": ["MSP430F", "MSP430G"],
                "esp": ["ESP8266", "ESP32", "ESP32-S2", "ESP32-C3"]
            },
            "analog_families": {
                "opamp": ["LM", "TL", "AD", "OPA", "MAX"],
                "regulator": ["LM78", "LM79", "LM317", "AMS1117", "LP29", "XL"],
                "comparator": ["LM393", "LM339", "MAX9"],
                "reference": ["LM4040", "AD1580", "MAX6"]
            },
            "passive_ranges": {
                "resistor": {"1206": [10, 100000], "0805": [10, 100000], "0603": [10, 100000]},
                "capacitor": {"ceramic": [1e-12, 10e-6], "electrolytic": [1e-6, 10e-3]},
                "inductor": {"power": [1e-6, 10e-3], "rf": [1e-9, 1e-6]}
            }
        }
    
    async def _get_database_recommendations(self, component: str, category: str, max_count: int) -> List[Dict[str, Any]]:
        """Get recommendations based on actual usage patterns from database"""
        recommendations = []
        
        try:
            from ..db.session import get_session
            from ..db.models import RFQItem, Component
            from sqlalchemy import func, and_
            
            session = next(get_session())
            try:
                # Find components frequently requested together
                similar_components = session.query(
                    Component.manufacturer_part_number,
                    Component.category,
                    Component.description,
                    func.count().label('frequency')
                ).join(RFQItem).filter(
                    and_(
                        Component.manufacturer_part_number != component,
                        RFQItem.rfq_id.in_(
                            session.query(RFQItem.rfq_id).join(Component).filter(
                                Component.manufacturer_part_number.ilike(f"%{component}%")
                            )
                        )
                    )
                ).group_by(
                    Component.manufacturer_part_number,
                    Component.category,
                    Component.description
                ).order_by(func.count().desc()).limit(max_count).all()
                
                for comp in similar_components:
                    confidence = min(0.9, 0.4 + (comp.frequency / 10))  # Scale based on frequency
                    recommendations.append({
                        "component": comp.manufacturer_part_number,
                        "reason": f"Frequently used with {component} ({comp.frequency} times)",
                        "confidence": confidence,
                        "category": comp.category or "unknown",
                        "recommendation_type": "frequently_together",
                        "description": comp.description,
                        "usage_frequency": comp.frequency
                    })
                    
            finally:
                session.close()
                
        except Exception as e:
            logger.warning(f"Database recommendations failed: {e}")
        
        return recommendations
    
    def _get_pattern_based_recommendations(self, component: str, category: str, max_count: int) -> List[Dict[str, Any]]:
        """Get recommendations based on component family patterns"""
        recommendations = []
        component_upper = component.upper()
        
        # Check microcontroller families
        for family, members in self.component_patterns.get("microcontroller_families", {}).items():
            if any(member in component_upper for member in members):
                # Recommend other family members
                for member in members:
                    if member not in component_upper and len(recommendations) < max_count:
                        # Generate specific part numbers based on pattern
                        similar_parts = self._generate_similar_parts(component, member)
                        for part in similar_parts[:2]:  # Limit to 2 per family member
                            recommendations.append({
                                "component": part,
                                "reason": f"Same family as {component} ({family})",
                                "confidence": 0.7,
                                "category": "microcontroller",
                                "recommendation_type": "family_member"
                            })
        
        # Check analog families
        for family, prefixes in self.component_patterns.get("analog_families", {}).items():
            for prefix in prefixes:
                if component_upper.startswith(prefix):
                    # Find similar components with same prefix but different suffix
                    similar_parts = self._generate_analog_alternatives(component, prefix)
                    for part in similar_parts:
                        if len(recommendations) < max_count:
                            recommendations.append({
                                "component": part,
                                "reason": f"Similar {family} component",
                                "confidence": 0.6,
                                "category": "analog",
                                "recommendation_type": "functional_equivalent"
                            })
        
        return recommendations[:max_count]
    
    def _get_similarity_based_recommendations(self, component: str, max_count: int) -> List[Dict[str, Any]]:
        """Generate recommendations based on component similarity patterns"""
        recommendations = []
        component_upper = component.upper()
        
        # Extract numeric patterns and suggest variations
        import re
        
        # Pattern for components like STM32F429ZIT6
        mcu_pattern = r'([A-Z]+)(\d+)([A-Z])(\d+)([A-Z]+)(\d+)'
        match = re.match(mcu_pattern, component_upper)
        
        if match:
            prefix, series, letter, number, suffix, package = match.groups()
            
            # Suggest different series numbers
            for new_series in [str(int(series) + 1), str(int(series) - 1)]:
                if new_series != series:
                    variant = f"{prefix}{new_series}{letter}{number}{suffix}{package}"
                    recommendations.append({
                        "component": variant,
                        "reason": f"Different series variant of {component}",
                        "confidence": 0.5,
                        "category": "microcontroller",
                        "recommendation_type": "series_variant"
                    })
                    if len(recommendations) >= max_count:
                        break
        
        # Pattern for simple components like LM358
        simple_pattern = r'([A-Z]+)(\d+)([A-Z]*)'
        match = re.match(simple_pattern, component_upper)
        
        if match and len(recommendations) < max_count:
            prefix, number, suffix = match.groups()
            
            # Suggest similar numbers
            base_num = int(number)
            for offset in [1, -1, 10, -10]:
                new_num = base_num + offset
                if new_num > 0:
                    variant = f"{prefix}{new_num}{suffix}"
                    recommendations.append({
                        "component": variant,
                        "reason": f"Similar component to {component}",
                        "confidence": 0.4,
                        "category": "analog",
                        "recommendation_type": "numerical_variant"
                    })
                    if len(recommendations) >= max_count:
                        break
        
        return recommendations
    
    def _generate_similar_parts(self, base_component: str, family_member: str) -> List[str]:
        """Generate specific part numbers based on family patterns"""
        parts = []
        base_upper = base_component.upper()
        
        # Extract suffix patterns from base component
        import re
        
        if "STM32" in base_upper:
            # STM32 pattern: STM32F429ZIT6
            match = re.search(r'STM32([A-Z]\d+)([A-Z]+)(\d+)', base_upper)
            if match:
                series, variant, package = match.groups()
                # Generate alternatives with different series
                for new_series in ["F4", "F7", "H7", "L4"]:
                    if new_series != series[:2]:
                        parts.append(f"STM32{new_series}29{variant}{package}")
                        
        elif "ATMEGA" in base_upper:
            # ATMEGA pattern
            match = re.search(r'ATMEGA(\d+)([A-Z]*)', base_upper)
            if match:
                number, suffix = match.groups()
                for new_num in ["328", "2560", "32U4"]:
                    if new_num != number:
                        parts.append(f"ATMEGA{new_num}{suffix}")
        
        return parts[:3]  # Limit to 3 alternatives
    
    def _generate_analog_alternatives(self, base_component: str, prefix: str) -> List[str]:
        """Generate analog component alternatives"""
        parts = []
        base_upper = base_component.upper()
        
        # Common analog alternatives
        analog_alternatives = {
            "LM358": ["LM324", "TL072", "TL074", "AD8066"],
            "LM317": ["LM1117", "AMS1117", "LP2950", "LM2596"],
            "LM393": ["LM339", "MAX9061", "AD8561"],
            "LM741": ["TL071", "AD711", "OP07"]
        }
        
        # Check for direct alternatives
        for base, alts in analog_alternatives.items():
            if base in base_upper:
                parts.extend(alts)
                break
        
        # Generate pattern-based alternatives
        import re
        match = re.search(r'([A-Z]+)(\d+)', base_upper)
        if match:
            prefix_part, number = match.groups()
            base_num = int(number)
            
            # Generate numerical variants
            for offset in [1, -1, 10, -10]:
                new_num = base_num + offset
                if new_num > 0:
                    parts.append(f"{prefix_part}{new_num}")
        
        return parts[:4]  # Limit to 4 alternatives
    
    def _get_specification_based_recommendations(
        self, component: str, specifications: Dict[str, Any], max_count: int
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on specifications"""
        recommendations = []
        
        # Simple spec-based logic (would be more sophisticated in production)
        voltage = specifications.get('voltage', '')
        current = specifications.get('current', '')
        
        if '5v' in voltage.lower():
            recommendations.append({
                "component": "LM7805",
                "reason": "5V voltage regulator for power supply",
                "confidence": 0.6,
                "category": "power",
                "recommendation_type": "complementary",
                "specifications": {"output_voltage": "5V", "max_current": "1A"}
            })
        
        if '3.3v' in voltage.lower():
            recommendations.append({
                "component": "AMS1117-3.3",
                "reason": "3.3V voltage regulator for power supply", 
                "confidence": 0.6,
                "category": "power",
                "recommendation_type": "complementary",
                "specifications": {"output_voltage": "3.3V", "max_current": "1A"}
            })
        
        # Add decoupling capacitors for microcontrollers
        if any(mcu in component.upper() for mcu in ['STM32', 'ATMEGA', 'PIC']):
            recommendations.extend([
                {
                    "component": "100nF Ceramic Capacitor",
                    "reason": "Decoupling capacitor for MCU",
                    "confidence": 0.7,
                    "category": "passives",
                    "recommendation_type": "essential",
                    "specifications": {"capacitance": "100nF", "type": "ceramic"}
                },
                {
                    "component": "10uF Electrolytic Capacitor",
                    "reason": "Bulk decoupling capacitor",
                    "confidence": 0.6,
                    "category": "passives", 
                    "recommendation_type": "recommended",
                    "specifications": {"capacitance": "10uF", "type": "electrolytic"}
                }
            ])
        
        return recommendations[:max_count]
    
    def _deduplicate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate recommendations"""
        seen = set()
        unique_recommendations = []
        
        for rec in recommendations:
            component = rec['component'].lower().strip()
            if component not in seen:
                unique_recommendations.append(rec)
                seen.add(component)
        
        return unique_recommendations
    
    async def _filter_by_price(self, recommendations: List[Dict[str, Any]], price_range: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter recommendations by price range using real pricing data"""
        min_price = price_range.get('min', 0)
        max_price = price_range.get('max', float('inf'))
        
        filtered = []
        for rec in recommendations:
            # Get actual price from database if available
            actual_price = await self._get_component_price(rec['component'])
            rec['actual_price'] = actual_price
            rec['price_source'] = 'database' if actual_price else 'estimated'
            
            # Use actual price if available, otherwise estimate
            price_to_check = actual_price if actual_price else self._estimate_component_price(rec['component'])
            rec['price_used_for_filtering'] = price_to_check
            
            if min_price <= price_to_check <= max_price:
                filtered.append(rec)
        
        return filtered
    
    async def _get_component_price(self, component: str) -> Optional[float]:
        """Get actual component price from database"""
        try:
            from ..db.session import get_session
            from ..db.models import PriceHistory, Component
            from sqlalchemy import and_, func, desc
            
            session = next(get_session())
            try:
                # Find component by part number or similar match
                component_obj = session.query(Component).filter(
                    Component.manufacturer_part_number.ilike(f"%{component}%")
                ).first()
                
                if not component_obj:
                    return None
                
                # Get most recent price
                latest_price = session.query(PriceHistory).filter(
                    PriceHistory.component_id == component_obj.id
                ).order_by(desc(PriceHistory.effective_date)).first()
                
                if latest_price:
                    return float(latest_price.unit_price)
                    
            finally:
                session.close()
                
        except Exception as e:
            logger.warning(f"Failed to get component price for {component}: {e}")
            
        return None
    
    def _estimate_component_price(self, component: str) -> float:
        """Enhanced price estimation using multiple factors"""
        component_lower = component.lower()
        base_price = 1.0
        
        # Microcontroller pricing by series and complexity
        if 'stm32' in component_lower:
            if any(series in component_lower for series in ['h7', 'f7']):
                base_price = 12.0  # High-performance series
            elif any(series in component_lower for series in ['f4', 'f3']):
                base_price = 8.0   # Mid-range series
            else:
                base_price = 5.0   # Entry-level series
                
        elif 'atmega' in component_lower:
            if '2560' in component_lower:
                base_price = 6.0
            elif any(num in component_lower for num in ['328', '32u4']):
                base_price = 3.0
            else:
                base_price = 2.0
                
        # Analog ICs by complexity and brand
        elif any(ic in component_lower for ic in ['lm', 'tl']):
            base_price = 1.5   # Basic analog ICs
        elif any(ic in component_lower for ic in ['ad', 'adi']):
            base_price = 4.0   # Analog Devices - premium
        elif 'max' in component_lower:
            base_price = 3.0   # Maxim - mid-premium
        elif 'opa' in component_lower:
            base_price = 2.5   # Precision op-amps
            
        # Passives by package and precision
        elif 'capacitor' in component_lower:
            if 'ceramic' in component_lower:
                base_price = 0.05
            elif 'tantalum' in component_lower:
                base_price = 0.3
            else:
                base_price = 0.1  # Electrolytic default
                
        elif 'resistor' in component_lower:
            if any(precision in component_lower for precision in ['0.1%', '0.01%']):
                base_price = 0.2  # Precision resistors
            else:
                base_price = 0.03 # Standard resistors
                
        elif 'inductor' in component_lower:
            base_price = 0.5
            
        # Power components
        elif any(power in component_lower for power in ['regulator', 'converter']):
            base_price = 3.5
            
        # Memory components
        elif any(mem in component_lower for mem in ['flash', 'eeprom', 'sram']):
            base_price = 2.0
            
        # Connectors and mechanical
        elif any(conn in component_lower for conn in ['connector', 'header', 'socket']):
            base_price = 1.5
            
        return base_price