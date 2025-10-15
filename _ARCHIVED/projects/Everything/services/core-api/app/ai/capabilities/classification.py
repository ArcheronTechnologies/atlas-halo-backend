"""
Classification Capabilities

Intent classification and category classification with rule-based fallbacks.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from collections import Counter

from .base import AICapability, CapabilityResult, CapabilityConfig, CapabilityStatus

logger = logging.getLogger(__name__)


class IntentClassifier(AICapability):
    """
    Rule-based intent classifier for supply chain and procurement emails/messages.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="intent_classifier",
            version="1.0.0",
            timeout_seconds=10.0,
            retry_count=1
        )
        super().__init__(config)
        
        # Intent patterns with weights
        self.intent_patterns = {
            'rfq': {
                'patterns': [
                    r'\b(rfq|request for quote|quotation|quote request)\b',
                    r'\b(pricing|price list|cost estimate)\b',
                    r'\b(bid|tender|proposal)\b',
                    r'\b(quantity|qty|pieces|units)\s+\d+',
                    r'\b(delivery|lead time|availability)\b'
                ],
                'keywords': ['quote', 'pricing', 'quantity', 'delivery', 'specification'],
                'indicators': ['?', 'need', 'require', 'looking for']
            },
            'purchase_order': {
                'patterns': [
                    r'\b(purchase order|po|order number)\b',
                    r'\b(order|buy|purchase)\b',
                    r'\b(confirmation|confirm)\b',
                    r'\bpo[-\s#]*\d+\b'
                ],
                'keywords': ['order', 'purchase', 'buy', 'confirmation'],
                'indicators': ['confirming', 'ordering', 'placing order']
            },
            'technical_inquiry': {
                'patterns': [
                    r'\b(specification|datasheet|technical|specs)\b',
                    r'\b(compatibility|compatible|substitute)\b',
                    r'\b(pinout|package|footprint)\b',
                    r'\b(operating|voltage|current|temperature)\b'
                ],
                'keywords': ['technical', 'datasheet', 'specifications', 'compatibility'],
                'indicators': ['how', 'what', 'which', 'specification']
            },
            'availability_check': {
                'patterns': [
                    r'\b(availability|available|in stock|stock level)\b',
                    r'\b(lead time|delivery time|shipping)\b',
                    r'\b(when|eta|expected)\b'
                ],
                'keywords': ['availability', 'stock', 'delivery', 'when'],
                'indicators': ['available?', 'in stock?', 'delivery time']
            },
            'complaint': {
                'patterns': [
                    r'\b(complaint|issue|problem|defective|faulty)\b',
                    r'\b(return|rma|warranty|claim)\b',
                    r'\b(damaged|broken|not working)\b'
                ],
                'keywords': ['complaint', 'issue', 'problem', 'defective', 'return'],
                'indicators': ['problem with', 'not working', 'defective']
            },
            'support': {
                'patterns': [
                    r'\b(help|support|assistance|question)\b',
                    r'\b(how to|tutorial|guide|manual)\b',
                    r'\b(troubleshoot|debug|fix)\b'
                ],
                'keywords': ['help', 'support', 'question', 'how'],
                'indicators': ['need help', 'how to', 'can you help']
            },
            'shipping': {
                'patterns': [
                    r'\b(shipping|shipment|delivery|transport)\b',
                    r'\b(tracking|track|status)\b',
                    r'\b(received|delivered|arrived)\b'
                ],
                'keywords': ['shipping', 'delivery', 'tracking', 'received'],
                'indicators': ['shipped?', 'tracking number', 'delivery status']
            }
        }
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.5
    
    async def initialize(self) -> bool:
        """Initialize the classifier"""
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Classify intent from text.
        
        Expected payload:
        {
            "text": "Text to classify",
            "subject": "Optional email subject",  # Optional, gives extra context
            "top_k": 3  # Return top K intents
        }
        """
        text = payload.get('text', '')
        subject = payload.get('subject', '')
        top_k = payload.get('top_k', 3)
        
        if not text:
            return CapabilityResult(
                success=False,
                error="No text provided for intent classification"
            )
        
        try:
            # Combine text and subject for analysis
            combined_text = f"{subject} {text}".lower()
            
            # Score each intent
            intent_scores = self._score_intents(combined_text)
            
            # Get top K intents
            top_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Determine primary intent and confidence
            if top_intents:
                primary_intent, primary_score = top_intents[0]
                confidence = min(0.95, primary_score)
                
                # Adjust confidence based on score distribution
                if len(top_intents) > 1:
                    second_score = top_intents[1][1]
                    if primary_score - second_score < 0.1:  # Close scores = less confidence
                        confidence *= 0.8
                
                result_data = {
                    "primary_intent": primary_intent,
                    "confidence": confidence,
                    "all_intents": [
                        {"intent": intent, "score": score} 
                        for intent, score in top_intents
                    ],
                    "classification_method": "rule_based"
                }
                
                # Add explanation for high-confidence predictions
                if confidence >= self.high_confidence_threshold:
                    result_data["explanation"] = self._explain_classification(primary_intent, combined_text)
                
                return CapabilityResult(
                    success=True,
                    data=result_data,
                    confidence=confidence
                )
            else:
                return CapabilityResult(
                    success=True,
                    data={
                        "primary_intent": "unknown",
                        "confidence": 0.1,
                        "all_intents": [],
                        "classification_method": "rule_based"
                    },
                    confidence=0.1,
                    warnings=["No intent patterns matched"]
                )
                
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Intent classification failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check with sample classification"""
        try:
            test_cases = [
                "I need a quote for 100 pieces of STM32F429ZIT6",
                "What is the datasheet for this component?",
                "Is this part available in stock?"
            ]
            
            for text in test_cases:
                result = await self.execute({"text": text})
                if not result.success or result.data["primary_intent"] == "unknown":
                    return False
            
            return True
        except:
            return False
    
    def _score_intents(self, text: str) -> Dict[str, float]:
        """Score all intents for the given text"""
        intent_scores = {}
        
        for intent_name, intent_config in self.intent_patterns.items():
            score = 0.0
            matched_patterns = 0
            
            # Pattern matching (highest weight)
            for pattern in intent_config['patterns']:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                if matches > 0:
                    score += matches * 0.4
                    matched_patterns += 1
            
            # Keyword matching (medium weight)
            keyword_matches = 0
            for keyword in intent_config['keywords']:
                if keyword.lower() in text:
                    score += 0.2
                    keyword_matches += 1
            
            # Indicator matching (lower weight but important for context)
            indicator_matches = 0
            for indicator in intent_config['indicators']:
                if indicator.lower() in text:
                    score += 0.1
                    indicator_matches += 1
            
            # Bonus for multiple types of matches
            match_types = sum([
                1 if matched_patterns > 0 else 0,
                1 if keyword_matches > 0 else 0,
                1 if indicator_matches > 0 else 0
            ])
            
            if match_types >= 2:
                score *= 1.2  # 20% bonus for multiple match types
            
            # Normalize score
            max_possible_score = len(intent_config['patterns']) * 0.4 + len(intent_config['keywords']) * 0.2 + len(intent_config['indicators']) * 0.1
            if max_possible_score > 0:
                score = min(1.0, score / max_possible_score)
            
            intent_scores[intent_name] = score
        
        return intent_scores
    
    def _explain_classification(self, intent: str, text: str) -> str:
        """Provide explanation for the classification"""
        if intent not in self.intent_patterns:
            return "Classification based on unknown patterns"
        
        config = self.intent_patterns[intent]
        matched_elements = []
        
        # Find what matched
        for pattern in config['patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                matched_elements.append(f"pattern '{pattern}'")
        
        for keyword in config['keywords']:
            if keyword.lower() in text:
                matched_elements.append(f"keyword '{keyword}'")
        
        if matched_elements:
            return f"Classified as '{intent}' based on: {', '.join(matched_elements[:3])}"
        else:
            return f"Classified as '{intent}' with low confidence"


class CategoryClassifier(AICapability):
    """
    Component category classifier using hierarchical rules and patterns.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="category_classifier",
            version="1.0.0",
            timeout_seconds=8.0,
            retry_count=1
        )
        super().__init__(config)
        
        # Dynamic category structure - can be loaded from database/config
        self.category_hierarchy = self._load_category_hierarchy()
        self.pattern_cache = {}
        self.learning_enabled = True
        
        # Flatten patterns for faster matching
        self._flat_patterns = self._flatten_patterns()
    
    async def initialize(self) -> bool:
        """Initialize the classifier"""
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Classify component category.
        
        Expected payload:
        {
            "text": "Component description or part number",
            "part_number": "Optional specific part number",  # Optional
            "context": "Optional additional context",  # Optional
            "include_subcategories": true  # Include subcategory classification
        }
        """
        text = payload.get('text', '')
        part_number = payload.get('part_number', '')
        context = payload.get('context', '')
        include_subcategories = payload.get('include_subcategories', True)
        
        if not text and not part_number:
            return CapabilityResult(
                success=False,
                error="No text or part number provided for classification"
            )
        
        try:
            # Combine all available text
            combined_text = f"{text} {part_number} {context}".lower().strip()
            
            # Classify using hierarchical matching
            categories = self._classify_hierarchical(combined_text, include_subcategories)
            
            if categories:
                primary_category = categories[0]
                confidence = primary_category['confidence']
                
                return CapabilityResult(
                    success=True,
                    data={
                        "primary_category": primary_category['category'],
                        "subcategory": primary_category.get('subcategory'),
                        "confidence": confidence,
                        "all_categories": categories,
                        "classification_method": "hierarchical_rules"
                    },
                    confidence=confidence
                )
            else:
                return CapabilityResult(
                    success=True,
                    data={
                        "primary_category": "unknown",
                        "subcategory": None,
                        "confidence": 0.1,
                        "all_categories": [],
                        "classification_method": "hierarchical_rules"
                    },
                    confidence=0.1,
                    warnings=["No category patterns matched"]
                )
                
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Category classification failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check with sample classifications"""
        try:
            test_cases = [
                {"text": "STM32F429ZIT6 microcontroller"},
                {"text": "10uF capacitor ceramic"},
                {"text": "LM358 operational amplifier"}
            ]
            
            for case in test_cases:
                result = await self.execute(case)
                if not result.success or result.data["primary_category"] == "unknown":
                    return False
            
            return True
        except:
            return False
    
    def _load_category_hierarchy(self) -> Dict[str, Any]:
        """Load category hierarchy from database or config with fallback"""
        try:
            # Try to load from database first
            hierarchy = self._load_from_database()
            if hierarchy:
                return hierarchy
        except Exception as e:
            logger.warning(f"Failed to load categories from database: {e}")
        
        # Fallback to base categories with expansion capability
        return {
            'semiconductors': {
                'microcontrollers': {
                    'patterns': [r'STM32', r'ATMEGA', r'PIC\d+', r'microcontroller', r'MCU', r'ESP32', r'MSP430'],
                    'subcategories': ['arm_cortex', '8bit_mcu', '16bit_mcu', '32bit_mcu', 'risc_v'],
                    'expandable': True
                },
                'memory': {
                    'patterns': [r'EEPROM', r'FLASH', r'SRAM', r'DRAM', r'memory', r'DDR\d+', r'LPDDR'],
                    'subcategories': ['volatile', 'non_volatile', 'cache', 'embedded'],
                    'expandable': True
                },
                'analog': {
                    'patterns': [r'LM\d+', r'AD\d+', r'MAX\d+', r'amplifier', r'ADC', r'DAC', r'TL\d+', r'OPA\d+'],
                    'subcategories': ['op_amps', 'data_converters', 'power_management', 'references'],
                    'expandable': True
                },
                'logic': {
                    'patterns': [r'74\w+', r'logic', r'gate', r'buffer', r'driver', r'SN74', r'HC\d+'],
                    'subcategories': ['ttl', 'cmos', 'buffers', 'gates', 'translators'],
                    'expandable': True
                },
                'power': {
                    'patterns': [r'regulator', r'converter', r'LDO', r'SMPS', r'switching', r'linear'],
                    'subcategories': ['linear_reg', 'switching_reg', 'dc_dc', 'ac_dc'],
                    'expandable': True
                }
            },
            'passives': {
                'resistors': {
                    'patterns': [r'\d+[KMG]?[RΩ]', r'resistor', r'ohm', r'potentiometer', r'rheostat'],
                    'subcategories': ['fixed', 'variable', 'precision', 'power', 'current_sense'],
                    'expandable': True
                },
                'capacitors': {
                    'patterns': [r'\d+[pnμumF]F', r'capacitor', r'cap', r'ceramic', r'electrolytic', r'tantalum'],
                    'subcategories': ['ceramic', 'electrolytic', 'tantalum', 'film', 'supercap'],
                    'expandable': True
                },
                'inductors': {
                    'patterns': [r'\d+[μmH]H', r'inductor', r'choke', r'coil', r'transformer'],
                    'subcategories': ['power', 'rf', 'common_mode', 'ferrite'],
                    'expandable': True
                }
            },
            'connectors': {
                'headers': {
                    'patterns': [r'header', r'pin header', r'socket', r'connector'],
                    'subcategories': ['male', 'female', 'right_angle', 'board_to_board'],
                    'expandable': True
                },
                'cables': {
                    'patterns': [r'cable', r'wire', r'harness', r'jumper'],
                    'subcategories': ['ribbon', 'coaxial', 'twisted_pair', 'fiber'],
                    'expandable': True
                },
                'interfaces': {
                    'patterns': [r'USB', r'HDMI', r'ethernet', r'RJ45', r'DB\d+'],
                    'subcategories': ['usb', 'hdmi', 'ethernet', 'serial', 'audio'],
                    'expandable': True
                }
            },
            'mechanical': {
                'enclosures': {
                    'patterns': [r'enclosure', r'case', r'box', r'housing', r'chassis'],
                    'subcategories': ['plastic', 'metal', 'rack_mount', 'waterproof'],
                    'expandable': True
                },
                'hardware': {
                    'patterns': [r'screw', r'standoff', r'washer', r'nut', r'bolt'],
                    'subcategories': ['fasteners', 'spacers', 'mounts', 'thermal'],
                    'expandable': True
                }
            },
            'sensors': {
                'temperature': {
                    'patterns': [r'temperature', r'thermal', r'thermocouple', r'thermistor', r'DS18B20'],
                    'subcategories': ['analog', 'digital', 'contact', 'non_contact'],
                    'expandable': True
                },
                'motion': {
                    'patterns': [r'accelerometer', r'gyroscope', r'IMU', r'motion', r'vibration'],
                    'subcategories': ['accelerometer', 'gyroscope', 'imu', 'magnetometer'],
                    'expandable': True
                },
                'environmental': {
                    'patterns': [r'pressure', r'humidity', r'gas', r'pH', r'optical'],
                    'subcategories': ['pressure', 'humidity', 'gas', 'light', 'proximity'],
                    'expandable': True
                }
            },
            'displays': {
                'lcd': {
                    'patterns': [r'LCD', r'display', r'screen', r'TFT', r'character'],
                    'subcategories': ['character', 'graphic', 'tft', 'oled'],
                    'expandable': True
                },
                'led': {
                    'patterns': [r'LED', r'light', r'RGB', r'strip', r'matrix'],
                    'subcategories': ['single', 'rgb', 'matrix', 'strip'],
                    'expandable': True
                }
            }
        }
    
    def _load_from_database(self) -> Optional[Dict[str, Any]]:
        """Load category hierarchy from database"""
        try:
            from ..db.session import get_session
            from ..db.models import ComponentCategory
            from sqlalchemy import func
            
            session = next(get_session())
            try:
                # Get all categories with their patterns and usage counts
                categories = session.query(
                    ComponentCategory.main_category,
                    ComponentCategory.subcategory,
                    func.count().label('usage_count')
                ).group_by(
                    ComponentCategory.main_category,
                    ComponentCategory.subcategory
                ).all()
                
                hierarchy = {}
                for cat in categories:
                    main_cat = cat.main_category
                    sub_cat = cat.subcategory or 'general'
                    
                    if main_cat not in hierarchy:
                        hierarchy[main_cat] = {}
                    
                    if sub_cat not in hierarchy[main_cat]:
                        # Generate patterns based on actual component data
                        patterns = self._generate_patterns_for_category(session, main_cat, sub_cat)
                        hierarchy[main_cat][sub_cat] = {
                            'patterns': patterns,
                            'subcategories': [sub_cat],
                            'usage_count': cat.usage_count,
                            'expandable': True
                        }
                
                return hierarchy if hierarchy else None
                
            finally:
                session.close()
                
        except Exception as e:
            logger.warning(f"Database category loading failed: {e}")
            return None
    
    def _generate_patterns_for_category(self, session, main_category: str, subcategory: str) -> List[str]:
        """Generate patterns based on actual component data in the database"""
        try:
            from ..db.models import Component
            import re
            
            # Get sample components from this category
            components = session.query(Component.manufacturer_part_number).filter(
                Component.category == main_category,
                Component.subcategory == subcategory
            ).limit(50).all()
            
            patterns = set()
            
            for comp in components:
                part_number = comp.manufacturer_part_number
                if part_number:
                    # Extract common prefixes and patterns
                    prefix_match = re.match(r'^([A-Z]+)', part_number.upper())
                    if prefix_match:
                        prefix = prefix_match.group(1)
                        if len(prefix) >= 2:
                            patterns.add(prefix)
                    
                    # Extract numeric patterns
                    if re.search(r'\d+', part_number):
                        patterns.add(r'\d+')
            
            # Add category-specific keywords
            category_keywords = {
                'microcontrollers': ['MCU', 'micro', 'controller'],
                'memory': ['memory', 'RAM', 'ROM'],
                'analog': ['amplifier', 'amp', 'analog'],
                'power': ['regulator', 'power', 'supply'],
                'sensors': ['sensor', 'detect']
            }
            
            if subcategory in category_keywords:
                patterns.update(category_keywords[subcategory])
            
            return list(patterns)[:10]  # Limit to 10 patterns
            
        except Exception as e:
            logger.warning(f"Pattern generation failed: {e}")
            return []
    
    def _flatten_patterns(self) -> List[Tuple[str, str, str, str]]:
        """Flatten hierarchy into (main_cat, sub_cat, pattern, full_path) tuples"""
        patterns = []
        
        for main_cat, main_config in self.category_hierarchy.items():
            for sub_cat, sub_config in main_config.items():
                if isinstance(sub_config, dict) and 'patterns' in sub_config:
                    for pattern in sub_config['patterns']:
                        patterns.append((main_cat, sub_cat, pattern, f"{main_cat}.{sub_cat}"))
        
        return patterns
    
    async def _learn_from_classification(self, text: str, classified_category: str, confidence: float):
        """Learn and update patterns from successful classifications"""
        if not self.learning_enabled or confidence < 0.7:
            return
        
        try:
            # Extract potential new patterns from the text
            import re
            
            # Look for manufacturer prefixes
            prefixes = re.findall(r'\b[A-Z]{2,6}(?=\d)', text.upper())
            
            # Look for part number patterns
            part_patterns = re.findall(r'\b[A-Z0-9]{4,15}\b', text.upper())
            
            # Update category hierarchy if new patterns found
            main_cat, sub_cat = classified_category.split('.') if '.' in classified_category else (classified_category, 'general')
            
            if main_cat in self.category_hierarchy and sub_cat in self.category_hierarchy[main_cat]:
                config = self.category_hierarchy[main_cat][sub_cat]
                if config.get('expandable', False):
                    existing_patterns = set(config['patterns'])
                    new_patterns = set(prefixes + part_patterns[:3])  # Limit new patterns
                    
                    # Add new patterns that aren't too generic
                    for pattern in new_patterns:
                        if len(pattern) >= 3 and pattern not in existing_patterns:
                            config['patterns'].append(pattern)
                            logger.debug(f"Learned new pattern '{pattern}' for {classified_category}")
                            
                            # Limit total patterns to prevent bloat
                            if len(config['patterns']) > 20:
                                config['patterns'] = config['patterns'][-20:]
        
        except Exception as e:
            logger.warning(f"Pattern learning failed: {e}")
    
    def _classify_hierarchical(self, text: str, include_subcategories: bool) -> List[Dict[str, Any]]:
        """Classify using hierarchical pattern matching"""
        category_scores = {}
        
        # Score all patterns
        for main_cat, sub_cat, pattern, full_path in self._flat_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                # Calculate score based on pattern specificity and matches
                pattern_specificity = len(pattern) / 20.0  # Longer patterns are more specific
                match_score = min(1.0, matches * 0.3)
                total_score = (pattern_specificity + match_score) / 2
                
                if full_path not in category_scores:
                    category_scores[full_path] = {
                        'main_category': main_cat,
                        'subcategory': sub_cat,
                        'score': 0,
                        'matched_patterns': []
                    }
                
                category_scores[full_path]['score'] += total_score
                category_scores[full_path]['matched_patterns'].append(pattern)
        
        # Convert to sorted list
        categories = []
        for path, data in category_scores.items():
            # Normalize score
            max_score = len(data['matched_patterns']) * 1.0
            normalized_score = min(0.95, data['score'] / max_score if max_score > 0 else 0)
            
            category_info = {
                'category': data['main_category'],
                'confidence': normalized_score,
                'matched_patterns': data['matched_patterns']
            }
            
            if include_subcategories:
                category_info['subcategory'] = data['subcategory']
            
            categories.append(category_info)
        
        # Sort by confidence
        categories.sort(key=lambda x: x['confidence'], reverse=True)
        
        return categories