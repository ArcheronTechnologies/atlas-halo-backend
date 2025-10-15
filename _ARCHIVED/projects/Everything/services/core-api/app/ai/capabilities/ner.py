"""
Named Entity Recognition (NER) Capabilities

Component part number extraction and entity recognition with multiple
fallback strategies for robust performance.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set
import asyncio

from .base import AICapability, CapabilityResult, CapabilityConfig, CapabilityStatus

logger = logging.getLogger(__name__)


class PartNumberExtractor(AICapability):
    """
    Fast regex-based part number extractor with common electronic component patterns.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="part_number_extractor",
            version="1.0.0",
            timeout_seconds=5.0,
            retry_count=1
        )
        super().__init__(config)
        
        # Common electronic part number patterns
        self.patterns = [
            # Standard patterns (manufacturer + part number)
            r'\b[A-Z]{2,6}[0-9]{3,10}[A-Z]{0,3}\b',  # General pattern like STM32F429ZIT6
            r'\b[A-Z]{1,3}[0-9]{2,8}[A-Z]{1,4}[0-9]{0,3}\b',  # Variations
            
            # Specific manufacturer patterns
            r'\bSTM32[A-Z0-9]{6,10}\b',  # STMicroelectronics
            r'\bATMEGA[0-9]{1,4}[A-Z]{1,3}\b',  # Atmel/Microchip
            r'\bPIC[0-9]{2}[A-Z0-9]{2,6}\b',  # Microchip PIC
            r'\bLM[0-9]{3,4}[A-Z]{0,2}\b',  # Linear/analog ICs
            r'\b74[A-Z]{1,3}[0-9]{2,4}\b',  # Logic ICs
            r'\bSN74[A-Z0-9]{4,8}\b',  # Texas Instruments
            r'\bAD[0-9]{3,4}[A-Z]{0,3}\b',  # Analog Devices
            r'\bLTC[0-9]{3,4}[A-Z]{0,2}\b',  # Linear Technology
            r'\bMAX[0-9]{3,5}[A-Z]{0,2}\b',  # Maxim
            
            # Resistor/Capacitor patterns
            r'\b[0-9]{1,3}[KMG]?[RΩ]\b',  # Resistors
            r'\b[0-9]+[np]?[FμuµF]\b',  # Capacitors
            
            # Package-specific patterns
            r'\b[A-Z0-9]{4,12}-(TQFP|QFN|BGA|SOIC|DIP|SOP)[0-9]{0,3}\b',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns]
        
        # Common false positives to filter out
        self.false_positives = {
            'HTTP', 'HTTPS', 'JSON', 'XML', 'HTML', 'CSV', 'PDF', 'ZIP',
            'GET', 'POST', 'PUT', 'DELETE', 'API', 'URL', 'URI', 'UUID',
            'ASCII', 'UTF8', 'UTF16', 'BASE64', 'MD5', 'SHA1', 'SHA256'
        }
    
    async def initialize(self) -> bool:
        """Initialize the extractor"""
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Extract part numbers from text.
        
        Expected payload:
        {
            "text": "Text containing part numbers",
            "max_results": 50,  # Optional
            "min_confidence": 0.5  # Optional
        }
        """
        text = payload.get('text', '')
        max_results = payload.get('max_results', 50)
        min_confidence = payload.get('min_confidence', 0.1)
        
        if not text:
            return CapabilityResult(
                success=False,
                error="No text provided for part number extraction"
            )
        
        try:
            part_numbers = self._extract_part_numbers(text, max_results, min_confidence)
            
            return CapabilityResult(
                success=True,
                data={
                    "part_numbers": part_numbers,
                    "total_found": len(part_numbers),
                    "extraction_method": "regex"
                },
                confidence=0.8  # Regex patterns have good precision but may miss some
            )
            
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Part number extraction failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check by testing extraction on sample text"""
        try:
            test_result = await self.execute({
                "text": "Need STM32F429ZIT6 and LM358 for the circuit"
            })
            return test_result.success and len(test_result.data.get("part_numbers", [])) >= 2
        except:
            return False
    
    def _extract_part_numbers(self, text: str, max_results: int, min_confidence: float) -> List[Dict[str, Any]]:
        """Extract part numbers using regex patterns"""
        matches = []
        seen = set()
        
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                part_number = match.group().strip().upper()
                
                # Skip if already found or is false positive
                if part_number in seen or part_number in self.false_positives:
                    continue
                
                # Calculate confidence based on pattern specificity and length
                confidence = self._calculate_confidence(part_number, pattern)
                
                if confidence >= min_confidence:
                    matches.append({
                        "part_number": part_number,
                        "confidence": confidence,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "context": self._get_context(text, match.start(), match.end())
                    })
                    seen.add(part_number)
                
                if len(matches) >= max_results:
                    break
            
            if len(matches) >= max_results:
                break
        
        # Sort by confidence descending
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches
    
    def _calculate_confidence(self, part_number: str, pattern: re.Pattern) -> float:
        """Calculate confidence score for a part number match"""
        base_confidence = 0.5
        
        # Length bonus (longer part numbers are typically more specific)
        length_bonus = min(0.3, len(part_number) * 0.02)
        
        # Pattern specificity bonus
        pattern_bonus = 0.0
        pattern_str = pattern.pattern
        
        # Higher confidence for manufacturer-specific patterns
        if any(mfg in pattern_str for mfg in ['STM32', 'ATMEGA', 'PIC', 'SN74']):
            pattern_bonus = 0.3
        elif any(generic in pattern_str for generic in ['LM', 'AD', 'MAX']):
            pattern_bonus = 0.2
        
        # Character composition bonus (good mix of letters and numbers)
        alpha_count = sum(1 for c in part_number if c.isalpha())
        digit_count = sum(1 for c in part_number if c.isdigit())
        if alpha_count > 0 and digit_count > 0:
            composition_bonus = 0.1
        else:
            composition_bonus = 0.0
        
        return min(0.95, base_confidence + length_bonus + pattern_bonus + composition_bonus)
    
    def _get_context(self, text: str, start: int, end: int, window: int = 20) -> str:
        """Get context around the matched part number"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()


class ComponentNERCapability(AICapability):
    """
    Advanced NER capability for electronic components with fallback to simpler methods.
    """
    
    def __init__(self):
        config = CapabilityConfig(
            name="component_ner",
            version="1.0.0",
            timeout_seconds=15.0,
            retry_count=2,
            fallback_capability="part_number_extractor"
        )
        super().__init__(config)
        
        self.part_extractor = PartNumberExtractor()
        self.spacy_available = False
        self.nlp_model = None
        
        # Component categories for classification
        self.component_categories = {
            'microcontroller': ['MCU', 'microcontroller', 'processor', 'STM32', 'ATMEGA', 'PIC'],
            'memory': ['RAM', 'ROM', 'FLASH', 'EEPROM', 'memory'],
            'analog': ['amplifier', 'ADC', 'DAC', 'LM', 'OP-AMP', 'analog'],
            'logic': ['logic', '74', 'gate', 'buffer', 'driver'],
            'power': ['regulator', 'converter', 'power', 'LDO', 'switching'],
            'passive': ['resistor', 'capacitor', 'inductor', 'R', 'C', 'L'],
            'connector': ['connector', 'header', 'socket', 'plug'],
            'sensor': ['sensor', 'temperature', 'pressure', 'accelerometer']
        }
    
    async def initialize(self) -> bool:
        """Initialize NER models with fallback"""
        try:
            # Try to initialize part extractor first
            await self.part_extractor.initialize()
            
            # Try to load spaCy model if available
            try:
                import spacy
                self.nlp_model = spacy.load("en_core_web_sm")
                self.spacy_available = True
                logger.info("spaCy model loaded successfully")
            except (ImportError, OSError):
                logger.warning("spaCy not available, using regex-only fallback")
                self.spacy_available = False
            
            await self.update_status(CapabilityStatus.HEALTHY)
            return True
            
        except Exception as e:
            logger.error(f"ComponentNER initialization failed: {e}")
            await self.update_status(CapabilityStatus.DEGRADED)
            return False
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Extract component entities from text.
        
        Expected payload:
        {
            "text": "Text containing component references",
            "extract_categories": true,  # Whether to classify components
            "max_results": 50
        }
        """
        text = payload.get('text', '')
        extract_categories = payload.get('extract_categories', True)
        max_results = payload.get('max_results', 50)
        
        if not text:
            return CapabilityResult(
                success=False,
                error="No text provided for component NER"
            )
        
        try:
            # Always use part number extraction as the base
            part_result = await self.part_extractor.execute({
                "text": text,
                "max_results": max_results
            })
            
            entities = []
            
            if part_result.success:
                part_numbers = part_result.data.get("part_numbers", [])
                
                # Enhance with categories if requested
                for pn in part_numbers:
                    entity = {
                        "text": pn["part_number"],
                        "label": "COMPONENT",
                        "confidence": pn["confidence"],
                        "start_pos": pn["start_pos"],
                        "end_pos": pn["end_pos"],
                        "context": pn["context"]
                    }
                    
                    if extract_categories:
                        category = self._classify_component(pn["part_number"], pn["context"])
                        entity["category"] = category
                    
                    entities.append(entity)
            
            # If spaCy is available, also extract general technical terms
            if self.spacy_available and self.nlp_model:
                spacy_entities = await self._extract_with_spacy(text)
                entities.extend(spacy_entities)
            
            # Remove duplicates and sort by confidence
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x['confidence'], reverse=True)
            
            return CapabilityResult(
                success=True,
                data={
                    "entities": entities[:max_results],
                    "total_found": len(entities),
                    "methods_used": ["regex"] + (["spacy"] if self.spacy_available else [])
                },
                confidence=0.85 if self.spacy_available else 0.75
            )
            
        except Exception as e:
            return CapabilityResult(
                success=False,
                error=f"Component NER failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Health check"""
        try:
            test_result = await self.execute({
                "text": "Circuit needs STM32F429ZIT6 microcontroller and LM358 op-amp"
            })
            return test_result.success and len(test_result.data.get("entities", [])) >= 2
        except:
            return False
    
    async def _extract_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NLP model"""
        if not self.spacy_available:
            return []
        
        try:
            # Run spaCy processing in thread pool to avoid blocking
            import asyncio
            import concurrent.futures
            
            def process_text():
                doc = self.nlp_model(text)
                entities = []
                
                for ent in doc.ents:
                    # Focus on technical/product entities
                    if ent.label_ in ['PRODUCT', 'ORG', 'MONEY', 'QUANTITY']:
                        # Check if it looks technical
                        if self._is_technical_entity(ent.text):
                            entities.append({
                                "text": ent.text,
                                "label": f"TECH_{ent.label_}",
                                "confidence": 0.6,  # Lower confidence for general NER
                                "start_pos": ent.start_char,
                                "end_pos": ent.end_char,
                                "context": text[max(0, ent.start_char-20):ent.end_char+20].strip()
                            })
                
                return entities
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(process_text)
                return await asyncio.wrap_future(future)
                
        except Exception as e:
            logger.warning(f"spaCy processing failed: {e}")
            return []
    
    def _classify_component(self, part_number: str, context: str) -> str:
        """Classify component type based on part number and context"""
        combined_text = f"{part_number} {context}".lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.component_categories.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    # Give higher weight to part number matches
                    if keyword.lower() in part_number.lower():
                        score += 2
                    else:
                        score += 1
            category_scores[category] = score
        
        # Return highest scoring category, or 'unknown' if no matches
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0] if best_category[1] > 0 else 'unknown'
        
        return 'unknown'
    
    def _is_technical_entity(self, text: str) -> bool:
        """Check if entity looks technical/electronic"""
        text_lower = text.lower()
        
        # Technical indicators
        technical_words = [
            'circuit', 'board', 'chip', 'module', 'sensor', 'controller',
            'processor', 'memory', 'voltage', 'current', 'frequency',
            'amplifier', 'regulator', 'converter', 'interface'
        ]
        
        return (
            any(word in text_lower for word in technical_words) or
            bool(re.search(r'[0-9]+[vmhzaf]', text_lower)) or  # Units like 5V, 10MHz
            len(re.findall(r'[A-Z0-9]', text)) > len(text) * 0.3  # High ratio of caps/digits
        )
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on text similarity"""
        unique_entities = []
        seen_texts = set()
        
        for entity in entities:
            text = entity['text'].lower().strip()
            if text not in seen_texts:
                unique_entities.append(entity)
                seen_texts.add(text)
        
        return unique_entities