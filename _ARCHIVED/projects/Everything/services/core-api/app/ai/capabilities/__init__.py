"""
AI Capabilities Framework

Modular, composable AI capabilities with orchestration, fallbacks, and observability.
"""

import logging
import asyncio
from typing import Dict, Any, List

from .base import AICapability, CapabilityResult, CapabilityConfig, CapabilityStatus
from .registry import CapabilityRegistry, capability_registry
from .orchestrator import CapabilityOrchestrator, Pipeline, PipelineConfig, PipelineStep
from .ner import ComponentNERCapability, PartNumberExtractor
from .classification import IntentClassifier, CategoryClassifier
from .recommendation import ComponentRecommender
from .forecasting import PriceForecastCapability, DemandForecastCapability
from .scenario import ScenarioAnalysisCapability
from .supplier_analysis import AdvancedSupplierAnalysis
from .evaluation import CapabilityEvaluator, TestFixtures, EvaluationSuite
from .observability import observability_manager, ObservabilityManager

logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator = CapabilityOrchestrator(capability_registry)

async def initialize_capabilities() -> Dict[str, bool]:
    """
    Initialize all AI capabilities and register them.
    
    Returns:
        Dict mapping capability names to initialization success
    """
    logger.info("Initializing AI capabilities framework...")
    
    # Create capability instances
    capabilities = [
        PartNumberExtractor(),
        ComponentNERCapability(),
        IntentClassifier(),
        CategoryClassifier(), 
        ComponentRecommender(),
        PriceForecastCapability(),
        DemandForecastCapability(),
        ScenarioAnalysisCapability(),
        AdvancedSupplierAnalysis()
    ]
    
    # Register capabilities with dependencies and observability
    for capability in capabilities:
        observability_manager.register_capability(capability)
    
    capability_registry.register(
        capabilities[0],  # PartNumberExtractor
        priority=10,
        tags={'ner', 'extraction', 'fast'}
    )
    
    capability_registry.register(
        capabilities[1],  # ComponentNERCapability
        dependencies=['part_number_extractor'],
        priority=20,
        tags={'ner', 'extraction', 'advanced'}
    )
    
    capability_registry.register(
        capabilities[2],  # IntentClassifier
        priority=30,
        tags={'classification', 'intent'}
    )
    
    capability_registry.register(
        capabilities[3],  # CategoryClassifier
        priority=30,
        tags={'classification', 'category'}
    )
    
    capability_registry.register(
        capabilities[4],  # ComponentRecommender
        dependencies=['category_classifier'],
        priority=50,
        tags={'recommendation', 'ml'}
    )
    
    capability_registry.register(
        capabilities[5],  # PriceForecastCapability
        priority=60,
        tags={'forecasting', 'price', 'ml'}
    )
    
    capability_registry.register(
        capabilities[6],  # DemandForecastCapability
        priority=60,
        tags={'forecasting', 'demand', 'ml'}
    )
    
    capability_registry.register(
        capabilities[7],  # ScenarioAnalysisCapability
        priority=40,
        tags={'scenario', 'risk', 'analysis'}
    )
    
    capability_registry.register(
        capabilities[8],  # AdvancedSupplierAnalysis
        priority=45,
        tags={'supplier', 'analysis', 'scoring'}
    )
    
    # Register common pipelines
    _register_common_pipelines()
    
    # Initialize all capabilities
    results = await capability_registry.initialize_all()
    
    # Log summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    logger.info(f"AI capabilities initialization complete: {successful}/{total} successful")
    
    return results

def _register_common_pipelines():
    """Register common capability pipelines"""
    
    # Email processing pipeline
    email_pipeline = PipelineConfig(
        name="email_processing",
        version="1.0.0",
        steps=[
            PipelineStep(
                capability_name="component_ner",
                input_mapping={"text": "email_body", "subject": "email_subject"},
                output_mapping={"entities": "extracted_components"},
                parallel_group="extraction"
            ),
            PipelineStep(
                capability_name="intent_classifier", 
                input_mapping={"text": "email_body", "subject": "email_subject"},
                output_mapping={"primary_intent": "email_intent"},
                parallel_group="extraction"
            ),
            PipelineStep(
                capability_name="component_recommender",
                input_mapping={"component": "primary_component"},
                output_mapping={"recommendations": "suggested_components"},
                condition="len(extracted_components) > 0",
                required=False
            )
        ],
        timeout_seconds=60.0,
        parallel_execution=True
    )
    
    # RFQ processing pipeline  
    rfq_pipeline = PipelineConfig(
        name="rfq_processing",
        version="1.0.0",
        steps=[
            PipelineStep(
                capability_name="component_ner",
                input_mapping={"text": "rfq_description"},
                output_mapping={"entities": "rfq_components"}
            ),
            PipelineStep(
                capability_name="category_classifier",
                input_mapping={"text": "component_text"},
                output_mapping={"primary_category": "component_category"},
                parallel_group="classification"
            ),
            PipelineStep(
                capability_name="price_forecast",
                input_mapping={"component_id": "component_id"},
                output_mapping={"forecast": "price_forecast"},
                parallel_group="forecasting",
                required=False
            ),
            PipelineStep(
                capability_name="demand_forecast",
                input_mapping={"component_id": "component_id"},
                output_mapping={"forecast": "demand_forecast"},
                parallel_group="forecasting", 
                required=False
            ),
            PipelineStep(
                capability_name="component_recommender",
                input_mapping={"component": "primary_component", "category": "component_category"},
                output_mapping={"recommendations": "alternative_components"}
            )
        ],
        timeout_seconds=90.0,
        parallel_execution=True
    )
    
    # Register pipelines
    orchestrator.register_pipeline(email_pipeline)
    orchestrator.register_pipeline(rfq_pipeline)
    
    logger.info("Registered common capability pipelines")

async def cleanup_capabilities():
    """Cleanup all capabilities"""
    logger.info("Shutting down AI capabilities...")
    await capability_registry.shutdown_all()

# Convenience functions for common operations
async def extract_components(text: str, **kwargs) -> CapabilityResult:
    """Extract components from text using the best available NER capability"""
    return await orchestrator.execute_capability(
        "component_ner", 
        {"text": text, **kwargs},
        fallback_name="part_number_extractor"
    )

async def classify_intent(text: str, subject: str = "", **kwargs) -> CapabilityResult:
    """Classify intent from text"""
    return await orchestrator.execute_capability(
        "intent_classifier",
        {"text": text, "subject": subject, **kwargs}
    )

async def get_recommendations(component: str, **kwargs) -> CapabilityResult:
    """Get component recommendations"""
    return await orchestrator.execute_capability(
        "component_recommender",
        {"component": component, **kwargs}
    )

async def analyze_scenario(scenario_type: str, parameters: Dict[str, Any], affected_components: List[str], **kwargs) -> CapabilityResult:
    """Analyze supply chain scenario impacts"""
    return await orchestrator.execute_capability(
        "scenario_analysis",
        {
            "scenario_type": scenario_type,
            "parameters": parameters, 
            "affected_components": affected_components,
            **kwargs
        }
    )

async def process_email(email_body: str, email_subject: str = "") -> Dict[str, Any]:
    """Process email using the email processing pipeline"""
    result = await orchestrator.execute_pipeline(
        "email_processing",
        {"email_body": email_body, "email_subject": email_subject}
    )
    return result.data if result.success else {}

async def process_rfq(rfq_description: str, **kwargs) -> Dict[str, Any]:
    """Process RFQ using the RFQ processing pipeline"""
    result = await orchestrator.execute_pipeline(
        "rfq_processing", 
        {"rfq_description": rfq_description, **kwargs}
    )
    return result.data if result.success else {}

__all__ = [
    'AICapability',
    'CapabilityResult', 
    'CapabilityConfig',
    'CapabilityStatus',
    'CapabilityRegistry',
    'capability_registry',
    'CapabilityOrchestrator',
    'orchestrator',
    'Pipeline',
    'PipelineConfig',
    'PipelineStep',
    'ComponentNERCapability',
    'PartNumberExtractor',
    'IntentClassifier',
    'CategoryClassifier',
    'ComponentRecommender',
    'PriceForecastCapability',
    'DemandForecastCapability',
    'ScenarioAnalysisCapability',
    'AdvancedSupplierAnalysis',
    'CapabilityEvaluator',
    'TestFixtures', 
    'EvaluationSuite',
    'observability_manager',
    'ObservabilityManager',
    'initialize_capabilities',
    'cleanup_capabilities',
    'extract_components',
    'classify_intent',
    'get_recommendations',
    'analyze_scenario',
    'process_email',
    'process_rfq'
]