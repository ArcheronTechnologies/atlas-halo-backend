#!/usr/bin/env python3
"""
Quick test script for the AI capabilities framework.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_capabilities():
    """Test the AI capabilities framework"""
    print("Testing AI Capabilities Framework...")
    
    try:
        # Import capabilities
        from app.ai.capabilities import (
            initialize_capabilities, 
            extract_components,
            classify_intent,
            get_recommendations,
            analyze_scenario,
            process_email,
            orchestrator,
            observability_manager,
            CapabilityEvaluator,
            capability_registry
        )
        
        print("✓ Successfully imported capabilities")
        
        # Initialize capabilities
        print("\nInitializing capabilities...")
        results = await initialize_capabilities()
        print(f"Initialization results: {results}")
        
        # Test component extraction
        print("\n--- Testing Component Extraction ---")
        test_text = "I need a quote for 100 pieces of STM32F429ZIT6 microcontroller and LM358 op-amp"
        result = await extract_components(test_text)
        print(f"NER Result: {result.success}")
        if result.success:
            entities = result.data.get("entities", [])
            print(f"Found {len(entities)} entities: {[e.get('text', '') for e in entities[:3]]}")
        
        # Test intent classification
        print("\n--- Testing Intent Classification ---")
        result = await classify_intent(test_text, "RFQ Request")
        print(f"Intent Result: {result.success}")
        if result.success:
            intent = result.data.get("primary_intent", "unknown")
            confidence = result.data.get("confidence", 0)
            print(f"Detected intent: {intent} (confidence: {confidence:.2f})")
        
        # Test recommendations
        print("\n--- Testing Recommendations ---")
        result = await get_recommendations("STM32F429ZIT6")
        print(f"Recommendation Result: {result.success}")
        if result.success:
            recs = result.data.get("recommendations", [])
            print(f"Found {len(recs)} recommendations: {[r.get('component', '') for r in recs[:3]]}")
        
        # Test scenario analysis
        print("\n--- Testing Scenario Analysis ---")
        scenario_result = await analyze_scenario(
            "geopolitical",
            {
                "scenario_type": "geopolitical",
                "geographic_scope": ["APAC"],
                "industry_sectors": ["semiconductors"],
                "timeframe": "medium_term",
                "probability": 0.3
            },
            ["test_component_1", "test_component_2"]
        )
        print(f"Scenario Analysis Result: {scenario_result.success}")
        if scenario_result.success:
            risk_level = scenario_result.data.get("risk_assessment", {}).get("overall_risk_level", "unknown")
            print(f"Risk level: {risk_level}")
        
        # Test email processing pipeline
        print("\n--- Testing Email Processing Pipeline ---")
        email_data = await process_email(test_text, "RFQ Request")
        print(f"Email processing result keys: {list(email_data.keys())}")
        
        # Get observability metrics
        print("\n--- Observability and Metrics ---")
        system_health = observability_manager.get_system_health()
        print(f"System health status: {system_health.get('health_status', 'unknown')}")
        print(f"Health score: {system_health.get('health_score', 0):.3f}")
        print(f"Total capabilities monitored: {system_health.get('total_capabilities', 0)}")
        print(f"Active alerts: {system_health.get('active_alerts', 0)}")
        
        # Get capability metrics
        print("\n--- Capability Registry Metrics ---")
        metrics = orchestrator.registry.get_metrics()
        print(f"Total capabilities: {metrics.get('total_capabilities', 0)}")
        print(f"Healthy capabilities: {metrics.get('healthy_capabilities', 0)}")
        
        # Quick evaluation test
        print("\n--- Quick Evaluation Test ---")
        evaluator = CapabilityEvaluator(capability_registry)
        # Load just a simple test suite for demonstration
        from app.ai.capabilities.evaluation import EvaluationSuite, TestCase
        
        simple_suite = EvaluationSuite(
            name="demo_evaluation",
            description="Demo evaluation suite",
            capability_name="component_ner",
            test_cases=[
                TestCase(
                    name="simple_ner_test",
                    description="Simple NER test",
                    input_payload={"text": "STM32F429ZIT6 microcontroller"},
                    expected_success=True,
                    max_execution_time_ms=5000
                )
            ]
        )
        
        evaluator.add_test_suite(simple_suite)
        eval_report = await evaluator.run_suite("demo_evaluation")
        print(f"Evaluation: {eval_report.passed_tests}/{eval_report.total_tests} tests passed ({eval_report.success_rate:.1%})")
        
        print("\n✓ All enhanced tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_capabilities())
    exit(0 if success else 1)