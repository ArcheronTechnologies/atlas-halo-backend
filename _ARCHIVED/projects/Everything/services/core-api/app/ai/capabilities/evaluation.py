"""
AI Capabilities Evaluation Harness

Comprehensive testing and evaluation framework for AI capabilities with fixtures,
benchmarks, and performance monitoring.
"""

import logging
import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from .base import AICapability, CapabilityResult
from .registry import CapabilityRegistry

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Individual test case for capability evaluation"""
    name: str
    description: str
    input_payload: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    expected_success: bool = True
    min_confidence: Optional[float] = None
    max_execution_time_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    category: str = "functional"  # functional, performance, stress, integration


@dataclass
class TestResult:
    """Result of a single test case execution"""
    test_case: TestCase
    capability_result: CapabilityResult
    passed: bool
    execution_time_ms: float
    timestamp: datetime
    failure_reason: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSuite:
    """Collection of test cases for a capability"""
    name: str
    description: str
    capability_name: str
    test_cases: List[TestCase]
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    timeout_seconds: float = 300.0


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    suite_name: str
    capability_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    test_results: List[TestResult]
    performance_summary: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    timestamp: datetime
    duration_seconds: float


class TestFixtures:
    """Built-in test fixtures for common scenarios"""
    
    @staticmethod
    def ner_test_cases() -> List[TestCase]:
        """NER capability test cases"""
        return [
            TestCase(
                name="basic_component_extraction",
                description="Extract common electronic components",
                input_payload={
                    "text": "I need 100 pieces of STM32F429ZIT6 microcontroller and 50 LM358 op-amps"
                },
                expected_success=True,
                min_confidence=0.7,
                max_execution_time_ms=5000,
                tags=["basic", "components"],
                category="functional"
            ),
            TestCase(
                name="complex_text_extraction",
                description="Extract components from complex technical text",
                input_payload={
                    "text": "Our PCB design requires the following components: STM32F429ZIT6 microcontroller (QFP-144), "
                           "LM358N dual op-amp (DIP-8), 100uF electrolytic capacitors (25V), "
                           "10k ohm resistors (0603 SMD), and crystal oscillator 8MHz (HC-49/S)"
                },
                expected_success=True,
                min_confidence=0.6,
                max_execution_time_ms=8000,
                tags=["complex", "technical"],
                category="functional"
            ),
            TestCase(
                name="empty_input",
                description="Handle empty input gracefully",
                input_payload={"text": ""},
                expected_success=True,
                max_execution_time_ms=1000,
                tags=["edge_case"],
                category="functional"
            ),
            TestCase(
                name="no_components",
                description="Handle text with no components",
                input_payload={
                    "text": "This is a general discussion about supply chain management without specific components."
                },
                expected_success=True,
                max_execution_time_ms=3000,
                tags=["edge_case"],
                category="functional"
            )
        ]
    
    @staticmethod
    def classification_test_cases() -> List[TestCase]:
        """Classification capability test cases"""
        return [
            TestCase(
                name="rfq_intent_detection",
                description="Detect RFQ intent in email",
                input_payload={
                    "text": "Could you please provide a quote for 1000 pieces of STM32F429ZIT6?",
                    "subject": "RFQ for Microcontrollers"
                },
                expected_success=True,
                min_confidence=0.8,
                max_execution_time_ms=3000,
                tags=["intent", "rfq"],
                category="functional"
            ),
            TestCase(
                name="technical_inquiry",
                description="Classify technical inquiry",
                input_payload={
                    "text": "What are the specifications of the STM32F429ZIT6 microcontroller?",
                    "subject": "Technical Question"
                },
                expected_success=True,
                min_confidence=0.7,
                max_execution_time_ms=3000,
                tags=["intent", "technical"],
                category="functional"
            ),
            TestCase(
                name="general_inquiry",
                description="Classify general business inquiry",
                input_payload={
                    "text": "What are your business hours and location?",
                    "subject": "General Question"
                },
                expected_success=True,
                max_execution_time_ms=3000,
                tags=["intent", "general"],
                category="functional"
            )
        ]
    
    @staticmethod
    def recommendation_test_cases() -> List[TestCase]:
        """Recommendation capability test cases"""
        return [
            TestCase(
                name="microcontroller_alternatives",
                description="Find alternatives for microcontroller",
                input_payload={
                    "component": "STM32F429ZIT6",
                    "category": "microcontroller",
                    "max_recommendations": 5
                },
                expected_success=True,
                min_confidence=0.6,
                max_execution_time_ms=10000,
                tags=["recommendations", "microcontroller"],
                category="functional"
            ),
            TestCase(
                name="analog_component_alternatives",
                description="Find alternatives for analog component",
                input_payload={
                    "component": "LM358",
                    "category": "analog",
                    "max_recommendations": 3
                },
                expected_success=True,
                min_confidence=0.5,
                max_execution_time_ms=8000,
                tags=["recommendations", "analog"],
                category="functional"
            ),
            TestCase(
                name="unknown_component",
                description="Handle unknown component gracefully",
                input_payload={
                    "component": "UNKNOWN123XYZ",
                    "max_recommendations": 5
                },
                expected_success=True,
                max_execution_time_ms=5000,
                tags=["edge_case", "unknown"],
                category="functional"
            )
        ]
    
    @staticmethod
    def scenario_test_cases() -> List[TestCase]:
        """Scenario analysis test cases"""
        return [
            TestCase(
                name="geopolitical_scenario",
                description="Analyze geopolitical disruption scenario",
                input_payload={
                    "scenario_type": "geopolitical",
                    "parameters": {
                        "scenario_type": "geopolitical",
                        "geographic_scope": ["APAC"],
                        "industry_sectors": ["semiconductors"],
                        "timeframe": "medium_term",
                        "probability": 0.3
                    },
                    "affected_components": ["comp_1", "comp_2"],
                    "analysis_depth": "standard"
                },
                expected_success=True,
                min_confidence=0.6,
                max_execution_time_ms=15000,
                tags=["scenario", "geopolitical"],
                category="functional"
            ),
            TestCase(
                name="natural_disaster_scenario",
                description="Analyze natural disaster scenario",
                input_payload={
                    "scenario_type": "natural_disaster",
                    "parameters": {
                        "scenario_type": "natural_disaster",
                        "geographic_scope": ["Japan"],
                        "industry_sectors": ["semiconductors"],
                        "timeframe": "immediate",
                        "probability": 0.15
                    },
                    "affected_components": ["comp_1"],
                    "analysis_depth": "comprehensive"
                },
                expected_success=True,
                min_confidence=0.7,
                max_execution_time_ms=20000,
                tags=["scenario", "disaster"],
                category="functional"
            )
        ]
    
    @staticmethod
    def supplier_analysis_test_cases() -> List[TestCase]:
        """Supplier analysis test cases"""
        return [
            TestCase(
                name="comprehensive_supplier_analysis",
                description="Full supplier analysis with all modules",
                input_payload={
                    "supplier_id": "test_supplier_1",
                    "analysis_modules": ["financial_health", "delivery_performance", "quality"],
                    "time_period": "12m",
                    "include_recommendations": True
                },
                expected_success=True,
                min_confidence=0.6,
                max_execution_time_ms=12000,
                tags=["supplier", "comprehensive"],
                category="functional"
            ),
            TestCase(
                name="quick_supplier_analysis",
                description="Quick supplier analysis with limited modules",
                input_payload={
                    "supplier_id": "test_supplier_2",
                    "analysis_modules": ["financial_health", "quality"],
                    "time_period": "6m",
                    "include_recommendations": False
                },
                expected_success=True,
                max_execution_time_ms=8000,
                tags=["supplier", "quick"],
                category="performance"
            ),
            TestCase(
                name="nonexistent_supplier",
                description="Handle nonexistent supplier gracefully",
                input_payload={
                    "supplier_id": "nonexistent_supplier_999",
                    "include_recommendations": False
                },
                expected_success=False,  # Should fail gracefully
                max_execution_time_ms=3000,
                tags=["supplier", "edge_case"],
                category="functional"
            )
        ]
    
    @staticmethod
    def performance_test_cases() -> List[TestCase]:
        """Performance and stress test cases"""
        return [
            TestCase(
                name="concurrent_ner_requests",
                description="Handle multiple concurrent NER requests",
                input_payload={
                    "text": "STM32F429ZIT6 microcontroller and LM358 op-amp",
                    "concurrent_requests": 10
                },
                expected_success=True,
                max_execution_time_ms=15000,
                tags=["performance", "concurrent"],
                category="stress"
            ),
            TestCase(
                name="large_text_processing",
                description="Process large text input",
                input_payload={
                    "text": "This is a very long text with many components: " + 
                           " ".join([f"Component_{i} STM32F{i}29ZIT6" for i in range(100)])
                },
                expected_success=True,
                max_execution_time_ms=20000,
                tags=["performance", "large_input"],
                category="stress"
            )
        ]


class CapabilityEvaluator:
    """Main evaluation harness for AI capabilities"""
    
    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry
        self.test_suites: Dict[str, EvaluationSuite] = {}
        self.fixtures = TestFixtures()
        
    def add_test_suite(self, suite: EvaluationSuite) -> None:
        """Add a test suite to the evaluator"""
        self.test_suites[suite.name] = suite
        logger.info(f"Added test suite '{suite.name}' with {len(suite.test_cases)} test cases")
    
    def load_default_suites(self) -> None:
        """Load default test suites for all capabilities"""
        # NER test suite
        ner_suite = EvaluationSuite(
            name="ner_evaluation",
            description="Named Entity Recognition capability evaluation",
            capability_name="component_ner",
            test_cases=self.fixtures.ner_test_cases()
        )
        self.add_test_suite(ner_suite)
        
        # Classification test suite
        classification_suite = EvaluationSuite(
            name="classification_evaluation",
            description="Intent classification capability evaluation",
            capability_name="intent_classifier",
            test_cases=self.fixtures.classification_test_cases()
        )
        self.add_test_suite(classification_suite)
        
        # Recommendation test suite
        recommendation_suite = EvaluationSuite(
            name="recommendation_evaluation",
            description="Component recommendation capability evaluation",
            capability_name="component_recommender",
            test_cases=self.fixtures.recommendation_test_cases()
        )
        self.add_test_suite(recommendation_suite)
        
        # Scenario analysis test suite
        scenario_suite = EvaluationSuite(
            name="scenario_evaluation",
            description="Scenario analysis capability evaluation",
            capability_name="scenario_analysis",
            test_cases=self.fixtures.scenario_test_cases()
        )
        self.add_test_suite(scenario_suite)
        
        # Supplier analysis test suite
        supplier_suite = EvaluationSuite(
            name="supplier_evaluation",
            description="Supplier analysis capability evaluation",
            capability_name="advanced_supplier_analysis",
            test_cases=self.fixtures.supplier_analysis_test_cases()
        )
        self.add_test_suite(supplier_suite)
        
        # Performance test suite
        performance_suite = EvaluationSuite(
            name="performance_evaluation",
            description="Performance and stress testing",
            capability_name="component_ner",  # Use NER for performance tests
            test_cases=self.fixtures.performance_test_cases()
        )
        self.add_test_suite(performance_suite)
    
    async def run_test_case(self, capability: AICapability, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        start_time = time.time()
        
        try:
            # Handle concurrent requests for stress testing
            if test_case.input_payload.get('concurrent_requests'):
                concurrent_count = test_case.input_payload.pop('concurrent_requests')
                tasks = []
                for _ in range(concurrent_count):
                    tasks.append(capability.run(test_case.input_payload.copy()))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Use the first successful result or first result if all failed
                capability_result = next((r for r in results if isinstance(r, CapabilityResult) and r.success), 
                                       results[0] if results else None)
                
                if not isinstance(capability_result, CapabilityResult):
                    capability_result = CapabilityResult(
                        success=False,
                        error=f"Concurrent execution failed: {capability_result}"
                    )
                    
            else:
                # Normal single request
                capability_result = await capability.run(test_case.input_payload)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Evaluate test result
            passed, failure_reason = self._evaluate_test_result(test_case, capability_result, execution_time)
            
            return TestResult(
                test_case=test_case,
                capability_result=capability_result,
                passed=passed,
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc),
                failure_reason=failure_reason,
                performance_metrics={
                    'memory_usage': None,  # Could add memory monitoring
                    'cpu_usage': None      # Could add CPU monitoring
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Test case '{test_case.name}' failed with exception: {e}")
            
            return TestResult(
                test_case=test_case,
                capability_result=CapabilityResult(success=False, error=str(e)),
                passed=False,
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc),
                failure_reason=f"Exception during execution: {str(e)}"
            )
    
    def _evaluate_test_result(
        self, test_case: TestCase, result: CapabilityResult, execution_time: float
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate if a test result passes the test case criteria"""
        
        # Check basic success expectation
        if test_case.expected_success and not result.success:
            return False, f"Expected success but got failure: {result.error}"
        
        if not test_case.expected_success and result.success:
            return False, "Expected failure but got success"
        
        # Check execution time
        if test_case.max_execution_time_ms and execution_time > test_case.max_execution_time_ms:
            return False, f"Execution time {execution_time:.1f}ms exceeded limit {test_case.max_execution_time_ms}ms"
        
        # Check minimum confidence
        if test_case.min_confidence and result.confidence and result.confidence < test_case.min_confidence:
            return False, f"Confidence {result.confidence} below minimum {test_case.min_confidence}"
        
        # Check expected output if specified
        if test_case.expected_output and result.success:
            if not self._compare_outputs(test_case.expected_output, result.data):
                return False, "Output does not match expected result"
        
        return True, None
    
    def _compare_outputs(self, expected: Dict[str, Any], actual: Any) -> bool:
        """Compare expected vs actual outputs (basic implementation)"""
        # This is a basic comparison - could be enhanced with more sophisticated matching
        if isinstance(actual, dict):
            for key, value in expected.items():
                if key not in actual:
                    return False
                if value != actual[key]:
                    return False
        return True
    
    async def run_suite(self, suite_name: str) -> EvaluationReport:
        """Run a complete test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        suite = self.test_suites[suite_name]
        start_time = time.time()
        
        logger.info(f"Starting evaluation suite '{suite_name}' with {len(suite.test_cases)} test cases")
        
        # Get capability
        capability = self.registry.get_capability(suite.capability_name)
        if not capability:
            raise ValueError(f"Capability '{suite.capability_name}' not found in registry")
        
        # Run setup if provided
        if suite.setup_function:
            await suite.setup_function()
        
        try:
            # Run all test cases
            test_results = []
            for test_case in suite.test_cases:
                logger.debug(f"Running test case: {test_case.name}")
                result = await self.run_test_case(capability, test_case)
                test_results.append(result)
            
            # Generate report
            duration = time.time() - start_time
            report = self._generate_report(suite, test_results, duration)
            
            logger.info(
                f"Suite '{suite_name}' completed: {report.passed_tests}/{report.total_tests} tests passed "
                f"({report.success_rate:.1%}) in {duration:.2f}s"
            )
            
            return report
            
        finally:
            # Run teardown if provided
            if suite.teardown_function:
                await suite.teardown_function()
    
    def _generate_report(
        self, suite: EvaluationSuite, test_results: List[TestResult], duration: float
    ) -> EvaluationReport:
        """Generate evaluation report from test results"""
        
        passed_tests = sum(1 for r in test_results if r.passed)
        failed_tests = len(test_results) - passed_tests
        
        execution_times = [r.execution_time_ms for r in test_results]
        
        errors = [r.failure_reason for r in test_results if r.failure_reason]
        warnings = []
        
        # Collect warnings from capability results
        for result in test_results:
            if result.capability_result.warnings:
                warnings.extend(result.capability_result.warnings)
        
        # Performance analysis
        performance_summary = {
            "avg_execution_time_ms": statistics.mean(execution_times) if execution_times else 0,
            "median_execution_time_ms": statistics.median(execution_times) if execution_times else 0,
            "p95_execution_time_ms": statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 10 else max(execution_times, default=0),
            "fastest_test": min(test_results, key=lambda x: x.execution_time_ms).test_case.name if test_results else None,
            "slowest_test": max(test_results, key=lambda x: x.execution_time_ms).test_case.name if test_results else None
        }
        
        # Category breakdown
        categories = {}
        for result in test_results:
            category = result.test_case.category
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0}
            categories[category]["total"] += 1
            if result.passed:
                categories[category]["passed"] += 1
        
        performance_summary["category_breakdown"] = categories
        
        return EvaluationReport(
            suite_name=suite.name,
            capability_name=suite.capability_name,
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=passed_tests / len(test_results) if test_results else 0,
            avg_execution_time=statistics.mean(execution_times) if execution_times else 0,
            min_execution_time=min(execution_times) if execution_times else 0,
            max_execution_time=max(execution_times) if execution_times else 0,
            test_results=test_results,
            performance_summary=performance_summary,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(timezone.utc),
            duration_seconds=duration
        )
    
    async def run_all_suites(self) -> Dict[str, EvaluationReport]:
        """Run all loaded test suites"""
        reports = {}
        
        logger.info(f"Running {len(self.test_suites)} test suites")
        
        for suite_name in self.test_suites:
            try:
                report = await self.run_suite(suite_name)
                reports[suite_name] = report
            except Exception as e:
                logger.error(f"Failed to run suite '{suite_name}': {e}")
                # Create error report
                reports[suite_name] = EvaluationReport(
                    suite_name=suite_name,
                    capability_name=self.test_suites[suite_name].capability_name,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    success_rate=0.0,
                    avg_execution_time=0.0,
                    min_execution_time=0.0,
                    max_execution_time=0.0,
                    test_results=[],
                    performance_summary={},
                    errors=[str(e)],
                    warnings=[],
                    timestamp=datetime.now(timezone.utc),
                    duration_seconds=0.0
                )
        
        return reports
    
    def export_report(self, report: EvaluationReport, output_path: str) -> None:
        """Export evaluation report to JSON file"""
        report_data = asdict(report)
        
        # Convert datetime objects to ISO strings
        report_data['timestamp'] = report.timestamp.isoformat()
        for test_result in report_data['test_results']:
            test_result['timestamp'] = test_result['timestamp'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation report exported to {output_path}")
    
    def generate_summary_report(self, reports: Dict[str, EvaluationReport]) -> Dict[str, Any]:
        """Generate overall summary from multiple suite reports"""
        total_tests = sum(r.total_tests for r in reports.values())
        total_passed = sum(r.passed_tests for r in reports.values())
        total_failed = sum(r.failed_tests for r in reports.values())
        
        suite_summaries = {}
        for name, report in reports.items():
            suite_summaries[name] = {
                "success_rate": report.success_rate,
                "avg_execution_time": report.avg_execution_time,
                "test_count": report.total_tests,
                "status": "PASS" if report.success_rate >= 0.8 else "FAIL"
            }
        
        return {
            "overall_summary": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "overall_success_rate": total_passed / total_tests if total_tests > 0 else 0,
                "suites_passed": sum(1 for r in reports.values() if r.success_rate >= 0.8),
                "suites_failed": sum(1 for r in reports.values() if r.success_rate < 0.8)
            },
            "suite_summaries": suite_summaries,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }