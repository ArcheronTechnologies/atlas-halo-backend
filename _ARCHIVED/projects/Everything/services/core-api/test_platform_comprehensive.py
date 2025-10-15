#!/usr/bin/env python3
"""
Comprehensive Large-Scale Test Suite for SCIP Platform
Tests all major features, endpoints, and integrations
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import httpx

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add app to path  
sys.path.insert(0, str(Path(__file__).parent / "app"))
os.environ.setdefault("SCIP_MINIMAL_STARTUP", "1")

class PlatformTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.auth_token = None
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "performance": {},
            "endpoints_tested": []
        }
        
    async def authenticate(self):
        """Get authentication token"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/auth/login",
                    json={"email": "user@admin", "password": "dev"}
                )
                if response.status_code == 200:
                    data = response.json()
                    self.auth_token = data.get("accessToken")
                    return True
                else:
                    self.log_error("Authentication failed", f"Status: {response.status_code}")
                    return False
        except Exception as e:
            self.log_error("Authentication error", str(e))
            return False
            
    def headers(self):
        """Get headers with auth token"""
        if not self.auth_token:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}
        
    def log_error(self, test_name: str, error: str):
        """Log an error"""
        self.results["errors"].append({"test": test_name, "error": error})
        print(f"❌ {test_name}: {error}")
        
    def log_success(self, test_name: str, details: str = ""):
        """Log a success"""
        print(f"✅ {test_name}: {details}")
        
    async def test_endpoint(self, method: str, endpoint: str, test_name: str, 
                          params: Dict = None, json_data: Dict = None, 
                          expected_status: int = 200) -> bool:
        """Test a single endpoint"""
        self.results["total_tests"] += 1
        self.results["endpoints_tested"].append(endpoint)
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                kwargs = {"headers": self.headers()}
                if params:
                    kwargs["params"] = params
                if json_data:
                    kwargs["json"] = json_data
                    
                response = await client.request(method, f"{self.base_url}{endpoint}", **kwargs)
                
                # Record performance
                response_time = time.time() - start_time
                self.results["performance"][test_name] = response_time
                
                if response.status_code == expected_status:
                    self.results["passed"] += 1
                    try:
                        data = response.json()
                        details = f"Status {response.status_code}, Response time: {response_time:.3f}s"
                        if isinstance(data, dict):
                            if "parts" in data:
                                details += f", Parts: {len(data.get('parts', []))}"
                            if "hits" in data:
                                details += f", Hits: {data.get('hits', 0)}"
                            if "results" in data:
                                details += f", Results: {len(data.get('results', []))}"
                        self.log_success(test_name, details)
                        return True
                    except:
                        self.log_success(test_name, f"Status {response.status_code}, Response time: {response_time:.3f}s")
                        return True
                else:
                    self.results["failed"] += 1
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    except:
                        error_msg = f"HTTP {response.status_code}"
                    self.log_error(test_name, f"{error_msg} (Expected {expected_status}, got {response.status_code})")
                    return False
                    
        except Exception as e:
            self.results["failed"] += 1
            self.log_error(test_name, str(e))
            return False
            
    async def test_health_endpoints(self):
        """Test basic health and system endpoints"""
        print("\n🏥 TESTING HEALTH & SYSTEM ENDPOINTS")
        print("=" * 50)
        
        await self.test_endpoint("GET", "/health", "Health Check")
        await self.test_endpoint("GET", "/ready", "Readiness Check")
        await self.test_endpoint("GET", "/metrics", "Metrics Endpoint")
        
    async def test_authentication(self):
        """Test authentication system"""
        print("\n🔐 TESTING AUTHENTICATION SYSTEM")
        print("=" * 50)
        
        # Test login
        auth_success = await self.authenticate()
        if auth_success:
            self.log_success("User Login", f"Token received: {self.auth_token[:20]}...")
        
        # Test protected endpoint without auth
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/v1/components")
            if response.status_code == 401:
                self.log_success("Auth Protection", "Unauthorized access properly blocked")
            else:
                self.log_error("Auth Protection", f"Expected 401, got {response.status_code}")
                
    async def test_octopart_endpoints(self):
        """Test all Octopart/Nexar API endpoints extensively"""
        print("\n🔍 TESTING OCTOPART/NEXAR API ENDPOINTS")
        print("=" * 50)
        
        test_queries = [
            ("STM32F429ZIT6", "Specific STM32 MCU"),
            ("STM32", "STM32 family search"),
            ("LM358", "Op-amp search"),
            ("74HC04", "Logic gate search"),
            ("ESP32", "ESP32 modules"),
        ]
        
        # Test Total Availability
        for query, description in test_queries:
            await self.test_endpoint(
                "GET", "/v1/market/total-availability",
                f"Total Availability: {description}",
                params={"q": query, "country": "US", "limit": 5}
            )
            
        # Test Pricing Breaks
        for query, description in test_queries:
            await self.test_endpoint(
                "GET", "/v1/market/pricing-breaks", 
                f"Pricing Breaks: {description}",
                params={"q": query, "limit": 3, "currency": "USD"}
            )
            
        # Test Detailed Offers
        specific_parts = ["STM32F429ZIT6", "LM358N", "74HC04D"]
        for part in specific_parts:
            await self.test_endpoint(
                "GET", "/v1/market/offers",
                f"Detailed Offers: {part}",
                params={"mpn": part, "country": "US", "currency": "USD"}
            )
            
        # Test Spec Attributes
        filter_tests = [
            ({"case_package": ["SSOP"]}, "SSOP package filter"),
            ({"manufacturer": ["Texas Instruments"]}, "TI manufacturer filter"),
            ({}, "No filters")
        ]
        for filters, description in filter_tests:
            params = {"q": "ADS", "limit": 3}
            if filters:
                params["filters"] = json.dumps(filters)
            await self.test_endpoint(
                "GET", "/v1/market/spec-attributes",
                f"Spec Attributes: {description}",
                params=params
            )
            
    async def test_core_business_endpoints(self):
        """Test core business functionality"""
        print("\n🏢 TESTING CORE BUSINESS ENDPOINTS")
        print("=" * 50)
        
        # Test Components
        await self.test_endpoint("GET", "/v1/components", "List Components")
        
        # Test Companies
        await self.test_endpoint("GET", "/v1/companies", "List Companies")
        
        # Test Users (admin access)
        await self.test_endpoint("GET", "/v1/users", "List Users")
        
        # Test Inventory
        await self.test_endpoint("GET", "/v1/inventory", "List Inventory")
        
        # Test RFQs
        await self.test_endpoint("GET", "/v1/rfqs", "List RFQs")
        
        # Test Purchase Orders
        await self.test_endpoint("GET", "/v1/purchase-orders", "List Purchase Orders")
        
    async def test_advanced_features(self):
        """Test advanced platform features"""
        print("\n🧠 TESTING ADVANCED FEATURES")
        print("=" * 50)
        
        # Test Search
        await self.test_endpoint("GET", "/v1/search", "Global Search", 
                                params={"q": "STM32", "limit": 10})
        
        # Test Intelligence
        await self.test_endpoint("GET", "/v1/intelligence/market-trends", "Market Trends")
        await self.test_endpoint("GET", "/v1/intelligence/supply-risks", "Supply Risk Analysis")
        await self.test_endpoint("GET", "/v1/intelligence/price-predictions", "Price Predictions")
        
        # Test Graph Database
        await self.test_endpoint("GET", "/v1/graph/relationships", "Graph Relationships")
        await self.test_endpoint("GET", "/v1/graph/suppliers", "Supplier Graph")
        
        # Test Audit
        await self.test_endpoint("GET", "/v1/audit/events", "Audit Events")
        
    async def test_integration_endpoints(self):
        """Test integration endpoints"""
        print("\n🔗 TESTING INTEGRATION ENDPOINTS")
        print("=" * 50)
        
        # Test Microsoft 365 integration
        await self.test_endpoint("GET", "/v1/integrations/microsoft365/status", 
                                "Microsoft 365 Status", expected_status=200)
        
        # Test ERP integrations
        await self.test_endpoint("GET", "/v1/integrations/visma/status",
                                "Visma ERP Status", expected_status=200)
        await self.test_endpoint("GET", "/v1/integrations/business-central/status",
                                "Business Central Status", expected_status=200)
        
        # Test C3 and HubSpot
        await self.test_endpoint("GET", "/v1/integrations/c3/status",
                                "C3 Integration Status", expected_status=200)
        await self.test_endpoint("GET", "/v1/integrations/hubspot/status", 
                                "HubSpot Integration Status", expected_status=200)
                                
    async def test_ingestion_layer(self):
        """Test comprehensive data ingestion layer"""
        print("\n📥 TESTING DATA INGESTION LAYER")
        print("=" * 50)
        
        # Test email ingestion with various scenarios
        email_test_cases = [
            {
                "name": "RFQ Response Email",
                "data": {
                    "from": "supplier@example.com",
                    "subject": "RFQ Response - STM32F429ZIT6",
                    "body": "We can supply 1000 units at $12.50 each with 2-week lead time",
                    "attachments": [],
                    "timestamp": "2025-01-15T10:30:00Z"
                }
            },
            {
                "name": "Quote Email with PDF",
                "data": {
                    "from": "distributor@avnet.com", 
                    "subject": "Quote #Q12345 - Multiple Components",
                    "body": "Please find attached quote for your requirements",
                    "attachments": [{"filename": "quote.pdf", "content_type": "application/pdf"}],
                    "timestamp": "2025-01-15T11:00:00Z"
                }
            },
            {
                "name": "Availability Update",
                "data": {
                    "from": "inventory@mouser.com",
                    "subject": "Stock Alert - ESP32-WROOM-32",
                    "body": "ESP32-WROOM-32 now in stock: 5000 units available",
                    "attachments": [],
                    "timestamp": "2025-01-15T12:00:00Z"
                }
            }
        ]
        
        for test_case in email_test_cases:
            await self.test_endpoint("POST", "/v1/ingestion/email",
                                    f"Email Ingestion: {test_case['name']}",
                                    json_data=test_case["data"])
        
        # Test BOM ingestion with different formats
        bom_test_cases = [
            {
                "name": "Simple BOM",
                "data": {
                    "filename": "simple_bom.xlsx",
                    "format": "xlsx",
                    "parts": [
                        {"mpn": "STM32F429ZIT6", "quantity": 100, "reference": "U1"},
                        {"mpn": "LM358N", "quantity": 50, "reference": "U2-U51"}
                    ]
                }
            },
            {
                "name": "Complex BOM with Alternates",
                "data": {
                    "filename": "complex_bom.csv",
                    "format": "csv", 
                    "parts": [
                        {
                            "mpn": "STM32F407VGT6",
                            "quantity": 200,
                            "reference": "U1-U200",
                            "alternates": ["STM32F407VET6", "STM32F407VGT7"],
                            "package": "LQFP-100"
                        },
                        {
                            "mpn": "74HC04D",
                            "quantity": 300,
                            "reference": "U201-U500",
                            "description": "Hex Inverter",
                            "package": "SOIC-14"
                        }
                    ]
                }
            }
        ]
        
        for test_case in bom_test_cases:
            await self.test_endpoint("POST", "/v1/ingestion/bom",
                                    f"BOM Ingestion: {test_case['name']}",
                                    json_data=test_case["data"])
        
        # Test ERP data ingestion
        await self.test_endpoint("POST", "/v1/ingestion/erp/visma",
                                "Visma ERP Data Ingestion",
                                json_data={
                                    "transaction_type": "purchase_order",
                                    "data": {
                                        "po_number": "PO-2025-001",
                                        "supplier": "Arrow Electronics",
                                        "items": [
                                            {"mpn": "STM32F429ZIT6", "quantity": 500, "unit_price": 12.45}
                                        ],
                                        "total_amount": 6225.00
                                    }
                                })
        
        # Test Microsoft 365 data ingestion  
        await self.test_endpoint("POST", "/v1/ingestion/microsoft365/teams",
                                "Teams Chat Ingestion",
                                json_data={
                                    "chat_id": "chat123",
                                    "messages": [
                                        {
                                            "from": "john.doe@company.com",
                                            "content": "We need 1000 units of STM32F429ZIT6 by end of month",
                                            "timestamp": "2025-01-15T14:30:00Z"
                                        }
                                    ]
                                })
        
        # Test web crawling results ingestion
        await self.test_endpoint("POST", "/v1/ingestion/web-intelligence",
                                "Web Intelligence Ingestion",
                                json_data={
                                    "source": "semiconductor-news.com",
                                    "url": "https://example.com/news/stm32-shortage",
                                    "title": "STM32 Microcontroller Shortage Expected",
                                    "content": "Industry analysis suggests STM32F4 series may face shortages...",
                                    "extracted_data": {
                                        "mentioned_parts": ["STM32F429ZIT6", "STM32F407VGT6"],
                                        "sentiment": "negative",
                                        "risk_level": "medium"
                                    },
                                    "crawled_at": "2025-01-15T16:00:00Z"
                                })
                                
    async def test_analysis_models(self):
        """Test AI/ML analysis models and capabilities"""
        print("\n🧠 TESTING ANALYSIS MODELS & AI CAPABILITIES")
        print("=" * 50)
        
        # Test NLP/NER models for component extraction
        await self.test_endpoint("POST", "/v1/ai/extract-components",
                                "Component NER Model",
                                json_data={
                                    "text": "We need quotes for STM32F429ZIT6, LM358N operational amplifiers, and 74HC04 hex inverters for our new project",
                                    "confidence_threshold": 0.8
                                })
        
        # Test sentiment analysis
        await self.test_endpoint("POST", "/v1/ai/analyze-sentiment", 
                                "Sentiment Analysis Model",
                                json_data={
                                    "text": "The delivery was delayed again and quality issues persist with this supplier",
                                    "context": "supplier_feedback"
                                })
        
        # Test price prediction models
        await self.test_endpoint("POST", "/v1/ai/predict-prices",
                                "Price Prediction Model",
                                json_data={
                                    "components": [
                                        {"mpn": "STM32F429ZIT6", "quantity": 1000},
                                        {"mpn": "LM358N", "quantity": 500}
                                    ],
                                    "time_horizon": "3_months",
                                    "market_conditions": {
                                        "demand_forecast": "high",
                                        "supply_risk": "medium"
                                    }
                                })
        
        # Test supply risk analysis
        await self.test_endpoint("POST", "/v1/ai/analyze-supply-risk",
                                "Supply Risk Analysis Model", 
                                json_data={
                                    "components": ["STM32F429ZIT6", "ESP32-WROOM-32"],
                                    "factors": ["geopolitical", "manufacturing", "demand"],
                                    "time_horizon": "6_months"
                                })
        
        # Test demand forecasting
        await self.test_endpoint("POST", "/v1/ai/forecast-demand",
                                "Demand Forecasting Model",
                                json_data={
                                    "component": "STM32F429ZIT6",
                                    "historical_data": {
                                        "months": 12,
                                        "include_seasonality": True,
                                        "include_market_trends": True
                                    },
                                    "forecast_horizon": "6_months"
                                })
        
        # Test BOM optimization
        await self.test_endpoint("POST", "/v1/ai/optimize-bom",
                                "BOM Optimization Model",
                                json_data={
                                    "bom": [
                                        {"mpn": "STM32F429ZIT6", "quantity": 1000, "current_supplier": "Mouser"},
                                        {"mpn": "LM358N", "quantity": 500, "current_supplier": "Digi-Key"}
                                    ],
                                    "optimization_criteria": ["cost", "availability", "lead_time"],
                                    "constraints": {
                                        "max_suppliers": 3,
                                        "min_availability": 90
                                    }
                                })
        
        # Test supplier scoring model
        await self.test_endpoint("POST", "/v1/ai/score-suppliers",
                                "Supplier Scoring Model",
                                json_data={
                                    "suppliers": ["Mouser", "Digi-Key", "Arrow", "Avnet"],
                                    "component": "STM32F429ZIT6",
                                    "criteria": {
                                        "price_weight": 0.3,
                                        "availability_weight": 0.3,
                                        "reliability_weight": 0.2,
                                        "lead_time_weight": 0.2
                                    }
                                })
        
        # Test anomaly detection
        await self.test_endpoint("POST", "/v1/ai/detect-anomalies",
                                "Anomaly Detection Model",
                                json_data={
                                    "data_type": "pricing",
                                    "component": "STM32F429ZIT6",
                                    "time_series_data": [
                                        {"date": "2025-01-01", "price": 12.50},
                                        {"date": "2025-01-02", "price": 12.45},
                                        {"date": "2025-01-03", "price": 25.00},  # Potential anomaly
                                        {"date": "2025-01-04", "price": 12.40}
                                    ]
                                })
        
        # Test market intelligence synthesis
        await self.test_endpoint("POST", "/v1/ai/synthesize-intelligence",
                                "Market Intelligence Synthesis",
                                json_data={
                                    "components": ["STM32F429ZIT6"],
                                    "data_sources": ["pricing", "availability", "news", "social_media"],
                                    "analysis_depth": "comprehensive",
                                    "include_predictions": True
                                })
        
        # Test geopolitical risk assessment
        await self.test_endpoint("POST", "/v1/ai/assess-geopolitical-risk",
                                "Geopolitical Risk Assessment",
                                json_data={
                                    "supply_chain": {
                                        "manufacturing_locations": ["Taiwan", "China", "Malaysia"],
                                        "key_suppliers": ["TSMC", "GlobalFoundries"],
                                        "components": ["STM32F429ZIT6", "ESP32-WROOM-32"]
                                    },
                                    "risk_factors": ["trade_tensions", "natural_disasters", "regulatory_changes"],
                                    "time_horizon": "12_months"
                                })
                                
    async def performance_test(self):
        """Run performance tests"""
        print("\n⚡ RUNNING PERFORMANCE TESTS")
        print("=" * 50)
        
        # Concurrent requests test
        concurrent_tasks = []
        for i in range(10):
            task = self.test_endpoint("GET", "/v1/market/total-availability",
                                    f"Concurrent Request {i+1}",
                                    params={"q": f"STM32F{i}", "limit": 3})
            concurrent_tasks.append(task)
            
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful = sum(1 for r in results if r is True)
        print(f"🚀 Concurrent Test: {successful}/10 successful in {total_time:.2f}s")
        
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE PLATFORM TEST REPORT")
        print("=" * 60)
        
        total = self.results["total_tests"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"📊 SUMMARY:")
        print(f"   Total Tests: {total}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Performance summary
        if self.results["performance"]:
            avg_response = sum(self.results["performance"].values()) / len(self.results["performance"])
            slowest = max(self.results["performance"].items(), key=lambda x: x[1])
            fastest = min(self.results["performance"].items(), key=lambda x: x[1])
            
            print(f"\n⚡ PERFORMANCE:")
            print(f"   Average Response Time: {avg_response:.3f}s")
            print(f"   Fastest: {fastest[0]} ({fastest[1]:.3f}s)")
            print(f"   Slowest: {slowest[0]} ({slowest[1]:.3f}s)")
        
        # Endpoints tested
        print(f"\n🔗 ENDPOINTS TESTED ({len(set(self.results['endpoints_tested']))}):")
        unique_endpoints = sorted(set(self.results["endpoints_tested"]))
        for endpoint in unique_endpoints:
            count = self.results["endpoints_tested"].count(endpoint)
            print(f"   {endpoint} ({count}x)")
        
        # Errors
        if self.results["errors"]:
            print(f"\n❌ ERRORS ({len(self.results['errors'])}):")
            for error in self.results["errors"]:
                print(f"   {error['test']}: {error['error']}")
        else:
            print(f"\n✅ NO ERRORS FOUND!")
            
        # Platform status
        print(f"\n🏆 PLATFORM STATUS:")
        if success_rate >= 90:
            print("   🟢 EXCELLENT - Platform is production ready!")
        elif success_rate >= 75:
            print("   🟡 GOOD - Minor issues to address")
        elif success_rate >= 50:
            print("   🟠 FAIR - Several issues need fixing")
        else:
            print("   🔴 POOR - Major issues require attention")
            
        return self.results

async def main():
    """Run comprehensive platform tests"""
    print("🚀 STARTING COMPREHENSIVE SCIP PLATFORM TEST")
    print("=" * 60)
    
    tester = PlatformTester()
    
    # Run all test suites
    await tester.test_health_endpoints()
    await tester.test_authentication()
    await tester.test_octopart_endpoints()
    await tester.test_core_business_endpoints()
    await tester.test_advanced_features()
    await tester.test_integration_endpoints()
    await tester.test_ingestion_layer()
    await tester.test_analysis_models()
    await tester.performance_test()
    
    # Generate final report
    results = tester.generate_report()
    
    # Save results to file
    with open("platform_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: platform_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())