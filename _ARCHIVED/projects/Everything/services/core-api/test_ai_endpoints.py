#!/usr/bin/env python3
"""
Test AI/ML Analysis Endpoints
"""

import asyncio
import httpx
import json

async def test_ai_endpoints():
    """Test the newly implemented AI endpoints"""
    base_url = "http://localhost:8000"
    
    # Get auth token
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/auth/login",
            json={"email": "user@admin", "password": "dev"}
        )
        if response.status_code != 200:
            print("❌ Authentication failed")
            return
        
        token = response.json().get("accessToken")
        headers = {"Authorization": f"Bearer {token}"}
        
        print("🤖 Testing AI/ML Analysis Endpoints")
        print("=" * 50)
        
        # Test 1: Component extraction
        print("\n🔍 Testing Component NER Extraction...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/extract-components",
                headers=headers,
                json={
                    "text": "I need STM32F429ZIT6 microcontroller and LM358 op-amp for my project",
                    "confidence_threshold": 0.8
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Found {len(data.get('components', []))} components")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 2: Sentiment analysis
        print("\n😊 Testing Sentiment Analysis...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/analyze-sentiment",
                headers=headers,
                json={
                    "text": "The delivery was delayed again, this is very frustrating!",
                    "context": "supplier communication"
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Sentiment = {data.get('sentiment')}")
                print(f"   Confidence: {data.get('confidence', 0):.2f}")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: Price prediction
        print("\n💰 Testing Price Prediction...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/predict-prices",
                headers=headers,
                json={
                    "component_id": "STM32F429ZIT6",
                    "forecast_horizon_days": 90
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Generated {len(data.get('predictions', []))} predictions")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 4: Supply risk analysis
        print("\n⚠️  Testing Supply Risk Analysis...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/analyze-supply-risk",
                headers=headers,
                json={
                    "component_ids": ["STM32F429ZIT6", "LM358"],
                    "suppliers": ["STMicroelectronics", "Texas Instruments"],
                    "risk_factors": ["geopolitical", "capacity"]
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Risk score = {data.get('risk_score', 0):.2f}")
                print(f"   Mitigation strategies: {len(data.get('mitigation_strategies', []))}")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 5: Demand forecasting
        print("\n📈 Testing Demand Forecasting...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/forecast-demand",
                headers=headers,
                json={
                    "component_id": "STM32F429ZIT6",
                    "forecast_horizon_days": 90
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Generated {len(data.get('forecast', []))} forecast points")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 6: BOM optimization
        print("\n🔧 Testing BOM Optimization...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/optimize-bom",
                headers=headers,
                json={
                    "bom_components": [
                        {"part_number": "STM32F429ZIT6", "quantity": 100},
                        {"part_number": "LM358", "quantity": 50}
                    ],
                    "optimization_criteria": ["cost", "availability", "risk"]
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Optimized {len(data.get('optimized_bom', []))} components")
                print(f"   Cost savings: ${data.get('cost_savings', 0):.2f}")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 7: Supplier scoring
        print("\n🏆 Testing Supplier Scoring...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/score-suppliers",
                headers=headers,
                json={
                    "supplier_id": "STMicroelectronics",
                    "evaluation_criteria": ["delivery", "quality", "cost", "risk"]
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Overall score = {data.get('overall_score', 0):.2f}")
                print(f"   Category scores: {len(data.get('category_scores', {}))}")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 8: Market intelligence synthesis
        print("\n🌐 Testing Market Intelligence Synthesis...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/synthesize-intelligence",
                headers=headers,
                json={
                    "components": ["STM32F429ZIT6", "LM358"],
                    "intelligence_types": ["price", "availability", "trends"],
                    "time_horizon": "3months"
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Analyzed {data.get('intelligence_summary', {}).get('components_analyzed', 0)} components")
                print(f"   Market trends: {len(data.get('market_trends', []))}")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 9: Geopolitical risk assessment
        print("\n🌍 Testing Geopolitical Risk Assessment...")
        try:
            response = await client.post(
                f"{base_url}/v1/ai/assess-geopolitical-risk",
                headers=headers,
                json={
                    "suppliers": ["STMicroelectronics", "Texas Instruments", "Infineon"],
                    "components": ["STM32F429ZIT6", "LM358"],
                    "risk_scenarios": ["trade_war", "sanctions", "natural_disaster"]
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Assessed {len(data.get('country_risks', {})) } countries")
                print(f"   Scenario impacts: {len(data.get('scenario_impacts', []))}")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 10: AI capabilities overview
        print("\n🤖 Testing AI Capabilities Overview...")
        try:
            response = await client.get(
                f"{base_url}/v1/ai/capabilities",
                headers=headers
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: Found {data.get('total_capabilities', 0)} capabilities")
                print(f"   Healthy capabilities: {data.get('healthy_capabilities', 0)}")
                print(f"   Pipelines: {len(data.get('pipelines', {}))}")
                print(f"   Framework version: {data.get('framework_version', 'unknown')}")
            else:
                print(f"❌ Failed: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print(f"\n🎯 AI Endpoints Testing Complete!")

if __name__ == "__main__":
    asyncio.run(test_ai_endpoints())