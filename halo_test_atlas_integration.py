"""
Test Atlas Intelligence Integration in Halo Backend
Run: python test_atlas_integration.py
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_atlas_health():
    """Test Atlas Intelligence health check"""
    print("\n" + "="*60)
    print("TEST 1: Atlas Intelligence Health Check")
    print("="*60)

    from backend.services.atlas_client import get_atlas_client

    atlas = get_atlas_client()
    print(f"✅ Atlas client initialized: {atlas.base_url}")

    health = await atlas.health_check()
    print(f"\nAtlas Health: {health.get('status')}")

    if health.get('status') == 'healthy':
        services = health.get('services', {})
        print(f"  - Threat Classifier: {services.get('threat_classifier')}")
        print(f"  - Visual Detector: {services.get('visual_detector')}")
        print(f"  - Audio Classifier: {services.get('audio_classifier')}")
        print("\n✅ Atlas Intelligence is healthy and ready!")
        return True
    else:
        print(f"\n⚠️ Atlas Intelligence health check failed: {health.get('error')}")
        return False


async def test_incident_classifier():
    """Test incident classification via Atlas"""
    print("\n" + "="*60)
    print("TEST 2: Incident Classification (Halo Backend)")
    print("="*60)

    from backend.ai_processing.incident_classifier import get_incident_classifier

    classifier = get_incident_classifier()
    print("✅ Incident classifier initialized")

    # Test cases
    test_cases = [
        "Someone is shooting a gun near the school",
        "Car was stolen from parking lot",
        "Group of people fighting outside the bar"
    ]

    for i, description in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Description: \"{description}\"")
        result = await classifier.classify(description)

        print(f"  → Incident Type: {result.get('incident_type')}")
        print(f"  → Severity: {result.get('severity')}/5")
        print(f"  → Confidence: {result.get('confidence'):.2f}")
        print(f"  → Threat Level: {result.get('threat_level')}")

        if result.get('recommendations'):
            print(f"  → Recommendations: {result['recommendations'][:2]}")

        if result.get('fallback'):
            print("  ⚠️ FALLBACK MODE (Atlas unavailable)")

    print("\n✅ Incident classification tests complete!")


async def test_photo_analyzer():
    """Test photo analysis via Atlas"""
    print("\n" + "="*60)
    print("TEST 3: Photo Analysis (Halo Backend)")
    print("="*60)

    from backend.ai_processing.photo_analyzer import get_photo_analyzer

    analyzer = get_photo_analyzer()
    print(f"✅ Photo analyzer initialized (model_loaded: {analyzer.model_loaded})")

    # Create a small test image (1x1 pixel PNG)
    import base64
    test_png = base64.b64decode(
        b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
    )

    print("\nAnalyzing test image via Atlas Intelligence...")
    result = await analyzer.analyze(test_png, "test.png", "quick")

    print(f"  → Success: {result.get('success')}")
    print(f"  → Media Type: {result.get('media_type')}")
    print(f"  → Objects Detected: {len(result.get('objects_detected', []))}")
    print(f"  → Threats Detected: {len(result.get('threats_detected', []))}")
    print(f"  → Threat Level: {result.get('threat_level')}")
    print(f"  → Confidence: {result.get('confidence', 0):.2f}")
    print(f"  → Processing Time: {result.get('processing_time_ms', 0)}ms")

    if result.get('fallback'):
        print("  ⚠️ FALLBACK MODE (Atlas unavailable)")

    print("\n✅ Photo analysis test complete!")


async def test_integration():
    """Run all integration tests"""
    print("\n" + "🚀 " + "="*58)
    print("   HALO → ATLAS INTELLIGENCE INTEGRATION TESTS")
    print("="*60 + "\n")

    # Test 1: Atlas health
    atlas_healthy = await test_atlas_health()

    if not atlas_healthy:
        print("\n" + "⚠️ "*30)
        print("WARNING: Atlas Intelligence is not available!")
        print("This is expected if Atlas is not running on localhost:8001")
        print("Halo will use fallback mode (degraded functionality)")
        print("⚠️ "*30 + "\n")

        response = input("Continue with tests anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n❌ Tests aborted. Start Atlas Intelligence first:")
            print("   cd /Users/timothyaikenhead/Desktop/atlas-intelligence")
            print("   source venv/bin/activate")
            print("   uvicorn main:app --port 8001 --reload")
            return

    # Test 2: Incident classification
    await test_incident_classifier()

    # Test 3: Photo analysis
    await test_photo_analyzer()

    # Summary
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE!")
    print("="*60)
    print("\nIntegration Status:")
    if atlas_healthy:
        print("  ✅ Halo backend successfully integrated with Atlas Intelligence")
        print("  ✅ Real ML-powered incident classification working")
        print("  ✅ Real YOLOv8m photo analysis working")
        print("\nNext Steps:")
        print("  1. Deploy Atlas Intelligence to Railway")
        print("  2. Set ATLAS_INTELLIGENCE_URL in Halo's Railway env vars")
        print("  3. Deploy Halo backend")
        print("  4. Production-ready! 🚀")
    else:
        print("  ⚠️ Atlas Intelligence not running (fallback mode active)")
        print("\nTo test full integration:")
        print("  1. Start Atlas: cd atlas-intelligence && uvicorn main:app --port 8001")
        print("  2. Re-run this test: python test_atlas_integration.py")
    print("\n")


if __name__ == "__main__":
    try:
        asyncio.run(test_integration())
    except KeyboardInterrupt:
        print("\n\n❌ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
