#!/usr/bin/env python3
"""Direct test of Octopart API integration functions"""

import asyncio
import sys
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Set environment
os.environ.setdefault("SCIP_MINIMAL_STARTUP", "1")

from app.integrations.market_providers.octopart_total import fetch_total_availability
from app.integrations.market_providers.octopart_pricing import fetch_pricing_breaks
from app.integrations.market_providers.octopart_offers import fetch_offers
from app.integrations.market_providers.octopart_specs import fetch_spec_attributes


async def test_octopart_integration():
    """Test all Octopart API endpoints directly"""
    
    print("🔍 Testing Octopart API Integration")
    print("=" * 50)
    
    # Test 1: Total Availability
    print("\n1. Testing Total Availability for STM32F429ZIT6...")
    try:
        result = await fetch_total_availability("STM32F429ZIT6", country="US", limit=3)
        print(f"✅ Success! Found {len(result.get('parts', []))} parts")
        print(f"   Total availability: {result.get('sumTotalAvail', 0)}")
        for part in result.get('parts', [])[:2]:  # Show first 2
            print(f"   - {part.get('mpn')}: {part.get('totalAvail')} available")
        if result.get('note'):
            print(f"   Note: {result['note']}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: Pricing Breaks
    print("\n2. Testing Pricing Breaks for STM32...")
    try:
        result = await fetch_pricing_breaks("STM32", limit=2, currency="USD")
        print(f"✅ Success! Found {result.get('hits', 0)} hits")
        print(f"   Parts with pricing: {len(result.get('parts', []))}")
        for part in result.get('parts', [])[:1]:  # Show first part
            print(f"   Part: {part.get('mpn')}")
            for seller in part.get('sellers', [])[:2]:  # First 2 sellers
                print(f"     Seller: {seller.get('company')}")
                print(f"     Price breaks: {len(seller.get('priceBreaks', []))}")
        if result.get('note'):
            print(f"   Note: {result['note']}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: Detailed Offers
    print("\n3. Testing Detailed Offers for STM32F429ZIT6...")
    try:
        result = await fetch_offers("STM32F429ZIT6", country="US", currency="USD")
        print(f"✅ Success! Found {len(result.get('sellers', []))} sellers")
        for seller in result.get('sellers', [])[:2]:  # First 2 sellers
            print(f"   Seller: {seller.get('company')}")
            print(f"   Offers: {len(seller.get('offers', []))}")
            for offer in seller.get('offers', [])[:1]:  # First offer
                print(f"     Stock: {offer.get('inStockQuantity')}")
                print(f"     MOQ: {offer.get('moq')}")
                print(f"     Price breaks: {len(offer.get('priceBreaks', []))}")
        if result.get('note'):
            print(f"   Note: {result['note']}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 4: Spec Attributes
    print("\n4. Testing Spec Attributes for ADS with SSOP package...")
    try:
        filters = {"case_package": ["SSOP"]}
        result = await fetch_spec_attributes("ADS", filters=filters, limit=2)
        print(f"✅ Success! Found {result.get('hits', 0)} hits")
        print(f"   Parts with specs: {len(result.get('parts', []))}")
        for part in result.get('parts', [])[:1]:  # First part
            print(f"   Part: {part.get('mpn')}")
            print(f"   Specs: {len(part.get('specs', []))}")
            for spec in part.get('specs', [])[:3]:  # First 3 specs
                attr = spec.get('attribute', {})
                print(f"     {attr.get('name')}: {spec.get('displayValue')}")
        if result.get('note'):
            print(f"   Note: {result['note']}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 50)
    print("🎉 Octopart API testing complete!")


if __name__ == "__main__":
    asyncio.run(test_octopart_integration())