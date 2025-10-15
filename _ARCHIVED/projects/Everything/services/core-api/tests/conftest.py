import os
import sys
from pathlib import Path
import pytest
import httpx

# Minimize external dependencies/init during tests
os.environ.setdefault("SCIP_MINIMAL_STARTUP", "1")

# Ensure the service root (containing the 'app' package) is on sys.path
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


@pytest.fixture(autouse=True)
def mock_httpx_and_env(monkeypatch):
    """Mock Octopart HTTP calls and set safe env defaults for all tests."""
    # Allow disabling mocks when explicitly requested
    if os.getenv("SCIP_DISABLE_HTTPX_MOCK") == "1":
        return
    # Ensure an API key exists so code paths don't early-exit
    os.environ.setdefault("OCTOPART_API_KEY", "test-mock-token")

    class FakeResp:
        def __init__(self, payload: dict, status_code: int = 200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("error", request=None, response=None)

        def json(self):
            return self._payload

    async def fake_post(self, url: str, json: dict | None = None, headers: dict | None = None):
        query_str = (json or {}).get("query", "")
        # Offers via mpnView
        if "mpnView" in query_str:
            payload = {
                "data": {
                    "mpnView": {
                        "part": {
                            "sellers": [
                                {
                                    "company": {"name": "Digi-Key"},
                                    "offers": [
                                        {
                                            "sku": "DK-123",
                                            "clickUrl": "https://example.com/offer",
                                            "inventoryLevel": 250,
                                            "moq": 1,
                                            "orderMultiple": 1,
                                            "leadTimeDays": 5,
                                            "prices": [
                                                {"quantity": 1, "price": 12.34},
                                                {"quantity": 100, "price": 10.99},
                                            ],
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                }
            }
            return FakeResp(payload)

        # Spec attributes via supSearchMpn specs
        if "supSearchMpn" in query_str and "specs" in query_str:
            payload = {
                "data": {
                    "supSearchMpn": {
                        "hits": 2,
                        "results": [
                            {
                                "part": {
                                    "mpn": "ADS1234",
                                    "specs": [
                                        {
                                            "attribute": {"name": "Case/Package", "id": "case_package", "shortname": "case_package"},
                                            "displayValue": "SSOP",
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                }
            }
            return FakeResp(payload)

        # Total availability via supSearchMpn totalAvail
        if "supSearchMpn" in query_str and "totalAvail" in query_str:
            payload = {
                "data": {
                    "supSearchMpn": {
                        "results": [
                            {"description": "MCU STM32F4", "part": {"totalAvail": 1234, "mpn": "STM32F429ZIT6"}},
                            {"description": "MCU STM32F4 alt", "part": {"totalAvail": 200, "mpn": "STM32F439ZIT6"}},
                        ]
                    }
                }
            }
            return FakeResp(payload)

        # Pricing via supSearchMpn sellers/offers/prices
        if "supSearchMpn" in query_str and "sellers" in query_str and "offers" in query_str:
            payload = {
                "data": {
                    "supSearchMpn": {
                        "hits": 1,
                        "results": [
                            {
                                "part": {
                                    "mpn": "STM32F429ZIT6",
                                    "sellers": [
                                        {
                                            "company": {"name": "Mouser"},
                                            "offers": [
                                                {"prices": [{"quantity": 1, "price": 13.21}, {"quantity": 100, "price": 11.05}]}
                                            ],
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                }
            }
            return FakeResp(payload)

        # Default minimal successful response
        return FakeResp({"data": {}})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post, raising=True)
