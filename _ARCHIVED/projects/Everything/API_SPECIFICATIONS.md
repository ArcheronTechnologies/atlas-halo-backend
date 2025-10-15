# API Specifications - Supply Chain Intelligence Platform

## API Architecture Overview

### Base Configuration
- **Protocol**: HTTPS only
- **Format**: REST with JSON payloads
- **Authentication**: OAuth 2.0 + JWT tokens
- **Rate Limiting**: 1000 requests/minute per API key
- **Versioning**: URL versioning (v1, v2, etc.)
- **Base URL**: `https://api.scip.company/v1`

## Core API Modules

### 1. Authentication & Authorization

#### POST /auth/login
```json
{
  "email": "user@company.com",
  "password": "securePassword",
  "mfaToken": "123456"
}
```
Response:
```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIs...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIs...",
  "expiresIn": 3600,
  "user": {
    "id": "uuid",
    "email": "user@company.com",
    "role": "analyst",
    "permissions": ["read:components", "write:rfqs"]
  }
}
```

#### POST /auth/refresh
```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIs..."
}
```

### 2. Component Management

#### GET /components
Query Parameters:
- `search`: Full-text search
- `category`: Component category filter
- `manufacturer`: Manufacturer ID filter
- `lifecycle`: active, nrnd, obsolete
- `limit`: Results per page (default: 50)
- `offset`: Pagination offset

Response:
```json
{
  "data": [{
    "id": "uuid",
    "manufacturerPartNumber": "STM32F429ZIT6",
    "manufacturer": {
      "id": "uuid",
      "name": "STMicroelectronics"
    },
    "category": "Microcontrollers",
    "description": "ARM Cortex-M4 MCU with FPU",
    "lifecycleStatus": "active",
    "rohsCompliant": true,
    "datasheet": "https://...",
    "specifications": {
      "coreType": "ARM Cortex-M4",
      "clockSpeed": "180MHz",
      "flashMemory": "2MB",
      "packageType": "LQFP-144"
    },
    "alternativeParts": ["STM32F439ZIT6", "STM32F469NIT6"],
    "createdAt": "2024-01-15T10:30:00Z",
    "updatedAt": "2024-01-20T15:45:00Z"
  }],
  "pagination": {
    "total": 10000,
    "limit": 50,
    "offset": 0,
    "hasMore": true
  }
}
```

#### POST /components
```json
{
  "manufacturerPartNumber": "STM32F429ZIT6",
  "manufacturerId": "uuid",
  "category": "Microcontrollers",
  "subcategory": "ARM Cortex-M4",
  "description": "ARM Cortex-M4 MCU with FPU",
  "specifications": {
    "coreType": "ARM Cortex-M4",
    "clockSpeed": "180MHz"
  },
  "datasheetUrl": "https://...",
  "rohsCompliant": true
}
```

#### GET /components/{id}/pricing
Response:
```json
{
  "componentId": "uuid",
  "currentPricing": [{
    "supplier": {
      "id": "uuid",
      "name": "Digi-Key"
    },
    "quantityBreaks": [{
      "quantity": 1,
      "unitPrice": 15.99,
      "currency": "USD"
    }, {
      "quantity": 100,
      "unitPrice": 14.50,
      "currency": "USD"
    }],
    "availability": "In Stock",
    "leadTimeWeeks": 0,
    "lastUpdated": "2024-01-20T10:00:00Z"
  }],
  "priceHistory": [{
    "date": "2024-01-01",
    "averagePrice": 16.25,
    "priceRange": {
      "min": 15.00,
      "max": 17.50
    }
  }]
}
```

### 3. RFQ Management

#### GET /rfqs
Query Parameters:
- `status`: open, quoted, won, lost, expired
- `customerId`: Filter by customer
- `dateFrom`: Start date filter
- `dateTo`: End date filter
- `sort`: created_at, value, due_date
- `order`: asc, desc

Response:
```json
{
  "data": [{
    "id": "uuid",
    "rfqNumber": "RFQ-2024-001",
    "customer": {
      "id": "uuid",
      "name": "Apple Inc."
    },
    "status": "open",
    "totalValue": 125000.00,
    "currency": "USD",
    "requiredDate": "2024-02-15",
    "itemCount": 5,
    "source": "email",
    "urgency": "high",
    "aiRecommendations": [{
      "type": "pricing_strategy",
      "message": "Historical data suggests 15% markup optimal",
      "confidence": 0.87
    }],
    "createdAt": "2024-01-20T09:15:00Z"
  }]
}
```

#### POST /rfqs
```json
{
  "customerId": "uuid",
  "rfqNumber": "RFQ-2024-001",
  "requiredDate": "2024-02-15",
  "specialRequirements": ["RoHS compliant", "Date code < 6 months"],
  "items": [{
    "componentId": "uuid",
    "customerPartNumber": "CUST-12345",
    "quantity": 1000,
    "targetPrice": 12.50,
    "leadTimeWeeks": 8,
    "packaging": "Tape & Reel"
  }],
  "source": "email",
  "sourceReference": "email-message-id-123"
}
```

#### POST /rfqs/{id}/quote
```json
{
  "quotedItems": [{
    "rfqItemId": "uuid",
    "unitPrice": 13.25,
    "totalPrice": 13250.00,
    "leadTimeWeeks": 6,
    "alternativeComponent": {
      "componentId": "uuid",
      "reason": "Better availability",
      "unitPrice": 12.90
    }
  }],
  "validUntil": "2024-02-01",
  "terms": "Net 30",
  "notes": "Special pricing for volume commitment"
}
```

### 4. AI Intelligence API

#### GET /intelligence/market-trends
Query Parameters:
- `component`: Component ID or part number
- `timeframe`: 7d, 30d, 90d, 1y
- `region`: Global, APAC, EMEA, Americas

Response:
```json
{
  "component": {
    "id": "uuid",
    "manufacturerPartNumber": "STM32F429ZIT6"
  },
  "trends": {
    "priceMovement": {
      "direction": "increasing",
      "changePercent": 12.5,
      "confidence": 0.92,
      "drivers": ["Supply shortage in Asia", "Increased automotive demand"]
    },
    "availabilityForecast": [{
      "date": "2024-02-01",
      "availability": "Constrained",
      "confidence": 0.88
    }],
    "demandForecast": {
      "nextQuarter": {
        "increase": 15,
        "confidence": 0.85
      }
    }
  },
  "recommendations": [{
    "action": "stock_up",
    "reason": "Price increase expected in 30 days",
    "urgency": "high",
    "suggestedQuantity": 5000,
    "expectedSavings": 25000.00
  }],
  "generatedAt": "2024-01-20T16:30:00Z"
}
```

#### GET /intelligence/supplier-analysis
Query Parameters:
- `supplierId`: Specific supplier analysis
- `component`: Component-specific supplier comparison
- `riskLevel`: low, medium, high

Response:
```json
{
  "supplier": {
    "id": "uuid",
    "name": "Electronic Components Ltd"
  },
  "analysis": {
    "overallScore": 8.2,
    "riskAssessment": {
      "financialHealth": {
        "score": 7.5,
        "trend": "stable",
        "indicators": ["Strong cash flow", "Low debt ratio"]
      },
      "deliveryPerformance": {
        "onTimeRate": 94.2,
        "qualityScore": 8.8,
        "communicationRating": 9.1
      },
      "geopoliticalRisk": {
        "score": 3.2,
        "factors": ["Low country risk", "Stable trade relations"]
      }
    },
    "recommendations": [{
      "action": "increase_allocation",
      "reason": "Excellent performance metrics",
      "confidence": 0.91
    }],
    "alternativeSuppliers": [{
      "id": "uuid",
      "name": "Global Electronics",
      "score": 7.8,
      "advantage": "Lower pricing"
    }]
  }
}
```

#### POST /intelligence/scenario-analysis
```json
{
  "scenario": {
    "type": "geopolitical",
    "event": "Taiwan Strait tension escalation",
    "probability": 0.3,
    "timeframe": "6_months"
  },
  "components": ["STM32F429ZIT6", "ATMEGA328P"],
  "analysisDepth": "detailed"
}
```

Response:
```json
{
  "scenarioId": "uuid",
  "impact": {
    "overall": "high",
    "confidence": 0.85,
    "affectedComponents": [{
      "componentId": "uuid",
      "impact": "severe",
      "currentSuppliers": 3,
      "atRiskSuppliers": 2,
      "alternativeOptions": [{
        "supplierId": "uuid",
        "location": "Europe",
        "capacityAvailable": true,
        "priceImpact": 15.5
      }]
    }],
    "recommendations": [{
      "priority": "urgent",
      "action": "diversify_suppliers",
      "timeline": "immediate",
      "estimatedCost": 50000.00,
      "riskReduction": 70
    }]
  }
}
```

### 5. Data Ingestion API

#### POST /ingestion/email-batch
```json
{
  "emails": [{
    "messageId": "email-123",
    "sender": "supplier@company.com",
    "subject": "Price List Update Q1 2024",
    "body": "Please find attached...",
    "receivedAt": "2024-01-20T10:00:00Z",
    "attachments": [{
      "filename": "pricelist.pdf",
      "contentType": "application/pdf",
      "base64Content": "JVBERi0xLjQK..."
    }]
  }]
}
```

#### POST /ingestion/web-data
```json
{
  "url": "https://digikey.com/products/...",
  "contentType": "product_listing",
  "extractedData": {
    "components": [{
      "partNumber": "STM32F429ZIT6",
      "price": 15.99,
      "availability": "In Stock",
      "leadTime": "Ships Today"
    }]
  },
  "crawledAt": "2024-01-20T12:00:00Z"
}
```

### 6. Real-time Notifications

#### WebSocket: /ws/notifications
Authentication: JWT token in query parameter

Message Types:
```json
// Price Alert
{
  "type": "price_alert",
  "componentId": "uuid",
  "partNumber": "STM32F429ZIT6",
  "priceChange": {
    "oldPrice": 15.99,
    "newPrice": 17.25,
    "changePercent": 7.9
  },
  "supplier": "Digi-Key",
  "timestamp": "2024-01-20T14:30:00Z"
}

// Supply Chain Alert
{
  "type": "supply_alert",
  "severity": "high",
  "title": "Semiconductor shortage detected",
  "affectedComponents": ["STM32F429ZIT6", "ATMEGA328P"],
  "estimatedImpact": "2-3 month delay",
  "recommendations": ["Stock up immediately", "Find alternatives"],
  "timestamp": "2024-01-20T15:45:00Z"
}

// RFQ Update
{
  "type": "rfq_update",
  "rfqId": "uuid",
  "status": "quoted",
  "value": 125000.00,
  "aiRecommendations": ["Accept quote", "High win probability"],
  "timestamp": "2024-01-20T16:00:00Z"
}
```

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [{
      "field": "quantity",
      "message": "Must be greater than 0"
    }],
    "requestId": "req-uuid-123",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `409`: Conflict
- `429`: Rate Limited
- `500`: Internal Server Error
- `503`: Service Unavailable

## API Security

### Authentication Headers
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
X-API-Key: scip-api-key-123
X-Request-ID: req-uuid-123
```

### Request Signing (for sensitive operations)
```
X-Signature: sha256=a8b4c2d1e3f5g7h9...
X-Timestamp: 1642684800
```

### Rate Limiting Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642684860
```

## SDK Examples

### Python SDK Usage
```python
from scip_sdk import SCIPClient

client = SCIPClient(
    api_key="your-api-key",
    base_url="https://api.scip.company/v1"
)

# Search components
components = client.components.search(
    query="STM32F4",
    category="Microcontrollers"
)

# Get market intelligence
trends = client.intelligence.market_trends(
    component_id=components[0].id,
    timeframe="30d"
)

# Create RFQ
rfq = client.rfqs.create({
    "customer_id": "uuid",
    "items": [{
        "component_id": "uuid",
        "quantity": 1000
    }]
})
```

### JavaScript SDK Usage
```javascript
import { SCIPClient } from '@scip/sdk';

const client = new SCIPClient({
    apiKey: 'your-api-key',
    baseUrl: 'https://api.scip.company/v1'
});

// Real-time notifications
const notifications = client.notifications.subscribe({
    types: ['price_alert', 'supply_alert']
});

notifications.on('price_alert', (alert) => {
    console.log('Price changed:', alert);
});
```