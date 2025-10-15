# Database Schemas - Supply Chain Intelligence Platform

## PostgreSQL Schema (Transactional Data)

### Core Business Entities

```sql
-- Companies (Suppliers, Customers, Manufacturers)
CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type ENUM('supplier', 'customer', 'manufacturer', 'distributor'),
    tax_id VARCHAR(50),
    website VARCHAR(255),
    headquarters_country VARCHAR(2),
    risk_score DECIMAL(3,2),
    relationship_strength DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Components/Parts
CREATE TABLE components (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    manufacturer_part_number VARCHAR(100) NOT NULL,
    manufacturer_id UUID REFERENCES companies(id),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    description TEXT,
    datasheet_url VARCHAR(500),
    lifecycle_status ENUM('active', 'nrnd', 'obsolete'),
    rohs_compliant BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RFQs (Request for Quotes)
CREATE TABLE rfqs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID REFERENCES companies(id),
    rfq_number VARCHAR(50),
    email_message_id VARCHAR(255),
    status ENUM('open', 'quoted', 'won', 'lost', 'expired'),
    required_date DATE,
    total_value DECIMAL(12,2),
    source_system VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RFQ Line Items
CREATE TABLE rfq_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rfq_id UUID REFERENCES rfqs(id),
    component_id UUID REFERENCES components(id),
    customer_part_number VARCHAR(100),
    quantity INTEGER NOT NULL,
    target_price DECIMAL(10,4),
    lead_time_weeks INTEGER,
    packaging VARCHAR(50),
    date_code_requirement VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Purchase Orders
CREATE TABLE purchase_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID REFERENCES companies(id),
    po_number VARCHAR(50) NOT NULL,
    status ENUM('draft', 'sent', 'acknowledged', 'shipped', 'received', 'cancelled'),
    total_value DECIMAL(12,2),
    currency VARCHAR(3) DEFAULT 'USD',
    payment_terms VARCHAR(100),
    delivery_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Purchase Order Line Items
CREATE TABLE po_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    po_id UUID REFERENCES purchase_orders(id),
    component_id UUID REFERENCES components(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,4),
    lead_time_weeks INTEGER,
    manufacturer_lot_code VARCHAR(50),
    date_code VARCHAR(20),
    packaging VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Price History
CREATE TABLE price_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_id UUID REFERENCES components(id),
    supplier_id UUID REFERENCES companies(id),
    quantity_break INTEGER,
    unit_price DECIMAL(10,4),
    currency VARCHAR(3) DEFAULT 'USD',
    valid_from DATE,
    valid_until DATE,
    source_type ENUM('email', 'quote', 'purchase', 'web_crawl'),
    source_reference VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inventory
CREATE TABLE inventory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_id UUID REFERENCES components(id),
    location VARCHAR(100),
    quantity_available INTEGER,
    quantity_reserved INTEGER,
    cost_per_unit DECIMAL(10,4),
    date_code VARCHAR(20),
    lot_code VARCHAR(50),
    expiry_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### AI/ML Training Data

```sql
-- Email Processing
CREATE TABLE processed_emails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id VARCHAR(255) UNIQUE,
    sender_email VARCHAR(255),
    subject TEXT,
    body_text TEXT,
    body_html TEXT,
    received_at TIMESTAMP,
    classification VARCHAR(50),
    confidence_score DECIMAL(3,2),
    extracted_entities JSONB,
    attachments JSONB,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market Intelligence
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_id UUID REFERENCES components(id),
    data_source VARCHAR(100),
    metric_type VARCHAR(50),
    metric_value DECIMAL(12,4),
    unit VARCHAR(20),
    collection_date DATE,
    source_url VARCHAR(500),
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML Model Performance
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100),
    model_version VARCHAR(20),
    metric_name VARCHAR(50),
    metric_value DECIMAL(6,4),
    evaluation_date DATE,
    dataset_size INTEGER,
    training_duration_minutes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## MongoDB Schema (Document Store)

### Email Documents
```javascript
// Email collection
{
  _id: ObjectId,
  messageId: String,
  threadId: String,
  sender: {
    email: String,
    name: String,
    company: String
  },
  recipients: [{
    email: String,
    name: String,
    type: "to" | "cc" | "bcc"
  }],
  subject: String,
  bodyText: String,
  bodyHtml: String,
  receivedAt: Date,
  attachments: [{
    filename: String,
    contentType: String,
    size: Number,
    storageUrl: String,
    extractedText: String
  }],
  classification: {
    type: "rfq" | "quote" | "purchase_order" | "price_list" | "other",
    confidence: Number,
    extractedData: {
      components: [{
        partNumber: String,
        quantity: Number,
        targetPrice: Number,
        description: String
      }],
      dueDate: Date,
      specialRequirements: [String]
    }
  },
  processingStatus: "pending" | "processed" | "error",
  createdAt: Date,
  updatedAt: Date
}
```

### Web Intelligence Documents
```javascript
// Web scraping results
{
  _id: ObjectId,
  url: String,
  domain: String,
  contentType: String,
  title: String,
  content: String,
  extractedData: {
    components: [{
      partNumber: String,
      manufacturer: String,
      price: Number,
      availability: String,
      specifications: Object
    }],
    news: [{
      headline: String,
      summary: String,
      relevanceScore: Number,
      entities: [String]
    }]
  },
  crawledAt: Date,
  lastModified: Date,
  hash: String,
  classification: {
    category: String,
    relevanceScore: Number,
    confidenceScore: Number
  }
}
```

### Geopolitical Intelligence
```javascript
// Geopolitical events
{
  _id: ObjectId,
  eventType: "trade_policy" | "sanctions" | "conflict" | "economic",
  title: String,
  description: String,
  affectedCountries: [String],
  affectedIndustries: [String],
  severityScore: Number,
  impactAssessment: {
    supplyChainImpact: Number,
    affectedComponents: [String],
    alternativeSources: [{
      country: String,
      suppliers: [String],
      feasibilityScore: Number
    }]
  },
  sources: [{
    url: String,
    credibilityScore: Number,
    publishedAt: Date
  }],
  predictedImpact: {
    timeframe: String,
    probability: Number,
    severity: Number
  },
  createdAt: Date,
  updatedAt: Date
}
```

## Neo4j Schema (Graph Database)

### Node Types and Relationships
```cypher
// Companies
CREATE CONSTRAINT company_id FOR (c:Company) REQUIRE c.id IS UNIQUE;
CREATE INDEX company_name FOR (c:Company) ON (c.name);

// Components
CREATE CONSTRAINT component_id FOR (c:Component) REQUIRE c.id IS UNIQUE;
CREATE INDEX component_mpn FOR (c:Component) ON (c.manufacturerPartNumber);

// Relationships
CREATE (supplier:Company)-[:SUPPLIES]->(component:Component)
CREATE (manufacturer:Company)-[:MANUFACTURES]->(component:Component)
CREATE (customer:Company)-[:PURCHASES]->(component:Component)
CREATE (component1:Component)-[:ALTERNATIVE_TO]->(component2:Component)
CREATE (component:Component)-[:SUCCESSOR_OF]->(oldComponent:Component)
```

### Graph Queries Examples
```cypher
// Find alternative suppliers for a component
MATCH (c:Component {manufacturerPartNumber: 'STM32F4'})-[:SUPPLIED_BY]->(s:Supplier)
RETURN s.name, s.riskScore, s.averageLeadTime
ORDER BY s.riskScore ASC, s.averageLeadTime ASC;

// Identify supply chain vulnerabilities
MATCH (country:Country)-[:HOSTS]->(supplier:Company)-[:SUPPLIES]->(component:Component)
WHERE country.riskScore > 0.7
RETURN component.manufacturerPartNumber, COUNT(supplier) as supplierCount
HAVING supplierCount < 3;
```

## Elasticsearch Schema

### Component Index
```json
{
  "mappings": {
    "properties": {
      "manufacturerPartNumber": {"type": "keyword"},
      "description": {"type": "text", "analyzer": "standard"},
      "category": {"type": "keyword"},
      "specifications": {"type": "nested"},
      "datasheet": {"type": "text"},
      "alternativeParts": {"type": "keyword"},
      "suppliers": {
        "type": "nested",
        "properties": {
          "name": {"type": "keyword"},
          "price": {"type": "double"},
          "leadTime": {"type": "integer"},
          "availability": {"type": "keyword"}
        }
      },
      "priceHistory": {
        "type": "nested",
        "properties": {
          "date": {"type": "date"},
          "price": {"type": "double"},
          "quantity": {"type": "integer"}
        }
      }
    }
  }
}
```

### Document Index
```json
{
  "mappings": {
    "properties": {
      "title": {"type": "text", "analyzer": "standard"},
      "content": {"type": "text", "analyzer": "standard"},
      "url": {"type": "keyword"},
      "domain": {"type": "keyword"},
      "publishedAt": {"type": "date"},
      "entities": {"type": "keyword"},
      "relevanceScore": {"type": "double"},
      "classification": {"type": "keyword"},
      "extractedComponents": {
        "type": "nested",
        "properties": {
          "partNumber": {"type": "keyword"},
          "context": {"type": "text"}
        }
      }
    }
  }
}
```

## Data Relationships

### Cross-Database Relationships
- PostgreSQL `component.id` ↔ MongoDB documents `extractedData.components.componentId`
- PostgreSQL `company.id` ↔ Neo4j `Company.id`
- Elasticsearch component index ↔ PostgreSQL `components` table via `manufacturerPartNumber`

### Data Flow Patterns
1. **Ingestion**: Raw data → MongoDB → Processing → PostgreSQL/Neo4j
2. **Search**: Elasticsearch indexes from PostgreSQL for fast retrieval
3. **Analytics**: Neo4j for relationship queries, PostgreSQL for aggregations
4. **ML Training**: All sources → Feature store → Model training