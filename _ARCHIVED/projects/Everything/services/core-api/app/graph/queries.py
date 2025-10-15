"""
Neo4j Graph Queries

Advanced graph queries for supply chain intelligence including supplier
relationships, component alternatives, market analysis, and risk assessment.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .neo4j_client import Neo4jClient, QueryResult
# from .models import NodeType, RelationType
# Temporarily inline the required enums
from enum import Enum

class NodeType(Enum):
    COMPONENT = "Component"
    COMPANY = "Company" 
    USER = "User"

class RelationType(Enum):
    SUPPLIES = "SUPPLIES"
    MANUFACTURES = "MANUFACTURES"

logger = logging.getLogger(__name__)


class GraphQueries:
    """Advanced graph query operations for supply chain intelligence"""
    
    def __init__(self, client: Neo4jClient):
        self.client = client
    
    # Supplier and Customer Queries
    
    async def find_suppliers_for_component(
        self,
        component_id: str,
        include_inactive: bool = False,
        max_lead_time_days: Optional[int] = None,
        max_price: Optional[float] = None,
        min_availability_score: Optional[float] = None
    ) -> QueryResult:
        """Find all suppliers for a specific component with filtering options"""
        
        query = """
        MATCH (c:Component {id: $component_id})
        MATCH (c)<-[s:SUPPLIES]-(supplier:Company)
        WHERE ($include_inactive OR s.availability_status <> 'inactive')
        AND ($max_lead_time_days IS NULL OR s.lead_time_days <= $max_lead_time_days)
        AND ($max_price IS NULL OR s.price <= $max_price)
        OPTIONAL MATCH (supplier)-[:LOCATED_IN]->(location:Location)
        RETURN 
            supplier.id AS supplier_id,
            supplier.name AS supplier_name,
            supplier.type AS supplier_type,
            supplier.country AS country,
            supplier.risk_score AS risk_score,
            s.price AS price,
            s.currency AS currency,
            s.lead_time_days AS lead_time_days,
            s.minimum_order_quantity AS moq,
            s.availability_status AS availability,
            s.last_updated AS last_updated,
            location.name AS location_name
        ORDER BY 
            CASE WHEN s.price IS NULL THEN 1 ELSE 0 END,
            s.price ASC,
            s.lead_time_days ASC,
            supplier.risk_score ASC
        """
        
        parameters = {
            'component_id': component_id,
            'include_inactive': include_inactive,
            'max_lead_time_days': max_lead_time_days,
            'max_price': max_price,
            'min_availability_score': min_availability_score
        }
        
        return await self.client.execute_query(query, parameters)
    
    async def find_alternative_suppliers(
        self,
        primary_supplier_id: str,
        component_categories: Optional[List[str]] = None,
        exclude_regions: Optional[List[str]] = None,
        min_risk_score: Optional[float] = None
    ) -> QueryResult:
        """Find alternative suppliers for components supplied by a primary supplier"""
        
        query = """
        MATCH (primary:Company {id: $primary_supplier_id})-[s1:SUPPLIES]->(c:Component)
        MATCH (c)<-[s2:SUPPLIES]-(alternative:Company)
        WHERE alternative.id <> primary.id
        AND ($component_categories IS NULL OR c.category IN $component_categories)
        AND ($exclude_regions IS NULL OR NOT alternative.region IN $exclude_regions)
        AND ($min_risk_score IS NULL OR alternative.risk_score >= $min_risk_score)
        WITH 
            alternative,
            count(c) AS shared_components,
            collect({
                component_id: c.id,
                component_name: c.description,
                primary_price: s1.price,
                alternative_price: s2.price,
                price_difference: s2.price - s1.price,
                primary_lead_time: s1.lead_time_days,
                alternative_lead_time: s2.lead_time_days
            }) AS component_comparisons
        RETURN 
            alternative.id AS supplier_id,
            alternative.name AS supplier_name,
            alternative.country AS country,
            alternative.region AS region,
            alternative.risk_score AS risk_score,
            shared_components,
            component_comparisons
        ORDER BY shared_components DESC, alternative.risk_score ASC
        LIMIT 20
        """
        
        parameters = {
            'primary_supplier_id': primary_supplier_id,
            'component_categories': component_categories,
            'exclude_regions': exclude_regions,
            'min_risk_score': min_risk_score
        }
        
        return await self.client.execute_query(query, parameters)
    
    async def analyze_supplier_dependencies(
        self, 
        customer_id: str,
        depth: int = 2
    ) -> QueryResult:
        """Analyze supplier dependency risks for a customer"""
        
        query = """
        MATCH (customer:Company {id: $customer_id})-[:CREATED_BY|SUBMITTED_TO*1..2]->
              (rfq:RFQ)-[:CONTAINS]->(c:Component)<-[s:SUPPLIES]-(supplier:Company)
        WITH customer, supplier, collect(DISTINCT c) AS components, count(DISTINCT rfq) AS rfq_count
        OPTIONAL MATCH (supplier)<-[:SUPPLIES]-(competitor:Company)-[:SUPPLIES]->(alt:Component)
        WHERE alt.category IN [comp IN components | comp.category]
        WITH 
            customer,
            supplier,
            components,
            rfq_count,
            count(DISTINCT competitor) AS alternative_suppliers,
            count(DISTINCT alt) AS alternative_components
        RETURN 
            supplier.id AS supplier_id,
            supplier.name AS supplier_name,
            supplier.country AS country,
            supplier.risk_score AS risk_score,
            size(components) AS component_count,
            rfq_count,
            alternative_suppliers,
            alternative_components,
            CASE 
                WHEN alternative_suppliers = 0 THEN 'CRITICAL'
                WHEN alternative_suppliers <= 2 THEN 'HIGH'
                WHEN alternative_suppliers <= 5 THEN 'MEDIUM'
                ELSE 'LOW'
            END AS dependency_risk
        ORDER BY 
            CASE dependency_risk 
                WHEN 'CRITICAL' THEN 1 
                WHEN 'HIGH' THEN 2 
                WHEN 'MEDIUM' THEN 3 
                ELSE 4 
            END,
            component_count DESC
        """
        
        parameters = {
            'customer_id': customer_id,
            'depth': depth
        }
        
        return await self.client.execute_query(query, parameters)
    
    # Component Alternative Queries
    
    async def find_component_alternatives(
        self,
        component_id: str,
        min_confidence: float = 0.7,
        compatibility_levels: Optional[List[str]] = None,
        verified_only: bool = False
    ) -> QueryResult:
        """Find alternative components with compatibility analysis"""
        
        query = """
        MATCH (c:Component {id: $component_id})
        MATCH (c)-[alt:ALTERNATIVE_TO]-(alternative:Component)
        WHERE alt.confidence_score >= $min_confidence
        AND ($compatibility_levels IS NULL OR alt.compatibility_level IN $compatibility_levels)
        AND ($verified_only = false OR alt.verified = true)
        OPTIONAL MATCH (alternative)<-[s:SUPPLIES]-(supplier:Company)
        WITH 
            alternative,
            alt,
            collect({
                supplier_id: supplier.id,
                supplier_name: supplier.name,
                price: s.price,
                currency: s.currency,
                lead_time: s.lead_time_days,
                availability: s.availability_status
            }) AS suppliers
        RETURN 
            alternative.id AS component_id,
            alternative.manufacturer_part_number AS part_number,
            alternative.description AS description,
            alternative.category AS category,
            alternative.specifications AS specifications,
            alt.confidence_score AS confidence_score,
            alt.compatibility_level AS compatibility_level,
            alt.verified AS verified,
            alt.verified_by AS verified_by,
            alt.notes AS notes,
            suppliers,
            size(suppliers) AS supplier_count
        ORDER BY alt.confidence_score DESC, size(suppliers) DESC
        """
        
        parameters = {
            'component_id': component_id,
            'min_confidence': min_confidence,
            'compatibility_levels': compatibility_levels,
            'verified_only': verified_only
        }
        
        return await self.client.execute_query(query, parameters)
    
    async def suggest_component_alternatives(
        self,
        component_id: str,
        specification_weights: Optional[Dict[str, float]] = None
    ) -> QueryResult:
        """AI-powered component alternative suggestions based on specifications"""
        
        # This query uses specification similarity scoring
        query = """
        MATCH (source:Component {id: $component_id})
        MATCH (candidate:Component)
        WHERE candidate.id <> source.id 
        AND candidate.category = source.category
        
        // Calculate specification similarity score
        WITH source, candidate,
        REDUCE(score = 0.0, key IN keys(source.specifications) |
            CASE 
                WHEN key IN keys(candidate.specifications) 
                THEN score + 
                    CASE 
                        WHEN source.specifications[key] = candidate.specifications[key] THEN 1.0
                        WHEN toString(source.specifications[key]) CONTAINS toString(candidate.specifications[key]) 
                             OR toString(candidate.specifications[key]) CONTAINS toString(source.specifications[key])
                        THEN 0.7
                        ELSE 0.0
                    END
                ELSE score
            END
        ) AS similarity_score,
        size(keys(source.specifications)) AS total_specs
        
        WHERE similarity_score / total_specs >= 0.6  // 60% similarity threshold
        
        OPTIONAL MATCH (candidate)<-[s:SUPPLIES]-(supplier:Company)
        WITH 
            candidate,
            similarity_score / total_specs AS confidence_score,
            collect({
                supplier_id: supplier.id,
                supplier_name: supplier.name,
                price: s.price,
                lead_time: s.lead_time_days
            }) AS suppliers
        
        RETURN 
            candidate.id AS component_id,
            candidate.manufacturer_part_number AS part_number,
            candidate.description AS description,
            candidate.specifications AS specifications,
            confidence_score,
            suppliers,
            size(suppliers) AS supplier_count
        ORDER BY confidence_score DESC, size(suppliers) DESC
        LIMIT 10
        """
        
        parameters = {
            'component_id': component_id
        }
        
        return await self.client.execute_query(query, parameters)
    
    # Market Analysis Queries
    
    async def analyze_component_market(
        self,
        component_category: str,
        time_window_days: int = 90,
        include_price_trends: bool = True
    ) -> QueryResult:
        """Analyze market conditions for a component category"""
        
        query = """
        MATCH (c:Component {category: $component_category})
        OPTIONAL MATCH (c)<-[s:SUPPLIES]-(supplier:Company)
        WHERE s.last_updated >= date() - duration({days: $time_window_days})
        
        WITH 
            c,
            collect({
                supplier_id: supplier.id,
                supplier_name: supplier.name,
                price: s.price,
                currency: s.currency,
                lead_time: s.lead_time_days,
                last_updated: s.last_updated
            }) AS current_suppliers
        
        // Get historical pricing if requested
        OPTIONAL MATCH (c)<-[:AFFECTS]-(price_event:MarketEvent)
        WHERE $include_price_trends = true 
        AND price_event.event_type = 'price_change'
        AND price_event.created_at >= date() - duration({days: $time_window_days})
        
        RETURN 
            c.id AS component_id,
            c.manufacturer_part_number AS part_number,
            c.description AS description,
            current_suppliers,
            size(current_suppliers) AS supplier_count,
            
            // Market metrics
            CASE WHEN size(current_suppliers) > 0 
                THEN [s IN current_suppliers WHERE s.price IS NOT NULL | s.price]
                ELSE []
            END AS prices,
            
            CASE WHEN size([s IN current_suppliers WHERE s.price IS NOT NULL]) > 0
                THEN reduce(sum = 0.0, s IN current_suppliers | 
                    sum + CASE WHEN s.price IS NOT NULL THEN s.price ELSE 0 END
                ) / size([s IN current_suppliers WHERE s.price IS NOT NULL])
                ELSE null
            END AS average_price,
            
            CASE WHEN size([s IN current_suppliers WHERE s.lead_time IS NOT NULL]) > 0
                THEN reduce(sum = 0, s IN current_suppliers | 
                    sum + CASE WHEN s.lead_time IS NOT NULL THEN s.lead_time ELSE 0 END
                ) / size([s IN current_suppliers WHERE s.lead_time IS NOT NULL])
                ELSE null
            END AS average_lead_time,
            
            collect(price_event) AS price_events
            
        ORDER BY supplier_count DESC, average_price ASC
        """
        
        parameters = {
            'component_category': component_category,
            'time_window_days': time_window_days,
            'include_price_trends': include_price_trends
        }
        
        return await self.client.execute_query(query, parameters)
    
    async def identify_supply_risks(
        self,
        regions: Optional[List[str]] = None,
        component_categories: Optional[List[str]] = None,
        risk_threshold: float = 0.7
    ) -> QueryResult:
        """Identify supply chain risks by region and category"""
        
        query = """
        MATCH (supplier:Company)-[s:SUPPLIES]->(c:Component)
        WHERE ($regions IS NULL OR supplier.region IN $regions)
        AND ($component_categories IS NULL OR c.category IN $component_categories)
        AND supplier.risk_score >= $risk_threshold
        
        // Find alternative suppliers for risk assessment
        MATCH (c)<-[alt_s:SUPPLIES]-(alt_supplier:Company)
        WHERE alt_supplier.id <> supplier.id
        AND alt_supplier.risk_score < $risk_threshold
        
        WITH 
            supplier,
            c,
            s,
            count(DISTINCT alt_supplier) AS alternative_count,
            collect(DISTINCT alt_supplier.region) AS alternative_regions
        
        // Check for geopolitical concentrations
        WITH 
            supplier,
            c,
            s,
            alternative_count,
            alternative_regions,
            CASE 
                WHEN alternative_count = 0 THEN 'CRITICAL'
                WHEN alternative_count <= 2 THEN 'HIGH'
                WHEN alternative_count <= 5 THEN 'MEDIUM'
                ELSE 'LOW'
            END AS supply_risk_level
        
        RETURN 
            c.category AS component_category,
            c.id AS component_id,
            c.manufacturer_part_number AS part_number,
            supplier.id AS supplier_id,
            supplier.name AS supplier_name,
            supplier.country AS supplier_country,
            supplier.region AS supplier_region,
            supplier.risk_score AS risk_score,
            s.price AS price,
            s.lead_time_days AS lead_time,
            alternative_count,
            alternative_regions,
            supply_risk_level
        ORDER BY 
            CASE supply_risk_level 
                WHEN 'CRITICAL' THEN 1 
                WHEN 'HIGH' THEN 2 
                WHEN 'MEDIUM' THEN 3 
                ELSE 4 
            END,
            supplier.risk_score DESC
        """
        
        parameters = {
            'regions': regions,
            'component_categories': component_categories,
            'risk_threshold': risk_threshold
        }
        
        return await self.client.execute_query(query, parameters)
    
    # Advanced Relationship Queries
    
    async def find_supplier_network(
        self,
        supplier_id: str,
        network_depth: int = 2
    ) -> QueryResult:
        """Map supplier's network including partners, competitors, and shared customers"""
        
        query = """
        MATCH (supplier:Company {id: $supplier_id})
        
        // Find direct relationships
        OPTIONAL MATCH (supplier)-[partner:PARTNERS_WITH]-(partners:Company)
        OPTIONAL MATCH (supplier)-[compete:COMPETES_WITH]-(competitors:Company)
        
        // Find shared customers through RFQs
        OPTIONAL MATCH (supplier)<-[:SUBMITTED_TO]-(rfq:RFQ)-[:CREATED_BY]->(customer:Company)
        OPTIONAL MATCH (customer)-[:CREATED_BY]->(other_rfq:RFQ)-[:SUBMITTED_TO]->(other_supplier:Company)
        WHERE other_supplier.id <> supplier.id
        
        WITH 
            supplier,
            collect(DISTINCT {
                partner_id: partners.id,
                partner_name: partners.name,
                partnership_type: partner.partnership_type,
                partnership_status: partner.partnership_status
            }) AS partnerships,
            collect(DISTINCT {
                competitor_id: competitors.id,
                competitor_name: competitors.name,
                market_overlap: compete.market_overlap,
                competitive_strength: compete.competitive_strength
            }) AS competitors,
            collect(DISTINCT {
                customer_id: customer.id,
                customer_name: customer.name
            }) AS customers,
            collect(DISTINCT {
                competitor_supplier_id: other_supplier.id,
                competitor_supplier_name: other_supplier.name
            }) AS competitor_suppliers
        
        // Find component overlap with competitors
        MATCH (supplier)-[:SUPPLIES]->(comp:Component)<-[:SUPPLIES]-(comp_supplier:Company)
        WHERE comp_supplier.id <> supplier.id
        WITH 
            supplier, partnerships, competitors, customers, competitor_suppliers,
            collect({
                competitor_id: comp_supplier.id,
                competitor_name: comp_supplier.name,
                shared_component: comp.id,
                shared_component_name: comp.description
            }) AS component_competitors
        
        RETURN 
            supplier.id AS supplier_id,
            supplier.name AS supplier_name,
            partnerships,
            competitors,
            customers,
            competitor_suppliers,
            component_competitors,
            size(partnerships) AS partnership_count,
            size(competitors) AS competitor_count,
            size(customers) AS customer_count
        """
        
        parameters = {
            'supplier_id': supplier_id,
            'network_depth': network_depth
        }
        
        return await self.client.execute_query(query, parameters)
    
    async def recommend_strategic_partnerships(
        self,
        company_id: str,
        partnership_types: Optional[List[str]] = None,
        min_market_overlap: float = 0.3,
        exclude_existing: bool = True
    ) -> QueryResult:
        """Recommend potential strategic partnerships based on market analysis"""
        
        query = """
        MATCH (company:Company {id: $company_id})
        
        // Find companies in similar markets
        MATCH (company)-[:SUPPLIES|MANUFACTURES]->(comp:Component)
        MATCH (comp)<-[:SUPPLIES|MANUFACTURES]-(potential:Company)
        WHERE potential.id <> company.id
        
        // Calculate market overlap
        WITH 
            company, potential,
            collect(DISTINCT comp.category) AS company_categories,
            count(DISTINCT comp) AS shared_components
        
        MATCH (potential)-[:SUPPLIES|MANUFACTURES]->(pot_comp:Component)
        WITH 
            company, potential, company_categories, shared_components,
            collect(DISTINCT pot_comp.category) AS potential_categories,
            count(DISTINCT pot_comp) AS potential_components
        
        WITH 
            company, potential, shared_components, potential_components,
            [cat IN company_categories WHERE cat IN potential_categories] AS overlap_categories,
            company_categories, potential_categories
        
        // Calculate overlap metrics
        WITH 
            company, potential, shared_components,
            size(overlap_categories) AS category_overlap,
            size(company_categories) AS company_category_count,
            size(potential_categories) AS potential_category_count,
            (shared_components * 1.0) / (potential_components * 1.0) AS component_overlap_ratio
        
        WHERE (category_overlap * 1.0) / (company_category_count * 1.0) >= $min_market_overlap
        
        // Exclude existing partnerships if requested
        """ + ("""
        AND $exclude_existing = false 
        OR NOT EXISTS {
            MATCH (company)-[:PARTNERS_WITH]-(potential)
        }
        """ if exclude_existing else "") + """
        
        // Get additional company metrics
        OPTIONAL MATCH (potential)-[:LOCATED_IN]->(location)
        
        RETURN 
            potential.id AS partner_id,
            potential.name AS partner_name,
            potential.type AS partner_type,
            potential.country AS country,
            potential.region AS region,
            potential.risk_score AS risk_score,
            shared_components,
            category_overlap,
            component_overlap_ratio,
            overlap_categories AS shared_categories,
            
            // Partnership potential score
            (category_overlap * 1.0) / (company_category_count * 1.0) * 0.4 +
            component_overlap_ratio * 0.3 +
            (1.0 - COALESCE(potential.risk_score, 0.5)) * 0.3 AS partnership_score
            
        ORDER BY partnership_score DESC
        LIMIT 15
        """
        
        parameters = {
            'company_id': company_id,
            'partnership_types': partnership_types,
            'min_market_overlap': min_market_overlap,
            'exclude_existing': exclude_existing
        }
        
        return await self.client.execute_query(query, parameters)