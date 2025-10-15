from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..db.session import get_session
from ..repositories.components_repo import ComponentsRepository
from ..repositories.companies_repo import CompaniesRepository
from ..repositories.rfqs_repo import RFQsRepository
from ..db.models import Component as ComponentORM
from ..core.auth import require_scopes, require_api_key_or_bearer
from ..search.indexer import search_components as es_search
from ..db.mongo import get_mongo_db
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/components", dependencies=[Depends(require_scopes(["read:components"]))])
def search_components(query: str = Query(..., min_length=2), limit: int = 20, offset: int = 0, session: Session = Depends(get_session)):
    repo = ComponentsRepository(session)
    ids = es_search(query, limit=limit, offset=offset)
    if ids:
        items = repo.get_by_ids(ids)
    else:
        # Fallback to DB ILIKE search
        items, _ = repo.list(search=query, limit=limit, offset=offset)
    results: List[dict] = [
        {
            "id": c.id,
            "manufacturerPartNumber": c.manufacturer_part_number,
            "category": c.category,
            "description": c.description,
        }
        for c in items
    ]
    return {"data": results}


# Advanced Search Models
class GlobalSearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    entity_types: Optional[List[str]] = None  # components, companies, rfqs, emails, etc.
    limit: int = 20
    offset: int = 0
    include_ai_insights: bool = False


class SearchFacet(BaseModel):
    name: str
    values: List[Dict[str, Any]]


class GlobalSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    facets: List[SearchFacet]
    search_time_ms: float
    ai_insights: Optional[Dict[str, Any]] = None


# Global Search Endpoint
@router.get("/")
async def global_search(
    q: str = Query(..., min_length=2, description="Search query"),
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types to search"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_ai: bool = Query(False, description="Include AI-powered insights"),
    current_user: Dict = Depends(require_api_key_or_bearer),
    session: Session = Depends(get_session)
):
    """
    Global search across all platform entities
    """
    try:
        import time
        start_time = time.time()
        
        # Parse entity types
        search_entities = ["components", "companies", "rfqs"]
        if entity_types:
            search_entities = [e.strip() for e in entity_types.split(",")]
        
        results = []
        total_count = 0
        facets = []
        
        # Search components
        if "components" in search_entities:
            components_repo = ComponentsRepository(session)
            components, component_count = components_repo.list(search=q, limit=limit//len(search_entities), offset=offset)
            for component in components:
                results.append({
                    "id": component.id,
                    "type": "component",
                    "title": component.manufacturer_part_number,
                    "description": component.description,
                    "category": component.category,
                    "url": f"/v1/components/{component.id}",
                    "score": 1.0
                })
            total_count += component_count
        
        # Search companies
        if "companies" in search_entities:
            companies_repo = CompaniesRepository(session)
            companies = companies_repo.list(search=q)[:limit//len(search_entities)]
            for company in companies:
                results.append({
                    "id": company.id,
                    "type": "company",
                    "title": company.name,
                    "description": f"{company.type} - {getattr(company, 'country', 'Unknown')}",
                    "website": company.website,
                    "url": f"/v1/companies/{company.id}",
                    "score": 0.9
                })
            total_count += len(companies)
        
        # Search RFQs
        if "rfqs" in search_entities:
            rfqs_repo = RFQsRepository(session)
            rfqs = rfqs_repo.list()[:limit//len(search_entities)]
            for rfq in rfqs:
                # Basic search filter (simplified since list() doesn't accept search)
                if q.lower() in (rfq.rfq_number or "").lower():
                    results.append({
                        "id": rfq.id,
                        "type": "rfq",
                        "title": f"RFQ {rfq.rfq_number}",
                        "description": f"Status: {rfq.status}",
                        "customer_id": rfq.customer_id,
                        "url": f"/v1/rfqs/{rfq.id}",
                        "score": 0.8
                    })
            total_count += len([r for r in results if r.get("type") == "rfq"])
        
        # Search MongoDB collections (emails, teams messages, etc.)
        mongo = get_mongo_db()
        if mongo and any(et in ["emails", "teams", "web_intelligence"] for et in search_entities):
            mongo_results = await _search_mongo_collections(mongo, q, search_entities, limit//2)
            results.extend(mongo_results)
        
        # Sort by score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:limit]
        
        # Generate facets
        facets = _generate_search_facets(results)
        
        # AI insights if requested
        ai_insights = None
        if include_ai:
            ai_insights = await _generate_ai_insights(q, results)
        
        search_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "total_count": total_count,
            "facets": facets,
            "search_time_ms": search_time,
            "ai_insights": ai_insights
        }
        
    except Exception as e:
        logger.error(f"Global search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.post("/advanced")
async def advanced_search(
    request: GlobalSearchRequest,
    current_user: Dict = Depends(require_api_key_or_bearer),
    session: Session = Depends(get_session)
):
    """
    Advanced search with complex filters and AI insights
    """
    try:
        import time
        start_time = time.time()
        
        results = []
        total_count = 0
        
        # Apply filters and entity type restrictions
        entity_types = request.entity_types or ["components", "companies", "rfqs"]
        
        # Enhanced search with filters
        if "components" in entity_types:
            components_results = await _search_components_advanced(
                session, request.query, request.filters, request.limit, request.offset
            )
            results.extend(components_results)
        
        if "companies" in entity_types:
            companies_results = await _search_companies_advanced(
                session, request.query, request.filters, request.limit, request.offset
            )
            results.extend(companies_results)
        
        # Apply MongoDB searches
        mongo = get_mongo_db()
        if mongo:
            mongo_results = await _search_mongo_advanced(
                mongo, request.query, request.filters, entity_types, request.limit
            )
            results.extend(mongo_results)
        
        # Generate facets
        facets = _generate_search_facets(results)
        
        # AI insights
        ai_insights = None
        if request.include_ai_insights:
            ai_insights = await _generate_ai_insights(request.query, results)
        
        search_time = (time.time() - start_time) * 1000
        
        return GlobalSearchResponse(
            results=results[:request.limit],
            total_count=len(results),
            facets=facets,
            search_time_ms=search_time,
            ai_insights=ai_insights
        )
        
    except Exception as e:
        logger.error(f"Advanced search error: {e}")
        raise HTTPException(status_code=500, detail="Advanced search failed")


# Helper functions

async def _search_mongo_collections(mongo, query: str, entity_types: List[str], limit: int) -> List[Dict]:
    """Search MongoDB collections"""
    results = []
    
    try:
        # Search emails
        if "emails" in entity_types:
            emails_coll = mongo.get_collection("ingested_emails")
            email_cursor = emails_coll.find({
                "$or": [
                    {"subject": {"$regex": query, "$options": "i"}},
                    {"body": {"$regex": query, "$options": "i"}}
                ]
            }).limit(limit // 3)
            
            async for email in email_cursor:
                results.append({
                    "id": str(email.get("_id")),
                    "type": "email",
                    "title": email.get("subject", "No Subject"),
                    "description": email.get("body", "")[:200],
                    "sender": email.get("sender"),
                    "url": f"/v1/emails/{email.get('_id')}",
                    "score": 0.7
                })
        
        # Search Teams messages
        if "teams" in entity_types:
            teams_coll = mongo.get_collection("teams_messages")
            teams_cursor = teams_coll.find({
                "content": {"$regex": query, "$options": "i"}
            }).limit(limit // 3)
            
            async for message in teams_cursor:
                results.append({
                    "id": str(message.get("_id")),
                    "type": "teams_message",
                    "title": f"Teams Message from {message.get('sender')}",
                    "description": message.get("content", "")[:200],
                    "channel": message.get("channelId"),
                    "url": f"/v1/teams/{message.get('_id')}",
                    "score": 0.6
                })
        
        # Search web intelligence
        if "web_intelligence" in entity_types:
            web_coll = mongo.get_collection("web_intelligence")
            web_cursor = web_coll.find({
                "$or": [
                    {"content": {"$regex": query, "$options": "i"}},
                    {"url": {"$regex": query, "$options": "i"}}
                ]
            }).limit(limit // 3)
            
            async for web_doc in web_cursor:
                results.append({
                    "id": str(web_doc.get("_id")),
                    "type": "web_intelligence",
                    "title": web_doc.get("url", "Web Document"),
                    "description": web_doc.get("content", "")[:200],
                    "source_type": web_doc.get("sourceType"),
                    "url": web_doc.get("url"),
                    "score": 0.5
                })
        
    except Exception as e:
        logger.error(f"MongoDB search error: {e}")
    
    return results


async def _search_components_advanced(session: Session, query: str, filters: Dict, limit: int, offset: int) -> List[Dict]:
    """Advanced component search with filters"""
    components_repo = ComponentsRepository(session)
    
    # Use the correct method signature for ComponentsRepository.list()
    components, _ = components_repo.list(search=query, limit=limit, offset=offset)
    
    # Apply filters manually since the repo doesn't support them directly
    if filters:
        if "category" in filters:
            components = [c for c in components if c.category == filters["category"]]
        if "manufacturer" in filters:
            components = [c for c in components if getattr(c, 'manufacturer', '') == filters["manufacturer"]]
    
    return [{
        "id": c.id,
        "type": "component",
        "title": c.manufacturer_part_number,
        "description": c.description,
        "category": c.category,
        "manufacturer": getattr(c, 'manufacturer', 'Unknown'),
        "url": f"/v1/components/{c.id}",
        "score": 1.0
    } for c in components]


async def _search_companies_advanced(session: Session, query: str, filters: Dict, limit: int, offset: int) -> List[Dict]:
    """Advanced company search with filters"""
    companies_repo = CompaniesRepository(session)
    
    # Use the correct method signature - CompaniesRepository.list() has different parameters
    type_filter = filters.get("type") if filters else None
    companies = companies_repo.list(search=query, type_=type_filter)[:limit]
    
    # Apply additional filters manually
    if filters and "country" in filters:
        companies = [c for c in companies if getattr(c, 'country', '') == filters["country"]]
    
    return [{
        "id": c.id,
        "type": "company",
        "title": c.name,
        "description": f"{c.type} company",
        "country": getattr(c, 'country', 'Unknown'),
        "website": c.website,
        "url": f"/v1/companies/{c.id}",
        "score": 0.9
    } for c in companies]


async def _search_mongo_advanced(mongo, query: str, filters: Dict, entity_types: List[str], limit: int) -> List[Dict]:
    """Advanced MongoDB search with filters"""
    results = []
    
    try:
        # Build MongoDB query with filters
        base_query = {"$or": [
            {"content": {"$regex": query, "$options": "i"}},
            {"subject": {"$regex": query, "$options": "i"}},
            {"body": {"$regex": query, "$options": "i"}}
        ]}
        
        # Add filters
        if filters:
            if "date_range" in filters:
                date_filter = filters["date_range"]
                if "start" in date_filter:
                    base_query["ingestedAt"] = {"$gte": date_filter["start"]}
                if "end" in date_filter:
                    base_query.setdefault("ingestedAt", {})["$lte"] = date_filter["end"]
        
        # Search each requested collection
        for entity_type in entity_types:
            if entity_type in ["emails", "teams", "web_intelligence"]:
                collection_name = f"ingested_{entity_type}" if entity_type == "emails" else entity_type
                if entity_type == "teams":
                    collection_name = "teams_messages"
                
                coll = mongo.get_collection(collection_name)
                cursor = coll.find(base_query).limit(limit // len(entity_types))
                
                async for doc in cursor:
                    results.append({
                        "id": str(doc.get("_id")),
                        "type": entity_type,
                        "title": doc.get("subject") or doc.get("url") or f"{entity_type.title()} Document",
                        "description": (doc.get("content") or doc.get("body") or "")[:200],
                        "timestamp": doc.get("ingestedAt"),
                        "score": 0.7
                    })
        
    except Exception as e:
        logger.error(f"Advanced MongoDB search error: {e}")
    
    return results


def _generate_search_facets(results: List[Dict]) -> List[SearchFacet]:
    """Generate search facets from results"""
    facets = []
    
    # Type facet
    type_counts = {}
    for result in results:
        result_type = result.get("type", "unknown")
        type_counts[result_type] = type_counts.get(result_type, 0) + 1
    
    facets.append(SearchFacet(
        name="type",
        values=[{"value": k, "count": v} for k, v in type_counts.items()]
    ))
    
    # Category facet (for components)
    category_counts = {}
    for result in results:
        if result.get("type") == "component" and result.get("category"):
            category = result["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
    
    if category_counts:
        facets.append(SearchFacet(
            name="category",
            values=[{"value": k, "count": v} for k, v in category_counts.items()]
        ))
    
    return facets


async def _generate_ai_insights(query: str, results: List[Dict]) -> Dict[str, Any]:
    """Generate AI insights for search results"""
    try:
        from ..ai.capabilities import classify_intent, extract_components
        
        # Classify search intent
        intent_result = await classify_intent(query)
        
        # Extract components from query
        components_result = await extract_components(query)
        
        # Analyze result patterns
        result_types = [r.get("type") for r in results]
        most_common_type = max(set(result_types), key=result_types.count) if result_types else "none"
        
        insights = {
            "search_intent": intent_result.data if intent_result.success else {},
            "extracted_components": components_result.data if components_result.success else [],
            "result_analysis": {
                "total_results": len(results),
                "most_common_type": most_common_type,
                "type_distribution": {t: result_types.count(t) for t in set(result_types)}
            },
            "suggestions": _generate_search_suggestions(query, results)
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"AI insights error: {e}")
        return {"error": "AI insights unavailable"}


def _generate_search_suggestions(query: str, results: List[Dict]) -> List[str]:
    """Generate search suggestions based on query and results"""
    suggestions = []
    
    # If few results, suggest broader search
    if len(results) < 5:
        suggestions.append(f"Try searching for broader terms related to '{query}'")
    
    # If many component results, suggest filtering
    component_count = sum(1 for r in results if r.get("type") == "component")
    if component_count > 10:
        suggestions.append("Consider filtering by component category or manufacturer")
    
    # Suggest related searches based on result content
    if any("STM32" in str(r.get("title", "")) for r in results):
        suggestions.append("Related: STM32 microcontrollers, ARM Cortex-M4")
    
    return suggestions

