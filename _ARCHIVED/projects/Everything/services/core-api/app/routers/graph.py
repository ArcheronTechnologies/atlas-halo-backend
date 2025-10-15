"""
Graph Database Router

Advanced graph database operations for supply chain intelligence including
supplier relationships, component alternatives, market analysis, and network visualization.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from ..db.session import get_session
from ..db.models import Component, Company
from ..core.auth import require_scopes
from ..graph import neo4j_client, GraphQueries
from ..graph.sync import GraphSynchronizer

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize graph components
graph_queries = GraphQueries(neo4j_client)
graph_synchronizer = GraphSynchronizer(neo4j_client)


@router.post("/sync", response_model=dict, dependencies=[Depends(require_scopes(["write:graph"]))])
async def sync_graph(
    background_tasks: BackgroundTasks,
    full_sync: bool = Query(False, description="Perform full sync instead of incremental"),
    batch_size: int = Query(1000, description="Batch size for sync operations")
):
    """Synchronize data from SQL database to Neo4j graph database"""
    if not neo4j_client.is_healthy():
        return {"synced": False, "message": "Neo4j not configured or available"}
    
    try:
        if full_sync:
            # Run full sync in background
            background_tasks.add_task(graph_synchronizer.full_sync, batch_size)
            return {
                "synced": True,
                "message": "Full synchronization started in background",
                "type": "full"
            }
        else:
            # Run incremental sync immediately
            since = datetime.utcnow() - timedelta(hours=1)  # Last hour
            result = await graph_synchronizer.incremental_sync(since, batch_size)
            return {
                "synced": True,
                "message": "Incremental synchronization completed",
                "type": "incremental",
                "metrics": result
            }
    except Exception as e:
        logger.error(f"Graph sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/sync/status", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def get_sync_status():
    """Get synchronization status and metrics"""
    if not neo4j_client.is_healthy():
        return {"available": False, "message": "Neo4j not configured"}
    
    health = neo4j_client.get_health()
    metrics = neo4j_client.get_metrics()
    sync_metrics = graph_synchronizer.get_metrics()
    
    return {
        "available": True,
        "healthy": health.is_healthy,
        "last_check": health.last_check.isoformat() if health.last_check else None,
        "response_time": health.response_time,
        "server_info": health.server_info,
        "query_metrics": metrics,
        "sync_metrics": sync_metrics
    }


@router.post("/validate", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def validate_graph():
    """Validate graph data consistency"""
    if not neo4j_client.is_healthy():
        return {"validated": False, "message": "Neo4j not configured"}
    
    try:
        validation_results = await graph_synchronizer.validate_sync()
        return {
            "validated": True,
            "results": validation_results
        }
    except Exception as e:
        logger.error(f"Graph validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# Component Analysis Endpoints

@router.get("/components/{component_id}/neighbors", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
def component_neighbors(component_id: str, session: Session = Depends(get_session)):
    """Get component neighbors (legacy endpoint for backwards compatibility)"""
    comp = session.get(Component, component_id)
    if not comp:
        raise HTTPException(404, detail="Component not found")
    neighbors = []
    if comp.manufacturer_id:
        mfg = session.get(Company, comp.manufacturer_id)
        if mfg:
            neighbors.append({"type": "MANUFACTURER", "company": {"id": mfg.id, "name": mfg.name}})
    return {
        "component": {
            "id": comp.id,
            "manufacturerPartNumber": comp.manufacturer_part_number,
        },
        "neighbors": neighbors,
    }


@router.get("/components/{component_id}/suppliers", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def get_component_suppliers(
    component_id: str,
    include_inactive: bool = Query(False, description="Include inactive suppliers"),
    max_lead_time_days: Optional[int] = Query(None, description="Maximum lead time in days"),
    max_price: Optional[float] = Query(None, description="Maximum price filter"),
    min_availability_score: Optional[float] = Query(None, description="Minimum availability score")
):
    """Find suppliers for a specific component with advanced filtering"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.find_suppliers_for_component(
            component_id=component_id,
            include_inactive=include_inactive,
            max_lead_time_days=max_lead_time_days,
            max_price=max_price,
            min_availability_score=min_availability_score
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "component_id": component_id,
            "suppliers": result.data or [],
            "count": len(result.data or []),
            "execution_time": result.execution_time
        }
    except Exception as e:
        logger.error(f"Failed to get suppliers for component {component_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/components/{component_id}/alternatives", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def get_component_alternatives(
    component_id: str,
    min_confidence: float = Query(0.7, description="Minimum confidence score"),
    compatibility_levels: Optional[List[str]] = Query(None, description="Compatibility levels to include"),
    verified_only: bool = Query(False, description="Show only verified alternatives")
):
    """Find alternative components with compatibility analysis"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.find_component_alternatives(
            component_id=component_id,
            min_confidence=min_confidence,
            compatibility_levels=compatibility_levels,
            verified_only=verified_only
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "component_id": component_id,
            "alternatives": result.data or [],
            "count": len(result.data or []),
            "execution_time": result.execution_time
        }
    except Exception as e:
        logger.error(f"Failed to get alternatives for component {component_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/components/{component_id}/suggested-alternatives", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def get_suggested_alternatives(component_id: str):
    """Get AI-powered component alternative suggestions"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.suggest_component_alternatives(component_id)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "component_id": component_id,
            "suggested_alternatives": result.data or [],
            "count": len(result.data or []),
            "execution_time": result.execution_time
        }
    except Exception as e:
        logger.error(f"Failed to get suggested alternatives for component {component_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Supplier Analysis Endpoints

@router.get("/suppliers/{supplier_id}/alternatives", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def find_alternative_suppliers(
    supplier_id: str,
    component_categories: Optional[List[str]] = Query(None, description="Component categories to include"),
    exclude_regions: Optional[List[str]] = Query(None, description="Regions to exclude"),
    min_risk_score: Optional[float] = Query(None, description="Minimum risk score")
):
    """Find alternative suppliers for components supplied by a primary supplier"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.find_alternative_suppliers(
            primary_supplier_id=supplier_id,
            component_categories=component_categories,
            exclude_regions=exclude_regions,
            min_risk_score=min_risk_score
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "primary_supplier_id": supplier_id,
            "alternative_suppliers": result.data or [],
            "count": len(result.data or []),
            "execution_time": result.execution_time
        }
    except Exception as e:
        logger.error(f"Failed to find alternative suppliers for {supplier_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suppliers/{supplier_id}/network", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def get_supplier_network(
    supplier_id: str,
    network_depth: int = Query(2, description="Network traversal depth")
):
    """Map supplier's network including partners, competitors, and shared customers"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.find_supplier_network(
            supplier_id=supplier_id,
            network_depth=network_depth
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "supplier_id": supplier_id,
            "network": result.data[0] if result.data else {},
            "execution_time": result.execution_time
        }
    except Exception as e:
        logger.error(f"Failed to get supplier network for {supplier_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Customer Analysis Endpoints

@router.get("/customers/{customer_id}/dependency-analysis", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def analyze_supplier_dependencies(
    customer_id: str,
    depth: int = Query(2, description="Analysis depth")
):
    """Analyze supplier dependency risks for a customer"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.analyze_supplier_dependencies(
            customer_id=customer_id,
            depth=depth
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "customer_id": customer_id,
            "dependencies": result.data or [],
            "count": len(result.data or []),
            "execution_time": result.execution_time
        }
    except Exception as e:
        logger.error(f"Failed to analyze dependencies for customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Market Analysis Endpoints

@router.get("/market/components/{category}", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def analyze_component_market(
    category: str,
    time_window_days: int = Query(90, description="Time window for analysis in days"),
    include_price_trends: bool = Query(True, description="Include price trend analysis")
):
    """Analyze market conditions for a component category"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.analyze_component_market(
            component_category=category,
            time_window_days=time_window_days,
            include_price_trends=include_price_trends
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "category": category,
            "market_analysis": result.data or [],
            "count": len(result.data or []),
            "execution_time": result.execution_time
        }
    except Exception as e:
        logger.error(f"Failed to analyze market for category {category}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/supply-risks", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def identify_supply_risks(
    regions: Optional[List[str]] = Query(None, description="Regions to analyze"),
    component_categories: Optional[List[str]] = Query(None, description="Component categories to analyze"),
    risk_threshold: float = Query(0.7, description="Risk score threshold")
):
    """Identify supply chain risks by region and category"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.identify_supply_risks(
            regions=regions,
            component_categories=component_categories,
            risk_threshold=risk_threshold
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "supply_risks": result.data or [],
            "count": len(result.data or []),
            "execution_time": result.execution_time,
            "filters": {
                "regions": regions,
                "categories": component_categories,
                "risk_threshold": risk_threshold
            }
        }
    except Exception as e:
        logger.error(f"Failed to identify supply risks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Partnership Analysis Endpoints

@router.get("/companies/{company_id}/partnership-recommendations", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def recommend_partnerships(
    company_id: str,
    partnership_types: Optional[List[str]] = Query(None, description="Partnership types to consider"),
    min_market_overlap: float = Query(0.3, description="Minimum market overlap threshold"),
    exclude_existing: bool = Query(True, description="Exclude existing partnerships")
):
    """Recommend potential strategic partnerships based on market analysis"""
    if not neo4j_client.is_healthy():
        raise HTTPException(status_code=503, detail="Graph database not available")
    
    try:
        result = await graph_queries.recommend_strategic_partnerships(
            company_id=company_id,
            partnership_types=partnership_types,
            min_market_overlap=min_market_overlap,
            exclude_existing=exclude_existing
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return {
            "company_id": company_id,
            "recommendations": result.data or [],
            "count": len(result.data or []),
            "execution_time": result.execution_time
        }
    except Exception as e:
        logger.error(f"Failed to generate partnership recommendations for {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Missing endpoints from Phase 2 requirements

@router.get("/relationships", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def get_graph_relationships(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type"),
    limit: int = Query(100, description="Maximum number of relationships to return"),
    depth: int = Query(1, description="Relationship traversal depth")
):
    """Get graph relationships with filtering and traversal options"""
    if not neo4j_client.is_healthy():
        return {
            "available": False,
            "message": "Graph database not available",
            "relationships": [],
            "summary": {
                "total_nodes": 0,
                "total_relationships": 0,
                "node_types": {},
                "relationship_types": {}
            }
        }
    
    try:
        # Get basic graph statistics and relationships
        result = await graph_queries.get_graph_overview(
            entity_type=entity_type,
            relationship_type=relationship_type,
            limit=limit,
            depth=depth
        )
        
        if not result.success:
            return {
                "available": True,
                "relationships": [],
                "message": result.error or "No relationships found",
                "summary": {"total_nodes": 0, "total_relationships": 0}
            }
        
        return {
            "available": True,
            "relationships": result.data.get("relationships", []),
            "summary": result.data.get("summary", {}),
            "execution_time": result.execution_time,
            "filters": {
                "entity_type": entity_type,
                "relationship_type": relationship_type,
                "limit": limit,
                "depth": depth
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph relationships: {e}")
        return {
            "available": True,
            "relationships": [],
            "message": f"Error: {str(e)}",
            "summary": {"total_nodes": 0, "total_relationships": 0}
        }


@router.get("/suppliers", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def get_suppliers_graph(
    component_category: Optional[str] = Query(None, description="Filter by component category"),
    region: Optional[str] = Query(None, description="Filter by supplier region"),
    min_reliability_score: Optional[float] = Query(None, description="Minimum reliability score"),
    include_metrics: bool = Query(True, description="Include supplier performance metrics")
):
    """Get suppliers network graph with performance metrics"""
    if not neo4j_client.is_healthy():
        return {
            "available": False,
            "message": "Graph database not available",
            "suppliers": [],
            "metrics": {}
        }
    
    try:
        result = await graph_queries.get_suppliers_network(
            component_category=component_category,
            region=region,
            min_reliability_score=min_reliability_score,
            include_metrics=include_metrics
        )
        
        if not result.success:
            return {
                "available": True,
                "suppliers": [],
                "message": result.error or "No suppliers found",
                "metrics": {}
            }
        
        return {
            "available": True,
            "suppliers": result.data.get("suppliers", []),
            "metrics": result.data.get("metrics", {}),
            "network_analysis": result.data.get("network_analysis", {}),
            "execution_time": result.execution_time,
            "filters": {
                "component_category": component_category,
                "region": region,
                "min_reliability_score": min_reliability_score
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get suppliers graph: {e}")
        return {
            "available": True,
            "suppliers": [],
            "message": f"Error: {str(e)}",
            "metrics": {}
        }


@router.get("/network-visualization", response_model=dict, dependencies=[Depends(require_scopes(["read:graph"]))])
async def get_network_visualization(
    center_entity_id: Optional[str] = Query(None, description="Center entity for visualization"),
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types to include"),
    max_nodes: int = Query(100, description="Maximum number of nodes"),
    max_depth: int = Query(2, description="Maximum traversal depth")
):
    """Get network visualization data for graph rendering"""
    if not neo4j_client.is_healthy():
        return {
            "available": False,
            "message": "Graph database not available",
            "nodes": [],
            "edges": []
        }
    
    try:
        # Parse entity types
        parsed_entity_types = None
        if entity_types:
            parsed_entity_types = [t.strip() for t in entity_types.split(",")]
        
        result = await graph_queries.get_network_visualization(
            center_entity_id=center_entity_id,
            entity_types=parsed_entity_types,
            max_nodes=max_nodes,
            max_depth=max_depth
        )
        
        if not result.success:
            return {
                "available": True,
                "nodes": [],
                "edges": [],
                "message": result.error or "No network data found"
            }
        
        return {
            "available": True,
            "nodes": result.data.get("nodes", []),
            "edges": result.data.get("edges", []),
            "layout_hints": result.data.get("layout_hints", {}),
            "statistics": result.data.get("statistics", {}),
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Failed to get network visualization: {e}")
        return {
            "available": True,
            "nodes": [],
            "edges": [],
            "message": f"Error: {str(e)}"
        }

