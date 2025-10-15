from fastapi import APIRouter, Query, HTTPException, Depends
from sqlalchemy.orm import Session
from ..models.intelligence import MarketTrendsResponse, SupplierAnalysisResponse, ScenarioRequest, ScenarioResponse
from ..ai.market_intelligence import market_intelligence, ComponentData
from ..ai.web_intelligence import web_intelligence, get_intelligence_dashboard, search_web_intelligence
from ..db.session import get_session
from ..db.models import Component as ComponentORM
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/market-trends", response_model=MarketTrendsResponse)
async def market_trends(
    component: str | None = None, 
    timeframe: str = Query(default="30d"), 
    region: str | None = None,
    session: Session = Depends(get_session)
):
    """
    Get AI-powered market trend analysis for components
    """
    try:
        # If component ID provided, fetch from database
        component_data = None
        if component:
            component_orm = session.get(ComponentORM, component)
            if component_orm:
                component_data = ComponentData(
                    id=component_orm.id,
                    part_number=component_orm.manufacturer_part_number,
                    category=component_orm.category or "Unknown",
                    description=component_orm.description,
                    manufacturer=component_orm.manufacturer_id
                )
        
        # Fallback to example component if not found
        if not component_data:
            component_data = ComponentData(
                id=component or "example-component",
                part_number="STM32F429ZIT6",
                category="Microcontrollers",
                description="ARM Cortex-M4 MCU"
            )
        
        # Get AI analysis
        analysis = await market_intelligence.analyze_market_trends(
            component_data, timeframe, region or "Global"
        )
        
        return MarketTrendsResponse(**analysis)
        
    except Exception as e:
        logger.error(f"Error in market trends analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to analyze market trends. Please try again later."
        )


@router.get("/supplier-analysis", response_model=SupplierAnalysisResponse)
async def supplier_analysis(
    supplierId: str | None = None, 
    component: str | None = None, 
    riskLevel: str | None = None,
    session: Session = Depends(get_session)
):
    """
    Get AI-powered supplier performance analysis
    """
    try:
        if not supplierId:
            raise HTTPException(status_code=400, detail="supplierId parameter is required")
        
        # Get component filters if specified
        component_filters = [component] if component else None
        
        # Run AI analysis
        analysis = await market_intelligence.analyze_supplier_performance(
            supplierId, component_filters, time_period="90d"
        )
        
        return SupplierAnalysisResponse(**analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in supplier analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to analyze supplier performance. Please try again later."
        )


@router.post("/scenario-analysis", response_model=ScenarioResponse)
async def scenario_analysis(body: ScenarioRequest):
    """
    Run AI-powered scenario analysis for geopolitical and market disruptions
    """
    try:
        # Extract scenario parameters
        scenario_type = body.scenario.type if body.scenario else "geopolitical"
        scenario_params = {
            "event": body.scenario.event if body.scenario else "Unknown event",
            "probability": body.scenario.probability if body.scenario else 0.5,
            "timeframe": body.scenario.timeframe if body.scenario else "6_months"
        }
        
        # Get affected components
        affected_components = body.components or ["example-component"]
        
        # Run AI scenario analysis
        analysis = await market_intelligence.run_scenario_analysis(
            scenario_type, scenario_params, affected_components
        )
        
        return ScenarioResponse(**analysis)
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to run scenario analysis. Please try again later."
        )


# Web Intelligence Endpoints

@router.get("/web-intelligence/dashboard")
async def get_web_intelligence_dashboard():
    """Get web intelligence dashboard with latest market insights"""
    try:
        dashboard_data = await get_intelligence_dashboard()
        return dashboard_data
    except Exception as e:
        logger.error(f"Error getting web intelligence dashboard: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get web intelligence dashboard"
        )


@router.get("/web-intelligence/search")
async def search_intelligence_items(
    query: str = Query(..., description="Search query for intelligence items"),
    days_back: int = Query(7, ge=1, le=90, description="Number of days to search back"),
    item_type: str = Query(None, regex="^(news|supplier|market)$", description="Filter by item type")
):
    """Search web intelligence items"""
    try:
        results = await search_web_intelligence(query, days_back)
        
        # Filter by type if specified
        if item_type:
            results = [item for item in results if item.get('item_type') == item_type]
        
        return {
            "query": query,
            "days_back": days_back,
            "total_results": len(results),
            "items": results[:20]  # Limit to 20 results
        }
    except Exception as e:
        logger.error(f"Error searching web intelligence: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to search web intelligence"
        )


@router.get("/web-intelligence/market-sentiment")
async def get_market_sentiment():
    """Get current market sentiment analysis"""
    try:
        sentiment = await web_intelligence.market_collector.analyze_market_sentiment()
        return sentiment
    except Exception as e:
        logger.error(f"Error getting market sentiment: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze market sentiment"
        )


@router.get("/web-intelligence/geopolitical-alerts")
async def get_geopolitical_alerts(hours_back: int = Query(48, ge=1, le=168)):
    """Get geopolitical risk alerts from web intelligence"""
    try:
        from datetime import datetime, timezone, timedelta
        from ..db.mongo import get_mongo_db
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        mongo = get_mongo_db()
        if not mongo:
            return {"alerts": [], "total": 0}
        
        collection = mongo.get_collection("web_intelligence")
        
        # Query for high-risk geopolitical items
        cursor = collection.find({
            'discovered_at': {'$gte': cutoff_time.isoformat()},
            'geopolitical_impact.risk_level': {'$in': ['high', 'critical']}
        }).sort('relevance_score', -1).limit(20)
        
        alerts = await cursor.to_list(length=20)
        
        return {
            "period_hours": hours_back,
            "total": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting geopolitical alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get geopolitical alerts"
        )


@router.get("/web-intelligence/supply-chain-alerts")
async def get_supply_chain_alerts(hours_back: int = Query(24, ge=1, le=168)):
    """Get supply chain disruption alerts"""
    try:
        from datetime import datetime, timezone, timedelta
        from ..db.mongo import get_mongo_db
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        mongo = get_mongo_db()
        if not mongo:
            return {"alerts": [], "total": 0}
        
        collection = mongo.get_collection("web_intelligence")
        
        # Query for supply chain issues
        cursor = collection.find({
            'discovered_at': {'$gte': cutoff_time.isoformat()},
            '$or': [
                {'market_impact.supply_impact': 'shortage'},
                {'keywords': {'$in': ['shortage', 'supply chain', 'disruption', 'allocation']}},
                {'relevance_score': {'$gte': 0.8}}
            ]
        }).sort('relevance_score', -1).limit(15)
        
        alerts = await cursor.to_list(length=15)
        
        return {
            "period_hours": hours_back,
            "total": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting supply chain alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get supply chain alerts"
        )

