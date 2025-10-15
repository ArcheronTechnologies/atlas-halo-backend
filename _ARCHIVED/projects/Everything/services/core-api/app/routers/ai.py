"""
AI/ML Analysis Router

Endpoints for AI/ML capabilities including NER, classification, forecasting, and analysis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..ai.capabilities import (
    extract_components,
    classify_intent, 
    get_recommendations,
    analyze_scenario,
    process_email,
    process_rfq,
    orchestrator,
    capability_registry
)
from ..core.auth import require_api_key_or_bearer

router = APIRouter()

# Request/Response Models
class ComponentExtractionRequest(BaseModel):
    text: str
    context: Optional[str] = None
    confidence_threshold: Optional[float] = 0.8

class ComponentExtractionResponse(BaseModel):
    success: bool
    components: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    processing_time_ms: float

class SentimentAnalysisRequest(BaseModel):
    text: str
    context: Optional[str] = None

class SentimentAnalysisResponse(BaseModel):
    success: bool
    sentiment: str
    confidence: float
    emotions: Dict[str, float]
    processing_time_ms: float

class PricePredictionRequest(BaseModel):
    component_id: str
    historical_data: Optional[List[Dict[str, Any]]] = None
    forecast_horizon_days: Optional[int] = 90

class PricePredictionResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    confidence_interval: Dict[str, float]
    model_metrics: Dict[str, float]
    processing_time_ms: float

class SupplyRiskRequest(BaseModel):
    component_ids: List[str]
    suppliers: Optional[List[str]] = None
    risk_factors: Optional[List[str]] = None

class SupplyRiskResponse(BaseModel):
    success: bool
    risk_assessment: Dict[str, Any]
    risk_score: float
    mitigation_strategies: List[str]
    processing_time_ms: float

class DemandForecastRequest(BaseModel):
    component_id: str
    historical_demand: Optional[List[Dict[str, Any]]] = None
    forecast_horizon_days: Optional[int] = 90

class DemandForecastResponse(BaseModel):
    success: bool
    forecast: List[Dict[str, Any]]
    seasonality: Dict[str, Any]
    confidence_intervals: Dict[str, float]
    processing_time_ms: float

class BOMOptimizationRequest(BaseModel):
    bom_components: List[Dict[str, Any]]
    optimization_criteria: List[str] = ["cost", "availability", "risk"]
    constraints: Optional[Dict[str, Any]] = None

class BOMOptimizationResponse(BaseModel):
    success: bool
    optimized_bom: List[Dict[str, Any]]
    cost_savings: float
    risk_reduction: float
    alternatives: List[Dict[str, Any]]
    processing_time_ms: float

class SupplierScoringRequest(BaseModel):
    supplier_id: str
    evaluation_criteria: Optional[List[str]] = None
    historical_performance: Optional[Dict[str, Any]] = None

class SupplierScoringResponse(BaseModel):
    success: bool
    overall_score: float
    category_scores: Dict[str, float]
    benchmarks: Dict[str, float]
    recommendations: List[str]
    processing_time_ms: float

class AnomalyDetectionRequest(BaseModel):
    data_type: str
    data_points: List[Dict[str, Any]]
    sensitivity: Optional[float] = 0.05

class AnomalyDetectionResponse(BaseModel):
    success: bool
    anomalies: List[Dict[str, Any]]
    anomaly_scores: List[float]
    baseline_metrics: Dict[str, float]
    processing_time_ms: float

class MarketIntelligenceRequest(BaseModel):
    components: List[str]
    intelligence_types: List[str] = ["price", "availability", "trends"]
    time_horizon: Optional[str] = "3months"

class MarketIntelligenceResponse(BaseModel):
    success: bool
    intelligence_summary: Dict[str, Any]
    market_trends: List[Dict[str, Any]]
    competitive_analysis: Dict[str, Any]
    processing_time_ms: float

class GeopoliticalRiskRequest(BaseModel):
    suppliers: List[str]
    components: List[str]
    risk_scenarios: Optional[List[str]] = None

class GeopoliticalRiskResponse(BaseModel):
    success: bool
    risk_assessment: Dict[str, Any]
    country_risks: Dict[str, float]
    scenario_impacts: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    processing_time_ms: float


@router.post("/extract-components", response_model=ComponentExtractionResponse)
async def extract_components_endpoint(
    request: ComponentExtractionRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Extract component part numbers and specifications from text using NER models"""
    try:
        import time
        start_time = time.time()
        
        result = await extract_components(
            text=request.text,
            context=request.context,
            confidence_threshold=request.confidence_threshold
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.success:
            return ComponentExtractionResponse(
                success=True,
                components=result.data.get("entities", []),
                confidence_scores=result.data.get("confidence_scores", {}),
                processing_time_ms=processing_time
            )
        else:
            return ComponentExtractionResponse(
                success=False,
                components=[],
                confidence_scores={},
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Component extraction failed: {str(e)}")


@router.post("/analyze-sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment_endpoint(
    request: SentimentAnalysisRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Analyze sentiment of supplier communications and market signals"""
    try:
        import time
        start_time = time.time()
        
        # Use intent classification as a proxy for sentiment analysis
        result = await classify_intent(
            text=request.text,
            context=request.context or ""
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.success:
            # Map intent to sentiment (simplified)
            intent = result.data.get("primary_intent", "neutral")
            sentiment_map = {
                "complaint": "negative",
                "urgent": "negative", 
                "request": "neutral",
                "inquiry": "neutral",
                "positive": "positive",
                "approval": "positive"
            }
            sentiment = sentiment_map.get(intent, "neutral")
            
            return SentimentAnalysisResponse(
                success=True,
                sentiment=sentiment,
                confidence=result.data.get("confidence", 0.5),
                emotions=result.data.get("emotions", {}),
                processing_time_ms=processing_time
            )
        else:
            return SentimentAnalysisResponse(
                success=False,
                sentiment="unknown",
                confidence=0.0,
                emotions={},
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@router.post("/predict-prices", response_model=PricePredictionResponse)
async def predict_prices_endpoint(
    request: PricePredictionRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Predict component pricing trends using ML models"""
    try:
        import time
        start_time = time.time()
        
        result = await orchestrator.execute_capability(
            "price_forecast",
            {
                "component_id": request.component_id,
                "historical_data": request.historical_data,
                "forecast_horizon_days": request.forecast_horizon_days
            }
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.success:
            return PricePredictionResponse(
                success=True,
                predictions=result.data.get("predictions", []),
                confidence_interval=result.data.get("confidence_interval", {}),
                model_metrics=result.data.get("model_metrics", {}),
                processing_time_ms=processing_time
            )
        else:
            return PricePredictionResponse(
                success=False,
                predictions=[],
                confidence_interval={},
                model_metrics={},
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Price prediction failed: {str(e)}")


@router.post("/analyze-supply-risk", response_model=SupplyRiskResponse)
async def analyze_supply_risk_endpoint(
    request: SupplyRiskRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Analyze supply chain risks for components and suppliers"""
    try:
        import time
        start_time = time.time()
        
        result = await analyze_scenario(
            scenario_type="supply_risk",
            parameters={
                "suppliers": request.suppliers,
                "risk_factors": request.risk_factors
            },
            affected_components=request.component_ids
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.success:
            return SupplyRiskResponse(
                success=True,
                risk_assessment=result.data.get("assessment", {}),
                risk_score=result.data.get("risk_score", 0.0),
                mitigation_strategies=result.data.get("mitigation_strategies", []),
                processing_time_ms=processing_time
            )
        else:
            return SupplyRiskResponse(
                success=False,
                risk_assessment={},
                risk_score=0.0,
                mitigation_strategies=[],
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supply risk analysis failed: {str(e)}")


@router.post("/forecast-demand", response_model=DemandForecastResponse)
async def forecast_demand_endpoint(
    request: DemandForecastRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Forecast component demand using time series models"""
    try:
        import time
        start_time = time.time()
        
        result = await orchestrator.execute_capability(
            "demand_forecast",
            {
                "component_id": request.component_id,
                "historical_demand": request.historical_demand,
                "forecast_horizon_days": request.forecast_horizon_days
            }
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.success:
            return DemandForecastResponse(
                success=True,
                forecast=result.data.get("forecast", []),
                seasonality=result.data.get("seasonality", {}),
                confidence_intervals=result.data.get("confidence_intervals", {}),
                processing_time_ms=processing_time
            )
        else:
            return DemandForecastResponse(
                success=False,
                forecast=[],
                seasonality={},
                confidence_intervals={},
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand forecasting failed: {str(e)}")


@router.post("/optimize-bom", response_model=BOMOptimizationResponse)
async def optimize_bom_endpoint(
    request: BOMOptimizationRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Optimize Bill of Materials for cost, availability, and risk"""
    try:
        import time
        start_time = time.time()
        
        # Use component recommender for BOM optimization
        optimization_results = []
        for component in request.bom_components:
            result = await get_recommendations(
                component=component.get("part_number", ""),
                optimization_criteria=request.optimization_criteria,
                constraints=request.constraints
            )
            if result.success:
                optimization_results.append(result.data)
        
        processing_time = (time.time() - start_time) * 1000
        
        return BOMOptimizationResponse(
            success=True,
            optimized_bom=optimization_results,
            cost_savings=sum(r.get("cost_savings", 0) for r in optimization_results),
            risk_reduction=sum(r.get("risk_reduction", 0) for r in optimization_results) / len(optimization_results) if optimization_results else 0,
            alternatives=[r.get("alternatives", []) for r in optimization_results],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BOM optimization failed: {str(e)}")


@router.post("/score-suppliers", response_model=SupplierScoringResponse)
async def score_suppliers_endpoint(
    request: SupplierScoringRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Score and rank suppliers using advanced analytics"""
    try:
        import time
        start_time = time.time()
        
        result = await orchestrator.execute_capability(
            "supplier_analysis",
            {
                "supplier_id": request.supplier_id,
                "evaluation_criteria": request.evaluation_criteria,
                "historical_performance": request.historical_performance
            }
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.success:
            return SupplierScoringResponse(
                success=True,
                overall_score=result.data.get("overall_score", 0.0),
                category_scores=result.data.get("category_scores", {}),
                benchmarks=result.data.get("benchmarks", {}),
                recommendations=result.data.get("recommendations", []),
                processing_time_ms=processing_time
            )
        else:
            return SupplierScoringResponse(
                success=False,
                overall_score=0.0,
                category_scores={},
                benchmarks={},
                recommendations=[],
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supplier scoring failed: {str(e)}")


@router.post("/detect-anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies_endpoint(
    request: AnomalyDetectionRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Detect anomalies in pricing, demand, or supply patterns"""
    try:
        import time
        start_time = time.time()
        
        result = await analyze_scenario(
            scenario_type="anomaly_detection",
            parameters={
                "data_type": request.data_type,
                "sensitivity": request.sensitivity
            },
            affected_components=[dp.get("component_id", "") for dp in request.data_points]
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.success:
            return AnomalyDetectionResponse(
                success=True,
                anomalies=result.data.get("anomalies", []),
                anomaly_scores=result.data.get("anomaly_scores", []),
                baseline_metrics=result.data.get("baseline_metrics", {}),
                processing_time_ms=processing_time
            )
        else:
            return AnomalyDetectionResponse(
                success=False,
                anomalies=[],
                anomaly_scores=[],
                baseline_metrics={},
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@router.post("/synthesize-intelligence", response_model=MarketIntelligenceResponse)
async def synthesize_market_intelligence_endpoint(
    request: MarketIntelligenceRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Synthesize market intelligence from multiple data sources"""
    try:
        import time
        start_time = time.time()
        
        # Process each component through the RFQ pipeline for comprehensive analysis
        intelligence_data = []
        for component in request.components:
            result = await process_rfq(
                rfq_description=f"Market analysis for {component}",
                component_id=component,
                intelligence_types=request.intelligence_types,
                time_horizon=request.time_horizon
            )
            intelligence_data.append(result)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Aggregate intelligence
        aggregated = {
            "components_analyzed": len(request.components),
            "intelligence_types": request.intelligence_types,
            "time_horizon": request.time_horizon,
            "summary": "Market intelligence synthesis completed"
        }
        
        return MarketIntelligenceResponse(
            success=True,
            intelligence_summary=aggregated,
            market_trends=[d.get("market_trends", {}) for d in intelligence_data],
            competitive_analysis={"analysis": "Aggregated competitive intelligence"},
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market intelligence synthesis failed: {str(e)}")


@router.post("/assess-geopolitical-risk", response_model=GeopoliticalRiskResponse)
async def assess_geopolitical_risk_endpoint(
    request: GeopoliticalRiskRequest,
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Assess geopolitical risks for suppliers and supply chains"""
    try:
        import time
        start_time = time.time()
        
        result = await analyze_scenario(
            scenario_type="geopolitical_risk",
            parameters={
                "suppliers": request.suppliers,
                "risk_scenarios": request.risk_scenarios
            },
            affected_components=request.components
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.success:
            return GeopoliticalRiskResponse(
                success=True,
                risk_assessment=result.data.get("risk_assessment", {}),
                country_risks=result.data.get("country_risks", {}),
                scenario_impacts=result.data.get("scenario_impacts", []),
                mitigation_strategies=result.data.get("mitigation_strategies", []),
                processing_time_ms=processing_time
            )
        else:
            return GeopoliticalRiskResponse(
                success=False,
                risk_assessment={},
                country_risks={},
                scenario_impacts=[],
                mitigation_strategies=[],
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geopolitical risk assessment failed: {str(e)}")


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_ai_capabilities(
    current_user: Dict = Depends(require_api_key_or_bearer)
):
    """Get status and information about available AI capabilities"""
    try:
        capabilities_info = {}
        
        # Get all registered capabilities
        for name, registration in capability_registry._capabilities.items():
            capability = registration.capability
            capabilities_info[name] = {
                "name": capability.config.name,
                "version": capability.config.version,
                "status": capability.status.value,
                "description": capability.config.description,
                "tags": list(registration.tags) if registration.tags else [],
                "dependencies": list(registration.dependencies),
                "priority": registration.priority
            }
        
        # Get pipeline information
        pipelines_info = {}
        for name, pipeline in orchestrator._pipelines.items():
            pipelines_info[name] = {
                "name": pipeline.config.name,
                "version": pipeline.config.version,
                "steps": len(pipeline.config.steps),
                "parallel_execution": pipeline.config.parallel_execution,
                "timeout_seconds": pipeline.config.timeout_seconds
            }
        
        return {
            "capabilities": capabilities_info,
            "pipelines": pipelines_info,
            "total_capabilities": len(capabilities_info),
            "healthy_capabilities": sum(1 for c in capabilities_info.values() if c["status"] == "healthy"),
            "framework_version": "2.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities info: {str(e)}")