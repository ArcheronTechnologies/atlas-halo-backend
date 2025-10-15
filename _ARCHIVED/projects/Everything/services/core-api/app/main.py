from fastapi import FastAPI, WebSocket, Depends, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .routers.auth import router as auth_router
from .routers.components import router as components_router
from .routers.rfqs import router as rfqs_router
from .routers.intelligence import router as intelligence_router
from .routers.ingestion import router as ingestion_router
from .routers.notifications import router as notifications_router
from .routers.companies import router as companies_router
from .routers.monitoring import router as monitoring_router
from .routers.users import router as users_router
from .routers.inventory import router as inventory_router
from .routers.purchase_orders import router as purchase_orders_router
from .routers.graph import router as graph_router
from .routers.audit import router as audit_router
from .routers.integrations import router as integrations_router
from .routers.search import router as search_router
from .routers.market import router as market_router
from .routers.integrations import router as integrations_router
from .routers.health import router as health_router
from .routers.ai import router as ai_router
from .db.base import Base
from .db.session import engine
from .core.auth import require_api_key_or_bearer
from .core.config import settings
from .search.indexer import ensure_indices
from .middleware.rate_limit import RateLimiter
from .middleware.metrics import PrometheusRequestMetricsMiddleware
from datetime import datetime, timezone
import uuid
import asyncio
from .db.mongo import ensure_mongo_indexes
from starlette.middleware.cors import CORSMiddleware
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    minimal_mode = os.getenv("SCIP_MINIMAL_STARTUP") == "1"
    # Startup warnings for production auth posture
    try:
        if os.getenv("SCIP_ENV") == "production":
            if not os.getenv("OIDC_ISSUER"):
                print("Warning: Running in production without OIDC_ISSUER configured; JWT verification may be insecure.")
            if os.getenv("SCIP_AUTH_MODE", "mixed").lower() != "bearer_only":
                print("Warning: In production, set SCIP_AUTH_MODE=bearer_only to disable API keys.")
    except Exception:
        pass
    # Startup
    Base.metadata.create_all(bind=engine)
    ensure_indices()
    try:
        await ensure_mongo_indexes()
    except Exception:
        pass
    
    # Initialize performance monitoring
    from .monitoring.performance import performance_monitor, PerformanceMiddleware
    from .monitoring.distributed import start_distributed_monitoring
    from .cache.redis_cache import init_cache
    from .cache.advanced_cache import start_advanced_caching
    from .scaling.load_balancer import service_registry
    from .ai.web_intelligence import start_web_intelligence
    
    # Initialize cache connections
    if not minimal_mode:
        try:
            await init_cache()
            await start_advanced_caching()
            print("✓ Advanced caching system initialized")
        except Exception as e:
            print(f"Warning: Cache initialization failed: {e}")
    
    # Start distributed monitoring
    if not minimal_mode:
        try:
            await start_distributed_monitoring()
            print("✓ Distributed monitoring started")
        except Exception as e:
            print(f"Warning: Distributed monitoring failed to start: {e}")
    
    # Start service registry and load balancer health checks
    if not minimal_mode:
        try:
            await service_registry.start_all_health_checks()
            print("✓ Load balancer health checks started")
        except Exception as e:
            print(f"Warning: Load balancer initialization failed: {e}")
    
    # Start web intelligence gathering
    if not minimal_mode:
        try:
            await start_web_intelligence()
            print("✓ Web intelligence gathering started")
        except Exception as e:
            print(f"Warning: Web intelligence initialization failed: {e}")
    
    # Initialize AI capabilities framework
    if not minimal_mode:
        try:
            from .ai.capabilities import initialize_capabilities
            capability_results = await initialize_capabilities()
            successful_capabilities = sum(1 for success in capability_results.values() if success)
            total_capabilities = len(capability_results)
            print(f"✓ AI capabilities framework initialized: {successful_capabilities}/{total_capabilities} successful")
        except Exception as e:
            print(f"Warning: AI capabilities initialization failed: {e}")
    
    # Initialize Neo4j graph database
    if not minimal_mode:
        try:
            from .graph import neo4j_client
            await neo4j_client.initialize()
            print("✓ Neo4j graph database initialized")
        except Exception as e:
            print(f"Warning: Neo4j initialization failed: {e}")
    
    # Initialize Kafka client
    if not minimal_mode:
        try:
            from .events import kafka_client
            await kafka_client.start()
            print("✓ Kafka client started")
        except Exception as e:
            print(f"Warning: Kafka client initialization failed: {e}")
    
    # Initialize observability
    if not minimal_mode:
        try:
            from .observability import init_tracing, init_metrics, init_structured_logging, health_monitor
            from .observability.health import init_health_checks
            
            # Initialize tracing
            tracing_success = init_tracing()
            print(f"✓ OpenTelemetry tracing {'enabled' if tracing_success else 'disabled'}")
            
            # Initialize metrics
            metrics_success = init_metrics()
            print(f"✓ Prometheus metrics {'enabled' if metrics_success else 'disabled'}")
            
            # Initialize structured logging
            logging_success = init_structured_logging(
                level="INFO",
                service_name=settings.otel_service_name,
                enable_json_format=True
            )
            print(f"✓ Structured logging {'enabled' if logging_success else 'disabled'}")
            
            # Initialize health checks
            init_health_checks()
            await health_monitor.start_monitoring()
            print("✓ Health monitoring started")
            
        except Exception as e:
            print(f"Warning: Observability initialization failed: {e}")
    
    # Initialize notification system
    if not minimal_mode:
        try:
            from .notifications import notification_manager
            await notification_manager.start_background_tasks()
            print("✓ Notification system started")
        except Exception as e:
            print(f"Warning: Notification system initialization failed: {e}")
    
    yield
    
    # Shutdown - cleanup
    from .monitoring.performance import shutdown_performance_monitoring
    if not minimal_mode:
        from .monitoring.distributed import stop_distributed_monitoring
        from .cache.redis_cache import cleanup_cache
        from .cache.advanced_cache import stop_advanced_caching
        from .ai.web_intelligence import stop_web_intelligence
    
    try:
        shutdown_performance_monitoring()
        if not minimal_mode:
            await stop_distributed_monitoring()
            await stop_advanced_caching()
            await cleanup_cache()
            await service_registry.stop_all_health_checks()
            await stop_web_intelligence()
            
            # Cleanup Neo4j and Kafka
            from .graph import neo4j_client
            from .events import kafka_client
            await neo4j_client.close()
            await kafka_client.stop()
            
            # Cleanup AI capabilities
            from .ai.capabilities import cleanup_capabilities
            await cleanup_capabilities()
            
            # Cleanup observability and notifications
            from .observability import health_monitor
            from .notifications import notification_manager
            await health_monitor.stop_monitoring()
            await notification_manager.stop_background_tasks()
        
        print("✓ All services shut down cleanly")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")

def create_app() -> FastAPI:
    app = FastAPI(title="SCIP Core API", version="0.1.0", lifespan=lifespan)
    # Dev CORS for local frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit","X-RateLimit-Remaining","X-RateLimit-Reset","X-Request-ID"],
    )

    # Health and metrics endpoints (no auth required)
    app.include_router(health_router, tags=["health"])

    # Routers (v1)
    app.include_router(auth_router, prefix="/v1/auth", tags=["auth"])
    app.include_router(components_router, prefix="/v1/components", tags=["components"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(companies_router, prefix="/v1/companies", tags=["companies"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(users_router, prefix="/v1/users", tags=["users"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(inventory_router, prefix="/v1/inventory", tags=["inventory"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(purchase_orders_router, prefix="/v1/purchase-orders", tags=["purchase_orders"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(rfqs_router, prefix="/v1/rfqs", tags=["rfqs"], dependencies=[Depends(require_api_key_or_bearer)])
    # Require market intelligence read scope for intelligence endpoints
    from .core.auth import require_scopes
    app.include_router(intelligence_router, prefix="/v1/intelligence", tags=["intelligence"], dependencies=[Depends(require_scopes(["read:market_intelligence"]))])
    app.include_router(graph_router, prefix="/v1/graph", tags=["graph"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(search_router, prefix="/v1/search", tags=["search"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(integrations_router, prefix="/v1/integrations", tags=["integrations"], dependencies=[Depends(require_api_key_or_bearer)])
    # Require integrations read scope for market provider endpoints
    app.include_router(market_router, prefix="/v1/market", tags=["market"], dependencies=[Depends(require_scopes(["read:integrations"]))])
    app.include_router(ingestion_router, prefix="/v1/ingestion", tags=["ingestion"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(notifications_router, prefix="/v1", tags=["notifications"])
    app.include_router(monitoring_router, prefix="/v1/monitoring", tags=["monitoring"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(audit_router, prefix="/v1/audit", tags=["audit"], dependencies=[Depends(require_api_key_or_bearer)])
    app.include_router(ai_router, prefix="/v1/ai", tags=["ai"], dependencies=[Depends(require_api_key_or_bearer)])

    # Add performance monitoring middleware
    from .monitoring.performance import performance_monitor, PerformanceMiddleware
    app.add_middleware(PerformanceMiddleware, monitor=performance_monitor)
    app.add_middleware(PrometheusRequestMetricsMiddleware)

    # Rate limit middleware (in-memory, per minute)
    app.add_middleware(RateLimiter, rate_per_minute=settings.api_rate_limit_per_min)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc) or "Unexpected server error",
                    "requestId": rid,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
            headers={"X-Request-ID": rid},
        )

    from fastapi import HTTPException

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            409: "CONFLICT",
            429: "RATE_LIMITED",
        }
        code = code_map.get(exc.status_code, "ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": exc.detail,
                    "requestId": rid,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
            headers={"X-Request-ID": rid},
        )

    return app


app = create_app()
