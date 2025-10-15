"""
Prometheus Metrics Exporter
Exposes system metrics for Prometheus scraping
Updated for graceful degradation without psutil
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
from prometheus_client.core import CollectorRegistry
from fastapi import APIRouter, Response
from datetime import datetime
import time

# Optional psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    import logging
    logging.warning("⚠️ psutil not available - system metrics disabled")

# Create metrics router
router = APIRouter(prefix="/metrics", tags=["monitoring"])

# =============================================================================
# DATA INGESTION METRICS
# =============================================================================

# Incident collection metrics
incidents_collected_total = Counter(
    'atlas_incidents_collected_total',
    'Total number of incidents collected from all sources',
    ['source', 'city']
)

incidents_stored_total = Counter(
    'atlas_incidents_stored_total',
    'Total number of incidents successfully stored',
    ['source']
)

incidents_duplicate_total = Counter(
    'atlas_incidents_duplicate_total',
    'Total number of duplicate incidents detected',
    ['source']
)

incidents_rejected_total = Counter(
    'atlas_incidents_rejected_total',
    'Total number of incidents rejected (quality/validation)',
    ['reason']
)

# Collection performance
collection_duration_seconds = Histogram(
    'atlas_collection_duration_seconds',
    'Time taken to collect incidents from a source',
    ['source', 'city'],
    buckets=(1, 5, 10, 30, 60, 120, 300)
)

# =============================================================================
# DATA QUALITY METRICS
# =============================================================================

data_quality_score = Histogram(
    'atlas_data_quality_score',
    'Quality score of incidents (0.0-1.0)',
    ['source'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

validation_failures_total = Counter(
    'atlas_validation_failures_total',
    'Total number of validation failures',
    ['validation_type']
)

# =============================================================================
# WEBSOCKET METRICS
# =============================================================================

websocket_connections_active = Gauge(
    'atlas_websocket_connections_active',
    'Number of active WebSocket connections'
)

websocket_messages_sent_total = Counter(
    'atlas_websocket_messages_sent_total',
    'Total number of WebSocket messages sent',
    ['message_type']
)

websocket_broadcast_duration_seconds = Histogram(
    'atlas_websocket_broadcast_duration_seconds',
    'Time taken to broadcast a message',
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
)

# =============================================================================
# DATABASE METRICS
# =============================================================================

database_queries_total = Counter(
    'atlas_database_queries_total',
    'Total number of database queries',
    ['query_type']
)

database_query_duration_seconds = Histogram(
    'atlas_database_query_duration_seconds',
    'Time taken to execute database queries',
    ['query_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
)

database_connections_active = Gauge(
    'atlas_database_connections_active',
    'Number of active database connections'
)

database_errors_total = Counter(
    'atlas_database_errors_total',
    'Total number of database errors',
    ['error_type']
)

# =============================================================================
# MEDIA STORAGE METRICS
# =============================================================================

media_files_uploaded_total = Counter(
    'atlas_media_files_uploaded_total',
    'Total number of media files uploaded',
    ['media_type']
)

media_compression_ratio = Histogram(
    'atlas_media_compression_ratio',
    'Compression ratio achieved for media files',
    ['media_type'],
    buckets=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0)
)

media_storage_bytes_saved_total = Counter(
    'atlas_media_storage_bytes_saved_total',
    'Total bytes saved through compression',
    ['media_type']
)

# =============================================================================
# ML MODEL METRICS
# =============================================================================

model_predictions_total = Counter(
    'atlas_model_predictions_total',
    'Total number of predictions made',
    ['model_type']
)

model_prediction_confidence = Histogram(
    'atlas_model_prediction_confidence',
    'Confidence score of predictions',
    ['model_type'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

model_training_duration_seconds = Histogram(
    'atlas_model_training_duration_seconds',
    'Time taken to train ML models',
    ['model_type'],
    buckets=(60, 300, 600, 1800, 3600, 7200)  # 1min to 2hrs
)

model_accuracy = Gauge(
    'atlas_model_accuracy',
    'Current model accuracy',
    ['model_type']
)

# =============================================================================
# API METRICS
# =============================================================================

api_requests_total = Counter(
    'atlas_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration_seconds = Histogram(
    'atlas_api_request_duration_seconds',
    'Time taken to process API requests',
    ['method', 'endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

api_errors_total = Counter(
    'atlas_api_errors_total',
    'Total number of API errors',
    ['endpoint', 'error_type']
)

# =============================================================================
# SYSTEM METRICS
# =============================================================================

system_cpu_usage_percent = Gauge(
    'atlas_system_cpu_usage_percent',
    'CPU usage percentage'
)

system_memory_usage_bytes = Gauge(
    'atlas_system_memory_usage_bytes',
    'Memory usage in bytes'
)

system_memory_usage_percent = Gauge(
    'atlas_system_memory_usage_percent',
    'Memory usage percentage'
)

system_disk_usage_bytes = Gauge(
    'atlas_system_disk_usage_bytes',
    'Disk usage in bytes',
    ['mount_point']
)

system_disk_usage_percent = Gauge(
    'atlas_system_disk_usage_percent',
    'Disk usage percentage',
    ['mount_point']
)

# =============================================================================
# APPLICATION INFO
# =============================================================================

app_info = Info(
    'atlas_app',
    'Atlas AI application information'
)

app_info.info({
    'version': '1.0.0',
    'name': 'Atlas AI',
    'environment': 'development'
})

app_uptime_seconds = Gauge(
    'atlas_app_uptime_seconds',
    'Application uptime in seconds'
)

# Store start time
_app_start_time = time.time()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def update_system_metrics():
    """Update system-level metrics"""
    if not PSUTIL_AVAILABLE:
        # Only update uptime when psutil unavailable
        uptime = time.time() - _app_start_time
        app_uptime_seconds.set(uptime)
        return

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    system_cpu_usage_percent.set(cpu_percent)

    # Memory usage
    memory = psutil.virtual_memory()
    system_memory_usage_bytes.set(memory.used)
    system_memory_usage_percent.set(memory.percent)

    # Disk usage
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            system_disk_usage_bytes.labels(mount_point=partition.mountpoint).set(usage.used)
            system_disk_usage_percent.labels(mount_point=partition.mountpoint).set(usage.percent)
        except PermissionError:
            pass

    # Uptime
    uptime = time.time() - _app_start_time
    app_uptime_seconds.set(uptime)


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/prometheus")
async def get_metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus text format
    """
    # Update system metrics before scraping
    update_system_metrics()

    # Generate and return metrics
    metrics_output = generate_latest(REGISTRY)
    return Response(content=metrics_output, media_type="text/plain; charset=utf-8")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - _app_start_time
    }


# =============================================================================
# METRIC RECORDING FUNCTIONS
# =============================================================================

def record_incident_collection(source: str, city: str, count: int, duration: float):
    """Record incident collection metrics"""
    incidents_collected_total.labels(source=source, city=city).inc(count)
    collection_duration_seconds.labels(source=source, city=city).observe(duration)


def record_incident_storage(source: str, stored: int, duplicates: int):
    """Record incident storage metrics"""
    incidents_stored_total.labels(source=source).inc(stored)
    incidents_duplicate_total.labels(source=source).inc(duplicates)


def record_incident_rejection(reason: str):
    """Record incident rejection"""
    incidents_rejected_total.labels(reason=reason).inc()


def record_data_quality(source: str, quality_score: float):
    """Record data quality score"""
    data_quality_score.labels(source=source).observe(quality_score)


def record_websocket_connection(delta: int):
    """Record WebSocket connection change"""
    if delta > 0:
        websocket_connections_active.inc(delta)
    else:
        websocket_connections_active.dec(abs(delta))


def record_websocket_broadcast(message_type: str, duration: float):
    """Record WebSocket broadcast"""
    websocket_messages_sent_total.labels(message_type=message_type).inc()
    websocket_broadcast_duration_seconds.observe(duration)


def record_database_query(query_type: str, duration: float):
    """Record database query"""
    database_queries_total.labels(query_type=query_type).inc()
    database_query_duration_seconds.labels(query_type=query_type).observe(duration)


def record_database_error(error_type: str):
    """Record database error"""
    database_errors_total.labels(error_type=error_type).inc()


def record_media_upload(media_type: str, compression_ratio: float, bytes_saved: int):
    """Record media upload"""
    media_files_uploaded_total.labels(media_type=media_type).inc()
    media_compression_ratio.labels(media_type=media_type).observe(compression_ratio)
    media_storage_bytes_saved_total.labels(media_type=media_type).inc(bytes_saved)


def record_model_prediction(model_type: str, confidence: float):
    """Record ML model prediction"""
    model_predictions_total.labels(model_type=model_type).inc()
    model_prediction_confidence.labels(model_type=model_type).observe(confidence)


def record_api_request(method: str, endpoint: str, status: int, duration: float):
    """Record API request"""
    api_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


def record_api_error(endpoint: str, error_type: str):
    """Record API error"""
    api_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()