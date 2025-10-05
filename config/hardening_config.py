"""
Production Hardening Configuration
Security, performance, and reliability settings
"""

import os
from typing import Dict, List

# =============================================================================
# RATE LIMITING
# =============================================================================

RATE_LIMIT_CONFIG = {
    # Global rate limits (per IP)
    'global': {
        'requests': 100,
        'period': 60,  # seconds
    },

    # API endpoint rate limits
    'endpoints': {
        '/api/v1/ai/analyze/photo': {
            'requests': 10,
            'period': 60,
        },
        '/api/v1/ai/analyze/video': {
            'requests': 5,
            'period': 60,
        },
        '/api/v1/media/upload': {
            'requests': 20,
            'period': 60,
        },
        '/api/v1/mobile/incidents': {
            'requests': 60,
            'period': 60,
        },
    },

    # Authenticated user limits (higher)
    'authenticated': {
        'requests': 300,
        'period': 60,
    }
}

# =============================================================================
# DATABASE CONNECTION POOLING
# =============================================================================

DATABASE_POOL_CONFIG = {
    'min_size': int(os.getenv('POSTGRES_POOL_MIN', '5')),
    'max_size': int(os.getenv('POSTGRES_POOL_SIZE', '20')),
    'max_queries': 50000,  # Close connection after N queries
    'max_inactive_connection_lifetime': 300,  # 5 minutes
    'timeout': 10,  # Connection timeout
    'command_timeout': 30,  # Query timeout
}

# =============================================================================
# CACHING
# =============================================================================

CACHE_CONFIG = {
    # In-memory caching for frequently accessed data
    'enabled': True,
    'backend': 'memory',  # 'memory' or 'redis' for production
    'ttl': {
        'incidents': 60,  # 1 minute
        'predictions': 300,  # 5 minutes
        'ai_analysis': 3600,  # 1 hour (same file = same analysis)
        'user_profile': 600,  # 10 minutes
    },
    'max_size': 1000,  # Max items in memory cache
}

# Redis configuration (for production)
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': int(os.getenv('REDIS_DB', '0')),
    'password': os.getenv('REDIS_PASSWORD'),
    'max_connections': 50,
}

# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

CIRCUIT_BREAKER_CONFIG = {
    # Circuit breaker for external services
    'failure_threshold': 5,  # Open circuit after N failures
    'recovery_timeout': 60,  # Try to recover after N seconds
    'expected_exception': Exception,
}

# =============================================================================
# REQUEST VALIDATION
# =============================================================================

VALIDATION_CONFIG = {
    # File upload limits
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'allowed_image_types': {'image/jpeg', 'image/png', 'image/heic'},
    'allowed_video_types': {'video/mp4', 'video/quicktime', 'video/x-m4v'},
    'allowed_audio_types': {'audio/mpeg', 'audio/mp4', 'audio/x-m4a'},

    # Geo queries
    'max_radius': 50000,  # 50km max search radius
    'max_results': 1000,  # Max results per query

    # Text fields
    'max_description_length': 5000,
    'max_category_length': 100,
}

# =============================================================================
# CORS CONFIGURATION
# =============================================================================

CORS_CONFIG = {
    'allow_origins': [
        'http://localhost:8081',  # Expo dev server
        'exp://localhost:8081',
        'http://192.168.0.142:8081',  # Local network
        'exp://192.168.0.142:8081',
    ],
    'allow_credentials': True,
    'allow_methods': ['*'],
    'allow_headers': ['*'],
    'expose_headers': ['X-Total-Count', 'X-Page-Count'],
}

# Production CORS (more restrictive)
if os.getenv('ENV') == 'production':
    CORS_CONFIG['allow_origins'] = [
        os.getenv('FRONTEND_URL', 'https://atlasai.app'),
        os.getenv('MOBILE_APP_URL', 'https://app.atlasai.app'),
    ]

# =============================================================================
# SECURITY HEADERS
# =============================================================================

SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'",
}

# =============================================================================
# MONITORING & OBSERVABILITY
# =============================================================================

MONITORING_CONFIG = {
    'enable_metrics': True,
    'enable_tracing': False,  # Distributed tracing (future)
    'enable_profiling': os.getenv('ENV') != 'production',

    # Prometheus metrics
    'metrics_port': int(os.getenv('METRICS_PORT', '9090')),
    'metrics_path': '/metrics',

    # Slow query logging
    'slow_query_threshold_ms': 1000,
    'log_slow_queries': True,

    # Error tracking
    'enable_sentry': os.getenv('SENTRY_DSN') is not None,
    'sentry_dsn': os.getenv('SENTRY_DSN'),
    'sentry_environment': os.getenv('ENV', 'development'),
}

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

PERFORMANCE_CONFIG = {
    # Query optimization
    'enable_query_caching': True,
    'explain_slow_queries': True,

    # Media processing
    'image_compression_quality': 85,
    'image_max_dimension': 2048,
    'thumbnail_size': (300, 300),
    'enable_lazy_loading': True,

    # AI inference
    'ai_batch_size': 1,  # Process N images at once
    'ai_timeout': 30,  # Max inference time
    'enable_ai_caching': True,  # Cache identical file analysis

    # WebSocket
    'ws_heartbeat_interval': 30,
    'ws_max_connections': 10000,
    'ws_message_size_limit': 1024 * 1024,  # 1MB
}

# =============================================================================
# BACKUP & RECOVERY
# =============================================================================

BACKUP_CONFIG = {
    'enable_auto_backup': True,
    'backup_interval_hours': 24,
    'backup_retention_days': 30,
    'backup_path': os.getenv('BACKUP_PATH', './backups'),
}

# =============================================================================
# FEATURE FLAGS
# =============================================================================

FEATURE_FLAGS = {
    'enable_ai_analysis': True,
    'enable_push_notifications': True,
    'enable_user_uploads': True,
    'enable_anonymous_reports': True,
    'enable_community_verification': True,
    'enable_incident_clustering': True,
    'enable_ml_predictions': True,
    'enable_real_time_updates': True,
}
