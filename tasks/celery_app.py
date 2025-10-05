"""
Celery Application Configuration
Async task processing for AI analysis and media processing
"""

import os
from celery import Celery

# Redis as message broker
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'atlas_ai',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        'backend.tasks.ai_tasks',
        'backend.tasks.media_tasks',
        'backend.tasks.notification_tasks',
        'backend.tasks.data_ingestion_tasks'
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks (prevent memory leaks)
)

# Task routing
celery_app.conf.task_routes = {
    'backend.tasks.ai_tasks.*': {'queue': 'ai_analysis'},
    'backend.tasks.media_tasks.*': {'queue': 'media_processing'},
    'backend.tasks.notification_tasks.*': {'queue': 'notifications'},
    'backend.tasks.data_ingestion_tasks.*': {'queue': 'data_ingestion'},
}

# Celery Beat schedule for periodic tasks
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'fetch-polisen-data-hourly': {
        'task': 'fetch_polisen_incidents',
        'schedule': crontab(minute=0),  # Every hour at the top of the hour
        'options': {
            'queue': 'data_ingestion',
            'priority': 7  # High priority for fresh data
        }
    },
    'validate-predictions-hourly': {
        'task': 'validate_predictions_against_actuals',
        'schedule': crontab(minute=15),  # Every hour at :15 (after data fetch)
        'options': {
            'queue': 'ai_analysis',
            'priority': 6
        }
    },
    'compute-temporal-risk-daily': {
        'task': 'compute_temporal_risk_scores',
        'schedule': crontab(hour=2, minute=0),  # Daily at 02:00 UTC
        'options': {
            'queue': 'ai_analysis',
            'priority': 5  # Medium-high priority
        }
    },
    'retrain-prediction-model-weekly': {
        'task': 'retrain_prediction_model',
        'schedule': crontab(hour=3, minute=0, day_of_week=1),  # Monday at 03:00 UTC
        'options': {
            'queue': 'ai_analysis',
            'priority': 4
        }
    },
}

if __name__ == '__main__':
    celery_app.start()
