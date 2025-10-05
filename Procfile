web: hypercorn main:app --bind 0.0.0.0:$PORT --workers 4
worker: celery -A backend.tasks.celery_app worker --loglevel=info
beat: celery -A backend.tasks.celery_app beat --loglevel=info
