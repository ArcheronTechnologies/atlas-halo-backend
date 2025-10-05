web: uvicorn main:app --host 0.0.0.0 --port $PORT --ws none
worker: celery -A backend.tasks.celery_app worker --loglevel=info
beat: celery -A backend.tasks.celery_app beat --loglevel=info
