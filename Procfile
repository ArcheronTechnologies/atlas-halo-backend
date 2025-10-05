web: gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
worker: celery -A backend.tasks.celery_app worker --loglevel=info
beat: celery -A backend.tasks.celery_app beat --loglevel=info
