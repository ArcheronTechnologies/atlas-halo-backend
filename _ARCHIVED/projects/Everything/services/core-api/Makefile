run:
	uvicorn app.main:app --reload --port 8000

install:
	python -m pip install -e .

lint:
	ruff check app || true

test:
	pytest -q

test-offline:
	SCIP_MINIMAL_STARTUP=1 SCIP_SKIP_NETWORK=1 pytest -q

seed:
	python scripts/seed.py

migrate:
	alembic upgrade head

revision:
	alembic revision --autogenerate -m "auto"

es-ensure:
	python scripts/es.py ensure

es-reindex:
	python scripts/es.py reindex-components
