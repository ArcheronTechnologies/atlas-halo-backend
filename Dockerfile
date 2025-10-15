FROM python:3.12.11 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app


RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt
FROM python:3.12.11-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
RUN find .venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN find .venv -name "*.pyc" -delete 2>/dev/null || true
COPY . .
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
