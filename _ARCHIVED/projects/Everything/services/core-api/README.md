# SCIP Core API

FastAPI-based core API implementing the initial endpoints from API_SPECIFICATIONS.md.

This is an MVP scaffold with in-memory storage so we can iterate on API design before wiring databases and Kafka.

## Run (dev)

1. Ensure Python 3.11+
2. Create a virtual environment and install deps

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. Configure environment

By default, the API uses a local SQLite database at `./scip.db`. To use PostgreSQL, set `DATABASE_URL` in `.env`:

```
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/scip
```

Optional integrations:

- OIDC JWT verification (production):
  - `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_PATH` (local JWKS file)
- MongoDB for ingestion docs:
  - `MONGODB_URL`, `MONGODB_DB` (default `scip`)
- Elasticsearch indexing:
  - `ELASTICSEARCH_URL`
- Octopart API (market intelligence):
  - `OCTOPART_API_KEY` (required for Octopart endpoints)
  - `OCTOPART_ENDPOINT` (default `https://api.octopart.com/v4/graphql`)
  - `OCTOPART_TIMEOUT_SECONDS` (default `15`)
  - `OCTOPART_MAX_CONCURRENCY` (default `5`)
  - `OCTOPART_CACHE_TTL_SECONDS` (default `300`)
  - `OCTOPART_RATE_LIMIT_PER_MIN` (default `0` = disabled)
- Redis-backed rate limiting (production):
  - `REDIS_URL` (e.g., redis://localhost:6379/0)
- API keys:
  - Create via `POST /v1/users/{id}/api-keys` (admin scope); use with header `X-API-Key: <key>`
 - Kafka events (optional):
   - `KAFKA_BROKERS` (comma-separated)
 - Neo4j graph (optional):
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

4. (Optional) Seed sample data

```
make seed
```

5. Start server

```
uvicorn app.main:app --reload --port 8000
```

4. Open docs at `http://localhost:8000/docs`

## Notes
- Auth tokens are stubbed for now; replace with real OAuth2/JWT later.
- Storage is in-memory; swap with PostgreSQL/MongoDB per DATABASE_SCHEMAS.md.
- WebSocket notifications are a no-op echo for now.

Scopes enforced per endpoint:
- Components: `read:components`, `write:components`
- RFQs: `read:rfqs`, `write:rfqs`
- Companies: `read:companies`, `write:companies`

Rate limiting:
- Default `SCIP_RATE_LIMIT=1000` requests/min per token/IP.
- Responses include `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` headers.

## Next
- Add MongoDB (motor) for unstructured email/web docs
- Implement real authentication (OAuth2/OIDC + JWT verification)
- Define Kafka topics and producers/consumers for ingestion
- Add Elasticsearch indexing for components/documents

## Auth

- Obtain token via `POST /v1/auth/login` with body:

```
{
  "email": "user@company.com",
  "password": "dev"
}
```

- Use the `accessToken` as a bearer token:

```
Authorization: Bearer <accessToken>
```

- WebSocket auth: connect to `/v1/ws/notifications?token=<accessToken>`
- Alternatively, use API keys with `X-API-Key: <key>` on protected endpoints
 - OIDC scaffolding:
   - `GET /v1/auth/oidc/authorize?redirect_uri=...` → returns authorizationUrl + sets cookies
   - `GET /v1/auth/oidc/callback?code=...&state=...&redirect_uri=...` exchanges code and returns API tokens
   - Env: `OIDC_AUTH_URL`, `OIDC_TOKEN_URL`, `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET` (optional for public clients)

To use OIDC (recommended for prod), set issuer, audience, and JWKS file path.

Production auth mode
- Set `SCIP_AUTH_MODE=bearer_only` to require JWT bearer tokens and disallow API keys.
- Default is `mixed` (accepts both bearer and API key) for development convenience.

## Database migrations (Alembic)

Generate a new migration after model changes:

```
cd services/core-api
alembic revision --autogenerate -m "your message"
```

Apply migrations:

```
make migrate
```

The dev server also creates tables automatically for SQLite; in production use Alembic migrations.

## Endpoints (highlights)

- Components
  - `GET /v1/components?search=...&limit=&offset=`
  - `POST /v1/components`
  - `PUT /v1/components/{id}`
  - `DELETE /v1/components/{id}`
  - `GET /v1/components/{id}/pricing`
- Companies
  - `GET /v1/companies?search=&type=`
  - `POST /v1/companies`
  - `GET /v1/companies/{id}`
  - `PUT /v1/companies/{id}`
  - `DELETE /v1/companies/{id}`
- RFQs
  - `GET /v1/rfqs`
  - `POST /v1/rfqs`
  - `POST /v1/rfqs/{id}/quote`
  - `GET /v1/rfqs/{id}`
  - `POST /v1/rfqs/{id}/status` with `{ "status": "won|lost|expired|quoted|open" }`
 - Inventory
   - `GET /v1/inventory?componentId=`
   - `POST /v1/inventory`
   - `DELETE /v1/inventory/{id}`
- Purchase Orders
  - `GET /v1/purchase-orders`
  - `POST /v1/purchase-orders`
  - `GET /v1/purchase-orders/{id}`
  - `POST /v1/purchase-orders/{id}/status`
  - `POST /v1/purchase-orders/{id}/items`
  - `DELETE /v1/purchase-orders/{id}/items/{item_id}`
- Users (admin)
  - `GET /v1/users`
  - `POST /v1/users`
  - `POST /v1/users/{id}/roles/{role}`
  - `POST /v1/users/{id}/api-keys`
- Audit (admin)
  - `GET /v1/audit?entityType=&entityId=&limit=&offset=`
- Graph (optional)
  - `POST /v1/graph/sync`
  - `GET /v1/graph/components/{component_id}/neighbors`
- Market (Octopart)
  - `GET /v1/market/total-availability?q=&country=&limit=`
  - `GET /v1/market/pricing-breaks?q=&limit=&currency=`
  - `GET /v1/market/offers?mpn=&country=&currency=`
  - `GET /v1/market/spec-attributes?q=&limit=&filters=`
    - `filters` is a JSON string, e.g. `{"case_package":["SSOP"]}`. Attribute shortnames per Octopart docs.

Production behavior
- All Octopart calls use:
  - Async concurrency limit (`OCTOPART_MAX_CONCURRENCY`)
  - Exponential backoff retries on failure
  - Optional Redis-backed response caching (`OCTOPART_CACHE_TTL_SECONDS`)
  - Optional per-minute rate limiting via Redis (`OCTOPART_RATE_LIMIT_PER_MIN`)
  - Request timeouts (`OCTOPART_TIMEOUT_SECONDS`)
- To disable external calls (local dev/CI): set `SCIP_SKIP_NETWORK=1`.
- To minimize startup (skip Kafka/Neo4j/observability/etc): set `SCIP_MINIMAL_STARTUP=1`.

## Search CLI (Elasticsearch)

Ensure indices:

```
make es-ensure
```

Reindex components:

```
make es-reindex
```
