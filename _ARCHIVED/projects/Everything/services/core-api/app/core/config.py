import os
from dataclasses import dataclass
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    from pathlib import Path
    # Load nearest .env from CWD upward
    loaded = load_dotenv(find_dotenv(), override=False)
    # Fallback: load service-local .env (two levels up from this file)
    if not loaded:
        service_env = Path(__file__).resolve().parents[2] / ".env"
        if service_env.exists():
            load_dotenv(service_env, override=False)
except Exception:
    # dotenv is optional; ignore if unavailable
    pass


@dataclass
class Settings:
    secret_key: str = os.getenv("SCIP_SECRET_KEY", "dev-secret-key-change-me")
    api_rate_limit_per_min: int = int(os.getenv("SCIP_RATE_LIMIT", "1000"))
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./scip.db")
    jwt_leeway: int = int(os.getenv("JWT_LEEWAY", "120"))
    redis_url: str | None = os.getenv("REDIS_URL")
    
    # OIDC Configuration
    oidc_issuer: str = os.getenv("OIDC_ISSUER", "")
    oidc_client_id: str = os.getenv("OIDC_CLIENT_ID", "")
    oidc_client_secret: str = os.getenv("OIDC_CLIENT_SECRET", "")
    oidc_auth_endpoint: str = os.getenv("OIDC_AUTH_ENDPOINT", "")
    oidc_token_endpoint: str = os.getenv("OIDC_TOKEN_ENDPOINT", "")
    oidc_userinfo_endpoint: str = os.getenv("OIDC_USERINFO_ENDPOINT", "")
    oidc_jwks_uri: str = os.getenv("OIDC_JWKS_URI", "")
    oidc_end_session_endpoint: str = os.getenv("OIDC_END_SESSION_ENDPOINT", "")
    oidc_provider: str = os.getenv("OIDC_PROVIDER", "azure_ad")  # azure_ad, auth0, keycloak
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_security_protocol: str = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    kafka_sasl_mechanism: str = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
    kafka_username: str = os.getenv("KAFKA_USERNAME", "")
    kafka_password: str = os.getenv("KAFKA_PASSWORD", "")
    kafka_topic_prefix: str = os.getenv("KAFKA_TOPIC_PREFIX", "scip")
    
    # Neo4j Configuration
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # OpenTelemetry Configuration
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "scip-api")
    otel_endpoint: str = os.getenv("OTEL_ENDPOINT", "")
    otel_headers: str = os.getenv("OTEL_HEADERS", "")
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "8001"))


settings = Settings()
