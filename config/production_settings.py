"""
Atlas AI Production Configuration Management
Complete configuration system with validation and environment support
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "atlas_ai"
    username: str = "atlas_user"
    password: str = "secure_password"
    ssl_mode: str = "prefer"
    pool_min_size: int = 1
    pool_max_size: int = 20
    
    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    database: int = 0
    ssl: bool = False
    
    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.database}"


@dataclass 
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "change-me-in-production-very-long-secret-key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    rate_limit_requests_per_minute: int = 100
    ssl_required: bool = False


@dataclass
class AIConfig:
    """AI and ML configuration."""
    model_storage_path: str = "models"
    threat_detection_threshold: float = 0.7
    retraining_trigger_threshold: int = 100
    max_concurrent_training_jobs: int = 2


class ProductionConfigManager:
    """Production-ready configuration manager."""
    
    def __init__(self):
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Database
        self._database_config = DatabaseConfig(
            host=os.getenv("DATABASE_HOST", "localhost"),
            port=int(os.getenv("DATABASE_PORT", "5432")),
            database=os.getenv("DATABASE_NAME", "atlas_ai"),
            username=os.getenv("DATABASE_USER", "atlas_user"),
            password=os.getenv("DATABASE_PASSWORD", "secure_password"),
            ssl_mode=os.getenv("DATABASE_SSL_MODE", "prefer")
        )
        
        # Redis
        self._redis_config = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            database=int(os.getenv("REDIS_DATABASE", "0")),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true"
        )
        
        # Security
        cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        
        self._security_config = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", "change-me-in-production-very-long-secret-key"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            cors_origins=cors_origins,
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "100")),
            ssl_required=os.getenv("SSL_REQUIRED", "false").lower() == "true"
        )
        
        # AI
        self._ai_config = AIConfig(
            model_storage_path=os.getenv("MODEL_STORAGE_PATH", "models"),
            threat_detection_threshold=float(os.getenv("THREAT_DETECTION_THRESHOLD", "0.7")),
            retraining_trigger_threshold=int(os.getenv("RETRAINING_TRIGGER", "100")),
            max_concurrent_training_jobs=int(os.getenv("MAX_TRAINING_JOBS", "2"))
        )
        
        # Ensure model directory exists
        os.makedirs(self._ai_config.model_storage_path, exist_ok=True)
        
        # Log level
        self.log_level = LogLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return self._database_config
    
    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration."""
        return self._redis_config
    
    @property
    def security(self) -> SecurityConfig:
        """Get security configuration."""
        return self._security_config
    
    @property
    def ai(self) -> AIConfig:
        """Get AI configuration."""
        return self._ai_config
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return issues."""
        issues = []
        
        # Security validation
        if len(self.security.secret_key) < 32:
            issues.append("Secret key should be at least 32 characters")
        
        if self.is_production():
            if "change-me" in self.security.secret_key:
                issues.append("Secret key must be changed in production")
            
            if not self.security.ssl_required:
                issues.append("SSL should be required in production") 
            
            if self.debug:
                issues.append("Debug mode should be disabled in production")
        
        return issues
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=getattr(logging, self.log_level.value),
            format=log_format
        )
        
        if self.is_development():
            logging.getLogger("uvicorn").setLevel(logging.INFO)
        
        if self.is_production():
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)


# Global configuration instance
_config: Optional[ProductionConfigManager] = None


def get_config() -> ProductionConfigManager:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ProductionConfigManager()
    return _config


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database


def get_redis_config() -> RedisConfig:
    """Get Redis configuration."""
    return get_config().redis


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_config().security


def is_production() -> bool:
    """Check if running in production."""
    return get_config().is_production()


def is_development() -> bool:
    """Check if running in development."""
    return get_config().is_development()


# Environment validation
def validate_environment():
    """Validate environment configuration."""
    config = get_config()
    issues = config.validate_configuration()
    
    if issues:
        logger = logging.getLogger(__name__)
        logger.warning("Configuration issues found:")
        for issue in issues:
            logger.warning(f"⚠️ {issue}")
        
        if config.is_production():
            raise RuntimeError("Critical configuration issues in production environment")
    
    return len(issues) == 0