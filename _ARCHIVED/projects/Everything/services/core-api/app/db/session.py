from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from ..core.config import settings


# Enhanced connection configuration for better performance
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

# Add connection pooling and timeouts for non-SQLite databases
if not settings.database_url.startswith("sqlite"):
    connect_args.update({
        "connect_timeout": 10,
        "application_name": "scip_api",
    })

engine = create_engine(
    settings.database_url, 
    future=True, 
    connect_args=connect_args,
    poolclass=QueuePool,
    pool_size=10,  # Number of connections to maintain
    max_overflow=20,  # Additional connections under load
    pool_pre_ping=True,  # Validate connections before use
    pool_recycle=3600,  # Recycle connections every hour
    pool_timeout=30,  # Timeout waiting for connection
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with automatic rollback on error"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def get_async_session():
    """Async session getter (placeholder for future async SQLAlchemy support)"""
    # For now, return sync session - can be enhanced with async SQLAlchemy later
    return get_session()
