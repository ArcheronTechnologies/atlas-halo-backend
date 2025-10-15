import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
import aiosqlite
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    class_name = Column(String, index=True)
    confidence = Column(Float)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_width = Column(Float)
    bbox_height = Column(Float)

class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    is_active = Column(Boolean, default=False)

class Config(Base):
    __tablename__ = "config"
    
    key = Column(String, primary_key=True)
    value = Column(Text)

class DatabaseManager:
    def __init__(self, db_path: str = "atlas.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    @asynccontextmanager
    async def get_session(self):
        async with aiosqlite.connect(self.db_path) as db:
            yield db
    
    async def log_detection(self, class_name: str, confidence: float, bbox: List[float]) -> bool:
        try:
            async with self.get_session() as db:
                await db.execute("""
                    INSERT INTO detections (timestamp, class_name, confidence, bbox_x, bbox_y, bbox_width, bbox_height)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (datetime.utcnow().isoformat(), class_name, confidence, bbox[0], bbox[1], bbox[2], bbox[3]))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error logging detection: {e}")
            return False
    
    async def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        try:
            async with self.get_session() as db:
                async with db.execute("""
                    SELECT * FROM detections 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching detections: {e}")
            return []
    
    async def add_model(self, name: str, path: str, accuracy: float = 0.0) -> bool:
        try:
            async with self.get_session() as db:
                await db.execute("""
                    INSERT INTO models (name, path, created_at, accuracy, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, path, datetime.utcnow().isoformat(), accuracy, False))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            return False
    
    async def get_models(self) -> List[Dict]:
        try:
            async with self.get_session() as db:
                async with db.execute("SELECT * FROM models ORDER BY created_at DESC") as cursor:
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []
    
    async def set_active_model(self, model_id: int) -> bool:
        try:
            async with self.get_session() as db:
                await db.execute("UPDATE models SET is_active = 0")
                await db.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error setting active model: {e}")
            return False
    
    async def get_active_model(self) -> Optional[Dict]:
        try:
            async with self.get_session() as db:
                async with db.execute("SELECT * FROM models WHERE is_active = 1 LIMIT 1") as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [desc[0] for desc in cursor.description]
                        return dict(zip(columns, row))
            return None
        except Exception as e:
            logger.error(f"Error fetching active model: {e}")
            return None
    
    async def set_config(self, key: str, value: Any) -> bool:
        try:
            async with self.get_session() as db:
                value_str = json.dumps(value) if not isinstance(value, str) else value
                await db.execute("""
                    INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)
                """, (key, value_str))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error setting config: {e}")
            return False
    
    async def get_config(self, key: str, default: Any = None) -> Any:
        try:
            async with self.get_session() as db:
                async with db.execute("SELECT value FROM config WHERE key = ?", (key,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        try:
                            return json.loads(row[0])
                        except json.JSONDecodeError:
                            return row[0]
            return default
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            return default
    
    async def get_detection_stats(self, hours: int = 24) -> Dict:
        try:
            cutoff_time = datetime.utcnow().replace(microsecond=0) - timedelta(hours=hours)
            async with self.get_session() as db:
                async with db.execute("""
                    SELECT class_name, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM detections 
                    WHERE timestamp > ?
                    GROUP BY class_name
                    ORDER BY count DESC
                """, (cutoff_time.isoformat(),)) as cursor:
                    rows = await cursor.fetchall()
                    stats = {}
                    for row in rows:
                        stats[row[0]] = {
                            'count': row[1],
                            'avg_confidence': round(row[2], 3)
                        }
                    return stats
        except Exception as e:
            logger.error(f"Error getting detection stats: {e}")
            return {}

db_manager = DatabaseManager()