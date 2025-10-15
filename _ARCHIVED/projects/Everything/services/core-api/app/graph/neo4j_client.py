import os
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of a Neo4j query"""
    success: bool
    data: List[Dict[str, Any]]
    error: Optional[str] = None
    execution_time: Optional[float] = None
    total_records: int = 0

_driver = None


class Neo4jClient:
    """Neo4j client for graph database operations"""
    
    def __init__(self):
        self._driver = None
        self._healthy = False
    
    async def initialize(self):
        """Initialize Neo4j driver"""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not (uri and user and password):
            logger.warning("Neo4j credentials not configured, client disabled")
            return
        
        try:
            from neo4j import GraphDatabase  # type: ignore
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            await self._driver.verify_connectivity()
            self._healthy = True
            logger.info("Neo4j client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j client: {e}")
            self._driver = None
            self._healthy = False
    
    async def close(self):
        """Close Neo4j driver"""
        if self._driver is not None:
            try:
                await self._driver.close()
                logger.info("Neo4j client closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j client: {e}")
            finally:
                self._driver = None
                self._healthy = False
    
    def is_healthy(self) -> bool:
        """Check if Neo4j client is healthy"""
        return self._healthy and self._driver is not None
    
    async def run_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Run a Cypher query and return results"""
        if not self.is_healthy():
            logger.warning("Neo4j client not healthy, skipping query")
            return []
        
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error running Neo4j query: {e}")
            return []


# Global client instance
neo4j_client = Neo4jClient()

# Legacy functions for backward compatibility
async def initialize():
    await neo4j_client.initialize()

async def close():
    await neo4j_client.close()

