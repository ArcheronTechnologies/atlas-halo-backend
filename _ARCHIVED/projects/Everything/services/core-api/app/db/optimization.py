"""
Database Optimization and Performance Utilities

This module provides production-ready database optimization features including
connection pooling, query optimization, indexing strategies, and performance monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import time
from sqlalchemy import text, event, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool
from ..core.config import settings

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Production database optimization manager"""
    
    def __init__(self):
        self.query_stats = {}
        self.slow_queries = []
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'peak_connections': 0,
            'avg_query_time': 0.0
        }
        self.performance_thresholds = {
            'slow_query_threshold': 1.0,  # seconds
            'connection_warning_threshold': 80,  # percentage of pool
            'cache_hit_ratio_warning': 0.8  # below this is concerning
        }
    
    def setup_connection_pool(self, database_url: str) -> Dict[str, Any]:
        """Configure optimized connection pool settings"""
        pool_config = {
            'poolclass': QueuePool,
            'pool_size': 20,  # Base number of connections
            'max_overflow': 30,  # Additional connections under load
            'pool_pre_ping': True,  # Validate connections before use
            'pool_recycle': 3600,  # Recycle connections every hour
            'pool_timeout': 30,  # Timeout waiting for connection
            'echo': False,  # Set to True for SQL debugging
            'echo_pool': False,  # Set to True for pool debugging
            'connect_args': {
                'connect_timeout': 10,
                'command_timeout': 30,
                'server_settings': {
                    'application_name': 'scip_api',
                    'jit': 'off'  # Disable JIT for consistency
                }
            }
        }
        
        # Production-specific optimizations
        if 'postgresql' in database_url:
            pool_config['connect_args']['server_settings'].update({
                'shared_preload_libraries': 'pg_stat_statements',
                'track_activity_query_size': '2048',
                'log_min_duration_statement': '1000'  # Log slow queries
            })
        
        return pool_config
    
    def setup_query_monitoring(self, engine: Engine):
        """Set up query performance monitoring"""
        
        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - context._query_start_time
            
            # Update query statistics
            query_type = statement.strip().split()[0].upper()
            if query_type not in self.query_stats:
                self.query_stats[query_type] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'max_time': 0.0
                }
            
            stats = self.query_stats[query_type]
            stats['count'] += 1
            stats['total_time'] += total
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['max_time'] = max(stats['max_time'], total)
            
            # Track slow queries
            if total > self.performance_thresholds['slow_query_threshold']:
                slow_query = {
                    'statement': statement[:500] + '...' if len(statement) > 500 else statement,
                    'duration': total,
                    'timestamp': datetime.now(timezone.utc),
                    'parameters': str(parameters)[:200] if parameters else None
                }
                self.slow_queries.append(slow_query)
                
                # Keep only last 100 slow queries
                if len(self.slow_queries) > 100:
                    self.slow_queries = self.slow_queries[-100:]
                
                logger.warning(f"Slow query detected: {total:.2f}s - {statement[:100]}...")
    
    async def analyze_table_statistics(self, session: Session) -> Dict[str, Any]:
        """Analyze table statistics and suggest optimizations"""
        try:
            stats = {}
            
            # Get table sizes and row counts
            if 'postgresql' in str(session.bind.url):
                result = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    ORDER BY tablename, attname;
                """))
                
                table_stats = result.fetchall()
                for row in table_stats:
                    table_name = row.tablename
                    if table_name not in stats:
                        stats[table_name] = {'columns': {}}
                    
                    stats[table_name]['columns'][row.attname] = {
                        'n_distinct': row.n_distinct,
                        'correlation': row.correlation
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing table statistics: {e}")
            return {}
    
    async def suggest_indexes(self, session: Session) -> List[Dict[str, str]]:
        """Suggest indexes based on query patterns and table statistics"""
        suggestions = []
        
        try:
            # Analyze slow queries for index opportunities
            for query in self.slow_queries[-50:]:  # Last 50 slow queries
                statement = query['statement'].lower()
                
                # Look for WHERE clauses without indexes
                if 'where' in statement:
                    # Extract table and column patterns
                    # This is a simplified version - production would use proper SQL parsing
                    if 'components' in statement and 'manufacturer_part_number' in statement:
                        suggestions.append({
                            'table': 'components',
                            'columns': ['manufacturer_part_number'],
                            'type': 'btree',
                            'reason': 'Frequent WHERE clause on manufacturer_part_number'
                        })
                    
                    if 'rfqs' in statement and 'customer_id' in statement and 'created_at' in statement:
                        suggestions.append({
                            'table': 'rfqs',
                            'columns': ['customer_id', 'created_at'],
                            'type': 'btree',
                            'reason': 'Composite index for customer queries with date filtering'
                        })
            
            # Remove duplicates
            unique_suggestions = []
            seen = set()
            for suggestion in suggestions:
                key = f"{suggestion['table']}:{','.join(suggestion['columns'])}"
                if key not in seen:
                    seen.add(key)
                    unique_suggestions.append(suggestion)
            
            return unique_suggestions
            
        except Exception as e:
            logger.error(f"Error generating index suggestions: {e}")
            return []
    
    async def optimize_queries(self, session: Session) -> Dict[str, Any]:
        """Run query optimization analysis"""
        optimizations = {
            'vacuum_needed': [],
            'analyze_needed': [],
            'reindex_needed': [],
            'statistics_update': []
        }
        
        try:
            if 'postgresql' in str(session.bind.url):
                # Check for tables needing VACUUM
                result = await session.execute(text("""
                    SELECT schemaname, tablename, n_dead_tup, n_live_tup
                    FROM pg_stat_user_tables
                    WHERE n_dead_tup > n_live_tup * 0.1
                    ORDER BY n_dead_tup DESC;
                """))
                
                for row in result.fetchall():
                    optimizations['vacuum_needed'].append({
                        'table': f"{row.schemaname}.{row.tablename}",
                        'dead_tuples': row.n_dead_tup,
                        'live_tuples': row.n_live_tup,
                        'ratio': row.n_dead_tup / (row.n_live_tup or 1)
                    })
                
                # Check for missing statistics
                result = await session.execute(text("""
                    SELECT schemaname, tablename, last_autoanalyze, last_analyze
                    FROM pg_stat_user_tables
                    WHERE last_analyze IS NULL OR last_analyze < NOW() - INTERVAL '7 days'
                    ORDER BY tablename;
                """))
                
                for row in result.fetchall():
                    optimizations['analyze_needed'].append({
                        'table': f"{row.schemaname}.{row.tablename}",
                        'last_analyze': row.last_analyze,
                        'last_autoanalyze': row.last_autoanalyze
                    })
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing queries: {e}")
            return optimizations
    
    async def get_performance_report(self, session: Session) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query_statistics': self.query_stats,
            'slow_queries': self.slow_queries[-20:],  # Last 20 slow queries
            'connection_stats': self.connection_stats,
            'recommendations': []
        }
        
        try:
            # Add database-specific metrics
            if 'postgresql' in str(session.bind.url):
                # Connection statistics
                result = await session.execute(text("""
                    SELECT state, count(*) 
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                    GROUP BY state;
                """))
                
                conn_states = {row.state: row.count for row in result.fetchall()}
                report['database_connections'] = conn_states
                
                # Cache hit ratio
                result = await session.execute(text("""
                    SELECT 
                        sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
                    FROM pg_statio_user_tables;
                """))
                
                cache_ratio = result.scalar()
                if cache_ratio:
                    report['cache_hit_ratio'] = float(cache_ratio)
                    
                    if cache_ratio < self.performance_thresholds['cache_hit_ratio_warning']:
                        report['recommendations'].append({
                            'type': 'performance',
                            'priority': 'high',
                            'issue': 'Low cache hit ratio',
                            'details': f'Cache hit ratio is {cache_ratio:.2%}, consider increasing shared_buffers',
                            'action': 'Increase PostgreSQL shared_buffers setting'
                        })
            
            # Generate recommendations based on query stats
            for query_type, stats in self.query_stats.items():
                if stats['avg_time'] > 0.5:  # Average query time > 500ms
                    report['recommendations'].append({
                        'type': 'query_optimization',
                        'priority': 'medium',
                        'issue': f'Slow {query_type} queries',
                        'details': f'Average {query_type} query time: {stats["avg_time"]:.2f}s',
                        'action': f'Review and optimize {query_type} queries, consider indexing'
                    })
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return report
    
    async def run_maintenance_tasks(self, session: Session) -> Dict[str, Any]:
        """Run automated maintenance tasks"""
        results = {
            'vacuum_completed': [],
            'analyze_completed': [],
            'errors': []
        }
        
        try:
            optimizations = await self.optimize_queries(session)
            
            # Run VACUUM on tables that need it
            for table_info in optimizations['vacuum_needed'][:5]:  # Limit to 5 tables
                try:
                    table_name = table_info['table']
                    await session.execute(text(f"VACUUM ANALYZE {table_name};"))
                    results['vacuum_completed'].append(table_name)
                    logger.info(f"VACUUM ANALYZE completed for {table_name}")
                except Exception as e:
                    results['errors'].append(f"VACUUM failed for {table_name}: {str(e)}")
            
            # Update statistics for tables that need it
            for table_info in optimizations['analyze_needed'][:10]:  # Limit to 10 tables
                try:
                    table_name = table_info['table']
                    await session.execute(text(f"ANALYZE {table_name};"))
                    results['analyze_completed'].append(table_name)
                    logger.info(f"ANALYZE completed for {table_name}")
                except Exception as e:
                    results['errors'].append(f"ANALYZE failed for {table_name}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running maintenance tasks: {e}")
            results['errors'].append(str(e))
            return results


class QueryOptimizer:
    """Query optimization utilities"""
    
    @staticmethod
    def optimize_pagination_query(base_query, limit: int, offset: int):
        """Optimize pagination queries using cursor-based approach when possible"""
        # For large offsets, cursor-based pagination is more efficient
        if offset > 10000:
            logger.warning(f"Large offset detected ({offset}), consider cursor-based pagination")
        
        return base_query.limit(limit).offset(offset)
    
    @staticmethod
    def add_query_hints(query, hints: List[str]):
        """Add database-specific query hints"""
        # PostgreSQL query hints would be added here
        # This is a placeholder for production implementation
        return query
    
    @staticmethod
    def optimize_join_order(query):
        """Optimize JOIN order based on table sizes and selectivity"""
        # This would analyze the query plan and suggest optimal JOIN order
        # Placeholder for production implementation
        return query


class ConnectionPool:
    """Enhanced connection pool manager"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.optimizer = DatabaseOptimizer()
        self.pool_config = self.optimizer.setup_connection_pool(database_url)
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup"""
        session = None
        try:
            # This would create a session from the configured pool
            session = Session()  # Placeholder - would use actual pool
            yield session
            session.commit()
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            if session:
                session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        health = {
            'status': 'healthy',
            'response_time': 0.0,
            'active_connections': 0,
            'pool_status': 'ok'
        }
        
        start_time = time.time()
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                health['response_time'] = time.time() - start_time
                
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error(f"Database health check failed: {e}")
        
        return health


# Global optimizer instance
db_optimizer = DatabaseOptimizer()


# Utility functions for common optimization tasks
async def analyze_slow_queries(session: Session, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent slow queries for analysis"""
    return db_optimizer.slow_queries[-limit:] if db_optimizer.slow_queries else []


async def get_table_sizes(session: Session) -> Dict[str, Dict[str, Any]]:
    """Get table size information"""
    sizes = {}
    try:
        if 'postgresql' in str(session.bind.url):
            result = await session.execute(text("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
                    pg_total_relation_size(tablename::regclass) as size_bytes
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(tablename::regclass) DESC;
            """))
            
            for row in result.fetchall():
                sizes[row.tablename] = {
                    'size_pretty': row.size,
                    'size_bytes': row.size_bytes
                }
    
    except Exception as e:
        logger.error(f"Error getting table sizes: {e}")
    
    return sizes


async def optimize_database_settings(session: Session) -> Dict[str, str]:
    """Get recommended database settings for production"""
    recommendations = {}
    
    try:
        if 'postgresql' in str(session.bind.url):
            # Get current settings
            result = await session.execute(text("SHOW ALL;"))
            current_settings = {row.name: row.setting for row in result.fetchall()}
            
            # Recommend optimizations based on workload
            memory_gb = 16  # Would be detected from system
            
            recommendations.update({
                'shared_buffers': f'{int(memory_gb * 0.25)}GB',
                'effective_cache_size': f'{int(memory_gb * 0.75)}GB',
                'work_mem': '256MB',
                'maintenance_work_mem': '2GB',
                'checkpoint_completion_target': '0.9',
                'wal_buffers': '64MB',
                'default_statistics_target': '500',
                'random_page_cost': '1.1',
                'effective_io_concurrency': '200'
            })
    
    except Exception as e:
        logger.error(f"Error getting database settings: {e}")
    
    return recommendations