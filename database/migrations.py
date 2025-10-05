"""
Atlas AI Database Migration System
Manages database schema changes with version control and rollback support
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .postgis_database import DatabaseConfig, PostGISDatabase
from ..config.production_settings import get_config


@dataclass
class Migration:
    """Database migration definition."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    checksum: str
    created_at: datetime
    applied_at: Optional[datetime] = None
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'Migration':
        """Create migration from SQL file."""
        content = file_path.read_text(encoding='utf-8')
        
        # Parse migration metadata from comments
        lines = content.split('\n')
        metadata = {}
        up_sql = []
        down_sql = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Parse metadata comments
            if line.startswith('-- @'):
                key, value = line[3:].split(':', 1)
                metadata[key.strip()] = value.strip()
            elif line.startswith('-- UP'):
                current_section = 'up'
                continue
            elif line.startswith('-- DOWN'):
                current_section = 'down'
                continue
            elif line and not line.startswith('--'):
                if current_section == 'up':
                    up_sql.append(line)
                elif current_section == 'down':
                    down_sql.append(line)
        
        up_sql_text = '\n'.join(up_sql)
        down_sql_text = '\n'.join(down_sql)
        checksum = hashlib.md5(up_sql_text.encode()).hexdigest()
        
        return cls(
            version=metadata.get('version', file_path.stem),
            name=metadata.get('name', file_path.stem),
            description=metadata.get('description', ''),
            up_sql=up_sql_text,
            down_sql=down_sql_text,
            checksum=checksum,
            created_at=datetime.now()
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.applied_at:
            data['applied_at'] = self.applied_at.isoformat()
        return data


class MigrationManager:
    """Database migration manager."""
    
    def __init__(self, db: PostGISDatabase, migrations_dir: str = "migrations"):
        self.db = db
        self.migrations_dir = Path(migrations_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(exist_ok=True)
    
    async def initialize_migration_table(self):
        """Create migration tracking table."""
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                checksum VARCHAR(32) NOT NULL,
                up_sql TEXT NOT NULL,
                down_sql TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
                created_at TIMESTAMP NOT NULL,
                execution_time_ms INTEGER
            )
        """)
        
        await self.db.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at 
            ON schema_migrations(applied_at)
        """)
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        result = await self.db.execute_query("""
            SELECT version FROM schema_migrations 
            ORDER BY applied_at ASC
        """)
        return [row['version'] for row in result] if result else []
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied_versions = set(await self.get_applied_migrations())
        all_migrations = self.discover_migrations()
        
        pending = []
        for migration in all_migrations:
            if migration.version not in applied_versions:
                pending.append(migration)
        
        return sorted(pending, key=lambda m: m.version)
    
    def discover_migrations(self) -> List[Migration]:
        """Discover migration files in migrations directory."""
        migrations = []
        
        for file_path in sorted(self.migrations_dir.glob("*.sql")):
            try:
                migration = Migration.from_file(file_path)
                migrations.append(migration)
            except Exception as e:
                self.logger.error(f"Failed to parse migration {file_path}: {e}")
        
        return migrations
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration."""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Applying migration {migration.version}: {migration.name}")
            
            # Check if already applied
            applied_migrations = await self.get_applied_migrations()
            if migration.version in applied_migrations:
                self.logger.warning(f"Migration {migration.version} already applied")
                return True
            
            # Validate checksum if migration exists in database
            existing = await self.db.execute_query(
                "SELECT checksum FROM schema_migrations WHERE version = $1",
                migration.version
            )
            
            if existing and existing[0]['checksum'] != migration.checksum:
                raise Exception(f"Checksum mismatch for migration {migration.version}")
            
            # Execute migration in transaction
            async with self.db.pool.acquire() as conn:
                async with conn.transaction():
                    # Apply the migration SQL
                    await conn.execute(migration.up_sql)
                    
                    # Record migration
                    execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    
                    await conn.execute("""
                        INSERT INTO schema_migrations 
                        (version, name, description, checksum, up_sql, down_sql, 
                         applied_at, created_at, execution_time_ms)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (version) DO UPDATE SET
                            applied_at = EXCLUDED.applied_at,
                            execution_time_ms = EXCLUDED.execution_time_ms
                    """, 
                        migration.version,
                        migration.name,
                        migration.description,
                        migration.checksum,
                        migration.up_sql,
                        migration.down_sql,
                        datetime.now(),
                        migration.created_at,
                        execution_time
                    )
            
            self.logger.info(
                f"✅ Applied migration {migration.version} in {execution_time}ms"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply migration {migration.version}: {e}")
            return False
    
    async def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration."""
        try:
            self.logger.info(f"Rolling back migration {version}")
            
            # Get migration details
            result = await self.db.execute_query(
                "SELECT * FROM schema_migrations WHERE version = $1",
                version
            )
            
            if not result:
                self.logger.error(f"Migration {version} not found")
                return False
            
            migration_data = result[0]
            
            # Execute rollback in transaction
            async with self.db.pool.acquire() as conn:
                async with conn.transaction():
                    # Execute down SQL
                    await conn.execute(migration_data['down_sql'])
                    
                    # Remove migration record
                    await conn.execute(
                        "DELETE FROM schema_migrations WHERE version = $1",
                        version
                    )
            
            self.logger.info(f"✅ Rolled back migration {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to rollback migration {version}: {e}")
            return False
    
    async def migrate_up(self, target_version: Optional[str] = None) -> bool:
        """Apply pending migrations up to target version."""
        try:
            await self.initialize_migration_table()
            
            pending_migrations = await self.get_pending_migrations()
            
            if not pending_migrations:
                self.logger.info("No pending migrations")
                return True
            
            # Filter to target version if specified
            if target_version:
                pending_migrations = [
                    m for m in pending_migrations 
                    if m.version <= target_version
                ]
            
            self.logger.info(f"Applying {len(pending_migrations)} migrations")
            
            for migration in pending_migrations:
                success = await self.apply_migration(migration)
                if not success:
                    self.logger.error("Migration failed, stopping")
                    return False
            
            self.logger.info("✅ All migrations applied successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Migration process failed: {e}")
            return False
    
    async def migrate_down(self, target_version: str) -> bool:
        """Rollback migrations down to target version."""
        try:
            applied_migrations = await self.get_applied_migrations()
            
            # Find migrations to rollback (in reverse order)
            to_rollback = [
                version for version in reversed(applied_migrations)
                if version > target_version
            ]
            
            if not to_rollback:
                self.logger.info("No migrations to rollback")
                return True
            
            self.logger.info(f"Rolling back {len(to_rollback)} migrations")
            
            for version in to_rollback:
                success = await self.rollback_migration(version)
                if not success:
                    self.logger.error("Rollback failed, stopping")
                    return False
            
            self.logger.info("✅ All rollbacks completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Rollback process failed: {e}")
            return False
    
    async def get_migration_status(self) -> Dict:
        """Get current migration status."""
        try:
            await self.initialize_migration_table()
            
            applied_migrations = await self.get_applied_migrations()
            pending_migrations = await self.get_pending_migrations()
            all_migrations = self.discover_migrations()
            
            # Get latest applied migration info
            latest_applied = None
            if applied_migrations:
                result = await self.db.execute_query("""
                    SELECT version, name, applied_at 
                    FROM schema_migrations 
                    ORDER BY applied_at DESC 
                    LIMIT 1
                """)
                if result:
                    latest_applied = result[0]
            
            return {
                'total_migrations': len(all_migrations),
                'applied_count': len(applied_migrations),
                'pending_count': len(pending_migrations),
                'applied_migrations': applied_migrations,
                'pending_migrations': [m.version for m in pending_migrations],
                'latest_applied': latest_applied,
                'database_up_to_date': len(pending_migrations) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get migration status: {e}")
            return {'error': str(e)}
    
    def create_migration(self, name: str, description: str = "") -> Path:
        """Create a new migration file template."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{timestamp}_{name}"
        filename = f"{version}.sql"
        file_path = self.migrations_dir / filename
        
        template = f"""-- @version: {version}
-- @name: {name}
-- @description: {description}
-- 
-- Atlas AI Database Migration
-- Created: {datetime.now().isoformat()}

-- UP
-- Add your migration SQL here



-- DOWN  
-- Add your rollback SQL here


"""
        
        file_path.write_text(template, encoding='utf-8')
        self.logger.info(f"Created migration file: {file_path}")
        
        return file_path
    
    async def validate_migrations(self) -> List[str]:
        """Validate all migration files."""
        issues = []
        
        try:
            migrations = self.discover_migrations()
            versions = set()
            
            for migration in migrations:
                # Check for duplicate versions
                if migration.version in versions:
                    issues.append(f"Duplicate version: {migration.version}")
                versions.add(migration.version)
                
                # Validate SQL syntax (basic check)
                if not migration.up_sql.strip():
                    issues.append(f"Empty UP SQL in {migration.version}")
                
                if not migration.down_sql.strip():
                    issues.append(f"Empty DOWN SQL in {migration.version}")
                
                # Check for dangerous operations
                dangerous_ops = ['DROP DATABASE', 'DROP SCHEMA', 'TRUNCATE']
                for op in dangerous_ops:
                    if op in migration.up_sql.upper():
                        issues.append(f"Dangerous operation '{op}' in {migration.version}")
            
            return issues
            
        except Exception as e:
            return [f"Validation failed: {e}"]


async def main():
    """CLI interface for migration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Atlas AI Database Migration Manager')
    parser.add_argument('command', choices=['status', 'up', 'down', 'create', 'validate'], 
                       help='Migration command')
    parser.add_argument('--target', help='Target migration version')
    parser.add_argument('--name', help='Migration name (for create command)')
    parser.add_argument('--description', default='', help='Migration description')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database and migration manager
    config = get_config()
    db = PostGISDatabase(config.database)
    migration_manager = MigrationManager(db)
    
    try:
        if args.command == 'status':
            status = await migration_manager.get_migration_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.command == 'up':
            success = await migration_manager.migrate_up(args.target)
            exit(0 if success else 1)
            
        elif args.command == 'down':
            if not args.target:
                print("Target version required for down migration")
                exit(1)
            success = await migration_manager.migrate_down(args.target)
            exit(0 if success else 1)
            
        elif args.command == 'create':
            if not args.name:
                print("Migration name required")
                exit(1)
            file_path = migration_manager.create_migration(args.name, args.description)
            print(f"Created migration: {file_path}")
            
        elif args.command == 'validate':
            issues = await migration_manager.validate_migrations()
            if issues:
                print("Migration validation issues:")
                for issue in issues:
                    print(f"  ❌ {issue}")
                exit(1)
            else:
                print("✅ All migrations are valid")
                
    except Exception as e:
        logging.error(f"Command failed: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())