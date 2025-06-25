"""
Enhanced memory_agent_streamlit_fix.py
Streamlit-compatible memory agent with Redis fallback and robust error handling
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis import with fallback
try:
    import redis
    REDIS_AVAILABLE = True
    logger.info("✅ Redis library available")
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis library not available - using SQLite only")

class MemoryBucket(Enum):
    SESSION = "session"
    KNOWLEDGE = "knowledge"
    VECTOR = "vector"
    CACHE = "cache"
    AUDIT = "audit"
    USER_PROFILE = "user_profile"

class MemoryPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TEMPORARY = "temporary"

@dataclass
class MemoryContext:
    """Context for memory operations"""
    user_id: str
    session_id: str
    operation_id: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class RedisManager:
    """Robust Redis connection manager with fallback"""

    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = None
        self.connection_pool = None
        self.last_error = None
        self.available = False

        if REDIS_AVAILABLE:
            self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection with error handling"""
        try:
            redis_config = self.config.get("redis", {})

            # Create connection pool for better performance
            self.connection_pool = redis.ConnectionPool(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                db=redis_config.get("db", 0),
                password=redis_config.get("password"),
                socket_timeout=redis_config.get("socket_timeout", 5),
                socket_connect_timeout=redis_config.get("socket_connect_timeout", 5),
                retry_on_timeout=True,
                max_connections=redis_config.get("max_connections", 10)
            )

            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=True
            )

            # Test connection
            self.redis_client.ping()
            self.available = True
            logger.info("✅ Redis connection established successfully")

        except Exception as e:
            self.last_error = str(e)
            self.available = False
            self.redis_client = None
            logger.warning(f"⚠️ Redis connection failed: {e}")

    def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client if available"""
        if self.available and self.redis_client:
            try:
                # Test connection
                self.redis_client.ping()
                return self.redis_client
            except Exception as e:
                logger.warning(f"Redis connection lost: {e}")
                self.available = False
                return None
        return None

    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self.available and self.redis_client is not None

    def reconnect(self) -> bool:
        """Attempt to reconnect to Redis"""
        if REDIS_AVAILABLE:
            self._initialize_redis()
        return self.available

class HybridMemoryAgent:
    """
    Enhanced hybrid memory management with Redis fallback and Streamlit compatibility
    Features:
    - Redis for fast caching with SQLite fallback
    - Automatic failover between storage methods
    - Streamlit-compatible (no async conflicts)
    - Robust error handling and recovery
    """

    def __init__(self, mcp_client=None, config: Dict = None):
        self.mcp_client = mcp_client
        self.config = config or self._default_config()
        self.logger = logger

        # Initialize Redis manager
        self.redis_manager = RedisManager(self.config)
        self.redis_available = self.redis_manager.is_available()

        # Initialize SQLite as fallback
        self.db_path = self.config.get("db_path", "enhanced_memory.db")
        self._init_sqlite_database()

        # In-memory cache for faster access
        self.memory_cache = {}
        self.cache_timestamps = {}
        self.max_cache_size = self.config.get("max_cache_size", 1000)

        # Statistics tracking
        self.stats = {
            "redis_hits": 0,
            "redis_misses": 0,
            "sqlite_hits": 0,
            "sqlite_misses": 0,
            "cache_hits": 0,
            "total_operations": 0,
            "redis_errors": 0,
            "last_redis_check": datetime.now()
        }

        # Background maintenance
        self._start_maintenance_thread()

        logger.info(f"✅ Enhanced Memory Agent initialized - Redis: {'Available' if self.redis_available else 'Unavailable'}")

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "db_path": "enhanced_memory.db",
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "socket_timeout": 5,
                "max_connections": 10
            },
            "cache_ttl": {
                "session": 3600,      # 1 hour
                "knowledge": 86400,   # 24 hours
                "cache": 1800,        # 30 minutes
                "temporary": 300      # 5 minutes
            },
            "max_cache_size": 1000,
            "maintenance_interval": 300  # 5 minutes
        }

    def _init_sqlite_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        key TEXT PRIMARY KEY,
                        bucket TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        priority TEXT DEFAULT 'medium',
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_memory_bucket ON memory_entries(bucket);
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_entries(user_id);
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries(session_id);
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_memory_expires ON memory_entries(expires_at);
                ''')

                conn.commit()
                logger.info("✅ SQLite database initialized successfully")
        except Exception as e:
            logger.error(f"❌ SQLite initialization failed: {e}")
            raise

    def store_memory(self, bucket: str, data: Dict, context: MemoryContext = None, **kwargs) -> Dict:
        """
        Store data with automatic failover (Streamlit-compatible, no async)
        Priority: Memory Cache -> Redis -> SQLite
        """
        try:
            self.stats["total_operations"] += 1

            # Generate key
            key = self._generate_key(bucket, context, kwargs.get("key_suffix", ""))

            # Determine TTL
            ttl = self._get_ttl(bucket, kwargs.get("ttl"))
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

            # Store in memory cache first
            self._store_in_cache(key, data, expires_at)

            # Try Redis storage
            redis_success = self._store_in_redis(key, data, ttl)

            # Always store in SQLite as backup
            sqlite_success = self._store_in_sqlite(key, bucket, data, context, expires_at, kwargs)

            return {
                "success": True,
                "key": key,
                "storage_methods": {
                    "cache": True,
                    "redis": redis_success,
                    "sqlite": sqlite_success
                },
                "expires_at": expires_at.isoformat() if expires_at else None
            }

        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def retrieve_memory(self, bucket: str = None, context: MemoryContext = None, key: str = None, **kwargs) -> Dict:
        """
        Retrieve data with automatic failover (Streamlit-compatible, no async)
        Priority: Memory Cache -> Redis -> SQLite
        """
        try:
            self.stats["total_operations"] += 1

            # Generate key if not provided
            if not key:
                key = self._generate_key(bucket, context, kwargs.get("key_suffix", ""))

            # Try memory cache first
            cached_data = self._retrieve_from_cache(key)
            if cached_data is not None:
                self.stats["cache_hits"] += 1
                return {
                    "success": True,
                    "data": cached_data,
                    "source": "cache",
                    "key": key
                }

            # Try Redis
            redis_data = self._retrieve_from_redis(key)
            if redis_data is not None:
                self.stats["redis_hits"] += 1
                # Store back in cache
                self._store_in_cache(key, redis_data)
                return {
                    "success": True,
                    "data": redis_data,
                    "source": "redis",
                    "key": key
                }
            else:
                self.stats["redis_misses"] += 1

            # Try SQLite
            sqlite_data = self._retrieve_from_sqlite(key)
            if sqlite_data is not None:
                self.stats["sqlite_hits"] += 1
                # Store back in cache and Redis
                self._store_in_cache(key, sqlite_data)
                if self.redis_available:
                    self._store_in_redis(key, sqlite_data, 3600)  # 1 hour default
                return {
                    "success": True,
                    "data": sqlite_data,
                    "source": "sqlite",
                    "key": key
                }
            else:
                self.stats["sqlite_misses"] += 1

            return {
                "success": False,
                "error": "Data not found",
                "key": key
            }

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_key(self, bucket: str, context: MemoryContext = None, suffix: str = "") -> str:
        """Generate a unique key for storage"""
        if context:
            base_key = f"{bucket}:{context.user_id}:{context.session_id}"
        else:
            base_key = f"{bucket}:default:default"

        if suffix:
            base_key += f":{suffix}"

        return base_key

    def _get_ttl(self, bucket: str, custom_ttl: int = None) -> int:
        """Get TTL for a bucket"""
        if custom_ttl:
            return custom_ttl

        ttl_config = self.config.get("cache_ttl", {})
        return ttl_config.get(bucket, 3600)  # Default 1 hour

    def _store_in_cache(self, key: str, data: Dict, expires_at: datetime = None):
        """Store data in memory cache"""
        try:
            # Clean cache if too large
            if len(self.memory_cache) >= self.max_cache_size:
                self._cleanup_cache()

            self.memory_cache[key] = data
            self.cache_timestamps[key] = {
                "stored_at": datetime.now(),
                "expires_at": expires_at
            }
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def _retrieve_from_cache(self, key: str) -> Optional[Dict]:
        """Retrieve data from memory cache"""
        try:
            if key in self.memory_cache:
                # Check expiration
                timestamp_info = self.cache_timestamps.get(key, {})
                expires_at = timestamp_info.get("expires_at")

                if expires_at and datetime.now() > expires_at:
                    # Expired - remove from cache
                    del self.memory_cache[key]
                    del self.cache_timestamps[key]
                    return None

                return self.memory_cache[key]
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    def _store_in_redis(self, key: str, data: Dict, ttl: int = 3600) -> bool:
        """Store data in Redis"""
        try:
            if not self.redis_available:
                return False

            redis_client = self.redis_manager.get_client()
            if not redis_client:
                self.redis_available = False
                return False

            # Serialize data
            serialized_data = json.dumps(data, default=str)

            # Store with TTL
            if ttl:
                redis_client.setex(key, ttl, serialized_data)
            else:
                redis_client.set(key, serialized_data)

            return True

        except Exception as e:
            logger.warning(f"Redis storage failed: {e}")
            self.stats["redis_errors"] += 1
            self.redis_available = False
            return False

    def _retrieve_from_redis(self, key: str) -> Optional[Dict]:
        """Retrieve data from Redis"""
        try:
            if not self.redis_available:
                return None

            redis_client = self.redis_manager.get_client()
            if not redis_client:
                self.redis_available = False
                return None

            # Retrieve data
            cached_data = redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)

        except Exception as e:
            logger.warning(f"Redis retrieval failed: {e}")
            self.stats["redis_errors"] += 1
            self.redis_available = False

        return None

    def _store_in_sqlite(self, key: str, bucket: str, data: Dict, context: MemoryContext,
                        expires_at: datetime, kwargs: Dict) -> bool:
        """Store data in SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Upsert operation
                cursor.execute('''
                    INSERT OR REPLACE INTO memory_entries 
                    (key, bucket, user_id, session_id, data, expires_at, priority, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    key,
                    bucket,
                    context.user_id if context else None,
                    context.session_id if context else None,
                    json.dumps(data, default=str),
                    expires_at.isoformat() if expires_at else None,
                    kwargs.get("priority", "medium")
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"SQLite storage failed: {e}")
            return False

    def _retrieve_from_sqlite(self, key: str) -> Optional[Dict]:
        """Retrieve data from SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT data, expires_at, access_count FROM memory_entries 
                    WHERE key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ''', (key,))

                row = cursor.fetchone()
                if row:
                    data, expires_at, access_count = row

                    # Update access statistics
                    cursor.execute('''
                        UPDATE memory_entries 
                        SET access_count = ?, last_accessed = CURRENT_TIMESTAMP
                        WHERE key = ?
                    ''', (access_count + 1, key))

                    conn.commit()

                    return json.loads(data)

        except Exception as e:
            logger.error(f"SQLite retrieval failed: {e}")

        return None

    def _cleanup_cache(self):
        """Clean up expired entries from memory cache"""
        try:
            current_time = datetime.now()
            expired_keys = []

            for key, timestamp_info in self.cache_timestamps.items():
                expires_at = timestamp_info.get("expires_at")
                if expires_at and current_time > expires_at:
                    expired_keys.append(key)

            # Remove expired entries
            for key in expired_keys:
                self.memory_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)

            # If still too large, remove oldest entries
            if len(self.memory_cache) >= self.max_cache_size:
                # Sort by stored_at and remove oldest 20%
                sorted_keys = sorted(
                    self.cache_timestamps.keys(),
                    key=lambda k: self.cache_timestamps[k]["stored_at"]
                )

                keys_to_remove = sorted_keys[:int(self.max_cache_size * 0.2)]
                for key in keys_to_remove:
                    self.memory_cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)

        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")

    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        def maintenance_worker():
            while True:
                try:
                    time.sleep(self.config.get("maintenance_interval", 300))

                    # Cleanup expired cache entries
                    self._cleanup_cache()

                    # Try to reconnect to Redis if unavailable
                    if not self.redis_available:
                        self.redis_available = self.redis_manager.reconnect()
                        if self.redis_available:
                            logger.info("✅ Redis connection restored")

                    # Clean up expired SQLite entries
                    self._cleanup_sqlite()

                except Exception as e:
                    logger.warning(f"Maintenance error: {e}")

        # Start daemon thread
        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()

    def _cleanup_sqlite(self):
        """Clean up expired entries from SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM memory_entries 
                    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                ''')
                deleted_count = cursor.rowcount
                conn.commit()

                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired SQLite entries")

        except Exception as e:
            logger.warning(f"SQLite cleanup failed: {e}")

    def get_statistics(self) -> Dict:
        """Get memory agent statistics"""
        return {
            "redis_available": self.redis_available,
            "cache_size": len(self.memory_cache),
            "max_cache_size": self.max_cache_size,
            "stats": self.stats.copy(),
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(self.stats["total_operations"], 1) * 100
            ),
            "redis_hit_rate": (
                self.stats["redis_hits"] / max(self.stats["redis_hits"] + self.stats["redis_misses"], 1) * 100
            ) if self.redis_available else 0,
            "sqlite_hit_rate": (
                self.stats["sqlite_hits"] / max(self.stats["sqlite_hits"] + self.stats["sqlite_misses"], 1) * 100
            )
        }

    def clear_session(self, session_id: str) -> Dict:
        """Clear all data for a specific session"""
        try:
            cleared_count = 0

            # Clear from cache
            keys_to_remove = [k for k in self.memory_cache.keys() if session_id in k]
            for key in keys_to_remove:
                self.memory_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
                cleared_count += 1

            # Clear from Redis
            if self.redis_available:
                try:
                    redis_client = self.redis_manager.get_client()
                    if redis_client:
                        pattern = f"*:{session_id}:*"
                        keys = redis_client.keys(pattern)
                        if keys:
                            redis_client.delete(*keys)
                            cleared_count += len(keys)
                except Exception as e:
                    logger.warning(f"Redis session cleanup failed: {e}")

            # Clear from SQLite
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM memory_entries WHERE session_id = ?', (session_id,))
                    cleared_count += cursor.rowcount
                    conn.commit()
            except Exception as e:
                logger.warning(f"SQLite session cleanup failed: {e}")

            return {
                "success": True,
                "cleared_entries": cleared_count,
                "session_id": session_id
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Factory function for easy creation
def create_streamlit_memory_agent(config: Dict = None) -> HybridMemoryAgent:
    """Create an enhanced Streamlit-compatible memory agent"""
    return HybridMemoryAgent(None, config)

# For compatibility with existing imports
StreamlitMemoryAgent = HybridMemoryAgent

# Export all necessary components
__all__ = [
    'HybridMemoryAgent',
    'StreamlitMemoryAgent',
    'MemoryContext',
    'MemoryBucket',
    'MemoryPriority',
    'create_streamlit_memory_agent',
    'RedisManager'
]