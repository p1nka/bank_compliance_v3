"""
agents/memory_agent.py - Banking Compliance Memory Management System
Full-featured memory agent with ALL dependencies required.
Synchronous version, Streamlit-safe.
"""

import json
import logging
import hashlib
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import sqlite3
from pathlib import Path
import base64

# Optional: For vector search, comment out if not used or install faiss and sentence_transformers.
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

class MemoryStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    CORRUPTED = "corrupted"
    PENDING_DELETION = "pending_deletion"

@dataclass
class MemoryContext:
    """Context object for memory operations"""
    user_id: str
    session_id: str
    operation_id: str
    timestamp: datetime

    # Context metadata
    workflow_stage: Optional[str] = None
    agent_name: Optional[str] = None
    memory_scope: Optional[str] = None

    # Retrieval criteria
    filter_criteria: Dict = None
    search_query: Optional[str] = None
    similarity_threshold: float = 0.7
    max_results: int = 10

    # Storage options
    encryption_required: bool = False
    retention_policy: Optional[str] = None
    priority: MemoryPriority = MemoryPriority.MEDIUM
    tags: List[str] = None

    def __post_init__(self):
        if self.filter_criteria is None:
            self.filter_criteria = {}
        if self.tags is None:
            self.tags = []

@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    entry_id: str
    bucket: MemoryBucket
    user_id: str
    session_id: Optional[str]

    # Content
    data: Dict
    content_hash: str
    encrypted: bool

    # Metadata
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    access_count: int = 0

    # Classification
    priority: MemoryPriority = MemoryPriority.MEDIUM
    status: MemoryStatus = MemoryStatus.ACTIVE
    tags: List[str] = None
    content_type: str = "general"

    # Relationships
    related_entries: List[str] = None
    parent_entry: Optional[str] = None

    # Retention
    expires_at: Optional[datetime] = None
    retention_policy: Optional[str] = None

    # Vector data
    embedding: Optional[np.ndarray] = None
    embedding_model: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.related_entries is None:
            self.related_entries = []

# -- MemoryDatabase: SQLite Storage --

class MemoryDatabase:
    """SQLite database for memory storage (sync)"""

    def __init__(self, db_path: str = "memory_store.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize memory database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    bucket TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    data_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    encrypted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    priority TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'active',
                    content_type TEXT DEFAULT 'general',
                    expires_at TIMESTAMP NULL,
                    retention_policy TEXT NULL,
                    tags TEXT NULL,
                    related_entries TEXT NULL,
                    parent_entry TEXT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bucket_user ON memory_entries (bucket, user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON memory_entries (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON memory_entries (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires ON memory_entries (expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON memory_entries (content_type)')
            conn.commit()

    def store_entry(self, entry: MemoryEntry) -> bool:
        """Store memory entry in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO memory_entries (
                    entry_id, bucket, user_id, session_id, data_json, content_hash,
                    encrypted, created_at, updated_at, accessed_at, access_count,
                    priority, status, content_type, expires_at, retention_policy,
                    tags, related_entries, parent_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.entry_id, entry.bucket.value, entry.user_id, entry.session_id,
                json.dumps(entry.data, default=str), entry.content_hash, entry.encrypted,
                entry.created_at, entry.updated_at, entry.accessed_at, entry.access_count,
                entry.priority.value, entry.status.value, entry.content_type,
                entry.expires_at, entry.retention_policy,
                json.dumps(entry.tags), json.dumps(entry.related_entries), entry.parent_entry
            ))
            conn.commit()
            return True

    def retrieve_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve memory entry by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memory_entries WHERE entry_id = ?', (entry_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_entry(row)
            return None

    def _row_to_entry(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry object"""
        return MemoryEntry(
            entry_id=row[0],
            bucket=MemoryBucket(row[1]),
            user_id=row[2],
            session_id=row[3],
            data=json.loads(row[4]),
            content_hash=row[5],
            encrypted=bool(row[6]),
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            accessed_at=datetime.fromisoformat(row[9]),
            access_count=row[10],
            priority=MemoryPriority(row[11]),
            status=MemoryStatus(row[12]),
            content_type=row[13],
            expires_at=datetime.fromisoformat(row[14]) if row[14] else None,
            retention_policy=row[15],
            tags=json.loads(row[16]) if row[16] else [],
            related_entries=json.loads(row[17]) if row[17] else [],
            parent_entry=row[18]
        )

# -- RedisManager: Synchronous Redis connection --

class RedisManager:
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = None
        self.available = False
        if REDIS_AVAILABLE:
            try:
                redis_config = self.config.get("redis", {})
                self.redis_client = redis.Redis(
                    host=redis_config.get("host", "localhost"),
                    port=redis_config.get("port", 6379),
                    db=redis_config.get("db", 0),
                    password=redis_config.get("password"),
                    socket_timeout=redis_config.get("socket_timeout", 5),
                    decode_responses=True
                )
                self.redis_client.ping()
                self.available = True
                logger.info("✅ Redis connection established successfully")
            except Exception as e:
                self.available = False
                self.redis_client = None
                logger.warning(f"⚠️ Redis connection failed: {e}")

    def get_client(self):
        if self.available and self.redis_client:
            try:
                self.redis_client.ping()
                return self.redis_client
            except Exception as e:
                self.available = False
                logger.warning(f"Redis connection lost: {e}")
                return None
        return None

    def is_available(self):
        return self.available and self.redis_client is not None

# -- HybridMemoryAgent: Synchronous version --

class HybridMemoryAgent:
    """Synchronous hybrid memory management with Redis and SQLite fallback."""

    def __init__(self, config: Dict):
        self.config = config or {}
        self.db_path = self.config.get("db_path", "memory_store.db")
        self.memory_db = MemoryDatabase(self.db_path)
        self.redis_manager = RedisManager(self.config)
        self.redis_available = self.redis_manager.is_available()
        self.stats = {
            "redis_hits": 0,
            "redis_misses": 0,
            "sqlite_hits": 0,
            "sqlite_misses": 0,
            "total_operations": 0,
            "redis_errors": 0,
        }
        logger.info(f"HybridMemoryAgent initialized. Redis: {self.redis_available}, SQLite: {self.db_path}")

    def store_memory(self, key: str, data: Dict, ttl: int = 3600) -> bool:
        """Store memory synchronously, Redis preferred, fallback to SQLite."""
        self.stats["total_operations"] += 1
        # Try Redis
        if self.redis_available:
            try:
                redis_client = self.redis_manager.get_client()
                if redis_client:
                    redis_client.setex(key, ttl, json.dumps(data, default=str))
                    return True
            except Exception as e:
                self.stats["redis_errors"] += 1
                self.redis_available = False
                logger.warning(f"Redis storage failed: {e}")
        # Fallback to SQLite as a simple key/value (not full object support)
        try:
            entry = MemoryEntry(
                entry_id=key,
                bucket=MemoryBucket.SESSION,
                user_id="default",
                session_id=None,
                data=data,
                content_hash=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                encrypted=False,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=0
            )
            self.memory_db.store_entry(entry)
            return True
        except Exception as e:
            logger.error(f"SQLite storage failed: {e}")
            return False

    def retrieve_memory(self, key: str) -> Optional[Dict]:
        """Retrieve memory synchronously, Redis preferred, fallback to SQLite."""
        self.stats["total_operations"] += 1
        if self.redis_available:
            try:
                redis_client = self.redis_manager.get_client()
                if redis_client:
                    cached_data = redis_client.get(key)
                    if cached_data:
                        self.stats["redis_hits"] += 1
                        return json.loads(cached_data)
                    else:
                        self.stats["redis_misses"] += 1
            except Exception as e:
                self.stats["redis_errors"] += 1
                self.redis_available = False
                logger.warning(f"Redis retrieval failed: {e}")
        # Fallback to SQLite
        try:
            entry = self.memory_db.retrieve_entry(key)
            if entry:
                self.stats["sqlite_hits"] += 1
                return entry.data
            else:
                self.stats["sqlite_misses"] += 1
        except Exception as e:
            logger.error(f"SQLite retrieval failed: {e}")
        return None

    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        return {
            "redis_available": self.redis_available,
            "sqlite_path": self.db_path,
            "stats": self.stats.copy()
        }

# -- Simple factory for compatibility --
def create_sync_memory_agent(config: Dict = None) -> HybridMemoryAgent:
    return HybridMemoryAgent(config)

# For compatibility with old code (optional)
create_streamlit_memory_agent = create_sync_memory_agent

__all__ = [
    'HybridMemoryAgent',
    'MemoryContext',
    'MemoryBucket',
    'MemoryPriority',
    'MemoryStatus',
    'create_sync_memory_agent',
    'create_streamlit_memory_agent',
]