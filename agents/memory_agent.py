"""
agents/memory_agent.py - Banking Compliance Memory Management System
Full-featured memory agent with ALL dependencies required
"""

import asyncio
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
import aiofiles
from pathlib import Path
import faiss
import st
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END
from langsmith import traceable, Client as LangSmithClient

# MCP imports
from mcp_client import MCPClient
from utils.redis_manager import RedisManager

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


class MemoryEncryption:
    """Handles encryption/decryption of sensitive memory data"""

    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self.key = encryption_key.encode()
        else:
            password = b"banking_compliance_memory_key_2024"
            salt = b"memory_agent_salt"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.key = key

        self.fernet = Fernet(self.key)

    def encrypt_data(self, data: Dict) -> str:
        """Encrypt memory data"""
        json_data = json.dumps(data, default=str)
        encrypted_data = self.fernet.encrypt(json_data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()

    def decrypt_data(self, encrypted_data: str) -> Dict:
        """Decrypt memory data"""
        decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.fernet.decrypt(decoded_data)
        return json.loads(decrypted_data.decode())


class VectorMemoryStore:
    """Vector storage for semantic memory search"""

    def __init__(self, dimension: int = 384, index_path: str = "memory_index.faiss"):
        self.dimension = dimension
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dimension)
        self.id_mapping = {}
        self.reverse_mapping = {}
        self.next_id = 0
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._load_index()

    def _load_index(self):
        """Load existing FAISS index"""
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            mapping_path = self.index_path.replace('.faiss', '_mappings.json')
            if Path(mapping_path).exists():
                with open(mapping_path, 'r') as f:
                    mappings = json.load(f)
                    self.id_mapping = {int(k): v for k, v in mappings.get("id_mapping", {}).items()}
                    self.reverse_mapping = mappings.get("reverse_mapping", {})
                    self.next_id = mappings.get("next_id", 0)

    def _save_index(self):
        """Save FAISS index and mappings"""
        faiss.write_index(self.index, self.index_path)
        mapping_path = self.index_path.replace('.faiss', '_mappings.json')
        mappings = {
            "id_mapping": self.id_mapping,
            "reverse_mapping": self.reverse_mapping,
            "next_id": self.next_id
        }
        with open(mapping_path, 'w') as f:
            json.dump(mappings, f)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        embedding = self.embedding_model.encode([text])
        return embedding[0].astype(np.float32)

    def add_entry(self, entry_id: str, text_content: str) -> bool:
        """Add new entry to vector store"""
        embedding = self.generate_embedding(text_content)
        faiss_id = self.next_id
        self.index.add(embedding.reshape(1, -1))
        self.id_mapping[faiss_id] = entry_id
        self.reverse_mapping[entry_id] = faiss_id
        self.next_id += 1
        self._save_index()
        return True

    def search_similar(self, query_text: str, top_k: int = 10, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Search for similar entries"""
        if self.index.ntotal == 0:
            return []

        query_embedding = self.generate_embedding(query_text)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            if score >= threshold:
                entry_id = self.id_mapping.get(idx)
                if entry_id:
                    results.append((entry_id, float(score)))

        return results

    def remove_entry(self, entry_id: str) -> bool:
        """Remove entry from vector store"""
        faiss_id = self.reverse_mapping.get(entry_id)
        if faiss_id is not None:
            del self.id_mapping[faiss_id]
            del self.reverse_mapping[entry_id]
            self._save_index()
            return True
        return False


class MemoryDatabase:
    """SQLite database for memory storage"""

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

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL,
                    metadata_key TEXT NOT NULL,
                    metadata_value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entry_id) REFERENCES memory_entries (entry_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT,
                    FOREIGN KEY (entry_id) REFERENCES memory_entries (entry_id)
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

    def search_entries(self, bucket: MemoryBucket, user_id: str,
                       filter_criteria: Dict = None, limit: int = 100) -> List[MemoryEntry]:
        """Search memory entries with filters"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = '''
                SELECT * FROM memory_entries 
                WHERE bucket = ? AND user_id = ? AND status = 'active'
            '''
            params = [bucket.value, user_id]

            if filter_criteria:
                if filter_criteria.get("session_id"):
                    query += " AND session_id = ?"
                    params.append(filter_criteria["session_id"])

                if filter_criteria.get("content_type"):
                    query += " AND content_type = ?"
                    params.append(filter_criteria["content_type"])

                if filter_criteria.get("priority"):
                    query += " AND priority = ?"
                    params.append(filter_criteria["priority"])

                if filter_criteria.get("tags"):
                    for tag in filter_criteria["tags"]:
                        query += " AND tags LIKE ?"
                        params.append(f'%"{tag}"%')

                if filter_criteria.get("created_after"):
                    query += " AND created_at > ?"
                    params.append(filter_criteria["created_after"])

            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_entry(row) for row in rows]

    def update_access(self, entry_id: str):
        """Update access timestamp and count"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE memory_entries 
                SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE entry_id = ?
            ''', (entry_id,))
            conn.commit()

    def delete_entry(self, entry_id: str) -> bool:
        """Delete memory entry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memory_entries WHERE entry_id = ?', (entry_id,))
            cursor.execute('DELETE FROM memory_metadata WHERE entry_id = ?', (entry_id,))
            conn.commit()
            return cursor.rowcount > 0

    def cleanup_expired(self) -> int:
        """Clean up expired memory entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM memory_entries 
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
            ''')
            conn.commit()
            return cursor.rowcount

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


# agents/memory_agent.py (Enhanced)
class HybridMemoryAgent:
    """Enhanced hybrid memory management with robust Redis"""

    def __init__(self, mcp_client: MCPClient, config: Dict):
        self.mcp_client = mcp_client
        self.config = config

        # Initialize Redis with error handling
        self.redis_manager = RedisManager(config)
        self.redis_client = self.redis_manager.get_client()

        # Fallback mode if Redis is unavailable
        self.redis_available = self.redis_client is not None

        if not self.redis_available:
            st.warning("⚠️ Redis unavailable - using fallback mode")

    async def store_memory_with_fallback(self, key: str, data: Dict, ttl: int = 3600):
        """Store data with Redis fallback to SQLite"""
        try:
            if self.redis_available and self.redis_client:
                # Try Redis first
                await self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(data, default=str)
                )
                return True
        except Exception as e:
            self.logger.warning(f"Redis storage failed: {e}")
            self.redis_available = False

        # Fallback to SQLite
        return await self._store_in_sqlite(key, data)

    async def retrieve_memory_with_fallback(self, key: str) -> Optional[Dict]:
        """Retrieve data with Redis fallback"""
        try:
            if self.redis_available and self.redis_client:
                # Try Redis first
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    return json.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"Redis retrieval failed: {e}")
            self.redis_available = False

        # Fallback to SQLite
        return await self._retrieve_from_sqlite(key)