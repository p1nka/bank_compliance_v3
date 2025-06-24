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


class HybridMemoryAgent:
    """Advanced hybrid memory management system"""

    def __init__(self, mcp_client: MCPClient, config: Dict):
        self.mcp_client = mcp_client
        self.config = config

        # Initialize components
        self.encryption = MemoryEncryption(self.config.get("encryption_key"))
        self.database = MemoryDatabase(self.config["db_path"])
        self.vector_store = VectorMemoryStore(
            dimension=self.config["vector_dimension"],
            index_path=self.config["vector_index_path"]
        )

        # Initialize Redis
        self.redis_client = redis.Redis(
            host=self.config["redis_host"],
            port=self.config["redis_port"],
            db=self.config["redis_db"],
            decode_responses=True
        )

        # LangSmith client
        self.langsmith_client = LangSmithClient()

        # Memory statistics
        self.access_stats = {
            "total_stores": 0,
            "total_retrievals": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())

    @traceable(name="store_memory")
    async def store_memory(self, bucket: str, data: Dict, context: MemoryContext = None,
                           encrypt_sensitive: bool = False, **kwargs) -> Dict:
        """Store data in specified memory bucket"""
        bucket_enum = MemoryBucket(bucket)

        # Generate entry ID and content hash
        entry_id = secrets.token_hex(16)
        content_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

        # Prepare memory entry
        now = datetime.now()
        entry = MemoryEntry(
            entry_id=entry_id,
            bucket=bucket_enum,
            user_id=context.user_id if context else kwargs.get("user_id", "system"),
            session_id=context.session_id if context else kwargs.get("session_id"),
            data=data,
            content_hash=content_hash,
            created_at=now,
            updated_at=now,
            accessed_at=now,
            priority=kwargs.get("priority", MemoryPriority.MEDIUM),
            content_type=kwargs.get("content_type", "general"),
            tags=kwargs.get("tags", [])
        )

        # Set expiration
        retention_policy = kwargs.get("retention_policy", bucket)
        if retention_policy in self.config["retention_policies"]:
            ttl = self.config["retention_policies"][retention_policy]["default_ttl"]
            entry.expires_at = now + timedelta(seconds=ttl)

        # Encrypt data if required
        if encrypt_sensitive:
            entry.data = {"encrypted_content": self.encryption.encrypt_data(data)}
            entry.encrypted = True

        # Store in database
        success = self.database.store_entry(entry)
        if not success:
            return {"success": False, "error": "Database storage failed"}

        # Add to vector store
        if bucket_enum in [MemoryBucket.KNOWLEDGE, MemoryBucket.SESSION]:
            text_content = self._extract_text_content(data)
            if text_content:
                self.vector_store.add_entry(entry_id, text_content)

        # Cache in Redis
        cache_key = f"memory:{bucket}:{entry_id}"
        await self.redis_client.setex(
            cache_key,
            timedelta(minutes=30),
            json.dumps(asdict(entry), default=str)
        )

        # Call MCP tool
        mcp_result = await self.mcp_client.call_tool("memory_store", {
            "bucket": bucket,
            "entry_id": entry_id,
            "data": data,
            "user_id": entry.user_id,
            "content_type": entry.content_type
        })

        self.access_stats["total_stores"] += 1

        return {
            "success": True,
            "entry_id": entry_id,
            "content_hash": content_hash,
            "bucket": bucket,
            "encrypted": entry.encrypted,
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None
        }

    @traceable(name="retrieve_memory")
    async def retrieve_memory(self, bucket: str, filter_criteria: Dict = None,
                              context: MemoryContext = None, **kwargs) -> Dict:
        """Retrieve data from specified memory bucket"""
        bucket_enum = MemoryBucket(bucket)
        user_id = context.user_id if context else kwargs.get("user_id", "system")

        # Try cache first
        cached_results = None
        if filter_criteria:
            cache_key = f"search:{bucket}:{user_id}:{hashlib.md5(str(filter_criteria).encode()).hexdigest()}"
            cached_results = await self.redis_client.get(cache_key)
            if cached_results:
                self.access_stats["cache_hits"] += 1
                return json.loads(cached_results)

        self.access_stats["cache_misses"] += 1

        # Search database
        entries = self.database.search_entries(
            bucket=bucket_enum,
            user_id=user_id,
            filter_criteria=filter_criteria or {},
            limit=kwargs.get("max_results", 100)
        )

        # Semantic search if query provided
        if context and context.search_query and bucket_enum in [MemoryBucket.KNOWLEDGE, MemoryBucket.SESSION]:
            similar_entries = self.vector_store.search_similar(
                query_text=context.search_query,
                top_k=context.max_results,
                threshold=context.similarity_threshold
            )

            # Filter entries by similarity results
            similar_ids = {entry_id for entry_id, _ in similar_entries}
            entries = [entry for entry in entries if entry.entry_id in similar_ids]

            # Sort by similarity score
            similarity_scores = {entry_id: score for entry_id, score in similar_entries}
            entries.sort(key=lambda e: similarity_scores.get(e.entry_id, 0), reverse=True)

        # Update access timestamps
        for entry in entries:
            self.database.update_access(entry.entry_id)

        # Decrypt data if needed
        retrieved_data = []
        for entry in entries:
            if entry.encrypted:
                decrypted_data = self.encryption.decrypt_data(entry.data["encrypted_content"])
                entry.data = decrypted_data

            retrieved_data.append({
                "entry_id": entry.entry_id,
                "data": entry.data,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "access_count": entry.access_count,
                "priority": entry.priority.value,
                "content_type": entry.content_type,
                "tags": entry.tags
            })

        # Call MCP tool
        mcp_result = await self.mcp_client.call_tool("memory_retrieve", {
            "bucket": bucket,
            "filter_criteria": filter_criteria or {},
            "user_id": user_id,
            "results_count": len(retrieved_data)
        })

        result = {
            "success": True,
            "data": retrieved_data,
            "total_results": len(retrieved_data),
            "bucket": bucket,
            "search_query": context.search_query if context else None
        }

        # Cache results
        if filter_criteria:
            await self.redis_client.setex(
                cache_key,
                timedelta(minutes=10),
                json.dumps(result, default=str)
            )

        self.access_stats["total_retrievals"] += 1

        return result

    @traceable(name="prune_memory")
    async def prune_memory(self, bucket: str, policy: Dict, context: MemoryContext = None) -> Dict:
        """Prune memory based on policy"""
        bucket_enum = MemoryBucket(bucket)
        user_id = context.user_id if context else policy.get("user_id", "system")

        pruned_count = 0

        # Age-based pruning
        if policy.get("max_age_days"):
            cutoff_date = datetime.now() - timedelta(days=policy["max_age_days"])
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM memory_entries 
                    WHERE bucket = ? AND user_id = ? AND created_at < ?
                ''', (bucket_enum.value, user_id, cutoff_date))
                pruned_count += cursor.rowcount
                conn.commit()

        # Count-based pruning
        if policy.get("max_entries"):
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM memory_entries 
                    WHERE entry_id IN (
                        SELECT entry_id FROM memory_entries 
                        WHERE bucket = ? AND user_id = ?
                        ORDER BY updated_at DESC 
                        LIMIT -1 OFFSET ?
                    )
                ''', (bucket_enum.value, user_id, policy["max_entries"]))
                pruned_count += cursor.rowcount
                conn.commit()

        # Priority-based pruning
        if policy.get("remove_low_priority"):
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM memory_entries 
                    WHERE bucket = ? AND user_id = ? AND priority = 'low'
                ''', (bucket_enum.value, user_id))
                pruned_count += cursor.rowcount
                conn.commit()

        # Clean up expired entries
        expired_count = self.database.cleanup_expired()
        pruned_count += expired_count

        # Call MCP tool
        mcp_result = await self.mcp_client.call_tool("prune_memory", {
            "bucket": bucket,
            "policy": policy,
            "user_id": user_id,
            "pruned_count": pruned_count
        })

        return {
            "success": True,
            "pruned_count": pruned_count,
            "bucket": bucket,
            "policy_applied": policy
        }

    async def create_memory_context(self, user_id: str, session_id: str,
                                    agent_name: str = None, **kwargs) -> MemoryContext:
        """Create memory context for operations"""
        return MemoryContext(
            user_id=user_id,
            session_id=session_id,
            operation_id=secrets.token_hex(8),
            timestamp=datetime.now(),
            agent_name=agent_name,
            **kwargs
        )

    async def get_user_memory_summary(self, user_id: str) -> Dict:
        """Get summary of user's memory usage"""
        summary = {}

        for bucket in MemoryBucket:
            entries = self.database.search_entries(
                bucket=bucket,
                user_id=user_id,
                limit=1000
            )

            summary[bucket.value] = {
                "total_entries": len(entries),
                "total_size_bytes": sum(len(json.dumps(e.data)) for e in entries),
                "last_access": max(e.accessed_at for e in entries) if entries else None,
                "oldest_entry": min(e.created_at for e in entries) if entries else None,
                "newest_entry": max(e.created_at for e in entries) if entries else None
            }

        return {
            "success": True,
            "user_id": user_id,
            "memory_summary": summary,
            "total_entries": sum(s["total_entries"] for s in summary.values()),
            "total_size_bytes": sum(s["total_size_bytes"] for s in summary.values())
        }

    def _extract_text_content(self, data: Dict) -> str:
        """Extract text content for vector embedding"""
        text_parts = []

        def extract_recursive(obj, depth=0):
            if depth > 5:
                return

            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and len(value) > 10:
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item, depth + 1)
            elif isinstance(obj, str) and len(obj) > 10:
                text_parts.append(obj)

        extract_recursive(data)
        return " ".join(text_parts[:10])

    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            await asyncio.sleep(self.config["cleanup_interval"])
            expired_count = self.database.cleanup_expired()
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired memory entries")

    def get_memory_statistics(self) -> Dict:
        """Get memory system statistics"""
        return {
            "access_statistics": self.access_stats,
            "cache_hit_rate": self.access_stats["cache_hits"] / max(1,
                                                                    self.access_stats["cache_hits"] + self.access_stats[
                                                                        "cache_misses"]),
            "vector_store_size": self.vector_store.index.ntotal,
            "redis_available": self.redis_client is not None,
            "retention_policies": self.config["retention_policies"]
        }