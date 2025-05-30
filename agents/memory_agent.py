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
    SESSION = "session"  # Short-term session memory
    KNOWLEDGE = "knowledge"  # Long-term knowledge patterns
    VECTOR = "vector"  # Semantic vector embeddings
    CACHE = "cache"  # Performance caching
    AUDIT = "audit"  # Audit trail memory
    USER_PROFILE = "user_profile"  # User-specific patterns


class MemoryPriority(Enum):
    CRITICAL = "critical"  # Never expire, highest priority
    HIGH = "high"  # Long retention, high priority
    MEDIUM = "medium"  # Standard retention
    LOW = "low"  # Short retention, can be evicted
    TEMPORARY = "temporary"  # Very short retention


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
    encrypted: bool = False

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

    # Vector data (for semantic search)
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
            # Generate a key from password
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
        try:
            json_data = json.dumps(data, default=str)
            encrypted_data = self.fernet.encrypt(json_data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt_data(self, encrypted_data: str) -> Dict:
        """Decrypt memory data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise


class VectorMemoryStore:
    """Vector storage for semantic memory search"""

    def __init__(self, dimension: int = 384, index_path: str = "memory_index.faiss"):
        self.dimension = dimension
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        self.id_mapping = {}  # Maps FAISS IDs to memory entry IDs
        self.reverse_mapping = {}  # Maps memory entry IDs to FAISS IDs
        self.next_id = 0

        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load existing index if available
        self._load_index()

    def _load_index(self):
        """Load existing FAISS index"""
        try:
            if Path(self.index_path).exists():
                self.index = faiss.read_index(self.index_path)

                # Load mappings
                mapping_path = self.index_path.replace('.faiss', '_mappings.json')
                if Path(mapping_path).exists():
                    with open(mapping_path, 'r') as f:
                        mappings = json.load(f)
                        self.id_mapping = {int(k): v for k, v in mapping.get("id_mapping", {}).items()}
                        s

                        self.reverse_mapping = mappings.get("reverse_mapping", {})
                        self.next_id = mappings.get("next_id", 0)

                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Could not load existing index: {str(e)}")
            self.index = faiss.IndexFlatIP(self.dimension)

    def _save_index(self):
        """Save FAISS index and mappings"""
        try:
            faiss.write_index(self.index, self.index_path)

            # Save mappings
            mapping_path = self.index_path.replace('.faiss', '_mappings.json')
            mappings = {
                "id_mapping": self.id_mapping,
                "reverse_mapping": self.reverse_mapping,
                "next_id": self.next_id
            }
            with open(mapping_path, 'w') as f:
                json.dump(mappings, f)

        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return np.zeros(self.dimension, dtype=np.float32)

    def add_entry(self, entry_id: str, text_content: str) -> bool:
        """Add new entry to vector store"""
        try:
            embedding = self.generate_embedding(text_content)

            faiss_id = self.next_id
            self.index.add(embedding.reshape(1, -1))

            self.id_mapping[faiss_id] = entry_id
            self.reverse_mapping[entry_id] = faiss_id
            self.next_id += 1

            self._save_index()
            return True

        except Exception as e:
            logger.error(f"Failed to add entry to vector store: {str(e)}")
            return False

    def search_similar(self, query_text: str, top_k: int = 10, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Search for similar entries"""
        try:
            if self.index.ntotal == 0:
                return []

            query_embedding = self.generate_embedding(query_text)
            scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                if score >= threshold:
                    entry_id = self.id_mapping.get(idx)
                    if entry_id:
                        results.append((entry_id, float(score)))

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def remove_entry(self, entry_id: str) -> bool:
        """Remove entry from vector store"""
        try:
            faiss_id = self.reverse_mapping.get(entry_id)
            if faiss_id is not None:
                # FAISS doesn't support direct deletion, so we mark as deleted
                del self.id_mapping[faiss_id]
                del self.reverse_mapping[entry_id]
                self._save_index()
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to remove entry from vector store: {str(e)}")
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

            # Memory entries table
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

            # Memory metadata table
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

            # Memory access log
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

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bucket_user ON memory_entries (bucket, user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON memory_entries (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON memory_entries (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires ON memory_entries (expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON memory_entries (content_type)')

            conn.commit()

    def store_entry(self, entry: MemoryEntry) -> bool:
        """Store memory entry in database"""
        try:
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

        except Exception as e:
            logger.error(f"Failed to store memory entry: {str(e)}")
            return False

    def retrieve_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve memory entry by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM memory_entries WHERE entry_id = ?', (entry_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_entry(row)
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve memory entry: {str(e)}")
            return None

    def search_entries(self, bucket: MemoryBucket, user_id: str,
                       filter_criteria: Dict = None, limit: int = 100) -> List[MemoryEntry]:
        """Search memory entries with filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = '''
                    SELECT * FROM memory_entries 
                    WHERE bucket = ? AND user_id = ? AND status = 'active'
                '''
                params = [bucket.value, user_id]

                # Add filter criteria
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

        except Exception as e:
            logger.error(f"Failed to search memory entries: {str(e)}")
            return []

    def update_access(self, entry_id: str):
        """Update access timestamp and count"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE memory_entries 
                    SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE entry_id = ?
                ''', (entry_id,))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to update access: {str(e)}")

    def delete_entry(self, entry_id: str) -> bool:
        """Delete memory entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM memory_entries WHERE entry_id = ?', (entry_id,))
                cursor.execute('DELETE FROM memory_metadata WHERE entry_id = ?', (entry_id,))
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to delete memory entry: {str(e)}")
            return False

    def cleanup_expired(self) -> int:
        """Clean up expired memory entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM memory_entries 
                    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                ''')
                conn.commit()
                return cursor.rowcount

        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {str(e)}")
            return 0

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

    def __init__(self, mcp_client: MCPClient, config: Dict = None):
        self.mcp_client = mcp_client
        self.config = config or self._default_config()

        # Initialize components
        self.encryption = MemoryEncryption(self.config.get("encryption_key"))
        self.database = MemoryDatabase(self.config.get("db_path", "memory_store.db"))
        self.vector_store = VectorMemoryStore(
            dimension=self.config.get("vector_dimension", 384),
            index_path=self.config.get("vector_index_path", "memory_index.faiss")
        )

        # Initialize Redis for caching if available
        self.redis_client = None
        if self.config.get("redis_enabled", False):
            try:
                self.redis_client = redis.Redis(
                    host=self.config.get("redis_host", "localhost"),
                    port=self.config.get("redis_port", 6379),
                    db=self.config.get("redis_db", 0),
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}")

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

    def _default_config(self) -> Dict:
        """Default memory agent configuration"""
        return {
            "retention_policies": {
                "session": {"default_ttl": 3600 * 8},  # 8 hours
                "knowledge": {"default_ttl": 3600 * 24 * 30},  # 30 days
                "cache": {"default_ttl": 3600},  # 1 hour
                "audit": {"default_ttl": 3600 * 24 * 365}  # 1 year
            },
            "vector_similarity_threshold": 0.7,
            "max_memory_entries_per_user": 10000,
            "cleanup_interval": 3600,  # 1 hour
            "enable_encryption": True,
            "redis_enabled": False
        }

    @traceable(name="store_memory")
    async def store_memory(self, bucket: str, data: Dict, context: MemoryContext = None,
                           encrypt_sensitive: bool = False, **kwargs) -> Dict:
        """Store data in specified memory bucket"""

        try:
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

            # Set expiration based on retention policy
            retention_policy = kwargs.get("retention_policy", bucket)
            if retention_policy in self.config["retention_policies"]:
                ttl = self.config["retention_policies"][retention_policy]["default_ttl"]
                entry.expires_at = now + timedelta(seconds=ttl)

            # Encrypt sensitive data if required
            if encrypt_sensitive or self.config.get("enable_encryption", False):
                entry.data = {"encrypted_content": self.encryption.encrypt_data(data)}
                entry.encrypted = True

            # Store in database
            success = self.database.store_entry(entry)
            if not success:
                return {"success": False, "error": "Database storage failed"}

            # Add to vector store for semantic search
            if bucket_enum in [MemoryBucket.KNOWLEDGE, MemoryBucket.SESSION]:
                text_content = self._extract_text_content(data)
                if text_content:
                    self.vector_store.add_entry(entry_id, text_content)

            # Cache in Redis if available
            if self.redis_client:
                try:
                    cache_key = f"memory:{bucket}:{entry_id}"
                    await self.redis_client.setex(
                        cache_key,
                        timedelta(minutes=30),
                        json.dumps(asdict(entry), default=str)
                    )
                except Exception as e:
                    logger.warning(f"Redis caching failed: {str(e)}")

            # Call MCP tool for additional processing
            mcp_result = await self.mcp_client.call_tool("memory_store", {
                "bucket": bucket,
                "entry_id": entry_id,
                "data": data,
                "user_id": entry.user_id,
                "content_type": entry.content_type
            })

            # Update statistics
            self.access_stats["total_stores"] += 1

            # Log memory storage
            logger.info(f"Memory stored: {entry_id} in {bucket} for user {entry.user_id}")

            return {
                "success": True,
                "entry_id": entry_id,
                "content_hash": content_hash,
                "bucket": bucket,
                "encrypted": entry.encrypted,
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None
            }

        except Exception as e:
            logger.error(f"Memory storage failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    @traceable(name="retrieve_memory")
    async def retrieve_memory(self, bucket: str, filter_criteria: Dict = None,
                              context: MemoryContext = None, **kwargs) -> Dict:
        """Retrieve data from specified memory bucket"""

        try:
            bucket_enum = MemoryBucket(bucket)
            user_id = context.user_id if context else kwargs.get("user_id", "system")

            # Try cache first if available
            cached_results = None
            if self.redis_client and filter_criteria:
                cache_key = f"search:{bucket}:{user_id}:{hashlib.md5(str(filter_criteria).encode()).hexdigest()}"
                try:
                    cached_results = await self.redis_client.get(cache_key)
                    if cached_results:
                        self.access_stats["cache_hits"] += 1
                        return json.loads(cached_results)
                except Exception:
                    pass

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
                    try:
                        decrypted_data = self.encryption.decrypt_data(entry.data["encrypted_content"])
                        entry.data = decrypted_data
                    except Exception as e:
                        logger.error(f"Decryption failed for entry {entry.entry_id}: {str(e)}")
                        continue

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

            # Call MCP tool for additional processing
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

            # Cache results if Redis available
            if self.redis_client and filter_criteria:
                try:
                    await self.redis_client.setex(
                        cache_key,
                        timedelta(minutes=10),
                        json.dumps(result, default=str)
                    )
                except Exception:
                    pass

            # Update statistics
            self.access_stats["total_retrievals"] += 1

            logger.info(f"Memory retrieved: {len(retrieved_data)} entries from {bucket} for user {user_id}")

            return result

        except Exception as e:
            logger.error(f"Memory retrieval failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    @traceable(name="prune_memory")
    async def prune_memory(self, bucket: str, policy: Dict, context: MemoryContext = None) -> Dict:
        """Prune memory based on policy"""

        try:
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

            # Count-based pruning (keep only most recent N entries)
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

            # Priority-based pruning (remove low priority entries)
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

            logger.info(f"Memory pruned: {pruned_count} entries from {bucket} for user {user_id}")

            return {
                "success": True,
                "pruned_count": pruned_count,
                "bucket": bucket,
                "policy_applied": policy
            }

        except Exception as e:
            logger.error(f"Memory pruning failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
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

        try:
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

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_text_content(self, data: Dict) -> str:
        """Extract text content for vector embedding"""

        text_parts = []

        def extract_recursive(obj, depth=0):
            if depth > 5:  # Prevent infinite recursion
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
        return " ".join(text_parts[:10])  # Limit to first 10 text parts

    async def _periodic_cleanup(self):
        """Periodic cleanup task"""

        while True:
            try:
                await asyncio.sleep(self.config.get("cleanup_interval", 3600))

                # Clean up expired entries
                expired_count = self.database.cleanup_expired()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired memory entries")

                # Clean up cache if Redis available
                if self.redis_client:
                    try:
                        # Remove old cache entries (implementation depends on Redis setup)
                        pass
                    except Exception:
                        pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task failed: {str(e)}")

    def get_memory_statistics(self) -> Dict:
        """Get memory system statistics"""

        return {
            "access_statistics": self.access_stats,
            "cache_hit_rate": self.access_stats["cache_hits"] / max(1,
                                                                    self.access_stats["cache_hits"] + self.access_stats[
                                                                        "cache_misses"]),
            "vector_store_size": self.vector_store.index.ntotal,
            "redis_available": self.redis_client is not None,
            "encryption_enabled": self.config.get("enable_encryption", False),
            "retention_policies": self.config["retention_policies"]
        }


# Pre/Post Memory Hooks for All Agents
class MemoryHookManager:
    """Manages pre and post memory hooks for all workflow agents"""

    def __init__(self, memory_agent: HybridMemoryAgent):
        self.memory_agent = memory_agent

        # Hook configurations for different agent types
        self.hook_configs = {
            "data_processing_agent": {
                "pre_hooks": ["load_processing_patterns", "load_quality_benchmarks", "load_user_preferences"],
                "post_hooks": ["store_processing_results", "store_quality_patterns", "store_performance_metrics"]
            },
            "dormancy_analysis_agent": {
                "pre_hooks": ["load_dormancy_patterns", "load_historical_insights", "load_compliance_benchmarks"],
                "post_hooks": ["store_dormancy_results", "store_pattern_analysis", "store_compliance_data"]
            },
            "compliance_verification_agent": {
                "pre_hooks": ["load_compliance_patterns", "load_regulatory_updates", "load_violation_history"],
                "post_hooks": ["store_compliance_results", "store_violation_patterns", "store_remediation_actions"]
            },
            "risk_assessment_agent": {
                "pre_hooks": ["load_risk_models", "load_historical_assessments", "load_risk_thresholds"],
                "post_hooks": ["store_risk_results", "store_risk_patterns", "store_mitigation_strategies"]
            },
            "supervisor_agent": {
                "pre_hooks": ["load_decision_patterns", "load_escalation_rules", "load_workflow_history"],
                "post_hooks": ["store_decisions", "store_decision_patterns", "store_workflow_outcomes"]
            },
            "reporting_agent": {
                "pre_hooks": ["load_report_templates", "load_user_preferences", "load_previous_reports"],
                "post_hooks": ["store_report_results", "store_report_patterns", "store_user_feedback"]
            },
            "notification_agent": {
                "pre_hooks": ["load_notification_preferences", "load_delivery_history", "load_channel_performance"],
                "post_hooks": ["store_delivery_results", "store_notification_patterns", "store_channel_analytics"]
            },
            "error_handler_agent": {
                "pre_hooks": ["load_error_patterns", "load_recovery_strategies", "load_escalation_rules"],
                "post_hooks": ["store_error_data", "store_recovery_results", "store_error_patterns"]
            }
        }

    @traceable(name="execute_pre_memory_hooks")
    async def execute_pre_hooks(self, agent_name: str, user_id: str, session_id: str,
                                workflow_context: Dict = None) -> Dict:
        """Execute pre-processing memory hooks for an agent"""

        try:
            context = await self.memory_agent.create_memory_context(
                user_id=user_id,
                session_id=session_id,
                agent_name=agent_name,
                workflow_stage="pre_processing"
            )

            hook_results = {}
            agent_config = self.hook_configs.get(agent_name, {})

            for hook_name in agent_config.get("pre_hooks", []):
                try:
                    result = await self._execute_specific_pre_hook(
                        hook_name, agent_name, context, workflow_context
                    )
                    hook_results[hook_name] = result
                except Exception as e:
                    logger.warning(f"Pre-hook {hook_name} failed for {agent_name}: {str(e)}")
                    hook_results[hook_name] = {"success": False, "error": str(e)}

            return {
                "success": True,
                "agent_name": agent_name,
                "hooks_executed": list(hook_results.keys()),
                "hook_results": hook_results,
                "context_loaded": sum(1 for r in hook_results.values() if r.get("success", False))
            }

        except Exception as e:
            logger.error(f"Pre-hooks execution failed for {agent_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent_name": agent_name
            }

    @traceable(name="execute_post_memory_hooks")
    async def execute_post_hooks(self, agent_name: str, user_id: str, session_id: str,
                                 agent_results: Dict, workflow_context: Dict = None) -> Dict:
        """Execute post-processing memory hooks for an agent"""

        try:
            context = await self.memory_agent.create_memory_context(
                user_id=user_id,
                session_id=session_id,
                agent_name=agent_name,
                workflow_stage="post_processing"
            )

            hook_results = {}
            agent_config = self.hook_configs.get(agent_name, {})

            for hook_name in agent_config.get("post_hooks", []):
                try:
                    result = await self._execute_specific_post_hook(
                        hook_name, agent_name, context, agent_results, workflow_context
                    )
                    hook_results[hook_name] = result
                except Exception as e:
                    logger.warning(f"Post-hook {hook_name} failed for {agent_name}: {str(e)}")
                    hook_results[hook_name] = {"success": False, "error": str(e)}

            return {
                "success": True,
                "agent_name": agent_name,
                "hooks_executed": list(hook_results.keys()),
                "hook_results": hook_results,
                "data_stored": sum(1 for r in hook_results.values() if r.get("success", False))
            }

        except Exception as e:
            logger.error(f"Post-hooks execution failed for {agent_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent_name": agent_name
            }

    async def _execute_specific_pre_hook(self, hook_name: str, agent_name: str,
                                         context: MemoryContext, workflow_context: Dict) -> Dict:
        """Execute specific pre-processing hook"""

        if hook_name == "load_processing_patterns":
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "data_processing_patterns",
                    "user_id": context.user_id,
                    "content_type": "processing_optimization"
                },
                context=context
            )

        elif hook_name == "load_quality_benchmarks":
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "quality_benchmarks",
                    "user_id": context.user_id,
                    "content_type": "data_quality"
                },
                context=context
            )

        elif hook_name == "load_dormancy_patterns":
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "dormancy_patterns",
                    "user_id": context.user_id,
                    "content_type": "dormancy_analysis"
                },
                context=context
            )

        elif hook_name == "load_compliance_patterns":
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "compliance_patterns",
                    "user_id": context.user_id,
                    "content_type": "compliance_verification"
                },
                context=context
            )

        elif hook_name == "load_risk_models":
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "risk_models",
                    "user_id": context.user_id,
                    "content_type": "risk_assessment"
                },
                context=context
            )

        elif hook_name == "load_user_preferences":
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.USER_PROFILE.value,
                filter_criteria={
                    "type": "user_preferences",
                    "user_id": context.user_id,
                    "agent_name": agent_name
                },
                context=context
            )

        elif hook_name == "load_notification_preferences":
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.USER_PROFILE.value,
                filter_criteria={
                    "type": "notification_preferences",
                    "user_id": context.user_id
                },
                context=context
            )

        elif hook_name == "load_decision_patterns":
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "decision_patterns",
                    "user_id": context.user_id,
                    "content_type": "supervisor_decisions"
                },
                context=context
            )

        else:
            # Generic pattern loading
            return await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": hook_name.replace("load_", ""),
                    "user_id": context.user_id
                },
                context=context
            )

    async def _execute_specific_post_hook(self, hook_name: str, agent_name: str,
                                          context: MemoryContext, agent_results: Dict,
                                          workflow_context: Dict) -> Dict:
        """Execute specific post-processing hook"""

        if hook_name == "store_processing_results":
            return await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data={
                    "type": "data_processing_results",
                    "agent_name": agent_name,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "results": agent_results,
                    "timestamp": datetime.now().isoformat()
                },
                context=context,
                content_type="processing_results",
                tags=["data_processing", "session_results"]
            )

        elif hook_name == "store_quality_patterns":
            if agent_results.get("quality_score", 0) > 0.8:  # Only store high-quality patterns
                return await self.memory_agent.store_memory(
                    bucket=MemoryBucket.KNOWLEDGE.value,
                    data={
                        "type": "quality_patterns",
                        "user_id": context.user_id,
                        "quality_metrics": agent_results.get("quality_metrics", {}),
                        "successful_patterns": agent_results.get("successful_patterns", {}),
                        "optimization_insights": agent_results.get("optimization_insights", {}),
                        "timestamp": datetime.now().isoformat()
                    },
                    context=context,
                    content_type="quality_optimization",
                    priority=MemoryPriority.HIGH,
                    tags=["quality", "patterns", "optimization"]
                )

        elif hook_name == "store_dormancy_results":
            return await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data={
                    "type": "dormancy_analysis_results",
                    "agent_name": agent_name,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "dormancy_summary": agent_results.get("dormancy_summary", {}),
                    "compliance_breakdown": agent_results.get("compliance_breakdown", {}),
                    "risk_indicators": agent_results.get("risk_indicators", {}),
                    "timestamp": datetime.now().isoformat()
                },
                context=context,
                content_type="dormancy_results",
                tags=["dormancy", "analysis", "session_results"]
            )

        elif hook_name == "store_pattern_analysis":
            if agent_results.get("pattern_analysis"):
                return await self.memory_agent.store_memory(
                    bucket=MemoryBucket.KNOWLEDGE.value,
                    data={
                        "type": "dormancy_patterns",
                        "user_id": context.user_id,
                        "pattern_insights": agent_results["pattern_analysis"].get("insights", []),
                        "seasonal_patterns": agent_results["pattern_analysis"].get("seasonal_patterns", {}),
                        "reactivation_probability": agent_results["pattern_analysis"].get("reactivation_probability",
                                                                                          {}),
                        "effectiveness_metrics": agent_results.get("performance_metrics", {}),
                        "timestamp": datetime.now().isoformat()
                    },
                    context=context,
                    content_type="dormancy_patterns",
                    priority=MemoryPriority.HIGH,
                    tags=["dormancy", "patterns", "analysis"]
                )

        elif hook_name == "store_compliance_results":
            return await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data={
                    "type": "compliance_verification_results",
                    "agent_name": agent_name,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "compliance_status": agent_results.get("compliance_status", {}),
                    "violations_found": agent_results.get("violations", []),
                    "remediation_actions": agent_results.get("remediation_actions", []),
                    "compliance_score": agent_results.get("compliance_score", 0),
                    "timestamp": datetime.now().isoformat()
                },
                context=context,
                content_type="compliance_results",
                tags=["compliance", "verification", "session_results"]
            )

        elif hook_name == "store_risk_results":
            return await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data={
                    "type": "risk_assessment_results",
                    "agent_name": agent_name,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "overall_risk_score": agent_results.get("overall_risk_score", 0),
                    "risk_breakdown": agent_results.get("risk_breakdown", {}),
                    "high_risk_accounts": agent_results.get("high_risk_accounts", []),
                    "mitigation_strategies": agent_results.get("mitigation_strategies", []),
                    "timestamp": datetime.now().isoformat()
                },
                context=context,
                content_type="risk_results",
                tags=["risk", "assessment", "session_results"]
            )

        elif hook_name == "store_decisions":
            return await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data={
                    "type": "supervisor_decisions",
                    "agent_name": agent_name,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "decisions_made": agent_results.get("decisions", []),
                    "escalations": agent_results.get("escalations", []),
                    "routing_decisions": agent_results.get("routing_decisions", {}),
                    "confidence_scores": agent_results.get("confidence_scores", {}),
                    "timestamp": datetime.now().isoformat()
                },
                context=context,
                content_type="supervisor_decisions",
                tags=["supervisor", "decisions", "session_results"]
            )

        elif hook_name == "store_delivery_results":
            return await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data={
                    "type": "notification_delivery_results",
                    "agent_name": agent_name,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "delivery_summary": agent_results.get("delivery_summary", {}),
                    "channel_performance": agent_results.get("channel_performance", {}),
                    "successful_deliveries": agent_results.get("successful_deliveries", 0),
                    "failed_deliveries": agent_results.get("failed_deliveries", 0),
                    "timestamp": datetime.now().isoformat()
                },
                context=context,
                content_type="notification_results",
                tags=["notifications", "delivery", "session_results"]
            )

        else:
            # Generic result storage
            return await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data={
                    "type": f"{agent_name}_results",
                    "agent_name": agent_name,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "results": agent_results,
                    "hook_name": hook_name,
                    "timestamp": datetime.now().isoformat()
                },
                context=context,
                content_type="agent_results",
                tags=[agent_name, "results"]
            )


# Memory Management API for External Access
class MemoryManagementAPI:
    """API interface for memory management operations"""

    def __init__(self, memory_agent: HybridMemoryAgent):
        self.memory_agent = memory_agent

    async def get_user_memory_dashboard(self, user_id: str) -> Dict:
        """Get comprehensive memory dashboard for user"""

        try:
            # Get memory summary
            summary = await self.memory_agent.get_user_memory_summary(user_id)

            # Get recent activity
            recent_activity = []
            for bucket in MemoryBucket:
                recent_entries = await self.memory_agent.retrieve_memory(
                    bucket=bucket.value,
                    filter_criteria={
                        "user_id": user_id,
                        "created_after": (datetime.now() - timedelta(days=7)).isoformat()
                    },
                    max_results=10
                )

                if recent_entries.get("success") and recent_entries.get("data"):
                    recent_activity.extend([
                        {
                            "bucket": bucket.value,
                            "entry_id": entry["entry_id"],
                            "content_type": entry["content_type"],
                            "created_at": entry["created_at"],
                            "access_count": entry["access_count"]
                        }
                        for entry in recent_entries["data"]
                    ])

            # Sort by creation date
            recent_activity.sort(key=lambda x: x["created_at"], reverse=True)

            # Get memory statistics
            stats = self.memory_agent.get_memory_statistics()

            return {
                "success": True,
                "user_id": user_id,
                "memory_summary": summary.get("memory_summary", {}),
                "total_entries": summary.get("total_entries", 0),
                "total_size_bytes": summary.get("total_size_bytes", 0),
                "recent_activity": recent_activity[:20],
                "system_statistics": stats,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }

    async def export_user_memory(self, user_id: str, export_format: str = "json") -> Dict:
        """Export all user memory data"""

        try:
            exported_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "format": export_format,
                "buckets": {}
            }

            total_entries = 0

            for bucket in MemoryBucket:
                bucket_data = await self.memory_agent.retrieve_memory(
                    bucket=bucket.value,
                    filter_criteria={"user_id": user_id},
                    max_results=10000
                )

                if bucket_data.get("success"):
                    exported_data["buckets"][bucket.value] = bucket_data.get("data", [])
                    total_entries += len(bucket_data.get("data", []))

            return {
                "success": True,
                "export_data": exported_data,
                "total_entries_exported": total_entries,
                "export_size_bytes": len(json.dumps(exported_data, default=str))
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }

    async def purge_user_memory(self, user_id: str, confirm_purge: bool = False) -> Dict:
        """Purge all memory data for a user (GDPR compliance)"""

        if not confirm_purge:
            return {
                "success": False,
                "error": "Purge confirmation required",
                "user_id": user_id
            }

        try:
            purged_entries = 0

            for bucket in MemoryBucket:
                # Get all entries for user
                entries = self.memory_agent.database.search_entries(
                    bucket=bucket,
                    user_id=user_id,
                    limit=100000
                )

                # Delete each entry
                for entry in entries:
                    if self.memory_agent.database.delete_entry(entry.entry_id):
                        purged_entries += 1

                        # Remove from vector store
                        self.memory_agent.vector_store.remove_entry(entry.entry_id)

            # Clear Redis cache for user
            if self.memory_agent.redis_client:
                try:
                    # This would need to be implemented based on Redis key patterns
                    pass
                except Exception:
                    pass

            return {
                "success": True,
                "user_id": user_id,
                "purged_entries": purged_entries,
                "purge_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }



