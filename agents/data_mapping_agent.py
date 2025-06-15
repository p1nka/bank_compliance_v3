"""
agents/data_mapping_agent.py - Advanced Data Mapping Agent with BGE Embeddings
Intelligent field mapping for banking compliance data using semantic similarity
Integrated with the existing hybrid memory and MCP framework
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import pickle
from pathlib import Path
import aiofiles
import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import re

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END
from langsmith import traceable, Client as LangSmithClient

# MCP imports
from mcp_client import MCPClient

# Memory agent import
from agents.memory_agent import HybridMemoryAgent, MemoryContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MappingConfidence(Enum):
    HIGH = "high"  # 90-100%
    MEDIUM = "medium"  # 70-89%
    LOW = "low"  # 50-69%
    VERY_LOW = "very_low"  # <50%


class MappingStrategy(Enum):
    AUTOMATIC = "automatic"  # Auto-map high confidence
    MANUAL = "manual"  # User manual mapping
    LLM_ASSISTED = "llm_assisted"  # LLM-based mapping
    HYBRID = "hybrid"  # Combination approach


class MappingStatus(Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    REQUIRES_USER_INPUT = "requires_user_input"
    LLM_PROCESSING = "llm_processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FieldMapping:
    """Individual field mapping result"""

    source_field: str
    target_field: str
    confidence_score: float
    confidence_level: MappingConfidence
    mapping_strategy: MappingStrategy

    # Semantic analysis
    semantic_similarity: float
    embedding_vector: Optional[np.ndarray] = None

    # Additional context
    data_type_match: bool = False
    sample_values: List[str] = None
    business_rules: List[str] = None

    # User decisions
    user_confirmed: Optional[bool] = None
    user_override: Optional[str] = None
    llm_suggestion: Optional[str] = None

    # Metadata
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.sample_values is None:
            self.sample_values = []
        if self.business_rules is None:
            self.business_rules = []


@dataclass
class DataMappingState:
    """Comprehensive state for data mapping workflow"""

    session_id: str
    user_id: str
    mapping_id: str
    timestamp: datetime

    # Input data
    source_schema: Optional[Dict] = None
    target_schema: Optional[Dict] = None
    source_data_sample: Optional[pd.DataFrame] = None
    mapping_config: Optional[Dict] = None

    # Processing results
    field_mappings: List[FieldMapping] = None
    mapping_summary: Optional[Dict] = None
    unmapped_fields: List[str] = None

    # Status tracking
    mapping_status: MappingStatus = MappingStatus.PENDING
    total_fields: int = 0
    high_confidence_mappings: int = 0
    requires_user_input: int = 0
    requires_llm_assistance: int = 0

    # Memory context
    memory_context: Dict = None
    retrieved_patterns: Dict = None
    historical_mappings: Dict = None

    # Performance metrics
    processing_time: float = 0.0
    embedding_time: float = 0.0
    similarity_calculation_time: float = 0.0

    # Audit trail
    mapping_log: List[Dict] = None
    error_log: List[Dict] = None
    user_decisions: List[Dict] = None

    def __post_init__(self):
        if self.field_mappings is None:
            self.field_mappings = []
        if self.unmapped_fields is None:
            self.unmapped_fields = []
        if self.memory_context is None:
            self.memory_context = {}
        if self.retrieved_patterns is None:
            self.retrieved_patterns = {}
        if self.historical_mappings is None:
            self.historical_mappings = {}
        if self.mapping_log is None:
            self.mapping_log = []
        if self.error_log is None:
            self.error_log = []
        if self.user_decisions is None:
            self.user_decisions = []


class BGEEmbeddingManager:
    """BGE (BAAI General Embedding) manager for semantic field analysis"""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize BGE embedding manager
        BGE models are state-of-the-art for semantic similarity tasks
        """
        self.model_name = model_name
        self.model = None
        self.embedding_cache = {}
        self.cache_file = "bge_embedding_cache.pkl"

        # Load embedding cache if exists
        self._load_cache()

    async def initialize_model(self):
        """Initialize the BGE model asynchronously"""
        try:
            logger.info(f"Loading BGE model: {self.model_name}")
            # Use asyncio to prevent blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(self.model_name)
            )
            logger.info("BGE model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load BGE model: {str(e)}")
            # Fallback to a smaller model
            try:
                self.model_name = "BAAI/bge-base-en-v1.5"
                self.model = await loop.run_in_executor(
                    None, lambda: SentenceTransformer(self.model_name)
                )
                logger.info(f"Loaded fallback BGE model: {self.model_name}")
                return True
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {str(e2)}")
                return False

    def _load_cache(self):
        """Load embedding cache from disk"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {str(e)}")
            self.embedding_cache = {}

    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")

    def _create_field_context(self, field_name: str, sample_values: List[str] = None,
                              data_type: str = None) -> str:
        """Create rich context for field embedding"""

        # Start with the field name
        context_parts = [field_name]

        # Add cleaned field name (remove underscores, camelCase, etc.)
        clean_name = re.sub(r'[_-]', ' ', field_name)
        clean_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean_name)
        if clean_name.lower() != field_name.lower():
            context_parts.append(clean_name)

        # Add data type context
        if data_type:
            context_parts.append(f"data type: {data_type}")

        # Add sample values context (limited to avoid noise)
        if sample_values:
            # Take up to 3 unique sample values
            unique_samples = list(set(str(v) for v in sample_values if v is not None))[:3]
            if unique_samples:
                context_parts.append(f"examples: {', '.join(unique_samples)}")

        # Join with clear separators
        return " | ".join(context_parts)

    async def get_field_embedding(self, field_name: str, sample_values: List[str] = None,
                                  data_type: str = None, use_cache: bool = True) -> np.ndarray:
        """Get BGE embedding for a field with context"""

        # Create rich context for the field
        field_context = self._create_field_context(field_name, sample_values, data_type)

        # Check cache first
        cache_key = f"{field_context}|{self.model_name}"
        if use_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        if self.model is None:
            await self.initialize_model()

        try:
            # Generate embedding using BGE model
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.model.encode([field_context])
            )

            embedding_vector = embedding[0]

            # Cache the result
            if use_cache:
                self.embedding_cache[cache_key] = embedding_vector
                # Periodically save cache
                if len(self.embedding_cache) % 50 == 0:
                    self._save_cache()

            return embedding_vector

        except Exception as e:
            logger.error(f"Failed to generate embedding for field '{field_name}': {str(e)}")
            # Return zero vector as fallback
            return np.zeros(1024)  # BGE-large has 1024 dimensions

    async def calculate_semantic_similarity(self, source_embedding: np.ndarray,
                                            target_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Reshape for sklearn if needed
            if source_embedding.ndim == 1:
                source_embedding = source_embedding.reshape(1, -1)
            if target_embedding.ndim == 1:
                target_embedding = target_embedding.reshape(1, -1)

            # Calculate cosine similarity
            similarity = cosine_similarity(source_embedding, target_embedding)[0][0]

            # Ensure the result is in [0, 1] range
            return max(0, min(1, (similarity + 1) / 2))

        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0

    def cleanup(self):
        """Cleanup resources and save cache"""
        self._save_cache()


class LLMAssistantMapper:
    """LLM-assisted mapping for low confidence cases"""

    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key

    async def suggest_mapping(self, source_field: str, target_fields: List[str],
                              source_samples: List[str] = None,
                              target_field_info: Dict = None,
                              business_context: str = None) -> Dict:
        """Use LLM to suggest field mapping"""

        try:
            # Prepare context for LLM
            prompt = self._create_mapping_prompt(
                source_field, target_fields, source_samples,
                target_field_info, business_context
            )

            response = await self._call_openai(prompt)

            # Parse LLM response
            mapping_suggestion = self._parse_llm_response(response, target_fields)

            return {
                "success": True,
                "suggested_mapping": mapping_suggestion,
                "confidence": mapping_suggestion.get("confidence", 0.5),
                "reasoning": mapping_suggestion.get("reasoning", ""),
                "alternative_suggestions": mapping_suggestion.get("alternatives", [])
            }

        except Exception as e:
            logger.error(f"LLM mapping suggestion failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "suggested_mapping": None
            }

    def _create_mapping_prompt(self, source_field: str, target_fields: List[str],
                               source_samples: List[str] = None,
                               target_field_info: Dict = None,
                               business_context: str = None) -> str:
        """Create a comprehensive prompt for LLM mapping"""

        prompt = f"""You are an expert data analyst specializing in banking compliance data mapping.

SOURCE FIELD TO MAP:
- Field Name: "{source_field}"
"""

        if source_samples:
            prompt += f"- Sample Values: {', '.join(str(v) for v in source_samples[:5])}\n"

        prompt += f"""
AVAILABLE TARGET FIELDS:
"""

        for i, target_field in enumerate(target_fields, 1):
            prompt += f"{i}. {target_field}"
            if target_field_info and target_field in target_field_info:
                info = target_field_info[target_field]
                if info.get("description"):
                    prompt += f" - {info['description']}"
                if info.get("data_type"):
                    prompt += f" (Type: {info['data_type']})"
                if info.get("samples"):
                    prompt += f" (Examples: {', '.join(str(v) for v in info['samples'][:3])})"
            prompt += "\n"

        if business_context:
            prompt += f"\nBUSINESS CONTEXT:\n{business_context}\n"

        prompt += """
TASK:
1. Analyze the source field and determine the BEST target field mapping
2. Provide a confidence score (0.0 to 1.0)
3. Explain your reasoning
4. Suggest up to 2 alternative mappings if applicable

Respond in this JSON format:
{
    "best_match": "target_field_name",
    "confidence": 0.85,
    "reasoning": "Detailed explanation of why this mapping makes sense",
    "alternatives": [
        {"field": "alternative_field", "confidence": 0.6, "reason": "Why this could also work"}
    ]
}

Focus on:
- Semantic meaning and business logic
- Data type compatibility
- Sample value patterns
- Banking domain knowledge
- Compliance requirements

Response:"""

        return prompt

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with retry logic"""

        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system",
                     "content": "You are an expert data analyst specializing in banking compliance data mapping. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            # Fallback to a simpler model or local processing
            raise

    def _parse_llm_response(self, response: str, target_fields: List[str]) -> Dict:
        """Parse and validate LLM response"""

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = json.loads(response)

            # Validate the response
            best_match = response_data.get("best_match")
            if best_match not in target_fields:
                # Try to find closest match
                best_match = self._find_closest_field_name(best_match, target_fields)

            return {
                "target_field": best_match,
                "confidence": min(1.0, max(0.0, response_data.get("confidence", 0.5))),
                "reasoning": response_data.get("reasoning", ""),
                "alternatives": response_data.get("alternatives", [])
            }

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return {
                "target_field": target_fields[0] if target_fields else None,
                "confidence": 0.3,
                "reasoning": "Failed to parse LLM response",
                "alternatives": []
            }

    def _find_closest_field_name(self, suggested_name: str, target_fields: List[str]) -> str:
        """Find the closest matching field name"""

        if not suggested_name or not target_fields:
            return target_fields[0] if target_fields else None

        # Simple string similarity check
        suggested_lower = suggested_name.lower()
        best_match = target_fields[0]
        best_score = 0

        for field in target_fields:
            field_lower = field.lower()

            # Check for exact substring match
            if suggested_lower in field_lower or field_lower in suggested_lower:
                return field

            # Check for word overlap
            suggested_words = set(suggested_lower.split('_'))
            field_words = set(field_lower.split('_'))
            overlap = len(suggested_words & field_words)

            if overlap > best_score:
                best_score = overlap
                best_match = field

        return best_match


class DataMappingAgent:
    """Advanced data mapping agent with BGE embeddings and LLM assistance"""

    def __init__(self, memory_agent: HybridMemoryAgent, mcp_client: MCPClient,
                 db_session=None, openai_api_key: str = None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

        # Initialize components
        self.bge_manager = BGEEmbeddingManager()
        self.llm_assistant = LLMAssistantMapper(openai_api_key)

        # Mapping configuration
        self.confidence_thresholds = {
            MappingConfidence.HIGH: 0.90,
            MappingConfidence.MEDIUM: 0.70,
            MappingConfidence.LOW: 0.50
        }

        # Target schema for banking compliance (CBUAE)
        self.target_schema = {
            "Account_ID": {
                "description": "Unique account identifier",
                "data_type": "string",
                "required": True,
                "samples": ["ACC001", "12345678", "ACCT-2024-001"]
            },
            "Account_Type": {
                "description": "Type of account (Current, Saving, Call, Fixed, Term, Investment, safe_deposit_box)",
                "data_type": "string",
                "required": True,
                "samples": ["Current", "Saving", "Fixed"]
            },
            "Current_Balance": {
                "description": "Current account balance in AED",
                "data_type": "float",
                "required": True,
                "samples": [1000.50, 25000.00, 500.75]
            },
            "Date_Last_Cust_Initiated_Activity": {
                "description": "Date of last customer-initiated transaction or activity",
                "data_type": "date",
                "required": True,
                "samples": ["2024-01-15", "2023-06-30", "2022-12-01"]
            },
            "Date_Last_Customer_Communication_Any_Type": {
                "description": "Date of last communication from customer",
                "data_type": "date",
                "required": False,
                "samples": ["2024-02-01", "2023-07-15", "2022-11-20"]
            },
            "Customer_Has_Active_Liability_Account": {
                "description": "Whether customer has active liability accounts",
                "data_type": "boolean",
                "required": False,
                "samples": ["yes", "no", "true", "false"]
            },
            "FTD_Maturity_Date": {
                "description": "Fixed Term Deposit maturity date",
                "data_type": "date",
                "required": False,
                "samples": ["2024-12-31", "2025-06-30", "2023-12-15"]
            },
            "FTD_Auto_Renewal": {
                "description": "Whether FTD has auto-renewal enabled",
                "data_type": "boolean",
                "required": False,
                "samples": ["yes", "no", "true", "false"]
            },
            "Date_Last_FTD_Renewal_Claim_Request": {
                "description": "Date of last FTD renewal or claim request",
                "data_type": "date",
                "required": False,
                "samples": ["2024-01-01", "2023-12-31", "2023-06-15"]
            },
            "Inv_Maturity_Redemption_Date": {
                "description": "Investment maturity or redemption date",
                "data_type": "date",
                "required": False,
                "samples": ["2024-03-31", "2025-01-15", "2023-09-30"]
            },
            "SDB_Charges_Outstanding": {
                "description": "Outstanding charges for Safe Deposit Box",
                "data_type": "string",
                "required": False,
                "samples": ["yes", "no", "500.00"]
            },
            "Date_SDB_Charges_Became_Outstanding": {
                "description": "Date when SDB charges became outstanding",
                "data_type": "date",
                "required": False,
                "samples": ["2023-01-01", "2022-12-31", "2024-01-15"]
            },
            "SDB_Tenant_Communication_Received": {
                "description": "Whether communication received from SDB tenant",
                "data_type": "boolean",
                "required": False,
                "samples": ["yes", "no", "true", "false"]
            },
            "Unclaimed_Item_Trigger_Date": {
                "description": "Date when item became unclaimed",
                "data_type": "date",
                "required": False,
                "samples": ["2023-01-01", "2022-06-15", "2024-02-01"]
            },
            "Unclaimed_Item_Amount": {
                "description": "Amount of unclaimed item",
                "data_type": "float",
                "required": False,
                "samples": [1000.00, 5000.50, 250.75]
            },
            "Expected_Account_Dormant": {
                "description": "Expected dormancy status based on analysis",
                "data_type": "boolean",
                "required": False,
                "samples": ["yes", "no", "true", "false"]
            },
            "Bank_Contact_Attempted_Post_Dormancy_Trigger": {
                "description": "Whether bank attempted contact after dormancy trigger",
                "data_type": "boolean",
                "required": False,
                "samples": ["yes", "no", "true", "false"]
            },
            "Date_Last_Bank_Contact_Attempt": {
                "description": "Date of last bank contact attempt",
                "data_type": "date",
                "required": False,
                "samples": ["2024-01-15", "2023-11-30", "2023-08-01"]
            },
            "Customer_Address_Known": {
                "description": "Whether customer address is known",
                "data_type": "boolean",
                "required": False,
                "samples": ["yes", "no", "true", "false"]
            }
        }

    @traceable(name="data_mapping_pre_hook")
    async def pre_mapping_hook(self, state: DataMappingState) -> DataMappingState:
        """Enhanced pre-mapping memory hook"""

        try:
            # Retrieve historical mapping patterns
            historical_mappings = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": "field_mapping_patterns",
                    "user_id": state.user_id
                }
            )

            if historical_mappings.get("success"):
                state.historical_mappings = historical_mappings.get("data", {})
                logger.info("Retrieved historical mapping patterns from memory")

            # Retrieve user mapping preferences
            user_preferences = await self.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "type": "mapping_preferences",
                    "user_id": state.user_id
                }
            )

            if user_preferences.get("success"):
                state.memory_context["preferences"] = user_preferences.get("data", {})

            # Log pre-hook execution
            state.mapping_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_mapping_hook",
                "action": "memory_retrieval",
                "status": "completed",
                "historical_patterns_loaded": len(state.historical_mappings),
                "preferences_loaded": len(state.memory_context.get("preferences", {}))
            })

        except Exception as e:
            logger.error(f"Pre-mapping hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_mapping_hook",
                "error": str(e)
            })

        return state

    @traceable(name="analyze_source_schema")
    async def analyze_source_schema(self, state: DataMappingState) -> DataMappingState:
        """Analyze source schema and extract field information"""

        try:
            state.mapping_status = MappingStatus.ANALYZING

            source_fields = []

            if state.source_data_sample is not None and not state.source_data_sample.empty:
                # Extract schema from sample data
                for column in state.source_data_sample.columns:
                    # Get sample values (non-null, limited)
                    sample_values = state.source_data_sample[column].dropna().head(5).tolist()

                    # Determine data type
                    data_type = str(state.source_data_sample[column].dtype)

                    source_fields.append({
                        "field_name": column,
                        "data_type": data_type,
                        "sample_values": sample_values,
                        "null_count": state.source_data_sample[column].isnull().sum(),
                        "unique_count": state.source_data_sample[column].nunique()
                    })

            elif state.source_schema:
                # Use provided schema
                for field_name, field_info in state.source_schema.items():
                    source_fields.append({
                        "field_name": field_name,
                        "data_type": field_info.get("data_type", "unknown"),
                        "sample_values": field_info.get("samples", []),
                        "description": field_info.get("description", "")
                    })

            state.source_schema = {field["field_name"]: field for field in source_fields}
            state.total_fields = len(source_fields)

            # Log analysis completion
            state.mapping_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "schema_analysis",
                "action": "source_schema_analyzed",
                "total_fields": state.total_fields,
                "fields": [f["field_name"] for f in source_fields]
            })

        except Exception as e:
            state.mapping_status = MappingStatus.FAILED
            error_msg = str(e)
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "schema_analysis",
                "error": error_msg
            })
            logger.error(f"Schema analysis failed: {error_msg}")

        return state

    @traceable(name="perform_semantic_mapping")
    async def perform_semantic_mapping(self, state: DataMappingState) -> DataMappingState:
        """Perform semantic mapping using BGE embeddings"""

        try:
            start_time = datetime.now()

            # Initialize BGE model
            if not await self.bge_manager.initialize_model():
                raise ValueError("Failed to initialize BGE embedding model")

            # Generate embeddings for target fields
            target_embeddings = {}
            for target_field, target_info in self.target_schema.items():
                embedding = await self.bge_manager.get_field_embedding(
                    target_field,
                    sample_values=target_info.get("samples", []),
                    data_type=target_info.get("data_type")
                )
                target_embeddings[target_field] = embedding

            state.embedding_time = (datetime.now() - start_time).total_seconds()

            # Process each source field
            similarity_start = datetime.now()

            for source_field_name, source_field_info in state.source_schema.items():
                # Generate embedding for source field
                source_embedding = await self.bge_manager.get_field_embedding(
                    source_field_name,
                    sample_values=source_field_info.get("sample_values", []),
                    data_type=source_field_info.get("data_type")
                )

                # Find best matching target field
                best_match = None
                best_similarity = 0.0

                for target_field, target_embedding in target_embeddings.items():
                    similarity = await self.bge_manager.calculate_semantic_similarity(
                        source_embedding, target_embedding
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = target_field

                # Determine confidence level and strategy
                confidence_level = self._determine_confidence_level(best_similarity)
                mapping_strategy = self._determine_mapping_strategy(confidence_level, state)

                # Check data type compatibility
                data_type_match = self._check_data_type_compatibility(
                    source_field_info.get("data_type", ""),
                    self.target_schema[best_match].get("data_type", "")
                )

                # Create field mapping
                field_mapping = FieldMapping(
                    source_field=source_field_name,
                    target_field=best_match,
                    confidence_score=best_similarity,
                    confidence_level=confidence_level,
                    mapping_strategy=mapping_strategy,
                    semantic_similarity=best_similarity,
                    embedding_vector=source_embedding,
                    data_type_match=data_type_match,
                    sample_values=source_field_info.get("sample_values", []),
                    business_rules=self._get_business_rules(best_match)
                )

                state.field_mappings.append(field_mapping)

                # Update counters
                if confidence_level == MappingConfidence.HIGH:
                    state.high_confidence_mappings += 1
                elif mapping_strategy == MappingStrategy.MANUAL:
                    state.requires_user_input += 1
                elif mapping_strategy == MappingStrategy.LLM_ASSISTED:
                    state.requires_llm_assistance += 1

            state.similarity_calculation_time = (datetime.now() - similarity_start).total_seconds()

            # Generate mapping summary
            state.mapping_summary = self._generate_mapping_summary(state)

            # Determine next steps
            if state.requires_user_input > 0 or state.requires_llm_assistance > 0:
                state.mapping_status = MappingStatus.REQUIRES_USER_INPUT
            else:
                state.mapping_status = MappingStatus.COMPLETED

            # Log mapping completion
            state.mapping_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "semantic_mapping",
                "action": "mapping_completed",
                "total_mappings": len(state.field_mappings),
                "high_confidence": state.high_confidence_mappings,
                "requires_user_input": state.requires_user_input,
                "requires_llm": state.requires_llm_assistance,
                "embedding_time": state.embedding_time,
                "similarity_time": state.similarity_calculation_time
            })

        except Exception as e:
            state.mapping_status = MappingStatus.FAILED
            error_msg = str(e)
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "semantic_mapping",
                "error": error_msg
            })
            logger.error(f"Semantic mapping failed: {error_msg}")

        return state

    def _determine_confidence_level(self, similarity_score: float) -> MappingConfidence:
        """Determine confidence level based on similarity score"""

        if similarity_score >= self.confidence_thresholds[MappingConfidence.HIGH]:
            return MappingConfidence.HIGH
        elif similarity_score >= self.confidence_thresholds[MappingConfidence.MEDIUM]:
            return MappingConfidence.MEDIUM
        elif similarity_score >= self.confidence_thresholds[MappingConfidence.LOW]:
            return MappingConfidence.LOW
        else:
            return MappingConfidence.VERY_LOW

    def _determine_mapping_strategy(self, confidence_level: MappingConfidence,
                                    state: DataMappingState) -> MappingStrategy:
        """Determine mapping strategy based on confidence and user preferences"""

        user_prefs = state.memory_context.get("preferences", {})

        if confidence_level == MappingConfidence.HIGH:
            return MappingStrategy.AUTOMATIC
        elif confidence_level == MappingConfidence.MEDIUM:
            # Check user preference for medium confidence
            if user_prefs.get("auto_map_medium_confidence", False):
                return MappingStrategy.AUTOMATIC
            else:
                return MappingStrategy.MANUAL
        elif confidence_level == MappingConfidence.LOW:
            # Offer LLM assistance for low confidence
            if user_prefs.get("use_llm_for_low_confidence", True):
                return MappingStrategy.LLM_ASSISTED
            else:
                return MappingStrategy.MANUAL
        else:  # VERY_LOW
            return MappingStrategy.MANUAL

    def _check_data_type_compatibility(self, source_type: str, target_type: str) -> bool:
        """Check if source and target data types are compatible"""

        # Data type compatibility mapping
        compatibility_map = {
            "object": ["string", "text", "varchar"],
            "string": ["object", "text", "varchar"],
            "int64": ["integer", "int", "number", "float"],
            "float64": ["float", "number", "decimal", "int"],
            "datetime64": ["date", "datetime", "timestamp"],
            "bool": ["boolean", "bool", "binary"],
            "category": ["string", "categorical"]
        }

        source_lower = source_type.lower()
        target_lower = target_type.lower()

        # Direct match
        if source_lower == target_lower:
            return True

        # Check compatibility map
        for source_pattern, compatible_targets in compatibility_map.items():
            if source_pattern in source_lower:
                return any(target in target_lower for target in compatible_targets)

        return False

    def _get_business_rules(self, target_field: str) -> List[str]:
        """Get business rules for target field"""

        business_rules = {
            "Account_ID": [
                "Must be unique across all accounts",
                "Required field for all account records"
            ],
            "Account_Type": [
                "Must be one of: Current, Saving, Call, Fixed, Term, Investment, safe_deposit_box",
                "Used for dormancy classification rules"
            ],
            "Current_Balance": [
                "Must be non-negative number",
                "Used for high-value account identification"
            ],
            "Date_Last_Cust_Initiated_Activity": [
                "Must be valid date format",
                "Critical for dormancy period calculation"
            ],
            "Customer_Has_Active_Liability_Account": [
                "Boolean field: yes/no or true/false",
                "Affects dormancy classification per Article 2.1.1"
            ]
        }

        return business_rules.get(target_field, [])

    def _generate_mapping_summary(self, state: DataMappingState) -> Dict:
        """Generate comprehensive mapping summary"""

        summary = {
            "total_fields": state.total_fields,
            "mapped_fields": len(state.field_mappings),
            "mapping_rate": len(state.field_mappings) / max(state.total_fields, 1),
            "confidence_distribution": {
                "high": state.high_confidence_mappings,
                "medium": len([m for m in state.field_mappings if m.confidence_level == MappingConfidence.MEDIUM]),
                "low": len([m for m in state.field_mappings if m.confidence_level == MappingConfidence.LOW]),
                "very_low": len([m for m in state.field_mappings if m.confidence_level == MappingConfidence.VERY_LOW])
            },
            "strategy_distribution": {
                "automatic": len([m for m in state.field_mappings if m.mapping_strategy == MappingStrategy.AUTOMATIC]),
                "manual": state.requires_user_input,
                "llm_assisted": state.requires_llm_assistance
            },
            "data_type_compatibility": {
                "compatible": len([m for m in state.field_mappings if m.data_type_match]),
                "incompatible": len([m for m in state.field_mappings if not m.data_type_match])
            },
            "average_confidence": sum(m.confidence_score for m in state.field_mappings) / max(len(state.field_mappings),
                                                                                              1),
            "next_steps": self._determine_next_steps(state)
        }

        return summary

    def _determine_next_steps(self, state: DataMappingState) -> List[str]:
        """Determine next steps based on mapping results"""

        next_steps = []

        if state.high_confidence_mappings > 0:
            next_steps.append(f"Automatically map {state.high_confidence_mappings} high-confidence fields")

        if state.requires_user_input > 0:
            next_steps.append(f"Review {state.requires_user_input} fields requiring manual mapping")

        if state.requires_llm_assistance > 0:
            next_steps.append(f"Process {state.requires_llm_assistance} fields with LLM assistance")

        incompatible_types = len([m for m in state.field_mappings if not m.data_type_match])
        if incompatible_types > 0:
            next_steps.append(f"Review {incompatible_types} fields with data type mismatches")

        if not next_steps:
            next_steps.append("All fields mapped successfully - proceed to data transformation")

        return next_steps

    @traceable(name="process_llm_assistance")
    async def process_llm_assistance(self, state: DataMappingState,
                                     field_mapping: FieldMapping) -> FieldMapping:
        """Process LLM assistance for low confidence mappings"""

        try:
            # Get fields requiring LLM assistance
            target_fields = list(self.target_schema.keys())

            # Get business context
            business_context = f"""
Banking Compliance Data Mapping for CBUAE Regulations:
- Target schema follows CBUAE dormancy analysis requirements
- Fields are used for Articles 2.1.1, 2.2, 2.3, 2.4, 2.6, and 8 compliance
- Data will be analyzed for account dormancy detection
- Accuracy is critical for regulatory compliance
"""

            # Call LLM for suggestion
            llm_result = await self.llm_assistant.suggest_mapping(
                source_field=field_mapping.source_field,
                target_fields=target_fields,
                source_samples=field_mapping.sample_values,
                target_field_info=self.target_schema,
                business_context=business_context
            )

            if llm_result.get("success"):
                suggestion = llm_result["suggested_mapping"]

                # Update field mapping with LLM suggestion
                field_mapping.llm_suggestion = suggestion.get("target_field")
                field_mapping.confidence_score = max(
                    field_mapping.confidence_score,
                    suggestion.get("confidence", 0.5)
                )

                # Update confidence level if LLM improved it
                new_confidence_level = self._determine_confidence_level(field_mapping.confidence_score)
                if new_confidence_level != field_mapping.confidence_level:
                    field_mapping.confidence_level = new_confidence_level
                    field_mapping.mapping_strategy = self._determine_mapping_strategy(
                        new_confidence_level, state
                    )

                # Add LLM reasoning to business rules
                reasoning = suggestion.get("reasoning", "")
                if reasoning:
                    field_mapping.business_rules.append(f"LLM Analysis: {reasoning}")

                # Log LLM assistance
                state.mapping_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "llm_assistance",
                    "action": "llm_suggestion_received",
                    "source_field": field_mapping.source_field,
                    "original_target": field_mapping.target_field,
                    "llm_suggested_target": field_mapping.llm_suggestion,
                    "confidence_improvement": suggestion.get("confidence", 0) - field_mapping.semantic_similarity,
                    "reasoning": reasoning
                })

            else:
                # LLM assistance failed
                state.error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "llm_assistance",
                    "source_field": field_mapping.source_field,
                    "error": llm_result.get("error", "LLM assistance failed")
                })

        except Exception as e:
            logger.error(f"LLM assistance failed for field {field_mapping.source_field}: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "llm_assistance",
                "source_field": field_mapping.source_field,
                "error": str(e)
            })

        return field_mapping

    @traceable(name="process_user_decisions")
    async def process_user_decisions(self, state: DataMappingState,
                                     user_decisions: List[Dict]) -> DataMappingState:
        """Process user manual mapping decisions"""

        try:
            for decision in user_decisions:
                source_field = decision.get("source_field")
                target_field = decision.get("target_field")
                user_confirmed = decision.get("confirmed", True)
                user_override = decision.get("override_target")

                # Find the corresponding field mapping
                field_mapping = None
                for mapping in state.field_mappings:
                    if mapping.source_field == source_field:
                        field_mapping = mapping
                        break

                if field_mapping:
                    # Apply user decision
                    field_mapping.user_confirmed = user_confirmed

                    if user_override and user_override in self.target_schema:
                        field_mapping.user_override = user_override
                        field_mapping.target_field = user_override
                        field_mapping.mapping_strategy = MappingStrategy.MANUAL

                        # Recalculate confidence for user override
                        if user_confirmed:
                            field_mapping.confidence_score = 1.0
                            field_mapping.confidence_level = MappingConfidence.HIGH

                    field_mapping.updated_at = datetime.now()

                    # Log user decision
                    state.user_decisions.append({
                        "timestamp": datetime.now().isoformat(),
                        "source_field": source_field,
                        "original_target": target_field,
                        "final_target": field_mapping.target_field,
                        "user_confirmed": user_confirmed,
                        "user_override": user_override,
                        "decision_type": "manual_mapping"
                    })

                    # Update counters
                    if field_mapping.confidence_level == MappingConfidence.HIGH:
                        if source_field not in [m.source_field for m in
                                                state.field_mappings[:state.field_mappings.index(field_mapping)] if
                                                m.confidence_level == MappingConfidence.HIGH]:
                            state.high_confidence_mappings += 1
                            state.requires_user_input = max(0, state.requires_user_input - 1)

            # Update mapping summary
            state.mapping_summary = self._generate_mapping_summary(state)

            # Check if all user inputs are resolved
            remaining_user_inputs = len([m for m in state.field_mappings
                                         if m.mapping_strategy == MappingStrategy.MANUAL and not m.user_confirmed])

            if remaining_user_inputs == 0:
                state.mapping_status = MappingStatus.COMPLETED

            # Log user decision processing
            state.mapping_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "user_decisions",
                "action": "decisions_processed",
                "decisions_count": len(user_decisions),
                "remaining_user_inputs": remaining_user_inputs
            })

        except Exception as e:
            error_msg = str(e)
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "user_decisions",
                "error": error_msg
            })
            logger.error(f"User decision processing failed: {error_msg}")

        return state

    @traceable(name="generate_mapping_results")
    async def generate_mapping_results(self, state: DataMappingState) -> Dict:
        """Generate final mapping results for data transformation"""

        try:
            # Create field mapping dictionary
            field_map = {}
            transformation_rules = {}
            validation_rules = {}

            for mapping in state.field_mappings:
                # Determine final target field
                final_target = mapping.user_override or mapping.llm_suggestion or mapping.target_field

                # Only include confirmed or high-confidence mappings
                if (mapping.user_confirmed or
                        mapping.confidence_level == MappingConfidence.HIGH or
                        (mapping.user_confirmed is None and mapping.confidence_score >= 0.8)):

                    field_map[mapping.source_field] = final_target

                    # Add transformation rules based on data types
                    if not mapping.data_type_match:
                        transformation_rules[mapping.source_field] = self._get_transformation_rule(
                            mapping, self.target_schema[final_target]
                        )

                    # Add validation rules
                    validation_rules[final_target] = {
                        "required": self.target_schema[final_target].get("required", False),
                        "data_type": self.target_schema[final_target].get("data_type"),
                        "business_rules": mapping.business_rules
                    }
                else:
                    # Add to unmapped fields
                    state.unmapped_fields.append(mapping.source_field)

            # Generate comprehensive results
            results = {
                "mapping_id": state.mapping_id,
                "session_id": state.session_id,
                "user_id": state.user_id,
                "generated_at": datetime.now().isoformat(),
                "field_mappings": field_map,
                "transformation_rules": transformation_rules,
                "validation_rules": validation_rules,
                "unmapped_fields": state.unmapped_fields,
                "mapping_summary": state.mapping_summary,
                "confidence_statistics": {
                    "total_fields": state.total_fields,
                    "mapped_fields": len(field_map),
                    "unmapped_fields": len(state.unmapped_fields),
                    "average_confidence": sum(m.confidence_score for m in state.field_mappings) / max(
                        len(state.field_mappings), 1),
                    "high_confidence_count": state.high_confidence_mappings,
                    "user_decisions_count": len([m for m in state.field_mappings if m.user_confirmed is not None])
                },
                "data_transformation_script": self._generate_transformation_script(field_map, transformation_rules),
                "validation_script": self._generate_validation_script(validation_rules),
                "next_steps": [
                    "Apply field mappings to transform source data",
                    "Execute data validation rules",
                    "Proceed with dormancy analysis workflow"
                ]
            }

            return results

        except Exception as e:
            logger.error(f"Failed to generate mapping results: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mapping_id": state.mapping_id
            }

    def _get_transformation_rule(self, mapping: FieldMapping, target_schema: Dict) -> Dict:
        """Generate transformation rule for data type conversion"""

        source_type = mapping.sample_values[0].__class__.__name__ if mapping.sample_values else "str"
        target_type = target_schema.get("data_type", "string")

        transformation_rule = {
            "source_type": source_type,
            "target_type": target_type,
            "transformation": "direct_copy"
        }

        # Define transformation logic based on type conversion
        if target_type == "date" and source_type in ["str", "object"]:
            transformation_rule.update({
                "transformation": "parse_date",
                "date_formats": ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"],
                "default_format": "YYYY-MM-DD"
            })

        elif target_type == "boolean" and source_type in ["str", "object"]:
            transformation_rule.update({
                "transformation": "parse_boolean",
                "true_values": ["yes", "true", "1", "y"],
                "false_values": ["no", "false", "0", "n"],
                "case_sensitive": False
            })

        elif target_type in ["float", "number"] and source_type in ["str", "object"]:
            transformation_rule.update({
                "transformation": "parse_numeric",
                "remove_currency_symbols": True,
                "decimal_separator": "."
            })

        return transformation_rule

    def _generate_transformation_script(self, field_map: Dict, transformation_rules: Dict) -> str:
        """Generate Python script for data transformation"""

        script = """
# Auto-generated data transformation script
import pandas as pd
import numpy as np
from datetime import datetime

def transform_data(source_df):
    \"\"\"Transform source data according to field mappings\"\"\"

    # Create target DataFrame
    target_df = pd.DataFrame()

    # Field mappings
"""

        for source_field, target_field in field_map.items():
            if source_field in transformation_rules:
                rule = transformation_rules[source_field]
                transformation = rule.get("transformation", "direct_copy")

                if transformation == "parse_date":
                    script += f"""
    # Transform {source_field} -> {target_field} (date conversion)
    target_df['{target_field}'] = pd.to_datetime(source_df['{source_field}'], errors='coerce')
"""
                elif transformation == "parse_boolean":
                    script += f"""
    # Transform {source_field} -> {target_field} (boolean conversion)
    target_df['{target_field}'] = source_df['{source_field}'].astype(str).str.lower().map({{
        'yes': True, 'true': True, '1': True, 'y': True,
        'no': False, 'false': False, '0': False, 'n': False
    }})
"""
                elif transformation == "parse_numeric":
                    script += f"""
    # Transform {source_field} -> {target_field} (numeric conversion)
    target_df['{target_field}'] = pd.to_numeric(
        source_df['{source_field}'].astype(str).str.replace('[^0-9.-]', '', regex=True), 
        errors='coerce'
    )
"""
                else:
                    script += f"""
    # Transform {source_field} -> {target_field} (direct copy)
    target_df['{target_field}'] = source_df['{source_field}']
"""
            else:
                script += f"""
    # Transform {source_field} -> {target_field} (direct copy)
    target_df['{target_field}'] = source_df['{source_field}']
"""

        script += """

    return target_df

# Usage example:
# transformed_data = transform_data(your_source_dataframe)
"""

        return script

    def _generate_validation_script(self, validation_rules: Dict) -> str:
        """Generate Python script for data validation"""

        script = """
# Auto-generated data validation script
import pandas as pd
import numpy as np

def validate_data(df):
    \"\"\"Validate transformed data according to business rules\"\"\"

    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

"""

        for field, rules in validation_rules.items():
            if rules.get("required"):
                script += f"""
    # Validate required field: {field}
    if '{field}' not in df.columns:
        validation_results['errors'].append('Required field missing: {field}')
        validation_results['valid'] = False
    elif df['{field}'].isnull().any():
        validation_results['errors'].append('Required field {field} has null values')
        validation_results['valid'] = False
"""

            data_type = rules.get("data_type")
            if data_type == "date":
                script += f"""
    # Validate date format: {field}
    if '{field}' in df.columns:
        try:
            pd.to_datetime(df['{field}'], errors='raise')
        except:
            validation_results['warnings'].append('Invalid date format in field: {field}')
"""
            elif data_type == "boolean":
                script += f"""
    # Validate boolean values: {field}
    if '{field}' in df.columns:
        valid_boolean_values = [True, False, 'yes', 'no', 'true', 'false', 1, 0]
        invalid_values = df['{field}'][~df['{field}'].isin(valid_boolean_values)]
        if not invalid_values.empty:
            validation_results['warnings'].append(f'Invalid boolean values in {field}: {{invalid_values.unique()}}')
"""

        script += """

    return validation_results

# Usage example:
# validation_result = validate_data(your_transformed_dataframe)
# if not validation_result['valid']:
#     print("Validation errors:", validation_result['errors'])
"""

        return script

    @traceable(name="data_mapping_post_hook")
    async def post_mapping_hook(self, state: DataMappingState) -> DataMappingState:
        """Enhanced post-mapping memory hook"""

        try:
            # Store mapping session data
            session_data = {
                "session_id": state.session_id,
                "mapping_id": state.mapping_id,
                "user_id": state.user_id,
                "mapping_results": {
                    "status": state.mapping_status.value,
                    "total_fields": state.total_fields,
                    "high_confidence_mappings": state.high_confidence_mappings,
                    "user_decisions": len(state.user_decisions),
                    "processing_time": state.processing_time,
                    "average_confidence": state.mapping_summary.get("average_confidence",
                                                                    0) if state.mapping_summary else 0
                },
                "performance_metrics": {
                    "embedding_time": state.embedding_time,
                    "similarity_calculation_time": state.similarity_calculation_time,
                    "total_processing_time": state.processing_time
                }
            }

            await self.memory_agent.store_memory(
                bucket="session",
                data=session_data,
                encrypt_sensitive=True
            )

            # Store successful mapping patterns in knowledge memory
            if state.mapping_status == MappingStatus.COMPLETED:
                successful_mappings = {
                    field_map.source_field: {
                        "target_field": field_map.target_field,
                        "confidence_score": field_map.confidence_score,
                        "strategy": field_map.mapping_strategy.value,
                        "data_type_match": field_map.data_type_match,
                        "user_confirmed": field_map.user_confirmed
                    }
                    for field_map in state.field_mappings
                    if field_map.confidence_level == MappingConfidence.HIGH or field_map.user_confirmed
                }

                pattern_data = {
                    "type": "field_mapping_patterns",
                    "user_id": state.user_id,
                    "successful_mappings": successful_mappings,
                    "mapping_statistics": {
                        "total_fields": state.total_fields,
                        "success_rate": len(successful_mappings) / max(state.total_fields, 1),
                        "average_confidence": sum(m["confidence_score"] for m in successful_mappings.values()) / max(
                            len(successful_mappings), 1),
                        "embedding_model": self.bge_manager.model_name
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket="knowledge",
                    data=pattern_data
                )

            # Store user preferences based on decisions
            if state.user_decisions:
                user_preferences = {
                    "type": "mapping_preferences",
                    "user_id": state.user_id,
                    "preferences": {
                        "prefers_manual_review": len(
                            [d for d in state.user_decisions if not d.get("confirmed", True)]) > 0,
                        "accepts_llm_suggestions": len(
                            [m for m in state.field_mappings if m.llm_suggestion and m.user_confirmed]) > 0,
                        "confidence_threshold": min(
                            m.confidence_score for m in state.field_mappings if m.user_confirmed) if any(
                            m.user_confirmed for m in state.field_mappings) else 0.9
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket="session",
                    data=user_preferences
                )

            # Log post-hook completion
            state.mapping_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_mapping_hook",
                "action": "memory_storage",
                "status": "completed",
                "session_data_stored": True,
                "patterns_stored": state.mapping_status == MappingStatus.COMPLETED,
                "preferences_stored": len(state.user_decisions) > 0
            })

        except Exception as e:
            logger.error(f"Post-mapping hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_mapping_hook",
                "error": str(e)
            })

        return state

    @traceable(name="execute_mapping_workflow")
    async def execute_mapping_workflow(self, user_id: str, source_data: Union[pd.DataFrame, Dict],
                                       mapping_options: Dict = None) -> Dict:
        """Execute complete data mapping workflow"""

        try:
            # Initialize mapping state
            mapping_id = secrets.token_hex(16)
            session_id = secrets.token_hex(16)

            state = DataMappingState(
                session_id=session_id,
                user_id=user_id,
                mapping_id=mapping_id,
                timestamp=datetime.now(),
                source_data_sample=source_data if isinstance(source_data, pd.DataFrame) else None,
                source_schema=source_data if isinstance(source_data, dict) else None,
                mapping_config=mapping_options or {}
            )

            start_time = datetime.now()

            # Execute workflow stages
            state = await self.pre_mapping_hook(state)
            state = await self.analyze_source_schema(state)
            state = await self.perform_semantic_mapping(state)

            # Process LLM assistance for low confidence mappings
            for field_mapping in state.field_mappings:
                if field_mapping.mapping_strategy == MappingStrategy.LLM_ASSISTED:
                    field_mapping = await self.process_llm_assistance(state, field_mapping)

            # Update processing time
            state.processing_time = (datetime.now() - start_time).total_seconds()

            # Execute post-hook
            state = await self.post_mapping_hook(state)

            # Generate final results
            mapping_results = await self.generate_mapping_results(state)

            # Prepare response
            response = {
                "success": True,
                "mapping_id": mapping_id,
                "session_id": session_id,
                "status": state.mapping_status.value,
                "processing_time": state.processing_time,
                "field_mappings": [
                    {
                        "source_field": m.source_field,
                        "target_field": m.target_field,
                        "confidence_score": round(m.confidence_score, 4),
                        "confidence_level": m.confidence_level.value,
                        "mapping_strategy": m.mapping_strategy.value,
                        "data_type_match": m.data_type_match,
                        "sample_values": m.sample_values[:3],  # Limit sample values
                        "business_rules": m.business_rules,
                        "llm_suggestion": m.llm_suggestion,
                        "user_confirmed": m.user_confirmed
                    }
                    for m in state.field_mappings
                ],
                "mapping_summary": state.mapping_summary,
                "requires_user_input": state.requires_user_input > 0,
                "requires_llm_assistance": state.requires_llm_assistance > 0,
                "next_steps": state.mapping_summary.get("next_steps", []) if state.mapping_summary else [],
                "transformation_ready": state.mapping_status == MappingStatus.COMPLETED,
                "mapping_results": mapping_results if state.mapping_status == MappingStatus.COMPLETED else None,
                "performance_metrics": {
                    "total_processing_time": state.processing_time,
                    "embedding_time": state.embedding_time,
                    "similarity_calculation_time": state.similarity_calculation_time,
                    "bge_model_used": self.bge_manager.model_name
                },
                "errors": state.error_log if state.error_log else None
            }

            return response

        except Exception as e:
            logger.error(f"Data mapping workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "mapping_id": mapping_id if 'mapping_id' in locals() else None
            }

    async def process_user_mapping_decisions(self, mapping_id: str, user_decisions: List[Dict]) -> Dict:
        """Process user decisions for manual mapping"""

        try:
            # Retrieve mapping state from memory
            mapping_data = await self.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "mapping_id": mapping_id
                }
            )

            if not mapping_data.get("success"):
                return {
                    "success": False,
                    "error": "Mapping session not found"
                }

            # Reconstruct state (simplified - in production you'd store full state)
            # For now, create a minimal state for processing decisions
            state = DataMappingState(
                session_id=mapping_data["data"]["session_id"],
                user_id=mapping_data["data"]["user_id"],
                mapping_id=mapping_id,
                timestamp=datetime.now()
            )

            # Process user decisions
            state = await self.process_user_decisions(state, user_decisions)

            # Generate updated results
            mapping_results = await self.generate_mapping_results(state)

            return {
                "success": True,
                "mapping_id": mapping_id,
                "decisions_processed": len(user_decisions),
                "updated_mappings": mapping_results,
                "status": "completed" if state.mapping_status == MappingStatus.COMPLETED else "requires_more_input"
            }

        except Exception as e:
            logger.error(f"Failed to process user decisions: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mapping_id": mapping_id
            }

    async def apply_data_transformation(self, mapping_id: str, source_data: pd.DataFrame) -> Dict:
        """Apply the finalized mapping to transform source data"""

        try:
            # Retrieve mapping results
            mapping_results = await self.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "mapping_id": mapping_id,
                    "type": "mapping_results"
                }
            )

            if not mapping_results.get("success"):
                return {
                    "success": False,
                    "error": "Mapping results not found"
                }

            field_mappings = mapping_results["data"]["field_mappings"]
            transformation_rules = mapping_results["data"]["transformation_rules"]
            validation_rules = mapping_results["data"]["validation_rules"]

            # Create target DataFrame
            target_df = pd.DataFrame()
            transformation_log = []
            validation_errors = []

            # Apply field mappings
            for source_field, target_field in field_mappings.items():
                if source_field in source_data.columns:
                    if source_field in transformation_rules:
                        # Apply transformation rule
                        rule = transformation_rules[source_field]
                        target_df[target_field] = self._apply_transformation_rule(
                            source_data[source_field], rule
                        )
                        transformation_log.append({
                            "source_field": source_field,
                            "target_field": target_field,
                            "transformation": rule.get("transformation", "direct_copy"),
                            "status": "success"
                        })
                    else:
                        # Direct copy
                        target_df[target_field] = source_data[source_field]
                        transformation_log.append({
                            "source_field": source_field,
                            "target_field": target_field,
                            "transformation": "direct_copy",
                            "status": "success"
                        })
                else:
                    transformation_log.append({
                        "source_field": source_field,
                        "target_field": target_field,
                        "transformation": "skipped",
                        "status": "error",
                        "error": "Source field not found in data"
                    })

            # Validate transformed data
            validation_results = self._validate_transformed_data(target_df, validation_rules)

            # Calculate transformation statistics
            transformation_stats = {
                "source_records": len(source_data),
                "target_records": len(target_df),
                "source_fields": len(source_data.columns),
                "target_fields": len(target_df.columns),
                "successful_transformations": len([log for log in transformation_log if log["status"] == "success"]),
                "failed_transformations": len([log for log in transformation_log if log["status"] == "error"]),
                "validation_errors": len(validation_results.get("errors", [])),
                "validation_warnings": len(validation_results.get("warnings", []))
            }

            return {
                "success": True,
                "mapping_id": mapping_id,
                "transformed_data": target_df.to_dict('records'),
                "transformation_log": transformation_log,
                "validation_results": validation_results,
                "transformation_statistics": transformation_stats,
                "ready_for_dormancy_analysis": validation_results.get("valid", False),
                "next_steps": [
                    "Review validation results",
                    "Proceed to dormancy analysis" if validation_results.get("valid",
                                                                             False) else "Fix validation errors"
                ]
            }

        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mapping_id": mapping_id
            }

    def _apply_transformation_rule(self, source_series: pd.Series, rule: Dict) -> pd.Series:
        """Apply specific transformation rule to a pandas Series"""

        transformation = rule.get("transformation", "direct_copy")

        try:
            if transformation == "parse_date":
                # Date parsing with multiple format support
                result = pd.to_datetime(source_series, errors='coerce')
                return result

            elif transformation == "parse_boolean":
                # Boolean parsing with multiple value support
                true_values = rule.get("true_values", ["yes", "true", "1", "y"])
                false_values = rule.get("false_values", ["no", "false", "0", "n"])
                case_sensitive = rule.get("case_sensitive", False)

                if not case_sensitive:
                    source_lower = source_series.astype(str).str.lower()
                    true_values = [str(v).lower() for v in true_values]
                    false_values = [str(v).lower() for v in false_values]
                    comparison_series = source_lower
                else:
                    comparison_series = source_series.astype(str)

                result = comparison_series.map(lambda x:
                                               True if x in true_values else
                                               False if x in false_values else
                                               None
                                               )
                return result

            elif transformation == "parse_numeric":
                # Numeric parsing with currency symbol removal
                if rule.get("remove_currency_symbols", True):
                    cleaned = source_series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
                else:
                    cleaned = source_series

                result = pd.to_numeric(cleaned, errors='coerce')
                return result

            else:
                # Direct copy
                return source_series.copy()

        except Exception as e:
            logger.error(f"Transformation rule '{transformation}' failed: {str(e)}")
            return source_series.copy()

    def _validate_transformed_data(self, df: pd.DataFrame, validation_rules: Dict) -> Dict:
        """Validate transformed data against business rules"""

        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        try:
            for field, rules in validation_rules.items():
                # Check required fields
                if rules.get("required", False):
                    if field not in df.columns:
                        validation_results["errors"].append(f"Required field missing: {field}")
                        validation_results["valid"] = False
                    elif df[field].isnull().any():
                        null_count = df[field].isnull().sum()
                        validation_results["errors"].append(f"Required field '{field}' has {null_count} null values")
                        validation_results["valid"] = False

                # Check data types
                if field in df.columns:
                    expected_type = rules.get("data_type")
                    if expected_type == "date":
                        try:
                            pd.to_datetime(df[field], errors='raise')
                        except:
                            validation_results["warnings"].append(f"Invalid date format in field: {field}")

                    elif expected_type == "boolean":
                        valid_boolean_values = [True, False, 'yes', 'no', 'true', 'false', 1, 0, None]
                        invalid_mask = ~df[field].isin(valid_boolean_values)
                        if invalid_mask.any():
                            invalid_count = invalid_mask.sum()
                            validation_results["warnings"].append(
                                f"Field '{field}' has {invalid_count} invalid boolean values")

                    elif expected_type in ["float", "number"]:
                        non_numeric_mask = pd.to_numeric(df[field], errors='coerce').isnull() & df[field].notnull()
                        if non_numeric_mask.any():
                            invalid_count = non_numeric_mask.sum()
                            validation_results["warnings"].append(
                                f"Field '{field}' has {invalid_count} non-numeric values")

                # Apply business rules
                business_rules = rules.get("business_rules", [])
                for rule in business_rules:
                    if "Must be unique" in rule and field in df.columns:
                        if df[field].duplicated().any():
                            duplicate_count = df[field].duplicated().sum()
                            validation_results["errors"].append(
                                f"Field '{field}' has {duplicate_count} duplicate values (must be unique)")
                            validation_results["valid"] = False

                    elif "Must be non-negative" in rule and field in df.columns:
                        numeric_values = pd.to_numeric(df[field], errors='coerce')
                        negative_mask = numeric_values < 0
                        if negative_mask.any():
                            negative_count = negative_mask.sum()
                            validation_results["warnings"].append(
                                f"Field '{field}' has {negative_count} negative values")

            return validation_results

        except Exception as e:
            validation_results["errors"].append(f"Validation process failed: {str(e)}")
            validation_results["valid"] = False
            return validation_results

    def cleanup_resources(self):
        """Cleanup BGE model and other resources"""
        try:
            if self.bge_manager:
                self.bge_manager.cleanup()
            logger.info("Data mapping agent resources cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup resources: {str(e)}")

    def get_mapping_statistics(self) -> Dict:
        """Get mapping agent statistics"""

        return {
            "bge_model": self.bge_manager.model_name if self.bge_manager else None,
            "embedding_cache_size": len(self.bge_manager.embedding_cache) if self.bge_manager else 0,
            "target_schema_fields": len(self.target_schema),
            "confidence_thresholds": {k.value: v for k, v in self.confidence_thresholds.items()},
            "supported_transformations": [
                "parse_date",
                "parse_boolean",
                "parse_numeric",
                "direct_copy"
            ],
            "llm_assistant_available": self.llm_assistant.openai_api_key is not None
        }


# ========================= API INTERFACE FUNCTIONS =========================

async def create_mapping_agent(memory_agent: HybridMemoryAgent, mcp_client: MCPClient,
                               openai_api_key: str = None) -> DataMappingAgent:
    """Create and initialize data mapping agent"""

    agent = DataMappingAgent(
        memory_agent=memory_agent,
        mcp_client=mcp_client,
        openai_api_key=openai_api_key
    )

    # Initialize BGE model
    await agent.bge_manager.initialize_model()

    return agent


async def quick_field_mapping(source_fields: List[str], openai_api_key: str = None) -> Dict:
    """Quick field mapping for simple use cases"""

    try:
        # Create minimal memory agent for quick mapping
        from mcp_client import MCPClient
        mcp_client = MCPClient()
        mcp_client.set_mock_mode(True)

        # Create temporary memory agent (simplified)
        memory_agent = type('MockMemoryAgent', (), {
            'retrieve_memory': lambda *args, **kwargs: asyncio.create_task(
                asyncio.coroutine(lambda: {"success": False})()
            ),
            'store_memory': lambda *args, **kwargs: asyncio.create_task(
                asyncio.coroutine(lambda: {"success": True})()
            )
        })()

        # Create mapping agent
        agent = DataMappingAgent(memory_agent, mcp_client, openai_api_key=openai_api_key)

        # Create source schema from field names
        source_schema = {field: {"field_name": field, "data_type": "string"} for field in source_fields}

        # Execute mapping
        result = await agent.execute_mapping_workflow(
            user_id="quick_mapping",
            source_data=source_schema
        )

        # Cleanup
        agent.cleanup_resources()

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ========================= EXAMPLE USAGE =========================

async def main():
    """Example usage of the data mapping agent"""

    print("Banking Compliance Data Mapping Agent with BGE Embeddings")
    print("=" * 70)
    print("Features:")
    print("- BGE (BAAI General Embedding) for semantic field analysis")
    print("- 90%+ confidence threshold for automatic mapping")
    print("- LLM assistance for low confidence mappings")
    print("- Manual user mapping for complex cases")
    print("- Comprehensive data transformation pipeline")
    print("- CBUAE banking compliance schema target")
    print("- Hybrid memory integration for pattern learning")
    print("- Performance optimization with embedding caching")

    # Example source data
    sample_data = pd.DataFrame({
        'acc_id': ['ACC001', 'ACC002', 'ACC003'],
        'account_type': ['Current', 'Savings', 'Fixed'],
        'balance': [1000.50, 25000.00, 50000.00],
        'last_activity': ['2024-01-15', '2023-06-30', '2022-12-01'],
        'customer_contact': ['2024-02-01', '2023-07-15', '2022-11-20']
    })

    print(f"\nExample Source Data Shape: {sample_data.shape}")
    print(f"Source Fields: {list(sample_data.columns)}")
    print(f"Target Schema Fields: 17 CBUAE compliance fields")

    # Quick mapping demonstration
    print("\n" + "=" * 50)
    print("Quick Mapping Demo (without full initialization)")

    quick_result = await quick_field_mapping(list(sample_data.columns))

    if quick_result.get("success"):
        print(f"✓ Quick mapping completed in {quick_result.get('processing_time', 0):.2f}s")
        print(
            f"✓ High confidence mappings: {quick_result.get('mapping_summary', {}).get('confidence_distribution', {}).get('high', 0)}")
        print(f"✓ Requires user input: {quick_result.get('requires_user_input', False)}")

        # Show sample mappings
        mappings = quick_result.get('field_mappings', [])[:3]
        print("\nSample Field Mappings:")
        for mapping in mappings:
            print(f"  {mapping['source_field']} → {mapping['target_field']} "
                  f"(confidence: {mapping['confidence_score']:.3f}, "
                  f"strategy: {mapping['mapping_strategy']})")
    else:
        print(f"✗ Quick mapping failed: {quick_result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())