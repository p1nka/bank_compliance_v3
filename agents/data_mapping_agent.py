"""
agents/data_mapping_agent.py - Advanced Data Mapping Agent with BGE Embeddings
Intelligent field mapping for banking compliance data using semantic similarity
Updated to use LangChain with Llama 3.3 70B Versatile instead of OpenAI
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
import re

# LangChain imports
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain.schema.output_parser import OutputParserException

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


class LlamaCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for Llama model calls"""

    def __init__(self):
        super().__init__()
        self.tokens_used = 0
        self.call_duration = 0

    async def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info("Starting Llama 3.3 70B call for field mapping")
        self.start_time = datetime.now()

    async def on_llm_end(self, response, **kwargs):
        self.call_duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Llama call completed in {self.call_duration:.2f}s")

    async def on_llm_error(self, error, **kwargs):
        logger.error(f"Llama call failed: {str(error)}")


class LlamaAssistantMapper:
    """Llama 3.3 70B-assisted mapping for low confidence cases using LangChain"""

    def __init__(self, groq_api_key: str = None, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize Llama assistant mapper with Groq API

        Args:
            groq_api_key: Groq API key for Llama access
            model_name: Llama model to use (default: llama-3.3-70b-versatile)
        """
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.callback_handler = LlamaCallbackHandler()

        if groq_api_key:
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=0.1,
                max_tokens=800,
                streaming=False,
                callbacks=[self.callback_handler]
            )
        else:
            self.llm = None
            logger.warning("No Groq API key provided - LLM assistance will be unavailable")

    async def suggest_mapping(self, source_field: str, target_fields: List[str],
                              source_samples: List[str] = None,
                              target_field_info: Dict = None,
                              business_context: str = None) -> Dict:
        """Use Llama 3.3 70B to suggest field mapping"""

        if not self.llm:
            return {
                "success": False,
                "error": "Llama model not configured - missing Groq API key",
                "suggested_mapping": None
            }

        try:
            # Prepare messages for LangChain
            system_message = SystemMessage(content=self._create_system_prompt())
            human_message = HumanMessage(content=self._create_mapping_prompt(
                source_field, target_fields, source_samples,
                target_field_info, business_context
            ))

            # Call Llama through LangChain
            response = await self.llm.ainvoke([system_message, human_message])
            response_text = response.content.strip()

            # Parse Llama response
            mapping_suggestion = self._parse_llm_response(response_text, target_fields)

            return {
                "success": True,
                "suggested_mapping": mapping_suggestion,
                "confidence": mapping_suggestion.get("confidence", 0.5),
                "reasoning": mapping_suggestion.get("reasoning", ""),
                "alternative_suggestions": mapping_suggestion.get("alternatives", []),
                "model_used": self.model_name,
                "processing_time": self.callback_handler.call_duration
            }

        except OutputParserException as e:
            logger.error(f"Failed to parse Llama response: {str(e)}")
            return {
                "success": False,
                "error": f"Response parsing failed: {str(e)}",
                "suggested_mapping": None
            }
        except Exception as e:
            logger.error(f"Llama mapping suggestion failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "suggested_mapping": None
            }

    def _create_system_prompt(self) -> str:
        """Create system prompt optimized for Llama 3.3 70B"""
        return """You are an expert data analyst specializing in banking compliance data mapping with deep knowledge of:

- Banking and financial services domain
- UAE regulatory requirements (CBUAE)
- Data mapping and schema transformation
- Field naming conventions and patterns
- Semantic field relationships

Your task is to analyze source fields and suggest the best target field mappings based on:
1. Semantic meaning and business logic
2. Data type compatibility  
3. Sample value patterns
4. Banking domain knowledge
5. Compliance requirements

Always respond with valid JSON in the exact format requested. Be precise and confident in your analysis."""

    def _create_mapping_prompt(self, source_field: str, target_fields: List[str],
                               source_samples: List[str] = None,
                               target_field_info: Dict = None,
                               business_context: str = None) -> str:
        """Create a comprehensive prompt for Llama mapping optimized for banking domain"""

        prompt = f"""FIELD MAPPING ANALYSIS

SOURCE FIELD TO MAP:
- Field Name: "{source_field}"
"""

        if source_samples:
            prompt += f"- Sample Values: {', '.join(str(v) for v in source_samples[:5])}\n"

        prompt += f"""
AVAILABLE TARGET FIELDS ({len(target_fields)} options):
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
                    samples_str = ', '.join(str(v) for v in info['samples'][:3])
                    prompt += f" (Examples: {samples_str})"
                if info.get("business_rules"):
                    rules_str = '; '.join(info['business_rules'][:2])
                    prompt += f" [Rules: {rules_str}]"
            prompt += "\n"

        if business_context:
            prompt += f"\nBUSINESS CONTEXT:\n{business_context}\n"

        prompt += """
ANALYSIS REQUIREMENTS:
1. Find the BEST semantic match considering banking domain context
2. Evaluate data type compatibility and sample value patterns
3. Consider UAE banking compliance requirements
4. Provide confidence score (0.0 to 1.0) based on match quality
5. Explain reasoning with specific field analysis
6. Suggest up to 2 alternatives if confidence < 0.9

RESPONSE FORMAT (JSON only):
{
    "best_match": "exact_target_field_name",
    "confidence": 0.85,
    "reasoning": "Detailed explanation focusing on semantic meaning, data patterns, and banking domain relevance",
    "alternatives": [
        {"field": "alternative_field_name", "confidence": 0.6, "reason": "Why this could also work"}
    ]
}

IMPORTANT: 
- Use exact field names from the target list
- Focus on banking/financial domain semantics
- Consider UAE compliance context
- Provide specific reasoning not generic explanations

JSON Response:"""

        return prompt

    def _parse_llm_response(self, response: str, target_fields: List[str]) -> Dict:
        """Parse and validate Llama response with robust error handling"""

        try:
            # Clean response text
            response = response.strip()

            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_text = json_match.group()
            else:
                json_text = response

            # Parse JSON
            try:
                response_data = json.loads(json_text)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
                response_data = json.loads(json_text)

            # Validate and clean the response
            best_match = response_data.get("best_match", "").strip()

            # Find exact match or closest match
            if best_match not in target_fields:
                closest_match = self._find_closest_field_name(best_match, target_fields)
                if closest_match:
                    logger.warning(f"Llama suggested '{best_match}' but using closest match '{closest_match}'")
                    best_match = closest_match
                else:
                    logger.warning(f"Llama suggested invalid field '{best_match}', using first available")
                    best_match = target_fields[0] if target_fields else None

            # Validate confidence score
            confidence = response_data.get("confidence", 0.5)
            confidence = max(0.0, min(1.0, float(confidence)))

            # Process alternatives
            alternatives = []
            for alt in response_data.get("alternatives", [])[:2]:  # Max 2 alternatives
                alt_field = alt.get("field", "").strip()
                if alt_field in target_fields:
                    alternatives.append({
                        "field": alt_field,
                        "confidence": max(0.0, min(1.0, float(alt.get("confidence", 0.3)))),
                        "reason": alt.get("reason", "")[:200]  # Limit reason length
                    })

            return {
                "target_field": best_match,
                "confidence": confidence,
                "reasoning": response_data.get("reasoning", "")[:500],  # Limit reasoning length
                "alternatives": alternatives
            }

        except Exception as e:
            logger.error(f"Failed to parse Llama response: {str(e)}")
            logger.debug(f"Raw response: {response}")

            # Return fallback response
            return {
                "target_field": target_fields[0] if target_fields else None,
                "confidence": 0.3,
                "reasoning": f"Failed to parse Llama response: {str(e)}",
                "alternatives": []
            }

    def _find_closest_field_name(self, suggested_name: str, target_fields: List[str]) -> Optional[str]:
        """Find the closest matching field name using fuzzy matching"""

        if not suggested_name or not target_fields:
            return None

        suggested_lower = suggested_name.lower().replace('_', ' ').replace('-', ' ')
        best_match = None
        best_score = 0

        for field in target_fields:
            field_lower = field.lower().replace('_', ' ').replace('-', ' ')

            # Check for exact substring match
            if suggested_lower in field_lower or field_lower in suggested_lower:
                return field

            # Check for word overlap
            suggested_words = set(suggested_lower.split())
            field_words = set(field_lower.split())
            overlap = len(suggested_words & field_words)

            # Calculate similarity score
            total_words = len(suggested_words | field_words)
            similarity = overlap / total_words if total_words > 0 else 0

            if similarity > best_score and similarity > 0.3:  # Minimum 30% similarity
                best_score = similarity
                best_match = field

        return best_match


class DataMappingAgent:
    """Advanced data mapping agent with BGE embeddings and Llama 3.3 70B assistance"""

    def __init__(self, memory_agent: HybridMemoryAgent, mcp_client: MCPClient,
                 db_session=None, groq_api_key: str = None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

        # Initialize components
        self.bge_manager = BGEEmbeddingManager()
        self.llama_assistant = LlamaAssistantMapper(groq_api_key)

        # Mapping configuration
        self.confidence_thresholds = {
            MappingConfidence.HIGH: 0.90,
            MappingConfidence.MEDIUM: 0.70,
            MappingConfidence.LOW: 0.50
        }

        # Target schema based on actual CSV columns with business context
        self.target_schema = {
            "customer_id": {
                "description": "Unique customer identifier",
                "data_type": "string",
                "required": True,
                "samples": ["CUS770487", "CUS865179", "CUS133659"],
                "business_rules": ["Must be unique across all customers", "Required for customer identification"]
            },
            "customer_type": {
                "description": "Type of customer (Individual or Corporate)",
                "data_type": "string",
                "required": True,
                "samples": ["INDIVIDUAL", "CORPORATE"],
                "business_rules": ["Must be INDIVIDUAL or CORPORATE", "Affects compliance requirements"]
            },
            "full_name_en": {
                "description": "Customer full name in English",
                "data_type": "string",
                "required": True,
                "samples": ["Ali Al-Zaabi", "Fatima Al-Shamsi", "Fatima Al-Zaabi"],
                "business_rules": ["Required for identification", "Must match KYC documents"]
            },
            "full_name_ar": {
                "description": "Customer full name in Arabic",
                "data_type": "string",
                "required": False,
                "samples": ["Ali Al-Zaabi", "Fatima Al-Shamsi", "Fatima Al-Zaabi"],
                "business_rules": ["Optional Arabic name", "Should match English name semantically"]
            },
            "id_number": {
                "description": "Customer identification number",
                "data_type": "number",
                "required": True,
                "samples": [78419988078673, 78419814823498, 78419845443951],
                "business_rules": ["Must be valid Emirates ID or passport number", "Required for compliance"]
            },
            "id_type": {
                "description": "Type of identification document",
                "data_type": "string",
                "required": True,
                "samples": ["EMIRATES_ID", "PASSPORT", "VISA"],
                "business_rules": ["Must be valid ID type", "Links to id_number field"]
            },
            "date_of_birth": {
                "description": "Customer date of birth",
                "data_type": "date",
                "required": True,
                "samples": ["1971-01-03", "1994-05-03", "1974-04-24"],
                "business_rules": ["Must be valid date", "Used for age verification"]
            },
            "nationality": {
                "description": "Customer nationality",
                "data_type": "string",
                "required": True,
                "samples": ["EGYPT", "INDIA", "PHILIPPINES"],
                "business_rules": ["Must be valid country code", "Affects compliance screening"]
            },
            "address_line1": {
                "description": "Primary address line",
                "data_type": "string",
                "required": True,
                "samples": ["Building 224, Street 15", "Building 876, Street 15"],
                "business_rules": ["Required for customer address", "Must be current address"]
            },
            "address_line2": {
                "description": "Secondary address line",
                "data_type": "string",
                "required": False,
                "samples": ["Area 17", "Area 4", "Area 9"],
                "business_rules": ["Optional address details", "Supplements address_line1"]
            },
            "city": {
                "description": "Customer city",
                "data_type": "string",
                "required": True,
                "samples": ["Dubai", "Ajman", "Sharjah"],
                "business_rules": ["Must be valid UAE city", "Part of complete address"]
            },
            "emirate": {
                "description": "UAE emirate",
                "data_type": "string",
                "required": True,
                "samples": ["FUJAIRAH", "SHARJAH", "UMM_AL_QUWAIN"],
                "business_rules": ["Must be valid UAE emirate", "Geographic classification"]
            },
            "country": {
                "description": "Country of residence",
                "data_type": "string",
                "required": True,
                "samples": ["UAE"],
                "business_rules": ["Must be UAE for local customers", "Links to emirate field"]
            },
            "postal_code": {
                "description": "Postal code for address",
                "data_type": "number",
                "required": False,
                "samples": [36062, 69429, 62350],
                "business_rules": ["Optional postal code", "Must be valid format if provided"]
            },
            "phone_primary": {
                "description": "Primary phone number",
                "data_type": "string",
                "required": True,
                "samples": ["+97159142600", "+97156120868", "+97156073292"],
                "business_rules": ["Required contact method", "Must be valid UAE format"]
            },
            "phone_secondary": {
                "description": "Secondary phone number",
                "data_type": "string",
                "required": False,
                "samples": ["+97158897858", "+97150120642"],
                "business_rules": ["Optional backup contact", "Must be valid format if provided"]
            },
            "email_primary": {
                "description": "Primary email address",
                "data_type": "string",
                "required": True,
                "samples": ["ali.al-zaabi@hotmail.com", "fatima.al-shamsi@outlook.com"],
                "business_rules": ["Required for digital communication", "Must be valid email format"]
            },
            "email_secondary": {
                "description": "Secondary email address",
                "data_type": "string",
                "required": False,
                "samples": ["hassan.al-suwaidi@gmail.com", "sara.al-suwaidi@gmail.com"],
                "business_rules": ["Optional backup email", "Must be valid format if provided"]
            },
            "address_known": {
                "description": "Whether customer address is known and valid",
                "data_type": "boolean",
                "required": True,
                "samples": ["YES", "NO"],
                "business_rules": ["Critical for contact attempts", "Affects dormancy communication"]
            },
            "last_contact_date": {
                "description": "Date of last communication with customer",
                "data_type": "date",
                "required": False,
                "samples": ["2021-12-26", "2018-09-30", "2021-02-20"],
                "business_rules": ["Important for dormancy calculation", "Must be valid date if provided"]
            },
            "last_contact_method": {
                "description": "Method used for last customer communication",
                "data_type": "string",
                "required": False,
                "samples": ["EMAIL", "PHONE", "LETTER"],
                "business_rules": ["Links to last_contact_date", "Must be valid communication method"]
            },
            "kyc_status": {
                "description": "Know Your Customer compliance status",
                "data_type": "string",
                "required": True,
                "samples": ["PENDING", "COMPLIANT", "EXPIRED"],
                "business_rules": ["Critical for compliance", "Must be current status"]
            },
            "kyc_expiry_date": {
                "description": "KYC documentation expiry date",
                "data_type": "date",
                "required": True,
                "samples": ["2025-06-09", "2026-03-18", "2024-11-06"],
                "business_rules": ["Must be monitored for renewal", "Affects account status"]
            },
            "risk_rating": {
                "description": "Customer risk assessment rating",
                "data_type": "string",
                "required": True,
                "samples": ["LOW", "HIGH", "MEDIUM"],
                "business_rules": ["Must be LOW, MEDIUM, or HIGH", "Affects monitoring requirements"]
            },
            "account_id": {
                "description": "Unique account identifier",
                "data_type": "string",
                "required": True,
                "samples": ["ACC2867825", "ACC8707870", "ACC6292423"],
                "business_rules": ["Must be unique across all accounts", "Primary key for account records"]
            },
            "account_type": {
                "description": "Type of banking account",
                "data_type": "string",
                "required": True,
                "samples": ["CURRENT", "INVESTMENT", "SAVINGS"],
                "business_rules": ["Must be valid account type", "Affects dormancy rules"]
            },
            "account_subtype": {
                "description": "Account subtype classification",
                "data_type": "string",
                "required": False,
                "samples": ["JOINT", "CORPORATE", "SECURITIES"],
                "business_rules": ["Additional account classification", "Supplements account_type"]
            },
            "account_name": {
                "description": "Descriptive name for the account",
                "data_type": "string",
                "required": True,
                "samples": ["CURRENT - Ali Al-Zaabi", "INVESTMENT - Fatima Al-Zaabi"],
                "business_rules": ["Should include account type and customer name", "For identification purposes"]
            },
            "currency": {
                "description": "Account currency code",
                "data_type": "string",
                "required": True,
                "samples": ["USD", "GBP", "EUR"],
                "business_rules": ["Must be valid ISO currency code", "Affects balance calculations"]
            },
            "account_status": {
                "description": "Current status of the account",
                "data_type": "string",
                "required": True,
                "samples": ["DORMANT", "ACTIVE", "CLOSED"],
                "business_rules": ["Critical for dormancy analysis", "Must be current status"]
            },
            "dormancy_status": {
                "description": "Specific dormancy classification",
                "data_type": "string",
                "required": False,
                "samples": ["FLAGGED", "CONTACTED", "WAITING"],
                "business_rules": ["Links to account_status", "Tracks dormancy process stage"]
            },
            "opening_date": {
                "description": "Date when account was opened",
                "data_type": "date",
                "required": True,
                "samples": ["2016-05-01", "2019-11-12", "2018-04-28"],
                "business_rules": ["Must be valid historical date", "Used for account age calculation"]
            },
            "closing_date": {
                "description": "Date when account was closed",
                "data_type": "date",
                "required": False,
                "samples": ["2024-01-15", "2023-06-30"],
                "business_rules": ["Only for closed accounts", "Must be after opening_date"]
            },
            "last_transaction_date": {
                "description": "Date of last customer-initiated transaction",
                "data_type": "date",
                "required": True,
                "samples": ["2021-02-20", "2018-05-25", "2020-02-27"],
                "business_rules": ["Critical for dormancy calculation", "Must be valid transaction date"]
            },
            "last_system_transaction_date": {
                "description": "Date of last system-generated transaction",
                "data_type": "date",
                "required": False,
                "samples": ["2021-03-22", "2018-05-30", "2020-03-17"],
                "business_rules": ["System transactions don't count for dormancy", "Usually after last_transaction_date"]
            },
            "balance_current": {
                "description": "Current account balance",
                "data_type": "float",
                "required": True,
                "samples": [36849.91, 30964.14, 10710.06],
                "business_rules": ["Must be non-negative for most account types", "Critical for transfer decisions"]
            },
            "balance_available": {
                "description": "Available balance for transactions",
                "data_type": "float",
                "required": True,
                "samples": [35106.87, 27634.01, 8704.13],
                "business_rules": ["Usually less than or equal to current balance", "Accounts for holds and restrictions"]
            },
            "balance_minimum": {
                "description": "Minimum required balance",
                "data_type": "number",
                "required": False,
                "samples": [500, 0],
                "business_rules": ["Account-type specific", "Must be maintained to avoid fees"]
            },
            "interest_rate": {
                "description": "Current interest rate applied to account",
                "data_type": "float",
                "required": False,
                "samples": [0, 2.03, 0.82],
                "business_rules": ["Percentage value", "May be 0 for non-interest bearing accounts"]
            },
            "interest_accrued": {
                "description": "Total interest accrued",
                "data_type": "float",
                "required": False,
                "samples": [0, 533.84, 1908.72],
                "business_rules": ["Cumulative interest earned", "May be 0 for current accounts"]
            },
            "is_joint_account": {
                "description": "Whether account has multiple holders",
                "data_type": "boolean",
                "required": True,
                "samples": ["YES", "NO"],
                "business_rules": ["Affects communication requirements", "Links to joint_account_holders count"]
            },
            "joint_account_holders": {
                "description": "Number of joint account holders",
                "data_type": "number",
                "required": False,
                "samples": [1, 2, 3],
                "business_rules": ["Only relevant if is_joint_account = YES", "Must be >= 1 for joint accounts"]
            },
            "has_outstanding_facilities": {
                "description": "Whether customer has outstanding credit facilities",
                "data_type": "boolean",
                "required": True,
                "samples": ["YES", "NO"],
                "business_rules": ["Affects dormancy classification per Article 2.1.1", "Critical compliance field"]
            },
            "maturity_date": {
                "description": "Maturity date for fixed term deposits",
                "data_type": "date",
                "required": False,
                "samples": ["2021-04-10", "2020-01-06", "2025-01-13"],
                "business_rules": ["Only for term deposits and investments", "Must be future date for active FTDs"]
            },
            "auto_renewal": {
                "description": "Whether account has auto-renewal enabled",
                "data_type": "boolean",
                "required": False,
                "samples": ["YES", "NO"],
                "business_rules": ["Only for term deposits", "Affects dormancy calculation at maturity"]
            },
            "last_statement_date": {
                "description": "Date of last account statement",
                "data_type": "date",
                "required": True,
                "samples": ["2020-12-30", "2018-03-13", "2019-12-16"],
                "business_rules": ["Regular statement generation", "Frequency depends on account type"]
            },
            "statement_frequency": {
                "description": "How often statements are generated",
                "data_type": "string",
                "required": True,
                "samples": ["QUARTERLY", "ANNUAL", "MONTHLY"],
                "business_rules": ["Must be valid frequency", "Links to last_statement_date"]
            },
            "tracking_id": {
                "description": "Internal tracking identifier for dormancy process",
                "data_type": "string",
                "required": True,
                "samples": ["TRK733052", "TRK983794", "TRK887352"],
                "business_rules": ["Unique tracking reference", "Used for dormancy workflow management"]
            },
            "dormancy_trigger_date": {
                "description": "Date when account became dormant",
                "data_type": "date",
                "required": False,
                "samples": ["2024-02-20", "2021-05-24", "2023-02-26"],
                "business_rules": ["Usually 3 years after last transaction", "Critical for compliance timeline"]
            },
            "dormancy_period_start": {
                "description": "Start date of dormancy period",
                "data_type": "date",
                "required": False,
                "samples": ["2024-02-20", "2021-05-24", "2023-02-26"],
                "business_rules": ["Usually same as dormancy_trigger_date", "Begins dormancy workflow"]
            },
            "dormancy_period_months": {
                "description": "Number of months since dormancy trigger",
                "data_type": "number",
                "required": False,
                "samples": [52, 39, 53],
                "business_rules": ["Calculated from dormancy_trigger_date", "Used for transfer eligibility"]
            },
            "dormancy_classification_date": {
                "description": "Date when dormancy was officially classified",
                "data_type": "date",
                "required": False,
                "samples": ["2024-02-20", "2021-05-24", "2023-02-26"],
                "business_rules": ["Administrative classification date", "Usually same as trigger date"]
            },
            "transfer_eligibility_date": {
                "description": "Date when account becomes eligible for central bank transfer",
                "data_type": "date",
                "required": False,
                "samples": ["2023-05-24", "2023-01-19", "2023-10-20"],
                "business_rules": ["Usually 15 years after dormancy trigger", "Per CBUAE Article 8"]
            },
            "current_stage": {
                "description": "Current stage in dormancy workflow",
                "data_type": "string",
                "required": True,
                "samples": ["FLAGGED", "CONTACTED", "WAITING"],
                "business_rules": ["Must be valid workflow stage", "Tracks process progress"]
            },
            "contact_attempts_made": {
                "description": "Number of contact attempts made to customer",
                "data_type": "number",
                "required": True,
                "samples": [5, 4, 3],
                "business_rules": ["Must be non-negative", "Links to contact attempt dates"]
            },
            "last_contact_attempt_date": {
                "description": "Date of most recent contact attempt",
                "data_type": "date",
                "required": False,
                "samples": ["2021-12-26", "2018-09-30", "2021-02-20"],
                "business_rules": ["Must be valid date if provided", "Should align with contact_attempts_made"]
            },
            "waiting_period_start": {
                "description": "Start date of waiting period after contact",
                "data_type": "date",
                "required": False,
                "samples": ["2021-12-27", "2018-10-01", "2021-02-21"],
                "business_rules": ["Usually day after last contact attempt", "Part of compliance process"]
            },
            "waiting_period_end": {
                "description": "End date of waiting period",
                "data_type": "date",
                "required": False,
                "samples": ["2022-03-26", "2018-12-29", "2021-05-21"],
                "business_rules": ["Usually 3 months after waiting period start", "Determines next action"]
            },
            "transferred_to_ledger_date": {
                "description": "Date when funds were transferred to separate ledger",
                "data_type": "date",
                "required": False,
                "samples": ["2024-01-15", "2023-06-30"],
                "business_rules": ["Intermediate step before central bank transfer", "Per CBUAE process"]
            },
            "transferred_to_cb_date": {
                "description": "Date when funds were transferred to central bank",
                "data_type": "date",
                "required": False,
                "samples": ["2024-06-15", "2023-12-30"],
                "business_rules": ["Final step in dormancy process", "Must be after transfer_eligibility_date"]
            },
            "cb_transfer_amount": {
                "description": "Amount transferred to central bank",
                "data_type": "float",
                "required": False,
                "samples": [30964.14, 10137.44, 33127.38],
                "business_rules": ["Should match final account balance", "Only for transferred accounts"]
            },
            "cb_transfer_reference": {
                "description": "Central bank transfer reference number",
                "data_type": "string",
                "required": False,
                "samples": ["CB961722", "CB790993", "CB373432"],
                "business_rules": ["Unique reference from central bank", "Links to cb_transfer_amount"]
            },
            "exclusion_reason": {
                "description": "Reason for exclusion from dormancy process",
                "data_type": "string",
                "required": False,
                "samples": ["ACTIVE_LIABILITY", "COURT_ORDER", "DECEASED"],
                "business_rules": ["Valid exclusion reasons per regulation", "Prevents automatic processing"]
            },
            "created_date": {
                "description": "Record creation timestamp",
                "data_type": "datetime",
                "required": True,
                "samples": ["2016-05-01 00:00:00", "2019-11-12 00:00:00"],
                "business_rules": ["System generated timestamp", "Should match or be after opening_date"]
            },
            "updated_date": {
                "description": "Last record update timestamp",
                "data_type": "datetime",
                "required": True,
                "samples": ["2024-12-01 00:00:00"],
                "business_rules": ["System generated timestamp", "Should be recent for active records"]
            },
            "updated_by": {
                "description": "System or user who last updated the record",
                "data_type": "string",
                "required": True,
                "samples": ["SYSTEM", "USER123", "ADMIN"],
                "business_rules": ["Audit trail information", "Links to user management system"]
            }
        }

    @traceable
    async def analyze_source_schema(self, state: DataMappingState) -> DataMappingState:
        """Analyze source schema and extract field information"""

        logger.info(f"Analyzing source schema for mapping {state.mapping_id}")

        try:
            start_time = datetime.now()

            # Extract source fields from DataFrame or schema
            if state.source_data_sample is not None:
                source_fields = list(state.source_data_sample.columns)

                # Extract sample values and data types for each field
                source_field_info = {}
                for field in source_fields:
                    column_data = state.source_data_sample[field]

                    # Get sample values (non-null, unique, limited)
                    sample_values = column_data.dropna().unique()[:5].tolist()

                    # Infer data type
                    if pd.api.types.is_datetime64_any_dtype(column_data):
                        data_type = "datetime"
                    elif pd.api.types.is_numeric_dtype(column_data):
                        data_type = "number" if column_data.dtype in ['int64', 'int32'] else "float"
                    elif pd.api.types.is_bool_dtype(column_data):
                        data_type = "boolean"
                    else:
                        data_type = "string"

                    source_field_info[field] = {
                        "data_type": data_type,
                        "sample_values": sample_values,
                        "null_count": column_data.isnull().sum(),
                        "total_count": len(column_data)
                    }

                state.source_schema = source_field_info
                state.total_fields = len(source_fields)

            elif state.source_schema:
                state.total_fields = len(state.source_schema)
            else:
                raise ValueError("No source schema or data sample provided")

            # Update processing time
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.mapping_status = MappingStatus.ANALYZING

            # Log analysis results
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "schema_analysis_completed",
                "details": f"Analyzed {state.total_fields} source fields",
                "processing_time": state.processing_time
            })

            logger.info(f"Source schema analysis completed: {state.total_fields} fields identified")

        except Exception as e:
            logger.error(f"Source schema analysis failed: {str(e)}")
            state.mapping_status = MappingStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "step": "schema_analysis"
            })

        return state

    @traceable
    async def generate_embeddings(self, state: DataMappingState) -> DataMappingState:
        """Generate BGE embeddings for source and target fields"""

        logger.info(f"Generating BGE embeddings for mapping {state.mapping_id}")

        try:
            start_time = datetime.now()

            # Initialize BGE model if not already done
            if not self.bge_manager.model:
                await self.bge_manager.initialize_model()

            # Generate embeddings for source fields
            source_embeddings = {}
            target_embeddings = {}

            # Process source fields
            for field_name, field_info in state.source_schema.items():
                sample_values = field_info.get("sample_values", [])
                data_type = field_info.get("data_type", "string")

                embedding = await self.bge_manager.get_field_embedding(
                    field_name, sample_values, data_type
                )
                source_embeddings[field_name] = embedding

            # Generate embeddings for target fields (cached)
            for target_field, target_info in self.target_schema.items():
                sample_values = target_info.get("samples", [])
                data_type = target_info.get("data_type", "string")

                embedding = await self.bge_manager.get_field_embedding(
                    target_field, sample_values, data_type
                )
                target_embeddings[target_field] = embedding

            # Store embeddings in state
            state.memory_context["source_embeddings"] = source_embeddings
            state.memory_context["target_embeddings"] = target_embeddings

            # Update timing
            state.embedding_time = (datetime.now() - start_time).total_seconds()

            # Log completion
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "embeddings_generated",
                "details": f"Generated embeddings for {len(source_embeddings)} source and {len(target_embeddings)} target fields",
                "processing_time": state.embedding_time
            })

            logger.info(f"BGE embeddings generated in {state.embedding_time:.2f}s")

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "step": "embedding_generation"
            })

        return state

    @traceable
    async def calculate_similarities(self, state: DataMappingState) -> DataMappingState:
        """Calculate semantic similarities between source and target fields"""

        logger.info(f"Calculating semantic similarities for mapping {state.mapping_id}")

        try:
            start_time = datetime.now()

            source_embeddings = state.memory_context.get("source_embeddings", {})
            target_embeddings = state.memory_context.get("target_embeddings", {})

            # Calculate similarities for each source field
            field_similarities = {}

            for source_field, source_embedding in source_embeddings.items():
                similarities = {}

                for target_field, target_embedding in target_embeddings.items():
                    similarity = await self.bge_manager.calculate_semantic_similarity(
                        source_embedding, target_embedding
                    )
                    similarities[target_field] = similarity

                # Sort by similarity score
                sorted_similarities = sorted(
                    similarities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                field_similarities[source_field] = sorted_similarities

            # Store similarity results
            state.memory_context["field_similarities"] = field_similarities

            # Update timing
            state.similarity_calculation_time = (datetime.now() - start_time).total_seconds()

            # Log completion
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "similarities_calculated",
                "details": f"Calculated similarities for {len(field_similarities)} source fields",
                "processing_time": state.similarity_calculation_time
            })

            logger.info(f"Similarity calculation completed in {state.similarity_calculation_time:.2f}s")

        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "step": "similarity_calculation"
            })

        return state

    @traceable
    async def create_initial_mappings(self, state: DataMappingState) -> DataMappingState:
        """Create initial field mappings based on semantic similarity"""

        logger.info(f"Creating initial mappings for mapping {state.mapping_id}")

        try:
            field_similarities = state.memory_context.get("field_similarities", {})
            source_embeddings = state.memory_context.get("source_embeddings", {})

            field_mappings = []
            high_confidence_count = 0
            requires_llm_count = 0

            for source_field, similarities in field_similarities.items():
                if not similarities:
                    continue

                # Get best match
                best_target_field, best_similarity = similarities[0]

                # Determine confidence level
                if best_similarity >= self.confidence_thresholds[MappingConfidence.HIGH]:
                    confidence_level = MappingConfidence.HIGH
                    mapping_strategy = MappingStrategy.AUTOMATIC
                    high_confidence_count += 1
                elif best_similarity >= self.confidence_thresholds[MappingConfidence.MEDIUM]:
                    confidence_level = MappingConfidence.MEDIUM
                    mapping_strategy = MappingStrategy.HYBRID
                elif best_similarity >= self.confidence_thresholds[MappingConfidence.LOW]:
                    confidence_level = MappingConfidence.LOW
                    mapping_strategy = MappingStrategy.LLM_ASSISTED
                    requires_llm_count += 1
                else:
                    confidence_level = MappingConfidence.VERY_LOW
                    mapping_strategy = MappingStrategy.MANUAL

                # Check data type compatibility
                source_info = state.source_schema.get(source_field, {})
                target_info = self.target_schema.get(best_target_field, {})
                data_type_match = source_info.get("data_type") == target_info.get("data_type")

                # Create field mapping
                field_mapping = FieldMapping(
                    source_field=source_field,
                    target_field=best_target_field,
                    confidence_score=best_similarity,
                    confidence_level=confidence_level,
                    mapping_strategy=mapping_strategy,
                    semantic_similarity=best_similarity,
                    embedding_vector=source_embeddings.get(source_field),
                    data_type_match=data_type_match,
                    sample_values=source_info.get("sample_values", []),
                    business_rules=target_info.get("business_rules", [])
                )

                field_mappings.append(field_mapping)

            # Update state
            state.field_mappings = field_mappings
            state.high_confidence_mappings = high_confidence_count
            state.requires_llm_assistance = requires_llm_count

            # Log results
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "initial_mappings_created",
                "details": f"Created {len(field_mappings)} initial mappings, {high_confidence_count} high confidence, {requires_llm_count} need LLM assistance"
            })

            logger.info(f"Initial mappings created: {len(field_mappings)} total, {high_confidence_count} high confidence")

        except Exception as e:
            logger.error(f"Initial mapping creation failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "step": "initial_mapping_creation"
            })

        return state

    @traceable
    async def llm_assistance_for_low_confidence(self, state: DataMappingState) -> DataMappingState:
        """Use Llama 3.3 70B for low confidence mappings"""

        logger.info(f"Applying LLM assistance for low confidence mappings")

        try:
            # Get mappings that need LLM assistance
            llm_mappings = [
                mapping for mapping in state.field_mappings
                if mapping.mapping_strategy == MappingStrategy.LLM_ASSISTED
            ]

            if not llm_mappings:
                logger.info("No mappings require LLM assistance")
                return state

            logger.info(f"Processing {len(llm_mappings)} mappings with Llama 3.3 70B")

            # Process each low confidence mapping
            for mapping in llm_mappings:
                try:
                    # Get source field info
                    source_info = state.source_schema.get(mapping.source_field, {})
                    source_samples = source_info.get("sample_values", [])

                    # Get top target field candidates
                    field_similarities = state.memory_context.get("field_similarities", {})
                    similarities = field_similarities.get(mapping.source_field, [])

                    # Take top 5 candidates for LLM consideration
                    target_candidates = [field for field, _ in similarities[:5]]

                    # Create business context
                    business_context = f"""
UAE Banking Compliance Context:
- This is for dormant account compliance per CBUAE regulations
- Customer identification and contact information is critical
- Account status and transaction history determines dormancy classification
- Risk assessment and KYC status affect compliance requirements
                    """

                    # Call Llama for suggestion
                    llm_result = await self.llama_assistant.suggest_mapping(
                        source_field=mapping.source_field,
                        target_fields=target_candidates,
                        source_samples=source_samples,
                        target_field_info=self.target_schema,
                        business_context=business_context
                    )

                    if llm_result["success"]:
                        suggested_mapping = llm_result["suggested_mapping"]

                        # Update mapping with LLM suggestion
                        mapping.target_field = suggested_mapping["target_field"]
                        mapping.confidence_score = suggested_mapping["confidence"]
                        mapping.llm_suggestion = suggested_mapping["reasoning"]

                        # Update confidence level based on LLM confidence
                        if suggested_mapping["confidence"] >= 0.8:
                            mapping.confidence_level = MappingConfidence.HIGH
                            mapping.mapping_strategy = MappingStrategy.HYBRID
                        elif suggested_mapping["confidence"] >= 0.6:
                            mapping.confidence_level = MappingConfidence.MEDIUM
                            mapping.mapping_strategy = MappingStrategy.HYBRID
                        else:
                            mapping.confidence_level = MappingConfidence.LOW
                            mapping.mapping_strategy = MappingStrategy.MANUAL

                        mapping.updated_at = datetime.now()

                        logger.info(f"LLM improved mapping for '{mapping.source_field}' -> '{mapping.target_field}' (confidence: {suggested_mapping['confidence']:.2f})")

                    else:
                        logger.warning(f"LLM assistance failed for field '{mapping.source_field}': {llm_result['error']}")
                        mapping.mapping_strategy = MappingStrategy.MANUAL

                except Exception as e:
                    logger.error(f"LLM processing failed for field '{mapping.source_field}': {str(e)}")
                    mapping.mapping_strategy = MappingStrategy.MANUAL

            # Update counts
            state.high_confidence_mappings = sum(1 for m in state.field_mappings if m.confidence_level == MappingConfidence.HIGH)
            state.requires_llm_assistance = sum(1 for m in state.field_mappings if m.mapping_strategy == MappingStrategy.LLM_ASSISTED)
            state.requires_user_input = sum(1 for m in state.field_mappings if m.mapping_strategy == MappingStrategy.MANUAL)

            # Log completion
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "llm_assistance_completed",
                "details": f"LLM processed {len(llm_mappings)} mappings, {state.high_confidence_mappings} now high confidence"
            })

        except Exception as e:
            logger.error(f"LLM assistance failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "step": "llm_assistance"
            })

        return state

    @traceable
    async def finalize_mappings(self, state: DataMappingState) -> DataMappingState:
        """Finalize the mapping results and create summary"""

        logger.info(f"Finalizing mappings for mapping {state.mapping_id}")

        try:
            # Create mapping summary
            total_mappings = len(state.field_mappings)
            high_confidence = sum(1 for m in state.field_mappings if m.confidence_level == MappingConfidence.HIGH)
            medium_confidence = sum(1 for m in state.field_mappings if m.confidence_level == MappingConfidence.MEDIUM)
            low_confidence = sum(1 for m in state.field_mappings if m.confidence_level == MappingConfidence.LOW)
            very_low_confidence = sum(1 for m in state.field_mappings if m.confidence_level == MappingConfidence.VERY_LOW)

            automatic_mappings = sum(1 for m in state.field_mappings if m.mapping_strategy == MappingStrategy.AUTOMATIC)
            hybrid_mappings = sum(1 for m in state.field_mappings if m.mapping_strategy == MappingStrategy.HYBRID)
            manual_mappings = sum(1 for m in state.field_mappings if m.mapping_strategy == MappingStrategy.MANUAL)

            # Identify unmapped source fields
            mapped_source_fields = {m.source_field for m in state.field_mappings}
            all_source_fields = set(state.source_schema.keys())
            unmapped_fields = list(all_source_fields - mapped_source_fields)

            # Create comprehensive summary
            mapping_summary = {
                "total_source_fields": len(all_source_fields),
                "total_mappings": total_mappings,
                "unmapped_fields_count": len(unmapped_fields),
                "confidence_distribution": {
                    "high": high_confidence,
                    "medium": medium_confidence,
                    "low": low_confidence,
                    "very_low": very_low_confidence
                },
                "strategy_distribution": {
                    "automatic": automatic_mappings,
                    "hybrid": hybrid_mappings,
                    "manual": manual_mappings
                },
                "mapping_quality_score": (high_confidence * 1.0 + medium_confidence * 0.7 + low_confidence * 0.4) / total_mappings if total_mappings > 0 else 0,
                "requires_user_review": manual_mappings + low_confidence,
                "ready_for_automation": automatic_mappings + hybrid_mappings
            }

            # Update state with final results
            state.mapping_summary = mapping_summary
            state.unmapped_fields = unmapped_fields
            state.mapping_status = MappingStatus.COMPLETED

            # Calculate total processing time
            state.processing_time = (datetime.now() - state.timestamp).total_seconds()

            # Final log entry
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "mapping_finalized",
                "details": f"Completed mapping: {total_mappings} mappings, quality score: {mapping_summary['mapping_quality_score']:.2f}",
                "processing_time": state.processing_time
            })

            logger.info(f"Mapping finalized: {total_mappings} mappings, quality score: {mapping_summary['mapping_quality_score']:.2f}")

        except Exception as e:
            logger.error(f"Mapping finalization failed: {str(e)}")
            state.mapping_status = MappingStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "step": "finalization"
            })

        return state

    async def save_mapping_results(self, state: DataMappingState) -> bool:
        """Save mapping results to memory and database"""

        try:
            # Prepare memory context for storage
            memory_context = MemoryContext(
                session_id=state.session_id,
                user_id=state.user_id,
                timestamp=datetime.now(),
                context_type="data_mapping",
                content={
                    "mapping_id": state.mapping_id,
                    "source_schema": state.source_schema,
                    "field_mappings": [asdict(mapping) for mapping in state.field_mappings],
                    "mapping_summary": state.mapping_summary,
                    "processing_metrics": {
                        "total_processing_time": state.processing_time,
                        "embedding_time": state.embedding_time,
                        "similarity_calculation_time": state.similarity_calculation_time
                    }
                },
                metadata={
                    "mapping_status": state.mapping_status.value,
                    "total_fields": state.total_fields,
                    "high_confidence_mappings": state.high_confidence_mappings,
                    "requires_user_input": state.requires_user_input
                }
            )

            # Store in memory agent
            await self.memory_agent.store_memory(memory_context)

            # Save to database if available
            if self.db_session:
                # Implementation depends on your database schema
                # This is a placeholder for database storage
                pass

            logger.info(f"Mapping results saved for mapping {state.mapping_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save mapping results: {str(e)}")
            return False

    async def retrieve_historical_mappings(self, source_fields: List[str], limit: int = 10) -> Dict:
        """Retrieve historical mapping patterns for similar fields"""

        try:
            # Search for similar mapping contexts in memory
            search_results = await self.memory_agent.search_memories(
                query=f"data mapping fields: {', '.join(source_fields[:5])}",
                context_type="data_mapping",
                limit=limit
            )

            historical_patterns = {}
            for result in search_results:
                content = result.get("content", {})
                field_mappings = content.get("field_mappings", [])

                for mapping in field_mappings:
                    source_field = mapping.get("source_field")
                    if source_field in source_fields:
                        if source_field not in historical_patterns:
                            historical_patterns[source_field] = []

                        historical_patterns[source_field].append({
                            "target_field": mapping.get("target_field"),
                            "confidence": mapping.get("confidence_score"),
                            "strategy": mapping.get("mapping_strategy"),
                            "timestamp": result.get("timestamp")
                        })

            return historical_patterns

        except Exception as e:
            logger.error(f"Failed to retrieve historical mappings: {str(e)}")
            return {}

    def create_mapping_workflow(self) -> StateGraph:
        """Create LangGraph workflow for data mapping process"""

        workflow = StateGraph(DataMappingState)

        # Add nodes for each step
        workflow.add_node("analyze_schema", self.analyze_source_schema)
        workflow.add_node("generate_embeddings", self.generate_embeddings)
        workflow.add_node("calculate_similarities", self.calculate_similarities)
        workflow.add_node("create_mappings", self.create_initial_mappings)
        workflow.add_node("llm_assistance", self.llm_assistance_for_low_confidence)
        workflow.add_node("finalize", self.finalize_mappings)

        # Define workflow edges
        workflow.add_edge("analyze_schema", "generate_embeddings")
        workflow.add_edge("generate_embeddings", "calculate_similarities")
        workflow.add_edge("calculate_similarities", "create_mappings")
        workflow.add_edge("create_mappings", "llm_assistance")
        workflow.add_edge("llm_assistance", "finalize")
        workflow.add_edge("finalize", END)

        # Set entry point
        workflow.set_entry_point("analyze_schema")

        return workflow.compile()

    @traceable
    async def execute_mapping(self, source_data: pd.DataFrame, mapping_config: Dict = None) -> DataMappingState:
        """Execute the complete data mapping workflow"""

        # Generate unique IDs
        session_id = secrets.token_hex(8)
        mapping_id = f"MAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
        user_id = mapping_config.get("user_id", "anonymous") if mapping_config else "anonymous"

        # Initialize state
        initial_state = DataMappingState(
            session_id=session_id,
            user_id=user_id,
            mapping_id=mapping_id,
            timestamp=datetime.now(),
            source_data_sample=source_data,
            mapping_config=mapping_config or {}
        )

        logger.info(f"Starting data mapping workflow: {mapping_id}")

        try:
            # Create and run workflow
            workflow = self.create_mapping_workflow()
            final_state = await workflow.ainvoke(initial_state)

            # Save results
            await self.save_mapping_results(final_state)

            logger.info(f"Data mapping workflow completed: {mapping_id}")
            return final_state

        except Exception as e:
            logger.error(f"Data mapping workflow failed: {str(e)}")
            initial_state.mapping_status = MappingStatus.FAILED
            initial_state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "step": "workflow_execution"
            })
            return initial_state

    async def get_mapping_suggestions_for_user_review(self, mapping_id: str) -> Dict:
        """Get mapping suggestions formatted for user review"""

        try:
            # Retrieve mapping state from memory
            search_results = await self.memory_agent.search_memories(
                query=f"mapping_id: {mapping_id}",
                context_type="data_mapping",
                limit=1
            )

            if not search_results:
                return {"error": "Mapping not found"}

            mapping_data = search_results[0]["content"]
            field_mappings = mapping_data.get("field_mappings", [])

            # Organize mappings by confidence level for user review
            review_data = {
                "mapping_id": mapping_id,
                "total_fields": len(field_mappings),
                "high_confidence": [],
                "medium_confidence": [],
                "low_confidence": [],
                "requires_manual_review": []
            }

            for mapping in field_mappings:
                confidence_level = mapping.get("confidence_level")
                mapping_strategy = mapping.get("mapping_strategy")

                review_item = {
                    "source_field": mapping.get("source_field"),
                    "suggested_target": mapping.get("target_field"),
                    "confidence_score": mapping.get("confidence_score"),
                    "reasoning": mapping.get("llm_suggestion", "Semantic similarity match"),
                    "sample_values": mapping.get("sample_values", []),
                    "data_type_match": mapping.get("data_type_match", False)
                }

                if confidence_level == "high":
                    review_data["high_confidence"].append(review_item)
                elif confidence_level == "medium":
                    review_data["medium_confidence"].append(review_item)
                elif confidence_level == "low":
                    review_data["low_confidence"].append(review_item)
                else:
                    review_data["requires_manual_review"].append(review_item)

            return review_data

        except Exception as e:
            logger.error(f"Failed to get mapping suggestions: {str(e)}")
            return {"error": str(e)}

    async def apply_user_feedback(self, mapping_id: str, user_decisions: List[Dict]) -> bool:
        """Apply user feedback to mapping results"""

        try:
            # Update mapping state with user decisions
            for decision in user_decisions:
                source_field = decision.get("source_field")
                user_target = decision.get("user_target_field")
                user_confirmed = decision.get("confirmed", False)

                # Store user decision in memory
                decision_context = MemoryContext(
                    session_id=f"feedback_{mapping_id}",
                    user_id=decision.get("user_id", "anonymous"),
                    timestamp=datetime.now(),
                    context_type="mapping_feedback",
                    content={
                        "mapping_id": mapping_id,
                        "source_field": source_field,
                        "user_target_field": user_target,
                        "user_confirmed": user_confirmed,
                        "feedback_type": "field_mapping_decision"
                    }
                )

                await self.memory_agent.store_memory(decision_context)

            logger.info(f"Applied user feedback for mapping {mapping_id}: {len(user_decisions)} decisions")
            return True

        except Exception as e:
            logger.error(f"Failed to apply user feedback: {str(e)}")
            return False

    def cleanup_resources(self):
        """Cleanup BGE model and other resources"""
        try:
            if self.bge_manager:
                self.bge_manager.cleanup()
            logger.info("Data mapping agent resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")


# Example usage and testing functions
async def example_usage():
    """Example of how to use the DataMappingAgent with Llama 3.3 70B"""

    # Initialize required components (placeholders)
    memory_agent = None  # HybridMemoryAgent instance
    mcp_client = None    # MCPClient instance
    groq_api_key = "your_groq_api_key_here"  # Get from Groq

    # Create mapping agent
    mapping_agent = DataMappingAgent(
        memory_agent=memory_agent,
        mcp_client=mcp_client,
        groq_api_key=groq_api_key
    )

    # Sample source data (would come from uploaded CSV)
    sample_data = pd.DataFrame({
        "cust_id": ["C001", "C002", "C003"],
        "cust_name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "phone": ["+971501234567", "+971502345678", "+971503456789"],
        "email": ["john@example.com", "jane@example.com", "bob@example.com"],
        "account_number": ["ACC001", "ACC002", "ACC003"],
        "balance": [1000.50, 2500.75, 750.00],
        "last_txn_date": ["2023-01-15", "2023-02-20", "2023-01-10"]
    })

    # Configuration
    mapping_config = {
        "user_id": "data_analyst_01",
        "confidence_threshold": 0.7,
        "use_llm_assistance": True,
        "save_patterns": True
    }

    try:
        # Execute mapping workflow
        print("Starting data mapping workflow...")
        mapping_result = await mapping_agent.execute_mapping(
            source_data=sample_data,
            mapping_config=mapping_config
        )

        # Display results
        print(f"\nMapping completed with status: {mapping_result.mapping_status}")
        print(f"Total fields processed: {mapping_result.total_fields}")
        print(f"High confidence mappings: {mapping_result.high_confidence_mappings}")
        print(f"Requires user input: {mapping_result.requires_user_input}")

        # Show mapping summary
        if mapping_result.mapping_summary:
            summary = mapping_result.mapping_summary
            print(f"\nMapping Quality Score: {summary['mapping_quality_score']:.2f}")
            print(f"Ready for automation: {summary['ready_for_automation']} fields")
            print(f"Requires review: {summary['requires_user_review']} fields")

        # Display individual mappings
        print("\nField Mappings:")
        for mapping in mapping_result.field_mappings:
            print(f"  {mapping.source_field} -> {mapping.target_field}")
            print(f"    Confidence: {mapping.confidence_score:.2f} ({mapping.confidence_level.value})")
            print(f"    Strategy: {mapping.mapping_strategy.value}")
            if mapping.llm_suggestion:
                print(f"    LLM Reasoning: {mapping.llm_suggestion[:100]}...")
            print()

        # Get suggestions for user review
        review_data = await mapping_agent.get_mapping_suggestions_for_user_review(
            mapping_result.mapping_id
        )

        print(f"\nFields requiring manual review: {len(review_data.get('requires_manual_review', []))}")

        # Cleanup
        mapping_agent.cleanup_resources()

        return mapping_result

    except Exception as e:
        print(f"Mapping workflow failed: {str(e)}")
        mapping_agent.cleanup_resources()
        return None


if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())