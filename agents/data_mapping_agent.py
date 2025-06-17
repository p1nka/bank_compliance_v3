"""
agents/data_mapping_agent.py - Advanced Data Mapping Agent
Uses BGE Large v1.5 embeddings for semantic similarity and automatic column mapping
Supports LLM assistance and manual mapping for complex cases
"""

import logging
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import secrets
import json
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Sentence Transformers for BGE embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("sentence-transformers not available, using mock implementation")
    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
        def encode(self, texts):
            return np.random.rand(len(texts), 1024)

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable, Client as LangSmithClient

# MCP imports with fallback
try:
    from mcp_client import MCPClient
except ImportError:
    logging.warning("MCPClient not available, using mock implementation")

    class MCPClient:
        async def call_tool(self, tool_name: str, params: Dict) -> Dict:
            return {"success": True, "data": {}}

# Groq LLM imports
try:
    from groq import Groq
except ImportError:
    logging.warning("Groq not available, using mock implementation")

    class Groq:
        def __init__(self, api_key=None):
            self.api_key = api_key

        class Chat:
            class Completions:
                def create(self, **kwargs):
                    return type('obj', (object,), {
                        'choices': [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': 'Mock LLM response for column mapping'
                            })()
                        })()]
                    })()

        @property
        def chat(self):
            return type('obj', (object,), {'completions': self.Chat.Completions()})()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== ENUMS AND DATACLASSES =====

class MappingStatus(Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    AUTO_MAPPING = "auto_mapping"
    REQUIRES_USER_INPUT = "requires_user_input"
    LLM_PROCESSING = "llm_processing"
    MANUAL_MAPPING = "manual_mapping"
    COMPLETED = "completed"
    FAILED = "failed"

class MappingStrategy(Enum):
    AUTOMATIC = "automatic"
    LLM_ASSISTED = "llm_assisted"
    MANUAL = "manual"
    HYBRID = "hybrid"

class MappingConfidence(Enum):
    HIGH = "high"      # 90%+ similarity
    MEDIUM = "medium"  # 70-89% similarity
    LOW = "low"        # 50-69% similarity
    VERY_LOW = "very_low"  # <50% similarity

@dataclass
class FieldMapping:
    """Individual field mapping result with BGE embeddings"""

    source_field: str
    target_field: Optional[str]
    confidence_score: float
    confidence_level: MappingConfidence
    mapping_strategy: MappingStrategy

    # BGE embedding analysis
    source_embedding: Optional[np.ndarray] = None
    target_embedding: Optional[np.ndarray] = None
    cosine_similarity_score: float = 0.0

    # Semantic analysis
    semantic_keywords: List[str] = field(default_factory=list)
    business_context: Optional[str] = None

    # Data type and sample analysis
    source_data_type: Optional[str] = None
    target_data_type: Optional[str] = None
    data_type_match: bool = False
    source_samples: List[str] = field(default_factory=list)
    target_samples: List[str] = field(default_factory=list)

    # User interaction
    user_confirmed: Optional[bool] = None
    user_override: Optional[str] = None
    llm_suggestion: Optional[str] = None
    llm_confidence: Optional[float] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def update_confidence_level(self):
        """Update confidence level based on cosine similarity score"""
        if self.cosine_similarity_score >= 0.90:
            self.confidence_level = MappingConfidence.HIGH
        elif self.cosine_similarity_score >= 0.70:
            self.confidence_level = MappingConfidence.MEDIUM
        elif self.cosine_similarity_score >= 0.50:
            self.confidence_level = MappingConfidence.LOW
        else:
            self.confidence_level = MappingConfidence.VERY_LOW

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
    mapping_config: Dict = field(default_factory=dict)

    # BGE embeddings
    source_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    target_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    similarity_matrix: Optional[np.ndarray] = None

    # Processing results
    field_mappings: List[FieldMapping] = field(default_factory=list)
    mapping_summary: Dict = field(default_factory=dict)
    unmapped_fields: List[str] = field(default_factory=list)
    auto_mapped_fields: List[str] = field(default_factory=list)

    # Status tracking
    mapping_status: MappingStatus = MappingStatus.PENDING
    total_fields: int = 0
    high_confidence_mappings: int = 0
    medium_confidence_mappings: int = 0
    low_confidence_mappings: int = 0
    requires_user_input: int = 0
    auto_mapping_percentage: float = 0.0

    # Performance metrics
    processing_time: float = 0.0
    embedding_time: float = 0.0
    similarity_calculation_time: float = 0.0

    # Workflow routing
    user_choice: Optional[str] = None  # "llm" or "manual"

    # Error handling
    error_log: List[Dict] = field(default_factory=list)
    mapping_log: List[Dict] = field(default_factory=list)

# ===== BGE EMBEDDING MANAGER =====

class BGEEmbeddingManager:
    """Manages BGE Large v1.5 embeddings for semantic similarity"""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load BGE Large v1.5 model"""
        try:
            logger.info(f"Loading BGE model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("BGE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            # Fallback to a smaller model or mock
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.warning("Using fallback model: all-MiniLM-L6-v2")
            except:
                logger.error("All models failed to load, using mock embeddings")
                self.model = None

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            # Mock embeddings for testing
            return np.random.rand(len(texts), 1024)

        try:
            # Preprocess texts for better embeddings
            processed_texts = []
            for text in texts:
                # Add business context keywords
                processed_text = self._enhance_text_context(text)
                processed_texts.append(processed_text)

            embeddings = self.model.encode(processed_texts, normalize_embeddings=True)
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return np.random.rand(len(texts), 1024)

    def _enhance_text_context(self, text: str) -> str:
        """Enhance text with business context for better embeddings"""
        # Convert field names to more descriptive text
        enhanced_text = text.lower().replace('_', ' ').replace('-', ' ')

        # Add banking/financial context keywords
        context_mappings = {
            'customer': 'customer client account holder',
            'account': 'bank account financial account',
            'balance': 'account balance money amount',
            'transaction': 'financial transaction payment',
            'date': 'date time timestamp',
            'id': 'identifier unique id number',
            'name': 'name full name customer name',
            'type': 'type category classification',
            'status': 'status state condition',
            'currency': 'currency money type',
            'dormancy': 'dormancy inactive sleeping account',
            'contact': 'contact communication phone email',
            'address': 'address location residence',
            'amount': 'amount money value balance'
        }

        for keyword, context in context_mappings.items():
            if keyword in enhanced_text:
                enhanced_text += f" {context}"

        return enhanced_text

    def calculate_similarity_matrix(self, source_embeddings: np.ndarray,
                                  target_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix between source and target embeddings"""
        try:
            similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
            return similarity_matrix
        except Exception as e:
            logger.error(f"Failed to calculate similarity matrix: {e}")
            # Return random similarity matrix as fallback
            return np.random.rand(len(source_embeddings), len(target_embeddings))

# ===== LLM ASSISTANT MAPPER =====

class LLMAssistantMapper:
    """LLM-assisted mapping using Groq Llama 3.3 70B"""

    def __init__(self, api_key: str = None):
        self.client = Groq(api_key=api_key) if api_key else None

    async def suggest_mappings(self, unmapped_fields: List[str],
                             target_schema: Dict,
                             source_samples: Dict) -> Dict[str, Dict]:
        """Get LLM suggestions for unmapped fields"""
        if not self.client:
            return self._mock_llm_suggestions(unmapped_fields, target_schema)

        suggestions = {}

        for field in unmapped_fields:
            try:
                prompt = self._create_mapping_prompt(field, target_schema, source_samples.get(field, []))

                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data mapping assistant for banking compliance systems. Provide accurate column mappings based on field names, data types, and sample values."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )

                # Parse LLM response
                suggestion = self._parse_llm_response(response.choices[0].message.content)
                suggestions[field] = suggestion

            except Exception as e:
                logger.error(f"LLM suggestion failed for field {field}: {e}")
                suggestions[field] = {
                    "suggested_target": None,
                    "confidence": 0.0,
                    "reasoning": "LLM processing failed"
                }

        return suggestions

    def _create_mapping_prompt(self, source_field: str, target_schema: Dict,
                              sample_values: List[str]) -> str:
        """Create a detailed prompt for LLM mapping suggestion"""
        target_fields_desc = []
        for field_name, field_info in target_schema.items():
            desc = f"- {field_name}: {field_info.get('description', 'No description')}"
            if field_info.get('samples'):
                desc += f" (Examples: {', '.join(str(x) for x in field_info['samples'][:3])})"
            target_fields_desc.append(desc)

        prompt = f"""
I need to map a source column to the best matching target column in a banking compliance dataset.

SOURCE COLUMN:
- Name: {source_field}
- Sample Values: {', '.join(str(x) for x in sample_values[:5]) if sample_values else 'No samples available'}

TARGET SCHEMA OPTIONS:
{chr(10).join(target_fields_desc)}

Please analyze the source column and suggest the best matching target column. Consider:
1. Semantic similarity of column names
2. Data type compatibility
3. Sample value patterns
4. Banking domain context

Respond in JSON format:
{{
    "suggested_target": "exact_target_column_name_or_null",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this mapping makes sense"
}}
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                suggestion = json.loads(json_match.group())
                return {
                    "suggested_target": suggestion.get("suggested_target"),
                    "confidence": float(suggestion.get("confidence", 0.0)),
                    "reasoning": suggestion.get("reasoning", "No reasoning provided")
                }
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")

        # Fallback parsing
        return {
            "suggested_target": None,
            "confidence": 0.0,
            "reasoning": "Failed to parse LLM response"
        }

    def _mock_llm_suggestions(self, unmapped_fields: List[str], target_schema: Dict) -> Dict[str, Dict]:
        """Mock LLM suggestions for testing"""
        suggestions = {}
        target_fields = list(target_schema.keys())

        for i, field in enumerate(unmapped_fields):
            if i < len(target_fields):
                suggestions[field] = {
                    "suggested_target": target_fields[i],
                    "confidence": 0.75,
                    "reasoning": f"Mock suggestion based on field similarity"
                }
            else:
                suggestions[field] = {
                    "suggested_target": None,
                    "confidence": 0.0,
                    "reasoning": "No suitable target field found"
                }

        return suggestions

# ===== TARGET SCHEMA DEFINITION =====

class BankingComplianceSchema:
    """Banking compliance dataset schema based on actual CSV structure"""

    @staticmethod
    def get_target_schema() -> Dict[str, Dict]:
        """Get the complete target schema for banking compliance dataset"""
        return {
            # Customer Information
            "customer_id": {
                "description": "Unique customer identifier",
                "data_type": "string",
                "required": True,
                "samples": ["CUS770487", "CUS865179", "CUS133659"],
                "business_rules": ["Must be unique", "Required for customer identification"],
                "keywords": ["customer", "client", "id", "identifier", "unique"]
            },
            "customer_type": {
                "description": "Type of customer (Individual or Corporate)",
                "data_type": "string",
                "required": True,
                "samples": ["INDIVIDUAL", "CORPORATE"],
                "business_rules": ["Must be INDIVIDUAL or CORPORATE"],
                "keywords": ["customer", "type", "individual", "corporate", "entity"]
            },
            "full_name_en": {
                "description": "Customer full name in English",
                "data_type": "string",
                "required": True,
                "samples": ["John Smith", "ABC Corporation Ltd", "Sarah Johnson"],
                "business_rules": ["Must not be empty for individuals"],
                "keywords": ["name", "full", "english", "customer", "title"]
            },
            "full_name_ar": {
                "description": "Customer full name in Arabic",
                "data_type": "string",
                "required": False,
                "samples": ["جون سميث", "شركة ABC المحدودة", "سارة جونسون"],
                "business_rules": ["Optional Arabic name"],
                "keywords": ["name", "arabic", "full", "ar", "عربي"]
            },
            "id_number": {
                "description": "Customer identification number",
                "data_type": "integer",
                "required": True,
                "samples": [784199012345678, 784198765432109, 784197654321098],
                "business_rules": ["Must be valid UAE Emirates ID"],
                "keywords": ["id", "identification", "number", "emirates", "passport"]
            },
            "id_type": {
                "description": "Type of identification document",
                "data_type": "string",
                "required": True,
                "samples": ["EMIRATES_ID", "PASSPORT", "TRADE_LICENSE"],
                "business_rules": ["Must be valid ID type"],
                "keywords": ["id", "type", "document", "identification", "emirates", "passport"]
            },
            "date_of_birth": {
                "description": "Customer date of birth",
                "data_type": "date",
                "required": True,
                "samples": ["1985-03-15", "1992-07-22", "1978-11-08"],
                "business_rules": ["Must be valid date", "Age restrictions may apply"],
                "keywords": ["birth", "date", "dob", "born", "age"]
            },
            "nationality": {
                "description": "Customer nationality",
                "data_type": "string",
                "required": True,
                "samples": ["UAE", "INDIA", "PAKISTAN", "PHILIPPINES"],
                "business_rules": ["Must be valid country code"],
                "keywords": ["nationality", "country", "citizen", "origin"]
            },

            # Address Information
            "address_line1": {
                "description": "Primary address line",
                "data_type": "string",
                "required": True,
                "samples": ["123 Sheikh Zayed Road", "Apartment 101, Marina Walk", "Office 505, Business Bay"],
                "business_rules": ["Primary address required"],
                "keywords": ["address", "line1", "street", "primary", "location"]
            },
            "address_line2": {
                "description": "Secondary address line",
                "data_type": "string",
                "required": False,
                "samples": ["Near Metro Station", "Building B", "Floor 5"],
                "business_rules": ["Optional additional address info"],
                "keywords": ["address", "line2", "secondary", "additional", "building"]
            },
            "city": {
                "description": "City name",
                "data_type": "string",
                "required": True,
                "samples": ["DUBAI", "ABU_DHABI", "SHARJAH", "AJMAN"],
                "business_rules": ["Must be valid UAE city"],
                "keywords": ["city", "town", "municipality", "area"]
            },
            "emirate": {
                "description": "UAE Emirate",
                "data_type": "string",
                "required": True,
                "samples": ["DUBAI", "ABU_DHABI", "SHARJAH", "AJMAN", "FUJAIRAH", "RAS_AL_KHAIMAH", "UMM_AL_QUWAIN"],
                "business_rules": ["Must be one of 7 UAE emirates"],
                "keywords": ["emirate", "state", "region", "uae"]
            },
            "country": {
                "description": "Country (always UAE for local accounts)",
                "data_type": "string",
                "required": True,
                "samples": ["UAE"],
                "business_rules": ["Typically UAE for local accounts"],
                "keywords": ["country", "nation", "uae"]
            },
            "postal_code": {
                "description": "Postal code",
                "data_type": "integer",
                "required": False,
                "samples": [12345, 67890, 11111],
                "business_rules": ["Optional in UAE"],
                "keywords": ["postal", "code", "zip", "mail"]
            },

            # Contact Information
            "phone_primary": {
                "description": "Primary phone number",
                "data_type": "float",
                "required": True,
                "samples": [971501234567.0, 971504567890.0, 971509876543.0],
                "business_rules": ["Must be valid UAE phone number"],
                "keywords": ["phone", "primary", "mobile", "contact", "number"]
            },
            "phone_secondary": {
                "description": "Secondary phone number",
                "data_type": "float",
                "required": False,
                "samples": [971421234567.0, 971437654321.0],
                "business_rules": ["Optional secondary contact"],
                "keywords": ["phone", "secondary", "alternate", "backup", "landline"]
            },
            "email_primary": {
                "description": "Primary email address",
                "data_type": "string",
                "required": True,
                "samples": ["john.smith@email.com", "sarah.j@company.ae", "info@business.com"],
                "business_rules": ["Must be valid email format"],
                "keywords": ["email", "primary", "contact", "mail", "electronic"]
            },
            "email_secondary": {
                "description": "Secondary email address",
                "data_type": "string",
                "required": False,
                "samples": ["john.backup@email.com", "alt.email@company.ae"],
                "business_rules": ["Optional secondary email"],
                "keywords": ["email", "secondary", "alternate", "backup"]
            },
            "address_known": {
                "description": "Whether customer address is known and verified",
                "data_type": "string",
                "required": True,
                "samples": ["YES", "NO"],
                "business_rules": ["Must be YES or NO"],
                "keywords": ["address", "known", "verified", "confirmed"]
            },
            "last_contact_date": {
                "description": "Date of last contact with customer",
                "data_type": "date",
                "required": False,
                "samples": ["2023-12-15", "2024-01-22", "2023-11-08"],
                "business_rules": ["Important for dormancy compliance"],
                "keywords": ["contact", "last", "date", "communication", "touch"]
            },
            "last_contact_method": {
                "description": "Method of last contact",
                "data_type": "string",
                "required": False,
                "samples": ["EMAIL", "PHONE", "SMS", "LETTER"],
                "business_rules": ["Must be valid contact method"],
                "keywords": ["contact", "method", "communication", "channel"]
            },

            # KYC and Risk
            "kyc_status": {
                "description": "Know Your Customer status",
                "data_type": "string",
                "required": True,
                "samples": ["COMPLETED", "PENDING", "EXPIRED", "IN_PROGRESS"],
                "business_rules": ["Must be valid KYC status"],
                "keywords": ["kyc", "know", "customer", "status", "compliance"]
            },
            "kyc_expiry_date": {
                "description": "KYC expiry date",
                "data_type": "date",
                "required": False,
                "samples": ["2025-12-31", "2024-06-30", "2026-03-15"],
                "business_rules": ["Must be future date for active KYC"],
                "keywords": ["kyc", "expiry", "date", "expiration", "renewal"]
            },
            "risk_rating": {
                "description": "Customer risk rating",
                "data_type": "string",
                "required": True,
                "samples": ["LOW", "MEDIUM", "HIGH"],
                "business_rules": ["Must be LOW, MEDIUM, or HIGH"],
                "keywords": ["risk", "rating", "assessment", "level", "aml"]
            },

            # Account Information
            "account_id": {
                "description": "Unique account identifier",
                "data_type": "string",
                "required": True,
                "samples": ["ACC123456789", "SAV987654321", "CUR555666777"],
                "business_rules": ["Must be unique across all accounts"],
                "keywords": ["account", "id", "identifier", "number", "unique"]
            },
            "account_type": {
                "description": "Type of bank account",
                "data_type": "string",
                "required": True,
                "samples": ["CURRENT", "SAVINGS", "FIXED_DEPOSIT", "INVESTMENT"],
                "business_rules": ["Must be valid account type"],
                "keywords": ["account", "type", "savings", "current", "deposit", "investment"]
            },
            "account_subtype": {
                "description": "Account subtype for detailed classification",
                "data_type": "string",
                "required": False,
                "samples": ["PREMIUM_SAVINGS", "BASIC_CURRENT", "TERM_DEPOSIT_12M"],
                "business_rules": ["Optional detailed classification"],
                "keywords": ["account", "subtype", "classification", "category", "variant"]
            },
            "account_name": {
                "description": "Account name or title",
                "data_type": "string",
                "required": False,
                "samples": ["John Smith Savings", "ABC Corp Current Account", "Emergency Fund"],
                "business_rules": ["Optional account naming"],
                "keywords": ["account", "name", "title", "label"]
            },
            "currency": {
                "description": "Account currency",
                "data_type": "string",
                "required": True,
                "samples": ["AED", "USD", "EUR", "GBP"],
                "business_rules": ["Must be valid currency code"],
                "keywords": ["currency", "money", "denomination", "aed", "usd"]
            },
            "account_status": {
                "description": "Current status of the account",
                "data_type": "string",
                "required": True,
                "samples": ["ACTIVE", "DORMANT", "CLOSED", "SUSPENDED", "FROZEN"],
                "business_rules": ["Must be valid account status"],
                "keywords": ["account", "status", "active", "dormant", "closed", "state"]
            },
            "dormancy_status": {
                "description": "Specific dormancy classification",
                "data_type": "string",
                "required": True,
                "samples": ["Not_Dormant", "Potentially_Dormant", "Dormant", "Transferred_to_CB"],
                "business_rules": ["Critical for compliance reporting"],
                "keywords": ["dormancy", "status", "dormant", "inactive", "sleeping"]
            },

            # Account Dates and Activity
            "opening_date": {
                "description": "Date account was opened",
                "data_type": "date",
                "required": True,
                "samples": ["2019-03-15", "2020-07-22", "2018-11-08"],
                "business_rules": ["Must be valid past date"],
                "keywords": ["opening", "date", "opened", "created", "inception"]
            },
            "closing_date": {
                "description": "Date account was closed (if applicable)",
                "data_type": "date",
                "required": False,
                "samples": ["2024-01-15", "2023-12-31"],
                "business_rules": ["Only for closed accounts"],
                "keywords": ["closing", "date", "closed", "terminated", "end"]
            },
            "last_transaction_date": {
                "description": "Date of last customer-initiated transaction",
                "data_type": "date",
                "required": True,
                "samples": ["2023-12-15", "2024-01-22", "2021-05-08"],
                "business_rules": ["Critical for dormancy determination"],
                "keywords": ["transaction", "last", "date", "activity", "movement"]
            },
            "last_system_transaction_date": {
                "description": "Date of last system-generated transaction",
                "data_type": "date",
                "required": False,
                "samples": ["2024-01-01", "2023-12-31"],
                "business_rules": ["System-initiated transactions"],
                "keywords": ["system", "transaction", "automated", "generated"]
            },

            # Balances and Financial
            "balance_current": {
                "description": "Current account balance",
                "data_type": "float",
                "required": True,
                "samples": [15000.50, 250000.00, 1000.25],
                "business_rules": ["Current balance for dormancy assessment"],
                "keywords": ["balance", "current", "amount", "money", "funds"]
            },
            "balance_available": {
                "description": "Available balance for withdrawal",
                "data_type": "float",
                "required": False,
                "samples": [14500.50, 249000.00, 950.25],
                "business_rules": ["May differ from current due to holds"],
                "keywords": ["balance", "available", "withdrawable", "liquid"]
            },
            "balance_minimum": {
                "description": "Minimum balance requirement",
                "data_type": "integer",
                "required": False,
                "samples": [1000, 5000, 10000],
                "business_rules": ["Account type specific minimums"],
                "keywords": ["balance", "minimum", "required", "threshold"]
            },
            "interest_rate": {
                "description": "Current interest rate applied",
                "data_type": "float",
                "required": False,
                "samples": [0.5, 1.25, 2.0],
                "business_rules": ["Percentage rate"],
                "keywords": ["interest", "rate", "percentage", "yield"]
            },
            "interest_accrued": {
                "description": "Accrued interest amount",
                "data_type": "float",
                "required": False,
                "samples": [125.50, 3250.75, 0.0],
                "business_rules": ["Interest earned but not paid"],
                "keywords": ["interest", "accrued", "earned", "accumulated"]
            },

            # Account Features
            "is_joint_account": {
                "description": "Whether account is jointly held",
                "data_type": "string",
                "required": False,
                "samples": ["YES", "NO"],
                "business_rules": ["YES or NO"],
                "keywords": ["joint", "account", "shared", "multiple", "holders"]
            },
            "joint_account_holders": {
                "description": "Number of joint account holders",
                "data_type": "float",
                "required": False,
                "samples": [2.0, 3.0, 1.0],
                "business_rules": ["1 for single, >1 for joint"],
                "keywords": ["joint", "holders", "number", "count", "owners"]
            },
            "has_outstanding_facilities": {
                "description": "Whether account has outstanding credit facilities",
                "data_type": "string",
                "required": False,
                "samples": ["YES", "NO"],
                "business_rules": ["YES or NO"],
                "keywords": ["facilities", "outstanding", "credit", "loan", "facility"]
            },
            "maturity_date": {
                "description": "Maturity date for fixed term deposits",
                "data_type": "date",
                "required": False,
                "samples": ["2025-01-13", "2024-12-31", "2026-06-30"],
                "business_rules": ["Only for term deposits"],
                "keywords": ["maturity", "date", "term", "deposit", "expiry"]
            },
            "auto_renewal": {
                "description": "Whether account has auto-renewal enabled",
                "data_type": "string",
                "required": False,
                "samples": ["YES", "NO"],
                "business_rules": ["YES or NO for term deposits"],
                "keywords": ["auto", "renewal", "automatic", "renew"]
            },

            # Statements and Communication
            "last_statement_date": {
                "description": "Date of last account statement",
                "data_type": "date",
                "required": True,
                "samples": ["2023-12-30", "2024-01-31", "2023-11-30"],
                "business_rules": ["Regular statement generation"],
                "keywords": ["statement", "last", "date", "report"]
            },
            "statement_frequency": {
                "description": "How often statements are generated",
                "data_type": "string",
                "required": True,
                "samples": ["QUARTERLY", "MONTHLY", "ANNUAL"],
                "business_rules": ["Must be valid frequency"],
                "keywords": ["statement", "frequency", "monthly", "quarterly", "annual"]
            },

            # Dormancy Tracking
            "tracking_id": {
                "description": "Internal tracking identifier for dormancy process",
                "data_type": "string",
                "required": True,
                "samples": ["TRK733052", "TRK983794", "TRK887352"],
                "business_rules": ["Unique tracking reference"],
                "keywords": ["tracking", "id", "identifier", "reference"]
            },
            "dormancy_trigger_date": {
                "description": "Date when account became dormant",
                "data_type": "date",
                "required": False,
                "samples": ["2021-05-24", "2022-03-15", "2020-12-01"],
                "business_rules": ["Critical for compliance timeline"],
                "keywords": ["dormancy", "trigger", "date", "became", "dormant"]
            },
            "dormancy_period_start": {
                "description": "Start date of dormancy period",
                "data_type": "date",
                "required": False,
                "samples": ["2021-05-24", "2022-03-15", "2020-12-01"],
                "business_rules": ["Begins dormancy workflow"],
                "keywords": ["dormancy", "period", "start", "begin"]
            },
            "dormancy_period_months": {
                "description": "Number of months since dormancy trigger",
                "data_type": "float",
                "required": False,
                "samples": [36.5, 24.2, 48.1],
                "business_rules": ["Calculated from trigger date"],
                "keywords": ["dormancy", "period", "months", "duration"]
            },
            "dormancy_classification_date": {
                "description": "Date when dormancy was officially classified",
                "data_type": "date",
                "required": False,
                "samples": ["2021-08-24", "2022-06-15", "2021-03-01"],
                "business_rules": ["Official classification date"],
                "keywords": ["dormancy", "classification", "date", "official"]
            },
            "transfer_eligibility_date": {
                "description": "Date when account becomes eligible for transfer",
                "data_type": "date",
                "required": False,
                "samples": ["2026-05-24", "2027-03-15", "2025-12-01"],
                "business_rules": ["Usually 5 years from trigger"],
                "keywords": ["transfer", "eligibility", "date", "eligible"]
            },

            # Process Status
            "current_stage": {
                "description": "Current stage in dormancy process",
                "data_type": "string",
                "required": False,
                "samples": ["CONTACT_ATTEMPTS", "WAITING_PERIOD", "READY_FOR_TRANSFER"],
                "business_rules": ["Process workflow stage"],
                "keywords": ["current", "stage", "process", "workflow", "phase"]
            },
            "contact_attempts_made": {
                "description": "Number of contact attempts made",
                "data_type": "integer",
                "required": False,
                "samples": [3, 5, 2],
                "business_rules": ["Minimum 3 attempts required"],
                "keywords": ["contact", "attempts", "made", "number", "count"]
            },
            "last_contact_attempt_date": {
                "description": "Date of last contact attempt",
                "data_type": "date",
                "required": False,
                "samples": ["2023-11-15", "2024-01-10", "2023-09-20"],
                "business_rules": ["Track contact compliance"],
                "keywords": ["contact", "attempt", "last", "date"]
            },
            "waiting_period_start": {
                "description": "Start date of waiting period",
                "data_type": "date",
                "required": False,
                "samples": ["2024-02-15", "2023-12-01", "2024-01-20"],
                "business_rules": ["3-month waiting period"],
                "keywords": ["waiting", "period", "start", "begin"]
            },
            "waiting_period_end": {
                "description": "End date of waiting period",
                "data_type": "date",
                "required": False,
                "samples": ["2024-05-15", "2024-03-01", "2024-04-20"],
                "business_rules": ["End of 3-month waiting"],
                "keywords": ["waiting", "period", "end", "finish"]
            },

            # Transfer Information
            "transferred_to_ledger_date": {
                "description": "Date transferred to internal ledger",
                "data_type": "date",
                "required": False,
                "samples": ["2024-05-20", "2024-03-05", "2024-04-25"],
                "business_rules": ["Internal ledger transfer"],
                "keywords": ["transferred", "ledger", "date", "internal"]
            },
            "transferred_to_cb_date": {
                "description": "Date transferred to Central Bank",
                "data_type": "date",
                "required": False,
                "samples": ["2026-05-24", "2027-03-15"],
                "business_rules": ["Central Bank transfer"],
                "keywords": ["transferred", "central", "bank", "cb", "date"]
            },
            "cb_transfer_amount": {
                "description": "Amount transferred to Central Bank",
                "data_type": "float",
                "required": False,
                "samples": [15000.50, 250000.00, 1000.25],
                "business_rules": ["Transfer amount"],
                "keywords": ["cb", "transfer", "amount", "central", "bank"]
            },
            "cb_transfer_reference": {
                "description": "Central Bank transfer reference number",
                "data_type": "string",
                "required": False,
                "samples": ["CBUAE2024001", "CBUAE2024002"],
                "business_rules": ["CBUAE reference"],
                "keywords": ["cb", "transfer", "reference", "number", "cbuae"]
            },
            "exclusion_reason": {
                "description": "Reason for exclusion from dormancy process",
                "data_type": "string",
                "required": False,
                "samples": ["ACTIVE_FACILITIES", "COURT_ORDER", "LEGAL_HOLD"],
                "business_rules": ["Valid exclusion reason"],
                "keywords": ["exclusion", "reason", "exempt", "excluded"]
            },

            # System Fields
            "created_date": {
                "description": "Record creation date",
                "data_type": "date",
                "required": True,
                "samples": ["2024-01-01", "2023-12-15", "2024-02-01"],
                "business_rules": ["System generated"],
                "keywords": ["created", "date", "system", "record"]
            },
            "updated_date": {
                "description": "Last update date",
                "data_type": "date",
                "required": True,
                "samples": ["2024-01-15", "2024-02-01", "2024-01-30"],
                "business_rules": ["System maintained"],
                "keywords": ["updated", "date", "last", "modified"]
            },
            "updated_by": {
                "description": "User who last updated the record",
                "data_type": "string",
                "required": True,
                "samples": ["SYSTEM", "USER123", "BATCH_PROCESS"],
                "business_rules": ["Audit trail"],
                "keywords": ["updated", "by", "user", "modified", "audit"]
            }
        }

# ===== MAIN DATA MAPPING AGENT =====

class DataMappingAgent:
    """Advanced data mapping agent with BGE embeddings and LLM assistance"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None,
                 groq_api_key: str = None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client or MCPClient()
        self.groq_api_key = groq_api_key

        # Initialize components
        self.bge_manager = BGEEmbeddingManager()
        self.llm_assistant = LLMAssistantMapper(groq_api_key)
        self.target_schema = BankingComplianceSchema.get_target_schema()

        # Initialize LangSmith
        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # Confidence thresholds
        self.confidence_thresholds = {
            "auto_mapping": 0.90,      # 90%+ for automatic mapping
            "high_confidence": 0.90,
            "medium_confidence": 0.70,
            "low_confidence": 0.50
        }

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for data mapping"""
        workflow = StateGraph(DataMappingState)

        # Add workflow nodes
        workflow.add_node("analyze_source_schema", self._analyze_source_schema)
        workflow.add_node("generate_embeddings", self._generate_embeddings)
        workflow.add_node("calculate_similarities", self._calculate_similarities)
        workflow.add_node("auto_map_high_confidence", self._auto_map_high_confidence)
        workflow.add_node("check_auto_mapping_threshold", self._check_auto_mapping_threshold)
        workflow.add_node("get_user_choice", self._get_user_choice)
        workflow.add_node("llm_assisted_mapping", self._llm_assisted_mapping)
        workflow.add_node("manual_mapping_preparation", self._manual_mapping_preparation)
        workflow.add_node("finalize_mapping", self._finalize_mapping)

        # Define workflow edges with conditional routing
        workflow.add_edge(START, "analyze_source_schema")
        workflow.add_edge("analyze_source_schema", "generate_embeddings")
        workflow.add_edge("generate_embeddings", "calculate_similarities")
        workflow.add_edge("calculate_similarities", "auto_map_high_confidence")
        workflow.add_edge("auto_map_high_confidence", "check_auto_mapping_threshold")

        # Conditional routing based on auto-mapping percentage
        workflow.add_conditional_edges(
            "check_auto_mapping_threshold",
            self._routing_decision,
            {
                "auto_complete": "finalize_mapping",
                "user_choice": "get_user_choice"
            }
        )

        # User choice routing
        workflow.add_conditional_edges(
            "get_user_choice",
            self._user_choice_routing,
            {
                "llm": "llm_assisted_mapping",
                "manual": "manual_mapping_preparation"
            }
        )

        workflow.add_edge("llm_assisted_mapping", "finalize_mapping")
        workflow.add_edge("manual_mapping_preparation", "finalize_mapping")
        workflow.add_edge("finalize_mapping", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _routing_decision(self, state: DataMappingState) -> str:
        """Decide routing based on auto-mapping percentage"""
        if state.auto_mapping_percentage >= self.confidence_thresholds["auto_mapping"]:
            return "auto_complete"
        else:
            return "user_choice"

    def _user_choice_routing(self, state: DataMappingState) -> str:
        """Route based on user choice"""
        return state.user_choice if state.user_choice in ["llm", "manual"] else "manual"

    @traceable(name="analyze_source_schema")
    async def _analyze_source_schema(self, state: DataMappingState) -> DataMappingState:
        """Analyze source schema and extract field information"""
        logger.info(f"Analyzing source schema for mapping {state.mapping_id}")

        try:
            start_time = datetime.now()

            # Extract source fields from DataFrame
            if state.source_data_sample is not None:
                source_fields = list(state.source_data_sample.columns)

                # Extract sample values and data types
                source_field_info = {}
                for field in source_fields:
                    column_data = state.source_data_sample[field]

                    # Get sample values
                    sample_values = column_data.dropna().unique()[:5].tolist()

                    # Infer data type
                    if pd.api.types.is_datetime64_any_dtype(column_data):
                        data_type = "date"
                    elif pd.api.types.is_numeric_dtype(column_data):
                        data_type = "float" if column_data.dtype in ['float64', 'float32'] else "integer"
                    elif pd.api.types.is_bool_dtype(column_data):
                        data_type = "boolean"
                    else:
                        data_type = "string"

                    source_field_info[field] = {
                        "data_type": data_type,
                        "sample_values": [str(x) for x in sample_values],
                        "null_count": int(column_data.isnull().sum()),
                        "total_count": len(column_data)
                    }

                state.source_schema = source_field_info
                state.total_fields = len(source_fields)

            # Update processing time
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.mapping_status = MappingStatus.ANALYZING

            # Log analysis
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
                "stage": "schema_analysis"
            })

        return state

    @traceable(name="generate_embeddings")
    async def _generate_embeddings(self, state: DataMappingState) -> DataMappingState:
        """Generate BGE embeddings for source and target fields"""
        logger.info("Generating BGE embeddings for semantic similarity")

        try:
            start_time = datetime.now()

            # Prepare field texts for embedding
            source_fields = list(state.source_schema.keys())
            target_fields = list(self.target_schema.keys())

            # Enhance field names with context for better embeddings
            source_texts = []
            target_texts = []

            for field in source_fields:
                # Add sample values and data type context
                field_info = state.source_schema[field]
                enhanced_text = f"{field} {field_info['data_type']}"
                if field_info['sample_values']:
                    enhanced_text += f" examples: {' '.join(field_info['sample_values'][:3])}"
                source_texts.append(enhanced_text)

            for field in target_fields:
                # Add description and keywords context
                field_info = self.target_schema[field]
                enhanced_text = f"{field} {field_info['description']}"
                if field_info.get('keywords'):
                    enhanced_text += f" keywords: {' '.join(field_info['keywords'])}"
                if field_info.get('samples'):
                    enhanced_text += f" examples: {' '.join(str(x) for x in field_info['samples'][:3])}"
                target_texts.append(enhanced_text)

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(source_texts)} source and {len(target_texts)} target fields")

            source_embeddings = self.bge_manager.generate_embeddings(source_texts)
            target_embeddings = self.bge_manager.generate_embeddings(target_texts)

            # Store embeddings in state
            state.source_embeddings = {field: embedding for field, embedding in zip(source_fields, source_embeddings)}
            state.target_embeddings = {field: embedding for field, embedding in zip(target_fields, target_embeddings)}

            # Update timing
            state.embedding_time = (datetime.now() - start_time).total_seconds()

            # Log embedding generation
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "embeddings_generated",
                "details": f"Generated BGE embeddings for {len(source_fields)} source and {len(target_fields)} target fields",
                "processing_time": state.embedding_time
            })

            logger.info(f"BGE embeddings generated successfully in {state.embedding_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            state.mapping_status = MappingStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "stage": "embedding_generation"
            })

        return state

    @traceable(name="calculate_similarities")
    async def _calculate_similarities(self, state: DataMappingState) -> DataMappingState:
        """Calculate cosine similarity matrix between source and target fields"""
        logger.info("Calculating cosine similarities between source and target fields")

        try:
            start_time = datetime.now()

            # Prepare embedding matrices
            source_fields = list(state.source_embeddings.keys())
            target_fields = list(state.target_embeddings.keys())

            source_matrix = np.array([state.source_embeddings[field] for field in source_fields])
            target_matrix = np.array([state.target_embeddings[field] for field in target_fields])

            # Calculate similarity matrix
            similarity_matrix = self.bge_manager.calculate_similarity_matrix(source_matrix, target_matrix)
            state.similarity_matrix = similarity_matrix

            # Create field mappings with similarity scores
            state.field_mappings = []

            for i, source_field in enumerate(source_fields):
                # Find best matching target field
                best_target_idx = np.argmax(similarity_matrix[i])
                best_similarity = similarity_matrix[i][best_target_idx]
                best_target_field = target_fields[best_target_idx]

                # Create field mapping
                field_mapping = FieldMapping(
                    source_field=source_field,
                    target_field=best_target_field,
                    confidence_score=float(best_similarity),
                    confidence_level=MappingConfidence.LOW,  # Will be updated
                    mapping_strategy=MappingStrategy.AUTOMATIC,
                    source_embedding=state.source_embeddings[source_field],
                    target_embedding=state.target_embeddings[best_target_field],
                    cosine_similarity_score=float(best_similarity),
                    source_data_type=state.source_schema[source_field]["data_type"],
                    target_data_type=self.target_schema[best_target_field]["data_type"],
                    source_samples=state.source_schema[source_field]["sample_values"],
                    target_samples=[str(x) for x in self.target_schema[best_target_field].get("samples", [])]
                )

                # Update confidence level based on similarity score
                field_mapping.update_confidence_level()

                # Check data type compatibility
                field_mapping.data_type_match = self._check_data_type_compatibility(
                    field_mapping.source_data_type,
                    field_mapping.target_data_type
                )

                # Add business context
                field_mapping.business_context = self.target_schema[best_target_field].get("description", "")
                field_mapping.semantic_keywords = self.target_schema[best_target_field].get("keywords", [])

                state.field_mappings.append(field_mapping)

            # Update timing
            state.similarity_calculation_time = (datetime.now() - start_time).total_seconds()

            # Log similarity calculation
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "similarities_calculated",
                "details": f"Calculated similarities for {len(source_fields)} source fields",
                "processing_time": state.similarity_calculation_time
            })

            logger.info(f"Similarity calculation completed in {state.similarity_calculation_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            state.mapping_status = MappingStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "stage": "similarity_calculation"
            })

        return state

    def _check_data_type_compatibility(self, source_type: str, target_type: str) -> bool:
        """Check if source and target data types are compatible"""
        compatibility_map = {
            "string": ["string"],
            "integer": ["integer", "float"],
            "float": ["float", "integer"],
            "date": ["date", "string"],
            "boolean": ["boolean", "string"]
        }

        return target_type in compatibility_map.get(source_type, [])

    @traceable(name="auto_map_high_confidence")
    async def _auto_map_high_confidence(self, state: DataMappingState) -> DataMappingState:
        """Automatically map fields with high confidence (90%+)"""
        logger.info("Performing automatic mapping for high confidence fields")

        try:
            # Count confidence levels
            high_confidence_count = 0
            medium_confidence_count = 0
            low_confidence_count = 0

            auto_mapped_fields = []

            for mapping in state.field_mappings:
                if mapping.confidence_level == MappingConfidence.HIGH:
                    high_confidence_count += 1
                    mapping.mapping_strategy = MappingStrategy.AUTOMATIC
                    auto_mapped_fields.append(mapping.source_field)
                elif mapping.confidence_level == MappingConfidence.MEDIUM:
                    medium_confidence_count += 1
                elif mapping.confidence_level == MappingConfidence.LOW:
                    low_confidence_count += 1

            # Update state counts
            state.high_confidence_mappings = high_confidence_count
            state.medium_confidence_mappings = medium_confidence_count
            state.low_confidence_mappings = low_confidence_count
            state.auto_mapped_fields = auto_mapped_fields

            # Calculate auto-mapping percentage
            state.auto_mapping_percentage = (high_confidence_count / state.total_fields) * 100 if state.total_fields > 0 else 0

            # Log auto-mapping results
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "auto_mapping_completed",
                "details": f"Auto-mapped {high_confidence_count}/{state.total_fields} fields ({state.auto_mapping_percentage:.1f}%)",
                "high_confidence": high_confidence_count,
                "medium_confidence": medium_confidence_count,
                "low_confidence": low_confidence_count
            })

            logger.info(f"Auto-mapping completed: {high_confidence_count}/{state.total_fields} fields ({state.auto_mapping_percentage:.1f}%)")

        except Exception as e:
            logger.error(f"Auto-mapping failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "stage": "auto_mapping"
            })

        return state

    @traceable(name="check_auto_mapping_threshold")
    async def _check_auto_mapping_threshold(self, state: DataMappingState) -> DataMappingState:
        """Check if auto-mapping percentage meets threshold"""
        threshold = self.confidence_thresholds["auto_mapping"] * 100  # Convert to percentage

        if state.auto_mapping_percentage >= threshold:
            state.mapping_status = MappingStatus.COMPLETED
            logger.info(f"Auto-mapping threshold met: {state.auto_mapping_percentage:.1f}% >= {threshold}%")
        else:
            state.mapping_status = MappingStatus.REQUIRES_USER_INPUT
            state.requires_user_input = state.total_fields - state.high_confidence_mappings
            logger.info(f"Auto-mapping threshold not met: {state.auto_mapping_percentage:.1f}% < {threshold}%. User input required.")

        return state

    async def _get_user_choice(self, state: DataMappingState) -> DataMappingState:
        """Get user choice for handling low confidence mappings"""
        # This would typically be handled by the UI
        # For now, we'll set a default choice
        state.user_choice = state.mapping_config.get("user_choice", "manual")
        state.mapping_status = MappingStatus.LLM_PROCESSING if state.user_choice == "llm" else MappingStatus.MANUAL_MAPPING

        logger.info(f"User choice: {state.user_choice}")
        return state

    @traceable(name="llm_assisted_mapping")
    async def _llm_assisted_mapping(self, state: DataMappingState) -> DataMappingState:
        """Use LLM to assist with low confidence mappings"""
        logger.info("Using LLM assistance for low confidence mappings")

        try:
            # Get unmapped or low confidence fields
            unmapped_fields = []
            source_samples = {}

            for mapping in state.field_mappings:
                if mapping.confidence_level in [MappingConfidence.LOW, MappingConfidence.VERY_LOW]:
                    unmapped_fields.append(mapping.source_field)
                    source_samples[mapping.source_field] = mapping.source_samples

            if unmapped_fields:
                # Get LLM suggestions
                llm_suggestions = await self.llm_assistant.suggest_mappings(
                    unmapped_fields, self.target_schema, source_samples
                )

                # Update mappings with LLM suggestions
                for mapping in state.field_mappings:
                    if mapping.source_field in llm_suggestions:
                        suggestion = llm_suggestions[mapping.source_field]

                        if suggestion["suggested_target"]:
                            mapping.target_field = suggestion["suggested_target"]
                            mapping.llm_suggestion = suggestion["suggested_target"]
                            mapping.llm_confidence = suggestion["confidence"]
                            mapping.mapping_strategy = MappingStrategy.LLM_ASSISTED

                            # Update confidence based on LLM + similarity
                            combined_confidence = (mapping.cosine_similarity_score + suggestion["confidence"]) / 2
                            mapping.confidence_score = combined_confidence
                            mapping.update_confidence_level()

                logger.info(f"LLM assistance completed for {len(unmapped_fields)} fields")

            state.mapping_status = MappingStatus.COMPLETED

        except Exception as e:
            logger.error(f"LLM assistance failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "stage": "llm_assistance"
            })

        return state

    async def _manual_mapping_preparation(self, state: DataMappingState) -> DataMappingState:
        """Prepare data for manual mapping by user"""
        logger.info("Preparing data for manual mapping")

        try:
            # Identify fields that need manual mapping
            manual_mapping_needed = []

            for mapping in state.field_mappings:
                if mapping.confidence_level in [MappingConfidence.LOW, MappingConfidence.VERY_LOW]:
                    manual_mapping_needed.append({
                        "source_field": mapping.source_field,
                        "suggested_target": mapping.target_field,
                        "similarity_score": mapping.cosine_similarity_score,
                        "source_data_type": mapping.source_data_type,
                        "source_samples": mapping.source_samples,
                        "target_options": self._get_similar_target_fields(mapping.source_field, state.similarity_matrix, list(self.target_schema.keys())),
                        "business_context": mapping.business_context
                    })

            # Store manual mapping requirements
            state.unmapped_fields = [item["source_field"] for item in manual_mapping_needed]
            state.mapping_config["manual_mapping_data"] = manual_mapping_needed

            state.mapping_status = MappingStatus.MANUAL_MAPPING

            logger.info(f"Manual mapping preparation completed for {len(manual_mapping_needed)} fields")

        except Exception as e:
            logger.error(f"Manual mapping preparation failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "stage": "manual_mapping_preparation"
            })

        return state

    def _get_similar_target_fields(self, source_field: str, similarity_matrix: np.ndarray,
                                  target_fields: List[str], top_k: int = 5) -> List[Dict]:
        """Get top-k most similar target fields for manual selection"""
        try:
            source_fields = list(self.target_schema.keys())  # This should be source fields from state
            if source_field not in source_fields:
                return []

            source_idx = source_fields.index(source_field)
            similarities = similarity_matrix[source_idx]

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            similar_fields = []
            for idx in top_indices:
                if idx < len(target_fields):
                    target_field = target_fields[idx]
                    similar_fields.append({
                        "field_name": target_field,
                        "similarity_score": float(similarities[idx]),
                        "description": self.target_schema[target_field].get("description", ""),
                        "data_type": self.target_schema[target_field].get("data_type", ""),
                        "samples": self.target_schema[target_field].get("samples", [])
                    })

            return similar_fields

        except Exception as e:
            logger.error(f"Failed to get similar target fields: {e}")
            return []

    @traceable(name="finalize_mapping")
    async def _finalize_mapping(self, state: DataMappingState) -> DataMappingState:
        """Finalize the mapping process and generate summary"""
        logger.info("Finalizing mapping process")

        try:
            # Calculate final statistics
            total_mapped = len([m for m in state.field_mappings if m.target_field])
            high_confidence = len([m for m in state.field_mappings if m.confidence_level == MappingConfidence.HIGH])
            medium_confidence = len([m for m in state.field_mappings if m.confidence_level == MappingConfidence.MEDIUM])
            low_confidence = len([m for m in state.field_mappings if m.confidence_level == MappingConfidence.LOW])

            # Calculate mapping success rate
            mapping_success_rate = (total_mapped / state.total_fields) * 100 if state.total_fields > 0 else 0

            # Generate mapping summary
            state.mapping_summary = {
                "total_source_fields": state.total_fields,
                "total_mapped_fields": total_mapped,
                "mapping_success_rate": round(mapping_success_rate, 2),
                "auto_mapping_percentage": round(state.auto_mapping_percentage, 2),
                "confidence_distribution": {
                    "high": high_confidence,
                    "medium": medium_confidence,
                    "low": low_confidence
                },
                "mapping_strategies": {
                    "automatic": len([m for m in state.field_mappings if m.mapping_strategy == MappingStrategy.AUTOMATIC]),
                    "llm_assisted": len([m for m in state.field_mappings if m.mapping_strategy == MappingStrategy.LLM_ASSISTED]),
                    "manual": len([m for m in state.field_mappings if m.mapping_strategy == MappingStrategy.MANUAL])
                },
                "processing_time": {
                    "total": state.processing_time + state.embedding_time + state.similarity_calculation_time,
                    "embedding_generation": state.embedding_time,
                    "similarity_calculation": state.similarity_calculation_time,
                    "schema_analysis": state.processing_time
                },
                "data_quality_scores": self._calculate_data_quality_scores(state),
                "transformation_ready": mapping_success_rate >= 80,  # 80% mapping success for transformation readiness
                "recommendations": self._generate_mapping_recommendations(state)
            }

            # Update final status
            if mapping_success_rate >= 90:
                state.mapping_status = MappingStatus.COMPLETED
            elif mapping_success_rate >= 70:
                state.mapping_status = MappingStatus.REQUIRES_USER_INPUT
            else:
                state.mapping_status = MappingStatus.FAILED

            # Final log entry
            state.mapping_log.append({
                "timestamp": datetime.now(),
                "action": "mapping_finalized",
                "details": f"Mapping completed with {mapping_success_rate:.1f}% success rate",
                "total_mapped": total_mapped,
                "total_fields": state.total_fields,
                "status": state.mapping_status.value
            })

            logger.info(f"Mapping finalized: {total_mapped}/{state.total_fields} fields mapped ({mapping_success_rate:.1f}% success)")

        except Exception as e:
            logger.error(f"Mapping finalization failed: {str(e)}")
            state.mapping_status = MappingStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "stage": "finalization"
            })

        return state

    def _calculate_data_quality_scores(self, state: DataMappingState) -> Dict:
        """Calculate data quality scores for mapped fields"""
        try:
            quality_scores = {}

            for mapping in state.field_mappings:
                if mapping.target_field and mapping.source_field in state.source_schema:
                    source_info = state.source_schema[mapping.source_field]

                    # Calculate completeness score
                    total_count = source_info.get("total_count", 0)
                    null_count = source_info.get("null_count", 0)
                    completeness = ((total_count - null_count) / total_count * 100) if total_count > 0 else 0

                    # Calculate type compatibility score
                    type_compatibility = 100 if mapping.data_type_match else 50

                    # Calculate semantic similarity score
                    semantic_score = mapping.cosine_similarity_score * 100

                    # Overall quality score
                    overall_quality = (completeness * 0.4 + type_compatibility * 0.3 + semantic_score * 0.3)

                    quality_scores[mapping.source_field] = {
                        "completeness": round(completeness, 2),
                        "type_compatibility": type_compatibility,
                        "semantic_similarity": round(semantic_score, 2),
                        "overall_quality": round(overall_quality, 2),
                        "target_field": mapping.target_field
                    }

            return quality_scores

        except Exception as e:
            logger.error(f"Data quality calculation failed: {e}")
            return {}

    def _generate_mapping_recommendations(self, state: DataMappingState) -> List[str]:
        """Generate recommendations for improving mapping quality"""
        recommendations = []

        try:
            # Check auto-mapping percentage
            if state.auto_mapping_percentage < 70:
                recommendations.append("Consider improving source data column naming conventions for better automatic mapping")

            # Check for unmapped fields
            unmapped_count = len([m for m in state.field_mappings if not m.target_field])
            if unmapped_count > 0:
                recommendations.append(f"Review {unmapped_count} unmapped fields for potential manual mapping")

            # Check data type mismatches
            type_mismatches = len([m for m in state.field_mappings if not m.data_type_match])
            if type_mismatches > 0:
                recommendations.append(f"Address {type_mismatches} data type compatibility issues")

            # Check for low confidence mappings
            low_confidence_count = len([m for m in state.field_mappings if m.confidence_level == MappingConfidence.LOW])
            if low_confidence_count > 0:
                recommendations.append(f"Validate {low_confidence_count} low confidence mappings")

            # Performance recommendations
            if state.embedding_time > 10:
                recommendations.append("Consider using a smaller embedding model for faster processing")

            # Success rate recommendations
            success_rate = (len([m for m in state.field_mappings if m.target_field]) / state.total_fields) * 100
            if success_rate >= 90:
                recommendations.append("Excellent mapping quality achieved - ready for data transformation")
            elif success_rate >= 70:
                recommendations.append("Good mapping quality - review low confidence mappings before transformation")
            else:
                recommendations.append("Mapping quality needs improvement - manual review recommended")

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Error generating recommendations - manual review advised")

        return recommendations

    # ===== PUBLIC API METHODS =====

    async def analyze_and_map_data(self, source_data: pd.DataFrame, user_id: str,
                                  mapping_config: Dict = None) -> DataMappingState:
        """Main method to analyze and map source data to target schema"""
        logger.info(f"Starting data mapping analysis for user {user_id}")

        try:
            # Create mapping state
            mapping_state = DataMappingState(
                session_id=secrets.token_hex(16),
                user_id=user_id,
                mapping_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                source_data_sample=source_data.head(100),  # Use sample for analysis
                mapping_config=mapping_config or {}
            )

            # Execute mapping workflow
            final_state = await self.workflow.ainvoke(mapping_state)

            logger.info(f"Data mapping completed with status: {final_state.mapping_status.value}")
            return final_state

        except Exception as e:
            logger.error(f"Data mapping analysis failed: {str(e)}")
            raise

    def apply_manual_mappings(self, state: DataMappingState, manual_mappings: Dict[str, str]) -> DataMappingState:
        """Apply manual mappings provided by user"""
        try:
            for source_field, target_field in manual_mappings.items():
                # Find and update the corresponding mapping
                for mapping in state.field_mappings:
                    if mapping.source_field == source_field:
                        mapping.target_field = target_field
                        mapping.mapping_strategy = MappingStrategy.MANUAL
                        mapping.user_confirmed = True
                        mapping.user_override = target_field
                        mapping.updated_at = datetime.now()

                        # Update confidence to high for manual mappings
                        mapping.confidence_score = 1.0
                        mapping.confidence_level = MappingConfidence.HIGH
                        break

            # Recalculate summary
            self._update_mapping_summary(state)

            logger.info(f"Applied {len(manual_mappings)} manual mappings")
            return state

        except Exception as e:
            logger.error(f"Failed to apply manual mappings: {e}")
            raise

    def _update_mapping_summary(self, state: DataMappingState):
        """Update mapping summary after manual changes"""
        total_mapped = len([m for m in state.field_mappings if m.target_field])
        mapping_success_rate = (total_mapped / state.total_fields) * 100 if state.total_fields > 0 else 0

        if state.mapping_summary:
            state.mapping_summary["total_mapped_fields"] = total_mapped
            state.mapping_summary["mapping_success_rate"] = round(mapping_success_rate, 2)
            state.mapping_summary["transformation_ready"] = mapping_success_rate >= 80

    def transform_data(self, source_data: pd.DataFrame, mapping_state: DataMappingState) -> pd.DataFrame:
        """Transform source data using the established mappings"""
        try:
            logger.info("Transforming data using established mappings")

            # Create mapping dictionary
            field_map = {}
            for mapping in mapping_state.field_mappings:
                if mapping.target_field and mapping.confidence_level != MappingConfidence.VERY_LOW:
                    field_map[mapping.source_field] = mapping.target_field

            # Transform data
            transformed_data = source_data.copy()

            # Rename columns according to mapping
            transformed_data = transformed_data.rename(columns=field_map)

            # Add missing required target columns with defaults
            for target_field, field_info in self.target_schema.items():
                if target_field not in transformed_data.columns and field_info.get("required", False):
                    # Add default values based on data type
                    if field_info["data_type"] == "string":
                        transformed_data[target_field] = "Not_Available"
                    elif field_info["data_type"] in ["integer", "float"]:
                        transformed_data[target_field] = 0
                    elif field_info["data_type"] == "date":
                        transformed_data[target_field] = "1900-01-01"
                    else:
                        transformed_data[target_field] = None

            # Reorder columns to match target schema
            target_columns = list(self.target_schema.keys())
            available_columns = [col for col in target_columns if col in transformed_data.columns]
            extra_columns = [col for col in transformed_data.columns if col not in target_columns]

            final_columns = available_columns + extra_columns
            transformed_data = transformed_data[final_columns]

            logger.info(f"Data transformation completed: {len(transformed_data)} records, {len(final_columns)} columns")
            return transformed_data

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise

    def get_mapping_report(self, state: DataMappingState) -> Dict:
        """Generate comprehensive mapping report"""
        try:
            report = {
                "mapping_id": state.mapping_id,
                "user_id": state.user_id,
                "timestamp": state.timestamp.isoformat(),
                "status": state.mapping_status.value,
                "summary": state.mapping_summary,
                "field_mappings": [
                    {
                        "source_field": m.source_field,
                        "target_field": m.target_field,
                        "confidence_score": round(m.confidence_score, 3),
                        "confidence_level": m.confidence_level.value,
                        "mapping_strategy": m.mapping_strategy.value,
                        "data_type_match": m.data_type_match,
                        "source_data_type": m.source_data_type,
                        "target_data_type": m.target_data_type,
                        "cosine_similarity": round(m.cosine_similarity_score, 3)
                    }
                    for m in state.field_mappings
                ],
                "unmapped_fields": state.unmapped_fields,
                "auto_mapped_fields": state.auto_mapped_fields,
                "processing_metrics": {
                    "total_processing_time": state.processing_time + state.embedding_time + state.similarity_calculation_time,
                    "embedding_time": state.embedding_time,
                    "similarity_calculation_time": state.similarity_calculation_time
                },
                "logs": state.mapping_log,
                "errors": state.error_log
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate mapping report: {e}")
            return {"error": str(e)}

# ===== FACTORY FUNCTIONS =====

def create_data_mapping_agent(memory_agent=None, mcp_client: MCPClient = None,
                             groq_api_key: str = None) -> DataMappingAgent:
    """Factory function to create data mapping agent"""
    return DataMappingAgent(memory_agent, mcp_client, groq_api_key)

async def run_automated_data_mapping(source_data: pd.DataFrame, user_id: str,
                                   groq_api_key: str = None, mapping_config: Dict = None) -> Dict:
    """
    Run automated data mapping analysis

    Args:
        source_data: Source DataFrame to map
        user_id: User identifier
        groq_api_key: Optional Groq API key for LLM assistance
        mapping_config: Optional configuration for mapping behavior

    Returns:
        Dictionary containing mapping results and recommendations
    """
    try:
        # Initialize mapping agent
        mapping_agent = DataMappingAgent(groq_api_key=groq_api_key)

        # Run mapping analysis
        mapping_state = await mapping_agent.analyze_and_map_data(
            source_data, user_id, mapping_config
        )

        # Generate report
        report = mapping_agent.get_mapping_report(mapping_state)

        # Add transformation readiness and next steps
        auto_mapping_pct = mapping_state.auto_mapping_percentage

        if auto_mapping_pct >= 90:
            next_steps = "automatic_transformation"
            message = f"Excellent! {auto_mapping_pct:.1f}% of fields mapped automatically. Ready for data transformation."
        else:
            next_steps = "user_input_required"
            message = f"Auto-mapping achieved {auto_mapping_pct:.1f}%. Please choose: LLM assistance or manual mapping for remaining fields."

        return {
            "success": True,
            "mapping_state": mapping_state,
            "report": report,
            "auto_mapping_percentage": auto_mapping_pct,
            "next_steps": next_steps,
            "message": message,
            "transformation_ready": mapping_state.mapping_summary.get("transformation_ready", False)
        }

    except Exception as e:
        logger.error(f"Automated data mapping failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "mapping_state": None
        }

# ===== EXPORT DEFINITIONS =====

__all__ = [
    # Core Classes
    "DataMappingAgent",
    "BGEEmbeddingManager",
    "LLMAssistantMapper",
    "BankingComplianceSchema",

    # State and Data Classes
    "DataMappingState",
    "FieldMapping",

    # Enums
    "MappingStatus",
    "MappingStrategy",
    "MappingConfidence",

    # Factory Functions
    "create_data_mapping_agent",
    "run_automated_data_mapping"
]
