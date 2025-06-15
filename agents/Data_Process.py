"""
data_process.py - Enhanced Data Processing Agent
Integrated with Hybrid Memory Agent, LangGraph, and MCP Tools
Handles banking data validation, quality assessment, and preparation for dormancy analysis
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import secrets
from pathlib import Path
import aiofiles
import openpyxl
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable, Client as LangSmithClient
from langsmith.wrappers import wrap_openai

# MCP imports
from mcp_client import MCPClient
from mcp_implementation import MCPServer

# Pydantic for data validation
from pydantic import BaseModel, validator, Field
from typing_extensions import Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Data Processing States and Models
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"
    PARTIAL_SUCCESS = "partial_success"


class DataQualityLevel(Enum):
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 70-89%
    FAIR = "fair"  # 50-69%
    POOR = "poor"  # Below 50%


@dataclass
class DataProcessingState:
    """Enhanced state for data processing workflow"""
    session_id: str
    user_id: str
    processing_id: str
    timestamp: datetime

    # Input data
    raw_data: Optional[Dict] = None
    data_source: Optional[str] = None
    file_metadata: Optional[Dict] = None

    # Processing results
    processed_data: Optional[Dict] = None
    validation_results: Optional[Dict] = None
    quality_metrics: Optional[Dict] = None
    data_schema: Optional[Dict] = None

    # Status tracking
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    quality_level: Optional[DataQualityLevel] = None
    quality_score: float = 0.0
    error_count: int = 0
    warning_count: int = 0

    # Memory context
    memory_context: Dict = None
    retrieved_patterns: Dict = None

    # Audit and logging
    processing_log: List[Dict] = None
    error_log: List[Dict] = None
    performance_metrics: Dict = None

    def __post_init__(self):
        if self.memory_context is None:
            self.memory_context = {}
        if self.retrieved_patterns is None:
            self.retrieved_patterns = {}
        if self.processing_log is None:
            self.processing_log = []
        if self.error_log is None:
            self.error_log = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


# Data Schema Validation Models
class AccountSchema(BaseModel):
    """Pydantic model for account data validation"""
    Account_ID: str = Field(..., min_length=1, max_length=50)
    Account_Type: str = Field(..., pattern=r'^(Current|Saving|Call|Fixed|Term|Investment|safe_deposit_box).*')
    Current_Balance: Optional[float] = Field(None, ge=0)
    Date_Last_Cust_Initiated_Activity: Optional[str] = None
    Date_Last_Customer_Communication_Any_Type: Optional[str] = None
    Customer_Has_Active_Liability_Account: Optional[str] = Field(None, pattern=r'^(yes|no|true|false|1|0)$')

    @validator('Current_Balance')
    def validate_balance(cls, v):
        if v is not None and (v < 0 or v > 1e12):  # Reasonable upper limit
            raise ValueError('Balance must be between 0 and 1 trillion')
        return v


class PaymentInstrumentSchema(BaseModel):
    """Schema for payment instruments (cheques, drafts, etc.)"""
    Account_ID: str = Field(..., min_length=1)
    Account_Type: str = Field(..., pattern=r'.*(Bankers_Cheque|Bank_Draft|Cashier_Order).*')
    Unclaimed_Item_Trigger_Date: Optional[str] = None
    Unclaimed_Item_Amount: Optional[float] = Field(None, gt=0)


# Data Quality Analyzer
class DataQualityAnalyzer:
    """Advanced data quality analysis with ML-based patterns"""

    def __init__(self, memory_agent):
        self.memory_agent = memory_agent
        self.quality_thresholds = {
            "completeness": 0.8,
            "accuracy": 0.9,
            "consistency": 0.85,
            "validity": 0.9,
            "timeliness": 0.7
        }

    @traceable(name="analyze_data_quality")
    async def analyze_quality(self, df: pd.DataFrame, schema_type: str = "account") -> Dict:
        """Comprehensive data quality analysis"""
        try:
            start_time = datetime.now()

            quality_metrics = {
                "completeness": await self._assess_completeness(df),
                "accuracy": await self._assess_accuracy(df, schema_type),
                "consistency": await self._assess_consistency(df),
                "validity": await self._assess_validity(df, schema_type),
                "timeliness": await self._assess_timeliness(df),
                "uniqueness": await self._assess_uniqueness(df)
            }

            # Calculate overall quality score
            weights = {"completeness": 0.2, "accuracy": 0.25, "consistency": 0.2,
                       "validity": 0.2, "timeliness": 0.15}

            overall_score = sum(quality_metrics[metric] * weights[metric]
                                for metric in weights.keys())

            # Determine quality level
            if overall_score >= 0.9:
                quality_level = DataQualityLevel.EXCELLENT
            elif overall_score >= 0.7:
                quality_level = DataQualityLevel.GOOD
            elif overall_score >= 0.5:
                quality_level = DataQualityLevel.FAIR
            else:
                quality_level = DataQualityLevel.POOR

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "overall_score": round(overall_score, 4),
                "quality_level": quality_level.value,
                "metrics": quality_metrics,
                "processing_time_seconds": processing_time,
                "record_count": len(df),
                "recommendations": await self._generate_quality_recommendations(quality_metrics)
            }

        except Exception as e:
            logger.error(f"Data quality analysis failed: {str(e)}")
            return {
                "overall_score": 0.0,
                "quality_level": DataQualityLevel.POOR.value,
                "error": str(e)
            }

    async def _assess_completeness(self, df: pd.DataFrame) -> float:
        """Assess data completeness"""
        if df.empty:
            return 0.0

        total_cells = df.size
        non_null_cells = df.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0

    async def _assess_accuracy(self, df: pd.DataFrame, schema_type: str) -> float:
        """Assess data accuracy based on business rules"""
        if df.empty:
            return 0.0

        accuracy_checks = []

        # Check date formats
        date_columns = [col for col in df.columns if 'Date' in col or 'date' in col]
        for col in date_columns:
            if col in df.columns:
                valid_dates = pd.to_datetime(df[col], errors='coerce').notna().sum()
                total_dates = df[col].notna().sum()
                if total_dates > 0:
                    accuracy_checks.append(valid_dates / total_dates)

        # Check balance values
        if 'Current_Balance' in df.columns:
            valid_balances = (pd.to_numeric(df['Current_Balance'], errors='coerce') >= 0).sum()
            total_balances = df['Current_Balance'].notna().sum()
            if total_balances > 0:
                accuracy_checks.append(valid_balances / total_balances)

        return np.mean(accuracy_checks) if accuracy_checks else 1.0

    async def _assess_consistency(self, df: pd.DataFrame) -> float:
        """Assess data consistency"""
        if df.empty:
            return 0.0

        consistency_score = 1.0

        # Check Account_Type consistency
        if 'Account_Type' in df.columns:
            valid_types = df['Account_Type'].str.contains(
                r'Current|Saving|Call|Fixed|Term|Investment|safe_deposit_box',
                case=False, na=False
            ).sum()
            total_types = df['Account_Type'].notna().sum()
            if total_types > 0:
                consistency_score *= valid_types / total_types

        return consistency_score

    async def _assess_validity(self, df: pd.DataFrame, schema_type: str) -> float:
        """Assess data validity against schema"""
        if df.empty:
            return 0.0

        validity_checks = []

        # Check required fields based on schema type
        if schema_type == "account":
            required_fields = ['Account_ID', 'Account_Type']
        else:
            required_fields = ['Account_ID']

        for field in required_fields:
            if field in df.columns:
                valid_count = df[field].notna().sum()
                validity_checks.append(valid_count / len(df))

        return np.mean(validity_checks) if validity_checks else 0.0

    async def _assess_timeliness(self, df: pd.DataFrame) -> float:
        """Assess data timeliness"""
        if df.empty:
            return 0.0

        # Check if activity dates are within reasonable timeframe
        date_columns = [col for col in df.columns if 'Date' in col]
        timeliness_scores = []

        current_date = datetime.now()

        for col in date_columns:
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors='coerce')
                valid_dates = dates.dropna()

                if not valid_dates.empty:
                    # Calculate days since last activity
                    days_old = (current_date - valid_dates.max()).days

                    # Timeliness score decreases with age
                    if days_old <= 365:  # Within 1 year
                        timeliness_scores.append(1.0)
                    elif days_old <= 1095:  # Within 3 years
                        timeliness_scores.append(0.7)
                    elif days_old <= 1825:  # Within 5 years
                        timeliness_scores.append(0.4)
                    else:
                        timeliness_scores.append(0.1)

        return np.mean(timeliness_scores) if timeliness_scores else 0.5

    async def _assess_uniqueness(self, df: pd.DataFrame) -> float:
        """Assess data uniqueness"""
        if df.empty or 'Account_ID' not in df.columns:
            return 0.0

        unique_accounts = df['Account_ID'].nunique()
        total_accounts = len(df['Account_ID'].dropna())

        return unique_accounts / total_accounts if total_accounts > 0 else 0.0

    async def _generate_quality_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on quality metrics"""
        recommendations = []

        if metrics.get("completeness", 0) < 0.8:
            recommendations.append("Improve data completeness by filling missing values")

        if metrics.get("accuracy", 0) < 0.9:
            recommendations.append("Validate data accuracy, especially date formats and numeric values")

        if metrics.get("consistency", 0) < 0.85:
            recommendations.append("Standardize account type classifications")

        if metrics.get("timeliness", 0) < 0.7:
            recommendations.append("Update stale data records with recent information")

        return recommendations


# Enhanced Data Processing Agent
class DataProcessingAgent:
    """Enhanced data processing agent with hybrid memory integration"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session: AsyncSession):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.quality_analyzer = DataQualityAnalyzer(memory_agent)
        self.langsmith_client = LangSmithClient()

        # Supported file formats
        self.supported_formats = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}

        # Data transformation rules
        self.transformation_rules = {
            'account_type_mapping': {
                'current': 'Current', 'savings': 'Saving', 'saving': 'Saving',
                'call': 'Call', 'fixed': 'Fixed', 'term': 'Term',
                'investment': 'Investment', 'sdb': 'safe_deposit_box'
            },
            'boolean_mapping': {
                'yes': 'yes', 'no': 'no', 'true': 'yes', 'false': 'no',
                '1': 'yes', '0': 'no', 'y': 'yes', 'n': 'no'
            }
        }

    @traceable(name="data_processing_pre_hook")
    async def pre_processing_hook(self, state: DataProcessingState) -> DataProcessingState:
        """Enhanced pre-processing memory hook with pattern retrieval"""
        try:
            # Retrieve data processing patterns from knowledge memory
            processing_patterns = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": "data_processing_patterns",
                    "user_id": state.user_id
                }
            )

            if processing_patterns.get("success"):
                state.retrieved_patterns["processing"] = processing_patterns.get("data", {})
                logger.info("Retrieved data processing patterns from memory")

            # Retrieve quality benchmarks
            quality_benchmarks = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": "quality_benchmarks",
                    "user_id": state.user_id
                }
            )

            if quality_benchmarks.get("success"):
                state.retrieved_patterns["quality"] = quality_benchmarks.get("data", {})
                logger.info("Retrieved quality benchmarks from memory")

            # Retrieve user preferences
            user_preferences = await self.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "type": "user_preferences",
                    "user_id": state.user_id
                }
            )

            if user_preferences.get("success"):
                state.memory_context["preferences"] = user_preferences.get("data", {})

            # Log pre-processing hook execution
            state.processing_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_processing_hook",
                "action": "memory_retrieval",
                "status": "completed",
                "patterns_retrieved": len(state.retrieved_patterns),
                "context_loaded": len(state.memory_context)
            })

        except Exception as e:
            logger.error(f"Pre-processing hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_processing_hook",
                "error": str(e)
            })

        return state

    @traceable(name="validate_data_schema")
    async def validate_data_schema(self, data: Dict, schema_type: str = "account") -> Dict:
        """Validate data against predefined schemas"""
        try:
            validation_results = {
                "valid_records": 0,
                "invalid_records": 0,
                "validation_errors": [],
                "schema_compliance": 0.0
            }

            if not data or not isinstance(data, dict):
                return {"error": "Invalid data format"}

            # Convert to DataFrame for validation
            if 'accounts' in data:
                df = pd.DataFrame(data['accounts'])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])

            if df.empty:
                return {"error": "No data to validate"}

            # Schema validation based on type
            schema_class = AccountSchema if schema_type == "account" else PaymentInstrumentSchema

            for index, row in df.iterrows():
                try:
                    # Convert row to dict and validate
                    row_dict = row.to_dict()
                    validated_record = schema_class(**row_dict)
                    validation_results["valid_records"] += 1
                except Exception as e:
                    validation_results["invalid_records"] += 1
                    validation_results["validation_errors"].append({
                        "record_index": index,
                        "error": str(e),
                        "record_data": row_dict
                    })

            total_records = len(df)
            validation_results["schema_compliance"] = (
                    validation_results["valid_records"] / total_records
            ) if total_records > 0 else 0.0

            return validation_results

        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return {"error": str(e)}

    @traceable(name="process_data_file")
    async def process_data_file(self, file_path: str, file_type: str = None) -> Dict:
        """Process data file with format detection and parsing"""
        try:
            path_obj = Path(file_path)

            if not path_obj.exists():
                return {"error": "File not found"}

            # Detect file type if not provided
            if not file_type:
                file_type = path_obj.suffix.lower()

            if file_type not in self.supported_formats:
                return {"error": f"Unsupported file format: {file_type}"}

            # File metadata
            file_stats = path_obj.stat()
            file_metadata = {
                "file_name": path_obj.name,
                "file_size_bytes": file_stats.st_size,
                "file_type": file_type,
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            }

            # Parse file based on type
            if file_type == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_type in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, engine='openpyxl' if file_type == '.xlsx' else 'xlrd')
            elif file_type == '.json':
                async with aiofiles.open(file_path, 'r') as f:
                    json_data = json.loads(await f.read())
                df = pd.json_normalize(json_data)
            elif file_type == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                return {"error": f"Parser not implemented for {file_type}"}

            # Convert DataFrame to dict format
            if df.empty:
                return {"error": "File contains no data"}

            # Data preprocessing
            df = self._preprocess_dataframe(df)

            processed_data = {
                "accounts": df.to_dict('records'),
                "metadata": {
                    "record_count": len(df),
                    "column_count": len(df.columns),
                    "columns": list(df.columns),
                    "file_metadata": file_metadata
                }
            }

            return {"success": True, "data": processed_data}

        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            return {"error": str(e)}

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame with data cleaning and transformation"""
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')

            # Standardize column names
            df.columns = df.columns.str.strip().str.replace(' ', '_')

            # Apply transformation rules
            if 'Account_Type' in df.columns:
                df['Account_Type'] = df['Account_Type'].str.lower().map(
                    self.transformation_rules['account_type_mapping']
                ).fillna(df['Account_Type'])

            # Standardize boolean columns
            boolean_columns = [
                'Customer_Has_Active_Liability_Account',
                'FTD_Auto_Renewal',
                'Bank_Contact_Attempted_Post_Dormancy_Trigger'
            ]

            for col in boolean_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.lower().map(
                        self.transformation_rules['boolean_mapping']
                    ).fillna(df[col])

            # Convert numeric columns
            numeric_columns = ['Current_Balance', 'Unclaimed_Item_Amount']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Standardize date columns
            date_columns = [col for col in df.columns if 'Date' in col or 'date' in col]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Convert back to string for JSON serialization
                    df[col] = df[col].dt.strftime('%Y-%m-%d')

            return df

        except Exception as e:
            logger.error(f"DataFrame preprocessing failed: {str(e)}")
            return df

    @traceable(name="process_banking_data")
    async def process_banking_data(self, state: DataProcessingState) -> DataProcessingState:
        """Main data processing workflow with comprehensive validation"""
        try:
            start_time = datetime.now()
            state.processing_status = ProcessingStatus.PROCESSING

            # Step 1: Process input data
            if state.data_source and Path(state.data_source).exists():
                # Process from file
                file_result = await self.process_data_file(state.data_source)
                if file_result.get("error"):
                    raise ValueError(f"File processing failed: {file_result['error']}")
                state.processed_data = file_result["data"]
                state.file_metadata = file_result["data"]["metadata"]["file_metadata"]
            elif state.raw_data:
                # Process from raw data
                state.processed_data = state.raw_data
            else:
                raise ValueError("No data source provided")

            # Step 2: Schema validation
            validation_result = await self.validate_data_schema(
                state.processed_data,
                schema_type="account"
            )

            if validation_result.get("error"):
                raise ValueError(f"Schema validation failed: {validation_result['error']}")

            state.validation_results = validation_result

            # Step 3: Quality analysis
            df = pd.DataFrame(state.processed_data.get("accounts", []))
            quality_result = await self.quality_analyzer.analyze_quality(df, "account")

            state.quality_metrics = quality_result
            state.quality_score = quality_result.get("overall_score", 0.0)
            state.quality_level = DataQualityLevel(quality_result.get("quality_level", "poor"))

            # Step 4: Call MCP tool for additional processing
            mcp_result = await self.mcp_client.call_tool("process_banking_data", {
                "data": state.processed_data,
                "validation_results": state.validation_results,
                "quality_metrics": state.quality_metrics,
                "processing_context": state.memory_context,
                "retrieved_patterns": state.retrieved_patterns
            })

            if not mcp_result.get("success"):
                logger.warning(f"MCP processing warning: {mcp_result.get('error')}")

            # Step 5: Generate data schema
            state.data_schema = self._generate_data_schema(df)

            # Step 6: Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "records_processed": len(df) if not df.empty else 0,
                "records_per_second": len(df) / processing_time if processing_time > 0 else 0,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024) if not df.empty else 0
            }

            # Determine final status
            if state.quality_score >= 0.8:
                state.processing_status = ProcessingStatus.COMPLETED
            elif state.quality_score >= 0.5:
                state.processing_status = ProcessingStatus.PARTIAL_SUCCESS
            else:
                state.processing_status = ProcessingStatus.REQUIRES_REVIEW

            # Log successful processing
            state.processing_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "data_processing",
                "action": "process_banking_data",
                "status": state.processing_status.value,
                "quality_score": state.quality_score,
                "records_processed": len(df) if not df.empty else 0,
                "processing_time": processing_time
            })

        except Exception as e:
            state.processing_status = ProcessingStatus.FAILED
            error_msg = str(e)
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "data_processing",
                "error": error_msg,
                "error_type": type(e).__name__
            })
            logger.error(f"Data processing failed: {error_msg}")

        return state

    def _generate_data_schema(self, df: pd.DataFrame) -> Dict:
        """Generate data schema information"""
        if df.empty:
            return {}

        schema = {
            "columns": {},
            "record_count": len(df),
            "null_counts": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        for column in df.columns:
            col_info = {
                "data_type": str(df[column].dtype),
                "null_count": df[column].isnull().sum(),
                "unique_count": df[column].nunique(),
                "sample_values": df[column].dropna().head(3).tolist()
            }

            if df[column].dtype in ['int64', 'float64']:
                col_info.update({
                    "min": df[column].min(),
                    "max": df[column].max(),
                    "mean": df[column].mean(),
                    "std": df[column].std()
                })

            schema["columns"][column] = col_info

        return schema

    @traceable(name="data_processing_post_hook")
    async def post_processing_hook(self, state: DataProcessingState) -> DataProcessingState:
        """Enhanced post-processing memory hook with pattern storage"""
        try:
            # Store processing results in session memory
            session_data = {
                "session_id": state.session_id,
                "processing_id": state.processing_id,
                "user_id": state.user_id,
                "processing_results": {
                    "status": state.processing_status.value,
                    "quality_score": state.quality_score,
                    "quality_level": state.quality_level.value if state.quality_level else None,
                    "records_processed": state.performance_metrics.get("records_processed", 0),
                    "processing_time": state.performance_metrics.get("processing_time_seconds", 0)
                },
                "validation_summary": {
                    "schema_compliance": state.validation_results.get("schema_compliance", 0),
                    "error_count": len(state.error_log),
                    "warning_count": state.warning_count
                }
            }

            await self.memory_agent.store_memory(
                bucket="session",
                data=session_data,
                encrypt_sensitive=True
            )

            # Store processing patterns in knowledge memory
            if state.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.PARTIAL_SUCCESS]:
                knowledge_data = {
                    "type": "data_processing_patterns",
                    "user_id": state.user_id,
                    "processing_patterns": {
                        "quality_score": state.quality_score,
                        "common_issues": [error.get("error") for error in state.error_log[:5]],
                        "data_characteristics": state.data_schema,
                        "performance_benchmark": state.performance_metrics
                    },
                    "success_factors": {
                        "file_type": state.file_metadata.get("file_type") if state.file_metadata else "unknown",
                        "record_count": state.performance_metrics.get("records_processed", 0),
                        "processing_approach": "standard_validation"
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket="knowledge",
                    data=knowledge_data
                )

            # Store quality benchmarks
            if state.quality_metrics:
                quality_benchmark = {
                    "type": "quality_benchmarks",
                    "user_id": state.user_id,
                    "benchmark_data": {
                        "overall_score": state.quality_score,
                        "metrics": state.quality_metrics.get("metrics", {}),
                        "recommendations": state.quality_metrics.get("recommendations", [])
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket="knowledge",
                    data=quality_benchmark
                )

            # Log post-processing hook completion
            state.processing_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_processing_hook",
                "action": "memory_storage",
                "status": "completed",
                "session_data_stored": True,
                "knowledge_patterns_stored": state.processing_status in [ProcessingStatus.COMPLETED,
                                                                         ProcessingStatus.PARTIAL_SUCCESS]
            })

        except Exception as e:
            logger.error(f"Post-processing hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_processing_hook",
                "error": str(e)
            })

        return state

    @traceable(name="execute_data_processing_workflow")
    async def execute_workflow(self, user_id: str, data_source: Union[str, Dict],
                               processing_options: Dict = None) -> Dict:
        """Execute complete data processing workflow"""
        try:
            # Initialize processing state
            processing_id = secrets.token_hex(16)
            session_id = secrets.token_hex(16)

            state = DataProcessingState(
                session_id=session_id,
                user_id=user_id,
                processing_id=processing_id,
                timestamp=datetime.now(),
                data_source=data_source if isinstance(data_source, str) else None,
                raw_data=data_source if isinstance(data_source, dict) else None
            )

            # Execute workflow stages
            state = await self.pre_processing_hook(state)
            state = await self.process_banking_data(state)
            state = await self.post_processing_hook(state)

            # Prepare response
            response = {
                "success": state.processing_status != ProcessingStatus.FAILED,
                "processing_id": processing_id,
                "session_id": session_id,
                "status": state.processing_status.value,
                "quality_score": state.quality_score,
                "quality_level": state.quality_level.value if state.quality_level else None,
                "records_processed": state.performance_metrics.get("records_processed", 0),
                "processing_time": state.performance_metrics.get("processing_time_seconds", 0),
                "validation_results": state.validation_results,
                "error_count": len(state.error_log),
                "processing_log": state.processing_log[-5:],  # Last 5 entries
                "recommendations": state.quality_metrics.get("recommendations", []) if state.quality_metrics else []
            }

            if state.processing_status == ProcessingStatus.FAILED:
                response["errors"] = state.error_log

            return response

        except Exception as e:
            logger.error(f"Data processing workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }


