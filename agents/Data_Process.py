"""
Unified Data Processing Agent for Banking Compliance Analysis
Integrates: Data Upload (4 methods), Quality Analysis, BGE Mapping, and Memory Management
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
import hashlib
import secrets
import tempfile
import io
import os
import re
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

# BGE and similarity imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    logging.warning("BGE dependencies not available. Install sentence-transformers and scikit-learn")

# Cloud storage imports
try:
    from azure.storage.blob import BlobServiceClient
    import boto3
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    import hdfs
    CLOUD_DEPENDENCIES_AVAILABLE = True
except ImportError:
    CLOUD_DEPENDENCIES_AVAILABLE = False
    logging.warning("Cloud dependencies not available for some upload methods")

# Memory agent imports
try:
    from memory_agent import HybridMemoryAgent, MemoryContext, MemoryBucket, MemoryPriority
    MEMORY_AGENT_AVAILABLE = True
except ImportError:
    MEMORY_AGENT_AVAILABLE = False
    logging.warning("Memory agent not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class DataQualityLevel(Enum):
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 70-89%
    FAIR = "fair"  # 50-69%
    POOR = "poor"  # Below 50%


class MappingConfidence(Enum):
    HIGH = "high"      # >0.8 similarity
    MEDIUM = "medium"  # 0.6-0.8 similarity
    LOW = "low"        # 0.4-0.6 similarity
    NONE = "none"      # <0.4 similarity


@dataclass
class UploadResult:
    """Data structure for upload results"""
    success: bool
    data: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class QualityResult:
    """Data structure for quality analysis results"""
    success: bool
    overall_score: float = 0.0
    quality_level: str = "poor"
    metrics: Dict[str, float] = None
    missing_percentage: float = 0.0
    duplicate_records: int = 0
    recommendations: List[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class MappingResult:
    """Data structure for column mapping results"""
    success: bool
    mappings: Dict[str, str] = None
    mapping_sheet: Optional[pd.DataFrame] = None
    auto_mapping_percentage: float = 0.0
    method: str = "unknown"
    confidence_distribution: Dict[str, int] = None
    processing_time: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if self.mappings is None:
            self.mappings = {}
        if self.confidence_distribution is None:
            self.confidence_distribution = {}


class UnifiedDataProcessingAgent:
    """
    Unified agent that combines:
    1. Data Upload (4 methods: Files, Drive, DataLake, HDFS)
    2. Data Quality Analysis
    3. BGE-based Column Mapping
    4. Memory Management Integration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified data processing agent"""
        self.config = config or self._default_config()

        # Initialize memory agent if available
        self.memory_agent = None
        if MEMORY_AGENT_AVAILABLE and self.config.get("enable_memory", True):
            try:
                from mcp_client import MCPClient
                mcp_client = MCPClient()
                self.memory_agent = HybridMemoryAgent(mcp_client, self.config.get("memory_config", {}))
                logger.info("✅ Memory agent initialized")
            except Exception as e:
                logger.warning(f"Memory agent initialization failed: {e}")

        # Initialize BGE model for semantic mapping
        self.bge_model = None
        if BGE_AVAILABLE and self.config.get("enable_bge", True):
            try:
                model_name = self.config.get("bge_model", "BAAI/bge-large-en-v1.5")
                self.bge_model = SentenceTransformer(model_name)
                logger.info(f"✅ BGE model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"BGE model loading failed: {e}")

        # Banking compliance schema for mapping
        self.banking_schema = self._load_banking_schema()

        # Supported file formats
        self.supported_formats = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt'}

        # Column mapping patterns
        self.column_mapping_patterns = self._load_column_patterns()

        # Processing statistics
        self.stats = {
            "uploads_processed": 0,
            "quality_analyses": 0,
            "mappings_performed": 0,
            "memory_operations": 0,
            "total_processing_time": 0.0
        }

    def _default_config(self) -> Dict:
        """Default configuration for the agent"""
        return {
            "enable_memory": True,
            "enable_bge": True,
            "bge_model": "BAAI/bge-large-en-v1.5",
            "quality_thresholds": {
                "completeness": 0.8,
                "accuracy": 0.9,
                "consistency": 0.85,
                "validity": 0.9
            },
            "mapping_thresholds": {
                "high_confidence": 0.8,
                "medium_confidence": 0.6,
                "low_confidence": 0.4
            },
            "cloud_configs": {
                "azure": {},
                "aws": {},
                "google": {},
                "hdfs": {}
            },
            "memory_config": {
                "retention_policies": {
                    "session": {"default_ttl": 3600 * 8},  # 8 hours
                    "knowledge": {"default_ttl": 3600 * 24 * 30},  # 30 days
                    "cache": {"default_ttl": 3600}  # 1 hour
                }
            }
        }

    def _load_banking_schema(self) -> Dict[str, Dict]:
        """Load CBUAE banking compliance schema"""
        return {
            'customer_id': {
                'description': 'Unique customer identifier',
                'required': True,
                'type': 'string',
                'keywords': ['customer', 'client', 'id', 'identifier', 'cust']
            },
            'customer_type': {
                'description': 'Type of customer (Individual/Corporate)',
                'required': True,
                'type': 'string',
                'keywords': ['customer_type', 'client_type', 'type', 'individual', 'corporate']
            },
            'full_name_en': {
                'description': 'Customer full name in English',
                'required': True,
                'type': 'string',
                'keywords': ['name', 'full_name', 'customer_name', 'client_name', 'english']
            },
            'account_id': {
                'description': 'Unique account identifier',
                'required': True,
                'type': 'string',
                'keywords': ['account_id', 'account_number', 'acc_id', 'account', 'number']
            },
            'account_type': {
                'description': 'Type of account (Savings/Current/Fixed)',
                'required': True,
                'type': 'string',
                'keywords': ['account_type', 'type', 'savings', 'current', 'fixed', 'deposit']
            },
            'account_status': {
                'description': 'Current status of account',
                'required': True,
                'type': 'string',
                'keywords': ['status', 'account_status', 'active', 'dormant', 'closed']
            },
            'dormancy_status': {
                'description': 'Dormancy classification status',
                'required': True,
                'type': 'string',
                'keywords': ['dormancy', 'dormant', 'dormancy_status', 'classification']
            },
            'balance_current': {
                'description': 'Current account balance',
                'required': True,
                'type': 'float',
                'keywords': ['balance', 'current_balance', 'amount', 'balance_current']
            },
            'last_transaction_date': {
                'description': 'Date of last customer transaction',
                'required': True,
                'type': 'date',
                'keywords': ['transaction_date', 'last_transaction', 'activity_date', 'last_activity']
            },
            'phone_primary': {
                'description': 'Primary phone number',
                'required': False,
                'type': 'string',
                'keywords': ['phone', 'mobile', 'telephone', 'contact', 'primary_phone']
            },
            'email_primary': {
                'description': 'Primary email address',
                'required': False,
                'type': 'string',
                'keywords': ['email', 'email_address', 'contact_email', 'primary_email']
            },
            'date_of_birth': {
                'description': 'Customer date of birth',
                'required': False,
                'type': 'date',
                'keywords': ['birth_date', 'dob', 'date_of_birth', 'birthday']
            },
            'nationality': {
                'description': 'Customer nationality',
                'required': False,
                'type': 'string',
                'keywords': ['nationality', 'country', 'citizenship', 'nation']
            },
            'currency': {
                'description': 'Account currency',
                'required': False,
                'type': 'string',
                'keywords': ['currency', 'curr', 'aed', 'usd', 'eur']
            },
            'opening_date': {
                'description': 'Account opening date',
                'required': False,
                'type': 'date',
                'keywords': ['opening_date', 'opened', 'start_date', 'creation_date']
            },
            'kyc_status': {
                'description': 'KYC compliance status',
                'required': False,
                'type': 'string',
                'keywords': ['kyc', 'kyc_status', 'compliance', 'verification']
            },
            'risk_rating': {
                'description': 'Customer risk rating',
                'required': False,
                'type': 'string',
                'keywords': ['risk', 'rating', 'risk_rating', 'risk_level']
            }
        }

    def _load_column_patterns(self) -> Dict[str, List[str]]:
        """Load common column name patterns for mapping"""
        return {
            'customer_id': ['customer_id', 'cust_id', 'client_id', 'customer_number', 'clientid'],
            'account_id': ['account_id', 'account_number', 'acc_id', 'account_no', 'accountid'],
            'full_name_en': ['name', 'full_name', 'customer_name', 'client_name', 'fullname'],
            'account_type': ['account_type', 'acc_type', 'type', 'product_type', 'accounttype'],
            'balance_current': ['balance', 'current_balance', 'amount', 'bal', 'currentbalance'],
            'account_status': ['status', 'account_status', 'acc_status', 'state'],
            'dormancy_status': ['dormancy_status', 'dormant', 'dormancy', 'classification'],
            'last_transaction_date': ['last_transaction_date', 'transaction_date', 'last_activity', 'activity_date'],
            'phone_primary': ['phone', 'mobile', 'telephone', 'contact_number', 'phone_number'],
            'email_primary': ['email', 'email_address', 'contact_email', 'emailaddress']
        }

    # =================== DATA UPLOAD METHODS ===================

    async def upload_data(self, upload_method: str, source: Union[str, io.IOBase],
                         user_id: str, session_id: str, **kwargs) -> UploadResult:
        """
        Main upload method supporting 4 different data sources

        Args:
            upload_method: 'file', 'drive', 'datalake', or 'hdfs'
            source: File path, URL, file object, or identifier
            user_id: User identifier for memory management
            session_id: Session identifier
            **kwargs: Additional parameters specific to each method

        Returns:
            UploadResult object with data and metadata
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting upload via {upload_method}: {str(source)[:100]}")

            # Create memory context
            memory_context = None
            if self.memory_agent:
                memory_context = await self.memory_agent.create_memory_context(
                    user_id=user_id,
                    session_id=session_id,
                    agent_name="unified_data_processing",
                    workflow_stage="data_upload"
                )

            # Load previous upload patterns from memory
            upload_patterns = await self._load_upload_patterns(memory_context)

            # Route to appropriate upload method
            if upload_method.lower() == 'file':
                result = await self._upload_flat_file(source, **kwargs)
            elif upload_method.lower() == 'drive':
                result = await self._upload_from_drive(source, **kwargs)
            elif upload_method.lower() == 'datalake':
                result = await self._upload_from_datalake(source, **kwargs)
            elif upload_method.lower() == 'hdfs':
                result = await self._upload_from_hdfs(source, **kwargs)
            else:
                return UploadResult(
                    success=False,
                    error=f"Unsupported upload method: {upload_method}"
                )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            # Store upload results in memory
            if result.success and self.memory_agent:
                await self._store_upload_results(result, memory_context, upload_method)

            # Update statistics
            self.stats["uploads_processed"] += 1
            self.stats["total_processing_time"] += processing_time

            return result

        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return UploadResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    async def _upload_flat_file(self, source: Union[str, io.IOBase], **kwargs) -> UploadResult:
        """Upload from flat files (CSV, Excel, JSON, Parquet)"""
        try:
            # Handle different source types
            if hasattr(source, 'read'):
                # File-like object (e.g., Streamlit file uploader)
                return await self._process_file_object(source, **kwargs)
            elif isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    # URL
                    return await self._process_url(source, **kwargs)
                else:
                    # Local file path
                    return await self._process_local_file(source, **kwargs)
            else:
                return UploadResult(
                    success=False,
                    error="Invalid source type. Expected file path, URL, or file object."
                )

        except Exception as e:
            logger.error(f"Flat file upload failed: {str(e)}")
            return UploadResult(success=False, error=str(e))

    async def _process_local_file(self, file_path: str, **kwargs) -> UploadResult:
        """Process local file"""
        path_obj = Path(file_path)

        if not path_obj.exists():
            return UploadResult(success=False, error=f"File not found: {file_path}")

        # Detect file format
        file_extension = path_obj.suffix.lower()
        if file_extension not in self.supported_formats:
            return UploadResult(
                success=False,
                error=f"Unsupported format: {file_extension}. Supported: {self.supported_formats}"
            )

        # Read file
        data = await self._read_file_by_extension(str(path_obj), file_extension, **kwargs)

        if data is None:
            return UploadResult(success=False, error="Failed to read file")

        # Process and validate
        processed_data, warnings = await self._process_banking_data(data)

        # Generate metadata
        file_stats = path_obj.stat()
        metadata = {
            'source_type': 'local_file',
            'file_path': str(path_obj),
            'file_name': path_obj.name,
            'file_size_bytes': file_stats.st_size,
            'file_extension': file_extension,
            'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            'upload_timestamp': datetime.now().isoformat(),
            'total_records': len(processed_data),
            'total_columns': len(processed_data.columns)
        }

        return UploadResult(
            success=True,
            data=processed_data,
            metadata=metadata,
            warnings=warnings
        )

    async def _process_file_object(self, file_obj, **kwargs) -> UploadResult:
        """Process file-like object (e.g., Streamlit upload)"""
        try:
            # Reset file pointer
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)

            # Determine file format
            file_extension = kwargs.get('file_extension', '.csv')
            if hasattr(file_obj, 'name'):
                file_extension = Path(file_obj.name).suffix.lower()

            # Read file
            data = await self._read_file_by_extension(file_obj, file_extension, **kwargs)

            if data is None:
                return UploadResult(success=False, error="Failed to read file object")

            # Process and validate
            processed_data, warnings = await self._process_banking_data(data)

            metadata = {
                'source_type': 'file_object',
                'file_extension': file_extension,
                'file_name': getattr(file_obj, 'name', 'unknown'),
                'upload_timestamp': datetime.now().isoformat(),
                'total_records': len(processed_data),
                'total_columns': len(processed_data.columns)
            }

            return UploadResult(
                success=True,
                data=processed_data,
                metadata=metadata,
                warnings=warnings
            )

        except Exception as e:
            return UploadResult(success=False, error=f"File object processing failed: {str(e)}")

    async def _process_url(self, url: str, **kwargs) -> UploadResult:
        """Process file from URL"""
        try:
            # Parse URL to determine file type
            parsed_url = urlparse(url)
            file_extension = Path(parsed_url.path).suffix.lower()

            if not file_extension:
                file_extension = kwargs.get('file_format', '.csv')

            # Download file with timeout
            timeout = kwargs.get('timeout', 30)
            headers = kwargs.get('headers', {})

            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            try:
                # Read file
                data = await self._read_file_by_extension(tmp_file_path, file_extension, **kwargs)

                if data is None:
                    return UploadResult(success=False, error="Failed to read downloaded file")

                # Process and validate
                processed_data, warnings = await self._process_banking_data(data)

                metadata = {
                    'source_type': 'url',
                    'url': url,
                    'file_extension': file_extension,
                    'content_length': len(response.content),
                    'response_headers': dict(response.headers),
                    'upload_timestamp': datetime.now().isoformat(),
                    'total_records': len(processed_data),
                    'total_columns': len(processed_data.columns)
                }

                return UploadResult(
                    success=True,
                    data=processed_data,
                    metadata=metadata,
                    warnings=warnings
                )

            finally:
                # Cleanup
                os.unlink(tmp_file_path)

        except requests.RequestException as e:
            return UploadResult(success=False, error=f"Failed to download from URL: {str(e)}")
        except Exception as e:
            return UploadResult(success=False, error=f"URL processing failed: {str(e)}")

    async def _upload_from_drive(self, drive_link: str, **kwargs) -> UploadResult:
        """Upload from Google Drive sharing link"""
        if not CLOUD_DEPENDENCIES_AVAILABLE:
            return UploadResult(
                success=False,
                error="Google Drive dependencies not available"
            )

        try:
            # Extract file ID from Drive link
            file_id = self._extract_drive_file_id(drive_link)

            # Attempt direct download
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(direct_url, timeout=30)

            if response.status_code == 200:
                # Determine file format
                content_type = response.headers.get('content-type', '')
                if 'csv' in content_type:
                    file_extension = '.csv'
                elif 'excel' in content_type or 'spreadsheet' in content_type:
                    file_extension = '.xlsx'
                else:
                    file_extension = kwargs.get('file_format', '.csv')

                # Process downloaded content
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                try:
                    data = await self._read_file_by_extension(tmp_file_path, file_extension, **kwargs)

                    if data is None:
                        return UploadResult(success=False, error="Failed to read Google Drive file")

                    processed_data, warnings = await self._process_banking_data(data)

                    metadata = {
                        'source_type': 'google_drive',
                        'drive_link': drive_link,
                        'file_id': file_id,
                        'file_extension': file_extension,
                        'download_method': 'direct',
                        'content_length': len(response.content),
                        'upload_timestamp': datetime.now().isoformat(),
                        'total_records': len(processed_data),
                        'total_columns': len(processed_data.columns)
                    }

                    return UploadResult(
                        success=True,
                        data=processed_data,
                        metadata=metadata,
                        warnings=warnings
                    )

                finally:
                    os.unlink(tmp_file_path)
            else:
                return UploadResult(
                    success=False,
                    error=f"Failed to access Google Drive file. Status: {response.status_code}"
                )

        except Exception as e:
            return UploadResult(success=False, error=f"Google Drive upload failed: {str(e)}")

    async def _upload_from_datalake(self, data_path: str, **kwargs) -> UploadResult:
        """Upload from Data Lake (Azure Data Lake or AWS S3)"""
        if not CLOUD_DEPENDENCIES_AVAILABLE:
            return UploadResult(
                success=False,
                error="Data Lake dependencies not available"
            )

        platform = kwargs.get('platform', 'azure').lower()

        if platform == 'azure':
            return await self._upload_from_azure(data_path, **kwargs)
        elif platform == 'aws':
            return await self._upload_from_s3(data_path, **kwargs)
        else:
            return UploadResult(
                success=False,
                error=f"Unsupported data lake platform: {platform}"
            )

    async def _upload_from_azure(self, data_path: str, **kwargs) -> UploadResult:
        """Upload from Azure Data Lake/Blob Storage"""
        try:
            # Get Azure configuration
            config = self.config.get("cloud_configs", {}).get("azure", {})
            account_name = config.get('account_name') or kwargs.get('account_name')
            account_key = config.get('account_key') or kwargs.get('account_key')
            container_name = config.get('container_name') or kwargs.get('container_name')

            if not all([account_name, account_key, container_name]):
                return UploadResult(
                    success=False,
                    error="Missing Azure credentials"
                )

            # Initialize client and download
            account_url = f"https://{account_name}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=data_path)

            if not blob_client.exists():
                return UploadResult(success=False, error=f"Blob not found: {data_path}")

            blob_data = blob_client.download_blob().readall()
            file_extension = Path(data_path).suffix.lower() or kwargs.get('file_format', '.csv')

            # Process data
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                tmp_file.write(blob_data)
                tmp_file_path = tmp_file.name

            try:
                data = await self._read_file_by_extension(tmp_file_path, file_extension, **kwargs)

                if data is None:
                    return UploadResult(success=False, error="Failed to read Azure blob")

                processed_data, warnings = await self._process_banking_data(data)

                metadata = {
                    'source_type': 'azure_datalake',
                    'account_name': account_name,
                    'container_name': container_name,
                    'blob_path': data_path,
                    'blob_size': len(blob_data),
                    'file_extension': file_extension,
                    'upload_timestamp': datetime.now().isoformat(),
                    'total_records': len(processed_data),
                    'total_columns': len(processed_data.columns)
                }

                return UploadResult(
                    success=True,
                    data=processed_data,
                    metadata=metadata,
                    warnings=warnings
                )

            finally:
                os.unlink(tmp_file_path)

        except Exception as e:
            return UploadResult(success=False, error=f"Azure Data Lake upload failed: {str(e)}")

    async def _upload_from_s3(self, data_path: str, **kwargs) -> UploadResult:
        """Upload from AWS S3"""
        try:
            # Get AWS configuration
            config = self.config.get("cloud_configs", {}).get("aws", {})
            bucket_name = config.get('bucket_name') or kwargs.get('bucket_name')
            aws_access_key_id = config.get('access_key_id') or kwargs.get('aws_access_key_id')
            aws_secret_access_key = config.get('secret_access_key') or kwargs.get('aws_secret_access_key')
            region_name = config.get('region') or kwargs.get('region', 'us-east-1')

            if not all([bucket_name, aws_access_key_id, aws_secret_access_key]):
                return UploadResult(success=False, error="Missing AWS credentials")

            # Initialize S3 client and download
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )

            # Check if object exists
            try:
                s3_client.head_object(Bucket=bucket_name, Key=data_path)
            except s3_client.exceptions.NoSuchKey:
                return UploadResult(success=False, error=f"Object not found: {data_path}")

            file_extension = Path(data_path).suffix.lower() or kwargs.get('file_format', '.csv')

            # Download object
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                s3_client.download_fileobj(bucket_name, data_path, tmp_file)
                tmp_file_path = tmp_file.name

            try:
                data = await self._read_file_by_extension(tmp_file_path, file_extension, **kwargs)

                if data is None:
                    return UploadResult(success=False, error="Failed to read S3 object")

                processed_data, warnings = await self._process_banking_data(data)

                metadata = {
                    'source_type': 'aws_s3',
                    'bucket_name': bucket_name,
                    'object_key': data_path,
                    'file_extension': file_extension,
                    'upload_timestamp': datetime.now().isoformat(),
                    'total_records': len(processed_data),
                    'total_columns': len(processed_data.columns)
                }

                return UploadResult(
                    success=True,
                    data=processed_data,
                    metadata=metadata,
                    warnings=warnings
                )

            finally:
                os.unlink(tmp_file_path)

        except Exception as e:
            return UploadResult(success=False, error=f"AWS S3 upload failed: {str(e)}")

    async def _upload_from_hdfs(self, hdfs_path: str, **kwargs) -> UploadResult:
        """Upload from HDFS"""
        if not CLOUD_DEPENDENCIES_AVAILABLE:
            return UploadResult(success=False, error="HDFS dependencies not available")

        try:
            # Get HDFS configuration
            config = self.config.get("cloud_configs", {}).get("hdfs", {})
            hdfs_host = config.get('host') or kwargs.get('hdfs_host', 'localhost')
            hdfs_port = config.get('port') or kwargs.get('hdfs_port', 9870)
            hdfs_user = config.get('user') or kwargs.get('hdfs_user', 'hdfs')

            # Initialize HDFS client
            from hdfs import InsecureClient
            hdfs_url = f'http://{hdfs_host}:{hdfs_port}'
            hdfs_client = InsecureClient(url=hdfs_url, user=hdfs_user)

            # Check if file exists
            try:
                hdfs_client.status(hdfs_path)
            except Exception:
                return UploadResult(success=False, error=f"File not found in HDFS: {hdfs_path}")

            file_extension = Path(hdfs_path).suffix.lower() or kwargs.get('file_format', '.csv')

            # Download file
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                hdfs_client.download(hdfs_path, tmp_file.name, overwrite=True)
                tmp_file_path = tmp_file.name

            try:
                data = await self._read_file_by_extension(tmp_file_path, file_extension, **kwargs)

                if data is None:
                    return UploadResult(success=False, error="Failed to read HDFS file")

                processed_data, warnings = await self._process_banking_data(data)

                metadata = {
                    'source_type': 'hdfs',
                    'hdfs_host': hdfs_host,
                    'hdfs_port': hdfs_port,
                    'hdfs_path': hdfs_path,
                    'file_extension': file_extension,
                    'upload_timestamp': datetime.now().isoformat(),
                    'total_records': len(processed_data),
                    'total_columns': len(processed_data.columns)
                }

                return UploadResult(
                    success=True,
                    data=processed_data,
                    metadata=metadata,
                    warnings=warnings
                )

            finally:
                os.unlink(tmp_file_path)

        except Exception as e:
            return UploadResult(success=False, error=f"HDFS upload failed: {str(e)}")

    # =================== DATA QUALITY ANALYSIS ===================

    async def analyze_data_quality(self, data: pd.DataFrame, user_id: str,
                                  session_id: str, **kwargs) -> QualityResult:
        """
        Comprehensive data quality analysis

        Args:
            data: DataFrame to analyze
            user_id: User identifier for memory management
            session_id: Session identifier
            **kwargs: Additional options

        Returns:
            QualityResult with quality metrics and recommendations
        """
        start_time = datetime.now()

        try:
            if data.empty:
                return QualityResult(success=False, error="Data is empty")

            logger.info(f"Analyzing data quality for {len(data)} records, {len(data.columns)} columns")

            # Create memory context
            memory_context = None
            if self.memory_agent:
                memory_context = await self.memory_agent.create_memory_context(
                    user_id=user_id,
                    session_id=session_id,
                    agent_name="unified_data_processing",
                    workflow_stage="quality_analysis"
                )

            # Load quality benchmarks from memory
            quality_benchmarks = await self._load_quality_benchmarks(memory_context)

            # Perform quality analysis
            completeness_score = await self._assess_completeness(data)
            accuracy_score = await self._assess_accuracy(data)
            consistency_score = await self._assess_consistency(data)
            validity_score = await self._assess_validity(data)
            uniqueness_score = await self._assess_uniqueness(data)

            # Calculate overall score
            weights = {
                "completeness": 0.25,
                "accuracy": 0.25,
                "consistency": 0.20,
                "validity": 0.20,
                "uniqueness": 0.10
            }

            metrics = {
                "completeness": completeness_score,
                "accuracy": accuracy_score,
                "consistency": consistency_score,
                "validity": validity_score,
                "uniqueness": uniqueness_score
            }

            overall_score = sum(metrics[metric] * weights[metric] for metric in weights.keys())

            # Determine quality level
            if overall_score >= 0.9:
                quality_level = DataQualityLevel.EXCELLENT.value
            elif overall_score >= 0.7:
                quality_level = DataQualityLevel.GOOD.value
            elif overall_score >= 0.5:
                quality_level = DataQualityLevel.FAIR.value
            else:
                quality_level = DataQualityLevel.POOR.value

            # Calculate additional metrics
            missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            duplicate_records = data.duplicated().sum()

            # Generate recommendations
            recommendations = await self._generate_quality_recommendations(metrics, data)

            processing_time = (datetime.now() - start_time).total_seconds()

            result = QualityResult(
                success=True,
                overall_score=overall_score,
                quality_level=quality_level,
                metrics=metrics,
                missing_percentage=missing_percentage,
                duplicate_records=int(duplicate_records),
                recommendations=recommendations,
                processing_time=processing_time
            )

            # Store quality results in memory
            if self.memory_agent:
                await self._store_quality_results(result, memory_context, data)

            # Update statistics
            self.stats["quality_analyses"] += 1
            self.stats["total_processing_time"] += processing_time

            return result

        except Exception as e:
            logger.error(f"Data quality analysis failed: {str(e)}")
            return QualityResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    async def _assess_completeness(self, data: pd.DataFrame) -> float:
        """Assess data completeness"""
        if data.empty:
            return 0.0

        total_cells = data.size
        non_null_cells = data.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0

    async def _assess_accuracy(self, data: pd.DataFrame) -> float:
        """Assess data accuracy based on business rules"""
        if data.empty:
            return 0.0

        accuracy_checks = []

        # Check date formats
        date_columns = [col for col in data.columns if any(word in col.lower() for word in ['date', 'time', 'created', 'updated'])]
        for col in date_columns:
            if col in data.columns:
                valid_dates = pd.to_datetime(data[col], errors='coerce').notna().sum()
                total_dates = data[col].notna().sum()
                if total_dates > 0:
                    accuracy_checks.append(valid_dates / total_dates)

        # Check balance values
        balance_columns = [col for col in data.columns if any(word in col.lower() for word in ['balance', 'amount', 'value'])]
        for col in balance_columns:
            if col in data.columns:
                numeric_data = pd.to_numeric(data[col], errors='coerce')
                valid_balances = (numeric_data >= 0).sum()
                total_balances = data[col].notna().sum()
                if total_balances > 0:
                    accuracy_checks.append(valid_balances / total_balances)

        return np.mean(accuracy_checks) if accuracy_checks else 1.0

    async def _assess_consistency(self, data: pd.DataFrame) -> float:
        """Assess data consistency"""
        if data.empty:
            return 0.0

        consistency_score = 1.0

        # Check account type consistency
        account_type_cols = [col for col in data.columns if 'type' in col.lower()]
        for col in account_type_cols:
            if col in data.columns:
                # Check for consistent formatting
                if data[col].dtype == 'object':
                    values = data[col].dropna()
                    if len(values) > 0:
                        # Check case consistency
                        case_consistent = len(values.str.isupper().unique()) <= 1
                        consistency_score *= 1.0 if case_consistent else 0.8

        return consistency_score

    async def _assess_validity(self, data: pd.DataFrame) -> float:
        """Assess data validity against banking schema"""
        if data.empty:
            return 0.0

        validity_checks = []

        # Check for required banking fields
        required_patterns = ['id', 'account', 'customer', 'balance']
        for pattern in required_patterns:
            pattern_found = any(pattern in col.lower() for col in data.columns)
            validity_checks.append(1.0 if pattern_found else 0.0)

        return np.mean(validity_checks) if validity_checks else 0.0

    async def _assess_uniqueness(self, data: pd.DataFrame) -> float:
        """Assess data uniqueness"""
        if data.empty:
            return 0.0

        # Check for ID columns uniqueness
        id_columns = [col for col in data.columns if 'id' in col.lower()]

        if not id_columns:
            return 0.8  # Default score if no ID columns found

        uniqueness_scores = []
        for col in id_columns:
            if col in data.columns:
                unique_count = data[col].nunique()
                total_count = data[col].notna().sum()
                if total_count > 0:
                    uniqueness_scores.append(unique_count / total_count)

        return np.mean(uniqueness_scores) if uniqueness_scores else 0.8

    async def _generate_quality_recommendations(self, metrics: Dict[str, float], data: pd.DataFrame) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        if metrics.get("completeness", 0) < 0.8:
            missing_cols = data.columns[data.isnull().any()].tolist()
            recommendations.append(f"Improve data completeness. Columns with missing data: {missing_cols[:5]}")

        if metrics.get("accuracy", 0) < 0.9:
            recommendations.append("Validate data accuracy, especially date formats and numeric values")

        if metrics.get("consistency", 0) < 0.85:
            recommendations.append("Standardize data formats and value representations")

        if metrics.get("validity", 0) < 0.9:
            recommendations.append("Ensure data conforms to banking compliance schema requirements")

        if metrics.get("uniqueness", 0) < 0.9:
            recommendations.append("Review and remove duplicate records, ensure ID uniqueness")

        # Data-specific recommendations
        if len(data) < 100:
            recommendations.append("Dataset is small - consider collecting more data for robust analysis")
        elif len(data) > 100000:
            recommendations.append("Large dataset - consider data sampling for initial analysis")

        return recommendations

    # =================== COLUMN MAPPING WITH BGE ===================

    async def map_columns(self, data: pd.DataFrame, user_id: str, session_id: str,
                         use_llm: bool = False, llm_api_key: Optional[str] = None, **kwargs) -> MappingResult:
        """
        Map data columns to banking schema using BGE embeddings and cosine similarity

        Args:
            data: DataFrame to map
            user_id: User identifier for memory management
            session_id: Session identifier
            use_llm: Whether to use LLM for enhanced mapping
            llm_api_key: API key for LLM service (optional)
            **kwargs: Additional options

        Returns:
            MappingResult with column mappings and confidence scores
        """
        start_time = datetime.now()

        try:
            if data.empty:
                return MappingResult(success=False, error="Data is empty")

            if not BGE_AVAILABLE:
                return MappingResult(success=False, error="BGE dependencies not available")

            if not self.bge_model:
                return MappingResult(success=False, error="BGE model not loaded")

            logger.info(f"Mapping {len(data.columns)} columns using BGE embeddings")

            # Create memory context
            memory_context = None
            if self.memory_agent:
                memory_context = await self.memory_agent.create_memory_context(
                    user_id=user_id,
                    session_id=session_id,
                    agent_name="unified_data_processing",
                    workflow_stage="column_mapping"
                )

            # Load mapping patterns from memory
            mapping_patterns = await self._load_mapping_patterns(memory_context)

            # Perform BGE-based semantic mapping
            mappings = await self._perform_bge_mapping(data.columns.tolist())

            # Apply keyword-based mapping as fallback
            keyword_mappings = await self._perform_keyword_mapping(data.columns.tolist())

            # Merge mappings (BGE takes priority)
            final_mappings = {**keyword_mappings, **mappings}

            # Create mapping sheet
            mapping_sheet = await self._create_mapping_sheet(data.columns.tolist(), final_mappings)

            # Calculate statistics
            auto_mapping_percentage = (len(final_mappings) / len(data.columns)) * 100 if len(data.columns) > 0 else 0

            confidence_distribution = {
                "high": len([m for m in mapping_sheet.to_dict('records') if m.get('Confidence_Level') == 'high']),
                "medium": len([m for m in mapping_sheet.to_dict('records') if m.get('Confidence_Level') == 'medium']),
                "low": len([m for m in mapping_sheet.to_dict('records') if m.get('Confidence_Level') == 'low'])
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            # Enhance with LLM if requested
            method = "BGE Semantic + Keyword"
            if use_llm and llm_api_key:
                try:
                    enhanced_mappings = await self._enhance_with_llm(mapping_sheet, llm_api_key)
                    if enhanced_mappings:
                        mapping_sheet = enhanced_mappings
                        method = "BGE + LLM Enhanced"
                except Exception as e:
                    logger.warning(f"LLM enhancement failed: {e}")

            result = MappingResult(
                success=True,
                mappings=final_mappings,
                mapping_sheet=mapping_sheet,
                auto_mapping_percentage=auto_mapping_percentage,
                method=method,
                confidence_distribution=confidence_distribution,
                processing_time=processing_time
            )

            # Store mapping results in memory
            if self.memory_agent:
                await self._store_mapping_results(result, memory_context, data)

            # Update statistics
            self.stats["mappings_performed"] += 1
            self.stats["total_processing_time"] += processing_time

            return result

        except Exception as e:
            logger.error(f"Column mapping failed: {str(e)}")
            return MappingResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    async def _perform_bge_mapping(self, source_columns: List[str]) -> Dict[str, str]:
        """Perform BGE-based semantic mapping"""
        try:
            # Prepare texts for embedding
            source_texts = [col.replace('_', ' ').lower() for col in source_columns]
            target_texts = []
            target_fields = []

            for field, info in self.banking_schema.items():
                target_fields.append(field)
                enhanced_text = f"{field.replace('_', ' ')} {info['description']} {' '.join(info['keywords'][:3])}"
                target_texts.append(enhanced_text)

            # Generate embeddings
            source_embeddings = self.bge_model.encode(source_texts, normalize_embeddings=True)
            target_embeddings = self.bge_model.encode(target_texts, normalize_embeddings=True)

            # Calculate similarities
            similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)

            # Create mappings based on similarity scores
            mappings = {}
            used_targets = set()
            thresholds = self.config["mapping_thresholds"]

            # Sort by highest similarity first
            source_target_pairs = []
            for i, source_col in enumerate(source_columns):
                for j, target_field in enumerate(target_fields):
                    similarity = similarity_matrix[i][j]
                    source_target_pairs.append((i, j, similarity, source_col, target_field))

            source_target_pairs.sort(key=lambda x: x[2], reverse=True)
            used_sources = set()

            for i, j, similarity, source_col, target_field in source_target_pairs:
                if (source_col not in used_sources and
                    target_field not in used_targets and
                    similarity >= thresholds["low_confidence"]):

                    mappings[source_col] = target_field
                    used_sources.add(source_col)
                    used_targets.add(target_field)

            return mappings

        except Exception as e:
            logger.error(f"BGE mapping failed: {e}")
            return {}

    async def _perform_keyword_mapping(self, source_columns: List[str]) -> Dict[str, str]:
        """Perform keyword-based mapping as fallback"""
        mappings = {}

        for source_col in source_columns:
            source_lower = source_col.lower().replace('_', '').replace('-', '').replace(' ', '')

            best_match = None
            best_score = 0

            for target_field, patterns in self.column_mapping_patterns.items():
                for pattern in patterns:
                    pattern_clean = pattern.lower().replace('_', '').replace('-', '').replace(' ', '')

                    # Exact match
                    if source_lower == pattern_clean:
                        mappings[source_col] = target_field
                        break

                    # Substring match
                    elif pattern_clean in source_lower or source_lower in pattern_clean:
                        score = len(pattern_clean) / len(source_lower) if source_lower else 0
                        if score > best_score:
                            best_match = target_field
                            best_score = score

                if source_col in mappings:
                    break

            # Use best match if no exact match found
            if source_col not in mappings and best_match and best_score > 0.5:
                mappings[source_col] = best_match

        return mappings

    async def _create_mapping_sheet(self, source_columns: List[str], mappings: Dict[str, str]) -> pd.DataFrame:
        """Create detailed mapping sheet"""
        mapping_data = []

        for source_col in source_columns:
            target_field = mappings.get(source_col, "")

            # Determine confidence level
            if target_field:
                confidence = "high"  # Default for now, could be enhanced with similarity scores
                similarity_score = 0.85  # Placeholder
            else:
                confidence = "none"
                similarity_score = 0.0

            # Check if required field
            required = target_field in [f for f, info in self.banking_schema.items() if info.get('required', False)]

            mapping_data.append({
                'Source_Column': source_col,
                'Target_Field': target_field,
                'Confidence_Level': confidence,
                'Similarity_Score': similarity_score,
                'Data_Type': self.banking_schema.get(target_field, {}).get('type', 'unknown'),
                'Required': required,
                'Description': self.banking_schema.get(target_field, {}).get('description', '')
            })

        return pd.DataFrame(mapping_data)

    async def _enhance_with_llm(self, mapping_sheet: pd.DataFrame, api_key: str) -> Optional[pd.DataFrame]:
        """Enhance mapping using LLM (placeholder for now)"""
        # This would integrate with LLM service like Groq
        # For now, return original mapping sheet
        logger.info("LLM enhancement requested but not implemented yet")
        return mapping_sheet

    # =================== MEMORY INTEGRATION ===================

    async def _load_upload_patterns(self, memory_context: Optional[MemoryContext]) -> Dict:
        """Load upload patterns from memory"""
        if not self.memory_agent or not memory_context:
            return {}

        try:
            result = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "upload_patterns",
                    "user_id": memory_context.user_id
                },
                context=memory_context
            )

            if result.get("success") and result.get("data"):
                self.stats["memory_operations"] += 1
                return result["data"][0].get("data", {}) if result["data"] else {}

            return {}
        except Exception as e:
            logger.warning(f"Failed to load upload patterns: {e}")
            return {}

    async def _load_quality_benchmarks(self, memory_context: Optional[MemoryContext]) -> Dict:
        """Load quality benchmarks from memory"""
        if not self.memory_agent or not memory_context:
            return {}

        try:
            result = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "quality_benchmarks",
                    "user_id": memory_context.user_id
                },
                context=memory_context
            )

            if result.get("success") and result.get("data"):
                self.stats["memory_operations"] += 1
                return result["data"][0].get("data", {}) if result["data"] else {}

            return {}
        except Exception as e:
            logger.warning(f"Failed to load quality benchmarks: {e}")
            return {}

    async def _load_mapping_patterns(self, memory_context: Optional[MemoryContext]) -> Dict:
        """Load mapping patterns from memory"""
        if not self.memory_agent or not memory_context:
            return {}

        try:
            result = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "mapping_patterns",
                    "user_id": memory_context.user_id
                },
                context=memory_context
            )

            if result.get("success") and result.get("data"):
                self.stats["memory_operations"] += 1
                return result["data"][0].get("data", {}) if result["data"] else {}

            return {}
        except Exception as e:
            logger.warning(f"Failed to load mapping patterns: {e}")
            return {}

    async def _store_upload_results(self, result: UploadResult, memory_context: MemoryContext,
                                   upload_method: str) -> None:
        """Store upload results in memory"""
        if not self.memory_agent or not result.success:
            return

        try:
            upload_data = {
                "type": "upload_results",
                "user_id": memory_context.user_id,
                "session_id": memory_context.session_id,
                "upload_method": upload_method,
                "metadata": result.metadata,
                "processing_time": result.processing_time,
                "warnings": result.warnings,
                "timestamp": datetime.now().isoformat()
            }

            await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data=upload_data,
                context=memory_context,
                content_type="upload_results",
                tags=["upload", upload_method, "session_results"]
            )

            self.stats["memory_operations"] += 1

        except Exception as e:
            logger.warning(f"Failed to store upload results: {e}")

    async def _store_quality_results(self, result: QualityResult, memory_context: MemoryContext,
                                    data: pd.DataFrame) -> None:
        """Store quality results in memory"""
        if not self.memory_agent or not result.success:
            return

        try:
            quality_data = {
                "type": "quality_results",
                "user_id": memory_context.user_id,
                "session_id": memory_context.session_id,
                "overall_score": result.overall_score,
                "quality_level": result.quality_level,
                "metrics": result.metrics,
                "missing_percentage": result.missing_percentage,
                "duplicate_records": result.duplicate_records,
                "recommendations": result.recommendations,
                "data_shape": {"rows": len(data), "columns": len(data.columns)},
                "processing_time": result.processing_time,
                "timestamp": datetime.now().isoformat()
            }

            await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data=quality_data,
                context=memory_context,
                content_type="quality_results",
                tags=["quality", "analysis", "session_results"]
            )

            # Store as knowledge if high quality
            if result.overall_score > 0.8:
                knowledge_data = {
                    "type": "quality_benchmarks",
                    "user_id": memory_context.user_id,
                    "benchmark_score": result.overall_score,
                    "successful_patterns": result.metrics,
                    "data_characteristics": {
                        "columns": len(data.columns),
                        "rows": len(data),
                        "column_types": data.dtypes.astype(str).to_dict()
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.KNOWLEDGE.value,
                    data=knowledge_data,
                    context=memory_context,
                    content_type="quality_benchmarks",
                    priority=MemoryPriority.HIGH,
                    tags=["quality", "benchmarks", "patterns"]
                )

            self.stats["memory_operations"] += 2

        except Exception as e:
            logger.warning(f"Failed to store quality results: {e}")

    async def _store_mapping_results(self, result: MappingResult, memory_context: MemoryContext,
                                    data: pd.DataFrame) -> None:
        """Store mapping results in memory"""
        if not self.memory_agent or not result.success:
            return

        try:
            mapping_data = {
                "type": "mapping_results",
                "user_id": memory_context.user_id,
                "session_id": memory_context.session_id,
                "mappings": result.mappings,
                "auto_mapping_percentage": result.auto_mapping_percentage,
                "method": result.method,
                "confidence_distribution": result.confidence_distribution,
                "source_columns": data.columns.tolist(),
                "processing_time": result.processing_time,
                "timestamp": datetime.now().isoformat()
            }

            await self.memory_agent.store_memory(
                bucket=MemoryBucket.SESSION.value,
                data=mapping_data,
                context=memory_context,
                content_type="mapping_results",
                tags=["mapping", "bge", "session_results"]
            )

            # Store successful patterns as knowledge
            if result.auto_mapping_percentage > 70:
                pattern_data = {
                    "type": "mapping_patterns",
                    "user_id": memory_context.user_id,
                    "successful_mappings": result.mappings,
                    "source_patterns": data.columns.tolist(),
                    "mapping_confidence": result.auto_mapping_percentage,
                    "method_used": result.method,
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.KNOWLEDGE.value,
                    data=pattern_data,
                    context=memory_context,
                    content_type="mapping_patterns",
                    priority=MemoryPriority.HIGH,
                    tags=["mapping", "patterns", "successful"]
                )

            self.stats["memory_operations"] += 2

        except Exception as e:
            logger.warning(f"Failed to store mapping results: {e}")

    # =================== UTILITY METHODS ===================

    async def _read_file_by_extension(self, file_path_or_obj: Union[str, io.IOBase],
                                     file_extension: str, **kwargs) -> Optional[pd.DataFrame]:
        """Read file based on extension"""
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path_or_obj, encoding='utf-8-sig', **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path_or_obj, **kwargs)
            elif file_extension == '.json':
                return pd.read_json(file_path_or_obj, **kwargs)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path_or_obj, **kwargs)
            elif file_extension == '.txt':
                delimiter = kwargs.get('delimiter', '\t')
                return pd.read_csv(file_path_or_obj, delimiter=delimiter, **kwargs)
            else:
                logger.error(f"Unsupported file extension: {file_extension}")
                return None
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None

    async def _process_banking_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Process and clean banking data"""
        warnings = []

        try:
            if data.empty:
                raise ValueError("Data is empty")

            # Clean column names
            data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

            # Remove completely empty rows and columns
            initial_shape = data.shape
            data = data.dropna(how='all').dropna(axis=1, how='all')

            if data.shape != initial_shape:
                warnings.append(f"Removed empty rows/columns. Shape changed from {initial_shape} to {data.shape}")

            # Data type conversions
            for col in data.columns:
                # Convert date columns
                if any(word in col for word in ['date', 'time', 'created', 'updated']):
                    try:
                        data[col] = pd.to_datetime(data[col], errors='coerce')
                    except:
                        pass

                # Convert numeric columns
                elif any(word in col for word in ['balance', 'amount', 'value', 'number']):
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except:
                        pass

            # Remove duplicates
            initial_count = len(data)
            data = data.drop_duplicates()
            if len(data) < initial_count:
                warnings.append(f"Removed {initial_count - len(data)} duplicate records")

            return data, warnings

        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return data, [f"Processing error: {str(e)}"]

    def _extract_drive_file_id(self, drive_link: str) -> str:
        """Extract file ID from Google Drive sharing link"""
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/open\?id=([a-zA-Z0-9-_]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, drive_link)
            if match:
                return match.group(1)

        raise ValueError("Could not extract file ID from Google Drive link")

    # =================== COMPREHENSIVE WORKFLOW ===================

    async def process_data_comprehensive(self, upload_method: str, source: Union[str, io.IOBase],
                                       user_id: str, session_id: str,
                                       run_quality_analysis: bool = True,
                                       run_column_mapping: bool = True,
                                       use_llm_mapping: bool = False,
                                       llm_api_key: Optional[str] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive data processing workflow:
        1. Data Upload
        2. Quality Analysis
        3. Column Mapping
        4. Memory Storage

        Returns:
            Dictionary with all results and summary
        """
        workflow_start = datetime.now()

        try:
            results = {
                "workflow_id": secrets.token_hex(8),
                "user_id": user_id,
                "session_id": session_id,
                "start_time": workflow_start.isoformat(),
                "upload_result": None,
                "quality_result": None,
                "mapping_result": None,
                "summary": {},
                "success": False
            }

            # Step 1: Data Upload
            logger.info(f"Starting comprehensive workflow: Upload via {upload_method}")
            upload_result = await self.upload_data(upload_method, source, user_id, session_id, **kwargs)
            results["upload_result"] = asdict(upload_result)

            if not upload_result.success:
                results["error"] = f"Upload failed: {upload_result.error}"
                return results

            data = upload_result.data

            # Step 2: Quality Analysis (optional)
            if run_quality_analysis:
                logger.info("Running quality analysis...")
                quality_result = await self.analyze_data_quality(data, user_id, session_id, **kwargs)
                results["quality_result"] = asdict(quality_result)

                if not quality_result.success:
                    logger.warning(f"Quality analysis failed: {quality_result.error}")

            # Step 3: Column Mapping (optional)
            if run_column_mapping:
                logger.info("Running column mapping...")
                mapping_result = await self.map_columns(data, user_id, session_id, use_llm_mapping, llm_api_key, **kwargs)
                results["mapping_result"] = asdict(mapping_result)

                if not mapping_result.success:
                    logger.warning(f"Column mapping failed: {mapping_result.error}")

            # Calculate workflow summary
            workflow_time = (datetime.now() - workflow_start).total_seconds()

            results["summary"] = {
                "total_processing_time": workflow_time,
                "records_processed": len(data),
                "columns_processed": len(data.columns),
                "upload_method": upload_method,
                "quality_score": results.get("quality_result", {}).get("overall_score", 0),
                "mapping_percentage": results.get("mapping_result", {}).get("auto_mapping_percentage", 0),
                "workflow_completed": datetime.now().isoformat()
            }

            results["success"] = True
            results["end_time"] = datetime.now().isoformat()

            logger.info(f"Comprehensive workflow completed in {workflow_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Comprehensive workflow failed: {str(e)}")
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            return results

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            "agent_stats": self.stats.copy(),
            "memory_available": self.memory_agent is not None,
            "bge_available": self.bge_model is not None,
            "cloud_dependencies": CLOUD_DEPENDENCIES_AVAILABLE,
            "supported_formats": list(self.supported_formats),
            "schema_fields": len(self.banking_schema),
            "mapping_patterns": len(self.column_mapping_patterns),
            "configuration": {
                "enable_memory": self.config.get("enable_memory"),
                "enable_bge": self.config.get("enable_bge"),
                "bge_model": self.config.get("bge_model")
            }
        }


# Example usage and factory functions
def create_unified_data_processing_agent(config: Optional[Dict] = None) -> UnifiedDataProcessingAgent:
    """Factory function to create unified data processing agent"""
    return UnifiedDataProcessingAgent(config)


async def quick_process_data(file_path: str, user_id: str = "default_user") -> Dict[str, Any]:
    """Quick helper function for simple data processing"""
    agent = create_unified_data_processing_agent()
    session_id = secrets.token_hex(8)

    return await agent.process_data_comprehensive(
        upload_method="file",
        source=file_path,
        user_id=user_id,
        session_id=session_id,
        run_quality_analysis=True,
        run_column_mapping=True
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create agent
        agent = create_unified_data_processing_agent()

        # Example comprehensive processing
        # result = await agent.process_data_comprehensive(
        #     upload_method="file",
        #     source="banking_data.csv",
        #     user_id="user123",
        #     session_id="session456"
        # )

        # Print statistics
        stats = agent.get_agent_statistics()
        print("Agent Statistics:", json.dumps(stats, indent=2))

    # Run example
    # asyncio.run(main())
    print("Unified Data Processing Agent initialized successfully!")
    print("Features: 4 Upload Methods + Quality Analysis + BGE Mapping + Memory Integration")