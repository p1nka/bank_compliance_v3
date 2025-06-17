"""
Data Upload System for Banking Compliance Analysis
Supports 4 upload methods: Flat Files, Drive Links, Data Lake, and HDFS
"""

import os
import io
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime
import logging
from pathlib import Path
import requests
from urllib.parse import urlparse, parse_qs
import tempfile
import json
import asyncio
from dataclasses import dataclass
import re

# Cloud storage and big data imports
try:
    from azure.storage.blob import BlobServiceClient
    from azure.storage.filedatalake import DataLakeServiceClient
    import hdfs
    from hdfs import InsecureClient
    import boto3
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    CLOUD_DEPENDENCIES_AVAILABLE = True
except ImportError:
    CLOUD_DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


class BankingComplianceUploader:
    """
    Comprehensive data upload handler for Banking Compliance System
    Supports: Flat Files, Google Drive, Azure Data Lake, AWS S3, HDFS
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the uploader with configuration"""
        self.config = config or {}

        # Banking compliance schema from the dataset
        self.expected_columns = [
            'customer_id', 'customer_type', 'full_name_en', 'full_name_ar', 'id_number',
            'id_type', 'date_of_birth', 'nationality', 'address_line1', 'address_line2',
            'city', 'emirate', 'country', 'postal_code', 'phone_primary', 'phone_secondary',
            'email_primary', 'email_secondary', 'address_known', 'last_contact_date',
            'last_contact_method', 'kyc_status', 'kyc_expiry_date', 'risk_rating',
            'account_id', 'account_type', 'account_subtype', 'account_name', 'currency',
            'account_status', 'dormancy_status', 'opening_date', 'closing_date',
            'last_transaction_date', 'last_system_transaction_date', 'balance_current',
            'balance_available', 'balance_minimum', 'interest_rate', 'interest_accrued',
            'is_joint_account', 'joint_account_holders', 'has_outstanding_facilities',
            'maturity_date', 'auto_renewal', 'last_statement_date', 'statement_frequency',
            'tracking_id', 'dormancy_trigger_date', 'dormancy_period_start',
            'dormancy_period_months', 'dormancy_classification_date', 'transfer_eligibility_date',
            'current_stage', 'contact_attempts_made', 'last_contact_attempt_date',
            'waiting_period_start', 'waiting_period_end', 'transferred_to_ledger_date',
            'transferred_to_cb_date', 'cb_transfer_amount', 'cb_transfer_reference',
            'exclusion_reason', 'created_date', 'updated_date', 'updated_by'
        ]

        # Supported file formats
        self.supported_formats = {
            'csv': self._read_csv,
            'xlsx': self._read_excel,
            'xls': self._read_excel,
            'json': self._read_json,
            'parquet': self._read_parquet,
            'txt': self._read_text
        }

        # Column mapping for common variations
        self.column_mapping = {
            'cust_id': 'customer_id',
            'customer_number': 'customer_id',
            'client_id': 'customer_id',
            'acc_id': 'account_id',
            'account_number': 'account_id',
            'account_no': 'account_id',
            'acc_type': 'account_type',
            'account_category': 'account_type',
            'acc_status': 'account_status',
            'status': 'account_status',
            'full_name': 'full_name_en',
            'name': 'full_name_en',
            'customer_name': 'full_name_en',
            'client_name': 'full_name_en',
            'dob': 'date_of_birth',
            'birth_date': 'date_of_birth',
            'phone': 'phone_primary',
            'mobile': 'phone_primary',
            'telephone': 'phone_primary',
            'email': 'email_primary',
            'email_address': 'email_primary',
            'balance': 'balance_current',
            'current_balance': 'balance_current',
            'available_balance': 'balance_available',
            'available_bal': 'balance_available'
        }

    def upload_data(self, upload_method: str, source: str, **kwargs) -> UploadResult:
        """
        Main upload method supporting 4 different data sources

        Args:
            upload_method: 'file', 'drive', 'datalake', or 'hdfs'
            source: File path, URL, or identifier
            **kwargs: Additional parameters specific to each method

        Returns:
            UploadResult object with data and metadata
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting upload via {upload_method}: {source}")

            # Route to appropriate upload method
            if upload_method.lower() == 'file':
                result = self._upload_flat_file(source, **kwargs)
            elif upload_method.lower() == 'drive':
                result = self._upload_from_drive(source, **kwargs)
            elif upload_method.lower() == 'datalake':
                result = self._upload_from_datalake(source, **kwargs)
            elif upload_method.lower() == 'hdfs':
                result = self._upload_from_hdfs(source, **kwargs)
            else:
                return UploadResult(
                    success=False,
                    error=f"Unsupported upload method: {upload_method}"
                )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            return result

        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return UploadResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def _upload_flat_file(self, source: Union[str, io.IOBase], **kwargs) -> UploadResult:
        """
        Upload from flat files (CSV, Excel, JSON, Parquet)
        Supports local files, URLs, and file objects (e.g., Streamlit uploads)
        """
        try:
            # Handle different source types
            if hasattr(source, 'read'):
                # File-like object (e.g., Streamlit file uploader)
                return self._process_file_object(source, **kwargs)
            elif isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    # URL
                    return self._process_url(source, **kwargs)
                else:
                    # Local file path
                    return self._process_local_file(source, **kwargs)
            else:
                return UploadResult(
                    success=False,
                    error="Invalid source type. Expected file path, URL, or file object."
                )

        except Exception as e:
            logger.error(f"Flat file upload failed: {str(e)}")
            return UploadResult(success=False, error=str(e))

    def _process_local_file(self, file_path: str, **kwargs) -> UploadResult:
        """Process local file"""
        path_obj = Path(file_path)

        if not path_obj.exists():
            return UploadResult(success=False, error=f"File not found: {file_path}")

        # Detect file format
        file_extension = path_obj.suffix.lower().lstrip('.')
        if file_extension not in self.supported_formats:
            return UploadResult(
                success=False,
                error=f"Unsupported format: {file_extension}. Supported: {list(self.supported_formats.keys())}"
            )

        # Read file
        read_function = self.supported_formats[file_extension]
        data = read_function(str(path_obj), **kwargs)

        # Process and validate
        processed_data, warnings = self._process_banking_data(data)

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

    def _process_url(self, url: str, **kwargs) -> UploadResult:
        """Process file from URL"""
        try:
            # Parse URL to determine file type
            parsed_url = urlparse(url)
            file_extension = Path(parsed_url.path).suffix.lower().lstrip('.')

            if not file_extension:
                file_extension = kwargs.get('file_format', 'csv')

            # Download file with timeout
            timeout = kwargs.get('timeout', 30)
            headers = kwargs.get('headers', {})

            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            try:
                # Read file
                read_function = self.supported_formats.get(file_extension, self._read_csv)
                data = read_function(tmp_file_path, **kwargs)

                # Process and validate
                processed_data, warnings = self._process_banking_data(data)

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

    def _process_file_object(self, file_obj, **kwargs) -> UploadResult:
        """Process file-like object (e.g., Streamlit upload)"""
        try:
            # Reset file pointer
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)

            # Determine file format
            file_extension = kwargs.get('file_extension', 'csv')
            if hasattr(file_obj, 'name'):
                file_extension = Path(file_obj.name).suffix.lower().lstrip('.')

            # Read file
            read_function = self.supported_formats.get(file_extension, self._read_csv)
            data = read_function(file_obj, **kwargs)

            # Process and validate
            processed_data, warnings = self._process_banking_data(data)

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

    def _upload_from_drive(self, drive_link: str, **kwargs) -> UploadResult:
        """
        Upload from Google Drive sharing link
        Supports both direct download and API access
        """
        if not CLOUD_DEPENDENCIES_AVAILABLE:
            return UploadResult(
                success=False,
                error="Google Drive dependencies not available. Install google-api-python-client"
            )

        try:
            # Try direct download first (for public files)
            file_id = self._extract_drive_file_id(drive_link)

            # Attempt direct download
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            response = requests.get(direct_url, timeout=30)

            if response.status_code == 200:
                # Direct download successful
                content_type = response.headers.get('content-type', '')

                # Determine file format from content type or URL
                if 'csv' in content_type or 'text/csv' in content_type:
                    file_extension = 'csv'
                elif 'excel' in content_type or 'spreadsheet' in content_type:
                    file_extension = 'xlsx'
                else:
                    file_extension = kwargs.get('file_format', 'csv')

                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                try:
                    # Read file
                    read_function = self.supported_formats.get(file_extension, self._read_csv)
                    data = read_function(tmp_file_path, **kwargs)

                    # Process and validate
                    processed_data, warnings = self._process_banking_data(data)

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
                    error=f"Failed to access Google Drive file. Status: {response.status_code}. "
                          "Make sure the file is publicly accessible or provide API credentials."
                )

        except Exception as e:
            return UploadResult(success=False, error=f"Google Drive upload failed: {str(e)}")

    def _upload_from_datalake(self, data_path: str, **kwargs) -> UploadResult:
        """
        Upload from Data Lake (Azure Data Lake or AWS S3)
        """
        if not CLOUD_DEPENDENCIES_AVAILABLE:
            return UploadResult(
                success=False,
                error="Data Lake dependencies not available. Install azure-storage-blob or boto3"
            )

        platform = kwargs.get('platform', 'azure').lower()

        if platform == 'azure':
            return self._upload_from_azure(data_path, **kwargs)
        elif platform == 'aws':
            return self._upload_from_s3(data_path, **kwargs)
        else:
            return UploadResult(
                success=False,
                error=f"Unsupported data lake platform: {platform}. Use 'azure' or 'aws'"
            )

    def _upload_from_azure(self, data_path: str, **kwargs) -> UploadResult:
        """Upload from Azure Data Lake/Blob Storage"""
        try:
            # Get Azure configuration
            account_name = self.config.get('azure_account_name') or kwargs.get('account_name')
            account_key = self.config.get('azure_account_key') or kwargs.get('account_key')
            container_name = self.config.get('azure_container_name') or kwargs.get('container_name')

            if not all([account_name, account_key, container_name]):
                return UploadResult(
                    success=False,
                    error="Missing Azure credentials. Provide account_name, account_key, and container_name"
                )

            # Initialize client
            account_url = f"https://{account_name}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=account_key
            )

            # Get blob client
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=data_path
            )

            # Check if blob exists
            if not blob_client.exists():
                return UploadResult(
                    success=False,
                    error=f"Blob not found: {data_path} in container {container_name}"
                )

            # Get blob properties
            blob_properties = blob_client.get_blob_properties()

            # Download blob
            blob_data = blob_client.download_blob().readall()

            # Determine file format
            file_extension = Path(data_path).suffix.lower().lstrip('.')
            if not file_extension:
                file_extension = kwargs.get('file_format', 'csv')

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                tmp_file.write(blob_data)
                tmp_file_path = tmp_file.name

            try:
                # Read file
                read_function = self.supported_formats.get(file_extension, self._read_csv)
                data = read_function(tmp_file_path, **kwargs)

                # Process and validate
                processed_data, warnings = self._process_banking_data(data)

                metadata = {
                    'source_type': 'azure_datalake',
                    'account_name': account_name,
                    'container_name': container_name,
                    'blob_path': data_path,
                    'blob_size': blob_properties.size,
                    'last_modified': blob_properties.last_modified.isoformat() if blob_properties.last_modified else None,
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

    def _upload_from_s3(self, data_path: str, **kwargs) -> UploadResult:
        """Upload from AWS S3"""
        try:
            # Get AWS configuration
            bucket_name = self.config.get('aws_bucket_name') or kwargs.get('bucket_name')
            aws_access_key_id = self.config.get('aws_access_key_id') or kwargs.get('aws_access_key_id')
            aws_secret_access_key = self.config.get('aws_secret_access_key') or kwargs.get('aws_secret_access_key')
            region_name = self.config.get('aws_region') or kwargs.get('region', 'us-east-1')

            if not all([bucket_name, aws_access_key_id, aws_secret_access_key]):
                return UploadResult(
                    success=False,
                    error="Missing AWS credentials. Provide bucket_name, aws_access_key_id, and aws_secret_access_key"
                )

            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )

            # Check if object exists
            try:
                response = s3_client.head_object(Bucket=bucket_name, Key=data_path)
            except s3_client.exceptions.NoSuchKey:
                return UploadResult(
                    success=False,
                    error=f"Object not found: {data_path} in bucket {bucket_name}"
                )

            # Determine file format
            file_extension = Path(data_path).suffix.lower().lstrip('.')
            if not file_extension:
                file_extension = kwargs.get('file_format', 'csv')

            # Download object
            with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                s3_client.download_fileobj(bucket_name, data_path, tmp_file)
                tmp_file_path = tmp_file.name

            try:
                # Read file
                read_function = self.supported_formats.get(file_extension, self._read_csv)
                data = read_function(tmp_file_path, **kwargs)

                # Process and validate
                processed_data, warnings = self._process_banking_data(data)

                metadata = {
                    'source_type': 'aws_s3',
                    'bucket_name': bucket_name,
                    'object_key': data_path,
                    'object_size': response['ContentLength'],
                    'last_modified': response['LastModified'].isoformat(),
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

    def _upload_from_hdfs(self, hdfs_path: str, **kwargs) -> UploadResult:
        """
        Upload from HDFS (Hadoop Distributed File System)
        """
        if not CLOUD_DEPENDENCIES_AVAILABLE:
            return UploadResult(
                success=False,
                error="HDFS dependencies not available. Install hdfs3 or hdfs"
            )

        try:
            # Get HDFS configuration
            hdfs_host = self.config.get('hdfs_host') or kwargs.get('hdfs_host', 'localhost')
            hdfs_port = self.config.get('hdfs_port') or kwargs.get('hdfs_port', 9870)
            hdfs_user = self.config.get('hdfs_user') or kwargs.get('hdfs_user', 'hdfs')

            # Initialize HDFS client
            hdfs_url = f'http://{hdfs_host}:{hdfs_port}'
            hdfs_client = InsecureClient(url=hdfs_url, user=hdfs_user)

            # Check if file exists
            try:
                file_status = hdfs_client.status(hdfs_path)
            except Exception:
                return UploadResult(
                    success=False,
                    error=f"File not found in HDFS: {hdfs_path}"
                )

            # Determine file format
            file_extension = Path(hdfs_path).suffix.lower().lstrip('.')
            if not file_extension:
                file_extension = kwargs.get('file_format', 'csv')

            # Download file
            with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                hdfs_client.download(hdfs_path, tmp_file.name, overwrite=True)
                tmp_file_path = tmp_file.name

            try:
                # Read file
                read_function = self.supported_formats.get(file_extension, self._read_csv)
                data = read_function(tmp_file_path, **kwargs)

                # Process and validate
                processed_data, warnings = self._process_banking_data(data)

                metadata = {
                    'source_type': 'hdfs',
                    'hdfs_host': hdfs_host,
                    'hdfs_port': hdfs_port,
                    'hdfs_path': hdfs_path,
                    'file_size': file_status['length'],
                    'modification_time': file_status['modificationTime'],
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

    def _process_banking_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Process and validate banking compliance data

        Returns:
            Tuple of (processed_dataframe, warnings_list)
        """
        warnings = []

        try:
            if data.empty:
                raise ValueError("Data is empty")

            logger.info(f"Processing data with shape: {data.shape}")

            # Clean column names
            original_columns = data.columns.tolist()
            data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

            # Apply column mapping
            data = data.rename(columns=self.column_mapping)

            # Check for critical columns
            critical_columns = ['customer_id', 'account_id']
            missing_critical = [col for col in critical_columns if col not in data.columns]

            if missing_critical:
                warnings.append(f"Missing critical columns: {missing_critical}")

            # Data type conversions
            data = self._convert_data_types(data, warnings)

            # Data quality checks
            quality_issues = self._check_data_quality(data)
            warnings.extend(quality_issues)

            # Remove duplicates
            initial_count = len(data)
            data = data.drop_duplicates()
            if len(data) < initial_count:
                warnings.append(f"Removed {initial_count - len(data)} duplicate records")

            logger.info(f"Data processing completed. Final shape: {data.shape}")

            return data, warnings

        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            return data, [f"Processing error: {str(e)}"]

    def _convert_data_types(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """Convert data types according to banking schema"""
        try:
            # Date columns
            date_columns = [
                'date_of_birth', 'last_contact_date', 'kyc_expiry_date',
                'opening_date', 'closing_date', 'last_transaction_date',
                'last_system_transaction_date', 'maturity_date', 'last_statement_date',
                'dormancy_trigger_date', 'dormancy_period_start', 'dormancy_classification_date',
                'transfer_eligibility_date', 'last_contact_attempt_date', 'waiting_period_start',
                'waiting_period_end', 'transferred_to_ledger_date', 'transferred_to_cb_date',
                'created_date', 'updated_date'
            ]

            for col in date_columns:
                if col in data.columns:
                    before_count = data[col].notna().sum()
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    after_count = data[col].notna().sum()
                    if after_count < before_count:
                        warnings.append(f"Date conversion failed for {before_count - after_count} values in {col}")

            # Numeric columns
            numeric_columns = [
                'id_number', 'postal_code', 'phone_primary', 'phone_secondary',
                'balance_current', 'balance_available', 'balance_minimum',
                'interest_rate', 'interest_accrued', 'joint_account_holders',
                'dormancy_period_months', 'contact_attempts_made', 'cb_transfer_amount'
            ]

            for col in numeric_columns:
                if col in data.columns:
                    before_count = data[col].notna().sum()
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    after_count = data[col].notna().sum()
                    if after_count < before_count:
                        warnings.append(f"Numeric conversion failed for {before_count - after_count} values in {col}")

            # Boolean columns
            boolean_columns = ['is_joint_account', 'has_outstanding_facilities', 'auto_renewal']

            for col in boolean_columns:
                if col in data.columns:
                    data[col] = data[col].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])

            return data

        except Exception as e:
            warnings.append(f"Data type conversion error: {str(e)}")
            return data

    def _check_data_quality(self, data: pd.DataFrame) -> List[str]:
        """Perform data quality checks and return list of issues"""
        issues = []

        try:
            # Check for missing values in critical columns
            critical_cols = ['customer_id', 'account_id', 'account_status']
            for col in critical_cols:
                if col in data.columns:
                    missing_count = data[col].isnull().sum()
                    if missing_count > 0:
                        issues.append(f"Missing values in critical column '{col}': {missing_count} records")

            # Check for invalid balance values
            balance_cols = ['balance_current', 'balance_available']
            for col in balance_cols:
                if col in data.columns:
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        issues.append(f"Negative balance values in '{col}': {negative_count} records")

            # Check for future dates
            date_cols = ['date_of_birth', 'opening_date']
            current_date = pd.Timestamp.now()
            for col in date_cols:
                if col in data.columns:
                    future_count = (data[col] > current_date).sum()
                    if future_count > 0:
                        issues.append(f"Future dates in '{col}': {future_count} records")

            # Check for duplicate customer/account combinations
            if 'customer_id' in data.columns and 'account_id' in data.columns:
                duplicate_accounts = data.duplicated(subset=['customer_id', 'account_id']).sum()
                if duplicate_accounts > 0:
                    issues.append(f"Duplicate customer-account combinations: {duplicate_accounts} records")

            return issues

        except Exception as e:
            return [f"Data quality check error: {str(e)}"]

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

    # File reading methods
    def _read_csv(self, file_path_or_obj, **kwargs):
        """Read CSV file with robust encoding detection"""
        default_params = {
            'encoding': 'utf-8',
            'low_memory': False,
            'parse_dates': False,  # We'll handle dates manually
            'na_values': ['', 'NULL', 'null', 'NA', 'N/A', '#N/A', 'nan', 'NaN']
        }
        default_params.update(kwargs)

        try:
            return pd.read_csv(file_path_or_obj, **default_params)
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    default_params['encoding'] = encoding
                    return pd.read_csv(file_path_or_obj, **default_params)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode file with any supported encoding")

    def _read_excel(self, file_path_or_obj, **kwargs):
        """Read Excel file"""
        default_params = {
            'sheet_name': kwargs.get('sheet_name', 0),
            'header': kwargs.get('header', 0),
            'na_values': ['', 'NULL', 'null', 'NA', 'N/A', '#N/A', 'nan', 'NaN']
        }
        return pd.read_excel(file_path_or_obj, **default_params)

    def _read_json(self, file_path_or_obj, **kwargs):
        """Read JSON file"""
        default_params = {
            'orient': kwargs.get('orient', 'records'),
            'lines': kwargs.get('lines', False)
        }
        return pd.read_json(file_path_or_obj, **default_params)

    def _read_parquet(self, file_path_or_obj, **kwargs):
        """Read Parquet file"""
        return pd.read_parquet(file_path_or_obj, **kwargs)

    def _read_text(self, file_path_or_obj, **kwargs):
        """Read text file (assuming delimited format)"""
        delimiter = kwargs.get('delimiter', '\t')
        return pd.read_csv(file_path_or_obj, delimiter=delimiter, **kwargs)

    def get_upload_summary(self, result: UploadResult) -> Dict[str, Any]:
        """Generate a comprehensive upload summary"""
        if not result.success:
            return {
                'status': 'failed',
                'error': result.error,
                'processing_time': result.processing_time
            }

        data = result.data
        metadata = result.metadata or {}

        # Calculate additional statistics
        memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        # Column analysis
        column_analysis = {
            'total_columns': len(data.columns),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'text_columns': len(data.select_dtypes(include=['object']).columns),
            'date_columns': len(data.select_dtypes(include=['datetime']).columns),
        }

        # Data quality metrics
        quality_metrics = {
            'total_records': len(data),
            'complete_records': len(data.dropna()),
            'missing_values_total': data.isnull().sum().sum(),
            'duplicate_records': data.duplicated().sum(),
            'memory_usage_mb': round(memory_usage, 2)
        }

        # Banking-specific metrics
        banking_metrics = {}
        if 'account_status' in data.columns:
            banking_metrics['account_status_distribution'] = data['account_status'].value_counts().to_dict()

        if 'customer_type' in data.columns:
            banking_metrics['customer_type_distribution'] = data['customer_type'].value_counts().to_dict()

        if 'balance_current' in data.columns:
            balance_stats = data['balance_current'].describe()
            banking_metrics['balance_statistics'] = {
                'mean': round(balance_stats['mean'], 2) if not pd.isna(balance_stats['mean']) else 0,
                'median': round(data['balance_current'].median(), 2) if not pd.isna(
                    data['balance_current'].median()) else 0,
                'min': round(balance_stats['min'], 2) if not pd.isna(balance_stats['min']) else 0,
                'max': round(balance_stats['max'], 2) if not pd.isna(balance_stats['max']) else 0
            }

        return {
            'status': 'success',
            'metadata': metadata,
            'column_analysis': column_analysis,
            'quality_metrics': quality_metrics,
            'banking_metrics': banking_metrics,
            'warnings': result.warnings,
            'processing_time': result.processing_time,
            'recommendations': self._generate_recommendations(data, result.warnings)
        }

    def _generate_recommendations(self, data: pd.DataFrame, warnings: List[str]) -> List[str]:
        """Generate recommendations based on data analysis"""
        recommendations = []

        # Check data completeness
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_percentage > 10:
            recommendations.append(f"High missing data rate ({missing_percentage:.1f}%). Consider data cleaning.")

        # Check for critical banking columns
        critical_banking_cols = ['customer_id', 'account_id', 'account_status', 'balance_current']
        missing_critical = [col for col in critical_banking_cols if col not in data.columns]
        if missing_critical:
            recommendations.append(f"Missing critical banking columns: {missing_critical}. Verify data source.")

        # Check record count
        if len(data) < 100:
            recommendations.append("Small dataset detected. Consider aggregating more data for robust analysis.")
        elif len(data) > 100000:
            recommendations.append("Large dataset detected. Consider data sampling for initial analysis.")

        # Check for dormancy analysis readiness
        dormancy_cols = ['last_transaction_date', 'dormancy_status', 'account_status']
        has_dormancy_cols = any(col in data.columns for col in dormancy_cols)
        if has_dormancy_cols:
            recommendations.append("Dataset appears ready for dormancy analysis.")
        else:
            recommendations.append(
                "Limited dormancy analysis capability. Ensure transaction date fields are available.")

        return recommendations


# Streamlit UI Integration Functions
def create_upload_interface():
    """Create Streamlit interface for data upload"""
    st.title("🔄 Banking Compliance Data Upload")
    st.markdown("Upload your banking data from multiple sources for compliance analysis.")

    # Initialize uploader
    uploader = BankingComplianceUploader()

    # Upload method selection
    upload_method = st.selectbox(
        "Select Upload Method:",
        ["📄 Flat Files", "🔗 Google Drive", "☁️ Data Lake", "🗄️ HDFS"],
        help="Choose your preferred data source method"
    )

    if upload_method == "📄 Flat Files":
        return _handle_flat_file_upload(uploader)
    elif upload_method == "🔗 Google Drive":
        return _handle_drive_upload(uploader)
    elif upload_method == "☁️ Data Lake":
        return _handle_datalake_upload(uploader)
    elif upload_method == "🗄️ HDFS":
        return _handle_hdfs_upload(uploader)


def _handle_flat_file_upload(uploader: BankingComplianceUploader):
    """Handle flat file upload interface"""
    st.markdown("### 📄 File Upload Options")

    # File upload tabs
    tab1, tab2 = st.tabs(["Upload File", "From URL"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Upload CSV, Excel, JSON, or Parquet files"
        )

        if uploaded_file:
            file_extension = Path(uploaded_file.name).suffix.lower().lstrip('.')

            # Additional options
            with st.expander("Upload Options"):
                if file_extension == 'xlsx':
                    sheet_name = st.text_input("Sheet Name (optional)", value="0")
                elif file_extension == 'csv':
                    encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"])
                    delimiter = st.text_input("Delimiter", value=",")
                else:
                    sheet_name = None
                    encoding = None
                    delimiter = None

            if st.button("🚀 Upload File", type="primary"):
                with st.spinner("Processing file..."):
                    kwargs = {'file_extension': file_extension}
                    if sheet_name and file_extension == 'xlsx':
                        try:
                            kwargs['sheet_name'] = int(sheet_name)
                        except ValueError:
                            kwargs['sheet_name'] = sheet_name
                    if encoding and file_extension == 'csv':
                        kwargs['encoding'] = encoding
                    if delimiter and file_extension == 'csv':
                        kwargs['delimiter'] = delimiter

                    result = uploader.upload_data('file', uploaded_file, **kwargs)
                    _display_upload_results(result, uploader)

    with tab2:
        url = st.text_input(
            "Enter File URL:",
            placeholder="https://example.com/data.csv",
            help="Provide a direct link to your data file"
        )

        file_format = st.selectbox(
            "File Format:",
            ["csv", "xlsx", "json", "parquet"],
            help="Specify the file format if not clear from URL"
        )

        if url and st.button("📥 Import from URL", type="primary"):
            with st.spinner("Downloading and processing file..."):
                result = uploader.upload_data('file', url, file_format=file_format)
                _display_upload_results(result, uploader)


def _handle_drive_upload(uploader: BankingComplianceUploader):
    """Handle Google Drive upload interface"""
    st.markdown("### 🔗 Google Drive Upload")

    drive_link = st.text_input(
        "Google Drive Sharing Link:",
        placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing",
        help="Paste the Google Drive sharing link. Make sure the file is publicly accessible."
    )

    file_format = st.selectbox(
        "Expected File Format:",
        ["csv", "xlsx", "json"],
        help="Specify the expected file format"
    )

    if drive_link and st.button("📥 Import from Drive", type="primary"):
        with st.spinner("Accessing Google Drive file..."):
            result = uploader.upload_data('drive', drive_link, file_format=file_format)
            _display_upload_results(result, uploader)


def _handle_datalake_upload(uploader: BankingComplianceUploader):
    """Handle Data Lake upload interface"""
    st.markdown("### ☁️ Data Lake Upload")

    platform = st.selectbox("Platform:", ["Azure", "AWS S3"])

    if platform == "Azure":
        st.markdown("#### Azure Data Lake Configuration")
        account_name = st.text_input("Account Name:")
        account_key = st.text_input("Account Key:", type="password")
        container_name = st.text_input("Container Name:")
        blob_path = st.text_input("Blob Path:", placeholder="data/banking_data.csv")

        if all([account_name, account_key, container_name, blob_path]):
            if st.button("📥 Import from Azure", type="primary"):
                with st.spinner("Accessing Azure Data Lake..."):
                    result = uploader.upload_data(
                        'datalake', blob_path,
                        platform='azure',
                        account_name=account_name,
                        account_key=account_key,
                        container_name=container_name
                    )
                    _display_upload_results(result, uploader)

    elif platform == "AWS S3":
        st.markdown("#### AWS S3 Configuration")
        bucket_name = st.text_input("Bucket Name:")
        aws_access_key_id = st.text_input("Access Key ID:")
        aws_secret_access_key = st.text_input("Secret Access Key:", type="password")
        object_key = st.text_input("Object Key:", placeholder="data/banking_data.csv")
        region = st.text_input("Region:", value="us-east-1")

        if all([bucket_name, aws_access_key_id, aws_secret_access_key, object_key]):
            if st.button("📥 Import from S3", type="primary"):
                with st.spinner("Accessing AWS S3..."):
                    result = uploader.upload_data(
                        'datalake', object_key,
                        platform='aws',
                        bucket_name=bucket_name,
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region=region
                    )
                    _display_upload_results(result, uploader)


def _handle_hdfs_upload(uploader: BankingComplianceUploader):
    """Handle HDFS upload interface"""
    st.markdown("### 🗄️ HDFS Upload")

    hdfs_host = st.text_input("HDFS Host:", value="localhost")
    hdfs_port = st.number_input("HDFS Port:", value=9870, min_value=1, max_value=65535)
    hdfs_user = st.text_input("HDFS User:", value="hdfs")
    hdfs_path = st.text_input("HDFS File Path:", placeholder="/data/banking_data.csv")

    if hdfs_path and st.button("📥 Import from HDFS", type="primary"):
        with st.spinner("Accessing HDFS..."):
            result = uploader.upload_data(
                'hdfs', hdfs_path,
                hdfs_host=hdfs_host,
                hdfs_port=hdfs_port,
                hdfs_user=hdfs_user
            )
            _display_upload_results(result, uploader)


def _display_upload_results(result: UploadResult, uploader: BankingComplianceUploader):
    """Display upload results in Streamlit"""
    if result.success:
        st.success(f"✅ Data uploaded successfully!")

        # Get comprehensive summary
        summary = uploader.get_upload_summary(result)

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Records", f"{summary['quality_metrics']['total_records']:,}")
        with col2:
            st.metric("Columns", summary['column_analysis']['total_columns'])
        with col3:
            st.metric("Memory", f"{summary['quality_metrics']['memory_usage_mb']:.1f} MB")
        with col4:
            processing_time = summary.get('processing_time', 0)
            st.metric("Processing Time", f"{processing_time:.2f}s")

        # Display warnings if any
        if result.warnings:
            st.warning("⚠️ Upload completed with warnings:")
            for warning in result.warnings:
                st.write(f"• {warning}")

        # Display recommendations
        if summary.get('recommendations'):
            st.info("💡 Recommendations:")
            for rec in summary['recommendations']:
                st.write(f"• {rec}")

        # Data preview
        st.markdown("### 📊 Data Preview")
        st.dataframe(result.data.head(10), use_container_width=True)

        # Store in session state for other parts of the app
        st.session_state.uploaded_data = result.data
        st.session_state.upload_metadata = result.metadata
        st.session_state.upload_summary = summary

    else:
        st.error(f"❌ Upload failed: {result.error}")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'azure_account_name': 'your_azure_account',
        'azure_account_key': 'your_azure_key',
        'azure_container_name': 'compliance-data',
        'aws_bucket_name': 'banking-compliance',
        'aws_access_key_id': 'your_aws_key',
        'aws_secret_access_key': 'your_aws_secret',
        'hdfs_host': 'hadoop-cluster.example.com',
        'hdfs_port': 9870,
        'hdfs_user': 'hdfs'
    }

    # Initialize uploader
    uploader = BankingComplianceUploader(config)

    # Example usage - uncomment to test

    # Test flat file upload
    # result = uploader.upload_data('file', 'banking_compliance_dataset_5000_rows.csv')
    # print(f"Upload result: {result.success}")
    # if result.success:
    #     summary = uploader.get_upload_summary(result)
    #     print(f"Processed {summary['quality_metrics']['total_records']} records")

    # Test Google Drive upload
    # drive_link = "https://drive.google.com/file/d/your_file_id/view"
    # result = uploader.upload_data('drive', drive_link)

    # Test Azure Data Lake upload
    # result = uploader.upload_data('datalake', 'data/banking_data.csv',
    #                              platform='azure', account_name='account',
    #                              account_key='key', container_name='container')

    # Test HDFS upload
    # result = uploader.upload_data('hdfs', '/data/banking_data.csv',
    #                              hdfs_host='localhost', hdfs_port=9870)

    print("Data upload system initialized successfully!")
    print("Supported methods: Flat Files, Google Drive, Data Lake (Azure/AWS), HDFS")
    print("Use create_upload_interface() to create Streamlit UI")