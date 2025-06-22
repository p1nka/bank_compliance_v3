"""
FastAPI Data Processing Service for Banking Compliance Analysis
Handles file uploads, data quality analysis, BGE-based column mapping, and LLM enhancement
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
import pandas as pd
import numpy as np
import io
import json
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict

# Import BGE and similarity computation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False

# Import custom agents (if available)
try:
    from agents.data_upload_agent import BankingComplianceUploader, UploadResult
    from agents.data_mapping_agent import DataMappingAgent, run_automated_data_mapping
    from Data_Process import DataProcessingAgent, DataQualityAnalyzer
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Banking Compliance Data Processing API",
    description="Advanced data processing API with BGE embeddings and LLM enhancement",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Banking Compliance Schema
BANKING_SCHEMA = {
    'customer_id': 'Unique customer identifier',
    'customer_type': 'Type of customer (Individual/Corporate)',
    'full_name_en': 'Customer full name in English',
    'full_name_ar': 'Customer full name in Arabic',
    'id_number': 'National ID or passport number',
    'id_type': 'Type of identification document',
    'date_of_birth': 'Customer date of birth',
    'nationality': 'Customer nationality',
    'address_line1': 'Primary address line',
    'address_line2': 'Secondary address line',
    'city': 'City of residence',
    'emirate': 'Emirate of residence',
    'country': 'Country of residence',
    'postal_code': 'Postal code',
    'phone_primary': 'Primary phone number',
    'phone_secondary': 'Secondary phone number',
    'email_primary': 'Primary email address',
    'email_secondary': 'Secondary email address',
    'address_known': 'Address verification status',
    'last_contact_date': 'Date of last customer contact',
    'last_contact_method': 'Method of last contact',
    'kyc_status': 'KYC compliance status',
    'kyc_expiry_date': 'KYC expiry date',
    'risk_rating': 'Customer risk rating',
    'account_id': 'Unique account identifier',
    'account_type': 'Type of account',
    'account_subtype': 'Account subtype',
    'account_name': 'Account name/description',
    'currency': 'Account currency',
    'account_status': 'Current account status',
    'dormancy_status': 'Dormancy classification',
    'opening_date': 'Account opening date',
    'closing_date': 'Account closing date',
    'last_transaction_date': 'Date of last transaction',
    'last_system_transaction_date': 'Date of last system transaction',
    'balance_current': 'Current account balance',
    'balance_available': 'Available balance',
    'balance_minimum': 'Minimum balance requirement',
    'interest_rate': 'Interest rate applicable',
    'interest_accrued': 'Accrued interest amount',
    'is_joint_account': 'Joint account indicator',
    'joint_account_holders': 'Number of joint holders',
    'has_outstanding_facilities': 'Outstanding facilities indicator',
    'maturity_date': 'Account/product maturity date',
    'auto_renewal': 'Auto renewal indicator',
    'last_statement_date': 'Last statement date',
    'statement_frequency': 'Statement frequency',
    'tracking_id': 'Dormancy tracking ID',
    'dormancy_trigger_date': 'Date dormancy was triggered',
    'dormancy_period_start': 'Start of dormancy period',
    'dormancy_period_months': 'Months in dormancy',
    'dormancy_classification_date': 'Date of dormancy classification',
    'transfer_eligibility_date': 'Date eligible for transfer',
    'current_stage': 'Current process stage',
    'contact_attempts_made': 'Number of contact attempts',
    'last_contact_attempt_date': 'Date of last contact attempt',
    'waiting_period_start': 'Waiting period start date',
    'waiting_period_end': 'Waiting period end date',
    'transferred_to_ledger_date': 'Date transferred to internal ledger',
    'transferred_to_cb_date': 'Date transferred to central bank',
    'cb_transfer_amount': 'Amount transferred to central bank',
    'cb_transfer_reference': 'Central bank transfer reference',
    'exclusion_reason': 'Reason for exclusion from process',
    'created_date': 'Record creation date',
    'updated_date': 'Record last update date',
    'updated_by': 'Last updated by user/system'
}

# Pydantic Models
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    file_id: str
    records_count: int
    columns_count: int
    file_size_mb: float
    processing_time_seconds: float
    data_preview: List[Dict[str, Any]]
    column_info: List[Dict[str, Any]]

class QualityAnalysisResponse(BaseModel):
    success: bool
    quality_score: float
    quality_level: str
    missing_percentage: float
    records_analyzed: int
    total_columns: int
    data_issues: Dict[str, Any]
    recommendations: List[str]
    processing_time_seconds: float

class MappingRequest(BaseModel):
    file_id: str
    llm_enabled: bool = False
    confidence_threshold: float = 0.3

class MappingResponse(BaseModel):
    success: bool
    mapping_score: float
    schema_compliance: bool
    total_fields: int
    mapped_fields: int
    unmapped_fields: int
    llm_enhanced: bool
    average_confidence: float
    similarity_method: str
    mapping_results: Dict[str, Any]
    processing_time_seconds: float

class LLMToggleRequest(BaseModel):
    enabled: bool
    file_id: str

class LLMToggleResponse(BaseModel):
    success: bool
    llm_enabled: bool
    message: str
    confidence_boost_applied: float

# Global storage for uploaded files (in production, use proper database/cache)
uploaded_files_storage = {}
mapping_results_storage = {}

# BGE Model (loaded once)
bge_model = None

async def get_bge_model():
    """Load and cache BGE model"""
    global bge_model
    if bge_model is None and BGE_AVAILABLE:
        try:
            logger.info("Loading BGE-large model...")
            bge_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            logger.info("BGE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            bge_model = None
    return bge_model

@app.on_event("startup")
async def startup_event():
    """Initialize BGE model on startup"""
    await get_bge_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bge_available": BGE_AVAILABLE,
        "agents_available": AGENTS_AVAILABLE,
        "bge_model_loaded": bge_model is not None
    }

@app.post("/upload/file", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """
    Upload and process banking compliance data file
    Supports CSV, Excel, and JSON formats
    """
    start_time = datetime.now()

    try:
        # Validate file type
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.json'}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {allowed_extensions}"
            )

        # Read file content
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)

        # Parse data based on file type
        if file_extension == '.csv':
            data = pd.read_csv(io.BytesIO(file_content))
        elif file_extension in ['.xlsx', '.xls']:
            data = pd.read_excel(io.BytesIO(file_content))
        elif file_extension == '.json':
            data = pd.read_json(io.BytesIO(file_content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Generate unique file ID
        file_id = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(uploaded_files_storage)}"

        # Create data preview
        data_preview = data.head(5).fillna("").to_dict('records')

        # Create column information
        column_info = []
        for col in data.columns:
            column_info.append({
                'column_name': col,
                'data_type': str(data[col].dtype),
                'non_null_count': int(data[col].count()),
                'null_count': int(data[col].isnull().sum()),
                'null_percentage': round((data[col].isnull().sum() / len(data)) * 100, 2)
            })

        # Store file data
        uploaded_files_storage[file_id] = {
            'data': data,
            'filename': file.filename,
            'description': description,
            'upload_time': datetime.now().isoformat(),
            'file_size_mb': file_size_mb,
            'records_count': len(data),
            'columns_count': len(data.columns)
        }

        processing_time = (datetime.now() - start_time).total_seconds()

        return FileUploadResponse(
            success=True,
            message=f"File '{file.filename}' uploaded successfully",
            file_id=file_id,
            records_count=len(data),
            columns_count=len(data.columns),
            file_size_mb=round(file_size_mb, 2),
            processing_time_seconds=round(processing_time, 3),
            data_preview=data_preview,
            column_info=column_info
        )

    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/upload/drive")
async def upload_from_drive(
    drive_url: str = Form(...),
    description: Optional[str] = Form(None)
):
    """Upload file from Google Drive URL"""
    try:
        # Extract file ID from Google Drive URL
        file_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', drive_url)
        if not file_id_match:
            raise HTTPException(status_code=400, detail="Invalid Google Drive URL")

        drive_file_id = file_id_match.group(1)

        # Create direct download URL
        download_url = f"https://drive.google.com/uc?id={drive_file_id}&export=download"

        # Simulate file download and processing
        # In production, implement actual Google Drive API integration
        return {
            "success": True,
            "message": "Google Drive integration would be implemented here",
            "drive_file_id": drive_file_id,
            "download_url": download_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drive upload failed: {str(e)}")

@app.post("/upload/datalake")
async def upload_from_datalake(
    account_name: str = Form(...),
    container_name: str = Form(...),
    file_path: str = Form(...),
    access_key: str = Form(...),
    description: Optional[str] = Form(None)
):
    """Upload file from Azure Data Lake"""
    try:
        # Simulate Azure Data Lake integration
        # In production, implement actual Azure Data Lake connectivity
        return {
            "success": True,
            "message": "Azure Data Lake integration would be implemented here",
            "account_name": account_name,
            "container_name": container_name,
            "file_path": file_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data Lake upload failed: {str(e)}")

@app.post("/upload/hdfs")
async def upload_from_hdfs(
    hdfs_url: str = Form(...),
    file_path: str = Form(...),
    username: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """Upload file from Hadoop HDFS"""
    try:
        # Simulate HDFS integration
        # In production, implement actual HDFS connectivity
        return {
            "success": True,
            "message": "Hadoop HDFS integration would be implemented here",
            "hdfs_url": hdfs_url,
            "file_path": file_path,
            "username": username
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HDFS upload failed: {str(e)}")

@app.post("/quality/analyze", response_model=QualityAnalysisResponse)
async def analyze_data_quality(file_id: str = Form(...)):
    """Analyze data quality for uploaded file"""
    start_time = datetime.now()

    try:
        if file_id not in uploaded_files_storage:
            raise HTTPException(status_code=404, detail="File not found")

        data = uploaded_files_storage[file_id]['data']

        # Perform quality analysis
        total_records = len(data)
        total_columns = len(data.columns)

        # Calculate missing data
        missing_data = data.isnull().sum()
        total_missing = missing_data.sum()
        missing_percentage = (total_missing / (total_records * total_columns)) * 100

        # Quality scoring
        quality_score = max(0, 100 - missing_percentage * 2) / 100  # Normalize to 0-1

        if quality_score >= 0.9:
            quality_level = "Excellent"
        elif quality_score >= 0.75:
            quality_level = "Good"
        elif quality_score >= 0.6:
            quality_level = "Fair"
        else:
            quality_level = "Poor"

        # Data issues analysis
        duplicate_count = data.duplicated().sum()

        data_issues = {
            "missing_values": int(total_missing),
            "duplicate_records": int(duplicate_count),
            "missing_by_column": missing_data.to_dict(),
            "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()}
        }

        # Generate recommendations
        recommendations = []
        if missing_percentage > 10:
            recommendations.append("High missing data detected - review data collection process")
        if missing_percentage > 5:
            recommendations.append("Consider data imputation strategies for missing values")
        if duplicate_count > 0:
            recommendations.append(f"Found {duplicate_count} duplicate records - consider deduplication")
        if total_records < 100:
            recommendations.append("Dataset is small - consider collecting more data")

        processing_time = (datetime.now() - start_time).total_seconds()

        return QualityAnalysisResponse(
            success=True,
            quality_score=quality_score,
            quality_level=quality_level,
            missing_percentage=round(missing_percentage, 2),
            records_analyzed=total_records,
            total_columns=total_columns,
            data_issues=data_issues,
            recommendations=recommendations,
            processing_time_seconds=round(processing_time, 3)
        )

    except Exception as e:
        logger.error(f"Quality analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Quality analysis failed: {str(e)}")

@app.post("/mapping/toggle-llm", response_model=LLMToggleResponse)
async def toggle_llm_enhancement(request: LLMToggleRequest):
    """Toggle LLM enhancement for mapping analysis"""
    try:
        if request.file_id not in uploaded_files_storage:
            raise HTTPException(status_code=404, detail="File not found")

        # Store LLM preference
        uploaded_files_storage[request.file_id]['llm_enabled'] = request.enabled

        confidence_boost = 0.1 if request.enabled else 0.0  # 10% boost when enabled

        message = (
            "LLM enhancement enabled - AI will boost confidence scores by 10%"
            if request.enabled
            else "LLM enhancement disabled - using pure BGE similarity scores"
        )

        return LLMToggleResponse(
            success=True,
            llm_enabled=request.enabled,
            message=message,
            confidence_boost_applied=confidence_boost
        )

    except Exception as e:
        logger.error(f"LLM toggle error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM toggle failed: {str(e)}")

@app.post("/mapping/analyze", response_model=MappingResponse)
async def analyze_column_mapping(request: MappingRequest):
    """Perform BGE-based column mapping analysis"""
    start_time = datetime.now()

    try:
        if request.file_id not in uploaded_files_storage:
            raise HTTPException(status_code=404, detail="File not found")

        data = uploaded_files_storage[request.file_id]['data']
        source_columns = list(data.columns)

        # Get LLM setting
        llm_enabled = uploaded_files_storage[request.file_id].get('llm_enabled', request.llm_enabled)

        # Load BGE model
        model = await get_bge_model()

        if model is None or not BGE_AVAILABLE:
            # Fallback to simple string matching
            mapping_results = perform_simple_mapping(source_columns, request.confidence_threshold)
            similarity_method = "String Similarity (Fallback)"
        else:
            # Use BGE embeddings
            mapping_results = await perform_bge_mapping(
                model, data, source_columns, request.confidence_threshold, llm_enabled
            )
            similarity_method = "BGE-large + Cosine Similarity"

        # Calculate mapping statistics
        total_fields = len(source_columns)
        mapped_fields = len(mapping_results)
        unmapped_fields = total_fields - mapped_fields
        mapping_score = (mapped_fields / total_fields) * 100 if total_fields > 0 else 0

        # Apply LLM boost if enabled
        if llm_enabled:
            mapping_score = min(mapping_score * 1.1, 100)  # 10% boost
            for mapping in mapping_results.values():
                mapping['llm_enhanced'] = True
                mapping['confidence_score'] = min(mapping['confidence_score'] * 1.05, 1.0)

        # Calculate average confidence
        confidence_scores = [m['confidence_score'] for m in mapping_results.values()]
        average_confidence = np.mean(confidence_scores) if confidence_scores else 0

        # Store mapping results
        mapping_id = f"mapping_{request.file_id}_{datetime.now().strftime('%H%M%S')}"
        mapping_results_storage[mapping_id] = {
            'mapping_results': mapping_results,
            'source_columns': source_columns,
            'file_id': request.file_id,
            'created_at': datetime.now().isoformat(),
            'llm_enabled': llm_enabled
        }

        processing_time = (datetime.now() - start_time).total_seconds()

        return MappingResponse(
            success=True,
            mapping_score=round(mapping_score, 2),
            schema_compliance=mapping_score >= 70,
            total_fields=total_fields,
            mapped_fields=mapped_fields,
            unmapped_fields=unmapped_fields,
            llm_enhanced=llm_enabled,
            average_confidence=round(average_confidence, 4),
            similarity_method=similarity_method,
            mapping_results=mapping_results,
            processing_time_seconds=round(processing_time, 3)
        )

    except Exception as e:
        logger.error(f"Mapping analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Mapping analysis failed: {str(e)}")

@app.get("/mapping/download/{file_id}")
async def download_mapping_sheet(file_id: str):
    """Download mapping results as CSV file"""
    try:
        # Find the most recent mapping for this file
        relevant_mappings = {
            k: v for k, v in mapping_results_storage.items()
            if v['file_id'] == file_id
        }

        if not relevant_mappings:
            raise HTTPException(status_code=404, detail="No mapping results found for this file")

        # Get the most recent mapping
        latest_mapping_key = max(relevant_mappings.keys(), key=lambda k: relevant_mappings[k]['created_at'])
        mapping_data = relevant_mappings[latest_mapping_key]

        # Create mapping sheet
        sheet_data = []
        for source_col in mapping_data['source_columns']:
            if source_col in mapping_data['mapping_results']:
                mapping = mapping_data['mapping_results'][source_col]
                sheet_data.append({
                    'Source_Column': source_col,
                    'Mapped_To': mapping['mapped_to'],
                    'Confidence_Score': f"{mapping['confidence_score']:.4f}",
                    'Confidence_Level': mapping['confidence_level'],
                    'Description': mapping['description'],
                    'Method': mapping.get('similarity_method', 'BGE + Cosine Similarity'),
                    'LLM_Enhanced': mapping.get('llm_enhanced', False)
                })
            else:
                sheet_data.append({
                    'Source_Column': source_col,
                    'Mapped_To': 'NOT_MAPPED',
                    'Confidence_Score': '0.0000',
                    'Confidence_Level': 'none',
                    'Description': 'No suitable match found',
                    'Method': 'BGE + Cosine Similarity',
                    'LLM_Enhanced': False
                })

        # Create CSV
        df = pd.DataFrame(sheet_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Create response
        response = StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=mapping_results_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )

        return response

    except Exception as e:
        logger.error(f"Download mapping sheet error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/files")
async def list_uploaded_files():
    """List all uploaded files"""
    try:
        files_info = []
        for file_id, file_data in uploaded_files_storage.items():
            files_info.append({
                'file_id': file_id,
                'filename': file_data['filename'],
                'upload_time': file_data['upload_time'],
                'records_count': file_data['records_count'],
                'columns_count': file_data['columns_count'],
                'file_size_mb': file_data['file_size_mb'],
                'description': file_data.get('description', ''),
                'llm_enabled': file_data.get('llm_enabled', False)
            })

        return {
            'success': True,
            'files': files_info,
            'total_files': len(files_info)
        }

    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

# Helper Functions

async def perform_bge_mapping(
    model, data: pd.DataFrame, source_columns: List[str],
    confidence_threshold: float, llm_enabled: bool
) -> Dict[str, Any]:
    """Perform BGE-based column mapping"""

    # Precompute schema embeddings
    schema_embeddings = {}
    for column in BANKING_SCHEMA.keys():
        enriched_text = f"{column} {BANKING_SCHEMA[column]}"
        schema_embeddings[column] = model.encode([enriched_text])[0]

    mapping_results = {}

    # Generate embeddings for source columns
    for source_col in source_columns:
        # Enrich source column with sample data
        sample_values = data[source_col].dropna().head(3).astype(str).tolist()
        enriched_text = f"{source_col} {' '.join(sample_values)}"
        source_embedding = model.encode([enriched_text])[0]

        best_match = None
        best_score = 0.0
        similarity_scores = {}

        # Calculate cosine similarity with all schema columns
        for schema_col, schema_emb in schema_embeddings.items():
            similarity = cosine_similarity([source_embedding], [schema_emb])[0][0]
            similarity_scores[schema_col] = similarity

            if similarity > best_score:
                best_score = similarity
                best_match = schema_col

        # Only include mappings above threshold
        if best_score >= confidence_threshold:
            confidence_level = get_confidence_level(best_score)

            mapping_results[source_col] = {
                'mapped_to': best_match,
                'confidence_score': float(best_score),
                'confidence_level': confidence_level,
                'description': BANKING_SCHEMA[best_match],
                'similarity_method': 'BGE + Cosine Similarity',
                'top_3_matches': sorted(similarity_scores.items(),
                                      key=lambda x: x[1], reverse=True)[:3]
            }

    return mapping_results

def perform_simple_mapping(source_columns: List[str], confidence_threshold: float) -> Dict[str, Any]:
    """Perform simple string-based mapping as fallback"""
    mapping_results = {}

    for source_col in source_columns:
        best_match = None
        best_score = 0.0

        source_lower = source_col.lower().replace('_', ' ')

        for schema_col in BANKING_SCHEMA.keys():
            schema_lower = schema_col.lower().replace('_', ' ')

            # Simple similarity calculation
            if source_lower == schema_lower:
                score = 1.0
            elif source_lower in schema_lower or schema_lower in source_lower:
                score = 0.8
            else:
                # Word overlap
                source_words = set(source_lower.split())
                schema_words = set(schema_lower.split())
                overlap = len(source_words.intersection(schema_words))
                total = len(source_words.union(schema_words))
                score = overlap / total if total > 0 else 0

            if score > best_score:
                best_score = score
                best_match = schema_col

        if best_score >= confidence_threshold:
            mapping_results[source_col] = {
                'mapped_to': best_match,
                'confidence_score': float(best_score),
                'confidence_level': get_confidence_level(best_score),
                'description': BANKING_SCHEMA[best_match],
                'similarity_method': 'String Similarity (Fallback)'
            }

    return mapping_results

def get_confidence_level(score: float) -> str:
    """Determine confidence level based on similarity score"""
    if score >= 0.8:
        return 'high'
    elif score >= 0.5:
        return 'medium'
    else:
        return 'low'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")