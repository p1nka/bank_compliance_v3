# banking_compliance_unified_api.py
import pandas as pd
import numpy as np
import io
import json
import os
import tempfile
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import re

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import BGE and similarity computation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    logging.warning("BGE dependencies (sentence-transformers, scikit-learn) not available. Some functionalities may be limited.")

# Import Llama CPP (for local LLM)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    # NOTE: Model path for Llama might need to be absolute or configured via environment variables.
    # The 'mistral-7b-instruct-v0.1.Q4_K_M.gguf' file should be accessible.
    # For production, consider using a dedicated inference service or a single global instance.
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("Llama CPP not available. LLM-based recommendations in agents will be limited.")

# Import custom agents and schema manager
# Ensure relative imports are handled correctly based on your project structure
from agents.Data_Process import ProductionDataProcessingAgent
# Note: Dormant_agent.py has a top-level `llm("Tell me a joke...")` call which can cause issues.
# It's recommended to remove such calls from agent files if they are not inside a function.
from agents.Dormant_agent import DormantAccountOrchestrator, ActivityStatus, AccountType, CustomerTier, ContactMethod
from agents.compliance_verification_agent import ComplianceOrchestrator
from agents.unified_schema import UnifiedBankingSchemaManager
from mcp_client import MCPClient
from agents.memory_agent import HybridMemoryAgent, MemoryContext, MemoryBucket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Banking Compliance API",
    description="Comprehensive API for data processing, dormancy analysis, and compliance verification.",
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

# Global instances (initialized at startup)
unified_schema_manager: Optional[UnifiedBankingSchemaManager] = None
mcp_client_instance: Optional[MCPClient] = None
memory_agent_instance: Optional[HybridMemoryAgent] = None
data_processing_agent: Optional[ProductionDataProcessingAgent] = None
dormancy_orchestrator: Optional[DormantAccountOrchestrator] = None
compliance_orchestrator: Optional[ComplianceOrchestrator] = None
bge_model_instance: Optional[SentenceTransformer] = None
# llm_model_instance is not directly used here but agents internally initialize Llama.

# --- Pydantic Models for API ---
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

class DormancyAnalysisRequest(BaseModel):
    file_id: str
    # Add other parameters specific to dormancy analysis if needed

class DormancyAnalysisResponse(BaseModel):
    success: bool
    summary: str
    results: Dict[str, Any]
    processing_time_seconds: float

class ComplianceVerificationRequest(BaseModel):
    file_id: str
    # Add other parameters specific to compliance verification if needed

class ComplianceVerificationResponse(BaseModel):
    success: bool
    summary: str
    results: Dict[str, Any]
    processing_time_seconds: float


# --- In-memory storage for uploaded files (for demonstration) ---
uploaded_files_storage = {}

# --- Helper Functions (Copied/adapted from original banking_compliance_fastapi.py) ---
def get_confidence_level(score: float) -> str:
    """Determines confidence level based on similarity score."""
    if score >= 0.8: return 'high'
    elif score >= 0.5: return 'medium'
    else: return 'low'

# This function is not directly used by the ProductionDataProcessingAgent in its current form
# as the agent performs its own BGE mapping internally. It's kept for reference from original.
async def perform_bge_mapping(
    model, data: pd.DataFrame, source_columns: List[str],
    confidence_threshold: float, llm_enabled: bool # llm_enabled is for conceptual LLM-based mapping boost
) -> Dict[str, Any]:
    """Perform BGE-based column mapping (simplified, relies on global schema manager)."""
    global unified_schema_manager
    if not unified_schema_manager:
        logger.error("Schema manager not initialized for BGE mapping.")
        return {}

    schema_embeddings = {}
    for column_name, field_def in unified_schema_manager.fields.items():
        enriched_text = f"{column_name} {field_def.description} {' '.join(field_def.keywords)} {' '.join(field_def.aliases)}"
        schema_embeddings[column_name] = model.encode([enriched_text], normalize_embeddings=True)[0]

    mapping_results = {}
    for source_col in source_columns:
        # Avoid processing if column is not found in data
        if source_col not in data.columns:
            continue

        sample_values = data[source_col].dropna().head(3).astype(str).tolist()
        enriched_text = f"{source_col} {' '.join(sample_values)}"
        source_embedding = model.encode([enriched_text], normalize_embeddings=True)[0]

        best_match = None
        best_score = 0.0
        similarity_scores = {}

        for schema_col, schema_emb in schema_embeddings.items():
            similarity = float(cosine_similarity([source_embedding], [schema_emb])[0][0])
            similarity_scores[schema_col] = similarity
            if similarity > best_score:
                best_score = similarity
                best_match = schema_col

        if best_score >= confidence_threshold:
            confidence_level = get_confidence_level(best_score)
            mapped_field_def = unified_schema_manager.get_field(best_match)
            mapping_results[source_col] = {
                'mapped_to': best_match,
                'confidence_score': best_score,
                'confidence_level': confidence_level,
                'description': mapped_field_def.description if mapped_field_def else "N/A",
                'similarity_method': 'BGE + Cosine Similarity',
                'top_3_matches': sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            }
    return mapping_results

# This function is not directly used by the ProductionDataProcessingAgent in its current form
# as the agent performs its own mapping internally. It's kept for reference from original.
def perform_simple_mapping(source_columns: List[str], confidence_threshold: float) -> Dict[str, Any]:
    """Perform simple string-based mapping as fallback (simplified, relies on global schema manager)."""
    global unified_schema_manager
    if not unified_schema_manager:
        logger.error("Schema manager not initialized for simple mapping.")
        return {}

    mapping_results = {}
    schema_fields_info = {
        field.name.lower().replace('_', ' '): field.name
        for field in unified_schema_manager.fields.values()
    }

    for source_col in source_columns:
        best_match = None
        best_score = 0.0
        source_lower = source_col.lower().replace('_', ' ')

        for schema_clean, schema_original in schema_fields_info.items():
            score = 0.0
            if source_lower == schema_clean:
                score = 1.0
            elif source_lower in schema_clean or schema_clean in source_lower:
                score = 0.8
            else:
                source_words = set(source_lower.split())
                schema_words = set(schema_clean.split())
                overlap = len(source_words.intersection(schema_words))
                total = len(source_words.union(schema_words))
                score = overlap / total if total > 0 else 0

            if score > best_score:
                best_score = score
                best_match = schema_original

        if best_score >= confidence_threshold:
            mapped_field_def = unified_schema_manager.get_field(best_match)
            mapping_results[source_col] = {
                'mapped_to': best_match,
                'confidence_score': float(best_score),
                'confidence_level': get_confidence_level(best_score),
                'description': mapped_field_def.description if mapped_field_def else "N/A",
                'similarity_method': 'String Similarity (Fallback)'
            }
    return mapping_results


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initializes global components and agents on application startup."""
    global unified_schema_manager, mcp_client_instance, memory_agent_instance, \
           data_processing_agent, dormancy_orchestrator, compliance_orchestrator, \
           bge_model_instance, LLAMA_CPP_AVAILABLE, BGE_AVAILABLE # <--- Declare globals here

    logger.info("Initializing application components...")

    # ... (other initializations) ...

    # 5. Initialize Llama CPP model availability and instance (if LLAMA_CPP_AVAILABLE is True initially)
    if LLAMA_CPP_AVAILABLE: # This checks the module-level variable
        try:
            model_path = os.getenv("LLAMA_MODEL_PATH", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
            # For this example, I'll assume the model is directly accessible or configured.
            # In a real setup, ensure this path is correct relative to the running FastAPI app.
            # Note: Agents still load their own Llama instances internally.
            # This global instance is here for potential future injection or direct use.
            # llm_model_instance = Llama(model_path=model_path, n_ctx=2048, verbose=False) # This line was in the generated code, but not explicitly used by agents currently
            logger.info(f"Llama CPP library available. Agents may load model: '{model_path}'.")
        except Exception as e:
            logger.error(f"Failed to confirm Llama CPP model path or setup: {e}")
            LLAMA_CPP_AVAILABLE = False # Set global to False if an issue arises
    else:
        logger.info("Llama CPP not available (due to import error).")

    # ... (initialization of dormancy_orchestrator and compliance_orchestrator) ...
    # These agents will attempt to load Llama internally if LLAMA_CPP_AVAILABLE is True.

    # 8. Load BGE Model (if BGE_AVAILABLE is True initially and config enables it)
    if BGE_AVAILABLE and data_processing_agent.config.get("enable_bge", True):
        try:
            bge_model_instance = SentenceTransformer('BAAI/bge-large-en-v1.5')
            logger.info("BGE model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            BGE_AVAILABLE = False # <--- This correctly updates the global flag
            bge_model_instance = None
    else:
        logger.info("BGE model loading skipped (BGE not available or disabled).")

    logger.info("Application startup complete.")

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status and component availability."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bge_available": BGE_AVAILABLE,
        "llama_cpp_available": LLAMA_CPP_AVAILABLE, # Indicates if the library is installed
        "mcp_client_connected": mcp_client_instance.connected if mcp_client_instance else False,
        "memory_agent_stats": memory_agent_instance.get_statistics() if memory_agent_instance else {}
    }


# --- Data Processing Endpoints (Adapted from banking_compliance_fastapi.py) ---
@app.post("/upload/file", response_model=FileUploadResponse)
async def upload_file_endpoint(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    user_id: str = Form("default_user"),
    session_id: Optional[str] = Form(None)
):
    """
    Uploads and processes banking compliance data file.
    Supports CSV, Excel, and JSON formats.
    """
    if not data_processing_agent:
        raise HTTPException(status_code=500, detail="Data processing agent not initialized.")

    try:
        # ProductionDataProcessingAgent's upload_data expects io.IOBase or path
        # file.file is a SpooledTemporaryFile, which acts as io.IOBase
        upload_result = await data_processing_agent.upload_data(
            upload_method="file",
            source=file.file,
            user_id=user_id,
            session_id=session_id or f"sess_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        if not upload_result.success or upload_result.data is None:
            raise HTTPException(status_code=500, detail=f"File upload failed: {upload_result.error}")

        # Store the processed dataframe in in-memory storage using file_id for subsequent steps
        file_id = upload_result.metadata.get("operation_id", f"file_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        uploaded_files_storage[file_id] = {
            'data': upload_result.data,
            'filename': file.filename,
            'description': description,
            'upload_time': datetime.now().isoformat(),
            'file_size_mb': round(upload_result.metadata.get('file_size', 0) / (1024 * 1024), 2),
            'records_count': len(upload_result.data),
            'columns_count': len(upload_result.data.columns)
        }

        data_preview = upload_result.data.head(5).fillna("").to_dict('records')
        column_info = []
        for col in upload_result.data.columns:
            column_info.append({
                'column_name': col,
                'data_type': str(upload_result.data[col].dtype),
                'non_null_count': int(upload_result.data[col].count()),
                'null_count': int(upload_result.data[col].isnull().sum()),
                'null_percentage': round((upload_result.data[col].isnull().sum() / len(upload_result.data)) * 100, 2)
            })

        return FileUploadResponse(
            success=True,
            message=f"File '{file.filename}' uploaded successfully",
            file_id=file_id,
            records_count=len(upload_result.data),
            columns_count=len(upload_result.data.columns),
            file_size_mb=round(upload_result.metadata.get('file_size', 0) / (1024 * 1024), 2),
            processing_time_seconds=upload_result.processing_time or 0.0,
            data_preview=data_preview,
            column_info=column_info
        )

    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/data/quality/analyze", response_model=QualityAnalysisResponse)
async def analyze_data_quality_endpoint(file_id: str = Form(...), user_id: str = Form("default_user"), session_id: Optional[str] = Form(None)):
    """Analyzes data quality for a previously uploaded file."""
    if not data_processing_agent:
        raise HTTPException(status_code=500, detail="Data processing agent not initialized.")
    if file_id not in uploaded_files_storage:
        raise HTTPException(status_code=404, detail="File not found in storage. Please upload the file first.")

    data = uploaded_files_storage[file_id]['data']

    quality_result = await data_processing_agent.analyze_data_quality(
        data=data,
        user_id=user_id,
        session_id=session_id or f"sess_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    if not quality_result.success:
        raise HTTPException(status_code=500, detail=f"Quality analysis failed: {quality_result.error}")

    # Store quality results
    uploaded_files_storage[f"quality_results_{file_id}"] = asdict(quality_result)

    return QualityAnalysisResponse(
        success=quality_result.success,
        quality_score=quality_result.overall_score,
        quality_level=quality_result.quality_level,
        missing_percentage=quality_result.missing_data_analysis.get("overall_missing_percentage", 0.0),
        records_analyzed=quality_result.missing_data_analysis.get("total_records", 0),
        total_columns=len(data.columns),
        data_issues=quality_result.detailed_metrics,
        recommendations=quality_result.recommendations,
        processing_time_seconds=quality_result.processing_time or 0.0
    )


@app.post("/mapping/analyze", response_model=MappingResponse)
async def analyze_column_mapping_endpoint(request: MappingRequest, user_id: str = Form("default_user"), session_id: Optional[str] = Form(None)):
    """
    Performs BGE-based column mapping analysis for an uploaded file.
    Note: The 'llm_enabled' flag here is conceptual, as the DataProcessAgent internally manages LLM/BGE use for mapping.
    """
    if not data_processing_agent:
        raise HTTPException(status_code=500, detail="Data processing agent not initialized.")
    if request.file_id not in uploaded_files_storage:
        raise HTTPException(status_code=404, detail="File not found in storage. Please upload the file first.")

    data = uploaded_files_storage[request.file_id]['data']

    mapping_result = await data_processing_agent.map_columns_to_schema(
        data=data,
        user_id=user_id,
        session_id=session_id or f"sess_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        use_bge=bge_model_instance is not None # Only use BGE if model is loaded globally
    )

    if not mapping_result.success:
        raise HTTPException(status_code=500, detail=f"Mapping analysis failed: {mapping_result.error}")

    # Store mapping results for download and subsequent steps
    mapping_results_storage_key = f"mapping_results_{request.file_id}"
    uploaded_files_storage[mapping_results_storage_key] = asdict(mapping_result)

    confidence_scores_list = list(mapping_result.confidence_scores.values())
    average_confidence = np.mean(confidence_scores_list) if confidence_scores_list else 0

    return MappingResponse(
        success=mapping_result.success,
        mapping_score=(len(mapping_result.field_mappings) / len(data.columns) * 100) if len(data.columns) > 0 else 0,
        schema_compliance=len(mapping_result.missing_required_fields) == 0, # Assuming full compliance if no required fields are missing
        total_fields=len(data.columns),
        mapped_fields=len(mapping_result.field_mappings),
        unmapped_fields=len(mapping_result.unmapped_columns),
        llm_enhanced=request.llm_enabled, # This reflects the user's toggle, not necessarily agent's actual LLM use.
        average_confidence=round(average_confidence, 4),
        similarity_method=mapping_result.mapping_method,
        mapping_results=mapping_result.field_mappings, # Only sending the mapped fields for brevity
        processing_time_seconds=mapping_result.processing_time or 0.0
    )


@app.post("/mapping/toggle-llm", response_model=LLMToggleResponse)
async def toggle_llm_enhancement_endpoint(request: LLMToggleRequest):
    """
    Toggles LLM enhancement for mapping analysis. This is a conceptual flag as the underlying
    DataProcessAgent's BGE mapping internally decides LLM usage.
    """
    if request.file_id not in uploaded_files_storage:
        raise HTTPException(status_code=404, detail="File not found")

    # Update stored file metadata if needed for future calls
    # This preference is primarily for the UI and might influence agent behavior if implemented in DataProcessAgent.
    uploaded_files_storage[request.file_id]['llm_enabled_for_mapping_preference'] = request.enabled

    confidence_boost = 0.1 if request.enabled else 0.0
    message = (
        "LLM enhancement for mapping preference enabled. Actual agent behavior depends on its implementation."
        if request.enabled
        else "LLM enhancement for mapping preference disabled."
    )

    return LLMToggleResponse(
        success=True,
        llm_enabled=request.enabled,
        message=message,
        confidence_boost_applied=confidence_boost
    )

@app.get("/mapping/download/{file_id}")
async def download_mapping_sheet_endpoint(file_id: str):
    """Downloads mapping results as a CSV file for a given uploaded file."""

    # Retrieve mapping results
    mapping_results_key = f"mapping_results_{file_id}"
    if mapping_results_key not in uploaded_files_storage:
        raise HTTPException(status_code=404, detail="No mapping results found for this file. Please run mapping analysis first.")

    mapping_data = uploaded_files_storage[mapping_results_key]

    sheet_data = []
    # Use original columns from the uploaded file for the sheet
    original_data_cols = uploaded_files_storage[file_id]['data'].columns.tolist()

    for source_col in original_data_cols:
        mapped_to = mapping_data['field_mappings'].get(source_col)
        confidence = mapping_data['confidence_scores'].get(source_col, 0.0)

        field_def = unified_schema_manager.get_field(mapped_to) if mapped_to else None

        sheet_data.append({
            'Source_Column': source_col,
            'Mapped_To': mapped_to if mapped_to else 'NOT_MAPPED',
            'Confidence_Score': f"{confidence:.4f}",
            'Confidence_Level': get_confidence_level(confidence),
            'Field_Description': field_def.description if field_def else 'No suitable match found',
            'Similarity_Method': mapping_data.get('mapping_method', 'Unknown'),
            'LLM_Enhanced': mapping_data.get('llm_enhanced', False) # Check if LLM was involved in mapping
        })

    df = pd.DataFrame(sheet_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    response = StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=mapping_results_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )
    return response

@app.get("/files")
async def list_uploaded_files_endpoint():
    """Lists all uploaded files currently in memory storage."""
    files_info = []
    for file_id, file_data in uploaded_files_storage.items():
        # Exclude mapping and quality results which are also stored in uploaded_files_storage
        if not file_id.startswith("mapping_results_") and not file_id.startswith("quality_results_") and not file_id.startswith("dormancy_results_") and not file_id.startswith("compliance_results_"):
            files_info.append({
                'file_id': file_id,
                'filename': file_data.get('filename', 'N/A'),
                'upload_time': file_data.get('upload_time', 'N/A'),
                'records_count': file_data.get('records_count', 0),
                'columns_count': file_data.get('columns_count', 0),
                'file_size_mb': file_data.get('file_size_mb', 0.0),
                'description': file_data.get('description', ''),
                'llm_enabled_preference': file_data.get('llm_enabled_for_mapping_preference', False)
            })
    return {
        'success': True,
        'files': files_info,
        'total_files': len(files_info)
    }

# --- Dormancy Analysis Endpoint ---
@app.post("/dormancy/analyze", response_model=DormancyAnalysisResponse)
async def analyze_dormancy_endpoint(request: DormancyAnalysisRequest, user_id: str = Form("default_user")):
    """
    Performs dormancy analysis on processed account data.
    Requires data to be uploaded and processed (or mapped) first.
    """
    if not dormancy_orchestrator:
        raise HTTPException(status_code=500, detail="Dormancy analysis agent not initialized.")
    if request.file_id not in uploaded_files_storage:
        raise HTTPException(status_code=404, detail="Data for dormancy analysis not found. Please upload and process/map data first.")

    # Retrieve the original DataFrame associated with the file_id
    processed_data_df = uploaded_files_storage[request.file_id]['data']

    # Convert DataFrame rows to list of dictionaries, as dormancy agents expect this format.
    accounts_data_list = processed_data_df.to_dict(orient='records')

    # Ensure necessary fields are present and correctly typed for DormantAgent's internal logic.
    # This is a crucial step to adapt DataFrame data to agent's expected format.
    for account in accounts_data_list:
        # Example: Ensure dates are datetime objects for calculation
        if 'last_transaction_date' in account:
            if isinstance(account['last_transaction_date'], str):
                try: account['last_activity_date'] = datetime.fromisoformat(account['last_transaction_date'])
                except ValueError: account['last_activity_date'] = datetime.now() - timedelta(days=1000) # Default to old
            elif isinstance(account['last_transaction_date'], pd.Timestamp):
                 account['last_activity_date'] = account['last_transaction_date'].to_pydatetime()
            else: account['last_activity_date'] = datetime.now() - timedelta(days=1000) # Default if not found
        else: account['last_activity_date'] = datetime.now() - timedelta(days=1000) # Default if not found

        # Ensure account_value is numeric (mapping 'balance_current' or similar)
        if 'balance_current' in account:
            try: account['account_value'] = float(account['balance_current'])
            except (ValueError, TypeError): account['account_value'] = 0.0
        else: account['account_value'] = 0.0

        # Map 'account_type' to DormantAgent's AccountType Enum values if necessary
        if 'account_type' in account and isinstance(account['account_type'], str):
            # Convert string to enum for the dormant agent if needed. Default to 'DEMAND_DEPOSIT'.
            try: account['account_type'] = AccountType[account['account_type'].upper()].value
            except KeyError: account['account_type'] = AccountType.DEMAND_DEPOSIT.value
        else: account['account_type'] = AccountType.DEMAND_DEPOSIT.value

        # Mock maturity date if not present for fixed deposit agent
        if account.get('account_type') == AccountType.FIXED_DEPOSIT.value and 'maturity_date' not in account:
            account['maturity_date'] = datetime.now() - timedelta(days=1500) # Assume past maturity for demo

        # Mock transition history for dormant-to-active transitions agent
        if 'transition_history' not in account:
            account['transition_history'] = []

        # Ensure 'customer_type' is available for contact attempts agent
        if 'customer_type' not in account:
            account['customer_type'] = 'individual'

        # Ensure 'contact_attempts' is available
        if 'contact_attempts_made' in account:
            try: account['contact_attempts'] = int(account['contact_attempts_made'])
            except (ValueError, TypeError): account['contact_attempts'] = 0
        else: account['contact_attempts'] = 0

        # Ensure 'article3_stage' for Article 3 Process Agent
        if 'current_stage' in account:
            account['article3_stage'] = account['current_stage']
        else:
            account['article3_stage'] = 'STAGE_1' # Default for agent

    dormancy_results_list = []
    start_time = datetime.now()
    for account_data in accounts_data_list:
        results = dormancy_orchestrator.process_account(account_data)
        dormancy_results_list.append({
            "account_id": account_data.get('account_id', 'N/A'),
            "analysis_results": results
        })
    processing_time = (datetime.now() - start_time).total_seconds()

    # Aggregate summary
    total_dormant_flagged = 0
    high_value_dormant_count = 0
    for res in dormancy_results_list:
        # Check specific agent results for dormancy status and high value
        if res.get('analysis_results', {}).get('demand_deposit_inactivity', {}).get('status') == ActivityStatus.DORMANT.value:
            total_dormant_flagged += 1

        # Check high value agent's customer tier
        customer_tier = res.get('analysis_results', {}).get('high_value_dormant', {}).get('customer_tier')
        if customer_tier in [CustomerTier.HIGH_VALUE.value, CustomerTier.PREMIUM.value, CustomerTier.VIP.value, CustomerTier.PRIVATE_BANKING.value]:
            high_value_dormant_count += 1

    summary_message = f"Dormancy analysis completed for {len(accounts_data_list)} accounts. Found {total_dormant_flagged} accounts flagged as dormant, including {high_value_dormant_count} high-value dormant accounts."

    # Store dormancy results for subsequent compliance verification
    uploaded_files_storage[f"dormancy_results_{request.file_id}"] = dormancy_results_list

    return DormancyAnalysisResponse(
        success=True,
        summary=summary_message,
        results={"detailed_results": dormancy_results_list},
        processing_time_seconds=processing_time
    )

# --- Compliance Verification Endpoint ---
@app.post("/compliance/verify", response_model=ComplianceVerificationResponse)
async def verify_compliance_endpoint(request: ComplianceVerificationRequest, user_id: str = Form("default_user")):
    """
    Performs compliance verification on account data, optionally using dormancy analysis results.
    Requires dormancy analysis to be completed first if its results are to be used.
    """
    if not compliance_orchestrator:
        raise HTTPException(status_code=500, detail="Compliance verification agent not initialized.")

    # Retrieve dormancy results (optional, but typically follows dormancy analysis)
    dormancy_results_key = f"dormancy_results_{request.file_id}"
    dormancy_analysis_output = uploaded_files_storage.get(dormancy_results_key)

    if not dormancy_analysis_output:
        # If no dormancy results, we can still proceed with original data, but advise user
        logger.warning(f"No dormancy analysis results found for file_id {request.file_id}. Compliance verification might be less comprehensive.")
        # Attempt to get raw processed data instead
        if request.file_id not in uploaded_files_storage:
             raise HTTPException(status_code=404, detail="No processed data or dormancy analysis results found. Please upload and process/analyze data first.")

        original_data_df = uploaded_files_storage[request.file_id]['data']
        # If no dormancy output, create a list of mock entries for iteration
        dormancy_analysis_output = [{"account_id": row.get('account_id', f"ACC_{i}"), "analysis_results": {}} for i, row in original_data_df.head(100).to_dict(orient='records')]


    compliance_verification_results = []
    start_time = datetime.now()

    # Iterate through each account (from dormancy output or raw data)
    for entry in dormancy_analysis_output:
        account_id = entry.get('account_id', 'N/A')
        dormancy_single_account_results = entry.get('analysis_results', {})

        # Retrieve the original full account data for this specific account_id from the initial upload
        original_data_df = uploaded_files_storage[request.file_id]['data']

        # Assuming 'account_id' is a column in the original data, find the matching row.
        # This mapping should ideally come from the mapping step. For simplicity, we search directly.
        # Example: search for 'account_id' column or map from common names.

        # Try to find a 'account_id' column dynamically
        account_id_col_name = None
        for col in original_data_df.columns:
            if 'account_id' in col.lower() or 'account_no' in col.lower():
                account_id_col_name = col
                break

        account_row_data = {}
        if account_id_col_name and account_id != 'N/A' and account_id in original_data_df[account_id_col_name].values:
            account_row_data = original_data_df[original_data_df[account_id_col_name] == account_id].iloc[0].to_dict()
        else:
            # Fallback if specific account row not found, use a generic structure or raise error
            logger.warning(f"Original data for account_id {account_id} not found; using a generic structure for compliance verification.")
            account_row_data = {'account_id': account_id, 'balance_current': 0.0, 'dormancy_status': 'unknown'}

        # --- Crucial: Ensure `account_row_data` has all fields expected by Compliance Agents ---
        # The Compliance Agents (in compliance_verification_agent.py) expect a wide range of fields.
        # We need to populate default/mock values for any missing required fields to prevent errors.
        account_row_data.setdefault('customer_id', account_row_data.get('account_id', 'CUST_MOCK'))
        account_row_data.setdefault('account_type', 'savings')
        account_row_data.setdefault('balance_current', float(account_row_data.get('balance_current', 0.0)))
        account_row_data.setdefault('dormancy_status', account_row_data.get('dormancy_status', 'dormant'))
        account_row_data.setdefault('last_transaction_date', (datetime.now() - timedelta(days=1500)).isoformat())
        account_row_data.setdefault('dormancy_trigger_date', (datetime.now() - timedelta(days=1200)).isoformat())
        account_row_data.setdefault('contact_attempts_made', 3)
        account_row_data.setdefault('customer_type', 'individual')
        account_row_data.setdefault('currency', 'AED')
        account_row_data.setdefault('created_date', (datetime.now() - timedelta(days=2000)).isoformat())
        account_row_data.setdefault('updated_date', (datetime.now() - timedelta(days=50)).isoformat())
        account_row_data.setdefault('updated_by', 'SYSTEM_API')
        account_row_data.setdefault('kyc_status', 'complete')
        account_row_data.setdefault('kyc_expiry_date', (datetime.now() + timedelta(days=180)).isoformat())
        account_row_data.setdefault('risk_rating', 'medium')
        account_row_data.setdefault('address_known', True)
        account_row_data.setdefault('phone_primary', '+971501234567')
        account_row_data.setdefault('email_primary', 'test@example.com')
        account_row_data.setdefault('has_outstanding_facilities', False)
        account_row_data.setdefault('transfer_eligibility_date', (datetime.now() - timedelta(days=30)).isoformat())
        account_row_data.setdefault('transferred_to_cb_date', None)
        account_row_data.setdefault('contact_attempts_made', 3)
        account_row_data.setdefault('last_contact_attempt_date', (datetime.now() - timedelta(days=100)).isoformat())
        account_row_data.setdefault('current_stage', 'transfer_ready')
        account_row_data.setdefault('statement_frequency', 'monthly')
        account_row_data.setdefault('last_statement_date', (datetime.now() - timedelta(days=40)).isoformat())
        account_row_data.setdefault('is_joint_account', False)
        account_row_data.setdefault('interest_accrued', 0.0)
        account_row_data.setdefault('interest_rate', 0.01)
        account_row_data.setdefault('balance_available', account_row_data['balance_current'])


        compliance_results_for_account = compliance_orchestrator.process_account(
            account_data=account_row_data,
            dormancy_results=dormancy_single_account_results # Pass single account's dormancy analysis result
        )
        overall_compliance_summary = compliance_orchestrator.generate_compliance_summary(compliance_results_for_account)

        compliance_verification_results.append({
            "account_id": account_id,
            "detailed_compliance_results": compliance_results_for_account,
            "overall_summary": overall_compliance_summary
        })
    processing_time = (datetime.now() - start_time).total_seconds()

    total_violations = sum(item['overall_summary']['total_violations'] for item in compliance_verification_results)
    critical_violations = sum(item['overall_summary']['critical_violations'] for item in compliance_verification_results)
    overall_compliance_status_counts = {}
    for item in compliance_verification_results:
        status = item['overall_summary']['overall_status']
        overall_compliance_status_counts[status] = overall_compliance_status_counts.get(status, 0) + 1


    summary_message = (
        f"Compliance verification completed for {len(compliance_verification_results)} accounts. "
        f"Total violations found: {total_violations}. Critical violations: {critical_violations}. "
        f"Overall statuses: {overall_compliance_status_counts}"
    )

    # Store compliance results
    uploaded_files_storage[f"compliance_results_{request.file_id}"] = compliance_verification_results


    return ComplianceVerificationResponse(
        success=True,
        summary=summary_message,
        results={"detailed_compliance_reports": compliance_verification_results},
        processing_time_seconds=processing_time
    )

# --- Main entry point for local development ---
if __name__ == "__main__":
    import uvicorn
    # To run the FastAPI application:
    # 1. Ensure all Python dependencies from requirements.txt are installed.
    # 2. Place this file (banking_compliance_unified_api.py) in your project root.
    # 3. Ensure the 'agents' directory and its contents are correctly structured relative to this file.
    # 4. Make sure 'mistral-7b-instruct-v0.1.Q4_K_M.gguf' model file is accessible if LLAMA_CPP_AVAILABLE is True.
    #    You might need to set the LLAMA_MODEL_PATH environment variable or modify the startup_event to point to it.
    # 5. Run the command in your terminal from the project root:
    #    uvicorn banking_compliance_unified_api:app --host 0.0.0.0 --port 8080 --reload
    # This will start the API server. You can then access the documentation at http://localhost:8080/docs
    # and interact with the endpoints.

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")