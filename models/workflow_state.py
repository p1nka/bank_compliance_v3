from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class WorkflowState(BaseModel):
    """Represents the state of the banking compliance workflow."""
    initial_data: Optional[Dict[str, Any]] = None
    processed_data: Optional[Dict[str, Any]] = None
    dormancy_analysis_results: Optional[Dict[str, Any]] = None
    compliance_status: Optional[Dict[str, Any]] = None
    risk_assessment_results: Optional[Dict[str, Any]] = None
    final_report: Optional[Dict[str, Any]] = None
    notifications: List[str] = []
    error: Optional[str] = None
    current_agent: Optional[str] = None
    session_memory: Dict[str, Any] = {}
    knowledge_memory: Dict[str, Any] = {}
    # Add any other state variables your agents might need to pass around
    # For RAG, you might add:
    retrieved_documents: Optional[List[Dict[str, Any]]] = None
    query_for_rag: Optional[str] = None

class BankingDataInput(BaseModel):
    """Input model for banking data."""
    file_content: str # Base64 encoded file content or raw text
    file_name: str
    file_type: str
    user_query: Optional[str] = None # For RAG-based queries
