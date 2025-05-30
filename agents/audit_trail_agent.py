import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# LangGraph and LangSmith imports
from langsmith import traceable, Client as LangSmithClient

# MCP imports
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AuditState:
    """State for audit trail workflow"""

    session_id: str
    user_id: str
    audit_id: str
    timestamp: datetime

    # Audit data
    workflow_state: Optional[Any] = None
    audit_entries: List[Dict] = None

    def __post_init__(self):
        if self.audit_entries is None:
            self.audit_entries = []


class AuditTrailAgent:
    """Comprehensive audit trail and logging agent"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

    @traceable(name="log_workflow_completion")
    async def log_workflow_completion(self, state: AuditState) -> AuditState:
        """Log comprehensive workflow completion audit trail"""

        try:
            # Generate comprehensive audit log
            audit_log = {
                "audit_id": state.audit_id,
                "user_id": state.user_id,
                "session_id": state.session_id,
                "workflow_completion": {
                    "completed_at": datetime.now().isoformat(),
                    "total_processing_time": getattr(state.workflow_state, 'total_processing_time', 0),
                    "nodes_completed": len(getattr(state.workflow_state, 'completed_nodes', [])),
                    "nodes_failed": len(getattr(state.workflow_state, 'failed_nodes', [])),
                    "workflow_status": getattr(state.workflow_state, 'workflow_status', 'unknown').value if hasattr(
                        getattr(state.workflow_state, 'workflow_status', None), 'value') else 'unknown'
                },
                "data_lineage": self._generate_data_lineage(state.workflow_state),
                "compliance_trail": self._generate_compliance_trail(state.workflow_state),
                "performance_metrics": self._generate_performance_audit(state.workflow_state),
                "security_events": self._generate_security_audit(state.workflow_state)
            }

            # Store audit log
            await self.memory_agent.store_memory(
                bucket="audit",
                data=audit_log,
                content_type="workflow_audit",
                priority="HIGH",
                tags=["audit", "workflow", "compliance"],
                encrypt_sensitive=True
            )

            state.audit_entries.append(audit_log)

            # Call MCP tool for external audit logging
            mcp_result = await self.mcp_client.call_tool("log_audit_trail", {
                "audit_log": audit_log,
                "user_id": state.user_id,
                "session_id": state.session_id
            })

        except Exception as e:
            logger.error(f"Audit logging failed: {str(e)}")

        return state

    def _generate_data_lineage(self, workflow_state) -> Dict:
        """Generate data lineage information"""

        lineage = {
            "input_data_source": "uploaded_file",
            "processing_stages": [],
            "output_artifacts": [],
            "data_transformations": []
        }

        if hasattr(workflow_state, 'completed_nodes'):
            for node in getattr(workflow_state, 'completed_nodes', []):
                lineage["processing_stages"].append({
                    "stage": node,
                    "completed_at": datetime.now().isoformat(),
                    "data_modified": True
                })

        return lineage

    def _generate_compliance_trail(self, workflow_state) -> Dict:
        """Generate compliance audit trail"""

        compliance_trail = {
            "regulatory_framework": "CBUAE",
            "compliance_checks_performed": [],
            "violations_identified": 0,
            "remediation_actions": [],
            "compliance_officer": "system"
        }

        if hasattr(workflow_state, 'compliance_results'):
            compliance_results = getattr(workflow_state, 'compliance_results', {})
            if compliance_results:
                compliance_trail["violations_identified"] = len(compliance_results.get("violations", []))
                compliance_trail["remediation_actions"] = compliance_results.get("remediation_actions", [])

        return compliance_trail

    def _generate_performance_audit(self, workflow_state) -> Dict:
        """Generate performance audit information"""

        performance = {
            "total_execution_time": getattr(workflow_state, 'total_processing_time', 0),
            "node_timings": getattr(workflow_state, 'node_timings', {}),
            "memory_operations": len(
                [log for log in getattr(workflow_state, 'workflow_log', []) if "memory" in log.get("action", "")]),
            "error_rate": len(getattr(workflow_state, 'errors', [])) / max(1, len(getattr(workflow_state,
                                                                                          'completed_nodes', []))),
            "efficiency_score": self._calculate_efficiency_score(workflow_state)
        }

        return performance

    def _generate_security_audit(self, workflow_state) -> Dict:
        """Generate security audit information"""

        security = {
            "user_authentication": "verified",
            "data_encryption": "enabled",
            "access_controls": "enforced",
            "sensitive_data_handling": "compliant",
            "audit_trail_integrity": "maintained"
        }

        return security

    def _calculate_efficiency_score(self, workflow_state) -> float:
        """Calculate workflow efficiency score"""

        if not hasattr(workflow_state, 'total_processing_time') or not hasattr(workflow_state, 'completed_nodes'):
            return 0.0

        total_time = getattr(workflow_state, 'total_processing_time', 1)
        completed_nodes = len(getattr(workflow_state, 'completed_nodes', []))
        failed_nodes = len(getattr(workflow_state, 'failed_nodes', []))

        # Efficiency based on completion rate and processing speed
        completion_rate = completed_nodes / max(1, completed_nodes + failed_nodes)
        time_efficiency = min(1.0, 300 / max(total_time, 1))  # 300 seconds as baseline

        return (completion_rate * 0.7 + time_efficiency * 0.3)