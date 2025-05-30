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
class ErrorState:
    """State for error handling workflow"""

    session_id: str
    user_id: str
    error_id: str
    timestamp: datetime

    # Error information
    errors: List[Dict] = None
    failed_node: Optional[str] = None
    workflow_context: Optional[Any] = None

    # Recovery results
    recovery_action: Optional[str] = None
    recovery_success: bool = False

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ErrorHandlerAgent:
    """Centralized error handling and recovery agent"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

    @traceable(name="handle_workflow_error")
    async def handle_workflow_error(self, state: ErrorState) -> ErrorState:
        """Handle workflow errors and determine recovery actions"""

        try:
            # Analyze error severity and type
            error_analysis = self._analyze_errors(state.errors)

            # Determine recovery strategy
            state.recovery_action = self._determine_recovery_action(error_analysis, state.failed_node)

            # Log error for future pattern analysis
            await self._log_error_pattern(state)

            # Call MCP tool for error handling
            mcp_result = await self.mcp_client.call_tool("handle_error", {
                "errors": state.errors,
                "failed_node": state.failed_node,
                "recovery_action": state.recovery_action,
                "user_id": state.user_id
            })

            state.recovery_success = True

        except Exception as e:
            logger.error(f"Error handling failed: {str(e)}")
            state.recovery_action = "terminate"
            state.recovery_success = False

        return state

    def _analyze_errors(self, errors: List[Dict]) -> Dict:
        """Analyze errors to determine severity and patterns"""

        analysis = {
            "total_errors": len(errors),
            "critical_errors": 0,
            "recoverable_errors": 0,
            "error_types": {},
            "severity": "low"
        }

        for error in errors:
            error_type = error.get("error_type", "unknown")
            analysis["error_types"][error_type] = analysis["error_types"].get(error_type, 0) + 1

            if error.get("critical", False):
                analysis["critical_errors"] += 1
            else:
                analysis["recoverable_errors"] += 1

        # Determine overall severity
        if analysis["critical_errors"] > 0:
            analysis["severity"] = "critical"
        elif analysis["total_errors"] > 5:
            analysis["severity"] = "high"
        elif analysis["total_errors"] > 2:
            analysis["severity"] = "medium"

        return analysis

    def _determine_recovery_action(self, error_analysis: Dict, failed_node: str) -> str:
        """Determine appropriate recovery action"""

        if error_analysis["severity"] == "critical":
            return "escalate"
        elif error_analysis["recoverable_errors"] > 0 and failed_node:
            return "retry"
        elif error_analysis["total_errors"] < 3:
            return "retry"
        else:
            return "terminate"

    async def _log_error_pattern(self, state: ErrorState):
        """Log error patterns for future analysis"""

        try:
            error_pattern = {
                "type": "error_pattern",
                "user_id": state.user_id,
                "session_id": state.session_id,
                "failed_node": state.failed_node,
                "errors": state.errors,
                "recovery_action": state.recovery_action,
                "timestamp": datetime.now().isoformat()
            }

            await self.memory_agent.store_memory(
                bucket="knowledge",
                data=error_pattern,
                content_type="error_analysis",
                tags=["errors", "patterns", "recovery"]
            )

        except Exception as e:
            logger.warning(f"Failed to log error pattern: {str(e)}")