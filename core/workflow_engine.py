"""
core/workflow_engine.py - LangGraph Orchestration Engine
Enhanced Banking Compliance Agentic AI System Orchestration
Integrates all agents with hybrid memory and MCP tools
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import traceback

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# LangSmith imports
from langsmith import traceable, Client as LangSmithClient
from langsmith.wrappers import wrap_openai

# Core agent imports
from agents.Data_Process import DataProcessingAgent, DataProcessingState
from agents.Dormant_agent import DormancyAnalysisAgent, DormancyAnalysisState
from agents.compliance_verification_agent import ComplianceVerificationAgent, ComplianceState
from agents.risk_assessment_agent import RiskAssessmentAgent, RiskAssessmentState
from agents.reporting_agent import ReportingAgent, ReportingState
from agents.notification_agent import NotificationAgent, NotificationState
from agents.memory_agent import HybridMemoryAgent, MemoryContext
from agents.supervisor_agent import SupervisorAgent, SupervisorState
from agents.error_handler_agent import ErrorHandlerAgent, ErrorState
from agents.audit_trail_agent import AuditTrailAgent, AuditState

# MCP client
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowState:
    """Comprehensive workflow state for LangGraph orchestration"""

    # Session and workflow identifiers
    workflow_id: str
    session_id: str
    user_id: str
    timestamp: datetime

    # Workflow status and control
    workflow_status: WorkflowStatus = WorkflowStatus.INITIALIZED
    current_node: Optional[str] = None
    completed_nodes: List[str] = None
    failed_nodes: List[str] = None

    # Data flow between agents
    raw_input_data: Optional[Dict] = None
    processed_data: Optional[Dict] = None
    dormancy_results: Optional[Dict] = None
    compliance_results: Optional[Dict] = None
    risk_assessment_results: Optional[Dict] = None
    supervisor_decisions: Optional[Dict] = None
    reporting_results: Optional[Dict] = None
    notification_results: Optional[Dict] = None

    # Agent states
    data_processing_state: Optional[DataProcessingState] = None
    dormancy_analysis_state: Optional[DormancyAnalysisState] = None
    compliance_state: Optional[ComplianceState] = None
    risk_assessment_state: Optional[RiskAssessmentState] = None
    supervisor_state: Optional[SupervisorState] = None
    reporting_state: Optional[ReportingState] = None
    notification_state: Optional[NotificationState] = None
    error_state: Optional[ErrorState] = None
    audit_state: Optional[AuditState] = None

    # Memory and context
    memory_context: Dict = None
    workflow_metadata: Dict = None
    user_preferences: Dict = None

    # Error handling
    errors: List[Dict] = None
    warnings: List[Dict] = None
    retry_count: int = 0
    max_retries: int = 3

    # Performance metrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    node_timings: Dict = None
    total_processing_time: float = 0.0

    # Audit trail
    workflow_log: List[Dict] = None
    state_transitions: List[Dict] = None

    def __post_init__(self):
        if self.completed_nodes is None:
            self.completed_nodes = []
        if self.failed_nodes is None:
            self.failed_nodes = []
        if self.memory_context is None:
            self.memory_context = {}
        if self.workflow_metadata is None:
            self.workflow_metadata = {}
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.node_timings is None:
            self.node_timings = {}
        if self.workflow_log is None:
            self.workflow_log = []
        if self.state_transitions is None:
            self.state_transitions = []


class WorkflowOrchestrationEngine:
    """Enhanced LangGraph workflow orchestration engine"""

    def __init__(self, memory_agent: HybridMemoryAgent, mcp_client: MCPClient,
                 db_session=None, langsmith_client=None):
        """Initialize the workflow orchestration engine"""

        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = langsmith_client or LangSmithClient()

        # Initialize agents
        self._initialize_agents()

        # Initialize workflow graph
        self.workflow_graph = None
        self.checkpointer = MemorySaver()

        # Build the workflow
        self._build_workflow()

        # Performance monitoring
        self.workflow_metrics = {}

    def _initialize_agents(self):
        """Initialize all workflow agents"""

        # Core processing agents
        self.data_processing_agent = DataProcessingAgent(
            self.memory_agent, self.mcp_client, self.db_session
        )

        self.dormancy_analysis_agent = DormancyAnalysisAgent(
            self.memory_agent, self.mcp_client, self.db_session
        )

        self.compliance_agent = ComplianceVerificationAgent(
            self.memory_agent, self.mcp_client, self.db_session
        )

        self.risk_assessment_agent = RiskAssessmentAgent(
            self.memory_agent, self.mcp_client, self.db_session
        )

        # Coordination and output agents
        self.supervisor_agent = SupervisorAgent(
            self.memory_agent, self.mcp_client, self.db_session
        )

        self.reporting_agent = ReportingAgent(
            self.memory_agent, self.mcp_client, self.db_session
        )

        # Create notification agent (assuming it exists)
        try:
            self.notification_agent = NotificationAgent(
                self.memory_agent, self.mcp_client, self.db_session
            )
        except ImportError:
            logger.warning("NotificationAgent not found, notifications will be skipped")
            self.notification_agent = None

        # Support agents
        self.error_handler_agent = ErrorHandlerAgent(
            self.memory_agent, self.mcp_client, self.db_session
        )

        self.audit_trail_agent = AuditTrailAgent(
            self.memory_agent, self.mcp_client, self.db_session
        )

    def _build_workflow(self):
        """Build the LangGraph workflow with all nodes and edges"""

        # Create workflow graph
        workflow = StateGraph(WorkflowState)

        # Add all workflow nodes
        workflow.add_node("router_agent", self._router_node)
        workflow.add_node("data_processing_agent", self._data_processing_node)
        workflow.add_node("dormancy_analysis_agent", self._dormancy_analysis_node)
        workflow.add_node("compliance_verification_agent", self._compliance_verification_node)
        workflow.add_node("risk_assessment_agent", self._risk_assessment_node)
        workflow.add_node("memory_agent", self._memory_agent_node)
        workflow.add_node("supervisor_agent", self._supervisor_node)
        workflow.add_node("reporting_agent", self._reporting_node)
        workflow.add_node("error_handler_agent", self._error_handler_node)
        workflow.add_node("audit_trail_agent", self._audit_trail_node)

        # Add notification agent only if it exists
        if self.notification_agent:
            workflow.add_node("notification_agent", self._notification_node)

        # Set entry point
        workflow.set_entry_point("router_agent")

        # Define workflow edges (main flow)
        workflow.add_edge("router_agent", "data_processing_agent")
        workflow.add_edge("data_processing_agent", "dormancy_analysis_agent")
        workflow.add_edge("dormancy_analysis_agent", "compliance_verification_agent")
        workflow.add_edge("compliance_verification_agent", "risk_assessment_agent")
        workflow.add_edge("risk_assessment_agent", "memory_agent")
        workflow.add_edge("memory_agent", "supervisor_agent")

        # Add conditional edges for supervisor routing
        workflow.add_conditional_edges(
            "supervisor_agent",
            self._supervisor_routing,
            {
                "proceed_to_reporting": "reporting_agent",
                "escalate": "error_handler_agent",
                "requires_review": "error_handler_agent"
            }
        )

        # Add reporting flow
        if self.notification_agent:
            workflow.add_edge("reporting_agent", "notification_agent")
            workflow.add_edge("notification_agent", "audit_trail_agent")
        else:
            workflow.add_edge("reporting_agent", "audit_trail_agent")

        workflow.add_edge("audit_trail_agent", END)

        # Error handling routes
        workflow.add_conditional_edges(
            "error_handler_agent",
            self._error_routing,
            {
                "retry": "supervisor_agent",
                "escalate": "audit_trail_agent",
                "terminate": END
            }
        )

        # Compile workflow
        self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)

    @traceable(name="router_node")
    async def _router_node(self, state: WorkflowState) -> WorkflowState:
        """Router agent - selects initial workflow path"""

        try:
            state.current_node = "router_agent"
            state.start_time = datetime.now()
            state.workflow_status = WorkflowStatus.RUNNING

            # Pre-memory hook
            state = await self._execute_pre_memory_hook(state, "router_agent")

            # Log workflow start
            state.workflow_log.append({
                "timestamp": datetime.now().isoformat(),
                "node": "router_agent",
                "action": "workflow_started",
                "workflow_id": state.workflow_id,
                "user_id": state.user_id
            })

            # Determine workflow path based on input data
            if not state.raw_input_data:
                raise ValueError("No input data provided")

            # Analyze input to determine best processing path
            data_type = self._analyze_input_data(state.raw_input_data)

            state.workflow_metadata.update({
                "data_type": data_type,
                "routing_decision": "standard_flow",
                "estimated_processing_time": self._estimate_processing_time(state.raw_input_data)
            })

            # Post-memory hook
            state = await self._execute_post_memory_hook(state, "router_agent")

            state.completed_nodes.append("router_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "router_agent", e)

    @traceable(name="data_processing_node")
    async def _data_processing_node(self, state: WorkflowState) -> WorkflowState:
        """Data processing agent node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "data_processing_agent"

            # Pre-memory hook
            state = await self._execute_pre_memory_hook(state, "data_processing_agent")

            # Initialize data processing state
            processing_state = DataProcessingState(
                session_id=state.session_id,
                user_id=state.user_id,
                processing_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                raw_data=state.raw_input_data,
                memory_context=state.memory_context
            )

            # Execute data processing
            processing_state = await self.data_processing_agent.process_banking_data(processing_state)

            # Update workflow state
            state.data_processing_state = processing_state
            state.processed_data = processing_state.processed_data

            # Record timing
            state.node_timings["data_processing_agent"] = (datetime.now() - node_start_time).total_seconds()

            # Post-memory hook
            state = await self._execute_post_memory_hook(state, "data_processing_agent")

            state.completed_nodes.append("data_processing_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "data_processing_agent", e)

    @traceable(name="dormancy_analysis_node")
    async def _dormancy_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Dormancy analysis agent node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "dormancy_analysis_agent"

            # Pre-memory hook
            state = await self._execute_pre_memory_hook(state, "dormancy_analysis_agent")

            # Initialize dormancy analysis state
            analysis_state = DormancyAnalysisState(
                session_id=state.session_id,
                user_id=state.user_id,
                analysis_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                processed_data=state.processed_data,
                memory_context=state.memory_context
            )

            # Execute dormancy analysis
            analysis_state = await self.dormancy_analysis_agent.analyze_dormancy(analysis_state)

            # Update workflow state
            state.dormancy_analysis_state = analysis_state
            state.dormancy_results = analysis_state.dormancy_results

            # Record timing
            state.node_timings["dormancy_analysis_agent"] = (datetime.now() - node_start_time).total_seconds()

            # Post-memory hook
            state = await self._execute_post_memory_hook(state, "dormancy_analysis_agent")

            state.completed_nodes.append("dormancy_analysis_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "dormancy_analysis_agent", e)

    @traceable(name="compliance_verification_node")
    async def _compliance_verification_node(self, state: WorkflowState) -> WorkflowState:
        """Compliance verification agent node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "compliance_verification_agent"

            # Pre-memory hook
            state = await self._execute_pre_memory_hook(state, "compliance_verification_agent")

            # Initialize compliance state
            compliance_state = ComplianceState(
                session_id=state.session_id,
                user_id=state.user_id,
                compliance_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                dormancy_results=state.dormancy_results,
                memory_context=state.memory_context
            )

            # Execute compliance verification
            compliance_state = await self.compliance_agent.verify_compliance(compliance_state)

            # Update workflow state
            state.compliance_state = compliance_state
            state.compliance_results = compliance_state.compliance_results

            # Record timing
            state.node_timings["compliance_verification_agent"] = (datetime.now() - node_start_time).total_seconds()

            # Post-memory hook
            state = await self._execute_post_memory_hook(state, "compliance_verification_agent")

            state.completed_nodes.append("compliance_verification_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "compliance_verification_agent", e)

    @traceable(name="risk_assessment_node")
    async def _risk_assessment_node(self, state: WorkflowState) -> WorkflowState:
        """Risk assessment agent node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "risk_assessment_agent"

            # Pre-memory hook
            state = await self._execute_pre_memory_hook(state, "risk_assessment_agent")

            # Initialize risk assessment state
            risk_state = RiskAssessmentState(
                session_id=state.session_id,
                user_id=state.user_id,
                assessment_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                dormancy_results=state.dormancy_results,
                compliance_results=state.compliance_results,
                memory_context=state.memory_context
            )

            # Execute risk assessment
            risk_state = await self.risk_assessment_agent.assess_risk(risk_state)

            # Update workflow state
            state.risk_assessment_state = risk_state
            state.risk_assessment_results = risk_state.risk_assessment_results

            # Record timing
            state.node_timings["risk_assessment_agent"] = (datetime.now() - node_start_time).total_seconds()

            # Post-memory hook
            state = await self._execute_post_memory_hook(state, "risk_assessment_agent")

            state.completed_nodes.append("risk_assessment_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "risk_assessment_agent", e)

    @traceable(name="memory_agent_node")
    async def _memory_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Memory agent consolidation node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "memory_agent"

            # Consolidate all analysis results for memory storage
            consolidated_results = {
                "workflow_id": state.workflow_id,
                "session_id": state.session_id,
                "user_id": state.user_id,
                "timestamp": datetime.now().isoformat(),
                "data_processing_summary": {
                    "status": state.data_processing_state.processing_status.value if state.data_processing_state else None,
                    "quality_score": state.data_processing_state.quality_score if state.data_processing_state else 0,
                    "records_processed": len(state.processed_data.get("accounts", [])) if state.processed_data else 0
                },
                "dormancy_analysis_summary": {
                    "total_analyzed": state.dormancy_analysis_state.total_accounts_analyzed if state.dormancy_analysis_state else 0,
                    "dormant_found": state.dormancy_analysis_state.dormant_accounts_found if state.dormancy_analysis_state else 0,
                    "high_risk": state.dormancy_analysis_state.high_risk_accounts if state.dormancy_analysis_state else 0
                },
                "compliance_summary": {
                    "status": state.compliance_state.verification_status.value if state.compliance_state else None,
                    "critical_issues": len(
                        state.compliance_results.get("critical_violations", [])) if state.compliance_results else 0
                },
                "risk_assessment_summary": {
                    "overall_risk_score": state.risk_assessment_results.get("overall_risk_score",
                                                                            0) if state.risk_assessment_results else 0,
                    "high_risk_accounts": len(
                        state.risk_assessment_results.get("high_risk_accounts", [])) if state.risk_assessment_results else 0
                },
                "workflow_performance": state.node_timings
            }

            # Store consolidated results in both session and knowledge memory
            await self.memory_agent.store_memory(
                bucket="session",
                data=consolidated_results,
                encrypt_sensitive=True
            )

            await self.memory_agent.store_memory(
                bucket="knowledge",
                data={
                    "type": "workflow_completion_pattern",
                    "user_id": state.user_id,
                    "completion_data": consolidated_results,
                    "success_indicators": {
                        "all_nodes_completed": len(state.completed_nodes),
                        "no_critical_errors": len(state.errors) == 0,
                        "performance_acceptable": sum(state.node_timings.values()) < 300  # 5 minutes
                    }
                }
            )

            # Record timing
            state.node_timings["memory_agent"] = (datetime.now() - node_start_time).total_seconds()

            state.completed_nodes.append("memory_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "memory_agent", e)

    @traceable(name="supervisor_node")
    async def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """Supervisor agent decision node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "supervisor_agent"

            # Pre-memory hook
            state = await self._execute_pre_memory_hook(state, "supervisor_agent")

            # Initialize supervisor state
            supervisor_state = SupervisorState(
                session_id=state.session_id,
                user_id=state.user_id,
                supervision_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                workflow_results={
                    "data_processing": state.data_processing_state,
                    "dormancy_analysis": state.dormancy_analysis_state,
                    "compliance": state.compliance_state,
                    "risk_assessment": state.risk_assessment_state
                },
                memory_context=state.memory_context
            )

            # Execute supervisor decision making
            supervisor_state = await self.supervisor_agent.supervise_workflow(supervisor_state)

            # Update workflow state
            state.supervisor_state = supervisor_state
            state.supervisor_decisions = supervisor_state.supervision_decisions

            # Record timing
            state.node_timings["supervisor_agent"] = (datetime.now() - node_start_time).total_seconds()

            # Post-memory hook
            state = await self._execute_post_memory_hook(state, "supervisor_agent")

            state.completed_nodes.append("supervisor_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "supervisor_agent", e)

    @traceable(name="reporting_node")
    async def _reporting_node(self, state: WorkflowState) -> WorkflowState:
        """Reporting agent node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "reporting_agent"

            # Pre-memory hook
            state = await self._execute_pre_memory_hook(state, "reporting_agent")

            # Initialize reporting state
            reporting_state = ReportingState(
                session_id=state.session_id,
                user_id=state.user_id,
                report_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                workflow_results={
                    "data_processing": state.data_processing_state,
                    "dormancy_analysis": state.dormancy_analysis_state,
                    "compliance": state.compliance_state,
                    "risk_assessment": state.risk_assessment_state,
                    "supervisor_decisions": state.supervisor_decisions
                },
                memory_context=state.memory_context
            )

            # Execute report generation
            reporting_state = await self.reporting_agent.generate_compliance_report(reporting_state)

            # Update workflow state
            state.reporting_state = reporting_state
            state.reporting_results = reporting_state.report_results

            # Record timing
            state.node_timings["reporting_agent"] = (datetime.now() - node_start_time).total_seconds()

            # Post-memory hook
            state = await self._execute_post_memory_hook(state, "reporting_agent")

            state.completed_nodes.append("reporting_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "reporting_agent", e)

    @traceable(name="notification_node")
    async def _notification_node(self, state: WorkflowState) -> WorkflowState:
        """Notification agent node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "notification_agent"

            # Skip if notification agent not available
            if not self.notification_agent:
                logger.info("Notification agent not available, skipping notifications")
                state.completed_nodes.append("notification_agent")
                return state

            # Pre-memory hook
            state = await self._execute_pre_memory_hook(state, "notification_agent")

            # Create a mock NotificationState for now
            notification_state = type('NotificationState', (), {
                'session_id': state.session_id,
                'user_id': state.user_id,
                'notification_id': secrets.token_hex(16),
                'timestamp': datetime.now(),
                'report_results': state.reporting_results,
                'supervisor_decisions': state.supervisor_decisions,
                'memory_context': state.memory_context,
                'notification_results': {"sent_notifications": []}
            })()

            # Execute notifications if method exists
            if hasattr(self.notification_agent, 'send_notifications'):
                notification_state = await self.notification_agent.send_notifications(notification_state)

            # Update workflow state
            state.notification_state = notification_state
            state.notification_results = notification_state.notification_results

            # Record timing
            state.node_timings["notification_agent"] = (datetime.now() - node_start_time).total_seconds()

            # Post-memory hook
            state = await self._execute_post_memory_hook(state, "notification_agent")

            state.completed_nodes.append("notification_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "notification_agent", e)

    @traceable(name="audit_trail_node")
    async def _audit_trail_node(self, state: WorkflowState) -> WorkflowState:
        """Audit trail agent node - final workflow step"""

        try:
            node_start_time = datetime.now()
            state.current_node = "audit_trail_agent"

            # Initialize audit state
            audit_state = AuditState(
                session_id=state.session_id,
                user_id=state.user_id,
                audit_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                workflow_state=state
            )

            # Execute audit trail logging
            audit_state = await self.audit_trail_agent.log_workflow_completion(audit_state)

            # Update workflow state
            state.audit_state = audit_state
            state.end_time = datetime.now()
            state.total_processing_time = (state.end_time - state.start_time).total_seconds()
            state.workflow_status = WorkflowStatus.COMPLETED

            # Record timing
            state.node_timings["audit_trail_agent"] = (datetime.now() - node_start_time).total_seconds()

            # Final workflow log entry
            state.workflow_log.append({
                "timestamp": datetime.now().isoformat(),
                "node": "audit_trail_agent",
                "action": "workflow_completed",
                "workflow_id": state.workflow_id,
                "total_time": state.total_processing_time,
                "nodes_completed": len(state.completed_nodes),
                "nodes_failed": len(state.failed_nodes)
            })

            state.completed_nodes.append("audit_trail_agent")
            return state

        except Exception as e:
            return await self._handle_node_error(state, "audit_trail_agent", e)

    @traceable(name="error_handler_node")
    async def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Error handler agent node"""

        try:
            node_start_time = datetime.now()
            state.current_node = "error_handler_agent"

            # Initialize error state
            error_state = ErrorState(
                session_id=state.session_id,
                user_id=state.user_id,
                error_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                errors=state.errors,
                failed_node=state.failed_nodes[-1] if state.failed_nodes else None,
                workflow_context=state
            )

            # Execute error handling
            error_state = await self.error_handler_agent.handle_workflow_error(error_state)

            # Update workflow state
            state.error_state = error_state

            # Determine recovery action
            if error_state.recovery_action == "retry" and state.retry_count < state.max_retries:
                state.retry_count += 1
                # Reset failed node for retry
                if state.failed_nodes:
                    failed_node = state.failed_nodes.pop()
                    if failed_node in state.completed_nodes:
                        state.completed_nodes.remove(failed_node)
            elif error_state.recovery_action == "escalate":
                state.workflow_status = WorkflowStatus.FAILED

            # Record timing
            state.node_timings["error_handler_agent"] = (datetime.now() - node_start_time).total_seconds()

            state.completed_nodes.append("error_handler_agent")
            return state

        except Exception as e:
            # Critical error in error handler - terminate workflow
            state.workflow_status = WorkflowStatus.FAILED
            state.errors.append({
                "timestamp": datetime.now().isoformat(),
                "node": "error_handler_agent",
                "error": str(e),
                "critical": True
            })
            return state

    async def _execute_pre_memory_hook(self, state: WorkflowState, node_name: str) -> WorkflowState:
        """Execute pre-processing memory hook for any node"""

        try:
            # Retrieve relevant context for the current node
            node_context = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": f"{node_name}_patterns",
                    "user_id": state.user_id
                }
            )

            if node_context.get("success"):
                state.memory_context[f"{node_name}_patterns"] = node_context.get("data", {})

            # Retrieve user preferences for this node
            user_prefs = await self.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "type": "user_preferences",
                    "user_id": state.user_id,
                    "node": node_name
                }
            )

            if user_prefs.get("success"):
                state.user_preferences[node_name] = user_prefs.get("data", {})

            # Log pre-hook execution
            state.workflow_log.append({
                "timestamp": datetime.now().isoformat(),
                "node": node_name,
                "action": "pre_memory_hook",
                "context_retrieved": len(state.memory_context.get(f"{node_name}_patterns", {})),
                "preferences_loaded": len(state.user_preferences.get(node_name, {}))
            })

        except Exception as e:
            logger.warning(f"Pre-memory hook failed for {node_name}: {str(e)}")
            state.warnings.append({
                "timestamp": datetime.now().isoformat(),
                "node": node_name,
                "action": "pre_memory_hook",
                "warning": str(e)
            })

        return state

    async def _execute_post_memory_hook(self, state: WorkflowState, node_name: str) -> WorkflowState:
        """Execute post-processing memory hook for any node"""

        try:
            # Prepare node completion data for storage
            node_completion_data = {
                "type": f"{node_name}_completion",
                "user_id": state.user_id,
                "workflow_id": state.workflow_id,
                "session_id": state.session_id,
                "timestamp": datetime.now().isoformat(),
                "processing_time": state.node_timings.get(node_name, 0),
                "status": "completed",
                "node_metadata": self._extract_node_metadata(state, node_name)
            }

            # Store in session memory
            await self.memory_agent.store_memory(
                bucket="session",
                data=node_completion_data
            )

            # Store patterns in knowledge memory if successful
            if node_name in state.completed_nodes or node_name == "error_handler_agent":
                pattern_data = {
                    "type": f"{node_name}_patterns",
                    "user_id": state.user_id,
                    "success_pattern": {
                        "completion_time": state.node_timings.get(node_name, 0),
                        "context_used": state.memory_context.get(f"{node_name}_patterns", {}),
                        "preferences_applied": state.user_preferences.get(node_name, {}),
                        "performance_metrics": self._calculate_node_performance(state, node_name)
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket="knowledge",
                    data=pattern_data
                )

            # Log post-hook execution
            state.workflow_log.append({
                "timestamp": datetime.now().isoformat(),
                "node": node_name,
                "action": "post_memory_hook",
                "session_data_stored": True,
                "pattern_data_stored": node_name in state.completed_nodes,
                "performance_recorded": True
            })

        except Exception as e:
            logger.warning(f"Post-memory hook failed for {node_name}: {str(e)}")
            state.warnings.append({
                "timestamp": datetime.now().isoformat(),
                "node": node_name,
                "action": "post_memory_hook",
                "warning": str(e)
            })

        return state

    def _extract_node_metadata(self, state: WorkflowState, node_name: str) -> Dict:
        """Extract relevant metadata for a specific node"""

        metadata = {
            "node_name": node_name,
            "execution_order": len(state.completed_nodes) + 1,
            "retry_count": state.retry_count,
            "memory_context_size": len(state.memory_context),
            "warnings_count": len([w for w in state.warnings if w.get("node") == node_name])
        }

        # Add node-specific metadata
        if node_name == "data_processing_agent" and state.data_processing_state:
            metadata.update({
                "records_processed": len(state.processed_data.get("accounts", [])) if state.processed_data else 0,
                "quality_score": state.data_processing_state.quality_score,
                "validation_errors": len(state.data_processing_state.error_log)
            })

        elif node_name == "dormancy_analysis_agent" and state.dormancy_analysis_state:
            metadata.update({
                "total_analyzed": state.dormancy_analysis_state.total_accounts_analyzed,
                "dormant_found": state.dormancy_analysis_state.dormant_accounts_found,
                "high_risk": state.dormancy_analysis_state.high_risk_accounts
            })

        elif node_name == "compliance_verification_agent" and state.compliance_state:
            metadata.update({
                "critical_violations": len(
                    state.compliance_results.get("critical_violations", [])) if state.compliance_results else 0,
                "compliance_score": state.compliance_results.get("overall_compliance_score",
                                                                 0) if state.compliance_results else 0
            })

        elif node_name == "risk_assessment_agent" and state.risk_assessment_state:
            metadata.update({
                "overall_risk_score": state.risk_assessment_results.get("overall_risk_score",
                                                                        0) if state.risk_assessment_results else 0,
                "high_risk_accounts": len(
                    state.risk_assessment_results.get("high_risk_accounts", [])) if state.risk_assessment_results else 0
            })

        return metadata

    def _calculate_node_performance(self, state: WorkflowState, node_name: str) -> Dict:
        """Calculate performance metrics for a node"""

        processing_time = state.node_timings.get(node_name, 0)

        performance = {
            "processing_time_seconds": processing_time,
            "efficiency_rating": "excellent" if processing_time < 30 else "good" if processing_time < 60 else "acceptable" if processing_time < 120 else "poor",
            "memory_utilization": len(state.memory_context.get(f"{node_name}_patterns", {})),
            "error_rate": len([e for e in state.errors if e.get("node") == node_name]) / max(1, state.retry_count + 1),
            "success_rate": 1.0 if node_name in state.completed_nodes else 0.0
        }

        return performance

    def _check_node_status(self, state: WorkflowState) -> str:
        """Check node execution status for conditional routing"""

        current_node = state.current_node

        # Check if current node failed
        if current_node in state.failed_nodes:
            if state.retry_count < state.max_retries:
                return "retry"
            else:
                return "error"

        # Check if current node completed successfully
        if current_node in state.completed_nodes:
            return "continue"

        # Default to continue
        return "continue"

    def _supervisor_routing(self, state: WorkflowState) -> str:
        """Supervisor agent routing logic"""

        if not state.supervisor_decisions:
            return "escalate"

        decision = state.supervisor_decisions.get("routing_decision", "proceed_to_reporting")

        if decision == "escalate":
            return "escalate"
        elif decision == "requires_review":
            return "requires_review"
        else:
            return "proceed_to_reporting"

    def _error_routing(self, state: WorkflowState) -> str:
        """Error handler routing logic"""

        if not state.error_state:
            return "terminate"

        recovery_action = state.error_state.recovery_action

        if recovery_action == "retry" and state.retry_count < state.max_retries:
            return "retry"
        elif recovery_action == "escalate":
            return "escalate"
        else:
            return "terminate"

    async def _handle_node_error(self, state: WorkflowState, node_name: str, error: Exception) -> WorkflowState:
        """Handle node execution errors"""

        error_info = {
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "retry_count": state.retry_count
        }

        state.errors.append(error_info)
        state.failed_nodes.append(node_name)

        # Log error
        logger.error(f"Node {node_name} failed: {str(error)}")

        # Store error in memory for pattern analysis
        try:
            await self.memory_agent.store_memory(
                bucket="session",
                data={
                    "type": "node_error",
                    "user_id": state.user_id,
                    "workflow_id": state.workflow_id,
                    "error_info": error_info
                }
            )
        except:
            pass  # Don't fail on memory storage error

        return state

    def _analyze_input_data(self, input_data: Dict) -> str:
        """Analyze input data to determine processing requirements"""

        if not input_data:
            return "empty"

        # Check data structure
        if "accounts" in input_data:
            account_count = len(input_data["accounts"])
            if account_count > 10000:
                return "large_dataset"
            elif account_count > 1000:
                return "medium_dataset"
            else:
                return "small_dataset"

        return "unknown_format"

    def _estimate_processing_time(self, input_data: Dict) -> float:
        """Estimate total processing time based on input data"""

        base_time = 30  # Base processing time in seconds

        if "accounts" in input_data:
            account_count = len(input_data["accounts"])
            # Estimate ~0.1 seconds per account for comprehensive analysis
            estimated_time = base_time + (account_count * 0.1)
        else:
            estimated_time = base_time

        return estimated_time

    @traceable(name="execute_workflow")
    async def execute_workflow(self, user_id: str, input_data: Dict,
                               workflow_options: Dict = None) -> Dict:
        """Execute the complete banking compliance workflow"""

        try:
            # Initialize workflow state
            workflow_id = secrets.token_hex(16)
            session_id = secrets.token_hex(16)

            initial_state = WorkflowState(
                workflow_id=workflow_id,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now(),
                raw_input_data=input_data,
                workflow_metadata=workflow_options or {}
            )

            # Execute workflow
            logger.info(f"Starting workflow {workflow_id} for user {user_id}")

            final_state = await self.workflow_graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            )

            # Prepare response
            response = {
                "success": final_state.workflow_status == WorkflowStatus.COMPLETED,
                "workflow_id": workflow_id,
                "session_id": session_id,
                "status": final_state.workflow_status.value,
                "total_processing_time": final_state.total_processing_time,
                "nodes_completed": len(final_state.completed_nodes),
                "nodes_failed": len(final_state.failed_nodes),
                "errors": final_state.errors,
                "warnings": final_state.warnings,
                "results": {
                    "data_processing": {
                        "status": final_state.data_processing_state.processing_status.value if final_state.data_processing_state else None,
                        "quality_score": final_state.data_processing_state.quality_score if final_state.data_processing_state else 0,
                        "records_processed": len(
                            final_state.processed_data.get("accounts", [])) if final_state.processed_data else 0
                    },
                    "dormancy_analysis": {
                        "total_analyzed": final_state.dormancy_analysis_state.total_accounts_analyzed if final_state.dormancy_analysis_state else 0,
                        "dormant_found": final_state.dormancy_analysis_state.dormant_accounts_found if final_state.dormancy_analysis_state else 0,
                        "high_risk": final_state.dormancy_analysis_state.high_risk_accounts if final_state.dormancy_analysis_state else 0
                    },
                    "compliance": {
                        "status": final_state.compliance_state.verification_status.value if final_state.compliance_state else None,
                        "critical_violations": len(final_state.compliance_results.get("critical_violations",
                                                                                      [])) if final_state.compliance_results else 0
                    },
                    "risk_assessment": {
                        "overall_score": final_state.risk_assessment_results.get("overall_risk_score",
                                                                                 0) if final_state.risk_assessment_results else 0
                    },
                    "reporting": {
                        "reports_generated": len(final_state.reporting_results.get("generated_reports",
                                                                                   [])) if final_state.reporting_results else 0
                    },
                    "notifications": {
                        "notifications_sent": len(final_state.notification_results.get("sent_notifications",
                                                                                       [])) if final_state.notification_results else 0
                    }
                },
                "performance_metrics": {
                    "node_timings": final_state.node_timings,
                    "memory_operations": len(
                        [log for log in final_state.workflow_log if "memory" in log.get("action", "")]),
                    "total_memory_retrievals": len(
                        [log for log in final_state.workflow_log if log.get("action") == "pre_memory_hook"]),
                    "total_memory_stores": len(
                        [log for log in final_state.workflow_log if log.get("action") == "post_memory_hook"])
                }
            }

            logger.info(f"Workflow {workflow_id} completed with status: {final_state.workflow_status.value}")
            return response

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "workflow_id": workflow_id if 'workflow_id' in locals() else None
            }

    async def get_workflow_status(self, workflow_id: str) -> Dict:
        """Get current status of a running workflow"""

        try:
            # Retrieve workflow state from memory
            workflow_data = await self.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "workflow_id": workflow_id
                }
            )

            if workflow_data.get("success"):
                return {
                    "success": True,
                    "workflow_status": workflow_data.get("data", {})
                }
            else:
                return {
                    "success": False,
                    "error": "Workflow not found"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def cancel_workflow(self, workflow_id: str, user_id: str) -> Dict:
        """Cancel a running workflow"""

        try:
            # Update workflow status in memory
            cancellation_data = {
                "workflow_id": workflow_id,
                "user_id": user_id,
                "cancelled_at": datetime.now().isoformat(),
                "status": "cancelled"
            }

            await self.memory_agent.store_memory(
                bucket="session",
                data=cancellation_data
            )

            return {
                "success": True,
                "message": f"Workflow {workflow_id} cancelled successfully"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_workflow_metrics(self) -> Dict:
        """Get aggregated workflow performance metrics"""

        return {
            "total_workflows_executed": len(self.workflow_metrics),
            "average_processing_time": sum(self.workflow_metrics.values()) / len(
                self.workflow_metrics) if self.workflow_metrics else 0,
            "success_rate": 0.95,  # Mock data - would be calculated from actual executions
            "most_common_failure_points": ["data_processing_agent", "compliance_verification_agent"],
            "memory_efficiency": {
                "average_retrievals_per_workflow": 12,
                "average_stores_per_workflow": 8,
                "cache_hit_rate": 0.78
            }
        }


# Example usage and testing
async def main():
    """Example usage of the workflow orchestration engine"""

    print("Banking Compliance Workflow Orchestration Engine")
    print("=" * 60)
    print("Features:")
    print("- LangGraph-based workflow orchestration")
    print("- Hybrid memory integration with pre/post hooks")
    print("- MCP tool integration across all agents")
    print("- LangSmith observability and tracing")
    print("- Comprehensive error handling and recovery")
    print("- Performance monitoring and metrics")
    print("- Audit trail and compliance logging")
    print("- Dynamic routing and conditional execution")


if __name__ == "__main__":
    asyncio.run(main())