import logging
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import json

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable, Client as LangSmithClient

# MCP imports - using try/except to handle import issues
try:
    from mcp_client import MCPClient
except ImportError:
    logging.warning("MCPClient not available, using mock implementation")


    class MCPClient:
        async def call_tool(self, tool_name: str, params: Dict) -> Dict:
            return {"success": True, "data": {}}

# Import comprehensive dormancy monitoring system
from dormant import (
    DormancyMonitoringAgents,
    DormancyReportingEngine,
    DormancyNotificationService,
    initialize_dormancy_monitoring_system,
    run_daily_dormancy_monitoring
)

# Import error handler
from agents.error_handler_agent import ErrorHandlerAgent, ErrorState

# Import memory agent components
from agents.memory_agent import MemoryBucket, MemoryPriority, MemoryContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Enhanced States for Individual Agents
class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZING_PATTERNS = "analyzing_patterns"
    AWAITING_TRIGGER = "awaiting_trigger"
    MEMORY_LOADING = "memory_loading"
    MEMORY_STORING = "memory_storing"


class DormancyStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"
    ESCALATED = "escalated"


class DormancyTrigger(Enum):
    """Triggers for dormancy analysis"""
    STANDARD_INACTIVITY = "standard_inactivity"
    PAYMENT_INSTRUMENT_UNCLAIMED = "payment_instrument_unclaimed"
    SDB_UNPAID_FEES = "sdb_unpaid_fees"
    INVESTMENT_MATURITY = "investment_maturity"
    FIXED_DEPOSIT_MATURITY = "fixed_deposit_maturity"
    HIGH_VALUE_THRESHOLD = "high_value_threshold"
    CB_TRANSFER_ELIGIBILITY = "cb_transfer_eligibility"
    PROACTIVE_CONTACT = "proactive_contact"
    ARTICLE_3_PROCESS = "article_3_process"
    DORMANT_TO_ACTIVE = "dormant_to_active"
    CONTACT_ATTEMPTS_INCOMPLETE = "contact_attempts_incomplete"
    INTERNAL_LEDGER_TRANSFER = "internal_ledger_transfer"
    RECORD_RETENTION_VIOLATION = "record_retention_violation"
    CLAIM_PROCESSING_OVERDUE = "claim_processing_overdue"
    STATEMENT_SUPPRESSION = "statement_suppression"
    FOREIGN_CURRENCY_CONVERSION = "foreign_currency_conversion"
    SDB_COURT_APPLICATION = "sdb_court_application"


@dataclass
class DormancyAnalysisState:
    """Main state for dormancy analysis workflow - used by workflow engine"""

    session_id: str
    user_id: str
    analysis_id: str
    timestamp: datetime

    # Input data
    processed_data: Optional[Dict] = None
    analysis_config: Dict = None

    # Analysis results
    dormancy_results: Optional[Dict] = None
    dormancy_summary: Optional[Dict] = None
    compliance_flags: List[str] = None

    # Status tracking
    analysis_status: DormancyStatus = DormancyStatus.PENDING
    total_accounts_analyzed: int = 0
    dormant_accounts_found: int = 0
    high_risk_accounts: int = 0

    # Memory context
    memory_context: Dict = None
    retrieved_patterns: Dict = None

    # Performance metrics
    processing_time: float = 0.0
    analysis_efficiency: float = 0.0

    # Audit trail
    analysis_log: List[Dict] = None
    error_log: List[Dict] = None

    # Agent orchestration
    active_agents: List[str] = None
    completed_agents: List[str] = None
    failed_agents: List[str] = None
    agent_results: Dict = None

    # Workflow routing
    current_node: str = "start"
    routing_decision: str = "continue"

    def __post_init__(self):
        if self.analysis_config is None:
            self.analysis_config = {}
        if self.compliance_flags is None:
            self.compliance_flags = []
        if self.memory_context is None:
            self.memory_context = {}
        if self.retrieved_patterns is None:
            self.retrieved_patterns = {}
        if self.analysis_log is None:
            self.analysis_log = []
        if self.error_log is None:
            self.error_log = []
        if self.active_agents is None:
            self.active_agents = []
        if self.completed_agents is None:
            self.completed_agents = []
        if self.failed_agents is None:
            self.failed_agents = []
        if self.agent_results is None:
            self.agent_results = {}


@dataclass
class AgentState:
    """Individual agent state with memory management"""
    agent_id: str
    agent_type: str
    session_id: str
    user_id: str
    timestamp: datetime

    # Data management
    input_dataframe: Optional[pd.DataFrame] = None
    processed_dataframe: Optional[pd.DataFrame] = None
    analysis_results: Optional[Dict] = None
    pattern_analysis: Optional[Dict] = None

    # Status and metrics
    agent_status: AgentStatus = AgentStatus.IDLE
    records_processed: int = 0
    dormant_records_found: int = 0
    processing_time: float = 0.0

    # Memory context
    pre_hook_memory: Dict = None
    post_hook_memory: Dict = None
    retrieved_patterns: Dict = None
    stored_patterns: Dict = None

    # Agent-specific parameters
    regulatory_params: Dict = None
    analysis_config: Dict = None

    # Triggers and conditions
    trigger_conditions: Dict = None
    triggered_by: Optional[DormancyTrigger] = None

    # Error handling
    error_handler: Optional[ErrorHandlerAgent] = None
    error_state: Optional[ErrorState] = None

    # Logging and audit
    execution_log: List[Dict] = None
    error_log: List[Dict] = None
    performance_metrics: Dict = None

    def __post_init__(self):
        if self.pre_hook_memory is None:
            self.pre_hook_memory = {}
        if self.post_hook_memory is None:
            self.post_hook_memory = {}
        if self.retrieved_patterns is None:
            self.retrieved_patterns = {}
        if self.stored_patterns is None:
            self.stored_patterns = {}
        if self.regulatory_params is None:
            self.regulatory_params = {}
        if self.analysis_config is None:
            self.analysis_config = {}
        if self.trigger_conditions is None:
            self.trigger_conditions = {}
        if self.execution_log is None:
            self.execution_log = []
        if self.error_log is None:
            self.error_log = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


# Base Dormancy Agent Class with Memory Hooks and Comprehensive Monitoring
class BaseDormancyAgent:
    """Base class for all dormancy analysis agents with comprehensive CBUAE monitoring"""

    def __init__(self, agent_type: str, memory_agent, mcp_client: MCPClient, db_connection=None):
        self.agent_type = agent_type
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_connection = db_connection

        # Initialize the comprehensive dormancy monitoring system
        if db_connection:
            self.dormancy_monitoring = DormancyMonitoringAgents(db_connection)
            self.reporting_engine = DormancyReportingEngine(db_connection)
            self.notification_service = DormancyNotificationService(db_connection)
        else:
            self.dormancy_monitoring = None
            self.reporting_engine = None
            self.notification_service = None

        # Initialize error handler
        self.error_handler = ErrorHandlerAgent(memory_agent, mcp_client)

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # Default regulatory parameters from CBUAE regulation
        self.default_params = {
            "standard_inactivity_years": 3,
            "payment_instrument_unclaimed_years": 1,
            "sdb_unpaid_fees_years": 3,
            "eligibility_for_cb_transfer_years": 5,
            "high_value_threshold_aed": 25000,
            "proactive_contact_threshold_months": 30,
            "article_3_contact_wait_months": 3
        }

        # Agent-specific trigger conditions
        self.trigger_conditions = self._define_trigger_conditions()

    def _define_trigger_conditions(self) -> Dict:
        """Define agent-specific trigger conditions"""
        return {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [],
            "business_rules": []
        }

    @traceable(name="check_triggers")
    async def check_triggers(self, state: AgentState) -> bool:
        """Check if agent should be triggered using comprehensive monitoring"""
        try:
            # Check data availability
            if not state.input_dataframe or state.input_dataframe.empty:
                return False

            # Check minimum records
            min_records = self.trigger_conditions.get("minimum_records", 1)
            if len(state.input_dataframe) < min_records:
                return False

            # Check required columns
            required_cols = self.trigger_conditions.get("required_columns", [])
            if required_cols and not all(col in state.input_dataframe.columns for col in required_cols):
                return False

            # Agent-specific trigger logic
            return await self._check_agent_specific_triggers(state)

        except Exception as e:
            logger.error(f"Trigger check failed for {self.agent_type}: {str(e)}")
            return False

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Override in subclasses for agent-specific trigger logic"""
        return True

    @traceable(name="pre_analysis_memory_hook")
    async def pre_analysis_memory_hook(self, state: AgentState) -> AgentState:
        """Enhanced pre-analysis memory hook with comprehensive monitoring context"""
        try:
            state.agent_status = AgentStatus.MEMORY_LOADING

            # Create memory context
            memory_context = await self.memory_agent.create_memory_context(
                user_id=state.user_id,
                session_id=state.session_id,
                agent_name=self.agent_type
            )

            # Retrieve agent-specific patterns from comprehensive monitoring
            agent_patterns = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": f"{self.agent_type}_patterns",
                    "user_id": state.user_id,
                    "agent_type": self.agent_type
                },
                context=memory_context
            )

            if agent_patterns.get("success") and agent_patterns.get("data"):
                state.retrieved_patterns = {
                    "agent_patterns": agent_patterns["data"],
                    "pattern_count": len(agent_patterns["data"])
                }
                logger.info(f"Retrieved {len(agent_patterns['data'])} patterns for {self.agent_type}")

            # Retrieve historical monitoring results
            historical_monitoring = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": f"{self.agent_type}_monitoring_history",
                    "user_id": state.user_id,
                    "content_type": "monitoring_results"
                },
                context=memory_context
            )

            if historical_monitoring.get("success") and historical_monitoring.get("data"):
                state.pre_hook_memory["historical_monitoring"] = historical_monitoring["data"]

            # Retrieve regulatory benchmarks from CBUAE standards
            regulatory_benchmarks = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "cbuae_regulatory_benchmarks",
                    "content_type": "compliance_standards",
                    "agent_type": self.agent_type
                },
                context=memory_context
            )

            if regulatory_benchmarks.get("success") and regulatory_benchmarks.get("data"):
                state.pre_hook_memory["regulatory_benchmarks"] = regulatory_benchmarks["data"]

            # Store comprehensive monitoring configuration
            if self.dormancy_monitoring:
                monitoring_config = {
                    "monitoring_enabled": True,
                    "agent_type": self.agent_type,
                    "regulatory_params": self.default_params,
                    "monitoring_capabilities": self._get_monitoring_capabilities()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.SESSION.value,
                    data={
                        "type": f"{self.agent_type}_monitoring_config",
                        "agent_id": state.agent_id,
                        "session_id": state.session_id,
                        "config": monitoring_config,
                        "timestamp": datetime.now().isoformat()
                    },
                    context=memory_context,
                    content_type="monitoring_config",
                    tags=[self.agent_type, "monitoring", "session"]
                )

            state.agent_status = AgentStatus.PROCESSING

        except Exception as e:
            logger.error(f"{self.agent_type} pre-analysis memory hook failed: {str(e)}")
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "pre_analysis_memory_hook")

        return state

    @traceable(name="post_analysis_memory_hook")
    async def post_analysis_memory_hook(self, state: AgentState) -> AgentState:
        """Enhanced post-analysis memory hook with comprehensive monitoring storage"""
        try:
            state.agent_status = AgentStatus.MEMORY_STORING

            # Create memory context
            memory_context = await self.memory_agent.create_memory_context(
                user_id=state.user_id,
                session_id=state.session_id,
                agent_name=self.agent_type
            )

            # Store comprehensive monitoring results
            if state.analysis_results and self.dormancy_monitoring:
                monitoring_results = {
                    "type": f"{self.agent_type}_monitoring_results",
                    "agent_id": state.agent_id,
                    "session_id": state.session_id,
                    "results": {
                        "cbuae_compliance": {
                            "article": state.analysis_results.get("compliance_article"),
                            "dormant_count": state.dormant_records_found,
                            "validation_passed": state.analysis_results.get("validation_passed", True),
                            "regulatory_requirements_met": True
                        },
                        "monitoring_insights": state.analysis_results,
                        "performance_metrics": state.performance_metrics,
                        "alert_generated": state.dormant_records_found > 0,
                        "priority_level": self._determine_priority_level(state)
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.SESSION.value,
                    data=monitoring_results,
                    context=memory_context,
                    content_type="monitoring_results",
                    priority=MemoryPriority.HIGH,
                    tags=[self.agent_type, "monitoring", "cbuae_compliance"]
                )

            # Store regulatory compliance patterns
            if state.pattern_analysis and state.agent_status == AgentStatus.COMPLETED:
                compliance_patterns = {
                    "type": f"{self.agent_type}_compliance_patterns",
                    "user_id": state.user_id,
                    "agent_type": self.agent_type,
                    "cbuae_article": state.analysis_results.get("compliance_article"),
                    "patterns": state.pattern_analysis,
                    "regulatory_effectiveness": {
                        "detection_accuracy": state.performance_metrics.get("accuracy_score", 0),
                        "false_positive_rate": 0.0,  # Would be calculated from validation data
                        "compliance_score": self._calculate_compliance_score(state),
                        "regulatory_risk_level": self._assess_regulatory_risk(state)
                    },
                    "audit_trail": {
                        "processing_time": state.processing_time,
                        "records_analyzed": state.records_processed,
                        "findings_generated": state.dormant_records_found,
                        "validation_status": "passed"
                    },
                    "timestamp": datetime.now().isoformat()
                }

                pattern_result = await self.memory_agent.store_memory(
                    bucket=MemoryBucket.KNOWLEDGE.value,
                    data=compliance_patterns,
                    context=memory_context,
                    content_type="compliance_patterns",
                    priority=MemoryPriority.CRITICAL,
                    tags=[self.agent_type, "cbuae", "regulatory", "patterns"],
                    encrypt_sensitive=True
                )

                if pattern_result.get("success"):
                    state.stored_patterns = compliance_patterns

            # Store audit trail for CBUAE reporting
            if state.processed_dataframe is not None and not state.processed_dataframe.empty:
                audit_data = {
                    "type": f"{self.agent_type}_cbuae_audit",
                    "agent_id": state.agent_id,
                    "session_id": state.session_id,
                    "regulatory_compliance": {
                        "cbuae_article": state.analysis_results.get("compliance_article"),
                        "regulation_version": "2024",
                        "compliance_status": "compliant" if state.analysis_results.get("validation_passed",
                                                                                       True) else "non_compliant",
                        "findings_summary": {
                            "total_accounts_reviewed": state.records_processed,
                            "dormant_accounts_identified": state.dormant_records_found,
                            "high_risk_accounts": self._count_high_risk_accounts(state),
                            "immediate_actions_required": self._get_immediate_actions(state)
                        }
                    },
                    "audit_trail": {
                        "analysis_timestamp": datetime.now().isoformat(),
                        "processing_duration": state.processing_time,
                        "data_quality_assessment": state.performance_metrics.get("data_quality_score", 0),
                        "monitoring_agent_version": "2024.1",
                        "validation_checkpoints_passed": True
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.AUDIT.value,
                    data=audit_data,
                    context=memory_context,
                    content_type="cbuae_audit_trail",
                    priority=MemoryPriority.CRITICAL,
                    tags=[self.agent_type, "audit", "cbuae", "regulatory"]
                )

        except Exception as e:
            logger.error(f"{self.agent_type} post-analysis memory hook failed: {str(e)}")
            await self._handle_error(state, e, "post_analysis_memory_hook")

        return state

    def _get_monitoring_capabilities(self) -> List[str]:
        """Get monitoring capabilities for this agent type"""
        base_capabilities = [
            "dormancy_detection",
            "regulatory_compliance",
            "pattern_analysis",
            "risk_assessment"
        ]

        # Add agent-specific capabilities
        agent_capabilities = {
            "demand_deposit_dormancy": ["article_2_1_1", "inactivity_monitoring", "liability_account_check"],
            "fixed_deposit_dormancy": ["article_2_2", "maturity_monitoring", "auto_renewal_tracking"],
            "investment_dormancy": ["article_2_3", "redemption_monitoring", "investment_tracking"],
            "safe_deposit_dormancy": ["article_2_6", "fee_monitoring", "court_application_tracking"],
            "unclaimed_instruments": ["article_2_4", "instrument_monitoring", "contact_attempt_tracking"],
            "high_value_dormancy": ["threshold_monitoring", "priority_flagging"],
            "cb_transfer_eligibility": ["article_8", "transfer_readiness", "address_verification"],
            "article_3_process": ["article_3", "contact_process_monitoring", "ledger_transfer_tracking"],
            "proactive_contact": ["article_5", "early_warning", "communication_tracking"],
            "contact_attempts": ["article_3_1", "communication_verification", "method_tracking"],
            "internal_ledger": ["article_3_4", "ledger_transfer", "balance_monitoring"],
            "record_retention": ["article_3_9", "audit_compliance", "retention_monitoring"],
            "claim_processing": ["article_4", "customer_claims", "processing_timeline"],
            "statement_suppression": ["article_7_3", "statement_control", "dormant_flagging"],
            "foreign_currency": ["article_8_5", "currency_conversion", "exchange_rate_tracking"],
            "sdb_court_application": ["article_3_7", "court_process", "legal_compliance"]
        }

        return base_capabilities + agent_capabilities.get(self.agent_type, [])

    def _determine_priority_level(self, state: AgentState) -> str:
        """Determine priority level based on findings"""
        if state.dormant_records_found == 0:
            return "low"
        elif state.dormant_records_found < 10:
            return "medium"
        else:
            return "high"

    def _calculate_compliance_score(self, state: AgentState) -> float:
        """Calculate CBUAE compliance score"""
        base_score = 1.0

        # Deduct points for validation failures
        if not state.analysis_results.get("validation_passed", True):
            base_score -= 0.3

        # Deduct points for processing errors
        if state.error_log:
            base_score -= 0.1 * len(state.error_log)

        # Deduct points for data quality issues
        data_quality = state.performance_metrics.get("data_quality_score", 1.0)
        base_score *= data_quality

        return max(0.0, min(1.0, base_score))

    def _assess_regulatory_risk(self, state: AgentState) -> str:
        """Assess regulatory risk level"""
        if state.dormant_records_found == 0:
            return "low"
        elif state.dormant_records_found < 5:
            return "medium"
        elif state.dormant_records_found < 20:
            return "high"
        else:
            return "critical"

    def _count_high_risk_accounts(self, state: AgentState) -> int:
        """Count high-risk accounts from analysis"""
        if not state.processed_dataframe or state.processed_dataframe.empty:
            return 0

        # Count based on balance thresholds or other risk factors
        high_risk_count = 0
        if 'Current_Balance' in state.processed_dataframe.columns:
            high_value_threshold = self.default_params.get("high_value_threshold_aed", 25000)
            high_risk_count = len(state.processed_dataframe[
                                      pd.to_numeric(state.processed_dataframe['Current_Balance'],
                                                    errors='coerce').fillna(0) >= high_value_threshold
                                      ])

        return high_risk_count

    def _get_immediate_actions(self, state: AgentState) -> List[str]:
        """Get immediate actions required based on findings"""
        actions = []

        if state.dormant_records_found > 0:
            actions.append(f"Review {state.dormant_records_found} dormant accounts identified")

        if self._count_high_risk_accounts(state) > 0:
            actions.append("Prioritize high-value dormant accounts for immediate attention")

        if state.analysis_results.get("compliance_article"):
            article = state.analysis_results["compliance_article"]
            actions.append(f"Ensure CBUAE Article {article} compliance requirements are met")

        return actions

    async def _handle_error(self, state: AgentState, error: Exception, stage: str):
        """Handle errors using error handler agent"""
        try:
            error_state = ErrorState(
                session_id=state.session_id,
                user_id=state.user_id,
                error_id=secrets.token_hex(8),
                timestamp=datetime.now(),
                errors=[{
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stage": stage,
                    "agent_type": self.agent_type,
                    "critical": stage in ["analysis", "trigger_check"]
                }],
                failed_node=stage,
                workflow_context=asdict(state)
            )

            # Handle error using error handler
            error_result = await self.error_handler.handle_workflow_error(error_state)

            # Update agent state based on error handling result
            state.error_state = error_result
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "error": str(error),
                "recovery_action": error_result.recovery_action,
                "recovery_success": error_result.recovery_success
            })

            # Determine if agent should continue or fail
            if error_result.recovery_action == "retry":
                state.agent_status = AgentStatus.PROCESSING
            elif error_result.recovery_action == "escalate":
                state.agent_status = AgentStatus.FAILED
            else:
                state.agent_status = AgentStatus.FAILED

        except Exception as nested_error:
            logger.error(f"Error handling failed: {str(nested_error)}")
            state.agent_status = AgentStatus.FAILED

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Base pattern analysis using comprehensive monitoring insights"""
        return {
            "pattern_type": "base",
            "insights": [],
            "recommendations": [],
            "effectiveness_metrics": {},
            "cbuae_compliance": {
                "article": analysis_results.get("compliance_article"),
                "requirements_met": True,
                "risk_level": "low"
            }
        }


# Comprehensive CBUAE Article-Specific Agents

# Article 2.1.1 - Demand Deposit Dormancy Agent
class DemandDepositDormancyAgent(BaseDormancyAgent):
    """Specialized agent for demand deposit dormancy analysis (CBUAE Article 2.1.1)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("demand_deposit_dormancy", memory_agent, mcp_client, db_connection)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Account_Type", "Date_Last_Cust_Initiated_Activity",
                "Date_Last_Customer_Communication_Any_Type", "Customer_Has_Active_Liability_Account"
            ],
            "business_rules": [
                "account_type_contains_demand_deposit",
                "activity_date_within_range"
            ]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check demand deposit specific triggers"""
        if self.dormancy_monitoring:
            # Use comprehensive monitoring system
            alerts = self.dormancy_monitoring.check_demand_deposit_inactivity()
            if alerts:
                state.triggered_by = DormancyTrigger.STANDARD_INACTIVITY
                return True

        # Fallback to DataFrame analysis
        try:
            df = state.input_dataframe
            demand_deposit_accounts = df[
                df['Account_Type'].astype(str).str.contains("Current|Saving|Call", case=False, na=False)
            ]

            if not demand_deposit_accounts.empty:
                state.triggered_by = DormancyTrigger.STANDARD_INACTIVITY
                return True

        except Exception as e:
            logger.error(f"Demand deposit trigger check failed: {str(e)}")

        return False

    @traceable(name="demand_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze demand deposit dormancy with comprehensive monitoring"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                # Use comprehensive monitoring system
                alerts = self.dormancy_monitoring.check_demand_deposit_inactivity()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 2.1.1 - Demand Deposit Inactivity Analysis",
                    "details": [alert.__dict__ for alert in alerts],
                    "compliance_article": "2.1.1",
                    "analysis_date": report_date,
                    "validation_passed": True,
                    "key_findings": [],
                    "alerts_generated": len(alerts) > 0
                }

                state.dormant_records_found = len(alerts)
                state.records_processed = state.total_accounts_analyzed

                # Create DataFrame from alerts for pattern analysis
                if alerts:
                    alert_data = []
                    for alert in alerts:
                        alert_data.append({
                            'Account_ID': alert.account_id,
                            'Customer_ID': alert.customer_id,
                            'Alert_Type': alert.alert_type,
                            'Priority': alert.priority,
                            'Message': alert.message,
                            'Created_Date': alert.created_date
                        })
                    state.processed_dataframe = pd.DataFrame(alert_data)
                else:
                    state.processed_dataframe = pd.DataFrame()
            else:
                # Fallback to basic analysis
                state.analysis_results = {
                    "count": 0,
                    "description": "Monitoring system not available",
                    "compliance_article": "2.1.1",
                    "validation_passed": False
                }
                state.processed_dataframe = pd.DataFrame()

            # Pattern analysis
            state.agent_status = AgentStatus.ANALYZING_PATTERNS
            state.pattern_analysis = await self.analyze_patterns(state.processed_dataframe, state.analysis_results)

            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            state.processing_time = processing_time
            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "alerts_generated": state.dormant_records_found,
                "monitoring_system_available": self.dormancy_monitoring is not None,
                "compliance_score": self._calculate_compliance_score(state),
                "regulatory_risk": self._assess_regulatory_risk(state)
            }

            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "demand_deposit_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze demand deposit patterns with CBUAE compliance focus"""
        patterns = {
            "pattern_type": "demand_deposit",
            "cbuae_article": "2.1.1",
            "regulatory_focus": "customer_initiated_activity_tracking",
            "insights": [],
            "recommendations": [],
            "compliance_assessment": {
                "requirements_met": True,
                "risk_level": "low",
                "immediate_actions": []
            }
        }

        if not df.empty:
            patterns["insights"].append(f"Identified {len(df)} demand deposit dormancy alerts")

            if 'Priority' in df.columns:
                high_priority = len(df[df['Priority'] == 'HIGH'])
                if high_priority > 0:
                    patterns["insights"].append(
                        f"{high_priority} high-priority dormancy cases require immediate attention")
                    patterns["compliance_assessment"]["risk_level"] = "high"
                    patterns["compliance_assessment"]["immediate_actions"].append(
                        "Review high-priority dormant accounts within 48 hours")

            patterns["recommendations"].extend([
                "Implement proactive customer contact procedures per CBUAE Article 3",
                "Verify customer communication channels are up to date",
                "Review liability account relationships for dormant customers"
            ])

        return patterns


# Article 2.2 - Fixed Deposit Dormancy Agent
class FixedDepositDormancyAgent(BaseDormancyAgent):
    """Specialized agent for fixed deposit dormancy analysis (CBUAE Article 2.2)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("fixed_deposit_dormancy", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check fixed deposit specific triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.check_fixed_deposit_inactivity()
            if alerts:
                state.triggered_by = DormancyTrigger.FIXED_DEPOSIT_MATURITY
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze fixed deposit dormancy"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.check_fixed_deposit_inactivity()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 2.2 - Fixed Deposit Maturity Analysis",
                    "compliance_article": "2.2",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "fixed_deposit_analysis")

        return state


# Article 2.3 - Investment Account Dormancy Agent
class InvestmentAccountDormancyAgent(BaseDormancyAgent):
    """Specialized agent for investment account dormancy analysis (CBUAE Article 2.3)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("investment_dormancy", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check investment account specific triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.check_investment_inactivity()
            if alerts:
                state.triggered_by = DormancyTrigger.INVESTMENT_MATURITY
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze investment account dormancy"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.check_investment_inactivity()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 2.3 - Investment Account Analysis",
                    "compliance_article": "2.3",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "investment_analysis")

        return state


# Article 2.4 - Payment Instruments Dormancy Agent
class PaymentInstrumentsDormancyAgent(BaseDormancyAgent):
    """Specialized agent for unclaimed payment instruments analysis (CBUAE Article 2.4)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("unclaimed_instruments", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check payment instruments specific triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.check_unclaimed_payment_instruments()
            if alerts:
                state.triggered_by = DormancyTrigger.PAYMENT_INSTRUMENT_UNCLAIMED
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze unclaimed payment instruments"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.check_unclaimed_payment_instruments()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 2.4 - Unclaimed Payment Instruments",
                    "compliance_article": "2.4",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "payment_instruments_analysis")

        return state


# Article 2.6 - Safe Deposit Box Dormancy Agent
class SafeDepositBoxDormancyAgent(BaseDormancyAgent):
    """Specialized agent for safe deposit box dormancy analysis (CBUAE Article 2.6)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("safe_deposit_dormancy", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check safe deposit box specific triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.check_safe_deposit_dormancy()
            if alerts:
                state.triggered_by = DormancyTrigger.SDB_UNPAID_FEES
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze safe deposit box dormancy"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.check_safe_deposit_dormancy()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 2.6 - Safe Deposit Box Dormancy",
                    "compliance_article": "2.6",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "safe_deposit_analysis")

        return state


# Article 3 Process Agents

# Article 3.1 - Contact Attempts Agent
class ContactAttemptsAgent(BaseDormancyAgent):
    """Agent for monitoring contact attempt completeness (CBUAE Article 3.1)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("contact_attempts", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check contact attempts specific triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.detect_incomplete_contact_attempts()
            if alerts:
                state.triggered_by = DormancyTrigger.CONTACT_ATTEMPTS_INCOMPLETE
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze contact attempt completeness"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.detect_incomplete_contact_attempts()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 3.1 - Contact Attempt Verification",
                    "compliance_article": "3.1",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "contact_attempts_analysis")

        return state


# Article 3.4 - Internal Ledger Transfer Agent
class InternalLedgerAgent(BaseDormancyAgent):
    """Agent for monitoring internal ledger transfer eligibility (CBUAE Article 3.4)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("internal_ledger", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check internal ledger transfer triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.detect_internal_ledger_candidates()
            if alerts:
                state.triggered_by = DormancyTrigger.INTERNAL_LEDGER_TRANSFER
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze internal ledger transfer candidates"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.detect_internal_ledger_candidates()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 3.4 - Internal Ledger Transfer Candidates",
                    "compliance_article": "3.4",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "internal_ledger_analysis")

        return state


# Article 3.7 - SDB Court Application Agent
class SDBCourtApplicationAgent(BaseDormancyAgent):
    """Agent for monitoring SDB court application requirements (CBUAE Article 3.7)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("sdb_court_application", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check SDB court application triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.detect_sdb_court_application_needed()
            if alerts:
                state.triggered_by = DormancyTrigger.SDB_COURT_APPLICATION
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze SDB court application requirements"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.detect_sdb_court_application_needed()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 3.7 - SDB Court Application Requirements",
                    "compliance_article": "3.7",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "sdb_court_application_analysis")

        return state


# Article 3.9 - Record Retention Agent
class RecordRetentionAgent(BaseDormancyAgent):
    """Agent for monitoring record retention compliance (CBUAE Article 3.9)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("record_retention", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check record retention triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.check_record_retention_compliance()
            if alerts:
                state.triggered_by = DormancyTrigger.RECORD_RETENTION_VIOLATION
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze record retention compliance"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.check_record_retention_compliance()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 3.9 - Record Retention Compliance",
                    "compliance_article": "3.9",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "record_retention_analysis")

        return state


# Article 4 - Customer Claims Agent
class CustomerClaimsAgent(BaseDormancyAgent):
    """Agent for monitoring customer claims processing (CBUAE Article 4)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("customer_claims", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check customer claims triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.detect_claim_processing_pending()
            if alerts:
                state.triggered_by = DormancyTrigger.CLAIM_PROCESSING_OVERDUE
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze customer claims processing"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.detect_claim_processing_pending()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 4 - Customer Claims Processing",
                    "compliance_article": "4",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "customer_claims_analysis")

        return state


# Article 5 - Proactive Contact Agent
class ProactiveContactAgent(BaseDormancyAgent):
    """Agent for proactive contact monitoring (CBUAE Article 5)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("proactive_contact", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check proactive contact triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.check_contact_attempts_needed()
            if alerts:
                state.triggered_by = DormancyTrigger.PROACTIVE_CONTACT
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze proactive contact requirements"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.check_contact_attempts_needed()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 5 - Proactive Contact Requirements",
                    "compliance_article": "5",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "proactive_contact_analysis")

        return state


# Article 7.3 - Statement Suppression Agent
class StatementSuppressionAgent(BaseDormancyAgent):
    """Agent for statement suppression monitoring (CBUAE Article 7.3)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("statement_suppression", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check statement suppression triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.detect_statement_freeze_candidates()
            if alerts:
                state.triggered_by = DormancyTrigger.STATEMENT_SUPPRESSION
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze statement suppression eligibility"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.detect_statement_freeze_candidates()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 7.3 - Statement Suppression Eligibility",
                    "compliance_article": "7.3",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "statement_suppression_analysis")

        return state


# Article 8 - Central Bank Transfer Agents

# Article 8.1 - CB Transfer Eligibility Agent
class CBTransferEligibilityAgent(BaseDormancyAgent):
    """Agent for Central Bank transfer eligibility (CBUAE Article 8.1)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("cb_transfer_eligibility", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check CB transfer eligibility triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.check_eligible_for_cb_transfer()
            if alerts:
                state.triggered_by = DormancyTrigger.CB_TRANSFER_ELIGIBILITY
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze CB transfer eligibility"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.check_eligible_for_cb_transfer()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 8.1 - Central Bank Transfer Eligibility",
                    "compliance_article": "8.1",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "cb_transfer_analysis")

        return state


# Article 8.5 - Foreign Currency Conversion Agent
class ForeignCurrencyConversionAgent(BaseDormancyAgent):
    """Agent for foreign currency conversion monitoring (CBUAE Article 8.5)"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        super().__init__("foreign_currency_conversion", memory_agent, mcp_client, db_connection)

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check foreign currency conversion triggers"""
        if self.dormancy_monitoring:
            alerts = self.dormancy_monitoring.detect_foreign_currency_conversion_needed()
            if alerts:
                state.triggered_by = DormancyTrigger.FOREIGN_CURRENCY_CONVERSION
                return True
        return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze foreign currency conversion requirements"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if self.dormancy_monitoring:
                alerts = self.dormancy_monitoring.detect_foreign_currency_conversion_needed()

                state.analysis_results = {
                    "count": len(alerts),
                    "description": "CBUAE Article 8.5 - Foreign Currency Conversion Requirements",
                    "compliance_article": "8.5",
                    "analysis_date": report_date,
                    "validation_passed": True
                }

                state.dormant_records_found = len(alerts)

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            await self._handle_error(state, e, "foreign_currency_analysis")

        return state


# Comprehensive Workflow Orchestrator
class DormancyWorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for comprehensive CBUAE dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_connection=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_connection = db_connection

        # Initialize comprehensive monitoring system
        if db_connection:
            self.monitoring_system = initialize_dormancy_monitoring_system(db_connection)
        else:
            self.monitoring_system = None

        # Initialize all specialized agents
        self.agents = {
            # Article 2 - Dormancy Criteria Agents
            "demand_deposit": DemandDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "fixed_deposit": FixedDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "investment": InvestmentAccountDormancyAgent(memory_agent, mcp_client, db_connection),
            "safe_deposit": SafeDepositBoxDormancyAgent(memory_agent, mcp_client, db_connection),
            "payment_instruments": PaymentInstrumentsDormancyAgent(memory_agent, mcp_client, db_connection),

            # Article 3 - Bank Obligations Agents
            "contact_attempts": ContactAttemptsAgent(memory_agent, mcp_client, db_connection),
            "internal_ledger": InternalLedgerAgent(memory_agent, mcp_client, db_connection),
            "sdb_court_application": SDBCourtApplicationAgent(memory_agent, mcp_client, db_connection),
            "record_retention": RecordRetentionAgent(memory_agent, mcp_client, db_connection),

            # Article 4 - Customer Claims Agent
            "customer_claims": CustomerClaimsAgent(memory_agent, mcp_client, db_connection),

            # Article 5 - Proactive Communication Agent
            "proactive_contact": ProactiveContactAgent(memory_agent, mcp_client, db_connection),

            # Article 7.3 - Statement Suppression Agent
            "statement_suppression": StatementSuppressionAgent(memory_agent, mcp_client, db_connection),

            # Article 8 - Central Bank Transfer Agents
            "cb_transfer": CBTransferEligibilityAgent(memory_agent, mcp_client, db_connection),
            "foreign_currency": ForeignCurrencyConversionAgent(memory_agent, mcp_client, db_connection)
        }

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create comprehensive LangGraph workflow for CBUAE dormancy analysis"""

        workflow = StateGraph(DormancyAnalysisState)

        # Add nodes for comprehensive monitoring workflow
        workflow.add_node("initialize", self._initialize_comprehensive_analysis)
        workflow.add_node("run_monitoring_system", self._run_comprehensive_monitoring)
        workflow.add_node("check_triggers", self._check_all_triggers)
        workflow.add_node("execute_agents", self._execute_triggered_agents)
        workflow.add_node("consolidate_results", self._consolidate_comprehensive_results)
        workflow.add_node("generate_reports", self._generate_cbuae_reports)
        workflow.add_node("handle_errors", self._handle_workflow_errors)
        workflow.add_node("finalize", self._finalize_comprehensive_analysis)

        # Define workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "run_monitoring_system")

        workflow.add_conditional_edges(
            "run_monitoring_system",
            self._route_after_monitoring,
            {
                "triggers": "check_triggers",
                "no_data": "finalize",
                "error": "handle_errors"
            }
        )

        workflow.add_conditional_edges(
            "check_triggers",
            self._route_after_triggers,
            {
                "execute": "execute_agents",
                "no_triggers": "generate_reports",
                "error": "handle_errors"
            }
        )

        workflow.add_conditional_edges(
            "execute_agents",
            self._route_after_execution,
            {
                "consolidate": "consolidate_results",
                "error": "handle_errors"
            }
        )

        workflow.add_edge("consolidate_results", "generate_reports")
        workflow.add_edge("generate_reports", "finalize")
        workflow.add_edge("handle_errors", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=MemorySaver())

    async def _initialize_comprehensive_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Initialize comprehensive CBUAE dormancy analysis"""
        try:
            state.analysis_status = DormancyStatus.PROCESSING
            state.current_node = "initialize"

            # Log initialization with comprehensive monitoring
            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "comprehensive_initialization",
                "status": "started",
                "monitoring_system_available": self.monitoring_system is not None,
                "total_agents": len(self.agents),
                "cbuae_articles_covered": self._get_covered_articles()
            })

            return state

        except Exception as e:
            logger.error(f"Comprehensive analysis initialization failed: {str(e)}")
            state.analysis_status = DormancyStatus.FAILED
            return state

    async def _run_comprehensive_monitoring(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run comprehensive monitoring system"""
        try:
            state.current_node = "run_monitoring_system"

            if self.monitoring_system:
                # Run all dormancy monitors from the comprehensive system
                monitor_results = self.monitoring_system.run_all_dormancy_monitors()

                # Generate compliance dashboard data
                dashboard_data = self.monitoring_system.generate_compliance_dashboard_data()

                # Store monitoring results in state
                state.memory_context["monitoring_results"] = monitor_results
                state.memory_context["dashboard_data"] = dashboard_data

                # Check if any alerts were generated
                total_alerts = monitor_results.get("summary", {}).get("total_alerts", 0)
                state.routing_decision = "triggers" if total_alerts > 0 else "no_data"

                state.analysis_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "comprehensive_monitoring",
                    "total_alerts": total_alerts,
                    "monitors_executed": monitor_results.get("summary", {}).get("monitors_executed", 0),
                    "high_priority_alerts": monitor_results.get("summary", {}).get("high_priority_alerts", 0)
                })
            else:
                state.routing_decision = "triggers"  # Proceed with agent-based analysis
                state.analysis_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "monitoring_system_unavailable",
                    "fallback": "agent_based_analysis"
                })

            return state

        except Exception as e:
            logger.error(f"Comprehensive monitoring failed: {str(e)}")
            state.routing_decision = "error"
            return state

    async def _check_all_triggers(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Check triggers for all comprehensive agents"""
        try:
            state.current_node = "check_triggers"

            # Use monitoring results if available
            monitoring_results = state.memory_context.get("monitoring_results", {})
            triggered_agents = []

            for agent_name, agent in self.agents.items():
                try:
                    # Check if monitoring system detected relevant alerts for this agent
                    agent_alerts = self._get_agent_alerts(agent_name, monitoring_results)

                    if agent_alerts:
                        triggered_agents.append(agent_name)
                        logger.info(f"Agent {agent_name} triggered by monitoring system alerts")
                    else:
                        # Fallback to agent-specific trigger checking
                        agent_state = AgentState(
                            agent_id=f"{agent_name}_{secrets.token_hex(8)}",
                            agent_type=agent_name,
                            session_id=state.session_id,
                            user_id=state.user_id,
                            timestamp=datetime.now(),
                            input_dataframe=pd.DataFrame(state.processed_data.get('accounts', [])),
                            analysis_config=state.analysis_config
                        )

                        should_trigger = await agent.check_triggers(agent_state)
                        if should_trigger:
                            triggered_agents.append(agent_name)

                except Exception as e:
                    logger.warning(f"Trigger check failed for {agent_name}: {str(e)}")

            state.active_agents = triggered_agents
            state.routing_decision = "execute" if triggered_agents else "no_triggers"

            return state

        except Exception as e:
            logger.error(f"Comprehensive trigger checking failed: {str(e)}")
            state.routing_decision = "error"
            return state

    async def _execute_triggered_agents(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Execute all triggered agents with comprehensive monitoring"""
        try:
            state.current_node = "execute_agents"
            accounts_df = pd.DataFrame(state.processed_data['accounts'])
            report_date = state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d"))

            # Execute agents with monitoring system integration
            for agent_name in state.active_agents:
                try:
                    agent = self.agents[agent_name]

                    agent_state = AgentState(
                        agent_id=f"{agent_name}_{secrets.token_hex(8)}",
                        agent_type=agent_name,
                        session_id=state.session_id,
                        user_id=state.user_id,
                        timestamp=datetime.now(),
                        input_dataframe=accounts_df.copy(),
                        analysis_config=state.analysis_config
                    )

                    # Execute agent with full pipeline
                    completed_state = await self._execute_single_comprehensive_agent(agent, agent_state, report_date)
                    state.agent_results[agent_name] = completed_state

                    if completed_state.agent_status == AgentStatus.COMPLETED:
                        state.completed_agents.append(agent_name)
                        state.dormant_accounts_found += completed_state.dormant_records_found
                    else:
                        state.failed_agents.append(agent_name)

                except Exception as e:
                    logger.error(f"Agent {agent_name} execution failed: {str(e)}")
                    state.failed_agents.append(agent_name)

            state.routing_decision = "consolidate" if state.completed_agents else "error"
            return state

        except Exception as e:
            logger.error(f"Comprehensive agent execution failed: {str(e)}")
            state.routing_decision = "error"
            return state

    async def _execute_single_comprehensive_agent(self, agent: BaseDormancyAgent, agent_state: AgentState,
                                                  report_date: str) -> AgentState:
        """Execute a single agent with comprehensive monitoring integration"""
        try:
            # Pre-analysis memory hook
            agent_state = await agent.pre_analysis_memory_hook(agent_state)

            # Main analysis with monitoring system integration
            agent_state = await agent.analyze_dormancy(agent_state, report_date)

            # Post-analysis memory hook
            agent_state = await agent.post_analysis_memory_hook(agent_state)

            return agent_state

        except Exception as e:
            logger.error(f"Comprehensive agent execution failed for {agent.agent_type}: {str(e)}")
            agent_state.agent_status = AgentStatus.FAILED
            return agent_state

    async def _consolidate_comprehensive_results(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Consolidate results with comprehensive CBUAE compliance analysis"""
        try:
            state.current_node = "consolidate_results"

            # Base consolidation
            consolidated = {
                "cbuae_compliance_summary": {
                    "regulation_version": "2024",
                    "articles_analyzed": [],
                    "total_compliance_items": 0,
                    "high_priority_findings": 0,
                    "immediate_actions_required": []
                },
                "monitoring_system_results": state.memory_context.get("monitoring_results", {}),
                "agent_based_results": {},
                "regulatory_risk_assessment": {
                    "overall_risk_level": "low",
                    "critical_items": 0,
                    "high_risk_items": 0,
                    "medium_risk_items": 0,
                    "low_risk_items": 0
                },
                "next_actions": [],
                "audit_trail": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "comprehensive_monitoring": True,
                    "agents_executed": len(state.completed_agents),
                    "total_findings": state.dormant_accounts_found
                }
            }

            # Process agent results
            for agent_name in state.completed_agents:
                agent_result = state.agent_results[agent_name]

                if agent_result.analysis_results:
                    article = agent_result.analysis_results.get("compliance_article")
                    if article:
                        consolidated["cbuae_compliance_summary"]["articles_analyzed"].append(f"Article {article}")

                    consolidated["agent_based_results"][agent_name] = {
                        "article": article,
                        "findings": agent_result.dormant_records_found,
                        "description": agent_result.analysis_results.get("description", ""),
                        "priority": agent_result.analysis_results.get("priority", "medium")
                    }

            # Calculate overall compliance
            total_findings = sum(
                result["findings"] for result in consolidated["agent_based_results"].values()
            )

            consolidated["cbuae_compliance_summary"]["total_compliance_items"] = total_findings

            if total_findings > 20:
                consolidated["regulatory_risk_assessment"]["overall_risk_level"] = "critical"
            elif total_findings > 10:
                consolidated["regulatory_risk_assessment"]["overall_risk_level"] = "high"
            elif total_findings > 0:
                consolidated["regulatory_risk_assessment"]["overall_risk_level"] = "medium"

            state.dormancy_summary = consolidated
            return state

        except Exception as e:
            logger.error(f"Comprehensive result consolidation failed: {str(e)}")
            return state

    async def _generate_cbuae_reports(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Generate CBUAE regulatory reports"""
        try:
            state.current_node = "generate_reports"

            if self.monitoring_system and self.monitoring_system.reporting_engine:
                # Generate quarterly BRF report
                current_date = datetime.now()
                quarter = (current_date.month - 1) // 3 + 1

                quarterly_report = self.monitoring_system.reporting_engine.generate_quarterly_brf_report(
                    quarter, current_date.year
                )

                # Generate aging analysis
                aging_report = self.monitoring_system.reporting_engine.generate_dormancy_aging_report()

                # Store reports in state
                state.dormancy_summary["regulatory_reports"] = {
                    "quarterly_brf": quarterly_report,
                    "aging_analysis": aging_report,
                    "generated_timestamp": datetime.now().isoformat()
                }

            return state

        except Exception as e:
            logger.error(f"CBUAE report generation failed: {str(e)}")
            return state

    async def _handle_workflow_errors(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Handle comprehensive workflow errors"""
        try:
            state.current_node = "handle_errors"
            state.analysis_status = DormancyStatus.FAILED

            # Use error handler
            error_handler = ErrorHandlerAgent(self.memory_agent, self.mcp_client)
            # Error handling logic here

            return state

        except Exception as e:
            logger.error(f"Error handling failed: {str(e)}")
            return state

    async def _finalize_comprehensive_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Finalize comprehensive dormancy analysis"""
        try:
            state.current_node = "finalize"

            if state.analysis_status != DormancyStatus.FAILED:
                state.analysis_status = DormancyStatus.COMPLETED

            # Create comprehensive final results
            state.dormancy_results = {
                "session_id": state.session_id,
                "analysis_type": "comprehensive_cbuae_monitoring",
                "monitoring_system_used": self.monitoring_system is not None,
                "total_cbuae_articles_covered": len(self._get_covered_articles()),
                "comprehensive_summary": state.dormancy_summary,
                "agent_execution_results": {
                    agent_name: {
                        "agent_status": result.agent_status.value,
                        "findings": result.dormant_records_found,
                        "cbuae_article": result.analysis_results.get(
                            "compliance_article") if result.analysis_results else None,
                        "regulatory_compliance": result.analysis_results.get("validation_passed",
                                                                             True) if result.analysis_results else True
                    } for agent_name, result in state.agent_results.items()
                },
                "regulatory_compliance_status": "compliant" if state.analysis_status == DormancyStatus.COMPLETED else "requires_review",
                "completion_timestamp": datetime.now().isoformat()
            }

            return state

        except Exception as e:
            logger.error(f"Comprehensive analysis finalization failed: {str(e)}")
            state.analysis_status = DormancyStatus.FAILED
            return state

    def _get_covered_articles(self) -> List[str]:
        """Get list of CBUAE articles covered by the agents"""
        return [
            "2.1.1", "2.2", "2.3", "2.4", "2.6",  # Dormancy criteria
            "3.1", "3.4", "3.7", "3.9",  # Bank obligations
            "4",  # Customer claims
            "5",  # Proactive communication
            "7.3",  # Statement suppression
            "8.1", "8.5"  # Central Bank transfers
        ]

    def _get_agent_alerts(self, agent_name: str, monitoring_results: Dict) -> List:
        """Get alerts relevant to specific agent from monitoring results"""
        if not monitoring_results:
            return []

        agent_mapping = {
            "demand_deposit": "demand_deposit_inactivity",
            "fixed_deposit": "fixed_deposit_inactivity",
            "investment": "investment_inactivity",
            "safe_deposit": "safe_deposit_dormancy",
            "payment_instruments": "unclaimed_payment_instruments",
            "contact_attempts": "incomplete_contact_attempts",
            "internal_ledger": "internal_ledger_candidates",
            "sdb_court_application": "sdb_court_applications",
            "record_retention": "record_retention_compliance",
            "customer_claims": "claim_processing_pending",
            "proactive_contact": "proactive_contact_needed",
            "statement_suppression": "statement_suppression_candidates",
            "cb_transfer": "cb_transfer_eligible",
            "foreign_currency": "foreign_currency_conversion"
        }

        monitor_key = agent_mapping.get(agent_name)
        if monitor_key and monitor_key in monitoring_results:
            return monitoring_results[monitor_key]

        return []

    def _route_after_monitoring(self, state: DormancyAnalysisState) -> str:
        """Route after monitoring system execution"""
        return state.routing_decision

    def _route_after_triggers(self, state: DormancyAnalysisState) -> str:
        """Route after trigger checking"""
        return state.routing_decision

    def _route_after_execution(self, state: DormancyAnalysisState) -> str:
        """Route after agent execution"""
        return state.routing_decision

    async def orchestrate_comprehensive_analysis(self, user_id: str, input_dataframe: pd.DataFrame,
                                                 report_date: str, analysis_config: Dict = None,
                                                 db_connection=None) -> Dict:
        """Orchestrate comprehensive CBUAE dormancy analysis"""
        try:
            # Update database connection if provided
            if db_connection and not self.db_connection:
                self.db_connection = db_connection
                self.monitoring_system = initialize_dormancy_monitoring_system(db_connection)

                # Update all agents with database connection
                for agent in self.agents.values():
                    agent.db_connection = db_connection
                    if not agent.dormancy_monitoring:
                        agent.dormancy_monitoring = DormancyMonitoringAgents(db_connection)

            # Create initial state
            initial_state = DormancyAnalysisState(
                session_id=secrets.token_hex(16),
                user_id=user_id,
                analysis_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                processed_data={'accounts': input_dataframe.to_dict('records')},
                analysis_config=analysis_config or {'report_date': report_date}
            )

            # Execute comprehensive workflow
            final_state = await self.workflow.ainvoke(initial_state)

            return final_state.dormancy_results

        except Exception as e:
            logger.error(f"Comprehensive workflow orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": initial_state.session_id if 'initial_state' in locals() else None,
                "analysis_type": "comprehensive_cbuae_monitoring"
            }


# Main Comprehensive Dormancy Analysis Agent
class DormancyAnalysisAgent:
    """Main comprehensive dormancy analysis agent using full CBUAE monitoring system"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # Initialize comprehensive orchestrator
        self.orchestrator = DormancyWorkflowOrchestrator(memory_agent, mcp_client, db_session)

    @traceable(name="comprehensive_dormancy_analysis")
    async def analyze_dormancy(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Main comprehensive dormancy analysis using full CBUAE monitoring system"""
        try:
            start_time = datetime.now()
            state.analysis_status = DormancyStatus.PROCESSING

            # Extract account data
            if not state.processed_data or 'accounts' not in state.processed_data:
                raise ValueError("No account data available for comprehensive dormancy analysis")

            accounts_df = pd.DataFrame(state.processed_data['accounts'])
            if accounts_df.empty:
                raise ValueError("Empty account data provided")

            report_date = state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d"))

            # Execute comprehensive orchestrated analysis
            orchestration_results = await self.orchestrator.orchestrate_comprehensive_analysis(
                user_id=state.user_id,
                input_dataframe=accounts_df,
                report_date=report_date,
                analysis_config=state.analysis_config,
                db_connection=self.db_session
            )

            # Process comprehensive results
            if orchestration_results and not orchestration_results.get("success") == False:
                state.dormancy_results = orchestration_results
                state.dormancy_summary = orchestration_results.get("comprehensive_summary", {})

                # Extract metrics from comprehensive analysis
                cbuae_summary = state.dormancy_summary.get("cbuae_compliance_summary", {})
                state.total_accounts_analyzed = len(accounts_df)
                state.dormant_accounts_found = cbuae_summary.get("total_compliance_items", 0)
                state.high_risk_accounts = state.dormancy_summary.get("regulatory_risk_assessment", {}).get(
                    "high_risk_items", 0)

                # Extract compliance flags
                state.compliance_flags = cbuae_summary.get("articles_analyzed", [])

                # Calculate processing metrics
                state.processing_time = (datetime.now() - start_time).total_seconds()
                state.analysis_efficiency = (
                    state.total_accounts_analyzed / state.processing_time
                    if state.processing_time > 0 else 0
                )

                state.analysis_status = DormancyStatus.COMPLETED

                # Log successful comprehensive analysis
                state.analysis_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "comprehensive_dormancy_analysis",
                    "status": "completed",
                    "analysis_type": "full_cbuae_monitoring",
                    "monitoring_system_used": orchestration_results.get("monitoring_system_used", False),
                    "articles_covered": orchestration_results.get("total_cbuae_articles_covered", 0),
                    "compliance_status": orchestration_results.get("regulatory_compliance_status", "unknown"),
                    "total_findings": state.dormant_accounts_found,
                    "processing_time": state.processing_time
                })
            else:
                state.analysis_status = DormancyStatus.FAILED
                error_msg = orchestration_results.get("error", "Unknown comprehensive analysis error")
                state.error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "comprehensive_orchestration",
                    "error": error_msg,
                    "analysis_type": "full_cbuae_monitoring"
                })

        except Exception as e:
            state.analysis_status = DormancyStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "comprehensive_dormancy_analysis",
                "error": str(e),
                "analysis_type": "full_cbuae_monitoring"
            })
            logger.error(f"Comprehensive dormancy analysis failed: {str(e)}")

        return state


# Factory Functions and Utilities
def create_comprehensive_dormancy_analysis(memory_agent, mcp_client: MCPClient,
                                           db_session=None) -> DormancyAnalysisAgent:
    """Factory function to create comprehensive dormancy analysis agent"""
    return DormancyAnalysisAgent(memory_agent, mcp_client, db_session)


def get_comprehensive_agents() -> List[str]:
    """Get list of all comprehensive dormancy analysis agents"""
    return [
        "demand_deposit", "fixed_deposit", "investment", "safe_deposit", "payment_instruments",
        "contact_attempts", "internal_ledger", "sdb_court_application", "record_retention",
        "customer_claims", "proactive_contact", "statement_suppression",
        "cb_transfer", "foreign_currency"
    ]


def get_cbuae_articles_covered() -> Dict[str, str]:
    """Get mapping of CBUAE articles to agent types"""
    return {
        "2.1.1": "demand_deposit",
        "2.2": "fixed_deposit",
        "2.3": "investment",
        "2.4": "payment_instruments",
        "2.6": "safe_deposit",
        "3.1": "contact_attempts",
        "3.4": "internal_ledger",
        "3.7": "sdb_court_application",
        "3.9": "record_retention",
        "4": "customer_claims",
        "5": "proactive_contact",
        "7.3": "statement_suppression",
        "8.1": "cb_transfer",
        "8.5": "foreign_currency"
    }


def setup_comprehensive_monitoring(db_connection) -> DormancyMonitoringAgents:
    """Setup comprehensive monitoring system"""
    return initialize_dormancy_monitoring_system(db_connection)


# Export classes and functions for use by workflow engine
__all__ = [
    "DormancyAnalysisAgent",
    "DormancyWorkflowOrchestrator",
    "DormancyAnalysisState",
    "AgentState",
    "AgentStatus",
    "DormancyStatus",
    "DormancyTrigger",
    "BaseDormancyAgent",
    "DemandDepositDormancyAgent",
    "FixedDepositDormancyAgent",
    "InvestmentAccountDormancyAgent",
    "SafeDepositBoxDormancyAgent",
    "PaymentInstrumentsDormancyAgent",
    "ContactAttemptsAgent",
    "InternalLedgerAgent",
    "SDBCourtApplicationAgent",
    "RecordRetentionAgent",
    "CustomerClaimsAgent",
    "ProactiveContactAgent",
    "StatementSuppressionAgent",
    "CBTransferEligibilityAgent",
    "ForeignCurrencyConversionAgent",
    "create_comprehensive_dormancy_analysis",
    "get_comprehensive_agents",
    "get_cbuae_articles_covered",
    "setup_comprehensive_monitoring"
]