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

# Import dormancy analysis functions
from agents.dormant import (
    check_safe_deposit_dormancy,
    check_investment_inactivity,
    check_fixed_deposit_inactivity,
    check_demand_deposit_inactivity,
    check_unclaimed_payment_instruments,
    check_eligible_for_cb_transfer,
    check_art3_process_needed,
    check_contact_attempts_needed,
    check_high_value_dormant_accounts,
    check_dormant_to_active_transitions,
    run_all_dormant_identification_checks
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


# Base Dormancy Agent Class with Memory Hooks
class BaseDormancyAgent:
    """Base class for all dormancy analysis agents with memory hooks"""

    def __init__(self, agent_type: str, memory_agent, mcp_client: MCPClient):
        self.agent_type = agent_type
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client

        # Initialize error handler
        self.error_handler = ErrorHandlerAgent(memory_agent, mcp_client)

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # Default regulatory parameters
        self.default_params = {
            "standard_inactivity_years": 3,
            "payment_instrument_unclaimed_years": 1,
            "sdb_unpaid_fees_years": 3,
            "eligibility_for_cb_transfer_years": 5,
            "high_value_threshold_aed": 25000
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
        """Check if agent should be triggered"""
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
        """Enhanced pre-analysis memory hook with agent-specific patterns"""
        try:
            state.agent_status = AgentStatus.MEMORY_LOADING

            # Create memory context
            memory_context = await self.memory_agent.create_memory_context(
                user_id=state.user_id,
                session_id=state.session_id,
                agent_name=self.agent_type
            )

            # Retrieve agent-specific patterns
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

            # Retrieve historical performance data
            historical_data = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": f"{self.agent_type}_performance",
                    "user_id": state.user_id,
                    "content_type": "performance_metrics"
                },
                context=memory_context
            )

            if historical_data.get("success") and historical_data.get("data"):
                state.pre_hook_memory["historical_performance"] = historical_data["data"]

            # Retrieve user preferences
            user_preferences = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.USER_PROFILE.value,
                filter_criteria={
                    "type": "agent_preferences",
                    "user_id": state.user_id,
                    "agent_type": self.agent_type
                },
                context=memory_context
            )

            if user_preferences.get("success") and user_preferences.get("data"):
                state.pre_hook_memory["user_preferences"] = user_preferences["data"]

            # Load regulatory benchmarks
            regulatory_benchmarks = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE.value,
                filter_criteria={
                    "type": "regulatory_benchmarks",
                    "content_type": "compliance_standards",
                    "agent_type": self.agent_type
                },
                context=memory_context
            )

            if regulatory_benchmarks.get("success") and regulatory_benchmarks.get("data"):
                state.pre_hook_memory["regulatory_benchmarks"] = regulatory_benchmarks["data"]

            # Store session input data
            if state.input_dataframe is not None:
                df_summary = {
                    "total_records": len(state.input_dataframe),
                    "columns": list(state.input_dataframe.columns),
                    "data_types": state.input_dataframe.dtypes.to_dict(),
                    "memory_usage": state.input_dataframe.memory_usage(deep=True).sum()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.SESSION.value,
                    data={
                        "type": f"{self.agent_type}_input_summary",
                        "agent_id": state.agent_id,
                        "session_id": state.session_id,
                        "summary": df_summary,
                        "timestamp": datetime.now().isoformat()
                    },
                    context=memory_context,
                    content_type="input_data",
                    tags=[self.agent_type, "input", "session"]
                )

            # Log pre-analysis execution
            state.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_analysis_memory_hook",
                "agent_type": self.agent_type,
                "action": "memory_retrieval_completed",
                "patterns_retrieved": len(state.retrieved_patterns.get("agent_patterns", [])),
                "historical_data_loaded": bool(state.pre_hook_memory.get("historical_performance")),
                "user_preferences_loaded": bool(state.pre_hook_memory.get("user_preferences")),
                "regulatory_benchmarks_loaded": bool(state.pre_hook_memory.get("regulatory_benchmarks"))
            })

            state.agent_status = AgentStatus.PROCESSING

        except Exception as e:
            logger.error(f"{self.agent_type} pre-analysis memory hook failed: {str(e)}")
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "pre_analysis_memory_hook")

        return state

    @traceable(name="post_analysis_memory_hook")
    async def post_analysis_memory_hook(self, state: AgentState) -> AgentState:
        """Enhanced post-analysis memory hook with pattern storage"""
        try:
            state.agent_status = AgentStatus.MEMORY_STORING

            # Create memory context
            memory_context = await self.memory_agent.create_memory_context(
                user_id=state.user_id,
                session_id=state.session_id,
                agent_name=self.agent_type
            )

            # Store session results
            if state.analysis_results:
                session_results = {
                    "type": f"{self.agent_type}_session_results",
                    "agent_id": state.agent_id,
                    "session_id": state.session_id,
                    "results": {
                        "records_processed": state.records_processed,
                        "dormant_found": state.dormant_records_found,
                        "processing_time": state.processing_time,
                        "analysis_results": state.analysis_results,
                        "performance_metrics": state.performance_metrics
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.SESSION.value,
                    data=session_results,
                    context=memory_context,
                    content_type="session_results",
                    priority=MemoryPriority.HIGH,
                    tags=[self.agent_type, "results", "session"]
                )

            # Store pattern analysis in knowledge base
            if state.pattern_analysis and state.agent_status == AgentStatus.COMPLETED:
                pattern_knowledge = {
                    "type": f"{self.agent_type}_patterns",
                    "user_id": state.user_id,
                    "agent_type": self.agent_type,
                    "patterns": state.pattern_analysis,
                    "effectiveness_metrics": {
                        "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0,
                        "processing_efficiency": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
                        "accuracy_score": state.performance_metrics.get("accuracy_score", 0)
                    },
                    "regulatory_compliance": {
                        "compliance_article": state.analysis_results.get("compliance_article"),
                        "validation_passed": state.analysis_results.get("validation_passed", True)
                    },
                    "timestamp": datetime.now().isoformat()
                }

                pattern_result = await self.memory_agent.store_memory(
                    bucket=MemoryBucket.KNOWLEDGE.value,
                    data=pattern_knowledge,
                    context=memory_context,
                    content_type="dormancy_patterns",
                    priority=MemoryPriority.HIGH,
                    tags=[self.agent_type, "patterns", "knowledge"],
                    encrypt_sensitive=True
                )

                if pattern_result.get("success"):
                    state.stored_patterns = pattern_knowledge

            # Store performance metrics for future optimization
            if state.performance_metrics and state.agent_status == AgentStatus.COMPLETED:
                performance_data = {
                    "type": f"{self.agent_type}_performance",
                    "user_id": state.user_id,
                    "agent_type": self.agent_type,
                    "metrics": state.performance_metrics,
                    "optimization_insights": {
                        "processing_speed": state.performance_metrics.get("records_per_second", 0),
                        "memory_efficiency": state.performance_metrics.get("memory_usage", 0),
                        "accuracy": state.performance_metrics.get("accuracy_score", 0)
                    },
                    "benchmark_comparison": self._compare_with_benchmarks(state),
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.KNOWLEDGE.value,
                    data=performance_data,
                    context=memory_context,
                    content_type="performance_metrics",
                    priority=MemoryPriority.MEDIUM,
                    tags=[self.agent_type, "performance", "optimization"]
                )

            # Store processed data summary for audit
            if state.processed_dataframe is not None and not state.processed_dataframe.empty:
                audit_summary = {
                    "type": f"{self.agent_type}_audit_trail",
                    "agent_id": state.agent_id,
                    "session_id": state.session_id,
                    "processing_summary": {
                        "input_records": len(state.input_dataframe) if state.input_dataframe is not None else 0,
                        "output_records": len(state.processed_dataframe),
                        "dormant_identified": state.dormant_records_found,
                        "processing_time": state.processing_time,
                        "data_quality_score": state.performance_metrics.get("data_quality_score", 0)
                    },
                    "regulatory_compliance": {
                        "article": state.analysis_results.get("compliance_article"),
                        "validation_status": "passed" if state.analysis_results.get("validation_passed",
                                                                                    True) else "failed"
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.AUDIT.value,
                    data=audit_summary,
                    context=memory_context,
                    content_type="audit_trail",
                    priority=MemoryPriority.CRITICAL,
                    tags=[self.agent_type, "audit", "compliance"]
                )

            # Log post-analysis execution
            state.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_analysis_memory_hook",
                "agent_type": self.agent_type,
                "action": "memory_storage_completed",
                "session_results_stored": bool(state.analysis_results),
                "patterns_stored": bool(state.stored_patterns),
                "performance_data_stored": bool(state.performance_metrics),
                "audit_trail_created": bool(state.processed_dataframe is not None)
            })

        except Exception as e:
            logger.error(f"{self.agent_type} post-analysis memory hook failed: {str(e)}")
            await self._handle_error(state, e, "post_analysis_memory_hook")

        return state

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

    def _compare_with_benchmarks(self, state: AgentState) -> Dict:
        """Compare current performance with historical benchmarks"""
        benchmarks = state.pre_hook_memory.get("historical_performance", [])

        if not benchmarks:
            return {"status": "no_benchmarks_available"}

        current_metrics = state.performance_metrics
        if not current_metrics:
            return {"status": "no_current_metrics"}

        # Calculate performance comparison
        comparison = {
            "processing_speed_improvement": 0.0,
            "accuracy_improvement": 0.0,
            "efficiency_improvement": 0.0,
            "overall_performance": "stable"
        }

        # Implement benchmark comparison logic here
        # This is a simplified version
        for benchmark in benchmarks[-5:]:  # Last 5 runs
            if isinstance(benchmark, dict) and "data" in benchmark:
                bench_data = benchmark["data"]
                if isinstance(bench_data, dict) and "metrics" in bench_data:
                    bench_metrics = bench_data["metrics"]

                    # Compare key metrics
                    current_speed = current_metrics.get("records_per_second", 0)
                    bench_speed = bench_metrics.get("records_per_second", 0)

                    if bench_speed > 0:
                        comparison["processing_speed_improvement"] = (current_speed - bench_speed) / bench_speed

        return comparison

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Base pattern analysis - to be overridden by specific agents"""
        return {
            "pattern_type": "base",
            "insights": [],
            "recommendations": [],
            "effectiveness_metrics": {}
        }


# Demand Deposit Dormancy Agent (Article 2.1.1)
class DemandDepositDormancyAgent(BaseDormancyAgent):
    """Specialized agent for demand deposit dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("demand_deposit_dormancy", memory_agent, mcp_client)

        # Define specific trigger conditions
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
        try:
            # Check if there are any demand deposit accounts
            df = state.input_dataframe
            demand_deposit_accounts = df[
                df['Account_Type'].astype(str).str.contains("Current|Saving|Call", case=False, na=False)
            ]

            if demand_deposit_accounts.empty:
                logger.info("No demand deposit accounts found, skipping agent")
                return False

            # Check if there are accounts with activity within the dormancy threshold
            report_date = datetime.strptime(
                state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d")),
                "%Y-%m-%d"
            )
            threshold_date = report_date - timedelta(days=3 * 365)  # 3 years

            if 'Date_Last_Cust_Initiated_Activity' in df.columns:
                df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df['Date_Last_Cust_Initiated_Activity'], errors='coerce'
                )

                potentially_dormant = demand_deposit_accounts[
                    demand_deposit_accounts['Date_Last_Cust_Initiated_Activity'] < threshold_date
                    ]

                if potentially_dormant.empty:
                    logger.info("No potentially dormant demand deposit accounts found")
                    return False

            state.triggered_by = DormancyTrigger.STANDARD_INACTIVITY
            return True

        except Exception as e:
            logger.error(f"Demand deposit trigger check failed: {str(e)}")
            return False

    @traceable(name="demand_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze demand deposit dormancy with enhanced pattern recognition"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for demand deposit analysis")

            # Execute CBUAE Article 2.1.1 analysis
            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_demand_deposit_inactivity(
                state.input_dataframe, report_date_obj
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count

            # Store analysis results
            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "compliance_article": "2.1.1",
                "analysis_date": report_date,
                "validation_passed": True,
                "key_findings": []
            }

            # Enhanced pattern analysis with historical data
            state.agent_status = AgentStatus.ANALYZING_PATTERNS
            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)

            # Performance metrics calculation
            processing_time = (datetime.now() - start_time).total_seconds()
            state.processing_time = processing_time

            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "records_per_second": state.records_processed / processing_time if processing_time > 0 else 0,
                "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0,
                "analysis_efficiency": count / processing_time if processing_time > 0 else 0,
                "memory_usage": state.input_dataframe.memory_usage(
                    deep=True).sum() if state.input_dataframe is not None else 0,
                "data_quality_score": self._calculate_data_quality_score(state.input_dataframe),
                "accuracy_score": self._calculate_accuracy_score(processed_df, state.analysis_results)
            }

            # Call MCP tool for additional processing
            if self.mcp_client:
                mcp_result = await self.mcp_client.call_tool("analyze_account_dormancy", {
                    "accounts": processed_df.to_dict('records') if not processed_df.empty else [],
                    "report_date": report_date,
                    "regulatory_params": {
                        "article": "2.1.1",
                        "type": "demand_deposit",
                        "agent_type": self.agent_type
                    },
                    "performance_metrics": state.performance_metrics
                })

                if mcp_result.get("success"):
                    state.analysis_results["mcp_insights"] = mcp_result.get("data", {})

            state.agent_status = AgentStatus.COMPLETED

            # Generate key findings
            if count > 0:
                state.analysis_results["key_findings"].extend([
                    f"Identified {count} demand deposit accounts meeting dormancy criteria (Article 2.1.1)",
                    f"Dormancy rate: {(count / state.records_processed) * 100:.1f}%",
                    f"Average processing time per account: {processing_time / state.records_processed:.4f} seconds",
                    f"Data quality score: {state.performance_metrics['data_quality_score']:.2f}/1.0"
                ])

            # Log successful analysis
            state.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "demand_deposit_analysis",
                "agent_type": self.agent_type,
                "status": "completed",
                "records_processed": state.records_processed,
                "dormant_found": state.dormant_records_found,
                "processing_time": processing_time,
                "triggered_by": state.triggered_by.value if state.triggered_by else None
            })

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "demand_deposit_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze demand deposit specific patterns with historical context"""
        if df.empty:
            return {"pattern_type": "demand_deposit", "insights": [], "recommendations": []}

        patterns = {
            "pattern_type": "demand_deposit",
            "account_type_distribution": {},
            "balance_patterns": {},
            "activity_patterns": {},
            "communication_patterns": {},
            "risk_indicators": {},
            "insights": [],
            "recommendations": [],
            "effectiveness_metrics": {}
        }

        try:
            # Account type analysis
            if 'Account_Type' in df.columns:
                type_dist = df['Account_Type'].value_counts().to_dict()
                patterns["account_type_distribution"] = type_dist

                most_dormant_type = max(type_dist, key=type_dist.get) if type_dist else "Unknown"
                patterns["insights"].append(
                    f"Most dormant account type: {most_dormant_type} ({type_dist.get(most_dormant_type, 0)} accounts)")

            # Balance pattern analysis
            if 'Current_Balance' in df.columns:
                balances = pd.to_numeric(df['Current_Balance'], errors='coerce').dropna()
                if not balances.empty:
                    patterns["balance_patterns"] = {
                        "average_balance": float(balances.mean()),
                        "median_balance": float(balances.median()),
                        "high_balance_accounts": int((balances > 25000).sum()),
                        "low_balance_accounts": int((balances < 1000).sum()),
                        "zero_balance_accounts": int((balances == 0).sum()),
                        "total_dormant_value": float(balances.sum())
                    }

                    if patterns["balance_patterns"]["high_balance_accounts"] > 0:
                        patterns["insights"].append(
                            f"Found {patterns['balance_patterns']['high_balance_accounts']} high-value dormant accounts (>AED 25,000)")
                        patterns["recommendations"].append(
                            "Prioritize immediate reactivation of high-balance dormant accounts")

                    if patterns["balance_patterns"]["zero_balance_accounts"] > len(balances) * 0.3:
                        patterns["insights"].append("High percentage of zero-balance dormant accounts detected")
                        patterns["recommendations"].append(
                            "Review account closure procedures for zero-balance dormant accounts")

            # Activity pattern analysis
            if 'Date_Last_Cust_Initiated_Activity' in df.columns:
                activity_dates = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'], errors='coerce').dropna()
                if not activity_dates.empty:
                    now = datetime.now()
                    inactivity_periods = (now - activity_dates).dt.days

                    patterns["activity_patterns"] = {
                        "average_inactivity_days": float(inactivity_periods.mean()),
                        "max_inactivity_days": int(inactivity_periods.max()),
                        "min_inactivity_days": int(inactivity_periods.min()),
                        "accounts_inactive_3_to_4_years": int(
                            ((inactivity_periods >= 1095) & (inactivity_periods < 1460)).sum()),
                        "accounts_inactive_4_to_5_years": int(
                            ((inactivity_periods >= 1460) & (inactivity_periods < 1825)).sum()),
                        "accounts_inactive_over_5_years": int((inactivity_periods >= 1825).sum())
                    }

                    if patterns["activity_patterns"]["accounts_inactive_over_5_years"] > 0:
                        patterns["insights"].append(
                            f"{patterns['activity_patterns']['accounts_inactive_over_5_years']} accounts inactive for over 5 years - eligible for CB transfer")
                        patterns["recommendations"].append(
                            "Initiate Central Bank transfer process for accounts inactive >5 years")

            # Communication pattern analysis
            if 'Date_Last_Customer_Communication_Any_Type' in df.columns:
                comm_dates = pd.to_datetime(df['Date_Last_Customer_Communication_Any_Type'], errors='coerce').dropna()
                if not comm_dates.empty:
                    comm_gaps = (datetime.now() - comm_dates).dt.days

                    patterns["communication_patterns"] = {
                        "average_communication_gap": float(comm_gaps.mean()),
                        "no_communication_3_years": int((comm_gaps >= 1095).sum()),
                        "no_communication_5_years": int((comm_gaps >= 1825).sum())
                    }

            # Risk indicator analysis
            if 'Customer_Has_Active_Liability_Account' in df.columns:
                liability_analysis = df['Customer_Has_Active_Liability_Account'].value_counts().to_dict()
                patterns["risk_indicators"] = {
                    "customers_with_active_liability": liability_analysis.get('yes', 0) + liability_analysis.get('true',
                                                                                                                 0) + liability_analysis.get(
                        '1', 0),
                    "customers_without_liability": len(df) - (
                                liability_analysis.get('yes', 0) + liability_analysis.get('true',
                                                                                          0) + liability_analysis.get(
                            '1', 0))
                }

                if patterns["risk_indicators"]["customers_without_liability"] > 0:
                    patterns["insights"].append(
                        f"{patterns['risk_indicators']['customers_without_liability']} dormant customers have no active liability accounts")

            # Effectiveness metrics
            patterns["effectiveness_metrics"] = {
                "detection_accuracy": 1.0,  # Assuming regulatory function is accurate
                "false_positive_rate": 0.0,  # Would need historical validation data
                "processing_efficiency": len(df) / analysis_results.get("processing_time", 1) if analysis_results.get(
                    "processing_time") else 0
            }

            # Generate strategic recommendations
            if len(patterns["insights"]) > 3:
                patterns["recommendations"].append(
                    "Implement targeted reactivation campaign based on identified patterns")

            if patterns["balance_patterns"].get("total_dormant_value", 0) > 1000000:  # > 1M AED
                patterns["recommendations"].append("Escalate high-value dormant portfolio to senior management")

        except Exception as e:
            patterns["insights"].append(f"Pattern analysis error: {str(e)}")
            logger.error(f"Demand deposit pattern analysis error: {str(e)}")

        return patterns

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score for input dataframe"""
        if df.empty:
            return 0.0

        total_cells = df.size
        non_null_cells = df.count().sum()
        completeness_score = non_null_cells / total_cells if total_cells > 0 else 0

        # Check for required columns presence
        required_cols = self.trigger_conditions.get("required_columns", [])
        present_cols = [col for col in required_cols if col in df.columns]
        column_completeness = len(present_cols) / len(required_cols) if required_cols else 1.0

        # Simple data quality score combining completeness factors
        return (completeness_score + column_completeness) / 2

    def _calculate_accuracy_score(self, processed_df: pd.DataFrame, analysis_results: Dict) -> float:
        """Calculate accuracy score based on analysis results"""
        # This would typically involve validation against known ground truth
        # For now, return a score based on data consistency
        if processed_df.empty:
            return 1.0  # Perfect accuracy if no false positives

        # Check for data consistency in results
        expected_count = analysis_results.get("count", 0)
        actual_count = len(processed_df)

        if expected_count == actual_count:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(expected_count - actual_count) / max(expected_count, actual_count, 1))


# Fixed Deposit Dormancy Agent (Article 2.2)
class FixedDepositDormancyAgent(BaseDormancyAgent):
    """Specialized agent for fixed deposit dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("fixed_deposit_dormancy", memory_agent, mcp_client)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Account_Type", "FTD_Maturity_Date", "FTD_Auto_Renewal",
                "Date_Last_FTD_Renewal_Claim_Request", "Date_Last_Customer_Communication_Any_Type"
            ],
            "business_rules": ["account_type_contains_fixed_term", "maturity_date_passed"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check fixed deposit specific triggers"""
        try:
            df = state.input_dataframe
            fixed_deposit_accounts = df[
                df['Account_Type'].astype(str).str.contains("Fixed|Term", case=False, na=False)
            ]

            if fixed_deposit_accounts.empty:
                return False

            # Check for matured deposits
            if 'FTD_Maturity_Date' in df.columns:
                df['FTD_Maturity_Date'] = pd.to_datetime(df['FTD_Maturity_Date'], errors='coerce')
                report_date = datetime.strptime(
                    state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d")),
                    "%Y-%m-%d"
                )

                matured_deposits = fixed_deposit_accounts[
                    fixed_deposit_accounts['FTD_Maturity_Date'] < report_date
                    ]

                if matured_deposits.empty:
                    return False

            state.triggered_by = DormancyTrigger.FIXED_DEPOSIT_MATURITY
            return True

        except Exception as e:
            logger.error(f"Fixed deposit trigger check failed: {str(e)}")
            return False

    @traceable(name="fixed_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze fixed deposit dormancy"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for fixed deposit analysis")

            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_fixed_deposit_inactivity(
                state.input_dataframe, report_date_obj
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count

            processing_time = (datetime.now() - start_time).total_seconds()
            state.processing_time = processing_time

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "compliance_article": "2.2",
                "analysis_date": report_date,
                "validation_passed": True,
                "key_findings": []
            }

            # Pattern analysis
            state.agent_status = AgentStatus.ANALYZING_PATTERNS
            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)

            # Performance metrics
            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "records_per_second": state.records_processed / processing_time if processing_time > 0 else 0,
                "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0,
                "data_quality_score": self._calculate_data_quality_score(state.input_dataframe),
                "accuracy_score": self._calculate_accuracy_score(processed_df, state.analysis_results)
            }

            # MCP tool call
            if self.mcp_client:
                mcp_result = await self.mcp_client.call_tool("analyze_account_dormancy", {
                    "accounts": processed_df.to_dict('records') if not processed_df.empty else [],
                    "report_date": report_date,
                    "regulatory_params": {"article": "2.2", "type": "fixed_deposit"}
                })

                if mcp_result.get("success"):
                    state.analysis_results["mcp_insights"] = mcp_result.get("data", {})

            state.agent_status = AgentStatus.COMPLETED

            if count > 0:
                state.analysis_results["key_findings"].extend([
                    f"Identified {count} fixed deposit accounts meeting dormancy criteria (Article 2.2)",
                    f"Dormancy rate: {(count / state.records_processed) * 100:.1f}%"
                ])

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "fixed_deposit_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze fixed deposit specific patterns"""
        if df.empty:
            return {"pattern_type": "fixed_deposit", "insights": [], "recommendations": []}

        patterns = {
            "pattern_type": "fixed_deposit",
            "maturity_patterns": {},
            "auto_renewal_patterns": {},
            "insights": [],
            "recommendations": []
        }

        try:
            # Auto-renewal analysis
            if 'FTD_Auto_Renewal' in df.columns:
                auto_renewal_dist = df['FTD_Auto_Renewal'].value_counts().to_dict()
                patterns["auto_renewal_patterns"] = auto_renewal_dist

                auto_renewal_count = auto_renewal_dist.get('yes', 0) + auto_renewal_dist.get('true', 0)
                patterns["insights"].append(f"Auto-renewal accounts: {auto_renewal_count}")

            # Maturity date analysis
            if 'FTD_Maturity_Date' in df.columns:
                maturity_dates = pd.to_datetime(df['FTD_Maturity_Date'], errors='coerce').dropna()
                if not maturity_dates.empty:
                    now = datetime.now()
                    overdue_periods = (now - maturity_dates).dt.days

                    patterns["maturity_patterns"] = {
                        "average_overdue_days": float(overdue_periods.mean()),
                        "max_overdue_days": int(overdue_periods.max()),
                        "accounts_overdue_3_to_5_years": int(
                            ((overdue_periods >= 1095) & (overdue_periods < 1825)).sum()),
                        "accounts_overdue_over_5_years": int((overdue_periods >= 1825).sum())
                    }

        except Exception as e:
            patterns["insights"].append(f"Pattern analysis error: {str(e)}")

        return patterns

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if df.empty:
            return 0.0

        required_cols = self.trigger_conditions.get("required_columns", [])
        present_cols = [col for col in required_cols if col in df.columns]
        return len(present_cols) / len(required_cols) if required_cols else 1.0

    def _calculate_accuracy_score(self, processed_df: pd.DataFrame, analysis_results: Dict) -> float:
        """Calculate accuracy score"""
        expected_count = analysis_results.get("count", 0)
        actual_count = len(processed_df)

        if expected_count == actual_count:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(expected_count - actual_count) / max(expected_count, actual_count, 1))


# Investment Account Dormancy Agent (Article 2.3)
class InvestmentAccountDormancyAgent(BaseDormancyAgent):
    """Specialized agent for investment account dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("investment_dormancy", memory_agent, mcp_client)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Account_Type", "Inv_Maturity_Redemption_Date",
                "Date_Last_Customer_Communication_Any_Type"
            ],
            "business_rules": ["account_type_contains_investment", "maturity_redemption_passed"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check investment account specific triggers"""
        try:
            df = state.input_dataframe
            investment_accounts = df[
                df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)
            ]

            if investment_accounts.empty:
                return False

            state.triggered_by = DormancyTrigger.INVESTMENT_MATURITY
            return True

        except Exception as e:
            logger.error(f"Investment account trigger check failed: {str(e)}")
            return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze investment account dormancy"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for investment analysis")

            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_investment_inactivity(
                state.input_dataframe, report_date_obj
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count
            state.processing_time = (datetime.now() - start_time).total_seconds()

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "compliance_article": "2.3",
                "analysis_date": report_date,
                "validation_passed": True
            }

            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)
            state.performance_metrics = self._calculate_performance_metrics(state)
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "investment_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze investment account patterns"""
        return {"pattern_type": "investment", "insights": [], "recommendations": []}

    def _calculate_performance_metrics(self, state: AgentState) -> Dict:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": state.processing_time,
            "records_per_second": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
            "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
        }


# Safe Deposit Box Dormancy Agent (Article 2.6)
class SafeDepositBoxDormancyAgent(BaseDormancyAgent):
    """Specialized agent for safe deposit box dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("safe_deposit_dormancy", memory_agent, mcp_client)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Account_Type", "SDB_Charges_Outstanding",
                "Date_SDB_Charges_Became_Outstanding", "SDB_Tenant_Communication_Received"
            ],
            "business_rules": ["account_type_safe_deposit_box", "charges_outstanding"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check safe deposit box specific triggers"""
        try:
            df = state.input_dataframe
            sdb_accounts = df[
                df['Account_Type'].astype(str).str.lower() == "safe_deposit_box"
                ]

            if sdb_accounts.empty:
                return False

            state.triggered_by = DormancyTrigger.SDB_UNPAID_FEES
            return True

        except Exception as e:
            logger.error(f"Safe deposit box trigger check failed: {str(e)}")
            return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze safe deposit box dormancy"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for safe deposit box analysis")

            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_safe_deposit_dormancy(
                state.input_dataframe, report_date_obj
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count
            state.processing_time = (datetime.now() - start_time).total_seconds()

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "compliance_article": "2.6",
                "analysis_date": report_date,
                "validation_passed": True
            }

            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)
            state.performance_metrics = self._calculate_performance_metrics(state)
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "safe_deposit_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze safe deposit box patterns"""
        return {"pattern_type": "safe_deposit", "insights": [], "recommendations": []}

    def _calculate_performance_metrics(self, state: AgentState) -> Dict:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": state.processing_time,
            "records_per_second": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
            "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
        }


# Payment Instruments Dormancy Agent (Article 2.4)
class PaymentInstrumentsDormancyAgent(BaseDormancyAgent):
    """Specialized agent for unclaimed payment instruments analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("unclaimed_instruments", memory_agent, mcp_client)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Account_Type", "Unclaimed_Item_Trigger_Date", "Unclaimed_Item_Amount"
            ],
            "business_rules": ["account_type_payment_instrument", "unclaimed_period_exceeded"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check payment instruments specific triggers"""
        try:
            df = state.input_dataframe
            payment_instruments = df[
                df['Account_Type'].astype(str).str.contains("Bankers_Cheque|Bank_Draft|Cashier_Order", case=False,
                                                            na=False)
            ]

            if payment_instruments.empty:
                return False

            state.triggered_by = DormancyTrigger.PAYMENT_INSTRUMENT_UNCLAIMED
            return True

        except Exception as e:
            logger.error(f"Payment instruments trigger check failed: {str(e)}")
            return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze unclaimed payment instruments"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for payment instruments analysis")

            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_unclaimed_payment_instruments(
                state.input_dataframe, report_date_obj
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count
            state.processing_time = (datetime.now() - start_time).total_seconds()

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "compliance_article": "2.4",
                "analysis_date": report_date,
                "validation_passed": True
            }

            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)
            state.performance_metrics = self._calculate_performance_metrics(state)
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "payment_instruments_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze payment instruments patterns"""
        return {"pattern_type": "payment_instruments", "insights": [], "recommendations": []}

    def _calculate_performance_metrics(self, state: AgentState) -> Dict:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": state.processing_time,
            "records_per_second": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
            "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
        }


# High Value Dormancy Agent
class HighValueDormancyAgent(BaseDormancyAgent):
    """Specialized agent for high-value dormant accounts analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient, threshold_balance: float = 25000):
        super().__init__("high_value_dormancy", memory_agent, mcp_client)
        self.threshold_balance = threshold_balance

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": ["Account_ID", "Current_Balance", "Expected_Account_Dormant"],
            "business_rules": ["high_balance_dormant_accounts"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check high value dormancy specific triggers"""
        try:
            df = state.input_dataframe
            if 'Current_Balance' not in df.columns or 'Expected_Account_Dormant' not in df.columns:
                return False

            high_value_dormant = df[
                (pd.to_numeric(df['Current_Balance'], errors='coerce').fillna(0) >= self.threshold_balance) &
                (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1']))
                ]

            if high_value_dormant.empty:
                return False

            state.triggered_by = DormancyTrigger.HIGH_VALUE_THRESHOLD
            return True

        except Exception as e:
            logger.error(f"High value dormancy trigger check failed: {str(e)}")
            return False

    async def analyze_dormancy(self, state: AgentState, report_date: str = None) -> AgentState:
        """Analyze high-value dormant accounts"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for high-value analysis")

            processed_df, count, description, details = check_high_value_dormant_accounts(
                state.input_dataframe, self.threshold_balance
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count
            state.processing_time = (datetime.now() - start_time).total_seconds()

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "threshold_balance": self.threshold_balance,
                "analysis_date": report_date or datetime.now().strftime("%Y-%m-%d"),
                "validation_passed": True
            }

            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)
            state.performance_metrics = self._calculate_performance_metrics(state)
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "high_value_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze high-value dormant account patterns"""
        return {"pattern_type": "high_value", "insights": [], "recommendations": []}

    def _calculate_performance_metrics(self, state: AgentState) -> Dict:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": state.processing_time,
            "records_per_second": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
            "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
        }


# CB Transfer Eligibility Agent (Article 8)
class CBTransferEligibilityAgent(BaseDormancyAgent):
    """Specialized agent for Central Bank transfer eligibility analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("cb_transfer_eligibility", memory_agent, mcp_client)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Account_Type", "Date_Last_Cust_Initiated_Activity",
                "Customer_Has_Active_Liability_Account", "Customer_Address_Known"
            ],
            "business_rules": ["dormant_5_years", "no_active_accounts", "address_unknown"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check CB transfer eligibility specific triggers"""
        try:
            df = state.input_dataframe
            report_date = datetime.strptime(
                state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d")),
                "%Y-%m-%d"
            )
            threshold_5_years = report_date - timedelta(days=5 * 365)

            # Check for accounts dormant for 5+ years
            if 'Date_Last_Cust_Initiated_Activity' in df.columns:
                df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df['Date_Last_Cust_Initiated_Activity'], errors='coerce'
                )

                eligible_accounts = df[
                    df['Date_Last_Cust_Initiated_Activity'] < threshold_5_years
                    ]

                if eligible_accounts.empty:
                    return False

            state.triggered_by = DormancyTrigger.CB_TRANSFER_ELIGIBILITY
            return True

        except Exception as e:
            logger.error(f"CB transfer eligibility trigger check failed: {str(e)}")
            return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze CB transfer eligibility"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for CB transfer analysis")

            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_eligible_for_cb_transfer(
                state.input_dataframe, report_date_obj
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count
            state.processing_time = (datetime.now() - start_time).total_seconds()

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "compliance_article": "8",
                "analysis_date": report_date,
                "validation_passed": True
            }

            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)
            state.performance_metrics = self._calculate_performance_metrics(state)
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "cb_transfer_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze CB transfer eligibility patterns"""
        return {"pattern_type": "cb_transfer", "insights": [], "recommendations": []}

    def _calculate_performance_metrics(self, state: AgentState) -> Dict:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": state.processing_time,
            "records_per_second": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
            "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
        }


# Article 3 Process Agent
class Article3ProcessAgent(BaseDormancyAgent):
    """Specialized agent for Article 3 process analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("article_3_process", memory_agent, mcp_client)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Expected_Account_Dormant",
                "Bank_Contact_Attempted_Post_Dormancy_Trigger", "Date_Last_Bank_Contact_Attempt"
            ],
            "business_rules": ["account_dormant", "contact_process_needed"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check Article 3 process specific triggers"""
        try:
            df = state.input_dataframe
            dormant_accounts = df[
                df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])
            ]

            if dormant_accounts.empty:
                return False

            state.triggered_by = DormancyTrigger.ARTICLE_3_PROCESS
            return True

        except Exception as e:
            logger.error(f"Article 3 process trigger check failed: {str(e)}")
            return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze Article 3 process requirements"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for Article 3 process analysis")

            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_art3_process_needed(
                state.input_dataframe, report_date_obj
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count
            state.processing_time = (datetime.now() - start_time).total_seconds()

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "compliance_article": "3",
                "analysis_date": report_date,
                "validation_passed": True
            }

            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)
            state.performance_metrics = self._calculate_performance_metrics(state)
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "article_3_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze Article 3 process patterns"""
        return {"pattern_type": "article_3", "insights": [], "recommendations": []}

    def _calculate_performance_metrics(self, state: AgentState) -> Dict:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": state.processing_time,
            "records_per_second": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
            "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
        }


# Proactive Contact Agent
class ProactiveContactAgent(BaseDormancyAgent):
    """Specialized agent for proactive contact analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("proactive_contact", memory_agent, mcp_client)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Date_Last_Cust_Initiated_Activity",
                "Date_Last_Customer_Communication_Any_Type", "Bank_Contact_Attempted_Post_Dormancy_Trigger"
            ],
            "business_rules": ["nearing_dormancy", "no_recent_contact"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check proactive contact specific triggers"""
        try:
            df = state.input_dataframe
            report_date = datetime.strptime(
                state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d")),
                "%Y-%m-%d"
            )

            # Check for accounts nearing dormancy (e.g., 2.5 years inactive)
            warning_threshold = report_date - timedelta(days=2.5 * 365)

            if 'Date_Last_Cust_Initiated_Activity' in df.columns:
                df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df['Date_Last_Cust_Initiated_Activity'], errors='coerce'
                )

                nearing_dormant = df[
                    df['Date_Last_Cust_Initiated_Activity'] < warning_threshold
                    ]

                if nearing_dormant.empty:
                    return False

            state.triggered_by = DormancyTrigger.PROACTIVE_CONTACT
            return True

        except Exception as e:
            logger.error(f"Proactive contact trigger check failed: {str(e)}")
            return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze proactive contact requirements"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for proactive contact analysis")

            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_contact_attempts_needed(
                state.input_dataframe, report_date_obj
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count
            state.processing_time = (datetime.now() - start_time).total_seconds()

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "compliance_article": "5",
                "analysis_date": report_date,
                "validation_passed": True
            }

            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)
            state.performance_metrics = self._calculate_performance_metrics(state)
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "proactive_contact_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze proactive contact patterns"""
        return {"pattern_type": "proactive_contact", "insights": [], "recommendations": []}

    def _calculate_performance_metrics(self, state: AgentState) -> Dict:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": state.processing_time,
            "records_per_second": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
            "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
        }


# Dormant to Active Transition Agent
class DormantToActiveAgent(BaseDormancyAgent):
    """Specialized agent for dormant-to-active transition analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("dormant_to_active", memory_agent, mcp_client)

        self.trigger_conditions = {
            "data_availability": True,
            "minimum_records": 1,
            "required_columns": [
                "Account_ID", "Date_Last_Cust_Initiated_Activity", "Expected_Account_Dormant"
            ],
            "business_rules": ["recent_activity", "previously_dormant"]
        }

    async def _check_agent_specific_triggers(self, state: AgentState) -> bool:
        """Check dormant-to-active transition specific triggers"""
        try:
            df = state.input_dataframe
            report_date = datetime.strptime(
                state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d")),
                "%Y-%m-%d"
            )

            # Check for recent activity (last 30 days)
            recent_threshold = report_date - timedelta(days=30)

            if 'Date_Last_Cust_Initiated_Activity' in df.columns:
                df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(
                    df['Date_Last_Cust_Initiated_Activity'], errors='coerce'
                )

                recently_active = df[
                    (df['Date_Last_Cust_Initiated_Activity'] >= recent_threshold) &
                    (~df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1']))
                    ]

                if recently_active.empty:
                    return False

            state.triggered_by = DormancyTrigger.DORMANT_TO_ACTIVE
            return True

        except Exception as e:
            logger.error(f"Dormant-to-active trigger check failed: {str(e)}")
            return False

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze dormant-to-active transitions"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for dormant-to-active analysis")

            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")

            # For this analysis, we need historical dormancy flags
            # This would typically come from the memory system or a separate data source
            dormant_flags_history_df = None  # Would be loaded from memory or database

            processed_df, count, description, details = check_dormant_to_active_transitions(
                state.input_dataframe, report_date_obj, dormant_flags_history_df
            )

            state.processed_dataframe = processed_df
            state.records_processed = len(state.input_dataframe)
            state.dormant_records_found = count
            state.processing_time = (datetime.now() - start_time).total_seconds()

            state.analysis_results = {
                "count": count,
                "description": description,
                "details": details,
                "analysis_date": report_date,
                "validation_passed": True
            }

            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)
            state.performance_metrics = self._calculate_performance_metrics(state)
            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "dormant_to_active_analysis")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze dormant-to-active transition patterns"""
        return {"pattern_type": "dormant_to_active", "insights": [], "recommendations": []}

    def _calculate_performance_metrics(self, state: AgentState) -> Dict:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": state.processing_time,
            "records_per_second": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
            "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
        }


# LangGraph Workflow Orchestrator
class DormancyWorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client

        # Initialize all specialized agents
        self.agents = {
            "demand_deposit": DemandDepositDormancyAgent(memory_agent, mcp_client),
            "fixed_deposit": FixedDepositDormancyAgent(memory_agent, mcp_client),
            "investment": InvestmentAccountDormancyAgent(memory_agent, mcp_client),
            "safe_deposit": SafeDepositBoxDormancyAgent(memory_agent, mcp_client),
            "payment_instruments": PaymentInstrumentsDormancyAgent(memory_agent, mcp_client),
            "high_value": HighValueDormancyAgent(memory_agent, mcp_client),
            "cb_transfer": CBTransferEligibilityAgent(memory_agent, mcp_client),
            "article_3": Article3ProcessAgent(memory_agent, mcp_client),
            "proactive_contact": ProactiveContactAgent(memory_agent, mcp_client),
            "dormant_to_active": DormantToActiveAgent(memory_agent, mcp_client)
        }

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for dormancy analysis"""

        # Define workflow state
        workflow = StateGraph(DormancyAnalysisState)

        # Add nodes for each step
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("check_triggers", self._check_all_triggers)
        workflow.add_node("execute_agents", self._execute_triggered_agents)
        workflow.add_node("consolidate_results", self._consolidate_all_results)
        workflow.add_node("handle_errors", self._handle_workflow_errors)
        workflow.add_node("finalize", self._finalize_analysis)

        # Define workflow edges with conditional routing
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "check_triggers")

        # Conditional routing based on triggers
        workflow.add_conditional_edges(
            "check_triggers",
            self._route_after_triggers,
            {
                "execute": "execute_agents",
                "no_triggers": "finalize",
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

        workflow.add_conditional_edges(
            "consolidate_results",
            self._route_after_consolidation,
            {
                "finalize": "finalize",
                "error": "handle_errors"
            }
        )

        workflow.add_edge("handle_errors", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=MemorySaver())

    async def _initialize_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Initialize dormancy analysis"""
        try:
            state.analysis_status = DormancyStatus.PROCESSING
            state.current_node = "initialize"

            # Log initialization
            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "initialization",
                "status": "started",
                "total_agents": len(self.agents)
            })

            # Validate input data
            if not state.processed_data or 'accounts' not in state.processed_data:
                raise ValueError("No account data available for dormancy analysis")

            accounts_df = pd.DataFrame(state.processed_data['accounts'])
            if accounts_df.empty:
                raise ValueError("Empty account data provided")

            state.total_accounts_analyzed = len(accounts_df)

            return state

        except Exception as e:
            logger.error(f"Analysis initialization failed: {str(e)}")
            state.analysis_status = DormancyStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "initialization",
                "error": str(e)
            })
            return state

    async def _check_all_triggers(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Check triggers for all agents"""
        try:
            state.current_node = "check_triggers"
            accounts_df = pd.DataFrame(state.processed_data['accounts'])
            report_date = state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d"))

            triggered_agents = []

            for agent_name, agent in self.agents.items():
                try:
                    # Create agent state for trigger checking
                    agent_state = AgentState(
                        agent_id=f"{agent_name}_{secrets.token_hex(8)}",
                        agent_type=agent_name,
                        session_id=state.session_id,
                        user_id=state.user_id,
                        timestamp=datetime.now(),
                        input_dataframe=accounts_df.copy(),
                        analysis_config=state.analysis_config
                    )

                    # Check if agent should be triggered
                    should_trigger = await agent.check_triggers(agent_state)

                    if should_trigger:
                        triggered_agents.append(agent_name)
                        logger.info(f"Agent {agent_name} triggered by {agent_state.triggered_by}")

                except Exception as e:
                    logger.warning(f"Trigger check failed for {agent_name}: {str(e)}")
                    state.error_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "trigger_check",
                        "agent": agent_name,
                        "error": str(e)
                    })

            state.active_agents = triggered_agents

            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "trigger_check",
                "triggered_agents": triggered_agents,
                "total_triggered": len(triggered_agents)
            })

            if not triggered_agents:
                state.routing_decision = "no_triggers"
            else:
                state.routing_decision = "execute"

            return state

        except Exception as e:
            logger.error(f"Trigger checking failed: {str(e)}")
            state.routing_decision = "error"
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "trigger_check",
                "error": str(e)
            })
            return state

    async def _execute_triggered_agents(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Execute all triggered agents"""
        try:
            state.current_node = "execute_agents"
            accounts_df = pd.DataFrame(state.processed_data['accounts'])
            report_date = state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d"))

            # Execute agents in parallel
            agent_tasks = []
            for agent_name in state.active_agents:
                agent = self.agents[agent_name]

                # Create agent state
                agent_state = AgentState(
                    agent_id=f"{agent_name}_{secrets.token_hex(8)}",
                    agent_type=agent_name,
                    session_id=state.session_id,
                    user_id=state.user_id,
                    timestamp=datetime.now(),
                    input_dataframe=accounts_df.copy(),
                    analysis_config=state.analysis_config
                )

                # Create task for agent execution
                task = self._execute_single_agent(agent, agent_state, report_date)
                agent_tasks.append((agent_name, task))

            # Wait for all agents to complete
            for agent_name, task in agent_tasks:
                try:
                    completed_state = await task
                    state.agent_results[agent_name] = completed_state

                    if completed_state.agent_status == AgentStatus.COMPLETED:
                        state.completed_agents.append(agent_name)
                        state.dormant_accounts_found += completed_state.dormant_records_found
                    else:
                        state.failed_agents.append(agent_name)

                except Exception as e:
                    logger.error(f"Agent {agent_name} execution failed: {str(e)}")
                    state.failed_agents.append(agent_name)
                    state.error_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "agent_execution",
                        "agent": agent_name,
                        "error": str(e)
                    })

            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "agent_execution",
                "completed_agents": state.completed_agents,
                "failed_agents": state.failed_agents,
                "total_dormant_found": state.dormant_accounts_found
            })

            if state.completed_agents:
                state.routing_decision = "consolidate"
            else:
                state.routing_decision = "error"

            return state

        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            state.routing_decision = "error"
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "agent_execution",
                "error": str(e)
            })
            return state

    async def _execute_single_agent(self, agent: BaseDormancyAgent, agent_state: AgentState,
                                    report_date: str) -> AgentState:
        """Execute a single agent with full pipeline"""
        try:
            # Pre-analysis memory hook
            agent_state = await agent.pre_analysis_memory_hook(agent_state)

            # Main analysis
            agent_state = await agent.analyze_dormancy(agent_state, report_date)

            # Post-analysis memory hook
            agent_state = await agent.post_analysis_memory_hook(agent_state)

            return agent_state

        except Exception as e:
            logger.error(f"Single agent execution failed for {agent.agent_type}: {str(e)}")
            agent_state.agent_status = AgentStatus.FAILED
            await agent._handle_error(agent_state, e, "single_agent_execution")
            return agent_state

    async def _consolidate_all_results(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Consolidate results from all agents"""
        try:
            state.current_node = "consolidate_results"

            consolidated = {
                "total_records_analyzed": state.total_accounts_analyzed,
                "total_dormant_found": state.dormant_accounts_found,
                "dormancy_breakdown": {},
                "pattern_insights": [],
                "recommendations": [],
                "risk_assessment": {
                    "high_risk_items": 0,
                    "medium_risk_items": 0,
                    "low_risk_items": 0
                },
                "compliance_summary": {},
                "next_actions": []
            }

            # Process results from each completed agent
            for agent_name in state.completed_agents:
                agent_result = state.agent_results[agent_name]

                if agent_result.analysis_results:
                    # Add to breakdown
                    consolidated["dormancy_breakdown"][agent_name] = {
                        "count": agent_result.dormant_records_found,
                        "description": agent_result.analysis_results.get("description", ""),
                        "article": agent_result.analysis_results.get("compliance_article", "")
                    }

                    # Collect pattern insights
                    if agent_result.pattern_analysis and agent_result.pattern_analysis.get("insights"):
                        for insight in agent_result.pattern_analysis["insights"]:
                            consolidated["pattern_insights"].append(f"[{agent_name.upper()}] {insight}")

                    # Collect recommendations
                    if agent_result.pattern_analysis and agent_result.pattern_analysis.get("recommendations"):
                        for rec in agent_result.pattern_analysis["recommendations"]:
                            consolidated["recommendations"].append(f"[{agent_name.upper()}] {rec}")

                    # Assess risk levels
                    if agent_result.dormant_records_found > 0:
                        if agent_name in ["high_value", "cb_transfer"]:
                            consolidated["risk_assessment"]["high_risk_items"] += agent_result.dormant_records_found
                        elif agent_name in ["safe_deposit", "payment_instruments"]:
                            consolidated["risk_assessment"]["medium_risk_items"] += agent_result.dormant_records_found
                        else:
                            consolidated["risk_assessment"]["low_risk_items"] += agent_result.dormant_records_found

            # Generate compliance summary
            articles_with_findings = []
            for agent_name, breakdown in consolidated["dormancy_breakdown"].items():
                if breakdown["count"] > 0 and breakdown.get("article"):
                    articles_with_findings.append(f"Article {breakdown['article']}")

            consolidated["compliance_summary"] = {
                "articles_with_findings": articles_with_findings,
                "compliance_status": "ATTENTION_REQUIRED" if articles_with_findings else "COMPLIANT",
                "total_compliance_items": len(articles_with_findings)
            }

            # Generate next actions
            if consolidated["risk_assessment"]["high_risk_items"] > 0:
                consolidated["next_actions"].append("URGENT: Address high-risk dormant accounts immediately")

            if consolidated["compliance_summary"]["total_compliance_items"] > 0:
                consolidated["next_actions"].append(
                    f"Review {consolidated['compliance_summary']['total_compliance_items']} CBUAE compliance items")

            if not consolidated["next_actions"]:
                consolidated["next_actions"].append("Continue monitoring - no immediate actions required")

            # Store consolidated results
            state.dormancy_summary = consolidated
            state.compliance_flags = articles_with_findings
            state.high_risk_accounts = consolidated["risk_assessment"]["high_risk_items"]

            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "consolidation",
                "total_findings": len(consolidated["dormancy_breakdown"]),
                "compliance_items": len(articles_with_findings),
                "high_risk_items": consolidated["risk_assessment"]["high_risk_items"]
            })

            state.routing_decision = "finalize"
            return state

        except Exception as e:
            logger.error(f"Result consolidation failed: {str(e)}")
            state.routing_decision = "error"
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "consolidation",
                "error": str(e)
            })
            return state

    async def _handle_workflow_errors(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Handle workflow-level errors"""
        try:
            state.current_node = "handle_errors"
            state.analysis_status = DormancyStatus.FAILED

            # Create error state for error handler
            error_state = ErrorState(
                session_id=state.session_id,
                user_id=state.user_id,
                error_id=secrets.token_hex(8),
                timestamp=datetime.now(),
                errors=[{
                    "error_type": "workflow_error",
                    "error_message": "Workflow execution failed",
                    "stage": state.current_node,
                    "critical": True
                }],
                failed_node=state.current_node,
                workflow_context=asdict(state)
            )

            # Initialize error handler
            error_handler = ErrorHandlerAgent(self.memory_agent, self.mcp_client)
            error_result = await error_handler.handle_workflow_error(error_state)

            # Update state based on error handling
            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "error_handling",
                "recovery_action": error_result.recovery_action,
                "recovery_success": error_result.recovery_success
            })

            return state

        except Exception as e:
            logger.error(f"Error handling failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "error_handling",
                "error": str(e)
            })
            return state

    async def _finalize_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Finalize dormancy analysis"""
        try:
            state.current_node = "finalize"

            # Calculate final metrics
            if state.analysis_status != DormancyStatus.FAILED:
                state.analysis_status = DormancyStatus.COMPLETED

            # Calculate processing efficiency
            total_processing_time = sum(
                result.processing_time for result in state.agent_results.values()
                if hasattr(result, 'processing_time')
            )

            if total_processing_time > 0:
                state.analysis_efficiency = state.total_accounts_analyzed / total_processing_time

            # Create final dormancy results
            state.dormancy_results = {
                "session_id": state.session_id,
                "analysis_timestamp": state.timestamp.isoformat(),
                "total_processing_time": total_processing_time,
                "agents_executed": len(state.active_agents),
                "agents_completed": len(state.completed_agents),
                "agents_failed": len(state.failed_agents),
                "completed_agents": state.completed_agents,
                "failed_agents": state.failed_agents,
                "agent_results": {
                    agent_name: {
                        "agent_id": result.agent_id,
                        "agent_status": result.agent_status.value,
                        "records_processed": result.records_processed,
                        "dormant_found": result.dormant_records_found,
                        "processing_time": result.processing_time,
                        "triggered_by": result.triggered_by.value if result.triggered_by else None,
                        "analysis_results": result.analysis_results,
                        "pattern_analysis": result.pattern_analysis,
                        "performance_metrics": result.performance_metrics
                    } for agent_name, result in state.agent_results.items()
                },
                "consolidated_summary": state.dormancy_summary
            }

            # Final log entry
            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "finalization",
                "status": state.analysis_status.value,
                "total_dormant_found": state.dormant_accounts_found,
                "analysis_efficiency": state.analysis_efficiency,
                "completion_time": datetime.now().isoformat()
            })

            logger.info(
                f"Dormancy analysis completed: {len(state.completed_agents)}/{len(state.active_agents)} agents successful")

            return state

        except Exception as e:
            logger.error(f"Analysis finalization failed: {str(e)}")
            state.analysis_status = DormancyStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "finalization",
                "error": str(e)
            })
            return state

    def _route_after_triggers(self, state: DormancyAnalysisState) -> str:
        """Route after trigger checking"""
        return state.routing_decision

    def _route_after_execution(self, state: DormancyAnalysisState) -> str:
        """Route after agent execution"""
        return state.routing_decision

    def _route_after_consolidation(self, state: DormancyAnalysisState) -> str:
        """Route after result consolidation"""
        return state.routing_decision

    async def orchestrate_analysis(self, user_id: str, input_dataframe: pd.DataFrame,
                                   report_date: str, analysis_config: Dict = None) -> Dict:
        """Orchestrate dormancy analysis using LangGraph workflow"""
        try:
            # Create initial state
            initial_state = DormancyAnalysisState(
                session_id=secrets.token_hex(16),
                user_id=user_id,
                analysis_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                processed_data={'accounts': input_dataframe.to_dict('records')},
                analysis_config=analysis_config or {'report_date': report_date}
            )

            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Return results
            return final_state.dormancy_results

        except Exception as e:
            logger.error(f"Workflow orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": initial_state.session_id if 'initial_state' in locals() else None
            }


# Main Dormancy Analysis Agent (used by workflow engine)
class DormancyAnalysisAgent:
    """Main dormancy analysis agent that orchestrates all sub-agents using LangGraph"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # Initialize the LangGraph orchestrator
        self.orchestrator = DormancyWorkflowOrchestrator(memory_agent, mcp_client)

    @traceable(name="analyze_dormancy")
    async def analyze_dormancy(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Main dormancy analysis method called by workflow engine"""

        try:
            start_time = datetime.now()
            state.analysis_status = DormancyStatus.PROCESSING

            # Extract account data from processed_data
            if not state.processed_data or 'accounts' not in state.processed_data:
                raise ValueError("No account data available for dormancy analysis")

            # Convert to DataFrame
            accounts_df = pd.DataFrame(state.processed_data['accounts'])

            if accounts_df.empty:
                raise ValueError("Empty account data provided")

            # Use current date if not specified in config
            report_date = state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d"))

            # Execute orchestrated analysis using LangGraph workflow
            orchestration_results = await self.orchestrator.orchestrate_analysis(
                user_id=state.user_id,
                input_dataframe=accounts_df,
                report_date=report_date,
                analysis_config=state.analysis_config
            )

            # Extract results from orchestration
            if orchestration_results and not orchestration_results.get("success") == False:
                state.dormancy_results = orchestration_results
                state.dormancy_summary = orchestration_results.get("consolidated_summary", {})

                # Update state metrics
                summary = state.dormancy_summary
                state.total_accounts_analyzed = summary.get("total_records_analyzed", len(accounts_df))
                state.dormant_accounts_found = summary.get("total_dormant_found", 0)

                # Count high-risk accounts
                risk_assessment = summary.get("risk_assessment", {})
                state.high_risk_accounts = risk_assessment.get("high_risk_items", 0)

                # Extract compliance flags
                compliance_summary = summary.get("compliance_summary", {})
                state.compliance_flags = compliance_summary.get("articles_with_findings", [])

                # Calculate processing metrics
                state.processing_time = (datetime.now() - start_time).total_seconds()
                state.analysis_efficiency = (
                    state.total_accounts_analyzed / state.processing_time
                    if state.processing_time > 0 else 0
                )

                state.analysis_status = DormancyStatus.COMPLETED

                # Log successful analysis
                state.analysis_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "main_dormancy_analysis",
                    "status": "completed",
                    "accounts_analyzed": state.total_accounts_analyzed,
                    "dormant_found": state.dormant_accounts_found,
                    "high_risk": state.high_risk_accounts,
                    "processing_time": state.processing_time,
                    "workflow_orchestration": "langgraph_success"
                })

            else:
                state.analysis_status = DormancyStatus.FAILED
                error_msg = orchestration_results.get("error", "Unknown orchestration error")
                state.error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "orchestration",
                    "error": error_msg,
                    "workflow_type": "langgraph"
                })

        except Exception as e:
            state.analysis_status = DormancyStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "main_dormancy_analysis",
                "error": str(e),
                "workflow_type": "langgraph"
            })
            logger.error(f"Dormancy analysis failed: {str(e)}")

        return state


# Additional utility functions
def create_comprehensive_dormancy_analysis(memory_agent, mcp_client: MCPClient) -> DormancyAnalysisAgent:
    """Factory function to create a comprehensive dormancy analysis agent"""
    return DormancyAnalysisAgent(memory_agent, mcp_client)


def get_available_agents() -> List[str]:
    """Get list of available dormancy analysis agents"""
    return [
        "demand_deposit", "fixed_deposit", "investment", "safe_deposit",
        "payment_instruments", "high_value", "cb_transfer", "article_3",
        "proactive_contact", "dormant_to_active"
    ]


def get_agent_trigger_conditions(agent_type: str) -> Dict:
    """Get trigger conditions for a specific agent type"""
    agent_configs = {
        "demand_deposit": DemandDepositDormancyAgent,
        "fixed_deposit": FixedDepositDormancyAgent,
        "investment": InvestmentAccountDormancyAgent,
        "safe_deposit": SafeDepositBoxDormancyAgent,
        "payment_instruments": PaymentInstrumentsDormancyAgent,
        "high_value": HighValueDormancyAgent,
        "cb_transfer": CBTransferEligibilityAgent,
        "article_3": Article3ProcessAgent,
        "proactive_contact": ProactiveContactAgent,
        "dormant_to_active": DormantToActiveAgent
    }

    if agent_type in agent_configs:
        # Create a temporary instance to get trigger conditions
        temp_agent = agent_configs[agent_type](None, None)
        return temp_agent.trigger_conditions

    return {}


# Export the main classes for use by workflow engine
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
    "HighValueDormancyAgent",
    "CBTransferEligibilityAgent",
    "Article3ProcessAgent",
    "ProactiveContactAgent",
    "DormantToActiveAgent",
    "create_comprehensive_dormancy_analysis",
    "get_available_agents",
    "get_agent_trigger_conditions"
]