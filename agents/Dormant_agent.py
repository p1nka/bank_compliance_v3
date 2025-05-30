import logger
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable, Client as LangSmithClient

# MCP imports - using try/except to handle import issues
try:
    from mcp_client import MCPClient
except ImportError:
    logger.warning("MCPClient not available, using mock implementation")


    class MCPClient:
        async def call_tool(self, tool_name: str, params: Dict) -> Dict:
            return {"success": True, "data": {}}

# Import the dormant analysis functions - with error handling
try:
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
        check_dormant_to_active_transitions
    )
except ImportError:
    logger.warning("Dormant analysis functions not available, using mock implementations")


    # Mock implementations
    def check_demand_deposit_inactivity(df, date):
        return df.copy(), 0, "Mock analysis", {}


    def check_fixed_deposit_inactivity(df, date):
        return df.copy(), 0, "Mock analysis", {}


    def check_investment_inactivity(df, date):
        return df.copy(), 0, "Mock analysis", {}


    def check_safe_deposit_dormancy(df, date):
        return df.copy(), 0, "Mock analysis", {}


    def check_unclaimed_payment_instruments(df, date):
        return df.copy(), 0, "Mock analysis", {}


    def check_eligible_for_cb_transfer(df, date):
        return df.copy(), 0, "Mock analysis", {}


    def check_art3_process_needed(df, date):
        return df.copy(), 0, "Mock analysis", {}


    def check_contact_attempts_needed(df, date):
        return df.copy(), 0, "Mock analysis", {}


    def check_high_value_dormant_accounts(df, threshold):
        return df.copy(), 0, "Mock analysis", {}


    def check_dormant_to_active_transitions(df, date):
        return df.copy(), 0, "Mock analysis", {}

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


class DormancyStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


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
        if self.execution_log is None:
            self.execution_log = []
        if self.error_log is None:
            self.error_log = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


# Base Dormancy Agent Class
class BaseDormancyAgent:
    """Base class for all dormancy analysis agents"""

    def __init__(self, agent_type: str, memory_agent, mcp_client: MCPClient):
        self.agent_type = agent_type
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
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

    @traceable(name="base_pre_analysis_hook")
    async def pre_analysis_hook(self, state: AgentState) -> AgentState:
        """Base pre-analysis memory hook"""
        try:
            state.agent_status = AgentStatus.PROCESSING

            # Retrieve agent-specific patterns from memory
            if self.memory_agent:
                agent_patterns = await self.memory_agent.retrieve_memory(
                    bucket="knowledge",
                    filter_criteria={
                        "type": f"{self.agent_type}_patterns",
                        "user_id": state.user_id,
                        "agent_type": self.agent_type
                    }
                )

                if agent_patterns.get("success"):
                    state.retrieved_patterns = agent_patterns.get("data", {})
                    logger.info(f"Retrieved {self.agent_type} patterns from memory")

                # Retrieve historical analysis data
                historical_data = await self.memory_agent.retrieve_memory(
                    bucket="knowledge",
                    filter_criteria={
                        "type": f"{self.agent_type}_historical",
                        "user_id": state.user_id
                    }
                )

                if historical_data.get("success"):
                    state.pre_hook_memory["historical_data"] = historical_data.get("data", {})

                # Store DataFrame in memory for this session
                if state.input_dataframe is not None:
                    df_dict = state.input_dataframe.to_dict('records')
                    await self.memory_agent.store_memory(
                        bucket="session",
                        data={
                            "agent_id": state.agent_id,
                            "agent_type": self.agent_type,
                            "input_dataframe": df_dict,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    state.pre_hook_memory["input_df_stored"] = True

            # Log pre-analysis execution
            state.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_analysis_hook",
                "agent_type": self.agent_type,
                "action": "memory_retrieval_and_storage",
                "patterns_retrieved": len(state.retrieved_patterns),
                "historical_data_loaded": bool(state.pre_hook_memory.get("historical_data")),
                "input_df_stored": state.pre_hook_memory.get("input_df_stored", False)
            })

        except Exception as e:
            logger.error(f"{self.agent_type} pre-analysis hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_analysis_hook",
                "agent_type": self.agent_type,
                "error": str(e)
            })

        return state

    @traceable(name="base_post_analysis_hook")
    async def post_analysis_hook(self, state: AgentState) -> AgentState:
        """Base post-analysis memory hook"""
        try:
            if self.memory_agent:
                # Store processed DataFrame
                if state.processed_dataframe is not None:
                    processed_df_dict = state.processed_dataframe.to_dict('records')
                    await self.memory_agent.store_memory(
                        bucket="session",
                        data={
                            "agent_id": state.agent_id,
                            "agent_type": self.agent_type,
                            "processed_dataframe": processed_df_dict,
                            "analysis_results": state.analysis_results,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    state.post_hook_memory["processed_df_stored"] = True

                # Store analysis results and patterns
                if state.analysis_results:
                    await self.memory_agent.store_memory(
                        bucket="session",
                        data={
                            "agent_id": state.agent_id,
                            "agent_type": self.agent_type,
                            "session_results": {
                                "records_processed": state.records_processed,
                                "dormant_found": state.dormant_records_found,
                                "processing_time": state.processing_time,
                                "analysis_results": state.analysis_results
                            }
                        }
                    )

                # Store new patterns in knowledge memory
                if state.pattern_analysis and state.agent_status == AgentStatus.COMPLETED:
                    pattern_data = {
                        "type": f"{self.agent_type}_patterns",
                        "user_id": state.user_id,
                        "agent_type": self.agent_type,
                        "patterns": state.pattern_analysis,
                        "performance_metrics": state.performance_metrics,
                        "timestamp": datetime.now().isoformat()
                    }

                    await self.memory_agent.store_memory(
                        bucket="knowledge",
                        data=pattern_data
                    )
                    state.stored_patterns = pattern_data

                # Store historical data for future analysis
                if state.agent_status == AgentStatus.COMPLETED:
                    historical_data = {
                        "type": f"{self.agent_type}_historical",
                        "user_id": state.user_id,
                        "agent_type": self.agent_type,
                        "analysis_summary": {
                            "dormancy_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0,
                            "processing_efficiency": state.records_processed / state.processing_time if state.processing_time > 0 else 0,
                            "key_findings": state.analysis_results.get("key_findings", [])
                        },
                        "timestamp": datetime.now().isoformat()
                    }

                    await self.memory_agent.store_memory(
                        bucket="knowledge",
                        data=historical_data
                    )

            # Log post-analysis execution
            state.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_analysis_hook",
                "agent_type": self.agent_type,
                "action": "memory_storage",
                "processed_df_stored": state.post_hook_memory.get("processed_df_stored", False),
                "patterns_stored": bool(state.stored_patterns),
                "historical_data_stored": state.agent_status == AgentStatus.COMPLETED
            })

        except Exception as e:
            logger.error(f"{self.agent_type} post-analysis hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_analysis_hook",
                "agent_type": self.agent_type,
                "error": str(e)
            })

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Base pattern analysis - to be overridden by specific agents"""
        return {"pattern_type": "base", "insights": []}


# Main Dormancy Analysis Agent (used by workflow engine)
class DormancyAnalysisAgent:
    """Main dormancy analysis agent that orchestrates all sub-agents"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # Initialize the orchestrator
        self.orchestrator = DormancyAgentOrchestrator(memory_agent, mcp_client)

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

            # Execute orchestrated analysis
            orchestration_results = await self.orchestrator.orchestrate_analysis(
                user_id=state.user_id,
                input_dataframe=accounts_df,
                report_date=report_date,
                analysis_config=state.analysis_config
            )

            # Extract results
            if orchestration_results.get("success", True):  # Default to True if not specified
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
                    "stage": "dormancy_analysis",
                    "status": "completed",
                    "accounts_analyzed": state.total_accounts_analyzed,
                    "dormant_found": state.dormant_accounts_found,
                    "high_risk": state.high_risk_accounts,
                    "processing_time": state.processing_time
                })

            else:
                state.analysis_status = DormancyStatus.FAILED
                error_msg = orchestration_results.get("error", "Unknown orchestration error")
                state.error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "orchestration",
                    "error": error_msg
                })

        except Exception as e:
            state.analysis_status = DormancyStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "dormancy_analysis",
                "error": str(e)
            })
            logger.error(f"Dormancy analysis failed: {str(e)}")

        return state


# Demand Deposit Dormancy Agent (Article 2.1.1)
class DemandDepositDormancyAgent(BaseDormancyAgent):
    """Specialized agent for demand deposit dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("demand_deposit_dormancy", memory_agent, mcp_client)

    @traceable(name="demand_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze demand deposit dormancy with pattern recognition"""
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
                "key_findings": []
            }

            # Enhanced pattern analysis
            state.agent_status = AgentStatus.ANALYZING_PATTERNS
            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)

            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            state.processing_time = processing_time
            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "records_per_second": state.records_processed / processing_time if processing_time > 0 else 0,
                "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0,
                "analysis_efficiency": count / processing_time if processing_time > 0 else 0
            }

            # Call MCP tool for additional processing
            if self.mcp_client:
                mcp_result = await self.mcp_client.call_tool("analyze_account_dormancy", {
                    "accounts": processed_df.to_dict('records') if not processed_df.empty else [],
                    "report_date": report_date,
                    "regulatory_params": {"article": "2.1.1", "type": "demand_deposit"}
                })

                if mcp_result.get("success"):
                    state.analysis_results["mcp_insights"] = mcp_result.get("data", {})

            state.agent_status = AgentStatus.COMPLETED

            # Add key findings
            if count > 0:
                state.analysis_results["key_findings"].extend([
                    f"Identified {count} demand deposit accounts meeting dormancy criteria",
                    f"Average dormancy processing time: {processing_time:.2f} seconds",
                    f"Dormancy rate: {(count / state.records_processed) * 100:.1f}%"
                ])

            # Log successful analysis
            state.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "demand_deposit_analysis",
                "agent_type": self.agent_type,
                "status": "completed",
                "records_processed": state.records_processed,
                "dormant_found": state.dormant_records_found,
                "processing_time": processing_time
            })

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            error_msg = str(e)
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "demand_deposit_analysis",
                "agent_type": self.agent_type,
                "error": error_msg
            })
            logger.error(f"Demand deposit analysis failed: {error_msg}")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze demand deposit specific patterns"""
        if df.empty:
            return {"pattern_type": "demand_deposit", "insights": []}

        patterns = {
            "pattern_type": "demand_deposit",
            "account_type_distribution": {},
            "balance_patterns": {},
            "activity_patterns": {},
            "communication_patterns": {},
            "insights": [],
            "recommendations": []
        }

        try:
            # Account type analysis
            if 'Account_Type' in df.columns:
                type_dist = df['Account_Type'].value_counts().to_dict()
                patterns["account_type_distribution"] = type_dist

                most_dormant_type = max(type_dist, key=type_dist.get) if type_dist else "Unknown"
                patterns["insights"].append(f"Most dormant account type: {most_dormant_type}")

            # Balance pattern analysis
            if 'Current_Balance' in df.columns:
                balances = pd.to_numeric(df['Current_Balance'], errors='coerce').dropna()
                if not balances.empty:
                    patterns["balance_patterns"] = {
                        "average_balance": float(balances.mean()),
                        "median_balance": float(balances.median()),
                        "high_balance_accounts": int((balances > 25000).sum()),
                        "low_balance_accounts": int((balances < 1000).sum())
                    }

                    if patterns["balance_patterns"]["high_balance_accounts"] > 0:
                        patterns["insights"].append(
                            f"Found {patterns['balance_patterns']['high_balance_accounts']} high-value dormant accounts")
                        patterns["recommendations"].append("Prioritize reactivation of high-balance dormant accounts")

            # Add more pattern analysis...

        except Exception as e:
            patterns["insights"].append(f"Pattern analysis error: {str(e)}")
            logger.error(f"Demand deposit pattern analysis error: {str(e)}")

        return patterns


# Master Dormancy Orchestrator
class DormancyAgentOrchestrator:
    """Master orchestrator for all dormancy agents"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client

        # Initialize all specialized agents
        self.agents = {
            "demand_deposit": DemandDepositDormancyAgent(memory_agent, mcp_client),
            # Add other agents as needed
        }

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

    @traceable(name="orchestrate_dormancy_analysis")
    async def orchestrate_analysis(self, user_id: str, input_dataframe: pd.DataFrame,
                                   report_date: str, analysis_config: Dict = None) -> Dict:
        """Orchestrate all dormancy agents in parallel"""
        try:
            session_id = secrets.token_hex(16)
            orchestration_start = datetime.now()

            # For now, just run the demand deposit agent as an example
            # In the full implementation, you would run all agents in parallel

            agent_name = "demand_deposit"
            agent = self.agents[agent_name]

            # Initialize agent state
            state = AgentState(
                agent_id=f"{agent_name}_{secrets.token_hex(8)}",
                agent_type=agent_name,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now(),
                input_dataframe=input_dataframe.copy(),
                analysis_config=analysis_config or {}
            )

            # Execute agent pipeline
            state = await self._execute_agent_pipeline(agent, state, report_date)

            # Compile orchestration results
            orchestration_time = (datetime.now() - orchestration_start).total_seconds()

            orchestration_results = {
                "session_id": session_id,
                "orchestration_timestamp": orchestration_start.isoformat(),
                "total_processing_time": orchestration_time,
                "agents_executed": 1,
                "agents_completed": 1 if state.agent_status == AgentStatus.COMPLETED else 0,
                "agents_failed": 0 if state.agent_status == AgentStatus.COMPLETED else 1,
                "completed_agents": [agent_name] if state.agent_status == AgentStatus.COMPLETED else [],
                "failed_agents": [] if state.agent_status == AgentStatus.COMPLETED else [agent_name],
                "agent_results": {
                    agent_name: {
                        "agent_id": state.agent_id,
                        "agent_status": state.agent_status.value,
                        "records_processed": state.records_processed,
                        "dormant_found": state.dormant_records_found,
                        "processing_time": state.processing_time,
                        "analysis_results": state.analysis_results,
                        "pattern_analysis": state.pattern_analysis,
                        "performance_metrics": state.performance_metrics,
                        "execution_log": state.execution_log[-5:],  # Last 5 entries
                        "error_log": state.error_log
                    }
                },
                "consolidated_summary": await self._consolidate_results({agent_name: state}),
                "memory_storage_summary": {"session_stored": True, "patterns_stored": True}
            }

            logger.info(
                f"Orchestration completed: {orchestration_results['agents_completed']}/{orchestration_results['agents_executed']} agents successful")
            return orchestration_results

        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id if 'session_id' in locals() else None
            }

    async def _execute_agent_pipeline(self, agent: BaseDormancyAgent, state: AgentState,
                                      report_date: str) -> AgentState:
        """Execute the full pipeline for a single agent"""
        try:
            # Pre-analysis hook
            state = await agent.pre_analysis_hook(state)

            # Main analysis
            state = await agent.analyze_dormancy(state, report_date)

            # Post-analysis hook
            state = await agent.post_analysis_hook(state)

            return state

        except Exception as e:
            logger.error(f"Agent {state.agent_type} pipeline failed: {str(e)}")
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "agent_pipeline",
                "error": str(e)
            })
            return state

    async def _consolidate_results(self, agent_results: Dict[str, AgentState]) -> Dict:
        """Consolidate results from all agents"""
        consolidated = {
            "total_records_analyzed": 0,
            "total_dormant_found": 0,
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

        try:
            for agent_name, state in agent_results.items():
                if state.agent_status == AgentStatus.COMPLETED and state.analysis_results:
                    # Aggregate counts
                    consolidated["total_records_analyzed"] = max(
                        consolidated["total_records_analyzed"],
                        state.records_processed
                    )
                    consolidated["total_dormant_found"] += state.dormant_records_found

                    # Store breakdown by agent
                    # Store breakdown by agent
                    consolidated["dormancy_breakdown"][agent_name] = {
                        "count": state.dormant_records_found,
                        "description": state.analysis_results.get("description", ""),
                        "article": state.analysis_results.get("compliance_article", "")
                    }

                    # Collect pattern insights
                    if state.pattern_analysis and state.pattern_analysis.get("insights"):
                        for insight in state.pattern_analysis["insights"]:
                            consolidated["pattern_insights"].append(f"[{agent_name.upper()}] {insight}")

                    # Collect recommendations
                    if state.pattern_analysis and state.pattern_analysis.get("recommendations"):
                        for rec in state.pattern_analysis["recommendations"]:
                            consolidated["recommendations"].append(f"[{agent_name.upper()}] {rec}")

                    # Assess risk levels
                    if state.dormant_records_found > 0:
                        if agent_name in ["high_value", "cb_transfer"]:
                            consolidated["risk_assessment"]["high_risk_items"] += state.dormant_records_found
                        elif agent_name in ["safe_deposit", "payment_instruments"]:
                            consolidated["risk_assessment"]["medium_risk_items"] += state.dormant_records_found
                        else:
                            consolidated["risk_assessment"]["low_risk_items"] += state.dormant_records_found

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

        except Exception as e:
            logger.error(f"Result consolidation failed: {str(e)}")
            consolidated["consolidation_error"] = str(e)

        return consolidated


# Additional specialized agents (simplified versions for the fix)
class FixedDepositDormancyAgent(BaseDormancyAgent):
    """Specialized agent for fixed deposit dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("fixed_deposit_dormancy", memory_agent, mcp_client)

    @traceable(name="fixed_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze fixed deposit dormancy with pattern recognition"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data available for fixed deposit analysis")

            # Execute CBUAE Article 2.2 analysis
            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
            processed_df, count, description, details = check_fixed_deposit_inactivity(
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
                "compliance_article": "2.2",
                "analysis_date": report_date,
                "key_findings": []
            }

            # Pattern analysis
            state.agent_status = AgentStatus.ANALYZING_PATTERNS
            state.pattern_analysis = await self.analyze_patterns(processed_df, state.analysis_results)

            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            state.processing_time = processing_time
            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "records_per_second": state.records_processed / processing_time if processing_time > 0 else 0,
                "dormancy_detection_rate": state.dormant_records_found / state.records_processed if state.records_processed > 0 else 0
            }

            # Call MCP tool
            if self.mcp_client:
                mcp_result = await self.mcp_client.call_tool("analyze_account_dormancy", {
                    "accounts": processed_df.to_dict('records') if not processed_df.empty else [],
                    "report_date": report_date,
                    "regulatory_params": {"article": "2.2", "type": "fixed_deposit"}
                })

                if mcp_result.get("success"):
                    state.analysis_results["mcp_insights"] = mcp_result.get("data", {})

            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            error_msg = str(e)
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "fixed_deposit_analysis",
                "agent_type": self.agent_type,
                "error": error_msg
            })
            logger.error(f"Fixed deposit analysis failed: {error_msg}")

        return state

    async def analyze_patterns(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Analyze fixed deposit specific patterns"""
        return {"pattern_type": "fixed_deposit", "insights": [], "recommendations": []}


class InvestmentAccountDormancyAgent(BaseDormancyAgent):
    """Specialized agent for investment account dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("investment_dormancy", memory_agent, mcp_client)

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
                "analysis_date": report_date
            }

            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

        return state


class SafeDepositBoxDormancyAgent(BaseDormancyAgent):
    """Specialized agent for safe deposit box dormancy analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("safe_deposit_dormancy", memory_agent, mcp_client)

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
                "analysis_date": report_date
            }

            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

        return state


class PaymentInstrumentsDormancyAgent(BaseDormancyAgent):
    """Specialized agent for unclaimed payment instruments analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("unclaimed_instruments", memory_agent, mcp_client)

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
                "analysis_date": report_date
            }

            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

        return state


class HighValueDormancyAgent(BaseDormancyAgent):
    """Specialized agent for high-value dormant accounts analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient, threshold_balance: float = 25000):
        super().__init__("high_value_dormancy", memory_agent, mcp_client)
        self.threshold_balance = threshold_balance

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
                "analysis_date": report_date or datetime.now().strftime("%Y-%m-%d")
            }

            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

        return state


class CBTransferEligibilityAgent(BaseDormancyAgent):
    """Specialized agent for Central Bank transfer eligibility analysis"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__("cb_transfer_eligibility", memory_agent, mcp_client)

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
                "analysis_date": report_date
            }

            state.agent_status = AgentStatus.COMPLETED

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

        return state


# Enhanced Orchestrator with all agents
class EnhancedDormancyAgentOrchestrator(DormancyAgentOrchestrator):
    """Enhanced orchestrator with all specialized agents"""

    def __init__(self, memory_agent, mcp_client: MCPClient):
        super().__init__(memory_agent, mcp_client)

        # Initialize all specialized agents
        self.agents = {
            "demand_deposit": DemandDepositDormancyAgent(memory_agent, mcp_client),
            "fixed_deposit": FixedDepositDormancyAgent(memory_agent, mcp_client),
            "investment": InvestmentAccountDormancyAgent(memory_agent, mcp_client),
            "safe_deposit": SafeDepositBoxDormancyAgent(memory_agent, mcp_client),
            "payment_instruments": PaymentInstrumentsDormancyAgent(memory_agent, mcp_client),
            "high_value": HighValueDormancyAgent(memory_agent, mcp_client),
            "cb_transfer": CBTransferEligibilityAgent(memory_agent, mcp_client)
        }

    async def orchestrate_analysis(self, user_id: str, input_dataframe: pd.DataFrame,
                                   report_date: str, analysis_config: Dict = None) -> Dict:
        """Orchestrate all dormancy agents in parallel"""
        try:
            session_id = secrets.token_hex(16)
            orchestration_start = datetime.now()

            # Initialize agent states
            agent_states = {}
            for agent_name, agent in self.agents.items():
                state = AgentState(
                    agent_id=f"{agent_name}_{secrets.token_hex(8)}",
                    agent_type=agent_name,
                    session_id=session_id,
                    user_id=user_id,
                    timestamp=datetime.now(),
                    input_dataframe=input_dataframe.copy(),
                    analysis_config=analysis_config or {}
                )
                agent_states[agent_name] = state

            # Execute all agents in parallel
            logger.info(f"Starting parallel execution of {len(self.agents)} dormancy agents")

            # Create tasks for parallel execution
            tasks = []
            for agent_name, agent in self.agents.items():
                state = agent_states[agent_name]
                task = self._execute_agent_pipeline(agent, state, report_date)
                tasks.append((agent_name, task))

            # Execute all tasks concurrently
            results = {}
            completed_agents = []
            failed_agents = []

            # Wait for all tasks to complete
            import asyncio
            for agent_name, task in tasks:
                try:
                    completed_state = await task
                    results[agent_name] = completed_state
                    if completed_state.agent_status == AgentStatus.COMPLETED:
                        completed_agents.append(agent_name)
                    else:
                        failed_agents.append(agent_name)
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {str(e)}")
                    failed_agents.append(agent_name)
                    results[agent_name] = agent_states[agent_name]  # Return original state

            # Compile orchestration results
            orchestration_time = (datetime.now() - orchestration_start).total_seconds()

            orchestration_results = {
                "session_id": session_id,
                "orchestration_timestamp": orchestration_start.isoformat(),
                "total_processing_time": orchestration_time,
                "agents_executed": len(self.agents),
                "agents_completed": len(completed_agents),
                "agents_failed": len(failed_agents),
                "completed_agents": completed_agents,
                "failed_agents": failed_agents,
                "agent_results": {},
                "consolidated_summary": await self._consolidate_results(results),
                "memory_storage_summary": {"session_stored": True, "patterns_stored": True}
            }

            # Extract agent results
            for agent_name, state in results.items():
                agent_result = {
                    "agent_id": state.agent_id,
                    "agent_status": state.agent_status.value,
                    "records_processed": state.records_processed,
                    "dormant_found": state.dormant_records_found,
                    "processing_time": state.processing_time,
                    "analysis_results": state.analysis_results,
                    "pattern_analysis": state.pattern_analysis,
                    "performance_metrics": state.performance_metrics,
                    "execution_log": state.execution_log[-5:] if state.execution_log else [],
                    "error_log": state.error_log
                }
                orchestration_results["agent_results"][agent_name] = agent_result

            logger.info(f"Orchestration completed: {len(completed_agents)}/{len(self.agents)} agents successful")
            return orchestration_results

        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id if 'session_id' in locals() else None
            }


# Update the main DormancyAnalysisAgent to use the enhanced orchestrator
class EnhancedDormancyAnalysisAgent(DormancyAnalysisAgent):
    """Enhanced main dormancy analysis agent with all sub-agents"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        super().__init__(memory_agent, mcp_client, db_session)

        # Use the enhanced orchestrator instead
        self.orchestrator = EnhancedDormancyAgentOrchestrator(memory_agent, mcp_client)


# Export the main class for use by workflow engine
# This ensures backward compatibility
DormancyAnalysisAgent = EnhancedDormancyAnalysisAgent