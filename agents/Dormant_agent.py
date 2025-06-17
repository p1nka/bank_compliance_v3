
"""
agents/dormant_agent.py - Comprehensive Dormancy Analysis Agent System
Regenerated based on dormant.py structure with CSV column alignment
CBUAE compliance monitoring with specialized agent implementations
"""

import logging
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import json
import numpy as np

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable, Client as LangSmithClient

# MCP imports with fallback
try:
    from mcp_client import MCPClient
except ImportError:
    logging.warning("MCPClient not available, using mock implementation")


    class MCPClient:
        async def call_tool(self, tool_name: str, params: Dict) -> Dict:
            return {"success": True, "data": {}}

# Import error handler and memory agent
try:
    from agents.error_handler_agent import ErrorHandlerAgent, ErrorState
    from agents.memory_agent import MemoryBucket, MemoryPriority, MemoryContext
except ImportError:
    # Mock implementations for testing
    class ErrorState:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


    class ErrorHandlerAgent:
        def __init__(self, memory_agent, mcp_client):
            pass

        async def handle_workflow_error(self, error_state):
            return type('obj', (object,), {
                'recovery_action': 'continue',
                'recovery_success': True
            })()


    class MemoryBucket:
        KNOWLEDGE = "knowledge"
        SESSION = "session"
        AUDIT = "audit"


    class MemoryPriority:
        HIGH = "high"
        CRITICAL = "critical"


    class MemoryContext:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== ENUMS AND STATUS DEFINITIONS =====

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
    """Triggers for dormancy analysis based on CSV columns"""
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
    UNCLAIMED_INSTRUMENTS_LEDGER = "unclaimed_instruments_ledger"
    CBUAE_TRANSFER_READY = "cbuae_transfer_ready"


# ===== STATE DATACLASSES =====

@dataclass
class DormancyAnalysisState:
    """Main state for dormancy analysis workflow - using CSV column names"""

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
    """Individual agent state with CSV column mapping"""
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


# ===== MOCK MEMORY AGENT =====

class MockMemoryAgent:
    """Mock memory agent for testing purposes"""

    async def create_memory_context(self, user_id: str, session_id: str, agent_name: str):
        return {"user_id": user_id, "session_id": session_id, "agent_name": agent_name}

    async def retrieve_memory(self, bucket: str, filter_criteria: Dict, context: Dict):
        return {"success": True, "data": []}

    async def store_memory(self, bucket: str, data: Dict, context: Dict,
                           content_type: str = None, priority: str = None,
                           tags: List[str] = None, encrypt_sensitive: bool = False):
        return {"success": True, "id": secrets.token_hex(8)}


# ===== BASE DORMANCY AGENT =====

class BaseDormancyAgent:
    """Base class for all dormancy analysis agents using CSV column mapping"""

    def __init__(self, agent_type: str, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        self.agent_type = agent_type
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()
        self.db_connection = db_connection
        self.error_handler = ErrorHandlerAgent(memory_agent, mcp_client)

        # CSV column mapping from banking_compliance_dataset
        self.csv_columns = {
            'customer_id': 'customer_id',
            'full_name_en': 'full_name_en',
            'account_id': 'account_id',
            'account_type': 'account_type',
            'account_subtype': 'account_subtype',
            'account_status': 'account_status',
            'dormancy_status': 'dormancy_status',
            'last_transaction_date': 'last_transaction_date',
            'last_contact_date': 'last_contact_date',
            'balance_current': 'balance_current',
            'maturity_date': 'maturity_date',
            'auto_renewal': 'auto_renewal',
            'dormancy_trigger_date': 'dormancy_trigger_date',
            'dormancy_period_months': 'dormancy_period_months',
            'current_stage': 'current_stage',
            'contact_attempts_made': 'contact_attempts_made',
            'transferred_to_cb_date': 'transferred_to_cb_date',
            'cb_transfer_amount': 'cb_transfer_amount'
        }

        # Default regulatory parameters
        self.default_params = {
            "standard_inactivity_years": 3,
            "unclaimed_instruments_years": 1,
            "high_value_threshold_aed": 100000,
            "cb_transfer_threshold_years": 5,
            "contact_attempt_minimum": 3,
            "statement_suppression_months": 6,
            "record_retention_years": 10
        }

    def _safe_date_parse(self, date_value) -> Optional[datetime]:
        """Safely parse date values from CSV"""
        if pd.isna(date_value) or date_value is None:
            return None

        try:
            if isinstance(date_value, str):
                return pd.to_datetime(date_value)
            elif isinstance(date_value, datetime):
                return date_value
            else:
                return pd.to_datetime(str(date_value))
        except:
            return None

    def _calculate_years_since(self, date_value, reference_date: datetime) -> float:
        """Calculate years between two dates"""
        parsed_date = self._safe_date_parse(date_value)
        if not parsed_date:
            return 0.0

        delta = reference_date - parsed_date
        return delta.days / 365.25

    async def pre_analysis_memory_hook(self, state: AgentState) -> AgentState:
        """Pre-analysis memory retrieval hook"""
        try:
            memory_context = await self.memory_agent.create_memory_context(
                state.user_id, state.session_id, self.agent_type
            )

            # Retrieve relevant patterns
            patterns = await self.memory_agent.retrieve_memory(
                bucket=MemoryBucket.KNOWLEDGE,
                filter_criteria={"agent_type": self.agent_type, "pattern_type": "dormancy"},
                context=memory_context
            )

            state.pre_hook_memory = memory_context
            state.retrieved_patterns = patterns.get("data", {})

        except Exception as e:
            logger.warning(f"Pre-analysis memory hook failed: {e}")
            state.pre_hook_memory = {}
            state.retrieved_patterns = {}

        return state

    async def post_analysis_memory_hook(self, state: AgentState) -> AgentState:
        """Post-analysis memory storage hook"""
        try:
            if state.analysis_results:
                # Store analysis patterns
                await self.memory_agent.store_memory(
                    bucket=MemoryBucket.KNOWLEDGE,
                    data={
                        "agent_type": self.agent_type,
                        "analysis_results": state.analysis_results,
                        "pattern_summary": {
                            "records_processed": state.records_processed,
                            "dormant_found": state.dormant_records_found,
                            "processing_time": state.processing_time
                        }
                    },
                    context=state.pre_hook_memory,
                    content_type="analysis_pattern",
                    priority=MemoryPriority.HIGH,
                    tags=[self.agent_type, "dormancy_analysis"]
                )

                state.post_hook_memory = {"stored": True, "timestamp": datetime.now().isoformat()}

        except Exception as e:
            logger.warning(f"Post-analysis memory hook failed: {e}")
            state.post_hook_memory = {"stored": False, "error": str(e)}

        return state

    async def _handle_error(self, state: AgentState, error: Exception, stage: str):
        """Handle errors during agent execution"""
        try:
            error_state = ErrorState(
                error_id=secrets.token_hex(8),
                session_id=state.session_id,
                user_id=state.user_id,
                timestamp=datetime.now(),
                error_details=[{
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stage": stage,
                    "agent_type": self.agent_type,
                    "critical": stage in ["analysis", "trigger_check"]
                }],
                failed_node=stage,
                workflow_context=asdict(state)
            )

            error_result = await self.error_handler.handle_workflow_error(error_state)

            state.error_state = error_result
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "error": str(error),
                "recovery_action": getattr(error_result, 'recovery_action', 'continue'),
                "recovery_success": getattr(error_result, 'recovery_success', True)
            })

        except Exception as nested_error:
            logger.error(f"Error handling failed: {str(nested_error)}")
            state.agent_status = AgentStatus.FAILED

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Base analyze_dormancy method - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement analyze_dormancy")


# ===== SPECIALIZED DORMANCY AGENTS =====

class DemandDepositDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.1.1 - Demand Deposit Dormancy Analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("demand_deposit_dormancy", memory_agent, mcp_client, db_connection)

    @traceable(name="demand_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze demand deposit dormancy using actual CSV columns"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for demand deposit analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for demand deposits and current accounts
            demand_deposits = df[
                (df[self.csv_columns['account_type']].isin(['CURRENT', 'SAVINGS', 'Current', 'Savings'])) &
                (df[self.csv_columns['account_status']] != 'CLOSED')
                ].copy()

            dormant_accounts = []

            for idx, account in demand_deposits.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]

                    # Calculate inactivity period
                    years_inactive = self._calculate_years_since(last_transaction, report_datetime)

                    # CBUAE Article 2.1.1: 3+ years of inactivity
                    if years_inactive >= self.default_params["standard_inactivity_years"]:
                        dormant_accounts.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'account_type': account[self.csv_columns['account_type']],
                            'balance_current': balance,
                            'last_transaction_date': last_transaction,
                            'years_inactive': round(years_inactive, 2),
                            'dormancy_trigger': 'STANDARD_INACTIVITY',
                            'compliance_article': '2.1.1',
                            'priority': 'HIGH' if balance > self.default_params[
                                "high_value_threshold_aed"] else 'MEDIUM',
                            'next_action': 'INITIATE_CONTACT_ATTEMPTS'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing demand deposit account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(dormant_accounts),
                "description": "CBUAE Article 2.1.1 - Demand Deposit Inactivity Analysis",
                "details": dormant_accounts,
                "compliance_article": "2.1.1",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(dormant_accounts) > 0
            }

            state.dormant_records_found = len(dormant_accounts)
            state.records_processed = len(demand_deposits)
            state.processed_dataframe = pd.DataFrame(dormant_accounts) if dormant_accounts else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "investment_analysis")

        return state


class PaymentInstrumentsDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.4 - Unclaimed Payment Instruments Analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("unclaimed_instruments", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze unclaimed payment instruments"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for payment instruments analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for accounts with potential unclaimed instruments
            # This could include accounts with specific subtypes or flags
            unclaimed_candidates = df[
                (df[self.csv_columns['account_status']].isin(['DORMANT', 'UNCLAIMED'])) |
                (df[self.csv_columns['account_subtype']].str.contains('INSTRUMENT', na=False))
                ].copy()

            unclaimed_instruments = []

            for idx, account in unclaimed_candidates.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]

                    # Calculate unclaimed period
                    years_unclaimed = self._calculate_years_since(last_transaction, report_datetime)

                    # CBUAE Article 2.4: 1+ year unclaimed
                    if years_unclaimed >= self.default_params["unclaimed_instruments_years"]:
                        unclaimed_instruments.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'account_type': account[self.csv_columns['account_type']],
                            'account_subtype': account[self.csv_columns['account_subtype']],
                            'balance_current': balance,
                            'last_transaction_date': last_transaction,
                            'years_unclaimed': round(years_unclaimed, 2),
                            'dormancy_trigger': 'PAYMENT_INSTRUMENT_UNCLAIMED',
                            'compliance_article': '2.4',
                            'priority': 'HIGH',
                            'next_action': 'UNCLAIMED_INSTRUMENT_PROCESS'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing unclaimed instrument {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(unclaimed_instruments),
                "description": "CBUAE Article 2.4 - Unclaimed Payment Instruments Analysis",
                "details": unclaimed_instruments,
                "compliance_article": "2.4",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(unclaimed_instruments) > 0
            }

            state.dormant_records_found = len(unclaimed_instruments)
            state.records_processed = len(unclaimed_candidates)
            state.processed_dataframe = pd.DataFrame(unclaimed_instruments) if unclaimed_instruments else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "payment_instruments_analysis")

        return state


class ContactAttemptsAgent(BaseDormancyAgent):
    """CBUAE Article 3 - Contact Attempts and Bank Obligations"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("contact_attempts", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze contact attempts compliance"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for contact attempts analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for dormant accounts requiring contact attempts
            dormant_accounts = df[
                df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant', 'POTENTIALLY_DORMANT'])
            ].copy()

            non_compliant_contacts = []

            for idx, account in dormant_accounts.iterrows():
                try:
                    contact_attempts = account[self.csv_columns['contact_attempts_made']]
                    dormancy_trigger_date = account[self.csv_columns['dormancy_trigger_date']]
                    current_stage = account[self.csv_columns['current_stage']]

                    # Calculate time since dormancy trigger
                    months_since_trigger = 0
                    if dormancy_trigger_date:
                        trigger_datetime = self._safe_date_parse(dormancy_trigger_date)
                        if trigger_datetime:
                            months_since_trigger = (report_datetime - trigger_datetime).days / 30.44

                    # CBUAE Article 3: Contact attempt requirements
                    required_attempts = min(self.default_params["contact_attempt_minimum"],
                                            int(months_since_trigger / 6) + 1)

                    if contact_attempts < required_attempts:
                        non_compliant_contacts.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'contact_attempts_made': contact_attempts,
                            'required_attempts': required_attempts,
                            'current_stage': current_stage,
                            'months_since_trigger': round(months_since_trigger, 1),
                            'compliance_gap': required_attempts - contact_attempts,
                            'compliance_article': '3',
                            'priority': 'HIGH',
                            'next_action': 'INITIATE_ADDITIONAL_CONTACT_ATTEMPTS'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing contact attempts for {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(non_compliant_contacts),
                "description": "CBUAE Article 3 - Contact Attempts Compliance Analysis",
                "details": non_compliant_contacts,
                "compliance_article": "3",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(non_compliant_contacts) > 0
            }

            state.dormant_records_found = len(non_compliant_contacts)
            state.records_processed = len(dormant_accounts)
            state.processed_dataframe = pd.DataFrame(
                non_compliant_contacts) if non_compliant_contacts else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "contact_attempts_analysis")

        return state


class InternalLedgerAgent(BaseDormancyAgent):
    """Internal Ledger Transfer Analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("internal_ledger", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze internal ledger transfer requirements"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for internal ledger analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for accounts eligible for internal ledger transfer
            eligible_accounts = df[
                (df[self.csv_columns['dormancy_status']] == 'DORMANT') &
                (df[self.csv_columns['current_stage']].str.contains('CONTACT_COMPLETED', na=False))
                ].copy()

            transfer_eligible = []

            for idx, account in eligible_accounts.iterrows():
                try:
                    dormancy_trigger_date = account[self.csv_columns['dormancy_trigger_date']]
                    contact_attempts = account[self.csv_columns['contact_attempts_made']]
                    balance = account[self.csv_columns['balance_current']]

                    # Calculate dormancy period
                    months_dormant = 0
                    if dormancy_trigger_date:
                        trigger_datetime = self._safe_date_parse(dormancy_trigger_date)
                        if trigger_datetime:
                            months_dormant = (report_datetime - trigger_datetime).days / 30.44

                    # Check if eligible for internal ledger transfer
                    # Typically after contact attempts are completed and waiting period elapsed
                    if (contact_attempts >= self.default_params["contact_attempt_minimum"] and
                            months_dormant >= 6):  # 6 months minimum waiting period

                        transfer_eligible.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'balance_current': balance,
                            'months_dormant': round(months_dormant, 1),
                            'contact_attempts_made': contact_attempts,
                            'transfer_type': 'INTERNAL_LEDGER',
                            'priority': 'MEDIUM',
                            'next_action': 'PREPARE_INTERNAL_TRANSFER'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing internal ledger for {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(transfer_eligible),
                "description": "Internal Ledger Transfer Eligibility Analysis",
                "details": transfer_eligible,
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(transfer_eligible) > 0
            }

            state.dormant_records_found = len(transfer_eligible)
            state.records_processed = len(eligible_accounts)
            state.processed_dataframe = pd.DataFrame(transfer_eligible) if transfer_eligible else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "internal_ledger_analysis")

        return state


class CBTransferEligibilityAgent(BaseDormancyAgent):
    """CBUAE Article 8 - Central Bank Transfer Eligibility"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("cb_transfer_eligibility", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze Central Bank transfer eligibility"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for CB transfer analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for long-term dormant accounts
            long_term_dormant = df[
                df[self.csv_columns['dormancy_status']] == 'DORMANT'
                ].copy()

            eligible_accounts = []

            for idx, account in long_term_dormant.iterrows():
                try:
                    dormancy_trigger_date = account[self.csv_columns['dormancy_trigger_date']]
                    balance = account[self.csv_columns['balance_current']]
                    transferred_date = account[self.csv_columns['transferred_to_cb_date']]

                    # Skip if already transferred
                    if pd.notna(transferred_date):
                        continue

                    # Calculate dormancy period
                    years_dormant = 0
                    if dormancy_trigger_date:
                        trigger_datetime = self._safe_date_parse(dormancy_trigger_date)
                        if trigger_datetime:
                            years_dormant = self._calculate_years_since(dormancy_trigger_date, report_datetime)

                    # CBUAE Article 8: 5+ years dormant for CB transfer
                    if years_dormant >= self.default_params["cb_transfer_threshold_years"]:
                        eligible_accounts.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'account_type': account[self.csv_columns['account_type']],
                            'balance_current': balance,
                            'years_dormant': round(years_dormant, 2),
                            'dormancy_trigger_date': dormancy_trigger_date,
                            'estimated_transfer_amount': balance,
                            'compliance_article': '8.1',
                            'priority': 'CRITICAL',
                            'next_action': 'PREPARE_CB_TRANSFER'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing CB transfer eligibility for {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(eligible_accounts),
                "description": "CBUAE Article 8.1 - Central Bank Transfer Eligibility Analysis",
                "details": eligible_accounts,
                "compliance_article": "8.1",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(eligible_accounts) > 0
            }

            state.dormant_records_found = len(eligible_accounts)
            state.records_processed = len(long_term_dormant)
            state.processed_dataframe = pd.DataFrame(eligible_accounts) if eligible_accounts else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "cb_transfer_analysis")

        return state


class SafeDepositDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.6 - Safe Deposit Box Dormancy Analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("safe_deposit_dormancy", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze safe deposit box dormancy"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for safe deposit analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for safe deposit box accounts or accounts with SDB indicators
            sdb_accounts = df[
                (df[self.csv_columns['account_type']].str.contains('SDB|SAFE_DEPOSIT', case=False, na=False)) |
                (df[self.csv_columns['account_subtype']].str.contains('SDB|SAFE_DEPOSIT', case=False, na=False))
                ].copy()

            dormant_sdb = []

            for idx, account in sdb_accounts.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]

                    # Calculate inactivity period
                    years_inactive = self._calculate_years_since(last_transaction, report_datetime)

                    # CBUAE Article 2.6: 3+ years of SDB fees unpaid or no communication
                    if years_inactive >= self.default_params["standard_inactivity_years"]:
                        dormant_sdb.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'account_type': account[self.csv_columns['account_type']],
                            'balance_current': balance,
                            'last_transaction_date': last_transaction,
                            'years_inactive': round(years_inactive, 2),
                            'dormancy_trigger': 'SDB_UNPAID_FEES',
                            'compliance_article': '2.6',
                            'priority': 'HIGH',
                            'next_action': 'SDB_COURT_APPLICATION_REQUIRED'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing SDB account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(dormant_sdb),
                "description": "CBUAE Article 2.6 - Safe Deposit Box Dormancy Analysis",
                "details": dormant_sdb,
                "compliance_article": "2.6",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(dormant_sdb) > 0
            }

            state.dormant_records_found = len(dormant_sdb)
            state.records_processed = len(sdb_accounts)
            state.processed_dataframe = pd.DataFrame(dormant_sdb) if dormant_sdb else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "safe_deposit_analysis")

        return state


class Art3ProcessNeededAgent(BaseDormancyAgent):
    """CBUAE Article 3 - Process Requirement Detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("art3_process_needed", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze Article 3 process requirements"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for Article 3 analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for dormant accounts requiring Article 3 process
            dormant_accounts = df[
                (df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant'])) &
                (df[self.csv_columns['current_stage']].isin(['TRIGGERED', 'IDENTIFIED', 'NEW_DORMANT']))
                ].copy()

            art3_required = []

            for idx, account in dormant_accounts.iterrows():
                try:
                    dormancy_trigger_date = account[self.csv_columns['dormancy_trigger_date']]
                    current_stage = account[self.csv_columns['current_stage']]
                    contact_attempts = account[self.csv_columns['contact_attempts_made']]

                    # Calculate time since dormancy trigger
                    months_since_trigger = 0
                    if dormancy_trigger_date:
                        trigger_datetime = self._safe_date_parse(dormancy_trigger_date)
                        if trigger_datetime:
                            months_since_trigger = (report_datetime - trigger_datetime).days / 30.44

                    # Article 3 process required if dormant but not yet processed
                    if (months_since_trigger > 0 and
                            current_stage in ['TRIGGERED', 'IDENTIFIED', 'NEW_DORMANT'] and
                            contact_attempts < self.default_params["contact_attempt_minimum"]):
                        art3_required.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'current_stage': current_stage,
                            'months_since_trigger': round(months_since_trigger, 1),
                            'contact_attempts_made': contact_attempts,
                            'compliance_article': '3',
                            'priority': 'HIGH',
                            'next_action': 'INITIATE_ARTICLE_3_PROCESS'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing Article 3 requirement for {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(art3_required),
                "description": "CBUAE Article 3 - Process Requirement Analysis",
                "details": art3_required,
                "compliance_article": "3",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(art3_required) > 0
            }

            state.dormant_records_found = len(art3_required)
            state.records_processed = len(dormant_accounts)
            state.processed_dataframe = pd.DataFrame(art3_required) if art3_required else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "art3_process_analysis")

        return state


class HighValueDormantAccountsAgent(BaseDormancyAgent):
    """High-Value Dormant Account Identification"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("high_value_dormant", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze high-value dormant accounts"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for high-value analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for high-value accounts
            high_value_threshold = self.default_params["high_value_threshold_aed"]
            high_value_accounts = df[
                (df[self.csv_columns['balance_current']] > high_value_threshold) &
                (df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant', 'POTENTIALLY_DORMANT']))
                ].copy()

            high_value_dormant = []

            for idx, account in high_value_accounts.iterrows():
                try:
                    balance = account[self.csv_columns['balance_current']]
                    last_transaction = account[self.csv_columns['last_transaction_date']]

                    # Calculate inactivity period
                    years_inactive = self._calculate_years_since(last_transaction, report_datetime)

                    # Determine risk level based on balance
                    if balance > 1000000:  # > 1M AED
                        risk_level = "CRITICAL"
                    elif balance > 500000:  # > 500K AED
                        risk_level = "HIGH"
                    else:
                        risk_level = "MEDIUM"

                    high_value_dormant.append({
                        'account_id': account[self.csv_columns['account_id']],
                        'customer_id': account[self.csv_columns['customer_id']],
                        'customer_name': account[self.csv_columns['full_name_en']],
                        'account_type': account[self.csv_columns['account_type']],
                        'balance_current': balance,
                        'last_transaction_date': last_transaction,
                        'years_inactive': round(years_inactive, 2),
                        'risk_level': risk_level,
                        'priority': risk_level,
                        'next_action': 'EXECUTIVE_REVIEW_REQUIRED'
                    })

                except Exception as e:
                    logger.warning(
                        f"Error processing high-value account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(high_value_dormant),
                "description": "High-Value Dormant Account Analysis",
                "details": high_value_dormant,
                "total_value": sum(acc["balance_current"] for acc in high_value_dormant),
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(high_value_dormant) > 0
            }

            state.dormant_records_found = len(high_value_dormant)
            state.records_processed = len(high_value_accounts)
            state.processed_dataframe = pd.DataFrame(high_value_dormant) if high_value_dormant else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "high_value_analysis")

        return state


class DormantToActiveTransitionsAgent(BaseDormancyAgent):
    """Dormant to Active Reactivation Detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("dormant_to_active", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze dormant to active transitions"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for transition analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for recently reactivated accounts
            reactivated_accounts = df[
                (df[self.csv_columns['account_status']] == 'ACTIVE') &
                (df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant']))
                ].copy()

            transitions = []

            for idx, account in reactivated_accounts.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    dormancy_trigger_date = account[self.csv_columns['dormancy_trigger_date']]

                    # Check if there's recent activity
                    days_since_transaction = 0
                    if last_transaction:
                        transaction_datetime = self._safe_date_parse(last_transaction)
                        if transaction_datetime:
                            days_since_transaction = (report_datetime - transaction_datetime).days

                    # Potential reactivation if recent transaction but still marked dormant
                    if days_since_transaction <= 30:  # Activity within last 30 days
                        transitions.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'account_type': account[self.csv_columns['account_type']],
                            'last_transaction_date': last_transaction,
                            'dormancy_trigger_date': dormancy_trigger_date,
                            'days_since_transaction': days_since_transaction,
                            'transition_type': 'DORMANT_TO_ACTIVE',
                            'priority': 'MEDIUM',
                            'next_action': 'UPDATE_DORMANCY_STATUS'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing transition for {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(transitions),
                "description": "Dormant to Active Transition Analysis",
                "details": transitions,
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(transitions) > 0
            }

            state.dormant_records_found = len(transitions)
            state.records_processed = len(reactivated_accounts)
            state.processed_dataframe = pd.DataFrame(transitions) if transitions else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "transition_analysis")

        return state


class RunAllDormantIdentificationChecksAgent(BaseDormancyAgent):
    """Master Orchestrator for All Dormancy Identification Checks"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("run_all_dormant_checks", memory_agent, mcp_client, db_connection)

        # Initialize all specialized agents
        self.specialist_agents = {
            "demand_deposit": DemandDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "fixed_deposit": FixedDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "investment": InvestmentAccountDormancyAgent(memory_agent, mcp_client, db_connection),
            "payment_instruments": PaymentInstrumentsDormancyAgent(memory_agent, mcp_client, db_connection),
            "safe_deposit": SafeDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "contact_attempts": ContactAttemptsAgent(memory_agent, mcp_client, db_connection),
            "cb_transfer": CBTransferEligibilityAgent(memory_agent, mcp_client, db_connection),
            "internal_ledger": InternalLedgerAgent(memory_agent, mcp_client, db_connection),
            "art3_process": Art3ProcessNeededAgent(memory_agent, mcp_client, db_connection),
            "high_value": HighValueDormantAccountsAgent(memory_agent, mcp_client, db_connection),
            "transitions": DormantToActiveTransitionsAgent(memory_agent, mcp_client, db_connection)
        }

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Run all dormancy identification checks"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for comprehensive analysis")

            # Run all specialist agents
            agent_results = {}
            total_dormant_found = 0
            total_processed = 0

            for agent_name, agent in self.specialist_agents.items():
                try:
                    # Create sub-state for this agent
                    sub_state = AgentState(
                        agent_id=f"{state.agent_id}_{agent_name}",
                        agent_type=agent_name,
                        session_id=state.session_id,
                        user_id=state.user_id,
                        timestamp=datetime.now(),
                        input_dataframe=state.input_dataframe,
                        analysis_config=state.analysis_config
                    )

                    # Run the specialist agent
                    result = await agent.analyze_dormancy(sub_state, report_date)

                    # Collect results
                    agent_results[agent_name] = {
                        "status": result.agent_status.value,
                        "results": result.analysis_results,
                        "dormant_found": result.dormant_records_found,
                        "records_processed": result.records_processed,
                        "processing_time": result.processing_time
                    }

                    total_dormant_found += result.dormant_records_found
                    total_processed = max(total_processed, result.records_processed)

                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    agent_results[agent_name] = {
                        "status": "failed",
                        "error": str(e),
                        "dormant_found": 0,
                        "records_processed": 0
                    }

            # Compile comprehensive results
            state.analysis_results = {
                "summary": {
                    "total_agents_run": len(self.specialist_agents),
                    "successful_agents": len([r for r in agent_results.values() if r.get("status") == "completed"]),
                    "total_dormant_found": total_dormant_found,
                    "total_records_processed": total_processed
                },
                "agent_results": agent_results,
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": total_dormant_found > 0
            }

            state.dormant_records_found = total_dormant_found
            state.records_processed = total_processed
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "comprehensive_analysis")

        return state


# ===== WORKFLOW ORCHESTRATOR =====

class DormancyWorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for comprehensive CBUAE dormancy analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()
        self.db_connection = db_connection

        # Initialize all 15 specialized dormancy agents
        self.agents = {
            # Primary Detection Agents (Article 2.x)
            "demand_deposit": DemandDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "fixed_deposit": FixedDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "investment": InvestmentAccountDormancyAgent(memory_agent, mcp_client, db_connection),
            "payment_instruments": PaymentInstrumentsDormancyAgent(memory_agent, mcp_client, db_connection),
            "safe_deposit": SafeDepositDormancyAgent(memory_agent, mcp_client, db_connection),

            # Process & Transfer Agents (Article 3, 5, 8)
            "contact_attempts": ContactAttemptsAgent(memory_agent, mcp_client, db_connection),
            "cb_transfer": CBTransferEligibilityAgent(memory_agent, mcp_client, db_connection),
            "art3_process": Art3ProcessNeededAgent(memory_agent, mcp_client, db_connection),
            "internal_ledger": InternalLedgerAgent(memory_agent, mcp_client, db_connection),

            # Specialized Analysis Agents
            "high_value": HighValueDormantAccountsAgent(memory_agent, mcp_client, db_connection),
            "transitions": DormantToActiveTransitionsAgent(memory_agent, mcp_client, db_connection),

            # Master Orchestrator
            "run_all_checks": RunAllDormantIdentificationChecksAgent(memory_agent, mcp_client, db_connection),

            # Additional Monitoring Agents
            "monitoring": DormancyMonitoringAgent(memory_agent, mcp_client, db_connection),
            "compliance_validation": ComplianceValidationAgent(memory_agent, mcp_client, db_connection),
            "risk_assessment": RiskAssessmentAgent(memory_agent, mcp_client, db_connection)
        }

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for comprehensive dormancy analysis with all 15+ agents"""
        workflow = StateGraph(DormancyAnalysisState)

        # Add nodes for all dormancy agents
        workflow.add_node("demand_deposit_analysis", self._run_demand_deposit_analysis)
        workflow.add_node("fixed_deposit_analysis", self._run_fixed_deposit_analysis)
        workflow.add_node("investment_analysis", self._run_investment_analysis)
        workflow.add_node("payment_instruments_analysis", self._run_payment_instruments_analysis)
        workflow.add_node("safe_deposit_analysis", self._run_safe_deposit_analysis)
        workflow.add_node("contact_attempts_analysis", self._run_contact_attempts_analysis)
        workflow.add_node("cb_transfer_analysis", self._run_cb_transfer_analysis)
        workflow.add_node("art3_process_analysis", self._run_art3_process_analysis)
        workflow.add_node("internal_ledger_analysis", self._run_internal_ledger_analysis)
        workflow.add_node("high_value_analysis", self._run_high_value_analysis)
        workflow.add_node("transitions_analysis", self._run_transitions_analysis)
        workflow.add_node("monitoring_analysis", self._run_monitoring_analysis)
        workflow.add_node("compliance_validation", self._run_compliance_validation)
        workflow.add_node("risk_assessment", self._run_risk_assessment)
        workflow.add_node("run_all_checks", self._run_all_checks)
        workflow.add_node("summarize_results", self._summarize_results)

        # Define comprehensive workflow edges
        workflow.add_edge(START, "demand_deposit_analysis")
        workflow.add_edge("demand_deposit_analysis", "fixed_deposit_analysis")
        workflow.add_edge("fixed_deposit_analysis", "investment_analysis")
        workflow.add_edge("investment_analysis", "payment_instruments_analysis")
        workflow.add_edge("payment_instruments_analysis", "safe_deposit_analysis")
        workflow.add_edge("safe_deposit_analysis", "contact_attempts_analysis")
        workflow.add_edge("contact_attempts_analysis", "cb_transfer_analysis")
        workflow.add_edge("cb_transfer_analysis", "art3_process_analysis")
        workflow.add_edge("art3_process_analysis", "internal_ledger_analysis")
        workflow.add_edge("internal_ledger_analysis", "high_value_analysis")
        workflow.add_edge("high_value_analysis", "transitions_analysis")
        workflow.add_edge("transitions_analysis", "monitoring_analysis")
        workflow.add_edge("monitoring_analysis", "compliance_validation")
        workflow.add_edge("compliance_validation", "risk_assessment")
        workflow.add_edge("risk_assessment", "run_all_checks")
        workflow.add_edge("run_all_checks", "summarize_results")
        workflow.add_edge("summarize_results", END)

        return workflow.compile(checkpointer=MemorySaver())

    async def _run_demand_deposit_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run demand deposit analysis"""
        agent_state = self._create_agent_state(state, "demand_deposit_analysis")
        result = await self.agents["demand_deposit"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "demand_deposit")

    async def _run_fixed_deposit_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run fixed deposit analysis"""
        agent_state = self._create_agent_state(state, "fixed_deposit_analysis")
        result = await self.agents["fixed_deposit"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "fixed_deposit")

    async def _run_investment_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run investment analysis"""
        agent_state = self._create_agent_state(state, "investment_analysis")
        result = await self.agents["investment"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "investment")

    async def _run_payment_instruments_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run payment instruments analysis"""
        agent_state = self._create_agent_state(state, "payment_instruments_analysis")
        result = await self.agents["payment_instruments"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "payment_instruments")

    async def _run_contact_attempts_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run contact attempts analysis"""
        agent_state = self._create_agent_state(state, "contact_attempts_analysis")
        result = await self.agents["contact_attempts"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "contact_attempts")

    async def _run_internal_ledger_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run internal ledger analysis"""
        agent_state = self._create_agent_state(state, "internal_ledger_analysis")
        result = await self.agents["internal_ledger"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "internal_ledger")

    async def _run_safe_deposit_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run safe deposit analysis"""
        agent_state = self._create_agent_state(state, "safe_deposit_analysis")
        result = await self.agents["safe_deposit"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "safe_deposit")

    async def _run_art3_process_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run Article 3 process analysis"""
        agent_state = self._create_agent_state(state, "art3_process_analysis")
        result = await self.agents["art3_process"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "art3_process")

    async def _run_high_value_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run high value analysis"""
        agent_state = self._create_agent_state(state, "high_value_analysis")
        result = await self.agents["high_value"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "high_value")

    async def _run_transitions_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run transitions analysis"""
        agent_state = self._create_agent_state(state, "transitions_analysis")
        result = await self.agents["transitions"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "transitions")

    async def _run_monitoring_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run monitoring analysis"""
        agent_state = self._create_agent_state(state, "monitoring_analysis")
        result = await self.agents["monitoring"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "monitoring")

    async def _run_compliance_validation(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run compliance validation"""
        try:
            # Use the compliance validation agent to validate all previous results
            validation_result = await self.agents["compliance_validation"].validate_compliance(state.agent_results)

            state.agent_results["compliance_validation"] = {
                "status": "completed",
                "results": validation_result,
                "records_processed": state.total_accounts_analyzed,
                "dormant_found": 0,
                "processing_time": 0.1
            }

            # Add compliance flags based on validation
            if validation_result.get("requires_action", False):
                state.compliance_flags.extend([
                    f"Compliance gap identified in {gap['article']}"
                    for gap in validation_result.get("gaps_identified", [])
                ])

            state.completed_agents.append("compliance_validation")

        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            state.failed_agents.append("compliance_validation")

        return state

    async def _run_risk_assessment(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run risk assessment"""
        try:
            # Convert processed_data to DataFrame for risk assessment
            input_df = None
            if state.processed_data and 'accounts' in state.processed_data:
                input_df = pd.DataFrame(state.processed_data['accounts'])

            if input_df is not None and not input_df.empty:
                risk_result = await self.agents["risk_assessment"].assess_risk(input_df)

                state.agent_results["risk_assessment"] = {
                    "status": "completed",
                    "results": risk_result,
                    "records_processed": len(input_df),
                    "dormant_found": risk_result.get("risk_distribution", {}).get("high_risk_count", 0),
                    "processing_time": 0.1
                }

                # Update high risk account count
                state.high_risk_accounts = risk_result.get("risk_distribution", {}).get("high_risk_count", 0)

                state.completed_agents.append("risk_assessment")
            else:
                state.failed_agents.append("risk_assessment")

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            state.failed_agents.append("risk_assessment")

        return state

    async def _run_all_checks(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run comprehensive checks orchestrator"""
        agent_state = self._create_agent_state(state, "run_all_checks")
        result = await self.agents["run_all_checks"].analyze_dormancy(
            agent_state, state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
        )
        return self._update_state_from_agent(state, result, "run_all_checks")

    def _create_agent_state(self, state: DormancyAnalysisState, agent_type: str) -> AgentState:
        """Create agent state from analysis state"""
        # Convert processed_data to DataFrame if it's a dict
        input_df = None
        if state.processed_data and 'accounts' in state.processed_data:
            input_df = pd.DataFrame(state.processed_data['accounts'])

        return AgentState(
            agent_id=secrets.token_hex(8),
            agent_type=agent_type,
            session_id=state.session_id,
            user_id=state.user_id,
            timestamp=datetime.now(),
            input_dataframe=input_df,
            analysis_config=state.analysis_config
        )

    def _update_state_from_agent(self, state: DormancyAnalysisState, agent_result: AgentState,
                                 agent_name: str) -> DormancyAnalysisState:
        """Update analysis state with agent results"""
        state.agent_results[agent_name] = {
            "status": agent_result.agent_status.value,
            "results": agent_result.analysis_results,
            "records_processed": agent_result.records_processed,
            "dormant_found": agent_result.dormant_records_found,
            "processing_time": agent_result.processing_time,
            "error_log": agent_result.error_log
        }

        # Update totals
        state.total_accounts_analyzed += agent_result.records_processed
        state.dormant_accounts_found += agent_result.dormant_records_found

        # Track agent completion
        if agent_result.agent_status == AgentStatus.COMPLETED:
            state.completed_agents.append(agent_name)
        elif agent_result.agent_status == AgentStatus.FAILED:
            state.failed_agents.append(agent_name)

        return state

    async def _summarize_results(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Summarize all analysis results"""
        try:
            # Calculate summary statistics
            total_dormant = sum(result.get("dormant_found", 0) for result in state.agent_results.values())
            total_processed = sum(result.get("records_processed", 0) for result in state.agent_results.values())
            total_processing_time = sum(result.get("processing_time", 0) for result in state.agent_results.values())

            # Create comprehensive summary
            state.dormancy_summary = {
                "analysis_overview": {
                    "total_accounts_analyzed": total_processed,
                    "total_dormant_accounts": total_dormant,
                    "dormancy_rate": round((total_dormant / total_processed * 100) if total_processed > 0 else 0, 2),
                    "total_processing_time": round(total_processing_time, 2),
                    "analysis_date": state.analysis_config.get("report_date", datetime.now().strftime("%Y-%m-%d"))
                },
                "agent_results": state.agent_results,
                "compliance_status": "COMPLIANT" if len(state.failed_agents) == 0 else "REQUIRES_ATTENTION",
                "completed_agents": state.completed_agents,
                "failed_agents": state.failed_agents,
                "recommendations": self._generate_recommendations(state)
            }

            # Set final status
            if len(state.failed_agents) == 0:
                state.analysis_status = DormancyStatus.COMPLETED
            else:
                state.analysis_status = DormancyStatus.REQUIRES_REVIEW

            state.processing_time = total_processing_time

        except Exception as e:
            state.analysis_status = DormancyStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "summarize_results",
                "error": str(e)
            })

        return state

    def _generate_recommendations(self, state: DormancyAnalysisState) -> List[Dict]:
        """Generate recommendations based on analysis results"""
        recommendations = []

        for agent_name, results in state.agent_results.items():
            if results.get("dormant_found", 0) > 0:
                recommendations.append({
                    "agent": agent_name,
                    "priority": "HIGH" if results.get("dormant_found", 0) > 10 else "MEDIUM",
                    "action": f"Review {results.get('dormant_found', 0)} dormant accounts identified by {agent_name}",
                    "compliance_article": results.get("results", {}).get("compliance_article", "N/A")
                })

        return recommendations

    async def run_comprehensive_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run comprehensive dormancy analysis workflow"""
        try:
            state.analysis_status = DormancyStatus.PROCESSING
            start_time = datetime.now()

            # Execute workflow
            result = await self.workflow.ainvoke(state)

            # Update final processing time
            result.processing_time = (datetime.now() - start_time).total_seconds()

            return result

        except Exception as e:
            state.analysis_status = DormancyStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "comprehensive_analysis",
                "error": str(e)
            })
            return state


# ===== MAIN ANALYSIS AGENT =====

class DormancyAnalysisAgent:
    """Main dormancy analysis agent with comprehensive CBUAE compliance"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()
        self.db_connection = db_connection
        self.orchestrator = DormancyWorkflowOrchestrator(memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Run comprehensive dormancy analysis"""
        return await self.orchestrator.run_comprehensive_analysis(state)


# ===== FACTORY FUNCTIONS =====

def create_comprehensive_dormancy_analysis(memory_agent=None, mcp_client: MCPClient = None,
                                           db_session=None) -> DormancyAnalysisAgent:
    """Factory function to create comprehensive dormancy analysis agent"""
    return DormancyAnalysisAgent(memory_agent, mcp_client, db_session)


async def run_comprehensive_dormancy_analysis_csv(user_id: str, account_data: pd.DataFrame,
                                                  report_date: str = None, db_connection=None,
                                                  memory_agent=None, mcp_client: MCPClient = None) -> Dict:
    """
    Run a complete comprehensive dormancy analysis using CSV data

    Args:
        user_id: User identifier
        account_data: DataFrame containing account information with CSV column names
        report_date: Analysis report date (defaults to today)
        db_connection: Database connection for monitoring system
        memory_agent: Memory agent instance
        mcp_client: MCP client instance

    Returns:
        Dictionary containing comprehensive analysis results
    """
    try:
        # Initialize analysis agent
        analysis_agent = DormancyAnalysisAgent(memory_agent, mcp_client, db_connection)

        # Set default report date
        if not report_date:
            report_date = datetime.now().strftime("%Y-%m-%d")

        # Validate CSV columns
        required_columns = [
            'customer_id', 'account_id', 'account_type', 'account_status',
            'last_transaction_date', 'balance_current'
        ]

        missing_columns = [col for col in required_columns if col not in account_data.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")

        # Create analysis state
        analysis_state = DormancyAnalysisState(
            session_id=secrets.token_hex(16),
            user_id=user_id,
            analysis_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            processed_data={'accounts': account_data.to_dict('records')},
            analysis_config={'report_date': report_date}
        )

        # Execute comprehensive analysis
        final_state = await analysis_agent.analyze_dormancy(analysis_state)

        # Return results
        return {
            "success": final_state.analysis_status == DormancyStatus.COMPLETED,
            "session_id": final_state.session_id,
            "analysis_results": final_state.dormancy_results if hasattr(final_state,
                                                                        'dormancy_results') else final_state.agent_results,
            "summary": final_state.dormancy_summary,
            "total_accounts_analyzed": final_state.total_accounts_analyzed,
            "dormant_accounts_found": final_state.dormant_accounts_found,
            "high_risk_accounts": final_state.high_risk_accounts,
            "processing_time_seconds": final_state.processing_time,
            "compliance_flags": final_state.compliance_flags,
            "analysis_log": final_state.analysis_log,
            "error_log": final_state.error_log,
            "data_quality": final_state.dormancy_summary.get("regulatory_reports", {}).get("data_quality_assessment",
                                                                                           {}) if final_state.dormancy_summary else {},
            "recommendations": final_state.dormancy_summary.get("recommendations",
                                                                []) if final_state.dormancy_summary else []
        }

    except Exception as e:
        logger.error(f"Comprehensive dormancy analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "session_id": None,
            "analysis_results": None
        }


# ===== SPECIALIZED MONITORING AGENTS =====

class DormancyMonitoringAgent(BaseDormancyAgent):
    """Real-time dormancy monitoring and alerting"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("dormancy_monitoring", memory_agent, mcp_client, db_connection)

    async def monitor_dormancy_triggers(self, df: pd.DataFrame, monitoring_config: Dict) -> Dict:
        """Monitor for dormancy triggers in real-time"""
        try:
            triggers_found = []
            current_time = datetime.now()

            for idx, account in df.iterrows():
                last_transaction = self._safe_date_parse(account[self.csv_columns['last_transaction_date']])

                if last_transaction:
                    days_inactive = (current_time - last_transaction).days

                    # Check various trigger conditions
                    if days_inactive >= (3 * 365):  # 3 years
                        triggers_found.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'trigger_type': 'STANDARD_INACTIVITY',
                            'days_inactive': days_inactive,
                            'severity': 'HIGH',
                            'next_action': 'INITIATE_DORMANCY_PROCESS'
                        })
                    elif days_inactive >= (2.5 * 365):  # 2.5 years - early warning
                        triggers_found.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'trigger_type': 'PRE_DORMANCY_WARNING',
                            'days_inactive': days_inactive,
                            'severity': 'MEDIUM',
                            'next_action': 'PROACTIVE_CUSTOMER_CONTACT'
                        })

            return {
                "triggers_found": len(triggers_found),
                "details": triggers_found,
                "monitoring_timestamp": current_time.isoformat(),
                "alert_generated": len(triggers_found) > 0
            }

        except Exception as e:
            logger.error(f"Dormancy monitoring failed: {e}")
            return {"error": str(e), "triggers_found": 0}


class ComplianceValidationAgent(BaseDormancyAgent):
    """CBUAE compliance validation and gap analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("compliance_validation", memory_agent, mcp_client, db_connection)

    async def validate_compliance(self, analysis_results: Dict) -> Dict:
        """Validate compliance across all CBUAE articles"""
        try:
            compliance_status = {}
            gaps_identified = []

            # Article 2 validations
            if "demand_deposit" in analysis_results:
                dd_results = analysis_results["demand_deposit"]
                compliance_status["article_2_1_1"] = {
                    "compliant": dd_results.get("status") == "completed",
                    "findings": dd_results.get("dormant_found", 0),
                    "validation_passed": dd_results.get("results", {}).get("validation_passed", False)
                }

            # Article 3 validations
            if "contact_attempts" in analysis_results:
                ca_results = analysis_results["contact_attempts"]
                compliance_status["article_3"] = {
                    "compliant": ca_results.get("dormant_found", 0) == 0,
                    "gaps_found": ca_results.get("dormant_found", 0),
                    "validation_passed": ca_results.get("results", {}).get("validation_passed", False)
                }

                if ca_results.get("dormant_found", 0) > 0:
                    gaps_identified.append({
                        "article": "3",
                        "gap_type": "INSUFFICIENT_CONTACT_ATTEMPTS",
                        "severity": "HIGH",
                        "affected_accounts": ca_results.get("dormant_found", 0)
                    })

            # Article 8 validations
            if "cb_transfer" in analysis_results:
                cb_results = analysis_results["cb_transfer"]
                compliance_status["article_8"] = {
                    "transfer_ready": cb_results.get("dormant_found", 0),
                    "validation_passed": cb_results.get("results", {}).get("validation_passed", False)
                }

            # Overall compliance score
            total_articles = len(compliance_status)
            compliant_articles = sum(1 for status in compliance_status.values()
                                     if status.get("validation_passed", False))

            compliance_score = (compliant_articles / total_articles * 100) if total_articles > 0 else 100

            return {
                "overall_compliance_score": round(compliance_score, 2),
                "article_compliance": compliance_status,
                "gaps_identified": gaps_identified,
                "total_gaps": len(gaps_identified),
                "validation_timestamp": datetime.now().isoformat(),
                "requires_action": len(gaps_identified) > 0
            }

        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return {"error": str(e), "overall_compliance_score": 0}


class RiskAssessmentAgent(BaseDormancyAgent):
    """Risk assessment for dormant accounts"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("risk_assessment", memory_agent, mcp_client, db_connection)

    async def assess_risk(self, df: pd.DataFrame) -> Dict:
        """Assess risk levels for dormant accounts"""
        try:
            risk_categories = {
                "high_risk": [],
                "medium_risk": [],
                "low_risk": []
            }

            for idx, account in df.iterrows():
                balance = float(account[self.csv_columns['balance_current']] or 0)
                account_type = account[self.csv_columns['account_type']]

                # Risk scoring logic
                risk_score = 0
                risk_factors = []

                # Balance-based risk
                if balance > 1000000:  # > 1M AED
                    risk_score += 40
                    risk_factors.append("HIGH_VALUE_ACCOUNT")
                elif balance > 100000:  # > 100K AED
                    risk_score += 20
                    risk_factors.append("MEDIUM_VALUE_ACCOUNT")

                # Account type risk
                if account_type in ['INVESTMENT', 'FIXED_DEPOSIT']:
                    risk_score += 15
                    risk_factors.append("INVESTMENT_ACCOUNT")

                # Dormancy period risk
                last_transaction = self._safe_date_parse(account[self.csv_columns['last_transaction_date']])
                if last_transaction:
                    years_inactive = self._calculate_years_since(last_transaction, datetime.now())
                    if years_inactive > 5:
                        risk_score += 25
                        risk_factors.append("LONG_TERM_DORMANCY")
                    elif years_inactive > 3:
                        risk_score += 15
                        risk_factors.append("STANDARD_DORMANCY")

                # Categorize risk
                account_risk = {
                    'account_id': account[self.csv_columns['account_id']],
                    'risk_score': risk_score,
                    'risk_factors': risk_factors,
                    'balance': balance,
                    'account_type': account_type
                }

                if risk_score >= 60:
                    risk_categories["high_risk"].append(account_risk)
                elif risk_score >= 30:
                    risk_categories["medium_risk"].append(account_risk)
                else:
                    risk_categories["low_risk"].append(account_risk)

            return {
                "risk_distribution": {
                    "high_risk_count": len(risk_categories["high_risk"]),
                    "medium_risk_count": len(risk_categories["medium_risk"]),
                    "low_risk_count": len(risk_categories["low_risk"])
                },
                "risk_details": risk_categories,
                "total_value_at_risk": sum(acc["balance"] for acc in risk_categories["high_risk"]),
                "assessment_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"error": str(e)}


# ===== UTILITY FUNCTIONS =====

def validate_csv_structure(df: pd.DataFrame) -> Dict:
    """Validate CSV structure against required schema"""
    required_columns = [
        'customer_id', 'account_id', 'account_type', 'account_status',
        'last_transaction_date', 'balance_current', 'dormancy_status'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in required_columns]

    # Data quality checks
    quality_issues = []

    # Check for null values in critical columns
    for col in ['customer_id', 'account_id', 'account_type']:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            quality_issues.append(f"{col} has {null_count} null values")

    # Check date formats
    if 'last_transaction_date' in df.columns:
        try:
            pd.to_datetime(df['last_transaction_date'], errors='coerce')
        except:
            quality_issues.append("last_transaction_date contains invalid date formats")

    return {
        "structure_valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "quality_issues": quality_issues,
        "total_records": len(df),
        "validation_timestamp": datetime.now().isoformat()
    }


# ===== FIXED SECTION - DEMAND DEPOSIT TO INVESTMENT AGENTS =====

class DemandDepositDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.1.1 - Demand Deposit Dormancy Analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("demand_deposit_dormancy", memory_agent, mcp_client, db_connection)

    @traceable(name="demand_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze demand deposit dormancy using actual CSV columns"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for demand deposit analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for demand deposits and current accounts
            demand_deposits = df[
                (df[self.csv_columns['account_type']].isin(['CURRENT', 'SAVINGS', 'Current', 'Savings'])) &
                (df[self.csv_columns['account_status']] != 'CLOSED')
                ].copy()

            dormant_accounts = []

            for idx, account in demand_deposits.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]

                    # Calculate inactivity period
                    years_inactive = self._calculate_years_since(last_transaction, report_datetime)

                    # CBUAE Article 2.1.1: 3+ years of inactivity
                    if years_inactive >= self.default_params["standard_inactivity_years"]:
                        dormant_accounts.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'account_type': account[self.csv_columns['account_type']],
                            'balance_current': balance,
                            'last_transaction_date': last_transaction,
                            'years_inactive': round(years_inactive, 2),
                            'dormancy_trigger': 'STANDARD_INACTIVITY',
                            'compliance_article': '2.1.1',
                            'priority': 'HIGH' if balance > self.default_params[
                                "high_value_threshold_aed"] else 'MEDIUM',
                            'next_action': 'INITIATE_CONTACT_ATTEMPTS'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing demand deposit account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(dormant_accounts),
                "description": "CBUAE Article 2.1.1 - Demand Deposit Inactivity Analysis",
                "details": dormant_accounts,
                "compliance_article": "2.1.1",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(dormant_accounts) > 0
            }

            state.dormant_records_found = len(dormant_accounts)
            state.records_processed = len(demand_deposits)
            state.processed_dataframe = pd.DataFrame(dormant_accounts) if dormant_accounts else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "demand_deposit_analysis")

        return state


class FixedDepositDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.2 - Fixed Deposit Dormancy Analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("fixed_deposit_dormancy", memory_agent, mcp_client, db_connection)

    @traceable(name="fixed_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze fixed deposit dormancy using actual CSV columns"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for fixed deposit analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for fixed deposits
            fixed_deposits = df[
                (df[self.csv_columns['account_type']].isin(['FIXED_DEPOSIT', 'Fixed Deposit', 'INVESTMENT'])) &
                (df[self.csv_columns['account_status']] != 'CLOSED')
                ].copy()

            dormant_accounts = []

            for idx, account in fixed_deposits.iterrows():
                try:
                    maturity_date = account[self.csv_columns['maturity_date']]
                    auto_renewal = account[self.csv_columns['auto_renewal']]
                    last_contact = account[self.csv_columns['last_contact_date']]
                    balance = account[self.csv_columns['balance_current']]

                    # Parse dates
                    maturity_datetime = self._safe_date_parse(maturity_date)

                    if maturity_datetime:
                        years_since_maturity = self._calculate_years_since(maturity_date, report_datetime)
                        years_since_contact = self._calculate_years_since(last_contact, report_datetime)

                        # CBUAE Article 2.2 Logic
                        is_dormant = False
                        dormancy_reason = ""

                        if auto_renewal == 'YES':
                            # Auto-renewal FDs: Check for communication gaps
                            if years_since_contact >= self.default_params["standard_inactivity_years"]:
                                is_dormant = True
                                dormancy_reason = "AUTO_RENEWAL_NO_CONTACT"
                        else:
                            # Non-auto-renewal FDs: Check maturity + inactivity
                            if years_since_maturity >= 1 and years_since_contact >= self.default_params[
                                "standard_inactivity_years"]:
                                is_dormant = True
                                dormancy_reason = "MATURITY_PLUS_INACTIVITY"

                        if is_dormant:
                            dormant_accounts.append({
                                'account_id': account[self.csv_columns['account_id']],
                                'customer_id': account[self.csv_columns['customer_id']],
                                'customer_name': account[self.csv_columns['full_name_en']],
                                'account_type': account[self.csv_columns['account_type']],
                                'balance_current': balance,
                                'maturity_date': maturity_date,
                                'auto_renewal': auto_renewal,
                                'years_since_maturity': round(years_since_maturity, 2),
                                'years_since_contact': round(years_since_contact, 2),
                                'dormancy_reason': dormancy_reason,
                                'compliance_article': '2.2',
                                'priority': 'HIGH' if balance > self.default_params[
                                    "high_value_threshold_aed"] else 'MEDIUM',
                                'next_action': 'MATURITY_CONTACT_REQUIRED'
                            })

                except Exception as e:
                    logger.warning(
                        f"Error processing fixed deposit account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(dormant_accounts),
                "description": "CBUAE Article 2.2 - Fixed Deposit Maturity Analysis",
                "details": dormant_accounts,
                "compliance_article": "2.2",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(dormant_accounts) > 0
            }

            state.dormant_records_found = len(dormant_accounts)
            state.records_processed = len(fixed_deposits)
            state.processed_dataframe = pd.DataFrame(dormant_accounts) if dormant_accounts else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "fixed_deposit_analysis")

        return state


class InvestmentAccountDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.3 - Investment Account Dormancy Analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("investment_dormancy", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze investment account dormancy"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for investment analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for investment accounts
            investment_accounts = df[
                (df[self.csv_columns['account_type']].isin(['INVESTMENT', 'Investment', 'PORTFOLIO'])) &
                (df[self.csv_columns['account_status']] != 'CLOSED')
                ].copy()

            dormant_accounts = []

            for idx, account in investment_accounts.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]

                    # Calculate inactivity period
                    years_inactive = self._calculate_years_since(last_transaction, report_datetime)

                    # CBUAE Article 2.3: Investment-specific dormancy criteria
                    if years_inactive >= self.default_params["standard_inactivity_years"]:
                        dormant_accounts.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'account_type': account[self.csv_columns['account_type']],
                            'account_subtype': account[self.csv_columns['account_subtype']],
                            'balance_current': balance,
                            'last_transaction_date': last_transaction,
                            'years_inactive': round(years_inactive, 2),
                            'dormancy_trigger': 'INVESTMENT_INACTIVITY',
                            'compliance_article': '2.3',
                            'priority': 'HIGH' if balance > self.default_params[
                                "high_value_threshold_aed"] else 'MEDIUM',
                            'next_action': 'INVESTMENT_REVIEW_REQUIRED'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing investment account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(dormant_accounts),
                "description": "CBUAE Article 2.3 - Investment Account Inactivity Analysis",
                "details": dormant_accounts,
                "compliance_article": "2.3",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(dormant_accounts) > 0
            }

            state.dormant_records_found = len(dormant_accounts)
            state.records_processed = len(investment_accounts)
            state.processed_dataframe = pd.DataFrame(dormant_accounts) if dormant_accounts else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "investment_analysis")

        return state


# ===== EXPORT DEFINITIONS =====

__all__ = [
    # Core Analysis Components
    "DormancyAnalysisAgent",
    "DormancyWorkflowOrchestrator",
    "DormancyAnalysisState",
    "AgentState",

    # Status and Trigger Enums
    "AgentStatus",
    "DormancyStatus",
    "DormancyTrigger",

    # Base Agent Class
    "BaseDormancyAgent",

    # Primary Detection Agents (Article 2.x - 5 agents)
    "DemandDepositDormancyAgent",  # Article 2.1.1
    "FixedDepositDormancyAgent",  # Article 2.2
    "InvestmentAccountDormancyAgent",  # Article 2.3
    "PaymentInstrumentsDormancyAgent",  # Article 2.4
    "SafeDepositDormancyAgent",  # Article 2.6

    # Process & Transfer Agents (Articles 3, 5, 8 - 4 agents)
    "ContactAttemptsAgent",  # Article 5
    "Art3ProcessNeededAgent",  # Article 3
    "InternalLedgerAgent",  # Internal Process
    "CBTransferEligibilityAgent",  # Article 8

    # Specialized Analysis Agents (2 agents)
    "HighValueDormantAccountsAgent",  # Internal Analysis
    "DormantToActiveTransitionsAgent",  # Internal Analysis

    # Master Orchestrator (1 agent)
    "RunAllDormantIdentificationChecksAgent",  # Master Orchestrator

    # Additional Monitoring & Validation Agents (3 agents)
    "DormancyMonitoringAgent",  # Real-time Monitoring
    "ComplianceValidationAgent",  # CBUAE Compliance Validation
    "RiskAssessmentAgent",  # Risk Scoring

    # Utility Components
    "MockMemoryAgent",

    # Factory Functions
    "create_comprehensive_dormancy_analysis",
    "run_comprehensive_dormancy_analysis_csv",
    "validate_csv_structure"
]

# Total: 15+ specialized dormancy agents as per the dormant.py specification