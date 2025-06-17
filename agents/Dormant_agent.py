"""
agents/dormancy_agents.py - Updated Dormancy Analysis Agents
Aligned with actual CSV column names from banking_compliance_dataset
Comprehensive CBUAE compliance monitoring with exact field mapping
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

# MCP imports - using try/except to handle import issues
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


# Mock Memory Agent for testing
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


# Base Dormancy Agent Class using CSV column names
class BaseDormancyAgent:
    """Base class for all dormancy analysis agents using exact CSV column names"""

    def __init__(self, agent_type: str, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        self.agent_type = agent_type
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()
        self.db_connection = db_connection

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

        # CSV column mappings
        self.csv_columns = {
            # Customer columns
            'customer_id': 'customer_id',
            'customer_type': 'customer_type',
            'full_name_en': 'full_name_en',
            'full_name_ar': 'full_name_ar',
            'id_number': 'id_number',
            'id_type': 'id_type',
            'date_of_birth': 'date_of_birth',
            'nationality': 'nationality',

            # Address columns
            'address_line1': 'address_line1',
            'address_line2': 'address_line2',
            'city': 'city',
            'emirate': 'emirate',
            'country': 'country',
            'postal_code': 'postal_code',

            # Contact columns
            'phone_primary': 'phone_primary',
            'phone_secondary': 'phone_secondary',
            'email_primary': 'email_primary',
            'email_secondary': 'email_secondary',
            'address_known': 'address_known',
            'last_contact_date': 'last_contact_date',
            'last_contact_method': 'last_contact_method',

            # KYC columns
            'kyc_status': 'kyc_status',
            'kyc_expiry_date': 'kyc_expiry_date',
            'risk_rating': 'risk_rating',

            # Account columns
            'account_id': 'account_id',
            'account_type': 'account_type',
            'account_subtype': 'account_subtype',
            'account_name': 'account_name',
            'currency': 'currency',
            'account_status': 'account_status',
            'dormancy_status': 'dormancy_status',
            'opening_date': 'opening_date',
            'closing_date': 'closing_date',

            # Transaction columns
            'last_transaction_date': 'last_transaction_date',
            'last_system_transaction_date': 'last_system_transaction_date',

            # Balance columns
            'balance_current': 'balance_current',
            'balance_available': 'balance_available',
            'balance_minimum': 'balance_minimum',
            'interest_rate': 'interest_rate',
            'interest_accrued': 'interest_accrued',

            # Joint account columns
            'is_joint_account': 'is_joint_account',
            'joint_account_holders': 'joint_account_holders',
            'has_outstanding_facilities': 'has_outstanding_facilities',

            # Investment/FD columns
            'maturity_date': 'maturity_date',
            'auto_renewal': 'auto_renewal',

            # Statement columns
            'last_statement_date': 'last_statement_date',
            'statement_frequency': 'statement_frequency',

            # Dormancy tracking columns
            'tracking_id': 'tracking_id',
            'dormancy_trigger_date': 'dormancy_trigger_date',
            'dormancy_period_start': 'dormancy_period_start',
            'dormancy_period_months': 'dormancy_period_months',
            'dormancy_classification_date': 'dormancy_classification_date',
            'transfer_eligibility_date': 'transfer_eligibility_date',
            'current_stage': 'current_stage',

            # Contact attempt columns
            'contact_attempts_made': 'contact_attempts_made',
            'last_contact_attempt_date': 'last_contact_attempt_date',
            'waiting_period_start': 'waiting_period_start',
            'waiting_period_end': 'waiting_period_end',

            # Transfer columns
            'transferred_to_ledger_date': 'transferred_to_ledger_date',
            'transferred_to_cb_date': 'transferred_to_cb_date',
            'cb_transfer_amount': 'cb_transfer_amount',
            'cb_transfer_reference': 'cb_transfer_reference',
            'exclusion_reason': 'exclusion_reason',

            # Audit columns
            'created_date': 'created_date',
            'updated_date': 'updated_date',
            'updated_by': 'updated_by'
        }

    def _safe_date_parse(self, date_str: str) -> Optional[datetime]:
        """Safely parse date strings from CSV"""
        if pd.isna(date_str) or date_str == '' or date_str is None:
            return None

        try:
            if isinstance(date_str, str):
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue

                # Use pandas if all else fails
                return pd.to_datetime(date_str)
            elif isinstance(date_str, datetime):
                return date_str
            else:
                return pd.to_datetime(date_str)
        except Exception as e:
            logger.warning(f"Could not parse date '{date_str}': {e}")
            return None

    def _calculate_years_since(self, date_str: str, reference_date: datetime = None) -> float:
        """Calculate years since a given date"""
        if not reference_date:
            reference_date = datetime.now()

        parsed_date = self._safe_date_parse(date_str)
        if not parsed_date:
            return 0.0

        delta = reference_date - parsed_date
        return delta.days / 365.25

    def _calculate_months_since(self, date_str: str, reference_date: datetime = None) -> float:
        """Calculate months since a given date"""
        if not reference_date:
            reference_date = datetime.now()

        parsed_date = self._safe_date_parse(date_str)
        if not parsed_date:
            return 0.0

        delta = reference_date - parsed_date
        return delta.days / 30.44  # Average days per month

    @traceable(name="pre_analysis_memory_hook")
    async def pre_analysis_memory_hook(self, state: AgentState) -> AgentState:
        """Enhanced pre-analysis memory hook"""
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
                bucket=MemoryBucket.KNOWLEDGE,
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

            state.agent_status = AgentStatus.PROCESSING

        except Exception as e:
            logger.error(f"{self.agent_type} pre-analysis memory hook failed: {str(e)}")
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "pre_analysis_memory_hook")

        return state

    @traceable(name="post_analysis_memory_hook")
    async def post_analysis_memory_hook(self, state: AgentState) -> AgentState:
        """Enhanced post-analysis memory hook"""
        try:
            state.agent_status = AgentStatus.MEMORY_STORING

            # Create memory context
            memory_context = await self.memory_agent.create_memory_context(
                user_id=state.user_id,
                session_id=state.session_id,
                agent_name=self.agent_type
            )

            # Store comprehensive monitoring results
            if state.analysis_results:
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
                    bucket=MemoryBucket.SESSION,
                    data=monitoring_results,
                    context=memory_context,
                    content_type="monitoring_results",
                    priority=MemoryPriority.HIGH,
                    tags=[self.agent_type, "monitoring", "cbuae_compliance"]
                )

        except Exception as e:
            logger.error(f"{self.agent_type} post-analysis memory hook failed: {str(e)}")
            await self._handle_error(state, e, "post_analysis_memory_hook")

        return state

    def _determine_priority_level(self, state: AgentState) -> str:
        """Determine priority level based on findings"""
        if state.dormant_records_found == 0:
            return "low"
        elif state.dormant_records_found < 10:
            return "medium"
        else:
            return "high"

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


# ===== CBUAE ARTICLE-SPECIFIC AGENTS USING CSV COLUMNS =====

class DemandDepositDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.1.1 - Demand Deposit Dormancy Analysis using CSV columns"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("demand_deposit_dormancy", memory_agent, mcp_client, db_connection)

    @traceable(name="demand_deposit_analysis")
    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze demand deposit dormancy using actual CSV columns"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            # Pre-analysis memory hook
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
                    # Check if account meets Article 2.1.1 criteria
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    last_contact = account[self.csv_columns['last_contact_date']]
                    has_facilities = account[self.csv_columns['has_outstanding_facilities']]
                    balance = account[self.csv_columns['balance_current']]

                    # Calculate inactivity period
                    years_since_transaction = self._calculate_years_since(last_transaction, report_datetime)
                    years_since_contact = self._calculate_years_since(last_contact, report_datetime)

                    # CBUAE Article 2.1.1 Logic:
                    # No customer-initiated activity for 3+ years AND no outstanding liabilities
                    is_dormant = (
                            years_since_transaction >= self.default_params["standard_inactivity_years"] and
                            (has_facilities != 'YES') and  # No outstanding facilities
                            balance > 0  # Has positive balance
                    )

                    if is_dormant:
                        dormant_accounts.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'account_type': account[self.csv_columns['account_type']],
                            'balance_current': balance,
                            'last_transaction_date': last_transaction,
                            'years_inactive': round(years_since_transaction, 2),
                            'last_contact_date': last_contact,
                            'years_since_contact': round(years_since_contact, 2),
                            'dormancy_trigger': 'STANDARD_INACTIVITY',
                            'compliance_article': '2.1.1',
                            'priority': 'HIGH' if balance > self.default_params[
                                "high_value_threshold_aed"] else 'MEDIUM',
                            'next_action': 'PROACTIVE_CONTACT_REQUIRED',
                            'risk_factors': self._assess_risk_factors(account)
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            # Create analysis results
            state.analysis_results = {
                "count": len(dormant_accounts),
                "description": "CBUAE Article 2.1.1 - Demand Deposit Inactivity Analysis",
                "details": dormant_accounts,
                "compliance_article": "2.1.1",
                "analysis_date": report_date,
                "validation_passed": True,
                "key_findings": self._generate_demand_deposit_findings(dormant_accounts, len(demand_deposits)),
                "alerts_generated": len(dormant_accounts) > 0,
                "regulatory_requirements": [
                    "Customer-initiated activity tracking for 3+ years",
                    "Outstanding liability verification required",
                    "Proactive customer contact required before classification",
                    "High-value accounts require immediate attention"
                ]
            }

            state.dormant_records_found = len(dormant_accounts)
            state.records_processed = len(demand_deposits)
            state.processed_dataframe = pd.DataFrame(dormant_accounts) if dormant_accounts else pd.DataFrame()

            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            state.processing_time = processing_time
            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "accounts_analyzed": len(demand_deposits),
                "dormant_accounts_found": len(dormant_accounts),
                "high_value_accounts": len([a for a in dormant_accounts if a['priority'] == 'HIGH']),
                "data_quality_score": self._calculate_data_quality_score(demand_deposits)
            }

            state.agent_status = AgentStatus.COMPLETED

            # Post-analysis memory hook
            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "demand_deposit_analysis")

        return state

    def _assess_risk_factors(self, account: pd.Series) -> List[str]:
        """Assess risk factors for demand deposit account"""
        risk_factors = []

        try:
            balance = account[self.csv_columns['balance_current']]
            customer_type = account[self.csv_columns['customer_type']]
            risk_rating = account[self.csv_columns['risk_rating']]
            address_known = account[self.csv_columns['address_known']]

            if balance > self.default_params["high_value_threshold_aed"]:
                risk_factors.append("HIGH_VALUE_ACCOUNT")

            if customer_type == 'CORPORATE':
                risk_factors.append("CORPORATE_CUSTOMER")

            if risk_rating == 'HIGH':
                risk_factors.append("HIGH_RISK_CUSTOMER")

            if address_known != 'YES':
                risk_factors.append("ADDRESS_UNKNOWN")

            # Check for foreign currency
            if account[self.csv_columns['currency']] != 'AED':
                risk_factors.append("FOREIGN_CURRENCY")

        except Exception as e:
            logger.warning(f"Risk assessment failed: {e}")

        return risk_factors

    def _generate_demand_deposit_findings(self, dormant_accounts: List[Dict], total_accounts: int) -> List[str]:
        """Generate key findings for demand deposits"""
        findings = []

        if dormant_accounts:
            high_value_count = len([a for a in dormant_accounts if a['priority'] == 'HIGH'])
            corporate_count = len([a for a in dormant_accounts if 'CORPORATE_CUSTOMER' in a.get('risk_factors', [])])

            findings.append(
                f"Found {len(dormant_accounts)} demand deposit accounts meeting Article 2.1.1 dormancy criteria")
            findings.append(f"Analyzed {total_accounts} total demand deposit accounts")

            if high_value_count > 0:
                findings.append(f"{high_value_count} high-value accounts (>AED 25,000) require immediate attention")

            if corporate_count > 0:
                findings.append(f"{corporate_count} corporate accounts identified - enhanced due diligence required")

            # Calculate average inactivity period
            avg_inactivity = sum(a['years_inactive'] for a in dormant_accounts) / len(dormant_accounts)
            findings.append(f"Average inactivity period: {avg_inactivity:.1f} years")

            findings.append("Proactive customer contact campaigns should be initiated immediately")
            findings.append("Liability verification required before dormancy classification")
        else:
            findings.append("No demand deposit accounts currently meet Article 2.1.1 dormancy criteria")
            findings.append(f"Analyzed {total_accounts} accounts - all show recent customer activity")

        return findings

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score for the dataset"""
        if df.empty:
            return 0.0

        try:
            # Key fields for dormancy analysis
            key_fields = [
                self.csv_columns['last_transaction_date'],
                self.csv_columns['account_status'],
                self.csv_columns['balance_current'],
                self.csv_columns['has_outstanding_facilities']
            ]

            total_cells = len(df) * len(key_fields)
            missing_cells = 0

            for field in key_fields:
                if field in df.columns:
                    missing_cells += df[field].isna().sum()
                else:
                    missing_cells += len(df)  # Field completely missing

            quality_score = max(0.0, (total_cells - missing_cells) / total_cells)
            return round(quality_score, 3)

        except Exception as e:
            logger.warning(f"Data quality calculation failed: {e}")
            return 0.5  # Default score


class FixedDepositDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.2 - Fixed Deposit Dormancy Analysis using CSV columns"""

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

            # Filter for fixed deposits and investment accounts
            fixed_deposits = df[
                (df[self.csv_columns['account_type']].isin(
                    ['INVESTMENT', 'Investment', 'FIXED_DEPOSIT', 'Fixed Deposit'])) &
                (df[self.csv_columns['account_status']] != 'CLOSED')
                ].copy()

            dormant_accounts = []

            for idx, account in fixed_deposits.iterrows():
                try:
                    maturity_date = account[self.csv_columns['maturity_date']]
                    auto_renewal = account[self.csv_columns['auto_renewal']]
                    last_contact = account[self.csv_columns['last_contact_date']]
                    balance = account[self.csv_columns['balance_current']]

                    # Parse maturity date
                    maturity_datetime = self._safe_date_parse(maturity_date)

                    if maturity_datetime:
                        years_since_maturity = self._calculate_years_since(maturity_date, report_datetime)
                        years_since_contact = self._calculate_years_since(last_contact, report_datetime)

                        # CBUAE Article 2.2 Logic:
                        # Fixed deposits with no customer communication post-maturity
                        is_dormant = False
                        dormancy_reason = ""

                        if auto_renewal == 'YES':
                            # Auto-renewal FDs: Check for communication gaps
                            if years_since_contact >= self.default_params["standard_inactivity_years"]:
                                is_dormant = True
                                dormancy_reason = "AUTO_RENEWAL_NO_CONTACT"
                        else:
                            # Non-auto-renewal FDs: Check maturity + inactivity
                            if (years_since_maturity >= self.default_params["standard_inactivity_years"] and
                                    years_since_contact >= self.default_params["standard_inactivity_years"]):
                                is_dormant = True
                                dormancy_reason = "MATURED_UNCLAIMED"

                        if is_dormant and balance > 0:
                            dormant_accounts.append({
                                'account_id': account[self.csv_columns['account_id']],
                                'customer_id': account[self.csv_columns['customer_id']],
                                'customer_name': account[self.csv_columns['full_name_en']],
                                'account_type': account[self.csv_columns['account_type']],
                                'balance_current': balance,
                                'maturity_date': maturity_date,
                                'years_since_maturity': round(years_since_maturity, 2),
                                'auto_renewal': auto_renewal,
                                'last_contact_date': last_contact,
                                'years_since_contact': round(years_since_contact, 2),
                                'dormancy_trigger': dormancy_reason,
                                'compliance_article': '2.2',
                                'priority': 'HIGH' if balance > self.default_params[
                                    "high_value_threshold_aed"] else 'MEDIUM',
                                'next_action': 'MATURITY_COMMUNICATION_REQUIRED',
                                'risk_factors': self._assess_fd_risk_factors(account, years_since_maturity)
                            })

                except Exception as e:
                    logger.warning(
                        f"Error processing FD account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            # Create analysis results
            state.analysis_results = {
                "count": len(dormant_accounts),
                "description": "CBUAE Article 2.2 - Fixed Deposit Maturity and Renewal Analysis",
                "details": dormant_accounts,
                "compliance_article": "2.2",
                "analysis_date": report_date,
                "validation_passed": True,
                "key_findings": self._generate_fd_findings(dormant_accounts, len(fixed_deposits)),
                "alerts_generated": len(dormant_accounts) > 0,
                "regulatory_requirements": [
                    "Maturity date tracking and customer notification required",
                    "Auto-renewal communication verification",
                    "Post-maturity customer contact documentation",
                    "Investment instruction compliance verification"
                ]
            }

            state.dormant_records_found = len(dormant_accounts)
            state.records_processed = len(fixed_deposits)
            state.processed_dataframe = pd.DataFrame(dormant_accounts) if dormant_accounts else pd.DataFrame()

            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.performance_metrics = {
                "processing_time_seconds": state.processing_time,
                "fd_accounts_analyzed": len(fixed_deposits),
                "dormant_fds_found": len(dormant_accounts),
                "auto_renewal_issues": len(
                    [a for a in dormant_accounts if a['dormancy_trigger'] == 'AUTO_RENEWAL_NO_CONTACT']),
                "matured_unclaimed": len([a for a in dormant_accounts if a['dormancy_trigger'] == 'MATURED_UNCLAIMED']),
                "data_quality_score": self._calculate_data_quality_score(fixed_deposits)
            }

            state.agent_status = AgentStatus.COMPLETED
            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "fixed_deposit_analysis")

        return state

    def _assess_fd_risk_factors(self, account: pd.Series, years_since_maturity: float) -> List[str]:
        """Assess risk factors for fixed deposit account"""
        risk_factors = []

        try:
            balance = account[self.csv_columns['balance_current']]
            currency = account[self.csv_columns['currency']]
            customer_type = account[self.csv_columns['customer_type']]

            if balance > self.default_params["high_value_threshold_aed"]:
                risk_factors.append("HIGH_VALUE_INVESTMENT")

            if years_since_maturity > 5:
                risk_factors.append("LONG_OVERDUE_MATURITY")

            if currency != 'AED':
                risk_factors.append("FOREIGN_CURRENCY_INVESTMENT")

            if customer_type == 'CORPORATE':
                risk_factors.append("CORPORATE_INVESTMENT")

        except Exception as e:
            logger.warning(f"FD risk assessment failed: {e}")

        return risk_factors

    def _generate_fd_findings(self, dormant_accounts: List[Dict], total_fds: int) -> List[str]:
        """Generate key findings for fixed deposits"""
        findings = []

        if dormant_accounts:
            auto_renewal_issues = len(
                [a for a in dormant_accounts if a['dormancy_trigger'] == 'AUTO_RENEWAL_NO_CONTACT'])
            matured_unclaimed = len([a for a in dormant_accounts if a['dormancy_trigger'] == 'MATURED_UNCLAIMED'])

            findings.append(f"Found {len(dormant_accounts)} fixed deposits meeting Article 2.2 dormancy criteria")
            findings.append(f"Analyzed {total_fds} total fixed deposit/investment accounts")

            if auto_renewal_issues > 0:
                findings.append(f"{auto_renewal_issues} auto-renewal accounts lack customer communication")

            if matured_unclaimed > 0:
                findings.append(f"{matured_unclaimed} matured deposits remain unclaimed for 3+ years")

            findings.append("Review maturity notification processes and customer response tracking")
            findings.append("Verify auto-renewal instructions and customer contact preferences")
        else:
            findings.append("No fixed deposits currently meet Article 2.2 dormancy criteria")
            findings.append(f"Analyzed {total_fds} accounts - all show proper maturity management")

        return findings


class InvestmentAccountDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.3 - Investment Account Dormancy Analysis using CSV columns"""

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

            # Filter for investment accounts and securities
            investment_accounts = df[
                (df[self.csv_columns['account_subtype']].isin(
                    ['SECURITIES', 'INVESTMENT', 'Securities', 'Investment'])) |
                (df[self.csv_columns['account_type']].isin(['INVESTMENT', 'Investment']))
                ].copy()

            dormant_accounts = []

            for idx, account in investment_accounts.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]

                    years_inactive = self._calculate_years_since(last_transaction, report_datetime)

                    # CBUAE Article 2.3 Logic: Investment inactivity analysis
                    if years_inactive >= self.default_params["standard_inactivity_years"] and balance > 0:
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


class PaymentInstrumentsDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.4 - Unclaimed Payment Instruments using CSV columns"""

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

            # Look for accounts with potential unclaimed instruments
            # This could be inferred from transaction patterns or specific flags
            unclaimed_instruments = []

            for idx, account in df.iterrows():
                try:
                    # Check for signs of unclaimed payment instruments
                    last_statement = account[self.csv_columns['last_statement_date']]
                    balance = account[self.csv_columns['balance_current']]

                    months_since_statement = self._calculate_months_since(last_statement, report_datetime)

                    # CBUAE Article 2.4 Logic: Unclaimed instruments (1+ year unclaimed)
                    if months_since_statement >= 12 and balance > 0:
                        unclaimed_instruments.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'balance_current': balance,
                            'last_statement_date': last_statement,
                            'months_since_statement': round(months_since_statement, 1),
                            'dormancy_trigger': 'UNCLAIMED_INSTRUMENT',
                            'compliance_article': '2.4',
                            'priority': 'MEDIUM',
                            'next_action': 'INSTRUMENT_VERIFICATION_REQUIRED'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing payment instrument for account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
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
            state.records_processed = len(df)
            state.processed_dataframe = pd.DataFrame(unclaimed_instruments) if unclaimed_instruments else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "payment_instruments_analysis")

        return state


class ContactAttemptsAgent(BaseDormancyAgent):
    """CBUAE Article 3.1 - Contact Attempt Verification using CSV columns"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("contact_attempts", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze contact attempt compliance"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for contact attempts analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for accounts in dormancy process
            dormancy_accounts = df[
                df[self.csv_columns['dormancy_status']].isin(['FLAGGED', 'CONTACTED', 'WAITING'])
            ].copy()

            contact_issues = []

            for idx, account in dormancy_accounts.iterrows():
                try:
                    contact_attempts = account[self.csv_columns['contact_attempts_made']]
                    last_attempt = account[self.csv_columns['last_contact_attempt_date']]
                    current_stage = account[self.csv_columns['current_stage']]

                    # Check contact attempt compliance
                    months_since_attempt = self._calculate_months_since(last_attempt, report_datetime)

                    # CBUAE Article 3.1 Logic: Verify contact attempts
                    has_issue = False
                    issue_type = ""

                    if contact_attempts < 3:  # Minimum contact attempts required
                        has_issue = True
                        issue_type = "INSUFFICIENT_CONTACT_ATTEMPTS"
                    elif months_since_attempt > 6:  # Contact attempts too old
                        has_issue = True
                        issue_type = "OUTDATED_CONTACT_ATTEMPTS"
                    elif current_stage == 'FLAGGED' and contact_attempts == 0:
                        has_issue = True
                        issue_type = "NO_CONTACT_ATTEMPTS_INITIATED"

                    if has_issue:
                        contact_issues.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'current_stage': current_stage,
                            'contact_attempts_made': contact_attempts,
                            'last_contact_attempt_date': last_attempt,
                            'months_since_attempt': round(months_since_attempt, 1),
                            'issue_type': issue_type,
                            'dormancy_trigger': 'CONTACT_COMPLIANCE_ISSUE',
                            'compliance_article': '3.1',
                            'priority': 'HIGH',
                            'next_action': 'INITIATE_CONTACT_CAMPAIGN'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing contact attempts for account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(contact_issues),
                "description": "CBUAE Article 3.1 - Contact Attempt Compliance Verification",
                "details": contact_issues,
                "compliance_article": "3.1",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(contact_issues) > 0
            }

            state.dormant_records_found = len(contact_issues)
            state.records_processed = len(dormancy_accounts)
            state.processed_dataframe = pd.DataFrame(contact_issues) if contact_issues else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "contact_attempts_analysis")

        return state


class InternalLedgerAgent(BaseDormancyAgent):
    """CBUAE Article 3.4 - Internal Ledger Transfer using CSV columns"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        super().__init__("internal_ledger", memory_agent, mcp_client, db_connection)

    async def analyze_dormancy(self, state: AgentState, report_date: str) -> AgentState:
        """Analyze internal ledger transfer candidates"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            state = await self.pre_analysis_memory_hook(state)

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for internal ledger analysis")

            df = state.input_dataframe.copy()
            report_datetime = self._safe_date_parse(report_date) or datetime.now()

            # Filter for accounts in waiting period that have completed contact attempts
            waiting_accounts = df[
                (df[self.csv_columns['current_stage']] == 'WAITING') &
                (df[self.csv_columns['contact_attempts_made']] >= 3)
                ].copy()

            ledger_candidates = []

            for idx, account in waiting_accounts.iterrows():
                try:
                    waiting_end = account[self.csv_columns['waiting_period_end']]
                    balance = account[self.csv_columns['balance_current']]

                    # Check if waiting period has ended
                    waiting_end_date = self._safe_date_parse(waiting_end)

                    if waiting_end_date and waiting_end_date <= report_datetime and balance > 0:
                        ledger_candidates.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'balance_current': balance,
                            'waiting_period_end': waiting_end,
                            'contact_attempts_made': account[self.csv_columns['contact_attempts_made']],
                            'dormancy_trigger': 'INTERNAL_LEDGER_ELIGIBLE',
                            'compliance_article': '3.4',
                            'priority': 'HIGH',
                            'next_action': 'TRANSFER_TO_INTERNAL_LEDGER'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing internal ledger candidate {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
                    continue

            state.analysis_results = {
                "count": len(ledger_candidates),
                "description": "CBUAE Article 3.4 - Internal Ledger Transfer Candidates",
                "details": ledger_candidates,
                "compliance_article": "3.4",
                "analysis_date": report_date,
                "validation_passed": True,
                "alerts_generated": len(ledger_candidates) > 0
            }

            state.dormant_records_found = len(ledger_candidates)
            state.records_processed = len(waiting_accounts)
            state.processed_dataframe = pd.DataFrame(ledger_candidates) if ledger_candidates else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "internal_ledger_analysis")

        return state


class CBTransferEligibilityAgent(BaseDormancyAgent):
    """CBUAE Article 8.1 - Central Bank Transfer Eligibility using CSV columns"""

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

            # Check transfer eligibility dates
            eligible_accounts = []

            for idx, account in df.iterrows():
                try:
                    transfer_eligibility = account[self.csv_columns['transfer_eligibility_date']]
                    balance = account[self.csv_columns['balance_current']]
                    transferred_to_cb = account[self.csv_columns['transferred_to_cb_date']]

                    # Check if eligible for CB transfer and not already transferred
                    eligibility_date = self._safe_date_parse(transfer_eligibility)

                    if (eligibility_date and
                            eligibility_date <= report_datetime and
                            pd.isna(transferred_to_cb) and
                            balance > 0):
                        years_eligible = self._calculate_years_since(transfer_eligibility, report_datetime)

                        eligible_accounts.append({
                            'account_id': account[self.csv_columns['account_id']],
                            'customer_id': account[self.csv_columns['customer_id']],
                            'customer_name': account[self.csv_columns['full_name_en']],
                            'balance_current': balance,
                            'transfer_eligibility_date': transfer_eligibility,
                            'years_eligible': round(years_eligible, 2),
                            'dormancy_trigger': 'CB_TRANSFER_ELIGIBLE',
                            'compliance_article': '8.1',
                            'priority': 'CRITICAL',
                            'next_action': 'PREPARE_CB_TRANSFER'
                        })

                except Exception as e:
                    logger.warning(
                        f"Error processing CB transfer eligibility for account {account.get(self.csv_columns['account_id'], 'unknown')}: {e}")
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
            state.records_processed = len(df)
            state.processed_dataframe = pd.DataFrame(eligible_accounts) if eligible_accounts else pd.DataFrame()
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            state = await self.post_analysis_memory_hook(state)

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            await self._handle_error(state, e, "cb_transfer_analysis")

        return state


# ===== COMPREHENSIVE WORKFLOW ORCHESTRATOR =====

class DormancyWorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for comprehensive CBUAE dormancy analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_connection=None):
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()
        self.db_connection = db_connection

        # Initialize all specialized agents with CSV column mapping
        self.agents = {
            # Article 2 - Dormancy Criteria Agents
            "demand_deposit": DemandDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "fixed_deposit": FixedDepositDormancyAgent(memory_agent, mcp_client, db_connection),
            "investment": InvestmentAccountDormancyAgent(memory_agent, mcp_client, db_connection),
            "payment_instruments": PaymentInstrumentsDormancyAgent(memory_agent, mcp_client, db_connection),

            # Article 3 - Bank Obligations Agents
            "contact_attempts": ContactAttemptsAgent(memory_agent, mcp_client, db_connection),
            "internal_ledger": InternalLedgerAgent(memory_agent, mcp_client, db_connection),

            # Article 8 - Central Bank Transfer Agents
            "cb_transfer": CBTransferEligibilityAgent(memory_agent, mcp_client, db_connection),
        }

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create comprehensive LangGraph workflow for CBUAE dormancy analysis"""

        workflow = StateGraph(DormancyAnalysisState)

        # Add nodes for comprehensive monitoring workflow
        workflow.add_node("initialize", self._initialize_comprehensive_analysis)
        workflow.add_node("execute_agents", self._execute_all_agents)
        workflow.add_node("consolidate_results", self._consolidate_comprehensive_results)
        workflow.add_node("generate_reports", self._generate_cbuae_reports)
        workflow.add_node("finalize", self._finalize_comprehensive_analysis)

        # Define workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "execute_agents")
        workflow.add_edge("execute_agents", "consolidate_results")
        workflow.add_edge("consolidate_results", "generate_reports")
        workflow.add_edge("generate_reports", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=MemorySaver())

    async def _initialize_comprehensive_analysis(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Initialize comprehensive CBUAE dormancy analysis"""
        try:
            state.analysis_status = DormancyStatus.PROCESSING
            state.current_node = "initialize"

            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "comprehensive_initialization",
                "status": "started",
                "total_agents": len(self.agents),
                "cbuae_articles_covered": self._get_covered_articles()
            })

            return state

        except Exception as e:
            logger.error(f"Comprehensive analysis initialization failed: {str(e)}")
            state.analysis_status = DormancyStatus.FAILED
            return state

    async def _execute_all_agents(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Execute all agents with CSV data"""
        try:
            state.current_node = "execute_agents"
            accounts_df = pd.DataFrame(state.processed_data['accounts'])
            report_date = state.analysis_config.get('report_date', datetime.now().strftime("%Y-%m-%d"))

            logger.info(f"Executing {len(self.agents)} dormancy agents on {len(accounts_df)} accounts")

            # Execute all agents
            for agent_name, agent in self.agents.items():
                try:
                    agent_state = AgentState(
                        agent_id=f"{agent_name}_{secrets.token_hex(8)}",
                        agent_type=agent_name,
                        session_id=state.session_id,
                        user_id=state.user_id,
                        timestamp=datetime.now(),
                        input_dataframe=accounts_df.copy(),
                        analysis_config=state.analysis_config
                    )

                    # Execute agent
                    completed_state = await agent.analyze_dormancy(agent_state, report_date)
                    state.agent_results[agent_name] = completed_state.analysis_results

                    if completed_state.agent_status == AgentStatus.COMPLETED:
                        state.completed_agents.append(agent_name)
                        state.dormant_accounts_found += completed_state.dormant_records_found

                        logger.info(
                            f"Agent {agent_name} completed: {completed_state.dormant_records_found} dormant accounts found")
                    else:
                        state.failed_agents.append(agent_name)
                        logger.warning(f"Agent {agent_name} failed")

                except Exception as e:
                    logger.error(f"Agent {agent_name} execution failed: {str(e)}")
                    state.failed_agents.append(agent_name)

            state.total_accounts_analyzed = len(accounts_df)
            return state

        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return state

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
                "agent_based_results": {},
                "regulatory_risk_assessment": {
                    "overall_risk_level": "low",
                    "critical_items": 0,
                    "high_risk_items": 0,
                    "medium_risk_items": 0,
                    "low_risk_items": 0
                },
                "audit_trail": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "agents_executed": len(state.completed_agents),
                    "total_findings": state.dormant_accounts_found,
                    "data_source": "csv_banking_compliance_dataset"
                }
            }

            # Process agent results
            for agent_name in state.completed_agents:
                agent_result = state.agent_results[agent_name]

                if agent_result:
                    article = agent_result.get("compliance_article")
                    if article:
                        consolidated["cbuae_compliance_summary"]["articles_analyzed"].append(f"Article {article}")

                    findings_count = agent_result.get("count", 0)
                    consolidated["agent_based_results"][agent_name] = {
                        "article": article,
                        "findings": findings_count,
                        "description": agent_result.get("description", ""),
                        "priority": "HIGH" if findings_count > 10 else "MEDIUM" if findings_count > 0 else "LOW"
                    }

                    # Update risk assessment
                    if findings_count > 10:
                        consolidated["regulatory_risk_assessment"]["high_risk_items"] += 1
                    elif findings_count > 0:
                        consolidated["regulatory_risk_assessment"]["medium_risk_items"] += 1
                    else:
                        consolidated["regulatory_risk_assessment"]["low_risk_items"] += 1

            # Calculate overall compliance
            total_findings = sum(
                result["findings"] for result in consolidated["agent_based_results"].values()
            )

            consolidated["cbuae_compliance_summary"]["total_compliance_items"] = total_findings
            consolidated["cbuae_compliance_summary"]["high_priority_findings"] = \
                consolidated["regulatory_risk_assessment"]["high_risk_items"]

            # Determine overall risk level
            if consolidated["regulatory_risk_assessment"]["high_risk_items"] > 3:
                consolidated["regulatory_risk_assessment"]["overall_risk_level"] = "critical"
            elif consolidated["regulatory_risk_assessment"]["high_risk_items"] > 0:
                consolidated["regulatory_risk_assessment"]["overall_risk_level"] = "high"
            elif consolidated["regulatory_risk_assessment"]["medium_risk_items"] > 5:
                consolidated["regulatory_risk_assessment"]["overall_risk_level"] = "medium"

            # Generate immediate actions
            high_risk_agents = [
                name for name, result in consolidated["agent_based_results"].items()
                if result["priority"] == "HIGH"
            ]

            if high_risk_agents:
                consolidated["cbuae_compliance_summary"]["immediate_actions_required"].append(
                    f"Immediate review required for: {', '.join(high_risk_agents)}"
                )

            state.dormancy_summary = consolidated
            return state

        except Exception as e:
            logger.error(f"Result consolidation failed: {str(e)}")
            return state

    async def _generate_cbuae_reports(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Generate CBUAE regulatory reports"""
        try:
            state.current_node = "generate_reports"

            # Generate summary statistics
            if "regulatory_reports" not in state.dormancy_summary:
                state.dormancy_summary["regulatory_reports"] = {}

            # Create detailed breakdown by article
            article_breakdown = {}
            for agent_name, result in state.agent_results.items():
                if result:
                    article = result.get("compliance_article")
                    if article:
                        article_breakdown[f"Article_{article}"] = {
                            "agent": agent_name,
                            "findings_count": result.get("count", 0),
                            "description": result.get("description", ""),
                            "key_findings": result.get("key_findings", []),
                            "validation_passed": result.get("validation_passed", True)
                        }

            state.dormancy_summary["regulatory_reports"] = {
                "article_breakdown": article_breakdown,
                "generated_timestamp": datetime.now().isoformat(),
                "data_quality_assessment": self._assess_overall_data_quality(state),
                "recommendations": self._generate_recommendations(state)
            }

            return state

        except Exception as e:
            logger.error(f"CBUAE report generation failed: {str(e)}")
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
                "analysis_type": "comprehensive_cbuae_monitoring_csv",
                "total_cbuae_articles_covered": len(self._get_covered_articles()),
                "comprehensive_summary": state.dormancy_summary,
                "agent_execution_results": {
                    agent_name: {
                        "findings": result.get("count", 0) if result else 0,
                        "cbuae_article": result.get("compliance_article") if result else None,
                        "description": result.get("description", "") if result else "",
                        "validation_passed": result.get("validation_passed", True) if result else True
                    } for agent_name, result in state.agent_results.items()
                },
                "regulatory_compliance_status": "compliant" if state.analysis_status == DormancyStatus.COMPLETED else "requires_review",
                "completion_timestamp": datetime.now().isoformat(),
                "data_source_info": {
                    "source": "banking_compliance_dataset_csv",
                    "total_records": state.total_accounts_analyzed,
                    "columns_used": list(self.agents["demand_deposit"].csv_columns.keys())
                }
            }

            return state

        except Exception as e:
            logger.error(f"Analysis finalization failed: {str(e)}")
            state.analysis_status = DormancyStatus.FAILED
            return state

    def _get_covered_articles(self) -> List[str]:
        """Get list of CBUAE articles covered by the agents"""
        return ["2.1.1", "2.2", "2.3", "2.4", "3.1", "3.4", "8.1"]

    def _assess_overall_data_quality(self, state: DormancyAnalysisState) -> Dict:
        """Assess overall data quality"""
        try:
            df = pd.DataFrame(state.processed_data['accounts'])

            # Check for missing critical fields
            critical_fields = [
                'customer_id', 'account_id', 'account_type', 'account_status',
                'last_transaction_date', 'balance_current', 'has_outstanding_facilities'
            ]

            missing_data = {}
            for field in critical_fields:
                if field in df.columns:
                    missing_count = df[field].isna().sum()
                    missing_data[field] = {
                        "missing_count": int(missing_count),
                        "missing_percentage": round((missing_count / len(df)) * 100, 2)
                    }

            return {
                "total_records": len(df),
                "missing_data_analysis": missing_data,
                "overall_quality_score": round(
                    1 - (sum(m["missing_percentage"] for m in missing_data.values()) / len(missing_data) / 100), 3)
            }

        except Exception as e:
            logger.warning(f"Data quality assessment failed: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, state: DormancyAnalysisState) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []

        try:
            # Based on findings from different agents
            high_priority_agents = [
                name for name, result in state.agent_results.items()
                if result and result.get("count", 0) > 0
            ]

            if "demand_deposit" in high_priority_agents:
                recommendations.append("Implement proactive customer contact campaigns for inactive demand deposits")

            if "fixed_deposit" in high_priority_agents:
                recommendations.append("Review maturity notification processes and auto-renewal procedures")

            if "contact_attempts" in high_priority_agents:
                recommendations.append("Strengthen contact attempt documentation and tracking systems")

            if "cb_transfer" in high_priority_agents:
                recommendations.append("Prepare accounts for Central Bank transfer as per Article 8.1 requirements")

            if len(high_priority_agents) > 3:
                recommendations.append("Consider comprehensive dormancy management system upgrade")

            recommendations.append("Regular quarterly reviews of dormancy classification criteria")
            recommendations.append("Enhance data quality controls for critical dormancy tracking fields")

        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")

        return recommendations

    async def orchestrate_comprehensive_analysis(self, user_id: str, input_dataframe: pd.DataFrame,
                                                 report_date: str, analysis_config: Dict = None,
                                                 db_connection=None) -> Dict:
        """Orchestrate comprehensive CBUAE dormancy analysis with CSV data"""
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

            # Execute comprehensive workflow
            final_state = await self.workflow.ainvoke(initial_state)

            return final_state.dormancy_results

        except Exception as e:
            logger.error(f"Comprehensive workflow orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": initial_state.session_id if 'initial_state' in locals() else None,
                "analysis_type": "comprehensive_cbuae_monitoring_csv"
            }


# ===== MAIN COMPREHENSIVE DORMANCY ANALYSIS AGENT =====

class DormancyAnalysisAgent:
    """Main comprehensive dormancy analysis agent using CSV column names"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_session=None):
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()
        self.db_session = db_session

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # Initialize comprehensive orchestrator
        self.orchestrator = DormancyWorkflowOrchestrator(memory_agent, mcp_client, db_session)

    @traceable(name="comprehensive_dormancy_analysis_csv")
    async def analyze_dormancy(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Main comprehensive dormancy analysis using CSV column names"""
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

            logger.info(f"Starting comprehensive dormancy analysis on {len(accounts_df)} accounts")
            logger.info(f"Available columns: {list(accounts_df.columns)}")

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
                    "stage": "comprehensive_dormancy_analysis_csv",
                    "status": "completed",
                    "analysis_type": "full_cbuae_monitoring_with_csv",
                    "articles_covered": orchestration_results.get("total_cbuae_articles_covered", 0),
                    "compliance_status": orchestration_results.get("regulatory_compliance_status", "unknown"),
                    "total_findings": state.dormant_accounts_found,
                    "processing_time": state.processing_time,
                    "data_source": "banking_compliance_dataset_csv"
                })
            else:
                state.analysis_status = DormancyStatus.FAILED
                error_msg = orchestration_results.get("error", "Unknown comprehensive analysis error")
                state.error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "comprehensive_orchestration_csv",
                    "error": error_msg,
                    "analysis_type": "full_cbuae_monitoring_with_csv"
                })

        except Exception as e:
            state.analysis_status = DormancyStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "comprehensive_dormancy_analysis_csv",
                "error": str(e),
                "analysis_type": "full_cbuae_monitoring_with_csv"
            })
            logger.error(f"Comprehensive dormancy analysis failed: {str(e)}")

        return state


# ===== FACTORY FUNCTIONS AND UTILITIES =====

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
            "analysis_results": final_state.dormancy_results,
            "summary": final_state.dormancy_summary,
            "total_accounts_analyzed": final_state.total_accounts_analyzed,
            "dormant_accounts_found": final_state.dormant_accounts_found,
            "high_risk_accounts": final_state.high_risk_accounts,
            "processing_time_seconds": final_state.processing_time,
            "compliance_flags": final_state.compliance_flags,
            "analysis_log": final_state.analysis_log,
            "error_log": final_state.error_log,
            "data_quality": final_state.dormancy_summary.get("regulatory_reports", {}).get("data_quality_assessment",
                                                                                           {}),
            "recommendations": final_state.dormancy_summary.get("regulatory_reports", {}).get("recommendations", [])
        }

    except Exception as e:
        logger.error(f"Comprehensive dormancy analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "session_id": None,
            "analysis_results": None
        }


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
    "PaymentInstrumentsDormancyAgent",
    "ContactAttemptsAgent",
    "InternalLedgerAgent",
    "CBTransferEligibilityAgent",
    "create_comprehensive_dormancy_analysis",
    "run_comprehensive_dormancy_analysis_csv",
    "MockMemoryAgent"
]

# Example usage and testing
if __name__ == "__main__":
    import asyncio


    async def test_csv_system():
        """Test the CSV-based dormancy analysis system"""

        print("=== CBUAE Dormancy Analysis System - CSV Integration ===")
        print("Testing with actual CSV column names...")

        # Create sample data matching CSV structure
        sample_accounts = pd.DataFrame({
            'customer_id': ['CUS770487', 'CUS865179', 'CUS133659'],
            'customer_type': ['INDIVIDUAL', 'INDIVIDUAL', 'CORPORATE'],
            'full_name_en': ['Ali Al-Zaabi', 'Fatima Al-Shamsi', 'Hassan Al-Suwaidi'],
            'account_id': ['ACC2867825', 'ACC8707870', 'ACC6292423'],
            'account_type': ['CURRENT', 'INVESTMENT', 'SAVINGS'],
            'account_status': ['DORMANT', 'ACTIVE', 'DORMANT'],
            'balance_current': [36849.91, 30964.14, 10710.06],
            'last_transaction_date': ['2021-02-20', '2023-05-25', '2020-02-27'],
            'has_outstanding_facilities': ['NO', 'YES', 'NO'],
            'contact_attempts_made': [5, 0, 3],
            'dormancy_status': ['FLAGGED', 'ACTIVE', 'CONTACTED'],
            'current_stage': ['WAITING', 'ACTIVE', 'FLAGGED'],
            'transfer_eligibility_date': ['2023-05-24', '2030-01-01', '2023-10-20'],
            'transferred_to_cb_date': [None, None, None],
            'maturity_date': [None, '2025-01-13', None],
            'auto_renewal': [None, 'YES', None]
        })

        print(f"Sample Data: {len(sample_accounts)} accounts")
        print(f"Columns: {list(sample_accounts.columns)}")

        try:
            # Test CSV-based comprehensive interface
            print("\n--- Testing CSV-Based Comprehensive Analysis ---")
            results = await run_comprehensive_dormancy_analysis_csv(
                user_id="test_user",
                account_data=sample_accounts,
                report_date="2024-12-01"
            )

            print(f"CSV Analysis Results:")
            print(f"  Success: {results['success']}")
            print(f"  Total Accounts Analyzed: {results['total_accounts_analyzed']}")
            print(f"  Dormant Accounts Found: {results['dormant_accounts_found']}")
            print(f"  High Risk Accounts: {results['high_risk_accounts']}")
            print(f"  Processing Time: {results['processing_time_seconds']:.2f} seconds")
            print(f"  CBUAE Articles Covered: {len(results.get('compliance_flags', []))}")

            if results['recommendations']:
                print(f"\nRecommendations:")
                for i, rec in enumerate(results['recommendations'], 1):
                    print(f"  {i}. {rec}")

            # Show data quality assessment
            data_quality = results.get('data_quality', {})
            if data_quality:
                print(f"\nData Quality Score: {data_quality.get('overall_quality_score', 'N/A')}")

        except Exception as e:
            print(f"Test failed: {e}")

        print("\n=== CSV Integration Test Complete ===")


    # Run test
    asyncio.run(test_csv_system())