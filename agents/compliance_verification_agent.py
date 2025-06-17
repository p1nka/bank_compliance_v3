"""
agents/compliance_verification_agent.py - CBUAE Compliance Verification Agent
Aligned with dormancy_agents.py structure and CSV column names
Comprehensive compliance verification for banking dormancy regulations
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
    # Mock implementations
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

# Enums for compliance verification
class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    ACTION_REQUIRED = "action_required"
    UNDER_REVIEW = "under_review"
    ESCALATED = "escalated"
    PENDING = "pending"

class ActionPriority(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ComplianceCategory(Enum):
    CONTACT_COMMUNICATION = "Contact & Communication"
    PROCESS_MANAGEMENT = "Process Management"
    SPECIALIZED_COMPLIANCE = "Specialized Compliance"
    REPORTING_RETENTION = "Reporting & Retention"
    UTILITY = "Utility"

class CBUAEArticle(Enum):
    """CBUAE regulation articles"""
    ARTICLE_2_1_1 = "Art. 2.1.1"  # Demand Deposit Dormancy
    ARTICLE_2_2 = "Art. 2.2"      # Fixed Deposit Dormancy
    ARTICLE_2_3 = "Art. 2.3"      # Investment Account Dormancy
    ARTICLE_2_4 = "Art. 2.4"      # Unclaimed Payment Instruments
    ARTICLE_2_6 = "Art. 2.6"      # Safe Deposit Box Dormancy
    ARTICLE_3_1 = "Art. 3.1"      # Contact Attempt Requirements
    ARTICLE_3_4 = "Art. 3.4"      # Internal Ledger Transfer
    ARTICLE_3_5 = "Art. 3.5"      # Ledger Maintenance
    ARTICLE_3_6 = "Art. 3.6"      # Unclaimed Instruments Ledger
    ARTICLE_3_7 = "Art. 3.7"      # Safe Deposit Box Court Application
    ARTICLE_3_9 = "Art. 3.9"      # Record Retention
    ARTICLE_3_10 = "Art. 3.10"    # Annual Reporting
    ARTICLE_4 = "Art. 4"          # Customer Claims
    ARTICLE_5 = "Art. 5"          # Proactive Communication
    ARTICLE_7_3 = "Art. 7.3"      # Statement Suppression
    ARTICLE_8 = "Art. 8"          # Central Bank Transfer
    ARTICLE_8_1 = "Art. 8.1"      # Transfer Eligibility
    ARTICLE_8_5 = "Art. 8.5"      # Foreign Currency Conversion

@dataclass
class ComplianceAction:
    """Data structure for compliance actions"""
    account_id: str
    customer_id: str
    action_type: str
    priority: ActionPriority
    deadline: datetime
    description: str
    cbuae_article: str
    assigned_agent: str
    status: ComplianceStatus = ComplianceStatus.ACTION_REQUIRED
    created_date: datetime = None
    estimated_effort_hours: float = 1.0
    dependencies: List[str] = None
    validation_criteria: List[str] = None

    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
        if self.validation_criteria is None:
            self.validation_criteria = []

@dataclass
class ComplianceResult:
    """Data structure for compliance analysis results"""
    agent_name: str
    category: ComplianceCategory
    cbuae_article: str
    accounts_processed: int
    violations_found: int
    actions_generated: List[ComplianceAction]
    processing_time: float
    success: bool
    compliance_rate: float = 0.0
    risk_level: str = "LOW"
    error_message: Optional[str] = None
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

        # Calculate compliance rate
        if self.accounts_processed > 0:
            self.compliance_rate = ((self.accounts_processed - self.violations_found) / self.accounts_processed) * 100

        # Determine risk level
        if self.violations_found == 0:
            self.risk_level = "LOW"
        elif self.violations_found <= self.accounts_processed * 0.1:
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "HIGH"

@dataclass
class ComplianceState:
    """State for compliance verification workflow"""
    session_id: str
    user_id: str
    verification_id: str
    timestamp: datetime

    # Input data
    processed_data: Optional[Dict] = None
    verification_config: Dict = None

    # Verification results
    compliance_results: Optional[Dict] = None
    compliance_summary: Optional[Dict] = None
    action_plan: Optional[Dict] = None

    # Status tracking
    verification_status: ComplianceStatus = ComplianceStatus.PENDING
    total_accounts_verified: int = 0
    total_violations_found: int = 0
    total_actions_generated: int = 0

    # Performance metrics
    processing_time: float = 0.0
    verification_efficiency: float = 0.0

    # Audit trail
    verification_log: List[Dict] = None
    error_log: List[Dict] = None

    # Agent orchestration
    active_agents: List[str] = None
    completed_agents: List[str] = None
    failed_agents: List[str] = None
    agent_results: Dict = None

    def __post_init__(self):
        if self.verification_config is None:
            self.verification_config = {}
        if self.verification_log is None:
            self.verification_log = []
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

# Mock Memory Agent
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

# Base Compliance Agent Class
class BaseComplianceAgent:
    """Base class for all compliance verification agents"""

    def __init__(self, agent_name: str, category: ComplianceCategory, cbuae_article: str,
                 memory_agent=None, mcp_client: MCPClient = None):
        self.agent_name = agent_name
        self.category = category
        self.cbuae_article = cbuae_article
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()

        # Initialize error handler
        self.error_handler = ErrorHandlerAgent(memory_agent, mcp_client)

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # CSV column mappings (same as dormancy agents)
        self.csv_columns = {
            'customer_id': 'customer_id',
            'account_id': 'account_id',
            'account_type': 'account_type',
            'account_status': 'account_status',
            'balance_current': 'balance_current',
            'last_transaction_date': 'last_transaction_date',
            'dormancy_status': 'dormancy_status',
            'dormancy_period_months': 'dormancy_period_months',
            'contact_attempts_made': 'contact_attempts_made',
            'last_contact_date': 'last_contact_date',
            'last_contact_attempt_date': 'last_contact_attempt_date',
            'current_stage': 'current_stage',
            'waiting_period_end': 'waiting_period_end',
            'transferred_to_ledger_date': 'transferred_to_ledger_date',
            'transferred_to_cb_date': 'transferred_to_cb_date',
            'transfer_eligibility_date': 'transfer_eligibility_date',
            'statement_frequency': 'statement_frequency',
            'currency': 'currency',
            'address_known': 'address_known',
            'maturity_date': 'maturity_date',
            'auto_renewal': 'auto_renewal'
        }

    def _safe_date_parse(self, date_str: str) -> Optional[datetime]:
        """Safely parse date strings from CSV"""
        if pd.isna(date_str) or date_str == '' or date_str is None:
            return None

        try:
            if isinstance(date_str, str):
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                return pd.to_datetime(date_str)
            elif isinstance(date_str, datetime):
                return date_str
            else:
                return pd.to_datetime(date_str)
        except Exception as e:
            logger.warning(f"Could not parse date '{date_str}': {e}")
            return None

    def _calculate_months_since(self, date_str: str, reference_date: datetime = None) -> float:
        """Calculate months since a given date"""
        if not reference_date:
            reference_date = datetime.now()

        parsed_date = self._safe_date_parse(date_str)
        if not parsed_date:
            return 0.0

        delta = reference_date - parsed_date
        return delta.days / 30.44  # Average days per month

    def generate_action(self, account_row: pd.Series, action_type: str,
                       priority: ActionPriority, days_to_deadline: int,
                       description: str, estimated_hours: float = 1.0) -> ComplianceAction:
        """Generate a compliance action for an account"""
        deadline = datetime.now() + timedelta(days=days_to_deadline)

        return ComplianceAction(
            account_id=account_row[self.csv_columns['account_id']],
            customer_id=account_row[self.csv_columns['customer_id']],
            action_type=action_type,
            priority=priority,
            deadline=deadline,
            description=description,
            cbuae_article=self.cbuae_article,
            assigned_agent=self.agent_name,
            estimated_effort_hours=estimated_hours
        )

    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Base analyze compliance method - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement analyze_compliance")

# ===== CONTACT & COMMUNICATION AGENTS =====

class DetectIncompleteContactAttemptsAgent(BaseComplianceAgent):
    """Contact & Communication - Art. 3.1, 5: Insufficient contact detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_incomplete_contact_attempts",
            category=ComplianceCategory.CONTACT_COMMUNICATION,
            cbuae_article=CBUAEArticle.ARTICLE_3_1.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="contact_attempts_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze contact attempt compliance using CSV columns"""
        start_time = datetime.now()
        actions = []

        try:
            # Find accounts with insufficient contact attempts
            insufficient_contact = accounts_df[
                (accounts_df[self.csv_columns['contact_attempts_made']] < 3) |
                (accounts_df[self.csv_columns['contact_attempts_made']].isna()) |
                (accounts_df[self.csv_columns['last_contact_date']].isna())
            ].copy()

            for _, account in insufficient_contact.iterrows():
                attempts_made = account.get(self.csv_columns['contact_attempts_made'], 0)

                if pd.isna(attempts_made) or attempts_made == 0:
                    action = self.generate_action(
                        account,
                        "INITIATE_CONTACT_ATTEMPTS",
                        ActionPriority.CRITICAL,
                        1,  # 1 day deadline
                        f"No contact attempts recorded. Initiate minimum 3 contact attempts via multiple channels.",
                        2.0  # 2 hours estimated
                    )
                elif attempts_made < 3:
                    remaining = 3 - int(attempts_made)
                    action = self.generate_action(
                        account,
                        "COMPLETE_CONTACT_ATTEMPTS",
                        ActionPriority.CRITICAL,
                        3,  # 3 days deadline
                        f"Complete {remaining} additional contact attempts. Current: {int(attempts_made)}/3",
                        1.5  # 1.5 hours estimated
                    )
                else:
                    action = self.generate_action(
                        account,
                        "DOCUMENT_CONTACT_ATTEMPTS",
                        ActionPriority.HIGH,
                        7,  # 7 days deadline
                        "Document and verify all contact attempts are properly recorded.",
                        0.5  # 0.5 hours estimated
                    )

                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(insufficient_contact),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Implement automated contact attempt tracking",
                    "Establish multi-channel communication protocols",
                    "Regular training on CBUAE contact requirements"
                ]
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

class DetectUnflaggedDormantCandidatesAgent(BaseComplianceAgent):
    """Contact & Communication - Art. 2: Unflagged dormancy detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_unflagged_dormant_candidates",
            category=ComplianceCategory.CONTACT_COMMUNICATION,
            cbuae_article=CBUAEArticle.ARTICLE_2_1_1.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="unflagged_dormancy_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze unflagged dormancy compliance using CSV columns"""
        start_time = datetime.now()
        actions = []

        try:
            # Find accounts that meet dormancy criteria but aren't flagged
            unflagged_dormant = accounts_df[
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 36) &
                ((accounts_df[self.csv_columns['dormancy_status']] != 'DORMANT') |
                 (accounts_df[self.csv_columns['dormancy_status']].isna()))
            ].copy()

            for _, account in unflagged_dormant.iterrows():
                dormant_months = account.get(self.csv_columns['dormancy_period_months'], 0)
                current_status = account.get(self.csv_columns['dormancy_status'], 'Unknown')

                action = self.generate_action(
                    account,
                    "UPDATE_DORMANCY_STATUS",
                    ActionPriority.HIGH,
                    2,  # 2 days deadline
                    f"Update account status to DORMANT (inactive for {dormant_months:.0f} months, current status: {current_status})",
                    1.0  # 1 hour estimated
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(unflagged_dormant),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Implement automated dormancy status updates",
                    "Regular review of dormancy classification criteria",
                    "System alerts for dormancy threshold breaches"
                ]
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

# ===== PROCESS MANAGEMENT AGENTS =====

class DetectInternalLedgerCandidatesAgent(BaseComplianceAgent):
    """Process Management - Art. 3.4, 3.5: Internal ledger transfer detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_internal_ledger_candidates",
            category=ComplianceCategory.PROCESS_MANAGEMENT,
            cbuae_article=CBUAEArticle.ARTICLE_3_4.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="internal_ledger_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze internal ledger transfer compliance using CSV columns"""
        start_time = datetime.now()
        actions = []

        try:
            # Accounts eligible for internal ledger transfer
            ledger_candidates = accounts_df[
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 39) &  # 3 years + 3 month waiting
                (accounts_df[self.csv_columns['contact_attempts_made']] >= 3) &
                (accounts_df[self.csv_columns['balance_current']] > 0) &
                ((accounts_df[self.csv_columns['transferred_to_ledger_date']].isna()) |
                 (accounts_df[self.csv_columns['transferred_to_ledger_date']] == ''))
            ].copy()

            for _, account in ledger_candidates.iterrows():
                balance = account.get(self.csv_columns['balance_current'], 0)
                dormant_months = account.get(self.csv_columns['dormancy_period_months'], 0)

                action = self.generate_action(
                    account,
                    "TRANSFER_TO_INTERNAL_LEDGER",
                    ActionPriority.HIGH,
                    5,  # 5 days deadline
                    f"Transfer to internal dormant ledger. Balance: {balance:.2f}, Dormant: {dormant_months:.0f} months",
                    3.0  # 3 hours estimated
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(ledger_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Automate internal ledger transfer processes",
                    "Establish clear ledger management procedures",
                    "Regular audit of ledger transfers"
                ]
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

class DetectStatementFreezeCandidatesAgent(BaseComplianceAgent):
    """Process Management - Art. 7.3: Statement suppression detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_statement_freeze_candidates",
            category=ComplianceCategory.PROCESS_MANAGEMENT,
            cbuae_article=CBUAEArticle.ARTICLE_7_3.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="statement_freeze_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze statement suppression compliance using CSV columns"""
        start_time = datetime.now()
        actions = []

        try:
            # Accounts eligible for statement suppression
            freeze_candidates = accounts_df[
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 36) &
                (accounts_df[self.csv_columns['statement_frequency']].isin(['MONTHLY', 'QUARTERLY'])) &
                (accounts_df[self.csv_columns['statement_frequency']] != 'SUPPRESSED')
            ].copy()

            for _, account in freeze_candidates.iterrows():
                current_frequency = account.get(self.csv_columns['statement_frequency'], 'Unknown')

                action = self.generate_action(
                    account,
                    "SUPPRESS_STATEMENT_GENERATION",
                    ActionPriority.MEDIUM,
                    10,  # 10 days deadline
                    f"Suppress regular statement generation. Current: {current_frequency} → SUPPRESSED",
                    0.5  # 0.5 hours estimated
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(freeze_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Automate statement suppression for dormant accounts",
                    "Cost-benefit analysis of statement generation",
                    "Customer notification protocols for statement changes"
                ]
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

class DetectCBUAETransferCandidatesAgent(BaseComplianceAgent):
    """Process Management - Art. 8: CBUAE transfer detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_cbuae_transfer_candidates",
            category=ComplianceCategory.PROCESS_MANAGEMENT,
            cbuae_article=CBUAEArticle.ARTICLE_8.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="cbuae_transfer_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze CBUAE transfer compliance using CSV columns"""
        start_time = datetime.now()
        actions = []

        try:
            # Accounts eligible for CBUAE transfer
            transfer_candidates = accounts_df[
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 60) &  # 5+ years
                (accounts_df[self.csv_columns['balance_current']] > 0) &
                (accounts_df[self.csv_columns['address_known']] == 'No') &
                ((accounts_df[self.csv_columns['transferred_to_cb_date']].isna()) |
                 (accounts_df[self.csv_columns['transferred_to_cb_date']] == ''))
            ].copy()

            for _, account in transfer_candidates.iterrows():
                balance = account.get(self.csv_columns['balance_current'], 0)
                dormant_months = account.get(self.csv_columns['dormancy_period_months'], 0)
                currency = account.get(self.csv_columns['currency'], 'AED')

                if currency != 'AED':
                    action = self.generate_action(
                        account,
                        "CONVERT_CURRENCY_FOR_CB_TRANSFER",
                        ActionPriority.HIGH,
                        7,  # 7 days deadline
                        f"Convert {currency} balance to AED before CBUAE transfer. Balance: {balance:.2f} {currency}",
                        2.0  # 2 hours estimated
                    )
                else:
                    action = self.generate_action(
                        account,
                        "PREPARE_CBUAE_TRANSFER",
                        ActionPriority.HIGH,
                        14,  # 14 days deadline
                        f"Prepare for CBUAE transfer. Balance: {balance:.2f} AED, Dormant: {dormant_months:.0f} months",
                        4.0  # 4 hours estimated
                    )

                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(transfer_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Establish CBUAE transfer preparation workflows",
                    "Currency conversion protocols for foreign accounts",
                    "Documentation requirements for CB transfers"
                ]
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

# ===== SPECIALIZED COMPLIANCE AGENTS =====

class DetectForeignCurrencyConversionNeededAgent(BaseComplianceAgent):
    """Specialized Compliance - Art. 8.5: FX conversion detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_foreign_currency_conversion_needed",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article=CBUAEArticle.ARTICLE_8_5.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

        # CBUAE exchange rates (example rates)
        self.exchange_rates = {
            'USD': 3.67,
            'EUR': 4.0,
            'GBP': 4.5,
            'SAR': 0.98,
            'AED': 1.0
        }

    @traceable(name="fx_conversion_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze foreign currency conversion compliance using CSV columns"""
        start_time = datetime.now()
        actions = []

        try:
            # Foreign currency accounts needing conversion
            fx_conversion_needed = accounts_df[
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 60) &
                (accounts_df[self.csv_columns['currency']] != 'AED') &
                (accounts_df[self.csv_columns['currency']].notna()) &
                (accounts_df[self.csv_columns['balance_current']] > 0)
            ].copy()

            for _, account in fx_conversion_needed.iterrows():
                currency = account.get(self.csv_columns['currency'], 'Unknown')
                balance = account.get(self.csv_columns['balance_current'], 0)
                exchange_rate = self.exchange_rates.get(currency, 1.0)
                aed_equivalent = balance * exchange_rate

                action = self.generate_action(
                    account,
                    "CONVERT_FOREIGN_CURRENCY",
                    ActionPriority.MEDIUM,
                    7,  # 7 days deadline
                    f"Convert {balance:.2f} {currency} to {aed_equivalent:.2f} AED (Rate: {exchange_rate})",
                    1.5  # 1.5 hours estimated
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(fx_conversion_needed),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Establish FX conversion protocols for dormant accounts",
                    "Regular review of exchange rates",
                    "Automated currency conversion workflows"
                ]
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

class CheckRecordRetentionComplianceAgent(BaseComplianceAgent):
    """Specialized Compliance - Art. 3.9: Record retention compliance"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="check_record_retention_compliance",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article=CBUAEArticle.ARTICLE_3_9.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="record_retention_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze record retention compliance using CSV columns"""
        start_time = datetime.now()
        actions = []

        try:
            # Check for missing critical documentation
            critical_fields = [
                self.csv_columns['last_contact_date'],
                self.csv_columns['contact_attempts_made'],
                self.csv_columns['dormancy_trigger_date'],
                self.csv_columns['dormancy_classification_date']
            ]

            for _, account in accounts_df.iterrows():
                missing_fields = []

                for field in critical_fields:
                    if field in account and (pd.isna(account[field]) or account[field] == ''):
                        missing_fields.append(field)

                if missing_fields:
                    action = self.generate_action(
                        account,
                        "COMPLETE_RECORD_DOCUMENTATION",
                        ActionPriority.CRITICAL,
                        2,  # 2 days deadline
                        f"Complete missing documentation: {', '.join(missing_fields)}",
                        2.0  # 2 hours estimated
                    )
                    actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(actions),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Implement mandatory field validation",
                    "Regular data quality audits",
                    "Staff training on documentation requirements"
                ]
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

# ===== REPORTING & RETENTION AGENTS =====

class GenerateAnnualCBUAEReportSummaryAgent(BaseComplianceAgent):
    """Reporting & Retention - Art. 3.10: Annual CBUAE reporting"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="generate_annual_cbuae_report_summary",
            category=ComplianceCategory.REPORTING_RETENTION,
            cbuae_article=CBUAEArticle.ARTICLE_3_10.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="annual_reporting_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze annual reporting compliance using CSV columns"""
        start_time = datetime.now()
        actions = []

        try:
            # All dormant accounts with positive balances need to be in annual report
            reporting_candidates = accounts_df[
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 12) &
                (accounts_df[self.csv_columns['balance_current']] > 0)
            ].copy()

            if len(reporting_candidates) > 0:
                # Generate single action for annual reporting compilation
                total_balance = reporting_candidates[self.csv_columns['balance_current']].sum()
                account_count = len(reporting_candidates)

                # Use first account as representative (for action generation)
                representative_account = reporting_candidates.iloc[0]

                action = self.generate_action(
                    representative_account,
                    "COMPILE_ANNUAL_CBUAE_REPORT",
                    ActionPriority.MEDIUM,
                    30,  # 30 days deadline
                    f"Compile annual CBUAE report for {account_count} dormant accounts. Total balance: {total_balance:.2f} AED",
                    8.0  # 8 hours estimated
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(reporting_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Automate annual report generation",
                    "Establish reporting calendar and deadlines",
                    "Quality assurance for regulatory reporting"
                ]
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

# ===== MASTER COMPLIANCE ORCHESTRATOR =====

class ComplianceVerificationOrchestrator:
    """LangGraph-based workflow orchestrator for comprehensive compliance verification"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()

        # Initialize all compliance agents
        self.agents = {
            # Contact & Communication (2)
            "contact_attempts": DetectIncompleteContactAttemptsAgent(memory_agent, mcp_client),
            "unflagged_dormant": DetectUnflaggedDormantCandidatesAgent(memory_agent, mcp_client),

            # Process Management (3)
            "internal_ledger": DetectInternalLedgerCandidatesAgent(memory_agent, mcp_client),
            "statement_freeze": DetectStatementFreezeCandidatesAgent(memory_agent, mcp_client),
            "cbuae_transfer": DetectCBUAETransferCandidatesAgent(memory_agent, mcp_client),

            # Specialized Compliance (2)
            "fx_conversion": DetectForeignCurrencyConversionNeededAgent(memory_agent, mcp_client),
            "record_retention": CheckRecordRetentionComplianceAgent(memory_agent, mcp_client),

            # Reporting & Retention (1)
            "annual_reporting": GenerateAnnualCBUAEReportSummaryAgent(memory_agent, mcp_client)
        }

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create comprehensive LangGraph workflow for compliance verification"""

        workflow = StateGraph(ComplianceState)

        # Add nodes for compliance verification workflow
        workflow.add_node("initialize", self._initialize_compliance_verification)
        workflow.add_node("execute_agents", self._execute_all_compliance_agents)
        workflow.add_node("consolidate_results", self._consolidate_compliance_results)
        workflow.add_node("generate_action_plan", self._generate_action_plan)
        workflow.add_node("finalize", self._finalize_compliance_verification)

        # Define workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "execute_agents")
        workflow.add_edge("execute_agents", "consolidate_results")
        workflow.add_edge("consolidate_results", "generate_action_plan")
        workflow.add_edge("generate_action_plan", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=MemorySaver())

    async def _initialize_compliance_verification(self, state: ComplianceState) -> ComplianceState:
        """Initialize compliance verification workflow"""
        try:
            state.verification_status = ComplianceStatus.UNDER_REVIEW

            state.verification_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "compliance_verification_initialization",
                "status": "started",
                "total_agents": len(self.agents),
                "cbuae_articles_covered": self._get_covered_articles()
            })

            return state

        except Exception as e:
            logger.error(f"Compliance verification initialization failed: {str(e)}")
            state.verification_status = ComplianceStatus.NON_COMPLIANT
            return state

    async def _execute_all_compliance_agents(self, state: ComplianceState) -> ComplianceState:
        """Execute all compliance agents"""
        try:
            accounts_df = pd.DataFrame(state.processed_data['accounts'])

            logger.info(f"Executing {len(self.agents)} compliance agents on {len(accounts_df)} accounts")

            # Execute all agents
            for agent_name, agent in self.agents.items():
                try:
                    logger.info(f"Executing compliance agent: {agent_name}")

                    # Execute agent
                    compliance_result = await agent.analyze_compliance(accounts_df)
                    state.agent_results[agent_name] = compliance_result

                    if compliance_result.success:
                        state.completed_agents.append(agent_name)
                        state.total_violations_found += compliance_result.violations_found
                        state.total_actions_generated += len(compliance_result.actions_generated)

                        logger.info(f"Agent {agent_name} completed: {compliance_result.violations_found} violations found")
                    else:
                        state.failed_agents.append(agent_name)
                        logger.warning(f"Agent {agent_name} failed: {compliance_result.error_message}")

                except Exception as e:
                    logger.error(f"Agent {agent_name} execution failed: {str(e)}")
                    state.failed_agents.append(agent_name)

            state.total_accounts_verified = len(accounts_df)
            return state

        except Exception as e:
            logger.error(f"Compliance agent execution failed: {str(e)}")
            return state

    async def _consolidate_compliance_results(self, state: ComplianceState) -> ComplianceState:
        """Consolidate compliance verification results"""
        try:
            # Consolidate all compliance results
            consolidated = {
                "compliance_summary": {
                    "total_agents_executed": len(state.completed_agents),
                    "total_accounts_verified": state.total_accounts_verified,
                    "total_violations_found": state.total_violations_found,
                    "total_actions_generated": state.total_actions_generated,
                    "overall_compliance_rate": self._calculate_overall_compliance_rate(state),
                    "risk_assessment": self._assess_overall_risk(state)
                },
                "agent_results": {},
                "category_breakdown": {},
                "priority_breakdown": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
                "consolidated_actions": []
            }

            # Process agent results
            for agent_name in state.completed_agents:
                agent_result = state.agent_results[agent_name]

                consolidated["agent_results"][agent_name] = {
                    "category": agent_result.category.value,
                    "cbuae_article": agent_result.cbuae_article,
                    "accounts_processed": agent_result.accounts_processed,
                    "violations_found": agent_result.violations_found,
                    "actions_generated": len(agent_result.actions_generated),
                    "compliance_rate": agent_result.compliance_rate,
                    "risk_level": agent_result.risk_level,
                    "processing_time": agent_result.processing_time,
                    "recommendations": agent_result.recommendations
                }

                # Add actions to consolidated list
                consolidated["consolidated_actions"].extend(agent_result.actions_generated)

                # Update priority breakdown
                for action in agent_result.actions_generated:
                    consolidated["priority_breakdown"][action.priority.value] += 1

                # Update category breakdown
                category = agent_result.category.value
                if category not in consolidated["category_breakdown"]:
                    consolidated["category_breakdown"][category] = {
                        "agents": 0,
                        "violations": 0,
                        "actions": 0,
                        "avg_compliance_rate": 0.0
                    }

                consolidated["category_breakdown"][category]["agents"] += 1
                consolidated["category_breakdown"][category]["violations"] += agent_result.violations_found
                consolidated["category_breakdown"][category]["actions"] += len(agent_result.actions_generated)

            # Calculate average compliance rates by category
            for category, stats in consolidated["category_breakdown"].items():
                category_agents = [
                    result for result in state.agent_results.values()
                    if result.category.value == category and result.success
                ]
                if category_agents:
                    stats["avg_compliance_rate"] = sum(r.compliance_rate for r in category_agents) / len(category_agents)

            state.compliance_summary = consolidated
            return state

        except Exception as e:
            logger.error(f"Compliance result consolidation failed: {str(e)}")
            return state

    async def _generate_action_plan(self, state: ComplianceState) -> ComplianceState:
        """Generate prioritized action plan"""
        try:
            actions = state.compliance_summary.get("consolidated_actions", [])

            if not actions:
                state.action_plan = {
                    "action_plan": [],
                    "execution_timeline": {},
                    "resource_requirements": {},
                    "priority_summary": {}
                }
                return state

            # Sort actions by priority and deadline
            priority_order = {
                ActionPriority.CRITICAL: 0,
                ActionPriority.HIGH: 1,
                ActionPriority.MEDIUM: 2,
                ActionPriority.LOW: 3
            }

            sorted_actions = sorted(actions, key=lambda x: (priority_order[x.priority], x.deadline))

            # Group actions by timeline
            timeline_groups = {
                'immediate': [],  # Due within 24 hours
                'urgent': [],     # Due within 7 days
                'short_term': [], # Due within 30 days
                'long_term': []   # Due beyond 30 days
            }

            now = datetime.now()

            for action in sorted_actions:
                days_to_deadline = (action.deadline - now).days

                if days_to_deadline <= 1:
                    timeline_groups['immediate'].append(action)
                elif days_to_deadline <= 7:
                    timeline_groups['urgent'].append(action)
                elif days_to_deadline <= 30:
                    timeline_groups['short_term'].append(action)
                else:
                    timeline_groups['long_term'].append(action)

            # Generate resource requirements estimate
            resource_requirements = {
                'compliance_officers': len([a for a in actions if a.priority in [ActionPriority.CRITICAL, ActionPriority.HIGH]]),
                'estimated_total_hours': sum(a.estimated_effort_hours for a in actions),
                'critical_actions': len([a for a in actions if a.priority == ActionPriority.CRITICAL]),
                'system_updates_required': len([a for a in actions if any(
                    keyword in a.action_type.lower() for keyword in ['status', 'suppress', 'transfer', 'convert'])]),
                'customer_contact_required': len([a for a in actions if 'contact' in a.action_type.lower()])
            }

            state.action_plan = {
                'action_plan': [
                    {
                        'account_id': action.account_id,
                        'customer_id': action.customer_id,
                        'action_type': action.action_type,
                        'priority': action.priority.value,
                        'deadline': action.deadline.isoformat(),
                        'description': action.description,
                        'cbuae_article': action.cbuae_article,
                        'assigned_agent': action.assigned_agent,
                        'estimated_hours': action.estimated_effort_hours,
                        'days_to_deadline': (action.deadline - now).days
                    } for action in sorted_actions
                ],
                'execution_timeline': {
                    'immediate_actions': len(timeline_groups['immediate']),
                    'urgent_actions': len(timeline_groups['urgent']),
                    'short_term_actions': len(timeline_groups['short_term']),
                    'long_term_actions': len(timeline_groups['long_term'])
                },
                'resource_requirements': resource_requirements,
                'priority_summary': state.compliance_summary.get('priority_breakdown', {})
            }

            return state

        except Exception as e:
            logger.error(f"Action plan generation failed: {str(e)}")
            return state

    async def _finalize_compliance_verification(self, state: ComplianceState) -> ComplianceState:
        """Finalize compliance verification"""
        try:
            if state.verification_status != ComplianceStatus.NON_COMPLIANT:
                # Determine final compliance status
                overall_compliance_rate = state.compliance_summary.get("compliance_summary", {}).get("overall_compliance_rate", 0)

                if overall_compliance_rate >= 95:
                    state.verification_status = ComplianceStatus.COMPLIANT
                elif overall_compliance_rate >= 80:
                    state.verification_status = ComplianceStatus.ACTION_REQUIRED
                else:
                    state.verification_status = ComplianceStatus.NON_COMPLIANT

            # Create comprehensive final results
            state.compliance_results = {
                "session_id": state.session_id,
                "verification_type": "comprehensive_cbuae_compliance_verification",
                "total_cbuae_articles_covered": len(self._get_covered_articles()),
                "verification_summary": state.compliance_summary,
                "action_plan": state.action_plan,
                "final_compliance_status": state.verification_status.value,
                "completion_timestamp": datetime.now().isoformat(),
                "data_source_info": {
                    "source": "banking_compliance_dataset_csv",
                    "total_records": state.total_accounts_verified,
                    "agents_executed": len(state.completed_agents),
                    "agents_failed": len(state.failed_agents)
                }
            }

            return state

        except Exception as e:
            logger.error(f"Compliance verification finalization failed: {str(e)}")
            state.verification_status = ComplianceStatus.NON_COMPLIANT
            return state

    def _get_covered_articles(self) -> List[str]:
        """Get list of CBUAE articles covered by the agents"""
        return [
            "Art. 2.1.1", "Art. 3.1", "Art. 3.4", "Art. 3.9", "Art. 3.10",
            "Art. 7.3", "Art. 8", "Art. 8.5"
        ]

    def _calculate_overall_compliance_rate(self, state: ComplianceState) -> float:
        """Calculate overall compliance rate"""
        if state.total_accounts_verified == 0:
            return 100.0

        compliant_accounts = state.total_accounts_verified - state.total_violations_found
        return (compliant_accounts / state.total_accounts_verified) * 100

    def _assess_overall_risk(self, state: ComplianceState) -> str:
        """Assess overall compliance risk"""
        compliance_rate = self._calculate_overall_compliance_rate(state)

        if compliance_rate >= 95:
            return "LOW"
        elif compliance_rate >= 80:
            return "MEDIUM"
        elif compliance_rate >= 60:
            return "HIGH"
        else:
            return "CRITICAL"

    async def orchestrate_compliance_verification(self, user_id: str, input_dataframe: pd.DataFrame,
                                                  verification_config: Dict = None) -> Dict:
        """Orchestrate comprehensive compliance verification"""
        try:
            # Create initial state
            initial_state = ComplianceState(
                session_id=secrets.token_hex(16),
                user_id=user_id,
                verification_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                processed_data={'accounts': input_dataframe.to_dict('records')},
                verification_config=verification_config or {}
            )

            # Execute comprehensive workflow
            final_state = await self.workflow.ainvoke(initial_state)

            return final_state.compliance_results

        except Exception as e:
            logger.error(f"Compliance verification orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": initial_state.session_id if 'initial_state' in locals() else None,
                "verification_type": "comprehensive_cbuae_compliance_verification"
            }

# ===== MAIN COMPLIANCE VERIFICATION AGENT =====

class ComplianceVerificationAgent:
    """Main comprehensive compliance verification agent"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None, db_session=None):
        self.memory_agent = memory_agent or MockMemoryAgent()
        self.mcp_client = mcp_client or MCPClient()
        self.db_session = db_session

        try:
            self.langsmith_client = LangSmithClient()
        except:
            self.langsmith_client = None

        # Initialize compliance orchestrator
        self.orchestrator = ComplianceVerificationOrchestrator(memory_agent, mcp_client)

    @traceable(name="comprehensive_compliance_verification")
    async def verify_compliance(self, state: ComplianceState) -> ComplianceState:
        """Main comprehensive compliance verification"""
        try:
            start_time = datetime.now()
            state.verification_status = ComplianceStatus.UNDER_REVIEW

            # Extract account data
            if not state.processed_data or 'accounts' not in state.processed_data:
                raise ValueError("No account data available for compliance verification")

            accounts_df = pd.DataFrame(state.processed_data['accounts'])
            if accounts_df.empty:
                raise ValueError("Empty account data provided")

            logger.info(f"Starting comprehensive compliance verification on {len(accounts_df)} accounts")

            # Execute comprehensive orchestrated verification
            verification_results = await self.orchestrator.orchestrate_compliance_verification(
                user_id=state.user_id,
                input_dataframe=accounts_df,
                verification_config=state.verification_config
            )

            # Process verification results
            if verification_results and not verification_results.get("success") == False:
                state.compliance_results = verification_results
                state.compliance_summary = verification_results.get("verification_summary", {})
                state.action_plan = verification_results.get("action_plan", {})

                # Extract metrics
                summary = state.compliance_summary.get("compliance_summary", {})
                state.total_accounts_verified = summary.get("total_accounts_verified", 0)
                state.total_violations_found = summary.get("total_violations_found", 0)
                state.total_actions_generated = summary.get("total_actions_generated", 0)

                # Calculate processing metrics
                state.processing_time = (datetime.now() - start_time).total_seconds()
                state.verification_efficiency = (
                    state.total_accounts_verified / state.processing_time
                    if state.processing_time > 0 else 0
                )

                # Determine final status
                compliance_rate = summary.get("overall_compliance_rate", 0)
                if compliance_rate >= 95:
                    state.verification_status = ComplianceStatus.COMPLIANT
                elif compliance_rate >= 80:
                    state.verification_status = ComplianceStatus.ACTION_REQUIRED
                else:
                    state.verification_status = ComplianceStatus.NON_COMPLIANT

                # Log successful verification
                state.verification_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "comprehensive_compliance_verification",
                    "status": "completed",
                    "verification_type": "full_cbuae_compliance_verification",
                    "compliance_status": state.verification_status.value,
                    "total_violations": state.total_violations_found,
                    "total_actions": state.total_actions_generated,
                    "processing_time": state.processing_time,
                    "compliance_rate": compliance_rate
                })
            else:
                state.verification_status = ComplianceStatus.NON_COMPLIANT
                error_msg = verification_results.get("error", "Unknown verification error")
                state.error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "compliance_verification_orchestration",
                    "error": error_msg,
                    "verification_type": "full_cbuae_compliance_verification"
                })

        except Exception as e:
            state.verification_status = ComplianceStatus.NON_COMPLIANT
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "comprehensive_compliance_verification",
                "error": str(e),
                "verification_type": "full_cbuae_compliance_verification"
            })
            logger.error(f"Comprehensive compliance verification failed: {str(e)}")

        return state

# ===== FACTORY FUNCTIONS AND UTILITIES =====

def create_compliance_verification_agent(
    memory_agent=None,
    mcp_client: MCPClient = None,
    db_session=None
) -> ComplianceVerificationAgent:
    """Factory function to create compliance verification agent"""
    return ComplianceVerificationAgent(
        memory_agent=memory_agent,
        mcp_client=mcp_client,
        db_session=db_session
    )

def get_all_compliance_agents() -> List[BaseComplianceAgent]:
    """Get list of all compliance agents"""
    return [
        # Contact & Communication (2)
        DetectIncompleteContactAttemptsAgent(),
        DetectUnflaggedDormantCandidatesAgent(),

        # Process Management (3)
        DetectInternalLedgerCandidatesAgent(),
        DetectStatementFreezeCandidatesAgent(),
        DetectCBUAETransferCandidatesAgent(),

        # Specialized Compliance (2)
        DetectForeignCurrencyConversionNeededAgent(),
        CheckRecordRetentionComplianceAgent(),

        # Reporting & Retention (1)
        GenerateAnnualCBUAEReportSummaryAgent()
    ]

def get_compliance_agents_by_category() -> Dict[ComplianceCategory, List[BaseComplianceAgent]]:
    """Get compliance agents grouped by category"""
    agents = get_all_compliance_agents()
    categorized = {category: [] for category in ComplianceCategory}

    for agent in agents:
        categorized[agent.category].append(agent)

    return categorized

def export_compliance_actions_to_csv(compliance_results: Dict[str, Any],
                                   filename: Optional[str] = None) -> str:
    """Export compliance actions to CSV format"""

    # Extract actions from compliance results
    actions = []
    if 'verification_summary' in compliance_results:
        actions = compliance_results['verification_summary'].get('consolidated_actions', [])
    elif 'consolidated_actions' in compliance_results:
        actions = compliance_results['consolidated_actions']

    if not actions:
        csv_header = "account_id,customer_id,action_type,priority,deadline,description,cbuae_article,assigned_agent,status,estimated_hours\n"
        if filename:
            with open(filename, 'w') as f:
                f.write(csv_header)
            return f"Empty actions exported to {filename}"
        return csv_header

    csv_data = []
    for action in actions:
        csv_data.append({
            'account_id': action.account_id,
            'customer_id': action.customer_id,
            'action_type': action.action_type,
            'priority': action.priority.value,
            'deadline': action.deadline.strftime('%Y-%m-%d %H:%M:%S'),
            'description': action.description,
            'cbuae_article': action.cbuae_article,
            'assigned_agent': action.assigned_agent,
            'status': action.status.value,
            'estimated_hours': action.estimated_effort_hours
        })

    df = pd.DataFrame(csv_data)

    if filename:
        df.to_csv(filename, index=False)
        return f"Actions exported to {filename}"
    else:
        return df.to_csv(index=False)

def get_compliance_agent_coverage() -> Dict[str, Any]:
    """Get coverage information for compliance agents"""

    agents = get_all_compliance_agents()

    coverage = {
        'total_agents': len(agents),
        'categories': {},
        'cbuae_articles': set(),
        'agent_details': []
    }

    for agent in agents:
        # Category breakdown
        category_name = agent.category.value
        if category_name not in coverage['categories']:
            coverage['categories'][category_name] = 0
        coverage['categories'][category_name] += 1

        # CBUAE articles
        coverage['cbuae_articles'].add(agent.cbuae_article)

        # Agent details
        coverage['agent_details'].append({
            'name': agent.agent_name,
            'category': category_name,
            'article': agent.cbuae_article
        })

    coverage['cbuae_articles'] = list(coverage['cbuae_articles'])
    coverage['total_articles_covered'] = len(coverage['cbuae_articles'])

    return coverage

async def run_standalone_compliance_verification(
    accounts_data: Union[pd.DataFrame, List[Dict], str],
    user_id: str = "standalone_user",
    verification_config: Dict = None
) -> Dict:
    """Run standalone compliance verification without workflow orchestration"""

    try:
        # Handle different input types
        if isinstance(accounts_data, str):
            # Assume it's a CSV file path
            accounts_df = pd.read_csv(accounts_data)
        elif isinstance(accounts_data, list):
            # Convert list of dicts to DataFrame
            accounts_df = pd.DataFrame(accounts_data)
        elif isinstance(accounts_data, pd.DataFrame):
            accounts_df = accounts_data.copy()
        else:
            raise ValueError("Invalid accounts_data type. Expected DataFrame, list of dicts, or CSV file path")

        if accounts_df.empty:
            raise ValueError("No account data provided for compliance verification")

        # Create orchestrator
        orchestrator = ComplianceVerificationOrchestrator()

        # Run verification
        results = await orchestrator.orchestrate_compliance_verification(
            user_id=user_id,
            input_dataframe=accounts_df,
            verification_config=verification_config or {}
        )

        return results

    except Exception as e:
        logger.error(f"Standalone compliance verification failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "verification_type": "standalone_compliance_verification"
        }

def generate_compliance_report(compliance_results: Dict) -> str:
    """Generate a comprehensive compliance report in markdown format"""

    if not compliance_results or compliance_results.get("success") == False:
        return "# Compliance Verification Report\n\n**Status:** Failed\n\n**Error:** " + \
               compliance_results.get("error", "Unknown error")

    verification_summary = compliance_results.get("verification_summary", {})
    compliance_summary = verification_summary.get("compliance_summary", {})
    action_plan = compliance_results.get("action_plan", {})

    report = []
    report.append("# CBUAE Compliance Verification Report")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"**Verification Date:** {compliance_results.get('completion_timestamp', 'N/A')}")
    report.append(f"**Total Accounts Verified:** {compliance_summary.get('total_accounts_verified', 0):,}")
    report.append(f"**Total Violations Found:** {compliance_summary.get('total_violations_found', 0):,}")
    report.append(f"**Total Actions Generated:** {compliance_summary.get('total_actions_generated', 0):,}")
    report.append(f"**Overall Compliance Rate:** {compliance_summary.get('overall_compliance_rate', 0):.1f}%")
    report.append(f"**Risk Assessment:** {compliance_summary.get('risk_assessment', 'Unknown')}")
    report.append("")

    # CBUAE Articles Coverage
    total_articles = compliance_results.get("total_cbuae_articles_covered", 0)
    report.append("## CBUAE Regulation Coverage")
    report.append("")
    report.append(f"**Total CBUAE Articles Covered:** {total_articles}")
    report.append("**Articles Analyzed:**")
    for article in ["Art. 2.1.1", "Art. 3.1", "Art. 3.4", "Art. 3.9", "Art. 3.10", "Art. 7.3", "Art. 8", "Art. 8.5"]:
        report.append(f"- {article}")
    report.append("")

    # Category Breakdown
    category_breakdown = verification_summary.get("category_breakdown", {})
    if category_breakdown:
        report.append("## Compliance by Category")
        report.append("")
        report.append("| Category | Agents | Violations | Actions | Avg Compliance Rate |")
        report.append("|----------|--------|------------|---------|---------------------|")
        for category, stats in category_breakdown.items():
            compliance_rate = stats.get('avg_compliance_rate', 0)
            report.append(f"| {category} | {stats['agents']} | {stats['violations']} | {stats['actions']} | {compliance_rate:.1f}% |")
        report.append("")

    # Priority Breakdown
    priority_breakdown = verification_summary.get("priority_breakdown", {})
    if priority_breakdown:
        report.append("## Actions by Priority")
        report.append("")
        report.append("| Priority | Count | Percentage |")
        report.append("|----------|-------|------------|")
        total_actions = sum(priority_breakdown.values())
        for priority, count in priority_breakdown.items():
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            report.append(f"| {priority} | {count} | {percentage:.1f}% |")
        report.append("")

    # Action Plan Timeline
    if action_plan and 'execution_timeline' in action_plan:
        timeline = action_plan['execution_timeline']
        report.append("## Action Plan Timeline")
        report.append("")
        report.append(f"**Immediate Actions (≤1 day):** {timeline.get('immediate_actions', 0)}")
        report.append(f"**Urgent Actions (≤7 days):** {timeline.get('urgent_actions', 0)}")
        report.append(f"**Short-term Actions (≤30 days):** {timeline.get('short_term_actions', 0)}")
        report.append(f"**Long-term Actions (>30 days):** {timeline.get('long_term_actions', 0)}")
        report.append("")

    # Resource Requirements
    if action_plan and 'resource_requirements' in action_plan:
        resources = action_plan['resource_requirements']
        report.append("## Resource Requirements")
        report.append("")
        report.append(f"**Compliance Officers Needed:** {resources.get('compliance_officers', 0)}")
        report.append(f"**Estimated Total Hours:** {resources.get('estimated_total_hours', 0):.1f}")
        report.append(f"**Critical Actions:** {resources.get('critical_actions', 0)}")
        report.append(f"**System Updates Required:** {resources.get('system_updates_required', 0)}")
        report.append(f"**Customer Contact Required:** {resources.get('customer_contact_required', 0)}")
        report.append("")

    # Agent Performance
    agent_results = verification_summary.get("agent_results", {})
    if agent_results:
        report.append("## Agent Performance Summary")
        report.append("")
        report.append("| Agent | Article | Violations | Actions | Compliance Rate | Risk Level |")
        report.append("|-------|---------|------------|---------|-----------------|------------|")
        for agent_name, result in agent_results.items():
            compliance_rate = result.get('compliance_rate', 0)
            report.append(f"| {agent_name} | {result.get('cbuae_article', 'N/A')} | {result.get('violations_found', 0)} | {result.get('actions_generated', 0)} | {compliance_rate:.1f}% | {result.get('risk_level', 'N/A')} |")
        report.append("")

    # Recommendations
    report.append("## Key Recommendations")
    report.append("")

    # Collect recommendations from all agents
    all_recommendations = set()
    for agent_name, result in agent_results.items():
        recommendations = result.get('recommendations', [])
        all_recommendations.update(recommendations)

    if all_recommendations:
        for i, recommendation in enumerate(sorted(all_recommendations), 1):
            report.append(f"{i}. {recommendation}")
    else:
        report.append("No specific recommendations generated.")

    report.append("")

    # Footer
    report.append("---")
    report.append("")
    report.append("*This report was generated by the CBUAE Compliance Verification Agent System*")
    report.append("")

    return "\n".join(report)

# ===== MOCK DATA GENERATOR FOR TESTING =====

def generate_sample_dormant_accounts_data(num_accounts: int = 100) -> pd.DataFrame:
    """Generate sample dormant accounts data for testing compliance verification"""

    np.random.seed(42)  # For reproducible results

    account_types = ['SAVINGS', 'CURRENT', 'FIXED_DEPOSIT', 'INVESTMENT']
    currencies = ['AED', 'USD', 'EUR', 'GBP', 'SAR']
    dormancy_statuses = ['DORMANT', 'ACTIVE', 'INACTIVE', None]
    statement_frequencies = ['MONTHLY', 'QUARTERLY', 'SUPPRESSED', 'ANNUALLY']

    data = []

    for i in range(num_accounts):
        account_id = f"ACC{i+1:06d}"
        customer_id = f"CUST{i+1:06d}"

        # Generate dormancy period (months)
        dormancy_period = np.random.choice([
            np.random.randint(0, 36),      # 30% not yet dormant
            np.random.randint(36, 48),     # 25% newly dormant
            np.random.randint(48, 60),     # 25% mid-stage dormant
            np.random.randint(60, 120)     # 20% long-term dormant
        ], p=[0.3, 0.25, 0.25, 0.2])

        # Generate realistic balance
        balance = max(0, np.random.lognormal(7, 2))  # Log-normal distribution

        # Generate contact attempts based on dormancy period
        if dormancy_period >= 36:
            contact_attempts = np.random.randint(0, 5)
        else:
            contact_attempts = np.random.randint(0, 2)

        # Generate dates
        last_transaction_date = datetime.now() - timedelta(days=dormancy_period * 30)
        last_contact_date = None
        if contact_attempts > 0:
            last_contact_date = (datetime.now() - timedelta(days=np.random.randint(1, dormancy_period * 30))).strftime('%Y-%m-%d')

        # Generate other fields
        dormancy_status = np.random.choice(dormancy_statuses, p=[0.7, 0.1, 0.1, 0.1])
        currency = np.random.choice(currencies, p=[0.6, 0.15, 0.1, 0.1, 0.05])
        address_known = np.random.choice(['Yes', 'No'], p=[0.7, 0.3])

        account_data = {
            'customer_id': customer_id,
            'account_id': account_id,
            'account_type': np.random.choice(account_types),
            'account_status': 'ACTIVE' if dormancy_period < 36 else 'DORMANT',
            'balance_current': round(balance, 2),
            'last_transaction_date': last_transaction_date.strftime('%Y-%m-%d'),
            'dormancy_status': dormancy_status,
            'dormancy_period_months': dormancy_period,
            'contact_attempts_made': contact_attempts,
            'last_contact_date': last_contact_date,
            'last_contact_attempt_date': last_contact_date,
            'current_stage': 'DORMANT' if dormancy_period >= 36 else 'ACTIVE',
            'waiting_period_end': (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d'),
            'transferred_to_ledger_date': None,
            'transferred_to_cb_date': None,
            'transfer_eligibility_date': (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d'),
            'statement_frequency': np.random.choice(statement_frequencies),
            'currency': currency,
            'address_known': address_known,
            'maturity_date': None,
            'auto_renewal': 'No'
        }

        data.append(account_data)

    return pd.DataFrame(data)

# ===== MAIN EXECUTION AND TESTING =====

async def main():
    """Main function for testing compliance verification"""

    print("CBUAE Compliance Verification Agent System")
    print("=" * 60)

    # Show agent coverage
    coverage = get_compliance_agent_coverage()
    print(f"Total Compliance Agents: {coverage['total_agents']}")
    print(f"Categories: {list(coverage['categories'].keys())}")
    print(f"CBUAE Articles Covered: {coverage['total_articles_covered']}")
    print()

    # Category breakdown
    print("Agent Distribution by Category:")
    for category, count in coverage['categories'].items():
        print(f"  {category}: {count} agents")
    print()

    # Generate sample data
    print("Generating sample dormant accounts data...")
    sample_accounts = generate_sample_dormant_accounts_data(50)
    print(f"Generated {len(sample_accounts)} sample accounts")
    print()

    # Run comprehensive compliance verification
    print("Running comprehensive compliance verification...")
    verification_results = await run_standalone_compliance_verification(
        accounts_data=sample_accounts,
        user_id="test_user",
        verification_config={
            "include_recommendations": True,
            "detailed_analysis": True
        }
    )

    if verification_results and not verification_results.get("success") == False:
        print("Compliance Verification Results:")
        print("-" * 40)

        verification_summary = verification_results.get("verification_summary", {})
        compliance_summary = verification_summary.get("compliance_summary", {})

        print(f"Total Accounts Verified: {compliance_summary.get('total_accounts_verified', 0):,}")
        print(f"Total Violations Found: {compliance_summary.get('total_violations_found', 0):,}")
        print(f"Total Actions Generated: {compliance_summary.get('total_actions_generated', 0):,}")
        print(f"Overall Compliance Rate: {compliance_summary.get('overall_compliance_rate', 0):.1f}%")
        print(f"Risk Assessment: {compliance_summary.get('risk_assessment', 'Unknown')}")
        print()

        # Priority breakdown
        priority_breakdown = verification_summary.get("priority_breakdown", {})
        print("Actions by Priority:")
        for priority, count in priority_breakdown.items():
            print(f"  {priority}: {count}")
        print()

        # Generate and display report
        print("Generating comprehensive compliance report...")
        report = generate_compliance_report(verification_results)

        # Save report to file
        with open("compliance_verification_report.md", "w") as f:
            f.write(report)
        print("Report saved to: compliance_verification_report.md")
        print()

        # Export actions to CSV
        csv_output = export_compliance_actions_to_csv(verification_results, "compliance_actions.csv")
        print(f"Actions exported: {csv_output}")

    else:
        print("Compliance verification failed!")
        print(f"Error: {verification_results.get('error', 'Unknown error')}")

    print("\nCompliance verification completed.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())