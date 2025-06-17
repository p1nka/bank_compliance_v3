"""
agents/compliance_analyzer_agents.py - 17 Compliance Analyzer Agents
Works on dormant agent results and analyzes CBUAE compliance requirements
Aligned with CSV column names from banking_compliance_dataset
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

# ===== ENUMS AND DATACLASSES =====

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
    ARTICLE_2 = "Art. 2"
    ARTICLE_3_1 = "Art. 3.1"
    ARTICLE_3_4 = "Art. 3.4"
    ARTICLE_3_5 = "Art. 3.5"
    ARTICLE_3_6 = "Art. 3.6"
    ARTICLE_3_7 = "Art. 3.7"
    ARTICLE_3_9 = "Art. 3.9"
    ARTICLE_3_10 = "Art. 3.10"
    ARTICLE_4 = "Art. 4"
    ARTICLE_5 = "Art. 5"
    ARTICLE_7_3 = "Art. 7.3"
    ARTICLE_8 = "Art. 8"
    ARTICLE_8_5 = "Art. 8.5"

@dataclass
class ComplianceAction:
    """Individual compliance action to be taken"""
    account_id: str
    action_type: str
    priority: ActionPriority
    deadline_days: int
    description: str
    estimated_hours: float
    created_date: datetime
    assigned_to: str = "COMPLIANCE_TEAM"
    status: str = "PENDING"

@dataclass
class ComplianceResult:
    """Result from compliance analysis"""
    agent_name: str
    category: ComplianceCategory
    cbuae_article: str
    accounts_processed: int
    violations_found: int
    actions_generated: List[ComplianceAction]
    processing_time: float
    success: bool
    recommendations: List[str] = None
    error_message: str = None

@dataclass
class ComplianceAnalysisState:
    """Main state for compliance analysis workflow"""
    session_id: str
    user_id: str
    analysis_id: str
    timestamp: datetime
    
    # Input data from dormancy analysis
    dormancy_results: Optional[Dict] = None
    processed_accounts: Optional[pd.DataFrame] = None
    
    # Compliance analysis results
    compliance_results: Dict = None
    compliance_summary: Dict = None
    
    # Status tracking
    analysis_status: ComplianceStatus = ComplianceStatus.PENDING
    total_violations: int = 0
    total_actions: int = 0
    
    # Performance metrics
    processing_time: float = 0.0
    
    # Agent orchestration
    completed_agents: List[str] = None
    failed_agents: List[str] = None
    
    def __post_init__(self):
        if self.compliance_results is None:
            self.compliance_results = {}
        if self.completed_agents is None:
            self.completed_agents = []
        if self.failed_agents is None:
            self.failed_agents = []

# ===== BASE COMPLIANCE AGENT =====

class BaseComplianceAgent:
    """Base class for all compliance analyzer agents"""
    
    def __init__(self, agent_name: str, category: ComplianceCategory, cbuae_article: str,
                 memory_agent=None, mcp_client: MCPClient = None):
        self.agent_name = agent_name
        self.category = category
        self.cbuae_article = cbuae_article
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client or MCPClient()
        
        # CSV column mapping from banking_compliance_dataset
        self.csv_columns = {
            'customer_id': 'customer_id',
            'full_name_en': 'full_name_en',
            'account_id': 'account_id',
            'account_type': 'account_type',
            'account_status': 'account_status',
            'dormancy_status': 'dormancy_status',
            'last_transaction_date': 'last_transaction_date',
            'last_contact_date': 'last_contact_date',
            'contact_attempts_made': 'contact_attempts_made',
            'balance_current': 'balance_current',
            'dormancy_trigger_date': 'dormancy_trigger_date',
            'dormancy_period_months': 'dormancy_period_months',
            'current_stage': 'current_stage',
            'transferred_to_ledger_date': 'transferred_to_ledger_date',
            'transferred_to_cb_date': 'transferred_to_cb_date',
            'cb_transfer_amount': 'cb_transfer_amount',
            'currency': 'currency',
            'maturity_date': 'maturity_date',
            'last_statement_date': 'last_statement_date',
            'statement_frequency': 'statement_frequency'
        }
        
        # Default compliance parameters
        self.compliance_params = {
            "minimum_contact_attempts": 3,
            "waiting_period_months": 3,
            "cb_transfer_threshold_years": 5,
            "statement_suppression_months": 6,
            "record_retention_years": 10,
            "claim_processing_days": 30
        }
    
    def generate_action(self, account: pd.Series, action_type: str, priority: ActionPriority,
                       deadline_days: int, description: str, estimated_hours: float = 1.0) -> ComplianceAction:
        """Generate a compliance action for an account"""
        return ComplianceAction(
            account_id=account[self.csv_columns['account_id']],
            action_type=action_type,
            priority=priority,
            deadline_days=deadline_days,
            description=description,
            estimated_hours=estimated_hours,
            created_date=datetime.now()
        )
    
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Base compliance analysis method - to be overridden by subclasses"""
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
                (accounts_df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant'])) &
                ((accounts_df[self.csv_columns['contact_attempts_made']] < self.compliance_params["minimum_contact_attempts"]) |
                 (accounts_df[self.csv_columns['contact_attempts_made']].isna()) |
                 (accounts_df[self.csv_columns['last_contact_date']].isna()))
            ].copy()
            
            for _, account in insufficient_contact.iterrows():
                attempts_made = account.get(self.csv_columns['contact_attempts_made'], 0)
                
                if pd.isna(attempts_made) or attempts_made == 0:
                    action = self.generate_action(
                        account,
                        "INITIATE_CONTACT_ATTEMPTS",
                        ActionPriority.CRITICAL,
                        1,  # 1 day deadline
                        "No contact attempts recorded. Initiate minimum 3 contact attempts via multiple channels.",
                        2.0  # 2 hours estimated
                    )
                elif attempts_made < self.compliance_params["minimum_contact_attempts"]:
                    remaining = self.compliance_params["minimum_contact_attempts"] - int(attempts_made)
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
            cbuae_article=CBUAEArticle.ARTICLE_2.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )
    
    @traceable(name="unflagged_dormant_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Detect accounts that should be flagged as dormant but aren't"""
        start_time = datetime.now()
        actions = []
        
        try:
            # Find accounts that meet dormancy criteria but aren't flagged
            unflagged_dormant = accounts_df[
                (accounts_df[self.csv_columns['dormancy_status']].isin(['Not_Dormant', 'ACTIVE', ''])) &
                (accounts_df[self.csv_columns['account_status']] == 'ACTIVE')
            ].copy()
            
            # Calculate inactivity for each account
            current_date = datetime.now()
            
            for _, account in unflagged_dormant.iterrows():
                last_transaction = pd.to_datetime(account[self.csv_columns['last_transaction_date']], errors='coerce')
                
                if pd.notna(last_transaction):
                    days_inactive = (current_date - last_transaction).days
                    years_inactive = days_inactive / 365.25
                    
                    # Should be flagged if 3+ years inactive
                    if years_inactive >= 3:
                        action = self.generate_action(
                            account,
                            "FLAG_AS_DORMANT",
                            ActionPriority.HIGH,
                            2,  # 2 days deadline
                            f"Account inactive for {years_inactive:.1f} years. Flag as dormant per Article 2.",
                            1.0  # 1 hour estimated
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
                (accounts_df[self.csv_columns['contact_attempts_made']] >= self.compliance_params["minimum_contact_attempts"]) &
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
                    f"Transfer to internal dormant ledger. Balance: {balance:,.2f}, Dormant: {dormant_months} months",
                    2.0  # 2 hours estimated
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
                    "Automate internal ledger transfer process",
                    "Establish clear waiting period tracking",
                    "Regular review of transfer candidates"
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
        """Analyze statement suppression compliance"""
        start_time = datetime.now()
        actions = []
        
        try:
            # Accounts eligible for statement suppression
            current_date = datetime.now()
            
            freeze_candidates = accounts_df[
                (accounts_df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant'])) &
                (accounts_df[self.csv_columns['dormancy_period_months']] >= self.compliance_params["statement_suppression_months"])
            ].copy()
            
            for _, account in freeze_candidates.iterrows():
                last_statement = pd.to_datetime(account[self.csv_columns['last_statement_date']], errors='coerce')
                dormant_months = account.get(self.csv_columns['dormancy_period_months'], 0)
                
                # Check if statements are still being generated
                if pd.notna(last_statement):
                    months_since_statement = (current_date - last_statement).days / 30.44
                    
                    if months_since_statement < 3:  # Recent statement generation
                        action = self.generate_action(
                            account,
                            "SUPPRESS_STATEMENTS",
                            ActionPriority.MEDIUM,
                            7,  # 7 days deadline
                            f"Suppress statement generation. Account dormant for {dormant_months} months.",
                            0.5  # 0.5 hours estimated
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
                    "Automate statement suppression for dormant accounts",
                    "Establish clear dormancy period thresholds",
                    "Monitor statement generation costs"
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
        """Analyze CBUAE transfer compliance"""
        start_time = datetime.now()
        actions = []
        
        try:
            # Accounts eligible for CBUAE transfer (5+ years dormant)
            cb_transfer_months = self.compliance_params["cb_transfer_threshold_years"] * 12
            
            transfer_candidates = accounts_df[
                (accounts_df[self.csv_columns['dormancy_period_months']] >= cb_transfer_months) &
                (accounts_df[self.csv_columns['balance_current']] > 0) &
                ((accounts_df[self.csv_columns['transferred_to_cb_date']].isna()) |
                 (accounts_df[self.csv_columns['transferred_to_cb_date']] == ''))
            ].copy()
            
            for _, account in transfer_candidates.iterrows():
                balance = account.get(self.csv_columns['balance_current'], 0)
                dormant_months = account.get(self.csv_columns['dormancy_period_months'], 0)
                dormant_years = dormant_months / 12
                
                action = self.generate_action(
                    account,
                    "TRANSFER_TO_CBUAE",
                    ActionPriority.CRITICAL,
                    3,  # 3 days deadline
                    f"Transfer to CBUAE. Balance: {balance:,.2f}, Dormant: {dormant_years:.1f} years",
                    3.0  # 3 hours estimated
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
                    "Establish automated CBUAE transfer process",
                    "Implement 5-year dormancy alerts",
                    "Regular coordination with CBUAE for transfers"
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
    
    @traceable(name="fx_conversion_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze foreign currency conversion requirements"""
        start_time = datetime.now()
        actions = []
        
        try:
            # Foreign currency accounts requiring conversion before CBUAE transfer
            fx_conversion_candidates = accounts_df[
                (accounts_df[self.csv_columns['currency']] != 'AED') &
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 48) &  # Near 5-year threshold
                (accounts_df[self.csv_columns['balance_current']] > 0)
            ].copy()
            
            for _, account in fx_conversion_candidates.iterrows():
                currency = account.get(self.csv_columns['currency'], 'UNKNOWN')
                balance = account.get(self.csv_columns['balance_current'], 0)
                dormant_months = account.get(self.csv_columns['dormancy_period_months'], 0)
                
                action = self.generate_action(
                    account,
                    "CONVERT_TO_AED",
                    ActionPriority.HIGH,
                    10,  # 10 days deadline
                    f"Convert {currency} {balance:,.2f} to AED before CBUAE transfer. Dormant: {dormant_months} months",
                    1.5  # 1.5 hours estimated
                )
                actions.append(action)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(fx_conversion_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Establish FX conversion protocols for dormant accounts",
                    "Monitor exchange rate impacts",
                    "Early identification of FX conversion needs"
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


class DetectSDBCourtApplicationNeededAgent(BaseComplianceAgent):
    """Specialized Compliance - Art. 3.7: Safe Deposit Box court application"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_sdb_court_application_needed",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article=CBUAEArticle.ARTICLE_3_7.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="sdb_court_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze Safe Deposit Box court application requirements"""
        start_time = datetime.now()
        actions = []

        try:
            # Safe Deposit Box accounts requiring court application
            sdb_court_candidates = accounts_df[
                (accounts_df[self.csv_columns['account_type']].str.contains('SDB|SAFE_DEPOSIT', case=False, na=False)) &
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 36) &  # 3+ years
                (accounts_df[self.csv_columns['contact_attempts_made']] >= self.compliance_params[
                    "minimum_contact_attempts"])
                ].copy()

            for _, account in sdb_court_candidates.iterrows():
                dormant_months = account.get(self.csv_columns['dormancy_period_months'], 0)
                dormant_years = dormant_months / 12

                action = self.generate_action(
                    account,
                    "INITIATE_COURT_APPLICATION",
                    ActionPriority.HIGH,
                    14,  # 14 days deadline
                    f"Initiate court application for SDB access. Dormant: {dormant_years:.1f} years",
                    4.0  # 4 hours estimated
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(sdb_court_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Establish SDB court application procedures",
                    "Legal team coordination for court processes",
                    "Track court application timelines"
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


class DetectUnclaimedPaymentInstrumentsLedgerAgent(BaseComplianceAgent):
    """Specialized Compliance - Art. 3.6: Unclaimed instruments ledger"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_unclaimed_payment_instruments_ledger",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article=CBUAEArticle.ARTICLE_3_6.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="unclaimed_instruments_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze unclaimed payment instruments ledger requirements"""
        start_time = datetime.now()
        actions = []

        try:
            # Unclaimed payment instruments requiring ledger transfer
            unclaimed_candidates = accounts_df[
                (accounts_df[self.csv_columns['account_subtype']].str.contains('INSTRUMENT|CHEQUE|DRAFT', case=False,
                                                                               na=False)) &
                (accounts_df[self.csv_columns['dormancy_period_months']] >= 12) &  # 1+ year unclaimed
                (accounts_df[self.csv_columns['balance_current']] > 0)
                ].copy()

            for _, account in unclaimed_candidates.iterrows():
                balance = account.get(self.csv_columns['balance_current'], 0)
                dormant_months = account.get(self.csv_columns['dormancy_period_months'], 0)

                action = self.generate_action(
                    account,
                    "TRANSFER_TO_UNCLAIMED_LEDGER",
                    ActionPriority.MEDIUM,
                    7,  # 7 days deadline
                    f"Transfer unclaimed instrument to ledger. Amount: {balance:,.2f}, Unclaimed: {dormant_months} months",
                    1.0  # 1 hour estimated
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(unclaimed_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Establish unclaimed instruments tracking",
                    "Regular review of payment instrument status",
                    "Automated ledger transfer process"
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


class DetectClaimProcessingPendingAgent(BaseComplianceAgent):
    """Specialized Compliance - Art. 4: Customer claims processing"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="detect_claim_processing_pending",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article=CBUAEArticle.ARTICLE_4.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="claim_processing_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Analyze customer claims processing compliance"""
        start_time = datetime.now()
        actions = []

        try:
            # Accounts with potential pending claims (based on recent activity on dormant accounts)
            current_date = datetime.now()

            claim_candidates = accounts_df[
                (accounts_df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant'])) &
                (accounts_df[self.csv_columns['account_status']] == 'ACTIVE')  # Recently activated dormant accounts
                ].copy()

            for _, account in claim_candidates.iterrows():
                last_transaction = pd.to_datetime(account[self.csv_columns['last_transaction_date']], errors='coerce')

                if pd.notna(last_transaction):
                    days_since_transaction = (current_date - last_transaction).days

                    # Recent activity on previously dormant account suggests claim
                    if days_since_transaction <= 30:
                        action = self.generate_action(
                            account,
                            "PROCESS_DORMANT_CLAIM",
                            ActionPriority.HIGH,
                            self.compliance_params["claim_processing_days"],
                            f"Process dormant account claim. Recent activity: {days_since_transaction} days ago",
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
                    "Establish claim processing procedures",
                    "Track claim resolution timelines",
                    "Customer communication for claims"
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

    @traceable(name="annual_report_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Generate annual CBUAE report summary"""
        start_time = datetime.now()
        actions = []

        try:
            current_date = datetime.now()
            current_year = current_date.year

            # Calculate annual statistics
            total_dormant = len(
                accounts_df[accounts_df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant'])])
            total_transferred_cb = len(accounts_df[accounts_df[self.csv_columns['transferred_to_cb_date']].notna()])
            total_value_dormant = \
            accounts_df[accounts_df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant'])][
                self.csv_columns['balance_current']].sum()

            # Check if annual report is due (December each year)
            if current_date.month == 12:
                action = ComplianceAction(
                    account_id="ANNUAL_REPORT",
                    action_type="GENERATE_ANNUAL_REPORT",
                    priority=ActionPriority.CRITICAL,
                    deadline_days=31,  # End of year deadline
                    description=f"Generate {current_year} annual CBUAE dormancy report. Total dormant: {total_dormant}, CB transfers: {total_transferred_cb}, Value: {total_value_dormant:,.2f}",
                    estimated_hours=8.0,
                    created_date=datetime.now()
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
                    "Establish annual reporting calendar",
                    "Automate report data compilation",
                    "CBUAE submission process documentation"
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
    """Reporting & Retention - Art. 3.9: Record retention compliance"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="check_record_retention_compliance",
            category=ComplianceCategory.REPORTING_RETENTION,
            cbuae_article=CBUAEArticle.ARTICLE_3_9.value,
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="record_retention_compliance")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Check record retention compliance"""
        start_time = datetime.now()
        actions = []

        try:
            current_date = datetime.now()
            retention_years = self.compliance_params["record_retention_years"]

            # Check accounts transferred to CB more than retention period ago
            cb_transferred = accounts_df[accounts_df[self.csv_columns['transferred_to_cb_date']].notna()].copy()

            for _, account in cb_transferred.iterrows():
                transfer_date = pd.to_datetime(account[self.csv_columns['transferred_to_cb_date']], errors='coerce')

                if pd.notna(transfer_date):
                    years_since_transfer = (current_date - transfer_date).days / 365.25

                    if years_since_transfer >= retention_years:
                        action = self.generate_action(
                            account,
                            "ARCHIVE_RECORDS",
                            ActionPriority.LOW,
                            30,  # 30 days deadline
                            f"Archive records for CB transferred account. Transfer: {years_since_transfer:.1f} years ago",
                            0.5  # 0.5 hours estimated
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
                    "Implement automated record archiving",
                    "Establish retention policy procedures",
                    "Regular review of retention requirements"
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


# ===== UTILITY AGENTS =====

class LogFlagInstructionsAgent(BaseComplianceAgent):
    """Utility - Internal: Flagging instruction logger"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="log_flag_instructions",
            category=ComplianceCategory.UTILITY,
            cbuae_article="Internal",
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

    @traceable(name="flag_logging_utility")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Log flagging instructions for compliance tracking"""
        start_time = datetime.now()
        actions = []

        try:
            # Generate flagging instructions for all compliance actions
            flagging_candidates = accounts_df[
                accounts_df[self.csv_columns['dormancy_status']].isin(['DORMANT', 'Dormant', 'POTENTIALLY_DORMANT'])
            ].copy()

            for _, account in flagging_candidates.iterrows():
                action = self.generate_action(
                    account,
                    "LOG_COMPLIANCE_FLAG",
                    ActionPriority.LOW,
                    1,  # 1 day deadline
                    "Log compliance flagging instruction for audit trail",
                    0.1  # 0.1 hours estimated
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=len(flagging_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    "Automate compliance flag logging",
                    "Establish audit trail procedures",
                    "Regular review of flagging instructions"
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


# ===== UTILITY ALIAS AGENTS =====

class DetectFlagCandidatesAgent(DetectUnflaggedDormantCandidatesAgent):
    """Utility - Art. 2: Alias for unflagged dormant detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(memory_agent, mcp_client)
        self.agent_name = "detect_flag_candidates"


class DetectLedgerCandidatesAgent(DetectInternalLedgerCandidatesAgent):
    """Utility - Art. 3.4, 3.5: Alias for internal ledger detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(memory_agent, mcp_client)
        self.agent_name = "detect_ledger_candidates"


class DetectFreezeCandidatesAgent(DetectStatementFreezeCandidatesAgent):
    """Utility - Art. 7.3: Alias for statement freeze detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(memory_agent, mcp_client)
        self.agent_name = "detect_freeze_candidates"


class DetectTransferCandidatesToCBAgent(DetectCBUAETransferCandidatesAgent):
    """Utility - Art. 8: Alias for CBUAE transfer detection"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(memory_agent, mcp_client)
        self.agent_name = "detect_transfer_candidates_to_cb"


# ===== MASTER ORCHESTRATOR =====

class RunAllComplianceChecksAgent(BaseComplianceAgent):
    """Master Orchestrator - All: Main compliance analysis orchestrator"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        super().__init__(
            agent_name="run_all_compliance_checks",
            category=ComplianceCategory.UTILITY,
            cbuae_article="All",
            memory_agent=memory_agent,
            mcp_client=mcp_client
        )

        # Initialize all compliance agents
        self.compliance_agents = {
            # Contact & Communication (2 agents)
            "incomplete_contact": DetectIncompleteContactAttemptsAgent(memory_agent, mcp_client),
            "unflagged_dormant": DetectUnflaggedDormantCandidatesAgent(memory_agent, mcp_client),

            # Process Management (3 agents)
            "internal_ledger": DetectInternalLedgerCandidatesAgent(memory_agent, mcp_client),
            "statement_freeze": DetectStatementFreezeCandidatesAgent(memory_agent, mcp_client),
            "cbuae_transfer": DetectCBUAETransferCandidatesAgent(memory_agent, mcp_client),

            # Specialized Compliance (4 agents)
            "fx_conversion": DetectForeignCurrencyConversionNeededAgent(memory_agent, mcp_client),
            "sdb_court": DetectSDBCourtApplicationNeededAgent(memory_agent, mcp_client),
            "unclaimed_instruments": DetectUnclaimedPaymentInstrumentsLedgerAgent(memory_agent, mcp_client),
            "claim_processing": DetectClaimProcessingPendingAgent(memory_agent, mcp_client),

            # Reporting & Retention (2 agents)
            "annual_report": GenerateAnnualCBUAEReportSummaryAgent(memory_agent, mcp_client),
            "record_retention": CheckRecordRetentionComplianceAgent(memory_agent, mcp_client),

            # Utility (5 agents)
            "log_flags": LogFlagInstructionsAgent(memory_agent, mcp_client),
            "flag_candidates": DetectFlagCandidatesAgent(memory_agent, mcp_client),
            "ledger_candidates": DetectLedgerCandidatesAgent(memory_agent, mcp_client),
            "freeze_candidates": DetectFreezeCandidatesAgent(memory_agent, mcp_client),
            "transfer_candidates": DetectTransferCandidatesToCBAgent(memory_agent, mcp_client)
        }

    @traceable(name="comprehensive_compliance_analysis")
    async def analyze_compliance(self, accounts_df: pd.DataFrame) -> ComplianceResult:
        """Run comprehensive compliance analysis using all 17 agents"""
        start_time = datetime.now()
        all_actions = []
        agent_results = {}

        try:
            # Run all compliance agents
            for agent_name, agent in self.compliance_agents.items():
                try:
                    result = await agent.analyze_compliance(accounts_df)
                    agent_results[agent_name] = result
                    all_actions.extend(result.actions_generated)

                except Exception as e:
                    logger.error(f"Compliance agent {agent_name} failed: {e}")
                    agent_results[agent_name] = ComplianceResult(
                        agent_name=agent_name,
                        category=agent.category,
                        cbuae_article=agent.cbuae_article,
                        accounts_processed=len(accounts_df),
                        violations_found=0,
                        actions_generated=[],
                        processing_time=0,
                        success=False,
                        error_message=str(e)
                    )

            # Calculate summary statistics
            total_violations = sum(result.violations_found for result in agent_results.values())
            successful_agents = sum(1 for result in agent_results.values() if result.success)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(accounts_df),
                violations_found=total_violations,
                actions_generated=all_actions,
                processing_time=processing_time,
                success=True,
                recommendations=[
                    f"Successfully analyzed {successful_agents}/{len(self.compliance_agents)} compliance areas",
                    f"Generated {len(all_actions)} compliance actions across all areas",
                    "Prioritize CRITICAL and HIGH priority actions for immediate attention",
                    "Establish automated compliance monitoring workflows"
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


# ===== COMPLIANCE WORKFLOW ORCHESTRATOR =====

class ComplianceWorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for comprehensive compliance analysis"""

    def __init__(self, memory_agent=None, mcp_client: MCPClient = None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client or MCPClient()

        # Initialize master orchestrator
        self.master_agent = RunAllComplianceChecksAgent(memory_agent, mcp_client)

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for compliance analysis"""
        workflow = StateGraph(ComplianceAnalysisState)

        # Add workflow nodes
        workflow.add_node("run_compliance_analysis", self._run_compliance_analysis)
        workflow.add_node("summarize_compliance", self._summarize_compliance)

        # Define workflow edges
        workflow.add_edge(START, "run_compliance_analysis")
        workflow.add_edge("run_compliance_analysis", "summarize_compliance")
        workflow.add_edge("summarize_compliance", END)

        return workflow.compile(checkpointer=MemorySaver())

    async def _run_compliance_analysis(self, state: ComplianceAnalysisState) -> ComplianceAnalysisState:
        """Run comprehensive compliance analysis"""
        try:
            if state.processed_accounts is not None:
                result = await self.master_agent.analyze_compliance(state.processed_accounts)

                state.compliance_results["master_analysis"] = result
                state.total_violations = result.violations_found
                state.total_actions = len(result.actions_generated)

                if result.success:
                    state.completed_agents.append("run_all_compliance_checks")
                else:
                    state.failed_agents.append("run_all_compliance_checks")

        except Exception as e:
            logger.error(f"Compliance analysis failed: {e}")
            state.failed_agents.append("run_all_compliance_checks")

        return state

    async def _summarize_compliance(self, state: ComplianceAnalysisState) -> ComplianceAnalysisState:
        """Summarize compliance analysis results"""
        try:
            # Generate compliance summary
            if state.compliance_results:
                master_result = state.compliance_results.get("master_analysis")

                if master_result:
                    # Categorize actions by priority
                    priority_counts = {
                        "CRITICAL": len(
                            [a for a in master_result.actions_generated if a.priority == ActionPriority.CRITICAL]),
                        "HIGH": len([a for a in master_result.actions_generated if a.priority == ActionPriority.HIGH]),
                        "MEDIUM": len(
                            [a for a in master_result.actions_generated if a.priority == ActionPriority.MEDIUM]),
                        "LOW": len([a for a in master_result.actions_generated if a.priority == ActionPriority.LOW])
                    }

                    state.compliance_summary = {
                        "total_accounts_analyzed": master_result.accounts_processed,
                        "total_violations_found": master_result.violations_found,
                        "total_actions_generated": len(master_result.actions_generated),
                        "priority_breakdown": priority_counts,
                        "processing_time": master_result.processing_time,
                        "success_rate": f"{len(state.completed_agents)}/{len(state.completed_agents) + len(state.failed_agents)}",
                        "recommendations": master_result.recommendations,
                        "analysis_timestamp": datetime.now().isoformat()
                    }

                    state.analysis_status = ComplianceStatus.COMPLIANT if state.total_violations == 0 else ComplianceStatus.ACTION_REQUIRED
                else:
                    state.analysis_status = ComplianceStatus.PENDING

            state.processing_time = sum(
                result.processing_time for result in state.compliance_results.values()
                if hasattr(result, 'processing_time')
            )

        except Exception as e:
            logger.error(f"Compliance summary failed: {e}")
            state.analysis_status = ComplianceStatus.PENDING

        return state

    async def run_comprehensive_compliance_analysis(self, state: ComplianceAnalysisState) -> ComplianceAnalysisState:
        """Run comprehensive compliance analysis workflow"""
        try:
            start_time = datetime.now()

            # Execute workflow
            result = await self.workflow.ainvoke(state)

            # Update final processing time
            result.processing_time = (datetime.now() - start_time).total_seconds()

            return result

        except Exception as e:
            state.analysis_status = ComplianceStatus.PENDING
            logger.error(f"Comprehensive compliance analysis failed: {e}")
            return state


# ===== FACTORY FUNCTIONS =====

def create_compliance_analysis_agent(memory_agent=None, mcp_client: MCPClient = None) -> ComplianceWorkflowOrchestrator:
    """Factory function to create comprehensive compliance analysis agent"""
    return ComplianceWorkflowOrchestrator(memory_agent, mcp_client)


async def run_comprehensive_compliance_analysis_csv(user_id: str, dormancy_results: Dict,
                                                    accounts_df: pd.DataFrame,
                                                    memory_agent=None, mcp_client: MCPClient = None) -> Dict:
    """
    Run comprehensive compliance analysis on dormancy results using CSV data

    Args:
        user_id: User identifier
        dormancy_results: Results from dormancy analysis agents
        accounts_df: DataFrame containing account information with CSV column names
        memory_agent: Memory agent instance
        mcp_client: MCP client instance

    Returns:
        Dictionary containing comprehensive compliance analysis results
    """
    try:
        # Initialize compliance analysis agent
        compliance_agent = ComplianceWorkflowOrchestrator(memory_agent, mcp_client)

        # Create compliance analysis state
        compliance_state = ComplianceAnalysisState(
            session_id=secrets.token_hex(16),
            user_id=user_id,
            analysis_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            dormancy_results=dormancy_results,
            processed_accounts=accounts_df
        )

        # Execute comprehensive compliance analysis
        final_state = await compliance_agent.run_comprehensive_compliance_analysis(compliance_state)

        # Return results
        return {
            "success": final_state.analysis_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.ACTION_REQUIRED],
            "session_id": final_state.session_id,
            "compliance_status": final_state.analysis_status.value,
            "total_violations": final_state.total_violations,
            "total_actions": final_state.total_actions,
            "compliance_results": final_state.compliance_results,
            "compliance_summary": final_state.compliance_summary,
            "processing_time_seconds": final_state.processing_time,
            "completed_agents": final_state.completed_agents,
            "failed_agents": final_state.failed_agents,
            "analysis_timestamp": final_state.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Comprehensive compliance analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "session_id": None,
            "compliance_results": None
        }


def validate_compliance_requirements(accounts_df: pd.DataFrame) -> Dict:
    """Validate that the DataFrame meets compliance analysis requirements"""
    required_columns = [
        'customer_id', 'account_id', 'account_type', 'account_status',
        'dormancy_status', 'contact_attempts_made', 'balance_current'
    ]

    missing_columns = [col for col in required_columns if col not in accounts_df.columns]

    # Data quality checks
    quality_issues = []

    # Check for null values in critical columns
    for col in ['customer_id', 'account_id', 'dormancy_status']:
        if col in accounts_df.columns:
            null_count = accounts_df[col].isnull().sum()
            if null_count > 0:
                quality_issues.append(f"{col} has {null_count} null values")

    # Check dormancy status values
    if 'dormancy_status' in accounts_df.columns:
        valid_statuses = ['DORMANT', 'Dormant', 'Not_Dormant', 'ACTIVE', 'POTENTIALLY_DORMANT']
        invalid_statuses = accounts_df[~accounts_df['dormancy_status'].isin(valid_statuses + ['', None])][
            'dormancy_status'].unique()
        if len(invalid_statuses) > 0:
            quality_issues.append(f"Invalid dormancy statuses found: {list(invalid_statuses)}")

    return {
        "requirements_met": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "quality_issues": quality_issues,
        "total_records": len(accounts_df),
        "dormant_records": len(accounts_df[accounts_df.get('dormancy_status', '').isin(
            ['DORMANT', 'Dormant'])]) if 'dormancy_status' in accounts_df.columns else 0,
        "validation_timestamp": datetime.now().isoformat()
    }


# ===== EXPORT DEFINITIONS =====

__all__ = [
    # Core Analysis Components
    "ComplianceWorkflowOrchestrator",
    "ComplianceAnalysisState",
    "ComplianceResult",
    "ComplianceAction",

    # Status and Category Enums
    "ComplianceStatus",
    "ActionPriority",
    "ComplianceCategory",
    "CBUAEArticle",

    # Base Agent Class
    "BaseComplianceAgent",

    # Contact & Communication Agents (2 agents)
    "DetectIncompleteContactAttemptsAgent",  # Art. 3.1, 5
    "DetectUnflaggedDormantCandidatesAgent",  # Art. 2

    # Process Management Agents (3 agents)
    "DetectInternalLedgerCandidatesAgent",  # Art. 3.4, 3.5
    "DetectStatementFreezeCandidatesAgent",  # Art. 7.3
    "DetectCBUAETransferCandidatesAgent",  # Art. 8

    # Specialized Compliance Agents (4 agents)
    "DetectForeignCurrencyConversionNeededAgent",  # Art. 8.5
    "DetectSDBCourtApplicationNeededAgent",  # Art. 3.7
    "DetectUnclaimedPaymentInstrumentsLedgerAgent",  # Art. 3.6
    "DetectClaimProcessingPendingAgent",  # Art. 4

    # Reporting & Retention Agents (2 agents)
    "GenerateAnnualCBUAEReportSummaryAgent",  # Art. 3.10
    "CheckRecordRetentionComplianceAgent",  # Art. 3.9

    # Utility Agents (5 agents)
    "LogFlagInstructionsAgent",  # Internal
    "DetectFlagCandidatesAgent",  # Art. 2 (Alias)
    "DetectLedgerCandidatesAgent",  # Art. 3.4, 3.5 (Alias)
    "DetectFreezeCandidatesAgent",  # Art. 7.3 (Alias)
    "DetectTransferCandidatesToCBAgent",  # Art. 8 (Alias)

    # Master Orchestrator (1 agent)
    "RunAllComplianceChecksAgent",  # All Articles

    # Factory Functions
    "create_compliance_analysis_agent",
    "run_comprehensive_compliance_analysis_csv",
    "validate_compliance_requirements"
]

# Total: 17 compliance analyzer agents as per the specification
# Categories:
# - Contact & Communication: 2 agents
# - Process Management: 3 agents
# - Specialized Compliance: 4 agents
# - Reporting & Retention: 2 agents
# - Utility: 5 agents
# - Master Orchestrator: 1 agent