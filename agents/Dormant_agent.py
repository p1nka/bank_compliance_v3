"""
agents/Dormant_agent.py - CBUAE Dormancy Analysis Agents
Clean module matching the UI dashboard - No mock responses
"""

import pandas as pd
import numpy as np

import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ENUMS =====

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DormancyStatus(Enum):
    PENDING_REVIEW = "pending_review"
    ACTION_REQUIRED = "action_required"
    UP_TO_DATE = "up_to_date"
    PROCESSING = "processing"
    CRITICAL = "critical"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    URGENT = "urgent"
    PRIORITY = "priority"
    MONITORED = "monitored"

# ===== DATA CLASSES =====

@dataclass
class AgentState:
    agent_id: str
    agent_type: str
    session_id: str
    user_id: str
    timestamp: datetime
    input_dataframe: Optional[pd.DataFrame] = None
    agent_status: AgentStatus = AgentStatus.IDLE
    records_processed: int = 0
    dormant_records_found: int = 0
    processing_time: float = 0.0
    processed_dataframe: Optional[pd.DataFrame] = None
    analysis_results: Optional[Dict] = None
    error_log: List[Dict] = field(default_factory=list)

@dataclass
class AgentResult:
    agent_name: str
    agent_type: str
    cbuae_article: str
    records_processed: int
    dormant_records_found: int
    processing_time: float
    success: bool
    status: DormancyStatus
    detailed_results_df: Optional[pd.DataFrame] = None
    analysis_summary: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

# ===== UTILITY FUNCTIONS =====

def safe_date_parse(date_string: Union[str, datetime, None]) -> Optional[datetime]:
    """Safely parse date string to datetime object"""
    if date_string is None or pd.isna(date_string):
        return None
    if isinstance(date_string, datetime):
        return date_string
    try:
        return pd.to_datetime(date_string)
    except:
        return None

def calculate_dormancy_days(last_activity_date: Union[str, datetime],
                          report_date: Union[str, datetime] = None) -> int:
    """Calculate number of days account has been dormant"""
    if report_date is None:
        report_date = datetime.now()

    last_date = safe_date_parse(last_activity_date)
    report_date = safe_date_parse(report_date)

    if last_date is None or report_date is None:
        return 0

    return (report_date - last_date).days

def create_csv_download_data(df: pd.DataFrame, filename: str) -> Dict:
    """Create CSV download data structure"""
    if df is None or df.empty:
        return {"available": False, "filename": filename, "records": 0}

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    csv_bytes = csv_string.encode('utf-8')
    csv_base64 = base64.b64encode(csv_bytes).decode('utf-8')

    return {
        "available": True,
        "filename": filename,
        "records": len(df),
        "csv_data": csv_string,
        "csv_base64": csv_base64,
        "download_link": f"data:text/csv;base64,{csv_base64}",
        "file_size_kb": len(csv_bytes) / 1024
    }

# ===== BASE AGENT CLASS =====

class BaseDormancyAgent:
    """Base class for all dormancy analysis agents"""

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.agent_id = f"{agent_type}_{secrets.token_hex(8)}"

        # Standard column mappings for banking data
        self.csv_columns = {
            'customer_id': 'customer_id',
            'account_id': 'account_id',
            'account_type': 'account_type',
            'account_status': 'account_status',
            'last_transaction_date': 'last_transaction_date',
            'balance_current': 'balance_current',
            'dormancy_status': 'dormancy_status',
            'currency': 'currency',
            'contact_attempts_made': 'contact_attempts_made',
            'last_contact_date': 'last_contact_date'
        }

    def create_agent_state(self, user_id: str, dataframe: pd.DataFrame) -> AgentState:
        """Create agent state for analysis"""
        return AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            session_id=f"session_{secrets.token_hex(8)}",
            user_id=user_id,
            timestamp=datetime.now(),
            input_dataframe=dataframe,
            agent_status=AgentStatus.IDLE
        )

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Base analyze method - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement analyze_dormancy")

# ===== SPECIFIC DORMANCY AGENTS MATCHING UI =====

class SafeDepositDormancyAgent(BaseDormancyAgent):
    """Safe Deposit Dormancy Analysis - CBUAE Article 3.7"""

    def __init__(self):
        super().__init__("safe_deposit_dormancy")
        self.cbuae_article = "CBUAE Art. 3.7"
        self.ui_status = DormancyStatus.PENDING_REVIEW

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze safe deposit boxes requiring court applications"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Filter for safe deposit box accounts
            safe_deposits = df[
                df[self.csv_columns['account_type']].str.contains(
                    'SAFE_DEPOSIT|SDB|Safe Deposit', case=False, na=False
                )
            ].copy()

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in safe_deposits.iterrows():
                last_transaction = account[self.csv_columns['last_transaction_date']]
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 3.7: 2+ years for court application
                if dormancy_days >= 730:
                    dormant_accounts.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'dormancy_days': dormancy_days,
                        'compliance_article': self.cbuae_article,
                        'action_required': 'File court application for box access',
                        'priority': 'High',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(safe_deposits)
            state.dormant_records_found = len(dormant_accounts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if dormant_accounts:
                state.processed_dataframe = pd.DataFrame(dormant_accounts)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"safe_deposit_dormancy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'Safe deposit dormancy analysis per CBUAE Article 3.7',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'dormant_accounts': dormant_accounts,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(safe_deposits),
                    'dormant_found': len(dormant_accounts)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "safe_deposit_analysis"
            })
            logger.error(f"Safe deposit analysis failed: {e}")
            return state

class InvestmentAccountInactivityAgent(BaseDormancyAgent):
    """Investment Account Inactivity Analysis - CBUAE Article 2.2"""

    def __init__(self):
        super().__init__("investment_account_inactivity")
        self.cbuae_article = "CBUAE Art. 2.2"
        self.ui_status = DormancyStatus.ACTION_REQUIRED

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze investment account inactivity requiring customer contact"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Filter for investment accounts
            investment_accounts = df[
                df[self.csv_columns['account_type']].str.contains(
                    'INVESTMENT|PORTFOLIO|MUTUAL|SECURITIES', case=False, na=False
                )
            ].copy()

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in investment_accounts.iterrows():
                last_transaction = account[self.csv_columns['last_transaction_date']]
                balance = account[self.csv_columns['balance_current']]
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 2.2: 12 months for investment products
                if dormancy_days >= 365:
                    dormant_accounts.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Review investment product status and contact customer',
                        'priority': 'High',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(investment_accounts)
            state.dormant_records_found = len(dormant_accounts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if dormant_accounts:
                state.processed_dataframe = pd.DataFrame(dormant_accounts)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"investment_inactivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'Investment account inactivity analysis per CBUAE Article 2.2',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'dormant_accounts': dormant_accounts,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(investment_accounts),
                    'dormant_found': len(dormant_accounts)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "investment_inactivity_analysis"
            })
            logger.error(f"Investment inactivity analysis failed: {e}")
            return state

class FixedDepositInactivityAgent(BaseDormancyAgent):
    """Fixed Deposit Inactivity Analysis - CBUAE Article 2.1.2"""

    def __init__(self):
        super().__init__("fixed_deposit_inactivity")
        self.cbuae_article = "CBUAE Art. 2.1.2"
        self.ui_status = DormancyStatus.UP_TO_DATE

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze fixed deposit maturity monitoring"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Filter for fixed deposits
            fixed_deposits = df[
                df[self.csv_columns['account_type']].str.contains(
                    'FIXED_DEPOSIT|TERM_DEPOSIT|CD|CERTIFICATE', case=False, na=False
                )
            ].copy()

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in fixed_deposits.iterrows():
                last_transaction = account[self.csv_columns['last_transaction_date']]
                balance = account[self.csv_columns['balance_current']]
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 2.1.2: 12 months post-maturity
                if dormancy_days >= 365:
                    dormant_accounts.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Monitor maturity dates and contact customer',
                        'priority': 'Medium',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(fixed_deposits)
            state.dormant_records_found = len(dormant_accounts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if dormant_accounts:
                state.processed_dataframe = pd.DataFrame(dormant_accounts)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"fixed_deposit_inactivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'Fixed deposit inactivity analysis per CBUAE Article 2.1.2',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'dormant_accounts': dormant_accounts,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(fixed_deposits),
                    'dormant_found': len(dormant_accounts)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "fixed_deposit_inactivity_analysis"
            })
            logger.error(f"Fixed deposit inactivity analysis failed: {e}")
            return state

class DemandDepositInactivityAgent(BaseDormancyAgent):
    """Demand Deposit Inactivity Analysis - CBUAE Article 2.1.1"""

    def __init__(self):
        super().__init__("demand_deposit_inactivity")
        self.cbuae_article = "CBUAE Art. 2.1.1"
        self.ui_status = DormancyStatus.PROCESSING

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze demand deposit accounts for dormancy flagging"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Filter for demand deposits (current/savings accounts)
            demand_deposits = df[
                df[self.csv_columns['account_type']].str.contains(
                    'CURRENT|SAVINGS|CHECKING', case=False, na=False
                )
            ].copy()

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in demand_deposits.iterrows():
                last_transaction = account[self.csv_columns['last_transaction_date']]
                balance = account[self.csv_columns['balance_current']]
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 2.1.1: 12 months inactivity
                if dormancy_days >= 365:
                    dormant_accounts.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Flag as dormant and initiate contact',
                        'priority': 'Medium',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(demand_deposits)
            state.dormant_records_found = len(dormant_accounts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if dormant_accounts:
                state.processed_dataframe = pd.DataFrame(dormant_accounts)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"demand_deposit_inactivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'Demand deposit inactivity analysis per CBUAE Article 2.1.1',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'dormant_accounts': dormant_accounts,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(demand_deposits),
                    'dormant_found': len(dormant_accounts)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "demand_deposit_inactivity_analysis"
            })
            logger.error(f"Demand deposit inactivity analysis failed: {e}")
            return state

class UnclaimedPaymentInstrumentsAgent(BaseDormancyAgent):
    """Unclaimed Payment Instruments Analysis - CBUAE Article 3.6"""

    def __init__(self):
        super().__init__("unclaimed_payment_instruments")
        self.cbuae_article = "CBUAE Art. 3.6"
        self.ui_status = DormancyStatus.CRITICAL

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze unclaimed payment instruments requiring ledger transfer"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Filter for payment instruments
            payment_instruments = df[
                df[self.csv_columns['account_type']].str.contains(
                    'PAYMENT|CHECK|DRAFT|INSTRUMENT|CHEQUE', case=False, na=False
                )
            ].copy()

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in payment_instruments.iterrows():
                last_transaction = account[self.csv_columns['last_transaction_date']]
                balance = account[self.csv_columns['balance_current']]
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 3.6: 1+ year unclaimed
                if dormancy_days >= 365:
                    dormant_accounts.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Process for ledger transfer',
                        'priority': 'Critical',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(payment_instruments)
            state.dormant_records_found = len(dormant_accounts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if dormant_accounts:
                state.processed_dataframe = pd.DataFrame(dormant_accounts)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"unclaimed_payment_instruments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'Unclaimed payment instruments analysis per CBUAE Article 3.6',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'dormant_accounts': dormant_accounts,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(payment_instruments),
                    'dormant_found': len(dormant_accounts)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "unclaimed_payment_instruments_analysis"
            })
            logger.error(f"Unclaimed payment instruments analysis failed: {e}")
            return state

class EligibleForCBUAETransferAgent(BaseDormancyAgent):
    """Eligible for CBUAE Transfer Analysis - CBUAE Article 8"""

    def __init__(self):
        super().__init__("eligible_for_cbuae_transfer")
        self.cbuae_article = "CBUAE Art. 8"
        self.ui_status = DormancyStatus.READY

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze accounts ready for Central Bank transfer"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Look for dormant accounts eligible for CBUAE transfer
            dormant_accounts = df[
                df[self.csv_columns['dormancy_status']].str.contains(
                    'DORMANT|Dormant', case=False, na=False
                )
            ].copy()

            transfer_eligible = []
            report_date = datetime.now()

            for idx, account in dormant_accounts.iterrows():
                last_transaction = account[self.csv_columns['last_transaction_date']]
                balance = account[self.csv_columns['balance_current']]
                contact_attempts = account.get(self.csv_columns['contact_attempts_made'], 0)
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 8: 5+ years with completed contact attempts
                if dormancy_days >= 1825 and contact_attempts >= 3:
                    transfer_eligible.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'contact_attempts': int(contact_attempts),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Prepare transfer documentation',
                        'priority': 'High',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(dormant_accounts)
            state.dormant_records_found = len(transfer_eligible)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if transfer_eligible:
                state.processed_dataframe = pd.DataFrame(transfer_eligible)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"cbuae_transfer_eligible_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'CBUAE transfer eligibility analysis per CBUAE Article 8',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'transfer_eligible': transfer_eligible,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(dormant_accounts),
                    'transfer_eligible': len(transfer_eligible)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "cbuae_transfer_analysis"
            })
            logger.error(f"CBUAE transfer analysis failed: {e}")
            return state

class Article3ProcessNeededAgent(BaseDormancyAgent):
    """Article 3 Process Needed Analysis - CBUAE Article 3"""

    def __init__(self):
        super().__init__("article_3_process_needed")
        self.cbuae_article = "CBUAE Art. 3"
        self.ui_status = DormancyStatus.IN_PROGRESS

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze accounts requiring Article 3 dormancy processes"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Look for accounts requiring Article 3 processes
            article3_candidates = []
            report_date = datetime.now()

            for idx, account in df.iterrows():
                dormancy_status = account.get(self.csv_columns['dormancy_status'], '')
                last_transaction = account[self.csv_columns['last_transaction_date']]
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # Article 3 processes for dormant accounts
                if 'dormant' in str(dormancy_status).lower() and dormancy_days >= 365:
                    article3_candidates.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'dormancy_days': dormancy_days,
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Apply Article 3 dormancy processes',
                        'priority': 'High',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(df)
            state.dormant_records_found = len(article3_candidates)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if article3_candidates:
                state.processed_dataframe = pd.DataFrame(article3_candidates)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"article3_process_needed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'Article 3 process requirements analysis',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'article3_candidates': article3_candidates,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(df),
                    'process_needed': len(article3_candidates)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "article3_process_analysis"
            })
            logger.error(f"Article 3 process analysis failed: {e}")
            return state

class ContactAttemptsNeededAgent(BaseDormancyAgent):
    """Contact Attempts Needed Analysis - CBUAE Article 5"""

    def __init__(self):
        super().__init__("contact_attempts_needed")
        self.cbuae_article = "CBUAE Art. 5"
        self.ui_status = DormancyStatus.URGENT

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze accounts requiring customer contact attempts"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Look for dormant accounts with insufficient contact attempts
            incomplete_contacts = []

            # Filter for dormant accounts
            dormant_accounts = df[
                df[self.csv_columns['dormancy_status']].str.contains(
                    'DORMANT|Dormant', case=False, na=False
                )
            ].copy()

            for idx, account in dormant_accounts.iterrows():
                contact_attempts = account.get(self.csv_columns['contact_attempts_made'], 0)

                # CBUAE Article 5: Minimum 3 contact attempts required
                if contact_attempts < 3:
                    incomplete_contacts.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'contact_attempts_made': int(contact_attempts),
                        'required_attempts': 3,
                        'remaining_attempts': max(0, 3 - int(contact_attempts)),
                        'compliance_article': self.cbuae_article,
                        'action_required': f'Complete {3 - int(contact_attempts)} additional contact attempts',
                        'priority': 'Urgent',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(dormant_accounts)
            state.dormant_records_found = len(incomplete_contacts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if incomplete_contacts:
                state.processed_dataframe = pd.DataFrame(incomplete_contacts)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"contact_attempts_needed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'Contact attempts compliance analysis per CBUAE Article 5',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'incomplete_contacts': incomplete_contacts,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_dormant_reviewed': len(dormant_accounts),
                    'insufficient_contacts': len(incomplete_contacts)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "contact_attempts_analysis"
            })
            logger.error(f"Contact attempts analysis failed: {e}")
            return state

class HighValueDormantAgent(BaseDormancyAgent):
    """High Value Dormant (≥25K AED) Analysis"""

    def __init__(self):
        super().__init__("high_value_dormant")
        self.cbuae_article = "Risk Management"
        self.ui_status = DormancyStatus.PRIORITY

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze high value dormant accounts requiring escalated monitoring"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Filter for high value dormant accounts (≥25K AED)
            high_value_dormant = df[
                (df[self.csv_columns['dormancy_status']].str.contains(
                    'DORMANT|Dormant', case=False, na=False
                )) &
                (df[self.csv_columns['balance_current']] >= 25000)
            ].copy()

            dormant_accounts = []

            for idx, account in high_value_dormant.iterrows():
                balance = account[self.csv_columns['balance_current']]
                balance_value = float(balance) if pd.notna(balance) else 0.0

                if balance_value >= 25000:
                    dormant_accounts.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'balance_current': balance_value,
                        'currency': account.get(self.csv_columns['currency'], 'AED'),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Escalated monitoring and priority contact',
                        'priority': 'Critical' if balance_value >= 100000 else 'High',
                        'risk_category': 'High Value Dormant',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(high_value_dormant)
            state.dormant_records_found = len(dormant_accounts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if dormant_accounts:
                state.processed_dataframe = pd.DataFrame(dormant_accounts)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"high_value_dormant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'High value dormant accounts analysis (≥25K AED)',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'dormant_accounts': dormant_accounts,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(high_value_dormant),
                    'high_value_found': len(dormant_accounts),
                    'total_value': sum([acc['balance_current'] for acc in dormant_accounts])
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "high_value_dormant_analysis"
            })
            logger.error(f"High value dormant analysis failed: {e}")
            return state

class DormantToActiveTransitionsAgent(BaseDormancyAgent):
    """Dormant to Active Transitions Analysis"""

    def __init__(self):
        super().__init__("dormant_to_active_transitions")
        self.cbuae_article = "Monitoring"
        self.ui_status = DormancyStatus.MONITORED

    def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze accounts transitioning from dormant to active status"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            df = state.input_dataframe
            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Look for recent activity on accounts with dormancy history
            transitions = []
            report_date = datetime.now()

            for idx, account in df.iterrows():
                dormancy_status = account.get(self.csv_columns['dormancy_status'], '')
                last_transaction = account[self.csv_columns['last_transaction_date']]

                # Check for recent activity (within last 30 days)
                recent_activity_days = calculate_dormancy_days(last_transaction, report_date)

                # Look for transitions from dormant to active
                if (recent_activity_days <= 30 and
                    'dormant' in str(dormancy_status).lower()):

                    transitions.append({
                        'customer_id': account[self.csv_columns['customer_id']],
                        'account_id': account[self.csv_columns['account_id']],
                        'account_type': account[self.csv_columns['account_type']],
                        'last_transaction_date': str(last_transaction),
                        'days_since_activity': recent_activity_days,
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Update dormancy status tracking',
                        'priority': 'Medium',
                        'transition_type': 'Dormant to Active',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Update state
            state.records_processed = len(df)
            state.dormant_records_found = len(transitions)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            if transitions:
                state.processed_dataframe = pd.DataFrame(transitions)

            # Create CSV export
            csv_export = create_csv_download_data(
                state.processed_dataframe or pd.DataFrame(),
                f"dormant_to_active_transitions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            state.analysis_results = {
                'description': 'Dormant to active transitions monitoring',
                'compliance_article': self.cbuae_article,
                'status': self.ui_status.value,
                'transitions': transitions,
                'csv_export': csv_export,
                'summary_stats': {
                    'total_processed': len(df),
                    'transitions_found': len(transitions)
                }
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "dormant_to_active_transitions_analysis"
            })
            logger.error(f"Dormant to active transitions analysis failed: {e}")
            return state

# ===== ORCHESTRATOR CLASS =====

class DormancyWorkflowOrchestrator:
    """Orchestrator for all dormancy analysis agents"""

    def __init__(self):
        self.agents = {
            'safe_deposit_dormancy': SafeDepositDormancyAgent(),
            'investment_account_inactivity': InvestmentAccountInactivityAgent(),
            'fixed_deposit_inactivity': FixedDepositInactivityAgent(),
            'demand_deposit_inactivity': DemandDepositInactivityAgent(),
            'unclaimed_payment_instruments': UnclaimedPaymentInstrumentsAgent(),
            'eligible_for_cbuae_transfer': EligibleForCBUAETransferAgent(),
            'article_3_process_needed': Article3ProcessNeededAgent(),
            'contact_attempts_needed': ContactAttemptsNeededAgent(),
            'high_value_dormant': HighValueDormantAgent(),
            'dormant_to_active_transitions': DormantToActiveTransitionsAgent()
        }

    def run_comprehensive_analysis(self, state) -> Dict:
        """Run comprehensive analysis with all agents"""
        try:
            start_time = datetime.now()

            results = {
                "success": True,
                "session_id": state.session_id if hasattr(state, 'session_id') else secrets.token_hex(8),
                "agent_results": {},
                "csv_exports": {},
                "summary": {},
                "processing_time": 0.0
            }

            # Execute all agents
            for agent_name, agent in self.agents.items():
                try:
                    # Create agent state
                    agent_state = agent.create_agent_state(
                        user_id=state.user_id if hasattr(state, 'user_id') else "default_user",
                        dataframe=state.raw_data if hasattr(state, 'raw_data') else state
                    )

                    # Run agent analysis
                    result_state = agent.analyze_dormancy(agent_state)

                    # Store results
                    results["agent_results"][agent_name] = {
                        'agent_type': result_state.agent_type,
                        'records_processed': result_state.records_processed,
                        'dormant_records_found': result_state.dormant_records_found,
                        'processing_time': result_state.processing_time,
                        'success': result_state.agent_status == AgentStatus.COMPLETED,
                        'analysis_results': result_state.analysis_results,
                        'processed_dataframe': result_state.processed_dataframe,
                        'ui_status': agent.ui_status.value,
                        'cbuae_article': agent.cbuae_article
                    }

                    # Store CSV export data
                    if result_state.analysis_results and 'csv_export' in result_state.analysis_results:
                        results["csv_exports"][agent_name] = result_state.analysis_results['csv_export']

                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    results["agent_results"][agent_name] = {
                        'success': False,
                        'error': str(e),
                        'records_processed': 0,
                        'dormant_records_found': 0,
                        'ui_status': 'failed',
                        'cbuae_article': 'Unknown'
                    }

            # Create summary
            total_processed = sum([r.get('records_processed', 0) for r in results["agent_results"].values()])
            total_dormant = sum([r.get('dormant_records_found', 0) for r in results["agent_results"].values()])

            results["summary"] = {
                "total_accounts_analyzed": total_processed,
                "total_dormant_accounts_found": total_dormant,
                "agents_executed": len([r for r in results["agent_results"].values() if r.get('success', False)]),
                "csv_exports_available": len(results["csv_exports"]),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

            results["processing_time"] = results["summary"]["processing_time"]

            return results

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_results": {},
                "csv_exports": {}
            }

    def get_agent_by_name(self, agent_name: str):
        """Get agent by name for individual execution"""
        return self.agents.get(agent_name)

    def get_all_agent_info(self) -> Dict:
        """Get information about all available agents"""
        agent_info = {}
        for name, agent in self.agents.items():
            agent_info[name] = {
                'agent_type': agent.agent_type,
                'cbuae_article': agent.cbuae_article,
                'ui_status': agent.ui_status.value,
                'description': f"{agent.agent_type.replace('_', ' ').title()} Analysis"
            }
        return agent_info

# ===== MAIN EXECUTION FUNCTIONS =====

def run_comprehensive_dormancy_analysis_with_csv(user_id: str, account_data: pd.DataFrame,
                                                      report_date: str = None) -> Dict:
    """Run comprehensive dormancy analysis with CSV export capability"""
    try:
        # Initialize orchestrator
        orchestrator = DormancyWorkflowOrchestrator()

        # Create mock state object
        class MockState:
            def __init__(self):
                self.session_id = secrets.token_hex(16)
                self.user_id = user_id
                self.raw_data = account_data

        state = MockState()

        # Execute comprehensive analysis
        result = orchestrator.run_comprehensive_analysis(state)

        return result

    except Exception as e:
        logger.error(f"Comprehensive dormancy analysis with CSV failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "session_id": None,
            "agent_results": None,
            "csv_exports": {}
        }

def run_individual_agent_analysis(agent_name: str, user_id: str, account_data: pd.DataFrame) -> Dict:
    """Run individual agent analysis"""
    try:
        orchestrator = DormancyWorkflowOrchestrator()
        agent = orchestrator.get_agent_by_name(agent_name)

        if not agent:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found",
                "available_agents": list(orchestrator.agents.keys())
            }

        # Create agent state
        agent_state = agent.create_agent_state(user_id, account_data)

        # Run analysis
        result_state = agent.analyze_dormancy(agent_state)

        return {
            "success": result_state.agent_status == AgentStatus.COMPLETED,
            "agent_name": agent_name,
            "agent_type": result_state.agent_type,
            "records_processed": result_state.records_processed,
            "dormant_records_found": result_state.dormant_records_found,
            "processing_time": result_state.processing_time,
            "analysis_results": result_state.analysis_results,
            "csv_export": result_state.analysis_results.get('csv_export') if result_state.analysis_results else None
        }

    except Exception as e:
        logger.error(f"Individual agent analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ===== CSV UTILITY FUNCTIONS =====

def get_agent_csv_data(agent_results: Dict, agent_name: str) -> Optional[Dict]:
    """Get CSV export data for a specific agent"""
    if agent_name in agent_results and 'analysis_results' in agent_results[agent_name]:
        return agent_results[agent_name]['analysis_results'].get('csv_export')
    return None

def get_all_csv_download_info(analysis_results: Dict) -> Dict:
    """Get download information for all agent CSV files"""
    download_info = {}

    if 'csv_exports' in analysis_results:
        for agent_name, csv_data in analysis_results['csv_exports'].items():
            if csv_data.get('available', False):
                download_info[agent_name] = {
                    'filename': csv_data['filename'],
                    'records': csv_data['records'],
                    'file_size_kb': csv_data['file_size_kb'],
                    'download_ready': True
                }
            else:
                download_info[agent_name] = {
                    'download_ready': False,
                    'reason': 'No data found'
                }

    return download_info

# ===== MODULE EXPORTS =====

__all__ = [
    # Base Classes
    'BaseDormancyAgent',
    'DormancyWorkflowOrchestrator',

    # UI-Matched Agents (10 total)
    'SafeDepositDormancyAgent',
    'InvestmentAccountInactivityAgent',
    'FixedDepositInactivityAgent',
    'DemandDepositInactivityAgent',
    'UnclaimedPaymentInstrumentsAgent',
    'EligibleForCBUAETransferAgent',
    'Article3ProcessNeededAgent',
    'ContactAttemptsNeededAgent',
    'HighValueDormantAgent',
    'DormantToActiveTransitionsAgent',

    # Data Classes
    'AgentResult',
    'AgentState',
    'AgentStatus',
    'DormancyStatus',

    # Main Functions
    'run_comprehensive_dormancy_analysis_with_csv',
    'run_individual_agent_analysis',

    # Utility Functions
    'safe_date_parse',
    'calculate_dormancy_days',
    'create_csv_download_data',
    'get_agent_csv_data',
    'get_all_csv_download_info'
]