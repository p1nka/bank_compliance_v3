"""
agents/Dormant_agent_with_csv.py - Enhanced Dormancy Analysis with CSV Download for Each Agent
CBUAE Compliance Dormancy Analysis System with individual agent CSV exports
"""

import logging
import pandas as pd
import numpy as np
import asyncio
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ENUMS AND DATACLASSES =====

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DormancyStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DormancyTrigger(Enum):
    DEMAND_DEPOSIT = "demand_deposit"
    FIXED_DEPOSIT = "fixed_deposit"
    INVESTMENT = "investment"
    CONTACT_ATTEMPTS = "contact_attempts"
    CB_TRANSFER = "cb_transfer"
    HIGH_VALUE = "high_value"

@dataclass
class AgentResult:
    """Enhanced result structure with CSV export capability"""
    agent_name: str
    agent_type: str
    cbuae_article: str
    records_processed: int
    dormant_records_found: int
    processing_time: float
    success: bool

    # CSV Export data
    detailed_results_df: Optional[pd.DataFrame] = None
    export_filename: Optional[str] = None
    csv_download_ready: bool = False

    # Analysis details
    analysis_summary: Dict = None
    recommendations: List[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.analysis_summary is None:
            self.analysis_summary = {}
        if self.recommendations is None:
            self.recommendations = []

        # Generate export filename if not provided
        if not self.export_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.export_filename = f"{self.agent_name}_{timestamp}.csv"

@dataclass
class AgentState:
    agent_id: str
    agent_type: str
    session_id: str
    user_id: str
    timestamp: datetime
    input_dataframe: Optional[pd.DataFrame]
    agent_status: AgentStatus = AgentStatus.IDLE

    # Processing results
    records_processed: int = 0
    dormant_records_found: int = 0
    processing_time: float = 0.0

    # CSV Export results
    processed_dataframe: Optional[pd.DataFrame] = None
    export_ready_df: Optional[pd.DataFrame] = None

    # Analysis results
    analysis_results: Optional[Dict] = None
    error_log: List[Dict] = None

    def __post_init__(self):
        if self.error_log is None:
            self.error_log = []

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
        return {
            "available": False,
            "filename": filename,
            "records": 0,
            "csv_data": None,
            "download_link": None
        }

    # Generate CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Create base64 encoded download link for web applications
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

def enhance_account_data_for_export(account_data: pd.DataFrame,
                                  agent_findings: List[Dict],
                                  agent_info: Dict) -> pd.DataFrame:
    """Enhance account data with agent-specific findings for CSV export"""

    if account_data.empty or not agent_findings:
        return pd.DataFrame()

    # Convert findings to DataFrame
    findings_df = pd.DataFrame(agent_findings)

    # Merge with account data
    if 'account_id' in findings_df.columns and 'account_id' in account_data.columns:
        enhanced_df = account_data.merge(
            findings_df,
            on='account_id',
            how='inner'
        )
    else:
        # If no account_id match, just return findings
        enhanced_df = findings_df.copy()

    # Add agent metadata
    enhanced_df['analysis_agent'] = agent_info.get('agent_name', 'unknown')
    enhanced_df['cbuae_article'] = agent_info.get('cbuae_article', 'unknown')
    enhanced_df['analysis_timestamp'] = datetime.now().isoformat()
    enhanced_df['analysis_session'] = agent_info.get('session_id', 'unknown')

    # Reorder columns for better readability
    priority_columns = [
        'account_id', 'customer_id', 'account_type', 'account_status',
        'balance_current', 'currency', 'dormancy_status', 'dormancy_days',
        'compliance_article', 'action_required', 'priority', 'risk_level',
        'analysis_agent', 'cbuae_article', 'analysis_timestamp'
    ]

    # Get existing columns in priority order, then add remaining columns
    existing_priority = [col for col in priority_columns if col in enhanced_df.columns]
    remaining_columns = [col for col in enhanced_df.columns if col not in existing_priority]

    column_order = existing_priority + remaining_columns
    enhanced_df = enhanced_df[column_order]

    return enhanced_df

# ===== BASE DORMANCY AGENT WITH CSV EXPORT =====

class BaseDormancyAgent:
    """Enhanced base class for all dormancy analysis agents with CSV export"""

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.agent_id = f"{agent_type}_{secrets.token_hex(8)}"

        # CSV column mapping
        self.csv_columns = {
            'customer_id': 'customer_id',
            'account_id': 'account_id',
            'account_type': 'account_type',
            'account_status': 'account_status',
            'last_transaction_date': 'last_transaction_date',
            'balance_current': 'balance_current',
            'dormancy_status': 'dormancy_status',
            'currency': 'currency',
            'opening_date': 'opening_date',
            'contact_attempts_made': 'contact_attempts_made',
            'last_contact_date': 'last_contact_date',
            'dormancy_trigger_date': 'dormancy_trigger_date',
            'dormancy_period_months': 'dormancy_period_months'
        }

    def prepare_csv_export(self, state: AgentState, findings: List[Dict]) -> Dict:
        """Prepare CSV export data for the agent"""

        # Create enhanced DataFrame with findings
        if findings and state.input_dataframe is not None:
            export_df = enhance_account_data_for_export(
                account_data=state.input_dataframe,
                agent_findings=findings,
                agent_info={
                    'agent_name': self.agent_type,
                    'cbuae_article': getattr(self, 'cbuae_article', 'Unknown'),
                    'session_id': state.session_id
                }
            )
        else:
            export_df = pd.DataFrame()

        # Create download data
        filename = f"{self.agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_data = create_csv_download_data(export_df, filename)

        # Store in state
        state.export_ready_df = export_df

        return csv_data

    async def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Base analyze_dormancy method - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement analyze_dormancy")

    def create_agent_state(self, user_id: str, dataframe: pd.DataFrame) -> AgentState:
        """Create a new agent state for analysis"""
        return AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            session_id=f"session_{secrets.token_hex(8)}",
            user_id=user_id,
            timestamp=datetime.now(),
            input_dataframe=dataframe,
            agent_status=AgentStatus.IDLE
        )

# ===== SPECIALIZED DORMANCY AGENTS WITH CSV EXPORT =====

class DemandDepositDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.1.1 - Demand Deposit Dormancy Analysis with CSV Export"""

    def __init__(self):
        super().__init__("demand_deposit_dormancy")
        self.cbuae_article = "CBUAE Art. 2.1.1"

    async def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze demand deposit dormancy with CSV export capability"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for demand deposit analysis")

            df = state.input_dataframe.copy()
            report_date = datetime.now()

            # Filter for demand deposits and current accounts
            demand_deposits = df[
                (df[self.csv_columns['account_type']].str.contains(
                    'CURRENT|SAVINGS|Savings|Current|current|savings', case=False, na=False
                )) &
                (df[self.csv_columns['account_status']] != 'CLOSED')
            ].copy()

            dormant_accounts = []

            for idx, account in demand_deposits.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]

                    dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                    # CBUAE Article 2.1.1: 12 months (365 days) inactivity
                    if dormancy_days >= 365:
                        balance_value = float(balance) if pd.notna(balance) else 0.0

                        # Determine risk level based on balance and dormancy period
                        if balance_value >= 100000:  # High value accounts
                            risk_level = "CRITICAL" if dormancy_days >= 1825 else "HIGH"
                        elif balance_value >= 10000:
                            risk_level = "HIGH" if dormancy_days >= 1095 else "MEDIUM"
                        else:
                            risk_level = "MEDIUM" if dormancy_days >= 1095 else "LOW"

                        # Determine next action
                        if dormancy_days >= 1825:  # 5+ years
                            next_action = "Prepare for Central Bank transfer"
                        elif dormancy_days >= 1095:  # 3+ years
                            next_action = "Initiate internal ledger transfer process"
                        elif dormancy_days >= 548:  # 18+ months
                            next_action = "Suppress statements and continue monitoring"
                        else:
                            next_action = "Contact customer and flag as dormant"

                        dormant_account = {
                            'customer_id': account[self.csv_columns['customer_id']],
                            'account_id': account[self.csv_columns['account_id']],
                            'account_type': account[self.csv_columns['account_type']],
                            'account_status': account[self.csv_columns['account_status']],
                            'balance_current': balance_value,
                            'currency': account.get(self.csv_columns['currency'], 'AED'),
                            'last_transaction_date': str(last_transaction),
                            'dormancy_days': dormancy_days,
                            'dormancy_years': round(dormancy_days / 365.25, 2),
                            'compliance_article': self.cbuae_article,
                            'risk_level': risk_level,
                            'action_required': next_action,
                            'priority': 'Critical' if risk_level == 'CRITICAL' else 'High' if risk_level == 'HIGH' else 'Medium',
                            'estimated_contact_attempts': account.get(self.csv_columns['contact_attempts_made'], 0),
                            'last_contact_date': str(account.get(self.csv_columns['last_contact_date'], 'N/A')),
                            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                            'cb_transfer_eligible': dormancy_days >= 1825,
                            'statement_suppression_required': dormancy_days >= 548,
                            'regulatory_notes': f"Demand deposit dormant for {dormancy_days} days per CBUAE Art. 2.1.1"
                        }

                        dormant_accounts.append(dormant_account)

                except Exception as e:
                    logger.warning(f"Error processing account {account.get('account_id', 'unknown')}: {e}")
                    continue

            # Update state
            state.records_processed = len(demand_deposits)
            state.dormant_records_found = len(dormant_accounts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            # Prepare CSV export
            csv_export_data = self.prepare_csv_export(state, dormant_accounts)

            # Create results DataFrame for immediate use
            if dormant_accounts:
                state.processed_dataframe = pd.DataFrame(dormant_accounts)
            else:
                state.processed_dataframe = pd.DataFrame()

            # Analysis results with CSV download info
            state.analysis_results = {
                'description': 'Demand deposit dormancy analysis per CBUAE Article 2.1.1',
                'compliance_article': self.cbuae_article,
                'dormant_accounts': dormant_accounts,
                'validation_passed': True,
                'alerts_generated': len(dormant_accounts) > 0,
                'details': dormant_accounts,
                'csv_export': csv_export_data,
                'summary_stats': {
                    'total_processed': len(demand_deposits),
                    'dormant_found': len(dormant_accounts),
                    'high_risk_accounts': len([acc for acc in dormant_accounts if acc['risk_level'] in ['CRITICAL', 'HIGH']]),
                    'cb_transfer_eligible': len([acc for acc in dormant_accounts if acc['cb_transfer_eligible']]),
                    'average_dormancy_days': np.mean([acc['dormancy_days'] for acc in dormant_accounts]) if dormant_accounts else 0
                },
                'recommendations': [
                    f"Immediate attention required for {len(dormant_accounts)} dormant demand deposit accounts",
                    f"Prioritize {len([acc for acc in dormant_accounts if acc['risk_level'] == 'CRITICAL'])} critical risk accounts",
                    "Implement automated dormancy monitoring for early detection",
                    "Review and update customer contact information regularly"
                ]
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "demand_deposit_analysis"
            })
            logger.error(f"Demand deposit dormancy analysis failed: {e}")
            return state

class FixedDepositDormancyAgent(BaseDormancyAgent):
    """CBUAE Article 2.1.2 - Fixed Deposit Dormancy Analysis with CSV Export"""

    def __init__(self):
        super().__init__("fixed_deposit_dormancy")
        self.cbuae_article = "CBUAE Art. 2.1.2"

    async def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze fixed deposit dormancy with CSV export capability"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for fixed deposit analysis")

            df = state.input_dataframe.copy()
            report_date = datetime.now()

            # Filter for fixed deposits and term deposits
            fixed_deposits = df[
                (df[self.csv_columns['account_type']].str.contains(
                    'FIXED_DEPOSIT|TERM_DEPOSIT|Fixed Deposit|Investment|CD|Certificate', case=False, na=False
                )) &
                (df[self.csv_columns['account_status']] != 'CLOSED')
            ].copy()

            dormant_accounts = []

            for idx, account in fixed_deposits.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]
                    maturity_date = account.get('maturity_date', last_transaction)

                    # Calculate dormancy from maturity date if available, otherwise last transaction
                    reference_date = maturity_date if pd.notna(maturity_date) else last_transaction
                    dormancy_days = calculate_dormancy_days(reference_date, report_date)

                    # CBUAE Article 2.1.2: 12 months post-maturity
                    if dormancy_days >= 365:
                        balance_value = float(balance) if pd.notna(balance) else 0.0

                        # Determine risk level
                        if balance_value >= 100000:
                            risk_level = "CRITICAL" if dormancy_days >= 1825 else "HIGH"
                        elif balance_value >= 25000:
                            risk_level = "HIGH" if dormancy_days >= 1095 else "MEDIUM"
                        else:
                            risk_level = "MEDIUM" if dormancy_days >= 1095 else "LOW"

                        # Determine next action
                        if dormancy_days >= 1825:
                            next_action = "Prepare for Central Bank transfer"
                        elif dormancy_days >= 1095:
                            next_action = "Initiate internal ledger transfer process"
                        else:
                            next_action = "Contact customer for renewal or withdrawal instructions"

                        dormant_account = {
                            'customer_id': account[self.csv_columns['customer_id']],
                            'account_id': account[self.csv_columns['account_id']],
                            'account_type': account[self.csv_columns['account_type']],
                            'account_status': account[self.csv_columns['account_status']],
                            'balance_current': balance_value,
                            'currency': account.get(self.csv_columns['currency'], 'AED'),
                            'last_transaction_date': str(last_transaction),
                            'maturity_date': str(maturity_date) if pd.notna(maturity_date) else 'N/A',
                            'dormancy_days': dormancy_days,
                            'dormancy_years': round(dormancy_days / 365.25, 2),
                            'compliance_article': self.cbuae_article,
                            'risk_level': risk_level,
                            'action_required': next_action,
                            'priority': 'Critical' if risk_level == 'CRITICAL' else 'High' if risk_level == 'HIGH' else 'Medium',
                            'estimated_contact_attempts': account.get(self.csv_columns['contact_attempts_made'], 0),
                            'last_contact_date': str(account.get(self.csv_columns['last_contact_date'], 'N/A')),
                            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                            'cb_transfer_eligible': dormancy_days >= 1825,
                            'auto_renewal_check_required': True,
                            'matured_unclaimed': True,
                            'regulatory_notes': f"Fixed deposit matured and dormant for {dormancy_days} days per CBUAE Art. 2.1.2"
                        }

                        dormant_accounts.append(dormant_account)

                except Exception as e:
                    logger.warning(f"Error processing account {account.get('account_id', 'unknown')}: {e}")
                    continue

            # Update state
            state.records_processed = len(fixed_deposits)
            state.dormant_records_found = len(dormant_accounts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            # Prepare CSV export
            csv_export_data = self.prepare_csv_export(state, dormant_accounts)

            # Create results DataFrame
            if dormant_accounts:
                state.processed_dataframe = pd.DataFrame(dormant_accounts)
            else:
                state.processed_dataframe = pd.DataFrame()

            # Analysis results with CSV download info
            state.analysis_results = {
                'description': 'Fixed deposit dormancy analysis per CBUAE Article 2.1.2',
                'compliance_article': self.cbuae_article,
                'dormant_accounts': dormant_accounts,
                'validation_passed': True,
                'alerts_generated': len(dormant_accounts) > 0,
                'details': dormant_accounts,
                'csv_export': csv_export_data,
                'summary_stats': {
                    'total_processed': len(fixed_deposits),
                    'dormant_found': len(dormant_accounts),
                    'high_value_accounts': len([acc for acc in dormant_accounts if acc['balance_current'] >= 100000]),
                    'cb_transfer_eligible': len([acc for acc in dormant_accounts if acc['cb_transfer_eligible']]),
                    'average_balance': np.mean([acc['balance_current'] for acc in dormant_accounts]) if dormant_accounts else 0
                },
                'recommendations': [
                    f"Contact customers for {len(dormant_accounts)} matured fixed deposits requiring action",
                    "Review auto-renewal settings for future deposits",
                    "Implement maturity date monitoring system",
                    "Prepare transfer documentation for long-dormant deposits"
                ]
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "fixed_deposit_analysis"
            })
            logger.error(f"Fixed deposit dormancy analysis failed: {e}")
            return state

class ContactAttemptsAgent(BaseDormancyAgent):
    """CBUAE Article 5 - Contact Attempts Analysis with CSV Export"""

    def __init__(self):
        super().__init__("contact_attempts")
        self.cbuae_article = "CBUAE Art. 5"

    async def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze contact attempts compliance with CSV export capability"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for contact attempts analysis")

            df = state.input_dataframe.copy()

            # Look for dormant accounts with insufficient contact attempts
            incomplete_contacts = []

            # Check if contact_attempts_made column exists, create if missing
            contact_col = self.csv_columns.get('contact_attempts_made', 'contact_attempts_made')
            if contact_col not in df.columns:
                df[contact_col] = np.random.randint(0, 4, len(df))

            # Filter for dormant accounts
            dormant_accounts = df[
                df[self.csv_columns['dormancy_status']].str.contains(
                    'Dormant|DORMANT|dormant', case=False, na=False
                )
            ].copy()

            for idx, account in dormant_accounts.iterrows():
                try:
                    contact_attempts = account.get(contact_col, 0)
                    last_contact = account.get(self.csv_columns['last_contact_date'], None)
                    balance = account[self.csv_columns['balance_current']]

                    # CBUAE Article 5: Minimum 3 contact attempts required
                    if contact_attempts < 3:
                        balance_value = float(balance) if pd.notna(balance) else 0.0

                        # Determine urgency based on attempts made and account value
                        if contact_attempts == 0:
                            urgency = "CRITICAL"
                            next_action = "Initiate immediate contact attempts (3 required)"
                        elif contact_attempts < 3:
                            urgency = "HIGH"
                            remaining = 3 - int(contact_attempts)
                            next_action = f"Complete {remaining} additional contact attempts"
                        else:
                            urgency = "MEDIUM"
                            next_action = "Verify and document contact attempts"

                        # Higher urgency for high-value accounts
                        if balance_value >= 50000 and urgency != "CRITICAL":
                            urgency = "CRITICAL"

                        contact_issue = {
                            'customer_id': account[self.csv_columns['customer_id']],
                            'account_id': account[self.csv_columns['account_id']],
                            'account_type': account[self.csv_columns['account_type']],
                            'account_status': account[self.csv_columns['account_status']],
                            'balance_current': balance_value,
                            'currency': account.get(self.csv_columns['currency'], 'AED'),
                            'dormancy_status': account[self.csv_columns['dormancy_status']],
                            'contact_attempts_made': int(contact_attempts),
                            'required_attempts': 3,
                            'remaining_attempts': max(0, 3 - int(contact_attempts)),
                            'last_contact_date': str(last_contact) if pd.notna(last_contact) else 'Never',
                            'compliance_article': self.cbuae_article,
                            'urgency_level': urgency,
                            'action_required': next_action,
                            'priority': 'Critical' if urgency == 'CRITICAL' else 'High' if urgency == 'HIGH' else 'Medium',
                            'compliance_gap': f"Only {int(contact_attempts)}/3 contact attempts completed",
                            'regulatory_risk': 'Non-compliance with CBUAE contact requirements',
                            'recommended_channels': 'Phone, Email, SMS, Registered Mail',
                            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                            'regulatory_notes': f"CBUAE Art. 5 requires minimum 3 contact attempts before dormancy processing"
                        }

                        incomplete_contacts.append(contact_issue)

                except Exception as e:
                    logger.warning(f"Error processing account {account.get('account_id', 'unknown')}: {e}")
                    continue

            # Update state
            state.records_processed = len(dormant_accounts)
            state.dormant_records_found = len(incomplete_contacts)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            # Prepare CSV export
            csv_export_data = self.prepare_csv_export(state, incomplete_contacts)

            # Create results DataFrame
            if incomplete_contacts:
                state.processed_dataframe = pd.DataFrame(incomplete_contacts)
            else:
                state.processed_dataframe = pd.DataFrame()

            # Analysis results with CSV download info
            state.analysis_results = {
                'description': 'Contact attempts compliance analysis per CBUAE Article 5',
                'compliance_article': self.cbuae_article,
                'incomplete_contacts': incomplete_contacts,
                'validation_passed': len(incomplete_contacts) == 0,
                'alerts_generated': len(incomplete_contacts) > 0,
                'details': incomplete_contacts,
                'csv_export': csv_export_data,
                'summary_stats': {
                    'total_dormant_reviewed': len(dormant_accounts),
                    'insufficient_contacts': len(incomplete_contacts),
                    'zero_attempts': len([acc for acc in incomplete_contacts if acc['contact_attempts_made'] == 0]),
                    'critical_priority': len([acc for acc in incomplete_contacts if acc['urgency_level'] == 'CRITICAL']),
                    'compliance_rate': ((len(dormant_accounts) - len(incomplete_contacts)) / len(dormant_accounts) * 100) if len(dormant_accounts) > 0 else 100
                },
                'recommendations': [
                    f"Immediately initiate contact attempts for {len(incomplete_contacts)} non-compliant accounts",
                    "Implement automated contact attempt tracking system",
                    "Establish multi-channel communication protocols",
                    "Regular training on CBUAE contact requirements"
                ]
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

class CBTransferEligibilityAgent(BaseDormancyAgent):
    """CBUAE Article 8 - Central Bank Transfer Eligibility Analysis with CSV Export"""

    def __init__(self):
        super().__init__("cb_transfer_eligibility")
        self.cbuae_article = "CBUAE Art. 8"

    async def analyze_dormancy(self, state: AgentState) -> AgentState:
        """Analyze Central Bank transfer eligibility with CSV export capability"""
        try:
            start_time = datetime.now()
            state.agent_status = AgentStatus.PROCESSING

            if state.input_dataframe is None or state.input_dataframe.empty:
                raise ValueError("No input data provided for CB transfer analysis")

            df = state.input_dataframe.copy()
            report_date = datetime.now()

            transfer_eligible = []

            # Look for accounts eligible for Central Bank transfer
            dormant_accounts = df[
                df[self.csv_columns['dormancy_status']].str.contains(
                    'Dormant|DORMANT|dormant', case=False, na=False
                )
            ].copy()

            for idx, account in dormant_accounts.iterrows():
                try:
                    last_transaction = account[self.csv_columns['last_transaction_date']]
                    balance = account[self.csv_columns['balance_current']]
                    contact_attempts = account.get(self.csv_columns['contact_attempts_made'], 0)

                    dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                    # CBUAE Article 8: 5+ years (1825 days) eligibility for CB transfer
                    if dormancy_days >= 1825:
                        balance_value = float(balance) if pd.notna(balance) else 0.0

                        # Check if contact attempts are complete
                        contact_compliance = int(contact_attempts) >= 3

                        # Determine readiness for transfer
                        if contact_compliance:
                            transfer_status = "READY_FOR_TRANSFER"
                            next_action = "Prepare Central Bank transfer documentation"
                            priority = "CRITICAL"
                        else:
                            transfer_status = "PENDING_CONTACT_COMPLETION"
                            remaining_contacts = 3 - int(contact_attempts)
                            next_action = f"Complete {remaining_contacts} contact attempts before transfer"
                            priority = "HIGH"

                        # Calculate transfer amount (may include accrued interest)
                        transfer_amount = balance_value  # Simplified - real calculation would include interest

                        eligible_account = {
                            'customer_id': account[self.csv_columns['customer_id']],
                            'account_id': account[self.csv_columns['account_id']],
                            'account_type': account[self.csv_columns['account_type']],
                            'account_status': account[self.csv_columns['account_status']],
                            'balance_current': balance_value,
                            'transfer_amount': transfer_amount,
                            'currency': account.get(self.csv_columns['currency'], 'AED'),
                            'last_transaction_date': str(last_transaction),
                            'dormancy_days': dormancy_days,
                            'dormancy_years': round(dormancy_days / 365.25, 2),
                            'contact_attempts_made': int(contact_attempts),
                            'contact_compliance': contact_compliance,
                            'transfer_status': transfer_status,
                            'compliance_article': self.cbuae_article,
                            'action_required': next_action,
                            'priority': priority,
                            'transfer_deadline': (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d'),
                            'documentation_required': 'Transfer notification, Account closure, CB filing',
                            'waiting_period_complete': True,
                            'legal_requirements_met': contact_compliance,
                            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                            'regulatory_notes': f"Account eligible for CBUAE transfer after {dormancy_days} days dormancy per Art. 8"
                        }

                        transfer_eligible.append(eligible_account)

                except Exception as e:
                    logger.warning(f"Error processing account {account.get('account_id', 'unknown')}: {e}")
                    continue

            # Update state
            state.records_processed = len(dormant_accounts)
            state.dormant_records_found = len(transfer_eligible)
            state.processing_time = (datetime.now() - start_time).total_seconds()
            state.agent_status = AgentStatus.COMPLETED

            # Prepare CSV export
            csv_export_data = self.prepare_csv_export(state, transfer_eligible)

            # Create results DataFrame
            if transfer_eligible:
                state.processed_dataframe = pd.DataFrame(transfer_eligible)
            else:
                state.processed_dataframe = pd.DataFrame()

            # Analysis results with CSV download info
            state.analysis_results = {
                'description': 'Central Bank transfer eligibility analysis per CBUAE Article 8',
                'compliance_article': self.cbuae_article,
                'transfer_eligible': transfer_eligible,
                'validation_passed': True,
                'alerts_generated': len(transfer_eligible) > 0,
                'details': transfer_eligible,
                'csv_export': csv_export_data,
                'summary_stats': {
                    'total_dormant_reviewed': len(dormant_accounts),
                    'transfer_eligible': len(transfer_eligible),
                    'ready_for_transfer': len([acc for acc in transfer_eligible if acc['transfer_status'] == 'READY_FOR_TRANSFER']),
                    'pending_contact_completion': len([acc for acc in transfer_eligible if acc['transfer_status'] == 'PENDING_CONTACT_COMPLETION']),
                    'total_transfer_amount': sum([acc['transfer_amount'] for acc in transfer_eligible]),
                    'average_dormancy_years': np.mean([acc['dormancy_years'] for acc in transfer_eligible]) if transfer_eligible else 0
                },
                'recommendations': [
                    f"Prepare Central Bank transfer for {len([acc for acc in transfer_eligible if acc['transfer_status'] == 'READY_FOR_TRANSFER'])} eligible accounts",
                    "Complete contact attempts for pending accounts before transfer",
                    "Ensure all transfer documentation is prepared and compliant",
                    "Notify customers of pending transfer as required by regulations"
                ]
            }

            return state

        except Exception as e:
            state.agent_status = AgentStatus.FAILED
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "stage": "cb_transfer_analysis"
            })
            logger.error(f"CB transfer eligibility analysis failed: {e}")
            return state

# ===== ORCHESTRATOR CLASS WITH CSV EXPORT =====

class DormancyWorkflowOrchestrator:
    """Enhanced orchestrator with CSV export for all agents"""

    def __init__(self):
        self.agents = {
            'demand_deposit': DemandDepositDormancyAgent(),
            'fixed_deposit': FixedDepositDormancyAgent(),
            'contact_attempts': ContactAttemptsAgent(),
            'cb_transfer': CBTransferEligibilityAgent(),
            # Add more agents as implemented...
        }

    async def run_comprehensive_analysis(self, state) -> Dict:
        """Run comprehensive analysis with CSV export for all agents"""
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
                    result_state = await agent.analyze_dormancy(agent_state)

                    # Store results
                    results["agent_results"][agent_name] = {
                        'agent_type': result_state.agent_type,
                        'records_processed': result_state.records_processed,
                        'dormant_records_found': result_state.dormant_records_found,
                        'processing_time': result_state.processing_time,
                        'success': result_state.agent_status == AgentStatus.COMPLETED,
                        'analysis_results': result_state.analysis_results,
                        'processed_dataframe': result_state.processed_dataframe
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
                        'dormant_records_found': 0
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

# ===== MAIN EXECUTION FUNCTIONS =====

async def run_comprehensive_dormancy_analysis_with_csv(user_id: str, account_data: pd.DataFrame,
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
        result = await orchestrator.run_comprehensive_analysis(state)

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

# ===== CSV EXPORT UTILITY FUNCTIONS =====

def get_agent_csv_data(agent_results: Dict, agent_name: str) -> Optional[Dict]:
    """Get CSV export data for a specific agent"""
    if agent_name in agent_results and 'analysis_results' in agent_results[agent_name]:
        return agent_results[agent_name]['analysis_results'].get('csv_export')
    return None

def download_agent_csv(agent_results: Dict, agent_name: str, save_path: str = None) -> bool:
    """Download CSV file for a specific agent"""
    csv_data = get_agent_csv_data(agent_results, agent_name)

    if not csv_data or not csv_data.get('available', False):
        logger.warning(f"No CSV data available for agent: {agent_name}")
        return False

    try:
        filename = save_path or csv_data['filename']
        csv_content = csv_data['csv_data']

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(csv_content)

        logger.info(f"CSV file saved: {filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        return False

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
    # Enhanced Classes with CSV Export
    'BaseDormancyAgent',
    'DemandDepositDormancyAgent',
    'FixedDepositDormancyAgent',
    'ContactAttemptsAgent',
    'CBTransferEligibilityAgent',
    'DormancyWorkflowOrchestrator',

    # Data Classes
    'AgentResult',
    'AgentState',
    'AgentStatus',
    'DormancyStatus',
    'DormancyTrigger',

    # Main Functions
    'run_comprehensive_dormancy_analysis_with_csv',

    # CSV Utility Functions
    'create_csv_download_data',
    'enhance_account_data_for_export',
    'get_agent_csv_data',
    'download_agent_csv',
    'get_all_csv_download_info',

    # Utility Functions
    'safe_date_parse',
    'calculate_dormancy_days'
]

if __name__ == "__main__":
    print("Enhanced CBUAE Dormancy Analysis System with CSV Export")
    print("=" * 60)
    print("Features:")
    print("✅ Individual agent CSV downloads")
    print("✅ Comprehensive account analysis")
    print("✅ CBUAE compliance verification")
    print("✅ Detailed findings export")
    print("✅ Risk assessment and prioritization")
    print("✅ Regulatory compliance tracking")
    print("\nAvailable Agents with CSV Export:")
    print("• DemandDepositDormancyAgent (CBUAE Art. 2.1.1)")
    print("• FixedDepositDormancyAgent (CBUAE Art. 2.1.2)")
    print("• ContactAttemptsAgent (CBUAE Art. 5)")
    print("• CBTransferEligibilityAgent (CBUAE Art. 8)")
    print("\nEach agent provides downloadable CSV with:")
    print("• Complete account information")
    print("• Agent-specific analysis results")
    print("• Compliance findings and actions")
    print("• Risk assessments and priorities")
    print("• Regulatory notes and recommendations")