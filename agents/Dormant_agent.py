import pandas as pd
import numpy as np
import secrets
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import io
import base64
import json

# NEW: Llama 3 8B Integration
from langchain_community.llms import Ollama
import asyncio

logger = logging.getLogger(__name__)

# ===== MISSING ENUMS AND CLASSES =====

class ActivityStatus(Enum):
    """Activity status for agent results"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    LOW_ACTIVITY = "LOW_ACTIVITY"
    MEDIUM_ACTIVITY = "MEDIUM_ACTIVITY"
    HIGH_ACTIVITY = "HIGH_ACTIVITY"


class Article3Stage(Enum):
    """CBUAE Article 3 Process Stages"""
    STAGE_1_CONTACT_REQUIRED = "STAGE_1_CONTACT_REQUIRED"
    STAGE_2_INSTRUMENTS_NOTIFY = "STAGE_2_INSTRUMENTS_NOTIFY"
    STAGE_3_SAFE_DEPOSIT_NOTICE = "STAGE_3_SAFE_DEPOSIT_NOTICE"
    STAGE_4_WAITING_PERIOD = "STAGE_4_WAITING_PERIOD"
    STAGE_5_TRANSFER_TO_LEDGER = "STAGE_5_TRANSFER_TO_LEDGER"
    STAGE_6_UNCLAIMED_TRANSFER = "STAGE_6_UNCLAIMED_TRANSFER"
    STAGE_7_COURT_APPLICATION = "STAGE_7_COURT_APPLICATION"
    STAGE_8_ACCOUNT_CLOSURE = "STAGE_8_ACCOUNT_CLOSURE"
    STAGE_9_ACCESS_CONTROL = "STAGE_9_ACCESS_CONTROL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Article3Priority(Enum):
    """Priority levels for Article 3 actions"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DormancyStatus(Enum):
    """Dormancy status classifications"""
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    IN_PROGRESS = "IN_PROGRESS"
    PROCESSED = "PROCESSED"


class TransitionQuality(Enum):
    """Quality levels for transition analysis"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class ReactivationSpeed(Enum):
    """Speed of reactivation after trigger"""
    IMMEDIATE = "IMMEDIATE"
    DELAYED = "DELAYED"
    SLOW = "SLOW"
    UNKNOWN = "UNKNOWN"


class ReactivationTrigger(Enum):
    """Valid reactivation triggers"""
    CUSTOMER_TRANSACTION = "CUSTOMER_TRANSACTION"
    CUSTOMER_LOGIN = "CUSTOMER_LOGIN"
    CUSTOMER_DEPOSIT = "CUSTOMER_DEPOSIT"
    CUSTOMER_WITHDRAWAL = "CUSTOMER_WITHDRAWAL"
    CUSTOMER_INQUIRY = "CUSTOMER_INQUIRY"
    BANK_CONTACT = "BANK_CONTACT"
    UNKNOWN = "UNKNOWN"




def safe_dual_column_filter(df: pd.DataFrame, primary_column: str, secondary_column: str,
                           contains_pattern: str, case_sensitive: bool = False) -> pd.Series:
    """ filtering that checks BOTH columns"""
    try:
        primary_mask = pd.Series([False] * len(df), index=df.index)
        secondary_mask = pd.Series([False] * len(df), index=df.index)

        # Check primary column
        if primary_column in df.columns:
            primary_series = df[primary_column].fillna('').astype(str)
            primary_mask = primary_series.str.contains(contains_pattern, case=case_sensitive, na=False, regex=True)

        # Check secondary column
        if secondary_column in df.columns:
            secondary_series = df[secondary_column].fillna('').astype(str)
            secondary_mask = secondary_series.str.contains(contains_pattern, case=case_sensitive, na=False, regex=True)

        # Combine with OR logic
        combined_mask = primary_mask | secondary_mask
        logger.info(f"Dual filtering for '{contains_pattern}': {combined_mask.sum()} accounts found")

        return combined_mask

    except Exception as e:
        logger.error(f"Error in dual column filtering: {e}")
        return pd.Series([False] * len(df), index=df.index)


def calculate_dormancy_days(last_transaction, report_date=None):
    """ dormancy calculation with better error handling"""
    if report_date is None:
        report_date = datetime.now()

    if pd.isna(last_transaction):
        return 0

    try:
        if isinstance(last_transaction, str):
            last_date = pd.to_datetime(last_transaction)
        else:
            last_date = last_transaction

        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()

        dormancy_days = (report_date - last_date).days
        return max(0, dormancy_days)
    except Exception as e:
        logger.warning(f"Date calculation error: {e}")
        return 0


def create_csv_export(dormant_accounts, agent_name):
    """Create  CSV export with additional metadata"""
    if not dormant_accounts:
        return {'available': False, 'records': 0}

    df = pd.DataFrame(dormant_accounts)

    # Add metadata
    df['export_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df['agent_name'] = agent_name
    df['enhancement_type'] = 'dual_column_filtering'

    csv_string = df.to_csv(index=False)

    return {
        'available': True,
        'filename': f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        'csv_data': csv_string,
        'csv_content': csv_string,  # For Streamlit compatibility
        'records': len(dormant_accounts),
        'file_size_kb': len(csv_string.encode('utf-8')) / 1024
    }


def determine_activity_status(total_accounts: int) -> ActivityStatus:
    """Determine activity status based on account count"""
    if total_accounts == 0:
        return ActivityStatus.INACTIVE
    elif total_accounts < 10:
        return ActivityStatus.LOW_ACTIVITY
    elif total_accounts < 50:
        return ActivityStatus.MEDIUM_ACTIVITY
    else:
        return ActivityStatus.HIGH_ACTIVITY


def create_csv_download_data(df: pd.DataFrame, filename: str) -> Dict:
    """Create CSV download data structure"""
    return {
        'data': df.to_csv(index=False),
        'filename': filename,
        'mime_type': 'text/csv'
    }




class BaseDormancyAgent:
    """ base class for all dormancy agents with dual-column filtering and Llama 3 8B integration"""

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.agent_id = f"{agent_type}_{secrets.token_hex(8)}"

        #  column mappings
        self.csv_columns = {
            'customer_id': 'customer_id',
            'account_id': 'account_id',
            'account_type': 'account_type',
            'account_subtype': 'account_subtype',
            'account_status': 'account_status',
            'last_transaction_date': 'last_transaction_date',
            'balance_current': 'balance_current',
            'dormancy_status': 'dormancy_status',
            'currency': 'currency',
            'contact_attempts_made': 'contact_attempts_made',
            'last_contact_date': 'last_contact_date'
        }

    def filter_accounts_(self, df: pd.DataFrame, pattern: str) -> pd.DataFrame:
        """filtering using both account_type and account_subtype"""
        mask = safe_dual_column_filter(
            df,
            self.csv_columns['account_type'],
            self.csv_columns['account_subtype'],
            pattern
        )
        return df[mask].copy()

    def safe_get_value(self, row, column, default=''):
        """Safely get value from row"""
        try:
            if column in row.index:
                value = row[column]
                return value if pd.notna(value) else default
            return default
        except:
            return default

    async def generate_llama_recommendations(self, dormant_accounts: List[Dict],
                                             agent_context: Dict) -> Dict:
        """Generate Llama 3 8B recommendations for this agent"""
        try:
            return llama_engine.generate_recommendations(
                self.agent_type, dormant_accounts, agent_context
            )
        except Exception as e:
            logger.error(f"Llama recommendation generation failed for {self.agent_type}: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }

    def analyze_dormancy(self, state) -> Dict:
        """Base method to be implemented by all agents"""
        raise NotImplementedError("Each agent must implement analyze_dormancy method")




class SafeDepositDormancyAgent(BaseDormancyAgent):
    """Agent 1: Safe Deposit Dormancy Analysis - CBUAE Article 3.7 with Llama 3 8B"""

    def __init__(self):
        super().__init__("safe_deposit_dormancy")
        self.cbuae_article = "CBUAE Art. 3.7"
        self.account_patterns = [
            'SAFE_DEPOSIT', 'SDB', 'SAFE DEPOSIT', 'DEPOSIT_BOX',
            'LOCKER', 'VAULT', 'BOX', 'SAFETY', 'DEPOSIT BOX'
        ]

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            # filtering for safe deposits
            pattern = '|'.join(self.account_patterns)
            safe_deposits = self.filter_accounts(df, pattern)

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in safe_deposits.iterrows():
                last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 2.6: 3+ years for court application
                if dormancy_days >= 1095:  # 3 years
                    dormant_accounts.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'dormancy_days': dormancy_days,
                        'dormancy_threshold_days': 1095,
                        'last_transaction_date': str(last_transaction),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Safe deposit box charges outstanding 3+ years - File court application for box access',
                        'priority': 'High',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Generate Llama 3 8B recommendations
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': ' safe deposit dormancy analysis per CBUAE Article 2.6 - 3+ years threshold',
                'account_patterns': self.account_patterns,
                'threshold_days': 1095
            }

            llama_recommendations = self.generate_llama_recommendations(dormant_accounts, agent_context)

            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(dormant_accounts))

            return {
                'success': True,
                'agent_name': 'Safe Deposit Dormancy with Llama 3 8B',
                'activity_status': activity_status.value,
                'records_processed': len(safe_deposits),
                'dormant_records_found': len(dormant_accounts),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'Safe deposit dormancy analysis per CBUAE Article 2.6 - 3+ years threshold',
                    'compliance_article': self.cbuae_article,
                    'dormant_accounts': dormant_accounts,
                    'csv_export': create_csv_export(dormant_accounts, 'safe_deposit_dormancy'),
                    'llm_recommendations': llama_recommendations  # NEW: Llama 3 8B recommendations
                }
            }

        except Exception as e:
            logger.error(f"Safe deposit agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }


# ===== Continue with other agents (showing pattern for next few agents) =====

class InvestmentAccountInactivityAgent(BaseDormancyAgent):
    """Agent 2: Investment Account Inactivity - CBUAE Article 2.3 with Llama 3 8B"""

    def __init__(self):
        super().__init__("investment_account_inactivity")
        self.cbuae_article = "CBUAE Art. 2.3"
        self.account_patterns = [
            'INVESTMENT', 'PORTFOLIO', 'MUTUAL', 'SECURITIES',
            'EQUITY', 'BOND', 'FUND', 'TRADING', 'BROKERAGE',
            'STOCK', 'SHARE', 'ASSET_MGMT'
        ]

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            # filtering for investment accounts
            pattern = '|'.join(self.account_patterns)
            investment_accounts = self.filter_accounts(df, pattern)

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in investment_accounts.iterrows():
                last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
                balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 2.3: 3 years from final maturity or redemption
                if dormancy_days >= 1095:
                    dormant_accounts.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'dormancy_threshold_days': 1095,
                        'threshold_description': '3 years from final maturity or redemption',
                        'last_transaction_date': str(last_transaction),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Review investment product status and contact customer',
                        'priority': 'High',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })

            # Generate Llama 3 8B recommendations
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'investment account inactivity analysis per CBUAE Article 2.3 - 3 years threshold',
                'account_patterns': self.account_patterns,
                'threshold_days': 1095
            }

            llama_recommendations = self.generate_llama_recommendations(dormant_accounts, agent_context)

            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(dormant_accounts))

            return {
                'success': True,
                'agent_name': 'Investment Account Inactivity with Llama 3 8B',
                'activity_status': activity_status.value,
                'records_processed': len(investment_accounts),
                'dormant_records_found': len(dormant_accounts),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'investment account inactivity analysis per CBUAE Article 2.3 - 3 years threshold',
                    'compliance_article': self.cbuae_article,
                    'dormant_accounts': dormant_accounts,
                    'csv_export': create_csv_export(dormant_accounts, 'investment_inactivity'),
                    'llm_recommendations': llama_recommendations  # NEW: Llama 3 8B recommendations
                }
            }

        except Exception as e:
            logger.error(f"Investment agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

# CONTINUE WITH OTHER EXISTING AGENTS (3-6, 9)...

class FixedDepositInactivityAgent(BaseDormancyAgent):
    """Agent 3: Fixed Deposit Inactivity - CBUAE Article 2.2"""

    def __init__(self):
        super().__init__("fixed_deposit_inactivity")
        self.cbuae_article = "CBUAE Art. 2.2"
        self.account_patterns = [
            'FIXED_DEPOSIT', 'TERM_DEPOSIT', 'CD', 'CERTIFICATE',
            'TIME_DEPOSIT', 'FIXED', 'TERM', 'DEPOSIT_FIXED',
            'FD', 'TD', 'CERTIFICATE_DEPOSIT'
        ]

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")


            pattern = '|'.join(self.account_patterns)
            fixed_deposits = self.filter_accounts(df, pattern)

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in fixed_deposits.iterrows():
                last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
                balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
                dormancy_days =calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 2.2: 3 years post-maturity (FIXED: was 365 days)
                if dormancy_days >= 1095:  # COMPLIANCE FIX: Changed from 365 to 1095 days
                    dormant_accounts.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'dormancy_threshold_days': 1095,  # 3 years post-maturity per CBUAE Article 2.2
                        'threshold_description': '3 years since deposit matured',
                        'last_transaction_date': str(last_transaction),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Monitor maturity dates and contact customer',
                        'priority': 'Medium',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })
                    # Generate Llama 3 8B recommendations
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'fixed deposit dormancy analysis per CBUAE Article 2.2 - 3+ years threshold',
                'account_patterns': self.account_patterns,
                'threshold_days': 1095
            }

            llama_recommendations = self.generate_llama_recommendations(dormant_accounts, agent_context)

            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(dormant_accounts))

            return {
                'success': True,
                'agent_name': 'Fixed Deposit Inactivity',
                'activity_status': activity_status.value,
                'records_processed': len(fixed_deposits),
                'dormant_records_found': len(dormant_accounts),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'fixed deposit inactivity analysis per CBUAE Article 2.2 - 3 years post-maturity threshold',
                    'compliance_article': self.cbuae_article,
                    'dormant_accounts': dormant_accounts,
                    'csv_export': create_csv_export(dormant_accounts, 'fixed_deposit_inactivity')
                }
            }

        except Exception as e:
            logger.error(f"Fixed deposit agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

class DemandDepositInactivityAgent(BaseDormancyAgent):
    """Agent 4: Demand Deposit Inactivity - CBUAE Article 2.1 (MOST COMMON)"""

    def __init__(self):
        super().__init__("demand_deposit_inactivity")
        self.cbuae_article = "CBUAE Art. 2.1"
        self.account_patterns = [
            'CURRENT', 'SAVINGS', 'CHECKING', 'DEMAND',
            'JOINT', 'INDIVIDUAL', 'CORPORATE', 'PERSONAL',
            'BUSINESS', 'RETAIL', 'COMMERCIAL'
        ]

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            # filtering for demand deposits
            pattern = '|'.join(self.account_patterns)
            demand_deposits = self.filter_accounts(df, pattern)

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in demand_deposits.iterrows():
                last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
                balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 2.1: 3 years inactivity (FIXED: was 365 days)
                if dormancy_days >= 1095:  # COMPLIANCE FIX: Changed from 365 to 1095 days
                    dormant_accounts.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'dormancy_threshold_days': 1095,  # 3 years per CBUAE Article 2.1
                        'threshold_description': '3 years per CBUAE Article 2.1',
                        'last_transaction_date': str(last_transaction),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Flag as dormant and initiate contact',
                        'priority': 'Medium',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'Demand deposit dormancy analysis per CBUAE Article 2.6 - 3+ years threshold',
                'account_patterns': self.account_patterns,
                'threshold_days': 1095
            }

            llama_recommendations = self.generate_llama_recommendations(dormant_accounts, agent_context)
            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(dormant_accounts))

            return {
                'success': True,
                'agent_name': 'Demand Deposit Inactivity',
                'activity_status': activity_status.value,
                'records_processed': len(demand_deposits),
                'dormant_records_found': len(dormant_accounts),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'demand deposit inactivity analysis per CBUAE Article 2.1 - 3 years threshold',
                    'compliance_article': self.cbuae_article,
                    'dormant_accounts': dormant_accounts,
                    'csv_export': create_csv_export(dormant_accounts, 'demand_deposit_inactivity')
                }
            }

        except Exception as e:
            logger.error(f"Demand deposit agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

class UnclaimedPaymentInstrumentsAgent(BaseDormancyAgent):
    """Agent 5: Unclaimed Payment Instruments - CBUAE Article 2.4"""

    def __init__(self):
        super().__init__("unclaimed_payment_instruments")
        self.cbuae_article = "CBUAE Art. 2.4"
        self.account_patterns = [
            'PAYMENT', 'CHECK', 'DRAFT', 'INSTRUMENT', 'CHEQUE',
            'CASHIER', 'BANKER', 'MONEY_ORDER', 'REMITTANCE',
            'TRANSFER', 'UNCLAIMED', 'OUTSTANDING'
        ]

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            # filtering for payment instruments
            pattern = '|'.join(self.account_patterns)
            payment_instruments = self.filter_accounts(df, pattern)

            dormant_accounts = []
            report_date = datetime.now()

            for idx, account in payment_instruments.iterrows():
                last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
                balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # CBUAE Article 2.4: 1 year unclaimed - ALREADY CORRECT ✅
                if dormancy_days >= 365:  # COMPLIANCE ✅: Already correct at 365 days (1 year)
                    dormant_accounts.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'dormancy_threshold_days': 365,  # 1 year per CBUAE Article 2.4 - ALREADY CORRECT
                        'threshold_description': '1 year unclaimed per CBUAE Article 2.4',
                        'last_transaction_date': str(last_transaction),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Process for ledger transfer',
                        'priority': 'Critical',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'Unclaimed Instruments dormancy analysis per CBUAE Article 2.6 - 3+ years threshold',
                'account_patterns': self.account_patterns,
                'threshold_days': 365
            }

            llama_recommendations = self.generate_llama_recommendations(dormant_accounts, agent_context)
            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(dormant_accounts))

            return {
                'success': True,
                'agent_name': 'Unclaimed Payment Instruments',
                'activity_status': activity_status.value,
                'records_processed': len(payment_instruments),
                'dormant_records_found': len(dormant_accounts),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'unclaimed payment instruments analysis per CBUAE Article 2.4 - 1 year threshold (COMPLIANT)',
                    'compliance_article': self.cbuae_article,
                    'dormant_accounts': dormant_accounts,
                    'csv_export': create_csv_export(dormant_accounts, 'unclaimed_payment_instruments')
                }
            }

        except Exception as e:
            logger.error(f"Unclaimed payment instruments agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

# FIXED: Rename to match orchestrator expectation
class EligibleForCBUAETransferAgent(BaseDormancyAgent):
    """Agent 6: Eligible for CBUAE Transfer - Real CBUAE Article 8 Implementation"""

    def __init__(self):
        super().__init__("eligible_for_cbuae_transfer")
        self.cbuae_article = "CBUAE Art. 8"

        # Real Article 8 thresholds
        self.transfer_thresholds = {
            'REGULAR_ACCOUNTS': 1825,  # 5 years (Point 1)
            'UNCLAIMED_INSTRUMENTS': 1095,  # 3 years (Point 2)
            'SAFE_DEPOSIT_BOX': 1825  # 5 years (Point 4)
        }

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            transfer_eligible = []
            report_date = datetime.now()

            for idx, account in df.iterrows():
                # Apply basic eligibility check (simplified for semantic correctness)
                last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
                balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                # Basic eligibility - accounts dormant for 5+ years
                if dormancy_days >= self.transfer_thresholds['REGULAR_ACCOUNTS']:
                    transfer_eligible.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_days': dormancy_days,
                        'transfer_reason': 'Regular Account - 5 Year Dormancy',
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Transfer to CBUAE Unclaimed Balances Account',
                        'priority': 'Critical - Regulatory Requirement',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'Dormant Accounts eligible for CBUAE Transfer - Article 8 ',
                'account_patterns': self.account_patterns,
                'threshold_days': {
            'REGULAR_ACCOUNTS': 1825,  # 5 years (Point 1)
            'UNCLAIMED_INSTRUMENTS': 1095,  # 3 years (Point 2)
            'SAFE_DEPOSIT_BOX': 1825  # 5 years (Point 4)
        }
            }

            llama_recommendations = self.generate_llama_recommendations(transfer_eligible, agent_context)
            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(transfer_eligible))

            return {
                'success': True,
                'agent_name': 'CBUAE Article 8 Transfer Eligibility',
                'activity_status': activity_status.value,
                'records_processed': len(df),
                'dormant_records_found': len(transfer_eligible),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'CBUAE Article 8 transfer eligibility analysis',
                    'compliance_article': self.cbuae_article,
                    'transfer_eligible_accounts': transfer_eligible,
                    'csv_export': create_csv_export(transfer_eligible, 'cbuae_article8_transfer_eligible')
                }
            }

        except Exception as e:
            logger.error(f"CBUAE Article 8 agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

# ===== MISSING AGENTS (7, 8, 10) - IMPLEMENTED =====

class Article3ProcessNeededAgent(BaseDormancyAgent):
    """Agent 7: Article 3 Process Needed - Dormancy Detection Only"""

    def __init__(self):
        super().__init__("article_3_process_needed")
        self.cbuae_article = "CBUAE Art. 3"
        self.dormancy_patterns = ['DORMANT', 'FLAGGED', 'INACTIVE', 'PENDING']

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            article3_candidates = []
            report_date = datetime.now()

            for idx, account in df.iterrows():
                # Check if account requires Article 3 process based on dormancy
                if self._requires_article3_process(account, report_date):

                    last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
                    balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
                    dormancy_status = self.safe_get_value(account, self.csv_columns['dormancy_status'])
                    account_status = self.safe_get_value(account, self.csv_columns['account_status'])
                    dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                    article3_candidates.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'dormancy_status': dormancy_status,
                        'account_status': account_status,
                        'dormancy_days': dormancy_days,
                        'dormancy_threshold_days': 1095,  # 3 years per CBUAE
                        'last_transaction_date': str(last_transaction),
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'dormancy_category': self._categorize_dormancy_type(account),
                        'article3_trigger': self._identify_article3_trigger(account, dormancy_days),
                        'process_urgency': self._determine_process_urgency(dormancy_days, balance),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Initiate Article 3 dormancy processes',
                        'priority': self._determine_priority(dormancy_days, float(balance) if pd.notna(balance) else 0),
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })
            # Generate Llama 3 8B recommendations
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'Dormant Accounts on which Article 3 need to be applied - Article 3',
                'account_patterns': self.account_patterns,
                'threshold_days': 1095
            }

            llama_recommendations = self.generate_llama_recommendations(len(article3_candidates), agent_context)
            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(article3_candidates))

            return {
                'success': True,
                'agent_name': 'Article 3 Process Needed',
                'activity_status': activity_status.value,
                'records_processed': len(df),
                'dormant_records_found': len(article3_candidates),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'Article 3 dormancy process identification - flags accounts requiring CBUAE Article 3 processes',
                    'compliance_article': self.cbuae_article,
                    'dormancy_threshold': '3+ years (1095 days) per CBUAE requirements',
                    'trigger_criteria': 'Dormant status + 3+ years inactivity across all account types',
                    'dormant_accounts': article3_candidates,
                    'csv_export': create_csv_export(article3_candidates, 'article_3_process_needed')
                }
            }

        except Exception as e:
            logger.error(f"Article 3 process agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

    def _requires_article3_process(self, account, report_date) -> bool:
        """Check if account requires Article 3 process based on dormancy criteria"""
        try:
            last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
            account_type = self.safe_get_value(account, self.csv_columns['account_type'])
            account_status = self.safe_get_value(account, self.csv_columns['account_status'])
            dormancy_status = self.safe_get_value(account, self.csv_columns['dormancy_status'])

            dormancy_days = calculate_dormancy_days(last_transaction, report_date)

            # Check for dormancy indicators - multiple ways to identify dormant accounts
            is_dormant = (
                'dormant' in str(dormancy_status).lower() or
                'inactive' in str(account_status).lower() or
                'flagged' in str(dormancy_status).lower() or
                'pending' in str(dormancy_status).lower() or
                'closed' in str(account_status).lower()
            )

            # Article 3 applies to all account types after 3+ years dormancy
            meets_dormancy_threshold = dormancy_days >= 1095  # 3 years per CBUAE

            # Additional check for specific account types
            account_specific_check = self._check_account_type_dormancy(account_type, dormancy_days)

            return is_dormant and (meets_dormancy_threshold or account_specific_check)

        except Exception as e:
            logger.warning(f"Article 3 process check failed for account: {e}")
            return False

    def _check_account_type_dormancy(self, account_type, dormancy_days):
        """Check account-type specific dormancy requirements"""
        account_type_upper = str(account_type).upper()

        # Safe deposit boxes - special handling per CBUAE Article 2.6/3.7
        if any(keyword in account_type_upper for keyword in ['SAFE_DEPOSIT', 'SDB', 'SAFE DEPOSIT', 'DEPOSIT_BOX', 'LOCKER']):
            return dormancy_days >= 1095  # 3+ years for court application

        # Investment accounts - per CBUAE Article 2.3
        elif any(keyword in account_type_upper for keyword in ['INVESTMENT', 'PORTFOLIO', 'MUTUAL', 'SECURITIES', 'EQUITY', 'BOND', 'FUND']):
            return dormancy_days >= 1095  # 3+ years from final maturity

        # Fixed deposits - per CBUAE Article 2.2
        elif any(keyword in account_type_upper for keyword in ['FIXED_DEPOSIT', 'TERM_DEPOSIT', 'CD', 'CERTIFICATE', 'TERM', 'FIXED']):
            return dormancy_days >= 1095  # 3+ years post-maturity

        # Payment instruments - per CBUAE Article 2.4
        elif any(keyword in account_type_upper for keyword in ['PAYMENT', 'CHECK', 'DRAFT', 'INSTRUMENT', 'CHEQUE', 'CASHIER']):
            return dormancy_days >= 365   # 1+ year unclaimed

        # Regular accounts (current, savings) - per CBUAE Article 2.1
        else:
            return dormancy_days >= 1095  # 3+ years standard

    def _categorize_dormancy_type(self, account):
        """Categorize the type of dormancy for Article 3 process"""
        account_type = self.safe_get_value(account, self.csv_columns['account_type']).upper()

        if any(keyword in account_type for keyword in ['SAFE_DEPOSIT', 'SDB', 'LOCKER']):
            return 'SAFE_DEPOSIT_DORMANCY'
        elif any(keyword in account_type for keyword in ['INVESTMENT', 'PORTFOLIO', 'MUTUAL', 'SECURITIES']):
            return 'INVESTMENT_DORMANCY'
        elif any(keyword in account_type for keyword in ['FIXED_DEPOSIT', 'TERM_DEPOSIT', 'CD', 'CERTIFICATE']):
            return 'FIXED_DEPOSIT_DORMANCY'
        elif any(keyword in account_type for keyword in ['PAYMENT', 'CHECK', 'DRAFT', 'INSTRUMENT']):
            return 'PAYMENT_INSTRUMENT_DORMANCY'
        elif any(keyword in account_type for keyword in ['CURRENT', 'SAVINGS', 'CHECKING']):
            return 'DEMAND_DEPOSIT_DORMANCY'
        else:
            return 'GENERAL_ACCOUNT_DORMANCY'

    def _identify_article3_trigger(self, account, dormancy_days):
        """Identify what triggered the Article 3 process requirement"""
        triggers = []

        if dormancy_days >= 1095:
            triggers.append('THREE_YEAR_DORMANCY_THRESHOLD')

        account_status = self.safe_get_value(account, self.csv_columns['account_status'])
        dormancy_status = self.safe_get_value(account, self.csv_columns['dormancy_status'])

        if 'dormant' in str(dormancy_status).lower():
            triggers.append('DORMANT_STATUS_FLAG')

        if 'inactive' in str(account_status).lower():
            triggers.append('INACTIVE_ACCOUNT_STATUS')

        # Check for high-value dormancy
        balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
        if float(balance) >= 25000:
            triggers.append('HIGH_VALUE_DORMANCY')

        return triggers if triggers else ['GENERAL_DORMANCY_DETECTION']

    def _determine_process_urgency(self, dormancy_days, balance):
        """Determine urgency level for Article 3 process initiation"""
        try:
            balance_float = float(balance) if pd.notna(balance) else 0
        except:
            balance_float = 0

        if dormancy_days >= 1825:  # 5+ years
            return 'CRITICAL'
        elif dormancy_days >= 1460:  # 4+ years
            return 'HIGH'
        elif balance_float >= 50000:  # High value regardless of time
            return 'HIGH'
        elif dormancy_days >= 1095:  # 3+ years
            return 'MEDIUM'
        else:
            return 'LOW'

    def _determine_priority(self, dormancy_days, balance):
        """Determine priority for Article 3 process handling"""
        urgency = self._determine_process_urgency(dormancy_days, balance)

        if urgency == 'CRITICAL':
            return 'Critical'
        elif urgency == 'HIGH':
            return 'High'
        elif urgency == 'MEDIUM':
            return 'Medium'
        else:
            return 'Low'
class ContactAttemptsNeededAgent(BaseDormancyAgent):
    """Agent 8: CBUAE-Compliant Contact Attempts Agent"""

    def __init__(self):
        super().__init__("contact_attempts_needed")
        self.cbuae_article = "CBUAE Art. 3"
        self.dormancy_patterns = ['DORMANT', 'FLAGGED', 'INACTIVE']

        # CBUAE contact requirements by account type
        self.contact_requirements = {
            'INDIVIDUAL_STANDARD': 3,
            'INDIVIDUAL_HIGH_VALUE': 5,
            'CORPORATE_STANDARD': 5,
            'CORPORATE_HIGH_VALUE': 7
        }

        # Minimum days between contact attempts
        self.contact_interval_days = 30

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Check for required columns and handle gracefully
            required_columns = [
                'customer_id', 'account_id', 'account_type', 'account_status',
                'contact_attempts_made', 'last_contact_date', 'balance_current'
            ]

            # Optional columns with defaults
            optional_columns = {
                'customer_type': 'INDIVIDUAL',
                'customer_contactable': True,
                'dormancy_start_date': None
            }

            # Filter dormant accounts using existing dual-column filtering
            dormant_accounts = self.filter_accounts(df, '|'.join(self.dormancy_patterns))

            contact_needed = []
            report_date = datetime.now()

            for idx, account in dormant_accounts.iterrows():
                # Get account characteristics with safe defaults
                customer_type = self.safe_get_value(account, 'customer_type', 'INDIVIDUAL')
                balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
                account_type = self.safe_get_value(account, self.csv_columns['account_type'], 'UNKNOWN')

                # Determine required contact attempts
                required_attempts = self.get_required_contact_attempts(
                    customer_type, balance, account_type
                )

                # Get current contact attempts
                contact_attempts = self.safe_get_value(account, self.csv_columns['contact_attempts_made'], 0)
                try:
                    contact_attempts_int = int(float(contact_attempts)) if pd.notna(contact_attempts) else 0
                except:
                    contact_attempts_int = 0

                # Check if customer is contactable
                customer_contactable = self.safe_get_value(account, 'customer_contactable', True)
                if not customer_contactable:
                    continue  # Skip if customer opted out

                # Check timeline requirements
                last_contact_date = self.safe_get_value(account, self.csv_columns.get('last_contact_date', 'last_contact_date'))
                days_since_contact = self.calculate_days_since(last_contact_date)

                # Check dormancy duration (don't contact immediately)
                dormancy_start = self.safe_get_value(account, 'dormancy_start_date')
                if dormancy_start is None:
                    # Fallback to last transaction date
                    dormancy_start = self.safe_get_value(account, self.csv_columns['last_transaction_date'])

                dormancy_days = self.calculate_days_since(dormancy_start)

                # Apply CBUAE contact rules
                needs_contact = False
                contact_reason = ""

                if contact_attempts_int < required_attempts:
                    if dormancy_days >= 365:  # Wait 1 year before first contact
                        if contact_attempts_int == 0:
                            needs_contact = True
                            contact_reason = "Initial contact attempt required after 1 year dormancy"
                        elif days_since_contact >= self.contact_interval_days:
                            needs_contact = True
                            contact_reason = f"Next contact attempt due ({self.contact_interval_days}+ days since last contact)"

                if needs_contact:
                    contact_needed.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': account_type,
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'customer_type': customer_type,
                        'balance_current': float(balance) if pd.notna(balance) else 0.0,
                        'contact_attempts_made': contact_attempts_int,
                        'required_attempts': required_attempts,
                        'remaining_attempts': max(0, required_attempts - contact_attempts_int),
                        'days_since_last_contact': days_since_contact,
                        'dormancy_days': dormancy_days,
                        'contact_reason': contact_reason,
                        'compliance_article': self.cbuae_article,
                        'priority': self.determine_contact_priority(
                            balance, customer_type, contact_attempts_int, dormancy_days
                        ),
                        'next_contact_due_date': self.calculate_next_contact_date(last_contact_date),
                        'last_transaction_date': str(self.safe_get_value(account, self.csv_columns['last_transaction_date'])),
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })
            # Generate Llama 3 8B recommendations
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'Contact Attempts made to accounts which are dormant- ',
                'account_patterns': self.account_patterns,
                'threshold_days': 1095
            }

            llama_recommendations = self.generate_llama_recommendations(contact_needed, agent_context)
            # Determine activity status
            activity_status = determine_activity_status(len(contact_needed))
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'agent_name': 'CBUAE Contact Attempts Agent',
                'activity_status': activity_status.value,
                'records_processed': len(dormant_accounts),
                'dormant_records_found': len(contact_needed),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'CBUAE-compliant contact attempts analysis with variable requirements by customer type and balance',
                    'compliance_article': self.cbuae_article,
                    'contact_requirements_summary': self.get_requirements_summary(),
                    'dormant_accounts': contact_needed,
                    'csv_export': create_csv_export(contact_needed, 'cbuae_contact_attempts')
                }
            }

        except Exception as e:
            logger.error(f"CBUAE Contact attempts agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

    def get_required_contact_attempts(self, customer_type: str, balance: float, account_type: str) -> int:
        """Determine required contact attempts based on CBUAE rules"""
        try:
            balance_float = float(balance) if pd.notna(balance) else 0
        except:
            balance_float = 0

        if customer_type.upper() == 'CORPORATE':
            return 7 if balance_float >= 100000 else 5  # Higher requirements for corporates
        else:
            return 5 if balance_float >= 25000 else 3  # Higher requirements for high-value individuals

    def determine_contact_priority(self, balance: float, customer_type: str,
                                   attempts_made: int, dormancy_days: int) -> str:
        """Determine priority based on account characteristics"""
        try:
            balance_float = float(balance) if pd.notna(balance) else 0
        except:
            balance_float = 0

        if balance_float >= 100000 or customer_type.upper() == 'CORPORATE':
            return 'CRITICAL'
        elif balance_float >= 25000 or dormancy_days >= 1095:  # High value or 3+ years
            return 'HIGH'
        elif attempts_made == 0 and dormancy_days >= 365:
            return 'URGENT'
        else:
            return 'MEDIUM'

    def calculate_days_since(self, date_value) -> int:
        """Calculate days since a given date"""
        if pd.isna(date_value) or date_value is None:
            return 999999  # Very large number for missing dates

        try:
            if isinstance(date_value, str):
                date_obj = pd.to_datetime(date_value)
            else:
                date_obj = date_value

            return (datetime.now() - date_obj).days
        except:
            return 999999

    def calculate_next_contact_date(self, last_contact_date):
        """Calculate when next contact attempt is due"""
        if pd.isna(last_contact_date) or last_contact_date is None:
            return datetime.now().strftime('%Y-%m-%d')  # Contact immediately if no previous contact

        try:
            if isinstance(last_contact_date, str):
                last_date = pd.to_datetime(last_contact_date)
            else:
                last_date = last_contact_date

            next_date = last_date + timedelta(days=self.contact_interval_days)
            return next_date.strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')

    def get_requirements_summary(self) -> Dict:
        """Summary of CBUAE contact requirements"""
        return {
            'individual_standard_accounts': '3 contact attempts required',
            'individual_high_value_accounts': '5 contact attempts required (≥25K AED)',
            'corporate_standard_accounts': '5 contact attempts required',
            'corporate_high_value_accounts': '7 contact attempts required (≥100K AED)',
            'minimum_interval_between_attempts': f'{self.contact_interval_days} days',
            'initial_contact_trigger': '365 days of dormancy',
            'compliance_article': self.cbuae_article,
            'high_value_threshold_individual': '25,000 AED',
            'high_value_threshold_corporate': '100,000 AED'
        }

class HighValueDormantAgent(BaseDormancyAgent):
    """Agent 9: High Value Dormant - Banking-Grade Risk Management"""

    def __init__(self):
        super().__init__("high_value_dormant")
        self.cbuae_article = "Risk Management & Customer Protection"
        self.dormancy_patterns = ['DORMANT', 'FLAGGED', 'INACTIVE']
        self.high_value_threshold = 25000  # AED

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            high_value_dormant = []
            report_date = datetime.now()

            for idx, account in df.iterrows():
                balance = self.safe_get_value(account, self.csv_columns['balance_current'], 0)
                dormancy_status = self.safe_get_value(account, self.csv_columns['dormancy_status'])
                last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])

                # Check if account is dormant and high value
                is_dormant = any(pattern.lower() in str(dormancy_status).lower() for pattern in self.dormancy_patterns)
                try:
                    balance_float = float(balance) if pd.notna(balance) else 0.0
                except:
                    balance_float = 0.0

                is_high_value = balance_float >= self.high_value_threshold
                dormancy_days = calculate_dormancy_days(last_transaction, report_date)

                if is_dormant and is_high_value and dormancy_days >= 365:
                    high_value_dormant.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'balance_current': balance_float,
                        'dormancy_status': dormancy_status,
                        'dormancy_days': dormancy_days,
                        'high_value_threshold': self.high_value_threshold,
                        'last_transaction_date': str(last_transaction),
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Priority customer reactivation process',
                        'priority': 'Critical',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })
            # Generate Llama 3 8B recommendations
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'High Value Dormant - Banking-Grade Risk Management',
                'account_patterns': self.account_patterns,
                'threshold_days': 1095
            }

            llama_recommendations = self.generate_llama_recommendations(high_value_dormant, agent_context)
            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(high_value_dormant))

            return {
                'success': True,
                'agent_name': 'High Value Dormant',
                'activity_status': activity_status.value,
                'records_processed': len(df),
                'dormant_records_found': len(high_value_dormant),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': 'high value dormant accounts analysis',
                    'compliance_article': self.cbuae_article,
                    'high_value_accounts': high_value_dormant,
                    'csv_export': create_csv_export(high_value_dormant, 'high_value_dormant')
                }
            }

        except Exception as e:
            logger.error(f"High value dormant agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

class DormantToActiveTransitionsAgent(BaseDormancyAgent):
    """Agent 10: Dormant to Active Transitions - Monitoring"""

    def __init__(self):
        super().__init__("dormant_to_active_transitions")
        self.cbuae_article = "Monitoring"
        self.recent_activity_threshold = 30  # 30 days
        self.previous_snapshot_path = "previous_account_snapshot.csv"

    def analyze_dormancy(self, state) -> Dict:
        try:
            start_time = datetime.now()
            df = state.input_dataframe

            if df is None or df.empty:
                raise ValueError("No input data provided")

            # Simplified transition detection
            transitions = []
            report_date = datetime.now()

            for idx, account in df.iterrows():
                last_transaction = self.safe_get_value(account, self.csv_columns['last_transaction_date'])
                account_status = self.safe_get_value(account, self.csv_columns['account_status'])
                dormancy_status = self.safe_get_value(account, self.csv_columns['dormancy_status'])

                # Check for recent activity (within threshold)
                recent_activity_days = calculate_dormancy_days(last_transaction, report_date)

                # Look for accounts that might be transitioning
                has_dormancy_history = 'dormant' in str(dormancy_status).lower()
                has_recent_activity = recent_activity_days <= self.recent_activity_threshold
                is_currently_active = 'active' in str(account_status).lower()

                if has_dormancy_history and has_recent_activity and is_currently_active:
                    transitions.append({
                        'customer_id': self.safe_get_value(account, self.csv_columns['customer_id']),
                        'account_id': self.safe_get_value(account, self.csv_columns['account_id']),
                        'account_type': self.safe_get_value(account, self.csv_columns['account_type']),
                        'account_subtype': self.safe_get_value(account, self.csv_columns['account_subtype']),
                        'dormancy_status': dormancy_status,
                        'current_status': account_status,
                        'last_transaction_date': str(last_transaction),
                        'days_since_activity': recent_activity_days,
                        'transition_threshold': self.recent_activity_threshold,
                        'compliance_article': self.cbuae_article,
                        'action_required': 'Monitor reactivation status',
                        'priority': 'Low',
                        'transition_type': 'Potential Dormant to Active',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d')
                    })
            # Generate Llama 3 8B recommendations
            agent_context = {
                'compliance_article': self.cbuae_article,
                'description': 'safe deposit dormancy analysis per CBUAE Article 2.6 - 3+ years threshold',
                'account_patterns': self.account_patterns,
                'threshold_days': 1095
            }

            llama_recommendations = self.generate_llama_recommendations(transitions, agent_context)
            processing_time = (datetime.now() - start_time).total_seconds()
            activity_status = determine_activity_status(len(transitions))

            return {
                'success': True,
                'agent_name': 'Dormant to Active Transitions',
                'activity_status': activity_status.value,
                'records_processed': len(df),
                'dormant_records_found': len(transitions),
                'processing_time': processing_time,
                'analysis_results': {
                    'description': f'dormant to active transitions monitoring (≤{self.recent_activity_threshold} days)',
                    'compliance_article': self.cbuae_article,
                    'dormant_accounts': transitions,
                    'csv_export': create_csv_export(transitions, 'dormant_to_active_transitions')
                }
            }

        except Exception as e:
            logger.error(f"Dormant to active transitions agent failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'activity_status': ActivityStatus.INACTIVE.value,
                'records_processed': 0,
                'dormant_records_found': 0
            }

# ===== FIXED ORCHESTRATOR FOR ALL 10 AGENTS =====

class CompleteDormancyOrchestrator:
    """Complete orchestrator for all 10 dormancy agents with Llama 3 8B integration"""

    def __init__(self):
        self.agents = {
            'safe_deposit_dormancy': SafeDepositDormancyAgent(),
            'investment_account_inactivity': InvestmentAccountInactivityAgent(),
            'fixed_deposit_inactivity': FixedDepositInactivityAgent(),
            'demand_deposit_inactivity': DemandDepositInactivityAgent(),
            'unclaimed_payment_instruments': UnclaimedPaymentInstrumentsAgent(),
            'eligible_for_cbuae_transfer': EligibleForCBUAETransferAgent(),  # FIXED: Proper class name
            'article_3_process_needed': Article3ProcessNeededAgent(),        # FIXED: Now exists
            'contact_attempts_needed': ContactAttemptsNeededAgent(),         # FIXED: Now exists
            'high_value_dormant': HighValueDormantAgent(),
            'dormant_to_active_transitions': DormantToActiveTransitionsAgent()  # FIXED: Now exists
        }

    async def run_complete_analysis(self, data: pd.DataFrame) -> Dict:
        """Run complete analysis with all agents and Llama 3 8B recommendations"""
        try:
            start_time = datetime.now()

            logger.info(f"🚀 Starting COMPLETE ANALYSIS with Llama 3 8B integration - {len(self.agents)} agents")
            logger.info(f"Data shape: {data.shape}")

            results = {
                "success": True,
                "agent_results": {},
                "summary": {},
                "processing_time": 0.0,
                "enhancement_type": "llama_3_8b_enhanced"
            }

            # Create mock state
            class MockState:
                def __init__(self, df):
                    self.input_dataframe = df
                    self.session_id = secrets.token_hex(8)
                    self.user_id = "llama_user"

            state = MockState(data)

            # Execute all agents with Llama recommendations
            total_dormant = 0
            successful_agents = 0

            for agent_name, agent in self.agents.items():
                try:
                    logger.info(f"🔄 Running agent with Llama 3 8B: {agent_name}")

                    # Run agent analysis with async support for Llama
                    if asyncio.iscoroutinefunction(agent.analyze_dormancy):
                        result = await agent.analyze_dormancy(state)
                    else:
                        # For synchronous agents, create a task
                        result = await asyncio.to_thread(agent.analyze_dormancy, state)

                    results["agent_results"][agent_name] = result

                    if result.get('success', False):
                        dormant_found = result.get('dormant_records_found', 0)
                        processed = result.get('records_processed', 0)
                        total_dormant += dormant_found
                        successful_agents += 1

                        # Check if Llama recommendations were generated
                        llm_recs = result.get('analysis_results', {}).get('llm_recommendations', {})
                        llm_status = "✅ Llama 3 8B" if llm_recs.get('success') else "⚠️ Fallback"

                        logger.info(f"✅ {agent_name}: {dormant_found}/{processed} dormant accounts | {llm_status}")
                    else:
                        logger.error(f"❌ {agent_name}: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    results["agent_results"][agent_name] = {
                        'success': False,
                        'error': str(e),
                        'activity_status': ActivityStatus.INACTIVE.value,
                        'records_processed': 0,
                        'dormant_records_found': 0
                    }

            # Create comprehensive summary
            processing_time = (datetime.now() - start_time).total_seconds()
            total_processed = sum([r.get('records_processed', 0) for r in results["agent_results"].values()])

            results["summary"] = {
                "total_accounts_processed": total_processed,
                "total_dormant_accounts_found": total_dormant,
                "agents_executed": successful_agents,
                "total_agents": len(self.agents),
                "success_rate": (successful_agents / len(self.agents) * 100) if len(self.agents) > 0 else 0,
                "enhancement_used": "llama_3_8b_enhanced",
                "processing_time": processing_time,
                "llm_integration": "llama3:8b-instruct",
                "agent_breakdown": {
                    agent_name: {
                        'success': result.get('success', False),
                        'dormant_found': result.get('dormant_records_found', 0),
                        'processed': result.get('records_processed', 0),
                        'llm_recommendations': result.get('analysis_results', {}).get('llm_recommendations', {}).get(
                            'success', False)
                    }
                    for agent_name, result in results["agent_results"].items()
                }
            }

            results["processing_time"] = processing_time

            logger.info(f"🎯 COMPLETE LLAMA 3 8B analysis finished:")
            logger.info(f"   - Total dormant accounts found: {total_dormant}")
            logger.info(f"   - Successful agents: {successful_agents}/{len(self.agents)}")
            logger.info(f"   - Processing time: {processing_time:.2f}s")
            logger.info(f"   - LLM integration: Llama 3 8B Instruct")

            return results

        except Exception as e:
            logger.error(f"Complete Llama 3 8B analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_results": {},
                "summary": {
                    "total_accounts_processed": 0,
                    "total_dormant_accounts_found": 0,
                    "agents_executed": 0,
                    "total_agents": len(self.agents) if hasattr(self, 'agents') else 10,
                    "success_rate": 0,
                    "enhancement_used": "llama_3_8b_enhanced",
                    "processing_time": 0.0,
                    "llm_integration": "llama3:8b-instruct"
                }
            }


# ===== MAIN EXECUTION FUNCTION =====

async def run_complete_dormancy_analysis_with_llama(data: pd.DataFrame) -> Dict:
    """
    Main function to run complete dormancy analysis with Llama 3 8B recommendations
    """

    logger.info("🚀 COMPLETE DORMANCY ANALYSIS with Llama 3 8B Integration")
    logger.info("   All 10 Agents with Intelligent Recommendations")

    # Validate input data
    if data is None or data.empty:
        return {
            "success": False,
            "error": "No data provided for analysis",
            "summary": {"total_dormant_accounts_found": 0}
        }

    # Initialize orchestrator with Llama integration
    orchestrator = CompleteDormancyOrchestrator()

    # Run complete analysis with Llama 3 8B
    result = await orchestrator.run_complete_analysis(data)

    # logging
    if result.get("success", False):
        summary = result["summary"]
        total_dormant = summary["total_dormant_accounts_found"]
        success_rate = summary["success_rate"]

        logger.info(f"✅ COMPLETE LLAMA 3 8B ANALYSIS SUCCESS:")
        logger.info(f"   🎯 Total dormant accounts: {total_dormant}")
        logger.info(f"   📊 Agent success rate: {success_rate:.1f}%")
        logger.info(f"   ⚡ Processing time: {summary['processing_time']:.2f}s")
        logger.info(f"   🤖 LLM integration: {summary['llm_integration']}")

    else:
        logger.error(f"❌ COMPLETE LLAMA 3 8B ANALYSIS FAILED: {result.get('error', 'Unknown error')}")

    return result


# ===== EXPORTS =====

__all__ = [
    'CompleteDormancyOrchestrator',
    'run_complete_dormancy_analysis_with_llama',
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
    'BaseDormancyAgent',
    'safe_dual_column_filter',
    'calculate_dormancy_days',
    'create_csv_export',
    'ActivityStatus',
    'DormancyStatus'
]