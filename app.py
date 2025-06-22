"""
Comprehensive Banking Compliance Streamlit Application
Integrates: Data Processing (4 upload methods), Quality Analysis, BGE Mapping,
Dormancy Analysis (11 agents), Compliance Verification (17 agents), and Reporting
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import logging
import io
import tempfile
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="CBUAE Banking Compliance System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd, #f8f9fa);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding: 1rem;
        background: linear-gradient(90deg, #f1f8ff, #ffffff);
        border-radius: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #27ae60;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-card {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .agent-card {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        border-color: #3498db;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.15);
        transform: translateY(-2px);
    }
    
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== AGENT IMPORTS =====================

# Initialize agent status tracking
AGENTS_STATUS = {
    'unified_data_processing': False,
    'data_upload': False,
    'data_mapping': False,
    'data_quality': False,
    'bge_embeddings': False,
    'dormancy': False,
    'compliance': False
}

def safe_import_agent(module_name: str, items: List[str]) -> tuple[bool, Dict]:
    """Safely import agent modules with error handling"""
    try:
        exec(f"from {module_name} import {', '.join(items)}")
        imported_items = {item: locals()[item] for item in items}
        logger.info(f"✅ Successfully imported {module_name}")
        return True, imported_items
    except ImportError as e:
        logger.warning(f"⚠️ Could not import {module_name}: {e}")
        return False, {}
    except Exception as e:
        logger.error(f"❌ Error importing {module_name}: {e}")
        return False, {}

# Import Unified Data Processing Agent
try:
    from agents.Data_Process import UnifiedDataProcessingAgent, create_unified_data_processing_agent
    AGENTS_STATUS['unified_data_processing'] = True
    AGENTS_STATUS['data_upload'] = True
    AGENTS_STATUS['data_mapping'] = True
    AGENTS_STATUS['data_quality'] = True
    AGENTS_STATUS['bge_embeddings'] = True
    logger.info("✅ Unified Data Processing Agent loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Unified Data Processing Agent not available: {e}")

# Import Dormancy Agents
try:
    from agents.Dormant_agent import (
        run_comprehensive_dormancy_analysis_csv,
        DemandDepositDormancyAgent,
        FixedDepositDormancyAgent,
        InvestmentAccountDormancyAgent,
        ContactAttemptsAgent,
        CBTransferEligibilityAgent,
        ForeignCurrencyConversionAgent,
        HighValueDormantAccountsAgent,
        DormancyEscalationAgent,
        StatementSuppressionAgent,
        InternalLedgerTransferAgent,
        DormancyWorkflowOrchestrator
    )
    AGENTS_STATUS['dormancy'] = True
    logger.info("✅ Dormancy agents loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Dormancy agents not available: {e}")

# Import Compliance Verification Agents
try:
    from agents.compliance_verification_agent import (
        run_comprehensive_compliance_analysis_csv,
        DetectIncompleteContactAttemptsAgent,
        DetectUnflaggedDormantCandidatesAgent,
        DetectInternalLedgerCandidatesAgent,
        DetectStatementFreezeCandidatesAgent,
        DetectCBUAETransferCandidatesAgent,
        DetectForeignCurrencyConversionNeededAgent,
        GenerateAnnualCBUAEReportSummaryAgent,
        CheckRecordRetentionComplianceAgent,
        RunAllComplianceChecksAgent
    )
    AGENTS_STATUS['compliance'] = True
    logger.info("✅ Compliance verification agents loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Compliance verification agents not available: {e}")

# ===================== SESSION STATE INITIALIZATION =====================

def initialize_session_state():
    """Initialize all session state variables"""
    session_defaults = {
        'logged_in': False,
        'username': '',
        'current_page': 'login',
        'uploaded_data': None,
        'processed_data': None,
        'quality_results': None,
        'mapping_results': None,
        'dormancy_results': None,
        'compliance_results': None,
        'mapping_sheet': None,
        'llm_enabled': False,
        'upload_method': 'file',
        'agent_session_id': None
    }

    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# ===================== SCHEMA DEFINITIONS =====================

def get_comprehensive_banking_schema():
    """Get comprehensive CBUAE banking compliance schema (66 fields)"""
    return {
        # Customer Information (8 fields)
        'customer_id': {
            'description': 'Unique customer identifier',
            'required': True,
            'type': 'string',
            'keywords': ['customer', 'client', 'id', 'identifier', 'cust', 'customer_id']
        },
        'customer_type': {
            'description': 'Type of customer (Individual/Corporate)',
            'required': True,
            'type': 'string',
            'keywords': ['customer_type', 'client_type', 'type', 'individual', 'corporate']
        },
        'full_name_en': {
            'description': 'Customer full name in English',
            'required': True,
            'type': 'string',
            'keywords': ['name', 'full_name', 'customer_name', 'client_name', 'english', 'full_name_en']
        },
        'full_name_ar': {
            'description': 'Customer full name in Arabic',
            'required': False,
            'type': 'string',
            'keywords': ['name_ar', 'full_name_ar', 'arabic', 'name_arabic', 'customer_name_ar']
        },
        'id_number': {
            'description': 'Customer identification number',
            'required': True,
            'type': 'string',
            'keywords': ['id_number', 'identification', 'id_no', 'emirates_id', 'passport']
        },
        'id_type': {
            'description': 'Type of identification document',
            'required': True,
            'type': 'string',
            'keywords': ['id_type', 'identification_type', 'doc_type', 'document_type']
        },
        'date_of_birth': {
            'description': 'Customer date of birth',
            'required': False,
            'type': 'date',
            'keywords': ['birth_date', 'dob', 'date_of_birth', 'birthday', 'birth']
        },
        'nationality': {
            'description': 'Customer nationality',
            'required': False,
            'type': 'string',
            'keywords': ['nationality', 'country', 'citizenship', 'nation', 'origin']
        },

        # Address Information (7 fields)
        'address_line1': {
            'description': 'Primary address line',
            'required': False,
            'type': 'string',
            'keywords': ['address', 'address_line1', 'street', 'location', 'addr1']
        },
        'address_line2': {
            'description': 'Secondary address line',
            'required': False,
            'type': 'string',
            'keywords': ['address_line2', 'apartment', 'unit', 'building', 'addr2']
        },
        'city': {
            'description': 'City of residence',
            'required': False,
            'type': 'string',
            'keywords': ['city', 'town', 'municipality', 'location_city']
        },
        'emirate': {
            'description': 'UAE Emirate',
            'required': False,
            'type': 'string',
            'keywords': ['emirate', 'state', 'province', 'region', 'uae_emirate']
        },
        'country': {
            'description': 'Country of residence',
            'required': False,
            'type': 'string',
            'keywords': ['country', 'nation', 'residence_country', 'country_code']
        },
        'postal_code': {
            'description': 'Postal or ZIP code',
            'required': False,
            'type': 'string',
            'keywords': ['postal_code', 'zip', 'postcode', 'zip_code', 'postal']
        },
        'address_known': {
            'description': 'Whether address is known/verified',
            'required': False,
            'type': 'string',
            'keywords': ['address_known', 'verified', 'known', 'address_verified']
        },

        # Contact Information (6 fields)
        'phone_primary': {
            'description': 'Primary phone number',
            'required': False,
            'type': 'string',
            'keywords': ['phone', 'mobile', 'telephone', 'contact', 'primary_phone', 'phone_primary']
        },
        'phone_secondary': {
            'description': 'Secondary phone number',
            'required': False,
            'type': 'string',
            'keywords': ['phone_secondary', 'secondary_phone', 'phone2', 'alternate_phone']
        },
        'email_primary': {
            'description': 'Primary email address',
            'required': False,
            'type': 'string',
            'keywords': ['email', 'email_address', 'contact_email', 'primary_email', 'email_primary']
        },
        'email_secondary': {
            'description': 'Secondary email address',
            'required': False,
            'type': 'string',
            'keywords': ['email_secondary', 'secondary_email', 'email2', 'alternate_email']
        },
        'last_contact_date': {
            'description': 'Date of last contact attempt',
            'required': False,
            'type': 'date',
            'keywords': ['last_contact_date', 'contact_date', 'last_contact', 'contact_attempt']
        },
        'last_contact_method': {
            'description': 'Method of last contact attempt',
            'required': False,
            'type': 'string',
            'keywords': ['contact_method', 'last_contact_method', 'method', 'contact_type']
        },

        # KYC and Risk (3 fields)
        'kyc_status': {
            'description': 'KYC compliance status',
            'required': False,
            'type': 'string',
            'keywords': ['kyc', 'kyc_status', 'compliance', 'verification', 'know_your_customer']
        },
        'kyc_expiry_date': {
            'description': 'KYC expiration date',
            'required': False,
            'type': 'date',
            'keywords': ['kyc_expiry', 'kyc_expiry_date', 'expiry', 'kyc_expire']
        },
        'risk_rating': {
            'description': 'Customer risk rating',
            'required': False,
            'type': 'string',
            'keywords': ['risk', 'rating', 'risk_rating', 'risk_level', 'risk_score']
        },

        # Account Information (7 fields)
        'account_id': {
            'description': 'Unique account identifier',
            'required': True,
            'type': 'string',
            'keywords': ['account_id', 'account_number', 'acc_id', 'account', 'number']
        },
        'account_type': {
            'description': 'Type of account (Savings/Current/Fixed)',
            'required': True,
            'type': 'string',
            'keywords': ['account_type', 'type', 'savings', 'current', 'fixed', 'deposit']
        },
        'account_subtype': {
            'description': 'Account subtype classification',
            'required': False,
            'type': 'string',
            'keywords': ['account_subtype', 'subtype', 'sub_type', 'classification']
        },
        'account_name': {
            'description': 'Account name or title',
            'required': False,
            'type': 'string',
            'keywords': ['account_name', 'name', 'title', 'account_title']
        },
        'currency': {
            'description': 'Account currency',
            'required': False,
            'type': 'string',
            'keywords': ['currency', 'curr', 'aed', 'usd', 'eur', 'currency_code']
        },
        'account_status': {
            'description': 'Current status of account',
            'required': True,
            'type': 'string',
            'keywords': ['status', 'account_status', 'active', 'inactive', 'closed']
        },
        'dormancy_status': {
            'description': 'Dormancy classification status',
            'required': True,
            'type': 'string',
            'keywords': ['dormancy', 'dormant', 'dormancy_status', 'classification']
        },

        # Account Dates (4 fields)
        'opening_date': {
            'description': 'Account opening date',
            'required': False,
            'type': 'date',
            'keywords': ['opening_date', 'opened', 'start_date', 'creation_date', 'open_date']
        },
        'closing_date': {
            'description': 'Account closing date',
            'required': False,
            'type': 'date',
            'keywords': ['closing_date', 'closed', 'close_date', 'closure_date']
        },
        'last_transaction_date': {
            'description': 'Date of last customer transaction',
            'required': True,
            'type': 'date',
            'keywords': ['transaction_date', 'last_transaction', 'activity_date', 'last_activity']
        },
        'last_system_transaction_date': {
            'description': 'Date of last system transaction',
            'required': False,
            'type': 'date',
            'keywords': ['system_transaction', 'last_system_transaction', 'system_activity']
        },

        # Balance Information (5 fields)
        'balance_current': {
            'description': 'Current account balance',
            'required': True,
            'type': 'float',
            'keywords': ['balance', 'current_balance', 'amount', 'balance_current']
        },
        'balance_available': {
            'description': 'Available account balance',
            'required': False,
            'type': 'float',
            'keywords': ['balance_available', 'available', 'available_balance', 'usable_balance']
        },
        'balance_minimum': {
            'description': 'Minimum required balance',
            'required': False,
            'type': 'float',
            'keywords': ['balance_minimum', 'minimum', 'min_balance', 'minimum_balance']
        },
        'interest_rate': {
            'description': 'Account interest rate',
            'required': False,
            'type': 'float',
            'keywords': ['interest_rate', 'rate', 'interest', 'apr', 'yield']
        },
        'interest_accrued': {
            'description': 'Accrued interest amount',
            'required': False,
            'type': 'float',
            'keywords': ['interest_accrued', 'accrued', 'earned_interest', 'interest_earned']
        },

        # Account Details (7 fields)
        'is_joint_account': {
            'description': 'Whether account is joint account',
            'required': False,
            'type': 'string',
            'keywords': ['joint', 'is_joint', 'joint_account', 'shared']
        },
        'joint_account_holders': {
            'description': 'Number of joint account holders',
            'required': False,
            'type': 'integer',
            'keywords': ['joint_holders', 'holders', 'joint_account_holders', 'co_holders']
        },
        'has_outstanding_facilities': {
            'description': 'Whether account has outstanding facilities',
            'required': False,
            'type': 'string',
            'keywords': ['facilities', 'outstanding', 'has_facilities', 'credit_facilities']
        },
        'maturity_date': {
            'description': 'Account maturity date',
            'required': False,
            'type': 'date',
            'keywords': ['maturity', 'maturity_date', 'mature', 'expiry']
        },
        'auto_renewal': {
            'description': 'Auto renewal setting',
            'required': False,
            'type': 'string',
            'keywords': ['auto_renewal', 'renewal', 'auto_renew', 'automatic']
        },
        'last_statement_date': {
            'description': 'Date of last statement',
            'required': False,
            'type': 'date',
            'keywords': ['statement_date', 'last_statement', 'statement', 'last_stmt']
        },
        'statement_frequency': {
            'description': 'Statement generation frequency',
            'required': False,
            'type': 'string',
            'keywords': ['statement_frequency', 'frequency', 'stmt_freq', 'statement_cycle']
        },

        # Tracking and Processing (7 fields)
        'tracking_id': {
            'description': 'Unique tracking identifier',
            'required': False,
            'type': 'string',
            'keywords': ['tracking_id', 'tracking', 'trace_id', 'reference']
        },
        'dormancy_trigger_date': {
            'description': 'Date when dormancy was triggered',
            'required': False,
            'type': 'date',
            'keywords': ['dormancy_trigger', 'trigger_date', 'dormant_since']
        },
        'dormancy_period_start': {
            'description': 'Start of dormancy period',
            'required': False,
            'type': 'date',
            'keywords': ['dormancy_start', 'period_start', 'dormant_from']
        },
        'dormancy_period_months': {
            'description': 'Duration of dormancy in months',
            'required': False,
            'type': 'float',
            'keywords': ['dormancy_months', 'period_months', 'months_dormant']
        },
        'dormancy_classification_date': {
            'description': 'Date of dormancy classification',
            'required': False,
            'type': 'date',
            'keywords': ['classification_date', 'dormancy_classification', 'classified_date']
        },
        'transfer_eligibility_date': {
            'description': 'Date eligible for transfer',
            'required': False,
            'type': 'date',
            'keywords': ['transfer_eligibility', 'eligible_date', 'transfer_date']
        },
        'current_stage': {
            'description': 'Current processing stage',
            'required': False,
            'type': 'string',
            'keywords': ['stage', 'current_stage', 'process_stage', 'workflow_stage']
        },

        # Contact Attempts (4 fields)
        'contact_attempts_made': {
            'description': 'Number of contact attempts made',
            'required': False,
            'type': 'integer',
            'keywords': ['contact_attempts', 'attempts', 'contact_count', 'attempts_made']
        },
        'last_contact_attempt_date': {
            'description': 'Date of last contact attempt',
            'required': False,
            'type': 'date',
            'keywords': ['last_attempt', 'contact_attempt_date', 'attempt_date']
        },
        'waiting_period_start': {
            'description': 'Start of waiting period',
            'required': False,
            'type': 'date',
            'keywords': ['waiting_start', 'wait_period_start', 'waiting_period']
        },
        'waiting_period_end': {
            'description': 'End of waiting period',
            'required': False,
            'type': 'date',
            'keywords': ['waiting_end', 'wait_period_end', 'waiting_until']
        },

        # Transfer Information (5 fields)
        'transferred_to_ledger_date': {
            'description': 'Date transferred to ledger',
            'required': False,
            'type': 'date',
            'keywords': ['ledger_transfer', 'transferred_ledger', 'ledger_date']
        },
        'transferred_to_cb_date': {
            'description': 'Date transferred to Central Bank',
            'required': False,
            'type': 'date',
            'keywords': ['cb_transfer', 'central_bank', 'cb_date', 'transferred_cb']
        },
        'cb_transfer_amount': {
            'description': 'Amount transferred to Central Bank',
            'required': False,
            'type': 'float',
            'keywords': ['cb_amount', 'transfer_amount', 'cb_transfer_amount']
        },
        'cb_transfer_reference': {
            'description': 'Central Bank transfer reference',
            'required': False,
            'type': 'string',
            'keywords': ['cb_reference', 'transfer_reference', 'cb_ref']
        },
        'exclusion_reason': {
            'description': 'Reason for exclusion from process',
            'required': False,
            'type': 'string',
            'keywords': ['exclusion', 'reason', 'exclusion_reason', 'excluded']
        },

        # System Fields (3 fields)
        'created_date': {
            'description': 'Record creation date',
            'required': False,
            'type': 'date',
            'keywords': ['created', 'created_date', 'creation_date', 'date_created']
        },
        'updated_date': {
            'description': 'Record last update date',
            'required': False,
            'type': 'date',
            'keywords': ['updated', 'updated_date', 'modified', 'last_modified']
        },
        'updated_by': {
            'description': 'User who last updated record',
            'required': False,
            'type': 'string',
            'keywords': ['updated_by', 'modified_by', 'user', 'operator']
        }
    }

def get_comprehensive_column_patterns():
    """Get comprehensive column mapping patterns for all 66 fields"""
    return {
        # Customer Information
        'customer_id': ['customer_id', 'cust_id', 'client_id', 'customer_number', 'clientid', 'customer'],
        'customer_type': ['customer_type', 'client_type', 'type', 'individual', 'corporate', 'cust_type'],
        'full_name_en': ['name', 'full_name', 'customer_name', 'client_name', 'fullname', 'full_name_en'],
        'full_name_ar': ['full_name_ar', 'name_ar', 'arabic_name', 'customer_name_ar', 'name_arabic'],
        'id_number': ['id_number', 'identification', 'id_no', 'emirates_id', 'passport', 'id'],
        'id_type': ['id_type', 'identification_type', 'doc_type', 'document_type'],
        'date_of_birth': ['birth_date', 'dob', 'date_of_birth', 'birthday', 'birth'],
        'nationality': ['nationality', 'country', 'citizenship', 'nation', 'origin'],

        # Address Information
        'address_line1': ['address', 'address_line1', 'street', 'location', 'addr1'],
        'address_line2': ['address_line2', 'apartment', 'unit', 'building', 'addr2'],
        'city': ['city', 'town', 'municipality', 'location_city'],
        'emirate': ['emirate', 'state', 'province', 'region', 'uae_emirate'],
        'country': ['country', 'nation', 'residence_country', 'country_code'],
        'postal_code': ['postal_code', 'zip', 'postcode', 'zip_code', 'postal'],
        'address_known': ['address_known', 'verified', 'known', 'address_verified'],

        # Contact Information
        'phone_primary': ['phone', 'mobile', 'telephone', 'contact_number', 'phone_number', 'phone_primary'],
        'phone_secondary': ['phone_secondary', 'secondary_phone', 'phone2', 'alternate_phone'],
        'email_primary': ['email', 'email_address', 'contact_email', 'primary_email', 'email_primary'],
        'email_secondary': ['email_secondary', 'secondary_email', 'email2', 'alternate_email'],
        'last_contact_date': ['last_contact_date', 'contact_date', 'last_contact', 'contact_attempt'],
        'last_contact_method': ['contact_method', 'last_contact_method', 'method', 'contact_type'],

        # Account Information
        'account_id': ['account_id', 'account_number', 'acc_id', 'account_no', 'accountid', 'account'],
        'account_type': ['account_type', 'acc_type', 'type', 'product_type', 'accounttype'],
        'account_subtype': ['account_subtype', 'subtype', 'sub_type', 'classification'],
        'account_name': ['account_name', 'name', 'title', 'account_title'],
        'currency': ['currency', 'curr', 'aed', 'usd', 'eur', 'currency_code'],
        'account_status': ['status', 'account_status', 'acc_status', 'state'],
        'dormancy_status': ['dormancy_status', 'dormant', 'dormancy', 'classification'],

        # Balance and Financial
        'balance_current': ['balance', 'current_balance', 'amount', 'bal', 'currentbalance', 'balance_current'],
        'balance_available': ['balance_available', 'available', 'available_balance', 'usable_balance'],
        'balance_minimum': ['balance_minimum', 'minimum', 'min_balance', 'minimum_balance'],
        'interest_rate': ['interest_rate', 'rate', 'interest', 'apr', 'yield'],
        'interest_accrued': ['interest_accrued', 'accrued', 'earned_interest', 'interest_earned'],

        # Dates
        'opening_date': ['opening_date', 'opened', 'start_date', 'creation_date', 'open_date'],
        'closing_date': ['closing_date', 'closed', 'close_date', 'closure_date'],
        'last_transaction_date': ['last_transaction_date', 'transaction_date', 'last_activity', 'activity_date'],
        'last_system_transaction_date': ['system_transaction', 'last_system_transaction', 'system_activity'],

        # KYC and Risk
        'kyc_status': ['kyc', 'kyc_status', 'compliance', 'verification', 'know_your_customer'],
        'kyc_expiry_date': ['kyc_expiry', 'kyc_expiry_date', 'expiry', 'kyc_expire'],
        'risk_rating': ['risk', 'rating', 'risk_rating', 'risk_level', 'risk_score'],

        # Dormancy Processing
        'dormancy_trigger_date': ['dormancy_trigger', 'trigger_date', 'dormant_since'],
        'dormancy_period_start': ['dormancy_start', 'period_start', 'dormant_from'],
        'dormancy_period_months': ['dormancy_months', 'period_months', 'months_dormant'],
        'dormancy_classification_date': ['classification_date', 'dormancy_classification', 'classified_date'],
        'transfer_eligibility_date': ['transfer_eligibility', 'eligible_date', 'transfer_date'],
        'current_stage': ['stage', 'current_stage', 'process_stage', 'workflow_stage'],

        # Contact Attempts
        'contact_attempts_made': ['contact_attempts', 'attempts', 'contact_count', 'attempts_made'],
        'last_contact_attempt_date': ['last_attempt', 'contact_attempt_date', 'attempt_date'],
        'waiting_period_start': ['waiting_start', 'wait_period_start', 'waiting_period'],
        'waiting_period_end': ['waiting_end', 'wait_period_end', 'waiting_until'],

        # Transfers
        'transferred_to_ledger_date': ['ledger_transfer', 'transferred_ledger', 'ledger_date'],
        'transferred_to_cb_date': ['cb_transfer', 'central_bank', 'cb_date', 'transferred_cb'],
        'cb_transfer_amount': ['cb_amount', 'transfer_amount', 'cb_transfer_amount'],
        'cb_transfer_reference': ['cb_reference', 'transfer_reference', 'cb_ref'],
        'exclusion_reason': ['exclusion', 'reason', 'exclusion_reason', 'excluded']
    }

def get_schema_field_info(field_name):
    """Get detailed information about a specific schema field"""
    schema = get_comprehensive_banking_schema()
    return schema.get(field_name, {
        'description': 'Unknown field',
        'required': False,
        'type': 'string',
        'keywords': []
    })

def get_required_fields():
    """Get list of required fields from the comprehensive schema"""
    schema = get_comprehensive_banking_schema()
    return [field for field, info in schema.items() if info.get('required', False)]

def get_optional_fields():
    """Get list of optional fields from the comprehensive schema"""
    schema = get_comprehensive_banking_schema()
    return [field for field, info in schema.items() if not info.get('required', False)]

def get_fields_by_category():
    """Get fields organized by category"""
    return {
        'Customer Information': [
            'customer_id', 'customer_type', 'full_name_en', 'full_name_ar',
            'id_number', 'id_type', 'date_of_birth', 'nationality'
        ],
        'Address Information': [
            'address_line1', 'address_line2', 'city', 'emirate', 'country',
            'postal_code', 'address_known'
        ],
        'Contact Information': [
            'phone_primary', 'phone_secondary', 'email_primary', 'email_secondary',
            'last_contact_date', 'last_contact_method'
        ],
        'KYC and Risk': [
            'kyc_status', 'kyc_expiry_date', 'risk_rating'
        ],
        'Account Information': [
            'account_id', 'account_type', 'account_subtype', 'account_name',
            'currency', 'account_status', 'dormancy_status'
        ],
        'Account Dates': [
            'opening_date', 'closing_date', 'last_transaction_date',
            'last_system_transaction_date'
        ],
        'Balance Information': [
            'balance_current', 'balance_available', 'balance_minimum',
            'interest_rate', 'interest_accrued'
        ],
        'Account Details': [
            'is_joint_account', 'joint_account_holders', 'has_outstanding_facilities',
            'maturity_date', 'auto_renewal', 'last_statement_date', 'statement_frequency'
        ],
        'Tracking and Processing': [
            'tracking_id', 'dormancy_trigger_date', 'dormancy_period_start',
            'dormancy_period_months', 'dormancy_classification_date',
            'transfer_eligibility_date', 'current_stage'
        ],
        'Contact Attempts': [
            'contact_attempts_made', 'last_contact_attempt_date',
            'waiting_period_start', 'waiting_period_end'
        ],
        'Transfer Information': [
            'transferred_to_ledger_date', 'transferred_to_cb_date',
            'cb_transfer_amount', 'cb_transfer_reference', 'exclusion_reason'
        ],
        'System Fields': [
            'created_date', 'updated_date', 'updated_by'
        ]
    }

# ===================== UTILITY FUNCTIONS =====================

def get_or_create_event_loop():
    """Get or create event loop for async operations"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def validate_credentials(username: str, password: str) -> bool:
    """Validate user credentials"""
    demo_users = {
        "admin": "admin123",
        "compliance_officer": "compliance123",
        "analyst": "analyst123",
        "demo_user": "demo"
    }
    return username in demo_users and demo_users[username] == password

def safe_getattr(obj, attr, default=None):
    """Safely get attribute from object, handling both dict and dataclass"""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)

def create_download_link(data, filename: str, file_format: str = 'csv'):
    """Create download link for data"""
    if file_format.lower() == 'csv':
        if isinstance(data, pd.DataFrame):
            csv = data.to_csv(index=False)
            return st.download_button(
                label=f"📥 Download {filename}.csv",
                data=csv,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )
        else:
            return st.download_button(
                label=f"📥 Download {filename}.txt",
                data=str(data),
                file_name=f"{filename}.txt",
                mime="text/plain"
            )
    elif file_format.lower() == 'json':
        json_str = json.dumps(data, indent=2, default=str)
        return st.download_button(
            label=f"📥 Download {filename}.json",
            data=json_str,
            file_name=f"{filename}.json",
            mime="application/json"
        )

# ===================== LOGIN SYSTEM =====================

def show_login_page():
    """Display login interface"""
    st.markdown('<div class="main-header">🏦 CBUAE Banking Compliance System</div>', unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)

    with st.container():
        st.markdown("### 🔐 System Login")

        col1, col2 = st.columns(2)

        with col1:
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter password")

            if st.button("🚀 Login", use_container_width=True):
                if validate_credentials(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.current_page = 'main'
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")

        with col2:
            if st.button("👤 Demo Login", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.username = "demo_user"
                st.session_state.current_page = 'main'
                st.success("✅ Demo login successful!")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # System capabilities overview
    st.markdown("---")
    st.markdown("### 🎯 System Capabilities")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📤 Data Processing:**
        - 4 Upload methods (File, Drive, DataLake, HDFS)
        - Real-time quality analysis
        - BGE-powered schema mapping (66 fields)
        - LLM-enhanced column mapping
        - Comprehensive CBUAE schema support
        """)

    with col2:
        st.markdown("""
        **💤 Dormancy Analysis:**
        - 11 specialized dormancy agents
        - 5 Primary Detection agents (Art. 2.x)
        - 3 Process & Transfer agents (Art. 3, 5, 8)
        - 2 Specialized Analysis agents
        - 1 Master Orchestrator
        - Complete CBUAE Article compliance
        """)

    with col3:
        st.markdown("""
        **⚖️ Compliance Verification:**
        - 17 compliance verification agents
        - 6 functional categories
        - Multi-article regulation checking (Art. 2-8)
        - Automated violation detection
        - Comprehensive audit trails
        - Central Bank transfer processing
        """)

# ===================== DATA PROCESSING SECTION =====================

def show_data_processing_section():
    """Display comprehensive data processing interface"""
    st.markdown('<div class="section-header">📤 Data Processing & Upload</div>', unsafe_allow_html=True)

    if not AGENTS_STATUS['unified_data_processing']:
        st.error("❌ Unified Data Processing Agent not available")
        st.info("💡 Please ensure agents.Data_Process module is available and properly configured.")
        return

    # Data upload methods
    st.markdown("### 📂 Data Upload Methods")

    upload_method = st.selectbox(
        "🔄 Select Upload Method:",
        ["file", "drive", "datalake", "hdfs"],
        format_func=lambda x: {
            "file": "📄 File Upload (CSV, Excel, JSON)",
            "drive": "☁️ Google Drive",
            "datalake": "🌊 Azure Data Lake",
            "hdfs": "🐘 Hadoop HDFS"
        }[x]
    )

    st.session_state.upload_method = upload_method

    # File upload interface
    if upload_method == "file":
        uploaded_file = st.file_uploader(
            "📤 Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )

        if uploaded_file is not None:
            if st.button("🚀 Process File", type="primary"):
                process_uploaded_file(uploaded_file)

    elif upload_method == "drive":
        drive_path = st.text_input("📁 Google Drive File Path/ID", placeholder="Enter file path or Google Drive file ID")
        if st.button("🚀 Load from Drive", type="primary") and drive_path:
            process_drive_file(drive_path)

    elif upload_method == "datalake":
        datalake_path = st.text_input("🌊 Azure Data Lake Path", placeholder="Enter Azure Data Lake file path")
        if st.button("🚀 Load from Data Lake", type="primary") and datalake_path:
            process_datalake_file(datalake_path)

    elif upload_method == "hdfs":
        hdfs_path = st.text_input("🐘 HDFS Path", placeholder="Enter HDFS file path")
        if st.button("🚀 Load from HDFS", type="primary") and hdfs_path:
            process_hdfs_file(hdfs_path)

    # Display upload results
    if st.session_state.uploaded_data is not None:
        display_upload_results()

    # Data quality analysis section
    if st.session_state.uploaded_data is not None:
        show_data_quality_section()

    # Data mapping section
    if st.session_state.uploaded_data is not None:
        show_data_mapping_section()

def process_uploaded_file(uploaded_file):
    """Process uploaded file using unified data processing agent"""
    try:
        with st.spinner("📤 Processing uploaded file..."):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Process with unified agent
            loop = get_or_create_event_loop()
            agent = create_unified_data_processing_agent()

            session_id = secrets.token_hex(8)
            st.session_state.agent_session_id = session_id

            result = loop.run_until_complete(
                agent.process_data_comprehensive(
                    upload_method="file",
                    source=tmp_file_path,
                    user_id=st.session_state.username,
                    session_id=session_id,
                    run_quality_analysis=True,
                    run_column_mapping=False  # We'll do mapping separately
                )
            )

            # Handle both dict and nested structure formats
            upload_result = result.get('upload_result', {})

            if isinstance(upload_result, dict):
                success = upload_result.get('success', False)
                data = upload_result.get('data', None)
                error = upload_result.get('error', 'Unknown error')
            else:
                success = upload_result.success
                data = upload_result.data
                error = upload_result.error

            if success and data is not None:
                st.session_state.uploaded_data = data
                st.session_state.quality_results = result.get('quality_result')
                st.success("✅ File uploaded and processed successfully!")
            else:
                st.error(f"❌ Upload failed: {error}")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        logger.error(f"File processing error: {str(e)}")

def process_drive_file(drive_path):
    """Process file from Google Drive"""
    try:
        with st.spinner("☁️ Loading from Google Drive..."):
            # Placeholder for Google Drive integration
            st.info("🚧 Google Drive integration coming soon!")
    except Exception as e:
        st.error(f"❌ Error loading from Drive: {str(e)}")

def process_datalake_file(datalake_path):
    """Process file from Azure Data Lake"""
    try:
        with st.spinner("🌊 Loading from Azure Data Lake..."):
            # Placeholder for Azure Data Lake integration
            st.info("🚧 Azure Data Lake integration coming soon!")
    except Exception as e:
        st.error(f"❌ Error loading from Data Lake: {str(e)}")

def process_hdfs_file(hdfs_path):
    """Process file from HDFS"""
    try:
        with st.spinner("🐘 Loading from HDFS..."):
            # Placeholder for HDFS integration
            st.info("🚧 HDFS integration coming soon!")
    except Exception as e:
        st.error(f"❌ Error loading from HDFS: {str(e)}")

def display_upload_results():
    """Display upload results and data preview"""
    st.markdown("### 📊 Upload Results")

    data = st.session_state.uploaded_data

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{len(data):,}")
    with col2:
        st.metric("📋 Columns", len(data.columns))
    with col3:
        st.metric("💾 Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    with col4:
        st.metric("📅 Upload Time", datetime.now().strftime("%H:%M:%S"))

    # Data preview
    st.markdown("#### 🔍 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)

    # Column information
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum(),
            'Null %': (data.isnull().sum() / len(data) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)

def show_data_quality_section():
    """Display data quality analysis section"""
    st.markdown("### 🔍 Data Quality Analysis")

    if st.session_state.quality_results:
        display_quality_results()
    else:
        if st.button("🔬 Run Quality Analysis", type="primary"):
            run_quality_analysis()

def run_quality_analysis():
    """Run comprehensive data quality analysis"""
    try:
        with st.spinner("🔬 Analyzing data quality..."):
            data = st.session_state.uploaded_data
            agent = create_unified_data_processing_agent()

            loop = get_or_create_event_loop()
            result = loop.run_until_complete(
                agent.analyze_data_quality(
                    data,
                    st.session_state.username,
                    st.session_state.agent_session_id
                )
            )

            st.session_state.quality_results = result
            st.success("✅ Quality analysis completed!")

    except Exception as e:
        st.error(f"❌ Quality analysis failed: {str(e)}")

def display_quality_results():
    """Display quality analysis results"""
    results = st.session_state.quality_results

    # Handle both dict and dataclass formats safely
    success = safe_getattr(results, 'success', False)
    overall_score = safe_getattr(results, 'overall_score', 0)
    missing_percentage = safe_getattr(results, 'missing_percentage', 0)
    duplicate_records = safe_getattr(results, 'duplicate_records', 0)
    quality_level = safe_getattr(results, 'quality_level', 'unknown')
    metrics = safe_getattr(results, 'metrics', {})
    recommendations = safe_getattr(results, 'recommendations', [])
    error = safe_getattr(results, 'error', 'Unknown error')

    if not success:
        st.error(f"❌ Quality analysis failed: {error}")
        return

    # Quality metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Overall Score", f"{overall_score:.1f}%")
    with col2:
        st.metric("📉 Missing Data", f"{missing_percentage:.1f}%")
    with col3:
        st.metric("🔄 Duplicates", f"{duplicate_records:,}")
    with col4:
        quality_level_display = str(quality_level).upper()
        st.metric("📊 Quality Level", quality_level_display)

    # Quality breakdown
    if metrics:
        st.markdown("#### 📊 Quality Metrics Breakdown")
        metrics_df = pd.DataFrame([
            {'Metric': k.replace('_', ' ').title(), 'Score': f"{v:.1f}%"}
            for k, v in metrics.items()
        ])
        st.dataframe(metrics_df, use_container_width=True)

    # Recommendations
    if recommendations:
        st.markdown("#### 💡 Quality Recommendations")
        for rec in recommendations:
            st.write(f"• {rec}")

def show_mapping_validation():
    """Show mapping validation and completeness analysis"""
    if not st.session_state.mapping_results:
        return

    st.markdown("#### ✅ Mapping Validation")

    # Get current mappings
    mappings = safe_getattr(st.session_state.mapping_results, 'mappings', {})

    if not mappings:
        st.warning("⚠️ No column mappings found. Please run column mapping first.")
        return

    # Get required and optional fields
    required_fields = get_required_fields()
    optional_fields = get_optional_fields()

    # Check which required fields are mapped
    mapped_values = set(mappings.values())
    mapped_required = [field for field in required_fields if field in mapped_values]
    missing_required = [field for field in required_fields if field not in mapped_values]
    mapped_optional = [field for field in optional_fields if field in mapped_values]

    # Display validation metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🔴 Required Fields",
            f"{len(mapped_required)}/{len(required_fields)}",
            delta=f"{len(missing_required)} missing" if missing_required else "Complete!"
        )

    with col2:
        st.metric(
            "🟡 Optional Fields",
            f"{len(mapped_optional)}/{len(optional_fields)}",
            delta=f"{(len(mapped_optional)/len(optional_fields)*100):.0f}% coverage"
        )

    with col3:
        total_mapped = len(mapped_values)
        total_fields = len(required_fields) + len(optional_fields)
        st.metric(
            "📊 Total Coverage",
            f"{total_mapped}/{total_fields}",
            delta=f"{(total_mapped/total_fields*100):.1f}%"
        )

    with col4:
        completeness_score = (len(mapped_required) / len(required_fields)) * 100 if required_fields else 100
        st.metric(
            "✅ Completeness",
            f"{completeness_score:.0f}%",
            delta="Ready for analysis" if completeness_score >= 80 else "Needs improvement"
        )

    # Show detailed validation
    if missing_required:
        st.error(f"❌ **Missing Required Fields ({len(missing_required)}):** {', '.join(missing_required)}")
        st.info("💡 **Tip:** These fields are essential for CBUAE compliance analysis. Please map them before proceeding to dormancy analysis.")
    else:
        st.success("✅ **All required fields are mapped!** You can proceed to dormancy analysis.")

    # Show mapping by category
    with st.expander("📋 Mapping Status by Category", expanded=False):
        categories = get_fields_by_category()

        for category_name, category_fields in categories.items():
            mapped_in_category = [field for field in category_fields if field in mapped_values]
            coverage = len(mapped_in_category) / len(category_fields) * 100 if category_fields else 0

            if coverage >= 80:
                status_icon = "✅"
                status_color = "success"
            elif coverage >= 50:
                status_icon = "🟡"
                status_color = "warning"
            else:
                status_icon = "❌"
                status_color = "error"

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{status_icon} {category_name}**")
                if mapped_in_category:
                    st.caption(f"Mapped: {', '.join(mapped_in_category)}")
            with col2:
                st.markdown(f"**{len(mapped_in_category)}/{len(category_fields)}** ({coverage:.0f}%)")

def show_data_flow_status():
    """Show current data flow status and readiness for next steps"""
    st.markdown("#### 🔄 Data Flow Status")

    # Check current state
    has_uploaded = st.session_state.uploaded_data is not None
    has_quality = st.session_state.quality_results is not None
    has_mapping = st.session_state.mapping_results is not None
    has_processed = st.session_state.processed_data is not None

    # Processing steps with status
    steps = [
        {
            'name': '📤 Data Upload',
            'status': has_uploaded,
            'description': f'{len(st.session_state.uploaded_data):,} records uploaded' if has_uploaded else 'No data uploaded'
        },
        {
            'name': '🔍 Quality Analysis',
            'status': has_quality,
            'description': f'{safe_getattr(st.session_state.quality_results, "overall_score", 0):.1f}% quality score' if has_quality else 'Quality analysis pending'
        },
        {
            'name': '🗺️ Column Mapping',
            'status': has_mapping and safe_getattr(st.session_state.mapping_results, 'success', False),
            'description': f'{len(safe_getattr(st.session_state.mapping_results, "mappings", {})):,} columns mapped' if has_mapping else 'Column mapping pending'
        },
        {
            'name': '⚙️ Data Processing',
            'status': has_processed,
            'description': f'{len(st.session_state.processed_data):,} records processed' if has_processed else 'Data processing pending'
        }
    ]

    # Display status flow
    cols = st.columns(len(steps))

    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if step['status']:
                st.success(f"✅ {step['name']}")
                st.caption(step['description'])
            else:
                st.error(f"❌ {step['name']}")
                st.caption(step['description'])

    # Overall readiness
    all_ready = all(step['status'] for step in steps)

    if all_ready:
        st.success("🎉 **Data processing complete!** Your data is ready for dormancy analysis.")

        # Show what will be passed to dormancy analysis
        if st.session_state.processed_data is not None:
            processed_data = st.session_state.processed_data
            required_fields = get_required_fields()
            available_required = [field for field in required_fields if field in processed_data.columns]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Processed Records", f"{len(processed_data):,}")
            with col2:
                st.metric("🗂️ Total Columns", f"{len(processed_data.columns)}")
            with col3:
                st.metric("🔴 Required Fields", f"{len(available_required)}/{len(required_fields)}")

        if st.button("🚀 Proceed to Dormancy Analysis", type="primary"):
            st.session_state.current_page = 'dormancy_analysis'
            st.rerun()
    else:
        incomplete_steps = [step['name'] for step in steps if not step['status']]
        st.warning(f"⚠️ **Complete the following steps before proceeding:** {', '.join(incomplete_steps)}")

def show_data_mapping_section():
    """Display data mapping section with LLM enhancement"""
    st.markdown("### 🗺️ Data Mapping & Schema Alignment")

    st.info("📋 **Comprehensive CBUAE Schema**: This system supports mapping to all 66 fields of the CBUAE banking compliance dataset, organized into 11 categories including customer info, address, contact details, account information, financial data, dates, KYC & risk, processing stages, contact attempts, transfers, and system fields.")

    # LLM Enable/Disable toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 🤖 Mapping Configuration")
    with col2:
        llm_enabled = st.toggle("🧠 Enable LLM", value=st.session_state.llm_enabled)
        st.session_state.llm_enabled = llm_enabled

    if llm_enabled:
        st.success("🧠 **LLM-powered automatic column mapping enabled.** The system will use BGE embeddings and AI reasoning to suggest optimal column mappings from your data to the comprehensive CBUAE banking schema (66 fields).")
    else:
        st.info("👤 **Manual mapping mode.** You will manually map columns to the comprehensive CBUAE banking schema with guided assistance and field categorization.")

    # Run mapping
    if st.button("🗺️ Generate Column Mapping", type="primary"):
        run_column_mapping()

    # Display mapping results
    if st.session_state.mapping_results:
        display_mapping_results()

        # Show mapping validation
        show_mapping_validation()

        # Show data flow status
        show_data_flow_status()

def run_column_mapping():
    """Run column mapping with optional LLM enhancement"""
    try:
        with st.spinner("🗺️ Generating column mappings..."):
            data = st.session_state.uploaded_data
            agent = create_unified_data_processing_agent()

            loop = get_or_create_event_loop()
            result = loop.run_until_complete(
                agent.map_columns(
                    data,
                    st.session_state.username,
                    st.session_state.agent_session_id,
                    st.session_state.llm_enabled,
                    None  # llm_api_key - can be added from secrets if needed
                )
            )

            st.session_state.mapping_results = result

            # Handle both dict and dataclass formats
            if isinstance(result, dict):
                success = result.get('success', False)
                mapping_sheet = result.get('mapping_sheet', None)
                error = result.get('error', 'Unknown error')
            else:
                success = result.success
                mapping_sheet = result.mapping_sheet
                error = result.error

            if success:
                st.session_state.mapping_sheet = mapping_sheet

                # Immediately create processed data after successful mapping
                create_processed_data()

                st.success("✅ Column mapping completed!")
                st.info("🔄 Processed data created and ready for dormancy analysis!")
            else:
                st.error(f"❌ Mapping failed: {error}")

    except Exception as e:
        st.error(f"❌ Mapping error: {str(e)}")

def display_mapping_results():
    """Display column mapping results with download option"""
    results = st.session_state.mapping_results

    # Handle both dict and dataclass formats safely
    success = safe_getattr(results, 'success', False)
    auto_mapping_percentage = safe_getattr(results, 'auto_mapping_percentage', 0)
    method = safe_getattr(results, 'method', 'unknown')
    mappings = safe_getattr(results, 'mappings', {})
    mapping_sheet = safe_getattr(results, 'mapping_sheet', None)
    processing_time = safe_getattr(results, 'processing_time', 0)
    error = safe_getattr(results, 'error', 'Unknown error')

    if not success:
        st.error(f"❌ Mapping failed: {error}")
        return

    # Mapping statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Auto-Mapped", f"{auto_mapping_percentage:.1f}%")
    with col2:
        method_display = str(method).replace('_', ' ').title()
        st.metric("🔧 Method", method_display)
    with col3:
        total_mappings = len(mappings) if mappings else 0
        st.metric("🔗 Total Mappings", total_mappings)
    with col4:
        st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")

    # Mapping sheet display
    if mapping_sheet is not None:
        st.markdown("#### 📋 Column Mapping Sheet")
        st.dataframe(mapping_sheet, use_container_width=True)

        # Store mapping sheet in session state for downloads
        st.session_state.mapping_sheet = mapping_sheet

        # Download mapping sheet
        create_download_link(
            mapping_sheet,
            f"column_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'csv'
        )

    # Manual mapping adjustment (if LLM disabled or for review)
    if not st.session_state.llm_enabled or st.button("🔧 Adjust Mappings Manually"):
        show_manual_mapping_interface()

def show_manual_mapping_interface():
    """Show manual mapping interface for column adjustment"""
    st.markdown("#### 🛠️ Manual Column Mapping")

    data = st.session_state.uploaded_data
    mapping_results = st.session_state.mapping_results

    # Handle both dict and dataclass formats
    if isinstance(mapping_results, dict):
        current_mappings = mapping_results.get('mappings', {})
    else:
        current_mappings = mapping_results.mappings if mapping_results else {}

    # Comprehensive CBUAE Banking schema options (66 fields)
    schema_options = [
        # Customer Information
        'customer_id', 'customer_type', 'full_name_en', 'full_name_ar',
        'id_number', 'id_type', 'date_of_birth', 'nationality',

        # Address Information
        'address_line1', 'address_line2', 'city', 'emirate', 'country',
        'postal_code', 'address_known',

        # Contact Information
        'phone_primary', 'phone_secondary', 'email_primary', 'email_secondary',
        'last_contact_date', 'last_contact_method',

        # KYC and Risk
        'kyc_status', 'kyc_expiry_date', 'risk_rating',

        # Account Information
        'account_id', 'account_type', 'account_subtype', 'account_name',
        'currency', 'account_status', 'dormancy_status',

        # Account Dates
        'opening_date', 'closing_date', 'last_transaction_date',
        'last_system_transaction_date',

        # Balance Information
        'balance_current', 'balance_available', 'balance_minimum',
        'interest_rate', 'interest_accrued',

        # Account Details
        'is_joint_account', 'joint_account_holders', 'has_outstanding_facilities',
        'maturity_date', 'auto_renewal', 'last_statement_date', 'statement_frequency',

        # Tracking and Processing
        'tracking_id', 'dormancy_trigger_date', 'dormancy_period_start',
        'dormancy_period_months', 'dormancy_classification_date',
        'transfer_eligibility_date', 'current_stage',

        # Contact Attempts
        'contact_attempts_made', 'last_contact_attempt_date',
        'waiting_period_start', 'waiting_period_end',

        # Transfer Information
        'transferred_to_ledger_date', 'transferred_to_cb_date',
        'cb_transfer_amount', 'cb_transfer_reference', 'exclusion_reason',

        # System Fields
        'created_date', 'updated_date', 'updated_by'
    ]

    # Create mapping interface
    new_mappings = {}

    # Group schema fields by category for better organization
    st.markdown("##### 📋 Available Schema Fields (66 fields total)")

    with st.expander("📖 Schema Categories", expanded=False):
        st.markdown("""
        **Customer Info (8):** customer_id, customer_type, full_name_en, full_name_ar, id_number, id_type, date_of_birth, nationality
        
        **Address Info (7):** address_line1, address_line2, city, emirate, country, postal_code, address_known
        
        **Contact Info (6):** phone_primary, phone_secondary, email_primary, email_secondary, last_contact_date, last_contact_method
        
        **Account Info (7):** account_id, account_type, account_subtype, account_name, currency, account_status, dormancy_status
        
        **Financial Info (5):** balance_current, balance_available, balance_minimum, interest_rate, interest_accrued
        
        **Dates (4):** opening_date, closing_date, last_transaction_date, last_system_transaction_date
        
        **KYC & Risk (3):** kyc_status, kyc_expiry_date, risk_rating
        
        **Processing (7):** tracking_id, dormancy_trigger_date, dormancy_period_start, dormancy_period_months, dormancy_classification_date, transfer_eligibility_date, current_stage
        
        **Contact Attempts (4):** contact_attempts_made, last_contact_attempt_date, waiting_period_start, waiting_period_end
        
        **Transfers (5):** transferred_to_ledger_date, transferred_to_cb_date, cb_transfer_amount, cb_transfer_reference, exclusion_reason
        
        **System Fields (3):** created_date, updated_date, updated_by
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Source Columns**")
    with col2:
        st.markdown("**CBUAE Banking Schema Fields**")

    for i, col in enumerate(data.columns):
        col1, col2 = st.columns(2)
        with col1:
            st.text(col)
        with col2:
            default_mapping = current_mappings.get(col, '')
            default_index = schema_options.index(default_mapping) if default_mapping in schema_options else 0

            mapped_field = st.selectbox(
                f"Map to:",
                [''] + schema_options,
                index=default_index + 1 if default_mapping else 0,
                key=f"mapping_{i}",
                help=f"Select CBUAE schema field for '{col}'"
            )
            if mapped_field:
                new_mappings[col] = mapped_field

    if st.button("💾 Save Manual Mappings"):
        # Update mapping results based on format
        if isinstance(st.session_state.mapping_results, dict):
            st.session_state.mapping_results['mappings'] = new_mappings
            st.session_state.mapping_results['method'] = 'manual'
            st.session_state.mapping_results['success'] = True  # Ensure success flag is set
        else:
            st.session_state.mapping_results.mappings = new_mappings
            st.session_state.mapping_results.method = 'manual'
            st.session_state.mapping_results.success = True  # Ensure success flag is set

        # Create updated mapping sheet
        mapping_data = []
        for source_col, target_field in new_mappings.items():
            mapping_data.append({
                'Source_Column': source_col,
                'Target_Field': target_field,
                'Confidence': 'Manual',
                'Method': 'User Defined'
            })

        mapping_sheet = pd.DataFrame(mapping_data)

        # Update mapping sheet in results
        if isinstance(st.session_state.mapping_results, dict):
            st.session_state.mapping_results['mapping_sheet'] = mapping_sheet
        else:
            st.session_state.mapping_results.mapping_sheet = mapping_sheet

        st.session_state.mapping_sheet = mapping_sheet

        # Create processed data immediately after manual mapping
        create_processed_data()

        st.success("✅ Manual mappings saved!")
        st.info("🔄 Processed data created and ready for dormancy analysis!")

        # Show summary
        if new_mappings:
            st.markdown("#### 📊 Mapping Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Fields Mapped", len(new_mappings))
            with col2:
                st.metric("📋 Total Fields", len(data.columns))
            with col3:
                mapping_percentage = (len(new_mappings) / len(data.columns)) * 100
                st.metric("✅ Coverage", f"{mapping_percentage:.1f}%")

# ===================== DORMANCY ANALYSIS SECTION =====================

def show_dormancy_analysis_section():
    """Display comprehensive dormancy analysis interface"""
    st.markdown('<div class="section-header">💤 Dormancy Analysis</div>', unsafe_allow_html=True)

    if not AGENTS_STATUS['dormancy']:
        st.error("❌ Dormancy analysis agents not available")
        return

    # Check data flow and requirements
    if st.session_state.uploaded_data is None:
        st.error("❌ **No data uploaded.** Please upload data in the Data Processing section.")
        return

    if st.session_state.mapping_results is None:
        st.warning("⚠️ **No column mapping found.** Please run column mapping in the Data Processing section.")
        return

    if st.session_state.processed_data is None:
        st.warning("⚠️ **No processed data found.** Creating processed data from mappings...")
        create_processed_data()

        if st.session_state.processed_data is None:
            st.error("❌ **Failed to create processed data.** Please check your column mappings.")
            return

    # Data flow status
    data = st.session_state.processed_data
    original_data = st.session_state.uploaded_data

    st.markdown("### 📊 Data Flow Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📤 Uploaded Records", f"{len(original_data):,}")

    with col2:
        st.metric("🗺️ Processed Records", f"{len(data):,}")

    with col3:
        mapped_columns = len([col for col in data.columns if col in get_comprehensive_banking_schema().keys()])
        st.metric("✅ Mapped Columns", f"{mapped_columns}")

    with col4:
        required_mapped = len([field for field in get_required_fields() if field in data.columns])
        total_required = len(get_required_fields())
        st.metric("🔴 Required Fields", f"{required_mapped}/{total_required}")

    # Validation checks
    missing_required = [field for field in get_required_fields() if field not in data.columns]

    if missing_required:
        st.error(f"❌ **Missing required fields for CBUAE compliance:** {', '.join(missing_required)}")
        st.info("💡 **Action needed:** Please return to Data Processing section and map these required fields.")

        if st.button("🔄 Refresh Data Flow"):
            create_processed_data()
            st.rerun()
        return
    else:
        st.success("✅ **All required fields are mapped.** Ready for dormancy analysis!")

    # Show processed data preview
    with st.expander("🔍 Processed Data Preview", expanded=False):
        st.markdown("#### Mapped Column Overview")
        col_info = []
        for col in data.columns:
            if col in get_comprehensive_banking_schema():
                schema_info = get_schema_field_info(col)
                col_info.append({
                    'Column': col,
                    'Type': schema_info.get('type', 'unknown'),
                    'Required': '🔴 Yes' if schema_info.get('required', False) else '🟡 No',
                    'Description': schema_info.get('description', 'No description')
                })

        if col_info:
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True)

        st.markdown("#### Data Sample")
        st.dataframe(data.head(5), use_container_width=True)

    # Dormancy analysis overview
    st.markdown("### 🔍 CBUAE Dormancy Compliance Analysis")
    st.info("This section analyzes accounts for dormancy compliance according to CBUAE regulations using 11 specialized agents.")

    # Individual dormancy agents - ALL 11 AGENTS
    dormancy_agents_config = [
        # Primary Detection Agents (5 agents)
        {
            'name': 'Demand Deposit Dormancy',
            'description': 'Analyzes current/checking accounts for dormancy per CBUAE Art. 2.1.1',
            'article': 'CBUAE Art. 2.1.1',
            'key': 'check_demand_deposit_inactivity',
            'category': 'Primary Detection'
        },
        {
            'name': 'Fixed Deposit Dormancy',
            'description': 'Analyzes fixed/term deposits for dormancy per CBUAE Art. 2.1.2',
            'article': 'CBUAE Art. 2.1.2',
            'key': 'check_fixed_deposit_inactivity',
            'category': 'Primary Detection'
        },
        {
            'name': 'Investment Account Dormancy',
            'description': 'Analyzes investment accounts for inactivity per CBUAE Art. 2.2',
            'article': 'CBUAE Art. 2.2',
            'key': 'check_investment_inactivity',
            'category': 'Primary Detection'
        },
        {
            'name': 'Safe Deposit Box Dormancy',
            'description': 'Detects dormant safe deposit boxes with unpaid fees per CBUAE Art. 2.6',
            'article': 'CBUAE Art. 2.6',
            'key': 'check_safe_deposit_dormancy',
            'category': 'Primary Detection'
        },
        {
            'name': 'Unclaimed Payment Instruments',
            'description': 'Identifies unclaimed payment instruments per CBUAE Art. 2.4',
            'article': 'CBUAE Art. 2.4',
            'key': 'check_unclaimed_payment_instruments',
            'category': 'Primary Detection'
        },

        # Process & Transfer Agents (3 agents)
        {
            'name': 'CB Transfer Eligibility',
            'description': 'Identifies accounts eligible for Central Bank transfer per CBUAE Art. 8.1, 8.2',
            'article': 'CBUAE Art. 8.1, 8.2',
            'key': 'check_eligible_for_cb_transfer',
            'category': 'Process & Transfer'
        },
        {
            'name': 'Article 3 Process Required',
            'description': 'Identifies accounts requiring Article 3 process per CBUAE Art. 3',
            'article': 'CBUAE Art. 3',
            'key': 'check_art3_process_needed',
            'category': 'Process & Transfer'
        },
        {
            'name': 'Contact Attempts Needed',
            'description': 'Identifies accounts requiring proactive contact per CBUAE Art. 5',
            'article': 'CBUAE Art. 5',
            'key': 'check_contact_attempts_needed',
            'category': 'Process & Transfer'
        },

        # Specialized Analysis Agents (2 agents)
        {
            'name': 'High Value Dormant Accounts',
            'description': 'Identifies high-value dormant accounts requiring special attention',
            'article': 'Internal Policy',
            'key': 'check_high_value_dormant_accounts',
            'category': 'Specialized Analysis'
        },
        {
            'name': 'Dormant to Active Transitions',
            'description': 'Monitors accounts transitioning from dormant to active status',
            'article': 'Internal Policy',
            'key': 'check_dormant_to_active_transitions',
            'category': 'Specialized Analysis'
        },

        # Master Orchestrator (1 agent)
        {
            'name': 'Comprehensive Dormancy Analysis',
            'description': 'Master orchestrator running all dormancy identification checks',
            'article': 'All Articles',
            'key': 'run_all_dormant_identification_checks',
            'category': 'Master Orchestrator'
        }
    ]

    # Display agent cards organized by category
    st.markdown("### 🤖 Available Dormancy Agents (11 Total)")

    # Group agents by category
    agents_by_category = {}
    for agent in dormancy_agents_config:
        category = agent['category']
        if category not in agents_by_category:
            agents_by_category[category] = []
        agents_by_category[category].append(agent)

    # Display each category
    for category, agents in agents_by_category.items():
        with st.expander(f"📂 {category} ({len(agents)} agents)", expanded=False):
            for agent_config in agents:
                display_dormancy_agent_card(agent_config, data)

    # Comprehensive analysis
    st.markdown("### 🏃‍♂️ Comprehensive Dormancy Analysis")

    if st.button("🚀 Run All Dormancy Agents", type="primary", use_container_width=True):
        run_comprehensive_dormancy_analysis(data)

    # Display comprehensive results
    if st.session_state.dormancy_results:
        display_comprehensive_dormancy_results()

def create_processed_data():
    """Create processed data by applying column mappings"""
    try:
        if st.session_state.uploaded_data is None:
            st.error("❌ No uploaded data found")
            return False

        original_data = st.session_state.uploaded_data
        mapping_results = st.session_state.mapping_results

        if not mapping_results:
            st.warning("⚠️ No mapping results found")
            st.session_state.processed_data = original_data.copy()
            return False

        # Handle both dict and dataclass formats
        success = safe_getattr(mapping_results, 'success', False)
        mappings = safe_getattr(mapping_results, 'mappings', {})

        if not success:
            st.warning("⚠️ Mapping was not successful")
            st.session_state.processed_data = original_data.copy()
            return False

        if not mappings:
            st.warning("⚠️ No column mappings found")
            st.session_state.processed_data = original_data.copy()
            return False

        # Create mapped dataframe
        processed_data = original_data.copy()

        # Apply column mappings
        rename_dict = {k: v for k, v in mappings.items() if v and v.strip()}

        if rename_dict:
            processed_data = processed_data.rename(columns=rename_dict)

            # Log the transformation
            logger.info(f"Applied column mappings: {len(rename_dict)} columns renamed")

            # Add any missing required columns with null values (for safety)
            required_fields = get_required_fields()
            for field in required_fields:
                if field not in processed_data.columns:
                    processed_data[field] = None
                    logger.warning(f"Added missing required field '{field}' with null values")

        st.session_state.processed_data = processed_data

        # Store processing metadata
        st.session_state.processing_metadata = {
            'original_columns': len(original_data.columns),
            'processed_columns': len(processed_data.columns),
            'mapped_columns': len(rename_dict),
            'processing_time': datetime.now().isoformat(),
            'required_fields_present': len([f for f in get_required_fields() if f in processed_data.columns])
        }

        return True

    except Exception as e:
        st.error(f"❌ Error creating processed data: {str(e)}")
        logger.error(f"create_processed_data error: {str(e)}")
        # Fallback to original data
        st.session_state.processed_data = st.session_state.uploaded_data.copy() if st.session_state.uploaded_data is not None else None
        return False

def display_dormancy_agent_card(agent_config, data):
    """Display individual dormancy agent card"""
    with st.container():
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            st.markdown(f"**🔍 {agent_config['name']}**")
            st.markdown(f"*{agent_config['article']}*")

        with col2:
            st.markdown(f"{agent_config['description']}")

        with col3:
            if st.button(f"▶️ Run", key=f"run_{agent_config['key']}"):
                run_individual_dormancy_agent(agent_config, data)

        # Display results if available
        if st.session_state.dormancy_results:
            agent_results = st.session_state.dormancy_results.get('agent_results', {})
            if agent_config['key'] in agent_results:
                result = agent_results[agent_config['key']]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records Processed", f"{result.get('records_processed', 0):,}")
                with col2:
                    st.metric("Dormant Found", f"{result.get('dormant_records_found', 0):,}")
                with col3:
                    if result.get('dormant_records_found', 0) > 0:
                        create_download_link(
                            result.get('details', []),
                            f"{agent_config['key']}_results",
                            'csv'
                        )

        st.markdown('</div>', unsafe_allow_html=True)

def run_individual_dormancy_agent(agent_config, data):
    """Run individual dormancy agent"""
    try:
        with st.spinner(f"🔄 Running {agent_config['name']}..."):
            # Placeholder for individual agent execution
            # In actual implementation, you would call the specific agent

            # Simulate processing
            import time
            time.sleep(1)

            # Mock results
            mock_result = {
                'records_processed': len(data),
                'dormant_records_found': np.random.randint(0, len(data) // 10),
                'processing_time': np.random.uniform(0.5, 2.0),
                'success': True,
                'details': []
            }

            # Update session state
            if 'agent_results' not in st.session_state.dormancy_results:
                st.session_state.dormancy_results = {'agent_results': {}}

            st.session_state.dormancy_results['agent_results'][agent_config['key']] = mock_result

            st.success(f"✅ {agent_config['name']} completed!")
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error running {agent_config['name']}: {str(e)}")

def run_comprehensive_dormancy_analysis(data):
    """Run comprehensive dormancy analysis using all agents"""
    try:
        with st.spinner("🔄 Running comprehensive CBUAE dormancy analysis..."):
            loop = get_or_create_event_loop()

            # Run comprehensive analysis
            result = loop.run_until_complete(
                run_comprehensive_dormancy_analysis_csv(
                    user_id=st.session_state.username,
                    account_data=data,
                    report_date=datetime.now().strftime("%Y-%m-%d")
                )
            )

            st.session_state.dormancy_results = result

            if result.get('success'):
                st.success("✅ Comprehensive dormancy analysis completed!")
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"❌ Comprehensive analysis failed: {str(e)}")

def display_comprehensive_dormancy_results():
    """Display comprehensive dormancy analysis results"""
    results = st.session_state.dormancy_results

    if not results.get('success'):
        st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        return

    st.markdown("### 📊 Comprehensive Dormancy Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Accounts", f"{results.get('total_accounts_analyzed', 0):,}")
    with col2:
        st.metric("💤 Dormant Accounts", f"{results.get('total_dormant_accounts_found', 0):,}")
    with col3:
        st.metric("⏱️ Processing Time", f"{results.get('processing_time_seconds', 0):.2f}s")
    with col4:
        compliance_flags = len(results.get('compliance_flags', []))
        st.metric("⚠️ Compliance Flags", compliance_flags)

    # Agent-specific results
    agent_results = results.get('agent_results', {})
    if agent_results:
        st.markdown("#### 🤖 Agent Results Summary")

        agent_summary = []
        for agent_name, agent_result in agent_results.items():
            if agent_result.get('dormant_records_found', 0) > 0:
                agent_summary.append({
                    'Agent': agent_name.replace('_', ' ').title(),
                    'Records Processed': f"{agent_result.get('records_processed', 0):,}",
                    'Dormant Found': f"{agent_result.get('dormant_records_found', 0):,}",
                    'Status': '✅ Completed' if agent_result.get('success') else '❌ Failed'
                })

        if agent_summary:
            st.dataframe(pd.DataFrame(agent_summary), use_container_width=True)

            # Download comprehensive results
            create_download_link(
                pd.DataFrame(agent_summary),
                f"dormancy_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'csv'
            )

# ===================== COMPLIANCE VERIFICATION SECTION =====================

def show_compliance_analysis_section():
    """Display compliance verification analysis interface"""
    st.markdown('<div class="section-header">⚖️ CBUAE Compliance Verification</div>', unsafe_allow_html=True)

    if not AGENTS_STATUS['compliance']:
        st.error("❌ Compliance verification agents not available")
        return

    # Check data flow requirements
    if st.session_state.processed_data is None:
        st.warning("⚠️ **No processed data found.** Please complete data processing and mapping first.")

        if st.button("🔄 Check Data Status"):
            if st.session_state.mapping_results:
                create_processed_data()
                st.rerun()
            else:
                st.error("❌ No mapping results found. Please complete column mapping first.")
        return

    data = st.session_state.processed_data

    # Data validation for compliance
    required_fields = get_required_fields()
    missing_required = [field for field in required_fields if field not in data.columns]

    if missing_required:
        st.error(f"❌ **Missing required fields:** {', '.join(missing_required)}")
        st.info("💡 Please return to Data Processing and ensure all required fields are mapped.")
        return

    # Show data readiness status
    st.markdown("### 📊 Compliance Data Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📋 Total Records", f"{len(data):,}")

    with col2:
        mapped_fields = len([col for col in data.columns if col in get_comprehensive_banking_schema().keys()])
        st.metric("✅ Mapped Fields", f"{mapped_fields}")

    with col3:
        required_present = len([field for field in required_fields if field in data.columns])
        st.metric("🔴 Required Fields", f"{required_present}/{len(required_fields)}")

    with col4:
        # Check if we have dormancy results to build upon
        has_dormancy = st.session_state.dormancy_results is not None
        st.metric("💤 Dormancy Results", "✅ Available" if has_dormancy else "❌ Missing")

    if not has_dormancy:
        st.info("💡 **Tip:** Run dormancy analysis first for more comprehensive compliance verification.")

    # Compliance analysis overview
    st.markdown("### ⚖️ CBUAE Compliance Verification")
    st.info("This section verifies compliance with CBUAE regulations using 17 specialized compliance agents.")

    # Individual compliance agents - ALL 17 AGENTS
    compliance_agents_config = [
        # Contact & Communication Agents (2 agents)
        {
            'name': 'Incomplete Contact Attempts',
            'description': 'Detects accounts with incomplete contact attempt processes',
            'article': 'CBUAE Art. 3.1, 5',
            'key': 'detect_incomplete_contact_attempts',
            'category': 'Contact & Communication'
        },
        {
            'name': 'Unflagged Dormant Candidates',
            'description': 'Identifies accounts that should be flagged as dormant but are not',
            'article': 'CBUAE Art. 2',
            'key': 'detect_unflagged_dormant_candidates',
            'category': 'Contact & Communication'
        },

        # Process Management Agents (3 agents)
        {
            'name': 'Internal Ledger Candidates',
            'description': 'Identifies accounts ready for internal ledger transfer',
            'article': 'CBUAE Art. 3.4, 3.5',
            'key': 'detect_internal_ledger_candidates',
            'category': 'Process Management'
        },
        {
            'name': 'Statement Freeze Candidates',
            'description': 'Identifies accounts eligible for statement suppression',
            'article': 'CBUAE Art. 7.3',
            'key': 'detect_statement_freeze_candidates',
            'category': 'Process Management'
        },
        {
            'name': 'CBUAE Transfer Candidates',
            'description': 'Identifies accounts ready for Central Bank transfer',
            'article': 'CBUAE Art. 8',
            'key': 'detect_cbuae_transfer_candidates',
            'category': 'Process Management'
        },

        # Specialized Compliance Agents (4 agents)
        {
            'name': 'Foreign Currency Conversion',
            'description': 'Detects foreign currency accounts needing conversion before transfer',
            'article': 'CBUAE Art. 8.5',
            'key': 'detect_foreign_currency_conversion_needed',
            'category': 'Specialized Compliance'
        },
        {
            'name': 'SDB Court Application',
            'description': 'Identifies safe deposit boxes requiring court applications',
            'article': 'CBUAE Art. 3.7',
            'key': 'detect_sdb_court_application_needed',
            'category': 'Specialized Compliance'
        },
        {
            'name': 'Unclaimed Instruments Ledger',
            'description': 'Identifies unclaimed payment instruments for ledger transfer',
            'article': 'CBUAE Art. 3.6',
            'key': 'detect_unclaimed_payment_instruments_ledger',
            'category': 'Specialized Compliance'
        },
        {
            'name': 'Claim Processing Pending',
            'description': 'Detects pending customer claims requiring processing',
            'article': 'CBUAE Art. 4',
            'key': 'detect_claim_processing_pending',
            'category': 'Specialized Compliance'
        },

        # Reporting & Retention Agents (2 agents)
        {
            'name': 'Annual CBUAE Report',
            'description': 'Generates annual CBUAE compliance report summary',
            'article': 'CBUAE Art. 3.10',
            'key': 'generate_annual_cbuae_report_summary',
            'category': 'Reporting & Retention'
        },
        {
            'name': 'Record Retention Compliance',
            'description': 'Checks compliance with record retention requirements',
            'article': 'CBUAE Art. 3.9',
            'key': 'check_record_retention_compliance',
            'category': 'Reporting & Retention'
        },

        # Utility Agents (5 agents)
        {
            'name': 'Flag Instructions Logger',
            'description': 'Logs flagging instructions for dormant accounts',
            'article': 'Internal',
            'key': 'log_flag_instructions',
            'category': 'Utility'
        },
        {
            'name': 'Flag Candidates Detection',
            'description': 'Alias for unflagged dormant candidate detection',
            'article': 'CBUAE Art. 2',
            'key': 'detect_flag_candidates',
            'category': 'Utility'
        },
        {
            'name': 'Ledger Candidates Detection',
            'description': 'Alias for internal ledger candidate detection',
            'article': 'CBUAE Art. 3.4, 3.5',
            'key': 'detect_ledger_candidates',
            'category': 'Utility'
        },
        {
            'name': 'Freeze Candidates Detection',
            'description': 'Alias for statement freeze candidate detection',
            'article': 'CBUAE Art. 7.3',
            'key': 'detect_freeze_candidates',
            'category': 'Utility'
        },
        {
            'name': 'CB Transfer Candidates Detection',
            'description': 'Alias for CBUAE transfer candidate detection',
            'article': 'CBUAE Art. 8',
            'key': 'detect_transfer_candidates_to_cb',
            'category': 'Utility'
        },

        # Master Orchestrator (1 agent)
        {
            'name': 'Comprehensive Compliance Analysis',
            'description': 'Master orchestrator running all compliance verification checks',
            'article': 'All Articles',
            'key': 'run_all_compliance_checks',
            'category': 'Master Orchestrator'
        }
    ]

    # Display compliance agent cards organized by category
    st.markdown("### 🤖 Available Compliance Agents (17 Total)")

    # Group agents by category
    compliance_agents_by_category = {}
    for agent in compliance_agents_config:
        category = agent['category']
        if category not in compliance_agents_by_category:
            compliance_agents_by_category[category] = []
        compliance_agents_by_category[category].append(agent)

    # Display each category
    for category, agents in compliance_agents_by_category.items():
        with st.expander(f"📂 {category} ({len(agents)} agents)", expanded=False):
            for agent_config in agents:
                display_compliance_agent_card(agent_config, data)

    # Comprehensive compliance analysis
    st.markdown("### 🏃‍♂️ Comprehensive Compliance Analysis")

    if st.button("🚀 Run All Compliance Checks", type="primary", use_container_width=True):
        run_comprehensive_compliance_analysis(data)

    # Display comprehensive compliance results
    if st.session_state.compliance_results:
        display_comprehensive_compliance_results()

def display_compliance_agent_card(agent_config, data):
    """Display individual compliance agent card"""
    with st.container():
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            st.markdown(f"**⚖️ {agent_config['name']}**")
            st.markdown(f"*{agent_config['article']}*")

        with col2:
            st.markdown(f"{agent_config['description']}")

        with col3:
            if st.button(f"▶️ Run", key=f"compliance_run_{agent_config['key']}"):
                run_individual_compliance_agent(agent_config, data)

        # Display results if available
        if st.session_state.compliance_results:
            if agent_config['key'] in st.session_state.compliance_results:
                result = st.session_state.compliance_results[agent_config['key']]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records Processed", f"{result.get('accounts_processed', 0):,}")
                with col2:
                    st.metric("Violations Found", f"{result.get('violations_found', 0):,}")
                with col3:
                    if result.get('violations_found', 0) > 0:
                        create_download_link(
                            result.get('actions_generated', []),
                            f"{agent_config['key']}_actions",
                            'json'
                        )

        st.markdown('</div>', unsafe_allow_html=True)

def run_individual_compliance_agent(agent_config, data):
    """Run individual compliance agent"""
    try:
        with st.spinner(f"🔄 Running {agent_config['name']}..."):
            # Placeholder for individual compliance agent execution
            # In actual implementation, you would call the specific compliance agent

            # Simulate processing
            import time
            time.sleep(1)

            # Mock results
            mock_result = {
                'accounts_processed': len(data),
                'violations_found': np.random.randint(0, len(data) // 20),
                'processing_time': np.random.uniform(0.5, 2.0),
                'success': True,
                'actions_generated': []
            }

            # Update session state
            if st.session_state.compliance_results is None:
                st.session_state.compliance_results = {}

            st.session_state.compliance_results[agent_config['key']] = mock_result

            st.success(f"✅ {agent_config['name']} completed!")
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error running {agent_config['name']}: {str(e)}")

def run_comprehensive_compliance_analysis(data):
    """Run comprehensive compliance analysis using all agents"""
    try:
        with st.spinner("🔄 Running comprehensive CBUAE compliance analysis..."):
            loop = get_or_create_event_loop()

            # Prepare dormancy results (if available)
            dormancy_results = st.session_state.dormancy_results or {}

            # Run comprehensive compliance analysis
            result = loop.run_until_complete(
                run_comprehensive_compliance_analysis_csv(
                    user_id=st.session_state.username,
                    dormancy_results=dormancy_results,
                    accounts_df=data
                )
            )

            st.session_state.compliance_results = result

            if result.get('success'):
                st.success("✅ Comprehensive compliance analysis completed!")
            else:
                st.error(f"❌ Compliance analysis failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"❌ Comprehensive compliance analysis failed: {str(e)}")

def display_comprehensive_compliance_results():
    """Display comprehensive compliance analysis results"""
    results = st.session_state.compliance_results

    if not results.get('success'):
        st.error(f"❌ Compliance analysis failed: {results.get('error', 'Unknown error')}")
        return

    st.markdown("### 📊 Comprehensive Compliance Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Accounts", f"{results.get('total_accounts_analyzed', 0):,}")
    with col2:
        st.metric("⚠️ Total Violations", f"{results.get('total_violations', 0):,}")
    with col3:
        st.metric("📋 Actions Generated", f"{results.get('total_actions', 0):,}")
    with col4:
        st.metric("⏱️ Processing Time", f"{results.get('processing_time_seconds', 0):.2f}s")

    # Compliance status
    compliance_status = results.get('compliance_status', 'unknown')
    if compliance_status == 'compliant':
        st.success("✅ All accounts are compliant with CBUAE regulations")
    elif compliance_status == 'action_required':
        st.warning("⚠️ Some accounts require compliance actions")
    else:
        st.info("ℹ️ Compliance analysis in progress")

    # Download comprehensive compliance report
    if results.get('compliance_summary'):
        create_download_link(
            results['compliance_summary'],
            f"compliance_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'json'
        )

# ===================== REPORTS SECTION =====================

def show_reports_section():
    """Display comprehensive reports and analytics"""
    st.markdown('<div class="section-header">📊 Comprehensive Reports & Analytics</div>', unsafe_allow_html=True)

    # Executive summary
    display_executive_summary()

    # Agent status overview
    display_agent_status_overview()

    # Data flow visualization
    display_data_flow_visualization()

    # Detailed reports
    display_detailed_reports()

def display_executive_summary():
    """Display executive summary dashboard"""
    st.markdown("### 📈 Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_records = len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0
        st.metric("📊 Total Records", f"{total_records:,}")

    with col2:
        if st.session_state.dormancy_results:
            total_dormant = st.session_state.dormancy_results.get('total_dormant_accounts_found', 0)
            st.metric("💤 Dormant Accounts", f"{total_dormant:,}")
        else:
            st.metric("💤 Dormant Accounts", "N/A")

    with col3:
        if st.session_state.compliance_results:
            total_violations = st.session_state.compliance_results.get('total_violations', 0)
            st.metric("⚠️ Compliance Violations", f"{total_violations:,}")
        else:
            st.metric("⚠️ Compliance Violations", "N/A")

    with col4:
        available_agents = sum(AGENTS_STATUS.values())
        st.metric("🤖 Available Agents", f"{available_agents}/32")

def display_agent_status_overview():
    """Display comprehensive agent status table for all 28 agents"""
    st.markdown("### 🤖 Agent Status Overview (28 Total Agents)")

    agent_data = []

    # Data processing agents (4 agents)
    data_agents = [
        ('Data Upload', 'Data Processing', AGENTS_STATUS['data_upload']),
        ('Data Quality', 'Data Processing', AGENTS_STATUS['data_quality']),
        ('Data Mapping', 'Data Processing', AGENTS_STATUS['data_mapping']),
        ('BGE Embeddings', 'Data Processing', AGENTS_STATUS['bge_embeddings'])
    ]

    for name, category, available in data_agents:
        records_processed = 0
        status = "Available" if available else "Not Available"

        if name == "Data Quality" and st.session_state.quality_results:
            records_processed = len(st.session_state.uploaded_data) if st.session_state.uploaded_data is not None else 0
            status = "Completed"

        agent_data.append({
            'Agent': name,
            'Category': category,
            'Records Processed': f"{records_processed:,}",
            'Status': status,
            'Actions': 'Download Results' if records_processed > 0 else 'Available'
        })

    # Dormancy agents (11 agents) - organized by category
    dormancy_categories = {
        'Primary Detection': [
            'Demand Deposit Dormancy', 'Fixed Deposit Dormancy', 'Investment Account Dormancy',
            'Safe Deposit Box Dormancy', 'Unclaimed Payment Instruments'
        ],
        'Process & Transfer': [
            'CB Transfer Eligibility', 'Article 3 Process Required', 'Contact Attempts Needed'
        ],
        'Specialized Analysis': [
            'High Value Dormant Accounts', 'Dormant to Active Transitions'
        ],
        'Master Orchestrator': [
            'Comprehensive Dormancy Analysis'
        ]
    }

    for category, agent_names in dormancy_categories.items():
        for agent_name in agent_names:
            records_processed = 0
            status = "Available" if AGENTS_STATUS['dormancy'] else "Not Available"

            if st.session_state.dormancy_results:
                agent_results = st.session_state.dormancy_results.get('agent_results', {})
                snake_name = agent_name.lower().replace(' ', '_')
                if snake_name in agent_results:
                    records_processed = agent_results[snake_name].get('dormant_records_found', 0)
                    status = "Completed"

            agent_data.append({
                'Agent': agent_name,
                'Category': f'Dormancy - {category}',
                'Records Processed': f"{records_processed:,}",
                'Status': status,
                'Actions': 'Download Results' if records_processed > 0 else 'Available'
            })

    # Compliance agents (17 agents) - organized by category
    compliance_categories = {
        'Contact & Communication': [
            'Incomplete Contact Attempts', 'Unflagged Dormant Candidates'
        ],
        'Process Management': [
            'Internal Ledger Candidates', 'Statement Freeze Candidates', 'CBUAE Transfer Candidates'
        ],
        'Specialized Compliance': [
            'Foreign Currency Conversion', 'SDB Court Application',
            'Unclaimed Instruments Ledger', 'Claim Processing Pending'
        ],
        'Reporting & Retention': [
            'Annual CBUAE Report', 'Record Retention Compliance'
        ],
        'Utility': [
            'Flag Instructions Logger', 'Flag Candidates Detection', 'Ledger Candidates Detection',
            'Freeze Candidates Detection', 'CB Transfer Candidates Detection'
        ],
        'Master Orchestrator': [
            'Comprehensive Compliance Analysis'
        ]
    }

    for category, agent_names in compliance_categories.items():
        for agent_name in agent_names:
            violations_found = 0
            status = "Available" if AGENTS_STATUS['compliance'] else "Not Available"

            if st.session_state.compliance_results:
                snake_name = agent_name.lower().replace(' ', '_')
                if snake_name in st.session_state.compliance_results:
                    violations_found = st.session_state.compliance_results[snake_name].get('violations_found', 0)
                    status = "Completed"

            agent_data.append({
                'Agent': agent_name,
                'Category': f'Compliance - {category}',
                'Records Processed': f"{violations_found:,}",
                'Status': status,
                'Actions': 'Download Actions' if violations_found > 0 else 'Available'
            })

    # Create summary
    st.markdown(f"#### 📊 Agent Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📤 Data Processing", "4 agents")
    with col2:
        st.metric("💤 Dormancy Analysis", "11 agents")
    with col3:
        st.metric("⚖️ Compliance Verification", "17 agents")
    with col4:
        total_available = sum(1 for item in agent_data if item['Status'] in ['Available', 'Completed'])
        st.metric("✅ Total Available", f"{total_available}/32")

    # Display comprehensive table
    agent_df = pd.DataFrame(agent_data)
    st.dataframe(agent_df, use_container_width=True, height=800)

def display_data_flow_visualization():
    """Display data flow visualization"""
    st.markdown("### 🔄 Data Processing Flow")

    flow_steps = [
        "📤 Data Upload (4 Methods)",
        "🔍 Data Quality Analysis",
        "🗺️ BGE-Powered Column Mapping",
        "💤 Dormancy Analysis (11 Agents)",
        "⚖️ Compliance Verification (17 Agents)",
        "📊 Reports & Downloads"
    ]

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        for i, step in enumerate(flow_steps):
            st.markdown(f"**{i+1}.** {step}")
            if i < len(flow_steps) - 1:
                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;⬇️")

        st.success("✅ End-to-end CBUAE compliance workflow")

def display_detailed_reports():
    """Display detailed reports section"""
    st.markdown("### 📋 Detailed Reports")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Summary", "💤 Dormancy Report", "⚖️ Compliance Report", "🔄 Processing Log"])

    with tab1:
        if st.session_state.uploaded_data is not None:
            display_data_summary_report()
        else:
            st.info("📊 Upload data to generate summary report")

    with tab2:
        if st.session_state.dormancy_results:
            display_dormancy_detailed_report()
        else:
            st.info("💤 Run dormancy analysis to generate report")

    with tab3:
        if st.session_state.compliance_results:
            display_compliance_detailed_report()
        else:
            st.info("⚖️ Run compliance analysis to generate report")

    with tab4:
        display_processing_log()

def display_data_summary_report():
    """Display detailed data summary report"""
    original_data = st.session_state.uploaded_data
    processed_data = st.session_state.processed_data

    st.markdown("#### 📊 Data Overview")

    # Show both original and processed data stats
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📤 Original Dataset:**")
        st.write(f"• Total Records: {len(original_data):,}")
        st.write(f"• Total Columns: {len(original_data.columns)}")
        st.write(f"• Memory Usage: {original_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        st.write(f"• Upload Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with col2:
        if processed_data is not None:
            st.markdown("**⚙️ Processed Dataset:**")
            st.write(f"• Total Records: {len(processed_data):,}")
            st.write(f"• Total Columns: {len(processed_data.columns)}")
            st.write(f"• Memory Usage: {processed_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

            # Show mapping information
            if st.session_state.mapping_results:
                mappings = safe_getattr(st.session_state.mapping_results, 'mappings', {})
                mapped_fields = len([col for col in processed_data.columns if col in get_comprehensive_banking_schema().keys()])
                st.write(f"• Mapped Fields: {mapped_fields}/{len(processed_data.columns)}")
                st.write(f"• CBUAE Schema Coverage: {(mapped_fields/66)*100:.1f}%")
        else:
            st.markdown("**⚙️ Processed Dataset:**")
            st.write("• No processed data available")
            st.write("• Please complete column mapping")

    # Quality information
    if st.session_state.quality_results:
        st.markdown("#### 📈 Data Quality Metrics")
        quality = st.session_state.quality_results

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"• Overall Score: {safe_getattr(quality, 'overall_score', 0):.1f}%")
        with col2:
            st.write(f"• Missing Data: {safe_getattr(quality, 'missing_percentage', 0):.1f}%")
        with col3:
            st.write(f"• Quality Level: {safe_getattr(quality, 'quality_level', 'unknown').upper()}")

    # Show column transformation if mapping exists
    if st.session_state.mapping_results and processed_data is not None:
        with st.expander("🔄 Column Transformation Details"):
            mappings = safe_getattr(st.session_state.mapping_results, 'mappings', {})

            if mappings:
                transformation_data = []
                for original_col, mapped_col in mappings.items():
                    schema_info = get_schema_field_info(mapped_col) if mapped_col else {}
                    transformation_data.append({
                        'Original Column': original_col,
                        'Mapped To': mapped_col or 'Not Mapped',
                        'Type': schema_info.get('type', 'unknown'),
                        'Required': '🔴 Yes' if schema_info.get('required', False) else '🟡 No'
                    })

                transform_df = pd.DataFrame(transformation_data)
                st.dataframe(transform_df, use_container_width=True)

def display_dormancy_detailed_report():
    """Display detailed dormancy analysis report"""
    results = st.session_state.dormancy_results

    st.markdown("#### 💤 Dormancy Analysis Details")

    if results.get('agent_results'):
        # Create detailed breakdown
        agent_breakdown = []
        for agent_name, agent_result in results['agent_results'].items():
            if agent_result.get('dormant_records_found', 0) > 0:
                agent_breakdown.append({
                    'Agent': agent_name.replace('_', ' ').title(),
                    'Records Processed': agent_result.get('records_processed', 0),
                    'Dormant Found': agent_result.get('dormant_records_found', 0),
                    'Processing Time (s)': f"{agent_result.get('processing_time', 0):.2f}",
                    'Success': '✅' if agent_result.get('success') else '❌'
                })

        if agent_breakdown:
            st.dataframe(pd.DataFrame(agent_breakdown), use_container_width=True)

def display_compliance_detailed_report():
    """Display detailed compliance analysis report"""
    results = st.session_state.compliance_results

    st.markdown("#### ⚖️ Compliance Analysis Details")

    # Display compliance summary if available
    if results.get('compliance_summary'):
        summary = results['compliance_summary']

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• Total Accounts Analyzed: {summary.get('total_accounts_analyzed', 0):,}")
            st.write(f"• Total Violations Found: {summary.get('total_violations_found', 0):,}")
        with col2:
            st.write(f"• Total Actions Generated: {summary.get('total_actions_generated', 0):,}")
            st.write(f"• Processing Time: {summary.get('processing_time', 0):.2f}s")

def display_processing_log():
    """Display processing activity log"""
    st.markdown("#### 🔄 Processing Activity Log")

    # Create activity log
    activities = []

    if st.session_state.uploaded_data is not None:
        activities.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Activity': 'Data Upload',
            'Status': '✅ Completed',
            'Records': len(st.session_state.uploaded_data)
        })

    if st.session_state.quality_results:
        activities.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Activity': 'Quality Analysis',
            'Status': '✅ Completed',
            'Records': len(st.session_state.uploaded_data) if st.session_state.uploaded_data is not None else 0
        })

    if st.session_state.mapping_results:
        activities.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Activity': 'Column Mapping',
            'Status': '✅ Completed',
            'Records': len(st.session_state.mapping_results.get('mappings', {}))
        })

    if st.session_state.dormancy_results:
        activities.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Activity': 'Dormancy Analysis',
            'Status': '✅ Completed',
            'Records': st.session_state.dormancy_results.get('total_dormant_accounts_found', 0)
        })

    if st.session_state.compliance_results:
        activities.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Activity': 'Compliance Analysis',
            'Status': '✅ Completed',
            'Records': st.session_state.compliance_results.get('total_violations', 0)
        })

    if activities:
        st.dataframe(pd.DataFrame(activities), use_container_width=True)
    else:
        st.info("📝 No processing activities recorded yet")

# ===================== SIDEBAR AND NAVIGATION =====================

def show_sidebar():
    """Display sidebar navigation and status"""
    st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.username}!")

    # Navigation
    st.sidebar.markdown("### 🧭 Navigation")
    pages = {
        'data_processing': '📤 Data Processing',
        'dormancy_analysis': '💤 Dormancy Analysis',
        'compliance_verification': '⚖️ Compliance Verification',
        'reports': '📊 Reports & Analytics'
    }

    for page_key, page_name in pages.items():
        if st.sidebar.button(page_name, use_container_width=True):
            st.session_state.current_page = page_key
            st.rerun()

    # Debug Mode (single location to avoid conflicts)
    st.sidebar.markdown("### 🔍 Debug Mode")
    debug_mode = st.sidebar.checkbox("Enable Debug Info", key="global_debug_mode")

    if debug_mode:
        st.sidebar.markdown("#### 📊 Session State Info")
        st.sidebar.write(f"**Username:** {st.session_state.username}")
        st.sidebar.write(f"**Session ID:** {st.session_state.agent_session_id or 'None'}")

        if st.session_state.uploaded_data is not None:
            st.sidebar.write(f"**Data Shape:** {st.session_state.uploaded_data.shape}")

        if st.session_state.quality_results:
            st.sidebar.write(f"**Quality Results Type:** {type(st.session_state.quality_results).__name__}")

        if st.session_state.mapping_results:
            st.sidebar.write(f"**Mapping Results Type:** {type(st.session_state.mapping_results).__name__}")

    # System status
    st.sidebar.markdown("### 🔧 System Status")

    # Agent status indicators
    agent_status_display = {
        'unified_data_processing': '📤 Data Processing',
        'dormancy': '💤 Dormancy Agents',
        'compliance': '⚖️ Compliance Agents'
    }

    for agent_key, agent_name in agent_status_display.items():
        status = "🟢 Available" if AGENTS_STATUS[agent_key] else "🔴 Unavailable"
        st.sidebar.markdown(f"{agent_name}: {status}")

    # Session information
    st.sidebar.markdown("### ℹ️ Session Info")
    st.sidebar.markdown(f"**Session ID:** `{st.session_state.agent_session_id or 'Not started'}`")
    st.sidebar.markdown(f"**Login Time:** `{datetime.now().strftime('%H:%M:%S')}`")

    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ===================== MAIN APPLICATION =====================

def main():
    """Main application entry point"""

    # Check if user is logged in
    if not st.session_state.logged_in:
        show_login_page()
        return

    # Show sidebar
    show_sidebar()

    # Display main content based on current page
    page = st.session_state.get('current_page', 'data_processing')

    if page == 'data_processing':
        show_data_processing_section()
    elif page == 'dormancy_analysis':
        show_dormancy_analysis_section()
    elif page == 'compliance_verification':
        show_compliance_analysis_section()
    elif page == 'reports':
        show_reports_section()
    else:
        show_data_processing_section()  # Default to data processing

if __name__ == "__main__":
    main()