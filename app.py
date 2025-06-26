"""
CBUAE Banking Compliance Analysis System - Clean Streamlit Application
Fixed version with proper syntax and function definitions
Features:
- Real agent integration (no async functions)
- Enhanced data mapping with LLM toggle
- Manual column mapping interface
- Complete data flow through all agents
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime, timedelta
import secrets
import io
import zipfile
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple
import base64
import sys
import os
import asyncio
import concurrent.futures

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CBUAE Banking Compliance System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1f4e79 0%, #2d5aa0 50%, #1f4e79 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(31, 78, 121, 0.3);
    }
    .section-header {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 1.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .agent-status {
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .agent-success { 
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%); 
        color: #155724; 
        border-left-color: #28a745;
    }
    .agent-warning { 
        background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%); 
        color: #856404;
        border-left-color: #ffc107;
    }
    .agent-error { 
        background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%); 
        color: #721c24;
        border-left-color: #dc3545;
    }
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    .data-upload-zone {
        border: 2px dashed #1f4e79;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    .mapping-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .llm-toggle {
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .quality-excellent { background: #d1f2eb; color: #0e5e3b; }
    .quality-good { background: #d4edda; color: #155724; }
    .quality-fair { background: #fff3cd; color: #856404; }
    .quality-poor { background: #f8d7da; color: #721c24; }
    .confidence-high { background: #d4edda; color: #155724; }
    .confidence-medium { background: #fff3cd; color: #856404; }
    .confidence-low { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Import real agents with proper error handling
try:
    from login import SecureLoginManager
    LOGIN_AVAILABLE = True
    logger.info("✅ Login system imported successfully")
except ImportError as e:
    LOGIN_AVAILABLE = False
    logger.warning(f"⚠️ Login system not available: {e}")

# Import REAL data processing agents - CRITICAL
try:
    from agents.Data_Process import (
        UnifiedDataProcessingAgent,
        create_unified_data_processing_agent,
        UploadResult,
        QualityResult,
        MappingResult,
        ProcessingStatus,
        DataQualityLevel,
        MappingConfidence
    )
    DATA_PROCESSING_AVAILABLE = True
    logger.info("✅ Real Data Processing agents imported successfully")
except ImportError as e:
    DATA_PROCESSING_AVAILABLE = False
    logger.error(f"❌ CRITICAL: Real Data Processing agents not available: {e}")

# Import memory agent
try:
    from agents.memory_agent_streamlit_fix import (
        create_streamlit_memory_agent,
        MemoryContext,
        MemoryBucket,
        StreamlitMemoryAgent
    )
    MEMORY_AGENT_AVAILABLE = True
    logger.info("✅ Memory agent imported successfully")
except ImportError as e:
    try:
        from agents.memory_agent import HybridMemoryAgent, MemoryContext, MemoryBucket

        def create_streamlit_memory_agent(config):
            return HybridMemoryAgent({})

        MEMORY_AGENT_AVAILABLE = True
        logger.info("✅ Original memory agent imported successfully")
    except ImportError as e2:
        MEMORY_AGENT_AVAILABLE = False
        logger.warning(f"⚠️ Memory agent not available: {e}")

        def create_streamlit_memory_agent(config):
            return {"session_data": {}, "type": "dummy_fallback"}

        MemoryContext = None
        MemoryBucket = None

# Import dormancy agents
try:
    from agents.Dormant_agent import (
        DormancyWorkflowOrchestrator,
        run_comprehensive_dormancy_analysis_with_csv,
        run_individual_agent_analysis
    )
    DORMANCY_AGENTS_AVAILABLE = True
    logger.info("✅ Dormancy agents imported successfully")
except ImportError as e:
    DORMANCY_AGENTS_AVAILABLE = False
    logger.error(f"❌ Dormancy agents not available: {e}")

# Import compliance agents
try:
    from agents.compliance_verification_agent import (
        ComplianceWorkflowOrchestrator,
        run_comprehensive_compliance_analysis_with_csv,
        get_all_compliance_agents_info
    )
    COMPLIANCE_AGENTS_AVAILABLE = True
    logger.info("✅ Compliance agents imported successfully")
except ImportError as e:
    COMPLIANCE_AGENTS_AVAILABLE = False
    logger.error(f"❌ Compliance agents not available: {e}")

# CBUAE Banking Schema - Complete Updated Schema (65 Fields)
CBUAE_BANKING_SCHEMA = {
    # Customer Information
    'customer_id': ['customer_id', 'cust_id', 'customer_number', 'client_id'],
    'customer_type': ['customer_type', 'client_type', 'customer_category', 'entity_type'],
    'full_name_en': ['full_name_en', 'name_english', 'customer_name_en', 'client_name_en'],
    'full_name_ar': ['full_name_ar', 'name_arabic', 'customer_name_ar', 'client_name_ar'],
    'id_number': ['id_number', 'national_id', 'emirates_id', 'identification_number'],
    'id_type': ['id_type', 'identification_type', 'id_document_type', 'document_type'],
    'date_of_birth': ['date_of_birth', 'birth_date', 'dob', 'birthdate'],
    'nationality': ['nationality', 'country_of_birth', 'citizenship', 'national_origin'],

    # Address Information
    'address_line1': ['address_line1', 'address1', 'street_address', 'primary_address'],
    'address_line2': ['address_line2', 'address2', 'secondary_address', 'apartment'],
    'city': ['city', 'town', 'municipality', 'locality'],
    'emirate': ['emirate', 'state', 'province', 'region'],
    'country': ['country', 'nation', 'country_code', 'country_name'],
    'postal_code': ['postal_code', 'zip_code', 'postcode', 'zip'],

    # Contact Information
    'phone_primary': ['phone_primary', 'primary_phone', 'main_phone', 'mobile_number'],
    'phone_secondary': ['phone_secondary', 'secondary_phone', 'alternate_phone', 'home_phone'],
    'email_primary': ['email_primary', 'primary_email', 'main_email', 'email_address'],
    'email_secondary': ['email_secondary', 'secondary_email', 'alternate_email', 'backup_email'],
    'address_known': ['address_known', 'address_verified', 'valid_address', 'address_status'],
    'last_contact_date': ['last_contact_date', 'last_contacted', 'contact_date', 'last_communication'],
    'last_contact_method': ['last_contact_method', 'contact_method', 'communication_method', 'contact_type'],

    # KYC and Risk Information
    'kyc_status': ['kyc_status', 'kyc', 'know_your_customer', 'compliance_status'],
    'kyc_expiry_date': ['kyc_expiry_date', 'kyc_expiration', 'kyc_renewal_date', 'compliance_expiry'],
    'risk_rating': ['risk_rating', 'risk_level', 'risk_category', 'risk_score'],

    # Account Information
    'account_id': ['account_id', 'account_number', 'acc_id', 'acc_no'],
    'account_type': ['account_type', 'acc_type', 'product_type', 'account_category'],
    'account_subtype': ['account_subtype', 'account_sub_category', 'product_subtype', 'acc_subtype'],
    'account_name': ['account_name', 'account_title', 'acc_name', 'product_name'],
    'currency': ['currency', 'ccy', 'currency_code', 'curr'],
    'account_status': ['account_status', 'status', 'acc_status', 'state'],
    'dormancy_status': ['dormancy_status', 'dormant_status', 'activity_status', 'dormancy_flag'],

    # Date Information
    'opening_date': ['opening_date', 'open_date', 'account_open_date', 'start_date'],
    'closing_date': ['closing_date', 'close_date', 'account_close_date', 'end_date'],
    'last_transaction_date': ['last_transaction_date', 'last_txn_date', 'latest_transaction', 'last_activity'],
    'last_system_transaction_date': ['last_system_transaction_date', 'last_sys_txn', 'system_last_transaction', 'last_automated_txn'],

    # Balance Information
    'balance_current': ['balance_current', 'current_balance', 'available_balance', 'balance'],
    'balance_available': ['balance_available', 'usable_balance', 'accessible_balance', 'liquid_balance'],
    'balance_minimum': ['balance_minimum', 'min_balance', 'minimum_required', 'min_required_balance'],
    'interest_rate': ['interest_rate', 'rate', 'annual_rate', 'interest_percentage'],
    'interest_accrued': ['interest_accrued', 'accrued_interest', 'earned_interest', 'interest_earned'],

    # Account Features
    'is_joint_account': ['is_joint_account', 'joint_account', 'shared_account', 'multiple_holders'],
    'joint_account_holders': ['joint_account_holders', 'co_holders', 'additional_holders', 'joint_holders'],
    'has_outstanding_facilities': ['has_outstanding_facilities', 'outstanding_loans', 'active_facilities', 'loan_facilities'],
    'maturity_date': ['maturity_date', 'expiry_date', 'term_end_date', 'maturation_date'],
    'auto_renewal': ['auto_renewal', 'automatic_renewal', 'auto_rollover', 'renewal_flag'],
    'last_statement_date': ['last_statement_date', 'statement_date', 'last_stmt_date', 'recent_statement'],
    'statement_frequency': ['statement_frequency', 'stmt_frequency', 'statement_cycle', 'reporting_frequency'],

    # Dormancy Tracking
    'tracking_id': ['tracking_id', 'dormancy_tracking_id', 'reference_id', 'tracking_reference'],
    'dormancy_trigger_date': ['dormancy_trigger_date', 'dormancy_start', 'inactive_since', 'dormancy_began'],
    'dormancy_period_start': ['dormancy_period_start', 'dormant_period_begin', 'inactivity_start', 'dormancy_commencement'],
    'dormancy_period_months': ['dormancy_period_months', 'months_dormant', 'dormancy_duration', 'inactive_months'],
    'dormancy_classification_date': ['dormancy_classification_date', 'classified_dormant_date', 'dormancy_confirmed_date', 'classification_date'],
    'transfer_eligibility_date': ['transfer_eligibility_date', 'eligible_for_transfer', 'transfer_ready_date', 'transfer_due_date'],
    'current_stage': ['current_stage', 'processing_stage', 'workflow_stage', 'status_stage'],
    'contact_attempts_made': ['contact_attempts_made', 'contact_tries', 'outreach_attempts', 'communication_attempts'],
    'last_contact_attempt_date': ['last_contact_attempt_date', 'last_outreach_date', 'recent_contact_attempt', 'last_tried_contact'],
    'waiting_period_start': ['waiting_period_start', 'waiting_began', 'hold_period_start', 'waiting_commenced'],
    'waiting_period_end': ['waiting_period_end', 'waiting_expires', 'hold_period_end', 'waiting_concludes'],

    # Transfer Information
    'transferred_to_ledger_date': ['transferred_to_ledger_date', 'ledger_transfer_date', 'moved_to_ledger', 'ledger_entry_date'],
    'transferred_to_cb_date': ['transferred_to_cb_date', 'central_bank_transfer_date', 'cb_handover_date', 'transferred_cb'],
    'cb_transfer_amount': ['cb_transfer_amount', 'central_bank_amount', 'transferred_amount', 'cb_amount'],
    'cb_transfer_reference': ['cb_transfer_reference', 'cb_reference', 'central_bank_ref', 'transfer_reference'],
    'exclusion_reason': ['exclusion_reason', 'exempt_reason', 'exception_reason', 'non_transfer_reason'],

    # Audit Information
    'created_date': ['created_date', 'creation_date', 'record_created', 'date_created'],
    'updated_date': ['updated_date', 'modification_date', 'last_updated', 'date_modified'],
    'updated_by': ['updated_by', 'modified_by', 'last_modified_by', 'updater']
}

# System status check
def print_system_status():
    """Print the current system status"""
    print("\n" + "=" * 60)
    print("🏦 CBUAE Banking Compliance System Status")
    print("=" * 60)
    print(f"🔐 Login System: {'✅ Available' if LOGIN_AVAILABLE else '❌ Not Available'}")
    print(f"📊 Data Processing: {'✅ REAL AGENTS' if DATA_PROCESSING_AVAILABLE else '❌ CRITICAL MISSING'}")
    print(f"💾 Memory Agent: {'✅ Available' if MEMORY_AGENT_AVAILABLE else '❌ Not Available'}")
    print(f"🏃 Dormancy Agents: {'✅ Available' if DORMANCY_AGENTS_AVAILABLE else '❌ Not Available'}")
    print(f"⚖️ Compliance Agents: {'✅ Available' if COMPLIANCE_AGENTS_AVAILABLE else '❌ Not Available'}")
    print("=" * 60)

print_system_status()

# Synchronous wrapper for async agent methods
def run_sync_agent_method(agent_method, *args, **kwargs):
    """Synchronous wrapper for async agent methods - Streamlit compatible"""
    try:
        # Check if there's an existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If there's a running loop, use thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, agent_method(*args, **kwargs))
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(agent_method(*args, **kwargs))
    except Exception as e:
        logger.error(f"Sync wrapper failed: {e}")
        raise e

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = secrets.token_hex(16)
    if 'data_processing_agent' not in st.session_state:
        st.session_state.data_processing_agent = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'quality_results' not in st.session_state:
        st.session_state.quality_results = None
    if 'mapping_results' not in st.session_state:
        st.session_state.mapping_results = None
    if 'manual_mapping' not in st.session_state:
        st.session_state.manual_mapping = {}
    if 'llm_enabled' not in st.session_state:
        st.session_state.llm_enabled = True
    if 'dormancy_results' not in st.session_state:
        st.session_state.dormancy_results = None
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = None
    if 'memory_agent' not in st.session_state:
        st.session_state.memory_agent = None

def initialize_memory_agent():
    """Initialize enhanced memory agent"""
    if st.session_state.memory_agent is None:
        if MEMORY_AGENT_AVAILABLE:
            try:
                # Enhanced configuration
                memory_config = {
                    "db_path": "enhanced_banking_memory.db",
                    "redis_host": "localhost",
                    "redis_port": 6379,
                    "redis_db": 0,
                    "socket_timeout": 5,
                    "cache_ttl": {
                        "session": 3600,
                        "knowledge": 86400,
                        "cache": 1800
                    },
                    "max_cache_size": 1000
                }

                # Create enhanced memory agent
                if hasattr(create_streamlit_memory_agent, '__call__'):
                    st.session_state.memory_agent = create_streamlit_memory_agent(memory_config)
                else:
                    # Fallback to basic memory agent
                    st.session_state.memory_agent = {
                        "session_data": {},
                        "type": "basic_memory"
                    }

                # Show statistics
                if hasattr(st.session_state.memory_agent, 'get_statistics'):
                    stats = st.session_state.memory_agent.get_statistics()
                    logger.info(f"✅ Enhanced Memory Agent initialized - Redis: {stats.get('redis_available', False)}")
                else:
                    logger.info("✅ Memory Agent initialized (basic mode)")

            except Exception as e:
                logger.error(f"❌ Enhanced memory agent failed: {e}")
                # Simple fallback
                st.session_state.memory_agent = {
                    "session_data": {},
                    "type": "simple_fallback"
                }
        else:
            st.session_state.memory_agent = {
                "session_data": {},
                "type": "simple_fallback"
            }

    return st.session_state.memory_agent

def initialize_data_processing_agent():
    """Initialize the real data processing agent"""
    try:
        # Check if session state exists
        if not hasattr(st, 'session_state'):
            logger.error("Streamlit session state not available")
            return False

        # Check if agent already initialized
        if hasattr(st.session_state, 'data_processing_agent') and st.session_state.data_processing_agent is not None:
            logger.info("Data processing agent already initialized")
            return True

        # Check if Data Processing is available
        if not DATA_PROCESSING_AVAILABLE:
            logger.error("DATA_PROCESSING_AVAILABLE is False")
            st.session_state.data_processing_agent = None
            return False

        # Check if the factory function exists
        if 'create_unified_data_processing_agent' not in globals():
            logger.error("create_unified_data_processing_agent function not found in globals")
            st.session_state.data_processing_agent = None
            return False

        agent_config = {
            "enable_memory": MEMORY_AGENT_AVAILABLE,
            "enable_bge": True,  # Enable BGE embeddings for LLM mapping
            "bge_model": "BAAI/bge-large-en-v1.5",
            "similarity_threshold": 0.7,
            "quality_thresholds": {
                "excellent": 0.9,
                "good": 0.7,
                "fair": 0.5,
                "poor": 0.0
            },
            "mapping_thresholds": {
                "high_confidence": 0.8,
                "medium_confidence": 0.6,
                "low_confidence": 0.4
            },
            "banking_schema": CBUAE_BANKING_SCHEMA,  # Pass the updated schema
            "auto_mapping_enabled": True,
            "max_file_size_mb": 100,
            "supported_formats": ["csv", "xlsx", "json", "parquet"]
        }

        logger.info("Creating unified data processing agent...")
        st.session_state.data_processing_agent = create_unified_data_processing_agent(agent_config)

        if st.session_state.data_processing_agent is None:
            logger.error("Agent creation returned None")
            return False

        logger.info("✅ Real Data Processing Agent initialized successfully")
        return True

    except NameError as ne:
        logger.error(f"❌ NameError during agent initialization: {ne}")
        st.session_state.data_processing_agent = None
        return False
    except Exception as e:
        logger.error(f"❌ Data processing agent initialization failed: {e}")
        st.session_state.data_processing_agent = None
        return False

def check_agent_availability():
    """Debug function to check what agents are available"""
    status = {
        "DATA_PROCESSING_AVAILABLE": DATA_PROCESSING_AVAILABLE,
        "MEMORY_AGENT_AVAILABLE": MEMORY_AGENT_AVAILABLE,
        "DORMANCY_AGENTS_AVAILABLE": DORMANCY_AGENTS_AVAILABLE,
        "COMPLIANCE_AGENTS_AVAILABLE": COMPLIANCE_AGENTS_AVAILABLE,
        "create_unified_data_processing_agent_available": 'create_unified_data_processing_agent' in globals(),
        "create_streamlit_memory_agent_available": 'create_streamlit_memory_agent' in globals()
    }
    logger.info(f"Agent availability status: {status}")
    return status

def verify_functions():
    """Verify that all required functions are defined"""
    required_functions = [
        'initialize_session_state',
        'initialize_memory_agent',
        'initialize_data_processing_agent',
        'run_sync_agent_method',
        'check_agent_availability'
    ]

    missing_functions = []
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)

    if missing_functions:
        logger.error(f"Missing functions: {missing_functions}")
        return False
    else:
        logger.info("✅ All required functions are defined")
        return True

def test_function_definitions():
    """Test that functions are properly defined"""
    test_results = {}

    functions_to_test = [
        'initialize_session_state',
        'initialize_memory_agent',
        'initialize_data_processing_agent',
        'run_sync_agent_method'
    ]

    for func_name in functions_to_test:
        try:
            func = globals().get(func_name)
            if func and callable(func):
                test_results[func_name] = "✅ Defined and callable"
            else:
                test_results[func_name] = "❌ Not found or not callable"
        except Exception as e:
            test_results[func_name] = f"❌ Error: {e}"

    return test_results

# Simple login system fallback
class SimpleLoginManager:
    """Simplified login system"""
    def __init__(self):
        self.users = {
            "demo": {"password": "demo123", "role": "analyst"},
            "admin": {"password": "admin123", "role": "admin"},
            "compliance": {"password": "compliance123", "role": "compliance_officer"},
            "analyst": {"password": "analyst123", "role": "analyst"}
        }

    def authenticate_user(self, username: str, password: str) -> Dict:
        """Simple authentication"""
        if username in self.users and self.users[username]["password"] == password:
            return {
                'user_id': hash(username) % 10000,
                'username': username,
                'role': self.users[username]["role"],
                'authenticated': True
            }
        return None

# Login page
def render_login_page():
    """Render secure login interface"""
    st.markdown('<div class="main-header"><h1>🏦 CBUAE Banking Compliance System</h1><p>Secure Login Required</p></div>',
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("🔐 Authentication")

            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            col_login, col_demo = st.columns(2)

            with col_login:
                if st.button("🚀 Login", type="primary", use_container_width=True):
                    if username and password:
                        try:
                            # Try secure login first, fallback to simple login
                            login_manager = None
                            user_data = None

                            if LOGIN_AVAILABLE:
                                try:
                                    login_manager = SecureLoginManager()
                                    user_data = login_manager.authenticate_user(username, password)
                                except Exception as e:
                                    st.warning(f"⚠️ Secure login failed, using simple login: {str(e)}")
                                    login_manager = SimpleLoginManager()
                                    user_data = login_manager.authenticate_user(username, password)
                            else:
                                login_manager = SimpleLoginManager()
                                user_data = login_manager.authenticate_user(username, password)

                            if user_data and user_data.get('authenticated'):
                                st.session_state.authenticated = True
                                st.session_state.user_data = user_data
                                st.success(f"✅ Welcome back, {user_data['username']}!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials")

                        except Exception as e:
                            st.error(f"❌ Authentication failed: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter both username and password")

            with col_demo:
                if st.button("🎯 Demo Login", use_container_width=True):
                    try:
                        login_manager = SimpleLoginManager()
                        user_data = login_manager.authenticate_user("demo", "demo123")

                        if user_data:
                            st.session_state.authenticated = True
                            st.session_state.user_data = user_data
                            st.success("✅ Demo login successful!")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Demo login failed: {str(e)}")

            st.markdown("---")
            with st.expander("📋 Available Credentials"):
                st.code("""
Demo User:     demo / demo123
Admin User:    admin / admin123
Compliance:    compliance / compliance123
Analyst:       analyst / analyst123
                """)

        st.markdown('</div>', unsafe_allow_html=True)

# Main application header
def render_main_header():
    """Render main application header"""
    st.markdown(f'''
    <div class="main-header">
        <h1>🏦 CBUAE Banking Compliance Agentic AI System</h1>
        <p>Intelligent Banking Compliance with Real AI Agents & Advanced Data Mapping</p>
        <p>👤 Welcome, {st.session_state.user_data["username"]} | 🎯 Role: {st.session_state.user_data["role"]} | 🆔 Session: {st.session_state.session_id[:8]}</p>
    </div>
    ''', unsafe_allow_html=True)

# Data processing section
def render_data_processing_section():
    """Enhanced data processing section with real data mapping agent"""
    st.markdown('<div class="section-header"><h2>📊 Data Processing & Advanced Mapping</h2></div>',
                unsafe_allow_html=True)

    # Check if real agent is available
    if not DATA_PROCESSING_AVAILABLE:
        st.error("❌ CRITICAL: Real Data Processing Agent not available. Please check agents/Data_Process.py imports.")
        st.info("💡 Required: UnifiedDataProcessingAgent, MappingResult, QualityResult classes")
        return

    # Initialize agent
    agent_initialized = initialize_data_processing_agent()
    if not agent_initialized:
        st.error("❌ Failed to initialize Data Processing Agent")
        return

    agent = st.session_state.data_processing_agent
    st.success("✅ Real Data Processing Agent Loaded & Ready")

    # Data Upload Section - 3 Methods
    st.subheader("📁 Data Upload Agent (3 Real Methods)")

    upload_method = st.radio(
        "Select Upload Method:",
        ["File Upload", "Database Connection", "API Endpoint"],
        horizontal=True
    )

    uploaded_data = None

    if upload_method == "File Upload":
        st.markdown('<div class="data-upload-zone">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Banking Data",
            type=['csv', 'xlsx', 'json'],
            help="Upload account data for dormancy analysis and compliance verification"
        )

        if uploaded_file:
            try:
                with st.spinner("Processing file with real data upload agent..."):
                    # Save uploaded file temporarily
                    temp_file_path = f"temp_{st.session_state.session_id}_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Use synchronous wrapper for agent upload
                    upload_result = run_sync_agent_method(
                        agent.upload_data,
                        source=temp_file_path,
                        upload_method="file",
                        user_id=st.session_state.user_data.get("username", "unknown"),
                        session_id=st.session_state.session_id
                    )

                    # Clean up temp file
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass

                if upload_result and upload_result.success:
                    uploaded_data = upload_result.data
                    st.session_state.uploaded_data = uploaded_data

                    st.success(f"✅ File processed successfully: {uploaded_file.name}")
                    st.info(f"📊 Data shape: {uploaded_data.shape[0]} rows × {uploaded_data.shape[1]} columns")

                    # Show upload metadata
                    if hasattr(upload_result, 'metadata') and upload_result.metadata:
                        with st.expander("🔍 Upload Details"):
                            st.json(upload_result.metadata)
                else:
                    error_msg = getattr(upload_result, 'error', 'Unknown upload error') if upload_result else 'Upload failed'
                    st.error(f"❌ File upload failed: {error_msg}")

            except Exception as e:
                st.error(f"❌ File upload failed: {str(e)}")
                logger.error(f"Upload error: {e}")

                # Show detailed error for debugging
                with st.expander("🔍 Error Details"):
                    st.code(f"Error Type: {type(e).__name__}")
                    st.code(f"Error Message: {str(e)}")
                    st.markdown("**Possible Solutions:**")
                    st.markdown("1. Check file format (CSV, XLSX, JSON supported)")
                    st.markdown("2. Ensure file is not corrupted or empty")
                    st.markdown("3. Verify agent dependencies are installed")

        st.markdown('</div>', unsafe_allow_html=True)

    elif upload_method == "Database Connection":
        st.info("🔗 Real database connection interface")

        with st.form("database_form"):
            col1, col2 = st.columns(2)
            with col1:
                db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQL Server", "Oracle"])
                host = st.text_input("Host", value="localhost")
                port = st.number_input("Port", value=5432)
            with col2:
                database = st.text_input("Database Name")
                username = st.text_input("DB Username")
                password = st.text_input("DB Password", type="password")

            table_name = st.text_input("Table Name")
            sql_query = st.text_area("Custom SQL Query (Optional)",
                                    placeholder="SELECT * FROM accounts WHERE status = 'active'")

            if st.form_submit_button("🔗 Connect & Load Data", type="primary"):
                if all([host, database, username, password, table_name]):
                    try:
                        with st.spinner("Connecting to database..."):
                            # Database configuration
                            db_config = {
                                "db_type": db_type,
                                "host": host,
                                "port": port,
                                "database": database,
                                "username": username,
                                "password": password,
                                "table_name": table_name,
                                "sql_query": sql_query if sql_query else None
                            }

                            # Use synchronous wrapper for database upload
                            upload_result = run_sync_agent_method(
                                agent.upload_data,
                                source=db_config,
                                upload_method="database",
                                user_id=st.session_state.user_data.get("username", "unknown"),
                                session_id=st.session_state.session_id
                            )

                            if upload_result and upload_result.success:
                                uploaded_data = upload_result.data
                                st.session_state.uploaded_data = uploaded_data
                                st.success(f"✅ Database connection successful! Loaded {len(uploaded_data)} records")
                            else:
                                st.error(f"❌ Database connection failed: {getattr(upload_result, 'error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"❌ Database connection failed: {str(e)}")
                        logger.error(f"Database connection error: {e}")
                else:
                    st.warning("⚠️ Please fill all required fields")

    elif upload_method == "API Endpoint":
        st.info("🌐 Real API endpoint interface")

        with st.form("api_form"):
            api_url = st.text_input("API Endpoint URL", placeholder="https://api.example.com/banking-data")
            auth_method = st.selectbox("Authentication", ["None", "API Key", "Bearer Token", "Basic Auth"])

            auth_config = {}
            if auth_method == "API Key":
                key_name = st.text_input("API Key Header", value="X-API-Key")
                api_key = st.text_input("API Key", type="password")
                auth_config = {"type": "api_key", "header": key_name, "key": api_key}
            elif auth_method == "Bearer Token":
                bearer_token = st.text_input("Bearer Token", type="password")
                auth_config = {"type": "bearer", "token": bearer_token}
            elif auth_method == "Basic Auth":
                api_username = st.text_input("API Username")
                api_password = st.text_input("API Password", type="password")
                auth_config = {"type": "basic", "username": api_username, "password": api_password}

            if st.form_submit_button("🌐 Fetch Data from API", type="primary"):
                if api_url:
                    try:
                        with st.spinner("Fetching data from API..."):
                            api_config = {
                                "url": api_url,
                                "auth": auth_config,
                                "method": "GET"
                            }

                            # Use synchronous wrapper for API upload
                            upload_result = run_sync_agent_method(
                                agent.upload_data,
                                source=api_config,
                                upload_method="api",
                                user_id=st.session_state.user_data.get("username", "unknown"),
                                session_id=st.session_state.session_id
                            )

                            if upload_result and upload_result.success:
                                uploaded_data = upload_result.data
                                st.session_state.uploaded_data = uploaded_data
                                st.success(f"✅ API data fetched successfully! Loaded {len(uploaded_data)} records")
                            else:
                                st.error(f"❌ API fetch failed: {getattr(upload_result, 'error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"❌ API fetch failed: {str(e)}")
                        logger.error(f"API fetch error: {e}")
                else:
                    st.warning("⚠️ Please provide API endpoint URL")

    # Process uploaded data if available
    if st.session_state.uploaded_data is not None:
        uploaded_data = st.session_state.uploaded_data

        # Display data preview
        with st.expander("👀 Data Preview", expanded=True):
            st.dataframe(uploaded_data.head(10), use_container_width=True)

        # Data Quality Analysis
        st.subheader("🔍 Data Quality Analysis")

        if st.button("🚀 Run Data Quality Analysis", type="primary"):
            with st.spinner("Running real data quality analysis..."):
                try:
                    # Use synchronous wrapper for quality analysis
                    quality_result = run_sync_agent_method(
                        agent.analyze_quality,
                        data=uploaded_data,
                        user_id=st.session_state.user_data.get("username", "unknown"),
                        session_id=st.session_state.session_id
                    )

                    if quality_result and quality_result.success:
                        st.session_state.quality_results = quality_result

                        # Display quality metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("📊 Total Records", f"{len(uploaded_data):,}")
                        with col2:
                            missing_pct = getattr(quality_result, 'missing_percentage', 0)
                            st.metric("🔍 Missing Data", f"{missing_pct:.1f}%")
                        with col3:
                            duplicate_count = getattr(quality_result, 'duplicate_records', 0)
                            st.metric("📋 Duplicate Records", f"{duplicate_count:,}")
                        with col4:
                            quality_level = getattr(quality_result, 'quality_level', 'unknown').lower()
                            overall_score = getattr(quality_result, 'overall_score', 0)
                            quality_class = f"quality-{quality_level}"
                            st.markdown(f"""
                            <div class="metric-card {quality_class}">
                                <h3>⭐ Quality Score</h3>
                                <h2>{overall_score:.1%}</h2>
                                <p>{quality_level.title()}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Detailed metrics
                        if hasattr(quality_result, 'metrics') and quality_result.metrics:
                            with st.expander("📊 Detailed Quality Metrics"):
                                metrics_df = pd.DataFrame([quality_result.metrics]).T
                                metrics_df.columns = ['Score']
                                st.dataframe(metrics_df, use_container_width=True)

                        # Recommendations
                        if hasattr(quality_result, 'recommendations') and quality_result.recommendations:
                            st.subheader("💡 Quality Improvement Recommendations")
                            for i, rec in enumerate(quality_result.recommendations, 1):
                                st.markdown(f"{i}. {rec}")

                    else:
                        error_msg = getattr(quality_result, 'error', 'Unknown error') if quality_result else 'Quality analysis failed'
                        st.error(f"❌ Quality analysis failed: {error_msg}")

                except Exception as e:
                    st.error(f"❌ Quality analysis failed: {str(e)}")
                    logger.error(f"Quality analysis error: {e}")

                    # Show detailed error info
                    with st.expander("🔍 Error Details"):
                        st.code(f"Error Type: {type(e).__name__}")
                        st.code(f"Error Message: {str(e)}")
                        st.markdown("**Possible Solutions:**")
                        st.markdown("1. Ensure data is properly formatted")
                        st.markdown("2. Check for data type compatibility")
                        st.markdown("3. Verify agent is properly initialized")

        # Enhanced Data Mapping Section
        st.subheader("🗺️ Enhanced Data Mapping Agent")

        # LLM Toggle Section
        st.markdown('<div class="llm-toggle">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("#### 🤖 LLM-Powered Intelligent Column Mapping")
            st.markdown("Enable AI-powered mapping using BGE embeddings for semantic similarity matching")

        with col2:
            llm_enabled = st.checkbox(
                "🤖 Enable LLM Mapping",
                value=st.session_state.llm_enabled,
                help="Use AI to automatically map columns based on semantic similarity"
            )
            st.session_state.llm_enabled = llm_enabled

        st.markdown('</div>', unsafe_allow_html=True)

        # Show current column mapping approach
        if llm_enabled:
            st.info("🤖 **LLM Mode Enabled**: AI will automatically map columns using BGE embeddings and semantic similarity")
        else:
            st.info("✏️ **Manual Mode**: You will manually map each column to the CBUAE banking schema")

        # Run data mapping
        if st.button("🎯 Run Data Mapping", type="primary"):
            with st.spinner("Running advanced data mapping..."):
                try:
                    # Use synchronous wrapper for mapping
                    mapping_result = run_sync_agent_method(
                        agent.map_columns,
                        data=uploaded_data,
                        user_id=st.session_state.user_data.get("username", "unknown"),
                        session_id=st.session_state.session_id,
                        use_llm=llm_enabled,
                        llm_api_key=None  # Can be configured if needed
                    )

                    if mapping_result and mapping_result.success:
                        st.session_state.mapping_results = mapping_result
                        st.success("✅ Data mapping completed successfully!")

                        # Display mapping results
                        display_mapping_results(mapping_result, llm_enabled)

                    else:
                        error_msg = getattr(mapping_result, 'error', 'Unknown error') if mapping_result else 'Mapping failed'
                        st.error(f"❌ Data mapping failed: {error_msg}")

                except Exception as e:
                    st.error(f"❌ Data mapping failed: {str(e)}")
                    logger.error(f"Mapping error: {e}")

                    # Show more detailed error info
                    with st.expander("🔍 Error Details"):
                        st.code(f"Error Type: {type(e).__name__}")
                        st.code(f"Error Message: {str(e)}")
                        st.markdown("**Possible Solutions:**")
                        st.markdown("1. Ensure BGE dependencies are installed")
                        st.markdown("2. Check that the agent is properly initialized")
                        st.markdown("3. Verify data format compatibility")

        # Manual mapping interface (when LLM is disabled)
        if not llm_enabled and st.session_state.uploaded_data is not None:
            render_manual_mapping_interface()

        # Store raw data if no mapping done yet
        if st.session_state.processed_data is None and uploaded_data is not None:
            st.session_state.processed_data = uploaded_data

def display_mapping_results(mapping_result, llm_enabled):
    """Display data mapping results with download options"""
    st.subheader("📋 Data Mapping Results")

    # Mapping statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_columns = len(mapping_result.mapping_sheet) if hasattr(mapping_result, 'mapping_sheet') and mapping_result.mapping_sheet is not None else 0
        st.metric("📊 Total Columns", total_columns)

    with col2:
        if llm_enabled and hasattr(mapping_result, 'auto_mapping_percentage'):
            auto_mapped = mapping_result.auto_mapping_percentage
            st.metric("🎯 Auto-mapped", f"{auto_mapped:.1f}%")
        else:
            manual_mapped = len(mapping_result.mappings) if hasattr(mapping_result, 'mappings') and mapping_result.mappings else 0
            st.metric("✏️ Mapped", manual_mapped)

    with col3:
        method = getattr(mapping_result, 'method', 'LLM/BGE' if llm_enabled else 'Manual')
        st.metric("🔧 Method", method)

    with col4:
        if hasattr(mapping_result, 'confidence_distribution') and mapping_result.confidence_distribution:
            high_conf = mapping_result.confidence_distribution.get('high', 0)
            st.metric("✅ High Confidence", high_conf)
        else:
            st.metric("✅ High Confidence", "N/A")

    # Display mapping sheet
    if hasattr(mapping_result, 'mapping_sheet') and mapping_result.mapping_sheet is not None:
        st.subheader("📊 Column Mapping Sheet")

        mapping_df = mapping_result.mapping_sheet

        # Apply styling for confidence scores if LLM enabled
        if llm_enabled and 'Confidence_Level' in mapping_df.columns:
            def style_confidence(val):
                if val == 'high':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'medium':
                    return 'background-color: #fff3cd; color: #856404'
                elif val == 'low':
                    return 'background-color: #f8d7da; color: #721c24'
                return ''

            styled_mapping = mapping_df.style.applymap(
                style_confidence,
                subset=['Confidence_Level']
            )
            st.dataframe(styled_mapping, use_container_width=True)
        else:
            st.dataframe(mapping_df, use_container_width=True)

        # Download mapping sheet
        csv_mapping = mapping_df.to_csv(index=False)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📄 Download Mapping Sheet (CSV)",
                csv_mapping,
                f"column_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                help="Download the column mapping for future reference"
            )

        with col2:
            # Download as JSON for programmatic use
            mapping_json = mapping_df.to_json(indent=2)
            st.download_button(
                "📋 Download Mapping (JSON)",
                mapping_json,
                f"column_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                help="Download mapping in JSON format for API integration"
            )

        # Show processing time if available
        if hasattr(mapping_result, 'processing_time') and mapping_result.processing_time:
            st.info(f"⏱️ Processing completed in {mapping_result.processing_time:.2f} seconds")

        # Apply mapping to data
        if st.button("✅ Apply Mapping to Data", type="secondary"):
            try:
                # Apply the mapping to create processed data using the mappings
                mapped_data = apply_column_mapping_from_result(st.session_state.uploaded_data, mapping_result)
                st.session_state.processed_data = mapped_data
                st.success("✅ Column mapping applied successfully! Data is ready for analysis.")

                # Show sample of mapped data
                with st.expander("👀 Mapped Data Preview"):
                    st.dataframe(mapped_data.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to apply mapping: {str(e)}")
    else:
        st.warning("⚠️ No mapping sheet available in the result")

        # Show raw mappings if available
        if hasattr(mapping_result, 'mappings') and mapping_result.mappings:
            st.subheader("🔗 Raw Mappings")
            st.json(mapping_result.mappings)

def apply_column_mapping_from_result(data, mapping_result):
    """Apply column mapping from MappingResult to data"""
    mapped_data = data.copy()

    # Try to get mappings from different possible sources
    mappings = {}

    if hasattr(mapping_result, 'mappings') and mapping_result.mappings:
        mappings = mapping_result.mappings
    elif hasattr(mapping_result, 'mapping_sheet') and mapping_result.mapping_sheet is not None:
        # Extract mappings from mapping sheet
        mapping_df = mapping_result.mapping_sheet
        if 'Source_Column' in mapping_df.columns and 'Target_Field' in mapping_df.columns:
            mappings = dict(zip(mapping_df['Source_Column'], mapping_df['Target_Field']))
        elif 'source_column' in mapping_df.columns and 'target_field' in mapping_df.columns:
            mappings = dict(zip(mapping_df['source_column'], mapping_df['target_field']))

    # Apply mappings
    if mappings:
        # Create column rename mapping, excluding None/empty values
        rename_map = {}
        for source_col, target_field in mappings.items():
            if (source_col in mapped_data.columns and
                target_field and
                target_field != '[Skip]' and
                target_field != 'None'):
                rename_map[source_col] = target_field

        # Rename columns
        if rename_map:
            mapped_data = mapped_data.rename(columns=rename_map)
            logger.info(f"Applied {len(rename_map)} column mappings")

    return mapped_data

def render_manual_mapping_interface():
    """Render manual column mapping interface"""
    st.subheader("✏️ Manual Column Mapping")
    st.info("Map each column from your data to the CBUAE banking schema fields")

    uploaded_data = st.session_state.uploaded_data
    source_columns = uploaded_data.columns.tolist()
    target_fields = list(CBUAE_BANKING_SCHEMA.keys())

    st.markdown('<div class="mapping-container">', unsafe_allow_html=True)

    # Create mapping interface
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Your Data Columns**")
        for col in source_columns:
            st.markdown(f"• {col}")

    with col2:
        st.markdown("**🏦 CBUAE Schema Fields**")
        for field in target_fields:
            st.markdown(f"• {field}")

    st.markdown("---")
    st.markdown("**🔗 Create Mappings**")

    # Mapping form
    with st.form("manual_mapping_form"):
        mappings = {}

        for source_col in source_columns:
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.text_input(f"Source", value=source_col, disabled=True, key=f"src_{source_col}")

            with col2:
                target_options = ["[Skip]"] + target_fields
                current_mapping = st.session_state.manual_mapping.get(source_col, "[Skip]")

                selected_target = st.selectbox(
                    f"Target",
                    target_options,
                    index=target_options.index(current_mapping) if current_mapping in target_options else 0,
                    key=f"tgt_{source_col}"
                )

                if selected_target != "[Skip]":
                    mappings[source_col] = selected_target

            with col3:
                # Show sample data
                sample_value = str(uploaded_data[source_col].iloc[0]) if len(uploaded_data) > 0 else "N/A"
                st.text_input("Sample", value=sample_value[:20] + "..." if len(sample_value) > 20 else sample_value,
                            disabled=True, key=f"sample_{source_col}")

        if st.form_submit_button("💾 Save Manual Mapping", type="primary"):
            st.session_state.manual_mapping = mappings

            # Create mapping result
            mapping_data = []
            for source_col, target_field in mappings.items():
                mapping_data.append({
                    'source_column': source_col,
                    'target_field': target_field,
                    'mapping_type': 'manual',
                    'confidence_score': 1.0  # Manual mappings have 100% confidence
                })

            mapping_df = pd.DataFrame(mapping_data)

            # Create mock mapping result
            from types import SimpleNamespace
            mapping_result = SimpleNamespace(
                success=True,
                mapping_sheet=mapping_df,
                method="manual",
                auto_mapping_percentage=0,
                confidence_distribution={'high': len(mappings), 'medium': 0, 'low': 0}
            )

            st.session_state.mapping_results = mapping_result
            st.success(f"✅ Manual mapping saved! {len(mappings)} columns mapped.")

            # Display the results
            display_mapping_results(mapping_result, False)

    st.markdown('</div>', unsafe_allow_html=True)

# Dormancy analysis section
def render_dormancy_analysis_section():
    """Render dormancy analysis section with real agents"""
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please complete data processing and mapping first")
        return

    st.markdown('<div class="section-header"><h2>🏃 Dormancy Analysis Agents</h2></div>', unsafe_allow_html=True)

    if not DORMANCY_AGENTS_AVAILABLE:
        st.error("❌ Dormancy agents not available. Please check agents/Dormant_agent.py imports.")
        return

    # Display available dormancy agents
    try:
        orchestrator = DormancyWorkflowOrchestrator()
        agent_info = orchestrator.get_all_agent_info()

        st.subheader("🤖 Available Dormancy Agents")
        st.info(f"📊 Total Agents: {len(agent_info)} | 🎯 Ready for Analysis")

        # Show agent cards
        agent_cols = st.columns(min(3, len(agent_info)))
        for idx, (agent_name, info) in enumerate(agent_info.items()):
            with agent_cols[idx % len(agent_cols)]:
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 {agent_name.replace('_', ' ').title()}</h4>
                        <p><strong>Article:</strong> {info['cbuae_article']}</p>
                        <p><strong>Status:</strong> {info['ui_status']}</p>
                        <p>{info['description'][:100]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading dormancy agents: {e}")
        return

    # Run dormancy analysis
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🚀 Execute Comprehensive Dormancy Analysis")
        st.info("This will run all available dormancy agents on your processed data")

    with col2:
        if st.button("▶️ Run All Agents", type="primary", use_container_width=True):
            with st.spinner("Running comprehensive dormancy analysis..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Update progress
                    progress_bar.progress(0.2)
                    status_text.text("🔄 Initializing dormancy agents...")

                    dormancy_results = run_comprehensive_dormancy_analysis_with_csv(
                        user_id=f"streamlit_{st.session_state.session_id[:8]}",
                        account_data=st.session_state.processed_data,
                        report_date=datetime.now().strftime('%Y-%m-%d')
                    )

                    progress_bar.progress(1.0)
                    status_text.text("✅ Dormancy analysis completed!")

                    if dormancy_results.get("success"):
                        st.session_state.dormancy_results = dormancy_results
                        st.success("🎉 Dormancy analysis completed successfully!")

                        # Display summary results
                        display_dormancy_results(dormancy_results)

                    else:
                        st.error(f"❌ Dormancy analysis failed: {dormancy_results.get('error')}")

                except Exception as e:
                    st.error(f"❌ Dormancy analysis failed: {str(e)}")
                    logger.error(f"Dormancy analysis error: {e}")

def display_dormancy_results(results):
    """Display dormancy analysis results with download options"""
    if not results or not results.get("success"):
        return

    st.subheader("📊 Dormancy Analysis Results")

    # Summary metrics
    summary = results.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Accounts", f"{summary.get('total_accounts_processed', 0):,}")
    with col2:
        st.metric("🏃 Dormant Accounts", f"{summary.get('total_dormant_accounts', 0):,}")
    with col3:
        processing_time = results.get("processing_time", 0)
        st.metric("⏱️ Processing Time", f"{processing_time:.1f}s")
    with col4:
        dormancy_rate = (summary.get('total_dormant_accounts', 0) /
                         max(summary.get('total_accounts_processed', 1), 1)) * 100
        st.metric("📈 Dormancy Rate", f"{dormancy_rate:.1f}%")

    # Individual agent results
    agent_results = results.get("agent_results", {})
    if agent_results:
        st.subheader("🤖 Individual Agent Results & Downloads")

        for agent_name, agent_result in agent_results.items():
            if agent_result.get("success"):
                with st.expander(f"🎯 {agent_name.replace('_', ' ').title()}", expanded=False):

                    # Agent metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Records Processed", f"{agent_result.get('records_processed', 0):,}")
                    with col2:
                        st.metric("Dormant Found", f"{agent_result.get('dormant_accounts_found', 0):,}")
                    with col3:
                        st.metric("Processing Time", f"{agent_result.get('processing_time', 0):.2f}s")

                    # Download buttons
                    col1, col2 = st.columns(2)

                    with col1:
                        # CSV download
                        csv_data = agent_result.get('csv_export', {})
                        if csv_data.get('available'):
                            csv_content = csv_data.get('csv_content', '')
                            if csv_content:
                                st.download_button(
                                    f"📄 Download CSV Results",
                                    csv_content,
                                    f"{agent_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv",
                                    key=f"csv_download_{agent_name}",
                                    help="Download detailed results in CSV format"
                                )

                    with col2:
                        # Summary download
                        summary_text = agent_result.get('summary', 'No summary available')
                        if summary_text:
                            st.download_button(
                                f"📋 Download Summary",
                                summary_text,
                                f"{agent_name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                "text/plain",
                                key=f"summary_download_{agent_name}",
                                help="Download agent summary report"
                            )

                    # Agent summary (crisp)
                    st.markdown("**🔍 Agent Summary:**")
                    st.text_area(
                        "",
                        summary_text[:500] + "..." if len(summary_text) > 500 else summary_text,
                        height=100,
                        disabled=True,
                        key=f"display_summary_{agent_name}"
                    )

# Compliance analysis section
def render_compliance_analysis_section():
    """Render compliance analysis section"""
    if st.session_state.dormancy_results is None:
        st.warning("⚠️ Please complete dormancy analysis first")
        return

    st.markdown('<div class="section-header"><h2>⚖️ Compliance Verification Agents</h2></div>', unsafe_allow_html=True)

    if not COMPLIANCE_AGENTS_AVAILABLE:
        st.error("❌ Compliance agents not available. Please check agents/compliance_verification_agent.py imports.")
        return

    # Display available compliance agents
    try:
        compliance_info = get_all_compliance_agents_info()

        st.subheader("🤖 Available Compliance Agents")
        st.info(
            f"📊 Total Agents: {compliance_info['total_agents']} | 📋 CBUAE Articles: {len(compliance_info['cbuae_articles_covered'])}")

        # Group agents by category
        for category, agents in compliance_info['agents_by_category'].items():
            with st.expander(f"📂 {category.replace('_', ' ').title()} ({len(agents)} agents)"):
                for agent in agents:
                    st.markdown(f"""
                    <div class="agent-status agent-success">
                        <strong>{agent['agent_name']}</strong> (Article: {agent['cbuae_article']})<br>
                        {agent['description']}
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading compliance agents: {e}")
        return

    # Run compliance analysis
    if st.button("🚀 Run Compliance Analysis", type="primary"):
        with st.spinner("Running comprehensive compliance verification..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                progress_bar.progress(0.2)
                status_text.text("🔄 Initializing compliance agents...")

                # Run real compliance analysis
                compliance_results = run_comprehensive_compliance_analysis_with_csv(
                    user_id=f"streamlit_{st.session_state.session_id[:8]}",
                    dormancy_results=st.session_state.dormancy_results,
                    accounts_df=st.session_state.processed_data
                )

                progress_bar.progress(1.0)
                status_text.text("✅ Compliance analysis completed!")

                if compliance_results.get("success"):
                    st.session_state.compliance_results = compliance_results
                    st.success("🎉 Compliance analysis completed successfully!")

                    # Display compliance results
                    display_compliance_results(compliance_results)

                else:
                    st.error(f"❌ Compliance analysis failed: {compliance_results.get('error')}")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Compliance analysis error: {e}")

def display_compliance_results(results):
    """Display compliance analysis results with download options"""
    if not results or not results.get("success"):
        return

    st.subheader("📊 Compliance Verification Results")

    # Summary metrics
    summary = results.get("compliance_summary", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Accounts Analyzed", f"{summary.get('total_accounts_analyzed', 0):,}")
    with col2:
        st.metric("⚠️ Violations Found", f"{summary.get('total_violations_found', 0):,}")
    with col3:
        st.metric("🎯 Actions Generated", f"{summary.get('total_actions_generated', 0):,}")
    with col4:
        compliance_status = summary.get('overall_compliance_status', 'UNKNOWN')
        status_color = "🟢" if compliance_status == "COMPLIANT" else "🔴"
        st.metric("📋 Status", f"{status_color} {compliance_status}")

    # Individual agent results
    agent_results = results.get("agent_results", {})
    if agent_results:
        st.subheader("🤖 Individual Compliance Agent Results")

        for agent_name, agent_result in agent_results.items():
            if agent_result.get("success"):
                with st.expander(f"⚖️ {agent_name.replace('_', ' ').title()}", expanded=False):

                    # Agent metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Accounts Processed", f"{agent_result.get('accounts_processed', 0):,}")
                    with col2:
                        st.metric("Violations Found", f"{agent_result.get('violations_found', 0):,}")
                    with col3:
                        st.metric("Processing Time", f"{agent_result.get('processing_time', 0):.2f}s")

                    # Download buttons
                    col1, col2 = st.columns(2)

                    with col1:
                        # CSV download
                        if agent_result.get('csv_download_ready'):
                            csv_export = agent_result.get('compliance_summary', {}).get('csv_export', {})
                            if csv_export.get('available'):
                                csv_content = csv_export.get('csv_content', '')
                                if csv_content:
                                    st.download_button(
                                        f"📄 Download CSV Results",
                                        csv_content,
                                        f"{agent_name}_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv",
                                        key=f"compliance_csv_download_{agent_name}"
                                    )

                    with col2:
                        # Summary download
                        summary_text = agent_result.get('summary', 'No summary available')
                        if summary_text:
                            st.download_button(
                                f"📋 Download Summary",
                                summary_text,
                                f"{agent_name}_compliance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                "text/plain",
                                key=f"compliance_summary_download_{agent_name}"
                            )

                    # Recommendations
                    recommendations = agent_result.get('recommendations', [])
                    if recommendations:
                        st.subheader("💡 Recommendations")
                        for rec in recommendations:
                            st.markdown(f"• {rec}")

# Reports section
def render_reports_section():
    """Render comprehensive reports section"""
    st.markdown('<div class="section-header"><h2>📈 Comprehensive Reports & Analytics</h2></div>', unsafe_allow_html=True)

    # Executive summary
    st.subheader("📋 Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        accounts_count = len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0
        st.metric("📊 Total Accounts", f"{accounts_count:,}")

    with col2:
        dormant_count = 0
        if st.session_state.dormancy_results:
            dormant_count = st.session_state.dormancy_results.get("summary", {}).get("total_dormant_accounts", 0)
        st.metric("🏃 Dormant Accounts", f"{dormant_count:,}")

    with col3:
        violations_count = 0
        if st.session_state.compliance_results:
            violations_count = st.session_state.compliance_results.get("compliance_summary", {}).get("total_violations_found", 0)
        st.metric("⚠️ Violations Found", f"{violations_count:,}")

    with col4:
        total_agents = 0
        if DORMANCY_AGENTS_AVAILABLE:
            try:
                orchestrator = DormancyWorkflowOrchestrator()
                total_agents += len(orchestrator.get_all_agent_info())
            except:
                pass
        if COMPLIANCE_AGENTS_AVAILABLE:
            try:
                compliance_info = get_all_compliance_agents_info()
                total_agents += compliance_info['total_agents']
            except:
                pass
        st.metric("🤖 Total Agents", f"{total_agents}")

    # Detailed agent status
    st.subheader("🤖 All Agents Status & Account Numbers")

    # Data Processing Agent Status
    st.markdown("#### 📊 Data Processing Agent")
    if DATA_PROCESSING_AVAILABLE and st.session_state.data_processing_agent:
        processed_accounts = len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0
        quality_status = "✅ Completed" if st.session_state.quality_results else "⏳ Pending"
        mapping_status = "✅ Completed" if st.session_state.mapping_results else "⏳ Pending"

        st.markdown(f"""
        <div class="agent-status agent-success">
            <strong>📊 Unified Data Processing Agent</strong><br>
            Status: ✅ Available & Active<br>
            Accounts Processed: {processed_accounts:,}<br>
            Quality Analysis: {quality_status}<br>
            Column Mapping: {mapping_status}<br>
            Possible Actions: Data Upload, Quality Check, Column Mapping, Data Export
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="agent-status agent-error">
            <strong>📊 Data Processing Agent</strong><br>
            Status: ❌ Not Available<br>
            Accounts Processed: 0<br>
            Possible Actions: None
        </div>
        """, unsafe_allow_html=True)

    # Generate comprehensive report
    st.subheader("📊 Generate Comprehensive Report")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate Full Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                report_data = generate_comprehensive_report()

                if report_data:
                    st.success("✅ Report generated successfully!")

                    # Display report summary
                    st.subheader("📋 Report Summary")

                    # Key metrics from report
                    data_processing = report_data.get("data_processing", {})
                    dormancy_analysis = report_data.get("dormancy_analysis", {})
                    compliance_analysis = report_data.get("compliance_analysis", {})

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records Processed", f"{data_processing.get('total_records', 0):,}")
                    with col2:
                        st.metric("Dormant Accounts", f"{dormancy_analysis.get('total_dormant_accounts', 0):,}")
                    with col3:
                        st.metric("Compliance Violations", f"{compliance_analysis.get('total_violations', 0):,}")

                    # Download options
                    report_json = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        "📄 Download Full Report (JSON)",
                        report_json,
                        f"banking_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        help="Download comprehensive analysis report"
                    )

                    # Show detailed report
                    with st.expander("📊 Detailed Report Data"):
                        st.json(report_data)

def generate_comprehensive_report() -> Dict:
    """Generate comprehensive analysis report"""
    try:
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "session_id": st.session_state.session_id,
                "user": st.session_state.user_data.get("username") if st.session_state.user_data else "unknown",
                "system_version": "2.0.0-real-agents"
            },
            "data_processing": {
                "agent_available": DATA_PROCESSING_AVAILABLE,
                "agent_loaded": st.session_state.data_processing_agent is not None,
                "total_records": len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0,
                "quality_analysis_completed": st.session_state.quality_results is not None,
                "mapping_completed": st.session_state.mapping_results is not None,
                "llm_mapping_enabled": st.session_state.llm_enabled
            },
            "quality_analysis": {},
            "mapping_results": {},
            "dormancy_analysis": {},
            "compliance_analysis": {},
            "agent_status": {
                "data_processing_available": DATA_PROCESSING_AVAILABLE,
                "dormancy_agents_available": DORMANCY_AGENTS_AVAILABLE,
                "compliance_agents_available": COMPLIANCE_AGENTS_AVAILABLE,
                "memory_agent_available": MEMORY_AGENT_AVAILABLE
            }
        }

        # Add quality results
        if st.session_state.quality_results:
            report["quality_analysis"] = {
                "success": st.session_state.quality_results.success,
                "overall_score": st.session_state.quality_results.overall_score,
                "quality_level": st.session_state.quality_results.quality_level,
                "missing_percentage": st.session_state.quality_results.missing_percentage,
                "duplicate_records": getattr(st.session_state.quality_results, 'duplicate_records', 0),
                "metrics": getattr(st.session_state.quality_results, 'metrics', {}),
                "recommendations_count": len(getattr(st.session_state.quality_results, 'recommendations', []))
            }

        # Add mapping results
        if st.session_state.mapping_results:
            report["mapping_results"] = {
                "success": st.session_state.mapping_results.success,
                "method": getattr(st.session_state.mapping_results, 'method', 'unknown'),
                "llm_enabled": st.session_state.llm_enabled,
                "auto_mapping_percentage": getattr(st.session_state.mapping_results, 'auto_mapping_percentage', 0),
                "confidence_distribution": getattr(st.session_state.mapping_results, 'confidence_distribution', {}),
                "total_mappings": len(st.session_state.manual_mapping) if not st.session_state.llm_enabled else 0
            }

        # Add dormancy results
        if st.session_state.dormancy_results:
            report["dormancy_analysis"] = {
                "success": st.session_state.dormancy_results.get("success"),
                "total_dormant_accounts": st.session_state.dormancy_results.get("summary", {}).get("total_dormant_accounts", 0),
                "agents_executed": len(st.session_state.dormancy_results.get("agent_results", {})),
                "processing_time": st.session_state.dormancy_results.get("processing_time", 0)
            }

        # Add compliance results
        if st.session_state.compliance_results:
            report["compliance_analysis"] = {
                "success": st.session_state.compliance_results.get("success"),
                "total_violations": st.session_state.compliance_results.get("compliance_summary", {}).get("total_violations_found", 0),
                "agents_executed": len(st.session_state.compliance_results.get("agent_results", {})),
                "compliance_status": st.session_state.compliance_results.get("compliance_summary", {}).get("overall_compliance_status", "UNKNOWN")
            }

        return report

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {"error": str(e)}

# Sidebar navigation
def render_sidebar():
    """Render navigation sidebar with enhanced status"""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")

        pages = [
            "📊 Data Processing",
            "🏃 Dormancy Analysis",
            "⚖️ Compliance Verification",
            "📈 Reports & Analytics"
        ]

        selected_page = st.radio("Select Section:", pages)

        st.markdown("---")
        st.markdown("### 📋 Session Info")
        st.text(f"Session: {st.session_state.session_id[:8]}")
        st.text(f"User: {st.session_state.user_data['username'] if st.session_state.user_data else 'Unknown'}")

        # Progress tracking
        st.markdown("### 📊 Progress")
        progress_items = [
            ("Data Upload", st.session_state.uploaded_data is not None),
            ("Quality Check", st.session_state.quality_results is not None),
            ("Column Mapping", st.session_state.mapping_results is not None),
            ("Dormancy Analysis", st.session_state.dormancy_results is not None),
            ("Compliance Check", st.session_state.compliance_results is not None)
        ]

        for item, completed in progress_items:
            status = "✅" if completed else "⏳"
            st.text(f"{status} {item}")

        # System status
        st.markdown("### 🤖 System Status")
        st.text(f"Data Processing: {'✅' if DATA_PROCESSING_AVAILABLE else '❌'}")
        st.text(f"Dormancy Agents: {'✅' if DORMANCY_AGENTS_AVAILABLE else '❌'}")
        st.text(f"Compliance Agents: {'✅' if COMPLIANCE_AGENTS_AVAILABLE else '❌'}")
        st.text(f"Memory Agent: {'✅' if MEMORY_AGENT_AVAILABLE else '❌'}")

        # Logout button
        if st.button("🚪 Logout", type="secondary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        return selected_page

# Main application logic
def main():
    """Main application entry point"""
    initialize_session_state()

    # Check authentication
    if not st.session_state.authenticated:
        render_login_page()
        return

    # Check critical dependencies
    if not DATA_PROCESSING_AVAILABLE:
        st.error("❌ CRITICAL: Data Processing Agent not available. Please check agents/Data_Process.py")
        st.info("💡 The application requires real agents to function properly. Please ensure all agent imports are working.")
        return

    # Debug: Check agent availability
    agent_status = check_agent_availability()
    logger.info(f"System check - Agent status: {agent_status}")

    # Debug: Check function definitions
    test_results = test_function_definitions()
    logger.info(f"Function definition test results: {test_results}")

    # Initialize agents with enhanced error handling
    try:
        logger.info("Starting agent initialization...")

        # Verify functions exist before calling them
        if 'initialize_memory_agent' not in globals():
            raise NameError("initialize_memory_agent function not defined")
        if 'initialize_data_processing_agent' not in globals():
            raise NameError("initialize_data_processing_agent function not defined")

        # Initialize memory agent first
        memory_result = initialize_memory_agent()
        logger.info(f"Memory agent initialized: {memory_result is not None}")

        # Initialize data processing agent
        data_agent_result = initialize_data_processing_agent()
        logger.info(f"Data processing agent initialized: {data_agent_result}")

        if not data_agent_result:
            st.error("❌ Failed to initialize Data Processing Agent")
            st.info("💡 Please check the error details below and ensure agents.Data_Process is properly imported")
            return

    except Exception as e:
        st.error(f"❌ Agent initialization failed: {str(e)}")
        logger.error(f"Agent initialization error: {e}")

        # Show detailed error for debugging
        with st.expander("🔍 Initialization Error Details"):
            st.code(f"Error Type: {type(e).__name__}")
            st.code(f"Error Message: {str(e)}")

            # Show function definition test results
            st.markdown("**Function Definition Test Results:**")
            test_results = test_function_definitions()
            for func_name, result in test_results.items():
                st.text(f"{func_name}: {result}")

            # Show agent availability status
            st.markdown("**Agent Availability Status:**")
            status = check_agent_availability()
            for key, value in status.items():
                status_icon = "✅" if value else "❌"
                st.text(f"{status_icon} {key}: {value}")

            st.markdown("**Available Functions in Global Scope:**")
            available_funcs = [name for name in globals().keys() if callable(globals()[name]) and not name.startswith('_')]
            st.text(f"Functions found: {len(available_funcs)}")
            for func in sorted(available_funcs):
                st.text(f"  - {func}")

            st.markdown("**Expected Functions:**")
            expected = ['initialize_memory_agent', 'initialize_data_processing_agent', 'run_sync_agent_method']
            for func in expected:
                exists = func in globals()
                status_icon = "✅" if exists else "❌"
                st.text(f"{status_icon} {func}: {'Found' if exists else 'Missing'}")

            st.markdown("**Possible Solutions:**")
            st.markdown("1. Check that agents.Data_Process is properly imported")
            st.markdown("2. Verify all dependencies are installed")
            st.markdown("3. Ensure the Data Processing Agent class is available")
            st.markdown("4. Check Python path and module accessibility")
        return

    # Render main interface
    render_main_header()
    selected_page = render_sidebar()

    # Render selected page
    try:
        if selected_page == "📊 Data Processing":
            render_data_processing_section()
        elif selected_page == "🏃 Dormancy Analysis":
            render_dormancy_analysis_section()
        elif selected_page == "⚖️ Compliance Verification":
            render_compliance_analysis_section()
        elif selected_page == "📈 Reports & Analytics":
            render_reports_section()

    except Exception as e:
        st.error(f"❌ Page rendering failed: {str(e)}")
        logger.error(f"Page rendering error: {e}")

        with st.expander("🔍 Error Details"):
            st.code(f"Error: {str(e)}")
            st.markdown("**Possible Solutions:**")
            st.markdown("1. Check that all required agents are properly imported")
            st.markdown("2. Verify database connections and file permissions")
            st.markdown("3. Try refreshing the page or restarting the application")

# Call verification at module level
if __name__ != "__main__":
    verify_functions()

# Run the application
if __name__ == "__main__":
    main()