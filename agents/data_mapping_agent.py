"""
Banking Compliance Analysis - Real-Time Streamlit Application
Uses actual CSV structure and real-time data analysis with BGE embeddings
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import json
import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Banking Compliance Analysis System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add agents directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.join(current_dir, 'agents')
if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 2rem;
        color: #2c5aa0;
        border-left: 5px solid #2c5aa0;
        padding-left: 1rem;
        margin: 1.5rem 0;
    }
    
    .agent-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-box {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-success { background: #d4edda; color: #155724; }
    .status-warning { background: #fff3cd; color: #856404; }
    .status-error { background: #f8d7da; color: #721c24; }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .high-confidence { background-color: #d4edda; }
    .medium-confidence { background-color: #fff3cd; }
    .low-confidence { background-color: #f8d7da; }
</style>
""", unsafe_allow_html=True)

# Initialize agent availability flags
DATA_AGENTS_AVAILABLE = False
DORMANCY_AGENTS_AVAILABLE = False
COMPLIANCE_AGENTS_AVAILABLE = False

# Session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'mapping_results' not in st.session_state:
        st.session_state.mapping_results = None
    if 'dormancy_results' not in st.session_state:
        st.session_state.dormancy_results = {}
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = {}
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Processing"
    if 'llm_enabled' not in st.session_state:
        st.session_state.llm_enabled = False

# Try importing real agents
try:
    # Import data processing agents
    from agents.data_upload_agent import BankingComplianceUploader
    from agents.data_mapping_agent import (
        DataMappingAgent,
        run_automated_data_mapping,
        apply_llm_assistance,
        create_data_mapping_agent
    )

    # Import data processing agent
    try:
        from Data_Process import DataProcessingAgent
        DATA_PROCESSING_AVAILABLE = True
    except ImportError:
        DATA_PROCESSING_AVAILABLE = False
        logger.warning("Data processing agent not available")

    DATA_AGENTS_AVAILABLE = True
    logger.info("✅ Data processing agents imported successfully")

except Exception as e:
    logger.warning(f"⚠️ Data processing agents not available: {e}")
    DATA_AGENTS_AVAILABLE = False

# Try importing dormancy agents
try:
    from agents.Dormant_agent import (
        DemandDepositDormancyAgent,
        FixedDepositDormancyAgent,
        InvestmentAccountDormancyAgent,
        PaymentInstrumentsDormancyAgent,
        SafeDepositDormancyAgent,
        ContactAttemptsAgent,
        CBTransferEligibilityAgent,
        HighValueDormantAccountsAgent,
        run_comprehensive_dormancy_analysis_csv
    )

    DORMANCY_AGENTS_AVAILABLE = True
    logger.info("✅ Dormancy agents imported successfully")

except Exception as e:
    logger.warning(f"⚠️ Dormancy agents not available: {e}")
    DORMANCY_AGENTS_AVAILABLE = False

# Try importing compliance agents
try:
    from agents.compliance_verification_agent import (
        DetectIncompleteContactAttemptsAgent,
        DetectUnflaggedDormantCandidatesAgent,
        DetectInternalLedgerCandidatesAgent,
        DetectStatementFreezeCandidatesAgent,
        run_comprehensive_compliance_analysis_csv
    )

    COMPLIANCE_AGENTS_AVAILABLE = True
    logger.info("✅ Compliance agents imported successfully")

except Exception as e:
    logger.warning(f"⚠️ Compliance agents not available: {e}")
    COMPLIANCE_AGENTS_AVAILABLE = False

# Login function
def show_login_page():
    """Display login page"""
    st.markdown('<div class="main-header">🏦 Banking Compliance Analysis System</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("## 🔐 System Login")
        st.markdown("Enter your credentials to access the banking compliance analysis system.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            col_login, col_demo = st.columns(2)
            with col_login:
                login_button = st.form_submit_button("🚪 Login", use_container_width=True)
            with col_demo:
                demo_button = st.form_submit_button("🎯 Demo Mode", use_container_width=True)

            if login_button:
                if username and password:
                    # Simple authentication (replace with actual auth logic)
                    if username == "admin" and password == "admin123":
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Try admin/admin123 for demo.")
                else:
                    st.error("❌ Please enter both username and password")

            if demo_button:
                st.session_state.logged_in = True
                st.session_state.username = "demo_user"
                st.success("✅ Demo mode activated!")
                time.sleep(1)
                st.rerun()

        # Demo credentials info
        with st.expander("🔍 Demo Credentials"):
            st.info("""
            **Demo Credentials:**
            - Username: admin
            - Password: admin123
            
            Or use the Demo Mode button for quick access.
            """)

# Data Upload Section
def show_data_upload_section():
    """Display data upload interface"""
    st.markdown('<div class="section-header">📤 Data Upload</div>', unsafe_allow_html=True)

    # Upload method selection
    upload_method = st.selectbox(
        "Select Upload Method:",
        ["📄 File Upload", "🔗 Google Drive", "☁️ Azure Data Lake", "🗄️ HDFS"],
        help="Choose your preferred data source method"
    )

    uploaded_data = None

    if upload_method == "📄 File Upload":
        uploaded_data = handle_file_upload()
    elif upload_method == "🔗 Google Drive":
        uploaded_data = handle_drive_upload()
    elif upload_method == "☁️ Azure Data Lake":
        uploaded_data = handle_datalake_upload()
    elif upload_method == "🗄️ HDFS":
        uploaded_data = handle_hdfs_upload()

    return uploaded_data

def handle_file_upload():
    """Handle file upload"""
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload CSV, Excel, or JSON files"
    )

    if uploaded_file:
        try:
            # Determine file type and read accordingly
            file_extension = Path(uploaded_file.name).suffix.lower()

            if file_extension == '.csv':
                data = pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(uploaded_file)
            elif file_extension == '.json':
                data = pd.read_json(uploaded_file)
            else:
                st.error("❌ Unsupported file format")
                return None

            st.success(f"✅ File uploaded successfully! {len(data)} records loaded.")

            # Show data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(data.head(10))
                st.info(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

                # Show column information
                col_info = []
                for col in data.columns:
                    col_info.append({
                        'Column': col,
                        'Type': str(data[col].dtype),
                        'Non-Null': data[col].count(),
                        'Null %': f"{(data[col].isnull().sum() / len(data) * 100):.1f}%"
                    })

                st.markdown("#### Column Information")
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)

            st.session_state.uploaded_data = data
            return data

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

    return None

def handle_drive_upload():
    """Handle Google Drive upload"""
    drive_url = st.text_input("Google Drive Share URL", placeholder="Paste your Google Drive sharing link")

    if drive_url and st.button("📥 Load from Drive"):
        if DATA_AGENTS_AVAILABLE:
            try:
                uploader = BankingComplianceUploader()
                result = uploader.upload_from_drive(drive_url)
                if result.success:
                    st.success("✅ Data loaded from Google Drive successfully!")
                    st.session_state.uploaded_data = result.data
                    return result.data
                else:
                    st.error(f"❌ Failed to load from Drive: {result.error}")
            except Exception as e:
                st.error(f"❌ Drive upload error: {str(e)}")
        else:
            st.error("❌ Data upload agent not available")

    return None

def handle_datalake_upload():
    """Handle Azure Data Lake upload"""
    col1, col2 = st.columns(2)

    with col1:
        account_name = st.text_input("Storage Account Name")
        container_name = st.text_input("Container Name")

    with col2:
        file_path = st.text_input("File Path")
        access_key = st.text_input("Access Key", type="password")

    if st.button("🔗 Connect to Data Lake"):
        if DATA_AGENTS_AVAILABLE and all([account_name, container_name, file_path, access_key]):
            try:
                uploader = BankingComplianceUploader()
                result = uploader.upload_from_datalake(account_name, container_name, file_path, access_key)
                if result.success:
                    st.success("✅ Data loaded from Azure Data Lake successfully!")
                    st.session_state.uploaded_data = result.data
                    return result.data
                else:
                    st.error(f"❌ Failed to load from Data Lake: {result.error}")
            except Exception as e:
                st.error(f"❌ Data Lake upload error: {str(e)}")
        else:
            st.error("❌ Please fill all required fields or check agent availability")

    return None

def handle_hdfs_upload():
    """Handle HDFS upload"""
    col1, col2 = st.columns(2)

    with col1:
        hdfs_url = st.text_input("HDFS URL", placeholder="hdfs://namenode:port")
        file_path = st.text_input("File Path", placeholder="/path/to/file.csv")

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

    if st.button("🔗 Connect to HDFS"):
        if DATA_AGENTS_AVAILABLE and all([hdfs_url, file_path]):
            try:
                uploader = BankingComplianceUploader()
                result = uploader.upload_from_hdfs(hdfs_url, file_path, username, password)
                if result.success:
                    st.success("✅ Data loaded from HDFS successfully!")
                    st.session_state.uploaded_data = result.data
                    return result.data
                else:
                    st.error(f"❌ Failed to load from HDFS: {result.error}")
            except Exception as e:
                st.error(f"❌ HDFS upload error: {str(e)}")
        else:
            st.error("❌ Please fill required fields or check agent availability")

    return None

def create_actual_banking_data():
    """Create actual banking data matching the real CSV structure"""
    np.random.seed(42)
    n_records = 5000

    # Generate realistic data matching the exact CSV structure
    data = {}

    # Customer Information
    data['customer_id'] = [f'CUS{str(i).zfill(6)}' for i in range(1, n_records + 1)]
    data['customer_type'] = np.random.choice(['INDIVIDUAL', 'CORPORATE'], n_records, p=[0.85, 0.15])
    data['full_name_en'] = [f'Customer {i} Name' for i in range(1, n_records + 1)]
    data['full_name_ar'] = [f'العميل {i}' for i in range(1, n_records + 1)]
    data['id_number'] = [784199000000000 + i for i in range(n_records)]
    data['id_type'] = np.random.choice(['EMIRATES_ID', 'PASSPORT', 'TRADE_LICENSE'], n_records, p=[0.7, 0.25, 0.05])
    data['date_of_birth'] = pd.date_range('1970-01-01', '2000-01-01', periods=n_records).strftime('%Y-%m-%d')
    data['nationality'] = np.random.choice(['UAE', 'INDIA', 'PAKISTAN', 'PHILIPPINES', 'BANGLADESH'], n_records, p=[0.4, 0.2, 0.15, 0.15, 0.1])

    # Address Information
    emirates = ['DUBAI', 'ABU_DHABI', 'SHARJAH', 'AJMAN', 'FUJAIRAH', 'RAS_AL_KHAIMAH', 'UMM_AL_QUWAIN']
    data['address_line1'] = [f'Building {np.random.randint(1,500)}, Street {np.random.randint(1,100)}' for _ in range(n_records)]
    data['address_line2'] = [f'Apartment {np.random.randint(101,999)}' if np.random.random() > 0.3 else '' for _ in range(n_records)]
    data['city'] = np.random.choice(['DUBAI', 'ABU_DHABI', 'SHARJAH', 'AJMAN'], n_records, p=[0.5, 0.25, 0.15, 0.1])
    data['emirate'] = np.random.choice(emirates, n_records, p=[0.5, 0.25, 0.15, 0.05, 0.02, 0.02, 0.01])
    data['country'] = ['UAE'] * n_records
    data['postal_code'] = np.random.randint(10000, 99999, n_records)

    # Contact Information
    data['phone_primary'] = [971500000000 + np.random.randint(1000000, 9999999) for _ in range(n_records)]
    data['phone_secondary'] = [971420000000 + np.random.randint(1000000, 9999999) if np.random.random() > 0.4 else np.nan for _ in range(n_records)]
    data['email_primary'] = [f'customer{i}@email.com' for i in range(1, n_records + 1)]
    data['email_secondary'] = [f'alt{i}@email.com' if np.random.random() > 0.6 else '' for i in range(1, n_records + 1)]
    data['address_known'] = np.random.choice(['YES', 'NO'], n_records, p=[0.95, 0.05])
    data['last_contact_date'] = pd.date_range('2020-01-01', '2024-06-01', periods=n_records).strftime('%Y-%m-%d')
    data['last_contact_method'] = np.random.choice(['EMAIL', 'PHONE', 'SMS', 'LETTER'], n_records, p=[0.4, 0.35, 0.2, 0.05])

    # KYC and Risk
    data['kyc_status'] = np.random.choice(['COMPLETED', 'PENDING', 'EXPIRED', 'IN_PROGRESS'], n_records, p=[0.8, 0.1, 0.05, 0.05])
    data['kyc_expiry_date'] = pd.date_range('2024-07-01', '2026-12-31', periods=n_records).strftime('%Y-%m-%d')
    data['risk_rating'] = np.random.choice(['LOW', 'MEDIUM', 'HIGH'], n_records, p=[0.7, 0.25, 0.05])

    # Account Information
    data['account_id'] = [f'ACC{str(i).zfill(9)}' for i in range(1, n_records + 1)]
    data['account_type'] = np.random.choice(['SAVINGS', 'CURRENT', 'FIXED_DEPOSIT', 'INVESTMENT'], n_records, p=[0.5, 0.3, 0.15, 0.05])
    data['account_subtype'] = np.random.choice(['PREMIUM', 'BASIC', 'STANDARD', ''], n_records, p=[0.2, 0.4, 0.3, 0.1])
    data['account_name'] = [f'Account {i}' for i in range(1, n_records + 1)]
    data['currency'] = np.random.choice(['AED', 'USD', 'EUR'], n_records, p=[0.8, 0.15, 0.05])
    data['account_status'] = np.random.choice(['ACTIVE', 'DORMANT', 'CLOSED', 'SUSPENDED'], n_records, p=[0.65, 0.25, 0.05, 0.05])
    data['dormancy_status'] = np.random.choice(['Not_Dormant', 'Potentially_Dormant', 'Dormant', 'Transferred_to_CB'], n_records, p=[0.65, 0.15, 0.18, 0.02])

    # Account Dates and Activity
    data['opening_date'] = pd.date_range('2010-01-01', '2023-01-01', periods=n_records).strftime('%Y-%m-%d')
    data['closing_date'] = [pd.Timestamp('2024-01-01').strftime('%Y-%m-%d') if np.random.random() > 0.95 else '' for _ in range(n_records)]
    data['last_transaction_date'] = pd.date_range('2018-01-01', '2024-06-01', periods=n_records).strftime('%Y-%m-%d')
    data['last_system_transaction_date'] = pd.date_range('2023-01-01', '2024-06-01', periods=n_records).strftime('%Y-%m-%d')

    # Balances and Financial
    data['balance_current'] = np.random.uniform(100, 500000, n_records).round(2)
    data['balance_available'] = [max(0, bal - np.random.uniform(0, 1000)) for bal in data['balance_current']]
    data['balance_minimum'] = np.random.choice([1000, 5000, 10000, 25000], n_records, p=[0.5, 0.3, 0.15, 0.05])
    data['interest_rate'] = np.random.uniform(0.5, 3.5, n_records).round(2)
    data['interest_accrued'] = np.random.uniform(0, 5000, n_records).round(2)

    # Account Features
    data['is_joint_account'] = np.random.choice(['YES', 'NO'], n_records, p=[0.15, 0.85])
    data['joint_account_holders'] = [2.0 if joint == 'YES' else 1.0 for joint in data['is_joint_account']]
    data['has_outstanding_facilities'] = np.random.choice(['YES', 'NO'], n_records, p=[0.2, 0.8])
    data['maturity_date'] = pd.date_range('2024-07-01', '2030-01-01', periods=n_records).strftime('%Y-%m-%d')
    data['auto_renewal'] = np.random.choice(['YES', 'NO'], n_records, p=[0.6, 0.4])

    # Statements and Communication
    data['last_statement_date'] = pd.date_range('2024-01-01', '2024-06-01', periods=n_records).strftime('%Y-%m-%d')
    data['statement_frequency'] = np.random.choice(['MONTHLY', 'QUARTERLY', 'ANNUAL'], n_records, p=[0.7, 0.25, 0.05])

    # Dormancy Tracking
    data['tracking_id'] = [f'TRK{str(i).zfill(6)}' for i in range(1, n_records + 1)]
    data['dormancy_trigger_date'] = [pd.Timestamp('2021-01-01') + pd.Timedelta(days=np.random.randint(0, 1000)) if status == 'Dormant' else ''
                                   for status in data['dormancy_status']]
    data['dormancy_period_start'] = data['dormancy_trigger_date'].copy()

    # Calculate dormancy period months
    dormancy_months = []
    for trigger_date in data['dormancy_trigger_date']:
        if trigger_date:
            trigger = pd.Timestamp(trigger_date)
            months = (pd.Timestamp.now() - trigger).days / 30.44
            dormancy_months.append(round(months, 1))
        else:
            dormancy_months.append(0.0)
    data['dormancy_period_months'] = dormancy_months

    data['dormancy_classification_date'] = [pd.Timestamp(trigger) + pd.Timedelta(days=90) if trigger else ''
                                          for trigger in data['dormancy_trigger_date']]
    data['transfer_eligibility_date'] = [pd.Timestamp(trigger) + pd.Timedelta(days=1825) if trigger else ''  # 5 years
                                       for trigger in data['dormancy_trigger_date']]

    # Process Status
    data['current_stage'] = np.random.choice(['CONTACT_ATTEMPTS', 'WAITING_PERIOD', 'READY_FOR_TRANSFER', 'COMPLETED'], n_records, p=[0.4, 0.3, 0.2, 0.1])
    data['contact_attempts_made'] = np.random.randint(0, 6, n_records)
    data['last_contact_attempt_date'] = pd.date_range('2023-01-01', '2024-06-01', periods=n_records).strftime('%Y-%m-%d')
    data['waiting_period_start'] = pd.date_range('2024-01-01', '2024-06-01', periods=n_records).strftime('%Y-%m-%d')
    data['waiting_period_end'] = [pd.Timestamp(start) + pd.Timedelta(days=90) for start in data['waiting_period_start']]
    data['waiting_period_end'] = [date.strftime('%Y-%m-%d') for date in data['waiting_period_end']]

    # Transfer Information
    data['transferred_to_ledger_date'] = [pd.Timestamp('2024-01-01').strftime('%Y-%m-%d') if np.random.random() > 0.9 else '' for _ in range(n_records)]
    data['transferred_to_cb_date'] = [pd.Timestamp('2024-01-01').strftime('%Y-%m-%d') if np.random.random() > 0.98 else '' for _ in range(n_records)]
    data['cb_transfer_amount'] = [bal if transferred else 0.0 for bal, transferred in zip(data['balance_current'], data['transferred_to_cb_date'])]
    data['cb_transfer_reference'] = [f'CBUAE2024{str(i).zfill(6)}' if transferred else ''
                                   for i, transferred in enumerate(data['transferred_to_cb_date'])]
    data['exclusion_reason'] = np.random.choice(['', 'ACTIVE_FACILITIES', 'COURT_ORDER', 'LEGAL_HOLD'], n_records, p=[0.9, 0.05, 0.03, 0.02])

    # System Fields
    data['created_date'] = pd.date_range('2020-01-01', '2024-01-01', periods=n_records)
    data['updated_date'] = pd.date_range('2024-01-01', '2024-06-01', periods=n_records)
    data['updated_by'] = np.random.choice(['SYSTEM', 'USER123', 'BATCH_PROCESS', 'API_UPDATE'], n_records, p=[0.6, 0.2, 0.15, 0.05])

    return pd.DataFrame(data)

# Data Processing Section
def show_data_processing_section():
    """Display data processing interface"""
    st.markdown('<div class="section-header">⚙️ Data Processing</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first in the Data Upload section.")

        # Offer to generate sample data
        if st.button("🎲 Generate Sample Banking Data (5000 records)", type="secondary"):
            with st.spinner("Generating realistic banking compliance data..."):
                sample_data = create_actual_banking_data()
                st.session_state.uploaded_data = sample_data
                st.success("✅ Sample banking data generated successfully!")
                st.rerun()
        return

    data = st.session_state.uploaded_data

    # Data Quality Analysis
    st.markdown("### 🔍 Data Quality Analysis")

    if st.button("🔄 Run Quality Analysis", type="primary"):
        with st.spinner("Analyzing data quality..."):
            quality_results = run_real_data_quality_analysis(data)
            st.session_state.quality_results = quality_results

    if 'quality_results' in st.session_state:
        display_quality_results(st.session_state.quality_results)

    # Data Mapping Section
    st.markdown("### 🗺️ Data Mapping")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("Map your data columns to the banking compliance schema using BGE embeddings:")

    with col2:
        llm_enabled = st.toggle("🤖 Enable LLM", value=st.session_state.llm_enabled,
                               help="Use AI (Groq Llama 3.3 70B) to automatically suggest column mappings")
        st.session_state.llm_enabled = llm_enabled

    if st.button("🗺️ Start Data Mapping", type="primary"):
        with st.spinner("Running BGE embedding-based data mapping..."):
            mapping_results = run_real_data_mapping(data, llm_enabled)
            st.session_state.mapping_results = mapping_results

    if st.session_state.mapping_results:
        display_mapping_results(st.session_state.mapping_results)

def run_real_data_quality_analysis(data):
    """Run real data quality analysis using actual processing agent"""
    try:
        if DATA_PROCESSING_AVAILABLE:
            # Use actual data processing agent
            processor = DataProcessingAgent()

            # Run async method in sync context
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                processor.execute_workflow(
                    user_id=st.session_state.username,
                    data_source=data,
                    processing_options={"quality_check": True}
                )
            )

            if result.get("success"):
                return {
                    'total_records': len(data),
                    'total_columns': len(data.columns),
                    'missing_percentage': result.get("missing_percentage", 0),
                    'missing_by_column': data.isnull().sum().to_dict(),
                    'data_types': data.dtypes.astype(str).to_dict(),
                    'duplicate_records': data.duplicated().sum(),
                    'overall_score': result.get("quality_score", 0) * 100,
                    'quality_level': result.get("quality_level", "unknown"),
                    'validation_results': result.get("validation_results", {}),
                    'recommendations': result.get("recommendations", [])
                }

        # Fallback to manual calculation if agent not available
        return calculate_quality_manually(data)

    except Exception as e:
        logger.error(f"Data quality analysis failed: {e}")
        return calculate_quality_manually(data)

def calculate_quality_manually(data):
    """Manual data quality calculation"""
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data.sum() / (len(data) * len(data.columns))) * 100
    duplicate_records = data.duplicated().sum()

    # Calculate quality score
    completeness_score = 100 - missing_percentage
    uniqueness_score = 100 - (duplicate_records / len(data) * 100)
    overall_score = (completeness_score + uniqueness_score) / 2

    # Determine quality level
    if overall_score >= 90:
        quality_level = "Excellent"
    elif overall_score >= 75:
        quality_level = "Good"
    elif overall_score >= 60:
        quality_level = "Fair"
    else:
        quality_level = "Poor"

    return {
        'total_records': len(data),
        'total_columns': len(data.columns),
        'missing_percentage': missing_percentage,
        'missing_by_column': missing_data.to_dict(),
        'data_types': data.dtypes.astype(str).to_dict(),
        'duplicate_records': duplicate_records,
        'overall_score': overall_score,
        'quality_level': quality_level,
        'validation_results': {},
        'recommendations': []
    }

def display_quality_results(results):
    """Display data quality results"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{results['total_records']:,}")

    with col2:
        st.metric("Missing Data %", f"{results['missing_percentage']:.1f}%")

    with col3:
        st.metric("Duplicate Records", f"{results['duplicate_records']:,}")

    with col4:
        st.metric("Quality Score", f"{results['overall_score']:.1f}", delta=results['quality_level'])

    # Quality level indicator
    quality_level = results['quality_level']
    if quality_level == "Excellent":
        st.success(f"✅ Data Quality: {quality_level}")
    elif quality_level == "Good":
        st.info(f"ℹ️ Data Quality: {quality_level}")
    elif quality_level == "Fair":
        st.warning(f"⚠️ Data Quality: {quality_level}")
    else:
        st.error(f"❌ Data Quality: {quality_level}")

    # Missing data details
    if results['missing_percentage'] > 0:
        with st.expander("📊 Missing Data Details"):
            missing_df = pd.DataFrame([
                {'Column': col, 'Missing Count': count, 'Missing %': (count/results['total_records']*100)}
                for col, count in results['missing_by_column'].items() if count > 0
            ])

            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)

    # Show recommendations if available
    if results.get('recommendations'):
        with st.expander("💡 Recommendations"):
            for rec in results['recommendations']:
                st.write(f"• {rec}")

def run_real_data_mapping(data, llm_enabled=False):
    """Run real data mapping using BGE embeddings"""
    try:
        if DATA_AGENTS_AVAILABLE:
            # Get Groq API key if LLM enabled
            groq_api_key = None
            if llm_enabled:
                try:
                    groq_api_key = st.secrets.get("GROQ_API_KEY")
                    if not groq_api_key:
                        st.warning("⚠️ LLM enabled but no Groq API key found in secrets. Using BGE embeddings only.")
                except:
                    st.warning("⚠️ LLM enabled but no API key found. Using BGE embeddings only.")

            # Run async mapping in sync context
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            mapping_result = loop.run_until_complete(
                run_automated_data_mapping(
                    source_data=data,
                    user_id=st.session_state.username,
                    groq_api_key=groq_api_key
                )
            )

            return mapping_result
        else:
            st.error("❌ Data mapping agent not available")
            return None

    except Exception as e:
        st.error(f"❌ Data mapping failed: {str(e)}")
        logger.error(f"Data mapping error: {str(e)}")
        return None

def display_mapping_results(results):
    """Display mapping results"""
    if results and results.get('success'):
        col1, col2 = st.columns([3, 1])

        with col1:
            auto_pct = results.get('auto_mapping_percentage', 0)
            if auto_pct >= 90:
                st.success(f"✅ Excellent mapping! {auto_pct:.1f}% auto-mapped (≥90%)")
            elif auto_pct >= 70:
                st.warning(f"⚠️ Good mapping: {auto_pct:.1f}% auto-mapped (70-89%)")
            else:
                st.error(f"❌ Low mapping: {auto_pct:.1f}% auto-mapped (<70%)")

        with col2:
            if 'mapping_sheet' in results:
                mapping_sheet = results['mapping_sheet']
                csv_data = mapping_sheet.to_csv(index=False)
                st.download_button(
                    "📥 Download Mapping Sheet",
                    csv_data,
                    "column_mapping_results.csv",
                    "text/csv",
                    use_container_width=True
                )

        # Display mapping sheet with confidence highlighting
        if 'mapping_sheet' in results:
            st.markdown("#### 📋 Column Mapping Results")

            mapping_df = results['mapping_sheet'].copy()

            # Apply styling based on confidence levels
            def highlight_confidence(row):
                if row['Confidence_Level'] == 'high':
                    return ['background-color: #d4edda'] * len(row)
                elif row['Confidence_Level'] == 'medium':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)

            styled_df = mapping_df.style.apply(highlight_confidence, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            # Show mapping statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                high_count = len(mapping_df[mapping_df['Confidence_Level'] == 'high'])
                st.metric("High Confidence", high_count, delta="Auto-mapped")

            with col2:
                medium_count = len(mapping_df[mapping_df['Confidence_Level'] == 'medium'])
                st.metric("Medium Confidence", medium_count, delta="Review needed")

            with col3:
                low_count = len(mapping_df[mapping_df['Confidence_Level'] == 'low'])
                st.metric("Low Confidence", low_count, delta="Manual mapping")

            with col4:
                avg_similarity = mapping_df['Similarity_Score'].mean()
                st.metric("Avg Similarity", f"{avg_similarity:.3f}")

            # Show message and next steps
            st.info(results.get('message', 'Mapping completed'))

            # Manual mapping interface if needed
            if results.get('user_action_required') and not st.session_state.llm_enabled:
                show_manual_mapping_interface(mapping_df)
            elif results.get('user_action_required') and st.session_state.llm_enabled:
                if st.button("🤖 Apply LLM Assistance", type="secondary"):
                    apply_llm_assistance_to_mapping(results)

def show_manual_mapping_interface(mapping_sheet):
    """Show manual mapping interface for low confidence fields"""
    st.markdown("#### ✏️ Manual Column Mapping")

    low_confidence_fields = mapping_sheet[mapping_sheet['Confidence_Level'] == 'low']

    if len(low_confidence_fields) > 0:
        st.info(f"Please review and adjust {len(low_confidence_fields)} low confidence mappings:")

        # Target schema columns
        target_columns = [
            'customer_id', 'customer_type', 'full_name_en', 'full_name_ar', 'id_number',
            'account_id', 'account_type', 'account_status', 'dormancy_status',
            'balance_current', 'last_transaction_date', 'phone_primary', 'email_primary'
        ]

        manual_mappings = {}

        for idx, row in low_confidence_fields.iterrows():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.text(f"Source: {row['Source_Field']}")

            with col2:
                selected_target = st.selectbox(
                    f"Map to:",
                    ['unmapped'] + target_columns,
                    index=0,
                    key=f"mapping_{idx}"
                )
                if selected_target != 'unmapped':
                    manual_mappings[row['Source_Field']] = selected_target

            with col3:
                st.text(f"Score: {row['Similarity_Score']:.3f}")

        if manual_mappings and st.button("💾 Apply Manual Mappings"):
            st.success(f"✅ Applied {len(manual_mappings)} manual mappings")
            st.session_state.manual_mappings = manual_mappings
    else:
        st.success("✅ No manual mapping required - all fields have sufficient confidence!")

def apply_llm_assistance_to_mapping(mapping_results):
    """Apply LLM assistance to improve mapping results"""
    with st.spinner("Applying LLM assistance to improve mappings..."):
        try:
            if DATA_AGENTS_AVAILABLE:
                mapping_agent = create_data_mapping_agent(groq_api_key=st.secrets.get("GROQ_API_KEY"))

                # Run async LLM assistance
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                llm_result = loop.run_until_complete(
                    apply_llm_assistance(mapping_agent, mapping_results["mapping_state"])
                )

                if llm_result.get("success"):
                    st.session_state.mapping_results = llm_result
                    st.success(f"🤖 {llm_result.get('message', 'LLM assistance applied successfully')}")
                    st.rerun()
                else:
                    st.error(f"❌ LLM assistance failed: {llm_result.get('error')}")
            else:
                st.error("❌ Data mapping agent not available for LLM assistance")

        except Exception as e:
            st.error(f"❌ LLM assistance error: {str(e)}")

# Dormant Account Analysis Section
def show_dormant_analysis_section():
    """Display dormant account analysis using real agents"""
    st.markdown('<div class="section-header">😴 Dormant Account Analysis</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload and process data first.")
        return

    data = st.session_state.uploaded_data

    # Check for dormant accounts
    dormant_count = 0
    if 'dormancy_status' in data.columns:
        dormant_count = len(data[data['dormancy_status'] == 'Dormant'])
    elif 'account_status' in data.columns:
        dormant_count = len(data[data['account_status'] == 'DORMANT'])

    if dormant_count == 0:
        st.info("ℹ️ No dormant accounts detected in the current dataset.")
        return

    st.success(f"📊 Found {dormant_count:,} dormant accounts for analysis")

    # Available dormancy agents with real functionality
    dormancy_agents = [
        {
            'name': 'Demand Deposit Dormancy',
            'description': 'Analyzes dormancy for demand deposit accounts (savings, current)',
            'article': 'CBUAE Art. 2.1.1',
            'agent_class': DemandDepositDormancyAgent if DORMANCY_AGENTS_AVAILABLE else None,
            'available': DORMANCY_AGENTS_AVAILABLE
        },
        {
            'name': 'Fixed Deposit Dormancy',
            'description': 'Analyzes dormancy for fixed/term deposit accounts',
            'article': 'CBUAE Art. 2.2',
            'agent_class': FixedDepositDormancyAgent if DORMANCY_AGENTS_AVAILABLE else None,
            'available': DORMANCY_AGENTS_AVAILABLE
        },
        {
            'name': 'Investment Account Dormancy',
            'description': 'Analyzes dormancy for investment accounts',
            'article': 'CBUAE Art. 2.3',
            'agent_class': InvestmentAccountDormancyAgent if DORMANCY_AGENTS_AVAILABLE else None,
            'available': DORMANCY_AGENTS_AVAILABLE
        },
        {
            'name': 'Contact Attempts Analysis',
            'description': 'Analyzes customer contact attempt compliance',
            'article': 'CBUAE Art. 5',
            'agent_class': ContactAttemptsAgent if DORMANCY_AGENTS_AVAILABLE else None,
            'available': DORMANCY_AGENTS_AVAILABLE
        },
        {
            'name': 'CB Transfer Eligibility',
            'description': 'Identifies accounts eligible for Central Bank transfer',
            'article': 'CBUAE Art. 8.1-8.2',
            'agent_class': CBTransferEligibilityAgent if DORMANCY_AGENTS_AVAILABLE else None,
            'available': DORMANCY_AGENTS_AVAILABLE
        },
        {
            'name': 'High Value Dormant Accounts',
            'description': 'Identifies high-value dormant accounts requiring special attention',
            'article': 'Internal Policy',
            'agent_class': HighValueDormantAccountsAgent if DORMANCY_AGENTS_AVAILABLE else None,
            'available': DORMANCY_AGENTS_AVAILABLE
        }
    ]

    # Display agents with real analysis capability
    for agent in dormancy_agents:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{agent['name']}**")
                st.caption(f"{agent['description']} ({agent['article']})")
                if not agent['available']:
                    st.caption("❌ Agent not available - check imports")

            with col2:
                if agent['available'] and st.button(f"🔍 Analyze", key=f"dormant_{agent['name']}"):
                    run_real_dormancy_analysis(agent, data)

            with col3:
                if agent['name'] in st.session_state.dormancy_results:
                    results = st.session_state.dormancy_results[agent['name']]
                    if results.get('success') and results.get('csv_data'):
                        st.download_button(
                            "📥 CSV",
                            results['csv_data'],
                            f"dormant_{agent['name'].lower().replace(' ', '_')}.csv",
                            key=f"download_dormant_{agent['name']}"
                        )

    # Run comprehensive analysis
    if DORMANCY_AGENTS_AVAILABLE:
        st.markdown("---")
        if st.button("🚀 Run Comprehensive Dormancy Analysis", type="primary"):
            run_comprehensive_dormancy_analysis(data)

    # Display results
    if st.session_state.dormancy_results:
        display_dormancy_results(st.session_state.dormancy_results)

def run_real_dormancy_analysis(agent_info, data):
    """Run real dormancy analysis using actual agents"""
    with st.spinner(f"Running {agent_info['name']} analysis..."):
        try:
            if agent_info['available'] and agent_info['agent_class']:
                # Initialize the actual agent
                agent = agent_info['agent_class']()

                # Run analysis using the real agent
                # Note: This assumes the agents have a standard interface
                # You may need to adjust based on actual agent implementation
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Create agent state and run analysis
                from agents.Dormant_agent import AgentState

                state = AgentState(
                    session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    user_id=st.session_state.username,
                    input_dataframe=data,
                    agent_name=agent_info['name']
                )

                result_state = loop.run_until_complete(
                    agent.analyze_dormancy(state, datetime.now().strftime('%Y-%m-%d'))
                )

                if result_state.agent_status.name == 'COMPLETED':
                    results_df = result_state.processed_dataframe
                    csv_data = results_df.to_csv(index=False) if not results_df.empty else ""

                    st.session_state.dormancy_results[agent_info['name']] = {
                        'success': True,
                        'count': result_state.dormant_records_found,
                        'data': results_df,
                        'csv_data': csv_data,
                        'summary': f"Found {result_state.dormant_records_found} records using {agent_info['name']}",
                        'analysis_results': result_state.analysis_results
                    }

                    st.success(f"✅ {agent_info['name']} analysis completed! Found {result_state.dormant_records_found} records.")
                else:
                    st.error(f"❌ {agent_info['name']} analysis failed")
            else:
                st.error(f"❌ {agent_info['name']} agent not available")

        except Exception as e:
            logger.error(f"Dormancy analysis failed for {agent_info['name']}: {e}")
            st.error(f"❌ Analysis failed: {str(e)}")

def run_comprehensive_dormancy_analysis(data):
    """Run comprehensive dormancy analysis using all available agents"""
    with st.spinner("Running comprehensive dormancy analysis..."):
        try:
            if DORMANCY_AGENTS_AVAILABLE:
                # Use the comprehensive analysis function from the agents
                result = run_comprehensive_dormancy_analysis_csv(
                    data,
                    user_id=st.session_state.username,
                    report_date=datetime.now().strftime('%Y-%m-%d')
                )

                if result.get("success"):
                    st.session_state.comprehensive_dormancy_results = result
                    st.success(f"✅ Comprehensive analysis completed! Processed {result.get('total_records', 0)} records.")

                    # Store individual agent results
                    for agent_name, agent_result in result.get("agent_results", {}).items():
                        st.session_state.dormancy_results[agent_name] = agent_result

                else:
                    st.error(f"❌ Comprehensive analysis failed: {result.get('error')}")
            else:
                st.error("❌ Dormancy agents not available for comprehensive analysis")

        except Exception as e:
            logger.error(f"Comprehensive dormancy analysis failed: {e}")
            st.error(f"❌ Comprehensive analysis error: {str(e)}")

def display_dormancy_results(results):
    """Display dormancy analysis results"""
    st.markdown("### 📊 Dormancy Analysis Results")

    # Summary metrics
    total_analyzed = sum(r.get('count', 0) for r in results.values() if r.get('success'))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records Analyzed", f"{total_analyzed:,}")
    with col2:
        st.metric("Agents Executed", len([r for r in results.values() if r.get('success')]))
    with col3:
        high_risk_count = 0
        for result in results.values():
            if result.get('success') and 'data' in result:
                df = result['data']
                if 'Risk_Level' in df.columns:
                    high_risk_count += len(df[df['Risk_Level'] == 'HIGH'])
        st.metric("High Risk Accounts", high_risk_count)

    # Detailed results for each agent
    for agent_name, result in results.items():
        if result.get('success'):
            with st.expander(f"📋 {agent_name} Results ({result.get('count', 0)} records)"):
                if 'data' in result and not result['data'].empty:
                    st.dataframe(result['data'], use_container_width=True)

                    # Show analysis summary if available
                    if 'analysis_results' in result:
                        analysis = result['analysis_results']
                        st.markdown("**Analysis Summary:**")
                        st.write(f"• Description: {analysis.get('description', 'N/A')}")
                        st.write(f"• Compliance Article: {analysis.get('compliance_article', 'N/A')}")
                        st.write(f"• Validation Passed: {analysis.get('validation_passed', 'N/A')}")
                        if analysis.get('alerts_generated'):
                            st.warning("⚠️ Compliance alerts generated")

                st.caption(result.get('summary', ''))

# Compliance Analysis Section
def show_compliance_analysis_section():
    """Display compliance analysis using real agents"""
    st.markdown('<div class="section-header">⚖️ Compliance Analysis</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload and process data first.")
        return

    data = st.session_state.uploaded_data

    # Available compliance agents with real functionality
    compliance_agents = [
        {
            'name': 'Incomplete Contact Attempts',
            'description': 'Detects accounts with incomplete contact attempt processes',
            'article': 'CBUAE Art. 5',
            'applies_to': 'Dormant accounts with customer contact requirements',
            'agent_class': DetectIncompleteContactAttemptsAgent if COMPLIANCE_AGENTS_AVAILABLE else None,
            'available': COMPLIANCE_AGENTS_AVAILABLE
        },
        {
            'name': 'Unflagged Dormant Candidates',
            'description': 'Identifies accounts that should be flagged as dormant but are not',
            'article': 'CBUAE Art. 2.x',
            'applies_to': 'All account types with dormancy criteria',
            'agent_class': DetectUnflaggedDormantCandidatesAgent if COMPLIANCE_AGENTS_AVAILABLE else None,
            'available': COMPLIANCE_AGENTS_AVAILABLE
        },
        {
            'name': 'Internal Ledger Candidates',
            'description': 'Identifies accounts ready for internal ledger transfer',
            'article': 'CBUAE Art. 3',
            'applies_to': 'Dormant accounts after contact attempts',
            'agent_class': DetectInternalLedgerCandidatesAgent if COMPLIANCE_AGENTS_AVAILABLE else None,
            'available': COMPLIANCE_AGENTS_AVAILABLE
        },
        {
            'name': 'Statement Freeze Candidates',
            'description': 'Identifies accounts eligible for statement suppression',
            'article': 'CBUAE Art. 7.3',
            'applies_to': 'Dormant accounts meeting statement freeze criteria',
            'agent_class': DetectStatementFreezeCandidatesAgent if COMPLIANCE_AGENTS_AVAILABLE else None,
            'available': COMPLIANCE_AGENTS_AVAILABLE
        }
    ]

    st.info("🔍 Compliance agents analyze data for regulatory compliance issues.")

    # Display agents with real analysis capability
    for agent in compliance_agents:
        with st.container():
            st.markdown(f"#### {agent['name']}")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(agent['description'])
                st.caption(f"**Applies to:** {agent['applies_to']} | **Reference:** {agent['article']}")
                if not agent['available']:
                    st.caption("❌ Agent not available - check imports")

            with col2:
                col_analyze, col_download = st.columns(2)

                with col_analyze:
                    if agent['available'] and st.button("🔍 Analyze", key=f"compliance_{agent['name']}"):
                        run_real_compliance_analysis(agent, data)

                with col_download:
                    if agent['name'] in st.session_state.compliance_results:
                        results = st.session_state.compliance_results[agent['name']]
                        if results.get('success') and results.get('csv_data'):
                            st.download_button(
                                "📥 CSV",
                                results['csv_data'],
                                f"compliance_{agent['name'].lower().replace(' ', '_')}.csv",
                                key=f"download_compliance_{agent['name']}"
                            )

    # Run comprehensive compliance analysis
    if COMPLIANCE_AGENTS_AVAILABLE:
        st.markdown("---")
        if st.button("🚀 Run Comprehensive Compliance Analysis", type="primary"):
            run_comprehensive_compliance_analysis(data)

    # Display results
    if st.session_state.compliance_results:
        display_compliance_results(st.session_state.compliance_results)

def run_real_compliance_analysis(agent_info, data):
    """Run real compliance analysis using actual agents"""
    with st.spinner(f"Running {agent_info['name']} analysis..."):
        try:
            if agent_info['available'] and agent_info['agent_class']:
                # Initialize the actual compliance agent
                agent = agent_info['agent_class']()

                # Run analysis using the real agent
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Create agent state and run analysis
                from agents.compliance_verification_agent import AgentState

                state = AgentState(
                    session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    user_id=st.session_state.username,
                    input_dataframe=data,
                    agent_name=agent_info['name']
                )

                result_state = loop.run_until_complete(
                    agent.analyze_compliance(state, datetime.now().strftime('%Y-%m-%d'))
                )

                if result_state.agent_status.name == 'COMPLETED':
                    results_df = result_state.processed_dataframe
                    csv_data = results_df.to_csv(index=False) if not results_df.empty else ""

                    st.session_state.compliance_results[agent_info['name']] = {
                        'success': True,
                        'count': result_state.compliance_issues_found,
                        'data': results_df,
                        'csv_data': csv_data,
                        'summary': f"Found {result_state.compliance_issues_found} compliance issues",
                        'analysis_results': result_state.analysis_results
                    }

                    st.success(f"✅ {agent_info['name']} analysis completed! Found {result_state.compliance_issues_found} issues.")
                else:
                    st.error(f"❌ {agent_info['name']} analysis failed")
            else:
                st.error(f"❌ {agent_info['name']} agent not available")

        except Exception as e:
            logger.error(f"Compliance analysis failed for {agent_info['name']}: {e}")
            st.error(f"❌ Analysis failed: {str(e)}")

def run_comprehensive_compliance_analysis(data):
    """Run comprehensive compliance analysis using all available agents"""
    with st.spinner("Running comprehensive compliance analysis..."):
        try:
            if COMPLIANCE_AGENTS_AVAILABLE:
                # Use the comprehensive analysis function from the agents
                result = run_comprehensive_compliance_analysis_csv(
                    data,
                    user_id=st.session_state.username,
                    report_date=datetime.now().strftime('%Y-%m-%d')
                )

                if result.get("success"):
                    st.session_state.comprehensive_compliance_results = result
                    st.success(f"✅ Comprehensive compliance analysis completed! Found {result.get('total_issues', 0)} issues.")

                    # Store individual agent results
                    for agent_name, agent_result in result.get("agent_results", {}).items():
                        st.session_state.compliance_results[agent_name] = agent_result

                else:
                    st.error(f"❌ Comprehensive compliance analysis failed: {result.get('error')}")
            else:
                st.error("❌ Compliance agents not available for comprehensive analysis")

        except Exception as e:
            logger.error(f"Comprehensive compliance analysis failed: {e}")
            st.error(f"❌ Comprehensive compliance analysis error: {str(e)}")

def display_compliance_results(results):
    """Display compliance analysis results"""
    st.markdown("### ⚖️ Compliance Analysis Results")

    # Summary metrics
    total_issues = sum(r.get('count', 0) for r in results.values() if r.get('success'))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Issues Found", f"{total_issues:,}")
    with col2:
        st.metric("Compliance Checks", len([r for r in results.values() if r.get('success')]))
    with col3:
        critical_count = 0
        for result in results.values():
            if result.get('success') and 'data' in result:
                df = result['data']
                if 'Severity' in df.columns:
                    critical_count += len(df[df['Severity'] == 'CRITICAL'])
        st.metric("Critical Issues", critical_count)

    # Detailed results for each agent
    for agent_name, result in results.items():
        if result.get('success'):
            with st.expander(f"⚖️ {agent_name} Results ({result.get('count', 0)} issues)"):
                if 'data' in result and not result['data'].empty:
                    st.dataframe(result['data'], use_container_width=True)

                    # Show analysis summary if available
                    if 'analysis_results' in result:
                        analysis = result['analysis_results']
                        st.markdown("**Analysis Summary:**")
                        st.write(f"• Description: {analysis.get('description', 'N/A')}")
                        st.write(f"• Compliance Article: {analysis.get('compliance_article', 'N/A')}")
                        st.write(f"• Validation Passed: {analysis.get('validation_passed', 'N/A')}")
                        if analysis.get('alerts_generated'):
                            st.warning("⚠️ Compliance alerts generated")

                st.caption(result.get('summary', ''))

# Reports Section
def show_reports_section():
    """Display comprehensive reports"""
    st.markdown('<div class="section-header">📊 Comprehensive Reports</div>', unsafe_allow_html=True)

    # Summary dashboard
    create_summary_dashboard()

    # Agent status overview
    st.markdown("### 🤖 Agent Status Overview")

    agent_status_data = []

    # Data processing agents
    data_agents = ['Data Upload', 'Data Quality', 'Data Mapping']
    for agent in data_agents:
        status = "Available" if DATA_AGENTS_AVAILABLE else "Not Available"
        accounts_processed = 0

        if agent == "Data Mapping" and st.session_state.mapping_results:
            accounts_processed = st.session_state.mapping_results.get('mapping_state', {}).get('total_fields', 0)
            status = "Completed"

        agent_status_data.append({
            'Agent': agent,
            'Category': 'Data Processing',
            'Records Processed': accounts_processed,
            'Status': status,
            'Actions': 'Download Results' if accounts_processed > 0 else 'Run Analysis'
        })

    # Dormancy agents
    dormancy_agents = [
        'Demand Deposit Dormancy', 'Fixed Deposit Dormancy', 'Investment Account Dormancy',
        'Contact Attempts Analysis', 'CB Transfer Eligibility', 'High Value Dormant Accounts'
    ]

    for agent in dormancy_agents:
        accounts_processed = 0
        status = "Available" if DORMANCY_AGENTS_AVAILABLE else "Not Available"

        if agent in st.session_state.dormancy_results:
            accounts_processed = st.session_state.dormancy_results[agent].get('count', 0)
            status = "Completed"

        agent_status_data.append({
            'Agent': agent,
            'Category': 'Dormancy',
            'Records Processed': accounts_processed,
            'Status': status,
            'Actions': 'Download CSV, View Summary' if accounts_processed > 0 else 'Run Analysis'
        })

    # Compliance agents
    compliance_agents = [
        'Incomplete Contact Attempts', 'Unflagged Dormant Candidates',
        'Internal Ledger Candidates', 'Statement Freeze Candidates'
    ]

    for agent in compliance_agents:
        issues_found = 0
        status = "Available" if COMPLIANCE_AGENTS_AVAILABLE else "Not Available"

        if agent in st.session_state.compliance_results:
            issues_found = st.session_state.compliance_results[agent].get('count', 0)
            status = "Completed"

        agent_status_data.append({
            'Agent': agent,
            'Category': 'Compliance',
            'Records Processed': issues_found,
            'Status': status,
            'Actions': 'Download CSV, View Details' if issues_found > 0 else 'Run Analysis'
        })

    # Display agent status table
    agent_df = pd.DataFrame(agent_status_data)
    st.dataframe(agent_df, use_container_width=True)

    # Export options
    st.markdown("### 📥 Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export Summary Report"):
            summary_report = generate_summary_report()
            st.download_button(
                "📥 Download Summary (CSV)",
                summary_report,
                "banking_compliance_summary.csv",
                "text/csv"
            )

    with col2:
        if st.button("🔍 Export Dormancy Results"):
            dormancy_export = export_dormancy_results()
            if dormancy_export:
                st.download_button(
                    "📥 Download Dormancy (CSV)",
                    dormancy_export,
                    "dormancy_analysis_results.csv",
                    "text/csv"
                )

    with col3:
        if st.button("⚖️ Export Compliance Results"):
            compliance_export = export_compliance_results()
            if compliance_export:
                st.download_button(
                    "📥 Download Compliance (CSV)",
                    compliance_export,
                    "compliance_analysis_results.csv",
                    "text/csv"
                )

def create_summary_dashboard():
    """Create summary dashboard with key metrics"""
    st.markdown("### 📈 Executive Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_accounts = len(st.session_state.uploaded_data) if st.session_state.uploaded_data is not None else 0
        st.metric("Total Accounts", f"{total_accounts:,}")

    with col2:
        dormant_analyzed = sum(r.get('count', 0) for r in st.session_state.dormancy_results.values())
        st.metric("Dormant Analyzed", f"{dormant_analyzed:,}")

    with col3:
        compliance_issues = sum(r.get('count', 0) for r in st.session_state.compliance_results.values())
        st.metric("Compliance Issues", f"{compliance_issues:,}")

    with col4:
        agents_run = len(st.session_state.dormancy_results) + len(st.session_state.compliance_results)
        st.metric("Agents Executed", agents_run)

    # Charts
    if st.session_state.uploaded_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            # Account type distribution
            if 'account_type' in st.session_state.uploaded_data.columns:
                account_dist = st.session_state.uploaded_data['account_type'].value_counts()
                fig = px.pie(values=account_dist.values, names=account_dist.index,
                           title="Account Type Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Dormancy status distribution
            if 'dormancy_status' in st.session_state.uploaded_data.columns:
                dormancy_dist = st.session_state.uploaded_data['dormancy_status'].value_counts()
                fig = px.bar(x=dormancy_dist.index, y=dormancy_dist.values,
                           title="Dormancy Status Distribution")
                st.plotly_chart(fig, use_container_width=True)

def generate_summary_report():
    """Generate summary report"""
    summary_data = []

    # Add overall statistics
    total_accounts = len(st.session_state.uploaded_data) if st.session_state.uploaded_data is not None else 0
    dormant_analyzed = sum(r.get('count', 0) for r in st.session_state.dormancy_results.values())
    compliance_issues = sum(r.get('count', 0) for r in st.session_state.compliance_results.values())

    summary_data.append({
        'Metric': 'Total Accounts',
        'Value': total_accounts,
        'Category': 'Overview'
    })

    summary_data.append({
        'Metric': 'Dormant Accounts Analyzed',
        'Value': dormant_analyzed,
        'Category': 'Dormancy'
    })

    summary_data.append({
        'Metric': 'Compliance Issues Found',
        'Value': compliance_issues,
        'Category': 'Compliance'
    })

    return pd.DataFrame(summary_data).to_csv(index=False)

def export_dormancy_results():
    """Export all dormancy results"""
    if not st.session_state.dormancy_results:
        return None

    all_results = []
    for agent_name, results in st.session_state.dormancy_results.items():
        if results.get('success') and 'data' in results:
            agent_data = results['data'].copy()
            agent_data['Analysis_Agent'] = agent_name
            all_results.append(agent_data)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df.to_csv(index=False)

    return None

def export_compliance_results():
    """Export all compliance results"""
    if not st.session_state.compliance_results:
        return None

    all_results = []
    for agent_name, results in st.session_state.compliance_results.items():
        if results.get('success') and 'data' in results:
            agent_data = results['data'].copy()
            agent_data['Analysis_Agent'] = agent_name
            all_results.append(agent_data)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df.to_csv(index=False)

    return None

# Sidebar navigation
def show_sidebar():
    """Display sidebar navigation"""
    st.sidebar.title("🏦 Banking Compliance")
    st.sidebar.markdown(f"Welcome, **{st.session_state.username}**!")

    # Navigation
    pages = [
        "📤 Data Processing",
        "😴 Dormant Analysis",
        "⚖️ Compliance Analysis",
        "📊 Reports"
    ]

    selected_page = st.sidebar.selectbox("Navigate to:", pages,
                                       index=pages.index(f"📤 {st.session_state.current_page}") if f"📤 {st.session_state.current_page}" in pages else 0)

    st.session_state.current_page = selected_page.split(" ", 1)[1]

    # Data status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Status")

    if st.session_state.uploaded_data is not None:
        st.sidebar.success(f"✅ Data Loaded: {len(st.session_state.uploaded_data):,} records")

        # Show dormancy statistics
        if 'dormancy_status' in st.session_state.uploaded_data.columns:
            dormant_count = len(st.session_state.uploaded_data[st.session_state.uploaded_data['dormancy_status'] == 'Dormant'])
            st.sidebar.info(f"💤 Dormant Accounts: {dormant_count:,}")
    else:
        st.sidebar.warning("⚠️ No data loaded")

    if st.session_state.mapping_results:
        auto_pct = st.session_state.mapping_results.get('auto_mapping_percentage', 0)
        st.sidebar.info(f"🗺️ Mapping: {auto_pct:.1f}% auto-mapped")

    # Agent status
    st.sidebar.markdown("### 🤖 Agent Status")
    st.sidebar.info(f"Data: {'✅' if DATA_AGENTS_AVAILABLE else '❌'}")
    st.sidebar.info(f"Dormancy: {'✅' if DORMANCY_AGENTS_AVAILABLE else '❌'}")
    st.sidebar.info(f"Compliance: {'✅' if COMPLIANCE_AGENTS_AVAILABLE else '❌'}")

    # System information
    st.sidebar.markdown("### ℹ️ System Info")
    st.sidebar.caption(f"BGE Embeddings: {'✅ Available' if DATA_AGENTS_AVAILABLE else '❌ Unavailable'}")
    st.sidebar.caption(f"Real-time Analysis: {'✅ Enabled' if DORMANCY_AGENTS_AVAILABLE else '❌ Mock Mode'}")

    # Logout
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main application
def main():
    """Main application logic"""
    initialize_session_state()

    if not st.session_state.logged_in:
        show_login_page()
        return

    # Show sidebar
    show_sidebar()

    # Main content area
    current_page = st.session_state.current_page

    if current_page == "Data Processing":
        # Data upload section
        uploaded_data = show_data_upload_section()

        if uploaded_data is not None or st.session_state.uploaded_data is not None:
            st.markdown("---")
            # Data processing section
            show_data_processing_section()

    elif current_page == "Dormant Analysis":
        show_dormant_analysis_section()

    elif current_page == "Compliance Analysis":
        show_compliance_analysis_section()

    elif current_page == "Reports":
        show_reports_section()

if __name__ == "__main__":
    main()