"""
Comprehensive Banking Compliance Analysis Streamlit Application
Integrates all dormancy and compliance agents with full CSV export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import all agents from repository
try:
    # Dormancy Agents (10 agents)
    from agents.dormant_agent import (
        DormancyWorkflowOrchestrator,
        run_comprehensive_dormancy_analysis_with_csv,
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
        get_all_csv_download_info
    )
    DORMANCY_AGENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Dormancy agents not available: {e}")
    DORMANCY_AGENTS_AVAILABLE = False

try:
    # Compliance Agents (17 agents)
    from agents.compliance_verification_agent import (
        ComplianceWorkflowOrchestrator,
        run_comprehensive_compliance_analysis_with_csv,
        DetectIncompleteContactAttemptsAgent,
        DetectUnflaggedDormantCandidatesAgent,
        DetectStatementFreezeCandidatesAgent,
        DetectForeignCurrencyConversionNeededAgent,
        DetectSDBCourtApplicationNeededAgent,
        DetectUnclaimedPaymentInstrumentsLedgerAgent,
        DetectClaimProcessingPendingAgent,
        GenerateAnnualCBUAEReportSummaryAgent,
        CheckRecordRetentionComplianceAgent,
        LogFlagInstructionsAgent,
        get_all_compliance_csv_download_info,
        get_all_compliance_agents_info
    )
    COMPLIANCE_AGENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Compliance agents not available: {e}")
    COMPLIANCE_AGENTS_AVAILABLE = False

try:
    # Data Processing Agents
    from agents.Data_Process import (
        UnifiedDataProcessingAgent,
        DataQualityAnalyzer,
        UploadResult,
        QualityResult,
        MappingResult
    )
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Data processing agents not available: {e}")
    DATA_PROCESSING_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🏛️ CBUAE Banking Compliance System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f4e79;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem;
        border-left: 4px solid #2e8b57;
        background-color: #f0f8ff;
    }
    
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .download-button {
        background-color: #28a745;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .status-success { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Banking Compliance Schema (66 fields from project knowledge)
BANKING_SCHEMA = {
    'customer_id': {'required': True, 'type': 'string', 'description': 'Unique customer identifier'},
    'customer_type': {'required': False, 'type': 'string', 'description': 'Type of customer (Individual/Corporate)'},
    'full_name_en': {'required': False, 'type': 'string', 'description': 'Customer full name in English'},
    'full_name_ar': {'required': False, 'type': 'string', 'description': 'Customer full name in Arabic'},
    'id_number': {'required': False, 'type': 'integer', 'description': 'Customer ID number'},
    'id_type': {'required': False, 'type': 'string', 'description': 'Type of ID document'},
    'date_of_birth': {'required': False, 'type': 'date', 'description': 'Customer date of birth'},
    'nationality': {'required': False, 'type': 'string', 'description': 'Customer nationality'},
    'address_line1': {'required': False, 'type': 'string', 'description': 'Primary address line'},
    'address_line2': {'required': False, 'type': 'string', 'description': 'Secondary address line'},
    'city': {'required': False, 'type': 'string', 'description': 'City'},
    'emirate': {'required': False, 'type': 'string', 'description': 'Emirate'},
    'country': {'required': False, 'type': 'string', 'description': 'Country'},
    'postal_code': {'required': False, 'type': 'integer', 'description': 'Postal code'},
    'phone_primary': {'required': False, 'type': 'float', 'description': 'Primary phone number'},
    'phone_secondary': {'required': False, 'type': 'float', 'description': 'Secondary phone number'},
    'email_primary': {'required': False, 'type': 'string', 'description': 'Primary email address'},
    'email_secondary': {'required': False, 'type': 'string', 'description': 'Secondary email address'},
    'address_known': {'required': False, 'type': 'string', 'description': 'Address verification status'},
    'last_contact_date': {'required': True, 'type': 'date', 'description': 'Date of last customer contact'},
    'last_contact_method': {'required': False, 'type': 'string', 'description': 'Method of last contact'},
    'kyc_status': {'required': False, 'type': 'string', 'description': 'KYC compliance status'},
    'kyc_expiry_date': {'required': False, 'type': 'date', 'description': 'KYC expiry date'},
    'risk_rating': {'required': False, 'type': 'string', 'description': 'Customer risk rating'},
    'account_id': {'required': True, 'type': 'string', 'description': 'Unique account identifier'},
    'account_type': {'required': True, 'type': 'string', 'description': 'Type of account'},
    'account_subtype': {'required': False, 'type': 'string', 'description': 'Account subtype'},
    'account_name': {'required': False, 'type': 'string', 'description': 'Account name'},
    'currency': {'required': True, 'type': 'string', 'description': 'Account currency'},
    'account_status': {'required': True, 'type': 'string', 'description': 'Account status'},
    'dormancy_status': {'required': True, 'type': 'string', 'description': 'Dormancy classification'},
    'opening_date': {'required': True, 'type': 'date', 'description': 'Account opening date'},
    'closing_date': {'required': False, 'type': 'date', 'description': 'Account closing date'},
    'last_transaction_date': {'required': True, 'type': 'date', 'description': 'Date of last transaction'},
    'last_system_transaction_date': {'required': False, 'type': 'date', 'description': 'Date of last system transaction'},
    'balance_current': {'required': True, 'type': 'float', 'description': 'Current account balance'},
    'balance_available': {'required': False, 'type': 'float', 'description': 'Available balance'},
    'balance_minimum': {'required': False, 'type': 'integer', 'description': 'Minimum balance requirement'},
    'interest_rate': {'required': False, 'type': 'float', 'description': 'Interest rate'},
    'interest_accrued': {'required': False, 'type': 'float', 'description': 'Accrued interest'},
    'is_joint_account': {'required': False, 'type': 'string', 'description': 'Joint account indicator'},
    'joint_account_holders': {'required': False, 'type': 'float', 'description': 'Number of joint holders'},
    'has_outstanding_facilities': {'required': False, 'type': 'string', 'description': 'Outstanding facilities indicator'},
    'maturity_date': {'required': False, 'type': 'date', 'description': 'Account maturity date'},
    'auto_renewal': {'required': False, 'type': 'string', 'description': 'Auto renewal setting'},
    'last_statement_date': {'required': False, 'type': 'date', 'description': 'Last statement date'},
    'statement_frequency': {'required': False, 'type': 'string', 'description': 'Statement frequency'},
    'tracking_id': {'required': False, 'type': 'string', 'description': 'Tracking identifier'},
    'dormancy_trigger_date': {'required': True, 'type': 'date', 'description': 'Date dormancy triggered'},
    'dormancy_period_start': {'required': False, 'type': 'date', 'description': 'Dormancy period start'},
    'dormancy_period_months': {'required': False, 'type': 'float', 'description': 'Dormancy period in months'},
    'dormancy_classification_date': {'required': False, 'type': 'date', 'description': 'Date of dormancy classification'},
    'transfer_eligibility_date': {'required': False, 'type': 'date', 'description': 'Transfer eligibility date'},
    'current_stage': {'required': True, 'type': 'string', 'description': 'Current processing stage'},
    'contact_attempts_made': {'required': True, 'type': 'integer', 'description': 'Number of contact attempts'},
    'last_contact_attempt_date': {'required': False, 'type': 'date', 'description': 'Date of last contact attempt'},
    'waiting_period_start': {'required': False, 'type': 'date', 'description': 'Waiting period start date'},
    'waiting_period_end': {'required': False, 'type': 'date', 'description': 'Waiting period end date'},
    'transferred_to_ledger_date': {'required': False, 'type': 'date', 'description': 'Ledger transfer date'},
    'transferred_to_cb_date': {'required': False, 'type': 'date', 'description': 'Central Bank transfer date'},
    'cb_transfer_amount': {'required': False, 'type': 'float', 'description': 'Central Bank transfer amount'},
    'cb_transfer_reference': {'required': False, 'type': 'string', 'description': 'CB transfer reference'},
    'exclusion_reason': {'required': False, 'type': 'string', 'description': 'Exclusion reason'},
    'created_date': {'required': False, 'type': 'date', 'description': 'Record creation date'},
    'updated_date': {'required': False, 'type': 'date', 'description': 'Record update date'},
    'updated_by': {'required': False, 'type': 'string', 'description': 'Updated by user'}
}

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'data_quality_results' not in st.session_state:
        st.session_state.data_quality_results = None
    if 'mapping_results' not in st.session_state:
        st.session_state.mapping_results = None
    if 'dormancy_results' not in st.session_state:
        st.session_state.dormancy_results = None
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = None
    if 'processing_agent' not in st.session_state:
        st.session_state.processing_agent = None

# ===================== AUTHENTICATION =====================

def show_login_page():
    """Display login interface"""
    st.markdown('<div class="main-header">🏛️ CBUAE Banking Compliance System</div>', unsafe_allow_html=True)

    st.markdown("### 🔐 Authentication Required")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            st.markdown("#### Login to Access System")
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.form_submit_button("🚀 Login", use_container_width=True):
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.current_page = 'main'
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")

            with col_b:
                if st.form_submit_button("👤 Demo Login", use_container_width=True):
                    st.session_state.logged_in = True
                    st.session_state.username = "demo_user"
                    st.session_state.current_page = 'main'
                    st.success("✅ Demo login successful!")
                    st.rerun()

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
        - 10 specialized dormancy agents
        - Complete CBUAE Article compliance (Art. 2-8)
        - Individual CSV downloads per agent
        - Risk assessment and prioritization
        - Central Bank transfer processing
        """)

    with col3:
        st.markdown("""
        **⚖️ Compliance Verification:**
        - 17 compliance verification agents
        - Multi-article regulation checking
        - Automated violation detection
        - Comprehensive audit trails
        - Action-oriented recommendations
        """)

def authenticate_user(username: str, password: str) -> bool:
    """Simple authentication (replace with actual authentication system)"""
    # Demo credentials
    valid_credentials = {
        "admin": "admin123",
        "compliance": "compliance123",
        "demo": "demo123"
    }
    return username in valid_credentials and valid_credentials[username] == password

# ===================== DATA PROCESSING SECTION =====================

def show_data_processing_section():
    """Display comprehensive data processing interface"""
    st.markdown('<div class="section-header">📤 Data Processing & Upload</div>', unsafe_allow_html=True)

    if not DATA_PROCESSING_AVAILABLE:
        st.error("❌ Data Processing Agent not available")
        st.info("💡 Please ensure agents.Data_Process module is available and properly configured.")
        return

    # Initialize processing agent
    if st.session_state.processing_agent is None:
        st.session_state.processing_agent = UnifiedDataProcessingAgent()

    # Data Upload Section
    st.markdown("#### 📁 Data Upload Options")

    upload_method = st.selectbox(
        "Select Upload Method:",
        ["File Upload", "Google Drive", "Data Lake", "HDFS"],
        help="Choose your preferred data source method"
    )

    uploaded_data = None

    if upload_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Choose file (CSV, Excel, JSON, Parquet)",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Upload your banking compliance dataset"
        )

        if uploaded_file is not None:
            with st.spinner("🔄 Processing uploaded file..."):
                try:
                    # Process file based on type
                    if uploaded_file.name.endswith('.csv'):
                        uploaded_data = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        uploaded_data = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        uploaded_data = pd.read_json(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        uploaded_data = pd.read_parquet(uploaded_file)

                    st.session_state.uploaded_data = uploaded_data
                    st.success(f"✅ File uploaded successfully! Shape: {uploaded_data.shape}")

                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

    elif upload_method == "Google Drive":
        st.info("🔗 Google Drive integration - Enter file URL or ID")
        drive_url = st.text_input("Google Drive File URL/ID:")
        if drive_url and st.button("📥 Download from Drive"):
            st.warning("🚧 Google Drive integration requires authentication setup")

    elif upload_method == "Data Lake":
        st.info("☁️ Data Lake connection - Configure your data source")
        col1, col2 = st.columns(2)
        with col1:
            lake_endpoint = st.text_input("Data Lake Endpoint:")
        with col2:
            lake_path = st.text_input("File Path:")
        if lake_endpoint and lake_path and st.button("📥 Connect to Data Lake"):
            st.warning("🚧 Data Lake integration requires proper credentials")

    elif upload_method == "HDFS":
        st.info("🗂️ HDFS connection - Configure Hadoop cluster")
        col1, col2 = st.columns(2)
        with col1:
            hdfs_host = st.text_input("HDFS Host:")
        with col2:
            hdfs_path = st.text_input("HDFS Path:")
        if hdfs_host and hdfs_path and st.button("📥 Connect to HDFS"):
            st.warning("🚧 HDFS integration requires cluster access")

    # Data Quality Analysis
    if st.session_state.uploaded_data is not None:
        st.markdown("---")
        st.markdown("#### 🔍 Data Quality Analysis")

        if st.button("🔬 Analyze Data Quality", use_container_width=True):
            with st.spinner("🔄 Analyzing data quality..."):
                try:
                    analyzer = DataQualityAnalyzer()
                    quality_result = analyzer.analyze_data_quality(st.session_state.uploaded_data)
                    st.session_state.data_quality_results = quality_result

                    # Display quality metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Overall Score", f"{quality_result.overall_score:.1%}")
                    with col2:
                        st.metric("Quality Level", quality_result.quality_level.title())
                    with col3:
                        st.metric("Missing %", f"{quality_result.missing_percentage:.1%}")
                    with col4:
                        st.metric("Duplicates", quality_result.duplicate_records)

                    # Quality recommendations
                    if quality_result.recommendations:
                        st.markdown("**📋 Recommendations:**")
                        for rec in quality_result.recommendations:
                            st.write(f"• {rec}")

                except Exception as e:
                    st.error(f"❌ Quality analysis failed: {str(e)}")

        # Data Mapping Section
        st.markdown("---")
        st.markdown("#### 🗺️ Data Mapping")

        # Enable LLM toggle
        enable_llm = st.toggle(
            "🤖 Enable LLM Auto-Mapping",
            help="Use LLM to automatically map columns based on semantic similarity and reasoning"
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("🎯 Generate Column Mapping", use_container_width=True):
                with st.spinner("🔄 Generating column mapping..."):
                    try:
                        mapping_result = generate_column_mapping(
                            st.session_state.uploaded_data,
                            enable_llm=enable_llm
                        )
                        st.session_state.mapping_results = mapping_result

                        if mapping_result['success']:
                            st.success(f"✅ Mapping generated! Auto-mapped: {mapping_result['auto_mapped']}/{len(st.session_state.uploaded_data.columns)} columns")
                        else:
                            st.error(f"❌ Mapping failed: {mapping_result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"❌ Mapping generation failed: {str(e)}")

        with col2:
            if st.session_state.mapping_results:
                mapping_df = pd.DataFrame(st.session_state.mapping_results['mappings'])
                csv = mapping_df.to_csv(index=False)
                st.download_button(
                    "📄 Download Mapping",
                    csv,
                    f"column_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )

        # Display mapping results
        if st.session_state.mapping_results and st.session_state.mapping_results['success']:
            display_mapping_interface()

def generate_column_mapping(data: pd.DataFrame, enable_llm: bool = False) -> Dict:
    """Generate column mapping with optional LLM enhancement"""
    try:
        source_columns = list(data.columns)
        target_fields = list(BANKING_SCHEMA.keys())

        mappings = []
        auto_mapped = 0

        for source_col in source_columns:
            best_match = None
            best_score = 0.0

            # Simple string similarity matching (replace with BGE embeddings in production)
            for target_field in target_fields:
                # Basic similarity calculation
                source_lower = source_col.lower().replace('_', ' ').replace('-', ' ')
                target_lower = target_field.lower().replace('_', ' ').replace('-', ' ')

                # Check for exact matches or contains
                if source_lower == target_lower:
                    score = 1.0
                elif source_lower in target_lower or target_lower in source_lower:
                    score = 0.8
                elif any(word in target_lower.split() for word in source_lower.split()):
                    score = 0.6
                else:
                    score = 0.3

                if score > best_score:
                    best_score = score
                    best_match = target_field

            confidence = "high" if best_score >= 0.8 else "medium" if best_score >= 0.6 else "low"

            if best_score >= 0.6:
                auto_mapped += 1

            mappings.append({
                'source_column': source_col,
                'target_field': best_match,
                'confidence': confidence,
                'similarity_score': best_score,
                'required': BANKING_SCHEMA.get(best_match, {}).get('required', False),
                'description': BANKING_SCHEMA.get(best_match, {}).get('description', '')
            })

        return {
            'success': True,
            'mappings': mappings,
            'auto_mapped': auto_mapped,
            'total_columns': len(source_columns),
            'llm_enhanced': enable_llm
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'mappings': [],
            'auto_mapped': 0
        }

def display_mapping_interface():
    """Display interactive mapping interface"""
    if not st.session_state.mapping_results:
        return

    st.markdown("#### 🎯 Column Mapping Results")

    # Convert to DataFrame for display
    mapping_df = pd.DataFrame(st.session_state.mapping_results['mappings'])

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_confidence = st.selectbox("Filter by Confidence:", ["All", "High", "Medium", "Low"])
    with col2:
        filter_required = st.selectbox("Filter by Required:", ["All", "Required Only", "Optional Only"])
    with col3:
        show_unmapped = st.checkbox("Show Unmapped Only")

    # Apply filters
    filtered_df = mapping_df.copy()

    if filter_confidence != "All":
        filtered_df = filtered_df[filtered_df['confidence'] == filter_confidence.lower()]

    if filter_required == "Required Only":
        filtered_df = filtered_df[filtered_df['required'] == True]
    elif filter_required == "Optional Only":
        filtered_df = filtered_df[filtered_df['required'] == False]

    if show_unmapped:
        filtered_df = filtered_df[filtered_df['similarity_score'] < 0.6]

    # Display mapping table with edit capability
    st.markdown("**🗂️ Mapping Table** (Click to edit)")

    edited_df = st.data_editor(
        filtered_df,
        column_config={
            "source_column": st.column_config.TextColumn("Source Column", disabled=True),
            "target_field": st.column_config.SelectboxColumn(
                "Target Field",
                options=list(BANKING_SCHEMA.keys()),
                required=True
            ),
            "confidence": st.column_config.TextColumn("Confidence", disabled=True),
            "similarity_score": st.column_config.NumberColumn("Score", disabled=True, format="%.2f"),
            "required": st.column_config.CheckboxColumn("Required", disabled=True),
            "description": st.column_config.TextColumn("Description", disabled=True)
        },
        hide_index=True,
        use_container_width=True
    )

    # Update session state with edited mappings
    if not edited_df.equals(filtered_df):
        st.session_state.mapping_results['mappings'] = edited_df.to_dict('records')
        st.success("✅ Mapping updated!")

# ===================== DORMANCY ANALYSIS SECTION =====================

def show_dormancy_analysis_section():
    """Display dormancy analysis with all 10 agents"""
    st.markdown('<div class="section-header">💤 Dormancy Analysis</div>', unsafe_allow_html=True)

    if not DORMANCY_AGENTS_AVAILABLE:
        st.error("❌ Dormancy agents not available")
        return

    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first in the Data Processing section")
        return

    # Run dormancy analysis
    if st.button("🔬 Run Comprehensive Dormancy Analysis", use_container_width=True):
        with st.spinner("🔄 Running all 10 dormancy agents..."):
            try:
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                dormancy_results = loop.run_until_complete(
                    run_comprehensive_dormancy_analysis_with_csv(
                        user_id=st.session_state.username,
                        account_data=st.session_state.uploaded_data
                    )
                )

                st.session_state.dormancy_results = dormancy_results

                if dormancy_results['success']:
                    st.success("✅ Dormancy analysis completed successfully!")

                    # Display summary metrics
                    summary = dormancy_results.get('summary', {})
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Accounts", summary.get('total_accounts_analyzed', 0))
                    with col2:
                        st.metric("Dormant Found", summary.get('total_dormant_accounts_found', 0))
                    with col3:
                        st.metric("Agents Executed", summary.get('agents_executed', 0))
                    with col4:
                        st.metric("CSV Files", summary.get('csv_exports_available', 0))

                else:
                    st.error(f"❌ Dormancy analysis failed: {dormancy_results.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    # Display individual agent results
    if st.session_state.dormancy_results and st.session_state.dormancy_results['success']:
        display_dormancy_agent_results()

def display_dormancy_agent_results():
    """Display results from all dormancy agents with download options"""
    results = st.session_state.dormancy_results
    agent_results = results.get('agent_results', {})
    csv_exports = results.get('csv_exports', {})

    st.markdown("---")
    st.markdown("#### 🤖 Individual Dormancy Agent Results")

    # Agent information
    dormancy_agents_info = {
        'demand_deposit': {
            'name': 'Demand Deposit Dormancy',
            'article': 'CBUAE Art. 2.1.1',
            'description': 'Analyzes savings/checking accounts for 12-month dormancy'
        },
        'fixed_deposit': {
            'name': 'Fixed Deposit Dormancy',
            'article': 'CBUAE Art. 2.1.2',
            'description': 'Analyzes term deposits for post-maturity dormancy'
        },
        'investment_account': {
            'name': 'Investment Account Dormancy',
            'article': 'CBUAE Art. 2.2',
            'description': 'Analyzes investment portfolios and mutual funds'
        },
        'contact_attempts': {
            'name': 'Contact Attempts Verification',
            'article': 'CBUAE Art. 5',
            'description': 'Verifies minimum 3 contact attempts compliance'
        },
        'cb_transfer': {
            'name': 'CB Transfer Eligibility',
            'article': 'CBUAE Art. 8',
            'description': 'Identifies accounts for Central Bank transfer (5+ years)'
        },
        'foreign_currency': {
            'name': 'Foreign Currency Conversion',
            'article': 'CBUAE Art. 8.5',
            'description': 'Manages foreign currency conversion requirements'
        },
        'high_value_dormant': {
            'name': 'High Value Dormant Accounts',
            'article': 'High Value Monitoring',
            'description': 'Executive escalation for high-value accounts (500K+ AED)'
        },
        'dormancy_escalation': {
            'name': 'Dormancy Escalation',
            'article': 'Escalation Procedures',
            'description': 'Management escalation and timeline procedures'
        },
        'statement_suppression': {
            'name': 'Statement Suppression',
            'article': 'CBUAE Art. 7.3',
            'description': 'Statement suppression for 6+ month dormant accounts'
        },
        'internal_ledger_transfer': {
            'name': 'Internal Ledger Transfer',
            'article': 'CBUAE Art. 3',
            'description': 'Internal ledger transfer after contact attempts'
        }
    }

    # Display each agent
    for agent_key, agent_info in dormancy_agents_info.items():
        if agent_key in agent_results:
            agent_result = agent_results[agent_key]

            with st.expander(f"📊 {agent_info['name']} ({agent_info['article']})", expanded=False):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**Description:** {agent_info['description']}")
                    st.markdown(f"**Status:** {'✅ Success' if agent_result.get('success') else '❌ Failed'}")

                    if agent_result.get('success'):
                        st.markdown(f"**Records Processed:** {agent_result.get('records_processed', 0):,}")
                        st.markdown(f"**Dormant Found:** {agent_result.get('dormant_records_found', 0):,}")
                        st.markdown(f"**Processing Time:** {agent_result.get('processing_time', 0):.2f}s")

                with col2:
                    # Summary button
                    if st.button(f"📋 View Summary", key=f"summary_{agent_key}"):
                        if agent_result.get('analysis_results'):
                            st.json(agent_result['analysis_results'].get('summary_stats', {}))

                with col3:
                    # CSV download button
                    if agent_key in csv_exports and csv_exports[agent_key].get('available'):
                        csv_data = csv_exports[agent_key]

                        st.download_button(
                            "📄 Download CSV",
                            csv_data['csv_data'],
                            csv_data['filename'],
                            "text/csv",
                            key=f"download_{agent_key}",
                            use_container_width=True
                        )

                        st.caption(f"{csv_data['records']} records ({csv_data['file_size_kb']:.1f} KB)")
                    else:
                        st.caption("No data found")

# ===================== COMPLIANCE ANALYSIS SECTION =====================

def show_compliance_analysis_section():
    """Display compliance analysis with all 17 agents"""
    st.markdown('<div class="section-header">⚖️ Compliance Verification</div>', unsafe_allow_html=True)

    if not COMPLIANCE_AGENTS_AVAILABLE:
        st.error("❌ Compliance agents not available")
        return

    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first in the Data Processing section")
        return

    # Run compliance analysis
    if st.button("🔬 Run Comprehensive Compliance Analysis", use_container_width=True):
        with st.spinner("🔄 Running all 17 compliance agents..."):
            try:
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                compliance_results = loop.run_until_complete(
                    run_comprehensive_compliance_analysis_with_csv(
                        user_id=st.session_state.username,
                        dormancy_results=st.session_state.dormancy_results,
                        accounts_df=st.session_state.uploaded_data
                    )
                )

                st.session_state.compliance_results = compliance_results

                if compliance_results['success']:
                    st.success("✅ Compliance analysis completed successfully!")

                    # Display summary metrics
                    summary = compliance_results.get('compliance_summary', {})
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Accounts", summary.get('total_accounts_analyzed', 0))
                    with col2:
                        st.metric("Violations Found", summary.get('total_violations_found', 0))
                    with col3:
                        st.metric("Actions Generated", summary.get('total_actions_generated', 0))
                    with col4:
                        st.metric("Risk Level", summary.get('regulatory_risk_level', 'Unknown'))

                else:
                    st.error(f"❌ Compliance analysis failed: {compliance_results.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    # Display individual agent results
    if st.session_state.compliance_results and st.session_state.compliance_results['success']:
        display_compliance_agent_results()

def display_compliance_agent_results():
    """Display results from all compliance agents with download options"""
    results = st.session_state.compliance_results
    agent_results = results.get('agent_results', {})
    csv_exports = results.get('csv_exports', {})

    st.markdown("---")
    st.markdown("#### 🤖 Individual Compliance Agent Results")

    # Get agent info from the compliance module
    try:
        all_agents_info = get_all_compliance_agents_info()
        agents_by_category = all_agents_info.get('agents_by_category', {})
    except:
        # Fallback agent information
        agents_by_category = {
            "Contact & Communication": [
                {"agent_name": "incomplete_contact", "cbuae_article": "CBUAE Art. 3.1, 5"},
                {"agent_name": "unflagged_dormant", "cbuae_article": "CBUAE Art. 2"}
            ],
            "Process Management": [
                {"agent_name": "internal_ledger", "cbuae_article": "CBUAE Art. 3.4, 3.5"},
                {"agent_name": "statement_freeze", "cbuae_article": "CBUAE Art. 7.3"},
                {"agent_name": "cbuae_transfer", "cbuae_article": "CBUAE Art. 8"}
            ],
            "Specialized Compliance": [
                {"agent_name": "fx_conversion", "cbuae_article": "CBUAE Art. 8.5"},
                {"agent_name": "sdb_court", "cbuae_article": "CBUAE Art. 3.7"},
                {"agent_name": "unclaimed_instruments", "cbuae_article": "CBUAE Art. 3.6"},
                {"agent_name": "claim_processing", "cbuae_article": "CBUAE Art. 4"}
            ],
            "Reporting & Retention": [
                {"agent_name": "annual_report", "cbuae_article": "CBUAE Art. 3.10"},
                {"agent_name": "record_retention", "cbuae_article": "Record Retention"}
            ],
            "Utility": [
                {"agent_name": "log_flags", "cbuae_article": "Flag Logging"}
            ]
        }

    # Display by category
    for category, category_agents in agents_by_category.items():
        st.markdown(f"##### 📂 {category}")

        for agent_info in category_agents:
            agent_key = agent_info.get('agent_name', '')
            if agent_key in agent_results:
                agent_result = agent_results[agent_key]

                with st.expander(f"📊 {agent_key.replace('_', ' ').title()} ({agent_info.get('cbuae_article', 'Unknown')})", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.markdown(f"**Category:** {category}")
                        st.markdown(f"**Status:** {'✅ Success' if agent_result.get('success') else '❌ Failed'}")

                        if agent_result.get('success'):
                            st.markdown(f"**Accounts Processed:** {agent_result.get('accounts_processed', 0):,}")
                            st.markdown(f"**Violations Found:** {agent_result.get('violations_found', 0):,}")
                            st.markdown(f"**Actions Generated:** {agent_result.get('actions_generated', 0):,}")

                    with col2:
                        # Summary button
                        if st.button(f"📋 View Summary", key=f"comp_summary_{agent_key}"):
                            if agent_result.get('compliance_summary'):
                                st.json(agent_result['compliance_summary'])

                    with col3:
                        # CSV download button
                        if agent_key in csv_exports and csv_exports[agent_key].get('available'):
                            csv_data = csv_exports[agent_key]

                            st.download_button(
                                "📄 Download CSV",
                                csv_data['csv_data'],
                                csv_data['filename'],
                                "text/csv",
                                key=f"comp_download_{agent_key}",
                                use_container_width=True
                            )

                            st.caption(f"{csv_data['records']} records ({csv_data['file_size_kb']:.1f} KB)")
                        else:
                            st.caption("No violations found")

# ===================== REPORTS SECTION =====================

def show_reports_section():
    """Display comprehensive reports for all agents"""
    st.markdown('<div class="section-header">📊 Comprehensive Reports</div>', unsafe_allow_html=True)

    # Summary dashboard
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💤 Dormancy Analysis Summary")
        if st.session_state.dormancy_results:
            dormancy_summary = st.session_state.dormancy_results.get('summary', {})

            metrics = {
                "Total Accounts": dormancy_summary.get('total_accounts_analyzed', 0),
                "Dormant Found": dormancy_summary.get('total_dormant_accounts_found', 0),
                "Agents Executed": dormancy_summary.get('agents_executed', 0),
                "CSV Files": dormancy_summary.get('csv_exports_available', 0)
            }

            for metric, value in metrics.items():
                st.metric(metric, f"{value:,}")
        else:
            st.info("No dormancy analysis results available")

    with col2:
        st.markdown("#### ⚖️ Compliance Analysis Summary")
        if st.session_state.compliance_results:
            compliance_summary = st.session_state.compliance_results.get('compliance_summary', {})

            metrics = {
                "Total Violations": compliance_summary.get('total_violations_found', 0),
                "Actions Generated": compliance_summary.get('total_actions_generated', 0),
                "Risk Level": compliance_summary.get('regulatory_risk_level', 'Unknown'),
                "Agents Executed": compliance_summary.get('agents_executed', 0)
            }

            for metric, value in metrics.items():
                if isinstance(value, int):
                    st.metric(metric, f"{value:,}")
                else:
                    st.metric(metric, str(value))
        else:
            st.info("No compliance analysis results available")

    # Detailed agent breakdown
    st.markdown("---")
    st.markdown("#### 🤖 All Agents Overview")

    if st.session_state.dormancy_results or st.session_state.compliance_results:
        create_agent_overview_table()
    else:
        st.info("Please run dormancy and compliance analysis to view detailed reports")

def create_agent_overview_table():
    """Create comprehensive table showing all agents and their results"""
    all_agents_data = []

    # Add dormancy agents
    if st.session_state.dormancy_results:
        dormancy_agents = st.session_state.dormancy_results.get('agent_results', {})
        for agent_key, result in dormancy_agents.items():
            all_agents_data.append({
                'Agent Name': agent_key.replace('_', ' ').title(),
                'Type': 'Dormancy',
                'Accounts Processed': result.get('records_processed', 0),
                'Issues Found': result.get('dormant_records_found', 0),
                'Processing Time (s)': result.get('processing_time', 0),
                'Status': '✅ Success' if result.get('success') else '❌ Failed',
                'CSV Available': '✅' if agent_key in st.session_state.dormancy_results.get('csv_exports', {}) else '❌'
            })

    # Add compliance agents
    if st.session_state.compliance_results:
        compliance_agents = st.session_state.compliance_results.get('agent_results', {})
        for agent_key, result in compliance_agents.items():
            all_agents_data.append({
                'Agent Name': agent_key.replace('_', ' ').title(),
                'Type': 'Compliance',
                'Accounts Processed': result.get('accounts_processed', 0),
                'Issues Found': result.get('violations_found', 0),
                'Processing Time (s)': result.get('processing_time', 0),
                'Status': '✅ Success' if result.get('success') else '❌ Failed',
                'CSV Available': '✅' if agent_key in st.session_state.compliance_results.get('csv_exports', {}) else '❌'
            })

    if all_agents_data:
        agents_df = pd.DataFrame(all_agents_data)

        # Display filterable table
        col1, col2 = st.columns(2)
        with col1:
            type_filter = st.selectbox("Filter by Type:", ["All", "Dormancy", "Compliance"])
        with col2:
            status_filter = st.selectbox("Filter by Status:", ["All", "Success", "Failed"])

        # Apply filters
        filtered_df = agents_df.copy()
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df['Type'] == type_filter]
        if status_filter != "All":
            status_value = '✅ Success' if status_filter == 'Success' else '❌ Failed'
            filtered_df = filtered_df[filtered_df['Status'] == status_value]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Agents", len(filtered_df))
        with col2:
            st.metric("Total Issues Found", filtered_df['Issues Found'].sum())
        with col3:
            success_rate = (filtered_df['Status'] == '✅ Success').mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")

# ===================== SIDEBAR NAVIGATION =====================

def show_sidebar():
    """Display sidebar navigation"""
    with st.sidebar:
        st.markdown("### 🏛️ CBUAE Compliance System")
        st.markdown(f"**User:** {st.session_state.username}")
        st.markdown("---")

        # Navigation menu
        pages = {
            "📤 Data Processing": "data_processing",
            "💤 Dormancy Analysis": "dormancy_analysis",
            "⚖️ Compliance Verification": "compliance_verification",
            "📊 Reports": "reports"
        }

        for page_name, page_key in pages.items():
            if st.button(page_name, use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()

        st.markdown("---")

        # System status
        st.markdown("### 🔧 System Status")

        status_indicators = {
            "Data Processing": DATA_PROCESSING_AVAILABLE,
            "Dormancy Agents": DORMANCY_AGENTS_AVAILABLE,
            "Compliance Agents": COMPLIANCE_AGENTS_AVAILABLE
        }

        for system, available in status_indicators.items():
            status = "🟢 Available" if available else "🔴 Unavailable"
            st.markdown(f"**{system}:** {status}")

        st.markdown("---")

        # Data status
        if st.session_state.uploaded_data is not None:
            st.markdown("### 📊 Data Status")
            st.markdown(f"**Shape:** {st.session_state.uploaded_data.shape}")
            st.markdown(f"**Columns:** {len(st.session_state.uploaded_data.columns)}")

            if st.session_state.data_quality_results:
                quality = st.session_state.data_quality_results
                st.markdown(f"**Quality:** {quality.quality_level.title()}")
                st.markdown(f"**Score:** {quality.overall_score:.1%}")

        st.markdown("---")

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ===================== MAIN APPLICATION =====================

def main():
    """Main application function"""
    initialize_session_state()

    # Check authentication
    if not st.session_state.logged_in:
        show_login_page()
        return

    # Show sidebar
    show_sidebar()

    # Main content area
    if st.session_state.current_page == 'data_processing':
        show_data_processing_section()
    elif st.session_state.current_page == 'dormancy_analysis':
        show_dormancy_analysis_section()
    elif st.session_state.current_page == 'compliance_verification':
        show_compliance_analysis_section()
    elif st.session_state.current_page == 'reports':
        show_reports_section()
    else:
        # Default to data processing
        st.session_state.current_page = 'data_processing'
        show_data_processing_section()

if __name__ == "__main__":
    main()