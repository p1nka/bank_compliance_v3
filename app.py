"""
Enhanced Banking Compliance Analysis - Comprehensive Streamlit Application
Integrates all agents with proper data flow and login system
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
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import warnings
import secrets
import hashlib
import logging
import traceback

# Configure Streamlit page
st.set_page_config(
    page_title="CBUAE Banking Compliance Analysis System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all agents with error handling
AGENTS_STATUS = {}

try:
    from agents.data_upload_agent import BankingComplianceUploader, UploadResult
    AGENTS_STATUS['data_upload'] = True
except ImportError as e:
    AGENTS_STATUS['data_upload'] = False
    st.error(f"Data Upload Agent not available: {e}")

try:
    from agents.data_mapping_agent import DataMappingAgent, run_automated_data_mapping
    AGENTS_STATUS['data_mapping'] = True
except ImportError as e:
    AGENTS_STATUS['data_mapping'] = False
    st.error(f"Data Mapping Agent not available: {e}")

try:
    from agents.Data_Process import DataProcessingAgent, DataQualityAnalyzer
    AGENTS_STATUS['data_processing'] = True
except ImportError as e:
    AGENTS_STATUS['data_processing'] = False
    st.error(f"Data Processing Agent not available: {e}")

try:
    from agents.dormant_agent import (
        run_comprehensive_dormancy_analysis_csv,
        DormancyAnalysisAgent,
        validate_csv_structure,
        DemandDepositDormancyAgent,
        FixedDepositDormancyAgent,
        InvestmentAccountDormancyAgent,
        ContactAttemptsAgent,
        CBTransferEligibilityAgent,
        ForeignCurrencyConversionAgent
    )
    AGENTS_STATUS['dormancy'] = True
except ImportError as e:
    AGENTS_STATUS['dormancy'] = False
    st.error(f"Dormancy Agents not available: {e}")

try:
    from agents.compliance_verification_agent import (
        DetectIncompleteContactAttemptsAgent,
        DetectUnflaggedDormantCandidatesAgent,
        DetectInternalLedgerCandidatesAgent,
        DetectStatementFreezeCandidatesAgent
    )
    AGENTS_STATUS['compliance'] = True
except ImportError as e:
    AGENTS_STATUS['compliance'] = False
    st.error(f"Compliance Agents not available: {e}")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    AGENTS_STATUS['bge'] = True
except ImportError as e:
    AGENTS_STATUS['bge'] = False

warnings.filterwarnings('ignore')

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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .cbuae-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Load BGE model with caching
@st.cache_resource
def load_bge_model():
    """Load BGE model with caching"""
    if AGENTS_STATUS['bge']:
        try:
            st.info("🔄 Loading BGE-large model...")
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            st.success("✅ BGE model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Failed to load BGE model: {e}")
            return None
    return None

# Session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'logged_in': False,
        'username': '',
        'current_page': 'Data Processing',
        'uploaded_data': None,
        'processed_data': None,
        'mapped_data': None,
        'quality_results': None,
        'mapping_results': None,
        'dormancy_results': {},
        'compliance_results': {},
        'llm_enabled': False,
        'bge_model': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Authentication system
def show_login_page():
    """Display login page"""
    st.markdown('<div class="main-header">🏛️ CBUAE Banking Compliance System</div>', unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown("### 🔐 Secure Login")

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚪 Login", type="primary", use_container_width=True):
            if validate_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    with col2:
        if st.button("👤 Demo Login", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.username = "demo_user"
            st.success("✅ Demo login successful!")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 System Capabilities")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📤 Data Processing:**
        - Multi-source data upload (4 methods)
        - Real-time quality analysis
        - BGE-powered schema mapping
        - LLM-enhanced column mapping
        """)

    with col2:
        st.markdown("""
        **⚖️ Compliance Analysis:**
        - CBUAE dormancy regulations
        - Automated compliance checks
        - Comprehensive reporting
        - Downloadable audit trails
        """)

def validate_credentials(username: str, password: str) -> bool:
    """Validate user credentials"""
    # Demo credentials for testing
    demo_users = {
        "admin": "admin123",
        "compliance_officer": "compliance123",
        "analyst": "analyst123",
        "demo_user": "demo"
    }

    return username in demo_users and demo_users[username] == password

# Data Upload Section
def show_data_upload_section():
    """Display data upload interface with 4 methods"""
    st.markdown('<div class="section-header">📤 Data Upload</div>', unsafe_allow_html=True)

    # Upload method selection
    upload_method = st.selectbox(
        "Select Upload Method:",
        ["📄 Flat Files", "🔗 Google Drive", "☁️ Data Lake", "🗄️ HDFS"],
        help="Choose your preferred data source method"
    )

    if upload_method == "📄 Flat Files":
        return handle_flat_file_upload()
    elif upload_method == "🔗 Google Drive":
        return handle_drive_upload()
    elif upload_method == "☁️ Data Lake":
        return handle_datalake_upload()
    elif upload_method == "🗄️ HDFS":
        return handle_hdfs_upload()

def handle_flat_file_upload():
    """Handle flat file upload"""
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
        help="Upload CSV, Excel, JSON, or Parquet files"
    )

    if uploaded_file:
        try:
            with st.spinner("📂 Processing uploaded file..."):
                # Process based on file type
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    data = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    data = pd.read_excel(uploaded_file)
                elif file_extension == 'json':
                    data = pd.read_json(uploaded_file)
                elif file_extension == 'parquet':
                    data = pd.read_parquet(uploaded_file)

                st.session_state.uploaded_data = data
                st.success(f"✅ File uploaded successfully! {len(data):,} records loaded.")
                return data

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            return None

    # Sample data generation option
    if st.button("🎲 Generate Sample Banking Data", type="secondary"):
        with st.spinner("Generating sample data..."):
            sample_data = generate_sample_banking_data()
            st.session_state.uploaded_data = sample_data
            st.success(f"✅ Sample data generated! {len(sample_data):,} records created.")
            return sample_data

    return None

def handle_drive_upload():
    """Handle Google Drive upload"""
    st.info("🔗 Google Drive integration requires authentication setup")
    drive_url = st.text_input("Google Drive File URL", placeholder="https://drive.google.com/...")

    if st.button("📥 Download from Drive") and drive_url:
        if AGENTS_STATUS['data_upload']:
            try:
                uploader = BankingComplianceUploader()
                result = uploader.upload_from_drive(drive_url)
                if result.success:
                    st.session_state.uploaded_data = result.data
                    st.success("✅ Data loaded from Google Drive!")
                    return result.data
                else:
                    st.error(f"❌ Drive upload failed: {result.error}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.error("❌ Data upload agent not available")

    return None

def handle_datalake_upload():
    """Handle Data Lake upload"""
    st.info("☁️ Data Lake connection requires Azure/AWS credentials")

    col1, col2 = st.columns(2)
    with col1:
        container = st.text_input("Container/Bucket Name")
        file_path = st.text_input("File Path")

    with col2:
        connection_string = st.text_input("Connection String", type="password")

    if st.button("☁️ Connect to Data Lake") and all([container, file_path, connection_string]):
        st.info("🔄 Data Lake integration in progress...")
        # Implementation would go here

    return None

def handle_hdfs_upload():
    """Handle HDFS upload"""
    st.info("🗄️ HDFS connection requires cluster configuration")

    col1, col2 = st.columns(2)
    with col1:
        hdfs_host = st.text_input("HDFS Host", value="localhost")
        hdfs_port = st.number_input("HDFS Port", value=9000)

    with col2:
        hdfs_path = st.text_input("HDFS File Path", placeholder="/data/banking_data.csv")

    if st.button("🗄️ Connect to HDFS") and hdfs_path:
        st.info("🔄 HDFS integration in progress...")
        # Implementation would go here

    return None

def generate_sample_banking_data():
    """Generate realistic sample banking data"""
    np.random.seed(42)
    n_records = 5000

    # Generate customer data
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_records + 1)]
    customer_types = np.random.choice(['Individual', 'Corporate'], n_records, p=[0.8, 0.2])

    # Generate names
    first_names = ['Ahmed', 'Fatima', 'Mohammed', 'Aisha', 'Ali', 'Sara', 'Omar', 'Layla']
    last_names = ['Al-Maktoum', 'Al-Rashid', 'Al-Mansouri', 'Al-Zaabi', 'Al-Suwaidi']
    full_names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_records)]

    # Generate account data
    account_types = np.random.choice(['Savings', 'Current', 'Fixed Deposit', 'Investment'], n_records)
    balances = np.random.lognormal(8, 2, n_records)  # Log-normal distribution for realistic balances

    # Generate transaction dates (some accounts dormant)
    base_date = datetime.now()
    last_transaction_dates = []
    dormancy_statuses = []

    for i in range(n_records):
        # 20% chance of being dormant
        if np.random.random() < 0.2:
            # Dormant: no transaction in last 12+ months
            days_back = np.random.randint(365, 1095)  # 1-3 years back
            dormancy_statuses.append('Dormant')
        else:
            # Active: transaction within last 12 months
            days_back = np.random.randint(1, 365)
            dormancy_statuses.append('Active')

        last_transaction_dates.append(base_date - timedelta(days=days_back))

    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'customer_type': customer_types,
        'full_name_en': full_names,
        'account_type': account_types,
        'balance_current': balances,
        'last_transaction_date': last_transaction_dates,
        'dormancy_status': dormancy_statuses,
        'account_status': np.random.choice(['Active', 'Closed', 'Suspended'], n_records, p=[0.85, 0.1, 0.05]),
        'currency': np.random.choice(['AED', 'USD', 'EUR', 'GBP'], n_records, p=[0.7, 0.2, 0.05, 0.05]),
        'opening_date': [base_date - timedelta(days=np.random.randint(365, 3650)) for _ in range(n_records)],
        'contact_attempts_made': np.random.randint(0, 5, n_records),
        'kyc_status': np.random.choice(['Valid', 'Expired', 'Pending'], n_records, p=[0.8, 0.15, 0.05])
    })

    return data

# Data Processing Section
def show_data_processing_section():
    """Display data processing with quality analysis and mapping"""
    st.markdown('<div class="section-header">🔍 Data Processing & Quality Analysis</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first")
        return

    data = st.session_state.uploaded_data

    # Data Quality Analysis
    st.markdown("### 📊 Data Quality Analysis")

    if st.button("🔄 Run Quality Analysis", type="primary"):
        with st.spinner("Analyzing data quality..."):
            quality_results = run_data_quality_analysis(data)
            st.session_state.quality_results = quality_results

    if st.session_state.quality_results:
        display_quality_results(st.session_state.quality_results)

    # Data Mapping Section
    st.markdown("### 🗺️ Data Mapping")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("Map your data columns to the CBUAE banking compliance schema:")

    with col2:
        llm_enabled = st.toggle(
            "🤖 Enable LLM",
            value=st.session_state.llm_enabled,
            help="Use AI to automatically suggest optimal column mappings"
        )
        st.session_state.llm_enabled = llm_enabled

    if st.button("🗺️ Start Data Mapping", type="primary"):
        with st.spinner("Running BGE embedding-based data mapping..."):
            mapping_results = run_data_mapping(data, llm_enabled)
            st.session_state.mapping_results = mapping_results

            # Apply mapping to create processed data
            if mapping_results and mapping_results.get('success'):
                st.session_state.mapped_data = apply_column_mapping(data, mapping_results)
                st.session_state.processed_data = st.session_state.mapped_data

    if st.session_state.mapping_results:
        display_mapping_results(st.session_state.mapping_results)

def run_data_quality_analysis(data):
    """Run comprehensive data quality analysis"""
    try:
        if AGENTS_STATUS['data_processing']:
            # Use real data processing agent
            processor = DataProcessingAgent()

            # Run async method in sync context
            loop = get_or_create_event_loop()
            result = loop.run_until_complete(
                processor.execute_workflow(
                    user_id=st.session_state.username,
                    data_source=data,
                    processing_options={"quality_check": True}
                )
            )

            if result.get("success"):
                return format_quality_results(result, data)

        # Fallback to manual calculation
        return calculate_quality_manually(data)

    except Exception as e:
        logger.error(f"Data quality analysis failed: {e}")
        return calculate_quality_manually(data)

def calculate_quality_manually(data):
    """Manual data quality calculation"""
    missing_data = data.isnull().sum()
    total_cells = len(data) * len(data.columns)
    missing_percentage = (missing_data.sum() / total_cells) * 100
    duplicate_records = data.duplicated().sum()

    # Calculate completeness and uniqueness scores
    completeness_score = 100 - missing_percentage
    uniqueness_score = 100 - (duplicate_records / len(data) * 100) if len(data) > 0 else 100
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

    # Generate recommendations
    recommendations = []
    if missing_percentage > 10:
        recommendations.append("High missing data rate detected - review data collection processes")
    if missing_percentage > 5:
        recommendations.append("Consider implementing data validation rules")
    if duplicate_records > 0:
        recommendations.append(f"Found {duplicate_records} duplicate records - implement deduplication")
    if len(data) < 1000:
        recommendations.append("Small dataset - consider collecting more data for robust analysis")

    return {
        'total_records': len(data),
        'total_columns': len(data.columns),
        'missing_percentage': round(missing_percentage, 2),
        'missing_by_column': missing_data.to_dict(),
        'data_types': data.dtypes.astype(str).to_dict(),
        'duplicate_records': int(duplicate_records),
        'overall_score': round(overall_score, 1),
        'quality_level': quality_level,
        'recommendations': recommendations,
        'completeness_score': round(completeness_score, 1),
        'uniqueness_score': round(uniqueness_score, 1)
    }

def format_quality_results(agent_result, data):
    """Format agent results into standard format"""
    return {
        'total_records': len(data),
        'total_columns': len(data.columns),
        'missing_percentage': agent_result.get("missing_percentage", 0),
        'missing_by_column': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.astype(str).to_dict(),
        'duplicate_records': data.duplicated().sum(),
        'overall_score': agent_result.get("quality_score", 0) * 100,
        'quality_level': agent_result.get("quality_level", "Unknown"),
        'recommendations': agent_result.get("recommendations", [])
    }

def display_quality_results(results):
    """Display comprehensive quality results"""
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{results['total_records']:,}")

    with col2:
        st.metric("Missing Data", f"{results['missing_percentage']:.1f}%")

    with col3:
        st.metric("Duplicate Records", f"{results['duplicate_records']:,}")

    with col4:
        quality_delta = None
        if results['overall_score'] >= 85:
            quality_delta = "Excellent"
        elif results['overall_score'] >= 70:
            quality_delta = "Good"
        else:
            quality_delta = "Needs Improvement"

        st.metric("Quality Score", f"{results['overall_score']:.1f}", delta=quality_delta)

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

    # Detailed analysis
    with st.expander("📊 Detailed Quality Analysis"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Completeness Analysis:**")
            if results['missing_percentage'] > 0:
                missing_df = pd.DataFrame([
                    {'Column': col, 'Missing Count': count, 'Missing %': round(count/results['total_records']*100, 2)}
                    for col, count in results['missing_by_column'].items() if count > 0
                ]).sort_values('Missing %', ascending=False)

                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✅ No missing data detected")
            else:
                st.success("✅ No missing data detected")

        with col2:
            st.markdown("**Data Type Distribution:**")
            dtype_df = pd.DataFrame([
                {'Data Type': dtype, 'Column Count': len([col for col, dt in results['data_types'].items() if dt == dtype])}
                for dtype in set(results['data_types'].values())
            ])

            fig = px.pie(dtype_df, values='Column Count', names='Data Type', title="Data Types")
            st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    if results.get('recommendations'):
        with st.expander("💡 Quality Improvement Recommendations"):
            for i, rec in enumerate(results['recommendations'], 1):
                st.markdown(f"{i}. {rec}")

def run_data_mapping(data, llm_enabled=False):
    """Run data mapping using BGE embeddings and optional LLM"""
    try:
        if AGENTS_STATUS['data_mapping']:
            # Get API key if LLM enabled
            groq_api_key = None
            if llm_enabled:
                try:
                    groq_api_key = st.secrets.get("GROQ_API_KEY")
                    if not groq_api_key:
                        st.warning("⚠️ LLM enabled but no Groq API key found. Using BGE only.")
                except:
                    st.warning("⚠️ LLM enabled but no API key found. Using BGE only.")

            # Run async mapping
            loop = get_or_create_event_loop()
            mapping_result = loop.run_until_complete(
                run_automated_data_mapping(
                    source_data=data,
                    user_id=st.session_state.username,
                    groq_api_key=groq_api_key
                )
            )

            return mapping_result
        else:
            # Fallback manual mapping
            return run_manual_mapping(data)

    except Exception as e:
        st.error(f"❌ Data mapping failed: {str(e)}")
        logger.error(f"Data mapping error: {str(e)}")
        return run_manual_mapping(data)

def run_manual_mapping(data):
    """Fallback manual mapping implementation"""
    # Simple keyword-based mapping
    source_columns = list(data.columns)
    mappings = {}

    # Define basic mapping rules
    mapping_rules = {
        'customer_id': ['customer_id', 'cust_id', 'id', 'customer_number'],
        'full_name_en': ['name', 'customer_name', 'full_name'],
        'account_type': ['account_type', 'type', 'product_type'],
        'balance_current': ['balance', 'current_balance', 'amount'],
        'last_transaction_date': ['last_transaction', 'transaction_date', 'last_activity'],
        'dormancy_status': ['dormancy_status', 'status', 'dormant']
    }

    for target_field, keywords in mapping_rules.items():
        for source_col in source_columns:
            if any(keyword.lower() in source_col.lower() for keyword in keywords):
                mappings[source_col] = target_field
                break

    auto_mapping_percentage = (len(mappings) / len(source_columns)) * 100

    return {
        'success': True,
        'mappings': mappings,
        'auto_mapping_percentage': auto_mapping_percentage,
        'method': 'Keyword Matching',
        'mapping_sheet': create_mapping_sheet(source_columns, mappings)
    }

def create_mapping_sheet(source_columns, mappings):
    """Create mapping sheet DataFrame"""
    mapping_data = []

    for source_col in source_columns:
        target_field = mappings.get(source_col, "")
        confidence = "High" if target_field else "Low"

        mapping_data.append({
            'Source_Column': source_col,
            'Target_Field': target_field,
            'Confidence_Level': confidence,
            'Data_Type': 'Mixed',
            'Required': target_field in ['customer_id', 'account_type', 'balance_current']
        })

    return pd.DataFrame(mapping_data)

def display_mapping_results(results):
    """Display mapping results with download option"""
    if results and results.get('success'):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            auto_pct = results.get('auto_mapping_percentage', 0)
            method = results.get('method', 'Unknown')

            if auto_pct >= 85:
                st.success(f"✅ Excellent mapping! {auto_pct:.1f}% auto-mapped using {method}")
            elif auto_pct >= 70:
                st.info(f"ℹ️ Good mapping: {auto_pct:.1f}% auto-mapped using {method}")
            else:
                st.warning(f"⚠️ Manual review needed: {auto_pct:.1f}% auto-mapped using {method}")

        with col2:
            # Download mapping sheet
            if 'mapping_sheet' in results:
                csv_data = results['mapping_sheet'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Mapping",
                    data=csv_data,
                    file_name=f"column_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="secondary"
                )

        with col3:
            if st.button("✏️ Manual Mapping", type="secondary"):
                show_manual_mapping_interface(results)

        # Display mapping table
        if 'mapping_sheet' in results:
            st.markdown("### 📋 Column Mapping Results")

            def highlight_confidence(row):
                if row['Confidence_Level'] == 'High':
                    return ['background-color: #d4f4dd'] * len(row)
                elif row['Confidence_Level'] == 'Medium':
                    return ['background-color: #fff2cc'] * len(row)
                else:
                    return ['background-color: #ffe6e6'] * len(row)

            styled_df = results['mapping_sheet'].style.apply(highlight_confidence, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)

def show_manual_mapping_interface(results):
    """Show manual mapping interface"""
    st.markdown("### ✏️ Manual Column Mapping")

    if 'mapping_sheet' in results:
        mapping_df = results['mapping_sheet'].copy()

        # CBUAE schema fields
        schema_fields = [
            "", "customer_id", "customer_type", "full_name_en", "account_type",
            "balance_current", "last_transaction_date", "dormancy_status",
            "account_status", "currency", "opening_date", "contact_attempts_made"
        ]

        # Allow manual mapping
        for idx, row in mapping_df.iterrows():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.text(row['Source_Column'])

            with col2:
                new_mapping = st.selectbox(
                    f"Map to:",
                    schema_fields,
                    index=schema_fields.index(row['Target_Field']) if row['Target_Field'] in schema_fields else 0,
                    key=f"mapping_{idx}"
                )
                mapping_df.at[idx, 'Target_Field'] = new_mapping

            with col3:
                if new_mapping:
                    st.success("✅")
                else:
                    st.warning("⚠️")

        if st.button("💾 Save Manual Mapping"):
            st.session_state.mapping_results['mapping_sheet'] = mapping_df
            st.success("✅ Manual mapping saved!")

def apply_column_mapping(data, mapping_results):
    """Apply column mapping to create processed data"""
    if not mapping_results or not mapping_results.get('success'):
        return data

    mapping_sheet = mapping_results.get('mapping_sheet')
    if mapping_sheet is None:
        return data

    # Create column mapping dictionary
    column_mapping = {}
    for _, row in mapping_sheet.iterrows():
        source_col = row['Source_Column']
        target_field = row['Target_Field']
        if target_field and target_field.strip():
            column_mapping[source_col] = target_field

    # Apply mapping
    mapped_data = data.copy()
    if column_mapping:
        mapped_data = mapped_data.rename(columns=column_mapping)

    return mapped_data

# Dormant Analysis Section
def show_dormant_analysis_section():
    """Display dormant account analysis with real agents"""
    st.markdown('<div class="cbuae-banner">💤 CBUAE Dormancy Analysis System</div>', unsafe_allow_html=True)

    if not AGENTS_STATUS['dormancy']:
        st.error("❌ CBUAE Dormancy System not available")
        return

    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process data first in the Data Processing section.")
        return

    data = st.session_state.processed_data

    # Data validation for dormancy analysis
    st.markdown("### 📊 Data Validation for Dormancy Analysis")

    if st.button("🔍 Validate Data Structure"):
        try:
            validation_results = validate_csv_structure(data)
            display_validation_results(validation_results)
        except Exception as e:
            st.error(f"❌ Validation failed: {str(e)}")

    # Run dormancy analysis
    st.markdown("### 🏃‍♂️ Run Dormancy Analysis")

    if st.button("🚀 Start Comprehensive Dormancy Analysis", type="primary"):
        with st.spinner("🔄 Running comprehensive CBUAE dormancy analysis..."):
            try:
                # Run the comprehensive analysis
                loop = get_or_create_event_loop()
                results = loop.run_until_complete(
                    run_comprehensive_dormancy_analysis_csv(
                        user_id=st.session_state.username,
                        account_data=data,
                        report_date=datetime.now().strftime('%Y-%m-%d')
                    )
                )

                st.session_state.dormancy_results = results
                st.success("✅ Dormancy analysis completed!")

            except Exception as e:
                st.error(f"❌ Dormancy analysis failed: {str(e)}")
                logger.error(f"Dormancy analysis error: {traceback.format_exc()}")

    # Display dormancy results
    if st.session_state.dormancy_results:
        display_dormancy_results(st.session_state.dormancy_results)

def display_validation_results(validation_results):
    """Display CSV validation results"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Structure Valid", "✅ Yes" if validation_results.get('structure_valid') else "❌ No")

    with col2:
        st.metric("Total Records", validation_results.get('total_records', 0))

    with col3:
        issues_count = len(validation_results.get('quality_issues', []))
        st.metric("Quality Issues", issues_count)

    if validation_results.get('missing_columns'):
        st.warning(f"⚠️ Missing required columns: {', '.join(validation_results['missing_columns'])}")

    if validation_results.get('quality_issues'):
        with st.expander("💡 Quality Recommendations"):
            for issue in validation_results['quality_issues']:
                st.write(f"• {issue}")

def display_dormancy_results(results):
    """Display comprehensive dormancy analysis results"""
    st.markdown("### 📊 Dormancy Analysis Results")

    if not results or not results.get('success'):
        st.error("❌ No valid dormancy results available")
        return

    # Summary metrics
    summary = results.get('summary', {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Accounts", summary.get('total_accounts', 0))

    with col2:
        st.metric("Dormant Accounts", summary.get('total_dormant', 0))

    with col3:
        dormancy_rate = summary.get('dormancy_rate', 0)
        st.metric("Dormancy Rate", f"{dormancy_rate:.1f}%")

    with col4:
        compliance_score = summary.get('compliance_score', 0)
        st.metric("Compliance Score", f"{compliance_score:.1f}%")

    # Agent results
    agent_results = results.get('agent_results', {})

    if agent_results:
        st.markdown("### 🤖 Dormancy Agent Results")

        for agent_name, agent_result in agent_results.items():
            if agent_result and agent_result.get('dormant_records_found', 0) > 0:
                with st.expander(f"📋 {agent_name.replace('_', ' ').title()} ({agent_result.get('dormant_records_found', 0)} accounts)"):
                    display_agent_result(agent_name, agent_result)

def display_agent_result(agent_name, agent_result):
    """Display individual agent result"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Description:** {agent_result.get('description', 'N/A')}")
        st.markdown(f"**Compliance Article:** {agent_result.get('compliance_article', 'N/A')}")
        st.markdown(f"**Records Processed:** {agent_result.get('records_processed', 0):,}")
        st.markdown(f"**Dormant Found:** {agent_result.get('dormant_records_found', 0):,}")
        st.markdown(f"**Processing Time:** {agent_result.get('processing_time', 0):.2f}s")

    with col2:
        # Download buttons
        if agent_result.get('processed_dataframe') is not None:
            df = agent_result['processed_dataframe']
            if not df.empty:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{agent_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_csv_{agent_name}"
                )

        # Generate summary button
        if st.button("📄 Generate Summary", key=f"summary_{agent_name}"):
            summary = generate_agent_summary(agent_name, agent_result)
            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name=f"{agent_name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key=f"download_summary_{agent_name}"
            )

    # Display details if available
    if agent_result.get('details'):
        details_df = pd.DataFrame(agent_result['details'])
        if not details_df.empty:
            st.dataframe(details_df.head(10), use_container_width=True)
            if len(details_df) > 10:
                st.caption(f"Showing first 10 of {len(details_df)} records")

def generate_agent_summary(agent_name, agent_result):
    """Generate text summary for agent results"""
    summary = f"""
CBUAE Dormancy Analysis Summary
==============================

Agent: {agent_name.replace('_', ' ').title()}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Compliance Article: {agent_result.get('compliance_article', 'N/A')}

Results Overview:
- Records Processed: {agent_result.get('records_processed', 0):,}
- Dormant Accounts Found: {agent_result.get('dormant_records_found', 0):,}
- Processing Time: {agent_result.get('processing_time', 0):.2f} seconds
- Analysis Status: {'Completed Successfully' if agent_result.get('validation_passed') else 'Completed with Issues'}

Description:
{agent_result.get('description', 'No description available')}

Key Findings:
"""

    if agent_result.get('details'):
        summary += f"- {len(agent_result['details'])} specific accounts identified\n"
        summary += "- Detailed breakdown available in CSV export\n"

    if agent_result.get('alerts_generated'):
        summary += "- Compliance alerts generated - immediate attention required\n"

    summary += f"""
Recommendations:
- Review all identified accounts for compliance requirements
- Update account statuses as per CBUAE guidelines
- Implement corrective actions within regulatory timeframes
- Maintain audit trail for all actions taken

Generated by CBUAE Banking Compliance Analysis System
"""

    return summary

# Compliance Analysis Section
def show_compliance_analysis_section():
    """Display compliance analysis with verification agents"""
    st.markdown('<div class="section-header">⚖️ CBUAE Compliance Verification</div>', unsafe_allow_html=True)

    if not AGENTS_STATUS['compliance']:
        st.error("❌ Compliance verification agents not available")
        return

    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process data first in the Data Processing section.")
        return

    data = st.session_state.processed_data

    # Available compliance agents
    compliance_agents = [
        {
            'name': 'Incomplete Contact Attempts',
            'description': 'Detects accounts with incomplete contact attempt processes',
            'article': 'CBUAE Art. 5',
            'agent_class': DetectIncompleteContactAttemptsAgent,
            'key': 'contact_attempts'
        },
        {
            'name': 'Unflagged Dormant Candidates',
            'description': 'Identifies accounts that should be flagged as dormant but are not',
            'article': 'CBUAE Art. 2',
            'agent_class': DetectUnflaggedDormantCandidatesAgent,
            'key': 'unflagged_dormant'
        },
        {
            'name': 'Internal Ledger Candidates',
            'description': 'Identifies accounts ready for internal ledger transfer',
            'article': 'CBUAE Art. 3',
            'agent_class': DetectInternalLedgerCandidatesAgent,
            'key': 'internal_ledger'
        },
        {
            'name': 'Statement Freeze Candidates',
            'description': 'Identifies accounts eligible for statement suppression',
            'article': 'CBUAE Art. 7.3',
            'agent_class': DetectStatementFreezeCandidatesAgent,
            'key': 'statement_freeze'
        }
    ]

    st.info("🔍 Compliance agents analyze data for regulatory compliance issues.")

    # Run all compliance checks
    if st.button("🚀 Run All Compliance Checks", type="primary"):
        with st.spinner("🔄 Running comprehensive compliance analysis..."):
            run_all_compliance_checks(data, compliance_agents)

    # Display compliance results
    if st.session_state.compliance_results:
        display_compliance_results(compliance_agents)

def run_all_compliance_checks(data, compliance_agents):
    """Run all compliance verification agents"""
    results = {}

    for agent_info in compliance_agents:
        try:
            agent_class = agent_info['agent_class']
            agent = agent_class()

            # Run agent analysis
            result = agent.analyze_compliance(data)
            results[agent_info['key']] = result

        except Exception as e:
            logger.error(f"Compliance agent {agent_info['name']} failed: {str(e)}")
            results[agent_info['key']] = {
                'success': False,
                'error': str(e),
                'violations_found': 0
            }

    st.session_state.compliance_results = results

def display_compliance_results(compliance_agents):
    """Display compliance analysis results"""
    st.markdown("### 📊 Compliance Analysis Results")

    # Summary metrics
    total_violations = sum(
        result.get('violations_found', 0)
        for result in st.session_state.compliance_results.values()
    )

    agents_run = len([r for r in st.session_state.compliance_results.values() if r.get('success')])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Agents Run", agents_run)

    with col2:
        st.metric("Total Violations", total_violations)

    with col3:
        compliance_status = "✅ Compliant" if total_violations == 0 else f"⚠️ {total_violations} Issues"
        st.metric("Compliance Status", compliance_status)

    # Individual agent results
    for agent_info in compliance_agents:
        key = agent_info['key']
        result = st.session_state.compliance_results.get(key, {})

        if result and result.get('success'):
            violations = result.get('violations_found', 0)

            with st.expander(f"📋 {agent_info['name']} ({violations} violations found)"):
                display_compliance_agent_result(agent_info, result)

def display_compliance_agent_result(agent_info, result):
    """Display individual compliance agent result"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Description:** {agent_info['description']}")
        st.markdown(f"**CBUAE Article:** {agent_info['article']}")
        st.markdown(f"**Accounts Processed:** {result.get('accounts_processed', 0):,}")
        st.markdown(f"**Violations Found:** {result.get('violations_found', 0):,}")
        st.markdown(f"**Processing Time:** {result.get('processing_time', 0):.2f}s")

    with col2:
        # Download action items
        if result.get('actions_generated'):
            actions_df = pd.DataFrame(result['actions_generated'])
            csv_data = actions_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Actions",
                data=csv_data,
                file_name=f"{agent_info['key']}_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Display violations if any
    if result.get('actions_generated'):
        actions_df = pd.DataFrame(result['actions_generated'])
        st.dataframe(actions_df.head(10), use_container_width=True)

        if len(actions_df) > 10:
            st.caption(f"Showing first 10 of {len(actions_df)} violations")

# Reports Section
def show_reports_section():
    """Display comprehensive reports and analytics"""
    st.markdown('<div class="section-header">📊 Comprehensive Reports & Analytics</div>', unsafe_allow_html=True)

    # Summary dashboard
    create_summary_dashboard()

    # Agent status overview
    st.markdown("### 🤖 Agent Status Overview")
    create_agent_status_table()

    # Data flow diagram
    st.markdown("### 🔄 Data Flow & Processing Pipeline")
    create_data_flow_visualization()

def create_summary_dashboard():
    """Create summary dashboard with key metrics"""
    st.markdown("### 📈 Executive Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    # Data metrics
    with col1:
        if st.session_state.uploaded_data is not None:
            record_count = len(st.session_state.uploaded_data)
            st.metric("Total Records", f"{record_count:,}")
        else:
            st.metric("Total Records", "0")

    # Quality metrics
    with col2:
        if st.session_state.quality_results:
            quality_score = st.session_state.quality_results.get('overall_score', 0)
            st.metric("Data Quality", f"{quality_score:.1f}%")
        else:
            st.metric("Data Quality", "N/A")

    # Mapping metrics
    with col3:
        if st.session_state.mapping_results:
            mapping_pct = st.session_state.mapping_results.get('auto_mapping_percentage', 0)
            st.metric("Auto Mapping", f"{mapping_pct:.1f}%")
        else:
            st.metric("Auto Mapping", "N/A")

    # Dormancy metrics
    with col4:
        if st.session_state.dormancy_results:
            total_dormant = st.session_state.dormancy_results.get('summary', {}).get('total_dormant', 0)
            st.metric("Dormant Accounts", f"{total_dormant:,}")
        else:
            st.metric("Dormant Accounts", "N/A")

def create_agent_status_table():
    """Create comprehensive agent status table"""
    agent_data = []

    # Data processing agents
    data_agents = [
        ('Data Upload', 'Data Processing', AGENTS_STATUS['data_upload']),
        ('Data Quality', 'Data Processing', AGENTS_STATUS['data_processing']),
        ('Data Mapping', 'Data Processing', AGENTS_STATUS['data_mapping']),
        ('BGE Embeddings', 'Data Processing', AGENTS_STATUS['bge'])
    ]

    for name, category, available in data_agents:
        records_processed = 0
        status = "Available" if available else "Not Available"

        if name == "Data Quality" and st.session_state.quality_results:
            records_processed = st.session_state.quality_results.get('total_records', 0)
            status = "Completed"

        agent_data.append({
            'Agent': name,
            'Category': category,
            'Records Processed': f"{records_processed:,}",
            'Status': status,
            'Actions': 'Download Results' if records_processed > 0 else 'Available'
        })

    # Dormancy agents
    dormancy_agents = [
        'Demand Deposit Dormancy',
        'Fixed Deposit Dormancy',
        'Investment Account Dormancy',
        'Contact Attempts Analysis',
        'CB Transfer Eligibility',
        'Foreign Currency Conversion'
    ]

    for agent_name in dormancy_agents:
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
            'Category': 'Dormancy Analysis',
            'Records Processed': f"{records_processed:,}",
            'Status': status,
            'Actions': 'Download Results' if records_processed > 0 else 'Available'
        })

    # Compliance agents
    compliance_agents = [
        'Incomplete Contact Attempts',
        'Unflagged Dormant Candidates',
        'Internal Ledger Candidates',
        'Statement Freeze Candidates'
    ]

    for agent_name in compliance_agents:
        violations_found = 0
        status = "Available" if AGENTS_STATUS['compliance'] else "Not Available"

        if st.session_state.compliance_results:
            snake_name = agent_name.lower().replace(' ', '_')
            if snake_name in st.session_state.compliance_results:
                violations_found = st.session_state.compliance_results[snake_name].get('violations_found', 0)
                status = "Completed"

        agent_data.append({
            'Agent': agent_name,
            'Category': 'Compliance Verification',
            'Records Processed': f"{violations_found:,}",
            'Status': status,
            'Actions': 'Download Actions' if violations_found > 0 else 'Available'
        })

    # Display table
    agent_df = pd.DataFrame(agent_data)
    st.dataframe(agent_df, use_container_width=True, height=600)

def create_data_flow_visualization():
    """Create data flow visualization"""
    # Simple flow diagram using text
    flow_steps = [
        "📤 Data Upload (4 Methods)",
        "🔍 Data Quality Analysis",
        "🗺️ BGE-Powered Column Mapping",
        "💤 Dormancy Analysis (6 Agents)",
        "⚖️ Compliance Verification (4 Agents)",
        "📊 Reports & Downloads"
    ]

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        for i, step in enumerate(flow_steps):
            st.markdown(f"**{i+1}.** {step}")
            if i < len(flow_steps) - 1:
                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;⬇️")

        st.success("✅ End-to-end CBUAE compliance workflow")

# Utility functions
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

def show_sidebar():
    """Display sidebar navigation and status"""
    st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.username}!")

    # Navigation
    pages = [
        "📤 Data Processing",
        "💤 Dormant Analysis",
        "⚖️ Compliance Analysis",
        "📊 Reports"
    ]

    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        pages,
        index=pages.index(f"📤 {st.session_state.current_page}") if f"📤 {st.session_state.current_page}" in pages else 0
    )

    st.session_state.current_page = selected_page.split(" ", 1)[1]

    # Data status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Status")

    if st.session_state.uploaded_data is not None:
        st.sidebar.success(f"✅ Data Loaded: {len(st.session_state.uploaded_data):,} records")

        # Show dormancy statistics if available
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
    for agent_type, status in AGENTS_STATUS.items():
        icon = "✅" if status else "❌"
        st.sidebar.caption(f"{icon} {agent_type.replace('_', ' ').title()}")

    # System information
    st.sidebar.markdown("### ℹ️ System Info")
    st.sidebar.caption(f"BGE Embeddings: {'✅ Available' if AGENTS_STATUS['bge'] else '❌ Unavailable'}")
    st.sidebar.caption(f"Real-time Analysis: {'✅ Enabled' if AGENTS_STATUS['dormancy'] else '❌ Mock Mode'}")

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
        show_data_upload_section()

        if st.session_state.uploaded_data is not None:
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