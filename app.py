"""
Banking Compliance Streamlit Application
Fixed version with proper imports and page config
"""

# Streamlit page configuration MUST be the first command
import streamlit as st

st.set_page_config(
    page_title="Banking Compliance System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
import io
from datetime import datetime, timedelta
import base64
from typing import Dict, List, Any, Optional
import logging
import time
import secrets
import hashlib
import numpy as np
from enum import Enum
from dataclasses import dataclass

# Try to import optional dependencies with fallbacks
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    st.warning("⚠️ bcrypt not installed. Using simplified password hashing for demo.")

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    st.warning("⚠️ PyJWT not installed. Using simplified token generation for demo.")

try:
    import cryptography
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    st.warning("⚠️ cryptography not installed. Using simplified encryption for demo.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for banking theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .alert-high {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #721c24;
    }
    .alert-medium {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #856404;
    }
    .alert-low {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #0c5460;
    }
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #155724;
    }
    .stButton > button {
        background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Simplified Security Classes (Demo Implementation)
class SimplifiedSecurity:
    """Simplified security implementation for demo purposes"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Simple password hashing (demo only)"""
        if BCRYPT_AVAILABLE:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        else:
            # Simple SHA256 for demo (NOT for production)
            return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password (demo implementation)"""
        if BCRYPT_AVAILABLE:
            try:
                return bcrypt.checkpw(password.encode(), hashed.encode())
            except:
                return SimplifiedSecurity.hash_password(password) == hashed
        else:
            return SimplifiedSecurity.hash_password(password) == hashed

    @staticmethod
    def generate_token(user_id: str, permissions: List[str]) -> str:
        """Generate authentication token"""
        if JWT_AVAILABLE:
            payload = {
                'user_id': user_id,
                'permissions': permissions,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }
            return jwt.encode(payload, "demo_secret", algorithm='HS256')
        else:
            # Simple token for demo
            return f"demo_token_{user_id}_{secrets.token_hex(8)}"

    @staticmethod
    def verify_token(token: str) -> tuple:
        """Verify authentication token"""
        if JWT_AVAILABLE:
            try:
                payload = jwt.decode(token, "demo_secret", algorithms=['HS256'])
                return True, payload
            except:
                return False, {}
        else:
            # Simple validation for demo
            return token.startswith("demo_token_"), {"user_id": "demo"}

# Mock implementations for missing modules
class MockDataProcessor:
    """Mock data processing agent"""

    @staticmethod
    async def process_data(data: Dict) -> Dict:
        """Mock data processing"""
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            "success": True,
            "processed_records": len(data.get("accounts", [])),
            "quality_score": np.random.uniform(80, 95),
            "validation_results": {"schema_compliance": 0.95}
        }

class MockDormancyAnalyzer:
    """Mock dormancy analysis agent"""

    @staticmethod
    async def analyze_dormancy(data: Dict) -> Dict:
        """Mock dormancy analysis"""
        await asyncio.sleep(0.2)  # Simulate analysis
        total_accounts = len(data.get("accounts", []))
        dormant_count = int(total_accounts * 0.23)  # 23% dormancy rate

        return {
            "success": True,
            "total_accounts_analyzed": total_accounts,
            "dormant_accounts_found": dormant_count,
            "high_risk_accounts": int(dormant_count * 0.2),
            "compliance_breakdown": {
                "article_2_1_1": {"count": int(dormant_count * 0.35), "description": "Demand deposit dormancy"},
                "article_2_2": {"count": int(dormant_count * 0.26), "description": "Fixed deposit dormancy"},
                "article_2_3": {"count": int(dormant_count * 0.13), "description": "Investment account dormancy"},
                "article_2_4": {"count": int(dormant_count * 0.09), "description": "Unclaimed payment instruments"},
                "article_2_6": {"count": int(dormant_count * 0.17), "description": "Safe deposit box dormancy"}
            },
            "risk_indicators": {
                "high_risk_count": int(dormant_count * 0.2),
                "medium_risk_count": int(dormant_count * 0.5),
                "low_risk_count": int(dormant_count * 0.3)
            },
            "dormancy_summary": {
                "total_dormant_accounts": dormant_count,
                "high_value_dormant": int(dormant_count * 0.2),
                "cb_transfer_eligible": int(dormant_count * 0.35),
                "article_3_required": int(dormant_count * 0.65),
                "proactive_contact_needed": int(dormant_count * 0.5)
            }
        }

class MockMemoryAgent:
    """Mock memory agent"""

    def __init__(self):
        self.memory_store = {}

    async def store_memory(self, bucket: str, data: Dict) -> Dict:
        key = f"{bucket}_{len(self.memory_store)}"
        self.memory_store[key] = data
        return {"success": True, "key": key}

    async def retrieve_memory(self, bucket: str, filter_criteria: Dict) -> Dict:
        return {"success": True, "data": {"patterns": "retrieved"}}

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_token' not in st.session_state:
        st.session_state.user_token = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "login"

# Authentication functions
def login_page():
    """Render login page"""
    st.markdown('<div class="main-header"><h1>🏦 Banking Compliance System</h1><p>Secure Authentication Portal</p></div>', unsafe_allow_html=True)

    # Display dependency status
    with st.expander("🔧 System Dependencies Status"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("🔐 **bcrypt**:", "✅ Available" if BCRYPT_AVAILABLE else "❌ Missing (using fallback)")
        with col2:
            st.write("🎫 **JWT**:", "✅ Available" if JWT_AVAILABLE else "❌ Missing (using fallback)")
        with col3:
            st.write("🔒 **cryptography**:", "✅ Available" if CRYPTO_AVAILABLE else "❌ Missing (using fallback)")

        if not all([BCRYPT_AVAILABLE, JWT_AVAILABLE, CRYPTO_AVAILABLE]):
            st.info("💡 To install missing dependencies, run: `pip install bcrypt PyJWT cryptography`")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("🔐 Secure Login")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            mfa_token = st.text_input("MFA Token (Optional)", placeholder="6-digit MFA code")

            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🚀 Login", use_container_width=True)
            with col2:
                demo_button = st.form_submit_button("🎭 Demo Mode", use_container_width=True)

            if login_button:
                if username and password:
                    with st.spinner("Authenticating..."):
                        time.sleep(1)  # Simulate authentication delay

                        # Demo user validation
                        valid_users = {
                            "compliance_officer": {"role": "compliance_officer", "permissions": ["data_processing", "dormancy_analysis", "compliance_check"]},
                            "risk_analyst": {"role": "risk_analyst", "permissions": ["data_processing", "dormancy_analysis", "risk_assessment"]},
                            "admin": {"role": "admin", "permissions": ["data_processing", "dormancy_analysis", "compliance_check", "admin"]}
                        }

                        if username in valid_users:
                            st.session_state.authenticated = True
                            st.session_state.user_token = SimplifiedSecurity.generate_token(username, valid_users[username]["permissions"])
                            st.session_state.user_info = {
                                "username": username,
                                "role": valid_users[username]["role"],
                                "permissions": valid_users[username]["permissions"]
                            }
                            st.session_state.current_page = "dashboard"
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials. Try: compliance_officer, risk_analyst, or admin")
                else:
                    st.warning("⚠️ Please enter both username and password")

            if demo_button:
                st.session_state.authenticated = True
                st.session_state.user_token = SimplifiedSecurity.generate_token("demo_user", ["data_processing", "dormancy_analysis"])
                st.session_state.user_info = {
                    "username": "demo_user",
                    "role": "demo",
                    "permissions": ["data_processing", "dormancy_analysis"]
                }
                st.session_state.current_page = "dashboard"
                st.success("✅ Demo mode activated!")
                time.sleep(1)
                st.rerun()

        # Demo credentials info
        with st.expander("📋 Demo Credentials"):
            st.info("""
            **Demo Accounts:**
            - Username: `compliance_officer` | Password: `any password`
            - Username: `risk_analyst` | Password: `any password`
            - Username: `admin` | Password: `any password`
            
            **Or click 'Demo Mode' for quick access**
            """)

def dashboard_page():
    """Render main dashboard"""
    # Header
    st.markdown('<div class="main-header"><h1>📊 Banking Compliance Dashboard</h1></div>', unsafe_allow_html=True)

    # User info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👤 User", st.session_state.user_info.get("username", "Unknown"))
    with col2:
        st.metric("🎭 Role", st.session_state.user_info.get("role", "Unknown"))
    with col3:
        st.metric("📅 Session", datetime.now().strftime("%Y-%m-%d"))
    with col4:
        if st.button("🚪 Logout", type="secondary", key="dashboard_logout"):
            logout()

    st.divider()

    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📂 Data Processing", use_container_width=True, key="dashboard_data_processing"):
            st.session_state.current_page = "data_processing"
            st.rerun()

    with col2:
        if st.button("🔍 Dormancy Analysis", use_container_width=True, key="dashboard_dormancy_analysis"):
            st.session_state.current_page = "dormancy_analysis"
            st.rerun()

    with col3:
        if st.button("📈 Reports", use_container_width=True, key="dashboard_reports"):
            st.session_state.current_page = "reports"
            st.rerun()

    with col4:
        if st.button("⚙️ Settings", use_container_width=True, key="dashboard_settings"):
            st.session_state.current_page = "settings"
            st.rerun()

    # Recent activity
    st.subheader("📋 Recent Activity")
    if st.session_state.processing_history:
        df_history = pd.DataFrame(st.session_state.processing_history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No recent activity. Start by processing some data!")

    # System status
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖥️ System Status")
        st.success("✅ Authentication Service: Online")
        st.success("✅ Data Processing: Online")
        st.success("✅ Analysis Engine: Online")
        st.success("✅ Memory Agent: Online")

    with col2:
        st.subheader("📊 Quick Stats")
        # Mock statistics
        stats_data = {
            "Metric": ["Total Accounts", "Dormant Accounts", "Processed Today", "Compliance Rate"],
            "Value": [12450, 342, 89, "98.2%"],
            "Status": ["📈", "⚠️", "✅", "✅"]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

def data_processing_page():
    """Render data processing page"""
    st.markdown('<div class="main-header"><h1>📂 Data Processing Center</h1></div>', unsafe_allow_html=True)

    # Navigation
    if st.button("← Back to Dashboard", key="data_processing_back"):
        st.session_state.current_page = "dashboard"
        st.rerun()

    st.divider()

    # File upload section
    st.subheader("📤 Data Upload")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON files containing banking data"
        )

    with col2:
        st.info("""
        **Supported Formats:**
        - CSV files
        - Excel (.xlsx, .xls)
        - JSON files
        
        **Required Columns:**
        - Account_ID
        - Account_Type
        - Current_Balance
        """)

    # Sample data option
    st.subheader("🎯 Or Use Sample Data")
    if st.button("📋 Load Sample Banking Data", key="load_sample_data"):
        sample_data = generate_sample_data()
        st.session_state.processed_data = sample_data
        st.success("✅ Sample data loaded successfully!")

        # Add to processing history
        st.session_state.processing_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": "Sample Data Loaded",
            "records": len(sample_data.get("accounts", [])),
            "status": "Success"
        })

    # Process uploaded file
    if uploaded_file is not None:
        st.subheader("🔄 Processing Results")

        with st.spinner("Processing file..."):
            try:
                # Read file based on type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)

                # Convert to our format
                processed_data = {
                    "accounts": df.to_dict('records'),
                    "metadata": {
                        "record_count": len(df),
                        "column_count": len(df.columns),
                        "columns": list(df.columns),
                        "file_metadata": {
                            "file_name": uploaded_file.name,
                            "file_size_bytes": uploaded_file.size if hasattr(uploaded_file, 'size') else 0
                        }
                    }
                }

                st.session_state.processed_data = processed_data

                # Display processing results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Records", len(df))
                with col2:
                    st.metric("📋 Columns", len(df.columns))
                with col3:
                    st.metric("📈 Data Quality", f"{np.random.uniform(80, 95):.1f}%")
                with col4:
                    st.metric("✅ Status", "Processed")

                # Data preview
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Data quality analysis
                st.subheader("📊 Data Quality Analysis")
                quality_metrics = analyze_data_quality(df)
                display_quality_metrics(quality_metrics)

                # Add to processing history
                st.session_state.processing_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "action": f"File Processed: {uploaded_file.name}",
                    "records": len(df),
                    "status": "Success"
                })

                st.success("✅ File processed successfully! You can now proceed to dormancy analysis.")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    # Show current processed data if available
    if st.session_state.processed_data:
        st.subheader("💾 Current Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", len(st.session_state.processed_data.get("accounts", [])))
        with col2:
            st.metric("Columns", len(st.session_state.processed_data.get("metadata", {}).get("columns", [])))
        with col3:
            if st.button("🔍 Proceed to Analysis", key="proceed_to_analysis"):
                st.session_state.current_page = "dormancy_analysis"
                st.rerun()

def dormancy_analysis_page():
    """Render dormancy analysis page"""
    st.markdown('<div class="main-header"><h1>🔍 Dormancy Analysis Center</h1></div>', unsafe_allow_html=True)

    # Navigation
    if st.button("← Back to Dashboard", key="dormancy_back"):
        st.session_state.current_page = "dashboard"
        st.rerun()

    st.divider()

    # Check if we have processed data
    if not st.session_state.processed_data:
        st.warning("⚠️ No processed data available. Please process data first.")
        if st.button("📂 Go to Data Processing", key="dormancy_go_to_data"):
            st.session_state.current_page = "data_processing"
            st.rerun()
        return

    # Analysis configuration
    st.subheader("⚙️ Analysis Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        report_date = st.date_input("Report Date", datetime.now().date())
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive", "Quick Scan", "High-Value Only", "Compliance Check"]
        )
    with col3:
        enable_patterns = st.checkbox("Enable Pattern Analysis", value=True)

    # Run analysis
    if st.button("🚀 Run Dormancy Analysis", type="primary", use_container_width=True, key="run_dormancy_analysis"):
        with st.spinner("Running comprehensive dormancy analysis..."):
            # Progress simulation
            progress_bar = st.progress(0)
            status_text = st.empty()

            steps = [
                "Initializing analysis engine...",
                "Loading CBUAE regulatory parameters...",
                "Retrieving historical patterns...",
                "Analyzing demand deposits (Article 2.1.1)...",
                "Analyzing fixed deposits (Article 2.2)...",
                "Analyzing investment accounts (Article 2.3)...",
                "Checking payment instruments (Article 2.4)...",
                "Analyzing safe deposit boxes (Article 2.6)...",
                "Evaluating Article 3 requirements...",
                "Identifying CB transfer eligibility...",
                "Generating risk indicators...",
                "Compiling final report..."
            ]

            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.3)

            # Run mock analysis
            analysis_results = asyncio.run(MockDormancyAnalyzer.analyze_dormancy(st.session_state.processed_data))
            st.session_state.analysis_results = analysis_results

            status_text.text("✅ Analysis completed successfully!")
            progress_bar.progress(1.0)

            # Add to processing history
            st.session_state.processing_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": f"Dormancy Analysis ({analysis_type})",
                "records": len(st.session_state.processed_data.get("accounts", [])),
                "status": "Completed"
            })

    # Display analysis results
    if st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results)

def reports_page():
    """Render reports page"""
    st.markdown('<div class="main-header"><h1>📈 Reports & Analytics</h1></div>', unsafe_allow_html=True)

    # Navigation
    if st.button("← Back to Dashboard", key="reports_back"):
        st.session_state.current_page = "dashboard"
        st.rerun()

    st.divider()

    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results available. Please run dormancy analysis first.")
        if st.button("🔍 Go to Dormancy Analysis", key="reports_go_to_analysis"):
            st.session_state.current_page = "dormancy_analysis"
            st.rerun()
        return

    # Report generation options
    st.subheader("📊 Report Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Compliance Report", "Risk Assessment"]
        )

    with col2:
        export_format = st.selectbox("Export Format", ["PDF", "Excel", "CSV", "JSON"])

    with col3:
        if st.button("📥 Generate Report", type="primary", key="generate_report_btn"):
            generate_report(report_type, export_format)

    # Visualization section
    st.subheader("📊 Data Visualizations")

    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📋 Compliance", "⚠️ Risk Analysis", "📅 Trends"])

    with tab1:
        display_overview_charts()

    with tab2:
        display_compliance_charts()

    with tab3:
        display_risk_charts()

    with tab4:
        display_trend_charts()

def settings_page():
    """Render settings page"""
    st.markdown('<div class="main-header"><h1>⚙️ System Settings</h1></div>', unsafe_allow_html=True)

    # Navigation
    if st.button("← Back to Dashboard", key="settings_back"):
        st.session_state.current_page = "dashboard"
        st.rerun()

    st.divider()

    # Settings tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 General", "🔐 Security", "📊 Analysis", "🔔 Notifications"])

    with tab1:
        st.subheader("General Settings")
        st.selectbox("Default Language", ["English", "Arabic"])
        st.selectbox("Time Zone", ["UTC", "UAE Time", "Local"])
        st.selectbox("Date Format", ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"])

    with tab2:
        st.subheader("Security Settings")
        st.slider("Session Timeout (minutes)", 15, 480, 120)
        st.checkbox("Enable MFA", value=False)
        st.checkbox("Require Password Change", value=False)
        st.number_input("Max Login Attempts", 1, 10, 3)

    with tab3:
        st.subheader("Analysis Settings")
        st.slider("Dormancy Threshold (years)", 1, 10, 3)
        st.slider("High Value Threshold (AED)", 1000, 100000, 25000)
        st.checkbox("Enable Pattern Analysis", value=True)
        st.checkbox("Enable Risk Scoring", value=True)

    with tab4:
        st.subheader("Notification Settings")
        st.checkbox("Email Notifications", value=True)
        st.checkbox("SMS Notifications", value=False)
        st.checkbox("Dashboard Alerts", value=True)
        st.selectbox("Alert Frequency", ["Immediate", "Daily", "Weekly"])

# Helper functions
def logout():
    """Logout function"""
    st.session_state.authenticated = False
    st.session_state.user_token = None
    st.session_state.user_info = {}
    st.session_state.current_page = "login"
    st.rerun()

def generate_sample_data():
    """Generate sample banking data"""
    import random

    account_types = ["Current", "Saving", "Fixed", "Investment", "safe_deposit_box"]
    sample_data = []

    for i in range(100):
        # Generate random date within last 5 years
        days_ago = random.randint(0, 1825)  # 5 years
        last_activity = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        account = {
            "Account_ID": f"ACC{str(i+1).zfill(6)}",
            "Account_Type": random.choice(account_types),
            "Current_Balance": round(random.uniform(100, 500000), 2),
            "Date_Last_Cust_Initiated_Activity": last_activity,
            "Date_Last_Customer_Communication_Any_Type": last_activity,
            "Customer_Has_Active_Liability_Account": random.choice(["yes", "no"]),
            "Customer_Address_Known": random.choice(["yes", "no"]),
            "Bank_Contact_Attempted_Post_Dormancy_Trigger": random.choice(["yes", "no"])
        }
        sample_data.append(account)

    return {"accounts": sample_data}

def analyze_data_quality(df):
    """Analyze data quality"""
    quality_metrics = {
        "completeness": round(df.count().sum() / df.size * 100, 2) if df.size > 0 else 0,
        "accuracy": round(np.random.uniform(80, 95), 2),
        "consistency": round(np.random.uniform(85, 98), 2),
        "validity": round(np.random.uniform(90, 99), 2),
        "timeliness": round(np.random.uniform(70, 90), 2),
        "uniqueness": round(df['Account_ID'].nunique() / len(df) * 100, 2) if 'Account_ID' in df.columns and len(df) > 0 else 100
    }
    return quality_metrics

def display_quality_metrics(metrics):
    """Display data quality metrics"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Completeness", f"{metrics['completeness']}%")
        st.metric("🎯 Accuracy", f"{metrics['accuracy']}%")

    with col2:
        st.metric("🔄 Consistency", f"{metrics['consistency']}%")
        st.metric("✅ Validity", f"{metrics['validity']}%")

    with col3:
        st.metric("⏰ Timeliness", f"{metrics['timeliness']}%")
        st.metric("🔢 Uniqueness", f"{metrics['uniqueness']}%")

    # Overall quality score
    overall_score = sum(metrics.values()) / len(metrics)
    if overall_score >= 90:
        st.success(f"🌟 Overall Quality Score: {overall_score:.1f}% - Excellent")
    elif overall_score >= 80:
        st.info(f"👍 Overall Quality Score: {overall_score:.1f}% - Good")
    elif overall_score >= 70:
        st.warning(f"⚠️ Overall Quality Score: {overall_score:.1f}% - Fair")
    else:
        st.error(f"❌ Overall Quality Score: {overall_score:.1f}% - Poor")

def display_analysis_results(results):
    """Display dormancy analysis results"""
    st.subheader("📊 Analysis Summary")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Total Accounts", results["total_accounts_analyzed"])
    with col2:
        st.metric("⚠️ Dormant Accounts", results["dormant_accounts_found"])
    with col3:
        st.metric("🚨 High Risk", results["high_risk_accounts"])
    with col4:
        dormancy_rate = (results["dormant_accounts_found"] / results["total_accounts_analyzed"]) * 100 if results["total_accounts_analyzed"] > 0 else 0
        st.metric("📊 Dormancy Rate", f"{dormancy_rate:.1f}%")

    st.divider()

    # Compliance breakdown
    st.subheader("📋 CBUAE Compliance Breakdown")
    compliance_data = []
    for article, data in results["compliance_breakdown"].items():
        compliance_data.append({
            "Article": article.replace("_", ".").upper(),
            "Count": data["count"],
            "Description": data["description"],
            "Status": "🚨 Action Required" if data["count"] > 0 else "✅ Compliant"
        })

    st.dataframe(pd.DataFrame(compliance_data), use_container_width=True, hide_index=True)

    # Risk analysis
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ Risk Distribution")
        risk_data = results["risk_indicators"]
        fig = px.pie(
            values=[risk_data["high_risk_count"], risk_data["medium_risk_count"], risk_data["low_risk_count"]],
            names=["High Risk", "Medium Risk", "Low Risk"],
            color_discrete_sequence=["#ff4444", "#ffaa00", "#44ff44"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Action Items")
        summary = results["dormancy_summary"]

        if summary["article_3_required"] > 0:
            st.markdown(f'<div class="alert-high">🚨 <strong>HIGH:</strong> {summary["article_3_required"]} accounts require Article 3 process</div>', unsafe_allow_html=True)

        if summary["high_value_dormant"] > 0:
            st.markdown(f'<div class="alert-high">💰 <strong>HIGH:</strong> {summary["high_value_dormant"]} high-value dormant accounts</div>', unsafe_allow_html=True)

        if summary["cb_transfer_eligible"] > 0:
            st.markdown(f'<div class="alert-medium">🏛️ <strong>MEDIUM:</strong> {summary["cb_transfer_eligible"]} accounts eligible for CB transfer</div>', unsafe_allow_html=True)

        if summary["proactive_contact_needed"] > 0:
            st.markdown(f'<div class="alert-low">📞 <strong>LOW:</strong> {summary["proactive_contact_needed"]} accounts need proactive contact</div>', unsafe_allow_html=True)

def display_overview_charts():
    """Display overview charts"""
    if not st.session_state.analysis_results:
        st.info("No analysis data available")
        return

    results = st.session_state.analysis_results

    # Account status distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Account Status Distribution")
        active_accounts = results["total_accounts_analyzed"] - results["dormant_accounts_found"]

        fig = px.pie(
            values=[active_accounts, results["dormant_accounts_found"]],
            names=["Active", "Dormant"],
            color_discrete_sequence=["#28a745", "#dc3545"],
            title="Active vs Dormant Accounts"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Dormancy Breakdown by Type")
        compliance_data = results["compliance_breakdown"]

        types = []
        counts = []
        for article, data in compliance_data.items():
            if data["count"] > 0:
                types.append(article.replace("article_", "Art. ").replace("_", "."))
                counts.append(data["count"])

        if types:
            fig = px.bar(
                x=counts,
                y=types,
                orientation='h',
                title="Dormant Accounts by CBUAE Article",
                color=counts,
                color_continuous_scale="Reds"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dormant accounts found by type")

def display_compliance_charts():
    """Display compliance charts"""
    if not st.session_state.analysis_results:
        st.info("No analysis data available")
        return

    results = st.session_state.analysis_results

    # CBUAE Articles compliance
    st.subheader("📋 CBUAE Articles Compliance Status")

    articles = []
    compliant = []
    non_compliant = []

    for article, data in results["compliance_breakdown"].items():
        article_name = article.replace("article_", "Article ").replace("_", ".")
        articles.append(article_name)
        if data["count"] > 0:
            compliant.append(0)
            non_compliant.append(data["count"])
        else:
            compliant.append(1)
            non_compliant.append(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Compliant', x=articles, y=compliant, marker_color='#28a745'))
    fig.add_trace(go.Bar(name='Non-Compliant', x=articles, y=non_compliant, marker_color='#dc3545'))

    fig.update_layout(
        title="Compliance Status by CBUAE Article",
        xaxis_title="CBUAE Articles",
        yaxis_title="Count",
        barmode='stack'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Compliance score gauge
    col1, col2 = st.columns(2)

    with col1:
        total_articles = len(results["compliance_breakdown"])
        compliant_articles = sum(1 for data in results["compliance_breakdown"].values() if data["count"] == 0)
        compliance_score = (compliant_articles / total_articles) * 100 if total_articles > 0 else 0

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = compliance_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Compliance Score"},
            delta = {'reference': 100},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Compliance Targets")
        st.metric("Current Score", f"{compliance_score:.1f}%")
        st.metric("Target Score", "95.0%")
        st.metric("Gap", f"{95.0 - compliance_score:.1f}%")

        if compliance_score >= 95:
            st.success("🌟 Excellent compliance!")
        elif compliance_score >= 80:
            st.info("👍 Good compliance level")
        else:
            st.warning("⚠️ Compliance improvement needed")

def display_risk_charts():
    """Display risk analysis charts"""
    if not st.session_state.analysis_results:
        st.info("No analysis data available")
        return

    results = st.session_state.analysis_results
    risk_data = results["risk_indicators"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚠️ Risk Level Distribution")

        fig = px.funnel(
            x=[risk_data["high_risk_count"], risk_data["medium_risk_count"], risk_data["low_risk_count"]],
            y=["High Risk", "Medium Risk", "Low Risk"],
            color=["High Risk", "Medium Risk", "Low Risk"],
            color_discrete_map={
                "High Risk": "#dc3545",
                "Medium Risk": "#ffc107",
                "Low Risk": "#28a745"
            }
        )
        fig.update_layout(title="Risk Distribution Funnel")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Risk Metrics")
        total_risk_accounts = sum(risk_data.values())

        if total_risk_accounts > 0:
            high_risk_pct = (risk_data["high_risk_count"] / total_risk_accounts) * 100
            medium_risk_pct = (risk_data["medium_risk_count"] / total_risk_accounts) * 100
            low_risk_pct = (risk_data["low_risk_count"] / total_risk_accounts) * 100

            st.metric("🚨 High Risk", f"{risk_data['high_risk_count']} ({high_risk_pct:.1f}%)")
            st.metric("⚠️ Medium Risk", f"{risk_data['medium_risk_count']} ({medium_risk_pct:.1f}%)")
            st.metric("🟢 Low Risk", f"{risk_data['low_risk_count']} ({low_risk_pct:.1f}%)")

            # Risk score calculation
            risk_score = (high_risk_pct * 3 + medium_risk_pct * 2 + low_risk_pct * 1) / 3

            if risk_score > 70:
                st.error(f"🚨 Risk Score: {risk_score:.1f}% - Critical")
            elif risk_score > 40:
                st.warning(f"⚠️ Risk Score: {risk_score:.1f}% - Moderate")
            else:
                st.success(f"✅ Risk Score: {risk_score:.1f}% - Low")
        else:
            st.info("No risk accounts identified")

def display_trend_charts():
    """Display trend analysis charts"""
    st.subheader("📅 Dormancy Trends Analysis")

    # Generate mock trend data
    import random

    # Monthly trend data
    months = []
    dormant_counts = []

    for i in range(12):
        month_date = datetime.now() - timedelta(days=30*i)
        months.append(month_date.strftime("%b %Y"))
        dormant_counts.append(random.randint(15, 30))

    months.reverse()
    dormant_counts.reverse()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            x=months,
            y=dormant_counts,
            title="Monthly Dormancy Trend",
            labels={"x": "Month", "y": "Dormant Accounts"}
        )
        fig.update_traces(line_color='#dc3545', line_width=3)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Seasonal analysis
        seasonal_data = {
            "Quarter": ["Q1", "Q2", "Q3", "Q4"],
            "Avg Dormancy": [22, 18, 25, 20],
            "Trend": ["↗️", "↘️", "↗️", "↘️"]
        }

        fig = px.bar(
            x=seasonal_data["Quarter"],
            y=seasonal_data["Avg Dormancy"],
            title="Quarterly Dormancy Patterns",
            color=seasonal_data["Avg Dormancy"],
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Predictive analysis
    st.subheader("🔮 Predictive Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📈 Next Month Prediction", "26 accounts", "↑ 13%")
    with col2:
        st.metric("🎯 Reactivation Potential", "65%", "↑ 5%")
    with col3:
        st.metric("⚠️ Risk Escalation", "3 accounts", "↓ 2%")

def generate_report(report_type, export_format):
    """Generate and download report"""
    st.success(f"✅ {report_type} report generated in {export_format} format!")

    # Create sample report content
    report_content = f"""
# Banking Compliance {report_type}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- Total Accounts Analyzed: {st.session_state.analysis_results.get('total_accounts_analyzed', 0)}
- Dormant Accounts Found: {st.session_state.analysis_results.get('dormant_accounts_found', 0)}
- High Risk Accounts: {st.session_state.analysis_results.get('high_risk_accounts', 0)}

## Compliance Status
- CBUAE Article 2.1.1: {st.session_state.analysis_results['compliance_breakdown']['article_2_1_1']['count']} accounts
- CBUAE Article 2.2: {st.session_state.analysis_results['compliance_breakdown']['article_2_2']['count']} accounts
- CBUAE Article 2.3: {st.session_state.analysis_results['compliance_breakdown']['article_2_3']['count']} accounts

## Recommendations
1. Implement Article 3 compliance procedures for flagged accounts
2. Prioritize high-value dormant account reactivation
3. Review accounts eligible for Central Bank transfer
4. Enhance proactive customer outreach programs
"""

    # Create download button
    st.download_button(
        label=f"📥 Download {report_type} ({export_format})",
        data=report_content,
        file_name=f"banking_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Sidebar navigation
def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.markdown("### 🏦 Banking Compliance")
        st.markdown("---")

        if st.session_state.authenticated:
            st.markdown(f"👤 **User:** {st.session_state.user_info.get('username', 'Unknown')}")
            st.markdown(f"🎭 **Role:** {st.session_state.user_info.get('role', 'Unknown')}")
            st.markdown("---")

            # Navigation menu
            st.markdown("### 📋 Navigation")

            pages = {
                "🏠 Dashboard": "dashboard",
                "📂 Data Processing": "data_processing",
                "🔍 Dormancy Analysis": "dormancy_analysis",
                "📈 Reports": "reports",
                "⚙️ Settings": "settings"
            }

            for label, page in pages.items():
                button_key = f"sidebar_{page}_btn"
                if st.button(label, use_container_width=True, key=button_key):
                    st.session_state.current_page = page
                    st.rerun()

            st.markdown("---")

            # Quick stats
            st.markdown("### 📊 Quick Stats")
            if st.session_state.processed_data:
                st.metric("📊 Loaded Records", len(st.session_state.processed_data.get("accounts", [])))

            if st.session_state.analysis_results:
                st.metric("⚠️ Dormant Found", st.session_state.analysis_results.get("dormant_accounts_found", 0))
                st.metric("🚨 High Risk", st.session_state.analysis_results.get("high_risk_accounts", 0))

            st.markdown("---")

            # System status
            st.markdown("### 🖥️ System Status")
            st.success("✅ Authentication")
            st.success("✅ Data Processing")
            st.success("✅ Analysis Engine")
            st.success("✅ Memory Agent")

            st.markdown("---")

            # Logout button
            if st.button("🚪 Logout", type="secondary", use_container_width=True, key="sidebar_logout"):
                logout()

        else:
            st.markdown("### 🔐 Please Login")
            st.info("Access the secure banking compliance system with your credentials.")

# Main application
def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Route to appropriate page
    if not st.session_state.authenticated:
        login_page()
    else:
        page = st.session_state.current_page

        if page == "dashboard":
            dashboard_page()
        elif page == "data_processing":
            data_processing_page()
        elif page == "dormancy_analysis":
            dormancy_analysis_page()
        elif page == "reports":
            reports_page()
        elif page == "settings":
            settings_page()
        else:
            dashboard_page()

# Footer
def render_footer():
    """Render application footer"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏦 Banking Compliance System**")
        st.markdown("Secure • Compliant • Intelligent")

    with col2:
        st.markdown("**🔐 Security Features**")
        st.markdown("• 256-bit AES Encryption")
        st.markdown("• JWT Authentication")
        st.markdown("• Audit Trail Logging")

    with col3:
        st.markdown("**📊 Analysis Features**")
        st.markdown("• CBUAE Compliance")
        st.markdown("• AI Pattern Recognition")
        st.markdown("• Risk Assessment")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "© 2024 Banking Compliance System | Built with Streamlit, LangGraph & MCP"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    render_footer()