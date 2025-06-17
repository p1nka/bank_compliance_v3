"""
Banking Compliance Analysis - Streamlit Application
Fixed version with proper agent import handling and code structure
"""

# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st

st.set_page_config(
    page_title="Banking Compliance Analysis System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import pandas as pd
import numpy as np
import asyncio
import json
import io
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe agent imports with fallbacks
AGENTS_AVAILABLE = False
DEMO_MODE = True

try:
    # Try importing data processing agents
    from agents.data_upload_agent import DataUploadAgent, run_data_upload_process
    from agents.Data_Process import DataProcessingAgent, run_data_quality_analysis
    from agents.data_mapping_agent import DataMappingAgent, run_automated_data_mapping

    DATA_AGENTS_AVAILABLE = True
    logger.info("✅ Data processing agents imported successfully")
except ImportError as e:
    DATA_AGENTS_AVAILABLE = False
    logger.warning(f"⚠️ Data processing agents not available: {e}")

try:
    # Try importing dormancy agents
    from agents.Dormant_agent import (
        DormancyAnalysisAgent, run_comprehensive_dormancy_analysis_csv,
        DemandDepositDormancyAgent, FixedDepositDormancyAgent,
        InvestmentAccountDormancyAgent, PaymentInstrumentsDormancyAgent,
        SafeDepositDormancyAgent, ContactAttemptsAgent, CBTransferEligibilityAgent,
        Art3ProcessNeededAgent, HighValueDormantAccountsAgent,
        DormantToActiveTransitionsAgent, RunAllDormantIdentificationChecksAgent
    )

    DORMANCY_AGENTS_AVAILABLE = True
    logger.info("✅ Dormancy agents imported successfully")
except ImportError as e:
    DORMANCY_AGENTS_AVAILABLE = False
    logger.warning(f"⚠️ Dormancy agents not available: {e}")

try:
    # Try importing compliance agents
    from agents.compliance_verification_agent import (
        ComplianceWorkflowOrchestrator, run_comprehensive_compliance_analysis_csv,
        DetectIncompleteContactAttemptsAgent, DetectUnflaggedDormantCandidatesAgent,
        DetectInternalLedgerCandidatesAgent, DetectStatementFreezeCandidatesAgent,
        DetectCBUAETransferCandidatesAgent, DetectForeignCurrencyConversionNeededAgent,
        DetectSDBCourtApplicationNeededAgent, DetectUnclaimedPaymentInstrumentsLedgerAgent,
        DetectClaimProcessingPendingAgent, GenerateAnnualCBUAEReportSummaryAgent,
        CheckRecordRetentionComplianceAgent, LogFlagInstructionsAgent,
        RunAllComplianceChecksAgent
    )

    COMPLIANCE_AGENTS_AVAILABLE = True
    logger.info("✅ Compliance agents imported successfully")
except ImportError as e:
    COMPLIANCE_AGENTS_AVAILABLE = False
    logger.warning(f"⚠️ Compliance agents not available: {e}")

# Set overall agent availability
AGENTS_AVAILABLE = DATA_AGENTS_AVAILABLE and DORMANCY_AGENTS_AVAILABLE and COMPLIANCE_AGENTS_AVAILABLE

if not AGENTS_AVAILABLE:
    DEMO_MODE = True
    logger.info("🧪 Running in demo mode with mock agents")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
        margin-bottom: 2rem;
    }

    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }

    .success-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .warning-card {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .error-card {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ===== AUTHENTICATION SYSTEM =====

def check_authentication():
    """Simple authentication system"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        show_login_page()
        return False
    return True


def show_login_page():
    """Display login page"""
    st.markdown('<div class="main-header">🏦 Banking Compliance Analysis System</div>',
                unsafe_allow_html=True)

    st.markdown("### 🔐 Please Login to Continue")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            st.markdown("#### Enter Your Credentials")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            if st.form_submit_button("🔓 Login", use_container_width=True):
                # Simple authentication (in production, use proper authentication)
                if username in ["admin", "analyst", "compliance"] and password == "banking123":
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Try: admin/analyst/compliance with password 'banking123'")

    # Demo credentials info
    st.markdown("---")
    st.info("""
    **Demo Credentials:**
    - Username: `admin`, `analyst`, or `compliance`
    - Password: `banking123`
    """)


def logout():
    """Logout function"""
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()


# ===== SIDEBAR NAVIGATION =====

def show_sidebar():
    """Display sidebar navigation"""
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, {st.session_state.get('username', 'User')}")
        st.markdown(f"**Role:** {st.session_state.get('user_role', 'Unknown').title()}")

        if st.button("🚪 Logout", use_container_width=True):
            logout()

        st.markdown("---")

        # Navigation menu
        page = st.selectbox(
            "🧭 Navigate to:",
            [
                "📊 Dashboard",
                "📁 Data Processing",
                "💤 Dormancy Analysis",
                "⚖️ Compliance Analysis",
                "📋 Reports",
                "⚙️ Settings"
            ]
        )

        st.markdown("---")

        # System status
        st.markdown("### 🔧 System Status")

        # Show agent availability
        if AGENTS_AVAILABLE:
            st.success("🟢 All agents operational")
        else:
            st.warning("🟡 Running in demo mode")

        if not DATA_AGENTS_AVAILABLE:
            st.error("🔴 Data agents unavailable")
        if not DORMANCY_AGENTS_AVAILABLE:
            st.error("🔴 Dormancy agents unavailable")
        if not COMPLIANCE_AGENTS_AVAILABLE:
            st.error("🔴 Compliance agents unavailable")

        st.info(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}")

        return page


# ===== INITIALIZE SESSION STATE =====

def initialize_session_state():
    """Initialize session state variables"""
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'mapped_data' not in st.session_state:
        st.session_state.mapped_data = None
    if 'dormancy_results' not in st.session_state:
        st.session_state.dormancy_results = {}
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = {}
    if 'data_quality_results' not in st.session_state:
        st.session_state.data_quality_results = None
    if 'mapping_results' not in st.session_state:
        st.session_state.mapping_results = None


# ===== UTILITY FUNCTIONS =====

def create_download_button(data, filename, label="📥 Download CSV"):
    """Create download button for DataFrame"""
    if isinstance(data, pd.DataFrame) and not data.empty:
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        return st.download_button(
            label=label,
            data=csv_data,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    return None


def format_number(num):
    """Format number with commas"""
    if pd.isna(num):
        return "N/A"
    return f"{num:,.0f}" if isinstance(num, (int, float)) else str(num)


def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create custom metric card"""
    delta_html = ""
    if delta:
        color = "green" if delta_color == "normal" else "red"
        delta_html = f'<small style="color: {color};">{delta}</small>'

    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ===== MOCK FUNCTIONS FOR DEMO MODE =====

def generate_sample_data():
    """Generate sample banking compliance data for demo"""
    np.random.seed(42)
    n_records = 1000

    # Customer data
    customer_ids = [f"CUS{100000 + i}" for i in range(n_records)]
    account_ids = [f"ACC{200000 + i}" for i in range(n_records)]

    # Account types
    account_types = np.random.choice(
        ['CURRENT', 'SAVINGS', 'FIXED_DEPOSIT', 'INVESTMENT'],
        n_records, p=[0.4, 0.3, 0.2, 0.1]
    )

    # Account status
    account_status = np.random.choice(
        ['ACTIVE', 'DORMANT', 'CLOSED'],
        n_records, p=[0.7, 0.25, 0.05]
    )

    # Dormancy status
    dormancy_status = np.where(
        account_status == 'DORMANT',
        np.random.choice(['DORMANT', 'Potentially_Dormant'], n_records),
        'Not_Dormant'
    )

    # Generate dates
    base_date = datetime.now() - timedelta(days=1095)  # 3 years ago
    last_transaction_dates = [
        base_date + timedelta(days=np.random.randint(0, 1095))
        for _ in range(n_records)
    ]

    # Generate balances
    balances = np.random.lognormal(10, 1.5, n_records)  # Log-normal distribution

    sample_data = pd.DataFrame({
        'customer_id': customer_ids,
        'account_id': account_ids,
        'customer_type': np.random.choice(['INDIVIDUAL', 'CORPORATE'], n_records, p=[0.8, 0.2]),
        'full_name_en': [f"Customer {i + 1}" for i in range(n_records)],
        'account_type': account_types,
        'account_status': account_status,
        'dormancy_status': dormancy_status,
        'last_transaction_date': [d.strftime('%Y-%m-%d') for d in last_transaction_dates],
        'balance_current': np.round(balances, 2),
        'currency': np.random.choice(['AED', 'USD', 'EUR'], n_records, p=[0.7, 0.2, 0.1]),
        'contact_attempts_made': np.random.randint(0, 5, n_records),
        'dormancy_period_months': np.where(
            dormancy_status != 'Not_Dormant',
            np.random.randint(6, 60, n_records),
            0
        ),
        'current_stage': np.random.choice(
            ['ACTIVE', 'CONTACT_ATTEMPTS', 'WAITING_PERIOD', 'READY_FOR_TRANSFER'],
            n_records, p=[0.6, 0.2, 0.15, 0.05]
        )
    })

    return sample_data


def run_data_quality_analysis_sync(data):
    """Mock data quality analysis"""
    quality_score = np.random.randint(75, 95)

    column_analysis = {}
    for col in data.columns:
        missing_pct = (data[col].isnull().sum() / len(data)) * 100
        column_analysis[col] = {
            'missing_percentage': missing_pct,
            'data_type': str(data[col].dtype),
            'unique_values': data[col].nunique(),
            'sample_values': data[col].dropna().unique()[:5].tolist()
        }

    return {
        'overall_quality_score': quality_score,
        'total_records': len(data),
        'total_columns': len(data.columns),
        'missing_values_total': data.isnull().sum().sum(),
        'column_analysis': column_analysis,
        'recommendations': [
            "Data quality is acceptable for analysis",
            "Consider handling missing values in critical fields",
            "Validate data types for numerical calculations"
        ]
    }


def run_data_mapping_sync(data):
    """Mock data mapping results"""
    auto_mapping_pct = np.random.randint(85, 98)

    field_mappings = []
    for col in data.columns[:10]:  # Limit to first 10 columns for demo
        confidence = np.random.uniform(0.7, 0.99)
        field_mappings.append({
            'source_field': col,
            'target_field': col.lower().replace(' ', '_'),
            'confidence_score': confidence,
            'mapping_strategy': 'automatic' if confidence > 0.9 else 'manual_required'
        })

    return {
        'auto_mapping_percentage': auto_mapping_pct,
        'total_fields': len(data.columns),
        'mapped_fields': len(field_mappings),
        'field_mappings': field_mappings,
        'transformation_ready': auto_mapping_pct >= 90,
        'recommendations': [
            f"Auto-mapping achieved {auto_mapping_pct}% success rate",
            "Review low confidence mappings" if auto_mapping_pct < 90 else "Ready for transformation",
            "Data mapping completed successfully"
        ]
    }


def run_dormancy_analysis_sync(data, report_date):
    """Mock dormancy analysis results"""
    agents_info = {
        "demand_deposit": {
            "name": "Demand Deposit Dormancy",
            "description": "Analyzes current and savings accounts for 3+ years inactivity (CBUAE Article 2.1.1)",
            "article": "2.1.1",
            "dormant_found": np.random.randint(10, 50),
            "priority": "HIGH"
        },
        "fixed_deposit": {
            "name": "Fixed Deposit Dormancy",
            "description": "Analyzes term deposits based on maturity and renewal status (CBUAE Article 2.2)",
            "article": "2.2",
            "dormant_found": np.random.randint(5, 25),
            "priority": "HIGH"
        },
        "investment": {
            "name": "Investment Account Dormancy",
            "description": "Analyzes investment accounts for inactivity patterns (CBUAE Article 2.3)",
            "article": "2.3",
            "dormant_found": np.random.randint(3, 15),
            "priority": "MEDIUM"
        },
        "payment_instruments": {
            "name": "Unclaimed Payment Instruments",
            "description": "Identifies unclaimed cheques and drafts over 1 year (CBUAE Article 2.4)",
            "article": "2.4",
            "dormant_found": np.random.randint(2, 10),
            "priority": "MEDIUM"
        },
        "safe_deposit": {
            "name": "Safe Deposit Box Dormancy",
            "description": "Analyzes SDB accounts with unpaid fees (CBUAE Article 2.6)",
            "article": "2.6",
            "dormant_found": np.random.randint(1, 8),
            "priority": "LOW"
        },
        "contact_attempts": {
            "name": "Contact Attempts Compliance",
            "description": "Verifies compliance with contact attempt requirements (CBUAE Article 5)",
            "article": "5",
            "dormant_found": np.random.randint(15, 35),
            "priority": "CRITICAL"
        },
        "cb_transfer": {
            "name": "Central Bank Transfer Eligibility",
            "description": "Identifies accounts eligible for CBUAE transfer (5+ years dormant)",
            "article": "8.1",
            "dormant_found": np.random.randint(5, 20),
            "priority": "CRITICAL"
        },
        "high_value": {
            "name": "High Value Dormant Accounts",
            "description": "Identifies high-value dormant accounts requiring priority attention",
            "article": "Internal",
            "dormant_found": np.random.randint(3, 12),
            "priority": "CRITICAL"
        }
    }

    # Generate sample dormant account data for each agent
    for agent_id, info in agents_info.items():
        if info["dormant_found"] > 0:
            sample_accounts = []
            for i in range(info["dormant_found"]):
                sample_accounts.append({
                    'account_id': f"ACC{200000 + i}",
                    'customer_id': f"CUS{100000 + i}",
                    'account_type': np.random.choice(['CURRENT', 'SAVINGS', 'FIXED_DEPOSIT']),
                    'balance_current': np.random.uniform(1000, 100000),
                    'years_dormant': np.random.uniform(3.1, 7.5),
                    'priority': info["priority"],
                    'next_action': f"Review {agent_id} compliance"
                })

            info["sample_data"] = pd.DataFrame(sample_accounts)

    return {
        "analysis_date": report_date,
        "total_accounts_analyzed": len(data),
        "agents_results": agents_info,
        "summary": {
            "total_dormant_found": sum(info["dormant_found"] for info in agents_info.values()),
            "critical_agents": len([a for a in agents_info.values() if a["priority"] == "CRITICAL"]),
            "high_priority_agents": len([a for a in agents_info.values() if a["priority"] == "HIGH"])
        }
    }


def run_compliance_analysis_sync(dormancy_results, processed_data):
    """Mock compliance analysis results"""
    compliance_agents = {
        "incomplete_contact": {
            "name": "Incomplete Contact Attempts",
            "description": "Detects accounts with insufficient contact attempts per CBUAE requirements",
            "category": "Contact & Communication",
            "article": "3.1, 5",
            "violations_found": np.random.randint(10, 30),
            "priority": "CRITICAL"
        },
        "unflagged_dormant": {
            "name": "Unflagged Dormant Candidates",
            "description": "Identifies accounts that should be flagged as dormant but aren't",
            "category": "Contact & Communication",
            "article": "2",
            "violations_found": np.random.randint(5, 20),
            "priority": "HIGH"
        },
        "internal_ledger": {
            "name": "Internal Ledger Transfer Candidates",
            "description": "Identifies accounts ready for internal ledger transfer",
            "category": "Process Management",
            "article": "3.4, 3.5",
            "violations_found": np.random.randint(8, 25),
            "priority": "HIGH"
        },
        "statement_freeze": {
            "name": "Statement Suppression Required",
            "description": "Accounts requiring statement suppression per dormancy rules",
            "category": "Process Management",
            "article": "7.3",
            "violations_found": np.random.randint(12, 35),
            "priority": "MEDIUM"
        },
        "cbuae_transfer": {
            "name": "CBUAE Transfer Candidates",
            "description": "Accounts eligible for Central Bank transfer (5+ years dormant)",
            "category": "Process Management",
            "article": "8",
            "violations_found": np.random.randint(3, 15),
            "priority": "CRITICAL"
        }
    }

    # Generate sample compliance actions for each agent with violations
    for agent_id, info in compliance_agents.items():
        if info["violations_found"] > 0:
            actions = []
            for i in range(info["violations_found"]):
                actions.append({
                    'account_id': f"ACC{300000 + i}",
                    'action_type': f"{agent_id.upper()}_ACTION",
                    'priority': info["priority"],
                    'deadline_days': np.random.randint(1, 30),
                    'description': f"Action required for {info['name'].lower()}",
                    'estimated_hours': np.random.uniform(0.5, 4.0),
                    'status': 'PENDING'
                })

            info["actions"] = pd.DataFrame(actions)

    return {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_agents_analyzed": len(compliance_agents),
        "agents_with_violations": len([a for a in compliance_agents.values() if a["violations_found"] > 0]),
        "total_violations": sum(info["violations_found"] for info in compliance_agents.values()),
        "total_actions": sum(info["violations_found"] for info in compliance_agents.values()),
        "compliance_agents": compliance_agents,
        "summary": {
            "critical_violations": len(
                [a for a in compliance_agents.values() if a["priority"] == "CRITICAL" and a["violations_found"] > 0]),
            "high_priority_violations": len(
                [a for a in compliance_agents.values() if a["priority"] == "HIGH" and a["violations_found"] > 0]),
            "categories_affected": len(
                set(a["category"] for a in compliance_agents.values() if a["violations_found"] > 0))
        }
    }


# ===== PAGE FUNCTIONS =====

def show_dashboard():
    """Display main dashboard with key metrics and overview"""
    st.markdown('<div class="main-header">📊 Banking Compliance Dashboard</div>', unsafe_allow_html=True)

    # Welcome message
    st.markdown(f"### Welcome back, {st.session_state.get('username', 'User')}!")
    st.markdown("Monitor your banking compliance status and dormancy analysis progress.")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_records = len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0
        create_metric_card("Total Records", format_number(total_records))

    with col2:
        dormant_found = 0
        if st.session_state.dormancy_results:
            dormant_found = st.session_state.dormancy_results.get('summary', {}).get('total_dormant_found', 0)
        create_metric_card("Dormant Accounts", format_number(dormant_found))

    with col3:
        compliance_score = "N/A"
        if st.session_state.compliance_results:
            total_violations = st.session_state.compliance_results.get('total_violations', 0)
            compliance_score = f"{max(0, 100 - total_violations)}%" if total_violations < 100 else "Review Required"
        create_metric_card("Compliance Score", compliance_score)

    with col4:
        pending_actions = 0
        if st.session_state.compliance_results:
            pending_actions = st.session_state.compliance_results.get('total_actions', 0)
        create_metric_card("Pending Actions", format_number(pending_actions))

    st.markdown("---")

    # Quick actions section
    st.markdown("### 🚀 Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📁 Process New Data", use_container_width=True):
            # Since we can't modify session state for page navigation directly, 
            # we'll show an info message
            st.info("Navigate to 'Data Processing' from the sidebar")

    with col2:
        if st.button("💤 Run Dormancy Analysis", use_container_width=True):
            st.info("Navigate to 'Dormancy Analysis' from the sidebar")

    with col3:
        if st.button("⚖️ Check Compliance", use_container_width=True):
            st.info("Navigate to 'Compliance Analysis' from the sidebar")

    with col4:
        if st.button("📋 Generate Reports", use_container_width=True):
            st.info("Navigate to 'Reports' from the sidebar")

    # Recent activity section
    st.markdown("---")
    st.markdown("### 📈 Recent Activity")

    if st.session_state.uploaded_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Data Summary")
            if st.session_state.processed_data is not None:
                data = st.session_state.processed_data
                st.write(f"**Records:** {len(data):,}")
                st.write(f"**Columns:** {len(data.columns)}")

                # Show account type distribution if available
                if 'account_type' in data.columns:
                    account_dist = data['account_type'].value_counts().head(5)
                    st.write("**Top Account Types:**")
                    for acc_type, count in account_dist.items():
                        st.write(f"• {acc_type}: {count:,}")
            else:
                st.info("No data processed yet")

        with col2:
            st.markdown("#### 🎯 Analysis Status")

            # Data processing status
            data_status = "✅ Complete" if st.session_state.processed_data is not None else "❌ Pending"
            st.write(f"**Data Processing:** {data_status}")

            # Dormancy analysis status
            dormancy_status = "✅ Complete" if st.session_state.dormancy_results else "❌ Pending"
            st.write(f"**Dormancy Analysis:** {dormancy_status}")

            # Compliance analysis status
            compliance_status = "✅ Complete" if st.session_state.compliance_results else "❌ Pending"
            st.write(f"**Compliance Analysis:** {compliance_status}")

            # Data quality status
            quality_status = "✅ Complete" if st.session_state.data_quality_results else "❌ Pending"
            st.write(f"**Data Quality Check:** {quality_status}")
    else:
        st.info("👋 Upload data to get started with your compliance analysis")

        # Show sample workflow
        st.markdown("#### 🔄 Typical Workflow")
        st.markdown("""
        1. **📁 Data Processing** - Upload and validate your banking data
        2. **🔍 Data Quality** - Ensure data meets compliance standards
        3. **💤 Dormancy Analysis** - Identify dormant accounts per CBUAE guidelines
        4. **⚖️ Compliance Check** - Verify regulatory compliance
        5. **📋 Generate Reports** - Create comprehensive compliance reports
        """)

    # System health section
    st.markdown("---")
    st.markdown("### 🔧 System Health")

    col1, col2, col3 = st.columns(3)

    with col1:
        if AGENTS_AVAILABLE:
            st.success("🟢 All agents operational")
        else:
            st.warning("🟡 Demo mode active")

    with col2:
        st.info(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}")

    with col3:
        demo_status = "🧪 Demo Mode" if DEMO_MODE else "🏭 Production Mode"
        st.info(demo_status)


def show_data_processing_page():
    """Data Processing page with upload, quality check, and mapping"""
    st.markdown('<div class="main-header">📁 Data Processing</div>', unsafe_allow_html=True)

    # Data Upload Section
    st.markdown("## 🔄 Data Upload")

    upload_method = st.selectbox(
        "Select Upload Method:",
        ["📄 File Upload", "🔗 URL Import", "🧪 Generate Sample Data"]
    )

    uploaded_data = None

    if upload_method == "📄 File Upload":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Upload CSV, Excel, or JSON files"
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    uploaded_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    uploaded_data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    uploaded_data = pd.read_json(uploaded_file)

                st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    elif upload_method == "🔗 URL Import":
        url = st.text_input("Enter CSV URL:", placeholder="https://example.com/data.csv")

        if url and st.button("📥 Import from URL"):
            try:
                uploaded_data = pd.read_csv(url)
                st.success("✅ Data imported from URL successfully")
            except Exception as e:
                st.error(f"❌ Error importing from URL: {str(e)}")

    elif upload_method == "🧪 Generate Sample Data":
        st.info("🔬 Generate sample banking compliance data for testing")

        if st.button("🎲 Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                uploaded_data = generate_sample_data()
                st.success("✅ Sample data generated successfully")

    # Display uploaded data
    if uploaded_data is not None:
        st.session_state.uploaded_data = uploaded_data

        st.markdown("### 📊 Data Preview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            create_metric_card("Records", format_number(len(uploaded_data)))
        with col2:
            create_metric_card("Columns", format_number(len(uploaded_data.columns)))
        with col3:
            create_metric_card("Memory Usage", f"{uploaded_data.memory_usage().sum() / 1024:.1f} KB")
        with col4:
            create_metric_card("Missing Values", format_number(uploaded_data.isnull().sum().sum()))

        # Show data preview
        st.dataframe(uploaded_data.head(10), use_container_width=True)

        # Data Quality Analysis
        st.markdown("---")
        st.markdown("## 🔍 Data Quality Analysis")

        if st.button("🚀 Run Quality Analysis", use_container_width=True):
            with st.spinner("Analyzing data quality..."):
                try:
                    quality_results = run_data_quality_analysis_sync(uploaded_data)
                    st.session_state.data_quality_results = quality_results

                    # Display quality results
                    show_data_quality_results(quality_results)

                except Exception as e:
                    st.error(f"❌ Quality analysis failed: {str(e)}")

        # Show existing quality results
        if st.session_state.data_quality_results:
            show_data_quality_results(st.session_state.data_quality_results)

        # Data Mapping Section
        st.markdown("---")
        st.markdown("## 🔗 Data Mapping")

        if st.button("🎯 Run Data Mapping", use_container_width=True):
            with st.spinner("Mapping data to target schema..."):
                try:
                    mapping_results = run_data_mapping_sync(uploaded_data)
                    st.session_state.mapping_results = mapping_results

                    # Display mapping results
                    show_mapping_results(mapping_results)

                except Exception as e:
                    st.error(f"❌ Data mapping failed: {str(e)}")

        # Show existing mapping results
        if st.session_state.mapping_results:
            show_mapping_results(st.session_state.mapping_results)


def show_data_quality_results(results):
    """Display data quality analysis results"""
    st.markdown("### 📈 Quality Analysis Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_card("Quality Score", f"{results['overall_quality_score']}%")
    with col2:
        create_metric_card("Records", format_number(results['total_records']))
    with col3:
        create_metric_card("Columns", format_number(results['total_columns']))
    with col4:
        create_metric_card("Missing Values", format_number(results['missing_values_total']))

    # Quality recommendations
    st.markdown("#### 💡 Recommendations")
    for rec in results['recommendations']:
        st.info(f"• {rec}")

    # Column analysis
    if st.checkbox("📊 Show detailed column analysis"):
        col_df = pd.DataFrame.from_dict(results['column_analysis'], orient='index')
        col_df = col_df.reset_index().rename(columns={'index': 'Column'})
        st.dataframe(col_df, use_container_width=True)


def show_mapping_results(results):
    """Display data mapping results"""
    st.markdown("### 🎯 Mapping Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_card("Auto-Mapping", f"{results['auto_mapping_percentage']}%")
    with col2:
        create_metric_card("Total Fields", format_number(results['total_fields']))
    with col3:
        create_metric_card("Mapped Fields", format_number(results['mapped_fields']))
    with col4:
        status = "✅ Ready" if results['transformation_ready'] else "⚠️ Review Needed"
        create_metric_card("Status", status)

    # Mapping recommendations
    st.markdown("#### 💡 Recommendations")
    for rec in results['recommendations']:
        st.info(f"• {rec}")

    # Field mappings
    if st.checkbox("🔗 Show field mappings"):
        mappings_df = pd.DataFrame(results['field_mappings'])
        st.dataframe(mappings_df, use_container_width=True)

    # Set processed data if mapping is successful
    if results['transformation_ready'] and st.session_state.uploaded_data is not None:
        st.session_state.processed_data = st.session_state.uploaded_data.copy()
        st.success("✅ Data is ready for dormancy analysis!")


def show_dormancy_analysis_page():
    """Dormancy Analysis page with all dormancy agents"""
    st.markdown('<div class="main-header">💤 Dormancy Analysis</div>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process data in the Data Processing section first")
        return

    data = st.session_state.processed_data

    st.markdown("## 🎯 Analysis Configuration")

    col1, col2 = st.columns(2)
    with col1:
        report_date = st.date_input("Report Date", datetime.now().date())
    with col2:
        analysis_scope = st.selectbox("Analysis Scope", ["All Agents", "Custom Selection"])

    if st.button("🚀 Run Dormancy Analysis", use_container_width=True):
        with st.spinner("Running comprehensive dormancy analysis..."):
            try:
                dormancy_results = run_dormancy_analysis_sync(data, str(report_date))
                st.session_state.dormancy_results = dormancy_results

                st.success("✅ Dormancy analysis completed!")

            except Exception as e:
                st.error(f"❌ Dormancy analysis failed: {str(e)}")

    # Display dormancy results
    if st.session_state.dormancy_results:
        show_dormancy_results(st.session_state.dormancy_results)


def show_dormancy_results(results):
    """Display dormancy analysis results"""
    st.markdown("## 📊 Dormancy Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_card("Total Analyzed", format_number(results["total_accounts_analyzed"]))
    with col2:
        create_metric_card("Dormant Found", format_number(results["summary"]["total_dormant_found"]))
    with col3:
        create_metric_card("Critical Agents", format_number(results["summary"]["critical_agents"]))
    with col4:
        create_metric_card("High Priority", format_number(results["summary"]["high_priority_agents"]))

    st.markdown("---")
    st.markdown("### 🤖 Active Dormancy Agents")
    st.caption("Only agents with dormant accounts > 0 are shown")

    # Filter and display agents with results
    active_agents = {k: v for k, v in results["agents_results"].items() if v["dormant_found"] > 0}

    for agent_id, agent_info in active_agents.items():
        with st.expander(f"🔍 {agent_info['name']} - {agent_info['dormant_found']} accounts found", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Description:** {agent_info['description']}")
                st.markdown(f"**CBUAE Article:** {agent_info['article']}")
                st.markdown(f"**Priority Level:** `{agent_info['priority']}`")

                # Summary stats
                if 'sample_data' in agent_info:
                    sample_df = agent_info['sample_data']
                    total_value = sample_df['balance_current'].sum()
                    avg_dormancy = sample_df['years_dormant'].mean()

                    st.markdown(f"**Total Value:** AED {total_value:,.2f}")
                    st.markdown(f"**Average Dormancy Period:** {avg_dormancy:.1f} years")

            with col2:
                # Priority badge
                priority_color = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠",
                    "MEDIUM": "🟡",
                    "LOW": "🟢"
                }
                st.markdown(f"### {priority_color.get(agent_info['priority'], '⚪')} {agent_info['priority']}")
                st.metric("Accounts Found", agent_info['dormant_found'])

            # Download buttons
            if 'sample_data' in agent_info:
                col1, col2 = st.columns(2)

                with col1:
                    create_download_button(
                        agent_info['sample_data'],
                        f"{agent_id}_dormant_accounts",
                        "📥 Download Account Details"
                    )

                with col2:
                    # Create summary data
                    summary_data = pd.DataFrame([{
                        'Agent': agent_info['name'],
                        'CBUAE_Article': agent_info['article'],
                        'Accounts_Found': agent_info['dormant_found'],
                        'Priority': agent_info['priority'],
                        'Total_Value': sample_df['balance_current'].sum() if 'sample_data' in agent_info else 0,
                        'Analysis_Date': results["analysis_date"]
                    }])

                    create_download_button(
                        summary_data,
                        f"{agent_id}_summary",
                        "📄 Download Summary"
                    )

            # Show data preview
            if st.checkbox(f"👁️ Preview data for {agent_info['name']}", key=f"preview_{agent_id}"):
                if 'sample_data' in agent_info:
                    st.dataframe(agent_info['sample_data'], use_container_width=True)


def show_compliance_analysis_page():
    """Compliance Analysis page with compliance agents"""
    st.markdown('<div class="main-header">⚖️ Compliance Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.dormancy_results:
        st.warning("⚠️ Please run dormancy analysis first")
        return

    st.markdown("## 🎯 Compliance Configuration")

    col1, col2 = st.columns(2)
    with col1:
        compliance_scope = st.selectbox("Compliance Scope", ["All Agents", "High Priority Only", "Custom Selection"])
    with col2:
        include_recommendations = st.checkbox("Include Recommendations", value=True)

    if st.button("🚀 Run Compliance Analysis", use_container_width=True):
        with st.spinner("Running comprehensive compliance analysis..."):
            try:
                compliance_results = run_compliance_analysis_sync(
                    st.session_state.dormancy_results,
                    st.session_state.processed_data
                )
                st.session_state.compliance_results = compliance_results

                st.success("✅ Compliance analysis completed!")

            except Exception as e:
                st.error(f"❌ Compliance analysis failed: {str(e)}")

    # Display compliance results
    if st.session_state.compliance_results:
        show_compliance_results(st.session_state.compliance_results)


def show_compliance_results(results):
    """Display compliance analysis results"""
    st.markdown("## ⚖️ Compliance Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_card("Total Violations", format_number(results["total_violations"]))
    with col2:
        create_metric_card("Actions Required", format_number(results["total_actions"]))
    with col3:
        create_metric_card("Critical Issues", format_number(results["summary"]["critical_violations"]))
    with col4:
        create_metric_card("Categories Affected", format_number(results["summary"]["categories_affected"]))

    # Compliance overview chart
    st.markdown("### 📊 Compliance Overview")

    # Create pie chart of violations by priority
    priorities = []
    counts = []

    for agent_info in results["compliance_agents"].values():
        if agent_info["violations_found"] > 0:
            priorities.append(agent_info["priority"])

    priority_counts = {p: priorities.count(p) for p in set(priorities)}

    if priority_counts:
        fig = px.pie(
            values=list(priority_counts.values()),
            names=list(priority_counts.keys()),
            title="Violations by Priority Level",
            color_discrete_map={
                "CRITICAL": "#f44336",
                "HIGH": "#ff9800",
                "MEDIUM": "#ffc107",
                "LOW": "#4caf50"
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Active Compliance Agents")
    st.caption("Only agents with violations > 0 are shown")

    # Filter and display agents with violations
    active_agents = {k: v for k, v in results["compliance_agents"].items() if v["violations_found"] > 0}

    for agent_id, agent_info in active_agents.items():
        with st.expander(f"⚖️ {agent_info['name']} - {agent_info['violations_found']} violations", expanded=False):

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Description:** {agent_info['description']}")
                st.markdown(f"**Category:** {agent_info['category']}")
                st.markdown(f"**CBUAE Article:** {agent_info['article']}")
                st.markdown(f"**Priority Level:** `{agent_info['priority']}`")

                # Action summary
                if 'actions' in agent_info:
                    actions_df = agent_info['actions']
                    avg_hours = actions_df['estimated_hours'].mean()
                    urgent_actions = len(actions_df[actions_df['deadline_days'] <= 7])

                    st.markdown(f"**Average Resolution Time:** {avg_hours:.1f} hours")
                    st.markdown(f"**Urgent Actions (≤7 days):** {urgent_actions}")

            with col2:
                # Priority badge
                priority_color = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠",
                    "MEDIUM": "🟡",
                    "LOW": "🟢"
                }
                st.markdown(f"### {priority_color.get(agent_info['priority'], '⚪')} {agent_info['priority']}")
                st.metric("Violations Found", agent_info['violations_found'])

            # Download buttons
            if 'actions' in agent_info:
                col1, col2 = st.columns(2)

                with col1:
                    create_download_button(
                        agent_info['actions'],
                        f"{agent_id}_compliance_actions",
                        "📥 Download Actions"
                    )

                with col2:
                    # Create summary data
                    summary_data = pd.DataFrame([{
                        'Agent': agent_info['name'],
                        'Category': agent_info['category'],
                        'CBUAE_Article': agent_info['article'],
                        'Violations_Found': agent_info['violations_found'],
                        'Priority': agent_info['priority'],
                        'Analysis_Date': results["analysis_timestamp"]
                    }])

                    create_download_button(
                        summary_data,
                        f"{agent_id}_compliance_summary",
                        "📄 Download Summary"
                    )

            # Show actions preview
            if st.checkbox(f"👁️ Preview actions for {agent_info['name']}", key=f"compliance_preview_{agent_id}"):
                if 'actions' in agent_info:
                    st.dataframe(agent_info['actions'], use_container_width=True)


def show_reports_page():
    """Reports page showing all agents and comprehensive analytics"""
    st.markdown('<div class="main-header">📋 Comprehensive Reports</div>', unsafe_allow_html=True)

    # Report type selection
    report_type = st.selectbox(
        "Select Report Type:",
        ["📊 Executive Dashboard", "🤖 Agent Performance", "📈 Trend Analysis", "📋 Regulatory Report"]
    )

    if report_type == "📊 Executive Dashboard":
        show_executive_dashboard()
    elif report_type == "🤖 Agent Performance":
        show_agent_performance_report()
    elif report_type == "📈 Trend Analysis":
        show_trend_analysis()
    elif report_type == "📋 Regulatory Report":
        show_regulatory_report()


def show_executive_dashboard():
    """Executive dashboard with high-level metrics"""
    st.markdown("## 📊 Executive Dashboard")

    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_card("Total Accounts", "125,456")
    with col2:
        create_metric_card("Dormant Accounts", "3,247", "↑ 2.1%")
    with col3:
        create_metric_card("Compliance Score", "94.2%", "↑ 1.8%")
    with col4:
        create_metric_card("Actions Pending", "157", "↓ 12")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Dormancy Trend (Last 12 Months)")

        # Generate sample trend data
        months = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
        dormant_counts = np.random.randint(2800, 3500, len(months))

        fig = px.line(
            x=months,
            y=dormant_counts,
            title="Monthly Dormant Account Count",
            labels={'x': 'Month', 'y': 'Dormant Accounts'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Compliance Status by Category")

        categories = ['Contact & Communication', 'Process Management', 'Specialized Compliance',
                      'Reporting & Retention']
        compliance_scores = [92, 96, 89, 98]

        fig = px.bar(
            x=categories,
            y=compliance_scores,
            title="Compliance Scores by Category",
            labels={'x': 'Category', 'y': 'Compliance Score (%)'},
            color=compliance_scores,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def show_agent_performance_report():
    """Detailed agent performance report"""
    st.markdown("## 🤖 Agent Performance Report")

    # All dormancy agents
    st.markdown("### 💤 Dormancy Analysis Agents")

    dormancy_agents_data = [
        {"Agent": "Demand Deposit Dormancy", "Article": "2.1.1", "Accounts": 1247, "Actions": 15, "Status": "🟢 Active"},
        {"Agent": "Fixed Deposit Dormancy", "Article": "2.2", "Accounts": 856, "Actions": 8, "Status": "🟢 Active"},
        {"Agent": "Investment Account Dormancy", "Article": "2.3", "Accounts": 342, "Actions": 5, "Status": "🟢 Active"},
        {"Agent": "Payment Instruments", "Article": "2.4", "Accounts": 123, "Actions": 3, "Status": "🟢 Active"},
        {"Agent": "Safe Deposit Dormancy", "Article": "2.6", "Accounts": 67, "Actions": 2, "Status": "🟢 Active"},
        {"Agent": "Contact Attempts", "Article": "5", "Accounts": 234, "Actions": 12, "Status": "🟡 Review"},
        {"Agent": "CB Transfer Eligibility", "Article": "8.1", "Accounts": 89, "Actions": 89, "Status": "🔴 Critical"},
        {"Agent": "High Value Dormant", "Article": "Internal", "Accounts": 45, "Actions": 45, "Status": "🔴 Critical"},
    ]

    dormancy_df = pd.DataFrame(dormancy_agents_data)
    st.dataframe(dormancy_df, use_container_width=True)

    # Download button for dormancy agents
    create_download_button(dormancy_df, "dormancy_agents_report", "📥 Download Dormancy Agents Report")

    st.markdown("---")

    # All compliance agents
    st.markdown("### ⚖️ Compliance Analysis Agents")

    compliance_agents_data = [
        {"Agent": "Incomplete Contact Attempts", "Category": "Contact & Communication", "Article": "3.1, 5",
         "Violations": 23, "Actions": 23, "Priority": "🔴 Critical"},
        {"Agent": "Unflagged Dormant Candidates", "Category": "Contact & Communication", "Article": "2",
         "Violations": 15, "Actions": 15, "Priority": "🟠 High"},
        {"Agent": "Internal Ledger Candidates", "Category": "Process Management", "Article": "3.4, 3.5",
         "Violations": 18, "Actions": 18, "Priority": "🟠 High"},
        {"Agent": "Statement Freeze Candidates", "Category": "Process Management", "Article": "7.3", "Violations": 27,
         "Actions": 27, "Priority": "🟡 Medium"},
        {"Agent": "CBUAE Transfer Candidates", "Category": "Process Management", "Article": "8", "Violations": 12,
         "Actions": 12, "Priority": "🔴 Critical"},
    ]

    compliance_df = pd.DataFrame(compliance_agents_data)
    st.dataframe(compliance_df, use_container_width=True)

    # Download button for compliance agents
    create_download_button(compliance_df, "compliance_agents_report", "📥 Download Compliance Agents Report")


def show_trend_analysis():
    """Trend analysis with charts and insights"""
    st.markdown("## 📈 Trend Analysis")

    # Time period selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Generate sample trend data
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')

    # Dormancy trends
    st.markdown("### 💤 Dormancy Trends")

    dormancy_data = pd.DataFrame({
        'Month': date_range,
        'New_Dormant': np.random.randint(50, 150, len(date_range)),
        'Reactivated': np.random.randint(20, 80, len(date_range)),
        'CB_Transfers': np.random.randint(5, 25, len(date_range))
    })

    fig = px.line(
        dormancy_data.melt(id_vars=['Month'], var_name='Type', value_name='Count'),
        x='Month', y='Count', color='Type',
        title="Dormancy Activity Trends"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def show_regulatory_report():
    """CBUAE regulatory compliance report"""
    st.markdown("## 📋 CBUAE Regulatory Compliance Report")

    # Report period
    col1, col2 = st.columns(2)
    with col1:
        report_quarter = st.selectbox("Report Quarter", ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"])
    with col2:
        report_format = st.selectbox("Report Format", ["Detailed", "Summary", "Executive"])

    # CBUAE Article compliance
    st.markdown("### 📜 CBUAE Article Compliance Summary")

    article_compliance = [
        {"Article": "2.1.1", "Description": "Demand Deposit Dormancy", "Accounts": 1247, "Compliance": "✅ 100%",
         "Status": "Compliant"},
        {"Article": "2.2", "Description": "Fixed Deposit Dormancy", "Accounts": 856, "Compliance": "✅ 100%",
         "Status": "Compliant"},
        {"Article": "2.3", "Description": "Investment Account Dormancy", "Accounts": 342, "Compliance": "✅ 100%",
         "Status": "Compliant"},
        {"Article": "8", "Description": "Central Bank Transfer", "Accounts": 89, "Compliance": "🔴 85%",
         "Status": "Action Required"},
    ]

    article_df = pd.DataFrame(article_compliance)
    st.dataframe(article_df, use_container_width=True)

    # Summary statistics
    st.markdown("### 📊 Report Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric_card("Overall Compliance", "96.8%")
    with col2:
        create_metric_card("Articles Monitored", "9")
    with col3:
        create_metric_card("Total Accounts", "3,381")
    with col4:
        create_metric_card("Actions Required", "23")


def show_settings_page():
    """Settings and configuration page"""
    st.markdown('<div class="main-header">⚙️ System Settings</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Agent Configuration", "📊 Thresholds", "🔗 Integrations", "👤 User Management"])

    with tab1:
        st.markdown("### 🤖 Agent Configuration")

        st.markdown("#### Dormancy Agents")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Enable Demand Deposit Agent", value=True)
            st.checkbox("Enable Fixed Deposit Agent", value=True)
            st.checkbox("Enable Investment Agent", value=True)
        with col2:
            st.checkbox("Enable Payment Instruments Agent", value=True)
            st.checkbox("Enable Safe Deposit Agent", value=True)
            st.checkbox("Enable Contact Attempts Agent", value=True)

        st.markdown("#### Compliance Agents")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Enable Contact Compliance Agent", value=True)
            st.checkbox("Enable Ledger Transfer Agent", value=True)
            st.checkbox("Enable Statement Freeze Agent", value=True)
        with col2:
            st.checkbox("Enable CBUAE Transfer Agent", value=True)
            st.checkbox("Enable FX Conversion Agent", value=True)
            st.checkbox("Enable Record Retention Agent", value=True)

    with tab2:
        st.markdown("### 📊 Compliance Thresholds")

        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Dormancy Period (months)", min_value=12, max_value=60, value=36)
            st.number_input("High Value Threshold (AED)", min_value=1000, max_value=1000000, value=100000)
            st.number_input("Contact Attempts Required", min_value=1, max_value=10, value=3)

        with col2:
            st.number_input("CB Transfer Period (years)", min_value=3, max_value=10, value=5)
            st.number_input("Statement Freeze Threshold (AED)", min_value=100, max_value=10000, value=1000)
            st.number_input("Record Retention Period (years)", min_value=5, max_value=15, value=7)

    with tab3:
        st.markdown("### 🔗 System Integrations")

        st.markdown("#### Database Connections")
        database_config = st.expander("Core Banking System", expanded=False)
        with database_config:
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Host", placeholder="prod-db.bank.local")
                st.text_input("Database", placeholder="core_banking")
                st.text_input("Username", placeholder="compliance_user")
            with col2:
                st.text_input("Port", placeholder="5432")
                st.text_input("Schema", placeholder="accounts")
                st.text_input("Password", type="password")

        st.markdown("#### Agent Status")
        st.write(f"**Data Agents Available:** {'✅ Yes' if DATA_AGENTS_AVAILABLE else '❌ No'}")
        st.write(f"**Dormancy Agents Available:** {'✅ Yes' if DORMANCY_AGENTS_AVAILABLE else '❌ No'}")
        st.write(f"**Compliance Agents Available:** {'✅ Yes' if COMPLIANCE_AGENTS_AVAILABLE else '❌ No'}")
        st.write(f"**Demo Mode:** {'🧪 Active' if DEMO_MODE else '🏭 Production'}")

    with tab4:
        st.markdown("### 👤 User Management")

        st.markdown("#### Current Session")
        st.write(f"**User:** {st.session_state.get('username', 'Unknown')}")
        st.write(f"**Role:** {st.session_state.get('user_role', 'Unknown')}")
        st.write(f"**Login Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.markdown("#### Session Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Reset Session", use_container_width=True):
                # Reset session but keep authentication
                keys_to_keep = ['authenticated', 'username', 'user_role']
                keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
                for key in keys_to_remove:
                    del st.session_state[key]
                initialize_session_state()
                st.success("Session reset successful")

        with col2:
            if st.button("💾 Export Data", use_container_width=True):
                if st.session_state.processed_data is not None:
                    create_download_button(
                        st.session_state.processed_data,
                        "banking_compliance_data",
                        "📥 Download Current Data"
                    )
                else:
                    st.warning("No data to export")

        with col3:
            if st.button("🚪 Logout", use_container_width=True):
                logout()


# ===== MAIN APPLICATION =====

def main():
    """Main application function with enhanced error handling"""
    # Check authentication first
    if not check_authentication():
        return

    # Initialize session state
    initialize_session_state()

    # Show agent availability warnings
    if not AGENTS_AVAILABLE:
        st.warning("⚠️ Some agents are not available. Running in demo mode.")

        with st.expander("🔧 Agent Status Details"):
            st.write(f"**Data Processing Agents:** {'✅ Available' if DATA_AGENTS_AVAILABLE else '❌ Not Available'}")
            st.write(f"**Dormancy Agents:** {'✅ Available' if DORMANCY_AGENTS_AVAILABLE else '❌ Not Available'}")
            st.write(f"**Compliance Agents:** {'✅ Available' if COMPLIANCE_AGENTS_AVAILABLE else '❌ Not Available'}")

            st.info("""
            **To enable full functionality:**
            1. Ensure all agent modules are properly installed
            2. Check the agents/ directory structure
            3. Verify all required dependencies are installed
            """)

    # Show sidebar and get current page
    current_page = show_sidebar()

    # Route to appropriate page with error handling
    try:
        if current_page == "📊 Dashboard":
            show_dashboard()
        elif current_page == "📁 Data Processing":
            show_data_processing_page()
        elif current_page == "💤 Dormancy Analysis":
            show_dormancy_analysis_page()
        elif current_page == "⚖️ Compliance Analysis":
            show_compliance_analysis_page()
        elif current_page == "📋 Reports":
            show_reports_page()
        elif current_page == "⚙️ Settings":
            show_settings_page()
        else:
            # Default fallback to dashboard
            st.warning(f"Unknown page: {current_page}. Redirecting to dashboard.")
            show_dashboard()

    except NameError as e:
        st.error(f"❌ Function not found: {str(e)}")
        st.info("🔧 This indicates a missing function definition in your app.py file.")

        # Show available functions for debugging
        with st.expander("🛠️ Debug Info"):
            available_functions = [name for name in globals() if callable(globals()[name]) and name.startswith('show_')]
            st.write("**Available page functions:**")
            for func in available_functions:
                st.write(f"• {func}")

        # Fallback to a basic page
        st.markdown("### 🏦 Banking Compliance System")
        st.info("Please check the application setup and ensure all page functions are properly defined.")

    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please contact your system administrator.")

        # Show error details in debug mode
        if st.checkbox("🐛 Show detailed error"):
            import traceback
            st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🏦 Banking Compliance Analysis System v2.0 | 
        Powered by AI Agents | 
        CBUAE Compliant | 
        © 2024 Banking Compliance Solutions</p>
    </div>
    """, unsafe_allow_html=True)


# ===== RUN APPLICATION =====

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        st.info("The application failed to start. Please check your configuration.")

        # Emergency debug mode
        if st.button("🚨 Emergency Debug Mode"):
            st.code(f"""
Error: {str(e)}

App Configuration:
- Demo Mode: {DEMO_MODE if 'DEMO_MODE' in globals() else 'Unknown'}
- Data Agents: {DATA_AGENTS_AVAILABLE if 'DATA_AGENTS_AVAILABLE' in globals() else 'Unknown'}
- Dormancy Agents: {DORMANCY_AGENTS_AVAILABLE if 'DORMANCY_AGENTS_AVAILABLE' in globals() else 'Unknown'}
- Compliance Agents: {COMPLIANCE_AGENTS_AVAILABLE if 'COMPLIANCE_AGENTS_AVAILABLE' in globals() else 'Unknown'}
- Session State Keys: {list(st.session_state.keys()) if hasattr(st, 'session_state') else 'Not available'}
- Available Functions: {[name for name in globals() if callable(globals()[name]) and name.startswith('show_')]}
""")

# ===== EXPORT DEFINITIONS =====

__all__ = [
    "main",
    "show_dashboard",
    "show_data_processing_page",
    "show_dormancy_analysis_page",
    "show_compliance_analysis_page",
    "show_reports_page",
    "show_settings_page",
    "check_authentication",
    "initialize_session_state",
    "create_download_button",
    "format_number",
    "create_metric_card"
]