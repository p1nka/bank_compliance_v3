"""
app.py - Banking Compliance Agentic AI System Streamlit Application
Comprehensive Streamlit web application integrating all agents and features
Enhanced with LangGraph workflows, hybrid memory, and MCP tools
"""

import asyncio
import json
import logging
import os
import secrets
import tempfile
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx

# Core system imports
try:
    from core.workflow_engine import WorkflowOrchestrationEngine, WorkflowState, WorkflowStatus
    from agents.memory_agent import HybridMemoryAgent, MemoryContext, MemoryBucket, MemoryPriority
    from agents.Data_Process import DataProcessingAgent, DataProcessingState
    from agents.Dormant_agent import DormancyAnalysisAgent, run_comprehensive_dormancy_analysis_csv
    from agents.compliance_verification_agent import (
        ComplianceVerificationAgent,
        run_standalone_compliance_verification,
        generate_compliance_report,
        get_compliance_agent_coverage
    )
    from agents.data_mapping_agent import DataMappingAgent
    from agents.notification_agent import NotificationAgent
    from agents.reporting_agent import ReportingAgent
    from agents.risk_assessment_agent import RiskAssessmentAgent
    from agents.supervisor_agent import SupervisorAgent
    from agents.error_handler_agent import ErrorHandlerAgent
    from agents.audit_trail_agent import AuditTrailAgent
    from mcp_client import MCPClient, create_mcp_client
except ImportError as e:
    st.error(f"Failed to import core modules: {e}")
    st.info("Running in demo mode with mock data")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================= STREAMLIT CONFIGURATION =========================

st.set_page_config(
    page_title="Banking Compliance AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/banking-compliance-ai',
        'Report a bug': 'https://github.com/your-repo/banking-compliance-ai/issues',
        'About': """
        # Banking Compliance Agentic AI System
        
        A comprehensive AI-powered system for banking compliance analysis, 
        dormancy detection, and regulatory reporting.
        
        **Features:**
        - Multi-agent AI workflow orchestration
        - CBUAE compliance verification
        - Automated dormancy analysis
        - Risk assessment and reporting
        - Real-time notifications
        """
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    
    .workflow-step {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        background: #f8f9fa;
    }
    
    .step-completed {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .step-running {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    .step-failed {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ========================= GLOBAL STATE MANAGEMENT =========================

@st.cache_resource
def initialize_system():
    """Initialize the core system components"""

    try:
        logger.info("Initializing Banking Compliance Agentic AI System...")

        # Initialize with mock client for demo
        mcp_client = type('MockMCPClient', (), {
            'is_mock_mode': lambda: True,
            'call_tool': lambda self, tool_name, params=None: asyncio.run(mock_tool_call(tool_name, params)),
            'get_statistics': lambda self: {'connected': True, 'mock_mode': True}
        })()

        # Initialize memory agent (mock mode)
        memory_agent = type('MockMemoryAgent', (), {
            'store_memory': lambda self, *args, **kwargs: {'success': True},
            'retrieve_memory': lambda self, *args, **kwargs: {'success': True, 'data': []},
            'get_memory_statistics': lambda self: {'total_entries': 0}
        })()

        # Initialize workflow engine (mock mode)
        workflow_engine = type('MockWorkflowEngine', (), {
            'execute_workflow': lambda self, user_id, data, options=None: asyncio.run(mock_workflow_execution(user_id, data, options))
        })()

        return {
            'mcp_client': mcp_client,
            'memory_agent': memory_agent,
            'workflow_engine': workflow_engine,
            'initialized': True,
            'mode': 'demo'
        }

    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        return {
            'initialized': False,
            'error': str(e),
            'mode': 'error'
        }

async def mock_tool_call(tool_name: str, params: Optional[Dict] = None):
    """Mock tool call for demo purposes"""
    await asyncio.sleep(0.1)  # Simulate processing

    return {
        'success': True,
        'data': {'tool': tool_name, 'mock': True},
        'timestamp': datetime.now().isoformat()
    }

async def mock_workflow_execution(user_id: str, data: Dict, options: Optional[Dict] = None):
    """Mock workflow execution for demo purposes"""

    # Simulate processing time
    await asyncio.sleep(2)

    # Generate mock results
    total_accounts = len(data.get('accounts', []))
    dormant_accounts = max(1, int(total_accounts * 0.23))  # 23% dormancy rate

    return {
        'success': True,
        'workflow_id': secrets.token_hex(16),
        'status': 'completed',
        'total_processing_time': 2.1,
        'nodes_completed': 8,
        'nodes_failed': 0,
        'errors': [],
        'warnings': [],
        'results': {
            'data_processing': {
                'status': 'completed',
                'quality_score': 0.95,
                'records_processed': total_accounts
            },
            'dormancy_analysis': {
                'total_analyzed': total_accounts,
                'dormant_found': dormant_accounts,
                'high_risk': max(1, int(dormant_accounts * 0.15))
            },
            'compliance': {
                'status': 'compliant',
                'critical_violations': 0
            },
            'risk_assessment': {
                'overall_score': 0.25
            }
        }
    }

# Initialize system
if 'system' not in st.session_state:
    st.session_state.system = initialize_system()

# ========================= AUTHENTICATION =========================

def init_auth():
    """Initialize authentication state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.user_role = None

def login_form():
    """Display login form"""
    st.markdown('<div class="main-header">🏦 Banking Compliance AI System</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 System Login")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            role = st.selectbox("Role", ["Compliance Analyst", "Risk Manager", "Supervisor", "Administrator"])

            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🚀 Login", use_container_width=True)
            with col2:
                demo_button = st.form_submit_button("📊 Demo Mode", use_container_width=True)

            if login_button:
                if username and password:
                    # Mock authentication - in production, this would validate against a database
                    st.session_state.authenticated = True
                    st.session_state.user = username
                    st.session_state.user_role = role
                    st.session_state.auth_mode = 'authenticated'
                    st.rerun()
                else:
                    st.error("Please enter both username and password")

            if demo_button:
                st.session_state.authenticated = True
                st.session_state.user = "Demo User"
                st.session_state.user_role = "Compliance Analyst"
                st.session_state.auth_mode = 'demo'
                st.rerun()

        # System information
        with st.expander("ℹ️ System Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Demo Credentials:**\n- Username: analyst\n- Password: demo123")
            with col2:
                system_status = "🟢 Online" if st.session_state.system.get('initialized') else "🔴 Offline"
                st.info(f"**System Status:** {system_status}\n**Mode:** {st.session_state.system.get('mode', 'unknown')}")

# ========================= MAIN APPLICATION =========================

def main_app():
    """Main application interface"""

    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, {st.session_state.user}")
        st.markdown(f"**Role:** {st.session_state.user_role}")

        # Navigation menu
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "Dashboard",
                "Data Upload",
                "Data Mapping",
                "Workflow Monitor",
                "Compliance Reports",
                "Risk Analysis",
                "Notifications",
                "System Admin"
            ],
            icons=[
                "speedometer2",
                "cloud-upload",
                "diagram-3",
                "gear",
                "file-earmark-check",
                "exclamation-triangle",
                "bell",
                "tools"
            ],
            menu_icon="bank",
            default_index=0
        )

        # System status
        st.markdown("---")
        st.markdown("### 🔧 System Status")

        system = st.session_state.system
        if system.get('initialized'):
            st.success("✅ System Online")
            st.info(f"Mode: {system.get('mode', 'unknown').title()}")
        else:
            st.error("❌ System Offline")
            st.error(f"Error: {system.get('error', 'Unknown')}")

        # Quick stats
        if 'workflow_stats' in st.session_state:
            stats = st.session_state.workflow_stats
            st.metric("Total Workflows", stats.get('total', 0))
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")

        # Logout button
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main content area
    if selected == "Dashboard":
        show_dashboard()
    elif selected == "Data Upload":
        show_data_upload()
    elif selected == "Data Mapping":
        show_data_mapping()
    elif selected == "Workflow Monitor":
        show_workflow_monitor()
    elif selected == "Compliance Reports":
        show_compliance_reports()
    elif selected == "Risk Analysis":
        show_risk_analysis()
    elif selected == "Notifications":
        show_notifications()
    elif selected == "System Admin":
        show_system_admin()

# ========================= PAGE FUNCTIONS =========================

def show_dashboard():
    """Display main dashboard"""

    st.markdown('<div class="main-header">📊 Banking Compliance Dashboard</div>', unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Total Accounts Analyzed",
            value="12,847",
            delta="342 this week"
        )

    with col2:
        st.metric(
            label="⚠️ Dormant Accounts",
            value="2,954",
            delta="-23 from last month",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="🎯 Compliance Score",
            value="94.7%",
            delta="2.1% improvement"
        )

    with col4:
        st.metric(
            label="🔍 Risk Level",
            value="Medium",
            delta="Stable"
        )

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Dormancy Trends")

        # Generate sample trend data
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        dormancy_data = pd.DataFrame({
            'Date': dates,
            'Dormant_Accounts': np.random.randint(2800, 3200, len(dates)),
            'New_Dormant': np.random.randint(50, 150, len(dates)),
            'Reactivated': np.random.randint(30, 100, len(dates))
        })

        fig = px.line(dormancy_data, x='Date', y=['Dormant_Accounts', 'New_Dormant', 'Reactivated'],
                     title="Dormancy Trends Over Time")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏛️ CBUAE Compliance Breakdown")

        # Compliance by article
        compliance_data = pd.DataFrame({
            'Article': ['Art. 2.1.1', 'Art. 2.2', 'Art. 2.3', 'Art. 3.1', 'Art. 8.1'],
            'Compliance_Rate': [96.5, 94.2, 98.1, 92.8, 89.3],
            'Issues': [12, 23, 7, 34, 45]
        })

        fig = px.bar(compliance_data, x='Article', y='Compliance_Rate',
                    title="Compliance Rate by CBUAE Article",
                    color='Compliance_Rate',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Recent activity and alerts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔔 Recent Alerts")

        alerts = [
            {"time": "2 hours ago", "type": "High Value", "message": "High-value dormant account detected (AED 125,000)", "severity": "warning"},
            {"time": "5 hours ago", "type": "Compliance", "message": "Article 3.1 contact attempt overdue", "severity": "error"},
            {"time": "1 day ago", "type": "System", "message": "Monthly compliance report generated", "severity": "info"},
            {"time": "2 days ago", "type": "Risk", "message": "Risk score increased for retail portfolio", "severity": "warning"}
        ]

        for alert in alerts:
            if alert["severity"] == "error":
                st.error(f"🚨 **{alert['type']}** ({alert['time']})\n{alert['message']}")
            elif alert["severity"] == "warning":
                st.warning(f"⚠️ **{alert['type']}** ({alert['time']})\n{alert['message']}")
            else:
                st.info(f"ℹ️ **{alert['type']}** ({alert['time']})\n{alert['message']}")

    with col2:
        st.subheader("⚡ Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📁 Upload New Data", use_container_width=True):
                st.session_state.nav_override = "Data Upload"
                st.rerun()

            if st.button("🔍 Run Analysis", use_container_width=True):
                st.session_state.nav_override = "Workflow Monitor"
                st.rerun()

        with col2:
            if st.button("📋 Generate Report", use_container_width=True):
                st.session_state.nav_override = "Compliance Reports"
                st.rerun()

            if st.button("⚙️ System Status", use_container_width=True):
                st.session_state.nav_override = "System Admin"
                st.rerun()

        st.markdown("---")

        # System health indicators
        st.markdown("**🏥 System Health**")

        health_metrics = [
            ("Memory Agent", "🟢", "Healthy"),
            ("MCP Client", "🟢", "Connected"),
            ("Workflow Engine", "🟢", "Running"),
            ("Database", "🟢", "Online")
        ]

        for metric, status, description in health_metrics:
            st.markdown(f"{status} **{metric}**: {description}")

def show_data_upload():
    """Display data upload interface"""

    st.markdown('<div class="main-header">📁 Data Upload & Processing</div>', unsafe_allow_html=True)

    # File upload section
    st.subheader("📂 Upload Banking Data")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files containing banking compliance data"
    )

    col1, col2 = st.columns(2)

    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            options=[
                "comprehensive",
                "dormancy_only",
                "compliance_only",
                "risk_only"
            ],
            format_func=lambda x: {
                "comprehensive": "🔍 Comprehensive Analysis",
                "dormancy_only": "😴 Dormancy Analysis Only",
                "compliance_only": "✅ Compliance Verification Only",
                "risk_only": "⚠️ Risk Assessment Only"
            }[x]
        )

    with col2:
        report_date = st.date_input(
            "Report Date",
            value=datetime.now().date(),
            help="Date for the analysis report"
        )

    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            include_notifications = st.checkbox("📧 Send Notifications", value=True)
            auto_mapping = st.checkbox("🔗 Auto Data Mapping", value=True)
            quality_checks = st.checkbox("✅ Enhanced Quality Checks", value=True)

        with col2:
            confidence_threshold = st.slider("Mapping Confidence Threshold", 0.5, 1.0, 0.8)
            max_processing_time = st.number_input("Max Processing Time (minutes)", 1, 60, 15)
            priority_level = st.selectbox("Priority Level", ["Low", "Medium", "High"], index=1)

    if uploaded_file is not None:
        # Display file information
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # File preview
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Data summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            with col4:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

            # Column analysis
            with st.expander("📊 Column Analysis"):
                col_analysis = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                st.dataframe(col_analysis, use_container_width=True)

            # Process button
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button("🚀 Start Processing", use_container_width=True, type="primary"):
                    process_data_workflow(df, analysis_type, report_date, {
                        'include_notifications': include_notifications,
                        'auto_mapping': auto_mapping,
                        'quality_checks': quality_checks,
                        'confidence_threshold': confidence_threshold,
                        'max_processing_time': max_processing_time,
                        'priority_level': priority_level
                    })

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure the file is a valid CSV or Excel file with proper formatting.")

    # Sample data section
    st.markdown("---")
    st.subheader("📝 Sample Data Templates")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download Sample CSV", use_container_width=True):
            sample_data = generate_sample_data()
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sample_banking_data.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📋 View Data Schema", use_container_width=True):
            show_data_schema()

    with col3:
        if st.button("📖 Upload Guide", use_container_width=True):
            show_upload_guide()

def process_data_workflow(df: pd.DataFrame, analysis_type: str, report_date, options: Dict):
    """Process data through the workflow engine"""

    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create workflow execution tracking
    workflow_id = secrets.token_hex(16)

    if 'active_workflows' not in st.session_state:
        st.session_state.active_workflows = {}

    st.session_state.active_workflows[workflow_id] = {
        'status': 'starting',
        'progress': 0,
        'start_time': datetime.now(),
        'analysis_type': analysis_type,
        'total_records': len(df)
    }

    try:
        # Phase 1: Data preparation
        status_text.text("🔄 Preparing data for processing...")
        progress_bar.progress(10)

        # Convert DataFrame to the expected format
        processed_data = {
            'accounts': df.to_dict('records'),
            'metadata': {
                'source': 'user_upload',
                'total_records': len(df),
                'columns': list(df.columns),
                'upload_time': datetime.now().isoformat()
            }
        }

        # Phase 2: Execute workflow
        status_text.text("🚀 Executing workflow...")
        progress_bar.progress(30)

        # Mock workflow execution for demo
        import time
        time.sleep(1)  # Simulate processing

        # Phase 3: Generate results
        status_text.text("📊 Generating results...")
        progress_bar.progress(70)

        # Mock results
        results = {
            'success': True,
            'workflow_id': workflow_id,
            'status': 'completed',
            'total_processing_time': 2.5,
            'nodes_completed': 8,
            'nodes_failed': 0,
            'results': {
                'data_processing': {
                    'status': 'completed',
                    'quality_score': 0.95,
                    'records_processed': len(df)
                },
                'dormancy_analysis': {
                    'total_analyzed': len(df),
                    'dormant_found': max(1, int(len(df) * 0.23)),
                    'high_risk': max(1, int(len(df) * 0.05))
                },
                'compliance': {
                    'status': 'compliant',
                    'critical_violations': 0
                },
                'risk_assessment': {
                    'overall_score': 0.25
                }
            }
        }

        # Phase 4: Complete
        status_text.text("✅ Processing completed successfully!")
        progress_bar.progress(100)

        # Update workflow state
        st.session_state.active_workflows[workflow_id].update({
            'status': 'completed',
            'progress': 100,
            'results': results,
            'end_time': datetime.now()
        })

        # Show results
        time.sleep(1)
        show_workflow_results(results)

    except Exception as e:
        status_text.text(f"❌ Processing failed: {str(e)}")
        st.error(f"Workflow execution failed: {str(e)}")

        # Update workflow state
        st.session_state.active_workflows[workflow_id].update({
            'status': 'failed',
            'error': str(e),
            'end_time': datetime.now()
        })

def show_workflow_results(results: Dict):
    """Display workflow execution results"""

    st.markdown("---")
    st.subheader("🎉 Workflow Execution Results")

    if results.get('success'):
        st.success(f"✅ Workflow completed successfully!")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Processing Time",
                f"{results.get('total_processing_time', 0):.1f}s"
            )

        with col2:
            st.metric(
                "Nodes Completed",
                results.get('nodes_completed', 0)
            )

        with col3:
            records_processed = results.get('results', {}).get('data_processing', {}).get('records_processed', 0)
            st.metric(
                "Records Processed",
                f"{records_processed:,}"
            )

        with col4:
            quality_score = results.get('results', {}).get('data_processing', {}).get('quality_score', 0)
            st.metric(
                "Data Quality",
                f"{quality_score:.1%}"
            )

        # Detailed results
        st.subheader("📋 Detailed Results")

        workflow_results = results.get('results', {})

        # Data Processing Results
        if 'data_processing' in workflow_results:
            with st.expander("🔧 Data Processing Results"):
                dp_results = workflow_results['data_processing']
                st.json(dp_results)

        # Dormancy Analysis Results
        if 'dormancy_analysis' in workflow_results:
            with st.expander("😴 Dormancy Analysis Results"):
                da_results = workflow_results['dormancy_analysis']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyzed", da_results.get('total_analyzed', 0))
                with col2:
                    st.metric("Dormant Found", da_results.get('dormant_found', 0))
                with col3:
                    st.metric("High Risk", da_results.get('high_risk', 0))

                # Dormancy breakdown chart
                if da_results.get('dormant_found', 0) > 0:
                    dormancy_breakdown = pd.DataFrame({
                        'Category': ['Active', 'Dormant', 'High Risk'],
                        'Count': [
                            da_results.get('total_analyzed', 0) - da_results.get('dormant_found', 0),
                            da_results.get('dormant_found', 0) - da_results.get('high_risk', 0),
                            da_results.get('high_risk', 0)
                        ]
                    })

                    fig = px.pie(dormancy_breakdown, values='Count', names='Category',
                               title="Account Status Distribution")
                    st.plotly_chart(fig, use_container_width=True)

        # Compliance Results
        if 'compliance' in workflow_results:
            with st.expander("✅ Compliance Verification Results"):
                comp_results = workflow_results['compliance']

                if comp_results.get('status') == 'compliant':
                    st.success("✅ All compliance checks passed!")
                else:
                    st.warning("⚠️ Compliance issues detected")

                violations = comp_results.get('critical_violations', 0)
                if violations > 0:
                    st.error(f"🚨 {violations} critical violations found")
                else:
                    st.success("✅ No critical violations detected")

        # Risk Assessment Results
        if 'risk_assessment' in workflow_results:
            with st.expander("⚠️ Risk Assessment Results"):
                risk_results = workflow_results['risk_assessment']

                risk_score = risk_results.get('overall_score', 0)

                if risk_score < 0.3:
                    st.success(f"✅ Low Risk (Score: {risk_score:.2f})")
                elif risk_score < 0.7:
                    st.warning(f"⚠️ Medium Risk (Score: {risk_score:.2f})")
                else:
                    st.error(f"🚨 High Risk (Score: {risk_score:.2f})")

                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Risk Score"},
                    delta = {'reference': 50},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': "darkblue"},
                             'steps': [
                                 {'range': [0, 30], 'color': "lightgreen"},
                                 {'range': [30, 70], 'color': "yellow"},
                                 {'range': [70, 100], 'color': "red"}],
                             'threshold': {'line': {'color': "red", 'width': 4},
                                          'thickness': 0.75, 'value': 90}}))

                st.plotly_chart(fig, use_container_width=True)

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Generate Report", use_container_width=True):
                generate_analysis_report(results)

        with col2:
            if st.button("📧 Send Notifications", use_container_width=True):
                send_notifications(results)

        with col3:
            if st.button("💾 Save Results", use_container_width=True):
                save_workflow_results(results)

    else:
        st.error("❌ Workflow execution failed")
        if results.get('errors'):
            for error in results['errors']:
                st.error(f"🚨 {error}")

def generate_sample_data():
    """Generate sample banking data for download"""

    np.random.seed(42)

    # Generate sample accounts
    n_accounts = 100

    data = {
        'customer_id': [f'CUS{i:06d}' for i in range(1, n_accounts + 1)],
        'account_id': [f'ACC{i:06d}' for i in range(1, n_accounts + 1)],
        'customer_type': np.random.choice(['INDIVIDUAL', 'CORPORATE'], n_accounts, p=[0.8, 0.2]),
        'full_name_en': [f'Customer {i}' for i in range(1, n_accounts + 1)],
        'account_type': np.random.choice(['CURRENT', 'SAVINGS', 'INVESTMENT'], n_accounts, p=[0.4, 0.4, 0.2]),
        'account_status': np.random.choice(['ACTIVE', 'DORMANT', 'CLOSED'], n_accounts, p=[0.7, 0.25, 0.05]),
        'balance_current': np.random.lognormal(8, 1.5, n_accounts).round(2),
        'currency': np.random.choice(['AED', 'USD', 'EUR'], n_accounts, p=[0.7, 0.2, 0.1]),
        'last_transaction_date': pd.date_range(
            start='2020-01-01',
            end='2024-12-01',
            periods=n_accounts
        ).strftime('%Y-%m-%d'),
        'dormancy_period_months': np.random.randint(0, 60, n_accounts),
        'has_outstanding_facilities': np.random.choice(['YES', 'NO'], n_accounts, p=[0.3, 0.7]),
        'risk_rating': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], n_accounts, p=[0.6, 0.3, 0.1]),
        'last_contact_date': pd.date_range(
            start='2021-01-01',
            end='2024-11-01',
            periods=n_accounts
        ).strftime('%Y-%m-%d'),
        'address_known': np.random.choice(['YES', 'NO'], n_accounts, p=[0.85, 0.15])
    }

    return pd.DataFrame(data)

def show_data_schema():
    """Display the expected data schema"""

    st.subheader("📋 Expected Data Schema")

    schema_info = {
        'Column Name': [
            'customer_id', 'account_id', 'customer_type', 'full_name_en',
            'account_type', 'account_status', 'balance_current', 'currency',
            'last_transaction_date', 'dormancy_period_months', 'has_outstanding_facilities',
            'risk_rating', 'last_contact_date', 'address_known'
        ],
        'Data Type': [
            'String', 'String', 'String', 'String',
            'String', 'String', 'Float', 'String',
            'Date', 'Integer', 'Boolean', 'String', 'Date', 'Boolean'
        ],
        'Required': [
            'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'No',
            'Yes', 'No', 'No', 'No', 'No', 'No'
        ],
        'Description': [
            'Unique customer identifier',
            'Unique account identifier',
            'INDIVIDUAL or CORPORATE',
            'Customer name in English',
            'Account type (CURRENT, SAVINGS, etc.)',
            'Current account status',
            'Current account balance',
            'Account currency (AED, USD, etc.)',
            'Date of last customer transaction',
            'Months since account became dormant',
            'Whether customer has outstanding facilities',
            'Customer risk rating',
            'Date of last customer contact',
            'Whether customer address is known'
        ]
    }

    schema_df = pd.DataFrame(schema_info)
    st.dataframe(schema_df, use_container_width=True)

def show_upload_guide():
    """Display upload guide and best practices"""

    st.subheader("📖 Data Upload Guide")

    st.markdown("""
    ### 📁 Supported File Formats
    - **CSV files** (.csv) - Recommended format
    - **Excel files** (.xlsx, .xls) - Supported with automatic conversion
    
    ### 📋 Data Requirements
    
    #### Mandatory Fields
    - `customer_id` - Unique identifier for each customer
    - `account_id` - Unique identifier for each account  
    - `account_type` - Type of account (CURRENT, SAVINGS, INVESTMENT, etc.)
    - `account_status` - Current status (ACTIVE, DORMANT, CLOSED)
    - `balance_current` - Current account balance
    - `last_transaction_date` - Date of last customer-initiated transaction
    
    #### Recommended Fields
    - `customer_type` - INDIVIDUAL or CORPORATE
    - `currency` - Account currency (AED, USD, EUR, etc.)
    - `risk_rating` - Customer risk rating (LOW, MEDIUM, HIGH)
    - `has_outstanding_facilities` - YES/NO for outstanding credit facilities
    - `address_known` - Whether current address is known
    
    ### ✅ Best Practices
    
    1. **Data Quality**
       - Ensure all mandatory fields are populated
       - Use consistent date formats (YYYY-MM-DD recommended)
       - Standardize categorical values (e.g., always use "CURRENT" not "Current" or "current")
    
    2. **File Preparation**
       - Remove any merged cells in Excel files
       - Ensure headers are in the first row
       - Avoid special characters in column names
       - Keep file size under 50MB for optimal performance
    
    3. **Data Privacy**
       - Remove or anonymize sensitive customer information
       - Ensure compliance with data protection regulations
       - Use customer IDs instead of names where possible
    
    ### 🔧 Troubleshooting
    
    **Common Issues:**
    - **"Column not found" errors** - Check column names match expected schema
    - **Date parsing errors** - Ensure dates are in YYYY-MM-DD format
    - **Memory errors** - Split large files into smaller batches
    - **Encoding issues** - Save CSV files with UTF-8 encoding
    
    **Performance Tips:**
    - Files with 10,000+ records may take several minutes to process
    - Enable "Enhanced Quality Checks" for thorough validation
    - Use "Auto Data Mapping" for automatic field detection
    """)

def show_data_mapping():
    """Display data mapping interface"""

    st.markdown('<div class="main-header">🔗 Intelligent Data Mapping</div>', unsafe_allow_html=True)

    st.markdown("""
    The data mapping system uses AI to automatically map your data fields to the standard 
    banking compliance schema. Upload your file and let our system intelligently match fields.
    """)

    # Upload section
    uploaded_file = st.file_uploader(
        "Choose a file for mapping analysis",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file to analyze and map its fields"
    )

    col1, col2 = st.columns(2)

    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.5, 1.0, 0.8, 0.05,
            help="Minimum confidence level for automatic mapping"
        )

        use_llm_assistance = st.checkbox(
            "🤖 Use AI Assistance",
            value=True,
            help="Use large language model for low-confidence mappings"
        )

    with col2:
        save_patterns = st.checkbox(
            "💾 Save Mapping Patterns",
            value=True,
            help="Save successful mappings for future use"
        )

        preview_mappings = st.checkbox(
            "👁️ Preview Before Apply",
            value=True,
            help="Review mappings before applying them"
        )

    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")

            # Show column preview
            st.subheader("📊 Source Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Perform mapping analysis
            if st.button("🔍 Analyze Field Mappings", type="primary"):

                with st.spinner("🤖 Analyzing field mappings..."):
                    mapping_results = perform_data_mapping_analysis(
                        df, confidence_threshold, use_llm_assistance
                    )

                show_mapping_results(mapping_results, df)

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def perform_data_mapping_analysis(df: pd.DataFrame, confidence_threshold: float, use_llm: bool):
    """Perform intelligent data mapping analysis"""

    # Mock mapping analysis for demo
    source_columns = df.columns.tolist()

    # Standard target schema
    target_schema = [
        'customer_id', 'account_id', 'customer_type', 'full_name_en',
        'account_type', 'account_status', 'balance_current', 'currency',
        'last_transaction_date', 'dormancy_period_months', 'has_outstanding_facilities',
        'risk_rating', 'last_contact_date', 'address_known'
    ]

    # Simple mapping logic for demo
    mappings = []

    for source_col in source_columns:
        source_lower = source_col.lower().replace('_', ' ').replace('-', ' ')

        best_match = None
        best_confidence = 0.0

        for target_col in target_schema:
            target_lower = target_col.lower().replace('_', ' ')

            # Simple similarity based on common words
            source_words = set(source_lower.split())
            target_words = set(target_lower.split())

            if source_words & target_words:
                confidence = len(source_words & target_words) / len(source_words | target_words)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = target_col

        # Boost confidence for exact or near-exact matches
        if source_lower in target_lower or target_lower in source_lower:
            best_confidence = min(1.0, best_confidence + 0.3)

        mappings.append({
            'source_field': source_col,
            'target_field': best_match,
            'confidence': best_confidence,
            'confidence_level': 'High' if best_confidence >= 0.8 else 'Medium' if best_confidence >= 0.6 else 'Low',
            'mapping_strategy': 'Automatic' if best_confidence >= confidence_threshold else 'Manual',
            'sample_values': df[source_col].dropna().head(3).tolist()
        })

    return {
        'mappings': mappings,
        'total_fields': len(source_columns),
        'high_confidence': len([m for m in mappings if m['confidence'] >= 0.8]),
        'medium_confidence': len([m for m in mappings if 0.6 <= m['confidence'] < 0.8]),
        'low_confidence': len([m for m in mappings if m['confidence'] < 0.6]),
        'unmapped_fields': [m['source_field'] for m in mappings if m['target_field'] is None]
    }

def show_mapping_results(results: Dict, df: pd.DataFrame):
    """Display mapping analysis results"""

    st.subheader("🎯 Mapping Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Fields", results['total_fields'])
    with col2:
        st.metric("High Confidence", results['high_confidence'])
    with col3:
        st.metric("Medium Confidence", results['medium_confidence'])
    with col4:
        st.metric("Low Confidence", results['low_confidence'])

    # Confidence distribution chart
    confidence_data = pd.DataFrame({
        'Confidence Level': ['High (≥80%)', 'Medium (60-80%)', 'Low (<60%)'],
        'Count': [results['high_confidence'], results['medium_confidence'], results['low_confidence']]
    })

    fig = px.bar(confidence_data, x='Confidence Level', y='Count',
                title="Mapping Confidence Distribution",
                color='Count', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)

    # Detailed mappings table
    st.subheader("📋 Detailed Field Mappings")

    mappings_df = pd.DataFrame(results['mappings'])

    # Color code by confidence
    def highlight_confidence(row):
        if row['confidence'] >= 0.8:
            return ['background-color: #d4edda'] * len(row)
        elif row['confidence'] >= 0.6:
            return ['background-color: #fff3cd'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)

    styled_df = mappings_df.style.apply(highlight_confidence, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    # Allow manual adjustments
    if st.checkbox("🔧 Enable Manual Adjustments"):
        st.subheader("✏️ Manual Field Mapping")

        # Create manual mapping interface
        manual_mappings = {}

        target_options = ['None'] + [
            'customer_id', 'account_id', 'customer_type', 'full_name_en',
            'account_type', 'account_status', 'balance_current', 'currency',
            'last_transaction_date', 'dormancy_period_months', 'has_outstanding_facilities',
            'risk_rating', 'last_contact_date', 'address_known'
        ]

        for i, mapping in enumerate(results['mappings']):
            if mapping['confidence'] < 0.8:  # Only show low/medium confidence mappings
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.text(f"Source: {mapping['source_field']}")
                    st.caption(f"Sample: {', '.join(map(str, mapping['sample_values']))}")

                with col2:
                    default_idx = 0
                    if mapping['target_field'] and mapping['target_field'] in target_options:
                        default_idx = target_options.index(mapping['target_field'])

                    manual_mappings[mapping['source_field']] = st.selectbox(
                        f"Map to:",
                        target_options,
                        index=default_idx,
                        key=f"manual_mapping_{i}"
                    )

                with col3:
                    st.metric("Confidence", f"{mapping['confidence']:.0%}")

    # Apply mappings button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("✅ Apply Mappings and Continue", use_container_width=True, type="primary"):
            # Apply mappings and redirect to processing
            st.success("🎉 Mappings applied successfully!")
            st.info("Redirecting to data processing...")

            # Here you would apply the mappings and continue with processing
            # For demo, just show a success message
            st.balloons()

def show_workflow_monitor():
    """Display workflow monitoring interface"""

    st.markdown('<div class="main-header">⚙️ Workflow Monitor</div>', unsafe_allow_html=True)

    # Active workflows section
    st.subheader("🔄 Active Workflows")

    if 'active_workflows' in st.session_state and st.session_state.active_workflows:
        for workflow_id, workflow_info in st.session_state.active_workflows.items():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                with col1:
                    st.markdown(f"**Workflow ID:** `{workflow_id[:12]}...`")
                    st.caption(f"Type: {workflow_info.get('analysis_type', 'Unknown')}")

                with col2:
                    status = workflow_info.get('status', 'unknown')
                    if status == 'completed':
                        st.success(f"✅ {status.title()}")
                    elif status == 'failed':
                        st.error(f"❌ {status.title()}")
                    elif status in ['starting', 'running']:
                        st.info(f"🔄 {status.title()}")
                    else:
                        st.warning(f"⚠️ {status.title()}")

                with col3:
                    progress = workflow_info.get('progress', 0)
                    st.progress(progress / 100)
                    st.caption(f"Progress: {progress}%")

                with col4:
                    if st.button("👁️", key=f"view_{workflow_id}"):
                        show_workflow_details(workflow_id, workflow_info)

                st.markdown("---")
    else:
        st.info("📭 No active workflows found")

        if st.button("🚀 Start New Workflow", use_container_width=True):
            st.session_state.nav_override = "Data Upload"
            st.rerun()

    # Workflow history
    st.subheader("📈 Workflow Statistics")

    # Generate mock statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Workflows", "47", delta="3 this week")
    with col2:
        st.metric("Success Rate", "94.7%", delta="2.1%")
    with col3:
        st.metric("Avg Processing Time", "3.2 min", delta="-0.5 min")
    with col4:
        st.metric("Data Quality Score", "0.95", delta="0.02")

    # Performance charts
    col1, col2 = st.columns(2)

    with col1:
        # Workflow volume over time
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='W')
        workflow_volume = pd.DataFrame({
            'Date': dates,
            'Workflows': np.random.randint(5, 15, len(dates)),
            'Success': np.random.randint(4, 14, len(dates))
        })

        fig = px.line(workflow_volume, x='Date', y=['Workflows', 'Success'],
                     title="Weekly Workflow Volume")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Processing time distribution
        processing_times = np.random.lognormal(1, 0.5, 100)

        fig = px.histogram(x=processing_times, nbins=20,
                          title="Processing Time Distribution",
                          labels={'x': 'Processing Time (minutes)', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

    # System health monitoring
    st.subheader("🏥 System Health")

    health_data = [
        {"Component": "Workflow Engine", "Status": "Healthy", "Uptime": "99.9%", "Last Check": "2 min ago"},
        {"Component": "Memory Agent", "Status": "Healthy", "Uptime": "99.8%", "Last Check": "1 min ago"},
        {"Component": "MCP Client", "Status": "Connected", "Uptime": "99.7%", "Last Check": "30 sec ago"},
        {"Component": "Database", "Status": "Healthy", "Uptime": "100%", "Last Check": "1 min ago"},
        {"Component": "Notification Service", "Status": "Healthy", "Uptime": "99.5%", "Last Check": "2 min ago"}
    ]

    health_df = pd.DataFrame(health_data)

    # Style the health status
    def style_status(val):
        if val in ['Healthy', 'Connected']:
            return 'color: green'
        elif val in ['Warning']:
            return 'color: orange'
        elif val in ['Error', 'Disconnected']:
            return 'color: red'
        return ''

    styled_health = health_df.style.applymap(style_status, subset=['Status'])
    st.dataframe(styled_health, use_container_width=True)

def show_workflow_details(workflow_id: str, workflow_info: Dict):
    """Show detailed workflow information"""

    st.markdown(f"### 📊 Workflow Details: `{workflow_id[:12]}...`")

    # Basic information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Workflow Information:**")
        st.text(f"ID: {workflow_id}")
        st.text(f"Type: {workflow_info.get('analysis_type', 'Unknown')}")
        st.text(f"Status: {workflow_info.get('status', 'Unknown')}")
        st.text(f"Records: {workflow_info.get('total_records', 0):,}")

    with col2:
        st.markdown("**Timing Information:**")
        start_time = workflow_info.get('start_time')
        if start_time:
            st.text(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        end_time = workflow_info.get('end_time')
        if end_time:
            st.text(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = end_time - start_time
            st.text(f"Duration: {duration.total_seconds():.1f}s")

    # Results if available
    if 'results' in workflow_info:
        st.markdown("**Results:**")
        st.json(workflow_info['results'])

    # Error information if failed
    if workflow_info.get('status') == 'failed' and 'error' in workflow_info:
        st.error(f"Error: {workflow_info['error']}")

def show_compliance_reports():
    """Display compliance reports interface"""

    st.markdown('<div class="main-header">📋 Compliance Reports</div>', unsafe_allow_html=True)

    # Report generation section
    st.subheader("📄 Generate New Report")

    col1, col2 = st.columns(2)

    with col1:
        report_type = st.selectbox(
            "Report Type",
            options=[
                "comprehensive_compliance",
                "dormancy_analysis",
                "risk_assessment",
                "cbuae_summary",
                "executive_dashboard"
            ],
            format_func=lambda x: {
                "comprehensive_compliance": "📊 Comprehensive Compliance Report",
                "dormancy_analysis": "😴 Dormancy Analysis Report",
                "risk_assessment": "⚠️ Risk Assessment Report",
                "cbuae_summary": "🏛️ CBUAE Regulatory Summary",
                "executive_dashboard": "👔 Executive Dashboard"
            }[x]
        )

        report_period = st.selectbox(
            "Report Period",
            ["Last 30 Days", "Last Quarter", "Last 6 Months", "Last Year", "Custom Range"]
        )

    with col2:
        include_charts = st.checkbox("📊 Include Visualizations", value=True)
        include_recommendations = st.checkbox("💡 Include Recommendations", value=True)
        export_format = st.selectbox("Export Format", ["PDF", "Excel", "Word", "HTML"])

        if report_period == "Custom Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")

    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        generate_compliance_report_interface(report_type, report_period, include_charts, include_recommendations, export_format)

    st.markdown("---")

    # Recent reports
    st.subheader("📚 Recent Reports")

    # Mock recent reports data
    recent_reports = [
        {
            "Title": "Monthly Compliance Summary - November 2024",
            "Type": "Comprehensive Compliance",
            "Generated": "2024-12-01 09:30",
            "Status": "Completed",
            "Size": "2.3 MB"
        },
        {
            "Title": "Q4 Dormancy Analysis Report",
            "Type": "Dormancy Analysis",
            "Generated": "2024-11-28 14:15",
            "Status": "Completed",
            "Size": "1.8 MB"
        },
        {
            "Title": "Risk Assessment - High Value Accounts",
            "Type": "Risk Assessment",
            "Generated": "2024-11-25 11:45",
            "Status": "Completed",
            "Size": "945 KB"
        }
    ]

    for report in recent_reports:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])

            with col1:
                st.markdown(f"**{report['Title']}**")
                st.caption(f"Type: {report['Type']}")

            with col2:
                st.text(report['Generated'])

            with col3:
                st.success(f"✅ {report['Status']}")

            with col4:
                st.text(report['Size'])

            with col5:
                if st.button("📥", key=f"download_{report['Title']}", help="Download Report"):
                    st.success("📥 Download started!")

            st.markdown("---")

    # Compliance analytics
    st.subheader("📈 Compliance Analytics")

    # CBUAE article compliance over time
    col1, col2 = st.columns(2)

    with col1:
        # Compliance trends
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        compliance_trends = pd.DataFrame({
            'Month': dates,
            'Overall_Compliance': np.random.uniform(0.85, 0.98, len(dates)),
            'Article_2_1_1': np.random.uniform(0.88, 0.96, len(dates)),
            'Article_3_1': np.random.uniform(0.82, 0.94, len(dates)),
            'Article_8_1': np.random.uniform(0.79, 0.91, len(dates))
        })

        fig = px.line(compliance_trends, x='Month',
                     y=['Overall_Compliance', 'Article_2_1_1', 'Article_3_1', 'Article_8_1'],
                     title="Compliance Trends by CBUAE Article")
        fig.update_layout(yaxis_title="Compliance Rate", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Violation breakdown
        violation_data = pd.DataFrame({
            'Article': ['Art. 2.1.1', 'Art. 2.2', 'Art. 3.1', 'Art. 3.4', 'Art. 8.1'],
            'Violations': [5, 12, 8, 3, 15],
            'Severity': ['Medium', 'High', 'Low', 'Medium', 'High']
        })

        fig = px.bar(violation_data, x='Article', y='Violations', color='Severity',
                    title="Violations by CBUAE Article",
                    color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def generate_compliance_report_interface(report_type: str, report_period: str,
                                        include_charts: bool, include_recommendations: bool,
                                        export_format: str):
    """Generate compliance report interface"""

    with st.spinner("📄 Generating compliance report..."):
        # Simulate report generation
        import time
        time.sleep(2)

        # Mock report data
        report_data = {
            'title': f'{report_type.replace("_", " ").title()} Report',
            'period': report_period,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_accounts': 12847,
            'dormant_accounts': 2954,
            'compliance_score': 94.7,
            'critical_violations': 0,
            'recommendations': [
                'Implement automated contact attempt tracking',
                'Review high-value dormant account procedures',
                'Update customer address verification process'
            ] if include_recommendations else []
        }

        st.success("✅ Report generated successfully!")

        # Display report preview
        st.subheader("📋 Report Preview")

        st.markdown(f"""
        **Report Title:** {report_data['title']}  
        **Period:** {report_data['period']}  
        **Generated:** {report_data['generated_at']}  
        
        **Executive Summary:**
        - Total Accounts Analyzed: {report_data['total_accounts']:,}
        - Dormant Accounts Identified: {report_data['dormant_accounts']:,}
        - Overall Compliance Score: {report_data['compliance_score']}%
        - Critical Violations: {report_data['critical_violations']}
        """)

        if include_charts:
            # Sample chart
            fig = px.pie(
                values=[report_data['total_accounts'] - report_data['dormant_accounts'], report_data['dormant_accounts']],
                names=['Active', 'Dormant'],
                title="Account Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        if include_recommendations and report_data['recommendations']:
            st.markdown("**Key Recommendations:**")
            for i, rec in enumerate(report_data['recommendations'], 1):
                st.markdown(f"{i}. {rec}")

        # Download button
        if st.button(f"📥 Download {export_format} Report", use_container_width=True):
            st.success(f"📥 {export_format} report download started!")

def show_risk_analysis():
    """Display risk analysis interface"""

    st.markdown('<div class="main-header">⚠️ Risk Analysis Dashboard</div>', unsafe_allow_html=True)

    # Risk overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Risk Score", "0.25", delta="-0.03", delta_color="inverse")
    with col2:
        st.metric("High Risk Accounts", "127", delta="+5")
    with col3:
        st.metric("Critical Alerts", "3", delta="-2", delta_color="inverse")
    with col4:
        st.metric("Risk Trend", "Decreasing", delta="Stable", delta_color="inverse")

    # Risk distribution
    st.subheader("📊 Risk Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Risk level distribution
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
            'Account Count': [8456, 3214, 127, 3],
            'Percentage': [71.2, 27.1, 1.1, 0.6]
        })

        fig = px.pie(risk_data, values='Account Count', names='Risk Level',
                    title="Risk Level Distribution",
                    color_discrete_map={
                        'Low': '#28a745',
                        'Medium': '#ffc107',
                        'High': '#fd7e14',
                        'Critical': '#dc3545'
                    })
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk factors
        risk_factors = pd.DataFrame({
            'Factor': ['High Value Dormant', 'Compliance Violations', 'Operational Risk',
                      'Reputational Risk', 'Regulatory Risk'],
            'Impact Score': [0.85, 0.72, 0.45, 0.38, 0.29],
            'Trend': ['↑', '↓', '→', '↓', '→']
        })

        fig = px.bar(risk_factors, x='Factor', y='Impact Score',
                    title="Risk Factor Impact Analysis",
                    color='Impact Score',
                    color_continuous_scale='Reds')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # High risk accounts
    st.subheader("🚨 High Risk Accounts")

    # Mock high risk accounts data
    high_risk_accounts = pd.DataFrame({
        'Account ID': ['ACC001234', 'ACC005678', 'ACC009012'],
        'Customer Name': ['Customer A', 'Customer B', 'Customer C'],
        'Risk Score': [0.92, 0.87, 0.83],
        'Primary Risk': ['High Value Dormant', 'Compliance Violation', 'Operational Risk'],
        'Balance (AED)': [125000, 87500, 245000],
        'Last Contact': ['2021-03-15', '2020-11-22', '2022-01-08'],
        'Status': ['Requires Immediate Action', 'Under Review', 'Monitoring']
    })

    # Style the dataframe
    def highlight_risk_score(val):
        if val >= 0.9:
            return 'background-color: #f8d7da'
        elif val >= 0.8:
            return 'background-color: #fff3cd'
        return ''

    styled_df = high_risk_accounts.style.applymap(highlight_risk_score, subset=['Risk Score'])
    st.dataframe(styled_df, use_container_width=True)

    # Risk trends over time
    st.subheader("📈 Risk Trends")

    # Generate risk trend data
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='W')
    risk_trends = pd.DataFrame({
        'Date': dates,
        'Overall_Risk': np.random.uniform(0.15, 0.35, len(dates)),
        'High_Value_Risk': np.random.uniform(0.25, 0.45, len(dates)),
        'Compliance_Risk': np.random.uniform(0.10, 0.30, len(dates)),
        'Operational_Risk': np.random.uniform(0.05, 0.25, len(dates))
    })

    fig = px.line(risk_trends, x='Date',
                 y=['Overall_Risk', 'High_Value_Risk', 'Compliance_Risk', 'Operational_Risk'],
                 title="Risk Score Trends Over Time")
    fig.update_layout(yaxis_title="Risk Score", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Risk mitigation actions
    st.subheader("🛡️ Risk Mitigation Actions")

    mitigation_actions = [
        {
            "Priority": "Critical",
            "Action": "Contact high-value dormant account holders",
            "Target Date": "2024-12-15",
            "Owner": "Customer Relations Team",
            "Status": "In Progress"
        },
        {
            "Priority": "High",
            "Action": "Review compliance violation procedures",
            "Target Date": "2024-12-20",
            "Owner": "Compliance Team",
            "Status": "Not Started"
        },
        {
            "Priority": "Medium",
            "Action": "Update operational risk controls",
            "Target Date": "2024-12-31",
            "Owner": "Risk Management",
            "Status": "Planning"
        }
    ]

    for action in mitigation_actions:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])

            with col1:
                if action["Priority"] == "Critical":
                    st.error(f"🚨 {action['Priority']}")
                elif action["Priority"] == "High":
                    st.warning(f"⚠️ {action['Priority']}")
                else:
                    st.info(f"ℹ️ {action['Priority']}")

            with col2:
                st.markdown(f"**{action['Action']}**")

            with col3:
                st.text(f"Due: {action['Target Date']}")

            with col4:
                st.text(f"Owner: {action['Owner']}")

            with col5:
                if action["Status"] == "In Progress":
                    st.success(f"✅ {action['Status']}")
                elif action["Status"] == "Not Started":
                    st.error(f"❌ {action['Status']}")
                else:
                    st.info(f"📋 {action['Status']}")

            st.markdown("---")

def show_notifications():
    """Display notifications interface"""

    st.markdown('<div class="main-header">📧 Notifications & Alerts</div>', unsafe_allow_html=True)

    # Notification preferences
    st.subheader("⚙️ Notification Preferences")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Channel Preferences:**")
        email_enabled = st.checkbox("📧 Email Notifications", value=True)
        slack_enabled = st.checkbox("💬 Slack Integration", value=False)
        dashboard_enabled = st.checkbox("📊 Dashboard Alerts", value=True)
        mobile_enabled = st.checkbox("📱 Mobile Push", value=False)

    with col2:
        st.markdown("**Alert Thresholds:**")
        priority_threshold = st.selectbox(
            "Minimum Priority Level",
            ["Low", "Medium", "High", "Critical"],
            index=1
        )

        quiet_hours = st.checkbox("🌙 Enable Quiet Hours", value=True)
        if quiet_hours:
            col1, col2 = st.columns(2)
            with col1:
                quiet_start = st.time_input("Quiet Hours Start", value=datetime.strptime("22:00", "%H:%M").time())
            with col2:
                quiet_end = st.time_input("Quiet Hours End", value=datetime.strptime("06:00", "%H:%M").time())

    if st.button("💾 Save Preferences", type="primary"):
        st.success("✅ Notification preferences saved!")

    st.markdown("---")

    # Recent notifications
    st.subheader("🔔 Recent Notifications")

    # Notification filters
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_priority = st.selectbox("Filter by Priority", ["All", "Critical", "High", "Medium", "Low"])
    with col2:
        filter_type = st.selectbox("Filter by Type", ["All", "Compliance", "Risk", "System", "Dormancy"])
    with col3:
        filter_status = st.selectbox("Filter by Status", ["All", "Unread", "Read", "Acknowledged"])

    # Mock notifications data
    notifications = [
        {
            "timestamp": "2024-12-01 14:30",
            "priority": "Critical",
            "type": "Risk",
            "title": "High-Value Dormant Account Alert",
            "message": "Account ACC123456 with balance AED 150,000 has been dormant for 42 months",
            "status": "Unread",
            "actions": ["Contact Customer", "Escalate", "Mark as Reviewed"]
        },
        {
            "timestamp": "2024-12-01 11:15",
            "priority": "High",
            "type": "Compliance",
            "title": "Article 3.1 Compliance Issue",
            "message": "15 accounts require contact attempts per CBUAE Article 3.1 requirements",
            "status": "Read",
            "actions": ["Generate Contact List", "Schedule Campaign", "Mark as Reviewed"]
        },
        {
            "timestamp": "2024-12-01 09:45",
            "priority": "Medium",
            "type": "System",
            "title": "Weekly Compliance Report Available",
            "message": "Your weekly compliance analysis report has been generated and is ready for download",
            "status": "Read",
            "actions": ["Download Report", "Share Report", "Archive"]
        },
        {
            "timestamp": "2024-11-30 16:20",
            "priority": "High",
            "type": "Dormancy",
            "title": "Quarterly Dormancy Review",
            "message": "Q4 dormancy analysis complete: 127 new dormant accounts identified",
            "status": "Acknowledged",
            "actions": ["View Details", "Export Data", "Create Action Plan"]
        }
    ]

    # Apply filters
    filtered_notifications = notifications
    if filter_priority != "All":
        filtered_notifications = [n for n in filtered_notifications if n["priority"] == filter_priority]
    if filter_type != "All":
        filtered_notifications = [n for n in filtered_notifications if n["type"] == filter_type]
    if filter_status != "All":
        filtered_notifications = [n for n in filtered_notifications if n["status"] == filter_status]

    # Display notifications
    for i, notification in enumerate(filtered_notifications):
        with st.container():
            # Header row
            col1, col2, col3, col4 = st.columns([1, 3, 1, 1])

            with col1:
                if notification["priority"] == "Critical":
                    st.error(f"🚨 {notification['priority']}")
                elif notification["priority"] == "High":
                    st.warning(f"⚠️ {notification['priority']}")
                elif notification["priority"] == "Medium":
                    st.info(f"ℹ️ {notification['priority']}")
                else:
                    st.success(f"✅ {notification['priority']}")

            with col2:
                st.markdown(f"**{notification['title']}**")
                st.caption(f"{notification['type']} • {notification['timestamp']}")

            with col3:
                if notification["status"] == "Unread":
                    st.markdown("🔴 **Unread**")
                elif notification["status"] == "Read":
                    st.markdown("🔵 Read")
                else:
                    st.markdown("✅ Acknowledged")

            with col4:
                if st.button("⋯", key=f"actions_{i}", help="Actions"):
                    st.write("Action menu would appear here")

            # Message content
            st.markdown(notification["message"])

            # Action buttons
            if notification["status"] == "Unread":
                action_cols = st.columns(len(notification["actions"]) + 1)

                for j, action in enumerate(notification["actions"]):
                    with action_cols[j]:
                        if st.button(action, key=f"action_{i}_{j}", size="sm"):
                            st.success(f"✅ {action} executed!")

                with action_cols[-1]:
                    if st.button("✓ Mark Read", key=f"mark_read_{i}", size="sm"):
                        st.success("✅ Marked as read!")

            st.markdown("---")

    if not filtered_notifications:
        st.info("📭 No notifications match the current filters")

    # Notification statistics
    st.subheader("📊 Notification Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Notifications", len(notifications))
    with col2:
        unread_count = len([n for n in notifications if n["status"] == "Unread"])
        st.metric("Unread", unread_count)
    with col3:
        critical_count = len([n for n in notifications if n["priority"] == "Critical"])
        st.metric("Critical Alerts", critical_count)
    with col4:
        today_count = len([n for n in notifications if n["timestamp"].startswith("2024-12-01")])
        st.metric("Today", today_count)

def show_system_admin():
    """Display system administration interface"""

    st.markdown('<div class="main-header">🔧 System Administration</div>', unsafe_allow_html=True)

    # System status overview
    st.subheader("🏥 System Health Overview")

    # System components status
    components = [
        {"name": "Workflow Engine", "status": "Healthy", "uptime": "99.9%", "version": "1.0.0"},
        {"name": "Memory Agent", "status": "Healthy", "uptime": "99.8%", "version": "1.0.0"},
        {"name": "MCP Client", "status": "Connected", "uptime": "99.7%", "version": "1.0.0"},
        {"name": "Data Processing Agent", "status": "Healthy", "uptime": "99.9%", "version": "1.0.0"},
        {"name": "Dormancy Analysis Agent", "status": "Healthy", "uptime": "99.8%", "version": "1.0.0"},
        {"name": "Compliance Agent", "status": "Healthy", "uptime": "99.6%", "version": "1.0.0"},
        {"name": "Risk Assessment Agent", "status": "Healthy", "uptime": "99.7%", "version": "1.0.0"},
        {"name": "Notification Agent", "status": "Healthy", "uptime": "99.5%", "version": "1.0.0"}
    ]

    # Display in a grid
    cols = st.columns(2)
    for i, component in enumerate(components):
        with cols[i % 2]:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    if component["status"] in ["Healthy", "Connected"]:
                        st.success(f"✅ {component['name']}")
                    else:
                        st.error(f"❌ {component['name']}")
                    st.caption(f"Version: {component['version']}")

                with col2:
                    st.metric("Uptime", component["uptime"])

                with col3:
                    if st.button("🔄", key=f"restart_{i}", help="Restart Component"):
                        st.success(f"🔄 {component['name']} restarted!")

    st.markdown("---")

    # Performance metrics
    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("CPU Usage", "23%", delta="-5%", delta_color="inverse")
    with col2:
        st.metric("Memory Usage", "1.2 GB", delta="0.1 GB")
    with col3:
        st.metric("Active Sessions", "47", delta="3")
    with col4:
        st.metric("Response Time", "245ms", delta="-15ms", delta_color="inverse")

    # Performance charts
    col1, col2 = st.columns(2)

    with col1:
        # CPU and Memory usage over time
        times = pd.date_range(start='2024-12-01 00:00', end='2024-12-01 23:59', freq='H')
        perf_data = pd.DataFrame({
            'Time': times,
            'CPU_Usage': np.random.uniform(15, 35, len(times)),
            'Memory_Usage': np.random.uniform(0.8, 1.5, len(times))
        })

        fig = px.line(perf_data, x='Time', y=['CPU_Usage', 'Memory_Usage'],
                     title="System Resource Usage (24h)")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Response time distribution
        response_times = np.random.gamma(2, 50, 200)

        fig = px.histogram(x=response_times, nbins=20,
                          title="Response Time Distribution",
                          labels={'x': 'Response Time (ms)', 'y': 'Frequency'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Configuration management
    st.subheader("⚙️ System Configuration")

    with st.expander("🔧 Agent Configuration"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data Processing Agent:**")
            max_file_size = st.number_input("Max File Size (MB)", value=50)
            quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.8)
            auto_validation = st.checkbox("Auto Validation", value=True)

        with col2:
            st.markdown("**Workflow Engine:**")
            max_concurrent = st.number_input("Max Concurrent Workflows", value=10)
            timeout_minutes = st.number_input("Workflow Timeout (minutes)", value=30)
            auto_retry = st.checkbox("Auto Retry Failed Workflows", value=True)

        if st.button("💾 Save Configuration"):
            st.success("✅ Configuration saved successfully!")

    with st.expander("🔐 Security Settings"):
        col1, col2 = st.columns(2)

        with col1:
            session_timeout = st.number_input("Session Timeout (minutes)", value=480)
            max_login_attempts = st.number_input("Max Login Attempts", value=5)
            require_2fa = st.checkbox("Require Two-Factor Authentication", value=False)

        with col2:
            password_complexity = st.selectbox("Password Complexity", ["Basic", "Standard", "High"])
            audit_logging = st.checkbox("Enable Audit Logging", value=True)
            encrypt_sensitive = st.checkbox("Encrypt Sensitive Data", value=True)

        if st.button("🔒 Update Security Settings"):
            st.success("✅ Security settings updated!")

    # System maintenance
    st.subheader("🧹 System Maintenance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Database Maintenance:**")
        if st.button("🗂️ Optimize Database", use_container_width=True):
            with st.spinner("Optimizing database..."):
                import time
                time.sleep(2)
                st.success("✅ Database optimized!")

        if st.button("🧹 Clean Old Logs", use_container_width=True):
            st.success("✅ Old logs cleaned!")

    with col2:
        st.markdown("**Memory Management:**")
        if st.button("💾 Clear Cache", use_container_width=True):
            st.success("✅ Cache cleared!")

        if st.button("🔄 Restart Memory Agent", use_container_width=True):
            st.success("✅ Memory agent restarted!")

    with col3:
        st.markdown("**System Updates:**")
        if st.button("🔍 Check for Updates", use_container_width=True):
            st.info("ℹ️ System is up to date!")

        if st.button("📊 Generate Health Report", use_container_width=True):
            st.success("✅ Health report generated!")

    # Logs viewer
    st.subheader("📜 System Logs")

    log_type = st.selectbox("Log Type", ["Application", "Error", "Audit", "Performance"])
    log_level = st.selectbox("Log Level", ["All", "INFO", "WARNING", "ERROR", "CRITICAL"])

    # Mock log entries
    mock_logs = [
        {"timestamp": "2024-12-01 14:35:22", "level": "INFO", "component": "Workflow Engine", "message": "Workflow abc123 completed successfully"},
        {"timestamp": "2024-12-01 14:30:15", "level": "WARNING", "component": "Memory Agent", "message": "Memory usage approaching threshold (85%)"},
        {"timestamp": "2024-12-01 14:25:08", "level": "INFO", "component": "Data Processing", "message": "Processed 1,247 records in 2.3 seconds"},
        {"timestamp": "2024-12-01 14:20:45", "level": "ERROR", "component": "MCP Client", "message": "Connection timeout to external service"},
        {"timestamp": "2024-12-01 14:15:33", "level": "INFO", "component": "Compliance Agent", "message": "Compliance verification completed for 847 accounts"}
    ]

    # Display logs
    for log in mock_logs:
        col1, col2, col3, col4 = st.columns([2, 1, 2, 4])

        with col1:
            st.text(log["timestamp"])

        with col2:
            if log["level"] == "ERROR":
                st.error(log["level"])
            elif log["level"] == "WARNING":
                st.warning(log["level"])
            else:
                st.success(log["level"])

        with col3:
            st.text(log["component"])

        with col4:
            st.text(log["message"])

def generate_analysis_report(results: Dict):
    """Generate analysis report"""
    st.success("📄 Generating comprehensive analysis report...")
    # In a real implementation, this would generate a proper report
    st.balloons()

def send_notifications(results: Dict):
    """Send notifications based on results"""
    st.success("📧 Sending notifications to stakeholders...")
    # In a real implementation, this would send actual notifications
    st.balloons()

def save_workflow_results(results: Dict):
    """Save workflow results"""
    st.success("💾 Workflow results saved successfully!")
    # In a real implementation, this would save to database
    st.balloons()

# ========================= MAIN APPLICATION ENTRY POINT =========================

def main():
    """Main application entry point"""

    # Initialize authentication
    init_auth()

    # Check authentication status
    if not st.session_state.authenticated:
        login_form()
    else:
        main_app()

if __name__ == "__main__":
    main()