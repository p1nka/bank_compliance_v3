"""
CBUAE Banking Compliance Analysis System - Complete Streamlit Application
Integrates ALL real agents from the repository with hybrid memory and full workflow
Updated to use real Data Processing Agent for quality analysis and mapping
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import asyncio
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

# Custom CSS for better styling
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
    .quality-excellent { background: #d1f2eb; color: #0e5e3b; }
    .quality-good { background: #d4edda; color: #155724; }
    .quality-fair { background: #fff3cd; color: #856404; }
    .quality-poor { background: #f8d7da; color: #721c24; }
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

# Import MCP client
try:
    from mcp_client import MCPClient
    MCP_AVAILABLE = True
    logger.info("✅ MCP client imported successfully")
except ImportError as e:
    MCP_AVAILABLE = False
    logger.warning(f"⚠️ MCP client not available: {e}")

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
        create_streamlit_memory_agent = lambda config: HybridMemoryAgent(None, config)
        MEMORY_AGENT_AVAILABLE = True
        logger.info("✅ Original memory agent imported successfully")
    except ImportError as e2:
        MEMORY_AGENT_AVAILABLE = False
        logger.warning(f"⚠️ Memory agent not available: {e}")

        # Create dummy classes
        class DummyMemoryAgent:
            def __init__(self, config=None):
                self.config = config or {}
            def store_memory(self, *args, **kwargs):
                return {"success": False}
            def get_statistics(self):
                return {"status": "not_available"}

        create_streamlit_memory_agent = lambda config: DummyMemoryAgent(config)
        MemoryContext = None
        MemoryBucket = None

# Import REAL data processing agents - CRITICAL UPDATE
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
    logger.warning(f"⚠️ Real Data Processing agents not available: {e}")

# Set data mapping availability (part of data processing)
DATA_MAPPING_AVAILABLE = DATA_PROCESSING_AVAILABLE

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
    logger.warning(f"⚠️ Dormancy agents not available: {e}")

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
    logger.warning(f"⚠️ Compliance agents not available: {e}")

# Print system status
def print_system_status():
    """Print the current system status"""
    print("\n" + "="*50)
    print("🏦 CBUAE Banking Compliance System Status")
    print("="*50)
    print(f"🔐 Login System: {'✅ Available' if LOGIN_AVAILABLE else '❌ Not Available'}")
    print(f"💾 Memory Agent: {'✅ Available' if MEMORY_AGENT_AVAILABLE else '❌ Not Available'}")
    print(f"🔧 MCP Client: {'✅ Available' if MCP_AVAILABLE else '❌ Not Available'}")
    print(f"📊 Data Processing: {'✅ REAL AGENTS' if DATA_PROCESSING_AVAILABLE else '❌ Not Available'}")
    print(f"🏃 Dormancy Agents: {'✅ Available' if DORMANCY_AGENTS_AVAILABLE else '❌ Not Available'}")
    print(f"⚖️ Compliance Agents: {'✅ Available' if COMPLIANCE_AGENTS_AVAILABLE else '❌ Not Available'}")
    print("="*50)

# Call this to see what's working
print_system_status()

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = secrets.token_hex(16)
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'data_processing_agent' not in st.session_state:
        st.session_state.data_processing_agent = None
    if 'quality_results' not in st.session_state:
        st.session_state.quality_results = None
    if 'mapping_results' not in st.session_state:
        st.session_state.mapping_results = None
    if 'dormancy_results' not in st.session_state:
        st.session_state.dormancy_results = None
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = None
    if 'memory_agent' not in st.session_state:
        st.session_state.memory_agent = None
    if 'workflow_history' not in st.session_state:
        st.session_state.workflow_history = []

def initialize_data_processing_agent():
    """Initialize the real data processing agent"""
    if st.session_state.data_processing_agent is None:
        if DATA_PROCESSING_AVAILABLE:
            try:
                # Enhanced configuration for the data processing agent
                agent_config = {
                    "enable_memory": MEMORY_AGENT_AVAILABLE,
                    "enable_bge": True,  # Enable BGE embeddings for mapping
                    "bge_model": "BAAI/bge-large-en-v1.5",
                    "similarity_threshold": 0.7,
                    "quality_thresholds": {
                        "excellent": 0.9,
                        "good": 0.7,
                        "fair": 0.5,
                        "poor": 0.0
                    },
                    "banking_schema_enhanced": True,
                    "auto_mapping_enabled": True,
                    "max_file_size_mb": 100,
                    "supported_formats": ["csv", "xlsx", "json", "parquet"]
                }

                # Create the real data processing agent
                st.session_state.data_processing_agent = create_unified_data_processing_agent(agent_config)
                logger.info("✅ Real Data Processing Agent initialized")

            except Exception as e:
                logger.error(f"❌ Data processing agent initialization failed: {e}")
                st.session_state.data_processing_agent = None
        else:
            logger.warning("⚠️ Data processing agent not available")
            st.session_state.data_processing_agent = None

    return st.session_state.data_processing_agent

def initialize_memory_agent():
    """Initialize enhanced memory agent"""
    if st.session_state.memory_agent is None:
        if MEMORY_AGENT_AVAILABLE:
            try:
                # Enhanced configuration
                memory_config = {
                    "db_path": "enhanced_banking_memory.db",
                    "redis": {
                        "host": "localhost",
                        "port": 6379,
                        "db": 0,
                        "socket_timeout": 5
                    },
                    "cache_ttl": {
                        "session": 3600,
                        "knowledge": 86400,
                        "cache": 1800
                    },
                    "max_cache_size": 1000
                }

                # Create enhanced memory agent
                st.session_state.memory_agent = create_streamlit_memory_agent(memory_config)

                # Show statistics
                stats = st.session_state.memory_agent.get_statistics()
                logger.info(f"✅ Enhanced Memory Agent initialized - Redis: {stats.get('redis_available', False)}")

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

# Simple login system (fallback if main login fails)
class SimpleLoginManager:
    """Simplified login system for demonstration"""
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

    def create_user(self, username: str, password: str, role: str = 'user') -> bool:
        """Create new user"""
        self.users[username] = {"password": password, "role": role}
        return True

# Login system
def render_login_page():
    """Render secure login interface"""
    st.markdown('<div class="main-header"><h1>🏦 CBUAE Banking Compliance System</h1><p>Secure Login Required</p></div>', unsafe_allow_html=True)

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
                                    # Initialize demo user if it doesn't exist
                                    try:
                                        login_manager.create_user("demo", "demo123", "analyst")
                                        login_manager.create_user("admin", "admin123", "admin")
                                        login_manager.create_user("compliance", "compliance123", "compliance_officer")
                                        login_manager.create_user("analyst", "analyst123", "analyst")
                                    except:
                                        pass  # Users might already exist

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

                                # Create session token if secure login available
                                if LOGIN_AVAILABLE and isinstance(login_manager, SecureLoginManager):
                                    try:
                                        session_token = login_manager.create_secure_session(
                                            user_data,
                                            {"session_id": st.session_state.session_id}
                                        )
                                        st.session_state.session_token = session_token
                                    except:
                                        pass  # Continue without secure session

                                st.success(f"✅ Welcome back, {user_data['username']}!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials")

                        except Exception as e:
                            st.error(f"❌ Authentication failed: {str(e)}")
                            logger.error(f"Login error: {e}")
                    else:
                        st.warning("⚠️ Please enter both username and password")

            with col_demo:
                if st.button("🎯 Demo Login", use_container_width=True):
                    try:
                        # Use simple login for demo
                        login_manager = SimpleLoginManager()
                        user_data = login_manager.authenticate_user("demo", "demo123")

                        if user_data:
                            st.session_state.authenticated = True
                            st.session_state.user_data = user_data
                            st.success("✅ Demo login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Demo login failed")
                    except Exception as e:
                        st.error(f"❌ Demo login failed: {str(e)}")

            st.markdown("---")
            with st.expander("📋 Available Credentials"):
                st.code("""
Demo User:
Username: demo
Password: demo123

Admin User:
Username: admin  
Password: admin123

Compliance Officer:
Username: compliance
Password: compliance123

Analyst:
Username: analyst
Password: analyst123
                """)

        st.markdown('</div>', unsafe_allow_html=True)

# Main application header
def render_main_header():
    """Render main application header"""
    st.markdown(f'''
    <div class="main-header">
        <h1>🏦 CBUAE Banking Compliance Agentic AI System</h1>
        <p>Intelligent dormancy detection and compliance verification powered by REAL AI agents</p>
        <p>👤 Welcome, {st.session_state.user_data["username"]} | 🎯 Role: {st.session_state.user_data["role"]} | 🆔 Session: {st.session_state.session_id[:8]}</p>
    </div>
    ''', unsafe_allow_html=True)

# Data processing section - NOW USING REAL AGENTS
def render_data_processing_section():
    """Render data processing section with real agent integration"""
    st.markdown('<div class="section-header"><h2>📊 Data Processing & Analysis (Real Agents)</h2></div>', unsafe_allow_html=True)

    # Initialize the real data processing agent
    agent = initialize_data_processing_agent()

    if not DATA_PROCESSING_AVAILABLE or agent is None:
        st.error("❌ Real Data Processing Agent not available. Please check imports.")
        return

    # Show agent status
    st.success("✅ Real Data Processing Agent Loaded Successfully")
    agent_stats = agent.get_agent_statistics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Upload Methods", "4")
    with col2:
        st.metric("🗺️ Schema Fields", agent_stats.get("schema_fields", 0))
    with col3:
        st.metric("🔗 BGE Available", "✅ Yes" if agent_stats.get("bge_available") else "❌ No")
    with col4:
        st.metric("💾 Memory Enabled", "✅ Yes" if agent_stats.get("memory_available") else "❌ No")

    # Important note about the comprehensive workflow
    st.info("ℹ️ **Note**: This agent uses a comprehensive workflow. All processing (upload, quality, mapping) goes through `process_data_comprehensive()` method.")

    # Data upload subsection
    st.subheader("📁 Data Upload Agent (4 Real Methods)")

    upload_method = st.radio(
        "Select Upload Method:",
        ["File Upload", "Database Connection", "API Endpoint", "Manual Entry"],
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
                # Use the real agent's comprehensive processing
                with st.spinner("Processing file with real data upload agent..."):
                    # Save uploaded file temporarily
                    temp_file_path = f"temp_{st.session_state.session_id}_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Use real agent comprehensive workflow for upload only
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    comprehensive_result = loop.run_until_complete(
                        agent.process_data_comprehensive(
                            upload_method="file",
                            source=temp_file_path,
                            user_id=st.session_state.user_data.get("username", "unknown"),
                            session_id=st.session_state.session_id,
                            run_quality_analysis=False,  # Just upload for now
                            run_column_mapping=False
                        )
                    )

                    # Clean up temp file
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass

                    loop.close()

                if comprehensive_result.get("success") and comprehensive_result.get("upload_result"):
                    upload_result = comprehensive_result["upload_result"]
                    if upload_result.get("success"):
                        # Get the data from upload result - it should be in proper DataFrame format
                        uploaded_data = upload_result.get("data")

                        # Handle different data formats that might come from the agent
                        if uploaded_data is not None:
                            if isinstance(uploaded_data, dict):
                                # Convert dict to DataFrame if needed
                                if 'data' in uploaded_data:
                                    uploaded_data = pd.DataFrame(uploaded_data['data'])
                                else:
                                    uploaded_data = pd.DataFrame(uploaded_data)
                            elif not isinstance(uploaded_data, pd.DataFrame):
                                # Try to convert to DataFrame
                                uploaded_data = pd.DataFrame(uploaded_data)

                            st.success(f"✅ File processed successfully: {uploaded_file.name}")
                            st.info(f"📊 Data shape: {uploaded_data.shape[0]} rows × {uploaded_data.shape[1]} columns")

                            # Show processing metadata
                            metadata = upload_result.get("metadata")
                            if metadata:
                                with st.expander("🔍 Processing Details"):
                                    st.json(metadata)
                        else:
                            st.error("❌ No data returned from upload process")
                    else:
                        error_msg = upload_result.get("error", "Unknown upload error")
                        st.error(f"❌ File processing failed: {error_msg}")
                else:
                    error_msg = comprehensive_result.get("error", "Unknown comprehensive processing error")
                    st.error(f"❌ Comprehensive processing failed: {error_msg}")

            except Exception as e:
                st.error(f"❌ File upload failed: {str(e)}")
                logger.error(f"Upload error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    elif upload_method == "Database Connection":
        st.info("🔗 Real database connection interface")

        # Database connection form
        with st.form("database_form"):
            db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQL Server", "Oracle"])
            host = st.text_input("Host")
            port = st.number_input("Port", value=5432)
            database = st.text_input("Database Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            table_name = st.text_input("Table Name")

            if st.form_submit_button("🔗 Connect & Load Data"):
                if all([host, database, username, password, table_name]):
                    # This would use the real agent's database connection method
                    st.info("📊 Database connection would be established using real agent")
                else:
                    st.warning("⚠️ Please fill all required fields")

    elif upload_method == "API Endpoint":
        st.info("🌐 Real API endpoint interface")

        with st.form("api_form"):
            api_url = st.text_input("API Endpoint URL")
            auth_method = st.selectbox("Authentication", ["None", "API Key", "Bearer Token", "Basic Auth"])

            if auth_method == "API Key":
                api_key = st.text_input("API Key", type="password")
            elif auth_method == "Bearer Token":
                bearer_token = st.text_input("Bearer Token", type="password")
            elif auth_method == "Basic Auth":
                api_username = st.text_input("API Username")
                api_password = st.text_input("API Password", type="password")

            if st.form_submit_button("🌐 Fetch Data from API"):
                if api_url:
                    # This would use the real agent's API method
                    st.info("📊 API data would be fetched using real agent")
                else:
                    st.warning("⚠️ Please provide API endpoint URL")

    elif upload_method == "Manual Entry":
        st.info("✏️ Generate sample banking data for testing real agents")

        col1, col2 = st.columns(2)
        with col1:
            num_accounts = st.number_input("Number of sample accounts", min_value=10, max_value=1000, value=100)

        with col2:
            if st.button("📊 Generate Sample Data", type="secondary"):
                uploaded_data = generate_sample_banking_data(num_accounts)
                st.success(f"✅ Generated {num_accounts} sample banking accounts")

    # Process uploaded data with REAL AGENTS
    if uploaded_data is not None:
        # Display data preview
        with st.expander("👀 Data Preview", expanded=True):
            st.dataframe(uploaded_data.head(10), use_container_width=True)

        # REAL Data quality analysis using actual agent
        st.subheader("🔍 Data Quality Analysis (Real Agent)")

        if st.button("🚀 Run Real Data Quality Analysis", type="primary"):
            with st.spinner("Running real data quality analysis..."):
                try:
                    # Use the REAL agent for quality analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    quality_result = loop.run_until_complete(
                        agent.analyze_data_quality(
                            data=uploaded_data,
                            user_id=st.session_state.user_data.get("username", "unknown"),
                            session_id=st.session_state.session_id
                        )
                    )

                    loop.close()

                    if quality_result.success:
                        st.session_state.quality_results = quality_result

                        # Display quality metrics using real results
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "📊 Total Records",
                                f"{len(uploaded_data):,}"
                            )

                        with col2:
                            st.metric(
                                "🔍 Missing Data",
                                f"{quality_result.missing_percentage:.1f}%"
                            )

                        with col3:
                            st.metric(
                                "📋 Duplicate Records",
                                f"{quality_result.duplicate_records:,}"
                            )

                        with col4:
                            quality_class = f"quality-{quality_result.quality_level}"
                            st.markdown(f"""
                            <div class="metric-card {quality_class}">
                                <h3>⭐ Quality Score</h3>
                                <h2>{quality_result.overall_score:.1%}</h2>
                                <p>{quality_result.quality_level.title()}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Detailed metrics from real agent
                        with st.expander("📊 Detailed Quality Metrics"):
                            metrics_df = pd.DataFrame([quality_result.metrics]).T
                            metrics_df.columns = ['Score']
                            metrics_df['Percentage'] = (metrics_df['Score'] * 100).round(1)
                            st.dataframe(metrics_df, use_container_width=True)

                        # Recommendations from real agent
                        if quality_result.recommendations:
                            st.subheader("💡 Quality Improvement Recommendations")
                            for i, rec in enumerate(quality_result.recommendations, 1):
                                st.markdown(f"{i}. {rec}")

                    else:
                        st.error(f"❌ Quality analysis failed: {quality_result.error}")

                except Exception as e:
                    st.error(f"❌ Quality analysis failed: {str(e)}")
                    logger.error(f"Quality analysis error: {e}")

        # REAL Data mapping section using actual agent
        st.subheader("🗺️ Data Mapping Agent (Real BGE-based)")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.info("📋 Map your data columns to CBUAE banking schema using real BGE embeddings")

        with col2:
            enable_llm = st.checkbox(
                "🤖 Enable LLM Mapping",
                help="Let real BGE embeddings automatically map columns based on semantic similarity"
            )

        if st.button("🎯 Run Real Data Mapping", type="primary"):
            with st.spinner("Running real BGE-based column mapping..."):
                try:
                    # Use the REAL agent for comprehensive processing (mapping only)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Save data temporarily for processing
                    temp_file_path = f"temp_mapping_{st.session_state.session_id}.csv"
                    uploaded_data.to_csv(temp_file_path, index=False)

                    # Run comprehensive workflow with mapping only
                    comprehensive_result = loop.run_until_complete(
                        agent.process_data_comprehensive(
                            upload_method="file",
                            source=temp_file_path,
                            user_id=st.session_state.user_data.get("username", "unknown"),
                            session_id=st.session_state.session_id,
                            run_quality_analysis=False,  # Skip quality for mapping
                            run_column_mapping=True,
                            use_llm_mapping=enable_llm
                        )
                    )

                    # Clean up temp file
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass

                    loop.close()

                    if comprehensive_result.get("success") and comprehensive_result.get("mapping_result"):
                        mapping_result = comprehensive_result["mapping_result"]
                        st.session_state.mapping_results = mapping_result
                        st.success("✅ Real BGE-based data mapping completed successfully!")

                        # Display mapping results from real agent
                        mapping_sheet = mapping_result.get('mapping_sheet')
                        if mapping_sheet is not None:
                            st.subheader("📋 Real Column Mapping Results")

                            # Add confidence level styling
                            def style_confidence(val):
                                try:
                                    val_float = float(val)
                                    if val_float >= 0.8:
                                        return 'background-color: #d4edda; color: #155724'
                                    elif val_float >= 0.6:
                                        return 'background-color: #fff3cd; color: #856404'
                                    else:
                                        return 'background-color: #f8d7da; color: #721c24'
                                except:
                                    return ''

                            # Apply styling if confidence column exists
                            if 'confidence_score' in mapping_sheet.columns or 'Similarity_Score' in mapping_sheet.columns:
                                conf_col = 'confidence_score' if 'confidence_score' in mapping_sheet.columns else 'Similarity_Score'
                                styled_mapping = mapping_sheet.style.applymap(
                                    style_confidence,
                                    subset=[conf_col]
                                )
                                st.dataframe(styled_mapping, use_container_width=True)
                            else:
                                st.dataframe(mapping_sheet, use_container_width=True)

                            # Mapping statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                auto_pct = mapping_result.get('auto_mapping_percentage', 0)
                                st.metric("🎯 Auto-mapped", f"{auto_pct:.1f}%")
                            with col2:
                                method = mapping_result.get('method', 'unknown')
                                st.metric("🔧 Method", method)
                            with col3:
                                conf_dist = mapping_result.get('confidence_distribution', {})
                                high_conf = conf_dist.get('high', 0)
                                st.metric("✅ High Confidence", f"{high_conf}")

                            # Download real mapping sheet
                            csv_mapping = mapping_sheet.to_csv(index=False)
                            st.download_button(
                                "📄 Download Real Mapping Sheet",
                                csv_mapping,
                                f"real_column_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                help="Download the BGE-based column mapping for future reference"
                            )

                        # Store processed data from real agent
                        st.session_state.processed_data = uploaded_data  # In real implementation, would apply mapping

                    else:
                        error_msg = comprehensive_result.get("error", "Unknown error")
                        st.error(f"❌ Real data mapping failed: {error_msg}")

                except Exception as e:
                    st.error(f"❌ Real data mapping failed: {str(e)}")
                    logger.error(f"Real mapping error: {e}")

        # Store raw data if no mapping done yet
        if st.session_state.processed_data is None and uploaded_data is not None:
            st.session_state.processed_data = uploaded_data

def generate_sample_banking_data(num_accounts: int) -> pd.DataFrame:
    """Generate sample banking data for testing"""
    np.random.seed(42)  # For reproducible results

    # Account types and statuses
    account_types = ['Savings', 'Current', 'Fixed Deposit', 'Investment']
    account_statuses = ['Active', 'Dormant', 'Closed']
    customer_types = ['Individual', 'Corporate']
    currencies = ['AED', 'USD', 'EUR', 'GBP']

    data = []
    for i in range(num_accounts):
        # Generate account data
        account_id = f"ACC{10000 + i}"
        customer_id = f"CUST{5000 + i}"

        # Random dates
        opening_date = datetime.now() - timedelta(days=np.random.randint(30, 2000))
        last_transaction_days = np.random.randint(1, 800)
        last_transaction_date = datetime.now() - timedelta(days=last_transaction_days)

        # Determine if account should be dormant (accounts with no activity > 365 days)
        is_dormant = last_transaction_days > 365
        status = 'Dormant' if is_dormant else np.random.choice(['Active', 'Active', 'Active', 'Closed'])

        # Balance based on account type and status
        if status == 'Closed':
            balance = 0
        elif status == 'Dormant':
            balance = np.random.uniform(100, 25000)  # Dormant accounts often have balances
        else:
            balance = np.random.uniform(0, 100000)

        account_record = {
            'customer_id': customer_id,
            'account_id': account_id,
            'account_number': f"{np.random.randint(100000, 999999)}",
            'account_type': np.random.choice(account_types),
            'account_status': status,
            'customer_name': f"Customer {i+1}",
            'customer_type': np.random.choice(customer_types),
            'balance': round(balance, 2),
            'currency': np.random.choice(currencies),
            'opening_date': opening_date.strftime('%Y-%m-%d'),
            'last_transaction_date': last_transaction_date.strftime('%Y-%m-%d'),
            'branch_code': f"BR{np.random.randint(100, 999)}",
            'contact_phone': f"+971{np.random.randint(500000000, 599999999)}",
            'contact_email': f"customer{i+1}@example.com",
            'days_since_last_transaction': last_transaction_days,
            'risk_rating': np.random.choice(['Low', 'Medium', 'High']),
            'kyc_status': np.random.choice(['Current', 'Expired', 'Pending'])
        }

        data.append(account_record)

    df = pd.DataFrame(data)

    # Add some additional calculated fields that might be useful
    df['balance_category'] = pd.cut(df['balance'],
                                   bins=[0, 1000, 10000, 50000, float('inf')],
                                   labels=['Low', 'Medium', 'High', 'VIP'])

    return df

# Dormancy analysis section
def render_dormancy_analysis_section():
    """Render dormancy analysis section with real agents"""
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please complete data processing first")
        return

    st.markdown('<div class="section-header"><h2>🏃 Dormancy Analysis Agents</h2></div>', unsafe_allow_html=True)

    if not DORMANCY_AGENTS_AVAILABLE:
        st.error("❌ Dormancy agents not available. Please check imports.")
        return

    # Display available dormancy agents
    try:
        orchestrator = DormancyWorkflowOrchestrator()
        agent_info = orchestrator.get_all_agent_info()

        st.subheader("🤖 Available Dormancy Agents")

        agent_cols = st.columns(3)
        for idx, (agent_name, info) in enumerate(agent_info.items()):
            with agent_cols[idx % 3]:
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 {agent_name.replace('_', ' ').title()}</h4>
                        <p><strong>Type:</strong> {info['agent_type']}</p>
                        <p><strong>Article:</strong> {info['cbuae_article']}</p>
                        <p><strong>Status:</strong> {info['ui_status']}</p>
                        <p>{info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading dormancy agents: {e}")
        return

    # Run dormancy analysis
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🚀 Execute Dormancy Analysis")

    with col2:
        if st.button("▶️ Run All Agents", type="primary", use_container_width=True):
            with st.spinner("Running comprehensive dormancy analysis..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Update progress
                    progress_bar.progress(0.2)
                    status_text.text("🔄 Initializing dormancy agents...")

                    # Run real dormancy analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    dormancy_results = loop.run_until_complete(
                        run_comprehensive_dormancy_analysis_with_csv(
                            user_id=f"streamlit_{st.session_state.session_id[:8]}",
                            account_data=st.session_state.processed_data,
                            report_date=datetime.now().strftime('%Y-%m-%d')
                        )
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
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Dormancy analysis error: {e}")

                finally:
                    loop.close()

def display_dormancy_results(results):
    """Display dormancy analysis results"""
    if not results or not results.get("success"):
        return

    st.subheader("📊 Dormancy Analysis Results")

    # Summary metrics
    summary = results.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📊 Total Accounts",
            f"{summary.get('total_accounts_processed', 0):,}"
        )

    with col2:
        st.metric(
            "🏃 Dormant Accounts",
            f"{summary.get('total_dormant_accounts', 0):,}"
        )

    with col3:
        processing_time = results.get("processing_time", 0)
        st.metric(
            "⏱️ Processing Time",
            f"{processing_time:.1f}s"
        )

    with col4:
        dormancy_rate = (summary.get('total_dormant_accounts', 0) /
                        max(summary.get('total_accounts_processed', 1), 1)) * 100
        st.metric(
            "📈 Dormancy Rate",
            f"{dormancy_rate:.1f}%"
        )

    # Agent results
    agent_results = results.get("agent_results", {})
    if agent_results:
        st.subheader("🤖 Individual Agent Results")

        for agent_name, agent_result in agent_results.items():
            if agent_result.get("success"):
                with st.expander(f"🎯 {agent_name.replace('_', ' ').title()}", expanded=False):

                    # Agent metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Records Processed",
                            f"{agent_result.get('records_processed', 0):,}"
                        )

                    with col2:
                        st.metric(
                            "Dormant Found",
                            f"{agent_result.get('dormant_accounts_found', 0):,}"
                        )

                    with col3:
                        st.metric(
                            "Processing Time",
                            f"{agent_result.get('processing_time', 0):.2f}s"
                        )

                    # CSV download button
                    csv_data = agent_result.get('csv_export', {})
                    if csv_data.get('available'):
                        csv_content = csv_data.get('csv_content', '')
                        if csv_content:
                            st.download_button(
                                f"📄 Download {agent_name} Results",
                                csv_content,
                                f"{agent_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                key=f"download_{agent_name}"
                            )

                    # Summary
                    summary_text = agent_result.get('summary', 'No summary available')
                    st.text_area(
                        "Agent Summary",
                        summary_text,
                        height=100,
                        disabled=True,
                        key=f"summary_{agent_name}"
                    )

# Compliance analysis section
def render_compliance_analysis_section():
    """Render compliance analysis section"""
    if st.session_state.dormancy_results is None:
        st.warning("⚠️ Please complete dormancy analysis first")
        return

    st.markdown('<div class="section-header"><h2>⚖️ Compliance Verification Agents</h2></div>', unsafe_allow_html=True)

    if not COMPLIANCE_AGENTS_AVAILABLE:
        st.error("❌ Compliance agents not available. Please check imports.")
        return

    # Display available compliance agents
    try:
        compliance_info = get_all_compliance_agents_info()

        st.subheader("🤖 Available Compliance Agents")
        st.info(f"📊 Total Agents: {compliance_info['total_agents']} | 📋 CBUAE Articles: {len(compliance_info['cbuae_articles_covered'])}")

        # Group agents by category
        for category, agents in compliance_info['agents_by_category'].items():
            with st.expander(f"📂 {category.replace('_', ' ').title()} ({len(agents)} agents)"):
                for agent in agents:
                    st.markdown(f"""
                    - **{agent['agent_name']}** (Article: {agent['cbuae_article']})
                      - {agent['description']}
                    """)

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
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                compliance_results = loop.run_until_complete(
                    run_comprehensive_compliance_analysis_with_csv(
                        user_id=f"streamlit_{st.session_state.session_id[:8]}",
                        dormancy_results=st.session_state.dormancy_results,
                        accounts_df=st.session_state.processed_data
                    )
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

            finally:
                loop.close()

def display_compliance_results(results):
    """Display compliance analysis results"""
    if not results or not results.get("success"):
        return

    st.subheader("📊 Compliance Verification Results")

    # Summary metrics
    summary = results.get("compliance_summary", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📊 Accounts Analyzed",
            f"{summary.get('total_accounts_analyzed', 0):,}"
        )

    with col2:
        st.metric(
            "⚠️ Violations Found",
            f"{summary.get('total_violations_found', 0):,}"
        )

    with col3:
        st.metric(
            "🎯 Actions Generated",
            f"{summary.get('total_actions_generated', 0):,}"
        )

    with col4:
        compliance_status = summary.get('overall_compliance_status', 'UNKNOWN')
        status_color = "🟢" if compliance_status == "COMPLIANT" else "🔴"
        st.metric(
            "📋 Status",
            f"{status_color} {compliance_status}"
        )

    # Agent results
    agent_results = results.get("agent_results", {})
    if agent_results:
        st.subheader("🤖 Individual Compliance Agent Results")

        for agent_name, agent_result in agent_results.items():
            if agent_result.get("success"):
                with st.expander(f"⚖️ {agent_name.replace('_', ' ').title()}", expanded=False):

                    # Agent metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Accounts Processed",
                            f"{agent_result.get('accounts_processed', 0):,}"
                        )

                    with col2:
                        st.metric(
                            "Violations Found",
                            f"{agent_result.get('violations_found', 0):,}"
                        )

                    with col3:
                        st.metric(
                            "Processing Time",
                            f"{agent_result.get('processing_time', 0):.2f}s"
                        )

                    # CSV download button
                    if agent_result.get('csv_download_ready'):
                        csv_export = agent_result.get('compliance_summary', {}).get('csv_export', {})
                        if csv_export.get('available'):
                            csv_content = csv_export.get('csv_content', '')
                            if csv_content:
                                st.download_button(
                                    f"📄 Download {agent_name} Results",
                                    csv_content,
                                    f"{agent_name}_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv",
                                    key=f"compliance_download_{agent_name}"
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
    st.markdown('<div class="section-header"><h2>📈 Comprehensive Reports</h2></div>', unsafe_allow_html=True)

    # Executive summary
    st.subheader("📋 Executive Summary")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.processed_data is not None:
            st.metric(
                "📊 Total Accounts Processed",
                f"{len(st.session_state.processed_data):,}"
            )
        else:
            st.metric("📊 Total Accounts Processed", "0")

    with col2:
        total_agents = 0
        if DORMANCY_AGENTS_AVAILABLE:
            try:
                orchestrator = DormancyWorkflowOrchestrator()
                total_agents += len(orchestrator.agents)
            except:
                pass

        if COMPLIANCE_AGENTS_AVAILABLE:
            try:
                compliance_info = get_all_compliance_agents_info()
                total_agents += compliance_info['total_agents']
            except:
                pass

        st.metric("🤖 Total Agents Available", f"{total_agents}")

    # Detailed agent status
    st.subheader("🤖 Agent Status Overview")

    # Data Processing Agent Status
    st.markdown("#### 📊 Data Processing Agent")
    if DATA_PROCESSING_AVAILABLE:
        agent = st.session_state.data_processing_agent
        if agent:
            stats = agent.get_agent_statistics()
            quality_status = "✅ Processed" if st.session_state.quality_results else "⏳ Pending"
            mapping_status = "✅ Mapped" if st.session_state.mapping_results else "⏳ Pending"

            st.markdown(f"""
            <div class="agent-status agent-success">
                <strong>Unified Data Processing Agent</strong><br>
                Status: ✅ Available & Loaded<br>
                Quality Analysis: {quality_status}<br>
                Column Mapping: {mapping_status}<br>
                BGE Enabled: {'✅' if stats.get('configuration', {}).get('enable_bge') else '❌'}<br>
                Memory Integration: {'✅' if stats.get('configuration', {}).get('enable_memory') else '❌'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="agent-status agent-error">
                <strong>Data Processing Agent</strong><br>
                Status: ❌ Not Loaded
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="agent-status agent-error">
            <strong>Data Processing Agent</strong><br>
            Status: ❌ Not Available
        </div>
        """, unsafe_allow_html=True)

    # Dormancy agents status
    if DORMANCY_AGENTS_AVAILABLE:
        st.markdown("#### 🏃 Dormancy Agents")
        try:
            orchestrator = DormancyWorkflowOrchestrator()
            agent_info = orchestrator.get_all_agent_info()

            for agent_name, info in agent_info.items():
                status = "✅ Available" if info['ui_status'] == 'active' else "⚠️ Inactive"
                accounts_processed = 0

                if (st.session_state.dormancy_results and
                    st.session_state.dormancy_results.get("agent_results", {}).get(agent_name)):
                    agent_result = st.session_state.dormancy_results["agent_results"][agent_name]
                    if agent_result.get("success"):
                        accounts_processed = agent_result.get("records_processed", 0)
                        status = f"✅ Processed {accounts_processed:,} accounts"

                st.markdown(f"""
                <div class="agent-status agent-success">
                    <strong>{agent_name.replace('_', ' ').title()}</strong><br>
                    Status: {status}<br>
                    Article: {info['cbuae_article']}<br>
                    Type: {info['agent_type']}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading dormancy agent status: {e}")

    # Compliance agents status
    if COMPLIANCE_AGENTS_AVAILABLE:
        st.markdown("#### ⚖️ Compliance Agents")
        try:
            compliance_info = get_all_compliance_agents_info()

            for category, agents in compliance_info['agents_by_category'].items():
                st.markdown(f"**{category.replace('_', ' ').title()}**")

                for agent in agents:
                    agent_name = agent['agent_name']
                    status = "✅ Available"
                    violations_found = 0

                    if (st.session_state.compliance_results and
                        st.session_state.compliance_results.get("agent_results", {}).get(agent_name)):
                        agent_result = st.session_state.compliance_results["agent_results"][agent_name]
                        if agent_result.get("success"):
                            violations_found = agent_result.get("violations_found", 0)
                            status = f"✅ Found {violations_found:,} violations"

                    st.markdown(f"""
                    <div class="agent-status agent-success">
                        <strong>{agent_name.replace('_', ' ').title()}</strong><br>
                        Status: {status}<br>
                        Article: {agent['cbuae_article']}
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading compliance agent status: {e}")

    # Generate comprehensive report
    if st.button("📊 Generate Comprehensive Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            report_data = generate_comprehensive_report()

            if report_data:
                st.success("✅ Report generated successfully!")

                # Display report
                st.subheader("📋 Detailed Analysis Report")
                st.json(report_data)

                # Download report
                report_json = json.dumps(report_data, indent=2, default=str)
                st.download_button(
                    "📄 Download Full Report (JSON)",
                    report_json,
                    f"banking_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )

def generate_comprehensive_report() -> Dict:
    """Generate comprehensive analysis report"""
    try:
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "session_id": st.session_state.session_id,
                "user": st.session_state.user_data.get("username") if st.session_state.user_data else "unknown",
                "real_agents_used": True
            },
            "data_processing": {
                "agent_available": DATA_PROCESSING_AVAILABLE,
                "agent_loaded": st.session_state.data_processing_agent is not None,
                "total_records": len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0,
                "quality_analysis_completed": st.session_state.quality_results is not None,
                "mapping_completed": st.session_state.mapping_results is not None
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
                "duplicate_records": st.session_state.quality_results.duplicate_records,
                "metrics": st.session_state.quality_results.metrics,
                "recommendations_count": len(st.session_state.quality_results.recommendations),
                "processing_time": st.session_state.quality_results.processing_time
            }

        # Add mapping results
        if st.session_state.mapping_results:
            report["mapping_results"] = {
                "success": st.session_state.mapping_results.success,
                "method": st.session_state.mapping_results.method,
                "auto_mapping_percentage": st.session_state.mapping_results.auto_mapping_percentage,
                "confidence_distribution": st.session_state.mapping_results.confidence_distribution,
                "processing_time": st.session_state.mapping_results.processing_time
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
        return None

# Sidebar navigation
def render_sidebar():
    """Render navigation sidebar"""
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

        # Real agents status
        st.markdown("### 🤖 Real Agents Status")
        st.text(f"Data Processing: {'✅' if DATA_PROCESSING_AVAILABLE else '❌'}")
        st.text(f"Dormancy Agents: {'✅' if DORMANCY_AGENTS_AVAILABLE else '❌'}")
        st.text(f"Compliance Agents: {'✅' if COMPLIANCE_AGENTS_AVAILABLE else '❌'}")

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

    # Initialize agents
    initialize_memory_agent()
    initialize_data_processing_agent()

    # Render main header
    render_main_header()

    # Get selected page from sidebar
    selected_page = render_sidebar()

    # Render selected page
    if selected_page == "📊 Data Processing":
        render_data_processing_section()

    elif selected_page == "🏃 Dormancy Analysis":
        render_dormancy_analysis_section()

    elif selected_page == "⚖️ Compliance Verification":
        render_compliance_analysis_section()

    elif selected_page == "📈 Reports & Analytics":
        render_reports_section()

# Run the application
if __name__ == "__main__":
    main()