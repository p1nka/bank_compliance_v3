import streamlit as st
import pandas as pd
import asyncio
import json
import io
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import real agent modules
from agents.compliance_verification_agent import *

@dataclass
class AgentMemory:
    """Shared memory system for agents"""
    session_id: str
    data_processing_results: Optional[Dict[str, Any]] = None
    dormancy_analysis_results: Optional[Dict[str, Any]] = None
    compliance_results: Optional[Dict[str, Any]] = None
    processed_data: Optional[pd.DataFrame] = None
    dormant_accounts: Optional[List[Dict]] = None
    mapping_results: Optional[Dict[str, str]] = None
    created_at: datetime = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.last_updated = datetime.now()

    def update_memory(self, agent_type: str, results: Any):
        """Update memory with agent results"""
        self.last_updated = datetime.now()

        if agent_type == "data_processing":
            self.data_processing_results = results
        elif agent_type == "dormancy_analysis":
            self.dormancy_analysis_results = results
            if isinstance(results, dict) and 'dormant_accounts' in results:
                self.dormant_accounts = results['dormant_accounts']
        elif agent_type == "compliance":
            self.compliance_results = results
        elif agent_type == "processed_data":
            self.processed_data = results
        elif agent_type == "mapping":
            self.mapping_results = results

@dataclass
class AgentResult:
    agent_name: str
    success: bool
    records_processed: int
    results: Dict[str, Any]
    processing_time: float
    summary: str
    agent_type: str

@dataclass
class WorkflowState:
    session_id: str
    user_id: str
    current_stage: str
    memory: AgentMemory
    status: str = "initialized"

    def get_memory(self, session_id: str) -> AgentMemory:
        """Get or create agent memory for session"""
        if session_id not in self.agent_memory:
            self.agent_memory[session_id] = AgentMemory(session_id=session_id)
        return self.agent_memory[session_id]

    def process_data(self, data: pd.DataFrame, session_id: str) -> AgentResult:
        """Process raw data and update memory"""
        memory = self.get_memory(session_id)
        start_time = datetime.now()

        try:
            # Basic data processing
            processed_data = data.copy()

            # Ensure required columns exist with reasonable defaults
            required_columns = {
                'account_id': lambda: processed_data.index.astype(str),
                'balance_current': lambda: processed_data.get('balance', 0),
                'last_transaction_date': lambda: processed_data.get('last_transaction_date', '2020-01-01'),
                'contact_attempts_made': lambda: 0,
                'last_contact_date': lambda: None,
                'currency': lambda: 'AED',
                'status': lambda: 'ACTIVE',
                'customer_address_known': lambda: True,
                'transferred_to_cbuae': lambda: False,
                'statement_generation': lambda: True,
                'dormancy_trigger_date': lambda: '2020-01-01',
                'first_contact_date': lambda: None,
                'current_stage': lambda: 'INITIAL',
                'available_documents': lambda: ['account_opening_form']
            }

            for col, default_func in required_columns.items():
                if col not in processed_data.columns:
                    processed_data[col] = default_func()

            processing_time = (datetime.now() - start_time).total_seconds()

            # Update memory
            memory.update_memory("processed_data", processed_data)

            results = {
                'processed_records': len(processed_data),
                'columns_processed': list(processed_data.columns),
                'data_shape': processed_data.shape
            }

            memory.update_memory("data_processing", results)

            return AgentResult(
                agent_name="Data Processing Agent",
                success=True,
                records_processed=len(processed_data),
                results=results,
                processing_time=processing_time,
                summary=f"Successfully processed {len(processed_data)} records",
                agent_type="data_processing"
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_name="Data Processing Agent",
                success=False,
                records_processed=0,
                results={'error': str(e)},
                processing_time=processing_time,
                summary=f"Failed to process data: {str(e)}",
                agent_type="data_processing"
            )

    def analyze_dormancy(self, session_id: str) -> AgentResult:
        """Analyze dormancy using processed data from memory"""
        memory = self.get_memory(session_id)
        start_time = datetime.now()

        if memory.processed_data is None:
            return AgentResult(
                agent_name="Dormancy Analysis Agent",
                success=False,
                records_processed=0,
                results={'error': 'No processed data available'},
                processing_time=0,
                summary="No processed data available for dormancy analysis",
                agent_type="dormancy_analysis"
            )

        try:
            data = memory.processed_data
            dormant_accounts = []

            # Real dormancy analysis logic
            for _, row in data.iterrows():
                try:
                    # Convert last transaction date
                    if pd.notna(row.get('last_transaction_date')):
                        if isinstance(row['last_transaction_date'], str):
                            last_transaction = pd.to_datetime(row['last_transaction_date'])
                        else:
                            last_transaction = row['last_transaction_date']
                    else:
                        last_transaction = pd.to_datetime('2020-01-01')

                    days_inactive = (datetime.now() - last_transaction).days

                    # Determine if account is dormant (over 3 years = 1095 days)
                    if days_inactive >= 1095:
                        account_data = {
                            'account_id': str(row.get('account_id', f'ACC_{len(dormant_accounts)+1:06d}')),
                            'balance_current': float(row.get('balance_current', 0)),
                            'last_transaction_date': last_transaction.strftime('%Y-%m-%d'),
                            'contact_attempts_made': int(row.get('contact_attempts_made', 0)),
                            'last_contact_date': row.get('last_contact_date'),
                            'currency': str(row.get('currency', 'AED')),
                            'status': str(row.get('status', 'DORMANT')),
                            'customer_address_known': bool(row.get('customer_address_known', True)),
                            'transferred_to_cbuae': bool(row.get('transferred_to_cbuae', False)),
                            'statement_generation': bool(row.get('statement_generation', True)),
                            'dormancy_trigger_date': str(row.get('dormancy_trigger_date', '2020-01-01')),
                            'first_contact_date': row.get('first_contact_date'),
                            'current_stage': str(row.get('current_stage', 'DORMANT')),
                            'available_documents': row.get('available_documents', ['account_opening_form'])
                        }
                        dormant_accounts.append(account_data)

                except Exception as e:
                    # Skip problematic rows but continue processing
                    continue

            processing_time = (datetime.now() - start_time).total_seconds()

            results = {
                'total_accounts_analyzed': len(data),
                'dormant_accounts_found': len(dormant_accounts),
                'dormant_accounts': dormant_accounts,
                'dormancy_rate': (len(dormant_accounts) / len(data) * 100) if len(data) > 0 else 0,
                'total_dormant_balance': sum(acc['balance_current'] for acc in dormant_accounts)
            }

            # Update memory with results
            memory.update_memory("dormancy_analysis", results)

            return AgentResult(
                agent_name="Dormancy Analysis Agent",
                success=True,
                records_processed=len(data),
                results=results,
                processing_time=processing_time,
                summary=f"Found {len(dormant_accounts)} dormant accounts out of {len(data)} analyzed",
                agent_type="dormancy_analysis"
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_name="Dormancy Analysis Agent",
                success=False,
                records_processed=0,
                results={'error': str(e)},
                processing_time=processing_time,
                summary=f"Failed to analyze dormancy: {str(e)}",
                agent_type="dormancy_analysis"
            )

    def run_compliance_verification(self, session_id: str) -> Dict[str, AgentResult]:
        """Run compliance verification using dormant accounts from memory"""
        memory = self.get_memory(session_id)
        results = {}

        if memory.dormant_accounts is None or len(memory.dormant_accounts) == 0:
            # Return empty results if no dormant accounts
            for agent_name in self.compliance_agents.keys():
                results[agent_name] = AgentResult(
                    agent_name=self.compliance_agents[agent_name]['description'],
                    success=False,
                    records_processed=0,
                    results={'message': 'No dormant accounts available for analysis'},
                    processing_time=0,
                    summary="No dormant accounts to analyze",
                    agent_type="compliance"
                )
            return results

        dormant_accounts = memory.dormant_accounts

        # Run each compliance agent with appropriate data
        for agent_name, agent_config in self.compliance_agents.items():
            start_time = datetime.now()

            try:
                agent_function = agent_config['function']
                input_type = agent_config['input_type']

                # Prepare input data based on agent requirements
                if input_type == 'dormant_accounts':
                    input_data = dormant_accounts
                elif input_type == 'active_accounts':
                    # Convert some dormant accounts to active for testing unflagged detection
                    input_data = [
                        {**acc, 'status': 'ACTIVE'} for acc in dormant_accounts[:min(5, len(dormant_accounts))]
                    ]
                elif input_type == 'banking_data':
                    input_data = {
                        'records': dormant_accounts
                    }
                elif input_type == 'safe_deposit_boxes':
                    # Create SDB data from dormant accounts if needed
                    input_data = [
                        {
                            'sdb_number': f"SDB{i+1:03d}",
                            'customer_id': acc['account_id'],
                            'outstanding_charges': 1500.0,
                            'last_payment_date': '2020-12-15',
                            'contact_attempts_complete': True
                        }
                        for i, acc in enumerate(dormant_accounts[:3])  # Limit to 3 for demo
                    ]
                elif input_type == 'payment_instruments':
                    # Create payment instrument data from dormant accounts if needed
                    input_data = [
                        {
                            'instrument_number': f"CHQ{i+1:06d}",
                            'type': 'CHEQUE',
                            'amount': acc['balance_current'],
                            'beneficiary': f"Customer_{acc['account_id']}",
                            'customer_id': acc['account_id'],
                            'issue_date': '2023-01-15',
                            'claimed': False,
                            'contact_attempts': acc['contact_attempts_made']
                        }
                        for i, acc in enumerate(dormant_accounts[:5])  # Limit to 5 for demo
                    ]
                elif input_type == 'customer_claims':
                    # Create claims data from dormant accounts if needed
                    input_data = [
                        {
                            'claim_id': f"CLM{i+1:06d}",
                            'customer_id': acc['account_id'],
                            'account_id': acc['account_id'],
                            'type': 'REACTIVATION',
                            'submission_date': '2024-10-01',
                            'status': 'PENDING',
                            'assigned_to': 'Officer001'
                        }
                        for i, acc in enumerate(dormant_accounts[:3])  # Limit to 3 for demo
                    ]
                elif input_type == 'annual_data':
                    # Create annual data summary
                    input_data = {
                        'bank_license': 'UAE001',
                        'bank_name': 'Sample Bank',
                        'dormant_start_year': len(dormant_accounts) + 10,
                        'new_dormant': len(dormant_accounts),
                        'reactivated': 5,
                        'cbuae_transfers': 2,
                        'dormant_end_year': len(dormant_accounts) + 3,
                        'total_balances': sum(acc['balance_current'] for acc in dormant_accounts),
                        'cbuae_transfer_amount': 50000.0,
                        'customer_returns': 10000.0,
                        'fx_conversions': 3,
                        'contact_completion_rate': 85.5,
                        'avg_processing_days': 45,
                        'prepared_by': 'Compliance Officer',
                        'approved_by': 'Chief Compliance Officer'
                    }
                elif input_type == 'account_records':
                    # Create account records data
                    input_data = [
                        {
                            'account_id': acc['account_id'],
                            'closure_date': '2020-01-01',
                            'available_records': ['account_opening_form', 'kyc_documents'],
                            'storage_location': 'Archive_A',
                            'digital_backup': True
                        }
                        for acc in dormant_accounts[:5]  # Limit to 5 for demo
                    ]
                elif input_type == 'compliance_processes':
                    # Create compliance process data
                    input_data = [
                        {
                            'process_id': f"PROC{i+1:06d}",
                            'type': 'DORMANCY_CONTACT',
                            'account_id': acc['account_id'],
                            'deadline': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                            'assigned_officer': 'Officer001',
                            'required_action': 'CONTACT_CUSTOMER'
                        }
                        for i, acc in enumerate(dormant_accounts[:5])  # Limit to 5 for demo
                    ]
                elif input_type == 'system_activities':
                    # Create system activity data
                    input_data = [
                        {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'DORMANCY_FLAG',
                            'action': 'FLAG_ACCOUNT_DORMANT',
                            'entity_id': acc['account_id'],
                            'user_id': 'SYSTEM',
                            'session_id': session_id
                        }
                        for acc in dormant_accounts[:5]  # Limit to 5 for demo
                    ]
                elif input_type == 'flagging_activities':
                    # Create flagging activity data
                    input_data = [
                        {
                            'account_id': acc['account_id'],
                            'type': 'FLAG',
                            'old_status': 'ACTIVE',
                            'new_status': 'DORMANT',
                            'reason': 'EXCEEDED_DORMANCY_THRESHOLD',
                            'user_id': 'SYSTEM'
                        }
                        for acc in dormant_accounts[:5]  # Limit to 5 for demo
                    ]
                else:
                    input_data = dormant_accounts

                # Execute the agent
                if input_data:
                    agent_results = agent_function(input_data)
                    processing_time = (datetime.now() - start_time).total_seconds()

                    results[agent_name] = AgentResult(
                        agent_name=agent_config['description'],
                        success=True,
                        records_processed=len(input_data) if isinstance(input_data, list) else 1,
                        results=agent_results if isinstance(agent_results, list) else [agent_results] if agent_results else [],
                        processing_time=processing_time,
                        summary=f"Processed {len(input_data) if isinstance(input_data, list) else 1} records, found {len(agent_results) if isinstance(agent_results, list) else (1 if agent_results else 0)} issues",
                        agent_type="compliance"
                    )
                else:
                    results[agent_name] = AgentResult(
                        agent_name=agent_config['description'],
                        success=False,
                        records_processed=0,
                        results=[],
                        processing_time=0,
                        summary="No input data available",
                        agent_type="compliance"
                    )

            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                results[agent_name] = AgentResult(
                    agent_name=agent_config['description'],
                    success=False,
                    records_processed=0,
                    results={'error': str(e)},
                    processing_time=processing_time,
                    summary=f"Failed: {str(e)}",
                    agent_type="compliance"
                )

        # Update memory with compliance results
        memory.update_memory("compliance", results)

        return results

class BankingComplianceApp:
    """Main Streamlit application for Banking Compliance Agentic AI"""

    def __init__(self):
        self.orchestrator = RealAgentOrchestrator()
        self.setup_session_state()
        self.setup_page_config()

    def setup_session_state(self):
        """Initialize session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'session_id' not in st.session_state:
            st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()
        if 'workflow_state' not in st.session_state:
            st.session_state.workflow_state = None
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'dormancy_results' not in st.session_state:
            st.session_state.dormancy_results = None
        if 'compliance_results' not in st.session_state:
            st.session_state.compliance_results = {}
        if 'column_mapping' not in st.session_state:
            st.session_state.column_mapping = {}

    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Banking Compliance Agentic AI",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79 0%, #2e8b57 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .agent-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success-agent { border-left: 4px solid #28a745; }
        .failed-agent { border-left: 4px solid #dc3545; }
        .processing-agent { border-left: 4px solid #ffc107; }
        .memory-display { 
            background: #f0f8ff; 
            border: 1px solid #b0c4de; 
            border-radius: 5px; 
            padding: 1rem; 
            margin: 0.5rem 0; 
        }
        </style>
        """, unsafe_allow_html=True)

    def run(self):
        """Main application entry point"""
        if not st.session_state.authenticated:
            self.show_login()
        else:
            self.show_main_app()

    def show_login(self):
        """Display login interface"""
        st.markdown("""
        <div class="main-header">
        <h1>🏦 Banking Compliance Agentic AI System</h1>
        <p>Real-time dormancy detection and compliance verification with interconnected AI agents</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.subheader("🔐 User Authentication")

                with st.form("login_form"):
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    submitted = st.form_submit_button("Login", type="primary")

                    if submitted:
                        if self.authenticate_user(username, password):
                            st.session_state.authenticated = True
                            st.session_state.user_id = username
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Please try again.")

    def authenticate_user(self, username: str, password: str) -> bool:
        """Simple authentication (replace with proper authentication in production)"""
        valid_users = {
            "admin": "admin123",
            "compliance_officer": "compliance123",
            "analyst": "analyst123"
        }
        return username in valid_users and valid_users[username] == password

    def show_main_app(self):
        """Display main application interface"""
        # Header
        st.markdown(f"""
        <div class="main-header">
        <h1>🤖 Banking Compliance Agentic AI System</h1>
        <p>Welcome, {st.session_state.user_id}! Session: {st.session_state.session_id[:8]}...</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar navigation
        with st.sidebar:
            st.header("🎛️ Navigation")
            page = st.selectbox("Select Page", [
                "Data Processing",
                "Dormancy Analysis",
                "Compliance Verification",
                "Agent Memory",
                "Reports Dashboard"
            ])

            # Agent Memory Status
            memory = self.orchestrator.get_memory(st.session_state.session_id)
            st.subheader("🧠 Agent Memory Status")
            st.write(f"**Session:** {memory.session_id[:8]}...")
            st.write(f"**Created:** {memory.created_at.strftime('%H:%M:%S') if memory.created_at else 'N/A'}")
            st.write(f"**Last Updated:** {memory.last_updated.strftime('%H:%M:%S') if memory.last_updated else 'N/A'}")

            status_indicators = {
                "Data Processed": "✅" if memory.processed_data is not None else "❌",
                "Dormancy Analyzed": "✅" if memory.dormancy_analysis_results is not None else "❌",
                "Compliance Verified": "✅" if memory.compliance_results is not None else "❌"
            }

            for status, indicator in status_indicators.items():
                st.write(f"{indicator} {status}")

            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.session_state.user_id = None
                st.rerun()

        # Page routing
        if page == "Data Processing":
            self.show_data_processing_page()
        elif page == "Dormancy Analysis":
            self.show_dormancy_analysis_page()
        elif page == "Compliance Verification":
            self.show_compliance_verification_page()
        elif page == "Agent Memory":
            self.show_agent_memory_page()
        elif page == "Reports Dashboard":
            self.show_reports_dashboard()

    def show_data_processing_page(self):
        """Data processing section with upload and mapping"""
        st.header("📊 Data Processing & Mapping")

        # Data Upload Section
        st.subheader("📁 Data Upload Agent")

        upload_method = st.selectbox("Select Upload Method", [
            "File Upload (CSV/Excel)",
            "Database Connection",
            "API Integration",
            "Manual Entry"
        ])

        if upload_method == "File Upload (CSV/Excel)":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload account data for analysis"
            )

            if uploaded_file:
                try:
                    if uploaded_file.type == "text/csv":
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)

                    st.session_state.uploaded_data = data
                    st.success(f"✅ File uploaded successfully! Shape: {data.shape}")

                    # Data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(data.head(), use_container_width=True)

                    # Data quality assessment
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Missing data analysis
                        missing_data = data.isnull().sum()
                        if missing_data.sum() > 0:
                            st.subheader("⚠️ Missing Data Analysis")
                            missing_df = pd.DataFrame({
                                'Column': missing_data.index,
                                'Missing Count': missing_data.values,
                                'Missing %': (missing_data.values / len(data)) * 100
                            })
                            missing_df = missing_df[missing_df['Missing Count'] > 0]
                            st.dataframe(missing_df, use_container_width=True)

                    with col2:
                        st.metric("Total Records", len(data))
                        st.metric("Total Columns", len(data.columns))
                        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                        st.metric("Missing Data %", f"{missing_percentage:.1f}%")

                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")

        elif upload_method == "Database Connection":
            st.info("🔗 Database connection functionality - Connect to your banking database")

        elif upload_method == "API Integration":
            st.info("🌐 API integration functionality - Connect to banking APIs")

        elif upload_method == "Manual Entry":
            st.info("✍️ Manual data entry functionality - Enter data manually")

        # Data Mapping Section
        if st.session_state.uploaded_data is not None:
            st.subheader("🗂️ Data Mapping Agent")

            data = st.session_state.uploaded_data

            # Enable LLM toggle
            enable_llm = st.toggle("🤖 Enable LLM Auto-Mapping",
                                 help="Let AI automatically map columns based on content analysis")

            # Required schema
            required_schema = {
                'account_id': 'Unique account identifier',
                'balance': 'Account balance (will be mapped to balance_current)',
                'last_transaction_date': 'Date of last transaction',
                'customer_id': 'Customer identifier',
                'account_type': 'Type of account',
                'currency': 'Account currency',
                'status': 'Account status'
            }

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Available Columns:**")
                for col in data.columns:
                    st.write(f"• {col}")

            with col2:
                st.write("**Schema Mapping:**")
                mapping = {}

                for schema_field, description in required_schema.items():
                    mapping[schema_field] = st.selectbox(
                        f"{schema_field}",
                        options=[""] + list(data.columns),
                        help=description,
                        key=f"mapping_{schema_field}"
                    )

                st.session_state.column_mapping = mapping

            # Process and map data
            if st.button("✅ Process Data", type="primary"):
                if any(mapping.values() for mapping in st.session_state.column_mapping.values()):
                    with st.spinner("Processing data with Data Processing Agent..."):
                        # Apply mapping
                        mapped_data = pd.DataFrame()
                        for schema_field, source_column in st.session_state.column_mapping.items():
                            if source_column and source_column in data.columns:
                                if schema_field == 'balance':
                                    mapped_data['balance_current'] = pd.to_numeric(data[source_column], errors='coerce')
                                else:
                                    mapped_data[schema_field] = data[source_column]

                        # Process with real agent
                        processing_result = self.orchestrator.process_data(mapped_data, st.session_state.session_id)

                        if processing_result.success:
                            st.session_state.processed_data = processing_result
                            st.success("✅ Data processed successfully by Data Processing Agent!")
                            st.json(processing_result.results)
                        else:
                            st.error(f"❌ Data processing failed: {processing_result.summary}")
                else:
                    st.error("❌ Please map at least one column before proceeding")

    def show_dormancy_analysis_page(self):
        """Dormancy analysis section"""
        st.header("🔍 Dormancy Analysis Agent")

        memory = self.orchestrator.get_memory(st.session_state.session_id)

        if memory.processed_data is None:
            st.warning("⚠️ Please complete data processing first")
            return

        # Display memory state
        st.subheader("🧠 Agent Memory - Data Processing Results")
        if memory.data_processing_results:
            st.json(memory.data_processing_results)

        # Run dormancy analysis
        if st.button("🚀 Run Dormancy Analysis", type="primary"):
            with st.spinner("Dormancy Analysis Agent is analyzing accounts..."):
                dormancy_result = self.orchestrator.analyze_dormancy(st.session_state.session_id)
                st.session_state.dormancy_results = dormancy_result

        # Display results
        if st.session_state.dormancy_results:
            result = st.session_state.dormancy_results

            # Result summary
            st.subheader("📊 Dormancy Analysis Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Accounts", result.records_processed)
            with col2:
                if result.success and 'dormant_accounts_found' in result.results:
                    st.metric("Dormant Accounts", result.results['dormant_accounts_found'])
                else:
                    st.metric("Dormant Accounts", 0)
            with col3:
                if result.success and 'dormancy_rate' in result.results:
                    st.metric("Dormancy Rate", f"{result.results['dormancy_rate']:.1f}%")
                else:
                    st.metric("Dormancy Rate", "0%")
            with col4:
                st.metric("Processing Time", f"{result.processing_time:.2f}s")

            # Status indicator
            if result.success:
                st.success(f"✅ {result.summary}")
            else:
                st.error(f"❌ {result.summary}")

            # Detailed results
            if result.success and 'dormant_accounts' in result.results:
                st.subheader("🔍 Dormant Accounts Found")

                dormant_accounts = result.results['dormant_accounts']
                if dormant_accounts:
                    dormant_df = pd.DataFrame(dormant_accounts)
                    st.dataframe(dormant_df, use_container_width=True)

                    # Download options
                    col1, col2 = st.columns(2)

                    with col1:
                        csv = dormant_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Dormant Accounts CSV",
                            data=csv,
                            file_name=f"dormant_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                    with col2:
                        summary_text = f"""
Dormancy Analysis Report
========================
Session ID: {st.session_state.session_id}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Agent: {result.agent_name}

Summary:
- Total Accounts Analyzed: {result.records_processed}
- Dormant Accounts Found: {result.results.get('dormant_accounts_found', 0)}
- Dormancy Rate: {result.results.get('dormancy_rate', 0):.1f}%
- Total Dormant Balance: {result.results.get('total_dormant_balance', 0):,.2f} AED
- Processing Time: {result.processing_time:.2f} seconds

Status: {'SUCCESS' if result.success else 'FAILED'}
                        """

                        st.download_button(
                            "📄 Download Analysis Summary",
                            data=summary_text,
                            file_name=f"dormancy_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                else:
                    st.info("No dormant accounts found in the analyzed data.")

            # Update memory display
            st.subheader("🧠 Updated Agent Memory")
            updated_memory = self.orchestrator.get_memory(st.session_state.session_id)
            if updated_memory.dormancy_analysis_results:
                st.json(updated_memory.dormancy_analysis_results)

    def show_compliance_verification_page(self):
        """Compliance verification section"""
        st.header("⚖️ Compliance Verification Agents")

        memory = self.orchestrator.get_memory(st.session_state.session_id)

        if memory.dormant_accounts is None or len(memory.dormant_accounts) == 0:
            st.warning("⚠️ Please complete dormancy analysis first. No dormant accounts available for compliance verification.")
            return

        # Display dormancy results from memory
        st.subheader("🧠 Agent Memory - Dormancy Analysis Results")
        st.write(f"**Dormant Accounts Available:** {len(memory.dormant_accounts)}")

        if memory.dormancy_analysis_results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analyzed", memory.dormancy_analysis_results.get('total_accounts_analyzed', 0))
            with col2:
                st.metric("Dormant Found", memory.dormancy_analysis_results.get('dormant_accounts_found', 0))
            with col3:
                st.metric("Dormancy Rate", f"{memory.dormancy_analysis_results.get('dormancy_rate', 0):.1f}%")

        # Run compliance verification
        if st.button("🚀 Run All Compliance Agents", type="primary"):
            with st.spinner("Running 17 Compliance Verification Agents..."):
                compliance_results = self.orchestrator.run_compliance_verification(st.session_state.session_id)
                st.session_state.compliance_results = compliance_results

        # Display compliance results
        if st.session_state.compliance_results:
            st.subheader("🤖 Compliance Agent Results")

            # Summary metrics
            total_agents = len(st.session_state.compliance_results)
            successful_agents = sum(1 for result in st.session_state.compliance_results.values() if result.success)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Agents", total_agents)
            with col2:
                st.metric("Successful", successful_agents)
            with col3:
                st.metric("Success Rate", f"{(successful_agents/total_agents*100):.1f}%" if total_agents > 0 else "0%")

            # Individual agent results
            for agent_name, result in st.session_state.compliance_results.items():
                if result.success and len(result.results) > 0:
                    with st.expander(f"✅ {result.agent_name} - Found {len(result.results)} issues"):

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**Summary:** {result.summary}")
                            st.write(f"**Records Processed:** {result.records_processed}")
                            st.write(f"**Processing Time:** {result.processing_time:.2f}s")

                            # Display results
                            if isinstance(result.results, list) and result.results:
                                # Show first few results
                                if len(result.results) > 0:
                                    st.subheader("Sample Results:")
                                    for i, item in enumerate(result.results[:3]):  # Show first 3
                                        st.json(item)
                                        if i < min(2, len(result.results) - 1):
                                            st.write("---")

                                    if len(result.results) > 3:
                                        st.write(f"... and {len(result.results) - 3} more items")

                        with col2:
                            # Download buttons
                            if isinstance(result.results, list) and result.results:
                                # CSV download
                                try:
                                    df = pd.DataFrame(result.results)
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download CSV",
                                        data=csv,
                                        file_name=f"{agent_name}_results.csv",
                                        mime="text/csv",
                                        key=f"csv_{agent_name}"
                                    )
                                except:
                                    # Fallback for complex nested data
                                    json_data = json.dumps(result.results, indent=2, default=str)
                                    st.download_button(
                                        "📥 Download JSON",
                                        data=json_data,
                                        file_name=f"{agent_name}_results.json",
                                        mime="application/json",
                                        key=f"json_{agent_name}"
                                    )

                                # Summary download
                                summary_text = f"""
Compliance Agent Report: {result.agent_name}
============================================
Session ID: {st.session_state.session_id}
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary: {result.summary}
Records Processed: {result.records_processed}
Issues Found: {len(result.results)}
Processing Time: {result.processing_time:.2f} seconds
Status: {'SUCCESS' if result.success else 'FAILED'}

Results Summary:
{json.dumps(result.results, indent=2, default=str)}
                                """

                                st.download_button(
                                    "📄 Download Summary",
                                    data=summary_text,
                                    file_name=f"{agent_name}_summary.txt",
                                    mime="text/plain",
                                    key=f"summary_{agent_name}"
                                )

                elif result.success and len(result.results) == 0:
                    st.info(f"✅ {result.agent_name}: No issues found - {result.summary}")

                else:
                    st.error(f"❌ {result.agent_name}: {result.summary}")

    def show_agent_memory_page(self):
        """Display agent memory and interconnections"""
        st.header("🧠 Agent Memory & Interconnections")

        memory = self.orchestrator.get_memory(st.session_state.session_id)

        # Memory overview
        st.subheader("📊 Memory Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="memory-display">
            <h4>🔄 Data Processing Memory</h4>
            """, unsafe_allow_html=True)

            if memory.data_processing_results:
                st.write("✅ **Status:** Available")
                st.write(f"**Records:** {memory.data_processing_results.get('processed_records', 0)}")
                st.write(f"**Columns:** {len(memory.data_processing_results.get('columns_processed', []))}")
            else:
                st.write("❌ **Status:** Not Available")

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="memory-display">
            <h4>🔍 Dormancy Analysis Memory</h4>
            """, unsafe_allow_html=True)

            if memory.dormancy_analysis_results:
                st.write("✅ **Status:** Available")
                st.write(f"**Analyzed:** {memory.dormancy_analysis_results.get('total_accounts_analyzed', 0)}")
                st.write(f"**Dormant:** {memory.dormancy_analysis_results.get('dormant_accounts_found', 0)}")
                st.write(f"**Rate:** {memory.dormancy_analysis_results.get('dormancy_rate', 0):.1f}%")
            else:
                st.write("❌ **Status:** Not Available")

            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="memory-display">
            <h4>⚖️ Compliance Memory</h4>
            """, unsafe_allow_html=True)

            if memory.compliance_results:
                st.write("✅ **Status:** Available")
                successful = sum(1 for r in memory.compliance_results.values() if r.success)
                st.write(f"**Agents Run:** {len(memory.compliance_results)}")
                st.write(f"**Successful:** {successful}")
            else:
                st.write("❌ **Status:** Not Available")

            st.markdown("</div>", unsafe_allow_html=True)

        # Data flow visualization
        st.subheader("🔄 Agent Data Flow")

        # Create flow diagram using Plotly
        fig = go.Figure()

        # Define positions
        positions = {
            'Data Upload': (1, 3),
            'Data Processing': (2, 3),
            'Dormancy Analysis': (3, 3),
            'Compliance Verification': (4, 3),
            'Memory Store': (2.5, 2)
        }

        # Add nodes
        for agent, (x, y) in positions.items():
            color = 'lightgreen' if agent in ['Data Processing', 'Dormancy Analysis'] and memory.dormancy_analysis_results else 'lightcoral'
            if agent == 'Compliance Verification' and memory.compliance_results:
                color = 'lightgreen'
            elif agent == 'Memory Store':
                color = 'lightblue'

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=50, color=color),
                text=agent,
                textposition="middle center",
                showlegend=False
            ))

        # Add arrows (simplified)
        arrows = [
            ('Data Upload', 'Data Processing'),
            ('Data Processing', 'Dormancy Analysis'),
            ('Dormancy Analysis', 'Compliance Verification'),
            ('Data Processing', 'Memory Store'),
            ('Dormancy Analysis', 'Memory Store'),
            ('Compliance Verification', 'Memory Store')
        ]

        for start, end in arrows:
            start_pos = positions[start]
            end_pos = positions[end]

            fig.add_annotation(
                x=end_pos[0], y=end_pos[1],
                ax=start_pos[0], ay=start_pos[1],
                xref='x', yref='y',
                axref='x', ayref='y',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='blue'
            )

        fig.update_layout(
            title="Agent Data Flow and Memory Interconnections",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed memory contents
        st.subheader("📋 Detailed Memory Contents")

        memory_tabs = st.tabs(["Data Processing", "Dormancy Analysis", "Compliance Results", "Raw Data"])

        with memory_tabs[0]:
            if memory.data_processing_results:
                st.json(memory.data_processing_results)
            else:
                st.info("No data processing results in memory")

        with memory_tabs[1]:
            if memory.dormancy_analysis_results:
                st.json(memory.dormancy_analysis_results)
            else:
                st.info("No dormancy analysis results in memory")

        with memory_tabs[2]:
            if memory.compliance_results:
                # Show summary of compliance results
                compliance_summary = {}
                for agent_name, result in memory.compliance_results.items():
                    compliance_summary[agent_name] = {
                        'success': result.success,
                        'records_processed': result.records_processed,
                        'issues_found': len(result.results) if isinstance(result.results, list) else 0,
                        'processing_time': result.processing_time,
                        'summary': result.summary
                    }
                st.json(compliance_summary)
            else:
                st.info("No compliance results in memory")

        with memory_tabs[3]:
            if memory.processed_data is not None:
                st.dataframe(memory.processed_data.head(10), use_container_width=True)
                st.write(f"**Shape:** {memory.processed_data.shape}")
            else:
                st.info("No processed data in memory")

    def show_reports_dashboard(self):
        """Reports and analytics dashboard"""
        st.header("📊 Reports Dashboard")

        memory = self.orchestrator.get_memory(st.session_state.session_id)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_records = 0
            if memory.data_processing_results:
                total_records = memory.data_processing_results.get('processed_records', 0)
            st.metric("Total Records", total_records)

        with col2:
            dormant_count = 0
            if memory.dormancy_analysis_results:
                dormant_count = memory.dormancy_analysis_results.get('dormant_accounts_found', 0)
            st.metric("Dormant Accounts", dormant_count)

        with col3:
            compliance_agents = 0
            if memory.compliance_results:
                compliance_agents = len(memory.compliance_results)
            st.metric("Compliance Agents", f"{compliance_agents}/17")

        with col4:
            total_balance = 0
            if memory.dormancy_analysis_results:
                total_balance = memory.dormancy_analysis_results.get('total_dormant_balance', 0)
            st.metric("Dormant Balance", f"{total_balance:,.2f} AED")

        # Agent performance overview
        if memory.compliance_results:
            st.subheader("🤖 Agent Performance Overview")

            # Create performance data
            agent_data = []
            for agent_name, result in memory.compliance_results.items():
                agent_data.append({
                    'Agent': result.agent_name[:30] + "..." if len(result.agent_name) > 30 else result.agent_name,
                    'Status': 'Success' if result.success else 'Failed',
                    'Records Processed': result.records_processed,
                    'Issues Found': len(result.results) if isinstance(result.results, list) else 0,
                    'Processing Time (s)': result.processing_time
                })

            if agent_data:
                agent_df = pd.DataFrame(agent_data)

                # Performance chart
                col1, col2 = st.columns(2)

                with col1:
                    # Success rate pie chart
                    success_counts = agent_df['Status'].value_counts()
                    fig_pie = px.pie(
                        values=success_counts.values,
                        names=success_counts.index,
                        title="Agent Success Rate"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Issues found bar chart
                    top_agents = agent_df.nlargest(10, 'Issues Found')
                    fig_bar = px.bar(
                        top_agents,
                        x='Issues Found',
                        y='Agent',
                        orientation='h',
                        title="Top 10 Agents by Issues Found"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Detailed agent table
                st.subheader("📋 Detailed Agent Results")
                st.dataframe(agent_df, use_container_width=True)

                # Download comprehensive report
                if st.button("📥 Download Comprehensive Report"):
                    report_data = {
                        'session_id': st.session_state.session_id,
                        'user_id': st.session_state.user_id,
                        'report_generated': datetime.now().isoformat(),
                        'summary': {
                            'total_records': total_records,
                            'dormant_accounts': dormant_count,
                            'agents_executed': compliance_agents,
                            'total_dormant_balance': total_balance
                        },
                        'agent_results': agent_data,
                        'memory_state': {
                            'data_processing_available': memory.data_processing_results is not None,
                            'dormancy_analysis_available': memory.dormancy_analysis_results is not None,
                            'compliance_results_available': memory.compliance_results is not None
                        }
                    }

                    report_json = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        "📊 Download Complete Report (JSON)",
                        data=report_json,
                        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

        else:
            st.info("📋 No compliance results available. Please run the compliance verification workflow first.")

# Main application
if __name__ == "__main__":
    app = BankingComplianceApp()
    app.run()