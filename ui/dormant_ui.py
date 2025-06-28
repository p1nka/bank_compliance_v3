import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import hashlib
import secrets
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import all agents and utilities
from agents.Dormant_agent import *
from agents.memory_agent import create_sync_memory_agent
from agents.error_handler_agent import ErrorHandlerAgent, ErrorState
from agents.supervisor_agent import SupervisorAgent, SupervisorState
from agents.notification_agent import NotificationAgent, NotificationState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Mock Llama 3 8B Integration
class LlamaRecommendationEngine:
    """Mock Llama 3 8B Instruct for generating recommendations"""

    def __init__(self):
        self.model_name = "Llama 3 8B Instruct"

    def generate_recommendations(self, agent_name: str, dormant_accounts: List[Dict], summary_stats: Dict) -> str:
        """Generate intelligent recommendations based on dormancy findings"""

        recommendations = {
            "safe_deposit_dormancy": [
                "📋 Initiate CBUAE Article 3.7 safe deposit box access procedures",
                "📞 Contact customers using last known contact information",
                "⚖️ Prepare court application documentation for 3+ year dormant boxes",
                "💰 Calculate outstanding charges and fees",
                "📨 Send formal notifications per regulatory requirements"
            ],
            "investment_account_inactivity": [
                "📊 Review investment product maturity and redemption status",
                "👤 Contact customers regarding dormant investment accounts",
                "💹 Assess current market value of dormant investments",
                "📋 Prepare transfer documentation for unclaimed investments",
                "🔄 Implement systematic review process for investment dormancy"
            ],
            "demand_deposit_inactivity": [
                "📞 Initiate customer contact attempts per CBUAE Article 2.1",
                "📧 Send notification letters to last known addresses",
                "🏦 Flag accounts for dormant account ledger transfer",
                "💰 Monitor accounts for reactivation triggers",
                "📝 Document all contact attempts and responses"
            ],
            "high_value_dormant": [
                "🚨 PRIORITY: Immediate escalation for high-value accounts",
                "👔 Assign dedicated relationship manager for outreach",
                "🔍 Enhanced due diligence on account status",
                "💼 Executive-level customer retention initiatives",
                "⚡ Fast-track reactivation procedures"
            ]
        }

        base_recommendations = recommendations.get(agent_name, [
            "📋 Review account status and customer contact information",
            "📞 Initiate customer outreach program",
            "💰 Assess balances and dormancy periods",
            "📝 Document findings and actions taken",
            "🔄 Implement monitoring for reactivation"
        ])

        # Add specific recommendations based on findings
        account_count = len(dormant_accounts)
        total_balance = summary_stats.get('total_amount_dormant', 0)

        specific_recs = []
        if account_count > 100:
            specific_recs.append("📈 High volume detected - implement batch processing procedures")
        if total_balance > 1000000:
            specific_recs.append("💰 High value detected - escalate to senior management")
        if account_count > 0:
            specific_recs.append(f"📊 Process {account_count} accounts through compliance workflow")

        return "\n".join(base_recommendations + specific_recs)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
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
    if 'memory_agent' not in st.session_state:
        st.session_state.memory_agent = create_sync_memory_agent()
    if 'llama_engine' not in st.session_state:
        st.session_state.llama_engine = LlamaRecommendationEngine()


class DataUploadAgent:
    """Handles multiple data upload methods"""

    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'json', 'txt']

    def upload_data(self) -> Optional[pd.DataFrame]:
        """Handle 4 different ways to upload data"""

        st.subheader("📤 Data Upload Methods")

        upload_method = st.radio(
            "Choose upload method:",
            ["File Upload", "Paste Data", "Sample Data", "Database Connection"],
            horizontal=True
        )

        data = None

        if upload_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload your banking data file",
                type=self.supported_formats,
                help="Supported formats: CSV, Excel, JSON, TXT"
            )
            if uploaded_file:
                data = self._process_uploaded_file(uploaded_file)

        elif upload_method == "Paste Data":
            pasted_data = st.text_area(
                "Paste your CSV data here:",
                height=200,
                placeholder="customer_id,account_id,account_type,balance,last_transaction_date\n..."
            )
            if pasted_data:
                data = self._process_pasted_data(pasted_data)

        elif upload_method == "Sample Data":
            if st.button("Load Sample Banking Data"):
                data = self._generate_sample_data()
                st.success("Sample data loaded successfully!")

        elif upload_method == "Database Connection":
            st.info("🚧 Database connection feature coming soon!")
            # Mock database connection
            if st.button("Connect to Database"):
                data = self._generate_sample_data()
                st.success("Mock database data loaded!")

        return data

    def _process_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """Process uploaded file based on type"""
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                return pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None

    def _process_pasted_data(self, pasted_data: str) -> pd.DataFrame:
        """Process pasted CSV data"""
        try:
            return pd.read_csv(io.StringIO(pasted_data))
        except Exception as e:
            st.error(f"Error processing pasted data: {str(e)}")
            return None

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample banking data"""
        np.random.seed(42)
        n_accounts = 1000

        return pd.DataFrame({
            'customer_id': [f'CUST{str(i).zfill(6)}' for i in range(1, n_accounts + 1)],
            'account_id': [f'ACC{str(i).zfill(8)}' for i in range(1, n_accounts + 1)],
            'account_type': np.random.choice(['SAVINGS', 'CURRENT', 'FIXED_DEPOSIT', 'INVESTMENT', 'SAFE_DEPOSIT'],
                                             n_accounts),
            'account_subtype': np.random.choice(['INDIVIDUAL', 'JOINT', 'CORPORATE', 'PREMIUM'], n_accounts),
            'balance_current': np.random.exponential(10000, n_accounts),
            'last_transaction_date': pd.date_range(end='2024-01-01', periods=n_accounts, freq='-1D').strftime(
                '%Y-%m-%d'),
            'account_status': np.random.choice(['ACTIVE', 'INACTIVE', 'DORMANT'], n_accounts, p=[0.7, 0.2, 0.1]),
            'currency': 'AED',
            'contact_attempts_made': np.random.randint(0, 6, n_accounts),
            'last_contact_date': pd.date_range(end='2023-12-01', periods=n_accounts, freq='-2D').strftime('%Y-%m-%d'),
            'dormancy_status': np.random.choice(['ACTIVE', 'DORMANT', 'FLAGGED'], n_accounts, p=[0.75, 0.15, 0.1])
        })


class DataProcessingAgent:
    """Enhanced data processing with quality assessment"""

    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Process and assess data quality"""

        # Pre-memory hook - load processing patterns
        processing_context = self._pre_memory_hook()

        # Data quality assessment
        quality_report = self._assess_data_quality(data)

        # Data cleaning and standardization
        cleaned_data = self._clean_data(data)

        # Post-memory hook - store processing results
        processing_results = {
            'original_shape': data.shape,
            'cleaned_shape': cleaned_data.shape,
            'quality_score': quality_report['overall_score'],
            'missing_percentage': quality_report['missing_percentage'],
            'data_types': quality_report['data_types'],
            'quality_issues': quality_report['issues'],
            'recommendations': quality_report['recommendations']
        }

        self._post_memory_hook(processing_results)

        return {
            'processed_data': cleaned_data,
            'quality_report': quality_report,
            'processing_results': processing_results
        }

    def _pre_memory_hook(self) -> Dict:
        """Load processing patterns from memory"""
        context = self.memory_agent.retrieve_memory("data_processing_patterns")
        return context or {}

    def _post_memory_hook(self, results: Dict):
        """Store processing results in memory"""
        self.memory_agent.store_memory(
            "data_processing_results",
            {
                'timestamp': datetime.now().isoformat(),
                'processing_results': results,
                'session_id': st.session_state.get('session_id', secrets.token_hex(8))
            }
        )

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""

        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100

        # Column-wise analysis
        column_quality = {}
        for col in data.columns:
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            column_quality[col] = {
                'missing_percentage': missing_pct,
                'data_type': str(data[col].dtype),
                'unique_values': data[col].nunique(),
                'quality_score': max(0, 100 - missing_pct)
            }

        # Overall quality score
        overall_score = max(0, 100 - missing_percentage)

        # Issues and recommendations
        issues = []
        recommendations = []

        if missing_percentage > 20:
            issues.append("High percentage of missing data")
            recommendations.append("Consider data imputation or source validation")

        if missing_percentage > 10:
            issues.append("Moderate missing data detected")
            recommendations.append("Review data collection processes")

        return {
            'overall_score': overall_score,
            'missing_percentage': missing_percentage,
            'column_quality': column_quality,
            'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'issues': issues,
            'recommendations': recommendations
        }

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        cleaned = data.copy()

        # Standardize column names
        cleaned.columns = cleaned.columns.str.lower().str.replace(' ', '_')

        # Handle missing values
        for col in cleaned.columns:
            if cleaned[col].dtype == 'object':
                cleaned[col] = cleaned[col].fillna('UNKNOWN')
            else:
                cleaned[col] = cleaned[col].fillna(0)

        return cleaned


class DataMappingAgent:
    """Intelligent data mapping with LLM assistance"""

    def __init__(self, memory_agent, llama_engine):
        self.memory_agent = memory_agent
        self.llama_engine = llama_engine
        self.standard_schema = {
            'customer_id': 'Unique customer identifier',
            'account_id': 'Unique account identifier',
            'account_type': 'Type of account (SAVINGS, CURRENT, etc.)',
            'account_subtype': 'Account subtype classification',
            'balance_current': 'Current account balance',
            'last_transaction_date': 'Date of last transaction',
            'account_status': 'Current status of account',
            'currency': 'Account currency',
            'contact_attempts_made': 'Number of contact attempts',
            'last_contact_date': 'Date of last contact attempt',
            'dormancy_status': 'Dormancy classification'
        }

    def map_data(self, data: pd.DataFrame, enable_llm: bool = False) -> Dict[str, Any]:
        """Map data columns to standard schema"""

        st.subheader("🗺️ Data Mapping Configuration")

        if enable_llm:
            mapping = self._llm_assisted_mapping(data)
            st.success("🤖 LLM-assisted mapping completed!")
        else:
            mapping = self._manual_mapping(data)

        # Apply mapping
        mapped_data = self._apply_mapping(data, mapping)

        # Generate mapping sheet
        mapping_sheet = self._create_mapping_sheet(mapping)

        return {
            'mapped_data': mapped_data,
            'mapping': mapping,
            'mapping_sheet': mapping_sheet
        }

    def _llm_assisted_mapping(self, data: pd.DataFrame) -> Dict[str, str]:
        """Use LLM to suggest optimal column mapping"""

        # Mock LLM mapping logic
        mapping = {}
        data_columns = list(data.columns)

        # Simple fuzzy matching for demo
        for schema_col in self.standard_schema:
            best_match = None
            best_score = 0

            for data_col in data_columns:
                score = self._calculate_similarity(schema_col.lower(), data_col.lower())
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = data_col

            if best_match:
                mapping[schema_col] = best_match
                data_columns.remove(best_match)

        # Display LLM suggestions
        st.write("🤖 **LLM Mapping Suggestions:**")
        for schema_col, data_col in mapping.items():
            st.write(f"• {schema_col} ← {data_col}")

        return mapping

    def _manual_mapping(self, data: pd.DataFrame) -> Dict[str, str]:
        """Manual column mapping interface"""

        mapping = {}
        st.write("🔧 **Manual Column Mapping:**")

        for schema_col, description in self.standard_schema.items():
            options = ['-- Select Column --'] + list(data.columns)
            selected = st.selectbox(
                f"{schema_col}",
                options,
                help=description,
                key=f"mapping_{schema_col}"
            )
            if selected != '-- Select Column --':
                mapping[schema_col] = selected

        return mapping

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity for column matching"""
        import difflib
        return difflib.SequenceMatcher(None, str1, str2).ratio()

    def _apply_mapping(self, data: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply column mapping to data"""
        mapped_data = pd.DataFrame()

        for schema_col, data_col in mapping.items():
            if data_col in data.columns:
                mapped_data[schema_col] = data[data_col]

        return mapped_data

    def _create_mapping_sheet(self, mapping: Dict[str, str]) -> pd.DataFrame:
        """Create downloadable mapping sheet"""
        return pd.DataFrame([
            {'Standard_Column': k, 'Source_Column': v, 'Description': self.standard_schema[k]}
            for k, v in mapping.items()
        ])


class EnhancedDormancyOrchestrator:
    """Enhanced orchestrator with memory hooks and error handling"""

    def __init__(self, memory_agent, error_handler, supervisor_agent, llama_engine):
        self.memory_agent = memory_agent
        self.error_handler = error_handler
        self.supervisor_agent = supervisor_agent
        self.llama_engine = llama_engine

        # Initialize all dormancy agents
        self.agents = {
            'safe_deposit_dormancy': Enhanced1_SafeDepositDormancyAgent(),
            'investment_account_inactivity': Enhanced2_InvestmentAccountInactivityAgent(),
            'fixed_deposit_inactivity': Enhanced3_FixedDepositInactivityAgent(),
            'demand_deposit_inactivity': Enhanced4_DemandDepositInactivityAgent(),
            'unclaimed_payment_instruments': Enhanced5_UnclaimedPaymentInstrumentsAgent(),
            'eligible_for_cbuae_transfer': Enhanced6_EligibleForCBUAETransferAgent(),
            'article_3_process_needed': Enhanced7_Article3ProcessNeededAgent(),
            'contact_attempts_needed': Enhanced8_ContactAttemptsNeededAgent(),
            'high_value_dormant': Enhanced9_HighValueDormantAgent(),
            'dormant_to_active_transitions': Enhanced10_DormantToActiveTransitionsAgent()
        }

    def run_dormancy_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive dormancy analysis with memory hooks"""

        # Pre-memory hook - load data processing results
        processing_context = self._pre_memory_hook()

        results = {}
        active_agents = []

        # Create mock state
        class MockState:
            def __init__(self, df):
                self.input_dataframe = df
                self.session_id = secrets.token_hex(8)
                self.user_id = st.session_state.username or "anonymous"

        state = MockState(data)

        # Run each agent with error handling
        for agent_name, agent in self.agents.items():
            try:
                # Execute agent analysis
                result = agent.analyze_dormancy(state)

                if result.get('success', False) and result.get('dormant_records_found', 0) > 0:
                    # Generate LLM recommendations
                    dormant_accounts = result.get('analysis_results', {}).get('dormant_accounts', [])
                    summary_stats = {
                        'total_dormant_accounts_found': result.get('dormant_records_found', 0),
                        'total_amount_dormant': sum([acc.get('balance_current', 0) for acc in dormant_accounts])
                    }

                    recommendations = self.llama_engine.generate_recommendations(
                        agent_name, dormant_accounts, summary_stats
                    )

                    result['llm_recommendations'] = recommendations
                    results[agent_name] = result
                    active_agents.append(agent_name)

            except Exception as e:
                # Handle errors through error handler
                error_state = ErrorState(
                    session_id=state.session_id,
                    user_id=state.user_id,
                    error_id=secrets.token_hex(8),
                    timestamp=datetime.now(),
                    errors=[{'agent': agent_name, 'error': str(e)}],
                    failed_node=agent_name
                )
                # In real implementation, would call error_handler.handle_workflow_error(error_state)
                logger.error(f"Agent {agent_name} failed: {e}")

        # Post-memory hook - store all results
        self._post_memory_hook(results, active_agents)

        # Supervisor evaluation
        supervisor_results = self._supervisor_evaluation(results, active_agents)

        return {
            'agent_results': results,
            'active_agents': active_agents,
            'supervisor_summary': supervisor_results,
            'total_agents_run': len(active_agents),
            'total_dormant_found': sum([r.get('dormant_records_found', 0) for r in results.values()])
        }

    def _pre_memory_hook(self) -> Dict:
        """Load data processing context from memory"""
        context = self.memory_agent.retrieve_memory("data_processing_results")
        return context or {}

    def _post_memory_hook(self, results: Dict, active_agents: List[str]):
        """Store dormancy analysis results in memory"""
        memory_data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.get('session_id', secrets.token_hex(8)),
            'active_agents': active_agents,
            'results_summary': {
                agent: {
                    'dormant_found': result.get('dormant_records_found', 0),
                    'success': result.get('success', False),
                    'processing_time': result.get('processing_time', 0)
                } for agent, result in results.items()
            },
            'total_dormant_accounts': sum([r.get('dormant_records_found', 0) for r in results.values()])
        }

        self.memory_agent.store_memory("dormancy_analysis_results", memory_data)

    def _supervisor_evaluation(self, results: Dict, active_agents: List[str]) -> Dict:
        """Supervisor evaluation of dormancy analysis"""

        total_dormant = sum([r.get('dormant_records_found', 0) for r in results.values()])
        total_agents = len(active_agents)

        # Mock supervisor logic
        if total_dormant > 100:
            priority = "HIGH"
            action = "Immediate management review required"
        elif total_dormant > 50:
            priority = "MEDIUM"
            action = "Schedule compliance review"
        else:
            priority = "LOW"
            action = "Regular monitoring"

        return {
            'evaluation_priority': priority,
            'recommended_action': action,
            'total_dormant_accounts': total_dormant,
            'agents_executed': total_agents,
            'supervisor_notes': f"Analysis completed successfully with {total_agents} agents identifying {total_dormant} dormant accounts"
        }


def render_login():
    """Render login interface"""
    st.title("🏦 Banking Compliance Agentic AI")
    st.subheader("🔐 Secure Login")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("🚀 Login", type="primary", use_container_width=True):
            # Mock authentication
            if username and password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.session_id = secrets.token_hex(16)
                st.rerun()
            else:
                st.error("Please enter both username and password")


def render_agent_card(agent_name: str, result: Dict, llama_engine):
    """Render agent result card with summary"""

    agent_display_names = {
        'safe_deposit_dormancy': '🔐 Safe Deposit Dormancy',
        'investment_account_inactivity': '📈 Investment Account Inactivity',
        'fixed_deposit_inactivity': '💰 Fixed Deposit Inactivity',
        'demand_deposit_inactivity': '🏦 Demand Deposit Inactivity',
        'unclaimed_payment_instruments': '📄 Unclaimed Payment Instruments',
        'eligible_for_cbuae_transfer': '🔄 CBUAE Transfer Eligible',
        'article_3_process_needed': '📋 Article 3 Process Needed',
        'contact_attempts_needed': '📞 Contact Attempts Needed',
        'high_value_dormant': '💎 High Value Dormant',
        'dormant_to_active_transitions': '🔄 Dormant to Active Transitions'
    }

    dormant_found = result.get('dormant_records_found', 0)
    agent_title = agent_display_names.get(agent_name, agent_name.replace('_', ' ').title())

    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"### {agent_title}")
            st.markdown(f"**Dormant Accounts Found:** {dormant_found}")

        with col2:
            if st.button("📊 Summary", key=f"summary_{agent_name}"):
                st.session_state[f"show_summary_{agent_name}"] = True

        with col3:
            # Download CSV
            if dormant_found > 0:
                dormant_accounts = result.get('analysis_results', {}).get('dormant_accounts', [])
                csv_data = pd.DataFrame(dormant_accounts).to_csv(index=False)
                st.download_button(
                    "📥 CSV",
                    csv_data,
                    f"{agent_name}_dormant_accounts.csv",
                    "text/csv",
                    key=f"download_{agent_name}"
                )

        # Show summary if requested
        if st.session_state.get(f"show_summary_{agent_name}", False):
            with st.expander(f"📋 {agent_title} - Detailed Summary", expanded=True):

                st.markdown("#### 1. Title & Description")
                st.write(result.get('analysis_results', {}).get('description', 'No description available'))

                st.markdown("#### 2. Dormant Accounts Found")
                st.metric("Total Dormant Accounts", dormant_found)

                if dormant_found > 0:
                    dormant_accounts = result.get('analysis_results', {}).get('dormant_accounts', [])

                    # Sample accounts preview
                    st.markdown("**Sample Accounts:**")
                    sample_df = pd.DataFrame(dormant_accounts[:5])  # Show first 5
                    st.dataframe(sample_df)

                st.markdown("#### 3. Recommended Actions")
                recommendations = result.get('llm_recommendations', 'No recommendations available')
                st.markdown(recommendations)

                # Download summary
                summary_text = f"""
{agent_title} - Analysis Summary

1. TITLE & DESCRIPTION:
{result.get('analysis_results', {}).get('description', 'No description available')}

2. DORMANT ACCOUNTS FOUND:
Total: {dormant_found} accounts

3. RECOMMENDED ACTIONS:
{recommendations}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                st.download_button(
                    "📥 Download Summary",
                    summary_text,
                    f"{agent_name}_summary.txt",
                    "text/plain",
                    key=f"summary_download_{agent_name}"
                )

                if st.button("❌ Close", key=f"close_{agent_name}"):
                    st.session_state[f"show_summary_{agent_name}"] = False
                    st.rerun()


def main():
    """Main application logic"""

    # Initialize session state
    initialize_session_state()

    # Check authentication
    if not st.session_state.authenticated:
        render_login()
        return

    # Main application
    st.set_page_config(
        page_title="Banking Compliance Agentic AI",
        page_icon="🏦",
        layout="wide"
    )

    # Header
    st.title("🤖 Banking Compliance Agentic AI System")
    st.markdown(
        f"**Welcome, {st.session_state.username}** | Session: {st.session_state.get('session_id', 'N/A')[:8]}...")

    # Initialize agents
    memory_agent = st.session_state.memory_agent
    llama_engine = st.session_state.llama_engine

    # Mock other agents for now
    error_handler = None  # Would be ErrorHandlerAgent(memory_agent, mcp_client)
    supervisor_agent = None  # Would be SupervisorAgent(memory_agent, mcp_client)

    # Main sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Data Processing",
        "🔍 Dormancy Analysis",
        "⚖️ Compliance",
        "📊 Reports"
    ])

    with tab1:
        st.header("📤 Data Processing Section")

        # Data Upload
        upload_agent = DataUploadAgent()
        uploaded_data = upload_agent.upload_data()

        if uploaded_data is not None:
            st.session_state.uploaded_data = uploaded_data
            st.success(f"✅ Data uploaded successfully! Shape: {uploaded_data.shape}")

            # Data Processing
            st.subheader("🔧 Data Quality Assessment")
            processing_agent = DataProcessingAgent(memory_agent)
            processing_results = processing_agent.process_data(uploaded_data)
            st.session_state.processed_data = processing_results

            # Display quality metrics
            quality_report = processing_results['quality_report']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Quality Score", f"{quality_report['overall_score']:.1f}%")
            with col2:
                st.metric("Missing Data", f"{quality_report['missing_percentage']:.1f}%")
            with col3:
                st.metric("Total Rows", uploaded_data.shape[0])
            with col4:
                st.metric("Total Columns", uploaded_data.shape[1])

            # Data Mapping
            st.subheader("🗺️ Data Mapping")
            enable_llm = st.checkbox("🤖 Enable LLM-Assisted Mapping", value=True)

            if st.button("🚀 Start Data Mapping"):
                mapping_agent = DataMappingAgent(memory_agent, llama_engine)
                mapping_results = mapping_agent.map_data(uploaded_data, enable_llm)
                st.session_state.mapped_data = mapping_results

                # Download mapping sheet
                mapping_sheet = mapping_results['mapping_sheet']
                csv_data = mapping_sheet.to_csv(index=False)
                st.download_button(
                    "📥 Download Mapping Sheet",
                    csv_data,
                    "data_mapping_sheet.csv",
                    "text/csv"
                )

    with tab2:
        st.header("🔍 Dormancy Analysis Section")

        if st.session_state.mapped_data is not None:
            mapped_data = st.session_state.mapped_data['mapped_data']

            if st.button("🚀 Run Dormancy Analysis", type="primary"):
                # Initialize orchestrator
                orchestrator = EnhancedDormancyOrchestrator(
                    memory_agent, error_handler, supervisor_agent, llama_engine
                )

                # Run analysis
                with st.spinner("🤖 Running dormancy analysis..."):
                    dormancy_results = orchestrator.run_dormancy_analysis(mapped_data)
                    st.session_state.dormancy_results = dormancy_results

            # Display results
            if st.session_state.dormancy_results:
                results = st.session_state.dormancy_results

                # Summary metrics
                st.subheader("📊 Analysis Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Active Agents", results['total_agents_run'])
                with col2:
                    st.metric("Total Dormant Found", results['total_dormant_found'])
                with col3:
                    supervisor_priority = results['supervisor_summary']['evaluation_priority']
                    st.metric("Priority Level", supervisor_priority)

                # Agent results (only show agents with dormant accounts found)
                st.subheader("🤖 Agent Results (Active Agents Only)")

                for agent_name in results['active_agents']:
                    agent_result = results['agent_results'][agent_name]
                    render_agent_card(agent_name, agent_result, llama_engine)
                    st.divider()
        else:
            st.info("⚠️ Please complete data processing and mapping first.")

    with tab3:
        st.header("⚖️ Compliance Verification")
        st.info("🚧 Compliance agents integration coming soon!")

        # Mock compliance results for now
        if st.session_state.dormancy_results:
            st.subheader("📋 Compliance Status")

            compliance_agents = [
                "Article 2 Compliance",
                "Article 3.1 Process Management",
                "Article 3.4 Transfer Requirements",
                "Contact Verification",
                "Amount Verification"
            ]

            for agent in compliance_agents:
                with st.expander(f"⚖️ {agent}"):
                    st.write("Compliance check results would appear here...")
                    st.progress(0.85, text="85% Compliant")

    with tab4:
        st.header("📊 Reports Section")

        if st.session_state.dormancy_results:
            results = st.session_state.dormancy_results

            # All agents summary
            st.subheader("📈 All Agents Summary")

            all_agents_data = []
            for agent_name, agent_result in results['agent_results'].items():
                all_agents_data.append({
                    'Agent': agent_name.replace('_', ' ').title(),
                    'Accounts Found': agent_result.get('dormant_records_found', 0),
                    'Status': 'Active' if agent_result.get('success', False) else 'Failed',
                    'Processing Time': f"{agent_result.get('processing_time', 0):.2f}s",
                    'Recommended Actions': 'View Details'
                })

            summary_df = pd.DataFrame(all_agents_data)
            st.dataframe(summary_df, use_container_width=True)

            # Download comprehensive report
            if st.button("📥 Generate Comprehensive Report"):
                report_data = {
                    'Analysis Summary': results['supervisor_summary'],
                    'Agent Results': results['agent_results'],
                    'Generated On': datetime.now().isoformat()
                }

                st.download_button(
                    "📥 Download Full Report (JSON)",
                    json.dumps(report_data, indent=2, default=str),
                    "comprehensive_dormancy_report.json",
                    "application/json"
                )
        else:
            st.info("⚠️ No analysis results available. Please run dormancy analysis first.")


if __name__ == "__main__":
    main()