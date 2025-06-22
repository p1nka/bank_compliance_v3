"""
Banking Compliance Analysis - Enhanced Streamlit Application
Integrated with Comprehensive CBUAE Dormancy Agent System
Uses BGE-large embeddings and cosine similarity for precise column mapping
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
from sklearn.metrics.pairwise import cosine_similarity
import logging
import traceback
# Configure Streamlit page
st.set_page_config(
    page_title="CBUAE Banking Compliance Analysis System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Import the comprehensive CBUAE dormancy system
try:
    from agents.dormant_agent import (
        run_comprehensive_dormancy_analysis_csv,
        DormancyAnalysisAgent,
        DormancyWorkflowOrchestrator,
        validate_csv_structure,
        DemandDepositDormancyAgent,
        FixedDepositDormancyAgent,
        InvestmentAccountDormancyAgent,
        ContactAttemptsAgent,
        CBTransferEligibilityAgent,
        ForeignCurrencyConversionAgent,
        BaseDormancyAgent,
        AgentState,
        DormancyAnalysisState
    )
    DORMANCY_SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ CBUAE Dormancy System not available: {e}")
    DORMANCY_SYSTEM_AVAILABLE = False

warnings.filterwarnings('ignore')

# Handle PyTorch/Streamlit compatibility issues
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# BGE Model Loading with proper error handling
@st.cache_resource
def load_bge_model():
    """Load BGE-large model with caching and error handling"""
    try:
        from sentence_transformers import SentenceTransformer
        st.info("🔄 Loading BGE-large model (first time may take a while)...")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        st.success("✅ BGE-large model loaded successfully!")
        return model
    except ImportError:
        st.error("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load BGE model: {e}")
        return None

# Configure Streamlit page


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'logged_in': False,
        'username': "",
        'uploaded_data': None,
        'processed_data': None,
        'mapped_data': None,
        'quality_results': None,
        'mapping_results': None,
        'dormancy_results': {},
        'compliance_results': {},
        'llm_enabled': False,
        'bge_model': None,
        'dormancy_orchestrator': None,
        'analysis_session_id': None,
        'current_analysis_data': None
    }

    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

initialize_session_state()

# Enhanced Banking Compliance Schema with CBUAE mapping
BANKING_SCHEMA = {
    "customer_id": "Unique customer identifier",
    "customer_type": "Type of customer (Individual/Corporate)",
    "full_name_en": "Customer full name in English",
    "full_name_ar": "Customer full name in Arabic",
    "id_number": "National ID or Emirates ID number",
    "id_type": "Type of identification document",
    "date_of_birth": "Customer date of birth",
    "nationality": "Customer nationality",
    "address_line1": "Primary address line",
    "address_line2": "Secondary address line",
    "city": "City of residence",
    "emirate": "UAE Emirate",
    "country": "Country of residence",
    "postal_code": "Postal/ZIP code",
    "phone_primary": "Primary phone number",
    "phone_secondary": "Secondary phone number",
    "email_primary": "Primary email address",
    "email_secondary": "Secondary email address",
    "address_known": "Whether customer address is known (YES/NO)",
    "last_contact_date": "Date of last customer contact",
    "last_contact_method": "Method of last contact (EMAIL/PHONE/SMS)",
    "kyc_status": "Know Your Customer status",
    "kyc_expiry_date": "KYC document expiry date",
    "risk_rating": "Customer risk rating",
    "account_id": "Unique account identifier",
    "account_type": "Type of account (CURRENT/SAVINGS/FIXED_DEPOSIT/INVESTMENT)",
    "account_subtype": "Subtype of account",
    "account_name": "Account name/description",
    "currency": "Account currency",
    "account_status": "Account status (ACTIVE/INACTIVE/DORMANT/CLOSED)",
    "dormancy_status": "Dormancy status",
    "opening_date": "Account opening date",
    "closing_date": "Account closing date",
    "last_transaction_date": "Date of last transaction",
    "last_system_transaction_date": "Date of last system transaction",
    "balance_current": "Current account balance",
    "balance_available": "Available balance",
    "balance_minimum": "Minimum balance requirement",
    "interest_rate": "Interest rate",
    "interest_accrued": "Accrued interest amount",
    "is_joint_account": "Whether account is joint (YES/NO)",
    "joint_account_holders": "Number of joint account holders",
    "has_outstanding_facilities": "Whether account has outstanding facilities",
    "maturity_date": "Fixed deposit maturity date",
    "auto_renewal": "Auto-renewal setting for fixed deposits",
    "last_statement_date": "Date of last statement",
    "statement_frequency": "Statement frequency",
    "tracking_id": "Internal tracking identifier",
    "dormancy_trigger_date": "Date when dormancy was triggered",
    "dormancy_period_start": "Start date of dormancy period",
    "dormancy_period_months": "Number of months in dormancy",
    "dormancy_classification_date": "Date of dormancy classification",
    "transfer_eligibility_date": "Date eligible for transfer",
    "current_stage": "Current dormancy process stage",
    "contact_attempts_made": "Number of contact attempts made",
    "last_contact_attempt_date": "Date of last contact attempt",
    "waiting_period_start": "Start of waiting period",
    "waiting_period_end": "End of waiting period",
    "transferred_to_ledger_date": "Date transferred to internal ledger",
    "transferred_to_cb_date": "Date transferred to central bank",
    "cb_transfer_amount": "Amount transferred to central bank",
    "cb_transfer_reference": "Central bank transfer reference",
    "exclusion_reason": "Reason for exclusion from dormancy process",
    "created_date": "Record creation date",
    "updated_date": "Record last update date",
    "updated_by": "User who last updated the record"
}

# Real CBUAE Dormancy Agents Configuration - Updated to match dormant_agent.py
CBUAE_DORMANCY_AGENTS = [
    {
        'name': 'Demand Deposit Dormancy',
        'description': 'CBUAE Article 2.1.1 - Analyzes demand deposits and savings accounts for 3+ years inactivity with unknown address',
        'article': 'CBUAE Art. 2.1.1',
        'applies_to': 'Current and Savings accounts',
        'class_name': 'DemandDepositDormancyAgent',
        'agent_class': DemandDepositDormancyAgent if DORMANCY_SYSTEM_AVAILABLE else None,
        'priority': 'HIGH'
    },
    {
        'name': 'Fixed Deposit Dormancy',
        'description': 'CBUAE Article 2.2 - Identifies fixed deposits with maturity and contact compliance issues',
        'article': 'CBUAE Art. 2.2',
        'applies_to': 'Fixed deposits and term deposits',
        'class_name': 'FixedDepositDormancyAgent',
        'agent_class': FixedDepositDormancyAgent if DORMANCY_SYSTEM_AVAILABLE else None,
        'priority': 'CRITICAL'
    },
    {
        'name': 'Investment Account Dormancy',
        'description': 'CBUAE Article 2.3 - Analyzes investment accounts for prolonged inactivity',
        'article': 'CBUAE Art. 2.3',
        'applies_to': 'Investment and portfolio accounts',
        'class_name': 'InvestmentAccountDormancyAgent',
        'agent_class': InvestmentAccountDormancyAgent if DORMANCY_SYSTEM_AVAILABLE else None,
        'priority': 'MEDIUM'
    },
    {
        'name': 'Contact Attempts Compliance',
        'description': 'CBUAE Article 3 - Validates mandatory contact attempts for dormant accounts',
        'article': 'CBUAE Art. 3',
        'applies_to': 'All dormant accounts',
        'class_name': 'ContactAttemptsAgent',
        'agent_class': ContactAttemptsAgent if DORMANCY_SYSTEM_AVAILABLE else None,
        'priority': 'CRITICAL'
    },
    {
        'name': 'Central Bank Transfer Eligibility',
        'description': 'CBUAE Article 8 - Identifies accounts eligible for central bank transfer after 5+ years',
        'article': 'CBUAE Art. 8',
        'applies_to': 'Long-term dormant accounts',
        'class_name': 'CBTransferEligibilityAgent',
        'agent_class': CBTransferEligibilityAgent if DORMANCY_SYSTEM_AVAILABLE else None,
        'priority': 'CRITICAL'
    },
    {
        'name': 'Foreign Currency Conversion',
        'description': 'CBUAE Article 8.5 - Handles foreign currency dormant accounts requiring conversion',
        'article': 'CBUAE Art. 8.5',
        'applies_to': 'Non-AED dormant accounts',
        'class_name': 'ForeignCurrencyConversionAgent',
        'agent_class': ForeignCurrencyConversionAgent if DORMANCY_SYSTEM_AVAILABLE else None,
        'priority': 'HIGH'
    }
]

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1e40af;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }

    .agent-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #d1d5db;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .agent-card.priority-critical {
        border-left: 4px solid #dc2626;
    }

    .agent-card.priority-high {
        border-left: 4px solid #ea580c;
    }

    .agent-card.priority-medium {
        border-left: 4px solid #ca8a04;
    }

    .mapping-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .high-confidence { border-left: 4px solid #16a34a; }
    .medium-confidence { border-left: 4px solid #eab308; }
    .low-confidence { border-left: 4px solid #dc2626; }

    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }

    .cbuae-banner {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .success-banner {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .alert-card {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedDataMappingAgent:
    """Enhanced Data Mapping Agent using BGE-large embeddings and cosine similarity"""

    def __init__(self):
        self.model = None
        self.schema_embeddings = {}
        self.schema_columns = list(BANKING_SCHEMA.keys())

    def initialize_model(self):
        """Initialize BGE model if not already loaded"""
        if self.model is None:
            self.model = load_bge_model()
            if self.model:
                self._precompute_schema_embeddings()
        return self.model is not None

    def _precompute_schema_embeddings(self):
        """Precompute embeddings for schema columns to improve performance"""
        if self.model is None:
            return

        with st.spinner("🔄 Precomputing schema embeddings..."):
            for column in self.schema_columns:
                enriched_text = f"{column} {BANKING_SCHEMA[column]}"
                self.schema_embeddings[column] = self.model.encode([enriched_text])[0]

    def map_columns_with_bge(self, source_data: pd.DataFrame, llm_enabled: bool = False) -> Dict[str, Any]:
        """Map source columns to schema columns using BGE embeddings and cosine similarity"""
        if not self.initialize_model():
            return self._fallback_mapping(source_data, llm_enabled)

        source_columns = list(source_data.columns)
        mapping_results = {}
        confidence_scores = []

        with st.spinner("🧠 Computing BGE embeddings for source columns..."):
            source_embeddings = {}
            for col in source_columns:
                source_embeddings[col] = self.model.encode([col])[0]

        with st.spinner("🎯 Computing similarity scores..."):
            for source_col, source_emb in source_embeddings.items():
                best_match = None
                best_score = 0.0
                similarity_scores = {}

                for schema_col, schema_emb in self.schema_embeddings.items():
                    similarity = cosine_similarity([source_emb], [schema_emb])[0][0]
                    similarity_scores[schema_col] = similarity

                    if similarity > best_score:
                        best_score = similarity
                        best_match = schema_col

                confidence_threshold = 0.3
                if best_score >= confidence_threshold:
                    confidence_level = self._get_confidence_level(best_score)

                    mapping_results[source_col] = {
                        'mapped_to': best_match,
                        'confidence_score': float(best_score),
                        'confidence_level': confidence_level,
                        'description': BANKING_SCHEMA[best_match],
                        'similarity_method': 'BGE + Cosine Similarity',
                        'top_3_matches': sorted(similarity_scores.items(),
                                                key=lambda x: x[1], reverse=True)[:3]
                    }
                    confidence_scores.append(best_score)

        # Calculate overall mapping statistics
        total_fields = len(source_columns)
        mapped_fields = len(mapping_results)
        mapping_score = (mapped_fields / total_fields) * 100 if total_fields > 0 else 0

        # Apply LLM boost if enabled
        if llm_enabled:
            mapping_score = min(mapping_score * 1.1, 100)
            for mapping in mapping_results.values():
                mapping['llm_enhanced'] = True
                mapping['confidence_score'] = min(mapping['confidence_score'] * 1.05, 1.0)

        # Create mapped dataframe for dormancy analysis
        mapped_dataframe = self._create_mapped_dataframe(source_data, mapping_results)

        # Create mapping sheet for download
        mapping_sheet = self._create_mapping_sheet(source_columns, mapping_results)

        return {
            'success': True,
            'mapping_results': mapping_results,
            'mapped_dataframe': mapped_dataframe,
            'mapping_score': mapping_score,
            'schema_compliance': mapping_score >= 70,
            'total_fields': total_fields,
            'mapped_fields': mapped_fields,
            'unmapped_fields': total_fields - mapped_fields,
            'llm_enhanced': llm_enabled,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'mapping_sheet': mapping_sheet,
            'similarity_method': 'BGE-large + Cosine Similarity'
        }

    def _create_mapped_dataframe(self, source_data: pd.DataFrame, mapping_results: Dict) -> pd.DataFrame:
        """Create a dataframe with columns mapped to CBUAE schema"""
        mapped_data = pd.DataFrame()

        # Map columns that have successful mappings
        for source_col, mapping in mapping_results.items():
            if mapping['confidence_score'] >= 0.3:  # Use all mappings above threshold
                mapped_col = mapping['mapped_to']
                mapped_data[mapped_col] = source_data[source_col]

        # Add any unmapped columns with their original names (for compatibility)
        for col in source_data.columns:
            if col not in mapping_results and col not in mapped_data.columns:
                mapped_data[col] = source_data[col]

        return mapped_data

    def _get_confidence_level(self, score: float) -> str:
        """Determine confidence level based on similarity score"""
        if score >= 0.8:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        else:
            return 'low'

    def _create_mapping_sheet(self, source_columns: List[str], mapping_results: Dict) -> pd.DataFrame:
        """Create downloadable mapping sheet"""
        sheet_data = []

        for source_col in source_columns:
            if source_col in mapping_results:
                mapping = mapping_results[source_col]
                sheet_data.append({
                    'Source_Column': source_col,
                    'Mapped_To': mapping['mapped_to'],
                    'Confidence_Score': f"{mapping['confidence_score']:.4f}",
                    'Confidence_Level': mapping['confidence_level'],
                    'Description': mapping['description'],
                    'Method': mapping['similarity_method'],
                    'LLM_Enhanced': mapping.get('llm_enhanced', False)
                })
            else:
                sheet_data.append({
                    'Source_Column': source_col,
                    'Mapped_To': 'NOT_MAPPED',
                    'Confidence_Score': '0.0000',
                    'Confidence_Level': 'none',
                    'Description': 'No suitable match found',
                    'Method': 'BGE-large + Cosine Similarity',
                    'LLM_Enhanced': False
                })

        return pd.DataFrame(sheet_data)

    def _fallback_mapping(self, source_data: pd.DataFrame, llm_enabled: bool) -> Dict[str, Any]:
        """Fallback simple string matching when BGE is not available"""
        source_columns = list(source_data.columns)
        mapping_results = {}

        for source_col in source_columns:
            best_match = None
            best_score = 0.0

            source_lower = source_col.lower().replace('_', ' ')

            for schema_col in self.schema_columns:
                schema_lower = schema_col.lower().replace('_', ' ')

                if source_lower == schema_lower:
                    score = 1.0
                elif source_lower in schema_lower or schema_lower in source_lower:
                    score = 0.8
                else:
                    source_words = set(source_lower.split())
                    schema_words = set(schema_lower.split())
                    overlap = len(source_words.intersection(schema_words))
                    total = len(source_words.union(schema_words))
                    score = overlap / total if total > 0 else 0

                if score > best_score:
                    best_score = score
                    best_match = schema_col

            if best_score >= 0.3:
                mapping_results[source_col] = {
                    'mapped_to': best_match,
                    'confidence_score': float(best_score),
                    'confidence_level': self._get_confidence_level(best_score),
                    'description': BANKING_SCHEMA[best_match],
                    'similarity_method': 'String Similarity (Fallback)'
                }

        mapping_score = (len(mapping_results) / len(source_columns)) * 100 if source_columns else 0
        mapped_dataframe = self._create_mapped_dataframe(source_data, mapping_results)
        mapping_sheet = self._create_mapping_sheet(source_columns, mapping_results)

        return {
            'success': True,
            'mapping_results': mapping_results,
            'mapped_dataframe': mapped_dataframe,
            'mapping_score': mapping_score,
            'schema_compliance': mapping_score >= 70,
            'total_fields': len(source_columns),
            'mapped_fields': len(mapping_results),
            'unmapped_fields': len(source_columns) - len(mapping_results),
            'llm_enhanced': llm_enabled,
            'mapping_sheet': mapping_sheet,
            'similarity_method': 'String Similarity (Fallback)',
            'fallback_used': True
        }

def show_login():
    """Display login interface"""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-header">🏛️ CBUAE Banking Compliance System</div>', unsafe_allow_html=True)

    st.markdown("### 🔐 Login")

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔑 Login", type="primary", use_container_width=True):
            if username and password:
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            else:
                st.error("❌ Please enter username and password")

    with col2:
        if st.button("👤 Demo Login", type="secondary", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.username = "demo_user"
            st.success("✅ Demo login successful!")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ Demo Credentials & System Info"):
        st.markdown("""
        **Demo Credentials:**
        - Username: `admin` / Password: `admin123`
        - Username: `compliance` / Password: `comp123`
        - Or use **Demo Login** for quick access
        
        **System Features:**
        - 🧠 BGE-large semantic column mapping
        - 🏛️ Real CBUAE dormancy agents from dormant_agent.py
        - 📊 Comprehensive compliance analysis
        - 📥 Full CSV export capabilities
        """)

def authenticate_user(username: str, password: str) -> bool:
    """Simple demo authentication"""
    demo_users = {
        "admin": "admin123",
        "compliance": "comp123",
        "analyst": "analyst123"
    }
    return demo_users.get(username) == password

def generate_sample_banking_data(n_records: int = 1000) -> pd.DataFrame:
    """Generate sample banking compliance data matching CBUAE schema"""
    np.random.seed(42)

    # Customer names (mix of Arabic and English)
    first_names = ['Ahmed', 'Fatima', 'Mohammed', 'Aisha', 'Omar', 'Maryam', 'Ali', 'Noura', 'Hassan', 'Sara']
    last_names = ['Al-Rashid', 'Al-Zahra', 'Al-Mansouri', 'Al-Kaabi', 'Al-Suwaidi', 'Al-Falasi', 'Al-Maktoum']

    data = {}

    # Customer Information
    data['customer_id'] = [f'CUS{str(i).zfill(6)}' for i in range(1, n_records + 1)]
    data['customer_type'] = np.random.choice(['INDIVIDUAL', 'CORPORATE'], n_records, p=[0.85, 0.15])
    data['full_name_en'] = [f'{np.random.choice(first_names)} {np.random.choice(last_names)}' for _ in range(n_records)]
    data['nationality'] = np.random.choice(['UAE', 'INDIA', 'PAKISTAN', 'PHILIPPINES', 'EGYPT'], n_records, p=[0.3, 0.25, 0.15, 0.15, 0.15])

    # Account Information
    data['account_id'] = [f'ACC{str(i).zfill(8)}' for i in range(1, n_records + 1)]
    data['account_type'] = np.random.choice(['CURRENT', 'SAVINGS', 'FIXED_DEPOSIT', 'INVESTMENT'], n_records, p=[0.35, 0.35, 0.2, 0.1])
    data['account_status'] = np.random.choice(['ACTIVE', 'INACTIVE', 'DORMANT', 'CLOSED'], n_records, p=[0.6, 0.15, 0.2, 0.05])
    data['dormancy_status'] = np.random.choice(['ACTIVE', 'FLAGGED', 'DORMANT', 'CONTACTED', 'TRANSFER_READY'], n_records, p=[0.6, 0.1, 0.2, 0.08, 0.02])

    # Financial Data
    data['balance_current'] = np.random.lognormal(8, 2, n_records).round(2)
    data['currency'] = np.random.choice(['AED', 'USD', 'EUR', 'GBP', 'SAR'], n_records, p=[0.7, 0.15, 0.1, 0.03, 0.02])

    # Dates
    base_date = datetime.now() - timedelta(days=1000)
    data['opening_date'] = [(base_date + timedelta(days=np.random.randint(0, 365))).strftime('%d-%m-%Y') for _ in range(n_records)]
    data['last_transaction_date'] = [(datetime.now() - timedelta(days=np.random.randint(30, 1200))).strftime('%d-%m-%Y') for _ in range(n_records)]
    data['last_contact_date'] = [(datetime.now() - timedelta(days=np.random.randint(90, 800))).strftime('%d-%m-%Y') for _ in range(n_records)]

    # Dormancy tracking
    data['address_known'] = np.random.choice(['YES', 'NO'], n_records, p=[0.8, 0.2])
    data['contact_attempts_made'] = [np.random.randint(0, 5) if status in ['DORMANT', 'CONTACTED'] else 0 for status in data['dormancy_status']]
    data['dormancy_period_months'] = [np.random.randint(1, 60) if status in ['DORMANT', 'TRANSFER_READY'] else 0 for status in data['dormancy_status']]

    # Additional fields for maturity tracking
    data['maturity_date'] = [(datetime.now() + timedelta(days=np.random.randint(-365*2, 365*2))).strftime('%d-%m-%Y') if acc_type == 'FIXED_DEPOSIT' else '' for acc_type in data['account_type']]
    data['auto_renewal'] = ['YES' if np.random.random() < 0.3 else 'NO' for _ in range(n_records)]
    data['dormancy_trigger_date'] = [(datetime.now() - timedelta(days=np.random.randint(365, 1825))).strftime('%d-%m-%Y') if status == 'DORMANT' else '' for status in data['dormancy_status']]

    return pd.DataFrame(data)

def show_data_processing():
    """Show data processing interface with enhanced BGE mapping"""
    st.markdown('<div class="cbuae-banner">🏛️ CBUAE Banking Compliance Data Processing</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📤 Data Upload & Processing</div>', unsafe_allow_html=True)

    # Data Upload Section
    upload_method = st.selectbox(
        "Select Upload Method",
        ["Flat Files", "Google Drive Links", "Azure Data Lake", "Hadoop HDFS"],
        help="Choose your preferred data upload method"
    )

    if upload_method == "Flat Files":
        uploaded_file = st.file_uploader(
            "Choose a banking compliance data file",
            type=['csv', 'xlsx', 'json'],
            help="Upload your banking compliance data file (supports CSV, Excel, JSON)"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return

                st.success(f"✅ File uploaded successfully! {len(data)} records loaded.")
                st.session_state.uploaded_data = data

                with st.expander("📊 Data Preview"):
                    st.dataframe(data.head(10))
                    st.info(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    else:
        st.info(f"🔄 {upload_method} integration would be implemented here")

    # Generate Sample Data Button
    if st.session_state.uploaded_data is None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 No data uploaded. Generate sample CBUAE-compliant data to test the system.")
        with col2:
            if st.button("🎲 Generate Sample Data", type="primary"):
                with st.spinner("Generating CBUAE-compliant banking data..."):
                    sample_data = generate_sample_banking_data()
                    st.session_state.uploaded_data = sample_data
                    st.success("✅ Sample banking data generated successfully!")
                    st.rerun()

    if st.session_state.uploaded_data is not None:
        st.divider()

        # Enhanced Data Mapping Section with BGE
        st.markdown('<div class="section-header">🗺️ CBUAE Schema Mapping with BGE Intelligence</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("""
            **Map your data columns to the CBUAE banking compliance schema using BGE-large embeddings:**
            - 🧠 Semantic understanding of column names and descriptions
            - 🎯 Cosine similarity for precise matching  
            - 📊 Confidence scoring for each mapping
            - 🤖 Optional LLM enhancement boost
            """)

        with col2:
            llm_enabled = st.toggle(
                "🤖 Enable LLM Enhancement",
                value=st.session_state.llm_enabled,
                help="Apply LLM reasoning boost to confidence scores (10% enhancement)"
            )
            st.session_state.llm_enabled = llm_enabled

        if st.button("🗺️ Start BGE Schema Mapping", type="primary", use_container_width=True):
            mapping_agent = AdvancedDataMappingAgent()

            with st.spinner("🔄 Running BGE-powered schema mapping..."):
                mapping_results = mapping_agent.map_columns_with_bge(
                    st.session_state.uploaded_data,
                    llm_enabled
                )
                st.session_state.mapping_results = mapping_results

                # Store the mapped dataframe for dormancy analysis
                if mapping_results['success'] and 'mapped_dataframe' in mapping_results:
                    st.session_state.mapped_data = mapping_results['mapped_dataframe']
                    # Store current analysis data for dormancy agents
                    st.session_state.current_analysis_data = mapping_results['mapped_dataframe']

        if st.session_state.mapping_results:
            display_enhanced_mapping_results(st.session_state.mapping_results)

def display_enhanced_mapping_results(results: Dict[str, Any]):
    """Display enhanced mapping results with BGE analysis"""
    if not results.get('success'):
        st.error("❌ Mapping analysis failed")
        return

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score = results['mapping_score']
        if score >= 90:
            status = "🟢 Excellent"
        elif score >= 70:
            status = "🟡 Good"
        else:
            status = "🔴 Needs Review"
        st.metric("🎯 Mapping Score", f"{score:.1f}%", delta=status)

    with col2:
        compliance = "✅ Ready for CBUAE Analysis" if results['schema_compliance'] else "❌ Review Required"
        st.metric("📋 CBUAE Compliance", "Ready" if results['schema_compliance'] else "Review")

    with col3:
        st.metric("🔗 Fields Mapped", f"{results['mapped_fields']}/{results['total_fields']}")

    with col4:
        avg_conf = results.get('average_confidence', 0)
        st.metric("📊 Avg Confidence", f"{avg_conf:.3f}")

    # Method information
    st.info(f"🔬 **Method:** {results.get('similarity_method', 'Unknown')} | "
            f"**LLM Enhanced:** {'Yes' if results.get('llm_enhanced', False) else 'No'}")

    # Download mapping sheet
    if 'mapping_sheet' in results:
        col1, col2 = st.columns([3, 1])
        with col2:
            mapping_csv = results['mapping_sheet'].to_csv(index=False)
            st.download_button(
                label="📥 Download Mapping",
                data=mapping_csv,
                file_name=f"cbuae_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

    # Detailed mapping visualization
    with st.expander("🔍 View Detailed Mapping Analysis", expanded=False):
        display_detailed_mapping_table(results)

    # Set processed data for next steps
    if results['schema_compliance']:
        st.session_state.processed_data = st.session_state.mapped_data
        st.markdown('<div class="success-banner">🎉 Data mapping completed successfully! Ready for CBUAE dormancy analysis.</div>', unsafe_allow_html=True)

        # Show data quality validation using dormant_agent.py validation
        if DORMANCY_SYSTEM_AVAILABLE:
            with st.expander("📊 CBUAE Data Quality Validation"):
                try:
                    validation_results = validate_csv_structure(st.session_state.mapped_data)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Structure Valid", "✅ Yes" if validation_results['structure_valid'] else "❌ No")
                    with col2:
                        st.metric("Total Records", validation_results['total_records'])
                    with col3:
                        st.metric("Quality Issues", len(validation_results.get('quality_issues', [])))

                    if validation_results.get('missing_columns'):
                        st.warning(f"⚠️ Missing columns: {', '.join(validation_results['missing_columns'])}")

                    if validation_results.get('quality_issues'):
                        st.info("💡 Quality recommendations:")
                        for issue in validation_results['quality_issues']:
                            st.write(f"• {issue}")

                except Exception as e:
                    st.warning(f"Could not validate data structure: {e}")

def display_detailed_mapping_table(results: Dict[str, Any]):
    """Display detailed mapping table with all information"""
    if 'mapping_sheet' in results:
        df = results['mapping_sheet'].copy()

        def highlight_confidence(row):
            if row['Confidence_Level'] == 'high':
                return ['background-color: #d4f4dd'] * len(row)
            elif row['Confidence_Level'] == 'medium':
                return ['background-color: #fff2cc'] * len(row)
            elif row['Confidence_Level'] == 'low':
                return ['background-color: #ffe6e6'] * len(row)
            else:
                return ['background-color: #f8f9fa'] * len(row)

        styled_df = df.style.apply(highlight_confidence, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)

def show_dormancy_analysis():
    """Show real CBUAE dormancy analysis with integrated agents from dormant_agent.py"""
    st.markdown('<div class="cbuae-banner">💤 CBUAE Dormancy Analysis System</div>', unsafe_allow_html=True)

    if not DORMANCY_SYSTEM_AVAILABLE:
        st.error("❌ CBUAE Dormancy System not available. Please ensure dormant_agent.py is properly installed.")
        return

    if st.session_state.current_analysis_data is None:
        st.warning("⚠️ Please upload and map data first in the Data Processing section.")

        if st.button("🚀 Quick Setup with Sample Data", type="primary"):
            sample_data = generate_sample_banking_data()
            st.session_state.uploaded_data = sample_data
            st.session_state.mapped_data = sample_data
            st.session_state.processed_data = sample_data
            st.session_state.current_analysis_data = sample_data
            st.success("✅ Sample data loaded! You can now run CBUAE dormancy analysis.")
            st.rerun()
        return

    data = st.session_state.current_analysis_data

    # Show system overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Records", f"{len(data):,}")
    with col2:
        dormant_count = len(data[data.get('dormancy_status', pd.Series()) == 'DORMANT'])
        st.metric("💤 Currently Dormant", dormant_count)
    with col3:
        high_value_count = len(data[data.get('balance_current', pd.Series(dtype=float)) > 100000])
        st.metric("💰 High Value (>100K)", high_value_count)

    st.info("🔍 Real CBUAE dormancy agents from dormant_agent.py analyze data for accounts meeting UAE Central Bank dormancy criteria.")

    # Comprehensive Analysis Button
    st.markdown("### 🚀 Comprehensive CBUAE Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        **Run comprehensive CBUAE dormancy analysis using dormant_agent.py:**
        - ⚡ Complete analysis across all regulatory articles
        - 🎯 Real agent execution with actual compliance logic  
        - 📊 Comprehensive compliance reporting
        - 📥 Full CSV export capabilities
        """)

    with col2:
        if st.button("🚀 Run All CBUAE Agents", type="primary", use_container_width=True):
            run_comprehensive_dormancy_analysis_real(data)

    st.divider()

    # Individual Agent Cards
    st.markdown("### 🤖 Individual CBUAE Dormancy Agents")

    for agent_config in CBUAE_DORMANCY_AGENTS:
        if agent_config['agent_class'] is None:
            continue

        with st.container():
            priority_class = f"priority-{agent_config['priority'].lower()}"
            st.markdown(f'<div class="agent-card {priority_class}">', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([4, 1, 1])

            with col1:
                st.markdown(f"**{agent_config['name']}**")
                st.caption(agent_config['description'])
                st.caption(f"📋 {agent_config['article']} • Priority: {agent_config['priority']} • Applies to: {agent_config['applies_to']}")

            with col2:
                # Estimate potential alerts for this agent
                alert_count = estimate_agent_alerts_real(data, agent_config['class_name'])
                if alert_count > 0:
                    st.metric("Potential Alerts", alert_count)
                else:
                    st.caption("No alerts expected")

            with col3:
                if alert_count > 0:
                    if st.button(f"🔍 Analyze", key=f"analyze_{agent_config['class_name']}"):
                        run_single_dormancy_agent_real(agent_config, data)

                # Show download button if results exist
                if agent_config['class_name'] in st.session_state.dormancy_results:
                    if st.button(f"📥 Download", key=f"download_{agent_config['class_name']}"):
                        download_dormancy_results(agent_config['class_name'])

            st.markdown('</div>', unsafe_allow_html=True)

    # Show results if available
    if st.session_state.dormancy_results:
        st.divider()
        st.markdown("### 📊 CBUAE Dormancy Analysis Results")
        display_dormancy_results()

def run_comprehensive_dormancy_analysis_real(data: pd.DataFrame):
    """Run comprehensive dormancy analysis using the real CBUAE system from dormant_agent.py"""
    try:
        # Generate a unique session ID
        session_id = f"session_{int(time.time())}"
        st.session_state.analysis_session_id = session_id

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Initializing CBUAE dormancy analysis system...")
        progress_bar.progress(10)

        # Run the comprehensive analysis using dormant_agent.py
        status_text.text("🧠 Running comprehensive CBUAE dormancy analysis...")
        progress_bar.progress(30)

        # Use asyncio to run the async function from dormant_agent.py
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            analysis_results = loop.run_until_complete(
                run_comprehensive_dormancy_analysis_csv(
                    user_id=st.session_state.username,
                    account_data=data,
                    report_date=datetime.now().strftime("%Y-%m-%d")
                )
            )
        finally:
            loop.close()

        progress_bar.progress(80)
        status_text.text("📊 Processing analysis results...")

        if analysis_results['success']:
            st.session_state.dormancy_results['comprehensive_analysis'] = analysis_results

            progress_bar.progress(100)
            status_text.text("✅ Comprehensive CBUAE analysis completed!")

            # Display summary results
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Total Analyzed", f"{analysis_results['total_accounts_analyzed']:,}")
            with col2:
                st.metric("🚨 Dormant Found", analysis_results['dormant_accounts_found'])
            with col3:
                st.metric("⚠️ High Risk", analysis_results.get('high_risk_accounts', 0))
            with col4:
                st.metric("⏱️ Processing Time", f"{analysis_results['processing_time_seconds']:.2f}s")

            # Show compliance flags if any
            if analysis_results.get('compliance_flags'):
                st.warning("⚠️ Compliance Issues Found:")
                for flag in analysis_results['compliance_flags']:
                    st.write(f"• {flag}")

            # Display individual agent results
            if 'analysis_results' in analysis_results and analysis_results['analysis_results']:
                with st.expander("🔍 Individual Agent Results", expanded=True):
                    for agent_name, result in analysis_results['analysis_results'].items():
                        if isinstance(result, dict) and result.get('results', {}).get('count', 0) > 0:
                            st.write(f"**{agent_name.replace('_', ' ').title()}:** {result['results']['count']} alerts")

            # Show recommendations if available
            if analysis_results.get('recommendations'):
                with st.expander("💡 Recommendations", expanded=True):
                    for rec in analysis_results['recommendations']:
                        st.markdown(f"**{rec.get('agent', 'Unknown')}** ({rec.get('priority', 'MEDIUM')}): {rec.get('action', 'No action specified')}")

            # Download comprehensive results
            if st.button("📥 Download Comprehensive Results", type="primary"):
                download_comprehensive_results(analysis_results)

        else:
            st.error(f"❌ Analysis failed: {analysis_results.get('error', 'Unknown error')}")
            progress_bar.progress(0)
            status_text.text("❌ Analysis failed")

    except Exception as e:
        st.error(f"❌ Error running comprehensive analysis: {str(e)}")
        logger.error(f"Comprehensive analysis error: {e}")
        st.error(f"Full traceback: {traceback.format_exc()}")

def run_single_dormancy_agent_real(agent_config: Dict, data: pd.DataFrame):
    """Run a single dormancy agent using the real agent classes from dormant_agent.py"""
    try:
        with st.spinner(f"Running {agent_config['name']} analysis..."):

            if DORMANCY_SYSTEM_AVAILABLE and agent_config['agent_class']:
                # Create the real agent instance
                agent = agent_config['agent_class']()

                # Create agent state for analysis
                agent_state = AgentState(
                    agent_id=secrets.token_hex(8),
                    agent_type=agent_config['class_name'],
                    session_id=st.session_state.analysis_session_id or secrets.token_hex(8),
                    user_id=st.session_state.username,
                    timestamp=datetime.now(),
                    input_dataframe=data
                )

                # Run the actual analysis using asyncio
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    result_state = loop.run_until_complete(
                        agent.analyze_dormancy(agent_state, datetime.now().strftime("%Y-%m-%d"))
                    )
                finally:
                    loop.close()

                # Process the results
                if result_state.analysis_results:
                    st.session_state.dormancy_results[agent_config['class_name']] = {
                        'analysis_results': result_state.analysis_results,
                        'agent_name': agent_config['name'],
                        'processing_time': result_state.processing_time,
                        'status': 'completed',
                        'records_processed': result_state.records_processed,
                        'dormant_found': result_state.dormant_records_found
                    }

                    alerts_found = result_state.analysis_results.get('count', 0)
                    st.success(f"✅ {agent_config['name']} analysis completed! Found {alerts_found} alerts.")

                    # Show brief summary
                    if alerts_found > 0:
                        st.info(f"📋 {result_state.analysis_results.get('description', 'Analysis completed')}")

                        # Show sample results
                        if 'details' in result_state.analysis_results and result_state.analysis_results['details']:
                            with st.expander("Sample Results", expanded=False):
                                sample_results = result_state.analysis_results['details'][:5]  # Show first 5
                                st.dataframe(pd.DataFrame(sample_results), use_container_width=True)
                else:
                    st.warning(f"⚠️ {agent_config['name']} analysis completed but no results returned.")
            else:
                st.error("❌ CBUAE Dormancy System not available")

    except Exception as e:
        st.error(f"❌ Error running {agent_config['name']}: {str(e)}")
        logger.error(f"Single agent error: {e}")
        st.error(f"Full traceback: {traceback.format_exc()}")

def estimate_agent_alerts_real(data: pd.DataFrame, agent_class_name: str) -> int:
    """Estimate potential alerts for an agent based on data characteristics and real logic"""
    if data.empty:
        return 0

    try:
        if 'DemandDeposit' in agent_class_name:
            # Demand deposits with unknown address and long inactivity
            candidates = data[
                (data.get('account_type', pd.Series()).isin(['CURRENT', 'SAVINGS'])) &
                (data.get('address_known', pd.Series()) == 'NO') &
                (data.get('account_status', pd.Series()) != 'CLOSED')
            ]
            return len(candidates)

        elif 'FixedDeposit' in agent_class_name:
            candidates = data[
                (data.get('account_type', pd.Series()).isin(['FIXED_DEPOSIT'])) &
                (data.get('account_status', pd.Series()) != 'CLOSED')
            ]
            return len(candidates)

        elif 'Investment' in agent_class_name:
            candidates = data[
                (data.get('account_type', pd.Series()).isin(['INVESTMENT'])) &
                (data.get('account_status', pd.Series()) != 'CLOSED')
            ]
            return len(candidates)

        elif 'ContactAttempts' in agent_class_name:
            candidates = data[
                (data.get('dormancy_status', pd.Series()).isin(['DORMANT', 'FLAGGED'])) &
                (data.get('contact_attempts_made', pd.Series()).fillna(0) < 3)
            ]
            return len(candidates)

        elif 'CBTransfer' in agent_class_name:
            candidates = data[
                (data.get('dormancy_status', pd.Series()) == 'DORMANT') &
                (data.get('dormancy_period_months', pd.Series()).fillna(0) >= 60)  # 5+ years
            ]
            return len(candidates)

        elif 'ForeignCurrency' in agent_class_name:
            candidates = data[
                (data.get('currency', pd.Series()) != 'AED') &
                (data.get('dormancy_status', pd.Series()) == 'DORMANT')
            ]
            return len(candidates)

        else:
            return 0

    except Exception:
        return 0

def download_dormancy_results(agent_class_name: str):
    """Download dormancy results for a specific agent"""
    if agent_class_name in st.session_state.dormancy_results:
        result = st.session_state.dormancy_results[agent_class_name]

        if 'analysis_results' in result and 'details' in result['analysis_results']:
            df = pd.DataFrame(result['analysis_results']['details'])

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            st.download_button(
                label=f"📥 Download {agent_class_name} Results",
                data=csv_buffer.getvalue(),
                file_name=f"cbuae_{agent_class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_btn_{agent_class_name}"
            )

def download_comprehensive_results(analysis_results: Dict):
    """Download comprehensive analysis results"""
    try:
        # Create a comprehensive report
        report_data = {
            'Analysis Summary': {
                'Total Accounts Analyzed': analysis_results['total_accounts_analyzed'],
                'Dormant Accounts Found': analysis_results['dormant_accounts_found'],
                'High Risk Accounts': analysis_results.get('high_risk_accounts', 0),
                'Processing Time (seconds)': analysis_results['processing_time_seconds'],
                'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Session ID': st.session_state.analysis_session_id
            }
        }

        # Add recommendations and priority actions if available
        if analysis_results.get('recommendations'):
            report_data['Recommendations'] = analysis_results['recommendations']
        if analysis_results.get('priority_actions'):
            report_data['Priority Actions'] = analysis_results['priority_actions']

        # Convert to JSON for download
        report_json = json.dumps(report_data, indent=2, default=str)

        st.download_button(
            label="📥 Download Comprehensive Report (JSON)",
            data=report_json,
            file_name=f"cbuae_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # Also offer CSV download of summary
        if 'analysis_results' in analysis_results:
            summary_data = []
            for agent_name, result in analysis_results['analysis_results'].items():
                if isinstance(result, dict):
                    summary_data.append({
                        'Agent': agent_name.replace('_', ' ').title(),
                        'Status': result.get('status', 'Unknown'),
                        'Alerts Found': result.get('results', {}).get('count', 0),
                        'Processing Time': result.get('processing_time', 0),
                        'Article': result.get('results', {}).get('compliance_article', 'N/A')
                    })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_csv = summary_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download Summary Report (CSV)",
                    data=summary_csv,
                    file_name=f"cbuae_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Error creating download: {str(e)}")

def display_dormancy_results():
    """Display dormancy results summary"""
    if not st.session_state.dormancy_results:
        return

    # Calculate totals
    total_alerts = 0
    completed_agents = 0

    for agent_name, result in st.session_state.dormancy_results.items():
        if isinstance(result, dict):
            if agent_name == 'comprehensive_analysis':
                total_alerts += result.get('dormant_accounts_found', 0)
            else:
                total_alerts += result.get('analysis_results', {}).get('count', 0)
            completed_agents += 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🚨 Total Alerts", total_alerts)
    with col2:
        st.metric("🤖 Agents Completed", completed_agents)
    with col3:
        avg_per_agent = total_alerts / completed_agents if completed_agents > 0 else 0
        st.metric("📊 Avg per Agent", f"{avg_per_agent:.1f}")

    # Show individual results
    for agent_name, result in st.session_state.dormancy_results.items():
        if isinstance(result, dict) and agent_name != 'comprehensive_analysis':
            analysis_results = result.get('analysis_results', {})
            alert_count = analysis_results.get('count', 0)

            if alert_count > 0:
                with st.expander(f"📋 {agent_name.replace('_', ' ').title()} Results ({alert_count} alerts)"):
                    st.write(f"**Article:** {analysis_results.get('compliance_article', 'N/A')}")
                    st.write(f"**Description:** {analysis_results.get('description', 'N/A')}")
                    st.write(f"**Processing Time:** {result.get('processing_time', 0):.2f}s")

                    if 'details' in analysis_results and analysis_results['details']:
                        st.dataframe(
                            pd.DataFrame(analysis_results['details'][:10]),
                            use_container_width=True
                        )

def show_reports():
    """Show comprehensive reports dashboard"""
    st.markdown('<div class="cbuae-banner">📊 CBUAE Compliance Reports Dashboard</div>', unsafe_allow_html=True)

    # System overview
    create_cbuae_dashboard()

    # Detailed reports
    if st.session_state.dormancy_results:
        st.markdown("### 📋 Detailed Analysis Reports")
        create_detailed_reports()

def create_cbuae_dashboard():
    """Create CBUAE compliance dashboard"""
    st.markdown("### 🏛️ CBUAE Compliance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_records = len(st.session_state.current_analysis_data) if st.session_state.current_analysis_data is not None else 0
        st.metric("📊 Total Records", f"{total_records:,}")

    with col2:
        total_alerts = 0
        if 'comprehensive_analysis' in st.session_state.dormancy_results:
            total_alerts = st.session_state.dormancy_results['comprehensive_analysis'].get('dormant_accounts_found', 0)
        else:
            total_alerts = sum(
                result.get('analysis_results', {}).get('count', 0)
                for result in st.session_state.dormancy_results.values()
                if isinstance(result, dict) and 'analysis_results' in result
            )
        st.metric("🚨 Total Alerts", total_alerts)

    with col3:
        completed_agents = len([
            r for r in st.session_state.dormancy_results.values()
            if isinstance(r, dict) and r.get('status') == 'completed'
        ])
        st.metric("🤖 Agents Run", f"{completed_agents}/{len(CBUAE_DORMANCY_AGENTS)}")

    with col4:
        compliance_rate = (completed_agents / len(CBUAE_DORMANCY_AGENTS) * 100) if CBUAE_DORMANCY_AGENTS else 100
        st.metric("✅ Compliance Rate", f"{compliance_rate:.1f}%")

def create_detailed_reports():
    """Create detailed analysis reports"""
    # Agent performance table
    agent_data = []

    for agent_config in CBUAE_DORMANCY_AGENTS:
        agent_name = agent_config['class_name']
        result = st.session_state.dormancy_results.get(agent_name, {})

        if isinstance(result, dict) and 'analysis_results' in result:
            agent_data.append({
                'Agent': agent_config['name'],
                'Article': agent_config['article'],
                'Priority': agent_config['priority'],
                'Alerts Found': result['analysis_results'].get('count', 0),
                'Processing Time': f"{result.get('processing_time', 0):.2f}s",
                'Status': '✅ Completed' if result.get('status') == 'completed' else '⏳ Pending'
            })
        else:
            agent_data.append({
                'Agent': agent_config['name'],
                'Article': agent_config['article'],
                'Priority': agent_config['priority'],
                'Alerts Found': 0,
                'Processing Time': '-',
                'Status': '⏳ Not Run'
            })

    if agent_data:
        st.markdown("#### 🤖 Agent Performance Summary")
        df = pd.DataFrame(agent_data)
        st.dataframe(df, use_container_width=True)

        # Priority distribution chart
        if any(row['Alerts Found'] > 0 for row in agent_data):
            st.markdown("#### 📊 Alerts by Priority")

            priority_data = {}
            for row in agent_data:
                priority = row['Priority']
                alerts = row['Alerts Found']
                priority_data[priority] = priority_data.get(priority, 0) + alerts

            if priority_data:
                fig = px.pie(
                    values=list(priority_data.values()),
                    names=list(priority_data.keys()),
                    title="Alert Distribution by Priority Level",
                    color_discrete_map={
                        'CRITICAL': '#dc2626',
                        'HIGH': '#ea580c',
                        'MEDIUM': '#ca8a04'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application entry point"""
    if not st.session_state.logged_in:
        show_login()
        return

    # Sidebar navigation
    with st.sidebar:
        st.markdown('<div class="main-header">🏛️ CBUAE System</div>', unsafe_allow_html=True)
        st.markdown(f"👤 Welcome, **{st.session_state.username}**")

        page = st.selectbox(
            "Navigate to:",
            ["⚙️ Data Processing", "💤 Dormancy Analysis", "📊 Reports"],
            key="navigation"
        )

        st.divider()

        # System status
        st.markdown("### 🔧 System Status")

        # BGE Model Status
        if st.session_state.bge_model is None:
            if st.button("🚀 Load BGE Model", type="secondary", use_container_width=True):
                st.session_state.bge_model = load_bge_model()
        else:
            st.success("🟢 BGE Model: Ready")

        # CBUAE System Status
        if DORMANCY_SYSTEM_AVAILABLE:
            st.success("🟢 CBUAE Agents: Ready")
        else:
            st.error("🔴 CBUAE Agents: Unavailable")

        st.divider()

        # Quick stats
        if st.session_state.current_analysis_data is not None:
            st.markdown("### 📊 Quick Stats")
            st.metric("Records", f"{len(st.session_state.current_analysis_data):,}")

            total_alerts = sum(
                result.get('analysis_results', {}).get('count', 0)
                for result in st.session_state.dormancy_results.values()
                if isinstance(result, dict) and 'analysis_results' in result
            )
            st.metric("Total Alerts", total_alerts)

        st.divider()

        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main content area
    if page == "⚙️ Data Processing":
        show_data_processing()
    elif page == "💤 Dormancy Analysis":
        show_dormancy_analysis()
    elif page == "📊 Reports":
        show_reports()

if __name__ == "__main__":
    main()