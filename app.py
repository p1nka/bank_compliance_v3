"""
Enhanced Banking Compliance Multi-Agent System with CBUAE Dormancy Monitoring
Complete Streamlit Application with Intelligent Agent Invocation
"""

import streamlit as st

# Configure page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="CBUAE Banking Compliance System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
import io
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import secrets
import hashlib
from pathlib import Path
import logging

# Check for advanced features
try:
    import faiss
    import sentence_transformers
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False

# Check for dormancy agents with proper error handling
try:
    import langgraph
    from dormant_agents import (
        DormancyMonitoringAgents,
        DormancyAlert,
        DormancyReportingEngine,
        DormancyNotificationService,
        initialize_dormancy_monitoring_system,
        run_daily_dormancy_monitoring
    )
    DORMANCY_AGENTS_AVAILABLE = True
except ImportError as e:
    DORMANCY_AGENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #fafafa;
    }
    .workflow-step {
        text-align: center;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .workflow-step:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    .alert-high {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .alert-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .alert-low {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .step-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Intelligent CBUAE Agent Manager
class CBUAEIntelligentAgentManager:
    """Intelligent CBUAE agent manager that only invokes relevant agents"""

    def __init__(self):
        self.agent_definitions = {
            'article_2_1_demand_deposits': {
                'name': 'Article 2.1 - Demand Deposit Dormancy Monitor',
                'criteria': 'Demand deposits with no customer-initiated activity for 3+ years',
                'article_reference': 'CBUAE Article 2.1.1',
                'account_types': ['Current', 'Savings', 'Call'],
                'dormancy_threshold_days': 1095  # 3 years
            },
            'article_2_2_fixed_deposits': {
                'name': 'Article 2.2 - Fixed/Term Deposit Dormancy Monitor',
                'criteria': 'Fixed deposits unclaimed after maturity + 3 years',
                'article_reference': 'CBUAE Article 2.2',
                'account_types': ['Fixed', 'Term'],
                'dormancy_threshold_days': 1095
            },
            'article_2_3_investments': {
                'name': 'Article 2.3 - Investment Account Dormancy Monitor',
                'criteria': 'Investment accounts with no activity for 3+ years',
                'article_reference': 'CBUAE Article 2.3',
                'account_types': ['Investment'],
                'dormancy_threshold_days': 1095
            },
            'article_2_4_unclaimed_instruments': {
                'name': 'Article 2.4 - Unclaimed Payment Instruments Monitor',
                'criteria': 'Payment instruments unclaimed for 3+ years',
                'article_reference': 'CBUAE Article 2.4',
                'account_types': ['All'],
                'dormancy_threshold_days': 1095
            },
            'article_2_6_safe_deposit_boxes': {
                'name': 'Article 2.6 - Safe Deposit Box Dormancy Monitor',
                'criteria': 'SDB with outstanding charges for 3+ years',
                'article_reference': 'CBUAE Article 2.6',
                'account_types': ['All'],
                'dormancy_threshold_days': 1095
            },
            'article_3_contact_obligations': {
                'name': 'Article 3 - Bank Contact Obligations Monitor',
                'criteria': 'Dormant accounts requiring mandatory contact attempts',
                'article_reference': 'CBUAE Article 3.1',
                'account_types': ['All'],
                'dormancy_threshold_days': 1095
            },
            'article_8_cb_transfers': {
                'name': 'Article 8 - Central Bank Transfer Monitor',
                'criteria': 'Accounts eligible for CB transfer (5+ years dormant)',
                'article_reference': 'CBUAE Article 8.1',
                'account_types': ['All'],
                'dormancy_threshold_days': 1825  # 5 years
            }
        }

        self.article_guidance = {
            'article_2_1_demand_deposits': {
                'title': 'Article 2.1 - Demand Deposit Dormancy Compliance',
                'summary': 'Current and savings accounts with no customer-initiated activity for 3+ years are considered dormant.',
                'obligations': [
                    'Monitor demand deposit accounts for 3-year inactivity period',
                    'Exclude bank-initiated transactions from activity calculations',
                    'Apply dormancy classification uniformly across all demand deposits'
                ],
                'next_steps': [
                    'Review identified dormant demand deposit accounts',
                    'Verify last customer-initiated transaction dates',
                    'Proceed to Article 3 contact obligations for confirmed dormant accounts',
                    'Update account status in core banking system'
                ],
                'regulatory_timeline': '30 days to complete dormancy classification review'
            },
            'article_2_2_fixed_deposits': {
                'title': 'Article 2.2 - Fixed/Term Deposit Dormancy Compliance',
                'summary': 'Fixed deposits unclaimed for 3+ years after maturity are considered dormant.',
                'obligations': [
                    'Monitor fixed deposits post-maturity for 3-year unclaimed period',
                    'Track customer communication regarding maturity notifications',
                    'Consider auto-renewal instructions and their validity'
                ],
                'next_steps': [
                    'Review maturity dates and post-maturity periods',
                    'Verify customer notification attempts regarding maturity',
                    'Assess auto-renewal terms and customer instructions',
                    'Initiate Article 3 contact procedures for unclaimed deposits'
                ],
                'regulatory_timeline': '30 days to review and classify unclaimed fixed deposits'
            },
            'article_2_3_investments': {
                'title': 'Article 2.3 - Investment Account Dormancy Compliance',
                'summary': 'Investment accounts with no activity for 3+ years are considered dormant.',
                'obligations': [
                    'Monitor investment account activity including trading and communications',
                    'Consider dividend collection and portfolio adjustments as activity',
                    'Apply dormancy rules to all investment vehicles consistently'
                ],
                'next_steps': [
                    'Review investment account activity logs',
                    'Verify customer engagement with investment services',
                    'Check for automated investment instructions',
                    'Implement Article 3 contact procedures for dormant investment accounts'
                ],
                'regulatory_timeline': '45 days for investment account dormancy review'
            },
            'article_2_4_unclaimed_instruments': {
                'title': 'Article 2.4 - Unclaimed Payment Instruments Compliance',
                'summary': 'Payment instruments (drafts, checks, etc.) unclaimed for 3+ years require special handling.',
                'obligations': [
                    'Track all outstanding payment instruments by issue date',
                    'Monitor customer collection attempts and communications',
                    'Maintain detailed records of instrument status'
                ],
                'next_steps': [
                    'Compile list of outstanding payment instruments over 3 years',
                    'Verify customer notification attempts',
                    'Assess instrument validity and legal constraints',
                    'Prepare for potential Central Bank transfer procedures'
                ],
                'regulatory_timeline': '60 days to complete instrument review and classification'
            },
            'article_2_6_safe_deposit_boxes': {
                'title': 'Article 2.6 - Safe Deposit Box Dormancy Compliance',
                'summary': 'Safe deposit boxes with outstanding charges for 3+ years require dormancy procedures.',
                'obligations': [
                    'Monitor safe deposit box rental charge payment status',
                    'Track customer communication regarding outstanding charges',
                    'Maintain detailed access logs and payment histories'
                ],
                'next_steps': [
                    'Review safe deposit box payment histories',
                    'Verify customer contact attempts regarding outstanding charges',
                    'Assess legal requirements for box access procedures',
                    'Initiate court proceedings if required for box opening'
                ],
                'regulatory_timeline': '90 days to complete SDB dormancy procedures'
            },
            'article_3_contact_obligations': {
                'title': 'Article 3 - Bank Contact Obligations Compliance',
                'summary': 'Banks must attempt to contact dormant account holders through multiple channels.',
                'obligations': [
                    'Attempt contact through last known address via registered mail',
                    'Use alternative contact methods (phone, email) if available',
                    'Document all contact attempts with dates and methods',
                    'Allow reasonable time for customer response'
                ],
                'next_steps': [
                    'Prepare contact attempt strategy for each dormant account',
                    'Send registered mail to last known addresses',
                    'Attempt alternative contact methods (phone, email)',
                    'Document all communication attempts in customer records',
                    'Wait 60 days for customer response before proceeding',
                    'Prepare accounts for internal ledger transfer if no response'
                ],
                'regulatory_timeline': '120 days to complete all contact obligations'
            },
            'article_8_cb_transfers': {
                'title': 'Article 8 - Central Bank Transfer Compliance',
                'summary': 'Accounts dormant for 5+ years with unsuccessful contact must be transferred to CBUAE.',
                'obligations': [
                    'Identify accounts dormant for 5+ years',
                    'Verify completion of Article 3 contact obligations',
                    'Prepare detailed transfer documentation',
                    'Convert foreign currency balances to AED'
                ],
                'next_steps': [
                    'Compile list of accounts eligible for CB transfer',
                    'Verify all contact obligations have been completed',
                    'Prepare CBUAE transfer forms and documentation',
                    'Convert foreign currency balances to AED at current rates',
                    'Submit transfer requests to CBUAE within required timeframes',
                    'Maintain transfer records for audit purposes'
                ],
                'regulatory_timeline': '30 days from eligibility determination to CBUAE transfer'
            }
        }

    def analyze_account_data(self, uploaded_data: Dict) -> Dict:
        """Analyze uploaded data to determine which agents should be invoked"""

        analysis_results = {
            'total_accounts_analyzed': 0,
            'agent_eligibility': {},
            'account_breakdown': {},
            'analysis_summary': {}
        }

        try:
            # Extract all accounts from uploaded data
            all_accounts = []
            for file_name, file_data in uploaded_data.items():
                if isinstance(file_data, dict) and 'accounts' in file_data:
                    accounts = file_data['accounts']
                    all_accounts.extend(accounts)

            analysis_results['total_accounts_analyzed'] = len(all_accounts)

            # Analyze each agent's eligibility
            for agent_id, agent_config in self.agent_definitions.items():
                eligible_accounts = self._find_eligible_accounts(
                    all_accounts, agent_id, agent_config
                )

                analysis_results['agent_eligibility'][agent_id] = {
                    'should_invoke': len(eligible_accounts) > 0,
                    'eligible_account_count': len(eligible_accounts),
                    'agent_name': agent_config['name'],
                    'criteria': agent_config['criteria'],
                    'eligible_accounts': eligible_accounts[:10]  # First 10 for preview
                }

            # Generate account breakdown by type
            account_types = {}
            for account in all_accounts:
                acc_type = account.get('Account_Type', account.get('account_type', 'Unknown'))
                account_types[acc_type] = account_types.get(acc_type, 0) + 1

            analysis_results['account_breakdown'] = account_types

            # Generate analysis summary
            total_eligible_agents = sum(
                1 for agent_data in analysis_results['agent_eligibility'].values()
                if agent_data['should_invoke']
            )

            analysis_results['analysis_summary'] = {
                'total_agents_available': len(self.agent_definitions),
                'agents_with_eligible_accounts': total_eligible_agents,
                'agents_to_skip': len(self.agent_definitions) - total_eligible_agents
            }

        except Exception as e:
            analysis_results['error'] = str(e)

        return analysis_results

    def _find_eligible_accounts(self, accounts: List[Dict], agent_id: str, agent_config: Dict) -> List[Dict]:
        """Find accounts eligible for a specific agent"""

        eligible_accounts = []

        for account in accounts:
            if self._is_account_eligible_for_agent(account, agent_id, agent_config):
                eligible_accounts.append(account)

        return eligible_accounts

    def _is_account_eligible_for_agent(self, account: Dict, agent_id: str, agent_config: Dict) -> bool:
        """Check if an account is eligible for a specific agent"""

        try:
            # Get account type
            account_type = account.get('Account_Type', account.get('account_type', 'Unknown'))

            # Check if account type matches agent criteria
            if agent_config['account_types'] != ['All']:
                if account_type not in agent_config['account_types']:
                    return False

            # Get last activity date
            last_activity_field = account.get('Date_Last_Cust_Initiated_Activity',
                                            account.get('last_activity'))

            if not last_activity_field:
                return False

            # Parse date
            if isinstance(last_activity_field, str):
                last_activity = datetime.strptime(last_activity_field, '%Y-%m-%d').date()
            else:
                last_activity = last_activity_field

            # Calculate days inactive
            today = datetime.now().date()
            days_inactive = (today - last_activity).days

            # Agent-specific eligibility checks
            if agent_id == 'article_2_1_demand_deposits':
                return (account_type in ['Current', 'Savings', 'Call'] and
                       days_inactive >= agent_config['dormancy_threshold_days'])

            elif agent_id == 'article_2_2_fixed_deposits':
                if account_type not in ['Fixed', 'Term']:
                    return False
                # Check if matured and unclaimed
                maturity_date_field = account.get('FTD_Maturity_Date')
                if maturity_date_field:
                    if isinstance(maturity_date_field, str):
                        maturity_date = datetime.strptime(maturity_date_field, '%Y-%m-%d').date()
                    else:
                        maturity_date = maturity_date_field

                    days_since_maturity = (today - maturity_date).days
                    return days_since_maturity >= agent_config['dormancy_threshold_days']
                return False

            elif agent_id == 'article_2_3_investments':
                return (account_type == 'Investment' and
                       days_inactive >= agent_config['dormancy_threshold_days'])

            elif agent_id == 'article_2_4_unclaimed_instruments':
                # Check for unclaimed payment instruments
                unclaimed_trigger = account.get('Unclaimed_Item_Trigger_Date')
                if unclaimed_trigger:
                    if isinstance(unclaimed_trigger, str):
                        trigger_date = datetime.strptime(unclaimed_trigger, '%Y-%m-%d').date()
                    else:
                        trigger_date = unclaimed_trigger

                    days_since_trigger = (today - trigger_date).days
                    return days_since_trigger >= agent_config['dormancy_threshold_days']
                return False

            elif agent_id == 'article_2_6_safe_deposit_boxes':
                # Check for SDB with outstanding charges
                sdb_charges = account.get('SDB_Charges_Outstanding')
                if sdb_charges and float(sdb_charges) > 0:
                    charges_date_field = account.get('Date_SDB_Charges_Became_Outstanding')
                    if charges_date_field:
                        if isinstance(charges_date_field, str):
                            charges_date = datetime.strptime(charges_date_field, '%Y-%m-%d').date()
                        else:
                            charges_date = charges_date_field

                        days_since_charges = (today - charges_date).days
                        return days_since_charges >= agent_config['dormancy_threshold_days']
                return False

            elif agent_id == 'article_3_contact_obligations':
                # Any dormant account needs contact obligations
                is_dormant = account.get('Expected_Account_Dormant', 'no').lower() == 'yes'
                return is_dormant and days_inactive >= agent_config['dormancy_threshold_days']

            elif agent_id == 'article_8_cb_transfers':
                # 5+ years dormant with completed contact attempts
                if days_inactive >= agent_config['dormancy_threshold_days']:  # 5 years
                    contact_attempted = account.get('Bank_Contact_Attempted_Post_Dormancy_Trigger', 'no')
                    return contact_attempted.lower() == 'yes'
                return False

            return False

        except Exception:
            return False

    def invoke_eligible_agents(self, analysis_results: Dict) -> Dict:
        """Invoke only the agents that have eligible accounts"""

        invocation_results = {
            'agents_invoked': [],
            'agents_skipped': [],
            'total_alerts_generated': 0,
            'article_specific_results': {},
            'summary': {}
        }

        try:
            for agent_id, eligibility_data in analysis_results['agent_eligibility'].items():
                if eligibility_data['should_invoke']:
                    # Invoke the agent
                    agent_result = self._invoke_agent(agent_id, eligibility_data)
                    invocation_results['agents_invoked'].append(agent_result)
                    invocation_results['article_specific_results'][agent_id] = agent_result
                    invocation_results['total_alerts_generated'] += agent_result['alerts_generated']
                else:
                    # Skip the agent
                    skip_result = {
                        'agent_id': agent_id,
                        'agent_name': eligibility_data['agent_name'],
                        'reason_skipped': 'No eligible accounts found',
                        'eligible_account_count': 0
                    }
                    invocation_results['agents_skipped'].append(skip_result)

            # Generate summary
            invocation_results['summary'] = {
                'total_agents_available': len(self.agent_definitions),
                'agents_invoked_count': len(invocation_results['agents_invoked']),
                'agents_skipped_count': len(invocation_results['agents_skipped']),
                'total_alerts': invocation_results['total_alerts_generated'],
                'execution_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            invocation_results['error'] = str(e)

        return invocation_results

    def _invoke_agent(self, agent_id: str, eligibility_data: Dict) -> Dict:
        """Simulate invoking a specific agent and return results"""

        agent_config = self.agent_definitions[agent_id]
        eligible_count = eligibility_data['eligible_account_count']

        # Simulate agent processing
        alerts_generated = min(eligible_count, np.random.randint(1, eligible_count + 1))

        # Generate priority distribution
        high_priority = max(1, int(alerts_generated * 0.2))
        medium_priority = max(1, int(alerts_generated * 0.5))
        low_priority = alerts_generated - high_priority - medium_priority

        return {
            'agent_id': agent_id,
            'agent_name': agent_config['name'],
            'article_reference': agent_config['article_reference'],
            'eligible_accounts_processed': eligible_count,
            'alerts_generated': alerts_generated,
            'alert_breakdown': {
                'high_priority': high_priority,
                'medium_priority': medium_priority,
                'low_priority': low_priority
            },
            'execution_status': 'completed',
            'processing_time_seconds': np.random.uniform(2.5, 8.0)
        }

    def generate_article_specific_guidance(self, invocation_results: Dict) -> Dict:
        """Generate detailed guidance only for articles that were actually invoked"""

        guidance_results = {
            'applicable_articles': [],
            'action_items': [],
            'regulatory_timeline': {},
            'compliance_checklist': {}
        }

        try:
            for agent_result in invocation_results['agents_invoked']:
                agent_id = agent_result['agent_id']

                if agent_id in self.article_guidance:
                    article_guidance = self.article_guidance[agent_id]

                    guidance_item = {
                        'agent_id': agent_id,
                        'article_reference': agent_result['article_reference'],
                        'title': article_guidance['title'],
                        'summary': article_guidance['summary'],
                        'alerts_count': agent_result['alerts_generated'],
                        'priority_breakdown': agent_result['alert_breakdown'],
                        'obligations': article_guidance['obligations'],
                        'next_steps': article_guidance['next_steps'],
                        'regulatory_timeline': article_guidance['regulatory_timeline']
                    }

                    guidance_results['applicable_articles'].append(guidance_item)

                    # Generate specific action items
                    for step in article_guidance['next_steps']:
                        action_item = {
                            'article': agent_result['article_reference'],
                            'action': step,
                            'affected_accounts': agent_result['eligible_accounts_processed'],
                            'priority': 'HIGH' if agent_result['alert_breakdown']['high_priority'] > 0 else 'MEDIUM',
                            'timeline': article_guidance['regulatory_timeline']
                        }
                        guidance_results['action_items'].append(action_item)

            # Sort action items by priority and affected accounts
            guidance_results['action_items'].sort(
                key=lambda x: (x['priority'] == 'HIGH', x['affected_accounts']),
                reverse=True
            )

        except Exception as e:
            guidance_results['error'] = str(e)

        return guidance_results

# Simple Login Manager
class SecureLoginManager:
    def __init__(self):
        self.users = {
            "admin": {"password": "Admin123!", "role": "admin", "user_id": 1},
            "analyst": {"password": "Analyst123!", "role": "analyst", "user_id": 2},
            "compliance": {"password": "Compliance123!", "role": "compliance_officer", "user_id": 3},
            "manager": {"password": "Manager123!", "role": "manager", "user_id": 4}
        }
        self.sessions = {}

    def create_user(self, username, password, role):
        if username in self.users:
            raise ValueError("User already exists")
        self.users[username] = {"password": password, "role": role, "user_id": len(self.users) + 1}
        return True

    def authenticate_user(self, username, password, client_ip=None):
        if username in self.users and self.users[username]["password"] == password:
            return {
                "username": username,
                "role": self.users[username]["role"],
                "user_id": self.users[username]["user_id"],
                "authenticated": True
            }
        raise ValueError("Invalid credentials")

    def create_secure_session(self, user_data, session_data=None):
        token = secrets.token_hex(32)
        self.sessions[token] = {
            "user_data": user_data,
            "created_at": datetime.now(),
            "session_data": session_data or {}
        }
        return token

    def validate_session(self, token):
        if token in self.sessions:
            session = self.sessions[token]
            expires_at = session["created_at"] + timedelta(hours=8)
            return {
                "session_valid": True,
                "expires_at": expires_at.isoformat(),
                "user_data": session["user_data"]
            }
        raise ValueError("Invalid session")

    def logout_user(self, token):
        return self.sessions.pop(token, None) is not None

# Mock Database Connection for Demonstration
class MockDatabaseConnection:
    def __init__(self):
        self.mock_mode = True
        self.data = {}

    def execute(self, query, params=None):
        """Mock database execution"""
        return []

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""

    session_vars = {
        'login_manager': SecureLoginManager(),
        'authenticated': False,
        'user_data': None,
        'session_token': None,
        'db_connection': MockDatabaseConnection(),
        'uploaded_data': None,
        'processed_data': None,
        'intelligent_agent_manager': None,
        'agent_eligibility_analysis': None,
        'agent_invocation_results': None,
        'article_guidance_results': None,
        'monitoring_results': None,
        'dashboard_data': None
    }

    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize session state
initialize_session_state()

# Login Interface
def show_login():
    """Display enhanced login interface"""

    st.markdown('<div class="main-header">🏦 CBUAE Banking Compliance System</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header" style="font-size: 1.2rem;">Secure Authentication Required</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        # Login tabs
        tab1, tab2 = st.tabs(["🔐 Login", "👤 Register"])

        with tab1:
            st.subheader("Login to Your Account")

            # Demo credentials info
            with st.expander("🔑 Demo Credentials"):
                st.write("**Available Demo Accounts:**")
                st.write("• **admin** / Admin123! (Administrator)")
                st.write("• **analyst** / Analyst123! (Data Analyst)")
                st.write("• **compliance** / Compliance123! (Compliance Officer)")
                st.write("• **manager** / Manager123! (Manager)")

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                remember_me = st.checkbox("Remember me")

                login_button = st.form_submit_button("🔑 Login", use_container_width=True)

                if login_button:
                    if username and password:
                        try:
                            user_data = st.session_state.login_manager.authenticate_user(
                                username, password, st.session_state.get('client_ip', '127.0.0.1')
                            )

                            # Create secure session
                            session_token = st.session_state.login_manager.create_secure_session(
                                user_data,
                                {"workspace": "banking_compliance", "permissions": ["read", "write"]}
                            )

                            # Update session state
                            st.session_state.authenticated = True
                            st.session_state.user_data = user_data
                            st.session_state.session_token = session_token

                            st.success(f"Welcome back, {user_data['username']}!")
                            st.rerun()

                        except ValueError as e:
                            st.error(f"Login failed: {str(e)}")
                    else:
                        st.warning("Please enter both username and password")

        with tab2:
            st.subheader("Create New Account")

            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                role = st.selectbox("Role", ["analyst", "compliance_officer", "manager", "admin"])

                register_button = st.form_submit_button("📝 Register", use_container_width=True)

                if register_button:
                    if new_username and new_password and confirm_password:
                        if new_password == confirm_password:
                            if len(new_password) >= 8:
                                try:
                                    success = st.session_state.login_manager.create_user(
                                        new_username, new_password, role
                                    )
                                    if success:
                                        st.success("Account created successfully! Please login.")
                                except ValueError as e:
                                    st.error(f"Registration failed: {str(e)}")
                            else:
                                st.error("Password must be at least 8 characters long")
                        else:
                            st.error("Passwords do not match")
                    else:
                        st.warning("Please fill in all fields")

        st.markdown('</div>', unsafe_allow_html=True)

# Data Upload Interface (keeping existing functionality)
def show_data_upload():
    """Display enhanced data upload interface"""

    st.header("📁 Data Upload & Processing")
    st.markdown("Upload your banking data files for CBUAE compliance analysis:")

    # Upload method selection
    upload_method = st.radio(
        "Select Upload Method:",
        ["📤 Direct File Upload", "🔄 Generate Sample Data", "🗄️ Database Connection"],
        horizontal=True
    )

    uploaded_data = None
    data_source_info = {}

    if upload_method == "📤 Direct File Upload":
        uploaded_data, data_source_info = handle_direct_upload()
    elif upload_method == "🔄 Generate Sample Data":
        uploaded_data, data_source_info = handle_sample_data_generation()
    elif upload_method == "🗄️ Database Connection":
        uploaded_data, data_source_info = handle_database_connection()

    return uploaded_data, data_source_info

def handle_direct_upload():
    """Handle direct file upload"""

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Direct File Upload")

    # File uploader with multiple formats
    uploaded_files = st.file_uploader(
        "Choose your banking data files",
        type=['csv', 'xlsx', 'xls', 'json'],
        accept_multiple_files=True,
        help="Supported formats: CSV, Excel (XLSX/XLS), JSON"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")

        # Display file information
        for file in uploaded_files:
            st.write(f"📄 **{file.name}** ({file.size:,} bytes)")

        # Process files
        processed_data = {}
        for file in uploaded_files:
            try:
                # Process file based on type
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                elif file.name.endswith('.json'):
                    json_data = json.load(file)
                    df = pd.json_normalize(json_data) if isinstance(json_data, list) else pd.DataFrame([json_data])
                else:
                    st.error(f"Unsupported file format: {file.name}")
                    continue

                processed_data[file.name] = {
                    "accounts": df.to_dict('records'),
                    "metadata": {
                        "record_count": len(df),
                        "column_count": len(df.columns),
                        "columns": list(df.columns),
                        "file_name": file.name,
                        "file_size": file.size
                    }
                }
                st.success(f"✅ {file.name} processed successfully - {len(df)} records")

                # Show preview
                with st.expander(f"Preview: {file.name}"):
                    st.dataframe(df.head(10))

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

        return processed_data, {
            "source": "direct_upload",
            "files": [f.name for f in uploaded_files],
            "total_files": len(uploaded_files)
        }

    st.markdown('</div>', unsafe_allow_html=True)
    return None, {}

def handle_sample_data_generation():
    """Handle sample data generation"""

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("🔄 Generate Sample Banking Data")

    col1, col2 = st.columns(2)

    with col1:
        num_records = st.slider("Number of Records", 100, 5000, 1000, 100)
        include_dormant = st.checkbox("Include Dormant Accounts", value=True)

    with col2:
        data_quality = st.selectbox("Data Quality", ["High", "Medium", "Low"])
        include_cbuae_fields = st.checkbox("Include CBUAE Fields", value=True)

    if st.button("🚀 Generate Sample Data", type="primary", use_container_width=True):
        with st.spinner("Generating sample banking data..."):
            mock_data = generate_mock_banking_data(
                num_records, include_dormant, data_quality, include_cbuae_fields
            )

            st.success(f"✅ Generated {len(mock_data)} sample records!")

            # Show preview
            st.subheader("📊 Data Preview")
            st.dataframe(mock_data.head(10))

            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(mock_data))
            with col2:
                dormant_count = len(mock_data[mock_data['Expected_Account_Dormant'] == 'yes'])
                st.metric("Dormant Accounts", dormant_count)
            with col3:
                avg_balance = mock_data['Current_Balance'].mean()
                st.metric("Avg Balance", f"${avg_balance:,.2f}")
            with col4:
                account_types = mock_data['Account_Type'].nunique()
                st.metric("Account Types", account_types)

        st.markdown('</div>', unsafe_allow_html=True)

        return {
            "sample_data.csv": {
                "accounts": mock_data.to_dict('records'),
                "metadata": {
                    "record_count": len(mock_data),
                    "column_count": len(mock_data.columns),
                    "columns": list(mock_data.columns),
                    "file_name": "sample_data.csv",
                    "generated": True
                }
            }
        }, {
            "source": "sample_generation",
            "records": num_records,
            "quality": data_quality
        }

    st.markdown('</div>', unsafe_allow_html=True)
    return None, {}

def handle_database_connection():
    """Handle database connection"""

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("🗄️ Database Connection")

    st.info("🚧 Database integration available in mock mode for demonstration.")

    if st.button("🔗 Connect to Mock Database"):
        # Generate mock database data
        mock_data = generate_database_mock_data()
        st.success("✅ Connected to mock database successfully!")
        st.dataframe(mock_data.head(10))

        st.markdown('</div>', unsafe_allow_html=True)

        return {
            "database_export.csv": {
                "accounts": mock_data.to_dict('records'),
                "metadata": {
                    "record_count": len(mock_data),
                    "column_count": len(mock_data.columns),
                    "columns": list(mock_data.columns),
                    "file_name": "database_export.csv",
                    "source": "database"
                }
            }
        }, {"source": "database", "record_count": len(mock_data)}

    st.markdown('</div>', unsafe_allow_html=True)
    return None, {}

def generate_mock_banking_data(num_records: int, include_dormant: bool = True,
                              data_quality: str = "High", include_cbuae_fields: bool = True) -> pd.DataFrame:
    """Generate realistic mock banking data for CBUAE compliance testing"""

    np.random.seed(42)  # For reproducible results

    # Account types based on CBUAE regulations
    account_types = ['Current', 'Savings', 'Call', 'Fixed', 'Term', 'Investment']

    # Generate realistic data
    data = []
    for i in range(num_records):
        # Generate account ID in CBUAE format
        account_id = f"ACC{str(i+1).zfill(8)}"
        customer_id = f"CUST{str(i+1).zfill(6)}"

        # Random account type
        account_type = np.random.choice(account_types)

        # Generate realistic balance based on account type
        if account_type in ['Current', 'Savings']:
            balance = np.random.lognormal(8, 1.5)  # Log-normal for realistic distribution
        elif account_type in ['Fixed', 'Term']:
            balance = np.random.lognormal(10, 1)  # Higher balances for term deposits
        else:
            balance = np.random.lognormal(9, 2)

        # Generate activity dates
        last_activity = datetime.now() - timedelta(days=np.random.randint(0, 2500))
        last_communication = last_activity - timedelta(days=np.random.randint(0, 180))

        # Determine if account is dormant (simplified logic)
        days_inactive = (datetime.now() - last_activity).days
        is_dormant = days_inactive > 1095 if include_dormant else False  # 3 years

        # Base record with CBUAE standard fields
        record = {
            'Account_ID': account_id,
            'Customer_ID': customer_id,
            'Account_Type': account_type,
            'Current_Balance': round(balance, 2),
            'Date_Last_Cust_Initiated_Activity': last_activity.strftime('%Y-%m-%d'),
            'Date_Last_Customer_Communication_Any_Type': last_communication.strftime('%Y-%m-%d'),
            'Customer_Has_Active_Liability_Account': np.random.choice(['yes', 'no'], p=[0.3, 0.7]),
            'Expected_Account_Dormant': 'yes' if is_dormant else 'no',
            'Customer_Address_Known': np.random.choice(['yes', 'no'], p=[0.8, 0.2]),
        }

        # Add CBUAE specific fields if requested
        if include_cbuae_fields:
            # Fixed/Term Deposit specific fields
            if account_type in ['Fixed', 'Term']:
                record['FTD_Maturity_Date'] = (last_activity + timedelta(days=365)).strftime('%Y-%m-%d')
                record['FTD_Auto_Renewal'] = np.random.choice(['yes', 'no'])
                record['Date_Last_FTD_Renewal_Claim_Request'] = last_communication.strftime('%Y-%m-%d')

            # Investment account specific fields
            if account_type == 'Investment':
                record['Inv_Maturity_Redemption_Date'] = (last_activity + timedelta(days=730)).strftime('%Y-%m-%d')

            # Safe Deposit Box fields (randomly assigned)
            if np.random.random() < 0.1:  # 10% chance of having SDB
                record['SDB_Charges_Outstanding'] = round(np.random.uniform(100, 5000), 2)
                record['Date_SDB_Charges_Became_Outstanding'] = last_communication.strftime('%Y-%m-%d')
                record['SDB_Tenant_Communication_Received'] = np.random.choice(['yes', 'no'])

            # Unclaimed items (randomly assigned)
            if np.random.random() < 0.05:  # 5% chance
                record['Unclaimed_Item_Trigger_Date'] = last_activity.strftime('%Y-%m-%d')
                record['Unclaimed_Item_Amount'] = round(np.random.uniform(500, 10000), 2)

            # Bank contact fields
            if is_dormant:
                record['Bank_Contact_Attempted_Post_Dormancy_Trigger'] = np.random.choice(['yes', 'no'])
                record['Date_Last_Bank_Contact_Attempt'] = (
                    last_activity + timedelta(days=np.random.randint(30, 180))
                ).strftime('%Y-%m-%d')

        # Add data quality issues if requested
        if data_quality == "Medium":
            # Introduce some missing values
            if np.random.random() < 0.05:
                record['Customer_Address_Known'] = None
        elif data_quality == "Low":
            # Introduce more missing values and inconsistencies
            if np.random.random() < 0.1:
                record['Customer_Address_Known'] = None
            if np.random.random() < 0.05:
                record['Date_Last_Customer_Communication_Any_Type'] = None

        data.append(record)

    return pd.DataFrame(data)

def generate_database_mock_data():
    """Generate mock data that simulates database export"""

    return generate_mock_banking_data(
        num_records=2000,
        include_dormant=True,
        data_quality="High",
        include_cbuae_fields=True
    )

# Enhanced CBUAE Intelligent Dormancy Analysis Interface
async def show_dormancy_analysis():
    """Enhanced dormancy analysis interface with intelligent agent invocation"""

    st.header("🏦 CBUAE Intelligent Dormancy Analysis")
    st.markdown("*Advanced system that only invokes agents with eligible accounts and provides article-specific guidance*")

    if not hasattr(st.session_state, 'uploaded_data') or not st.session_state.uploaded_data:
        st.warning("⚠️ No data uploaded. Please upload banking data first.")
        return

    # Initialize intelligent agent manager
    if st.session_state.intelligent_agent_manager is None:
        st.session_state.intelligent_agent_manager = CBUAEIntelligentAgentManager()

    agent_manager = st.session_state.intelligent_agent_manager

    # Configuration section
    with st.expander("🔧 Analysis Configuration"):
        col1, col2 = st.columns(2)

        with col1:
            analysis_date = st.date_input(
                "Analysis Date",
                value=datetime.now().date(),
                help="Date for dormancy calculations"
            )

            include_preview = st.checkbox(
                "Include Account Previews",
                value=True,
                help="Show sample eligible accounts for each agent"
            )

        with col2:
            high_value_threshold = st.number_input(
                "High Value Alert Threshold (AED)",
                min_value=1000, max_value=1000000, value=25000,
                help="Balance threshold for high-priority alerts"
            )

            strict_mode = st.checkbox(
                "Strict CBUAE Compliance Mode",
                value=True,
                help="Apply strictest interpretation of CBUAE regulations"
            )

    # Step indicator
    st.markdown('<div class="step-indicator">Step 1: Account Eligibility Analysis</div>', unsafe_allow_html=True)

    # Step 1: Analyze account data to determine agent eligibility
    if st.button("🔍 Analyze Account Eligibility", type="primary", use_container_width=True):

        with st.spinner("🔄 Analyzing account data to determine agent eligibility..."):
            analysis_results = agent_manager.analyze_account_data(st.session_state.uploaded_data)
            st.session_state.agent_eligibility_analysis = analysis_results

        if 'error' in analysis_results:
            st.error(f"❌ Analysis failed: {analysis_results['error']}")
            return

        st.success("✅ Account eligibility analysis completed!")

        # Display analysis summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Accounts", analysis_results['total_accounts_analyzed'])

        with col2:
            agents_to_invoke = analysis_results['analysis_summary']['agents_with_eligible_accounts']
            st.metric("Agents to Invoke", agents_to_invoke)

        with col3:
            agents_to_skip = analysis_results['analysis_summary']['agents_to_skip']
            st.metric("Agents to Skip", agents_to_skip)

        with col4:
            total_eligible = sum(
                data['eligible_account_count']
                for data in analysis_results['agent_eligibility'].values()
            )
            st.metric("Total Eligible Accounts", total_eligible)

        # Display detailed agent eligibility
        st.subheader("📋 Agent Eligibility Breakdown")

        # Agents to invoke
        agents_to_invoke = [
            (agent_id, data) for agent_id, data in analysis_results['agent_eligibility'].items()
            if data['should_invoke']
        ]

        if agents_to_invoke:
            st.markdown("#### ✅ Agents to Invoke (Have Eligible Accounts)")

            for agent_id, data in agents_to_invoke:
                with st.expander(f"🎯 {data['agent_name']} ({data['eligible_account_count']} accounts)"):
                    st.write(f"**Criteria:** {data['criteria']}")
                    st.write(f"**Eligible Accounts:** {data['eligible_account_count']}")

                    if include_preview and data['eligible_accounts']:
                        st.write("**Sample Eligible Accounts:**")
                        preview_df = pd.DataFrame(data['eligible_accounts'][:5])
                        st.dataframe(preview_df, use_container_width=True)

        # Agents to skip
        agents_to_skip = [
            (agent_id, data) for agent_id, data in analysis_results['agent_eligibility'].items()
            if not data['should_invoke']
        ]

        if agents_to_skip:
            st.markdown("#### ⏭️ Agents to Skip (No Eligible Accounts)")

            skip_info = []
            for agent_id, data in agents_to_skip:
                skip_info.append({
                    "Agent": data['agent_name'],
                    "Reason": "No accounts meet criteria",
                    "Criteria": data['criteria']
                })

            skip_df = pd.DataFrame(skip_info)
            st.dataframe(skip_df, use_container_width=True)

    # Step 2: Invoke only eligible agents
    if hasattr(st.session_state, 'agent_eligibility_analysis') and st.session_state.agent_eligibility_analysis:

        st.markdown('<div class="step-indicator">Step 2: Intelligent Agent Invocation</div>', unsafe_allow_html=True)

        if st.button("🚀 Invoke Eligible Agents Only", type="primary", use_container_width=True):

            with st.spinner("🔄 Invoking only agents with eligible accounts..."):
                invocation_results = agent_manager.invoke_eligible_agents(
                    st.session_state.agent_eligibility_analysis
                )
                st.session_state.agent_invocation_results = invocation_results

            if 'error' in invocation_results:
                st.error(f"❌ Agent invocation failed: {invocation_results['error']}")
                return

            st.success("✅ Intelligent agent invocation completed!")

            # Display invocation summary
            summary = invocation_results['summary']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Agents Invoked", summary['agents_invoked_count'])

            with col2:
                st.metric("Agents Skipped", summary['agents_skipped_count'])

            with col3:
                st.metric("Total Alerts", summary['total_alerts'])

            with col4:
                efficiency = (summary['agents_invoked_count'] / summary['total_agents_available']) * 100
                st.metric("Efficiency", f"{efficiency:.1f}%")

            # Display invoked agent results
            if invocation_results['agents_invoked']:
                st.subheader("🎯 Invoked Agent Results")

                for agent_result in invocation_results['agents_invoked']:
                    with st.expander(f"📊 {agent_result['agent_name']} - {agent_result['alerts_generated']} alerts"):

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Article Reference:** {agent_result['article_reference']}")
                            st.write(f"**Accounts Processed:** {agent_result['eligible_accounts_processed']}")
                            st.write(f"**Processing Time:** {agent_result['processing_time_seconds']:.1f}s")

                        with col2:
                            breakdown = agent_result['alert_breakdown']
                            st.write("**Alert Priority Breakdown:**")
                            st.write(f"🔴 High Priority: {breakdown['high_priority']}")
                            st.write(f"🟡 Medium Priority: {breakdown['medium_priority']}")
                            st.write(f"🟢 Low Priority: {breakdown['low_priority']}")

    # Step 3: Generate article-specific guidance
    if hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results:

        st.markdown('<div class="step-indicator">Step 3: Article-Specific Guidance Generation</div>', unsafe_allow_html=True)

        if st.button("📚 Generate Article-Specific Guidance", type="primary", use_container_width=True):

            with st.spinner("📖 Generating guidance for invoked articles only..."):
                guidance_results = agent_manager.generate_article_specific_guidance(
                    st.session_state.agent_invocation_results
                )
                st.session_state.article_guidance_results = guidance_results

            if 'error' in guidance_results:
                st.error(f"❌ Guidance generation failed: {guidance_results['error']}")
                return

            st.success("✅ Article-specific guidance generated!")

            # Display applicable articles
            if guidance_results['applicable_articles']:
                st.subheader("⚖️ Applicable CBUAE Articles & Guidance")

                for article_guidance in guidance_results['applicable_articles']:

                    # Article header with alert counts
                    st.markdown(f"### {article_guidance['title']}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Alerts", article_guidance['alerts_count'])
                    with col2:
                        high_priority = article_guidance['priority_breakdown']['high_priority']
                        st.metric("High Priority", high_priority)
                    with col3:
                        medium_priority = article_guidance['priority_breakdown']['medium_priority']
                        st.metric("Medium Priority", medium_priority)

                    # Article summary and obligations
                    st.markdown(f"**Summary:** {article_guidance['summary']}")

                    # Regulatory obligations
                    st.markdown("**📋 Regulatory Obligations:**")
                    for obligation in article_guidance['obligations']:
                        st.write(f"• {obligation}")

                    # Next steps
                    st.markdown("**🚀 Required Next Steps:**")
                    for i, step in enumerate(article_guidance['next_steps'], 1):
                        st.write(f"{i}. {step}")

                    # Regulatory timeline
                    st.markdown(f"**⏰ Regulatory Timeline:** {article_guidance['regulatory_timeline']}")

                    st.markdown("---")

                # Priority action items
                if guidance_results['action_items']:
                    st.subheader("🚨 Priority Action Items")

                    # Group by priority
                    high_priority_actions = [
                        item for item in guidance_results['action_items']
                        if item['priority'] == 'HIGH'
                    ]

                    medium_priority_actions = [
                        item for item in guidance_results['action_items']
                        if item['priority'] == 'MEDIUM'
                    ]

                    if high_priority_actions:
                        st.markdown("#### 🔴 HIGH PRIORITY ACTIONS")
                        for action in high_priority_actions:
                            st.markdown(f"""
                            <div class="alert-high">
                                <strong>Article:</strong> {action['article']}<br>
                                <strong>Action:</strong> {action['action']}<br>
                                <strong>Affected Accounts:</strong> {action['affected_accounts']}<br>
                                <strong>Timeline:</strong> {action['timeline']}
                            </div>
                            """, unsafe_allow_html=True)

                    if medium_priority_actions:
                        st.markdown("#### 🟡 MEDIUM PRIORITY ACTIONS")
                        for action in medium_priority_actions[:5]:  # Show top 5
                            st.markdown(f"""
                            <div class="alert-medium">
                                <strong>Article:</strong> {action['article']}<br>
                                <strong>Action:</strong> {action['action']}<br>
                                <strong>Affected Accounts:</strong> {action['affected_accounts']}<br>
                                <strong>Timeline:</strong> {action['timeline']}
                            </div>
                            """, unsafe_allow_html=True)

            else:
                st.info("ℹ️ No articles were invoked - no eligible accounts found for any CBUAE dormancy criteria.")

    # Export results
    if hasattr(st.session_state, 'article_guidance_results') and st.session_state.article_guidance_results:
        st.subheader("📥 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Export Guidance Report", use_container_width=True):
                # Compile comprehensive report
                report_data = {
                    'analysis_date': datetime.now().isoformat(),
                    'agent_eligibility': st.session_state.agent_eligibility_analysis,
                    'invocation_results': st.session_state.agent_invocation_results,
                    'article_guidance': st.session_state.article_guidance_results
                }

                st.download_button(
                    label="💾 Download Complete Report",
                    data=json.dumps(report_data, indent=2, default=str),
                    file_name=f"cbuae_intelligent_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

        with col2:
            if st.button("📋 Export Action Items", use_container_width=True):
                if st.session_state.article_guidance_results.get('action_items'):
                    action_df = pd.DataFrame(st.session_state.article_guidance_results['action_items'])
                    csv_data = action_df.to_csv(index=False)

                    st.download_button(
                        label="💾 Download Action Items CSV",
                        data=csv_data,
                        file_name=f"cbuae_action_items_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

        with col3:
            if st.button("📊 Export Summary", use_container_width=True):
                summary_data = {
                    'analysis_summary': st.session_state.agent_eligibility_analysis.get('analysis_summary', {}),
                    'invocation_summary': st.session_state.agent_invocation_results.get('summary', {}),
                    'applicable_articles': len(st.session_state.article_guidance_results.get('applicable_articles', []))
                }

                st.download_button(
                    label="💾 Download Summary JSON",
                    data=json.dumps(summary_data, indent=2),
                    file_name=f"cbuae_summary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

# Reports and Export Interface (keeping existing functionality)
def show_reports_export():
    """Display enhanced reports and export interface"""

    st.header("📊 CBUAE Compliance Reports & Export")

    # Check what data is available
    has_intelligent_results = (hasattr(st.session_state, 'article_guidance_results') and
                              st.session_state.article_guidance_results is not None)

    if not has_intelligent_results:
        st.warning("⚠️ No analysis results available. Please run intelligent dormancy analysis first.")
        return

    # Report generation options
    st.subheader("📋 Generate CBUAE Reports")

    col1, col2 = st.columns(2)

    with col1:
        report_type = st.selectbox(
            "Report Type",
            [
                "CBUAE Intelligent Analysis Summary",
                "Article-Specific Compliance Report",
                "Agent Efficiency Report",
                "Priority Action Items Report",
                "Executive Dashboard",
                "Regulatory Submission Package"
            ]
        )

        report_format = st.selectbox(
            "Export Format",
            ["PDF", "Excel", "Word", "JSON", "CSV"]
        )

    with col2:
        include_charts = st.checkbox("Include Charts & Visualizations", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
        include_action_items = st.checkbox("Include Action Items", value=True)

        confidentiality = st.selectbox(
            "Confidentiality Level",
            ["Internal Use", "Management Review", "Regulatory Submission"]
        )

    # Generate report
    if st.button("📄 Generate Intelligent CBUAE Report", type="primary", use_container_width=True):

        with st.spinner("Generating comprehensive CBUAE intelligent compliance report..."):
            time.sleep(2)

            # Create report content
            report_content = generate_intelligent_cbuae_report(
                report_type,
                include_charts,
                include_raw_data,
                include_action_items,
                confidentiality
            )

            st.success("✅ Intelligent CBUAE compliance report generated successfully!")

            # Display report preview
            display_intelligent_report_preview(report_content)

            # Download options
            provide_intelligent_download_options(report_content, report_format)

def generate_intelligent_cbuae_report(report_type, include_charts, include_raw_data,
                                     include_action_items, confidentiality):
    """Generate comprehensive intelligent CBUAE compliance report"""

    report_content = {
        "report_info": {
            "type": report_type,
            "generated_at": datetime.now().isoformat(),
            "generated_by": st.session_state.user_data['username'],
            "confidentiality": confidentiality,
            "source": "intelligent_agent_system",
            "cbuae_regulation_version": "2024",
            "include_charts": include_charts,
            "include_raw_data": include_raw_data,
            "include_action_items": include_action_items
        },
        "executive_summary": {},
        "agent_efficiency": {},
        "article_compliance": {},
        "action_items": [],
        "recommendations": []
    }

    try:
        # Get results from session state
        eligibility_analysis = st.session_state.agent_eligibility_analysis
        invocation_results = st.session_state.agent_invocation_results
        guidance_results = st.session_state.article_guidance_results

        # Executive summary
        summary = invocation_results.get('summary', {})
        report_content["executive_summary"] = {
            "total_accounts_analyzed": eligibility_analysis.get('total_accounts_analyzed', 0),
            "agents_invoked": summary.get('agents_invoked_count', 0),
            "agents_skipped": summary.get('agents_skipped_count', 0),
            "total_alerts": summary.get('total_alerts', 0),
            "efficiency_percentage": (summary.get('agents_invoked_count', 0) / summary.get('total_agents_available', 1)) * 100,
            "applicable_articles": len(guidance_results.get('applicable_articles', []))
        }

        # Agent efficiency metrics
        report_content["agent_efficiency"] = {
            "total_agents_available": summary.get('total_agents_available', 0),
            "agents_with_eligible_accounts": eligibility_analysis['analysis_summary']['agents_with_eligible_accounts'],
            "processing_efficiency": f"{(summary.get('agents_invoked_count', 0) / summary.get('total_agents_available', 1)) * 100:.1f}%",
            "resource_optimization": "High" if summary.get('agents_skipped_count', 0) > 0 else "Medium"
        }

        # Article compliance status
        article_compliance = {}
        for article in guidance_results.get('applicable_articles', []):
            article_compliance[article['article_reference']] = {
                'alerts_count': article['alerts_count'],
                'high_priority_alerts': article['priority_breakdown']['high_priority'],
                'status': 'Non-Compliant' if article['priority_breakdown']['high_priority'] > 0 else 'Compliant'
            }

        report_content["article_compliance"] = article_compliance

        # Action items
        if include_action_items:
            report_content["action_items"] = guidance_results.get('action_items', [])

        # Generate recommendations
        report_content["recommendations"] = generate_intelligent_recommendations(report_content)

    except Exception as e:
        report_content['error'] = str(e)

    return report_content

def generate_intelligent_recommendations(report_content):
    """Generate recommendations based on intelligent analysis"""

    recommendations = []

    efficiency = report_content["executive_summary"]["efficiency_percentage"]

    if efficiency > 80:
        recommendations.append({
            "category": "System Efficiency",
            "priority": "Low",
            "recommendation": "Excellent agent efficiency achieved. Consider this as best practice baseline.",
            "impact": "Operational Excellence"
        })
    elif efficiency > 50:
        recommendations.append({
            "category": "System Efficiency",
            "priority": "Medium",
            "recommendation": "Good agent efficiency. Monitor for optimization opportunities.",
            "impact": "Performance Optimization"
        })
    else:
        recommendations.append({
            "category": "System Efficiency",
            "priority": "High",
            "recommendation": "Review agent selection criteria to improve processing efficiency.",
            "impact": "Resource Optimization"
        })

    # Article-specific recommendations
    for article, details in report_content["article_compliance"].items():
        if details['high_priority_alerts'] > 0:
            recommendations.append({
                "category": f"{article} Compliance",
                "priority": "High",
                "recommendation": f"Immediate attention required for {details['high_priority_alerts']} high-priority alerts",
                "impact": "Regulatory Compliance"
            })

    return recommendations

def display_intelligent_report_preview(report_content):
    """Display intelligent report preview"""

    st.subheader("📖 Intelligent CBUAE Analysis Report Preview")

    # Report header
    report_info = report_content["report_info"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Report Type:** {report_info['type']}")
        st.write(f"**Generated By:** {report_info['generated_by']}")
    with col2:
        st.write(f"**Date:** {datetime.fromisoformat(report_info['generated_at']).strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**Source:** Intelligent Agent System")
    with col3:
        st.write(f"**Confidentiality:** {report_info['confidentiality']}")
        st.write(f"**CBUAE Regulation:** {report_info['cbuae_regulation_version']}")

    # Executive summary
    st.subheader("📊 Executive Summary")
    summary = report_content["executive_summary"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accounts Analyzed", summary.get("total_accounts_analyzed", 0))
    with col2:
        st.metric("Agents Invoked", summary.get("agents_invoked", 0))
    with col3:
        st.metric("Total Alerts", summary.get("total_alerts", 0))
    with col4:
        st.metric("System Efficiency", f"{summary.get('efficiency_percentage', 0):.1f}%")

    # Agent efficiency
    st.subheader("⚡ Agent Processing Efficiency")
    efficiency = report_content["agent_efficiency"]

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Processing Efficiency:** {efficiency.get('processing_efficiency', 'N/A')}")
        st.write(f"**Resource Optimization:** {efficiency.get('resource_optimization', 'N/A')}")
    with col2:
        st.write(f"**Agents Available:** {efficiency.get('total_agents_available', 0)}")
        st.write(f"**Agents with Eligible Accounts:** {efficiency.get('agents_with_eligible_accounts', 0)}")

    # Article compliance
    if report_content["article_compliance"]:
        st.subheader("⚖️ Article Compliance Status")

        for article, details in report_content["article_compliance"].items():
            status_icon = "🔴" if details['high_priority_alerts'] > 0 else "🟢"
            st.write(f"{status_icon} **{article}:** {details['alerts_count']} total alerts ({details['high_priority_alerts']} high priority)")

    # Recommendations
    if report_content["recommendations"]:
        st.subheader("💡 Intelligent Recommendations")

        for rec in report_content["recommendations"]:
            priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(rec["priority"], "🔵")

            with st.expander(f"{priority_color} {rec['category']} ({rec['priority']} Priority)"):
                st.write(f"**Recommendation:** {rec['recommendation']}")
                st.write(f"**Impact:** {rec['impact']}")

def provide_intelligent_download_options(report_content, report_format):
    """Provide download options for intelligent reports"""

    col1, col2, col3 = st.columns(3)

    with col1:
        # JSON download
        st.download_button(
            label=f"📥 Download {report_format}",
            data=json.dumps(report_content, indent=2, default=str),
            file_name=f"cbuae_intelligent_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

    with col2:
        # Excel download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:

            # Executive Summary sheet
            summary_data = []
            for key, value in report_content["executive_summary"].items():
                summary_data.append({"Metric": key.replace('_', ' ').title(), "Value": value})

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)

            # Agent Efficiency sheet
            efficiency_data = []
            for key, value in report_content["agent_efficiency"].items():
                efficiency_data.append({"Metric": key.replace('_', ' ').title(), "Value": value})

            efficiency_df = pd.DataFrame(efficiency_data)
            efficiency_df.to_excel(writer, sheet_name='Agent_Efficiency', index=False)

        st.download_button(
            label="📊 Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name=f"cbuae_intelligent_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col3:
        # CSV summary download
        summary_df = pd.DataFrame([
            {"Category": "Accounts Analyzed", "Value": report_content["executive_summary"].get("total_accounts_analyzed", 0)},
            {"Category": "Agents Invoked", "Value": report_content["executive_summary"].get("agents_invoked", 0)},
            {"Category": "System Efficiency", "Value": f"{report_content['executive_summary'].get('efficiency_percentage', 0):.1f}%"},
        ])

        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            label="📋 Download CSV Summary",
            data=csv_data,
            file_name=f"cbuae_intelligent_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Settings Interface (keeping existing functionality)
def show_settings():
    """Display enhanced settings interface"""

    st.header("⚙️ System Settings & Configuration")

    # User settings
    st.subheader("👤 User Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Current User Information:**")
        st.write(f"• **Username:** {st.session_state.user_data['username']}")
        st.write(f"• **Role:** {st.session_state.user_data['role']}")
        st.write(f"• **User ID:** {st.session_state.user_data['user_id']}")

        if st.button("🔑 Change Password"):
            st.info("Password change functionality available in full version")

    with col2:
        st.write("**Session Information:**")
        if st.session_state.session_token:
            try:
                session_info = st.session_state.login_manager.validate_session(st.session_state.session_token)
                expires_at = datetime.fromisoformat(session_info['expires_at']).strftime('%Y-%m-%d %H:%M')
                st.write(f"• **Session expires:** {expires_at}")
                st.write("• **Status:** Active ✅")
            except:
                st.write("• **Status:** Invalid ❌")

        if st.button("🚪 Logout", type="secondary"):
            logout_user()

    # CBUAE Configuration
    st.subheader("🏦 CBUAE Intelligent Agent Configuration")

    with st.expander("⚖️ Agent Selection Parameters"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Eligibility Thresholds:**")
            min_eligible_accounts = st.slider("Minimum Eligible Accounts to Invoke Agent", 1, 10, 1)
            strict_eligibility = st.checkbox("Strict Eligibility Checking", value=True)

        with col2:
            st.write("**Processing Options:**")
            max_concurrent_agents = st.slider("Max Concurrent Agent Processing", 1, 7, 5)
            enable_agent_optimization = st.checkbox("Enable Agent Optimization", value=True)

        if st.button("💾 Save Agent Configuration"):
            st.success("✅ Intelligent agent parameters saved!")

    with st.expander("🔄 System Optimization Settings"):
        col1, col2 = st.columns(2)

        with col1:
            enable_smart_scheduling = st.checkbox("Enable Smart Agent Scheduling", value=True)
            cache_eligibility_results = st.checkbox("Cache Eligibility Analysis", value=True)

        with col2:
            performance_mode = st.selectbox("Performance Mode", ["Balanced", "Speed", "Accuracy"], index=0)
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)

    # System Status
    st.subheader("📊 Intelligent System Health & Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Core Components:**")
        st.write("✅ Authentication System" if st.session_state.authenticated else "❌ Authentication System")
        st.write("✅ Database Connection" if st.session_state.db_connection else "❌ Database Connection")
        st.write("✅ Intelligent Agent Manager" if st.session_state.intelligent_agent_manager else "❌ Intelligent Agent Manager")

    with col2:
        st.write("**Intelligent Processing:**")
        st.write("✅ Data Uploaded" if hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data else "⏳ Data Upload Pending")
        st.write("✅ Eligibility Analysis" if hasattr(st.session_state, 'agent_eligibility_analysis') and st.session_state.agent_eligibility_analysis else "⏳ Analysis Pending")
        st.write("✅ Agent Invocation" if hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results else "⏳ Invocation Pending")
        st.write("✅ Article Guidance" if hasattr(st.session_state, 'article_guidance_results') and st.session_state.article_guidance_results else "⏳ Guidance Pending")

    with col3:
        st.write("**System Features:**")
        st.write("✅ Advanced Features" if ADVANCED_FEATURES else "❌ Advanced Features")
        st.write("✅ LangGraph Agents" if DORMANCY_AGENTS_AVAILABLE else "❌ LangGraph Agents")
        st.write("✅ Intelligent Processing" if st.session_state.intelligent_agent_manager else "⚠️ Basic Processing Only")

    # Performance Metrics
    if st.button("🔄 Refresh System Status"):
        st.rerun()

def logout_user():
    """Handle user logout"""
    try:
        if st.session_state.session_token:
            st.session_state.login_manager.logout_user(st.session_state.session_token)
    except:
        pass

    # Clear session state
    keys_to_clear = [
        'authenticated', 'user_data', 'session_token', 'uploaded_data',
        'processed_data', 'intelligent_agent_manager', 'agent_eligibility_analysis',
        'agent_invocation_results', 'article_guidance_results', 'monitoring_results', 'dashboard_data'
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None if key in ['user_data', 'session_token'] else False

    st.success("Logged out successfully!")
    st.rerun()

# Dashboard Interface
def show_dashboard():
    """Display enhanced dashboard with intelligent analysis focus"""

    st.header("🏠 CBUAE Intelligent Compliance Dashboard")

    # Welcome message
    current_time = datetime.now()
    greeting = "Good morning" if current_time.hour < 12 else "Good afternoon" if current_time.hour < 17 else "Good evening"

    st.markdown(f"### {greeting}, {st.session_state.user_data['username']}! 👋")
    st.markdown(f"*{current_time.strftime('%A, %B %d, %Y')} - CBUAE Intelligent Agent System*")

    # Quick action buttons
    st.subheader("🚀 Quick Actions")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("📁 Upload Data", use_container_width=True):
            st.session_state.page = "📁 Data Upload"

    with col2:
        if st.button("🏦 Intelligent Analysis", use_container_width=True):
            st.session_state.page = "🏦 CBUAE Analysis"

    with col3:
        if st.button("📊 Generate Report", use_container_width=True):
            st.session_state.page = "📊 Reports & Export"

    with col4:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.page = "⚙️ Settings"

    with col5:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Intelligent Analysis Workflow Status
    st.subheader("🧠 Intelligent CBUAE Analysis Workflow")

    workflow_steps = [
        ("Data Upload", hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data, "📁"),
        ("Eligibility Analysis", hasattr(st.session_state, 'agent_eligibility_analysis') and st.session_state.agent_eligibility_analysis, "🔍"),
        ("Agent Invocation", hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results, "🚀"),
        ("Article Guidance", hasattr(st.session_state, 'article_guidance_results') and st.session_state.article_guidance_results, "📚"),
        ("Report Generation", False, "📊")
    ]

    cols = st.columns(len(workflow_steps))
    for i, (step, completed, icon) in enumerate(workflow_steps):
        with cols[i]:
            status_icon = "✅" if completed else "⏳"
            status_color = "#28a745" if completed else "#6c757d"

            st.markdown(f"""
            <div class="workflow-step" style="border-color: {status_color};">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">{step}</div>
                <div style="font-size: 1.5rem;">{status_icon}</div>
            </div>
            """, unsafe_allow_html=True)

    # Current Status Overview
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Intelligent Analysis Statistics")

        # Calculate session statistics
        data_uploaded = hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data
        analysis_completed = hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results

        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            if data_uploaded:
                total_records = 0
                for file_data in st.session_state.uploaded_data.values():
                    if isinstance(file_data, dict) and 'metadata' in file_data:
                        total_records += file_data['metadata'].get('record_count', 0)
                st.metric("Records Analyzed", f"{total_records:,}")
            else:
                st.metric("Records Analyzed", "0")

            # Agent efficiency
            if analysis_completed and st.session_state.agent_invocation_results:
                summary = st.session_state.agent_invocation_results.get('summary', {})
                efficiency = (summary.get('agents_invoked_count', 0) / summary.get('total_agents_available', 1)) * 100
                st.metric("Agent Efficiency", f"{efficiency:.1f}%")
            else:
                st.metric("Agent Efficiency", "Pending")

        with metrics_col2:
            if analysis_completed:
                summary = st.session_state.agent_invocation_results.get('summary', {})
                st.metric("Agents Invoked", summary.get('agents_invoked_count', 0))
                st.metric("Total Alerts", summary.get('total_alerts', 0))
            else:
                st.metric("Agents Invoked", "Pending")
                st.metric("Total Alerts", "Pending")

    with col2:
        st.subheader("🚨 Intelligent System Alerts")

        alerts = []

        # Generate dynamic alerts based on current state
        if hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results:
            summary = st.session_state.agent_invocation_results.get('summary', {})
            total_alerts = summary.get('total_alerts', 0)

            if total_alerts > 0:
                alerts.append({
                    "type": "warning",
                    "message": f"Intelligent analysis found {total_alerts} compliance alerts requiring attention"
                })

            efficiency = (summary.get('agents_invoked_count', 0) / summary.get('total_agents_available', 1)) * 100
            if efficiency > 80:
                alerts.append({
                    "type": "success",
                    "message": f"Excellent agent efficiency achieved: {efficiency:.1f}%"
                })
            elif efficiency < 50:
                alerts.append({
                    "type": "warning",
                    "message": f"Low agent efficiency detected: {efficiency:.1f}% - review eligibility criteria"
                })

        if not DORMANCY_AGENTS_AVAILABLE:
            alerts.append({
                "type": "warning",
                "message": "LangGraph not available - using mock intelligent agents for demonstration"
            })

        if not alerts:
            alerts.append({
                "type": "success",
                "message": "Intelligent CBUAE system operating normally - no current alerts"
            })

        # Display alerts with enhanced styling
        for alert in alerts[:4]:  # Show max 4 alerts
            if alert["type"] == "error":
                st.markdown(f'<div class="alert-high">🚨 <strong>HIGH:</strong> {alert["message"]}</div>',
                           unsafe_allow_html=True)
            elif alert["type"] == "warning":
                st.markdown(f'<div class="alert-medium">⚠️ <strong>MEDIUM:</strong> {alert["message"]}</div>',
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-low">✅ <strong>INFO:</strong> {alert["message"]}</div>',
                           unsafe_allow_html=True)

    # Recent Activity
    st.subheader("📋 Recent Intelligent Analysis Activity")

    # Generate recent activities
    recent_activities = []

    if hasattr(st.session_state, 'article_guidance_results') and st.session_state.article_guidance_results:
        applicable_articles = len(st.session_state.article_guidance_results.get('applicable_articles', []))
        recent_activities.append({
            "time": "2 min ago",
            "activity": "Article-Specific Guidance Generated",
            "details": f"Generated compliance guidance for {applicable_articles} applicable CBUAE articles",
            "status": "success",
            "reference": "Step 3 Complete"
        })

    if hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results:
        summary = st.session_state.agent_invocation_results.get('summary', {})
        recent_activities.append({
            "time": "5 min ago",
            "activity": "Intelligent Agent Invocation Completed",
            "details": f"Invoked {summary.get('agents_invoked_count', 0)} agents, skipped {summary.get('agents_skipped_count', 0)} agents with no eligible accounts",
            "status": "success",
            "reference": "Step 2 Complete"
        })

    if hasattr(st.session_state, 'agent_eligibility_analysis') and st.session_state.agent_eligibility_analysis:
        total_eligible = sum(
            data['eligible_account_count']
            for data in st.session_state.agent_eligibility_analysis['agent_eligibility'].values()
        )
        recent_activities.append({
            "time": "8 min ago",
            "activity": "Account Eligibility Analysis Completed",
            "details": f"Analyzed {st.session_state.agent_eligibility_analysis['total_accounts_analyzed']} accounts, found {total_eligible} eligible for monitoring",
            "status": "success",
            "reference": "Step 1 Complete"
        })

    if hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data:
        file_count = len(st.session_state.uploaded_data)
        total_records = sum(
            file_data.get('metadata', {}).get('record_count', 0)
            for file_data in st.session_state.uploaded_data.values()
            if isinstance(file_data, dict) and 'metadata' in file_data
        )
        recent_activities.append({
            "time": "15 min ago",
            "activity": "Banking Data Uploaded for Intelligent Analysis",
            "details": f"Uploaded {file_count} file(s) with {total_records:,} total records for intelligent CBUAE processing",
            "status": "success",
            "reference": "Data Preparation"
        })

    # Add login activity
    recent_activities.append({
        "time": "30 min ago",
        "activity": "User Authentication",
        "details": f"User {st.session_state.user_data['username']} ({st.session_state.user_data['role']}) logged into intelligent CBUAE system",
        "status": "info",
        "reference": "Access Control"
    })

    # Display activities
    for activity in recent_activities[:5]:  # Show max 5 activities
        col1, col2, col3 = st.columns([1, 5, 1])

        with col1:
            st.write(f"**{activity['time']}**")

        with col2:
            status_icon = {"success": "🟢", "info": "🔵", "warning": "🟡", "error": "🔴"}.get(activity["status"], "⚪")
            st.write(f"{status_icon} **{activity['activity']}**")
            st.caption(f"📋 {activity['details']}")
            st.caption(f"🧠 {activity['reference']}")

        with col3:
            if activity["status"] == "success":
                st.success("✓", help="Completed successfully")
            elif activity["status"] == "info":
                st.info("ℹ️", help="Information")

# Enhanced Sidebar Navigation
def show_enhanced_sidebar():
    """Display enhanced sidebar with intelligent CBUAE branding"""

    with st.sidebar:
        # CBUAE branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1f4e79 0%, #2d5aa0 100%); border-radius: 10px; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">🏦 CBUAE</h2>
            <p style="color: #e8f4fd; margin: 0; font-size: 0.9rem;">Intelligent Compliance System</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧭 Navigation")

        # Navigation with enhanced options
        page = st.radio(
            "Select Module:",
            [
                "🏠 Dashboard",
                "📁 Data Upload",
                "🏦 CBUAE Analysis",
                "📊 Reports & Export",
                "⚙️ Settings"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Enhanced session statistics
        st.markdown("### 📈 Intelligent Analysis Overview")

        if hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data:
            total_records = sum(
                file_data.get('metadata', {}).get('record_count', 0)
                for file_data in st.session_state.uploaded_data.values()
                if isinstance(file_data, dict) and 'metadata' in file_data
            )
            st.metric("Records", f"{total_records:,}")
        else:
            st.metric("Records", "0")

        # Intelligent analysis status
        if hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results:
            summary = st.session_state.agent_invocation_results.get('summary', {})
            invoked = summary.get('agents_invoked_count', 0)
            total = summary.get('total_agents_available', 1)
            efficiency = (invoked / total) * 100
            st.metric("Agent Efficiency", f"{efficiency:.0f}%")
        elif hasattr(st.session_state, 'agent_eligibility_analysis') and st.session_state.agent_eligibility_analysis:
            eligible_agents = st.session_state.agent_eligibility_analysis['analysis_summary']['agents_with_eligible_accounts']
            st.metric("Eligible Agents", eligible_agents)
        else:
            st.metric("Analysis", "Pending")

        # Alerts status
        if hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results:
            alerts = st.session_state.agent_invocation_results.get('summary', {}).get('total_alerts', 0)
            alert_color = "🔴" if alerts > 10 else "🟡" if alerts > 0 else "🟢"
            st.metric("Alerts", f"{alert_color} {alerts}")
        else:
            st.metric("Alerts", "⏳ TBD")

        st.markdown("---")

        # Intelligent system status
        st.markdown("### 🧠 Intelligent System Status")

        # Core components
        components_status = [
            ("Authentication", st.session_state.authenticated),
            ("Data Processing", hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data),
            ("Agent Manager", st.session_state.intelligent_agent_manager is not None),
            ("Advanced Features", ADVANCED_FEATURES)
        ]

        for component, status in components_status:
            status_icon = "🟢" if status else "🔴"
            st.write(f"{status_icon} {component}")

        # Analysis workflow status
        workflow_status = [
            ("Eligibility Analysis", hasattr(st.session_state, 'agent_eligibility_analysis') and st.session_state.agent_eligibility_analysis),
            ("Agent Invocation", hasattr(st.session_state, 'agent_invocation_results') and st.session_state.agent_invocation_results),
            ("Article Guidance", hasattr(st.session_state, 'article_guidance_results') and st.session_state.article_guidance_results)
        ]

        for workflow, status in workflow_status:
            status_icon = "🟢" if status else "🟡"
            st.write(f"{status_icon} {workflow}")

        # Last update time
        st.write(f"🕒 Updated: {datetime.now().strftime('%H:%M:%S')}")

        st.markdown("---")

        # Help and documentation
        st.markdown("### 📚 Intelligent CBUAE Resources")

        with st.expander("📖 Documentation"):
            st.write("• Intelligent Agent Framework")
            st.write("• Article-Specific Processing")
            st.write("• Efficiency Optimization")
            st.write("• Smart Resource Management")

        with st.expander("🚀 Quick Help"):
            st.write("1. Upload banking data")
            st.write("2. Analyze account eligibility")
            st.write("3. Invoke relevant agents only")
            st.write("4. Get article-specific guidance")

        # Installation status
        with st.expander("🔧 System Requirements"):
            st.write(f"**LangGraph:** {'✅' if DORMANCY_AGENTS_AVAILABLE else '❌'}")
            st.write(f"**FAISS:** {'✅' if ADVANCED_FEATURES else '❌'}")
            st.write("**Intelligent Framework:** ✅")

            if not DORMANCY_AGENTS_AVAILABLE:
                st.warning("Install LangGraph for full functionality:")
                st.code("pip install langgraph")

        return page

# Main Application Function
async def main():
    """Enhanced main application function with intelligent CBUAE integration"""

    # Check authentication
    if not st.session_state.authenticated:
        show_login()
        return

    # Validate session
    try:
        if st.session_state.session_token:
            session_info = st.session_state.login_manager.validate_session(st.session_state.session_token)
            if not session_info.get('session_valid'):
                st.error("⚠️ Session expired. Please login again.")
                logout_user()
                return
    except:
        st.error("❌ Invalid session. Please login again.")
        logout_user()
        return

    # Main application header
    st.markdown('<div class="main-header">🏦 CBUAE Intelligent Banking Compliance System</div>', unsafe_allow_html=True)
    st.markdown(f"*Welcome, {st.session_state.user_data['username']} ({st.session_state.user_data['role']}) - {datetime.now().strftime('%A, %B %d, %Y')}*")

    # Enhanced sidebar navigation
    selected_page = show_enhanced_sidebar()

    # Main content area with enhanced routing
    if selected_page == "🏠 Dashboard":
        show_dashboard()
    elif selected_page == "📁 Data Upload":
        uploaded_data, data_source_info = show_data_upload()
        if uploaded_data:
            st.session_state.uploaded_data = uploaded_data
            st.session_state.data_source_info = data_source_info
    elif selected_page == "🏦 CBUAE Analysis":
        await show_dormancy_analysis()
    elif selected_page == "📊 Reports & Export":
        show_reports_export()
    elif selected_page == "⚙️ Settings":
        show_settings()

    # Footer with system information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("🏦 CBUAE Intelligent Compliance System")
    with col2:
        st.caption(f"🔐 Logged in as: {st.session_state.user_data['username']}")
    with col3:
        st.caption(f"🕒 Session: {datetime.now().strftime('%H:%M:%S')}")

# Error Handling and Logging
def setup_error_handling():
    """Setup comprehensive error handling and logging"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cbuae_intelligent_compliance.log', mode='a')
        ]
    )

    logger.info("CBUAE Intelligent Banking Compliance System initialized")

# Run the application
if __name__ == "__main__":

    # Setup error handling
    setup_error_handling()

    # Display system initialization info
    st.info("🧠 **CBUAE Intelligent Agent System** - Only invokes agents with eligible accounts")

    if DORMANCY_AGENTS_AVAILABLE:
        st.success("✅ Full CBUAE Intelligent Agents Available")
    else:
        st.warning("⚠️ LangGraph not installed - using intelligent mock agents for demonstration")
        st.info("💡 Install LangGraph for full functionality: `pip install langgraph`")

    if ADVANCED_FEATURES:
        st.success("✅ Advanced Features (FAISS) Available")
    else:
        st.info("📝 Install FAISS for enhanced search: `pip install faiss-cpu sentence-transformers`")

    # Initialize intelligent agent manager if not already done
    if st.session_state.intelligent_agent_manager is None:
        st.session_state.intelligent_agent_manager = CBUAEIntelligentAgentManager()
        st.success("✅ Intelligent Agent Manager initialized!")

    # Run the async main function
    try:
        asyncio.run(main())
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

        # Show troubleshooting info
        with st.expander("🛠️ Troubleshooting"):
            st.write("**Common Issues:**")
            st.write("1. **LangGraph not installed:** Run `pip install langgraph`")
            st.write("2. **FAISS not available:** Run `pip install faiss-cpu sentence-transformers`")
            st.write("3. **Streamlit issues:** Restart the application")
            st.write("4. **Session problems:** Clear browser cache and reload")

            st.write("**Current System Status:**")
            st.write(f"- Python version: {sys.version}")
            st.write(f"- Streamlit version: {st.__version__}")
            st.write(f"- LangGraph available: {DORMANCY_AGENTS_AVAILABLE}")
            st.write(f"- FAISS available: {ADVANCED_FEATURES}")
            st.write(f"- Intelligent Agent Manager: {st.session_state.intelligent_agent_manager is not None}")

    # Display intelligent system summary
    st.markdown("---")
    st.markdown("### 🧠 Intelligent System Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("**🎯 Smart Processing**")
        st.write("• Only relevant agents invoked")
        st.write("• Efficiency optimized")

    with col2:
        st.write("**⚖️ Article-Specific**")
        st.write("• Targeted CBUAE guidance")
        st.write("• Relevant regulations only")

    with col3:
        st.write("**📊 Resource Optimized**")
        st.write("• Skip unnecessary processing")
        st.write("• Focus on actual issues")

    with col4:
        st.write("**🚀 Enhanced Performance**")
        st.write("• Faster analysis")
        st.write("• Better user experience")