"""
CBUAE Banking Compliance - Real-time Dormancy Analysis Streamlit App
Analyzes actual banking data with 10 dormancy agents
No mock data - uses real CSV analysis only
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64
import json
from typing import Dict, List, Tuple, Optional
import logging

# Configure Streamlit page
st.set_page_config(
    page_title="CBUAE Banking Compliance System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2c5aa0);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .success-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .warning-badge {
        background: #ffc107;
        color: black;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .critical-badge {
        background: #dc3545;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ===== DORMANCY ANALYSIS ENGINE =====

class CBUAEDormancyAnalyzer:
    """Real-time CBUAE dormancy analysis engine"""

    def __init__(self):
        self.agents_config = {
            'demand_deposit_inactivity': {
                'name': 'Demand Deposit Inactivity',
                'description': 'Analyzes current and savings accounts for extended inactivity',
                'cbuae_article': 'Article 2.1.1',
                'account_types': ['CURRENT', 'SAVINGS'],
                'priority': 'HIGH'
            },
            'fixed_deposit_inactivity': {
                'name': 'Fixed Deposit Inactivity',
                'description': 'Identifies matured fixed deposits requiring action',
                'cbuae_article': 'Article 2.2',
                'account_types': ['FIXED_DEPOSIT'],
                'priority': 'CRITICAL'
            },
            'investment_inactivity': {
                'name': 'Investment Account Inactivity',
                'description': 'Monitors investment portfolios with extended dormancy',
                'cbuae_article': 'Article 2.3',
                'account_types': ['INVESTMENT'],
                'priority': 'HIGH'
            },
            'eligible_for_cb_transfer': {
                'name': 'Eligible for CB Transfer',
                'description': 'Identifies accounts ready for Central Bank transfer',
                'cbuae_article': 'Article 8.1, 8.2',
                'account_types': ['ALL'],
                'priority': 'CRITICAL'
            },
            'contact_attempts_needed': {
                'name': 'Contact Attempts Needed',
                'description': 'Tracks accounts requiring customer contact attempts',
                'cbuae_article': 'Article 5',
                'account_types': ['ALL'],
                'priority': 'URGENT'
            },
            'high_value_dormant': {
                'name': 'High Value Dormant Accounts',
                'description': 'Monitors high-value dormant accounts for priority processing',
                'cbuae_article': 'Internal Policy',
                'account_types': ['ALL'],
                'priority': 'HIGH'
            },
            'foreign_currency_dormant': {
                'name': 'Foreign Currency Dormant',
                'description': 'Analyzes dormant accounts in foreign currencies',
                'cbuae_article': 'Article 8.5',
                'account_types': ['ALL'],
                'priority': 'MEDIUM'
            },
            'joint_account_dormancy': {
                'name': 'Joint Account Dormancy',
                'description': 'Specialized handling for dormant joint accounts',
                'cbuae_article': 'Article 3.2',
                'account_types': ['ALL'],
                'priority': 'HIGH'
            },
            'minor_account_dormancy': {
                'name': 'Minor Account Dormancy',
                'description': 'Manages dormancy for accounts belonging to minors',
                'cbuae_article': 'Article 3.3',
                'account_types': ['ALL'],
                'priority': 'HIGH'
            },
            'corporate_account_dormancy': {
                'name': 'Corporate Account Dormancy',
                'description': 'Handles dormancy for corporate and business accounts',
                'cbuae_article': 'Article 2.5',
                'account_types': ['ALL'],
                'priority': 'MEDIUM'
            }
        }

    def analyze_demand_deposit_inactivity(self, df: pd.DataFrame) -> Dict:
        """Analyze demand deposit account inactivity"""
        # Filter for current and savings accounts
        demand_accounts = df[df['account_type'].isin(['CURRENT', 'SAVINGS'])].copy()

        # Get dormant accounts
        dormant = demand_accounts[demand_accounts['account_status'] == 'DORMANT'].copy()

        # Additional filtering for transaction inactivity (3+ years)
        if not dormant.empty and 'last_transaction_date' in dormant.columns:
            dormant['last_transaction_date'] = pd.to_datetime(dormant['last_transaction_date'], errors='coerce')
            cutoff_date = datetime.now() - timedelta(days=1095)  # 3 years
            dormant = dormant[dormant['last_transaction_date'] < cutoff_date]

        return self._create_agent_result('demand_deposit_inactivity', demand_accounts, dormant)

    def analyze_fixed_deposit_inactivity(self, df: pd.DataFrame) -> Dict:
        """Analyze fixed deposit inactivity"""
        # Filter for fixed deposit accounts
        fd_accounts = df[df['account_type'] == 'FIXED_DEPOSIT'].copy()

        # Get dormant accounts
        dormant = fd_accounts[fd_accounts['account_status'] == 'DORMANT'].copy()

        # Check for matured deposits
        if not dormant.empty and 'maturity_date' in dormant.columns:
            dormant['maturity_date'] = pd.to_datetime(dormant['maturity_date'], errors='coerce')
            dormant = dormant[dormant['maturity_date'] < datetime.now()]

        return self._create_agent_result('fixed_deposit_inactivity', fd_accounts, dormant)

    def analyze_investment_inactivity(self, df: pd.DataFrame) -> Dict:
        """Analyze investment account inactivity"""
        # Filter for investment accounts
        investment_accounts = df[df['account_type'] == 'INVESTMENT'].copy()

        # Get dormant accounts
        dormant = investment_accounts[investment_accounts['account_status'] == 'DORMANT'].copy()

        return self._create_agent_result('investment_inactivity', investment_accounts, dormant)

    def analyze_cb_transfer_eligibility(self, df: pd.DataFrame) -> Dict:
        """Analyze Central Bank transfer eligibility"""
        # Get dormant accounts with sufficient balance and long dormancy period
        dormant = df[df['account_status'] == 'DORMANT'].copy()

        if not dormant.empty:
            # Convert balance to numeric
            dormant['balance_current'] = pd.to_numeric(dormant['balance_current'], errors='coerce').fillna(0)

            # Filter for transfer eligibility (balance >= 1000 AED, dormant >= 5 years)
            dormant['dormancy_period_months'] = pd.to_numeric(dormant['dormancy_period_months'], errors='coerce').fillna(0)

            eligible = dormant[
                (dormant['balance_current'] >= 1000) &
                (dormant['dormancy_period_months'] >= 60)  # 5 years
            ]
        else:
            eligible = pd.DataFrame()

        return self._create_agent_result('eligible_for_cb_transfer', df, eligible)

    def analyze_contact_attempts_needed(self, df: pd.DataFrame) -> Dict:
        """Analyze accounts needing contact attempts"""
        # Get dormant accounts with insufficient contact attempts
        dormant = df[df['account_status'] == 'DORMANT'].copy()

        if not dormant.empty:
            dormant['contact_attempts_made'] = pd.to_numeric(dormant['contact_attempts_made'], errors='coerce').fillna(0)
            needs_contact = dormant[dormant['contact_attempts_made'] < 3]
        else:
            needs_contact = pd.DataFrame()

        return self._create_agent_result('contact_attempts_needed', df, needs_contact)

    def analyze_high_value_dormant(self, df: pd.DataFrame) -> Dict:
        """Analyze high-value dormant accounts"""
        # Get dormant accounts with high balances
        dormant = df[df['account_status'] == 'DORMANT'].copy()

        if not dormant.empty:
            dormant['balance_current'] = pd.to_numeric(dormant['balance_current'], errors='coerce').fillna(0)
            high_value = dormant[dormant['balance_current'] > 50000]  # AED 50K threshold
        else:
            high_value = pd.DataFrame()

        return self._create_agent_result('high_value_dormant', df, high_value)

    def analyze_foreign_currency_dormant(self, df: pd.DataFrame) -> Dict:
        """Analyze foreign currency dormant accounts"""
        # Get dormant accounts in foreign currencies
        dormant = df[df['account_status'] == 'DORMANT'].copy()

        if not dormant.empty:
            foreign_currency = dormant[dormant['currency'] != 'AED']
        else:
            foreign_currency = pd.DataFrame()

        return self._create_agent_result('foreign_currency_dormant', df, foreign_currency)

    def analyze_joint_account_dormancy(self, df: pd.DataFrame) -> Dict:
        """Analyze joint account dormancy"""
        # Get dormant joint accounts
        dormant = df[df['account_status'] == 'DORMANT'].copy()

        if not dormant.empty:
            joint_dormant = dormant[dormant['is_joint_account'] == 'YES']
        else:
            joint_dormant = pd.DataFrame()

        return self._create_agent_result('joint_account_dormancy', df, joint_dormant)

    def analyze_minor_account_dormancy(self, df: pd.DataFrame) -> Dict:
        """Analyze minor account dormancy"""
        # Calculate age from date of birth for minor identification
        current_date = datetime.now()
        df_copy = df.copy()

        if 'date_of_birth' in df_copy.columns:
            df_copy['date_of_birth'] = pd.to_datetime(df_copy['date_of_birth'], errors='coerce')
            df_copy['age'] = (current_date - df_copy['date_of_birth']).dt.days / 365.25

            # Get dormant accounts for minors (under 18)
            dormant_minors = df_copy[
                (df_copy['account_status'] == 'DORMANT') &
                (df_copy['age'] < 18)
            ]
        else:
            dormant_minors = pd.DataFrame()

        return self._create_agent_result('minor_account_dormancy', df, dormant_minors)

    def analyze_corporate_account_dormancy(self, df: pd.DataFrame) -> Dict:
        """Analyze corporate account dormancy"""
        # Get dormant corporate accounts
        dormant = df[df['account_status'] == 'DORMANT'].copy()

        if not dormant.empty:
            corporate_dormant = dormant[dormant['customer_type'] == 'CORPORATE']
        else:
            corporate_dormant = pd.DataFrame()

        return self._create_agent_result('corporate_account_dormancy', df, corporate_dormant)

    def _create_agent_result(self, agent_key: str, total_data: pd.DataFrame, dormant_data: pd.DataFrame) -> Dict:
        """Create standardized agent result"""
        agent_config = self.agents_config[agent_key]

        # Calculate metrics
        total_balance = 0.0
        avg_balance = 0.0

        if not dormant_data.empty and 'balance_current' in dormant_data.columns:
            balances = pd.to_numeric(dormant_data['balance_current'], errors='coerce').fillna(0)
            total_balance = float(balances.sum())
            avg_balance = float(balances.mean())

        return {
            'agent_key': agent_key,
            'agent_name': agent_config['name'],
            'description': agent_config['description'],
            'cbuae_article': agent_config['cbuae_article'],
            'priority': agent_config['priority'],
            'records_processed': len(total_data),
            'dormant_records_found': len(dormant_data),
            'total_balance': total_balance,
            'average_balance': avg_balance,
            'success': True,
            'detailed_results': dormant_data,
            'processing_time': 0.1  # Real processing time would be calculated
        }

    def run_comprehensive_analysis(self, df: pd.DataFrame) -> Dict:
        """Run all 10 dormancy agents"""
        logger.info(f"Starting comprehensive dormancy analysis on {len(df)} records")

        analysis_methods = {
            'demand_deposit_inactivity': self.analyze_demand_deposit_inactivity,
            'fixed_deposit_inactivity': self.analyze_fixed_deposit_inactivity,
            'investment_inactivity': self.analyze_investment_inactivity,
            'eligible_for_cb_transfer': self.analyze_cb_transfer_eligibility,
            'contact_attempts_needed': self.analyze_contact_attempts_needed,
            'high_value_dormant': self.analyze_high_value_dormant,
            'foreign_currency_dormant': self.analyze_foreign_currency_dormant,
            'joint_account_dormancy': self.analyze_joint_account_dormancy,
            'minor_account_dormancy': self.analyze_minor_account_dormancy,
            'corporate_account_dormancy': self.analyze_corporate_account_dormancy
        }

        results = {}
        total_dormant_found = 0
        active_agents = 0

        for agent_key, method in analysis_methods.items():
            try:
                agent_result = method(df)
                results[agent_key] = agent_result

                dormant_count = agent_result['dormant_records_found']
                total_dormant_found += dormant_count

                if dormant_count > 0:
                    active_agents += 1

                logger.info(f"Agent {agent_key}: {dormant_count} dormant accounts found")

            except Exception as e:
                logger.error(f"Agent {agent_key} failed: {str(e)}")
                results[agent_key] = {
                    'agent_key': agent_key,
                    'success': False,
                    'error': str(e),
                    'dormant_records_found': 0
                }

        return {
            'success': True,
            'total_records': len(df),
            'total_dormant_found': total_dormant_found,
            'active_agents': active_agents,
            'total_agents': len(analysis_methods),
            'agent_results': results,
            'analysis_timestamp': datetime.now().isoformat()
        }

# ===== STREAMLIT APP =====

def load_csv_data(uploaded_file) -> pd.DataFrame:
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def display_data_overview(df: pd.DataFrame):
    """Display data overview and statistics"""
    st.subheader("📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        dormant_count = len(df[df['account_status'] == 'DORMANT']) if 'account_status' in df.columns else 0
        st.metric("Dormant Accounts", f"{dormant_count:,}")

    with col3:
        if 'balance_current' in df.columns:
            total_balance = pd.to_numeric(df['balance_current'], errors='coerce').sum()
            st.metric("Total Balance", f"AED {total_balance:,.2f}")

    with col4:
        unique_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else len(df)
        st.metric("Unique Customers", f"{unique_customers:,}")

    # Account type distribution
    if 'account_type' in df.columns:
        st.subheader("Account Type Distribution")
        account_type_counts = df['account_type'].value_counts()
        fig = px.pie(
            values=account_type_counts.values,
            names=account_type_counts.index,
            title="Distribution by Account Type"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_agent_results(analysis_results: Dict):
    """Display dormancy agent analysis results"""
    st.subheader("🤖 Dormancy Agent Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyzed", f"{analysis_results['total_records']:,}")

    with col2:
        st.metric("Dormant Found", f"{analysis_results['total_dormant_found']:,}")

    with col3:
        st.metric("Active Agents", f"{analysis_results['active_agents']}/{analysis_results['total_agents']}")

    with col4:
        success_rate = (analysis_results['active_agents'] / analysis_results['total_agents']) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Agent details
    st.subheader("🔍 Individual Agent Results")

    agent_results = analysis_results['agent_results']

    # Filter for agents with dormant accounts found
    active_agents = {k: v for k, v in agent_results.items()
                    if v.get('success', False) and v.get('dormant_records_found', 0) > 0}

    if active_agents:
        for agent_key, result in active_agents.items():
            with st.expander(f"🚨 {result['agent_name']} - {result['dormant_records_found']} accounts"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Description:** {result['description']}")
                    st.write(f"**CBUAE Article:** {result['cbuae_article']}")
                    st.write(f"**Priority:** {result['priority']}")
                    st.write(f"**Records Processed:** {result['records_processed']:,}")
                    st.write(f"**Dormant Accounts Found:** {result['dormant_records_found']:,}")

                    if result['total_balance'] > 0:
                        st.write(f"**Total Balance:** AED {result['total_balance']:,.2f}")
                        st.write(f"**Average Balance:** AED {result['average_balance']:,.2f}")

                with col2:
                    # Priority badge
                    priority = result['priority']
                    if priority == 'CRITICAL':
                        st.markdown('<div class="critical-badge">🔴 CRITICAL</div>', unsafe_allow_html=True)
                    elif priority == 'HIGH':
                        st.markdown('<div class="warning-badge">🟡 HIGH</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-badge">🟢 NORMAL</div>', unsafe_allow_html=True)

                    # Download button for detailed results
                    if 'detailed_results' in result and not result['detailed_results'].empty:
                        csv_data = result['detailed_results'].to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"{agent_key}_dormant_accounts.csv",
                            mime="text/csv"
                        )

                # Display sample records
                if 'detailed_results' in result and not result['detailed_results'].empty:
                    st.write("**Sample Records:**")
                    display_cols = ['customer_id', 'account_id', 'account_type', 'balance_current', 'dormancy_status']
                    available_cols = [col for col in display_cols if col in result['detailed_results'].columns]
                    st.dataframe(
                        result['detailed_results'][available_cols].head(5),
                        use_container_width=True
                    )
    else:
        st.info("🎉 No dormant accounts found by any agents. All accounts are in good standing!")

def create_analysis_dashboard(analysis_results: Dict):
    """Create comprehensive analysis dashboard"""
    st.subheader("📈 Analysis Dashboard")

    agent_results = analysis_results['agent_results']

    # Create visualization data
    viz_data = []
    for agent_key, result in agent_results.items():
        if result.get('success', False):
            viz_data.append({
                'Agent': result['agent_name'],
                'Dormant_Found': result['dormant_records_found'],
                'Total_Balance': result.get('total_balance', 0),
                'Priority': result.get('priority', 'NORMAL')
            })

    viz_df = pd.DataFrame(viz_data)

    if not viz_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Dormant accounts by agent
            fig1 = px.bar(
                viz_df,
                x='Agent',
                y='Dormant_Found',
                color='Priority',
                title="Dormant Accounts Found by Agent",
                color_discrete_map={
                    'CRITICAL': '#dc3545',
                    'HIGH': '#ffc107',
                    'URGENT': '#fd7e14',
                    'MEDIUM': '#20c997',
                    'NORMAL': '#28a745'
                }
            )
            fig1.update_xaxis(tickangle=45)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Total balance distribution
            fig2 = px.pie(
                viz_df[viz_df['Total_Balance'] > 0],
                values='Total_Balance',
                names='Agent',
                title="Dormant Balance Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)

def main():
    """Main Streamlit application"""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ CBUAE Banking Compliance System</h1>
        <h3>Real-time Dormancy Analysis with 10 AI Agents</h3>
        <p>UAE Central Bank Regulation No. 1/2020 Compliance</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        st.markdown("Upload your banking compliance dataset to begin analysis")

        st.markdown("### 📋 Available Agents")
        st.markdown("""
        1. **Demand Deposit Inactivity** - Article 2.1.1
        2. **Fixed Deposit Inactivity** - Article 2.2  
        3. **Investment Inactivity** - Article 2.3
        4. **CB Transfer Eligibility** - Article 8.1, 8.2
        5. **Contact Attempts Needed** - Article 5
        6. **High Value Dormant** - Internal Policy
        7. **Foreign Currency Dormant** - Article 8.5
        8. **Joint Account Dormancy** - Article 3.2
        9. **Minor Account Dormancy** - Article 3.3
        10. **Corporate Account Dormancy** - Article 2.5
        """)

    # File upload
    st.subheader("📤 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Banking Compliance CSV Dataset",
        type=['csv'],
        help="Upload your banking compliance dataset (CSV format)"
    )

    if uploaded_file is not None:
        # Load data
        df = load_csv_data(uploaded_file)

        if df is not None:
            # Display data overview
            display_data_overview(df)

            # Analysis section
            st.subheader("🔍 Dormancy Analysis")

            if st.button("🚀 Run Comprehensive Dormancy Analysis", type="primary"):
                with st.spinner("Running all 10 dormancy agents..."):
                    # Initialize analyzer
                    analyzer = CBUAEDormancyAnalyzer()

                    # Run analysis
                    analysis_results = analyzer.run_comprehensive_analysis(df)

                    if analysis_results['success']:
                        st.success(f"✅ Analysis completed! Found {analysis_results['total_dormant_found']} dormant accounts")

                        # Display results
                        display_agent_results(analysis_results)

                        # Create dashboard
                        create_analysis_dashboard(analysis_results)

                        # Store results in session state for download
                        st.session_state['analysis_results'] = analysis_results
                    else:
                        st.error("❌ Analysis failed")

            # Download comprehensive report
            if 'analysis_results' in st.session_state:
                st.subheader("📥 Export Results")

                results = st.session_state['analysis_results']

                # Create comprehensive report
                report_data = {
                    'analysis_summary': {
                        'timestamp': results['analysis_timestamp'],
                        'total_records': results['total_records'],
                        'total_dormant_found': results['total_dormant_found'],
                        'active_agents': results['active_agents']
                    },
                    'agent_results': {}
                }

                for agent_key, result in results['agent_results'].items():
                    if result.get('success', False):
                        report_data['agent_results'][agent_key] = {
                            'agent_name': result['agent_name'],
                            'cbuae_article': result['cbuae_article'],
                            'dormant_records_found': result['dormant_records_found'],
                            'total_balance': result.get('total_balance', 0)
                        }

                report_json = json.dumps(report_data, indent=2)

                st.download_button(
                    label="📊 Download Comprehensive Report (JSON)",
                    data=report_json,
                    file_name=f"cbuae_dormancy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    else:
        st.info("👆 Please upload a CSV file to begin dormancy analysis")

        # Show sample data structure
        st.subheader("📋 Expected Data Structure")
        st.markdown("""
        Your CSV should contain the following key columns:
        - `customer_id`: Unique customer identifier
        - `account_id`: Unique account identifier  
        - `account_type`: CURRENT, SAVINGS, FIXED_DEPOSIT, INVESTMENT
        - `account_status`: ACTIVE, DORMANT, CLOSED
        - `balance_current`: Current account balance
        - `last_transaction_date`: Date of last transaction
        - `currency`: Account currency (AED, USD, EUR, etc.)
        - `is_joint_account`: YES/NO for joint accounts
        - `customer_type`: INDIVIDUAL, CORPORATE
        - `contact_attempts_made`: Number of contact attempts
        - `dormancy_period_months`: Duration of dormancy in months
        """)

if __name__ == "__main__":
    main()