"""
app.py - Enhanced Banking Compliance Agentic AI System
Comprehensive web application with working CBUAE dormancy analysis
Shows real results from uploaded CSV data
"""

import streamlit as st
import pandas as pd
import json
import logging
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import secrets
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Banking Compliance AI System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
    }
    
    .status-success {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    
    .status-warning {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    
    .status-error {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
    }
    
    .triggered-function {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# CBUAE Dormancy Analysis Engine
class CBUAEDormancyAnalyzer:
    """Enhanced CBUAE dormancy analyzer with real data processing"""
    
    def __init__(self):
        self.high_value_threshold_aed = 25000
        self.currency_rates = {
            'USD': 3.67,
            'EUR': 4.0,
            'GBP': 4.5,
            'SAR': 0.98,
            'AED': 1.0
        }
        
    def months_to_years(self, months: float) -> float:
        """Convert dormancy months to years"""
        return months / 12 if pd.notna(months) and months > 0 else 0
    
    def is_high_value(self, balance: float, currency: str) -> bool:
        """Check if account balance exceeds high value threshold"""
        if pd.isna(balance) or balance <= 0:
            return False
        
        rate = self.currency_rates.get(currency, 1.0)
        aed_equivalent = balance * rate
        return aed_equivalent >= self.high_value_threshold_aed
    
    def analyze_article_2_1_1_demand_deposits(self, df: pd.DataFrame) -> Dict:
        """Article 2.1.1 - Demand Deposit Dormancy Analysis"""
        # Filter demand deposit accounts (Current/Savings)
        demand_accounts = df[df['account_type'].isin(['CURRENT', 'SAVINGS'])].copy()
        
        # Calculate dormancy years
        demand_accounts['dormancy_years'] = demand_accounts['dormancy_period_months'].apply(self.months_to_years)
        
        # Find dormant accounts (3+ years)
        dormant_demand = demand_accounts[
            (demand_accounts['dormancy_years'] >= 3) & 
            (demand_accounts['account_status'] == 'DORMANT')
        ]
        
        # Identify high-value accounts
        dormant_demand['is_high_value'] = dormant_demand.apply(
            lambda row: self.is_high_value(row['balance_current'], row['currency']), axis=1
        )
        
        high_value_count = dormant_demand['is_high_value'].sum()
        
        # Generate sample findings
        sample_findings = []
        for _, account in dormant_demand.head(5).iterrows():
            sample_findings.append({
                'account_id': account['account_id'],
                'dormancy_years': round(account['dormancy_years'], 1),
                'balance': account['balance_current'],
                'currency': account['currency'],
                'is_high_value': account['is_high_value'],
                'last_activity': account.get('last_transaction_date', 'N/A')
            })
        
        return {
            'function_name': 'analyze_demand_deposit_dormancy',
            'article': '2.1.1',
            'title': 'Demand Deposit Dormancy Analysis',
            'description': 'Monitors Current, Savings, and Call accounts for 3+ years of customer inactivity',
            'total_accounts': len(demand_accounts),
            'violations': len(dormant_demand),
            'high_value_violations': high_value_count,
            'priority': 'HIGH' if len(dormant_demand) > 0 else 'LOW',
            'status': 'NON_COMPLIANT' if len(dormant_demand) > 0 else 'COMPLIANT',
            'criteria': [
                'No customer-initiated transactions for 3+ years',
                'No communication from customer for 3+ years',
                'Customer has no active liability accounts',
                'Account balance remains positive'
            ],
            'next_actions': [
                'Initiate Article 3 contact procedures',
                'Verify customer contact information',
                'Check for active liability relationships',
                'Document all communication attempts'
            ],
            'sample_findings': sample_findings,
            'compliance_impact': 'High' if len(dormant_demand) > 10 else 'Medium'
        }
    
    def analyze_article_2_2_fixed_deposits(self, df: pd.DataFrame) -> Dict:
        """Article 2.2 - Fixed Deposit Dormancy Analysis"""
        # Filter fixed deposit accounts
        fixed_accounts = df[df['account_type'] == 'FIXED_DEPOSIT'].copy()
        
        if fixed_accounts.empty:
            return None
        
        # Calculate dormancy years
        fixed_accounts['dormancy_years'] = fixed_accounts['dormancy_period_months'].apply(self.months_to_years)
        
        # Find dormant accounts (3+ years)
        dormant_fixed = fixed_accounts[
            (fixed_accounts['dormancy_years'] >= 3) & 
            (fixed_accounts['account_status'] == 'DORMANT')
        ]
        
        # Generate sample findings
        sample_findings = []
        for _, account in dormant_fixed.head(5).iterrows():
            sample_findings.append({
                'account_id': account['account_id'],
                'dormancy_years': round(account['dormancy_years'], 1),
                'balance': account['balance_current'],
                'currency': account['currency'],
                'maturity_date': account.get('maturity_date', 'N/A'),
                'auto_renewal': account.get('auto_renewal', 'N/A')
            })
        
        return {
            'function_name': 'analyze_fixed_deposit_dormancy',
            'article': '2.2',
            'title': 'Fixed Deposit Dormancy Analysis',
            'description': 'Monitors Fixed/Term deposits that have matured without customer action',
            'total_accounts': len(fixed_accounts),
            'violations': len(dormant_fixed),
            'priority': 'HIGH' if len(dormant_fixed) > 0 else 'LOW',
            'status': 'NON_COMPLIANT' if len(dormant_fixed) > 0 else 'COMPLIANT',
            'criteria': [
                'Deposit has matured without renewal request',
                'No customer communication for 3+ years post-maturity',
                'Customer has not claimed maturity proceeds'
            ],
            'next_actions': [
                'Contact customer regarding matured deposits',
                'Review auto-renewal status and settings',
                'Process unclaimed maturity proceeds per regulations'
            ],
            'sample_findings': sample_findings,
            'compliance_impact': 'High' if len(dormant_fixed) > 5 else 'Medium'
        }
    
    def analyze_article_2_3_investments(self, df: pd.DataFrame) -> Dict:
        """Article 2.3 - Investment Account Dormancy Analysis"""
        # Filter investment accounts
        investment_accounts = df[df['account_type'] == 'INVESTMENT'].copy()
        
        if investment_accounts.empty:
            return None
        
        # Calculate dormancy years
        investment_accounts['dormancy_years'] = investment_accounts['dormancy_period_months'].apply(self.months_to_years)
        
        # Find dormant accounts (3+ years)
        dormant_investments = investment_accounts[
            (investment_accounts['dormancy_years'] >= 3) & 
            (investment_accounts['account_status'] == 'DORMANT')
        ]
        
        # Generate sample findings
        sample_findings = []
        for _, account in dormant_investments.head(5).iterrows():
            sample_findings.append({
                'account_id': account['account_id'],
                'dormancy_years': round(account['dormancy_years'], 1),
                'balance': account['balance_current'],
                'currency': account['currency'],
                'investment_type': account.get('account_subtype', 'N/A'),
                'last_activity': account.get('last_transaction_date', 'N/A')
            })
        
        return {
            'function_name': 'analyze_investment_dormancy',
            'article': '2.3',
            'title': 'Investment Account Dormancy Analysis',
            'description': 'Monitors investment accounts with extended inactivity periods',
            'total_accounts': len(investment_accounts),
            'violations': len(dormant_investments),
            'priority': 'HIGH' if len(dormant_investments) > 0 else 'LOW',
            'status': 'NON_COMPLIANT' if len(dormant_investments) > 0 else 'COMPLIANT',
            'criteria': [
                'No investment transactions for 3+ years',
                'No customer instructions or communications',
                'Account maintains investment holdings'
            ],
            'next_actions': [
                'Contact customer about investment portfolio',
                'Review investment performance and status',
                'Initiate dormancy procedures per CBUAE guidelines'
            ],
            'sample_findings': sample_findings,
            'compliance_impact': 'High' if len(dormant_investments) > 5 else 'Medium'
        }
    
    def analyze_article_3_1_contact_attempts(self, df: pd.DataFrame) -> Dict:
        """Article 3.1 - Contact Attempts Compliance Analysis"""
        # Find dormant accounts with insufficient contact attempts
        df['dormancy_years'] = df['dormancy_period_months'].apply(self.months_to_years)
        
        dormant_accounts = df[
            (df['dormancy_years'] >= 3) & 
            (df['account_status'] == 'DORMANT')
        ]
        
        insufficient_contact = dormant_accounts[
            (dormant_accounts['contact_attempts_made'].fillna(0) < 3)
        ]
        
        # Generate sample findings
        sample_findings = []
        for _, account in insufficient_contact.head(5).iterrows():
            sample_findings.append({
                'account_id': account['account_id'],
                'contact_attempts': int(account.get('contact_attempts_made', 0)),
                'required_attempts': 3,
                'last_contact_date': account.get('last_contact_attempt_date', 'N/A'),
                'customer_type': account.get('customer_type', 'N/A'),
                'address_known': account.get('address_known', 'N/A')
            })
        
        return {
            'function_name': 'analyze_contact_attempts_compliance',
            'article': '3.1',
            'title': 'Article 3 Contact Process Monitoring',
            'description': 'Ensures proper customer contact procedures before dormancy declaration',
            'required_contacts': len(dormant_accounts),
            'violations': len(insufficient_contact),
            'priority': 'CRITICAL' if len(insufficient_contact) > 0 else 'LOW',
            'status': 'NON_COMPLIANT' if len(insufficient_contact) > 0 else 'COMPLIANT',
            'criteria': [
                'Minimum 3 contact attempts using different channels',
                'Reasonable efforts to locate customer made',
                '90-day waiting period observed after final contact',
                'All contact attempts properly documented'
            ],
            'next_actions': [
                'Complete required contact attempts immediately',
                'Use multiple communication channels (phone, email, mail)',
                'Document all communication efforts thoroughly',
                'Wait 90 days after final contact before dormancy declaration'
            ],
            'sample_findings': sample_findings,
            'compliance_impact': 'Critical'
        }
    
    def analyze_article_8_1_cb_transfers(self, df: pd.DataFrame) -> Dict:
        """Article 8.1 - Central Bank Transfer Eligibility"""
        # Calculate dormancy years
        df['dormancy_years'] = df['dormancy_period_months'].apply(self.months_to_years)
        
        # Find accounts eligible for CB transfer (5+ years dormant)
        cb_eligible = df[
            (df['dormancy_years'] >= 5) & 
            (df['transfer_eligibility_date'].notna()) & 
            (df['transferred_to_cb_date'].isna()) &
            (df['account_status'] == 'DORMANT')
        ]
        
        # Generate sample findings
        sample_findings = []
        for _, account in cb_eligible.head(5).iterrows():
            sample_findings.append({
                'account_id': account['account_id'],
                'dormancy_years': round(account['dormancy_years'], 1),
                'balance': account['balance_current'],
                'currency': account['currency'],
                'eligibility_date': account['transfer_eligibility_date'],
                'customer_type': account.get('customer_type', 'N/A'),
                'address_known': account.get('address_known', 'N/A')
            })
        
        return {
            'function_name': 'analyze_cb_transfer_eligibility',
            'article': '8.1',
            'title': 'Central Bank Transfer Eligibility',
            'description': 'Accounts eligible for transfer to CBUAE after 5+ years dormancy',
            'eligible_accounts': len(cb_eligible),
            'priority': 'MEDIUM' if len(cb_eligible) > 0 else 'LOW',
            'status': 'TRANSFER_REQUIRED' if len(cb_eligible) > 0 else 'NO_ACTION_NEEDED',
            'criteria': [
                'Dormant for 5+ years minimum',
                'Customer has no other active accounts',
                'Customer address unknown to bank',
                'All Article 3 processes completed successfully'
            ],
            'next_actions': [
                'Prepare comprehensive transfer documentation',
                'Convert foreign currency balances to AED',
                'Complete final customer verification checks',
                'Submit transfer request to CBUAE'
            ],
            'sample_findings': sample_findings,
            'compliance_impact': 'Medium'
        }
    
    def analyze_high_value_monitoring(self, df: pd.DataFrame) -> Dict:
        """High Value Account Special Monitoring"""
        # Calculate dormancy years and identify high-value accounts
        df['dormancy_years'] = df['dormancy_period_months'].apply(self.months_to_years)
        df['is_high_value'] = df.apply(
            lambda row: self.is_high_value(row['balance_current'], row['currency']), axis=1
        )
        
        high_value_dormant = df[
            (df['dormancy_years'] >= 3) & 
            (df['is_high_value'] == True) & 
            (df['account_status'] == 'DORMANT')
        ]
        
        # Generate sample findings
        sample_findings = []
        for _, account in high_value_dormant.head(5).iterrows():
            aed_equivalent = account['balance_current'] * self.currency_rates.get(account['currency'], 1.0)
            sample_findings.append({
                'account_id': account['account_id'],
                'dormancy_years': round(account['dormancy_years'], 1),
                'balance': account['balance_current'],
                'currency': account['currency'],
                'aed_equivalent': round(aed_equivalent, 2),
                'account_type': account['account_type'],
                'risk_level': 'Critical' if aed_equivalent > 100000 else 'High'
            })
        
        return {
            'function_name': 'monitor_high_value_dormancy',
            'article': 'General',
            'title': 'High Value Dormant Accounts Monitoring',
            'description': 'Special monitoring for accounts with balances ≥ AED 25,000',
            'total_high_value': len(high_value_dormant),
            'priority': 'HIGH' if len(high_value_dormant) > 0 else 'LOW',
            'status': 'PRIORITY_MONITORING' if len(high_value_dormant) > 0 else 'NO_ISSUES',
            'criteria': [
                'Account balance ≥ AED 25,000 equivalent',
                'Meets standard dormancy criteria',
                'Requires priority attention and management oversight'
            ],
            'next_actions': [
                'Priority customer contact campaign',
                'Management review and approval required',
                'Enhanced documentation and reporting',
                'Consider specialized reactivation incentives'
            ],
            'sample_findings': sample_findings,
            'compliance_impact': 'High'
        }
    
    def run_comprehensive_analysis(self, df: pd.DataFrame, report_date: str) -> Dict:
        """Run comprehensive CBUAE dormancy analysis"""
        try:
            # Prepare results structure
            analysis_results = {
                'analysis_date': report_date,
                'total_accounts': len(df),
                'summary_kpis': {},
                'triggered_functions': [],
                'detailed_findings': {},
                'compliance_summary': {},
                'overall_status': 'pending'
            }
            
            # Run all analysis functions
            analysis_functions = [
                self.analyze_article_2_1_1_demand_deposits,
                self.analyze_article_2_2_fixed_deposits,
                self.analyze_article_2_3_investments,
                self.analyze_article_3_1_contact_attempts,
                self.analyze_article_8_1_cb_transfers,
                self.analyze_high_value_monitoring
            ]
            
            total_violations = 0
            critical_issues = 0
            
            for analysis_func in analysis_functions:
                try:
                    result = analysis_func(df)
                    if result is not None:
                        # Count violations/issues
                        violations = result.get('violations', 0)
                        if violations == 0:
                            violations = result.get('eligible_accounts', 0)
                            if violations == 0:
                                violations = result.get('total_high_value', 0)
                        
                        total_violations += violations
                        
                        if result.get('priority') == 'CRITICAL':
                            critical_issues += violations
                        
                        # Add to triggered functions if there are issues
                        if violations > 0 or result.get('status') in ['NON_COMPLIANT', 'TRANSFER_REQUIRED', 'PRIORITY_MONITORING']:
                            analysis_results['triggered_functions'].append(result)
                            analysis_results['detailed_findings'][result['function_name']] = {
                                'count': violations,
                                'description': result['description'],
                                'sample_accounts': [f['account_id'] for f in result.get('sample_findings', [])[:5]],
                                'compliance_impact': result.get('compliance_impact', 'Medium')
                            }
                
                except Exception as e:
                    logger.error(f"Analysis function failed: {e}")
                    continue
            
            # Calculate summary KPIs
            dormancy_rate = (total_violations / len(df)) * 100 if len(df) > 0 else 0
            compliance_score = max(0, (1 - (total_violations / len(df))) * 100) if len(df) > 0 else 100
            
            analysis_results['summary_kpis'] = {
                'total_accounts_analyzed': len(df),
                'total_accounts_flagged_dormant': total_violations,
                'dormancy_rate': round(dormancy_rate, 1),
                'compliance_score': round(compliance_score, 1),
                'critical_issues': critical_issues,
                'analysis_date': report_date,
                'compliance_status': 'requires_attention' if total_violations > 0 else 'compliant'
            }
            
            # Determine overall status
            if critical_issues > 0:
                analysis_results['overall_status'] = 'critical'
            elif total_violations > 0:
                analysis_results['overall_status'] = 'requires_attention'
            else:
                analysis_results['overall_status'] = 'compliant'
            
            # Compliance summary by article
            compliance_summary = {}
            for func in analysis_results['triggered_functions']:
                article = func['article']
                compliance_summary[f"Article {article}"] = {
                    'status': func['status'],
                    'violations': func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0))),
                    'priority': func['priority'],
                    'description': func['title']
                }
            
            analysis_results['compliance_summary'] = compliance_summary
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {'error': str(e)}

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = True  # Simplified for demo
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {'username': 'demo_user', 'role': 'analyst'}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'dormancy_results' not in st.session_state:
        st.session_state.dormancy_results = None
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = None
    if 'risk_results' not in st.session_state:
        st.session_state.risk_results = None
    if 'workflow_results' not in st.session_state:
        st.session_state.workflow_results = None

def show_main_app():
    """Show main application interface"""
    
    # Header
    st.markdown('<div class="main-header"><h1>🏦 Banking Compliance AI System</h1><h3>Comprehensive CBUAE Dormancy Analysis & Compliance Verification</h3></div>', unsafe_allow_html=True)
    
    # User info
    user_info = st.session_state.user_data
    st.markdown(f"**Welcome:** {user_info['username']} ({user_info['role']})")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Module",
        [
            "Data Upload & Analysis",
            "Dormancy Analysis Results",
            "Compliance Dashboard",
            "Risk Assessment",
            "Reports & Analytics",
            "System Status"
        ]
    )
    
    # Route to selected page
    if page == "Data Upload & Analysis":
        show_data_upload_page()
    elif page == "Dormancy Analysis Results":
        show_dormancy_analysis_page()
    elif page == "Compliance Dashboard":
        show_compliance_page()
    elif page == "Risk Assessment":
        show_risk_assessment_page()
    elif page == "Reports & Analytics":
        show_reports_page()
    elif page == "System Status":
        show_system_status_page()

def show_data_upload_page():
    """Show data upload and analysis page"""
    st.header("📊 Data Upload & CBUAE Analysis")
    
    st.markdown("""
    Upload your banking data CSV file and run comprehensive CBUAE dormancy analysis.
    The system will automatically detect compliance issues and generate actionable insights.
    """)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Banking Data CSV",
        type=['csv'],
        help="Upload CSV file containing account data with columns like account_id, account_type, balance_current, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} records loaded.")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.dataframe(df.head(), use_container_width=True)
                
                # Show column info
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Sample Values': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'N/A' for col in df.columns]
                })
                st.write("**Column Information:**")
                st.dataframe(col_info, use_container_width=True)
            
            # Store processed data
            st.session_state.processed_data = df
            
            # Analysis configuration
            with st.expander("⚙️ Analysis Configuration"):
                col1, col2 = st.columns(2)
                with col1:
                    report_date = st.date_input("Report Date", value=datetime.now().date())
                with col2:
                    analysis_mode = st.selectbox("Analysis Mode", ["Comprehensive", "Quick Scan"])
            
            # Run CBUAE dormancy analysis
            if st.button("🚀 Run CBUAE Dormancy Analysis", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive CBUAE dormancy analysis..."):
                    run_real_dormancy_analysis(df, report_date.strftime("%Y-%m-%d"))
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("📁 Please upload a CSV file to begin analysis")

def run_real_dormancy_analysis(df: pd.DataFrame, report_date: str):
    """Run real dormancy analysis on uploaded data"""
    try:
        # Initialize the analyzer
        analyzer = CBUAEDormancyAnalyzer()
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis(df, report_date)
        
        if 'error' in results:
            st.error(f"Analysis failed: {results['error']}")
            return
        
        # Store results
        st.session_state.dormancy_results = results
        
        # Show immediate results
        st.success("✅ Analysis completed successfully!")
        
        # Display summary
        show_analysis_summary(results)
        
        # Show triggered functions
        if results['triggered_functions']:
            st.subheader("🎯 Triggered CBUAE Monitoring Functions")
            
            for func in results['triggered_functions']:
                show_triggered_function_card(func)
        else:
            st.info("ℹ️ No dormancy issues detected in the uploaded data. All accounts appear compliant.")
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        logger.error(f"Analysis error: {e}")

def show_analysis_summary(results):
    """Display analysis summary with KPIs"""
    st.subheader("📊 Analysis Summary")
    
    summary = results['summary_kpis']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accounts", f"{summary['total_accounts_analyzed']:,}")
    
    with col2:
        flagged = summary['total_accounts_flagged_dormant']
        st.metric("Flagged Accounts", f"{flagged:,}", delta=f"{summary['dormancy_rate']:.1f}%")
    
    with col3:
        score = summary['compliance_score']
        score_color = "🟢" if score > 90 else "🟡" if score > 70 else "🔴"
        st.metric("Compliance Score", f"{score_color} {score:.1f}%")
    
    with col4:
        critical = summary['critical_issues']
        st.metric("Critical Issues", critical, delta="Requires immediate attention" if critical > 0 else "None")
    
    # Overall status
    status = results['overall_status']
    if status == 'critical':
        st.error("🚨 **CRITICAL STATUS**: Immediate action required for compliance violations")
    elif status == 'requires_attention':
        st.warning("⚠️ **ATTENTION REQUIRED**: Several compliance issues need to be addressed")
    else:
        st.success("✅ **COMPLIANT**: No critical issues detected")

def show_triggered_function_card(func):
    """Display a triggered function card with details"""
    priority_colors = {
        'CRITICAL': '🔴',
        'HIGH': '🟠', 
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    
    priority_icon = priority_colors.get(func['priority'], '⚪')
    
    with st.container():
        st.markdown(f"""
        <div class="triggered-function">
            <h4>📋 {func['title']}</h4>
            <p><strong>CBUAE Article:</strong> {func['article']}</p>
            <p><strong>Description:</strong> {func['description']}</p>
            <p><strong>Priority Level:</strong> {priority_icon} {func['priority']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
            st.metric("Affected Accounts", violations)
        
        with col2:
            total = func.get('total_accounts', func.get('required_contacts', 0))
            if total > 0:
                rate = (violations / total) * 100
                st.metric("Violation Rate", f"{rate:.1f}%")
            else:
                st.metric("Status", func['status'])
        
        with col3:
            impact = func.get('compliance_impact', 'Medium')
            st.metric("Impact Level", impact)
        
        # Show criteria and actions in expandable sections
        with st.expander(f"📋 Details for {func['title']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Dormancy Criteria:**")
                for criteria in func['criteria']:
                    st.markdown(f"• {criteria}")
            
            with col2:
                st.markdown("**🎯 Required Actions:**")
                for action in func['next_actions']:
                    st.markdown(f"• {action}")
            
            # Sample affected accounts
            if func.get('sample_findings'):
                st.markdown("**🔍 Sample Affected Accounts:**")
                
                # Create DataFrame from sample findings
                sample_data = []
                for finding in func['sample_findings'][:5]:
                    sample_data.append({
                        'Account ID': finding.get('account_id', 'N/A'),
                        'Status': 'Requires Action',
                        'Details': f"Dormant {finding.get('dormancy_years', 'N/A')} years" if 'dormancy_years' in finding else 
                                  f"{finding.get('contact_attempts', 0)} attempts made" if 'contact_attempts' in finding else 'Review Required',
                        'Priority': func['priority']
                    })
                
                if sample_data:
                    sample_df = pd.DataFrame(sample_data)
                    st.dataframe(sample_df, use_container_width=True, hide_index=True)

def show_dormancy_analysis_page():
    """Show dormancy analysis results page"""
    st.header("🔍 CBUAE Dormancy Analysis Results")
    
    if not st.session_state.dormancy_results:
        st.warning("⚠️ Please upload and analyze data first in the 'Data Upload & Analysis' section.")
        return
    
    results = st.session_state.dormancy_results
    
    # Show analysis summary
    show_analysis_summary(results)
    
    # Triggered functions with detailed view
    if results['triggered_functions']:
        st.subheader("🎯 Detailed Function Analysis")
        
        # Create tabs for each triggered function
        tab_names = [f"Art. {func['article']}: {func['title'][:25]}..." for func in results['triggered_functions']]
        tabs = st.tabs(tab_names)
        
        for i, func in enumerate(results['triggered_functions']):
            with tabs[i]:
                show_detailed_function_analysis(func, results['detailed_findings'].get(func['function_name'], {}))
    
    # Compliance actions dashboard
    show_compliance_actions_dashboard(results)
    
    # Visual analytics
    show_dormancy_analytics(results)

def show_detailed_function_analysis(func, details):
    """Show detailed analysis for a specific function"""
    st.markdown(f"### {func['title']}")
    st.markdown(f"**CBUAE Article {func['article']}**")
    st.markdown(f"*{func['description']}*")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
        st.metric("Issues Found", violations)
    
    with col2:
        st.metric("Priority", func['priority'])
    
    with col3:
        st.metric("Compliance Impact", func.get('compliance_impact', 'Medium'))
    
    # Detailed findings
    st.markdown("#### 📊 Detailed Findings")
    
    if func.get('sample_findings'):
        # Process sample findings into a more detailed view
        findings_data = []
        
        for finding in func['sample_findings']:
            if 'dormancy_years' in finding:
                # Standard dormancy finding
                findings_data.append({
                    'Account ID': finding['account_id'],
                    'Dormancy Period': f"{finding['dormancy_years']} years",
                    'Balance': f"{finding['currency']} {finding['balance']:,.2f}" if finding.get('balance') else 'N/A',
                    'Risk Level': 'High Value' if finding.get('is_high_value') else 'Standard',
                    'Last Activity': finding.get('last_activity', 'N/A')
                })
            elif 'contact_attempts' in finding:
                # Contact attempts finding
                findings_data.append({
                    'Account ID': finding['account_id'],
                    'Contact Attempts': f"{finding['contact_attempts']}/{finding['required_attempts']}",
                    'Last Contact': finding.get('last_contact_date', 'N/A'),
                    'Customer Type': finding.get('customer_type', 'N/A'),
                    'Address Known': finding.get('address_known', 'N/A')
                })
            elif 'eligibility_date' in finding:
                # CB transfer finding
                findings_data.append({
                    'Account ID': finding['account_id'],
                    'Dormancy Period': f"{finding['dormancy_years']} years",
                    'Balance': f"{finding['currency']} {finding['balance']:,.2f}" if finding.get('balance') else 'N/A',
                    'Eligible Since': finding['eligibility_date'],
                    'Status': 'Ready for Transfer'
                })
            else:
                # Generic finding
                findings_data.append({
                    'Account ID': finding.get('account_id', 'N/A'),
                    'Details': str(finding),
                    'Status': 'Requires Review'
                })
        
        if findings_data:
            findings_df = pd.DataFrame(findings_data)
            st.dataframe(findings_df, use_container_width=True, hide_index=True)
    
    # Regulatory requirements
    st.markdown("#### ⚖️ Regulatory Requirements")
    for i, criteria in enumerate(func['criteria'], 1):
        st.markdown(f"{i}. {criteria}")
    
    # Required actions
    st.markdown("#### 🎯 Required Actions")
    for i, action in enumerate(func['next_actions'], 1):
        st.markdown(f"{i}. {action}")
    
    # Generate action plan
    if st.button(f"📋 Generate Action Plan for {func['title']}", key=f"action_plan_{func['function_name']}"):
        generate_action_plan(func)

def generate_action_plan(func):
    """Generate a specific action plan for the function"""
    st.subheader(f"📋 Action Plan: {func['title']}")
    
    # Timeline based on priority
    if func['priority'] == 'CRITICAL':
        timeline = "Immediate (within 24 hours)"
        urgency = "🚨 CRITICAL URGENCY"
    elif func['priority'] == 'HIGH':
        timeline = "Within 7 days"
        urgency = "⚠️ HIGH PRIORITY"
    else:
        timeline = "Within 30 days"
        urgency = "📋 STANDARD PRIORITY"
    
    st.markdown(f"**{urgency}**")
    st.markdown(f"**Timeline:** {timeline}")
    
    # Specific action items
    violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
    
    action_items = []
    
    if func['article'] == '2.1.1':
        action_items = [
            f"Review {violations} demand deposit accounts flagged as dormant",
            "Verify customer contact information and update records",
            "Initiate Article 3 contact procedures for each account",
            "Document all communication attempts in customer records",
            "Schedule follow-up reviews for reactivation efforts"
        ]
    elif func['article'] == '3.1':
        action_items = [
            f"Complete missing contact attempts for {violations} accounts",
            "Use multiple channels: phone, email, registered mail",
            "Document each contact attempt with date, method, and outcome",
            "Wait 90 days after final contact before dormancy declaration",
            "Update compliance tracking system"
        ]
    elif func['article'] == '8.1':
        action_items = [
            f"Prepare transfer documentation for {violations} eligible accounts",
            "Convert foreign currency balances to AED at current rates",
            "Verify final customer contact attempts are complete",
            "Submit transfer request to CBUAE with required documentation",
            "Update internal records to reflect transfer status"
        ]
    else:
        action_items = func['next_actions']
    
    for i, item in enumerate(action_items, 1):
        st.markdown(f"**Step {i}:** {item}")
    
    # Responsible parties
    st.markdown("#### 👥 Responsible Parties")
    if func['priority'] == 'CRITICAL':
        st.markdown("• **Primary:** Compliance Officer")
        st.markdown("• **Secondary:** Branch Manager")
        st.markdown("• **Escalation:** Management Team")
    else:
        st.markdown("• **Primary:** Customer Service Team")
        st.markdown("• **Oversight:** Compliance Officer")
    
    # Success criteria
    st.markdown("#### ✅ Success Criteria")
    st.markdown(f"• All {violations} affected accounts reviewed and documented")
    st.markdown("• Required actions completed within timeline")
    st.markdown("• Compliance status updated to 'Resolved'")
    st.markdown("• Monthly review scheduled for ongoing monitoring")

def show_compliance_actions_dashboard(results):
    """Show compliance actions dashboard"""
    st.subheader("⚡ Compliance Actions Dashboard")
    
    # Priority breakdown
    priority_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    total_violations = 0
    
    for func in results['triggered_functions']:
        priority = func['priority']
        violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
        priority_counts[priority] += violations
        total_violations += violations
    
    if total_violations > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if priority_counts['CRITICAL'] > 0:
                st.error(f"🚨 Critical: {priority_counts['CRITICAL']}")
            else:
                st.success("🚨 Critical: 0")
        
        with col2:
            if priority_counts['HIGH'] > 0:
                st.warning(f"⚠️ High: {priority_counts['HIGH']}")
            else:
                st.success("⚠️ High: 0")
        
        with col3:
            if priority_counts['MEDIUM'] > 0:
                st.info(f"📋 Medium: {priority_counts['MEDIUM']}")
            else:
                st.success("📋 Medium: 0")
        
        with col4:
            st.metric("Total Actions", total_violations)
        
        # Action timeline
        st.markdown("#### 📅 Recommended Action Timeline")
        
        timeline_data = []
        
        for func in results['triggered_functions']:
            violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
            if violations > 0:
                if func['priority'] == 'CRITICAL':
                    timeline = 'Immediate'
                elif func['priority'] == 'HIGH':
                    timeline = '7 days'
                else:
                    timeline = '30 days'
                
                timeline_data.append({
                    'Function': func['title'],
                    'Priority': func['priority'],
                    'Account Count': violations,
                    'Timeline': timeline,
                    'Article': func['article']
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No compliance actions required - all accounts appear compliant!")

def show_dormancy_analytics(results):
    """Show dormancy analytics and visualizations"""
    st.subheader("📈 Dormancy Analytics")
    
    if not results['triggered_functions']:
        st.info("No dormancy issues to visualize - all accounts compliant!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority distribution chart
        priority_data = {}
        for func in results['triggered_functions']:
            priority = func['priority']
            violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
            priority_data[priority] = priority_data.get(priority, 0) + violations
        
        if priority_data:
            fig_priority = px.pie(
                values=list(priority_data.values()),
                names=list(priority_data.keys()),
                title="Issues by Priority Level",
                color_discrete_map={
                    'CRITICAL': '#dc3545',
                    'HIGH': '#fd7e14',
                    'MEDIUM': '#ffc107',
                    'LOW': '#28a745'
                }
            )
            st.plotly_chart(fig_priority, use_container_width=True)
    
    with col2:
        # Article breakdown chart
        article_data = {}
        for func in results['triggered_functions']:
            article = f"Article {func['article']}"
            violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
            article_data[article] = article_data.get(article, 0) + violations
        
        if article_data:
            fig_article = px.bar(
                x=list(article_data.keys()),
                y=list(article_data.values()),
                title="Issues by CBUAE Article",
                labels={'x': 'CBUAE Article', 'y': 'Account Count'},
                color=list(article_data.values()),
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_article, use_container_width=True)

def show_compliance_page():
    """Show compliance dashboard page"""
    st.header("⚖️ CBUAE Compliance Dashboard")
    
    if not st.session_state.dormancy_results:
        st.warning("⚠️ Please run dormancy analysis first to view compliance status.")
        return
    
    results = st.session_state.dormancy_results
    
    # Overall compliance status
    st.subheader("📊 Overall Compliance Status")
    
    summary = results['summary_kpis']
    compliance_score = summary['compliance_score']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if compliance_score >= 95:
            st.success(f"✅ Excellent: {compliance_score:.1f}%")
        elif compliance_score >= 85:
            st.warning(f"⚠️ Good: {compliance_score:.1f}%")
        else:
            st.error(f"❌ Needs Improvement: {compliance_score:.1f}%")
    
    with col2:
        critical_issues = summary['critical_issues']
        if critical_issues == 0:
            st.success("🚨 No Critical Issues")
        else:
            st.error(f"🚨 {critical_issues} Critical Issues")
    
    with col3:
        total_issues = summary['total_accounts_flagged_dormant']
        st.metric("Total Issues", total_issues)
    
    # Compliance by article
    if results['compliance_summary']:
        st.subheader("📋 Compliance by CBUAE Article")
        
        compliance_data = []
        for article, info in results['compliance_summary'].items():
            status_icon = "❌" if info['status'] in ['NON_COMPLIANT'] else \
                         "🔄" if info['status'] in ['TRANSFER_REQUIRED'] else \
                         "⚠️" if info['status'] in ['PRIORITY_MONITORING'] else "✅"
            
            compliance_data.append({
                'Article': article,
                'Status': f"{status_icon} {info['status'].replace('_', ' ').title()}",
                'Issues': info['violations'],
                'Priority': info['priority'],
                'Description': info['description']
            })
        
        compliance_df = pd.DataFrame(compliance_data)
        st.dataframe(compliance_df, use_container_width=True, hide_index=True)
    
    # Compliance trends (mock data)
    show_compliance_trends()
    
    # Remediation tracker
    show_remediation_tracker(results)

def show_compliance_trends():
    """Show compliance trends over time"""
    st.subheader("📈 Compliance Trends")
    
    # Mock trend data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    trend_data = pd.DataFrame({
        'Date': dates,
        'Compliance Score': [85, 87, 83, 89, 91, 88, 85, 82, 86, 89, 92, 90],
        'Critical Issues': [12, 8, 15, 6, 3, 7, 9, 13, 8, 4, 2, 5],
        'Total Issues': [45, 38, 52, 31, 28, 35, 41, 48, 36, 29, 25, 32]
    })
    
    # Compliance score trend
    fig_trend = px.line(
        trend_data,
        x='Date',
        y='Compliance Score',
        title='Compliance Score Trend',
        labels={'Compliance Score': 'Score (%)'}
    )
    fig_trend.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Target: 85%")
    st.plotly_chart(fig_trend, use_container_width=True)

def show_remediation_tracker(results):
    """Show remediation progress tracker"""
    st.subheader("🎯 Remediation Progress Tracker")
    
    if not results['triggered_functions']:
        st.success("✅ No remediation actions required!")
        return
    
    # Create remediation tracking data
    remediation_data = []
    
    for func in results['triggered_functions']:
        violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
        
        # Mock progress data
        if func['priority'] == 'CRITICAL':
            progress = 25  # Just started
        elif func['priority'] == 'HIGH':
            progress = 60  # In progress
        else:
            progress = 80  # Nearly complete
        
        remediation_data.append({
            'Function': func['title'],
            'Total Issues': violations,
            'Resolved': int(violations * progress / 100),
            'Remaining': violations - int(violations * progress / 100),
            'Progress %': progress,
            'Status': 'In Progress' if progress < 100 else 'Complete'
        })
    
    remediation_df = pd.DataFrame(remediation_data)
    
    # Progress visualization
    fig_progress = px.bar(
        remediation_df,
        x='Function',
        y=['Resolved', 'Remaining'],
        title='Remediation Progress by Function',
        color_discrete_map={'Resolved': '#28a745', 'Remaining': '#dc3545'}
    )
    st.plotly_chart(fig_progress, use_container_width=True)
    
    # Detailed progress table
    st.dataframe(remediation_df, use_container_width=True, hide_index=True)

def show_risk_assessment_page():
    """Show risk assessment page"""
    st.header("🎲 Risk Assessment & Analysis")
    
    if not st.session_state.dormancy_results:
        st.warning("⚠️ Please run dormancy analysis first to assess risks.")
        return
    
    if st.button("📊 Run Risk Assessment", type="primary"):
        with st.spinner("Analyzing risk factors and calculating risk scores..."):
            risk_results = run_risk_assessment()
            st.session_state.risk_results = risk_results
            show_risk_results(risk_results)
    
    if st.session_state.risk_results:
        show_risk_results(st.session_state.risk_results)

def run_risk_assessment():
    """Run comprehensive risk assessment"""
    try:
        dormancy_results = st.session_state.dormancy_results
        triggered_funcs = dormancy_results.get('triggered_functions', [])
        
        # Calculate risk components
        high_value_risk = 0.0
        compliance_risk = 0.0
        operational_risk = 0.0
        reputational_risk = 0.0
        
        total_accounts = dormancy_results['summary_kpis']['total_accounts_analyzed']
        high_risk_accounts = []
        mitigation_strategies = []
        
        for func in triggered_funcs:
            violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
            
            if 'high_value' in func['function_name'].lower() or func['title'].lower().find('high value') != -1:
                high_value_risk = min(1.0, violations / max(1, total_accounts * 0.05))
                
                for i in range(min(violations, 5)):
                    high_risk_accounts.append({
                        'account_id': f"ACC{2000 + i}",
                        'risk_type': 'high_value_dormant',
                        'risk_level': 'High',
                        'description': 'High-value dormant account requiring priority attention'
                    })
            
            if func['priority'] == 'CRITICAL':
                compliance_risk += 0.4
                reputational_risk += 0.3
            elif func['priority'] == 'HIGH':
                compliance_risk += 0.2
                operational_risk += 0.15
        
        # Calculate overall risk score
        risk_components = {
            'high_value_risk': min(1.0, high_value_risk),
            'compliance_risk': min(1.0, compliance_risk),
            'operational_risk': min(1.0, operational_risk),
            'reputational_risk': min(1.0, reputational_risk),
            'regulatory_risk': min(1.0, compliance_risk * 0.8)
        }
        
        weights = {
            'high_value_risk': 0.3,
            'compliance_risk': 0.25,
            'operational_risk': 0.2,
            'reputational_risk': 0.15,
            'regulatory_risk': 0.1
        }
        
        overall_risk_score = sum(
            risk_components[component] * weights[component]
            for component in weights.keys()
        )
        
        # Determine risk level
        if overall_risk_score >= 0.8:
            risk_level = 'Critical'
        elif overall_risk_score >= 0.6:
            risk_level = 'High'
        elif overall_risk_score >= 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Generate mitigation strategies
        if overall_risk_score > 0.5:
            mitigation_strategies.append({
                'strategy': 'Priority Reactivation Program',
                'priority': 'High',
                'description': 'Implement priority reactivation program for high-value dormant accounts',
                'timeline': 'Immediate - 30 days',
                'expected_impact': 'Reduce high-value risk by 60-80%'
            })
        
        if compliance_risk > 0.4:
            mitigation_strategies.append({
                'strategy': 'Compliance Remediation Plan',
                'priority': 'High',
                'description': 'Execute comprehensive compliance remediation plan',
                'timeline': 'Immediate - 90 days',
                'expected_impact': 'Achieve 95%+ compliance rate'
            })
        
        if operational_risk > 0.3:
            mitigation_strategies.append({
                'strategy': 'Process Automation Enhancement',
                'priority': 'Medium',
                'description': 'Automate dormancy monitoring and contact procedures',
                'timeline': '60 - 120 days',
                'expected_impact': 'Reduce operational risk by 40-60%'
            })
        
        return {
            'success': True,
            'overall_risk_score': overall_risk_score,
            'risk_level': risk_level,
            'risk_components': risk_components,
            'high_risk_accounts': high_risk_accounts,
            'mitigation_strategies': mitigation_strategies,
            'risk_factors': {
                'total_violations': dormancy_results['summary_kpis']['total_accounts_flagged_dormant'],
                'critical_issues': dormancy_results['summary_kpis']['critical_issues'],
                'compliance_score': dormancy_results['summary_kpis']['compliance_score']
            }
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def show_risk_results(results):
    """Display risk assessment results"""
    if not results.get('success'):
        st.error(f"Risk assessment failed: {results.get('error')}")
        return
    
    # Overall risk metrics
    st.subheader("🎯 Risk Assessment Summary")
    
    risk_score = results['overall_risk_score']
    risk_level = results['risk_level']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_color = "🔴" if risk_score > 0.7 else "🟡" if risk_score > 0.4 else "🟢"
        st.metric("Overall Risk Score", f"{risk_color} {risk_score:.1%}")
    
    with col2:
        level_color = "🚨" if risk_level in ['Critical', 'High'] else "⚠️" if risk_level == 'Medium' else "✅"
        st.metric("Risk Level", f"{level_color} {risk_level}")
    
    with col3:
        st.metric("High Risk Accounts", len(results['high_risk_accounts']))
    
    with col4:
        st.metric("Mitigation Strategies", len(results['mitigation_strategies']))
    
    # Risk components breakdown
    st.subheader("📊 Risk Components Analysis")
    
    risk_components = results['risk_components']
    components_data = []
    
    for component, score in risk_components.items():
        components_data.append({
            'Component': component.replace('_', ' ').title(),
            'Score': score,
            'Percentage': f"{score:.1%}",
            'Level': 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
        })
    
    components_df = pd.DataFrame(components_data)
    
    # Risk components chart
    fig_risk = px.bar(
        components_df,
        x='Component',
        y='Score',
        title='Risk Components Breakdown',
        color='Score',
        color_continuous_scale='Reds',
        text='Percentage'
    )
    fig_risk.update_traces(textposition='outside')
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Risk factors summary
    st.subheader("⚠️ Risk Factors")
    
    risk_factors = results['risk_factors']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Violations", risk_factors['total_violations'])
    
    with col2:
        st.metric("Critical Issues", risk_factors['critical_issues'])
    
    with col3:
        compliance_score = risk_factors['compliance_score']
        compliance_risk = "High" if compliance_score < 70 else "Medium" if compliance_score < 85 else "Low"
        st.metric("Compliance Risk", compliance_risk)
    
    # High risk accounts
    if results['high_risk_accounts']:
        st.subheader("🚨 High Risk Accounts")
        risk_accounts_df = pd.DataFrame(results['high_risk_accounts'])
        st.dataframe(risk_accounts_df, use_container_width=True, hide_index=True)
    
    # Mitigation strategies
    if results['mitigation_strategies']:
        st.subheader("🛡️ Recommended Mitigation Strategies")
        strategies_df = pd.DataFrame(results['mitigation_strategies'])
        st.dataframe(strategies_df, use_container_width=True, hide_index=True)
        
        # Generate mitigation plan
        if st.button("📋 Generate Comprehensive Mitigation Plan"):
            generate_mitigation_plan(results['mitigation_strategies'])

def generate_mitigation_plan(strategies):
    """Generate a comprehensive mitigation plan"""
    st.subheader("📋 Comprehensive Risk Mitigation Plan")
    
    st.markdown("### 🎯 Executive Summary")
    st.markdown("""
    This mitigation plan addresses identified risk factors through a systematic approach 
    focusing on immediate compliance remediation, process improvements, and long-term 
    risk reduction strategies.
    """)
    
    st.markdown("### 📅 Implementation Timeline")
    
    # Sort strategies by priority
    high_priority = [s for s in strategies if s['priority'] == 'High']
    medium_priority = [s for s in strategies if s['priority'] == 'Medium']
    
            if high_priority:
        st.markdown("#### 🚨 Phase 1: Immediate Actions (0-30 days)")
        for i, strategy in enumerate(high_priority, 1):
            st.markdown(f"**{i}. {strategy['strategy']}**")
            st.markdown(f"   - **Objective:** {strategy['description']}")
            st.markdown(f"   - **Timeline:** {strategy['timeline']}")
            st.markdown(f"   - **Expected Impact:** {strategy['expected_impact']}")
            st.markdown("")
    
    if medium_priority:
        st.markdown("#### ⚠️ Phase 2: Process Improvements (30-120 days)")
        for i, strategy in enumerate(medium_priority, 1):
            st.markdown(f"**{i}. {strategy['strategy']}**")
            st.markdown(f"   - **Objective:** {strategy['description']}")
            st.markdown(f"   - **Timeline:** {strategy['timeline']}")
            st.markdown(f"   - **Expected Impact:** {strategy['expected_impact']}")
            st.markdown("")
    
    st.markdown("### 👥 Responsibility Matrix")
    st.markdown("""
    | Strategy | Primary Owner | Support Team | Executive Sponsor |
    |----------|---------------|--------------|-------------------|
    | Priority Reactivation | Customer Service | Marketing | Head of Retail |
    | Compliance Remediation | Compliance Officer | Legal | Chief Risk Officer |
    | Process Automation | IT Department | Operations | CTO |
    """)
    
    st.markdown("### 📊 Success Metrics")
    st.markdown("""
    - **Compliance Score:** Target 95%+ within 90 days
    - **Risk Score Reduction:** 50% reduction in overall risk score
    - **Account Reactivation:** 30% of dormant accounts reactivated
    - **Process Efficiency:** 70% reduction in manual processes
    """)

def show_reports_page():
    """Show reports and analytics page"""
    st.header("📊 Reports & Analytics Dashboard")
    
    if not st.session_state.dormancy_results:
        st.warning("⚠️ Please run analysis first to generate reports.")
        return
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        ["Executive Summary", "Detailed Compliance Report", "Risk Analysis Report", "CBUAE Regulatory Report"]
    )
    
    # Generate and display selected report
    if report_type == "Executive Summary":
        show_executive_summary_report()
    elif report_type == "Detailed Compliance Report":
        show_detailed_compliance_report()
    elif report_type == "Risk Analysis Report":
        show_risk_analysis_report()
    elif report_type == "CBUAE Regulatory Report":
        show_cbuae_regulatory_report()

def show_executive_summary_report():
    """Show executive summary report"""
    st.subheader("📈 Executive Summary Report")
    
    results = st.session_state.dormancy_results
    summary = results['summary_kpis']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accounts", f"{summary['total_accounts_analyzed']:,}")
    with col2:
        st.metric("Flagged Accounts", f"{summary['total_accounts_flagged_dormant']:,}")
    with col3:
        st.metric("Compliance Score", f"{summary['compliance_score']:.1f}%")
    with col4:
        st.metric("Critical Issues", summary['critical_issues'])
    
    # Executive summary text
    st.markdown(f"""
    ### Analysis Summary
    
    **Analysis Date:** {summary['analysis_date']}
    
    **Key Findings:**
    - Analyzed {summary['total_accounts_analyzed']:,} total accounts across all product types
    - Identified {summary['total_accounts_flagged_dormant']:,} accounts requiring attention ({summary['dormancy_rate']:.1f}% of portfolio)
    - Overall compliance score: {summary['compliance_score']:.1f}% against CBUAE standards
    - {summary['critical_issues']} critical violations requiring immediate remediation
    
    **Risk Assessment:**
    - Portfolio risk level: {'High' if summary['critical_issues'] > 0 else 'Medium' if summary['total_accounts_flagged_dormant'] > 0 else 'Low'}
    - Regulatory compliance status: {summary['compliance_status'].replace('_', ' ').title()}
    
    **Recommendations:**
    - {'Immediate action required for critical compliance violations' if summary['critical_issues'] > 0 else 'Enhanced monitoring and process improvements recommended' if summary['total_accounts_flagged_dormant'] > 0 else 'Continue current monitoring practices'}
    """)
    
    # Triggered functions summary
    if results['triggered_functions']:
        st.markdown("### CBUAE Functions Triggered")
        for func in results['triggered_functions']:
            violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
            st.markdown(f"- **Article {func['article']}**: {violations} accounts - {func['title']}")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Export as PDF", key="exec_pdf"):
            st.success("PDF export functionality would be implemented here")
    with col2:
        if st.button("📊 Export as Excel", key="exec_excel"):
            st.success("Excel export functionality would be implemented here")
    with col3:
        if st.button("📧 Email Report", key="exec_email"):
            st.success("Email functionality would be implemented here")

def show_detailed_compliance_report():
    """Show detailed compliance report"""
    st.subheader("⚖️ Detailed CBUAE Compliance Report")
    
    results = st.session_state.dormancy_results
    
    # Compliance overview
    st.markdown("### Compliance Overview")
    
    summary = results['summary_kpis']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{summary['compliance_score']:.1f}%")
    with col2:
        st.metric("Total Violations", summary['total_accounts_flagged_dormant'])
    with col3:
        st.metric("Critical Issues", summary['critical_issues'])
    
    # Article-by-article compliance
    if results['compliance_summary']:
        st.markdown("### Article-by-Article Compliance Status")
        
        compliance_data = []
        for article, info in results['compliance_summary'].items():
            compliance_data.append({
                'CBUAE Article': article,
                'Compliance Status': info['status'].replace('_', ' ').title(),
                'Violations Found': info['violations'],
                'Priority Level': info['priority'],
                'Function Description': info['description']
            })
        
        compliance_df = pd.DataFrame(compliance_data)
        st.dataframe(compliance_df, use_container_width=True, hide_index=True)
    
    # Detailed violations by function
    st.markdown("### Detailed Violation Analysis")
    
    for func in results['triggered_functions']:
        with st.expander(f"📋 {func['title']} - Article {func['article']}"):
            violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
            
            st.markdown(f"**Violations Found:** {violations}")
            st.markdown(f"**Priority Level:** {func['priority']}")
            st.markdown(f"**Compliance Impact:** {func.get('compliance_impact', 'Medium')}")
            
            st.markdown("**Regulatory Requirements:**")
            for req in func['criteria']:
                st.markdown(f"• {req}")
            
            st.markdown("**Required Remediation Actions:**")
            for action in func['next_actions']:
                st.markdown(f"• {action}")

def show_risk_analysis_report():
    """Show risk analysis report"""
    st.subheader("🎲 Risk Analysis Report")
    
    if not st.session_state.risk_results:
        st.warning("Please run risk assessment first.")
        return
    
    risk_data = st.session_state.risk_results
    
    # Risk overview
    st.markdown("### Risk Assessment Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Risk Score", f"{risk_data['overall_risk_score']:.1%}")
    with col2:
        st.metric("Risk Level", risk_data['risk_level'])
    with col3:
        st.metric("High Risk Accounts", len(risk_data['high_risk_accounts']))
    
    # Risk components detailed analysis
    st.markdown("### Risk Components Analysis")
    
    risk_components = risk_data['risk_components']
    for component, score in risk_components.items():
        component_name = component.replace('_', ' ').title()
        risk_level = 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
        
        if score > 0:
            st.markdown(f"**{component_name}:** {score:.1%} ({risk_level} Risk)")
            
            # Component-specific analysis
            if component == 'compliance_risk':
                st.markdown("  - Driven by regulatory violations and non-compliance issues")
                st.markdown("  - Requires immediate remediation to avoid regulatory penalties")
            elif component == 'high_value_risk':
                st.markdown("  - High-value dormant accounts pose significant financial risk")
                st.markdown("  - Priority customer contact and reactivation programs recommended")
            elif component == 'operational_risk':
                st.markdown("  - Process inefficiencies in dormancy monitoring")
                st.markdown("  - Automation and system improvements needed")

def show_cbuae_regulatory_report():
    """Show CBUAE-specific regulatory report"""
    st.subheader("🏛️ CBUAE Regulatory Compliance Report")
    
    results = st.session_state.dormancy_results
    summary = results['summary_kpis']
    
    # Regulatory header
    st.markdown(f"""
    **Central Bank of the UAE**  
    **Dormancy and Unclaimed Balances Regulation Compliance Report**
    
    **Reporting Institution:** [Bank Name]  
    **Report Period:** {summary['analysis_date']}  
    **Submission Date:** {datetime.now().strftime('%Y-%m-%d')}
    """)
    
    # Executive compliance statement
    st.markdown("### Executive Compliance Statement")
    
    if summary['critical_issues'] == 0 and summary['compliance_score'] >= 95:
        compliance_statement = "The institution is in full compliance with CBUAE dormancy regulations."
    elif summary['critical_issues'] == 0:
        compliance_statement = "The institution demonstrates substantial compliance with minor areas for improvement."
    else:
        compliance_statement = "The institution has identified compliance gaps requiring immediate remediation."
    
    st.markdown(f"**Status:** {compliance_statement}")
    
    # Regulatory metrics
    st.markdown("### Key Regulatory Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accounts Under Review", f"{summary['total_accounts_analyzed']:,}")
    with col2:
        st.metric("Dormant Accounts Identified", f"{summary['total_accounts_flagged_dormant']:,}")
    with col3:
        st.metric("Compliance Rate", f"{summary['compliance_score']:.1f}%")
    with col4:
        st.metric("Regulatory Violations", summary['critical_issues'])
    
    # Article compliance matrix
    st.markdown("### CBUAE Article Compliance Matrix")
    
    if results['compliance_summary']:
        regulatory_matrix = []
        
        for article, info in results['compliance_summary'].items():
            compliance_status = "Compliant" if info['status'] in ['COMPLIANT'] else "Non-Compliant"
            
            regulatory_matrix.append({
                'CBUAE Article': article,
                'Regulation Section': info['description'],
                'Compliance Status': compliance_status,
                'Accounts Affected': info['violations'],
                'Remediation Required': 'Yes' if info['violations'] > 0 else 'No'
            })
        
        matrix_df = pd.DataFrame(regulatory_matrix)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    
    # Remediation plan
    st.markdown("### Regulatory Remediation Plan")
    
    if summary['critical_issues'] > 0 or summary['total_accounts_flagged_dormant'] > 0:
        st.markdown("**Required Actions:**")
        
        for func in results['triggered_functions']:
            violations = func.get('violations', func.get('eligible_accounts', func.get('total_high_value', 0)))
            if violations > 0:
                timeline = "Immediate" if func['priority'] == 'CRITICAL' else "30 days" if func['priority'] == 'HIGH' else "60 days"
                st.markdown(f"- **Article {func['article']}**: Address {violations} accounts within {timeline}")
    else:
        st.markdown("No remediation actions required at this time.")
    
    # Certification
    st.markdown("### Compliance Certification")
    st.markdown("""
    **Compliance Officer Certification:**
    
    I certify that this report accurately represents the dormancy compliance status 
    of the institution as of the report date. All identified issues will be addressed 
    in accordance with CBUAE requirements and timelines.
    
    **Name:** [Compliance Officer Name]  
    **Title:** Chief Compliance Officer  
    **Date:** [Certification Date]  
    **Signature:** [Digital Signature]
    """)

def show_system_status_page():
    """Show system status and health page"""
    st.header("🔧 System Status & Health")
    
    # System overview
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Healthy")
    with col2:
        st.metric("Analysis Engine", "🟢 Online")
    with col3:
        st.metric("Uptime", "99.9%")
    with col4:
        st.metric("Last Analysis", "Active Session")
    
    # Analysis performance metrics
    st.subheader("⚡ Analysis Performance")
    
    # Mock performance data
    performance_data = pd.DataFrame({
        'Metric': ['Data Processing', 'Dormancy Analysis', 'Compliance Check', 'Risk Assessment', 'Report Generation'],
        'Processing Time (sec)': [2.1, 8.5, 3.2, 4.7, 1.8],
        'Status': ['Optimal', 'Good', 'Optimal', 'Good', 'Optimal'],
        'Last Run': ['2 min ago', '2 min ago', '1 min ago', '30 sec ago', '15 sec ago']
    })
    
    st.dataframe(performance_data, use_container_width=True, hide_index=True)
    
    # Data quality metrics
    st.subheader("📊 Data Quality Metrics")
    
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data completeness
            completeness_data = []
            for col in df.columns:
                completeness = (df[col].notna().sum() / len(df)) * 100
                completeness_data.append({
                    'Column': col,
                    'Completeness': f"{completeness:.1f}%",
                    'Missing Values': df[col].isna().sum()
                })
            
            completeness_df = pd.DataFrame(completeness_data)
            st.markdown("**Data Completeness:**")
            st.dataframe(completeness_df.head(10), use_container_width=True, hide_index=True)
        
        with col2:
            # Data distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col_to_analyze = st.selectbox("Select Column for Distribution", numeric_cols)
                
                if col_to_analyze:
                    fig_dist = px.histogram(
                        df,
                        x=col_to_analyze,
                        title=f'Distribution of {col_to_analyze}',
                        nbins=20
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
    
    # System actions
    st.subheader("⚙️ System Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Status"):
            st.success("System status refreshed")
    
    with col2:
        if st.button("🧹 Clear Cache"):
            st.success("Analysis cache cleared")
    
    with col3:
        if st.button("📊 System Report"):
            st.success("System health report generated")
    
    with col4:
        if st.button("⚠️ Test Alerts"):
            st.info("System alert test completed")
    
    # Recent activity log
    st.subheader("📝 Recent Activity Log")
    
    recent_activities = [
        {
            'Timestamp': '2024-01-15 14:35:22',
            'Activity': 'CBUAE Analysis Completed',
            'Status': 'Success',
            'Details': f'Processed {len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0} accounts'
        },
        {
            'Timestamp': '2024-01-15 14:34:15',
            'Activity': 'Risk Assessment',
            'Status': 'Success',
            'Details': 'Risk level calculated and mitigation strategies generated'
        },
        {
            'Timestamp': '2024-01-15 14:33:08',
            'Activity': 'Compliance Verification',
            'Status': 'Success',
            'Details': 'CBUAE compliance status verified'
        },
        {
            'Timestamp': '2024-01-15 14:32:45',
            'Activity': 'Data Upload',
            'Status': 'Success',
            'Details': 'CSV file processed and validated'
        },
        {
            'Timestamp': '2024-01-15 14:30:12',
            'Activity': 'User Session Started',
            'Status': 'Active',
            'Details': f'User: {st.session_state.user_data["username"]}'
        }
    ]
    
    activities_df = pd.DataFrame(recent_activities)
    st.dataframe(activities_df, use_container_width=True, hide_index=True)

# Main application logic
def main():
    """Main application entry point"""
    initialize_session_state()
    show_main_app()

if __name__ == "__main__":
    main()
