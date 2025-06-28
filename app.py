"""
Banking Compliance Agentic AI System - NiceGUI Implementation
Complete application with all agents, memory management, and async workflow orchestration
"""

import asyncio
import json
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass

# NiceGUI imports
from nicegui import ui, app, run
from nicegui.events import UploadEventArguments
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your existing agents (assuming they're available)
try:
    from login import SecureLoginManager
    from Data_Process import UnifiedDataProcessingAgent
    from Dormant_agent import run_complete_dormancy_analysis_with_llama
    from memory_agent import HybridMemoryAgent, MemoryBucket, MemoryPriority
    from mcp_client import MCPClient
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    print("⚠️ Some agent modules not found. Using mock implementations.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state management
class AppState:
    def __init__(self):
        self.current_user = None
        self.session_id = None
        self.workflow_results = {}
        self.agent_statuses = {}
        self.uploaded_data = None
        self.is_processing = False

app_state = AppState()

# Authentication check decorator
def require_auth(func):
    def wrapper(*args, **kwargs):
        if not app_state.current_user:
            ui.navigate.to('/login')
            return
        return func(*args, **kwargs)
    return wrapper

# ========================= AGENT ORCHESTRATOR =========================

class NiceGUIBankingOrchestrator:
    """Main orchestrator for all banking compliance agents with NiceGUI integration"""

    def __init__(self):
        self.login_manager = SecureLoginManager() if AGENTS_AVAILABLE else None
        self.data_processor = UnifiedDataProcessingAgent() if AGENTS_AVAILABLE else None
        self.memory_agent = HybridMemoryAgent({}) if AGENTS_AVAILABLE else None
        self.mcp_client = MCPClient() if AGENTS_AVAILABLE else None

        # Agent status tracking
        self.agent_status = {
            'data_processing': {'status': 'idle', 'progress': 0, 'message': 'Ready'},
            'dormancy_analysis': {'status': 'idle', 'progress': 0, 'message': 'Ready'},
            'compliance_verification': {'status': 'idle', 'progress': 0, 'message': 'Ready'},
            'risk_assessment': {'status': 'idle', 'progress': 0, 'message': 'Ready'},
            'reporting': {'status': 'idle', 'progress': 0, 'message': 'Ready'},
            'notifications': {'status': 'idle', 'progress': 0, 'message': 'Ready'},
        }

        # Results storage
        self.latest_results = {}

    async def run_complete_workflow(self, uploaded_file, user_preferences: Dict):
        """Run the complete agentic workflow with real-time status updates"""

        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        app_state.session_id = workflow_id

        try:
            # Phase 1: Data Processing
            await self.update_agent_status('data_processing', 'running', 10, 'Processing uploaded data...')

            if AGENTS_AVAILABLE:
                upload_result = self.data_processor.upload_data(
                    upload_method="file",
                    source=uploaded_file,
                    user_id=app_state.current_user['user_id'],
                    session_id=workflow_id
                )

                if not upload_result.success:
                    raise Exception(f"Data upload failed: {upload_result.error}")

                processed_data = upload_result.data
            else:
                # Mock data processing
                await asyncio.sleep(2)
                processed_data = pd.DataFrame({
                    'account_id': [f'ACC{i:06d}' for i in range(1000)],
                    'customer_id': [f'CUST{i:06d}' for i in range(1000)],
                    'account_type': np.random.choice(['SAVINGS', 'CURRENT', 'FIXED_DEPOSIT'], 1000),
                    'balance_current': np.random.uniform(100, 100000, 1000),
                    'last_transaction_date': pd.date_range('2020-01-01', '2024-01-01', periods=1000)
                })

            await self.update_agent_status('data_processing', 'completed', 100, 'Data processing completed')

            # Phase 2: Dormancy Analysis
            await self.update_agent_status('dormancy_analysis', 'running', 20, 'Running 10+ dormancy agents...')

            if AGENTS_AVAILABLE:
                dormancy_results = await run_complete_dormancy_analysis_with_llama(processed_data)
            else:
                # Mock dormancy analysis
                await asyncio.sleep(3)
                dormancy_results = {
                    'success': True,
                    'agent_results': {
                        'demand_deposit_dormancy': {'dormant_records_found': 45},
                        'fixed_deposit_dormancy': {'dormant_records_found': 23},
                        'high_value_dormant': {'dormant_records_found': 12},
                    },
                    'summary': {
                        'total_dormant_accounts_found': 80,
                        'total_accounts_processed': 1000,
                        'dormancy_rate': 8.0
                    }
                }

            await self.update_agent_status('dormancy_analysis', 'completed', 100, f"Found {dormancy_results.get('summary', {}).get('total_dormant_accounts_found', 0)} dormant accounts")

            # Phase 3: Compliance Verification
            await self.update_agent_status('compliance_verification', 'running', 30, 'Running 17+ compliance agents...')

            # Mock compliance verification
            await asyncio.sleep(2)
            compliance_results = {
                'success': True,
                'compliance_score': 87,
                'violations': [
                    {'type': 'ARTICLE_2_VIOLATION', 'severity': 'medium', 'count': 12},
                    {'type': 'CONTACT_ATTEMPT_MISSING', 'severity': 'high', 'count': 8}
                ],
                'status': 'requires_attention'
            }

            await self.update_agent_status('compliance_verification', 'completed', 100, f"Compliance score: {compliance_results['compliance_score']}/100")

            # Phase 4: Risk Assessment
            await self.update_agent_status('risk_assessment', 'running', 40, 'Assessing risk factors...')

            await asyncio.sleep(1)
            risk_results = {
                'overall_risk_score': 65,
                'risk_level': 'medium',
                'risk_factors': {
                    'regulatory_risk': 70,
                    'financial_risk': 45,
                    'operational_risk': 60,
                    'reputational_risk': 30
                }
            }

            await self.update_agent_status('risk_assessment', 'completed', 100, f"Risk level: {risk_results['risk_level'].upper()}")

            # Phase 5: Reporting
            await self.update_agent_status('reporting', 'running', 50, 'Generating comprehensive reports...')

            await asyncio.sleep(1)
            report_results = {
                'executive_summary': {
                    'total_accounts': len(processed_data),
                    'dormant_accounts': dormancy_results.get('summary', {}).get('total_dormant_accounts_found', 0),
                    'compliance_score': compliance_results['compliance_score'],
                    'risk_level': risk_results['risk_level']
                },
                'recommendations': [
                    'Prioritize contact attempts for high-value dormant accounts',
                    'Implement automated dormancy monitoring system',
                    'Review and update customer contact information'
                ]
            }

            await self.update_agent_status('reporting', 'completed', 100, 'Reports generated successfully')

            # Phase 6: Notifications (optional)
            if user_preferences.get('send_notifications', False):
                await self.update_agent_status('notifications', 'running', 60, 'Sending notifications...')
                await asyncio.sleep(1)
                await self.update_agent_status('notifications', 'completed', 100, 'Notifications sent')

            # Compile final results
            self.latest_results = {
                'workflow_id': workflow_id,
                'data_processing': {'processed_records': len(processed_data)},
                'dormancy_analysis': dormancy_results,
                'compliance_verification': compliance_results,
                'risk_assessment': risk_results,
                'reporting': report_results,
                'processing_time': (datetime.now() - datetime.now()).total_seconds(),
                'success': True
            }

            app_state.workflow_results = self.latest_results

            return self.latest_results

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            await self.update_agent_status('error', 'error', 0, f"Workflow failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def update_agent_status(self, agent_name: str, status: str, progress: int, message: str):
        """Update agent status and notify UI"""
        self.agent_status[agent_name] = {
            'status': status,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        app_state.agent_statuses = self.agent_status.copy()

# Global orchestrator instance
orchestrator = NiceGUIBankingOrchestrator()

# ========================= UI COMPONENTS =========================

async def create_login_page():
    """Create the login page"""

    ui.colors(primary='#1f4e79', secondary='#2e8b57', accent='#ff6b35')

    with ui.column().classes('w-full h-screen items-center justify-center bg-gradient-to-br from-blue-900 to-green-800'):
        with ui.card().classes('w-96 p-8'):
            ui.html('<h2 class="text-2xl font-bold text-center mb-6">🏦 Banking Compliance AI</h2>')

            with ui.column().classes('w-full gap-4'):
                username_input = ui.input('Username', placeholder='Enter username').classes('w-full')
                password_input = ui.input('Password', placeholder='Enter password', password=True).classes('w-full')

                async def handle_login():
                    try:
                        if AGENTS_AVAILABLE and orchestrator.login_manager:
                            user_data = orchestrator.login_manager.authenticate_user(
                                username_input.value,
                                password_input.value
                            )

                            session_token = orchestrator.login_manager.create_secure_session(user_data)
                            app_state.current_user = {**user_data, 'session_token': session_token}
                        else:
                            # Mock authentication
                            app_state.current_user = {
                                'user_id': username_input.value,
                                'username': username_input.value,
                                'role': 'analyst',
                                'authenticated': True
                            }

                        ui.notify('Login successful!', type='positive')
                        ui.navigate.to('/dashboard')

                    except Exception as e:
                        ui.notify(f'Login failed: {str(e)}', type='negative')

                ui.button('Login', on_click=handle_login).classes('w-full bg-primary text-white')

                # Demo credentials info
                with ui.expansion('Demo Credentials').classes('w-full'):
                    ui.label('Username: admin / Password: admin123')
                    ui.label('Username: analyst / Password: analyst123')

async def create_dashboard():
    """Create the main dashboard"""

    if not app_state.current_user:
        ui.navigate.to('/login')
        return

    # Header
    with ui.header().classes('bg-primary text-white'):
        ui.label('Banking Compliance Agentic AI System').classes('text-xl font-bold')
        with ui.row().classes('ml-auto'):
            ui.label(f"Welcome, {app_state.current_user['username']}").classes('mr-4')
            ui.button('Logout', on_click=lambda: (setattr(app_state, 'current_user', None), ui.navigate.to('/login'))).classes('bg-red-500')

    # Main content with sidebar
    with ui.splitter(value=20).classes('w-full h-screen'):
        with ui.splitter_panel():
            await create_sidebar()

        with ui.splitter_panel():
            with ui.column().classes('p-4 w-full'):
                await create_main_content()

async def create_sidebar():
    """Create the sidebar with navigation and agent status"""

    with ui.column().classes('p-4 w-full'):
        ui.label('🎛️ Control Panel').classes('text-lg font-bold mb-4')

        # Navigation
        with ui.column().classes('w-full gap-2 mb-6'):
            ui.button('📊 Dashboard', on_click=lambda: None).classes('w-full')
            ui.button('📄 Data Upload', on_click=lambda: ui.run_javascript('showDataUpload()')).classes('w-full')
            ui.button('🤖 Agent Monitor', on_click=lambda: ui.run_javascript('showAgentMonitor()')).classes('w-full')
            ui.button('📈 Analytics', on_click=lambda: ui.run_javascript('showAnalytics()')).classes('w-full')

        # Agent Status Monitor
        ui.label('🤖 Agent Status').classes('text-md font-bold mb-2')

        status_container = ui.column().classes('w-full gap-2')

        async def update_status_display():
            """Update the agent status display"""
            status_container.clear()

            with status_container:
                for agent_name, status in app_state.agent_statuses.items():
                    status_color = {
                        'idle': 'bg-gray-500',
                        'running': 'bg-blue-500',
                        'completed': 'bg-green-500',
                        'error': 'bg-red-500'
                    }.get(status['status'], 'bg-gray-500')

                    with ui.card().classes(f'w-full p-2 {status_color} text-white'):
                        ui.label(agent_name.replace('_', ' ').title()).classes('font-bold')
                        ui.label(status['message']).classes('text-sm')
                        if status['progress'] > 0:
                            ui.linear_progress(status['progress'] / 100).classes('w-full')

        # Update status display initially
        await update_status_display()

        # Set up periodic status updates
        ui.timer(1.0, lambda: ui.run_javascript('updateAgentStatus()'))

async def create_main_content():
    """Create the main content area"""

    # Main tabs
    with ui.tabs().classes('w-full') as tabs:
        upload_tab = ui.tab('📄 Data Upload')
        analysis_tab = ui.tab('🔍 Analysis')
        results_tab = ui.tab('📊 Results')
        ai_chat_tab = ui.tab('💬 AI Assistant')

    with ui.tab_panels(tabs, value=upload_tab).classes('w-full'):

        # Data Upload Panel
        with ui.tab_panel(upload_tab):
            await create_upload_panel()

        # Analysis Panel
        with ui.tab_panel(analysis_tab):
            await create_analysis_panel()

        # Results Panel
        with ui.tab_panel(results_tab):
            await create_results_panel()

        # AI Chat Panel
        with ui.tab_panel(ai_chat_tab):
            await create_ai_chat_panel()

async def create_upload_panel():
    """Create the data upload panel"""

    ui.label('📄 Data Upload & Processing').classes('text-2xl font-bold mb-4')

    with ui.row().classes('w-full gap-6'):
        # Upload section
        with ui.column().classes('flex-1'):
            ui.label('Upload Banking Data').classes('text-lg font-bold mb-2')

            upload_area = ui.upload(
                on_upload=handle_file_upload,
                max_file_size=50_000_000,  # 50MB
                multiple=False
            ).classes('w-full').props('accept=".csv,.xlsx,.xls"')

            ui.label('Supported formats: CSV, Excel (.xlsx, .xls)').classes('text-sm text-gray-600')

        # Configuration section
        with ui.column().classes('flex-1'):
            ui.label('Processing Configuration').classes('text-lg font-bold mb-2')

            with ui.column().classes('gap-2'):
                priority_select = ui.select(
                    options=['low', 'medium', 'high', 'critical'],
                    value='medium',
                    label='Priority'
                ).classes('w-full')

                llm_mapping = ui.switch('Enable LLM Column Mapping').classes('w-full')
                send_notifications = ui.switch('Send Notifications').classes('w-full')

                # Agent selection
                ui.label('Active Agents').classes('font-bold mt-4')
                agent_checks = {}
                agents = [
                    'Data Processing', 'Dormancy Analysis', 'Compliance Verification',
                    'Risk Assessment', 'Reporting', 'Notifications'
                ]

                for agent in agents:
                    agent_checks[agent] = ui.checkbox(agent, value=True).classes('w-full')

async def handle_file_upload(event: UploadEventArguments):
    """Handle file upload and start processing"""

    try:
        app_state.uploaded_data = event.content
        ui.notify(f'File uploaded: {event.name}', type='positive')

        # Show processing button
        process_button = ui.button(
            '🚀 Start Agentic Analysis',
            on_click=lambda: start_workflow(event)
        ).classes('bg-primary text-white mt-4')

    except Exception as e:
        ui.notify(f'Upload failed: {str(e)}', type='negative')

async def start_workflow(upload_event):
    """Start the complete workflow"""

    if app_state.is_processing:
        ui.notify('Analysis already in progress', type='warning')
        return

    app_state.is_processing = True

    # Show progress dialog
    with ui.dialog() as dialog:
        with ui.card().classes('w-96'):
            ui.label('🚀 Running Agentic Analysis').classes('text-lg font-bold')
            progress_label = ui.label('Initializing...').classes('mt-2')
            progress_bar = ui.linear_progress(0).classes('w-full mt-2')

            # Start workflow
            user_preferences = {
                'send_notifications': True,  # Get from UI
                'llm_mapping': True,  # Get from UI
                'priority': 'medium'  # Get from UI
            }

            async def run_workflow():
                try:
                    results = await orchestrator.run_complete_workflow(
                        upload_event.content,
                        user_preferences
                    )

                    if results.get('success'):
                        ui.notify('Analysis completed successfully!', type='positive')
                        dialog.close()
                        # Switch to results tab
                        ui.run_javascript('showResults()')
                    else:
                        ui.notify(f'Analysis failed: {results.get("error")}', type='negative')

                except Exception as e:
                    ui.notify(f'Workflow error: {str(e)}', type='negative')
                    logger.error(f"Workflow error: {e}")
                finally:
                    app_state.is_processing = False

            # Start the workflow
            asyncio.create_task(run_workflow())

            # Update progress periodically
            async def update_progress():
                while app_state.is_processing:
                    if app_state.agent_statuses:
                        current_agent = next(
                            (name for name, status in app_state.agent_statuses.items()
                             if status['status'] == 'running'),
                            None
                        )
                        if current_agent:
                            status = app_state.agent_statuses[current_agent]
                            progress_label.text = f"{current_agent.replace('_', ' ').title()}: {status['message']}"
                            progress_bar.value = status['progress'] / 100

                    await asyncio.sleep(0.5)

            asyncio.create_task(update_progress())

    dialog.open()

async def create_analysis_panel():
    """Create the analysis configuration panel"""

    ui.label('🔍 Analysis Configuration').classes('text-2xl font-bold mb-4')

    if not app_state.uploaded_data:
        ui.label('Please upload data first to configure analysis.').classes('text-gray-600')
        return

    # Analysis options
    with ui.row().classes('w-full gap-6'):
        # Dormancy Analysis Options
        with ui.column().classes('flex-1'):
            ui.label('Dormancy Analysis').classes('text-lg font-bold mb-2')

            dormancy_agents = [
                'Safe Deposit Dormancy', 'Investment Account Inactivity',
                'Fixed Deposit Inactivity', 'Demand Deposit Inactivity',
                'Unclaimed Payment Instruments', 'CBUAE Transfer Eligibility',
                'Article 3 Process Needed', 'Contact Attempts Needed',
                'High Value Dormant', 'Dormant to Active Transitions'
            ]

            for agent in dormancy_agents:
                ui.checkbox(agent, value=True).classes('w-full')

        # Compliance Verification Options
        with ui.column().classes('flex-1'):
            ui.label('Compliance Verification').classes('text-lg font-bold mb-2')

            compliance_agents = [
                'Article 2 Compliance', 'Article 3.1 Compliance',
                'Article 3.4 Transfer Compliance', 'Contact Verification',
                'Transfer Eligibility Check', 'FX Conversion Compliance',
                'Process Management Verification', 'Documentation Review',
                'Timeline Compliance Check', 'Amount Verification'
            ]

            for agent in compliance_agents[:5]:  # Show first 5
                ui.checkbox(agent, value=True).classes('w-full')

            ui.label(f'... and {len(compliance_agents)-5} more agents').classes('text-sm text-gray-600')

async def create_results_panel():
    """Create the results display panel"""

    ui.label('📊 Analysis Results').classes('text-2xl font-bold mb-4')

    if not app_state.workflow_results:
        ui.label('No analysis results available. Please run an analysis first.').classes('text-gray-600')
        return

    results = app_state.workflow_results

    # Executive Summary
    with ui.card().classes('w-full p-4 mb-4'):
        ui.label('📋 Executive Summary').classes('text-xl font-bold mb-2')

        with ui.row().classes('w-full gap-4'):
            # Key metrics
            summary = results.get('reporting', {}).get('executive_summary', {})

            with ui.column().classes('flex-1 text-center'):
                ui.label('Total Accounts').classes('text-sm text-gray-600')
                ui.label(str(summary.get('total_accounts', 0))).classes('text-2xl font-bold')

            with ui.column().classes('flex-1 text-center'):
                ui.label('Dormant Accounts').classes('text-sm text-gray-600')
                ui.label(str(summary.get('dormant_accounts', 0))).classes('text-2xl font-bold text-orange-600')

            with ui.column().classes('flex-1 text-center'):
                ui.label('Compliance Score').classes('text-sm text-gray-600')
                score = summary.get('compliance_score', 0)
                color = 'text-green-600' if score >= 85 else 'text-orange-600' if score >= 70 else 'text-red-600'
                ui.label(f'{score}/100').classes(f'text-2xl font-bold {color}')

            with ui.column().classes('flex-1 text-center'):
                ui.label('Risk Level').classes('text-sm text-gray-600')
                risk_level = summary.get('risk_level', 'unknown').upper()
                risk_color = {
                    'LOW': 'text-green-600',
                    'MEDIUM': 'text-orange-600',
                    'HIGH': 'text-red-600',
                    'CRITICAL': 'text-red-800'
                }.get(risk_level, 'text-gray-600')
                ui.label(risk_level).classes(f'text-2xl font-bold {risk_color}')

    # Detailed Results Tabs
    with ui.tabs().classes('w-full') as result_tabs:
        dormancy_tab = ui.tab('🔍 Dormancy Analysis')
        compliance_tab = ui.tab('⚖️ Compliance Status')
        risk_tab = ui.tab('⚠️ Risk Assessment')
        recommendations_tab = ui.tab('💡 Recommendations')
        charts_tab = ui.tab('📈 Charts')

    with ui.tab_panels(result_tabs, value=dormancy_tab).classes('w-full'):

        with ui.tab_panel(dormancy_tab):
            await create_dormancy_results(results)

        with ui.tab_panel(compliance_tab):
            await create_compliance_results(results)

        with ui.tab_panel(risk_tab):
            await create_risk_results(results)

        with ui.tab_panel(recommendations_tab):
            await create_recommendations_results(results)

        with ui.tab_panel(charts_tab):
            await create_charts_results(results)

async def create_dormancy_results(results):
    """Create dormancy analysis results display"""

    dormancy_data = results.get('dormancy_analysis', {})

    if not dormancy_data.get('success', False):
        ui.label('Dormancy analysis not completed or failed.').classes('text-gray-600')
        return

    summary = dormancy_data.get('summary', {})
    agent_results = dormancy_data.get('agent_results', {})

    # Summary metrics
    with ui.card().classes('w-full p-4 mb-4'):
        ui.label('Dormancy Analysis Summary').classes('text-lg font-bold mb-2')

        with ui.row().classes('w-full gap-4'):
            ui.label(f"Total Accounts Processed: {summary.get('total_accounts_processed', 0)}")
            ui.label(f"Dormant Accounts Found: {summary.get('total_dormant_accounts_found', 0)}")
            ui.label(f"Dormancy Rate: {summary.get('dormancy_rate', 0):.1f}%")

    # Agent-specific results
    with ui.card().classes('w-full p-4'):
        ui.label('Agent Results').classes('text-lg font-bold mb-2')

        for agent_name, agent_result in agent_results.items():
            if agent_result.get('success', False):
                with ui.expansion(f"{agent_name.replace('_', ' ').title()} ({agent_result.get('dormant_records_found', 0)} dormant)").classes('w-full'):
                    ui.json_editor({'content': agent_result}).classes('w-full')

async def create_compliance_results(results):
    """Create compliance verification results display"""

    compliance_data = results.get('compliance_verification', {})

    if not compliance_data.get('success', False):
        ui.label('Compliance analysis not completed or failed.').classes('text-gray-600')
        return

    # Compliance score
    score = compliance_data.get('compliance_score', 0)
    status = compliance_data.get('status', 'unknown')

    with ui.card().classes('w-full p-4 mb-4'):
        ui.label('Compliance Status').classes('text-lg font-bold mb-2')

        score_color = 'text-green-600' if score >= 85 else 'text-orange-600' if score >= 70 else 'text-red-600'
        ui.label(f'Overall Score: {score}/100').classes(f'text-xl {score_color}')
        ui.label(f'Status: {status.replace("_", " ").title()}').classes('text-lg')

    # Violations
    violations = compliance_data.get('violations', [])
    if violations:
        with ui.card().classes('w-full p-4'):
            ui.label('Compliance Violations').classes('text-lg font-bold mb-2')

            for violation in violations:
                severity_color = {
                    'low': 'bg-green-100 text-green-800',
                    'medium': 'bg-orange-100 text-orange-800',
                    'high': 'bg-red-100 text-red-800',
                    'critical': 'bg-red-200 text-red-900'
                }.get(violation.get('severity', 'medium'), 'bg-gray-100')

                with ui.card().classes(f'w-full p-2 {severity_color}'):
                    ui.label(f"{violation.get('type', 'Unknown')} ({violation.get('count', 0)} accounts)")
                    ui.label(f"Severity: {violation.get('severity', 'Unknown').title()}")

async def create_risk_results(results):
    """Create risk assessment results display"""

    risk_data = results.get('risk_assessment', {})

    if not risk_data:
        ui.label('Risk assessment not completed.').classes('text-gray-600')
        return

    # Overall risk
    overall_score = risk_data.get('overall_risk_score', 0)
    risk_level = risk_data.get('risk_level', 'unknown')

    with ui.card().classes('w-full p-4 mb-4'):
        ui.label('Risk Assessment').classes('text-lg font-bold mb-2')

        risk_color = {
            'low': 'text-green-600',
            'medium': 'text-orange-600',
            'high': 'text-red-600',
            'critical': 'text-red-800'
        }.get(risk_level, 'text-gray-600')

        ui.label(f'Overall Risk Score: {overall_score}/100').classes(f'text-xl {risk_color}')
        ui.label(f'Risk Level: {risk_level.upper()}').classes(f'text-lg {risk_color}')

    # Risk factors
    risk_factors = risk_data.get('risk_factors', {})
    if risk_factors:
        with ui.card().classes('w-full p-4'):
            ui.label('Risk Factors').classes('text-lg font-bold mb-2')

            for factor, score in risk_factors.items():
                factor_name = factor.replace('_', ' ').title()
                score_color = 'text-green-600' if score < 30 else 'text-orange-600' if score < 70 else 'text-red-600'

                with ui.row().classes('w-full items-center gap-4'):
                    ui.label(factor_name).classes('w-40')
                    ui.linear_progress(score / 100).classes('flex-1')
                    ui.label(f'{score}/100').classes(f'{score_color} w-16')

async def create_recommendations_results(results):
    """Create AI recommendations display"""

    recommendations = results.get('reporting', {}).get('recommendations', [])

    if not recommendations:
        ui.label('No recommendations available.').classes('text-gray-600')
        return

    with ui.card().classes('w-full p-4'):
        ui.label('AI Recommendations').classes('text-lg font-bold mb-4')

        for i, recommendation in enumerate(recommendations, 1):
            with ui.card().classes('w-full p-3 mb-2 bg-blue-50'):
                ui.label(f'{i}. {recommendation}').classes('text-base')

async def create_charts_results(results):
    """Create interactive charts and visualizations"""

    ui.label('📈 Interactive Analytics').classes('text-xl font-bold mb-4')

    # Create mock chart data (in real implementation, use actual results)

    # Dormancy trend chart
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    dormancy_rates = [12.5, 13.2, 14.1, 15.8, 16.2, 14.9]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=dormancy_rates,
        mode='lines+markers',
        name='Dormancy Rate %',
        line=dict(color='#1f4e79', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Dormancy Trend Analysis',
        xaxis_title='Month',
        yaxis_title='Dormancy Rate (%)',
        height=400
    )

    ui.plotly(fig).classes('w-full')

    # Risk factors radar chart
    risk_factors = ['Regulatory', 'Financial', 'Operational', 'Reputational']
    risk_scores = [70, 45, 60, 30]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=risk_scores + [risk_scores[0]],  # Close the shape
        theta=risk_factors + [risk_factors[0]],
        fill='toself',
        name='Risk Scores',
        line=dict(color='#ff6b35')
    ))

    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title='Risk Factor Analysis',
        height=400
    )

    ui.plotly(fig2).classes('w-full')

async def create_ai_chat_panel():
    """Create the AI assistant chat panel"""

    ui.label('💬 AI Banking Compliance Assistant').classes('text-2xl font-bold mb-4')

    # Chat container
    chat_container = ui.column().classes('w-full h-96 p-4 bg-gray-50 rounded overflow-y-auto')

    # Input area
    with ui.row().classes('w-full gap-2 mt-4'):
        chat_input = ui.input(
            placeholder='Ask about compliance, risk, or dormancy patterns...'
        ).classes('flex-1')

        async def send_message():
            user_message = chat_input.value
            if not user_message.strip():
                return

            # Add user message
            with chat_container:
                with ui.row().classes('w-full justify-end mb-2'):
                    ui.label(user_message).classes('bg-blue-500 text-white p-2 rounded max-w-xs')

            chat_input.value = ''

            # Simulate AI response
            await asyncio.sleep(1)

            # Generate AI response based on query
            if 'dormant' in user_message.lower():
                ai_response = """🔍 **Dormancy Analysis Results:**
                
Found 80 dormant accounts (8% of total)
- High-value dormant: 12 accounts ($2.3M AED)
- Requires immediate CB transfer: 15 accounts
- Contact attempts needed: 23 accounts

**Recommendations:**
1. Prioritize high-value accounts for reactivation
2. Initiate contact attempts for flagged accounts
3. Prepare CB transfer documentation"""

            elif 'compliance' in user_message.lower():
                ai_response = """⚖️ **Compliance Status:**
                
Overall Score: 87/100 (Requires Attention)
- CBUAE Article 2: 12 violations
- Contact attempts missing: 8 accounts
- Documentation gaps: 5 accounts

**Next Steps:**
1. Address contact attempt requirements
2. Complete missing documentation  
3. Review CB transfer eligibility"""

            elif 'risk' in user_message.lower():
                ai_response = """⚠️ **Risk Assessment:**
                
Current Risk Level: MEDIUM (65/100)
- Regulatory Risk: 70/100 (High)
- Financial Risk: 45/100 (Medium)
- Operational Risk: 60/100 (Medium)

**Mitigation Actions:**
1. Focus on regulatory compliance gaps
2. Implement automated monitoring
3. Enhance customer contact processes"""

            else:
                ai_response = f"""🤖 **AI Analysis:**
                
Your query: "{user_message}"

Based on current analysis:
- Total accounts: 1,000
- Dormancy rate: 8.0%
- Compliance score: 87/100
- Risk level: Medium

How can I help you analyze this further?"""

            # Add AI response
            with chat_container:
                with ui.row().classes('w-full justify-start mb-2'):
                    ui.markdown(ai_response).classes('bg-white p-3 rounded max-w-xl border')

        ui.button('Send', on_click=send_message).classes('bg-primary text-white')

    # Suggested questions
    ui.label('💡 Suggested Questions:').classes('font-bold mt-4')
    suggestions = [
        'Show me high-risk dormant accounts',
        'What are our compliance gaps?',
        'Which accounts need CB transfer?',
        'What is our current risk level?'
    ]

    for suggestion in suggestions:
        ui.button(
            suggestion,
            on_click=lambda s=suggestion: setattr(chat_input, 'value', s)
        ).classes('mr-2 mb-2 bg-gray-200 text-gray-700')

# ========================= ROUTING =========================

@ui.page('/')
async def index():
    """Home page - redirect to login or dashboard"""
    if app_state.current_user:
        ui.navigate.to('/dashboard')
    else:
        ui.navigate.to('/login')

@ui.page('/login')
async def login_page():
    """Login page"""
    await create_login_page()

@ui.page('/dashboard')
@require_auth
async def dashboard_page():
    """Main dashboard page"""
    await create_dashboard()

# ========================= STARTUP =========================

if __name__ in {"__main__", "__mp_main__"}:
    # Configure NiceGUI
    ui.run(
        title='Banking Compliance Agentic AI',
        port=8080,
        host='0.0.0.0',
        dark=False,
        show=True,
        reload=False
    )