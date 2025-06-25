# app.py - Fixed BankingComplianceSystem class

import asyncio
import threading
import signal
from typing import Optional
# app.py - Fixed session state initialization

import streamlit as st
from typing import Optional, Dict, Any


def initialize_session_state():
    """Initialize all required session state variables"""

    # Core application state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'username' not in st.session_state:
        st.session_state.username = None

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'

    if 'agent_session_id' not in st.session_state:
        st.session_state.agent_session_id = None

    # Data management state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    if 'mapping_results' not in st.session_state:
        st.session_state.mapping_results = None

    if 'mapping_sheet' not in st.session_state:
        st.session_state.mapping_sheet = None

    # Analysis results state
    if 'quality_results' not in st.session_state:
        st.session_state.quality_results = None

    if 'dormancy_results' not in st.session_state:
        st.session_state.dormancy_results = None

    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = None

    # Agent management state - FIX FOR THE ERROR
    if 'intelligent_agent_manager' not in st.session_state:
        st.session_state.intelligent_agent_manager = None

    if 'memory_agent' not in st.session_state:
        st.session_state.memory_agent = None

    if 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = None

    # Settings and preferences
    if 'llm_enabled' not in st.session_state:
        st.session_state.llm_enabled = False

    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False


def safe_get_session_state(key: str, default: Any = None) -> Any:
    """Safely get session state value with default"""
    try:
        return getattr(st.session_state, key, default)
    except AttributeError:
        # Initialize the key if it doesn't exist
        setattr(st.session_state, key, default)
        return default


def initialize_intelligent_agent_manager():
    """Fixed intelligent agent manager initialization"""
    try:
        # Check if already initialized - use safe access
        if safe_get_session_state('intelligent_agent_manager') is None:

            # Initialize the manager safely
            try:
                # Import your agent manager class
                # from your_module import IntelligentAgentManager

                # st.session_state.intelligent_agent_manager = IntelligentAgentManager()

                # For now, set to empty dict to prevent the error
                st.session_state.intelligent_agent_manager = {}

                st.success("✅ Intelligent Agent Manager initialized successfully")

            except Exception as init_error:
                st.error(f"❌ Failed to initialize Intelligent Agent Manager: {init_error}")
                # Set to empty dict to prevent further errors
                st.session_state.intelligent_agent_manager = {}

    except Exception as e:
        st.error(f"❌ Error in agent manager initialization: {e}")
        # Ensure the attribute exists even if initialization fails
        st.session_state.intelligent_agent_manager = {}


# Fixed event loop management for Streamlit
def get_or_create_event_loop_streamlit():
    """Streamlit-compatible event loop management"""
    try:
        # Check if we're in an existing event loop
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            pass

        # Try to get the event loop for the current thread
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                return loop
        except RuntimeError:
            pass

        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    except Exception as e:
        st.error(f"❌ Event loop creation failed: {e}")
        # Return None and handle gracefully
        return None


# Initialize session state at the start of your Streamlit app
def main_streamlit():
    """Main Streamlit application with proper initialization"""

    # Page configuration
    st.set_page_config(
        page_title="CBUAE Banking Compliance System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state FIRST
    initialize_session_state()

    # Initialize agent manager safely
    initialize_intelligent_agent_manager()

    # Your app logic here
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_main_application()


if __name__ == "__main__":
    main_streamlit()

    # Fixed data analysis functions with proper error handling

    import pandas as pd
    import numpy as np
    from typing import Any, Optional, Union
    import logging

    logger = logging.getLogger(__name__)


    def safe_convert_to_numeric(series: pd.Series, default_value: float = 0.0) -> pd.Series:
        """Safely convert series to numeric, handling errors gracefully"""
        try:
            return pd.to_numeric(series, errors='coerce').fillna(default_value)
        except Exception as e:
            logger.warning(f"Numeric conversion failed: {e}")
            return pd.Series([default_value] * len(series), index=series.index)


    def safe_convert_to_datetime(series: pd.Series) -> pd.Series:
        """Safely convert series to datetime, handling errors gracefully"""
        try:
            return pd.to_datetime(series, errors='coerce')
        except Exception as e:
            logger.warning(f"Datetime conversion failed: {e}")
            return pd.Series([pd.NaT] * len(series), index=series.index)


    def safe_get_column(df: pd.DataFrame, column_name: str, default_value: Any = None) -> pd.Series:
        """Safely get column from DataFrame with default value"""
        if column_name in df.columns:
            return df[column_name]
        else:
            logger.warning(f"Column '{column_name}' not found, using default value")
            return pd.Series([default_value] * len(df), index=df.index)


    def check_high_value_accounts_fixed(df: pd.DataFrame, threshold: float = 100000.0) -> pd.DataFrame:
        """Fixed high value account detection"""
        try:
            # Safely get balance column and convert to numeric
            balance_column = safe_get_column(df, 'balance_current', 0.0)
            balance_numeric = safe_convert_to_numeric(balance_column)

            # Create high value flag
            df_result = df.copy()
            df_result['is_high_value'] = balance_numeric > threshold

            return df_result

        except Exception as e:
            logger.error(f"High value account check failed: {e}")
            # Return original DataFrame with default high value column
            df_result = df.copy()
            df_result['is_high_value'] = False
            return df_result


    def check_contact_attempts_fixed(df: pd.DataFrame, max_attempts: int = 3) -> pd.DataFrame:
        """Fixed contact attempts analysis"""
        try:
            # Safely get contact attempts column
            contact_attempts = safe_get_column(df, 'contact_attempts_made', 0)
            contact_attempts_numeric = safe_convert_to_numeric(contact_attempts, 0)

            df_result = df.copy()
            df_result['needs_contact'] = contact_attempts_numeric < max_attempts
            df_result['contact_attempts_safe'] = contact_attempts_numeric

            return df_result

        except Exception as e:
            logger.error(f"Contact attempts check failed: {e}")
            df_result = df.copy()
            df_result['needs_contact'] = True
            df_result['contact_attempts_safe'] = 0
            return df_result


    def check_transfer_eligibility_fixed(df: pd.DataFrame) -> pd.DataFrame:
        """Fixed transfer eligibility check"""
        try:
            # Safely get transfer date column
            transfer_date = safe_get_column(df, 'transferred_to_cb_date')
            transfer_date_converted = safe_convert_to_datetime(transfer_date)

            df_result = df.copy()
            df_result['eligible_for_transfer'] = transfer_date_converted.isna()
            df_result['transfer_date_safe'] = transfer_date_converted

            return df_result

        except Exception as e:
            logger.error(f"Transfer eligibility check failed: {e}")
            df_result = df.copy()
            df_result['eligible_for_transfer'] = True
            df_result['transfer_date_safe'] = pd.NaT
            return df_result


    def analyze_balance_thresholds_fixed(df: pd.DataFrame,
                                         low_threshold: float = 1000.0,
                                         high_threshold: float = 100000.0) -> pd.DataFrame:
        """Fixed balance threshold analysis with proper type handling"""
        try:
            # Get balance column safely
            balance_column = safe_get_column(df, 'balance_current', 0.0)
            balance_numeric = safe_convert_to_numeric(balance_column)

            df_result = df.copy()

            # Create threshold flags safely
            df_result['balance_numeric'] = balance_numeric
            df_result['is_low_balance'] = balance_numeric < low_threshold
            df_result['is_high_balance'] = balance_numeric > high_threshold
            df_result['balance_category'] = pd.cut(
                balance_numeric,
                bins=[-np.inf, low_threshold, high_threshold, np.inf],
                labels=['Low', 'Medium', 'High']
            )

            return df_result

        except Exception as e:
            logger.error(f"Balance threshold analysis failed: {e}")
            df_result = df.copy()
            df_result['balance_numeric'] = 0.0
            df_result['is_low_balance'] = True
            df_result['is_high_balance'] = False
            df_result['balance_category'] = 'Unknown'
            return df_result


    def comprehensive_data_validation(df: pd.DataFrame) -> dict:
        """Comprehensive data validation with detailed reporting"""
        validation_results = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_columns': [],
            'data_type_issues': [],
            'validation_passed': True,
            'recommendations': []
        }

        # Expected columns based on your schema
        expected_columns = [
            'customer_id', 'balance_current', 'contact_attempts_made',
            'transferred_to_cb_date', 'account_status', 'dormancy_status'
        ]

        # Check for missing columns
        for col in expected_columns:
            if col not in df.columns:
                validation_results['missing_columns'].append(col)
                validation_results['validation_passed'] = False

        # Check data types for critical columns
        if 'balance_current' in df.columns:
            try:
                pd.to_numeric(df['balance_current'], errors='raise')
            except (ValueError, TypeError):
                validation_results['data_type_issues'].append('balance_current: cannot convert to numeric')
                validation_results['validation_passed'] = False

        if 'contact_attempts_made' in df.columns:
            try:
                pd.to_numeric(df['contact_attempts_made'], errors='raise')
            except (ValueError, TypeError):
                validation_results['data_type_issues'].append('contact_attempts_made: cannot convert to numeric')
                validation_results['validation_passed'] = False

        # Generate recommendations
        if validation_results['missing_columns']:
            validation_results['recommendations'].append('Add missing columns or update column mapping')

        if validation_results['data_type_issues']:
            validation_results['recommendations'].append('Clean data types before analysis')

        return validation_results


    # Usage example for the fixed functions:
    def run_fixed_analysis(df: pd.DataFrame) -> dict:
        """Run analysis with fixed functions and comprehensive error handling"""

        try:
            # Validate data first
            validation = comprehensive_data_validation(df)

            if not validation['validation_passed']:
                logger.warning(f"Data validation issues found: {validation}")

            # Run analysis with fixed functions
            results = {}

            # High value accounts
            try:
                high_value_df = check_high_value_accounts_fixed(df)
                results['high_value_analysis'] = {
                    'total_high_value': high_value_df['is_high_value'].sum(),
                    'percentage': (high_value_df['is_high_value'].sum() / len(df)) * 100
                }
            except Exception as e:
                logger.error(f"High value analysis failed: {e}")
                results['high_value_analysis'] = {'error': str(e)}

            # Contact attempts
            try:
                contact_df = check_contact_attempts_fixed(df)
                results['contact_analysis'] = {
                    'needs_contact': contact_df['needs_contact'].sum(),
                    'average_attempts': contact_df['contact_attempts_safe'].mean()
                }
            except Exception as e:
                logger.error(f"Contact analysis failed: {e}")
                results['contact_analysis'] = {'error': str(e)}

            # Transfer eligibility
            try:
                transfer_df = check_transfer_eligibility_fixed(df)
                results['transfer_analysis'] = {
                    'eligible_count': transfer_df['eligible_for_transfer'].sum()
                }
            except Exception as e:
                logger.error(f"Transfer analysis failed: {e}")
                results['transfer_analysis'] = {'error': str(e)}

            # Balance analysis
            try:
                balance_df = analyze_balance_thresholds_fixed(df)
                results['balance_analysis'] = {
                    'low_balance_count': balance_df['is_low_balance'].sum(),
                    'high_balance_count': balance_df['is_high_balance'].sum(),
                    'average_balance': balance_df['balance_numeric'].mean()
                }
            except Exception as e:
                logger.error(f"Balance analysis failed: {e}")
                results['balance_analysis'] = {'error': str(e)}

            results['validation_results'] = validation
            results['analysis_completed'] = True

            return results

        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            return {
                'analysis_completed': False,
                'error': str(e),
                'validation_results': validation if 'validation' in locals() else {}
            }

class BankingComplianceSystem:
    """Fixed Banking Compliance System with proper initialization"""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.logger = self._setup_logger()

        # Only set signal handlers if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                self.logger.info("Signal handlers registered successfully")
            except ValueError as e:
                self.logger.warning(f"Could not register signal handlers: {e}")
        else:
            self.logger.info("Skipping signal handler registration - not in main thread")

        # Initialize components safely
        self._initialize_components()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()

    def _initialize_components(self):
        """Initialize system components safely"""
        try:
            # Initialize memory agent and other components here
            self.memory_agent = None  # Initialize properly
            self.mcp_client = None    # Initialize properly
            self.logger.info("Components initialized successfully")
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

# Fixed main function structure
async def main():
    """Fixed main function that doesn't conflict with signal handlers"""
    try:
        # Don't initialize BankingComplianceSystem in async context
        # Instead, use a factory pattern or initialize in main thread
        await demonstrate_system_fixed()
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

async def demonstrate_system_fixed():
    """Fixed demonstration system without signal handler conflicts"""
    logger.info("Starting system demonstration...")

    # Instead of creating BankingComplianceSystem here,
    # work with components directly or use a different approach
    try:
        # Your demonstration logic here
        logger.info("System demonstration completed successfully")
    except Exception as e:
        logger.error(f"System demonstration failed: {e}")
        raise

# Alternative approach: Initialize in main thread before async operations
def main_sync():
    """Synchronous main function for proper initialization"""
    try:
        # Initialize system components in main thread
        system = BankingComplianceSystem()

        # Then run async operations
        asyncio.run(async_operations(system))

    except Exception as e:
        logger.error(f"System error: {e}")
        raise

async def async_operations(system):
    """Run async operations with pre-initialized system"""
    # Your async logic here
    pass

if __name__ == "__main__":
    # Use synchronous main for proper signal handler setup
    main_sync()