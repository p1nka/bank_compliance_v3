"""
Data Processor UI with Pre/Post Memory Hooks and Error Handler Integration
Enhanced Streamlit interface for the Unified Data Processing Agent
"""

import streamlit as st
import pandas as pd
import json
import time
import traceback
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the unified data processing agent
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HookType(Enum):
    PRE_UPLOAD = "pre_upload"
    POST_UPLOAD = "post_upload"
    PRE_QUALITY = "pre_quality"
    POST_QUALITY = "post_quality"
    PRE_MAPPING = "pre_mapping"
    POST_MAPPING = "post_mapping"
    PRE_MEMORY = "pre_memory"
    POST_MEMORY = "post_memory"
    ERROR_HANDLER = "error_handler"


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HookContext:
    """Context passed to hooks"""
    hook_type: str
    user_id: str
    session_id: str
    workflow_id: str
    timestamp: str
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


@dataclass
class ErrorContext:
    """Context for error handling"""
    error: Exception
    error_type: str
    severity: str
    component: str
    user_id: str
    session_id: str
    workflow_id: str
    timestamp: str
    stack_trace: str
    recovery_suggestions: List[str]


class ErrorHandlerAgent:
    """Advanced error handling agent with recovery strategies"""

    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies = self._initialize_recovery_strategies()

    def _initialize_recovery_strategies(self) -> Dict[str, List[str]]:
        """Initialize recovery strategies for different error types"""
        return {
            "upload_error": [
                "Check file format and encoding",
                "Verify file permissions and accessibility",
                "Try alternative upload method",
                "Reduce file size or split into chunks",
                "Check network connectivity for cloud sources"
            ],
            "quality_error": [
                "Verify data structure and format",
                "Check for minimum required columns",
                "Ensure numeric columns contain valid numbers",
                "Validate date formats",
                "Review data for corrupted entries"
            ],
            "mapping_error": [
                "Check BGE model availability",
                "Verify column names contain valid characters",
                "Try keyword-based mapping fallback",
                "Review schema configuration",
                "Ensure target schema is properly loaded"
            ],
            "memory_error": [
                "Check memory agent initialization",
                "Verify memory service connectivity",
                "Validate user credentials",
                "Try operation without memory storage",
                "Check memory service configuration"
            ],
            "system_error": [
                "Restart the application",
                "Check system resources (CPU, memory)",
                "Verify all dependencies are installed",
                "Check file system permissions",
                "Review system logs for additional details"
            ]
        }

    def handle_error(self, error: Exception, component: str, context: Dict[str, Any]) -> ErrorContext:
        """Handle error and provide recovery suggestions"""
        error_type = type(error).__name__
        severity = self._determine_severity(error, component)

        error_context = ErrorContext(
            error=error,
            error_type=error_type,
            severity=severity.value,
            component=component,
            user_id=context.get("user_id", "unknown"),
            session_id=context.get("session_id", "unknown"),
            workflow_id=context.get("workflow_id", "unknown"),
            timestamp=datetime.now().isoformat(),
            stack_trace=traceback.format_exc(),
            recovery_suggestions=self._get_recovery_suggestions(error, component)
        )

        self.error_history.append(error_context)

        # Log error
        logger.error(f"Error in {component}: {str(error)}")

        return error_context

    def _determine_severity(self, error: Exception, component: str) -> ErrorSeverity:
        """Determine error severity based on error type and component"""
        critical_errors = ["MemoryError", "SystemError", "OSError"]
        high_errors = ["ValueError", "TypeError", "FileNotFoundError"]
        medium_errors = ["ImportError", "AttributeError", "KeyError"]

        error_type = type(error).__name__

        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _get_recovery_suggestions(self, error: Exception, component: str) -> List[str]:
        """Get recovery suggestions based on error and component"""
        error_type = type(error).__name__.lower()
        component_key = f"{component}_error"

        # Get component-specific suggestions
        suggestions = self.recovery_strategies.get(component_key, [])

        # Add error-type specific suggestions
        if "file" in error_type or "permission" in str(error).lower():
            suggestions.extend(self.recovery_strategies.get("upload_error", []))
        elif "memory" in error_type or "resource" in str(error).lower():
            suggestions.extend(self.recovery_strategies.get("system_error", []))

        return list(set(suggestions))  # Remove duplicates

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns"""
        if not self.error_history:
            return {"total_errors": 0, "patterns": {}}

        total_errors = len(self.error_history)
        severity_counts = {}
        component_counts = {}
        error_type_counts = {}

        for error_ctx in self.error_history:
            # Count by severity
            severity_counts[error_ctx.severity] = severity_counts.get(error_ctx.severity, 0) + 1

            # Count by component
            component_counts[error_ctx.component] = component_counts.get(error_ctx.component, 0) + 1

            # Count by error type
            error_type_counts[error_ctx.error_type] = error_type_counts.get(error_ctx.error_type, 0) + 1

        return {
            "total_errors": total_errors,
            "severity_distribution": severity_counts,
            "component_distribution": component_counts,
            "error_type_distribution": error_type_counts,
            "recent_errors": [asdict(error_ctx) for error_ctx in self.error_history[-5:]]
        }


class MemoryHookManager:
    """Manager for pre and post memory hooks"""

    def __init__(self):
        self.hooks: Dict[str, List[Callable]] = {hook_type.value: [] for hook_type in HookType}
        self.hook_history: List[Dict[str, Any]] = []

    def register_hook(self, hook_type: HookType, hook_function: Callable):
        """Register a hook function"""
        self.hooks[hook_type.value].append(hook_function)
        logger.info(f"Registered hook: {hook_type.value}")

    def execute_hooks(self, hook_type: HookType, context: HookContext) -> HookContext:
        """Execute all hooks of a given type"""
        start_time = datetime.now()

        try:
            for hook_func in self.hooks[hook_type.value]:
                context = hook_func(context)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Record hook execution
            self.hook_history.append({
                "hook_type": hook_type.value,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat(),
                "success": True,
                "context_data_size": len(str(context.data)) if context.data else 0
            })

            return context

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            # Record hook failure
            self.hook_history.append({
                "hook_type": hook_type.value,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat(),
                "success": False,
                "error": str(e)
            })

            logger.error(f"Hook execution failed for {hook_type.value}: {str(e)}")
            raise

    def get_hook_statistics(self) -> Dict[str, Any]:
        """Get hook execution statistics"""
        if not self.hook_history:
            return {"total_executions": 0}

        total_executions = len(self.hook_history)
        successful_executions = len([h for h in self.hook_history if h.get("success", False)])
        average_execution_time = sum(h.get("execution_time", 0) for h in self.hook_history) / total_executions

        hook_type_counts = {}
        for history_item in self.hook_history:
            hook_type = history_item["hook_type"]
            hook_type_counts[hook_type] = hook_type_counts.get(hook_type, 0) + 1

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": (successful_executions / total_executions) * 100,
            "average_execution_time": average_execution_time,
            "hook_type_distribution": hook_type_counts,
            "recent_executions": self.hook_history[-10:]
        }


class DataProcessorUI:
    """Main UI class for data processing with hooks and error handling"""

    def __init__(self):
        self.agent: Optional[UnifiedDataProcessingAgent] = None
        self.hook_manager = MemoryHookManager()
        self.error_handler = ErrorHandlerAgent()
        self.workflow_history: List[Dict[str, Any]] = []

        # Initialize session state
        self._initialize_session_state()

        # Register default hooks
        self._register_default_hooks()

        # Initialize agent
        self._initialize_agent()

    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'user_id' not in st.session_state:
            st.session_state.user_id = f"user_{secrets.token_hex(4)}"

        if 'session_id' not in st.session_state:
            st.session_state.session_id = secrets.token_hex(8)

        if 'workflow_results' not in st.session_state:
            st.session_state.workflow_results = []

        if 'error_log' not in st.session_state:
            st.session_state.error_log = []

    def _initialize_agent(self):
        """Initialize the data processing agent with error handling"""
        try:
            config = {
                "enable_memory": True,
                "enable_bge": True,
                "bge_model": "BAAI/bge-large-en-v1.5",
                "quality_thresholds": {
                    "completeness": 0.8,
                    "accuracy": 0.9,
                    "consistency": 0.85,
                    "validity": 0.9
                }
            }

            self.agent = create_unified_data_processing_agent(config)
            logger.info("Data processing agent initialized successfully")

        except Exception as e:
            error_context = self.error_handler.handle_error(
                e, "agent_initialization",
                {"user_id": st.session_state.user_id, "session_id": st.session_state.session_id}
            )
            st.error(f"Failed to initialize agent: {str(e)}")
            st.session_state.error_log.append(asdict(error_context))

    def _register_default_hooks(self):
        """Register default memory hooks"""

        # Pre-memory hooks
        def pre_memory_validation(context: HookContext) -> HookContext:
            """Validate data before memory operations"""
            if context.data is not None:
                logger.info(f"Pre-memory validation for {context.hook_type}")
                # Add validation logic here
                context.metadata = context.metadata or {}
                context.metadata["pre_memory_validation"] = True
                context.metadata["validation_timestamp"] = datetime.now().isoformat()
            return context

        def pre_memory_preprocessing(context: HookContext) -> HookContext:
            """Preprocess data before memory storage"""
            if context.data is not None:
                logger.info(f"Pre-memory preprocessing for {context.hook_type}")
                # Add preprocessing logic here
                context.metadata = context.metadata or {}
                context.metadata["pre_memory_preprocessing"] = True
                context.metadata["preprocessing_timestamp"] = datetime.now().isoformat()
            return context

        # Post-memory hooks
        def post_memory_verification(context: HookContext) -> HookContext:
            """Verify memory operations completed successfully"""
            logger.info(f"Post-memory verification for {context.hook_type}")
            context.metadata = context.metadata or {}
            context.metadata["post_memory_verification"] = True
            context.metadata["verification_timestamp"] = datetime.now().isoformat()
            return context

        def post_memory_cleanup(context: HookContext) -> HookContext:
            """Cleanup after memory operations"""
            logger.info(f"Post-memory cleanup for {context.hook_type}")
            context.metadata = context.metadata or {}
            context.metadata["post_memory_cleanup"] = True
            context.metadata["cleanup_timestamp"] = datetime.now().isoformat()
            return context

        # Register hooks
        self.hook_manager.register_hook(HookType.PRE_MEMORY, pre_memory_validation)
        self.hook_manager.register_hook(HookType.PRE_MEMORY, pre_memory_preprocessing)
        self.hook_manager.register_hook(HookType.POST_MEMORY, post_memory_verification)
        self.hook_manager.register_hook(HookType.POST_MEMORY, post_memory_cleanup)

    def _execute_with_hooks_and_error_handling(self, operation_func: Callable,
                                               operation_name: str,
                                               context_data: Any = None) -> Any:
        """Execute operation with pre/post hooks and error handling"""
        workflow_id = secrets.token_hex(8)

        try:
            # Create hook context
            hook_context = HookContext(
                hook_type=f"pre_{operation_name}",
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                workflow_id=workflow_id,
                timestamp=datetime.now().isoformat(),
                data=context_data,
                metadata={}
            )

            # Execute pre-memory hooks
            hook_context = self.hook_manager.execute_hooks(HookType.PRE_MEMORY, hook_context)

            # Execute main operation
            result = operation_func()

            # Update context with result
            hook_context.data = result
            hook_context.hook_type = f"post_{operation_name}"
            hook_context.timestamp = datetime.now().isoformat()

            # Execute post-memory hooks
            hook_context = self.hook_manager.execute_hooks(HookType.POST_MEMORY, hook_context)

            return result

        except Exception as e:
            # Handle error
            error_context = self.error_handler.handle_error(
                e, operation_name,
                {
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "workflow_id": workflow_id
                }
            )

            # Add to session state error log
            st.session_state.error_log.append(asdict(error_context))

            # Show error in UI
            st.error(f"Operation failed: {str(e)}")

            # Show recovery suggestions
            if error_context.recovery_suggestions:
                st.warning("**Recovery Suggestions:**")
                for suggestion in error_context.recovery_suggestions:
                    st.write(f"• {suggestion}")

            raise

    def render_ui(self):
        """Render the main UI"""
        st.set_page_config(
            page_title="Advanced Data Processor with Memory Hooks",
            page_icon="🔄",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🔄 Advanced Data Processor")
        st.markdown("**Enhanced with Pre/Post Memory Hooks and Error Handler**")

        # Sidebar
        self._render_sidebar()

        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📤 Data Upload",
            "📊 Quality Analysis",
            "🔗 Column Mapping",
            "📈 Results Dashboard",
            "⚙️ System Monitor"
        ])

        with tab1:
            self._render_upload_tab()

        with tab2:
            self._render_quality_tab()

        with tab3:
            self._render_mapping_tab()

        with tab4:
            self._render_results_tab()

        with tab5:
            self._render_system_monitor_tab()

    def _render_sidebar(self):
        """Render sidebar with configuration and status"""
        with st.sidebar:
            st.header("🛠️ Configuration")

            # User and Session Info
            st.info(f"**User ID:** {st.session_state.user_id[:8]}...")
            st.info(f"**Session ID:** {st.session_state.session_id[:8]}...")

            # Agent Status
            agent_status = "✅ Ready" if self.agent else "❌ Not Initialized"
            st.metric("Agent Status", agent_status)

            # Error Summary
            error_count = len(st.session_state.error_log)
            st.metric("Errors", error_count)

            # Hook Statistics
            hook_stats = self.hook_manager.get_hook_statistics()
            if hook_stats.get("total_executions", 0) > 0:
                st.metric("Hook Executions", hook_stats["total_executions"])
                st.metric("Hook Success Rate", f"{hook_stats.get('success_rate', 0):.1f}%")

            # Configuration Options
            st.subheader("⚙️ Settings")

            enable_memory_hooks = st.checkbox("Enable Memory Hooks", value=True)
            enable_error_recovery = st.checkbox("Enable Error Recovery", value=True)
            auto_retry_on_error = st.checkbox("Auto Retry on Error", value=False)

            # Advanced Options
            with st.expander("Advanced Options"):
                max_retries = st.number_input("Max Retries", min_value=0, max_value=5, value=2)
                hook_timeout = st.number_input("Hook Timeout (s)", min_value=1, max_value=60, value=30)
                error_log_retention = st.number_input("Error Log Retention (hours)", min_value=1, max_value=168,
                                                      value=24)

    def _render_upload_tab(self):
        """Render data upload tab"""
        st.header("📤 Data Upload")

        # Upload method selection
        upload_methods = ["File Upload", "Google Drive", "Azure Data Lake", "AWS S3", "HDFS"]
        selected_method = st.selectbox("Select Upload Method", upload_methods)

        if selected_method == "File Upload":
            self._render_file_upload()
        elif selected_method == "Google Drive":
            self._render_drive_upload()
        elif selected_method == "Azure Data Lake":
            self._render_azure_upload()
        elif selected_method == "AWS S3":
            self._render_s3_upload()
        elif selected_method == "HDFS":
            self._render_hdfs_upload()

    def _render_file_upload(self):
        """Render file upload interface"""
        st.subheader("📁 File Upload")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'],
            help="Supported formats: CSV, Excel, JSON, Parquet, Text"
        )

        if uploaded_file is not None:
            # File details
            file_details = {
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": uploaded_file.size
            }

            st.json(file_details)

            if st.button("🚀 Process File", type="primary"):
                self._process_uploaded_file(uploaded_file)

    def _render_drive_upload(self):
        """Render Google Drive upload interface"""
        st.subheader("💾 Google Drive Upload")

        drive_link = st.text_input(
            "Google Drive Sharing Link",
            placeholder="https://drive.google.com/file/d/...",
            help="Enter a shareable Google Drive link"
        )

        if drive_link and st.button("🚀 Process from Drive", type="primary"):
            self._process_drive_file(drive_link)

    def _render_azure_upload(self):
        """Render Azure Data Lake upload interface"""
        st.subheader("☁️ Azure Data Lake Upload")

        col1, col2 = st.columns(2)

        with col1:
            account_name = st.text_input("Account Name")
            container_name = st.text_input("Container Name")

        with col2:
            account_key = st.text_input("Account Key", type="password")
            blob_path = st.text_input("Blob Path")

        if all([account_name, account_key, container_name, blob_path]):
            if st.button("🚀 Process from Azure", type="primary"):
                azure_config = {
                    "account_name": account_name,
                    "account_key": account_key,
                    "container_name": container_name
                }
                self._process_azure_file(blob_path, azure_config)

    def _render_s3_upload(self):
        """Render AWS S3 upload interface"""
        st.subheader("🪣 AWS S3 Upload")

        col1, col2 = st.columns(2)

        with col1:
            bucket_name = st.text_input("Bucket Name")
            object_key = st.text_input("Object Key")

        with col2:
            access_key = st.text_input("Access Key ID", type="password")
            secret_key = st.text_input("Secret Access Key", type="password")
            region = st.text_input("Region", value="us-east-1")

        if all([bucket_name, object_key, access_key, secret_key]):
            if st.button("🚀 Process from S3", type="primary"):
                s3_config = {
                    "bucket_name": bucket_name,
                    "aws_access_key_id": access_key,
                    "aws_secret_access_key": secret_key,
                    "region": region
                }
                self._process_s3_file(object_key, s3_config)

    def _render_hdfs_upload(self):
        """Render HDFS upload interface"""
        st.subheader("🗂️ HDFS Upload")

        col1, col2 = st.columns(2)

        with col1:
            hdfs_host = st.text_input("HDFS Host", value="localhost")
            hdfs_path = st.text_input("HDFS Path")

        with col2:
            hdfs_port = st.number_input("HDFS Port", value=9870)
            hdfs_user = st.text_input("HDFS User", value="hdfs")

        if hdfs_path:
            if st.button("🚀 Process from HDFS", type="primary"):
                hdfs_config = {
                    "hdfs_host": hdfs_host,
                    "hdfs_port": hdfs_port,
                    "hdfs_user": hdfs_user
                }
                self._process_hdfs_file(hdfs_path, hdfs_config)

    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded file with hooks and error handling"""

        def upload_operation():
            return self.agent.upload_data(
                upload_method="file",
                source=uploaded_file,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            )

        try:
            with st.spinner("Processing uploaded file..."):
                result = self._execute_with_hooks_and_error_handling(
                    upload_operation, "upload", uploaded_file
                )

                if result.success:
                    st.success("✅ File uploaded successfully!")
                    self._display_upload_results(result)

                    # Store in session state
                    st.session_state.workflow_results.append({
                        "type": "upload",
                        "method": "file",
                        "result": asdict(result),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    st.error(f"❌ Upload failed: {result.error}")

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")

    def _process_drive_file(self, drive_link):
        """Process Google Drive file with hooks and error handling"""

        def drive_operation():
            return self.agent.upload_data(
                upload_method="drive",
                source=drive_link,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            )

        try:
            with st.spinner("Processing Google Drive file..."):
                result = self._execute_with_hooks_and_error_handling(
                    drive_operation, "drive_upload", drive_link
                )

                if result.success:
                    st.success("✅ Google Drive file processed successfully!")
                    self._display_upload_results(result)
                else:
                    st.error(f"❌ Drive upload failed: {result.error}")

        except Exception as e:
            st.error(f"❌ Drive processing failed: {str(e)}")

    def _process_azure_file(self, blob_path, config):
        """Process Azure file with hooks and error handling"""

        def azure_operation():
            return self.agent.upload_data(
                upload_method="datalake",
                source=blob_path,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                platform="azure",
                **config
            )

        try:
            with st.spinner("Processing Azure Data Lake file..."):
                result = self._execute_with_hooks_and_error_handling(
                    azure_operation, "azure_upload", {"blob_path": blob_path, "config": config}
                )

                if result.success:
                    st.success("✅ Azure file processed successfully!")
                    self._display_upload_results(result)
                else:
                    st.error(f"❌ Azure upload failed: {result.error}")

        except Exception as e:
            st.error(f"❌ Azure processing failed: {str(e)}")

    def _process_s3_file(self, object_key, config):
        """Process S3 file with hooks and error handling"""

        def s3_operation():
            return self.agent.upload_data(
                upload_method="datalake",
                source=object_key,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                platform="aws",
                **config
            )

        try:
            with st.spinner("Processing AWS S3 file..."):
                result = self._execute_with_hooks_and_error_handling(
                    s3_operation, "s3_upload", {"object_key": object_key, "config": config}
                )

                if result.success:
                    st.success("✅ S3 file processed successfully!")
                    self._display_upload_results(result)
                else:
                    st.error(f"❌ S3 upload failed: {result.error}")

        except Exception as e:
            st.error(f"❌ S3 processing failed: {str(e)}")

    def _process_hdfs_file(self, hdfs_path, config):
        """Process HDFS file with hooks and error handling"""

        def hdfs_operation():
            return self.agent.upload_data(
                upload_method="hdfs",
                source=hdfs_path,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                **config
            )

        try:
            with st.spinner("Processing HDFS file..."):
                result = self._execute_with_hooks_and_error_handling(
                    hdfs_operation, "hdfs_upload", {"hdfs_path": hdfs_path, "config": config}
                )

                if result.success:
                    st.success("✅ HDFS file processed successfully!")
                    self._display_upload_results(result)
                else:
                    st.error(f"❌ HDFS upload failed: {result.error}")

        except Exception as e:
            st.error(f"❌ HDFS processing failed: {str(e)}")

    def _display_upload_results(self, result: UploadResult):
        """Display upload results"""
        if result.metadata:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Records", result.metadata.get("total_records", "N/A"))

            with col2:
                st.metric("Columns", result.metadata.get("total_columns", "N/A"))

            with col3:
                st.metric("File Size", f"{result.metadata.get('file_size_bytes', 0):,} bytes")

            with col4:
                st.metric("Processing Time", f"{result.processing_time:.2f}s")

            # Show data preview
            if result.data is not None and not result.data.empty:
                st.subheader("📋 Data Preview")
                st.dataframe(result.data.head(10), use_container_width=True)

        # Show warnings if any
        if result.warnings:
            st.warning("⚠️ **Warnings:**")
            for warning in result.warnings:
                st.write(f"• {warning}")

    def _render_quality_tab(self):
        """Render quality analysis tab"""
        st.header("📊 Data Quality Analysis")

        # Get last successful upload
        upload_results = [r for r in st.session_state.workflow_results if r["type"] == "upload"]

        if not upload_results:
            st.info("📤 Please upload data first to perform quality analysis.")
            return

        latest_upload = upload_results[-1]

        if st.button("🔍 Analyze Data Quality", type="primary"):
            self._perform_quality_analysis(latest_upload)

    def _perform_quality_analysis(self, upload_result):
        """Perform quality analysis with hooks and error handling"""

        def quality_operation():
            # Reconstruct DataFrame from stored result
            upload_data = upload_result["result"]
            if upload_data["data"] is None:
                raise ValueError("No data available for quality analysis")

            # For demo purposes, we'll create a simple DataFrame
            # In production, you'd reconstruct the actual data
            data = pd.DataFrame(upload_data["data"]) if isinstance(upload_data["data"], list) else pd.DataFrame()

            return self.agent.analyze_data_quality(
                data=data,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            )

        try:
            with st.spinner("Analyzing data quality..."):
                result = self._execute_with_hooks_and_error_handling(
                    quality_operation, "quality_analysis", upload_result
                )

                if result.success:
                    st.success("✅ Quality analysis completed!")
                    self._display_quality_results(result)

                    # Store in session state
                    st.session_state.workflow_results.append({
                        "type": "quality",
                        "result": asdict(result),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    st.error(f"❌ Quality analysis failed: {result.error}")

        except Exception as e:
            st.error(f"❌ Quality analysis failed: {str(e)}")

    def _display_quality_results(self, result: QualityResult):
        """Display quality analysis results"""
        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Overall Score", f"{result.overall_score:.2f}")

        with col2:
            st.metric("Quality Level", result.quality_level.title())

        with col3:
            st.metric("Missing %", f"{result.missing_percentage:.1f}%")

        with col4:
            st.metric("Duplicates", result.duplicate_records)

        # Quality metrics chart
        if result.metrics:
            st.subheader("📈 Quality Metrics Breakdown")

            metrics_df = pd.DataFrame([
                {"Metric": metric.replace("_", " ").title(), "Score": score}
                for metric, score in result.metrics.items()
            ])

            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Score",
                color="Score",
                color_continuous_scale="RdYlGn",
                title="Quality Metrics"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        if result.recommendations:
            st.subheader("💡 Recommendations")
            for i, recommendation in enumerate(result.recommendations, 1):
                st.write(f"{i}. {recommendation}")

    def _render_mapping_tab(self):
        """Render column mapping tab"""
        st.header("🔗 Column Mapping")

        # Get last successful upload
        upload_results = [r for r in st.session_state.workflow_results if r["type"] == "upload"]

        if not upload_results:
            st.info("📤 Please upload data first to perform column mapping.")
            return

        latest_upload = upload_results[-1]

        # Mapping options
        col1, col2 = st.columns(2)

        with col1:
            use_bge = st.checkbox("Use BGE Semantic Mapping", value=True)
            use_llm = st.checkbox("Enhance with LLM", value=False)

        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)
            if use_llm:
                llm_api_key = st.text_input("LLM API Key", type="password")

        if st.button("🗺️ Generate Column Mapping", type="primary"):
            self._perform_column_mapping(latest_upload, use_bge, use_llm,
                                         llm_api_key if use_llm else None)

    def _perform_column_mapping(self, upload_result, use_bge, use_llm, llm_api_key):
        """Perform column mapping with hooks and error handling"""

        def mapping_operation():
            # Reconstruct DataFrame from stored result
            upload_data = upload_result["result"]
            if upload_data["data"] is None:
                raise ValueError("No data available for column mapping")

            # For demo purposes, we'll create a simple DataFrame
            data = pd.DataFrame(upload_data["data"]) if isinstance(upload_data["data"], list) else pd.DataFrame()

            return self.agent.map_columns(
                data=data,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                use_llm=use_llm,
                llm_api_key=llm_api_key
            )

        try:
            with st.spinner("Generating column mapping..."):
                result = self._execute_with_hooks_and_error_handling(
                    mapping_operation, "column_mapping", {
                        "upload_result": upload_result,
                        "use_bge": use_bge,
                        "use_llm": use_llm
                    }
                )

                if result.success:
                    st.success("✅ Column mapping completed!")
                    self._display_mapping_results(result)

                    # Store in session state
                    st.session_state.workflow_results.append({
                        "type": "mapping",
                        "result": asdict(result),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    st.error(f"❌ Column mapping failed: {result.error}")

        except Exception as e:
            st.error(f"❌ Column mapping failed: {str(e)}")

    def _display_mapping_results(self, result: MappingResult):
        """Display column mapping results"""
        # Mapping statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Auto Mapping %", f"{result.auto_mapping_percentage:.1f}%")

        with col2:
            st.metric("Method", result.method)

        with col3:
            st.metric("Total Mappings", len(result.mappings))

        with col4:
            st.metric("Processing Time", f"{result.processing_time:.2f}s")

        # Confidence distribution
        if result.confidence_distribution:
            st.subheader("📊 Confidence Distribution")

            conf_df = pd.DataFrame([
                {"Confidence": conf.title(), "Count": count}
                for conf, count in result.confidence_distribution.items()
            ])

            fig = px.pie(
                conf_df,
                values="Count",
                names="Confidence",
                title="Mapping Confidence Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Mapping sheet
        if result.mapping_sheet is not None:
            st.subheader("🗺️ Column Mapping Sheet")
            st.dataframe(result.mapping_sheet, use_container_width=True)

            # Download mapping sheet
            csv = result.mapping_sheet.to_csv(index=False)
            st.download_button(
                "📥 Download Mapping Sheet",
                csv,
                "column_mapping.csv",
                "text/csv"
            )

    def _render_results_tab(self):
        """Render results dashboard"""
        st.header("📈 Results Dashboard")

        if not st.session_state.workflow_results:
            st.info("🔄 No workflow results available. Please process some data first.")
            return

        # Results summary
        total_workflows = len(st.session_state.workflow_results)
        upload_count = len([r for r in st.session_state.workflow_results if r["type"] == "upload"])
        quality_count = len([r for r in st.session_state.workflow_results if r["type"] == "quality"])
        mapping_count = len([r for r in st.session_state.workflow_results if r["type"] == "mapping"])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Workflows", total_workflows)

        with col2:
            st.metric("Uploads", upload_count)

        with col3:
            st.metric("Quality Analyses", quality_count)

        with col4:
            st.metric("Mappings", mapping_count)

        # Workflow timeline
        st.subheader("🕒 Workflow Timeline")

        timeline_data = []
        for result in st.session_state.workflow_results:
            timeline_data.append({
                "Type": result["type"].title(),
                "Timestamp": pd.to_datetime(result["timestamp"]),
                "Success": "✅" if result["result"].get("success", False) else "❌"
            })

        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)

            fig = px.timeline(
                timeline_df,
                x_start="Timestamp",
                x_end="Timestamp",
                y="Type",
                color="Success",
                title="Workflow Execution Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recent results
        st.subheader("📋 Recent Results")

        for i, result in enumerate(reversed(st.session_state.workflow_results[-5:]), 1):
            with st.expander(f"{result['type'].title()} - {result['timestamp'][:19]}"):
                st.json(result["result"])

    def _render_system_monitor_tab(self):
        """Render system monitoring tab"""
        st.header("⚙️ System Monitor")

        # Error statistics
        st.subheader("🚨 Error Statistics")
        error_stats = self.error_handler.get_error_statistics()

        if error_stats["total_errors"] > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Errors", error_stats["total_errors"])

            with col2:
                severity_dist = error_stats.get("severity_distribution", {})
                critical_errors = severity_dist.get("critical", 0)
                st.metric("Critical Errors", critical_errors)

            with col3:
                component_dist = error_stats.get("component_distribution", {})
                most_errors = max(component_dist.items(), key=lambda x: x[1]) if component_dist else ("None", 0)
                st.metric("Most Error-Prone", f"{most_errors[0]} ({most_errors[1]})")

            # Error distribution charts
            if severity_dist:
                col1, col2 = st.columns(2)

                with col1:
                    severity_df = pd.DataFrame([
                        {"Severity": sev.title(), "Count": count}
                        for sev, count in severity_dist.items()
                    ])

                    fig = px.bar(
                        severity_df,
                        x="Severity",
                        y="Count",
                        color="Severity",
                        title="Errors by Severity"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if component_dist:
                        comp_df = pd.DataFrame([
                            {"Component": comp.replace("_", " ").title(), "Count": count}
                            for comp, count in component_dist.items()
                        ])

                        fig = px.pie(
                            comp_df,
                            values="Count",
                            names="Component",
                            title="Errors by Component"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Recent errors
            st.subheader("📝 Recent Errors")
            for error in error_stats.get("recent_errors", [])[-3:]:
                with st.expander(f"❌ {error['error_type']} - {error['timestamp'][:19]}"):
                    st.write(f"**Component:** {error['component']}")
                    st.write(f"**Severity:** {error['severity'].upper()}")
                    st.write(f"**Error:** {str(error['error'])}")

                    if error['recovery_suggestions']:
                        st.write("**Recovery Suggestions:**")
                        for suggestion in error['recovery_suggestions']:
                            st.write(f"• {suggestion}")
        else:
            st.success("✅ No errors recorded!")

        # Hook statistics
        st.subheader("🔗 Hook Statistics")
        hook_stats = self.hook_manager.get_hook_statistics()

        if hook_stats.get("total_executions", 0) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Hook Executions", hook_stats["total_executions"])

            with col2:
                st.metric("Success Rate", f"{hook_stats.get('success_rate', 0):.1f}%")

            with col3:
                st.metric("Avg Execution Time", f"{hook_stats.get('average_execution_time', 0):.3f}s")

            # Hook type distribution
            hook_type_dist = hook_stats.get("hook_type_distribution", {})
            if hook_type_dist:
                hook_df = pd.DataFrame([
                    {"Hook Type": hook_type.replace("_", " ").title(), "Count": count}
                    for hook_type, count in hook_type_dist.items()
                ])

                fig = px.bar(
                    hook_df,
                    x="Hook Type",
                    y="Count",
                    title="Hook Executions by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No hook executions recorded yet.")

        # Agent statistics
        if self.agent:
            st.subheader("🤖 Agent Statistics")
            agent_stats = self.agent.get_agent_statistics()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Uploads Processed", agent_stats["agent_stats"]["uploads_processed"])

            with col2:
                st.metric("Quality Analyses", agent_stats["agent_stats"]["quality_analyses"])

            with col3:
                st.metric("Mappings Performed", agent_stats["agent_stats"]["mappings_performed"])

            with col4:
                st.metric("Memory Operations", agent_stats["agent_stats"]["memory_operations"])

            # Configuration details
            with st.expander("🔧 Agent Configuration"):
                st.json(agent_stats["configuration"])


def main():
    """Main function to run the Data Processor UI"""
    # Initialize and render UI
    ui = DataProcessorUI()
    ui.render_ui()


if __name__ == "__main__":
    main()
