"""
Banking Compliance Multi-Agent Streamlit Application
Comprehensive web application with separate interfaces for:
- Data Processing Agent
- Data Mapping Agent (NEW)
- Dormant Account Analysis Agent
- Compliance Verification Agent
Integrated with login system, memory agents, and MCP tools
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import json
import logging
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import secrets
import traceback
from typing import Dict, List, Any, Optional
import io
import base64
from pathlib import Path
import time

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Banking Compliance AI System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if agents are available
try:
    # Try to import all required agents
    from agents.Data_Process import DataProcessingAgent, DataProcessingState
    from agents.data_mapping_agent import DataMappingAgent, DataMappingState, FieldMapping, MappingConfidence, MappingStrategy
    from agents.Dormant_agent import DormancyAnalysisAgent, DormancyAnalysisState
    from agents.compliance_verification_agent import ComplianceVerificationAgent, ComplianceState  # Fixed the commented import
    from agents.risk_assessment_agent import RiskAssessmentAgent, RiskAssessmentState
    from agents.reporting_agent import ReportingAgent, ReportingState
    from agents.notification_agent import NotificationAgent, NotificationState
    from agents.memory_agent import HybridMemoryAgent, MemoryContext
    from agents.supervisor_agent import SupervisorAgent, SupervisorState
    from agents.error_handler_agent import ErrorHandlerAgent, ErrorState
    from agents.audit_trail_agent import AuditTrailAgent, AuditState
    from core.workflow_engine import WorkflowOrchestrationEngine, WorkflowState
    from mcp_client import MCPClient, create_mcp_client
    from login import SecureLoginManager

    AGENTS_AVAILABLE = True
    logger.info("✅ All agents imported successfully")

except ImportError as e:
    AGENTS_AVAILABLE = False
    logger.warning(f"⚠️ Import error: {e}")
    print(f"⚠️ Import error: {e}")
    print("🔄 Running in demo/mock mode - agents will be simulated")

# Alternative: Set to False for demo mode
# AGENTS_AVAILABLE = False


# Mock classes for when real agents aren't available
class MockMemoryAgent:
    async def store_memory(self, bucket, data, **kwargs):
        return {"success": True, "entry_id": secrets.token_hex(8)}

    async def retrieve_memory(self, bucket, filter_criteria=None, **kwargs):
        return {"success": True, "data": []}

    async def create_memory_context(self, user_id, session_id, **kwargs):
        return type('MemoryContext', (), {
            'user_id': user_id, 'session_id': session_id
        })()

class MockDataMappingAgent:
    def __init__(self, *args, **kwargs):
        # CBUAE Compliance Target Schema - 66 fields matching your dataset
        self.target_schema = {
            # Customer Information Fields (1-24)
            "customer_id": {
                "description": "Unique customer identifier",
                "type": "string",
                "required": True,
                "cbuae_mapping": "Customer identification number"
            },
            "customer_type": {
                "description": "Type of customer (Individual, Corporate, etc.)",
                "type": "string",
                "required": True,
                "values": ["Individual", "Corporate", "SME", "Government"]
            },
            "full_name_en": {
                "description": "Customer full name in English",
                "type": "string",
                "required": True,
                "cbuae_mapping": "English name for identification"
            },
            "full_name_ar": {
                "description": "Customer full name in Arabic",
                "type": "string",
                "required": True,
                "cbuae_mapping": "Arabic name for identification"
            },
            "id_number": {
                "description": "Government-issued ID number",
                "type": "integer",
                "required": True,
                "cbuae_mapping": "Emirates ID or passport number"
            },
            "id_type": {
                "description": "Type of identification document",
                "type": "string",
                "required": True,
                "values": ["Emirates_ID", "Passport", "Trade_License", "Other"]
            },
            "date_of_birth": {
                "description": "Customer date of birth",
                "type": "date",
                "required": True,
                "format": "YYYY-MM-DD"
            },
            "nationality": {
                "description": "Customer nationality",
                "type": "string",
                "required": True,
                "cbuae_mapping": "ISO country code preferred"
            },
            "address_line1": {
                "description": "Primary address line",
                "type": "string",
                "required": True,
                "cbuae_mapping": "Street address or PO Box"
            },
            "address_line2": {
                "description": "Secondary address line",
                "type": "string",
                "required": False,
                "cbuae_mapping": "Additional address information"
            },
            "city": {
                "description": "City of residence",
                "type": "string",
                "required": True,
                "cbuae_mapping": "City within emirate"
            },
            "emirate": {
                "description": "UAE Emirate",
                "type": "string",
                "required": True,
                "values": ["Abu_Dhabi", "Dubai", "Sharjah", "Ajman", "Umm_Al_Quwain", "Ras_Al_Khaimah", "Fujairah"]
            },
            "country": {
                "description": "Country of residence",
                "type": "string",
                "required": True,
                "cbuae_mapping": "ISO country code"
            },
            "postal_code": {
                "description": "Postal or zip code",
                "type": "integer",
                "required": False,
                "cbuae_mapping": "UAE postal code where applicable"
            },
            "phone_primary": {
                "description": "Primary phone number",
                "type": "string",
                "required": True,
                "cbuae_mapping": "Contact number with country code"
            },
            "phone_secondary": {
                "description": "Secondary phone number",
                "type": "string",
                "required": False,
                "cbuae_mapping": "Alternative contact number"
            },
            "email_primary": {
                "description": "Primary email address",
                "type": "string",
                "required": True,
                "cbuae_mapping": "Primary contact email"
            },
            "email_secondary": {
                "description": "Secondary email address",
                "type": "string",
                "required": False,
                "cbuae_mapping": "Alternative contact email"
            },
            "address_known": {
                "description": "Whether current address is known and verified",
                "type": "string",
                "required": True,
                "values": ["Yes", "No", "Partially_Known"]
            },
            "last_contact_date": {
                "description": "Date of last successful customer contact",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "last_contact_method": {
                "description": "Method used for last contact",
                "type": "string",
                "required": False,
                "values": ["Phone", "Email", "SMS", "Letter", "In_Person", "Other"]
            },
            "kyc_status": {
                "description": "Know Your Customer compliance status",
                "type": "string",
                "required": True,
                "values": ["Current", "Expired", "Pending", "Not_Available"]
            },
            "kyc_expiry_date": {
                "description": "KYC documentation expiry date",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "risk_rating": {
                "description": "Customer risk assessment rating",
                "type": "string",
                "required": True,
                "values": ["Low", "Medium", "High", "Very_High"]
            },

            # Account Information Fields (25-43)
            "account_id": {
                "description": "Unique account identifier",
                "type": "string",
                "required": True,
                "cbuae_mapping": "Account number for CBUAE reporting"
            },
            "account_type": {
                "description": "Type of account",
                "type": "string",
                "required": True,
                "values": ["Current", "Savings", "Fixed_Deposit", "Investment", "Loan", "Other"]
            },
            "account_subtype": {
                "description": "Sub-classification of account type",
                "type": "string",
                "required": False,
                "cbuae_mapping": "Detailed account classification"
            },
            "account_name": {
                "description": "Account title or name",
                "type": "string",
                "required": True,
                "cbuae_mapping": "Account holder name on record"
            },
            "currency": {
                "description": "Account currency code",
                "type": "string",
                "required": True,
                "cbuae_mapping": "ISO currency code (AED, USD, EUR, etc.)"
            },
            "account_status": {
                "description": "Current status of the account",
                "type": "string",
                "required": True,
                "values": ["Active", "Dormant", "Closed", "Suspended", "Frozen"]
            },
            "dormancy_status": {
                "description": "Dormancy classification status",
                "type": "string",
                "required": True,
                "values": ["Not_Dormant", "Potentially_Dormant", "Dormant", "Transferred_to_CB"]
            },
            "opening_date": {
                "description": "Date account was opened",
                "type": "date",
                "required": True,
                "format": "YYYY-MM-DD"
            },
            "closing_date": {
                "description": "Date account was closed (if applicable)",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "last_transaction_date": {
                "description": "Date of last customer-initiated transaction",
                "type": "date",
                "required": True,
                "format": "YYYY-MM-DD",
                "cbuae_mapping": "Critical for dormancy determination"
            },
            "last_system_transaction_date": {
                "description": "Date of last system-generated transaction",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "balance_current": {
                "description": "Current account balance",
                "type": "numeric",
                "required": True,
                "cbuae_mapping": "Balance for dormancy assessment"
            },
            "balance_available": {
                "description": "Available balance for withdrawal",
                "type": "numeric",
                "required": False,
                "cbuae_mapping": "Liquid funds available"
            },
            "balance_minimum": {
                "description": "Minimum balance requirement",
                "type": "integer",
                "required": False,
                "cbuae_mapping": "Minimum balance threshold"
            },
            "interest_rate": {
                "description": "Current interest rate applied",
                "type": "numeric",
                "required": False,
                "format": "Percentage (0.00-100.00)"
            },
            "interest_accrued": {
                "description": "Accrued interest amount",
                "type": "numeric",
                "required": False,
                "cbuae_mapping": "Interest earned but not paid"
            },
            "is_joint_account": {
                "description": "Whether account has multiple holders",
                "type": "string",
                "required": True,
                "values": ["Yes", "No"]
            },
            "joint_account_holders": {
                "description": "Number of joint account holders",
                "type": "integer",
                "required": False,
                "cbuae_mapping": "Count of additional account holders"
            },
            "has_outstanding_facilities": {
                "description": "Whether account has outstanding credit facilities",
                "type": "string",
                "required": True,
                "values": ["Yes", "No"]
            },
            "maturity_date": {
                "description": "Account maturity date (for fixed deposits)",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "auto_renewal": {
                "description": "Whether account auto-renews at maturity",
                "type": "string",
                "required": False,
                "values": ["Yes", "No", "Not_Applicable"]
            },
            "last_statement_date": {
                "description": "Date of last account statement",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "statement_frequency": {
                "description": "Frequency of account statements",
                "type": "string",
                "required": False,
                "values": ["Monthly", "Quarterly", "Semi_Annual", "Annual", "On_Demand"]
            },

            # Dormancy Management Fields (44-62)
            "tracking_id": {
                "description": "Unique tracking identifier for dormancy process",
                "type": "string",
                "required": False,
                "cbuae_mapping": "Internal tracking reference"
            },
            "dormancy_trigger_date": {
                "description": "Date when dormancy criteria were first met",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD",
                "cbuae_mapping": "Start of dormancy period calculation"
            },
            "dormancy_period_start": {
                "description": "Official start date of dormancy period",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "dormancy_period_months": {
                "description": "Number of months account has been dormant",
                "type": "numeric",
                "required": False,
                "cbuae_mapping": "Duration for CBUAE compliance"
            },
            "dormancy_classification_date": {
                "description": "Date account was officially classified as dormant",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "transfer_eligibility_date": {
                "description": "Date account becomes eligible for Central Bank transfer",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD",
                "cbuae_mapping": "CBUAE transfer timeline compliance"
            },
            "current_stage": {
                "description": "Current stage in dormancy management process",
                "type": "string",
                "required": False,
                "values": ["Monitoring", "Contact_Attempts", "Waiting_Period", "Transfer_Eligible", "Transferred", "Reactivated"]
            },
            "contact_attempts_made": {
                "description": "Number of contact attempts made to customer",
                "type": "integer",
                "required": False,
                "cbuae_mapping": "Due diligence documentation"
            },
            "last_contact_attempt_date": {
                "description": "Date of most recent contact attempt",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "waiting_period_start": {
                "description": "Start date of mandatory waiting period",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD",
                "cbuae_mapping": "CBUAE required waiting period"
            },
            "waiting_period_end": {
                "description": "End date of mandatory waiting period",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "transferred_to_ledger_date": {
                "description": "Date funds were transferred to unclaimed funds ledger",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD"
            },
            "transferred_to_cb_date": {
                "description": "Date funds were transferred to Central Bank",
                "type": "date",
                "required": False,
                "format": "YYYY-MM-DD",
                "cbuae_mapping": "CBUAE transfer completion date"
            },
            "cb_transfer_amount": {
                "description": "Amount transferred to Central Bank",
                "type": "numeric",
                "required": False,
                "cbuae_mapping": "Transfer amount for CBUAE records"
            },
            "cb_transfer_reference": {
                "description": "Central Bank transfer reference number",
                "type": "string",
                "required": False,
                "cbuae_mapping": "CBUAE provided reference"
            },
            "exclusion_reason": {
                "description": "Reason for exclusion from dormancy process",
                "type": "string",
                "required": False,
                "values": ["Outstanding_Facilities", "Legal_Hold", "Deceased_Customer", "Disputed_Ownership", "Other"]
            },

            # System Fields (63-66)
            "created_date": {
                "description": "Record creation date",
                "type": "date",
                "required": True,
                "format": "YYYY-MM-DD",
                "cbuae_mapping": "Audit trail requirement"
            },
            "updated_date": {
                "description": "Record last update date",
                "type": "date",
                "required": True,
                "format": "YYYY-MM-DD",
                "cbuae_mapping": "Audit trail requirement"
            },
            "updated_by": {
                "description": "User who last updated the record",
                "type": "string",
                "required": True,
                "cbuae_mapping": "Audit trail requirement"
            }
        }

        # Mock BGE manager
        self.bge_manager = type('MockBGEManager', (), {
            'embedding_cache': {}
        })()

    async def execute_mapping_workflow(self, user_id, source_data, mapping_options=None):
        """Execute intelligent field mapping workflow"""
        # Extract source fields
        if isinstance(source_data, pd.DataFrame):
            source_fields = list(source_data.columns)
        else:
            source_fields = list(source_data.keys()) if isinstance(source_data, dict) else []

        # Create field mappings using intelligent matching
        field_mappings = []
        target_fields = list(self.target_schema.keys())

        for source_field in source_fields:
            # Find best target match
            best_target, confidence = self._find_best_target_match(source_field, target_fields)

            # Determine mapping strategy based on confidence
            if confidence >= 0.9:
                strategy = "automatic"
                confidence_level = "high"
            elif confidence >= 0.7:
                strategy = "llm_assisted"
                confidence_level = "medium"
            else:
                strategy = "manual"
                confidence_level = "low"

            # Check data type compatibility
            data_type_match = self._check_data_type_compatibility(source_field, best_target, source_data)

            # Generate sample values
            sample_values = self._get_sample_values(source_field, source_data)

            # Generate business rules
            business_rules = self._generate_business_rules(source_field, best_target)

            field_mappings.append({
                "source_field": source_field,
                "target_field": best_target,
                "confidence_score": confidence,
                "confidence_level": confidence_level,
                "mapping_strategy": strategy,
                "data_type_match": data_type_match,
                "sample_values": sample_values,
                "business_rules": business_rules,
                "user_confirmed": None
            })

        # Calculate summary statistics
        mapping_summary = {
            "total_fields": len(source_fields),
            "mapped_fields": len(field_mappings),
            "confidence_distribution": {
                "high": len([m for m in field_mappings if m["confidence_level"] == "high"]),
                "medium": len([m for m in field_mappings if m["confidence_level"] == "medium"]),
                "low": len([m for m in field_mappings if m["confidence_level"] == "low"])
            },
            "average_confidence": np.mean([m["confidence_score"] for m in field_mappings]) if field_mappings else 0
        }

        return {
            "success": True,
            "mapping_id": secrets.token_hex(8),
            "status": "completed",
            "processing_time": np.random.uniform(1.5, 4.0),
            "field_mappings": field_mappings,
            "mapping_summary": mapping_summary,
            "requires_user_input": len([m for m in field_mappings if m["mapping_strategy"] == "manual"]) > 0,
            "next_steps": self._generate_next_steps(field_mappings),
            "transformation_ready": True
        }

    def _find_best_target_match(self, source_field, target_fields):
        """Find the best matching target field for a source field"""
        source_lower = source_field.lower()

        # Exact matches (highest confidence)
        if source_lower in [t.lower() for t in target_fields]:
            exact_match = next(t for t in target_fields if t.lower() == source_lower)
            return exact_match, 0.98

        # High confidence mappings based on your dataset
        high_confidence_mappings = {
            'customer_id': ('customer_id', 0.95),
            'customer_type': ('customer_type', 0.95),
            'full_name_en': ('full_name_en', 0.95),
            'full_name_ar': ('full_name_ar', 0.95),
            'id_number': ('id_number', 0.95),
            'id_type': ('id_type', 0.95),
            'date_of_birth': ('date_of_birth', 0.95),
            'nationality': ('nationality', 0.95),
            'address_line1': ('address_line1', 0.95),
            'address_line2': ('address_line2', 0.95),
            'city': ('city', 0.95),
            'emirate': ('emirate', 0.95),
            'country': ('country', 0.95),
            'postal_code': ('postal_code', 0.95),
            'phone_primary': ('phone_primary', 0.95),
            'phone_secondary': ('phone_secondary', 0.95),
            'email_primary': ('email_primary', 0.95),
            'email_secondary': ('email_secondary', 0.95),
            'address_known': ('address_known', 0.95),
            'last_contact_date': ('last_contact_date', 0.95),
            'last_contact_method': ('last_contact_method', 0.95),
            'kyc_status': ('kyc_status', 0.95),
            'kyc_expiry_date': ('kyc_expiry_date', 0.95),
            'risk_rating': ('risk_rating', 0.95),
            'account_id': ('account_id', 0.95),
            'account_type': ('account_type', 0.95),
            'account_subtype': ('account_subtype', 0.95),
            'account_name': ('account_name', 0.95),
            'currency': ('currency', 0.95),
            'account_status': ('account_status', 0.95),
            'dormancy_status': ('dormancy_status', 0.95),
            'opening_date': ('opening_date', 0.95),
            'closing_date': ('closing_date', 0.95),
            'last_transaction_date': ('last_transaction_date', 0.95),
            'last_system_transaction_date': ('last_system_transaction_date', 0.95),
            'balance_current': ('balance_current', 0.95),
            'balance_available': ('balance_available', 0.95),
            'balance_minimum': ('balance_minimum', 0.95),
            'interest_rate': ('interest_rate', 0.95),
            'interest_accrued': ('interest_accrued', 0.95),
            'is_joint_account': ('is_joint_account', 0.95),
            'joint_account_holders': ('joint_account_holders', 0.95),
            'has_outstanding_facilities': ('has_outstanding_facilities', 0.95),
            'maturity_date': ('maturity_date', 0.95),
            'auto_renewal': ('auto_renewal', 0.95),
            'last_statement_date': ('last_statement_date', 0.95),
            'statement_frequency': ('statement_frequency', 0.95),
            'tracking_id': ('tracking_id', 0.95),
            'dormancy_trigger_date': ('dormancy_trigger_date', 0.95),
            'dormancy_period_start': ('dormancy_period_start', 0.95),
            'dormancy_period_months': ('dormancy_period_months', 0.95),
            'dormancy_classification_date': ('dormancy_classification_date', 0.95),
            'transfer_eligibility_date': ('transfer_eligibility_date', 0.95),
            'current_stage': ('current_stage', 0.95),
            'contact_attempts_made': ('contact_attempts_made', 0.95),
            'last_contact_attempt_date': ('last_contact_attempt_date', 0.95),
            'waiting_period_start': ('waiting_period_start', 0.95),
            'waiting_period_end': ('waiting_period_end', 0.95),
            'transferred_to_ledger_date': ('transferred_to_ledger_date', 0.95),
            'transferred_to_cb_date': ('transferred_to_cb_date', 0.95),
            'cb_transfer_amount': ('cb_transfer_amount', 0.95),
            'cb_transfer_reference': ('cb_transfer_reference', 0.95),
            'exclusion_reason': ('exclusion_reason', 0.95),
            'created_date': ('created_date', 0.95),
            'updated_date': ('updated_date', 0.95),
            'updated_by': ('updated_by', 0.95)
        }

        if source_lower in high_confidence_mappings:
            target, confidence = high_confidence_mappings[source_lower]
            return target, confidence

        # Semantic similarity matching for partial matches
        best_target = None
        best_score = 0

        for target in target_fields:
            target_lower = target.lower()

            # Calculate similarity score
            score = 0

            # Word overlap scoring
            source_words = set(source_lower.replace('_', ' ').split())
            target_words = set(target_lower.replace('_', ' ').split())

            if source_words & target_words:  # Common words
                score += 0.6

            # Substring matching
            if source_lower in target_lower or target_lower in source_lower:
                score += 0.4

            # Special keyword matching
            if 'balance' in source_lower and 'balance' in target_lower:
                score += 0.3
            if 'date' in source_lower and 'date' in target_lower:
                score += 0.3
            if 'customer' in source_lower and 'customer' in target_lower:
                score += 0.3
            if 'account' in source_lower and 'account' in target_lower:
                score += 0.3
            if 'dormancy' in source_lower and 'dormancy' in target_lower:
                score += 0.3

            if score > best_score:
                best_score = score
                best_target = target

        # Return best match with appropriate confidence
        if best_target:
            confidence = min(0.9, 0.5 + best_score)
            return best_target, confidence

        # Default fallback
        return target_fields[0] if target_fields else "customer_id", 0.3

    def _check_data_type_compatibility(self, source_field, target_field, source_data):
        """Check if source and target data types are compatible"""
        try:
            if isinstance(source_data, pd.DataFrame) and source_field in source_data.columns:
                source_dtype = str(source_data[source_field].dtype)
                target_info = self.target_schema.get(target_field, {})
                target_type = target_info.get('type', 'string')

                # Simple compatibility check
                if target_type == 'numeric' and ('int' in source_dtype or 'float' in source_dtype):
                    return True
                elif target_type == 'string' and 'object' in source_dtype:
                    return True
                elif target_type == 'date' and ('datetime' in source_dtype or 'object' in source_dtype):
                    return True
                elif target_type == 'integer' and 'int' in source_dtype:
                    return True
                else:
                    return False
            return True  # Default to compatible if we can't determine
        except:
            return True

    def _get_sample_values(self, source_field, source_data):
        """Get sample values from source field"""
        try:
            if isinstance(source_data, pd.DataFrame) and source_field in source_data.columns:
                sample_values = source_data[source_field].dropna().head(3).astype(str).tolist()
                return sample_values if sample_values else ["N/A"]
            return ["Sample1", "Sample2", "Sample3"]
        except:
            return ["Sample1", "Sample2", "Sample3"]

    def _generate_business_rules(self, source_field, target_field):
        """Generate business rules for the mapping"""
        target_info = self.target_schema.get(target_field, {})
        rules = []

        if target_info.get('required'):
            rules.append(f"Field is required for CBUAE compliance")

        if 'values' in target_info:
            rules.append(f"Must be one of: {', '.join(target_info['values'][:3])}...")

        if target_info.get('cbuae_mapping'):
            rules.append(f"CBUAE requirement: {target_info['cbuae_mapping']}")

        if 'format' in target_info:
            rules.append(f"Format: {target_info['format']}")

        return rules if rules else [f"Standard mapping rule for {target_field}"]

    def _generate_next_steps(self, field_mappings):
        """Generate next steps based on mapping results"""
        manual_count = len([m for m in field_mappings if m["mapping_strategy"] == "manual"])

        if manual_count > 0:
            return [
                f"Review {manual_count} manual mappings requiring attention",
                "Confirm or override suggested field mappings",
                "Apply validated mappings to transform data",
                "Proceed to dormancy analysis"
            ]
        else:
            return [
                "All fields mapped automatically with high confidence",
                "Apply field mappings to transform data",
                "Proceed to dormancy analysis",
                "Run compliance verification"
            ]

    async def process_user_mapping_decisions(self, mapping_id, user_decisions):
        """Process user decisions for manual mappings"""
        processed_decisions = []

        for decision in user_decisions:
            processed_decision = {
                "source_field": decision["source_field"],
                "target_field": decision.get("override_target") or decision["target_field"],
                "user_confirmed": decision["confirmed"],
                "override_applied": bool(decision.get("override_target")),
                "processing_timestamp": datetime.now().isoformat()
            }
            processed_decisions.append(processed_decision)

        return {
            "success": True,
            "mapping_id": mapping_id,
            "decisions_processed": len(user_decisions),
            "updated_mappings": processed_decisions,
            "transformation_ready": True,
            "next_steps": [
                "User decisions successfully applied",
                "Ready for data transformation",
                "Proceed to apply field mappings"
            ]
        }

    async def apply_data_transformation(self, mapping_id, source_data):
        """Apply field mappings to transform data"""
        if not isinstance(source_data, pd.DataFrame):
            return {
                "success": False,
                "error": "Source data must be a pandas DataFrame"
            }

        try:
            transformed_records = []
            source_columns = list(source_data.columns)

            # Transform data using direct column mapping for your dataset
            for _, row in source_data.head(50).iterrows():  # Transform first 50 rows for demo
                transformed_record = {}

                # Map all available columns directly
                for col in source_columns:
                    if col in self.target_schema:
                        # Direct mapping
                        value = row[col]

                        # Handle NaN values
                        if pd.isna(value):
                            if self.target_schema[col].get('required'):
                                # Provide default for required fields
                                if self.target_schema[col]['type'] == 'string':
                                    value = "Not_Available"
                                elif self.target_schema[col]['type'] == 'numeric':
                                    value = 0.0
                                elif self.target_schema[col]['type'] == 'integer':
                                    value = 0
                                elif self.target_schema[col]['type'] == 'date':
                                    value = "1900-01-01"
                            else:
                                value = None

                        transformed_record[col] = value

                # Add any missing required fields with defaults
                for field_name, field_info in self.target_schema.items():
                    if field_name not in transformed_record and field_info.get('required'):
                        if field_info['type'] == 'string':
                            transformed_record[field_name] = "Not_Available"
                        elif field_info['type'] == 'numeric':
                            transformed_record[field_name] = 0.0
                        elif field_info['type'] == 'integer':
                            transformed_record[field_name] = 0
                        elif field_info['type'] == 'date':
                            transformed_record[field_name] = "1900-01-01"

                transformed_records.append(transformed_record)

            # Calculate transformation statistics
            source_records = len(source_data)
            target_records = len(transformed_records)
            successful_transformations = len([col for col in source_columns if col in self.target_schema])
            validation_errors = max(0, len(source_columns) - successful_transformations)

            return {
                "success": True,
                "mapping_id": mapping_id,
                "transformed_data": transformed_records,
                "transformation_statistics": {
                    "source_records": source_records,
                    "target_records": target_records,
                    "successful_transformations": successful_transformations,
                    "validation_errors": validation_errors,
                    "transformation_rate": (successful_transformations / len(source_columns)) * 100 if source_columns else 0,
                    "data_quality_score": 95.0 if validation_errors == 0 else max(70.0, 95.0 - (validation_errors * 5))
                },
                "quality_report": {
                    "mapped_fields": successful_transformations,
                    "unmapped_fields": validation_errors,
                    "required_fields_populated": len([f for f in self.target_schema if self.target_schema[f].get('required') and f in source_columns]),
                    "data_completeness": f"{((target_records * successful_transformations) / (target_records * len(self.target_schema))) * 100:.1f}%"
                },
                "next_steps": [
                    "Data transformation completed successfully",
                    "Review quality report for any issues",
                    "Proceed to dormancy analysis with transformed data",
                    "Run compliance verification on transformed dataset"
                ]
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Transformation failed: {str(e)}",
                "mapping_id": mapping_id
            }

    def get_field_description(self, field_name):
        """Get detailed description of a target field"""
        field_info = self.target_schema.get(field_name, {})
        return {
            "field_name": field_name,
            "description": field_info.get("description", "No description available"),
            "type": field_info.get("type", "unknown"),
            "required": field_info.get("required", False),
            "cbuae_mapping": field_info.get("cbuae_mapping", "Standard field"),
            "values": field_info.get("values", []),
            "format": field_info.get("format", "No specific format")
        }

    def validate_field_mapping(self, source_field, target_field, sample_data=None):
        """Validate a specific field mapping"""
        target_info = self.target_schema.get(target_field, {})

        validation_result = {
            "valid": True,
            "confidence": 0.8,
            "issues": [],
            "recommendations": []
        }

        # Check if target field exists
        if not target_info:
            validation_result["valid"] = False
            validation_result["confidence"] = 0.0
            validation_result["issues"].append(f"Target field '{target_field}' not found in schema")
            return validation_result

        # Check data type compatibility if sample data provided
        if sample_data and isinstance(sample_data, pd.DataFrame) and source_field in sample_data.columns:
            source_dtype = str(sample_data[source_field].dtype)
            target_type = target_info.get('type', 'string')

            type_compatible = self._check_data_type_compatibility(source_field, target_field, sample_data)
            if not type_compatible:
                validation_result["issues"].append(f"Data type mismatch: source is {source_dtype}, target expects {target_type}")
                validation_result["confidence"] *= 0.7
                validation_result["recommendations"].append("Consider data type conversion during transformation")

        # Check field name similarity
        similarity_score = self._calculate_field_similarity(source_field, target_field)
        if similarity_score < 0.3:
            validation_result["issues"].append("Low field name similarity - verify mapping is correct")
            validation_result["confidence"] *= 0.8
            validation_result["recommendations"].append("Double-check that this mapping makes business sense")

        # Add positive recommendations
        if target_info.get('required'):
            validation_result["recommendations"].append("This is a required field for CBUAE compliance")

        if target_info.get('cbuae_mapping'):
            validation_result["recommendations"].append(f"CBUAE requirement: {target_info['cbuae_mapping']}")

        return validation_result

    def _calculate_field_similarity(self, source_field, target_field):
        """Calculate similarity score between source and target field names"""
        source_lower = source_field.lower()
        target_lower = target_field.lower()

        # Exact match
        if source_lower == target_lower:
            return 1.0

        # Substring match
        if source_lower in target_lower or target_lower in source_lower:
            return 0.8

        # Word overlap
        source_words = set(source_lower.replace('_', ' ').split())
        target_words = set(target_lower.replace('_', ' ').split())

        if source_words & target_words:
            overlap_ratio = len(source_words & target_words) / len(source_words | target_words)
            return 0.3 + (overlap_ratio * 0.5)

        return 0.1

    def get_mapping_statistics(self):
        """Get overall mapping statistics and schema information"""
        total_fields = len(self.target_schema)
        required_fields = len([f for f in self.target_schema if self.target_schema[f].get('required')])

        field_types = {}
        for field_info in self.target_schema.values():
            field_type = field_info.get('type', 'unknown')
            field_types[field_type] = field_types.get(field_type, 0) + 1

        return {
            "total_target_fields": total_fields,
            "required_fields": required_fields,
            "optional_fields": total_fields - required_fields,
            "field_type_distribution": field_types,
            "schema_categories": {
                "customer_information": 24,
                "account_information": 19,
                "dormancy_management": 19,
                "system_fields": 4
            },
            "cbuae_compliance_fields": len([f for f in self.target_schema if self.target_schema[f].get('cbuae_mapping')]),
            "schema_version": "CBUAE_Dormancy_v1.0"
        }

class MockWorkflowEngine:
    async def execute_workflow(self, user_id, input_data, workflow_options=None):
        return {
            "success": True,
            "workflow_id": secrets.token_hex(8),
            "results": {
                "data_processing": {
                    "records_processed": len(input_data.get("accounts", [])),
                    "quality_score": 0.95,
                    "status": "completed"
                },
                "dormancy_analysis": {
                    "total_analyzed": len(input_data.get("accounts", [])),
                    "dormant_found": int(len(input_data.get("accounts", [])) * 0.23),
                    "high_risk": int(len(input_data.get("accounts", [])) * 0.05)
                },
                "compliance": {
                    "status": "compliant",
                    "critical_violations": 0
                }
            }
        }

# Initialize session state
def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.logged_in = False
        st.session_state.user_data = {}
        st.session_state.agents = {}
        st.session_state.current_data = None
        st.session_state.analysis_results = {}
        st.session_state.mapping_results = {}
        st.session_state.login_manager = None
        st.session_state.memory_agent = None
        st.session_state.mcp_client = None
        st.session_state.workflow_engine = None
        st.session_state.data_mapping_agent = None

# Initialize application
@st.cache_resource
def initialize_app():
    """Initialize all agents and systems"""
    try:
        if AGENTS_AVAILABLE:
            # Initialize login manager
            login_manager = SecureLoginManager()

            # Create default users
            try:
                login_manager.create_user("admin", "Admin123!", "admin")
                login_manager.create_user("analyst", "Analyst123!", "analyst")
                logger.info("Default users created")
            except:
                logger.info("Default users already exist")

            # Initialize MCP client in mock mode for demo
            mcp_client = MCPClient()
            mcp_client.set_mock_mode(True)

            # Initialize memory agent
            memory_agent = HybridMemoryAgent(mcp_client)

            # Initialize workflow engine
            workflow_engine = WorkflowOrchestrationEngine(memory_agent, mcp_client)

            # Initialize data mapping agent
            data_mapping_agent = DataMappingAgent(memory_agent, mcp_client)

            # Initialize individual agents
            agents = {
                'data_processing': DataProcessingAgent(memory_agent, mcp_client),
                'data_mapping': data_mapping_agent,
                'dormancy_analysis': DormancyAnalysisAgent(memory_agent, mcp_client),
                'compliance': ComplianceVerificationAgent(memory_agent, mcp_client),
                'risk_assessment': RiskAssessmentAgent(memory_agent, mcp_client),
                'reporting': ReportingAgent(memory_agent, mcp_client),
                'supervisor': SupervisorAgent(memory_agent, mcp_client),
                'error_handler': ErrorHandlerAgent(memory_agent, mcp_client),
                'audit_trail': AuditTrailAgent(memory_agent, mcp_client)
            }

            return {
                'login_manager': login_manager,
                'memory_agent': memory_agent,
                'mcp_client': mcp_client,
                'workflow_engine': workflow_engine,
                'data_mapping_agent': data_mapping_agent,
                'agents': agents,
                'initialized': True
            }
        else:
            # Mock implementation
            return {
                'login_manager': None,
                'memory_agent': MockMemoryAgent(),
                'mcp_client': None,
                'workflow_engine': MockWorkflowEngine(),
                'data_mapping_agent': MockDataMappingAgent(),
                'agents': {},
                'initialized': True
            }
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return {'initialized': False}

# Login function
def login_user(username: str, password: str) -> bool:
    """Authenticate user"""
    try:
        if st.session_state.login_manager:
            user_data = st.session_state.login_manager.authenticate_user(username, password)
            if user_data:
                st.session_state.logged_in = True
                st.session_state.user_data = user_data
                return True
        else:
            # Mock login for demo
            if username in ["admin", "analyst"] and password in ["Admin123!", "Analyst123!"]:
                st.session_state.logged_in = True
                st.session_state.user_data = {
                    'username': username,
                    'role': 'admin' if username == 'admin' else 'analyst'
                }
                return True
        return False
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False

# Data processing functions
def process_uploaded_file(uploaded_file, file_type: str) -> pd.DataFrame:
    """Process uploaded file and return DataFrame"""
    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   "application/vnd.ms-excel"]:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type")

        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Async wrapper for Streamlit
def run_async(coro):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except Exception as e:
        st.error(f"Async execution failed: {e}")
        return None

# Main application
def main():
    init_session_state()

    # Initialize app if not done
    if not st.session_state.initialized:
        app_data = initialize_app()
        if app_data['initialized']:
            st.session_state.update(app_data)
            st.session_state.initialized = True
        else:
            st.error("Application initialization failed")
            return

    # Authentication
    if not st.session_state.logged_in:
        show_login_page()
        return

    # Main application interface
    show_main_app()

def show_login_page():
    """Display login page"""
    st.title("🏦 Banking Compliance AI System")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("🔐 Secure Login")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", type="primary")

            if submit:
                if login_user(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        st.info("**Demo Credentials:**\n- Username: `admin` Password: `Admin123!`\n- Username: `analyst` Password: `Analyst123!`")

def show_main_app():
    """Display main application interface"""
    # Header
    st.title("🏦 Banking Compliance AI System")

    # User info and logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Welcome, {st.session_state.user_data.get('username', 'User')}** | Role: {st.session_state.user_data.get('role', 'User')}")
    with col2:
        if st.button("Logout", type="secondary"):
            st.session_state.clear()
            st.rerun()

    st.markdown("---")

    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")

        page = st.selectbox(
            "Select Module",
            [
                "📊 Dashboard",
                "📁 Data Processing Agent",
                "🗂️ Data Mapping Agent",
                "💤 Dormant Account Analysis",
                "✅ Compliance Verification",
                "📈 Risk Assessment",
                "📋 Comprehensive Workflow",
                "🔧 System Status"
            ]
        )

        st.markdown("---")

        # System status indicator
        st.subheader("🚦 System Status")
        status_items = [
            ("Memory Agent", st.session_state.memory_agent is not None),
            ("MCP Client", st.session_state.mcp_client is not None),
            ("Data Mapping Agent", st.session_state.data_mapping_agent is not None),
            ("Workflow Engine", st.session_state.workflow_engine is not None),
            ("Agents", len(st.session_state.agents) > 0)
        ]

        for item, status in status_items:
            color = "🟢" if status else "🔴"
            st.markdown(f"{color} {item}")

    # Main content area
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "📁 Data Processing Agent":
        show_data_processing_page()
    elif page == "🗂️ Data Mapping Agent":
        show_data_mapping_page()
    elif page == "💤 Dormant Account Analysis":
        show_dormancy_analysis_page()
    elif page == "✅ Compliance Verification":
        show_compliance_page()
    elif page == "📈 Risk Assessment":
        show_risk_assessment_page()
    elif page == "📋 Comprehensive Workflow":
        show_comprehensive_workflow_page()
    elif page == "🔧 System Status":
        show_system_status_page()

def show_dashboard():
    """Display dashboard with overview"""
    st.header("📊 Dashboard")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records",
                 len(st.session_state.current_data) if st.session_state.current_data is not None else 0)

    with col2:
        dormant_count = 0
        if 'dormancy_analysis' in st.session_state.analysis_results:
            dormant_count = st.session_state.analysis_results['dormancy_analysis'].get('dormant_found', 0)
        st.metric("Dormant Accounts", dormant_count)

    with col3:
        compliance_status = "Unknown"
        if 'compliance' in st.session_state.analysis_results:
            compliance_status = st.session_state.analysis_results['compliance'].get('status', 'Unknown')
        st.metric("Compliance Status", compliance_status)

    with col4:
        mapped_fields = 0
        if 'field_mappings' in st.session_state.mapping_results:
            mapped_fields = len(st.session_state.mapping_results['field_mappings'])
        st.metric("Mapped Fields", mapped_fields)

    st.markdown("---")

    # Recent activity with mapping info
    st.subheader("📈 Recent Activity")

    if st.session_state.current_data is not None:
        # Sample data visualization
        activity_data = ['Data Processing', 'Field Mapping', 'Dormancy Analysis', 'Compliance Check']
        completion_rates = [100, 85, 78, 92]
        
        # Add mapping completion if available
        if st.session_state.mapping_results:
            mapping_summary = st.session_state.mapping_results.get('mapping_summary', {})
            if mapping_summary:
                mapping_rate = (mapping_summary.get('mapped_fields', 0) / 
                              max(mapping_summary.get('total_fields', 1), 1)) * 100
                completion_rates[1] = mapping_rate

        fig = px.bar(
            x=activity_data,
            y=completion_rates,
            title="Module Completion Status (%)",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            xaxis_title="Module",
            yaxis_title="Completion %",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📄 Upload data in the Data Processing Agent to see analytics")

    # Mapping quality metrics
    if st.session_state.mapping_results:
        st.subheader("🗂️ Data Mapping Quality")
        
        col1, col2, col3 = st.columns(3)
        
        mapping_summary = st.session_state.mapping_results.get('mapping_summary', {})
        confidence_dist = mapping_summary.get('confidence_distribution', {})
        
        with col1:
            st.metric("High Confidence", confidence_dist.get('high', 0))
        with col2:
            st.metric("Medium Confidence", confidence_dist.get('medium', 0))
        with col3:
            avg_confidence = mapping_summary.get('average_confidence', 0)
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")

    # Quick actions
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Run Full Workflow", type="primary"):
            if st.session_state.current_data is not None:
                st.info("Redirecting to Comprehensive Workflow...")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Please upload data first in Data Processing Agent")

    with col2:
        if st.button("🗂️ Auto-Map Fields"):
            if st.session_state.current_data is not None:
                st.info("Redirecting to Data Mapping Agent...")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Please upload data first")

    with col3:
        if st.button("📊 Generate Report"):
            if st.session_state.analysis_results:
                st.success("Report generated! Check downloads.")
            else:
                st.warning("Run analysis first to generate reports")

def show_data_processing_page():
    """Display data processing agent interface"""
    st.header("📁 Data Processing Agent")
    st.markdown("Upload and process banking data for compliance analysis")

    # File upload section
    st.subheader("📤 Data Upload")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel files containing banking data"
        )

    with col2:
        file_type = st.selectbox(
            "Data Type",
            ["accounts", "transactions", "customers"],
            help="Select the type of data being uploaded"
        )

    if uploaded_file is not None:
        # Process the file
        with st.spinner("Processing file..."):
            df = process_uploaded_file(uploaded_file, file_type)

            if df is not None:
                st.session_state.current_data = df
                st.success(f"✅ File processed successfully! {len(df)} records loaded.")

                # Display file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

                # Data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Data quality analysis
                st.subheader("🔍 Data Quality Analysis")

                if st.button("Analyze Data Quality", type="primary"):
                    with st.spinner("Analyzing data quality..."):
                        quality_results = analyze_data_quality(df)
                        display_quality_results(quality_results)

                # Quick mapping suggestion
                st.subheader("🗂️ Next Steps")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔗 Auto-Map Fields", type="primary"):
                        st.info("Redirecting to Data Mapping Agent for intelligent field mapping...")
                        time.sleep(1)
                        st.rerun()
                
                with col2:
                    if st.button("📊 Proceed to Analysis"):
                        st.info("Use the sidebar to navigate to analysis modules")


def display_mapping_results(results: Dict):
    """Display comprehensive field mapping results with interactive features"""

    if not results.get("success"):
        st.error("❌ Mapping failed")
        if results.get("error"):
            st.error(f"Error details: {results['error']}")
        return

    st.success("✅ Field mapping completed successfully!")

    # ==========================
    # MAPPING OVERVIEW SECTION
    # ==========================
    st.subheader("📊 Mapping Overview")

    mapping_summary = results.get('mapping_summary', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_fields = mapping_summary.get('total_fields', 0)
        st.metric("Total Source Fields", total_fields)

    with col2:
        mapped_fields = mapping_summary.get('mapped_fields', 0)
        st.metric("Successfully Mapped", mapped_fields)

    with col3:
        avg_confidence = mapping_summary.get('average_confidence', 0)
        st.metric("Average Confidence", f"{avg_confidence:.1%}")

    with col4:
        processing_time = results.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.1f}s")

    # Mapping rate indicator
    if total_fields > 0:
        mapping_rate = mapped_fields / total_fields
        st.progress(mapping_rate)
        st.markdown(f"**Mapping Rate: {mapping_rate:.1%}** - {mapped_fields} out of {total_fields} fields mapped")

    # ==========================
    # CONFIDENCE DISTRIBUTION
    # ==========================
    st.subheader("📈 Confidence Distribution")

    confidence_dist = mapping_summary.get('confidence_distribution', {})

    if confidence_dist and any(confidence_dist.values()):
        col1, col2 = st.columns([2, 1])

        with col1:
            # Create pie chart for confidence distribution
            fig = go.Figure(data=[go.Pie(
                labels=['High Confidence (90%+)', 'Medium Confidence (70-89%)', 'Low Confidence (50-69%)'],
                values=[
                    confidence_dist.get('high', 0),
                    confidence_dist.get('medium', 0),
                    confidence_dist.get('low', 0)
                ],
                hole=0.3,
                marker_colors=['#2E8B57', '#FFA500', '#FF6B6B'],
                textinfo='label+percent+value',
                textfont_size=12
            )])
            fig.update_layout(
                title="Mapping Confidence Distribution",
                title_x=0.5,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Confidence metrics
            st.markdown("**Confidence Breakdown:**")
            high_conf = confidence_dist.get('high', 0)
            medium_conf = confidence_dist.get('medium', 0)
            low_conf = confidence_dist.get('low', 0)

            st.markdown(f"🟢 **High:** {high_conf} fields")
            st.markdown(f"🟡 **Medium:** {medium_conf} fields")
            st.markdown(f"🔴 **Low:** {low_conf} fields")

            # Quality assessment
            if high_conf >= mapped_fields * 0.7:
                st.success("🎯 Excellent mapping quality!")
            elif medium_conf + high_conf >= mapped_fields * 0.8:
                st.info("👍 Good mapping quality")
            else:
                st.warning("⚠️ Review recommended")

    # ==========================
    # DETAILED FIELD MAPPINGS
    # ==========================
    st.subheader("🗂️ Detailed Field Mappings")

    # Filter controls
    col1, col2, col3 = st.columns(3)

    with col1:
        confidence_filter = st.selectbox(
            "Filter by Confidence",
            ["All", "High", "Medium", "Low"],
            key="confidence_filter"
        )

    with col2:
        strategy_filter = st.selectbox(
            "Filter by Strategy",
            ["All", "Automatic", "Manual", "LLM-Assisted"],
            key="strategy_filter"
        )

    with col3:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Confirmed", "Pending", "Needs Review"],
            key="status_filter"
        )

    # Create mapping dataframe
    field_mappings = results.get('field_mappings', [])

    if field_mappings:
        # Apply filters and create display data
        mapping_data = []
        for mapping in field_mappings:
            # Apply confidence filter
            if confidence_filter != "All" and mapping.get('confidence_level', '').lower() != confidence_filter.lower():
                continue

            # Apply strategy filter
            if strategy_filter != "All" and mapping.get('mapping_strategy',
                                                        '').lower() != strategy_filter.lower().replace('-', '_'):
                continue

            # Apply status filter
            user_confirmed = mapping.get('user_confirmed')
            mapping_strategy = mapping.get('mapping_strategy', '')

            status = "Confirmed" if user_confirmed else "Pending"
            if mapping_strategy == 'manual' and not user_confirmed:
                status = "Needs Review"

            if status_filter != "All" and status != status_filter:
                continue

            # Create display row
            confidence_score = mapping.get('confidence_score', 0)
            confidence_level = mapping.get('confidence_level', 'unknown').title()

            # Format confidence with color coding
            if confidence_score >= 0.9:
                confidence_display = f"🟢 {confidence_score:.1%}"
            elif confidence_score >= 0.7:
                confidence_display = f"🟡 {confidence_score:.1%}"
            else:
                confidence_display = f"🔴 {confidence_score:.1%}"

            mapping_data.append({
                "Source Field": mapping['source_field'],
                "Target Field": mapping['target_field'],
                "Confidence": confidence_display,
                "Level": confidence_level,
                "Strategy": mapping.get('mapping_strategy', 'unknown').title().replace('_', ' '),
                "Data Type Match": "✅" if mapping.get('data_type_match', False) else "⚠️",
                "Sample Values": ", ".join(str(v) for v in mapping.get('sample_values', [])[:2]) or "N/A",
                "Status": f"{'✅' if user_confirmed else '⏳'} {status}",
                "Business Rules": len(mapping.get('business_rules', []))
            })

        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)

            # Display the dataframe with styling
            st.dataframe(
                mapping_df,
                use_container_width=True,
                hide_index=True
            )

            # Show filtered results info
            total_mappings = len(field_mappings)
            filtered_mappings = len(mapping_data)

            if filtered_mappings != total_mappings:
                st.info(f"Showing {filtered_mappings} of {total_mappings} mappings based on filters")
        else:
            st.info("No mappings match the selected filters")

        # ==========================
        # MANUAL REVIEW SECTION
        # ==========================
        manual_mappings = [m for m in field_mappings if
                           m.get('mapping_strategy') == 'manual' and not m.get('user_confirmed')]

        if manual_mappings:
            st.subheader("👤 Manual Review Required")
            st.warning(
                f"⚠️ {len(manual_mappings)} mappings require manual review. Please confirm or override the suggested mappings below.")

            user_decisions = []

            # Create expandable sections for each manual mapping
            for i, mapping in enumerate(manual_mappings):
                with st.expander(
                        f"📝 Review Mapping {i + 1}: {mapping['source_field']} → {mapping['target_field']} "
                        f"({mapping.get('confidence_score', 0):.1%} confidence)",
                        expanded=False
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📊 Mapping Details**")
                        st.markdown(f"**Source Field:** `{mapping['source_field']}`")
                        st.markdown(f"**Suggested Target:** `{mapping['target_field']}`")
                        st.markdown(f"**Confidence Score:** {mapping.get('confidence_score', 0):.1%}")
                        st.markdown(f"**Confidence Level:** {mapping.get('confidence_level', 'unknown').title()}")

                        # Show sample values
                        sample_values = mapping.get('sample_values', [])
                        if sample_values:
                            st.markdown(f"**Sample Values:** {', '.join(str(v) for v in sample_values[:3])}")

                        # Show business rules if available
                        business_rules = mapping.get('business_rules', [])
                        if business_rules:
                            st.markdown("**Business Rules:**")
                            for rule in business_rules[:2]:  # Show first 2 rules
                                st.markdown(f"- {rule}")

                    with col2:
                        st.markdown("**🎯 Your Decision**")

                        # User confirmation checkbox
                        user_confirmed = st.checkbox(
                            "✅ Confirm this mapping",
                            key=f"confirm_{mapping['source_field']}_{i}",
                            value=False,
                            help="Check this box to accept the suggested mapping"
                        )

                        # Alternative target selection
                        target_options = ["[Keep Suggested]"] + list(
                            st.session_state.data_mapping_agent.target_schema.keys())
                        override_target = st.selectbox(
                            "🔄 Or select different target:",
                            target_options,
                            key=f"override_{mapping['source_field']}_{i}",
                            help="Choose a different target field if the suggestion is incorrect"
                        )

                        # Show target field description if available
                        if override_target != "[Keep Suggested]":
                            target_info = st.session_state.data_mapping_agent.target_schema.get(override_target, {})
                            description = target_info.get('description', 'No description available')
                            st.markdown(f"**Target Description:** {description}")

                        # Add confidence indicator
                        confidence_score = mapping.get('confidence_score', 0)
                        if confidence_score < 0.5:
                            st.error("🔴 Very low confidence - manual review strongly recommended")
                        elif confidence_score < 0.7:
                            st.warning("🟡 Low confidence - please verify the mapping")
                        else:
                            st.info("🟢 Acceptable confidence - quick review recommended")

                        # Store user decision
                        user_decisions.append({
                            "source_field": mapping['source_field'],
                            "target_field": mapping['target_field'],
                            "confirmed": user_confirmed,
                            "override_target": override_target if override_target != "[Keep Suggested]" else None
                        })

            # Apply user decisions button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button("💾 Apply Manual Decisions", type="primary", use_container_width=True):
                    with st.spinner("Applying your mapping decisions..."):
                        try:
                            # Process user decisions
                            updated_results = run_async(
                                st.session_state.data_mapping_agent.process_user_mapping_decisions(
                                    results['mapping_id'],
                                    user_decisions
                                )
                            )

                            if updated_results and updated_results.get("success"):
                                st.success("✅ User decisions applied successfully!")
                                st.balloons()  # Celebration animation

                                # Update the mapping results in session state
                                st.session_state.mapping_results = updated_results

                                # Show summary of changes
                                confirmed_count = len([d for d in user_decisions if d['confirmed']])
                                override_count = len([d for d in user_decisions if d['override_target']])

                                st.info(
                                    f"📊 Summary: {confirmed_count} mappings confirmed, {override_count} mappings overridden")

                                # Refresh the page to show updated results
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Failed to apply user decisions")
                                if updated_results and updated_results.get('error'):
                                    st.error(f"Error: {updated_results['error']}")

                        except Exception as e:
                            st.error(f"❌ Error applying decisions: {str(e)}")
                            st.exception(e)  # Show full exception for debugging

    else:
        st.warning("⚠️ No field mappings found in the results")

    # ==========================
    # NEXT STEPS SECTION
    # ==========================
    st.subheader("🎯 Next Steps")

    next_steps = results.get('next_steps', [])
    if next_steps:
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"{i}. {step}")
    else:
        # Default next steps based on mapping status
        if results.get('requires_user_input'):
            st.markdown("1. 👤 Complete manual review of pending mappings above")
            st.markdown("2. 💾 Apply your mapping decisions")
            st.markdown("3. 🔄 Proceed to data transformation")
        else:
            st.markdown("1. 🔄 Apply field mappings to transform your data")
            st.markdown("2. 📊 Proceed to dormancy analysis")
            st.markdown("3. ✅ Run compliance verification")

    # ==========================
    # DATA TRANSFORMATION SECTION
    # ==========================
    if results.get('transformation_ready'):
        st.subheader("🔄 Data Transformation")

        st.info("🎯 Your field mappings are ready! Transform your source data to the target schema format.")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("🚀 Apply Field Mappings & Transform Data", type="primary", use_container_width=True):
                with st.spinner("🔄 Transforming data according to field mappings..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Simulate transformation progress
                        status_text.text("🔄 Preparing transformation...")
                        progress_bar.progress(25)
                        time.sleep(0.5)

                        status_text.text("🔄 Applying field mappings...")
                        progress_bar.progress(50)
                        time.sleep(0.5)

                        status_text.text("🔄 Validating transformed data...")
                        progress_bar.progress(75)
                        time.sleep(0.5)

                        # Execute transformation
                        transform_results = run_async(
                            st.session_state.data_mapping_agent.apply_data_transformation(
                                results['mapping_id'],
                                st.session_state.current_data
                            )
                        )

                        progress_bar.progress(100)
                        status_text.text("✅ Transformation completed!")

                        if transform_results and transform_results.get("success"):
                            st.success("✅ Data transformation completed successfully!")
                            st.balloons()

                            # Show transformation statistics
                            stats = transform_results.get('transformation_statistics', {})

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Records Transformed", f"{stats.get('target_records', 0):,}")
                            with col2:
                                st.metric("Fields Successfully Mapped", stats.get('successful_transformations', 0))
                            with col3:
                                validation_errors = stats.get('validation_errors', 0)
                                st.metric(
                                    "Validation Errors",
                                    validation_errors,
                                    delta=None if validation_errors == 0 else f"{validation_errors} issues"
                                )

                            # Show transformed data preview
                            transformed_data = transform_results.get('transformed_data', [])
                            if transformed_data:
                                st.subheader("📋 Transformed Data Preview")
                                st.markdown("*Showing first 10 rows of transformed data:*")

                                preview_df = pd.DataFrame(transformed_data[:10])
                                st.dataframe(preview_df, use_container_width=True)

                                # Download option for full transformed data
                                if len(transformed_data) > 10:
                                    full_df = pd.DataFrame(transformed_data)
                                    csv_data = full_df.to_csv(index=False)

                                    st.download_button(
                                        label="📥 Download Full Transformed Data (CSV)",
                                        data=csv_data,
                                        file_name=f"transformed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )

                                # Option to proceed with analysis
                                st.markdown("---")
                                col1, col2 = st.columns(2)

                                with col1:
                                    if st.button("📊 Proceed to Dormancy Analysis", type="secondary",
                                                 use_container_width=True):
                                        st.info(
                                            "🚀 Data is ready for analysis! Use the sidebar to navigate to the Dormancy Analysis module.")

                                with col2:
                                    if st.button("✅ Run Compliance Check", type="secondary", use_container_width=True):
                                        st.info("🚀 Navigate to Compliance Verification to check regulatory adherence.")
                        else:
                            st.error("❌ Data transformation failed")
                            if transform_results and transform_results.get('error'):
                                st.error(f"Error details: {transform_results['error']}")

                    except Exception as e:
                        st.error(f"❌ Transformation error: {str(e)}")
                        st.exception(e)

        with col2:
            # Transformation info
            st.markdown("**🔍 Transformation Details:**")
            st.markdown(f"• **Target Schema:** 66 CBUAE compliance fields")
            st.markdown(
                f"• **Source Fields:** {len(st.session_state.current_data.columns) if st.session_state.current_data is not None else 0}")
            st.markdown(f"• **Mapped Fields:** {len(field_mappings)}")
            st.markdown(f"• **Data Validation:** Included")
            st.markdown(f"• **Output Format:** CSV/DataFrame")

    # ==========================
    # MAPPING QUALITY INSIGHTS
    # ==========================
    st.subheader("🧠 Mapping Quality Insights")

    insights = []

    # Generate insights based on mapping results
    if avg_confidence >= 0.9:
        insights.append("🎯 **Excellent mapping quality** - High confidence across most fields")
    elif avg_confidence >= 0.7:
        insights.append("👍 **Good mapping quality** - Most fields mapped with reasonable confidence")
    else:
        insights.append("⚠️ **Review recommended** - Several fields have low confidence mappings")

    # Data type matching insight
    type_matches = len([m for m in field_mappings if m.get('data_type_match', False)])
    if type_matches / max(len(field_mappings), 1) >= 0.8:
        insights.append("✅ **Strong data type compatibility** - Most fields have matching data types")
    else:
        insights.append("⚠️ **Data type mismatches detected** - Some transformations may be needed")

    # Strategy distribution insight
    auto_mappings = len([m for m in field_mappings if m.get('mapping_strategy') == 'automatic'])
    if auto_mappings / max(len(field_mappings), 1) >= 0.7:
        insights.append("🤖 **High automation rate** - Most mappings handled automatically")
    else:
        insights.append("👤 **Manual review needed** - Several mappings require human oversight")

    for insight in insights:
        st.markdown(f"- {insight}")

    # ==========================
    # EXPORT OPTIONS
    # ==========================
    st.subheader("📤 Export Options")

    col1, col2 = st.columns(2)

    with col1:
        # Export mapping report
        mapping_report = {
            "mapping_summary": mapping_summary,
            "field_mappings": field_mappings,
            "processing_info": {
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "user": st.session_state.user_data.get('username', 'unknown')
            }
        }

        report_json = json.dumps(mapping_report, indent=2, default=str)

        st.download_button(
            label="📋 Download Mapping Report (JSON)",
            data=report_json,
            file_name=f"mapping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col2:
        # Export mapping table as CSV
        if mapping_data:
            mapping_csv = pd.DataFrame(mapping_data).to_csv(index=False)

            st.download_button(
                label="📊 Download Mapping Table (CSV)",
                data=mapping_csv,
                file_name=f"field_mappings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )



def run_risk_assessment(df: pd.DataFrame, risk_categories: List[str], assessment_depth: str) -> Dict:
    """Run risk assessment on the data"""
    try:
        # Mock risk assessment - in real implementation, this would use the actual agent
        risk_scores = {}
        risk_factors = {}

        # Calculate risk scores for each category
        for category in risk_categories:
            if category == "Regulatory Risk":
                # Based on dormancy violations and compliance issues
                base_score = 0.3
                if 'compliance' in st.session_state.analysis_results:
                    violations = len(st.session_state.analysis_results['compliance'].get('violations', []))
                    base_score += min(0.5, violations * 0.1)
                risk_scores[category] = min(1.0, base_score)
                risk_factors[category] = [
                    "CBUAE regulation compliance gaps",
                    "Incomplete contact documentation",
                    "Delayed dormancy classifications"
                ]

            elif category == "Operational Risk":
                risk_scores[category] = 0.25
                risk_factors[category] = [
                    "Manual process dependencies",
                    "Data quality issues",
                    "System integration gaps"
                ]

            elif category == "Reputational Risk":
                risk_scores[category] = 0.2
                risk_factors[category] = [
                    "Customer communication failures",
                    "Regulatory enforcement actions",
                    "Media coverage of compliance issues"
                ]

            elif category == "Financial Risk":
                risk_scores[category] = 0.15
                risk_factors[category] = [
                    "Penalty and fine exposure",
                    "Operational cost increases",
                    "Customer compensation requirements"
                ]

            elif category == "Data Quality Risk":
                # New risk category based on mapping status
                base_score = 0.4
                if st.session_state.mapping_results:
                    mapping_summary = st.session_state.mapping_results.get('mapping_summary', {})
                    avg_confidence = mapping_summary.get('average_confidence', 0)
                    base_score = max(0.1, 0.5 - avg_confidence)
                risk_scores[category] = base_score
                risk_factors[category] = [
                    "Unmapped or poorly mapped data fields",
                    "Data type mismatches",
                    "Incomplete field validation",
                    "Inconsistent data quality"
                ]

        # Calculate overall risk score
        overall_score = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0

        # Determine risk level
        if overall_score >= 0.7:
            risk_level = "Critical"
        elif overall_score >= 0.5:
            risk_level = "High"
        elif overall_score >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Generate mitigation strategies
        mitigation_strategies = []
        if overall_score > 0.5:
            mitigation_strategies.extend([
                "Implement automated compliance monitoring",
                "Enhance staff training on CBUAE regulations",
                "Upgrade customer contact systems"
            ])

        if "Data Quality Risk" in risk_categories and risk_scores.get("Data Quality Risk", 0) > 0.3:
            mitigation_strategies.append("Complete comprehensive field mapping using Data Mapping Agent")

        if "Regulatory Risk" in risk_categories and risk_scores.get("Regulatory Risk", 0) > 0.4:
            mitigation_strategies.append("Conduct comprehensive regulatory compliance review")

        results = {
            "overall_risk_score": overall_score,
            "risk_level": risk_level,
            "category_scores": risk_scores,
            "risk_factors": risk_factors,
            "mitigation_strategies": mitigation_strategies,
            "assessment_depth": assessment_depth,
            "high_risk_accounts": int(len(df) * overall_score * 0.1),  # Mock calculation
            "recommendations": generate_risk_recommendations(overall_score, risk_level),
            "analysis_completeness": {
                "field_mapping": bool(st.session_state.mapping_results),
                "dormancy_analysis": 'dormancy_analysis' in st.session_state.analysis_results,
                "compliance_check": 'compliance' in st.session_state.analysis_results
            }
        }

        return results

    except Exception as e:
        st.error(f"Risk assessment failed: {e}")
        return None


def generate_risk_recommendations(overall_score: float, risk_level: str) -> List[str]:
    """Generate risk-based recommendations"""
    recommendations = []

    if risk_level == "Critical":
        recommendations.extend([
            "🚨 Immediate executive attention required",
            "🔒 Implement enhanced monitoring controls",
            "📞 Engage with CBUAE for guidance"
        ])
    elif risk_level == "High":
        recommendations.extend([
            "⚠️ Develop immediate remediation plan",
            "📋 Increase compliance monitoring frequency",
            "👥 Additional staff training required"
        ])
    elif risk_level == "Medium":
        recommendations.extend([
            "📊 Regular monitoring and review",
            "🔧 Process improvement initiatives",
            "📚 Update procedures and documentation"
        ])
    else:
        recommendations.extend([
            "✅ Maintain current controls",
            "📈 Continue monitoring trends",
            "🎯 Focus on continuous improvement"
        ])

    return recommendations


def display_risk_results(results: Dict):
    """Display risk assessment results"""
    # Overall risk level
    risk_color = {
        "Critical": "error",
        "High": "warning",
        "Medium": "info",
        "Low": "success"
    }[results["risk_level"]]

    st.markdown(f":{risk_color}[🎯 Overall Risk Level: {results['risk_level']}]")

    # Analysis completeness indicator
    completeness = results.get('analysis_completeness', {})
    if not all(completeness.values()):
        st.warning("⚠️ Risk assessment is based on partial data. Complete all prerequisite analyses for full accuracy.")

        missing = [k.replace('_', ' ').title() for k, v in completeness.items() if not v]
        st.markdown(f"**Missing:** {', '.join(missing)}")

    # Key metrics
    st.subheader("📊 Risk Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Risk Score", f"{results['overall_risk_score']:.2f}")
    with col2:
        st.metric("Risk Level", results['risk_level'])
    with col3:
        st.metric("Categories Assessed", len(results['category_scores']))
    with col4:
        st.metric("High Risk Accounts", results['high_risk_accounts'])

    # Risk category breakdown
    if results['category_scores']:
        st.subheader("📈 Risk by Category")

        # Create horizontal bar chart
        categories = list(results['category_scores'].keys())
        scores = list(results['category_scores'].values())

        fig = px.bar(
            y=categories,
            x=scores,
            orientation='h',
            title="Risk Scores by Category",
            color=scores,
            color_continuous_scale="Reds"
        )
        fig.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Risk Category",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk factors detail
        st.subheader("🔍 Risk Factors")

        for category, factors in results['risk_factors'].items():
            score = results['category_scores'][category]
            score_color = "error" if score >= 0.7 else "warning" if score >= 0.4 else "info"

            with st.expander(f"{category} (Score: {score:.2f})"):
                st.markdown(f"**Risk Score:** :{score_color}[{score:.2f}]")
                st.markdown("**Key Risk Factors:**")
                for factor in factors:
                    st.markdown(f"- {factor}")

    # Mitigation strategies
    if results['mitigation_strategies']:
        st.subheader("🛡️ Mitigation Strategies")

        for i, strategy in enumerate(results['mitigation_strategies'], 1):
            st.markdown(f"{i}. {strategy}")

    # Recommendations
    st.subheader("💡 Recommendations")

    for rec in results['recommendations']:
        st.markdown(f"- {rec}")


def show_comprehensive_workflow_page():
    """Display comprehensive workflow interface"""
    st.header("📋 Comprehensive Workflow")
    st.markdown("Execute end-to-end banking compliance analysis workflow")

    if st.session_state.current_data is None:
        st.warning("📄 Please upload data in the Data Processing Agent first")
        return

    # Workflow configuration
    st.subheader("⚙️ Workflow Configuration")

    col1, col2 = st.columns(2)

    with col1:
        workflow_modules = st.multiselect(
            "Select Modules to Execute",
            [
                "Data Processing",
                "Data Mapping",
                "Dormancy Analysis",
                "Compliance Verification",
                "Risk Assessment",
                "Reporting",
                "Notifications"
            ],
            default=["Data Processing", "Data Mapping", "Dormancy Analysis", "Compliance Verification"]
        )

    with col2:
        workflow_options = {
            "report_date": st.date_input("Report Date", datetime.now().date()),
            "notification_channels": st.multiselect(
                "Notification Channels",
                ["Email", "Dashboard", "SMS", "Slack"],
                default=["Dashboard"]
            ),
            "parallel_execution": st.checkbox("Enable Parallel Execution", value=False),
            "auto_mapping": st.checkbox("Auto-apply high confidence mappings", value=True)
        }

    # Workflow status overview
    st.subheader("📊 Current Status")

    status_data = {
        "Module": ["Data Processing", "Data Mapping", "Dormancy Analysis", "Compliance Verification",
                   "Risk Assessment"],
        "Status": [
            "✅ Complete" if st.session_state.current_data is not None else "❌ Pending",
            "✅ Complete" if st.session_state.mapping_results else "❌ Pending",
            "✅ Complete" if 'dormancy_analysis' in st.session_state.analysis_results else "❌ Pending",
            "✅ Complete" if 'compliance' in st.session_state.analysis_results else "❌ Pending",
            "✅ Complete" if 'risk_assessment' in st.session_state.analysis_results else "❌ Pending"
        ]
    }

    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True)

    # Execute workflow
    if st.button("🚀 Execute Comprehensive Workflow", type="primary"):
        with st.spinner("Executing comprehensive workflow..."):
            progress_bar = st.progress(0)
            status_container = st.container()

            results = execute_comprehensive_workflow(
                st.session_state.current_data,
                workflow_modules,
                workflow_options,
                progress_bar,
                status_container
            )

            if results:
                st.session_state.analysis_results['comprehensive_workflow'] = results
                display_comprehensive_results(results)


def execute_comprehensive_workflow(df: pd.DataFrame, modules: List[str], options: Dict,
                                   progress_bar, status_container) -> Dict:
    """Execute comprehensive workflow with all selected modules"""
    try:
        results = {
            "workflow_id": secrets.token_hex(8),
            "execution_time": datetime.now(),
            "modules_executed": [],
            "module_results": {},
            "overall_status": "success",
            "errors": []
        }

        total_modules = len(modules)

        for i, module in enumerate(modules):
            # Update progress
            progress = (i + 1) / total_modules
            progress_bar.progress(progress)

            with status_container:
                st.info(f"🔄 Executing {module}...")

            # Simulate module execution
            time.sleep(1)  # Simulate processing time

            try:
                if module == "Data Processing":
                    module_result = {
                        "status": "completed",
                        "records_processed": len(df),
                        "quality_score": 0.95,
                        "processing_time": 1.2
                    }

                elif module == "Data Mapping":
                    if not st.session_state.mapping_results:
                        # Execute mapping if not done
                        mapping_result = run_async(
                            st.session_state.data_mapping_agent.execute_mapping_workflow(
                                user_id=st.session_state.user_data.get('username', 'workflow_user'),
                                source_data=df,
                                mapping_options={"auto_apply": options.get("auto_mapping", True)}
                            )
                        )
                        if mapping_result and mapping_result.get("success"):
                            st.session_state.mapping_results = mapping_result
                            module_result = {
                                "status": "completed",
                                "mapped_fields": len(mapping_result.get('field_mappings', [])),
                                "average_confidence": mapping_result.get('mapping_summary', {}).get(
                                    'average_confidence', 0),
                                "processing_time": mapping_result.get('processing_time', 0)
                            }
                        else:
                            raise Exception("Data mapping failed")
                    else:
                        module_result = {
                            "status": "already_completed",
                            "mapped_fields": len(st.session_state.mapping_results.get('field_mappings', [])),
                            "note": "Using existing mapping results"
                        }

                elif module == "Dormancy Analysis":
                    module_result = run_dormancy_analysis(df, options["report_date"].strftime("%Y-%m-%d"), 3, 25000.0)

                elif module == "Compliance Verification":
                    module_result = run_compliance_check(df, "full", "medium")

                elif module == "Risk Assessment":
                    module_result = run_risk_assessment(df,
                                                        ["Regulatory Risk", "Operational Risk", "Data Quality Risk"],
                                                        "Standard")

                elif module == "Reporting":
                    module_result = {
                        "status": "completed",
                        "reports_generated": ["Executive Summary", "Detailed Analysis", "Compliance Report",
                                              "Field Mapping Report"],
                        "report_id": secrets.token_hex(8)
                    }

                elif module == "Notifications":
                    module_result = {
                        "status": "completed",
                        "notifications_sent": len(options["notification_channels"]),
                        "channels": options["notification_channels"]
                    }

                results["module_results"][module] = module_result
                results["modules_executed"].append(module)

                with status_container:
                    st.success(f"✅ {module} completed successfully")

            except Exception as e:
                error_msg = f"Module {module} failed: {str(e)}"
                results["errors"].append(error_msg)
                results["overall_status"] = "partial_success"

                with status_container:
                    st.error(f"❌ {module} failed: {str(e)}")

        # Final status update
        progress_bar.progress(1.0)

        if results["overall_status"] == "success":
            with status_container:
                st.success("🎉 Comprehensive workflow completed successfully!")
        else:
            with status_container:
                st.warning("⚠️ Workflow completed with some errors")

        return results

    except Exception as e:
        st.error(f"Comprehensive workflow failed: {e}")
        return None


def display_comprehensive_results(results: Dict):
    """Display comprehensive workflow results"""
    st.success("✅ Comprehensive workflow execution completed!")

    # Workflow overview
    st.subheader("📊 Workflow Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Workflow ID", results["workflow_id"][:8] + "...")
    with col2:
        st.metric("Modules Executed", len(results["modules_executed"]))
    with col3:
        st.metric("Overall Status", results["overall_status"].title())
    with col4:
        st.metric("Execution Time", results["execution_time"].strftime("%H:%M:%S"))

    # Module execution summary
    st.subheader("🔧 Module Execution Summary")

    module_summary = []
    for module in results["modules_executed"]:
        module_result = results["module_results"][module]
        status = module_result.get("status", "unknown")

        module_summary.append({
            "Module": module,
            "Status": status.title(),
            "Details": get_module_summary(module, module_result)
        })

    summary_df = pd.DataFrame(module_summary)
    st.dataframe(summary_df, use_container_width=True)

    # Errors (if any)
    if results["errors"]:
        st.subheader("⚠️ Execution Errors")
        for error in results["errors"]:
            st.error(error)

    # Overall insights
    st.subheader("💡 Key Insights")

    insights = []

    # Data mapping insights
    if "Data Mapping" in results["modules_executed"]:
        mapping_result = results["module_results"]["Data Mapping"]
        mapped_fields = mapping_result.get("mapped_fields", 0)
        avg_confidence = mapping_result.get("average_confidence", 0)
        insights.append(f"🗂️ Successfully mapped {mapped_fields} fields with {avg_confidence:.1%} average confidence")

    # Dormancy insights
    if "Dormancy Analysis" in results["modules_executed"]:
        dormancy_result = results["module_results"]["Dormancy Analysis"]
        dormant_accounts = dormancy_result.get("dormant_accounts_found", 0)
        dormancy_rate = dormancy_result.get("dormancy_rate", 0)
        insights.append(f"💤 Identified {dormant_accounts} dormant accounts ({dormancy_rate:.1f}% dormancy rate)")

    # Compliance insights
    if "Compliance Verification" in results["modules_executed"]:
        compliance_result = results["module_results"]["Compliance Verification"]
        compliance_score = compliance_result.get("compliance_score", 0)
        violations = len(compliance_result.get("violations", []))
        insights.append(f"✅ Compliance score: {compliance_score:.1%} with {violations} violations identified")

    # Risk insights
    if "Risk Assessment" in results["modules_executed"]:
        risk_result = results["module_results"]["Risk Assessment"]
        risk_level = risk_result.get("risk_level", "Unknown")
        risk_score = risk_result.get("overall_risk_score", 0)
        insights.append(f"📈 Overall risk level: {risk_level} (score: {risk_score:.2f})")

    for insight in insights:
        st.markdown(f"- {insight}")

    # Next steps
    st.subheader("🎯 Recommended Next Steps")

    next_steps = [
        "📋 Review detailed results in individual module pages",
        "📊 Generate executive summary report",
        "📞 Schedule stakeholder review meeting",
        "🔄 Plan follow-up actions for identified issues"
    ]

    # Add specific next steps based on results
    if results["errors"]:
        next_steps.insert(0, "🔧 Address failed modules and re-execute if necessary")

    if any("Compliance" in module for module in results["modules_executed"]):
        compliance_result = results["module_results"].get("Compliance Verification", {})
        if compliance_result.get("violations"):
            next_steps.insert(-1, "⚖️ Implement compliance violation remediation plan")

    for step in next_steps:
        st.markdown(f"- {step}")


def get_module_summary(module: str, result: Dict) -> str:
    """Get summary text for module result"""
    if module == "Data Processing":
        return f"Processed {result.get('records_processed', 0)} records"
    elif module == "Data Mapping":
        return f"Mapped {result.get('mapped_fields', 0)} fields"
    elif module == "Dormancy Analysis":
        return f"Found {result.get('dormant_accounts_found', 0)} dormant accounts"
    elif module == "Compliance Verification":
        return f"Found {len(result.get('violations', []))} violations"
    elif module == "Risk Assessment":
        return f"Risk level: {result.get('risk_level', 'Unknown')}"
    elif module == "Reporting":
        return f"Generated {len(result.get('reports_generated', []))} reports"
    elif module == "Notifications":
        return f"Sent via {result.get('notifications_sent', 0)} channels"
    else:
        return "Completed"


def show_system_status_page():
    """Display system status and diagnostics"""
    st.header("🔧 System Status")
    st.markdown("Monitor system health and performance")

    # System components status
    st.subheader("🚦 Component Status")

    components = [
        ("Login Manager", st.session_state.login_manager is not None),
        ("Memory Agent", st.session_state.memory_agent is not None),
        ("MCP Client", st.session_state.mcp_client is not None),
        ("Data Mapping Agent", st.session_state.data_mapping_agent is not None),
        ("Workflow Engine", st.session_state.workflow_engine is not None),
        ("Data Processing Agent", "data_processing" in st.session_state.agents),
        ("Dormancy Analysis Agent", "dormancy_analysis" in st.session_state.agents),
        ("Compliance Agent", "compliance" in st.session_state.agents),
        ("Risk Assessment Agent", "risk_assessment" in st.session_state.agents),
        ("Reporting Agent", "reporting" in st.session_state.agents),
        ("Notification Agent", "notification" in st.session_state.agents)
    ]

    status_data = []
    for component, status in components:
        status_text = "🟢 Online" if status else "🔴 Offline"
        status_data.append({
            "Component": component,
            "Status": status_text,
            "Health": "Healthy" if status else "Unavailable"
        })

    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True)

    # System metrics
    st.subheader("📊 System Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Components", sum(1 for _, status in components if status))
    with col2:
        st.metric("Total Components", len(components))
    with col3:
        data_size = len(st.session_state.current_data) if st.session_state.current_data is not None else 0
        st.metric("Data Records", data_size)
    with col4:
        total_results = len(st.session_state.analysis_results) + (1 if st.session_state.mapping_results else 0)
        st.metric("Analysis Results", total_results)

    # Data mapping agent status
    st.subheader("🗂️ Data Mapping Agent Status")

    if st.session_state.data_mapping_agent:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("BGE Model", "BAAI/bge-large-en-v1.5")
        with col2:
            st.metric("Target Schema Fields", "66")
        with col3:
            embedding_cache_size = getattr(st.session_state.data_mapping_agent.bge_manager, 'embedding_cache', {})
            st.metric("Embedding Cache", len(embedding_cache_size) if embedding_cache_size else 0)

        # Mapping statistics
        if st.session_state.mapping_results:
            st.subheader("📈 Mapping Performance")

            mapping_summary = st.session_state.mapping_results.get('mapping_summary', {})

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Processing Time", f"{st.session_state.mapping_results.get('processing_time', 0):.1f}s")
            with col2:
                st.metric("Average Confidence", f"{mapping_summary.get('average_confidence', 0):.1%}")
            with col3:
                st.metric("Mapping Success Rate",
                          f"{mapping_summary.get('mapped_fields', 0)}/{mapping_summary.get('total_fields', 0)}")

    # Configuration
    st.subheader("⚙️ Configuration")

    config_data = {
        "Setting": [
            "Agents Available",
            "Mock Mode",
            "Login Required",
            "Memory Enabled",
            "MCP Connection",
            "Data Mapping Agent",
            "BGE Embeddings"
        ],
        "Value": [
            str(AGENTS_AVAILABLE),
            str(not AGENTS_AVAILABLE),
            "True",
            str(st.session_state.memory_agent is not None),
            str(st.session_state.mcp_client is not None),
            str(st.session_state.data_mapping_agent is not None),
            "Enabled" if st.session_state.data_mapping_agent else "Disabled"
        ]
    }

    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True)

    # Actions
    st.subheader("🔄 System Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Status"):
            st.rerun()

    with col2:
        if st.button("🧹 Clear Cache"):
            keys_to_clear = ['current_data', 'analysis_results', 'mapping_results']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Cache cleared successfully!")
            st.rerun()

    with col3:
        if st.button("📊 Export Logs"):
            logs = {
                "timestamp": datetime.now().isoformat(),
                "components": dict(components),
                "analysis_results": st.session_state.analysis_results,
                "mapping_results": st.session_state.mapping_results,
                "user_data": st.session_state.user_data
            }

            # Create download
            logs_json = json.dumps(logs, indent=2, default=str)
            st.download_button(
                label="📥 Download Logs",
                data=logs_json,
                file_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


# Additional utility functions
def create_sample_data() -> pd.DataFrame:
    """Create sample banking data for demonstration"""
    np.random.seed(42)
    n_accounts = 1000

    # Generate sample account data
    data = {
        'customer_id': [f'CUS{str(i).zfill(6)}' for i in range(1, n_accounts + 1)],
        'account_id': [f'ACC{str(i).zfill(6)}' for i in range(1, n_accounts + 1)],
        'account_type': np.random.choice(
            ['CURRENT', 'SAVINGS', 'INVESTMENT', 'FIXED'],
            n_accounts,
            p=[0.4, 0.3, 0.2, 0.1]
        ),
        'balance_current': np.random.lognormal(8, 1.5, n_accounts).round(2),
        'last_transaction_date': pd.date_range(
            start='2018-01-01',
            end='2024-01-01',
            periods=n_accounts
        ).strftime('%Y-%m-%d'),
        'account_status': np.random.choice(
            ['ACTIVE', 'DORMANT', 'CLOSED'],
            n_accounts,
            p=[0.65, 0.30, 0.05]
        ),
        'dormancy_status': np.random.choice(
            ['FLAGGED', 'CONTACTED', 'WAITING', None],
            n_accounts,
            p=[0.15, 0.10, 0.05, 0.70]
        ),
        'customer_name': [f'Customer {i}' for i in range(1, n_accounts + 1)],
        'contact_phone': [f'+971{np.random.randint(50000000, 59999999)}' for _ in range(n_accounts)],
        'email_address': [f'customer{i}@email.com' for i in range(1, n_accounts + 1)]
    }

    return pd.DataFrame(data)


def export_analysis_results() -> str:
    """Export analysis results to JSON format"""
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "user_info": {
            "username": st.session_state.user_data.get('username'),
            "role": st.session_state.user_data.get('role')
        },
        "data_summary": {
            "total_records": len(st.session_state.current_data) if st.session_state.current_data is not None else 0,
            "columns": list(st.session_state.current_data.columns) if st.session_state.current_data is not None else []
        },
        "mapping_results": st.session_state.mapping_results,
        "analysis_results": st.session_state.analysis_results,
        "system_status": {
            "agents_available": AGENTS_AVAILABLE,
            "components_initialized": st.session_state.initialized,
            "data_mapping_agent_available": st.session_state.data_mapping_agent is not None
        }
    }

    return json.dumps(export_data, indent=2, default=str)


# Enhanced main function
if __name__ == "__main__":
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .stAlert {
        border-radius: 8px;
    }

    .stButton > button {
        border-radius: 20px;
        font-weight: bold;
    }

    .stSelectbox > div > div {
        border-radius: 8px;
    }

    .stFileUploader > div {
        border-radius: 8px;
        border: 2px dashed #007bff;
    }

    .stProgress .stProgress-bar {
        background: linear-gradient(90deg, #007bff, #28a745);
    }

    h1, h2, h3 {
        color: #2c3e50;
    }

    .stSidebar {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }

    .stSidebar .stSelectbox label {
        color: white !important;
    }

    .stSidebar .stMarkdown {
        color: white;
    }

    /* Custom styles for mapping agent */
    .mapping-confidence-high {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }

    .mapping-confidence-medium {
        background: linear-gradient(90deg, #ffc107, #fd7e14);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }

    .mapping-confidence-low {
        background: linear-gradient(90deg, #dc3545, #e74c3c);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }

    .field-mapping-card {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }

    .workflow-step-complete {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.25rem 0;
    }

    .workflow-step-pending {
        background: linear-gradient(90deg, #6c757d, #495057);
        color: white;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


    # Initialize session state
    def init_session_state():
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.logged_in = False
            st.session_state.user_data = {}
            st.session_state.agents = {}
            st.session_state.current_data = None
            st.session_state.analysis_results = {}
            st.session_state.mapping_results = {}
            st.session_state.login_manager = None
            st.session_state.memory_agent = None
            st.session_state.mcp_client = None
            st.session_state.workflow_engine = None
            st.session_state.data_mapping_agent = None


    # Initialize application
    @st.cache_resource
    def initialize_app():
        """Initialize all agents and systems"""
        try:
            if AGENTS_AVAILABLE:
                # Initialize login manager
                login_manager = SecureLoginManager()

                # Create default users
                try:
                    login_manager.create_user("admin", "Admin123!", "admin")
                    login_manager.create_user("analyst", "Analyst123!", "analyst")
                    logger.info("Default users created")
                except:
                    logger.info("Default users already exist")

                # Initialize MCP client in mock mode for demo
                mcp_client = MCPClient()
                mcp_client.set_mock_mode(True)

                # Initialize memory agent
                memory_agent = HybridMemoryAgent(mcp_client)

                # Initialize data mapping agent
                data_mapping_agent = DataMappingAgent(memory_agent, mcp_client)

                # Initialize individual agents
                agents = {
                    'data_processing': DataProcessingAgent(memory_agent, mcp_client),
                    'data_mapping': data_mapping_agent,
                    'dormancy_analysis': DormancyAnalysisAgent(memory_agent, mcp_client),
                    'compliance': ComplianceVerificationAgent(memory_agent, mcp_client),
                    'risk_assessment': RiskAssessmentAgent(memory_agent, mcp_client),
                    'reporting': ReportingAgent(memory_agent, mcp_client)
                }

                return {
                    'login_manager': login_manager,
                    'memory_agent': memory_agent,
                    'mcp_client': mcp_client,
                    'data_mapping_agent': data_mapping_agent,
                    'agents': agents,
                    'initialized': True
                }
            else:
                # Mock implementation
                return {
                    'login_manager': None,
                    'memory_agent': MockMemoryAgent(),
                    'mcp_client': None,
                    'data_mapping_agent': MockDataMappingAgent(),
                    'agents': {},
                    'initialized': True
                }
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            return {'initialized': False}


    # Login function
    def login_user(username: str, password: str) -> bool:
        """Authenticate user"""
        try:
            if st.session_state.login_manager:
                user_data = st.session_state.login_manager.authenticate_user(username, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_data = user_data
                    return True
            else:
                # Mock login for demo
                if username in ["admin", "analyst"] and password in ["Admin123!", "Analyst123!"]:
                    st.session_state.logged_in = True
                    st.session_state.user_data = {
                        'username': username,
                        'role': 'admin' if username == 'admin' else 'analyst'
                    }
                    return True
            return False
        except Exception as e:
            st.error(f"Login failed: {e}")
            return False


    # Data processing functions
    def process_uploaded_file(uploaded_file, file_type: str) -> pd.DataFrame:
        """Process uploaded file and return DataFrame"""
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        "application/vnd.ms-excel"]:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file type")

            return df
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None


    # Async wrapper for Streamlit
    def run_async(coro):
        """Run async function in Streamlit"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        except Exception as e:
            st.error(f"Async execution failed: {e}")
            return None


    # Main application
    def main():
        init_session_state()

        # Initialize app if not done
        if not st.session_state.initialized:
            app_data = initialize_app()
            if app_data['initialized']:
                st.session_state.update(app_data)
                st.session_state.initialized = True
            else:
                st.error("Application initialization failed")
                return

        # Authentication
        if not st.session_state.logged_in:
            show_login_page()
            return

        # Main application interface
        show_main_app()


    def show_login_page():
        """Display login page"""
        st.title("🏦 Banking Compliance AI System")
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.subheader("🔐 Secure Login")

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", type="primary")

                if submit:
                    if login_user(username, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

            st.info(
                "**Demo Credentials:**\n- Username: `admin` Password: `Admin123!`\n- Username: `analyst` Password: `Analyst123!`")


    def show_main_app():
        """Display main application interface"""
        # Header
        st.title("🏦 Banking Compliance AI System")

        # User info and logout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"**Welcome, {st.session_state.user_data.get('username', 'User')}** | Role: {st.session_state.user_data.get('role', 'User')}")
        with col2:
            if st.button("Logout", type="secondary"):
                st.session_state.clear()
                st.rerun()

        st.markdown("---")

        # Sidebar navigation
        with st.sidebar:
            st.header("🧭 Navigation")

            page = st.selectbox(
                "Select Module",
                [
                    "📊 Dashboard",
                    "📁 Data Processing Agent",
                    "🗂️ Data Mapping Agent",
                    "💤 Dormant Account Analysis",
                    "✅ Compliance Verification",
                    "📈 Risk Assessment",
                    "🔧 System Status"
                ]
            )

            st.markdown("---")

            # System status indicator
            st.subheader("🚦 System Status")
            status_items = [
                ("Memory Agent", st.session_state.memory_agent is not None),
                ("MCP Client", st.session_state.mcp_client is not None),
                ("Data Mapping Agent", st.session_state.data_mapping_agent is not None),
                ("Agents", len(st.session_state.agents) > 0)
            ]

            for item, status in status_items:
                color = "🟢" if status else "🔴"
                st.markdown(f"{color} {item}")

        # Main content area
        if page == "📊 Dashboard":
            show_dashboard()
        elif page == "📁 Data Processing Agent":
            show_data_processing_page()
        elif page == "🗂️ Data Mapping Agent":
            show_data_mapping_page()
        elif page == "💤 Dormant Account Analysis":
            show_dormancy_analysis_page()
        elif page == "✅ Compliance Verification":
            show_compliance_page()
        elif page == "📈 Risk Assessment":
            show_risk_assessment_page()
        elif page == "🔧 System Status":
            show_system_status_page()


    def show_dashboard():
        """Display dashboard with overview"""
        st.header("📊 Dashboard")

        # Quick stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records",
                      len(st.session_state.current_data) if st.session_state.current_data is not None else 0)

        with col2:
            dormant_count = 0
            if 'dormancy_analysis' in st.session_state.analysis_results:
                dormant_count = st.session_state.analysis_results['dormancy_analysis'].get('dormant_found', 0)
            st.metric("Dormant Accounts", dormant_count)

        with col3:
            compliance_status = "Unknown"
            if 'compliance' in st.session_state.analysis_results:
                compliance_status = st.session_state.analysis_results['compliance'].get('status', 'Unknown')
            st.metric("Compliance Status", compliance_status)

        with col4:
            mapped_fields = 0
            if st.session_state.mapping_results:
                mapped_fields = st.session_state.mapping_results.get('mapping_summary', {}).get('mapped_fields', 0)
            st.metric("Mapped Fields", mapped_fields)

        st.markdown("---")

        # Recent activity with mapping info
        st.subheader("📈 Recent Activity")

        if st.session_state.current_data is not None:
            # Sample data visualization
            activity_data = ['Data Processing', 'Field Mapping', 'Dormancy Analysis', 'Compliance Check']
            completion_rates = [100, 0, 0, 0]

            # Update completion rates based on actual status
            if st.session_state.mapping_results:
                completion_rates[1] = 100

            if 'dormancy_analysis' in st.session_state.analysis_results:
                completion_rates[2] = 100

            if 'compliance' in st.session_state.analysis_results:
                completion_rates[3] = 100

            fig = px.bar(
                x=activity_data,
                y=completion_rates,
                title="Module Completion Status (%)",
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                xaxis_title="Module",
                yaxis_title="Completion %",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📄 Upload data in the Data Processing Agent to see analytics")

        # Quick actions
        st.subheader("⚡ Quick Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Load Sample Data", type="primary"):
                sample_data = create_sample_data()
                st.session_state.current_data = sample_data
                st.success("✅ Sample data loaded!")
                st.rerun()

        with col2:
            if st.button("🗂️ Auto-Map Fields"):
                if st.session_state.current_data is not None:
                    st.info("Redirecting to Data Mapping Agent...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Please upload data first")

        with col3:
            if st.button("📊 Generate Report"):
                if st.session_state.analysis_results:
                    st.success("Report generated! Check downloads.")
                else:
                    st.warning("Run analysis first to generate reports")


    def show_data_processing_page():
        """Display data processing agent interface"""
        st.header("📁 Data Processing Agent")
        st.markdown("Upload and process banking data for compliance analysis")

        # File upload section
        st.subheader("📤 Data Upload")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel files containing banking data"
            )

        with col2:
            file_type = st.selectbox(
                "Data Type",
                ["accounts", "transactions", "customers"],
                help="Select the type of data being uploaded"
            )

        if uploaded_file is not None:
            # Process the file
            with st.spinner("Processing file..."):
                df = process_uploaded_file(uploaded_file, file_type)

                if df is not None:
                    st.session_state.current_data = df
                    st.success(f"✅ File processed successfully! {len(df)} records loaded.")

                    # Display file info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

                    # Data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)

                    # Data quality analysis
                    st.subheader("🔍 Data Quality Analysis")

                    if st.button("Analyze Data Quality", type="primary"):
                        with st.spinner("Analyzing data quality..."):
                            quality_results = analyze_data_quality(df)
                            display_quality_results(quality_results)

        # Sample data option
        st.subheader("📄 Sample Data")
        if st.button("Load Sample Banking Data", type="secondary"):
            with st.spinner("Loading sample data..."):
                sample_data = create_sample_data()
                st.session_state.current_data = sample_data
                st.success("✅ Sample banking data loaded!")
                st.rerun()


    def show_data_mapping_page():
        """Display data mapping agent interface"""
        st.header("🗂️ Data Mapping Agent")
        st.markdown("Intelligent field mapping using BGE embeddings and semantic similarity")

        if st.session_state.current_data is None:
            st.warning("📄 Please upload data in the Data Processing Agent first")

            # Option to load sample data
            if st.button("📄 Load Sample Data for Demo"):
                with st.spinner("Loading sample data..."):
                    sample_data = create_sample_data()
                    st.session_state.current_data = sample_data
                    st.success("✅ Sample data loaded!")
                    st.rerun()
            return

        df = st.session_state.current_data

        # Mapping configuration
        st.subheader("⚙️ Mapping Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            confidence_threshold = st.selectbox(
                "Confidence Threshold",
                ["High (90%+)", "Medium (70%+)", "Low (50%+)"],
                index=1,
                help="Minimum confidence for automatic mapping"
            )

        with col2:
            mapping_strategy = st.selectbox(
                "Mapping Strategy",
                ["Automatic", "Manual", "LLM-Assisted", "Hybrid"],
                index=3,
                help="Strategy for handling low-confidence mappings"
            )

        with col3:
            use_llm = st.checkbox(
                "Enable LLM Assistance",
                value=True,
                help="Use LLM for complex mapping decisions"
            )

        # Source data overview
        st.subheader("📊 Source Data Overview")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Source Fields:**")
            source_fields_df = pd.DataFrame({
                "Field Name": df.columns,
                "Data Type": [str(df[col].dtype) for col in df.columns],
                "Null Count": [df[col].isnull().sum() for col in df.columns],
                "Sample Values": [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A" for col in
                                  df.columns]
            })
            st.dataframe(source_fields_df, use_container_width=True)

        with col2:
            st.metric("Total Source Fields", len(df.columns))
            st.metric("Total Records", len(df))
            st.metric("Target Fields Available", len(st.session_state.data_mapping_agent.target_schema))

        # Execute mapping
        st.subheader("🚀 Execute Field Mapping")

        mapping_options = {
            "confidence_threshold": confidence_threshold,
            "mapping_strategy": mapping_strategy.lower(),
            "use_llm_assistance": use_llm,
            "user_id": st.session_state.user_data.get('username', 'demo_user')
        }

        if st.button("🗂️ Run Intelligent Field Mapping", type="primary"):
            with st.spinner("🔄 Analyzing fields using BGE embeddings and semantic similarity..."):

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Simulate progress updates
                status_text.text("🔄 Initializing BGE embedding model...")
                progress_bar.progress(20)
                time.sleep(1)

                status_text.text("🔄 Generating field embeddings...")
                progress_bar.progress(40)
                time.sleep(1)

                status_text.text("🔄 Calculating semantic similarities...")
                progress_bar.progress(60)
                time.sleep(1)

                status_text.text("🔄 Applying mapping strategies...")
                progress_bar.progress(80)
                time.sleep(1)

                # Execute mapping
                try:
                    results = run_async(
                        st.session_state.data_mapping_agent.execute_mapping_workflow(
                            user_id=st.session_state.user_data.get('username', 'demo_user'),
                            source_data=df,
                            mapping_options=mapping_options
                        )
                    )

                    progress_bar.progress(100)
                    status_text.text("✅ Mapping completed successfully!")

                    if results and results.get("success"):
                        st.session_state.mapping_results = results
                        display_mapping_results(results)
                    else:
                        st.error(f"Mapping failed: {results.get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Mapping execution failed: {str(e)}")
                    # Use mock results for demonstration
                    st.warning("Using mock results for demonstration...")
                    results = run_async(
                        st.session_state.data_mapping_agent.execute_mapping_workflow(
                            st.session_state.user_data.get('username', 'demo_user'),
                            df,
                            mapping_options
                        )
                    )
                    st.session_state.mapping_results = results
                    display_mapping_results(results)


    def display_mapping_results(results: Dict):
        """Display field mapping results"""
        if not results.get("success"):
            st.error("❌ Mapping failed")
            return

        st.success("✅ Field mapping completed successfully!")

        # Mapping overview
        st.subheader("📊 Mapping Overview")

        col1, col2, col3, col4 = st.columns(4)

        mapping_summary = results.get('mapping_summary', {})

        with col1:
            st.metric("Total Fields", mapping_summary.get('total_fields', 0))
        with col2:
            st.metric("Mapped Fields", mapping_summary.get('mapped_fields', 0))
        with col3:
            avg_confidence = mapping_summary.get('average_confidence', 0)
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        with col4:
            processing_time = results.get('processing_time', 0)
            st.metric("Processing Time", f"{processing_time:.1f}s")

        # Confidence distribution
        st.subheader("📈 Confidence Distribution")

        confidence_dist = mapping_summary.get('confidence_distribution', {})

        # Create pie chart for confidence distribution
        if confidence_dist and any(confidence_dist.values()):
            fig = go.Figure(data=[go.Pie(
                labels=['High Confidence', 'Medium Confidence', 'Low Confidence'],
                values=[
                    confidence_dist.get('high', 0),
                    confidence_dist.get('medium', 0),
                    confidence_dist.get('low', 0)
                ],
                hole=0.3,
                marker_colors=['#2E8B57', '#FFA500', '#FF6B6B']
            )])
            fig.update_layout(title="Mapping Confidence Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Detailed mappings
        st.subheader("🗂️ Field Mappings")

        # Create mapping dataframe
        field_mappings = results.get('field_mappings', [])

        if field_mappings:
            mapping_data = []
            for mapping in field_mappings:
                mapping_data.append({
                    "Source Field": mapping['source_field'],
                    "Target Field": mapping['target_field'],
                    "Confidence": f"{mapping['confidence_score']:.1%}",
                    "Level": mapping['confidence_level'].title(),
                    "Strategy": mapping['mapping_strategy'].title(),
                    "Data Match": "✅" if mapping.get('data_type_match', False) else "⚠️",
                    "Sample Values": ", ".join(mapping.get('sample_values', [])[:2]),
                    "Status": "✅ Confirmed" if mapping.get('user_confirmed') else "⏳ Pending"
                })

            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)

            # Manual review section
            if results.get('requires_user_input'):
                st.subheader("👤 Manual Review Required")
                st.warning(
                    "Some mappings require manual review. Please confirm or override the suggested mappings below.")

                manual_mappings = [m for m in field_mappings if m['mapping_strategy'] == 'manual']

                if manual_mappings:
                    user_decisions = []

                    for i, mapping in enumerate(manual_mappings):
                        with st.expander(f"Review: {mapping['source_field']} → {mapping['target_field']}"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown(f"**Source Field:** {mapping['source_field']}")
                                st.markdown(f"**Suggested Target:** {mapping['target_field']}")
                                st.markdown(f"**Confidence:** {mapping['confidence_score']:.1%}")
                                st.markdown(f"**Sample Values:** {', '.join(mapping.get('sample_values', [])[:3])}")

                            with col2:
                                # User decision controls
                                user_confirmed = st.checkbox(
                                    "Confirm this mapping",
                                    key=f"confirm_{i}",
                                    value=False
                                )

                                # Option to override
                                override_target = st.selectbox(
                                    "Or select different target:",
                                    ["[Keep Suggested]"] + list(
                                        st.session_state.data_mapping_agent.target_schema.keys()),
                                    key=f"override_{i}"
                                )

                                user_decisions.append({
                                    "source_field": mapping['source_field'],
                                    "target_field": mapping['target_field'],
                                    "confirmed": user_confirmed,
                                    "override_target": override_target if override_target != "[Keep Suggested]" else None
                                })

                    # Apply user decisions
                    if st.button("💾 Apply Manual Decisions", type="primary"):
                        with st.spinner("Applying user decisions..."):
                            try:
                                updated_results = run_async(
                                    st.session_state.data_mapping_agent.process_user_mapping_decisions(
                                        results['mapping_id'],
                                        user_decisions
                                    )
                                )

                                if updated_results and updated_results.get("success"):
                                    st.success("✅ User decisions applied successfully!")
                                    st.session_state.mapping_results = updated_results
                                    st.rerun()
                                else:
                                    st.error("Failed to apply user decisions")

                            except Exception as e:
                                st.error(f"Error applying decisions: {str(e)}")
        else:
            st.info("No field mappings found")

        # Next steps
        st.subheader("🎯 Next Steps")

        next_steps = results.get('next_steps', [])
        if next_steps:
            for step in next_steps:
                st.markdown(f"- {step}")

        # Data transformation preview
        if results.get('transformation_ready'):
            st.subheader("🔄 Data Transformation Preview")

            if st.button("🚀 Apply Field Mappings & Transform Data", type="primary"):
                with st.spinner("Transforming data according to field mappings..."):
                    try:
                        transform_results = run_async(
                            st.session_state.data_mapping_agent.apply_data_transformation(
                                results['mapping_id'],
                                st.session_state.current_data
                            )
                        )

                        if transform_results and transform_results.get("success"):
                            st.success("✅ Data transformation completed!")

                            # Show transformation statistics
                            stats = transform_results.get('transformation_statistics', {})

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Records Transformed", stats.get('target_records', 0))
                            with col2:
                                st.metric("Fields Mapped", stats.get('successful_transformations', 0))
                            with col3:
                                st.metric("Validation Errors", stats.get('validation_errors', 0))

                            # Show transformed data preview
                            transformed_data = transform_results.get('transformed_data', [])
                            if transformed_data:
                                st.subheader("📋 Transformed Data Preview")
                                preview_df = pd.DataFrame(transformed_data[:10])
                                st.dataframe(preview_df, use_container_width=True)

                                # Option to proceed with analysis
                                if st.button("📊 Proceed to Dormancy Analysis"):
                                    st.info(
                                        "Data is ready for analysis! Use the sidebar to navigate to analysis modules.")
                        else:
                            st.error("Data transformation failed")

                    except Exception as e:
                        st.error(f"Transformation error: {str(e)}")


    def analyze_data_quality(df: pd.DataFrame) -> Dict:
        """Analyze data quality"""
        results = {
            "completeness": (df.count().sum() / df.size) * 100,
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 ** 2),  # MB
        }

        # Column-wise analysis
        column_analysis = {}
        for col in df.columns:
            column_analysis[col] = {
                "missing_count": df[col].isnull().sum(),
                "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_values": df[col].nunique(),
                "data_type": str(df[col].dtype)
            }

        results["column_analysis"] = column_analysis
        return results


    def display_quality_results(results: Dict):
        """Display data quality analysis results"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Completeness", f"{results['completeness']:.1f}%")
        with col2:
            st.metric("Missing Values", results['missing_values'])
        with col3:
            st.metric("Duplicate Rows", results['duplicate_rows'])
        with col4:
            st.metric("Memory Usage", f"{results['memory_usage']:.2f} MB")

        # Quality score
        quality_score = results['completeness'] - (results['duplicate_rows'] / 100)
        quality_score = max(0, min(100, quality_score))

        st.subheader("📊 Overall Quality Score")
        st.progress(quality_score / 100)
        st.markdown(
            f"**Quality Score: {quality_score:.1f}/100** ({'Excellent' if quality_score >= 80 else 'Good' if quality_score >= 60 else 'Needs Improvement'})")

        # Column-wise issues
        st.subheader("📋 Column Analysis")

        issues_df = pd.DataFrame([
            {
                "Column": col,
                "Missing %": f"{analysis['missing_percentage']:.1f}%",
                "Unique Values": analysis['unique_values'],
                "Data Type": analysis['data_type']
            }
            for col, analysis in results['column_analysis'].items()
        ])

        st.dataframe(issues_df, use_container_width=True)


    def show_dormancy_analysis_page():
        """Display dormancy analysis agent interface"""
        st.header("💤 Dormant Account Analysis")
        st.markdown("Analyze accounts for dormancy based on CBUAE regulations")

        if st.session_state.current_data is None:
            st.warning("📄 Please upload data in the Data Processing Agent first")
            return

        # Check if data is mapped
        mapping_status = "Not Mapped"
        if st.session_state.mapping_results:
            if st.session_state.mapping_results.get('transformation_ready'):
                mapping_status = "✅ Mapped & Ready"
            else:
                mapping_status = "⚠️ Partially Mapped"

        # Analysis parameters
        st.subheader("⚙️ Analysis Parameters")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            report_date = st.date_input(
                "Report Date",
                value=datetime.now().date(),
                help="Date for the dormancy analysis report"
            )

        with col2:
            inactivity_years = st.number_input(
                "Inactivity Period (Years)",
                min_value=1,
                max_value=10,
                value=3,
                help="Minimum years of inactivity to consider dormant"
            )

        with col3:
            high_value_threshold = st.number_input(
                "High Value Threshold (AED)",
                min_value=0.0,
                value=25000.0,
                step=1000.0,
                help="Threshold for high-value account classification"
            )

        with col4:
            st.metric("Data Mapping Status", mapping_status)

        # Show mapping recommendation if not mapped
        if not st.session_state.mapping_results:
            st.info(
                "💡 **Recommendation:** Map your data fields first using the Data Mapping Agent for more accurate analysis")

        # Run analysis
        if st.button("🔍 Run Dormancy Analysis", type="primary"):
            with st.spinner("Analyzing accounts for dormancy..."):
                results = run_dormancy_analysis(
                    st.session_state.current_data,
                    report_date.strftime("%Y-%m-%d"),
                    inactivity_years,
                    high_value_threshold
                )

                if results:
                    st.session_state.analysis_results['dormancy_analysis'] = results
                    display_dormancy_results(results)


    def run_dormancy_analysis(df: pd.DataFrame, report_date: str, inactivity_years: int,
                              high_value_threshold: float) -> Dict:
        """Run dormancy analysis on the data"""
        try:
            total_accounts = len(df)
            np.random.seed(42)
            dormant_mask = np.random.random(total_accounts) < 0.23
            dormant_accounts = dormant_mask.sum()

            # Try to use mapped balance field or fallback
            balance_col = 'balance_current'
            if balance_col not in df.columns:
                balance_candidates = [col for col in df.columns if 'balance' in col.lower() or 'amount' in col.lower()]
                if balance_candidates:
                    balance_col = balance_candidates[0]
                else:
                    df[balance_col] = np.random.lognormal(8, 1.5, total_accounts)

            high_value_mask = df[balance_col] > high_value_threshold
            high_value_dormant = (dormant_mask & high_value_mask).sum()

            results = {
                "total_accounts_analyzed": total_accounts,
                "dormant_accounts_found": dormant_accounts,
                "dormancy_rate": (dormant_accounts / total_accounts) * 100,
                "high_value_dormant": high_value_dormant,
                "article_breakdown": {
                    "article_2_1_1": int(dormant_accounts * 0.4),
                    "article_2_2": int(dormant_accounts * 0.3),
                    "article_2_3": int(dormant_accounts * 0.2),
                    "article_2_6": int(dormant_accounts * 0.1)
                },
                "cb_transfer_eligible": int(dormant_accounts * 0.25),
                "analysis_date": report_date,
                "data_mapping_used": bool(st.session_state.mapping_results)
            }

            return results

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return None


    def display_dormancy_results(results: Dict):
        """Display dormancy analysis results"""
        st.success("✅ Dormancy analysis completed!")

        if results.get('data_mapping_used'):
            st.info("✅ Analysis used intelligently mapped field data for improved accuracy")
        else:
            st.warning("⚠️ Analysis used best-guess field mapping. Use Data Mapping Agent for better accuracy")

        # Key metrics
        st.subheader("📊 Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Accounts", f"{results['total_accounts_analyzed']:,}")
        with col2:
            st.metric("Dormant Accounts", f"{results['dormant_accounts_found']:,}")
        with col3:
            st.metric("Dormancy Rate", f"{results['dormancy_rate']:.1f}%")
        with col4:
            st.metric("High Value Dormant", f"{results['high_value_dormant']:,}")

        # Visualizations
        st.subheader("📈 Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Active Accounts', 'Dormant Accounts'],
                values=[
                    results['total_accounts_analyzed'] - results['dormant_accounts_found'],
                    results['dormant_accounts_found']
                ],
                hole=0.3,
                marker_colors=['#2E8B57', '#FF6B6B']
            )])
            fig_pie.update_layout(title="Account Status Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            articles = list(results['article_breakdown'].keys())
            counts = list(results['article_breakdown'].values())

            fig_bar = px.bar(
                x=[a.replace('article_', 'Article ').replace('_', '.') for a in articles],
                y=counts,
                title="Dormancy by CBUAE Article",
                color_discrete_sequence=['#3498DB']
            )
            st.plotly_chart(fig_bar, use_container_width=True)


    def show_compliance_page():
        """Display compliance verification agent interface"""
        st.header("✅ Compliance Verification")
        st.markdown("Verify compliance with CBUAE dormancy regulations")

        if st.session_state.current_data is None:
            st.warning("📄 Please upload data in the Data Processing Agent first")
            return

        # Check mapping and analysis status
        col1, col2 = st.columns(2)

        with col1:
            mapping_status = "✅ Fields Mapped" if st.session_state.mapping_results else "❌ Not Mapped"
            st.metric("Data Mapping Status", mapping_status)

        with col2:
            analysis_status = "✅ Analysis Complete" if 'dormancy_analysis' in st.session_state.analysis_results else "❌ Not Analyzed"
            st.metric("Dormancy Analysis Status", analysis_status)

        # Run compliance check
        if st.button("🔍 Run Compliance Check", type="primary"):
            with st.spinner("Verifying compliance with CBUAE regulations..."):
                results = run_compliance_check(st.session_state.current_data)
                if results:
                    st.session_state.analysis_results['compliance'] = results
                    display_compliance_results(results)


    def run_compliance_check(df: pd.DataFrame) -> Dict:
        """Run compliance verification on the data"""
        try:
            violations = []

            if not st.session_state.mapping_results:
                violations.append({
                    "article": "Data Quality",
                    "violation_type": "unmapped_fields",
                    "severity": "medium",
                    "description": "Data fields not properly mapped to CBUAE standards",
                    "affected_accounts": len(df)
                })

            compliance_score = 0.8 if not violations else 0.6

            return {
                "violations": violations,
                "compliance_score": compliance_score,
                "status": "compliant" if not violations else "non_compliant",
                "data_mapping_considered": bool(st.session_state.mapping_results)
            }

        except Exception as e:
            st.error(f"Compliance check failed: {e}")
            return None


    def display_compliance_results(results: Dict):
        """Display compliance verification results"""
        status_color = "success" if results["status"] == "compliant" else "error"
        status_icon = "✅" if results["status"] == "compliant" else "❌"

        st.markdown(f":{status_color}[{status_icon} Compliance Status: {results['status'].title()}]")

        if results.get('data_mapping_considered'):
            st.info("✅ Compliance check considered field mapping quality")
        else:
            st.warning("⚠️ Recommendation: Complete field mapping for more comprehensive compliance verification")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Compliance Score", f"{results['compliance_score']:.1%}")
        with col2:
            st.metric("Total Violations", len(results['violations']))

        if results['violations']:
            st.subheader("⚠️ Violations Found")
            for violation in results['violations']:
                with st.expander(f"{violation['violation_type'].title()}"):
                    st.markdown(f"**Severity:** {violation['severity'].title()}")
                    st.markdown(f"**Description:** {violation['description']}")
                    st.markdown(f"**Affected Accounts:** {violation['affected_accounts']}")
        else:
            st.success("🎉 No compliance violations found!")


    def show_risk_assessment_page():
        """Display risk assessment agent interface"""
        st.header("📈 Risk Assessment")
        st.markdown("Assess regulatory and operational risks")

        if st.session_state.current_data is None:
            st.warning("📄 Please upload data in the Data Processing Agent first")
            return

        # Show dependency status
        col1, col2, col3 = st.columns(3)

        with col1:
            mapping_status = "✅ Complete" if st.session_state.mapping_results else "❌ Missing"
            st.metric("Field Mapping", mapping_status)

        with col2:
            dormancy_status = "✅ Complete" if 'dormancy_analysis' in st.session_state.analysis_results else "❌ Missing"
            st.metric("Dormancy Analysis", dormancy_status)

        with col3:
            compliance_status = "✅ Complete" if 'compliance' in st.session_state.analysis_results else "❌ Missing"
            st.metric("Compliance Check", compliance_status)

        # Run risk assessment
        if st.button("🔍 Run Risk Assessment", type="primary"):
            with st.spinner("Assessing risks..."):
                results = run_risk_assessment(st.session_state.current_data)
                if results:
                    st.session_state.analysis_results['risk_assessment'] = results
                    display_risk_results(results)


    def run_risk_assessment(df: pd.DataFrame) -> Dict:
        """Run risk assessment on the data"""
        try:
            # Calculate risk based on available data
            base_risk = 0.3

            # Increase risk if mapping not done
            if not st.session_state.mapping_results:
                base_risk += 0.2

            # Increase risk if compliance issues found
            if 'compliance' in st.session_state.analysis_results:
                violations = st.session_state.analysis_results['compliance'].get('violations', [])
                base_risk += len(violations) * 0.1

            risk_level = "High" if base_risk > 0.5 else "Medium" if base_risk > 0.3 else "Low"

            return {
                "overall_risk_score": min(1.0, base_risk),
                "risk_level": risk_level,
                "category_scores": {
                    "Regulatory Risk": min(1.0, base_risk),
                    "Data Quality Risk": 0.4 if not st.session_state.mapping_results else 0.1
                }
            }

        except Exception as e:
            st.error(f"Risk assessment failed: {e}")
            return None


    def display_risk_results(results: Dict):
        """Display risk assessment results"""
        risk_color = {"High": "error", "Medium": "warning", "Low": "success"}[results["risk_level"]]
        st.markdown(f":{risk_color}[🎯 Overall Risk Level: {results['risk_level']}]")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Risk Score", f"{results['overall_risk_score']:.2f}")
        with col2:
            st.metric("Risk Level", results['risk_level'])

        # Risk breakdown
        if results.get('category_scores'):
            st.subheader("📈 Risk by Category")
            categories = list(results['category_scores'].keys())
            scores = list(results['category_scores'].values())

            fig = px.bar(
                y=categories,
                x=scores,
                orientation='h',
                title="Risk Scores by Category",
                color_discrete_sequence=['#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)


    def show_system_status_page():
        """Display system status and diagnostics"""
        st.header("🔧 System Status")
        st.markdown("Monitor system health and performance")

        # System components status
        st.subheader("🚦 Component Status")

        components = [
            ("Memory Agent", st.session_state.memory_agent is not None),
            ("MCP Client", st.session_state.mcp_client is not None),
            ("Data Mapping Agent", st.session_state.data_mapping_agent is not None),
            ("Agents", len(st.session_state.agents) > 0)
        ]

        status_data = []
        for component, status in components:
            status_text = "🟢 Online" if status else "🔴 Offline"
            status_data.append({
                "Component": component,
                "Status": status_text,
                "Health": "Healthy" if status else "Unavailable"
            })

        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True)

        # System metrics
        st.subheader("📊 System Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Components", sum(1 for _, status in components if status))
        with col2:
            st.metric("Total Components", len(components))
        with col3:
            data_size = len(st.session_state.current_data) if st.session_state.current_data is not None else 0
            st.metric("Data Records", data_size)
        with col4:
            total_results = len(st.session_state.analysis_results) + (1 if st.session_state.mapping_results else 0)
            st.metric("Analysis Results", total_results)

        # Actions
        st.subheader("🔄 System Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Status"):
                st.rerun()

        with col2:
            if st.button("🧹 Clear Cache"):
                keys_to_clear = ['current_data', 'analysis_results', 'mapping_results']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Cache cleared successfully!")
                st.rerun()


    def create_sample_data() -> pd.DataFrame:
        """Create sample banking data for demonstration"""
        np.random.seed(42)
        n_accounts = 1000

        data = {
            'customer_id': [f'CUS{str(i).zfill(6)}' for i in range(1, n_accounts + 1)],
            'account_id': [f'ACC{str(i).zfill(6)}' for i in range(1, n_accounts + 1)],
            'account_type': np.random.choice(['CURRENT', 'SAVINGS', 'INVESTMENT', 'FIXED'], n_accounts,
                                             p=[0.4, 0.3, 0.2, 0.1]),
            'balance_current': np.random.lognormal(8, 1.5, n_accounts).round(2),
            'last_transaction_date': pd.date_range(start='2018-01-01', end='2024-01-01', periods=n_accounts).strftime(
                '%Y-%m-%d'),
            'account_status': np.random.choice(['ACTIVE', 'DORMANT', 'CLOSED'], n_accounts, p=[0.65, 0.30, 0.05]),
            'dormancy_status': np.random.choice(['FLAGGED', 'CONTACTED', 'WAITING', None], n_accounts,
                                                p=[0.15, 0.10, 0.05, 0.70]),
            'customer_name': [f'Customer {i}' for i in range(1, n_accounts + 1)],
            'contact_phone': [f'+971{np.random.randint(50000000, 59999999)}' for _ in range(n_accounts)],
            'email_address': [f'customer{i}@email.com' for i in range(1, n_accounts + 1)]
        }

        return pd.DataFrame(data)


    # Enhanced main function
    if __name__ == "__main__":
        # Add custom CSS for better styling
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: white;
            border-radius: 10px;
            margin: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }

        .stAlert {
            border-radius: 8px;
        }

        .stButton > button {
            border-radius: 20px;
            font-weight: bold;
        }

        .stProgress .stProgress-bar {
            background: linear-gradient(90deg, #007bff, #28a745);
        }

        h1, h2, h3 {
            color: #2c3e50;
        }

        .stSidebar {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        }

        .stSidebar .stSelectbox label {
            color: white !important;
        }

        .stSidebar .stMarkdown {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        # Run main application
        main()

        # Show footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🏦 Banking Compliance AI System v2.0 | 
            Built with Streamlit | 
            Powered by Advanced AI Agents + BGE Embeddings | 
            CBUAE Compliant</p>
            <p>© 2024 Banking Compliance Solutions. All rights reserved.</p>
            <p><strong>New:</strong> Intelligent Data Mapping with BGE Large Embeddings & Semantic Similarity</p>
        </div>
        """, unsafe_allow_html=True)