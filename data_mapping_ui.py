"""
Data Mapping UI Component
Modular UI component for CBUAE banking compliance data mapping
Can be integrated into larger Streamlit applications
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Try importing BGE for embeddings
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False


class MappingConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MappingResult:
    source_field: str
    target_field: str
    confidence_score: float
    confidence_level: MappingConfidence
    mapping_method: str
    is_required: bool = False


class DataMappingUI:
    """Reusable Data Mapping UI Component"""

    def __init__(self):
        self.cbuae_schema = self._load_cbuae_schema()
        self.bge_model = self._load_bge_model()

    def _load_cbuae_schema(self) -> Dict:
        """Load CBUAE banking compliance schema"""
        return {
            # Core Required Fields
            'customer_id': {
                'description': 'Unique customer identifier',
                'type': 'string',
                'required': True,
                'keywords': ['customer', 'client', 'cust', 'id', 'customer_id'],
                'examples': ['CUS123456', 'CUST000001']
            },
            'account_id': {
                'description': 'Unique account identifier',
                'type': 'string',
                'required': True,
                'keywords': ['account', 'acc', 'id', 'account_id'],
                'examples': ['ACC123456789', 'ACCT0001']
            },
            'account_type': {
                'description': 'Type of account (CURRENT, SAVINGS, etc.)',
                'type': 'string',
                'required': True,
                'keywords': ['type', 'account_type', 'product', 'category'],
                'examples': ['SAVINGS', 'CURRENT', 'FIXED_DEPOSIT']
            },
            'account_status': {
                'description': 'Account status (ACTIVE, DORMANT, CLOSED)',
                'type': 'string',
                'required': True,
                'keywords': ['status', 'state', 'account_status'],
                'examples': ['ACTIVE', 'DORMANT', 'CLOSED']
            },
            'dormancy_status': {
                'description': 'Dormancy classification',
                'type': 'string',
                'required': True,
                'keywords': ['dormancy', 'dormant', 'inactive', 'dormancy_status'],
                'examples': ['Not_Dormant', 'Potentially_Dormant', 'Dormant']
            },
            'balance_current': {
                'description': 'Current account balance',
                'type': 'numeric',
                'required': True,
                'keywords': ['balance', 'amount', 'current', 'balance_current'],
                'examples': ['15000.50', '250000.00', '1000.25']
            },
            'last_transaction_date': {
                'description': 'Date of last transaction',
                'type': 'date',
                'required': True,
                'keywords': ['transaction', 'last', 'date', 'activity', 'last_transaction_date'],
                'examples': ['2023-11-15', '2024-01-10', '2023-09-20']
            },
            # Additional important fields...
            'customer_type': {
                'description': 'Type of customer (INDIVIDUAL, CORPORATE)',
                'type': 'string',
                'required': False,
                'keywords': ['customer_type', 'client_type', 'type'],
                'examples': ['INDIVIDUAL', 'CORPORATE']
            },
            'full_name_en': {
                'description': 'Customer full name in English',
                'type': 'string',
                'required': False,
                'keywords': ['name', 'full_name', 'customer_name', 'full_name_en'],
                'examples': ['Ahmed Al Mahmoud', 'Sarah Johnson']
            },
            'phone_primary': {
                'description': 'Primary phone number',
                'type': 'numeric',
                'required': False,
                'keywords': ['phone', 'mobile', 'telephone', 'primary', 'phone_primary'],
                'examples': ['971501234567', '971509876543']
            },
            'email_primary': {
                'description': 'Primary email address',
                'type': 'string',
                'required': False,
                'keywords': ['email', 'mail', 'primary', 'email_primary'],
                'examples': ['customer@email.com', 'user@domain.com']
            },
            'nationality': {
                'description': 'Customer nationality',
                'type': 'string',
                'required': False,
                'keywords': ['nationality', 'country', 'citizen'],
                'examples': ['UAE', 'INDIA', 'PAKISTAN']
            },
            'currency': {
                'description': 'Account currency',
                'type': 'string',
                'required': False,
                'keywords': ['currency', 'ccy', 'cur'],
                'examples': ['AED', 'USD', 'EUR']
            },
            'opening_date': {
                'description': 'Account opening date',
                'type': 'date',
                'required': False,
                'keywords': ['opening', 'open', 'created', 'start', 'opening_date'],
                'examples': ['2020-01-15', '2019-05-20', '2021-03-10']
            },
            'dormancy_trigger_date': {
                'description': 'Date when dormancy was triggered',
                'type': 'date',
                'required': False,
                'keywords': ['trigger', 'dormancy', 'date', 'dormancy_trigger_date'],
                'examples': ['2021-05-15', '2020-08-20', '2022-01-10']
            },
            'contact_attempts_made': {
                'description': 'Number of contact attempts made',
                'type': 'numeric',
                'required': False,
                'keywords': ['contact', 'attempts', 'tried', 'contact_attempts_made'],
                'examples': ['3', '5', '2']
            },
            'cb_transfer_amount': {
                'description': 'Amount transferred to Central Bank',
                'type': 'numeric',
                'required': False,
                'keywords': ['transfer', 'amount', 'central', 'bank', 'cb_transfer_amount'],
                'examples': ['15000.50', '250000.00', '1000.25']
            }
        }

    def _load_bge_model(self):
        """Load BGE model for semantic mapping"""
        if not BGE_AVAILABLE:
            return None

        try:
            model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            return model
        except Exception:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                return model
            except Exception:
                return None

    def render_mapping_interface(self, data: pd.DataFrame, container_key: str = "mapping") -> Dict:
        """
        Main interface for data mapping

        Args:
            data: DataFrame to map
            container_key: Unique key for this mapping instance

        Returns:
            Dict with mapping results
        """
        st.markdown("### 🗺️ Data Mapping to CBUAE Schema")

        # Method selection
        mapping_method = st.selectbox(
            "Select Mapping Method:",
            ["🤖 Automatic (BGE Semantic)" if BGE_AVAILABLE else "🔤 Automatic (Keywords)",
             "✏️ Manual Mapping",
             "🔄 Hybrid (Auto + Manual)"],
            key=f"{container_key}_method"
        )

        # Show schema info
        self._show_schema_summary()

        # Execute mapping based on method
        if st.button("🚀 Start Mapping", key=f"{container_key}_start"):
            if "Automatic" in mapping_method:
                results = self._run_automatic_mapping(data, container_key)
            elif "Manual" in mapping_method:
                results = self._run_manual_mapping(data, container_key)
            else:  # Hybrid
                results = self._run_hybrid_mapping(data, container_key)

            # Store results in session state
            st.session_state[f"{container_key}_results"] = results
            return results

        # Show existing results if available
        if f"{container_key}_results" in st.session_state:
            results = st.session_state[f"{container_key}_results"]
            self._display_mapping_results(results, container_key)
            return results

        return {}

    def _show_schema_summary(self):
        """Show CBUAE schema summary"""
        with st.expander("📚 CBUAE Schema Overview", expanded=False):
            required_fields = [f for f, info in self.cbuae_schema.items() if info['required']]
            optional_fields = [f for f, info in self.cbuae_schema.items() if not info['required']]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔴 Required Fields:**")
                for field in required_fields:
                    st.write(f"• `{field}` - {self.cbuae_schema[field]['description']}")

            with col2:
                st.markdown("**⚪ Optional Fields (Sample):**")
                for field in optional_fields[:5]:
                    st.write(f"• `{field}` - {self.cbuae_schema[field]['description']}")
                if len(optional_fields) > 5:
                    st.caption(f"... and {len(optional_fields) - 5} more optional fields")

    def _run_automatic_mapping(self, data: pd.DataFrame, container_key: str) -> Dict:
        """Run automatic mapping using BGE or keyword matching"""
        with st.spinner("🔄 Running automatic mapping..."):
            source_columns = list(data.columns)

            if self.bge_model:
                mappings = self._bge_semantic_mapping(source_columns)
                method = "BGE Semantic"
            else:
                mappings = self._keyword_mapping(source_columns)
                method = "Keyword Matching"

            # Calculate statistics
            required_fields = [f for f, info in self.cbuae_schema.items() if info['required']]
            mapped_required = len([m for m in mappings if m.is_required])
            compliance_score = (mapped_required / len(required_fields)) * 100

            results = {
                'mappings': mappings,
                'method': method,
                'compliance_score': compliance_score,
                'required_mapped': mapped_required,
                'required_total': len(required_fields),
                'total_mapped': len(mappings),
                'timestamp': datetime.now()
            }

            if compliance_score >= 85:
                st.success(f"✅ Excellent mapping! {compliance_score:.1f}% CBUAE compliance achieved")
            elif compliance_score >= 70:
                st.warning(f"⚠️ Good mapping: {compliance_score:.1f}% compliance - review needed")
            else:
                st.error(f"❌ Low compliance: {compliance_score:.1f}% - manual mapping required")

            return results

    def _bge_semantic_mapping(self, source_columns: List[str]) -> List[MappingResult]:
        """BGE semantic mapping implementation"""
        # Prepare texts for embedding
        source_texts = [col.replace('_', ' ').lower() for col in source_columns]
        target_texts = []
        target_fields = []

        for field, info in self.cbuae_schema.items():
            target_fields.append(field)
            enhanced_text = f"{field.replace('_', ' ')} {info['description']} {' '.join(info['keywords'][:3])}"
            target_texts.append(enhanced_text)

        # Generate embeddings and calculate similarities
        source_embeddings = self.bge_model.encode(source_texts, normalize_embeddings=True)
        target_embeddings = self.bge_model.encode(target_texts, normalize_embeddings=True)
        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)

        # Create mappings
        mappings = []
        used_targets = set()

        # Sort by highest similarity first
        source_target_pairs = []
        for i, source_col in enumerate(source_columns):
            for j, target_field in enumerate(target_fields):
                similarity = similarity_matrix[i][j]
                source_target_pairs.append((i, j, similarity, source_col, target_field))

        source_target_pairs.sort(key=lambda x: x[2], reverse=True)
        used_sources = set()

        for i, j, similarity, source_col, target_field in source_target_pairs:
            if (source_col not in used_sources and
                    target_field not in used_targets and
                    similarity >= 0.3):

                # Determine confidence
                if similarity >= 0.7:
                    confidence = MappingConfidence.HIGH
                elif similarity >= 0.5:
                    confidence = MappingConfidence.MEDIUM
                else:
                    confidence = MappingConfidence.LOW

                mapping = MappingResult(
                    source_field=source_col,
                    target_field=target_field,
                    confidence_score=similarity,
                    confidence_level=confidence,
                    mapping_method="BGE Semantic",
                    is_required=self.cbuae_schema[target_field]['required']
                )

                mappings.append(mapping)
                used_sources.add(source_col)
                used_targets.add(target_field)

        return mappings

    def _keyword_mapping(self, source_columns: List[str]) -> List[MappingResult]:
        """Keyword-based mapping implementation"""
        mappings = []
        used_sources = set()

        for target_field, info in self.cbuae_schema.items():
            best_match = None
            best_score = 0

            for source_col in source_columns:
                if source_col in used_sources:
                    continue

                score = 0
                source_lower = source_col.lower()

                # Check keywords
                for keyword in info['keywords']:
                    if keyword in source_lower:
                        score += 10
                        if source_lower == keyword:
                            score += 20

                # Exact match bonus
                if source_lower == target_field.lower():
                    score += 50

                if score > best_score:
                    best_score = score
                    best_match = source_col

            if best_match and best_score >= 10:
                confidence = (MappingConfidence.HIGH if best_score >= 40 else
                              MappingConfidence.MEDIUM if best_score >= 25 else
                              MappingConfidence.LOW)

                mapping = MappingResult(
                    source_field=best_match,
                    target_field=target_field,
                    confidence_score=best_score / 100.0,
                    confidence_level=confidence,
                    mapping_method="Keyword Match",
                    is_required=info['required']
                )

                mappings.append(mapping)
                used_sources.add(best_match)

        return mappings

    def _run_manual_mapping(self, data: pd.DataFrame, container_key: str) -> Dict:
        """Run manual mapping interface"""
        st.markdown("#### ✏️ Manual Mapping Interface")

        source_columns = list(data.columns)
        mappings = []

        # Create mapping interface for required fields first
        required_fields = [(f, info) for f, info in self.cbuae_schema.items() if info['required']]
        optional_fields = [(f, info) for f, info in self.cbuae_schema.items() if not info['required']]

        st.markdown("**🔴 Required Fields (Must Map for Compliance):**")

        for target_field, info in required_fields:
            col1, col2, col3 = st.columns([2, 3, 1])

            with col1:
                st.markdown(f"**{target_field}** ⭐")
                st.caption(info['description'])

            with col2:
                selected_source = st.selectbox(
                    "Map to source field:",
                    ['-- Select --'] + source_columns,
                    key=f"{container_key}_manual_{target_field}"
                )

                if selected_source != '-- Select --':
                    mapping = MappingResult(
                        source_field=selected_source,
                        target_field=target_field,
                        confidence_score=1.0,
                        confidence_level=MappingConfidence.HIGH,
                        mapping_method="Manual",
                        is_required=True
                    )
                    mappings.append(mapping)

            with col3:
                st.text("Required")

        # Optional fields in expander
        with st.expander("⚪ Optional Fields", expanded=False):
            for target_field, info in optional_fields[:10]:  # Show first 10 optional fields
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown(f"**{target_field}**")
                    st.caption(info['description'])

                with col2:
                    selected_source = st.selectbox(
                        "Map to source field:",
                        ['-- Select --'] + source_columns,
                        key=f"{container_key}_optional_{target_field}"
                    )

                    if selected_source != '-- Select --':
                        mapping = MappingResult(
                            source_field=selected_source,
                            target_field=target_field,
                            confidence_score=1.0,
                            confidence_level=MappingConfidence.HIGH,
                            mapping_method="Manual",
                            is_required=False
                        )
                        mappings.append(mapping)

        # Calculate results
        required_mapped = len([m for m in mappings if m.is_required])
        required_total = len(required_fields)
        compliance_score = (required_mapped / required_total) * 100 if required_total > 0 else 0

        results = {
            'mappings': mappings,
            'method': 'Manual',
            'compliance_score': compliance_score,
            'required_mapped': required_mapped,
            'required_total': required_total,
            'total_mapped': len(mappings),
            'timestamp': datetime.now()
        }

        return results

    def _run_hybrid_mapping(self, data: pd.DataFrame, container_key: str) -> Dict:
        """Run hybrid mapping (automatic + manual review)"""
        # First run automatic
        auto_results = self._run_automatic_mapping(data, f"{container_key}_auto")

        st.markdown("---")
        st.markdown("#### 🔍 Review and Adjust Automatic Mappings")

        # Allow manual adjustments
        mappings = auto_results['mappings'].copy()
        source_columns = list(data.columns)

        for i, mapping in enumerate(mappings):
            if mapping.confidence_level != MappingConfidence.HIGH:
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                with col1:
                    st.text(f"Source: {mapping.source_field}")

                with col2:
                    st.text(f"Target: {mapping.target_field}")
                    st.caption(self.cbuae_schema[mapping.target_field]['description'])

                with col3:
                    new_target = st.selectbox(
                        "Adjust mapping:",
                        ['Keep Current'] + list(self.cbuae_schema.keys()),
                        key=f"{container_key}_hybrid_{i}"
                    )

                    if new_target != 'Keep Current':
                        mapping.target_field = new_target
                        mapping.mapping_method = "Hybrid"

                with col4:
                    confidence_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                    st.text(f"{confidence_colors[mapping.confidence_level.value]} {mapping.confidence_level.value}")

        # Recalculate statistics
        required_mapped = len([m for m in mappings if m.is_required])
        required_total = len([f for f, info in self.cbuae_schema.items() if info['required']])
        compliance_score = (required_mapped / required_total) * 100

        results = {
            'mappings': mappings,
            'method': 'Hybrid',
            'compliance_score': compliance_score,
            'required_mapped': required_mapped,
            'required_total': required_total,
            'total_mapped': len(mappings),
            'timestamp': datetime.now()
        }

        return results

    def _display_mapping_results(self, results: Dict, container_key: str):
        """Display mapping results with download options"""
        st.markdown("---")
        st.markdown("### 🎯 Mapping Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            compliance_color = ("green" if results['compliance_score'] >= 85 else
                                "orange" if results['compliance_score'] >= 70 else "red")
            st.metric(
                "CBUAE Compliance",
                f"{results['compliance_score']:.1f}%",
                delta=f"{results['required_mapped']}/{results['required_total']} required"
            )

        with col2:
            st.metric("Total Mapped", results['total_mapped'])

        with col3:
            high_conf = len([m for m in results['mappings'] if m.confidence_level == MappingConfidence.HIGH])
            st.metric("High Confidence", high_conf)

        with col4:
            st.metric("Method", results['method'])

        # Compliance status
        if results['compliance_score'] >= 85:
            st.success("🎉 Excellent CBUAE compliance!")
        elif results['compliance_score'] >= 70:
            st.warning("⚠️ Good compliance - minor improvements needed")
        else:
            st.error("❌ Below compliance threshold - additional mapping required")

        # Detailed mapping table
        if results['mappings']:
            st.markdown("#### 📋 Detailed Mapping Results")

            mapping_data = []
            for mapping in results['mappings']:
                mapping_data.append({
                    'Source Field': mapping.source_field,
                    'Target Field': mapping.target_field,
                    'Confidence': f"{mapping.confidence_score:.3f}",
                    'Level': mapping.confidence_level.value.title(),
                    'Method': mapping.mapping_method,
                    'Required': "✅" if mapping.is_required else "⚪",
                })

            df = pd.DataFrame(mapping_data)

            # Apply styling
            def style_confidence(row):
                if row['Level'] == 'High':
                    return ['background-color: #d4edda'] * len(row)
                elif row['Level'] == 'Medium':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)

            styled_df = df.style.apply(style_confidence, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            # Download options
            col1, col2 = st.columns(2)

            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"mapping_results_{container_key}.csv",
                    "text/csv",
                    key=f"{container_key}_download_csv"
                )

            with col2:
                json_data = {
                    'summary': {
                        'compliance_score': results['compliance_score'],
                        'method': results['method'],
                        'timestamp': results['timestamp'].isoformat()
                    },
                    'mappings': [
                        {
                            'source': m.source_field,
                            'target': m.target_field,
                            'confidence': m.confidence_score,
                            'required': m.is_required
                        } for m in results['mappings']
                    ]
                }

                st.download_button(
                    "📄 Download JSON",
                    json.dumps(json_data, indent=2),
                    f"mapping_results_{container_key}.json",
                    "application/json",
                    key=f"{container_key}_download_json"
                )


# Usage example for integration into main app
def integrate_mapping_component(data: pd.DataFrame, session_key: str = "main_mapping"):
    """
    Integration function for main Streamlit app

    Usage in main app:
    ```python
    from data_mapping_ui import integrate_mapping_component

    # In your main app
    if uploaded_data is not None:
        mapping_results = integrate_mapping_component(uploaded_data, "app_mapping")
        if mapping_results:
            st.write("Mapping completed!")
    ```
    """
    mapping_ui = DataMappingUI()
    return mapping_ui.render_mapping_interface(data, session_key)


if __name__ == "__main__":
    # Demo/testing interface
    st.title("🗺️ Data Mapping UI Component Demo")

    # Create sample data for testing
    sample_data = pd.DataFrame({
        'customer_id': {'required': True, 'description': 'Unique customer identifier'},
            'account_id': {'required': True, 'description': 'Unique account identifier'},
            'account_type': {'required': True, 'description': 'Type of account (CURRENT, SAVINGS, etc.)'},
            'account_status': {'required': True, 'description': 'Account status (ACTIVE, DORMANT, CLOSED)'},
            'dormancy_status': {'required': True, 'description': 'Dormancy classification'},
            'balance_current': {'required': True, 'description': 'Current account balance'},
            'last_transaction_date': {'required': True, 'description': 'Date of last transaction'},

            # Customer Information (High Priority)
            'customer_type': {'required': False, 'description': 'Type of customer (INDIVIDUAL, CORPORATE)'},
            'full_name_en': {'required': False, 'description': 'Customer full name in English'},
            'full_name_ar': {'required': False, 'description': 'Customer full name in Arabic'},
            'id_number': {'required': False, 'description': 'Customer ID number'},
            'id_type': {'required': False, 'description': 'Type of ID (EMIRATES_ID, PASSPORT, etc.)'},
            'date_of_birth': {'required': False, 'description': 'Customer date of birth'},
            'nationality': {'required': False, 'description': 'Customer nationality'},

            # Contact Information
            'phone_primary': {'required': False, 'description': 'Primary phone number'},
            'phone_secondary': {'required': False, 'description': 'Secondary phone number'},
            'email_primary': {'required': False, 'description': 'Primary email address'},
            'email_secondary': {'required': False, 'description': 'Secondary email address'},
            'address_line1': {'required': False, 'description': 'Primary address line'},
            'address_line2': {'required': False, 'description': 'Secondary address line'},
            'city': {'required': False, 'description': 'City'},
            'emirate': {'required': False, 'description': 'Emirate'},
            'country': {'required': False, 'description': 'Country'},
            'postal_code': {'required': False, 'description': 'Postal code'},
            'address_known': {'required': False, 'description': 'Whether customer address is known (YES/NO)'},
            'last_contact_date': {'required': False, 'description': 'Date of last contact with customer'},
            'last_contact_method': {'required': False, 'description': 'Method of last contact (EMAIL, PHONE, etc.)'},

            # KYC and Risk Management
            'kyc_status': {'required': False, 'description': 'Know Your Customer status'},
            'kyc_expiry_date': {'required': False, 'description': 'KYC expiry date'},
            'risk_rating': {'required': False, 'description': 'Customer risk rating (LOW, MEDIUM, HIGH)'},

            # Account Details
            'account_subtype': {'required': False, 'description': 'Account subtype classification'},
            'account_name': {'required': False, 'description': 'Account name or description'},
            'currency': {'required': False, 'description': 'Account currency'},
            'opening_date': {'required': False, 'description': 'Account opening date'},
            'closing_date': {'required': False, 'description': 'Account closing date (if applicable)'},
            'last_system_transaction_date': {'required': False, 'description': 'Date of last system transaction'},

            # Balance and Financial Information
            'balance_available': {'required': False, 'description': 'Available balance'},
            'balance_minimum': {'required': False, 'description': 'Minimum balance requirement'},
            'interest_rate': {'required': False, 'description': 'Interest rate'},
            'interest_accrued': {'required': False, 'description': 'Accrued interest amount'},

            # Account Features
            'is_joint_account': {'required': False, 'description': 'Whether account is joint (YES/NO)'},
            'joint_account_holders': {'required': False, 'description': 'Number of joint account holders'},
            'has_outstanding_facilities': {'required': False, 'description': 'Whether customer has outstanding facilities (YES/NO)'},
            'maturity_date': {'required': False, 'description': 'Account maturity date (for term deposits)'},
            'auto_renewal': {'required': False, 'description': 'Auto-renewal setting (YES/NO)'},

            # Statement and Communication
            'last_statement_date': {'required': False, 'description': 'Date of last statement'},
            'statement_frequency': {'required': False, 'description': 'Statement frequency (MONTHLY, QUARTERLY, etc.)'},

            # Dormancy Tracking (Critical for Compliance)
            'tracking_id': {'required': False, 'description': 'Internal dormancy tracking identifier'},
            'dormancy_trigger_date': {'required': False, 'description': 'Date when dormancy was triggered'},
            'dormancy_period_start': {'required': False, 'description': 'Start date of dormancy period'},
            'dormancy_period_months': {'required': False, 'description': 'Number of months since dormancy trigger'},
            'dormancy_classification_date': {'required': False, 'description': 'Date when dormancy was classified'},
            'transfer_eligibility_date': {'required': False, 'description': 'Date when account becomes eligible for CB transfer'},

            # Process Management
            'current_stage': {'required': False, 'description': 'Current stage in dormancy process'},
            'contact_attempts_made': {'required': False, 'description': 'Number of contact attempts made'},
            'last_contact_attempt_date': {'required': False, 'description': 'Date of last contact attempt'},
            'waiting_period_start': {'required': False, 'description': 'Start date of waiting period'},
            'waiting_period_end': {'required': False, 'description': 'End date of waiting period'},

            # Transfer Information
            'transferred_to_ledger_date': {'required': False, 'description': 'Date transferred to internal ledger'},
            'transferred_to_cb_date': {'required': False, 'description': 'Date transferred to Central Bank'},
            'cb_transfer_amount': {'required': False, 'description': 'Amount transferred to Central Bank'},
            'cb_transfer_reference': {'required': False, 'description': 'Central Bank transfer reference'},
            'exclusion_reason': {'required': False, 'description': 'Reason for exclusion from dormancy process'},

            # System Fields
            'created_date': {'required': False, 'description': 'Record creation date'},
            'updated_date': {'required': False, 'description': 'Record last update date'},
            'updated_by': {'required': False, 'description': 'User who last updated the record'}

    })

    st.info("📊 Demo with sample banking data")
    st.dataframe(sample_data)

    # Use the mapping component
    mapping_ui = DataMappingUI()
    results = mapping_ui.render_mapping_interface(sample_data, "demo")