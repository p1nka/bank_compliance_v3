from enum import Enum
import os
import csv
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from llama_cpp import Llama

# Initialize LLM
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)


# Constants and Enums
class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL_COMPLIANT = "partial_compliant"
    PENDING_REVIEW = "pending_review"
    CRITICAL_VIOLATION = "critical_violation"


class ViolationType(Enum):
    ARTICLE_2_VIOLATION = "article_2_violation"
    ARTICLE_3_1_VIOLATION = "article_3_1_violation"
    ARTICLE_3_4_VIOLATION = "article_3_4_violation"
    CONTACT_VIOLATION = "contact_violation"
    TRANSFER_VIOLATION = "transfer_violation"
    DOCUMENTATION_VIOLATION = "documentation_violation"
    TIMELINE_VIOLATION = "timeline_violation"
    AMOUNT_VIOLATION = "amount_violation"
    REPORTING_VIOLATION = "reporting_violation"


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CBUAEArticle(Enum):
    ARTICLE_2 = "article_2"
    ARTICLE_3_1 = "article_3_1"
    ARTICLE_3_4 = "article_3_4"
    ARTICLE_4 = "article_4"
    ARTICLE_5 = "article_5"


# Base Compliance Agent Class
class ComplianceAgent:
    def __init__(self):
        self.llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)

    def get_llm_recommendation(self, context: str) -> str:
        try:
            prompt = f"""As a CBUAE banking compliance expert, analyze the following compliance issue and provide specific regulatory recommendations:

Context: {context}

Provide:
1. Immediate actions required
2. Regulatory citation
3. Timeline for remediation
4. Risk assessment
5. Preventive measures

Response:"""

            response = self.llm(prompt, max_tokens=200)
            return response.get('choices', [{}])[0].get('text', 'No recommendation available').strip()
        except Exception as e:
            print(f"Error getting LLM recommendation: {e}")
            return "Recommendation unavailable due to system error"

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        raise NotImplementedError


# Agent 1: Article 2 Compliance - Dormant Account Detection
class Article2ComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'Article2Compliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Article 2 - Dormant Account Identification'
        }

        # Check dormancy classification compliance
        dormancy_status = account_data.get('dormancy_status', '')
        last_transaction_date = account_data.get('last_transaction_date')
        dormancy_trigger_date = account_data.get('dormancy_trigger_date')

        if last_transaction_date:
            days_inactive = (datetime.now() - datetime.fromisoformat(str(last_transaction_date))).days

            # Violation: Account should be dormant but not classified
            if days_inactive >= 365 and dormancy_status != 'dormant':
                result['violations'].append("Account meets dormancy criteria but not classified as dormant")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value
                result['risk_level'] = RiskLevel.HIGH.value

            # Violation: Missing dormancy trigger date
            if dormancy_status == 'dormant' and not dormancy_trigger_date:
                result['violations'].append("Dormant account missing trigger date")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value
                result['risk_level'] = RiskLevel.MEDIUM.value

        if result['violations']:
            result['action'] = "Update dormancy classification and trigger dates"
            context = f"Article 2 violations found: {result['violations']}. Account: {account_data.get('account_id', 'unknown')}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 2: Article 3.1 Compliance - Customer Contact Process
class Article31ComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'Article31Compliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Article 3.1 - Customer Contact Requirements'
        }

        customer_type = account_data.get('customer_type', 'individual')
        account_value = account_data.get('balance_current', 0)
        contact_attempts = account_data.get('contact_attempts_made', 0)
        dormancy_status = account_data.get('dormancy_status', '')

        if dormancy_status == 'dormant':
            # Determine required contact attempts
            required_attempts = 3  # Default
            if customer_type == 'individual' and account_value >= 25000:
                required_attempts = 5
            elif customer_type == 'corporate':
                required_attempts = 5 if account_value >= 100000 else 4

            # Check compliance
            if contact_attempts < required_attempts:
                result['violations'].append(
                    f"Insufficient contact attempts: {contact_attempts} of {required_attempts} required")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value
                result['risk_level'] = RiskLevel.HIGH.value
                result['action'] = f"Complete {required_attempts - contact_attempts} additional contact attempts"

            # Check contact method diversity
            last_contact_method = account_data.get('last_contact_method', '')
            if contact_attempts > 0 and len(last_contact_method.split(',')) < 2:
                result['violations'].append("Contact attempts must use diverse methods (email, phone, letter)")
                result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value

        if result['violations']:
            context = f"Article 3.1 violations: {result['violations']}. Customer type: {customer_type}, Value: {account_value}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 3: Article 3.4 Compliance - Central Bank Transfer
class Article34ComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'Article34Compliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Article 3.4 - Central Bank Transfer Requirements'
        }

        dormancy_trigger_date = account_data.get('dormancy_trigger_date')
        transfer_eligibility_date = account_data.get('transfer_eligibility_date')
        transferred_to_cb_date = account_data.get('transferred_to_cb_date')
        contact_attempts = account_data.get('contact_attempts_made', 0)

        if dormancy_trigger_date:
            dormancy_days = (datetime.now() - datetime.fromisoformat(str(dormancy_trigger_date))).days

            # Check if eligible for CB transfer (2+ years dormant + contact completed)
            if dormancy_days >= 730 and contact_attempts >= 3:
                if not transfer_eligibility_date:
                    result['violations'].append("Account eligible for CB transfer but eligibility date not set")
                    result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                    result['priority'] = Priority.HIGH.value
                    result['risk_level'] = RiskLevel.HIGH.value

                elif transfer_eligibility_date and not transferred_to_cb_date:
                    eligibility_days = (datetime.now() - datetime.fromisoformat(str(transfer_eligibility_date))).days
                    if eligibility_days > 90:  # 3 months overdue
                        result['violations'].append(f"CB transfer overdue by {eligibility_days - 90} days")
                        result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
                        result['priority'] = Priority.CRITICAL.value
                        result['risk_level'] = RiskLevel.CRITICAL.value
                        result['action'] = "Immediately initiate CB transfer process"

        if result['violations']:
            context = f"Article 3.4 violations: {result['violations']}. Dormancy days: {dormancy_days if 'dormancy_days' in locals() else 'unknown'}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 4: Contact Verification Compliance
class ContactVerificationAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'ContactVerification',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Contact Verification Standards'
        }

        # Verify contact information completeness
        required_fields = ['phone_primary', 'email_primary', 'address_line1']
        missing_contact_info = []

        for field in required_fields:
            if not account_data.get(field):
                missing_contact_info.append(field)

        if missing_contact_info:
            result['violations'].append(f"Missing required contact information: {missing_contact_info}")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value
            result['risk_level'] = RiskLevel.MEDIUM.value

        # Verify contact attempt documentation
        contact_attempts = account_data.get('contact_attempts_made', 0)
        last_contact_attempt_date = account_data.get('last_contact_attempt_date')

        if contact_attempts > 0 and not last_contact_attempt_date:
            result['violations'].append("Contact attempts recorded but missing attempt dates")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value

        # Check address verification
        address_known = account_data.get('address_known', True)
        if not address_known and contact_attempts < 5:
            result['violations'].append("Unknown address requires enhanced contact efforts")
            result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
            result['priority'] = Priority.HIGH.value
            result['action'] = "Implement address verification procedures"

        if result['violations']:
            context = f"Contact verification issues: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 5: Transfer Eligibility Compliance
class TransferEligibilityComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'TransferEligibilityCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Transfer Eligibility Requirements'
        }

        # Check transfer eligibility criteria
        has_outstanding_facilities = account_data.get('has_outstanding_facilities', False)
        cb_transfer_amount = account_data.get('cb_transfer_amount')
        balance_current = account_data.get('balance_current', 0)
        exclusion_reason = account_data.get('exclusion_reason')

        # Accounts with outstanding facilities should be excluded
        if has_outstanding_facilities and not exclusion_reason:
            result['violations'].append("Account has outstanding facilities but no exclusion reason documented")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.HIGH.value
            result['risk_level'] = RiskLevel.HIGH.value

        # Transfer amount should match current balance (minus fees)
        if cb_transfer_amount and balance_current:
            if abs(cb_transfer_amount - balance_current) > (balance_current * 0.02):  # 2% tolerance
                result['violations'].append("CB transfer amount does not match account balance")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value

        # Check minimum transfer threshold
        if balance_current > 0 and balance_current < 100:  # Minimum threshold 100 AED
            if not exclusion_reason:
                result['violations'].append("Below minimum transfer threshold but no exclusion documented")
                result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
                result['priority'] = Priority.LOW.value

        if result['violations']:
            result['action'] = "Review and correct transfer eligibility documentation"
            context = f"Transfer eligibility issues: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 6: FX Conversion Compliance
class FXConversionComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'FXConversionCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Foreign Exchange Conversion Standards'
        }

        currency = account_data.get('currency', 'AED')
        balance_current = account_data.get('balance_current', 0)
        cb_transfer_amount = account_data.get('cb_transfer_amount')

        # Check FX accounts compliance
        if currency != 'AED':
            # FX conversion rate documentation required
            if cb_transfer_amount and not account_data.get('fx_conversion_rate'):
                result['violations'].append("Foreign currency account missing conversion rate documentation")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value
                result['risk_level'] = RiskLevel.MEDIUM.value

            # FX conversion date required
            if cb_transfer_amount and not account_data.get('fx_conversion_date'):
                result['violations'].append("Missing FX conversion date for foreign currency transfer")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value

            # Check if conversion follows CBUAE rates
            if balance_current > 0:
                result['action'] = "Verify FX conversion uses official CBUAE rates"

        if result['violations']:
            context = f"FX conversion compliance issues: {result['violations']}. Currency: {currency}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 7: Process Management Compliance
class ProcessManagementComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'ProcessManagementCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Process Management Standards'
        }

        current_stage = account_data.get('current_stage', '')
        updated_by = account_data.get('updated_by', '')
        updated_date = account_data.get('updated_date')

        # Check process stage progression
        valid_stages = ['initial', 'contact_phase', 'waiting_period', 'transfer_eligible', 'transferred', 'completed']
        if current_stage and current_stage not in valid_stages:
            result['violations'].append(f"Invalid process stage: {current_stage}")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value

        # Check update tracking
        if not updated_by:
            result['violations'].append("Missing user tracking for account updates")
            result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
            result['priority'] = Priority.LOW.value

        if not updated_date:
            result['violations'].append("Missing timestamp for last update")
            result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
            result['priority'] = Priority.LOW.value

        # Check for stale records (no updates in 90 days for active dormant accounts)
        if updated_date and current_stage in ['contact_phase', 'waiting_period']:
            days_since_update = (datetime.now() - datetime.fromisoformat(str(updated_date))).days
            if days_since_update > 90:
                result['violations'].append(f"Process stalled: No updates for {days_since_update} days")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value
                result['risk_level'] = RiskLevel.HIGH.value
                result['action'] = "Review and update account status"

        if result['violations']:
            context = f"Process management issues: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 8: Documentation Compliance
class DocumentationComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'DocumentationCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Documentation Requirements'
        }

        # Check required documentation fields
        required_docs = {
            'tracking_id': 'Unique tracking identifier',
            'created_date': 'Account creation timestamp',
            'dormancy_classification_date': 'Dormancy classification date',
            'last_statement_date': 'Last statement generation date'
        }

        missing_docs = []
        for field, description in required_docs.items():
            if not account_data.get(field):
                missing_docs.append(description)

        if missing_docs:
            result['violations'].append(f"Missing required documentation: {missing_docs}")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value
            result['risk_level'] = RiskLevel.MEDIUM.value

        # Check statement frequency compliance
        statement_frequency = account_data.get('statement_frequency', '')
        last_statement_date = account_data.get('last_statement_date')

        if statement_frequency and last_statement_date:
            freq_days = {'monthly': 30, 'quarterly': 90, 'annual': 365}.get(statement_frequency, 30)
            days_since_statement = (datetime.now() - datetime.fromisoformat(str(last_statement_date))).days

            if days_since_statement > freq_days * 1.2:  # 20% tolerance
                result['violations'].append(f"Statement overdue by {days_since_statement - freq_days} days")
                result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
                result['priority'] = Priority.LOW.value

        if result['violations']:
            result['action'] = "Complete missing documentation requirements"
            context = f"Documentation compliance issues: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 9: Timeline Compliance
class TimelineComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'TimelineCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Timeline Requirements'
        }

        # Check key timeline milestones
        dormancy_trigger_date = account_data.get('dormancy_trigger_date')
        waiting_period_start = account_data.get('waiting_period_start')
        waiting_period_end = account_data.get('waiting_period_end')
        transfer_eligibility_date = account_data.get('transfer_eligibility_date')

        if dormancy_trigger_date:
            trigger_date = datetime.fromisoformat(str(dormancy_trigger_date))

            # Check 3-month waiting period compliance
            if waiting_period_start:
                waiting_start = datetime.fromisoformat(str(waiting_period_start))
                if (waiting_start - trigger_date).days > 30:  # Should start within 30 days
                    result['violations'].append("Waiting period started too late after dormancy trigger")
                    result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                    result['priority'] = Priority.MEDIUM.value

            # Check waiting period duration
            if waiting_period_start and waiting_period_end:
                waiting_duration = (datetime.fromisoformat(str(waiting_period_end)) -
                                    datetime.fromisoformat(str(waiting_period_start))).days
                if waiting_duration < 90:
                    result['violations'].append(f"Waiting period too short: {waiting_duration} days (minimum 90)")
                    result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
                    result['priority'] = Priority.CRITICAL.value
                    result['risk_level'] = RiskLevel.CRITICAL.value

            # Check transfer eligibility timing
            if transfer_eligibility_date:
                eligibility_date = datetime.fromisoformat(str(transfer_eligibility_date))
                dormancy_duration = (eligibility_date - trigger_date).days
                if dormancy_duration < 730:  # Minimum 2 years
                    result['violations'].append(f"Transfer eligibility set too early: {dormancy_duration} days")
                    result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
                    result['priority'] = Priority.CRITICAL.value
                    result['risk_level'] = RiskLevel.CRITICAL.value

        if result['violations']:
            result['action'] = "Correct timeline violations and update process dates"
            context = f"Timeline compliance violations: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 10: Amount Verification Compliance
class AmountVerificationComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'AmountVerificationCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Amount Verification Standards'
        }

        balance_current = account_data.get('balance_current', 0)
        balance_available = account_data.get('balance_available', 0)
        balance_minimum = account_data.get('balance_minimum', 0)
        interest_accrued = account_data.get('interest_accrued', 0)

        # Verify balance consistency
        if balance_available > balance_current:
            result['violations'].append("Available balance exceeds current balance")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.HIGH.value
            result['risk_level'] = RiskLevel.HIGH.value

        # Check minimum balance compliance
        if balance_minimum > 0 and balance_current < balance_minimum:
            result['violations'].append(
                f"Current balance below minimum requirement: {balance_current} < {balance_minimum}")
            result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value

        # Verify interest calculation
        interest_rate = account_data.get('interest_rate', 0)
        if interest_rate > 0 and balance_current > 0 and interest_accrued == 0:
            result['violations'].append("Interest-bearing account with zero accrued interest")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value

        # Check for negative balances
        if balance_current < 0:
            result['violations'].append("Account has negative balance")
            result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
            result['priority'] = Priority.CRITICAL.value
            result['risk_level'] = RiskLevel.CRITICAL.value

        if result['violations']:
            result['action'] = "Investigate and correct balance discrepancies"
            context = f"Amount verification issues: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 11: Claims Detection Compliance
class ClaimsDetectionComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'ClaimsDetectionCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Claims Detection Requirements'
        }

        has_outstanding_facilities = account_data.get('has_outstanding_facilities', False)
        is_joint_account = account_data.get('is_joint_account', False)
        joint_account_holders = account_data.get('joint_account_holders', '')
        exclusion_reason = account_data.get('exclusion_reason', '')

        # Check for outstanding facilities documentation
        if has_outstanding_facilities:
            if not exclusion_reason:
                result['violations'].append("Outstanding facilities detected but no exclusion reason provided")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value
                result['risk_level'] = RiskLevel.HIGH.value

        # Check joint account holder verification
        if is_joint_account:
            if not joint_account_holders:
                result['violations'].append("Joint account missing holder information")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value

            # Joint accounts require verification of all holders
            if joint_account_holders and len(joint_account_holders.split(',')) < 2:
                result['violations'].append("Joint account must have multiple verified holders")
                result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value

        # Check for legal claims indicators
        legal_indicators = ['litigation', 'garnishment', 'lien', 'court_order']
        for indicator in legal_indicators:
            if account_data.get(indicator, False):
                if not exclusion_reason or indicator not in exclusion_reason.lower():
                    result['violations'].append(
                        f"Legal claim indicator ({indicator}) not documented in exclusion reason")
                    result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                    result['priority'] = Priority.HIGH.value
                    result['risk_level'] = RiskLevel.HIGH.value

        if result['violations']:
            result['action'] = "Document and verify all outstanding claims"
            context = f"Claims detection issues: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 12: Flag Instructions Compliance
class FlagInstructionsComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'FlagInstructionsCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Flag Instructions Standards'
        }

        # Check for special handling flags
        special_flags = {
            'vip_customer': 'VIP customer requiring special handling',
            'regulatory_watch': 'Account under regulatory monitoring',
            'legal_hold': 'Account subject to legal proceedings',
            'deceased_customer': 'Deceased customer account',
            'dispute_pending': 'Customer dispute in progress'
        }

        active_flags = []
        for flag, description in special_flags.items():
            if account_data.get(flag, False):
                active_flags.append(flag)

        # Verify flag documentation
        if active_flags:
            flag_instructions = account_data.get('flag_instructions', '')
            if not flag_instructions:
                result['violations'].append(f"Active flags ({active_flags}) missing handling instructions")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value
                result['risk_level'] = RiskLevel.HIGH.value

        # Check deceased customer handling
        if account_data.get('deceased_customer', False):
            if not account_data.get('death_certificate_received', False):
                result['violations'].append("Deceased customer flag without death certificate verification")
                result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
                result['priority'] = Priority.CRITICAL.value
                result['risk_level'] = RiskLevel.CRITICAL.value

        # Check regulatory compliance for flagged accounts
        if account_data.get('regulatory_watch', False):
            if not account_data.get('compliance_officer_assigned', False):
                result['violations'].append("Regulatory watch account missing compliance officer assignment")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value

        if result['violations']:
            result['action'] = "Update flag instructions and verify special handling procedures"
            context = f"Flag instructions violations: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 13: Risk Assessment Compliance
class RiskAssessmentComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'RiskAssessmentCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Risk Assessment Requirements'
        }

        risk_rating = account_data.get('risk_rating', '')
        balance_current = account_data.get('balance_current', 0)
        nationality = account_data.get('nationality', '')
        kyc_status = account_data.get('kyc_status', '')
        kyc_expiry_date = account_data.get('kyc_expiry_date')

        # Check risk rating assignment
        if not risk_rating:
            result['violations'].append("Missing risk rating assignment")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.HIGH.value
            result['risk_level'] = RiskLevel.HIGH.value

        # Verify high-value account risk assessment
        if balance_current >= 100000 and risk_rating not in ['high', 'very_high']:
            result['violations'].append("High-value account with inadequate risk rating")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.HIGH.value
            result['risk_level'] = RiskLevel.HIGH.value

        # Check KYC compliance
        if kyc_status != 'compliant':
            result['violations'].append(f"Non-compliant KYC status: {kyc_status}")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.HIGH.value
            result['risk_level'] = RiskLevel.HIGH.value

        # Check KYC expiry
        if kyc_expiry_date:
            if datetime.fromisoformat(str(kyc_expiry_date)) < datetime.now():
                result['violations'].append("Expired KYC documentation")
                result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
                result['priority'] = Priority.CRITICAL.value
                result['risk_level'] = RiskLevel.CRITICAL.value

        # Check nationality-based risk factors
        high_risk_countries = ['XX', 'YY', 'ZZ']  # Placeholder for actual high-risk countries
        if nationality in high_risk_countries and risk_rating != 'high':
            result['violations'].append("High-risk nationality with inadequate risk rating")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.HIGH.value

        if result['violations']:
            result['action'] = "Update risk assessment and KYC documentation"
            context = f"Risk assessment violations: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 14: Regulatory Reporting Compliance
class RegulatoryReportingComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'RegulatoryReportingCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Regulatory Reporting Requirements'
        }

        # Check reporting data completeness
        required_reporting_fields = [
            'customer_id', 'account_id', 'balance_current', 'dormancy_status',
            'last_transaction_date', 'customer_type', 'nationality'
        ]

        missing_fields = []
        for field in required_reporting_fields:
            if not account_data.get(field):
                missing_fields.append(field)

        if missing_fields:
            result['violations'].append(f"Missing required reporting fields: {missing_fields}")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value
            result['risk_level'] = RiskLevel.MEDIUM.value

        # Check data quality for reporting
        customer_id = account_data.get('customer_id', '')
        account_id = account_data.get('account_id', '')

        if customer_id and len(customer_id) < 5:
            result['violations'].append("Customer ID format invalid for regulatory reporting")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value

        if account_id and len(account_id) < 8:
            result['violations'].append("Account ID format invalid for regulatory reporting")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value

        # Check reportable threshold compliance
        balance_current = account_data.get('balance_current', 0)
        if balance_current >= 3000:  # CBUAE reporting threshold
            if not account_data.get('reportable_to_cbuae', True):
                result['violations'].append("Account above reporting threshold but marked as non-reportable")
                result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
                result['priority'] = Priority.CRITICAL.value
                result['risk_level'] = RiskLevel.CRITICAL.value

        if result['violations']:
            result['action'] = "Correct data quality issues for regulatory reporting"
            context = f"Regulatory reporting violations: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 15: Audit Trail Compliance
class AuditTrailComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'AuditTrailCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Audit Trail Requirements'
        }

        # Check audit trail completeness
        audit_fields = ['created_date', 'updated_date', 'updated_by']
        missing_audit_fields = []

        for field in audit_fields:
            if not account_data.get(field):
                missing_audit_fields.append(field)

        if missing_audit_fields:
            result['violations'].append(f"Missing audit trail fields: {missing_audit_fields}")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value

        # Check change tracking
        if account_data.get('dormancy_status') == 'dormant':
            if not account_data.get('dormancy_classification_date'):
                result['violations'].append("Dormancy status change not properly tracked")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value
                result['risk_level'] = RiskLevel.HIGH.value

        # Check user authorization tracking
        updated_by = account_data.get('updated_by', '')
        if updated_by and not updated_by.startswith(('USER_', 'SYS_', 'ADMIN_')):
            result['violations'].append("Invalid user identifier format in audit trail")
            result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
            result['priority'] = Priority.LOW.value

        # Check timestamp consistency
        created_date = account_data.get('created_date')
        updated_date = account_data.get('updated_date')

        if created_date and updated_date:
            if datetime.fromisoformat(str(updated_date)) < datetime.fromisoformat(str(created_date)):
                result['violations'].append("Update timestamp earlier than creation timestamp")
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value

        if result['violations']:
            result['action'] = "Implement proper audit trail logging"
            context = f"Audit trail violations: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 16: Action Generation Compliance
class ActionGenerationComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'ActionGenerationCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Action Generation Standards'
        }

        current_stage = account_data.get('current_stage', '')
        dormancy_status = account_data.get('dormancy_status', '')
        contact_attempts = account_data.get('contact_attempts_made', 0)

        # Generate required actions based on current state
        required_actions = []

        if dormancy_status == 'dormant':
            if contact_attempts < 3:
                required_actions.append("CONTACT_CUSTOMER")

            if current_stage == 'contact_phase' and contact_attempts >= 3:
                required_actions.append("START_WAITING_PERIOD")

            dormancy_trigger_date = account_data.get('dormancy_trigger_date')
            if dormancy_trigger_date:
                dormancy_days = (datetime.now() - datetime.fromisoformat(str(dormancy_trigger_date))).days
                if dormancy_days >= 730 and contact_attempts >= 3:
                    required_actions.append("INITIATE_CB_TRANSFER")

        # Check if required actions are documented
        documented_actions = account_data.get('required_actions', '').split(',')
        documented_actions = [action.strip() for action in documented_actions if action.strip()]

        missing_actions = [action for action in required_actions if action not in documented_actions]

        if missing_actions:
            result['violations'].append(f"Missing required actions: {missing_actions}")
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.HIGH.value
            result['risk_level'] = RiskLevel.HIGH.value
            result['action'] = f"Document required actions: {missing_actions}"

        # Check for contradictory actions
        if 'INITIATE_CB_TRANSFER' in documented_actions and contact_attempts < 3:
            result['violations'].append("CB transfer action without completing required contact attempts")
            result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
            result['priority'] = Priority.CRITICAL.value
            result['risk_level'] = RiskLevel.CRITICAL.value

        if result['violations']:
            context = f"Action generation violations: {result['violations']}"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Agent 17: Final Verification Compliance
class FinalVerificationComplianceAgent(ComplianceAgent):
    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'FinalVerificationCompliance',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Final Verification Standards',
            'overall_compliance_score': 0
        }

        # Compile all violations from dormancy results if available
        all_violations = []
        compliance_scores = []

        if dormancy_results:
            for agent_name, agent_result in dormancy_results.items():
                if isinstance(agent_result, dict) and 'violations' in agent_result:
                    all_violations.extend(agent_result.get('violations', []))

                # Calculate compliance score
                violations_count = len(agent_result.get('violations', []))
                if violations_count == 0:
                    compliance_scores.append(100)
                elif violations_count <= 2:
                    compliance_scores.append(80)
                elif violations_count <= 5:
                    compliance_scores.append(60)
                else:
                    compliance_scores.append(40)

        # Calculate overall compliance score
        if compliance_scores:
            result['overall_compliance_score'] = sum(compliance_scores) / len(compliance_scores)
        else:
            result['overall_compliance_score'] = 100

        # Final verification checks
        critical_fields = [
            'customer_id', 'account_id', 'account_type', 'balance_current',
            'dormancy_status', 'last_transaction_date'
        ]

        critical_missing = []
        for field in critical_fields:
            if not account_data.get(field):
                critical_missing.append(field)

        if critical_missing:
            result['violations'].append(f"Critical fields missing: {critical_missing}")
            result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
            result['priority'] = Priority.CRITICAL.value
            result['risk_level'] = RiskLevel.CRITICAL.value

        # Overall compliance assessment
        if result['overall_compliance_score'] >= 95:
            result['compliance_status'] = ComplianceStatus.COMPLIANT.value
        elif result['overall_compliance_score'] >= 80:
            result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value
        elif result['overall_compliance_score'] >= 60:
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.HIGH.value
            result['risk_level'] = RiskLevel.HIGH.value
        else:
            result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
            result['priority'] = Priority.CRITICAL.value
            result['risk_level'] = RiskLevel.CRITICAL.value

        # Final action determination
        if result['overall_compliance_score'] < 100:
            result['action'] = "Address all compliance violations before proceeding"

        if result['violations'] or all_violations:
            context = f"Final verification found {len(all_violations)} total violations. Overall score: {result['overall_compliance_score']:.1f}%"
            result['recommendation'] = self.get_llm_recommendation(context)

        return result


# Compliance Orchestrator
class ComplianceOrchestrator:
    def __init__(self):
        self.agents = {
            'article_2_compliance': Article2ComplianceAgent(),
            'article_3_1_compliance': Article31ComplianceAgent(),
            'article_3_4_compliance': Article34ComplianceAgent(),
            'contact_verification': ContactVerificationAgent(),
            'transfer_eligibility_compliance': TransferEligibilityComplianceAgent(),
            'fx_conversion_compliance': FXConversionComplianceAgent(),
            'process_management_compliance': ProcessManagementComplianceAgent(),
            'documentation_compliance': DocumentationComplianceAgent(),
            'timeline_compliance': TimelineComplianceAgent(),
            'amount_verification_compliance': AmountVerificationComplianceAgent(),
            'claims_detection_compliance': ClaimsDetectionComplianceAgent(),
            'flag_instructions_compliance': FlagInstructionsComplianceAgent(),
            'risk_assessment_compliance': RiskAssessmentComplianceAgent(),
            'regulatory_reporting_compliance': RegulatoryReportingComplianceAgent(),
            'audit_trail_compliance': AuditTrailComplianceAgent(),
            'action_generation_compliance': ActionGenerationComplianceAgent(),
            'final_verification_compliance': FinalVerificationComplianceAgent()
        }

    def process_account(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        """Process account through all compliance agents"""
        results = {}

        # Run all compliance agents
        for agent_name, agent in self.agents.items():
            try:
                results[agent_name] = agent.execute(account_data, dormancy_results)
            except Exception as e:
                results[agent_name] = {
                    'agent': agent_name,
                    'error': str(e),
                    'compliance_status': ComplianceStatus.PENDING_REVIEW.value
                }

        return results

    def export_to_csv(self, account_data: Dict, results: Dict[str, Dict],
                      filename: str = "compliance_results.csv") -> None:
        """Export compliance results to CSV"""
        rows = []

        # Add account metadata to every agent's result
        for agent_name, agent_result in results.items():
            row = {
                "account_id": account_data.get("account_id", "N/A"),
                "customer_id": account_data.get("customer_id", "N/A"),
                "account_type": account_data.get("account_type", "N/A"),
                "balance_current": account_data.get("balance_current", "N/A"),
                "dormancy_status": account_data.get("dormancy_status", "N/A"),
                "agent": agent_name,
            }
            row.update(agent_result)
            rows.append(row)

        # Write to CSV
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            if rows:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"Compliance results exported to {filename}")

    def generate_compliance_summary(self, results: Dict[str, Dict]) -> Dict:
        """Generate overall compliance summary"""
        summary = {
            'total_agents': len(results),
            'compliant_agents': 0,
            'non_compliant_agents': 0,
            'critical_violations': 0,
            'total_violations': 0,
            'overall_status': ComplianceStatus.COMPLIANT.value,
            'critical_actions': [],
            'recommendations': []
        }

        for agent_name, agent_result in results.items():
            if agent_result.get('compliance_status') == ComplianceStatus.COMPLIANT.value:
                summary['compliant_agents'] += 1
            else:
                summary['non_compliant_agents'] += 1

            if agent_result.get('compliance_status') == ComplianceStatus.CRITICAL_VIOLATION.value:
                summary['critical_violations'] += 1

            violations = agent_result.get('violations', [])
            summary['total_violations'] += len(violations)

            if agent_result.get('priority') == Priority.CRITICAL.value:
                summary['critical_actions'].append(agent_result.get('action', ''))

            if agent_result.get('recommendation'):
                summary['recommendations'].append(f"{agent_name}: {agent_result['recommendation']}")

        # Determine overall status
        if summary['critical_violations'] > 0:
            summary['overall_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
        elif summary['non_compliant_agents'] > summary['compliant_agents']:
            summary['overall_status'] = ComplianceStatus.NON_COMPLIANT.value
        elif summary['non_compliant_agents'] > 0:
            summary['overall_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value

        return summary


# Example Usage
if __name__ == "__main__":
    # Sample account data
    sample_account = {
        'customer_id': 'CUST12345',
        'account_id': 'ACC98765',
        'account_type': 'savings',
        'balance_current': 15000,
        'dormancy_status': 'dormant',
        'last_transaction_date': '2022-01-15',
        'dormancy_trigger_date': '2022-02-15',
        'contact_attempts_made': 2,
        'customer_type': 'individual',
        'currency': 'AED',
        'created_date': '2020-01-01',
        'updated_date': '2024-01-01',
        'updated_by': 'USER_001'
    }

    # Create orchestrator and process account
    orchestrator = ComplianceOrchestrator()
    compliance_results = orchestrator.process_account(sample_account)

    # Generate summary
    summary = orchestrator.generate_compliance_summary(compliance_results)

    # Export results
    orchestrator.export_to_csv(sample_account, compliance_results, "compliance_check_results.csv")

    # Print results
    print("COMPLIANCE ANALYSIS RESULTS:")
    print("=" * 50)

    for agent_name, result in compliance_results.items():
        print(f"\n{agent_name.upper().replace('_', ' ')}:")
        print(f"Status: {result.get('compliance_status', 'Unknown')}")
        print(f"Priority: {result.get('priority', 'None')}")
        if result.get('violations'):
            print(f"Violations: {result['violations']}")
        if result.get('action'):
            print(f"Action: {result['action']}")

    print(f"\nOVERALL COMPLIANCE SUMMARY:")
    print(f"Status: {summary['overall_status']}")
    print(f"Compliant Agents: {summary['compliant_agents']}/{summary['total_agents']}")
    print(f"Total Violations: {summary['total_violations']}")
    print(f"Critical Violations: {summary['critical_violations']}")