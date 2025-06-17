"""
compliance_agent.py - CBUAE Compliance Analyzer Agents System
17 Specialized Agents for Processing Dormant Accounts and Taking Regulatory Actions

Based on the Banking Compliance AI Agents Classification Table:
- Contact & Communication (2 agents)
- Process Management (3 agents) 
- Specialized Compliance (4 agents)
- Reporting & Retention (2 agents)
- Utility (5 agents)
- Master Orchestrator (1 agent)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enums for agent states and priorities
class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    ACTION_REQUIRED = "action_required"
    UNDER_REVIEW = "under_review"
    ESCALATED = "escalated"


class ActionPriority(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ComplianceCategory(Enum):
    CONTACT_COMMUNICATION = "Contact & Communication"
    PROCESS_MANAGEMENT = "Process Management"
    SPECIALIZED_COMPLIANCE = "Specialized Compliance"
    REPORTING_RETENTION = "Reporting & Retention"
    UTILITY = "Utility"


@dataclass
class ComplianceAction:
    """Data class for compliance actions"""
    account_id: str
    action_type: str
    priority: ActionPriority
    deadline: datetime
    description: str
    cbuae_article: str
    assigned_agent: str
    status: ComplianceStatus = ComplianceStatus.ACTION_REQUIRED
    created_date: datetime = None

    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()


@dataclass
class ComplianceResult:
    """Data class for compliance analysis results"""
    agent_name: str
    category: ComplianceCategory
    cbuae_article: str
    accounts_processed: int
    violations_found: int
    actions_generated: List[ComplianceAction]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class ComplianceAgentBase:
    """Base class for all compliance agents"""

    def __init__(self, agent_name: str, category: ComplianceCategory, cbuae_article: str):
        self.agent_name = agent_name
        self.category = category
        self.cbuae_article = cbuae_article
        self.logger = logging.getLogger(f"ComplianceAgent.{agent_name}")

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        """Base analyze method to be implemented by each agent"""
        raise NotImplementedError("Each agent must implement the analyze method")

    def generate_action(self, account_row: pd.Series, action_type: str,
                        priority: ActionPriority, days_to_deadline: int,
                        description: str) -> ComplianceAction:
        """Generate a compliance action for an account"""
        deadline = datetime.now() + timedelta(days=days_to_deadline)

        return ComplianceAction(
            account_id=account_row['account_id'],
            action_type=action_type,
            priority=priority,
            deadline=deadline,
            description=description,
            cbuae_article=self.cbuae_article,
            assigned_agent=self.agent_name
        )


# ===== CONTACT & COMMUNICATION AGENTS (2) =====

class DetectIncompleteContactAttemptsAgent(ComplianceAgentBase):
    """Contact & Communication - Art. 3.1, 5: Insufficient contact detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_incomplete_contact_attempts",
            category=ComplianceCategory.CONTACT_COMMUNICATION,
            cbuae_article="Art. 3.1, 5"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Find accounts with insufficient contact attempts
            insufficient_contact = dormant_accounts[
                (dormant_accounts['contact_attempts_made'] < 3) |
                (dormant_accounts['contact_attempts_made'].isna()) |
                (dormant_accounts['last_contact_date'].isna())
                ]

            for _, account in insufficient_contact.iterrows():
                attempts_made = account.get('contact_attempts_made', 0)

                if pd.isna(attempts_made) or attempts_made == 0:
                    action = self.generate_action(
                        account,
                        "INITIATE_CONTACT_ATTEMPTS",
                        ActionPriority.CRITICAL,
                        1,  # 1 day deadline
                        f"No contact attempts recorded. Initiate minimum 3 contact attempts via multiple channels."
                    )
                elif attempts_made < 3:
                    remaining = 3 - int(attempts_made)
                    action = self.generate_action(
                        account,
                        "COMPLETE_CONTACT_ATTEMPTS",
                        ActionPriority.CRITICAL,
                        3,  # 3 days deadline
                        f"Complete {remaining} additional contact attempts. Current: {int(attempts_made)}/3"
                    )
                else:
                    action = self.generate_action(
                        account,
                        "DOCUMENT_CONTACT_ATTEMPTS",
                        ActionPriority.HIGH,
                        7,  # 7 days deadline
                        "Document and verify all contact attempts are properly recorded."
                    )

                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(insufficient_contact),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class DetectUnflaggedDormantCandidatesAgent(ComplianceAgentBase):
    """Contact & Communication - Art. 2: Unflagged dormancy detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_unflagged_dormant_candidates",
            category=ComplianceCategory.CONTACT_COMMUNICATION,
            cbuae_article="Art. 2"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Find accounts that meet dormancy criteria but aren't flagged
            unflagged_dormant = dormant_accounts[
                (dormant_accounts['dormancy_period_months'] >= 36) &
                ((dormant_accounts['dormancy_status'] != 'DORMANT') |
                 (dormant_accounts['dormancy_status'].isna()))
                ]

            for _, account in unflagged_dormant.iterrows():
                dormant_months = account.get('dormancy_period_months', 0)
                current_status = account.get('dormancy_status', 'Unknown')

                action = self.generate_action(
                    account,
                    "UPDATE_DORMANCY_STATUS",
                    ActionPriority.HIGH,
                    2,  # 2 days deadline
                    f"Update account status to DORMANT (inactive for {dormant_months:.0f} months, current status: {current_status})"
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(unflagged_dormant),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


# ===== PROCESS MANAGEMENT AGENTS (3) =====

class DetectInternalLedgerCandidatesAgent(ComplianceAgentBase):
    """Process Management - Art. 3.4, 3.5: Internal ledger transfer detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_internal_ledger_candidates",
            category=ComplianceCategory.PROCESS_MANAGEMENT,
            cbuae_article="Art. 3.4, 3.5"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Accounts eligible for internal ledger transfer
            ledger_candidates = dormant_accounts[
                (dormant_accounts['dormancy_period_months'] >= 39) &  # 3 years + 3 month waiting
                (dormant_accounts['contact_attempts_made'] >= 3) &
                (dormant_accounts['balance_current'] > 0) &
                ((dormant_accounts['transferred_to_ledger_date'].isna()) |
                 (dormant_accounts['transferred_to_ledger_date'] == ''))
                ]

            for _, account in ledger_candidates.iterrows():
                balance = account.get('balance_current', 0)
                dormant_months = account.get('dormancy_period_months', 0)

                action = self.generate_action(
                    account,
                    "TRANSFER_TO_INTERNAL_LEDGER",
                    ActionPriority.HIGH,
                    5,  # 5 days deadline
                    f"Transfer to internal dormant ledger. Balance: {balance:.2f}, Dormant: {dormant_months:.0f} months"
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(ledger_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class DetectStatementFreezeCandidatesAgent(ComplianceAgentBase):
    """Process Management - Art. 7.3: Statement suppression detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_statement_freeze_candidates",
            category=ComplianceCategory.PROCESS_MANAGEMENT,
            cbuae_article="Art. 7.3"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Accounts eligible for statement suppression
            freeze_candidates = dormant_accounts[
                (dormant_accounts['dormancy_period_months'] >= 36) &
                (dormant_accounts['statement_frequency'].isin(['MONTHLY', 'QUARTERLY'])) &
                (dormant_accounts['statement_frequency'] != 'SUPPRESSED')
                ]

            for _, account in freeze_candidates.iterrows():
                current_frequency = account.get('statement_frequency', 'Unknown')

                action = self.generate_action(
                    account,
                    "SUPPRESS_STATEMENT_GENERATION",
                    ActionPriority.MEDIUM,
                    10,  # 10 days deadline
                    f"Suppress regular statement generation. Current: {current_frequency} → SUPPRESSED"
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(freeze_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class DetectCBUAETransferCandidatesAgent(ComplianceAgentBase):
    """Process Management - Art. 8: CBUAE transfer detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_cbuae_transfer_candidates",
            category=ComplianceCategory.PROCESS_MANAGEMENT,
            cbuae_article="Art. 8"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Accounts eligible for CBUAE transfer
            transfer_candidates = dormant_accounts[
                (dormant_accounts['dormancy_period_months'] >= 60) &  # 5+ years
                (dormant_accounts['balance_current'] > 0) &
                (dormant_accounts['address_known'] == 'No') &
                ((dormant_accounts['transferred_to_cb_date'].isna()) |
                 (dormant_accounts['transferred_to_cb_date'] == ''))
                ]

            for _, account in transfer_candidates.iterrows():
                balance = account.get('balance_current', 0)
                dormant_months = account.get('dormancy_period_months', 0)
                currency = account.get('currency', 'AED')

                if currency != 'AED':
                    action = self.generate_action(
                        account,
                        "CONVERT_CURRENCY_FOR_CB_TRANSFER",
                        ActionPriority.HIGH,
                        7,  # 7 days deadline
                        f"Convert {currency} balance to AED before CBUAE transfer. Balance: {balance:.2f} {currency}"
                    )
                else:
                    action = self.generate_action(
                        account,
                        "PREPARE_CBUAE_TRANSFER",
                        ActionPriority.HIGH,
                        14,  # 14 days deadline
                        f"Prepare for CBUAE transfer. Balance: {balance:.2f} AED, Dormant: {dormant_months:.0f} months"
                    )

                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(transfer_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


# ===== SPECIALIZED COMPLIANCE AGENTS (4) =====

class DetectForeignCurrencyConversionNeededAgent(ComplianceAgentBase):
    """Specialized Compliance - Art. 8.5: FX conversion detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_foreign_currency_conversion_needed",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article="Art. 8.5"
        )

        # CBUAE exchange rates (example rates)
        self.exchange_rates = {
            'USD': 3.67,
            'EUR': 4.0,
            'GBP': 4.5,
            'SAR': 0.98,
            'AED': 1.0
        }

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Foreign currency accounts needing conversion
            fx_conversion_needed = dormant_accounts[
                (dormant_accounts['dormancy_period_months'] >= 60) &
                (dormant_accounts['currency'] != 'AED') &
                (dormant_accounts['currency'].notna()) &
                (dormant_accounts['balance_current'] > 0)
                ]

            for _, account in fx_conversion_needed.iterrows():
                currency = account.get('currency', 'Unknown')
                balance = account.get('balance_current', 0)
                exchange_rate = self.exchange_rates.get(currency, 1.0)
                aed_equivalent = balance * exchange_rate

                action = self.generate_action(
                    account,
                    "CONVERT_FOREIGN_CURRENCY",
                    ActionPriority.MEDIUM,
                    7,  # 7 days deadline
                    f"Convert {balance:.2f} {currency} to {aed_equivalent:.2f} AED (Rate: {exchange_rate})"
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(fx_conversion_needed),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class DetectSDBCourtApplicationNeededAgent(ComplianceAgentBase):
    """Specialized Compliance - Art. 3.7: Safe Deposit Box court application"""

    def __init__(self):
        super().__init__(
            agent_name="detect_sdb_court_application_needed",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article="Art. 3.7"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Safe deposit boxes needing court application
            court_application_needed = dormant_accounts[
                (dormant_accounts['box_id'].notna()) &
                (dormant_accounts['outstanding_charges'] > 0) &
                (dormant_accounts['dormancy_period_months'] >= 36) &
                ((dormant_accounts['court_order_date'].isna()) |
                 (dormant_accounts['court_order_date'] == ''))
                ]

            for _, account in court_application_needed.iterrows():
                box_id = account.get('box_id', 'Unknown')
                charges = account.get('outstanding_charges', 0)

                action = self.generate_action(
                    account,
                    "FILE_COURT_APPLICATION_SDB",
                    ActionPriority.CRITICAL,
                    3,  # 3 days deadline
                    f"File court application for SDB {box_id}. Outstanding charges: {charges:.2f} AED"
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(court_application_needed),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class DetectUnclaimedPaymentInstrumentsLedgerAgent(ComplianceAgentBase):
    """Specialized Compliance - Art. 3.6: Unclaimed instruments ledger"""

    def __init__(self):
        super().__init__(
            agent_name="detect_unclaimed_payment_instruments_ledger",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article="Art. 3.6"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Unclaimed instruments needing ledger transfer
            instruments_ledger = dormant_accounts[
                (dormant_accounts['instrument_type'].notna()) &
                (dormant_accounts['unclaimed_since'].notna()) &
                ((dormant_accounts['transferred_to_ledger_date'].isna()) |
                 (dormant_accounts['transferred_to_ledger_date'] == '')) &
                (dormant_accounts['amount'] > 0)
                ]

            for _, account in instruments_ledger.iterrows():
                instrument_type = account.get('instrument_type', 'Unknown')
                amount = account.get('amount', 0)
                unclaimed_since = account.get('unclaimed_since', 'Unknown')

                action = self.generate_action(
                    account,
                    "TRANSFER_UNCLAIMED_INSTRUMENT_TO_LEDGER",
                    ActionPriority.MEDIUM,
                    7,  # 7 days deadline
                    f"Transfer {instrument_type} to unclaimed ledger. Amount: {amount:.2f}, Unclaimed since: {unclaimed_since}"
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(instruments_ledger),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class DetectClaimProcessingPendingAgent(ComplianceAgentBase):
    """Specialized Compliance - Art. 4: Customer claims processing"""

    def __init__(self):
        super().__init__(
            agent_name="detect_claim_processing_pending",
            category=ComplianceCategory.SPECIALIZED_COMPLIANCE,
            cbuae_article="Art. 4"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Pending customer claims
            pending_claims = dormant_accounts[
                (dormant_accounts['reclaim_status'].notna()) &
                (dormant_accounts['reclaim_status'].isin(['PENDING', 'PROCESSING'])) &
                (dormant_accounts['reclaim_date'].notna())
                ]

            current_date = pd.Timestamp.now()

            for _, account in pending_claims.iterrows():
                reclaim_date = pd.to_datetime(account.get('reclaim_date'), errors='coerce')
                status = account.get('reclaim_status', 'Unknown')

                if pd.notna(reclaim_date):
                    days_pending = (current_date - reclaim_date).days

                    if days_pending > 30:
                        priority = ActionPriority.CRITICAL
                        deadline_days = 1
                    elif days_pending > 14:
                        priority = ActionPriority.HIGH
                        deadline_days = 3
                    else:
                        priority = ActionPriority.MEDIUM
                        deadline_days = 7

                    action = self.generate_action(
                        account,
                        "EXPEDITE_CLAIM_PROCESSING",
                        priority,
                        deadline_days,
                        f"Process overdue customer reclaim. Status: {status}, Pending: {days_pending} days"
                    )
                    actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(pending_claims),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


# ===== REPORTING & RETENTION AGENTS (2) =====

class CheckRecordRetentionComplianceAgent(ComplianceAgentBase):
    """Reporting & Retention - Art. 3.9: Record retention compliance"""

    def __init__(self):
        super().__init__(
            agent_name="check_record_retention_compliance",
            category=ComplianceCategory.REPORTING_RETENTION,
            cbuae_article="Art. 3.9"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Check for missing critical documentation
            critical_fields = [
                'last_contact_date', 'contact_attempts_made', 'dormancy_trigger_date',
                'dormancy_classification_date'
            ]

            for _, account in dormant_accounts.iterrows():
                missing_fields = []

                for field in critical_fields:
                    if field in account and (pd.isna(account[field]) or account[field] == ''):
                        missing_fields.append(field)

                if missing_fields:
                    action = self.generate_action(
                        account,
                        "COMPLETE_RECORD_DOCUMENTATION",
                        ActionPriority.CRITICAL,
                        2,  # 2 days deadline
                        f"Complete missing documentation: {', '.join(missing_fields)}"
                    )
                    actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(actions),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class GenerateAnnualCBUAEReportSummaryAgent(ComplianceAgentBase):
    """Reporting & Retention - Art. 3.10: Annual CBUAE reporting"""

    def __init__(self):
        super().__init__(
            agent_name="generate_annual_cbuae_report_summary",
            category=ComplianceCategory.REPORTING_RETENTION,
            cbuae_article="Art. 3.10"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # All dormant accounts with positive balances need to be in annual report
            reporting_candidates = dormant_accounts[
                (dormant_accounts['dormancy_period_months'] >= 12) &
                (dormant_accounts['balance_current'] > 0)
                ]

            if len(reporting_candidates) > 0:
                # Generate single action for annual reporting compilation
                total_balance = reporting_candidates['balance_current'].sum()
                account_count = len(reporting_candidates)

                # Use first account as representative (for action generation)
                representative_account = reporting_candidates.iloc[0]

                action = self.generate_action(
                    representative_account,
                    "COMPILE_ANNUAL_CBUAE_REPORT",
                    ActionPriority.MEDIUM,
                    30,  # 30 days deadline
                    f"Compile annual CBUAE report for {account_count} dormant accounts. Total balance: {total_balance:.2f} AED"
                )
                actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(reporting_candidates),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


# ===== UTILITY AGENTS (5) =====

class LogFlagInstructionsAgent(ComplianceAgentBase):
    """Utility - Internal: Flagging instruction logger"""

    def __init__(self):
        super().__init__(
            agent_name="log_flag_instructions",
            category=ComplianceCategory.UTILITY,
            cbuae_article="Internal"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        start_time = datetime.now()
        actions = []

        try:
            # Log flagging instructions for all dormant accounts
            for _, account in dormant_accounts.iterrows():
                dormant_months = account.get('dormancy_period_months', 0)
                status = account.get('dormancy_status', 'Unknown')

                if dormant_months >= 36 and status != 'DORMANT':
                    action = self.generate_action(
                        account,
                        "LOG_FLAGGING_INSTRUCTION",
                        ActionPriority.LOW,
                        30,  # 30 days deadline
                        f"Log flagging instruction for dormant account ({dormant_months:.0f} months inactive)"
                    )
                    actions.append(action)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=len(actions),
                actions_generated=actions,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")

            return ComplianceResult(
                agent_name=self.agent_name,
                category=self.category,
                cbuae_article=self.cbuae_article,
                accounts_processed=len(dormant_accounts),
                violations_found=0,
                actions_generated=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class DetectFlagCandidatesAgent(ComplianceAgentBase):
    """Utility - Art. 2: Alias for unflagged dormant detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_flag_candidates",
            category=ComplianceCategory.UTILITY,
            cbuae_article="Art. 2"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        # This is an alias/utility agent - delegates to main unflagged detection
        unflagged_agent = DetectUnflaggedDormantCandidatesAgent()
        result = unflagged_agent.analyze(dormant_accounts)

        # Update agent name to reflect this is the utility version
        result.agent_name = self.agent_name
        return result


class DetectLedgerCandidatesAgent(ComplianceAgentBase):
    """Utility - Art. 3.4, 3.5: Alias for internal ledger detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_ledger_candidates",
            category=ComplianceCategory.UTILITY,
            cbuae_article="Art. 3.4, 3.5"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        # This is an alias/utility agent - delegates to main ledger detection
        ledger_agent = DetectInternalLedgerCandidatesAgent()
        result = ledger_agent.analyze(dormant_accounts)

        # Update agent name to reflect this is the utility version
        result.agent_name = self.agent_name
        return result


class DetectFreezeCandidatesAgent(ComplianceAgentBase):
    """Utility - Art. 7.3: Alias for statement freeze detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_freeze_candidates",
            category=ComplianceCategory.UTILITY,
            cbuae_article="Art. 7.3"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        # This is an alias/utility agent - delegates to main statement freeze detection
        freeze_agent = DetectStatementFreezeCandidatesAgent()
        result = freeze_agent.analyze(dormant_accounts)

        # Update agent name to reflect this is the utility version
        result.agent_name = self.agent_name
        return result


class DetectTransferCandidatesToCBAgent(ComplianceAgentBase):
    """Utility - Art. 8: Alias for CBUAE transfer detection"""

    def __init__(self):
        super().__init__(
            agent_name="detect_transfer_candidates_to_cb",
            category=ComplianceCategory.UTILITY,
            cbuae_article="Art. 8"
        )

    def analyze(self, dormant_accounts: pd.DataFrame) -> ComplianceResult:
        # This is an alias/utility agent - delegates to main CBUAE transfer detection
        transfer_agent = DetectCBUAETransferCandidatesAgent()
        result = transfer_agent.analyze(dormant_accounts)

        # Update agent name to reflect this is the utility version
        result.agent_name = self.agent_name
        return result


# ===== MASTER ORCHESTRATOR (1) =====

class RunAllComplianceChecksAgent:
    """Master Orchestrator - All: Main compliance analysis orchestrator"""

    def __init__(self):
        self.agent_name = "run_all_compliance_checks"
        self.category = ComplianceCategory.UTILITY
        self.logger = logging.getLogger("ComplianceOrchestrator")

        # Initialize all compliance agents
        self.agents = [
            # Contact & Communication (2)
            DetectIncompleteContactAttemptsAgent(),
            DetectUnflaggedDormantCandidatesAgent(),

            # Process Management (3)
            DetectInternalLedgerCandidatesAgent(),
            DetectStatementFreezeCandidatesAgent(),
            DetectCBUAETransferCandidatesAgent(),

            # Specialized Compliance (4)
            DetectForeignCurrencyConversionNeededAgent(),
            DetectSDBCourtApplicationNeededAgent(),
            DetectUnclaimedPaymentInstrumentsLedgerAgent(),
            DetectClaimProcessingPendingAgent(),

            # Reporting & Retention (2)
            CheckRecordRetentionComplianceAgent(),
            GenerateAnnualCBUAEReportSummaryAgent(),

            # Utility (5) - Note: These are aliases, can be excluded to avoid duplication
            LogFlagInstructionsAgent(),
            # DetectFlagCandidatesAgent(),      # Alias - commented to avoid duplication
            # DetectLedgerCandidatesAgent(),    # Alias - commented to avoid duplication  
            # DetectFreezeCandidatesAgent(),    # Alias - commented to avoid duplication
            # DetectTransferCandidatesToCBAgent() # Alias - commented to avoid duplication
        ]

    def run_comprehensive_compliance_analysis(self, dormant_accounts: pd.DataFrame) -> Dict[str, Any]:
        """Run all compliance agents on dormant accounts"""
        start_time = datetime.now()

        self.logger.info(f"Starting comprehensive compliance analysis on {len(dormant_accounts)} dormant accounts")

        results = {
            'analysis_timestamp': start_time.isoformat(),
            'dormant_accounts_processed': len(dormant_accounts),
            'agents_executed': len(self.agents),
            'agent_results': {},
            'consolidated_actions': [],
            'summary_stats': {},
            'success': True,
            'errors': []
        }

        total_violations = 0
        total_actions = 0
        agent_performance = []

        for agent in self.agents:
            try:
                self.logger.info(f"Executing agent: {agent.agent_name}")

                agent_result = agent.analyze(dormant_accounts)

                results['agent_results'][agent.agent_name] = {
                    'category': agent_result.category.value,
                    'cbuae_article': agent_result.cbuae_article,
                    'accounts_processed': agent_result.accounts_processed,
                    'violations_found': agent_result.violations_found,
                    'actions_generated': len(agent_result.actions_generated),
                    'processing_time': agent_result.processing_time,
                    'success': agent_result.success,
                    'error_message': agent_result.error_message
                }

                if agent_result.success:
                    total_violations += agent_result.violations_found
                    total_actions += len(agent_result.actions_generated)
                    results['consolidated_actions'].extend(agent_result.actions_generated)

                    agent_performance.append({
                        'agent': agent.agent_name,
                        'category': agent_result.category.value,
                        'violations': agent_result.violations_found,
                        'actions': len(agent_result.actions_generated),
                        'processing_time': agent_result.processing_time
                    })
                else:
                    results['errors'].append({
                        'agent': agent.agent_name,
                        'error': agent_result.error_message
                    })

            except Exception as e:
                error_msg = f"Failed to execute agent {agent.agent_name}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append({
                    'agent': agent.agent_name,
                    'error': error_msg
                })
                results['success'] = False

        # Calculate summary statistics
        processing_time = (datetime.now() - start_time).total_seconds()

        results['summary_stats'] = {
            'total_violations_found': total_violations,
            'total_actions_generated': total_actions,
            'total_processing_time': processing_time,
            'average_processing_time_per_agent': processing_time / len(self.agents) if self.agents else 0,
            'agents_with_violations': len([r for r in results['agent_results'].values() if r['violations_found'] > 0]),
            'compliance_rate': ((len(dormant_accounts) - total_violations) / len(
                dormant_accounts) * 100) if dormant_accounts is not None and len(dormant_accounts) > 0 else 100
        }

        # Priority breakdown
        priority_breakdown = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for action in results['consolidated_actions']:
            priority_breakdown[action.priority.value] += 1

        results['priority_breakdown'] = priority_breakdown

        # Category breakdown
        category_breakdown = {}
        for agent_name, agent_result in results['agent_results'].items():
            category = agent_result['category']
            if category not in category_breakdown:
                category_breakdown[category] = {
                    'agents': 0,
                    'violations': 0,
                    'actions': 0
                }
            category_breakdown[category]['agents'] += 1
            category_breakdown[category]['violations'] += agent_result['violations_found']
            category_breakdown[category]['actions'] += agent_result['actions_generated']

        results['category_breakdown'] = category_breakdown

        self.logger.info(
            f"Compliance analysis completed. {total_violations} violations found, {total_actions} actions generated")

        return results

    def generate_action_plan(self, compliance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prioritized action plan from compliance results"""

        actions = compliance_results.get('consolidated_actions', [])

        if not actions:
            return {
                'action_plan': [],
                'execution_timeline': {},
                'resource_requirements': {},
                'priority_summary': {}
            }

        # Sort actions by priority and deadline
        priority_order = {
            ActionPriority.CRITICAL: 0,
            ActionPriority.HIGH: 1,
            ActionPriority.MEDIUM: 2,
            ActionPriority.LOW: 3
        }

        sorted_actions = sorted(actions, key=lambda x: (priority_order[x.priority], x.deadline))

        # Group actions by timeline
        timeline_groups = {
            'immediate': [],  # Due within 24 hours
            'urgent': [],  # Due within 7 days
            'short_term': [],  # Due within 30 days
            'long_term': []  # Due beyond 30 days
        }

        now = datetime.now()

        for action in sorted_actions:
            days_to_deadline = (action.deadline - now).days

            if days_to_deadline <= 1:
                timeline_groups['immediate'].append(action)
            elif days_to_deadline <= 7:
                timeline_groups['urgent'].append(action)
            elif days_to_deadline <= 30:
                timeline_groups['short_term'].append(action)
            else:
                timeline_groups['long_term'].append(action)

        # Generate resource requirements estimate
        resource_requirements = {
            'compliance_officers': len(
                [a for a in actions if a.priority in [ActionPriority.CRITICAL, ActionPriority.HIGH]]),
            'legal_review_required': len(
                [a for a in actions if 'court' in a.action_type.lower() or 'legal' in a.description.lower()]),
            'system_updates_required': len([a for a in actions if any(
                keyword in a.action_type.lower() for keyword in ['status', 'suppress', 'transfer', 'convert'])]),
            'customer_contact_required': len([a for a in actions if 'contact' in a.action_type.lower()]),
            'estimated_total_hours': len(actions) * 2  # Estimate 2 hours per action
        }

        return {
            'action_plan': [
                {
                    'account_id': action.account_id,
                    'action_type': action.action_type,
                    'priority': action.priority.value,
                    'deadline': action.deadline.isoformat(),
                    'description': action.description,
                    'cbuae_article': action.cbuae_article,
                    'assigned_agent': action.assigned_agent,
                    'days_to_deadline': (action.deadline - now).days
                } for action in sorted_actions
            ],
            'execution_timeline': {
                'immediate_actions': len(timeline_groups['immediate']),
                'urgent_actions': len(timeline_groups['urgent']),
                'short_term_actions': len(timeline_groups['short_term']),
                'long_term_actions': len(timeline_groups['long_term'])
            },
            'resource_requirements': resource_requirements,
            'priority_summary': compliance_results.get('priority_breakdown', {})
        }


# ===== HELPER FUNCTIONS =====

def get_all_compliance_agents() -> List[ComplianceAgentBase]:
    """Get list of all compliance agents"""
    return [
        # Contact & Communication (2)
        DetectIncompleteContactAttemptsAgent(),
        DetectUnflaggedDormantCandidatesAgent(),

        # Process Management (3)
        DetectInternalLedgerCandidatesAgent(),
        DetectStatementFreezeCandidatesAgent(),
        DetectCBUAETransferCandidatesAgent(),

        # Specialized Compliance (4)
        DetectForeignCurrencyConversionNeededAgent(),
        DetectSDBCourtApplicationNeededAgent(),
        DetectUnclaimedPaymentInstrumentsLedgerAgent(),
        DetectClaimProcessingPendingAgent(),

        # Reporting & Retention (2)
        CheckRecordRetentionComplianceAgent(),
        GenerateAnnualCBUAEReportSummaryAgent(),

        # Utility (1) - Only non-alias utility agent
        LogFlagInstructionsAgent()
    ]


def get_compliance_agents_by_category() -> Dict[ComplianceCategory, List[ComplianceAgentBase]]:
    """Get compliance agents grouped by category"""
    agents = get_all_compliance_agents()

    categorized = {category: [] for category in ComplianceCategory}

    for agent in agents:
        categorized[agent.category].append(agent)

    return categorized


def run_compliance_analysis_on_dormant_accounts(dormant_accounts: pd.DataFrame,
                                                include_action_plan: bool = True) -> Dict[str, Any]:
    """Main function to run compliance analysis on dormant accounts"""

    orchestrator = RunAllComplianceChecksAgent()

    # Run comprehensive compliance analysis
    compliance_results = orchestrator.run_comprehensive_compliance_analysis(dormant_accounts)

    # Generate action plan if requested
    if include_action_plan and compliance_results['success']:
        action_plan = orchestrator.generate_action_plan(compliance_results)
        compliance_results['action_plan'] = action_plan

    return compliance_results


def export_compliance_actions_to_csv(compliance_results: Dict[str, Any],
                                     filename: Optional[str] = None) -> str:
    """Export compliance actions to CSV format"""

    actions = compliance_results.get('consolidated_actions', [])

    if not actions:
        return "account_id,action_type,priority,deadline,description,cbuae_article,assigned_agent,status\n"

    csv_data = []
    for action in actions:
        csv_data.append({
            'account_id': action.account_id,
            'action_type': action.action_type,
            'priority': action.priority.value,
            'deadline': action.deadline.strftime('%Y-%m-%d %H:%M:%S'),
            'description': action.description,
            'cbuae_article': action.cbuae_article,
            'assigned_agent': action.assigned_agent,
            'status': action.status.value,
            'created_date': action.created_date.strftime('%Y-%m-%d %H:%M:%S')
        })

    df = pd.DataFrame(csv_data)

    if filename:
        df.to_csv(filename, index=False)
        return f"Actions exported to {filename}"
    else:
        return df.to_csv(index=False)


def get_compliance_agent_coverage() -> Dict[str, Any]:
    """Get coverage information for compliance agents"""

    agents = get_all_compliance_agents()

    coverage = {
        'total_agents': len(agents),
        'categories': {},
        'cbuae_articles': set(),
        'agent_details': []
    }

    for agent in agents:
        # Category breakdown
        category_name = agent.category.value
        if category_name not in coverage['categories']:
            coverage['categories'][category_name] = 0
        coverage['categories'][category_name] += 1

        # CBUAE articles
        coverage['cbuae_articles'].add(agent.cbuae_article)

        # Agent details
        coverage['agent_details'].append({
            'name': agent.agent_name,
            'category': category_name,
            'article': agent.cbuae_article
        })

    coverage['cbuae_articles'] = list(coverage['cbuae_articles'])
    coverage['total_articles_covered'] = len(coverage['cbuae_articles'])

    return coverage


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    # Demo usage
    import sys

    print("CBUAE Compliance Analyzer Agents System")
    print("=" * 50)

    # Show agent coverage
    coverage = get_compliance_agent_coverage()
    print(f"Total Agents: {coverage['total_agents']}")
    print(f"Categories: {list(coverage['categories'].keys())}")
    print(f"CBUAE Articles Covered: {coverage['total_articles_covered']}")
    print()

    # Category breakdown
    print("Agent Distribution by Category:")
    for category, count in coverage['categories'].items():
        print(f"  {category}: {count} agents")
    print()

    # Create sample dormant accounts data for testing
    sample_data = pd.DataFrame({
        'account_id': ['ACC001', 'ACC002', 'ACC003', 'ACC004', 'ACC005'],
        'dormancy_period_months': [40, 45, 38, 65, 42],
        'balance_current': [1500.0, 25000.0, 500.0, 12000.0, 3000.0],
        'currency': ['AED', 'USD', 'AED', 'EUR', 'AED'],
        'contact_attempts_made': [1, 0, 3, 2, 1],
        'last_contact_date': [None, None, '2023-01-15', None, '2022-12-10'],
        'dormancy_status': ['ACTIVE', None, 'DORMANT', 'ACTIVE', None],
        'statement_frequency': ['MONTHLY', 'QUARTERLY', 'SUPPRESSED', 'MONTHLY', 'QUARTERLY'],
        'address_known': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'transferred_to_cb_date': [None, None, None, None, None],
        'transferred_to_ledger_date': [None, None, None, None, None],
        'box_id': [None, None, None, 'BOX123', None],
        'outstanding_charges': [0, 0, 0, 500, 0],
        'reclaim_status': [None, 'PENDING', None, None, 'PROCESSING'],
        'reclaim_date': [None, '2024-01-15', None, None, '2024-02-01'],
        'instrument_type': [None, None, 'BANKERS_CHEQUE', None, None],
        'unclaimed_since': [None, None, '2023-06-01', None, None],
        'amount': [0, 0, 1000, 0, 0],
        'dormancy_trigger_date': ['2021-01-01', None, '2020-12-01', '2019-06-01', '2021-03-01']
    })

    print("Running compliance analysis on sample dormant accounts...")

    # Run compliance analysis
    results = run_compliance_analysis_on_dormant_accounts(sample_data)

    if results['success']:
        print(f"\nCompliance Analysis Results:")
        print(f"  Dormant Accounts Processed: {results['dormant_accounts_processed']}")
        print(f"  Total Violations Found: {results['summary_stats']['total_violations_found']}")
        print(f"  Total Actions Generated: {results['summary_stats']['total_actions_generated']}")
        print(f"  Compliance Rate: {results['summary_stats']['compliance_rate']:.1f}%")
        print(f"  Processing Time: {results['summary_stats']['total_processing_time']:.2f} seconds")

        print(f"\nPriority Breakdown:")
        for priority, count in results['priority_breakdown'].items():
            print(f"  {priority}: {count} actions")

        print(f"\nCategory Breakdown:")
        for category, stats in results['category_breakdown'].items():
            print(
                f"  {category}: {stats['agents']} agents, {stats['violations']} violations, {stats['actions']} actions")

        if 'action_plan' in results:
            action_plan = results['action_plan']
            print(f"\nAction Plan Timeline:")
            timeline = action_plan['execution_timeline']
            print(f"  Immediate (≤1 day): {timeline['immediate_actions']} actions")
            print(f"  Urgent (≤7 days): {timeline['urgent_actions']} actions")
            print(f"  Short-term (≤30 days): {timeline['short_term_actions']} actions")
            print(f"  Long-term (>30 days): {timeline['long_term_actions']} actions")

        if results['errors']:
            print(f"\nErrors Encountered:")
            for error in results['errors']:
                print(f"  {error['agent']}: {error['error']}")
    else:
        print("Compliance analysis failed!")
        for error in results['errors']:
            print(f"  Error: {error}")

    print("\nCompliance analysis completed.")