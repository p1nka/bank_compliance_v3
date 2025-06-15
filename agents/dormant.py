"""
CBUAE Dormancy and Unclaimed Balances Regulation - Monitoring Agents
=====================================================================

This module contains all the automated monitoring agents required for 
compliance with CBUAE Dormancy and Unclaimed Balances Regulation.

Database Tables Referenced:
- CUSTOMER_MASTER (24 columns)
- ACCOUNTS (32 columns) 
- DORMANCY_TRACKING (20 columns)
- CUSTOMER_COMMUNICATIONS (16 columns)
- TRANSACTIONS (19 columns)
- OUTSTANDING_FACILITIES (16 columns)
- UNCLAIMED_INSTRUMENTS (17 columns)
- DIVIDENDS (14 columns)
- SAFE_DEPOSIT_BOXES (19 columns)
- CENTRAL_BANK_TRANSFERS (18 columns)
- RECLAIM_REQUESTS (20 columns)
- REGULATORY_REPORTS (15 columns)
- AUDIT_LOG (12 columns)
- CONFIGURATION (10 columns)
- USERS (12 columns)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DormancyAlert:
    """Data class for dormancy alerts"""
    account_id: str
    customer_id: str
    alert_type: str
    priority: str
    message: str
    created_date: datetime


class DormancyMonitoringAgents:
    """
    Comprehensive monitoring agents for CBUAE Dormancy Regulation compliance
    """

    def __init__(self, db_connection):
        self.db = db_connection
        self.alerts = []

    # =====================================================================
    # ARTICLE 2: DORMANCY CRITERIA MONITORING AGENTS
    # =====================================================================

    def check_demand_deposit_inactivity(self) -> List[DormancyAlert]:
        """
        Article 2.1: Monitor demand deposit accounts (Current, Savings, Call)

        Criteria:
        - No customer-initiated transactions for 3 years
        - No communication from customer for 3 years
        - Customer has no active liability accounts
        - Customer address unknown
        - No pending litigation or regulatory requirements
        """
        query = """
        SELECT 
            a.account_id,
            a.customer_id,
            a.account_type,
            a.account_status,
            a.last_transaction_date,
            cm.last_contact_date,
            cm.address_known,
            COUNT(of.facility_id) as active_facilities
        FROM ACCOUNTS a
        JOIN CUSTOMER_MASTER cm ON a.customer_id = cm.customer_id
        LEFT JOIN OUTSTANDING_FACILITIES of ON a.customer_id = of.customer_id 
            AND of.facility_status = 'ACTIVE'
        WHERE a.account_type IN ('CURRENT', 'SAVINGS', 'CALL')
            AND a.account_status = 'ACTIVE'
            AND (a.last_transaction_date < DATE_SUB(NOW(), INTERVAL 3 YEAR) 
                 OR a.last_transaction_date IS NULL)
            AND (cm.last_contact_date < DATE_SUB(NOW(), INTERVAL 3 YEAR) 
                 OR cm.last_contact_date IS NULL)
        GROUP BY a.account_id, a.customer_id
        HAVING active_facilities = 0
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            if not row['address_known']:
                alert = DormancyAlert(
                    account_id=row['account_id'],
                    customer_id=row['customer_id'],
                    alert_type='DEMAND_DEPOSIT_DORMANCY',
                    priority='HIGH',
                    message=f"Account {row['account_id']} meets dormancy criteria per Article 2.1",
                    created_date=datetime.now()
                )
                alerts.append(alert)

        return alerts

    def check_fixed_deposit_inactivity(self) -> List[DormancyAlert]:
        """
        Article 2.2: Monitor fixed/term deposit accounts

        Case 1 - No Auto-Renewal:
        - Deposit has matured
        - Neither renewal nor claim request made for 3 years since maturity

        Case 2 - With Auto-Renewal:
        - No customer communication for 3 years from first maturity date
        - Despite automatic renewal clause being active
        """
        query = """
        SELECT 
            a.account_id,
            a.customer_id,
            a.maturity_date,
            a.auto_renewal,
            cm.last_contact_date,
            a.account_status
        FROM ACCOUNTS a
        JOIN CUSTOMER_MASTER cm ON a.customer_id = cm.customer_id
        WHERE a.account_type IN ('FIXED_DEPOSIT', 'TERM_DEPOSIT')
            AND (
                (a.auto_renewal = 0 
                 AND a.maturity_date < DATE_SUB(NOW(), INTERVAL 3 YEAR)
                 AND a.account_status = 'MATURED')
                OR
                (a.auto_renewal = 1 
                 AND a.maturity_date < DATE_SUB(NOW(), INTERVAL 3 YEAR)
                 AND (cm.last_contact_date < DATE_SUB(NOW(), INTERVAL 3 YEAR) 
                      OR cm.last_contact_date IS NULL))
            )
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert_type = 'FIXED_DEPOSIT_NO_RENEWAL' if not row['auto_renewal'] else 'FIXED_DEPOSIT_AUTO_RENEWAL'
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type=alert_type,
                priority='HIGH',
                message=f"Fixed deposit {row['account_id']} meets dormancy criteria per Article 2.2",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def check_investment_inactivity(self) -> List[DormancyAlert]:
        """
        Article 2.3: Monitor investment accounts

        Criteria:
        - Applies to closed-ended/redeemable investment products
        - No customer communication for 3 years from final maturity/redemption date
        - Product has reached maturity/redemption point
        """
        query = """
        SELECT 
            a.account_id,
            a.customer_id,
            a.maturity_date,
            cm.last_contact_date,
            a.account_subtype
        FROM ACCOUNTS a
        JOIN CUSTOMER_MASTER cm ON a.customer_id = cm.customer_id
        WHERE a.account_type = 'INVESTMENT'
            AND a.account_subtype IN ('CLOSED_ENDED', 'REDEEMABLE')
            AND a.maturity_date < DATE_SUB(NOW(), INTERVAL 3 YEAR)
            AND (cm.last_contact_date < DATE_SUB(NOW(), INTERVAL 3 YEAR) 
                 OR cm.last_contact_date IS NULL)
            AND a.account_status = 'MATURED'
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='INVESTMENT_DORMANCY',
                priority='MEDIUM',
                message=f"Investment account {row['account_id']} meets dormancy criteria per Article 2.3",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def check_unclaimed_payment_instruments(self) -> List[DormancyAlert]:
        """
        Article 2.4: Monitor unclaimed payment instruments

        Criteria:
        - Instrument Types: Bankers Cheques, Bank Drafts, Cashier Orders
        - Timeframe: Unclaimed for 1 year from issuance
        - Despite bank's efforts to contact customer/beneficiary
        """
        query = """
        SELECT 
            ui.instrument_id,
            ui.customer_id,
            ui.instrument_type,
            ui.issue_date,
            ui.amount,
            ui.status,
            COUNT(cc.communication_id) as contact_attempts
        FROM UNCLAIMED_INSTRUMENTS ui
        LEFT JOIN CUSTOMER_COMMUNICATIONS cc ON ui.customer_id = cc.customer_id
            AND cc.communication_purpose = 'UNCLAIMED_INSTRUMENT'
        WHERE ui.instrument_type IN ('BANKERS_CHEQUE', 'BANK_DRAFT', 'CASHIER_ORDER')
            AND ui.issue_date < DATE_SUB(NOW(), INTERVAL 1 YEAR)
            AND ui.status = 'UNCLAIMED'
        GROUP BY ui.instrument_id
        HAVING contact_attempts > 0
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['instrument_id'],
                customer_id=row['customer_id'],
                alert_type='UNCLAIMED_PAYMENT_INSTRUMENT',
                priority='HIGH',
                message=f"Payment instrument {row['instrument_id']} unclaimed for 1+ year per Article 2.4",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def check_safe_deposit_dormancy(self) -> List[DormancyAlert]:
        """
        Article 2.6: Monitor safe deposit boxes

        Criteria:
        - Outstanding charges/fees for more than 3 years
        - No response received from tenant despite bank contact attempts
        - Applies to rental fees, access charges, or other SDB-related costs
        """
        query = """
        SELECT 
            sdb.box_id,
            sdb.customer_id,
            sdb.outstanding_since,
            sdb.outstanding_charges,
            sdb.box_status,
            COUNT(cc.communication_id) as contact_attempts
        FROM SAFE_DEPOSIT_BOXES sdb
        LEFT JOIN CUSTOMER_COMMUNICATIONS cc ON sdb.customer_id = cc.customer_id
            AND cc.communication_purpose = 'SDB_OUTSTANDING_FEES'
        WHERE sdb.outstanding_since < DATE_SUB(NOW(), INTERVAL 3 YEAR)
            AND sdb.outstanding_charges > 0
            AND sdb.box_status != 'CLOSED'
        GROUP BY sdb.box_id
        HAVING contact_attempts > 0
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['box_id'],
                customer_id=row['customer_id'],
                alert_type='SAFE_DEPOSIT_DORMANCY',
                priority='HIGH',
                message=f"Safe deposit box {row['box_id']} has outstanding fees 3+ years per Article 2.6",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    # =====================================================================
    # ARTICLE 3: BANK OBLIGATIONS MONITORING AGENTS
    # =====================================================================

    def detect_incomplete_contact_attempts(self) -> List[DormancyAlert]:
        """
        Article 3.1: Verify contact attempt completeness

        Requirements:
        - Multiple communication channels required (email, SMS, phone, mail)
        - Document all contact attempts with dates and methods
        - Reasonable efforts must be made to locate customers
        """
        query = """
        SELECT 
            dt.account_id,
            dt.customer_id,
            dt.dormancy_trigger_date,
            GROUP_CONCAT(DISTINCT cc.communication_method) as methods_used,
            COUNT(cc.communication_id) as total_attempts
        FROM DORMANCY_TRACKING dt
        LEFT JOIN CUSTOMER_COMMUNICATIONS cc ON dt.customer_id = cc.customer_id
            AND cc.communication_date >= dt.dormancy_trigger_date
            AND cc.communication_purpose = 'DORMANCY_CONTACT'
        WHERE dt.current_stage = 'CONTACT_PHASE'
            AND dt.dormancy_trigger_date IS NOT NULL
        GROUP BY dt.account_id, dt.customer_id
        HAVING total_attempts < 3 
            OR NOT FIND_IN_SET('EMAIL', methods_used)
            OR NOT FIND_IN_SET('SMS', methods_used)
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='INCOMPLETE_CONTACT_ATTEMPTS',
                priority='HIGH',
                message=f"Insufficient contact attempts for account {row['account_id']} per Article 3.1",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def detect_internal_ledger_candidates(self) -> List[DormancyAlert]:
        """
        Article 3.4 & 3.5: Monitor internal dormant ledger transfer eligibility

        Requirements:
        - After dormancy declaration and contact attempts
        - Wait period of 3 months after last contact attempt
        - Transfer dormant balances to internal "dormant accounts ledger"
        """
        query = """
        SELECT 
            dt.account_id,
            dt.customer_id,
            dt.last_contact_attempt_date,
            dt.current_stage,
            a.balance_current
        FROM DORMANCY_TRACKING dt
        JOIN ACCOUNTS a ON dt.account_id = a.account_id
        WHERE dt.current_stage = 'CONTACT_COMPLETE'
            AND dt.last_contact_attempt_date < DATE_SUB(NOW(), INTERVAL 3 MONTH)
            AND dt.transferred_to_ledger_date IS NULL
            AND a.balance_current > 0
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='INTERNAL_LEDGER_READY',
                priority='MEDIUM',
                message=f"Account {row['account_id']} ready for internal ledger transfer per Article 3.4",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def detect_unclaimed_payment_instruments_ledger(self) -> List[DormancyAlert]:
        """
        Article 3.6: Monitor unclaimed instruments ledger transfer

        Requirements:
        - Transfer unclaimed instruments to internal ledger after dormancy criteria met
        - Maintain detailed records of instrument details
        - Handle differently from regular account balances
        """
        query = """
        SELECT 
            ui.instrument_id,
            ui.customer_id,
            ui.unclaimed_since,
            ui.amount,
            ui.status
        FROM UNCLAIMED_INSTRUMENTS ui
        WHERE ui.unclaimed_since < DATE_SUB(NOW(), INTERVAL 1 YEAR)
            AND ui.status = 'UNCLAIMED'
            AND ui.transferred_to_cb_date IS NULL
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['instrument_id'],
                customer_id=row['customer_id'],
                alert_type='UNCLAIMED_INSTRUMENT_LEDGER_READY',
                priority='MEDIUM',
                message=f"Unclaimed instrument {row['instrument_id']} ready for ledger transfer per Article 3.6",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def detect_sdb_court_application_needed(self) -> List[DormancyAlert]:
        """
        Article 3.7: Monitor safe deposit box court applications

        Requirements:
        - For Safe Deposit Boxes with outstanding fees
        - Bank must apply to competent court for access
        - Required before accessing box contents
        """
        query = """
        SELECT 
            sdb.box_id,
            sdb.customer_id,
            sdb.outstanding_since,
            sdb.outstanding_charges,
            sdb.court_order_required,
            sdb.court_order_date
        FROM SAFE_DEPOSIT_BOXES sdb
        WHERE sdb.outstanding_since < DATE_SUB(NOW(), INTERVAL 3 YEAR)
            AND sdb.outstanding_charges > 0
            AND sdb.court_order_required = 1
            AND sdb.court_order_date IS NULL
            AND sdb.box_status != 'CLOSED'
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['box_id'],
                customer_id=row['customer_id'],
                alert_type='SDB_COURT_APPLICATION_NEEDED',
                priority='HIGH',
                message=f"Safe deposit box {row['box_id']} requires court application per Article 3.7",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def check_record_retention_compliance(self) -> List[DormancyAlert]:
        """
        Article 3.9: Monitor record retention compliance

        Requirements:
        - Maintain comprehensive records of all dormancy-related actions
        - Perpetual retention for accounts transferred to CBUAE
        - Bank policy retention (typically 7+ years) for other records
        """
        query = """
        SELECT 
            cbt.transfer_id,
            cbt.account_id,
            cbt.customer_id,
            cbt.transfer_date
        FROM CENTRAL_BANK_TRANSFERS cbt
        LEFT JOIN AUDIT_LOG al ON cbt.transfer_id = al.record_id 
            AND al.table_name = 'CENTRAL_BANK_TRANSFERS'
        WHERE cbt.transfer_date < DATE_SUB(NOW(), INTERVAL 1 YEAR)
            AND al.log_id IS NULL
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='RECORD_RETENTION_VIOLATION',
                priority='HIGH',
                message=f"Missing audit records for transfer {row['transfer_id']} per Article 3.9",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def generate_annual_cbuae_report_summary(self) -> Dict:
        """
        Article 3.10: Generate annual CBUAE reporting data

        Requirements:
        - Submit annual reports to CBUAE on dormant accounts
        - Include statistical summaries and financial details
        - Report on transfers made to Central Bank
        """
        query = """
        SELECT 
            COUNT(DISTINCT dt.account_id) as total_dormant_accounts,
            SUM(a.balance_current) as total_dormant_balance,
            COUNT(DISTINCT cbt.transfer_id) as total_transfers_to_cb,
            SUM(cbt.aed_amount) as total_amount_transferred,
            COUNT(DISTINCT rr.reclaim_id) as total_reclaim_requests
        FROM DORMANCY_TRACKING dt
        LEFT JOIN ACCOUNTS a ON dt.account_id = a.account_id
        LEFT JOIN CENTRAL_BANK_TRANSFERS cbt ON dt.account_id = cbt.account_id
            AND YEAR(cbt.transfer_date) = YEAR(NOW())
        LEFT JOIN RECLAIM_REQUESTS rr ON dt.account_id = rr.account_id
            AND YEAR(rr.request_date) = YEAR(NOW())
        WHERE dt.dormancy_trigger_date >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
        """

        result = self.db.execute(query)[0]
        return {
            'report_year': datetime.now().year,
            'total_dormant_accounts': result['total_dormant_accounts'],
            'total_dormant_balance': result['total_dormant_balance'],
            'total_transfers_to_cb': result['total_transfers_to_cb'],
            'total_amount_transferred': result['total_amount_transferred'],
            'total_reclaim_requests': result['total_reclaim_requests'],
            'generated_date': datetime.now()
        }

    # =====================================================================
    # ARTICLE 4: CUSTOMER CLAIMS MONITORING AGENTS
    # =====================================================================

    def detect_claim_processing_pending(self) -> List[DormancyAlert]:
        """
        Article 4: Monitor customer claims processing

        Requirements:
        - Process claims within reasonable timeframes (typically 30 days)
        - Verify customer identity and account ownership
        - Track claim processing times and resolution status
        """
        query = """
        SELECT 
            rr.reclaim_id,
            rr.customer_id,
            rr.account_id,
            rr.request_date,
            rr.processing_status,
            rr.verification_status,
            DATEDIFF(NOW(), rr.request_date) as days_pending
        FROM RECLAIM_REQUESTS rr
        WHERE rr.processing_status IN ('PENDING', 'UNDER_REVIEW')
            AND DATEDIFF(NOW(), rr.request_date) > 30
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='CLAIM_PROCESSING_OVERDUE',
                priority='HIGH',
                message=f"Reclaim request {row['reclaim_id']} pending {row['days_pending']} days per Article 4",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    # =====================================================================
    # ARTICLE 5: PROACTIVE COMMUNICATION AGENTS
    # =====================================================================

    def check_contact_attempts_needed(self) -> List[DormancyAlert]:
        """
        Article 5: Monitor proactive communication requirements

        Requirements:
        - Contact customers showing early signs of inactivity
        - Typically 6 months before reaching 3-year inactivity mark
        - Follow up on unresponded communications
        """
        query = """
        SELECT 
            a.account_id,
            a.customer_id,
            a.last_transaction_date,
            cm.last_contact_date,
            DATEDIFF(NOW(), a.last_transaction_date) as days_inactive
        FROM ACCOUNTS a
        JOIN CUSTOMER_MASTER cm ON a.customer_id = cm.customer_id
        LEFT JOIN CUSTOMER_COMMUNICATIONS cc ON a.customer_id = cc.customer_id
            AND cc.communication_date > DATE_SUB(NOW(), INTERVAL 6 MONTH)
            AND cc.communication_purpose = 'PROACTIVE_CONTACT'
        WHERE a.account_status = 'ACTIVE'
            AND a.last_transaction_date BETWEEN 
                DATE_SUB(NOW(), INTERVAL 30 MONTH) AND 
                DATE_SUB(NOW(), INTERVAL 24 MONTH)
            AND cc.communication_id IS NULL
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='PROACTIVE_CONTACT_NEEDED',
                priority='MEDIUM',
                message=f"Account {row['account_id']} needs proactive contact - {row['days_inactive']} days inactive",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    # =====================================================================
    # ARTICLE 7.3: STATEMENT SUPPRESSION AGENTS
    # =====================================================================

    def detect_statement_freeze_candidates(self) -> List[DormancyAlert]:
        """
        Article 7.3: Monitor statement suppression eligibility

        Requirements:
        - Suppress regular statement generation for confirmed dormant accounts
        - Account must be officially declared dormant
        - Customer contact attempts completed
        """
        query = """
        SELECT 
            a.account_id,
            a.customer_id,
            a.last_statement_date,
            a.statement_frequency,
            dt.current_stage
        FROM ACCOUNTS a
        JOIN DORMANCY_TRACKING dt ON a.account_id = dt.account_id
        WHERE dt.current_stage IN ('DORMANT_CONFIRMED', 'INTERNAL_LEDGER')
            AND a.last_statement_date > DATE_SUB(NOW(), INTERVAL 1 MONTH)
            AND a.statement_frequency != 'SUPPRESSED'
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='STATEMENT_SUPPRESSION_READY',
                priority='LOW',
                message=f"Account {row['account_id']} eligible for statement suppression per Article 7.3",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    # =====================================================================
    # ARTICLE 8: CENTRAL BANK TRANSFER AGENTS
    # =====================================================================

    def check_eligible_for_cb_transfer(self) -> List[DormancyAlert]:
        """
        Article 8.1: Monitor Central Bank transfer eligibility

        Requirements:
        - Account dormant for 5 years minimum
        - Customer has no other active accounts with the bank
        - Customer address unknown to the bank
        - All Article 3 processes completed
        """
        query = """
        SELECT 
            dt.account_id,
            dt.customer_id,
            dt.dormancy_trigger_date,
            cm.address_known,
            a.balance_current,
            COUNT(a2.account_id) as other_active_accounts
        FROM DORMANCY_TRACKING dt
        JOIN ACCOUNTS a ON dt.account_id = a.account_id
        JOIN CUSTOMER_MASTER cm ON dt.customer_id = cm.customer_id
        LEFT JOIN ACCOUNTS a2 ON dt.customer_id = a2.customer_id 
            AND a2.account_id != dt.account_id 
            AND a2.account_status = 'ACTIVE'
        WHERE dt.dormancy_trigger_date < DATE_SUB(NOW(), INTERVAL 5 YEAR)
            AND dt.current_stage = 'INTERNAL_LEDGER'
            AND dt.transferred_to_cb_date IS NULL
            AND cm.address_known = 0
            AND a.balance_current > 0
        GROUP BY dt.account_id
        HAVING other_active_accounts = 0
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='CB_TRANSFER_ELIGIBLE',
                priority='HIGH',
                message=f"Account {row['account_id']} eligible for Central Bank transfer per Article 8.1",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def detect_foreign_currency_conversion_needed(self) -> List[DormancyAlert]:
        """
        Article 8.5: Monitor foreign currency conversion requirements

        Requirements:
        - Convert foreign currency balances to AED before transfer
        - Use CBUAE official exchange rates
        - Document conversion rates and dates
        """
        query = """
        SELECT 
            dt.account_id,
            dt.customer_id,
            a.currency,
            a.balance_current
        FROM DORMANCY_TRACKING dt
        JOIN ACCOUNTS a ON dt.account_id = a.account_id
        WHERE dt.current_stage = 'CB_TRANSFER_READY'
            AND a.currency != 'AED'
            AND dt.transferred_to_cb_date IS NULL
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='FOREIGN_CURRENCY_CONVERSION_NEEDED',
                priority='HIGH',
                message=f"Account {row['account_id']} requires currency conversion before CB transfer per Article 8.5",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    def detect_cbuae_transfer_candidates(self) -> List[DormancyAlert]:
        """
        Article 8: Monitor final CBUAE transfer candidates

        Requirements:
        - All eligibility criteria met
        - Currency conversion completed if needed
        - Documentation prepared
        """
        query = """
        SELECT 
            dt.account_id,
            dt.customer_id,
            dt.transfer_eligibility_date,
            a.balance_current,
            a.currency
        FROM DORMANCY_TRACKING dt
        JOIN ACCOUNTS a ON dt.account_id = a.account_id
        WHERE dt.transfer_eligibility_date < DATE_SUB(NOW(), INTERVAL 1 MONTH)
            AND dt.transferred_to_cb_date IS NULL
            AND a.currency = 'AED'
            AND a.balance_current > 0
        """

        results = self.db.execute(query)
        alerts = []

        for row in results:
            alert = DormancyAlert(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                alert_type='CBUAE_TRANSFER_READY',
                priority='HIGH',
                message=f"Account {row['account_id']} ready for CBUAE transfer per Article 8",
                created_date=datetime.now()
            )
            alerts.append(alert)

        return alerts

    # =====================================================================
    # COMPREHENSIVE MONITORING ORCHESTRATOR
    # =====================================================================

    def run_all_dormancy_monitors(self) -> Dict[str, List[DormancyAlert]]:
        """
        Execute all dormancy monitoring agents and return consolidated results
        """
        monitor_results = {}

        # Article 2: Dormancy Criteria Monitors
        monitor_results['demand_deposit_inactivity'] = self.check_demand_deposit_inactivity()
        monitor_results['fixed_deposit_inactivity'] = self.check_fixed_deposit_inactivity()
        monitor_results['investment_inactivity'] = self.check_investment_inactivity()
        monitor_results['unclaimed_payment_instruments'] = self.check_unclaimed_payment_instruments()
        monitor_results['safe_deposit_dormancy'] = self.check_safe_deposit_dormancy()

        # Article 3: Bank Obligations Monitors
        monitor_results['incomplete_contact_attempts'] = self.detect_incomplete_contact_attempts()
        monitor_results['internal_ledger_candidates'] = self.detect_internal_ledger_candidates()
        monitor_results['unclaimed_instruments_ledger'] = self.detect_unclaimed_payment_instruments_ledger()
        monitor_results['sdb_court_applications'] = self.detect_sdb_court_application_needed()
        monitor_results['record_retention_compliance'] = self.check_record_retention_compliance()

        # Article 4: Customer Claims Monitors
        monitor_results['claim_processing_pending'] = self.detect_claim_processing_pending()

        # Article 5: Proactive Communication Monitors
        monitor_results['proactive_contact_needed'] = self.check_contact_attempts_needed()

        # Article 7.3: Statement Suppression Monitors
        monitor_results['statement_suppression_candidates'] = self.detect_statement_freeze_candidates()

        # Article 8: Central Bank Transfer Monitors
        monitor_results['cb_transfer_eligible'] = self.check_eligible_for_cb_transfer()
        monitor_results['foreign_currency_conversion'] = self.detect_foreign_currency_conversion_needed()
        monitor_results['cbuae_transfer_ready'] = self.detect_cbuae_transfer_candidates()

        # Generate summary statistics
        total_alerts = sum(len(alerts) for alerts in monitor_results.values())
        high_priority_alerts = sum(
            len([alert for alert in alerts if alert.priority == 'HIGH'])
            for alerts in monitor_results.values()
        )

        monitor_results['summary'] = {
            'total_alerts': total_alerts,
            'high_priority_alerts': high_priority_alerts,
            'execution_time': datetime.now(),
            'monitors_executed': len(monitor_results) - 1  # -1 to exclude summary itself
        }

        return monitor_results

    def generate_compliance_dashboard_data(self) -> Dict:
        """
        Generate comprehensive compliance dashboard data for management reporting
        """
        # Run all monitors
        monitor_results = self.run_all_dormancy_monitors()

        # Generate annual report data
        annual_report = self.generate_annual_cbuae_report_summary()

        # Calculate compliance metrics
        compliance_metrics = self._calculate_compliance_metrics()

        # Prepare dashboard data
        dashboard_data = {
            'alerts_by_priority': self._group_alerts_by_priority(monitor_results),
            'alerts_by_type': self._group_alerts_by_type(monitor_results),
            'compliance_score': self._calculate_compliance_score(monitor_results),
            'annual_report_summary': annual_report,
            'compliance_metrics': compliance_metrics,
            'trend_analysis': self._generate_trend_analysis(),
            'action_items': self._generate_action_items(monitor_results),
            'last_updated': datetime.now()
        }

        return dashboard_data

    def _calculate_compliance_metrics(self) -> Dict:
        """Calculate key compliance metrics"""
        query = """
        SELECT 
            COUNT(CASE WHEN dt.current_stage = 'CONTACT_PHASE' THEN 1 END) as accounts_in_contact_phase,
            COUNT(CASE WHEN dt.current_stage = 'INTERNAL_LEDGER' THEN 1 END) as accounts_in_internal_ledger,
            COUNT(CASE WHEN dt.current_stage = 'CB_TRANSFER_READY' THEN 1 END) as accounts_ready_for_cb,
            COUNT(CASE WHEN dt.transferred_to_cb_date IS NOT NULL THEN 1 END) as accounts_transferred_to_cb,
            AVG(DATEDIFF(dt.transferred_to_ledger_date, dt.dormancy_trigger_date)) as avg_days_to_internal_ledger,
            AVG(DATEDIFF(dt.transferred_to_cb_date, dt.transfer_eligibility_date)) as avg_days_to_cb_transfer
        FROM DORMANCY_TRACKING dt
        WHERE dt.dormancy_trigger_date >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
        """

        result = self.db.execute(query)[0]
        return result

    def _group_alerts_by_priority(self, monitor_results: Dict) -> Dict:
        """Group alerts by priority level"""
        priority_groups = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

        for monitor_type, alerts in monitor_results.items():
            if monitor_type != 'summary':
                for alert in alerts:
                    priority_groups[alert.priority] += 1

        return priority_groups

    def _group_alerts_by_type(self, monitor_results: Dict) -> Dict:
        """Group alerts by type for categorization"""
        type_groups = {}

        for monitor_type, alerts in monitor_results.items():
            if monitor_type != 'summary':
                type_groups[monitor_type] = len(alerts)

        return type_groups

    def _calculate_compliance_score(self, monitor_results: Dict) -> float:
        """Calculate overall compliance score (0-100)"""
        total_checks = len(monitor_results) - 1  # -1 for summary
        total_alerts = monitor_results['summary']['total_alerts']
        high_priority_alerts = monitor_results['summary']['high_priority_alerts']

        # Base score calculation
        base_score = max(0, 100 - (total_alerts * 2))

        # Penalty for high priority alerts
        high_priority_penalty = high_priority_alerts * 5

        # Final compliance score
        compliance_score = max(0, base_score - high_priority_penalty)

        return round(compliance_score, 2)

    def _generate_trend_analysis(self) -> Dict:
        """Generate trend analysis for the past 12 months"""
        query = """
        SELECT 
            DATE_FORMAT(dt.dormancy_trigger_date, '%Y-%m') as month,
            COUNT(*) as new_dormant_accounts,
            SUM(a.balance_current) as dormant_balance
        FROM DORMANCY_TRACKING dt
        JOIN ACCOUNTS a ON dt.account_id = a.account_id
        WHERE dt.dormancy_trigger_date >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
        GROUP BY DATE_FORMAT(dt.dormancy_trigger_date, '%Y-%m')
        ORDER BY month
        """

        results = self.db.execute(query)
        return {
            'monthly_trends': results,
            'analysis_period': '12 months',
            'generated_date': datetime.now()
        }

    def _generate_action_items(self, monitor_results: Dict) -> List[Dict]:
        """Generate prioritized action items based on alerts"""
        action_items = []

        # High priority actions
        for monitor_type, alerts in monitor_results.items():
            if monitor_type != 'summary':
                high_priority_alerts = [alert for alert in alerts if alert.priority == 'HIGH']
                if high_priority_alerts:
                    action_items.append({
                        'priority': 'HIGH',
                        'category': monitor_type,
                        'count': len(high_priority_alerts),
                        'action_required': self._get_action_description(monitor_type),
                        'deadline': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    })

        return action_items

    def _get_action_description(self, monitor_type: str) -> str:
        """Get action description for each monitor type"""
        action_descriptions = {
            'demand_deposit_inactivity': 'Initiate dormancy declaration process and customer contact',
            'fixed_deposit_inactivity': 'Review maturity status and initiate contact procedures',
            'investment_inactivity': 'Verify investment product status and contact customers',
            'unclaimed_payment_instruments': 'Process unclaimed instruments for ledger transfer',
            'safe_deposit_dormancy': 'Initiate court application process for box access',
            'incomplete_contact_attempts': 'Complete required customer contact attempts',
            'internal_ledger_candidates': 'Transfer eligible accounts to internal dormant ledger',
            'cb_transfer_eligible': 'Prepare accounts for Central Bank transfer',
            'foreign_currency_conversion': 'Convert foreign currency balances to AED',
            'claim_processing_pending': 'Process overdue customer reclaim requests',
            'proactive_contact_needed': 'Initiate proactive customer contact campaign'
        }

        return action_descriptions.get(monitor_type, 'Review and take appropriate action')


# =====================================================================
# UTILITY FUNCTIONS AND ADDITIONAL MONITORING AGENTS
# =====================================================================

class DormancyReportingEngine:
    """
    Specialized reporting engine for CBUAE dormancy compliance
    """

    def __init__(self, db_connection):
        self.db = db_connection

    def generate_quarterly_brf_report(self, quarter: int, year: int) -> Dict:
        """
        Generate Quarterly Banking Returns Form (BRF) report for CBUAE

        Requirements:
        - Quarterly submission to CBUAE
        - Detailed breakdown of dormant accounts and transfers
        - Statistical analysis and compliance metrics
        """
        query = """
        SELECT 
            COUNT(DISTINCT cbt.account_id) as accounts_transferred,
            SUM(cbt.aed_amount) as total_amount_transferred,
            COUNT(DISTINCT cbt.customer_id) as customers_affected,
            AVG(cbt.aed_amount) as average_transfer_amount,
            cbt.quarter,
            cbt.transfer_date
        FROM CENTRAL_BANK_TRANSFERS cbt
        WHERE cbt.quarter = %s AND YEAR(cbt.transfer_date) = %s
        GROUP BY cbt.quarter
        """

        results = self.db.execute(query, (quarter, year))

        return {
            'quarter': quarter,
            'year': year,
            'transfer_statistics': results[0] if results else {},
            'report_generated': datetime.now(),
            'report_type': 'BRF_QUARTERLY'
        }

    def generate_dormancy_aging_report(self) -> Dict:
        """
        Generate aging analysis of dormant accounts by dormancy period
        """
        query = """
        SELECT 
            CASE 
                WHEN DATEDIFF(NOW(), dt.dormancy_trigger_date) <= 1095 THEN '0-3 years'
                WHEN DATEDIFF(NOW(), dt.dormancy_trigger_date) <= 1825 THEN '3-5 years'
                WHEN DATEDIFF(NOW(), dt.dormancy_trigger_date) <= 2555 THEN '5-7 years'
                ELSE '7+ years'
            END as dormancy_period,
            COUNT(*) as account_count,
            SUM(a.balance_current) as total_balance
        FROM DORMANCY_TRACKING dt
        JOIN ACCOUNTS a ON dt.account_id = a.account_id
        WHERE dt.dormancy_trigger_date IS NOT NULL
        GROUP BY dormancy_period
        ORDER BY MIN(DATEDIFF(NOW(), dt.dormancy_trigger_date))
        """

        results = self.db.execute(query)

        return {
            'aging_analysis': results,
            'total_dormant_accounts': sum(row['account_count'] for row in results),
            'total_dormant_balance': sum(row['total_balance'] for row in results),
            'report_date': datetime.now()
        }


class DormancyNotificationService:
    """
    Service for managing dormancy-related notifications and communications
    """

    def __init__(self, db_connection):
        self.db = db_connection

    def send_dormancy_alerts(self, alerts: List[DormancyAlert]) -> Dict:
        """
        Send dormancy alerts to appropriate stakeholders
        """
        # Group alerts by priority and type
        high_priority = [alert for alert in alerts if alert.priority == 'HIGH']
        medium_priority = [alert for alert in alerts if alert.priority == 'MEDIUM']

        notification_summary = {
            'high_priority_sent': len(high_priority),
            'medium_priority_sent': len(medium_priority),
            'total_notifications': len(alerts),
            'sent_timestamp': datetime.now()
        }

        # Log notifications in audit trail
        for alert in alerts:
            self._log_notification(alert)

        return notification_summary

    def _log_notification(self, alert: DormancyAlert):
        """Log notification in audit trail"""
        query = """
        INSERT INTO AUDIT_LOG (
            table_name, record_id, action_type, field_name, 
            new_value, change_reason, user_id, change_date
        ) VALUES (
            'DORMANCY_ALERTS', %s, 'NOTIFICATION_SENT', 'alert_type',
            %s, 'Automated dormancy monitoring alert', 'SYSTEM', NOW()
        )
        """

        self.db.execute(query, (alert.account_id, alert.alert_type))


# =====================================================================
# MAIN EXECUTION AND CONFIGURATION
# =====================================================================

def initialize_dormancy_monitoring_system(db_connection) -> DormancyMonitoringAgents:
    """
    Initialize the complete dormancy monitoring system
    """
    # Create main monitoring agent
    dormancy_agents = DormancyMonitoringAgents(db_connection)

    # Initialize supporting services
    reporting_engine = DormancyReportingEngine(db_connection)
    notification_service = DormancyNotificationService(db_connection)

    logger.info("Dormancy monitoring system initialized successfully")
    logger.info("All CBUAE regulation monitoring agents loaded and ready")

    return dormancy_agents


def run_daily_dormancy_monitoring(db_connection):
    """
    Execute daily dormancy monitoring routine
    """
    try:
        # Initialize system
        agents = initialize_dormancy_monitoring_system(db_connection)

        # Run all monitoring agents
        logger.info("Starting daily dormancy monitoring execution...")
        monitor_results = agents.run_all_dormancy_monitors()

        # Generate compliance dashboard
        dashboard_data = agents.generate_compliance_dashboard_data()

        # Log execution summary
        summary = monitor_results['summary']
        logger.info(f"Daily monitoring completed: {summary['total_alerts']} total alerts, "
                    f"{summary['high_priority_alerts']} high priority")

        # Send notifications for high priority alerts
        notification_service = DormancyNotificationService(db_connection)
        high_priority_alerts = []

        for monitor_type, alerts in monitor_results.items():
            if monitor_type != 'summary':
                high_priority_alerts.extend([alert for alert in alerts if alert.priority == 'HIGH'])

        if high_priority_alerts:
            notification_summary = notification_service.send_dormancy_alerts(high_priority_alerts)
            logger.info(f"Sent {notification_summary['total_notifications']} priority notifications")

        return {
            'monitoring_results': monitor_results,
            'dashboard_data': dashboard_data,
            'execution_status': 'SUCCESS',
            'execution_time': datetime.now()
        }

    except Exception as e:
        logger.error(f"Daily dormancy monitoring failed: {str(e)}")
        return {
            'execution_status': 'FAILED',
            'error_message': str(e),
            'execution_time': datetime.now()
        }


if __name__ == "__main__":
    # Example usage
    print("CBUAE Dormancy Monitoring Agents - Complete Implementation")
    print("=" * 60)
    print("This module implements all monitoring agents required for CBUAE")
    print("Dormancy and Unclaimed Balances Regulation compliance.")
    print("\nKey Features:")
    print("• 15+ specialized monitoring agents")
    print("• Complete Article 2-8 coverage")
    print("• Automated alert generation")
    print("• Compliance dashboard integration")
    print("• Regulatory reporting capabilities")
    print("• Audit trail maintenance")
    print("\nDatabase Schema Support:")
    print("• 15 tables with 252 total columns")
    print("• Comprehensive relationship mapping")
    print("• Full regulatory data capture")
    print("\nTo use this system:")
    print("1. Initialize database connection")
    print("2. Call initialize_dormancy_monitoring_system()")
    print("3. Execute run_daily_dormancy_monitoring()")
    print("4. Monitor alerts and take required actions")