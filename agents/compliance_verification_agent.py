import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END
from langsmith import traceable, Client as LangSmithClient

# MCP imports
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    PENDING_VERIFICATION = "pending_verification"
    PARTIALLY_COMPLIANT = "partially_compliant"


@dataclass
class ComplianceState:
    """State for compliance verification workflow"""

    session_id: str
    user_id: str
    compliance_id: str
    timestamp: datetime

    # Input data
    dormancy_results: Optional[Dict] = None
    account_data: Optional[Dict] = None
    verification_criteria: Optional[Dict] = None

    # Verification results
    compliance_results: Optional[Dict] = None
    violations: List[Dict] = None
    remediation_actions: List[Dict] = None

    # Status tracking
    verification_status: ComplianceStatus = ComplianceStatus.PENDING_VERIFICATION
    critical_violations: int = 0
    total_checks: int = 0
    compliance_score: float = 0.0

    # Memory context
    memory_context: Dict = None
    retrieved_patterns: Dict = None

    # Audit trail
    verification_log: List[Dict] = None
    error_log: List[Dict] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.remediation_actions is None:
            self.remediation_actions = []
        if self.memory_context is None:
            self.memory_context = {}
        if self.retrieved_patterns is None:
            self.retrieved_patterns = {}
        if self.verification_log is None:
            self.verification_log = []
        if self.error_log is None:
            self.error_log = []


class ComplianceVerificationAgent:
    """CBUAE compliance verification agent"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

        # CBUAE compliance rules
        self.compliance_rules = {
            "article_2_1_1": self._verify_demand_deposit_compliance,
            "article_2_2": self._verify_fixed_deposit_compliance,
            "article_2_3": self._verify_investment_compliance,
            "article_2_4": self._verify_payment_instrument_compliance,
            "article_2_6": self._verify_sdb_compliance,
            "article_3": self._verify_contact_procedures,
            "article_8": self._verify_cb_transfer_eligibility
        }

    @traceable(name="verify_compliance")
    async def verify_compliance(self, state: ComplianceState) -> ComplianceState:
        """Main compliance verification workflow"""

        try:
            start_time = datetime.now()
            state.verification_status = ComplianceStatus.PENDING_VERIFICATION

            # Execute all compliance checks
            verification_results = {}
            total_violations = 0

            for article, verify_func in self.compliance_rules.items():
                try:
                    result = await verify_func(state)
                    verification_results[article] = result

                    if result.get("violations"):
                        total_violations += len(result["violations"])
                        state.violations.extend(result["violations"])

                    if result.get("remediation_actions"):
                        state.remediation_actions.extend(result["remediation_actions"])

                    state.total_checks += 1

                except Exception as e:
                    logger.error(f"Compliance check {article} failed: {str(e)}")
                    state.error_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "article": article,
                        "error": str(e)
                    })

            # Calculate compliance score and status
            state.compliance_results = verification_results
            state.critical_violations = len([v for v in state.violations if v.get("severity") == "critical"])
            state.compliance_score = self._calculate_compliance_score(verification_results)

            # Determine overall status
            if state.critical_violations > 0:
                state.verification_status = ComplianceStatus.NON_COMPLIANT
            elif total_violations > 0:
                state.verification_status = ComplianceStatus.PARTIALLY_COMPLIANT
            elif state.compliance_score >= 0.95:
                state.verification_status = ComplianceStatus.COMPLIANT
            else:
                state.verification_status = ComplianceStatus.REQUIRES_REVIEW

            # Call MCP tool
            mcp_result = await self.mcp_client.call_tool("verify_compliance", {
                "compliance_results": verification_results,
                "violations": state.violations,
                "compliance_score": state.compliance_score,
                "user_id": state.user_id
            })

            processing_time = (datetime.now() - start_time).total_seconds()

            state.verification_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "compliance_verification",
                "status": state.verification_status.value,
                "total_checks": state.total_checks,
                "violations_found": len(state.violations),
                "critical_violations": state.critical_violations,
                "compliance_score": state.compliance_score,
                "processing_time": processing_time
            })

        except Exception as e:
            state.verification_status = ComplianceStatus.NON_COMPLIANT
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "compliance_verification",
                "error": str(e)
            })

            logger.error(f"Compliance verification failed: {str(e)}")

        return state

    async def _verify_demand_deposit_compliance(self, state: ComplianceState) -> Dict:
        """Verify Article 2.1.1 compliance for demand deposits"""

        violations = []
        remediation_actions = []

        if not state.dormancy_results:
            return {"violations": violations, "remediation_actions": remediation_actions}

        demand_deposit_data = state.dormancy_results.get("demand_deposit_dormant", {})

        if demand_deposit_data.get("count", 0) > 0:
            # Check for proper dormancy identification
            details = demand_deposit_data.get("details", {})

            # Violation: Accounts dormant without proper Article 3 process
            accounts_needing_article3 = state.dormancy_results.get("art3_process_needed", {}).get("count", 0)

            if accounts_needing_article3 > 0:
                violations.append({
                    "article": "2.1.1",
                    "violation_type": "missing_article_3_process",
                    "severity": "high",
                    "description": f"{accounts_needing_article3} demand deposit accounts require Article 3 contact process",
                    "affected_accounts": accounts_needing_article3,
                    "regulatory_requirement": "Customer contact required before dormancy classification"
                })

                remediation_actions.append({
                    "action": "initiate_article_3_process",
                    "priority": "high",
                    "description": "Contact customers for dormant demand deposit accounts",
                    "timeline": "Immediate",
                    "responsible_party": "Customer Relations Team"
                })

        return {"violations": violations, "remediation_actions": remediation_actions}

    async def _verify_fixed_deposit_compliance(self, state: ComplianceState) -> Dict:
        """Verify Article 2.2 compliance for fixed deposits"""

        violations = []
        remediation_actions = []

        if not state.dormancy_results:
            return {"violations": violations, "remediation_actions": remediation_actions}

        fixed_deposit_data = state.dormancy_results.get("fixed_deposit_dormant", {})

        if fixed_deposit_data.get("count", 0) > 0:
            violations.append({
                "article": "2.2",
                "violation_type": "fixed_deposit_dormancy",
                "severity": "medium",
                "description": f"{fixed_deposit_data['count']} fixed deposits meeting dormancy criteria",
                "affected_accounts": fixed_deposit_data["count"],
                "regulatory_requirement": "Fixed deposits dormant for 3+ years require special handling"
            })

            remediation_actions.append({
                "action": "review_fixed_deposit_maturity",
                "priority": "medium",
                "description": "Review maturity dates and renewal options for dormant fixed deposits",
                "timeline": "Within 30 days",
                "responsible_party": "Investment Operations Team"
            })

        return {"violations": violations, "remediation_actions": remediation_actions}

    async def _verify_investment_compliance(self, state: ComplianceState) -> Dict:
        """Verify Article 2.3 compliance for investment accounts"""

        violations = []
        remediation_actions = []

        investment_data = state.dormancy_results.get("investment_dormant", {})

        if investment_data.get("count", 0) > 0:
            violations.append({
                "article": "2.3",
                "violation_type": "investment_account_dormancy",
                "severity": "medium",
                "description": f"{investment_data['count']} investment accounts meeting dormancy criteria",
                "affected_accounts": investment_data["count"],
                "regulatory_requirement": "Investment accounts require customer communication verification"
            })

        return {"violations": violations, "remediation_actions": remediation_actions}

    async def _verify_payment_instrument_compliance(self, state: ComplianceState) -> Dict:
        """Verify Article 2.4 compliance for payment instruments"""

        violations = []
        remediation_actions = []

        instrument_data = state.dormancy_results.get("unclaimed_instruments", {})

        if instrument_data.get("count", 0) > 0:
            violations.append({
                "article": "2.4",
                "violation_type": "unclaimed_payment_instruments",
                "severity": "high",
                "description": f"{instrument_data['count']} unclaimed payment instruments",
                "affected_accounts": instrument_data["count"],
                "regulatory_requirement": "Bank must make efforts to contact beneficiaries"
            })

            remediation_actions.append({
                "action": "contact_instrument_beneficiaries",
                "priority": "high",
                "description": "Attempt contact with payment instrument beneficiaries",
                "timeline": "Immediate",
                "responsible_party": "Operations Team"
            })

        return {"violations": violations, "remediation_actions": remediation_actions}

    async def _verify_sdb_compliance(self, state: ComplianceState) -> Dict:
        """Verify Article 2.6 compliance for safe deposit boxes"""

        violations = []
        remediation_actions = []

        sdb_data = state.dormancy_results.get("sdb_dormant", {})

        if sdb_data.get("count", 0) > 0:
            violations.append({
                "article": "2.6",
                "violation_type": "sdb_dormancy",
                "severity": "medium",
                "description": f"{sdb_data['count']} safe deposit boxes with unpaid charges",
                "affected_accounts": sdb_data["count"],
                "regulatory_requirement": "SDB with unpaid charges >3 years require tenant contact"
            })

        return {"violations": violations, "remediation_actions": remediation_actions}

    async def _verify_contact_procedures(self, state: ComplianceState) -> Dict:
        """Verify Article 3 compliance for contact procedures"""

        violations = []
        remediation_actions = []

        article3_data = state.dormancy_results.get("art3_process_needed", {})

        if article3_data.get("count", 0) > 0:
            violations.append({
                "article": "3",
                "violation_type": "missing_contact_procedures",
                "severity": "critical",
                "description": f"{article3_data['count']} accounts require Article 3 contact process",
                "affected_accounts": article3_data["count"],
                "regulatory_requirement": "Must contact customers before classifying as dormant"
            })

            remediation_actions.append({
                "action": "execute_article_3_process",
                "priority": "critical",
                "description": "Contact customers and wait 90 days before final dormancy classification",
                "timeline": "Immediate",
                "responsible_party": "Compliance Team"
            })

        return {"violations": violations, "remediation_actions": remediation_actions}

    async def _verify_cb_transfer_eligibility(self, state: ComplianceState) -> Dict:
        """Verify Article 8 compliance for Central Bank transfers"""

        violations = []
        remediation_actions = []

        cb_transfer_data = state.dormancy_results.get("eligible_for_cb_transfer", {})

        if cb_transfer_data.get("count", 0) > 0:
            remediation_actions.append({
                "action": "prepare_cb_transfer",
                "priority": "medium",
                "description": f"Prepare {cb_transfer_data['count']} accounts for Central Bank transfer",
                "timeline": "Within 60 days",
                "responsible_party": "Treasury Operations"
            })

        return {"violations": violations, "remediation_actions": remediation_actions}

    def _calculate_compliance_score(self, verification_results: Dict) -> float:
        """Calculate overall compliance score"""

        total_articles = len(verification_results)
        compliant_articles = 0

        for article, result in verification_results.items():
            violations = result.get("violations", [])
            critical_violations = [v for v in violations if v.get("severity") == "critical"]

            if not critical_violations:
                compliant_articles += 1

        return compliant_articles / total_articles if total_articles > 0 else 0.0