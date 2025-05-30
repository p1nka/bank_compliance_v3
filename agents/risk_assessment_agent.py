import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# LangGraph and LangSmith imports
from langsmith import traceable, Client as LangSmithClient

# MCP imports
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessmentState:
    """State for risk assessment workflow"""

    session_id: str
    user_id: str
    assessment_id: str
    timestamp: datetime

    # Input data
    dormancy_results: Optional[Dict] = None
    compliance_results: Optional[Dict] = None
    account_data: Optional[Dict] = None

    # Assessment results
    risk_assessment_results: Optional[Dict] = None
    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    high_risk_accounts: List[Dict] = None
    mitigation_strategies: List[Dict] = None

    # Memory context
    memory_context: Dict = None

    # Audit trail
    assessment_log: List[Dict] = None
    error_log: List[Dict] = None

    def __post_init__(self):
        if self.high_risk_accounts is None:
            self.high_risk_accounts = []
        if self.mitigation_strategies is None:
            self.mitigation_strategies = []
        if self.memory_context is None:
            self.memory_context = {}
        if self.assessment_log is None:
            self.assessment_log = []
        if self.error_log is None:
            self.error_log = []


class RiskAssessmentAgent:
    """Comprehensive risk assessment agent"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

        # Risk factors and weights
        self.risk_factors = {
            "high_value_dormant": 0.3,
            "compliance_violations": 0.25,
            "operational_risk": 0.2,
            "reputational_risk": 0.15,
            "regulatory_risk": 0.1
        }

    @traceable(name="assess_risk")
    async def assess_risk(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """Main risk assessment workflow"""

        try:
            start_time = datetime.now()

            # Calculate individual risk components
            risk_components = {
                "high_value_risk": self._assess_high_value_risk(state),
                "compliance_risk": self._assess_compliance_risk(state),
                "operational_risk": self._assess_operational_risk(state),
                "reputational_risk": self._assess_reputational_risk(state),
                "regulatory_risk": self._assess_regulatory_risk(state)
            }

            # Calculate overall risk score
            state.overall_risk_score = sum(
                risk_components[component] * self.risk_factors[factor]
                for component, factor in [
                    ("high_value_risk", "high_value_dormant"),
                    ("compliance_risk", "compliance_violations"),
                    ("operational_risk", "operational_risk"),
                    ("reputational_risk", "reputational_risk"),
                    ("regulatory_risk", "regulatory_risk")
                ]
            )

            # Determine risk level
            if state.overall_risk_score >= 0.8:
                state.risk_level = RiskLevel.CRITICAL
            elif state.overall_risk_score >= 0.6:
                state.risk_level = RiskLevel.HIGH
            elif state.overall_risk_score >= 0.4:
                state.risk_level = RiskLevel.MEDIUM
            else:
                state.risk_level = RiskLevel.LOW

            # Identify high-risk accounts
            state.high_risk_accounts = self._identify_high_risk_accounts(state)

            # Generate mitigation strategies
            state.mitigation_strategies = self._generate_mitigation_strategies(state, risk_components)

            # Compile results
            state.risk_assessment_results = {
                "overall_risk_score": state.overall_risk_score,
                "risk_level": state.risk_level.value,
                "risk_components": risk_components,
                "high_risk_accounts": state.high_risk_accounts,
                "mitigation_strategies": state.mitigation_strategies,
                "assessment_metadata": {
                    "factors_analyzed": len(risk_components),
                    "accounts_reviewed": len(state.high_risk_accounts),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            }

            # Call MCP tool
            mcp_result = await self.mcp_client.call_tool("assess_risk", {
                "risk_assessment": state.risk_assessment_results,
                "user_id": state.user_id,
                "session_id": state.session_id
            })

        except Exception as e:
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            logger.error(f"Risk assessment failed: {str(e)}")

        return state

    def _assess_high_value_risk(self, state: RiskAssessmentState) -> float:
        """Assess risk from high-value dormant accounts"""

        if not state.dormancy_results:
            return 0.0

        high_value_count = state.dormancy_results.get("high_value_dormant", {}).get("count", 0)
        total_dormant = state.dormancy_results.get("summary_kpis", {}).get("total_accounts_flagged_dormant", 1)

        # Risk increases with proportion of high-value accounts
        proportion = high_value_count / max(total_dormant, 1)

        # Risk score based on proportion and absolute count
        if high_value_count > 50:
            return min(1.0, 0.8 + proportion * 0.2)
        elif high_value_count > 20:
            return min(0.8, 0.6 + proportion * 0.2)
        elif high_value_count > 5:
            return min(0.6, 0.4 + proportion * 0.2)
        else:
            return proportion * 0.4

    def _assess_compliance_risk(self, state: RiskAssessmentState) -> float:
        """Assess risk from compliance violations"""

        if not state.compliance_results:
            return 0.0

        critical_violations = len([v for v in state.compliance_results.get("violations", [])
                                   if v.get("severity") == "critical"])
        high_violations = len([v for v in state.compliance_results.get("violations", [])
                               if v.get("severity") == "high"])
        total_violations = len(state.compliance_results.get("violations", []))

        # Weight violations by severity
        weighted_score = (critical_violations * 1.0 + high_violations * 0.7 +
                          (total_violations - critical_violations - high_violations) * 0.3)

        # Normalize to 0-1 scale
        return min(1.0, weighted_score / 10)

    def _assess_operational_risk(self, state: RiskAssessmentState) -> float:
        """Assess operational risk factors"""

        risk_score = 0.0

        if state.dormancy_results:
            # Risk from Article 3 process gaps
            art3_needed = state.dormancy_results.get("art3_process_needed", {}).get("count", 0)
            if art3_needed > 0:
                risk_score += min(0.5, art3_needed / 100)

            # Risk from contact attempt failures
            contact_needed = state.dormancy_results.get("proactive_contact_needed", {}).get("count", 0)
            if contact_needed > 0:
                risk_score += min(0.3, contact_needed / 200)

        return min(1.0, risk_score)

    def _assess_reputational_risk(self, state: RiskAssessmentState) -> float:
        """Assess reputational risk factors"""

        risk_score = 0.0

        if state.compliance_results:
            # Reputational risk from public compliance failures
            public_violations = [v for v in state.compliance_results.get("violations", [])
                                 if v.get("article") in ["3", "8"]]  # Customer-facing articles

            risk_score = min(1.0, len(public_violations) / 5)

        return risk_score

    def _assess_regulatory_risk(self, state: RiskAssessmentState) -> float:
        """Assess regulatory risk factors"""

        risk_score = 0.0

        if state.dormancy_results:
            # Risk from CB transfer eligibility
            cb_eligible = state.dormancy_results.get("eligible_for_cb_transfer", {}).get("count", 0)
            if cb_eligible > 0:
                risk_score += min(0.6, cb_eligible / 50)

        return min(1.0, risk_score)

    def _identify_high_risk_accounts(self, state: RiskAssessmentState) -> List[Dict]:
        """Identify individual high-risk accounts"""

        high_risk_accounts = []

        if state.dormancy_results:
            # High-value dormant accounts
            high_value_data = state.dormancy_results.get("high_value_dormant", {})
            if high_value_data.get("details", {}).get("sample_accounts"):
                for account_id in high_value_data["details"]["sample_accounts"]:
                    high_risk_accounts.append({
                        "account_id": account_id,
                        "risk_type": "high_value_dormant",
                        "risk_level": "high",
                        "description": "High-value dormant account requiring priority attention"
                    })

            # Compliance violation accounts
            if state.compliance_results:
                for violation in state.compliance_results.get("violations", []):
                    if violation.get("severity") == "critical":
                        high_risk_accounts.append({
                            "account_id": f"Multiple ({violation.get('affected_accounts', 'Unknown')})",
                            "risk_type": "compliance_violation",
                            "risk_level": "critical",
                            "description": violation.get("description", "Critical compliance violation"),
                            "article": violation.get("article")
                        })

        return high_risk_accounts[:20]  # Limit to top 20

    def _generate_mitigation_strategies(self, state: RiskAssessmentState, risk_components: Dict) -> List[Dict]:
        """Generate risk mitigation strategies"""

        strategies = []

        # High-value account strategies
        if risk_components.get("high_value_risk", 0) > 0.5:
            strategies.append({
                "strategy": "priority_reactivation_program",
                "priority": "high",
                "description": "Implement priority reactivation program for high-value dormant accounts",
                "timeline": "Immediate - 30 days",
                "expected_impact": "Reduce high-value risk by 60-80%",
                "cost_estimate": "Medium",
                "responsible_team": "Customer Relations & Risk Management"
            })

        # Compliance strategies
        if risk_components.get("compliance_risk", 0) > 0.4:
            strategies.append({
                "strategy": "compliance_remediation_plan",
                "priority": "high",
                "description": "Execute comprehensive compliance remediation plan",
                "timeline": "Immediate - 90 days",
                "expected_impact": "Achieve 95%+ compliance rate",
                "cost_estimate": "High",
                "responsible_team": "Compliance & Legal"
            })

        # Operational strategies
        if risk_components.get("operational_risk", 0) > 0.3:
            strategies.append({
                "strategy": "process_automation",
                "priority": "medium",
                "description": "Automate dormancy monitoring and customer contact processes",
                "timeline": "3-6 months",
                "expected_impact": "Reduce operational risk by 40-60%",
                "cost_estimate": "Medium",
                "responsible_team": "IT & Operations"
            })

        return strategies