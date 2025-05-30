import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# LangGraph and LangSmith imports
from langsmith import traceable, Client as LangSmithClient

# MCP imports
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SupervisorState:
    """State for supervisor workflow"""

    session_id: str
    user_id: str
    supervision_id: str
    timestamp: datetime

    # Input data
    workflow_results: Optional[Dict] = None

    # Supervision results
    supervision_decisions: Optional[Dict] = None
    escalations: List[Dict] = None
    approvals: List[Dict] = None

    # Memory context
    memory_context: Dict = None

    def __post_init__(self):
        if self.escalations is None:
            self.escalations = []
        if self.approvals is None:
            self.approvals = []
        if self.memory_context is None:
            self.memory_context = {}


class SupervisorAgent:
    """Workflow supervision and decision-making agent"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

    @traceable(name="supervise_workflow")
    async def supervise_workflow(self, state: SupervisorState) -> SupervisorState:
        """Main supervision workflow"""

        try:
            # Analyze workflow results
            analysis = self._analyze_workflow_results(state.workflow_results)

            # Make supervision decisions
            decisions = self._make_supervision_decisions(analysis)

            # Generate escalations if needed
            escalations = self._generate_escalations(analysis, decisions)

            state.supervision_decisions = {
                "analysis": analysis,
                "decisions": decisions,
                "escalations": escalations,
                "routing_decision": self._determine_routing(decisions),
                "confidence_score": self._calculate_confidence(analysis)
            }

            state.escalations = escalations

        except Exception as e:
            logger.error(f"Supervision failed: {str(e)}")

        return state

    def _analyze_workflow_results(self, workflow_results: Dict) -> Dict:
        """Analyze overall workflow results"""

        analysis = {
            "data_quality": self._analyze_data_quality(workflow_results),
            "compliance_status": self._analyze_compliance_status(workflow_results),
            "risk_level": self._analyze_risk_level(workflow_results),
            "operational_issues": self._identify_operational_issues(workflow_results)
        }

        return analysis

    def _make_supervision_decisions(self, analysis: Dict) -> List[Dict]:
        """Make supervision decisions based on analysis"""

        decisions = []

        # Data quality decisions
        if analysis["data_quality"]["score"] < 0.7:
            decisions.append({
                "type": "data_quality_intervention",
                "action": "require_data_remediation",
                "priority": "high",
                "reason": "Data quality below acceptable threshold"
            })

        # Compliance decisions
        if analysis["compliance_status"]["critical_violations"] > 0:
            decisions.append({
                "type": "compliance_escalation",
                "action": "immediate_compliance_review",
                "priority": "critical",
                "reason": "Critical compliance violations detected"
            })

        return decisions

    def _determine_routing(self, decisions: List[Dict]) -> str:
        """Determine workflow routing based on decisions"""

        critical_decisions = [d for d in decisions if d.get("priority") == "critical"]

        if critical_decisions:
            return "escalate"
        elif any(d.get("priority") == "high" for d in decisions):
            return "requires_review"
        else:
            return "proceed_to_reporting"

    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence in supervision decisions"""

        confidence_factors = [
            analysis["data_quality"]["score"],
            1.0 - (analysis["compliance_status"]["critical_violations"] / 10),
            1.0 - (analysis["risk_level"]["score"] / 2)
        ]

        return sum(confidence_factors) / len(confidence_factors)

    def _analyze_data_quality(self, workflow_results: Dict) -> Dict:
        """Analyze data quality from workflow results"""

        data_processing = workflow_results.get("data_processing")
        if data_processing and hasattr(data_processing, 'quality_score'):
            return {
                "score": data_processing.quality_score,
                "status": "acceptable" if data_processing.quality_score >= 0.8 else "poor"
            }

        return {"score": 0.5, "status": "unknown"}

    def _analyze_compliance_status(self, workflow_results: Dict) -> Dict:
        """Analyze compliance status from workflow results"""

        compliance = workflow_results.get("compliance")
        if compliance and hasattr(compliance, 'critical_violations'):
            return {
                "critical_violations": compliance.critical_violations,
                "status": "non_compliant" if compliance.critical_violations > 0 else "compliant"
            }

        return {"critical_violations": 0, "status": "unknown"}

    def _analyze_risk_level(self, workflow_results: Dict) -> Dict:
        """Analyze risk level from workflow results"""

        risk_assessment = workflow_results.get("risk_assessment")
        if risk_assessment and hasattr(risk_assessment, 'overall_risk_score'):
            return {
                "score": risk_assessment.overall_risk_score,
                "level": risk_assessment.risk_level.value if hasattr(risk_assessment, 'risk_level') else "unknown"
            }

        return {"score": 0.0, "level": "unknown"}

    def _identify_operational_issues(self, workflow_results: Dict) -> List[Dict]:
        """Identify operational issues from workflow results"""

        issues = []

        # Check for processing errors
        for agent_name, agent_results in workflow_results.items():
            if hasattr(agent_results, 'error_log') and agent_results.error_log:
                issues.append({
                    "type": "processing_error",
                    "agent": agent_name,
                    "count": len(agent_results.error_log),
                    "severity": "medium"
                })

        return issues

    def _generate_escalations(self, analysis: Dict, decisions: List[Dict]) -> List[Dict]:
        """Generate escalations based on analysis and decisions"""

        escalations = []

        # Escalate critical compliance violations
        if analysis["compliance_status"]["critical_violations"] > 0:
            escalations.append({
                "type": "compliance_escalation",
                "level": "management",
                "reason": "Critical compliance violations require management review",
                "priority": "immediate",
                "details": analysis["compliance_status"]
            })

        # Escalate high risk situations
        if analysis["risk_level"]["score"] > 0.8:
            escalations.append({
                "type": "risk_escalation",
                "level": "risk_committee",
                "reason": "High risk score requires risk committee review",
                "priority": "urgent",
                "details": analysis["risk_level"]
            })

        return escalations