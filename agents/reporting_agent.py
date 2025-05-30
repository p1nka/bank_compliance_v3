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
class ReportingState:
    """State for reporting workflow"""

    session_id: str
    user_id: str
    report_id: str
    timestamp: datetime

    # Input data
    workflow_results: Optional[Dict] = None

    # Report results
    report_results: Optional[Dict] = None
    generated_reports: List[Dict] = None

    # Memory context
    memory_context: Dict = None

    def __post_init__(self):
        if self.generated_reports is None:
            self.generated_reports = []
        if self.memory_context is None:
            self.memory_context = {}


class ReportingAgent:
    """Comprehensive reporting agent"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

    @traceable(name="generate_compliance_report")
    async def generate_compliance_report(self, state: ReportingState) -> ReportingState:
        """Generate comprehensive compliance report"""

        try:
            # Generate executive summary
            executive_summary = self._generate_executive_summary(state.workflow_results)

            # Generate detailed findings
            detailed_findings = self._generate_detailed_findings(state.workflow_results)

            # Generate action items
            action_items = self._generate_action_items(state.workflow_results)

            # Compile final report
            report = {
                "report_id": state.report_id,
                "generated_at": datetime.now().isoformat(),
                "executive_summary": executive_summary,
                "detailed_findings": detailed_findings,
                "action_items": action_items,
                "compliance_flags": self._extract_compliance_flags(state.workflow_results),
                "performance_metrics": self._extract_performance_metrics(state.workflow_results)
            }

            state.report_results = report
            state.generated_reports.append(report)

            # Call MCP tool
            mcp_result = await self.mcp_client.call_tool("generate_compliance_report", {
                "report": report,
                "user_id": state.user_id,
                "session_id": state.session_id
            })

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")

        return state

    def _generate_executive_summary(self, workflow_results: Dict) -> Dict:
        """Generate executive summary"""

        # Extract key metrics from workflow results
        data_processing = workflow_results.get("data_processing")
        dormancy_analysis = workflow_results.get("dormancy_analysis")
        compliance = workflow_results.get("compliance")
        risk_assessment = workflow_results.get("risk_assessment")

        summary = {
            "total_accounts_analyzed": getattr(data_processing, 'total_accounts_analyzed', 0) if data_processing else 0,
            "dormant_accounts_identified": getattr(dormancy_analysis, 'dormant_accounts_found',
                                                   0) if dormancy_analysis else 0,
            "high_risk_accounts": getattr(risk_assessment, 'high_risk_accounts', []) if risk_assessment else [],
            "compliance_status": getattr(compliance, 'verification_status',
                                         'unknown').value if compliance else 'unknown',
            "overall_risk_score": getattr(risk_assessment, 'overall_risk_score', 0) if risk_assessment else 0,
            "critical_violations": getattr(compliance, 'critical_violations', 0) if compliance else 0,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "dormancy_rate_percentage": 0.0
        }

        # Calculate dormancy rate
        if summary["total_accounts_analyzed"] > 0:
            summary["dormancy_rate_percentage"] = round(
                (summary["dormant_accounts_identified"] / summary["total_accounts_analyzed"]) * 100, 2
            )

        return summary

    def _generate_detailed_findings(self, workflow_results: Dict) -> Dict:
        """Generate detailed findings section"""

        findings = {
            "data_quality_analysis": {},
            "dormancy_by_type": {},
            "compliance_breakdown": {},
            "risk_analysis": {},
            "operational_insights": []
        }

        # Extract detailed data from each workflow component
        if "data_processing" in workflow_results:
            dp = workflow_results["data_processing"]
            findings["data_quality_analysis"] = {
                "quality_score": getattr(dp, 'quality_score', 0),
                "data_issues": len(getattr(dp, 'error_log', [])),
                "processing_time": getattr(dp, 'processing_time', 0)
            }

        if "dormancy_analysis" in workflow_results:
            da = workflow_results["dormancy_analysis"]
            findings["dormancy_by_type"] = getattr(da, 'dormancy_summary', {}).get('by_type', {})

        if "compliance" in workflow_results:
            comp = workflow_results["compliance"]
            findings["compliance_breakdown"] = {
                "total_violations": len(getattr(comp, 'violations', [])),
                "critical_violations": getattr(comp, 'critical_violations', 0),
                "compliance_score": getattr(comp, 'compliance_score', 0)
            }

        if "risk_assessment" in workflow_results:
            risk = workflow_results["risk_assessment"]
            findings["risk_analysis"] = {
                "overall_score": getattr(risk, 'overall_risk_score', 0),
                "risk_level": getattr(risk, 'risk_level', 'unknown').value if hasattr(getattr(risk, 'risk_level', None),
                                                                                      'value') else 'unknown',
                "high_risk_count": len(getattr(risk, 'high_risk_accounts', []))
            }

        return findings

    def _generate_action_items(self, workflow_results: Dict) -> List[Dict]:
        """Generate action items based on workflow results"""

        action_items = []

        # Compliance-based actions
        if "compliance" in workflow_results:
            comp = workflow_results["compliance"]
            remediation_actions = getattr(comp, 'remediation_actions', [])

            for action in remediation_actions:
                action_items.append({
                    "priority": action.get("priority", "medium").upper(),
                    "action": action.get("action", "Review required"),
                    "description": action.get("description", ""),
                    "timeline": action.get("timeline", "TBD"),
                    "responsible_party": action.get("responsible_party", "TBD"),
                    "category": "compliance"
                })

        # Risk-based actions
        if "risk_assessment" in workflow_results:
            risk = workflow_results["risk_assessment"]
            mitigation_strategies = getattr(risk, 'mitigation_strategies', [])

            for strategy in mitigation_strategies:
                action_items.append({
                    "priority": strategy.get("priority", "medium").upper(),
                    "action": strategy.get("strategy", "Risk mitigation required"),
                    "description": strategy.get("description", ""),
                    "timeline": strategy.get("timeline", "TBD"),
                    "responsible_party": strategy.get("responsible_team", "TBD"),
                    "category": "risk_mitigation"
                })

        # Supervisor-based actions
        if "supervisor_decisions" in workflow_results:
            decisions = workflow_results["supervisor_decisions"]
            if isinstance(decisions, dict) and "escalations" in decisions:
                for escalation in decisions["escalations"]:
                    action_items.append({
                        "priority": escalation.get("priority", "medium").upper(),
                        "action": f"Management Review - {escalation.get('type', 'General')}",
                        "description": escalation.get("reason", ""),
                        "timeline": "Immediate",
                        "responsible_party": escalation.get("level", "Management"),
                        "category": "escalation"
                    })

        return action_items

    def _extract_compliance_flags(self, workflow_results: Dict) -> List[str]:
        """Extract compliance flags from workflow results"""

        flags = []

        if "dormancy_analysis" in workflow_results:
            da = workflow_results["dormancy_analysis"]
            compliance_flags = getattr(da, 'compliance_flags', [])
            flags.extend(compliance_flags)

        if "compliance" in workflow_results:
            comp = workflow_results["compliance"]
            violations = getattr(comp, 'violations', [])
            for violation in violations:
                flags.append(f"{violation.get('article', 'Unknown')}: {violation.get('violation_type', 'Violation')}")

        return flags

    def _extract_performance_metrics(self, workflow_results: Dict) -> Dict:
        """Extract performance metrics from workflow results"""

        metrics = {
            "total_processing_time": 0,
            "agents_executed": 0,
            "memory_operations": 0,
            "errors_encountered": 0
        }

        for agent_name, agent_result in workflow_results.items():
            if hasattr(agent_result, 'processing_time'):
                metrics["total_processing_time"] += getattr(agent_result, 'processing_time', 0)

            metrics["agents_executed"] += 1

            if hasattr(agent_result, 'error_log'):
                metrics["errors_encountered"] += len(getattr(agent_result, 'error_log', []))

        return metrics