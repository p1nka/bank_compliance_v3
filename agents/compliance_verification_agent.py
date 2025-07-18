from enum import Enum
import os
import csv
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class GroqComplianceLLM:
    """Enhanced Groq API client for compliance-specific AI analysis"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama3-8b-8192"

        if not self.api_key:
            logger.warning("GROQ_API_KEY not found. AI compliance features will be disabled.")

    def analyze_compliance_violation(self, violation_context: str, article: str = None) -> str:
        """Analyze compliance violations with AI-powered insights"""
        if not self.api_key:
            return "AI compliance analysis unavailable - API key not configured"

        try:
            system_prompt = f"""You are a CBUAE banking compliance expert specializing in UAE banking regulations. 
            Your expertise covers dormant account management, customer contact requirements, Central Bank transfers, 
            and regulatory reporting. Provide detailed compliance analysis with specific regulatory citations and remediation steps.

            Focus on:
            - CBUAE Article compliance requirements
            - Regulatory deadlines and timelines
            - Risk assessment and mitigation
            - Specific remediation actions
            - Compliance monitoring recommendations"""

            user_prompt = f"""
            Compliance Violation Analysis Required:

            Article Context: {article or 'General Banking Compliance'}
            Violation Details: {violation_context}

            Please provide a comprehensive compliance analysis including:

            1. REGULATORY ASSESSMENT:
               - Specific CBUAE article/regulation violated
               - Severity level and regulatory implications
               - Potential penalties or sanctions

            2. IMMEDIATE ACTIONS:
               - Critical steps to address violation
               - Timeline for remediation
               - Required documentation

            3. RISK MITIGATION:
               - Operational risks and controls
               - Regulatory risks and monitoring
               - Reputational risk management

            4. COMPLIANCE MONITORING:
               - Ongoing monitoring requirements
               - Reporting obligations
               - Key performance indicators

            5. PREVENTION MEASURES:
               - Process improvements
               - Control enhancements
               - Training requirements

            Provide specific, actionable recommendations with regulatory justification.
            """

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return "No compliance analysis generated"

        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
            return f"Compliance analysis unavailable due to API error: {str(e)}"
        except Exception as e:
            logger.error(f"Error in compliance analysis: {e}")
            return f"Compliance analysis unavailable: {str(e)}"

    def generate_compliance_summary(self, violations: List[Dict], overall_status: str) -> str:
        """Generate comprehensive compliance summary using AI"""
        if not self.api_key:
            return "AI compliance summary unavailable - API key not configured"

        try:
            violations_text = json.dumps(violations, indent=2, default=str)

            system_prompt = """You are a senior compliance officer preparing executive summaries for banking management. 
            Create concise, actionable compliance reports that highlight key risks, required actions, and strategic recommendations."""

            user_prompt = f"""
            Compliance Assessment Summary Required:

            Overall Compliance Status: {overall_status}
            Violations Identified: {violations_text}

            Create an executive compliance summary with:

            1. EXECUTIVE OVERVIEW (2-3 sentences)
            2. KEY VIOLATIONS & RISKS (prioritized list)
            3. IMMEDIATE ACTIONS REQUIRED (with timelines)
            4. STRATEGIC RECOMMENDATIONS (process improvements)
            5. COMPLIANCE SCORE & TREND

            Keep the summary professional, concise, and action-oriented.
            """

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.2,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return "No compliance summary generated"

        except Exception as e:
            logger.error(f"Error generating compliance summary: {e}")
            return f"Compliance summary unavailable: {str(e)}"


# Base Compliance Agent Class
class ComplianceAgent:
    def __init__(self):
        self.llm_client = GroqComplianceLLM()

    def get_llm_recommendation(self, context: str, article: str = None) -> str:
        """Get AI-generated compliance recommendation"""
        return self.llm_client.analyze_compliance_violation(context, article)

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
            'ai_analysis': None,
            'regulatory_citation': 'CBUAE Article 2 - Dormant Account Identification'
        }

        # Check dormancy classification compliance
        dormancy_status = account_data.get('dormancy_status', '')
        last_transaction_date = account_data.get('last_transaction_date')
        dormancy_trigger_date = account_data.get('dormancy_trigger_date')

        if last_transaction_date:
            if isinstance(last_transaction_date, str):
                last_transaction_date = datetime.fromisoformat(last_transaction_date)

            days_inactive = (datetime.now() - last_transaction_date).days

            # Violation: Account should be dormant but not classified
            if days_inactive >= 365 and dormancy_status != 'dormant':
                violation = "Account meets dormancy criteria but not classified as dormant"
                result['violations'].append(violation)
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value
                result['risk_level'] = RiskLevel.HIGH.value

            # Violation: Missing dormancy trigger date
            if dormancy_status == 'dormant' and not dormancy_trigger_date:
                violation = "Dormant account missing trigger date"
                result['violations'].append(violation)
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value
                result['risk_level'] = RiskLevel.MEDIUM.value

        if result['violations']:
            result['action'] = "Update dormancy classification and trigger dates"

            context = f"""
            CBUAE Article 2 Compliance Violation Analysis:

            Account Details:
            - Account ID: {account_data.get('account_id', 'unknown')}
            - Days Inactive: {days_inactive if 'days_inactive' in locals() else 'unknown'}
            - Current Dormancy Status: {dormancy_status}
            - Trigger Date Present: {bool(dormancy_trigger_date)}

            Violations Identified:
            {chr(10).join(f"- {v}" for v in result['violations'])}

            This relates to CBUAE Article 2 requirements for proper dormant account identification and classification.
            """

            result['ai_analysis'] = self.get_llm_recommendation(context, "CBUAE Article 2")

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
            'ai_analysis': None,
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
                violation = f"Insufficient contact attempts: {contact_attempts} of {required_attempts} required"
                result['violations'].append(violation)
                result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                result['priority'] = Priority.HIGH.value
                result['risk_level'] = RiskLevel.HIGH.value
                result['action'] = f"Complete {required_attempts - contact_attempts} additional contact attempts"

            # Check contact method diversity
            last_contact_method = account_data.get('last_contact_method', '')
            if contact_attempts > 0 and len(last_contact_method.split(',')) < 2:
                violation = "Contact attempts must use diverse methods (email, phone, letter)"
                result['violations'].append(violation)
                result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
                result['priority'] = Priority.MEDIUM.value

        if result['violations']:
            context = f"""
            CBUAE Article 3.1 Customer Contact Compliance Analysis:

            Account Profile:
            - Customer Type: {customer_type}
            - Account Value: {account_value:,.2f} AED
            - Required Contact Attempts: {required_attempts if 'required_attempts' in locals() else 'N/A'}
            - Completed Attempts: {contact_attempts}
            - Contact Methods Used: {last_contact_method or 'Not specified'}

            Compliance Violations:
            {chr(10).join(f"- {v}" for v in result['violations'])}

            This relates to CBUAE Article 3.1 mandatory customer contact requirements before dormancy transfer.
            """

            result['ai_analysis'] = self.get_llm_recommendation(context, "CBUAE Article 3.1")

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
            'ai_analysis': None,
            'regulatory_citation': 'CBUAE Article 3.4 - Central Bank Transfer Requirements'
        }

        dormancy_trigger_date = account_data.get('dormancy_trigger_date')
        transfer_eligibility_date = account_data.get('transfer_eligibility_date')
        transferred_to_cb_date = account_data.get('transferred_to_cb_date')
        contact_attempts = account_data.get('contact_attempts_made', 0)

        if dormancy_trigger_date:
            if isinstance(dormancy_trigger_date, str):
                dormancy_trigger_date = datetime.fromisoformat(dormancy_trigger_date)

            dormancy_days = (datetime.now() - dormancy_trigger_date).days

            # Check if eligible for CB transfer (2+ years dormant + contact completed)
            if dormancy_days >= 730 and contact_attempts >= 3:
                if not transfer_eligibility_date:
                    violation = "Account eligible for CB transfer but eligibility date not set"
                    result['violations'].append(violation)
                    result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
                    result['priority'] = Priority.HIGH.value
                    result['risk_level'] = RiskLevel.HIGH.value

                elif transfer_eligibility_date and not transferred_to_cb_date:
                    if isinstance(transfer_eligibility_date, str):
                        transfer_eligibility_date = datetime.fromisoformat(transfer_eligibility_date)

                    eligibility_days = (datetime.now() - transfer_eligibility_date).days
                    if eligibility_days > 90:  # 3 months overdue
                        violation = f"CB transfer overdue by {eligibility_days - 90} days"
                        result['violations'].append(violation)
                        result['compliance_status'] = ComplianceStatus.CRITICAL_VIOLATION.value
                        result['priority'] = Priority.CRITICAL.value
                        result['risk_level'] = RiskLevel.CRITICAL.value
                        result['action'] = "Immediately initiate CB transfer process"

        if result['violations']:
            context = f"""
            CBUAE Article 3.4 Central Bank Transfer Compliance Analysis:

            Transfer Timeline:
            - Dormancy Days: {dormancy_days if 'dormancy_days' in locals() else 'unknown'}
            - Contact Attempts Completed: {contact_attempts}
            - Transfer Eligibility Date: {transfer_eligibility_date or 'Not set'}
            - Actual Transfer Date: {transferred_to_cb_date or 'Not transferred'}
            - Days Overdue: {eligibility_days - 90 if 'eligibility_days' in locals() and eligibility_days > 90 else 0}

            Compliance Violations:
            {chr(10).join(f"- {v}" for v in result['violations'])}

            This relates to CBUAE Article 3.4 mandatory Central Bank transfer requirements and timelines.
            """

            result['ai_analysis'] = self.get_llm_recommendation(context, "CBUAE Article 3.4")

        return result


# Additional compliance agents following the same enhanced pattern...

# Agent 4: Documentation Compliance with AI Analysis
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
            'ai_analysis': None,
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
            violation = f"Missing required documentation: {missing_docs}"
            result['violations'].append(violation)
            result['compliance_status'] = ComplianceStatus.NON_COMPLIANT.value
            result['priority'] = Priority.MEDIUM.value
            result['risk_level'] = RiskLevel.MEDIUM.value

        # Check statement frequency compliance
        statement_frequency = account_data.get('statement_frequency', '')
        last_statement_date = account_data.get('last_statement_date')

        if statement_frequency and last_statement_date:
            freq_days = {'monthly': 30, 'quarterly': 90, 'annual': 365}.get(statement_frequency, 30)
            if isinstance(last_statement_date, str):
                last_statement_date = datetime.fromisoformat(last_statement_date)
            days_since_statement = (datetime.now() - last_statement_date).days

            if days_since_statement > freq_days * 1.2:  # 20% tolerance
                violation = f"Statement overdue by {days_since_statement - freq_days} days"
                result['violations'].append(violation)
                result['compliance_status'] = ComplianceStatus.PARTIAL_COMPLIANT.value
                result['priority'] = Priority.LOW.value

        if result['violations']:
            result['action'] = "Complete missing documentation requirements"

            context = f"""
            CBUAE Documentation Compliance Analysis:

            Documentation Status:
            - Missing Required Documents: {missing_docs}
            - Statement Frequency: {statement_frequency}
            - Days Since Last Statement: {days_since_statement if 'days_since_statement' in locals() else 'unknown'}

            Compliance Violations:
            {chr(10).join(f"- {v}" for v in result['violations'])}

            This relates to CBUAE documentation and record-keeping requirements for dormant accounts.
            """

            result['ai_analysis'] = self.get_llm_recommendation(context, "CBUAE Documentation Standards")

        return result


# Enhanced Compliance Orchestrator with AI Integration
class ComplianceOrchestrator:
    def __init__(self):
        self.agents = {
            'article_2_compliance': Article2ComplianceAgent(),
            'article_3_1_compliance': Article31ComplianceAgent(),
            'article_3_4_compliance': Article34ComplianceAgent(),
            'documentation_compliance': DocumentationComplianceAgent(),
            # Add other agents...
        }
        self.llm_client = GroqComplianceLLM()

    def process_account(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        """Process account through all compliance agents with AI analysis"""
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

        # Generate overall AI compliance summary
        results['overall_ai_compliance_summary'] = self._generate_overall_compliance_summary(results)

        return results

    def _generate_overall_compliance_summary(self, results: Dict[str, Dict]) -> str:
        """Generate comprehensive AI-powered compliance summary"""
        violations = []
        compliance_statuses = []

        for agent_name, agent_result in results.items():
            if isinstance(agent_result, dict):
                status = agent_result.get('compliance_status', 'unknown')
                compliance_statuses.append(status)

                if agent_result.get('violations'):
                    violations.extend([
                        {
                            'agent': agent_name,
                            'violation': v,
                            'priority': agent_result.get('priority', 'medium'),
                            'risk_level': agent_result.get('risk_level', 'medium')
                        }
                        for v in agent_result['violations']
                    ])

        # Determine overall status
        if ComplianceStatus.CRITICAL_VIOLATION.value in compliance_statuses:
            overall_status = "CRITICAL_VIOLATIONS_FOUND"
        elif ComplianceStatus.NON_COMPLIANT.value in compliance_statuses:
            overall_status = "NON_COMPLIANT"
        elif ComplianceStatus.PARTIAL_COMPLIANT.value in compliance_statuses:
            overall_status = "PARTIAL_COMPLIANT"
        else:
            overall_status = "COMPLIANT"

        return self.llm_client.generate_compliance_summary(violations, overall_status)

    def export_to_csv(self, account_data: Dict, results: Dict[str, Dict],
                      filename: str = "enhanced_compliance_results.csv") -> None:
        """Export compliance results to CSV with AI analysis"""
        rows = []

        # Add account metadata to every agent's result
        for agent_name, agent_result in results.items():
            if agent_name == 'overall_ai_compliance_summary':
                continue

            row = {
                "account_id": account_data.get("account_id", "N/A"),
                "customer_id": account_data.get("customer_id", "N/A"),
                "account_type": account_data.get("account_type", "N/A"),
                "balance_current": account_data.get("balance_current", "N/A"),
                "dormancy_status": account_data.get("dormancy_status", "N/A"),
                "agent": agent_name,
                "ai_analysis": agent_result.get("ai_analysis", "N/A"),
                "compliance_status": agent_result.get("compliance_status", "N/A"),
                "violations": "; ".join(agent_result.get("violations", [])),
                "priority": agent_result.get("priority", "N/A"),
                "risk_level": agent_result.get("risk_level", "N/A"),
                "regulatory_citation": agent_result.get("regulatory_citation", "N/A")
            }
            rows.append(row)

        # Add overall summary row
        rows.append({
            "account_id": account_data.get("account_id", "N/A"),
            "agent": "overall_compliance_summary",
            "ai_analysis": results.get('overall_ai_compliance_summary', 'N/A'),
            "compliance_status": "SUMMARY"
        })

        # Write to CSV
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            if rows:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"Enhanced compliance results with AI analysis exported to {filename}")

    def generate_executive_report(self, results: Dict[str, Dict]) -> Dict:
        """Generate executive compliance report with AI insights"""
        summary = {
            'total_agents': len([k for k in results.keys() if k != 'overall_ai_compliance_summary']),
            'compliant_agents': 0,
            'non_compliant_agents': 0,
            'critical_violations': 0,
            'total_violations': 0,
            'overall_status': ComplianceStatus.COMPLIANT.value,
            'critical_actions': [],
            'ai_recommendations': [],
            'executive_summary': results.get('overall_ai_compliance_summary', '')
        }

        for agent_name, agent_result in results.items():
            if agent_name == 'overall_ai_compliance_summary':
                continue

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

            if agent_result.get('ai_analysis'):
                summary['ai_recommendations'].append({
                    'agent': agent_name,
                    'analysis': agent_result['ai_analysis']
                })

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

    # Generate executive report
    executive_report = orchestrator.generate_executive_report(compliance_results)

    # Export enhanced results
    orchestrator.export_to_csv(sample_account, compliance_results, "ai_enhanced_compliance_results.csv")

    # Print results with AI analysis
    print("=== AI-ENHANCED COMPLIANCE ANALYSIS RESULTS ===")
    print("=" * 60)

    print(f"\nOVERALL AI COMPLIANCE SUMMARY:")
    print(compliance_results.get('overall_ai_compliance_summary', 'N/A'))

    print(f"\nINDIVIDUAL AGENT RESULTS:")
    for agent_name, result in compliance_results.items():
        if agent_name != 'overall_ai_compliance_summary':
            print(f"\n{agent_name.upper().replace('_', ' ')}:")
            print(f"Status: {result.get('compliance_status', 'Unknown')}")
            print(f"Priority: {result.get('priority', 'None')}")
            if result.get('violations'):
                print(f"Violations: {result['violations']}")
            if result.get('ai_analysis'):
                print(f"AI Analysis: {result['ai_analysis'][:200]}...")

    print(f"\nEXECUTIVE SUMMARY:")
    print(f"Overall Status: {executive_report['overall_status']}")
    print(f"Compliant Agents: {executive_report['compliant_agents']}/{executive_report['total_agents']}")
    print(f"Total Violations: {executive_report['total_violations']}")
    print(f"Critical Violations: {executive_report['critical_violations']}")