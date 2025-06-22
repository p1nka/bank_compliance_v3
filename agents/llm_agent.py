# agents/llm_agent.py
"""
LLM Agent for Banking Compliance Analysis
Provides AI-powered assistance for banking compliance queries
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BankingComplianceLLMAgent:
    """
    LLM Agent for Banking Compliance Analysis
    Provides intelligent responses to banking compliance queries
    """

    def __init__(self, memory_agent=None, mcp_client=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize the banking compliance knowledge base"""
        return {
            "cbuae_articles": {
                "2.1.1": "Demand deposit accounts become dormant after 3 years of inactivity",
                "2.2": "Fixed deposit accounts become dormant after maturity + 1 year if unclaimed",
                "2.3": "Investment accounts become dormant after 3 years of no customer-initiated activity",
                "3.1": "Banks must attempt to contact customers before declaring accounts dormant",
                "3.4": "Internal ledger transfers for dormant account management",
                "3.5": "Statement freeze procedures for dormant accounts",
                "5": "Contact attempt requirements and documentation",
                "8.1": "Central Bank transfer eligibility criteria"
            },
            "account_types": {
                "CURRENT": "Current account - operational banking",
                "SAVINGS": "Savings account - interest-bearing deposits",
                "FIXED_DEPOSIT": "Fixed deposit - term deposits with maturity",
                "INVESTMENT": "Investment account - securities and portfolio management"
            },
            "dormancy_statuses": {
                "Not_Dormant": "Active account with recent activity",
                "Potentially_Dormant": "Account approaching dormancy threshold",
                "Dormant": "Account declared dormant per CBUAE regulations",
                "Transferred_to_CB": "Account transferred to Central Bank"
            },
            "compliance_categories": {
                "Contact & Communication": "Customer outreach and notification requirements",
                "Process Management": "Internal processes and procedures",
                "Transfer Management": "Central Bank transfer procedures",
                "Documentation": "Record keeping and audit trail requirements"
            }
        }

    async def process_query(self, query: str, context_data: Dict = None) -> Dict[str, Any]:
        """
        Process a user query and provide intelligent response
        """
        try:
            # Analyze query intent
            intent = self._analyze_query_intent(query)

            # Generate response based on intent
            response = await self._generate_response(query, intent, context_data)

            return {
                "success": True,
                "response": response["answer"],
                "intent": intent,
                "sources": response.get("sources", []),
                "suggestions": response.get("suggestions", []),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"LLM query processing failed: {str(e)}")
            return {
                "success": False,
                "response": "I apologize, but I encountered an error processing your query. Please try rephrasing your question.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_query_intent(self, query: str) -> str:
        """Analyze the intent of the user query"""
        query_lower = query.lower()

        # Intent keywords mapping
        intent_keywords = {
            "data_summary": ["summary", "overview", "total", "count", "how many"],
            "dormancy_analysis": ["dormant", "dormancy", "inactive", "sleeping"],
            "compliance_check": ["compliance", "violation", "cbuae", "regulation", "article"],
            "account_details": ["account", "customer", "balance", "transaction"],
            "regulations": ["rule", "regulation", "requirement", "should", "must"],
            "recommendations": ["recommend", "suggest", "advice", "what should", "best practice"],
            "troubleshooting": ["error", "problem", "issue", "not working", "failed"]
        }

        # Check for intent matches
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent

        return "general_inquiry"

    async def _generate_response(self, query: str, intent: str, context_data: Dict = None) -> Dict[str, Any]:
        """Generate response based on query intent and context"""

        if intent == "data_summary":
            return self._generate_data_summary_response(context_data)
        elif intent == "dormancy_analysis":
            return self._generate_dormancy_response(query, context_data)
        elif intent == "compliance_check":
            return self._generate_compliance_response(query, context_data)
        elif intent == "account_details":
            return self._generate_account_details_response(query, context_data)
        elif intent == "regulations":
            return self._generate_regulations_response(query)
        elif intent == "recommendations":
            return self._generate_recommendations_response(context_data)
        elif intent == "troubleshooting":
            return self._generate_troubleshooting_response(query)
        else:
            return self._generate_general_response(query, context_data)

    def _generate_data_summary_response(self, context_data: Dict = None) -> Dict[str, Any]:
        """Generate data summary response"""
        if not context_data or "uploaded_data" not in context_data:
            return {
                "answer": "No data has been uploaded yet. Please upload your banking data first to get a summary.",
                "suggestions": ["Upload CSV/Excel file", "Generate sample data", "Check data format requirements"]
            }

        data = context_data["uploaded_data"]

        # Calculate summary statistics
        total_accounts = len(data)
        account_types = data.get('account_type', pd.Series()).value_counts().to_dict()
        dormant_count = len(data[data.get('dormancy_status', pd.Series()).isin(['Dormant', 'Transferred_to_CB'])])

        response = f"""📊 **Data Summary:**

**Total Accounts:** {total_accounts:,}
**Dormant Accounts:** {dormant_count:,} ({(dormant_count / total_accounts * 100):.1f}%)

**Account Types:**
"""

        for acc_type, count in account_types.items():
            percentage = (count / total_accounts) * 100
            response += f"• {acc_type}: {count:,} ({percentage:.1f}%)\n"

        if context_data.get("dormancy_results"):
            response += f"\n**Dormancy Analysis:** Completed with {len(context_data['dormancy_results'])} agents"

        if context_data.get("compliance_results"):
            total_violations = sum(r["violations_found"] for r in context_data["compliance_results"].values())
            response += f"\n**Compliance Check:** {total_violations} violations found"

        return {
            "answer": response,
            "sources": ["Uploaded Data Analysis"],
            "suggestions": ["Run dormancy analysis", "Check compliance violations", "Generate detailed report"]
        }

    def _generate_dormancy_response(self, query: str, context_data: Dict = None) -> Dict[str, Any]:
        """Generate dormancy-related response"""

        if not context_data or "dormancy_results" not in context_data:
            if "uploaded_data" in context_data:
                # Analyze uploaded data for dormancy indicators
                data = context_data["uploaded_data"]
                if 'dormancy_status' in data.columns:
                    dormant_accounts = data[data['dormancy_status'].isin(['Dormant', 'Transferred_to_CB'])]

                    response = f"""💤 **Dormancy Analysis from Uploaded Data:**

**Total Dormant Accounts:** {len(dormant_accounts):,}
**Dormancy Rate:** {(len(dormant_accounts) / len(data) * 100):.1f}%

**Dormancy Status Breakdown:**
"""
                    status_counts = data['dormancy_status'].value_counts()
                    for status, count in status_counts.items():
                        response += f"• {status}: {count:,}\n"

                    return {
                        "answer": response,
                        "sources": ["Uploaded Data"],
                        "suggestions": ["Run detailed dormancy analysis", "Check CBUAE compliance",
                                        "Review account contact attempts"]
                    }

            return {
                "answer": "No dormancy analysis has been performed yet. Please run the dormancy analysis first to get detailed insights about dormant accounts.",
                "suggestions": ["Run dormancy analysis", "Check data quality first",
                                "Review CBUAE Article 2 requirements"]
            }

        # Analyze dormancy results
        dormancy_results = context_data["dormancy_results"]
        total_dormant = sum(result["dormant_found"] for result in dormancy_results.values())

        response = f"""💤 **Dormancy Analysis Results:**

**Total Dormant Accounts Found:** {total_dormant:,}
**Agents Executed:** {len(dormancy_results)}

**Agent-wise Results:**
"""

        for agent_id, result in dormancy_results.items():
            agent_name = result["agent_info"]["name"]
            article = result["agent_info"]["article"]
            found = result["dormant_found"]
            response += f"• {agent_name} (Article {article}): {found:,} accounts\n"

        # Add CBUAE context
        response += f"\n**CBUAE Regulations:**\n"
        response += "• Demand deposits: Dormant after 3 years of inactivity\n"
        response += "• Fixed deposits: Dormant after maturity + 1 year\n"
        response += "• Investment accounts: Dormant after 3 years of inactivity\n"

        return {
            "answer": response,
            "sources": ["Dormancy Analysis Results", "CBUAE Regulations"],
            "suggestions": ["Review high-value dormant accounts", "Check contact attempt compliance",
                            "Generate dormancy report"]
        }

    def _generate_compliance_response(self, query: str, context_data: Dict = None) -> Dict[str, Any]:
        """Generate compliance-related response"""

        if not context_data or "compliance_results" not in context_data:
            return {
                "answer": "No compliance analysis has been performed yet. Please run the compliance verification first to identify potential violations.",
                "suggestions": ["Run compliance analysis", "Review CBUAE regulations",
                                "Check dormancy analysis results"]
            }

        compliance_results = context_data["compliance_results"]
        total_violations = sum(result["violations_found"] for result in compliance_results.values())

        response = f"""⚖️ **Compliance Analysis Results:**

**Total Violations Found:** {total_violations:,}
**Compliance Agents Executed:** {len(compliance_results)}

**Violation Breakdown:**
"""

        for agent_id, result in compliance_results.items():
            agent_name = result["agent_info"]["name"]
            category = result["agent_info"]["category"]
            article = result["agent_info"]["article"]
            violations = result["violations_found"]

            priority = "🔴 Critical" if violations > 15 else "🟠 High" if violations > 5 else "🟡 Medium"
            response += f"• {agent_name}: {violations:,} violations ({priority})\n"
            response += f"  Category: {category} | Article: {article}\n\n"

        # Add regulatory context
        response += "**Key CBUAE Requirements:**\n"
        response += "• Article 3.1: Customer contact before dormancy declaration\n"
        response += "• Article 5: Documented contact attempt requirements\n"
        response += "• Article 8.1: Central Bank transfer eligibility\n"

        return {
            "answer": response,
            "sources": ["Compliance Analysis Results", "CBUAE Regulations"],
            "suggestions": ["Address critical violations first", "Review contact procedures",
                            "Generate compliance report"]
        }

    def _generate_account_details_response(self, query: str, context_data: Dict = None) -> Dict[str, Any]:
        """Generate account details response"""

        if not context_data or "uploaded_data" not in context_data:
            return {
                "answer": "No account data available. Please upload your banking data first.",
                "suggestions": ["Upload account data", "Check data format", "Use sample data for testing"]
            }

        data = context_data["uploaded_data"]

        # Extract account information from query if specific account mentioned
        query_lower = query.lower()

        response = f"""🏦 **Account Information:**

**Dataset Overview:**
• Total Accounts: {len(data):,}
• Columns Available: {', '.join(data.columns)}

**Account Status Distribution:**
"""

        if 'account_status' in data.columns:
            status_counts = data['account_status'].value_counts()
            for status, count in status_counts.items():
                percentage = (count / len(data)) * 100
                response += f"• {status}: {count:,} ({percentage:.1f}%)\n"

        if 'balance_current' in data.columns:
            balance_stats = data['balance_current'].describe()
            response += f"\n**Balance Statistics:**\n"
            response += f"• Average Balance: {balance_stats['mean']:,.2f} AED\n"
            response += f"• Median Balance: {balance_stats['50%']:,.2f} AED\n"
            response += f"• Maximum Balance: {balance_stats['max']:,.2f} AED\n"

        return {
            "answer": response,
            "sources": ["Account Data Analysis"],
            "suggestions": ["Search specific account ID", "Filter by account type", "Analyze balance distribution"]
        }

    def _generate_regulations_response(self, query: str) -> Dict[str, Any]:
        """Generate regulations and requirements response"""

        query_lower = query.lower()

        response = """📜 **CBUAE Banking Regulations:**

**Dormancy Requirements:**
• **Article 2.1.1:** Demand deposit accounts become dormant after 3 years of customer inactivity
• **Article 2.2:** Fixed deposit accounts become dormant after maturity + 1 year if unclaimed
• **Article 2.3:** Investment accounts become dormant after 3 years of no customer-initiated activity

**Contact & Communication:**
• **Article 3.1:** Banks must attempt customer contact before declaring accounts dormant
• **Article 5:** Contact attempts must be documented and follow prescribed procedures

**Account Management:**
• **Article 3.4:** Internal ledger procedures for dormant account management
• **Article 3.5:** Statement freeze procedures for dormant accounts
• **Article 8.1:** Central Bank transfer eligibility criteria and procedures

**Key Compliance Points:**
✓ Proper customer notification before dormancy declaration
✓ Documented contact attempts with audit trail
✓ Timely transfer of eligible accounts to Central Bank
✓ Accurate dormancy classification by account type
✓ Maintenance of customer contact information
"""

        # Add specific guidance based on query
        if "contact" in query_lower:
            response += "\n**Contact Attempt Requirements:**\n"
            response += "• Multiple contact methods (phone, email, mail)\n"
            response += "• Documented attempt dates and methods\n"
            response += "• Grace period before dormancy declaration\n"

        return {
            "answer": response,
            "sources": ["CBUAE Regulations", "Banking Compliance Guidelines"],
            "suggestions": ["Check specific article details", "Review contact procedures", "Validate current practices"]
        }

    def _generate_recommendations_response(self, context_data: Dict = None) -> Dict[str, Any]:
        """Generate recommendations based on analysis results"""

        recommendations = []

        if context_data:
            # Data quality recommendations
            if "uploaded_data" in context_data:
                data = context_data["uploaded_data"]
                missing_cols = []
                required_cols = ['customer_id', 'account_id', 'account_type', 'dormancy_status']
                for col in required_cols:
                    if col not in data.columns:
                        missing_cols.append(col)

                if missing_cols:
                    recommendations.append(f"📋 **Data Quality:** Add missing columns: {', '.join(missing_cols)}")

            # Dormancy recommendations
            if "dormancy_results" in context_data:
                total_dormant = sum(r["dormant_found"] for r in context_data["dormancy_results"].values())
                if total_dormant > 0:
                    recommendations.append(
                        f"💤 **Dormancy Management:** Review {total_dormant:,} dormant accounts for CB transfer eligibility")
                    recommendations.append("📞 **Customer Contact:** Implement proactive customer outreach program")

            # Compliance recommendations
            if "compliance_results" in context_data:
                total_violations = sum(r["violations_found"] for r in context_data["compliance_results"].values())
                if total_violations > 0:
                    recommendations.append(
                        f"⚖️ **Compliance:** Address {total_violations:,} regulatory violations immediately")
                    recommendations.append("📚 **Training:** Provide staff training on CBUAE compliance requirements")

        # General recommendations
        if not recommendations:
            recommendations = [
                "📊 **Data Management:** Ensure regular data quality monitoring",
                "🔄 **Process Automation:** Implement automated dormancy detection",
                "📱 **Customer Communication:** Maintain updated customer contact information",
                "📋 **Compliance Monitoring:** Regular compliance audits and reviews",
                "🎯 **Performance Tracking:** Monitor dormancy rates and compliance metrics"
            ]

        response = "💡 **Recommendations:**\n\n" + "\n\n".join(recommendations)

        return {
            "answer": response,
            "sources": ["Best Practices", "Regulatory Guidelines"],
            "suggestions": ["Implement priority recommendations", "Set up monitoring dashboard",
                            "Schedule regular reviews"]
        }

    def _generate_troubleshooting_response(self, query: str) -> Dict[str, Any]:
        """Generate troubleshooting assistance"""

        query_lower = query.lower()

        troubleshooting_guides = {
            "agent": """🔧 **Agent Troubleshooting:**

**Common Agent Issues:**
• **Import Errors:** Check that agent files exist in `/agents/` directory
• **Missing Dependencies:** Install required packages (langgraph, pandas, numpy)
• **Module Path:** Verify Python path includes agents directory

**Solutions:**
1. Check file permissions and paths
2. Restart Streamlit application
3. Verify all required dependencies installed
4. Check agent class names match imports
""",
            "data": """📊 **Data Troubleshooting:**

**Data Upload Issues:**
• **File Format:** Ensure CSV/Excel files are properly formatted
• **Column Names:** Check for required columns (customer_id, account_id, etc.)
• **Data Types:** Verify date formats and numeric fields

**Solutions:**
1. Use sample data to test functionality
2. Check data preview after upload
3. Validate column mappings
4. Review data quality analysis results
""",
            "analysis": """🔍 **Analysis Troubleshooting:**

**Analysis Failures:**
• **Data Quality:** Low quality data may cause analysis failures
• **Memory Issues:** Large datasets may require optimization
• **Agent Configuration:** Check agent initialization parameters

**Solutions:**
1. Run data quality analysis first
2. Filter data to manageable size
3. Check system status on dashboard
4. Review error logs for specific issues
"""
        }

        # Select appropriate guide
        for keyword, guide in troubleshooting_guides.items():
            if keyword in query_lower:
                return {
                    "answer": guide,
                    "sources": ["Technical Documentation"],
                    "suggestions": ["Check system status", "Review logs", "Contact support if needed"]
                }

        # General troubleshooting
        response = """🛠️ **General Troubleshooting:**

**Step-by-Step Diagnosis:**
1. **Check System Status:** Verify all agents are available on Dashboard
2. **Data Validation:** Ensure data is uploaded and properly formatted
3. **Error Messages:** Review any error messages for specific guidance
4. **Browser Console:** Check browser console for JavaScript errors
5. **Restart Application:** Try refreshing or restarting Streamlit

**Common Solutions:**
• Clear browser cache and cookies
• Ensure stable internet connection
• Use supported file formats (CSV, Excel)
• Check data size limitations
• Verify column names and data types

**Need More Help?**
• Check the Dashboard for system status
• Review uploaded data format
• Try using sample data for testing
"""

        return {
            "answer": response,
            "sources": ["Technical Support"],
            "suggestions": ["Check Dashboard status", "Use sample data", "Review error messages"]
        }

    def _generate_general_response(self, query: str, context_data: Dict = None) -> Dict[str, Any]:
        """Generate general response for unspecified queries"""

        response = """🤖 **Banking Compliance Assistant**

I'm here to help you with banking compliance analysis! I can assist with:

**📊 Data Analysis:**
• Data quality assessment
• Account summaries and statistics
• Balance and transaction analysis

**💤 Dormancy Management:**
• CBUAE dormancy regulations
• Dormant account identification
• Contact attempt requirements

**⚖️ Compliance Verification:**
• Regulatory violation detection
• CBUAE article compliance
• Process improvement recommendations

**📋 Reporting:**
• Comprehensive analysis reports
• Compliance summaries
• Export capabilities

**Example Questions:**
• "What's the summary of my uploaded data?"
• "How many dormant accounts were found?"
• "What compliance violations were detected?"
• "What are the CBUAE requirements for contact attempts?"
• "Can you recommend next steps for my analysis?"

Feel free to ask me anything about your banking compliance analysis!
"""

        return {
            "answer": response,
            "sources": ["Banking Compliance Knowledge Base"],
            "suggestions": [
                "Ask about data summary",
                "Request dormancy analysis insights",
                "Check compliance violations",
                "Get regulatory guidance"
            ]
        }


# Utility functions for Streamlit integration
def create_llm_interface():
    """Create the LLM interface for Streamlit"""
    return BankingComplianceLLMAgent()


def format_llm_response(response: Dict[str, Any]) -> str:
    """Format LLM response for display in Streamlit"""
    if not response.get("success", False):
        return f"❌ Error: {response.get('response', 'Unknown error occurred')}"

    formatted_response = response["response"]

    # Add sources if available
    if response.get("sources"):
        formatted_response += f"\n\n**Sources:** {', '.join(response['sources'])}"

    return formatted_response