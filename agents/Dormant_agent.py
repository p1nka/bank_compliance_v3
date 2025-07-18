from enum import Enum
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants and Enums
class ActivityStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DORMANT = "dormant"
    UNCLAIMED = "unclaimed"


class AccountType(Enum):
    SAFE_DEPOSIT = "safe_deposit"
    INVESTMENT = "investment"
    FIXED_DEPOSIT = "fixed_deposit"
    DEMAND_DEPOSIT = "demand_deposit"
    UNCLAIMED_INSTRUMENT = "unclaimed_instrument"


class CustomerTier(Enum):
    STANDARD = "standard"
    HIGH_VALUE = "high_value"  # ≥25K AED
    PREMIUM = "premium"  # ≥100K AED
    VIP = "vip"  # ≥500K AED
    PRIVATE_BANKING = "private_banking"  # ≥1M AED


class ContactMethod(Enum):
    EMAIL = "email"
    SMS = "sms"
    PHONE = "phone"
    LETTER = "letter"


class GroqLLMClient:
    """Client for Groq API with Llama3 8B Instruct model"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama3-8b-8192"  # Llama3 8B Instruct model

        if not self.api_key:
            logger.warning("GROQ_API_KEY not found. LLM features will be disabled.")

    def get_recommendation(self, context: str, prompt_type: str = "banking_compliance") -> str:
        """Get AI-generated recommendation using Groq API"""
        if not self.api_key:
            return "AI recommendations unavailable - API key not configured"

        try:
            # Enhanced prompts for different scenarios
            system_prompts = {
                "banking_compliance": """You are a CBUAE banking compliance expert with deep knowledge of UAE banking regulations. 
                Provide specific, actionable recommendations for dormancy management and compliance issues. 
                Focus on regulatory requirements, risk mitigation, and operational efficiency.""",

                "dormancy_analysis": """You are a banking dormancy specialist. Analyze dormant account situations and provide 
                comprehensive recommendations including customer contact strategies, regulatory compliance steps, and risk assessments.""",

                "risk_assessment": """You are a banking risk management expert. Evaluate dormancy-related risks and provide 
                detailed mitigation strategies, focusing on operational, regulatory, and financial risks.""",

                "customer_engagement": """You are a customer relationship specialist in banking. Provide recommendations for 
                re-engaging dormant account holders, including communication strategies and reactivation approaches."""
            }

            system_prompt = system_prompts.get(prompt_type, system_prompts["banking_compliance"])

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"Banking compliance context: {context}\n\nProvide specific recommendations with:\n1. Immediate actions\n2. Regulatory considerations\n3. Risk mitigation steps\n4. Timeline for implementation\n5. Success metrics"}
                ],
                "max_tokens": 1000,
                "temperature": 0.1,  # Low temperature for consistent, factual responses
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
                return "No recommendation generated"

        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
            return f"Recommendation unavailable due to API error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting LLM recommendation: {e}")
            return f"Recommendation unavailable due to system error: {str(e)}"

    def generate_summary(self, data: Dict) -> str:
        """Generate AI-powered summary of dormancy analysis"""
        if not self.api_key:
            return "AI summary unavailable - API key not configured"

        try:
            context = f"""
            Dormancy Analysis Data:
            - Account Type: {data.get('account_type', 'Unknown')}
            - Dormancy Days: {data.get('dormancy_days', 0)}
            - Account Balance: {data.get('account_balance', 0)} AED
            - Contact Attempts: {data.get('contact_attempts', 0)}
            - Customer Tier: {data.get('customer_tier', 'Standard')}
            - Last Activity: {data.get('last_activity', 'Unknown')}
            - Risk Factors: {data.get('risk_factors', [])}
            """

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system",
                     "content": "You are a banking analyst. Create concise, professional summaries of dormancy analysis results focusing on key insights and implications."},
                    {"role": "user",
                     "content": f"Create a professional summary of this dormancy analysis:\n{context}\n\nProvide a concise summary highlighting key findings and implications."}
                ],
                "max_tokens": 300,
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
                return "Summary generation failed"

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Summary unavailable: {str(e)}"


# Base Agent Class
class DormantAgent:
    def __init__(self):
        self.llm_client = GroqLLMClient()

    def get_llm_recommendation(self, context: str, prompt_type: str = "banking_compliance") -> str:
        """Get AI-generated recommendation using Groq API"""
        return self.llm_client.get_recommendation(context, prompt_type)

    def generate_summary(self, data: Dict) -> str:
        """Generate AI-powered summary"""
        return self.llm_client.generate_summary(data)

    def execute(self, account_data: Dict) -> Dict:
        raise NotImplementedError


# Agent 1: Safe Deposit Dormancy
class SafeDepositDormancyAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        dormancy_days = (datetime.now() - account_data['last_activity_date']).days
        result = {
            'agent': 'SafeDepositDormancy',
            'status': ActivityStatus.ACTIVE.value,
            'action': None,
            'priority': None,
            'recommendation': None,
            'ai_summary': None,
            'dormancy_days': dormancy_days,
            'account_details': account_data
        }

        if dormancy_days >= 1095:  # 3+ years
            result['status'] = ActivityStatus.DORMANT.value
            result['action'] = "File court application"
            result['priority'] = "high"

            context = f"""
            Safe Deposit Box Dormancy Analysis:
            - Box has been dormant for {dormancy_days} days ({dormancy_days / 365:.1f} years)
            - Exceeded 3-year threshold requiring court application
            - Account Type: Safe Deposit Box
            - Current Action Required: File court application
            - Regulatory Requirement: CBUAE compliance for dormant safe deposit boxes
            """

            result['recommendation'] = self.get_llm_recommendation(context, "dormancy_analysis")
            result['ai_summary'] = self.generate_summary({
                'account_type': 'Safe Deposit Box',
                'dormancy_days': dormancy_days,
                'account_balance': account_data.get('outstanding_charges', 0),
                'risk_factors': ['Court application required', 'Long-term dormancy'],
                'last_activity': account_data.get('last_activity_date', 'Unknown')
            })

        return result


# Agent 2: Investment Account Inactivity
class InvestmentAccountInactivityAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        dormancy_days = (datetime.now() - account_data['last_activity_date']).days
        result = {
            'agent': 'InvestmentAccountInactivity',
            'status': ActivityStatus.ACTIVE.value,
            'action': None,
            'priority': None,
            'recommendation': None,
            'ai_summary': None,
            'dormancy_days': dormancy_days,
            'portfolio_value': account_data.get('portfolio_value', 0)
        }

        if dormancy_days >= 1095:  # 3 years
            result['status'] = ActivityStatus.DORMANT.value
            result['action'] = "Review investment status"
            result['priority'] = "high"

            context = f"""
            Investment Account Dormancy Analysis:
            - Account dormant for {dormancy_days} days ({dormancy_days / 365:.1f} years)
            - Portfolio Value: {account_data.get('portfolio_value', 'Unknown')} AED
            - Investment Type: {account_data.get('investment_type', 'Mixed Portfolio')}
            - Risk Level: High due to market exposure without active management
            - Required Action: Comprehensive investment review and customer contact
            """

            result['recommendation'] = self.get_llm_recommendation(context, "risk_assessment")
            result['ai_summary'] = self.generate_summary({
                'account_type': 'Investment Account',
                'dormancy_days': dormancy_days,
                'account_balance': account_data.get('portfolio_value', 0),
                'risk_factors': ['Market exposure', 'No active management', 'Potential losses'],
                'last_activity': account_data.get('last_activity_date', 'Unknown')
            })

        return result


# Agent 3: Fixed Deposit Inactivity
class FixedDepositInactivityAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        maturity_date = account_data.get('maturity_date')
        if not maturity_date:
            return {
                'agent': 'FixedDepositInactivity',
                'error': 'Missing maturity date',
                'status': ActivityStatus.ACTIVE.value,
                'ai_summary': 'Analysis incomplete due to missing maturity date'
            }

        dormancy_days = (datetime.now() - maturity_date).days
        result = {
            'agent': 'FixedDepositInactivity',
            'status': ActivityStatus.ACTIVE.value,
            'action': None,
            'priority': None,
            'recommendation': None,
            'ai_summary': None,
            'dormancy_days': dormancy_days,
            'maturity_amount': account_data.get('maturity_amount', 0)
        }

        if dormancy_days >= 1095:  # 3 years post-maturity
            result['status'] = ActivityStatus.DORMANT.value
            result['action'] = "Monitor maturity dates"
            result['priority'] = "medium"

            context = f"""
            Fixed Deposit Post-Maturity Dormancy:
            - Matured {dormancy_days} days ago ({dormancy_days / 365:.1f} years)
            - Original Amount: {account_data.get('original_amount', 'Unknown')} AED
            - Maturity Amount: {account_data.get('maturity_amount', 'Unknown')} AED
            - Interest Rate: {account_data.get('interest_rate', 'Unknown')}%
            - Customer has not claimed matured funds
            - Potential interest earnings lost due to non-renewal
            """

            result['recommendation'] = self.get_llm_recommendation(context, "customer_engagement")
            result['ai_summary'] = self.generate_summary({
                'account_type': 'Fixed Deposit (Post-Maturity)',
                'dormancy_days': dormancy_days,
                'account_balance': account_data.get('maturity_amount', 0),
                'risk_factors': ['Unclaimed maturity amount', 'Interest earnings lost'],
                'last_activity': maturity_date.strftime('%Y-%m-%d') if maturity_date else 'Unknown'
            })

        return result


# Agent 4: Demand Deposit Inactivity
class DemandDepositInactivityAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        dormancy_days = (datetime.now() - account_data['last_activity_date']).days
        result = {
            'agent': 'DemandDepositInactivity',
            'status': ActivityStatus.ACTIVE.value,
            'action': None,
            'priority': None,
            'recommendation': None,
            'ai_summary': None,
            'dormancy_days': dormancy_days,
            'account_balance': account_data.get('balance', 0)
        }

        if dormancy_days >= 1095:  # 3 years
            result['status'] = ActivityStatus.DORMANT.value
            result['action'] = "Flag as dormant and initiate contact"
            result['priority'] = "medium"

            context = f"""
            Demand Deposit Account Dormancy:
            - Account dormant for {dormancy_days} days ({dormancy_days / 365:.1f} years)
            - Current Balance: {account_data.get('balance', 'Unknown')} AED
            - Account Type: {account_data.get('account_type', 'Savings/Current')}
            - Customer Tier: {account_data.get('customer_tier', 'Standard')}
            - Last Transaction: {account_data.get('last_transaction_type', 'Unknown')}
            - Contact Information: {account_data.get('contact_status', 'Unknown')}
            """

            result['recommendation'] = self.get_llm_recommendation(context, "customer_engagement")
            result['ai_summary'] = self.generate_summary({
                'account_type': 'Demand Deposit',
                'dormancy_days': dormancy_days,
                'account_balance': account_data.get('balance', 0),
                'customer_tier': account_data.get('customer_tier', 'Standard'),
                'risk_factors': ['Long-term inactivity', 'Potential account closure'],
                'last_activity': account_data.get('last_activity_date', 'Unknown')
            })

        return result


# Agent 5: Unclaimed Payment Instruments
class UnclaimedInstrumentsAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        dormancy_days = (datetime.now() - account_data['last_activity_date']).days
        result = {
            'agent': 'UnclaimedInstruments',
            'status': ActivityStatus.ACTIVE.value,
            'action': None,
            'priority': None,
            'recommendation': None,
            'ai_summary': None,
            'dormancy_days': dormancy_days,
            'instrument_value': account_data.get('amount', 0)
        }

        if dormancy_days >= 365:  # 1 year
            result['status'] = ActivityStatus.UNCLAIMED.value
            result['action'] = "Process for ledger transfer"
            result['priority'] = "critical"

            context = f"""
            Unclaimed Payment Instrument Analysis:
            - Instrument unclaimed for {dormancy_days} days ({dormancy_days / 365:.1f} years)
            - Instrument Type: {account_data.get('instrument_type', 'Unknown')}
            - Amount: {account_data.get('amount', 'Unknown')} AED
            - Beneficiary: {account_data.get('beneficiary', 'Unknown')}
            - Issue Date: {account_data.get('issue_date', 'Unknown')}
            - Urgency: High - requires immediate ledger transfer processing
            """

            result['recommendation'] = self.get_llm_recommendation(context, "banking_compliance")
            result['ai_summary'] = self.generate_summary({
                'account_type': 'Unclaimed Payment Instrument',
                'dormancy_days': dormancy_days,
                'account_balance': account_data.get('amount', 0),
                'risk_factors': ['Regulatory non-compliance', 'Beneficiary rights'],
                'last_activity': account_data.get('last_activity_date', 'Unknown')
            })

        return result


# Continue with other agents following the same pattern...
# Agent 6: CBUAE Transfer Eligibility
class CBUAETransferEligibilityAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        account_type = account_data.get('account_type')
        dormancy_days = (datetime.now() - account_data['last_activity_date']).days
        result = {
            'agent': 'CBUAETransferEligibility',
            'eligible': False,
            'action': None,
            'priority': None,
            'recommendation': None,
            'ai_summary': None,
            'dormancy_days': dormancy_days
        }

        if account_type == AccountType.SAFE_DEPOSIT.value and dormancy_days >= 1825:  # 5 years
            result['eligible'] = True
            result['action'] = "Initiate CBUAE transfer for safe deposit"
            result['priority'] = "high"
        elif account_type == AccountType.UNCLAIMED_INSTRUMENT.value and dormancy_days >= 1095:  # 3 years
            result['eligible'] = True
            result['action'] = "Initiate CBUAE transfer for unclaimed instrument"
            result['priority'] = "high"
        elif dormancy_days >= 1825:  # 5 years for regular accounts
            result['eligible'] = True
            result['action'] = "Initiate CBUAE transfer"
            result['priority'] = "high"

        if result['eligible']:
            context = f"""
            CBUAE Transfer Eligibility Assessment:
            - Account Type: {account_type}
            - Dormancy Period: {dormancy_days} days ({dormancy_days / 365:.1f} years)
            - Transfer Eligible: Yes
            - Regulatory Requirement: CBUAE transfer mandatory
            - Account Balance: {account_data.get('balance', 'Unknown')} AED
            - Customer Contact Status: {account_data.get('contact_status', 'Unknown')}
            """

            result['recommendation'] = self.get_llm_recommendation(context, "banking_compliance")
            result['ai_summary'] = self.generate_summary({
                'account_type': account_type,
                'dormancy_days': dormancy_days,
                'account_balance': account_data.get('balance', 0),
                'risk_factors': ['Regulatory deadline', 'CBUAE transfer required'],
                'last_activity': account_data.get('last_activity_date', 'Unknown')
            })

        return result


# Additional agents following the same pattern...
# For brevity, I'll include the orchestrator with the updated structure

# Orchestrator
class DormantAccountOrchestrator:
    def __init__(self):
        self.agents = {
            'safe_deposit_dormancy': SafeDepositDormancyAgent(),
            'investment_account_inactivity': InvestmentAccountInactivityAgent(),
            'fixed_deposit_inactivity': FixedDepositInactivityAgent(),
            'demand_deposit_inactivity': DemandDepositInactivityAgent(),
            'unclaimed_instruments': UnclaimedInstrumentsAgent(),
            'cbuae_transfer_eligibility': CBUAETransferEligibilityAgent(),
            # Add other agents...
        }
        self.llm_client = GroqLLMClient()

    def process_account(self, account_data: Dict) -> Dict:
        results = {}

        # Process through relevant agents
        account_type = account_data.get('account_type')

        if account_type == AccountType.SAFE_DEPOSIT.value:
            results['safe_deposit_dormancy'] = self.agents['safe_deposit_dormancy'].execute(account_data)
        elif account_type == AccountType.INVESTMENT.value:
            results['investment_account_inactivity'] = self.agents['investment_account_inactivity'].execute(
                account_data)
        elif account_type == AccountType.FIXED_DEPOSIT.value:
            results['fixed_deposit_inactivity'] = self.agents['fixed_deposit_inactivity'].execute(account_data)
        elif account_type == AccountType.DEMAND_DEPOSIT.value:
            results['demand_deposit_inactivity'] = self.agents['demand_deposit_inactivity'].execute(account_data)
        elif account_type == AccountType.UNCLAIMED_INSTRUMENT.value:
            results['unclaimed_instruments'] = self.agents['unclaimed_instruments'].execute(account_data)

        # Always run transfer eligibility check
        results['cbuae_transfer_eligibility'] = self.agents['cbuae_transfer_eligibility'].execute(account_data)

        # Generate overall AI summary
        results['overall_ai_summary'] = self._generate_overall_summary(account_data, results)

        return results

    def _generate_overall_summary(self, account_data: Dict, results: Dict) -> str:
        """Generate an overall AI summary of all agent results"""
        context = f"""
        Comprehensive Dormancy Analysis Summary:
        Account Details: {account_data}
        Agent Results: {results}

        Provide a comprehensive executive summary of the dormancy analysis including:
        1. Key findings across all agents
        2. Priority actions required
        3. Regulatory compliance status
        4. Risk assessment
        5. Recommended next steps
        """

        return self.llm_client.get_recommendation(context, "dormancy_analysis")

    def export_to_csv(self, account_data: Dict, results: Dict[str, Dict],
                      filename: str = "dormant_agents_results.csv") -> None:
        """Export account data + agent results to CSV with AI summaries"""
        import csv

        rows = []
        for agent_name, agent_result in results.items():
            if agent_name == 'overall_ai_summary':
                continue

            row = {
                "account_no": account_data.get("account_no", "N/A"),
                "account_type": account_data.get("account_type", "N/A"),
                "last_activity_date": account_data.get("last_activity_date", "N/A"),
                "account_value": account_data.get("account_value", "N/A"),
                "agent": agent_name,
                "ai_summary": agent_result.get("ai_summary", "N/A"),
                "llm_recommendation": agent_result.get("recommendation", "N/A")
            }
            row.update(agent_result)
            rows.append(row)

        # Add overall summary row
        rows.append({
            "account_no": account_data.get("account_no", "N/A"),
            "account_type": account_data.get("account_type", "N/A"),
            "agent": "overall_summary",
            "ai_summary": results.get('overall_ai_summary', 'N/A'),
            "llm_recommendation": results.get('overall_ai_summary', 'N/A')
        })

        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            if rows:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"Enhanced results with AI summaries exported to {filename}")


# Example Usage
if __name__ == "__main__":
    # Sample account data
    account_data = {
        'account_no': 'ACC001',
        'account_type': 'demand_deposit',
        'last_activity_date': datetime(2020, 1, 15),
        'account_value': 25000,
        'balance': 25000,
        'customer_tier': 'high_value',
        'contact_status': 'address_known'
    }

    orchestrator = DormantAccountOrchestrator()
    results = orchestrator.process_account(account_data)

    print("=== ENHANCED DORMANCY ANALYSIS WITH AI SUMMARIES ===")
    print(f"Account: {account_data['account_no']}")
    print(f"Overall AI Summary: {results.get('overall_ai_summary', 'N/A')}")

    for agent, result in results.items():
        if agent != 'overall_ai_summary':
            print(f"\n{agent.upper()} RESULTS:")
            print(f"AI Summary: {result.get('ai_summary', 'N/A')}")
            print(f"Recommendation: {result.get('recommendation', 'N/A')}")

    # Export enhanced results
    orchestrator.export_to_csv(account_data, results, "enhanced_dormant_account_actions.csv")