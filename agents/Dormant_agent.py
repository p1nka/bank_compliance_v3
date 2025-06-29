from enum import Enum
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests  # For API calls to Llama3 8B Instruct
from llama_cpp import Llama
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)
response = llm("Tell me a joke.", max_tokens=50)
# Constants and Enums
class ActivityStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"  # Corrected from IMACTIVE
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

# Base Agent Class
class DormantAgent:
    def __init__(self):
        self.llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)  # Replace with actual endpoint

    def get_llm_recommendation(self, context: str) -> str:
        try:
            payload = {
                "model": "llama3",  # or whatever model you pulled with Ollama
                "prompt": f"Based on the following banking dormancy context, provide a professional recommendation: {context}"
            }
            response = llm("Tell me a joke.", max_tokens=50)
            response.raise_for_status()
            return response.json().get("response", "No recommendation available")
        except Exception as e:
            print(f"Error getting LLM recommendation: {e}")
            return "Recommendation unavailable due to system error"

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
            'recommendation': None
        }
        
        if dormancy_days >= 1095:  # 3+ years
            result['status'] = ActivityStatus.DORMANT.value
            result['action'] = "File court application"
            result['priority'] = "high"
            context = f"Safe deposit box dormant for {dormancy_days} days. Current action: {result['action']}"
            result['recommendation'] = self.get_llm_recommendation(context)
            
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
            'recommendation': None
        }
        
        if dormancy_days >= 1095:  # 3 years
            result['status'] = ActivityStatus.DORMANT.value
            result['action'] = "Review investment status"
            result['priority'] = "high"
            context = f"Investment account dormant for {dormancy_days} days. Portfolio value: {account_data.get('portfolio_value', 'unknown')}"
            result['recommendation'] = self.get_llm_recommendation(context)
            
        return result

# Agent 3: Fixed Deposit Inactivity
class FixedDepositInactivityAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        maturity_date = account_data.get('maturity_date')
        if not maturity_date:
            return {
                'agent': 'FixedDepositInactivity',
                'error': 'Missing maturity date',
                'status': ActivityStatus.ACTIVE.value
            }
            
        dormancy_days = (datetime.now() - maturity_date).days
        result = {
            'agent': 'FixedDepositInactivity',
            'status': ActivityStatus.ACTIVE.value,
            'action': None,
            'priority': None,
            'recommendation': None
        }
        
        if dormancy_days >= 1095:  # 3 years post-maturity
            result['status'] = ActivityStatus.DORMANT.value
            result['action'] = "Monitor maturity dates"
            result['priority'] = "medium"
            context = f"Fixed deposit dormant for {dormancy_days} days post-maturity. Original amount: {account_data.get('amount', 'unknown')}"
            result['recommendation'] = self.get_llm_recommendation(context)
            
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
            'recommendation': None
        }
        
        if dormancy_days >= 1095:  # 3 years
            result['status'] = ActivityStatus.DORMANT.value
            result['action'] = "Flag as dormant and initiate contact"
            result['priority'] = "medium"
            context = f"Demand deposit account dormant for {dormancy_days} days. Account type: {account_data.get('account_type', 'standard')}"
            result['recommendation'] = self.get_llm_recommendation(context)
            
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
            'recommendation': None
        }
        
        if dormancy_days >= 365:  # 1 year
            result['status'] = ActivityStatus.UNCLAIMED.value
            result['action'] = "Process for ledger transfer"
            result['priority'] = "critical"
            context = f"Unclaimed instrument dormant for {dormancy_days} days. Instrument type: {account_data.get('instrument_type', 'unknown')}"
            result['recommendation'] = self.get_llm_recommendation(context)
            
        return result

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
            'recommendation': None
        }
        
        if account_type == AccountType.SAFE_DEPOSIT.value and dormancy_days >= 1825:  # 5 years
            result['eligible'] = True
            result['action'] = "Initiate CBUAE transfer for safe deposit"
        elif account_type == AccountType.UNCLAIMED_INSTRUMENT.value and dormancy_days >= 1095:  # 3 years
            result['eligible'] = True
            result['action'] = "Initiate CBUAE transfer for unclaimed instrument"
        elif dormancy_days >= 1825:  # 5 years for regular accounts
            result['eligible'] = True
            result['action'] = "Initiate CBUAE transfer"
            
        if result['eligible']:
            result['priority'] = "high"
            context = f"Account eligible for CBUAE transfer after {dormancy_days} days. Account details: {account_data}"
            result['recommendation'] = self.get_llm_recommendation(context)
            
        return result

# Agent 7: Article 3 Process
class CBUAEArticle3ProcessAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        current_stage = account_data.get('article3_stage', 'STAGE_1')
        result = {
            'agent': 'CBUAEArticle3Process',
            'current_stage': current_stage,
            'next_stage': None,
            'action': None,
            'priority': None,
            'recommendation': None
        }
        
        stages = [
            'STAGE_1', 'STAGE_2', 'STAGE_3', 'STAGE_4', 
            'STAGE_5', 'STAGE_6', 'STAGE_7', 'STAGE_8', 'STAGE_9'
        ]
        
        current_index = stages.index(current_stage) if current_stage in stages else 0
        if current_index < len(stages) - 1:
            result['next_stage'] = stages[current_index + 1]
            result['action'] = f"Proceed to {result['next_stage']}"
            result['priority'] = "medium"
            
            # Specific actions for some stages
            if current_stage == 'STAGE_1':
                result['action'] = "Initial contact required"
            elif current_stage == 'STAGE_2':
                result['action'] = "Notify instrument issuers"
            elif current_stage == 'STAGE_3':
                result['action'] = "Safe deposit notice"
            elif current_stage == 'STAGE_4':
                result['action'] = "3-month waiting period"
                
            context = f"Article 3 process at {current_stage}. Next action: {result['action']}"
            result['recommendation'] = self.get_llm_recommendation(context)
            
        return result

# Agent 8: Contact Attempts
class CBUAEContactAttemptsAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        customer_type = account_data.get('customer_type', 'individual')
        value = account_data.get('account_value', 0)
        attempts_made = account_data.get('contact_attempts', 0)
        
        result = {
            'agent': 'CBUAEContactAttempts',
            'attempts_required': None,
            'attempts_made': attempts_made,
            'action': None,
            'priority': None,
            'recommendation': None
        }
        
        if customer_type == 'individual':
            if value >= 25000:  # High value individual
                result['attempts_required'] = 5
            else:
                result['attempts_required'] = 3
        else:  # corporate
            if value >= 100000:  # High value corporate
                result['attempts_required'] = 7
            else:
                result['attempts_required'] = 5
                
        if attempts_made < result['attempts_required']:
            result['action'] = f"Schedule contact attempt {attempts_made + 1} of {result['attempts_required']}"
            result['priority'] = "medium"
            context = f"Contact attempts for {customer_type} account (value: {value}). Made {attempts_made} of {result['attempts_required']} required."
            result['recommendation'] = self.get_llm_recommendation(context)
        else:
            result['action'] = "Required contact attempts completed"
            
        return result

# Agent 9: High Value Dormant
class HighValueDormantAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        value = account_data.get('account_value', 0)
        currency = account_data.get('currency', 'AED')
        result = {
            'agent': 'HighValueDormant',
            'customer_tier': CustomerTier.STANDARD.value,
            'action': None,
            'priority': None,
            'risk_score': 0,
            'recommendation': None
        }
        
        # Convert to AED if needed (simplified)
        if currency != 'AED':
            # In real implementation, use actual exchange rate
            value = value * 1.2  # Simplified conversion factor
            
        # Determine tier
        if value >= 1000000:
            result['customer_tier'] = CustomerTier.PRIVATE_BANKING.value
            result['priority'] = "highest"
            result['risk_score'] = 90
        elif value >= 500000:
            result['customer_tier'] = CustomerTier.VIP.value
            result['priority'] = "very_high"
            result['risk_score'] = 80
        elif value >= 100000:
            result['customer_tier'] = CustomerTier.PREMIUM.value
            result['priority'] = "high"
            result['risk_score'] = 70
        elif value >= 25000:
            result['customer_tier'] = CustomerTier.HIGH_VALUE.value
            result['priority'] = "medium_high"
            result['risk_score'] = 60
            
        if result['priority']:
            result['action'] = f"Assign relationship manager for {result['customer_tier']} customer"
            context = f"High value dormant account detected. Tier: {result['customer_tier']}, Value: {value} {currency}, Risk score: {result['risk_score']}"
            result['recommendation'] = self.get_llm_recommendation(context)
            
        return result

# Agent 10: Dormant-to-Active Transitions
class CBUAEDormantToActiveTransitionsAgent(DormantAgent):
    def execute(self, account_data: Dict) -> Dict:
        transition_history = account_data.get('transition_history', [])
        result = {
            'agent': 'DormantToActiveTransitions',
            'analysis': None,
            'action': None,
            'recommendation': None
        }
        
        if len(transition_history) > 0:
            last_transition = transition_history[-1]
            days_since_last_active = (datetime.now() - last_transition['activation_date']).days
            
            if days_since_last_active < 30:
                result['analysis'] = "Recent reactivation"
                result['action'] = "Monitor for sustained activity"
            elif days_since_last_active < 90:
                result['analysis'] = "Moderate reactivation"
                result['action'] = "Consider engagement campaign"
            else:
                result['analysis'] = "Historical reactivation"
                result['action'] = "Use for pattern analysis"
                
            context = f"Account transition history: {len(transition_history)} transitions. Last active {days_since_last_active} days ago."
            result['recommendation'] = self.get_llm_recommendation(context)
            
        return result

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
            'article_3_process_needed': CBUAEArticle3ProcessAgent(),
            'contact_attempts_needed': CBUAEContactAttemptsAgent(),
            'high_value_dormant': HighValueDormantAgent(),
            'dormant_to_active_transitions': CBUAEDormantToActiveTransitionsAgent()
        }
        
    def process_account(self, account_data: Dict) -> Dict:
        results = {}
        
        # First run standard dormancy agents based on account type
        account_type = account_data.get('account_type')
        
        if account_type == AccountType.SAFE_DEPOSIT.value:
            results['safe_deposit_dormancy'] = self.agents['safe_deposit_dormancy'].execute(account_data)
        elif account_type == AccountType.INVESTMENT.value:
            results['investment_account_inactivity'] = self.agents['investment_account_inactivity'].execute(account_data)
        elif account_type == AccountType.FIXED_DEPOSIT.value:
            results['fixed_deposit_inactivity'] = self.agents['fixed_deposit_inactivity'].execute(account_data)
        elif account_type == AccountType.DEMAND_DEPOSIT.value:
            results['demand_deposit_inactivity'] = self.agents['demand_deposit_inactivity'].execute(account_data)
        elif account_type == AccountType.UNCLAIMED_INSTRUMENT.value:
            results['unclaimed_instruments'] = self.agents['unclaimed_instruments'].execute(account_data)
            
        # Then run all other agents that might be relevant
        results['cbuae_transfer_eligibility'] = self.agents['cbuae_transfer_eligibility'].execute(account_data)
        
        if account_data.get('status') == ActivityStatus.DORMANT.value:
            results['article_3_process_needed'] = self.agents['article_3_process_needed'].execute(account_data)
            results['contact_attempts_needed'] = self.agents['contact_attempts_needed'].execute(account_data)
            results['high_value_dormant'] = self.agents['high_value_dormant'].execute(account_data)
            
        if account_data.get('transition_history'):
            results['dormant_to_active_transitions'] = self.agents['dormant_to_active_transitions'].execute(account_data)
            
        return results

    def export_to_csv(self, account_data: Dict, results: Dict[str, Dict],
                      filename: str = "dormant_agents_results.csv") -> None:
        """
        Export account data + agent results to CSV.

        Args:
            account_data: Original account data (e.g., account_no, last_activity_date).
            results: Agent-specific results (e.g., action, priority).
            filename: Output CSV path.
        """
        # Prepare rows for CSV
        rows = []

        # Add account metadata to every agent's result
        for agent_name, agent_result in results.items():
            row = {
                "account_no": account_data.get("account_no", "N/A"),
                "account_type": account_data.get("account_type", "N/A"),
                "last_activity_date": account_data.get("last_activity_date", "N/A"),
                "account_value": account_data.get("account_value", "N/A"),
                "agent": agent_name,
            }
            row.update(agent_result)  # Add agent-specific fields (action, priority, etc.)
            rows.append(row)

        # Write to CSV
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            if rows:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"Results exported to {filename}")

# Example Usage
if __name__ == "__main__":

    orchestrator = DormantAccountOrchestrator()
    results = orchestrator.process_account(account_data)
    orchestrator.export_to_csv(account_data, results, "dormant_account_actions.csv")
    print("Dormancy Analysis Results:")
    for agent, result in results.items():
        print(f"\n{agent.upper()} RESULTS:")
        for key, value in result.items():
            print(f"{key}: {value}")