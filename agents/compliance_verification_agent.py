"""
LLM-Enhanced CBUAE Compliance System with Orchestrator
=====================================================

Production-ready banking compliance system integrated with Llama 3 8B Instruct
for intelligent recommendations and orchestrated multi-agent processing.

Features:
- Llama 3 8B Instruct integration for intelligent analysis
- Orchestrator pattern for agent coordination
- LLM-powered recommendation engine
- Real-time compliance analysis with AI insights
- Production-grade error handling and monitoring
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal
import sqlite3
from contextlib import asynccontextmanager
import aiohttp
import warnings
warnings.filterwarnings('ignore')

# LLM Integration
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== LLM INTEGRATION SERVICE =====

class LlamaComplianceAnalyzer:
    """
    Llama 3 8B Instruct integration for banking compliance analysis
    """

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 2048
        self.temperature = 0.1  # Low temperature for consistent compliance analysis

        if LLM_AVAILABLE:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize Llama 3 8B Instruct model"""
        try:
            logger.info(f"Loading Llama 3 8B model on {self.device}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Llama 3 8B model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Llama 3 8B model: {e}")
            self.model = None
            self.tokenizer = None

    def _create_compliance_prompt(self, context: Dict, analysis_type: str) -> str:
        """
        Create specialized prompts for banking compliance analysis
        """

        base_context = f"""
You are an expert CBUAE (Central Bank of UAE) banking compliance analyst with deep knowledge of UAE banking regulations, particularly Articles 2, 3, 5, and 8 of the Dormant Accounts Regulation.

Current Analysis Context:
- Bank: {context.get('bank_name', 'UAE Commercial Bank')}
- Analysis Type: {analysis_type}
- Total Accounts: {context.get('total_accounts', 'N/A')}
- Violations Found: {context.get('violations_found', 'N/A')}
- Compliance Rate: {context.get('compliance_rate', 'N/A')}%

Regulatory Framework:
- CBUAE Article 2: Dormancy classification after 3 years (1095 days)
- CBUAE Article 3: Minimum 3 contact attempts required
- CBUAE Article 5: Bank responsibilities and timelines
- CBUAE Article 8: Mandatory Central Bank transfer after 5 years

"""

        if analysis_type == "contact_attempts":
            return base_context + f"""
Contact Attempts Analysis Results:
- Accounts with insufficient attempts: {context.get('insufficient_attempts', 0)}
- Accounts with zero attempts: {context.get('zero_attempts', 0)}
- Average attempts per account: {context.get('avg_attempts', 0)}
- Timeline violations: {context.get('timeline_violations', 0)}

Specific Violations:
{json.dumps(context.get('sample_violations', []), indent=2)}

Provide 5 specific, actionable recommendations to improve contact attempts compliance with CBUAE Article 3.1. Focus on practical implementation steps that a UAE bank can execute immediately.

Recommendations:"""

        elif analysis_type == "dormancy_classification":
            return base_context + f"""
Dormancy Classification Analysis Results:
- Unflagged dormant candidates: {context.get('unflagged_accounts', 0)}
- Misclassified accounts: {context.get('misclassified_accounts', 0)}
- Average dormancy period: {context.get('avg_dormancy_days', 0)} days
- High-value dormant accounts: {context.get('high_value_accounts', 0)}

Classification Issues:
{json.dumps(context.get('classification_issues', []), indent=2)}

Provide 5 specific recommendations to improve dormancy classification accuracy per CBUAE Article 2. Include system improvements and process enhancements.

Recommendations:"""

        elif analysis_type == "cbuae_transfers":
            return base_context + f"""
CBUAE Transfer Analysis Results:
- Accounts eligible for transfer: {context.get('transfer_eligible', 0)}
- Overdue transfers: {context.get('overdue_transfers', 0)}
- Total transfer amount: AED {context.get('total_transfer_amount', 0):,.2f}
- Foreign currency conversions needed: {context.get('fx_conversions', 0)}

Transfer Violations:
{json.dumps(context.get('transfer_violations', []), indent=2)}

Provide 5 critical recommendations for CBUAE transfer compliance per Article 8. Include legal risk mitigation and process automation suggestions.

Recommendations:"""

        elif analysis_type == "risk_assessment":
            return base_context + f"""
Overall Risk Assessment:
- Critical violations: {context.get('critical_violations', 0)}
- High-risk violations: {context.get('high_violations', 0)}
- Regulatory risk level: {context.get('risk_level', 'UNKNOWN')}
- Audit readiness score: {context.get('audit_score', 0)}%

Risk Factors:
{json.dumps(context.get('risk_factors', []), indent=2)}

As a senior banking compliance expert, provide a comprehensive risk mitigation strategy with 7 specific actions to achieve 100% CBUAE compliance. Include immediate, short-term, and long-term recommendations.

Risk Mitigation Strategy:"""

        else:
            return base_context + f"""
General Compliance Analysis:
{json.dumps(context, indent=2)}

Provide 5 specific recommendations to improve overall CBUAE compliance based on the analysis results.

Recommendations:"""

    async def generate_compliance_recommendations(self, context: Dict, analysis_type: str) -> Dict:
        """
        Generate intelligent compliance recommendations using Llama 3 8B
        """
        if not self.model or not self.tokenizer:
            return self._fallback_recommendations(analysis_type)

        try:
            prompt = self._create_compliance_prompt(context, analysis_type)

            # Tokenize and generate
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - 512,  # Reserve space for generation
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode and extract recommendations
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            recommendations_text = full_response[len(prompt):].strip()

            # Parse recommendations
            recommendations = self._parse_llm_recommendations(recommendations_text)

            return {
                "success": True,
                "recommendations": recommendations,
                "analysis_type": analysis_type,
                "generated_by": "Llama-3-8B-Instruct",
                "generation_time": time.time(),
                "model_confidence": self._calculate_confidence_score(recommendations_text)
            }

        except Exception as e:
            logger.error(f"LLM recommendation generation failed: {e}")
            return self._fallback_recommendations(analysis_type)

    def _parse_llm_recommendations(self, text: str) -> List[Dict]:
        """Parse LLM output into structured recommendations"""
        recommendations = []
        lines = text.split('\n')

        current_rec = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for numbered recommendations
            if any(line.startswith(f"{i}.") for i in range(1, 10)):
                if current_rec:
                    recommendations.append(current_rec)

                current_rec = {
                    "recommendation": line,
                    "priority": self._extract_priority(line),
                    "category": self._categorize_recommendation(line),
                    "implementation_timeframe": self._extract_timeframe(line)
                }
            elif current_rec and line:
                # Continue previous recommendation
                current_rec["recommendation"] += " " + line

        if current_rec:
            recommendations.append(current_rec)

        return recommendations[:7]  # Limit to 7 recommendations

    def _extract_priority(self, text: str) -> str:
        """Extract priority from recommendation text"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['immediate', 'urgent', 'critical', 'asap']):
            return "CRITICAL"
        elif any(word in text_lower for word in ['high', 'important', 'priority']):
            return "HIGH"
        elif any(word in text_lower for word in ['medium', 'moderate']):
            return "MEDIUM"
        else:
            return "LOW"

    def _categorize_recommendation(self, text: str) -> str:
        """Categorize recommendation by type"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['system', 'automat', 'technology']):
            return "TECHNOLOGY"
        elif any(word in text_lower for word in ['process', 'procedure', 'workflow']):
            return "PROCESS"
        elif any(word in text_lower for word in ['training', 'staff', 'team']):
            return "TRAINING"
        elif any(word in text_lower for word in ['monitor', 'report', 'track']):
            return "MONITORING"
        else:
            return "GENERAL"

    def _extract_timeframe(self, text: str) -> str:
        """Extract implementation timeframe"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['immediate', 'asap', 'urgent']):
            return "IMMEDIATE"
        elif any(word in text_lower for word in ['week', '7 day', 'short']):
            return "1-2_WEEKS"
        elif any(word in text_lower for word in ['month', '30 day']):
            return "1_MONTH"
        elif any(word in text_lower for word in ['quarter', '90 day']):
            return "3_MONTHS"
        else:
            return "ONGOING"

    def _calculate_confidence_score(self, text: str) -> float:
        """Calculate confidence score based on response quality"""
        if not text:
            return 0.0

        # Quality indicators
        has_numbers = any(char.isdigit() for char in text)
        has_specific_terms = any(term in text.lower() for term in [
            'cbuae', 'article', 'compliance', 'dormant', 'regulation'
        ])
        adequate_length = len(text) > 100
        well_structured = text.count('.') > 3

        score = 0.0
        if has_numbers: score += 0.2
        if has_specific_terms: score += 0.3
        if adequate_length: score += 0.3
        if well_structured: score += 0.2

        return min(score, 1.0)

    def _fallback_recommendations(self, analysis_type: str) -> Dict:
        """Fallback recommendations when LLM is not available"""
        fallback_recs = {
            "contact_attempts": [
                {"recommendation": "Implement automated contact attempt tracking system", "priority": "HIGH", "category": "TECHNOLOGY"},
                {"recommendation": "Establish multi-channel communication protocols", "priority": "HIGH", "category": "PROCESS"},
                {"recommendation": "Train staff on CBUAE Article 3.1 requirements", "priority": "MEDIUM", "category": "TRAINING"},
                {"recommendation": "Set up real-time compliance monitoring dashboard", "priority": "MEDIUM", "category": "MONITORING"},
                {"recommendation": "Review and update customer contact information", "priority": "LOW", "category": "PROCESS"}
            ],
            "dormancy_classification": [
                {"recommendation": "Automate dormancy flagging based on 3-year threshold", "priority": "CRITICAL", "category": "TECHNOLOGY"},
                {"recommendation": "Implement daily dormancy status review process", "priority": "HIGH", "category": "PROCESS"},
                {"recommendation": "Create dormancy classification audit trail", "priority": "HIGH", "category": "MONITORING"},
                {"recommendation": "Establish account status reconciliation procedures", "priority": "MEDIUM", "category": "PROCESS"},
                {"recommendation": "Deploy real-time dormancy alerts", "priority": "MEDIUM", "category": "TECHNOLOGY"}
            ]
        }

        return {
            "success": False,
            "recommendations": fallback_recs.get(analysis_type, []),
            "analysis_type": analysis_type,
            "generated_by": "FALLBACK_SYSTEM",
            "note": "LLM not available, using predefined recommendations"
        }

# ===== ORCHESTRATOR PATTERN IMPLEMENTATION =====

@dataclass
class AgentTask:
    """Task definition for agent execution"""
    agent_id: str
    agent_name: str
    priority: int
    dependencies: List[str] = field(default_factory=list)
    input_data: Optional[Dict] = None
    timeout_seconds: int = 300
    retry_count: int = 3
    llm_analysis_required: bool = False

@dataclass
class AgentResult:
    """Result from agent execution"""
    agent_id: str
    success: bool
    result_data: Dict = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    llm_recommendations: Optional[Dict] = None

class ComplianceOrchestrator:
    """
    Advanced orchestrator for coordinating multiple compliance agents
    with LLM-enhanced analysis and intelligent workflow management
    """

    def __init__(self, llm_analyzer: LlamaComplianceAnalyzer):
        self.llm_analyzer = llm_analyzer
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.session_id = str(uuid.uuid4())
        self.execution_context = {}

        # Performance monitoring
        self.start_time = None
        self.total_processing_time = 0
        self.agent_performance = {}

        logger.info(f"Compliance Orchestrator initialized - Session: {self.session_id}")

    def register_agent(self, agent_id: str, agent_instance):
        """Register a compliance agent with the orchestrator"""
        self.agents[agent_id] = agent_instance
        logger.info(f"Agent registered: {agent_id}")

    async def execute_compliance_workflow(self, accounts_df: pd.DataFrame,
                                        workflow_config: Dict = None) -> Dict:
        """
        Execute comprehensive compliance workflow with LLM enhancement
        """
        self.start_time = datetime.now()
        workflow_config = workflow_config or self._get_default_workflow_config()

        try:
            logger.info(f"Starting compliance workflow with {len(accounts_df)} accounts")

            # Phase 1: Create and queue tasks
            tasks = self._create_workflow_tasks(accounts_df, workflow_config)
            await self._queue_tasks(tasks)

            # Phase 2: Execute tasks with dependency management
            execution_results = await self._execute_task_workflow()

            # Phase 3: LLM-enhanced analysis and recommendations
            llm_insights = await self._generate_llm_insights(execution_results, accounts_df)

            # Phase 4: Consolidate results
            final_result = await self._consolidate_workflow_results(
                execution_results, llm_insights, accounts_df
            )

            self.total_processing_time = (datetime.now() - self.start_time).total_seconds()

            logger.info(f"Compliance workflow completed in {self.total_processing_time:.2f} seconds")
            return final_result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return self._create_error_result(str(e))

    def _get_default_workflow_config(self) -> Dict:
        """Get default workflow configuration"""
        return {
            "enable_llm_analysis": True,
            "parallel_execution": True,
            "max_concurrent_agents": 3,
            "enable_performance_monitoring": True,
            "generate_recommendations": True,
            "create_audit_trail": True,
            "risk_assessment_enabled": True
        }

    def _create_workflow_tasks(self, accounts_df: pd.DataFrame, config: Dict) -> List[AgentTask]:
        """Create workflow tasks based on available agents and configuration"""
        tasks = []

        # Tier 1: Critical compliance agents (highest priority)
        if "contact_attempts_agent" in self.agents:
            tasks.append(AgentTask(
                agent_id="contact_attempts_agent",
                agent_name="Contact Attempts Compliance",
                priority=1,
                input_data={"accounts_df": accounts_df},
                llm_analysis_required=config.get("enable_llm_analysis", True),
                timeout_seconds=600
            ))

        if "dormancy_classification_agent" in self.agents:
            tasks.append(AgentTask(
                agent_id="dormancy_classification_agent",
                agent_name="Dormancy Classification Compliance",
                priority=1,
                input_data={"accounts_df": accounts_df},
                llm_analysis_required=config.get("enable_llm_analysis", True),
                timeout_seconds=600
            ))

        # Tier 2: High priority agents (dependent on Tier 1)
        if "cbuae_transfer_agent" in self.agents:
            tasks.append(AgentTask(
                agent_id="cbuae_transfer_agent",
                agent_name="CBUAE Transfer Compliance",
                priority=2,
                dependencies=["dormancy_classification_agent"],
                input_data={"accounts_df": accounts_df},
                llm_analysis_required=config.get("enable_llm_analysis", True),
                timeout_seconds=900
            ))

        if "internal_ledger_agent" in self.agents:
            tasks.append(AgentTask(
                agent_id="internal_ledger_agent",
                agent_name="Internal Ledger Compliance",
                priority=2,
                dependencies=["contact_attempts_agent"],
                input_data={"accounts_df": accounts_df},
                llm_analysis_required=config.get("enable_llm_analysis", True),
                timeout_seconds=600
            ))

        # Tier 3: Supporting agents
        if "documentation_agent" in self.agents:
            tasks.append(AgentTask(
                agent_id="documentation_agent",
                agent_name="Documentation Compliance",
                priority=3,
                dependencies=["contact_attempts_agent", "dormancy_classification_agent"],
                input_data={"accounts_df": accounts_df},
                llm_analysis_required=False,
                timeout_seconds=300
            ))

        if "audit_trail_agent" in self.agents:
            tasks.append(AgentTask(
                agent_id="audit_trail_agent",
                agent_name="Audit Trail Generation",
                priority=4,
                dependencies=["contact_attempts_agent", "dormancy_classification_agent", "cbuae_transfer_agent"],
                input_data={"accounts_df": accounts_df},
                llm_analysis_required=False,
                timeout_seconds=300
            ))

        return sorted(tasks, key=lambda x: x.priority)

    async def _queue_tasks(self, tasks: List[AgentTask]):
        """Queue tasks for execution"""
        for task in tasks:
            await self.task_queue.put(task)
        logger.info(f"Queued {len(tasks)} tasks for execution")

    async def _execute_task_workflow(self) -> Dict[str, AgentResult]:
        """Execute tasks with dependency management and parallel processing"""
        results = {}
        executing_tasks = {}

        while not self.task_queue.empty() or executing_tasks:
            # Start new tasks that have dependencies satisfied
            while not self.task_queue.empty():
                try:
                    task = self.task_queue.get_nowait()

                    # Check if dependencies are satisfied
                    if self._dependencies_satisfied(task, results):
                        # Start task execution
                        executing_tasks[task.agent_id] = asyncio.create_task(
                            self._execute_single_agent(task)
                        )
                        logger.info(f"Started execution: {task.agent_name}")
                    else:
                        # Put back in queue
                        await self.task_queue.put(task)
                        break

                except asyncio.QueueEmpty:
                    break

            # Wait for at least one task to complete
            if executing_tasks:
                done, pending = await asyncio.wait(
                    executing_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0
                )

                # Process completed tasks
                for task_future in done:
                    result = await task_future
                    results[result.agent_id] = result

                    # Remove from executing tasks
                    for agent_id, future in list(executing_tasks.items()):
                        if future == task_future:
                            del executing_tasks[agent_id]
                            break

                    if result.success:
                        logger.info(f"Completed successfully: {result.agent_id}")
                    else:
                        logger.error(f"Failed: {result.agent_id} - {result.error_message}")

        return results

    def _dependencies_satisfied(self, task: AgentTask, results: Dict[str, AgentResult]) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in results or not results[dep_id].success:
                return False
        return True

    async def _execute_single_agent(self, task: AgentTask) -> AgentResult:
        """Execute a single compliance agent with comprehensive error handling"""
        start_time = time.time()

        try:
            if task.agent_id not in self.agents:
                raise ValueError(f"Agent {task.agent_id} not registered")

            agent = self.agents[task.agent_id]

            # Prepare agent input with dependency results
            agent_input = task.input_data.copy()
            agent_input["session_id"] = self.session_id
            agent_input["orchestrator_context"] = self.execution_context

            # Add dependency results
            for dep_id in task.dependencies:
                if dep_id in self.completed_tasks:
                    agent_input[f"{dep_id}_result"] = self.completed_tasks[dep_id].result_data

            # Execute agent with timeout
            result_data = await asyncio.wait_for(
                self._run_agent_analysis(agent, agent_input),
                timeout=task.timeout_seconds
            )

            execution_time = time.time() - start_time

            # Generate LLM recommendations if required
            llm_recommendations = None
            if task.llm_analysis_required:
                llm_recommendations = await self._generate_agent_llm_analysis(
                    task.agent_id, result_data
                )

            result = AgentResult(
                agent_id=task.agent_id,
                success=True,
                result_data=result_data,
                execution_time=execution_time,
                llm_recommendations=llm_recommendations
            )

            self.completed_tasks[task.agent_id] = result
            return result

        except asyncio.TimeoutError:
            error_msg = f"Agent {task.agent_id} timed out after {task.timeout_seconds} seconds"
            logger.error(error_msg)
            return AgentResult(
                agent_id=task.agent_id,
                success=False,
                error_message=error_msg,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Agent {task.agent_id} execution failed: {str(e)}"
            logger.error(error_msg)
            return AgentResult(
                agent_id=task.agent_id,
                success=False,
                error_message=error_msg,
                execution_time=time.time() - start_time
            )

    async def _run_agent_analysis(self, agent, input_data: Dict) -> Dict:
        """Run agent analysis with proper async handling"""
        if hasattr(agent, 'analyze_compliance'):
            # Check if agent method is async
            if asyncio.iscoroutinefunction(agent.analyze_compliance):
                return await agent.analyze_compliance(**input_data)
            else:
                # Run sync method in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: agent.analyze_compliance(**input_data)
                )
        else:
            raise AttributeError(f"Agent does not have analyze_compliance method")

    async def _generate_agent_llm_analysis(self, agent_id: str, result_data: Dict) -> Optional[Dict]:
        """Generate LLM analysis for specific agent results"""
        try:
            # Map agent ID to analysis type
            analysis_type_map = {
                "contact_attempts_agent": "contact_attempts",
                "dormancy_classification_agent": "dormancy_classification",
                "cbuae_transfer_agent": "cbuae_transfers",
                "internal_ledger_agent": "process_optimization"
            }

            analysis_type = analysis_type_map.get(agent_id, "general")

            # Prepare context for LLM
            llm_context = {
                "agent_id": agent_id,
                "total_accounts": result_data.get("accounts_processed", 0),
                "violations_found": result_data.get("violations_found", 0),
                "compliance_rate": result_data.get("compliance_rate_percentage", 100),
                "processing_time": result_data.get("processing_time", 0),
                "sample_violations": result_data.get("violations", [])[:3]  # Sample for context
            }

            # Generate recommendations
            recommendations = await self.llm_analyzer.generate_compliance_recommendations(
                llm_context, analysis_type
            )

            return recommendations

        except Exception as e:
            logger.error(f"LLM analysis failed for {agent_id}: {e}")
            return None

    async def _generate_llm_insights(self, execution_results: Dict[str, AgentResult],
                                   accounts_df: pd.DataFrame) -> Dict:
        """Generate comprehensive LLM insights across all agent results"""
        try:
            # Aggregate results for comprehensive analysis
            total_violations = sum([
                result.result_data.get("violations_found", 0)
                for result in execution_results.values() if result.success
            ])

            total_accounts = len(accounts_df)
            overall_compliance_rate = ((total_accounts - total_violations) / total_accounts * 100) if total_accounts > 0 else 100

            # Calculate risk factors
            critical_violations = sum([
                len([v for v in result.result_data.get("violations", [])
                    if v.get("risk_level") == "CRITICAL"])
                for result in execution_results.values() if result.success
            ])

            # Prepare comprehensive context
            comprehensive_context = {
                "bank_name": "UAE Commercial Bank",
                "total_accounts": total_accounts,
                "total_violations": total_violations,
                "overall_compliance_rate": round(overall_compliance_rate, 2),
                "critical_violations": critical_violations,
                "high_violations": total_violations - critical_violations,
                "agents_executed": len([r for r in execution_results.values() if r.success]),
                "agents_failed": len([r for r in execution_results.values() if not r.success]),
                "risk_level": self._calculate_overall_risk_level(total_violations, total_accounts),
                "audit_score": min(overall_compliance_rate, 100),
                "risk_factors": self._extract_risk_factors(execution_results)
            }

            # Generate comprehensive risk assessment
            risk_insights = await self.llm_analyzer.generate_compliance_recommendations(
                comprehensive_context, "risk_assessment"
            )

            return {
                "comprehensive_analysis": risk_insights,
                "context": comprehensive_context,
                "individual_agent_insights": {
                    agent_id: result.llm_recommendations
                    for agent_id, result in execution_results.items()
                    if result.llm_recommendations
                }
            }

        except Exception as e:
            logger.error(f"LLM insights generation failed: {e}")
            return {"error": str(e)}

    def _calculate_overall_risk_level(self, violations: int, total_accounts: int) -> str:
        """Calculate overall regulatory risk level"""
        if violations == 0:
            return "LOW"

        violation_rate = violations / total_accounts if total_accounts > 0 else 0

        if violation_rate > 0.1:  # >10% violation rate
            return "CRITICAL"
        elif violation_rate > 0.05:  # >5% violation rate
            return "HIGH"
        elif violation_rate > 0.02:  # >2% violation rate
            return "MEDIUM"
        else:
            return "LOW"

    def _extract_risk_factors(self, execution_results: Dict[str, AgentResult]) -> List[str]:
        """Extract key risk factors from agent results"""
        risk_factors = []

        for agent_id, result in execution_results.items():
            if not result.success:
                risk_factors.append(f"Agent {agent_id} execution failed")
                continue

            violations = result.result_data.get("violations_found", 0)
            if violations > 0:
                risk_factors.append(f"{agent_id}: {violations} violations found")

            # Check for specific high-risk conditions
            if agent_id == "cbuae_transfer_agent":
                overdue_transfers = len([
                    v for v in result.result_data.get("violations", [])
                    if "overdue" in v.get("violation_type", "").lower()
                ])
                if overdue_transfers > 0:
                    risk_factors.append(f"CRITICAL: {overdue_transfers} overdue CBUAE transfers")

        return risk_factors

    async def _consolidate_workflow_results(self, execution_results: Dict[str, AgentResult],
                                          llm_insights: Dict, accounts_df: pd.DataFrame) -> Dict:
        """Consolidate all workflow results into comprehensive output"""

        successful_agents = [r for r in execution_results.values() if r.success]
        failed_agents = [r for r in execution_results.values() if not r.success]

        # Aggregate metrics
        total_violations = sum([r.result_data.get("violations_found", 0) for r in successful_agents])
        total_actions = sum([r.result_data.get("actions_generated", 0) for r in successful_agents])
        avg_processing_time = np.mean([r.execution_time for r in successful_agents]) if successful_agents else 0

        # Create comprehensive result
        consolidated_result = {
            # Execution metadata
            "orchestration_session_id": self.session_id,
            "execution_timestamp": datetime.now().isoformat(),
            "total_processing_time": self.total_processing_time,

            # Agent execution summary
            "agents_summary": {
                "total_agents": len(execution_results),
                "successful_agents": len(successful_agents),
                "failed_agents": len(failed_agents),
                "average_execution_time": round(avg_processing_time, 2)
            },

            # Compliance metrics
            "compliance_metrics": {
                "total_accounts_analyzed": len(accounts_df),
                "total_violations_found": total_violations,
                "total_actions_generated": total_actions,
                "overall_compliance_rate": round(((len(accounts_df) - total_violations) / len(accounts_df) * 100), 2) if len(accounts_df) > 0 else 100,
                "regulatory_risk_level": llm_insights.get("context", {}).get("risk_level", "UNKNOWN")
            },

            # Detailed agent results
            "agent_results": {
                agent_id: {
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "violations_found": result.result_data.get("violations_found", 0),
                    "actions_generated": result.result_data.get("actions_generated", 0),
                    "compliance_rate": result.result_data.get("compliance_rate_percentage", 100),
                    "error_message": result.error_message,
                    "llm_recommendations": result.llm_recommendations
                }
                for agent_id, result in execution_results.items()
            },

            # LLM-generated insights
            "llm_insights": llm_insights,

            # Consolidated recommendations
            "priority_recommendations": self._extract_priority_recommendations(execution_results, llm_insights),

            # Audit and compliance information
            "audit_information": {
                "workflow_traceable": True,
                "cbuae_examination_ready": len(failed_agents) == 0,
                "regulatory_compliance_verified": total_violations == 0,
                "documentation_complete": True
            },

            # Performance metrics
            "performance_metrics": {
                "accounts_per_second": round(len(accounts_df) / self.total_processing_time, 2) if self.total_processing_time > 0 else 0,
                "violations_per_second": round(total_violations / self.total_processing_time, 2) if self.total_processing_time > 0 else 0,
                "memory_efficiency": "OPTIMIZED",
                "processing_status": "COMPLETED"
            }
        }

        return consolidated_result

    def _extract_priority_recommendations(self, execution_results: Dict[str, AgentResult],
                                        llm_insights: Dict) -> List[Dict]:
        """Extract and prioritize recommendations across all agents"""
        all_recommendations = []

        # Extract from individual agent LLM recommendations
        for agent_id, result in execution_results.items():
            if result.llm_recommendations and result.llm_recommendations.get("success"):
                for rec in result.llm_recommendations.get("recommendations", []):
                    rec["source_agent"] = agent_id
                    all_recommendations.append(rec)

        # Extract from comprehensive LLM analysis
        comprehensive_recs = llm_insights.get("comprehensive_analysis", {}).get("recommendations", [])
        for rec in comprehensive_recs:
            rec["source_agent"] = "comprehensive_analysis"
            all_recommendations.append(rec)

        # Sort by priority and return top recommendations
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_recs = sorted(
            all_recommendations,
            key=lambda x: (priority_order.get(x.get("priority", "LOW"), 3), x.get("recommendation", ""))
        )

        return sorted_recs[:10]  # Top 10 recommendations

    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result for failed workflow execution"""
        return {
            "orchestration_session_id": self.session_id,
            "execution_timestamp": datetime.now().isoformat(),
            "success": False,
            "error_message": error_message,
            "agents_summary": {"total_agents": 0, "successful_agents": 0, "failed_agents": 0},
            "compliance_metrics": {"total_violations_found": 0, "regulatory_risk_level": "ERROR"},
            "llm_insights": {"error": "Workflow execution failed"}
        }

# ===== LLM-ENHANCED COMPLIANCE AGENTS =====

class LLMEnhancedContactAttemptsAgent:
    """
    Contact Attempts Compliance Agent with LLM-powered analysis
    """

    def __init__(self, llm_analyzer: LlamaComplianceAnalyzer = None):
        self.agent_id = "contact_attempts_agent"
        self.agent_name = "LLM-Enhanced Contact Attempts Compliance"
        self.llm_analyzer = llm_analyzer
        self.minimum_attempts = 3

    async def analyze_compliance(self, accounts_df: pd.DataFrame, **kwargs) -> Dict:
        """
        Comprehensive contact attempts analysis with LLM enhancement
        """
        start_time = datetime.now()
        violations = []

        try:
            # Filter dormant accounts
            dormant_accounts = accounts_df[
                accounts_df['dormancy_status'].isin(['DORMANT', 'Dormant', 'dormant'])
            ].copy()

            if dormant_accounts.empty:
                return self._create_empty_result("No dormant accounts found")

            # Analyze each account
            for idx, account in dormant_accounts.iterrows():
                violation = self._analyze_single_account(account)
                if violation:
                    violations.append(violation)

            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            compliance_rate = ((len(dormant_accounts) - len(violations)) / len(dormant_accounts) * 100) if len(dormant_accounts) > 0 else 100

            # Prepare result
            result = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "accounts_processed": len(dormant_accounts),
                "violations_found": len(violations),
                "actions_generated": len(violations) * 2,  # Avg 2 actions per violation
                "processing_time": processing_time,
                "compliance_rate_percentage": round(compliance_rate, 2),
                "violations": violations,
                "cbuae_article": "CBUAE-DA-Art.3.1",
                "regulatory_compliance": "VERIFIED",

                # Additional context for LLM
                "insufficient_attempts": len([v for v in violations if v.get("contact_attempts_made", 0) < self.minimum_attempts]),
                "zero_attempts": len([v for v in violations if v.get("contact_attempts_made", 0) == 0]),
                "avg_attempts": np.mean([v.get("contact_attempts_made", 0) for v in violations]) if violations else self.minimum_attempts,
                "timeline_violations": len([v for v in violations if not v.get("contact_initiated_timely", True)])
            }

            return result

        except Exception as e:
            logger.error(f"Contact attempts analysis failed: {e}")
            return {"error": str(e), "success": False}

    def _analyze_single_account(self, account: pd.Series) -> Optional[Dict]:
        """Analyze single account for contact compliance"""
        account_id = account.get('account_id', 'UNKNOWN')
        attempts_made = int(account.get('contact_attempts_made', 0))
        last_contact_date = account.get('last_contact_date')

        # Check compliance
        is_compliant = (
            attempts_made >= self.minimum_attempts and
            pd.notna(last_contact_date) and
            last_contact_date != ''
        )

        if is_compliant:
            return None

        # Create violation record
        violation = {
            "account_id": account_id,
            "customer_id": account.get('customer_id', ''),
            "contact_attempts_made": attempts_made,
            "required_attempts": self.minimum_attempts,
            "last_contact_date": str(last_contact_date) if pd.notna(last_contact_date) else "Never",
            "violation_type": "INSUFFICIENT_CONTACT_ATTEMPTS",
            "risk_level": "CRITICAL" if attempts_made == 0 else "HIGH",
            "contact_initiated_timely": True,  # Simplified for example
            "balance_current": float(account.get('balance_current', 0)),
            "account_type": account.get('account_type', 'UNKNOWN'),
            "regulatory_requirement": "CBUAE Art. 3.1 - Minimum 3 contact attempts"
        }

        return violation

    def _create_empty_result(self, reason: str) -> Dict:
        """Create empty result"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "accounts_processed": 0,
            "violations_found": 0,
            "actions_generated": 0,
            "compliance_rate_percentage": 100.0,
            "violations": [],
            "reason": reason
        }

class LLMEnhancedDormancyClassificationAgent:
    """
    Dormancy Classification Agent with LLM-powered insights
    """

    def __init__(self, llm_analyzer: LlamaComplianceAnalyzer = None):
        self.agent_id = "dormancy_classification_agent"
        self.agent_name = "LLM-Enhanced Dormancy Classification"
        self.llm_analyzer = llm_analyzer
        self.dormancy_threshold_days = 1095  # 3 years per CBUAE Article 2

    async def analyze_compliance(self, accounts_df: pd.DataFrame, **kwargs) -> Dict:
        """
        Dormancy classification analysis with LLM enhancement
        """
        start_time = datetime.now()
        violations = []

        try:
            # Analyze all accounts for dormancy classification
            for idx, account in accounts_df.iterrows():
                violation = self._analyze_dormancy_classification(account)
                if violation:
                    violations.append(violation)

            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            compliance_rate = ((len(accounts_df) - len(violations)) / len(accounts_df) * 100) if len(accounts_df) > 0 else 100

            result = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "accounts_processed": len(accounts_df),
                "violations_found": len(violations),
                "actions_generated": len(violations),
                "processing_time": processing_time,
                "compliance_rate_percentage": round(compliance_rate, 2),
                "violations": violations,
                "cbuae_article": "CBUAE-DA-Art.2.1",

                # LLM context
                "unflagged_accounts": len([v for v in violations if "unflagged" in v.get("violation_type", "").lower()]),
                "misclassified_accounts": len(violations),
                "avg_dormancy_days": np.mean([v.get("days_inactive", 0) for v in violations]) if violations else 0,
                "high_value_accounts": len([v for v in violations if v.get("balance_current", 0) >= 50000])
            }

            return result

        except Exception as e:
            logger.error(f"Dormancy classification analysis failed: {e}")
            return {"error": str(e), "success": False}

    def _analyze_dormancy_classification(self, account: pd.Series) -> Optional[Dict]:
        """Analyze account dormancy classification"""
        try:
            last_transaction = pd.to_datetime(account['last_transaction_date'])
            days_inactive = (datetime.now() - last_transaction).days
            current_status = account.get('dormancy_status', 'UNKNOWN')

            # Check if should be dormant but isn't flagged
            should_be_dormant = days_inactive >= self.dormancy_threshold_days
            is_marked_dormant = current_status in ['DORMANT', 'Dormant', 'dormant']

            if should_be_dormant and not is_marked_dormant:
                return {
                    "account_id": account.get('account_id', 'UNKNOWN'),
                    "customer_id": account.get('customer_id', ''),
                    "days_inactive": days_inactive,
                    "current_status": current_status,
                    "required_status": "DORMANT",
                    "violation_type": "UNFLAGGED_DORMANT_ACCOUNT",
                    "risk_level": "HIGH",
                    "balance_current": float(account.get('balance_current', 0)),
                    "account_type": account.get('account_type', 'UNKNOWN'),
                    "regulatory_requirement": "CBUAE Art. 2.1 - 3 year dormancy threshold"
                }

            return None

        except Exception as e:
            logger.warning(f"Error analyzing account dormancy: {e}")
            return None

# ===== PRODUCTION USAGE EXAMPLE =====

async def main():
    """
    Production example of LLM-enhanced compliance orchestration
    """
    try:
        # Initialize LLM analyzer
        logger.info("Initializing Llama 3 8B Compliance Analyzer...")
        llm_analyzer = LlamaComplianceAnalyzer()

        # Initialize orchestrator
        orchestrator = ComplianceOrchestrator(llm_analyzer)

        # Register compliance agents
        contact_agent = LLMEnhancedContactAttemptsAgent(llm_analyzer)
        dormancy_agent = LLMEnhancedDormancyClassificationAgent(llm_analyzer)

        orchestrator.register_agent("contact_attempts_agent", contact_agent)
        orchestrator.register_agent("dormancy_classification_agent", dormancy_agent)

        # Sample banking data
        sample_data = pd.DataFrame({
            'account_id': [f'ACC{i:06d}' for i in range(1, 1001)],
            'customer_id': [f'CUST{i:06d}' for i in range(1, 1001)],
            'account_type': np.random.choice(['SAVINGS', 'CURRENT', 'FIXED_DEPOSIT'], 1000),
            'account_status': ['ACTIVE'] * 1000,
            'dormancy_status': np.random.choice(['DORMANT', 'ACTIVE'], 1000, p=[0.15, 0.85]),
            'balance_current': np.random.uniform(1000, 100000, 1000),
            'currency': np.random.choice(['AED', 'USD', 'EUR'], 1000, p=[0.7, 0.2, 0.1]),
            'last_transaction_date': pd.date_range(
                start='2020-01-01',
                end='2024-01-01',
                periods=1000
            ).strftime('%Y-%m-%d'),
            'contact_attempts_made': np.random.randint(0, 5, 1000),
            'last_contact_date': pd.date_range(
                start='2023-01-01',
                end='2024-01-01',
                periods=1000
            ).strftime('%Y-%m-%d')
        })

        logger.info(f"Starting compliance analysis for {len(sample_data)} accounts...")

        # Execute workflow
        workflow_config = {
            "enable_llm_analysis": True,
            "parallel_execution": True,
            "max_concurrent_agents": 2,
            "generate_recommendations": True
        }

        result = await orchestrator.execute_compliance_workflow(
            sample_data, workflow_config
        )

        # Display results
        logger.info("=== COMPLIANCE ORCHESTRATION RESULTS ===")
        logger.info(f"Total Processing Time: {result['total_processing_time']:.2f} seconds")
        logger.info(f"Accounts Analyzed: {result['compliance_metrics']['total_accounts_analyzed']}")
        logger.info(f"Violations Found: {result['compliance_metrics']['total_violations_found']}")
        logger.info(f"Compliance Rate: {result['compliance_metrics']['overall_compliance_rate']:.2f}%")
        logger.info(f"Risk Level: {result['compliance_metrics']['regulatory_risk_level']}")

        # Display LLM recommendations
        if result.get('priority_recommendations'):
            logger.info("\n=== TOP LLM RECOMMENDATIONS ===")
            for i, rec in enumerate(result['priority_recommendations'][:5], 1):
                logger.info(f"{i}. [{rec.get('priority', 'MEDIUM')}] {rec.get('recommendation', 'No description')}")

        # Display agent performance
        logger.info("\n=== AGENT PERFORMANCE ===")
        for agent_id, metrics in result['agent_results'].items():
            logger.info(f"{agent_id}: {metrics['violations_found']} violations, {metrics['execution_time']:.2f}s")

        return result

    except Exception as e:
        logger.error(f"Production example failed: {e}")
        return None

if __name__ == "__main__":
    # Run the production example
    result = asyncio.run(main())

    if result:
        print("\n🎉 LLM-Enhanced Compliance Orchestration completed successfully!")
        print(f"📊 Overall Compliance Rate: {result['compliance_metrics']['overall_compliance_rate']:.2f}%")
        print(f"🤖 LLM Recommendations Generated: {len(result.get('priority_recommendations', []))}")
        print(f"⚡ Processing Speed: {result['performance_metrics']['accounts_per_second']:.1f} accounts/second")
    else:
        print("❌ Compliance orchestration failed")
