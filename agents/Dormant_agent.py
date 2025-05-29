"""
Dormant_Agent.py - Enhanced Dormancy Analysis Agent
Integrated with Hybrid Memory Agent, LangGraph, and MCP Tools
Implements CBUAE dormancy regulations with intelligent pattern recognition
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
from pathlib import Path
import hashlib

# Import the dormant analysis functions from the original dormant.py
from dormant import (
    check_safe_deposit_dormancy,
    check_investment_inactivity,
    check_fixed_deposit_inactivity,
    check_demand_deposit_inactivity,
    check_unclaimed_payment_instruments,
    check_eligible_for_cb_transfer,
    check_art3_process_needed,
    check_contact_attempts_needed,
    check_high_value_dormant_accounts,
    check_dormant_to_active_transitions,
    run_all_dormant_identification_checks
)

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable, Client as LangSmithClient

# MCP imports
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dormancy Analysis States and Models
class DormancyStatus(Enum):
    ACTIVE = "active"
    APPROACHING_DORMANCY = "approaching_dormancy"
    DORMANT = "dormant"
    ELIGIBLE_FOR_TRANSFER = "eligible_for_transfer"
    REACTIVATED = "reactivated"
    UNKNOWN = "unknown"


class AnalysisStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class ComplianceArticle(Enum):
    ARTICLE_2_1_1 = "article_2_1_1"  # Demand deposits
    ARTICLE_2_2 = "article_2_2"  # Fixed deposits
    ARTICLE_2_3 = "article_2_3"  # Investment accounts
    ARTICLE_2_4 = "article_2_4"  # Payment instruments
    ARTICLE_2_6 = "article_2_6"  # Safe deposit boxes
    ARTICLE_3 = "article_3"  # Contact procedures
    ARTICLE_8 = "article_8"  # CB transfer eligibility


@dataclass
class DormancyAnalysisState:
    """Comprehensive state for dormancy analysis workflow"""
    session_id: str
    user_id: str
    analysis_id: str
    timestamp: datetime

    # Input data
    processed_data: Optional[Dict] = None
    report_date: Optional[str] = None
    analysis_parameters: Optional[Dict] = None

    # Analysis results
    dormancy_results: Optional[Dict] = None
    compliance_flags: List[str] = None
    risk_indicators: Dict = None
    dormancy_summary: Optional[Dict] = None

    # Status tracking
    analysis_status: AnalysisStatus = AnalysisStatus.PENDING
    total_accounts_analyzed: int = 0
    dormant_accounts_found: int = 0
    high_risk_accounts: int = 0

    # Memory context
    memory_context: Dict = None
    retrieved_patterns: Dict = None
    historical_insights: Dict = None

    # Performance and audit
    analysis_log: List[Dict] = None
    error_log: List[Dict] = None
    performance_metrics: Dict = None
    compliance_breakdown: Dict = None

    def __post_init__(self):
        if self.compliance_flags is None:
            self.compliance_flags = []
        if self.risk_indicators is None:
            self.risk_indicators = {}
        if self.memory_context is None:
            self.memory_context = {}
        if self.retrieved_patterns is None:
            self.retrieved_patterns = {}
        if self.historical_insights is None:
            self.historical_insights = {}
        if self.analysis_log is None:
            self.analysis_log = []
        if self.error_log is None:
            self.error_log = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.compliance_breakdown is None:
            self.compliance_breakdown = {}


# Enhanced Dormancy Pattern Recognition
class DormancyPatternAnalyzer:
    """AI-enhanced pattern recognition for dormancy analysis"""

    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

        # Pattern recognition thresholds
        self.pattern_thresholds = {
            "seasonal_dormancy": 0.3,
            "customer_behavior_similarity": 0.7,
            "reactivation_probability": 0.5,
            "risk_escalation_threshold": 0.8
        }

        # Customer behavior patterns
        self.behavior_patterns = {
            "seasonal_patterns": {},
            "communication_patterns": {},
            "transaction_patterns": {},
            "reactivation_patterns": {}
        }

    @traceable(name="analyze_dormancy_patterns")
    async def analyze_patterns(self, df: pd.DataFrame, historical_data: Dict = None) -> Dict:
        """Analyze dormancy patterns using ML-enhanced insights"""
        try:
            pattern_analysis = {
                "seasonal_patterns": await self._analyze_seasonal_patterns(df),
                "customer_segments": await self._analyze_customer_segments(df),
                "reactivation_probability": await self._predict_reactivation_probability(df, historical_data),
                "risk_indicators": await self._identify_risk_indicators(df),
                "communication_effectiveness": await self._analyze_communication_patterns(df, historical_data)
            }

            # Generate insights based on patterns
            insights = await self._generate_pattern_insights(pattern_analysis)

            return {
                "pattern_analysis": pattern_analysis,
                "insights": insights,
                "recommendations": await self._generate_pattern_recommendations(pattern_analysis)
            }

        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal dormancy patterns"""
        if df.empty or 'Date_Last_Cust_Initiated_Activity' not in df.columns:
            return {"seasonal_insights": "Insufficient data for seasonal analysis"}

        # Convert to datetime for analysis
        df['activity_date'] = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'], errors='coerce')
        valid_dates = df['activity_date'].dropna()

        if valid_dates.empty:
            return {"seasonal_insights": "No valid activity dates found"}

        # Extract seasonal patterns
        seasonal_data = {
            "quarters": valid_dates.dt.quarter.value_counts().to_dict(),
            "months": valid_dates.dt.month.value_counts().to_dict(),
            "days_of_week": valid_dates.dt.dayofweek.value_counts().to_dict()
        }

        # Identify peak dormancy periods
        peak_dormancy_month = min(seasonal_data["months"], key=seasonal_data["months"].get)
        peak_activity_month = max(seasonal_data["months"], key=seasonal_data["months"].get)

        return {
            "seasonal_data": seasonal_data,
            "peak_dormancy_month": peak_dormancy_month,
            "peak_activity_month": peak_activity_month,
            "seasonal_variance": np.std(list(seasonal_data["months"].values()))
        }

    async def _analyze_customer_segments(self, df: pd.DataFrame) -> Dict:
        """Segment customers based on dormancy characteristics"""
        segments = {
            "high_value_dormant": 0,
            "long_term_dormant": 0,
            "recent_dormant": 0,
            "approaching_dormancy": 0
        }

        if df.empty:
            return segments

        # High value dormant (assuming Current_Balance > 25000)
        if 'Current_Balance' in df.columns:
            high_value = pd.to_numeric(df['Current_Balance'], errors='coerce') > 25000
            segments["high_value_dormant"] = high_value.sum()

        # Long-term dormant (>5 years)
        if 'Date_Last_Cust_Initiated_Activity' in df.columns:
            activity_dates = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'], errors='coerce')
            five_years_ago = datetime.now() - timedelta(days=5 * 365)
            long_term = activity_dates < five_years_ago
            segments["long_term_dormant"] = long_term.sum()

            # Recent dormant (3-5 years)
            three_years_ago = datetime.now() - timedelta(days=3 * 365)
            recent_dormant = (activity_dates < three_years_ago) & (activity_dates >= five_years_ago)
            segments["recent_dormant"] = recent_dormant.sum()

            # Approaching dormancy (2.5-3 years)
            approaching_threshold = datetime.now() - timedelta(days=int(2.5 * 365))
            approaching = (activity_dates < approaching_threshold) & (activity_dates >= three_years_ago)
            segments["approaching_dormancy"] = approaching.sum()

        return segments

    async def _predict_reactivation_probability(self, df: pd.DataFrame, historical_data: Dict = None) -> Dict:
        """Predict reactivation probability based on historical patterns"""
        if df.empty:
            return {"average_probability": 0.0, "confidence": 0.0}

        # Simple heuristic-based prediction (can be enhanced with ML models)
        probability_factors = {
            "account_type_factor": 0.5,
            "balance_factor": 0.3,
            "communication_factor": 0.4,
            "historical_factor": 0.6
        }

        # Account type influence
        if 'Account_Type' in df.columns:
            current_savings = df['Account_Type'].str.contains('Current|Saving', case=False, na=False).mean()
            probability_factors["account_type_factor"] = 0.3 + (current_savings * 0.4)

        # Balance influence
        if 'Current_Balance' in df.columns:
            avg_balance = pd.to_numeric(df['Current_Balance'], errors='coerce').mean()
            if avg_balance > 10000:
                probability_factors["balance_factor"] = 0.6
            elif avg_balance > 1000:
                probability_factors["balance_factor"] = 0.4
            else:
                probability_factors["balance_factor"] = 0.2

        # Historical reactivation data
        if historical_data and 'reactivation_rate' in historical_data:
            probability_factors["historical_factor"] = historical_data['reactivation_rate']

        average_probability = np.mean(list(probability_factors.values()))
        confidence = 1.0 - np.std(list(probability_factors.values()))

        return {
            "average_probability": round(average_probability, 3),
            "confidence": round(confidence, 3),
            "factors": probability_factors
        }

    async def _identify_risk_indicators(self, df: pd.DataFrame) -> Dict:
        """Identify risk indicators for dormant accounts"""
        risk_indicators = {
            "high_balance_risk": 0,
            "compliance_risk": 0,
            "operational_risk": 0,
            "reputational_risk": 0
        }

        if df.empty:
            return risk_indicators

        # High balance risk
        if 'Current_Balance' in df.columns:
            high_balances = pd.to_numeric(df['Current_Balance'], errors='coerce') > 100000
            risk_indicators["high_balance_risk"] = high_balances.sum()

        # Compliance risk (accounts requiring Article 3 process)
        if 'Expected_Requires_Article_3_Process' in df.columns:
            compliance_required = df['Expected_Requires_Article_3_Process'].str.lower().isin(['yes', 'true', '1'])
            risk_indicators["compliance_risk"] = compliance_required.sum()

        # Operational risk (accounts with contact attempts failed)
        if 'Bank_Contact_Attempted_Post_Dormancy_Trigger' in df.columns:
            failed_contacts = df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].str.lower().isin(['no', 'false', '0'])
            risk_indicators["operational_risk"] = failed_contacts.sum()

        return risk_indicators

    async def _analyze_communication_patterns(self, df: pd.DataFrame, historical_data: Dict = None) -> Dict:
        """Analyze effectiveness of communication attempts"""
        if df.empty:
            return {"effectiveness_rate": 0.0, "insights": "No data available"}

        communication_analysis = {
            "total_attempts": 0,
            "successful_contacts": 0,
            "effectiveness_rate": 0.0,
            "preferred_channels": {},
            "response_timeframes": {}
        }

        if 'Bank_Contact_Attempted_Post_Dormancy_Trigger' in df.columns:
            attempts = df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].str.lower().isin(['yes', 'true', '1'])
            communication_analysis["total_attempts"] = attempts.sum()

        if 'Date_Last_Customer_Communication_Any_Type' in df.columns:
            recent_communications = pd.to_datetime(df['Date_Last_Customer_Communication_Any_Type'], errors='coerce')
            recent_threshold = datetime.now() - timedelta(days=90)
            recent_responses = (recent_communications > recent_threshold).sum()
            communication_analysis["successful_contacts"] = recent_responses

        if communication_analysis["total_attempts"] > 0:
            communication_analysis["effectiveness_rate"] = (
                    communication_analysis["successful_contacts"] / communication_analysis["total_attempts"]
            )

        return communication_analysis

    async def _generate_pattern_insights(self, pattern_analysis: Dict) -> List[str]:
        """Generate actionable insights from pattern analysis"""
        insights = []

        # Seasonal insights
        seasonal = pattern_analysis.get("seasonal_patterns", {})
        if "peak_dormancy_month" in seasonal:
            insights.append(
                f"Peak dormancy occurs in month {seasonal['peak_dormancy_month']} - consider proactive outreach")

        # Customer segment insights
        segments = pattern_analysis.get("customer_segments", {})
        if segments.get("high_value_dormant", 0) > 0:
            insights.append(
                f"{segments['high_value_dormant']} high-value accounts are dormant - prioritize reactivation")

        # Reactivation probability insights
        reactivation = pattern_analysis.get("reactivation_probability", {})
        if reactivation.get("average_probability", 0) > 0.5:
            insights.append("Good reactivation potential - implement targeted campaigns")
        elif reactivation.get("average_probability", 0) < 0.3:
            insights.append("Low reactivation probability - consider transfer procedures")

        # Risk indicator insights
        risks = pattern_analysis.get("risk_indicators", {})
        total_risk = sum(risks.values())
        if total_risk > 0:
            insights.append(f"Total risk indicators: {total_risk} - enhance monitoring and controls")

        return insights

    async def _generate_pattern_recommendations(self, pattern_analysis: Dict) -> List[str]:
        """Generate recommendations based on pattern analysis"""
        recommendations = []

        # Communication effectiveness recommendations
        comm_analysis = pattern_analysis.get("communication_effectiveness", {})
        effectiveness = comm_analysis.get("effectiveness_rate", 0)

        if effectiveness < 0.3:
            recommendations.append("Low communication effectiveness - review contact strategies and channels")
        elif effectiveness > 0.7:
            recommendations.append("High communication effectiveness - scale successful contact methods")

        # Segment-based recommendations
        segments = pattern_analysis.get("customer_segments", {})
        if segments.get("approaching_dormancy", 0) > 0:
            recommendations.append("Implement proactive retention for accounts approaching dormancy")

        if segments.get("long_term_dormant", 0) > 10:
            recommendations.append("Review long-term dormant accounts for potential Central Bank transfer")

        # Risk-based recommendations
        risks = pattern_analysis.get("risk_indicators", {})
        if risks.get("high_balance_risk", 0) > 0:
            recommendations.append("Prioritize high-balance dormant accounts for immediate attention")

        if risks.get("compliance_risk", 0) > 0:
            recommendations.append("Ensure Article 3 compliance procedures are followed")

        return recommendations


# Enhanced Dormancy Analysis Agent
class DormancyAnalysisAgent:
    """Enhanced dormancy analysis agent with AI-powered insights"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.pattern_analyzer = DormancyPatternAnalyzer(memory_agent)
        self.langsmith_client = LangSmithClient()

        # CBUAE regulatory parameters
        self.regulatory_params = {
            "standard_inactivity_years": 3,
            "payment_instrument_unclaimed_years": 1,
            "sdb_unpaid_fees_years": 3,
            "eligibility_for_cb_transfer_years": 5,
            "high_value_threshold_aed": 25000,
            "article_3_waiting_period_days": 90
        }

        # Analysis configuration
        self.analysis_config = {
            "enable_pattern_analysis": True,
            "enable_risk_scoring": True,
            "enable_predictive_insights": True,
            "parallel_processing": True,
            "memory_enhanced_analysis": True
        }

    @traceable(name="dormancy_analysis_pre_hook")
    async def pre_analysis_hook(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Enhanced pre-analysis memory hook with comprehensive context retrieval"""
        try:
            # Retrieve dormancy analysis patterns
            dormancy_patterns = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": "dormancy_patterns",
                    "user_id": state.user_id
                }
            )

            if dormancy_patterns.get("success"):
                state.retrieved_patterns["dormancy"] = dormancy_patterns.get("data", {})
                logger.info("Retrieved dormancy patterns from knowledge memory")

            # Retrieve historical dormancy insights
            historical_insights = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": "historical_dormancy_insights",
                    "user_id": state.user_id
                }
            )

            if historical_insights.get("success"):
                state.historical_insights = historical_insights.get("data", {})
                logger.info("Retrieved historical dormancy insights")

            # Retrieve compliance benchmarks
            compliance_benchmarks = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": "compliance_benchmarks",
                    "user_id": state.user_id
                }
            )

            if compliance_benchmarks.get("success"):
                state.retrieved_patterns["compliance"] = compliance_benchmarks.get("data", {})

            # Retrieve user-specific analysis preferences
            user_preferences = await self.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "type": "analysis_preferences",
                    "user_id": state.user_id
                }
            )

            if user_preferences.get("success"):
                state.memory_context["preferences"] = user_preferences.get("data", {})

            # Log pre-analysis hook execution
            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_analysis_hook",
                "action": "memory_retrieval",
                "status": "completed",
                "patterns_retrieved": len(state.retrieved_patterns),
                "historical_insights_loaded": bool(state.historical_insights),
                "context_loaded": len(state.memory_context)
            })

        except Exception as e:
            logger.error(f"Pre-analysis hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_analysis_hook",
                "error": str(e),
                "error_type": type(e).__name__
            })

        return state

    @traceable(name="analyze_account_dormancy")
    async def analyze_dormancy(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Comprehensive dormancy analysis using enhanced CBUAE compliance checks"""
        try:
            start_time = datetime.now()
            state.analysis_status = AnalysisStatus.PROCESSING

            # Validate input data
            if not state.processed_data:
                raise ValueError("No processed data available for dormancy analysis")

            # Prepare DataFrame
            if 'accounts' in state.processed_data:
                df = pd.DataFrame(state.processed_data['accounts'])
            else:
                df = pd.DataFrame(state.processed_data)

            if df.empty:
                raise ValueError("Empty dataset provided for analysis")

            state.total_accounts_analyzed = len(df)

            # Set report date
            report_date = state.report_date or datetime.now().strftime("%Y-%m-%d")

            # Execute comprehensive dormancy checks using original dormant.py functions
            dormancy_results = run_all_dormant_identification_checks(
                df=df,
                report_date_str=report_date,
                dormant_flags_history_df=None  # Can be enhanced with historical data
            )

            state.dormancy_results = dormancy_results

            # Enhanced pattern analysis if enabled
            if self.analysis_config["enable_pattern_analysis"]:
                pattern_results = await self.pattern_analyzer.analyze_patterns(
                    df, state.historical_insights
                )
                state.dormancy_results["pattern_analysis"] = pattern_results

            # Generate compliance flags
            state.compliance_flags = self._generate_compliance_flags(dormancy_results)

            # Calculate summary metrics
            state.dormancy_summary = self._calculate_dormancy_summary(dormancy_results)
            state.dormant_accounts_found = state.dormancy_summary["total_dormant_accounts"]

            # Risk indicator analysis
            if self.analysis_config["enable_risk_scoring"]:
                state.risk_indicators = await self._analyze_risk_indicators(df, dormancy_results)
                state.high_risk_accounts = state.risk_indicators.get("high_risk_count", 0)

            # Compliance breakdown
            state.compliance_breakdown = self._generate_compliance_breakdown(dormancy_results)

            # Call MCP tool for additional processing
            mcp_result = await self.mcp_client.call_tool("analyze_account_dormancy", {
                "data": state.processed_data,
                "analysis_results": dormancy_results,
                "regulatory_params": self.regulatory_params,
                "memory_context": state.memory_context,
                "retrieved_patterns": state.retrieved_patterns
            })

            if mcp_result.get("success"):
                # Merge MCP results if available
                mcp_insights = mcp_result.get("data", {})
                if mcp_insights:
                    state.dormancy_results["mcp_insights"] = mcp_insights
            else:
                logger.warning(f"MCP analysis warning: {mcp_result.get('error')}")

            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "accounts_per_second": state.total_accounts_analyzed / processing_time if processing_time > 0 else 0,
                "dormancy_detection_rate": state.dormant_accounts_found / state.total_accounts_analyzed if state.total_accounts_analyzed > 0 else 0,
                "compliance_checks_executed": len(dormancy_results),
                "pattern_analysis_enabled": self.analysis_config["enable_pattern_analysis"],
                "risk_scoring_enabled": self.analysis_config["enable_risk_scoring"]
            }

            # Determine final status
            if state.dormant_accounts_found > 0:
                state.analysis_status = AnalysisStatus.COMPLETED
            else:
                state.analysis_status = AnalysisStatus.COMPLETED  # No dormant accounts is also a valid result

            # Log successful analysis
            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "dormancy_analysis",
                "action": "analyze_dormancy",
                "status": state.analysis_status.value,
                "accounts_analyzed": state.total_accounts_analyzed,
                "dormant_accounts_found": state.dormant_accounts_found,
                "high_risk_accounts": state.high_risk_accounts,
                "processing_time": processing_time,
                "compliance_checks": len(state.compliance_flags)
            })

        except Exception as e:
            state.analysis_status = AnalysisStatus.FAILED
            error_msg = str(e)
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "dormancy_analysis",
                "error": error_msg,
                "error_type": type(e).__name__
            })
            logger.error(f"Dormancy analysis failed: {error_msg}")

        return state

    def _generate_compliance_flags(self, dormancy_results: Dict) -> List[str]:
        """Generate compliance flags based on analysis results"""
        flags = []

        # Check each compliance category
        compliance_checks = {
            "sdb_dormant": ComplianceArticle.ARTICLE_2_6,
            "investment_dormant": ComplianceArticle.ARTICLE_2_3,
            "fixed_deposit_dormant": ComplianceArticle.ARTICLE_2_2,
            "demand_deposit_dormant": ComplianceArticle.ARTICLE_2_1_1,
            "unclaimed_instruments": ComplianceArticle.ARTICLE_2_4,
            "art3_process_needed": ComplianceArticle.ARTICLE_3,
            "eligible_for_cb_transfer": ComplianceArticle.ARTICLE_8
        }

        for check_type, article in compliance_checks.items():
            if check_type in dormancy_results:
                count = dormancy_results[check_type].get("count", 0)
                if count > 0:
                    flags.append(f"{article.value}_{count}_accounts")

        return flags

    def _calculate_dormancy_summary(self, dormancy_results: Dict) -> Dict:
        """Calculate comprehensive dormancy summary"""
        summary = {
            "total_dormant_accounts": 0,
            "by_type": {},
            "high_value_dormant": 0,
            "cb_transfer_eligible": 0,
            "article_3_required": 0,
            "proactive_contact_needed": 0
        }

        # Count dormant accounts by type
        dormancy_types = [
            "sdb_dormant", "investment_dormant", "fixed_deposit_dormant",
            "demand_deposit_dormant", "unclaimed_instruments"
        ]

        for dormancy_type in dormancy_types:
            if dormancy_type in dormancy_results:
                count = dormancy_results[dormancy_type].get("count", 0)
                summary["by_type"][dormancy_type] = count
                summary["total_dormant_accounts"] += count

        # Special categories
        if "high_value_dormant" in dormancy_results:
            summary["high_value_dormant"] = dormancy_results["high_value_dormant"].get("count", 0)

        if "eligible_for_cb_transfer" in dormancy_results:
            summary["cb_transfer_eligible"] = dormancy_results["eligible_for_cb_transfer"].get("count", 0)

        if "art3_process_needed" in dormancy_results:
            summary["article_3_required"] = dormancy_results["art3_process_needed"].get("count", 0)

        if "proactive_contact_needed" in dormancy_results:
            summary["proactive_contact_needed"] = dormancy_results["proactive_contact_needed"].get("count", 0)

        return summary

    async def _analyze_risk_indicators(self, df: pd.DataFrame, dormancy_results: Dict) -> Dict:
        """Analyze risk indicators for dormant accounts"""
        risk_analysis = {
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "risk_factors": [],
            "recommendations": []
        }

        try:
            # High balance risk
            high_value_count = dormancy_results.get("high_value_dormant", {}).get("count", 0)
            if high_value_count > 0:
                risk_analysis["high_risk_count"] += high_value_count
                risk_analysis["risk_factors"].append(f"High-value dormant accounts: {high_value_count}")
                risk_analysis["recommendations"].append("Prioritize high-value account reactivation")

            # Compliance risk
            cb_eligible_count = dormancy_results.get("eligible_for_cb_transfer", {}).get("count", 0)
            if cb_eligible_count > 0:
                risk_analysis["high_risk_count"] += cb_eligible_count
                risk_analysis["risk_factors"].append(f"CB transfer eligible: {cb_eligible_count}")
                risk_analysis["recommendations"].append("Review accounts for Central Bank transfer")

            # Process compliance risk
            art3_needed_count = dormancy_results.get("art3_process_needed", {}).get("count", 0)
            if art3_needed_count > 0:
                risk_analysis["medium_risk_count"] += art3_needed_count
                risk_analysis["risk_factors"].append(f"Article 3 process required: {art3_needed_count}")
                risk_analysis["recommendations"].append("Complete Article 3 compliance procedures")

            # Proactive contact risk
            contact_needed_count = dormancy_results.get("proactive_contact_needed", {}).get("count", 0)
            if contact_needed_count > 0:
                risk_analysis["low_risk_count"] += contact_needed_count
                risk_analysis["risk_factors"].append(f"Proactive contact needed: {contact_needed_count}")
                risk_analysis["recommendations"].append("Implement proactive customer outreach")

        except Exception as e:
            logger.error(f"Risk analysis failed: {str(e)}")
            risk_analysis["error"] = str(e)

        return risk_analysis

    def _generate_compliance_breakdown(self, dormancy_results: Dict) -> Dict:
        """Generate detailed compliance breakdown"""
        breakdown = {
            "cbuae_articles": {},
            "total_compliance_items": 0,
            "critical_items": 0,
            "attention_required": []
        }

        # Map results to CBUAE articles
        article_mapping = {
            "article_2_1_1": dormancy_results.get("demand_deposit_dormant", {}),
            "article_2_2": dormancy_results.get("fixed_deposit_dormant", {}),
            "article_2_3": dormancy_results.get("investment_dormant", {}),
            "article_2_4": dormancy_results.get("unclaimed_instruments", {}),
            "article_2_6": dormancy_results.get("sdb_dormant", {}),
            "article_3": dormancy_results.get("art3_process_needed", {}),
            "article_8": dormancy_results.get("eligible_for_cb_transfer", {})
        }

        for article, data in article_mapping.items():
            count = data.get("count", 0)
            description = data.get("desc", "")

            breakdown["cbuae_articles"][article] = {
                "count": count,
                "description": description,
                "details": data.get("details", {}),
                "status": "critical" if count > 0 else "compliant"
            }

            breakdown["total_compliance_items"] += count

            if count > 0:
                breakdown["critical_items"] += 1
                breakdown["attention_required"].append({
                    "article": article,
                    "count": count,
                    "description": description
                })

        return breakdown

    @traceable(name="dormancy_analysis_post_hook")
    async def post_analysis_hook(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Enhanced post-analysis memory hook with comprehensive data storage"""
        try:
            # Store analysis results in session memory
            session_data = {
                "session_id": state.session_id,
                "analysis_id": state.analysis_id,
                "user_id": state.user_id,
                "analysis_results": {
                    "status": state.analysis_status.value,
                    "total_accounts_analyzed": state.total_accounts_analyzed,
                    "dormant_accounts_found": state.dormant_accounts_found,
                    "high_risk_accounts": state.high_risk_accounts,
                    "compliance_flags": state.compliance_flags,
                    "processing_time": state.performance_metrics.get("processing_time_seconds", 0)
                },
                "dormancy_summary": state.dormancy_summary,
                "compliance_breakdown": state.compliance_breakdown,
                "risk_indicators": state.risk_indicators
            }

            await self.memory_agent.store_memory(
                bucket="session",
                data=session_data,
                encrypt_sensitive=True
            )

            # Store dormancy patterns in knowledge memory
            if state.analysis_status == AnalysisStatus.COMPLETED:
                knowledge_data = {
                    "type": "dormancy_patterns",
                    "user_id": state.user_id,
                    "analysis_patterns": {
                        "dormancy_rate": state.dormant_accounts_found / state.total_accounts_analyzed if state.total_accounts_analyzed > 0 else 0,
                        "common_dormancy_types": {
                            dormancy_type: count for dormancy_type, count in state.dormancy_summary["by_type"].items()
                            if count > 0
                        },
                        "risk_distribution": {
                            "high_risk": state.risk_indicators.get("high_risk_count", 0),
                            "medium_risk": state.risk_indicators.get("medium_risk_count", 0),
                            "low_risk": state.risk_indicators.get("low_risk_count", 0)
                        },
                        "compliance_insights": state.compliance_breakdown
                    },
                    "performance_benchmark": state.performance_metrics,
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket="knowledge",
                    data=knowledge_data
                )

            # Store historical insights for future analysis
            if state.dormancy_results and "pattern_analysis" in state.dormancy_results:
                historical_data = {
                    "type": "historical_dormancy_insights",
                    "user_id": state.user_id,
                    "insights": {
                        "pattern_insights": state.dormancy_results["pattern_analysis"].get("insights", []),
                        "recommendations": state.dormancy_results["pattern_analysis"].get("recommendations", []),
                        "reactivation_probability": state.dormancy_results["pattern_analysis"]["pattern_analysis"].get(
                            "reactivation_probability", {}),
                        "seasonal_patterns": state.dormancy_results["pattern_analysis"]["pattern_analysis"].get(
                            "seasonal_patterns", {})
                    },
                    "analysis_date": datetime.now().isoformat(),
                    "accounts_analyzed": state.total_accounts_analyzed
                }

                await self.memory_agent.store_memory(
                    bucket="knowledge",
                    data=historical_data
                )

            # Store compliance benchmarks
            compliance_benchmark = {
                "type": "compliance_benchmarks",
                "user_id": state.user_id,
                "benchmarks": {
                    "total_compliance_items": state.compliance_breakdown.get("total_compliance_items", 0),
                    "critical_items": state.compliance_breakdown.get("critical_items", 0),
                    "compliance_rate": 1.0 - (state.compliance_breakdown.get("critical_items", 0) / 7),
                    # 7 CBUAE articles
                    "article_breakdown": state.compliance_breakdown.get("cbuae_articles", {})
                },
                "timestamp": datetime.now().isoformat()
            }

            await self.memory_agent.store_memory(
                bucket="knowledge",
                data=compliance_benchmark
            )

            # Log post-analysis hook completion
            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_analysis_hook",
                "action": "memory_storage",
                "status": "completed",
                "session_data_stored": True,
                "knowledge_patterns_stored": state.analysis_status == AnalysisStatus.COMPLETED,
                "historical_insights_stored": "pattern_analysis" in state.dormancy_results,
                "compliance_benchmarks_stored": True
            })

        except Exception as e:
            logger.error(f"Post-analysis hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_analysis_hook",
                "error": str(e),
                "error_type": type(e).__name__
            })

        return state

    @traceable(name="generate_dormancy_report")
    async def generate_analysis_report(self, state: DormancyAnalysisState) -> Dict:
        """Generate comprehensive dormancy analysis report"""
        try:
            # Executive summary
            executive_summary = {
                "total_accounts_analyzed": state.total_accounts_analyzed,
                "dormant_accounts_identified": state.dormant_accounts_found,
                "dormancy_rate_percentage": round(
                    (
                                state.dormant_accounts_found / state.total_accounts_analyzed * 100) if state.total_accounts_analyzed > 0 else 0,
                    2
                ),
                "high_risk_accounts": state.high_risk_accounts,
                "compliance_status": "ATTENTION_REQUIRED" if state.compliance_breakdown.get("critical_items",
                                                                                            0) > 0 else "COMPLIANT",
                "analysis_date": state.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Detailed findings
            detailed_findings = {
                "dormancy_by_type": state.dormancy_summary["by_type"],
                "compliance_breakdown": state.compliance_breakdown,
                "risk_analysis": state.risk_indicators,
                "performance_metrics": state.performance_metrics
            }

            # Action items
            action_items = []

            # Generate action items based on findings
            if state.dormancy_summary["article_3_required"] > 0:
                action_items.append({
                    "priority": "HIGH",
                    "action": "Complete Article 3 compliance procedures",
                    "accounts_affected": state.dormancy_summary["article_3_required"],
                    "timeline": "Immediate"
                })

            if state.dormancy_summary["high_value_dormant"] > 0:
                action_items.append({
                    "priority": "HIGH",
                    "action": "Prioritize high-value account reactivation",
                    "accounts_affected": state.dormancy_summary["high_value_dormant"],
                    "timeline": "Within 30 days"
                })

            if state.dormancy_summary["cb_transfer_eligible"] > 0:
                action_items.append({
                    "priority": "MEDIUM",
                    "action": "Review accounts for Central Bank transfer",
                    "accounts_affected": state.dormancy_summary["cb_transfer_eligible"],
                    "timeline": "Within 60 days"
                })

            if state.dormancy_summary["proactive_contact_needed"] > 0:
                action_items.append({
                    "priority": "LOW",
                    "action": "Implement proactive customer outreach",
                    "accounts_affected": state.dormancy_summary["proactive_contact_needed"],
                    "timeline": "Within 90 days"
                })

            # Pattern insights (if available)
            pattern_insights = {}
            if state.dormancy_results and "pattern_analysis" in state.dormancy_results:
                pattern_data = state.dormancy_results["pattern_analysis"]
                pattern_insights = {
                    "insights": pattern_data.get("insights", []),
                    "recommendations": pattern_data.get("recommendations", []),
                    "seasonal_patterns": pattern_data.get("pattern_analysis", {}).get("seasonal_patterns", {}),
                    "reactivation_probability": pattern_data.get("pattern_analysis", {}).get("reactivation_probability",
                                                                                             {})
                }

            # Compile final report
            report = {
                "report_id": secrets.token_hex(16),
                "analysis_id": state.analysis_id,
                "generated_at": datetime.now().isoformat(),
                "executive_summary": executive_summary,
                "detailed_findings": detailed_findings,
                "action_items": action_items,
                "pattern_insights": pattern_insights,
                "compliance_flags": state.compliance_flags,
                "processing_log": state.analysis_log,
                "errors": state.error_log if state.error_log else None
            }

            return {"success": True, "report": report}

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    @traceable(name="execute_dormancy_analysis_workflow")
    async def execute_workflow(self, user_id: str, processed_data: Dict,
                               analysis_options: Dict = None) -> Dict:
        """Execute complete dormancy analysis workflow"""
        try:
            # Initialize analysis state
            analysis_id = secrets.token_hex(16)
            session_id = secrets.token_hex(16)

            state = DormancyAnalysisState(
                session_id=session_id,
                user_id=user_id,
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                processed_data=processed_data,
                report_date=analysis_options.get("report_date") if analysis_options else None,
                analysis_parameters=analysis_options or {}
            )

            # Execute workflow stages
            state = await self.pre_analysis_hook(state)
            state = await self.analyze_dormancy(state)
            state = await self.post_analysis_hook(state)

            # Generate analysis report
            report_result = await self.generate_analysis_report(state)

            # Prepare response
            response = {
                "success": state.analysis_status != AnalysisStatus.FAILED,
                "analysis_id": analysis_id,
                "session_id": session_id,
                "status": state.analysis_status.value,
                "total_accounts_analyzed": state.total_accounts_analyzed,
                "dormant_accounts_found": state.dormant_accounts_found,
                "high_risk_accounts": state.high_risk_accounts,
                "compliance_flags": state.compliance_flags,
                "dormancy_summary": state.dormancy_summary,
                "compliance_breakdown": state.compliance_breakdown,
                "risk_indicators": state.risk_indicators,
                "performance_metrics": state.performance_metrics,
                "processing_time": state.performance_metrics.get("processing_time_seconds", 0),
                "analysis_log": state.analysis_log[-10:],  # Last 10 entries
                "report": report_result.get("report") if report_result.get("success") else None
            }

            if state.analysis_status == AnalysisStatus.FAILED:
                response["errors"] = state.error_log

            return response

        except Exception as e:
            logger.error(f"Dormancy analysis workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }


# LangGraph Workflow Integration
class DormancyWorkflowEngine:
    """LangGraph-based workflow engine for dormancy analysis"""

    def __init__(self, dormancy_agent: DormancyAnalysisAgent):
        self.dormancy_agent = dormancy_agent
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for dormancy analysis"""
        workflow = StateGraph(DormancyAnalysisState)

        # Add nodes
        workflow.add_node("pre_analysis", self._pre_analysis_node)
        workflow.add_node("dormancy_analysis", self._dormancy_analysis_node)
        workflow.add_node("post_analysis", self._post_analysis_node)
        workflow.add_node("report_generation", self._report_generation_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Add edges
        workflow.add_edge("pre_analysis", "dormancy_analysis")
        workflow.add_edge("dormancy_analysis", "post_analysis")
        workflow.add_edge("post_analysis", "report_generation")
        workflow.add_edge("report_generation", END)

        # Conditional edges for error handling
        workflow.add_conditional_edges(
            "dormancy_analysis",
            self._check_analysis_status,
            {
                "continue": "post_analysis",
                "error": "error_handler"
            }
        )

        workflow.add_edge("error_handler", END)

        # Set entry point
        workflow.set_entry_point("pre_analysis")

        return workflow.compile(checkpointer=self.checkpointer)

    async def _pre_analysis_node(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Pre-analysis workflow node"""
        return await self.dormancy_agent.pre_analysis_hook(state)

    async def _dormancy_analysis_node(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Main dormancy analysis workflow node"""
        return await self.dormancy_agent.analyze_dormancy(state)

    async def _post_analysis_node(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Post-analysis workflow node"""
        return await self.dormancy_agent.post_analysis_hook(state)

    async def _report_generation_node(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Report generation workflow node"""
        report_result = await self.dormancy_agent.generate_analysis_report(state)
        if report_result.get("success"):
            state.analysis_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "report_generation",
                "action": "generate_report",
                "status": "completed",
                "report_id": report_result["report"]["report_id"]
            })
        return state

    async def _error_handler_node(self, state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Error handling workflow node"""
        logger.error(f"Dormancy analysis workflow error for session {state.session_id}")
        state.analysis_status = AnalysisStatus.FAILED
        return state

    def _check_analysis_status(self, state: DormancyAnalysisState) -> str:
        """Check analysis status for conditional routing"""
        return "continue" if state.analysis_status != AnalysisStatus.FAILED else "error"

    @traceable(name="execute_dormancy_workflow")
    async def execute_workflow(self, initial_state: DormancyAnalysisState) -> DormancyAnalysisState:
        """Execute the dormancy analysis workflow"""
        return await self.workflow.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": initial_state.session_id}}
        )


# API Interface for Dormancy Analysis
class DormancyAnalysisAPI:
    """RESTful API interface for dormancy analysis"""

    def __init__(self, dormancy_agent: DormancyAnalysisAgent):
        self.dormancy_agent = dormancy_agent
        self.workflow_engine = DormancyWorkflowEngine(dormancy_agent)

    async def analyze_dormancy_endpoint(self, request_data: Dict) -> Dict:
        """Dormancy analysis endpoint"""
        try:
            user_id = request_data.get('user_id')
            processed_data = request_data.get('processed_data')
            analysis_options = request_data.get('analysis_options', {})

            if not user_id or not processed_data:
                return {"success": False, "error": "User ID and processed data required"}

            return await self.dormancy_agent.execute_workflow(
                user_id, processed_data, analysis_options
            )

        except Exception as e:
            logger.error(f"Dormancy analysis API error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_analysis_report_endpoint(self, request_data: Dict) -> Dict:
        """Get analysis report endpoint"""
        try:
            analysis_id = request_data.get('analysis_id')
            user_id = request_data.get('user_id')

            if not analysis_id or not user_id:
                return {"success": False, "error": "Analysis ID and User ID required"}

            # Retrieve analysis results from memory
            session_data = await self.dormancy_agent.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "analysis_id": analysis_id,
                    "user_id": user_id
                }
            )

            if session_data.get("success"):
                return {"success": True, "data": session_data.get("data")}
            else:
                return {"success": False, "error": "Analysis not found"}

        except Exception as e:
            logger.error(f"Get analysis report API error: {str(e)}")
            return {"success": False, "error": str(e)}


# Example usage and testing
async def main():
    """Example usage of the dormancy analysis agent"""
    print("Enhanced Banking Dormancy Analysis Agent")
    print("=" * 50)
    print("Features:")
    print("- CBUAE compliance analysis (Articles 2.1-2.6, 3, 8)")
    print("- AI-powered pattern recognition")
    print("- Risk indicator analysis")
    print("- Hybrid memory integration")
    print("- LangGraph workflow orchestration")
    print("- LangSmith observability")
    print("- MCP tool integration")
    print("- Comprehensive reporting")
    print("- Predictive insights")
    print("- Historical pattern learning")


if __name__ == "__main__":
    asyncio.run(main())