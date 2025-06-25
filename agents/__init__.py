"""
agents/__init__.py
Initialize the agents package properly
"""

# Make agents a proper Python package
__version__ = "1.0.0"

# Export main agent classes for easy imports
try:
    from .Data_Process import UnifiedDataProcessingAgent, create_unified_data_processing_agent
    DATA_PROCESS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data Processing Agent not available: {e}")
    DATA_PROCESS_AVAILABLE = False

try:
    from .Dormant_agent import (
        DormancyWorkflowOrchestrator,
        run_comprehensive_dormancy_analysis_with_csv,
        run_individual_agent_analysis
    )
    DORMANCY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dormancy Agents not available: {e}")
    DORMANCY_AVAILABLE = False

try:
    from .compliance_verification_agent import (
        ComplianceWorkflowOrchestrator,
        run_comprehensive_compliance_analysis_with_csv,
        get_all_compliance_agents_info
    )
    COMPLIANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Compliance Agents not available: {e}")
    COMPLIANCE_AVAILABLE = False

try:
    from .memory_agent import HybridMemoryAgent, MemoryContext, MemoryBucket
    MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Memory Agent not available: {e}")
    MEMORY_AVAILABLE = False

# Export availability flags
__all__ = [
    'DATA_PROCESS_AVAILABLE',
    'DORMANCY_AVAILABLE',
    'COMPLIANCE_AVAILABLE',
    'MEMORY_AVAILABLE'
]