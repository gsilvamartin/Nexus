"""
Edge AI Distribution System

Sistema de distribuição de IA na borda para tornar a IA mais acessível
e escalável através de particionamento inteligente de modelos e otimização de latência.
"""

from .orchestrator import EdgeComputeOrchestrator
from .partitioning import ModelPartitioningEngine
from .optimization import LatencyOptimizer
from .distribution import DistributedIntelligence

__all__ = [
    'EdgeComputeOrchestrator',
    'ModelPartitioningEngine',
    'LatencyOptimizer',
    'DistributedIntelligence'
]
