"""
Reasoning Systems - NEXUS Advanced Reasoning

Implementa os sistemas de raciocínio avançados do NEXUS, incluindo raciocínio causal,
análise contrafactual e inferência causal multi-dimensional.
"""

from nexus.reasoning.causal import CausalReasoningEngine
from nexus.reasoning.counterfactual import CounterfactualEngine
from nexus.reasoning.causal_graph import TemporalCausalGraph
from nexus.reasoning.structure_learning import CausalStructureLearner

__all__ = [
    "CausalReasoningEngine",
    "CounterfactualEngine", 
    "TemporalCausalGraph",
    "CausalStructureLearner",
]
