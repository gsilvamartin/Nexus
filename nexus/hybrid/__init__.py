"""
Quantum-Biological Hybrid Processing

Sistema de processamento híbrido que combina simulação quântica com modelos
biológicos para resolver problemas complexos de forma inovadora.
"""

from .quantum_simulator import QuantumCircuitSimulator
from .biological_models import (
    SpikingNeuralNetworks,
    EvolutionaryComputation,
    ArtificialImmuneSystem,
    SwarmOptimization
)
from .hybrid_processor import QuantumBiologicalProcessor
from .hybrid_solution import HybridSolution

__all__ = [
    'QuantumCircuitSimulator',
    'SpikingNeuralNetworks',
    'EvolutionaryComputation',
    'ArtificialImmuneSystem',
    'SwarmOptimization',
    'QuantumBiologicalProcessor',
    'HybridSolution'
]
