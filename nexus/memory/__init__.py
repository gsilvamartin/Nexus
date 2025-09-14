"""
Memory Systems - NEXUS Advanced Memory

Implementa os sistemas de memória avançados do NEXUS, incluindo memória episódica
persistente, consolidação de memória e recuperação baseada em padrões.
"""

from nexus.memory.episodic import EpisodicMemorySystem
from nexus.memory.temporal_graph import TemporalGraphDatabase
from nexus.memory.consolidation import MemoryConsolidationEngine
from nexus.memory.pattern_detection import PatternDetectionEngine

__all__ = [
    "EpisodicMemorySystem",
    "TemporalGraphDatabase",
    "MemoryConsolidationEngine",
    "PatternDetectionEngine",
]
