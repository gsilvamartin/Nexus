"""
Cognitive Substrate - NEXUS Camada 1

Implementa o substrato cognitivo fundamental do NEXUS, incluindo função executiva,
memória de trabalho, memória episódica e córtex de decisão.
"""

from nexus.cognitive.executive import ExecutiveFunction
from nexus.cognitive.substrate import CognitiveSubstrate
from nexus.cognitive.working_memory import WorkingMemory
from nexus.cognitive.attention import AttentionController
from nexus.cognitive.metacognition import MetaCognitiveController

__all__ = [
    "ExecutiveFunction",
    "CognitiveSubstrate", 
    "WorkingMemory",
    "AttentionController",
    "MetaCognitiveController",
]
