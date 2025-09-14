"""
Specialized Cortices - NEXUS Camada 2

Implementa os córtices especializados do NEXUS para diferentes domínios de
inteligência: especificação, arquitetura, implementação e verificação.
"""

from nexus.cortices.specification import SpecificationCortex
from nexus.cortices.architecture import ArchitectureCortex
from nexus.cortices.implementation import ImplementationCortex
from nexus.cortices.verification import VerificationCortex

__all__ = [
    "SpecificationCortex",
    "ArchitectureCortex",
    "ImplementationCortex", 
    "VerificationCortex",
]
