"""
Swarm Intelligence Architecture

Sistema de inteligência de enxame para coordenação descentralizada de agentes
com comportamentos emergentes e comunicação stigmértica.
"""

from .coordinator import DecentralizedSwarmCoordinator
from .emergence import EmergentBehaviorDetector
from .collective import CollectiveIntelligenceEngine
from .stigmergy import StigmergicCommunication
from .swarm_execution import SwarmExecution

__all__ = [
    'DecentralizedSwarmCoordinator',
    'EmergentBehaviorDetector', 
    'CollectiveIntelligenceEngine',
    'StigmergicCommunication',
    'SwarmExecution'
]
