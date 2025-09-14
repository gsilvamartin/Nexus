"""
NEXUS Communication Layer

Camada de comunicação unificada para integração com plataformas externas
como Slack, Discord, e outras ferramentas de colaboração.
"""

from .slack_integration import SlackIntegration
from .discord_integration import DiscordIntegration
from .communication_manager import CommunicationManager

__all__ = [
    'SlackIntegration',
    'DiscordIntegration', 
    'CommunicationManager'
]
