"""
NEXUS Configuration Module

Módulo de configuração para o sistema NEXUS.
"""

from .communication_config import (
    CommunicationConfig,
    SlackConfig,
    DiscordConfig,
    load_communication_config,
    get_default_communication_config,
    validate_communication_config
)

__all__ = [
    'CommunicationConfig',
    'SlackConfig', 
    'DiscordConfig',
    'load_communication_config',
    'get_default_communication_config',
    'validate_communication_config'
]
