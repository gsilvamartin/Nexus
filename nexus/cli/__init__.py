"""
CLI Interface - NEXUS Command Line Interface

Implementa interface de linha de comando para demonstrar e interagir com
o sistema NEXUS.
"""

from nexus.cli.main import main
from nexus.cli.commands import NexusCommands
from nexus.cli.demo import DemoRunner

__all__ = [
    "main",
    "NexusCommands",
    "DemoRunner",
]
