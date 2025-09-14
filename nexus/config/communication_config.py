"""
Communication Configuration

Configurações para integrações de comunicação (Slack, Discord, etc.)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class SlackConfig:
    """Configuração para integração com Slack."""
    
    enabled: bool = False
    bot_token: Optional[str] = None
    app_token: Optional[str] = None
    signing_secret: Optional[str] = None
    channels: List[str] = field(default_factory=list)
    default_channel: Optional[str] = None
    
    # Configurações de comandos
    enable_slash_commands: bool = True
    enable_mentions: bool = True
    enable_events: bool = True
    
    # Configurações de permissões
    admin_users: List[str] = field(default_factory=list)
    allowed_channels: List[str] = field(default_factory=list)
    
    # Configurações de notificações
    notify_on_errors: bool = True
    notify_on_completion: bool = True
    notify_on_status_change: bool = True


@dataclass
class DiscordConfig:
    """Configuração para integração com Discord."""
    
    enabled: bool = False
    bot_token: Optional[str] = None
    guild_id: Optional[str] = None
    channels: List[int] = field(default_factory=list)
    default_channel: Optional[int] = None
    
    # Configurações de comandos
    enable_slash_commands: bool = True
    enable_prefix_commands: bool = True
    command_prefix: str = "!nexus "
    
    # Configurações de permissões
    admin_role_id: Optional[str] = None
    allowed_roles: List[str] = field(default_factory=list)
    allowed_users: List[int] = field(default_factory=list)
    
    # Configurações de notificações
    notify_on_errors: bool = True
    notify_on_completion: bool = True
    notify_on_status_change: bool = True
    
    # Configurações de embed
    default_color: int = 0x0099ff
    success_color: int = 0x00ff00
    warning_color: int = 0xff9900
    error_color: int = 0xff0000


@dataclass
class CommunicationConfig:
    """Configuração geral de comunicação."""
    
    slack: SlackConfig = field(default_factory=SlackConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    
    # Configurações globais
    enable_cross_platform: bool = True
    message_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Configurações de logging
    log_messages: bool = True
    log_commands: bool = True
    log_errors: bool = True
    
    # Configurações de rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000


def load_communication_config(config_data: Dict[str, Any]) -> CommunicationConfig:
    """
    Carrega configuração de comunicação a partir de dicionário.
    
    Args:
        config_data: Dados de configuração
        
    Returns:
        Configuração de comunicação
    """
    
    # Configuração do Slack
    slack_data = config_data.get('slack', {})
    slack_config = SlackConfig(
        enabled=slack_data.get('enabled', False),
        bot_token=slack_data.get('bot_token'),
        app_token=slack_data.get('app_token'),
        signing_secret=slack_data.get('signing_secret'),
        channels=slack_data.get('channels', []),
        default_channel=slack_data.get('default_channel'),
        enable_slash_commands=slack_data.get('enable_slash_commands', True),
        enable_mentions=slack_data.get('enable_mentions', True),
        enable_events=slack_data.get('enable_events', True),
        admin_users=slack_data.get('admin_users', []),
        allowed_channels=slack_data.get('allowed_channels', []),
        notify_on_errors=slack_data.get('notify_on_errors', True),
        notify_on_completion=slack_data.get('notify_on_completion', True),
        notify_on_status_change=slack_data.get('notify_on_status_change', True)
    )
    
    # Configuração do Discord
    discord_data = config_data.get('discord', {})
    discord_config = DiscordConfig(
        enabled=discord_data.get('enabled', False),
        bot_token=discord_data.get('bot_token'),
        guild_id=discord_data.get('guild_id'),
        channels=discord_data.get('channels', []),
        default_channel=discord_data.get('default_channel'),
        enable_slash_commands=discord_data.get('enable_slash_commands', True),
        enable_prefix_commands=discord_data.get('enable_prefix_commands', True),
        command_prefix=discord_data.get('command_prefix', '!nexus '),
        admin_role_id=discord_data.get('admin_role_id'),
        allowed_roles=discord_data.get('allowed_roles', []),
        allowed_users=discord_data.get('allowed_users', []),
        notify_on_errors=discord_data.get('notify_on_errors', True),
        notify_on_completion=discord_data.get('notify_on_completion', True),
        notify_on_status_change=discord_data.get('notify_on_status_change', True),
        default_color=discord_data.get('default_color', 0x0099ff),
        success_color=discord_data.get('success_color', 0x00ff00),
        warning_color=discord_data.get('warning_color', 0xff9900),
        error_color=discord_data.get('error_color', 0xff0000)
    )
    
    # Configuração geral
    return CommunicationConfig(
        slack=slack_config,
        discord=discord_config,
        enable_cross_platform=config_data.get('enable_cross_platform', True),
        message_timeout=config_data.get('message_timeout', 30.0),
        retry_attempts=config_data.get('retry_attempts', 3),
        retry_delay=config_data.get('retry_delay', 1.0),
        log_messages=config_data.get('log_messages', True),
        log_commands=config_data.get('log_commands', True),
        log_errors=config_data.get('log_errors', True),
        rate_limit_enabled=config_data.get('rate_limit_enabled', True),
        rate_limit_per_minute=config_data.get('rate_limit_per_minute', 60),
        rate_limit_per_hour=config_data.get('rate_limit_per_hour', 1000)
    )


def get_default_communication_config() -> Dict[str, Any]:
    """
    Retorna configuração padrão de comunicação.
    
    Returns:
        Dicionário com configuração padrão
    """
    
    return {
        'slack': {
            'enabled': False,
            'bot_token': None,
            'app_token': None,
            'signing_secret': None,
            'channels': [],
            'default_channel': None,
            'enable_slash_commands': True,
            'enable_mentions': True,
            'enable_events': True,
            'admin_users': [],
            'allowed_channels': [],
            'notify_on_errors': True,
            'notify_on_completion': True,
            'notify_on_status_change': True
        },
        'discord': {
            'enabled': False,
            'bot_token': None,
            'guild_id': None,
            'channels': [],
            'default_channel': None,
            'enable_slash_commands': True,
            'enable_prefix_commands': True,
            'command_prefix': '!nexus ',
            'admin_role_id': None,
            'allowed_roles': [],
            'allowed_users': [],
            'notify_on_errors': True,
            'notify_on_completion': True,
            'notify_on_status_change': True,
            'default_color': 0x0099ff,
            'success_color': 0x00ff00,
            'warning_color': 0xff9900,
            'error_color': 0xff0000
        },
        'enable_cross_platform': True,
        'message_timeout': 30.0,
        'retry_attempts': 3,
        'retry_delay': 1.0,
        'log_messages': True,
        'log_commands': True,
        'log_errors': True,
        'rate_limit_enabled': True,
        'rate_limit_per_minute': 60,
        'rate_limit_per_hour': 1000
    }


def validate_communication_config(config: CommunicationConfig) -> List[str]:
    """
    Valida configuração de comunicação.
    
    Args:
        config: Configuração a ser validada
        
    Returns:
        Lista de erros encontrados
    """
    
    errors = []
    
    # Validar Slack
    if config.slack.enabled:
        if not config.slack.bot_token:
            errors.append("Slack bot_token é obrigatório quando Slack está habilitado")
        
        if config.slack.enable_events and not config.slack.app_token:
            errors.append("Slack app_token é obrigatório quando eventos estão habilitados")
        
        if not config.slack.channels and not config.slack.default_channel:
            errors.append("Pelo menos um canal deve ser configurado para Slack")
    
    # Validar Discord
    if config.discord.enabled:
        if not config.discord.bot_token:
            errors.append("Discord bot_token é obrigatório quando Discord está habilitado")
        
        if not config.discord.channels and not config.discord.default_channel:
            errors.append("Pelo menos um canal deve ser configurado para Discord")
    
    # Validar configurações globais
    if config.message_timeout <= 0:
        errors.append("message_timeout deve ser maior que 0")
    
    if config.retry_attempts < 0:
        errors.append("retry_attempts não pode ser negativo")
    
    if config.retry_delay < 0:
        errors.append("retry_delay não pode ser negativo")
    
    if config.rate_limit_per_minute <= 0:
        errors.append("rate_limit_per_minute deve ser maior que 0")
    
    if config.rate_limit_per_hour <= 0:
        errors.append("rate_limit_per_hour deve ser maior que 0")
    
    return errors
