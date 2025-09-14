"""
Communication Manager

Gerenciador unificado para todas as integrações de comunicação,
incluindo Slack, Discord e futuras plataformas.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from enum import Enum

from .slack_integration import SlackIntegration
from .discord_integration import DiscordIntegration

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Plataformas de comunicação suportadas."""
    SLACK = "slack"
    DISCORD = "discord"


class MessageType(Enum):
    """Tipos de mensagem."""
    TEXT = "text"
    EMBED = "embed"
    FILE = "file"
    STATUS = "status"
    NOTIFICATION = "notification"


class CommunicationManager:
    """
    Gerenciador unificado de comunicação para o sistema NEXUS.
    
    Coordena todas as integrações de comunicação, fornecendo uma interface
    unificada para envio de mensagens, notificações e processamento de comandos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa gerenciador de comunicação.
        
        Args:
            config: Configuração contendo tokens e configurações de todas as plataformas
        """
        self.config = config
        
        # Integrações
        self.slack: Optional[SlackIntegration] = None
        self.discord: Optional[DiscordIntegration] = None
        
        # Estado
        self.is_initialized = False
        self.active_platforms: List[Platform] = []
        
        # Callbacks globais
        self.message_callbacks: List[Callable] = []
        self.command_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # Mapeamento de canais/usuários
        self.channel_mapping: Dict[str, Dict[str, Any]] = {}
        self.user_mapping: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Communication Manager initialized")
    
    async def initialize(self) -> None:
        """Inicializa todas as integrações de comunicação."""
        
        logger.info("Initializing Communication Manager")
        
        try:
            # Inicializar Slack se configurado
            if self.config.get('slack', {}).get('enabled', False):
                await self._initialize_slack()
            
            # Inicializar Discord se configurado
            if self.config.get('discord', {}).get('enabled', False):
                await self._initialize_discord()
            
            # Configurar callbacks unificados
            self._setup_unified_callbacks()
            
            self.is_initialized = True
            logger.info(f"Communication Manager initialized with platforms: {[p.value for p in self.active_platforms]}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Communication Manager: {e}")
            raise
    
    async def _initialize_slack(self) -> None:
        """Inicializa integração com Slack."""
        
        slack_config = self.config.get('slack', {})
        
        self.slack = SlackIntegration(slack_config)
        await self.slack.initialize()
        
        self.active_platforms.append(Platform.SLACK)
        logger.info("Slack integration initialized")
    
    async def _initialize_discord(self) -> None:
        """Inicializa integração com Discord."""
        
        discord_config = self.config.get('discord', {})
        
        self.discord = DiscordIntegration(discord_config)
        
        # Iniciar bot Discord em task separada
        asyncio.create_task(self.discord.initialize())
        
        self.active_platforms.append(Platform.DISCORD)
        logger.info("Discord integration initialized")
    
    def _setup_unified_callbacks(self) -> None:
        """Configura callbacks unificados para todas as plataformas."""
        
        # Callback unificado para mensagens
        async def unified_message_callback(message_data: Dict[str, Any]):
            # Adicionar metadados de plataforma
            platform = self._detect_platform_from_message(message_data)
            message_data['platform'] = platform.value if platform else 'unknown'
            
            # Notificar callbacks globais
            for callback in self.message_callbacks:
                try:
                    await callback(message_data)
                except Exception as e:
                    logger.error(f"Error in unified message callback: {e}")
        
        # Callback unificado para status
        async def unified_status_callback(event_type: str, data: Dict[str, Any]):
            # Adicionar timestamp
            data['timestamp'] = datetime.utcnow().isoformat()
            data['event_type'] = event_type
            
            # Notificar callbacks globais
            for callback in self.status_callbacks:
                try:
                    await callback(event_type, data)
                except Exception as e:
                    logger.error(f"Error in unified status callback: {e}")
        
        # Configurar callbacks nas integrações
        if self.slack:
            self.slack.add_message_callback(unified_message_callback)
            self.slack.add_status_callback(unified_status_callback)
        
        if self.discord:
            self.discord.add_message_callback(unified_message_callback)
            self.discord.add_status_callback(unified_status_callback)
    
    def _detect_platform_from_message(self, message_data: Dict[str, Any]) -> Optional[Platform]:
        """Detecta plataforma baseada nos dados da mensagem."""
        
        # Slack tem 'channel' e 'user' como strings
        if 'channel' in message_data and isinstance(message_data.get('channel'), str):
            return Platform.SLACK
        
        # Discord tem 'channel_id' e 'guild_id' como ints
        if 'channel_id' in message_data and isinstance(message_data.get('channel_id'), int):
            return Platform.DISCORD
        
        return None
    
    async def send_message(
        self, 
        platform: Platform,
        channel: Union[str, int], 
        content: str,
        message_type: MessageType = MessageType.TEXT,
        **kwargs
    ) -> bool:
        """
        Envia mensagem para plataforma específica.
        
        Args:
            platform: Plataforma de destino
            channel: ID do canal
            content: Conteúdo da mensagem
            message_type: Tipo da mensagem
            **kwargs: Argumentos adicionais específicos da plataforma
            
        Returns:
            True se enviado com sucesso
        """
        
        try:
            if platform == Platform.SLACK and self.slack:
                if message_type == MessageType.EMBED:
                    return await self.slack.send_rich_message(
                        channel=channel,
                        title=kwargs.get('title', ''),
                        content=content,
                        color=kwargs.get('color', 'good'),
                        fields=kwargs.get('fields')
                    )
                else:
                    return await self.slack.send_message(
                        channel=channel,
                        text=content,
                        thread_ts=kwargs.get('thread_ts'),
                        blocks=kwargs.get('blocks')
                    )
            
            elif platform == Platform.DISCORD and self.discord:
                if message_type == MessageType.EMBED:
                    return await self.discord.send_embed(
                        channel_id=channel,
                        title=kwargs.get('title', ''),
                        description=content,
                        color=kwargs.get('color', 0x0099ff),
                        fields=kwargs.get('fields'),
                        thumbnail_url=kwargs.get('thumbnail_url')
                    )
                else:
                    return await self.discord.send_message(
                        channel_id=channel,
                        content=content,
                        embed=kwargs.get('embed')
                    )
            
            else:
                logger.error(f"Platform {platform.value} not available or not configured")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send message to {platform.value}: {e}")
            return False
    
    async def send_broadcast(
        self, 
        content: str,
        message_type: MessageType = MessageType.TEXT,
        platforms: Optional[List[Platform]] = None,
        **kwargs
    ) -> Dict[Platform, bool]:
        """
        Envia mensagem para múltiplas plataformas.
        
        Args:
            content: Conteúdo da mensagem
            message_type: Tipo da mensagem
            platforms: Lista de plataformas (None = todas ativas)
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário com status de envio por plataforma
        """
        
        if platforms is None:
            platforms = self.active_platforms
        
        results = {}
        
        for platform in platforms:
            # Obter canais configurados para esta plataforma
            channels = self._get_configured_channels(platform)
            
            for channel in channels:
                success = await self.send_message(
                    platform=platform,
                    channel=channel,
                    content=content,
                    message_type=message_type,
                    **kwargs
                )
                
                if platform not in results:
                    results[platform] = []
                
                results[platform].append(success)
        
        return results
    
    async def send_file(
        self,
        platform: Platform,
        channel: Union[str, int],
        file_path: str,
        **kwargs
    ) -> bool:
        """
        Envia arquivo para plataforma específica.
        
        Args:
            platform: Plataforma de destino
            channel: ID do canal
            file_path: Caminho do arquivo
            **kwargs: Argumentos adicionais
            
        Returns:
            True se enviado com sucesso
        """
        
        try:
            if platform == Platform.SLACK and self.slack:
                return await self.slack.send_file(
                    channel=channel,
                    file_path=file_path,
                    title=kwargs.get('title'),
                    comment=kwargs.get('comment')
                )
            
            elif platform == Platform.DISCORD and self.discord:
                return await self.discord.send_file(
                    channel_id=channel,
                    file_path=file_path,
                    filename=kwargs.get('filename'),
                    content=kwargs.get('content')
                )
            
            else:
                logger.error(f"Platform {platform.value} not available for file sending")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send file to {platform.value}: {e}")
            return False
    
    async def send_notification(
        self,
        title: str,
        message: str,
        level: str = "info",
        platforms: Optional[List[Platform]] = None,
        **kwargs
    ) -> Dict[Platform, bool]:
        """
        Envia notificação para plataformas configuradas.
        
        Args:
            title: Título da notificação
            message: Mensagem da notificação
            level: Nível da notificação (info, warning, error, success)
            platforms: Lista de plataformas (None = todas ativas)
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário com status de envio por plataforma
        """
        
        if platforms is None:
            platforms = self.active_platforms
        
        # Configurar cores baseadas no nível
        color_mapping = {
            'info': {'slack': 'good', 'discord': 0x0099ff},
            'warning': {'slack': 'warning', 'discord': 0xff9900},
            'error': {'slack': 'danger', 'discord': 0xff0000},
            'success': {'slack': 'good', 'discord': 0x00ff00}
        }
        
        results = {}
        
        for platform in platforms:
            channels = self._get_configured_channels(platform)
            
            for channel in channels:
                if platform == Platform.SLACK:
                    success = await self.slack.send_rich_message(
                        channel=channel,
                        title=title,
                        content=message,
                        color=color_mapping[level]['slack'],
                        fields=kwargs.get('fields')
                    )
                elif platform == Platform.DISCORD:
                    success = await self.discord.send_embed(
                        channel_id=channel,
                        title=title,
                        description=message,
                        color=color_mapping[level]['discord'],
                        fields=kwargs.get('fields')
                    )
                else:
                    success = False
                
                if platform not in results:
                    results[platform] = []
                
                results[platform].append(success)
        
        return results
    
    async def send_system_status(
        self,
        platforms: Optional[List[Platform]] = None,
        **kwargs
    ) -> Dict[Platform, bool]:
        """
        Envia status do sistema para plataformas configuradas.
        
        Args:
            platforms: Lista de plataformas (None = todas ativas)
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário com status de envio por plataforma
        """
        
        if platforms is None:
            platforms = self.active_platforms
        
        results = {}
        
        for platform in platforms:
            channels = self._get_configured_channels(platform)
            
            for channel in channels:
                if platform == Platform.SLACK:
                    status_text = await self.slack._get_system_status_text()
                    success = await self.slack.send_message(
                        channel=channel,
                        text=status_text
                    )
                elif platform == Platform.DISCORD:
                    status_text = await self.discord._get_system_status_text()
                    embed = self.discord._create_status_embed(status_text)
                    success = await self.discord.send_message(
                        channel_id=channel,
                        content="",
                        embed=embed
                    )
                else:
                    success = False
                
                if platform not in results:
                    results[platform] = []
                
                results[platform].append(success)
        
        return results
    
    def _get_configured_channels(self, platform: Platform) -> List[Union[str, int]]:
        """Obtém canais configurados para uma plataforma."""
        
        if platform == Platform.SLACK:
            return self.config.get('slack', {}).get('channels', [])
        elif platform == Platform.DISCORD:
            return self.config.get('discord', {}).get('channels', [])
        
        return []
    
    def add_message_callback(self, callback: Callable):
        """Adiciona callback global para mensagens recebidas."""
        self.message_callbacks.append(callback)
    
    def add_command_callback(self, callback: Callable):
        """Adiciona callback global para comandos recebidos."""
        self.command_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable):
        """Adiciona callback global para mudanças de status."""
        self.status_callbacks.append(callback)
    
    def register_channel_mapping(
        self, 
        platform: Platform, 
        channel_id: Union[str, int], 
        mapping_data: Dict[str, Any]
    ):
        """Registra mapeamento de canal para metadados."""
        
        key = f"{platform.value}:{channel_id}"
        self.channel_mapping[key] = mapping_data
    
    def register_user_mapping(
        self, 
        platform: Platform, 
        user_id: Union[str, int], 
        mapping_data: Dict[str, Any]
    ):
        """Registra mapeamento de usuário para metadados."""
        
        key = f"{platform.value}:{user_id}"
        self.user_mapping[key] = mapping_data
    
    def get_channel_mapping(
        self, 
        platform: Platform, 
        channel_id: Union[str, int]
    ) -> Optional[Dict[str, Any]]:
        """Obtém mapeamento de canal."""
        
        key = f"{platform.value}:{channel_id}"
        return self.channel_mapping.get(key)
    
    def get_user_mapping(
        self, 
        platform: Platform, 
        user_id: Union[str, int]
    ) -> Optional[Dict[str, Any]]:
        """Obtém mapeamento de usuário."""
        
        key = f"{platform.value}:{user_id}"
        return self.user_mapping.get(key)
    
    async def process_nexus_command(
        self, 
        command: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processa comando NEXUS recebido de qualquer plataforma.
        
        Args:
            command: Comando recebido
            context: Contexto da mensagem
            
        Returns:
            Resultado do processamento
        """
        
        # Notificar callbacks de comando
        for callback in self.command_callbacks:
            try:
                result = await callback(command, context)
                if result:
                    return result
            except Exception as e:
                logger.error(f"Error in command callback: {e}")
        
        # Processamento padrão
        return await self._default_command_processing(command, context)
    
    async def _default_command_processing(
        self, 
        command: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Processamento padrão de comandos."""
        
        command_lower = command.lower()
        
        if 'status' in command_lower:
            return {
                'type': 'status',
                'message': 'Status do sistema solicitado',
                'action': 'send_status'
            }
        
        elif 'help' in command_lower or 'ajuda' in command_lower:
            return {
                'type': 'help',
                'message': 'Ajuda solicitada',
                'action': 'send_help'
            }
        
        elif 'desenvolver' in command_lower or 'develop' in command_lower:
            return {
                'type': 'development',
                'message': 'Solicitação de desenvolvimento',
                'action': 'start_development'
            }
        
        elif 'analisar' in command_lower or 'analyze' in command_lower:
            return {
                'type': 'analysis',
                'message': 'Solicitação de análise',
                'action': 'start_analysis'
            }
        
        else:
            return {
                'type': 'unknown',
                'message': 'Comando não reconhecido',
                'action': 'send_help'
            }
    
    async def get_platform_status(self) -> Dict[str, Any]:
        """Obtém status de todas as plataformas."""
        
        status = {
            'initialized': self.is_initialized,
            'active_platforms': [p.value for p in self.active_platforms],
            'platforms': {}
        }
        
        if self.slack:
            status['platforms']['slack'] = {
                'connected': self.slack.is_connected,
                'channels': len(self.slack.connected_channels)
            }
        
        if self.discord:
            status['platforms']['discord'] = {
                'connected': self.discord.is_connected,
                'guilds': len(self.discord.connected_guilds)
            }
        
        return status
    
    async def shutdown(self) -> None:
        """Desliga gerenciador de comunicação."""
        
        logger.info("Shutting down Communication Manager")
        
        if self.slack:
            await self.slack.shutdown()
        
        if self.discord:
            await self.discord.shutdown()
        
        self.is_initialized = False
        self.active_platforms.clear()
        
        logger.info("Communication Manager shutdown complete")
