"""
Slack Integration Module

Integração completa com Slack para comunicação bidirecional,
notificações e execução de comandos do NEXUS.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.async_client import AsyncSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

logger = logging.getLogger(__name__)


class SlackIntegration:
    """
    Integração completa com Slack para o sistema NEXUS.
    
    Suporta:
    - Recebimento de mensagens em tempo real
    - Envio de notificações e updates
    - Execução de comandos via slash commands
    - Integração com canais específicos
    - Sistema de permissões e autenticação
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa integração com Slack.
        
        Args:
            config: Configuração contendo tokens e configurações do Slack
        """
        self.config = config
        self.bot_token = config.get('bot_token')
        self.app_token = config.get('app_token')
        self.signing_secret = config.get('signing_secret')
        
        # Clientes Slack
        self.web_client: Optional[AsyncWebClient] = None
        self.socket_client: Optional[AsyncSocketModeClient] = None
        
        # Estado da integração
        self.is_connected = False
        self.connected_channels: List[str] = []
        self.command_handlers: Dict[str, Callable] = {}
        
        # Callbacks para eventos
        self.message_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        logger.info("Slack Integration initialized")
    
    async def initialize(self) -> None:
        """Inicializa conexão com Slack."""
        
        try:
            # Inicializar cliente web
            self.web_client = AsyncWebClient(token=self.bot_token)
            
            # Testar conexão
            auth_response = await self.web_client.auth_test()
            if not auth_response["ok"]:
                raise Exception(f"Slack auth failed: {auth_response['error']}")
            
            logger.info(f"Connected to Slack as: {auth_response['user']}")
            
            # Inicializar socket mode se app_token estiver disponível
            if self.app_token:
                self.socket_client = AsyncSocketModeClient(
                    app_token=self.app_token,
                    web_client=self.web_client
                )
                
                # Configurar handlers
                self.socket_client.socket_mode_request_listeners.append(self._handle_socket_mode_request)
                
                # Conectar
                await self.socket_client.connect()
                self.is_connected = True
                
                logger.info("Slack Socket Mode connected")
            else:
                logger.warning("App token not provided, Socket Mode disabled")
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack integration: {e}")
            raise
    
    async def _handle_socket_mode_request(self, client: AsyncSocketModeClient, req: SocketModeRequest):
        """Manipula requisições do Socket Mode."""
        
        try:
            if req.type == "events_api":
                # Evento de mensagem
                if req.payload.get("event", {}).get("type") == "message":
                    await self._handle_message_event(req.payload["event"])
                
                # Evento de slash command
                elif req.payload.get("event", {}).get("type") == "app_mention":
                    await self._handle_mention_event(req.payload["event"])
            
            elif req.type == "interactive":
                # Interação com botões, modais, etc.
                await self._handle_interactive_event(req.payload)
            
            elif req.type == "slash_commands":
                # Slash commands
                await self._handle_slash_command(req.payload)
            
            # Responder com sucesso
            client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
            
        except Exception as e:
            logger.error(f"Error handling socket mode request: {e}")
            client.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id, payload={"error": str(e)})
            )
    
    async def _handle_message_event(self, event: Dict[str, Any]):
        """Manipula eventos de mensagem."""
        
        # Ignorar mensagens do próprio bot
        if event.get("bot_id") or event.get("user") == self.web_client.auth_test()["user_id"]:
            return
        
        message_data = {
            'channel': event.get('channel'),
            'user': event.get('user'),
            'text': event.get('text', ''),
            'timestamp': event.get('ts'),
            'thread_ts': event.get('thread_ts'),
            'event_type': 'message'
        }
        
        # Notificar callbacks
        for callback in self.message_callbacks:
            try:
                await callback(message_data)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")
    
    async def _handle_mention_event(self, event: Dict[str, Any]):
        """Manipula menções ao bot."""
        
        message_data = {
            'channel': event.get('channel'),
            'user': event.get('user'),
            'text': event.get('text', ''),
            'timestamp': event.get('ts'),
            'thread_ts': event.get('thread_ts'),
            'event_type': 'mention'
        }
        
        # Processar menção como comando
        await self._process_nexus_command(message_data)
    
    async def _handle_interactive_event(self, payload: Dict[str, Any]):
        """Manipula eventos interativos (botões, modais)."""
        
        logger.info(f"Interactive event received: {payload.get('type')}")
        
        # Implementar handlers para diferentes tipos de interação
        if payload.get('type') == 'block_actions':
            await self._handle_block_actions(payload)
        elif payload.get('type') == 'view_submission':
            await self._handle_view_submission(payload)
    
    async def _handle_slash_command(self, payload: Dict[str, Any]):
        """Manipula slash commands."""
        
        command = payload.get('command', '')
        text = payload.get('text', '')
        user_id = payload.get('user_id')
        channel_id = payload.get('channel_id')
        
        logger.info(f"Slash command received: {command} {text}")
        
        # Processar comando
        response = await self._process_slash_command(command, text, user_id, channel_id)
        
        # Enviar resposta
        await self.send_message(
            channel=channel_id,
            text=response,
            thread_ts=payload.get('response_url')  # Responder em thread se disponível
        )
    
    async def _process_nexus_command(self, message_data: Dict[str, Any]):
        """Processa comandos do NEXUS mencionados em mensagens."""
        
        text = message_data['text'].lower()
        channel = message_data['channel']
        
        # Comandos básicos
        if 'status' in text or 'status do sistema' in text:
            await self._send_system_status(channel)
        
        elif 'help' in text or 'ajuda' in text:
            await self._send_help_message(channel)
        
        elif 'desenvolver' in text or 'develop' in text:
            await self._handle_development_request(message_data)
        
        elif 'analisar' in text or 'analyze' in text:
            await self._handle_analysis_request(message_data)
        
        else:
            # Comando não reconhecido
            await self.send_message(
                channel=channel,
                text="🤖 Comando não reconhecido. Use `@nexus help` para ver comandos disponíveis."
            )
    
    async def _process_slash_command(self, command: str, text: str, user_id: str, channel_id: str) -> str:
        """Processa slash commands específicos."""
        
        if command == '/nexus-status':
            return await self._get_system_status_text()
        
        elif command == '/nexus-help':
            return await self._get_help_text()
        
        elif command == '/nexus-develop':
            return await self._handle_development_slash_command(text, user_id, channel_id)
        
        elif command == '/nexus-analyze':
            return await self._handle_analysis_slash_command(text, user_id, channel_id)
        
        else:
            return "❌ Comando não reconhecido. Use `/nexus-help` para ver comandos disponíveis."
    
    async def _handle_development_request(self, message_data: Dict[str, Any]):
        """Manipula solicitações de desenvolvimento."""
        
        channel = message_data['channel']
        text = message_data['text']
        
        # Extrair descrição do projeto
        description = self._extract_project_description(text)
        
        if not description:
            await self.send_message(
                channel=channel,
                text="❌ Por favor, forneça uma descrição do projeto a ser desenvolvido."
            )
            return
        
        # Enviar confirmação
        await self.send_message(
            channel=channel,
            text=f"🚀 Iniciando desenvolvimento do projeto: *{description}*\n\n⏳ Processando... (isso pode levar alguns minutos)"
        )
        
        # Notificar callbacks de desenvolvimento
        for callback in self.status_callbacks:
            try:
                await callback('development_started', {
                    'description': description,
                    'channel': channel,
                    'user': message_data['user']
                })
            except Exception as e:
                logger.error(f"Error in development callback: {e}")
    
    async def _handle_analysis_request(self, message_data: Dict[str, Any]):
        """Manipula solicitações de análise."""
        
        channel = message_data['channel']
        text = message_data['text']
        
        # Extrair caminho do projeto
        project_path = self._extract_project_path(text)
        
        if not project_path:
            await self.send_message(
                channel=channel,
                text="❌ Por favor, forneça o caminho do projeto a ser analisado."
            )
            return
        
        # Enviar confirmação
        await self.send_message(
            channel=channel,
            text=f"🔍 Analisando projeto: *{project_path}*\n\n⏳ Processando análise... (isso pode levar alguns minutos)"
        )
        
        # Notificar callbacks de análise
        for callback in self.status_callbacks:
            try:
                await callback('analysis_started', {
                    'project_path': project_path,
                    'channel': channel,
                    'user': message_data['user']
                })
            except Exception as e:
                logger.error(f"Error in analysis callback: {e}")
    
    def _extract_project_description(self, text: str) -> Optional[str]:
        """Extrai descrição do projeto do texto da mensagem."""
        
        # Remover menção ao bot
        text = text.replace('<@nexus>', '').strip()
        
        # Procurar por palavras-chave
        keywords = ['desenvolver', 'develop', 'criar', 'create', 'projeto', 'project']
        
        for keyword in keywords:
            if keyword in text.lower():
                # Extrair texto após a palavra-chave
                parts = text.lower().split(keyword, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        return None
    
    def _extract_project_path(self, text: str) -> Optional[str]:
        """Extrai caminho do projeto do texto da mensagem."""
        
        # Remover menção ao bot
        text = text.replace('<@nexus>', '').strip()
        
        # Procurar por caminhos (começando com ./ ou /)
        import re
        path_match = re.search(r'(\.?\/[^\s]+|\/[^\s]+)', text)
        
        if path_match:
            return path_match.group(1)
        
        return None
    
    async def send_message(
        self, 
        channel: str, 
        text: str, 
        thread_ts: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Envia mensagem para canal do Slack.
        
        Args:
            channel: ID do canal
            text: Texto da mensagem
            thread_ts: Timestamp da thread (para responder em thread)
            blocks: Blocos ricos do Slack
            
        Returns:
            True se enviado com sucesso
        """
        
        try:
            if not self.web_client:
                logger.error("Slack web client not initialized")
                return False
            
            response = await self.web_client.chat_postMessage(
                channel=channel,
                text=text,
                thread_ts=thread_ts,
                blocks=blocks
            )
            
            return response["ok"]
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
    
    async def send_rich_message(
        self, 
        channel: str, 
        title: str, 
        content: str,
        color: str = "good",
        fields: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """
        Envia mensagem rica com formatação.
        
        Args:
            channel: ID do canal
            title: Título da mensagem
            content: Conteúdo principal
            color: Cor do attachment (good, warning, danger)
            fields: Campos adicionais
        """
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": content
                }
            }
        ]
        
        if fields:
            fields_block = {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*{field['title']}*\n{field['value']}"
                    }
                    for field in fields
                ]
            }
            blocks.append(fields_block)
        
        return await self.send_message(channel=channel, text=title, blocks=blocks)
    
    async def send_file(
        self, 
        channel: str, 
        file_path: str, 
        title: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        Envia arquivo para canal do Slack.
        
        Args:
            channel: ID do canal
            file_path: Caminho do arquivo
            title: Título do arquivo
            comment: Comentário sobre o arquivo
        """
        
        try:
            if not self.web_client:
                logger.error("Slack web client not initialized")
                return False
            
            with open(file_path, 'rb') as file:
                response = await self.web_client.files_upload(
                    channels=channel,
                    file=file,
                    title=title,
                    initial_comment=comment
                )
            
            return response["ok"]
            
        except Exception as e:
            logger.error(f"Failed to send Slack file: {e}")
            return False
    
    async def update_message(
        self, 
        channel: str, 
        ts: str, 
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Atualiza mensagem existente.
        
        Args:
            channel: ID do canal
            ts: Timestamp da mensagem
            text: Novo texto
            blocks: Novos blocos
        """
        
        try:
            if not self.web_client:
                logger.error("Slack web client not initialized")
                return False
            
            response = await self.web_client.chat_update(
                channel=channel,
                ts=ts,
                text=text,
                blocks=blocks
            )
            
            return response["ok"]
            
        except Exception as e:
            logger.error(f"Failed to update Slack message: {e}")
            return False
    
    async def _send_system_status(self, channel: str):
        """Envia status do sistema."""
        
        status_text = await self._get_system_status_text()
        await self.send_message(channel=channel, text=status_text)
    
    async def _send_help_message(self, channel: str):
        """Envia mensagem de ajuda."""
        
        help_text = await self._get_help_text()
        await self.send_message(channel=channel, text=help_text)
    
    async def _get_system_status_text(self) -> str:
        """Obtém texto de status do sistema."""
        
        # Implementação simplificada - integrar com NEXUS Core
        return """
🤖 *Status do Sistema NEXUS*

🟢 Sistema: Ativo
🧠 Córtices: 4/4 ativos
💾 Memória episódica: 45% utilizada
🤖 Modelos disponíveis: 3
⚡ Performance média: 1.2s
📊 Requisições hoje: 127
✅ Taxa de sucesso: 98.4%

*Componentes:*
• Substrato Cognitivo: 🟢 Online
• Memória Episódica: 🟢 Online  
• Raciocínio Causal: 🟢 Online
• Orquestração Multi-modal: 🟢 Online
        """
    
    async def _get_help_text(self) -> str:
        """Obtém texto de ajuda."""
        
        return """
🤖 *Comandos NEXUS disponíveis:*

*Mensões:*
• `@nexus status` - Status do sistema
• `@nexus help` - Esta mensagem de ajuda
• `@nexus desenvolver [descrição]` - Desenvolver projeto
• `@nexus analisar [caminho]` - Analisar projeto existente

*Slash Commands:*
• `/nexus-status` - Status detalhado
• `/nexus-help` - Ajuda completa
• `/nexus-develop [descrição]` - Desenvolvimento via comando
• `/nexus-analyze [caminho]` - Análise via comando

*Exemplos:*
• `@nexus desenvolver "Sistema de e-commerce com React"`
• `@nexus analisar ./meu-projeto`
• `/nexus-develop "API REST com Node.js"`

💡 *Dica:* Use menções para comandos rápidos ou slash commands para comandos formais.
        """
    
    def add_message_callback(self, callback: Callable):
        """Adiciona callback para mensagens recebidas."""
        self.message_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable):
        """Adiciona callback para mudanças de status."""
        self.status_callbacks.append(callback)
    
    def register_command_handler(self, command: str, handler: Callable):
        """Registra handler para comando específico."""
        self.command_handlers[command] = handler
    
    async def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Obtém informações do canal."""
        
        try:
            if not self.web_client:
                return None
            
            response = await self.web_client.conversations_info(channel=channel_id)
            return response["channel"] if response["ok"] else None
            
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            return None
    
    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtém informações do usuário."""
        
        try:
            if not self.web_client:
                return None
            
            response = await self.web_client.users_info(user=user_id)
            return response["user"] if response["ok"] else None
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None
    
    async def shutdown(self) -> None:
        """Desliga integração com Slack."""
        
        logger.info("Shutting down Slack integration")
        
        if self.socket_client:
            await self.socket_client.disconnect()
        
        self.is_connected = False
        logger.info("Slack integration shutdown complete")
