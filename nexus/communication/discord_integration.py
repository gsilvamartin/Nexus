"""
Discord Integration Module

Integração completa com Discord para comunicação bidirecional,
notificações e execução de comandos do NEXUS.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json

import discord
from discord.ext import commands
from discord import app_commands

logger = logging.getLogger(__name__)


class DiscordIntegration:
    """
    Integração completa com Discord para o sistema NEXUS.
    
    Suporta:
    - Bot Discord com slash commands
    - Recebimento de mensagens em tempo real
    - Envio de notificações e updates
    - Sistema de permissões e roles
    - Integração com canais específicos
    - Embed messages ricas
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa integração com Discord.
        
        Args:
            config: Configuração contendo token e configurações do Discord
        """
        self.config = config
        self.bot_token = config.get('bot_token')
        self.guild_id = config.get('guild_id')
        self.admin_role_id = config.get('admin_role_id')
        
        # Bot Discord
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        
        self.bot = commands.Bot(
            command_prefix='!nexus ',
            intents=intents,
            help_command=None
        )
        
        # Estado da integração
        self.is_connected = False
        self.connected_guilds: List[int] = []
        self.command_handlers: Dict[str, Callable] = {}
        
        # Callbacks para eventos
        self.message_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # Configurar eventos
        self._setup_events()
        
        logger.info("Discord Integration initialized")
    
    def _setup_events(self):
        """Configura eventos do bot Discord."""
        
        @self.bot.event
        async def on_ready():
            logger.info(f'Discord bot logged in as {self.bot.user}')
            self.is_connected = True
            
            # Sincronizar slash commands
            if self.guild_id:
                guild = discord.Object(id=int(self.guild_id))
                await self.bot.tree.sync(guild=guild)
                logger.info(f"Slash commands synced for guild {self.guild_id}")
            else:
                await self.bot.tree.sync()
                logger.info("Slash commands synced globally")
        
        @self.bot.event
        async def on_message(message):
            # Ignorar mensagens do próprio bot
            if message.author == self.bot.user:
                return
            
            # Processar comandos de prefixo
            await self.bot.process_commands(message)
            
            # Processar menções ao bot
            if self.bot.user.mentioned_in(message):
                await self._handle_mention(message)
            
            # Notificar callbacks
            message_data = {
                'channel_id': message.channel.id,
                'guild_id': message.guild.id if message.guild else None,
                'user_id': message.author.id,
                'username': message.author.name,
                'content': message.content,
                'timestamp': message.created_at.isoformat(),
                'event_type': 'message'
            }
            
            for callback in self.message_callbacks:
                try:
                    await callback(message_data)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
        
        @self.bot.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.CommandNotFound):
                return
            logger.error(f"Command error: {error}")
            await ctx.send(f"❌ Erro ao executar comando: {error}")
        
        # Configurar slash commands
        self._setup_slash_commands()
    
    def _setup_slash_commands(self):
        """Configura slash commands do Discord."""
        
        @self.bot.tree.command(name="nexus-status", description="Mostra status do sistema NEXUS")
        async def nexus_status(interaction: discord.Interaction):
            await interaction.response.defer()
            
            status_text = await self._get_system_status_text()
            embed = self._create_status_embed(status_text)
            
            await interaction.followup.send(embed=embed)
        
        @self.bot.tree.command(name="nexus-help", description="Mostra ajuda dos comandos NEXUS")
        async def nexus_help(interaction: discord.Interaction):
            await interaction.response.defer()
            
            help_text = await self._get_help_text()
            embed = self._create_help_embed(help_text)
            
            await interaction.followup.send(embed=embed)
        
        @self.bot.tree.command(name="nexus-develop", description="Inicia desenvolvimento de projeto")
        @app_commands.describe(description="Descrição do projeto a ser desenvolvido")
        async def nexus_develop(interaction: discord.Interaction, description: str):
            await interaction.response.defer()
            
            # Verificar permissões
            if not await self._check_permissions(interaction):
                await interaction.followup.send("❌ Você não tem permissão para usar este comando.")
                return
            
            # Enviar confirmação
            embed = discord.Embed(
                title="🚀 Desenvolvimento Iniciado",
                description=f"**Projeto:** {description}",
                color=0x00ff00
            )
            embed.add_field(name="Status", value="⏳ Processando... (isso pode levar alguns minutos)", inline=False)
            
            await interaction.followup.send(embed=embed)
            
            # Notificar callbacks
            for callback in self.status_callbacks:
                try:
                    await callback('development_started', {
                        'description': description,
                        'channel_id': interaction.channel_id,
                        'user_id': interaction.user.id,
                        'guild_id': interaction.guild_id
                    })
                except Exception as e:
                    logger.error(f"Error in development callback: {e}")
        
        @self.bot.tree.command(name="nexus-analyze", description="Analisa projeto existente")
        @app_commands.describe(project_path="Caminho do projeto a ser analisado")
        async def nexus_analyze(interaction: discord.Interaction, project_path: str):
            await interaction.response.defer()
            
            # Verificar permissões
            if not await self._check_permissions(interaction):
                await interaction.followup.send("❌ Você não tem permissão para usar este comando.")
                return
            
            # Enviar confirmação
            embed = discord.Embed(
                title="🔍 Análise Iniciada",
                description=f"**Projeto:** {project_path}",
                color=0x0099ff
            )
            embed.add_field(name="Status", value="⏳ Processando análise... (isso pode levar alguns minutos)", inline=False)
            
            await interaction.followup.send(embed=embed)
            
            # Notificar callbacks
            for callback in self.status_callbacks:
                try:
                    await callback('analysis_started', {
                        'project_path': project_path,
                        'channel_id': interaction.channel_id,
                        'user_id': interaction.user.id,
                        'guild_id': interaction.guild_id
                    })
                except Exception as e:
                    logger.error(f"Error in analysis callback: {e}")
        
        @self.bot.tree.command(name="nexus-monitor", description="Inicia monitoramento de projeto")
        @app_commands.describe(project_path="Caminho do projeto para monitorar")
        async def nexus_monitor(interaction: discord.Interaction, project_path: str):
            await interaction.response.defer()
            
            # Verificar permissões
            if not await self._check_permissions(interaction):
                await interaction.followup.send("❌ Você não tem permissão para usar este comando.")
                return
            
            # Enviar confirmação
            embed = discord.Embed(
                title="📊 Monitoramento Iniciado",
                description=f"**Projeto:** {project_path}",
                color=0xff9900
            )
            embed.add_field(name="Status", value="⏳ Configurando monitoramento...", inline=False)
            
            await interaction.followup.send(embed=embed)
            
            # Notificar callbacks
            for callback in self.status_callbacks:
                try:
                    await callback('monitoring_started', {
                        'project_path': project_path,
                        'channel_id': interaction.channel_id,
                        'user_id': interaction.user.id,
                        'guild_id': interaction.guild_id
                    })
                except Exception as e:
                    logger.error(f"Error in monitoring callback: {e}")
    
    async def _handle_mention(self, message: discord.Message):
        """Manipula menções ao bot."""
        
        content = message.content.lower()
        channel = message.channel
        
        # Comandos básicos via menção
        if 'status' in content or 'status do sistema' in content:
            status_text = await self._get_system_status_text()
            embed = self._create_status_embed(status_text)
            await channel.send(embed=embed)
        
        elif 'help' in content or 'ajuda' in content:
            help_text = await self._get_help_text()
            embed = self._create_help_embed(help_text)
            await channel.send(embed=embed)
        
        elif 'desenvolver' in content or 'develop' in content:
            await self._handle_development_mention(message)
        
        elif 'analisar' in content or 'analyze' in content:
            await self._handle_analysis_mention(message)
        
        else:
            # Comando não reconhecido
            embed = discord.Embed(
                title="❓ Comando não reconhecido",
                description="Use `@nexus help` para ver comandos disponíveis ou `/nexus-help` para slash commands.",
                color=0xff0000
            )
            await channel.send(embed=embed)
    
    async def _handle_development_mention(self, message: discord.Message):
        """Manipula solicitações de desenvolvimento via menção."""
        
        content = message.content
        channel = message.channel
        
        # Extrair descrição do projeto
        description = self._extract_project_description(content)
        
        if not description:
            embed = discord.Embed(
                title="❌ Descrição necessária",
                description="Por favor, forneça uma descrição do projeto a ser desenvolvido.\n\nExemplo: `@nexus desenvolver Sistema de e-commerce com React`",
                color=0xff0000
            )
            await channel.send(embed=embed)
            return
        
        # Enviar confirmação
        embed = discord.Embed(
            title="🚀 Desenvolvimento Iniciado",
            description=f"**Projeto:** {description}",
            color=0x00ff00
        )
        embed.add_field(name="Status", value="⏳ Processando... (isso pode levar alguns minutos)", inline=False)
        
        await channel.send(embed=embed)
        
        # Notificar callbacks
        for callback in self.status_callbacks:
            try:
                await callback('development_started', {
                    'description': description,
                    'channel_id': channel.id,
                    'user_id': message.author.id,
                    'guild_id': message.guild.id if message.guild else None
                })
            except Exception as e:
                logger.error(f"Error in development callback: {e}")
    
    async def _handle_analysis_mention(self, message: discord.Message):
        """Manipula solicitações de análise via menção."""
        
        content = message.content
        channel = message.channel
        
        # Extrair caminho do projeto
        project_path = self._extract_project_path(content)
        
        if not project_path:
            embed = discord.Embed(
                title="❌ Caminho necessário",
                description="Por favor, forneça o caminho do projeto a ser analisado.\n\nExemplo: `@nexus analisar ./meu-projeto`",
                color=0xff0000
            )
            await channel.send(embed=embed)
            return
        
        # Enviar confirmação
        embed = discord.Embed(
            title="🔍 Análise Iniciada",
            description=f"**Projeto:** {project_path}",
            color=0x0099ff
        )
        embed.add_field(name="Status", value="⏳ Processando análise... (isso pode levar alguns minutos)", inline=False)
        
        await channel.send(embed=embed)
        
        # Notificar callbacks
        for callback in self.status_callbacks:
            try:
                await callback('analysis_started', {
                    'project_path': project_path,
                    'channel_id': channel.id,
                    'user_id': message.author.id,
                    'guild_id': message.guild.id if message.guild else None
                })
            except Exception as e:
                logger.error(f"Error in analysis callback: {e}")
    
    def _extract_project_description(self, text: str) -> Optional[str]:
        """Extrai descrição do projeto do texto da mensagem."""
        
        # Remover menção ao bot
        text = text.replace(f'<@{self.bot.user.id}>', '').strip()
        
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
        text = text.replace(f'<@{self.bot.user.id}>', '').strip()
        
        # Procurar por caminhos (começando com ./ ou /)
        import re
        path_match = re.search(r'(\.?\/[^\s]+|\/[^\s]+)', text)
        
        if path_match:
            return path_match.group(1)
        
        return None
    
    async def _check_permissions(self, interaction: discord.Interaction) -> bool:
        """Verifica se usuário tem permissões para usar comandos."""
        
        # Se não há role de admin definido, permitir para todos
        if not self.admin_role_id:
            return True
        
        # Verificar se usuário tem role de admin
        user_roles = [role.id for role in interaction.user.roles]
        return int(self.admin_role_id) in user_roles
    
    def _create_status_embed(self, status_text: str) -> discord.Embed:
        """Cria embed de status do sistema."""
        
        embed = discord.Embed(
            title="🤖 Status do Sistema NEXUS",
            color=0x00ff00
        )
        
        # Parse do texto de status (implementação simplificada)
        lines = status_text.strip().split('\n')
        
        for line in lines:
            if line.strip():
                if 'Sistema:' in line:
                    embed.add_field(name="Sistema", value=line.split('Sistema:')[1].strip(), inline=True)
                elif 'Córtices:' in line:
                    embed.add_field(name="Córtices", value=line.split('Córtices:')[1].strip(), inline=True)
                elif 'Memória episódica:' in line:
                    embed.add_field(name="Memória", value=line.split('Memória episódica:')[1].strip(), inline=True)
                elif 'Modelos disponíveis:' in line:
                    embed.add_field(name="Modelos", value=line.split('Modelos disponíveis:')[1].strip(), inline=True)
                elif 'Performance média:' in line:
                    embed.add_field(name="Performance", value=line.split('Performance média:')[1].strip(), inline=True)
                elif 'Requisições hoje:' in line:
                    embed.add_field(name="Requisições", value=line.split('Requisições hoje:')[1].strip(), inline=True)
                elif 'Taxa de sucesso:' in line:
                    embed.add_field(name="Taxa de Sucesso", value=line.split('Taxa de sucesso:')[1].strip(), inline=True)
        
        embed.set_footer(text="NEXUS - Sistema de Desenvolvimento Autônomo")
        embed.timestamp = datetime.utcnow()
        
        return embed
    
    def _create_help_embed(self, help_text: str) -> discord.Embed:
        """Cria embed de ajuda."""
        
        embed = discord.Embed(
            title="🤖 Comandos NEXUS",
            description="Sistema de Desenvolvimento Autônomo de Próxima Geração",
            color=0x0099ff
        )
        
        # Parse do texto de ajuda (implementação simplificada)
        sections = help_text.split('*')
        
        current_section = None
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if section.startswith('Mensões:') or section.startswith('Slash Commands:') or section.startswith('Exemplos:'):
                current_section = section
            elif current_section and section:
                embed.add_field(name=current_section, value=section, inline=False)
        
        embed.set_footer(text="Use @nexus para comandos rápidos ou /nexus-* para slash commands")
        embed.timestamp = datetime.utcnow()
        
        return embed
    
    async def send_message(
        self, 
        channel_id: int, 
        content: str, 
        embed: Optional[discord.Embed] = None
    ) -> bool:
        """
        Envia mensagem para canal do Discord.
        
        Args:
            channel_id: ID do canal
            content: Conteúdo da mensagem
            embed: Embed opcional
            
        Returns:
            True se enviado com sucesso
        """
        
        try:
            channel = self.bot.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return False
            
            await channel.send(content=content, embed=embed)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
            return False
    
    async def send_embed(
        self, 
        channel_id: int, 
        title: str, 
        description: str,
        color: int = 0x0099ff,
        fields: Optional[List[Dict[str, str]]] = None,
        thumbnail_url: Optional[str] = None
    ) -> bool:
        """
        Envia embed rica para canal do Discord.
        
        Args:
            channel_id: ID do canal
            title: Título do embed
            description: Descrição principal
            color: Cor do embed (hex)
            fields: Campos adicionais
            thumbnail_url: URL da thumbnail
        """
        
        try:
            embed = discord.Embed(
                title=title,
                description=description,
                color=color,
                timestamp=datetime.utcnow()
            )
            
            if fields:
                for field in fields:
                    embed.add_field(
                        name=field['name'],
                        value=field['value'],
                        inline=field.get('inline', False)
                    )
            
            if thumbnail_url:
                embed.set_thumbnail(url=thumbnail_url)
            
            embed.set_footer(text="NEXUS - Sistema de Desenvolvimento Autônomo")
            
            return await self.send_message(channel_id=channel_id, content="", embed=embed)
            
        except Exception as e:
            logger.error(f"Failed to send Discord embed: {e}")
            return False
    
    async def send_file(
        self, 
        channel_id: int, 
        file_path: str, 
        filename: Optional[str] = None,
        content: Optional[str] = None
    ) -> bool:
        """
        Envia arquivo para canal do Discord.
        
        Args:
            channel_id: ID do canal
            file_path: Caminho do arquivo
            filename: Nome do arquivo (opcional)
            content: Conteúdo da mensagem (opcional)
        """
        
        try:
            channel = self.bot.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return False
            
            file = discord.File(file_path, filename=filename)
            await channel.send(content=content, file=file)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord file: {e}")
            return False
    
    async def _get_system_status_text(self) -> str:
        """Obtém texto de status do sistema."""
        
        # Implementação simplificada - integrar com NEXUS Core
        return """
🤖 Status do Sistema NEXUS

🟢 Sistema: Ativo
🧠 Córtices: 4/4 ativos
💾 Memória episódica: 45% utilizada
🤖 Modelos disponíveis: 3
⚡ Performance média: 1.2s
📊 Requisições hoje: 127
✅ Taxa de sucesso: 98.4%

Componentes:
• Substrato Cognitivo: 🟢 Online
• Memória Episódica: 🟢 Online  
• Raciocínio Causal: 🟢 Online
• Orquestração Multi-modal: 🟢 Online
        """
    
    async def _get_help_text(self) -> str:
        """Obtém texto de ajuda."""
        
        return """
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
• `/nexus-monitor [caminho]` - Monitoramento de projeto

*Exemplos:*
• `@nexus desenvolver "Sistema de e-commerce com React"`
• `@nexus analisar ./meu-projeto`
• `/nexus-develop "API REST com Node.js"`

💡 Dica: Use menções para comandos rápidos ou slash commands para comandos formais.
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
    
    async def get_guild_info(self, guild_id: int) -> Optional[discord.Guild]:
        """Obtém informações do servidor."""
        return self.bot.get_guild(guild_id)
    
    async def get_channel_info(self, channel_id: int) -> Optional[discord.TextChannel]:
        """Obtém informações do canal."""
        return self.bot.get_channel(channel_id)
    
    async def get_user_info(self, user_id: int) -> Optional[discord.User]:
        """Obtém informações do usuário."""
        return self.bot.get_user(user_id)
    
    async def initialize(self) -> None:
        """Inicializa integração com Discord."""
        
        try:
            # Iniciar bot
            await self.bot.start(self.bot_token)
            
        except Exception as e:
            logger.error(f"Failed to initialize Discord integration: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Desliga integração com Discord."""
        
        logger.info("Shutting down Discord integration")
        
        if self.bot:
            await self.bot.close()
        
        self.is_connected = False
        logger.info("Discord integration shutdown complete")
