"""
Working Memory System

Implementa o sistema de memória de trabalho do NEXUS, responsável por manter
contexto ativo, estado e cache durante o processamento cognitivo.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Item individual na memória de trabalho."""
    
    key: str
    value: Any
    priority: float = 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Metadados
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class MemoryStats:
    """Estatísticas da memória de trabalho."""
    
    total_items: int
    total_size_bytes: int
    hit_rate: float
    miss_rate: float
    eviction_count: int
    average_access_time: float
    
    # Distribuição por prioridade
    priority_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Métricas temporais
    items_created_last_hour: int = 0
    items_accessed_last_hour: int = 0


class WorkingMemory:
    """
    Sistema de Memória de Trabalho do NEXUS.
    
    Mantém contexto ativo, estado e cache durante processamento cognitivo,
    com gerenciamento inteligente de capacidade e eviction policies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa a Memória de Trabalho.
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        
        # Configurações de capacidade
        self.max_items = self.config.get('max_items', 10000)
        self.max_size_bytes = self.config.get('max_size_bytes', 1024 * 1024 * 100)  # 100MB
        self.default_ttl_seconds = self.config.get('default_ttl_seconds', 3600)  # 1 hour
        
        # Armazenamento principal
        self.memory: Dict[str, MemoryItem] = {}
        
        # Índices para acesso eficiente
        self.priority_index: Dict[float, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.access_history: deque = deque(maxlen=1000)
        
        # Estatísticas
        self.stats = MemoryStats(0, 0, 0.0, 0.0, 0, 0.0)
        self.access_times: deque = deque(maxlen=100)
        
        # Contexto atual
        self.current_context: Dict[str, Any] = {}
        self.context_stack: List[Dict[str, Any]] = []
        
        # Locks para thread safety
        self._memory_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()
        
        # Task para limpeza periódica
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Working Memory initialized")
    
    async def initialize(self) -> None:
        """Inicializa a memória de trabalho."""
        
        # Iniciar limpeza periódica
        cleanup_interval = self.config.get('cleanup_interval_seconds', 300)  # 5 minutes
        self._cleanup_task = asyncio.create_task(
            self._periodic_cleanup(cleanup_interval)
        )
        
        logger.info("Working Memory initialization complete")
    
    async def store(
        self, 
        key: str, 
        value: Any, 
        priority: float = 1.0,
        ttl_seconds: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """
        Armazena um item na memória de trabalho.
        
        Args:
            key: Chave única do item
            value: Valor a ser armazenado
            priority: Prioridade do item (maior = mais importante)
            ttl_seconds: Tempo de vida em segundos
            tags: Tags para categorização
            
        Returns:
            True se armazenado com sucesso
        """
        async with self._memory_lock:
            
            # Calcular TTL
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds
            
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            # Calcular tamanho
            size_bytes = self._calculate_size(value)
            
            # Verificar se precisa fazer eviction
            if await self._needs_eviction(size_bytes):
                await self._evict_items(size_bytes)
            
            # Remover item existente se houver
            if key in self.memory:
                await self._remove_item(key)
            
            # Criar novo item
            item = MemoryItem(
                key=key,
                value=value,
                priority=priority,
                size_bytes=size_bytes,
                expires_at=expires_at,
                tags=tags or set()
            )
            
            # Armazenar item
            self.memory[key] = item
            
            # Atualizar índices
            self.priority_index[priority].add(key)
            for tag in item.tags:
                self.tag_index[tag].add(key)
            
            # Atualizar estatísticas
            await self._update_stats_on_store(item)
            
            logger.debug(f"Stored item in working memory: {key}")
            return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Recupera um item da memória de trabalho.
        
        Args:
            key: Chave do item
            
        Returns:
            Valor do item ou None se não encontrado
        """
        start_time = datetime.utcnow()
        
        async with self._memory_lock:
            
            item = self.memory.get(key)
            
            if item is None:
                # Cache miss
                await self._record_access(key, hit=False)
                return None
            
            # Verificar expiração
            if item.expires_at and datetime.utcnow() > item.expires_at:
                await self._remove_item(key)
                await self._record_access(key, hit=False)
                return None
            
            # Atualizar estatísticas de acesso
            item.access_count += 1
            item.last_accessed = datetime.utcnow()
            
            # Registrar acesso
            await self._record_access(key, hit=True)
            
            # Registrar tempo de acesso
            access_time = (datetime.utcnow() - start_time).total_seconds()
            self.access_times.append(access_time)
            
            return item.value
    
    async def update_context(self, context_updates: Dict[str, Any]) -> None:
        """
        Atualiza o contexto atual da memória de trabalho.
        
        Args:
            context_updates: Atualizações do contexto
        """
        async with self._memory_lock:
            
            # Atualizar contexto atual
            self.current_context.update(context_updates)
            
            # Armazenar contexto na memória com alta prioridade
            context_key = f"context_{datetime.utcnow().isoformat()}"
            await self.store(
                key=context_key,
                value=dict(self.current_context),
                priority=10.0,  # Alta prioridade
                tags={'context', 'system'}
            )
            
            logger.debug("Working memory context updated")
    
    async def push_context(self, new_context: Dict[str, Any]) -> None:
        """
        Empilha um novo contexto (salva o atual e ativa o novo).
        
        Args:
            new_context: Novo contexto a ser ativado
        """
        async with self._memory_lock:
            
            # Salvar contexto atual na pilha
            self.context_stack.append(dict(self.current_context))
            
            # Ativar novo contexto
            self.current_context = new_context.copy()
            
            logger.debug("Context pushed to working memory stack")
    
    async def pop_context(self) -> Optional[Dict[str, Any]]:
        """
        Desempilha o contexto (restaura o contexto anterior).
        
        Returns:
            Contexto anterior ou None se pilha vazia
        """
        async with self._memory_lock:
            
            if not self.context_stack:
                return None
            
            # Restaurar contexto anterior
            previous_context = self.context_stack.pop()
            self.current_context = previous_context
            
            logger.debug("Context popped from working memory stack")
            return previous_context
    
    async def search_by_tags(self, tags: Set[str]) -> List[Any]:
        """
        Busca itens por tags.
        
        Args:
            tags: Tags para busca
            
        Returns:
            Lista de valores dos itens encontrados
        """
        async with self._memory_lock:
            
            # Encontrar chaves que têm todas as tags
            matching_keys = None
            
            for tag in tags:
                tag_keys = self.tag_index.get(tag, set())
                if matching_keys is None:
                    matching_keys = tag_keys.copy()
                else:
                    matching_keys &= tag_keys
            
            if not matching_keys:
                return []
            
            # Recuperar valores
            results = []
            for key in matching_keys:
                value = await self.retrieve(key)
                if value is not None:
                    results.append(value)
            
            return results
    
    async def get_current_context(self) -> Dict[str, Any]:
        """
        Obtém o contexto atual.
        
        Returns:
            Contexto atual
        """
        return dict(self.current_context)
    
    async def calculate_load(self) -> float:
        """
        Calcula a carga atual da memória de trabalho.
        
        Returns:
            Carga normalizada (0-1)
        """
        async with self._memory_lock:
            
            # Carga baseada no número de itens
            item_load = len(self.memory) / self.max_items
            
            # Carga baseada no tamanho
            current_size = sum(item.size_bytes for item in self.memory.values())
            size_load = current_size / self.max_size_bytes
            
            # Carga combinada (pior caso)
            combined_load = max(item_load, size_load)
            
            return min(combined_load, 1.0)
    
    async def get_state(self) -> Dict[str, Any]:
        """
        Obtém o estado atual da memória de trabalho.
        
        Returns:
            Estado detalhado da memória
        """
        async with self._memory_lock:
            
            current_size = sum(item.size_bytes for item in self.memory.values())
            
            return {
                'item_count': len(self.memory),
                'total_size_bytes': current_size,
                'load_percentage': await self.calculate_load() * 100,
                'context_stack_depth': len(self.context_stack),
                'current_context_size': len(self.current_context),
                'stats': self.stats,
                'top_priorities': await self._get_top_priority_items(5)
            }
    
    async def optimize_parameters(self) -> None:
        """Otimiza parâmetros da memória de trabalho."""
        
        # Analisar padrões de acesso
        access_patterns = await self._analyze_access_patterns()
        
        # Ajustar TTL baseado nos padrões
        if access_patterns.get('frequent_expiration', False):
            self.default_ttl_seconds = int(self.default_ttl_seconds * 1.5)
            logger.info(f"Increased default TTL to {self.default_ttl_seconds}s")
        
        # Ajustar capacidade se necessário
        if access_patterns.get('frequent_eviction', False):
            self.max_items = int(self.max_items * 1.2)
            logger.info(f"Increased max items to {self.max_items}")
        
        # Executar limpeza agressiva se necessário
        if await self.calculate_load() > 0.9:
            await self._aggressive_cleanup()
    
    async def _needs_eviction(self, new_item_size: int) -> bool:
        """Verifica se precisa fazer eviction."""
        
        current_items = len(self.memory)
        current_size = sum(item.size_bytes for item in self.memory.values())
        
        would_exceed_items = current_items >= self.max_items
        would_exceed_size = (current_size + new_item_size) > self.max_size_bytes
        
        return would_exceed_items or would_exceed_size
    
    async def _evict_items(self, space_needed: int) -> None:
        """Executa eviction de itens."""
        
        # Estratégia: LRU com consideração de prioridade
        items_by_score = []
        
        for key, item in self.memory.items():
            # Score baseado em: prioridade (negativa), último acesso, contagem de acesso
            time_since_access = (datetime.utcnow() - item.last_accessed).total_seconds()
            score = (-item.priority * 1000) + time_since_access - (item.access_count * 10)
            items_by_score.append((score, key, item))
        
        # Ordenar por score (maior score = mais provável de ser removido)
        items_by_score.sort(reverse=True)
        
        # Remover itens até liberar espaço suficiente
        freed_space = 0
        items_removed = 0
        
        for score, key, item in items_by_score:
            if freed_space >= space_needed and items_removed > 0:
                break
            
            await self._remove_item(key)
            freed_space += item.size_bytes
            items_removed += 1
        
        self.stats.eviction_count += items_removed
        logger.debug(f"Evicted {items_removed} items, freed {freed_space} bytes")
    
    async def _remove_item(self, key: str) -> None:
        """Remove um item da memória."""
        
        item = self.memory.get(key)
        if not item:
            return
        
        # Remover do armazenamento principal
        del self.memory[key]
        
        # Remover dos índices
        self.priority_index[item.priority].discard(key)
        if not self.priority_index[item.priority]:
            del self.priority_index[item.priority]
        
        for tag in item.tags:
            self.tag_index[tag].discard(key)
            if not self.tag_index[tag]:
                del self.tag_index[tag]
    
    async def _calculate_size(self, value: Any) -> int:
        """Calcula o tamanho aproximado de um valor."""
        
        try:
            # Serializar para JSON para estimar tamanho
            json_str = json.dumps(value, default=str)
            return len(json_str.encode('utf-8'))
        except (TypeError, ValueError):
            # Fallback para tipos não serializáveis
            return len(str(value).encode('utf-8'))
    
    async def _record_access(self, key: str, hit: bool) -> None:
        """Registra um acesso para estatísticas."""
        
        self.access_history.append({
            'key': key,
            'hit': hit,
            'timestamp': datetime.utcnow()
        })
        
        # Atualizar estatísticas
        async with self._stats_lock:
            if len(self.access_history) >= 100:
                recent_accesses = list(self.access_history)[-100:]
                hits = sum(1 for access in recent_accesses if access['hit'])
                self.stats.hit_rate = hits / len(recent_accesses)
                self.stats.miss_rate = 1.0 - self.stats.hit_rate
    
    async def _update_stats_on_store(self, item: MemoryItem) -> None:
        """Atualiza estatísticas ao armazenar item."""
        
        async with self._stats_lock:
            self.stats.total_items = len(self.memory)
            self.stats.total_size_bytes = sum(i.size_bytes for i in self.memory.values())
            
            # Atualizar distribuição de prioridade
            priority_str = f"{item.priority:.1f}"
            if priority_str not in self.stats.priority_distribution:
                self.stats.priority_distribution[priority_str] = 0
            self.stats.priority_distribution[priority_str] += 1
    
    async def _get_top_priority_items(self, count: int) -> List[Dict[str, Any]]:
        """Obtém os itens de maior prioridade."""
        
        items_with_priority = [
            (item.priority, key, item) 
            for key, item in self.memory.items()
        ]
        items_with_priority.sort(reverse=True)
        
        return [
            {
                'key': key,
                'priority': priority,
                'access_count': item.access_count,
                'size_bytes': item.size_bytes
            }
            for priority, key, item in items_with_priority[:count]
        ]
    
    async def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analisa padrões de acesso."""
        
        patterns = {
            'frequent_expiration': False,
            'frequent_eviction': False,
            'hot_keys': []
        }
        
        if len(self.access_history) < 50:
            return patterns
        
        recent_accesses = list(self.access_history)[-100:]
        
        # Verificar expiração frequente
        miss_rate = sum(1 for access in recent_accesses if not access['hit']) / len(recent_accesses)
        patterns['frequent_expiration'] = miss_rate > 0.3
        
        # Verificar eviction frequente
        patterns['frequent_eviction'] = self.stats.eviction_count > 10
        
        # Identificar chaves quentes
        key_access_count = defaultdict(int)
        for access in recent_accesses:
            if access['hit']:
                key_access_count[access['key']] += 1
        
        hot_keys = sorted(key_access_count.items(), key=lambda x: x[1], reverse=True)[:5]
        patterns['hot_keys'] = [key for key, count in hot_keys]
        
        return patterns
    
    async def _periodic_cleanup(self, interval_seconds: int) -> None:
        """Limpeza periódica de itens expirados."""
        
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                
                # Remover itens expirados
                expired_keys = []
                current_time = datetime.utcnow()
                
                async with self._memory_lock:
                    for key, item in self.memory.items():
                        if item.expires_at and current_time > item.expires_at:
                            expired_keys.append(key)
                
                # Remover itens expirados
                for key in expired_keys:
                    async with self._memory_lock:
                        await self._remove_item(key)
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired items")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}", exc_info=True)
    
    async def _aggressive_cleanup(self) -> None:
        """Limpeza agressiva para liberar espaço."""
        
        async with self._memory_lock:
            
            # Remover itens de baixa prioridade e pouco acessados
            items_to_remove = []
            
            for key, item in self.memory.items():
                if item.priority < 2.0 and item.access_count < 2:
                    items_to_remove.append(key)
            
            # Remover até 30% dos itens
            max_remove = len(self.memory) // 3
            items_to_remove = items_to_remove[:max_remove]
            
            for key in items_to_remove:
                await self._remove_item(key)
            
            logger.info(f"Aggressive cleanup removed {len(items_to_remove)} items")
    
    async def shutdown(self) -> None:
        """Desliga a memória de trabalho."""
        
        logger.info("Shutting down Working Memory")
        
        # Cancelar limpeza periódica
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Limpar memória
        async with self._memory_lock:
            self.memory.clear()
            self.priority_index.clear()
            self.tag_index.clear()
            self.access_history.clear()
            self.current_context.clear()
            self.context_stack.clear()
        
        logger.info("Working Memory shutdown complete")
