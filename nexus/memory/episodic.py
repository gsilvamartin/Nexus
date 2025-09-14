"""
Episodic Memory System

Implementa o sistema de memória episódica do NEXUS com banco de dados temporal,
consolidação de memória e recuperação baseada em padrões.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
import numpy as np
from enum import Enum

from nexus.memory.temporal_graph import TemporalGraphDatabase
from nexus.memory.consolidation import MemoryConsolidationEngine
from nexus.memory.pattern_detection import PatternDetectionEngine
from nexus.memory.experience_encoder import ExperienceEncoder

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Tipos de memória episódica."""
    EXPERIENCE = "experience"
    LEARNING = "learning"
    DECISION = "decision"
    OUTCOME = "outcome"
    PATTERN = "pattern"
    INSIGHT = "insight"


class ConsolidationLevel(Enum):
    """Níveis de consolidação da memória."""
    WORKING = "working"        # Memória de trabalho temporária
    SHORT_TERM = "short_term"  # Memória de curto prazo
    LONG_TERM = "long_term"    # Memória de longo prazo
    PERMANENT = "permanent"    # Memória permanente


@dataclass
class MemoryTrace:
    """Traço de memória episódica."""
    
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    
    # Contexto temporal
    timestamp: datetime
    duration: Optional[timedelta] = None
    
    # Contexto causal
    causal_chain: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    
    # Contexto emocional/avaliativo
    success_metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    importance: float = 0.0
    
    # Contexto estratégico
    goal_context: Dict[str, Any] = field(default_factory=dict)
    strategy_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadados de consolidação
    consolidation_level: ConsolidationLevel = ConsolidationLevel.WORKING
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    # Relacionamentos
    related_memories: Set[str] = field(default_factory=set)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    
    # Encoding
    encoded_features: Optional[np.ndarray] = None
    embedding_version: str = "v1.0"


@dataclass
class MemoryQuery:
    """Query para recuperação de memórias."""
    
    # Critérios de busca
    memory_types: Optional[List[MemoryType]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    content_keywords: Optional[List[str]] = None
    
    # Contexto de busca
    current_context: Optional[Dict[str, Any]] = None
    goal_context: Optional[Dict[str, Any]] = None
    
    # Dimensões de similaridade
    similarity_dimensions: List[str] = field(default_factory=lambda: ['semantic', 'structural', 'temporal', 'causal'])
    
    # Parâmetros de busca
    max_results: int = 10
    min_similarity: float = 0.5
    consolidation_levels: List[ConsolidationLevel] = field(default_factory=lambda: list(ConsolidationLevel))


@dataclass
class MemoryInsight:
    """Insight derivado da análise de memórias."""
    
    insight_type: str
    description: str
    confidence: float
    
    # Evidências
    supporting_memories: List[str] = field(default_factory=list)
    pattern_evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Aplicabilidade
    applicable_contexts: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    # Metadados
    generated_at: datetime = field(default_factory=datetime.utcnow)
    validity_period: Optional[timedelta] = None


class EpisodicMemorySystem:
    """
    Sistema de Memória Episódica do NEXUS.
    
    Implementa memória episódica persistente com banco de dados temporal,
    consolidação automática e recuperação baseada em padrões multi-dimensionais.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o Sistema de Memória Episódica.
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        
        # Componentes especializados
        self.temporal_db = TemporalGraphDatabase(self.config.get('temporal_db', {}))
        self.experience_encoder = ExperienceEncoder(self.config.get('encoder', {}))
        self.pattern_detector = PatternDetectionEngine(self.config.get('pattern_detection', {}))
        self.consolidation_engine = MemoryConsolidationEngine(self.config.get('consolidation', {}))
        
        # Cache de memórias recentes
        self.recent_memories: Dict[str, MemoryTrace] = {}
        self.memory_index: Dict[str, Set[str]] = {}  # Índices para busca rápida
        
        # Configurações
        self.max_recent_memories = self.config.get('max_recent_memories', 1000)
        self.consolidation_threshold = self.config.get('consolidation_threshold', 100)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        
        # Estatísticas
        self.memory_stats = {
            'total_memories': 0,
            'memories_by_type': {},
            'consolidation_events': 0,
            'pattern_discoveries': 0
        }
        
        # Tasks assíncronas
        self._consolidation_task: Optional[asyncio.Task] = None
        self._pattern_detection_task: Optional[asyncio.Task] = None
        
        logger.info("Episodic Memory System initialized")
    
    async def initialize(self) -> None:
        """Inicializa o sistema de memória episódica."""
        
        # Inicializar componentes
        await self.temporal_db.initialize()
        await self.experience_encoder.initialize()
        await self.pattern_detector.initialize()
        await self.consolidation_engine.initialize()
        
        # Iniciar tarefas de background
        consolidation_interval = self.config.get('consolidation_interval_seconds', 300)  # 5 min
        pattern_detection_interval = self.config.get('pattern_detection_interval_seconds', 600)  # 10 min
        
        self._consolidation_task = asyncio.create_task(
            self._periodic_consolidation(consolidation_interval)
        )
        self._pattern_detection_task = asyncio.create_task(
            self._periodic_pattern_detection(pattern_detection_interval)
        )
        
        logger.info("Episodic Memory System initialization complete")
    
    async def store_experience(self, experience: Dict[str, Any]) -> str:
        """
        Armazena uma experiência na memória episódica.
        
        Args:
            experience: Dados da experiência
            
        Returns:
            ID da memória criada
        """
        # Gerar ID único para a memória
        memory_id = self._generate_memory_id(experience)
        
        # Codificar experiência com contexto rico
        encoded_experience = await self.experience_encoder.encode(
            experience,
            context={
                'temporal': experience.get('timestamp', datetime.utcnow()),
                'causal': experience.get('causal_chain', []),
                'emotional': experience.get('success_metrics', {}),
                'strategic': experience.get('goal_context', {})
            }
        )
        
        # Determinar tipo de memória
        memory_type = self._classify_memory_type(experience)
        
        # Criar traço de memória
        memory_trace = MemoryTrace(
            memory_id=memory_id,
            memory_type=memory_type,
            content=experience,
            timestamp=experience.get('timestamp', datetime.utcnow()),
            duration=experience.get('duration'),
            causal_chain=experience.get('causal_chain', []),
            success_metrics=experience.get('success_metrics', {}),
            confidence=experience.get('confidence', 0.0),
            importance=self._calculate_importance(experience),
            goal_context=experience.get('goal_context', {}),
            strategy_context=experience.get('strategy_context', {}),
            encoded_features=encoded_experience.get('features')
        )
        
        # Armazenar no cache recente
        self.recent_memories[memory_id] = memory_trace
        
        # Atualizar índices
        await self._update_memory_indices(memory_trace)
        
        # Armazenar no banco temporal
        await self.temporal_db.create_experience_node(memory_trace)
        
        # Vincular a experiências relacionadas
        await self._link_related_experiences(memory_trace)
        
        # Verificar se precisa consolidar
        if len(self.recent_memories) >= self.consolidation_threshold:
            asyncio.create_task(self._trigger_consolidation())
        
        # Atualizar estatísticas
        self._update_memory_statistics(memory_trace)
        
        logger.debug(f"Experience stored in episodic memory: {memory_id}")
        return memory_id
    
    async def retrieve_relevant_experiences(
        self, 
        current_context: Dict[str, Any],
        query: Optional[MemoryQuery] = None
    ) -> List[Tuple[MemoryTrace, float]]:
        """
        Recupera experiências relevantes para o contexto atual.
        
        Args:
            current_context: Contexto atual
            query: Query opcional para busca específica
            
        Returns:
            Lista de (memória, score_similaridade)
        """
        if query is None:
            query = MemoryQuery(current_context=current_context)
        
        # Busca multi-dimensional por similaridade
        similar_experiences = await self.temporal_db.similarity_search(
            current_context,
            dimensions=query.similarity_dimensions,
            max_results=query.max_results,
            min_similarity=query.min_similarity
        )
        
        # Busca baseada em padrões
        pattern_matches = await self.pattern_detector.find_pattern_matches(
            current_context, self.temporal_db
        )
        
        # Combinar e ranquear resultados
        combined_results = await self._combine_search_results(
            similar_experiences, pattern_matches, query
        )
        
        # Atualizar estatísticas de acesso
        for memory_trace, _ in combined_results:
            memory_trace.access_count += 1
            memory_trace.last_accessed = datetime.utcnow()
        
        logger.debug(f"Retrieved {len(combined_results)} relevant experiences")
        return combined_results
    
    async def synthesize_experience_insights(
        self, 
        experiences: List[MemoryTrace],
        context: Dict[str, Any]
    ) -> List[MemoryInsight]:
        """
        Sintetiza insights a partir de experiências.
        
        Args:
            experiences: Lista de experiências
            context: Contexto para síntese
            
        Returns:
            Lista de insights gerados
        """
        insights = []
        
        # Análise de padrões temporais
        temporal_insights = await self._analyze_temporal_patterns(experiences)
        insights.extend(temporal_insights)
        
        # Análise de padrões causais
        causal_insights = await self._analyze_causal_patterns(experiences)
        insights.extend(causal_insights)
        
        # Análise de padrões de sucesso
        success_insights = await self._analyze_success_patterns(experiences)
        insights.extend(success_insights)
        
        # Análise de padrões estratégicos
        strategic_insights = await self._analyze_strategic_patterns(experiences, context)
        insights.extend(strategic_insights)
        
        # Ranquear insights por relevância
        ranked_insights = await self._rank_insights_by_relevance(insights, context)
        
        logger.debug(f"Synthesized {len(ranked_insights)} insights from experiences")
        return ranked_insights
    
    async def consolidate_memories(self, force: bool = False) -> Dict[str, Any]:
        """
        Executa consolidação de memórias.
        
        Args:
            force: Forçar consolidação mesmo se não atingiu threshold
            
        Returns:
            Relatório de consolidação
        """
        if not force and len(self.recent_memories) < self.consolidation_threshold:
            return {'status': 'skipped', 'reason': 'threshold_not_met'}
        
        logger.info("Starting memory consolidation")
        
        # Selecionar memórias para consolidação
        memories_to_consolidate = await self._select_memories_for_consolidation()
        
        # Executar consolidação
        consolidation_result = await self.consolidation_engine.consolidate_memories(
            memories_to_consolidate
        )
        
        # Atualizar níveis de consolidação
        await self._update_consolidation_levels(consolidation_result)
        
        # Limpar cache de memórias recentes
        await self._cleanup_recent_memories(memories_to_consolidate)
        
        # Atualizar estatísticas
        self.memory_stats['consolidation_events'] += 1
        
        logger.info(f"Memory consolidation completed: {len(memories_to_consolidate)} memories processed")
        return consolidation_result
    
    async def discover_patterns(self) -> List[Dict[str, Any]]:
        """
        Descobre novos padrões nas memórias.
        
        Returns:
            Lista de padrões descobertos
        """
        logger.info("Starting pattern discovery")
        
        # Obter memórias para análise de padrões
        memories_for_analysis = await self._get_memories_for_pattern_analysis()
        
        # Executar detecção de padrões
        discovered_patterns = await self.pattern_detector.discover_patterns(
            memories_for_analysis
        )
        
        # Validar e filtrar padrões
        validated_patterns = await self._validate_discovered_patterns(discovered_patterns)
        
        # Armazenar padrões descobertos
        for pattern in validated_patterns:
            await self._store_discovered_pattern(pattern)
        
        # Atualizar estatísticas
        self.memory_stats['pattern_discoveries'] += len(validated_patterns)
        
        logger.info(f"Pattern discovery completed: {len(validated_patterns)} new patterns found")
        return validated_patterns
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Obtém estatísticas detalhadas da memória.
        
        Returns:
            Estatísticas da memória episódica
        """
        # Estatísticas básicas
        basic_stats = dict(self.memory_stats)
        
        # Estatísticas de consolidação
        consolidation_stats = await self.consolidation_engine.get_consolidation_statistics()
        
        # Estatísticas de padrões
        pattern_stats = await self.pattern_detector.get_pattern_statistics()
        
        # Estatísticas temporais
        temporal_stats = await self.temporal_db.get_temporal_statistics()
        
        # Distribuição por tipo de memória
        type_distribution = {}
        for memory in self.recent_memories.values():
            memory_type = memory.memory_type.value
            type_distribution[memory_type] = type_distribution.get(memory_type, 0) + 1
        
        # Métricas de performance
        performance_metrics = {
            'average_retrieval_time': await self._calculate_average_retrieval_time(),
            'memory_efficiency': await self._calculate_memory_efficiency(),
            'consolidation_effectiveness': await self._calculate_consolidation_effectiveness()
        }
        
        return {
            'basic_statistics': basic_stats,
            'consolidation_statistics': consolidation_stats,
            'pattern_statistics': pattern_stats,
            'temporal_statistics': temporal_stats,
            'type_distribution': type_distribution,
            'performance_metrics': performance_metrics,
            'system_health': await self._assess_memory_system_health()
        }
    
    def _generate_memory_id(self, experience: Dict[str, Any]) -> str:
        """Gera ID único para uma memória."""
        
        # Criar hash baseado no conteúdo e timestamp
        content_str = json.dumps(experience, sort_keys=True, default=str)
        timestamp_str = str(datetime.utcnow().timestamp())
        
        combined = f"{content_str}_{timestamp_str}"
        memory_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return f"mem_{memory_hash[:16]}"
    
    def _classify_memory_type(self, experience: Dict[str, Any]) -> MemoryType:
        """Classifica o tipo de memória baseado na experiência."""
        
        # Lógica de classificação baseada no conteúdo
        if 'learning' in experience or 'lesson' in experience:
            return MemoryType.LEARNING
        elif 'decision' in experience or 'choice' in experience:
            return MemoryType.DECISION
        elif 'outcome' in experience or 'result' in experience:
            return MemoryType.OUTCOME
        elif 'pattern' in experience:
            return MemoryType.PATTERN
        elif 'insight' in experience:
            return MemoryType.INSIGHT
        else:
            return MemoryType.EXPERIENCE
    
    def _calculate_importance(self, experience: Dict[str, Any]) -> float:
        """Calcula a importância de uma experiência."""
        
        importance = 0.5  # Base importance
        
        # Aumentar importância baseado em métricas de sucesso
        success_metrics = experience.get('success_metrics', {})
        if success_metrics:
            avg_success = sum(success_metrics.values()) / len(success_metrics)
            importance += avg_success * 0.3
        
        # Aumentar importância baseado na confiança
        confidence = experience.get('confidence', 0.0)
        importance += confidence * 0.2
        
        # Aumentar importância se há contexto estratégico
        if experience.get('goal_context') or experience.get('strategy_context'):
            importance += 0.2
        
        # Aumentar importância se há cadeia causal
        if experience.get('causal_chain'):
            importance += 0.1
        
        return min(importance, 1.0)
    
    async def _update_memory_indices(self, memory_trace: MemoryTrace) -> None:
        """Atualiza índices de busca para uma memória."""
        
        # Índice por tipo
        memory_type = memory_trace.memory_type.value
        if memory_type not in self.memory_index:
            self.memory_index[memory_type] = set()
        self.memory_index[memory_type].add(memory_trace.memory_id)
        
        # Índice por palavras-chave do conteúdo
        content_str = json.dumps(memory_trace.content, default=str).lower()
        words = content_str.split()
        
        for word in words:
            if len(word) > 3:  # Ignorar palavras muito curtas
                if word not in self.memory_index:
                    self.memory_index[word] = set()
                self.memory_index[word].add(memory_trace.memory_id)
    
    async def _link_related_experiences(self, memory_trace: MemoryTrace) -> None:
        """Vincula experiência a outras experiências relacionadas."""
        
        # Buscar experiências similares recentes
        similar_memories = []
        
        for other_id, other_memory in self.recent_memories.items():
            if other_id != memory_trace.memory_id:
                similarity = await self._calculate_memory_similarity(
                    memory_trace, other_memory
                )
                
                if similarity > self.similarity_threshold:
                    similar_memories.append((other_id, similarity))
        
        # Adicionar links bidirecionais
        for other_id, similarity in similar_memories:
            memory_trace.related_memories.add(other_id)
            memory_trace.similarity_scores[other_id] = similarity
            
            # Link reverso
            other_memory = self.recent_memories[other_id]
            other_memory.related_memories.add(memory_trace.memory_id)
            other_memory.similarity_scores[memory_trace.memory_id] = similarity
        
        # Atualizar no banco temporal
        await self.temporal_db.link_experiences(
            memory_trace, similarity_threshold=self.similarity_threshold
        )
    
    async def _calculate_memory_similarity(
        self, 
        memory1: MemoryTrace, 
        memory2: MemoryTrace
    ) -> float:
        """Calcula similaridade entre duas memórias."""
        
        similarity_scores = []
        
        # Similaridade semântica (baseada em features codificadas)
        if memory1.encoded_features is not None and memory2.encoded_features is not None:
            semantic_sim = np.dot(memory1.encoded_features, memory2.encoded_features)
            similarity_scores.append(semantic_sim)
        
        # Similaridade temporal
        time_diff = abs((memory1.timestamp - memory2.timestamp).total_seconds())
        temporal_sim = max(0, 1 - (time_diff / (24 * 3600)))  # Normalizar por 24h
        similarity_scores.append(temporal_sim * 0.3)
        
        # Similaridade de contexto estratégico
        if memory1.goal_context and memory2.goal_context:
            goal_overlap = len(set(memory1.goal_context.keys()) & set(memory2.goal_context.keys()))
            total_goals = len(set(memory1.goal_context.keys()) | set(memory2.goal_context.keys()))
            strategic_sim = goal_overlap / total_goals if total_goals > 0 else 0
            similarity_scores.append(strategic_sim * 0.4)
        
        # Similaridade causal
        if memory1.causal_chain and memory2.causal_chain:
            causal_overlap = len(set(memory1.causal_chain) & set(memory2.causal_chain))
            total_causal = len(set(memory1.causal_chain) | set(memory2.causal_chain))
            causal_sim = causal_overlap / total_causal if total_causal > 0 else 0
            similarity_scores.append(causal_sim * 0.3)
        
        # Retornar média ponderada
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _update_memory_statistics(self, memory_trace: MemoryTrace) -> None:
        """Atualiza estatísticas de memória."""
        
        self.memory_stats['total_memories'] += 1
        
        memory_type = memory_trace.memory_type.value
        if memory_type not in self.memory_stats['memories_by_type']:
            self.memory_stats['memories_by_type'][memory_type] = 0
        self.memory_stats['memories_by_type'][memory_type] += 1
    
    async def _combine_search_results(
        self,
        similar_experiences: List[Tuple[MemoryTrace, float]],
        pattern_matches: List[Tuple[MemoryTrace, float]],
        query: MemoryQuery
    ) -> List[Tuple[MemoryTrace, float]]:
        """Combina resultados de busca de diferentes fontes."""
        
        # Combinar resultados
        all_results = {}
        
        # Adicionar resultados de similaridade
        for memory, score in similar_experiences:
            all_results[memory.memory_id] = (memory, score * 0.7)  # Peso para similaridade
        
        # Adicionar resultados de padrões
        for memory, score in pattern_matches:
            if memory.memory_id in all_results:
                # Combinar scores
                existing_memory, existing_score = all_results[memory.memory_id]
                combined_score = existing_score + (score * 0.3)  # Peso para padrões
                all_results[memory.memory_id] = (existing_memory, combined_score)
            else:
                all_results[memory.memory_id] = (memory, score * 0.3)
        
        # Converter para lista e ordenar por score
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Aplicar filtros da query
        filtered_results = []
        for memory, score in combined_results:
            
            # Filtrar por tipo de memória
            if query.memory_types and memory.memory_type not in query.memory_types:
                continue
            
            # Filtrar por range temporal
            if query.time_range:
                start_time, end_time = query.time_range
                if not (start_time <= memory.timestamp <= end_time):
                    continue
            
            # Filtrar por nível de consolidação
            if memory.consolidation_level not in query.consolidation_levels:
                continue
            
            # Filtrar por similaridade mínima
            if score < query.min_similarity:
                continue
            
            filtered_results.append((memory, score))
            
            # Limitar número de resultados
            if len(filtered_results) >= query.max_results:
                break
        
        return filtered_results
    
    async def _analyze_temporal_patterns(self, experiences: List[MemoryTrace]) -> List[MemoryInsight]:
        """Analisa padrões temporais nas experiências."""
        
        insights = []
        
        if len(experiences) < 3:
            return insights
        
        # Analisar sequências temporais
        sorted_experiences = sorted(experiences, key=lambda x: x.timestamp)
        
        # Detectar padrões de timing
        time_intervals = []
        for i in range(1, len(sorted_experiences)):
            interval = (sorted_experiences[i].timestamp - sorted_experiences[i-1].timestamp).total_seconds()
            time_intervals.append(interval)
        
        if time_intervals:
            avg_interval = sum(time_intervals) / len(time_intervals)
            
            # Se há regularidade temporal
            if all(abs(interval - avg_interval) < avg_interval * 0.3 for interval in time_intervals):
                insights.append(MemoryInsight(
                    insight_type="temporal_regularity",
                    description=f"Regular pattern detected with average interval of {avg_interval/3600:.1f} hours",
                    confidence=0.8,
                    supporting_memories=[exp.memory_id for exp in sorted_experiences],
                    applicable_contexts=["scheduling", "planning"],
                    recommended_actions=["Consider leveraging this timing pattern for future planning"]
                ))
        
        return insights
    
    async def _analyze_causal_patterns(self, experiences: List[MemoryTrace]) -> List[MemoryInsight]:
        """Analisa padrões causais nas experiências."""
        
        insights = []
        
        # Coletar todas as cadeias causais
        all_causal_elements = []
        for exp in experiences:
            all_causal_elements.extend(exp.causal_chain)
        
        # Encontrar elementos causais frequentes
        causal_frequency = {}
        for element in all_causal_elements:
            causal_frequency[element] = causal_frequency.get(element, 0) + 1
        
        # Identificar elementos causais dominantes
        frequent_elements = [
            element for element, freq in causal_frequency.items() 
            if freq >= len(experiences) * 0.5  # Aparece em pelo menos 50% das experiências
        ]
        
        if frequent_elements:
            insights.append(MemoryInsight(
                insight_type="causal_pattern",
                description=f"Recurring causal elements identified: {', '.join(frequent_elements)}",
                confidence=0.7,
                supporting_memories=[exp.memory_id for exp in experiences],
                pattern_evidence={'frequent_elements': frequent_elements, 'frequencies': causal_frequency},
                applicable_contexts=["decision_making", "problem_solving"],
                recommended_actions=["Monitor these causal factors in future decisions"]
            ))
        
        return insights
    
    async def _analyze_success_patterns(self, experiences: List[MemoryTrace]) -> List[MemoryInsight]:
        """Analisa padrões de sucesso nas experiências."""
        
        insights = []
        
        # Separar experiências por nível de sucesso
        high_success = []
        low_success = []
        
        for exp in experiences:
            if exp.success_metrics:
                avg_success = sum(exp.success_metrics.values()) / len(exp.success_metrics)
                if avg_success > 0.7:
                    high_success.append(exp)
                elif avg_success < 0.4:
                    low_success.append(exp)
        
        # Analisar características de experiências de alto sucesso
        if len(high_success) >= 2:
            # Encontrar características comuns
            common_contexts = self._find_common_contexts(high_success)
            
            if common_contexts:
                insights.append(MemoryInsight(
                    insight_type="success_pattern",
                    description=f"High success associated with contexts: {', '.join(common_contexts)}",
                    confidence=0.8,
                    supporting_memories=[exp.memory_id for exp in high_success],
                    pattern_evidence={'common_contexts': common_contexts},
                    applicable_contexts=["strategy", "optimization"],
                    recommended_actions=["Prioritize these contexts in future activities"]
                ))
        
        return insights
    
    async def _analyze_strategic_patterns(
        self, 
        experiences: List[MemoryTrace], 
        context: Dict[str, Any]
    ) -> List[MemoryInsight]:
        """Analisa padrões estratégicos nas experiências."""
        
        insights = []
        
        # Analisar contextos de objetivos
        goal_patterns = {}
        for exp in experiences:
            for goal_key, goal_value in exp.goal_context.items():
                if goal_key not in goal_patterns:
                    goal_patterns[goal_key] = []
                goal_patterns[goal_key].append(goal_value)
        
        # Identificar objetivos recorrentes
        recurring_goals = {
            goal: values for goal, values in goal_patterns.items()
            if len(set(str(v) for v in values)) < len(values) * 0.7  # Há repetição
        }
        
        if recurring_goals:
            insights.append(MemoryInsight(
                insight_type="strategic_pattern",
                description=f"Recurring strategic goals identified: {list(recurring_goals.keys())}",
                confidence=0.7,
                supporting_memories=[exp.memory_id for exp in experiences],
                pattern_evidence={'recurring_goals': recurring_goals},
                applicable_contexts=["strategic_planning", "goal_setting"],
                recommended_actions=["Consider these recurring patterns in strategic planning"]
            ))
        
        return insights
    
    def _find_common_contexts(self, experiences: List[MemoryTrace]) -> List[str]:
        """Encontra contextos comuns entre experiências."""
        
        if not experiences:
            return []
        
        # Coletar todos os contextos
        all_contexts = set()
        for exp in experiences:
            # Contextos de objetivo
            all_contexts.update(exp.goal_context.keys())
            # Contextos de estratégia
            all_contexts.update(exp.strategy_context.keys())
        
        # Encontrar contextos que aparecem em múltiplas experiências
        common_contexts = []
        for context in all_contexts:
            count = sum(
                1 for exp in experiences
                if context in exp.goal_context or context in exp.strategy_context
            )
            if count >= len(experiences) * 0.6:  # Aparece em pelo menos 60%
                common_contexts.append(context)
        
        return common_contexts
    
    async def _rank_insights_by_relevance(
        self, 
        insights: List[MemoryInsight], 
        context: Dict[str, Any]
    ) -> List[MemoryInsight]:
        """Ranqueia insights por relevância ao contexto atual."""
        
        # Calcular score de relevância para cada insight
        scored_insights = []
        
        for insight in insights:
            relevance_score = insight.confidence  # Base score
            
            # Aumentar score se aplicável ao contexto atual
            current_contexts = set(context.keys())
            applicable_contexts = set(insight.applicable_contexts)
            
            context_overlap = len(current_contexts & applicable_contexts)
            if context_overlap > 0:
                relevance_score += context_overlap * 0.2
            
            # Aumentar score baseado no número de evidências
            evidence_boost = min(len(insight.supporting_memories) * 0.05, 0.3)
            relevance_score += evidence_boost
            
            scored_insights.append((insight, relevance_score))
        
        # Ordenar por score de relevância
        scored_insights.sort(key=lambda x: x[1], reverse=True)
        
        return [insight for insight, _ in scored_insights]
    
    async def _select_memories_for_consolidation(self) -> List[MemoryTrace]:
        """Seleciona memórias para consolidação."""
        
        # Critérios de seleção:
        # 1. Memórias antigas (mais de X tempo)
        # 2. Memórias importantes
        # 3. Memórias frequentemente acessadas
        
        consolidation_candidates = []
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # 24 horas
        
        for memory in self.recent_memories.values():
            # Verificar idade
            if memory.timestamp < cutoff_time:
                consolidation_candidates.append(memory)
            # Ou verificar importância alta
            elif memory.importance > 0.8:
                consolidation_candidates.append(memory)
            # Ou verificar acesso frequente
            elif memory.access_count > 5:
                consolidation_candidates.append(memory)
        
        return consolidation_candidates
    
    async def _update_consolidation_levels(self, consolidation_result: Dict[str, Any]) -> None:
        """Atualiza níveis de consolidação das memórias."""
        
        consolidated_memories = consolidation_result.get('consolidated_memories', [])
        
        for memory_id in consolidated_memories:
            if memory_id in self.recent_memories:
                memory = self.recent_memories[memory_id]
                
                # Promover nível de consolidação
                if memory.consolidation_level == ConsolidationLevel.WORKING:
                    memory.consolidation_level = ConsolidationLevel.SHORT_TERM
                elif memory.consolidation_level == ConsolidationLevel.SHORT_TERM:
                    memory.consolidation_level = ConsolidationLevel.LONG_TERM
                elif memory.consolidation_level == ConsolidationLevel.LONG_TERM:
                    if memory.importance > 0.9 or memory.access_count > 10:
                        memory.consolidation_level = ConsolidationLevel.PERMANENT
    
    async def _cleanup_recent_memories(self, consolidated_memories: List[MemoryTrace]) -> None:
        """Limpa memórias recentes que foram consolidadas."""
        
        for memory in consolidated_memories:
            # Remover memórias de nível SHORT_TERM ou superior do cache recente
            if memory.consolidation_level in [ConsolidationLevel.SHORT_TERM, ConsolidationLevel.LONG_TERM, ConsolidationLevel.PERMANENT]:
                if memory.memory_id in self.recent_memories:
                    del self.recent_memories[memory.memory_id]
                
                # Atualizar índices
                await self._remove_from_indices(memory)
    
    async def _remove_from_indices(self, memory: MemoryTrace) -> None:
        """Remove memória dos índices de busca."""
        
        # Remover de todos os índices que contêm esta memória
        indices_to_update = []
        
        for index_key, memory_set in self.memory_index.items():
            if memory.memory_id in memory_set:
                indices_to_update.append(index_key)
        
        for index_key in indices_to_update:
            self.memory_index[index_key].discard(memory.memory_id)
            # Remover índice vazio
            if not self.memory_index[index_key]:
                del self.memory_index[index_key]
    
    async def _get_memories_for_pattern_analysis(self) -> List[MemoryTrace]:
        """Obtém memórias para análise de padrões."""
        
        # Combinar memórias recentes com algumas memórias consolidadas
        analysis_memories = list(self.recent_memories.values())
        
        # Adicionar algumas memórias consolidadas importantes
        consolidated_memories = await self.temporal_db.get_consolidated_memories(
            limit=100, min_importance=0.7
        )
        analysis_memories.extend(consolidated_memories)
        
        return analysis_memories
    
    async def _validate_discovered_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Valida padrões descobertos."""
        
        validated_patterns = []
        
        for pattern in patterns:
            # Critérios de validação
            confidence = pattern.get('confidence', 0.0)
            support_count = pattern.get('support_count', 0)
            
            # Validar confiança mínima
            if confidence < 0.6:
                continue
            
            # Validar suporte mínimo
            if support_count < 3:
                continue
            
            # Validar novidade (não duplicar padrões existentes)
            is_novel = await self._check_pattern_novelty(pattern)
            if not is_novel:
                continue
            
            validated_patterns.append(pattern)
        
        return validated_patterns
    
    async def _check_pattern_novelty(self, pattern: Dict[str, Any]) -> bool:
        """Verifica se um padrão é novo."""
        
        # Implementação simplificada - verificar se padrão similar já existe
        pattern_signature = pattern.get('signature', '')
        
        existing_patterns = await self.temporal_db.get_existing_patterns()
        
        for existing_pattern in existing_patterns:
            existing_signature = existing_pattern.get('signature', '')
            
            # Calcular similaridade entre assinaturas
            similarity = self._calculate_pattern_similarity(pattern_signature, existing_signature)
            
            if similarity > 0.8:  # Muito similar a padrão existente
                return False
        
        return True
    
    def _calculate_pattern_similarity(self, signature1: str, signature2: str) -> float:
        """Calcula similaridade entre assinaturas de padrões."""
        
        # Implementação simplificada usando Jaccard similarity
        set1 = set(signature1.split())
        set2 = set(signature2.split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _store_discovered_pattern(self, pattern: Dict[str, Any]) -> None:
        """Armazena um padrão descoberto."""
        
        # Criar memória para o padrão
        pattern_memory = {
            'pattern': pattern,
            'timestamp': datetime.utcnow(),
            'discovery_method': 'automatic_pattern_detection',
            'confidence': pattern.get('confidence', 0.0)
        }
        
        # Armazenar como memória de padrão
        await self.store_experience(pattern_memory)
        
        # Também armazenar no banco de padrões
        await self.temporal_db.store_pattern(pattern)
    
    async def _calculate_average_retrieval_time(self) -> float:
        """Calcula tempo médio de recuperação."""
        # Implementação simplificada - retornar valor estimado
        return 0.05  # 50ms
    
    async def _calculate_memory_efficiency(self) -> float:
        """Calcula eficiência da memória."""
        
        # Eficiência baseada na razão hit/miss e uso de espaço
        total_memories = len(self.recent_memories)
        if total_memories == 0:
            return 1.0
        
        # Calcular uso eficiente baseado em acessos
        accessed_memories = sum(1 for memory in self.recent_memories.values() if memory.access_count > 0)
        access_efficiency = accessed_memories / total_memories
        
        # Calcular eficiência de consolidação
        consolidated_count = sum(
            1 for memory in self.recent_memories.values() 
            if memory.consolidation_level != ConsolidationLevel.WORKING
        )
        consolidation_efficiency = consolidated_count / total_memories
        
        # Eficiência combinada
        overall_efficiency = (access_efficiency + consolidation_efficiency) / 2
        
        return overall_efficiency
    
    async def _calculate_consolidation_effectiveness(self) -> float:
        """Calcula efetividade da consolidação."""
        
        # Baseado no número de consolidações vs memórias criadas
        total_memories = self.memory_stats['total_memories']
        consolidation_events = self.memory_stats['consolidation_events']
        
        if total_memories == 0:
            return 1.0
        
        # Razão ideal de consolidação (aproximadamente 1 consolidação por 100 memórias)
        ideal_ratio = total_memories / 100
        actual_ratio = consolidation_events
        
        effectiveness = min(actual_ratio / ideal_ratio, 1.0) if ideal_ratio > 0 else 1.0
        
        return effectiveness
    
    async def _assess_memory_system_health(self) -> Dict[str, Any]:
        """Avalia saúde do sistema de memória."""
        
        health_metrics = {
            'overall_health': 'good',
            'issues': [],
            'recommendations': []
        }
        
        # Verificar uso de memória
        if len(self.recent_memories) > self.max_recent_memories * 0.9:
            health_metrics['issues'].append('High memory usage')
            health_metrics['recommendations'].append('Increase consolidation frequency')
        
        # Verificar eficiência
        efficiency = await self._calculate_memory_efficiency()
        if efficiency < 0.6:
            health_metrics['issues'].append('Low memory efficiency')
            health_metrics['recommendations'].append('Review memory access patterns')
        
        # Verificar consolidação
        effectiveness = await self._calculate_consolidation_effectiveness()
        if effectiveness < 0.5:
            health_metrics['issues'].append('Ineffective consolidation')
            health_metrics['recommendations'].append('Adjust consolidation parameters')
        
        # Determinar saúde geral
        if len(health_metrics['issues']) == 0:
            health_metrics['overall_health'] = 'excellent'
        elif len(health_metrics['issues']) <= 2:
            health_metrics['overall_health'] = 'good'
        else:
            health_metrics['overall_health'] = 'needs_attention'
        
        return health_metrics
    
    async def _periodic_consolidation(self, interval_seconds: int) -> None:
        """Consolidação periódica de memórias."""
        
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                
                # Verificar se precisa consolidar
                if len(self.recent_memories) >= self.consolidation_threshold:
                    await self.consolidate_memories()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic consolidation: {e}", exc_info=True)
    
    async def _periodic_pattern_detection(self, interval_seconds: int) -> None:
        """Detecção periódica de padrões."""
        
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                
                # Executar detecção de padrões
                await self.discover_patterns()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic pattern detection: {e}", exc_info=True)
    
    async def _trigger_consolidation(self) -> None:
        """Dispara consolidação de memórias."""
        
        try:
            await self.consolidate_memories(force=True)
        except Exception as e:
            logger.error(f"Error in triggered consolidation: {e}", exc_info=True)
    
    async def shutdown(self) -> None:
        """Desliga o sistema de memória episódica."""
        
        logger.info("Shutting down Episodic Memory System")
        
        # Cancelar tarefas de background
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        
        if self._pattern_detection_task:
            self._pattern_detection_task.cancel()
            try:
                await self._pattern_detection_task
            except asyncio.CancelledError:
                pass
        
        # Consolidação final
        if self.recent_memories:
            await self.consolidate_memories(force=True)
        
        # Desligar componentes
        await self.temporal_db.shutdown()
        await self.experience_encoder.shutdown()
        await self.pattern_detector.shutdown()
        await self.consolidation_engine.shutdown()
        
        # Limpar caches
        self.recent_memories.clear()
        self.memory_index.clear()
        
        logger.info("Episodic Memory System shutdown complete")
