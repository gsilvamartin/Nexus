"""
Pattern Detection Engine

Implementa o motor de detecção de padrões para identificar padrões recorrentes
nas memórias episódicas e gerar insights.
"""

import asyncio
import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class PatternDetectionEngine:
    """Motor de detecção de padrões em memórias episódicas."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detected_patterns: List[Dict[str, Any]] = []
        self.pattern_stats = {
            'total_patterns_detected': 0,
            'pattern_types': defaultdict(int),
            'last_detection_run': None
        }
        
        logger.info("Pattern Detection Engine initialized")
    
    async def initialize(self) -> None:
        """Inicializa o motor de detecção de padrões."""
        logger.info("Pattern Detection Engine initialization complete")
    
    async def find_pattern_matches(
        self, 
        context: Dict[str, Any], 
        temporal_db: Any
    ) -> List[Tuple[Any, float]]:
        """Encontra memórias que correspondem a padrões conhecidos."""
        
        matches = []
        
        # Buscar padrões aplicáveis ao contexto
        applicable_patterns = await self._find_applicable_patterns(context)
        
        # Para cada padrão, buscar memórias correspondentes
        for pattern in applicable_patterns:
            pattern_matches = await self._find_memories_matching_pattern(
                pattern, temporal_db
            )
            matches.extend(pattern_matches)
        
        # Remover duplicatas e ordenar por score
        unique_matches = {}
        for memory, score in matches:
            if memory.memory_id not in unique_matches:
                unique_matches[memory.memory_id] = (memory, score)
            else:
                # Manter o maior score
                existing_score = unique_matches[memory.memory_id][1]
                if score > existing_score:
                    unique_matches[memory.memory_id] = (memory, score)
        
        result = list(unique_matches.values())
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    async def discover_patterns(self, memories: List[Any]) -> List[Dict[str, Any]]:
        """Descobre novos padrões nas memórias."""
        
        discovered_patterns = []
        
        # Detectar padrões temporais
        temporal_patterns = await self._detect_temporal_patterns(memories)
        discovered_patterns.extend(temporal_patterns)
        
        # Detectar padrões de sequência
        sequence_patterns = await self._detect_sequence_patterns(memories)
        discovered_patterns.extend(sequence_patterns)
        
        # Detectar padrões de contexto
        context_patterns = await self._detect_context_patterns(memories)
        discovered_patterns.extend(context_patterns)
        
        # Detectar padrões de sucesso/falha
        outcome_patterns = await self._detect_outcome_patterns(memories)
        discovered_patterns.extend(outcome_patterns)
        
        # Atualizar estatísticas
        self.pattern_stats['total_patterns_detected'] += len(discovered_patterns)
        self.pattern_stats['last_detection_run'] = datetime.utcnow()
        
        for pattern in discovered_patterns:
            pattern_type = pattern.get('pattern_type', 'unknown')
            self.pattern_stats['pattern_types'][pattern_type] += 1
        
        return discovered_patterns
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas de detecção de padrões."""
        
        return {
            'total_patterns_detected': self.pattern_stats['total_patterns_detected'],
            'pattern_types': dict(self.pattern_stats['pattern_types']),
            'last_detection_run': self.pattern_stats['last_detection_run'],
            'active_patterns': len(self.detected_patterns)
        }
    
    async def _find_applicable_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Encontra padrões aplicáveis ao contexto atual."""
        
        applicable = []
        
        for pattern in self.detected_patterns:
            # Verificar se o padrão é aplicável ao contexto
            pattern_context = pattern.get('context_requirements', {})
            
            # Calcular sobreposição de contexto
            context_overlap = self._calculate_context_overlap(context, pattern_context)
            
            if context_overlap > 0.5:  # Threshold de aplicabilidade
                applicable.append(pattern)
        
        return applicable
    
    async def _find_memories_matching_pattern(
        self, 
        pattern: Dict[str, Any], 
        temporal_db: Any
    ) -> List[Tuple[Any, float]]:
        """Encontra memórias que correspondem a um padrão específico."""
        
        matches = []
        
        # Obter critérios do padrão
        pattern_criteria = pattern.get('matching_criteria', {})
        
        # Buscar no banco temporal (implementação simplificada)
        # Em produção, isso seria uma query complexa no grafo temporal
        
        # Simular busca por padrão
        for node_id, node_data in temporal_db.nodes.items():
            match_score = await self._calculate_pattern_match_score(
                node_data, pattern_criteria
            )
            
            if match_score > 0.6:  # Threshold de correspondência
                # Criar objeto mock de memória
                mock_memory = type('MockMemory', (), {
                    'memory_id': node_id,
                    'content': node_data['content'],
                    'timestamp': node_data['timestamp']
                })()
                
                matches.append((mock_memory, match_score))
        
        return matches
    
    async def _detect_temporal_patterns(self, memories: List[Any]) -> List[Dict[str, Any]]:
        """Detecta padrões temporais nas memórias."""
        
        patterns = []
        
        if len(memories) < 3:
            return patterns
        
        # Ordenar memórias por timestamp
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        
        # Detectar intervalos regulares
        intervals = []
        for i in range(1, len(sorted_memories)):
            interval = (sorted_memories[i].timestamp - sorted_memories[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        # Verificar se há regularidade nos intervalos
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
            
            # Se baixa variância, há padrão temporal
            if variance < (avg_interval * 0.3) ** 2:
                patterns.append({
                    'pattern_type': 'temporal_regularity',
                    'pattern_id': f'temporal_{datetime.utcnow().timestamp()}',
                    'description': f'Regular temporal pattern with {avg_interval/3600:.1f}h intervals',
                    'confidence': 0.8,
                    'support_count': len(memories),
                    'signature': f'temporal_regular_{avg_interval}',
                    'matching_criteria': {
                        'temporal_interval': avg_interval,
                        'variance_threshold': variance
                    },
                    'context_requirements': {},
                    'discovered_at': datetime.utcnow()
                })
        
        return patterns
    
    async def _detect_sequence_patterns(self, memories: List[Any]) -> List[Dict[str, Any]]:
        """Detecta padrões de sequência nas memórias."""
        
        patterns = []
        
        # Analisar sequências de tipos de memória
        memory_types = [m.memory_type.value for m in memories]
        
        # Detectar subsequências frequentes
        frequent_sequences = await self._find_frequent_subsequences(memory_types)
        
        for sequence, frequency in frequent_sequences.items():
            if frequency >= 3:  # Mínimo 3 ocorrências
                patterns.append({
                    'pattern_type': 'sequence',
                    'pattern_id': f'sequence_{datetime.utcnow().timestamp()}',
                    'description': f'Frequent sequence: {" -> ".join(sequence)}',
                    'confidence': min(frequency / len(memories), 1.0),
                    'support_count': frequency,
                    'signature': f'sequence_{"_".join(sequence)}',
                    'matching_criteria': {
                        'sequence': list(sequence),
                        'min_frequency': frequency
                    },
                    'context_requirements': {},
                    'discovered_at': datetime.utcnow()
                })
        
        return patterns
    
    async def _detect_context_patterns(self, memories: List[Any]) -> List[Dict[str, Any]]:
        """Detecta padrões de contexto nas memórias."""
        
        patterns = []
        
        # Coletar todos os contextos
        all_contexts = []
        for memory in memories:
            context_keys = list(memory.goal_context.keys()) + list(memory.strategy_context.keys())
            all_contexts.extend(context_keys)
        
        # Encontrar contextos frequentes
        context_frequency = {}
        for context in all_contexts:
            context_frequency[context] = context_frequency.get(context, 0) + 1
        
        # Identificar contextos dominantes
        total_memories = len(memories)
        for context, frequency in context_frequency.items():
            if frequency >= total_memories * 0.6:  # Aparece em 60%+ das memórias
                patterns.append({
                    'pattern_type': 'context_dominance',
                    'pattern_id': f'context_{datetime.utcnow().timestamp()}',
                    'description': f'Dominant context: {context}',
                    'confidence': frequency / total_memories,
                    'support_count': frequency,
                    'signature': f'context_{context}',
                    'matching_criteria': {
                        'required_context': context,
                        'min_frequency': frequency
                    },
                    'context_requirements': {context: True},
                    'discovered_at': datetime.utcnow()
                })
        
        return patterns
    
    async def _detect_outcome_patterns(self, memories: List[Any]) -> List[Dict[str, Any]]:
        """Detecta padrões de resultado/sucesso nas memórias."""
        
        patterns = []
        
        # Separar memórias por nível de sucesso
        high_success_memories = []
        low_success_memories = []
        
        for memory in memories:
            if memory.success_metrics:
                avg_success = sum(memory.success_metrics.values()) / len(memory.success_metrics)
                if avg_success > 0.8:
                    high_success_memories.append(memory)
                elif avg_success < 0.3:
                    low_success_memories.append(memory)
        
        # Analisar características comuns de alto sucesso
        if len(high_success_memories) >= 3:
            common_characteristics = await self._find_common_characteristics(high_success_memories)
            
            if common_characteristics:
                patterns.append({
                    'pattern_type': 'success_pattern',
                    'pattern_id': f'success_{datetime.utcnow().timestamp()}',
                    'description': f'Success pattern with characteristics: {common_characteristics}',
                    'confidence': 0.8,
                    'support_count': len(high_success_memories),
                    'signature': f'success_{"_".join(common_characteristics)}',
                    'matching_criteria': {
                        'required_characteristics': common_characteristics,
                        'min_success_level': 0.8
                    },
                    'context_requirements': {},
                    'discovered_at': datetime.utcnow()
                })
        
        return patterns
    
    async def _find_frequent_subsequences(self, sequence: List[str]) -> Dict[Tuple[str, ...], int]:
        """Encontra subsequências frequentes em uma sequência."""
        
        subsequences = {}
        
        # Gerar subsequências de tamanho 2 e 3
        for length in [2, 3]:
            for i in range(len(sequence) - length + 1):
                subseq = tuple(sequence[i:i + length])
                subsequences[subseq] = subsequences.get(subseq, 0) + 1
        
        # Filtrar subsequências com frequência mínima
        frequent = {seq: freq for seq, freq in subsequences.items() if freq >= 2}
        
        return frequent
    
    async def _find_common_characteristics(self, memories: List[Any]) -> List[str]:
        """Encontra características comuns entre memórias."""
        
        characteristics = []
        
        # Analisar contextos comuns
        all_goal_contexts = set()
        all_strategy_contexts = set()
        
        for memory in memories:
            all_goal_contexts.update(memory.goal_context.keys())
            all_strategy_contexts.update(memory.strategy_context.keys())
        
        # Encontrar contextos que aparecem em múltiplas memórias
        for context in all_goal_contexts:
            count = sum(1 for memory in memories if context in memory.goal_context)
            if count >= len(memories) * 0.7:  # 70% das memórias
                characteristics.append(f'goal_{context}')
        
        for context in all_strategy_contexts:
            count = sum(1 for memory in memories if context in memory.strategy_context)
            if count >= len(memories) * 0.7:  # 70% das memórias
                characteristics.append(f'strategy_{context}')
        
        return characteristics
    
    def _calculate_context_overlap(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calcula sobreposição entre dois contextos."""
        
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)
        
        return intersection / union
    
    async def _calculate_pattern_match_score(
        self, 
        memory_data: Dict[str, Any], 
        pattern_criteria: Dict[str, Any]
    ) -> float:
        """Calcula score de correspondência entre memória e padrão."""
        
        score = 0.0
        
        # Verificar critérios temporais
        if 'temporal_interval' in pattern_criteria:
            # Implementação simplificada
            score += 0.3
        
        # Verificar critérios de sequência
        if 'sequence' in pattern_criteria:
            # Implementação simplificada
            score += 0.4
        
        # Verificar critérios de contexto
        if 'required_context' in pattern_criteria:
            memory_content = memory_data.get('content', {})
            required_context = pattern_criteria['required_context']
            
            if required_context in str(memory_content):
                score += 0.5
        
        # Verificar características de sucesso
        if 'required_characteristics' in pattern_criteria:
            # Implementação simplificada
            score += 0.3
        
        return min(score, 1.0)
    
    async def shutdown(self) -> None:
        """Desliga o motor de detecção de padrões."""
        logger.info("Pattern Detection Engine shutdown complete")
