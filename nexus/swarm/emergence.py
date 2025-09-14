"""
Emergent Behavior Detector

Detecta e analisa comportamentos emergentes em enxames de agentes
usando técnicas de análise de padrões e machine learning.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


@dataclass
class EmergentBehavior:
    """Representa um comportamento emergente detectado."""
    
    behavior_id: str
    behavior_type: str
    description: str
    participating_agents: List[str]
    emergence_timestamp: datetime
    confidence_score: float
    stability_score: float
    impact_score: float
    characteristics: Dict[str, Any] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BehaviorPattern:
    """Padrão de comportamento identificado."""
    
    pattern_id: str
    pattern_type: str
    frequency: float
    duration: float
    complexity: float
    effectiveness: float
    conditions: Dict[str, Any] = field(default_factory=dict)


class EmergentBehaviorDetector:
    """
    Detector de comportamentos emergentes.
    
    Analisa interações entre agentes para identificar padrões emergentes
    e comportamentos coletivos não programados explicitamente.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Configurações de detecção
        self.detection_window = config.get('detection_window', 300)  # 5 minutos
        self.min_participants = config.get('min_participants', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.stability_threshold = config.get('stability_threshold', 0.6)
        
        # Estado do detector
        self.detected_behaviors: Dict[str, EmergentBehavior] = {}
        self.behavior_patterns: List[BehaviorPattern] = []
        self.interaction_history: deque = deque(maxlen=10000)
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        
        # Métricas de análise
        self.analysis_metrics = {
            'total_interactions': 0,
            'patterns_detected': 0,
            'behaviors_emerged': 0,
            'false_positives': 0
        }
        
        logger.info("Emergent Behavior Detector initialized")
    
    async def initialize(self) -> None:
        """Inicializa o detector de comportamentos emergentes."""
        
        logger.info("Initializing Emergent Behavior Detector")
        
        # Inicializar algoritmos de detecção
        await self._initialize_detection_algorithms()
        
        logger.info("Emergent Behavior Detector initialization complete")
    
    async def monitor_emergence(
        self, 
        swarms: List[Dict[str, Any]], 
        coordination_patterns: List[Dict[str, Any]]
    ) -> List[EmergentBehavior]:
        """Monitora emergência de comportamentos em enxames."""
        
        logger.info("Monitoring emergent behaviors in swarms")
        
        # Coletar dados de interação
        interaction_data = await self._collect_interaction_data(swarms)
        
        # Analisar padrões de comportamento
        behavior_patterns = await self._analyze_behavior_patterns(interaction_data)
        
        # Detectar comportamentos emergentes
        emergent_behaviors = await self._detect_emergent_behaviors(
            behavior_patterns, coordination_patterns
        )
        
        # Validar e classificar comportamentos
        validated_behaviors = await self._validate_behaviors(emergent_behaviors)
        
        # Atualizar histórico
        for behavior in validated_behaviors:
            self.detected_behaviors[behavior.behavior_id] = behavior
            self.analysis_metrics['behaviors_emerged'] += 1
        
        logger.info(f"Detected {len(validated_behaviors)} emergent behaviors")
        return validated_behaviors
    
    async def _collect_interaction_data(self, swarms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coleta dados de interação dos enxames."""
        
        interaction_data = {
            'agent_interactions': defaultdict(list),
            'task_collaborations': defaultdict(list),
            'resource_sharing': defaultdict(list),
            'communication_patterns': defaultdict(list),
            'temporal_patterns': defaultdict(list)
        }
        
        for swarm in swarms:
            swarm_id = swarm.get('id', 'unknown')
            agents = swarm.get('agents', [])
            
            # Analisar interações entre agentes
            for i, agent1 in enumerate(agents):
                for j, agent2 in enumerate(agents[i+1:], i+1):
                    interaction = await self._analyze_agent_interaction(
                        agent1, agent2, swarm_id
                    )
                    if interaction:
                        interaction_data['agent_interactions'][swarm_id].append(interaction)
            
            # Analisar colaborações em tarefas
            tasks = swarm.get('tasks', [])
            for task in tasks:
                collaboration = await self._analyze_task_collaboration(task, agents)
                if collaboration:
                    interaction_data['task_collaborations'][swarm_id].append(collaboration)
            
            # Analisar padrões de comunicação
            communication = await self._analyze_communication_patterns(agents)
            if communication:
                interaction_data['communication_patterns'][swarm_id].append(communication)
        
        return interaction_data
    
    async def _analyze_agent_interaction(
        self, 
        agent1: Dict[str, Any], 
        agent2: Dict[str, Any], 
        swarm_id: str
    ) -> Optional[Dict[str, Any]]:
        """Analisa interação entre dois agentes."""
        
        # Verificar se há interação significativa
        interaction_strength = self._calculate_interaction_strength(agent1, agent2)
        
        if interaction_strength < 0.1:
            return None
        
        return {
            'agent1_id': agent1.get('id'),
            'agent2_id': agent2.get('id'),
            'swarm_id': swarm_id,
            'interaction_strength': interaction_strength,
            'interaction_type': self._classify_interaction_type(agent1, agent2),
            'timestamp': datetime.utcnow(),
            'context': {
                'agent1_capabilities': agent1.get('capabilities', []),
                'agent2_capabilities': agent2.get('capabilities', []),
                'shared_tasks': self._find_shared_tasks(agent1, agent2)
            }
        }
    
    def _calculate_interaction_strength(
        self, 
        agent1: Dict[str, Any], 
        agent2: Dict[str, Any]
    ) -> float:
        """Calcula força da interação entre agentes."""
        
        # Fatores que influenciam a força da interação
        shared_tasks = len(self._find_shared_tasks(agent1, agent2))
        capability_overlap = len(set(agent1.get('capabilities', [])) & 
                                set(agent2.get('capabilities', [])))
        communication_frequency = agent1.get('communication_frequency', {}).get(
            agent2.get('id'), 0
        )
        
        # Calcular score combinado
        strength = (
            shared_tasks * 0.4 +
            capability_overlap * 0.3 +
            communication_frequency * 0.3
        ) / 10.0  # Normalizar
        
        return min(strength, 1.0)
    
    def _classify_interaction_type(
        self, 
        agent1: Dict[str, Any], 
        agent2: Dict[str, Any]
    ) -> str:
        """Classifica tipo de interação entre agentes."""
        
        shared_tasks = self._find_shared_tasks(agent1, agent2)
        capability_overlap = set(agent1.get('capabilities', [])) & set(agent2.get('capabilities', []))
        
        if shared_tasks:
            return 'collaborative'
        elif capability_overlap:
            return 'competitive'
        else:
            return 'independent'
    
    def _find_shared_tasks(
        self, 
        agent1: Dict[str, Any], 
        agent2: Dict[str, Any]
    ) -> List[str]:
        """Encontra tarefas compartilhadas entre agentes."""
        
        tasks1 = set(agent1.get('current_tasks', []))
        tasks2 = set(agent2.get('current_tasks', []))
        
        return list(tasks1 & tasks2)
    
    async def _analyze_task_collaboration(
        self, 
        task: Dict[str, Any], 
        agents: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analisa colaboração em uma tarefa."""
        
        assigned_agents = task.get('assigned_agents', [])
        
        if len(assigned_agents) < 2:
            return None
        
        # Calcular métricas de colaboração
        collaboration_metrics = {
            'task_id': task.get('id'),
            'assigned_agents': assigned_agents,
            'collaboration_intensity': len(assigned_agents) / len(agents),
            'task_complexity': task.get('complexity', 0.5),
            'completion_rate': task.get('completion_rate', 0.0),
            'timestamp': datetime.utcnow()
        }
        
        return collaboration_metrics
    
    async def _analyze_communication_patterns(
        self, 
        agents: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analisa padrões de comunicação entre agentes."""
        
        communication_matrix = np.zeros((len(agents), len(agents)))
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    comm_freq = agent1.get('communication_frequency', {}).get(
                        agent2.get('id'), 0
                    )
                    communication_matrix[i][j] = comm_freq
        
        # Analisar padrões na matriz
        total_communication = np.sum(communication_matrix)
        if total_communication == 0:
            return None
        
        # Calcular métricas de rede
        network_metrics = {
            'total_communication': total_communication,
            'average_communication': np.mean(communication_matrix[communication_matrix > 0]),
            'communication_density': np.count_nonzero(communication_matrix) / (len(agents) * (len(agents) - 1)),
            'centrality_scores': self._calculate_centrality_scores(communication_matrix),
            'timestamp': datetime.utcnow()
        }
        
        return network_metrics
    
    def _calculate_centrality_scores(self, matrix: np.ndarray) -> List[float]:
        """Calcula scores de centralidade para cada agente."""
        
        # Centralidade de grau (degree centrality)
        degree_centrality = np.sum(matrix, axis=1)
        
        # Centralidade de proximidade (closeness centrality)
        # Implementação simplificada
        closeness_centrality = 1.0 / (np.sum(matrix, axis=1) + 1e-6)
        
        # Combinar scores
        centrality_scores = (degree_centrality + closeness_centrality) / 2
        
        return centrality_scores.tolist()
    
    async def _analyze_behavior_patterns(
        self, 
        interaction_data: Dict[str, Any]
    ) -> List[BehaviorPattern]:
        """Analisa padrões de comportamento nos dados de interação."""
        
        patterns = []
        
        # Analisar padrões de interação
        for swarm_id, interactions in interaction_data['agent_interactions'].items():
            if not interactions:
                continue
            
            # Detectar padrões temporais
            temporal_patterns = await self._detect_temporal_patterns(interactions)
            patterns.extend(temporal_patterns)
            
            # Detectar padrões de colaboração
            collaboration_patterns = await self._detect_collaboration_patterns(
                interaction_data['task_collaborations'].get(swarm_id, [])
            )
            patterns.extend(collaboration_patterns)
            
            # Detectar padrões de comunicação
            communication_patterns = await self._detect_communication_patterns(
                interaction_data['communication_patterns'].get(swarm_id, [])
            )
            patterns.extend(communication_patterns)
        
        return patterns
    
    async def _detect_temporal_patterns(
        self, 
        interactions: List[Dict[str, Any]]
    ) -> List[BehaviorPattern]:
        """Detecta padrões temporais nas interações."""
        
        patterns = []
        
        if len(interactions) < 3:
            return patterns
        
        # Agrupar interações por tipo
        interaction_types = defaultdict(list)
        for interaction in interactions:
            interaction_types[interaction['interaction_type']].append(interaction)
        
        # Analisar cada tipo de interação
        for interaction_type, type_interactions in interaction_types.items():
            if len(type_interactions) < 3:
                continue
            
            # Calcular frequência temporal
            timestamps = [i['timestamp'] for i in type_interactions]
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
            
            if not time_diffs:
                continue
            
            avg_interval = np.mean(time_diffs)
            frequency = 1.0 / avg_interval if avg_interval > 0 else 0
            
            # Detectar padrão se frequência for consistente
            if frequency > 0.1:  # Pelo menos uma interação a cada 10 segundos
                pattern = BehaviorPattern(
                    pattern_id=f"temporal_{interaction_type}_{len(patterns)}",
                    pattern_type='temporal',
                    frequency=frequency,
                    duration=avg_interval,
                    complexity=len(type_interactions) / 10.0,
                    effectiveness=min(frequency * 10, 1.0),
                    conditions={'interaction_type': interaction_type}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_collaboration_patterns(
        self, 
        collaborations: List[Dict[str, Any]]
    ) -> List[BehaviorPattern]:
        """Detecta padrões de colaboração."""
        
        patterns = []
        
        if not collaborations:
            return patterns
        
        # Analisar intensidade de colaboração
        collaboration_intensities = [c['collaboration_intensity'] for c in collaborations]
        avg_intensity = np.mean(collaboration_intensities)
        
        if avg_intensity > 0.3:  # Threshold para colaboração significativa
            pattern = BehaviorPattern(
                pattern_id=f"collaboration_{len(patterns)}",
                pattern_type='collaboration',
                frequency=avg_intensity,
                duration=1.0,  # Colaboração contínua
                complexity=avg_intensity,
                effectiveness=avg_intensity,
                conditions={'min_intensity': 0.3}
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_communication_patterns(
        self, 
        communications: List[Dict[str, Any]]
    ) -> List[BehaviorPattern]:
        """Detecta padrões de comunicação."""
        
        patterns = []
        
        if not communications:
            return patterns
        
        for comm in communications:
            density = comm.get('communication_density', 0)
            
            if density > 0.5:  # Alta densidade de comunicação
                pattern = BehaviorPattern(
                    pattern_id=f"communication_{len(patterns)}",
                    pattern_type='communication',
                    frequency=density,
                    duration=1.0,
                    complexity=density,
                    effectiveness=density,
                    conditions={'min_density': 0.5}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_emergent_behaviors(
        self, 
        behavior_patterns: List[BehaviorPattern],
        coordination_patterns: List[Dict[str, Any]]
    ) -> List[EmergentBehavior]:
        """Detecta comportamentos emergentes baseados nos padrões."""
        
        emergent_behaviors = []
        
        for pattern in behavior_patterns:
            # Verificar se o padrão é emergente
            if await self._is_emergent_pattern(pattern, coordination_patterns):
                behavior = await self._create_emergent_behavior(pattern)
                emergent_behaviors.append(behavior)
        
        return emergent_behaviors
    
    async def _is_emergent_pattern(
        self, 
        pattern: BehaviorPattern,
        coordination_patterns: List[Dict[str, Any]]
    ) -> bool:
        """Verifica se um padrão é emergente (não programado explicitamente)."""
        
        # Verificar se o padrão não está nos padrões de coordenação conhecidos
        for coord_pattern in coordination_patterns:
            if pattern.pattern_type == coord_pattern.get('type'):
                return False
        
        # Verificar critérios de emergência
        is_complex = pattern.complexity > 0.5
        is_effective = pattern.effectiveness > 0.6
        is_frequent = pattern.frequency > 0.1
        
        return is_complex and is_effective and is_frequent
    
    async def _create_emergent_behavior(self, pattern: BehaviorPattern) -> EmergentBehavior:
        """Cria comportamento emergente a partir de um padrão."""
        
        behavior = EmergentBehavior(
            behavior_id=f"emergent_{pattern.pattern_id}",
            behavior_type=pattern.pattern_type,
            description=f"Emergent {pattern.pattern_type} behavior",
            participating_agents=[],  # Será preenchido posteriormente
            emergence_timestamp=datetime.utcnow(),
            confidence_score=pattern.effectiveness,
            stability_score=min(pattern.frequency * 10, 1.0),
            impact_score=pattern.complexity,
            characteristics={
                'frequency': pattern.frequency,
                'duration': pattern.duration,
                'complexity': pattern.complexity,
                'effectiveness': pattern.effectiveness
            }
        )
        
        return behavior
    
    async def _validate_behaviors(
        self, 
        behaviors: List[EmergentBehavior]
    ) -> List[EmergentBehavior]:
        """Valida comportamentos emergentes detectados."""
        
        validated_behaviors = []
        
        for behavior in behaviors:
            # Verificar critérios de validação
            if (behavior.confidence_score >= self.confidence_threshold and
                behavior.stability_score >= self.stability_threshold):
                
                # Adicionar ao histórico de evolução
                behavior.evolution_history.append({
                    'timestamp': datetime.utcnow(),
                    'confidence': behavior.confidence_score,
                    'stability': behavior.stability_score,
                    'impact': behavior.impact_score
                })
                
                validated_behaviors.append(behavior)
            else:
                self.analysis_metrics['false_positives'] += 1
        
        return validated_behaviors
    
    async def _initialize_detection_algorithms(self) -> None:
        """Inicializa algoritmos de detecção."""
        
        # Configurar algoritmos de análise de padrões
        self.pattern_analyzer = {
            'temporal_analysis': True,
            'collaboration_analysis': True,
            'communication_analysis': True,
            'network_analysis': True
        }
        
        logger.info("Detection algorithms initialized")
    
    async def get_emergence_metrics(self) -> Dict[str, Any]:
        """Obtém métricas de emergência."""
        
        return {
            'total_behaviors': len(self.detected_behaviors),
            'active_behaviors': len([b for b in self.detected_behaviors.values() 
                                   if b.stability_score > 0.5]),
            'analysis_metrics': self.analysis_metrics,
            'behavior_types': list(set(b.behavior_type for b in self.detected_behaviors.values())),
            'average_confidence': np.mean([b.confidence_score for b in self.detected_behaviors.values()]) 
                                if self.detected_behaviors else 0.0
        }
    
    async def shutdown(self) -> None:
        """Desliga o detector de comportamentos emergentes."""
        
        logger.info("Shutting down Emergent Behavior Detector")
        
        # Salvar estado se necessário
        await self._save_detection_state()
        
        logger.info("Emergent Behavior Detector shutdown complete")
    
    async def _save_detection_state(self) -> None:
        """Salva estado do detector."""
        
        # Implementação simplificada
        logger.info("Detection state saved")
