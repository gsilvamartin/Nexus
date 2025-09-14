"""
Decentralized Swarm Coordinator

Coordenação descentralizada de enxames de agentes usando comunicação
stigmértica e algoritmos de consenso distribuído.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json
from enum import Enum

logger = logging.getLogger(__name__)


class SwarmRole(Enum):
    """Papéis dos agentes no enxame."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    OBSERVER = "observer"
    SPECIALIST = "specialist"


class CoordinationStyle(Enum):
    """Estilos de coordenação."""
    STIGMERGY = "stigmergy"  # Feromônios como formigas
    CONSENSUS = "consensus"  # Consenso distribuído
    HIERARCHICAL = "hierarchical"  # Hierarquia adaptativa
    EMERGENT = "emergent"  # Comportamento emergente


@dataclass
class Agent:
    """Representa um agente no enxame."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: SwarmRole = SwarmRole.WORKER
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    status: str = "idle"  # idle, working, waiting, error
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    stigmergy_trail: Dict[str, Any] = field(default_factory=dict)
    
    def is_alive(self, timeout_seconds: int = 30) -> bool:
        """Verifica se o agente está ativo."""
        return (datetime.utcnow() - self.last_heartbeat).total_seconds() < timeout_seconds


@dataclass
class SwarmTask:
    """Tarefa para o enxame."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    priority: int = 1
    deadline: Optional[datetime] = None
    assigned_agents: Set[str] = field(default_factory=set)
    status: str = "pending"  # pending, in_progress, completed, failed
    dependencies: List[str] = field(default_factory=list)
    stigmergy_markers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationPattern:
    """Padrão de coordenação emergente."""
    
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""
    participating_agents: Set[str] = field(default_factory=set)
    coordination_rules: Dict[str, Any] = field(default_factory=dict)
    effectiveness_score: float = 0.0
    emergence_timestamp: datetime = field(default_factory=datetime.utcnow)


class DecentralizedSwarmCoordinator:
    """
    Coordenador descentralizado de enxames de agentes.
    
    Implementa coordenação baseada em stigmergia (feromônios) e
    algoritmos de consenso distribuído para comportamento emergente.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.coordination_patterns: List[CoordinationPattern] = []
        self.stigmergy_environment: Dict[str, Any] = {}
        
        # Configurações
        self.heartbeat_interval = config.get('heartbeat_interval', 10)
        self.coordination_timeout = config.get('coordination_timeout', 60)
        self.emergence_threshold = config.get('emergence_threshold', 0.7)
        
        # Estado do sistema
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Decentralized Swarm Coordinator initialized")
    
    async def initialize(self) -> None:
        """Inicializa o coordenador de enxame."""
        
        logger.info("Initializing Decentralized Swarm Coordinator")
        
        # Inicializar ambiente stigmértico
        self.stigmergy_environment = {
            'pheromone_trails': {},
            'environmental_markers': {},
            'communication_hubs': {},
            'resource_maps': {}
        }
        
        # Iniciar monitoramento
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_swarm())
        
        logger.info("Decentralized Swarm Coordinator initialization complete")
    
    async def register_agent(
        self, 
        agent_id: str, 
        capabilities: List[str], 
        role: SwarmRole = SwarmRole.WORKER
    ) -> Agent:
        """Registra um novo agente no enxame."""
        
        agent = Agent(
            id=agent_id,
            role=role,
            capabilities=capabilities
        )
        
        self.agents[agent_id] = agent
        
        # Adicionar ao ambiente stigmértico
        self.stigmergy_environment['communication_hubs'][agent_id] = {
            'capabilities': capabilities,
            'last_activity': datetime.utcnow(),
            'message_queue': []
        }
        
        logger.info(f"Agent registered: {agent_id} with role {role.value}")
        return agent
    
    async def submit_task(self, task: SwarmTask) -> str:
        """Submete uma tarefa para o enxame."""
        
        self.tasks[task.id] = task
        
        # Criar marcadores stigmérticos para a tarefa
        await self._create_stigmergy_markers(task)
        
        # Iniciar coordenação para atribuir agentes
        await self._coordinate_task_assignment(task)
        
        logger.info(f"Task submitted: {task.id}")
        return task.id
    
    async def enable_coordination(
        self, 
        swarms: List[Dict[str, Any]], 
        coordination_style: str = 'stigmergy'
    ) -> List[CoordinationPattern]:
        """Habilita coordenação entre enxames."""
        
        coordination_patterns = []
        
        for swarm in swarms:
            pattern = await self._create_coordination_pattern(
                swarm, coordination_style
            )
            coordination_patterns.append(pattern)
            self.coordination_patterns.append(pattern)
        
        # Aplicar regras de coordenação
        await self._apply_coordination_rules(coordination_patterns)
        
        logger.info(f"Coordination enabled for {len(swarms)} swarms")
        return coordination_patterns
    
    async def _create_coordination_pattern(
        self, 
        swarm: Dict[str, Any], 
        coordination_style: str
    ) -> CoordinationPattern:
        """Cria padrão de coordenação para um enxame."""
        
        pattern = CoordinationPattern(
            pattern_type=coordination_style,
            participating_agents=set(swarm.get('agent_ids', [])),
            coordination_rules=self._get_coordination_rules(coordination_style)
        )
        
        return pattern
    
    def _get_coordination_rules(self, coordination_style: str) -> Dict[str, Any]:
        """Obtém regras de coordenação baseadas no estilo."""
        
        rules = {
            'stigmergy': {
                'pheromone_decay_rate': 0.1,
                'trail_strength_threshold': 0.5,
                'communication_range': 100,
                'update_frequency': 5
            },
            'consensus': {
                'consensus_threshold': 0.6,
                'voting_timeout': 30,
                'leader_election_interval': 60
            },
            'hierarchical': {
                'hierarchy_levels': 3,
                'command_chain_length': 5,
                'delegation_threshold': 0.8
            },
            'emergent': {
                'emergence_detection_window': 300,
                'pattern_evolution_rate': 0.05,
                'adaptation_threshold': 0.7
            }
        }
        
        return rules.get(coordination_style, rules['stigmergy'])
    
    async def _apply_coordination_rules(
        self, 
        patterns: List[CoordinationPattern]
    ) -> None:
        """Aplica regras de coordenação aos padrões."""
        
        for pattern in patterns:
            if pattern.pattern_type == 'stigmergy':
                await self._setup_stigmergic_coordination(pattern)
            elif pattern.pattern_type == 'consensus':
                await self._setup_consensus_coordination(pattern)
            elif pattern.pattern_type == 'hierarchical':
                await self._setup_hierarchical_coordination(pattern)
            elif pattern.pattern_type == 'emergent':
                await self._setup_emergent_coordination(pattern)
    
    async def _setup_stigmergic_coordination(self, pattern: CoordinationPattern) -> None:
        """Configura coordenação stigmértica."""
        
        # Criar trilhas de feromônios para cada agente
        for agent_id in pattern.participating_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.stigmergy_trail = {
                    'pheromone_strength': 1.0,
                    'last_update': datetime.utcnow(),
                    'trail_markers': []
                }
    
    async def _setup_consensus_coordination(self, pattern: CoordinationPattern) -> None:
        """Configura coordenação por consenso."""
        
        # Implementar eleição de líder
        leader = await self._elect_leader(pattern.participating_agents)
        pattern.coordination_rules['leader'] = leader
    
    async def _setup_hierarchical_coordination(self, pattern: CoordinationPattern) -> None:
        """Configura coordenação hierárquica."""
        
        # Criar hierarquia baseada em capacidades
        hierarchy = await self._create_hierarchy(pattern.participating_agents)
        pattern.coordination_rules['hierarchy'] = hierarchy
    
    async def _setup_emergent_coordination(self, pattern: CoordinationPattern) -> None:
        """Configura coordenação emergente."""
        
        # Configurar detecção de padrões emergentes
        pattern.coordination_rules['emergence_detector'] = {
            'enabled': True,
            'detection_interval': 10,
            'pattern_threshold': 0.7
        }
    
    async def _create_stigmergy_markers(self, task: SwarmTask) -> None:
        """Cria marcadores stigmérticos para uma tarefa."""
        
        # Criar trilha de feromônios para a tarefa
        task.stigmergy_markers = {
            'task_pheromone': {
                'strength': 1.0,
                'type': task.description,
                'requirements': task.requirements,
                'priority': task.priority
            },
            'environmental_cues': {
                'location': 'task_queue',
                'urgency': task.priority / 10.0,
                'complexity': len(task.requirements) / 10.0
            }
        }
        
        # Adicionar ao ambiente stigmértico
        self.stigmergy_environment['pheromone_trails'][task.id] = task.stigmergy_markers
    
    async def _coordinate_task_assignment(self, task: SwarmTask) -> None:
        """Coordena atribuição de tarefas usando stigmergia."""
        
        # Encontrar agentes adequados baseado em feromônios
        suitable_agents = await self._find_suitable_agents(task)
        
        # Atribuir tarefa aos agentes mais adequados
        for agent_id in suitable_agents[:3]:  # Atribuir a até 3 agentes
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.current_task = task.id
                agent.status = "working"
                task.assigned_agents.add(agent_id)
                task.status = "in_progress"
    
    async def _find_suitable_agents(self, task: SwarmTask) -> List[str]:
        """Encontra agentes adequados para uma tarefa usando stigmergia."""
        
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            if not agent.is_alive():
                continue
            
            # Calcular adequação baseada em feromônios e capacidades
            suitability_score = await self._calculate_agent_suitability(agent, task)
            
            if suitability_score > 0.5:
                suitable_agents.append((agent_id, suitability_score))
        
        # Ordenar por adequação
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        
        return [agent_id for agent_id, _ in suitable_agents]
    
    async def _calculate_agent_suitability(
        self, 
        agent: Agent, 
        task: SwarmTask
    ) -> float:
        """Calcula adequação de um agente para uma tarefa."""
        
        # Verificar capacidades
        capability_match = 0.0
        for requirement in task.requirements:
            if requirement in agent.capabilities:
                capability_match += 1.0
        
        capability_score = capability_match / len(task.requirements) if task.requirements else 1.0
        
        # Verificar trilha stigmértica
        stigmergy_score = 0.0
        if task.id in self.stigmergy_environment['pheromone_trails']:
            trail = self.stigmergy_environment['pheromone_trails'][task.id]
            stigmergy_score = trail['task_pheromone']['strength']
        
        # Verificar performance histórica
        performance_score = agent.performance_metrics.get('success_rate', 0.5)
        
        # Combinar scores
        suitability = (
            capability_score * 0.4 +
            stigmergy_score * 0.3 +
            performance_score * 0.3
        )
        
        return suitability
    
    async def _elect_leader(self, agent_ids: Set[str]) -> str:
        """Elege líder para coordenação por consenso."""
        
        # Implementação simplificada - eleger agente com melhor performance
        best_agent = None
        best_score = 0.0
        
        for agent_id in agent_ids:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                score = agent.performance_metrics.get('leadership_score', 0.5)
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
        
        return best_agent or list(agent_ids)[0]
    
    async def _create_hierarchy(self, agent_ids: Set[str]) -> Dict[str, Any]:
        """Cria hierarquia de agentes."""
        
        # Ordenar agentes por capacidades
        agents_by_capability = []
        for agent_id in agent_ids:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                capability_count = len(agent.capabilities)
                agents_by_capability.append((agent_id, capability_count))
        
        agents_by_capability.sort(key=lambda x: x[1], reverse=True)
        
        # Criar níveis hierárquicos
        hierarchy = {
            'level_1': [agents_by_capability[0][0]] if agents_by_capability else [],
            'level_2': [aid for aid, _ in agents_by_capability[1:3]],
            'level_3': [aid for aid, _ in agents_by_capability[3:]]
        }
        
        return hierarchy
    
    async def _monitor_swarm(self) -> None:
        """Monitora o enxame continuamente."""
        
        while self.is_running:
            try:
                # Atualizar trilhas stigmérticas
                await self._update_stigmergy_trails()
                
                # Detectar padrões emergentes
                await self._detect_emergent_patterns()
                
                # Verificar saúde dos agentes
                await self._check_agent_health()
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in swarm monitoring: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _update_stigmergy_trails(self) -> None:
        """Atualiza trilhas stigmérticas."""
        
        current_time = datetime.utcnow()
        
        # Decair feromônios
        for trail_id, trail in self.stigmergy_environment['pheromone_trails'].items():
            decay_rate = 0.1
            trail['task_pheromone']['strength'] *= (1 - decay_rate)
            
            # Remover trilhas muito fracas
            if trail['task_pheromone']['strength'] < 0.01:
                del self.stigmergy_environment['pheromone_trails'][trail_id]
    
    async def _detect_emergent_patterns(self) -> None:
        """Detecta padrões emergentes no enxame."""
        
        # Analisar interações entre agentes
        interactions = await self._analyze_agent_interactions()
        
        # Detectar padrões de coordenação
        patterns = await self._identify_coordination_patterns(interactions)
        
        # Atualizar padrões existentes
        for pattern in patterns:
            if pattern.effectiveness_score > self.emergence_threshold:
                self.coordination_patterns.append(pattern)
    
    async def _analyze_agent_interactions(self) -> Dict[str, Any]:
        """Analisa interações entre agentes."""
        
        interactions = {
            'communication_frequency': {},
            'task_collaboration': {},
            'resource_sharing': {}
        }
        
        # Implementação simplificada
        for agent_id, agent in self.agents.items():
            if agent.current_task:
                interactions['task_collaboration'][agent_id] = {
                    'task_id': agent.current_task,
                    'collaboration_level': 0.5
                }
        
        return interactions
    
    async def _identify_coordination_patterns(
        self, 
        interactions: Dict[str, Any]
    ) -> List[CoordinationPattern]:
        """Identifica padrões de coordenação emergentes."""
        
        patterns = []
        
        # Padrão de colaboração em tarefas
        if interactions['task_collaboration']:
            pattern = CoordinationPattern(
                pattern_type='task_collaboration',
                participating_agents=set(interactions['task_collaboration'].keys()),
                effectiveness_score=0.8
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _check_agent_health(self) -> None:
        """Verifica saúde dos agentes."""
        
        current_time = datetime.utcnow()
        
        for agent_id, agent in self.agents.items():
            if not agent.is_alive():
                logger.warning(f"Agent {agent_id} appears to be offline")
                
                # Reatribuir tarefas se necessário
                if agent.current_task:
                    task = self.tasks.get(agent.current_task)
                    if task:
                        task.assigned_agents.discard(agent_id)
                        await self._coordinate_task_assignment(task)
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Obtém status do enxame."""
        
        active_agents = sum(1 for agent in self.agents.values() if agent.is_alive())
        active_tasks = sum(1 for task in self.tasks.values() if task.status == "in_progress")
        
        return {
            'total_agents': len(self.agents),
            'active_agents': active_agents,
            'total_tasks': len(self.tasks),
            'active_tasks': active_tasks,
            'coordination_patterns': len(self.coordination_patterns),
            'stigmergy_environment': {
                'active_trails': len(self.stigmergy_environment['pheromone_trails']),
                'communication_hubs': len(self.stigmergy_environment['communication_hubs'])
            }
        }
    
    async def shutdown(self) -> None:
        """Desliga o coordenador de enxame."""
        
        logger.info("Shutting down Decentralized Swarm Coordinator")
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Decentralized Swarm Coordinator shutdown complete")
