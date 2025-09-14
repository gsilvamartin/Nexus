"""
Cognitive Types Module

Define tipos e estruturas de dados compartilhadas entre os módulos cognitivos.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ComplexityLevel(Enum):
    """Níveis de complexidade para análise meta-cognitiva."""
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    HIGHLY_COMPLEX = 5


class GoalStatus(Enum):
    """Status de um objetivo na hierarquia."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class GoalPriority(Enum):
    """Prioridade de um objetivo."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ComplexityAnalysis:
    """Análise de complexidade de um objetivo ou tarefa."""
    
    level: ComplexityLevel
    dimensions: Dict[str, float]  # Dimensões de complexidade (0-1)
    weights: Dict[str, float]     # Pesos para alocação de recursos
    confidence: float             # Confiança na análise (0-1)
    reasoning: List[str]          # Justificativas da análise
    
    # Métricas específicas
    cognitive_load: float = 0.0
    uncertainty: float = 0.0
    interdependencies: int = 0
    time_horizon: float = 0.0     # Em horas


@dataclass
class GoalNode:
    """
    Nó na hierarquia de objetivos.
    
    Representa um objetivo individual com suas dependências,
    métricas de progresso e contexto de execução.
    """
    
    # Identificação
    id: str
    name: str
    description: str
    
    # Hierarquia
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    
    # Status e prioridade
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    
    # Métricas
    progress: float = 0.0  # 0.0 - 1.0
    confidence: float = 0.0  # Confiança no sucesso
    complexity: Optional[ComplexityAnalysis] = None
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Recursos
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    required_resources: Dict[str, float] = field(default_factory=dict)
    
    # Contexto
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    success_criteria: List[str] = field(default_factory=list)
    
    # Métricas de aprendizado
    success_probability: float = 0.5
    risk_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Inicialização pós-criação."""
        if not self.id:
            # Gerar ID único baseado em timestamp e hash do nome
            import hashlib
            import time
            
            timestamp = str(int(time.time() * 1000))
            name_hash = hashlib.md5(self.name.encode()).hexdigest()[:8]
            self.id = f"goal_{timestamp}_{name_hash}"
    
    def add_child(self, child_id: str):
        """Adiciona um objetivo filho."""
        self.children_ids.add(child_id)
    
    def remove_child(self, child_id: str):
        """Remove um objetivo filho."""
        self.children_ids.discard(child_id)
    
    def is_leaf(self) -> bool:
        """Verifica se é um nó folha (sem filhos)."""
        return len(self.children_ids) == 0
    
    def is_root(self) -> bool:
        """Verifica se é um nó raiz (sem pai)."""
        return self.parent_id is None
    
    def can_start(self) -> bool:
        """Verifica se o objetivo pode ser iniciado."""
        return (
            self.status == GoalStatus.PENDING and
            len(self.dependencies) == 0  # Todas as dependências resolvidas
        )
    
    def update_progress(self, progress: float):
        """Atualiza o progresso do objetivo."""
        self.progress = max(0.0, min(1.0, progress))
        
        if self.progress >= 1.0 and self.status == GoalStatus.ACTIVE:
            self.status = GoalStatus.COMPLETED
            self.completed_at = datetime.now()
    
    def estimate_completion_time(self) -> Optional[float]:
        """Estima tempo para conclusão em horas."""
        if self.complexity is None:
            return None
        
        # Estimativa baseada na complexidade e progresso
        base_time = self.complexity.time_horizon
        remaining_work = 1.0 - self.progress
        
        return base_time * remaining_work
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'parent_id': self.parent_id,
            'children_ids': list(self.children_ids),
            'status': self.status.value,
            'priority': self.priority.value,
            'progress': self.progress,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'allocated_resources': self.allocated_resources,
            'required_resources': self.required_resources,
            'context': self.context,
            'dependencies': list(self.dependencies),
            'success_criteria': self.success_criteria,
            'success_probability': self.success_probability,
            'risk_factors': self.risk_factors
        }


@dataclass
class GoalTree:
    """
    Árvore hierárquica de objetivos.
    
    Gerencia a estrutura completa de objetivos e suas relações.
    """
    
    root_id: str
    goals: Dict[str, GoalNode] = field(default_factory=dict)
    
    def add_goal(self, goal: GoalNode, parent_id: Optional[str] = None):
        """Adiciona um objetivo à árvore."""
        if parent_id:
            if parent_id not in self.goals:
                raise ValueError(f"Objetivo pai {parent_id} não encontrado")
            
            goal.parent_id = parent_id
            self.goals[parent_id].add_child(goal.id)
        
        self.goals[goal.id] = goal
    
    def get_goal(self, goal_id: str) -> Optional[GoalNode]:
        """Obtém um objetivo pelo ID."""
        return self.goals.get(goal_id)
    
    def get_children(self, goal_id: str) -> List[GoalNode]:
        """Obtém os objetivos filhos de um objetivo."""
        goal = self.goals.get(goal_id)
        if not goal:
            return []
        
        return [self.goals[child_id] for child_id in goal.children_ids 
                if child_id in self.goals]
    
    def get_leaves(self) -> List[GoalNode]:
        """Obtém todos os objetivos folha (sem filhos)."""
        return [goal for goal in self.goals.values() if goal.is_leaf()]
    
    def get_active_goals(self) -> List[GoalNode]:
        """Obtém todos os objetivos ativos."""
        return [goal for goal in self.goals.values() 
                if goal.status == GoalStatus.ACTIVE]
    
    def calculate_overall_progress(self) -> float:
        """Calcula o progresso geral da árvore."""
        if not self.goals:
            return 0.0
        
        total_weight = 0.0
        weighted_progress = 0.0
        
        for goal in self.goals.values():
            weight = goal.priority.value
            total_weight += weight
            weighted_progress += goal.progress * weight
        
        return weighted_progress / total_weight if total_weight > 0 else 0.0


@dataclass
class ResourceAllocation:
    """Alocação de recursos para objetivos."""
    
    goal_id: str
    resource_type: str
    amount: float
    priority: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AttentionFocus:
    """Foco de atenção do sistema."""
    
    goal_id: str
    intensity: float  # 0.0 - 1.0
    duration: float   # Em segundos
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResourcePlan:
    """Plano de alocação de recursos do sistema."""
    
    computational_resources: Dict[str, Any] = field(default_factory=dict)
    model_allocations: Dict[str, Any] = field(default_factory=dict)
    time_allocations: Dict[str, Any] = field(default_factory=dict)
    priority_weights: Dict[str, float] = field(default_factory=dict)
    max_parallel_tasks: int = 10
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AttentionSchema:
    """Schema de distribuição de atenção do sistema."""
    
    focus_areas: Dict[str, float] = field(default_factory=dict)
    attention_span: float = 300.0  # Em segundos
    switching_cost: float = 0.1
    adaptive_threshold: float = 0.7
    interruption_tolerance: float = 0.3
    focus_stability: float = 0.8
    switching_efficiency: float = 0.9
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionStrategy:
    """Estratégia de execução para um conjunto de objetivos."""
    
    goal_tree: GoalTree
    resource_plan: ResourcePlan
    attention_schema: AttentionSchema
    complexity_analysis: ComplexityAnalysis
    estimated_duration: float  # Em horas
    confidence: float
    risk_assessment: Dict[str, float]
    
    # Campos avançados para otimização
    success_probability: float = 0.85
    contingency_plans: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_config: Optional[Dict[str, Any]] = None
    
    # Métricas de otimização
    pareto_efficiency_score: float = 0.8
    adaptation_capability: float = 0.7
    
    created_at: datetime = field(default_factory=datetime.now)
