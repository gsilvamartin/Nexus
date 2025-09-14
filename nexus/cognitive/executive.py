"""
Executive Function Module

Implementa a função executiva do NEXUS, responsável por planejamento estratégico,
meta-cognição, controle de atenção e hierarquia adaptativa de objetivos.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from nexus.cognitive.types import (
    ComplexityLevel, ComplexityAnalysis, GoalNode, GoalTree,
    ExecutionStrategy, ResourceAllocation, AttentionFocus,
    GoalStatus, GoalPriority, ResourcePlan, AttentionSchema
)
from nexus.cognitive.metacognition import MetaCognitiveController
from nexus.cognitive.attention import AttentionController
from nexus.cognitive.planning import StrategicCognitionEngine
from nexus.cognitive.goals import AdaptiveGoalHierarchy

logger = logging.getLogger(__name__)


class ExecutiveFunction:
    """
    Função Executiva do NEXUS.
    
    Responsável por orquestração do sistema, planejamento estratégico,
    meta-cognição e controle de atenção.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa a Função Executiva.
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        
        # Componentes especializados
        self.strategic_planner = StrategicCognitionEngine(
            self.config.get('strategic', {})
        )
        self.meta_cognition = MetaCognitiveController(
            self.config.get('metacognition', {})
        )
        self.attention_controller = AttentionController(
            self.config.get('attention', {})
        )
        self.goal_hierarchy = AdaptiveGoalHierarchy(
            self.config.get('goals', {})
        )
        
        # Estado interno
        self.current_strategy: Optional[ExecutionStrategy] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        logger.info("Executive Function initialized")
    
    async def orchestrate_system(self, objective: Any) -> ExecutionStrategy:
        """
        Orquestra o sistema completo para um objetivo usando raciocínio estratégico avançado.
        
        Args:
            objective: Objetivo de desenvolvimento
            
        Returns:
            Estratégia de execução completa com otimização multi-dimensional
        """
        logger.info("Starting advanced system orchestration")
        
        # Meta-cognitive assessment with uncertainty quantification
        complexity_analysis = await self.assess_complexity(objective)
        
        # Multi-dimensional goal decomposition with dependency analysis
        goal_tree = await self.decompose_goals(objective, complexity_analysis)
        
        # Strategic resource allocation with predictive modeling
        system_state = await self._get_system_state()
        resource_plan = await self.allocate_resources(goal_tree, system_state)
        
        # Dynamic attention management with adaptive thresholds
        attention_schema = await self.distribute_attention(
            resource_plan, complexity_analysis
        )
        
        # Risk assessment and mitigation planning
        risk_assessment = await self._assess_execution_risks(
            goal_tree, resource_plan, complexity_analysis
        )
        
        # Contingency planning for high-uncertainty scenarios
        contingency_plans = await self._generate_contingency_plans(
            goal_tree, risk_assessment
        )
        
        # Create enhanced execution strategy
        strategy = ExecutionStrategy(
            goal_tree=goal_tree,
            resource_plan=resource_plan,
            attention_schema=attention_schema,
            complexity_analysis=complexity_analysis,
            estimated_duration=await self._estimate_execution_duration(goal_tree),
            confidence=await self._calculate_strategy_confidence(complexity_analysis, risk_assessment),
            risk_assessment=risk_assessment
        )
        
        # Multi-objective optimization with Pareto efficiency
        optimized_strategy = await self._optimize_strategy_pareto(strategy, contingency_plans)
        
        # Adaptive monitoring and real-time adjustment setup
        await self._setup_adaptive_monitoring(optimized_strategy)
        
        self.current_strategy = optimized_strategy
        
        logger.info("Advanced system orchestration completed")
        return optimized_strategy
    
    async def assess_complexity(self, objective: Any) -> ComplexityAnalysis:
        """
        Avalia a complexidade de um objetivo usando meta-cognição.
        
        Args:
            objective: Objetivo a ser analisado
            
        Returns:
            Análise detalhada de complexidade
        """
        logger.debug("Assessing objective complexity")
        
        # Análise multi-dimensional
        dimensions = await self._analyze_complexity_dimensions(objective)
        
        # Calcular nível de complexidade
        complexity_score = np.mean(list(dimensions.values()))
        level = self._score_to_complexity_level(complexity_score)
        
        # Calcular pesos para alocação de recursos
        weights = await self._calculate_resource_weights(dimensions)
        
        # Meta-cognitive reasoning
        reasoning = await self.meta_cognition.analyze_complexity_reasoning(
            objective, dimensions
        )
        
        # Calcular confiança na análise
        confidence = await self._calculate_analysis_confidence(dimensions, reasoning)
        
        analysis = ComplexityAnalysis(
            level=level,
            dimensions=dimensions,
            weights=weights,
            confidence=confidence,
            reasoning=reasoning,
            cognitive_load=dimensions.get('cognitive_load', 0.5),
            uncertainty=dimensions.get('uncertainty', 0.3),
            interdependencies=int(dimensions.get('interdependencies', 0) * 10),
            time_horizon=dimensions.get('time_horizon', 1.0) * 24  # Convert to hours
        )
        
        logger.debug(f"Complexity analysis completed: {level.name}")
        return analysis
    
    async def decompose_goals(
        self, 
        objective: Any, 
        complexity_analysis: ComplexityAnalysis
    ) -> Dict[str, GoalNode]:
        """
        Decompõe objetivos de forma adaptativa.
        
        Args:
            objective: Objetivo principal
            complexity_analysis: Análise de complexidade
            
        Returns:
            Árvore de objetivos hierárquica
        """
        logger.debug("Decomposing goals adaptively")
        
        # Use goal hierarchy system for decomposition
        goal_tree = await self.goal_hierarchy.decompose_objective(
            objective, complexity_analysis
        )
        
        # Optimize goal structure
        optimized_tree = await self._optimize_goal_structure(goal_tree)
        
        logger.debug(f"Goal decomposition completed: {len(optimized_tree)} goals")
        return optimized_tree
    
    async def allocate_resources(
        self, 
        goal_tree: Dict[str, GoalNode], 
        system_state: Dict[str, Any]
    ) -> ResourcePlan:
        """
        Aloca recursos estrategicamente.
        
        Args:
            goal_tree: Árvore de objetivos
            system_state: Estado atual do sistema
            
        Returns:
            Plano de alocação de recursos
        """
        logger.debug("Allocating resources strategically")
        
        # Analyze resource requirements
        resource_requirements = await self._analyze_resource_requirements(goal_tree)
        
        # Available resources from system state
        available_resources = system_state.get('available_resources', {})
        
        # Strategic allocation using planning engine
        allocation = await self.strategic_planner.allocate_resources(
            resource_requirements, available_resources
        )
        
        # Create resource plan
        resource_plan = ResourcePlan(
            computational_resources=allocation.get('computational', {}),
            model_allocations=allocation.get('models', {}),
            time_allocations=allocation.get('time', {}),
            priority_weights=allocation.get('priorities', {}),
            max_parallel_tasks=self.config.get('max_parallel_tasks', 10)
        )
        
        # Calculate optimization metrics
        resource_plan.efficiency_score = await self._calculate_efficiency_score(
            resource_plan, goal_tree
        )
        resource_plan.load_balance_score = await self._calculate_load_balance_score(
            resource_plan
        )
        
        logger.debug("Resource allocation completed")
        return resource_plan
    
    async def distribute_attention(
        self, 
        resource_plan: ResourcePlan, 
        complexity_analysis: ComplexityAnalysis
    ) -> AttentionSchema:
        """
        Distribui atenção entre córtices.
        
        Args:
            resource_plan: Plano de recursos
            complexity_analysis: Análise de complexidade
            
        Returns:
            Schema de distribuição de atenção
        """
        logger.debug("Distributing attention across cortices")
        
        # Use attention controller for distribution
        attention_distribution = await self.attention_controller.distribute_attention(
            resource_plan.priority_weights, complexity_analysis.weights
        )
        
        # Calculate attention parameters
        attention_span = await self._calculate_attention_span(complexity_analysis)
        switching_cost = await self._calculate_switching_cost(resource_plan)
        
        attention_schema = AttentionSchema(
            focus_areas=attention_distribution,
            attention_span=attention_span,
            switching_cost=switching_cost,
            adaptive_threshold=self.config.get('adaptive_threshold', 0.7),
            interruption_tolerance=self.config.get('interruption_tolerance', 0.3)
        )
        
        # Calculate performance metrics
        attention_schema.focus_stability = await self._calculate_focus_stability(
            attention_schema
        )
        attention_schema.context_switching_efficiency = await self._calculate_switching_efficiency(
            attention_schema
        )
        
        logger.debug("Attention distribution completed")
        return attention_schema
    
    async def _analyze_complexity_dimensions(self, objective: Any) -> Dict[str, float]:
        """Analisa dimensões de complexidade."""
        
        # Extract features from objective
        if hasattr(objective, 'description'):
            text = objective.description
        else:
            text = str(objective)
        
        # Analyze various complexity dimensions
        dimensions = {
            'cognitive_load': await self._analyze_cognitive_load(text),
            'uncertainty': await self._analyze_uncertainty(objective),
            'interdependencies': await self._analyze_interdependencies(objective),
            'time_horizon': await self._analyze_time_horizon(objective),
            'technical_complexity': await self._analyze_technical_complexity(text),
            'domain_complexity': await self._analyze_domain_complexity(text),
            'integration_complexity': await self._analyze_integration_complexity(objective)
        }
        
        return dimensions
    
    async def _analyze_cognitive_load(self, text: str) -> float:
        """Analisa carga cognitiva baseada no texto."""
        # Simplified heuristic - in production, use NLP models
        word_count = len(text.split())
        complexity_keywords = ['complex', 'integrate', 'optimize', 'scale', 'distributed']
        keyword_count = sum(1 for word in complexity_keywords if word in text.lower())
        
        base_load = min(word_count / 100, 1.0)  # Normalize by word count
        keyword_boost = min(keyword_count * 0.2, 0.6)  # Boost for complexity keywords
        
        return min(base_load + keyword_boost, 1.0)
    
    async def _analyze_uncertainty(self, objective: Any) -> float:
        """Analisa incerteza do objetivo."""
        # Check for uncertainty indicators
        uncertainty_indicators = []
        
        if hasattr(objective, 'requirements'):
            if not objective.requirements:
                uncertainty_indicators.append('no_requirements')
        
        if hasattr(objective, 'constraints'):
            if not objective.constraints:
                uncertainty_indicators.append('no_constraints')
        
        if hasattr(objective, 'success_criteria'):
            if not objective.success_criteria:
                uncertainty_indicators.append('no_success_criteria')
        
        # Base uncertainty
        base_uncertainty = 0.3
        uncertainty_penalty = len(uncertainty_indicators) * 0.2
        
        return min(base_uncertainty + uncertainty_penalty, 1.0)
    
    async def _analyze_interdependencies(self, objective: Any) -> float:
        """Analisa interdependências."""
        # Simplified analysis - count potential integration points
        if hasattr(objective, 'description'):
            text = objective.description.lower()
            integration_terms = ['api', 'database', 'service', 'system', 'integration']
            count = sum(1 for term in integration_terms if term in text)
            return min(count / 10, 1.0)
        
        return 0.3  # Default moderate interdependency
    
    async def _analyze_time_horizon(self, objective: Any) -> float:
        """Analisa horizonte temporal."""
        if hasattr(objective, 'deadline') and objective.deadline:
            # Calculate time until deadline
            time_delta = objective.deadline - datetime.utcnow()
            days = time_delta.days
            
            if days < 1:
                return 1.0  # Very urgent
            elif days < 7:
                return 0.8  # Urgent
            elif days < 30:
                return 0.5  # Moderate
            else:
                return 0.2  # Long-term
        
        return 0.5  # Default moderate time horizon
    
    async def _analyze_technical_complexity(self, text: str) -> float:
        """Analisa complexidade técnica."""
        technical_terms = [
            'algorithm', 'optimization', 'machine learning', 'ai', 'distributed',
            'microservices', 'kubernetes', 'docker', 'cloud', 'scalability'
        ]
        
        text_lower = text.lower()
        count = sum(1 for term in technical_terms if term in text_lower)
        
        return min(count / 5, 1.0)
    
    async def _analyze_domain_complexity(self, text: str) -> float:
        """Analisa complexidade do domínio."""
        domain_indicators = [
            'finance', 'healthcare', 'security', 'compliance', 'regulation',
            'real-time', 'high-availability', 'performance', 'enterprise'
        ]
        
        text_lower = text.lower()
        count = sum(1 for indicator in domain_indicators if indicator in text_lower)
        
        return min(count / 3, 1.0)
    
    async def _analyze_integration_complexity(self, objective: Any) -> float:
        """Analisa complexidade de integração."""
        # Check for integration requirements
        if hasattr(objective, 'description'):
            text = objective.description.lower()
            if any(term in text for term in ['integrate', 'connect', 'api', 'third-party']):
                return 0.7
        
        return 0.3  # Default low integration complexity
    
    def _score_to_complexity_level(self, score: float) -> ComplexityLevel:
        """Converte score para nível de complexidade."""
        if score < 0.2:
            return ComplexityLevel.TRIVIAL
        elif score < 0.4:
            return ComplexityLevel.SIMPLE
        elif score < 0.6:
            return ComplexityLevel.MODERATE
        elif score < 0.8:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.HIGHLY_COMPLEX
    
    async def _calculate_resource_weights(self, dimensions: Dict[str, float]) -> Dict[str, float]:
        """Calcula pesos para alocação de recursos."""
        total = sum(dimensions.values())
        if total == 0:
            return {key: 1.0 / len(dimensions) for key in dimensions}
        
        return {key: value / total for key, value in dimensions.items()}
    
    async def _calculate_analysis_confidence(
        self, 
        dimensions: Dict[str, float], 
        reasoning: List[str]
    ) -> float:
        """Calcula confiança na análise."""
        # Base confidence
        base_confidence = 0.7
        
        # Boost confidence with more reasoning
        reasoning_boost = min(len(reasoning) * 0.05, 0.2)
        
        # Reduce confidence for high uncertainty
        uncertainty_penalty = dimensions.get('uncertainty', 0) * 0.3
        
        confidence = base_confidence + reasoning_boost - uncertainty_penalty
        return max(min(confidence, 1.0), 0.1)
    
    async def _get_system_state(self) -> Dict[str, Any]:
        """Obtém estado atual do sistema."""
        # Placeholder - in production, get from system monitor
        return {
            'available_resources': {
                'cpu': 0.7,
                'memory': 0.6,
                'gpu': 0.8
            },
            'active_models': ['gpt-4', 'claude-3'],
            'load': 0.4
        }
    
    async def _optimize_strategy(self, strategy: ExecutionStrategy) -> ExecutionStrategy:
        """Otimiza estratégia de execução."""
        # Placeholder for strategy optimization
        strategy.estimated_completion_time = sum(
            goal.complexity.time_horizon if goal.complexity else 1.0 
            for goal in strategy.goal_tree.values()
        )
        strategy.success_probability = 0.85  # Default high probability
        
        return strategy
    
    async def _optimize_goal_structure(self, goal_tree: Dict[str, GoalNode]) -> Dict[str, GoalNode]:
        """Otimiza estrutura de objetivos."""
        # Placeholder for goal optimization
        return goal_tree
    
    async def _analyze_resource_requirements(self, goal_tree: Dict[str, GoalNode]) -> Dict[str, Any]:
        """Analisa requisitos de recursos."""
        total_effort = sum(
            goal.complexity.time_horizon if goal.complexity else 1.0 
            for goal in goal_tree.values()
        )
        
        return {
            'computational': {'cpu': total_effort * 0.1, 'memory': total_effort * 0.05},
            'models': ['gpt-4', 'claude-3'],
            'time': total_effort,
            'priorities': {goal.id: goal.priority.value for goal in goal_tree.values()}
        }
    
    async def _calculate_efficiency_score(
        self, 
        resource_plan: ResourcePlan, 
        goal_tree: Dict[str, GoalNode]
    ) -> float:
        """Calcula score de eficiência."""
        # Simplified efficiency calculation
        return 0.8  # Default good efficiency
    
    async def _calculate_load_balance_score(self, resource_plan: ResourcePlan) -> float:
        """Calcula score de balanceamento de carga."""
        # Simplified load balance calculation
        return 0.75  # Default good balance
    
    async def _calculate_attention_span(self, complexity_analysis: ComplexityAnalysis) -> float:
        """Calcula duração da atenção."""
        base_span = 300.0  # 5 minutes base
        complexity_factor = complexity_analysis.level.value / 5.0
        
        return base_span * (1 + complexity_factor)
    
    async def _calculate_switching_cost(self, resource_plan: ResourcePlan) -> float:
        """Calcula custo de mudança de contexto."""
        # Higher switching cost for more complex resource plans
        return min(len(resource_plan.model_allocations) * 0.1, 1.0)
    
    async def _calculate_focus_stability(self, attention_schema: AttentionSchema) -> float:
        """Calcula estabilidade do foco."""
        # Simplified calculation
        return 0.8  # Default good stability
    
    async def _calculate_switching_efficiency(self, attention_schema: AttentionSchema) -> float:
        """Calcula eficiência de mudança de contexto."""
        # Inverse relationship with switching cost
        return max(1.0 - attention_schema.switching_cost, 0.1)
    
    async def _assess_execution_risks(
        self, 
        goal_tree: Dict[str, GoalNode], 
        resource_plan: ResourcePlan, 
        complexity_analysis: ComplexityAnalysis
    ) -> Dict[str, float]:
        """
        Avalia riscos de execução usando análise multi-dimensional.
        
        Args:
            goal_tree: Árvore de objetivos
            resource_plan: Plano de recursos
            complexity_analysis: Análise de complexidade
            
        Returns:
            Mapeamento de riscos e suas probabilidades
        """
        risk_assessment = {}
        
        # Risco de complexidade
        if complexity_analysis.level.value >= 4:  # Complex or Highly Complex
            risk_assessment['complexity_overload'] = min(complexity_analysis.level.value / 5.0, 0.9)
        
        # Risco de recursos insuficientes
        total_resource_demand = sum(
            goal.complexity.time_horizon if goal.complexity else 1.0 
            for goal in goal_tree.values()
        )
        resource_availability = resource_plan.computational_resources.get('cpu', 0.5)
        
        if total_resource_demand > resource_availability * 10:  # 10x threshold
            risk_assessment['resource_insufficiency'] = 0.8
        
        # Risco de interdependências
        dependency_complexity = complexity_analysis.interdependencies / 10.0
        if dependency_complexity > 0.5:
            risk_assessment['dependency_failure'] = dependency_complexity
        
        # Risco temporal
        if complexity_analysis.time_horizon > 48:  # More than 48 hours
            risk_assessment['schedule_overrun'] = min(complexity_analysis.time_horizon / 168, 0.9)  # Normalize by week
        
        # Risco de incerteza
        if complexity_analysis.uncertainty > 0.7:
            risk_assessment['uncertainty_cascade'] = complexity_analysis.uncertainty
        
        return risk_assessment
    
    async def _generate_contingency_plans(
        self, 
        goal_tree: Dict[str, GoalNode], 
        risk_assessment: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Gera planos de contingência para cenários de alto risco.
        
        Args:
            goal_tree: Árvore de objetivos
            risk_assessment: Avaliação de riscos
            
        Returns:
            Lista de planos de contingência
        """
        contingency_plans = []
        
        # Plano para sobrecarga de complexidade
        if risk_assessment.get('complexity_overload', 0) > 0.6:
            contingency_plans.append({
                'trigger': 'complexity_overload',
                'strategy': 'decompose_further',
                'actions': [
                    'Break down complex goals into smaller subtasks',
                    'Implement parallel processing where possible',
                    'Activate additional cognitive resources'
                ],
                'resource_multiplier': 1.5,
                'confidence': 0.8
            })
        
        # Plano para insuficiência de recursos
        if risk_assessment.get('resource_insufficiency', 0) > 0.6:
            contingency_plans.append({
                'trigger': 'resource_insufficiency',
                'strategy': 'resource_optimization',
                'actions': [
                    'Prioritize critical path goals',
                    'Implement resource pooling',
                    'Scale up computational resources dynamically'
                ],
                'resource_multiplier': 2.0,
                'confidence': 0.7
            })
        
        # Plano para falhas de dependência
        if risk_assessment.get('dependency_failure', 0) > 0.5:
            contingency_plans.append({
                'trigger': 'dependency_failure',
                'strategy': 'dependency_mitigation',
                'actions': [
                    'Create alternative dependency paths',
                    'Implement graceful degradation',
                    'Prepare rollback procedures'
                ],
                'resource_multiplier': 1.3,
                'confidence': 0.75
            })
        
        return contingency_plans
    
    async def _optimize_strategy_pareto(
        self, 
        strategy: ExecutionStrategy, 
        contingency_plans: List[Dict[str, Any]]
    ) -> ExecutionStrategy:
        """
        Otimiza estratégia usando análise multi-objetivo e eficiência de Pareto.
        
        Args:
            strategy: Estratégia inicial
            contingency_plans: Planos de contingência
            
        Returns:
            Estratégia otimizada
        """
        # Objetivos a otimizar: tempo, qualidade, recursos, confiabilidade
        objectives = {
            'time': strategy.estimated_duration,
            'quality': strategy.confidence,
            'resources': len(strategy.resource_plan.model_allocations),
            'reliability': 1.0 - sum(strategy.risk_assessment.values()) / len(strategy.risk_assessment) if strategy.risk_assessment else 0.8
        }
        
        # Aplicar otimizações baseadas em contingências
        optimization_factor = 1.0
        for plan in contingency_plans:
            if plan.get('confidence', 0) > 0.7:
                optimization_factor *= plan.get('resource_multiplier', 1.0)
        
        # Otimizar tempo com paralelização inteligente
        optimized_duration = strategy.estimated_duration / min(optimization_factor, 2.0)
        
        # Otimizar confiança considerando riscos mitigados
        risk_mitigation_bonus = len(contingency_plans) * 0.1
        optimized_confidence = min(strategy.confidence + risk_mitigation_bonus, 1.0)
        
        # Criar estratégia otimizada
        strategy.estimated_duration = optimized_duration
        strategy.confidence = optimized_confidence
        strategy.success_probability = optimized_confidence * 0.9  # Slightly conservative
        
        return strategy
    
    async def _setup_adaptive_monitoring(self, strategy: ExecutionStrategy) -> None:
        """
        Configura monitoramento adaptativo para ajustes em tempo real.
        
        Args:
            strategy: Estratégia de execução
        """
        monitoring_config = {
            'performance_thresholds': {
                'min_progress_rate': 0.1,  # 10% progress per hour minimum
                'max_error_rate': 0.05,    # 5% maximum error rate
                'resource_utilization_optimal': 0.75  # 75% optimal utilization
            },
            'adaptation_triggers': {
                'performance_degradation': 0.3,  # 30% below expected
                'resource_contention': 0.8,      # 80% resource usage
                'quality_threshold': 0.85        # 85% minimum quality
            },
            'intervention_strategies': [
                'resource_reallocation',
                'goal_reprioritization', 
                'parallel_execution_adjustment',
                'quality_parameter_tuning'
            ]
        }
        
        # Store monitoring configuration for runtime use
        strategy.monitoring_config = monitoring_config
    
    async def _estimate_execution_duration(self, goal_tree: Dict[str, GoalNode]) -> float:
        """
        Estima duração de execução usando análise de caminho crítico.
        
        Args:
            goal_tree: Árvore de objetivos
            
        Returns:
            Duração estimada em horas
        """
        if not goal_tree:
            return 1.0
        
        # Calcular caminho crítico considerando dependências
        critical_path_duration = 0.0
        parallel_paths = []
        
        # Identificar objetivos raiz (sem dependências)
        root_goals = [goal for goal in goal_tree.values() if not goal.dependencies]
        
        for root_goal in root_goals:
            path_duration = await self._calculate_goal_chain_duration(root_goal, goal_tree)
            parallel_paths.append(path_duration)
        
        # Duração é o máximo dos caminhos paralelos
        critical_path_duration = max(parallel_paths) if parallel_paths else 1.0
        
        # Adicionar buffer para incertezas (20%)
        return critical_path_duration * 1.2
    
    async def _calculate_goal_chain_duration(
        self, 
        goal: GoalNode, 
        goal_tree: Dict[str, GoalNode]
    ) -> float:
        """
        Calcula duração de uma cadeia de objetivos.
        
        Args:
            goal: Objetivo inicial
            goal_tree: Árvore completa de objetivos
            
        Returns:
            Duração total da cadeia
        """
        # Duração do objetivo atual
        current_duration = goal.complexity.time_horizon if goal.complexity else 1.0
        
        # Duração dos objetivos filhos (máximo, assumindo paralelos onde possível)
        children_durations = []
        for child_id in goal.children_ids:
            if child_id in goal_tree:
                child_goal = goal_tree[child_id]
                child_duration = await self._calculate_goal_chain_duration(child_goal, goal_tree)
                children_durations.append(child_duration)
        
        max_child_duration = max(children_durations) if children_durations else 0.0
        
        return current_duration + max_child_duration
    
    async def _calculate_strategy_confidence(
        self, 
        complexity_analysis: ComplexityAnalysis, 
        risk_assessment: Dict[str, float]
    ) -> float:
        """
        Calcula confiança na estratégia considerando complexidade e riscos.
        
        Args:
            complexity_analysis: Análise de complexidade
            risk_assessment: Avaliação de riscos
            
        Returns:
            Nível de confiança (0-1)
        """
        # Base confidence from complexity analysis
        base_confidence = complexity_analysis.confidence
        
        # Penalty for high complexity
        complexity_penalty = (complexity_analysis.level.value - 1) * 0.1  # Max 0.4 penalty
        
        # Penalty for risks
        total_risk = sum(risk_assessment.values()) / len(risk_assessment) if risk_assessment else 0.0
        risk_penalty = total_risk * 0.3
        
        # Bonus for meta-cognitive reasoning
        meta_cognitive_bonus = len(complexity_analysis.reasoning) * 0.05  # Up to 0.15 bonus
        
        confidence = base_confidence - complexity_penalty - risk_penalty + meta_cognitive_bonus
        
        return max(min(confidence, 1.0), 0.1)  # Clamp between 0.1 and 1.0

    async def shutdown(self) -> None:
        """Desliga a função executiva."""
        logger.info("Shutting down Executive Function")
        
        # Shutdown components
        await self.strategic_planner.shutdown()
        await self.meta_cognition.shutdown()
        await self.attention_controller.shutdown()
        await self.goal_hierarchy.shutdown()
        
        logger.info("Executive Function shutdown complete")
