"""
Multi-Modal Model Orchestration

Implementa orquestração inteligente de múltiplos modelos de IA especializados
com roteamento adaptativo e inferência ensemble.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from nexus.orchestration.model_router import NeuralModelRouter
from nexus.orchestration.ensemble import EnsembleInferenceEngine
from nexus.orchestration.performance_tracker import ModelPerformanceTracker

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos disponíveis."""
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    NATURAL_LANGUAGE = "natural_language"
    SPECIALIZED_DOMAIN = "specialized_domain"
    MULTIMODAL = "multimodal"


class TaskComplexity(Enum):
    """Níveis de complexidade de tarefas."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class ModelSpec:
    """Especificação de um modelo."""
    
    model_id: str
    model_name: str
    model_type: ModelType
    
    # Capacidades
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    
    # Configurações
    temperature: float = 0.0
    max_tokens: int = 4096
    context_window: int = 8192
    
    # Métricas de performance
    avg_response_time: float = 0.0
    success_rate: float = 0.0
    cost_per_token: float = 0.0
    
    # Disponibilidade
    is_available: bool = True
    rate_limit: Optional[int] = None
    
    # Metadados
    provider: str = ""
    version: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskProfile:
    """Perfil de uma tarefa para roteamento."""
    
    task_type: str
    complexity: TaskComplexity
    domain: str
    
    # Requisitos
    requires_reasoning: bool = False
    requires_code_generation: bool = False
    requires_analysis: bool = False
    requires_creativity: bool = False
    
    # Constraints
    max_response_time: Optional[float] = None
    max_cost: Optional[float] = None
    min_accuracy: Optional[float] = None
    
    # Contexto
    context_size: int = 0
    previous_interactions: List[str] = field(default_factory=list)
    
    # Metadados
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InferenceResult:
    """Resultado de inferência de modelo."""
    
    model_id: str
    result: Any
    confidence: float
    
    # Métricas
    response_time: float
    tokens_used: int
    cost: float
    
    # Metadados
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class OrchestrationStrategy:
    """Estratégia de orquestração para uma tarefa."""
    
    strategy_type: str  # single, ensemble, pipeline, adaptive
    selected_models: List[str]
    
    # Configurações de ensemble
    voting_strategy: Optional[str] = None
    confidence_weights: Optional[Dict[str, float]] = None
    
    # Configurações de pipeline
    pipeline_stages: Optional[List[Dict[str, Any]]] = None
    
    # Configurações adaptativas
    fallback_models: Optional[List[str]] = None
    retry_strategy: Optional[Dict[str, Any]] = None
    
    # Métricas esperadas
    expected_response_time: float = 0.0
    expected_cost: float = 0.0
    expected_accuracy: float = 0.0


class MultiModalOrchestrator:
    """
    Orquestrador Multi-Modal do NEXUS.
    
    Gerencia múltiplos modelos de IA especializados com roteamento inteligente,
    inferência ensemble e otimização adaptativa de performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o Orquestrador Multi-Modal.
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        
        # Pool de modelos disponíveis
        self.model_pool: Dict[str, ModelSpec] = {}
        
        # Componentes especializados
        self.router = NeuralModelRouter(self.config.get('router', {}))
        self.ensemble_engine = EnsembleInferenceEngine(self.config.get('ensemble', {}))
        self.performance_tracker = ModelPerformanceTracker(self.config.get('performance', {}))
        
        # Cache de estratégias
        self.strategy_cache: Dict[str, OrchestrationStrategy] = {}
        
        # Configurações
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.default_timeout = self.config.get('default_timeout_seconds', 30)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Estatísticas
        self.orchestration_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'ensemble_requests': 0,
            'single_model_requests': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
        
        # Semáforo para controle de concorrência
        self._concurrency_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Inicializar pool de modelos
        self._initialize_model_pool()
        
        logger.info("Multi-Modal Orchestrator initialized")
    
    def _initialize_model_pool(self) -> None:
        """Inicializa o pool de modelos disponíveis."""
        
        # Modelos de raciocínio estratégico
        self.model_pool['gpt4_reasoning'] = ModelSpec(
            model_id='gpt4_reasoning',
            model_name='GPT-4 Turbo',
            model_type=ModelType.REASONING,
            capabilities=['strategic_reasoning', 'complex_analysis', 'planning'],
            temperature=0.1,
            max_tokens=4096,
            context_window=128000,
            avg_response_time=2.5,
            success_rate=0.92,
            cost_per_token=0.00003,
            provider='openai'
        )
        
        self.model_pool['claude_tactical'] = ModelSpec(
            model_id='claude_tactical',
            model_name='Claude 3.5 Sonnet',
            model_type=ModelType.REASONING,
            capabilities=['tactical_planning', 'detailed_analysis', 'structured_thinking'],
            temperature=0.0,
            max_tokens=4096,
            context_window=200000,
            avg_response_time=1.8,
            success_rate=0.94,
            cost_per_token=0.000015,
            provider='anthropic'
        )
        
        # Modelos de geração de código
        self.model_pool['codellama_instruct'] = ModelSpec(
            model_id='codellama_instruct',
            model_name='Code Llama Instruct',
            model_type=ModelType.CODE_GENERATION,
            capabilities=['code_generation', 'code_completion', 'debugging'],
            specializations=['python', 'javascript', 'typescript', 'rust'],
            temperature=0.0,
            max_tokens=4096,
            context_window=16384,
            avg_response_time=1.2,
            success_rate=0.89,
            cost_per_token=0.000005,
            provider='meta'
        )
        
        self.model_pool['starcoder2'] = ModelSpec(
            model_id='starcoder2',
            model_name='StarCoder 2',
            model_type=ModelType.CODE_ANALYSIS,
            capabilities=['code_analysis', 'code_review', 'vulnerability_detection'],
            specializations=['security_analysis', 'performance_optimization'],
            temperature=0.0,
            max_tokens=2048,
            context_window=8192,
            avg_response_time=0.8,
            success_rate=0.91,
            cost_per_token=0.000003,
            provider='huggingface'
        )
        
        # Modelos especializados
        self.model_pool['sql_coder'] = ModelSpec(
            model_id='sql_coder',
            model_name='SQL Coder',
            model_type=ModelType.SPECIALIZED_DOMAIN,
            capabilities=['sql_generation', 'database_design', 'query_optimization'],
            specializations=['postgresql', 'mysql', 'sqlite'],
            temperature=0.0,
            max_tokens=2048,
            context_window=4096,
            avg_response_time=0.6,
            success_rate=0.95,
            cost_per_token=0.000002,
            provider='defog'
        )
        
        self.model_pool['security_analyst'] = ModelSpec(
            model_id='security_analyst',
            model_name='Security Analyst Model',
            model_type=ModelType.SPECIALIZED_DOMAIN,
            capabilities=['security_analysis', 'threat_detection', 'compliance_check'],
            specializations=['owasp', 'penetration_testing', 'secure_coding'],
            temperature=0.1,
            max_tokens=2048,
            context_window=8192,
            avg_response_time=1.0,
            success_rate=0.88,
            cost_per_token=0.000008,
            provider='custom'
        )
        
        logger.info(f"Initialized model pool with {len(self.model_pool)} models")
    
    async def initialize(self) -> None:
        """Inicializa o orquestrador."""
        
        # Inicializar componentes
        await self.router.initialize()
        await self.ensemble_engine.initialize()
        await self.performance_tracker.initialize()
        
        logger.info("Multi-Modal Orchestrator initialization complete")
    
    async def intelligent_dispatch(
        self, 
        task: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> InferenceResult:
        """
        Executa dispatch inteligente de tarefa para modelo(s) apropriado(s).
        
        Args:
            task: Descrição da tarefa
            context: Contexto da tarefa
            
        Returns:
            Resultado da inferência
        """
        async with self._concurrency_semaphore:
            start_time = datetime.utcnow()
            
            try:
                # Analisar perfil da tarefa
                task_profile = await self._analyze_task_profile(task, context)
                
                # Determinar estratégia de orquestração
                strategy = await self._determine_orchestration_strategy(task_profile)
                
                # Executar estratégia
                if strategy.strategy_type == 'single':
                    result = await self._execute_single_model_strategy(task, context, strategy)
                elif strategy.strategy_type == 'ensemble':
                    result = await self._execute_ensemble_strategy(task, context, strategy)
                elif strategy.strategy_type == 'pipeline':
                    result = await self._execute_pipeline_strategy(task, context, strategy)
                elif strategy.strategy_type == 'adaptive':
                    result = await self._execute_adaptive_strategy(task, context, strategy)
                else:
                    raise ValueError(f"Unknown strategy type: {strategy.strategy_type}")
                
                # Atualizar estatísticas
                response_time = (datetime.utcnow() - start_time).total_seconds()
                await self._update_orchestration_stats(True, response_time, strategy)
                
                # Registrar performance
                await self.performance_tracker.record_performance(
                    task_profile, result, strategy.selected_models
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error in intelligent dispatch: {e}", exc_info=True)
                
                # Atualizar estatísticas de erro
                response_time = (datetime.utcnow() - start_time).total_seconds()
                await self._update_orchestration_stats(False, response_time, None)
                
                # Retornar resultado de erro
                return InferenceResult(
                    model_id='error',
                    result=None,
                    confidence=0.0,
                    response_time=response_time,
                    tokens_used=0,
                    cost=0.0,
                    success=False,
                    error_message=str(e)
                )
    
    async def get_active_models(self) -> List[str]:
        """
        Obtém lista de modelos ativos.
        
        Returns:
            Lista de IDs de modelos ativos
        """
        return [
            model_id for model_id, model_spec in self.model_pool.items()
            if model_spec.is_available
        ]
    
    async def get_model_capabilities(self, model_id: str) -> Optional[List[str]]:
        """
        Obtém capacidades de um modelo específico.
        
        Args:
            model_id: ID do modelo
            
        Returns:
            Lista de capacidades ou None se modelo não encontrado
        """
        model_spec = self.model_pool.get(model_id)
        return model_spec.capabilities if model_spec else None
    
    async def update_model_performance(
        self, 
        model_id: str, 
        performance_metrics: Dict[str, float]
    ) -> None:
        """
        Atualiza métricas de performance de um modelo.
        
        Args:
            model_id: ID do modelo
            performance_metrics: Métricas de performance
        """
        if model_id in self.model_pool:
            model_spec = self.model_pool[model_id]
            
            # Atualizar métricas com média móvel
            alpha = 0.1  # Fator de suavização
            
            if 'response_time' in performance_metrics:
                model_spec.avg_response_time = (
                    (1 - alpha) * model_spec.avg_response_time +
                    alpha * performance_metrics['response_time']
                )
            
            if 'success_rate' in performance_metrics:
                model_spec.success_rate = (
                    (1 - alpha) * model_spec.success_rate +
                    alpha * performance_metrics['success_rate']
                )
            
            logger.debug(f"Updated performance for model {model_id}")
    
    async def add_model(self, model_spec: ModelSpec) -> None:
        """
        Adiciona novo modelo ao pool.
        
        Args:
            model_spec: Especificação do modelo
        """
        self.model_pool[model_spec.model_id] = model_spec
        logger.info(f"Added model to pool: {model_spec.model_id}")
    
    async def remove_model(self, model_id: str) -> None:
        """
        Remove modelo do pool.
        
        Args:
            model_id: ID do modelo a ser removido
        """
        if model_id in self.model_pool:
            del self.model_pool[model_id]
            logger.info(f"Removed model from pool: {model_id}")
    
    async def get_orchestration_statistics(self) -> Dict[str, Any]:
        """
        Obtém estatísticas de orquestração.
        
        Returns:
            Estatísticas detalhadas
        """
        # Estatísticas básicas
        basic_stats = dict(self.orchestration_stats)
        
        # Calcular métricas derivadas
        if basic_stats['total_requests'] > 0:
            basic_stats['success_rate'] = (
                basic_stats['successful_requests'] / basic_stats['total_requests']
            )
            basic_stats['ensemble_usage_rate'] = (
                basic_stats['ensemble_requests'] / basic_stats['total_requests']
            )
        else:
            basic_stats['success_rate'] = 0.0
            basic_stats['ensemble_usage_rate'] = 0.0
        
        # Estatísticas de modelos
        model_stats = {}
        for model_id, model_spec in self.model_pool.items():
            model_stats[model_id] = {
                'avg_response_time': model_spec.avg_response_time,
                'success_rate': model_spec.success_rate,
                'cost_per_token': model_spec.cost_per_token,
                'is_available': model_spec.is_available
            }
        
        # Estatísticas de performance
        performance_stats = await self.performance_tracker.get_performance_statistics()
        
        return {
            'basic_statistics': basic_stats,
            'model_statistics': model_stats,
            'performance_statistics': performance_stats,
            'cache_statistics': {
                'cache_size': len(self.strategy_cache),
                'cache_hit_rate': (
                    basic_stats['cache_hits'] / basic_stats['total_requests']
                    if basic_stats['total_requests'] > 0 else 0.0
                )
            }
        }
    
    async def _analyze_task_profile(
        self, 
        task: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> TaskProfile:
        """Analisa perfil da tarefa."""
        
        # Extrair informações da tarefa
        task_description = task.get('description', '')
        task_type = task.get('type', 'general')
        
        # Determinar complexidade
        complexity = await self._determine_task_complexity(task, context)
        
        # Determinar domínio
        domain = await self._determine_task_domain(task_description)
        
        # Analisar requisitos
        requires_reasoning = await self._requires_reasoning(task_description)
        requires_code_generation = await self._requires_code_generation(task_description)
        requires_analysis = await self._requires_analysis(task_description)
        requires_creativity = await self._requires_creativity(task_description)
        
        # Extrair constraints
        max_response_time = context.get('max_response_time')
        max_cost = context.get('max_cost')
        min_accuracy = context.get('min_accuracy')
        
        # Calcular tamanho do contexto
        context_size = len(str(context))
        
        return TaskProfile(
            task_type=task_type,
            complexity=complexity,
            domain=domain,
            requires_reasoning=requires_reasoning,
            requires_code_generation=requires_code_generation,
            requires_analysis=requires_analysis,
            requires_creativity=requires_creativity,
            max_response_time=max_response_time,
            max_cost=max_cost,
            min_accuracy=min_accuracy,
            context_size=context_size,
            priority=task.get('priority', 1)
        )
    
    async def _determine_task_complexity(
        self, 
        task: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> TaskComplexity:
        """Determina complexidade da tarefa."""
        
        complexity_indicators = 0
        
        # Indicadores de complexidade
        task_description = task.get('description', '').lower()
        
        # Palavras-chave de complexidade
        complex_keywords = [
            'complex', 'sophisticated', 'advanced', 'intricate', 'comprehensive',
            'multi-step', 'integrate', 'optimize', 'analyze', 'design'
        ]
        
        for keyword in complex_keywords:
            if keyword in task_description:
                complexity_indicators += 1
        
        # Tamanho do contexto
        context_size = len(str(context))
        if context_size > 10000:
            complexity_indicators += 2
        elif context_size > 5000:
            complexity_indicators += 1
        
        # Número de requisitos
        requirements = task.get('requirements', [])
        if len(requirements) > 5:
            complexity_indicators += 2
        elif len(requirements) > 2:
            complexity_indicators += 1
        
        # Mapear para enum
        if complexity_indicators >= 6:
            return TaskComplexity.HIGHLY_COMPLEX
        elif complexity_indicators >= 4:
            return TaskComplexity.COMPLEX
        elif complexity_indicators >= 2:
            return TaskComplexity.MODERATE
        elif complexity_indicators >= 1:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.TRIVIAL
    
    async def _determine_task_domain(self, task_description: str) -> str:
        """Determina domínio da tarefa."""
        
        task_lower = task_description.lower()
        
        # Mapeamento de palavras-chave para domínios
        domain_keywords = {
            'code': ['code', 'programming', 'function', 'class', 'algorithm'],
            'sql': ['sql', 'database', 'query', 'table', 'select'],
            'security': ['security', 'vulnerability', 'attack', 'encryption', 'auth'],
            'architecture': ['architecture', 'design', 'system', 'component', 'pattern'],
            'analysis': ['analyze', 'review', 'evaluate', 'assess', 'examine'],
            'planning': ['plan', 'strategy', 'roadmap', 'schedule', 'timeline'],
            'general': []
        }
        
        # Encontrar domínio com mais matches
        best_domain = 'general'
        max_matches = 0
        
        for domain, keywords in domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in task_lower)
            if matches > max_matches:
                max_matches = matches
                best_domain = domain
        
        return best_domain
    
    async def _requires_reasoning(self, task_description: str) -> bool:
        """Verifica se tarefa requer raciocínio."""
        
        reasoning_keywords = [
            'analyze', 'reason', 'think', 'evaluate', 'assess', 'consider',
            'plan', 'strategy', 'decide', 'choose', 'compare', 'weigh'
        ]
        
        task_lower = task_description.lower()
        return any(keyword in task_lower for keyword in reasoning_keywords)
    
    async def _requires_code_generation(self, task_description: str) -> bool:
        """Verifica se tarefa requer geração de código."""
        
        code_keywords = [
            'code', 'function', 'class', 'implement', 'write', 'create',
            'generate', 'build', 'develop', 'program'
        ]
        
        task_lower = task_description.lower()
        return any(keyword in task_lower for keyword in code_keywords)
    
    async def _requires_analysis(self, task_description: str) -> bool:
        """Verifica se tarefa requer análise."""
        
        analysis_keywords = [
            'analyze', 'review', 'examine', 'inspect', 'evaluate',
            'assess', 'check', 'validate', 'verify'
        ]
        
        task_lower = task_description.lower()
        return any(keyword in task_lower for keyword in analysis_keywords)
    
    async def _requires_creativity(self, task_description: str) -> bool:
        """Verifica se tarefa requer criatividade."""
        
        creativity_keywords = [
            'creative', 'innovative', 'novel', 'unique', 'original',
            'brainstorm', 'ideate', 'design', 'invent'
        ]
        
        task_lower = task_description.lower()
        return any(keyword in task_lower for keyword in creativity_keywords)
    
    async def _determine_orchestration_strategy(
        self, 
        task_profile: TaskProfile
    ) -> OrchestrationStrategy:
        """Determina estratégia de orquestração."""
        
        # Verificar cache
        cache_key = self._generate_strategy_cache_key(task_profile)
        if self.enable_caching and cache_key in self.strategy_cache:
            self.orchestration_stats['cache_hits'] += 1
            return self.strategy_cache[cache_key]
        
        # Determinar estratégia baseada no perfil
        if task_profile.complexity in [TaskComplexity.COMPLEX, TaskComplexity.HIGHLY_COMPLEX]:
            # Tarefas complexas usam ensemble
            strategy = await self._create_ensemble_strategy(task_profile)
        elif task_profile.requires_code_generation and task_profile.requires_analysis:
            # Tarefas que requerem geração e análise usam pipeline
            strategy = await self._create_pipeline_strategy(task_profile)
        else:
            # Tarefas simples usam modelo único
            strategy = await self._create_single_model_strategy(task_profile)
        
        # Cache da estratégia
        if self.enable_caching:
            self.strategy_cache[cache_key] = strategy
        
        return strategy
    
    async def _create_single_model_strategy(self, task_profile: TaskProfile) -> OrchestrationStrategy:
        """Cria estratégia de modelo único."""
        
        # Selecionar melhor modelo para a tarefa
        best_model = await self.router.select_best_model(task_profile)
        
        return OrchestrationStrategy(
            strategy_type='single',
            selected_models=[best_model],
            expected_response_time=self.model_pool[best_model].avg_response_time,
            expected_cost=self._estimate_cost(best_model, task_profile),
            expected_accuracy=self.model_pool[best_model].success_rate
        )
    
    async def _create_ensemble_strategy(self, task_profile: TaskProfile) -> OrchestrationStrategy:
        """Cria estratégia de ensemble."""
        
        # Selecionar modelos para ensemble
        ensemble_models = await self.router.select_ensemble(
            task_profile, 
            performance_history=await self.performance_tracker.get_history()
        )
        
        # Determinar estratégia de votação
        voting_strategy = 'weighted_confidence'
        
        # Calcular pesos de confiança
        confidence_weights = {}
        for model_id in ensemble_models:
            model_spec = self.model_pool[model_id]
            confidence_weights[model_id] = model_spec.success_rate
        
        return OrchestrationStrategy(
            strategy_type='ensemble',
            selected_models=ensemble_models,
            voting_strategy=voting_strategy,
            confidence_weights=confidence_weights,
            expected_response_time=max(
                self.model_pool[m].avg_response_time for m in ensemble_models
            ),
            expected_cost=sum(
                self._estimate_cost(m, task_profile) for m in ensemble_models
            ),
            expected_accuracy=0.95  # Ensemble geralmente tem alta precisão
        )
    
    async def _create_pipeline_strategy(self, task_profile: TaskProfile) -> OrchestrationStrategy:
        """Cria estratégia de pipeline."""
        
        pipeline_stages = []
        selected_models = []
        
        # Estágio 1: Análise/Planejamento
        if task_profile.requires_reasoning:
            reasoning_model = await self._select_model_by_capability('strategic_reasoning')
            pipeline_stages.append({
                'stage': 'analysis',
                'model': reasoning_model,
                'input_transform': 'pass_through',
                'output_transform': 'extract_plan'
            })
            selected_models.append(reasoning_model)
        
        # Estágio 2: Geração de código
        if task_profile.requires_code_generation:
            code_model = await self._select_model_by_capability('code_generation')
            pipeline_stages.append({
                'stage': 'generation',
                'model': code_model,
                'input_transform': 'plan_to_code_prompt',
                'output_transform': 'extract_code'
            })
            selected_models.append(code_model)
        
        # Estágio 3: Verificação
        if task_profile.requires_analysis:
            analysis_model = await self._select_model_by_capability('code_analysis')
            pipeline_stages.append({
                'stage': 'verification',
                'model': analysis_model,
                'input_transform': 'code_to_review_prompt',
                'output_transform': 'extract_feedback'
            })
            selected_models.append(analysis_model)
        
        return OrchestrationStrategy(
            strategy_type='pipeline',
            selected_models=selected_models,
            pipeline_stages=pipeline_stages,
            expected_response_time=sum(
                self.model_pool[m].avg_response_time for m in selected_models
            ),
            expected_cost=sum(
                self._estimate_cost(m, task_profile) for m in selected_models
            ),
            expected_accuracy=0.88  # Pipeline tem boa precisão
        )
    
    async def _select_model_by_capability(self, capability: str) -> str:
        """Seleciona modelo por capacidade específica."""
        
        candidates = [
            model_id for model_id, model_spec in self.model_pool.items()
            if capability in model_spec.capabilities and model_spec.is_available
        ]
        
        if not candidates:
            # Fallback para primeiro modelo disponível
            candidates = [
                model_id for model_id, model_spec in self.model_pool.items()
                if model_spec.is_available
            ]
        
        if not candidates:
            raise RuntimeError("No available models")
        
        # Selecionar modelo com melhor performance
        best_model = max(
            candidates,
            key=lambda m: self.model_pool[m].success_rate
        )
        
        return best_model
    
    def _estimate_cost(self, model_id: str, task_profile: TaskProfile) -> float:
        """Estima custo de execução."""
        
        model_spec = self.model_pool[model_id]
        
        # Estimar tokens baseado na complexidade
        estimated_tokens = {
            TaskComplexity.TRIVIAL: 100,
            TaskComplexity.SIMPLE: 300,
            TaskComplexity.MODERATE: 800,
            TaskComplexity.COMPLEX: 2000,
            TaskComplexity.HIGHLY_COMPLEX: 4000
        }
        
        tokens = estimated_tokens.get(task_profile.complexity, 1000)
        
        return tokens * model_spec.cost_per_token
    
    def _generate_strategy_cache_key(self, task_profile: TaskProfile) -> str:
        """Gera chave de cache para estratégia."""
        
        key_components = [
            task_profile.task_type,
            task_profile.complexity.value,
            task_profile.domain,
            str(task_profile.requires_reasoning),
            str(task_profile.requires_code_generation),
            str(task_profile.requires_analysis),
            str(task_profile.requires_creativity)
        ]
        
        return '|'.join(key_components)
    
    async def _execute_single_model_strategy(
        self, 
        task: Dict[str, Any], 
        context: Dict[str, Any], 
        strategy: OrchestrationStrategy
    ) -> InferenceResult:
        """Executa estratégia de modelo único."""
        
        model_id = strategy.selected_models[0]
        
        # Simular inferência do modelo
        start_time = datetime.utcnow()
        
        # Aqui seria a chamada real para o modelo
        # result = await self._call_model(model_id, task, context)
        
        # Simulação
        await asyncio.sleep(0.1)  # Simular latência
        
        result = {
            'response': f"Response from {model_id}",
            'reasoning': "Detailed reasoning process",
            'confidence': 0.85
        }
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Atualizar estatísticas
        self.orchestration_stats['single_model_requests'] += 1
        
        return InferenceResult(
            model_id=model_id,
            result=result,
            confidence=result['confidence'],
            response_time=response_time,
            tokens_used=500,  # Estimativa
            cost=self._estimate_cost(model_id, None)
        )
    
    async def _execute_ensemble_strategy(
        self, 
        task: Dict[str, Any], 
        context: Dict[str, Any], 
        strategy: OrchestrationStrategy
    ) -> InferenceResult:
        """Executa estratégia de ensemble."""
        
        # Executar inferência em paralelo
        ensemble_tasks = []
        
        for model_id in strategy.selected_models:
            ensemble_task = asyncio.create_task(
                self._simulate_model_inference(model_id, task, context)
            )
            ensemble_tasks.append(ensemble_task)
        
        # Aguardar todos os resultados
        individual_results = await asyncio.gather(*ensemble_tasks)
        
        # Combinar resultados usando engine de ensemble
        combined_result = await self.ensemble_engine.ensemble_inference(
            strategy.selected_models,
            task,
            voting_strategy=strategy.voting_strategy,
            individual_results=individual_results
        )
        
        # Atualizar estatísticas
        self.orchestration_stats['ensemble_requests'] += 1
        
        return combined_result
    
    async def _execute_pipeline_strategy(
        self, 
        task: Dict[str, Any], 
        context: Dict[str, Any], 
        strategy: OrchestrationStrategy
    ) -> InferenceResult:
        """Executa estratégia de pipeline."""
        
        current_input = task
        pipeline_results = []
        total_cost = 0.0
        total_time = 0.0
        
        # Executar cada estágio do pipeline
        for stage in strategy.pipeline_stages:
            model_id = stage['model']
            
            # Transformar entrada
            transformed_input = await self._transform_pipeline_input(
                current_input, stage['input_transform']
            )
            
            # Executar modelo
            stage_result = await self._simulate_model_inference(
                model_id, transformed_input, context
            )
            
            # Transformar saída
            transformed_output = await self._transform_pipeline_output(
                stage_result.result, stage['output_transform']
            )
            
            pipeline_results.append({
                'stage': stage['stage'],
                'model': model_id,
                'result': transformed_output
            })
            
            # Atualizar métricas
            total_cost += stage_result.cost
            total_time += stage_result.response_time
            
            # Preparar entrada para próximo estágio
            current_input = transformed_output
        
        # Combinar resultados do pipeline
        final_result = {
            'pipeline_results': pipeline_results,
            'final_output': current_input,
            'stages_completed': len(pipeline_results)
        }
        
        return InferenceResult(
            model_id='pipeline',
            result=final_result,
            confidence=0.88,  # Pipeline tem boa confiança
            response_time=total_time,
            tokens_used=sum(r.get('tokens_used', 0) for r in pipeline_results),
            cost=total_cost
        )
    
    async def _execute_adaptive_strategy(
        self, 
        task: Dict[str, Any], 
        context: Dict[str, Any], 
        strategy: OrchestrationStrategy
    ) -> InferenceResult:
        """Executa estratégia adaptativa."""
        
        # Tentar modelo principal primeiro
        primary_model = strategy.selected_models[0]
        
        try:
            result = await self._simulate_model_inference(primary_model, task, context)
            
            # Verificar se resultado é satisfatório
            if result.confidence > 0.7 and result.success:
                return result
            
        except Exception as e:
            logger.warning(f"Primary model {primary_model} failed: {e}")
        
        # Tentar modelos de fallback
        for fallback_model in strategy.fallback_models or []:
            try:
                result = await self._simulate_model_inference(fallback_model, task, context)
                
                if result.success:
                    return result
                    
            except Exception as e:
                logger.warning(f"Fallback model {fallback_model} failed: {e}")
        
        # Se todos falharam, retornar erro
        return InferenceResult(
            model_id='adaptive_failed',
            result=None,
            confidence=0.0,
            response_time=0.0,
            tokens_used=0,
            cost=0.0,
            success=False,
            error_message="All models in adaptive strategy failed"
        )
    
    async def _simulate_model_inference(
        self, 
        model_id: str, 
        task: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> InferenceResult:
        """Simula inferência de modelo."""
        
        model_spec = self.model_pool[model_id]
        
        # Simular latência
        await asyncio.sleep(model_spec.avg_response_time / 10)  # Escalar para simulação
        
        # Simular resultado
        result = {
            'response': f"Simulated response from {model_id}",
            'model_type': model_spec.model_type.value,
            'capabilities_used': model_spec.capabilities[:2]  # Primeiras 2 capacidades
        }
        
        return InferenceResult(
            model_id=model_id,
            result=result,
            confidence=model_spec.success_rate,
            response_time=model_spec.avg_response_time,
            tokens_used=500,  # Estimativa
            cost=self._estimate_cost(model_id, None)
        )
    
    async def _transform_pipeline_input(self, input_data: Any, transform_type: str) -> Any:
        """Transforma entrada para estágio do pipeline."""
        
        if transform_type == 'pass_through':
            return input_data
        elif transform_type == 'plan_to_code_prompt':
            return {
                'prompt': f"Generate code based on this plan: {input_data}",
                'context': 'code_generation'
            }
        elif transform_type == 'code_to_review_prompt':
            return {
                'prompt': f"Review this code for issues: {input_data}",
                'context': 'code_review'
            }
        else:
            return input_data
    
    async def _transform_pipeline_output(self, output_data: Any, transform_type: str) -> Any:
        """Transforma saída de estágio do pipeline."""
        
        if transform_type == 'extract_plan':
            return {
                'plan': output_data.get('response', ''),
                'steps': ['step1', 'step2', 'step3']  # Simulado
            }
        elif transform_type == 'extract_code':
            return {
                'code': output_data.get('response', ''),
                'language': 'python'  # Simulado
            }
        elif transform_type == 'extract_feedback':
            return {
                'feedback': output_data.get('response', ''),
                'issues': ['issue1', 'issue2']  # Simulado
            }
        else:
            return output_data
    
    async def _update_orchestration_stats(
        self, 
        success: bool, 
        response_time: float, 
        strategy: Optional[OrchestrationStrategy]
    ) -> None:
        """Atualiza estatísticas de orquestração."""
        
        self.orchestration_stats['total_requests'] += 1
        
        if success:
            self.orchestration_stats['successful_requests'] += 1
        
        # Atualizar tempo médio de resposta
        total_requests = self.orchestration_stats['total_requests']
        current_avg = self.orchestration_stats['average_response_time']
        
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.orchestration_stats['average_response_time'] = new_avg
    
    async def shutdown(self) -> None:
        """Desliga o orquestrador multi-modal."""
        
        logger.info("Shutting down Multi-Modal Orchestrator")
        
        # Desligar componentes
        await self.router.shutdown()
        await self.ensemble_engine.shutdown()
        await self.performance_tracker.shutdown()
        
        # Limpar caches
        self.strategy_cache.clear()
        
        logger.info("Multi-Modal Orchestrator shutdown complete")
