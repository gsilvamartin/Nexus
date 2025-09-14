"""
Agentic Workflow Orchestrator

Orquestra workflows agênticos usando padrões modulares de coordenação
para transformar chamadas AI isoladas em sistemas autônomos e adaptativos.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json
from enum import Enum

from .patterns import (
    ReflectionPattern,
    HierarchicalPlanningPattern,
    ToolUsePattern,
    CollaborationPattern,
    HumanInLoopPattern,
    SelfCorrectionPattern,
    MemoryPattern,
    SynthesisPattern,
    RoutingPattern
)

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status de um workflow."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowPriority(Enum):
    """Prioridade de um workflow."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkflowStep:
    """Passo individual de um workflow."""
    
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    pattern_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowDefinition:
    """Definição de um workflow agêntico."""
    
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    timeout: Optional[float] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WorkflowExecution:
    """Execução de um workflow."""
    
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_definition: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class AgenticWorkflowOrchestrator:
    """
    Orquestrador de Workflows Agênticos.
    
    Coordena execução de workflows usando padrões modulares de coordenação
    para criar sistemas autônomos e adaptativos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Configurações de orquestração
        self.max_concurrent_workflows = config.get('max_concurrent_workflows', 50)
        self.step_timeout = config.get('step_timeout', 60)
        self.workflow_timeout = config.get('workflow_timeout', 300)
        self.retry_delay = config.get('retry_delay', 5)
        
        # Estado do orquestrador
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.completed_executions: List[WorkflowExecution] = []
        
        # Padrões de workflow
        self.workflow_patterns = {
            'reflection': ReflectionPattern(config.get('reflection', {})),
            'planning': HierarchicalPlanningPattern(config.get('planning', {})),
            'tool_use': ToolUsePattern(config.get('tool_use', {})),
            'multi_agent_collaboration': CollaborationPattern(config.get('collaboration', {})),
            'human_in_loop': HumanInLoopPattern(config.get('human_in_loop', {})),
            'self_correction': SelfCorrectionPattern(config.get('self_correction', {})),
            'memory_consolidation': MemoryPattern(config.get('memory', {})),
            'knowledge_synthesis': SynthesisPattern(config.get('synthesis', {})),
            'adaptive_routing': RoutingPattern(config.get('routing', {}))
        }
        
        # Métricas de performance
        self.orchestration_metrics = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_execution_time': 0.0,
            'pattern_usage': {},
            'step_success_rate': 0.0
        }
        
        # Sistema de monitoramento
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("Agentic Workflow Orchestrator initialized")
    
    async def initialize(self) -> None:
        """Inicializa o orquestrador de workflows."""
        
        logger.info("Initializing Agentic Workflow Orchestrator")
        
        # Inicializar padrões de workflow
        await self._initialize_workflow_patterns()
        
        # Iniciar sistema de monitoramento
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_workflows())
        
        logger.info("Agentic Workflow Orchestrator initialization complete")
    
    async def create_workflow(
        self, 
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        priority: WorkflowPriority = WorkflowPriority.NORMAL,
        timeout: Optional[float] = None
    ) -> WorkflowDefinition:
        """Cria uma nova definição de workflow."""
        
        logger.info(f"Creating workflow: {name}")
        
        # Converter passos para objetos WorkflowStep
        workflow_steps = []
        for step_data in steps:
            step = WorkflowStep(
                name=step_data.get('name', ''),
                description=step_data.get('description', ''),
                pattern_type=step_data.get('pattern_type', ''),
                parameters=step_data.get('parameters', {}),
                dependencies=step_data.get('dependencies', []),
                max_retries=step_data.get('max_retries', 3)
            )
            workflow_steps.append(step)
        
        # Criar definição de workflow
        workflow = WorkflowDefinition(
            name=name,
            description=description,
            steps=workflow_steps,
            priority=priority,
            timeout=timeout,
            retry_policy={
                'max_retries': 3,
                'retry_delay': self.retry_delay,
                'exponential_backoff': True
            }
        )
        
        # Armazenar definição
        self.workflow_definitions[workflow.workflow_id] = workflow
        
        logger.info(f"Workflow created: {workflow.workflow_id}")
        return workflow
    
    async def execute_workflow(
        self, 
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Executa um workflow."""
        
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow_definition = self.workflow_definitions[workflow_id]
        
        logger.info(f"Executing workflow: {workflow_definition.name}")
        
        # Criar execução
        execution = WorkflowExecution(
            workflow_definition=workflow_definition,
            status=WorkflowStatus.PENDING
        )
        
        # Adicionar contexto
        if context:
            execution.results['context'] = context
        
        # Registrar execução
        self.active_executions[execution.execution_id] = execution
        
        try:
            # Executar workflow
            await self._execute_workflow_steps(execution)
            
            # Marcar como concluída
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.utcnow()
            
            # Mover para execuções concluídas
            self.completed_executions.append(execution)
            del self.active_executions[execution.execution_id]
            
            # Atualizar métricas
            self.orchestration_metrics['successful_workflows'] += 1
            self.orchestration_metrics['total_workflows'] += 1
            
            logger.info(f"Workflow completed: {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.utcnow()
            
            # Atualizar métricas
            self.orchestration_metrics['failed_workflows'] += 1
            self.orchestration_metrics['total_workflows'] += 1
        
        return execution
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution) -> None:
        """Executa passos de um workflow."""
        
        execution.status = WorkflowStatus.RUNNING
        execution.start_time = datetime.utcnow()
        
        workflow = execution.workflow_definition
        
        # Ordenar passos por dependências
        ordered_steps = await self._topological_sort_steps(workflow.steps)
        
        # Executar passos em ordem
        for step in ordered_steps:
            try:
                # Verificar se pode executar
                if not await self._can_execute_step(step, execution):
                    await asyncio.sleep(0.1)  # Aguardar dependências
                    continue
                
                # Executar passo
                await self._execute_step(step, execution)
                
                # Marcar como concluído
                execution.completed_steps.append(step.step_id)
                
            except Exception as e:
                logger.error(f"Step execution failed: {step.step_id} - {e}")
                
                # Aplicar política de retry
                if step.retry_count < step.max_retries:
                    step.retry_count += 1
                    step.status = "pending"
                    await asyncio.sleep(self.retry_delay * step.retry_count)
                    continue
                else:
                    step.status = "failed"
                    step.error = str(e)
                    execution.failed_steps.append(step.step_id)
                    
                    # Verificar se deve falhar o workflow inteiro
                    if workflow.retry_policy.get('fail_on_step_failure', True):
                        raise e
    
    async def _topological_sort_steps(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Ordena passos topologicamente baseado nas dependências."""
        
        # Implementação simplificada de ordenação topológica
        sorted_steps = []
        remaining_steps = steps.copy()
        
        while remaining_steps:
            # Encontrar passos sem dependências pendentes
            ready_steps = []
            for step in remaining_steps:
                if not step.dependencies or all(
                    dep in [s.step_id for s in sorted_steps] 
                    for dep in step.dependencies
                ):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Ciclo detectado ou dependência inválida
                break
            
            # Adicionar passos prontos
            sorted_steps.extend(ready_steps)
            for step in ready_steps:
                remaining_steps.remove(step)
        
        return sorted_steps
    
    async def _can_execute_step(
        self, 
        step: WorkflowStep, 
        execution: WorkflowExecution
    ) -> bool:
        """Verifica se um passo pode ser executado."""
        
        # Verificar dependências
        for dep_id in step.dependencies:
            if dep_id not in execution.completed_steps:
                return False
        
        # Verificar se não está em execução
        if step.status == "running":
            return False
        
        return True
    
    async def _execute_step(
        self, 
        step: WorkflowStep, 
        execution: WorkflowExecution
    ) -> None:
        """Executa um passo individual do workflow."""
        
        logger.info(f"Executing step: {step.name}")
        
        step.status = "running"
        execution.current_step = step.step_id
        
        start_time = datetime.utcnow()
        
        try:
            # Obter padrão de workflow
            pattern = self.workflow_patterns.get(step.pattern_type)
            if not pattern:
                raise ValueError(f"Unknown workflow pattern: {step.pattern_type}")
            
            # Preparar contexto
            context = {
                'step': step,
                'execution': execution,
                'workflow': execution.workflow_definition,
                'previous_results': execution.results
            }
            
            # Executar padrão
            result = await pattern.execute(step.parameters, context)
            
            # Armazenar resultado
            step.result = result
            step.status = "completed"
            execution.results[step.step_id] = result
            
            # Atualizar métricas do padrão
            pattern_name = step.pattern_type
            if pattern_name not in self.orchestration_metrics['pattern_usage']:
                self.orchestration_metrics['pattern_usage'][pattern_name] = 0
            self.orchestration_metrics['pattern_usage'][pattern_name] += 1
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            raise
        
        finally:
            # Calcular tempo de execução
            step.execution_time = (datetime.utcnow() - start_time).total_seconds()
    
    async def _initialize_workflow_patterns(self) -> None:
        """Inicializa padrões de workflow."""
        
        logger.info("Initializing workflow patterns")
        
        # Inicializar cada padrão
        for pattern_name, pattern in self.workflow_patterns.items():
            if hasattr(pattern, 'initialize'):
                await pattern.initialize()
        
        logger.info("Workflow patterns initialized")
    
    async def _monitor_workflows(self) -> None:
        """Monitora workflows ativos."""
        
        while self.is_running:
            try:
                # Verificar timeouts
                await self._check_workflow_timeouts()
                
                # Verificar passos travados
                await self._check_stuck_steps()
                
                # Atualizar métricas
                await self._update_metrics()
                
                # Aguardar próximo ciclo
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in workflow monitoring: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _check_workflow_timeouts(self) -> None:
        """Verifica timeouts de workflows."""
        
        current_time = datetime.utcnow()
        
        for execution_id, execution in self.active_executions.items():
            if execution.start_time:
                elapsed = (current_time - execution.start_time).total_seconds()
                timeout = execution.workflow_definition.timeout or self.workflow_timeout
                
                if elapsed > timeout:
                    logger.warning(f"Workflow timeout: {execution_id}")
                    execution.status = WorkflowStatus.FAILED
                    execution.error = "Workflow timeout"
                    execution.end_time = current_time
    
    async def _check_stuck_steps(self) -> None:
        """Verifica passos travados."""
        
        current_time = datetime.utcnow()
        
        for execution in self.active_executions.values():
            for step in execution.workflow_definition.steps:
                if step.status == "running":
                    # Verificar se está travado há muito tempo
                    if hasattr(step, 'start_time') and step.start_time:
                        elapsed = (current_time - step.start_time).total_seconds()
                        if elapsed > self.step_timeout:
                            logger.warning(f"Step timeout: {step.step_id}")
                            step.status = "failed"
                            step.error = "Step timeout"
    
    async def _update_metrics(self) -> None:
        """Atualiza métricas de orquestração."""
        
        # Calcular taxa de sucesso de passos
        total_steps = 0
        successful_steps = 0
        
        for execution in self.active_executions.values():
            for step in execution.workflow_definition.steps:
                total_steps += 1
                if step.status == "completed":
                    successful_steps += 1
        
        if total_steps > 0:
            self.orchestration_metrics['step_success_rate'] = successful_steps / total_steps
        
        # Calcular tempo médio de execução
        if self.completed_executions:
            total_time = sum(
                (execution.end_time - execution.start_time).total_seconds()
                for execution in self.completed_executions
                if execution.start_time and execution.end_time
            )
            self.orchestration_metrics['average_execution_time'] = total_time / len(self.completed_executions)
    
    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtém status de um workflow específico."""
        
        execution = self.active_executions.get(execution_id)
        if not execution:
            # Verificar execuções concluídas
            for completed in self.completed_executions:
                if completed.execution_id == execution_id:
                    execution = completed
                    break
        
        if not execution:
            return None
        
        return {
            'execution_id': execution.execution_id,
            'workflow_name': execution.workflow_definition.name,
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'current_step': execution.current_step,
            'completed_steps': len(execution.completed_steps),
            'total_steps': len(execution.workflow_definition.steps),
            'failed_steps': len(execution.failed_steps),
            'error': execution.error
        }
    
    async def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Obtém métricas de orquestração."""
        
        return {
            'orchestration_metrics': self.orchestration_metrics,
            'active_workflows': len(self.active_executions),
            'completed_workflows': len(self.completed_executions),
            'total_definitions': len(self.workflow_definitions)
        }
    
    async def shutdown(self) -> None:
        """Desliga o orquestrador de workflows."""
        
        logger.info("Shutting down Agentic Workflow Orchestrator")
        
        # Parar monitoramento
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Finalizar workflows ativos
        for execution in self.active_executions.values():
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.CANCELLED
                execution.end_time = datetime.utcnow()
        
        logger.info("Agentic Workflow Orchestrator shutdown complete")
