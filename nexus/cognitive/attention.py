"""
Attention Controller

Implementa o sistema de controle de atenção do NEXUS, gerenciando foco,
distribuição de recursos cognitivos e mudanças de contexto.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class AttentionMode(Enum):
    """Modos de atenção do sistema."""
    FOCUSED = "focused"           # Foco intenso em uma tarefa
    DISTRIBUTED = "distributed"   # Atenção distribuída entre múltiplas tarefas
    SCANNING = "scanning"         # Varredura para detectar prioridades
    ADAPTIVE = "adaptive"         # Modo adaptativo baseado no contexto


@dataclass
class AttentionTarget:
    """Alvo de atenção com metadados."""
    
    target_id: str
    priority: float
    allocated_resources: float
    focus_duration: timedelta
    
    # Contexto do alvo
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    # Métricas de performance
    performance_score: float = 0.0
    interruption_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_focused: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AttentionState:
    """Estado atual do sistema de atenção."""
    
    mode: AttentionMode
    primary_target: Optional[str]
    active_targets: Dict[str, AttentionTarget]
    
    # Métricas de atenção
    focus_stability: float = 0.0
    switching_frequency: float = 0.0
    cognitive_load: float = 0.0
    
    # Configurações dinâmicas
    interruption_threshold: float = 0.7
    context_switching_cost: float = 0.1
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AttentionController:
    """
    Controlador de Atenção do NEXUS.
    
    Gerencia a distribuição de atenção entre diferentes tarefas e córtices,
    otimizando foco e minimizando custos de mudança de contexto.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o Controlador de Atenção.
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        
        # Estado atual
        self.current_state = AttentionState(
            mode=AttentionMode.ADAPTIVE,
            primary_target=None,
            active_targets={}
        )
        
        # Histórico de estados
        self.attention_history: List[AttentionState] = []
        
        # Configurações
        self.max_concurrent_targets = self.config.get('max_concurrent_targets', 5)
        self.min_focus_duration = self.config.get('min_focus_duration_seconds', 30)
        self.max_focus_duration = self.config.get('max_focus_duration_seconds', 1800)  # 30 min
        
        # Métricas de performance
        self.performance_metrics: Dict[str, float] = {}
        self.switching_history: List[Dict[str, Any]] = []
        
        # Lock para thread safety
        self._attention_lock = asyncio.Lock()
        
        # Task para monitoramento
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Attention Controller initialized")
    
    async def initialize(self) -> None:
        """Inicializa o controlador de atenção."""
        
        # Iniciar monitoramento contínuo
        monitoring_interval = self.config.get('monitoring_interval_seconds', 5)
        self._monitoring_task = asyncio.create_task(
            self._continuous_monitoring(monitoring_interval)
        )
        
        logger.info("Attention Controller initialization complete")
    
    async def distribute_attention(
        self, 
        priority_weights: Dict[str, float],
        complexity_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Distribui atenção baseado em prioridades e complexidade.
        
        Args:
            priority_weights: Pesos de prioridade por área
            complexity_weights: Pesos de complexidade por área
            
        Returns:
            Distribuição de atenção normalizada
        """
        async with self._attention_lock:
            
            # Combinar pesos de prioridade e complexidade
            combined_weights = {}
            all_areas = set(priority_weights.keys()) | set(complexity_weights.keys())
            
            for area in all_areas:
                priority_weight = priority_weights.get(area, 0.0)
                complexity_weight = complexity_weights.get(area, 0.0)
                
                # Combinar pesos (prioridade tem peso maior)
                combined_weight = (priority_weight * 0.7) + (complexity_weight * 0.3)
                combined_weights[area] = combined_weight
            
            # Normalizar distribuição
            total_weight = sum(combined_weights.values())
            if total_weight == 0:
                # Distribuição uniforme se não há pesos
                uniform_weight = 1.0 / len(combined_weights) if combined_weights else 0.0
                return {area: uniform_weight for area in combined_weights}
            
            normalized_distribution = {
                area: weight / total_weight 
                for area, weight in combined_weights.items()
            }
            
            # Atualizar alvos de atenção
            await self._update_attention_targets(normalized_distribution)
            
            logger.debug(f"Attention distributed across {len(normalized_distribution)} areas")
            return normalized_distribution
    
    async def focus_on_target(self, target_id: str, duration_seconds: Optional[int] = None) -> bool:
        """
        Foca intensamente em um alvo específico.
        
        Args:
            target_id: ID do alvo para focar
            duration_seconds: Duração do foco (opcional)
            
        Returns:
            True se foco foi estabelecido com sucesso
        """
        async with self._attention_lock:
            
            # Verificar se alvo existe
            if target_id not in self.current_state.active_targets:
                logger.warning(f"Target not found for focus: {target_id}")
                return False
            
            # Calcular duração do foco
            if duration_seconds is None:
                target = self.current_state.active_targets[target_id]
                duration_seconds = min(
                    max(target.priority * 300, self.min_focus_duration),
                    self.max_focus_duration
                )
            
            # Mudar para modo focado
            previous_mode = self.current_state.mode
            self.current_state.mode = AttentionMode.FOCUSED
            self.current_state.primary_target = target_id
            
            # Atualizar alvo
            target = self.current_state.active_targets[target_id]
            target.allocated_resources = 0.8  # Alocar 80% dos recursos
            target.focus_duration = timedelta(seconds=duration_seconds)
            target.last_focused = datetime.utcnow()
            
            # Reduzir recursos dos outros alvos
            remaining_resources = 0.2
            other_targets = [t for tid, t in self.current_state.active_targets.items() if tid != target_id]
            
            if other_targets:
                resource_per_target = remaining_resources / len(other_targets)
                for other_target in other_targets:
                    other_target.allocated_resources = resource_per_target
            
            # Registrar mudança de foco
            await self._record_attention_switch(previous_mode, AttentionMode.FOCUSED, target_id)
            
            logger.info(f"Focused attention on target: {target_id} for {duration_seconds}s")
            return True
    
    async def switch_to_distributed_mode(self) -> None:
        """Muda para modo de atenção distribuída."""
        
        async with self._attention_lock:
            
            previous_mode = self.current_state.mode
            self.current_state.mode = AttentionMode.DISTRIBUTED
            self.current_state.primary_target = None
            
            # Redistribuir recursos igualmente
            if self.current_state.active_targets:
                resource_per_target = 1.0 / len(self.current_state.active_targets)
                for target in self.current_state.active_targets.values():
                    target.allocated_resources = resource_per_target
            
            # Registrar mudança
            await self._record_attention_switch(previous_mode, AttentionMode.DISTRIBUTED)
            
            logger.info("Switched to distributed attention mode")
    
    async def handle_interruption(self, interruption_priority: float) -> bool:
        """
        Lida com uma interrupção baseada na prioridade.
        
        Args:
            interruption_priority: Prioridade da interrupção (0-1)
            
        Returns:
            True se interrupção foi aceita
        """
        async with self._attention_lock:
            
            # Verificar se interrupção deve ser aceita
            should_accept = await self._should_accept_interruption(interruption_priority)
            
            if should_accept:
                # Aceitar interrupção
                if self.current_state.primary_target:
                    target = self.current_state.active_targets[self.current_state.primary_target]
                    target.interruption_count += 1
                
                # Mudar para modo de varredura temporariamente
                await self._temporary_scanning_mode()
                
                logger.info(f"Accepted interruption with priority {interruption_priority}")
                return True
            else:
                logger.debug(f"Rejected interruption with priority {interruption_priority}")
                return False
    
    async def optimize_attention_parameters(self) -> None:
        """Otimiza parâmetros de atenção baseado no histórico."""
        
        # Analisar padrões de performance
        performance_analysis = await self._analyze_attention_performance()
        
        # Ajustar threshold de interrupção
        if performance_analysis.get('excessive_interruptions', False):
            self.current_state.interruption_threshold = min(
                self.current_state.interruption_threshold + 0.1, 0.9
            )
            logger.info(f"Increased interruption threshold to {self.current_state.interruption_threshold}")
        
        # Ajustar custo de mudança de contexto
        if performance_analysis.get('frequent_switching', False):
            self.current_state.context_switching_cost = min(
                self.current_state.context_switching_cost + 0.05, 0.5
            )
            logger.info(f"Increased context switching cost to {self.current_state.context_switching_cost}")
        
        # Otimizar duração de foco
        optimal_focus_duration = performance_analysis.get('optimal_focus_duration')
        if optimal_focus_duration:
            self.min_focus_duration = max(optimal_focus_duration * 0.5, 10)
            self.max_focus_duration = min(optimal_focus_duration * 2, 3600)
    
    async def get_attention_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas detalhadas de atenção.
        
        Returns:
            Métricas de atenção e performance
        """
        async with self._attention_lock:
            
            # Calcular métricas atuais
            current_metrics = await self._calculate_current_metrics()
            
            # Métricas históricas
            historical_metrics = await self._calculate_historical_metrics()
            
            return {
                'current_state': self.current_state,
                'current_metrics': current_metrics,
                'historical_metrics': historical_metrics,
                'active_targets_count': len(self.current_state.active_targets),
                'primary_target': self.current_state.primary_target,
                'recommendations': await self._generate_attention_recommendations()
            }
    
    async def _update_attention_targets(self, distribution: Dict[str, float]) -> None:
        """Atualiza alvos de atenção baseado na distribuição."""
        
        current_time = datetime.utcnow()
        
        # Atualizar alvos existentes e criar novos
        for area, attention_weight in distribution.items():
            
            if area in self.current_state.active_targets:
                # Atualizar alvo existente
                target = self.current_state.active_targets[area]
                target.allocated_resources = attention_weight
                target.priority = attention_weight
            else:
                # Criar novo alvo
                target = AttentionTarget(
                    target_id=area,
                    priority=attention_weight,
                    allocated_resources=attention_weight,
                    focus_duration=timedelta(seconds=self.min_focus_duration)
                )
                self.current_state.active_targets[area] = target
        
        # Remover alvos que não estão mais na distribuição
        targets_to_remove = [
            target_id for target_id in self.current_state.active_targets
            if target_id not in distribution
        ]
        
        for target_id in targets_to_remove:
            del self.current_state.active_targets[target_id]
            if self.current_state.primary_target == target_id:
                self.current_state.primary_target = None
        
        # Limitar número de alvos simultâneos
        if len(self.current_state.active_targets) > self.max_concurrent_targets:
            await self._consolidate_targets()
    
    async def _consolidate_targets(self) -> None:
        """Consolida alvos quando há muitos ativos."""
        
        # Ordenar alvos por prioridade
        sorted_targets = sorted(
            self.current_state.active_targets.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        # Manter apenas os top N alvos
        targets_to_keep = dict(sorted_targets[:self.max_concurrent_targets])
        
        # Somar recursos dos alvos removidos e redistribuir
        removed_resources = sum(
            target.allocated_resources 
            for target_id, target in self.current_state.active_targets.items()
            if target_id not in targets_to_keep
        )
        
        # Atualizar alvos mantidos
        self.current_state.active_targets = targets_to_keep
        
        # Redistribuir recursos removidos
        if targets_to_keep and removed_resources > 0:
            bonus_per_target = removed_resources / len(targets_to_keep)
            for target in targets_to_keep.values():
                target.allocated_resources += bonus_per_target
        
        logger.debug(f"Consolidated to {len(targets_to_keep)} attention targets")
    
    async def _should_accept_interruption(self, interruption_priority: float) -> bool:
        """Determina se uma interrupção deve ser aceita."""
        
        # Verificar threshold básico
        if interruption_priority < self.current_state.interruption_threshold:
            return False
        
        # Considerar modo atual
        if self.current_state.mode == AttentionMode.FOCUSED:
            # Modo focado é mais resistente a interrupções
            if self.current_state.primary_target:
                primary_target = self.current_state.active_targets[self.current_state.primary_target]
                # Aceitar apenas se prioridade da interrupção for significativamente maior
                return interruption_priority > (primary_target.priority + 0.2)
        
        # Considerar histórico de interrupções recentes
        recent_interruptions = len([
            switch for switch in self.switching_history[-10:]
            if switch.get('reason') == 'interruption'
        ])
        
        if recent_interruptions > 3:
            # Muitas interrupções recentes, ser mais seletivo
            return interruption_priority > 0.8
        
        return True
    
    async def _temporary_scanning_mode(self) -> None:
        """Ativa modo de varredura temporário."""
        
        previous_mode = self.current_state.mode
        self.current_state.mode = AttentionMode.SCANNING
        
        # Agendar retorno ao modo anterior após breve período
        async def return_to_previous_mode():
            await asyncio.sleep(5)  # 5 segundos de varredura
            async with self._attention_lock:
                if self.current_state.mode == AttentionMode.SCANNING:
                    self.current_state.mode = previous_mode
        
        asyncio.create_task(return_to_previous_mode())
    
    async def _record_attention_switch(
        self, 
        from_mode: AttentionMode, 
        to_mode: AttentionMode,
        target_id: Optional[str] = None
    ) -> None:
        """Registra uma mudança de atenção."""
        
        switch_record = {
            'timestamp': datetime.utcnow(),
            'from_mode': from_mode,
            'to_mode': to_mode,
            'target_id': target_id,
            'switching_cost': self.current_state.context_switching_cost
        }
        
        self.switching_history.append(switch_record)
        
        # Manter histórico limitado
        if len(self.switching_history) > 100:
            self.switching_history = self.switching_history[-100:]
        
        # Atualizar frequência de mudança
        await self._update_switching_frequency()
    
    async def _update_switching_frequency(self) -> None:
        """Atualiza frequência de mudança de contexto."""
        
        if len(self.switching_history) < 2:
            self.current_state.switching_frequency = 0.0
            return
        
        # Calcular switches por minuto na última hora
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_switches = [
            switch for switch in self.switching_history
            if switch['timestamp'] > one_hour_ago
        ]
        
        self.current_state.switching_frequency = len(recent_switches) / 60.0  # switches per minute
    
    async def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calcula métricas atuais de atenção."""
        
        metrics = {
            'focus_stability': await self._calculate_focus_stability(),
            'resource_utilization': await self._calculate_resource_utilization(),
            'target_performance': await self._calculate_target_performance(),
            'attention_efficiency': await self._calculate_attention_efficiency()
        }
        
        return metrics
    
    async def _calculate_focus_stability(self) -> float:
        """Calcula estabilidade do foco."""
        
        if len(self.switching_history) < 5:
            return 1.0  # Assume estabilidade alta para sistemas novos
        
        # Estabilidade baseada na frequência de mudanças
        recent_switches = self.switching_history[-20:]  # Últimas 20 mudanças
        
        if not recent_switches:
            return 1.0
        
        # Calcular intervalo médio entre mudanças
        intervals = []
        for i in range(1, len(recent_switches)):
            interval = (recent_switches[i]['timestamp'] - recent_switches[i-1]['timestamp']).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 1.0
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Estabilidade inversamente proporcional à frequência de mudança
        # Intervalos maiores = maior estabilidade
        stability = min(avg_interval / 300.0, 1.0)  # Normalizar por 5 minutos
        
        return stability
    
    async def _calculate_resource_utilization(self) -> float:
        """Calcula utilização de recursos de atenção."""
        
        if not self.current_state.active_targets:
            return 0.0
        
        total_allocated = sum(
            target.allocated_resources 
            for target in self.current_state.active_targets.values()
        )
        
        return min(total_allocated, 1.0)
    
    async def _calculate_target_performance(self) -> float:
        """Calcula performance média dos alvos."""
        
        if not self.current_state.active_targets:
            return 0.0
        
        total_performance = sum(
            target.performance_score 
            for target in self.current_state.active_targets.values()
        )
        
        return total_performance / len(self.current_state.active_targets)
    
    async def _calculate_attention_efficiency(self) -> float:
        """Calcula eficiência geral da atenção."""
        
        stability = await self._calculate_focus_stability()
        utilization = await self._calculate_resource_utilization()
        performance = await self._calculate_target_performance()
        
        # Eficiência como média ponderada
        efficiency = (stability * 0.4) + (utilization * 0.3) + (performance * 0.3)
        
        return efficiency
    
    async def _calculate_historical_metrics(self) -> Dict[str, Any]:
        """Calcula métricas históricas."""
        
        return {
            'total_switches': len(self.switching_history),
            'average_switching_frequency': self.current_state.switching_frequency,
            'mode_distribution': await self._calculate_mode_distribution(),
            'interruption_acceptance_rate': await self._calculate_interruption_rate()
        }
    
    async def _calculate_mode_distribution(self) -> Dict[str, float]:
        """Calcula distribuição de tempo por modo."""
        
        if not self.attention_history:
            return {}
        
        mode_durations = {mode.value: 0.0 for mode in AttentionMode}
        
        for i in range(len(self.attention_history) - 1):
            current_state = self.attention_history[i]
            next_state = self.attention_history[i + 1]
            
            duration = (next_state.timestamp - current_state.timestamp).total_seconds()
            mode_durations[current_state.mode.value] += duration
        
        total_duration = sum(mode_durations.values())
        if total_duration == 0:
            return mode_durations
        
        return {mode: duration / total_duration for mode, duration in mode_durations.items()}
    
    async def _calculate_interruption_rate(self) -> float:
        """Calcula taxa de aceitação de interrupções."""
        
        interruption_switches = [
            switch for switch in self.switching_history
            if switch.get('reason') == 'interruption'
        ]
        
        if not self.switching_history:
            return 0.0
        
        return len(interruption_switches) / len(self.switching_history)
    
    async def _analyze_attention_performance(self) -> Dict[str, Any]:
        """Analisa performance da atenção."""
        
        analysis = {
            'excessive_interruptions': False,
            'frequent_switching': False,
            'optimal_focus_duration': None
        }
        
        # Verificar interrupções excessivas
        if self.current_state.switching_frequency > 2.0:  # Mais de 2 switches por minuto
            analysis['excessive_interruptions'] = True
        
        # Verificar mudanças frequentes
        if len(self.switching_history) > 50:
            recent_switches = self.switching_history[-50:]
            avg_interval = sum(
                (recent_switches[i]['timestamp'] - recent_switches[i-1]['timestamp']).total_seconds()
                for i in range(1, len(recent_switches))
            ) / (len(recent_switches) - 1)
            
            if avg_interval < 60:  # Menos de 1 minuto entre mudanças
                analysis['frequent_switching'] = True
        
        # Calcular duração ótima de foco
        if self.current_state.active_targets:
            focus_durations = [
                target.focus_duration.total_seconds()
                for target in self.current_state.active_targets.values()
                if target.performance_score > 0.7
            ]
            
            if focus_durations:
                analysis['optimal_focus_duration'] = sum(focus_durations) / len(focus_durations)
        
        return analysis
    
    async def _generate_attention_recommendations(self) -> List[str]:
        """Gera recomendações para otimização da atenção."""
        
        recommendations = []
        
        # Verificar estabilidade do foco
        stability = await self._calculate_focus_stability()
        if stability < 0.6:
            recommendations.append("Increase focus stability by reducing context switching")
        
        # Verificar utilização de recursos
        utilization = await self._calculate_resource_utilization()
        if utilization < 0.7:
            recommendations.append("Improve resource utilization by consolidating attention targets")
        
        # Verificar frequência de mudança
        if self.current_state.switching_frequency > 1.5:
            recommendations.append("Reduce switching frequency by increasing interruption threshold")
        
        # Verificar número de alvos
        if len(self.current_state.active_targets) > self.max_concurrent_targets * 0.8:
            recommendations.append("Consider reducing number of concurrent attention targets")
        
        return recommendations
    
    async def _continuous_monitoring(self, interval_seconds: int) -> None:
        """Monitoramento contínuo da atenção."""
        
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                
                # Salvar estado atual no histórico
                async with self._attention_lock:
                    self.attention_history.append(self.current_state)
                    
                    # Manter histórico limitado
                    if len(self.attention_history) > 1000:
                        self.attention_history = self.attention_history[-1000:]
                
                # Atualizar métricas de performance
                await self._update_performance_metrics()
                
                # Otimização automática se necessário
                efficiency = await self._calculate_attention_efficiency()
                if efficiency < 0.6:
                    await self.optimize_attention_parameters()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in attention monitoring: {e}", exc_info=True)
    
    async def _update_performance_metrics(self) -> None:
        """Atualiza métricas de performance."""
        
        current_metrics = await self._calculate_current_metrics()
        self.performance_metrics.update(current_metrics)
        
        # Atualizar performance dos alvos (simulado)
        for target in self.current_state.active_targets.values():
            # Performance baseada em recursos alocados e estabilidade
            base_performance = target.allocated_resources
            stability_bonus = await self._calculate_focus_stability() * 0.2
            interruption_penalty = min(target.interruption_count * 0.1, 0.3)
            
            target.performance_score = max(
                base_performance + stability_bonus - interruption_penalty, 0.0
            )
    
    async def shutdown(self) -> None:
        """Desliga o controlador de atenção."""
        
        logger.info("Shutting down Attention Controller")
        
        # Cancelar monitoramento
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Limpar estado
        async with self._attention_lock:
            self.current_state.active_targets.clear()
            self.attention_history.clear()
            self.switching_history.clear()
            self.performance_metrics.clear()
        
        logger.info("Attention Controller shutdown complete")
