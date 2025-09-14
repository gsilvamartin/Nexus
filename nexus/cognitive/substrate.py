"""
Cognitive Substrate

Implementa o substrato cognitivo fundamental do NEXUS, coordenando função executiva,
memória de trabalho e sistemas de decisão.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from nexus.cognitive.working_memory import WorkingMemory
from nexus.cognitive.decision_cortex import DecisionCortex

logger = logging.getLogger(__name__)


@dataclass
class CognitiveState:
    """Estado cognitivo atual do sistema."""
    
    attention_focus: Dict[str, float] = field(default_factory=dict)
    working_memory_load: float = 0.0
    decision_confidence: float = 0.0
    cognitive_load: float = 0.0
    
    # Contexto atual
    active_goals: List[str] = field(default_factory=list)
    current_strategy: Optional[str] = None
    
    # Métricas de performance
    processing_speed: float = 1.0
    accuracy_estimate: float = 0.8
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CognitiveSubstrate:
    """
    Substrato Cognitivo do NEXUS.
    
    Coordena os componentes cognitivos fundamentais e mantém o estado
    cognitivo global do sistema.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o Substrato Cognitivo.
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        self.substrate_id = str(uuid.uuid4())
        
        # Componentes cognitivos
        self.working_memory = WorkingMemory(self.config.get('working_memory', {}))
        self.decision_cortex = DecisionCortex(self.config.get('decision_cortex', {}))
        
        # Estado cognitivo
        self.current_state = CognitiveState(
            attention_focus={},
            working_memory_load=0.0,
            decision_confidence=0.0,
            cognitive_load=0.0,
            active_goals=[]
        )
        
        # Histórico de estados
        self.state_history: List[CognitiveState] = []
        
        # Configurações
        self.max_history_size = self.config.get('max_history_size', 1000)
        self.state_update_interval = self.config.get('state_update_interval', 1.0)
        
        # Task para monitoramento contínuo
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Cognitive Substrate initialized: {self.substrate_id}")
    
    async def initialize(self) -> None:
        """Inicializa o substrato cognitivo."""
        
        # Inicializar componentes
        await self.working_memory.initialize()
        await self.decision_cortex.initialize()
        
        # Iniciar monitoramento contínuo
        self._monitoring_task = asyncio.create_task(self._continuous_monitoring())
        
        logger.info("Cognitive Substrate initialization complete")
    
    async def process_cognitive_cycle(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa um ciclo cognitivo completo.
        
        Args:
            inputs: Entradas para processamento
            
        Returns:
            Resultados do processamento cognitivo
        """
        cycle_start = datetime.utcnow()
        
        # 1. Atualizar memória de trabalho
        await self.working_memory.update_context(inputs)
        
        # 2. Processar decisões
        decision_results = await self.decision_cortex.process_decisions(
            inputs, self.working_memory.get_current_context()
        )
        
        # 3. Atualizar estado cognitivo
        await self._update_cognitive_state(inputs, decision_results)
        
        # 4. Preparar saídas
        outputs = {
            'decisions': decision_results,
            'cognitive_state': self.current_state,
            'working_memory_state': self.working_memory.get_state(),
            'processing_time': (datetime.utcnow() - cycle_start).total_seconds()
        }
        
        return outputs
    
    async def get_cognitive_insights(self) -> Dict[str, Any]:
        """
        Obtém insights sobre o estado cognitivo atual.
        
        Returns:
            Insights cognitivos detalhados
        """
        
        # Análise de tendências
        trends = await self._analyze_cognitive_trends()
        
        # Métricas de performance
        performance_metrics = await self._calculate_performance_metrics()
        
        # Recomendações de otimização
        optimization_recommendations = await self._generate_optimization_recommendations()
        
        insights = {
            'current_state': self.current_state,
            'trends': trends,
            'performance_metrics': performance_metrics,
            'optimization_recommendations': optimization_recommendations,
            'system_health': await self._assess_system_health()
        }
        
        return insights
    
    async def optimize_cognitive_parameters(self) -> None:
        """Otimiza parâmetros cognitivos baseado no histórico."""
        
        # Analisar padrões de performance
        performance_patterns = await self._analyze_performance_patterns()
        
        # Ajustar parâmetros da memória de trabalho
        if performance_patterns.get('memory_bottleneck', False):
            await self.working_memory.optimize_parameters()
        
        # Ajustar parâmetros do córtex de decisão
        if performance_patterns.get('decision_latency', 0) > 1.0:
            await self.decision_cortex.optimize_parameters()
        
        logger.info("Cognitive parameters optimized")
    
    async def _update_cognitive_state(
        self, 
        inputs: Dict[str, Any], 
        decision_results: Dict[str, Any]
    ) -> None:
        """Atualiza o estado cognitivo atual."""
        
        # Calcular carga da memória de trabalho
        working_memory_load = await self.working_memory.calculate_load()
        
        # Calcular confiança das decisões
        decision_confidence = decision_results.get('confidence', 0.0)
        
        # Calcular carga cognitiva total
        cognitive_load = await self._calculate_cognitive_load(
            working_memory_load, decision_confidence
        )
        
        # Atualizar foco de atenção
        attention_focus = await self._calculate_attention_focus(inputs)
        
        # Criar novo estado
        new_state = CognitiveState(
            attention_focus=attention_focus,
            working_memory_load=working_memory_load,
            decision_confidence=decision_confidence,
            cognitive_load=cognitive_load,
            active_goals=inputs.get('active_goals', []),
            current_strategy=inputs.get('current_strategy')
        )
        
        # Atualizar estado atual
        self.current_state = new_state
        
        # Adicionar ao histórico
        self.state_history.append(new_state)
        
        # Manter tamanho do histórico
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
    
    async def _calculate_cognitive_load(
        self, 
        working_memory_load: float, 
        decision_confidence: float
    ) -> float:
        """Calcula a carga cognitiva total."""
        
        # Componentes da carga cognitiva
        memory_component = working_memory_load * 0.4
        decision_component = (1.0 - decision_confidence) * 0.3
        attention_component = len(self.current_state.attention_focus) * 0.1
        
        # Carga base do sistema
        base_load = 0.2
        
        total_load = base_load + memory_component + decision_component + attention_component
        
        return min(total_load, 1.0)
    
    async def _calculate_attention_focus(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Calcula o foco de atenção atual."""
        
        focus_areas = {}
        
        # Foco baseado em objetivos ativos
        active_goals = inputs.get('active_goals', [])
        if active_goals:
            goal_weight = 1.0 / len(active_goals)
            for goal in active_goals:
                focus_areas[f"goal_{goal}"] = goal_weight
        
        # Foco baseado em decisões pendentes
        pending_decisions = inputs.get('pending_decisions', [])
        if pending_decisions:
            decision_weight = 0.5 / len(pending_decisions)
            for decision in pending_decisions:
                focus_areas[f"decision_{decision}"] = decision_weight
        
        # Normalizar pesos
        total_weight = sum(focus_areas.values())
        if total_weight > 0:
            focus_areas = {k: v / total_weight for k, v in focus_areas.items()}
        
        return focus_areas
    
    async def _continuous_monitoring(self) -> None:
        """Monitoramento contínuo do estado cognitivo."""
        
        while True:
            try:
                # Verificar saúde do sistema
                system_health = await self._assess_system_health()
                
                # Otimizar se necessário
                if system_health.get('optimization_needed', False):
                    await self.optimize_cognitive_parameters()
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.state_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cognitive monitoring: {e}", exc_info=True)
                await asyncio.sleep(self.state_update_interval)
    
    async def _analyze_cognitive_trends(self) -> Dict[str, Any]:
        """Analisa tendências no histórico cognitivo."""
        
        if len(self.state_history) < 10:
            return {'insufficient_data': True}
        
        recent_states = self.state_history[-50:]  # Últimos 50 estados
        
        # Tendências de carga cognitiva
        cognitive_loads = [state.cognitive_load for state in recent_states]
        avg_load = sum(cognitive_loads) / len(cognitive_loads)
        load_trend = 'increasing' if cognitive_loads[-1] > avg_load else 'decreasing'
        
        # Tendências de confiança
        confidences = [state.decision_confidence for state in recent_states]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_trend = 'increasing' if confidences[-1] > avg_confidence else 'decreasing'
        
        return {
            'cognitive_load': {
                'average': avg_load,
                'current': self.current_state.cognitive_load,
                'trend': load_trend
            },
            'decision_confidence': {
                'average': avg_confidence,
                'current': self.current_state.decision_confidence,
                'trend': confidence_trend
            },
            'stability': await self._calculate_stability_metric(recent_states)
        }
    
    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calcula métricas de performance cognitiva."""
        
        if not self.state_history:
            return {}
        
        recent_states = self.state_history[-100:]  # Últimos 100 estados
        
        # Métricas básicas
        avg_cognitive_load = sum(s.cognitive_load for s in recent_states) / len(recent_states)
        avg_decision_confidence = sum(s.decision_confidence for s in recent_states) / len(recent_states)
        avg_processing_speed = sum(s.processing_speed for s in recent_states) / len(recent_states)
        
        # Métricas de eficiência
        efficiency = avg_processing_speed * avg_decision_confidence / max(avg_cognitive_load, 0.1)
        
        return {
            'average_cognitive_load': avg_cognitive_load,
            'average_decision_confidence': avg_decision_confidence,
            'average_processing_speed': avg_processing_speed,
            'cognitive_efficiency': efficiency,
            'system_utilization': await self._calculate_system_utilization()
        }
    
    async def _generate_optimization_recommendations(self) -> List[str]:
        """Gera recomendações de otimização."""
        
        recommendations = []
        
        # Verificar carga cognitiva alta
        if self.current_state.cognitive_load > 0.8:
            recommendations.append("Reduce cognitive load by simplifying active goals")
        
        # Verificar baixa confiança nas decisões
        if self.current_state.decision_confidence < 0.6:
            recommendations.append("Improve decision confidence through better context")
        
        # Verificar sobrecarga da memória de trabalho
        if self.current_state.working_memory_load > 0.9:
            recommendations.append("Optimize working memory by clearing unused context")
        
        # Verificar dispersão de atenção
        if len(self.current_state.attention_focus) > 5:
            recommendations.append("Focus attention by reducing concurrent tasks")
        
        return recommendations
    
    async def _assess_system_health(self) -> Dict[str, Any]:
        """Avalia a saúde do sistema cognitivo."""
        
        health_score = 1.0
        issues = []
        
        # Verificar carga cognitiva
        if self.current_state.cognitive_load > 0.9:
            health_score -= 0.3
            issues.append("High cognitive load")
        
        # Verificar confiança nas decisões
        if self.current_state.decision_confidence < 0.5:
            health_score -= 0.2
            issues.append("Low decision confidence")
        
        # Verificar memória de trabalho
        if self.current_state.working_memory_load > 0.95:
            health_score -= 0.2
            issues.append("Working memory overload")
        
        # Verificar estabilidade
        stability = await self._calculate_system_stability()
        if stability < 0.7:
            health_score -= 0.3
            issues.append("System instability")
        
        return {
            'health_score': max(health_score, 0.0),
            'issues': issues,
            'optimization_needed': health_score < 0.7,
            'stability': stability
        }
    
    async def _calculate_stability_metric(self, states: List[CognitiveState]) -> float:
        """Calcula métrica de estabilidade."""
        
        if len(states) < 2:
            return 1.0
        
        # Calcular variância da carga cognitiva
        loads = [state.cognitive_load for state in states]
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        
        # Estabilidade inversamente proporcional à variância
        stability = 1.0 / (1.0 + variance * 10)
        
        return stability
    
    async def _calculate_system_utilization(self) -> float:
        """Calcula utilização do sistema."""
        
        # Utilização baseada na carga cognitiva e atividade
        base_utilization = self.current_state.cognitive_load
        activity_factor = len(self.current_state.active_goals) * 0.1
        
        return min(base_utilization + activity_factor, 1.0)
    
    async def _calculate_system_stability(self) -> float:
        """Calcula estabilidade do sistema."""
        
        if len(self.state_history) < 10:
            return 0.8  # Default stability for new systems
        
        return await self._calculate_stability_metric(self.state_history[-20:])
    
    async def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analisa padrões de performance."""
        
        patterns = {
            'memory_bottleneck': False,
            'decision_latency': 0.0,
            'attention_fragmentation': False
        }
        
        if self.state_history:
            recent_states = self.state_history[-50:]
            
            # Verificar gargalo de memória
            high_memory_load_count = sum(
                1 for state in recent_states if state.working_memory_load > 0.8
            )
            patterns['memory_bottleneck'] = high_memory_load_count > len(recent_states) * 0.3
            
            # Verificar latência de decisão (simplificado)
            low_confidence_count = sum(
                1 for state in recent_states if state.decision_confidence < 0.6
            )
            patterns['decision_latency'] = low_confidence_count / len(recent_states)
            
            # Verificar fragmentação de atenção
            high_focus_count = sum(
                1 for state in recent_states if len(state.attention_focus) > 5
            )
            patterns['attention_fragmentation'] = high_focus_count > len(recent_states) * 0.4
        
        return patterns
    
    async def shutdown(self) -> None:
        """Desliga o substrato cognitivo."""
        
        logger.info("Shutting down Cognitive Substrate")
        
        # Cancelar monitoramento
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Desligar componentes
        await self.working_memory.shutdown()
        await self.decision_cortex.shutdown()
        
        logger.info("Cognitive Substrate shutdown complete")
