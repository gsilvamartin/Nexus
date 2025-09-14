"""
Meta-Cognitive Controller

Implementa o sistema de meta-cognição do NEXUS, responsável por monitoramento
e controle dos próprios processos cognitivos.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Tipos de insights meta-cognitivos."""
    PERFORMANCE_ANALYSIS = "performance_analysis"
    COGNITIVE_LOAD = "cognitive_load"
    DECISION_QUALITY = "decision_quality"
    LEARNING_EFFECTIVENESS = "learning_effectiveness"
    ATTENTION_PATTERNS = "attention_patterns"
    ERROR_ANALYSIS = "error_analysis"
    ADAPTATION_NEED = "adaptation_need"


class CognitiveState(Enum):
    """Estados cognitivos do sistema."""
    OPTIMAL = "optimal"
    STRESSED = "stressed"
    OVERLOADED = "overloaded"
    UNDERUTILIZED = "underutilized"
    CONFUSED = "confused"
    FOCUSED = "focused"
    SCATTERED = "scattered"


@dataclass
class MetaCognitiveInsight:
    """Insight meta-cognitivo sobre o sistema."""
    
    insight_type: InsightType
    confidence: float
    reasoning: List[str]
    recommendations: List[str]
    severity: str = "info"  # info, warning, critical
    impact_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Contexto adicional
    context: Dict[str, Any] = field(default_factory=dict)
    related_insights: List[str] = field(default_factory=list)


@dataclass
class CognitiveMetrics:
    """Métricas cognitivas do sistema."""
    
    # Performance geral
    overall_efficiency: float = 0.0
    decision_accuracy: float = 0.0
    learning_rate: float = 0.0
    
    # Carga cognitiva
    cognitive_load: float = 0.0
    attention_stability: float = 0.0
    context_switching_frequency: float = 0.0
    
    # Qualidade de processamento
    error_rate: float = 0.0
    confidence_variance: float = 0.0
    reasoning_depth: float = 0.0
    
    # Adaptabilidade
    adaptation_speed: float = 0.0
    pattern_recognition_accuracy: float = 0.0
    generalization_ability: float = 0.0
    
    # Timestamps
    measured_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CognitiveProfile:
    """Perfil cognitivo do sistema."""
    
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    learning_preferences: Dict[str, float] = field(default_factory=dict)
    optimal_conditions: Dict[str, Any] = field(default_factory=dict)
    adaptation_strategies: List[str] = field(default_factory=list)


class MetaCognitiveController:
    """
    Controlador Meta-Cognitivo do NEXUS.
    
    Monitora e controla os próprios processos cognitivos do sistema,
    fornecendo insights sobre performance e recomendações de melhoria.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializa o controlador meta-cognitivo."""
        self.config = config or {}
        
        # Estado interno
        self.current_state = CognitiveState.OPTIMAL
        self.cognitive_profile = CognitiveProfile()
        self.metrics_history: List[CognitiveMetrics] = []
        self.insights_history: List[MetaCognitiveInsight] = []
        
        # Configurações
        self.monitoring_interval = self.config.get('monitoring_interval', 30)  # segundos
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.7)
        self.performance_window = self.config.get('performance_window', 100)  # operações
        
        # Task de monitoramento
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        logger.info("Meta-Cognitive Controller initialized")
    
    async def initialize(self) -> None:
        """Inicializa o controlador meta-cognitivo."""
        
        # Iniciar monitoramento contínuo
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(
            self._continuous_monitoring()
        )
        
        logger.info("Meta-Cognitive Controller initialization complete")
    
    async def analyze_complexity_reasoning(
        self, 
        objective: Any, 
        dimensions: Dict[str, float]
    ) -> List[str]:
        """
        Analisa o raciocínio por trás da análise de complexidade.
        
        Args:
            objective: Objetivo analisado
            dimensions: Dimensões de complexidade
            
        Returns:
            Lista de justificativas do raciocínio
        """
        reasoning = []
        
        # Análise dimensional detalhada
        for dimension, value in dimensions.items():
            if value > 0.8:
                reasoning.append(f"Very high {dimension} detected ({value:.2f}) - requires significant resources")
            elif value > 0.6:
                reasoning.append(f"High {dimension} detected ({value:.2f}) - moderate resource allocation needed")
            elif value > 0.4:
                reasoning.append(f"Moderate {dimension} detected ({value:.2f}) - standard processing")
            elif value > 0.2:
                reasoning.append(f"Low {dimension} detected ({value:.2f}) - minimal resource requirements")
            else:
                reasoning.append(f"Very low {dimension} detected ({value:.2f}) - trivial processing")
        
        # Análise contextual avançada
        if hasattr(objective, 'description'):
            text = objective.description.lower()
            
            # Análise de complexidade linguística
            complexity_indicators = {
                'complex': ['complex', 'sophisticated', 'advanced', 'intricate'],
                'uncertainty': ['maybe', 'possibly', 'might', 'could', 'uncertain'],
                'integration': ['integrate', 'connect', 'combine', 'merge', 'link'],
                'optimization': ['optimize', 'improve', 'enhance', 'maximize', 'minimize'],
                'scaling': ['scale', 'expand', 'grow', 'increase', 'multiply']
            }
            
            for category, keywords in complexity_indicators.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches > 0:
                    reasoning.append(f"Found {matches} {category} indicators in description")
        
        # Análise de interdependências
        if hasattr(objective, 'requirements') and objective.requirements:
            reasoning.append(f"Objective has {len(objective.requirements)} explicit requirements")
        
        if hasattr(objective, 'constraints') and objective.constraints:
            reasoning.append(f"Objective has {len(objective.constraints)} constraints")
        
        # Análise de prazo
        if hasattr(objective, 'deadline') and objective.deadline:
            time_remaining = objective.deadline - datetime.utcnow()
            if time_remaining.days < 1:
                reasoning.append("Urgent deadline - high priority processing required")
            elif time_remaining.days < 7:
                reasoning.append("Short deadline - accelerated processing needed")
            elif time_remaining.days < 30:
                reasoning.append("Moderate timeline - standard processing")
            else:
                reasoning.append("Long timeline - can afford thorough analysis")
        
        return reasoning
    
    async def assess_cognitive_state(self, current_metrics: CognitiveMetrics) -> CognitiveState:
        """
        Avalia o estado cognitivo atual do sistema.
        
        Args:
            current_metrics: Métricas cognitivas atuais
            
        Returns:
            Estado cognitivo identificado
        """
        
        # Análise de sobrecarga cognitiva
        if current_metrics.cognitive_load > 0.8:
            if current_metrics.error_rate > 0.3:
                return CognitiveState.OVERLOADED
            else:
                return CognitiveState.STRESSED
        
        # Análise de subutilização
        if current_metrics.cognitive_load < 0.2 and current_metrics.overall_efficiency < 0.5:
            return CognitiveState.UNDERUTILIZED
        
        # Análise de confusão
        if current_metrics.confidence_variance > 0.5 and current_metrics.error_rate > 0.2:
            return CognitiveState.CONFUSED
        
        # Análise de foco
        if current_metrics.attention_stability > 0.8 and current_metrics.context_switching_frequency < 0.1:
            return CognitiveState.FOCUSED
        
        # Análise de dispersão
        if current_metrics.context_switching_frequency > 0.5:
            return CognitiveState.SCATTERED
        
        # Estado ótimo
        if (current_metrics.overall_efficiency > 0.7 and 
            current_metrics.cognitive_load > 0.3 and 
            current_metrics.cognitive_load < 0.7 and
            current_metrics.error_rate < 0.1):
            return CognitiveState.OPTIMAL
        
        return CognitiveState.OPTIMAL  # Default
    
    async def generate_insights(self, metrics: CognitiveMetrics) -> List[MetaCognitiveInsight]:
        """
        Gera insights meta-cognitivos baseados nas métricas.
        
        Args:
            metrics: Métricas cognitivas atuais
            
        Returns:
            Lista de insights gerados
        """
        insights = []
        
        # Insight de performance
        if metrics.overall_efficiency < 0.6:
            insights.append(MetaCognitiveInsight(
                insight_type=InsightType.PERFORMANCE_ANALYSIS,
                confidence=0.8,
                reasoning=[
                    f"Overall efficiency is {metrics.overall_efficiency:.2f}",
                    "Performance below optimal threshold"
                ],
                recommendations=[
                    "Review resource allocation",
                    "Consider task prioritization",
                    "Check for bottlenecks"
                ],
                severity="warning",
                impact_score=0.7
            ))
        
        # Insight de carga cognitiva
        if metrics.cognitive_load > 0.8:
            insights.append(MetaCognitiveInsight(
                insight_type=InsightType.COGNITIVE_LOAD,
                confidence=0.9,
                reasoning=[
                    f"Cognitive load is {metrics.cognitive_load:.2f}",
                    "System approaching capacity limits"
                ],
                recommendations=[
                    "Reduce concurrent tasks",
                    "Implement task queuing",
                    "Consider resource scaling"
                ],
                severity="critical",
                impact_score=0.9
            ))
        
        # Insight de qualidade de decisão
        if metrics.decision_accuracy < 0.7:
            insights.append(MetaCognitiveInsight(
                insight_type=InsightType.DECISION_QUALITY,
                confidence=0.75,
                reasoning=[
                    f"Decision accuracy is {metrics.decision_accuracy:.2f}",
                    "Decision quality below expected level"
                ],
                recommendations=[
                    "Improve data quality",
                    "Enhance reasoning algorithms",
                    "Increase confidence thresholds"
                ],
                severity="warning",
                impact_score=0.6
            ))
        
        # Insight de taxa de erro
        if metrics.error_rate > 0.15:
            insights.append(MetaCognitiveInsight(
                insight_type=InsightType.ERROR_ANALYSIS,
                confidence=0.85,
                reasoning=[
                    f"Error rate is {metrics.error_rate:.2f}",
                    "Error rate above acceptable threshold"
                ],
                recommendations=[
                    "Implement additional validation",
                    "Review error handling procedures",
                    "Increase monitoring frequency"
                ],
                severity="critical",
                impact_score=0.8
            ))
        
        # Insight de adaptabilidade
        if metrics.adaptation_speed < 0.5:
            insights.append(MetaCognitiveInsight(
                insight_type=InsightType.ADAPTATION_NEED,
                confidence=0.7,
                reasoning=[
                    f"Adaptation speed is {metrics.adaptation_speed:.2f}",
                    "System adapting slowly to changes"
                ],
                recommendations=[
                    "Increase learning rate",
                    "Improve pattern recognition",
                    "Enhance feedback mechanisms"
                ],
                severity="info",
                impact_score=0.4
            ))
        
        return insights
    
    async def update_cognitive_profile(self, insights: List[MetaCognitiveInsight]) -> None:
        """
        Atualiza o perfil cognitivo baseado nos insights.
        
        Args:
            insights: Lista de insights para análise
        """
        
        # Analisar padrões de força
        performance_insights = [i for i in insights if i.insight_type == InsightType.PERFORMANCE_ANALYSIS]
        if performance_insights:
            avg_confidence = np.mean([i.confidence for i in performance_insights])
            if avg_confidence > 0.8:
                if "High Performance" not in self.cognitive_profile.strengths:
                    self.cognitive_profile.strengths.append("High Performance")
        
        # Analisar padrões de fraqueza
        error_insights = [i for i in insights if i.insight_type == InsightType.ERROR_ANALYSIS]
        if error_insights:
            avg_impact = np.mean([i.impact_score for i in error_insights])
            if avg_impact > 0.7:
                if "Error Prone" not in self.cognitive_profile.weaknesses:
                    self.cognitive_profile.weaknesses.append("Error Prone")
        
        # Atualizar preferências de aprendizado
        learning_insights = [i for i in insights if i.insight_type == InsightType.LEARNING_EFFECTIVENESS]
        if learning_insights:
            # Ajustar preferências baseado em insights
            self.cognitive_profile.learning_preferences.update({
                'adaptation_rate': 0.1,
                'pattern_recognition_weight': 0.8,
                'feedback_sensitivity': 0.7
            })
    
    async def recommend_adaptations(self, current_state: CognitiveState) -> List[str]:
        """
        Recomenda adaptações baseadas no estado cognitivo atual.
        
        Args:
            current_state: Estado cognitivo atual
            
        Returns:
            Lista de recomendações de adaptação
        """
        recommendations = []
        
        if current_state == CognitiveState.OVERLOADED:
            recommendations.extend([
                "Reduce cognitive load by 30%",
                "Implement task prioritization",
                "Activate resource scaling",
                "Enable error recovery mode"
            ])
        
        elif current_state == CognitiveState.STRESSED:
            recommendations.extend([
                "Optimize resource allocation",
                "Reduce context switching",
                "Implement task batching",
                "Monitor performance closely"
            ])
        
        elif current_state == CognitiveState.UNDERUTILIZED:
            recommendations.extend([
                "Increase task complexity",
                "Activate additional capabilities",
                "Explore new problem domains",
                "Optimize for efficiency"
            ])
        
        elif current_state == CognitiveState.CONFUSED:
            recommendations.extend([
                "Improve data quality",
                "Enhance reasoning algorithms",
                "Reduce ambiguity in inputs",
                "Implement confidence checking"
            ])
        
        elif current_state == CognitiveState.SCATTERED:
            recommendations.extend([
                "Implement attention management",
                "Reduce task switching",
                "Focus on priority tasks",
                "Optimize workflow"
            ])
        
        return recommendations
    
    async def _continuous_monitoring(self) -> None:
        """Monitoramento contínuo do estado cognitivo."""
        
        while self._is_monitoring:
            try:
                # Coletar métricas atuais
                current_metrics = await self._collect_cognitive_metrics()
                
                # Avaliar estado cognitivo
                new_state = await self.assess_cognitive_state(current_metrics)
                if new_state != self.current_state:
                    logger.info(f"Cognitive state changed: {self.current_state} -> {new_state}")
                    self.current_state = new_state
                
                # Gerar insights
                insights = await self.generate_insights(current_metrics)
                if insights:
                    self.insights_history.extend(insights)
                    await self.update_cognitive_profile(insights)
                
                # Armazenar métricas
                self.metrics_history.append(current_metrics)
                
                # Manter histórico limitado
                if len(self.metrics_history) > self.performance_window:
                    self.metrics_history = self.metrics_history[-self.performance_window:]
                
                if len(self.insights_history) > 1000:
                    self.insights_history = self.insights_history[-1000:]
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}", exc_info=True)
                await asyncio.sleep(60)  # Aguardar antes de tentar novamente
    
    async def _collect_cognitive_metrics(self) -> CognitiveMetrics:
        """Coleta métricas cognitivas atuais."""
        
        # Implementação simplificada - em produção, coletar de sistemas reais
        return CognitiveMetrics(
            overall_efficiency=0.75,
            decision_accuracy=0.82,
            learning_rate=0.15,
            cognitive_load=0.45,
            attention_stability=0.70,
            context_switching_frequency=0.25,
            error_rate=0.08,
            confidence_variance=0.20,
            reasoning_depth=0.65,
            adaptation_speed=0.60,
            pattern_recognition_accuracy=0.78,
            generalization_ability=0.70
        )
    
    async def get_cognitive_status(self) -> Dict[str, Any]:
        """
        Obtém status cognitivo completo do sistema.
        
        Returns:
            Status cognitivo detalhado
        """
        
        current_metrics = self.metrics_history[-1] if self.metrics_history else await self._collect_cognitive_metrics()
        
        return {
            'current_state': self.current_state.value,
            'cognitive_profile': {
                'strengths': self.cognitive_profile.strengths,
                'weaknesses': self.cognitive_profile.weaknesses,
                'learning_preferences': self.cognitive_profile.learning_preferences
            },
            'current_metrics': {
                'overall_efficiency': current_metrics.overall_efficiency,
                'cognitive_load': current_metrics.cognitive_load,
                'error_rate': current_metrics.error_rate,
                'decision_accuracy': current_metrics.decision_accuracy
            },
            'recent_insights': [
                {
                    'type': insight.insight_type.value,
                    'confidence': insight.confidence,
                    'severity': insight.severity,
                    'timestamp': insight.timestamp.isoformat()
                }
                for insight in self.insights_history[-10:]  # Últimos 10 insights
            ],
            'recommendations': await self.recommend_adaptations(self.current_state)
        }
    
    async def shutdown(self) -> None:
        """Desliga o controlador meta-cognitivo."""
        
        logger.info("Shutting down Meta-Cognitive Controller")
        
        # Parar monitoramento
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Limpar dados
        self.metrics_history.clear()
        self.insights_history.clear()
        
        logger.info("Meta-Cognitive Controller shutdown complete")
