"""
Enhanced Decision Cortex

Implementa o córtex de decisão do substrato cognitivo com capacidades avançadas
de tomada de decisão, avaliação de riscos e raciocínio estratégico.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Tipos de decisão."""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    EMERGENCY = "emergency"


@dataclass
class DecisionOption:
    """Opção de decisão."""
    id: str
    description: str
    expected_outcome: Dict[str, Any]
    confidence: float
    risk_level: float
    resource_cost: float
    time_cost: float
    dependencies: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    """Resultado de uma decisão."""
    decision_id: str
    selected_option: DecisionOption
    reasoning: List[str]
    confidence: float
    risk_assessment: Dict[str, float]
    expected_benefits: Dict[str, Any]
    implementation_plan: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DecisionCortex:
    """
    Córtex de Decisão Avançado do NEXUS.
    
    Implementa capacidades avançadas de tomada de decisão com análise de riscos,
    raciocínio estratégico e otimização multi-objetivo.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializa o córtex de decisão."""
        self.config = config or {}
        
        # Configurações
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.risk_tolerance = self.config.get('risk_tolerance', 0.5)
        self.max_decision_time = self.config.get('max_decision_time', 30.0)
        
        # Estado interno
        self.active_decisions: Dict[str, DecisionResult] = {}
        self.decision_history: List[DecisionResult] = []
        self.decision_patterns: Dict[str, Any] = {}
        
        # Métricas de performance
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'average_confidence': 0.0,
            'average_decision_time': 0.0
        }
        
        logger.info("Enhanced Decision Cortex initialized")
    
    async def initialize(self) -> None:
        """Inicializa o córtex de decisão."""
        logger.info("Enhanced Decision Cortex initialization complete")
    
    async def process_decisions(
        self, 
        inputs: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processa decisões baseado em entradas e contexto.
        
        Args:
            inputs: Entradas para decisão
            context: Contexto atual
            
        Returns:
            Resultados das decisões
        """
        start_time = datetime.utcnow()
        
        # Identificar tipo de decisão necessária
        decision_type = await self._identify_decision_type(inputs, context)
        
        # Gerar opções de decisão
        options = await self._generate_decision_options(inputs, context, decision_type)
        
        # Avaliar opções
        evaluated_options = await self._evaluate_options(options, context)
        
        # Selecionar melhor opção
        selected_option = await self._select_best_option(evaluated_options, context)
        
        # Criar resultado da decisão
        decision_result = await self._create_decision_result(
            selected_option, inputs, context, decision_type
        )
        
        # Registrar decisão
        await self._record_decision(decision_result)
        
        # Atualizar métricas
        decision_time = (datetime.utcnow() - start_time).total_seconds()
        await self._update_metrics(decision_result, decision_time)
        
        return {
            'decisions': [decision_result],
            'confidence': decision_result.confidence,
            'reasoning': decision_result.reasoning,
            'decision_time': decision_time,
            'risk_level': decision_result.risk_assessment.get('overall', 0.0)
        }
    
    async def _identify_decision_type(
        self, 
        inputs: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> DecisionType:
        """Identifica o tipo de decisão necessária."""
        
        # Análise de urgência
        urgency_indicators = inputs.get('urgency_indicators', [])
        is_urgent = any(indicator in ['critical', 'emergency', 'immediate'] 
                       for indicator in urgency_indicators)
        
        # Análise de complexidade
        complexity = context.get('complexity', 0.5)
        
        # Análise de escopo
        scope = inputs.get('scope', 'operational')
        
        if is_urgent:
            return DecisionType.EMERGENCY
        elif scope in ['strategic', 'long_term'] or complexity > 0.8:
            return DecisionType.STRATEGIC
        elif scope in ['tactical', 'medium_term'] or complexity > 0.5:
            return DecisionType.TACTICAL
        else:
            return DecisionType.OPERATIONAL
    
    async def _generate_decision_options(
        self, 
        inputs: Dict[str, Any], 
        context: Dict[str, Any], 
        decision_type: DecisionType
    ) -> List[DecisionOption]:
        """Gera opções de decisão."""
        
        options = []
        
        # Opção 1: Ação conservadora
        conservative_option = DecisionOption(
            id="conservative",
            description="Ação conservadora com baixo risco",
            expected_outcome={
                'success_probability': 0.8,
                'benefit_level': 0.6,
                'stability': 0.9
            },
            confidence=0.8,
            risk_level=0.2,
            resource_cost=0.5,
            time_cost=0.7
        )
        options.append(conservative_option)
        
        # Opção 2: Ação balanceada
        balanced_option = DecisionOption(
            id="balanced",
            description="Ação balanceada entre risco e benefício",
            expected_outcome={
                'success_probability': 0.7,
                'benefit_level': 0.8,
                'stability': 0.7
            },
            confidence=0.75,
            risk_level=0.4,
            resource_cost=0.6,
            time_cost=0.5
        )
        options.append(balanced_option)
        
        # Opção 3: Ação agressiva (se apropriado)
        if decision_type != DecisionType.EMERGENCY and context.get('risk_tolerance', 0.5) > 0.6:
            aggressive_option = DecisionOption(
                id="aggressive",
                description="Ação agressiva com alto potencial",
                expected_outcome={
                    'success_probability': 0.6,
                    'benefit_level': 0.9,
                    'stability': 0.5
                },
                confidence=0.6,
                risk_level=0.7,
                resource_cost=0.8,
                time_cost=0.3
            )
            options.append(aggressive_option)
        
        return options
    
    async def _evaluate_options(
        self, 
        options: List[DecisionOption], 
        context: Dict[str, Any]
    ) -> List[Tuple[DecisionOption, float]]:
        """Avalia opções de decisão."""
        
        evaluated = []
        
        for option in options:
            # Calcular score multi-dimensional
            score = await self._calculate_option_score(option, context)
            evaluated.append((option, score))
        
        # Ordenar por score
        evaluated.sort(key=lambda x: x[1], reverse=True)
        
        return evaluated
    
    async def _calculate_option_score(
        self, 
        option: DecisionOption, 
        context: Dict[str, Any]
    ) -> float:
        """Calcula score de uma opção."""
        
        # Fatores de avaliação
        success_factor = option.expected_outcome.get('success_probability', 0.5)
        benefit_factor = option.expected_outcome.get('benefit_level', 0.5)
        stability_factor = option.expected_outcome.get('stability', 0.5)
        
        # Penalidades
        risk_penalty = option.risk_level * 0.3
        cost_penalty = (option.resource_cost + option.time_cost) * 0.1
        
        # Bonus por confiança
        confidence_bonus = option.confidence * 0.2
        
        # Score combinado
        base_score = (success_factor + benefit_factor + stability_factor) / 3
        adjusted_score = base_score - risk_penalty - cost_penalty + confidence_bonus
        
        return max(0.0, min(1.0, adjusted_score))
    
    async def _select_best_option(
        self, 
        evaluated_options: List[Tuple[DecisionOption, float]], 
        context: Dict[str, Any]
    ) -> DecisionOption:
        """Seleciona a melhor opção."""
        
        if not evaluated_options:
            # Fallback para opção padrão
            return DecisionOption(
                id="default",
                description="Opção padrão de fallback",
                expected_outcome={'success_probability': 0.5},
                confidence=0.5,
                risk_level=0.5,
                resource_cost=0.5,
                time_cost=0.5
            )
        
        # Selecionar opção com maior score
        best_option, best_score = evaluated_options[0]
        
        # Verificar se atende critérios mínimos
        if best_score < self.confidence_threshold:
            logger.warning(f"Best option score {best_score} below threshold {self.confidence_threshold}")
        
        return best_option
    
    async def _create_decision_result(
        self, 
        selected_option: DecisionOption, 
        inputs: Dict[str, Any], 
        context: Dict[str, Any], 
        decision_type: DecisionType
    ) -> DecisionResult:
        """Cria resultado da decisão."""
        
        decision_id = f"decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Gerar raciocínio
        reasoning = await self._generate_reasoning(selected_option, context, decision_type)
        
        # Avaliar riscos
        risk_assessment = await self._assess_risks(selected_option, context)
        
        # Calcular benefícios esperados
        expected_benefits = await self._calculate_expected_benefits(selected_option, context)
        
        # Criar plano de implementação
        implementation_plan = await self._create_implementation_plan(selected_option, context)
        
        return DecisionResult(
            decision_id=decision_id,
            selected_option=selected_option,
            reasoning=reasoning,
            confidence=selected_option.confidence,
            risk_assessment=risk_assessment,
            expected_benefits=expected_benefits,
            implementation_plan=implementation_plan
        )
    
    async def _generate_reasoning(
        self, 
        option: DecisionOption, 
        context: Dict[str, Any], 
        decision_type: DecisionType
    ) -> List[str]:
        """Gera raciocínio para a decisão."""
        
        reasoning = []
        
        # Raciocínio baseado no tipo de decisão
        if decision_type == DecisionType.STRATEGIC:
            reasoning.append("Decisão estratégica focada em longo prazo")
        elif decision_type == DecisionType.TACTICAL:
            reasoning.append("Decisão tática para implementação de médio prazo")
        elif decision_type == DecisionType.OPERATIONAL:
            reasoning.append("Decisão operacional para execução imediata")
        elif decision_type == DecisionType.EMERGENCY:
            reasoning.append("Decisão de emergência para resolução rápida")
        
        # Raciocínio baseado na opção selecionada
        reasoning.append(f"Opção selecionada: {option.description}")
        reasoning.append(f"Confiança: {option.confidence:.2f}")
        reasoning.append(f"Nível de risco: {option.risk_level:.2f}")
        
        # Raciocínio baseado no contexto
        if context.get('urgency', 0) > 0.7:
            reasoning.append("Contexto de alta urgência considerado")
        
        if context.get('resource_constraints', False):
            reasoning.append("Restrições de recursos consideradas")
        
        return reasoning
    
    async def _assess_risks(
        self, 
        option: DecisionOption, 
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Avalia riscos da opção selecionada."""
        
        risks = {
            'overall': option.risk_level,
            'execution_risk': option.risk_level * 0.8,
            'resource_risk': option.resource_cost * 0.6,
            'timeline_risk': option.time_cost * 0.4,
            'dependency_risk': len(option.dependencies) * 0.1
        }
        
        # Ajustar riscos baseado no contexto
        if context.get('uncertainty', 0) > 0.7:
            risks['uncertainty_risk'] = context['uncertainty']
            risks['overall'] = max(risks['overall'], context['uncertainty'])
        
        return risks
    
    async def _calculate_expected_benefits(
        self, 
        option: DecisionOption, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcula benefícios esperados."""
        
        return {
            'immediate_benefits': option.expected_outcome.get('benefit_level', 0.5),
            'long_term_value': option.expected_outcome.get('benefit_level', 0.5) * 1.2,
            'stability_improvement': option.expected_outcome.get('stability', 0.5),
            'efficiency_gain': 1.0 - option.resource_cost,
            'time_savings': 1.0 - option.time_cost
        }
    
    async def _create_implementation_plan(
        self, 
        option: DecisionOption, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cria plano de implementação."""
        
        return {
            'phases': [
                {
                    'phase': 'preparation',
                    'duration': option.time_cost * 0.2,
                    'resources': option.resource_cost * 0.3,
                    'description': 'Preparação e configuração inicial'
                },
                {
                    'phase': 'execution',
                    'duration': option.time_cost * 0.6,
                    'resources': option.resource_cost * 0.5,
                    'description': 'Execução principal da decisão'
                },
                {
                    'phase': 'validation',
                    'duration': option.time_cost * 0.2,
                    'resources': option.resource_cost * 0.2,
                    'description': 'Validação e verificação de resultados'
                }
            ],
            'dependencies': option.dependencies,
            'constraints': option.constraints,
            'success_criteria': [
                f"Confiança mínima: {option.confidence:.2f}",
                f"Risco máximo: {option.risk_level:.2f}",
                "Execução dentro do prazo estimado"
            ]
        }
    
    async def _record_decision(self, decision_result: DecisionResult) -> None:
        """Registra decisão no histórico."""
        
        self.active_decisions[decision_result.decision_id] = decision_result
        self.decision_history.append(decision_result)
        
        # Manter apenas as últimas 1000 decisões
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    async def _update_metrics(
        self, 
        decision_result: DecisionResult, 
        decision_time: float
    ) -> None:
        """Atualiza métricas de performance."""
        
        self.performance_metrics['total_decisions'] += 1
        
        if decision_result.confidence > self.confidence_threshold:
            self.performance_metrics['successful_decisions'] += 1
        
        # Atualizar média de confiança
        total_decisions = self.performance_metrics['total_decisions']
        current_avg = self.performance_metrics['average_confidence']
        self.performance_metrics['average_confidence'] = (
            (current_avg * (total_decisions - 1) + decision_result.confidence) / total_decisions
        )
        
        # Atualizar média de tempo de decisão
        current_avg_time = self.performance_metrics['average_decision_time']
        self.performance_metrics['average_decision_time'] = (
            (current_avg_time * (total_decisions - 1) + decision_time) / total_decisions
        )
    
    async def get_decision_status(self, decision_id: str) -> Optional[DecisionResult]:
        """Obtém status de uma decisão."""
        return self.active_decisions.get(decision_id)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtém métricas de performance."""
        return {
            **self.performance_metrics,
            'success_rate': (
                self.performance_metrics['successful_decisions'] / 
                max(self.performance_metrics['total_decisions'], 1)
            ),
            'active_decisions_count': len(self.active_decisions)
        }
    
    async def optimize_parameters(self) -> None:
        """Otimiza parâmetros do córtex de decisão."""
        
        # Analisar padrões de decisão
        if len(self.decision_history) > 10:
            patterns = await self._analyze_decision_patterns()
            
            # Ajustar threshold de confiança baseado no histórico
            if patterns.get('low_confidence_success_rate', 0) > 0.8:
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
                logger.info(f"Lowered confidence threshold to {self.confidence_threshold}")
            
            # Ajustar tolerância a risco baseado no sucesso
            if patterns.get('high_risk_success_rate', 0) > 0.7:
                self.risk_tolerance = min(0.8, self.risk_tolerance + 0.05)
                logger.info(f"Increased risk tolerance to {self.risk_tolerance}")
        
        logger.info("Enhanced Decision Cortex parameters optimized")
    
    async def _analyze_decision_patterns(self) -> Dict[str, float]:
        """Analisa padrões nas decisões."""
        
        if len(self.decision_history) < 5:
            return {}
        
        recent_decisions = self.decision_history[-20:]  # Últimas 20 decisões
        
        patterns = {}
        
        # Taxa de sucesso de decisões de baixa confiança
        low_confidence_decisions = [
            d for d in recent_decisions if d.confidence < self.confidence_threshold
        ]
        if low_confidence_decisions:
            successful_low_conf = sum(1 for d in low_confidence_decisions if d.confidence > 0.6)
            patterns['low_confidence_success_rate'] = successful_low_conf / len(low_confidence_decisions)
        
        # Taxa de sucesso de decisões de alto risco
        high_risk_decisions = [
            d for d in recent_decisions if d.selected_option.risk_level > 0.6
        ]
        if high_risk_decisions:
            successful_high_risk = sum(1 for d in high_risk_decisions if d.confidence > 0.7)
            patterns['high_risk_success_rate'] = successful_high_risk / len(high_risk_decisions)
        
        return patterns
    
    async def shutdown(self) -> None:
        """Desliga o córtex de decisão."""
        logger.info("Shutting down Enhanced Decision Cortex")
        
        # Salvar estado se necessário
        self.active_decisions.clear()
        
        logger.info("Enhanced Decision Cortex shutdown complete")
