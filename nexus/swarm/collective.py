"""
Collective Intelligence Engine

Motor de inteligência coletiva que combina insights de múltiplos agentes
para gerar soluções superiores através de síntese colaborativa.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class CollectiveInsight:
    """Insight coletivo gerado pela síntese de múltiplos agentes."""
    
    insight_id: str
    content: str
    confidence: float
    contributing_agents: List[str]
    synthesis_method: str
    timestamp: datetime
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentContribution:
    """Contribuição individual de um agente."""
    
    agent_id: str
    contribution_type: str
    content: Any
    confidence: float
    relevance_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisStrategy:
    """Estratégia de síntese de inteligência coletiva."""
    
    strategy_id: str
    strategy_type: str
    parameters: Dict[str, Any]
    effectiveness_score: float
    applicability_conditions: Dict[str, Any] = field(default_factory=dict)


class CollectiveIntelligenceEngine:
    """
    Motor de Inteligência Coletiva.
    
    Combina contribuições de múltiplos agentes para gerar insights
    e soluções superiores através de síntese colaborativa.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Configurações de síntese
        self.min_contributors = config.get('min_contributors', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.synthesis_timeout = config.get('synthesis_timeout', 30)
        
        # Estado do motor
        self.collective_insights: Dict[str, CollectiveInsight] = {}
        self.agent_contributions: Dict[str, List[AgentContribution]] = defaultdict(list)
        self.synthesis_strategies: Dict[str, SynthesisStrategy] = {}
        
        # Métricas de performance
        self.synthesis_metrics = {
            'total_insights': 0,
            'successful_syntheses': 0,
            'failed_syntheses': 0,
            'average_quality': 0.0,
            'strategy_effectiveness': {}
        }
        
        logger.info("Collective Intelligence Engine initialized")
    
    async def initialize(self) -> None:
        """Inicializa o motor de inteligência coletiva."""
        
        logger.info("Initializing Collective Intelligence Engine")
        
        # Inicializar estratégias de síntese
        await self._initialize_synthesis_strategies()
        
        logger.info("Collective Intelligence Engine initialization complete")
    
    async def synthesize_collective_intelligence(
        self, 
        contributions: List[AgentContribution],
        synthesis_type: str = 'adaptive'
    ) -> CollectiveInsight:
        """Sintetiza inteligência coletiva a partir de contribuições."""
        
        logger.info(f"Synthesizing collective intelligence from {len(contributions)} contributions")
        
        # Validar contribuições
        valid_contributions = await self._validate_contributions(contributions)
        
        if len(valid_contributions) < self.min_contributors:
            raise ValueError(f"Insufficient valid contributions: {len(valid_contributions)} < {self.min_contributors}")
        
        # Selecionar estratégia de síntese
        strategy = await self._select_synthesis_strategy(valid_contributions, synthesis_type)
        
        # Executar síntese
        collective_insight = await self._execute_synthesis(valid_contributions, strategy)
        
        # Armazenar insight
        self.collective_insights[collective_insight.insight_id] = collective_insight
        
        # Atualizar métricas
        self.synthesis_metrics['total_insights'] += 1
        self.synthesis_metrics['successful_syntheses'] += 1
        
        logger.info(f"Collective insight synthesized: {collective_insight.insight_id}")
        return collective_insight
    
    async def _validate_contributions(
        self, 
        contributions: List[AgentContribution]
    ) -> List[AgentContribution]:
        """Valida contribuições dos agentes."""
        
        valid_contributions = []
        
        for contribution in contributions:
            # Verificar confiança mínima
            if contribution.confidence < 0.3:
                continue
            
            # Verificar relevância
            if contribution.relevance_score < 0.2:
                continue
            
            # Verificar se não é duplicata
            if not await self._is_duplicate_contribution(contribution, valid_contributions):
                valid_contributions.append(contribution)
        
        return valid_contributions
    
    async def _is_duplicate_contribution(
        self, 
        contribution: AgentContribution,
        existing_contributions: List[AgentContribution]
    ) -> bool:
        """Verifica se uma contribuição é duplicata."""
        
        # Implementação simplificada - verificar similaridade de conteúdo
        for existing in existing_contributions:
            if (existing.agent_id == contribution.agent_id and
                existing.contribution_type == contribution.contribution_type):
                
                # Calcular similaridade de conteúdo
                similarity = await self._calculate_content_similarity(
                    existing.content, contribution.content
                )
                
                if similarity > 0.8:
                    return True
        
        return False
    
    async def _calculate_content_similarity(
        self, 
        content1: Any, 
        content2: Any
    ) -> float:
        """Calcula similaridade entre conteúdos."""
        
        # Implementação simplificada
        if isinstance(content1, str) and isinstance(content2, str):
            # Similaridade de Jaccard para strings
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
        
        # Para outros tipos, usar comparação direta
        return 1.0 if content1 == content2 else 0.0
    
    async def _select_synthesis_strategy(
        self, 
        contributions: List[AgentContribution],
        synthesis_type: str
    ) -> SynthesisStrategy:
        """Seleciona estratégia de síntese apropriada."""
        
        # Analisar características das contribuições
        contribution_analysis = await self._analyze_contributions(contributions)
        
        # Selecionar estratégia baseada no tipo e análise
        if synthesis_type == 'adaptive':
            strategy = await self._select_adaptive_strategy(contribution_analysis)
        elif synthesis_type == 'consensus':
            strategy = self.synthesis_strategies['consensus']
        elif synthesis_type == 'weighted':
            strategy = self.synthesis_strategies['weighted']
        else:
            strategy = self.synthesis_strategies['default']
        
        return strategy
    
    async def _analyze_contributions(
        self, 
        contributions: List[AgentContribution]
    ) -> Dict[str, Any]:
        """Analisa características das contribuições."""
        
        analysis = {
            'total_contributions': len(contributions),
            'contribution_types': list(set(c.contribution_type for c in contributions)),
            'average_confidence': np.mean([c.confidence for c in contributions]),
            'average_relevance': np.mean([c.relevance_score for c in contributions]),
            'confidence_variance': np.var([c.confidence for c in contributions]),
            'relevance_variance': np.var([c.relevance_score for c in contributions])
        }
        
        return analysis
    
    async def _select_adaptive_strategy(
        self, 
        analysis: Dict[str, Any]
    ) -> SynthesisStrategy:
        """Seleciona estratégia adaptativa baseada na análise."""
        
        # Lógica de seleção adaptativa
        if analysis['confidence_variance'] < 0.1:
            # Baixa variância - usar consenso
            return self.synthesis_strategies['consensus']
        elif analysis['average_confidence'] > 0.8:
            # Alta confiança - usar média ponderada
            return self.synthesis_strategies['weighted']
        else:
            # Caso geral - usar síntese híbrida
            return self.synthesis_strategies['hybrid']
    
    async def _execute_synthesis(
        self, 
        contributions: List[AgentContribution],
        strategy: SynthesisStrategy
    ) -> CollectiveInsight:
        """Executa síntese usando a estratégia selecionada."""
        
        if strategy.strategy_type == 'consensus':
            return await self._consensus_synthesis(contributions)
        elif strategy.strategy_type == 'weighted':
            return await self._weighted_synthesis(contributions)
        elif strategy.strategy_type == 'hybrid':
            return await self._hybrid_synthesis(contributions)
        else:
            return await self._default_synthesis(contributions)
    
    async def _consensus_synthesis(
        self, 
        contributions: List[AgentContribution]
    ) -> CollectiveInsight:
        """Síntese por consenso."""
        
        # Encontrar elementos comuns
        common_elements = await self._find_common_elements(contributions)
        
        # Calcular confiança do consenso
        consensus_confidence = np.mean([c.confidence for c in contributions])
        
        # Criar insight coletivo
        insight = CollectiveInsight(
            insight_id=f"consensus_{datetime.utcnow().timestamp()}",
            content=common_elements,
            confidence=consensus_confidence,
            contributing_agents=[c.agent_id for c in contributions],
            synthesis_method='consensus',
            timestamp=datetime.utcnow(),
            quality_metrics={
                'consensus_strength': len(common_elements) / len(contributions),
                'agreement_level': consensus_confidence
            }
        )
        
        return insight
    
    async def _weighted_synthesis(
        self, 
        contributions: List[AgentContribution]
    ) -> CollectiveInsight:
        """Síntese ponderada por confiança."""
        
        # Calcular pesos baseados na confiança
        weights = [c.confidence for c in contributions]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Combinar contribuições ponderadas
        combined_content = await self._combine_weighted_content(
            contributions, normalized_weights
        )
        
        # Calcular confiança ponderada
        weighted_confidence = sum(c.confidence * w for c, w in zip(contributions, normalized_weights))
        
        # Criar insight coletivo
        insight = CollectiveInsight(
            insight_id=f"weighted_{datetime.utcnow().timestamp()}",
            content=combined_content,
            confidence=weighted_confidence,
            contributing_agents=[c.agent_id for c in contributions],
            synthesis_method='weighted',
            timestamp=datetime.utcnow(),
            quality_metrics={
                'weight_distribution': normalized_weights,
                'weighted_confidence': weighted_confidence
            }
        )
        
        return insight
    
    async def _hybrid_synthesis(
        self, 
        contributions: List[AgentContribution]
    ) -> CollectiveInsight:
        """Síntese híbrida combinando múltiplas estratégias."""
        
        # Executar síntese por consenso
        consensus_insight = await self._consensus_synthesis(contributions)
        
        # Executar síntese ponderada
        weighted_insight = await self._weighted_synthesis(contributions)
        
        # Combinar insights
        hybrid_content = await self._combine_insights(
            consensus_insight.content, weighted_insight.content
        )
        
        # Calcular confiança híbrida
        hybrid_confidence = (consensus_insight.confidence + weighted_insight.confidence) / 2
        
        # Criar insight coletivo
        insight = CollectiveInsight(
            insight_id=f"hybrid_{datetime.utcnow().timestamp()}",
            content=hybrid_content,
            confidence=hybrid_confidence,
            contributing_agents=[c.agent_id for c in contributions],
            synthesis_method='hybrid',
            timestamp=datetime.utcnow(),
            quality_metrics={
                'consensus_confidence': consensus_insight.confidence,
                'weighted_confidence': weighted_insight.confidence,
                'hybrid_balance': 0.5
            }
        )
        
        return insight
    
    async def _default_synthesis(
        self, 
        contributions: List[AgentContribution]
    ) -> CollectiveInsight:
        """Síntese padrão (média simples)."""
        
        # Combinar todas as contribuições
        combined_content = await self._combine_all_content(contributions)
        
        # Calcular confiança média
        average_confidence = np.mean([c.confidence for c in contributions])
        
        # Criar insight coletivo
        insight = CollectiveInsight(
            insight_id=f"default_{datetime.utcnow().timestamp()}",
            content=combined_content,
            confidence=average_confidence,
            contributing_agents=[c.agent_id for c in contributions],
            synthesis_method='default',
            timestamp=datetime.utcnow(),
            quality_metrics={
                'contribution_count': len(contributions),
                'average_confidence': average_confidence
            }
        )
        
        return insight
    
    async def _find_common_elements(
        self, 
        contributions: List[AgentContribution]
    ) -> str:
        """Encontra elementos comuns entre contribuições."""
        
        # Implementação simplificada
        if not contributions:
            return ""
        
        # Para strings, encontrar palavras comuns
        if all(isinstance(c.content, str) for c in contributions):
            all_words = []
            for contribution in contributions:
                words = contribution.content.lower().split()
                all_words.extend(words)
            
            # Encontrar palavras que aparecem em todas as contribuições
            word_counts = defaultdict(int)
            for word in all_words:
                word_counts[word] += 1
            
            common_words = [word for word, count in word_counts.items() 
                          if count == len(contributions)]
            
            return " ".join(common_words)
        
        # Para outros tipos, retornar primeira contribuição
        return str(contributions[0].content)
    
    async def _combine_weighted_content(
        self, 
        contributions: List[AgentContribution],
        weights: List[float]
    ) -> str:
        """Combina conteúdo ponderado."""
        
        # Implementação simplificada
        combined_parts = []
        
        for contribution, weight in zip(contributions, weights):
            if isinstance(contribution.content, str):
                # Adicionar conteúdo com peso
                weighted_content = f"[{weight:.2f}] {contribution.content}"
                combined_parts.append(weighted_content)
        
        return " | ".join(combined_parts)
    
    async def _combine_insights(
        self, 
        consensus_content: str, 
        weighted_content: str
    ) -> str:
        """Combina insights de diferentes estratégias."""
        
        return f"Consensus: {consensus_content} | Weighted: {weighted_content}"
    
    async def _combine_all_content(
        self, 
        contributions: List[AgentContribution]
    ) -> str:
        """Combina todo o conteúdo das contribuições."""
        
        content_parts = []
        
        for contribution in contributions:
            if isinstance(contribution.content, str):
                content_parts.append(contribution.content)
        
        return " | ".join(content_parts)
    
    async def _initialize_synthesis_strategies(self) -> None:
        """Inicializa estratégias de síntese."""
        
        # Estratégia de consenso
        self.synthesis_strategies['consensus'] = SynthesisStrategy(
            strategy_id='consensus',
            strategy_type='consensus',
            parameters={
                'min_agreement': 0.6,
                'consensus_threshold': 0.7
            },
            effectiveness_score=0.8,
            applicability_conditions={
                'min_contributors': 3,
                'max_confidence_variance': 0.2
            }
        )
        
        # Estratégia ponderada
        self.synthesis_strategies['weighted'] = SynthesisStrategy(
            strategy_id='weighted',
            strategy_type='weighted',
            parameters={
                'confidence_weight': 0.7,
                'relevance_weight': 0.3
            },
            effectiveness_score=0.85,
            applicability_conditions={
                'min_contributors': 2,
                'min_average_confidence': 0.5
            }
        )
        
        # Estratégia híbrida
        self.synthesis_strategies['hybrid'] = SynthesisStrategy(
            strategy_id='hybrid',
            strategy_type='hybrid',
            parameters={
                'consensus_weight': 0.4,
                'weighted_weight': 0.6
            },
            effectiveness_score=0.9,
            applicability_conditions={
                'min_contributors': 4,
                'mixed_confidence_levels': True
            }
        )
        
        # Estratégia padrão
        self.synthesis_strategies['default'] = SynthesisStrategy(
            strategy_id='default',
            strategy_type='default',
            parameters={},
            effectiveness_score=0.7,
            applicability_conditions={}
        )
        
        logger.info("Synthesis strategies initialized")
    
    async def get_collective_intelligence_metrics(self) -> Dict[str, Any]:
        """Obtém métricas de inteligência coletiva."""
        
        return {
            'total_insights': len(self.collective_insights),
            'synthesis_metrics': self.synthesis_metrics,
            'strategy_usage': {
                strategy_id: strategy.effectiveness_score 
                for strategy_id, strategy in self.synthesis_strategies.items()
            },
            'average_insight_quality': np.mean([
                insight.quality_metrics.get('consensus_strength', 0.5) 
                for insight in self.collective_insights.values()
            ]) if self.collective_insights else 0.0
        }
    
    async def shutdown(self) -> None:
        """Desliga o motor de inteligência coletiva."""
        
        logger.info("Shutting down Collective Intelligence Engine")
        
        # Salvar estado se necessário
        await self._save_collective_state()
        
        logger.info("Collective Intelligence Engine shutdown complete")
    
    async def _save_collective_state(self) -> None:
        """Salva estado do motor coletivo."""
        
        # Implementação simplificada
        logger.info("Collective intelligence state saved")
