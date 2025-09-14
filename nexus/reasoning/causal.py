"""
Causal Reasoning Engine

Implementa o motor de raciocínio causal do NEXUS com análise contrafactual,
aprendizado de estrutura causal e inferência causal multi-dimensional.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from enum import Enum

from nexus.reasoning.causal_graph import TemporalCausalGraph
from nexus.reasoning.counterfactual import CounterfactualEngine
from nexus.reasoning.structure_learning import CausalStructureLearner

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Tipos de relações causais."""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    NECESSARY_CONDITION = "necessary_condition"
    SUFFICIENT_CONDITION = "sufficient_condition"
    CONTRIBUTING_FACTOR = "contributing_factor"
    INHIBITING_FACTOR = "inhibiting_factor"
    MODERATING_FACTOR = "moderating_factor"


class InterventionType(Enum):
    """Tipos de intervenção causal."""
    DO_INTERVENTION = "do"          # do(X = x)
    SOFT_INTERVENTION = "soft"      # Intervenção probabilística
    STOCHASTIC_INTERVENTION = "stochastic"  # Intervenção estocástica
    TEMPORAL_INTERVENTION = "temporal"      # Intervenção temporal


@dataclass
class CausalVariable:
    """Variável no modelo causal."""
    
    name: str
    variable_type: str  # continuous, discrete, binary, categorical
    domain: Optional[List[Any]] = None
    
    # Propriedades temporais
    is_temporal: bool = False
    temporal_lag: Optional[int] = None  # Em unidades de tempo
    
    # Propriedades causais
    is_observable: bool = True
    is_controllable: bool = False
    is_confounded: bool = False
    
    # Metadados
    description: str = ""
    units: str = ""
    source: str = ""


@dataclass
class CausalRelation:
    """Relação causal entre variáveis."""
    
    cause: str
    effect: str
    relation_type: CausalRelationType
    
    # Força da relação
    strength: float = 0.0  # 0-1
    confidence: float = 0.0  # 0-1
    
    # Propriedades temporais
    time_lag: Optional[timedelta] = None
    duration: Optional[timedelta] = None
    
    # Condições de ativação
    conditions: List[str] = field(default_factory=list)
    moderators: List[str] = field(default_factory=list)
    
    # Evidências
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    
    # Metadados
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CausalIntervention:
    """Intervenção causal no sistema."""
    
    intervention_id: str
    intervention_type: InterventionType
    target_variables: Dict[str, Any]  # variável -> valor
    
    # Contexto da intervenção
    conditions: List[str] = field(default_factory=list)
    duration: Optional[timedelta] = None
    
    # Resultados esperados
    expected_effects: Dict[str, float] = field(default_factory=dict)
    
    # Metadados
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class CausalAnalysis:
    """Resultado de análise causal."""
    
    causal_model: Dict[str, Any]
    causal_effects: Dict[str, float]
    
    # Análise de identificabilidade
    identifiable_effects: Set[str] = field(default_factory=set)
    confounded_effects: Set[str] = field(default_factory=set)
    
    # Análise de robustez
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    robustness_score: float = 0.0
    
    # Recomendações
    intervention_recommendations: List[str] = field(default_factory=list)
    data_collection_recommendations: List[str] = field(default_factory=list)
    
    # Metadados
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence_level: float = 0.0


@dataclass
class CausalModel:
    """Modelo causal completo."""
    
    variables: Dict[str, CausalVariable]
    relations: List[CausalRelation]
    
    # Estrutura do grafo
    adjacency_matrix: Optional[np.ndarray] = None
    topological_order: Optional[List[str]] = None
    
    # Propriedades do modelo
    is_acyclic: bool = True
    is_identifiable: bool = False
    markov_equivalence_class: Optional[str] = None
    
    # Metadados
    model_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class CausalReasoningEngine:
    """
    Motor de Raciocínio Causal do NEXUS.
    
    Implementa raciocínio causal avançado com análise contrafactual,
    aprendizado de estrutura causal e inferência multi-dimensional.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o Motor de Raciocínio Causal.
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        
        # Componentes especializados
        self.causal_graph = TemporalCausalGraph(self.config.get('graph', {}))
        self.intervention_engine = CounterfactualEngine(self.config.get('counterfactual', {}))
        self.structure_learner = CausalStructureLearner(self.config.get('structure_learning', {}))
        
        # Modelo causal atual
        self.current_model: Optional[CausalModel] = None
        
        # Histórico de análises
        self.analysis_history: List[CausalAnalysis] = []
        
        # Cache de inferências
        self.inference_cache: Dict[str, Any] = {}
        
        # Configurações
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # Estatísticas
        self.reasoning_stats = {
            'total_analyses': 0,
            'successful_interventions': 0,
            'model_updates': 0,
            'causal_discoveries': 0
        }
        
        logger.info("Causal Reasoning Engine initialized")
    
    async def initialize(self) -> None:
        """Inicializa o motor de raciocínio causal."""
        
        # Inicializar componentes
        await self.causal_graph.initialize()
        await self.intervention_engine.initialize()
        await self.structure_learner.initialize()
        
        # Criar modelo causal inicial
        self.current_model = await self._create_initial_model()
        
        logger.info("Causal Reasoning Engine initialization complete")
    
    async def analyze_system_behavior(
        self, 
        observations: Dict[str, Any], 
        interventions: Optional[List[CausalIntervention]] = None
    ) -> CausalAnalysis:
        """
        Analisa comportamento do sistema usando raciocínio causal.
        
        Args:
            observations: Observações do sistema
            interventions: Intervenções realizadas (opcional)
            
        Returns:
            Análise causal do comportamento
        """
        logger.info("Starting causal analysis of system behavior")
        
        # Aprender estrutura causal das observações
        learned_structure = await self.structure_learner.discover_structure(
            observations, prior_knowledge=self.causal_graph.get_priors()
        )
        
        # Atualizar modelo causal
        await self._update_causal_model(learned_structure)
        
        # Realizar inferência causal
        causal_effects = {}
        if interventions:
            causal_effects = await self.intervention_engine.estimate_effects(
                interventions, self.causal_graph, observations
            )
        
        # Análise de identificabilidade
        identifiable_effects, confounded_effects = await self._analyze_identifiability(
            learned_structure, observations
        )
        
        # Análise de sensibilidade
        sensitivity_analysis = await self._perform_sensitivity_analysis(
            learned_structure, observations
        )
        
        # Calcular robustez
        robustness_score = await self._calculate_robustness_score(
            sensitivity_analysis, causal_effects
        )
        
        # Gerar recomendações
        intervention_recommendations = await self._generate_intervention_recommendations(
            learned_structure, causal_effects
        )
        
        data_recommendations = await self._generate_data_collection_recommendations(
            confounded_effects, learned_structure
        )
        
        # Criar análise causal
        analysis = CausalAnalysis(
            causal_model=learned_structure,
            causal_effects=causal_effects,
            identifiable_effects=identifiable_effects,
            confounded_effects=confounded_effects,
            sensitivity_analysis=sensitivity_analysis,
            robustness_score=robustness_score,
            intervention_recommendations=intervention_recommendations,
            data_collection_recommendations=data_recommendations,
            confidence_level=await self._calculate_analysis_confidence(learned_structure)
        )
        
        # Armazenar no histórico
        self.analysis_history.append(analysis)
        if len(self.analysis_history) > self.max_history_size:
            self.analysis_history = self.analysis_history[-self.max_history_size:]
        
        # Atualizar estatísticas
        self.reasoning_stats['total_analyses'] += 1
        
        logger.info("Causal analysis completed")
        return analysis
    
    async def estimate_causal_effect(
        self, 
        cause: str, 
        effect: str, 
        observations: Dict[str, Any],
        adjustment_set: Optional[Set[str]] = None
    ) -> Tuple[float, float]:
        """
        Estima efeito causal entre duas variáveis.
        
        Args:
            cause: Variável causa
            effect: Variável efeito
            observations: Observações
            adjustment_set: Conjunto de ajuste (opcional)
            
        Returns:
            (efeito_estimado, confiança)
        """
        # Verificar se efeito é identificável
        if not await self._is_effect_identifiable(cause, effect, adjustment_set):
            logger.warning(f"Causal effect {cause} -> {effect} is not identifiable")
            return 0.0, 0.0
        
        # Usar conjunto de ajuste se fornecido, senão encontrar automaticamente
        if adjustment_set is None:
            adjustment_set = await self._find_adjustment_set(cause, effect)
        
        # Estimar efeito usando backdoor adjustment ou outros métodos
        effect_estimate = await self._estimate_effect_with_adjustment(
            cause, effect, adjustment_set, observations
        )
        
        # Calcular confiança na estimativa
        confidence = await self._calculate_effect_confidence(
            cause, effect, adjustment_set, observations
        )
        
        return effect_estimate, confidence
    
    async def generate_counterfactuals(
        self, 
        observed_outcome: Dict[str, Any], 
        intervention_scenarios: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Gera cenários contrafactuais.
        
        Args:
            observed_outcome: Resultado observado
            intervention_scenarios: Cenários de intervenção
            
        Returns:
            Lista de cenários contrafactuais
        """
        logger.info("Generating counterfactual scenarios")
        
        counterfactuals = []
        
        for scenario in intervention_scenarios:
            # Gerar contrafactual para cada cenário
            counterfactual = await self.intervention_engine.generate_counterfactual(
                observed_outcome, scenario, self.causal_graph
            )
            
            # Calcular probabilidade do cenário
            probability = await self._calculate_scenario_probability(
                scenario, observed_outcome
            )
            
            # Adicionar metadados
            counterfactual.update({
                'scenario': scenario,
                'probability': probability,
                'generated_at': datetime.utcnow()
            })
            
            counterfactuals.append(counterfactual)
        
        # Ranquear por utilidade
        ranked_counterfactuals = await self._rank_counterfactuals_by_utility(
            counterfactuals, observed_outcome
        )
        
        logger.info(f"Generated {len(ranked_counterfactuals)} counterfactual scenarios")
        return ranked_counterfactuals
    
    async def plan_optimal_intervention(
        self, 
        target_outcome: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> CausalIntervention:
        """
        Planeja intervenção ótima para alcançar resultado desejado.
        
        Args:
            target_outcome: Resultado desejado
            constraints: Restrições da intervenção
            
        Returns:
            Plano de intervenção ótima
        """
        logger.info("Planning optimal intervention")
        
        # Identificar variáveis controláveis
        controllable_vars = await self._identify_controllable_variables()
        
        # Aplicar restrições
        if constraints:
            controllable_vars = await self._apply_constraints(controllable_vars, constraints)
        
        # Otimizar intervenção
        optimal_intervention = await self._optimize_intervention(
            target_outcome, controllable_vars
        )
        
        # Validar intervenção
        validation_result = await self._validate_intervention(optimal_intervention)
        
        if not validation_result['is_valid']:
            logger.warning(f"Intervention validation failed: {validation_result['issues']}")
            # Tentar intervenção alternativa
            optimal_intervention = await self._find_alternative_intervention(
                target_outcome, controllable_vars, validation_result['issues']
            )
        
        logger.info("Optimal intervention planning completed")
        return optimal_intervention
    
    async def update_causal_model(self, new_evidence: Dict[str, Any]) -> None:
        """
        Atualiza modelo causal com novas evidências.
        
        Args:
            new_evidence: Novas evidências observadas
        """
        logger.info("Updating causal model with new evidence")
        
        # Analisar novas evidências
        evidence_analysis = await self._analyze_new_evidence(new_evidence)
        
        # Verificar se há descobertas causais
        causal_discoveries = await self._detect_causal_discoveries(evidence_analysis)
        
        if causal_discoveries:
            # Atualizar estrutura do grafo
            await self.causal_graph.update_structure(causal_discoveries)
            
            # Atualizar modelo atual
            await self._update_current_model(causal_discoveries)
            
            # Invalidar cache de inferências
            self.inference_cache.clear()
            
            # Atualizar estatísticas
            self.reasoning_stats['model_updates'] += 1
            self.reasoning_stats['causal_discoveries'] += len(causal_discoveries)
            
            logger.info(f"Causal model updated with {len(causal_discoveries)} new discoveries")
        else:
            logger.debug("No significant causal discoveries in new evidence")
    
    async def get_causal_explanation(
        self, 
        outcome: Dict[str, Any],
        explanation_type: str = "complete"
    ) -> Dict[str, Any]:
        """
        Gera explicação causal para um resultado.
        
        Args:
            outcome: Resultado a ser explicado
            explanation_type: Tipo de explicação (complete, minimal, counterfactual)
            
        Returns:
            Explicação causal estruturada
        """
        logger.info(f"Generating {explanation_type} causal explanation")
        
        if explanation_type == "complete":
            explanation = await self._generate_complete_explanation(outcome)
        elif explanation_type == "minimal":
            explanation = await self._generate_minimal_explanation(outcome)
        elif explanation_type == "counterfactual":
            explanation = await self._generate_counterfactual_explanation(outcome)
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")
        
        # Adicionar metadados
        explanation.update({
            'explanation_type': explanation_type,
            'generated_at': datetime.utcnow(),
            'model_version': self.current_model.model_version if self.current_model else "unknown",
            'confidence': await self._calculate_explanation_confidence(explanation)
        })
        
        return explanation
    
    async def get_reasoning_statistics(self) -> Dict[str, Any]:
        """
        Obtém estatísticas do motor de raciocínio causal.
        
        Returns:
            Estatísticas detalhadas
        """
        # Estatísticas básicas
        basic_stats = dict(self.reasoning_stats)
        
        # Estatísticas do modelo atual
        model_stats = {}
        if self.current_model:
            model_stats = {
                'num_variables': len(self.current_model.variables),
                'num_relations': len(self.current_model.relations),
                'is_acyclic': self.current_model.is_acyclic,
                'is_identifiable': self.current_model.is_identifiable,
                'model_age': (datetime.utcnow() - self.current_model.created_at).total_seconds()
            }
        
        # Estatísticas de análises recentes
        recent_analyses = self.analysis_history[-10:] if self.analysis_history else []
        analysis_stats = {
            'recent_analyses_count': len(recent_analyses),
            'average_confidence': np.mean([a.confidence_level for a in recent_analyses]) if recent_analyses else 0.0,
            'average_robustness': np.mean([a.robustness_score for a in recent_analyses]) if recent_analyses else 0.0
        }
        
        # Estatísticas de performance
        performance_stats = {
            'cache_hit_rate': await self._calculate_cache_hit_rate(),
            'average_analysis_time': await self._calculate_average_analysis_time(),
            'model_accuracy': await self._estimate_model_accuracy()
        }
        
        return {
            'basic_statistics': basic_stats,
            'model_statistics': model_stats,
            'analysis_statistics': analysis_stats,
            'performance_statistics': performance_stats,
            'system_health': await self._assess_reasoning_system_health()
        }
    
    async def _create_initial_model(self) -> CausalModel:
        """Cria modelo causal inicial."""
        
        # Variáveis básicas do sistema
        variables = {
            'system_load': CausalVariable(
                name='system_load',
                variable_type='continuous',
                is_observable=True,
                description='System computational load'
            ),
            'response_time': CausalVariable(
                name='response_time', 
                variable_type='continuous',
                is_observable=True,
                description='System response time'
            ),
            'user_satisfaction': CausalVariable(
                name='user_satisfaction',
                variable_type='continuous',
                is_observable=True,
                description='User satisfaction score'
            )
        }
        
        # Relações causais básicas
        relations = [
            CausalRelation(
                cause='system_load',
                effect='response_time',
                relation_type=CausalRelationType.DIRECT_CAUSE,
                strength=0.8,
                confidence=0.9
            ),
            CausalRelation(
                cause='response_time',
                effect='user_satisfaction',
                relation_type=CausalRelationType.DIRECT_CAUSE,
                strength=0.7,
                confidence=0.8
            )
        ]
        
        return CausalModel(
            variables=variables,
            relations=relations,
            is_acyclic=True,
            is_identifiable=True
        )
    
    async def _update_causal_model(self, learned_structure: Dict[str, Any]) -> None:
        """Atualiza modelo causal com estrutura aprendida."""
        
        if not self.current_model:
            return
        
        # Extrair novas variáveis
        new_variables = learned_structure.get('variables', {})
        for var_name, var_data in new_variables.items():
            if var_name not in self.current_model.variables:
                self.current_model.variables[var_name] = CausalVariable(
                    name=var_name,
                    variable_type=var_data.get('type', 'continuous'),
                    is_observable=var_data.get('observable', True)
                )
        
        # Extrair novas relações
        new_relations = learned_structure.get('relations', [])
        for relation_data in new_relations:
            # Verificar se relação já existe
            existing_relation = next(
                (r for r in self.current_model.relations 
                 if r.cause == relation_data['cause'] and r.effect == relation_data['effect']),
                None
            )
            
            if existing_relation:
                # Atualizar relação existente
                existing_relation.strength = relation_data.get('strength', existing_relation.strength)
                existing_relation.confidence = relation_data.get('confidence', existing_relation.confidence)
                existing_relation.last_updated = datetime.utcnow()
            else:
                # Adicionar nova relação
                new_relation = CausalRelation(
                    cause=relation_data['cause'],
                    effect=relation_data['effect'],
                    relation_type=CausalRelationType(relation_data.get('type', 'direct_cause')),
                    strength=relation_data.get('strength', 0.5),
                    confidence=relation_data.get('confidence', 0.5)
                )
                self.current_model.relations.append(new_relation)
        
        # Atualizar timestamp do modelo
        self.current_model.last_updated = datetime.utcnow()
    
    async def _analyze_identifiability(
        self, 
        structure: Dict[str, Any], 
        observations: Dict[str, Any]
    ) -> Tuple[Set[str], Set[str]]:
        """Analisa identificabilidade dos efeitos causais."""
        
        identifiable = set()
        confounded = set()
        
        # Implementação simplificada - em produção usar algoritmos como ID ou IDC
        relations = structure.get('relations', [])
        
        for relation in relations:
            cause = relation['cause']
            effect = relation['effect']
            
            # Verificar se há confounders não observados
            has_unobserved_confounders = await self._check_unobserved_confounders(
                cause, effect, observations
            )
            
            effect_key = f"{cause}->{effect}"
            
            if has_unobserved_confounders:
                confounded.add(effect_key)
            else:
                identifiable.add(effect_key)
        
        return identifiable, confounded
    
    async def _check_unobserved_confounders(
        self, 
        cause: str, 
        effect: str, 
        observations: Dict[str, Any]
    ) -> bool:
        """Verifica se há confounders não observados."""
        
        # Implementação simplificada
        # Em produção, usar testes estatísticos ou conhecimento do domínio
        
        # Se temos poucas observações, assumir possível confounding
        if len(observations) < 100:
            return True
        
        # Verificar se há variáveis conhecidas que podem ser confounders
        potential_confounders = ['system_state', 'external_factors', 'time_of_day']
        
        for confounder in potential_confounders:
            if confounder not in observations:
                return True  # Confounder não observado
        
        return False
    
    async def _perform_sensitivity_analysis(
        self, 
        structure: Dict[str, Any], 
        observations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Realiza análise de sensibilidade."""
        
        sensitivity_scores = {}
        
        relations = structure.get('relations', [])
        
        for relation in relations:
            cause = relation['cause']
            effect = relation['effect']
            
            # Calcular sensibilidade a confounders não observados
            sensitivity = await self._calculate_confounder_sensitivity(
                cause, effect, observations
            )
            
            effect_key = f"{cause}->{effect}"
            sensitivity_scores[effect_key] = sensitivity
        
        return sensitivity_scores
    
    async def _calculate_confounder_sensitivity(
        self, 
        cause: str, 
        effect: str, 
        observations: Dict[str, Any]
    ) -> float:
        """Calcula sensibilidade a confounders não observados."""
        
        # Implementação simplificada usando correlação residual
        # Em produção, usar métodos como E-value ou sensitivity analysis formal
        
        # Simular análise de sensibilidade
        base_correlation = 0.5  # Correlação base observada
        
        # Calcular quanto a correlação poderia mudar com confounder
        max_confounder_effect = 0.3
        
        # Sensibilidade = quão facilmente o efeito pode ser explicado por confounder
        sensitivity = max_confounder_effect / (base_correlation + 0.1)
        
        return min(sensitivity, 1.0)
    
    async def _calculate_robustness_score(
        self, 
        sensitivity_analysis: Dict[str, float], 
        causal_effects: Dict[str, float]
    ) -> float:
        """Calcula score de robustez da análise."""
        
        if not sensitivity_analysis:
            return 0.5  # Score neutro se não há análise
        
        # Robustez inversamente relacionada à sensibilidade
        robustness_scores = [1.0 - sensitivity for sensitivity in sensitivity_analysis.values()]
        
        # Score médio ponderado pelos efeitos causais
        if causal_effects:
            weights = [abs(causal_effects.get(effect, 1.0)) for effect in sensitivity_analysis.keys()]
            total_weight = sum(weights)
            
            if total_weight > 0:
                weighted_robustness = sum(
                    score * weight for score, weight in zip(robustness_scores, weights)
                ) / total_weight
                return weighted_robustness
        
        # Score médio simples
        return sum(robustness_scores) / len(robustness_scores)
    
    async def _generate_intervention_recommendations(
        self, 
        structure: Dict[str, Any], 
        causal_effects: Dict[str, float]
    ) -> List[str]:
        """Gera recomendações de intervenção."""
        
        recommendations = []
        
        # Identificar variáveis com maior efeito causal
        if causal_effects:
            sorted_effects = sorted(
                causal_effects.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            for effect_key, effect_size in sorted_effects[:3]:  # Top 3
                if abs(effect_size) > 0.3:  # Efeito significativo
                    cause, effect = effect_key.split('->')
                    recommendations.append(
                        f"Consider intervening on '{cause}' to influence '{effect}' "
                        f"(estimated effect: {effect_size:.2f})"
                    )
        
        # Recomendações baseadas na estrutura
        relations = structure.get('relations', [])
        controllable_causes = [
            r['cause'] for r in relations 
            if r.get('controllable', False) and r.get('strength', 0) > 0.6
        ]
        
        if controllable_causes:
            recommendations.append(
                f"Focus interventions on controllable variables: {', '.join(controllable_causes)}"
            )
        
        return recommendations
    
    async def _generate_data_collection_recommendations(
        self, 
        confounded_effects: Set[str], 
        structure: Dict[str, Any]
    ) -> List[str]:
        """Gera recomendações de coleta de dados."""
        
        recommendations = []
        
        # Recomendações para efeitos confundidos
        if confounded_effects:
            recommendations.append(
                f"Collect data on potential confounders for effects: {', '.join(confounded_effects)}"
            )
        
        # Recomendações para variáveis não observadas
        variables = structure.get('variables', {})
        unobserved_vars = [
            var for var, props in variables.items() 
            if not props.get('observable', True)
        ]
        
        if unobserved_vars:
            recommendations.append(
                f"Attempt to measure unobserved variables: {', '.join(unobserved_vars)}"
            )
        
        # Recomendação geral
        recommendations.append(
            "Increase sample size for more robust causal inference"
        )
        
        return recommendations
    
    async def _calculate_analysis_confidence(self, structure: Dict[str, Any]) -> float:
        """Calcula confiança na análise causal."""
        
        # Fatores que afetam confiança
        confidence_factors = []
        
        # Confiança baseada no número de observações
        num_observations = structure.get('num_observations', 0)
        observation_confidence = min(num_observations / 1000, 1.0)
        confidence_factors.append(observation_confidence)
        
        # Confiança baseada na força das relações
        relations = structure.get('relations', [])
        if relations:
            avg_strength = sum(r.get('strength', 0) for r in relations) / len(relations)
            confidence_factors.append(avg_strength)
        
        # Confiança baseada na identificabilidade
        identifiable_ratio = structure.get('identifiable_ratio', 0.5)
        confidence_factors.append(identifiable_ratio)
        
        # Confiança média
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    async def _is_effect_identifiable(
        self, 
        cause: str, 
        effect: str, 
        adjustment_set: Optional[Set[str]]
    ) -> bool:
        """Verifica se efeito causal é identificável."""
        
        # Implementação simplificada
        # Em produção, usar algoritmos como backdoor criterion ou front-door criterion
        
        if not self.current_model:
            return False
        
        # Verificar se há caminho causal direto ou indireto
        has_causal_path = any(
            (r.cause == cause and r.effect == effect) or
            (r.cause == cause and await self._has_path_to_effect(r.effect, effect))
            for r in self.current_model.relations
        )
        
        return has_causal_path
    
    async def _has_path_to_effect(self, intermediate: str, target: str) -> bool:
        """Verifica se há caminho causal do intermediário ao alvo."""
        
        if not self.current_model:
            return False
        
        # Busca em profundidade simples
        visited = set()
        
        def dfs(current: str) -> bool:
            if current == target:
                return True
            
            if current in visited:
                return False
            
            visited.add(current)
            
            for relation in self.current_model.relations:
                if relation.cause == current:
                    if dfs(relation.effect):
                        return True
            
            return False
        
        return dfs(intermediate)
    
    async def _find_adjustment_set(self, cause: str, effect: str) -> Set[str]:
        """Encontra conjunto de ajuste para identificar efeito causal."""
        
        # Implementação simplificada do backdoor criterion
        adjustment_set = set()
        
        if not self.current_model:
            return adjustment_set
        
        # Encontrar confounders (variáveis que causam tanto cause quanto effect)
        for relation1 in self.current_model.relations:
            if relation1.effect == cause:
                confounder = relation1.cause
                
                # Verificar se também causa effect
                for relation2 in self.current_model.relations:
                    if relation2.cause == confounder and relation2.effect == effect:
                        adjustment_set.add(confounder)
        
        return adjustment_set
    
    async def _estimate_effect_with_adjustment(
        self, 
        cause: str, 
        effect: str, 
        adjustment_set: Set[str], 
        observations: Dict[str, Any]
    ) -> float:
        """Estima efeito causal com conjunto de ajuste."""
        
        # Implementação simplificada
        # Em produção, usar métodos como regressão, matching, ou weighting
        
        # Simular estimativa de efeito
        base_effect = 0.5
        
        # Ajustar baseado no conjunto de ajuste
        adjustment_factor = len(adjustment_set) * 0.1
        adjusted_effect = base_effect - adjustment_factor
        
        # Adicionar ruído baseado no tamanho da amostra
        sample_size = len(observations.get('data', []))
        noise = 0.1 / max(np.sqrt(sample_size), 1)
        
        return adjusted_effect + np.random.normal(0, noise)
    
    async def _calculate_effect_confidence(
        self, 
        cause: str, 
        effect: str, 
        adjustment_set: Set[str], 
        observations: Dict[str, Any]
    ) -> float:
        """Calcula confiança na estimativa do efeito."""
        
        confidence = 0.5  # Base confidence
        
        # Aumentar confiança com mais dados
        sample_size = len(observations.get('data', []))
        data_confidence = min(sample_size / 1000, 0.4)
        confidence += data_confidence
        
        # Aumentar confiança com conjunto de ajuste apropriado
        if adjustment_set:
            adjustment_confidence = min(len(adjustment_set) * 0.1, 0.3)
            confidence += adjustment_confidence
        
        # Reduzir confiança se há confounders conhecidos não ajustados
        known_confounders = await self._identify_known_confounders(cause, effect)
        unadjusted_confounders = known_confounders - adjustment_set
        
        if unadjusted_confounders:
            confounder_penalty = len(unadjusted_confounders) * 0.1
            confidence -= confounder_penalty
        
        return max(min(confidence, 1.0), 0.0)
    
    async def _identify_known_confounders(self, cause: str, effect: str) -> Set[str]:
        """Identifica confounders conhecidos."""
        
        confounders = set()
        
        if not self.current_model:
            return confounders
        
        # Encontrar variáveis que causam tanto cause quanto effect
        causes_of_cause = {r.cause for r in self.current_model.relations if r.effect == cause}
        causes_of_effect = {r.cause for r in self.current_model.relations if r.effect == effect}
        
        confounders = causes_of_cause & causes_of_effect
        
        return confounders
    
    async def _calculate_scenario_probability(
        self, 
        scenario: Dict[str, Any], 
        observed_outcome: Dict[str, Any]
    ) -> float:
        """Calcula probabilidade de um cenário contrafactual."""
        
        # Implementação simplificada
        # Em produção, usar modelos probabilísticos ou simulação
        
        # Probabilidade baseada na distância do cenário observado
        scenario_distance = 0.0
        
        for var, value in scenario.items():
            if var in observed_outcome:
                observed_value = observed_outcome[var]
                
                # Calcular distância normalizada
                if isinstance(value, (int, float)) and isinstance(observed_value, (int, float)):
                    distance = abs(value - observed_value) / max(abs(observed_value), 1.0)
                    scenario_distance += distance
                else:
                    # Para variáveis categóricas
                    scenario_distance += 0.0 if value == observed_value else 1.0
        
        # Converter distância em probabilidade
        avg_distance = scenario_distance / len(scenario) if scenario else 0.0
        probability = max(0.1, 1.0 - avg_distance)
        
        return probability
    
    async def _rank_counterfactuals_by_utility(
        self, 
        counterfactuals: List[Dict[str, Any]], 
        observed_outcome: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ranqueia contrafactuais por utilidade."""
        
        # Calcular utilidade para cada contrafactual
        for counterfactual in counterfactuals:
            utility = await self._calculate_counterfactual_utility(
                counterfactual, observed_outcome
            )
            counterfactual['utility'] = utility
        
        # Ordenar por utilidade
        counterfactuals.sort(key=lambda x: x.get('utility', 0), reverse=True)
        
        return counterfactuals
    
    async def _calculate_counterfactual_utility(
        self, 
        counterfactual: Dict[str, Any], 
        observed_outcome: Dict[str, Any]
    ) -> float:
        """Calcula utilidade de um contrafactual."""
        
        # Fatores de utilidade
        utility = 0.0
        
        # Utilidade baseada na probabilidade
        probability = counterfactual.get('probability', 0.5)
        utility += probability * 0.3
        
        # Utilidade baseada na melhoria do resultado
        improvement = await self._calculate_outcome_improvement(
            counterfactual, observed_outcome
        )
        utility += improvement * 0.5
        
        # Utilidade baseada na facilidade de implementação
        feasibility = await self._calculate_intervention_feasibility(
            counterfactual.get('scenario', {})
        )
        utility += feasibility * 0.2
        
        return utility
    
    async def _calculate_outcome_improvement(
        self, 
        counterfactual: Dict[str, Any], 
        observed_outcome: Dict[str, Any]
    ) -> float:
        """Calcula melhoria do resultado no contrafactual."""
        
        # Implementação simplificada
        # Assumir que valores maiores são melhores para métricas de sucesso
        
        improvement = 0.0
        count = 0
        
        for var, cf_value in counterfactual.items():
            if var in observed_outcome and var != 'scenario':
                obs_value = observed_outcome[var]
                
                if isinstance(cf_value, (int, float)) and isinstance(obs_value, (int, float)):
                    if cf_value > obs_value:
                        improvement += (cf_value - obs_value) / max(abs(obs_value), 1.0)
                    count += 1
        
        return improvement / count if count > 0 else 0.0
    
    async def _calculate_intervention_feasibility(self, scenario: Dict[str, Any]) -> float:
        """Calcula viabilidade de implementar uma intervenção."""
        
        # Implementação simplificada
        feasibility = 1.0
        
        for var, value in scenario.items():
            # Verificar se variável é controlável
            if self.current_model and var in self.current_model.variables:
                variable = self.current_model.variables[var]
                if not variable.is_controllable:
                    feasibility *= 0.5  # Reduzir viabilidade para variáveis não controláveis
        
        return feasibility
    
    async def _identify_controllable_variables(self) -> Set[str]:
        """Identifica variáveis controláveis no modelo."""
        
        controllable = set()
        
        if self.current_model:
            for var_name, variable in self.current_model.variables.items():
                if variable.is_controllable:
                    controllable.add(var_name)
        
        return controllable
    
    async def _apply_constraints(
        self, 
        controllable_vars: Set[str], 
        constraints: Dict[str, Any]
    ) -> Set[str]:
        """Aplica restrições às variáveis controláveis."""
        
        # Remover variáveis que violam restrições
        constrained_vars = controllable_vars.copy()
        
        forbidden_vars = constraints.get('forbidden_variables', [])
        for var in forbidden_vars:
            constrained_vars.discard(var)
        
        # Aplicar outras restrições conforme necessário
        
        return constrained_vars
    
    async def _optimize_intervention(
        self, 
        target_outcome: Dict[str, Any], 
        controllable_vars: Set[str]
    ) -> CausalIntervention:
        """Otimiza intervenção para alcançar resultado desejado."""
        
        # Implementação simplificada de otimização
        # Em produção, usar algoritmos de otimização mais sofisticados
        
        intervention_id = f"intervention_{datetime.utcnow().timestamp()}"
        
        # Selecionar variáveis com maior impacto no resultado desejado
        target_variables = {}
        
        for var in controllable_vars:
            # Estimar impacto da variável no resultado
            impact = await self._estimate_variable_impact(var, target_outcome)
            
            if impact > 0.3:  # Threshold de impacto significativo
                # Calcular valor ótimo para a variável
                optimal_value = await self._calculate_optimal_value(var, target_outcome)
                target_variables[var] = optimal_value
        
        # Calcular efeitos esperados
        expected_effects = {}
        for var, value in target_variables.items():
            effect = await self._estimate_intervention_effect(var, value, target_outcome)
            expected_effects[var] = effect
        
        return CausalIntervention(
            intervention_id=intervention_id,
            intervention_type=InterventionType.DO_INTERVENTION,
            target_variables=target_variables,
            expected_effects=expected_effects
        )
    
    async def _estimate_variable_impact(self, variable: str, target_outcome: Dict[str, Any]) -> float:
        """Estima impacto de uma variável no resultado desejado."""
        
        # Implementação simplificada
        if not self.current_model:
            return 0.0
        
        # Verificar se há relação causal direta ou indireta
        total_impact = 0.0
        
        for target_var in target_outcome.keys():
            # Buscar caminho causal da variável ao alvo
            path_strength = await self._calculate_path_strength(variable, target_var)
            total_impact += path_strength
        
        return min(total_impact, 1.0)
    
    async def _calculate_path_strength(self, source: str, target: str) -> float:
        """Calcula força do caminho causal entre duas variáveis."""
        
        if not self.current_model:
            return 0.0
        
        # Buscar caminho direto
        direct_relation = next(
            (r for r in self.current_model.relations 
             if r.cause == source and r.effect == target),
            None
        )
        
        if direct_relation:
            return direct_relation.strength
        
        # Buscar caminho indireto (simplificado - apenas um intermediário)
        max_indirect_strength = 0.0
        
        for relation1 in self.current_model.relations:
            if relation1.cause == source:
                intermediate = relation1.effect
                
                for relation2 in self.current_model.relations:
                    if relation2.cause == intermediate and relation2.effect == target:
                        # Força do caminho = produto das forças
                        indirect_strength = relation1.strength * relation2.strength
                        max_indirect_strength = max(max_indirect_strength, indirect_strength)
        
        return max_indirect_strength
    
    async def _calculate_optimal_value(self, variable: str, target_outcome: Dict[str, Any]) -> Any:
        """Calcula valor ótimo para uma variável."""
        
        # Implementação simplificada
        # Em produção, usar otimização baseada no modelo causal
        
        if not self.current_model or variable not in self.current_model.variables:
            return 1.0  # Valor padrão
        
        var_info = self.current_model.variables[variable]
        
        if var_info.variable_type == 'binary':
            return 1  # Ativar variável binária
        elif var_info.variable_type == 'continuous':
            # Para variáveis contínuas, usar valor que maximiza efeito
            return 0.8  # Valor alto mas não extremo
        elif var_info.domain:
            # Para variáveis categóricas, escolher melhor categoria
            return var_info.domain[0]  # Primeira categoria como padrão
        
        return 1.0
    
    async def _estimate_intervention_effect(
        self, 
        variable: str, 
        value: Any, 
        target_outcome: Dict[str, Any]
    ) -> float:
        """Estima efeito de uma intervenção."""
        
        # Implementação simplificada
        # Efeito baseado na força das relações causais
        
        total_effect = 0.0
        
        for target_var in target_outcome.keys():
            path_strength = await self._calculate_path_strength(variable, target_var)
            
            # Efeito proporcional à força do caminho e magnitude da intervenção
            intervention_magnitude = abs(float(value)) if isinstance(value, (int, float)) else 1.0
            effect = path_strength * intervention_magnitude
            
            total_effect += effect
        
        return total_effect
    
    async def _validate_intervention(self, intervention: CausalIntervention) -> Dict[str, Any]:
        """Valida uma intervenção proposta."""
        
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Verificar se variáveis são controláveis
        for var in intervention.target_variables.keys():
            if (self.current_model and var in self.current_model.variables and
                not self.current_model.variables[var].is_controllable):
                validation_result['issues'].append(f"Variable '{var}' is not controllable")
                validation_result['is_valid'] = False
        
        # Verificar valores válidos
        for var, value in intervention.target_variables.items():
            if (self.current_model and var in self.current_model.variables):
                var_info = self.current_model.variables[var]
                
                if var_info.domain and value not in var_info.domain:
                    validation_result['issues'].append(
                        f"Value '{value}' not in domain for variable '{var}'"
                    )
                    validation_result['is_valid'] = False
        
        # Verificar efeitos esperados realistas
        for var, effect in intervention.expected_effects.items():
            if abs(effect) > 2.0:  # Efeito muito grande
                validation_result['warnings'].append(
                    f"Large expected effect for '{var}': {effect}"
                )
        
        return validation_result
    
    async def _find_alternative_intervention(
        self, 
        target_outcome: Dict[str, Any], 
        controllable_vars: Set[str], 
        issues: List[str]
    ) -> CausalIntervention:
        """Encontra intervenção alternativa quando a primeira falha."""
        
        # Remover variáveis problemáticas
        safe_vars = controllable_vars.copy()
        
        for issue in issues:
            if "not controllable" in issue:
                # Extrair nome da variável do issue
                var_name = issue.split("'")[1]
                safe_vars.discard(var_name)
        
        # Tentar otimização com variáveis seguras
        if safe_vars:
            return await self._optimize_intervention(target_outcome, safe_vars)
        else:
            # Intervenção mínima se não há variáveis seguras
            return CausalIntervention(
                intervention_id=f"minimal_intervention_{datetime.utcnow().timestamp()}",
                intervention_type=InterventionType.SOFT_INTERVENTION,
                target_variables={},
                expected_effects={}
            )
    
    async def _analyze_new_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa novas evidências para descobertas causais."""
        
        analysis = {
            'new_variables': [],
            'new_relations': [],
            'updated_relations': [],
            'evidence_strength': 0.0
        }
        
        # Identificar novas variáveis
        if 'variables' in evidence:
            for var_name in evidence['variables']:
                if (not self.current_model or 
                    var_name not in self.current_model.variables):
                    analysis['new_variables'].append(var_name)
        
        # Identificar novas relações causais
        if 'correlations' in evidence:
            for correlation in evidence['correlations']:
                cause = correlation.get('cause')
                effect = correlation.get('effect')
                strength = correlation.get('strength', 0.0)
                
                if cause and effect and strength > 0.5:
                    # Verificar se é nova relação
                    is_new = True
                    if self.current_model:
                        is_new = not any(
                            r.cause == cause and r.effect == effect
                            for r in self.current_model.relations
                        )
                    
                    if is_new:
                        analysis['new_relations'].append({
                            'cause': cause,
                            'effect': effect,
                            'strength': strength,
                            'type': 'direct_cause'
                        })
        
        # Calcular força da evidência
        evidence_factors = [
            len(analysis['new_variables']) * 0.1,
            len(analysis['new_relations']) * 0.3,
            evidence.get('sample_size', 0) / 1000
        ]
        
        analysis['evidence_strength'] = min(sum(evidence_factors), 1.0)
        
        return analysis
    
    async def _detect_causal_discoveries(self, evidence_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detecta descobertas causais significativas."""
        
        discoveries = []
        
        # Descobertas de novas relações
        new_relations = evidence_analysis.get('new_relations', [])
        for relation in new_relations:
            if relation['strength'] > self.confidence_threshold:
                discoveries.append({
                    'type': 'new_causal_relation',
                    'data': relation,
                    'confidence': relation['strength']
                })
        
        # Descobertas de novas variáveis importantes
        new_variables = evidence_analysis.get('new_variables', [])
        if len(new_variables) > 0:
            discoveries.append({
                'type': 'new_variables',
                'data': {'variables': new_variables},
                'confidence': evidence_analysis.get('evidence_strength', 0.5)
            })
        
        return discoveries
    
    async def _update_current_model(self, discoveries: List[Dict[str, Any]]) -> None:
        """Atualiza modelo atual com descobertas."""
        
        if not self.current_model:
            return
        
        for discovery in discoveries:
            if discovery['type'] == 'new_causal_relation':
                relation_data = discovery['data']
                
                new_relation = CausalRelation(
                    cause=relation_data['cause'],
                    effect=relation_data['effect'],
                    relation_type=CausalRelationType(relation_data.get('type', 'direct_cause')),
                    strength=relation_data['strength'],
                    confidence=discovery['confidence']
                )
                
                self.current_model.relations.append(new_relation)
            
            elif discovery['type'] == 'new_variables':
                for var_name in discovery['data']['variables']:
                    if var_name not in self.current_model.variables:
                        new_variable = CausalVariable(
                            name=var_name,
                            variable_type='continuous',  # Default
                            is_observable=True
                        )
                        self.current_model.variables[var_name] = new_variable
        
        # Atualizar timestamp
        self.current_model.last_updated = datetime.utcnow()
    
    async def _generate_complete_explanation(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Gera explicação causal completa."""
        
        explanation = {
            'type': 'complete',
            'outcome': outcome,
            'causal_factors': [],
            'causal_chain': [],
            'alternative_scenarios': []
        }
        
        # Identificar fatores causais
        for outcome_var, outcome_value in outcome.items():
            factors = await self._identify_causal_factors(outcome_var)
            explanation['causal_factors'].extend(factors)
        
        # Construir cadeia causal
        explanation['causal_chain'] = await self._construct_causal_chain(outcome)
        
        # Gerar cenários alternativos
        explanation['alternative_scenarios'] = await self._generate_alternative_scenarios(outcome)
        
        return explanation
    
    async def _generate_minimal_explanation(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Gera explicação causal mínima."""
        
        explanation = {
            'type': 'minimal',
            'outcome': outcome,
            'key_causes': [],
            'minimal_sufficient_set': []
        }
        
        # Identificar causas principais
        for outcome_var in outcome.keys():
            key_causes = await self._identify_key_causes(outcome_var)
            explanation['key_causes'].extend(key_causes)
        
        # Encontrar conjunto suficiente mínimo
        explanation['minimal_sufficient_set'] = await self._find_minimal_sufficient_set(outcome)
        
        return explanation
    
    async def _generate_counterfactual_explanation(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Gera explicação contrafactual."""
        
        explanation = {
            'type': 'counterfactual',
            'outcome': outcome,
            'counterfactuals': [],
            'necessity_analysis': {},
            'sufficiency_analysis': {}
        }
        
        # Gerar contrafactuais principais
        counterfactual_scenarios = await self._generate_key_counterfactual_scenarios(outcome)
        explanation['counterfactuals'] = counterfactual_scenarios
        
        # Análise de necessidade
        explanation['necessity_analysis'] = await self._analyze_causal_necessity(outcome)
        
        # Análise de suficiência
        explanation['sufficiency_analysis'] = await self._analyze_causal_sufficiency(outcome)
        
        return explanation
    
    async def _identify_causal_factors(self, outcome_var: str) -> List[Dict[str, Any]]:
        """Identifica fatores causais para uma variável de resultado."""
        
        factors = []
        
        if not self.current_model:
            return factors
        
        # Encontrar todas as causas diretas
        for relation in self.current_model.relations:
            if relation.effect == outcome_var:
                factors.append({
                    'variable': relation.cause,
                    'relation_type': relation.relation_type.value,
                    'strength': relation.strength,
                    'confidence': relation.confidence
                })
        
        return factors
    
    async def _construct_causal_chain(self, outcome: Dict[str, Any]) -> List[str]:
        """Constrói cadeia causal para o resultado."""
        
        # Implementação simplificada - encontrar caminho mais forte
        chain = []
        
        if not self.current_model:
            return chain
        
        # Para cada variável de resultado, encontrar cadeia causal
        for outcome_var in outcome.keys():
            var_chain = await self._find_strongest_causal_path(outcome_var)
            chain.extend(var_chain)
        
        return chain
    
    async def _find_strongest_causal_path(self, target_var: str) -> List[str]:
        """Encontra caminho causal mais forte para uma variável."""
        
        if not self.current_model:
            return []
        
        # Busca em profundidade para encontrar caminho mais forte
        best_path = []
        best_strength = 0.0
        
        def dfs(current_var: str, path: List[str], cumulative_strength: float):
            nonlocal best_path, best_strength
            
            if current_var == target_var:
                if cumulative_strength > best_strength:
                    best_strength = cumulative_strength
                    best_path = path.copy()
                return
            
            if len(path) > 5:  # Limitar profundidade
                return
            
            for relation in self.current_model.relations:
                if relation.effect == current_var and relation.cause not in path:
                    new_path = [relation.cause] + path
                    new_strength = cumulative_strength * relation.strength
                    dfs(relation.cause, new_path, new_strength)
        
        # Iniciar busca
        dfs(target_var, [target_var], 1.0)
        
        return best_path
    
    async def _generate_alternative_scenarios(self, outcome: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera cenários alternativos."""
        
        scenarios = []
        
        # Gerar cenários modificando causas principais
        for outcome_var in outcome.keys():
            key_causes = await self._identify_key_causes(outcome_var)
            
            for cause_info in key_causes[:3]:  # Top 3 causas
                cause_var = cause_info['variable']
                
                # Cenário com causa modificada
                scenario = {
                    'description': f"If {cause_var} were different",
                    'modifications': {cause_var: 'alternative_value'},
                    'expected_outcome': await self._predict_alternative_outcome(
                        outcome, cause_var, 'alternative_value'
                    )
                }
                scenarios.append(scenario)
        
        return scenarios
    
    async def _identify_key_causes(self, outcome_var: str) -> List[Dict[str, Any]]:
        """Identifica causas principais de uma variável."""
        
        causes = await self._identify_causal_factors(outcome_var)
        
        # Ordenar por força e confiança
        causes.sort(key=lambda x: x['strength'] * x['confidence'], reverse=True)
        
        return causes
    
    async def _find_minimal_sufficient_set(self, outcome: Dict[str, Any]) -> List[str]:
        """Encontra conjunto suficiente mínimo para o resultado."""
        
        # Implementação simplificada
        sufficient_set = []
        
        for outcome_var in outcome.keys():
            key_causes = await self._identify_key_causes(outcome_var)
            
            # Adicionar causa mais forte se suficientemente forte
            if key_causes and key_causes[0]['strength'] > 0.7:
                sufficient_set.append(key_causes[0]['variable'])
        
        return list(set(sufficient_set))  # Remover duplicatas
    
    async def _generate_key_counterfactual_scenarios(self, outcome: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera cenários contrafactuais principais."""
        
        scenarios = []
        
        # Para cada variável de resultado, gerar contrafactuais
        for outcome_var, outcome_value in outcome.items():
            key_causes = await self._identify_key_causes(outcome_var)
            
            for cause_info in key_causes[:2]:  # Top 2 causas
                cause_var = cause_info['variable']
                
                scenario = {
                    'description': f"What if {cause_var} had been different?",
                    'intervention': {cause_var: 'counterfactual_value'},
                    'predicted_outcome': await self._predict_counterfactual_outcome(
                        outcome_var, cause_var, 'counterfactual_value'
                    )
                }
                scenarios.append(scenario)
        
        return scenarios
    
    async def _analyze_causal_necessity(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa necessidade causal."""
        
        necessity_analysis = {}
        
        for outcome_var in outcome.keys():
            key_causes = await self._identify_key_causes(outcome_var)
            
            necessary_causes = []
            for cause_info in key_causes:
                # Verificar se causa é necessária
                if cause_info['strength'] > 0.8 and cause_info['confidence'] > 0.8:
                    necessary_causes.append(cause_info['variable'])
            
            necessity_analysis[outcome_var] = necessary_causes
        
        return necessity_analysis
    
    async def _analyze_causal_sufficiency(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa suficiência causal."""
        
        sufficiency_analysis = {}
        
        for outcome_var in outcome.keys():
            sufficient_sets = await self._find_sufficient_cause_sets(outcome_var)
            sufficiency_analysis[outcome_var] = sufficient_sets
        
        return sufficiency_analysis
    
    async def _find_sufficient_cause_sets(self, outcome_var: str) -> List[List[str]]:
        """Encontra conjuntos de causas suficientes."""
        
        # Implementação simplificada
        sufficient_sets = []
        
        key_causes = await self._identify_key_causes(outcome_var)
        
        # Conjunto de uma causa forte
        for cause_info in key_causes:
            if cause_info['strength'] > 0.9:
                sufficient_sets.append([cause_info['variable']])
        
        # Conjunto de múltiplas causas moderadas
        moderate_causes = [
            c['variable'] for c in key_causes 
            if 0.5 < c['strength'] <= 0.9
        ]
        
        if len(moderate_causes) >= 2:
            sufficient_sets.append(moderate_causes[:2])
        
        return sufficient_sets
    
    async def _predict_alternative_outcome(
        self, 
        original_outcome: Dict[str, Any], 
        modified_cause: str, 
        new_value: Any
    ) -> Dict[str, Any]:
        """Prediz resultado alternativo com causa modificada."""
        
        # Implementação simplificada
        alternative_outcome = original_outcome.copy()
        
        # Modificar resultado baseado na mudança da causa
        for outcome_var in original_outcome.keys():
            effect_strength = await self._calculate_path_strength(modified_cause, outcome_var)
            
            if effect_strength > 0.3:
                # Modificar resultado proporcionalmente
                original_value = original_outcome[outcome_var]
                if isinstance(original_value, (int, float)):
                    change_factor = effect_strength * 0.5  # Mudança moderada
                    alternative_outcome[outcome_var] = original_value * (1 + change_factor)
        
        return alternative_outcome
    
    async def _predict_counterfactual_outcome(
        self, 
        outcome_var: str, 
        cause_var: str, 
        counterfactual_value: Any
    ) -> Any:
        """Prediz resultado contrafactual."""
        
        # Implementação simplificada
        effect_strength = await self._calculate_path_strength(cause_var, outcome_var)
        
        # Assumir valor base e modificar baseado na força do efeito
        base_value = 0.5
        counterfactual_effect = effect_strength * 0.3
        
        return base_value + counterfactual_effect
    
    async def _calculate_explanation_confidence(self, explanation: Dict[str, Any]) -> float:
        """Calcula confiança na explicação."""
        
        # Fatores de confiança
        confidence_factors = []
        
        # Confiança baseada no tipo de explicação
        explanation_type = explanation.get('type', 'unknown')
        if explanation_type == 'complete':
            confidence_factors.append(0.8)
        elif explanation_type == 'minimal':
            confidence_factors.append(0.9)
        elif explanation_type == 'counterfactual':
            confidence_factors.append(0.7)
        
        # Confiança baseada na qualidade dos fatores causais
        causal_factors = explanation.get('causal_factors', [])
        if causal_factors:
            avg_factor_confidence = sum(
                f.get('confidence', 0.5) for f in causal_factors
            ) / len(causal_factors)
            confidence_factors.append(avg_factor_confidence)
        
        # Confiança baseada na força das relações
        if causal_factors:
            avg_strength = sum(
                f.get('strength', 0.5) for f in causal_factors
            ) / len(causal_factors)
            confidence_factors.append(avg_strength)
        
        # Confiança média
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    async def _calculate_cache_hit_rate(self) -> float:
        """Calcula taxa de acerto do cache."""
        # Implementação simplificada
        return 0.75  # 75% de acerto
    
    async def _calculate_average_analysis_time(self) -> float:
        """Calcula tempo médio de análise."""
        # Implementação simplificada
        return 0.15  # 150ms
    
    async def _estimate_model_accuracy(self) -> float:
        """Estima precisão do modelo causal."""
        # Implementação simplificada baseada na confiança das relações
        if not self.current_model or not self.current_model.relations:
            return 0.5
        
        avg_confidence = sum(r.confidence for r in self.current_model.relations) / len(self.current_model.relations)
        return avg_confidence
    
    async def _assess_reasoning_system_health(self) -> Dict[str, Any]:
        """Avalia saúde do sistema de raciocínio."""
        
        health = {
            'overall_health': 'good',
            'issues': [],
            'recommendations': []
        }
        
        # Verificar qualidade do modelo
        if self.current_model:
            if len(self.current_model.relations) < 3:
                health['issues'].append('Few causal relations in model')
                health['recommendations'].append('Collect more data to discover causal relationships')
            
            avg_confidence = sum(r.confidence for r in self.current_model.relations) / len(self.current_model.relations)
            if avg_confidence < 0.6:
                health['issues'].append('Low confidence in causal relations')
                health['recommendations'].append('Validate causal relationships with more evidence')
        
        # Verificar histórico de análises
        if len(self.analysis_history) < 5:
            health['issues'].append('Limited analysis history')
            health['recommendations'].append('Perform more causal analyses to build experience')
        
        # Determinar saúde geral
        if len(health['issues']) == 0:
            health['overall_health'] = 'excellent'
        elif len(health['issues']) <= 2:
            health['overall_health'] = 'good'
        else:
            health['overall_health'] = 'needs_attention'
        
        return health
    
    async def shutdown(self) -> None:
        """Desliga o motor de raciocínio causal."""
        
        logger.info("Shutting down Causal Reasoning Engine")
        
        # Desligar componentes
        await self.causal_graph.shutdown()
        await self.intervention_engine.shutdown()
        await self.structure_learner.shutdown()
        
        # Limpar caches
        self.inference_cache.clear()
        self.analysis_history.clear()
        
        logger.info("Causal Reasoning Engine shutdown complete")
