"""
Specification Cortex

Implementa o córtex de especificação do NEXUS, responsável por análise semântica,
modelagem de domínio e extração de regras de negócio.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import json
from enum import Enum

logger = logging.getLogger(__name__)


class RequirementType(Enum):
    """Tipos de requisitos identificados."""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS_RULE = "business_rule"
    CONSTRAINT = "constraint"
    QUALITY_ATTRIBUTE = "quality_attribute"


class IntentCategory(Enum):
    """Categorias de intenção identificadas."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    INTEGRATE = "integrate"
    OPTIMIZE = "optimize"
    SECURE = "secure"
    SCALE = "scale"


@dataclass
class SemanticEntity:
    """Entidade semântica identificada no texto."""
    
    name: str
    entity_type: str
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    context: str = ""


@dataclass
class RequirementSpec:
    """Especificação de um requisito."""
    
    id: str
    text: str
    requirement_type: RequirementType
    priority: int
    entities: List[SemanticEntity] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    
    # Metadados
    confidence: float = 0.0
    source: str = ""
    extracted_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DomainConcept:
    """Conceito do domínio de negócio."""
    
    name: str
    description: str
    concept_type: str  # entity, value_object, service, aggregate
    attributes: Dict[str, str] = field(default_factory=dict)
    behaviors: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    
    # Regras de negócio associadas
    business_rules: List[str] = field(default_factory=list)
    
    # Métricas
    complexity_score: float = 0.0
    importance_score: float = 0.0


@dataclass
class BusinessRule:
    """Regra de negócio extraída."""
    
    id: str
    description: str
    rule_type: str  # validation, calculation, workflow, constraint
    conditions: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    
    # Contexto
    domain_concepts: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    
    # Propriedades
    is_critical: bool = False
    is_configurable: bool = False
    priority: int = 1


@dataclass
class SemanticAnalysisResult:
    """Resultado da análise semântica."""
    
    entities: List[SemanticEntity]
    intents: List[IntentCategory]
    requirements: List[RequirementSpec]
    
    # Métricas de análise
    confidence_score: float = 0.0
    ambiguity_score: float = 0.0
    completeness_score: float = 0.0
    
    # Insights
    identified_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DomainModel:
    """Modelo do domínio de negócio."""
    
    domain_name: str
    concepts: List[DomainConcept]
    relationships: Dict[str, Dict[str, str]]  # concept -> {related_concept: relationship_type}
    
    # Arquitetura conceitual
    bounded_contexts: List[str] = field(default_factory=list)
    aggregates: List[str] = field(default_factory=list)
    
    # Métricas do modelo
    cohesion_score: float = 0.0
    coupling_score: float = 0.0
    complexity_score: float = 0.0


class SpecificationCortex:
    """
    Córtex de Especificação do NEXUS.
    
    Responsável por análise semântica profunda, modelagem de domínio
    e extração de regras de negócio a partir de requisitos.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o Córtex de Especificação.
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        
        # Componentes especializados
        self.semantic_analyzer = SemanticAnalyzer(self.config.get('semantic', {}))
        self.domain_modeler = DomainModeler(self.config.get('domain', {}))
        self.business_rule_extractor = BusinessRuleExtractor(self.config.get('rules', {}))
        
        # Cache de análises
        self.analysis_cache: Dict[str, SemanticAnalysisResult] = {}
        self.domain_model_cache: Dict[str, DomainModel] = {}
        
        # Conhecimento base
        self.domain_patterns: Dict[str, List[str]] = {}
        self.requirement_templates: List[Dict[str, Any]] = []
        
        logger.info("Specification Cortex initialized")
    
    async def analyze_semantics(
        self, 
        description: str, 
        requirements: List[str]
    ) -> SemanticAnalysisResult:
        """
        Executa análise semântica profunda de descrição e requisitos.
        
        Args:
            description: Descrição principal do projeto
            requirements: Lista de requisitos
            
        Returns:
            Resultado da análise semântica
        """
        logger.info("Starting semantic analysis")
        
        # Verificar cache
        cache_key = self._generate_cache_key(description, requirements)
        if cache_key in self.analysis_cache:
            logger.debug("Returning cached semantic analysis")
            return self.analysis_cache[cache_key]
        
        # Análise semântica do texto principal
        primary_analysis = await self.semantic_analyzer.analyze_text(description)
        
        # Análise de requisitos individuais
        requirement_analyses = []
        for req in requirements:
            req_analysis = await self.semantic_analyzer.analyze_requirement(req)
            requirement_analyses.append(req_analysis)
        
        # Consolidar resultados
        consolidated_result = await self._consolidate_semantic_analysis(
            primary_analysis, requirement_analyses
        )
        
        # Identificar gaps e gerar recomendações
        consolidated_result.identified_gaps = await self._identify_specification_gaps(
            consolidated_result
        )
        consolidated_result.recommendations = await self._generate_specification_recommendations(
            consolidated_result
        )
        
        # Cache resultado
        self.analysis_cache[cache_key] = consolidated_result
        
        logger.info(f"Semantic analysis completed - {len(consolidated_result.entities)} entities found")
        return consolidated_result
    
    async def model_domain(
        self, 
        semantic_analysis: SemanticAnalysisResult, 
        constraints: Dict[str, Any]
    ) -> DomainModel:
        """
        Cria modelo de domínio baseado na análise semântica.
        
        Args:
            semantic_analysis: Resultado da análise semântica
            constraints: Restrições do domínio
            
        Returns:
            Modelo de domínio estruturado
        """
        logger.info("Starting domain modeling")
        
        # Extrair conceitos do domínio das entidades semânticas
        domain_concepts = await self.domain_modeler.extract_concepts(
            semantic_analysis.entities, constraints
        )
        
        # Identificar relacionamentos entre conceitos
        relationships = await self.domain_modeler.identify_relationships(
            domain_concepts, semantic_analysis.requirements
        )
        
        # Identificar bounded contexts
        bounded_contexts = await self.domain_modeler.identify_bounded_contexts(
            domain_concepts, relationships
        )
        
        # Identificar aggregates
        aggregates = await self.domain_modeler.identify_aggregates(
            domain_concepts, relationships
        )
        
        # Criar modelo de domínio
        domain_model = DomainModel(
            domain_name=await self._extract_domain_name(semantic_analysis),
            concepts=domain_concepts,
            relationships=relationships,
            bounded_contexts=bounded_contexts,
            aggregates=aggregates
        )
        
        # Calcular métricas do modelo
        domain_model.cohesion_score = await self._calculate_cohesion(domain_model)
        domain_model.coupling_score = await self._calculate_coupling(domain_model)
        domain_model.complexity_score = await self._calculate_domain_complexity(domain_model)
        
        logger.info(f"Domain model created - {len(domain_concepts)} concepts, {len(bounded_contexts)} contexts")
        return domain_model
    
    async def extract_business_rules(
        self, 
        domain_model: DomainModel, 
        success_criteria: List[str]
    ) -> List[BusinessRule]:
        """
        Extrai regras de negócio do modelo de domínio.
        
        Args:
            domain_model: Modelo de domínio
            success_criteria: Critérios de sucesso
            
        Returns:
            Lista de regras de negócio extraídas
        """
        logger.info("Starting business rule extraction")
        
        # Extrair regras dos conceitos do domínio
        concept_rules = await self.business_rule_extractor.extract_from_concepts(
            domain_model.concepts
        )
        
        # Extrair regras dos critérios de sucesso
        criteria_rules = await self.business_rule_extractor.extract_from_criteria(
            success_criteria, domain_model
        )
        
        # Extrair regras dos relacionamentos
        relationship_rules = await self.business_rule_extractor.extract_from_relationships(
            domain_model.relationships, domain_model.concepts
        )
        
        # Consolidar e deduplificar regras
        all_rules = concept_rules + criteria_rules + relationship_rules
        consolidated_rules = await self._consolidate_business_rules(all_rules)
        
        # Priorizar regras
        prioritized_rules = await self._prioritize_business_rules(
            consolidated_rules, domain_model
        )
        
        logger.info(f"Business rule extraction completed - {len(prioritized_rules)} rules extracted")
        return prioritized_rules
    
    async def validate_specifications(
        self, 
        semantic_analysis: SemanticAnalysisResult,
        domain_model: DomainModel,
        business_rules: List[BusinessRule]
    ) -> Dict[str, Any]:
        """
        Valida especificações quanto à completude e consistência.
        
        Args:
            semantic_analysis: Análise semântica
            domain_model: Modelo de domínio
            business_rules: Regras de negócio
            
        Returns:
            Relatório de validação
        """
        validation_report = {
            'is_valid': True,
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Validar completude
        completeness_issues = await self._validate_completeness(
            semantic_analysis, domain_model, business_rules
        )
        validation_report['issues'].extend(completeness_issues)
        
        # Validar consistência
        consistency_issues = await self._validate_consistency(
            semantic_analysis, domain_model, business_rules
        )
        validation_report['issues'].extend(consistency_issues)
        
        # Calcular scores
        validation_report['completeness_score'] = await self._calculate_completeness_score(
            semantic_analysis, domain_model
        )
        validation_report['consistency_score'] = await self._calculate_consistency_score(
            domain_model, business_rules
        )
        
        # Determinar validade geral
        validation_report['is_valid'] = (
            validation_report['completeness_score'] > 0.7 and
            validation_report['consistency_score'] > 0.8 and
            len([issue for issue in validation_report['issues'] if issue.get('severity') == 'critical']) == 0
        )
        
        return validation_report
    
    async def _consolidate_semantic_analysis(
        self,
        primary_analysis: Dict[str, Any],
        requirement_analyses: List[Dict[str, Any]]
    ) -> SemanticAnalysisResult:
        """Consolida análises semânticas múltiplas."""
        
        # Consolidar entidades
        all_entities = primary_analysis.get('entities', [])
        for req_analysis in requirement_analyses:
            all_entities.extend(req_analysis.get('entities', []))
        
        # Deduplificar entidades
        unique_entities = await self._deduplicate_entities(all_entities)
        
        # Consolidar intenções
        all_intents = primary_analysis.get('intents', [])
        for req_analysis in requirement_analyses:
            all_intents.extend(req_analysis.get('intents', []))
        
        unique_intents = list(set(all_intents))
        
        # Consolidar requisitos
        all_requirements = []
        for i, req_analysis in enumerate(requirement_analyses):
            req_spec = RequirementSpec(
                id=f"req_{i}",
                text=req_analysis.get('text', ''),
                requirement_type=RequirementType.FUNCTIONAL,  # Default
                priority=req_analysis.get('priority', 1),
                entities=req_analysis.get('entities', []),
                confidence=req_analysis.get('confidence', 0.0)
            )
            all_requirements.append(req_spec)
        
        # Calcular métricas consolidadas
        confidence_score = sum(
            req.confidence for req in all_requirements
        ) / len(all_requirements) if all_requirements else 0.0
        
        return SemanticAnalysisResult(
            entities=unique_entities,
            intents=unique_intents,
            requirements=all_requirements,
            confidence_score=confidence_score,
            ambiguity_score=primary_analysis.get('ambiguity_score', 0.0),
            completeness_score=primary_analysis.get('completeness_score', 0.0)
        )
    
    async def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[SemanticEntity]:
        """Deduplifica entidades semânticas."""
        
        unique_entities = []
        seen_names = set()
        
        for entity_data in entities:
            name = entity_data.get('name', '').lower()
            if name not in seen_names:
                entity = SemanticEntity(
                    name=entity_data.get('name', ''),
                    entity_type=entity_data.get('type', 'unknown'),
                    confidence=entity_data.get('confidence', 0.0),
                    attributes=entity_data.get('attributes', {}),
                    relationships=entity_data.get('relationships', []),
                    context=entity_data.get('context', '')
                )
                unique_entities.append(entity)
                seen_names.add(name)
        
        return unique_entities
    
    async def _identify_specification_gaps(
        self, 
        analysis: SemanticAnalysisResult
    ) -> List[str]:
        """Identifica gaps nas especificações."""
        
        gaps = []
        
        # Verificar entidades sem atributos
        entities_without_attributes = [
            entity.name for entity in analysis.entities 
            if not entity.attributes
        ]
        if entities_without_attributes:
            gaps.append(f"Entities without attributes: {', '.join(entities_without_attributes)}")
        
        # Verificar requisitos sem critérios de aceitação
        reqs_without_criteria = [
            req.id for req in analysis.requirements 
            if not req.acceptance_criteria
        ]
        if reqs_without_criteria:
            gaps.append(f"Requirements without acceptance criteria: {len(reqs_without_criteria)}")
        
        # Verificar baixa confiança geral
        if analysis.confidence_score < 0.6:
            gaps.append("Low overall confidence in analysis")
        
        # Verificar alta ambiguidade
        if analysis.ambiguity_score > 0.7:
            gaps.append("High ambiguity detected in specifications")
        
        return gaps
    
    async def _generate_specification_recommendations(
        self, 
        analysis: SemanticAnalysisResult
    ) -> List[str]:
        """Gera recomendações para melhorar especificações."""
        
        recommendations = []
        
        # Recomendações baseadas em gaps
        if analysis.confidence_score < 0.7:
            recommendations.append("Add more detailed descriptions to improve confidence")
        
        if analysis.completeness_score < 0.8:
            recommendations.append("Consider adding missing functional requirements")
        
        # Recomendações baseadas em entidades
        if len(analysis.entities) < 3:
            recommendations.append("Consider identifying more domain entities")
        
        # Recomendações baseadas em requisitos
        functional_reqs = [r for r in analysis.requirements if r.requirement_type == RequirementType.FUNCTIONAL]
        if len(functional_reqs) < len(analysis.requirements) * 0.6:
            recommendations.append("Consider adding more functional requirements")
        
        return recommendations
    
    async def _extract_domain_name(self, analysis: SemanticAnalysisResult) -> str:
        """Extrai nome do domínio da análise."""
        
        # Buscar por entidades que podem representar o domínio
        domain_candidates = []
        
        for entity in analysis.entities:
            if entity.entity_type in ['domain', 'system', 'application']:
                domain_candidates.append(entity.name)
        
        if domain_candidates:
            return domain_candidates[0]
        
        # Fallback: usar primeira entidade
        if analysis.entities:
            return f"{analysis.entities[0].name} Domain"
        
        return "Unknown Domain"
    
    async def _calculate_cohesion(self, domain_model: DomainModel) -> float:
        """Calcula coesão do modelo de domínio."""
        
        if not domain_model.concepts:
            return 0.0
        
        # Coesão baseada em relacionamentos internos
        total_relationships = 0
        internal_relationships = 0
        
        for concept_name, relationships in domain_model.relationships.items():
            for related_concept, relationship_type in relationships.items():
                total_relationships += 1
                # Verificar se ambos os conceitos estão no mesmo bounded context
                concept_context = self._get_concept_context(concept_name, domain_model)
                related_context = self._get_concept_context(related_concept, domain_model)
                
                if concept_context == related_context:
                    internal_relationships += 1
        
        if total_relationships == 0:
            return 1.0
        
        return internal_relationships / total_relationships
    
    async def _calculate_coupling(self, domain_model: DomainModel) -> float:
        """Calcula acoplamento do modelo de domínio."""
        
        if len(domain_model.concepts) < 2:
            return 0.0
        
        # Acoplamento baseado em relacionamentos entre contextos
        cross_context_relationships = 0
        total_relationships = 0
        
        for concept_name, relationships in domain_model.relationships.items():
            for related_concept, relationship_type in relationships.items():
                total_relationships += 1
                concept_context = self._get_concept_context(concept_name, domain_model)
                related_context = self._get_concept_context(related_concept, domain_model)
                
                if concept_context != related_context:
                    cross_context_relationships += 1
        
        if total_relationships == 0:
            return 0.0
        
        return cross_context_relationships / total_relationships
    
    async def _calculate_domain_complexity(self, domain_model: DomainModel) -> float:
        """Calcula complexidade do modelo de domínio."""
        
        # Complexidade baseada em múltiplos fatores
        concept_complexity = len(domain_model.concepts) / 20.0  # Normalizar por 20 conceitos
        relationship_complexity = len(domain_model.relationships) / 50.0  # Normalizar por 50 relacionamentos
        context_complexity = len(domain_model.bounded_contexts) / 10.0  # Normalizar por 10 contextos
        
        total_complexity = (concept_complexity + relationship_complexity + context_complexity) / 3.0
        
        return min(total_complexity, 1.0)
    
    def _get_concept_context(self, concept_name: str, domain_model: DomainModel) -> str:
        """Obtém o contexto de um conceito."""
        
        # Implementação simplificada - em produção, usar lógica mais sofisticada
        for i, context in enumerate(domain_model.bounded_contexts):
            concepts_in_context = len(domain_model.concepts) // len(domain_model.bounded_contexts)
            if i * concepts_in_context <= hash(concept_name) % len(domain_model.concepts) < (i + 1) * concepts_in_context:
                return context
        
        return domain_model.bounded_contexts[0] if domain_model.bounded_contexts else "default"
    
    async def _consolidate_business_rules(self, rules: List[BusinessRule]) -> List[BusinessRule]:
        """Consolida e deduplifica regras de negócio."""
        
        # Implementação simplificada - deduplificar por descrição
        unique_rules = []
        seen_descriptions = set()
        
        for rule in rules:
            if rule.description not in seen_descriptions:
                unique_rules.append(rule)
                seen_descriptions.add(rule.description)
        
        return unique_rules
    
    async def _prioritize_business_rules(
        self, 
        rules: List[BusinessRule], 
        domain_model: DomainModel
    ) -> List[BusinessRule]:
        """Prioriza regras de negócio."""
        
        # Calcular prioridade baseada em criticidade e impacto
        for rule in rules:
            priority_score = 1
            
            if rule.is_critical:
                priority_score += 3
            
            # Prioridade baseada no número de conceitos afetados
            affected_concepts = len(rule.domain_concepts)
            priority_score += min(affected_concepts, 3)
            
            rule.priority = priority_score
        
        # Ordenar por prioridade
        return sorted(rules, key=lambda r: r.priority, reverse=True)
    
    async def _validate_completeness(
        self,
        semantic_analysis: SemanticAnalysisResult,
        domain_model: DomainModel,
        business_rules: List[BusinessRule]
    ) -> List[Dict[str, Any]]:
        """Valida completude das especificações."""
        
        issues = []
        
        # Verificar se há entidades suficientes
        if len(semantic_analysis.entities) < 3:
            issues.append({
                'type': 'completeness',
                'severity': 'warning',
                'message': 'Few domain entities identified'
            })
        
        # Verificar se há requisitos suficientes
        if len(semantic_analysis.requirements) < 5:
            issues.append({
                'type': 'completeness',
                'severity': 'warning',
                'message': 'Few requirements specified'
            })
        
        # Verificar se há regras de negócio
        if len(business_rules) == 0:
            issues.append({
                'type': 'completeness',
                'severity': 'critical',
                'message': 'No business rules identified'
            })
        
        return issues
    
    async def _validate_consistency(
        self,
        semantic_analysis: SemanticAnalysisResult,
        domain_model: DomainModel,
        business_rules: List[BusinessRule]
    ) -> List[Dict[str, Any]]:
        """Valida consistência das especificações."""
        
        issues = []
        
        # Verificar consistência entre entidades e conceitos
        entity_names = {entity.name.lower() for entity in semantic_analysis.entities}
        concept_names = {concept.name.lower() for concept in domain_model.concepts}
        
        missing_concepts = entity_names - concept_names
        if missing_concepts:
            issues.append({
                'type': 'consistency',
                'severity': 'warning',
                'message': f'Entities without corresponding concepts: {missing_concepts}'
            })
        
        # Verificar relacionamentos órfãos
        all_concept_names = {concept.name for concept in domain_model.concepts}
        for concept_name, relationships in domain_model.relationships.items():
            if concept_name not in all_concept_names:
                issues.append({
                    'type': 'consistency',
                    'severity': 'error',
                    'message': f'Orphaned relationship for concept: {concept_name}'
                })
        
        return issues
    
    async def _calculate_completeness_score(
        self,
        semantic_analysis: SemanticAnalysisResult,
        domain_model: DomainModel
    ) -> float:
        """Calcula score de completude."""
        
        score = 0.0
        
        # Score baseado em entidades
        if len(semantic_analysis.entities) >= 5:
            score += 0.3
        elif len(semantic_analysis.entities) >= 3:
            score += 0.2
        
        # Score baseado em requisitos
        if len(semantic_analysis.requirements) >= 10:
            score += 0.3
        elif len(semantic_analysis.requirements) >= 5:
            score += 0.2
        
        # Score baseado em conceitos do domínio
        if len(domain_model.concepts) >= 5:
            score += 0.2
        elif len(domain_model.concepts) >= 3:
            score += 0.1
        
        # Score baseado em relacionamentos
        if len(domain_model.relationships) >= 3:
            score += 0.2
        elif len(domain_model.relationships) >= 1:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _calculate_consistency_score(
        self,
        domain_model: DomainModel,
        business_rules: List[BusinessRule]
    ) -> float:
        """Calcula score de consistência."""
        
        score = 1.0  # Começar com score máximo
        
        # Penalizar relacionamentos órfãos
        all_concept_names = {concept.name for concept in domain_model.concepts}
        orphaned_relationships = 0
        
        for concept_name in domain_model.relationships.keys():
            if concept_name not in all_concept_names:
                orphaned_relationships += 1
        
        if orphaned_relationships > 0:
            score -= min(orphaned_relationships * 0.1, 0.3)
        
        # Penalizar regras sem conceitos associados
        rules_without_concepts = sum(
            1 for rule in business_rules if not rule.domain_concepts
        )
        
        if rules_without_concepts > 0:
            score -= min(rules_without_concepts * 0.05, 0.2)
        
        return max(score, 0.0)
    
    def _generate_cache_key(self, description: str, requirements: List[str]) -> str:
        """Gera chave de cache para análise."""
        
        content = description + "|".join(requirements)
        return str(hash(content))
    
    async def shutdown(self) -> None:
        """Desliga o córtex de especificação."""
        
        logger.info("Shutting down Specification Cortex")
        
        # Limpar caches
        self.analysis_cache.clear()
        self.domain_model_cache.clear()
        
        # Desligar componentes
        await self.semantic_analyzer.shutdown()
        await self.domain_modeler.shutdown()
        await self.business_rule_extractor.shutdown()
        
        logger.info("Specification Cortex shutdown complete")


class SemanticAnalyzer:
    """Analisador semântico para processamento de linguagem natural."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analisa texto e extrai informações semânticas."""
        
        # Implementação simplificada - em produção usar modelos NLP avançados
        entities = await self._extract_entities(text)
        intents = await self._classify_intents(text)
        
        return {
            'entities': entities,
            'intents': intents,
            'confidence': 0.8,
            'ambiguity_score': 0.3,
            'completeness_score': 0.7
        }
    
    async def analyze_requirement(self, requirement: str) -> Dict[str, Any]:
        """Analisa um requisito específico."""
        
        entities = await self._extract_entities(requirement)
        
        return {
            'text': requirement,
            'entities': entities,
            'confidence': 0.8,
            'priority': 1
        }
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extrai entidades do texto."""
        
        # Implementação simplificada usando regex
        entities = []
        
        # Buscar por substantivos capitalizados (possíveis entidades)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(entity_pattern, text)
        
        for match in matches:
            entities.append({
                'name': match,
                'type': 'entity',
                'confidence': 0.7,
                'attributes': {},
                'relationships': []
            })
        
        return entities
    
    async def _classify_intents(self, text: str) -> List[str]:
        """Classifica intenções no texto."""
        
        intents = []
        text_lower = text.lower()
        
        # Mapeamento simples de palavras-chave para intenções
        intent_keywords = {
            'create': ['create', 'build', 'develop', 'implement', 'make'],
            'update': ['update', 'modify', 'change', 'edit', 'improve'],
            'delete': ['delete', 'remove', 'eliminate'],
            'query': ['search', 'find', 'query', 'retrieve', 'get'],
            'integrate': ['integrate', 'connect', 'link', 'combine'],
            'optimize': ['optimize', 'improve', 'enhance', 'speed up'],
            'secure': ['secure', 'protect', 'authenticate', 'authorize'],
            'scale': ['scale', 'expand', 'grow', 'increase capacity']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                intents.append(intent)
        
        return intents
    
    async def shutdown(self) -> None:
        """Desliga o analisador semântico."""
        pass


class DomainModeler:
    """Modelador de domínio para criar modelos conceituais."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def extract_concepts(
        self, 
        entities: List[SemanticEntity], 
        constraints: Dict[str, Any]
    ) -> List[DomainConcept]:
        """Extrai conceitos do domínio das entidades."""
        
        concepts = []
        
        for entity in entities:
            concept = DomainConcept(
                name=entity.name,
                description=f"Domain concept for {entity.name}",
                concept_type="entity",  # Default type
                attributes=entity.attributes,
                relationships=entity.relationships,
                complexity_score=0.5,
                importance_score=entity.confidence
            )
            concepts.append(concept)
        
        return concepts
    
    async def identify_relationships(
        self, 
        concepts: List[DomainConcept], 
        requirements: List[RequirementSpec]
    ) -> Dict[str, Dict[str, str]]:
        """Identifica relacionamentos entre conceitos."""
        
        relationships = {}
        
        # Implementação simplificada - criar relacionamentos baseados em proximidade
        for i, concept1 in enumerate(concepts):
            relationships[concept1.name] = {}
            
            for j, concept2 in enumerate(concepts):
                if i != j and j < i + 2:  # Relacionar com próximos conceitos
                    relationships[concept1.name][concept2.name] = "associates_with"
        
        return relationships
    
    async def identify_bounded_contexts(
        self, 
        concepts: List[DomainConcept], 
        relationships: Dict[str, Dict[str, str]]
    ) -> List[str]:
        """Identifica bounded contexts."""
        
        # Implementação simplificada - criar contextos baseados em agrupamento
        contexts = []
        
        if len(concepts) <= 3:
            contexts.append("Core Context")
        else:
            contexts.extend(["Core Context", "Supporting Context", "Generic Context"])
        
        return contexts
    
    async def identify_aggregates(
        self, 
        concepts: List[DomainConcept], 
        relationships: Dict[str, Dict[str, str]]
    ) -> List[str]:
        """Identifica aggregates do domínio."""
        
        # Implementação simplificada
        aggregates = []
        
        for concept in concepts:
            if concept.importance_score > 0.7:
                aggregates.append(f"{concept.name} Aggregate")
        
        return aggregates
    
    async def shutdown(self) -> None:
        """Desliga o modelador de domínio."""
        pass


class BusinessRuleExtractor:
    """Extrator de regras de negócio."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def extract_from_concepts(self, concepts: List[DomainConcept]) -> List[BusinessRule]:
        """Extrai regras dos conceitos do domínio."""
        
        rules = []
        
        for concept in concepts:
            # Criar regra de validação básica para cada conceito
            rule = BusinessRule(
                id=f"rule_{concept.name.lower()}_validation",
                description=f"Validation rules for {concept.name}",
                rule_type="validation",
                domain_concepts=[concept.name],
                priority=1 if concept.importance_score > 0.7 else 2
            )
            rules.append(rule)
        
        return rules
    
    async def extract_from_criteria(
        self, 
        criteria: List[str], 
        domain_model: DomainModel
    ) -> List[BusinessRule]:
        """Extrai regras dos critérios de sucesso."""
        
        rules = []
        
        for i, criterion in enumerate(criteria):
            rule = BusinessRule(
                id=f"success_rule_{i}",
                description=f"Success criterion: {criterion}",
                rule_type="constraint",
                is_critical=True,
                priority=1
            )
            rules.append(rule)
        
        return rules
    
    async def extract_from_relationships(
        self, 
        relationships: Dict[str, Dict[str, str]], 
        concepts: List[DomainConcept]
    ) -> List[BusinessRule]:
        """Extrai regras dos relacionamentos."""
        
        rules = []
        
        for concept_name, related_concepts in relationships.items():
            for related_name, relationship_type in related_concepts.items():
                rule = BusinessRule(
                    id=f"relationship_rule_{concept_name}_{related_name}",
                    description=f"{concept_name} {relationship_type} {related_name}",
                    rule_type="workflow",
                    domain_concepts=[concept_name, related_name],
                    priority=2
                )
                rules.append(rule)
        
        return rules
    
    async def shutdown(self) -> None:
        """Desliga o extrator de regras."""
        pass
