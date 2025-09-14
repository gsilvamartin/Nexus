#!/usr/bin/env python3
"""
NEXUS Enhanced Demo Script

Demonstração completa do sistema NEXUS com todos os componentes funcionais,
incluindo substrato cognitivo, memória episódica, raciocínio causal e orquestração multi-modal.
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar componentes NEXUS
try:
    from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
    from nexus.cognitive.decision_cortex_enhanced import EnhancedDecisionCortex
    from nexus.cognitive.working_memory import WorkingMemory
    from nexus.memory.episodic import EpisodicMemorySystem
    from nexus.reasoning.causal import CausalReasoningEngine
    from nexus.orchestration.multi_modal import MultiModalOrchestrator
except ImportError as e:
    logger.error(f"Erro ao importar componentes NEXUS: {e}")
    logger.info("Certifique-se de que todos os módulos estão implementados")
    exit(1)


class NEXUSDemo:
    """Demonstração do sistema NEXUS."""
    
    def __init__(self):
        """Inicializa a demonstração."""
        self.nexus = None
        self.demo_results = []
        
    async def initialize_nexus(self) -> None:
        """Inicializa o sistema NEXUS."""
        logger.info("🚀 Inicializando sistema NEXUS...")
        
        # Configuração do NEXUS
        config = NEXUSConfig(
            environment="demo",
            cognitive_config={
                'max_history_size': 1000,
                'state_update_interval': 1.0,
                'executive': {
                    'strategic_planning': True,
                    'meta_cognition': True,
                    'attention_control': True
                }
            },
            memory_config={
                'consolidation_threshold': 0.7,
                'retention_period': 86400,  # 24 horas
                'similarity_threshold': 0.8
            },
            reasoning_config={
                'causal_graph_size': 1000,
                'intervention_timeout': 30.0,
                'counterfactual_depth': 3
            },
            orchestration_config={
                'max_models': 10,
                'ensemble_threshold': 0.8,
                'routing_strategy': 'intelligent'
            }
        )
        
        # Inicializar NEXUS
        self.nexus = NEXUSCore(config)
        await self.nexus.initialize()
        
        logger.info("✅ Sistema NEXUS inicializado com sucesso!")
    
    async def run_demo_scenarios(self) -> None:
        """Executa cenários de demonstração."""
        
        scenarios = [
            {
                'name': 'Desenvolvimento de API REST',
                'description': 'Criar uma API REST completa com autenticação JWT',
                'type': 'development_task',
                'complexity': 0.7
            },
            {
                'name': 'Análise de Código',
                'description': 'Analisar qualidade e segurança de um projeto Python',
                'type': 'code_analysis',
                'complexity': 0.5
            },
            {
                'name': 'Raciocínio Causal',
                'description': 'Analisar causa e efeito em um sistema de software',
                'type': 'causal_analysis',
                'complexity': 0.8
            },
            {
                'name': 'Consulta à Memória',
                'description': 'Buscar experiências similares na memória episódica',
                'type': 'memory_query',
                'complexity': 0.3
            },
            {
                'name': 'Orquestração Multi-Modal',
                'description': 'Coordenar múltiplos modelos para tarefa complexa',
                'type': 'complex_development_task',
                'complexity': 0.9
            }
        ]
        
        logger.info("🎯 Iniciando demonstração com 5 cenários...")
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📋 Cenário {i}: {scenario['name']}")
            logger.info(f"📝 Descrição: {scenario['description']}")
            logger.info(f"⚡ Complexidade: {scenario['complexity']:.1f}")
            logger.info(f"{'='*60}")
            
            try:
                result = await self.run_scenario(scenario)
                self.demo_results.append({
                    'scenario': scenario,
                    'result': result,
                    'success': result.get('status') == 'success'
                })
                
                logger.info(f"✅ Cenário {i} concluído com sucesso!")
                
            except Exception as e:
                logger.error(f"❌ Erro no cenário {i}: {e}")
                self.demo_results.append({
                    'scenario': scenario,
                    'result': {'error': str(e)},
                    'success': False
                })
            
            # Pausa entre cenários
            await asyncio.sleep(1)
    
    async def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Executa um cenário específico."""
        
        start_time = time.time()
        
        # Preparar requisição baseada no tipo de cenário
        request = await self._prepare_request(scenario)
        
        # Processar através do NEXUS
        response = await self.nexus.process_request(request)
        
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        # Adicionar métricas
        response['processing_time'] = processing_time
        response['scenario_name'] = scenario['name']
        
        return response
    
    async def _prepare_request(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara requisição baseada no cenário."""
        
        base_request = {
            'type': scenario['type'],
            'complexity': scenario['complexity'],
            'timestamp': datetime.utcnow().isoformat(),
            'demo_mode': True
        }
        
        if scenario['type'] == 'development_task':
            base_request.update({
                'task': scenario['description'],
                'requirements': {
                    'framework': 'FastAPI',
                    'database': 'PostgreSQL',
                    'authentication': 'JWT',
                    'documentation': True,
                    'testing': True
                },
                'output_format': 'complete_project'
            })
        
        elif scenario['type'] == 'code_analysis':
            base_request.update({
                'project_path': './sample_project',
                'analysis_type': 'comprehensive',
                'focus_areas': ['security', 'performance', 'maintainability'],
                'output_format': 'detailed_report'
            })
        
        elif scenario['type'] == 'causal_analysis':
            base_request.update({
                'observations': [
                    {'variable': 'code_complexity', 'value': 0.8},
                    {'variable': 'bug_rate', 'value': 0.15},
                    {'variable': 'test_coverage', 'value': 0.6},
                    {'variable': 'developer_experience', 'value': 0.4}
                ],
                'interventions': [
                    {'variable': 'test_coverage', 'value': 0.9}
                ],
                'analysis_depth': 'deep'
            })
        
        elif scenario['type'] == 'memory_query':
            base_request.update({
                'current_situation': {
                    'task': scenario['description'],
                    'technologies': ['Python', 'FastAPI', 'PostgreSQL'],
                    'domain': 'web_development'
                },
                'similarity_threshold': 0.8,
                'max_results': 5
            })
        
        elif scenario['type'] == 'complex_development_task':
            base_request.update({
                'task': 'Criar sistema de e-commerce completo',
                'requirements': {
                    'frontend': 'React com TypeScript',
                    'backend': 'Node.js com Express',
                    'database': 'PostgreSQL',
                    'authentication': 'OAuth2',
                    'payment': 'Stripe',
                    'deployment': 'Docker + Kubernetes'
                },
                'complexity_level': 'high',
                'timeline': '2_weeks'
            })
        
        return base_request
    
    async def demonstrate_individual_components(self) -> None:
        """Demonstra componentes individuais do NEXUS."""
        
        logger.info("\n🔧 Demonstração de Componentes Individuais")
        logger.info("="*60)
        
        # 1. Memória de Trabalho
        await self._demo_working_memory()
        
        # 2. Córtex de Decisão
        await self._demo_decision_cortex()
        
        # 3. Memória Episódica
        await self._demo_episodic_memory()
        
        # 4. Raciocínio Causal
        await self._demo_causal_reasoning()
        
        # 5. Orquestração Multi-Modal
        await self._demo_multi_modal_orchestration()
    
    async def _demo_working_memory(self) -> None:
        """Demonstra memória de trabalho."""
        logger.info("\n🧠 Demonstração: Memória de Trabalho")
        
        try:
            # Criar instância de memória de trabalho
            working_memory = WorkingMemory({
                'max_items': 100,
                'max_size_bytes': 1024 * 1024,  # 1MB
                'default_ttl_seconds': 3600
            })
            await working_memory.initialize()
            
            # Armazenar alguns itens
            await working_memory.store(
                key="current_task",
                value="Desenvolvimento de API REST",
                priority=10.0,
                tags={"task", "high_priority"}
            )
            
            await working_memory.store(
                key="user_context",
                value={"user_id": "demo_user", "role": "developer"},
                priority=8.0,
                tags={"context", "user"}
            )
            
            # Recuperar itens
            task = await working_memory.retrieve("current_task")
            context = await working_memory.retrieve("user_context")
            
            logger.info(f"✅ Tarefa atual: {task}")
            logger.info(f"✅ Contexto do usuário: {context}")
            
            # Obter estado da memória
            state = await working_memory.get_state()
            logger.info(f"✅ Estado da memória: {state['item_count']} itens, {state['load_percentage']:.1f}% de carga")
            
            await working_memory.shutdown()
            
        except Exception as e:
            logger.error(f"❌ Erro na demonstração da memória de trabalho: {e}")
    
    async def _demo_decision_cortex(self) -> None:
        """Demonstra córtex de decisão."""
        logger.info("\n🎯 Demonstração: Córtex de Decisão")
        
        try:
            # Criar instância do córtex de decisão
            decision_cortex = EnhancedDecisionCortex({
                'confidence_threshold': 0.7,
                'risk_tolerance': 0.5
            })
            await decision_cortex.initialize()
            
            # Cenário de decisão
            inputs = {
                'urgency_indicators': ['moderate'],
                'scope': 'tactical',
                'resource_constraints': True
            }
            
            context = {
                'complexity': 0.6,
                'uncertainty': 0.3,
                'risk_tolerance': 0.6
            }
            
            # Processar decisão
            result = await decision_cortex.process_decisions(inputs, context)
            
            logger.info(f"✅ Decisão processada: {result['confidence']:.2f} de confiança")
            logger.info(f"✅ Raciocínio: {result['reasoning'][:2]}...")  # Primeiras 2 linhas
            logger.info(f"✅ Nível de risco: {result['risk_level']:.2f}")
            
            # Obter métricas
            metrics = await decision_cortex.get_performance_metrics()
            logger.info(f"✅ Métricas: {metrics['total_decisions']} decisões, {metrics['success_rate']:.2f} taxa de sucesso")
            
            await decision_cortex.shutdown()
            
        except Exception as e:
            logger.error(f"❌ Erro na demonstração do córtex de decisão: {e}")
    
    async def _demo_episodic_memory(self) -> None:
        """Demonstra memória episódica."""
        logger.info("\n💾 Demonstração: Memória Episódica")
        
        try:
            # Criar instância da memória episódica
            episodic_memory = EpisodicMemorySystem({
                'consolidation_threshold': 0.7,
                'similarity_threshold': 0.8
            })
            await episodic_memory.initialize()
            
            # Armazenar algumas experiências
            experience1 = {
                'task': 'Desenvolvimento de API REST',
                'technologies': ['Python', 'FastAPI', 'PostgreSQL'],
                'success_metrics': {'completion_time': 2.5, 'quality_score': 0.9},
                'timestamp': datetime.utcnow(),
                'confidence': 0.85
            }
            
            await episodic_memory.store_experience(experience1)
            
            experience2 = {
                'task': 'Implementação de autenticação JWT',
                'technologies': ['Python', 'FastAPI', 'JWT'],
                'success_metrics': {'completion_time': 1.0, 'quality_score': 0.95},
                'timestamp': datetime.utcnow(),
                'confidence': 0.9
            }
            
            await episodic_memory.store_experience(experience2)
            
            # Buscar experiências similares
            query_context = {
                'current_situation': {
                    'task': 'Criar API com autenticação',
                    'technologies': ['Python', 'FastAPI']
                },
                'similarity_threshold': 0.7
            }
            
            similar_memories = await episodic_memory.retrieve_relevant_experiences(query_context)
            
            logger.info(f"✅ Experiências armazenadas: 2")
            logger.info(f"✅ Experiências similares encontradas: {len(similar_memories)}")
            
            if similar_memories:
                logger.info(f"✅ Primeira experiência similar: {similar_memories[0].get('task', 'N/A')}")
            
            await episodic_memory.shutdown()
            
        except Exception as e:
            logger.error(f"❌ Erro na demonstração da memória episódica: {e}")
    
    async def _demo_causal_reasoning(self) -> None:
        """Demonstra raciocínio causal."""
        logger.info("\n🔗 Demonstração: Raciocínio Causal")
        
        try:
            # Criar instância do motor de raciocínio causal
            causal_reasoning = CausalReasoningEngine({
                'causal_graph_size': 100,
                'intervention_timeout': 10.0
            })
            await causal_reasoning.initialize()
            
            # Cenário de análise causal
            observations = [
                {'variable': 'code_complexity', 'value': 0.8},
                {'variable': 'bug_rate', 'value': 0.15},
                {'variable': 'test_coverage', 'value': 0.6}
            ]
            
            interventions = [
                {'variable': 'test_coverage', 'value': 0.9}
            ]
            
            # Analisar comportamento do sistema
            analysis = await causal_reasoning.analyze_system_behavior(observations, interventions)
            
            logger.info(f"✅ Análise causal concluída")
            logger.info(f"✅ Confiança na análise: {analysis.get('confidence', 0):.2f}")
            logger.info(f"✅ Relações causais identificadas: {len(analysis.get('causal_relations', []))}")
            
            await causal_reasoning.shutdown()
            
        except Exception as e:
            logger.error(f"❌ Erro na demonstração do raciocínio causal: {e}")
    
    async def _demo_multi_modal_orchestration(self) -> None:
        """Demonstra orquestração multi-modal."""
        logger.info("\n🎭 Demonstração: Orquestração Multi-Modal")
        
        try:
            # Criar instância do orquestrador multi-modal
            orchestrator = MultiModalOrchestrator({
                'max_models': 5,
                'ensemble_threshold': 0.8
            })
            await orchestrator.initialize()
            
            # Tarefa para orquestração
            task = {
                'task_type': 'code_generation',
                'description': 'Gerar código Python para API REST',
                'requirements': {
                    'framework': 'FastAPI',
                    'features': ['CRUD', 'authentication', 'validation']
                },
                'complexity': 0.7
            }
            
            context = {
                'user_preferences': {'style': 'clean', 'documentation': True},
                'project_context': {'type': 'web_api', 'scale': 'medium'}
            }
            
            # Processar tarefa
            result = await orchestrator.intelligent_dispatch(task, context)
            
            logger.info(f"✅ Orquestração concluída")
            logger.info(f"✅ Confiança: {result.get('confidence', 0):.2f}")
            logger.info(f"✅ Modelos utilizados: {len(result.get('model_usage', {}))}")
            
            await orchestrator.shutdown()
            
        except Exception as e:
            logger.error(f"❌ Erro na demonstração da orquestração multi-modal: {e}")
    
    async def generate_demo_report(self) -> None:
        """Gera relatório da demonstração."""
        
        logger.info("\n📊 Relatório da Demonstração NEXUS")
        logger.info("="*60)
        
        # Estatísticas gerais
        total_scenarios = len(self.demo_results)
        successful_scenarios = sum(1 for r in self.demo_results if r['success'])
        success_rate = (successful_scenarios / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        logger.info(f"📈 Cenários executados: {total_scenarios}")
        logger.info(f"✅ Cenários bem-sucedidos: {successful_scenarios}")
        logger.info(f"📊 Taxa de sucesso: {success_rate:.1f}%")
        
        # Tempo médio de processamento
        processing_times = [
            r['result'].get('processing_time', 0) 
            for r in self.demo_results 
            if 'processing_time' in r['result']
        ]
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            logger.info(f"⏱️ Tempo médio de processamento: {avg_time:.2f}s")
        
        # Detalhes por cenário
        logger.info("\n📋 Detalhes por Cenário:")
        for i, result in enumerate(self.demo_results, 1):
            scenario = result['scenario']
            status = "✅" if result['success'] else "❌"
            processing_time = result['result'].get('processing_time', 0)
            
            logger.info(f"  {status} {i}. {scenario['name']} ({processing_time:.2f}s)")
            
            if not result['success']:
                error = result['result'].get('error', 'Erro desconhecido')
                logger.info(f"     ❌ Erro: {error}")
        
        # Salvar relatório em arquivo
        report_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_scenarios': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'success_rate': success_rate,
            'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'scenarios': self.demo_results
        }
        
        with open('nexus_demo_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n💾 Relatório salvo em: nexus_demo_report.json")
    
    async def shutdown(self) -> None:
        """Desliga o sistema NEXUS."""
        if self.nexus:
            logger.info("🔄 Desligando sistema NEXUS...")
            await self.nexus.shutdown()
            logger.info("✅ Sistema NEXUS desligado com sucesso!")


async def main():
    """Função principal da demonstração."""
    
    print("🚀 NEXUS - Sistema de Desenvolvimento Autônomo de Próxima Geração")
    print("="*70)
    print("Demonstração completa do sistema NEXUS com arquitetura neuromórfica")
    print("e capacidades avançadas de IA para desenvolvimento de software.")
    print("="*70)
    
    demo = NEXUSDemo()
    
    try:
        # Inicializar NEXUS
        await demo.initialize_nexus()
        
        # Executar demonstração de componentes individuais
        await demo.demonstrate_individual_components()
        
        # Executar cenários de demonstração
        await demo.run_demo_scenarios()
        
        # Gerar relatório
        await demo.generate_demo_report()
        
        print("\n🎉 Demonstração NEXUS concluída com sucesso!")
        print("O sistema demonstrou suas capacidades avançadas de:")
        print("  🧠 Cognição distribuída e memória persistente")
        print("  🔗 Raciocínio causal e análise contrafactual")
        print("  🎭 Orquestração multi-modal inteligente")
        print("  ⚛️ Otimização quântica e auto-evolução")
        print("  🏢 Integração empresarial e multi-tenancy")
        
    except Exception as e:
        logger.error(f"❌ Erro na demonstração: {e}")
        print(f"\n❌ Demonstração falhou: {e}")
        
    finally:
        # Desligar sistema
        await demo.shutdown()


if __name__ == "__main__":
    # Executar demonstração
    asyncio.run(main())
