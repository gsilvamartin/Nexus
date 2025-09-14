#!/usr/bin/env python3
"""
Teste Básico do NEXUS

Testa os componentes básicos do sistema NEXUS para verificar se estão funcionando corretamente.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_working_memory():
    """Testa a memória de trabalho."""
    logger.info("🧠 Testando Memória de Trabalho...")
    
    try:
        from nexus.cognitive.working_memory import WorkingMemory
        
        # Criar instância
        wm = WorkingMemory({
            'max_items': 10,
            'max_size_bytes': 1024 * 100,  # 100KB
            'default_ttl_seconds': 60
        })
        
        await wm.initialize()
        
        # Teste básico
        await wm.store("test_key", "test_value", priority=5.0)
        value = await wm.retrieve("test_key")
        
        assert value == "test_value", f"Valor esperado 'test_value', obtido '{value}'"
        
        # Teste de contexto
        await wm.update_context({"test_context": "working"})
        context = await wm.get_current_context()
        
        assert "test_context" in context, "Contexto não foi atualizado"
        
        await wm.shutdown()
        logger.info("✅ Memória de Trabalho: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memória de Trabalho: FALHOU - {e}")
        return False


async def test_decision_cortex():
    """Testa o córtex de decisão."""
    logger.info("🎯 Testando Córtex de Decisão...")
    
    try:
        from nexus.cognitive.decision_cortex_enhanced import EnhancedDecisionCortex
        
        # Criar instância
        dc = EnhancedDecisionCortex({
            'confidence_threshold': 0.7,
            'risk_tolerance': 0.5
        })
        
        await dc.initialize()
        
        # Teste básico
        inputs = {
            'urgency_indicators': ['moderate'],
            'scope': 'operational'
        }
        
        context = {
            'complexity': 0.5,
            'uncertainty': 0.3
        }
        
        result = await dc.process_decisions(inputs, context)
        
        assert 'decisions' in result, "Resultado deve conter 'decisions'"
        assert 'confidence' in result, "Resultado deve conter 'confidence'"
        assert 'reasoning' in result, "Resultado deve conter 'reasoning'"
        
        # Verificar métricas
        metrics = await dc.get_performance_metrics()
        assert 'total_decisions' in metrics, "Métricas devem conter 'total_decisions'"
        
        await dc.shutdown()
        logger.info("✅ Córtex de Decisão: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Córtex de Decisão: FALHOU - {e}")
        return False


async def test_episodic_memory():
    """Testa a memória episódica."""
    logger.info("💾 Testando Memória Episódica...")
    
    try:
        from nexus.memory.episodic import EpisodicMemorySystem
        
        # Criar instância
        em = EpisodicMemorySystem({
            'consolidation_threshold': 0.7,
            'similarity_threshold': 0.8
        })
        
        await em.initialize()
        
        # Teste básico
        experience = {
            'task': 'Teste de memória episódica',
            'technologies': ['Python', 'Test'],
            'success_metrics': {'completion_time': 1.0, 'quality_score': 0.9},
            'timestamp': datetime.utcnow(),
            'confidence': 0.8
        }
        
        memory_id = await em.store_experience(experience)
        assert memory_id is not None, "ID da memória não deve ser None"
        
        # Teste de busca
        query_context = {
            'current_situation': {
                'task': 'Teste de memória',
                'technologies': ['Python']
            },
            'similarity_threshold': 0.7
        }
        
        similar_memories = await em.retrieve_relevant_experiences(query_context)
        assert isinstance(similar_memories, list), "Resultado deve ser uma lista"
        
        await em.shutdown()
        logger.info("✅ Memória Episódica: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memória Episódica: FALHOU - {e}")
        return False


async def test_causal_reasoning():
    """Testa o raciocínio causal."""
    logger.info("🔗 Testando Raciocínio Causal...")
    
    try:
        from nexus.reasoning.causal import CausalReasoningEngine
        
        # Criar instância
        cr = CausalReasoningEngine({
            'causal_graph_size': 100,
            'intervention_timeout': 10.0
        })
        
        await cr.initialize()
        
        # Teste básico
        observations = [
            {'variable': 'test_var1', 'value': 0.5},
            {'variable': 'test_var2', 'value': 0.7}
        ]
        
        interventions = [
            {'variable': 'test_var1', 'value': 0.8}
        ]
        
        analysis = await cr.analyze_system_behavior(observations, interventions)
        
        assert isinstance(analysis, dict), "Análise deve ser um dicionário"
        assert 'confidence' in analysis, "Análise deve conter 'confidence'"
        
        await cr.shutdown()
        logger.info("✅ Raciocínio Causal: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Raciocínio Causal: FALHOU - {e}")
        return False


async def test_multi_modal_orchestration():
    """Testa a orquestração multi-modal."""
    logger.info("🎭 Testando Orquestração Multi-Modal...")
    
    try:
        from nexus.orchestration.multi_modal import MultiModalOrchestrator
        
        # Criar instância
        mmo = MultiModalOrchestrator({
            'max_models': 3,
            'ensemble_threshold': 0.8
        })
        
        await mmo.initialize()
        
        # Teste básico
        task = {
            'task_type': 'test_task',
            'description': 'Tarefa de teste',
            'complexity': 0.5
        }
        
        context = {
            'test_context': True
        }
        
        result = await mmo.intelligent_dispatch(task, context)
        
        assert isinstance(result, dict), "Resultado deve ser um dicionário"
        assert 'confidence' in result, "Resultado deve conter 'confidence'"
        
        await mmo.shutdown()
        logger.info("✅ Orquestração Multi-Modal: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Orquestração Multi-Modal: FALHOU - {e}")
        return False


async def test_nexus_core():
    """Testa o núcleo do NEXUS."""
    logger.info("🚀 Testando NEXUS Core...")
    
    try:
        from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
        
        # Configuração mínima
        config = NEXUSConfig(
            environment="test",
            cognitive_config={'max_history_size': 100},
            memory_config={'consolidation_threshold': 0.7},
            reasoning_config={'causal_graph_size': 100},
            orchestration_config={'max_models': 3}
        )
        
        # Criar instância
        nexus = NEXUSCore(config)
        await nexus.initialize()
        
        # Teste básico
        request = {
            'type': 'test_request',
            'content': 'Teste do NEXUS Core',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        response = await nexus.process_request(request)
        
        assert isinstance(response, dict), "Resposta deve ser um dicionário"
        assert 'status' in response, "Resposta deve conter 'status'"
        
        # Verificar status do sistema
        status = await nexus.get_system_status()
        assert 'status' in status, "Status deve conter 'status'"
        assert 'health_score' in status, "Status deve conter 'health_score'"
        
        await nexus.shutdown()
        logger.info("✅ NEXUS Core: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ NEXUS Core: FALHOU - {e}")
        return False


async def run_all_tests():
    """Executa todos os testes."""
    
    print("🧪 NEXUS - Teste Básico dos Componentes")
    print("="*50)
    
    tests = [
        ("Memória de Trabalho", test_working_memory),
        ("Córtex de Decisão", test_decision_cortex),
        ("Memória Episódica", test_episodic_memory),
        ("Raciocínio Causal", test_causal_reasoning),
        ("Orquestração Multi-Modal", test_multi_modal_orchestration),
        ("NEXUS Core", test_nexus_core)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Executando: {test_name}")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Relatório final
    print("\n" + "="*50)
    print("📊 RELATÓRIO FINAL DOS TESTES")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    failed_tests = total_tests - passed_tests
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Total: {total_tests}")
    print(f"✅ Passou: {passed_tests}")
    print(f"❌ Falhou: {failed_tests}")
    print(f"📊 Taxa de sucesso: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 Todos os testes passaram! O sistema NEXUS está funcionando corretamente.")
        return True
    else:
        print(f"\n⚠️ {failed_tests} teste(s) falharam. Verifique os logs para mais detalhes.")
        return False


async def main():
    """Função principal."""
    
    try:
        success = await run_all_tests()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
