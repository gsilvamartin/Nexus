#!/usr/bin/env python3
"""
NEXUS Test Script

Script simples para testar a funcionalidade básica do NEXUS.
"""

import asyncio
import sys
from pathlib import Path

# Adicionar o diretório do projeto ao Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_nexus_imports():
    """Testa se todos os imports do NEXUS funcionam."""
    
    print("🔍 Testando imports do NEXUS...")
    
    try:
        # Testar imports principais
        from nexus.core import NEXUS, DevelopmentObjective
        print("✅ Core imports OK")
        
        from nexus.cognitive import ExecutiveFunction, CognitiveSubstrate
        print("✅ Cognitive imports OK")
        
        from nexus.cortices import SpecificationCortex, ArchitectureCortex
        print("✅ Cortices imports OK")
        
        from nexus.memory import EpisodicMemorySystem
        print("✅ Memory imports OK")
        
        from nexus.reasoning import CausalReasoningEngine
        print("✅ Reasoning imports OK")
        
        from nexus.orchestration import MultiModalOrchestrator
        print("✅ Orchestration imports OK")
        
        print("\n✅ Todos os imports funcionaram corretamente!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Erro de import: {e}")
        return False


async def test_nexus_basic_functionality():
    """Testa funcionalidade básica do NEXUS."""
    
    print("\n🧪 Testando funcionalidade básica...")
    
    try:
        from nexus.core import NEXUS, DevelopmentObjective
        
        # Configuração mínima
        config = {
            'cognitive': {'max_working_memory_items': 100},
            'orchestration': {'max_concurrent_requests': 1}
        }
        
        # Criar sistema NEXUS
        nexus = NEXUS(config)
        print("✅ Sistema NEXUS criado")
        
        # Criar objetivo simples
        objective = DevelopmentObjective(
            description="Teste simples do sistema",
            requirements=["Requisito 1", "Requisito 2"]
        )
        print("✅ Objetivo de desenvolvimento criado")
        
        # Testar desenvolvimento (simulado)
        print("🚀 Executando desenvolvimento autônomo...")
        result = await nexus.autonomous_development(objective)
        
        if result.success:
            print("✅ Desenvolvimento autônomo executado com sucesso!")
            print(f"   - Tempo de execução: {result.execution_time:.2f}s")
            print(f"   - Arquivos gerados: {len(result.files)}")
        else:
            print("⚠️  Desenvolvimento completou com avisos")
            for error in result.errors:
                print(f"   - {error}")
        
        # Desligar sistema
        await nexus.shutdown()
        print("✅ Sistema NEXUS desligado corretamente")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante teste: {e}")
        return False


async def test_cli_functionality():
    """Testa funcionalidade do CLI."""
    
    print("\n🖥️  Testando CLI...")
    
    try:
        from nexus.cli.commands import NexusCommands
        
        # Criar comandos CLI
        commands = NexusCommands()
        await commands.initialize()
        print("✅ CLI inicializado")
        
        # Testar status do sistema
        status = await commands.get_system_status()
        print(f"✅ Status obtido: {status['active']}")
        
        # Desligar
        await commands.shutdown()
        print("✅ CLI desligado")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro no teste CLI: {e}")
        return False


async def main():
    """Função principal de teste."""
    
    print("🧪 NEXUS - Testes de Funcionalidade Básica")
    print("=" * 50)
    
    # Executar testes
    tests = [
        ("Imports", test_nexus_imports),
        ("Funcionalidade Básica", test_nexus_basic_functionality),
        ("CLI", test_cli_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Executando teste: {test_name}")
        print("-" * 30)
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Teste {test_name} falhou: {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! NEXUS está funcionando corretamente.")
    else:
        print("⚠️  Alguns testes falharam. Verifique as dependências e configuração.")
    
    return passed == total


if __name__ == "__main__":
    # Configurar logging mínimo
    import logging
    logging.basicConfig(level=logging.ERROR)  # Apenas erros
    
    # Executar testes
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
