"""
NEXUS CLI Main Interface

Interface principal de linha de comando para o sistema NEXUS.
"""

import asyncio
import logging
import sys
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

from nexus.cli.commands import NexusCommands
from nexus.cli.demo import DemoRunner

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)
console = Console()

# Criar aplicação Typer
app = typer.Typer(
    name="nexus",
    help="NEXUS - Sistema de Desenvolvimento Autônomo de Próxima Geração",
    rich_markup_mode="rich"
)

# Instância global dos comandos
nexus_commands: Optional[NexusCommands] = None


def print_banner():
    """Exibe banner do NEXUS."""
    
    banner_text = """
    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
    ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    """
    
    subtitle = "Sistema de Desenvolvimento Autônomo de Próxima Geração"
    version = "v0.1.0 - Arquitetura Neuromórfica Distribuída"
    
    banner_panel = Panel(
        Text(banner_text, style="bold blue") + "\n" +
        Text(subtitle, style="italic cyan", justify="center") + "\n" +
        Text(version, style="dim white", justify="center"),
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(banner_panel)


@app.command()
def develop(
    description: str = typer.Argument(..., help="Descrição do projeto a ser desenvolvido"),
    requirements: Optional[str] = typer.Option(None, "--requirements", "-r", help="Arquivo de requisitos"),
    output_dir: Optional[str] = typer.Option("./nexus_output", "--output", "-o", help="Diretório de saída"),
    complexity: Optional[str] = typer.Option("moderate", "--complexity", "-c", help="Nível de complexidade"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Modo verboso")
):
    """
    Executa desenvolvimento autônomo de software.
    
    Exemplo:
        nexus develop "Criar um sistema de e-commerce com React e Node.js"
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(f"\n🚀 [bold green]Iniciando desenvolvimento autônomo...[/bold green]")
    console.print(f"📝 Descrição: {description}")
    console.print(f"📁 Diretório de saída: {output_dir}")
    console.print(f"⚡ Complexidade: {complexity}")
    
    # Executar desenvolvimento
    result = asyncio.run(_run_autonomous_development(description, requirements, output_dir, complexity))
    
    if result['success']:
        console.print(f"\n✅ [bold green]Desenvolvimento concluído com sucesso![/bold green]")
        console.print(f"📊 Arquivos gerados: {result['files_count']}")
        console.print(f"🧪 Cobertura de testes: {result['test_coverage']:.1f}%")
        console.print(f"⏱️  Tempo de execução: {result['execution_time']:.1f}s")
    else:
        console.print(f"\n❌ [bold red]Desenvolvimento falhou:[/bold red] {result['error']}")


@app.command()
def analyze(
    project_path: str = typer.Argument(..., help="Caminho do projeto para análise"),
    analysis_type: str = typer.Option("complete", "--type", "-t", help="Tipo de análise"),
    output_format: str = typer.Option("console", "--format", "-f", help="Formato de saída")
):
    """
    Executa análise de projeto existente.
    
    Exemplo:
        nexus analyze ./my_project --type security --format json
    """
    console.print(f"\n🔍 [bold blue]Analisando projeto...[/bold blue]")
    console.print(f"📁 Projeto: {project_path}")
    console.print(f"🔬 Tipo de análise: {analysis_type}")
    
    # Executar análise
    result = asyncio.run(_run_project_analysis(project_path, analysis_type, output_format))
    
    if result['success']:
        console.print(f"\n✅ [bold green]Análise concluída![/bold green]")
        console.print(f"📊 Score de qualidade: {result['quality_score']:.1f}/10")
        console.print(f"🔒 Score de segurança: {result['security_score']:.1f}/10")
        console.print(f"⚡ Score de performance: {result['performance_score']:.1f}/10")
    else:
        console.print(f"\n❌ [bold red]Análise falhou:[/bold red] {result['error']}")


@app.command()
def demo(
    scenario: str = typer.Option("full", "--scenario", "-s", help="Cenário de demonstração"),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Modo interativo")
):
    """
    Executa demonstração das capacidades do NEXUS.
    
    Cenários disponíveis:
    - full: Demonstração completa
    - cognitive: Substrato cognitivo
    - cortices: Córtices especializados
    - memory: Sistema de memória episódica
    - reasoning: Motor de raciocínio causal
    """
    console.print(f"\n🎭 [bold magenta]Executando demonstração NEXUS...[/bold magenta]")
    console.print(f"🎬 Cenário: {scenario}")
    
    # Executar demonstração
    asyncio.run(_run_demo(scenario, interactive))


@app.command()
def status():
    """Exibe status do sistema NEXUS."""
    
    console.print(f"\n📊 [bold cyan]Status do Sistema NEXUS[/bold cyan]")
    
    # Obter status
    status_info = asyncio.run(_get_system_status())
    
    # Exibir informações
    console.print(f"🟢 Sistema: {'Ativo' if status_info['active'] else 'Inativo'}")
    console.print(f"🧠 Córtices: {status_info['cortices_active']}/4 ativos")
    console.print(f"💾 Memória episódica: {status_info['memory_usage']:.1f}% utilizada")
    console.print(f"🤖 Modelos disponíveis: {status_info['models_available']}")
    console.print(f"⚡ Performance média: {status_info['avg_performance']:.1f}ms")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Mostrar configuração atual"),
    set_key: Optional[str] = typer.Option(None, "--set", help="Definir chave de configuração"),
    value: Optional[str] = typer.Option(None, "--value", help="Valor da configuração")
):
    """Gerencia configuração do NEXUS."""
    
    if show:
        console.print(f"\n⚙️  [bold yellow]Configuração NEXUS[/bold yellow]")
        config_info = asyncio.run(_get_configuration())
        
        for section, settings in config_info.items():
            console.print(f"\n[bold]{section}:[/bold]")
            for key, value in settings.items():
                console.print(f"  {key}: {value}")
    
    elif set_key and value:
        console.print(f"\n⚙️  Definindo {set_key} = {value}")
        asyncio.run(_set_configuration(set_key, value))
        console.print("✅ Configuração atualizada!")
    
    else:
        console.print("Use --show para ver configuração ou --set/--value para definir")


@app.command()
def version():
    """Exibe informações de versão."""
    
    console.print(f"\n📋 [bold]NEXUS - Sistema de Desenvolvimento Autônomo[/bold]")
    console.print(f"Versão: 0.1.0")
    console.print(f"Arquitetura: Neuromórfica Distribuída")
    console.print(f"Python: {sys.version}")
    console.print(f"Plataforma: {sys.platform}")


async def _run_autonomous_development(
    description: str, 
    requirements: Optional[str], 
    output_dir: str, 
    complexity: str
) -> dict:
    """Executa desenvolvimento autônomo."""
    
    try:
        # Inicializar NEXUS se necessário
        global nexus_commands
        if nexus_commands is None:
            nexus_commands = NexusCommands()
            await nexus_commands.initialize()
        
        # Executar desenvolvimento
        result = await nexus_commands.autonomous_development(
            description=description,
            requirements_file=requirements,
            output_directory=output_dir,
            complexity_level=complexity
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erro no desenvolvimento autônomo: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def _run_project_analysis(project_path: str, analysis_type: str, output_format: str) -> dict:
    """Executa análise de projeto."""
    
    try:
        # Inicializar NEXUS se necessário
        global nexus_commands
        if nexus_commands is None:
            nexus_commands = NexusCommands()
            await nexus_commands.initialize()
        
        # Executar análise
        result = await nexus_commands.analyze_project(
            project_path=project_path,
            analysis_type=analysis_type,
            output_format=output_format
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def _run_demo(scenario: str, interactive: bool):
    """Executa demonstração."""
    
    try:
        demo_runner = DemoRunner()
        await demo_runner.run_demo(scenario, interactive)
        
    except Exception as e:
        logger.error(f"Erro na demonstração: {e}", exc_info=True)
        console.print(f"\n❌ [bold red]Erro na demonstração:[/bold red] {e}")


async def _get_system_status() -> dict:
    """Obtém status do sistema."""
    
    try:
        # Inicializar NEXUS se necessário
        global nexus_commands
        if nexus_commands is None:
            nexus_commands = NexusCommands()
            await nexus_commands.initialize()
        
        return await nexus_commands.get_system_status()
        
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}", exc_info=True)
        return {
            'active': False,
            'cortices_active': 0,
            'memory_usage': 0.0,
            'models_available': 0,
            'avg_performance': 0.0
        }


async def _get_configuration() -> dict:
    """Obtém configuração atual."""
    
    return {
        'cognitive': {
            'max_working_memory': '100MB',
            'consolidation_threshold': 100,
            'attention_span': '300s'
        },
        'models': {
            'default_provider': 'openai',
            'max_concurrent': 10,
            'timeout': '30s'
        },
        'execution': {
            'max_parallel_tasks': 5,
            'output_directory': './nexus_output',
            'enable_caching': True
        }
    }


async def _set_configuration(key: str, value: str):
    """Define configuração."""
    
    # Implementação simplificada
    logger.info(f"Configuração {key} definida para {value}")


def main():
    """Função principal do CLI."""
    
    # Exibir banner se não há argumentos
    if len(sys.argv) == 1:
        print_banner()
        console.print("\n💡 Use [bold]nexus --help[/bold] para ver comandos disponíveis")
        console.print("🎭 Use [bold]nexus demo[/bold] para ver demonstração")
        console.print("🚀 Use [bold]nexus develop \"descrição do projeto\"[/bold] para começar")
        return
    
    # Executar aplicação Typer
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n\n👋 [yellow]Operação cancelada pelo usuário[/yellow]")
    except Exception as e:
        console.print(f"\n❌ [bold red]Erro inesperado:[/bold red] {e}")
        logger.error(f"Erro inesperado: {e}", exc_info=True)
    finally:
        # Cleanup se necessário
        if nexus_commands:
            try:
                asyncio.run(nexus_commands.shutdown())
            except:
                pass


if __name__ == "__main__":
    main()
