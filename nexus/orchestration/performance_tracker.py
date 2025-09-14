"""
Model Performance Tracker

Rastreia performance de modelos para otimização de roteamento e seleção.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """Rastreador de performance de modelos."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Histórico de performance
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info("Model Performance Tracker initialized")
    
    async def initialize(self) -> None:
        """Inicializa o rastreador."""
        logger.info("Model Performance Tracker initialization complete")
    
    async def record_performance(
        self, 
        task_profile: Any, 
        result: Any, 
        models_used: List[str]
    ) -> None:
        """Registra performance de execução."""
        
        performance_record = {
            'timestamp': datetime.utcnow(),
            'task_type': task_profile.task_type,
            'complexity': task_profile.complexity.value,
            'models_used': models_used,
            'success': result.success,
            'confidence': result.confidence,
            'response_time': result.response_time,
            'cost': result.cost
        }
        
        # Registrar para cada modelo usado
        for model_id in models_used:
            self.performance_history[model_id].append(performance_record)
            
            # Atualizar métricas agregadas
            await self._update_model_metrics(model_id)
    
    async def get_history(self) -> Dict[str, Any]:
        """Obtém histórico de performance."""
        
        history = {}
        for model_id, records in self.performance_history.items():
            history[model_id] = list(records)[-10:]  # Últimos 10 registros
        
        return history
    
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas de performance."""
        
        stats = {}
        
        for model_id, metrics in self.model_metrics.items():
            stats[model_id] = dict(metrics)
        
        return stats
    
    async def _update_model_metrics(self, model_id: str) -> None:
        """Atualiza métricas agregadas de um modelo."""
        
        records = list(self.performance_history[model_id])
        
        if not records:
            return
        
        # Calcular métricas
        recent_records = records[-50:]  # Últimos 50 registros
        
        success_rate = sum(1 for r in recent_records if r['success']) / len(recent_records)
        avg_response_time = sum(r['response_time'] for r in recent_records) / len(recent_records)
        avg_confidence = sum(r['confidence'] for r in recent_records) / len(recent_records)
        avg_cost = sum(r['cost'] for r in recent_records) / len(recent_records)
        
        self.model_metrics[model_id] = {
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'avg_confidence': avg_confidence,
            'avg_cost': avg_cost,
            'total_requests': len(records)
        }
    
    async def shutdown(self) -> None:
        """Desliga o rastreador."""
        logger.info("Model Performance Tracker shutdown complete")
