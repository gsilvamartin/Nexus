"""
Experience Encoder

Implementa o codificador de experiências para transformar experiências em
representações vetoriais ricas para armazenamento e recuperação eficiente.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperienceEncoder:
    """Codificador de experiências para memória episódica."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encoding_dimension = config.get('encoding_dimension', 512)
        self.context_weights = config.get('context_weights', {
            'temporal': 0.2,
            'causal': 0.3,
            'emotional': 0.25,
            'strategic': 0.25
        })
        
        logger.info("Experience Encoder initialized")
    
    async def initialize(self) -> None:
        """Inicializa o codificador de experiências."""
        logger.info("Experience Encoder initialization complete")
    
    async def encode(
        self, 
        experience: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Codifica uma experiência com contexto rico.
        
        Args:
            experience: Dados da experiência
            context: Contexto adicional
            
        Returns:
            Experiência codificada com features vetoriais
        """
        # Codificar diferentes aspectos da experiência
        temporal_features = await self._encode_temporal_context(
            context.get('temporal', datetime.utcnow())
        )
        
        causal_features = await self._encode_causal_context(
            context.get('causal', [])
        )
        
        emotional_features = await self._encode_emotional_context(
            context.get('emotional', {})
        )
        
        strategic_features = await self._encode_strategic_context(
            context.get('strategic', {})
        )
        
        # Codificar conteúdo da experiência
        content_features = await self._encode_content(experience)
        
        # Combinar todas as features
        combined_features = await self._combine_features(
            temporal_features,
            causal_features,
            emotional_features,
            strategic_features,
            content_features
        )
        
        return {
            'features': combined_features,
            'temporal_features': temporal_features,
            'causal_features': causal_features,
            'emotional_features': emotional_features,
            'strategic_features': strategic_features,
            'content_features': content_features,
            'encoding_metadata': {
                'dimension': self.encoding_dimension,
                'encoded_at': datetime.utcnow(),
                'encoder_version': '1.0'
            }
        }
    
    async def _encode_temporal_context(self, timestamp: datetime) -> np.ndarray:
        """Codifica contexto temporal."""
        
        # Extrair características temporais
        hour = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 7.0
        day_of_month = timestamp.day / 31.0
        month = timestamp.month / 12.0
        
        # Codificação cíclica para características temporais
        hour_sin = np.sin(2 * np.pi * hour)
        hour_cos = np.cos(2 * np.pi * hour)
        
        day_sin = np.sin(2 * np.pi * day_of_week)
        day_cos = np.cos(2 * np.pi * day_of_week)
        
        month_sin = np.sin(2 * np.pi * month)
        month_cos = np.cos(2 * np.pi * month)
        
        # Criar vetor de features temporais
        temporal_dim = int(self.encoding_dimension * self.context_weights['temporal'])
        temporal_features = np.zeros(temporal_dim)
        
        # Preencher features básicas
        if temporal_dim >= 8:
            temporal_features[0] = hour_sin
            temporal_features[1] = hour_cos
            temporal_features[2] = day_sin
            temporal_features[3] = day_cos
            temporal_features[4] = month_sin
            temporal_features[5] = month_cos
            temporal_features[6] = day_of_month
            temporal_features[7] = timestamp.timestamp() / 1e10  # Normalizado
        
        # Preencher resto com ruído baixo para diversidade
        if temporal_dim > 8:
            temporal_features[8:] = np.random.normal(0, 0.1, temporal_dim - 8)
        
        return temporal_features
    
    async def _encode_causal_context(self, causal_chain: List[str]) -> np.ndarray:
        """Codifica contexto causal."""
        
        causal_dim = int(self.encoding_dimension * self.context_weights['causal'])
        causal_features = np.zeros(causal_dim)
        
        if not causal_chain:
            return causal_features
        
        # Codificar elementos causais usando hash simples
        for i, element in enumerate(causal_chain[:causal_dim//2]):
            # Hash simples do elemento causal
            element_hash = hash(element) % 1000000
            normalized_hash = element_hash / 1000000.0
            
            # Distribuir em duas posições para cada elemento
            pos1 = (i * 2) % causal_dim
            pos2 = (i * 2 + 1) % causal_dim
            
            causal_features[pos1] = normalized_hash
            causal_features[pos2] = len(causal_chain) / 10.0  # Comprimento da cadeia
        
        return causal_features
    
    async def _encode_emotional_context(self, success_metrics: Dict[str, float]) -> np.ndarray:
        """Codifica contexto emocional/avaliativo."""
        
        emotional_dim = int(self.encoding_dimension * self.context_weights['emotional'])
        emotional_features = np.zeros(emotional_dim)
        
        if not success_metrics:
            return emotional_features
        
        # Estatísticas básicas das métricas de sucesso
        values = list(success_metrics.values())
        
        if emotional_dim >= 5:
            emotional_features[0] = np.mean(values)  # Sucesso médio
            emotional_features[1] = np.std(values) if len(values) > 1 else 0  # Variabilidade
            emotional_features[2] = np.max(values)  # Melhor resultado
            emotional_features[3] = np.min(values)  # Pior resultado
            emotional_features[4] = len(values) / 10.0  # Número de métricas
        
        # Codificar métricas individuais
        for i, (metric_name, value) in enumerate(success_metrics.items()):
            if i + 5 < emotional_dim:
                emotional_features[i + 5] = value
        
        return emotional_features
    
    async def _encode_strategic_context(self, strategic_context: Dict[str, Any]) -> np.ndarray:
        """Codifica contexto estratégico."""
        
        strategic_dim = int(self.encoding_dimension * self.context_weights['strategic'])
        strategic_features = np.zeros(strategic_dim)
        
        if not strategic_context:
            return strategic_features
        
        # Codificar chaves e valores do contexto estratégico
        for i, (key, value) in enumerate(strategic_context.items()):
            if i * 2 + 1 < strategic_dim:
                # Hash da chave
                key_hash = hash(key) % 1000000
                strategic_features[i * 2] = key_hash / 1000000.0
                
                # Valor (tentativa de normalização)
                try:
                    if isinstance(value, (int, float)):
                        strategic_features[i * 2 + 1] = min(abs(value), 1.0)
                    else:
                        value_hash = hash(str(value)) % 1000000
                        strategic_features[i * 2 + 1] = value_hash / 1000000.0
                except:
                    strategic_features[i * 2 + 1] = 0.5  # Default
        
        return strategic_features
    
    async def _encode_content(self, experience: Dict[str, Any]) -> np.ndarray:
        """Codifica conteúdo da experiência."""
        
        # Dimensão restante para conteúdo
        used_dimension = sum(
            int(self.encoding_dimension * weight) 
            for weight in self.context_weights.values()
        )
        content_dim = self.encoding_dimension - used_dimension
        
        content_features = np.zeros(content_dim)
        
        # Codificar características básicas do conteúdo
        if content_dim >= 10:
            # Tamanho do conteúdo
            content_size = len(str(experience))
            content_features[0] = min(content_size / 1000.0, 1.0)
            
            # Número de chaves
            num_keys = len(experience) if isinstance(experience, dict) else 1
            content_features[1] = min(num_keys / 20.0, 1.0)
            
            # Presença de tipos específicos de dados
            content_str = str(experience).lower()
            
            content_features[2] = 1.0 if 'error' in content_str else 0.0
            content_features[3] = 1.0 if 'success' in content_str else 0.0
            content_features[4] = 1.0 if 'decision' in content_str else 0.0
            content_features[5] = 1.0 if 'learning' in content_str else 0.0
            content_features[6] = 1.0 if 'pattern' in content_str else 0.0
            content_features[7] = 1.0 if 'insight' in content_str else 0.0
            content_features[8] = 1.0 if 'outcome' in content_str else 0.0
            content_features[9] = 1.0 if 'experience' in content_str else 0.0
        
        # Hash features para o resto da dimensão
        if content_dim > 10:
            content_hash = hash(str(experience))
            np.random.seed(abs(content_hash) % 2**32)
            content_features[10:] = np.random.normal(0, 0.3, content_dim - 10)
        
        return content_features
    
    async def _combine_features(
        self,
        temporal_features: np.ndarray,
        causal_features: np.ndarray,
        emotional_features: np.ndarray,
        strategic_features: np.ndarray,
        content_features: np.ndarray
    ) -> np.ndarray:
        """Combina todas as features em um único vetor."""
        
        # Concatenar todas as features
        combined = np.concatenate([
            temporal_features,
            causal_features,
            emotional_features,
            strategic_features,
            content_features
        ])
        
        # Garantir dimensão correta
        if len(combined) > self.encoding_dimension:
            combined = combined[:self.encoding_dimension]
        elif len(combined) < self.encoding_dimension:
            # Preencher com zeros
            padding = np.zeros(self.encoding_dimension - len(combined))
            combined = np.concatenate([combined, padding])
        
        # Normalizar o vetor
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    async def calculate_similarity(
        self, 
        features1: np.ndarray, 
        features2: np.ndarray
    ) -> float:
        """Calcula similaridade entre duas representações de features."""
        
        # Similaridade do cosseno
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Garantir que está no range [0, 1]
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
    
    async def decode_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Decodifica features de volta para informações interpretáveis."""
        
        # Separar features por tipo
        temporal_dim = int(self.encoding_dimension * self.context_weights['temporal'])
        causal_dim = int(self.encoding_dimension * self.context_weights['causal'])
        emotional_dim = int(self.encoding_dimension * self.context_weights['emotional'])
        strategic_dim = int(self.encoding_dimension * self.context_weights['strategic'])
        
        temporal_features = features[:temporal_dim]
        causal_features = features[temporal_dim:temporal_dim + causal_dim]
        emotional_features = features[temporal_dim + causal_dim:temporal_dim + causal_dim + emotional_dim]
        strategic_features = features[temporal_dim + causal_dim + emotional_dim:temporal_dim + causal_dim + emotional_dim + strategic_dim]
        content_features = features[temporal_dim + causal_dim + emotional_dim + strategic_dim:]
        
        decoded = {
            'temporal_info': await self._decode_temporal_features(temporal_features),
            'causal_info': await self._decode_causal_features(causal_features),
            'emotional_info': await self._decode_emotional_features(emotional_features),
            'strategic_info': await self._decode_strategic_features(strategic_features),
            'content_info': await self._decode_content_features(content_features)
        }
        
        return decoded
    
    async def _decode_temporal_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Decodifica features temporais."""
        
        if len(features) < 8:
            return {}
        
        # Reconstruir informações temporais básicas
        hour_approx = np.arctan2(features[0], features[1]) / (2 * np.pi)
        if hour_approx < 0:
            hour_approx += 1
        
        day_approx = np.arctan2(features[2], features[3]) / (2 * np.pi)
        if day_approx < 0:
            day_approx += 1
        
        return {
            'approximate_hour': hour_approx * 24,
            'approximate_day_of_week': day_approx * 7,
            'approximate_day_of_month': features[6] * 31,
            'temporal_signature': features[7]
        }
    
    async def _decode_causal_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Decodifica features causais."""
        
        return {
            'causal_complexity': np.mean(features),
            'causal_signature': np.std(features)
        }
    
    async def _decode_emotional_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Decodifica features emocionais."""
        
        if len(features) < 5:
            return {}
        
        return {
            'average_success': features[0],
            'success_variability': features[1],
            'peak_success': features[2],
            'lowest_success': features[3],
            'metrics_count': features[4] * 10
        }
    
    async def _decode_strategic_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Decodifica features estratégicas."""
        
        return {
            'strategic_complexity': np.mean(features),
            'strategic_diversity': np.std(features)
        }
    
    async def _decode_content_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Decodifica features de conteúdo."""
        
        if len(features) < 10:
            return {}
        
        return {
            'content_size_indicator': features[0],
            'complexity_indicator': features[1],
            'contains_error': features[2] > 0.5,
            'contains_success': features[3] > 0.5,
            'contains_decision': features[4] > 0.5,
            'contains_learning': features[5] > 0.5,
            'contains_pattern': features[6] > 0.5,
            'contains_insight': features[7] > 0.5,
            'contains_outcome': features[8] > 0.5,
            'contains_experience': features[9] > 0.5
        }
    
    async def shutdown(self) -> None:
        """Desliga o codificador de experiências."""
        logger.info("Experience Encoder shutdown complete")
