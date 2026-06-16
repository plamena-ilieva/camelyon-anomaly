"""Проект 2 -- Разпознаване на медицинска аномалия (CAMELYON17).

Публично API на пакета. Виж PLAN.md за структурата.
"""

from projects.anomaly import data, eda, metrics, models, train
from projects.anomaly.models import VGG, SimpleCNN, UNetAutoencoder

__all__ = [
    'data',
    'eda',
    'metrics',
    'models',
    'train',
    'SimpleCNN',
    'VGG',
    'UNetAutoencoder',
]
