"""
Neural network models for federated learning.
Includes MobileNet implementation optimized for edge devices.
"""

from .mobilenet import ModelMobileNet
from .cnn_abstract import ModelCNNAbstract

__all__ = [
    'ModelMobileNet',
    'ModelCNNAbstract'
]
