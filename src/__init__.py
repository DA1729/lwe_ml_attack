"""
LWE Machine Learning Attack Package

This package provides tools for demonstrating machine learning attacks
against the Learning With Errors (LWE) cryptographic problem.
"""

from .lwe_crypto import LWECrypto, LWEDataGenerator, encrypt_message, decrypt_message
from .attack_engine import LWEAttackEngine, LWEPartialAttack
from .model_trainer import LWEModelTrainer

__version__ = "1.0.0"
__author__ = "LWE ML Attack Project"

__all__ = [
    'LWECrypto',
    'LWEDataGenerator', 
    'LWEAttackEngine',
    'LWEPartialAttack',
    'LWEModelTrainer',
    'encrypt_message',
    'decrypt_message'
]