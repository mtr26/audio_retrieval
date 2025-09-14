"""
Audio Retrieval System

A neural audio fingerprinting system for content-based audio retrieval.
"""

from .retrieval import AudioRetrieval
from .model import FingerPrintModel, ModelConfig
from .indexing import Indexer
from .preprocessing import load_mel_spectrogram, split_spectrogram

__version__ = "0.1.0"
__author__ = "Mohamed Traore"

__all__ = [
    "AudioRetrieval",
    "FingerPrintModel", 
    "ModelConfig",
    "Indexer",
    "load_mel_spectrogram",
    "split_spectrogram"
]